"""Tracklet-level cow identity with globally unique assignment and a saved gallery.

Replaces the greedy per-track matching of GlobalIdentityManager. Same public interface
(associate_tracks / get_summary / registry / track_to_global / last_consensus_energy).

Per synchronized step, for each camera (most reliable view first):
  1. Every ByteTrack tracklet keeps a running appearance embedding (DINOv2 + colour cue),
     so identity is decided from many frames, not one crop.
  2. A cost matrix tracklet x cow is built from
       appearance   1 - cos(tracklet embedding, cow prototype)
       position     distance to where that cow is *right now* in another camera, or
                    where it was last seen (gated by a max walking speed)
       continuity   bonus for keeping the ID the tracklet already has (hysteresis)
       new slot     fixed cost for opening an unused COW_xx (only 10 exist)
  3. Hungarian assignment => each cow at most once per camera per frame. A tracklet
     whose best cost is still poor stays unassigned (None) instead of guessing.
  4. Prototypes are updated only from confident, unambiguous assignments.

The gallery (prototypes) is saved to disk and loaded by the next run, so COW_03 in one
clip/day is COW_03 in the next. Seed it once from verified crops for best results.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .global_identity import GlobalCowRecord
from .local_tracker import LocalTrack
from ..calibration.coordinate_transform import CoordinateTransformer

BIG = 1e3


def color_descriptor(frame_bgr: np.ndarray, bbox) -> np.ndarray:
    """Small, illumination-tolerant coat descriptor (white face / brown tint / tags).

    Black Angus look alike to generic embeddings; what differs is white markings,
    reddish-brown coat tint and coloured ear tags. Returns an L2-normalised 36-d vector.
    """
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return np.zeros(36, np.float32)
    crop = cv2.resize(frame_bgr[y1:y2, x1:x2], (64, 64))
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(np.float32), hsv[..., 1].astype(np.float32), hsv[..., 2].astype(np.float32)
    animal = V > 25
    white = (V > 170) & (S < 50)
    brown = (H < 25) & (S > 60) & (V > 40) & (V < 170)
    tag = (S > 120) & (V > 90)  # saturated yellow/green/orange ear tags
    hist = cv2.calcHist([hsv], [0, 1], animal.astype(np.uint8), [8, 4], [0, 180, 0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-6)
    stats = np.array([white.mean(), brown.mean(), tag.mean(), V[animal].mean() / 255.0 if animal.any() else 0.0])
    v = np.concatenate([hist, 4.0 * stats]).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-6)


def fuse_embedding(dino: Optional[np.ndarray], color: Optional[np.ndarray], color_weight: float = 0.6) -> Optional[np.ndarray]:
    # Always the same length (dino + 36 colour dims): a tiny box gives an all-zero colour part, which is kept as zeros
    # rather than dropped, so a tracklet's running average never mixes vectors of different sizes.
    if dino is None or np.linalg.norm(dino) == 0:
        return None
    col = color if color is not None and np.linalg.norm(color) > 0 else np.zeros(36, np.float32)
    v = np.concatenate([dino / np.linalg.norm(dino), color_weight * col]).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-6)


@dataclass
class Tracklet:
    key: Tuple[str, int]
    emb: Optional[np.ndarray] = None
    n_obs: int = 0
    cow: Optional[str] = None
    n_with_cow: int = 0
    world: Optional[Tuple[float, float]] = None


@dataclass
class CowSlot:
    cow_id: str
    proto: Optional[np.ndarray] = None
    n_updates: int = 0
    last_world: Optional[Tuple[float, float]] = None
    last_time: float = -1e9
    step_world: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # this step, per camera


DEFAULTS = dict(
    camera_order=["cam1", "cam2", "cam3"],
    position_trust={"cam1": 1.0, "cam2": 1.0, "cam3": 0.0},  # cam3 metric map unverified
    w_app=1.0,
    w_pos=0.6,
    pos_scale_m=3.0,
    max_speed_mps=3.0,
    pos_gate_slack_m=3.0,
    pos_memory_sec=10.0,
    stick_bonus=0.35,
    stick_full_after=5,
    new_slot_cost=0.55,
    accept_max_cost=0.85,
    proto_update_margin=0.05,
    proto_lr=0.05,
    emb_ema=0.2,
    color_weight=0.6,
)


class TrackletIdentityManager:
    def __init__(self, transformer: CoordinateTransformer, num_cows: int = 10,
                 gallery_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        self.transformer = transformer
        self.cfg = {**DEFAULTS, **(config or {})}
        self.cow_ids = [f"COW_{i:02d}" for i in range(1, num_cows + 1)]
        self.slots: Dict[str, CowSlot] = {c: CowSlot(c) for c in self.cow_ids}
        self.registry: Dict[str, GlobalCowRecord] = {c: GlobalCowRecord(global_cow_id=c) for c in self.cow_ids}
        self.tracklets: Dict[Tuple[str, int], Tracklet] = {}
        self.track_to_global: Dict[Tuple[str, int], str] = {}
        self.track_confidence: Dict[Tuple[str, int], float] = {}  # this step's assignment confidence
        self.last_consensus_energy: float = 0.0  # kept for CSV compatibility: mean assignment cost
        self.gallery_path = gallery_path
        if gallery_path and os.path.exists(gallery_path):
            self.load_gallery(gallery_path)

    # ------------------------------------------------------------------ gallery IO
    def save_gallery(self, path: Optional[str] = None):
        path = path or self.gallery_path
        if not path:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        seen = [c for c in self.cow_ids if self.slots[c].proto is not None]
        np.savez(path, cow_ids=np.array(seen),
                 protos=np.stack([self.slots[c].proto for c in seen]) if seen else np.zeros((0, 1)),
                 n_updates=np.array([self.slots[c].n_updates for c in seen]))
        with open(os.path.splitext(path)[0] + ".json", "w") as f:
            json.dump({c: {"n_updates": self.slots[c].n_updates} for c in seen}, f, indent=2)

    def load_gallery(self, path: str):
        d = np.load(path, allow_pickle=False)
        for cid, p, n in zip(d["cow_ids"], d["protos"], d["n_updates"]):
            if str(cid) in self.slots:
                self.slots[str(cid)].proto = p.astype(np.float32)
                self.slots[str(cid)].n_updates = int(n)
        print(f"[IDENTITY] loaded {len(d['cow_ids'])} cow prototypes from {path}")

    # ------------------------------------------------------------------ main step
    def associate_tracks(self, camera_tracks: Dict[str, List[LocalTrack]],
                         track_embeddings: Dict[Tuple[str, int], np.ndarray],
                         current_timestamp: float,
                         frame_sizes: Optional[Dict[str, Tuple[int, int]]] = None) -> Dict[Tuple[str, int], str]:
        c = self.cfg
        for s in self.slots.values():
            s.step_world = {}
        costs_used = []
        order = [cam for cam in c["camera_order"] if cam in camera_tracks] + \
                [cam for cam in camera_tracks if cam not in c["camera_order"]]

        for cam in order:
            tracks = camera_tracks[cam]
            if not tracks:
                continue
            res = (frame_sizes or {}).get(cam)
            tls = []
            for t in tracks:
                key = (cam, t.track_id)
                tl = self.tracklets.setdefault(key, Tracklet(key))
                cx, by = (t.bbox[0] + t.bbox[2]) / 2.0, t.bbox[3]
                tl.world = self.transformer.pixel_to_world(cam, cx, by, current_res=res)
                t.world_pos = tl.world
                e = track_embeddings.get(key)
                if e is not None and np.linalg.norm(e) > 0:
                    a = 1.0 if tl.emb is None else max(c["emb_ema"], 1.0 / (tl.n_obs + 1))
                    tl.emb = e if tl.emb is None or tl.emb.shape != e.shape else (1 - a) * tl.emb + a * e
                    tl.emb = tl.emb / (np.linalg.norm(tl.emb) + 1e-6)
                tl.n_obs += 1
                tls.append((t, tl))

            C = np.array([[self._cost(tl, cam, cid, current_timestamp) for cid in self.cow_ids] for _, tl in tls])
            rows, cols = linear_sum_assignment(C)
            assigned = {}
            for r, k in zip(rows, cols):
                if C[r, k] <= c["accept_max_cost"]:
                    assigned[r] = k
            for r, (t, tl) in enumerate(tls):
                if r not in assigned:
                    t.global_cow_id = None
                    self.track_to_global.pop(tl.key, None)
                    continue
                k = assigned[r]
                cid = self.cow_ids[k]
                costs_used.append(float(C[r, k]))
                tl.n_with_cow = tl.n_with_cow + 1 if tl.cow == cid else 1
                tl.cow = cid
                t.global_cow_id = cid
                self._bind(cam, t.track_id, cid, confidence=float(np.clip(1 - C[r, k], 0.05, 0.99)))
                slot = self.slots[cid]
                if tl.world is not None and c["position_trust"].get(cam, 1.0) > 0:
                    slot.step_world[cam] = tl.world
                    slot.last_world, slot.last_time = tl.world, current_timestamp
                # prototype update only from unambiguous, visible assignments
                others = np.delete(C[r], k)
                margin = float(others.min() - C[r, k]) if len(others) else 1.0
                occluded = getattr(t, "occlusion_score", 0.0) or 0.0
                if tl.emb is not None and (slot.proto is None or (margin >= c["proto_update_margin"] and occluded < 0.3)):
                    lr = 1.0 if slot.proto is None else c["proto_lr"]
                    if slot.proto is not None and slot.proto.shape != tl.emb.shape:
                        lr = 1.0
                    p = tl.emb if lr == 1.0 else (1 - lr) * slot.proto + lr * tl.emb
                    slot.proto = p / (np.linalg.norm(p) + 1e-6)
                    slot.n_updates += 1
                rec = self.registry[cid]
                rec.first_seen_timestamp = min(rec.first_seen_timestamp, current_timestamp)
                rec.last_seen_timestamp = max(rec.last_seen_timestamp, current_timestamp)
                rec.cameras_observed.add(cam)
                if tl.world is not None:
                    rec.positions.append((current_timestamp, tl.world[0], tl.world[1]))

        self.last_consensus_energy = float(np.mean(costs_used)) if costs_used else 0.0
        return self.track_to_global

    def _cost(self, tl: Tracklet, cam: str, cid: str, now: float) -> float:
        c = self.cfg
        slot = self.slots[cid]
        if slot.proto is None:  # unused slot: open only if nothing existing fits
            return c["new_slot_cost"]
        d_app = 0.5
        if tl.emb is not None and slot.proto.shape == tl.emb.shape:
            d_app = float(np.clip(1.0 - float(np.dot(tl.emb, slot.proto)), 0.0, 1.0))
        cost = c["w_app"] * d_app

        trust = c["position_trust"].get(cam, 1.0)
        ref, age = None, None
        other = [(k, v) for k, v in slot.step_world.items() if k != cam and c["position_trust"].get(k, 1.0) > 0]
        if other:
            ref, age = other[0][1], 0.0
        elif slot.last_world is not None and now - slot.last_time <= c["pos_memory_sec"]:
            ref, age = slot.last_world, now - slot.last_time
        if trust > 0 and ref is not None and tl.world is not None:
            dist = float(np.hypot(tl.world[0] - ref[0], tl.world[1] - ref[1]))
            if dist > c["max_speed_mps"] * age + c["pos_gate_slack_m"]:
                return BIG  # physically impossible jump
            cost += trust * c["w_pos"] * min(1.0, dist / c["pos_scale_m"])
        if cam in slot.step_world:  # already used by another tracklet in this camera
            return BIG
        if tl.cow == cid:
            cost -= c["stick_bonus"] * min(1.0, tl.n_with_cow / c["stick_full_after"])
        return cost

    # ------------------------------------------------------------------ registry helpers
    def _bind(self, cam: str, track_id: int, cid: str, confidence: float):
        self.track_to_global[(cam, track_id)] = cid
        self.track_confidence[(cam, track_id)] = confidence
        rec = self.registry[cid]
        rec.camera_track_ids.setdefault(cam, [])
        if track_id not in rec.camera_track_ids[cam]:
            rec.camera_track_ids[cam].append(track_id)
        rec.identity_confidence = max(rec.identity_confidence, confidence)

    def get_summary(self) -> List[Dict[str, Any]]:
        out = []
        for cid, rec in self.registry.items():
            if rec.first_seen_timestamp != float("inf"):
                rec.total_observed_seconds = max(0.0, rec.last_seen_timestamp - rec.first_seen_timestamp)
            d = rec.to_dict()
            d["prototype_updates"] = self.slots[cid].n_updates
            out.append(d)
        return out
