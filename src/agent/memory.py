"""Per-cow memory cards: what the agent knows about each animal.

  outputs/agent/memory/cards.json           one card per cow (edit `confirmed`, `ear_tag`, `same_as`)
  outputs/agent/memory/crops/COW_xx/*.jpg   reference crops (several cameras / day + IR)
  outputs/agent/memory/prototypes.npz       mean appearance embedding per cow
  outputs/agent/memory/enrollment_sheet.jpg one row per cow, for the one-time human check

Enrollment collects the clearest crops of each tracked ID (large, confident, unoccluded,
not cut by the frame edge, spread over cameras and time). A person confirms the sheet once
(`confirmed: true`, or `same_as: COW_03` to merge a duplicate); after that the agent keeps
the cards up to date (last seen, notes, extra references).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

from .evidence import Embedder, FrameStore, crop, label, tile


@dataclass
class CowCard:
    cow_id: str
    confirmed: bool = False
    same_as: Optional[str] = None
    ear_tag: Optional[str] = None
    traits: str = ""
    references: List[str] = field(default_factory=list)
    last_seen: Dict = field(default_factory=dict)
    notes: List[Dict] = field(default_factory=list)
    stats: Dict = field(default_factory=dict)


class Memory:
    def __init__(self, root: str = "outputs/agent/memory"):
        self.root = root
        self.cards: Dict[str, CowCard] = {}
        self.protos: Dict[str, np.ndarray] = {}
        self.load()

    # ---------------------------------------------------------------- persistence
    @property
    def path(self):
        return os.path.join(self.root, "cards.json")

    def load(self):
        if os.path.exists(self.path):
            for d in json.load(open(self.path)):
                self.cards[d["cow_id"]] = CowCard(**d)
        p = os.path.join(self.root, "prototypes.npz")
        if os.path.exists(p):
            z = np.load(p)
            self.protos = {k: z[k] for k in z.files}

    def save(self):
        os.makedirs(self.root, exist_ok=True)
        json.dump([asdict(c) for c in sorted(self.cards.values(), key=lambda c: c.cow_id)],
                  open(self.path, "w"), indent=1)
        if self.protos:
            np.savez(os.path.join(self.root, "prototypes.npz"), **self.protos)

    # ---------------------------------------------------------------- queries
    def known_ids(self) -> List[str]:
        return sorted(c.cow_id for c in self.cards.values() if not c.same_as)

    def canonical(self, cid: Optional[str]) -> Optional[str]:
        seen = set()
        while cid in self.cards and self.cards[cid].same_as and cid not in seen:
            seen.add(cid)
            cid = self.cards[cid].same_as
        return cid

    def describe(self, cid: str) -> str:
        c = self.cards.get(cid)
        if c is None:
            return f"{cid}: no memory card"
        bits = [f"{cid}{' (human-confirmed)' if c.confirmed else ''}"]
        if c.ear_tag:
            bits.append(f"ear tag {c.ear_tag}")
        if c.traits:
            bits.append(c.traits)
        if c.last_seen:
            bits.append(f"last seen {c.last_seen.get('time')} on {c.last_seen.get('camera')}"
                        + (f" at ({c.last_seen['x']:.1f}, {c.last_seen['y']:.1f}) m" if c.last_seen.get("x") is not None else ""))
        if c.notes:
            bits.append("notes: " + "; ".join(n["text"] for n in c.notes[-3:]))
        return " | ".join(bits)

    def similarity(self, emb: np.ndarray, candidates: Optional[List[str]] = None) -> List[tuple]:
        out = []
        for cid, p in self.protos.items():
            if candidates and cid not in candidates:
                continue
            if self.cards.get(cid) and self.cards[cid].same_as:
                continue
            if p.shape != emb.shape:
                continue
            out.append((cid, float(np.dot(emb, p) / (np.linalg.norm(emb) * np.linalg.norm(p) + 1e-9))))
        return sorted(out, key=lambda x: -x[1])

    def add_note(self, cid: str, t: str, text: str, source: str = "agent"):
        card = self.cards.setdefault(cid, CowCard(cid))
        card.notes.append({"time": t, "text": text, "source": source})

    def update_last_seen(self, rec: pd.DataFrame):
        for cid, g in rec[rec.global_cow_id.notna()].groupby("global_cow_id"):
            if cid not in self.cards:  # only enrolled cows get cards (no ghost IDs)
                continue
            r = g.sort_values("t").iloc[-1]
            card = self.cards[cid]
            card.last_seen = {"time": str(pd.Timestamp(r.t, unit="s")), "camera": r.camera_id,
                              "x": None if pd.isna(r.world_x) else float(r.world_x),
                              "y": None if pd.isna(r.world_y) else float(r.world_y)}

    # ---------------------------------------------------------------- enrollment
    def enroll(self, rec: pd.DataFrame, frames: FrameStore, embedder: Optional[Embedder] = None,
               per_cow: int = 6, min_obs: int = 30) -> List[str]:
        """Builds / refreshes cards from the clearest crops of each tracked ID."""
        embedder = embedder or Embedder()
        r = rec[rec.global_cow_id.notna()].copy()
        r["area"] = (r.bbox_x2 - r.bbox_x1) * (r.bbox_y2 - r.bbox_y1) / (r.frame_width * r.frame_height)
        r["occ"] = pd.to_numeric(r.occlusion_status, errors="coerce").fillna(0.5)
        margin = 0.01
        r["edge"] = ((r.bbox_x1 < margin * r.frame_width) | (r.bbox_x2 > (1 - margin) * r.frame_width) |
                     (r.bbox_y1 < margin * r.frame_height) | (r.bbox_y2 > (1 - margin) * r.frame_height))
        r["quality"] = r.detection_confidence * np.sqrt(r.area.clip(0, 0.2)) * (1 - r.occ) * (~r.edge)
        made = []
        for cid, g in r.groupby("global_cow_id"):
            if len(g) < min_obs:
                continue
            g = g[g.quality > 0].sort_values("quality", ascending=False)
            picks, used = [], []
            for row in g.itertuples():  # spread over cameras and time (>= 20 s apart)
                if all(abs(row.t - u.t) > 20 or row.camera_id != u.camera_id for u in used):
                    used.append(row)
                if len(used) >= per_cow:
                    break
            cdir = os.path.join(self.root, "crops", cid)
            os.makedirs(cdir, exist_ok=True)
            embs = []
            card = self.cards.setdefault(cid, CowCard(cid))
            for row in used:
                fr = frames.frame(row.camera_id, row.t)
                if fr is None:
                    continue
                bb = (row.bbox_x1, row.bbox_y1, row.bbox_x2, row.bbox_y2)
                p = os.path.join(cdir, f"{row.camera_id}_{pd.Timestamp(row.t, unit='s'):%Y%m%d_%H%M%S}.jpg")
                cv2.imwrite(p, crop(fr, bb, pad=0.05, size=384))
                if p not in card.references:
                    card.references.append(p)
                embs.append(embedder(fr, bb))
            if embs:
                m = np.mean(embs, 0)
                self.protos[cid] = (m / (np.linalg.norm(m) + 1e-9)).astype(np.float32)
                card.stats["n_observations"] = int(len(g))
                card.stats["cameras"] = sorted(g.camera_id.unique().tolist())
                made.append(cid)
        self.update_last_seen(r)
        self.save()
        self.enrollment_sheet()
        return made

    def describe_traits(self, backend, overwrite: bool = False) -> int:
        """Ask the VLM to write each cow's distinguishing marks onto its card (ear tag colour /
        number, white face or patches, size, coat tint). Done once at enrollment."""
        from .vlm import parse_json
        from .evidence import to_jpeg_b64
        n = 0
        for cid in self.known_ids():
            card = self.cards[cid]
            if card.traits and not overwrite:
                continue
            img = self.reference_strip(cid, n=6, cell=256)
            msgs = [{"role": "system", "content": "You describe individual cattle for re-identification."},
                    {"role": "user", "content": [
                        {"type": "text", "text": (
                            f"These crops are meant to show the same Black Angus steer ({cid}) from different "
                            "cameras. List only features that would tell it apart from other black steers: ear tag "
                            "colour/number and which ear, white face or white patches, size/build, coat tint, "
                            "horns/scars. Say if the crops seem to show more than one animal. Reply JSON only: "
                            '{"traits": "<short>", "ear_tag": "<colour and number or null>", "mixed_animals": true/false}')},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + to_jpeg_b64(img, 1024)}}]}]
            try:
                out = parse_json(backend.complete(msgs)) or {}
            except Exception as e:
                print(f"[MEMORY] traits for {cid} failed: {e}")
                continue
            tr = out.get("traits", "")
            if isinstance(tr, dict):
                tr = "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in tr.items())
            elif isinstance(tr, list):
                tr = "; ".join(map(str, tr))
            card.traits = str(tr)[:300]
            if out.get("ear_tag") and str(out["ear_tag"]).lower() not in ("null", "none", ""):
                card.ear_tag = str(out["ear_tag"])[:60]
            if out.get("mixed_animals"):
                self.add_note(cid, "enrollment", "references may show more than one animal - check", source="vlm")
            n += 1
        self.save()
        return n

    def reference_strip(self, cid: str, n: int = 6, cell: int = 192) -> np.ndarray:
        card = self.cards.get(cid)
        ims = []
        for p in (card.references if card else [])[-n:]:
            im = cv2.imread(p)
            if im is not None:
                ims.append(label(im, os.path.basename(p).split("_")[0]))
        return tile(ims or [np.zeros((cell, cell, 3), np.uint8)], cols=max(1, len(ims)), cell=cell)

    def enrollment_sheet(self, out: Optional[str] = None) -> str:
        rows = []
        for cid in self.known_ids():
            strip = self.reference_strip(cid)
            head = np.zeros((strip.shape[0], 160, 3), np.uint8)
            cv2.putText(head, cid, (8, strip.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            rows.append(np.hstack([head, strip]))
        if not rows:
            return ""
        wmax = max(r.shape[1] for r in rows)
        rows = [np.hstack([r, np.zeros((r.shape[0], wmax - r.shape[1], 3), np.uint8)]) for r in rows]
        out = out or os.path.join(self.root, "enrollment_sheet.jpg")
        cv2.imwrite(out, np.vstack(rows))
        return out
