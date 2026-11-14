"""Tools the agent can call. Each returns text (+ optional images) and is tagged with the
kind of evidence it provides, which the verifier uses ("two independent kinds agree")."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

from .evidence import FrameStore, crop, draw_boxes, label, tile
from .memory import Memory


@dataclass
class ToolResult:
    text: str
    images: List[np.ndarray] = field(default_factory=list)
    data: Dict = field(default_factory=dict)


@dataclass
class Tool:
    name: str
    fn: Callable
    args: str
    description: str
    evidence: str  # visual | appearance_score | trajectory | classifier | chew_motion | memory


class Toolbox:
    def __init__(self, rec: pd.DataFrame, frames: FrameStore, memory: Memory,
                 timeline: Optional[pd.DataFrame] = None, embedder=None, video_index=None):
        self.rec = rec[rec.global_cow_id.notna() | rec.local_track_id.notna()]
        self.frames, self.memory, self.timeline = frames, memory, timeline
        self.embedder = embedder
        self.video_index = video_index or {}
        self.day = pd.Timestamp(rec.t.min(), unit="s").normalize() if len(rec) else pd.Timestamp("1970-01-01")
        self.tools: Dict[str, Tool] = {t.name: t for t in [
            Tool("memory", self.t_memory, '{"cow_ids": ["COW_01", ...]}',
                 "What the memory cards say about these cows (tag, traits, last seen, notes).", "memory"),
            Tool("scene", self.t_scene, '{"camera": "cam1", "time": "HH:MM:SS"}',
                 "Full camera frame at that time with every tracked box and its ID drawn.", "visual"),
            Tool("cow_strip", self.t_cow_strip,
                 '{"cow_id": "COW_03", "time": "HH:MM:SS", "camera": "cam1" (optional), "before": 4, "after": 4}',
                 "Crops of one cow at 6 moments from `before` s before to `after` s after, tiled in one image "
                 "(see how it moves / what it does).", "visual"),
            Tool("track_strip", self.t_track_strip,
                 '{"camera": "cam3", "track_id": 12, "time": "HH:MM:SS", "before": 4, "after": 4}',
                 "Same as cow_strip but for one tracker box regardless of its assigned ID.", "visual"),
            Tool("compare_to_cards", self.t_compare,
                 '{"camera": "cam3", "track_id": 12, "time": "HH:MM:SS", "candidates": ["COW_02", "COW_05"] (optional)}',
                 "Appearance similarity (DINOv2 + coat colour) of that box against each cow's memory card, "
                 "plus an image of the crop next to the best candidates' reference crops.", "appearance_score"),
            Tool("all_views", self.t_all_views, '{"time": "HH:MM:SS", "cow_id": "COW_03" (optional)}',
                 "The same moment from ALL cameras (cam1 front, cam2 bunk, cam3 back) side by side with every box "
                 "and ID drawn; with cow_id, its box is thick and its crops per camera are listed. Use it to check "
                 "that one cow has one consistent identity across views.", "visual"),
            Tool("references", self.t_references, '{"cow_ids": ["COW_01", "COW_04"]}',
                 "Reference crops stored on the memory cards of these cows.", "visual"),
            Tool("trajectory", self.t_trajectory, '{"cow_id": "COW_03", "t0": "HH:MM:SS", "t1": "HH:MM:SS"}',
                 "Where the cow was (per camera, pen metres where calibrated), its tracks, speeds and gaps.",
                 "trajectory"),
            Tool("who_is_near", self.t_near, '{"camera": "cam1", "time": "HH:MM:SS"}',
                 "All boxes in that camera at that time: ID, track, confidence, position.", "trajectory"),
            Tool("behavior", self.t_behavior, '{"cow_id": "COW_03", "t0": "HH:MM:SS", "t1": "HH:MM:SS"}',
                 "Behavior classifier output for the cow: per-second labels, confidences, speed, posture cue.",
                 "classifier"),
            Tool("chew_motion", self.t_chew, '{"cow_id": "COW_03", "time": "HH:MM:SS"}',
                 "Jaw-motion frequency test over ~12 s of video (rumination = periodic chewing 0.75-1.6 Hz). "
                 "Use this for rumination; chewing is not visible in still frames.", "chew_motion"),
        ]}

    # ------------------------------------------------------------------ helpers
    def t(self, v) -> float:
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip()
        try:
            ts = pd.Timestamp(s) if len(s) > 8 else self.day + pd.Timedelta(s)
        except Exception:
            ts = pd.Timestamp(s)
        return (ts - pd.Timestamp("1970-01-01")).total_seconds()

    def hms(self, t: float) -> str:
        return pd.Timestamp(t, unit="s").strftime("%H:%M:%S")

    def describe(self) -> str:
        return "\n".join(f"- {t.name} {t.args}: {t.description}" for t in self.tools.values())

    def call(self, name: str, args: Dict) -> ToolResult:
        if name not in self.tools:
            return ToolResult(f"unknown tool '{name}'. Available: {', '.join(self.tools)}")
        try:
            return self.tools[name].fn(**(args or {}))
        except TypeError as e:
            return ToolResult(f"bad arguments for {name}: {e}. Expected {self.tools[name].args}")
        except Exception as e:  # tools must never crash the loop
            return ToolResult(f"{name} failed: {type(e).__name__}: {e}")

    def _nearest(self, rows: pd.DataFrame, t: float, tol: float = 3.0) -> Optional[pd.Series]:
        if rows.empty:
            return None
        i = (rows.t - t).abs().idxmin()
        return rows.loc[i] if abs(rows.loc[i, "t"] - t) <= tol else None

    def _strip(self, rows: pd.DataFrame, camera: str, t: float, before: float, after: float, title: str) -> ToolResult:
        rows = rows[rows.camera_id == camera]
        if rows.empty:
            return ToolResult(f"no {title} boxes on {camera}")
        ims, lines = [], []
        for tt in np.linspace(t - before, t + after, 6):
            r = self._nearest(rows, tt, tol=1.5)
            if r is None:
                continue
            fr = self.frames.frame(camera, float(r.t))
            if fr is None:
                continue
            bb = (r.bbox_x1, r.bbox_y1, r.bbox_x2, r.bbox_y2)
            ims.append(label(crop(fr, bb, pad=0.25, size=320), f"{self.hms(r.t)} {r.global_cow_id} t{int(r.local_track_id)}"))
            lines.append(f"{self.hms(r.t)}: {r.global_cow_id} track {int(r.local_track_id)} conf {r.detection_confidence:.2f}")
        if not ims:
            return ToolResult(f"no video frames available for {title} on {camera}")
        return ToolResult(f"{title} on {camera}, {len(ims)} crops (left->right, top->bottom = time):\n" + "\n".join(lines),
                          [tile(ims, cols=3, cell=320)])

    # ------------------------------------------------------------------ tools
    def t_memory(self, cow_ids: List[str]):
        return ToolResult("\n".join(self.memory.describe(c) for c in cow_ids))

    def t_scene(self, camera: str, time):
        t = self.t(time)
        fr = self.frames.frame(camera, t)
        if fr is None:
            return ToolResult(f"no video for {camera} at {self.hms(t)}")
        rows = self.rec[(self.rec.camera_id == camera) & ((self.rec.t - t).abs() <= 0.6)]
        rows = rows.sort_values("t").groupby("local_track_id").head(1)
        txt = ", ".join(f"{r.global_cow_id}(t{int(r.local_track_id)})" for r in rows.itertuples()) or "no boxes"
        return ToolResult(f"{camera} at {self.hms(t)}: {len(rows)} boxes: {txt}", [draw_boxes(fr, rows)])

    def t_cow_strip(self, cow_id: str, time, camera: Optional[str] = None, before: float = 4, after: float = 4):
        t = self.t(time)
        rows = self.rec[(self.rec.global_cow_id == cow_id) & (self.rec.t.between(t - before - 2, t + after + 2))]
        if rows.empty:
            return ToolResult(f"{cow_id} has no boxes between {self.hms(t - before)} and {self.hms(t + after)}")
        camera = camera or rows.camera_id.value_counts().index[0]
        return self._strip(rows, camera, t, before, after, cow_id)

    def t_track_strip(self, camera: str, track_id: int, time, before: float = 4, after: float = 4):
        t = self.t(time)
        rows = self.rec[(self.rec.camera_id == camera) & (self.rec.local_track_id == int(track_id))
                        & self.rec.t.between(t - before - 2, t + after + 2)]
        return self._strip(rows, camera, t, before, after, f"track {track_id}")

    def t_compare(self, camera: str, track_id: int, time, candidates: Optional[List[str]] = None):
        t = self.t(time)
        rows = self.rec[(self.rec.camera_id == camera) & (self.rec.local_track_id == int(track_id))]
        r = self._nearest(rows, t, tol=10)
        if r is None:
            return ToolResult(f"track {track_id} not on {camera} near {self.hms(t)}")
        fr = self.frames.frame(camera, float(r.t))
        if fr is None or self.embedder is None:
            return ToolResult("no frame / embedder available")
        bb = (r.bbox_x1, r.bbox_y1, r.bbox_x2, r.bbox_y2)
        sims = self.memory.similarity(self.embedder(fr, bb), candidates)
        if not sims:
            return ToolResult("no memory-card prototypes to compare against (run enrollment)")
        top = sims[:3]
        txt = (f"track {track_id} on {camera} at {self.hms(r.t)} (currently {r.global_cow_id}): similarity "
               + ", ".join(f"{c} {s:.3f}" for c, s in sims[:6])
               + f". Margin best-vs-second {sims[0][1] - (sims[1][1] if len(sims) > 1 else 0):.3f} "
               "(< 0.02 means the score cannot separate them).")
        query = label(crop(fr, bb, pad=0.05, size=256), f"query t{track_id}")
        rows_img = [np.hstack([query, np.zeros((query.shape[0], 8, 3), np.uint8),
                               label(self.memory.reference_strip(c, n=3, cell=256)[:256], f"{c} {s:.3f}")])
                    for c, s in top]
        wmax = max(x.shape[1] for x in rows_img)
        img = np.vstack([np.hstack([x, np.zeros((x.shape[0], wmax - x.shape[1], 3), np.uint8)]) for x in rows_img])
        return ToolResult(txt, [img], {"ranking": sims})

    def t_all_views(self, time, cow_id: Optional[str] = None):
        t = self.t(time)
        ims, lines = [], []
        for cam in sorted(self.frames.clips):
            fr = self.frames.frame(cam, t)
            rows = self.rec[(self.rec.camera_id == cam) & ((self.rec.t - t).abs() <= 0.6)]
            rows = rows.sort_values("t").groupby("local_track_id").head(1)
            if fr is None:
                lines.append(f"{cam}: no video at {self.hms(t)}")
                continue
            img = draw_boxes(fr, rows, highlight=cow_id)
            s = 900 / max(img.shape[:2])
            ims.append(label(cv2.resize(img, None, fx=s, fy=s), f"{cam} {self.hms(t)}"))
            ids = ", ".join(f"{r.global_cow_id}(t{int(r.local_track_id)})" for r in rows.itertuples())
            mine = rows[rows.global_cow_id == cow_id] if cow_id else rows.iloc[:0]
            lines.append(f"{cam}: {len(rows)} boxes [{ids}]" + (f"; {cow_id} {'visible' if len(mine) else 'NOT visible'}" if cow_id else ""))
        if not ims:
            return ToolResult("no camera has video at " + self.hms(t))
        h = max(i.shape[0] for i in ims)
        ims = [np.vstack([i, np.zeros((h - i.shape[0], i.shape[1], 3), np.uint8)]) for i in ims]
        return ToolResult("\n".join(lines), [np.hstack(ims)])

    def t_references(self, cow_ids: List[str]):
        ims = [label(self.memory.reference_strip(c, n=4, cell=224), c) for c in cow_ids]
        wmax = max(x.shape[1] for x in ims)
        img = np.vstack([np.hstack([x, np.zeros((x.shape[0], wmax - x.shape[1], 3), np.uint8)]) for x in ims])
        return ToolResult("reference crops, one row per cow: " + ", ".join(cow_ids), [img])

    def t_trajectory(self, cow_id: str, t0, t1):
        a, b = self.t(t0), self.t(t1)
        g = self.rec[(self.rec.global_cow_id == cow_id) & self.rec.t.between(a, b)].sort_values("t")
        if g.empty:
            return ToolResult(f"{cow_id} not observed {self.hms(a)}-{self.hms(b)}")
        lines = []
        for cam, gc in g.groupby("camera_id"):
            segs = (gc.local_track_id != gc.local_track_id.shift()).cumsum()
            for _, s in gc.groupby(segs):
                p0, p1 = s.iloc[0], s.iloc[-1]
                pos = (f" world ({p0.world_x:.1f},{p0.world_y:.1f})->({p1.world_x:.1f},{p1.world_y:.1f}) m"
                       if pd.notna(p0.world_x) and pd.notna(p1.world_x) else "")
                lines.append(f"{cam} track {int(p0.local_track_id)}: {self.hms(p0.t)}-{self.hms(p1.t)} "
                             f"({len(s)} obs), px x {p0.center_x / p0.frame_width:.2f}->{p1.center_x / p1.frame_width:.2f}"
                             f"{pos}, id conf {s.identity_confidence.mean():.2f}")
        gw = g[g.world_x.notna()]
        if len(gw) > 1:
            dt = np.diff(gw.t.values)
            dd = np.hypot(np.diff(gw.world_x.values), np.diff(gw.world_y.values))
            sp = dd / np.maximum(dt, 1e-3)
            lines.append(f"max world speed {sp.max():.2f} m/s (steer walking ~1 m/s, running ~3 m/s)")
        return ToolResult(f"{cow_id} {self.hms(a)}-{self.hms(b)}:\n" + "\n".join(lines))

    def t_near(self, camera: str, time):
        t = self.t(time)
        rows = self.rec[(self.rec.camera_id == camera) & ((self.rec.t - t).abs() <= 0.6)]
        rows = rows.sort_values("t").groupby("local_track_id").head(1)
        if rows.empty:
            return ToolResult(f"no boxes on {camera} at {self.hms(t)}")
        return ToolResult("\n".join(
            f"{r.global_cow_id} track {int(r.local_track_id)} det {r.detection_confidence:.2f} "
            f"id {r.identity_confidence:.2f} px x {r.center_x / r.frame_width:.2f} y {r.center_y / r.frame_height:.2f}"
            + (f" world ({r.world_x:.1f},{r.world_y:.1f})" if pd.notna(r.world_x) else "")
            for r in rows.itertuples()))

    def t_behavior(self, cow_id: str, t0, t1):
        if self.timeline is None:
            return ToolResult("no behavior timeline loaded")
        a, b = self.t(t0), self.t(t1)
        g = self.timeline[(self.timeline.cow_id == cow_id) & self.timeline.t.between(a, b)]
        if g.empty:
            return ToolResult(f"no behavior rows for {cow_id} {self.hms(a)}-{self.hms(b)}")
        lines = [f"{beh}: {n} s" for beh, n in g.behavior.value_counts().items()]
        lines.append(f"mean confidence {g.confidence.mean():.2f}, observed share {g.observed.mean():.0%}")
        for col in ("speed_mps", "height_ratio", "nn_dist"):
            if col in g.columns and g[col].notna().any():
                lines.append(f"{col} median {g[col].median():.2f}")
        if "behavior_2" in g.columns:
            lines.append("second classifier: " + ", ".join(f"{k} {v} s" for k, v in g.behavior_2.value_counts().items()))
        return ToolResult(f"{cow_id} {self.hms(a)}-{self.hms(b)}:\n" + "\n".join(lines))

    def t_chew(self, cow_id: str, time):
        from ..behavior.rumination import chew_score, resolve_video
        from ..behavior.features import parse_clip_start
        t = self.t(time)
        rows = self.rec[(self.rec.global_cow_id == cow_id)]
        r = self._nearest(rows, t, tol=10)
        if r is None:
            return ToolResult(f"{cow_id} not visible near {self.hms(t)}")
        path = resolve_video(r.source_video, self.video_index)
        st = parse_clip_start(r.source_video)
        if path is None or st is None:
            return ToolResult("video not available for chew analysis")
        off = (pd.Timestamp(r.t, unit="s") - st).total_seconds()
        s = chew_score(path, off, (r.bbox_x1, r.bbox_y1, r.bbox_x2, r.bbox_y2))
        verdict = "periodic chewing (rumination likely)" if s.get("peak_ratio", 0) > 3 else "no clear chewing rhythm"
        return ToolResult(f"{cow_id} on {r.camera_id} from {self.hms(r.t)}: peak_ratio {s.get('peak_ratio', 0):.2f} at "
                          f"{s.get('peak_hz', 0):.2f} Hz -> {verdict}", data=s)
