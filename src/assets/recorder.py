"""Risk assets recorded during the tracking pass (frames are decoded anyway, so this is cheap).

Written to <run>/assets_raw/ keyed by camera + local track id + clip time, because cow IDs only
exist after identity resolution (scripts/build_risk_assets.py joins them later):

  frames.csv    per camera per step: wall time, daylight flag, bunk-trough signal (feed delivery)
  motion.csv    per box per step: motion energy inside the box vs the previous step (activity that does
                not depend on behavior labels; stillness for breathing windows; sudden jerks)
  gallery.csv + gallery/<cam>/*.jpg
                full-resolution crops of close daylight boxes (head / discharge / ear checks), at most one
                per track per camera every `gallery_every_s`
"""
from __future__ import annotations

import csv
import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import yaml

DEFAULTS = {
    "gallery_min_px": 450,      # box height in source pixels (discharge needs a close view)
    "gallery_every_s": 600,     # per track per camera
    "gallery_max_side": 1280,
    "gallery_quality": 88,
    "gallery_night": False,     # IR night crops cannot show discharge
    "motion_scale": 0.125,      # motion computed on a 1/8 size grey frame
    "bunk_camera": "cam2",
}


def is_daylight(img: np.ndarray) -> bool:
    if img is None or img.ndim != 3:
        return False
    s = cv2.resize(img, (64, 48)).astype(np.int16)
    return float(np.abs(s[..., 2] - s[..., 1]).mean() + np.abs(s[..., 1] - s[..., 0]).mean()) > 6.0


def clip_wall(source_video: Optional[str], offset: float) -> float:
    from ..behavior.features import parse_clip_start
    st = parse_clip_start(source_video) if source_video else None
    if st is None:
        return float("nan")
    return (st - pd.Timestamp("1970-01-01")).total_seconds() + float(offset)


class AssetRecorder:
    def __init__(self, out_dir: str, cfg: Optional[Dict] = None, zones_path: str = "configs/camera_zones.yaml"):
        self.c = {**DEFAULTS, **(cfg or {})}
        self.dir = os.path.join(out_dir, "assets_raw")
        os.makedirs(os.path.join(self.dir, "gallery"), exist_ok=True)
        self._f = {k: open(os.path.join(self.dir, f"{k}.csv"), "w", newline="") for k in ("frames", "motion", "gallery")}
        self._w = {k: csv.writer(f) for k, f in self._f.items()}
        self._w["frames"].writerow(["wall_t", "camera", "source_video", "offset_s", "daylight", "bunk_mean", "bunk_std", "frame_motion"])
        self._w["motion"].writerow(["wall_t", "camera", "source_video", "offset_s", "track_id", "x1", "y1", "x2", "y2", "box_h", "motion"])
        self._w["gallery"].writerow(["wall_t", "camera", "source_video", "offset_s", "track_id", "x1", "y1", "x2", "y2", "box_h", "file"])
        zones = yaml.safe_load(open(zones_path)) if os.path.exists(zones_path) else {}
        self.bunk_poly = {cam: np.array(z.get("feed_bunk"), float) for cam, z in (zones.get("cameras") or zones).items()
                          if isinstance(z, dict) and z.get("feed_bunk")}
        self.prev: Dict[str, Tuple[float, np.ndarray]] = {}
        self.cur: Dict[str, Dict] = {}
        self.last_gallery: Dict[Tuple[str, int], float] = {}
        self._mask_cache: Dict = {}
        self.n = {"frames": 0, "boxes": 0, "gallery": 0}

    def _bunk_mask(self, cam, shape):
        key = (cam, shape)
        if key not in self._mask_cache:
            m = np.zeros(shape, np.uint8)
            if cam in self.bunk_poly:
                pts = (self.bunk_poly[cam] * [shape[1], shape[0]]).astype(np.int32)
                cv2.fillPoly(m, [pts], 1)
            self._mask_cache[key] = m.astype(bool)
        return self._mask_cache[key]

    def frame(self, cam: str, frame: np.ndarray, source_video: Optional[str], offset: float):
        """Call once per camera per step, before the boxes."""
        wall = clip_wall(source_video, offset)
        s = self.c["motion_scale"]
        g = cv2.cvtColor(cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        g = cv2.GaussianBlur(g, (5, 5), 0)
        prev = self.prev.get(cam)
        diff = None
        if prev is not None and 0 < wall - prev[0] <= 2.5:
            diff = np.abs(g - prev[1])
        day = is_daylight(frame)
        bm = bs = np.nan
        if cam == self.c["bunk_camera"] or cam in self.bunk_poly:
            mk = self._bunk_mask(cam, g.shape)
            if mk.any():
                bm, bs = float(g[mk].mean()), float(g[mk].std())
        self._w["frames"].writerow([f"{wall:.1f}", cam, source_video, f"{offset:.2f}", int(day), _r(bm), _r(bs),
                                    _r(float(diff.mean()) if diff is not None else np.nan)])
        self.cur[cam] = {"wall": wall, "diff": diff, "day": day, "frame": frame, "sv": source_video, "off": offset}
        self.prev[cam] = (wall, g)
        self.n["frames"] += 1

    def track(self, cam: str, track_id: int, bbox):
        """Call per tracked box after `frame` for the same camera and step."""
        st = self.cur.get(cam)
        if st is None:
            return
        x1, y1, x2, y2 = [float(v) for v in bbox]
        h = y2 - y1
        mot = np.nan
        if st["diff"] is not None:
            s = self.c["motion_scale"]
            d = st["diff"][max(0, int(y1 * s)):max(1, int(y2 * s)), max(0, int(x1 * s)):max(1, int(x2 * s))]
            if d.size:
                mot = float(d.mean())
        self._w["motion"].writerow([f"{st['wall']:.1f}", cam, st["sv"], f"{st['off']:.2f}", int(track_id),
                                    int(x1), int(y1), int(x2), int(y2), int(h), _r(mot)])
        self.n["boxes"] += 1
        if h < self.c["gallery_min_px"] or (not st["day"] and not self.c["gallery_night"]):
            return
        key = (cam, int(track_id))
        if st["wall"] - self.last_gallery.get(key, -1e12) < self.c["gallery_every_s"]:
            return
        fr = st["frame"]
        H, W = fr.shape[:2]
        px, py = 0.06 * (x2 - x1), 0.06 * h
        crop = fr[int(max(0, y1 - py)):int(min(H, y2 + py)), int(max(0, x1 - px)):int(min(W, x2 + px))]
        if crop.size == 0:
            return
        m = max(crop.shape[:2])
        if m > self.c["gallery_max_side"]:
            crop = cv2.resize(crop, None, fx=self.c["gallery_max_side"] / m, fy=self.c["gallery_max_side"] / m, interpolation=cv2.INTER_AREA)
        rel = os.path.join("gallery", cam, f"{int(st['wall'])}_{int(track_id)}.jpg")
        os.makedirs(os.path.join(self.dir, "gallery", cam), exist_ok=True)
        cv2.imwrite(os.path.join(self.dir, rel), crop, [cv2.IMWRITE_JPEG_QUALITY, self.c["gallery_quality"]])
        self._w["gallery"].writerow([f"{st['wall']:.1f}", cam, st["sv"], f"{st['off']:.2f}", int(track_id),
                                     int(x1), int(y1), int(x2), int(y2), int(h), rel])
        self.last_gallery[key] = st["wall"]
        self.n["gallery"] += 1

    def reset_camera(self, cam: str):
        """Footage gap: do not compute motion across it."""
        self.prev.pop(cam, None)

    def close(self):
        for f in self._f.values():
            f.close()
        return dict(self.n)


def _r(v):
    return "" if v is None or not np.isfinite(v) else f"{v:.3f}"
