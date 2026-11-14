"""Hand-drawn per-camera pixel zones (pen ROI, feed bunk, waterer).

Drawn on real frames with scripts/draw_zones.py and stored normalized [0,1] in
configs/camera_zones.yaml. They take precedence over pen_layout.yaml ROIs and over the
metric (homography) zones, because a polygon drawn on the image is exact for that
camera even where the metric calibration is approximate (fisheye, cam3).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import cv2
import numpy as np
import yaml

DEFAULT_PATH = "configs/camera_zones.yaml"
LAYERS = ("roi", "feed_bunk", "waterer")


def load_zones(path: str = DEFAULT_PATH) -> Dict[str, Dict[str, List[List[float]]]]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return {cam: {k: v for k, v in (layers or {}).items() if k in LAYERS and v and len(v) >= 3}
            for cam, layers in data.get("cameras", {}).items()}


EDGE_SNAP = 0.015  # points this close to an image edge are put ON the edge


def _snap(v: float) -> float:
    """Cows cut off by the frame have boxes ending exactly at the border; a polygon that
    stops a few pixels short would exclude them, so near-edge clicks snap to the edge."""
    v = min(max(float(v), 0.0), 1.0)
    return 0.0 if v < EDGE_SNAP else 1.0 if v > 1.0 - EDGE_SNAP else v


def save_zones(zones: Dict[str, Dict[str, List[List[float]]]], path: str = DEFAULT_PATH, meta: Optional[Dict] = None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = {"cameras": {}}
    for cam in sorted(zones):
        out["cameras"][cam] = {k: [[round(_snap(x), 4), round(_snap(y), 4)] for x, y in pts]
                               for k, pts in zones[cam].items() if pts and len(pts) >= 3}
    if meta:
        out["meta"] = meta
    header = ("# Per-camera pixel zones drawn on real frames (normalized [0,1]).\n"
              "# Edit with: python scripts/draw_zones.py --camera <cam> --video <clip>\n"
              "#   roi       : Pen 6 ground (hoof contact must be inside; excludes Pen 7 / alley)\n"
              "#   feed_bunk : where heads go when eating (rail to trough)\n"
              "#   waterer   : waterer plus ~head reach around it\n")
    with open(path, "w") as f:
        f.write(header)
        yaml.safe_dump(out, f, sort_keys=False, default_flow_style=None)


def bbox_fraction_in(poly_norm, bbox, frame_w: float, frame_h: float, n: int = 7) -> float:
    """Fraction of an n x n grid of points over the bbox that falls inside the polygon."""
    poly = (np.asarray(poly_norm, np.float32) * np.array([frame_w, frame_h], np.float32)).reshape(-1, 1, 2)
    x1, y1, x2, y2 = bbox
    hit = 0
    for x in np.linspace(x1, x2, n):
        for y in np.linspace(y1, y2, n):
            hit += cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0
    return hit / float(n * n)
