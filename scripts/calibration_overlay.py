#!/usr/bin/env python3
"""Draw the calibrated pen model on a real frame to verify calibration visually.

  python scripts/calibration_overlay.py --camera cam1 --video "input/Day 1/Cam 01/Event20261107071945006.Avi" \
      --t 30 --out outputs/calibration_cam1.jpg

Draws: projected 2 m world grid + pen outline (cyan), metric zones (feed=red, drink=blue,
rest=green), the pixel ROI (yellow) and the pixel-space behavior zones (magenta).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import yaml

from src.calibration.camera_calibration import CameraCalibrator
from src.calibration.coordinate_transform import CoordinateTransformer
from src.video.video_reader import FrameReader


def polyline_world(tr, cam, pts_world, res, n=40):
    """Densely sample world segments so lens distortion shows as curves."""
    out = []
    for (x0, y0), (x1, y1) in zip(pts_world[:-1], pts_world[1:]):
        for a in np.linspace(0, 1, n):
            p = tr.world_to_pixel(cam, x0 + a * (x1 - x0), y0 + a * (y1 - y0), current_res=res)
            if p is not None and abs(p[0]) < 3 * res[0] and abs(p[1]) < 3 * res[1]:
                out.append(p)
    return np.array(out, np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--t", type=float, default=30.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1600)
    a = ap.parse_args()

    cap = cv2.VideoCapture(FrameReader(a.video).source)
    cap.set(cv2.CAP_PROP_POS_MSEC, a.t * 1000)
    ok, fr = cap.read()
    cap.release()
    assert ok, "could not read frame"
    h, w = fr.shape[:2]
    res = (w, h)
    tr = CoordinateTransformer(CameraCalibrator())
    with open("configs/pen_layout.yaml") as f:
        pen = yaml.safe_load(f)
    th = max(2, w // 800)

    for x in range(0, 13, 2):
        cv2.polylines(fr, [polyline_world(tr, a.camera, [(x, 0), (x, 30)], res)], False, (255, 255, 0), th)
    for y in range(0, 31, 2):
        cv2.polylines(fr, [polyline_world(tr, a.camera, [(0, y), (12, y)], res)], False, (255, 255, 0), th)
        p = tr.world_to_pixel(a.camera, 0.3, y, current_res=res)
        if p and 0 <= p[0] < w and 0 <= p[1] < h:
            cv2.putText(fr, f"Y{y}", (int(p[0]), int(p[1])), 0, w / 1600, (255, 255, 0), th)
    for name, col in (("feeding_zone", (0, 0, 255)), ("drinking_zone", (255, 0, 0)), ("resting_zone", (0, 200, 0))):
        poly = pen["zones"][name]["polygon_metric"]
        cv2.polylines(fr, [polyline_world(tr, a.camera, poly + [poly[0]], res)], False, col, th)

    from src.calibration.camera_zones import load_zones
    drawn = load_zones().get(a.camera, {})
    roi = drawn.get("roi") or pen["camera_rois"].get(a.camera, {}).get("normalized_polygon")
    if roi:
        cv2.polylines(fr, [(np.array(roi) * [w, h]).astype(np.int32)], True, (0, 255, 255), th)
    for zname in ("feed_bunk", "waterer"):  # hand-drawn image zones (magenta)
        if zname in drawn:
            P = (np.array(drawn[zname]) * [w, h]).astype(np.int32)
            cv2.polylines(fr, [P], True, (255, 0, 255), th)
            cv2.putText(fr, zname, tuple(P[0]), 0, w / 1600, (255, 0, 255), th)

    s = a.width / w
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    cv2.imwrite(a.out, cv2.resize(fr, None, fx=s, fy=s))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
