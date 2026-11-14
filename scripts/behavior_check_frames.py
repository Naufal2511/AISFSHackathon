#!/usr/bin/env python3
"""Visual sanity check: overlay cow ID + behavior label on a few frames per camera.

  python scripts/behavior_check_frames.py --records outputs/run_d1_0719/tracking_records.csv \
      --video-root input --n 6 --out outputs/run_d1_0719/behavior/check_sheet.jpg

Reads only n frames per camera (seeks), so it is cheap even on 4K footage.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np

from src.behavior.classifier import BehaviorClassifier
from src.behavior.rumination import index_videos, resolve_video, _open

COLORS = {
    "lying": (200, 120, 0), "standing": (0, 200, 0), "walking": (0, 220, 255), "feeding": (0, 0, 255),
    "drinking": (255, 0, 0), "ruminating (lying)": (200, 0, 200), "ruminating (standing)": (255, 0, 255),
    "grooming": (0, 140, 255), "social interaction": (255, 255, 0), "other / not visible": (128, 128, 128),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--video-root", default="input")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--tile-width", type=int, default=960)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    clf = BehaviorClassifier()
    tl = clf.run([a.records])
    rows = clf.rows.merge(tl[["global_cow_id", "sec", "behavior", "confidence", "evidence"]],
                          on=["global_cow_id", "sec"], how="left")
    vids = index_videos(a.video_root)
    cams = sorted(rows["camera_id"].unique())
    ts_all = np.sort(rows["timestamp"].unique())
    picks = ts_all[np.linspace(0, len(ts_all) - 1, a.n).astype(int)]

    grid = []
    for ts in picks:
        tiles = []
        for cam in cams:
            r = rows[(rows["camera_id"] == cam) & (rows["timestamp"] == ts)]
            src = rows[rows["camera_id"] == cam]["source_video"].iloc[0]
            cap = _open(resolve_video(src, vids))
            cap.set(cv2.CAP_PROP_POS_MSEC, float(ts) * 1000)
            ok, fr = cap.read()
            cap.release()
            if not ok:
                fr = np.zeros((1080, 1440, 3), np.uint8)
            s = a.tile_width / fr.shape[1]
            fr = cv2.resize(fr, None, fx=s, fy=s)
            for x in r.itertuples():
                c = COLORS.get(x.behavior, (255, 255, 255))
                p1 = (int(x.bbox_x1 * s), int(x.bbox_y1 * s)); p2 = (int(x.bbox_x2 * s), int(x.bbox_y2 * s))
                cv2.rectangle(fr, p1, p2, c, 2)
                lab = f"{x.global_cow_id} {x.behavior} {x.confidence:.2f}"
                cv2.putText(fr, lab, (p1[0], max(12, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
                cv2.putText(fr, lab, (p1[0], max(12, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
            cv2.putText(fr, f"{cam} t={ts:.0f}s", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            tiles.append(fr)
        h = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
        grid.append(np.hstack(tiles))
    sheet = np.vstack(grid)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    cv2.imwrite(a.out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"wrote {a.out} ({sheet.shape[1]}x{sheet.shape[0]})")


if __name__ == "__main__":
    main()
