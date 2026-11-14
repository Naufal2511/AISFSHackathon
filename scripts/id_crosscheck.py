#!/usr/bin/env python3
"""Cross-clip identity check: per COW_xx, crops from run A next to crops from run B.

  python scripts/id_crosscheck.py --runs outputs/run_d1_0719 outputs/run_d1_0734 \
      --camera cam1 --per-cow 3 --out outputs/run_d1_0734/id_crosscheck.jpg

If the gallery works, each row shows the same animal in every column.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import pandas as pd

from src.behavior.rumination import index_videos, resolve_video, _open

TILE = 180


def pick(df, n):
    """Largest boxes, spread over time."""
    df = df.assign(area=(df.bbox_x2 - df.bbox_x1) * (df.bbox_y2 - df.bbox_y1))
    df = df[df.area >= df.area.quantile(0.5)].sort_values("timestamp")
    if len(df) <= n:
        return df
    return df.iloc[np.linspace(0, len(df) - 1, n).astype(int)]


def crop(vp, ts, box):
    cap = _open(vp)
    cap.set(cv2.CAP_PROP_POS_MSEC, float(ts) * 1000)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return np.zeros((TILE, TILE, 3), np.uint8)
    x1, y1, x2, y2 = [int(max(0, v)) for v in box]
    c = fr[y1:y2, x1:x2]
    if c.size == 0:
        return np.zeros((TILE, TILE, 3), np.uint8)
    s = TILE / max(c.shape[:2])
    c = cv2.resize(c, None, fx=s, fy=s)
    out = np.zeros((TILE, TILE, 3), np.uint8)
    out[:c.shape[0], :c.shape[1]] = c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--camera", default="cam1")
    ap.add_argument("--per-cow", type=int, default=3)
    ap.add_argument("--video-root", default="input")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    vids = index_videos(a.video_root)
    recs = [pd.read_csv(os.path.join(r, "tracking_records.csv")) for r in a.runs]
    recs = [d[(d.camera_id == a.camera) & d.global_cow_id.notna()] for d in recs]
    cows = sorted(set().union(*[set(d.global_cow_id) for d in recs]))
    rows = []
    for cow in cows:
        tiles = []
        for d in recs:
            sel = pick(d[d.global_cow_id == cow], a.per_cow)
            for r in sel.itertuples():
                tiles.append(crop(resolve_video(r.source_video, vids), r.timestamp, (r.bbox_x1, r.bbox_y1, r.bbox_x2, r.bbox_y2)))
            tiles += [np.full((TILE, TILE, 3), 40, np.uint8)] * (a.per_cow - len(sel))
            tiles.append(np.full((TILE, 6, 3), 255, np.uint8))  # run separator
        label = np.full((TILE, 110, 3), 0, np.uint8)
        cv2.putText(label, cow, (6, TILE // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        rows.append(np.hstack([label] + tiles[:-1]))
    head = np.zeros((30, rows[0].shape[1], 3), np.uint8)
    x = 110
    for r in a.runs:
        cv2.putText(head, os.path.basename(r.rstrip("/")), (x + 4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        x += a.per_cow * TILE + 6
    sheet = np.vstack([head] + rows)
    cv2.imwrite(a.out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("wrote", a.out, sheet.shape[1], "x", sheet.shape[0])


if __name__ == "__main__":
    main()
