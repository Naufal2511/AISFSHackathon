#!/usr/bin/env python3
"""Learn identities by cross-camera agreement over a run, then relabel the tracking records.

  python scripts/resolve_identities.py --records outputs/run_d1_1600/tracking_records.csv \
      --out outputs/run_d1_1600/identity

Writes tracking_records_resolved.csv (same columns, global_cow_id replaced), identity_report.json
and identity_sheet.jpg (one row per cow: its crops from cam1 | cam2 | cam3).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import pandas as pd

from src.agent.evidence import Embedder, FrameStore, crop, label, tile
from src.behavior.features import load_records
from src.calibration.camera_calibration import CameraCalibrator
from src.calibration.coordinate_transform import CoordinateTransformer
from src.tracking.identity_resolver import IdentityResolver, cross_camera_consistency, world_positions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--video-root", default="input")
    ap.add_argument("--pen-size", type=int, default=10, help="herd size cap (use the anchor estimate, scripts/herd_size.py)")
    ap.add_argument("--crops-per-track", type=int, default=3)
    ap.add_argument("--no-appearance", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    raw = pd.read_csv(a.records)
    rec = load_records([a.records])
    rec = world_positions(rec, CoordinateTransformer(CameraCalibrator()))
    print(f"{len(rec)} boxes; consistency of the tracker's IDs:", cross_camera_consistency(rec))

    frames = FrameStore(rec, a.video_root)
    emb = None if a.no_appearance else Embedder()
    cache = {}

    def embed(key):
        cam, tid = key
        g = rec[res.row_key.reindex(rec.index).map(lambda k: k == key).astype(bool)]  # rows of this track piece
        area = (g.bbox_x2 - g.bbox_x1) * (g.bbox_y2 - g.bbox_y1)
        g = g.assign(q=g.detection_confidence * np.sqrt(area)).sort_values("q", ascending=False)
        picks = g.iloc[np.linspace(0, min(len(g), 30) - 1, min(a.crops_per_track, len(g))).astype(int)]
        vs = []
        for r in picks.itertuples():
            fr = frames.frame(cam, float(r.t))
            if fr is not None:
                vs.append(emb(fr, (r.bbox_x1, r.bbox_y1, r.bbox_x2, r.bbox_y2)))
                cache.setdefault(key, (fr, (r.bbox_x1, r.bbox_y1, r.bbox_x2, r.bbox_y2)))
        if not vs:
            return None
        v = np.mean(vs, 0)
        return v / (np.linalg.norm(v) + 1e-9)

    import yaml
    cam_pos = {k: v.get("mount_position_estimated") for k, v in yaml.safe_load(open("configs/cameras.yaml"))["cameras"].items()}
    from src.behavior.fusion import sampling_step
    step = sampling_step(rec.t.round().astype("int64").to_numpy())     # 1 at 1 fps, 5 at 1 frame / 5 s
    res = IdentityResolver({"pen_size": a.pen_size, "cam_pos": cam_pos, "sample_step": step})
    mapping = res.resolve(rec, None if emb is None else embed)

    key = list(res.row_key.reindex(rec.index))   # (camera, track piece) per row: tracks split at ID switches
    rec["resolved_id"] = [mapping.get(k) for k in key]
    # head / leg / sliver boxes carry the cow's ID but not a position (a head in the trough is not where the cow stands)
    is_part = np.array([k in res.fragment_keys for k in key])
    rec.loc[is_part, ["gx", "gy"]] = np.nan
    after = cross_camera_consistency(rec, "resolved_id")
    before = cross_camera_consistency(rec, "global_cow_id")
    report = {**res.report, "consistency_before": before, "consistency_after": after}
    json.dump(report, open(os.path.join(a.out, "identity_report.json"), "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in report.items() if k != "identities"}, indent=1, default=str))
    for cid, info in sorted(report["identities"].items()):
        print(f"  {cid}: {info['seconds']:4d} s, {info['tracklets']:2d} tracks, cameras {info['cameras']}")

    out = raw.copy()
    out["global_cow_id"] = [mapping.get(k) for k in key]
    out["identity_confidence"] = [res.confidence.get(k) for k in key]
    out["track_piece"] = [k[1] // 10000 if isinstance(k, tuple) else 0 for k in key]
    # positions from the current calibration of every camera (cam2/cam3 were refitted from cows)
    assert len(rec) == len(out) and (rec.local_track_id.to_numpy() == out.local_track_id.to_numpy()).all()
    out["world_x"], out["world_y"] = rec.gx.round(3).to_numpy(), rec.gy.round(3).to_numpy()
    out["box_role"] = np.where(is_part, "part", "body")
    out.to_csv(os.path.join(a.out, "tracking_records_resolved.csv"), index=False)

    # identity sheet: per cow, up to 3 crops per camera
    rows = []
    for cid in sorted(report["identities"]):
        cells = []
        for cam in ("cam1", "cam2", "cam3"):
            ks = [k for k, v in mapping.items() if v == cid and k[0] == cam]
            ks = sorted(ks, key=lambda k: -res.T[k].n)[:3]
            for k in ks:
                if k not in cache and emb is None:
                    continue
                if k in cache:
                    fr, bb = cache[k]
                    cells.append(label(crop(fr, bb, pad=0.05, size=200), f"{cam} t{k[1]}"))
            while len(cells) % 3:
                cells.append(np.zeros((200, 200, 3), np.uint8))
        rows.append(np.hstack([np.full((200, 120, 3), 30, np.uint8), tile(cells, cols=len(cells), cell=200)]))
        cv2.putText(rows[-1], cid, (6, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if rows:
        wmax = max(r.shape[1] for r in rows)
        sheet = np.vstack([np.hstack([r, np.zeros((r.shape[0], wmax - r.shape[1], 3), np.uint8)]) for r in rows])
        cv2.imwrite(os.path.join(a.out, "identity_sheet.jpg"), sheet)
    frames.close()
    print("->", a.out)


if __name__ == "__main__":
    main()
