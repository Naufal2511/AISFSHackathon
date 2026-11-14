#!/usr/bin/env python3
"""Build the top-view world state + per-camera visibility from identity-resolved records.

  python scripts/build_world_state.py --records outputs/level1_d1_1600/identity/tracking_records_resolved.csv \
      --out outputs/level1_d1_1600/state
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import yaml

from src.behavior.features import load_records
from src.calibration.camera_calibration import CameraCalibrator
from src.calibration.coordinate_transform import CoordinateTransformer
from src.calibration.pen_boundary import PenBoundaryManager
from src.state.world_state import WorldState, render_top_view


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t = time.time()
    rec = load_records([a.records])
    cams = yaml.safe_load(open("configs/cameras.yaml"))["cameras"]
    cam_pos = {k: v.get("mount_position_estimated") for k, v in cams.items()}
    ws = WorldState(CoordinateTransformer(CameraCalibrator()), cam_pos, PenBoundaryManager()).build(rec)
    ws.save(a.out)
    v = ws.visibility
    summ = {
        "cow_seconds": int(len(ws.state)), "observed_share": round(float(ws.state.observed.mean()), 3),
        "visibility_share": v.status.value_counts(normalize=True).round(3).to_dict(),
        "facing_share_when_visible": v[v.status.isin(["visible", "partial"])].facing.value_counts(normalize=True).round(3).to_dict(),
        "anomalies": ws.anomalies.type.value_counts().to_dict(),
        "seconds": round(time.time() - t, 1),
    }
    json.dump(summ, open(os.path.join(a.out, "state_summary.json"), "w"), indent=1)
    img = render_top_view(ws.state, sorted(ws.state.cow_id.unique()), int(ws.state.sec.min()), int(ws.state.sec.max()),
                          sum(ws.heat.values()) if ws.heat else None, ws.anomalies, cam_pos)
    cv2.imwrite(os.path.join(a.out, "trajectories_heatmap.jpg"), img)
    print(json.dumps(summ, indent=1))
    print(ws.anomalies.head(12).to_string())


if __name__ == "__main__":
    main()
