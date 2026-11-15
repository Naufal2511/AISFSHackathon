#!/usr/bin/env python3
"""Per-cow risk assets from a tracking run's assets_raw/ (recorded by main.py) + resolved identities.

  python scripts/build_risk_assets.py --raw outputs/run_d1_1600_h --run outputs/level1_d1_1600_h

Writes <run>/assets/: gallery.csv, motion.csv, breathing.csv (+ breathing/*.jpg spectra), jerks.csv,
feed_events.csv, clips.csv (+ clips/*.mp4), summary.json. Options: --skip-breathing --skip-clips.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import yaml

from src.agent.evidence import FrameStore
from src.assets import post
from src.behavior.features import load_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="tracking run dir with assets_raw/")
    ap.add_argument("--run", required=True, help="Level 1 run dir with identity/tracking_records_resolved.csv")
    ap.add_argument("--video-root", default="input")
    ap.add_argument("--skip-breathing", action="store_true")
    ap.add_argument("--skip-clips", action="store_true")
    ap.add_argument("--head-signs", action="store_true", help="label gallery crops with a fast TAMU VLM (head, ears, back, muzzle) + discharge image check")
    ap.add_argument("--head-backend", default="tamu_luna")
    ap.add_argument("--head-per-hour", type=int, default=2)
    a = ap.parse_args()
    cfg = yaml.safe_load(open("configs/risk.yaml"))["assets"]
    t0 = time.time()
    raw = os.path.join(a.raw, "assets_raw")
    out = os.path.join(a.run, "assets")
    os.makedirs(out, exist_ok=True)
    res_path = os.path.join(a.run, "identity", "tracking_records_resolved.csv")
    resolved = pd.read_csv(res_path, low_memory=False)
    summary = {}

    mot = post.join_ids(pd.read_csv(os.path.join(raw, "motion.csv")), resolved)
    mot.to_csv(os.path.join(out, "motion.csv"), index=False)
    summary["motion_rows"] = len(mot)
    gal = post.join_ids(pd.read_csv(os.path.join(raw, "gallery.csv")), resolved)
    gal["path"] = [os.path.relpath(os.path.join(raw, f)) for f in gal.file]
    gal.to_csv(os.path.join(out, "gallery.csv"), index=False)
    summary["gallery"] = gal.groupby("cow_id").size().to_dict()
    frames = pd.read_csv(os.path.join(raw, "frames.csv"))
    fe = post.feed_events(frames, cfg)
    fe.to_csv(os.path.join(out, "feed_events.csv"), index=False)
    summary["feed_events"] = len(fe)
    state_p = os.path.join(a.run, "contract", "cow_state.csv")
    state = None
    if os.path.exists(state_p):
        state = pd.read_csv(state_p, low_memory=False)
        state["sec"] = ((pd.to_datetime(state.wall_time) - pd.Timestamp("1970-01-01")).dt.total_seconds()).astype(int)
    j = post.jerks(mot, state, cfg)
    j.to_csv(os.path.join(out, "jerks.csv"), index=False)
    summary["jerks"] = len(j)

    fs = FrameStore(load_records([res_path]), a.video_root)
    if not a.skip_breathing:
        b = post.breathing(mot, fs, out, cfg)
        b.to_csv(os.path.join(out, "breathing.csv"), index=False)
        summary["breathing_windows"] = len(b)
        summary["breathing_valid"] = int(b.valid.sum()) if len(b) else 0
    if not a.skip_clips:
        c = post.clips(mot, fs, out, cfg)
        c.to_csv(os.path.join(out, "clips.csv"), index=False)
        summary["clips"] = len(c)
    if a.head_signs and len(gal):
        from src.assets import head_signs
        from src.risk.agent import Backends
        acfg = yaml.safe_load(open("configs/agent.yaml"))
        be = Backends(acfg, [a.head_backend, "tamu_flash", "tamu_opus"])
        hs = head_signs.run(gal, be, per_hour=a.head_per_hour, model_label=a.head_backend)
        if len(hs) and state is not None:   # a crop can hide the bunk: use the tracked position too
            ys = state.set_index(["cow_id", "sec"]).y
            hs["at_bunk"] = [bool(ys.get((c, int(t)), np.nan) <= 4.0) for c, t in zip(hs.cow_id, hs.wall_t)]
            hs["feeding"] = hs.feeding.astype(str).str.lower().eq("true") | hs.at_bunk
        hs.to_csv(os.path.join(out, "head_signs.csv"), index=False)
        summary["head_signs"] = len(hs)
        if len(hs):
            summary["head_signs_models"] = sorted(set(be.used))
            summary["discharge_flags"] = int(hs.discharge_flag.sum())
    summary["seconds"] = round(time.time() - t0, 1)
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1, default=str)
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
