#!/usr/bin/env python3
"""Export a run in the team data contract (docs/DATA_CONTRACT.md).

  python scripts/export_contract.py --run outputs/level1_d1_1600

Writes <run>/contract/observations.csv  (one row per box per second per camera)
       <run>/contract/cow_state.csv     (one row per cow per second, fused over cameras)
The behavior columns of observations.csv are left for the per-box behavior model; cow_state.csv
carries the current fused behavior (model = the classifier that produced it).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import yaml

from src.behavior.features import load_records
from src.behavior.taxonomy import BEHAVIORS

P_COLS = ["p_lying", "p_standing", "p_walking", "p_feeding", "p_drinking", "p_rum_lying", "p_rum_standing",
          "p_grooming", "p_social", "p_other"]
EVIDENCE = {0.95: "agreement", 0.85: "continuity", 0.8: "seed", 0.7: "appearance", 0.55: "attached", 0.5: "part"}


def iso(t):
    return pd.to_datetime(t, unit="s").dt.strftime("%Y-%m-%dT%H:%M:%S")


def zone_of(x, y, zones):
    from matplotlib.path import Path
    out = np.full(len(x), "other", object)
    for name, label in (("resting_zone", "rest"), ("drinking_zone", "waterer"), ("feeding_zone", "bunk")):
        poly = zones.get(name, {}).get("polygon_metric")
        if poly:
            inside = Path(poly).contains_points(np.c_[x, y])
            out[inside] = label
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--behavior-model", default=None, help="default: <run>/behavior/model.txt, else heuristic-fusion-v1")
    a = ap.parse_args()
    if a.behavior_model is None:
        mt = os.path.join(a.run, "behavior", "model.txt")
        a.behavior_model = open(mt).read().strip() if os.path.exists(mt) else "heuristic-fusion-v1"
    out_dir = os.path.join(a.run, "contract")
    os.makedirs(out_dir, exist_ok=True)
    cams = yaml.safe_load(open("configs/cameras.yaml"))["cameras"]
    zones = yaml.safe_load(open("configs/pen_layout.yaml")).get("zones", {})

    rec = load_records([os.path.join(a.run, "identity", "tracking_records_resolved.csv")])
    rec["sec"] = np.round(rec.t).astype(np.int64)
    vis = pd.read_csv(os.path.join(a.run, "state", "visibility.csv"))
    st = pd.read_csv(os.path.join(a.run, "state", "cow_state.csv"))
    an = pd.read_csv(os.path.join(a.run, "state", "anomalies.csv"))
    tl_path = next((p for p in (os.path.join(a.run, "agent", "behavior", "behavior_timeline_1s.csv"),
                                os.path.join(a.run, "behavior", "behavior_timeline_1s.csv")) if os.path.exists(p)), None)

    # ---------------- observations.csv
    o = pd.DataFrame({
        "wall_time": iso(rec.t), "file": rec.source_video.str.replace("remux_", "").str.replace(".mp4", ".Avi"),
        "offset_s": rec.timestamp.round(1), "camera": rec.camera_id,
        "camera_name": rec.camera_id.map(lambda c: cams.get(c, {}).get("name", c)),
        "track_id": rec.local_track_id.astype(int),
        "track_piece": rec.get("track_piece", pd.Series(0, index=rec.index)).fillna(0).astype(int),
        "cow_id": rec.global_cow_id, "id_conf": rec.identity_confidence.round(2),
        "id_evidence": rec.identity_confidence.round(2).map(lambda v: EVIDENCE.get(v, "")),
        "box_role": rec.get("box_role", pd.Series("body", index=rec.index)),
        "x1": rec.bbox_x1.round(0).astype(int), "y1": rec.bbox_y1.round(0).astype(int),
        "x2": rec.bbox_x2.round(0).astype(int), "y2": rec.bbox_y2.round(0).astype(int),
        "frame_w": rec.frame_width.astype(int), "frame_h": rec.frame_height.astype(int),
        "det_conf": rec.detection_confidence.round(3),
        "world_x": rec.world_x.round(2), "world_y": rec.world_y.round(2), "sec": rec.sec,
    })
    v = vis.rename(columns={"cow_id": "cow_id", "camera": "camera", "status": "visibility", "occluder": "occluded_by"})
    o = o.merge(v[["cow_id", "camera", "sec", "visibility", "occluded_by", "facing"]], how="left",
                on=["cow_id", "camera", "sec"])
    for c in ["behavior", "beh_conf"] + P_COLS + ["model"]:
        o[c] = ""
    o = o.drop(columns="sec").sort_values(["wall_time", "camera", "track_id"])
    o.to_csv(os.path.join(out_dir, "observations.csv"), index=False)

    # ---------------- cow_state.csv
    s = st.copy()
    s["wall_time"] = iso(s.sec)
    s["predicted"] = ~s.observed
    vv = vis[vis.status.isin(["visible", "partial"])]
    cams_vis = vv.groupby(["cow_id", "sec"]).camera.agg(lambda c: ",".join(sorted(set(c)))).rename("cams_visible")
    full = vis[vis.status == "visible"]
    best_id = full.sort_values("box_area", ascending=False).drop_duplicates(["cow_id", "sec"]).set_index(["cow_id", "sec"]).camera.rename("best_cam_identity")
    head = full.assign(k=(full.facing == "toward").astype(int) * 10 + full.box_area.fillna(0))
    best_head = head.sort_values("k", ascending=False).drop_duplicates(["cow_id", "sec"]).set_index(["cow_id", "sec"]).camera.rename("best_cam_head")
    s = s.join(cams_vis, on=["cow_id", "sec"]).join(best_id, on=["cow_id", "sec"]).join(best_head, on=["cow_id", "sec"])
    s["zone"] = zone_of(s.x.to_numpy(), s.y.to_numpy(), zones)
    s["nearest_cow"], s["nearest_m"] = "", np.nan
    for sec, g in s.groupby("sec"):
        if len(g) < 2:
            continue
        P = g[["x", "y"]].to_numpy()
        D = np.linalg.norm(P[:, None] - P[None], axis=2) + np.eye(len(P)) * 1e9
        j = D.argmin(1)
        s.loc[g.index, "nearest_cow"] = g.cow_id.to_numpy()[j]
        s.loc[g.index, "nearest_m"] = D[np.arange(len(P)), j].round(2)
    if tl_path:
        tl = pd.read_csv(tl_path)
        tl["sec"] = ((pd.to_datetime(tl.wall_time) - pd.Timestamp("1970-01-01")).dt.total_seconds()).astype(np.int64)
        pmap = {f"p::{b}": c for b, c in zip(BEHAVIORS, P_COLS)}
        keep = tl[["cow_id", "sec", "behavior", "confidence", "evidence"] + [k for k in pmap if k in tl]].rename(
            columns={"behavior": "behavior_fused", "confidence": "beh_conf_fused", "evidence": "beh_evidence", **pmap})
        s = s.merge(keep, how="left", on=["cow_id", "sec"])
        # sparse sampling (e.g. 1 frame / 5 s): the per-second state holds each sampled label until the next sample
        from src.behavior.fusion import sampling_step
        step = sampling_step(keep.sec.to_numpy())
        if step > 1:
            s = s.sort_values(["cow_id", "sec"])
            cols = [c for c in keep.columns if c not in ("cow_id", "sec")]
            s[cols] = s.groupby("cow_id")[cols].ffill(limit=step - 1)
        s["model"] = a.behavior_model
    anom = []
    for r in an.itertuples():
        for sec in range(int(r.t0), int(r.t1) + 1):
            anom.append((r.cow_id, sec, r.type))
    if anom:
        A = pd.DataFrame(anom, columns=["cow_id", "sec", "anomaly"]).groupby(["cow_id", "sec"]).anomaly.agg(lambda x: ",".join(sorted(set(x))))
        s = s.join(A, on=["cow_id", "sec"])
    cols = ["wall_time", "cow_id", "x", "y", "pos_sd_m", "speed", "heading_deg", "predicted", "n_cams", "cams_visible",
            "best_cam_identity", "best_cam_head", "zone", "nearest_cow", "nearest_m", "behavior_fused", "beh_conf_fused",
            "beh_evidence"] + P_COLS + ["model", "anomaly"]
    s = s.rename(columns={"speed": "speed_mps"})
    cols = [c.replace("speed", "speed_mps") if c == "speed" else c for c in cols]
    s[[c for c in cols if c in s.columns]].sort_values(["wall_time", "cow_id"]).to_csv(
        os.path.join(out_dir, "cow_state.csv"), index=False)
    meta = {"run": a.run, "observations_rows": int(len(o)), "cow_state_rows": int(len(s)),
            "cows": sorted(s.cow_id.unique().tolist()), "span": [str(s.wall_time.min()), str(s.wall_time.max())],
            "contract": "docs/DATA_CONTRACT.md"}
    json.dump(meta, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
