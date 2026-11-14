#!/usr/bin/env python3
"""Anchor moments: seconds where cows are confidently separate, so identity is unambiguous.

  python scripts/find_anchors.py --run outputs/level1_d1_1600 [--n 8]

A cow is CLEAR at second s when
  * it is >= --sep metres from every other observed cow (fused top-view position),
  * at least one camera sees it fully (visibility 'visible', not partial / occluded),
  * every camera that sees it agrees on its position within --agree metres,
  * the same track piece holds it for +-5 s in that camera (no switch around s).
A GLOBAL anchor is a second where all N cows are clear (the identity set is fixed there).
Also reports an ID-free check (cam1 boxes not touching any other box) so the result does not
depend on the identity resolver.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from src.behavior.features import load_records


def runs(mask_secs):
    """Lengths of consecutive-second runs."""
    s = np.sort(np.array(list(mask_secs)))
    if not len(s):
        return []
    br = np.flatnonzero(np.diff(s) > 1)
    starts = np.r_[0, br + 1]
    ends = np.r_[br, len(s) - 1]
    return [(int(s[a]), int(s[b]), int(s[b] - s[a] + 1)) for a, b in zip(starts, ends)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=None, help="number of animals (default: identities found)")
    ap.add_argument("--sep", type=float, default=2.5)
    ap.add_argument("--agree", type=float, default=2.0)
    a = ap.parse_args()

    rec = load_records([os.path.join(a.run, "identity", "tracking_records_resolved.csv")])
    rec["sec"] = np.round(rec.t).astype(np.int64)
    st = pd.read_csv(os.path.join(a.run, "state", "cow_state.csv"))
    vis = pd.read_csv(os.path.join(a.run, "state", "visibility.csv"))
    body = rec[rec.global_cow_id.notna() & (rec.get("box_role", "body") == "body")]
    n = a.n or body.global_cow_id.nunique()

    # stability: same (camera, track piece) for +-5 s
    piece = body.assign(piece=body.local_track_id * 100 + body.get("track_piece", 0))
    stable = set()
    for (cid, cam), g in piece.sort_values("sec").groupby(["global_cow_id", "camera_id"]):
        secs, pcs = g.sec.to_numpy(), g.piece.to_numpy()
        for i, s in enumerate(secs):
            lo, hi = np.searchsorted(secs, s - 5), np.searchsorted(secs, s + 5, side="right")
            if hi - lo >= 8 and len(set(pcs[lo:hi])) == 1:
                stable.add((cid, cam, s))
    # cross-camera agreement per (cow, sec)
    pos = body.dropna(subset=["world_x"]).groupby(["global_cow_id", "sec", "camera_id"])[["world_x", "world_y"]].mean()
    spread = {}
    for (cid, s), h in pos.groupby(level=[0, 1]):
        P = h.to_numpy()
        spread[(cid, s)] = float(np.max(np.linalg.norm(P[:, None] - P[None], axis=2))) if len(P) > 1 else 0.0
    vis_ok = vis[vis.status == "visible"]
    clear_view = {(r.cow_id, r.camera, r.sec) for r in vis_ok.itertuples()}

    obs = st[st.observed]
    per_sec = []
    clear_cows = {}
    for s, g in obs.groupby("sec"):
        P = g[["x", "y"]].to_numpy()
        D = np.linalg.norm(P[:, None] - P[None], axis=2) + np.eye(len(P)) * 1e9
        nn = D.min(1) if len(P) > 1 else np.full(len(P), 1e9)
        ok = []
        for cid, d in zip(g.cow_id, nn):
            cams = [c for c in ("cam1", "cam2", "cam3") if (cid, c, s) in clear_view and (cid, c, s) in stable]
            if d >= a.sep and cams and spread.get((cid, s), 0.0) <= a.agree:
                ok.append(cid)
        clear_cows[s] = ok
        per_sec.append((s, len(g), len(ok)))
    df = pd.DataFrame(per_sec, columns=["sec", "observed", "clear"])

    # ID-free check: cam1 boxes that touch no other box
    iso = []
    for s, g in rec[rec.camera_id == "cam1"].groupby("sec"):
        b = g[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy()
        pad = 0.01 * g.frame_width.iloc[0]
        free = 0
        for i in range(len(b)):
            touch = any(b[i, 0] - pad <= b[j, 2] and b[j, 0] <= b[i, 2] + pad and b[i, 1] - pad <= b[j, 3] and b[j, 1] <= b[i, 3] + pad
                        for j in range(len(b)) if j != i)
            free += not touch
        iso.append(free)

    out = {
        "seconds": int(len(df)), "animals": int(n),
        "clear_cows_per_second": {"median": float(df.clear.median()), "p90": float(df.clear.quantile(.9)), "max": int(df.clear.max())},
        "global_anchor_seconds": int((df.clear >= n).sum()),
        "seconds_with_clear>=n-1": int((df.clear >= n - 1).sum()),
        "seconds_with_clear>=n-2": int((df.clear >= n - 2).sum()),
        "global_anchor_runs": [(str(pd.Timestamp(a_, unit="s").time()), b_ - a_ + 1) for a_, b_, _ in runs(df[df.clear >= n].sec)][:10],
        "per_cow_clear_seconds": {c: sum(c in v for v in clear_cows.values()) for c in sorted(obs.cow_id.unique())},
        "id_free_cam1_isolated_boxes": {"median": float(np.median(iso)), "max": int(max(iso)) if iso else 0},
    }
    os.makedirs(os.path.join(a.run, "anchors"), exist_ok=True)
    df.to_csv(os.path.join(a.run, "anchors", "anchor_score.csv"), index=False)
    json.dump({str(k): v for k, v in clear_cows.items()}, open(os.path.join(a.run, "anchors", "clear_cows.json"), "w"))
    json.dump(out, open(os.path.join(a.run, "anchors", "anchor_summary.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
