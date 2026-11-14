#!/usr/bin/env python3
"""Can these cows be told apart by appearance? Train on the first part of a run, test on the
last part (temporal split, so neighbouring frames never leak between train and test).

  python scripts/appearance_test.py --run outputs/level1_d1_1600

Labels = identities from the agreement resolver, only confident, fully visible body boxes.
Features = DINOv2-S + coat colour (the same as the tracker). Classifier = logistic regression.
Reports accuracy per camera and across cameras (train cam1 -> test cam2, etc.); chance = 1/N.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

from src.agent.evidence import Embedder, FrameStore
from src.behavior.features import load_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--per-cow-cam", type=int, default=40)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--min-conf", type=float, default=0.8)
    a = ap.parse_args()

    rec = load_records([os.path.join(a.run, "identity", "tracking_records_resolved.csv")])
    rec["sec"] = np.round(rec.t).astype(np.int64)
    vis = pd.read_csv(os.path.join(a.run, "state", "visibility.csv"))
    clear = set(zip(vis[vis.status == "visible"].cow_id, vis[vis.status == "visible"].camera, vis[vis.status == "visible"].sec))
    b = rec[rec.global_cow_id.notna() & (rec.identity_confidence >= a.min_conf)]
    if "box_role" in b:
        b = b[b.box_role == "body"]
    b = b[[k in clear for k in zip(b.global_cow_id, b.camera_id, b.sec)]]
    t_split = rec.sec.min() + a.train_frac * (rec.sec.max() - rec.sec.min())

    # sample spread over time, both halves
    rows = []
    for (cid, cam), g in b.groupby(["global_cow_id", "camera_id"]):
        for part, h in (("train", g[g.sec < t_split]), ("test", g[g.sec >= t_split])):
            if len(h) == 0:
                continue
            k = min(len(h), a.per_cow_cam if part == "train" else a.per_cow_cam // 2)
            rows.append(h.iloc[np.linspace(0, len(h) - 1, k).astype(int)].assign(part=part))
    S = pd.concat(rows).sort_values(["camera_id", "t"])
    print(f"{len(S)} crops: " + ", ".join(f"{p} {n}" for p, n in S.part.value_counts().items()))

    frames, emb = FrameStore(rec), Embedder()
    X = []
    for r in S.itertuples():
        fr = frames.frame(r.camera_id, float(r.t))
        X.append(emb(fr, (r.bbox_x1, r.bbox_y1, r.bbox_x2, r.bbox_y2)) if fr is not None else None)
    keep = [x is not None for x in X]
    S, X = S[keep], normalize(np.array([x for x in X if x is not None]))
    y = S.global_cow_id.to_numpy()
    n_cows = len(set(y))

    def fit_eval(tr, te):
        if tr.sum() < 10 or te.sum() < 5 or len(set(y[tr])) < 2:
            return None
        clf = LogisticRegression(max_iter=2000, C=2.0).fit(X[tr], y[tr])
        known = np.isin(y[te], clf.classes_)
        return round(float((clf.predict(X[te][known]) == y[te][known]).mean()), 3), int(known.sum())

    res = {"cows": n_cows, "chance": round(1 / n_cows, 3)}
    tr, te = (S.part == "train").to_numpy(), (S.part == "test").to_numpy()
    res["all_cameras (train early -> test late)"] = fit_eval(tr, te)
    for cam in sorted(S.camera_id.unique()):
        c = (S.camera_id == cam).to_numpy()
        res[f"{cam} only"] = fit_eval(tr & c, te & c)
    for a_, b_ in (("cam1", "cam2"), ("cam1", "cam3"), ("cam2", "cam1")):
        res[f"train {a_} -> test {b_}"] = fit_eval((S.camera_id == a_).to_numpy(), (S.camera_id == b_).to_numpy())
    os.makedirs(os.path.join(a.run, "anchors"), exist_ok=True)
    json.dump(res, open(os.path.join(a.run, "anchors", "appearance_test.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
