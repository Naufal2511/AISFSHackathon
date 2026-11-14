#!/usr/bin/env python3
"""Calibrate a camera to pen metres from the COWS, using a calibrated reference camera.

At every second the reference camera (cam1) already places each cow on the pen map. The
target camera (cam2 / cam3) sees the same cows from elsewhere. We search for the lens + ground
homography that puts the target's cows on top of the reference's cows at the same moments,
over the whole run, without knowing which cow is which (RANSAC over order-preserving
pairings, then robust refinement on the inliers).

  python scripts/calibrate_from_tracks.py --records outputs/run_d1_1600/tracking_records.csv \
      --target cam3 --known "0.78,0.5855=0,0" "0.565,0.5297=12,0"

Ground point used for each box: bottom-centre (hooves), except for cameras looking down onto
the bunk (cam2), where the box centre is used. Writes the fitted model as calibration points
(undistorted pixels) into a YAML snippet and prints the fit quality.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def undistort(p, k1, w, h):
    c = np.array([w / 2, h / 2])
    R = np.hypot(w / 2, h / 2)
    d = p - c
    r2 = (np.hypot(d[:, 0], d[:, 1]) / R) ** 2
    return c + d / (1 + k1 * r2)[:, None]


def apply_h(H, p):
    q = np.c_[p, np.ones(len(p))] @ H.T
    return q[:, :2] / q[:, 2:3]


def fit_h(src, dst):
    import cv2
    H, _ = cv2.findHomography(src.astype(np.float32), dst.astype(np.float32), 0)
    return H


def ground_points(g, cam, overhead):
    if cam in overhead:
        return np.c_[(g.bbox_x1 + g.bbox_x2) / 2, (g.bbox_y1 + g.bbox_y2) / 2]
    return np.c_[(g.bbox_x1 + g.bbox_x2) / 2, g.bbox_y2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--reference", default="cam1")
    ap.add_argument("--known", nargs="*", default=[], help='"u,v=X,Y" normalized image -> pen metres')
    ap.add_argument("--k1", type=float, nargs="+", default=[0.0, -0.1, -0.2, -0.3, -0.4])
    ap.add_argument("--overhead", nargs="*", default=["cam2"])
    ap.add_argument("--tol", type=float, default=1.8, help="metres for a cow to count as matched")
    ap.add_argument("--hyp", type=int, default=6000, help="hypotheses per k1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--on-line", nargs="*", default=[],
                    help='"u,v=X:12" or "u,v=Y:0": image point known to lie on a fence line')
    ap.add_argument("--fix-k1", action="store_true", help="keep the lens value fixed (e.g. same lens as cam1)")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    d = pd.read_csv(a.records)
    ref = d[(d.camera_id == a.reference) & d.world_x.notna()]
    tgt = d[d.camera_id == a.target]
    W, Hh = int(tgt.frame_width.iloc[0]), int(tgt.frame_height.iloc[0])
    t_ref = {k: g[["world_x", "world_y"]].to_numpy() for k, g in ref.groupby("frame_number")}
    t_tgt = {k: ground_points(g, a.target, a.overhead) for k, g in tgt.groupby("frame_number")}
    steps = sorted(set(t_ref) & set(t_tgt))
    steps = [s for s in steps if len(t_ref[s]) >= 3 and len(t_tgt[s]) >= 3]
    eval_steps = steps[::3]
    print(f"{len(steps)} common seconds; scoring on {len(eval_steps)}")
    known = []
    for k in a.known:
        img, wd = k.split("=")
        u, v = map(float, img.split(","))
        known.append(((u * W, v * Hh), tuple(map(float, wd.split(",")))))

    lines = []
    for k in a.on_line:
        img, cond = k.split("=")
        u, v = map(float, img.split(","))
        ax, val = cond.split(":")
        lines.append(((u * W, v * Hh), 0 if ax.upper() == "X" else 1, float(val)))

    # flatten evaluation set
    ev_t = np.concatenate([[s] * len(t_tgt[s]) for s in eval_steps])
    ev_p = np.concatenate([t_tgt[s] for s in eval_steps])
    ref_pad = max(len(t_ref[s]) for s in eval_steps)
    R = np.full((len(eval_steps), ref_pad, 2), 1e6)
    for i, s in enumerate(eval_steps):
        R[i, : len(t_ref[s])] = t_ref[s]
    idx = {s: i for i, s in enumerate(eval_steps)}
    ev_i = np.array([idx[s] for s in ev_t])

    def score(H, k1, return_mask=False):
        wpt = apply_h(H, undistort(ev_p, k1, W, Hh))
        inside = (wpt[:, 0] > -1.5) & (wpt[:, 0] < 13.5) & (wpt[:, 1] > -1.5) & (wpt[:, 1] < 31.5)
        D = np.linalg.norm(R[ev_i] - wpt[:, None, :], axis=2)
        nn = np.argmin(D, axis=1)
        dist = D[np.arange(len(D)), nn]
        inl = (dist < a.tol) & inside
        # one-to-one: a reference cow can explain only ONE target box per second
        key = ev_i * 1000 + nn
        order = np.argsort(dist)
        seen, uniq = set(), np.zeros(len(inl), bool)
        for j in order:
            if inl[j] and key[j] not in seen:
                seen.add(key[j]); uniq[j] = True
        s = uniq.sum() - 0.5 * (~inside).sum() - 0.5 * (inl & ~uniq).sum()
        return (s, uniq, dist) if return_mask else s

    need = 4 - len(known)
    best = (-1e9, None, None)
    for k1 in a.k1:
        kp = undistort(np.array([p for p, _ in known]).reshape(-1, 2), k1, W, Hh) if known else np.zeros((0, 2))
        kw = np.array([w for _, w in known]).reshape(-1, 2)
        for _ in range(a.hyp):
            s = steps[rng.integers(len(steps))]
            P, Q = t_tgt[s], t_ref[s]
            if len(P) < need or len(Q) < need:
                continue
            pi = np.sort(rng.choice(len(P), need, replace=False))
            qi = rng.choice(len(Q), need, replace=False)
            src = np.r_[kp, undistort(P[pi], k1, W, Hh)]
            dst = np.r_[kw, Q[qi]]
            try:
                H = fit_h(src, dst)
            except Exception:
                continue
            if H is None or not np.all(np.isfinite(H)):
                continue
            sc = score(H, k1)
            if sc > best[0]:
                best = (sc, H, k1)
        print(f"k1 {k1:+.2f}: best score so far {best[0]:.0f} (k1 {best[2]})")

    sc, H, k1 = best
    # ---- refine: iterate nearest-neighbour matching + robust least squares on (H, k1)
    for it in range(5):
        _, inl, _ = score(H, k1, return_mask=True)
        wpt = apply_h(H, undistort(ev_p, k1, W, Hh))
        nn = np.argmin(np.linalg.norm(R[ev_i] - wpt[:, None, :], axis=2), axis=1)
        tgt_w = R[ev_i, nn]
        P, Q = ev_p[inl], tgt_w[inl]
        if len(known):
            P = np.r_[P, np.repeat(np.array([p for p, _ in known]), 20, 0)]
            Q = np.r_[Q, np.repeat(np.array([w for _, w in known]), 20, 0)]

        def res(x):
            Hx = np.r_[x[:8], 1.0].reshape(3, 3)
            kk = k1 if a.fix_k1 else x[8]
            r = (apply_h(Hx, undistort(P, kk, W, Hh)) - Q).ravel()
            if lines:
                lp = apply_h(Hx, undistort(np.array([p for p, _, _ in lines]), kk, W, Hh))
                lr = np.array([lp[i, ax] - val for i, (_, ax, val) in enumerate(lines)])
                r = np.r_[r, np.repeat(lr, 30)]
            return r
        x0 = np.r_[(H / H[2, 2]).ravel()[:8]] if a.fix_k1 else np.r_[(H / H[2, 2]).ravel()[:8], k1]
        lo = np.r_[np.full(8, -np.inf)] if a.fix_k1 else np.r_[np.full(8, -np.inf), -0.6]
        hi = np.r_[np.full(8, np.inf)] if a.fix_k1 else np.r_[np.full(8, np.inf), 0.2]
        x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)
        r = least_squares(res, x0, loss="soft_l1", f_scale=1.0, bounds=(lo, hi))
        H = np.r_[r.x[:8], 1.0].reshape(3, 3)
        k1 = k1 if a.fix_k1 else float(r.x[8])
    sc, inl, dist = score(H, k1, return_mask=True)
    print(f"\nfinal k1 {k1:.3f}; matched {inl.sum()}/{len(inl)} target boxes "
          f"({inl.mean():.0%}); median error of matched {np.median(dist[inl]):.2f} m")
    for (p, w) in known:
        got = apply_h(H, undistort(np.array([p]), k1, W, Hh))[0]
        print(f"known point {w} -> fitted {got.round(2)}")
    for (p, ax, val) in lines:
        got = apply_h(H, undistort(np.array([p]), k1, W, Hh))[0]
        print(f"line point {'XY'[ax]}={val} -> fitted {got.round(2)}")

    # express as 4 world corners in undistorted reference pixels
    Hinv = np.linalg.inv(H)
    corners = [(0, 0), (12, 0), (12, 30), (0, 30)]
    pts = apply_h(Hinv, np.array(corners, float))
    out = {"reference_resolution": [W, Hh], "distortion_k1": round(k1, 4),
           "calibration_points": [{"image_undistorted_px": [round(float(x), 1), round(float(y), 1)],
                                   "world": [float(c[0]), float(c[1])]} for (x, y), c in zip(pts, corners)],
           "fit": {"matched_share": round(float(inl.mean()), 3), "median_err_m": round(float(np.median(dist[inl])), 2),
                   "reference": a.reference, "records": a.records}}
    path = f"outputs/calibration/{a.target}_fit.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    print("->", path)


if __name__ == "__main__":
    main()
