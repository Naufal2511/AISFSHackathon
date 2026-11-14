"""Multi-camera cue fusion -> 10-class emissions -> sticky HMM smoothing per cow.

Pipeline per cow:
  rows (cow, camera, t)  --weighted by detection quality x camera expertise-->  fused cues per second
  fused cues  --factorised behavior model-->  P(class | cues) per second
  per-second probabilities  --HMM (bout-duration priors), forward-backward + Viterbi-->  labels + posteriors
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .taxonomy import BEHAVIORS, IDX, NOT_VISIBLE

CUE_TO_WEIGHT = {
    "p_lying": "posture",
    "p_moving": "motion",
    "in_feed": "feed",
    "in_drink": "drink",
}


def fuse_cameras(rows: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """Collapses all camera observations of a cow within one second into one cue vector."""
    rows = rows[rows["global_cow_id"].notna()].copy()
    cw = cfg["camera_weights"]
    for cue, key in CUE_TO_WEIGHT.items():
        w = rows["camera_id"].map(lambda c: cw.get(c, {}).get(key, 1.0))
        src = {"in_feed": "zone_src_feed", "in_drink": "zone_src_drink"}.get(cue)
        if src and src in rows:  # hand-drawn image zones are trusted on every camera
            w = w.where(rows[src] != "drawn", 1.0)
        w = w * rows["quality"]
        rows[f"_w_{cue}"] = w.where(rows[cue].notna(), 0.0)
        rows[f"_wx_{cue}"] = rows[f"_w_{cue}"] * rows[cue].fillna(0)

    agg = {f"_w_{c}": "sum" for c in CUE_TO_WEIGHT}
    agg.update({f"_wx_{c}": "sum" for c in CUE_TO_WEIGHT})
    agg.update(
        p_groom_cue="max",
        p_social_cue="max",
        quality="max",
        camera_id=lambda s: ",".join(sorted(set(s))),
        world_x="mean",
        world_y="mean",
        speed_mps="median",
        drink_dist="min",
        nn_dist="min",
        height_ratio="median",
        wall_time="first",
    )
    if "p_ruminating" in rows.columns:
        agg["p_ruminating"] = "max"
    if "source_video" in rows.columns:
        agg["source_video"] = "first"
    g = rows.groupby(["global_cow_id", "sec"]).agg(agg)
    for c in CUE_TO_WEIGHT:
        w = g[f"_w_{c}"]
        g[c] = (g[f"_wx_{c}"] / w).where(w > 0)
    g = g.drop(columns=[c for c in g.columns if c.startswith("_w")])
    g = g.rename(columns={"camera_id": "cameras"})
    g["n_cameras"] = g["cameras"].str.count(",") + 1
    return g.reset_index()


def behavior_probabilities(f: pd.DataFrame) -> np.ndarray:
    """Factorised behavior model. Returns (N, 10) probabilities.

    L lying, M moving, F feed zone, D drink zone, S social, G grooming, R rumination.
    Rumination R defaults to 0: it is only claimed when chewing was measured or a learned
    model provides it (see rumination.py / appearance.py).
    """
    L = f["p_lying"].fillna(0.3).to_numpy()
    M = f["p_moving"].fillna(0.1).to_numpy()
    F = f["in_feed"].fillna(0).to_numpy()
    D = f["in_drink"].fillna(0).to_numpy()
    S = f["p_social_cue"].fillna(0).to_numpy()
    G = f["p_groom_cue"].fillna(0).to_numpy()
    R = f["p_ruminating"].fillna(0).to_numpy() if "p_ruminating" in f else np.zeros(len(f))
    q = f["quality"].fillna(0.3).to_numpy()
    app_cols = [f"app::{b}" for b in BEHAVIORS]
    has_app = all(c in f.columns for c in app_cols)
    if has_app:  # learned expert can raise the cues geometry cannot measure
        Papp = f[app_cols].to_numpy(dtype=float)
        app_ok = np.isfinite(Papp).all(1)
        rum = Papp[:, IDX["ruminating (lying)"]] + Papp[:, IDX["ruminating (standing)"]]
        R = np.where(app_ok, np.maximum(R, rum), R)
        G = np.where(app_ok, np.maximum(G, Papp[:, IDX["grooming"]]), G)

    up = 1 - L
    P = np.zeros((len(f), len(BEHAVIORS)))
    P[:, IDX["social interaction"]] = up * S
    rest = up * (1 - S)
    P[:, IDX["walking"]] = rest * M
    still = rest * (1 - M)
    P[:, IDX["feeding"]] = still * F
    P[:, IDX["drinking"]] = still * (1 - F) * D
    free = still * (1 - F) * (1 - D)
    P[:, IDX["grooming"]] = free * G
    P[:, IDX["ruminating (standing)"]] = free * (1 - G) * R
    P[:, IDX["standing"]] = free * (1 - G) * (1 - R)
    P[:, IDX["ruminating (lying)"]] = L * R
    P[:, IDX["lying"]] = L * (1 - R)
    # Poor detections leak mass into "other / not visible"
    P[:, IDX[NOT_VISIBLE]] = 0.02 + 0.3 * (1 - q)
    P = P / P.sum(1, keepdims=True)
    if has_app:  # product of experts (geometry x appearance), floor keeps either from vetoing
        P = np.where(app_ok[:, None], (P + 0.02) * (np.nan_to_num(Papp) + 0.05), P)
        P = P / P.sum(1, keepdims=True)
    return P


def transition_matrix(cfg: Dict, dt: float) -> np.ndarray:
    taus = cfg["hmm"]["mean_bout_sec"]
    K = len(BEHAVIORS)
    A = np.zeros((K, K))
    for i, b in enumerate(BEHAVIORS):
        stay = float(np.exp(-dt / float(taus.get(b, 120))))
        A[i, :] = (1 - stay) / (K - 1)
        A[i, i] = stay
    # Physically implausible direct jumps (lying <-> walking needs a stand-up) are damped.
    for a, b in (("lying", "walking"), ("ruminating (lying)", "walking"), ("lying", "feeding"), ("lying", "drinking")):
        for x, y in ((a, b), (b, a)):
            A[IDX[x], IDX[y]] *= 0.1
    return A / A.sum(1, keepdims=True)


def _forward_backward(E: np.ndarray, A: np.ndarray, pi: np.ndarray):
    T, K = E.shape
    alpha = np.zeros((T, K)); c = np.zeros(T)
    a = pi * E[0]; c[0] = a.sum(); alpha[0] = a / c[0]
    for t in range(1, T):
        a = (alpha[t - 1] @ A) * E[t]
        c[t] = a.sum() + 1e-300
        alpha[t] = a / c[t]
    beta = np.ones((T, K))
    for t in range(T - 2, -1, -1):
        b = A @ (E[t + 1] * beta[t + 1])
        beta[t] = b / c[t + 1]
    post = alpha * beta
    return post / post.sum(1, keepdims=True)


def _viterbi(E: np.ndarray, A: np.ndarray, pi: np.ndarray) -> np.ndarray:
    T, K = E.shape
    logA = np.log(A + 1e-300); logE = np.log(E + 1e-300)
    delta = np.log(pi + 1e-300) + logE[0]
    back = np.zeros((T, K), dtype=np.int16)
    for t in range(1, T):
        s = delta[:, None] + logA
        back[t] = s.argmax(0)
        delta = s.max(0) + logE[t]
    path = np.zeros(T, dtype=np.int64)
    path[-1] = delta.argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def smooth_cow(fused_cow: pd.DataFrame, cfg: Dict, session_gap_sec: int = 300, temperature: float = 0.6) -> pd.DataFrame:
    """Builds a dense 1-s timeline for one cow and decodes it with the HMM."""
    tl = cfg["timeline"]
    step = int(tl["step_sec"])
    fused_cow = fused_cow.sort_values("sec")
    secs = fused_cow["sec"].to_numpy()
    probs = behavior_probabilities(fused_cow) ** temperature  # correlated frames -> temper evidence
    A = transition_matrix(cfg, step)
    K = len(BEHAVIORS)

    # Split into sessions (separate video clips); unrecorded time between them is "no_video".
    breaks = np.where(np.diff(secs) > session_gap_sec)[0]
    out = []
    for seg in np.split(np.arange(len(secs)), breaks + 1):
        s0, s1 = secs[seg[0]], secs[seg[-1]]
        grid = np.arange(s0, s1 + 1, step)
        E = np.ones((len(grid), K))
        pos = ((secs[seg] - s0) // step).astype(int)
        E[pos] = probs[seg]
        observed = np.zeros(len(grid), bool); observed[pos] = True
        # long unseen runs -> evidence for "not visible"; short ones are bridged (uniform emission)
        run_start = None
        for i in range(len(grid) + 1):
            if i < len(grid) and not observed[i]:
                run_start = i if run_start is None else run_start
            elif run_start is not None:
                if (i - run_start) * step > tl["gap_fill_sec"]:
                    E[run_start:i] = 0.02
                    E[run_start:i, IDX[NOT_VISIBLE]] = 1.0
                run_start = None
        pi = np.full(K, 1.0 / K)
        post = _forward_backward(E, A, pi)
        path = _viterbi(E, A, pi)
        o = pd.DataFrame({"sec": grid, "behavior": [BEHAVIORS[k] for k in path]})
        o["confidence"] = post[np.arange(len(grid)), path]
        o["observed"] = observed
        for k, b in enumerate(BEHAVIORS):
            o[f"p::{b}"] = post[:, k]
        out.append(o)
    res = pd.concat(out, ignore_index=True)
    keep = [c for c in ("cameras", "n_cameras", "world_x", "world_y", "speed_mps", "height_ratio",
                        "nn_dist", "drink_dist", "source_video", "p_ruminating") if c in fused_cow.columns]
    return res.merge(fused_cow[["sec"] + keep], on="sec", how="left")


# Classes whose evidence is a direct geometric measurement vs. a weak heuristic.
DIRECT = {"lying", "standing", "walking", "feeding", "drinking"}


def evidence_label(row) -> str:
    """Separates what the system OBSERVED from what it ESTIMATED (required by the brief)."""
    b = row["behavior"]
    if b == NOT_VISIBLE:
        return "not_visible"
    if not row["observed"]:
        return "estimated_gap_fill"
    if b in DIRECT:
        return "observed"
    if b.startswith("ruminating"):
        pr = row.get("p_ruminating")
        return "observed_chewing" if pd.notna(pr) and pr >= 0.5 else "estimated"
    return "estimated_heuristic"  # grooming / social interaction until the verifier confirms
