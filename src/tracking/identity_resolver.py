"""Identity by agreement: learn who is who from all cameras before labelling behavior.

The online tracker keeps boxes consistent WITHIN one camera (local tracks). Who is who ACROSS
cameras and across track breaks is decided here, over a learning window, by constraints that
do not depend on black cattle looking different:

  agreement (must-link)   two tracks in different cameras that stand at the same pen
                          position at the same seconds (>= `min_overlap` s, >= `min_agree` of
                          them within tolerance), and are each other's best partner
  continuity (link)       a track that ends and a track that starts in the same camera within
                          `max_gap` s at a reachable distance
  exclusion (cannot-link) two tracks in the same camera at the same time; two tracks in
                          different cameras that are co-visible but far apart
  appearance (tie-break)  DINOv2 + coat colour per camera, only between clusters seen by the
                          same camera (front / bunk / back views are not compared)

Clusters are grown greedily (strongest evidence first) and never merged across a cannot-link.
The pen holds `pen_size` animals: the largest clusters become COW_01..COW_N; leftovers are
attached to a compatible identity when the evidence allows it, otherwise left unassigned
(never forced onto a cow).

Chute rule: an identity last seen next to the chute fence that disappears is 'out'; a new
cluster that first appears next to the chute later is matched to the out identity (if one is
pending), so leaving and re-entering keeps the ID.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

DEFAULTS = dict(
    pen_size=10,
    min_track_obs=3,
    min_overlap=5,          # seconds co-visible before cameras may agree
    min_agree=0.6,          # share of those seconds within tolerance
    tol_m={"cam1-cam2": 2.0, "cam1-cam3": 2.8, "cam2-cam3": 2.8},
    far_m=5.0,              # co-visible and this far apart -> different animals
    max_gap=15.0,           # seconds for a same-camera continuation
    reach_mps=1.5,          # how far a steer can get during the gap (+1 m slack)
    app_merge_min=0.80,     # appearance-only merges need this similarity in a shared camera
    attach_min=0.70,
    chute_x_max=1.5,        # metres from the chute fence (X=0)
    ground_center_cameras=("cam2",),
)


@dataclass
class Tracklet:
    key: Tuple[str, int]
    secs: np.ndarray
    xy: np.ndarray            # world (n, 2), NaN where unknown
    n: int
    emb: Optional[np.ndarray] = None

    @property
    def cam(self):
        return self.key[0]

    @property
    def t0(self):
        return float(self.secs[0])

    @property
    def t1(self):
        return float(self.secs[-1])


@dataclass
class Cluster:
    members: List[Tuple[str, int]] = field(default_factory=list)
    occ: Dict[str, set] = field(default_factory=lambda: defaultdict(set))   # cam -> seconds
    n: int = 0


def world_positions(rec: pd.DataFrame, transformer, center_cams=("cam2",)) -> pd.DataFrame:
    rec = rec.copy()
    cx = (rec.bbox_x1 + rec.bbox_x2) / 2
    gy = np.where(rec.camera_id.isin(center_cams), (rec.bbox_y1 + rec.bbox_y2) / 2, rec.bbox_y2)
    wx, wy = [], []
    for cam, x, y, w, h in zip(rec.camera_id, cx, gy, rec.frame_width, rec.frame_height):
        p = transformer.pixel_to_world(cam, float(x), float(y), current_res=(int(w), int(h)))
        wx.append(np.nan if p is None else p[0])
        wy.append(np.nan if p is None else p[1])
    rec["gx"], rec["gy"] = wx, wy
    return rec


class IdentityResolver:
    def __init__(self, cfg: Optional[Dict] = None):
        self.c = {**DEFAULTS, **(cfg or {})}
        self.report: Dict = {}

    # ------------------------------------------------------------------ building blocks
    def tracklets(self, rec: pd.DataFrame) -> Dict[Tuple[str, int], Tracklet]:
        rec = rec.assign(sec=np.round(rec.t).astype(np.int64))
        out = {}
        for key, g in rec.groupby(["camera_id", "local_track_id"]):
            g = g.sort_values("sec").drop_duplicates("sec")
            if len(g) < self.c["min_track_obs"]:
                continue
            out[(key[0], int(key[1]))] = Tracklet((key[0], int(key[1])), g.sec.to_numpy(),
                                                  g[["gx", "gy"]].to_numpy(float), len(g))
        return out

    def _tol(self, a: str, b: str) -> float:
        return self.c["tol_m"].get("-".join(sorted((a, b))), 2.5)

    def pair_stats(self, A: Tracklet, B: Tracklet):
        common, ia, ib = np.intersect1d(A.secs, B.secs, return_indices=True)
        if len(common) == 0:
            return 0, None, None
        d = np.linalg.norm(A.xy[ia] - B.xy[ib], axis=1)
        d = d[np.isfinite(d)]
        if len(d) == 0:
            return len(common), None, None
        return len(common), float(np.mean(d < self._tol(A.cam, B.cam))), float(np.median(d))

    # ------------------------------------------------------------------ resolve
    def resolve(self, rec: pd.DataFrame, embed_fn=None) -> Dict[Tuple[str, int], Optional[str]]:
        c = self.c
        T = self.tracklets(rec)
        keys = list(T)
        if embed_fn is not None:
            for k in keys:
                T[k].emb = embed_fn(k)

        # --- agreement (cross-camera) and exclusion
        cannot = set()
        agree_edges = []
        best = defaultdict(lambda: (0.0, None))  # (track, other cam) -> (strength, partner)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                A, B = T[a], T[b]
                if A.t1 < B.t0 or B.t1 < A.t0:
                    continue
                if A.cam == B.cam:
                    if len(np.intersect1d(A.secs, B.secs)) > 1:
                        cannot.add((a, b)); cannot.add((b, a))
                    continue
                n, agree, med = self.pair_stats(A, B)
                if agree is None:
                    continue
                if n >= c["min_overlap"] and med > c["far_m"] and agree < 0.2:
                    cannot.add((a, b)); cannot.add((b, a))
                if n >= c["min_overlap"] and agree >= c["min_agree"]:
                    s = agree * n
                    agree_edges.append((s, a, b, agree, n, med))
                    if s > best[(a, B.cam)][0]:
                        best[(a, B.cam)] = (s, b)
                    if s > best[(b, A.cam)][0]:
                        best[(b, A.cam)] = (s, a)
        mutual = [(s, a, b) for s, a, b, *_ in agree_edges
                  if best[(a, T[b].cam)][1] == b and best[(b, T[a].cam)][1] == a]

        # --- continuity (same camera, short gap, reachable)
        cont = []
        for cam in {k[0] for k in keys}:
            ks = sorted([k for k in keys if k[0] == cam], key=lambda k: T[k].t0)
            for a in ks:
                for b in ks:
                    A, B = T[a], T[b]
                    gap = B.t0 - A.t1
                    if a == b or not (0 < gap <= c["max_gap"]):
                        continue
                    pa, pb = A.xy[-1], B.xy[0]
                    dist = float(np.linalg.norm(pa - pb)) if np.all(np.isfinite([*pa, *pb])) else np.nan
                    if np.isfinite(dist) and dist > c["reach_mps"] * gap + 1.0:
                        continue
                    sim = float(A.emb @ B.emb) if A.emb is not None and B.emb is not None else 0.5
                    score = 1.0 / (1 + gap) + (0 if not np.isfinite(dist) else 1.0 / (1 + dist)) + sim
                    cont.append((score, a, b))

        # --- greedy constrained union
        cl: Dict[int, Cluster] = {}
        of: Dict[Tuple[str, int], int] = {}
        for i, k in enumerate(keys):
            cl[i] = Cluster([k], defaultdict(set, {k[0]: set(T[k].secs.tolist())}), T[k].n)
            of[k] = i

        def conflict(x: int, y: int) -> bool:
            X, Y = cl[x], cl[y]
            for cam, secs in X.occ.items():
                if len(secs & Y.occ.get(cam, set())) > 1:
                    return True
            return any((m, n) in cannot for m in X.members for n in Y.members)

        def merge(x: int, y: int):
            X, Y = cl[x], cl.pop(y)
            X.members += Y.members
            for cam, secs in Y.occ.items():
                X.occ[cam] |= secs
            X.n += Y.n
            for m in Y.members:
                of[m] = x

        stats = {"agreement_merges": 0, "continuity_merges": 0, "appearance_merges": 0, "blocked": 0}
        for kind, edges in (("agreement_merges", sorted(mutual, reverse=True)),
                            ("continuity_merges", sorted(cont, reverse=True))):
            for _, a, b in edges:
                x, y = of[a], of[b]
                if x == y:
                    continue
                if conflict(x, y):
                    stats["blocked"] += 1
                    continue
                merge(x, y)
                stats[kind] += 1

        # --- appearance merges (same camera only), until the pen count is reached
        def cam_proto(x, cam):
            e = [T[m].emb * T[m].n for m in cl[x].members if m[0] == cam and T[m].emb is not None]
            if not e:
                return None
            v = np.sum(e, 0)
            return v / (np.linalg.norm(v) + 1e-9)

        def app_sim(x, y):
            sims = []
            for cam in set(cl[x].occ) & set(cl[y].occ):
                p, q = cam_proto(x, cam), cam_proto(y, cam)
                if p is not None and q is not None:
                    sims.append(float(p @ q))
            return max(sims) if sims else None

        big = lambda: sorted(cl, key=lambda x: -cl[x].n)
        while len([x for x in cl if cl[x].n >= 30]) > c["pen_size"]:
            cand = []
            ids = big()
            for i, x in enumerate(ids):
                for y in ids[i + 1:]:
                    s = app_sim(x, y)
                    if s is not None and s >= c["app_merge_min"] and not conflict(x, y):
                        cand.append((s, x, y))
            if not cand:
                break
            s, x, y = max(cand)
            merge(x, y)
            stats["appearance_merges"] += 1

        # --- identities = the largest clusters; attach leftovers when compatible
        order = big()
        core = order[: c["pen_size"]]
        names = {x: f"COW_{i + 1:02d}" for i, x in enumerate(sorted(core, key=lambda x: min(T[m].t0 for m in cl[x].members)))}
        attached = 0
        for x in order[c["pen_size"]:]:
            if x not in cl:
                continue
            opts = []
            for y in core:
                if conflict(x, y):
                    continue
                s = app_sim(x, y)
                # co-location with a core member in another camera is strong evidence
                co = 0.0
                for m in cl[x].members:
                    for n in cl[y].members:
                        if m[0] != n[0]:
                            k, agree, _ = self.pair_stats(T[m], T[n])
                            if agree is not None and k >= 3:
                                co = max(co, agree)
                score = max(co, s or 0.0)
                if score >= c["attach_min"]:
                    opts.append((score, y))
            if opts:
                opts.sort(reverse=True)
                if len(opts) == 1 or opts[0][0] - opts[1][0] > 0.05:  # unambiguous only
                    merge(opts[0][1], x)
                    attached += 1

        # --- chute rule: out-and-back-in keeps the identity
        chute = self._chute_events(cl, T, core, names)

        mapping = {k: None for k in keys}
        for x in core:
            if x in cl:
                for m in cl[x].members:
                    mapping[m] = names[x]
        n_boxes = sum(T[k].n for k in keys)
        n_assigned = sum(T[k].n for k in keys if mapping[k])
        self.report = {**stats, "tracklets": len(keys), "mutual_agreements": len(mutual),
                       "cannot_links": len(cannot) // 2, "continuity_candidates": len(cont),
                       "attached_leftovers": attached, "assigned_box_share": round(n_assigned / max(n_boxes, 1), 3),
                       "identities": {names[x]: {"cameras": sorted(cl[x].occ), "tracklets": len(cl[x].members),
                                                 "seconds": int(len(set().union(*cl[x].occ.values())))}
                                      for x in core if x in cl},
                       "chute_events": chute}
        self.clusters, self.names, self.T = cl, names, T
        return mapping

    def _chute_events(self, cl, T, core, names):
        ev = []
        span_end = max(t.t1 for t in T.values())
        for x in core:
            if x not in cl:
                continue
            last = max((T[m] for m in cl[x].members), key=lambda t: t.t1)
            p = last.xy[-1]
            if np.isfinite(p[0]) and p[0] <= self.c["chute_x_max"] and span_end - last.t1 > 60:
                ev.append({"cow_id": names[x], "event": "left via chute", "time": str(pd.Timestamp(last.t1, unit="s"))})
        return ev


def cross_camera_consistency(rec: pd.DataFrame, id_col: str = "global_cow_id") -> Dict:
    """For each second where an ID is seen by >= 2 cameras: distance between its positions."""
    r = rec[rec[id_col].notna() & rec.gx.notna()].assign(sec=np.round(rec.t).astype(np.int64))
    g = r.groupby([id_col, "sec", "camera_id"])[["gx", "gy"]].mean().reset_index()
    ds = []
    for (_, _), h in g.groupby([id_col, "sec"]):
        if len(h) >= 2:
            P = h[["gx", "gy"]].to_numpy()
            ds.append(np.max(np.linalg.norm(P[:, None] - P[None], axis=2)))
    ds = np.array(ds)
    if not len(ds):
        return {"pairs": 0}
    return {"multi_camera_seconds": int(len(ds)), "median_spread_m": round(float(np.median(ds)), 2),
            "share_within_3m": round(float(np.mean(ds < 3.0)), 3)}
