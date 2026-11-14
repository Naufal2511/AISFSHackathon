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
    far_m=5.0,              # (legacy fixed threshold; see far_margin_m)
    base_tol_m=1.5,         # cameras may disagree by base + range_coef * distance-to-camera ...
    range_coef={"cam1": 0.03, "cam2": 0.03, "cam3": 0.09},
    trusted_range_m={"cam1": 18.0, "cam2": 10.0, "cam3": 20.0},  # a camera DEFINES identities only near
                            # itself; far tracks may only join an identity (unambiguous agreement)
    far_range_m=15.0,       # beyond this distance from a camera its error grows faster:
    far_range_coef={"cam1": 0.10, "cam2": 0.06, "cam3": 0.12},   # extra metres per metre beyond far_range_m
    far_margin_m=3.0,       # ... and are only "different animals" beyond tolerance + this margin
    cam_pos={},             # camera mount positions (pen metres); enables distance-aware tolerance
    fragment_contained=0.7, # a box this much inside a bigger box is a piece of that cow
    small_contained=1.01,   # (off: it marked cows lying partly behind others as fragments)
    small_area_ratio=0.3,
    fragment_share=0.6,     # ... for this share of the track's life -> fragment of the host
    tiny_area_frac=0.2,     # median box below this fraction of the camera's median box -> sliver
    part_area_frac=0.0,     # touching-part rule OFF (it absorbed whole cows standing behind others);
                            # unsupported pieces like a head box simply stay unassigned
    part_max_obs=150,       # ... on a short track -> part of that neighbour
    split_jump_frac=0.08,       # candidate switch: the box jumps this share of the frame ...
    split_shape_change=0.9,     # ... AND its shape changes a lot (|log h|+|log w|), e.g. lying -> standing
    split_big_jump_frac=0.15,   # ... or jumps this far on its own
    split_jump_mps=4.0,         # ... or the ground position moves faster than a steer can
    split_confirm_s=5,          # confirmed only if the 5 s before and after differ ...
    split_confirm_m=1.5,        # ... by this many metres (median ground position)
    split_confirm_h=2.0,        # ... or by this factor in box height (lying vs standing)
    conflict_min_s=5,           # two boxes in one camera prove two animals only if seen together > 5 s
    sequential_gap_s=30,        # identity A ends, B starts within this gap (overlap <= conflict_min_s) ...
    walk_mps=1.6,               # ... and a steer can walk between them -> the same animal
    animal_min_s_two_cams=30,   # to COUNT as an animal: >= 30 s in each of two cameras ...
    animal_min_s_one_cam=150,   # ... or >= 150 s in one camera
    max_gap=15.0,           # seconds for a same-camera continuation
    reach_mps=1.5,          # how far a steer can get during the gap (+1 m slack)
    app_merge_min=1.01,     # appearance-only merges OFF: black cows look alike and a wrong merge inflates the head count
    attach_min=0.85,
    min_identity_obs=30,    # a cluster needs this many observations to count as an animal
    forced_merges=False,    # never push a cluster onto an animal just because it is the only one left
    attach_needs_colocation=True,  # leftovers join a cow only on positive position agreement ...
    attach_min_overlap=10,  # ... over at least this many shared seconds
    attach_min_obs=8,       # pieces shorter than this stay unassigned (often a head / limb box)
    chute_x_max=1.5,        # metres from the chute fence (X=0)
    ground_center_cameras=("cam2",),
)


@dataclass
class Tracklet:
    key: Tuple[str, int]
    secs: np.ndarray
    xy: np.ndarray            # world (n, 2), NaN where unknown
    n: int
    boxes: Optional[np.ndarray] = None   # (n, 4) pixel boxes
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
    def _split_points(self, g: pd.DataFrame) -> np.ndarray:
        """Segment index per row: the tracker sometimes slides from one cow onto a neighbour
        (box jumps and changes shape, e.g. a lying cow's track continues on a standing one)."""
        c = self.c
        if len(g) < 2:
            return np.zeros(len(g), int)
        W, H = g.frame_width.to_numpy(float), g.frame_height.to_numpy(float)
        cx, cy = ((g.bbox_x1 + g.bbox_x2) / 2).to_numpy() / W, ((g.bbox_y1 + g.bbox_y2) / 2).to_numpy() / H
        w = ((g.bbox_x2 - g.bbox_x1).to_numpy() / W).clip(1e-4)
        h = ((g.bbox_y2 - g.bbox_y1).to_numpy() / H).clip(1e-4)
        jump = np.hypot(np.diff(cx), np.diff(cy))
        shape = np.abs(np.diff(np.log(h))) + np.abs(np.diff(np.log(w)))
        dt = np.maximum(np.diff(g.sec.to_numpy()), 1)
        gx, gy = g.gx.to_numpy(float), g.gy.to_numpy(float)
        speed = np.hypot(np.diff(gx), np.diff(gy)) / dt
        cand = ((jump > c["split_jump_frac"]) & (shape > c["split_shape_change"])) | (jump > c["split_big_jump_frac"]) \
            | (np.nan_to_num(speed) > c["split_jump_mps"])
        n, k = len(g), c["split_confirm_s"]
        cut = np.zeros(n - 1, bool)
        for i in np.flatnonzero(cand):
            a, b = slice(max(0, i - k + 1), i + 1), slice(i + 1, min(n, i + 1 + k))
            if (b.stop - b.start) < 3 or (a.stop - a.start) < 3:
                continue
            pa = np.nanmedian(np.c_[gx[a], gy[a]], 0)
            pb = np.nanmedian(np.c_[gx[b], gy[b]], 0)
            moved = np.all(np.isfinite([*pa, *pb])) and np.hypot(*(pa - pb)) > c["split_confirm_m"]
            hr = np.median(h[b]) / np.median(h[a])
            if moved or hr > c["split_confirm_h"] or hr < 1 / c["split_confirm_h"]:
                cut[i] = True
        return np.r_[0, np.cumsum(cut)]

    def tracklets(self, rec: pd.DataFrame) -> Dict[Tuple[str, int], Tracklet]:
        rec = rec.assign(sec=np.round(rec.t).astype(np.int64))
        out = {}
        row_key = pd.Series(index=rec.index, dtype=object)
        self.splits = 0
        for key, g0 in rec.groupby(["camera_id", "local_track_id"]):
            g0 = g0.sort_values("sec")
            g1 = g0.drop_duplicates("sec")
            seg = pd.Series(self._split_points(g1), index=g1.index)
            seg_by_sec = dict(zip(g1.sec, seg))
            self.splits += int(seg.max()) if len(seg) else 0
            for si in sorted(set(seg)):
                k = (key[0], int(key[1]) + 10000 * int(si))   # piece si of tracker track key[1]
                rows = g0[g0.sec.map(seg_by_sec) == si]
                row_key.loc[rows.index] = [k] * len(rows)
                g = g1[seg == si]
                if len(g) < self.c["min_track_obs"]:
                    continue
                out[k] = Tracklet(k, g.sec.to_numpy(),
                                                  g[["gx", "gy"]].to_numpy(float), len(g),
                                                  g[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy(float))
        self.row_key = row_key
        return out

    def fragments(self, T) -> Dict[Tuple[str, int], Optional[Tuple[str, int]]]:
        """Tracks that are pieces of another cow (mostly inside its box) -> host; slivers -> None."""
        c = self.c
        by_cs = defaultdict(list)
        for k, t in T.items():
            for s, b in zip(t.secs, t.boxes):
                by_cs[(k[0], int(s))].append((k, b))
        cam_area = defaultdict(list)
        for k, t in T.items():
            cam_area[k[0]].extend(((t.boxes[:, 2] - t.boxes[:, 0]) * (t.boxes[:, 3] - t.boxes[:, 1])).tolist())
        med_area = {cam: float(np.median(v)) for cam, v in cam_area.items()}
        out = {}
        for k, t in T.items():
            areas = (t.boxes[:, 2] - t.boxes[:, 0]) * (t.boxes[:, 3] - t.boxes[:, 1])
            hosts = []
            for s, b, a in zip(t.secs, t.boxes, areas):
                best = None
                gap_px = 0.02 * max(b[2] - b[0], b[3] - b[1], 1) * 10
                for k2, b2 in by_cs[(k[0], int(s))]:
                    if k2 == k:
                        continue
                    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                    inter = max(0, min(b[2], b2[2]) - max(b[0], b2[0])) * max(0, min(b[3], b2[3]) - max(b[1], b2[1]))
                    if a2 > a and (inter / max(a, 1) >= c["fragment_contained"] or
                                   (a < c["small_area_ratio"] * a2 and inter / max(a, 1) >= c["small_contained"])):
                        best = k2
                        break
                    # a head / legs poking out of a much bigger neighbour's box (touching it)
                    touching = (b[0] <= b2[2] + gap_px and b2[0] <= b[2] + gap_px and
                                b[1] <= b2[3] + gap_px and b2[1] <= b[3] + gap_px)
                    if touching and a < c["part_area_frac"] * a2 and t.n <= c["part_max_obs"]:
                        best = k2
                hosts.append(best)
            inside = [h for h in hosts if h is not None]
            if len(inside) >= c["fragment_share"] * len(hosts):
                out[k] = max(set(inside), key=inside.count)
            elif np.median(areas) < c["tiny_area_frac"] * med_area[k[0]] and t.n < 120:
                out[k] = None
        return out

    def _tol(self, a: str, b: str) -> float:
        return self.c["tol_m"].get("-".join(sorted((a, b))), 2.5)

    def _tol_samples(self, A: Tracklet, B: Tracklet, pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
        """Distance-aware tolerance: far from a camera its metre estimate is less precise."""
        c = self.c
        if not c["cam_pos"]:
            return np.full(len(pa), self._tol(A.cam, B.cam))
        def rng(cam, p):
            cp = c["cam_pos"].get(cam)
            if not cp:
                return np.full(len(p), 10.0)
            return np.hypot(p[:, 0] - cp[0], p[:, 1] - cp[1])

        def err(cam, p):
            d = rng(cam, p)
            return (c["range_coef"].get(cam, 0.05) * d
                    + c["far_range_coef"].get(cam, 0.08) * np.clip(d - c["far_range_m"], 0, None))
        return c["base_tol_m"] + err(A.cam, pa) + err(B.cam, pb)

    def pair_stats(self, A: Tracklet, B: Tracklet, with_far: bool = False):
        common, ia, ib = np.intersect1d(A.secs, B.secs, return_indices=True)
        if len(common) == 0:
            return (0, None, None, None) if with_far else (0, None, None)
        pa, pb = A.xy[ia], B.xy[ib]
        ok = np.isfinite(pa).all(1) & np.isfinite(pb).all(1)
        if not ok.any():
            return (len(common), None, None, None) if with_far else (len(common), None, None)
        pa, pb = pa[ok], pb[ok]
        d = np.linalg.norm(pa - pb, axis=1)
        tol = self._tol_samples(A, B, pa, pb)
        agree, med = float(np.mean(d < tol)), float(np.median(d))
        if with_far:
            return len(common), agree, med, float(np.mean(d > tol + self.c["far_margin_m"]))
        return len(common), agree, med

    # ------------------------------------------------------------------ resolve
    def resolve(self, rec: pd.DataFrame, embed_fn=None) -> Dict[Tuple[str, int], Optional[str]]:
        c = self.c
        T_all = self.tracklets(rec)
        frag = self.fragments(T_all)
        T = {k: v for k, v in T_all.items() if k not in frag}
        keys = list(T)
        far_tracks = set()
        if c["cam_pos"]:
            for k, t in T.items():
                cp = c["cam_pos"].get(k[0])
                if cp and np.isfinite(t.xy).all(1).any():
                    d = np.nanmedian(np.hypot(t.xy[:, 0] - cp[0], t.xy[:, 1] - cp[1]))
                    if d > c["trusted_range_m"].get(k[0], 99):
                        far_tracks.add(k)
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
                    if len(np.intersect1d(A.secs, B.secs)) > c["conflict_min_s"]:  # proof of 2 animals
                        # (merges are still blocked by ANY overlap via the per-camera occupancy check)
                        cannot.add((a, b)); cannot.add((b, a))
                    continue
                n, agree, med, far = self.pair_stats(A, B, with_far=True)
                if agree is None:
                    continue
                if n >= c["min_overlap"] and far >= 0.8 and agree < 0.2:
                    cannot.add((a, b)); cannot.add((b, a))
                if n >= c["min_overlap"] and agree >= c["min_agree"]:
                    s = agree * n / (1.0 + med)   # among partners that all agree, the closest wins
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
                    if min(A.n, B.n) < c["attach_min_obs"]:   # a few-second piece cannot carry identity on
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

        def conflict(x: int, y: int, min_s: int = 1) -> bool:
            """Merging: any real overlap (> 1 s) in one camera forbids it (one ID = one box per frame).
            Counting animals uses a stricter proof (min_s = conflict_min_s)."""
            X, Y = cl[x], cl[y]
            for cam, secs in X.occ.items():
                if len(secs & Y.occ.get(cam, set())) > min_s:
                    return True
            return any((m, n) in cannot for m in X.members for n in Y.members)

        how: Dict[Tuple[str, int], str] = {}

        def merge(x: int, y: int, kind: str = ""):
            for m in cl[y].members + (cl[x].members if len(cl[x].members) == 1 else []):
                how.setdefault(m, kind)
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
                merge(x, y, kind)
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
            merge(x, y, "appearance_merges")
            stats["appearance_merges"] += 1

        # --- how many animals does the evidence REQUIRE? The largest set of clusters that are
        # pairwise provably different animals (seen together in one camera, or far apart at the
        # same moment). pen_size is only a cap: unused slots are never filled by splitting a cow.
        order = big()
        def near_occ(x):   # seconds from tracks within their camera's trusted range
            occ = defaultdict(set)
            for m in cl[x].members:
                if m not in far_tracks:
                    occ[m[0]] |= set(T[m].secs.tolist())
            return occ

        def supported(x):  # enough independent evidence that this is a whole animal
            secs = {cam: len(v) for cam, v in near_occ(x).items()}
            return (sum(v >= c["animal_min_s_two_cams"] for v in secs.values()) >= 2
                    or max(secs.values(), default=0) >= c["animal_min_s_one_cam"])
        bigs = [x for x in order if cl[x].n >= c["min_identity_obs"] and supported(x)]
        adj = {x: {y for y in bigs if y != x and conflict(x, y, c["conflict_min_s"])} for x in bigs}
        clique = max_clique(adj)
        # quality order: confirmed by >= 2 cameras first, then longest (a cam1-only fragment
        # must not take a slot from a real cow); pen_size = herd size from anchors when known
        def quality(x):
            secs = {cam: len(v) for cam, v in near_occ(x).items()}
            return (sum(v >= c["animal_min_s_two_cams"] for v in secs.values()) >= 2,
                    len(set().union(*near_occ(x).values())) if secs else 0)
        core = sorted(clique, key=quality, reverse=True)[: c["pen_size"]]
        names = {x: f"COW_{i + 1:02d}" for i, x in enumerate(sorted(core, key=lambda x: min(T[m].t0 for m in cl[x].members)))}
        attached, forced = 0, 0
        for x in [x for x in order if x not in core]:
            if x not in cl or cl[x].n < c["attach_min_obs"]:   # a few-second piece must not take a cow's ID
                continue
            compatible = [y for y in core if not conflict(x, y)]

            def colocation(y):
                best = None
                for m in cl[x].members:
                    for n in cl[y].members:
                        if m[0] != n[0]:
                            k, agree, _ = self.pair_stats(T[m], T[n])
                            if agree is not None and k >= 3:
                                best = agree if best is None else max(best, agree)
                return best
            if c.get("forced_merges", True) and len(compatible) == 1 and cl[x].n >= c["min_identity_obs"]:
                co = colocation(compatible[0])
                if (co is None and not c.get("forced_needs_colocation")) or (co is not None and co >= 0.5):
                    merge(compatible[0], x, "attached")
                    forced += 1
                    continue
            opts = []
            for y in compatible:
                s = app_sim(x, y)
                # co-location with a core member in another camera is strong evidence
                co, dist = 0.0, np.inf
                for m in cl[x].members:
                    for n in cl[y].members:
                        if m[0] != n[0]:
                            k, agree, med = self.pair_stats(T[m], T[n])
                            if agree is not None and k >= c.get("attach_min_overlap", 3) and \
                                    (agree > co or (agree == co and med < dist)):
                                co, dist = agree, med
                score = co if c.get("attach_needs_colocation") else max(co, s or 0.0)
                if score >= c["attach_min"]:
                    opts.append((dist, y))
            if opts:
                opts.sort(key=lambda o: o[0])   # closest agreeing animal
                if len(opts) == 1 or opts[0][0] < 0.6 * opts[1][0]:  # clearly closer than the runner-up
                    merge(opts[0][1], x, "attached")
                    attached += 1

        # --- sequential identities: A ends where/when B could start -> one animal walking
        seq = 0
        def span(x):
            ts = sorted(set().union(*cl[x].occ.values()))
            return ts[0], ts[-1]
        def end_pos(x, first=False):
            best = None
            for m in cl[x].members:
                t = T[m]
                ok = np.isfinite(t.xy).all(1)
                if not ok.any():
                    continue
                i = np.flatnonzero(ok)[0 if first else -1]
                cand = (t.secs[i], t.xy[i])
                if best is None or (cand[0] < best[0] if first else cand[0] > best[0]):
                    best = cand
            return best
        changed = True
        while changed:
            changed = False
            live = [x for x in core if x in cl]
            for x in sorted(live, key=lambda x: span(x)[1]):
                ex = end_pos(x)
                if ex is None:
                    continue
                opts = []
                for y in live:
                    if y == x or conflict(x, y):
                        continue
                    sy = end_pos(y, first=True)
                    if sy is None:
                        continue
                    gap = sy[0] - ex[0]
                    if not (-c["conflict_min_s"] <= gap <= c["sequential_gap_s"]):
                        continue
                    if span(y)[0] < span(x)[1] - c["conflict_min_s"]:   # y must really start after x ends
                        continue
                    dist = float(np.hypot(*(ex[1] - sy[1])))
                    if dist <= c["walk_mps"] * max(gap, 1) + 3.0:
                        opts.append((dist, y))
                if len(opts) == 1:      # unambiguous only
                    y = opts[0][1]
                    merge(x, y, "sequential")
                    core = [z for z in core if z != y]
                    seq += 1
                    changed = True
                    break

        # --- chute rule: out-and-back-in keeps the identity
        chute = self._chute_events(cl, T, core, names)

        names = {x: f"COW_{i + 1:02d}" for i, x in enumerate(sorted([x for x in core if x in cl],
                                                                      key=lambda x: min(T[m].t0 for m in cl[x].members)))}
        mapping = {k: None for k in keys}
        for x in core:
            if x in cl:
                for m in cl[x].members:
                    mapping[m] = names[x]
        # fragments inherit their host's identity (they are part of that cow); slivers stay unassigned
        for k, host in frag.items():
            h = host
            for _ in range(5):  # host may itself be a fragment
                if h in frag:
                    h = frag[h]
                else:
                    break
            mapping[k] = mapping.get(h) if h is not None else None
        n_boxes = sum(t.n for t in T_all.values())
        n_assigned = sum(T_all[k].n for k in mapping if mapping[k])
        self.report = {**stats, "tracklets": len(keys), "mutual_agreements": len(mutual),
                       "cannot_links": len(cannot) // 2, "continuity_candidates": len(cont),
                       "attached_leftovers": attached, "forced_by_constraints": forced,
                       "track_switches_split": self.splits, "far_tracks": len(far_tracks), "sequential_merges": seq,
                       "animals_required_by_evidence": len(clique), "pen_size_cap": c["pen_size"],
                       "fragments": {"part_of_another_cow": sum(1 for v in frag.values() if v is not None),
                                     "slivers": sum(1 for v in frag.values() if v is None)}, "assigned_box_share": round(n_assigned / max(n_boxes, 1), 3),
                       "identities": {names[x]: {"cameras": sorted(cl[x].occ), "tracklets": len(cl[x].members),
                                                 "seconds": int(len(set().union(*cl[x].occ.values())))}
                                      for x in core if x in cl},
                       "chute_events": chute}
        # how sure each track's identity is, by the strongest evidence that put it there
        agreed = {m for _, a, b in mutual for m in (a, b)}
        conf_of = {"continuity_merges": 0.85, "appearance_merges": 0.7, "attached": 0.55, "": 0.8}
        self.confidence = {k: (None if mapping.get(k) is None else 0.5 if k in frag else
                               0.95 if k in agreed else conf_of.get(how.get(k, ""), 0.8)) for k in mapping}
        self.how = how
        self.fragment_keys = set(frag)
        self.clusters, self.names, self.T = cl, names, T_all
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


def max_clique(adj: Dict) -> List:
    """Bron-Kerbosch with pivoting (graphs here have < 50 nodes)."""
    best: List = []

    def bk(R, P, X):
        nonlocal best
        if not P and not X:
            if len(R) > len(best):
                best = list(R)
            return
        if len(R) + len(P) <= len(best):
            return
        u = max(P | X, key=lambda v: len(adj[v] & P))
        for v in list(P - adj[u]):
            bk(R + [v], P & adj[v], X & adj[v])
            P = P - {v}
            X = X | {v}

    bk([], set(adj), set())
    return best


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
