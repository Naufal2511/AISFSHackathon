"""When should the agent look? Rules over tracking records and the behavior timeline.

Drastic events (always investigated):
  extra_cow       more distinct cow IDs than the pen holds (a 'new cow' appeared)
  new_cow_id      an ID that has no memory card yet
  cow_missing     a known cow not seen by any camera for `missing_min` minutes (left the pen?)
  duplicate_id    one ID on two boxes in the same camera frame (impossible)
  id_jump         an ID moves faster than a steer can (world coordinates, trusted cameras)
                  or is handed to a far-away track within seconds (same camera, pixels)
Low confidence:
  weak_id         a new track got its ID with low identity confidence
  behavior_check  low-confidence / gap-filled behavior bouts, rumination/grooming/social,
                  or disagreement with a second behavior classifier
Scheduled:
  id_audit        every `audit_every_min` minutes of wall clock
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

SEVERITY = {"extra_cow": 5, "new_cow_id": 5, "duplicate_id": 5, "id_jump": 4, "cow_missing": 4, "camera_conflict": 4,
            "id_audit": 3, "weak_id": 2, "behavior_check": 1}

DEFAULTS = dict(
    pen_size=10,
    missing_min=20.0,
    max_speed_mps=3.0,
    trusted_world_cameras=["cam1"],   # cam3 metric calibration is not verified yet
    handover_gap_sec=10.0,
    handover_jump_frac=0.25,          # of frame width
    weak_id_conf=0.60,
    behavior_conf=0.55,
    hard_behaviors=["ruminating (lying)", "ruminating (standing)", "grooming", "social interaction"],
    audit_every_min=30,
    max_flags_per_window=8,
)


@dataclass
class Flag:
    kind: str
    t0: float                       # epoch seconds (wall clock)
    t1: float
    cow_ids: List[str] = field(default_factory=list)
    camera: Optional[str] = None
    track_id: Optional[int] = None
    detail: str = ""
    data: Dict = field(default_factory=dict)

    @property
    def severity(self) -> int:
        return SEVERITY.get(self.kind, 0)

    @property
    def key(self) -> str:
        return f"{self.kind}|{','.join(self.cow_ids)}|{self.camera}|{self.track_id}|{int(self.t0)}"

    def to_dict(self):
        d = asdict(self)
        d["severity"] = self.severity
        return d


def _ts(t: float) -> str:
    return pd.Timestamp(t, unit="s").strftime("%H:%M:%S")


def find_flags(rec: pd.DataFrame, t0: float, t1: float, known_ids: List[str], cfg: Dict = None,
               timeline: Optional[pd.DataFrame] = None, last_seen: Optional[Dict[str, float]] = None,
               audit_times: Optional[List[float]] = None, world=None) -> List[Flag]:
    """rec: records with wall-clock column 't' (see behavior.features.load_records)."""
    c = {**DEFAULTS, **(cfg or {})}
    w = rec[(rec.t >= t0) & (rec.t < t1) & rec.global_cow_id.notna()]
    flags: List[Flag] = []

    # ---- drastic: more IDs than cows, unknown IDs
    ids = sorted(w.global_cow_id.unique())
    if len(ids) > c["pen_size"]:
        flags.append(Flag("extra_cow", t0, t1, ids, detail=f"{len(ids)} distinct IDs in window, pen holds {c['pen_size']}"))
    if known_ids:
        for cid in ids:
            if cid not in known_ids:
                first = w[w.global_cow_id == cid].sort_values("t").iloc[0]
                flags.append(Flag("new_cow_id", float(first.t), float(w[w.global_cow_id == cid].t.max()), [cid],
                                  first.camera_id, int(first.local_track_id),
                                  f"{cid} first seen {_ts(first.t)} on {first.camera_id}, no memory card"))

    # ---- drastic: duplicates inside one camera frame
    dup = w.groupby(["camera_id", "frame_number", "global_cow_id"]).local_track_id.nunique()
    for (cam, fn, cid), n in dup[dup > 1].items():
        r = w[(w.camera_id == cam) & (w.frame_number == fn) & (w.global_cow_id == cid)]
        flags.append(Flag("duplicate_id", float(r.t.min()), float(r.t.max()), [cid], cam,
                          detail=f"{cid} on {n} boxes in one {cam} frame",
                          data={"tracks": sorted(int(x) for x in r.local_track_id)}))
        break  # one is enough to trigger a look

    # ---- drastic: impossible jumps
    for cid, g in w.sort_values("t").groupby("global_cow_id"):
        gw = g[g.camera_id.isin(c["trusted_world_cameras"]) & g.world_x.notna()]
        if len(gw) > 1:
            dt = np.diff(gw.t.values)
            dd = np.hypot(np.diff(gw.world_x.values), np.diff(gw.world_y.values))
            ok = (dt > 0) & (dt <= 5)
            sp = np.where(ok, dd / np.maximum(dt, 1e-3), 0)
            if sp.max(initial=0) > c["max_speed_mps"]:
                i = int(sp.argmax())
                flags.append(Flag("id_jump", float(gw.t.values[i]), float(gw.t.values[i + 1]), [cid],
                                  gw.camera_id.values[i + 1], int(gw.local_track_id.values[i + 1]),
                                  f"{cid} moved {dd[i]:.1f} m in {dt[i]:.0f} s ({sp[i]:.1f} m/s)"))
                continue
        for cam, gc in g.groupby("camera_id"):  # handover to a far-away track in the same camera
            gc = gc.sort_values("t")
            prev = None
            for tid, gt in gc.groupby((gc.local_track_id != gc.local_track_id.shift()).cumsum()):
                if prev is not None:
                    gap = gt.t.iloc[0] - prev.t.iloc[-1]
                    jump = abs(gt.center_x.iloc[0] - prev.center_x.iloc[-1]) / max(gt.frame_width.iloc[0], 1)
                    if 0 <= gap <= c["handover_gap_sec"] and jump > c["handover_jump_frac"]:
                        flags.append(Flag("id_jump", float(prev.t.iloc[-1]), float(gt.t.iloc[0]), [cid], cam,
                                          int(gt.local_track_id.iloc[0]),
                                          f"{cid} handed from track {int(prev.local_track_id.iloc[-1])} to "
                                          f"{int(gt.local_track_id.iloc[0])} {jump:.0%} of the frame away after {gap:.0f} s"))
                prev = gt

    # ---- low confidence: weak ID on new tracks
    firsts = w.sort_values("t").groupby(["camera_id", "local_track_id"]).head(1)
    for r in firsts[firsts.identity_confidence < c["weak_id_conf"]].itertuples():
        flags.append(Flag("weak_id", float(r.t), float(r.t) + 5, [r.global_cow_id], r.camera_id, int(r.local_track_id),
                          f"track {r.local_track_id} on {r.camera_id} got {r.global_cow_id} with confidence "
                          f"{r.identity_confidence:.2f}"))

    # ---- drastic: missing cows (needs history across windows)
    if last_seen is not None:
        for cid in ids:
            last_seen[cid] = max(last_seen.get(cid, 0), float(w[w.global_cow_id == cid].t.max()))
        for cid in known_ids or []:
            ls = last_seen.get(cid)
            if ls is not None and t1 - ls > c["missing_min"] * 60:
                flags.append(Flag("cow_missing", ls, t1, [cid],
                                  detail=f"{cid} not seen by any camera since {_ts(ls)} "
                                         f"({(t1 - ls) / 60:.0f} min)"))

    # ---- low confidence behavior
    if timeline is not None and len(timeline):
        tl = timeline[(timeline.t >= t0) & (timeline.t < t1)]
        if "behavior_2" in tl.columns:  # second classifier (e.g. the YOLO behavior model)
            dis = tl[tl.behavior_2.notna() & (tl.behavior_2 != tl.behavior)]
            for cid, g in dis.groupby("cow_id"):
                if len(g) >= 20:
                    flags.append(Flag("behavior_check", float(g.t.min()), float(g.t.max()), [cid],
                                      detail=f"classifiers disagree for {len(g)} s: {g.behavior.mode()[0]} vs "
                                             f"{g.behavior_2.mode()[0]}"))
        for cid, g in tl.groupby("cow_id"):
            g = g.sort_values("t")
            seg = (g.behavior != g.behavior.shift()).cumsum()
            for _, b in g.groupby(seg):
                beh = b.behavior.iloc[0]
                conf = float(b.confidence.mean())
                hard = beh in c["hard_behaviors"]
                if len(b) >= 20 and (conf < c["behavior_conf"] or hard) and beh != "other / not visible":
                    flags.append(Flag("behavior_check", float(b.t.min()), float(b.t.max()), [cid],
                                      detail=f"{cid} {beh} for {len(b)} s, mean confidence {conf:.2f}"
                                             + (" (hard class)" if hard else ""),
                                      data={"behavior": beh, "confidence": round(conf, 3)}))

    # ---- world-state physics: jumps and real cross-camera conflicts (camera offsets are calibration, skipped)
    if world is not None and len(world.anomalies):
        an = world.anomalies
        an = an[(an.t0 >= t0) & (an.t0 < t1)]
        vis = world.visibility
        for r in an.itertuples():
            if r.type in ("camera_conflict", "camera_offset") and r.seconds < c.get("min_conflict_s", 8):
                continue
            if r.type == "camera_offset":
                odd = str(r.detail).split(" ")[0]
                try:
                    gap = float(str(r.detail).split("places it ")[1].split(" m")[0])
                except Exception:
                    gap = 99.0
                if odd == "cam3" and gap < 3.5:  # cam3's known far-range calibration error, not identity
                    continue
            if r.type not in ("jump", "camera_conflict", "camera_offset", "speed", "outside_pen"):
                continue
            v = vis[(vis.cow_id == r.cow_id) & (vis.sec == r.t0) & vis.track.notna()]
            tracks = {x.camera: int(x.track) for x in v.itertuples()}
            cam, tid = (next(iter(tracks.items())) if tracks else (None, None))
            flags.append(Flag("camera_conflict" if r.type in ("camera_conflict", "camera_offset") else "id_jump", float(r.t0), float(r.t1),
                              [r.cow_id], cam, tid, f"{r.type}: {r.detail}" + (f" ({r.seconds} s)" if r.seconds > 1 else ""),
                              data={"tracks": tracks}))

    # ---- scheduled audit
    for ta in audit_times or []:
        if t0 <= ta < t1:
            flags.append(Flag("id_audit", ta, ta, known_ids or ids, detail=f"scheduled ID check at {_ts(ta)}"))

    # one flag per (kind, cow, camera, track); most severe first, then budget
    uniq = {}
    for f in flags:
        uniq.setdefault(f.key, f)
    out = sorted(uniq.values(), key=lambda f: (-f.severity, f.t0))
    keep = [f for f in out if f.severity >= 3] + [f for f in out if f.severity < 3]
    return keep[: max(c["max_flags_per_window"], sum(f.severity >= 3 for f in out))]


def audit_schedule(t_start: float, t_end: float, every_min: int = 30) -> List[float]:
    step = every_min * 60
    first = (np.floor(t_start / step) + 1) * step
    return list(np.arange(first, t_end, step))
