"""Tracking-pipeline input -> store.

observations.csv  per camera, per second, per box: wall_time, file, offset_s, camera (cam1..), camera_name, track_id, cow_id, id_conf,
                  id_evidence, box (x1..y2), world_x/world_y (m), visibility, occluded_by ... (no behavior in this file)
cow_state.csv     per cow, per second, fused over cameras: wall_time, cow_id, speed_mps, zone, n_cams, behavior_fused, beh_conf_fused,
                  beh_evidence (observed | estimated_gap_fill | estimated_heuristic | not_visible), p_<class> ..., anomaly

What is used, and for what:
  * wall_time (= file start from the file name + offset_s, verified) is the common clock; the per-camera file names only tell which video
    file holds a second. file -> camera is stored so the video index does not depend on folder names.
  * cow_state -> the per-minute tier-1 row: top-k from the mean per-second probabilities, speed (m/s), zone fractions, stability metrics
    (dominant-label share, lowest confidence, switches, observed fraction), flags from `anomaly` and `beh_evidence`.
  * observations -> boxes (crops), world positions, per-camera visibility, identity confidence/evidence, the best camera + file + offset.
ASSUMPTIONS: day 1 starts at midnight of the first date; zone bunk -> feed_bunk, rest -> resting, other -> open (no waterer zone is given);
the camera alias (cam2 -> 'top' slot) is a naming choice, not a camera type.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from .config import AgentConfig
from .schemas import BEHAVIORS, CAMERAS, norm_label
from .store import MIN_PER_DAY, Store, entropy_of, t_abs
from .yolo import get_clock

P_COLS = {"p_lying": "lying", "p_standing": "standing", "p_walking": "walking", "p_feeding": "feeding", "p_drinking": "drinking",
          "p_rum_lying": "ruminating_lying", "p_rum_standing": "ruminating_standing", "p_grooming": "grooming",
          "p_social": "social_interaction", "p_other": "other_not_visible"}
ZONE = {"bunk": "zone_feed_bunk", "rest": "zone_resting", "other": "zone_open", "waterer": "zone_waterer"}
VIS_W = {"visible": 1.0, "partial": 0.6}


def _f(v) -> Optional[float]:
    try:
        x = float(v)
        return None if x != x else x
    except (TypeError, ValueError):
        return None


def _read(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _slots(obs: list[dict], alias: dict[str, str]) -> dict[str, str]:
    """camera id -> internal slot. Unlisted cameras take the next free slot in sorted order."""
    out = {}
    used = set(alias.values())
    free = [c for c in CAMERAS if c not in used]
    for cam in sorted({r["camera"] for r in obs}):
        out[cam] = alias.get(cam) or (free.pop(0) if free else cam)
    return out


def _night_by_minute(frames: str | Path, clock) -> dict[tuple[int, int], int]:
    """Stage 1 already decided day/night for every frame it tracked (assets_raw/frames.csv, `daylight` = colour
    frame, not IR). Minute is night when most of its frames, over all cameras, are not daylight."""
    from datetime import datetime, timedelta
    ep = datetime(1970, 1, 1)
    tot: Counter = Counter()
    dark: Counter = Counter()
    with open(frames, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            w = _f(r.get("wall_t"))
            if w is None:
                continue
            k = clock.to_dm(ep + timedelta(seconds=w))
            tot[k] += 1
            dark[k] += str(r.get("daylight", "")).strip().lower() not in ("1", "true")
    return {k: int(dark[k] * 2 > n) for k, n in tot.items() if n}


def _id_holds(trace_dir: str | Path, clock) -> set[tuple[str, int, int]]:
    """(cow, day, minute) the stage 1 identity agent could not settle (decision uncertain / escalate). Only those
    cow-minutes are held for a full check; every other cow is verified as usual."""
    import json as _json
    from datetime import datetime, timedelta
    ep, out = datetime(1970, 1, 1), set()
    for p in Path(trace_dir).glob("*.json"):
        try:
            t = _json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        f, fin = t.get("flag") or {}, t.get("final") or {}
        if fin.get("decision") not in ("uncertain", "escalate") or f.get("t0") is None:
            continue
        a, b = ep + timedelta(seconds=float(f["t0"])), ep + timedelta(seconds=float(f.get("t1") or f["t0"]))
        m = a.replace(second=0, microsecond=0)
        while m <= b:
            d, mi = clock.to_dm(m)
            for cow in f.get("cow_ids") or []:
                out.add((cow, d, mi))
            m += timedelta(minutes=1)
    return out


def import_tracking(db: Store, observations: str | Path, cow_state: str | Path, cfg: AgentConfig, index=None,
                    frames: str | Path | None = None, id_traces: str | Path | None = None) -> dict:
    obs, st = _read(observations), _read(cow_state)
    if not st:
        return {"rows": 0, "error": "cow_state file is empty"}
    from datetime import datetime
    ts = lambda s: datetime.fromisoformat(s)
    clock = get_clock(db, min(ts(r["wall_time"]) for r in st), cfg.video.day_origin)
    slot = _slots(obs, cfg.video.camera_alias)
    cam_names = {slot[r["camera"]]: r.get("camera_name") for r in obs}
    file_cam = {Path(r["file"]).name.lower(): slot[r["camera"]] for r in obs}
    db.set_meta("file_camera", file_cam)
    db.set_meta("camera_names", cam_names)
    db.set_meta("speed_unit", "m_s")
    # stage 1 classified rumination itself (DINOv2 classifier + jaw probe): the stage 2 jaw-motion screen would redo it
    db.set_meta("stage1_rumination", any((_f(r.get("p_rum_lying")) or 0) > 0 or (_f(r.get("p_rum_standing")) or 0) > 0 for r in st))
    night = _night_by_minute(frames, clock) if frames and Path(frames).exists() else {}
    db.set_meta("night_source", "stage1_frames" if night else "frame_check")
    holds = _id_holds(id_traces, clock) if id_traces and Path(id_traces).is_dir() else set()

    # ---- per-second fused state, indexed
    by_cow_sec: dict[tuple[str, str], dict] = {}
    for r in st:
        by_cow_sec[(r["cow_id"], r["wall_time"])] = r
    all_secs = sorted({r["wall_time"] for r in st})
    secs_in_min = Counter()
    for s in all_secs:
        d, m = clock.to_dm(ts(s))
        secs_in_min[(d, m)] += 1

    # ---- raw boxes -> records (identified boxes only; the unidentified are counted)
    recs, unid = [], 0
    for r in obs:
        cow = (r.get("cow_id") or "").strip()
        if not cow:
            unid += 1
            continue
        s = by_cow_sec.get((cow, r["wall_time"]))
        lab = norm_label(s["behavior_fused"]) if s else None
        lab = lab if lab in BEHAVIORS else None
        d, m = clock.to_dm(ts(r["wall_time"]))
        box = [_f(r.get(k)) for k in ("x1", "y1", "x2", "y2")]
        recs.append(dict(file=Path(r["file"]).name, camera=slot[r["camera"]], t_offset=_f(r.get("offset_s")), abs_time=r["wall_time"],
                         day=d, minute=m, cow_id=cow, behavior=lab, conf=_f(s["beh_conf_fused"]) if s else _f(r.get("det_conf")),
                         probs_json=None, x1=box[0], y1=box[1], x2=box[2], y2=box[3],
                         track_id=int(float(r["track_id"])) if r.get("track_id") not in (None, "") else None,
                         id_conf=_f(r.get("id_conf")), world_x=_f(r.get("world_x")), world_y=_f(r.get("world_y")),
                         speed_mps=_f(s["speed_mps"]) if s else None, visibility=r.get("visibility") or None,
                         id_evidence=r.get("id_evidence") or None))
    n_added = db.add_yolo_records(recs, dedupe=True)
    db._log_import("tracking_observations", "tracking", str(observations), len(obs))

    # ---- observation aggregates per (cow, day, minute)
    ob: dict[tuple, list[dict]] = defaultdict(list)
    for rec in recs:
        ob[(rec["cow_id"], rec["day"], rec["minute"])].append(rec)

    # ---- minute rows from the fused state
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in st:
        d, m = clock.to_dm(ts(r["wall_time"]))
        groups[(r["cow_id"], d, m)].append(r)
    rows, flag_count = [], Counter()
    for (cow, day, minute), rs in groups.items():
        n = len(rs)
        expect = max(secs_in_min[(day, minute)], n)
        vecs = []
        for r in rs:
            if r["beh_evidence"] == "not_visible":
                vecs.append({"other_not_visible": 1.0})
            else:
                vecs.append({lab: _f(r.get(pc)) or 0.0 for pc, lab in P_COLS.items()})
        mean = {lab: sum(v.get(lab, 0.0) for v in vecs) / n for lab in BEHAVIORS}
        top = sorted(((k, round(v, 4)) for k, v in mean.items() if v > 0.005), key=lambda x: -x[1])[:4] or [("other_not_visible", 1.0)]
        labels = [norm_label(r["behavior_fused"]) for r in rs]
        top_share = Counter(labels).most_common(1)[0][1] / n
        switches = sum(1 for a, b in zip(labels, labels[1:]) if a != b)
        spd = [_f(r["speed_mps"]) or 0.0 for r in rs]
        zc = Counter(ZONE.get(r["zone"]) for r in rs)
        evid = Counter(r["beh_evidence"] for r in rs)
        anom = Counter(a for r in rs for a in (r.get("anomaly") or "").split(",") if a)
        o_rows = ob.get((cow, day, minute), [])
        vis = {}
        for sl in CAMERAS:
            per_sec: dict[str, float] = {}
            for o in o_rows:
                if o["camera"] == sl:
                    per_sec[o["abs_time"]] = max(per_sec.get(o["abs_time"], 0.0), VIS_W.get(o["visibility"] or "", 0.5))
            vis[sl] = round(min(1.0, sum(per_sec.values()) / expect), 3) if per_sec else 0.0
        idc = [o["id_conf"] for o in o_rows if o["id_conf"] is not None]
        flags = []
        if anom.get("camera_conflict", 0) >= 3:
            flags.append("camera_conflict")
        if switches >= 2 or top_share < 0.8:
            flags.append("label_unstable")
        if evid.get("estimated_gap_fill", 0) + evid.get("estimated_heuristic", 0) >= 0.2 * n:
            flags.append("gap_fill")
        if evid.get("not_visible", 0) >= 0.5 * n:
            flags.append("not_visible")
        if (cow, day, minute) in holds:
            flags.append("id_unresolved")      # identity agent left this cow's ID open here: full check, no carry
        for f in flags:
            flag_count[f] += 1
        best = max(CAMERAS, key=lambda c: vis[c])
        src = None
        if o_rows:
            cand = [o for o in o_rows if o["camera"] == best] or o_rows
            centre = clock.to_dt(day, minute, 0.0)
            src = min(cand, key=lambda o: abs((ts(o["abs_time"]) - centre).total_seconds()))
        rows.append({"cow_id": cow, "day": day, "minute": minute, "t_abs": t_abs(day, minute), "is_night": night.get((day, minute)),
                     "tier1_top_json": json.dumps(top), "tier1_label": top[0][0], "tier1_conf": top[0][1], "tier1_entropy": entropy_of(top),
                     "zone_feed_bunk": zc.get("zone_feed_bunk", 0) / n, "zone_waterer": None, "zone_resting": zc.get("zone_resting", 0) / n,
                     "zone_open": zc.get("zone_open", 0) / n, "speed_px_s": round(float(np.mean(spd)), 4), "posture": None,
                     "vis_back": vis["back"], "vis_front": vis["front"], "vis_top": vis["top"],
                     "id_conf": round(sum(idc) / len(idc), 3) if idc else 0.5,
                     "camera_disagreement": int("camera_conflict" in flags),
                     "speed_p90": round(float(np.percentile(spd, 90)), 4), "stability": round(top_share, 4),
                     "min_conf": round(min((_f(r["beh_conf_fused"]) or 0.0) for r in rs), 4),
                     "obs_frac": round(min(1.0, evid.get("observed", 0) / expect), 4), "n_switches": switches,
                     "task": "verify", "src_file": src["file"] if src else None, "src_camera": src["camera"] if src else None,
                     "src_offset_s": src["t_offset"] if src else None, "src_abs_time": src["abs_time"] if src else None,
                     "yolo_flags": ";".join(flags) or None})
    extra = ["task", "src_file", "src_camera", "src_offset_s", "src_abs_time", "yolo_flags"]

    # ---- cows with no fused state at all in a minute of the window: the video may still show them
    cows = sorted({r["cow_id"] for r in st})
    window = sorted(secs_in_min)
    missing = [{"cow_id": c, "day": d, "minute": m, "t_abs": t_abs(d, m), "task": "find_missing", "yolo_flags": "no_yolo_record"}
               for c in cows for d, m in window if (c, d, m) not in groups]
    res = {"cows": cows, "minutes_with_state": len(rows), "boxes_added": n_added, "unidentified_boxes": unid,
           "cameras": {sl: cam_names.get(sl) for sl in sorted(cam_names)}, "speed_unit": "m_s", "flags": dict(flag_count),
           "night_minutes_from_stage1": sum(night.values()) if night else None,
           "id_hold_minutes": len(holds),
           "stage1_rumination": db.get_meta("stage1_rumination")}
    db.init_rows(cows, sorted({r["day"] for r in rows}), MIN_PER_DAY)
    res["rows"] = db.upsert_minute_records(rows, "minutes", "tracking_state", str(cow_state), 0, extra)["rows"]
    if missing:
        res["find_missing"] = db.upsert_minute_records(missing, "minutes", "tracking_gap", str(cow_state), 0, extra)["rows"]
    return res
