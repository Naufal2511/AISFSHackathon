"""Standalone-YOLO output -> `yolo_records` -> per-cow-minute upstream rows.

Input contract (CSV or JSONL, one row per cow per instant per camera). Column aliases are accepted:
  file        video file name, e.g. Event20261107033930006.Avi            (required)
  timestamp   seconds from file start | HH:MM:SS(.fff) | absolute datetime  (required)
  cow_id      the labelled cow                                              (required)
  behavior    YOLO's behaviour label for that cow at that time              (required)
  confidence  0..1                                                          (recommended)
  x1,y1,x2,y2 (or bbox "x1,y1,x2,y2")  box in pixels of that file           (recommended; needed for crops/tracks)
  camera, track_id, id_conf                                                 (optional)
Everything the agent needs beyond this (visibility, speed, neighbours, zones) is derived from the boxes and the video.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from .config import VideoCfg
from .store import Store, entropy_of, now, t_abs
from .schemas import BEHAVIORS, norm_label
from .video import Clock, VideoIndex, parse_event_filename, parse_timestamp

ALIASES = {
    "file": ["file", "video", "filename", "file_name", "video_file", "clip"],
    "timestamp": ["timestamp", "time", "t", "ts", "offset_s", "offset", "time_s", "frame_time"],
    "cow_id": ["cow_id", "cow", "cowid", "id", "cow_label", "label_id", "animal_id"],
    "behavior": ["behavior", "behaviour", "label", "class", "action", "pred", "behavior_label"],
    "conf": ["confidence", "conf", "score", "prob", "behavior_conf"],
    "camera": ["camera", "cam", "view"],
    "track_id": ["track_id", "track", "tid"],
    "id_conf": ["id_conf", "id_confidence", "identity_conf"],
    "x1": ["x1", "xmin", "left"], "y1": ["y1", "ymin", "top"], "x2": ["x2", "xmax", "right"], "y2": ["y2", "ymax", "bottom"],
    "bbox": ["bbox", "box", "xyxy"],
}


def _pick(row: dict, key: str):
    low = {k.lower().strip(): v for k, v in row.items()}
    for a in ALIASES[key]:
        if a in low and low[a] not in (None, ""):
            return low[a]
    return None


def _num(v) -> Optional[float]:
    try:
        x = float(v)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def _rows(path: Path):
    if path.suffix.lower() in (".jsonl", ".ndjson"):
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                yield json.loads(line)
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        yield from (data if isinstance(data, list) else data.get("records", []))
    else:
        with open(path, newline="", encoding="utf-8-sig") as f:
            yield from csv.DictReader(f)


def get_clock(db: Store, first: Optional[datetime], mode: str) -> Optional[Clock]:
    v = db.get_meta("clock_origin")
    if v:
        return Clock(datetime.fromisoformat(v))
    if first is None:
        return None
    c = Clock.from_first(first, mode)
    db.set_meta("clock_origin", c.origin.isoformat())
    return c


def import_yolo(db: Store, path: str | Path, index: Optional[VideoIndex], cfg: VideoCfg, default_camera: str = "front") -> dict:
    """Load YOLO records into `yolo_records`. Never silently drops: unparseable rows are counted by reason."""
    path = Path(path)
    parsed, skipped = [], Counter()
    for r in _rows(path):
        fname, cow = _pick(r, "file"), _pick(r, "cow_id")
        if not fname or not cow:
            skipped["missing_file_or_cow"] += 1; continue
        clip = index.find_by_name(fname) if index else None
        fstart = clip.start if clip else parse_event_filename(str(fname))
        off, absdt = parse_timestamp(_pick(r, "timestamp"), fstart)
        if absdt is None:
            skipped["bad_timestamp_or_filename"] += 1; continue
        lab = norm_label(_pick(r, "behavior") or "")
        if lab not in BEHAVIORS:
            skipped["unknown_behavior"] += 1; continue
        box = [_num(_pick(r, k)) for k in ("x1", "y1", "x2", "y2")]
        bb = _pick(r, "bbox")
        if bb and not any(b is not None for b in box):
            try:
                box = [float(x) for x in str(bb).strip("[]() ").replace(";", ",").split(",")][:4]
            except ValueError:
                box = [None] * 4
        cam = (_pick(r, "camera") or (clip.camera if clip else default_camera))
        parsed.append(dict(file=Path(str(fname)).name, camera=str(cam).lower(), t_offset=off, abs=absdt, cow=str(cow).strip(),
                           lab=lab, conf=_num(_pick(r, "conf")), box=box, track=_num(_pick(r, "track_id")), idc=_num(_pick(r, "id_conf"))))
    if not parsed:
        return {"rows": 0, "skipped": dict(skipped)}
    clock = get_clock(db, min(p["abs"] for p in parsed), cfg.day_origin)
    out = []
    for p in parsed:
        d, m = clock.to_dm(p["abs"])
        out.append(dict(file=p["file"], camera=p["camera"], t_offset=p["t_offset"], abs_time=p["abs"].isoformat(timespec="milliseconds"),
                        day=d, minute=m, cow_id=p["cow"], behavior=p["lab"], conf=p["conf"] if p["conf"] is not None else 1.0,
                        probs_json=None, x1=p["box"][0], y1=p["box"][1], x2=p["box"][2], y2=p["box"][3],
                        track_id=int(p["track"]) if p["track"] is not None else None, id_conf=p["idc"]))
    db.add_yolo_records(out)
    db._log_import("yolo_records", "standalone_yolo", str(path), len(out))
    return {"rows": len(out), "skipped": dict(skipped), "origin": clock.origin.isoformat()}


def infer_record_hz(db: Store, cfg: VideoCfg) -> float:
    """Records per second per cow per camera: configured, else the median spacing between consecutive record times of one
    cow in one file (so visibility is not misjudged when YOLO writes one row every few seconds)."""
    if cfg.record_hz:
        hz = float(cfg.record_hz)
    else:
        gaps = []
        prev = None
        n = 0
        for cow, day, minute, recs in db.iter_yolo():
            for r in sorted(recs, key=lambda r: (r.file, r.t_offset if r.t_offset is not None else -1)):
                if r.t_offset is None:
                    continue
                n += 1
                if prev and prev[0] == r.file and prev[1] == cow and 0 < r.t_offset - prev[2] < 30:
                    gaps.append(r.t_offset - prev[2])
                prev = (r.file, cow, r.t_offset)
            if n > 200000:
                break
        gaps.sort()
        hz = 1.0 / gaps[len(gaps) // 2] if gaps else 1.0
    db.set_meta("record_hz", round(hz, 4))
    return hz


# ----------------------------------------------------------------------------- aggregation
def _center(r) -> Optional[tuple[float, float]]:
    return None if None in (r["x1"], r["y1"], r["x2"], r["y2"]) else ((r["x1"] + r["x2"]) / 2, (r["y1"] + r["y2"]) / 2)


def _speed(recs: list[dict]) -> Optional[float]:
    """Median box-centre speed (px/s) on the camera with the most records; consecutive records <= 3 s apart only."""
    by_cam: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        if _center(r) and r["t_offset"] is not None:
            by_cam[r["camera"]].append(r)
    if not by_cam:
        return None
    best = max(by_cam.values(), key=len)
    best.sort(key=lambda r: (r["file"], r["t_offset"]))
    sp = []
    for a, b in zip(best, best[1:]):
        dt = b["t_offset"] - a["t_offset"]
        if a["file"] == b["file"] and 0 < dt <= 3.0:
            (ax, ay), (bx, by) = _center(a), _center(b)
            sp.append(math.hypot(bx - ax, by - ay) / dt)
    if not sp:
        return None
    sp.sort()
    return round(sp[len(sp) // 2], 2)


def _visibility(recs: list[dict], index: Optional[VideoIndex], hz: float) -> dict[str, float]:
    out = {}
    per_cam: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        per_cam[r["camera"]].append(r)
    for cam, rs in per_cam.items():
        presence = min(1.0, len({(r["file"], round(r["t_offset"] or 0)) for r in rs}) / max(1.0, 60 * hz))
        clip = next((c for c in (index.clips if index else []) if c.camera == cam and Path(c.path).name == rs[0]["file"]), None)
        trunc = 0.0
        if clip and clip.width and clip.height:
            edge = [r for r in rs if None not in (r["x1"], r["x2"]) and
                    (r["x1"] <= 2 or r["y1"] <= 2 or r["x2"] >= clip.width - 2 or r["y2"] >= clip.height - 2)]
            trunc = 0.4 * len(edge) / len(rs)        # ASSUMPTION: edge-touching boxes are ~partly out of frame
        out[cam] = round(max(0.0, presence * (1.0 - trunc)), 3)
    return out


def aggregate_minutes(db: Store, cfg: VideoCfg, index: Optional[VideoIndex] = None,
                      expected_cows: Optional[list[str]] = None) -> dict:
    """yolo_records -> minute_state upstream columns (+ source pointer + yolo_flags). Cow-minutes inside video coverage
    with NO YOLO record become task='find_missing' rows (the agent tries to recover them from the video)."""
    clock = get_clock(db, None, cfg.day_origin)
    if clock is None:
        return {"rows": 0, "find_missing": 0}
    hz = infer_record_hz(db, cfg)
    groups: dict[tuple, list] = defaultdict(list)
    for cow, day, minute, recs in db.iter_yolo():
        groups[(cow, day, minute)] = [r.as_dict() for r in recs]
    recs, flagged = [], Counter()
    for (cow, day, minute), rs in groups.items():
        w: dict[str, float] = defaultdict(float)
        for r in rs:
            w[r["behavior"]] += r["conf"] or 0.0
        tot = sum(w.values()) or 1.0
        mean_conf = tot / len(rs)
        top = sorted(((k, round(v / tot * min(1.0, mean_conf), 4)) for k, v in w.items()), key=lambda x: -x[1])[:4]
        rep = max((r for r in rs if r["behavior"] == top[0][0]), key=lambda r: r["conf"] or 0.0)
        flags = []
        seen = Counter((r["file"], r["camera"], round(r["t_offset"] or 0, 1)) for r in rs)
        if any(v > 1 for v in seen.values()):
            flags.append("duplicate_id")
        cam_top = {}
        for cam in {r["camera"] for r in rs}:
            ws: dict[str, float] = defaultdict(float)
            for r in rs:
                if r["camera"] == cam:
                    ws[r["behavior"]] += r["conf"] or 0.0
            cam_top[cam] = max(ws, key=ws.get)
        disagree = len(set(cam_top.values())) > 1
        if disagree:
            flags.append("camera_disagreement")
        vis = _visibility(rs, index, hz)
        idc = [r["id_conf"] for r in rs if r["id_conf"] is not None]
        for f in flags:
            flagged[f] += 1
        recs.append({"cow_id": cow, "day": day, "minute": minute, "t_abs": t_abs(day, minute), "is_night": None,
                     "tier1_top_json": json.dumps(top), "tier1_label": top[0][0], "tier1_conf": top[0][1],
                     "tier1_entropy": entropy_of(top), "speed_px_s": _speed(rs),
                     "vis_back": vis.get("back"), "vis_front": vis.get("front"), "vis_top": vis.get("top"),
                     "id_conf": round(sum(idc) / len(idc), 3) if idc else None, "camera_disagreement": int(disagree),
                     "task": "verify", "src_file": rep["file"], "src_camera": rep["camera"], "src_offset_s": rep["t_offset"],
                     "src_abs_time": rep["abs_time"], "yolo_flags": ";".join(flags) or None})
    extra = ["task", "src_file", "src_camera", "src_offset_s", "src_abs_time", "yolo_flags"]

    # misses: expected cows with no record in a minute that the video actually covers
    cows = sorted(set(expected_cows or []) | {k[0] for k in groups})
    missing = []
    if index and cows and groups:
        days_minutes = sorted({(k[1], k[2]) for k in groups})
        covered = set(days_minutes)
        for c in index.clips:                    # every minute a clip overlaps counts as covered
            t = c.start.replace(second=0, microsecond=0)
            while t < c.end:
                covered.add(clock.to_dm(t))
                t += timedelta(minutes=1)
        for cow in cows:
            for d, m in sorted(covered):
                if (cow, d, m) not in groups:
                    missing.append({"cow_id": cow, "day": d, "minute": m, "t_abs": t_abs(d, m), "task": "find_missing",
                                    "yolo_flags": "no_yolo_record"})
    res = {"rows": 0, "find_missing": 0}
    if recs:
        db.init_rows(cows, sorted({r["day"] for r in recs}), 1440)
        res["rows"] = db.upsert_minute_records(recs, "minutes", "yolo_records", "yolo_records", 0, extra)["rows"]
    if missing:
        res["find_missing"] = db.upsert_minute_records(missing, "minutes", "yolo_gap", "yolo_records", 0, extra)["rows"]
    res["flags"] = dict(flagged)
    return res
