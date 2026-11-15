"""File-based store (CSV + JSON/JSONL, no database).

store_dir/
  meta.json                        clock origin, inferred YOLO record rate, schema version
  cows.json                        cow ids
  imports.jsonl                    one line per import (what, from which file, how many rows)
  yolo/<cow>_d<day>.jsonl          raw records of the standalone YOLO process (one JSON object per line)
  minutes/<cow>_d<day>.csv         THE working table: one row per cow-minute (1440/day): upstream columns + agent result columns.
                                   Open it in Excel; it is also what the agent reads its per-cow history from.
  actions/<cow>_d<day>.jsonl       every action the agent took, with its reason (kept per run, reruns append)
  runs/<cow>_d<day>.jsonl          every decision ever written (reruns with another model/prompt stay comparable)
  human_labels.csv, track_obs.csv  optional evaluation labels / tracker output

Why these formats: the per-minute table is small (28,800 rows for 10 cows x 2 days) and wants to be human-readable -> CSV, one file per
cow-day. Raw YOLO records are many (up to millions) and append-only -> JSONL, one file per cow-day, loaded lazily. Actions/runs are
nested, append-only logs -> JSONL. Nothing needs cross-file transactions: a worker owns exactly one cow's files, writes are atomic
(temp file + rename), and a crash loses at most the last few unflushed minutes, which the resumable run simply redoes.
`t_abs = (day-1)*1440 + minute` is a continuous minute index; day 1 -> day 2 is assumed contiguous.
"""
from __future__ import annotations

import csv
import json
import math
import os
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from .config import HistoryCfg
from .schemas import BEHAVIORS, CAMERAS, Brief, Decision, norm_label

MIN_PER_DAY = 1440

UPSTREAM_COLS = ["is_night", "tier1_top_json", "tier1_label", "tier1_conf", "tier1_entropy",
                 "zone_feed_bunk", "zone_waterer", "zone_resting", "zone_open", "speed_px_s", "posture",
                 "vis_back", "vis_front", "vis_top", "id_conf", "camera_disagreement",
                 "speed_p90", "stability", "min_conf", "obs_frac", "n_switches"]
VIDEO_COLS = ["task", "src_file", "src_camera", "src_offset_s", "src_abs_time", "yolo_flags"]
AGENT_COLS = ["agent_status", "behavior", "confidence", "source", "tier", "reason", "correction", "rationale", "cameras_used", "flags",
              "behavior_tags", "used_history", "n_tool_calls", "n_vlm_calls", "tokens_in", "tokens_out", "trace_id", "provider", "model",
              "prompt_version", "updated_at", "id_verified", "id_suggested", "is_night_final"]
MINUTE_COLS = ["cow_id", "day", "minute", "t_abs", "upstream_run_id"] + UPSTREAM_COLS + VIDEO_COLS + AGENT_COLS
STALE_IF_CHANGED = ["tier1_top_json", "id_conf", "vis_back", "vis_front", "vis_top", "speed_px_s", "posture"]
INT_COLS = {"n_switches", "day", "minute", "t_abs", "upstream_run_id", "is_night", "camera_disagreement", "tier", "used_history", "n_tool_calls",
            "n_vlm_calls", "tokens_in", "tokens_out", "id_verified", "is_night_final"}
FLOAT_COLS = {"speed_p90", "stability", "min_conf", "obs_frac", "tier1_conf", "tier1_entropy", "zone_feed_bunk", "zone_waterer", "zone_resting", "zone_open", "speed_px_s", "vis_back",
              "vis_front", "vis_top", "id_conf", "confidence", "src_offset_s"}
RESET_COLS = [c for c in AGENT_COLS if c != "agent_status"]
YOLO_FIELDS = ["file", "camera", "t_offset", "abs_time", "day", "minute", "cow_id", "behavior", "conf", "probs_json",
               "x1", "y1", "x2", "y2", "track_id", "id_conf",
               "world_x", "world_y", "speed_mps", "visibility", "id_evidence"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def t_abs(day: int, minute: int) -> int:
    return (int(day) - 1) * MIN_PER_DAY + int(minute)


def _f(v: Any) -> Optional[float]:
    if v is None or str(v).strip() == "":
        return None
    try:
        x = float(v)
    except ValueError:
        return None
    return None if math.isnan(x) else x


def _b(v: Any) -> Optional[int]:
    if v is None or str(v).strip() == "":
        return None
    return 1 if str(v).strip().lower() in ("1", "true", "yes", "t", "y") else 0


def parse_tier1(v: Any) -> list[tuple[str, float]]:
    """Accepts JSON '[["standing",0.41],["feeding",0.33]]' or 'standing:0.41;feeding:0.33'."""
    if v is None or str(v).strip() == "":
        return []
    s = str(v).strip()
    pairs: list = []
    if s.startswith("["):
        pairs = json.loads(s)
    else:
        for part in s.split(";"):
            if ":" in part:
                a, b = part.rsplit(":", 1)
                pairs.append([a, b])
    out = []
    for item in pairs:
        lab = norm_label(item[0] if not isinstance(item, dict) else item.get("label", ""))
        p = _f(item[1] if not isinstance(item, dict) else item.get("p"))
        if lab in BEHAVIORS and p is not None:
            out.append((lab, max(0.0, min(1.0, p))))
    return sorted(out, key=lambda x: -x[1])


def entropy_of(top: list[tuple[str, float]]) -> float:
    ps = [p for _, p in top if p > 0]
    rest = max(0.0, 1.0 - sum(ps))
    if rest > 1e-9:
        ps.append(rest)
    return float(-sum(p * math.log(p) for p in ps))


def _cell(col: str, v: Any) -> Any:
    if v is None:
        return ""
    return v


def _parse_cell(col: str, s: str) -> Any:
    if s == "":
        return None
    if col in INT_COLS:
        try:
            return int(float(s))
        except ValueError:
            return None
    if col in FLOAT_COLS:
        return _f(s)
    return s


def _atomic_write(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}_{threading.get_ident()}")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer(f)
    os.replace(tmp, path)


class Rec:
    """One YOLO record, compact (millions can be loaded) but indexable like a dict: rec['x1']."""
    __slots__ = tuple(YOLO_FIELDS) + ("sec",)

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __getitem__(self, k):
        return getattr(self, k)

    def __setitem__(self, k, v):
        setattr(self, k, v)

    def get(self, k, default=None):
        v = getattr(self, k, None)
        return default if v is None else v

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in YOLO_FIELDS}


# process-wide cache of YOLO shards, shared by every Store/worker thread so each shard is read once
_YCACHE: dict[tuple[str, str, int], dict[int, list[Rec]]] = {}
_YLOCK = threading.RLock()


class Store:
    def __init__(self, path: str | Path):
        self.root = Path(path)
        self.path = str(self.root)
        for sub in ("yolo", "minutes", "actions", "runs"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._min: dict[tuple[str, int], dict[int, dict]] = {}
        self._dirty: set[tuple[str, int]] = set()
        self._writes = 0
        self._meta = self._read_json(self.root / "meta.json", {})
        if self._meta.get("schema_version") != 3:
            self._meta["schema_version"] = 3
            self._write_json(self.root / "meta.json", self._meta)

    # ------------------------------------------------------------------ small helpers
    @staticmethod
    def _read_json(p: Path, default):
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return default

    @staticmethod
    def _write_json(p: Path, obj) -> None:
        _atomic_write(p, lambda f: f.write(json.dumps(obj, indent=1, default=str)))

    def _log_import(self, kind: str, source: str, file: str, n: int) -> int:
        with self._lock:
            p = self.root / "imports.jsonl"
            run_id = (sum(1 for _ in open(p, encoding="utf-8")) if p.exists() else 0) + 1
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps({"run_id": run_id, "kind": kind, "source": source, "file": file, "n_rows": n, "imported_at": now()}) + "\n")
            return run_id

    def get_meta(self, key: str, default=None):
        self._meta = self._read_json(self.root / "meta.json", self._meta)
        return self._meta.get(key, default)

    def set_meta(self, key: str, value) -> None:
        with self._lock:
            self._meta = self._read_json(self.root / "meta.json", self._meta)
            self._meta[key] = value
            self._write_json(self.root / "meta.json", self._meta)

    def close(self):
        self.flush()

    def reload(self):
        """Drop in-memory copies so rows written by other workers/processes are re-read from disk."""
        with self._lock:
            self.flush()
            self._min.clear()

    # ------------------------------------------------------------------ minute files
    def _mpath(self, cow: str, day: int) -> Path:
        return self.root / "minutes" / f"{cow}_d{int(day)}.csv"

    def _load(self, cow: str, day: int) -> dict[int, dict]:
        key = (cow, int(day))
        with self._lock:
            if key not in self._min:
                rows: dict[int, dict] = {}
                p = self._mpath(cow, day)
                if p.exists():
                    with open(p, newline="", encoding="utf-8") as f:
                        for r in csv.DictReader(f):
                            row = {c: _parse_cell(c, r.get(c, "")) for c in MINUTE_COLS}
                            row["cow_id"] = cow
                            rows[row["minute"]] = row
                self._min[key] = rows
            return self._min[key]

    def flush(self) -> None:
        with self._lock:
            for cow, day in sorted(self._dirty):
                rows = self._min.get((cow, day), {})

                def w(f, rows=rows):
                    cw = csv.writer(f)
                    cw.writerow(MINUTE_COLS)
                    for m in sorted(rows):
                        cw.writerow([_cell(c, rows[m].get(c)) for c in MINUTE_COLS])
                _atomic_write(self._mpath(cow, day), w)
            self._dirty.clear()

    def _touch(self, cow: str, day: int, flush_every: int = 150) -> None:
        self._dirty.add((cow, int(day)))
        self._writes += 1
        if self._writes % flush_every == 0:
            self.flush()

    def _days_on_disk(self, cow: str) -> list[int]:
        return sorted(int(p.stem.rsplit("_d", 1)[1]) for p in (self.root / "minutes").glob(f"{cow}_d*.csv"))

    def _all_files(self) -> list[tuple[str, int]]:
        out = set(self._min) | self._dirty
        for p in (self.root / "minutes").glob("*_d*.csv"):
            cow, d = p.stem.rsplit("_d", 1)
            out.add((cow, int(d)))
        return sorted(out)

    def all_rows(self) -> Iterable[dict]:
        for cow, day in self._all_files():
            rows = self._load(cow, day)
            for m in sorted(rows):
                yield rows[m]

    # ------------------------------------------------------------------ setup / import
    def cows(self) -> list[str]:
        known = set(self._read_json(self.root / "cows.json", []))
        known |= {c for c, _ in self._all_files()}
        return sorted(known)

    def add_cows(self, cows: Iterable[str]) -> None:
        with self._lock:
            have = set(self._read_json(self.root / "cows.json", []))
            new = have | set(cows)
            if new != have:
                self._write_json(self.root / "cows.json", sorted(new))

    def init_rows(self, cows: Iterable[str], days: Iterable[int], n_minutes: int = MIN_PER_DAY) -> int:
        """Pre-create every cow-minute so the final activity table has no gaps even if upstream misses minutes."""
        cows, days = list(cows), list(days)
        self.add_cows(cows)
        n = 0
        for c in cows:
            for d in days:
                rows = self._load(c, d)
                for m in range(n_minutes):
                    if m not in rows:
                        rows[m] = {k: None for k in MINUTE_COLS}
                        rows[m].update(cow_id=c, day=d, minute=m, t_abs=t_abs(d, m), agent_status="pending", used_history=0,
                                       camera_disagreement=0)
                        n += 1
                self._dirty.add((c, d))
        self.flush()
        return n

    def import_minutes_csv(self, path: str | Path, source: str = "student1_tier1") -> dict:
        """Upsert UPSTREAM columns only. Columns: cow_id, day, minute, tier1_top | (tier1_label, tier1_conf), [tier1_entropy,
        is_night, zone_*, speed_px_s, posture, vis_back, vis_front, vis_top, id_conf, camera_disagreement]."""
        recs, bad = [], 0
        with open(path, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                try:
                    cow, day, minute = r["cow_id"].strip(), int(r["day"]), int(r["minute"])
                except (KeyError, ValueError, AttributeError):
                    bad += 1
                    continue
                top = parse_tier1(r.get("tier1_top"))
                if not top and r.get("tier1_label"):
                    lab, p = norm_label(r["tier1_label"]), _f(r.get("tier1_conf")) or 1.0
                    top = [(lab, p)] if lab in BEHAVIORS else []
                ent = _f(r.get("tier1_entropy"))
                recs.append({
                    "cow_id": cow, "day": day, "minute": minute, "t_abs": t_abs(day, minute),
                    "is_night": _b(r.get("is_night")),
                    "tier1_top_json": json.dumps(top) if top else None,
                    "tier1_label": top[0][0] if top else None, "tier1_conf": top[0][1] if top else None,
                    "tier1_entropy": ent if ent is not None else (entropy_of(top) if top else None),
                    "zone_feed_bunk": _f(r.get("zone_feed_bunk")), "zone_waterer": _f(r.get("zone_waterer")),
                    "zone_resting": _f(r.get("zone_resting")), "zone_open": _f(r.get("zone_open")),
                    "speed_px_s": _f(r.get("speed_px_s")),
                    "posture": (r.get("posture") or "").strip().lower() or None,
                    "vis_back": _f(r.get("vis_back")), "vis_front": _f(r.get("vis_front")), "vis_top": _f(r.get("vis_top")),
                    "id_conf": _f(r.get("id_conf")), "camera_disagreement": _b(r.get("camera_disagreement")) or 0})
        return self.upsert_minute_records(recs, "minutes", source, str(path), bad)

    def upsert_minute_records(self, recs: list[dict], kind: str, source: str, file: str, bad: int = 0,
                              extra_cols: Optional[list[str]] = None) -> dict:
        """Upsert UPSTREAM (and optional video) columns. A row the agent already finished becomes 'stale' when its
        tier-1/visibility inputs change materially (old result kept, re-processed on the next run)."""
        extra_cols = extra_cols or []
        run_id = self._log_import(kind, source, file, len(recs))
        touched: set[tuple[str, int]] = set()
        for r in recs:
            rows = self._load(r["cow_id"], r["day"])
            old = rows.get(r["minute"])
            new = {k: None for k in MINUTE_COLS} if old is None else old
            if old is None:
                new.update(agent_status="pending", used_history=0)
            was_final = old is not None and old.get("agent_status") not in (None, "pending", "stale")
            changed = old is not None and any(old.get(c) != r.get(c) for c in STALE_IF_CHANGED)
            new.update(cow_id=r["cow_id"], day=r["day"], minute=r["minute"], t_abs=t_abs(r["day"], r["minute"]), upstream_run_id=run_id)
            for c in UPSTREAM_COLS + extra_cols:
                new[c] = r.get(c)
            if was_final and changed:
                new["agent_status"] = "stale"
            rows[r["minute"]] = new
            touched.add((r["cow_id"], int(r["day"])))
        self._dirty |= touched
        self.add_cows({r["cow_id"] for r in recs})
        self.flush()
        return {"run_id": run_id, "rows": len(recs), "skipped_bad_rows": bad}

    def import_track_obs_csv(self, path: str | Path, source: str = "student1_tracker") -> dict:
        """Tracker output is kept as-is (store_dir/track_obs.csv) for backends that want it."""
        import shutil
        n = sum(1 for _ in open(path, encoding="utf-8")) - 1
        shutil.copyfile(path, self.root / "track_obs.csv")
        self._log_import("track_obs", source, str(path), n)
        return {"rows": n}

    def import_human_labels_csv(self, path: str | Path) -> dict:
        """Columns: cow_id, day, minute, behavior, [labeler, split]"""
        recs = {}
        p = self.root / "human_labels.csv"
        if p.exists():
            with open(p, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    recs[(r["cow_id"], int(r["day"]), int(r["minute"]), r["labeler"])] = r
        n = 0
        with open(path, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                lab = norm_label(r["behavior"])
                if lab not in BEHAVIORS:
                    continue
                key = (r["cow_id"].strip(), int(r["day"]), int(r["minute"]), (r.get("labeler") or "unknown").strip())
                recs[key] = {"cow_id": key[0], "day": key[1], "minute": key[2], "behavior": lab, "labeler": key[3],
                             "split": r.get("split") or "", "created_at": now()}
                n += 1

        def w(f):
            cw = csv.DictWriter(f, fieldnames=["cow_id", "day", "minute", "behavior", "labeler", "split", "created_at"])
            cw.writeheader()
            cw.writerows(recs.values())
        _atomic_write(p, w)
        return {"rows": n}

    def human_labels(self) -> list[dict]:
        p = self.root / "human_labels.csv"
        if not p.exists():
            return []
        with open(p, newline="", encoding="utf-8") as f:
            return [{**r, "day": int(r["day"]), "minute": int(r["minute"])} for r in csv.DictReader(f)]

    # ------------------------------------------------------------------ minute reads
    def get_row(self, cow: str, day: int, minute: int) -> Optional[dict]:
        r = self._load(cow, day).get(int(minute))
        return dict(r) if r else None

    def row_at(self, cow: str, t: int) -> Optional[dict]:
        day, minute = t // MIN_PER_DAY + 1, t % MIN_PER_DAY
        return self.get_row(cow, day, minute)

    def window(self, cow: str, lo: int, hi: int) -> list[dict]:
        out = []
        for t in range(max(0, lo), hi + 1):
            r = self._load(cow, t // MIN_PER_DAY + 1).get(t % MIN_PER_DAY)
            if r:
                out.append(r)
        return out

    def iter_rows(self, cow_id: str, days: Optional[list[int]] = None,
                  statuses: tuple[str, ...] = ("pending", "stale")) -> list[dict]:
        out = []
        for d in (days or self._days_on_disk(cow_id) or []):
            rows = self._load(cow_id, d)
            out += [dict(r) for r in rows.values() if r.get("agent_status") in statuses]
        return sorted(out, key=lambda r: r["t_abs"])

    def status(self) -> dict:
        rows = list(self.all_rows())
        by = lambda k: Counter(str(r[k]) for r in rows if r.get(k) is not None)
        s = lambda k: sum((r.get(k) or 0) for r in rows)
        pm = Counter(f"{r['provider']}/{r['model']}" for r in rows if r.get("provider"))
        return {"rows_total": len(rows), "rows_with_upstream": sum(1 for r in rows if r.get("tier1_top_json")),
                "by_status": dict(Counter(r["agent_status"] for r in rows)), "by_reason": dict(by("reason")),
                "by_provider_model": dict(pm), "tokens_in": s("tokens_in"), "tokens_out": s("tokens_out"),
                "vlm_calls": s("n_vlm_calls"), "used_history_rows": s("used_history")}

    # ------------------------------------------------------------------ write-back
    @staticmethod
    def _status_for(d: Decision) -> str:
        if d.reason in ("keep_tier1",):
            return "passthrough"
        if d.reason == "stable_carry":
            return "carried"
        if d.reason in ("auto_abstain", "no_upstream_data"):
            return "abstained"
        if d.reason in ("error_fallback", "budget_fallback"):
            return "fallback"
        return "done"

    def write_decision(self, d: Decision) -> None:
        """Write the agent's result into the cow-minute row and append the decision + its actions to the run logs."""
        with self._lock:
            rows = self._load(d.cow_id, d.day)
            row = rows.get(d.minute)
            if row is None:
                raise KeyError(f"no row for {d.cow_id} day {d.day} minute {d.minute} (run init first)")
            vals = {"agent_status": self._status_for(d), "behavior": d.behavior, "confidence": d.confidence, "source": d.source,
                    "tier": d.tier, "reason": d.reason, "cameras_used": ";".join(d.cameras_used), "flags": ";".join(d.flags),
                    "behavior_tags": json.dumps(d.behavior_tags), "used_history": int(d.used_history), "n_tool_calls": d.n_tool_calls,
                    "n_vlm_calls": d.n_vlm_calls, "tokens_in": d.tokens_in, "tokens_out": d.tokens_out, "trace_id": d.trace_id,
                    "provider": d.provider, "model": d.model, "prompt_version": d.prompt_version, "updated_at": now(),
                    "rationale": d.rationale, "correction": d.correction,
                    "id_verified": None if d.id_verified is None else int(d.id_verified), "id_suggested": d.id_suggested,
                    "is_night_final": int(d.is_night)}
            row.update(vals)
            self._touch(d.cow_id, d.day)
            run_id = time.time_ns()
            shard = f"{d.cow_id}_d{d.day}.jsonl"
            with open(self.root / "runs" / shard, "a", encoding="utf-8") as f:
                f.write(json.dumps({"run_id": run_id, "cow_id": d.cow_id, "day": d.day, "minute": d.minute, "behavior": d.behavior,
                                    "confidence": d.confidence, "source": d.source, "tier": d.tier, "reason": d.reason,
                                    "flags": vals["flags"], "correction": d.correction, "rationale": d.rationale,
                                    "n_tool_calls": d.n_tool_calls, "n_vlm_calls": d.n_vlm_calls, "provider": d.provider,
                                    "model": d.model, "prompt_version": d.prompt_version, "created_at": vals["updated_at"]}) + "\n")
            with open(self.root / "actions" / shard, "a", encoding="utf-8") as f:
                for a in d.actions:
                    f.write(json.dumps({"run_id": run_id, "cow_id": d.cow_id, "day": d.day, "minute": d.minute, "seq": a["seq"],
                                        "actor": a["actor"], "action": a["action"], "args": a.get("args") or {}, "reason": a["reason"],
                                        "outcome": str(a.get("outcome", ""))[:400], "source": a.get("source", "")}, default=str) + "\n")

    def actions_for(self, cow: str, day: int, minute: int) -> list[dict]:
        """Actions of the LATEST run for this cow-minute, in order."""
        p = self.root / "actions" / f"{cow}_d{day}.jsonl"
        if not p.exists():
            return []
        mine = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        mine = [a for a in mine if a["minute"] == minute]
        if not mine:
            return []
        last = max(a["run_id"] for a in mine)
        return sorted((a for a in mine if a["run_id"] == last), key=lambda a: a["seq"])

    def reset_agent(self, cows: Optional[list[str]] = None, days: Optional[list[int]] = None) -> int:
        """Clear agent columns (run logs are kept) so rows are re-processed."""
        n = 0
        for cow, day in self._all_files():
            if (cows and cow not in cows) or (days and day not in days):
                continue
            for r in self._load(cow, day).values():
                r["agent_status"] = "pending"
                for c in RESET_COLS:
                    r[c] = None
                r["used_history"] = 0
                n += 1
            self._dirty.add((cow, day))
        self.flush()
        return n

    # ------------------------------------------------------------------ bouts / export / eval
    def rebuild_bouts(self) -> list[dict]:
        bouts: list[dict] = []
        cur: Optional[dict] = None
        for r in self.all_rows():
            if r.get("behavior") is None:
                continue
            if cur and (r["cow_id"], r["day"], r["behavior"]) == (cur["cow_id"], cur["day"], cur["behavior"]) and r["minute"] == cur["end_minute"] + 1:
                cur["end_minute"] = r["minute"]; cur["_conf"].append(r["confidence"] or 0.0); cur["n_agent_minutes"] += int(r.get("tier") == 2)
            else:
                if cur:
                    bouts.append(self._close(cur))
                cur = {"cow_id": r["cow_id"], "day": r["day"], "behavior": r["behavior"], "start_minute": r["minute"],
                       "end_minute": r["minute"], "_conf": [r["confidence"] or 0.0], "n_agent_minutes": int(r.get("tier") == 2)}
        if cur:
            bouts.append(self._close(cur))
        return bouts

    @staticmethod
    def _close(c: dict) -> dict:
        n = c["end_minute"] - c["start_minute"] + 1
        return {"cow_id": c["cow_id"], "day": c["day"], "behavior": c["behavior"], "start_minute": c["start_minute"],
                "end_minute": c["end_minute"], "duration_min": n, "mean_confidence": round(sum(c["_conf"]) / n, 4),
                "n_agent_minutes": c["n_agent_minutes"]}

    def export_activity(self, out_dir: str | Path) -> dict:
        """activity_minute.csv (one row per cow-minute), activity.csv (one row per behavior bout), actions.csv (every action + reason)."""
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        bouts = self.rebuild_bouts()
        cols = ["cow_id", "day", "minute", "behavior", "confidence", "source", "tier", "reason", "correction", "tier1_label", "rationale",
                "behavior_tags", "flags", "cameras_used", "is_night_final", "id_verified", "id_suggested", "src_file", "src_camera", "src_offset_s"]
        n = missing = 0
        with open(out / "activity_minute.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(cols)
            for r in self.all_rows():
                if r.get("behavior") is None:
                    missing += 1
                    continue
                w.writerow([_cell(c, r.get(c)) for c in cols]); n += 1
        bcols = ["cow_id", "day", "behavior", "start_minute", "end_minute", "duration_min", "mean_confidence", "n_agent_minutes"]
        with open(out / "activity.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(bcols)
            for b in sorted(bouts, key=lambda b: (b["cow_id"], b["day"], b["start_minute"])):
                w.writerow([b[c] for c in bcols])
        acols = ["cow_id", "day", "minute", "run_id", "seq", "actor", "action", "args", "reason", "outcome", "source"]
        with open(out / "actions.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(acols)
            for p in sorted((self.root / "actions").glob("*.jsonl")):
                latest: dict[int, int] = {}
                items = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
                for a in items:
                    latest[a["minute"]] = max(latest.get(a["minute"], 0), a["run_id"])
                for a in items:
                    if a["run_id"] == latest[a["minute"]]:
                        w.writerow([a["cow_id"], a["day"], a["minute"], a["run_id"], a["seq"], a["actor"], a["action"],
                                    json.dumps(a["args"], default=str), a["reason"], a["outcome"], a["source"]])
        return {"minute_rows": n, "bouts": len(bouts), "rows_without_label": missing}

    def evaluate_vs_human(self) -> dict:
        """Accuracy of final labels vs hand labels, with the tier-1-only baseline on the same rows (the ablation)."""
        rows = []
        for h in self.human_labels():
            m = self.get_row(h["cow_id"], h["day"], h["minute"])
            if m and m.get("behavior") is not None:
                rows.append({"truth": h["behavior"], "final": m["behavior"], "t1": m.get("tier1_label"), "tier": m.get("tier"),
                             "night": int(m.get("is_night_final") if m.get("is_night_final") is not None else (m.get("is_night") or 0))})
        if not rows:
            return {"n": 0}
        acc = lambda pairs: round(sum(a == b for a, b in pairs) / len(pairs), 4) if pairs else None
        esc = [r for r in rows if r["tier"] == 2]
        per_class: dict[str, dict] = {}
        for lab in BEHAVIORS:
            sub = [r for r in rows if r["truth"] == lab]
            if sub:
                per_class[lab] = {"n": len(sub), "recall_final": acc([(r["truth"], r["final"]) for r in sub]),
                                  "recall_tier1": acc([(r["truth"], r["t1"]) for r in sub])}
        split = {}
        for name, flag in (("day", 0), ("night", 1)):
            sub = [r for r in rows if r["night"] == flag]
            if sub:
                split[name] = {"n": len(sub), "accuracy_final": acc([(r["truth"], r["final"]) for r in sub]),
                               "accuracy_tier1_only": acc([(r["truth"], r["t1"]) for r in sub])}
        return {"n": len(rows), "by_light": split, "accuracy_final": acc([(r["truth"], r["final"]) for r in rows]),
                "accuracy_tier1_only": acc([(r["truth"], r["t1"]) for r in rows]), "n_escalated": len(esc),
                "escalated_accuracy_final": acc([(r["truth"], r["final"]) for r in esc]),
                "escalated_accuracy_tier1_only": acc([(r["truth"], r["t1"]) for r in esc]), "per_class": per_class}

    # ------------------------------------------------------------------ raw YOLO records (JSONL shards, cached process-wide)
    def _ypath(self, cow: str, day: int) -> Path:
        return self.root / "yolo" / f"{cow}_d{int(day)}.jsonl"

    def _yshard(self, cow: str, day: int) -> dict[int, list[Rec]]:
        key = (self.path, cow, int(day))
        with _YLOCK:
            if key not in _YCACHE:
                by_min: dict[int, list[Rec]] = defaultdict(list)
                p = self._ypath(cow, day)
                if p.exists():
                    with open(p, encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                o = json.loads(line)
                                by_min[o["minute"]].append(Rec(**o))
                _YCACHE[key] = by_min
            return _YCACHE[key]

    def drop_cache(self) -> None:
        with _YLOCK:
            for k in [k for k in _YCACHE if k[0] == self.path]:
                del _YCACHE[k]

    @staticmethod
    def _ykey(o) -> tuple:
        g = o.get if isinstance(o, dict) else (lambda k: getattr(o, k))
        return (g("file"), g("camera"), g("t_offset"), g("track_id"), g("x1"), g("y1"))

    def add_yolo_records(self, recs: list[dict], dedupe: bool = False) -> int:
        """Append raw records. dedupe=True skips records already stored (same file/camera/time/track/box), so re-importing is safe."""
        by_shard: dict[tuple[str, int], list[dict]] = defaultdict(list)
        for r in recs:
            by_shard[(r["cow_id"], int(r["day"]))].append({k: r.get(k) for k in YOLO_FIELDS})
        added = 0
        for (cow, day), items in by_shard.items():
            if dedupe:
                have = {self._ykey(x) for recs_ in self._yshard(cow, day).values() for x in recs_}
                items = [o for o in items if self._ykey(o) not in have and not have.add(self._ykey(o))]
            added += len(items)
            with _YLOCK:
                with open(self._ypath(cow, day), "a", encoding="utf-8") as f:
                    for o in items:
                        f.write(json.dumps(o) + "\n")
                key = (self.path, cow, day)
                if key in _YCACHE:
                    for o in items:
                        _YCACHE[key][o["minute"]].append(Rec(**o))
        self.add_cows({r["cow_id"] for r in recs})
        return added

    def yolo_minute(self, cow: str, day: int, minute: int) -> list[Rec]:
        return list(self._yshard(cow, day).get(int(minute), []))

    def yolo_cow_range(self, cow: str, tabs_lo: int, tabs_hi: int) -> list[Rec]:
        out = []
        for t in range(tabs_lo, tabs_hi + 1):
            if t >= 0:
                out += self._yshard(cow, t // MIN_PER_DAY + 1).get(t % MIN_PER_DAY, [])
        return out

    def yolo_all_minute(self, day: int, minute: int, exclude_cow: Optional[str] = None) -> list[Rec]:
        out = []
        for cow in self.yolo_cows():
            if cow != exclude_cow:
                out += self._yshard(cow, day).get(int(minute), [])
        return out

    def yolo_cows(self) -> list[str]:
        return sorted({p.stem.rsplit("_d", 1)[0] for p in (self.root / "yolo").glob("*.jsonl")})

    def yolo_shards(self) -> list[tuple[str, int]]:
        return sorted((p.stem.rsplit("_d", 1)[0], int(p.stem.rsplit("_d", 1)[1])) for p in (self.root / "yolo").glob("*.jsonl"))

    def iter_yolo(self) -> Iterable[tuple[str, int, int, list[Rec]]]:
        """(cow, day, minute, records) for every cow-minute that has at least one record (with a behaviour)."""
        for cow, day in self.yolo_shards():
            sh = self._yshard(cow, day)
            for minute in sorted(sh):
                recs = [r for r in sh[minute] if r.behavior is not None]
                if recs:
                    yield cow, day, minute, recs


# ================================================================== history
def row_to_brief(row: dict, prev_label: Optional[str], next_label: Optional[str]) -> Optional[Brief]:
    """Build the agent's input from a minute row. Returns None when upstream has no tier-1 data for the minute."""
    task = row.get("task") or "verify"
    if not row.get("tier1_top_json") and task != "find_missing":
        return None
    zf = {k: row[f"zone_{k}"] for k in ("feed_bunk", "waterer", "resting", "open") if row.get(f"zone_{k}") is not None}
    vis = {c: row[f"vis_{c}"] for c in CAMERAS if row.get(f"vis_{c}") is not None}
    return Brief(cow_id=row["cow_id"], day=row["day"], minute=row["minute"],
                 tier1_top=json.loads(row["tier1_top_json"]) if row.get("tier1_top_json") else [], tier1_entropy=row.get("tier1_entropy") or 0.0,
                 zone_fractions=zf, speed_px_s=row.get("speed_px_s") or 0.0, posture=row.get("posture"),
                 visible_frac=vis, id_conf=row["id_conf"] if row.get("id_conf") is not None else 1.0,
                 is_night=bool(row.get("is_night")), prev_label=prev_label, next_label_tier1=next_label,
                 camera_disagreement=bool(row.get("camera_disagreement")), task=task, src_file=row.get("src_file"),
                 src_camera=row.get("src_camera"), src_offset_s=row.get("src_offset_s"), abs_time=row.get("src_abs_time"),
                 yolo_flags=[f for f in (row.get("yolo_flags") or "").split(";") if f])


TRACKED_FOR_SINCE = ["feeding", "drinking", "lying", "walking", "ruminating_lying", "ruminating_standing"]


class HistoryStore:
    """Read-only view over a cow's recent labelled minutes. Same-cow, short-window context only:
    no Day-1 habit priors and no herd priors, because those would flatten the signal Level 2 looks for.
    Labels prefer the agent's FINAL label; minutes not yet processed fall back to tier-1 and are marked provisional ('?')."""

    def __init__(self, db: Store, cfg: HistoryCfg):
        self.db, self.cfg = db, cfg

    def _window(self, cow_id: str, lo: int, hi: int) -> list[dict]:
        out = []
        for r in self.db.window(cow_id, lo, hi):
            final = r.get("behavior") is not None
            lab = r["behavior"] if final else r.get("tier1_label")
            if lab is not None:
                out.append({"t": r["t_abs"], "label": lab, "final": final, "conf": r["confidence"] if final else r.get("tier1_conf")})
        return out

    def prev_next(self, cow_id: str, day: int, minute: int) -> tuple[Optional[str], Optional[str]]:
        t = t_abs(day, minute)
        prev = self._window(cow_id, t - 1, t - 1)
        nxt = self.db.row_at(cow_id, t + 1)
        return (prev[0]["label"] if prev else None), (nxt.get("tier1_label") if nxt else None)

    @staticmethod
    def _rle(items: list[dict]) -> str:
        runs: list[list] = []
        for it in items:
            tag = it["label"] + ("" if it["final"] else "?")
            if runs and runs[-1][0] == tag and runs[-1][2] == it["t"] - 1:
                runs[-1][1] += 1; runs[-1][2] = it["t"]
            else:
                runs.append([tag, 1, it["t"]])
        return " > ".join(f"{a} x{n}" for a, n, _ in runs)

    def summary(self, cow_id: str, day: int, minute: int, lookback_min: Optional[int] = None) -> dict:
        c = self.cfg
        t = t_abs(day, minute)
        lookback = lookback_min or c.lookback_min
        back = max(lookback, c.bout_lookback_min, c.since_lookback_min)
        rows = self._window(cow_id, t - back, t - 1)
        if not rows:
            return {"available": True, "empty": True, "note": "no earlier labelled minutes for this cow"}
        recent = [r for r in rows if r["t"] >= t - lookback]
        last = rows[-1]
        bout_n, expect = 0, last["t"]
        for r in reversed(rows):
            if r["t"] == expect and r["label"] == last["label"] and bout_n < c.bout_lookback_min:
                bout_n += 1; expect -= 1
            else:
                break
        since = {}
        for b in TRACKED_FOR_SINCE:
            ts = [r["t"] for r in rows if r["label"] == b and r["t"] >= t - c.since_lookback_min]
            since[b] = (t - max(ts)) if ts else None
        counts60 = dict(Counter(r["label"] for r in rows if r["t"] >= t - 60))
        facts: dict[str, Any] = {
            "available": True, "recent_runs": self._rle(recent),
            "current_bout": {"label": last["label"], "minutes": bout_n, "ends_minutes_ago": t - last["t"], "final": last["final"]},
            "minutes_since": since, "counts_last_60": counts60,
            "n_provisional_in_window": sum(not r["final"] for r in recent),
            "note": "context only (same cow, recent minutes); the current minute's images and measurements decide. '?' = provisional tier-1 label"}
        if c.lookahead_min > 0:
            ahead = self.db.window(cow_id, t + 1, t + c.lookahead_min)
            facts["next_tier1"] = [r["tier1_label"] for r in ahead if r.get("tier1_label")]
        return facts
