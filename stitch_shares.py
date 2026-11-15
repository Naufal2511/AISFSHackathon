#!/usr/bin/env python3
"""Join the chunk runs of all three laptops into one 48 h result, then run Level 2 on it.

  # on one machine, at the repo root: unzip naufal_training.zip, afeef_training.zip, nooh_training.zip here
  python stitch_shares.py --videos "/Volumes/T7/all clips"          # videos: for Level 2's risk agent (frames)
  python stitch_shares.py --level2-skip-agent                        # no videos here: behavior-based risk only
  python stitch_shares.py --id-map configs/id_overrides.yaml         # after checking id_map.csv by eye

1. Finds every finished chunk under outputs/shares/<who>/<who>_cNN/.
2. Gives each cow ONE id across chunks. Every chunk numbers its cows itself (COW_01.. by its own identity pass).
   run_share.py makes each chunk also track 2 min before the window it owns, so in that lead-in both chunks have
   boxes on the same frames: overlapping boxes (IoU >= 0.5, same camera and second) vote, Hungarian on the votes.
   Cows the overlap cannot settle fall back to the memory-card prototypes (mean DINOv2 appearance per cow,
   stage1/level1/agent/memory/prototypes.npz; Hungarian on cosine similarity; each cow keeps every prototype it had).
   Black Angus look alike to DINOv2, so those appearance matches are flagged. Weak matches are flagged in id_map.csv;
   `--id-map` overrides any of them: {<chunk name>: {<local id>: <global id>}}.
3. Writes outputs/stitched/:
     final/activity.csv          one row per behavior bout (bouts joined across chunk edges)   <- the deliverable
     final/activity_minute.csv   one row per cow-minute (wall time + day/minute on one clock)
     final/actions.csv           every agent action with its reason
     final/time_budgets.csv      minutes and share per behavior, per cow per day and overall
     id_map.csv                  chunk, local id -> global id, similarity, margin, status
     traces/<chunk>/, id_agent/<chunk>/ (traces + 30-min ID-check sheets; id_check CSVs relabelled, legend.csv)
     runs/<chunk>/               Level 1 contract + assets relabelled to the global ids (Level 2 input)
     level2/                     brd_risk.csv, index.html, cards/ (scripts/run_level2.py over all chunks)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parent
SHARES = REPO / "outputs" / "shares"


# ----------------------------------------------------------------------------------------------- chunks
def find_chunks(root: Path) -> list[dict]:
    out = []
    for man in sorted(root.glob("*/*/pipeline_manifest.json")):
        run = man.parent
        m = json.loads(man.read_text(encoding="utf-8"))
        st = json.loads((run / "pipeline_state.json").read_text(encoding="utf-8")).get("steps", {}) \
            if (run / "pipeline_state.json").exists() else m.get("steps", {})
        if st.get("export", {}).get("status") != "ok":
            print(f"  skip {run.relative_to(REPO)}: export not finished")
            continue
        s = m.get("settings", {})
        if not s.get("start") or not s.get("end"):
            print(f"  WARNING {run.name}: no --start/--end recorded; its whole span is used")
        run_start = pd.Timestamp(s["start"]) if s.get("start") else pd.Timestamp.min
        cj = run / "chunk.json"          # run_share.py: the chunk OWNS [owned); it also tracked an overlap before it
        own = json.loads(cj.read_text(encoding="utf-8"))["owned"] if cj.exists() else [s.get("start"), s.get("end")]
        out.append({"name": run.name, "who": run.parent.name, "dir": run, "run_start": run_start,
                    "start": pd.Timestamp(own[0]) if own[0] else pd.Timestamp.min,
                    "end": pd.Timestamp(own[1]) if own[1] else pd.Timestamp.max})
    out.sort(key=lambda c: c["start"])
    for a, b in zip(out, out[1:]):
        if b["start"] > a["end"]:
            print(f"  NOTE gap between {a['name']} (ends {a['end']}) and {b['name']} (starts {b['start']})")
        if b["start"] < a["end"]:
            print(f"  NOTE {a['name']} and {b['name']} overlap; the earlier chunk keeps the overlap")
    return out


def _protos(chunk: dict) -> dict[str, np.ndarray]:
    p = chunk["dir"] / "stage1" / "level1" / "agent" / "memory" / "prototypes.npz"
    if not p.exists():
        return {}
    z = np.load(p)
    return {k: z[k] / (np.linalg.norm(z[k]) + 1e-9) for k in z.files}


def _boxes(ch: dict, t0, t1) -> pd.DataFrame:
    p = ch["dir"] / "stage1" / "level1" / "contract" / "observations.csv"
    if not p.exists():
        return pd.DataFrame()
    o = pd.read_csv(p, usecols=["wall_time", "camera", "cow_id", "x1", "y1", "x2", "y2"], low_memory=False)
    o = o[o.cow_id.notna()]
    ts = pd.to_datetime(o.wall_time)
    return o[(ts >= t0) & (ts < t1)]


def overlap_votes(prev: dict, ch: dict, prev_map: dict) -> tuple[pd.DataFrame, int]:
    """Both chunks tracked the same frames in ch's lead-in [run_start, start): a box in one is the same cow as the box
    it overlaps (IoU >= 0.5) in the other, same camera, same second. Returns votes (local id x global id) and the
    number of seconds compared."""
    t0, t1 = max(ch["run_start"], prev["start"]), min(ch["start"], prev["end"])
    if t1 <= t0:
        return pd.DataFrame(), 0
    a, b = _boxes(prev, t0, t1), _boxes(ch, t0, t1)
    if a.empty or b.empty:
        return pd.DataFrame(), 0
    a = a.assign(gid=a.cow_id.astype(str).map(prev_map).fillna(a.cow_id.astype(str)))
    j = b.merge(a, on=["wall_time", "camera"], suffixes=("", "_p"))
    ix = (np.minimum(j.x2, j.x2_p) - np.maximum(j.x1, j.x1_p)).clip(lower=0)
    iy = (np.minimum(j.y2, j.y2_p) - np.maximum(j.y1, j.y1_p)).clip(lower=0)
    inter = ix * iy
    union = (j.x2 - j.x1) * (j.y2 - j.y1) + (j.x2_p - j.x1_p) * (j.y2_p - j.y1_p) - inter
    j = j[(inter / union.clip(lower=1)) >= 0.5]
    return j.groupby([j.cow_id.astype(str), j.gid]).size().unstack(fill_value=0), j.wall_time.nunique()


def align_ids(chunks: list[dict], min_sim: float, min_margin: float, overrides: dict, min_votes: int = 10) -> pd.DataFrame:
    """Overlap first (same boxes in the same frames: near-certain), appearance prototypes for whatever is left (cows
    not seen in the overlap, chunks with a gap before them). Black Angus look alike to DINOv2, so appearance alone
    gives small margins; every such match is flagged."""
    from scipy.optimize import linear_sum_assignment
    bank: dict[str, list[np.ndarray]] = {}
    rows = []
    maps: dict[str, dict] = {}

    def new_id(taken=()) -> str:
        n = 1
        while f"COW_{n:02d}" in bank or f"COW_{n:02d}" in taken:
            n += 1
        return f"COW_{n:02d}"

    for ch in chunks:
        pr = _protos(ch)
        ov = {str(k): str(v) for k, v in (overrides.get(ch["name"]) or {}).items()}
        if not pr:
            print(f"  WARNING {ch['name']}: no memory-card prototypes; unmatched ids are kept as they are")
        gids = sorted(bank)
        first = not gids
        assign: dict[str, tuple] = {}
        by_overlap: dict[str, tuple] = {}
        prev = next((p for p in reversed(chunks[:chunks.index(ch)]) if p["end"] > ch["run_start"]), None)
        if prev is not None and ch["run_start"] < ch["start"]:
            V, nsec = overlap_votes(prev, ch, maps.get(prev["name"], {}))
            V = V.drop(index=[l for l in V.index if l in ov], errors="ignore")
            if len(V):
                ri, ci = linear_sum_assignment(-V.values)
                for i, k in zip(ri, ci):
                    v, tot = int(V.values[i, k]), int(V.values[i].sum())
                    if v >= min_votes and v >= 0.6 * tot:
                        by_overlap[V.index[i]] = (V.columns[k], v, tot)
            print(f"  {ch['name']}: {len(by_overlap)} cows matched by box overlap with {prev['name']} ({nsec} s compared)")
        local = sorted(set(pr) | set(by_overlap) | set(ov)) or sorted(_local_ids(ch))
        todo = [l for l in local if l not in ov and l in pr and l not in by_overlap]
        used = {g for g, _, _ in by_overlap.values()} | set(ov.values())
        gids_left = [g for g in gids if g not in used]
        if todo and gids_left:
            gids = gids_left
            S = np.array([[max(float(pr[l] @ q) for q in bank[g] if q.shape == pr[l].shape) if any(q.shape == pr[l].shape for q in bank[g]) else -1.0
                           for g in gids] for l in todo])
            ri, ci = linear_sum_assignment(-S)
            for i, j in zip(ri, ci):
                second = np.sort(S[i])[-2] if S.shape[1] > 1 else -1.0
                assign[todo[i]] = (gids[j], float(S[i, j]), float(S[i, j] - second))
        taken = set(ov.values()) | {g for g, _, _ in by_overlap.values()}
        for l in local:
            if l in ov:
                g, sim, mar, status = ov[l], None, None, "override"
            elif l in by_overlap:
                g, v, tot = by_overlap[l]
                sim, mar, status = round(v / max(tot, 1), 3), None, f"overlap ({v} boxes)"
            elif not pr:
                g, sim, mar, status = (l if l not in taken else new_id(taken)), None, None, "no_cards"
            elif l in assign and assign[l][1] >= min_sim and assign[l][0] not in taken:
                g, sim, mar = assign[l]
                status = "matched" if mar >= min_margin else "low_margin"
            else:
                g = l if (first and l not in taken) else new_id(taken)
                sim, mar = (assign[l][1], assign[l][2]) if l in assign else (None, None)
                status = "first_chunk" if first else "new"
            taken.add(g)
            bank.setdefault(g, [])
            if l in pr:
                bank[g].append(pr[l])
            maps.setdefault(ch["name"], {})[l] = g
            rows.append({"chunk": ch["name"], "who": ch["who"], "start": ch["start"], "local_id": l, "global_id": g,
                         "similarity": None if sim is None else round(sim, 3),
                         "margin": None if mar is None else round(mar, 3), "status": status})
    df = pd.DataFrame(rows)
    n = df.global_id.nunique() if len(df) else 0
    if n > 10:
        print(f"  WARNING {n} global ids for a pen of 10: check the 'new' / 'low_margin' rows of id_map.csv "
              f"and merge them with --id-map")
    return df


def _local_ids(ch: dict) -> set:
    p = ch["dir"] / "stage1" / "level1" / "final" / "activity.csv"
    return set(pd.read_csv(p, usecols=["cow_id"]).cow_id.dropna().astype(str)) if p.exists() else set()


# ----------------------------------------------------------------------------------------------- tables
def _in_window(ts: pd.Series, ch: dict) -> pd.Series:
    return (ts >= ch["start"]) & (ts < ch["end"])


def _day_minute(ts: pd.Series, origin: pd.Timestamp) -> tuple[pd.Series, pd.Series]:
    d = (ts.dt.normalize() - origin).dt.days + 1
    return d.astype(int), (ts.dt.hour * 60 + ts.dt.minute).astype(int)


def stitch_minutes(chunks, idmap, origin) -> pd.DataFrame:
    parts = []
    for ch in chunks:
        p = ch["dir"] / "stage1" / "level1" / "final" / "activity.csv"
        if not p.exists():
            print(f"  WARNING {ch['name']}: {p.relative_to(REPO)} missing (rerun its export step)")
            continue
        df = pd.read_csv(p, low_memory=False)
        ts = pd.to_datetime(df.minute_start)
        df = df[_in_window(ts, ch)].copy()
        df["cow_id"] = df.cow_id.astype(str).map(idmap[ch["name"]]).fillna(df.cow_id)
        df["chunk"] = ch["name"]
        parts.append(df)
    if not parts:
        raise SystemExit("no chunk has verified minutes")
    m = pd.concat(parts, ignore_index=True)
    ts = pd.to_datetime(m.minute_start)
    m["day"], m["minute"] = _day_minute(ts, origin)
    m = m.sort_values(["cow_id", "minute_start", "chunk"]).drop_duplicates(["cow_id", "minute_start"], keep="first")
    first = ["cow_id", "minute_start", "day", "minute", "behavior", "confidence"]
    return m[first + [c for c in m.columns if c not in first]].reset_index(drop=True)


def bouts(m: pd.DataFrame) -> pd.DataFrame:
    """Same rule as brd_agent Store.rebuild_bouts (same cow, day and behavior, consecutive minutes), on the joined
    minutes, so a bout that crosses a chunk edge is one bout."""
    rows, cur = [], None
    for r in m.sort_values(["cow_id", "day", "minute"]).itertuples(index=False):
        if cur and (r.cow_id, r.day, r.behavior) == (cur["cow_id"], cur["day"], cur["behavior"]) and r.minute == cur["end_minute"] + 1:
            cur["end_minute"] = r.minute
            cur["_c"].append(r.confidence if pd.notna(r.confidence) else 0.0)
            cur["n_agent_minutes"] += int(getattr(r, "tier", 0) == 2)
            continue
        if cur:
            rows.append(cur)
        cur = {"cow_id": r.cow_id, "day": r.day, "behavior": r.behavior, "start_minute": r.minute, "end_minute": r.minute,
               "_c": [r.confidence if pd.notna(r.confidence) else 0.0], "n_agent_minutes": int(getattr(r, "tier", 0) == 2)}
    if cur:
        rows.append(cur)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["duration_min"] = out.end_minute - out.start_minute + 1
    out["mean_confidence"] = [round(float(np.mean(c)), 4) for c in out._c]
    return out[["cow_id", "day", "behavior", "start_minute", "end_minute", "duration_min", "mean_confidence", "n_agent_minutes"]]


def time_budgets(m: pd.DataFrame) -> pd.DataFrame:
    g = m.groupby(["cow_id", "day", "behavior"]).size().rename("minutes").reset_index()
    tot = m.groupby(["cow_id", "behavior"]).size().rename("minutes").reset_index().assign(day="all")
    out = pd.concat([g, tot], ignore_index=True)
    out["share"] = (out.minutes / out.groupby(["cow_id", "day"]).minutes.transform("sum")).round(4)
    out["hours"] = (out.minutes / 60).round(2)
    return out.sort_values(["cow_id", "day", "minutes"], ascending=[True, True, False]).reset_index(drop=True)


def stitch_actions(chunks, idmap, origin) -> pd.DataFrame:
    parts = []
    for ch in chunks:
        p, meta = ch["dir"] / "final" / "actions.csv", ch["dir"] / "stage2" / "store" / "meta.json"
        if not p.exists() or not meta.exists():
            continue
        org = json.loads(meta.read_text(encoding="utf-8")).get("clock_origin")
        df = pd.read_csv(p, low_memory=False)
        if df.empty or not org:
            continue
        ts = pd.Timestamp(org) + pd.to_timedelta((df.day - 1) * 1440 + df.minute, unit="min")
        df = df[_in_window(ts, ch)].copy()
        ts = ts[df.index]
        df["minute_start"] = ts.dt.strftime("%Y-%m-%dT%H:%M:%S")
        df["day"], df["minute"] = _day_minute(ts, origin)
        df["cow_id"] = df.cow_id.astype(str).map(idmap[ch["name"]]).fillna(df.cow_id)
        df["chunk"] = ch["name"]
        parts.append(df)
    return pd.concat(parts, ignore_index=True).sort_values(["cow_id", "minute_start", "chunk", "run_id", "seq"]) if parts else pd.DataFrame()


# ----------------------------------------------------------------------------------------------- level 2 inputs
def relabel_run(ch: dict, cmap: dict, out: Path) -> Path:
    """stitched/runs/<chunk>/: the chunk's Level 1 contract, assets and verified minutes on the global ids,
    clipped to the chunk's window (Pen6Run / src/risk read exactly these files)."""
    src = ch["dir"] / "stage1" / "level1"
    dst = out / ch["name"]
    (dst / "contract").mkdir(parents=True, exist_ok=True)
    for name in ("observations.csv", "cow_state.csv"):
        df = pd.read_csv(src / "contract" / name, low_memory=False)
        df = df[_in_window(pd.to_datetime(df.wall_time), ch)].copy()
        df["cow_id"] = df.cow_id.map(lambda v: cmap.get(str(v), v) if pd.notna(v) else v)
        df.to_csv(dst / "contract" / name, index=False)
    rr = src / "identity" / "tracking_records_resolved.csv"      # frame lookups (FrameStore) by Level 2's tools
    if rr.exists():
        (dst / "identity").mkdir(exist_ok=True)
        df = pd.read_csv(rr, low_memory=False)
        if "global_cow_id" in df:
            df["global_cow_id"] = df.global_cow_id.map(lambda v: cmap.get(str(v), v) if pd.notna(v) else v)
        df.to_csv(dst / "identity" / rr.name, index=False)
    for f in (src / "contract").glob("*.json"):
        shutil.copy2(f, dst / "contract" / f.name)
    act = src / "final" / "activity.csv"
    if act.exists():
        df = pd.read_csv(act, low_memory=False)
        df = df[_in_window(pd.to_datetime(df.minute_start), ch)].copy()
        df["cow_id"] = df.cow_id.astype(str).map(cmap).fillna(df.cow_id)
        (dst / "final").mkdir(exist_ok=True)
        df.to_csv(dst / "final" / "activity.csv", index=False)
    if (src / "assets").is_dir():
        (dst / "assets").mkdir(exist_ok=True)
        for f in (src / "assets").glob("*.csv"):
            try:
                df = pd.read_csv(f, low_memory=False)
            except pd.errors.EmptyDataError:
                continue
            if "cow_id" in df:
                df["cow_id"] = df.cow_id.map(lambda v: cmap.get(str(v), v) if pd.notna(v) else v)
            for col in ("image", "file"):          # load_assets joins these to the assets dir: point at the originals
                if col in df and f.stem in ("breathing", "clips"):
                    df[col] = [str((src / "assets" / v).resolve()) if isinstance(v, str) else v for v in df[col]]
            df.to_csv(dst / "assets" / f.name, index=False)
    return dst


def copy_traces(ch: dict, cmap: dict, out: Path) -> None:
    t = ch["dir"] / "stage2" / "traces"
    if t.is_dir():
        shutil.copytree(t, out / "traces" / ch["name"], dirs_exist_ok=True)
    ag = ch["dir"] / "stage1" / "level1" / "agent"
    d = out / "id_agent" / ch["name"]
    for sub in ("traces", "id_checks"):
        if (ag / sub).is_dir():
            shutil.copytree(ag / sub, d / sub, dirs_exist_ok=True)
    ic = d / "id_checks"
    if ic.is_dir():
        for f in ic.glob("*.csv"):        # sheets are images (local ids drawn in); their CSVs get the global ids
            df = pd.read_csv(f)
            for c in [c for c in df.columns if c == "cow_id" or c.endswith("_best")]:
                df[c] = df[c].map(lambda v: cmap.get(str(v), v) if pd.notna(v) else v)
            df.to_csv(f, index=False)
        pd.DataFrame([{"local_id_on_sheet": k, "global_id": v} for k, v in sorted(cmap.items())]).to_csv(ic / "legend.csv", index=False)


# ----------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shares", default=str(SHARES), help="folder holding <who>/<who>_cNN/ chunk runs")
    ap.add_argument("--out", default="outputs/stitched")
    ap.add_argument("--videos", help="all clips (Level 2's risk agent reads frames); without it Level 2 runs behavior-only")
    ap.add_argument("--id-map", help="YAML overrides {chunk: {local_id: global_id}}")
    ap.add_argument("--min-sim", type=float, default=0.55, help="weaker prototype matches start a new id (flagged)")
    ap.add_argument("--min-margin", type=float, default=0.05, help="best minus second-best similarity below this is flagged")
    ap.add_argument("--no-level2", action="store_true")
    ap.add_argument("--level2-skip-agent", action="store_true")
    ap.add_argument("--level2-args", nargs=argparse.REMAINDER, default=[],
                    help="passed to scripts/run_level2.py, e.g. --level2-args --min-hours 0.1 (keep last)")
    a = ap.parse_args(argv)
    os.chdir(REPO)
    out = Path(a.out)
    (out / "final").mkdir(parents=True, exist_ok=True)

    chunks = find_chunks(Path(a.shares))
    if not chunks:
        raise SystemExit(f"no finished chunk under {a.shares} (unzip the laptops' <who>.zip at the repo root)")
    print(f"{len(chunks)} chunks: " + ", ".join(f"{c['name']} ({c['start']:%m-%d %H:%M}-{c['end']:%H:%M})" for c in chunks))

    overrides = yaml.safe_load(open(a.id_map)) if a.id_map else {}
    idm = align_ids(chunks, a.min_sim, a.min_margin, overrides or {})
    idm.to_csv(out / "id_map.csv", index=False)
    idmap = {c: dict(zip(g.local_id, g.global_id)) for c, g in idm.groupby("chunk")}
    for c in chunks:
        idmap.setdefault(c["name"], {})
    flagged = idm[idm.status.isin(["new", "low_margin"]) & (idm.chunk != chunks[0]["name"])]
    print(f"ids: {idm.global_id.nunique()} cows; {len(flagged)} weak/new matches to check in {out / 'id_map.csv'}")
    if len(flagged):
        print(flagged[["chunk", "local_id", "global_id", "similarity", "margin", "status"]].to_string(index=False))

    origin = min(c["start"] for c in chunks if c["start"] != pd.Timestamp.min).normalize() \
        if any(c["start"] != pd.Timestamp.min for c in chunks) else None
    if origin is None:
        raise SystemExit("chunks have no recorded windows")
    m = stitch_minutes(chunks, idmap, origin)
    m.to_csv(out / "final" / "activity_minute.csv", index=False)
    b = bouts(m)
    b.to_csv(out / "final" / "activity.csv", index=False)
    time_budgets(m).to_csv(out / "final" / "time_budgets.csv", index=False)
    acts = stitch_actions(chunks, idmap, origin)
    acts.to_csv(out / "final" / "actions.csv", index=False)
    print(f"final: {len(m)} cow-minutes, {len(b)} bouts, {len(acts)} actions, "
          f"{m.minute_start.min()} .. {m.minute_start.max()} (day 1 = {origin:%Y-%m-%d})")
    cover = m.groupby("cow_id").size().div(60).round(1)
    print("hours per cow:", cover.to_dict())

    runs = []
    if (out / "runs").exists():
        shutil.rmtree(out / "runs")
    for ch in chunks:
        runs.append(relabel_run(ch, idmap[ch["name"]], out / "runs"))
        copy_traces(ch, idmap[ch["name"]], out)
    json.dump({"chunks": [{k: str(v) for k, v in c.items()} for c in chunks], "day1": str(origin.date()),
               "cows": sorted(m.cow_id.unique().tolist())}, open(out / "stitch_manifest.json", "w"), indent=1)

    if a.no_level2:
        return 0
    skip_agent = a.level2_skip_agent or not a.videos
    cmd = [sys.executable, "scripts/run_level2.py", "--runs", *map(str, runs), "--out", str(out / "level2")]
    if a.videos:
        cmd += ["--video-root", str(Path(a.videos).expanduser().resolve())]
    if skip_agent:
        cmd.append("--skip-agent")
        if not a.videos:
            print("no --videos: Level 2 runs behavior-only (the risk agent needs frames)")
    cmd += list(a.level2_args)
    print("level2:", " ".join(cmd[1:3]), "...", flush=True)
    rc = subprocess.run(cmd, cwd=REPO).returncode
    print(f"{'done' if rc == 0 else 'Level 2 FAILED'} -> {out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
