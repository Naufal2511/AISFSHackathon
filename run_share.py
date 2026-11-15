#!/usr/bin/env python3
"""Split the 48 h of footage over the team's 3 laptops and run this laptop's share, several chunks in parallel.

  python run_share.py                       # interactive: pick who you are, the video folder, how many chunks at once
  python run_share.py --who naufal_training --videos "/Volumes/T7/Day 1+2" --parallel 2
  python run_share.py --plan-only --videos <all clips>     # just write outputs/shares/plan.json (send it to the others)
  python run_share.py --who afeef_training --pack          # zip this laptop's results for the stitcher

How it splits:
  * plan.json cuts the whole recording into 3 shares with the same amount of FOOTAGE (camera-minutes, so a gap in the
    recording does not leave one laptop idle), on whole-minute boundaries. The plan depends only on the clip names,
    so every laptop that sees the full folder computes the same plan; a laptop that only has its own clips uses the
    plan.json someone else sent (put it in outputs/shares/).
  * each chunk also tracks `--overlap-min` (2) minutes before the window it owns; stitch_shares.py joins the cow ids
    of neighbouring chunks (also across laptops) on the boxes both saw in that overlap.
  * `--chunk N` runs one chunk only (one task of a SLURM array job on HPRC).
  * each share is cut into chunks of ~`--chunk-hours`; `--parallel` of them run at the same time, each as its own
    `run_pipeline.py` (stage 1 -> identity agent -> assets -> stage 2 -> export), with the CPU threads divided between
    them. Level 2 is not run per chunk: it needs all 48 h together (stitch_shares.py).
  * every chunk is resumable: run the same command again after a crash and finished steps are skipped.

Output: outputs/shares/<who>/<who>_cNN/ (one pipeline run per chunk) + outputs/shares/<who>/share_status.json.
Then: `--pack` -> outputs/shares/<who>.zip; unzip all three at the repo root of one machine; `python stitch_shares.py`.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from pipeline import cameras as camlib  # noqa: E402

PEOPLE = ["naufal_training", "afeef_training", "nooh_training"]
SHARES = REPO / "outputs" / "shares"
PLAN = SHARES / "plan.json"
FMT = "%Y-%m-%d %H:%M"


# ----------------------------------------------------------------------------------------------- plan
def _intervals(clips) -> list[tuple[datetime, datetime]]:
    """Per camera, a clip runs until the next clip of that camera starts (capped at 15 min); the last clip gets the
    camera's median clip length. File names only: no ffprobe, so every laptop computes the same plan."""
    out = []
    for cam in camlib.CAMERAS:
        st = sorted(datetime.fromisoformat(c.start) for c in clips if c.camera == cam)
        if not st:
            continue
        gaps = [(b - a).total_seconds() for a, b in zip(st, st[1:]) if 0 < (b - a).total_seconds() <= 900]
        typ = sorted(gaps)[len(gaps) // 2] if gaps else 300.0
        for i, a in enumerate(st):
            nxt = st[i + 1] if i + 1 < len(st) else a + timedelta(seconds=typ)
            out.append((a, min(nxt, a + timedelta(seconds=900))))
    return out


HPRC_SHARE = "hprc_day2"
LAYOUTS = {
    "day1-laptops-day2-hprc": "Day 1 split by footage over the 3 laptops; Day 2 (and anything later) on HPRC",
    "all-laptops": "the whole recording split by footage over the 3 laptops",
}


def make_plan(videos: str, layout: str = "day1-laptops-day2-hprc") -> dict:
    """videos: the folder with ALL clips, or a .txt list of their paths (e.g. `rclone lsf -R gdrive:Pen6 > all.txt`)."""
    cfg = yaml.safe_load((REPO / "configs" / "pipeline.yaml").read_text(encoding="utf-8"))
    clips = camlib.assign_cameras(videos, cfg["cameras"])
    iv = _intervals(clips)
    t0 = min(a for a, _ in iv).replace(second=0, microsecond=0)
    t1 = max(b for _, b in iv)
    t1 = (t1 + timedelta(minutes=1)).replace(second=0, microsecond=0) if t1.second or t1.microsecond else t1
    n_min = int((t1 - t0).total_seconds() // 60)
    cover = [0.0] * n_min                                    # camera-seconds of footage in each minute
    for a, b in iv:
        i = int((a - t0).total_seconds() // 60)
        while i < n_min:
            m0 = t0 + timedelta(minutes=i)
            ov = (min(b, m0 + timedelta(minutes=1)) - max(a, m0)).total_seconds()
            if ov <= 0:
                break
            cover[i] += ov
            i += 1

    def split(lo: int, hi: int, names: list[str]) -> dict:
        """minutes [lo, hi) into len(names) parts with the same amount of footage, on whole minutes"""
        total = sum(cover[lo:hi])
        cuts, acc, k = [lo], 0.0, 1
        for i in range(lo, hi):
            acc += cover[i]
            while k < len(names) and acc >= total * k / len(names):
                cuts.append(i + 1)
                k += 1
        cuts += [hi] * (len(names) + 1 - len(cuts))
        out = {}
        for j, who in enumerate(names):
            s, e = t0 + timedelta(minutes=cuts[j]), t0 + timedelta(minutes=cuts[j + 1])
            out[who] = {"start": s.strftime(FMT), "end": e.strftime(FMT),
                        "footage_h": round(sum(cover[cuts[j]:cuts[j + 1]]) / 3600, 2),
                        "clips": sorted(c.name for c in camlib.clips_in_window(clips, s.isoformat(), e.isoformat()))}
        return out

    if layout == "all-laptops":
        shares = split(0, n_min, PEOPLE)
    elif layout == "day1-laptops-day2-hprc":
        midnight = int(((t0.replace(hour=0, minute=0) + timedelta(days=1)) - t0).total_seconds() // 60)
        if midnight >= n_min:
            raise SystemExit("the clips cover only one day: use --layout all-laptops")
        shares = {**split(0, midnight, PEOPLE), **split(midnight, n_min, [HPRC_SHARE])}
    else:
        raise SystemExit(f"unknown layout {layout}: {', '.join(LAYOUTS)}")
    return {"created": datetime.now().isoformat(timespec="seconds"), "videos_seen": str(Path(videos).resolve()),
            "layout": layout, "n_clips": len(clips), "span": [t0.strftime(FMT), t1.strftime(FMT)],
            "footage_h_total": round(sum(cover) / 3600, 2), "shares": shares}


def clip_start(name: str) -> datetime | None:
    m = camlib.EVENT_RE.search(name)
    return datetime.strptime(m.group(1), "%Y%m%d%H%M%S") if m else None


def missing_for_window(plan: dict, have: set[str], start: str, end: str) -> list[str]:
    """Clips of the plan (made from the FULL footage) that a window needs, incl. the one running into its start,
    that are not in this laptop's video folder."""
    s, e = datetime.strptime(start, FMT), datetime.strptime(end, FMT)
    need = [n for sh in plan["shares"].values() for n in sh["clips"]
            if (t := clip_start(n)) is not None and s - timedelta(seconds=900) <= t < e]
    return sorted({n for n in need if n.lower() not in have})


def chunks_of(share: dict, chunk_hours: float, parallel: int) -> list[tuple[str, str]]:
    s, e = datetime.strptime(share["start"], FMT), datetime.strptime(share["end"], FMT)
    mins = int((e - s).total_seconds() // 60)
    n = max(parallel, math.ceil(mins / (chunk_hours * 60)))
    edges = [s + timedelta(minutes=round(mins * i / n)) for i in range(n + 1)]
    return [(a.strftime(FMT), b.strftime(FMT)) for a, b in zip(edges, edges[1:]) if b > a]


# ----------------------------------------------------------------------------------------------- hardware
def hardware() -> dict:
    cores = os.cpu_count() or 4
    try:
        ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except (ValueError, OSError, AttributeError):
        ram_gb = 16.0
    dev = "cpu"
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    except Exception:
        pass
    # each chunk holds YOLO + DINOv2 + 3 decoded 4K frames (~3-4 GB); one GPU is shared well by 2 processes
    by_ram = max(1, int(ram_gb // 4))
    by_dev = 2 if dev in ("cuda", "mps") else max(1, cores // 4)
    return {"cores": cores, "ram_gb": round(ram_gb, 1), "device": dev, "suggested_parallel": max(1, min(4, by_ram, by_dev))}


# ----------------------------------------------------------------------------------------------- run
def lead_in(start: str, span_start: str, minutes: float) -> str:
    """A chunk also tracks `minutes` before the window it owns (not before the recording starts): in that overlap
    it and the previous chunk see the same boxes in the same frames, which is how stitch_shares.py joins cow ids."""
    s = datetime.strptime(start, FMT)
    return max(s - timedelta(minutes=minutes), datetime.strptime(span_start, FMT)).strftime(FMT)


def run_chunk(i: int, who: str, start: str, end: str, a, threads: int, workers: int, lock, status: dict) -> int:
    name = f"{who}_c{i + 1:02d}"
    root = SHARES / who
    log = root / f"{name}.log"
    run_start = lead_in(start, a.span_start, a.overlap_min)
    (root / name).mkdir(parents=True, exist_ok=True)
    (root / name / "chunk.json").write_text(json.dumps({"owned": [start, end], "run": [run_start, end],
                                                        "overlap_min": a.overlap_min, "who": who}, indent=1), encoding="utf-8")
    cmd = [sys.executable, "run_pipeline.py", "--videos", a.videos, "--name", name, "--out-root", str(root),
           "--start", run_start, "--end", end, "--until", "export", "--workers", str(workers), "--quiet"]
    if a.ffmpeg_dir:
        cmd += ["--ffmpeg-dir", a.ffmpeg_dir]
    if a.vlm:
        cmd += ["--vlm", a.vlm]
    if a.no_id_agent:
        cmd.append("--no-id-agent")
    env = {**os.environ, "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads),
           "OPENBLAS_NUM_THREADS": str(threads), "VECLIB_MAXIMUM_THREADS": str(threads), "PYTHONUNBUFFERED": "1"}
    with lock:
        status[name] = {"window": [start, end], "status": "running", "started": datetime.now().isoformat(timespec="seconds")}
        _save_status(who, status)
    print(f"  [{name}] {start} -> {end} started (log {log.relative_to(REPO)})", flush=True)
    t = time.time()
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n===== {datetime.now().isoformat(timespec='seconds')}  {' '.join(cmd)}\n")
        f.flush()
        rc = subprocess.run(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    with lock:
        status[name].update(status="ok" if rc == 0 else "failed", seconds=round(time.time() - t),
                            ended=datetime.now().isoformat(timespec="seconds"), run=str((root / name).relative_to(REPO)))
        _save_status(who, status)
    print(f"  [{name}] {'done' if rc == 0 else 'FAILED (exit %d), see the log; rerun to resume' % rc} "
          f"in {(time.time() - t) / 60:.0f} min", flush=True)
    return rc


def _save_status(who: str, status: dict) -> None:
    p = SHARES / who / "share_status.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(status, indent=1), encoding="utf-8")


def pack(who: str) -> Path:
    """Zip the results (not the clip links, not the remux cache) with repo-relative paths, so unzipping at the
    stitcher's repo root puts every file where the CSVs point."""
    root = SHARES / who
    out = SHARES / f"{who}.zip"
    skip = {"input", "cache"}
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for p in root.rglob("*"):
            rel = p.relative_to(root).parts
            if p.is_dir() or (len(rel) > 1 and rel[1] in skip) or p.suffix.lower() in (".avi", ".mp4") and "clips" not in rel:
                continue
            z.write(p, p.relative_to(REPO))
            n += 1
        if PLAN.exists():
            z.write(PLAN, PLAN.relative_to(REPO))
    print(f"packed {n} files -> {out} ({out.stat().st_size / 2 ** 20:.0f} MB)")
    return out


def smoke_test(a, share: dict, minutes: int = 5) -> dict:
    """5 minutes from the start of this share through the WHOLE pipeline (tracking, identity agent, assets, BRD agent,
    export), then checks the outputs. Separate folder (outputs/shares/_smoke/), never stitched."""
    root = SHARES / "_smoke"
    name = f"{a.who}_smoke"
    run = root / name
    res_p = run / "smoke_result.json"
    if res_p.exists():
        prev = json.loads(res_p.read_text(encoding="utf-8"))
        if prev.get("ok"):
            print(f"\nsmoke test already passed on {prev.get('time')} ({prev.get('minutes_per_footage_minute')} min per minute "
                  f"of footage); delete {run.relative_to(REPO)} to redo it")
            return prev
    s = datetime.strptime(share["start"], FMT)
    e = s + timedelta(minutes=minutes)
    if getattr(a, "plan", None) is not None and getattr(a, "have", None) is not None:
        miss = missing_for_window(a.plan, a.have, s.strftime(FMT), e.strftime(FMT))
        if miss:
            print(f"\nSTOP: the smoke-test window needs {len(miss)} clip(s) that are not in {a.videos}: {', '.join(miss[:4])}")
            return {"ok": False}
    print(f"\n== smoke test: {minutes} minutes of {a.who}'s share ({s:%Y-%m-%d %H:%M} -> {e:%H:%M}), full pipeline, real TAMU")
    print("   takes about 10-20 minutes; every step prints when it finishes")
    cmd = [sys.executable, "run_pipeline.py", "--videos", a.videos, "--name", name, "--out-root", str(root),
           "--start", s.strftime(FMT), "--end", e.strftime(FMT), "--until", "export", "--quiet", "--from", "prepare"]
    if a.ffmpeg_dir:
        cmd += ["--ffmpeg-dir", a.ffmpeg_dir]
    if a.vlm:
        cmd += ["--vlm", a.vlm]
    t = time.time()
    p = subprocess.run(cmd, cwd=REPO)
    took = time.time() - t
    checks, ok = [], p.returncode == 0
    def chk(cond, text, fix=""):
        nonlocal ok
        checks.append({"ok": bool(cond), "check": text, "fix": fix})
        ok = ok and bool(cond)
        print(f"   {'✔' if cond else '✖'} {text}" + ("" if cond or not fix else f"\n       FIX: {fix}"))
    print("\n== smoke test results")
    chk(p.returncode == 0, f"pipeline finished (exit {p.returncode})", f"read the newest log in {run / 'logs'}")
    fin = run / "final"
    import pandas as pd
    if (fin / "activity.csv").exists():
        b = pd.read_csv(fin / "activity.csv")
        m = pd.read_csv(fin / "activity_minute.csv")
        win = m[(pd.to_datetime(m.minute_start) >= s) & (pd.to_datetime(m.minute_start) < e)]
        cows = int(win[win.basis != "not_visible"].cow_id.nunique())
        chk(cows >= 5, f"cows identified and labelled: {cows} (pen of 10)",
            "too few cows: check this window's clips cover all 3 cameras; tell the team before running the share")
        rep_p = run / "stage1" / "level1" / "identity" / "identity_report.json"
        if rep_p.exists():
            share_id = float(json.loads(rep_p.read_text(encoding="utf-8")).get("assigned_box_share") or 0)
            chk(share_id >= 0.6, f"boxes given a cow ID: {share_id:.0%}",
                "identity is failing on this laptop: stop and tell the team (do not run the share)")
        seen = win.basis.value_counts().to_dict()
        chk(sum(v for k, v in seen.items() if k != "not_visible") > 0, f"labelled cow-minutes by basis: {seen}")
        chk(set(b.columns) >= {"cow_id", "start_time", "end_time", "behavior", "basis"}, f"activity.csv: {len(b)} bouts, brief format")
        chk((fin / "time_budgets.csv").exists(), "time_budgets.csv written")
    else:
        chk(False, "final/activity.csv written", f"read {run / 'logs'}")
    idsum = run / "stage1" / "level1" / "agent" / "agent_summary.json"
    if idsum.exists():
        j = json.loads(idsum.read_text(encoding="utf-8"))
        chk("tamu" in str(j.get("backend", "")).lower() or "chat-api" in str(j.get("backend", "")),
            f"identity agent on TAMU: {j.get('investigations')} cases, {j.get('vlm_calls')} calls", "TAMU_API_KEY in .env")
    s2 = (run / "logs" / "4_agent.log")
    if s2.exists():
        txt = s2.read_text(encoding="utf-8", errors="replace")
        n_err = txt.count("error_fallback")
        chk("gpt" in txt or "protected." in txt or a.vlm == "mock", "BRD agent called TAMU", "TAMU_API_KEY in .env; delete brd_agent/.env")
        chk(n_err == 0, f"BRD agent errors: {n_err}", f"see {s2}")
    per_min = took / 60 / minutes
    share_min = (datetime.strptime(share["end"], FMT) - datetime.strptime(share["start"], FMT)).total_seconds() / 60
    res = {"ok": ok, "time": datetime.now().isoformat(timespec="seconds"), "seconds": round(took),
           "minutes_per_footage_minute": round(per_min, 2), "checks": checks}
    run.mkdir(parents=True, exist_ok=True)
    res_p.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\n   took {took / 60:.1f} min for {minutes} min of footage ({per_min:.2f} min per minute).")
    print(f"   rough estimate for the whole share ({share_min / 60:.1f} h of wall clock): "
          f"{per_min * share_min / 60:.1f} h with 1 chunk at a time, ~{per_min * share_min / 60 / 1.4:.1f} h with 2 "
          f"(long chunks amortise start-up, so usually less)")
    print(f"   {'SMOKE TEST PASSED' if ok else 'SMOKE TEST FAILED: fix the lines marked ✖ and run again'}")
    print(f"   look at it: python scripts/run_viewer.py --run {run.relative_to(REPO)}   (then open {run.relative_to(REPO)}/viewer.html)")
    return res


def _ask(prompt: str, default: str | None = None) -> str:
    v = input(f"{prompt}{f' [{default}]' if default else ''}: ").strip()
    return v or (default or "")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--who", choices=PEOPLE + [HPRC_SHARE])
    ap.add_argument("--videos", help="folder with the Event*.Avi clips (all 48 h, or at least this laptop's share)")
    ap.add_argument("--parallel", type=int, help="chunks running at the same time (default: from this laptop's cores/RAM/GPU)")
    ap.add_argument("--chunk-hours", type=float, default=4.0, help="target chunk length (shorter = finer resume, more ID joins)")
    ap.add_argument("--overlap-min", type=float, default=2.0, help="minutes each chunk also tracks before its window (ID join)")
    ap.add_argument("--chunk", type=int, help="run only chunk N (1-based), e.g. one task of a SLURM array job")
    ap.add_argument("--ffmpeg-dir")
    ap.add_argument("--vlm", choices=["auto", "mock"])
    ap.add_argument("--no-id-agent", action="store_true")
    ap.add_argument("--replan", action="store_true", help="recompute plan.json from --videos (only before anyone started)")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--layout", choices=list(LAYOUTS), default="day1-laptops-day2-hprc", help="how plan.json splits the work")
    ap.add_argument("--pack", action="store_true", help="zip this laptop's results for stitch_shares.py and exit")
    ap.add_argument("--dry-run", action="store_true", help="print the chunks and exit")
    ap.add_argument("--yes", action="store_true", help="no confirmation prompt")
    ap.add_argument("--skip-checks", action="store_true", help="do not run check_setup.py first (not recommended)")
    ap.add_argument("--allow-missing", action="store_true", help="run chunks even if some of their clips are missing")
    ap.add_argument("--stage2-workers", type=int, help="BRD-agent threads per chunk (default 8 / parallel); HPRC: keep "
                    "(chunks at once x workers) under ~30 so TAMU is not flooded")
    ap.add_argument("--no-smoke", action="store_true", help="skip the 5-minute smoke test (only after it passed once)")
    ap.add_argument("--smoke-only", action="store_true", help="run the checks and the 5-minute smoke test, then stop")
    a = ap.parse_args(argv)
    os.chdir(REPO)
    interactive = sys.stdin.isatty() and not a.yes

    if a.who is None and not a.plan_only:
        if not interactive:
            raise SystemExit("--who is required when not interactive")
        print("Who is running this laptop?")
        for i, p in enumerate(PEOPLE, 1):
            print(f"  {i}) {p}")
        while a.who is None:
            v = _ask("choose 1-3")
            a.who = PEOPLE[int(v) - 1] if v in ("1", "2", "3") else (v if v in PEOPLE else None)
    if a.pack:
        pack(a.who)
        return 0

    plan = json.loads(PLAN.read_text(encoding="utf-8")) if PLAN.exists() and not a.replan else None
    if a.videos is None and interactive:
        a.videos = _ask("video folder (Event*.Avi, Cam 0N subfolders are fine)", os.environ.get("PEN6_VIDEOS"))
    if not a.videos:
        raise SystemExit("give --videos")
    a.videos = str(Path(a.videos).expanduser().resolve())
    if plan is None:
        plan = make_plan(a.videos, a.layout)
        PLAN.parent.mkdir(parents=True, exist_ok=True)
        PLAN.write_text(json.dumps(plan, indent=1), encoding="utf-8")
        print(f"wrote {PLAN.relative_to(REPO)}: {plan['n_clips']} clips, {plan['span'][0]} .. {plan['span'][1]}, "
              f"{plan['footage_h_total']} camera-hours. Send this file to the other two laptops if they do not "
              f"have every clip.")
    for who, s in plan["shares"].items():
        print(f"  {who:16s} {s['start']} -> {s['end']}  ({s['footage_h']} camera-h, {len(s['clips'])} clips)")
    if a.plan_only:
        return 0

    share = plan["shares"][a.who]
    a.span_start = plan["span"][0]
    a.plan = plan
    have = {p.name.lower() for p in Path(a.videos).rglob("*") if p.suffix.lower() in camlib.VIDEO_EXTS}
    a.have = have
    if not a.skip_checks:
        from check_setup import run_checks
        print()
        rep = run_checks(a.who, a.videos, a.ffmpeg_dir)
        if rep.failed:
            print("\nFix the FAIL lines above, then run this command again. Nothing was started.")
            return 1
        if interactive and _ask("\ncontinue? (y/n)", "y").lower() not in ("y", "yes"):
            return 0
    hw = hardware()
    if not a.no_smoke:
        res = smoke_test(a, share)
        if not res.get("ok"):
            return 1
        if a.smoke_only:
            return 0
        if interactive and _ask(f"\nsmoke test passed. Continue with {a.who}'s full share? (y/n)", "y").lower() not in ("y", "yes"):
            return 0
    if interactive and a.chunk is None:
        v = _ask("chunk length in hours (2 or 4)", str(int(a.chunk_hours)) if a.chunk_hours in (2, 4) else "4")
        a.chunk_hours = float(v) if v in ("2", "4", "2.0", "4.0") else a.chunk_hours
    if a.parallel is None:
        a.parallel = hw["suggested_parallel"]
        if interactive:
            a.parallel = int(_ask(f"chunks in parallel ({hw['cores']} cores, {hw['ram_gb']} GB RAM, {hw['device']})",
                                  str(a.parallel)))
    a.parallel = max(1, a.parallel)
    chunks = chunks_of(share, a.chunk_hours, a.parallel)
    threads = max(1, hw["cores"] // a.parallel)
    workers = a.stage2_workers or max(2, 8 // a.parallel)     # stage 2 VLM threads per chunk: total TAMU concurrency ~8
    print(f"\n{a.who}: {share['start']} -> {share['end']} in {len(chunks)} chunks, {a.parallel} at a time "
          f"({threads} CPU threads each, device {hw['device']}, {workers} stage 2 workers each)")
    blocked = []
    for i, (s, e) in enumerate(chunks):
        miss = missing_for_window(plan, a.have, lead_in(s, a.span_start, a.overlap_min), e)
        print(f"  {a.who}_c{i + 1:02d}: owns {s} -> {e}  (tracks from {lead_in(s, a.span_start, a.overlap_min)})  "
              + ("clips: all present" if not miss else f"MISSING {len(miss)} clip(s), e.g. {', '.join(miss[:3])}"))
        if miss:
            blocked.append((i, miss))
    if a.dry_run:
        return 0
    if blocked and not a.allow_missing:
        print(f"\nSTOP: {len(blocked)} chunk(s) need clips that are not in {a.videos}.")
        print("FIX: copy those clips into the video folder (keep the original names; any subfolder is fine), then run again.")
        print("     (--allow-missing runs anyway: those minutes come out as 'not visible')")
        lst = SHARES / a.who / "missing_clips.txt"
        lst.parent.mkdir(parents=True, exist_ok=True)
        lst.write_text("\n".join(sorted({n for _, m in blocked for n in m})) + "\n", encoding="utf-8")
        print(f"     full list: {lst.relative_to(REPO)}")
        return 1
    if a.dry_run:
        return 0
    if a.chunk is not None:
        if not 1 <= a.chunk <= len(chunks):
            raise SystemExit(f"--chunk must be 1..{len(chunks)}")
        sel = [(a.chunk - 1, chunks[a.chunk - 1])]
    else:
        sel = list(enumerate(chunks))
    if interactive and a.chunk is None and _ask("start? (y/n)", "y").lower() not in ("y", "yes"):
        return 0

    status_p = SHARES / a.who / "share_status.json"
    status = json.loads(status_p.read_text(encoding="utf-8")) if status_p.exists() else {}
    lock = threading.Lock()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.parallel) as ex:
        rcs = list(ex.map(lambda ic: run_chunk(ic[0], a.who, ic[1][0], ic[1][1], a, threads, workers, lock, status), sel))
    bad = sum(1 for r in rcs if r)
    print(f"\n{a.who}: {len(sel) - bad}/{len(sel)} chunks done in {(time.time() - t0) / 3600:.1f} h")
    if bad:
        print("rerun the same command to resume the failed chunks (finished steps are skipped)")
        return 1
    print(f"next: python run_share.py --who {a.who} --pack   -> send outputs/shares/{a.who}.zip to the stitcher")
    return 0


if __name__ == "__main__":
    sys.exit(main())
