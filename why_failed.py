#!/usr/bin/env python3
"""Why did a chunk (or pipeline run) stop? Reads only files on disk: safe to run while chunks are running.

  python why_failed.py                                  # every share on this machine
  python why_failed.py --who afeef_training             # one share
  python why_failed.py --run outputs/pipeline/demo_5min # one pipeline run folder

For each chunk it prints which step failed or was interrupted, the error and the traceback from that step's log,
seconds skipped by tracking (step_errors.log), whether a tracking checkpoint exists, and the likely cause with the
fix. Then: rerun the same command; finished steps, finished chunks and the tracking checkpoint are reused.
Full guide: docs/TROUBLESHOOTING.md
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
SHARES = REPO / "outputs" / "shares"
STEP_LOG = {"track": "1_track.log", "level1": "2_level1.log", "assets": "2b_assets.log", "import": "3_import.log",
            "agent": "4_agent.log", "export": "5_export.log", "level2": "6_level2.log"}
# Only used when the process cannot be checked (HPRC login node, no psutil): a 'running' step whose log has been
# silent this long is probably dead. A quiet log alone proves nothing: level1's behavior model prints nothing for
# ~40 min (c01: 39 min with the GPU to itself, longer while other chunks track).
STALE_MIN = 60
GONE = ("the process is gone without an error in its log: the terminal was closed, the laptop slept or shut down, "
        "or the OS killed it (out of memory)",
        "rerun the same command; keep the laptop awake (caffeinate -dims ... on a Mac, Sleep: Never on Windows); "
        "if it keeps happening, run fewer chunks at once")
UNSEEN = ("the log has been silent for a long time and this machine cannot see the process (HPRC login node, or "
          "psutil missing)",
          "HPRC: squeue -u $USER (job still listed = still running: wait), sacct -j <jobid> for how it ended; "
          "otherwise rerun the same command")


def alive(run: Path) -> list[int] | None:
    """PIDs of processes on THIS machine working on the run: its folder (every step's command has it) or
    run_pipeline.py's --name <run> in the command line. None when that cannot be checked: no psutil (it comes with
    ultralytics), or a cluster login node (squeue present), where the job runs on another node."""
    if shutil.which("squeue"):
        return None
    try:
        import psutil
    except ImportError:
        return None
    norm = (lambda t: t.lower()) if os.name == "nt" else (lambda t: t)
    pat = re.compile(re.escape(norm(str(run.resolve()))) + r"([\\/\s\"']|$)|--name " + re.escape(norm(run.name)) + r"(\s|$)")
    pids = []
    try:
        for p in psutil.process_iter(["pid", "cmdline"]):
            cmd = norm(" ".join(p.info.get("cmdline") or []))
            if p.info["pid"] != os.getpid() and pat.search(cmd):
                pids.append(p.info["pid"])
    except Exception:  # noqa: BLE001  (a process list we cannot read is "cannot check", not "dead")
        return None
    return pids

# (pattern in the error/traceback, likely cause, fix): first match wins, so specific patterns come first
CAUSES = [
    (r"could not be broadcast together with shapes \(\d+,\) \(\d+,\)",
     "a tracking bug in code older than commit 4124414 (appearance vectors of different lengths)",
     "git pull, then rerun the same command"),
    (r"steps in a row failed",
     "tracking skipped too many seconds in a row: something is really broken (see the errors below)",
     "read the step_errors.log lines below; fix that cause; rerun (it resumes from the checkpoint)"),
    (r"KeyboardInterrupt|Killed: 9|SIGTERM|exit -15|exit -9|exit 130|exit 137",
     "the process was stopped (Ctrl+C, terminal closed, laptop slept or ran out of memory, HPRC time limit)",
     "rerun the same command; keep the laptop awake (caffeinate -dims ... on a Mac); if it keeps dying with "
     "'Killed', run fewer chunks at once (--parallel 1)"),
    (r"out of memory|MPS backend out of memory|CUDA out of memory|MemoryError|Cannot allocate memory|OUT_OF_MEMORY",
     "not enough memory (RAM or GPU)",
     "run fewer chunks at once (--parallel 1), close other apps; HPRC: raise --mem in scripts/hprc/pipeline.slurm"),
    (r"No space left on device|Disk quota exceeded",
     "the disk (or HPRC quota) is full",
     "free space: delete <chunk>/cache/remux of finished chunks, old outputs/pipeline runs; HPRC: showquota"),
    (r"(TAMU|chat-api\.tamu\.ai).*(401|403)|\b401\b.*Unauthorized|error code: 1010|Forbidden",
     "TAMU AI rejected the request (wrong/expired key, or blocked client)",
     "check TAMU_API_KEY in .env (no quotes/spaces), then: python check_setup.py --who <you> --videos <clips>"),
    (r"\b429\b|Too Many Requests|rate.?limit",
     "TAMU rate limit (too many requests from this key at once)",
     "rerun; if it repeats, lower --stage2-workers and PEN6_ID_PARALLEL (e.g. 2)"),
    (r"OPENAI_API_KEY|api\.openai\.com.*(401|403|404)|model_not_found",
     "TAMU failed 3 times and the OpenAI fallback did not work either (no/invalid key or model name)",
     "rerun later (TAMU outage) or set OPENAI_API_KEY / OPENAI_MODEL in .env"),
    (r"ConnectionError|ConnectTimeout|ReadTimeout|Max retries exceeded|NameResolutionError|"
     r"nodename nor servname|Temporary failure in name resolution|SSLError|Connection reset",
     "network: TAMU (or OpenAI) could not be reached",
     "check Wi-Fi/VPN; HPRC compute nodes need the WebProxy module (scripts/hprc/env.sh); then rerun"),
    (r"No such file or directory: 'ffmpeg'|ffmpeg: command not found|ffmpeg.*not found",
     "ffmpeg is not installed or not on PATH",
     "install it (docs/LAPTOP_SETUP.md section 3) or pass --ffmpeg-dir <folder with ffmpeg>"),
    (r"No clips of .* fall in|missing clip|CameraError|No valid camera sources",
     "the footage for this time window is not in your video folder",
     "python fetch_clips.py --who <you> --dest <video folder>, or check --videos points at the right folder"),
    (r"invalid load key|UnpicklingError|PytorchStreamReader failed|version https://git-lfs",
     "a model file is a Git LFS pointer, not the real weights",
     "git lfs install && git lfs pull"),
    (r"ModuleNotFoundError|ImportError",
     "a Python package is missing (or the wrong Python/venv is active)",
     "activate the venv, then: pip install -r requirements.txt -r brd_agent/requirements.txt"),
    (r"plan\.json|fingerprint",
     "this machine's plan.json differs from the team's",
     "copy the team's outputs/shares/plan.json here (check_setup.py prints its fingerprint)"),
]


def tail_lines(p: Path, n: int = 4000, latest: bool = True) -> list[str]:
    """Last n lines; with latest, only the latest attempt (logs are appended to on every rerun, each attempt
    starts with a '===== ' header), so an old, already fixed error is not reported again."""
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    if latest:
        heads = [i for i, l in enumerate(lines) if l.startswith("===== ")]
        if heads:
            lines = lines[heads[-1]:]
    return lines[-n:]


def last_traceback(lines: list[str], max_lines: int = 30) -> list[str]:
    starts = [i for i, l in enumerate(lines) if "Traceback (most recent call last)" in l]
    if starts:
        tb = lines[starts[-1]:]
        end = next((i for i, l in enumerate(tb) if i > 0 and re.match(r"\s*\w*(Error|Exception|Interrupt)\b", l.strip())),
                   len(tb) - 1)
        tb = tb[:end + 1]
        return tb if len(tb) <= max_lines else tb[:4] + ["   ..."] + tb[-(max_lines - 5):]
    return [l for l in lines if re.search(r"error|failed|exception", l, re.I)][-8:]


def diagnose(text: str):
    for pat, cause, fix in CAUSES:
        if re.search(pat, text, re.I):
            return cause, fix
    return None


def age_min(p: Path) -> float:
    return (time.time() - p.stat().st_mtime) / 60 if p.exists() else float("inf")


def step_errors(run: Path) -> tuple[int, list[str]]:
    p = run / "stage1" / "tracking" / "step_errors.log"
    if not p.exists():
        return 0, []
    lines = tail_lines(p, 100000, latest=False)
    n = sum(1 for l in lines if l.startswith("==== "))
    excs = [l for l in lines if re.match(r"^\w+(Error|Exception)\b", l)]
    uniq = list(dict.fromkeys(excs))[-3:]
    return n, uniq


def report_run(run: Path, chunk_log: Path | None = None) -> bool:
    """Prints the run's state; True when everything finished."""
    st_p = run / "pipeline_state.json"
    print(f"\n== {run.relative_to(REPO) if run.is_relative_to(REPO) else run}")
    if not st_p.exists():
        lines = tail_lines(chunk_log) if chunk_log else []
        print("   no pipeline_state.json: the run never started its first step")
        if lines:
            tb = last_traceback(lines)
            print("   last lines of " + str(chunk_log.relative_to(REPO)) + ":\n      " + "\n      ".join(tb))
            d = diagnose("\n".join(lines[-200:]))
            if d:
                print(f"   LIKELY CAUSE: {d[0]}\n   FIX:          {d[1]}")
        return False
    steps = json.loads(st_p.read_text(encoding="utf-8")).get("steps", {})
    live = alive(run) if any(s.get("status") == "running" for s in steps.values()) else None
    bad, stopped = None, None
    for name, s in steps.items():
        status = s.get("status", "?")
        log = run / "logs" / STEP_LOG.get(name, f"{name}.log")
        quiet = age_min(log)
        if status == "running" and live:
            status = f"running (process {live[0]} alive; log updated {quiet:.0f} min ago)"
        elif status == "running" and live == []:
            status = "INTERRUPTED (no process for this chunk on this machine)"
            if bad is None:
                bad, stopped = (name, s, log), GONE
        elif status == "running" and quiet > STALE_MIN:
            status = f"INTERRUPTED? (log silent for {quiet:.0f} min; the process cannot be checked here)"
            if bad is None:
                bad, stopped = (name, s, log), UNSEEN
        elif status == "running":
            status = f"running (log updated {quiet:.0f} min ago)"
        elif status == "failed":
            status = "FAILED"
            bad = bad or (name, s, log)
        took = f"  {s['seconds'] / 60:.0f} min" if s.get("seconds") else ""
        print(f"   {name:8s} {status}{took}")
    n_err, excs = step_errors(run)
    if n_err:
        print(f"   tracking skipped {n_err} second(s) with errors (stage1/tracking/step_errors.log); last kinds:")
        for e in excs:
            print(f"      {e}")
    ck = run / "stage1" / "tracking" / "tracking_checkpoint.pkl"
    if ck.exists():
        print(f"   tracking checkpoint: yes ({ck.stat().st_size / 1e6:.0f} MB, saved {age_min(ck):.0f} min ago): "
              f"a rerun continues from it")
    if bad is None:
        done = steps and all(s.get("status") == "ok" for s in steps.values())
        print("   nothing failed" + (" (all steps done)" if done else " (still running or not finished)"))
        return bool(done)
    name, s, log = bad
    lines = tail_lines(log)
    text = "\n".join([s.get("error", "")] + lines[-300:])
    if chunk_log:
        text += "\n" + "\n".join(tail_lines(chunk_log, 60))
    print(f"\n   step '{name}': log {log.relative_to(REPO) if log.is_relative_to(REPO) else log}")
    if s.get("error"):
        print("   error: " + s["error"].splitlines()[0][:300])
    tb = last_traceback(lines)
    if tb:
        print("   traceback / last error lines:\n      " + "\n      ".join(l[:220] for l in tb))
    d = diagnose(text + "\n" + "\n".join(excs)) or stopped
    if d:
        print(f"\n   LIKELY CAUSE: {d[0]}\n   FIX:          {d[1]}")
    else:
        print("\n   LIKELY CAUSE: not a known one. Send this output and the log above to the team "
              "(never the .env file).")
    print("   THEN:         rerun the same command (finished steps/chunks are skipped"
          + (", tracking resumes from its checkpoint)" if ck.exists() else ")"))
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--who", help="share name (default: every share in outputs/shares/)")
    ap.add_argument("--run", help="one pipeline run folder instead of a share")
    a = ap.parse_args(argv)
    if a.run:
        return 0 if report_run(Path(a.run).resolve()) else 1
    whos = [a.who] if a.who else sorted(p.name for p in SHARES.iterdir() if p.is_dir() and not p.name.startswith("_")) \
        if SHARES.exists() else []
    if not whos:
        print("no shares in outputs/shares/ (use --run <folder> for a single pipeline run)")
        return 1
    ok = True
    for who in whos:
        d = SHARES / who
        status = {}
        if (d / "share_status.json").exists():
            status = json.loads((d / "share_status.json").read_text(encoding="utf-8"))
        chunks = sorted({p.name for p in d.iterdir() if p.is_dir() and p.name.startswith(who + "_c")} | set(status))
        print(f"\n######## {who}: {len(chunks)} chunk(s) started")
        for c in chunks:
            ok &= report_run(d / c, d / f"{c}.log")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
