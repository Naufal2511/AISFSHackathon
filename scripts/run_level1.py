#!/usr/bin/env python3
"""Level 1 chain on one tracking run: identity by agreement -> behavior -> supervisor agent.

  python scripts/run_level1.py --records outputs/run_d1_1600/tracking_records.csv \
      --out outputs/level1_d1_1600 --backend ollama

  <out>/identity/   tracking_records_resolved.csv, identity_report.json, identity_sheet.jpg
  <out>/behavior/   activity.csv, time_budgets.csv, ... (on the resolved IDs)
  <out>/agent/      traces, memory cards, ID checks, corrections, behavior/ after corrections
Watch it live:  python scripts/dashboard.py --run <out> --agent <out>/agent
"""
import argparse
import os
import subprocess
import sys
import time

PY = sys.executable


def step(name, cmd, log):
    t = time.time()
    print(f"== {name}: {' '.join(cmd[1:3])} ...", flush=True)
    with open(log, "a") as f:
        f.write(f"\n== {name}\n")
        f.flush()
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    print(f"   {'ok' if r.returncode == 0 else 'FAILED'} in {time.time() - t:.0f} s (log {log})", flush=True)
    if r.returncode:
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--video-root", default="input")
    ap.add_argument("--backend", default="ollama")
    ap.add_argument("--pen-size", default="8", help="herd size cap (anchor estimate: scripts/herd_size.py)")
    ap.add_argument("--max-flags", default=None)
    ap.add_argument("--skip-agent", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "agent"), exist_ok=True)
    log = os.path.join(a.out, "level1_log.txt")
    resolved = os.path.join(a.out, "identity", "tracking_records_resolved.csv")
    step("identity by agreement", [PY, "scripts/resolve_identities.py", "--records", a.records,
                                   "--out", os.path.join(a.out, "identity"), "--video-root", a.video_root,
                                   "--pen-size", a.pen_size], log)
    step("world state", [PY, "scripts/build_world_state.py", "--records", resolved,
                          "--out", os.path.join(a.out, "state")], log)
    step("anchors", [PY, "scripts/find_anchors.py", "--run", a.out], log)
    step("behavior", [PY, "classify_behavior.py", "--records", resolved, "--out", os.path.join(a.out, "behavior"),
                      "--video-root", a.video_root, "--probe-rumination"], log)
    step("contract export", [PY, "scripts/export_contract.py", "--run", a.out], log)
    if not a.skip_agent:
        cmd = [PY, "run_agent.py", "--records", resolved,
               "--behavior", os.path.join(a.out, "behavior", "behavior_timeline_1s.csv"),
               "--out", os.path.join(a.out, "agent"), "--video-root", a.video_root,
               "--backend", a.backend, "--rebuild-behavior", "--state", os.path.join(a.out, "state")]
        if a.max_flags:
            cmd += ["--max-flags", a.max_flags]
        with open(os.path.join(a.out, "agent", "agent_log.txt"), "w"):
            pass
        t = time.time()
        print("== supervisor agent ...", flush=True)
        r = subprocess.run(cmd, stdout=open(os.path.join(a.out, "agent", "agent_log.txt"), "a"),
                           stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        print(f"   {'ok' if r.returncode == 0 else 'FAILED'} in {time.time() - t:.0f} s", flush=True)
    print("done ->", a.out)


if __name__ == "__main__":
    main()
