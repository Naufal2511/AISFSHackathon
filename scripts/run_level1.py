#!/usr/bin/env python3
"""Level 1 chain on one tracking run: identity by agreement -> [identity agent] -> behavior -> contract [-> supervisor agent].

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
    ap.add_argument("--pen-size", default="10", help="herd size cap: the pen holds at most 10; the evidence count decides how many IDs are used")
    ap.add_argument("--max-flags", default=None)
    ap.add_argument("--skip-agent", action="store_true")
    ap.add_argument("--id-agent", action="store_true",
                    help="run the supervisor agent on identity flags only (duplicate_id, id_jump, extra_cow, cow_missing, "
                         "weak_id, 30-min id_audit) BEFORE behavior + contract export, so behavior is built on corrected "
                         "IDs; behavior is left to the stage 2 validation agent (brd_agent). Implies --skip-agent.")
    ap.add_argument("--id-cards", action="store_true",
                    help="only build the memory cards (per-cow appearance prototypes, no VLM): needed to align cow IDs "
                         "across runs (scripts/stitch_shares.py)")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "agent"), exist_ok=True)
    log = os.path.join(a.out, "level1_log.txt")
    resolved = os.path.join(a.out, "identity", "tracking_records_resolved.csv")
    step("identity by agreement", [PY, "scripts/resolve_identities.py", "--records", a.records,
                                   "--out", os.path.join(a.out, "identity"), "--video-root", a.video_root,
                                   "--pen-size", a.pen_size], log)
    state = os.path.join(a.out, "state")
    step("world state", [PY, "scripts/build_world_state.py", "--records", resolved, "--out", state], log)
    step("anchors", [PY, "scripts/find_anchors.py", "--run", a.out], log)
    agent_dir = os.path.join(a.out, "agent")
    if a.id_agent:
        # identity supervisor: no --behavior, so no behavior_check flags (stage 2 owns behavior)
        cmd = [PY, "run_agent.py", "--records", resolved, "--out", agent_dir, "--video-root", a.video_root,
               "--backend", a.backend, "--state", state]
        if a.max_flags:
            cmd += ["--max-flags", a.max_flags]
        step("identity agent", cmd, log)
        if _apply_corrections(resolved, os.path.join(agent_dir, "tracking_records_corrected.csv")):
            step("world state (corrected IDs)", [PY, "scripts/build_world_state.py", "--records", resolved, "--out", state], log)
            step("anchors (corrected IDs)", [PY, "scripts/find_anchors.py", "--run", a.out], log)
    elif a.id_cards:
        step("memory cards", [PY, "run_agent.py", "--records", resolved, "--out", agent_dir,
                              "--video-root", a.video_root, "--enroll-only"], log)
    step("behavior", [PY, "classify_behavior.py", "--records", resolved, "--out", os.path.join(a.out, "behavior"),
                      "--video-root", a.video_root, "--probe-rumination"], log)
    step("contract export", [PY, "scripts/export_contract.py", "--run", a.out], log)
    if not (a.skip_agent or a.id_agent or a.id_cards):
        cmd = [PY, "run_agent.py", "--records", resolved,
               "--behavior", os.path.join(a.out, "behavior", "behavior_timeline_1s.csv"),
               "--out", agent_dir, "--video-root", a.video_root,
               "--backend", a.backend, "--rebuild-behavior", "--state", state]
        if a.max_flags:
            cmd += ["--max-flags", a.max_flags]
        with open(os.path.join(agent_dir, "agent_log.txt"), "w"):
            pass
        t = time.time()
        print("== supervisor agent ...", flush=True)
        r = subprocess.run(cmd, stdout=open(os.path.join(agent_dir, "agent_log.txt"), "a"),
                           stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        print(f"   {'ok' if r.returncode == 0 else 'FAILED'} in {time.time() - t:.0f} s", flush=True)
    print("done ->", a.out)


def _apply_corrections(resolved: str, corrected: str) -> bool:
    """Replace the resolved records with the identity agent's corrected ones (original kept as *.pre_agent.csv).
    Returns True when at least one box changed, so world state is rebuilt only when needed."""
    import pandas as pd
    if not os.path.exists(corrected):
        return False
    fixed = pd.read_csv(corrected, low_memory=False)
    act = fixed.pop("agent_action") if "agent_action" in fixed else pd.Series(dtype=object)
    n = int(act.fillna("").astype(str).str.len().gt(0).sum())
    if not n:
        return False
    backup = resolved.replace(".csv", ".pre_agent.csv")
    if not os.path.exists(backup):
        os.replace(resolved, backup)
    fixed.to_csv(resolved, index=False)
    print(f"   identity agent corrected {n} boxes -> {resolved} (original: {backup})", flush=True)
    return True


if __name__ == "__main__":
    main()
