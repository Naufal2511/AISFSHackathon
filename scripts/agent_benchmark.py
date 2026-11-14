#!/usr/bin/env python3
"""Same investigations, different reasoning models: latency and quality side by side.

  python scripts/agent_benchmark.py --run outputs/level1_d1_1600_h --backends ollama tamu --n 8

Picks a fixed, diverse set of flags from the run (camera conflicts, jumps, weak IDs, behavior
checks, an ID audit), runs each with every backend on an identical copy of the memory cards,
and writes outputs/agent_bench/<run>/comparison.md + .json and the traces per backend.
Quality signals: decided within the step budget, passed the self-check on the first try,
evidence kinds used, and the trace text itself (read them side by side).
"""
import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import yaml

from src.agent.audit import audit_text, build_sheet
from src.agent.evidence import Embedder, FrameStore
from src.agent.loop import Agent
from src.agent.memory import Memory
from src.agent.tools import Toolbox
from src.agent.triggers import audit_schedule, find_flags
from src.agent.vlm import make_backend
from src.behavior.features import load_records
from src.behavior.rumination import index_videos
from src.state.world_state import WorldState


def pick_flags(rec, memory, tl, world, n):
    t_lo, t_hi = float(rec.t.min()), float(rec.t.max()) + 1
    audits = audit_schedule(t_lo, t_hi, 30) or [t_lo + (t_hi - t_lo) / 2]
    allf, last_seen = [], {}
    for w0 in np.arange(t_lo, t_hi, 300):
        allf += find_flags(rec, w0, min(w0 + 300, t_hi), memory.known_ids(), {"max_flags_per_window": 20},
                           tl, last_seen, audits, world=world)
    order = ["camera_conflict", "id_jump", "weak_id", "behavior_check", "id_audit", "new_cow_id", "cow_missing"]
    picked, used = [], set()
    while len(picked) < n and any(f.key not in used for f in allf):
        for kind in order:
            cand = [f for f in allf if f.kind == kind and f.key not in used]
            if cand and len(picked) < n:
                f = cand[len([p for p in picked if p.kind == kind]) % len(cand)]
                picked.append(f)
                used.add(f.key)
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--backends", nargs="+", default=["ollama", "tamu"])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--video-root", default="input")
    a = ap.parse_args()
    cfg = yaml.safe_load(open("configs/agent.yaml"))
    name = os.path.basename(a.run.rstrip("/"))
    out = os.path.join("outputs", "agent_bench", name)
    os.makedirs(out, exist_ok=True)

    rec = load_records([os.path.join(a.run, "identity", "tracking_records_resolved.csv")])
    tl = pd.read_csv(os.path.join(a.run, "behavior", "behavior_timeline_1s.csv"))
    tl["t"] = (pd.to_datetime(tl.wall_time) - pd.Timestamp("1970-01-01")).dt.total_seconds()
    world = WorldState.load(os.path.join(a.run, "state"))
    cam_pos = {k: v.get("mount_position_estimated") for k, v in yaml.safe_load(open("configs/cameras.yaml"))["cameras"].items()}
    frames, emb = FrameStore(rec, a.video_root), Embedder()
    mem0 = Memory(os.path.join(out, "memory"))
    if not mem0.cards:
        mem0.enroll(rec, frames, emb)
    flags = pick_flags(rec, mem0, tl, world, a.n)
    print("flags:", [f"{f.kind}({','.join(f.cow_ids[:2])})" for f in flags])

    rows, summary = [], {}
    for bname in a.backends:
        os.environ["AGENT_BACKEND"] = bname
        try:
            backend = make_backend(cfg)
        except Exception as e:
            print(f"[{bname}] cannot start: {e}")
            continue
        mem = copy.deepcopy(mem0)
        mem.root = os.path.join(out, bname, "memory")
        tb = Toolbox(rec, frames, mem, tl, emb, index_videos(a.video_root), world=world, cam_pos=cam_pos)
        agent = Agent(tb, mem, backend, os.path.join(out, bname), max_steps=cfg.get("agent", {}).get("max_steps", 6))
        t0 = time.time()
        for f in flags:
            imgs, txt = [], ""
            if f.kind == "id_audit":
                _, arows, sheet = build_sheet(f.t0, rec, frames, mem, emb, os.path.join(out, bname, "id_checks"))
                imgs, txt = [sheet], "Audit sheet (card | cam1 | cam2 | cam3):\n" + audit_text(arows)
            tr = agent.investigate(copy.deepcopy(f), imgs, txt)
            finals = [s for s in tr["steps"] if "final" in s]
            fin = tr.get("final") or {}
            kinds = sorted({s.get("evidence_kind") for s in tr["steps"] if s.get("evidence_kind") not in (None, "none")})
            rows.append({"backend": bname, "flag": f.kind, "cows": ",".join(f.cow_ids[:2]), "seconds": tr["seconds"],
                         "tool_calls": sum(1 for s in tr["steps"] if "tool" in s),
                         "decision": fin.get("decision"),
                         "budget_exhausted": "budget" in str(fin.get("summary", "")) or "step budget" in str(fin.get("summary", "")),
                         "self_check_first_try": bool(finals and finals[0].get("verified")),
                         "downgraded": "failed self-check" in str(fin.get("summary", "")),
                         "evidence_kinds": ",".join(kinds), "summary": str(fin.get("summary", ""))[:160]})
            print(f"[{bname}] {f.kind}: {fin.get('decision')} in {tr['seconds']} s - {str(fin.get('summary',''))[:90]}")
        d = pd.DataFrame([r for r in rows if r["backend"] == bname])
        summary[bname] = {
            "model": getattr(backend, "label", bname), "investigations": len(d),
            "mean_s": round(d.seconds.mean(), 1), "median_s": round(d.seconds.median(), 1),
            "s_per_model_call": round(backend.seconds / max(backend.calls, 1), 2), "model_calls": backend.calls,
            "decided_within_budget": f"{(~d.budget_exhausted).mean():.0%}",
            "self_check_pass_first_try": f"{d.self_check_first_try.mean():.0%}",
            "downgraded_by_self_check": int(d.downgraded.sum()),
            "decisions": d.decision.value_counts().to_dict(), "wall_s": round(time.time() - t0, 1)}
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "comparison_rows.csv"), index=False)
    json.dump(summary, open(os.path.join(out, "comparison.json"), "w"), indent=1, default=str)
    with open(os.path.join(out, "comparison.md"), "w") as fh:
        fh.write(f"# Agent backend comparison on {name}\n\n")
        fh.write(pd.DataFrame(summary).T.to_markdown() + "\n\n## Per investigation\n\n")
        fh.write(df.to_markdown(index=False) + "\n\nTraces: " + ", ".join(f"`{out}/{b}/traces/`" for b in summary) + "\n")
    print(json.dumps(summary, indent=1, default=str))
    print("->", os.path.join(out, "comparison.md"))


if __name__ == "__main__":
    main()
