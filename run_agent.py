#!/usr/bin/env python3
"""Run the supervisor agent over tracking output.

  # 1. memory cards (once per pen; then check outputs/agent/memory/enrollment_sheet.jpg)
  python run_agent.py --records outputs/run_d1_0719/tracking_records.csv --enroll-only

  # 2. supervise: flags -> VLM investigations -> verified corrections -> traces
  python run_agent.py --records outputs/run_d1_0719/tracking_records.csv \
      --behavior outputs/run_d1_0719/behavior/behavior_timeline_1s.csv --backend ollama --rebuild-behavior

Outputs (outputs/agent/ by default):
  traces/*.md|json        reasoning traces (thought, tool, evidence, decision, self-check)
  id_checks/*.jpg|csv     30-minute ID-check sheets
  corrections.csv         every non-trivial decision
  tracking_records_corrected.csv, behavior/ (activity.csv, time_budgets.csv, ... after corrections)
  agent_summary.json
"""
import argparse
import glob
import json
import os
import sys
import time

import pandas as pd
import yaml

from src.agent.evidence import Embedder, FrameStore
from src.agent.loop import Agent, apply_behavior_overrides, apply_id_corrections
from src.agent.memory import Memory
from src.agent.tools import Toolbox
from src.agent.vlm import make_backend
from src.behavior.features import load_records
from src.behavior.rumination import index_videos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", nargs="+", required=True, help="tracking_records.csv path(s) or globs")
    ap.add_argument("--behavior", help="behavior_timeline_1s.csv from classify_behavior.py (enables behavior checks)")
    ap.add_argument("--video-root", default="input")
    ap.add_argument("--state", help="world-state folder from scripts/build_world_state.py (enables cow_state, "
                                    "visibility, best_view, trajectory_map, anomalies tools and physics triggers)")
    ap.add_argument("--out", default="outputs/agent")
    ap.add_argument("--config", default="configs/agent.yaml")
    ap.add_argument("--backend", help="ollama | tamu | vllm | rules (overrides the config)")
    ap.add_argument("--enroll", action="store_true", help="rebuild memory cards before running")
    ap.add_argument("--enroll-only", action="store_true")
    ap.add_argument("--max-flags", type=int, help="stop after N investigations (quick tests)")
    ap.add_argument("--rebuild-behavior", action="store_true",
                    help="re-run the behavior classifier on the corrected records and apply behavior overrides")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    if a.backend:
        os.environ["AGENT_BACKEND"] = a.backend
    paths = sorted({p for g in a.records for p in glob.glob(g)})
    rec = load_records(paths)
    print(f"[AGENT] {len(rec)} records from {len(paths)} file(s), "
          f"{pd.Timestamp(rec.t.min(), unit='s')} .. {pd.Timestamp(rec.t.max(), unit='s')}")

    tl = None
    if a.behavior and os.path.exists(a.behavior):
        tl = pd.read_csv(a.behavior)
        tl["t"] = (pd.to_datetime(tl.wall_time) - pd.Timestamp("1970-01-01")).dt.total_seconds()

    frames = FrameStore(rec, a.video_root)
    embedder = Embedder()
    memory = Memory(os.path.join(a.out, "memory"))
    if a.enroll or a.enroll_only or not memory.cards:
        made = memory.enroll(rec, frames, embedder)
        print(f"[AGENT] memory cards: {', '.join(made)} -> {os.path.join(a.out, 'memory', 'enrollment_sheet.jpg')}")
        if a.enroll_only:
            print("Check the sheet; set `confirmed: true`, `ear_tag`, or `same_as: COW_xx` in "
                  f"{memory.path} for duplicates, then run without --enroll-only.")
            return

    backend = make_backend(cfg)
    print(f"[AGENT] reasoning backend: {getattr(backend, 'label', backend.name)}")
    if hasattr(backend, "complete"):
        n = memory.describe_traits(backend)
        if n:
            print(f"[AGENT] VLM wrote distinguishing traits on {n} memory cards")
    world, cam_pos = None, {}
    if a.state and os.path.exists(os.path.join(a.state, "cow_state.csv")):
        from src.state.world_state import WorldState
        world = WorldState.load(a.state)
        cam_pos = {k: v.get("mount_position_estimated") for k, v in
                   yaml.safe_load(open("configs/cameras.yaml"))["cameras"].items()}
        print(f"[AGENT] world state: {len(world.state)} cow-seconds, {len(world.anomalies)} anomaly events")
    tb = Toolbox(rec, frames, memory, tl, embedder, index_videos(a.video_root), world=world, cam_pos=cam_pos)
    ac = cfg.get("agent", {})
    trig = {**cfg.get("triggers", {}), "max_flags_per_window": ac.get("max_flags_per_window", 8),
            "audit_every_min": ac.get("audit_every_min", 30)}
    agent = Agent(tb, memory, backend, a.out, max_steps=ac.get("max_steps", 6))
    t0 = time.time()
    traces = agent.run(window_sec=ac.get("window_sec", 300), audit_every_min=ac.get("audit_every_min", 30),
                       trigger_cfg=trig, embedder=embedder, max_flags=a.max_flags,
                       parallel=ac.get("parallel_investigations", 1))
    frames.close()

    corr = pd.DataFrame(agent.corrections)
    corr.to_csv(os.path.join(a.out, "corrections.csv"), index=False)
    fixed = apply_id_corrections(rec, corr)
    orig_cols = list(pd.read_csv(paths[0], nrows=0).columns)
    fixed_path = os.path.join(a.out, "tracking_records_corrected.csv")
    fixed[orig_cols + ["agent_action"]].to_csv(fixed_path, index=False)

    if a.rebuild_behavior:
        from src.behavior.classifier import BehaviorClassifier
        from src.behavior.export import write_all
        clf = BehaviorClassifier()
        new_tl = clf.run([fixed_path], video_index=index_videos(a.video_root))
        new_tl = apply_behavior_overrides(new_tl, corr)
        out = write_all(os.path.join(a.out, "behavior"), new_tl, clf.cfg)
        print(f"[AGENT] behavior outputs after corrections -> {os.path.dirname(out['activity_csv'])}")

    decisions = pd.Series([(t.get("final") or {}).get("decision") for t in traces]).value_counts().to_dict()
    summary = {"backend": getattr(backend, "label", backend.name), "investigations": len(traces),
               "decisions": decisions, "flags": pd.Series([t["flag"]["kind"] for t in traces]).value_counts().to_dict(),
               "vlm_calls": getattr(backend, "calls", 0), "vlm_seconds": round(getattr(backend, "seconds", 0), 1),
               "mean_seconds_per_investigation": round(sum(t["seconds"] for t in traces) / max(len(traces), 1), 1),
               "wall_seconds": round(time.time() - t0, 1), "corrections_csv": os.path.join(a.out, "corrections.csv")}
    json.dump(summary, open(os.path.join(a.out, "agent_summary.json"), "w"), indent=1, default=str)
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    sys.exit(main())
