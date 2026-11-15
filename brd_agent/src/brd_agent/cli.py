"""Batch runner:  gate -> (agent | tier-1 passthrough | abstain) -> minute_labels_agent.csv + traces + summary.

  python -m brd_agent.cli --briefs examples/briefs_example.jsonl --out outputs/demo --backend mock --vlm mock
  python -m brd_agent.cli --briefs briefs.jsonl --out outputs/level1_agent --backend mypkg.adapter:RealBackend \
         --vlm auto --config configs/agent.yaml --workers 8 --max-agent-runs 4000
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .backend import DataBackend, MockBackend
from .config import load_config
from .gate import EscalationGate, select_within_budget
from .orchestrator import BehaviorAgent, tier1_decision
from .schemas import Brief
from .trace import TraceLogger
from .env import load_settings
from .vlm_client import MockVLM, make_vlm


def load_backend(spec: str) -> DataBackend:
    if spec == "mock":
        return MockBackend()
    mod, _, name = spec.partition(":")
    return getattr(importlib.import_module(mod), name)()      # class (or factory) constructible with no args


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--briefs", required=True, help="JSONL of Brief objects")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--backend", default="mock", help="'mock' or 'package.module:Class'")
    ap.add_argument("--vlm", default="mock", choices=["mock", "auto"], help="auto = provider from .env (USE_LOCAL_VLM)")
    ap.add_argument("--env", default=".env")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-agent-runs", type=int, default=None, help="cap VLM-escalated minutes (highest priority kept)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gate-only", action="store_true", help="only write gate_report.csv (tune thresholds, estimate cost)")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    briefs = [Brief.model_validate_json(l) for l in Path(a.briefs).read_text(encoding="utf-8").splitlines() if l.strip()][: a.limit]

    gate = EscalationGate(cfg.gate)
    gated = [(b, gate.decide(b)) for b in briefs]
    if a.max_agent_runs is not None:
        gated = select_within_budget(gated, a.max_agent_runs)

    with open(out / "gate_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["cow_id", "day", "minute", "action", "reasons", "priority"])
        for b, g in gated:
            w.writerow([b.cow_id, b.day, b.minute, g.action, ";".join(g.reasons), round(g.priority, 3)])
    n_agent = sum(g.action == "agent" for _, g in gated)
    print(f"gate: {len(briefs)} minutes -> {n_agent} to agent ({100 * n_agent / max(1, len(briefs)):.1f}%)")
    if a.gate_only:
        return

    backend = load_backend(a.backend)
    vlm = MockVLM() if a.vlm == "mock" else make_vlm(load_settings(a.env), cfg)
    agent = BehaviorAgent(cfg, backend, vlm, trace_logger=TraceLogger(out / "traces"), workdir=str(out / "work"))

    def work(item):
        b, g = item
        if g.action == "agent":
            return agent.run(b, g)
        return tier1_decision(b, g.action if g.action == "auto_abstain" else "keep_tier1", ["gate:" + r for r in g.reasons])

    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        decisions = list(ex.map(work, gated))

    cols = ["cow_id", "day", "minute", "behavior", "confidence", "source", "tier", "reason", "cameras_used",
            "tier1_label", "n_tool_calls", "n_vlm_calls", "trace_id", "flags"]
    with open(out / "minute_labels_agent.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(cols)
        for d in sorted(decisions, key=lambda d: (d.cow_id, d.day, d.minute)):
            row = d.model_dump()
            row["cameras_used"] = ";".join(row["cameras_used"]); row["flags"] = ";".join(row["flags"])
            w.writerow([row[c] for c in cols])

    changed = sum(1 for d in decisions if d.tier == 2 and d.behavior != d.tier1_label)
    summary = {"n_minutes": len(decisions), "n_agent": n_agent,
               "by_tier": dict(Counter(d.tier for d in decisions)),
               "by_reason": dict(Counter(d.reason for d in decisions)),
               "gate_reasons": dict(Counter(r for _, g in gated for r in g.reasons)),
               "agent_changed_tier1_label": changed,
               "total_tool_calls": sum(d.n_tool_calls for d in decisions),
               "total_vlm_calls": sum(d.n_vlm_calls for d in decisions)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
