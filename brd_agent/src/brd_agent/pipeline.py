"""DB-driven batch run. Per cow, minutes are processed strictly in time order so history (previous FINAL labels) exists;
parallelism is across cows. Each finished minute is written back to its row immediately (resumable: rerun skips done rows)."""
from __future__ import annotations

import json
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

from .backend import DataBackend
from .config import AgentConfig
from .calm import CalmChain
from .store import Store, HistoryStore, row_to_brief
from . import night
from .gate import EscalationGate, GateResult, select_within_budget
from .prescreen import prescreen
from .orchestrator import BehaviorAgent, tier1_decision
from .schemas import Decision
from .trace import TraceLogger
from .verifier import Verifier
from .vlm_client import VLMClient


def _no_upstream(row: dict) -> Decision:
    why = "Neither YOLO output nor video covers this cow-minute, so there is nothing to label (reported as not visible, not guessed)"
    return Decision(cow_id=row["cow_id"], day=row["day"], minute=row["minute"], behavior="other_not_visible",
                    confidence=0.0, source="not_visible", tier=1, reason="no_upstream_data", flags=["no_upstream"],
                    rationale=why, correction="miss_not_found",
                    actions=[{"seq": 1, "actor": "gate", "action": "no_upstream", "args": {}, "reason": why,
                              "outcome": "other_not_visible", "source": ""}])


def _prepass(db: Store, cows: list[str], days: Optional[list[int]], statuses, gate: EscalationGate,
             max_agent_runs: Optional[int]) -> dict[tuple[str, int], GateResult]:
    """Cheap read-only pass: gate every pending minute (tier-1 neighbours stand in for history) and, when a VLM budget
    is set, keep only the highest-priority escalations."""
    items, keys = [], []
    for cow in cows:
        rows = db.iter_rows(cow, days, statuses)
        for i, r in enumerate(rows):
            prev = rows[i - 1]["tier1_label"] if i and rows[i - 1]["t_abs"] == r["t_abs"] - 1 else None
            nxt = rows[i + 1]["tier1_label"] if i + 1 < len(rows) and rows[i + 1]["t_abs"] == r["t_abs"] + 1 else None
            b = row_to_brief(r, prev, nxt)
            if b is not None:
                items.append((b, gate.decide(b))); keys.append((cow, r["t_abs"]))
    if max_agent_runs is not None:
        items = select_within_budget(items, max_agent_runs)
    return {k: g for k, (_, g) in zip(keys, items)}


def _find_missing(brief, backend, agent: BehaviorAgent, cfg: AgentConfig) -> Decision:
    """YOLO produced nothing for this cow-minute although video exists. Try to recover her; label only from what is seen."""
    acts: list[dict] = []
    base = dict(cow_id=brief.cow_id, day=brief.day, minute=brief.minute, tier=1, source="not_visible", behavior="other_not_visible",
                confidence=0.0, tier1_label=None, flags=["yolo_miss"])
    if not cfg.verify.find_missing:
        why = "recovery of YOLO misses is switched off in config"
        return Decision(**base, reason="miss_not_recovered", rationale=why, correction="miss_not_found",
                        actions=[{"seq": 1, "actor": "gate", "action": "find_missing_disabled", "args": {}, "reason": why, "outcome": "skipped", "source": ""}])
    rec = backend.recover_box(brief.cow_id, brief.day, brief.minute)
    acts.append({"actor": "tool", "action": "tool:recover_box", "args": {}, "source": "find_missing", "outcome": json.dumps(rec, default=str)[:300],
                 "reason": "YOLO has no record for this cow although the video covers this minute: look for her with the detector, "
                           "continuing her last known position"})
    if not rec.get("ok"):
        why = f"YOLO missed this cow-minute and she could not be recovered: {rec.get('reason')}"
        acts.append({"actor": "resolver", "action": "final_decision", "args": {}, "reason": why, "outcome": "miss_not_found", "source": ""})
        return Decision(**base, reason="miss_not_recovered", rationale=why, correction="miss_not_found",
                        actions=[{"seq": i + 1, **a} for i, a in enumerate(acts)])
    vis = {rec["camera"]: cfg.video.recovered_visibility}
    b2 = brief.model_copy(update={"visible_frac": vis, "id_conf": 0.5, "yolo_flags": brief.yolo_flags + ["recovered_box"]})
    d = agent.run(b2)
    d.flags.append("recovered_by_detector")
    d.actions = [{"seq": 0, **a} for a in acts] + d.actions
    for i, a in enumerate(d.actions):
        a["seq"] = i + 1
    return d


def _passthrough(brief, g: GateResult, ps_acts: list[dict], reasons: list[str], night_act: Optional[dict]) -> Decision:
    d = tier1_decision(brief, "auto_abstain" if g.action == "auto_abstain" else "keep_tier1", ["gate:" + r for r in g.reasons])
    acts = []
    if night_act:
        acts.append(night_act)
    acts.append({"actor": "gate", "action": f"gate:{g.action}",
                 "args": {}, "reason": ("escalation rules: " + ", ".join(g.reasons)) if g.reasons else
                 "YOLO confident, no ambiguity, visibility/ID fine: nothing for the gate to escalate", "outcome": g.action, "source": ""})
    acts += ps_acts
    if reasons:
        acts.append({"actor": "gate", "action": "escalation_suppressed", "args": {}, "reason": "the agent run budget is exhausted, so "
                     "this flagged minute keeps YOLO's label at reduced confidence", "outcome": "; ".join(reasons), "source": ""})
        d.flags.append("over_budget_unverified")
        d.confidence = min(d.confidence, 0.6)
        d.rationale = "YOLO's label kept; rule checks raised " + "; ".join(reasons) + " but the agent budget was exhausted."
        d.correction = "unresolved"
    acts.append({"actor": "resolver", "action": "final_decision", "args": {"behavior": d.behavior}, "reason": d.rationale,
                 "outcome": f"{d.correction}: {brief.top_label} -> {d.behavior}", "source": ""})
    d.actions = [{"seq": i + 1, **a} for i, a in enumerate(acts)]
    return d


def run_db(store_dir: str, cfg: AgentConfig, backend_factory: Callable[[], DataBackend], vlm: VLMClient,
           trace_dir: Optional[str] = None, workers: int = 1, cows: Optional[list[str]] = None,
           days: Optional[list[int]] = None, max_agent_runs: Optional[int] = None, rerun: bool = False,
           limit_per_cow: Optional[int] = None, progress_every: int = 200) -> dict:
    t0 = time.time()
    db = Store(store_dir)
    if db.get_meta("speed_unit") == "m_s":          # tracking input: speeds are metric, so use the metric thresholds
        cfg = cfg.model_copy(deep=True)
        cfg.verifier.walk_speed_px_s, cfg.verifier.still_speed_px_s = cfg.verifier.walk_speed_m_s, cfg.verifier.still_speed_m_s
    cows = cows or db.cows()
    if rerun:
        db.reset_agent(cows, days)
    statuses = ("pending", "stale")
    gate = EscalationGate(cfg.gate)
    gate_map = _prepass(db, cows, days, statuses, gate, max_agent_runs)
    n_gate_agent = sum(g.action == "agent" for g in gate_map.values())
    db.close()

    lock = threading.Lock()
    done = {"n": 0}
    spare = {"n": (max(0, max_agent_runs - n_gate_agent) if max_agent_runs is not None else None)}   # budget for rule/audit escalations
    verifier = Verifier(cfg.verifier, cfg.night)
    counts: Counter = Counter()

    def process_cow(cow: str) -> None:
        cdb = Store(store_dir)
        try:
            hist = HistoryStore(cdb, cfg.history)
            backend = backend_factory()
            agent = BehaviorAgent(cfg, backend, vlm,
                                  trace_logger=TraceLogger(trace_dir) if trace_dir else TraceLogger(None),
                                  workdir=str((Path(trace_dir).parent if trace_dir else Path("work")) / "work" / cow),
                                  history=hist)
            rows = cdb.iter_rows(cow, days, statuses)[: limit_per_cow]
            chain = CalmChain()
            for row in rows:
                prev, nxt = hist.prev_next(cow, row["day"], row["minute"])
                brief = row_to_brief(row, prev, nxt)
                d = chain.try_carry(brief, row, cfg) if brief is not None else None
                if d is not None:
                    pass                                   # calm minute: inherits the bout's verified result, no video decode, no VLM
                elif brief is None:
                    d = _no_upstream(row)
                elif brief.task == "find_missing":
                    d = _find_missing(brief, backend, agent, cfg)
                else:
                    night_act = None
                    cam, st = (None, None)
                    try:
                        cam, st = night.scene_stats(backend, brief.cow_id, brief.day, brief.minute, brief.visible_frac, cfg.night)
                    except Exception:
                        pass
                    was_night = brief.is_night
                    if st and st["is_night"] and not brief.is_night:
                        brief = brief.model_copy(update={"is_night": True})
                    if st:
                        night_act = {"actor": "verifier", "action": "night_check", "args": {"camera": cam},
                                     "reason": "decide day/night from the frame itself so night handling does not depend on the clock",
                                     "outcome": f"brightness={st['brightness']} saturation={st['saturation']} ir_like={st['ir_like']} -> night={brief.is_night}",
                                     "source": ""}
                    g = gate_map.get((cow, row["t_abs"])) or gate.decide(brief)
                    if brief.is_night and not was_night and g.action == "keep_tier1":
                        g = gate.decide(brief)                  # re-gate with the night rules now that we know it is night
                    ps_acts, ps_reasons, ps_flags = [], [], []
                    if g.action != "auto_abstain":
                        ps_reasons, ps_acts = prescreen(brief, backend, cfg, verifier)    # always: cheap, and it records the check
                        ps_flags = [r for r in ps_reasons if r.startswith(("rule_", "yolo_flag"))]
                        if g.action == "agent":
                            ps_reasons = []
                        if ps_reasons:
                            with lock:
                                ok = spare["n"] is None or spare["n"] > 0
                                if ok and spare["n"] is not None:
                                    spare["n"] -= 1
                            if ok:
                                g = GateResult("agent", g.reasons + ps_reasons)
                                ps_reasons = []
                    if g.action == "agent":
                        flagged = [r for r in (g.reasons + ps_flags) if r.startswith(("rule_", "yolo_flag"))]
                        if flagged:
                            brief = brief.model_copy(update={"yolo_flags": list(dict.fromkeys(brief.yolo_flags + flagged))})
                        d = agent.run(brief, g)
                        if ps_acts:
                            d.actions = [{"seq": 0, **a} for a in ps_acts] + d.actions
                            for i, a in enumerate(d.actions):
                                a["seq"] = i + 1
                    else:
                        d = _passthrough(brief, g, ps_acts, ps_reasons, night_act)
                chain.observe(brief, row, d, cfg)
                cdb.write_decision(d)
                with lock:
                    counts[d.reason] += 1
                    done["n"] += 1
                    if progress_every and done["n"] % progress_every == 0:
                        print(f"  processed {done['n']} cow-minutes ({time.time() - t0:.0f}s)", flush=True)
        finally:
            cdb.close()

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        list(ex.map(process_cow, cows))          # list() re-raises worker exceptions

    db = Store(store_dir)          # fresh read: workers wrote the minute files
    n_bouts = len(db.rebuild_bouts())
    st = db.status()
    db.close()
    return {"processed": done["n"], "gate_escalations": n_gate_agent, "by_reason_this_run": dict(counts),
            "bouts": n_bouts, "elapsed_s": round(time.time() - t0, 1), "store_status": st}
