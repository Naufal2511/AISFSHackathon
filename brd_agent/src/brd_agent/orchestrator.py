"""The agent loop:  gate -> PLAN -> ACT(tools) -> HYPOTHESIZE -> VERIFY -> [SECOND LOOK -> CRITIQUE -> REVISE] -> RESOLVE.
Hard budgets bound tool calls and VLM calls per cow-minute; every failure path ends in a Decision (never an exception)."""
from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from . import night, prompts
from .backend import DataBackend
from .config import AgentConfig
from .gate import GateResult, pair_hint
from .memory import CowMemory
from .recipes import recipe_hints, required_calls
from .schemas import (TAGS, Brief, Critique, Decision, Hypothesis, Plan, ToolCall, ToolResult, VerifierReport)
from .tools import TOOLS, ToolContext, execute, tool_specs
from .trace import Trace, TraceLogger
from .verifier import Verifier
from .vlm_client import VLMClient

CAM_TOOLS = {"rumination_probe", "get_burst", "compare_to_exemplar"}

PAIR_TOOLS = {
    frozenset(("standing", "feeding")): [("zone_query", {}), ("zoom", {"region": "head"}), ("temporal_window", {})],
    frozenset(("standing", "ruminating_standing")): [("track_history", {}), ("rumination_probe", {}), ("zoom", {"region": "head"})],
    frozenset(("lying", "ruminating_lying")): [("track_history", {}), ("rumination_probe", {})],
    frozenset(("walking", "standing")): [("track_history", {}), ("temporal_window", {})],
    frozenset(("drinking", "standing")): [("zone_query", {}), ("zoom", {"region": "head"})],
    frozenset(("grooming", "standing")): [("zoom", {"region": "head"}), ("get_burst", {}), ("temporal_window", {})],
    frozenset(("social_interaction", "standing")): [("neighbors", {}), ("temporal_window", {}), ("zone_query", {})],
    frozenset(("social_interaction", "feeding")): [("neighbors", {}), ("zone_query", {}), ("zoom", {"region": "head"})],
}


TOOL_WHY = {
    "best_camera": "compare how well each camera sees the cow and pick the clearest",
    "get_frames": "look at the cow in still frames around this minute",
    "zoom": "inspect the head/body region at higher resolution",
    "temporal_window": "see whether the posture or activity changes before/after this moment",
    "get_burst": "measure motion over a short burst (grooming, licking, head movement)",
    "zone_query": "measure whether the head is at the feed bunk or waterer",
    "track_history": "measure speed, displacement and posture from the track",
    "neighbors": "measure distance and contact with nearby cows",
    "rumination_probe": "measure jaw cycling, which a still image cannot show",
    "enhance_night": "enhance contrast of a dark/infrared frame before trusting what it shows",
    "compare_to_exemplar": "check that the crop really is this cow (identity)",
    "cow_history": "read this cow's own recent behaviour from the database as context",
}

# Words in a hypothesis' "needs" that map to tools (only unrun tools are added).
NEEDS_MAP: list[tuple[str, str, dict]] = [
    (r"rumin|chew|jaw|cud", "rumination_probe", {}),
    (r"zoom|head|mouth|muzzle|face|ear|nose", "zoom", {"region": "head"}),
    (r"history|previous|earlier|duration|bout|how long", "cow_history", {}),
    (r"motion|moving|burst|lick|groom|rub", "get_burst", {}),
    (r"neighbo|contact|nearby|social|mount|push", "neighbors", {}),
    (r"bunk|water|zone|feed", "zone_query", {}),
    (r"identity|which cow|id\b|swap", "compare_to_exemplar", {}),
    (r"dark|night|contrast|bright|infrared|\bir\b", "enhance_night", {}),
    (r"before|after|sequence|temporal|time|frames", "temporal_window", {}),
    (r"speed|walk|displace|posture|track", "track_history", {}),
]
# what a pre-verification flag obliges the agent to re-measure
FLAG_TOOLS = [("jaw_motion", "rumination_probe"), ("moving_but_labelled_stationary", "track_history"), ("posture_", "track_history"),
              ("zone_", "zone_query"), ("cue_", "rumination_probe"), ("duplicate_id", "compare_to_exemplar")]
NEEDS_CAM_TOOLS = {"zoom", "get_frames", "rumination_probe", "get_burst", "compare_to_exemplar", "enhance_night"}


def needs_calls(needs: Optional[str], ran: list[ToolResult]) -> list[ToolCall]:
    if not needs or needs.strip().lower() in ("null", "none", "n/a", ""):
        return []
    cam = next((c for c in ("back", "front", "top") if re.search(rf"\b{c}\b", needs, re.I)), None)
    out = []
    for pat, tool, args in NEEDS_MAP:
        if not re.search(pat, needs, re.I) or tool in {c.tool for c in out}:
            continue
        seen = [r for r in ran if r.tool == tool and r.ok]
        # a tool is re-run only to look from a camera it has not used yet
        if seen and not (cam and tool in NEEDS_CAM_TOOLS and all(r.args.get("camera") != cam for r in seen)):
            continue
        if True:
            a = dict(args)
            if cam and tool in NEEDS_CAM_TOOLS:
                a["camera"] = cam
            out.append(ToolCall(tool=tool, args=a, reason=f"the hypothesis said it needs: \"{needs[:140]}\"", source="needs"))
    return out


def short(obj: Any, n: int = 200) -> str:
    s = json.dumps(obj, separators=(",", ":"), default=str)
    return s if len(s) <= n else s[:n] + "..."


@dataclass
class ActionLog:
    """Every action the agent takes, each with the reason for taking it (-> Decision.actions -> agent_actions table)."""
    items: list[dict] = field(default_factory=list)

    def add(self, actor: str, action: str, reason: str, args: Optional[dict] = None, outcome: str = "", source: str = ""):
        self.items.append({"seq": len(self.items) + 1, "actor": actor, "action": action,
                           "args": args or {}, "reason": (reason or "").strip() or "(no reason supplied)",
                           "outcome": outcome, "source": source})


class BudgetExceeded(Exception):
    pass


@dataclass
class Budget:
    tools_left: int
    vlm_left: int
    tools_used: int = 0
    vlm_used: int = 0
    tokens_in: int = 0
    tokens_out: int = 0

    def spend_tool(self) -> bool:
        if self.tools_left <= 0:
            return False
        self.tools_left -= 1; self.tools_used += 1
        return True

    def spend_vlm(self) -> bool:
        if self.vlm_left <= 0:
            return False
        self.vlm_left -= 1; self.vlm_used += 1
        return True


@dataclass
class Evidence:
    results: list[ToolResult] = field(default_factory=list)
    log: ActionLog = field(default_factory=ActionLog)

    def add(self, r: ToolResult):
        self.results.append(r)

    def facts(self, since: int = 0) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for r in self.results[since:]:
            out[r.tool] = r.facts if r.ok else {"_error": r.error}
        return out

    def facts_json(self, since: int = 0, cap: int = 4000) -> str:
        s = json.dumps(self.facts(since), separators=(",", ":"), default=str)
        return s if len(s) <= cap else s[:cap] + '..."truncated"'

    def images(self, k: int, since: int = 0) -> list[tuple[str, str]]:
        flat = [(p, c) for r in self.results[since:] if r.ok for p, c in zip(r.images, r.captions or [""] * len(r.images))]
        return flat[-k:]

    def tools_run(self) -> set[str]:
        return {r.tool for r in self.results if r.ok}

    def cameras_used(self) -> set[str]:
        cams = set()
        for r in self.results:
            c = r.facts.get("camera") if r.ok else None
            if c in ("back", "front", "top"):
                cams.add(c)
        return cams


TIER1_WHY = {
    "auto_abstain": "no camera sees the cow well enough, so no behaviour can be judged",
    "keep_tier1": "YOLO's label was confident and no risk signal fired; the rule checks found nothing to contradict it",
    "budget_fallback": "the VLM/tool budget ran out before the question was settled; YOLO's label kept at reduced confidence",
    "error_fallback": "the agent hit an error; YOLO's label kept at reduced confidence",
    "unresolved_fallback_tier1": "the evidence contradicted the proposed label and could not be settled; YOLO's label kept at reduced confidence",
}


def tier1_decision(b: Brief, reason: str, flags: Optional[list[str]] = None) -> Decision:
    flags = flags or []
    why = TIER1_WHY.get(reason, reason)
    if reason == "auto_abstain":
        return Decision(cow_id=b.cow_id, day=b.day, minute=b.minute, behavior="other_not_visible",
                        confidence=min(0.95, 1.0 - b.max_visible), source="not_visible", tier=1, reason=reason,
                        tier1_label=b.top_label, flags=flags, rationale=why, correction="abstained", is_night=b.is_night)
    return Decision(cow_id=b.cow_id, day=b.day, minute=b.minute, behavior=b.top_label or "other_not_visible",
                    confidence=b.top_prob if reason == "keep_tier1" else min(b.top_prob, 0.4),
                    source="model_estimate", tier=1, reason=reason, tier1_label=b.top_label, flags=flags,
                    rationale=why, correction="confirmed" if reason == "keep_tier1" else "unresolved", is_night=b.is_night)


def rules_plan(b: Brief, max_steps: int) -> Plan:
    steps: list[tuple[str, dict]] = []
    if b.is_night:
        steps.append(("enhance_night", {}))
    vis = list(b.visible_frac.values())
    if vis and max(vis) - min(vis) > 0.3:
        steps.append(("best_camera", {}))
    if len(b.tier1_top) >= 2:
        steps += PAIR_TOOLS.get(frozenset((b.tier1_top[0][0], b.tier1_top[1][0])), [])
    if len(steps) < 2:
        steps += [("zone_query", {}), ("track_history", {}), ("zoom", {"region": "head"})]
    if b.id_conf < 0.6:
        steps.append(("compare_to_exemplar", {}))
    top2 = " vs ".join(l for l, _ in b.tier1_top[:2]) or "unknown"
    calls = _dedupe([ToolCall(tool=t, args=a, source="planner",
                              reason=("the frame is dark/infrared, so enhance it before trusting images" if t == "enhance_night"
                                      else f"YOLO is torn between {top2}; {TOOL_WHY[t]}")) for t, a in steps])[:max_steps]
    return Plan(rationale=f"rules-based plan: gather evidence that separates {top2}", steps=calls)


def _dedupe(calls: list[ToolCall]) -> list[ToolCall]:
    seen, out = set(), []
    for c in calls:
        if c.tool not in TOOLS:
            continue
        k = (c.tool, json.dumps(c.args, sort_keys=True, default=str))
        if k not in seen:
            seen.add(k); out.append(c)
    return out


class BehaviorAgent:
    def __init__(self, cfg: AgentConfig, backend: DataBackend, vlm: VLMClient,
                 trace_logger: Optional[TraceLogger] = None, memory: Optional[CowMemory] = None,
                 workdir: Optional[str] = None, history: Optional[Any] = None):
        self.cfg, self.backend, self.vlm = cfg, backend, vlm
        self.history = history if cfg.history.enabled else None
        self.tl = trace_logger or TraceLogger(None)
        self.mem = memory or CowMemory()
        self.verifier = Verifier(cfg.verifier, cfg.night)
        self.system = prompts.system_prompt(cfg.prompt_version, cfg.behaviors_path)
        self.workdir = Path(workdir or tempfile.mkdtemp(prefix="agent_work_"))
        self.workdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ public
    def run(self, brief: Brief, gate: Optional[GateResult] = None) -> Decision:
        if brief.prev_label is None:
            brief = brief.model_copy(update={"prev_label": self.mem.prev_label(brief.cow_id, brief.day, brief.minute)})
        ev = Evidence()
        tr = self.tl.start(brief)
        if gate:
            tr.add("gate", action=gate.action, reasons=gate.reasons)
            ev.log.add("gate", f"gate:{gate.action}", "escalation rules: " + (", ".join(gate.reasons) or "nothing suspicious"),
                       outcome=gate.action)
        brief = self._night_check(brief, ev, tr)
        budget = Budget(self.cfg.budget.max_tool_calls, self.cfg.budget.max_vlm_calls)
        ctx = ToolContext(self.backend, brief, self.cfg, self.workdir, history=self.history)
        try:
            decision = self._loop(brief, ctx, ev, budget, tr)
        except BudgetExceeded as e:
            tr.add("budget_exceeded", where=str(e))
            decision = tier1_decision(brief, "budget_fallback", ["budget_exceeded"])
        except Exception as e:   # VLM/network/parse failure: never crash the batch
            tr.add("error", error=f"{type(e).__name__}: {e}")
            decision = tier1_decision(brief, "error_fallback", [f"error:{type(e).__name__}"])
            decision.rationale += f" [{type(e).__name__}: {str(e)[:300]}]"
        decision.n_tool_calls, decision.n_vlm_calls = budget.tools_used, budget.vlm_used
        decision.tokens_in, decision.tokens_out = budget.tokens_in, budget.tokens_out
        decision.used_history = "cow_history" in ev.tools_run()
        decision.provider, decision.model = getattr(self.vlm, "provider", None), getattr(self.vlm, "model", None)
        decision.prompt_version = self.cfg.prompt_version
        decision.cameras_used = sorted(ev.cameras_used())
        decision.trace_id = tr.trace_id
        decision.tier1_label = brief.top_label
        decision.is_night = brief.is_night
        self._identity_outcome(decision, ev)
        ev.log.add("resolver", "final_decision", decision.rationale or decision.reason,
                   args={"behavior": decision.behavior, "confidence": decision.confidence},
                   outcome=f"{decision.correction}: {brief.top_label} -> {decision.behavior}")
        decision.actions = ev.log.items
        tr.add("decision", behavior=decision.behavior, confidence=decision.confidence, reason=decision.reason,
               correction=decision.correction, rationale=decision.rationale)
        self.mem.update(brief.cow_id, brief.day, brief.minute, decision.behavior, decision.confidence)
        self.tl.finish(tr, decision)
        return decision

    def _night_check(self, b: Brief, ev: Evidence, tr: Trace) -> Brief:
        """Decide night/day from the FRAME (brightness, colour saturation), not from the clock or an upstream flag."""
        try:
            cam, st = night.scene_stats(self.backend, b.cow_id, b.day, b.minute, b.visible_frac, self.cfg.night)
        except Exception:
            cam, st = None, None
        if not st:
            ev.log.add("agent", "night_check", "no scene frame available to measure brightness; keeping the upstream night flag",
                       outcome=f"is_night={b.is_night} (unmeasured)")
            return b
        was = b.is_night
        now = bool(was or st["is_night"])
        ev.log.add("agent", "night_check", "measure brightness and colour saturation of the actual frame, so night/infrared handling "
                   "does not depend on the clock or an upstream flag", args={"camera": cam},
                   outcome=f"brightness={st['brightness']}, saturation={st['saturation']}, ir_like={st['ir_like']} -> night={now}"
                           + (" (upgraded from day)" if now and not was else ""))
        tr.add("night_check", camera=cam, **{**st, "is_night": now})
        return b.model_copy(update={"is_night": now}) if now != was else b

    def _identity_outcome(self, d: Decision, ev: Evidence):
        for r in reversed(ev.results):
            if r.tool == "compare_to_exemplar" and r.ok:
                d.id_verified = bool(r.facts.get("match")) and not r.facts.get("better_match")
                bm = r.facts.get("better_match")
                if bm:
                    d.id_suggested = bm
                    d.flags.append(f"id_swap_suspected:{bm}")
                break

    # ------------------------------------------------------------------ loop
    def _loop(self, b: Brief, ctx: ToolContext, ev: Evidence, budget: Budget, tr: Trace) -> Decision:
        plan = self._plan(b, budget, tr, ev)
        self._act(plan.steps, ctx, ev, budget, tr)
        h1 = self._hypothesize(b, ev, budget, tr, "first", self.cfg.blind_first_look)
        r1 = self._verify(h1, b, ev, tr, "first")

        # needs-driven round: if the model says what evidence would settle it, go and get exactly that
        extra = needs_calls(h1.needs, ev.results) if (h1.confidence < 0.85 or r1.hard_failures) else []
        if extra and budget.tools_left >= 1 and budget.vlm_left >= 3:
            ev.log.add("agent", "needs_round", f"the first hypothesis asked for more evidence: \"{(h1.needs or '')[:160]}\"",
                       outcome=f"running {[c.tool for c in extra]}", source="needs")
            tr.add("needs_round", needs=h1.needs, steps=[c.model_dump() for c in extra])
            self._act(extra, ctx, ev, budget, tr)
            h1 = self._hypothesize(b, ev, budget, tr, "after_needs", self.cfg.blind_first_look)
            r1 = self._verify(h1, b, ev, tr, "after_needs")
        final_h, final_r, reason = h1, r1, "agent_accept"

        disagrees = self.cfg.blind_first_look and b.top_label is not None and h1.label != b.top_label
        if r1.hard_failures or h1.confidence < self.cfg.critique_below or disagrees:
            if budget.vlm_left >= 1 and budget.tools_left >= 1:
                why = []
                if r1.hard_failures:
                    why.append("hard check failed: " + ", ".join(c.name for c in r1.hard_failures))
                if h1.confidence < self.cfg.critique_below:
                    why.append(f"confidence {h1.confidence:.2f} < {self.cfg.critique_below}")
                if disagrees:
                    why.append(f"independent look says '{h1.label}' but YOLO said '{b.top_label}'")
                ev.log.add("agent", "escalate_second_look", "; ".join(why), outcome="second look + critique", source="second_look")
                mark = len(ev.results)
                self._second_look(b, h1, ctx, ev, budget, tr)
                r1b = self._verify(h1, b, ev, tr, "after_second_look")
                crit = self._critique(b, h1, r1b, ev, mark, budget, tr)
                if crit.verdict == "accept" and not r1b.hard_failures:
                    final_h, final_r = h1, r1b
                elif budget.vlm_left >= 1:
                    self._critic_camera(crit, ctx, ev, budget, tr)
                    h2 = self._hypothesize(b, ev, budget, tr, "revised", self.cfg.blind_first_look, hint=crit.best_alternative)
                    r2 = self._verify(h2, b, ev, tr, "revised")
                    final_h, final_r, reason = (h2, r2, "agent_revised") if (not r2.hard_failures or r1b.hard_failures) else (h1, r1b, "agent_accept")
                else:
                    final_h, final_r = h1, r1b
                    reason = "budget_fallback"
        return self._resolve(b, final_h, final_r, reason, disagrees)

    def _critic_camera(self, crit: Critique, ctx: ToolContext, ev: Evidence, budget: Budget, tr: Trace):
        cam = crit.requested_camera
        if cam in ("back", "front", "top") and cam not in ev.cameras_used() and budget.tools_left >= 1:
            calls = [ToolCall(tool="zoom", args={"camera": cam, "region": "head"}, source="critic_request",
                              reason=f"the critic asked for the {cam} view: {crit.reason[:140]}"),
                     ToolCall(tool="get_frames", args={"camera": cam}, source="critic_request",
                              reason=f"the critic asked for the {cam} view: a still sequence from that camera")]
            self._act(calls, ctx, ev, budget, tr)

    # ------------------------------------------------------------------ stages
    def _ask(self, task: str, user: str, images: list[str], schema, budget: Budget, tr: Trace, meta: dict):
        if not budget.spend_vlm():
            raise BudgetExceeded(task)
        obj, info = self.vlm.ask(task, self.system, user, images, schema, meta)
        budget.tokens_in += int(info.get("prompt_tokens") or 0)
        budget.tokens_out += int(info.get("completion_tokens") or 0)
        tr.add("vlm", task=task, **{k: info.get(k) for k in ("prompt_tokens", "completion_tokens", "latency_s", "cached", "repaired")})
        return obj

    def _plan(self, b: Brief, budget: Budget, tr: Trace, ev: Evidence) -> Plan:
        max_steps = budget.tools_left
        if self.cfg.planner == "llm" and budget.vlm_left > 0:
            try:
                top2 = ", ".join(f"{l} ({p:.2f})" for l, p in b.tier1_top[:2]) or "unknown"
                user = prompts.render(self.cfg.prompt_version, "plan", brief_json=b.model_dump_json(), top2=top2,
                                      pair_hint=pair_hint(b), recipe_hints=recipe_hints(b), tool_specs=json.dumps(tool_specs()), max_steps=max_steps)
                plan = self._ask("plan", user, [], Plan, budget, tr, {"brief": b, "facts": {}})
                steps = _dedupe(plan.steps)[:max_steps]
                if steps:
                    for st in steps:
                        if not st.reason.strip():
                            st.reason = f"(planner gave no reason) {TOOL_WHY.get(st.tool, 'gather evidence')}"
                        st.source = "planner"
                    steps, added = self._augment(steps, b, max_steps)
                    tr.add("plan", source="llm", rationale=plan.rationale, steps=[s.model_dump() for s in steps],
                           mandatory_added=added)
                    ev.log.add("vlm", "plan", plan.rationale or "decide what evidence is needed to verify YOLO's claim",
                               outcome=f"{len(steps)} steps: {[s.tool for s in steps]}" + (f"; guardrail added {added}" if added else ""))
                    return Plan(rationale=plan.rationale, steps=steps)
                tr.add("plan_empty_fallback_to_rules")
            except BudgetExceeded:
                raise
            except Exception as e:
                tr.add("plan_failed_fallback_to_rules", error=f"{type(e).__name__}: {e}")
        plan = rules_plan(b, max_steps)
        steps, added = self._augment(plan.steps, b, max_steps)
        tr.add("plan", source="rules", rationale=plan.rationale, steps=[s.model_dump() for s in steps], mandatory_added=added)
        ev.log.add("planner", "plan", plan.rationale, outcome=f"{len(steps)} steps: {[s.tool for s in steps]}")
        return Plan(rationale=plan.rationale, steps=steps)

    def _augment(self, steps: list[ToolCall], b: Brief, max_steps: int) -> tuple[list[ToolCall], list[str]]:
        """Guardrail: a planner may not skip the tools a top-2 candidate label needs (recipes.py), nor skip night enhancement."""
        have = {s.tool for s in steps}
        need: list[ToolCall] = []
        for lab, _ in b.tier1_top[:2]:
            for call in required_calls(lab, include_history=self.history is not None):
                if call.tool not in have and call.tool not in {n.tool for n in need}:
                    call.reason = f"guardrail: '{lab}' is a top-2 candidate and needs {call.tool} to be confirmed or ruled out"
                    call.source = "guardrail"
                    need.append(call)
        for flag in b.yolo_flags:
            for key, tool in FLAG_TOOLS:
                if key in flag and tool not in have and tool not in {n.tool for n in need}:
                    need.append(ToolCall(tool=tool, source="guardrail",
                                         reason=f"guardrail: the pre-verification raised '{flag}', so {TOOL_WHY[tool]}"))
        if b.id_conf < self.cfg.verifier.min_id_conf and "compare_to_exemplar" not in have \
                and "compare_to_exemplar" not in {n.tool for n in need}:
            need.append(ToolCall(tool="compare_to_exemplar", source="guardrail",
                                 reason=f"guardrail: identity confidence is low ({b.id_conf:.2f}); verify the crop is really {b.cow_id}"))
        if b.is_night and "enhance_night" not in have:
            need.append(ToolCall(tool="enhance_night", source="guardrail",
                                 reason="guardrail: the frame is dark/infrared, so contrast must be enhanced before images are judged"))
        if not need:
            return steps, []
        return _dedupe(need + steps)[:max_steps], [c.tool for c in need]

    def _act(self, calls: list[ToolCall], ctx: ToolContext, ev: Evidence, budget: Budget, tr: Trace):
        for call in calls:
            if not call.reason.strip():
                call.reason = TOOL_WHY.get(call.tool, "gather evidence")
            if not budget.spend_tool():
                tr.add("tool_budget_exhausted")
                ev.log.add("agent", "tool_budget_exhausted", "the per-minute tool budget is used up", outcome=f"skipped {call.tool}")
                break
            res = execute(ctx, call)
            ev.add(res)
            tr.add("tool", tool=res.tool, args=res.args, ok=res.ok, facts=res.facts, error=res.error,
                   images=res.images, captions=res.captions, reason=res.reason, source=res.source)
            ev.log.add("tool", f"tool:{res.tool}", res.reason, args=res.args, source=res.source,
                       outcome=("ok " + short(res.facts)) if res.ok else f"error: {res.error}")

    def _manifest(self, imgs: list[tuple[str, str]]) -> str:
        return "\n".join(f"{i + 1}. {c or 'image'}" for i, (_, c) in enumerate(imgs)) or "(no images)"

    def _hypothesize(self, b: Brief, ev: Evidence, budget: Budget, tr: Trace, stage: str, blind: bool,
                     hint: Optional[str] = None) -> Hypothesis:
        imgs = ev.images(self.cfg.max_images)
        user = prompts.render(
            self.cfg.prompt_version, "hypothesis", cow_id=b.cow_id, day=b.day, minute=b.minute, is_night=b.is_night,
            prev_label=b.prev_label or "unknown",
            tier1_block="" if blind else f"- tier-1 guess (may be wrong): {b.tier1_top[:3]}\n",
            hint_block=(f"- a reviewer suggests the alternative '{hint}'. Test it against the evidence; do not assume it is right.\n" if hint else ""),
            evidence_json=ev.facts_json(), image_manifest=self._manifest(imgs))
        h = self._ask("hypothesis", user, [p for p, _ in imgs], Hypothesis, budget, tr,
                      {"brief": b, "facts": ev.facts(), "hint": hint})
        tr.add("hypothesis", stage=stage, **h.model_dump(mode="json"))
        ev.log.add("vlm", f"hypothesis:{stage}",
                   "independent look at the evidence without YOLO's guess" if blind else "judge the evidence (YOLO's guess shown)",
                   outcome=f"{h.label} conf={h.confidence:.2f}; {h.rationale or '; '.join(h.cues[:3])}")
        return h

    def _verify(self, h: Hypothesis, b: Brief, ev: Evidence, tr: Trace, stage: str) -> VerifierReport:
        rep = self.verifier.run(h, b, ev.facts())
        tr.add("verifier", stage=stage, checks=[c.model_dump() for c in rep.checks],
               n_hard=len(rep.hard_failures), n_soft=len(rep.soft_failures))
        bad = [f"{c.severity}:{c.name}" for c in rep.checks if not c.passed]
        ev.log.add("verifier", f"verify:{stage}", "deterministic rule check of the label against measurements, independent of the VLM",
                   outcome="all checks passed" if not bad else "failed " + ", ".join(bad))
        return rep

    def _second_look(self, b: Brief, h: Hypothesis, ctx: ToolContext, ev: Evidence, budget: Budget, tr: Trace):
        """Gather DIFFERENT evidence: an unused camera plus whatever tools can verify the proposed label."""
        vis = ctx.visibility()
        used = ev.cameras_used()
        cams = sorted(vis, key=vis.get, reverse=True)
        cam = next((c for c in cams if c not in used and vis[c] >= self.cfg.verifier.min_visible), cams[0] if cams else "auto")
        fresh = cam not in used
        run = ev.tools_run()
        calls = [ToolCall(tool=c.tool, args={**c.args, **({"camera": cam} if c.tool in CAM_TOOLS else {})}, source="second_look",
                          reason=f"proposed label '{h.label}' requires {c.tool}, which has not run yet")
                 for c in required_calls(h.label, include_history=self.history is not None) if c.tool not in run]
        if b.id_conf < self.cfg.verifier.min_id_conf and "compare_to_exemplar" not in run:
            calls.append(ToolCall(tool="compare_to_exemplar", args={"camera": cam}, source="second_look",
                                  reason=f"identity confidence is low ({b.id_conf:.2f}); compare the crop to this cow's gallery from {cam}"))
        view = f"{cam} ({'a camera not used before' if fresh else 'best remaining view'})"
        calls += [ToolCall(tool="get_frames", args={"camera": cam}, source="second_look",
                           reason=f"a second, independent look from the {view}, so one blind spot cannot decide the label"),
                  ToolCall(tool="zoom", args={"camera": cam, "region": "head"}, source="second_look",
                           reason=f"inspect the head from the {view} to test the cues behind '{h.label}'")]
        tr.add("second_look", camera=cam, steps=[c.model_dump() for c in calls])
        self._act(calls, ctx, ev, budget, tr)

    def _critique(self, b: Brief, h: Hypothesis, rep: VerifierReport, ev: Evidence, mark: int,
                  budget: Budget, tr: Trace) -> Critique:
        imgs = ev.images(self.cfg.max_images, since=mark)
        user = prompts.render(
            self.cfg.prompt_version, "critique", cow_id=b.cow_id, day=b.day, minute=b.minute,
            hypothesis_json=h.model_dump_json(),
            verifier_json=json.dumps([c.model_dump() for c in rep.checks if not c.passed]) or "[]",
            tier1_block=f"TIER-1 GUESS (cheap model, may be wrong): {b.tier1_top[:3]}\n",
            evidence_json=ev.facts_json(since=mark), image_manifest=self._manifest(imgs))
        c = self._ask("critique", user, [p for p, _ in imgs], Critique, budget, tr,
                      {"brief": b, "facts": ev.facts(), "hard_failed": bool(rep.hard_failures)})
        tr.add("critique", **c.model_dump(mode="json"))
        ev.log.add("vlm", "critique", "argue against the proposed label using the new evidence before it is accepted",
                   outcome=f"{c.verdict}" + (f" (alt={c.best_alternative})" if c.best_alternative else "") + f": {c.reason}")
        return c

    def _explain(self, b: Brief, h: Hypothesis, r: VerifierReport, conf: float, reason: str) -> tuple[str, str]:
        L, yolo = h.label, b.top_label
        cues = "; ".join(h.cues[:3]) or "no specific cue recorded"
        why = h.rationale or f"cues: {cues}"
        checks = "all automatic checks passed" if not r.soft_failures else "soft flags: " + ", ".join(c.name for c in r.soft_failures)
        night_note = " Night/infrared frame: confidence capped and fine cues need a measurement." if b.is_night else ""
        if b.task == "find_missing":
            if L == "other_not_visible":
                return f"YOLO produced no record for this cow-minute and the video does not show her clearly enough to label.{night_note}", "miss_not_found"
            return f"YOLO had no record for this cow-minute; recovered from the video: '{L}' ({why}); {checks}.{night_note}", "recovered_miss"
        if L == "other_not_visible":
            return f"Not visible enough to judge (YOLO said '{yolo}'): {why}.{night_note}", "abstained"
        if L == yolo:
            return f"YOLO's '{yolo}' confirmed: {why}; {checks}.{night_note}", "confirmed"
        return f"YOLO said '{yolo}' but the evidence supports '{L}': {why}; {checks}.{night_note}", "behavior_corrected"

    def _resolve(self, b: Brief, h: Hypothesis, r: VerifierReport, reason: str, disagrees: bool) -> Decision:
        flags = [f"soft:{c.name}" for c in r.soft_failures]
        if disagrees:
            flags.append("disagrees_with_tier1")
        if b.is_night:
            flags.append("night")
        if r.hard_failures:
            flags += [f"hard:{c.name}" for c in r.hard_failures]
            names = ", ".join(c.name for c in r.hard_failures)
            if any(c.name == "visibility" for c in r.hard_failures):
                return Decision(cow_id=b.cow_id, day=b.day, minute=b.minute, behavior="other_not_visible", confidence=0.5,
                                source="not_visible", reason="unresolved_fallback_tier1", flags=flags, is_night=b.is_night,
                                correction="abstained",
                                rationale=f"The cow is not visible enough in any camera to support '{h.label}' (failed: {names}).")
            d = tier1_decision(b, "unresolved_fallback_tier1", flags)
            d.tier = 2
            d.rationale = f"The proposed '{h.label}' was contradicted by measurements ({names}) and could not be repaired; YOLO's '{b.top_label}' kept at low confidence."
            return d
        conf = h.confidence * (self.cfg.verifier.soft_penalty ** len(r.soft_failures))
        if b.is_night:
            conf = min(conf, self.cfg.night.max_confidence)
        conf = round(conf, 4)
        tags = [t for t in h.tags if (not TAGS[t]["applies_to"] or h.label in TAGS[t]["applies_to"])] if h.label != "other_not_visible" else []
        flags += [f"clinical_candidate:{t}" for t in tags if TAGS[t]["clinical"]]
        rationale, correction = self._explain(b, h, r, conf, reason)
        return Decision(cow_id=b.cow_id, day=b.day, minute=b.minute, behavior=h.label, confidence=conf,
                        source="not_visible" if h.label == "other_not_visible" else "model_estimate",
                        reason=reason, flags=flags, behavior_tags=tags, rationale=rationale, correction=correction,
                        is_night=b.is_night)
