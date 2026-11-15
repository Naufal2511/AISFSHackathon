"""Escalation gate: rules only (no VLM). Decides which cow-minutes are worth an agent run.
Actions: keep_tier1 | agent | auto_abstain. Reasons are logged so you can report escalation rate by cause."""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import GateCfg
from .schemas import Brief

RARE = {"drinking", "grooming", "social_interaction", "ruminating_lying", "ruminating_standing"}
CONFUSER_PAIRS = {frozenset(p) for p in [
    ("standing", "feeding"), ("standing", "ruminating_standing"), ("lying", "ruminating_lying"),
    ("walking", "standing"), ("drinking", "standing"), ("grooming", "standing"),
    ("social_interaction", "standing"), ("social_interaction", "feeding"),
]}
PAIR_HINTS = {
    frozenset(("standing", "feeding")): "head-in-bunk fraction (zone_query), head zoom, how long she stays (temporal_window)",
    frozenset(("standing", "ruminating_standing")): "rumination_probe on a burst; stills cannot show chewing",
    frozenset(("lying", "ruminating_lying")): "rumination_probe on a burst; stills cannot show chewing",
    frozenset(("walking", "standing")): "track_history displacement and speed, temporal_window",
    frozenset(("drinking", "standing")): "zone_query head-in-waterer fraction, head zoom",
    frozenset(("grooming", "standing")): "head zoom, get_burst motion, temporal_window",
    frozenset(("social_interaction", "standing")): "neighbors contact seconds, temporal_window",
    frozenset(("social_interaction", "feeding")): "neighbors, zoom head, zone_query",
}


@dataclass
class GateResult:
    action: str                       # keep_tier1 | agent | auto_abstain
    reasons: list[str] = field(default_factory=list)
    priority: float = 0.0             # higher = escalate first when the VLM budget is limited


class EscalationGate:
    def __init__(self, cfg: GateCfg):
        self.c = cfg

    def decide(self, b: Brief) -> GateResult:
        c = self.c
        if b.visible_frac and b.max_visible < c.abstain_vis:
            return GateResult("auto_abstain", ["not_visible_any_camera"])
        r: list[str] = []
        if b.top_prob < c.conf_threshold:
            r.append("low_confidence")
        if b.tier1_entropy > c.entropy_threshold:
            r.append("high_entropy")
        if len(b.tier1_top) >= 2:
            pair = frozenset((b.tier1_top[0][0], b.tier1_top[1][0]))
            if pair in CONFUSER_PAIRS and (b.tier1_top[0][1] - b.tier1_top[1][1]) < c.pair_gap:
                r.append("confuser_pair")
        if b.camera_disagreement:
            r.append("camera_disagreement")
        if b.id_conf < c.min_id_conf:
            r.append("low_id_confidence")
        if b.is_night and b.top_prob < c.night_conf:
            r.append("night_low_confidence")
        if b.is_night and b.max_visible < c.night_vis:
            r.append("night_low_visibility")
        if b.top_label in RARE and b.top_prob < c.rare_conf:
            r.append("rare_class_modest_conf")
        if b.prev_label and b.next_label_tier1 and b.prev_label == b.next_label_tier1 != b.top_label:
            r.append("isolated_blip")
        if not r:
            return GateResult("keep_tier1", [])
        return GateResult("agent", r, priority=len(r) + (1.0 - b.top_prob))


def select_within_budget(results: list[tuple[Brief, GateResult]], max_agent_runs: int):
    """Keep the highest-priority escalations; the rest fall back to tier 1 (reason 'budget_fallback')."""
    agent = sorted([x for x in results if x[1].action == "agent"], key=lambda x: -x[1].priority)
    keep = {id(x[0]) for x in agent[:max_agent_runs]}
    return [(b, g if g.action != "agent" or id(b) in keep else GateResult("keep_tier1", g.reasons + ["over_budget"]))
            for b, g in results]


def pair_hint(b: Brief) -> str:
    if len(b.tier1_top) >= 2:
        return PAIR_HINTS.get(frozenset((b.tier1_top[0][0], b.tier1_top[1][0])), "any tools that separate the candidates")
    return "any tools that separate the candidates"
