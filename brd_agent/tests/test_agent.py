import json
from pathlib import Path

import numpy as np
import pytest

from brd_agent.backend import MockBackend
from brd_agent.config import AgentConfig
from brd_agent.gate import EscalationGate, GateResult, select_within_budget
from brd_agent.orchestrator import BehaviorAgent
from brd_agent.schemas import Brief, Hypothesis, ToolCall
from brd_agent.tools import ToolContext, execute, validate_args, TOOLS
from brd_agent.trace import TraceLogger
from brd_agent.verifier import Verifier
from brd_agent.vlm_client import MockVLM, default_mock_policy, extract_json

ROOT = Path(__file__).resolve().parents[1]


def cfg(**kw) -> AgentConfig:
    c = AgentConfig(behaviors_path=str(ROOT / "configs" / "behaviors.yaml"))
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def brief(top, minute=100, **kw) -> Brief:
    base = dict(cow_id="C01", day=1, minute=minute, tier1_top=top, tier1_entropy=1.3,
                visible_frac={"back": 0.3, "front": 0.8, "top": 0.6}, id_conf=0.9)
    base.update(kw)
    return Brief(**base)


def agent(scen=None, policy=None, c=None, tl=None):
    return BehaviorAgent(c or cfg(), MockBackend(scen), MockVLM(policy), trace_logger=tl)


# ---------------------------------------------------------------- happy paths
def test_feeding_resolved_and_trace_written(tmp_path):
    scen = {("C01", 1, 100): {"head_in_bunk_frac": 0.8, "zone_fractions": {"feed_bunk": 0.8, "waterer": 0, "resting": 0, "open": 0.2}}}
    d = agent(scen, tl=TraceLogger(tmp_path)).run(brief([("standing", 0.41), ("feeding", 0.33)]))
    assert d.behavior == "feeding" and d.tier == 2 and d.confidence > 0.5
    assert d.n_vlm_calls <= 4 and d.n_tool_calls <= 6
    tr = json.loads((tmp_path / f"{d.trace_id}.json").read_text(encoding="utf-8"))
    assert [e["type"] for e in tr["events"]][:3] == ["night_check", "vlm", "plan"]
    assert (tmp_path / f"{d.trace_id}.md").read_text(encoding="utf-8").startswith("# Trace")


def test_rumination_confirmed_by_probe():
    scen = {("C01", 1, 100): {"jaw_motion_score": 0.7, "periodicity_hz": 0.9, "posture_mode": "lying", "speed_mean": 1.0}}
    d = agent(scen).run(brief([("lying", 0.5), ("ruminating_lying", 0.4)], posture="lying"))
    assert d.behavior == "ruminating_lying"


# ---------------------------------------------------------------- verifier
def test_verifier_flags_hallucinated_chewing():
    h = Hypothesis(label="standing", confidence=0.8, cues=["jaw chewing visible"])
    rep = Verifier(cfg().verifier).run(h, brief([("standing", 0.5)]), {"rumination_probe": {"jaw_motion_score": 0.0, "rumination_likely": False}})
    assert any(c.name == "cue_chewing_hallucinated" for c in rep.hard_failures)


def test_negated_cues_are_not_claims():
    rp = {"rumination_probe": {"jaw_motion_score": 0.0, "rumination_likely": False}}
    for cue in ["no visible chewing", "not chewing", "no jaw motion visible", "without any cud chewing", "cannot see jaw movement"]:
        rep = Verifier(cfg().verifier).run(Hypothesis(label="lying", confidence=0.8, cues=[cue, "body flat on bedding"]), brief([("lying", 0.5)]), rp)
        assert not any(c.name == "cue_chewing_hallucinated" for c in rep.hard_failures), cue
    rep = Verifier(cfg().verifier).run(Hypothesis(label="lying", confidence=0.8, cues=["chewing visible"]), brief([("lying", 0.5)]), rp)
    assert any(c.name == "cue_chewing_hallucinated" for c in rep.hard_failures)


def test_verifier_zone_and_posture_rules():
    v = Verifier(cfg().verifier)
    b = brief([("standing", 0.5)])
    assert v.run(Hypothesis(label="feeding", confidence=.8), b, {"zone_query": {"head_in_bunk_frac": 0.0}}).hard_failures
    assert v.run(Hypothesis(label="lying", confidence=.8), b, {"track_history": {"posture_mode": "standing", "speed_mean": 1}}).hard_failures
    assert not v.run(Hypothesis(label="standing", confidence=.8), b, {"track_history": {"posture_mode": "standing", "speed_mean": 1}}).hard_failures


# ---------------------------------------------------------------- self-check + revision
def test_hard_failure_triggers_critique_and_revision():
    n = {"hyp": 0}

    def policy(task, user, images, meta):
        t = task.split(":")[0]
        if t == "hypothesis":
            n["hyp"] += 1
            if n["hyp"] == 1:   # first look hallucinates feeding although cow is not at the bunk
                return {"label": "feeding", "confidence": 0.85, "cues": ["head at bunk"], "alternatives": []}
            return {"label": "standing", "confidence": 0.8, "cues": ["upright"], "alternatives": []}
        if t == "critique":
            return {"verdict": "revise", "best_alternative": "standing", "reason": "not in bunk zone"}
        return default_mock_policy(task, user, images, meta)

    d = agent({("C01", 1, 100): {"head_in_bunk_frac": 0.0}}, policy).run(brief([("standing", 0.41), ("feeding", 0.33)]))
    assert d.behavior == "standing" and d.reason == "agent_revised"


def test_unresolved_falls_back_to_tier1_with_low_confidence():
    def policy(task, user, images, meta):
        t = task.split(":")[0]
        if t == "hypothesis":
            return {"label": "feeding", "confidence": 0.9, "cues": ["head at bunk"]}
        if t == "critique":
            return {"verdict": "accept"}
        return default_mock_policy(task, user, images, meta)

    d = agent({}, policy).run(brief([("standing", 0.41), ("feeding", 0.33)]))
    assert d.reason == "unresolved_fallback_tier1" and d.behavior == "standing" and d.confidence <= 0.4
    assert any(f.startswith("hard:") for f in d.flags)


# ---------------------------------------------------------------- robustness
def test_budget_exhaustion_does_not_crash():
    c = cfg(); c.budget.max_vlm_calls = 1      # plan uses it; hypothesis cannot run
    d = agent({}, c=c).run(brief([("standing", 0.41), ("feeding", 0.33)]))
    assert d.reason == "budget_fallback" and d.tier == 1


def test_invalid_json_is_repaired():
    def policy(task, user, images, meta):
        if task.startswith("hypothesis") and ":repair" not in task:
            return "sorry, I think she is eating"           # not JSON
        return default_mock_policy(task, user, images, meta)

    d = agent({}, policy).run(brief([("standing", 0.5), ("lying", 0.3)]))
    assert d.reason in ("agent_accept", "agent_revised") and d.behavior in ("standing", "lying", "walking")


def test_vlm_exception_becomes_error_fallback():
    def policy(task, user, images, meta):
        raise RuntimeError("server down")

    c = cfg(); c.planner = "rules"
    d = agent({}, policy, c=c).run(brief([("standing", 0.41), ("feeding", 0.33)]))
    assert d.reason == "error_fallback"


def test_rules_planner_uses_no_vlm_call_for_planning():
    c = cfg(); c.planner = "rules"
    vlm = MockVLM()
    a = BehaviorAgent(c, MockBackend({("C01", 1, 100): {"head_in_bunk_frac": 0.8}}), vlm)
    a.run(brief([("standing", 0.41), ("feeding", 0.33)]))
    assert all(t != "plan" for t, _ in vlm.calls)


# ---------------------------------------------------------------- gate / tools / utils
def test_gate_actions():
    g = EscalationGate(cfg().gate)
    assert g.decide(brief([("standing", 0.3)], visible_frac={"back": 0.0, "front": 0.02})).action == "auto_abstain"
    assert g.decide(brief([("lying", 0.95), ("standing", 0.03)], tier1_entropy=0.2)).action == "keep_tier1"
    r = g.decide(brief([("standing", 0.45), ("feeding", 0.40)]))
    assert r.action == "agent" and "confuser_pair" in r.reasons


def test_select_within_budget_keeps_highest_priority():
    bs = [(brief([("standing", p)], minute=i), GateResult("agent", ["low_confidence"], priority=pr)) for i, (p, pr) in enumerate([(.2, 3), (.5, 1), (.4, 2)])]
    out = select_within_budget(bs, 1)
    assert [g.action for _, g in out] == ["agent", "keep_tier1", "keep_tier1"]


def test_tool_args_validated_and_clamped(tmp_path):
    spec = TOOLS["temporal_window"].args
    a = validate_args(spec, {"before_s": 999, "n_frames": 50, "camera": "front"})
    assert a["before_s"] == 30 and a["n_frames"] == 6
    ctx = ToolContext(MockBackend(), brief([("standing", 0.5)]), cfg(), tmp_path)
    assert not execute(ctx, ToolCall(tool="nope")).ok
    assert not execute(ctx, ToolCall(tool="zoom", args={"camera": "side"})).ok
    r = execute(ctx, ToolCall(tool="enhance_night", args={}))
    assert r.ok and r.facts["brightness_after"] >= 0 and Path(r.images[0]).exists()


def test_extract_json_variants():
    assert extract_json('blah ```json\n{"a": 1}\n``` tail') == {"a": 1}
    assert extract_json('{"a": {"b": "}"}}') == {"a": {"b": "}"}}
    with pytest.raises(Exception):
        extract_json("no json here")


def test_label_normalisation():
    assert Hypothesis(label="Ruminating (lying)", confidence=1.4).label == "ruminating_lying"
    assert Hypothesis(label="other / not visible", confidence=-3).confidence == 0.0
    with pytest.raises(Exception):
        Hypothesis(label="sleeping", confidence=0.5)


# ---------------------------------------------------------------- reasons, night, needs, identity
def test_every_action_has_a_reason_and_decision_explains_itself():
    scen = {("C01", 1, 100): {"head_in_bunk_frac": 0.8, "zone_fractions": {"feed_bunk": 0.8, "waterer": 0, "resting": 0, "open": 0.2}}}
    d = agent(scen).run(brief([("standing", 0.41), ("feeding", 0.33)]), GateResult("agent", ["confuser_pair"]))
    assert d.actions and all(a["reason"] and a["reason"] != "(no reason supplied)" for a in d.actions)
    kinds = {a["actor"] for a in d.actions}
    assert {"gate", "agent", "vlm", "tool", "verifier", "resolver"} <= kinds
    assert d.rationale and d.correction in ("confirmed", "behavior_corrected")


def test_planner_steps_without_reason_still_get_one():
    def policy(task, user, images, meta):
        if task == "plan":
            return {"rationale": "", "steps": [{"tool": "zone_query", "args": {}}]}
        return default_mock_policy(task, user, images, meta)
    d = agent({}, policy).run(brief([("standing", 0.41), ("feeding", 0.33)]))
    tools = [a for a in d.actions if a["actor"] == "tool"]
    assert tools and all(a["reason"].strip() for a in tools)
    assert any(a["source"] == "guardrail" for a in tools)        # recipe tools the planner skipped were added, with a reason


def test_yolo_error_is_corrected_with_explanation():
    scen = {("C01", 1, 100): {"head_in_bunk_frac": 0.0, "posture_mode": "lying", "speed_mean": 0.5}}
    d = agent(scen).run(brief([("feeding", 0.60), ("standing", 0.30)], posture="lying"))
    assert d.behavior == "lying" and d.correction == "behavior_corrected"
    assert "YOLO said 'feeding'" in d.rationale and d.tier1_label == "feeding"


def test_night_frame_detected_from_pixels_enhanced_and_capped():
    scen = {("C01", 1, 100): {"brightness": 20.0, "head_in_bunk_frac": 0.8,
                              "zone_fractions": {"feed_bunk": 0.8, "waterer": 0, "resting": 0, "open": 0.2}}}
    d = agent(scen).run(brief([("standing", 0.41), ("feeding", 0.33)], is_night=False))
    assert d.is_night and "night" in d.flags
    assert d.confidence <= 0.85
    assert any(a["action"] == "night_check" and "night=True" in a["outcome"] for a in d.actions)
    assert any(a["action"] == "tool:enhance_night" and a["reason"] for a in d.actions)


def test_day_frame_is_not_marked_night():
    d = agent({}).run(brief([("standing", 0.41), ("feeding", 0.33)], is_night=False))
    assert not d.is_night and "night" not in d.flags


def test_night_fine_grained_label_without_measurement_is_flagged():
    h = Hypothesis(label="grooming", confidence=0.8, cues=["head turned to flank"])
    b = brief([("standing", 0.5), ("grooming", 0.3)], is_night=True)
    rep = Verifier(cfg().verifier, cfg().night).run(h, b, {})
    assert any(c.name == "night_unmeasured" for c in rep.soft_failures)


def test_needs_drive_an_extra_evidence_round():
    calls = {"n": 0}

    def policy(task, user, images, meta):
        if task.startswith("hypothesis"):
            calls["n"] += 1
            if calls["n"] == 1:
                out = default_mock_policy(task, user, images, meta)
                return {**out, "confidence": 0.6, "needs": "a close look at the head from the front camera"}
        return default_mock_policy(task, user, images, meta)
    scen = {("C01", 1, 100): {"head_in_bunk_frac": 0.8, "zone_fractions": {"feed_bunk": 0.8, "waterer": 0, "resting": 0, "open": 0.2}}}
    d = agent(scen, policy).run(brief([("standing", 0.41), ("feeding", 0.33)]))
    nr = [a for a in d.actions if a["source"] == "needs"]
    assert any(a["action"] == "needs_round" for a in nr)
    assert any(a["action"] == "tool:zoom" and a["args"].get("camera") == "front" for a in nr)


def test_id_swap_is_flagged_not_silently_fixed():
    scen = {("C01", 1, 100): {"exemplar_sim": 0.30, "exemplar_scores": {"C01": 0.30, "C07": 0.82}}}
    d = agent(scen).run(brief([("standing", 0.41), ("feeding", 0.33)], id_conf=0.4))
    assert d.id_suggested == "C07" and d.id_verified is False
    assert any(f.startswith("id_swap_suspected") for f in d.flags)
    assert d.cow_id == "C01"                                      # never auto-reassigned
