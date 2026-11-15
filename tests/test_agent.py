"""Agent triggers and self-check on synthetic records (no video, no model)."""
import copy
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.agent.loop import Agent, apply_id_corrections
from src.agent.memory import CowCard, Memory
from src.agent.triggers import audit_schedule, find_flags
from src.agent.vlm import parse_json

T0 = 1794036000.0


def rec_rows():
    rows = []
    for i in range(60):
        t = T0 + i
        for k in range(10):
            rows.append(dict(t=t, camera_id="cam1", frame_number=i, local_track_id=k + 1, global_cow_id=f"COW_{k+1:02d}",
                             center_x=300 * k + 100, frame_width=4000, frame_height=3000, world_x=1.0 * k,
                             world_y=5.0, identity_confidence=0.95, detection_confidence=0.9))
    return pd.DataFrame(rows)


class FakeTB:
    def __init__(self, rec):
        self.rec = rec
        self.tools = {}

    def t(self, v):
        return T0 + float(str(v).split(":")[-1]) if isinstance(v, str) else float(v)

    def hms(self, t):
        return str(t)

    def describe(self):
        return ""


class TestTriggers(unittest.TestCase):
    def test_quiet_window_has_no_flags(self):
        r = rec_rows()
        known = [f"COW_{k+1:02d}" for k in range(10)]
        self.assertEqual(find_flags(r, T0, T0 + 60, known), [])

    def test_drastic_events(self):
        r = rec_rows()
        known = [f"COW_{k+1:02d}" for k in range(10)]
        extra = r[r.global_cow_id == "COW_10"].copy()
        extra["global_cow_id"], extra["local_track_id"] = "COW_11", 99
        jump = r.global_cow_id.eq("COW_01") & (r.t == T0 + 30)
        r.loc[jump, "world_x"] = 9.0  # 9 m in 1 s
        r = pd.concat([r, extra])
        kinds = {f.kind for f in find_flags(r, T0, T0 + 60, known)}
        self.assertTrue({"extra_cow", "new_cow_id", "id_jump"} <= kinds)

    def test_weak_id_and_audit(self):
        r = rec_rows()
        r.loc[(r.global_cow_id == "COW_03") & (r.t == T0), "identity_confidence"] = 0.3
        fl = find_flags(r, T0, T0 + 60, [], audit_times=[T0 + 10])
        self.assertIn("weak_id", {f.kind for f in fl})
        self.assertIn("id_audit", {f.kind for f in fl})
        self.assertEqual(len(audit_schedule(T0, T0 + 3600, 30)), 2)


class TestSelfCheck(unittest.TestCase):
    def setUp(self):
        self.mem = Memory(tempfile.mkdtemp())
        for k in range(10):
            self.mem.cards[f"COW_{k+1:02d}"] = CowCard(f"COW_{k+1:02d}")
        self.agent = Agent(FakeTB(rec_rows()), self.mem, backend=None, out_dir=tempfile.mkdtemp())

    def state(self, *kinds, chew=None):
        steps = [{"tool": "x", "kind": k, "args": {}, "data": {}} for k in kinds]
        if chew is not None:
            steps.append({"tool": "chew_motion", "kind": "chew_motion", "args": {}, "data": {"peak_ratio": chew}})
        return {"steps": steps}

    def test_correction_needs_two_kinds(self):
        f = {"decision": "set_behavior", "cow_id": "COW_01", "behavior": "lying", "t0": "0", "t1": "9"}
        self.assertFalse(self.agent.verify(f, self.state("visual"))[0])
        self.assertTrue(self.agent.verify(f, self.state("visual", "classifier"))[0])

    def test_rumination_needs_measured_chewing(self):
        f = {"decision": "set_behavior", "cow_id": "COW_01", "behavior": "ruminating (lying)", "t0": "0", "t1": "9"}
        self.assertFalse(self.agent.verify(f, self.state("visual", "classifier", chew=1.2))[0])
        self.assertTrue(self.agent.verify(f, self.state("visual", "classifier", chew=4.5))[0])

    def test_relabel_cannot_create_duplicate(self):
        f = {"decision": "relabel", "camera": "cam1", "track_id": 1, "new_cow_id": "COW_02", "t0": "0", "t1": "30"}
        ok, why = self.agent.verify(f, self.state("visual", "appearance_score"))
        self.assertFalse(ok)
        self.assertIn("duplicate", why[0])

    def test_uncertain_removes_id(self):
        r = rec_rows()
        corr = pd.DataFrame([{"decision": "uncertain", "camera": "cam1", "track_id": 2,
                              "t0_epoch": T0, "t1_epoch": T0 + 10}])
        out = apply_id_corrections(r, corr)
        self.assertEqual(out[(out.local_track_id == 2) & (out.t <= T0 + 10)].global_cow_id.isna().sum(), 11)

    def test_parse_json_from_chatty_reply(self):
        self.assertEqual(parse_json('Sure!\n```json\n{"tool": "scene", "args": {"camera": "cam1"}}\n```')["tool"], "scene")


class DummyBackend:
    def __init__(self, name, action):
        self.name = name
        self.action = action
        self.calls = 0
        self.seconds = 0.0

    @property
    def label(self):
        return self.name

    def next_action(self, state):
        self.calls += 1
        return copy.deepcopy(self.action)


class TestTieredDecision(unittest.TestCase):
    def test_routine_flag_handled_by_quick_tier(self):
        from src.agent.vlm import TieredDecisionBackend
        fast = DummyBackend("fast_m", {"thought": "quick check", "final": {"decision": "confirm", "confidence": 0.85}})
        major = DummyBackend("major_m", {"thought": "deep check", "final": {"decision": "confirm", "confidence": 0.95}})
        tiered = TieredDecisionBackend(fast, major, escalate_severity=3)

        state = {"flag": {"kind": "behavior_check", "severity": 1}, "steps": [], "turns": []}
        act = tiered.next_action(state)
        self.assertEqual(act["_tier"], "quick")
        self.assertEqual(act["final"]["confidence"], 0.85)
        self.assertEqual(fast.calls, 1)
        self.assertEqual(major.calls, 0)

    def test_critical_flag_routes_direct_to_major_tier(self):
        from src.agent.vlm import TieredDecisionBackend
        fast = DummyBackend("fast_m", {"thought": "quick", "final": {"decision": "confirm", "confidence": 0.85}})
        major = DummyBackend("major_m", {"thought": "deep analysis", "final": {"decision": "relabel", "confidence": 0.90}})
        tiered = TieredDecisionBackend(fast, major, escalate_severity=3)

        state = {"flag": {"kind": "duplicate_id", "severity": 5}, "steps": [], "turns": []}
        act = tiered.next_action(state)
        self.assertEqual(act["_tier"], "major")
        self.assertEqual(act["final"]["decision"], "relabel")
        self.assertEqual(fast.calls, 0)
        self.assertEqual(major.calls, 1)

    def test_uncertain_quick_decision_escalates_to_major(self):
        from src.agent.vlm import TieredDecisionBackend
        fast = DummyBackend("fast_m", {"thought": "ambiguous", "final": {"decision": "uncertain", "confidence": 0.4}})
        major = DummyBackend("major_m", {"thought": "resolved by deep model", "final": {"decision": "confirm", "confidence": 0.88}})
        tiered = TieredDecisionBackend(fast, major, escalate_severity=3, min_confidence=0.70)

        state = {"flag": {"kind": "weak_id", "severity": 2}, "steps": [], "turns": []}
        act = tiered.next_action(state)
        self.assertEqual(act["_tier"], "major")
        self.assertEqual(act["final"]["decision"], "confirm")
        self.assertEqual(fast.calls, 1)
        self.assertEqual(major.calls, 1)
        self.assertEqual(tiered.escalations, 1)

    def test_self_check_failure_prompts_major_escalation(self):
        from src.agent.vlm import TieredDecisionBackend
        fast = DummyBackend("fast_m", {"thought": "tool step", "tool": "trajectory", "args": {}})
        major = DummyBackend("major_m", {"thought": "handling revision", "final": {"decision": "confirm", "confidence": 0.9}})
        tiered = TieredDecisionBackend(fast, major, escalate_severity=3)

        state = {"flag": {"kind": "weak_id", "severity": 2}, "steps": [{"tool": "compare_to_cards"}],
                 "turns": [{"role": "user", "text": "SELF-CHECK FAILED: relabel would create duplicate"}]}
        act = tiered.next_action(state)
        self.assertEqual(act["_tier"], "major")
        self.assertEqual(fast.calls, 0)
        self.assertEqual(major.calls, 1)

    def test_make_backend_decision(self):
        from src.agent.vlm import TieredDecisionBackend, make_backend
        cfg = {
            "backend": "decision",
            "decision": {
                "fast_backend": "rules",
                "major_backend": "rules",
                "escalate_severity": 3,
                "min_confidence": 0.70,
                "max_quick_steps": 2
            }
        }
        b = make_backend(cfg, "decision")
        self.assertIsInstance(b, TieredDecisionBackend)
        self.assertIn("tiered", b.label)


if __name__ == "__main__":
    unittest.main()


class TestToolImages(unittest.TestCase):
    def test_hcat_pads_different_heights(self):
        from src.agent.tools import hcat, vcat
        a, b = np.zeros((88, 100, 3), np.uint8), np.zeros((256, 300, 3), np.uint8)
        self.assertEqual(hcat([a, b]).shape, (256, 408, 3))
        self.assertEqual(vcat([a, b]).shape, (344, 300, 3))


class TrackIdFromModel(unittest.TestCase):
    """The identity agent's track_id comes from the model: nooh_training_c02 crashed on int("t250/t251")."""

    def test_track_ids_parsing(self):
        from src.agent.loop import track_ids
        self.assertEqual(track_ids(250), [250])
        self.assertEqual(track_ids(250.0), [250])
        self.assertEqual(track_ids("t250"), [250])
        self.assertEqual(track_ids("t250/t251"), [250, 251])
        self.assertEqual(track_ids("track ?"), [])
        self.assertEqual(track_ids(float("nan")), [])
        self.assertEqual(track_ids(None), [])

    def test_odd_track_ids_never_stop_the_run(self):
        r = rec_rows()
        corr = pd.DataFrame([
            dict(decision="uncertain", camera="cam1", track_id="t2/t3", t0_epoch=T0, t1_epoch=T0 + 10),
            dict(decision="uncertain", camera="cam1", track_id="track ?", t0_epoch=T0, t1_epoch=T0 + 10),
            dict(decision="relabel", camera="cam1", track_id="t4/t5", new_cow_id="COW_09", t0_epoch=T0, t1_epoch=T0 + 10),
            dict(decision="relabel", camera="cam1", track_id="t6", new_cow_id="COW_10", t0_epoch=T0 + 20, t1_epoch=T0 + 30),
        ])
        out = apply_id_corrections(r, corr)
        win = out.t <= T0 + 10
        self.assertEqual(out[out.local_track_id.isin([2, 3]) & win].global_cow_id.isna().sum(), 22)   # both tracks
        self.assertEqual(out.global_cow_id.isna().sum(), 22)                     # the unreadable one touched nothing
        self.assertTrue((out[(out.local_track_id == 4) & win].global_cow_id == "COW_04").all())   # 2-track relabel skipped
        self.assertTrue((out[(out.local_track_id == 6) & out.t.between(T0 + 20, T0 + 30)].global_cow_id == "COW_10").all())
