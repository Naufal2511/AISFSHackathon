"""Agent triggers and self-check on synthetic records (no video, no model)."""
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


if __name__ == "__main__":
    unittest.main()
