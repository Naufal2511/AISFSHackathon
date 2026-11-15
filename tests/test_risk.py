"""Level 2 risk scoring: input formats, score direction, coverage shrinkage, agent self-check."""
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.risk.agent import RiskAgent, visual_logit
from src.risk.data import BEHAVIORS, canon_behavior, load_activity
from src.risk.indicators import indicators
from src.risk.score import contributions, score_from
from src.risk.store import EvidenceStore

CFG = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "risk.yaml")))


def minute_table(n_cows=8, minutes=240, sick=None, sick_minutes=None):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_cows):
        cow = f"COW_{i + 1:02d}"
        for k in range(minutes if cow != sick or sick_minutes is None else sick_minutes):
            feed = rng.uniform(15, 25) if cow != sick else rng.uniform(0, 4)
            lie = rng.uniform(10, 20) if cow != sick else rng.uniform(35, 45)
            r = {"cow_id": cow, "minute": 1_700_000_000 + 60 * k, "seen_s": 60, "bunk_s": feed * 1.5,
                 "bunk_starts": 0.05, "water_starts": 0.01, "water_s": 1, "dist_m": rng.uniform(1, 3) * (0.3 if cow == sick else 1),
                 "nn_sum": rng.uniform(1.5, 2.5) * (2 if cow == sick else 1), "nn_n": 1, "iso_s": 0,
                 "lying_s": lie, "lying_starts": 0.02, "sync_s": 60, "desync_s": 0,
                 "edge_s": 0, "edge_or_alone_s": 0, "near_sum": 1.5 if cow != sick else 0.3, "near_n": 1,
                 "meal_starts": 0.1, "feeding_min_flag": 1 if feed >= 20 else 0}
            for b in BEHAVIORS:
                r[f"sec_{b}"] = 0.0
            r["sec_feeding"], r["sec_lying"] = feed, lie
            r["sec_standing"] = 60 - feed - lie
            r["beh_obs_s"] = 60
            rows.append(r)
    m = pd.DataFrame(rows)
    m.attrs["feed_rushes"] = []
    return m


class TestInputs(unittest.TestCase):
    def test_aliases(self):
        self.assertEqual(canon_behavior("rum_lying"), "ruminating (lying)")
        self.assertEqual(canon_behavior("Feeding"), "feeding")
        self.assertEqual(canon_behavior("social_interaction"), "social interaction")
        self.assertEqual(canon_behavior(None), "other / not visible")

    def test_json_shapes(self):
        rows = [{"cow_id": "COW_01", "timestamp": "2026-11-07T16:00:00", "behavior": "lying"},
                {"cow_id": "COW_01", "timestamp": "2026-11-07T16:01:00", "behavior": "feed"}]
        with tempfile.TemporaryDirectory() as d:
            for name, obj in [("a.json", rows), ("b.json", {"rows": rows}),
                              ("c.json", {"COW_01": [{k: v for k, v in r.items() if k != "cow_id"} for r in rows]})]:
                p = os.path.join(d, name)
                json.dump(obj, open(p, "w"))
                a = load_activity(p)
                self.assertEqual(len(a), 2)
                self.assertEqual(a.loc[1, "sec_feeding"], 60.0)
                self.assertEqual(a.loc[0, "sec_lying"], 60.0)


class TestScore(unittest.TestCase):
    def test_sick_cow_ranks_first(self):
        m = minute_table(sick="COW_05")
        sc = score_from(contributions(indicators(m, CFG), CFG), CFG)
        self.assertEqual(sc.behavior_score.idxmax(), "COW_05")
        self.assertGreater(sc.loc["COW_05", "behavior_score"], 0.4)
        self.assertLess(sc.drop("COW_05").behavior_score.max(), 0.25)

    def test_thin_coverage_is_shrunk(self):
        score = lambda m: score_from(contributions(indicators(m, CFG), CFG), CFG).loc["COW_05", "behavior_score"]
        self.assertLess(score(minute_table(sick="COW_05", sick_minutes=10)), score(minute_table(sick="COW_05")))

    def test_visual_logit(self):
        fin = {"signs": {"nasal_discharge": {"status": "present"}, "head_low": {"status": "absent"}}}
        v, why = visual_logit(fin, CFG)
        self.assertAlmostEqual(v, 0.9 - 0.15)
        self.assertEqual(why, ["nasal discharge"])


class TestSelfCheck(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.store = EvidenceStore(self.d)
        img = np.zeros((40, 40, 3), np.uint8)
        self.small = self.store.add_image(img, "COW_01", "crop", {"box_px": 120, "daylight": True, "time": 1}, "test")
        self.close_night = self.store.add_image(img, "COW_01", "crop", {"box_px": 700, "daylight": False, "time": 1}, "test")
        self.other = self.store.add_image(img, "COW_02", "crop", {"box_px": 700, "daylight": True, "time": 1}, "test")
        self.agent = RiskAgent.__new__(RiskAgent)
        self.agent.store, self.agent.cfg = self.store, CFG

    def test_rules(self):
        fin = {"signs": {"head_low": {"status": "present", "evidence": [self.close_night]},     # ok (posture at night)
                         "ears_droop": {"status": "present", "evidence": [self.small]},         # too small
                         "nasal_discharge": {"status": "present", "evidence": [self.close_night]},  # night
                         "gaunt": {"status": "absent", "evidence": [self.other]}},              # wrong cow
               "artifacts": [{"indicator": "feeding_share", "evidence": ["F-COW_01-feeding_share"], "why": "short"}],
               "key_evidence": ["I9999"]}
        clean, errs = self.agent.apply_rules("COW_01", fin)
        st = {k: v["status"] for k, v in clean["signs"].items()}
        self.assertEqual(st["head_low"], "present")
        self.assertEqual(st["ears_droop"], "not_assessable")
        self.assertEqual(st["nasal_discharge"], "not_assessable")
        self.assertEqual(st["gaunt"], "not_assessable")
        self.assertEqual(clean["artifacts"], [])        # coverage-only artifact rejected
        self.assertEqual(clean["key_evidence"], [])


if __name__ == "__main__":
    unittest.main()
