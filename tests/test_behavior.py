"""Behavior classification tests on a synthetic, physically plausible tracking log."""

import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.behavior.classifier import BehaviorClassifier
from src.behavior.export import write_all
from src.behavior.taxonomy import BEHAVIORS


def _row(t, cow, track, wx, wy, y2, h, w=260, x=1000, cam="cam1"):
    return dict(timestamp=float(t), camera_id=cam, frame_number=int(t * 15), local_track_id=track,
                global_cow_id=cow, bbox_x1=x, bbox_y1=y2 - h, bbox_x2=x + w, bbox_y2=y2,
                center_x=x + w / 2, center_y=y2 - h / 2, detection_confidence=0.9, tracking_confidence=0.9,
                segmentation_mask_reference="POLYGON_MASK", visibility_status="VISIBLE", occlusion_status="0.00",
                identity_confidence=0.9, world_x=wx, world_y=wy,
                source_video="Event20261107071945006.Avi", frame_width=3024, frame_height=1640)


def standing_h(y2):  # perspective: standing height grows with the image row
    return 0.25 * y2 + 20


def synth():
    rows = []
    rng = np.random.default_rng(0)
    for t in range(0, 600):
        # COW_01: lying in the resting area the whole time
        y2 = 900.0
        rows.append(_row(t, "COW_01", 1, 6.0, 15.0, y2, 0.5 * standing_h(y2) + rng.normal(0, 3)))
        # COW_02: feeding at bunk 0-299 s, walks to the waterer 300-324 s (15.2 m / 25 s = 0.6 m/s), drinks 325-449 s, then leaves view
        if t < 300:
            y2 = 1400.0; wx, wy, x = 6.0, 1.5, 1000
        elif t < 325:
            k = (t - 300) / 25.0
            wx, wy = 6.0 + 4.5 * k, 1.5 + 14.5 * k
            y2 = 1400.0 - 700 * k; x = 1000 + int(600 * k)  # mostly along the viewing axis
        elif t < 450:
            y2 = 700.0; wx, wy, x = 10.5, 16.0, 1600
        else:
            continue
        rows.append(_row(t, "COW_02", 2, wx, wy, y2, standing_h(y2) + rng.normal(0, 3), x=x))
        # extra standing cows across depths so the height model has an upper envelope everywhere
    for t in range(0, 600, 5):
        for k, y2 in enumerate((500, 700, 900, 1100, 1300, 1500)):
            rows.append(_row(t, f"COW_{3 + k:02d}", 10 + k, 2.0 + k, 8 + 2 * k, y2, standing_h(y2), x=100 + 300 * k))
    return pd.DataFrame(rows)


class TestBehavior(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.csv = os.path.join(cls.tmp, "tracking_records.csv")
        synth().to_csv(cls.csv, index=False)
        # isolated config copy: hand-drawn configs/camera_zones.yaml must not leak into synthetic tests
        cfg = os.path.join(cls.tmp, "configs")
        os.makedirs(cfg)
        for f in ("pen_layout.yaml", "cameras.yaml", "behavior.yaml"):
            shutil.copy(os.path.join("configs", f), cfg)
        cls.clf = BehaviorClassifier(os.path.join(cfg, "behavior.yaml"), os.path.join(cfg, "pen_layout.yaml"),
                                     os.path.join(cfg, "cameras.yaml"))
        cls.tl = cls.clf.run([cls.csv])

    def share(self, cow, a, b, label):
        d = self.tl[self.tl["global_cow_id"] == cow]
        t0 = d["sec"].min()
        d = d[(d["sec"] - t0 >= a) & (d["sec"] - t0 < b)]
        return (d["behavior"] == label).mean()

    def test_lying(self):
        self.assertGreater(self.share("COW_01", 0, 600, "lying"), 0.9)

    def test_feeding(self):
        self.assertGreater(self.share("COW_02", 10, 290, "feeding"), 0.9)

    def test_walking(self):
        self.assertGreater(self.share("COW_02", 303, 322, "walking"), 0.6)

    def test_drinking(self):
        self.assertGreater(self.share("COW_02", 340, 440, "drinking"), 0.8)

    def test_wallclock_anchor(self):
        first = pd.Timestamp("1970-01-01") + pd.Timedelta(seconds=int(self.tl["sec"].min()))
        self.assertEqual(str(first), "2026-11-07 07:19:45")

    def test_only_official_classes(self):
        self.assertTrue(set(self.tl["behavior"]).issubset(BEHAVIORS))

    def test_exports(self):
        paths = write_all(os.path.join(self.tmp, "out"), self.tl, self.clf.cfg, full_days=["2026-11-07"])
        act = pd.read_csv(paths["activity_csv"])
        self.assertEqual(len(act), 8 * 1440)  # every minute of the day for all 8 cows
        self.assertIn("no_video", set(act["evidence"]))
        tb = pd.read_csv(paths["time_budgets_csv"])
        self.assertIn("pct_feeding", tb.columns)


if __name__ == "__main__":
    unittest.main()
