"""Tests for hand-drawn camera zones: IO, bbox overlap, ROI override and behavior usage."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

from src.calibration.camera_zones import bbox_fraction_in, load_zones, save_zones


class TestCameraZones(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "camera_zones.yaml")

    def test_roundtrip_and_validation(self):
        save_zones({"cam1": {"roi": [[0, 0], [1, 0], [1, 1]], "waterer": [[0.1, 0.1], [0.2, 0.2]]}}, self.path)
        z = load_zones(self.path)
        self.assertIn("roi", z["cam1"])
        self.assertNotIn("waterer", z["cam1"])  # < 3 points is not a polygon

    def test_bbox_fraction(self):
        band = [[0, 0.8], [1, 0.8], [1, 1], [0, 1]]  # bottom 20% of the frame
        self.assertEqual(bbox_fraction_in(band, (0, 0, 100, 100), 1000, 1000), 0.0)
        self.assertGreater(bbox_fraction_in(band, (100, 700, 300, 1000), 1000, 1000), 0.5)

    def test_cli_set_and_save(self):
        r = subprocess.run([sys.executable, "scripts/draw_zones.py", "--camera", "cam9", "--zones", self.path,
                            "--set", "feed_bunk=0,0.8;1,0.8;1,1;0,1", "--save"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(len(load_zones(self.path)["cam9"]["feed_bunk"]), 4)

    def test_drawn_zone_drives_feeding(self):
        """A cam3 cow whose box sits in a drawn bunk band is feeding even though cam3 has no metric trust."""
        from src.behavior.classifier import BehaviorClassifier
        cfgdir = os.path.join(self.dir, "configs")
        os.makedirs(cfgdir)
        for f in ("pen_layout.yaml", "cameras.yaml", "behavior.yaml"):
            shutil.copy(os.path.join("configs", f), cfgdir)
        save_zones({"cam3": {"feed_bunk": [[0, 0.5], [1, 0.5], [1, 0.62], [0, 0.62]]}}, os.path.join(cfgdir, "camera_zones.yaml"))
        rows = []
        for t in range(120):
            for cow, (y1, y2) in {"COW_01": (1350, 1800), "COW_02": (2400, 2900)}.items():
                rows.append(dict(timestamp=float(t), camera_id="cam3", frame_number=t, local_track_id=int(cow[-1]),
                                 global_cow_id=cow, bbox_x1=1000, bbox_y1=y1, bbox_x2=1400, bbox_y2=y2,
                                 center_x=1200, center_y=(y1 + y2) / 2, detection_confidence=0.9, tracking_confidence=0.9,
                                 segmentation_mask_reference=None, visibility_status="VISIBLE", occlusion_status="0.00",
                                 identity_confidence=0.9, world_x=None, world_y=None,
                                 source_video="Event20261107071935003.Avi", frame_width=4000, frame_height=3000))
        csv = os.path.join(self.dir, "rec.csv")
        pd.DataFrame(rows).to_csv(csv, index=False)
        clf = BehaviorClassifier(os.path.join(cfgdir, "behavior.yaml"), os.path.join(cfgdir, "pen_layout.yaml"),
                                 os.path.join(cfgdir, "cameras.yaml"))
        tl = clf.run([csv])
        share = tl.groupby("global_cow_id")["behavior"].agg(lambda s: (s == "feeding").mean())
        self.assertGreater(share["COW_01"], 0.8)
        self.assertLess(share["COW_02"], 0.2)


if __name__ == "__main__":
    unittest.main()
