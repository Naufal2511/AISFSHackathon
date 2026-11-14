"""Unit tests for pen boundary ROI and containment testing."""

import unittest
import numpy as np
from src.calibration.pen_boundary import PenBoundaryManager


class TestPenBoundary(unittest.TestCase):
    def setUp(self):
        self.mgr = PenBoundaryManager(config_path="configs/pen_layout.yaml")

    def test_load_rois(self):
        self.assertIn("cam1", self.mgr.camera_rois)
        self.assertIn("cam2", self.mgr.camera_rois)
        self.assertIn("cam3", self.mgr.camera_rois)

    def test_cam1_containment(self):
        # Inside Pen 6 center
        w, h = 3024, 1640
        inside_pt = (w * 0.45, h * 0.50)
        self.assertTrue(self.mgr.is_point_inside("cam1", inside_pt, w, h))

        # Outside to the right in Pen 7
        outside_right = (w * 0.95, h * 0.50)
        self.assertFalse(self.mgr.is_point_inside("cam1", outside_right, w, h))

        # Outside to the left in the chute
        outside_left = (w * 0.05, h * 0.50)
        self.assertFalse(self.mgr.is_point_inside("cam1", outside_left, w, h))

    def test_ground_contact_tolerance(self):
        w, h = 3024, 1640
        poly = self.mgr.get_pixel_polygon("cam1", w, h)
        edge_pt = poly[0].astype(float)
        # Point right on the boundary edge (dist ~ 0) -> True
        self.assertTrue(self.mgr.is_point_inside("cam1", (edge_pt[0], edge_pt[1]), w, h, tolerance_px=5.0))

        # Far outside in Pen 7 (e.g. 50px away) -> False
        outside_pt = (w * 0.95, h * 0.50)
        self.assertFalse(self.mgr.is_point_inside("cam1", outside_pt, w, h, tolerance_px=5.0))
        # Bbox test with hooves touching ground
        bbox_outside = (w * 0.90, h * 0.40, w * 0.98, h * 0.60)
        self.assertFalse(self.mgr.is_bbox_inside("cam1", bbox_outside, w, h, tolerance_px=5.0))


if __name__ == "__main__":
    unittest.main()
