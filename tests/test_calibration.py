"""Unit tests for camera calibration and coordinate transformation."""

import unittest
from src.calibration.camera_calibration import CameraCalibrator
from src.calibration.coordinate_transform import CoordinateTransformer
from src.calibration.zone_mapping import ZoneMapper


class TestCalibration(unittest.TestCase):
    def setUp(self):
        self.calibrator = CameraCalibrator(config_path="configs/cameras.yaml")
        self.transformer = CoordinateTransformer(self.calibrator)
        self.zone_mapper = ZoneMapper(config_path="configs/pen_layout.yaml")

    def test_homography_exists(self):
        self.assertTrue(self.calibrator.has_calibration("cam1"))
        self.assertTrue(self.calibrator.has_calibration("cam2"))
        self.assertTrue(self.calibrator.has_calibration("cam3"))

    def test_reprojection_error(self):
        err = self.calibrator.get_reprojection_error("cam1")
        self.assertIsNotNone(err)
        # Should have reasonable reprojection error
        self.assertLess(err, 2.0)

    def test_pixel_to_world_and_back(self):
        # Test point in cam1
        ref_res = (3024, 1640)
        # Test point near front bunk center: X=6m, Y=0m
        norm_u, norm_v = 0.50, 0.86
        px, py = norm_u * 3024, norm_v * 1640

        world = self.transformer.pixel_to_world("cam1", px, py, current_res=ref_res)
        self.assertIsNotNone(world)
        self.assertGreaterEqual(world[0], 0.0)
        self.assertLessEqual(world[0], 12.0)

    def test_zone_mapping(self):
        # (6.0, 1.0) is in feeding apron
        zone = self.zone_mapper.get_zone(6.0, 1.0)
        self.assertEqual(zone, "Feeding Zone")

        # (10.5, 16.0) is at the waterer (challenge spec: X 8.5-11.5, Y 14-18)
        zone = self.zone_mapper.get_zone(10.5, 16.0)
        self.assertEqual(zone, "Drinking Zone")

        # (5.0, 16.0) is in resting zone
        zone = self.zone_mapper.get_zone(5.0, 16.0)
        self.assertEqual(zone, "Resting Zone")


if __name__ == "__main__":
    unittest.main()
