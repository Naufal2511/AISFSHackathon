"""Unit tests for cross-camera association and global cow registry."""

import unittest
import numpy as np
from src.calibration.camera_calibration import CameraCalibrator
from src.calibration.coordinate_transform import CoordinateTransformer
from src.tracking.local_tracker import LocalTrack, TrackState
from src.tracking.global_identity import GlobalIdentityManager


class TestGlobalIdentity(unittest.TestCase):
    def setUp(self):
        calibrator = CameraCalibrator(config_path="configs/cameras.yaml")
        transformer = CoordinateTransformer(calibrator)
        self.mgr = GlobalIdentityManager(transformer=transformer)

    def test_registry_initialization(self):
        self.assertEqual(len(self.mgr.registry), 10)
        self.assertIn("COW_01", self.mgr.registry)
        self.assertIn("COW_10", self.mgr.registry)

    def test_cross_camera_association(self):
        # Create simultaneous tracks from cam1 and cam3
        # Cam 1 track at front right (X=10.0, Y=5.0)
        t1 = LocalTrack(
            track_id=1,
            camera_id="cam1",
            state=TrackState.CONFIRMED,
            bbox=(2000.0, 1000.0, 2400.0, 1400.0),
            detection_confidence=0.85,
        )

        camera_tracks = {"cam1": [t1]}
        track_embeddings = {}

        mapping = self.mgr.associate_tracks(
            camera_tracks=camera_tracks,
            track_embeddings=track_embeddings,
            current_timestamp=1.0,
        )

        self.assertIn(("cam1", 1), mapping)
        global_id = mapping[("cam1", 1)]
        self.assertEqual(global_id, "COW_01")

        # Verify summary
        summary = self.mgr.get_summary()
        cow1_summary = next(s for s in summary if s["global_cow_id"] == "COW_01")
        self.assertIn(1, cow1_summary["camera_track_ids"].get("cam1", []))


if __name__ == "__main__":
    unittest.main()
