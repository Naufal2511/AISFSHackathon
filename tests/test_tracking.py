"""Unit tests for ByteTrack within-camera tracking."""

import unittest
from types import SimpleNamespace
from src.tracking.local_tracker import ByteTracker, KalmanBoxTracker, bbox_iou, TrackState


class TestTracking(unittest.TestCase):
    def test_kalman_box_tracker(self):
        bbox = (100.0, 100.0, 200.0, 200.0)
        tracker = KalmanBoxTracker(bbox)
        pred = tracker.predict()
        self.assertEqual(len(pred), 4)

        # Update with slightly moved box
        new_box = (102.0, 101.0, 202.0, 201.0)
        tracker.update(new_box)
        curr = tracker.get_bbox()
        self.assertAlmostEqual(curr[0], 102.0, delta=5.0)

    def test_bbox_iou(self):
        box1 = (0.0, 0.0, 10.0, 10.0)
        box2 = (0.0, 0.0, 10.0, 10.0)
        self.assertAlmostEqual(bbox_iou(box1, box2), 1.0)

        box3 = (20.0, 20.0, 30.0, 30.0)
        self.assertAlmostEqual(bbox_iou(box1, box3), 0.0)

    def test_bytetrack_lifecycle(self):
        tracker = ByteTracker(camera_id="cam1", min_hits=2)

        # Frame 1: Detection
        d1 = SimpleNamespace(
            bbox=(100.0, 100.0, 200.0, 200.0),
            confidence=0.85,
            ground_point=(150.0, 200.0),
        )
        tracks = tracker.update([d1], frame_number=1, timestamp_sec=0.1)
        # First frame is tentative, so active tracks empty until min_hits reached
        self.assertEqual(len(tracker.tracked_tracks), 1)
        self.assertEqual(tracker.tracked_tracks[0].state, TrackState.TENTATIVE)

        # Frame 2: Same detection -> confirmed
        d2 = SimpleNamespace(
            bbox=(102.0, 101.0, 202.0, 201.0),
            confidence=0.88,
            ground_point=(152.0, 201.0),
        )
        active_tracks = tracker.update([d2], frame_number=2, timestamp_sec=0.2)
        self.assertEqual(len(active_tracks), 1)
        self.assertEqual(active_tracks[0].state, TrackState.CONFIRMED)


if __name__ == "__main__":
    unittest.main()
