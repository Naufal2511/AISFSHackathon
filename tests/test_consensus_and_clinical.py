"""Unit tests for Lyapunov Consensus Protocol and Bovine Head Clinical Analyzer."""

import unittest
import numpy as np
from src.tracking.consensus_protocol import LyapunovConsensusProtocol
from src.detection.clinical_head_detector import (
    BovineHeadClinicalAnalyzer,
    ClinicalWindowAggregator,
    HeadCropResult,
)


class TestLyapunovConsensusProtocol(unittest.TestCase):
    def setUp(self):
        self.consensus = LyapunovConsensusProtocol(num_cows=10, consensus_rate_gamma=0.25)

    def test_lyapunov_energy_monotonic_decay(self):
        """Verifies that inter-camera disagreement energy strictly decreases under consensus iterations."""
        # Initial disagreement: Cam 1 believes track is COW_01, Cam 2 believes track is COW_02
        p1 = np.zeros(10, dtype=np.float32)
        p1[0] = 0.90
        p1[1] = 0.10
        self.consensus.beliefs[("cam1", 1)] = p1

        p2 = np.zeros(10, dtype=np.float32)
        p2[0] = 0.10
        p2[1] = 0.90
        self.consensus.beliefs[("cam2", 2)] = p2

        matches = [(("cam1", 1), ("cam2", 2))]
        evidence = {}

        # Initial energy
        diff = p1 - p2
        v_initial = float(np.dot(diff, diff))

        prev_energy = v_initial
        for k in range(5):
            agreed_ids, energy = self.consensus.update_consensus(evidence, matches)
            self.assertLessEqual(
                energy,
                prev_energy + 1e-6,
                f"Energy increased at iteration {k}: {energy} > {prev_energy}",
            )
            prev_energy = energy

        # After convergence, beliefs should be identical (energy near 0)
        self.assertLess(prev_energy, 0.05)


class TestClinicalHeadDetector(unittest.TestCase):
    def setUp(self):
        self.analyzer = BovineHeadClinicalAnalyzer(min_resolution_for_discharge=120)
        self.aggregator = ClinicalWindowAggregator(window_size_sec=10.0)

    def test_head_crop_extraction_and_scoring(self):
        # Create synthetic test frame (1080x1920 black image with dark cow body)
        frame = np.full((1080, 1920, 3), 40, dtype=np.uint8)
        # Cow bbox: [200, 200, 600, 700] -> (400w, 500h)
        cow_bbox = (200.0, 200.0, 600.0, 700.0)

        # In cam2 (overhead bunk), head is bottom half
        result_cam2 = self.analyzer.analyze_clinical_signs(
            frame_bgr=frame,
            cow_bbox=cow_bbox,
            camera_id="cam2",
            cow_id="COW_01",
            timestamp_sec=5.0,
            frame_number=10,
            save_crop=False,
        )

        self.assertIsNotNone(result_cam2)
        self.assertEqual(result_cam2.clinical_feasibility, "FEASIBLE")
        self.assertEqual(result_cam2.clip_10s_reference, "cam2_COW_01_clip_00000_00010.mp4")
        self.assertIn(result_cam2.wisconsin_nasal_score, [0, 1, 2, 3])
        self.assertIn(result_cam2.wisconsin_eye_score, [0, 1, 2, 3])

    def test_10s_window_aggregator(self):
        obs1 = HeadCropResult(
            camera_id="cam2",
            cow_id="COW_02",
            timestamp_sec=2.5,
            frame_number=5,
            head_resolution=(200, 220),
            nasal_box_local=(20, 60, 80, 100),
            left_eye_box_local=(5, 25, 45, 55),
            right_eye_box_local=(55, 25, 95, 55),
            nasal_moisture_index=0.15,
            ocular_discharge_index=0.10,
            wisconsin_nasal_score=0,
            wisconsin_eye_score=0,
            clinical_feasibility="FEASIBLE",
            clip_10s_reference="cam2_COW_02_clip_00000_00010.mp4",
        )
        obs2 = HeadCropResult(
            camera_id="cam2",
            cow_id="COW_02",
            timestamp_sec=7.0,
            frame_number=14,
            head_resolution=(205, 225),
            nasal_box_local=(20, 60, 80, 100),
            left_eye_box_local=(5, 25, 45, 55),
            right_eye_box_local=(55, 25, 95, 55),
            nasal_moisture_index=0.65,
            ocular_discharge_index=0.20,
            wisconsin_nasal_score=2,
            wisconsin_eye_score=0,
            clinical_feasibility="FEASIBLE",
            clip_10s_reference="cam2_COW_02_clip_00000_00010.mp4",
        )

        self.aggregator.add_observation(obs1)
        self.aggregator.add_observation(obs2)

        summary = self.aggregator.get_summary_records()
        self.assertEqual(len(summary), 1)
        rec = summary[0]
        self.assertEqual(rec["cow_id"], "COW_02")
        self.assertEqual(rec["window_index"], 0)
        self.assertEqual(rec["total_frames_sampled"], 2)
        self.assertEqual(rec["wisconsin_nasal_score_max"], 2)
        self.assertTrue(rec["brd_clinical_sign_detected"])


if __name__ == "__main__":
    unittest.main()
