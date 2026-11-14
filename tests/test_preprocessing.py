"""Unit tests for adaptive preprocessing and day/night enhancement."""

import unittest
import numpy as np
from src.detection.preprocessing import ImagePreprocessor


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.preprocessor = ImagePreprocessor()

    def test_day_detection(self):
        # Bright synthetic image (daytime)
        day_img = np.ones((100, 100, 3), dtype=np.uint8) * 180
        assessment = self.preprocessor.assess_frame(day_img)
        self.assertFalse(assessment["is_night"])
        self.assertGreater(assessment["mean_brightness"], 100.0)

    def test_night_detection_and_enhancement(self):
        # Dark synthetic image (nighttime)
        night_img = np.ones((100, 100, 3), dtype=np.uint8) * 40
        assessment = self.preprocessor.assess_frame(night_img)
        self.assertTrue(assessment["is_night"])

        enhanced, res = self.preprocessor.preprocess(night_img)
        self.assertEqual(res["enhancement_applied"], "CLAHE_LAB")
        self.assertEqual(enhanced.shape, night_img.shape)


if __name__ == "__main__":
    unittest.main()
