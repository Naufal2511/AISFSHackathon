"""Adaptive image preprocessing and day/night enhancement module."""

from typing import Tuple, Dict, Any
import cv2
import numpy as np


class ImagePreprocessor:
    """Detects lighting/quality conditions and applies targeted adaptive enhancement."""

    def __init__(
        self,
        brightness_threshold: float = 85.0,
        clahe_clip_limit: float = 3.0,
        clahe_grid_size: Tuple[int, int] = (8, 8),
        blur_laplacian_threshold: float = 100.0,
    ):
        self.brightness_threshold = brightness_threshold
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        self.blur_laplacian_threshold = blur_laplacian_threshold
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_grid_size)

    def assess_frame(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """Analyzes frame lighting and blur characteristics."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))
        is_night = mean_brightness < self.brightness_threshold

        # Laplacian variance measures edge sharpness (higher = sharper, lower = blurry)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        is_blurry = laplacian_var < self.blur_laplacian_threshold

        return {
            "mean_brightness": round(mean_brightness, 2),
            "is_night": is_night,
            "laplacian_variance": round(laplacian_var, 2),
            "is_blurry": is_blurry,
        }

    def preprocess(self, frame_bgr: np.ndarray, force_enhance: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Applies adaptive enhancement when beneficial (e.g. CLAHE on nighttime IR footage).
        Preserves original daytime frames to avoid introducing artifacts.
        """
        assessment = self.assess_frame(frame_bgr)
        enhanced = frame_bgr

        if assessment["is_night"] or force_enhance:
            # Convert to LAB and apply CLAHE strictly on the Lightness (L) channel
            lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            cl = self.clahe.apply(l)
            merged = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
            assessment["enhancement_applied"] = "CLAHE_LAB"
        else:
            assessment["enhancement_applied"] = "NONE"

        return enhanced, assessment
