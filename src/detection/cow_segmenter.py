"""Instance segmentation mask handling, contour extraction, and compact serialization."""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np


@dataclass
class CowMask:
    polygon: List[List[float]] # Simplified polygon [[x1, y1], [x2, y2], ...]
    area_pixels: float
    bbox: Tuple[float, float, float, float] # [x1, y1, x2, y2]
    centroid: Tuple[float, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "polygon": [[round(x, 1), round(y, 1)] for x, y in self.polygon],
            "area_pixels": round(self.area_pixels, 1),
            "centroid": [round(self.centroid[0], 1), round(self.centroid[1], 1)],
        }


class CowSegmenter:
    """Processes YOLO segmentation mask arrays into compact, lightweight polygon representations."""

    def __init__(self, simplify_tolerance: float = 1.5):
        self.simplify_tolerance = simplify_tolerance

    def extract_mask(
        self,
        binary_mask: np.ndarray,
        target_shape: Optional[Tuple[int, int]] = None,
    ) -> Optional[CowMask]:
        """
        Converts a 2D boolean/binary mask into a simplified polygon and computes centroid/area.
        """
        if binary_mask is None or not np.any(binary_mask):
            return None

        uint_mask = (binary_mask * 255).astype(np.uint8)

        if target_shape is not None and uint_mask.shape != target_shape:
            uint_mask = cv2.resize(
                uint_mask,
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        contours, _ = cv2.findContours(
            uint_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        # Take largest contour
        largest = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(largest))
        if area < 50.0:  # Noise threshold
            return None

        # Simplify contour using Ramer-Douglas-Peucker algorithm
        simplified = cv2.approxPolyDP(largest, self.simplify_tolerance, True)
        poly_pts = simplified.reshape(-1, 2).tolist()

        # Compute centroid using moments
        M = cv2.moments(largest)
        if M["m00"] != 0:
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(largest)
            cx, cy = x + w / 2.0, y + h / 2.0

        x, y, w, h = cv2.boundingRect(largest)
        bbox = (float(x), float(y), float(x + w), float(y + h))

        return CowMask(
            polygon=poly_pts,
            area_pixels=area,
            bbox=bbox,
            centroid=(cx, cy),
        )
