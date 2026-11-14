"""Multi-camera temporal synchronization and timeline alignment module."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class SynchronizedFrameGroup:
    timestamp_sec: float
    frames: Dict[str, np.ndarray]  # camera_id -> frame_bgr
    frame_indices: Dict[str, int]  # camera_id -> frame_index


class MultiCameraSynchronizer:
    """Aligns video frames across multiple asynchronous camera streams."""

    def __init__(self, camera_offsets_sec: Optional[Dict[str, float]] = None):
        self.offsets = camera_offsets_sec or {}

    def get_adjusted_timestamp(self, camera_id: str, raw_timestamp_sec: float) -> float:
        """Applies calibration offset to camera timestamp."""
        offset = self.offsets.get(camera_id, 0.0)
        return raw_timestamp_sec + offset

    def find_nearest_timestamp(
        self,
        query_timestamp: float,
        candidate_timestamps: List[float],
        max_tolerance_sec: float = 0.5,
    ) -> Optional[int]:
        """Finds closest frame index within tolerance window."""
        if not candidate_timestamps:
            return None

        arr = np.array(candidate_timestamps)
        diffs = np.abs(arr - query_timestamp)
        min_idx = int(np.argmin(diffs))
        if diffs[min_idx] <= max_tolerance_sec:
            return min_idx
        return None
