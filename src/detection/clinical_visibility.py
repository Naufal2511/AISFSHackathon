"""BRD Clinical Sign Visibility & Anatomical Resolution Assessment Module.

Evaluates whether individual cattle heads, eyes, and nasal regions have sufficient
optical resolution and visibility in a given camera view for clinical BRD inspection
(ocular/nasal discharge, ear droop).
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import numpy as np


@dataclass
class ClinicalVisibilityMetadata:
    camera_id: str
    cow_id: str
    timestamp_sec: float
    frame_number: int
    head_visible: bool
    head_bbox: Optional[Tuple[float, float, float, float]] # [x1, y1, x2, y2]
    head_resolution_pixels: Tuple[int, int]                # (width_px, height_px)
    optical_feasibility: str                               # 'FEASIBLE', 'MARGINAL', 'INSUFFICIENT'
    nasal_region_resolvable: bool                          # True if resolution >= threshold
    eye_region_resolvable: bool                            # True if resolution >= threshold
    best_camera_for_inspection: str                        # 'cam2' (top feed bunk)
    clip_10s_reference: str                                # 10s video window reference

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "cow_id": self.cow_id,
            "timestamp_sec": round(self.timestamp_sec, 2),
            "frame_number": self.frame_number,
            "head_visible": self.head_visible,
            "head_bbox": [round(c, 1) for c in self.head_bbox] if self.head_bbox else None,
            "head_resolution_pixels": list(self.head_resolution_pixels),
            "optical_feasibility": self.optical_feasibility,
            "nasal_region_resolvable": self.nasal_region_resolvable,
            "eye_region_resolvable": self.eye_region_resolvable,
            "best_camera_for_inspection": self.best_camera_for_inspection,
            "clip_10s_reference": self.clip_10s_reference,
        }


class ClinicalVisibilityAssessor:
    """
    Assesses physical resolution and visibility of bovine facial landmarks for BRD symptom detection.
    
    Veterinary Optical Requirements:
    - Nasal discharge (serous/mucopurulent mucus) requires >= 120x120px head crop.
    - Ocular discharge (lacrimation / conjunctivitis) requires >= 160x160px head crop.
    - Feed bunk overhead camera (Cam 2) provides 2-3m close-up views during feeding.
    - Overview cameras (Cam 1 & Cam 3) only resolve facial discharge when cow is at the front fence.
    """

    def __init__(
        self,
        min_head_px_discharge: int = 120,
        min_head_px_ocular: int = 160,
    ):
        self.min_head_px_discharge = min_head_px_discharge
        self.min_head_px_ocular = min_head_px_ocular

    def evaluate_cow(
        self,
        camera_id: str,
        cow_id: str,
        full_bbox: Tuple[float, float, float, float],
        frame_shape: Tuple[int, int],
        timestamp_sec: float,
        frame_number: int,
        world_pos: Optional[Tuple[float, float]] = None,
    ) -> ClinicalVisibilityMetadata:
        """
        Calculates anatomical head crop scale and discharge inspection feasibility.
        """
        x1, y1, x2, y2 = full_bbox
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)

        # Estimate head region based on camera perspective
        # In Cam 2 (overhead top view of feed bunk): the head is in the lower/feeding half
        # In Cam 1 / Cam 3: head occupies roughly top 30-35% of the body bbox
        if camera_id == "cam2":
            # Looking down into feed bunk trough
            head_h = bh * 0.45
            head_w = bw * 0.50
            head_box = (x1 + bw * 0.25, y1 + bh * 0.50, x1 + bw * 0.75, y2)
        else:
            head_h = bh * 0.35
            head_w = bw * 0.40
            head_box = (x1 + bw * 0.30, y1, x1 + bw * 0.70, y1 + head_h)

        head_px_w = int(head_w)
        head_px_h = int(head_h)
        min_dim = min(head_px_w, head_px_h)

        # Assess feasibility
        if min_dim >= self.min_head_px_ocular:
            feasibility = "FEASIBLE"
            nasal_resolvable = True
            eye_resolvable = True
        elif min_dim >= self.min_head_px_discharge:
            feasibility = "MARGINAL"
            nasal_resolvable = True
            eye_resolvable = False
        else:
            feasibility = "INSUFFICIENT"
            nasal_resolvable = False
            eye_resolvable = False

        # If cow is at feed bunk (Y <= 3.5m), Cam 2 provides the optimal view
        best_cam = "cam2" if (world_pos and world_pos[1] <= 3.5) else camera_id

        # 10-second clip reference window
        window_idx = int(timestamp_sec // 10)
        clip_ref = f"{camera_id}_{cow_id}_clip_{window_idx * 10:05d}_{(window_idx + 1) * 10:05d}.mp4"

        return ClinicalVisibilityMetadata(
            camera_id=camera_id,
            cow_id=cow_id,
            timestamp_sec=timestamp_sec,
            frame_number=frame_number,
            head_visible=(feasibility != "INSUFFICIENT"),
            head_bbox=head_box,
            head_resolution_pixels=(head_px_w, head_px_h),
            optical_feasibility=feasibility,
            nasal_region_resolvable=nasal_resolvable,
            eye_region_resolvable=eye_resolvable,
            best_camera_for_inspection=best_cam,
            clip_10s_reference=clip_ref,
        )
