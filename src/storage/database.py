"""Structured tracking record database and in-memory aggregation."""

from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import pandas as pd


@dataclass
class TrackingRecord:
    timestamp: float
    camera_id: str
    frame_number: int
    local_track_id: int
    global_cow_id: Optional[str]
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    center_x: float
    center_y: float
    detection_confidence: float
    tracking_confidence: float
    segmentation_mask_reference: Optional[str]
    visibility_status: str
    occlusion_status: str
    identity_confidence: Optional[float]
    world_x: Optional[float]
    world_y: Optional[float]
    discharge_inspection_feasible: Optional[bool] = None
    best_clinical_camera: Optional[str] = None
    clip_10s_reference: Optional[str] = None
    consensus_energy: Optional[float] = None
    # Provenance needed by the behavior layer (wall-clock anchoring, pixel geometry)
    source_video: Optional[str] = None
    frame_width: Optional[int] = None
    frame_height: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 3),
            "camera_id": self.camera_id,
            "frame_number": self.frame_number,
            "local_track_id": self.local_track_id,
            "global_cow_id": self.global_cow_id,
            "bbox_x1": round(self.bbox_x1, 1),
            "bbox_y1": round(self.bbox_y1, 1),
            "bbox_x2": round(self.bbox_x2, 1),
            "bbox_y2": round(self.bbox_y2, 1),
            "center_x": round(self.center_x, 1),
            "center_y": round(self.center_y, 1),
            "detection_confidence": round(self.detection_confidence, 3),
            "tracking_confidence": round(self.tracking_confidence, 3),
            "segmentation_mask_reference": self.segmentation_mask_reference,
            "visibility_status": self.visibility_status,
            "occlusion_status": self.occlusion_status,
            "identity_confidence": round(self.identity_confidence, 3)
            if self.identity_confidence is not None
            else None,
            "world_x": round(self.world_x, 2) if self.world_x is not None else None,
            "world_y": round(self.world_y, 2) if self.world_y is not None else None,
            "discharge_inspection_feasible": self.discharge_inspection_feasible,
            "best_clinical_camera": self.best_clinical_camera,
            "clip_10s_reference": self.clip_10s_reference,
            "consensus_energy": round(self.consensus_energy, 4)
            if self.consensus_energy is not None
            else None,
            "source_video": self.source_video,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
        }


class TrackingDatabase:
    """Stores and queries structured tracking records."""

    def __init__(self):
        self.records: List[TrackingRecord] = []

    def insert(self, record: TrackingRecord):
        self.records.append(record)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame(
                columns=[
                    "timestamp", "camera_id", "frame_number", "local_track_id",
                    "global_cow_id", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
                    "center_x", "center_y", "detection_confidence",
                    "tracking_confidence", "segmentation_mask_reference",
                    "visibility_status", "occlusion_status", "identity_confidence",
                    "world_x", "world_y"
                ]
            )
        return pd.DataFrame([r.to_dict() for r in self.records])
