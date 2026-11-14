"""Deep learning cow detection and instance segmentation pipeline."""

import os
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np
from ultralytics import YOLO

from .preprocessing import ImagePreprocessor
from .cow_segmenter import CowSegmenter, CowMask
from ..calibration.pen_boundary import PenBoundaryManager


@dataclass
class CowDetection:
    detection_id: int
    camera_id: str
    frame_number: int
    timestamp_sec: float
    bbox: Tuple[float, float, float, float]  # [x1, y1, x2, y2]
    confidence: float
    center: Tuple[float, float]             # (cx, cy)
    ground_point: Tuple[float, float]       # (cx, y2) hooves on ground
    mask: Optional[CowMask]
    occlusion_score: float                  # [0.0 = clear, 1.0 = heavily occluded]
    visibility_status: str                  # 'VISIBLE', 'PARTIALLY_OCCLUDED', 'HEAVILY_OCCLUDED'

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detection_id": self.detection_id,
            "camera_id": self.camera_id,
            "frame_number": self.frame_number,
            "timestamp_sec": round(self.timestamp_sec, 3),
            "bbox": [round(c, 1) for c in self.bbox],
            "confidence": round(self.confidence, 3),
            "center": [round(c, 1) for c in self.center],
            "ground_point": [round(c, 1) for c in self.ground_point],
            "occlusion_score": round(self.occlusion_score, 2),
            "visibility_status": self.visibility_status,
            "mask": self.mask.to_dict() if self.mask else None,
        }


class CowDetector:
    """Performs cow detection and segmentation with pen boundary filtering and occlusion analysis."""

    OVERHEAD_CAMERAS = ("cam2",)

    def __init__(
        self,
        model_path: str = "yolo11s-seg.pt",
        confidence_threshold: float = 0.20,
        iou_threshold: float = 0.45,
        target_classes: Optional[List[int]] = None,
        boundary_manager: Optional[PenBoundaryManager] = None,
        enable_preprocessing: bool = False,  # detector is trained on RAW frames: keep inference raw too
        imgsz: int = 640,
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        # COCO class 19 = cow
        self.target_classes = target_classes if target_classes is not None else [19]
        self.boundary_manager = boundary_manager or PenBoundaryManager()
        self.enable_preprocessing = enable_preprocessing
        # Inference resolution. 4000x3000 frames shrink 6x at 640, which loses cows that
        # are far away (cam3) or cut by the frame edge (bunk rear views); 1280 recovers them.
        self.imgsz = imgsz

        self.preprocessor = ImagePreprocessor()
        self.segmenter = CowSegmenter()

        # Load Ultralytics model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Detector weights not found: {model_path}")
        self.model = YOLO(model_path)
        # Ultralytics does not pick the Apple GPU by default (it falls back to CPU).
        import torch
        self.device = ("cuda:0" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available() else "cpu")

        # Resolve target classes from the model's own label map. COCO weights call the
        # class "cow" (id 19); the fine-tuned Angus weights have a single class
        # 0 = "angus_cow". Hard-coding 19 silently drops every fine-tuned detection.
        if target_classes is None:
            names = getattr(self.model, "names", {}) or {}
            cow_ids = [int(k) for k, v in names.items() if "cow" in str(v).lower() or "cattle" in str(v).lower()]
            if cow_ids:
                self.target_classes = cow_ids
            elif len(names) <= 2:
                self.target_classes = None  # custom single-class model: keep everything

    def detect(
        self,
        frame_bgr: np.ndarray,
        camera_id: str,
        frame_number: int,
        timestamp_sec: float,
        filter_outside_pen: bool = True,
    ) -> Tuple[List[CowDetection], Dict[str, Any]]:
        """
        Runs detection on a frame, applies adaptive preprocessing, filters by ROI,
        and computes occlusion metrics.
        """
        h, w = frame_bgr.shape[:2]

        # 1. Preprocessing assessment and enhancement
        if self.enable_preprocessing:
            processed_frame, assessment = self.preprocessor.preprocess(frame_bgr)
        else:
            processed_frame = frame_bgr
            assessment = {"enhancement_applied": "NONE", "is_night": False}

        # 2. Run YOLO inference
        results = self.model(
            processed_frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=self.target_classes,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]

        raw_detections: List[Dict[str, Any]] = []
        boxes = results.boxes
        masks = results.masks

        if boxes is not None and len(boxes) > 0:
            for i, box in enumerate(boxes):
                coords = box.xyxy[0].cpu().numpy().tolist()
                x1, y1, x2, y2 = coords
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())

                # Ground point estimation (bottom-center)
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                ground_pt = (cx, y2)

                # 3. Filter out detections outside valid pen ROI (e.g. adjacent pen cows)
                if filter_outside_pen:
                    # Overhead bunk camera: a feeding cow's head (in the trough) is the lowest
                    # point of her box, so "bbox bottom = hooves" is wrong there; test the body.
                    is_inside = self.boundary_manager.is_bbox_inside(
                        camera_id, (x1, y1, x2, y2), w, h,
                        use_ground_point=camera_id not in self.OVERHEAD_CAMERAS,
                    )
                    if not is_inside:
                        continue

                # 4. Extract segmentation mask if available
                cow_mask: Optional[CowMask] = None
                if masks is not None and i < len(masks):
                    mask_data = masks.data[i].cpu().numpy()
                    cow_mask = self.segmenter.extract_mask(mask_data, target_shape=(h, w))

                raw_detections.append(
                    {
                        "bbox": (x1, y1, x2, y2),
                        "confidence": conf,
                        "center": (cx, cy),
                        "ground_point": ground_pt,
                        "mask": cow_mask,
                    }
                )

        # 5. Compute pairwise occlusion scores
        detections: List[CowDetection] = []
        n_dets = len(raw_detections)

        for i, det in enumerate(raw_detections):
            box_i = det["bbox"]
            area_i = max(1.0, (box_i[2] - box_i[0]) * (box_i[3] - box_i[1]))
            max_overlap_ratio = 0.0

            for j, other in enumerate(raw_detections):
                if i == j:
                    continue
                box_j = other["bbox"]
                # Intersection
                ix1 = max(box_i[0], box_j[0])
                iy1 = max(box_i[1], box_j[1])
                ix2 = min(box_i[2], box_j[2])
                iy2 = min(box_i[3], box_j[3])

                if ix2 > ix1 and iy2 > iy1:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    # Fraction of cow i occluded by cow j
                    overlap_ratio = inter_area / area_i
                    if overlap_ratio > max_overlap_ratio:
                        max_overlap_ratio = overlap_ratio

            # Occlusion categorization
            if max_overlap_ratio < 0.15:
                vis_status = "VISIBLE"
            elif max_overlap_ratio < 0.50:
                vis_status = "PARTIALLY_OCCLUDED"
            else:
                vis_status = "HEAVILY_OCCLUDED"

            detections.append(
                CowDetection(
                    detection_id=i + 1,
                    camera_id=camera_id,
                    frame_number=frame_number,
                    timestamp_sec=timestamp_sec,
                    bbox=det["bbox"],
                    confidence=det["confidence"],
                    center=det["center"],
                    ground_point=det["ground_point"],
                    mask=det["mask"],
                    occlusion_score=max_overlap_ratio,
                    visibility_status=vis_status,
                )
            )

        return detections, assessment
