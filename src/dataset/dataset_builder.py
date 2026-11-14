"""Automated YOLO segmentation dataset creation with pseudo-labeling."""

import os
import glob
from typing import Dict, Any, Optional
import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from ..calibration.pen_boundary import PenBoundaryManager
from ..detection.preprocessing import ImagePreprocessor


class YOLODatasetBuilder:
    """Generates standard YOLOv8/11/26 segmentation and detection training datasets."""

    def __init__(
        self,
        base_dir: str = "data/training",
        model_path: str = "yolo26x-seg.pt",
        confidence_threshold: float = 0.25,
    ):
        self.base_dir = base_dir
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold

        self.boundary_manager = PenBoundaryManager()
        self.preprocessor = ImagePreprocessor()
        self.model = YOLO(model_path)

    def auto_annotate_split(self, split: str = "train", camera_id: str = "cam1") -> int:
        """
        Runs perception model on extracted frames, applies Pen ROI filtering,
        and generates normalized polygon label files (.txt) for YOLO segmentation training.
        """
        img_dir = os.path.join(self.base_dir, "images", split)
        lbl_dir = os.path.join(self.base_dir, "labels", split)
        os.makedirs(lbl_dir, exist_ok=True)

        img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png")))
        annotated_count = 0

        for img_path in img_files:
            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w = img.shape[:2]
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(lbl_dir, f"{base_name}.txt")

            # Apply adaptive CLAHE if night
            processed_img, _ = self.preprocessor.preprocess(img)

            # Detect cows
            res = self.model(
                processed_img,
                conf=self.confidence_threshold,
                classes=[19], # COCO cow
                verbose=False,
            )[0]

            lines = []
            boxes = res.boxes
            masks = res.masks

            if boxes is not None and len(boxes) > 0:
                for i, box in enumerate(boxes):
                    coords = box.xyxy[0].cpu().numpy().tolist()
                    x1, y1, x2, y2 = coords

                    # Infer camera ID from filename prefix if available (e.g. "cam2_frame_...")
                    active_cam = camera_id
                    prefix = base_name.split("_")[0]
                    if prefix in ["cam1", "cam2", "cam3"]:
                        active_cam = prefix

                    # Filter out cattle outside the pen ROI
                    if not self.boundary_manager.is_bbox_inside(active_cam, (x1, y1, x2, y2), w, h):
                        continue

                    # If segmentation mask exists, format as normalized polygon
                    if masks is not None and i < len(masks.xyn):
                        norm_poly = masks.xyn[i]
                        if len(norm_poly) >= 3:
                            poly_str = " ".join([f"{pt[0]:.6f} {pt[1]:.6f}" for pt in norm_poly])
                            lines.append(f"0 {poly_str}\n")
                    else:
                        # Fallback bounding box format: 0 cx cy w h
                        cx = ((x1 + x2) / 2.0) / w
                        cy = ((y1 + y2) / 2.0) / h
                        bw = (x2 - x1) / w
                        bh = (y2 - y1) / h
                        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            with open(label_path, "w") as f:
                f.writelines(lines)

            annotated_count += 1

        return annotated_count

    def write_dataset_yaml(self) -> str:
        """Generates standard YOLO dataset.yaml file."""
        abs_base = os.path.abspath(self.base_dir)
        yaml_content = {
            "path": abs_base,
            "train": "images/train",
            "val": "images/val",
            "names": {
                0: "angus_cow"
            }
        }
        yaml_path = os.path.join(self.base_dir, "dataset.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

        return yaml_path
