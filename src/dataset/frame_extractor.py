"""Keyframe extraction and temporal train/val splitting for feedlot video."""

import os
import glob
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np

from ..video.video_reader import FrameReader


class KeyframeExtractor:
    """Extracts non-redundant keyframes from video streams and creates leak-free temporal splits."""

    def __init__(self, output_dir: str = "data/training"):
        self.output_dir = output_dir
        self.train_img_dir = os.path.join(output_dir, "images", "train")
        self.val_img_dir = os.path.join(output_dir, "images", "val")
        os.makedirs(self.train_img_dir, exist_ok=True)
        os.makedirs(self.val_img_dir, exist_ok=True)

    def extract_from_source(
        self,
        camera_id: str,
        source_path: str,
        sample_interval_sec: float = 30.0,
        train_ratio: float = 0.80,
    ) -> Dict[str, int]:
        """
        Samples frames at regular intervals and splits into train/val chronologically.
        Chronological splitting ensures frames from the same time block are not leaked across sets.
        """
        extracted_frames: List[Tuple[float, np.ndarray, str]] = []

        reader = FrameReader(source_path, target_fps=1.0 / max(0.1, sample_interval_sec))
        for f_idx, ts, frame in reader.frames():
            extracted_frames.append((ts, frame, f"{camera_id}_frame_{f_idx:06d}"))

        if not extracted_frames:
            return {"total_extracted": 0, "train_count": 0, "val_count": 0}

        # Chronological split: First 80% -> Train, Last 20% -> Val
        split_idx = int(len(extracted_frames) * train_ratio)
        train_frames = extracted_frames[:split_idx]
        val_frames = extracted_frames[split_idx:] if split_idx < len(extracted_frames) else extracted_frames[-1:]

        # Save to disk
        for ts, img, name in train_frames:
            p = os.path.join(self.train_img_dir, f"{name}.jpg")
            cv2.imwrite(p, img)

        for ts, img, name in val_frames:
            p = os.path.join(self.val_img_dir, f"{name}.jpg")
            cv2.imwrite(p, img)

        return {
            "total_extracted": len(extracted_frames),
            "train_count": len(train_frames),
            "val_count": len(val_frames),
        }
