"""Video metadata inspection and computational resource estimation module."""

import os
import glob
from dataclasses import dataclass
from typing import Optional, Dict, Any
import cv2


@dataclass
class VideoInfo:
    source_path: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    file_size_bytes: int
    codec: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_path": self.source_path,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 2),
            "total_frames": self.total_frames,
            "duration_seconds": round(self.duration_seconds, 2),
            "file_size_mb": round(self.file_size_bytes / (1024 * 1024), 2),
            "codec": self.codec,
        }


class VideoMetadataInspector:
    """Inspects video metadata and computes runtime/storage requirements."""

    @staticmethod
    def inspect(source_path: str) -> VideoInfo:
        """Inspects a video file or directory of frames."""
        if os.path.isdir(source_path):
            return VideoMetadataInspector._inspect_image_directory(source_path)

        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Video file not found: {source_path}")

        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {source_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or fps > 240:
            fps = 15.0  # sensible fallback
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0

        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
        cap.release()

        file_size = os.path.getsize(source_path)

        return VideoInfo(
            source_path=source_path,
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration_seconds=duration,
            file_size_bytes=file_size,
            codec=codec,
        )

    @staticmethod
    def _inspect_image_directory(dir_path: str) -> VideoInfo:
        """Inspects a directory of image frames."""
        extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(dir_path, ext)))
        files.sort()

        if not files:
            raise ValueError(f"No image files found in directory: {dir_path}")

        sample = cv2.imread(files[0])
        if sample is None:
            raise ValueError(f"Unable to read image: {files[0]}")

        h, w = sample.shape[:2]
        total_size = sum(os.path.getsize(f) for f in files)
        fps = 15.0  # assumed reference FPS for frame sequence
        duration = len(files) / fps

        return VideoInfo(
            source_path=dir_path,
            width=w,
            height=h,
            fps=fps,
            total_frames=len(files),
            duration_seconds=duration,
            file_size_bytes=total_size,
            codec="IMAGE_SEQUENCE",
        )

    @staticmethod
    def estimate_processing_budget(
        total_video_gb: float = 50.0,
        total_hours: float = 48.0,
        native_fps: float = 15.0,
        target_fps: float = 1.0,
        inference_fps_per_gpu: float = 35.0,
    ) -> Dict[str, Any]:
        """Calculates computational time, frame count, and storage estimates."""
        total_seconds = total_hours * 3600.0
        total_native_frames = int(total_seconds * native_fps)
        sampled_frames = int(total_seconds * target_fps)
        downsample_ratio = native_fps / target_fps

        est_compute_seconds = sampled_frames / inference_fps_per_gpu
        est_compute_hours = est_compute_seconds / 3600.0

        # Output storage estimate: 10 cows * 50 bytes per record
        tracking_records = sampled_frames * 10
        est_csv_mb = (tracking_records * 120) / (1024 * 1024)

        return {
            "total_video_gb": total_video_gb,
            "total_video_hours": total_hours,
            "native_fps": native_fps,
            "sampling_target_fps": target_fps,
            "downsample_factor": f"{downsample_ratio:.1f}x",
            "total_native_frames": total_native_frames,
            "processed_frames_at_target_fps": sampled_frames,
            "estimated_inference_hours_on_gpu": round(est_compute_hours, 2),
            "estimated_csv_size_mb": round(est_csv_mb, 2),
        }
