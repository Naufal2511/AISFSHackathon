"""Frame generator supporting videos and image sequences with configurable sampling."""

import os
os.environ.setdefault("OPENCV_LOG_LEVEL", "OFF")
import glob
from typing import Generator, Tuple, Optional, List
import cv2
try:
    cv2.setLogLevel(0)
except Exception:
    pass
import numpy as np


class FrameReader:
    """Memory-efficient video/image sequence reader with temporal subsampling."""

    def __init__(
        self,
        source: str,
        target_fps: Optional[float] = None,
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
    ):
        self.source = source
        self.is_dir = os.path.isdir(source)
        self.target_fps = target_fps
        self.start_sec = max(0.0, start_sec)
        self.end_sec = end_sec

        if self.is_dir:
            self._init_dir()
        else:
            self._init_video()

    def _init_dir(self):
        extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        self.files: List[str] = []
        for ext in extensions:
            self.files.extend(glob.glob(os.path.join(self.source, ext)))
        self.files.sort()
        if not self.files:
            # Check if directory contains video files (.avi, .mp4, etc.)
            video_exts = ("*.avi", "*.Avi", "*.mp4", "*.mkv")
            v_files = []
            for vext in video_exts:
                v_files.extend(glob.glob(os.path.join(self.source, vext)))
                v_files.extend(glob.glob(os.path.join(self.source, "*", vext)))
            v_files.sort()
            if v_files:
                self.is_dir = False
                self.source = v_files[0]
                self._init_video()
                return
            raise ValueError(f"No image or video files in directory: {self.source}")

        sample = cv2.imread(self.files[0])
        self.height, self.width = sample.shape[:2]
        self.native_fps = 15.0
        self.total_frames = len(self.files)
        self.duration_sec = self.total_frames / self.native_fps
        self._calculate_sampling()

    def _auto_remux_hevc(self, video_path: str) -> Optional[str]:
        """Losslessly remuxes GHEC / proprietary AVI to standard MP4 container via stream copy."""
        import subprocess
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        from .clip_sequence import REMUX_DIR
        out_mp4 = os.path.join(REMUX_DIR, f"remux_{base_name}.mp4")
        if os.path.exists(out_mp4) and os.path.getsize(out_mp4) > 1000:
            return out_mp4

        cmd = [
            "ffmpeg", "-y", "-vcodec", "hevc", "-i", video_path,
            "-c:v", "copy", "-tag:v", "hvc1", out_mp4
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
            if res.returncode == 0 and os.path.exists(out_mp4):
                return out_mp4
        except Exception:
            pass
        return None

    def _init_video(self):
        if not os.path.exists(self.source):
            raise FileNotFoundError(f"Video file not found: {self.source}")

        # Attempt lossless HEVC container remux for proprietary AVI trail-cam footage before OpenCV
        if self.source.lower().endswith((".avi", ".ghec")):
            remuxed = self._auto_remux_hevc(self.source)
            if remuxed and os.path.exists(remuxed):
                self.source = remuxed

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.source}")

        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        self.native_fps = fps if (0 < fps <= 240) else 15.0
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_sec = self.total_frames / self.native_fps if self.native_fps > 0 else 0.0
        cap.release()
        self._calculate_sampling()

    def _calculate_sampling(self):
        if self.target_fps and self.target_fps > 0 and self.target_fps < self.native_fps:
            self.step = int(round(self.native_fps / self.target_fps))
        else:
            self.step = 1

        self.start_frame = int(self.start_sec * self.native_fps)
        if self.end_sec is not None:
            self.end_frame = min(self.total_frames, int(self.end_sec * self.native_fps))
        else:
            self.end_frame = self.total_frames

    def frames(self) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """Yields (frame_index, timestamp_seconds, frame_bgr)."""
        if self.is_dir:
            for idx in range(self.start_frame, self.end_frame, self.step):
                if idx >= len(self.files):
                    break
                path = self.files[idx]
                img = cv2.imread(path)
                if img is not None:
                    timestamp = idx / self.native_fps
                    yield idx, timestamp, img
        else:
            cap = cv2.VideoCapture(self.source)
            current_frame = self.start_frame

            while current_frame < self.end_frame:
                if self.step > 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = current_frame / self.native_fps
                yield current_frame, timestamp, frame

                current_frame += self.step
            cap.release()


def open_reader(source: str, target_fps: Optional[float] = None, start_sec: float = 0.0,
                end_sec: Optional[float] = None, **sequence_kwargs):
    """A folder (or glob) of Event-named clips -> ClipSequenceReader (one continuous stream,
    tracker never restarts at file boundaries); anything else -> FrameReader."""
    from .clip_sequence import ClipSequenceReader, is_clip_folder
    if is_clip_folder(source):
        return ClipSequenceReader(source, target_fps=target_fps, start_sec=start_sec,
                                  end_sec=end_sec, **sequence_kwargs)
    return FrameReader(source, target_fps=target_fps, start_sec=start_sec, end_sec=end_sec)
