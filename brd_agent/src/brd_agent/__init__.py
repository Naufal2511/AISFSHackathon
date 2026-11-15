"""Agentic behavior labeling for the BRD hackathon (tier 2 of the pipeline)."""
import os as _os

# Must be set before ANY module imports cv2 (night.py loads it before video.py): OpenCV reads these once at import.
_os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
_os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
_os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

__version__ = "0.1.0"
