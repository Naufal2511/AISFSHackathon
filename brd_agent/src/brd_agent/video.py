"""Video files: filename parsing, a time index over one or more cameras, frame reading, and the day/minute clock.

Filename convention (from the Merck-TAMU data): Event20261107033930006.Avi = "Event" + YYYYMMDDHHMMSSmmm
-> 2026-11-07 03:39:30.006 (wall-clock start of the clip). The camera is NOT in the name; it is taken from the parent
folder (configurable map) because that is the usual layout for multi-camera DVR exports. ASSUMPTION: confirm with the real tree.
"""
from __future__ import annotations

import functools
import os
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
try:
    import cv2
except ImportError:                     # pragma: no cover
    cv2 = None

from .schemas import CAMERAS

TS_RE = re.compile(r"(\d{17})")
VIDEO_EXT = {".avi", ".mp4", ".mkv", ".mov"}


def parse_event_filename(name: str) -> Optional[datetime]:
    """'Event20261107033930006.Avi' -> datetime(2026,11,7,3,39,30,6000). None if the name has no 17-digit stamp."""
    m = TS_RE.search(Path(str(name)).stem)
    if not m:
        return None
    s = m.group(1)
    try:
        return datetime(int(s[0:4]), int(s[4:6]), int(s[6:8]), int(s[8:10]), int(s[10:12]), int(s[12:14]), int(s[14:17]) * 1000)
    except ValueError:
        return None


def parse_timestamp(value, file_start: Optional[datetime]) -> tuple[Optional[float], Optional[datetime]]:
    """YOLO's timestamp for a record -> (offset seconds inside the file, absolute datetime).
    Accepts: a number (seconds from file start), 'HH:MM:SS(.fff)' (offset in the file) or a full date-time (absolute)."""
    if value is None or str(value).strip() == "":
        return None, None
    s = str(value).strip()
    try:
        off = float(s)
        return off, (file_start + timedelta(seconds=off)) if file_start else None
    except ValueError:
        pass
    m = re.fullmatch(r"(\d+):(\d{2}):(\d{2}(?:\.\d+)?)", s)
    if m:
        off = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
        return off, (file_start + timedelta(seconds=off)) if file_start else None
    m = re.fullmatch(r"(\d{2}):(\d{2}(?:\.\d+)?)", s)
    if m:
        off = int(m.group(1)) * 60 + float(m.group(2))
        return off, (file_start + timedelta(seconds=off)) if file_start else None
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y%m%d%H%M%S%f"):
        try:
            dt = datetime.strptime(s, fmt)
            return ((dt - file_start).total_seconds() if file_start else None), dt
        except ValueError:
            continue
    return None, None


@dataclass
class Clip:
    path: str
    camera: str
    start: datetime
    duration_s: float
    width: int
    height: int
    fps: float

    @property
    def end(self) -> datetime:
        return self.start + timedelta(seconds=self.duration_s)


class Clock:
    """Maps wall-clock time <-> (day, minute). day_origin='midnight': day 1 starts at 00:00 on the first date seen;
    'first_file_start': day 1 starts at the first clip's start time. Day 2 follows day 1 contiguously."""

    def __init__(self, origin: datetime):
        self.origin = origin

    @classmethod
    def from_first(cls, first: datetime, mode: str = "midnight") -> "Clock":
        return cls(first.replace(hour=0, minute=0, second=0, microsecond=0) if mode == "midnight" else first)

    def to_dm(self, dt: datetime) -> tuple[int, int]:
        mins = int((dt - self.origin).total_seconds() // 60)
        return mins // 1440 + 1, mins % 1440

    def to_dt(self, day: int, minute: int, offset_s: float = 0.0) -> datetime:
        return self.origin + timedelta(minutes=(day - 1) * 1440 + minute, seconds=30 + offset_s)   # minute centre + offset


class VideoIndex:
    def __init__(self, clips: list[Clip]):
        self.clips = sorted(clips, key=lambda c: (c.camera, c.start))
        self.by_name = {Path(c.path).name.lower(): c for c in self.clips}
        self.cameras = sorted({c.camera for c in self.clips})
        self._lock = threading.Lock()

    @classmethod
    def scan(cls, root: str | Path, camera_map: Optional[dict[str, str]] = None, default_camera: str = "front",
             file_camera: Optional[dict[str, str]] = None) -> "VideoIndex":
        """file_camera: {file name (lower case): camera slot}, e.g. from the tracking observations. Wins over folder names, so files
        can sit in any folder layout and have different start times per camera."""
        root = Path(root)
        files = [p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXT and parse_event_filename(p.name)]
        folders = sorted({p.parent.name for p in files if p.parent != root})
        cmap = {k.lower(): v for k, v in (camera_map or {}).items()}
        for i, f in enumerate(folders):                              # ASSUMPTION: unmapped folders -> back, front, top in order
            cmap.setdefault(f.lower(), CAMERAS[i] if i < len(CAMERAS) else f.lower())
        clips = []
        for p in files:
            cam = (file_camera or {}).get(p.name.lower()) or (cmap.get(p.parent.name.lower(), default_camera) if p.parent != root else default_camera)
            info = probe(p)
            clips.append(Clip(str(p), cam, parse_event_filename(p.name), info["duration_s"], info["w"], info["h"], info["fps"]))
        return cls(clips)

    def find_by_name(self, name: str) -> Optional[Clip]:
        return self.by_name.get(Path(str(name)).name.lower())

    def locate(self, camera: str, when: datetime) -> Optional[tuple[Clip, float]]:
        for c in self.clips:
            if c.camera == camera and c.start <= when < c.end:
                return c, (when - c.start).total_seconds()
        return None

    def cameras_at(self, when: datetime) -> list[str]:
        return [cam for cam in self.cameras if self.locate(cam, when)]

    def first_start(self) -> Optional[datetime]:
        return min((c.start for c in self.clips), default=None)


_STDERR_LOCK = threading.Lock()


def _open_quiet(path):
    """cv2.VideoCapture with fd 2 muted: OpenCV 5's FFMPEG plugin prints 'Could not find decoder' for HEVC/'GHEC' clips
    and ignores OPENCV_LOG_LEVEL. Those clips are expected to fail here (ffmpeg reads them instead), so the noise is useless."""
    with _STDERR_LOCK:
        try:
            saved = os.dup(2)
        except OSError:
            return cv2.VideoCapture(str(path))
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 2)
            os.close(devnull)
            return cv2.VideoCapture(str(path))
        finally:
            os.dup2(saved, 2)
            os.close(saved)


@functools.lru_cache(maxsize=4096)
def _cv_ok(path) -> bool:
    cap = _open_quiet(path) if cv2 is not None else None
    try:
        return bool(cap is not None and cap.isOpened() and (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) > 0)
    finally:
        if cap is not None:
            cap.release()


def _need(tool: str) -> None:
    import shutil
    if shutil.which(tool) is None:
        raise RuntimeError(f"'{tool}' was not found on PATH. The real footage (HEVC) needs ffmpeg: install it "
                           f"(Windows: `winget install Gyan.FFmpeg`, macOS: `brew install ffmpeg`, Linux: `apt install ffmpeg`), "
                           f"then open a NEW terminal and check `{tool} -version`.")


def _ff(args: list[str], timeout: int = 120) -> bytes:
    import subprocess
    _need("ffmpeg")
    r = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", *args], capture_output=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode(errors="replace")[:300])
    return r.stdout


def _codec_args(path) -> list[str]:
    """Some DVR exports tag HEVC with a non-standard FourCC (the sample clip says 'GHEC'), so decoders cannot find a codec.
    Force HEVC for those; ordinary files are left to autodetect."""
    return [] if _cv_ok(path) else ["-c:v", "hevc"]


_REMUX_LOCK = threading.Lock()


@functools.lru_cache(maxsize=4096)
def _readable(path) -> str:
    """Shared remux cache (opt-in): when PEN6_REMUX_DIR is set (run_pipeline.py sets it), a clip OpenCV cannot open is
    stream-copied ONCE into <dir>/remux_<stem>.mp4 (the same file name stage 1's tracker uses, so its remuxes are reused)
    and read with OpenCV from then on: ~0.3 s per seek+frame instead of ~1.8 s for an ffmpeg process per frame on 4K HEVC.
    Lossless (no re-encode), so frames are identical. Without the variable, or if the remux fails, the original path is
    returned and the ffmpeg fallback below is used exactly as before."""
    p = str(path)
    d = os.environ.get("PEN6_REMUX_DIR")
    if not d or cv2 is None or _cv_ok(p):
        return p
    out = Path(d) / f"remux_{Path(p).stem}.mp4"
    with _REMUX_LOCK:                                      # per-cow worker threads may ask for the same clip
        if not (out.exists() and out.stat().st_size > 1000):
            import shutil
            import subprocess
            if shutil.which("ffmpeg") is None:
                return p
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(out.stem + ".part.mp4")
            try:
                r = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-vcodec", "hevc", "-i", p,
                                    "-c:v", "copy", "-tag:v", "hvc1", str(tmp)], capture_output=True, timeout=300)
                if r.returncode != 0 or not tmp.exists():
                    return p
                os.replace(tmp, out)
            except Exception:
                return p
            finally:
                if tmp.exists():
                    tmp.unlink()
    return str(out) if _cv_ok(str(out)) else p


def probe(path: str | Path) -> dict:
    path = _readable(path)
    if _cv_ok(path):
        cap = cv2.VideoCapture(str(path))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
            return {"fps": float(fps) or 25.0, "duration_s": float(n / fps) if fps else 0.0,
                    "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)}
        finally:
            cap.release()
    import json as _json
    import subprocess
    _need("ffprobe")
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,r_frame_rate,duration,nb_frames", "-of", "json", str(path)], capture_output=True)
    st = (_json.loads(r.stdout or b"{}").get("streams") or [{}])[0]
    num, _, den = str(st.get("r_frame_rate", "25/1")).partition("/")
    fps = (float(num) / float(den or 1)) if float(den or 1) else 25.0
    dur = float(st.get("duration") or 0) or (float(st.get("nb_frames") or 0) / fps if fps else 0.0)
    return {"fps": fps or 25.0, "duration_s": dur, "w": int(st.get("width") or 0), "h": int(st.get("height") or 0)}


def read_frame(path: str, offset_s: float) -> Optional[Image.Image]:
    path = _readable(path)
    if not _cv_ok(path):
        import io
        try:
            return Image.open(io.BytesIO(_ff([*_codec_args(path), "-ss", f"{max(0.0, offset_s):.3f}", "-i", str(path),
                                              "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"]))).convert("RGB")
        except Exception:
            return None
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, offset_s) * 1000.0)
        ok, bgr = cap.read()
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)) if ok else None
    finally:
        cap.release()


def read_burst(path: str, start_s: float, seconds: float, sample_hz: float = 8.0, max_side: int = 1920) -> tuple[list[np.ndarray], float, float]:
    """Sample ~sample_hz frames/second from start_s. Frames are downscaled so the long side <= max_side (4000x3000 clips would
    otherwise be huge). Returns (BGR frames, effective_hz, scale) where scale = resized/native: multiply native box coordinates by it.
    Deliberately NOT routed through _readable(): for bursts the ffmpeg path is both faster (~2.3 s vs ~4.7 s per 4 s burst on 4K)
    and exact (fps filter = true 8 Hz; the OpenCV branch steps every round(fps/hz) frames = 7.5 Hz at 30 fps), and the
    jaw-motion / rumination thresholds were set on ffmpeg-sampled bursts."""
    info = probe(path)
    sc = min(1.0, max_side / max(info["w"], info["h"], 1))
    if not _cv_ok(path):
        w, h = max(2, int(info["w"] * sc) // 2 * 2), max(2, int(info["h"] * sc) // 2 * 2)
        raw = _ff([*_codec_args(path), "-ss", f"{max(0.0, start_s):.3f}", "-t", f"{seconds:.2f}", "-i", str(path), "-vf",
                   f"fps={sample_hz},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], timeout=300)
        n = len(raw) // (w * h * 3)
        frames = [np.frombuffer(raw, np.uint8, w * h * 3, i * w * h * 3).reshape(h, w, 3) for i in range(n)]
        return frames, float(sample_hz), w / max(1, info["w"])
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(round(fps / sample_hz)))
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start_s) * 1000.0)
        for i in range(int(seconds * fps)):
            ok, f = cap.read()
            if not ok:
                break
            if i % step == 0:
                frames.append(cv2.resize(f, None, fx=sc, fy=sc) if sc < 1.0 else f)
        return frames, fps / step, sc
    finally:
        cap.release()
