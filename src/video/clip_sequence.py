"""One continuous stream per camera over many back-to-back 5-minute clips.

The cameras write `Event<YYYYMMDDhhmmss><cam>.Avi` files every ~5 min. Processing them one
file at a time restarts the tracker at every file boundary, so identity has to be re-guessed
from appearance 12 times an hour. This reader plays a camera's clips in wall-clock order as
ONE stream:

* timestamps are seconds since a shared `origin` (a wall-clock datetime), so several cameras
  built with the same origin and fps produce frames on the same tick grid (cam1/cam3 clip
  names differ by ~10 s; ticks still line up);
* frames are decoded sequentially (grab/retrieve), ~2x faster than seeking per sample;
* `current_clip` / `clip_offset_sec` say which file and where inside it the last frame came
  from (records keep the convention wall_time = Event time of source_video + timestamp);
* `segment_id` increments when footage has a hole longer than `max_gap_sec`, so the caller
  can reset its tracker there instead of matching boxes across the gap;
* with `emit_gaps=True` the generator yields (idx, t, None) for ticks with no footage, which
  keeps several cameras in lock-step.
"""

from __future__ import annotations

import glob
import math
import os
import re
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Generator, Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np

EVENT_RE = re.compile(r"Event(\d{14})")
VIDEO_EXTS = (".avi", ".mp4", ".mkv", ".mov")
# Remuxed MP4 copies of the HEVC AVIs. /tmp on Mac/Linux (unchanged); on Windows "/tmp" means C:/tmp, which usually
# does not exist (every remux then fails and OpenCV cannot open the clips), so use the system temp folder there.
# PEN6_REMUX_DIR overrides both (run_pipeline.py points it at a per-run cache so later steps reuse the remuxes).
REMUX_DIR = os.environ.get("PEN6_REMUX_DIR") or ("/tmp" if os.name != "nt" else tempfile.gettempdir())
os.makedirs(REMUX_DIR, exist_ok=True)


def parse_event_time(name: str) -> Optional[datetime]:
    m = EVENT_RE.search(os.path.basename(name))
    return datetime.strptime(m.group(1), "%Y%m%d%H%M%S") if m else None


def parse_wall(value: Union[None, str, datetime]) -> Optional[datetime]:
    if value is None or isinstance(value, datetime):
        return value
    v = value.strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(v, fmt)
        except ValueError:
            pass
    raise ValueError(f"Unrecognised time '{value}' (use 'YYYY-MM-DD HH:MM[:SS]')")


@dataclass
class Clip:
    start: datetime
    path: str


def list_clips(source: Union[str, Iterable[str]]) -> List[Clip]:
    """Directory (searched recursively), glob pattern, single file or list of files ->
    clips sorted by their Event time, duplicates (same file name) removed."""
    if isinstance(source, str):
        if os.path.isdir(source):
            paths = [os.path.join(r, f) for r, _, fs in os.walk(source) for f in fs]
        elif any(ch in source for ch in "*?["):
            paths = glob.glob(source, recursive=True)
        else:
            paths = [source]
    else:
        paths = list(source)
    clips, seen = [], set()
    for p in paths:
        if not p.lower().endswith(VIDEO_EXTS) or os.path.basename(p).startswith("."):
            continue
        t = parse_event_time(p)
        if t is None or os.path.basename(p) in seen:
            continue
        seen.add(os.path.basename(p))
        clips.append(Clip(t, p))
    clips.sort(key=lambda c: c.start)
    return clips


def is_clip_folder(source: str) -> bool:
    """True when `source` is a folder (or glob) of Event-named video clips."""
    if isinstance(source, str) and (os.path.isdir(source) or any(ch in source for ch in "*?[")):
        return len(list_clips(source)) > 0
    return False


def readable_path(path: str) -> Tuple[str, bool]:
    """OpenCV cannot open the cameras' HEVC 'GHEC' AVI; stream-copy it into an MP4.
    Returns (path to open, whether this call created the file)."""
    if not path.lower().endswith((".avi", ".ghec")):
        return path, False
    out = os.path.join(REMUX_DIR, f"remux_{os.path.splitext(os.path.basename(path))[0]}.mp4")
    if os.path.exists(out) and os.path.getsize(out) > 1000:
        return out, False
    tmp = out + ".part.mp4"
    cmd = ["ffmpeg", "-v", "error", "-y", "-vcodec", "hevc", "-i", path,
           "-c:v", "copy", "-tag:v", "hvc1", tmp]
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        if r.returncode == 0 and os.path.exists(tmp):
            os.replace(tmp, out)
            return out, True
    except Exception:
        pass
    if os.path.exists(tmp):
        os.remove(tmp)
    return path, False


class ClipSequenceReader:
    """Continuous, wall-clock-aligned frame stream over a camera's clips (see module doc)."""

    def __init__(
        self,
        source: Union[str, Iterable[str]],
        target_fps: Optional[float] = None,
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
        origin: Union[None, str, datetime] = None,
        wall_start: Union[None, str, datetime] = None,
        wall_end: Union[None, str, datetime] = None,
        emit_gaps: bool = False,
        max_gap_sec: float = 10.0,
        keep_remux: Optional[bool] = None,
        max_clip_sec: float = 900.0,
        first_index: int = 0,
        segment_id: int = 0,
    ):
        # first_index / segment_id: continue a stream resumed from a tracking checkpoint (same frame numbers)
        self.first_index = first_index
        self.all_clips = list_clips(source)
        if not self.all_clips:
            raise ValueError(f"No Event-named video clips found in {source}")
        self.wall_start = parse_wall(wall_start)
        self.wall_end = parse_wall(wall_end)
        self.clips = [
            c for c in self.all_clips
            if (self.wall_end is None or c.start < self.wall_end)
            and (self.wall_start is None
                 or (c.start - self.wall_start).total_seconds() > -max_clip_sec)
        ]
        if not self.clips:
            raise ValueError(f"No clips of {source} fall in {self.wall_start} .. {self.wall_end}")
        self.origin = parse_wall(origin) or self.wall_start or self.clips[0].start
        self.target_fps = target_fps
        self.emit_gaps = emit_gaps
        self.max_gap_sec = max_gap_sec
        # None = PEN6_KEEP_REMUX env (run_pipeline.py sets 1 so identity / rumination / stage 2 reuse the remuxes)
        self.keep_remux = keep_remux if keep_remux is not None else os.environ.get("PEN6_KEEP_REMUX") == "1"

        t0 = max(0.0, start_sec)
        if self.wall_start is not None:
            t0 = max(t0, (self.wall_start - self.origin).total_seconds())
        t1 = math.inf if end_sec is None else end_sec
        if self.wall_end is not None:
            t1 = min(t1, (self.wall_end - self.origin).total_seconds())
        self.t_start, self.t_end = t0, t1

        # Geometry / fps from the first clip (all clips of one camera share them)
        path, made0 = readable_path(self.clips[0].path)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Could not open {self.clips[0].path}")
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        self.native_fps = fps if 0 < fps <= 240 else 15.0
        cap.release()

        self.is_dir = True
        self.source = self.clips[0].path  # compatibility with FrameReader users
        self.current_clip: Optional[str] = None
        self.clip_offset_sec = 0.0
        self.segment_id = segment_id
        self.duration_sec = (
            (self.clips[-1].start - self.origin).total_seconds() + 300.0 - self.t_start
        )
        self._prefetch: Optional[threading.Thread] = None
        self._made: set = {path} if made0 else set()  # remux files this reader created (deleted after use)

    # ------------------------------------------------------------------ helpers
    def _rel(self, c: Clip) -> float:
        return (c.start - self.origin).total_seconds()

    def _start_prefetch(self, i: int):
        if i < len(self.clips):
            def work(p=self.clips[i].path):
                out, created = readable_path(p)
                if created:
                    self._made.add(out)
            self._prefetch = threading.Thread(target=work, daemon=True)
            self._prefetch.start()

    def summary(self) -> str:
        return (f"{len(self.clips)} clips {self.clips[0].start:%Y-%m-%d %H:%M:%S} .. "
                f"{self.clips[-1].start:%H:%M:%S}, origin {self.origin:%Y-%m-%d %H:%M:%S}")

    # ------------------------------------------------------------------ stream
    def frames(self) -> Generator[Tuple[int, float, Optional[np.ndarray]], None, None]:
        """Yields (frame_index, seconds_since_origin, frame_bgr or None for a gap tick)."""
        dt = 1.0 / self.target_fps if self.target_fps and self.target_fps < self.native_fps else None
        gap_dt = dt or 1.0 / self.native_fps
        idx = self.first_index
        next_tick = math.ceil(self.t_start / gap_dt - 1e-9) * gap_dt  # shared grid anchored at origin
        last_t: Optional[float] = None

        for i, clip in enumerate(self.clips):
            S = self._rel(clip)
            nxt = self._rel(self.clips[i + 1]) if i + 1 < len(self.clips) else math.inf
            if S >= self.t_end:
                break
            if nxt <= self.t_start:
                continue
            if self._prefetch is not None:
                self._prefetch.join()
            path, created = readable_path(clip.path)
            created = created or path in self._made
            self._start_prefetch(i + 1)
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                continue
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or self.native_fps
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or int(300 * fps)
            clip_end = min(S + n / fps, nxt, self.t_end)

            # ticks before this clip starts: a hole in the footage
            if self.emit_gaps:
                while next_tick < S and next_tick < self.t_end:
                    yield idx, next_tick, None
                    idx += 1
                    next_tick += gap_dt
            if last_t is not None and S - last_t > self.max_gap_sec:
                self.segment_id += 1

            pos = 0
            if dt is None:  # every native frame
                first = max(0, int(math.ceil((max(S, self.t_start) - S) * fps - 1e-6)))
                while pos < first and cap.grab():
                    pos += 1
                while True:
                    t = S + pos / fps
                    if t >= clip_end:
                        break
                    ok, frame = cap.read()
                    if not ok:
                        break
                    pos += 1
                    self.current_clip, self.clip_offset_sec = os.path.basename(clip.path), t - S
                    last_t = t
                    yield idx, t, frame
                    idx += 1
                next_tick = max(next_tick, S + pos / fps)
            else:
                next_tick = max(next_tick, math.ceil(max(S, self.t_start) / dt - 1e-9) * dt)
                while next_tick < clip_end:
                    target = int(round((next_tick - S) * fps))
                    if target >= n:
                        break
                    ok = True
                    if target - pos > 2 * fps:  # long way in (start time inside a clip): seek once
                        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                        pos = target
                    while pos < target and ok:
                        ok = cap.grab()
                        pos += 1
                    if not ok:
                        break
                    ok, frame = cap.read()
                    pos += 1
                    if not ok:
                        break
                    self.current_clip, self.clip_offset_sec = os.path.basename(clip.path), next_tick - S
                    last_t = next_tick
                    yield idx, next_tick, frame
                    idx += 1
                    next_tick += dt
            cap.release()
            if created and not self.keep_remux and os.path.exists(path):
                os.remove(path)

        if self.emit_gaps and math.isfinite(self.t_end):
            while next_tick < self.t_end:
                yield idx, next_tick, None
                idx += 1
                next_tick += gap_dt
