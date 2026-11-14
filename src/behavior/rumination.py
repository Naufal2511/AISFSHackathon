"""Rumination tool: measure jaw movement instead of guessing it.

Ruminating cattle chew a regurgitated bolus at ~55-75 jaw movements / min (~0.9-1.25 Hz)
in long still bouts. At 1 fps this is invisible, so the tool decodes a SHORT native-fps
snippet (default 12 s) of the cow's bbox, builds a motion-energy signal per cell of a
3x3 grid over the box and looks for a dominant spectral peak in the chewing band. A
grid is used instead of a head crop because the head position is not detected.

Cost: one 12 s decode per probe; only stationary bouts >= 2 min are probed, a couple of
times each. It is opt-in (`--probe-rumination`) and doubles as an agent tool ("zoom in
on COW_03 at 07:21:10 and check whether she is chewing").
"""

from __future__ import annotations

import glob
import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .features import EVENT_RE


def index_videos(video_root: str) -> Dict[str, str]:
    """Maps Event id (and camera folder) -> raw video path."""
    idx = {}
    for p in glob.glob(os.path.join(video_root, "**", "*.[Aa]vi"), recursive=True) + glob.glob(
        os.path.join(video_root, "**", "*.mp4"), recursive=True
    ):
        m = EVENT_RE.search(os.path.basename(p))
        if m:
            idx[m.group(0)] = p
    return idx


def resolve_video(source_video: Optional[str], index: Dict[str, str]) -> Optional[str]:
    if not isinstance(source_video, str):
        return None
    m = EVENT_RE.search(source_video)
    if m and m.group(0) in index:
        return index[m.group(0)]
    remux = os.path.join("/tmp", source_video)
    return remux if os.path.exists(remux) else None


def _open(path: str) -> cv2.VideoCapture:
    if path.lower().endswith(".avi"):
        from ..video.video_reader import FrameReader  # reuses the HEVC remux path
        path = FrameReader(path).source
    return cv2.VideoCapture(path)


def chew_score(
    video_path: str,
    start_sec: float,
    bbox: Tuple[float, float, float, float],
    duration_sec: float = 12.0,
    band_hz: Tuple[float, float] = (0.75, 1.6),
    work_px: int = 96,
) -> Dict[str, float]:
    """Returns {'peak_ratio', 'peak_hz', 'fps'}; peak_ratio >> 1 means periodic chewing."""
    cap = _open(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start_sec) * 1000.0)
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    prev, sig = None, []
    for _ in range(int(duration_sec * fps)):
        ok, fr = cap.read()
        if not ok:
            break
        crop = fr[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
        if crop.size == 0:
            break
        g = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (work_px, work_px)).astype(np.float32)
        if prev is not None:
            d = np.abs(g - prev)
            s = work_px // 3
            sig.append([d[i * s:(i + 1) * s, j * s:(j + 1) * s].mean() for i in range(3) for j in range(3)])
        prev = g
    cap.release()
    if len(sig) < fps * 4:
        return {"peak_ratio": 0.0, "peak_hz": float("nan"), "fps": float(fps)}
    S = np.asarray(sig)
    S = S - S.mean(0)
    freqs = np.fft.rfftfreq(len(S), d=1.0 / fps)
    spec = np.abs(np.fft.rfft(S * np.hanning(len(S))[:, None], axis=0)) ** 2
    band = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    ref = (freqs > 0.2) & (freqs < min(5.0, fps / 2))
    if not band.any():
        return {"peak_ratio": 0.0, "peak_hz": float("nan"), "fps": float(fps)}
    ratios = spec[band].max(0) / (np.median(spec[ref], axis=0) + 1e-9)
    cell = int(ratios.argmax())
    return {
        "peak_ratio": float(ratios[cell]),
        "peak_hz": float(freqs[band][spec[band][:, cell].argmax()]),
        "fps": float(fps),
    }
