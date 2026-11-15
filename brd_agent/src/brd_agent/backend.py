"""Data access contract between the agent and your teammate's pipeline.

The agent never touches raw video. It asks a DataBackend for crops, zone facts, track history, etc.
Implement `DataBackend` against the real tracker/identity outputs; use `MockBackend` until they land.
All methods are keyed by (cow_id, day, minute); camera is one of 'back' | 'front' | 'top'.
Every returned image is a FILE PATH. Every fact must be a JSON-serialisable primitive.
"""
from __future__ import annotations

import tempfile
import zlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw

from .schemas import CAMERAS


class DataBackend(ABC):
    @abstractmethod
    def visibility(self, cow_id: str, day: int, minute: int) -> dict[str, float]:
        """Visible fraction of the cow per camera over the minute (0..1), after occlusion/edge handling."""

    @abstractmethod
    def crop(self, cow_id: str, day: int, minute: int, camera: str, t_offset_s: float = 0.0,
             region: str = "body", scale: float = 1.0) -> Optional[str]:
        """Path to a crop at (minute centre + t_offset_s). region: body | head | legs | scene. None if unavailable."""

    @abstractmethod
    def filmstrip(self, cow_id: str, day: int, minute: int, camera: str, before_s: float,
                  after_s: float, n_frames: int) -> list[str]:
        """n_frames evenly spaced body crops between centre-before_s and centre+after_s."""

    @abstractmethod
    def zone_info(self, cow_id: str, day: int, minute: int) -> dict[str, Any]:
        """Keys: zone_fractions{feed_bunk,waterer,resting,open}, head_in_bunk_frac, head_in_waterer_frac,
        dist_bunk_px, dist_waterer_px."""

    @abstractmethod
    def track_history(self, cow_id: str, day: int, minute: int, window_s: float) -> dict[str, Any]:
        """Keys: speed_mean, speed_max, displacement_px, posture_mode (lying|standing|unknown), posture_changes."""

    @abstractmethod
    def neighbors(self, cow_id: str, day: int, minute: int) -> dict[str, Any]:
        """Keys: nn_dist_px, n_within_radius, contact_s, nn_cow_id."""

    @abstractmethod
    def burst_motion(self, cow_id: str, day: int, minute: int, camera: str, seconds: float) -> dict[str, Any]:
        """High-fps burst analysis of the head ROI. Keys: jaw_motion_score (0..1), periodicity_hz,
        head_motion_score, body_motion_score. Wrap src/brd/behavior/rumination.py here."""

    @abstractmethod
    def exemplar_similarity(self, cow_id: str, crop_path: str) -> Optional[float]:
        """Similarity between the crop and this cow's gallery exemplars (0..1). None = no gallery available."""


    def recover_box(self, cow_id: str, day: int, minute: int) -> dict:
        """Optional: try to find a cow YOLO missed in this minute. Returns {ok, camera, box, reason}."""
        return {"ok": False, "reason": "this backend cannot recover missed detections"}

    def exemplar_scores(self, crop_path: str) -> Optional[dict[str, float]]:
        """Optional: similarity of the crop to EVERY cow's gallery (to detect an ID swap). None = not available."""
        return None


# ----------------------------------------------------------------------------- MOCK
_DEFAULT = {
    "head_in_bunk_frac": 0.0, "head_in_waterer_frac": 0.0,
    "zone_fractions": {"feed_bunk": 0.0, "waterer": 0.0, "resting": 0.3, "open": 0.7},
    "dist_bunk_px": 300.0, "dist_waterer_px": 400.0,
    "speed_mean": 3.0, "speed_max": 8.0, "displacement_px": 20.0, "posture_mode": "standing",
    "posture_changes": 0, "nn_dist_px": 200.0, "n_within_radius": 0, "contact_s": 0.0, "nn_cow_id": "C01",
    "jaw_motion_score": 0.05, "periodicity_hz": 0.0, "head_motion_score": 0.1, "body_motion_score": 0.05,
    "exemplar_sim": 0.85, "visibility": {"back": 0.6, "front": 0.8, "top": 0.7}, "brightness": 120.0,
}


class MockBackend(DataBackend):
    """Scenario-driven backend for development/tests. `scenarios[(cow, day, minute)]` overrides defaults."""

    def __init__(self, scenarios: Optional[dict[tuple, dict]] = None, workdir: Optional[str] = None):
        self.scenarios = scenarios or {}
        self.dir = Path(workdir or tempfile.mkdtemp(prefix="mockbackend_"))
        self.dir.mkdir(parents=True, exist_ok=True)

    def _s(self, cow_id, day, minute) -> dict:
        s = dict(_DEFAULT)
        s.update(self.scenarios.get((cow_id, day, minute), {}))
        return s

    def _img(self, text: str, dark: bool = False, size=(320, 240)) -> str:
        name = f"{zlib.crc32(text.encode()):08x}.png"
        p = self.dir / name
        if not p.exists():
            arr = np.full((size[1], size[0], 3), 25 if dark else 150, dtype=np.uint8)
            im = Image.fromarray(arr)
            ImageDraw.Draw(im).text((8, size[1] // 2), text, fill=(255, 255, 255))
            im.save(p)
        return str(p)

    def visibility(self, cow_id, day, minute):
        return dict(self._s(cow_id, day, minute)["visibility"])

    def crop(self, cow_id, day, minute, camera, t_offset_s=0.0, region="body", scale=1.0):
        s = self._s(cow_id, day, minute)
        if s["visibility"].get(camera, 0.0) <= 0.0:
            return None
        return self._img(f"{cow_id} d{day} m{minute} {camera} {region} t{t_offset_s:+.0f} x{scale}",
                         dark=s["brightness"] < 60)

    def filmstrip(self, cow_id, day, minute, camera, before_s, after_s, n_frames):
        ts = np.linspace(-before_s, after_s, n_frames)
        return [p for t in ts if (p := self.crop(cow_id, day, minute, camera, float(t))) is not None]

    def zone_info(self, cow_id, day, minute):
        s = self._s(cow_id, day, minute)
        return {k: s[k] for k in ("zone_fractions", "head_in_bunk_frac", "head_in_waterer_frac",
                                  "dist_bunk_px", "dist_waterer_px")}

    def track_history(self, cow_id, day, minute, window_s):
        s = self._s(cow_id, day, minute)
        return {k: s[k] for k in ("speed_mean", "speed_max", "displacement_px", "posture_mode", "posture_changes")}

    def neighbors(self, cow_id, day, minute):
        s = self._s(cow_id, day, minute)
        return {k: s[k] for k in ("nn_dist_px", "n_within_radius", "contact_s", "nn_cow_id")}

    def burst_motion(self, cow_id, day, minute, camera, seconds):
        s = self._s(cow_id, day, minute)
        return {k: s[k] for k in ("jaw_motion_score", "periodicity_hz", "head_motion_score", "body_motion_score")}

    def exemplar_scores(self, crop_path):
        for (_, _, _), s in self.scenarios.items():
            if "exemplar_scores" in s:
                return dict(s["exemplar_scores"])
        return None

    def exemplar_similarity(self, cow_id, crop_path):
        for (c, d, m), s in self.scenarios.items():
            if c == cow_id and "exemplar_sim" in s:
                return float(s["exemplar_sim"])
        return float(_DEFAULT["exemplar_sim"])
