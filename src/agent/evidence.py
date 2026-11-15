"""Evidence access for the agent: frames at a wall-clock time, crops, tiled strips, embeddings."""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

from ..behavior.features import parse_clip_start
from ..behavior.rumination import index_videos
from ..video.clip_sequence import readable_path

CAM_COLORS = [(0, 0, 255), (0, 200, 0), (255, 128, 0), (0, 200, 255), (255, 0, 255), (255, 255, 0),
              (128, 0, 255), (0, 128, 255), (180, 180, 0), (0, 255, 128), (255, 255, 255)]


def cow_color(cid: Optional[str]):
    try:
        return CAM_COLORS[int(str(cid).split("_")[-1]) % len(CAM_COLORS)]
    except ValueError:
        return (200, 200, 200)


class FrameStore:
    """Frame at (camera, wall time) from the raw clips referenced by the tracking records."""

    def __init__(self, rec: pd.DataFrame, video_root: str = "input", max_open: int = 4):
        self.index = index_videos(video_root)
        self.clips: Dict[str, List[Tuple[float, str]]] = {}
        for cam, g in rec.groupby("camera_id"):
            starts = []
            for sv in g.source_video.dropna().unique():
                st = parse_clip_start(sv)
                if st is not None:
                    starts.append(((st - pd.Timestamp("1970-01-01")).total_seconds(), sv))
            self.clips[cam] = sorted(starts)
        self._caps: "OrderedDict[str, cv2.VideoCapture]" = OrderedDict()
        self._lock = threading.RLock()
        self.max_open = max_open

    def _path(self, source_video: str) -> Optional[str]:
        from ..behavior.features import EVENT_RE
        m = EVENT_RE.search(source_video)
        p = self.index.get(m.group(0)) if m else None
        if p is None:
            from ..video.clip_sequence import REMUX_DIR
            p = os.path.join(REMUX_DIR, source_video)
            return p if os.path.exists(p) else None
        return readable_path(p)[0]

    def _cap(self, path: str) -> cv2.VideoCapture:
        if path in self._caps:
            self._caps.move_to_end(path)
            return self._caps[path]
        cap = cv2.VideoCapture(path)
        self._caps[path] = cap
        while len(self._caps) > self.max_open:
            _, old = self._caps.popitem(last=False)
            old.release()
        return cap

    def frame(self, camera: str, t: float) -> Optional[np.ndarray]:
        # the open VideoCaptures are shared: one seek + read at a time (investigations run in parallel threads)
        with self._lock:
            return self._frame(camera, t)

    def _frame(self, camera: str, t: float) -> Optional[np.ndarray]:
        cl = self.clips.get(camera) or []
        cand = [c for c in cl if c[0] <= t]
        if not cand:
            return None
        start, sv = cand[-1]
        path = self._path(sv)
        if path is None:
            return None
        cap = self._cap(path)
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t - start) * 1000.0)
        ok, fr = cap.read()
        return fr if ok else None

    def close(self):
        for c in self._caps.values():
            c.release()
        self._caps.clear()


def expand(bbox, w, h, pad=0.15):
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    return (int(max(0, x1 - pad * bw)), int(max(0, y1 - pad * bh)),
            int(min(w, x2 + pad * bw)), int(min(h, y2 + pad * bh)))


def crop(frame: np.ndarray, bbox, pad=0.15, size: int = 256) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = expand(bbox, w, h, pad)
    c = frame[y1:y2, x1:x2]
    if c.size == 0:
        return np.zeros((size, size, 3), np.uint8)
    s = size / max(c.shape[:2])
    return cv2.resize(c, (max(1, int(c.shape[1] * s)), max(1, int(c.shape[0] * s))))


def label(img: np.ndarray, text: str, color=(255, 255, 255)) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, text[:48], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def tile(images: Sequence[np.ndarray], cols: int, cell: int = 256) -> np.ndarray:
    if not images:
        return np.zeros((cell, cell, 3), np.uint8)
    cells = []
    for im in images:
        s = cell / max(im.shape[:2])
        r = cv2.resize(im, (max(1, int(im.shape[1] * s)), max(1, int(im.shape[0] * s))))
        canvas = np.zeros((cell, cell, 3), np.uint8)
        canvas[: r.shape[0], : r.shape[1]] = r
        cells.append(canvas)
    while len(cells) % cols:
        cells.append(np.zeros((cell, cell, 3), np.uint8))
    rows = [np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)]
    return np.vstack(rows)


def draw_boxes(frame: np.ndarray, rows: pd.DataFrame, highlight: Optional[str] = None) -> np.ndarray:
    out = frame.copy()
    th = max(2, out.shape[1] // 700)
    for r in rows.itertuples():
        col = cow_color(r.global_cow_id)
        w = th * (3 if highlight and r.global_cow_id == highlight else 1)
        cv2.rectangle(out, (int(r.bbox_x1), int(r.bbox_y1)), (int(r.bbox_x2), int(r.bbox_y2)), col, w)
        cv2.putText(out, f"{r.global_cow_id or '?'} t{int(r.local_track_id)}", (int(r.bbox_x1), int(r.bbox_y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, out.shape[1] / 1600, col, th, cv2.LINE_AA)
    return out


class Embedder:
    """Same appearance features as the tracker (DINOv2-S + coat colour descriptor)."""

    def __init__(self, use_dino: bool = True):
        self.dino = None
        if use_dino:
            try:
                from ..tracking.reidentification import CowAppearanceExtractor
                self.dino = CowAppearanceExtractor(model_name="dinov2_vits14", device="cpu")
            except Exception:
                self.dino = None

    _lock = threading.Lock()

    def __call__(self, frame: np.ndarray, bbox) -> np.ndarray:
        from ..tracking.tracklet_identity import color_descriptor, fuse_embedding
        with Embedder._lock:
            d = self.dino.extract(frame, bbox) if self.dino is not None else None
        e = fuse_embedding(d, color_descriptor(frame, bbox))
        return e if e is not None else np.zeros(1, np.float32)


def to_jpeg_b64(img: np.ndarray, max_side: int = 1024, quality: int = 85) -> str:
    import base64
    s = max_side / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()
