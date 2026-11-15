"""Night / infrared handling. Night footage is a separate domain: colour cues vanish, contrast collapses and fine cues
(jaw motion, ear position, nasal discharge) are unreliable. We (1) detect it from the frame itself (not the clock),
(2) enhance contrast before images reach the VLM, (3) cap confidence and demand measured support for fine-grained labels."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

try:                                    # OpenCV is optional; PIL fallback keeps tests/CI light
    import cv2
except ImportError:                     # pragma: no cover
    cv2 = None


def frame_stats(img: Image.Image, cfg) -> dict:
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    mx, mn = a.max(axis=2), a.min(axis=2)
    sat = float(((mx - mn) / (mx + 1e-6)).mean())
    brightness = float(lum.mean())
    ir_like = sat < cfg.ir_saturation_thresh
    is_night = brightness < cfg.brightness_thresh or (ir_like and brightness < cfg.brightness_thresh * 1.6)
    return {"brightness": round(brightness, 1), "saturation": round(sat, 3), "ir_like": bool(ir_like), "is_night": bool(is_night)}


def enhance(img: Image.Image, cfg) -> Image.Image:
    """CLAHE on luminance + gamma lift. Deterministic; does not invent detail, only stretches contrast."""
    rgb = img.convert("RGB")
    if cv2 is None:
        out = ImageOps.autocontrast(rgb, cutoff=1)
        return out.point(lambda v: int(255 * (v / 255.0) ** cfg.gamma))
    lab = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=(8, 8)).apply(lab[..., 0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    table = np.array([min(255, int(255 * (i / 255.0) ** cfg.gamma)) for i in range(256)], dtype=np.uint8)
    return Image.fromarray(cv2.LUT(out, table))


def scene_stats(backend, cow_id: str, day: int, minute: int, visible_frac: dict, cfg):
    """Brightness / saturation of a whole-scene frame at this minute, from the best-visible camera. (None, None) if no frame."""
    cam = max(visible_frac, key=visible_frac.get) if visible_frac else None
    if cam is None:
        for c in ("front", "back", "top"):
            try:
                if backend.crop(cow_id, day, minute, c, 0.0, "scene", 1.0):
                    cam = c
                    break
            except Exception:
                continue
    if cam is None:
        return None, None
    try:
        path = backend.crop(cow_id, day, minute, cam, 0.0, "scene", 1.0)
    except Exception:
        path = None
    if not path:
        return cam, None
    return cam, frame_stats(Image.open(path), cfg)
