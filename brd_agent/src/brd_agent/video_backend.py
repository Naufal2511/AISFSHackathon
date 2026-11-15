"""DataBackend over REAL video + the YOLO records: the agent's eyes. It never trusts YOLO's *label*, only its boxes/ids as
pointers to where the cow is, and measures everything else from pixels (crops, motion bursts) or box geometry (speed, contact).

ASSUMPTIONS (flagged because no real YOLO output/pen map was available when written):
  * box coordinates are in the pixel space of the video file they came from;
  * posture is NOT inferred from box shape (unreliable across cameras): it is left 'unknown' and judged from images;
  * zones need a pen map (YAML polygons per camera); without it zone facts are reported as unavailable, not as zero.
"""
from __future__ import annotations

import json
import math
import tempfile
import zlib
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import yaml
from PIL import Image

from .backend import DataBackend
from .config import VideoCfg
from .store import Store
from .video import Clock, Clip, VideoIndex, read_burst, read_frame

Box = tuple[float, float, float, float]
Detector = Callable[[Image.Image], list[tuple[float, float, float, float, float]]]   # -> [(x1,y1,x2,y2,conf)]


def _poly_frac(boxes: list[Box], poly: list[list[float]]) -> float:
    if not boxes or not poly:
        return 0.0
    import cv2
    pts = np.array(poly, dtype=np.float32)
    hit = sum(1 for b in boxes if cv2.pointPolygonTest(pts, ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), False) >= 0)
    return hit / len(boxes)


class VideoBackend(DataBackend):
    def __init__(self, store_dir: str, index: VideoIndex, cfg: VideoCfg, workdir: Optional[str] = None,
                 detector: Optional[Detector] = None, exemplar_dir: Optional[str] = None):
        self.store_dir, self.index, self.cfg, self.detector = str(store_dir), index, cfg, detector
        self.db = Store(store_dir)       # read-mostly; raw YOLO shards are cached process-wide, so this is cheap and thread-safe
        self.dir = Path(workdir or tempfile.mkdtemp(prefix="video_backend_"))
        self.dir.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        v = self.db.get_meta("clock_origin")
        self.clock = Clock(datetime.fromisoformat(v)) if v else Clock.from_first(index.first_start() or datetime.now(), cfg.day_origin)
        self.pen = yaml.safe_load(Path(cfg.pen_map).read_text(encoding="utf-8")) if cfg.pen_map and Path(cfg.pen_map).exists() else {}
        self.gallery = self._load_gallery(exemplar_dir) if exemplar_dir else {}

    # ------------------------------------------------------------------ records
    def _records(self, cow: str, day: int, minute: int, camera: Optional[str] = None, lo: float = 0.0, hi: float = 60.0) -> list[dict]:
        rows = [r.as_dict() for r in self.db.yolo_minute(cow, day, minute) if not camera or r.camera == camera]
        t0 = self.clock.to_dt(day, minute, -30.0)
        out = []
        for r in rows:
            from datetime import datetime
            sec = (datetime.fromisoformat(r["abs_time"]) - t0).total_seconds()
            if lo <= sec <= hi:
                r["sec"] = sec
                out.append(r)
        return sorted(out, key=lambda r: r["sec"])

    def _box_at(self, cow: str, day: int, minute: int, camera: str, when) -> Optional[Box]:
        """Nearest YOLO box for this cow in this camera within 6 s of `when`."""
        recs = [r for r in self._records(cow, day, minute, camera, -10, 70) if None not in (r["x1"], r["x2"])]
        if not recs:                                   # maybe the neighbouring minute
            return None
        from datetime import datetime
        best = min(recs, key=lambda r: abs((datetime.fromisoformat(r["abs_time"]) - when).total_seconds()))
        if abs((datetime.fromisoformat(best["abs_time"]) - when).total_seconds()) > 6.0:
            return None
        return (best["x1"], best["y1"], best["x2"], best["y2"])

    def _hz(self) -> float:
        if self.cfg.record_hz:
            return float(self.cfg.record_hz)
        v = self.db.get_meta("record_hz")
        return float(v) if v else 1.0

    # ------------------------------------------------------------------ DataBackend
    def visibility(self, cow_id, day, minute):
        r = self.db.get_row(cow_id, day, minute)
        return {c: (r.get(f"vis_{c}") or 0.0) for c in ("back", "front", "top")} if r else {}

    def _frame(self, camera: str, when) -> Optional[tuple[Image.Image, Clip, float]]:
        loc = self.index.locate(camera, when)
        if not loc:
            return None
        clip, off = loc
        im = read_frame(clip.path, off)
        return (im, clip, off) if im is not None else None

    def crop(self, cow_id, day, minute, camera, t_offset_s=0.0, region="body", scale=1.0):
        when = self.clock.to_dt(day, minute, t_offset_s)
        fr = self._frame(camera, when)
        if not fr:
            return None
        im, clip, off = fr
        key = f"{cow_id}{day}{minute}{camera}{t_offset_s}{region}{scale}"
        outp = self.dir / f"crop_{zlib.crc32(key.encode()):08x}.png"
        if region == "scene":
            im.thumbnail((self.cfg.frame_max_side, self.cfg.frame_max_side))
            im.save(outp)
            return str(outp)
        box = self._box_at(cow_id, day, minute, camera, when)
        if not box:
            return None
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        if region == "head":
            y2 = y1 + h * self.cfg.head_frac
        elif region == "legs":
            y1 = y2 - h * 0.35
        pad = self.cfg.crop_pad
        W, H = im.size
        c = im.crop((max(0, x1 - pad * w), max(0, y1 - pad * h), min(W, x2 + pad * w), min(H, y2 + pad * h)))
        if scale != 1.0 or min(c.size) < 224:
            f = max(scale, 224 / max(1, min(c.size)))
            c = c.resize((int(c.size[0] * f), int(c.size[1] * f)), Image.BICUBIC)
        c.save(outp)
        return str(outp)

    def filmstrip(self, cow_id, day, minute, camera, before_s, after_s, n_frames):
        ts = np.linspace(-before_s, after_s, n_frames)
        return [p for t in ts if (p := self.crop(cow_id, day, minute, camera, float(t))) is not None]

    def zone_info(self, cow_id, day, minute):
        if not self.pen:
            row = self.db.get_row(cow_id, day, minute) or {}
            if row.get("zone_feed_bunk") is not None:      # tracking input: zone per second already fused over cameras (body position)
                zf = {"feed_bunk": row["zone_feed_bunk"], "waterer": row.get("zone_waterer") or 0.0,
                      "resting": row.get("zone_resting") or 0.0, "open": row.get("zone_open") or 0.0}
                return {"available": True, "zone_fractions": zf, "head_in_bunk_frac": zf["feed_bunk"], "source": "fused_state",
                        "note": "zones come from the tracking state (body position, no separate waterer zone); head position not measured"}
            return {"available": False, "note": "no pen map configured"}
        res = {"available": True}
        best = None
        for cam, zones in self.pen.items():
            recs = [r for r in self._records(cow_id, day, minute, cam) if None not in (r["x1"], r["x2"])]
            if recs and (best is None or len(recs) > len(best[2])):
                best = (cam, zones, recs)
        if not best:
            return {"available": False, "note": "no boxes for this cow-minute"}
        cam, zones, recs = best
        boxes = [(r["x1"], r["y1"], r["x2"], r["y2"]) for r in recs]
        heads = [(b[0], b[1], b[2], b[1] + (b[3] - b[1]) * self.cfg.head_frac) for b in boxes]
        zf = {z: round(_poly_frac(boxes, zones.get(z, [])), 3) for z in ("feed_bunk", "waterer", "resting")}
        zf["open"] = round(max(0.0, 1.0 - sum(zf.values())), 3)
        res.update({"zone_fractions": zf, "camera": cam,
                    "head_in_bunk_frac": round(_poly_frac(heads, zones.get("feed_bunk", [])), 3),
                    "head_in_waterer_frac": round(_poly_frac(heads, zones.get("waterer", [])), 3)})
        return res

    def track_history(self, cow_id, day, minute, window_s):
        allrecs = self._records(cow_id, day, minute, None, 30 - window_s / 2, 30 + window_s / 2)
        fused = sorted({(round(r["sec"]), r["speed_mps"]) for r in allrecs if r["speed_mps"] is not None})
        if len(fused) >= 2:                      # tracking input: metric speed fused over cameras (m/s), independent of camera resolution
            sp = [v for _, v in fused]
            wx = [(r["sec"], r["world_x"], r["world_y"]) for r in allrecs if r["world_x"] is not None]
            disp = math.hypot(wx[-1][1] - wx[0][1], wx[-1][2] - wx[0][2]) if len(wx) >= 2 else 0.0
            return {"speed_mean": round(float(np.mean(sp)), 3), "speed_max": round(float(np.max(sp)), 3), "displacement_m": round(disp, 2),
                    "unit": "m/s", "posture_mode": "unknown", "posture_changes": 0, "n_records": len(sp),
                    "camera": max({r["camera"] for r in allrecs}, key=lambda c: sum(r["camera"] == c for r in allrecs))}
        recs = [r for r in allrecs if None not in (r["x1"], r["x2"])]
        if len(recs) < 2:
            return {"speed_mean": 0.0, "speed_max": 0.0, "displacement_px": 0.0, "posture_mode": "unknown", "posture_changes": 0,
                    "n_records": len(recs)}
        cam = max({r["camera"] for r in recs}, key=lambda c: sum(r["camera"] == c for r in recs))
        rs = [r for r in recs if r["camera"] == cam]
        sp, path = [], 0.0
        for a, b in zip(rs, rs[1:]):
            dt = b["sec"] - a["sec"]
            if 0 < dt <= 3.0:
                d = math.hypot((b["x1"] + b["x2"]) / 2 - (a["x1"] + a["x2"]) / 2, (b["y1"] + b["y2"]) / 2 - (a["y1"] + a["y2"]) / 2)
                sp.append(d / dt); path += d
        disp = math.hypot((rs[-1]["x1"] + rs[-1]["x2"]) / 2 - (rs[0]["x1"] + rs[0]["x2"]) / 2,
                          (rs[-1]["y1"] + rs[-1]["y2"]) / 2 - (rs[0]["y1"] + rs[0]["y2"]) / 2)
        return {"speed_mean": round(float(np.mean(sp)), 2) if sp else 0.0, "speed_max": round(float(np.max(sp)), 2) if sp else 0.0,
                "displacement_px": round(disp, 1), "path_px": round(path, 1), "posture_mode": "unknown", "posture_changes": 0,
                "camera": cam, "n_records": len(rs)}

    def neighbors(self, cow_id, day, minute):
        others = [r.as_dict() for r in self.db.yolo_all_minute(day, minute, exclude_cow=cow_id)]
        mine = [r for r in self._records(cow_id, day, minute) if None not in (r["x1"], r["x2"])]
        if not mine or not others:
            return {"nn_dist_px": None, "n_within_radius": 0, "contact_s": 0.0, "nn_cow_id": None}
        by_key = {}
        for o in others:
            by_key.setdefault((o["file"], round(o["t_offset"] or 0)), []).append(o)
        dists, contact, within = [], 0, set()
        for r in mine:
            for o in by_key.get((r["file"], round(r["t_offset"] or 0)), []):
                if None in (o["x1"], o["x2"]):
                    continue
                cx, cy = (r["x1"] + r["x2"]) / 2, (r["y1"] + r["y2"]) / 2
                ox, oy = (o["x1"] + o["x2"]) / 2, (o["y1"] + o["y2"]) / 2
                d = math.hypot(cx - ox, cy - oy)
                dists.append((d, o["cow_id"]))
                wd = max(r["x2"] - r["x1"], r["y2"] - r["y1"])
                if d < 1.5 * wd:
                    within.add(o["cow_id"])
                gap_x = max(0.0, max(r["x1"], o["x1"]) - min(r["x2"], o["x2"]))
                gap_y = max(0.0, max(r["y1"], o["y1"]) - min(r["y2"], o["y2"]))
                if math.hypot(gap_x, gap_y) < 0.1 * wd:
                    contact += 1
        if not dists:
            return {"nn_dist_px": None, "n_within_radius": 0, "contact_s": 0.0, "nn_cow_id": None}
        d, who = min(dists)
        return {"nn_dist_px": round(d, 1), "n_within_radius": len(within), "contact_s": round(contact / max(1e-6, self._hz()), 1),
                "nn_cow_id": who}

    def burst_motion(self, cow_id, day, minute, camera, seconds):
        import cv2
        when = self.clock.to_dt(day, minute, -seconds / 2)
        loc = self.index.locate(camera, when)
        box = self._box_at(cow_id, day, minute, camera, self.clock.to_dt(day, minute))
        if not loc or not box:
            return {"jaw_motion_score": 0.0, "periodicity_hz": 0.0, "head_motion_score": 0.0, "body_motion_score": 0.0, "available": False}
        clip, off = loc
        frames, hz, sc = read_burst(clip.path, off, seconds, sample_hz=8.0)
        if len(frames) < 8:
            return {"jaw_motion_score": 0.0, "periodicity_hz": 0.0, "head_motion_score": 0.0, "body_motion_score": 0.0, "available": False}
        x1, y1, x2, y2 = [int(v * sc) for v in box]      # boxes are in native pixels; frames were downscaled by sc
        hy2 = int(y1 + (y2 - y1) * self.cfg.head_frac)

        def roi(f, ya, yb):
            return cv2.resize(cv2.cvtColor(f[max(0, ya):max(ya + 2, yb), max(0, x1):max(x1 + 2, x2)], cv2.COLOR_BGR2GRAY), (48, 48)).astype(np.float32)
        head = [roi(f, y1, hy2) for f in frames]
        body = [roi(f, y1, y2) for f in frames]
        hd = np.array([np.abs(a - b).mean() for a, b in zip(head, head[1:])])
        bd = np.array([np.abs(a - b).mean() for a, b in zip(body, body[1:])])
        # dominant temporal component of the head ROI (PCA over time): periodic jaw/head movement shows up as one
        # oscillating component, whereas frame differencing would double the frequency of a rectified signal
        X = np.stack([h.ravel() for h in head]); X = X - X.mean(axis=0)
        u, sv, _ = np.linalg.svd(X, full_matrices=False)
        sig = u[:, 0] * sv[0]
        spec = np.abs(np.fft.rfft((sig - sig.mean()) * np.hanning(len(sig)))) ** 2
        freqs = np.fft.rfftfreq(len(sig), d=1.0 / hz)
        band = (freqs >= 0.6) & (freqs <= 1.8)           # cud-chewing is ~1 cycle/s
        tot = spec[1:].sum() + 1e-9
        ratio = float(spec[band].sum() / tot) if band.any() else 0.0
        peak = float(freqs[band][np.argmax(spec[band])]) if band.any() and spec[band].max() > 0 else 0.0
        head_motion = float(min(1.0, hd.mean() / 12.0))
        # chewing = strong periodic head motion in the cud band, but little whole-body motion (not walking)
        body_motion = float(min(1.0, bd.mean() / 12.0))
        jaw = float(min(1.0, ratio * 1.5) * min(1.0, head_motion * 4) * (1.0 - 0.5 * body_motion))
        return {"jaw_motion_score": round(jaw, 3), "periodicity_hz": round(peak, 2), "head_motion_score": round(head_motion, 3),
                "body_motion_score": round(body_motion, 3), "band_power_ratio": round(ratio, 3), "available": True}

    # ------------------------------------------------------------------ identity (optional gallery)
    @staticmethod
    def _hist(path: str) -> np.ndarray:
        import cv2
        bgr = cv2.imread(path)
        hsv = cv2.cvtColor(cv2.resize(bgr, (96, 96)), cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
        return h / (h.sum() + 1e-9)

    def _load_gallery(self, d: str) -> dict[str, list[np.ndarray]]:
        out = {}
        for sub in sorted(Path(d).iterdir()):
            if sub.is_dir():
                out[sub.name] = [self._hist(str(p)) for p in sorted(sub.glob("*")) if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        return out

    def exemplar_scores(self, crop_path):
        if not self.gallery:
            return None
        h = self._hist(crop_path)
        return {cow: max(float(np.minimum(h, g).sum()) for g in gs) for cow, gs in self.gallery.items() if gs}

    def exemplar_similarity(self, cow_id, crop_path):
        sc = self.exemplar_scores(crop_path)
        return None if not sc or cow_id not in sc else sc[cow_id]

    # ------------------------------------------------------------------ extras for verification / misses
    def cameras_with_video(self, day: int, minute: int) -> list[str]:
        return self.index.cameras_at(self.clock.to_dt(day, minute))

    def roster(self, day: int, minute: int) -> dict:
        out: dict[str, int] = {}
        for r in self.db.yolo_all_minute(day, minute):
            out[r.cow_id] = out.get(r.cow_id, 0) + 1
        return out

    @staticmethod
    def _iou(a, b) -> float:
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / u if u > 0 else 0.0

    def recover_box(self, cow_id, day, minute):
        """Look for a cow YOLO missed: run the detector, keep the detection that continues this cow's last/next known
        position and does not belong to another cow. A recovered box is stored (behavior NULL, probs_json.recovered) so the
        normal tools can crop it. Never invents an identity from appearance alone."""
        if self.detector is None:
            return {"ok": False, "reason": "no detector configured (set --detector-weights), cannot look for a missed cow"}
        from datetime import datetime
        tabs = (day - 1) * 1440 + minute
        known = [r.as_dict() for r in self.db.yolo_cow_range(cow_id, tabs - 5, tabs + 5) if r.behavior is not None and r.x1 is not None]
        if not known:
            return {"ok": False, "reason": "no known position of this cow within 5 minutes, nothing to continue from"}
        when = self.clock.to_dt(day, minute)
        for cam in self.cameras_with_video(day, minute):
            ref = [k for k in known if k["camera"] == cam]
            if not ref:
                continue
            ref.sort(key=lambda k: abs((datetime.fromisoformat(k["abs_time"]) - when).total_seconds()))
            exp = (ref[0]["x1"], ref[0]["y1"], ref[0]["x2"], ref[0]["y2"])
            dets = self.detect(cam, day, minute, 0.0) or []
            others = [(r.x1, r.y1, r.x2, r.y2) for r in self.db.yolo_all_minute(day, minute, exclude_cow=cow_id)
                      if r.camera == cam and r.x1 is not None]
            best = None
            for d in dets:
                bx = tuple(d["box"])
                if any(self._iou(bx, o) > 0.5 for o in others):
                    continue                                   # that box already belongs to another labelled cow
                w = max(exp[2] - exp[0], 1.0)
                dist = math.hypot((bx[0] + bx[2]) / 2 - (exp[0] + exp[2]) / 2, (bx[1] + bx[3]) / 2 - (exp[1] + exp[3]) / 2)
                if self._iou(bx, exp) > 0.25 or dist < 0.6 * w:
                    if best is None or dist < best[0]:
                        best = (dist, bx, d["conf"])
            if best:
                loc = self.index.locate(cam, when)
                clip, off = loc
                self.db.add_yolo_records([dict(file=Path(clip.path).name, camera=cam, t_offset=off,
                                               abs_time=when.isoformat(timespec="milliseconds"), day=day, minute=minute, cow_id=cow_id,
                                               behavior=None, conf=best[2], probs_json=json.dumps({"recovered": True}),
                                               x1=best[1][0], y1=best[1][1], x2=best[1][2], y2=best[1][3])])
                return {"ok": True, "camera": cam, "box": [round(v, 1) for v in best[1]], "detector_conf": best[2],
                        "reason": f"detector found a box continuing her last known position (distance {best[0]:.0f}px) that no other cow owns"}
        return {"ok": False, "reason": "detector found no box continuing her last known position that is not another cow's"}

    def detect(self, camera: str, day: int, minute: int, t_offset_s: float = 0.0) -> Optional[list[dict]]:
        """Run the optional detector on a frame (e.g. Ultralytics weights). None = no detector configured."""
        if self.detector is None:
            return None
        fr = self._frame(camera, self.clock.to_dt(day, minute, t_offset_s))
        if not fr:
            return []
        return [{"box": [round(v, 1) for v in d[:4]], "conf": round(d[4], 3)} for d in self.detector(fr[0])]


def ultralytics_detector(weights: str, conf: float = 0.25, cow_class: Optional[int] = None) -> Detector:
    """Wrap an Ultralytics YOLO model as a Detector. Import is lazy: the package is only needed if you use it."""
    from ultralytics import YOLO
    model = YOLO(weights)

    def run(img: Image.Image):
        res = model.predict(np.asarray(img.convert("RGB")), conf=conf, verbose=False)[0]
        out = []
        for b in res.boxes:
            if cow_class is None or int(b.cls) == cow_class:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                out.append((x1, y1, x2, y2, float(b.conf)))
        return out
    return run
