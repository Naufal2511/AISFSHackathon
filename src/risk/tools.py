"""Inspection tools for the BRD risk agent. Each returns (text, images, data) and logs every image it
shows the model as an `image` fact (camera, time, box size in source pixels, daylight) so later
claims can be checked against what was actually seen."""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

from ..api.flow import to_sec
from .data import BEHAVIORS, Herd, hms
from .indicators import coarse
from .score import fmt
from .store import EvidenceStore
from .longterm import LongTermTools, load_assets


@dataclass
class Result:
    text: str
    images: List[np.ndarray] = field(default_factory=list)
    data: Dict = field(default_factory=dict)


def is_daylight(img: np.ndarray) -> bool:
    """IR night video is grey: the colour channels are almost identical."""
    if img is None or img.ndim != 3:
        return False
    small = cv2.resize(img, (64, 48)).astype(np.int16)
    return float(np.abs(small[..., 2] - small[..., 1]).mean() + np.abs(small[..., 1] - small[..., 0]).mean()) > 6.0


def enhance(img: np.ndarray, size: int = 768) -> np.ndarray:
    """Upscale (Lanczos) + light denoise + local contrast (CLAHE) + unsharp mask. Makes ears, muzzle
    and discharge easier to see; it cannot add detail the camera never captured."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    if s > 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_LANCZOS4)
    img = cv2.fastNlMeansDenoisingColored(img, None, 4, 4, 7, 21)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(img, (0, 0), 2.0)
    return cv2.addWeighted(img, 1.5, blur, -0.5, 0)


def label(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
    scale = max(0.35, min(0.6, (img.shape[1] - 12) / (11.5 * max(len(text), 1))))  # fit narrow crops
    cv2.putText(img, text, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def hcat(ims: List[np.ndarray], h: int = 360) -> np.ndarray:
    ims = [cv2.resize(i, (int(i.shape[1] * h / i.shape[0]), h)) for i in ims if i is not None]
    gap = np.zeros((h, 6, 3), np.uint8)
    out = []
    for i in ims:
        out += [i, gap]
    return np.hstack(out[:-1]) if out else np.zeros((h, h, 3), np.uint8)


class RiskTools(LongTermTools):
    def __init__(self, herd: Herd, minutes: pd.DataFrame, ind: pd.DataFrame, con: pd.DataFrame,
                 scores: pd.DataFrame, cfg: Dict, store: EvidenceStore):
        self.h, self.m, self.ind, self.con, self.sc, self.cfg, self.store = herd, minutes, ind, con, scores, cfg, store
        self.obs = pd.concat([r.obs.assign(run=r.name) for r in herd.runs], ignore_index=True)
        self.obs = self.obs[self.obs.box_role.fillna("body") == "body"]
        self.obs["box_h"] = self.obs.y2 - self.obs.y1
        self.assets = load_assets([r.dir for r in herd.runs])
        self._R = Result
        self.tools = {n[2:]: getattr(self, n) for n in dir(self) if n.startswith("t_")}
        self._frame_cache: Dict = {}

    # ------------------------------------------------------------------ plumbing
    def describe(self) -> str:
        lines = []
        for name, fn in self.tools.items():
            sig = ", ".join(p for p in inspect.signature(fn).parameters)
            lines.append(f"- {name}({sig}): {(fn.__doc__ or '').strip().splitlines()[0]}")
        return "\n".join(lines)

    def call(self, name: str, args: Dict) -> Result:
        if name not in self.tools:
            return Result(f"unknown tool '{name}'. Tools: {', '.join(self.tools)}")
        try:
            return self.tools[name](**(args or {}))
        except TypeError as e:
            return Result(f"bad arguments for {name}: {e}")
        except Exception as e:  # a tool failure is information for the agent, not a crash
            return Result(f"{name} failed: {type(e).__name__}: {e}")

    def _frame(self, camera: str, sec: int) -> Optional[np.ndarray]:
        key = (camera, sec)
        if key not in self._frame_cache:
            run = self.h.run_at(sec)
            self._frame_cache[key] = None if run is None else run.frame(camera, sec)
            if len(self._frame_cache) > 64:
                self._frame_cache.pop(next(iter(self._frame_cache)))
        return self._frame_cache[key]

    def _box(self, cow_id: str, sec: int, camera: Optional[str] = None, tol: int = 2) -> Optional[pd.Series]:
        o = self.obs[(self.obs.cow_id == cow_id) & ((self.obs.sec - sec).abs() <= tol)]
        if camera:
            o = o[o.camera == camera]
        if o.empty:
            return None
        o = o.assign(dt=(o.sec - sec).abs()).sort_values(["dt", "box_h"], ascending=[True, False])
        return o.iloc[0]

    def _sec(self, time) -> int:
        """Accepts epoch seconds, ISO ('2026-11-07T16:07:25'), 'MM-DD HH:MM:SS' (as tools print it) or 'HH:MM:SS'."""
        import re
        if isinstance(time, (int, float, np.integer, np.floating)) or (isinstance(time, str) and time.strip().isdigit()):
            return int(float(time))
        s = str(time).strip()
        lo, hi = self.h.span()
        if re.fullmatch(r"\d{2}-\d{2}[ T]\d{1,2}:\d{2}(:\d{2})?", s):
            year = pd.to_datetime(lo, unit="s").year
            return int(to_sec(f"{year}-{s.replace(' ', 'T')}"))
        if re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", s):  # time of day: the matching day inside the data span
            days = pd.to_datetime(np.arange(lo, hi + 86400, 86400), unit="s").strftime("%Y-%m-%d").unique()
            cands = [int(to_sec(f"{d}T{s}")) for d in days]
            inside = [c for c in cands if lo - 60 <= c <= hi + 60]
            return (inside or cands)[0]
        return int(to_sec(s))

    def _log(self, img, cow_id, claim, data, source) -> str:
        return self.store.add_image(img, cow_id, claim, data, source)

    # ------------------------------------------------------------------ numbers
    def t_cow_profile(self, cow_id: str):
        """Behavior indicators of one cow vs the herd (value, herd median, z, hours observed, contribution) with fact ids."""
        d = self.con[self.con.cow_id == cow_id]
        if d.empty:
            return Result(f"no indicators for {cow_id}")
        s = self.sc.loc[cow_id] if cow_id in self.sc.index else None
        lines = [f"{cow_id}: behavior score {s.behavior_score:.2f} (rank {int(s['rank'])} of {len(self.sc)}"
                 + (f", top-3 in {100 * s.p_top3:.0f}% of time-resamples" if "p_top3" in s else "") + ")" if s is not None else cow_id,
                 "fact | indicator | value | herd median | z (sick-direction) | hours | contribution"]
        for _, r in d.sort_values("contrib", ascending=False).iterrows():
            base = r.indicator.replace("__trend", "")
            spec = self.cfg["indicators"][base]
            fid = f"F-{cow_id}-{r.indicator}"
            sz = "n/a" if pd.isna(r.signed) else f"{r.signed:+.1f}"
            lines.append(f"{fid} | {spec['label']}{' (day-to-day change)' if r.kind == 'trend' else ''} | "
                         f"{fmt(r.value, spec['unit'])} | {fmt(r.herd_median, spec['unit'])} | {sz} | "
                         f"{'' if pd.isna(r.hours) else f'{r.hours:.1f}'} | {r.contrib:+.2f}")
        cov = self.m[self.m.cow_id == cow_id]
        lines.append(f"coverage: seen {cov.seen_s.sum() / 3600:.1f} h, behavior labelled {cov.beh_obs_s.sum() / 3600:.1f} h "
                     f"between {hms(cov.minute.min())} and {hms(cov.minute.max())}")
        return Result("\n".join(lines))

    def t_herd_compare(self, indicator: str):
        """One indicator for every cow, sorted (to see whether the target is truly an outlier)."""
        if indicator not in self.ind:
            return Result(f"unknown indicator. Choose from: {', '.join(self.cfg['indicators'])}")
        spec = self.cfg["indicators"][indicator]
        v = self.ind[[indicator, f"{indicator}__hours"]].sort_values(indicator)
        return Result(f"{spec['label']} (sick direction: {'higher' if spec['direction'] > 0 else 'lower'})\n" +
                      "\n".join(f"{c}: {fmt(r[indicator], spec['unit'])} ({r[f'{indicator}__hours']:.1f} h)" for c, r in v.iterrows()))

    def t_timeline(self, cow_id: str):
        """Image: this cow's behavior per minute next to the herd's most common behavior, and its distance from the bunk."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        g = self.m[self.m.cow_id == cow_id].sort_values("minute")
        if g.empty:
            return Result(f"no minutes for {cow_id}")
        colors = {"lying": "#5b8def", "feeding": "#2ca02c", "upright": "#bbbbbb", "unknown": "#ffffff"}
        t = pd.to_datetime(g.minute, unit="s")
        fig, ax = plt.subplots(3, 1, figsize=(11, 4.2), sharex=True, gridspec_kw={"height_ratios": [1, 1, 2]})
        for i, col in enumerate(["coarse", "herd_mode"]):
            ax[i].bar(t, 1, width=1 / 1440, color=[colors.get(c, "#fff") for c in g[col].fillna("unknown")])
            ax[i].set_yticks([]); ax[i].set_ylabel(["this cow", "herd"][i], rotation=0, ha="right")
        ax[2].plot(t, g.y, ".", ms=2); ax[2].set_ylabel("m from bunk"); ax[2].invert_yaxis()
        ax[0].set_title(f"{cow_id}: green feeding, blue lying, grey upright, white unknown")
        fig.tight_layout()
        fig.canvas.draw()
        img = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[..., :3], cv2.COLOR_RGB2BGR)
        plt.close(fig)
        fid = self._log(img, cow_id, f"behavior timeline of {cow_id}", {"type": "chart"}, "timeline")
        out_of_step = int((g.desync_s > 0).sum())
        return Result(f"[{fid}] timeline {hms(g.minute.min())}-{hms(g.minute.max())}; minutes out of step with the herd: {out_of_step}", [img], {"fact": fid})

    def t_find_moments(self, cow_id: str, kind: str = "close_view", n: int = 4):
        """Times worth looking at. kind: close_view (largest box, daylight first), feeding, lying, isolated, out_of_step, walking, at_bunk."""
        o = self.obs[self.obs.cow_id == cow_id]
        if o.empty:
            return Result(f"{cow_id} has no boxes")
        g = self.m[self.m.cow_id == cow_id]
        if kind == "close_view":
            best = o.sort_values("box_h", ascending=False)
            hours = pd.to_datetime(best.sec, unit="s").dt.hour
            best = best.assign(day=hours.between(7, 18)).sort_values(["day", "box_h"], ascending=False)
            picks, used = [], []
            for _, r in best.iterrows():
                if all(abs(r.sec - u) > 120 for u in used):
                    picks.append(r); used.append(r.sec)
                if len(picks) >= n:
                    break
            return Result("\n".join(f"{hms(r.sec)} {r.camera} box {int(r.box_h)} px, facing {r.facing}, "
                                    f"{'day' if r.day else 'night'}" for r in picks),
                          data={"moments": [{"time": int(r.sec), "camera": r.camera} for r in picks]})
        col = {"feeding": "sec_feeding", "lying": "lying_s", "isolated": "iso_s", "out_of_step": "desync_s",
               "walking": "sec_walking", "at_bunk": "bunk_s"}.get(kind)
        if col is None:
            return Result("kind must be close_view, feeding, lying, isolated, out_of_step, walking or at_bunk")
        g = g[g[col] > 20].sort_values(col, ascending=False)
        picks, used = [], []
        for _, r in g.iterrows():
            if all(abs(r.minute - u) > 300 for u in used):
                picks.append(r); used.append(r.minute)
            if len(picks) >= n:
                break
        if not picks:
            return Result(f"no '{kind}' minutes found for {cow_id}")
        lines = []
        for r in picks:
            b = self._box(cow_id, int(r.minute) + 30, tol=30)
            cam = b.camera if b is not None else "?"
            lines.append(f"{hms(r.minute + 30)} ({kind} {int(r[col])} s that minute) best camera {cam}"
                         + (f", box {int(b.box_h)} px" if b is not None else ""))
        return Result("\n".join(lines), data={"moments": [int(r.minute) + 30 for r in picks]})

    def t_camera_info(self, cow_id: str, time):
        """Which cameras see the cow at that time: box size (source px), distance, facing, occlusion, day/night."""
        sec = self._sec(time)
        o = self.obs[(self.obs.cow_id == cow_id) & ((self.obs.sec - sec).abs() <= 1)]
        if o.empty:
            return Result(f"{cow_id} not detected at {hms(sec)} (try find_moments)")
        import yaml
        cpos = {k: v.get("mount_position_estimated") for k, v in yaml.safe_load(open("configs/cameras.yaml"))["cameras"].items()}
        lines = []
        for cam, g in o.groupby("camera"):
            r = g.sort_values("box_h", ascending=False).iloc[0]
            cp = cpos.get(cam)
            dist = np.hypot(r.world_x - cp[0], r.world_y - cp[1]) if cp and np.isfinite(r.world_x) else np.nan
            fr = self._frame(cam, int(r.sec))
            lines.append(f"{cam}: box {int(r.box_h)} px of {int(r.frame_h)}, {dist:.1f} m from camera, facing {r.facing}, "
                         f"{r.visibility}{' (occluded by ' + str(r.occluded_by) + ')' if isinstance(r.occluded_by, str) else ''}, "
                         f"{'daylight colour' if is_daylight(fr) else 'IR night / grey'}")
        mn = self.cfg["visual_min_box_px"]; dm = self.cfg["discharge_min_box_px"]
        lines.append(f"rule: posture signs need a box >= {mn} px; discharge needs >= {dm} px in daylight colour")
        return Result("\n".join(lines))

    def t_get_boxes(self, camera: str, time):
        """All detected cows in one camera at that second (id, box, size, facing, visibility)."""
        sec = self._sec(time)
        o = self.obs[(self.obs.camera == camera) & (self.obs.sec == sec)]
        if o.empty:
            return Result(f"no boxes in {camera} at {hms(sec)}")
        return Result("\n".join(f"{r.cow_id or '?'}: [{int(r.x1)},{int(r.y1)},{int(r.x2)},{int(r.y2)}] {int(r.box_h)} px, "
                                f"facing {r.facing}, {r.visibility}" for _, r in o.iterrows()))

    def t_get_frame(self, camera: str, time, boxes: bool = True):
        """Whole camera frame at that time with cow IDs drawn (context: where is the cow, what is the herd doing)."""
        sec = self._sec(time)
        fr = self._frame(camera, sec)
        if fr is None:
            return Result(f"no frame for {camera} at {hms(sec)}")
        img = fr.copy()
        if boxes:
            for _, r in self.obs[(self.obs.camera == camera) & (self.obs.sec == sec)].iterrows():
                cv2.rectangle(img, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), (0, 255, 255), 6)
                cv2.putText(img, str(r.cow_id), (int(r.x1), int(r.y1) - 12), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (0, 255, 255), 5)
        img = cv2.resize(img, (1280, int(img.shape[0] * 1280 / img.shape[1])))
        img = label(img, f"{camera} {hms(sec)} {'day' if is_daylight(fr) else 'IR night'}")
        fid = self._log(img, None, f"{camera} frame at {hms(sec)}", {"camera": camera, "time": sec, "daylight": is_daylight(fr)}, "get_frame")
        return Result(f"[{fid}] {camera} {hms(sec)}", [img], {"fact": fid})

    def _crop(self, cow_id: str, sec: int, camera: Optional[str], region: str, do_enhance: bool, size: int):
        b = self._box(cow_id, sec, camera)
        if b is None:
            return None, None, f"{cow_id} not detected at {hms(sec)}{' in ' + camera if camera else ''}"
        fr = self._frame(b.camera, int(b.sec))
        if fr is None:
            return None, None, f"no frame for {b.camera} at {hms(b.sec)}"
        x1, y1, x2, y2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)
        w, h = x2 - x1, y2 - y1
        reg = region
        if region == "head":  # head end from the facing estimate; side-on returns both ends
            reg = {"toward": "bottom_half", "away": "none"}.get(str(b.facing), "both_ends")
        if reg == "none":
            return None, b, f"{cow_id} faces away from {b.camera} at {hms(b.sec)}: head not visible; try another camera or time"
        boxes = {"body": [(x1, y1, x2, y2)], "top_half": [(x1, y1, x2, y1 + h // 2 + h // 8)],
                 "bottom_half": [(x1, y1 + h // 2 - h // 8, x2, y2)], "left_end": [(x1, y1, x1 + int(0.45 * w), y2)],
                 "right_end": [(x2 - int(0.45 * w), y1, x2, y2)]}
        # the head is at one end of the body's LONG axis (cameras look down: a cow is often tall in the image)
        boxes["both_ends"] = (boxes["left_end"] + boxes["right_end"]) if w >= h else \
            [(x1, y1, x2, y1 + int(0.5 * h)), (x1, y2 - int(0.5 * h), x2, y2)]
        if reg not in boxes:
            return None, b, "region must be body, head, top_half, bottom_half, left_end, right_end"
        H, W = fr.shape[:2]
        crops = []
        for (a, c, d, e) in boxes[reg]:
            pw, ph = int(0.08 * (d - a)), int(0.08 * (e - c))
            cr = fr[max(0, c - ph):min(H, e + ph), max(0, a - pw):min(W, d + pw)]
            crops.append(enhance(cr, size) if do_enhance else cr)
        img = hcat(crops, h=min(size, 720)) if len(crops) > 1 else crops[0]
        return img, b, None

    def t_get_crop(self, cow_id: str, time, camera: Optional[str] = None, region: str = "body", enhance: bool = True):
        """High-resolution crop of the cow from the 4000x3000 source frame, enhanced (upscale, denoise, contrast, sharpen). region: body | head | top_half | bottom_half | left_end | right_end."""
        sec = self._sec(time)
        img, b, err = self._crop(cow_id, sec, camera, region, enhance, 768)
        if err:
            return Result(err)
        fr = self._frame(b.camera, int(b.sec))
        day = is_daylight(fr)
        img = label(img, f"{cow_id} {b.camera} {hms(b.sec)} {region} box {int(b.box_h)}px {'day' if day else 'IR'}")
        fid = self._log(img, cow_id, f"{region} crop of {cow_id}", {"camera": b.camera, "time": int(b.sec), "region": region,
                                                                    "box_px": int(b.box_h), "daylight": day, "facing": b.facing,
                                                                    "enhanced": bool(enhance)}, "get_crop")
        return Result(f"[{fid}] {cow_id} {b.camera} {hms(b.sec)} {region}, box {int(b.box_h)} px, facing {b.facing}, "
                      f"{'daylight' if day else 'IR night'}", [img], {"fact": fid})

    def t_get_sequence(self, cow_id: str, time, span_s: int = 12, n: int = 6, camera: Optional[str] = None, region: str = "body"):
        """n crops over span_s seconds around a time (head carriage over time, flank movement / breathing effort, coughing posture)."""
        sec = self._sec(time)
        b0 = self._box(cow_id, sec, camera)
        if b0 is None:
            return Result(f"{cow_id} not detected at {hms(sec)}")
        ts = np.linspace(sec - span_s // 2, sec + span_s // 2, max(2, min(int(n), 8))).astype(int)
        crops, pxs = [], []
        for t in ts:
            img, b, err = self._crop(cow_id, int(t), b0.camera, region, True, 384)
            if img is not None:
                crops.append(label(img, f"{hms(b.sec)[-8:]}")); pxs.append(int(b.box_h))
        if not crops:
            return Result("no crops in that span")
        rows = [hcat(crops[i:i + 4], 300) for i in range(0, len(crops), 4)]
        wmax = max(r.shape[1] for r in rows)
        img = np.vstack([cv2.copyMakeBorder(r, 0, 6, 0, wmax - r.shape[1], cv2.BORDER_CONSTANT) for r in rows])
        day = is_daylight(self._frame(b0.camera, int(b0.sec)))
        fid = self._log(img, cow_id, f"{len(crops)}-frame sequence of {cow_id}",
                        {"camera": b0.camera, "time": sec, "span_s": span_s, "region": region, "box_px": int(np.median(pxs)),
                         "daylight": day}, "get_sequence")
        return Result(f"[{fid}] {len(crops)} crops of {cow_id} from {b0.camera} over {span_s} s, median box {int(np.median(pxs))} px", [img], {"fact": fid})

    def t_compare_herdmates(self, cow_id: str, time, camera: Optional[str] = None):
        """The cow next to up to 3 herdmates in the same camera at the same moment (relative head carriage, ears, body condition)."""
        sec = self._sec(time)
        b = self._box(cow_id, sec, camera)
        if b is None:
            return Result(f"{cow_id} not detected at {hms(sec)}")
        others = self.obs[(self.obs.camera == b.camera) & (self.obs.sec == b.sec) & (self.obs.cow_id != cow_id) & self.obs.cow_id.notna()]
        others = others.sort_values("box_h", ascending=False).head(3)
        crops = []
        for cid in [cow_id] + others.cow_id.tolist():
            img, bb, _ = self._crop(cid, int(b.sec), b.camera, "body", True, 384)
            if img is not None:
                crops.append(label(img, f"{cid}{' (target)' if cid == cow_id else ''} {int(bb.box_h)}px"))
        img = hcat(crops, 320)
        fid = self._log(img, cow_id, f"{cow_id} vs herdmates", {"camera": b.camera, "time": int(b.sec),
                                                                 "box_px": int(b.box_h), "daylight": is_daylight(self._frame(b.camera, int(b.sec)))},
                        "compare_herdmates")
        return Result(f"[{fid}] {cow_id} vs {', '.join(others.cow_id)} in {b.camera} at {hms(b.sec)}", [img], {"fact": fid})

    def t_record_fact(self, cow_id: str, claim: str, images: List[str], sign: Optional[str] = None,
                      status: Optional[str] = None, confidence: float = 0.5):
        """Save what you observed (with the image fact ids it is based on). sign/status optional: sign in visual_signs, status present|absent|not_assessable."""
        if isinstance(images, str):
            images = [images]
        bad = [i for i in images if not self.store.get(i) or self.store.get(i)["kind"] != "image"]
        if bad or not images:
            return Result(f"record_fact needs image fact ids you were shown (unknown: {bad or 'none given'})")
        fid = self.store.add("visual", cow_id, claim, {"sign": sign, "status": status, "confidence": float(confidence),
                                                       "based_on": images}, source="agent")
        return Result(f"saved {fid}", data={"fact": fid})
