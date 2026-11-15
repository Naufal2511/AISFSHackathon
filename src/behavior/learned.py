"""Fine-tuned per-box behavior classifier (beh-dinov2s-ft-v1) as the emission model.

Replaces the hand-built factorised cue model (fusion.behavior_probabilities) for every
cow-second that has a scored crop. Everything after the emission (HMM bout smoothing, gap
bridging, rumination chew probes, minute labels, agent review) is unchanged.

Weights: models/behavior/beh_finetuned.pt (git-ignored, pulled from Git LFS). The checkpoint
holds a timm ViT-S/14 DINOv2 state_dict with a 10-class head over taxonomy.BEHAVIORS and a
calibration temperature T.

Preprocess (as trained): RGB crop of the padded box -> 224x224 -> ImageNet mean/std; IR frames
are already 3-channel grey. Logits are averaged over a short window (2-5 s) before softmax / T.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from .taxonomy import BEHAVIORS

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
LEARN_COLS = [f"learn::{b}" for b in BEHAVIORS]
# data-contract column names (docs/DATA_CONTRACT.md), same order as BEHAVIORS
P_COLS = ["p_lying", "p_standing", "p_walking", "p_feeding", "p_drinking", "p_rum_lying", "p_rum_standing",
          "p_grooming", "p_social", "p_other"]


def _device(pref: Optional[str] = None) -> str:
    import torch
    if pref and pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class FineTunedBehaviorModel:
    def __init__(self, weights: str, device: Optional[str] = None):
        import timm
        import torch
        if not os.path.exists(weights) or os.path.getsize(weights) < 1_000_000:
            raise FileNotFoundError(f"{weights} missing or a Git LFS pointer; run `git lfs pull` on the weights branch")
        ck = torch.load(weights, map_location="cpu", weights_only=False)
        if list(ck["classes"]) != BEHAVIORS:
            raise ValueError(f"checkpoint classes {ck['classes']} != taxonomy {BEHAVIORS}")
        self.version = ck.get("version", os.path.basename(weights))
        self.T = float(ck.get("T", 1.0))
        self.img_size = int(ck.get("img_size", 224))
        self.device = _device(device)
        self.model = timm.create_model(ck["backbone"].split(".")[0], pretrained=False,
                                       num_classes=len(BEHAVIORS), img_size=self.img_size)
        self.model.load_state_dict(ck["state_dict"], strict=True)
        self.model.eval().to(self.device)
        self._torch = torch

    def crop(self, frame_bgr: np.ndarray, box: Tuple[float, float, float, float], pad: float) -> Optional[np.ndarray]:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = box
        pw, ph = pad * (x2 - x1), pad * (y2 - y1)
        x1, y1 = int(max(0, x1 - pw)), int(max(0, y1 - ph))
        x2, y2 = int(min(w, x2 + pw)), int(min(h, y2 + ph))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        c = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        c = cv2.resize(c, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        return ((c.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)

    def logits(self, batch: List[np.ndarray]) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            x = torch.from_numpy(np.stack(batch)).to(self.device)
            return self.model(x).float().cpu().numpy()

    def probs(self, logits: np.ndarray) -> np.ndarray:
        z = logits / self.T
        z = z - z.max(-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(-1, keepdims=True)


class _Seeker:
    """Reads frames at increasing clip offsets from one video, grabbing forward instead of
    re-seeking for short hops (HEVC seeks decode from the previous keyframe anyway)."""

    def __init__(self, path: str):
        from .rumination import _open
        self.cap = _open(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 15.0
        self.pos = -1e9  # seconds of the next frame to be read

    def at(self, t: float) -> Optional[np.ndarray]:
        if t < self.pos - 0.5 / self.fps or t - self.pos > 3.0:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
            self.pos = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        while self.pos < t - 0.5 / self.fps:
            if not self.cap.grab():
                return None
            self.pos = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        ok, fr = self.cap.read()
        self.pos = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        return fr if ok else None

    def close(self):
        self.cap.release()


def score_boxes(model: FineTunedBehaviorModel, rows: pd.DataFrame, video_index: Dict[str, str],
                cfg: Dict) -> pd.DataFrame:
    """Per-box logits for every body box at sampled seconds. Returns one row per scored box."""
    from .rumination import resolve_video
    every = int(cfg.get("every_sec", 2))
    pad = float(cfg.get("pad_frac", 0.1))
    bs = int(cfg.get("batch_size", 32))
    sub = rows[rows["global_cow_id"].notna() & (rows["sec"] % every == 0)]
    if "box_role" in sub:
        sub = sub[sub["box_role"].fillna("body") == "body"]
    out_idx, out_logits = [], []
    buf, buf_idx = [], []

    def flush():
        if buf:
            out_logits.append(model.logits(buf))
            out_idx.extend(buf_idx)
            buf.clear(); buf_idx.clear()

    t0, n_frames, missing = time.time(), 0, set()
    for vid, d in sub.groupby("source_video"):
        vp = resolve_video(vid, video_index)
        if vp is None:
            missing.add(vid)
            continue
        sk = _Seeker(vp)
        for ts, g in d.groupby("timestamp", sort=True):
            fr = sk.at(float(ts))
            if fr is None:
                continue
            n_frames += 1
            for x in g.itertuples():
                c = model.crop(fr, (x.bbox_x1, x.bbox_y1, x.bbox_x2, x.bbox_y2), pad)
                if c is not None:
                    buf.append(c); buf_idx.append(x.Index)
                    if len(buf) >= bs:
                        flush()
        sk.close()
    flush()
    if missing:
        print(f"[BEHAVIOR] learned model: {len(missing)} source videos not found (their boxes keep the heuristic)")
    if not out_idx:
        return pd.DataFrame()
    L = np.concatenate(out_logits)
    keep = [c for c in ("global_cow_id", "camera_id", "sec", "t", "timestamp", "local_track_id", "track_piece",
                        "quality", "source_video") if c in rows.columns]
    b = rows.loc[out_idx, keep].reset_index(drop=True)
    b[[f"logit::{c}" for c in BEHAVIORS]] = L
    print(f"[BEHAVIOR] learned model {model.version}: {len(b)} boxes from {n_frames} frames "
          f"in {time.time() - t0:.0f} s on {model.device}")
    return b


def fuse_scores(boxes: pd.DataFrame, model: FineTunedBehaviorModel, cfg: Dict) -> pd.DataFrame:
    """Boxes -> one calibrated 10-class vector per (cow, sampled second).

    Logits are averaged over all cameras of that cow and over +-window/2 seconds, weighted by
    detection quality x per-camera trust, then softmax(logit / T)."""
    lc = [f"logit::{c}" for c in BEHAVIORS]
    cw = cfg.get("camera_weight", {}) or {}
    w = boxes["quality"].fillna(0.3).to_numpy() * boxes["camera_id"].map(lambda c: float(cw.get(c, 1.0))).to_numpy()
    half = float(cfg.get("window_sec", 4)) / 2.0
    out = []
    for cow, d in boxes.assign(_w=w).groupby("global_cow_id"):
        secs = d["sec"].to_numpy(); L = d[lc].to_numpy(); wd = d["_w"].to_numpy()
        for s in np.unique(secs):
            m = np.abs(secs - s) <= half
            ww = wd[m] + 1e-6
            lg = (L[m] * ww[:, None]).sum(0) / ww.sum()
            out.append((cow, int(s), *model.probs(lg)))
    return pd.DataFrame(out, columns=["global_cow_id", "sec"] + LEARN_COLS)


def attach(fused: pd.DataFrame, learned: pd.DataFrame, hold_sec: float) -> pd.DataFrame:
    """Gives every observed cow-second the nearest learned vector within hold_sec (seconds
    between samples); seconds further away keep NaN and fall back to the heuristic."""
    fused = fused.drop(columns=[c for c in LEARN_COLS if c in fused.columns])
    if learned is None or learned.empty:
        return fused
    parts = []
    lk = learned.sort_values("sec")
    for cow, d in fused.groupby("global_cow_id", sort=False):
        lc = lk[lk["global_cow_id"] == cow].drop(columns="global_cow_id")
        d = d.sort_values("sec")
        if lc.empty:
            parts.append(d)
            continue
        parts.append(pd.merge_asof(d, lc, on="sec", direction="nearest", tolerance=int(hold_sec)))
    return pd.concat(parts, ignore_index=True)


def contract_predictions(boxes: pd.DataFrame, model: FineTunedBehaviorModel) -> pd.DataFrame:
    """Per-box probabilities in the DATA_CONTRACT shape accepted by RunFlow.submit_behavior."""
    P = model.probs(boxes[[f"logit::{c}" for c in BEHAVIORS]].to_numpy())
    o = pd.DataFrame({
        "wall_time": pd.to_datetime(np.round(boxes["t"]).astype(np.int64), unit="s").dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "camera": boxes["camera_id"], "track_id": boxes["local_track_id"].astype(int),
        "track_piece": boxes.get("track_piece", pd.Series(0, index=boxes.index)).fillna(0).astype(int),
        "cow_id": boxes["global_cow_id"],
    })
    o[P_COLS] = P.round(4)
    o["behavior"] = [BEHAVIORS[k] for k in P.argmax(1)]
    o["beh_conf"] = P.max(1).round(3)
    o["model"] = model.version
    return o
