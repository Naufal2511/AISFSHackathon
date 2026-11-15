"""Post-pass: turn the raw assets of a tracking run into per-cow risk assets.

  <level1 run>/assets/
    gallery.csv        close daylight crops with cow_id (path into the raw run's assets_raw/gallery)
    motion.csv         per cow per camera per second motion energy (+ box size)
    breathing.csv      breaths per minute from still, close windows (decoded at native fps) + spectrum images
    jerks.csv          sudden movement while not walking (cough / head-shake candidates)
    feed_events.csv    bunk-trough changes (cam2/cam1/cam3 trough polygons) = feed delivery candidates
    clips/             a short low-res clip per cow every N hours (closest view), clips.csv
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

from ..agent.evidence import FrameStore
from ..behavior.features import load_records

EPOCH = pd.Timestamp("1970-01-01")


def _key(df, cam="camera", tid="track_id", sv="source_video", off="offset_s"):
    return df[cam].astype(str) + "|" + df[tid].astype(int).astype(str) + "|" + df[sv].astype(str) + "|" + \
        (df[off].astype(float)).round(1).astype(str)


def join_ids(raw: pd.DataFrame, resolved: pd.DataFrame) -> pd.DataFrame:
    """Attach global_cow_id (body boxes only) to raw asset rows by camera, track, clip and clip time."""
    r = resolved.copy()
    if "box_role" in r:
        r = r[r.box_role.fillna("body") == "body"]
    r = r.assign(k=_key(r, "camera_id", "local_track_id", "source_video", "timestamp"))
    look = r.drop_duplicates("k").set_index("k").global_cow_id
    out = raw.assign(cow_id=_key(raw).map(look))
    return out[out.cow_id.notna()]


# ---------------------------------------------------------------- breathing
def breathing_rate(frames: List[np.ndarray], fps: float, lo_bpm=15, hi_bpm=90):
    """Dominant periodic intensity change over the still body (flank rise and fall).
    Returns (bpm, quality = peak / median power in band, spectrum, freqs, drift)."""
    g = np.stack([cv2.GaussianBlur(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (96, 96),
                                              interpolation=cv2.INTER_AREA).astype(np.float32), (5, 5), 0)
                  for f in frames])
    drift = float(np.abs(g[-1] - g[0]).mean())          # big = the cow moved: not a breathing window
    x = g.reshape(len(g), -1)
    # high-pass: remove anything slower than ~4 s per cycle (lighting drift, slow shifts of weight)
    from scipy.ndimage import uniform_filter1d
    x = x - uniform_filter1d(x, size=max(3, int(round(fps * 4))), axis=0, mode="nearest")
    x = x[int(fps * 2): len(x) - int(fps * 2)]          # drop filter edge effects
    var = x.var(0)
    sel = x[:, var >= np.quantile(var, 0.9)]
    win = np.hanning(len(sel))[:, None]
    spec = (np.abs(np.fft.rfft(sel * win, axis=0)) ** 2).mean(1)
    freqs = np.fft.rfftfreq(len(sel), 1 / fps) * 60
    band = (freqs >= lo_bpm) & (freqs <= hi_bpm)
    if band.sum() < 5:
        return np.nan, 0.0, spec, freqs, drift
    idx = np.where(band)[0]
    peaks = [i for i in idx[1:-1] if spec[i] >= spec[i - 1] and spec[i] >= spec[i + 1]]   # interior local maxima only
    if not peaks:
        return np.nan, 0.0, spec, freqs, drift
    i = max(peaks, key=lambda k: spec[k])
    q = float(spec[i] / max(np.median(spec[band]), 1e-9))
    # effort: amplitude of the breathing component (power in the peak +-1 bin) relative to body brightness
    amp = np.sqrt(spec[max(i - 1, 0):i + 2].sum() * 2) / max(np.hanning(len(sel)).sum(), 1)
    breathing_rate.effort = float(100 * amp / max(float(g.mean()), 1.0))
    return float(freqs[i]), q, spec, freqs, drift


def read_window(fs: FrameStore, source_video: str, offset: float, seconds: float, every: int = 3, box=None, pad=0.05):
    path = fs._path(source_video)
    if path is None:
        return [], 0.0
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, offset) * 1000)
    out, n = [], int(seconds * fps)
    for i in range(n):
        if i % every:
            if not cap.grab():
                break
            continue
        ok, fr = cap.read()
        if not ok:
            break
        if box is not None:
            x1, y1, x2, y2 = box
            px, py = pad * (x2 - x1), pad * (y2 - y1)
            H, W = fr.shape[:2]
            fr = fr[int(max(0, y1 - py)):int(min(H, y2 + py)), int(max(0, x1 - px)):int(min(W, x2 + px))]
        out.append(fr)
    cap.release()
    return out, fps / every


def still_windows(mot: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """Runs of >= window_s consecutive seconds of one cow in one camera, close and still."""
    m = mot[(mot.box_h >= cfg["breath_min_px"])].sort_values(["cow_id", "camera", "wall_t"])
    out = []
    for (cow, cam), g in m.groupby(["cow_id", "camera"]):
        thr = min(cfg["still_motion"], g.motion.quantile(0.3))
        still = (g.motion <= thr).to_numpy()
        t = g.wall_t.to_numpy()
        start = None
        for i in range(len(g)):
            ok = still[i] and (start is None or t[i] - t[i - 1] <= 1.5)
            if ok and start is None:
                start = i
            if (not ok or i == len(g) - 1) and start is not None:
                end = i if ok else i - 1
                if t[end] - t[start] >= cfg["breath_window_s"]:
                    r = g.iloc[start]
                    seg = g.iloc[start:end + 1]
                    out.append({"cow_id": cow, "camera": cam, "t0": float(t[start]), "t1": float(t[end]),
                                "source_video": r.source_video, "offset_s": float(r.offset_s),
                                "x1": seg.x1.median(), "y1": seg.y1.median(), "x2": seg.x2.median(), "y2": seg.y2.median(),
                                "box_h": float(seg.box_h.median()), "motion": float(seg.motion.median())})
                start = None if not ok else i
    return pd.DataFrame(out)


def breathing(mot: pd.DataFrame, fs: FrameStore, out_dir: str, cfg: Dict) -> pd.DataFrame:
    w = still_windows(mot, cfg)
    if w.empty:
        return w
    # at most N windows per cow per block, the closest first; spread over the day
    w["block"] = (w.t0 // (3600 * cfg["breath_block_h"])).astype(int)
    w = w.sort_values("box_h", ascending=False).groupby(["cow_id", "block"]).head(cfg["breath_per_block"])
    os.makedirs(os.path.join(out_dir, "breathing"), exist_ok=True)
    rows = []
    for _, r in w.iterrows():
        same_clip = r.source_video  # window must stay inside one clip
        frames, fps = read_window(fs, same_clip, r.offset_s, cfg["breath_window_s"], cfg["breath_every"],
                                  (r.x1, r.y1, r.x2, r.y2))
        if len(frames) < 20:
            continue
        breathing_rate.effort = np.nan
        bpm, q, spec, freqs, drift = breathing_rate(frames, fps)
        effort = breathing_rate.effort
        img = spectrum_png(frames[len(frames) // 2], spec, freqs, bpm, q)
        rel = os.path.join("breathing", f"{r.cow_id}_{int(r.t0)}.jpg")
        cv2.imwrite(os.path.join(out_dir, rel), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        rows.append({**{k: r[k] for k in ["cow_id", "camera", "t0", "t1", "box_h", "motion"]},
                     "bpm": round(bpm, 1) if np.isfinite(bpm) else np.nan, "quality": round(q, 2), "drift": round(drift, 2),
                     "effort": round(effort, 3) if np.isfinite(effort) else np.nan,
                     "fps": round(fps, 1), "n_frames": len(frames), "daylight": int(_day(frames[0])),
                     "valid": bool(np.isfinite(bpm) and q >= cfg["breath_min_quality"] and drift <= cfg["breath_max_drift"]),
                     "image": rel})
    return pd.DataFrame(rows)


def spectrum_png(frame, spec, freqs, bpm, q):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(8, 3), gridspec_kw={"width_ratios": [1, 1.4]})
    ax[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)); ax[0].axis("off")
    band = (freqs >= 6) & (freqs <= 120)
    ax[1].plot(freqs[band], spec[band]); ax[1].axvline(bpm, color="r", ls="--")
    ax[1].set_xlabel("cycles per minute"); ax[1].set_yticks([])
    ax[1].set_title(f"dominant {bpm:.0f}/min, peak/median {q:.1f}")
    fig.tight_layout(); fig.canvas.draw()
    img = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[..., :3], cv2.COLOR_RGB2BGR)
    plt.close(fig)
    return img


def _day(img):
    from .recorder import is_daylight
    return is_daylight(img)


# ---------------------------------------------------------------- jerks, feed events
def jerks(mot: pd.DataFrame, state: Optional[pd.DataFrame], cfg: Dict) -> pd.DataFrame:
    rows = []
    for (cow, cam), g in mot.groupby(["cow_id", "camera"]):
        g = g.dropna(subset=["motion"])
        if len(g) < 60:
            continue
        med = g.motion.median()
        mad = 1.4826 * (g.motion - med).abs().median() + 1e-6
        hit = g[(g.motion > med + cfg["jerk_mad"] * mad) & (g.motion > cfg["jerk_min"])]
        for _, r in hit.iterrows():
            rows.append({"cow_id": cow, "camera": cam, "wall_t": r.wall_t, "motion": r.motion, "z": (r.motion - med) / mad,
                         "box_h": r.box_h})
    j = pd.DataFrame(rows)
    if j.empty or state is None:
        return j
    st = state.set_index(["cow_id", "sec"])
    j["speed"] = [st.speed_mps.get((c, int(t)), np.nan) for c, t in zip(j.cow_id, j.wall_t)]
    j["y"] = [st.y.get((c, int(t)), np.nan) for c, t in zip(j.cow_id, j.wall_t)]
    j = j[~(j.speed > cfg["jerk_max_speed"])]          # walking / running is not a jerk
    j = j[~(j.y <= cfg.get("jerk_exclude_bunk_m", 4.0))]  # head movement while eating at the bunk is not a cough
    j = j.sort_values("wall_t")
    keep, last = [], {}
    for _, r in j.iterrows():                           # one event per cow per 10 s
        if r.wall_t - last.get(r.cow_id, -1e9) >= 10:
            keep.append(r); last[r.cow_id] = r.wall_t
    return pd.DataFrame(keep)


def feed_events(frames: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    rows = []
    for cam, g in frames.dropna(subset=["bunk_mean"]).groupby("camera"):
        g = g.sort_values("wall_t")
        s = g.set_index("wall_t").bunk_mean
        if len(s) < 300:
            continue
        t = s.index.to_numpy()
        v = s.to_numpy()
        W = cfg["feed_window_s"]
        deltas = []
        for i in range(0, len(t), 15):
            b = v[(t >= t[i] - W - 60) & (t < t[i] - 60)]
            a = v[(t > t[i] + 60) & (t <= t[i] + W + 60)]
            if len(b) > W / 3 and len(a) > W / 3:
                deltas.append((t[i], np.median(a) - np.median(b)))
        if not deltas:
            continue
        d = pd.DataFrame(deltas, columns=["wall_t", "delta"])
        mad = 1.4826 * (d.delta - d.delta.median()).abs().median() + 1e-6
        d["z"] = (d.delta - d.delta.median()) / mad
        cand = d[d.z.abs() >= cfg["feed_z"]].sort_values("z", key=np.abs, ascending=False)
        used = []
        for _, r in cand.iterrows():
            if all(abs(r.wall_t - u) > 1800 for u in used):
                used.append(r.wall_t)
                rows.append({"camera": cam, "wall_t": r.wall_t, "delta": r.delta, "z": r.z})
    return pd.DataFrame(rows)


def clips(mot: pd.DataFrame, fs: FrameStore, out_dir: str, cfg: Dict) -> pd.DataFrame:
    from ..risk.report import _write_clip
    os.makedirs(os.path.join(out_dir, "clips"), exist_ok=True)
    m = mot.assign(block=(mot.wall_t // (3600 * cfg["clip_every_h"])).astype(int))
    best = m.sort_values("box_h", ascending=False).groupby(["cow_id", "block"]).head(1)
    rows = []
    for _, r in best.iterrows():
        rel = os.path.join("clips", f"{r.cow_id}_{int(r.wall_t)}_{r.camera}.mp4")
        sub = mot[(mot.cow_id == r.cow_id) & (mot.camera == r.camera) & mot.wall_t.between(r.wall_t - 5, r.wall_t + 5)]
        n = _write_clip(fs, r.camera, sub.rename(columns={"wall_t": "sec"}), os.path.join(out_dir, rel), r.cow_id, fps_out=6, width=640)
        if n:
            rows.append({"cow_id": r.cow_id, "camera": r.camera, "t0": r.wall_t - 5, "t1": r.wall_t + 5, "box_h": r.box_h, "file": rel})
    return pd.DataFrame(rows)
