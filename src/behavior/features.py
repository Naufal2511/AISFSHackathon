"""Per-observation behavior cues computed from tracking_records.csv.

Every cue is derived from geometry the Level-1 pipeline already produces (bbox track,
calibrated ground position, pen zones), so no extra video decoding is needed:

  p_lying      standing-height model (bbox height vs expected standing height at that image row)
  speed_px_mps image speed of the ground contact, scaled by the expected standing height
               (orientation/posture invariant, robust to far-field homography error)
  speed_mps    calibrated ground speed (m/s)
  in_feed      on the feeding apron; cam2 adds "head over trough" overlap
  drink_dist   distance (m) to the waterer polygon
  width_cv     short-window oscillation of bbox width while stationary (self-grooming cue)
  social       nearest-neighbour contact from world distance + same-camera bbox overlap
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import yaml

EVENT_RE = re.compile(r"Event(\d{14})")


# ----------------------------------------------------------------------------- loading
def parse_clip_start(name: Optional[str]) -> Optional[pd.Timestamp]:
    """'Event20261107071945006.Avi' (or remux_Event...mp4) -> 2026-11-07 07:19:45."""
    if not isinstance(name, str):
        return None
    m = EVENT_RE.search(name)
    if not m:
        return None
    return pd.Timestamp(datetime.strptime(m.group(1), "%Y%m%d%H%M%S"))


def load_records(
    csv_paths,
    cameras_cfg: str = "configs/cameras.yaml",
    clip_start: Optional[str] = None,
) -> pd.DataFrame:
    """Loads one or more tracking_records CSVs and anchors every row to wall-clock time."""
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]
    frames = []
    for p in csv_paths:
        d = pd.read_csv(p)
        d["_source_csv"] = p
        frames.append(d)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        raise ValueError(
            f"No tracking rows in {csv_paths}. Run main.py first (tracking_records.csv is header-only)."
        )

    with open(cameras_cfg) as f:
        cams = yaml.safe_load(f)["cameras"]

    # Wall clock: prefer the Event timestamp in source_video, then --clip-start, else epoch 0.
    fallback = pd.Timestamp(clip_start) if clip_start else pd.Timestamp("1970-01-01")
    if "source_video" in df.columns:
        starts = df["source_video"].map(parse_clip_start)
        starts = starts.where(starts.notna(), fallback)
    else:
        starts = pd.Series([fallback] * len(df), index=df.index)
    offsets = df["camera_id"].map(lambda c: float(cams.get(c, {}).get("time_offset_seconds", 0.0)))
    df["wall_time"] = pd.to_datetime(starts) + pd.to_timedelta(df["timestamp"] + offsets, unit="s")
    df["t"] = (df["wall_time"] - pd.Timestamp("1970-01-01")).dt.total_seconds()
    df["sec"] = np.floor(df["t"]).astype(np.int64)

    # Frame geometry (older CSVs lack it -> camera reference resolution)
    for col, k in (("frame_width", 0), ("frame_height", 1)):
        ref = df["camera_id"].map(lambda c: cams.get(c, {}).get("reference_resolution", [3024, 1640])[k])
        df[col] = df[col].fillna(ref) if col in df.columns else ref

    df["occlusion"] = pd.to_numeric(df.get("occlusion_status"), errors="coerce").fillna(0.0).clip(0, 1)
    df["identity_confidence"] = pd.to_numeric(df.get("identity_confidence"), errors="coerce").fillna(0.6)
    df["detection_confidence"] = pd.to_numeric(df["detection_confidence"], errors="coerce").fillna(0.5)
    for c in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "world_x", "world_y"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["track_key"] = (
        df["camera_id"].astype(str) + "|" + df.get("source_video", df["_source_csv"]).astype(str)
        + "|" + df["local_track_id"].astype(str)
    )
    df["quality"] = (df["detection_confidence"] * (1 - 0.7 * df["occlusion"]) * df["identity_confidence"]).clip(0.02, 1)
    return df


# ----------------------------------------------------------------------------- helpers
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _load_pen(pen_cfg: str) -> Dict:
    with open(pen_cfg) as f:
        return yaml.safe_load(f)


def fit_standing_height(df_cam: pd.DataFrame, q: float, n_bins: int, min_rows: int):
    """Expected standing bbox height as a linear function of the bbox bottom row.

    On a ground plane the image height of an object of fixed physical height grows
    ~linearly with the image row of its ground contact. Standing animals dominate the
    upper quantile of heights in every row band, so the fit needs no labels.
    """
    d = df_cam.assign(h=df_cam["bbox_y2"] - df_cam["bbox_y1"])
    d = d[np.isfinite(d["bbox_y2"]) & np.isfinite(d["h"]) & (d["h"] > 2)]
    if len(d) < min_rows:
        return None
    edges = np.unique(np.quantile(d["bbox_y2"], np.linspace(0, 1, n_bins + 1)))
    d = d.assign(bin=np.clip(np.searchsorted(edges, d["bbox_y2"], side="right") - 1, 0, len(edges) - 2))
    # One value per (track, band): a long lying bout must not outvote standing animals.
    # Each track contributes its own tall end (it stood at some point if it ever did).
    per_track = d.groupby(["bin", "track_key"]).agg(y=("bbox_y2", "median"), h=("h", lambda s: s.quantile(0.9)))
    xs, ys, ws = [], [], []
    for _, g in per_track.groupby(level=0):
        if len(g) >= 2:
            xs.append(g["y"].median())
            ys.append(g["h"].quantile(q))
            ws.append(np.sqrt(len(g)))
    if len(xs) < 3:
        return None
    slope, intercept = np.polyfit(xs, ys, 1, w=ws)
    if slope <= 0:  # degenerate view: fall back to a constant standing height
        slope, intercept = 0.0, float(np.quantile(d["h"], q))
    return float(slope), float(intercept)


def _grid_fraction_in_poly(x1, y1, x2, y2, poly: np.ndarray, n: int = 5) -> float:
    xs = np.linspace(x1, x2, n)
    ys = np.linspace(y1 + 0.4 * (y2 - y1), y2, n)  # lower 60% of the box (head + neck when eating)
    hit = 0
    for x in xs:
        for y in ys:
            hit += cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0
    return hit / float(n * n)


# ----------------------------------------------------------------------------- cue extraction
def compute_cues(df: pd.DataFrame, cfg: Dict, pen_cfg: str = "configs/pen_layout.yaml") -> pd.DataFrame:
    df = df.copy()
    pen = _load_pen(pen_cfg)
    df["bw"] = (df["bbox_x2"] - df["bbox_x1"]).clip(lower=1)
    df["bh"] = (df["bbox_y2"] - df["bbox_y1"]).clip(lower=1)
    df["gx"] = (df["bbox_x1"] + df["bbox_x2"]) / 2.0  # ground-contact point
    df["gy"] = df["bbox_y2"]

    # ---- posture
    pc = cfg["posture"]
    df["height_ratio"] = np.nan
    df["p_lying"] = np.nan
    df["exp_h"] = np.nan  # expected standing height in px at this image row (metric ruler)
    fits = {}
    for cam, d in df.groupby("camera_id"):
        if cam in pc["overhead_cameras"]:
            df.loc[d.index, "p_lying"] = pc["overhead_lying_prior"]
            df.loc[d.index, "exp_h"] = np.nan
            continue
        fit = fit_standing_height(d, pc["standing_quantile"], pc["n_bins"], pc["min_rows_for_fit"])
        if fit is None:  # too few rows: aspect-ratio fallback (lying steer is long and flat)
            ar = d["bh"] / d["bw"]
            df.loc[d.index, "height_ratio"] = ar / max(np.quantile(ar, pc["standing_quantile"]), 1e-3)
            df.loc[d.index, "exp_h"] = d["bh"].rolling(30, min_periods=1).max()
        else:
            fits[cam] = fit
            exp_h = (fit[0] * d["bbox_y2"] + fit[1]).clip(lower=5)
            df.loc[d.index, "height_ratio"] = d["bh"] / exp_h
            df.loc[d.index, "exp_h"] = exp_h
        r = df.loc[d.index, "height_ratio"]
        df.loc[d.index, "p_lying"] = _sigmoid((pc["lying_ratio"] - r) / pc["lying_softness"])
    df.attrs["standing_height_fits"] = fits

    # ---- motion (per local track, time ordered)
    mc = cfg["motion"]
    df = df.sort_values(["track_key", "t"])
    g = df.groupby("track_key", sort=False)
    dt = g["t"].diff()
    # Ruler: expected standing height (~standing_height_m) at this image row. It does not
    # depend on body orientation or on the animal lying down, unlike bbox width/height.
    ruler = df["exp_h"].fillna(df[["bw", "bh"]].max(axis=1))
    dpx = np.hypot(g["gx"].diff(), g["gy"].diff())
    df["speed_px_mps_raw"] = (dpx / ruler * mc["standing_height_m"] / dt).where(dt > 0)
    dw = np.hypot(g["world_x"].diff(), g["world_y"].diff())
    df["speed_mps_raw"] = (dw / dt).where(dt > 0)
    fps = 1.0 / g["t"].diff().median() if len(df) > 1 else 1.0
    win = max(1, int(round(mc["smooth_window_sec"] * (fps if np.isfinite(fps) and fps > 0 else 1.0))))
    for c in ("speed_px_mps", "speed_mps"):
        df[c] = df.groupby("track_key", sort=False)[c + "_raw"].transform(
            lambda s: s.rolling(win, min_periods=1, center=True).median()
        )
    p_move_px = _sigmoid((df["speed_px_mps"] - mc["walk_speed_mps"]) / mc["walk_softness_mps"])
    p_move_w = _sigmoid((df["speed_mps"] - mc["world_walk_speed_mps"]) / mc["walk_softness_mps"])
    # World speed may boost the image cue (motion along the viewing axis is foreshortened
    # on screen) but cannot create motion alone: far-field homography jitter is ~1 m/px-row.
    blended = np.maximum(p_move_px, np.minimum(p_move_w, 3 * p_move_px))
    df["p_moving"] = np.where(df["camera_id"].isin(pc["overhead_cameras"]), p_move_w, blended)
    df["p_moving"] = pd.Series(df["p_moving"], index=df.index).fillna(0.1)

    # ---- grooming cue: width oscillation with a still ground contact
    gc = cfg["grooming"]
    gwin = max(3, int(round(gc["window_sec"] * (fps if np.isfinite(fps) and fps > 0 else 1.0))))
    roll = df.groupby("track_key", sort=False)["bw"]
    df["width_cv"] = roll.transform(lambda s: s.rolling(gwin, min_periods=3, center=True).std()) / roll.transform(
        lambda s: s.rolling(gwin, min_periods=3, center=True).mean()
    )
    df["p_groom_cue"] = (
        _sigmoid((df["width_cv"].fillna(0) - gc["width_cv_threshold"]) / 0.03) * (1 - df["p_moving"]) * gc["max_prob"]
    )

    # ---- zones
    zc = cfg["zones"]
    wx, wy = df["world_x"], df["world_y"]
    in_pen = wx.between(-0.5, 12.5) & wy.between(-1.5, 30.5)
    df["in_feed"] = (in_pen & (wy <= zc["bunk_y_max_m"])).astype(float)
    trough_pts = pen.get("camera_rois", {}).get("cam2", {}).get("trough_polygon")
    trough = np.array(trough_pts, dtype=np.float32) if trough_pts else None
    is_cam2 = df["camera_id"] == "cam2"
    if trough is not None and is_cam2.any():
        sub = df[is_cam2]
        frac = [
            _grid_fraction_in_poly(
                r.bbox_x1, r.bbox_y1, r.bbox_x2, r.bbox_y2,
                trough * np.array([r.frame_width, r.frame_height], dtype=np.float32),
            )
            for r in sub.itertuples()
        ]
        df.loc[sub.index, "head_in_trough"] = frac
        # cam2 only sees the apron, so presence there is itself strong feeding-zone evidence
        df.loc[sub.index, "in_feed"] = np.clip(0.6 + zc["head_in_trough_bonus"] * 4 * np.array(frac), 0, 1)
    df["head_in_trough"] = df.get("head_in_trough", pd.Series(np.nan, index=df.index))

    water = np.array(pen["landmarks"]["waterer"]["polygon_metric"], dtype=np.float32)
    dd = np.full(len(df), np.inf)
    ok = wx.notna() & wy.notna()
    dd[ok.to_numpy()] = [
        max(0.0, -cv2.pointPolygonTest(water, (float(x), float(y)), True))
        for x, y in zip(wx[ok], wy[ok])
    ]
    df["drink_dist"] = dd
    df["in_drink"] = (df["drink_dist"] <= zc["drink_radius_m"]).astype(float)

    df = _apply_drawn_zones(df, pen_cfg, zc)
    df = add_social_cue(df, cfg)
    return df.sort_values(["global_cow_id", "t"]).reset_index(drop=True)


def _apply_drawn_zones(df: pd.DataFrame, pen_cfg: str, zc: Dict) -> pd.DataFrame:
    """Hand-drawn image polygons (configs/camera_zones.yaml) override the metric zones.

    feeding  : >= feed_bbox_frac of the cow's box lies in the feed-bunk band (head/neck in the bunk)
    drinking : >= drink_bbox_frac of the box lies in the waterer polygon (head at the water)
    Rows decided this way get full zone weight in fusion, whatever the camera's metric trust.
    """
    from ..calibration.camera_zones import bbox_fraction_in, load_zones
    drawn = load_zones(os.path.join(os.path.dirname(pen_cfg) or ".", "camera_zones.yaml"))
    df["zone_src_feed"] = "metric"
    df["zone_src_drink"] = "metric"
    for cam, layers in drawn.items():
        m = df["camera_id"] == cam
        if not m.any():
            continue
        sub = df[m]
        for layer, col, src, thr in (("feed_bunk", "in_feed", "zone_src_feed", zc.get("feed_bbox_frac", 0.12)),
                                     ("waterer", "in_drink", "zone_src_drink", zc.get("drink_bbox_frac", 0.05))):
            if layer not in layers:
                continue
            frac = np.array([bbox_fraction_in(layers[layer], (r.bbox_x1, r.bbox_y1, r.bbox_x2, r.bbox_y2),
                                              r.frame_width, r.frame_height) for r in sub.itertuples()])
            df.loc[sub.index, col] = np.clip(frac / thr, 0, 1)
            df.loc[sub.index, src] = "drawn"
    return df


def add_social_cue(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """Nearest-neighbour contact per cow-second from world distance and same-camera overlap."""
    sc = cfg["social"]
    df["nn_dist"] = np.nan
    df["p_social_cue"] = 0.0
    known = df["global_cow_id"].notna()
    # world distance on per-(cow, sec) mean position (fused across cameras)
    pos = df[known].groupby(["sec", "global_cow_id"])[["world_x", "world_y"]].mean().dropna()
    nn = {}
    for sec, d in pos.groupby(level=0):
        if len(d) < 2:
            continue
        P = d.to_numpy()
        D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
        np.fill_diagonal(D, np.inf)
        for cow, v in zip(d.index.get_level_values(1), D.min(1)):
            nn[(sec, cow)] = v
    if nn:
        keys = list(zip(df["sec"], df["global_cow_id"]))
        df["nn_dist"] = [nn.get(k, np.nan) for k in keys]
    p_world = _sigmoid((sc["contact_dist_m"] - df["nn_dist"].fillna(99)) / 0.25)

    # same-camera bbox overlap at comparable depth (robust where the far-field homography is coarse)
    iou_max = np.zeros(len(df))
    idx_of = {ix: i for i, ix in enumerate(df.index)}
    for _, d in df[known].groupby(["camera_id", "sec"]):
        if len(d) < 2:
            continue
        b = d[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy()
        ix1 = np.maximum(b[:, None, 0], b[None, :, 0]); iy1 = np.maximum(b[:, None, 1], b[None, :, 1])
        ix2 = np.minimum(b[:, None, 2], b[None, :, 2]); iy2 = np.minimum(b[:, None, 3], b[None, :, 3])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        iou = inter / (area[:, None] + area[None, :] - inter + 1e-6)
        h = b[:, 3] - b[:, 1]
        same_depth = np.abs(b[:, None, 3] - b[None, :, 3]) < 0.25 * np.maximum(h[:, None], h[None, :])
        iou = np.where(same_depth, iou, 0.0)
        np.fill_diagonal(iou, 0.0)
        for ix, v in zip(d.index, iou.max(1)):
            iou_max[idx_of[ix]] = v
    df["bbox_contact_iou"] = iou_max
    p_iou = _sigmoid((iou_max - sc["bbox_iou_contact"]) / 0.05)

    p = np.maximum(p_world, 0.8 * p_iou)
    # Proximity alone is not interaction (steers idle shoulder-to-shoulder for minutes).
    # Require activity: this animal moved in the last few seconds, or the gap is closing.
    win = int(sc.get("activity_window_sec", 5))
    active = df.groupby("track_key", sort=False)["p_moving"].transform(
        lambda s: s.rolling(win, min_periods=1, center=True).max())
    per_cow = df[known].groupby(["global_cow_id", "sec"])["nn_dist"].min().rename("nn").reset_index()
    per_cow["closing"] = -per_cow.groupby("global_cow_id")["nn"].diff()  # m/s the gap shrinks
    closing = df[["global_cow_id", "sec"]].merge(per_cow, on=["global_cow_id", "sec"], how="left")["closing"]
    approach = _sigmoid((closing.fillna(0).to_numpy() - sc.get("approach_speed_mps", 0.25)) / 0.08)
    p = p * np.maximum(active.to_numpy(), approach)
    p = p * (1 - df["p_lying"].fillna(0)) * (1 - df["in_feed"])
    df["p_social_cue"] = np.clip(p, 0, 1) * sc["max_prob"]
    return df
