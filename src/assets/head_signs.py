"""Head signs from gallery crops: a VLM used as a sampled "pose sensor" + an image check for discharge.

We have no cattle keypoint model, so a fast TAMU model labels a few close daylight crops per cow per
hour (batched, several crops per call): standing / feeding, head carriage relative to the withers, ears,
back, and where the muzzle is. A deterministic image check then looks inside the muzzle box for
discharge-like pixels (light, low-saturation mucus or yellow-green pus streaks against the black Angus
muzzle). Both are screening signals; the risk agent still has to look at the crop.

Output: <run>/assets/head_signs.csv
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

PROMPT = """You label close camera crops of Angus feedlot cattle (black coat, often seen from above).
For EACH numbered image return one object. Be conservative: use "unclear"/"not_visible" when unsure.
Fields:
 "i": image number
 "standing": true | false | "unclear"
 "feeding": true if the head is in or over the feed bunk / eating, else false
 "head_position": "below_withers" | "level" | "above_withers" | "not_visible"   (muzzle relative to the top of the shoulders)
 "ears": "droop" | "normal" | "not_visible"     (droop = hanging down/backwards, not erect)
 "back": "arched" | "normal" | "unclear"
 "muzzle_box": [x1, y1, x2, y2] as fractions 0-1 of the image, or null if the nose is not visible
 "discharge": "yes" | "no" | "unclear"    (visible nasal or eye discharge; feed crumbs on the muzzle are NOT discharge)
 "note": at most 12 words
Reply with JSON only: {"labels": [ {...}, ... ]}"""

FIELDS = ["standing", "feeding", "head_position", "ears", "back", "muzzle_box", "discharge", "note"]


def discharge_score(img: np.ndarray, box) -> Dict:
    """Light / yellow-green elongated blobs in the muzzle box and just below it (nostril streaks).
    Returns pixel share and the longest vertical streak relative to the box height."""
    if img is None or box is None:
        return {"ip_share": np.nan, "ip_streak": np.nan, "muzzle_px": 0}
    H, W = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        return {"ip_share": np.nan, "ip_streak": np.nan, "muzzle_px": 0}
    bh = y2 - y1
    X1, X2 = int(x1 * W), int(x2 * W)
    Y1, Y2 = int(y1 * H), int(min(1.0, y2 + 0.5 * bh) * H)       # include the area below the nostrils
    roi = img[Y1:Y2, X1:X2]
    if roi.size == 0 or min(roi.shape[:2]) < 12:
        return {"ip_share": np.nan, "ip_streak": np.nan, "muzzle_px": int(min(roi.shape[:2]) if roi.size else 0)}
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    whitish = (v > 0.55) & (s < 0.30)
    yellow_green = (h >= 15) & (h <= 45) & (s > 0.35) & (v > 0.35)
    mask = (whitish | yellow_green).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    streak = 0.0
    for k in range(1, n):
        x, y, w, hh, area = stats[k]
        if hh >= 1.5 * w and area >= 0.002 * mask.size:            # elongated downward = a run, not a speck
            streak = max(streak, hh / mask.shape[0])
    return {"ip_share": float(mask.mean()), "ip_streak": float(streak), "muzzle_px": int(min(roi.shape[:2]))}


def pick(gallery: pd.DataFrame, per_hour: int = 2, min_px: int = 450) -> pd.DataFrame:
    g = gallery[gallery.box_h >= min_px].copy()
    g["hour"] = (g.wall_t // 3600 * 3600).astype(int)
    out = []
    for _, grp in g.sort_values("box_h", ascending=False).groupby(["cow_id", "hour"]):
        used = []
        for r in grp.itertuples():
            if all(abs(r.wall_t - u) >= 600 for u in used):
                out.append(r._asdict()); used.append(r.wall_t)
            if len(used) >= per_hour:
                break
    return pd.DataFrame(out)


def label_batch(backend, paths: List[str], max_side: int = 768) -> List[Dict]:
    from ..agent.evidence import to_jpeg_b64
    from ..agent.vlm import parse_json
    content = [{"type": "text", "text": PROMPT + f"\nThere are {len(paths)} images."}]
    for i, p in enumerate(paths, 1):
        content += [{"type": "text", "text": f"Image {i}:"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + to_jpeg_b64(cv2.imread(p), max_side)}}]
    text = backend.complete([{"role": "user", "content": content}])
    d = parse_json(text) or {}
    labels = d.get("labels") if isinstance(d, dict) else None
    return labels if isinstance(labels, list) else []


def run(gallery: pd.DataFrame, backend, per_hour: int = 2, batch: int = 4, min_px: int = 450,
        model_label: str = "") -> pd.DataFrame:
    sel = pick(gallery, per_hour, min_px)
    rows = []
    for i in range(0, len(sel), batch):
        chunk = sel.iloc[i:i + batch]
        paths = chunk.path.tolist()
        t0 = time.time()
        try:
            labels = label_batch(backend, paths)
        except Exception as e:
            print(f"[head_signs] batch {i // batch} failed: {e}")
            labels = []
        by_i = {int(l.get("i", k + 1)): l for k, l in enumerate(labels) if isinstance(l, dict)}
        for k, r in enumerate(chunk.itertuples(), 1):
            lab = by_i.get(k, {})
            img = cv2.imread(r.path)
            ds = discharge_score(img, lab.get("muzzle_box"))
            rows.append({"cow_id": r.cow_id, "wall_t": r.wall_t, "camera": r.camera, "box_h": r.box_h, "path": r.path,
                         **{f: (json.dumps(lab.get(f)) if f == "muzzle_box" else lab.get(f, "unclear")) for f in FIELDS},
                         **ds, "model": model_label, "seconds": round(time.time() - t0, 1)})
        print(f"[head_signs] {min(i + batch, len(sel))}/{len(sel)} crops", flush=True)
    df = pd.DataFrame(rows)
    if len(df):
        for f in ("head_position", "ears", "back", "discharge"):
            df[f] = df[f].astype(str).str.lower()
        # image check can raise a VLM "unclear" to a suspect, never overrule a clear "no"
        df["discharge_flag"] = (df.discharge == "yes") | ((df.discharge == "unclear") & (df.ip_streak >= 0.35))
    return df
