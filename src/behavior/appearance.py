"""Optional learned appearance expert: frozen DINOv2 crop embedding + linear probe.

Why this design (fast + robust for a 48-h hackathon):
  * DINOv2 ViT-S/14 is already loaded by the ReID stage -> no new heavy dependency.
  * A logistic-regression probe trains in seconds on CPU and works with a few hundred
    labels; external datasets add the classes geometry cannot see (rumination, grooming).
  * Its output is fused as a product-of-experts with the geometric model, never alone.

Training sources (all optional, combine any):
  1. Event crops you label: `export-crops` writes crops pre-sorted into folders named by the
     current geometric prediction; you only drag the wrong ones to the right folder.
  2. CVB (CSIRO, 8 Angus, 11 classes incl. ruminating-lying/standing, grooming), AVA csv.
  3. CBVD-5 (Kaggle, CC0, standing/lying/feeding/drinking/rumination), AVA-style csv.
"""

from __future__ import annotations

import glob
import os
import random
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

from .taxonomy import BEHAVIORS, CBVD5_ID_TO_NAME, CBVD5_TO_OURS, CVB_TO_OURS, LYING_FAMILY

SAFE = {b: b.replace(" / ", "_or_").replace(" ", "_").replace("(", "").replace(")", "") for b in BEHAVIORS}
UNSAFE = {v: k for k, v in SAFE.items()}


# ----------------------------------------------------------------------------- manifests
def manifest_from_folders(root: str, source: str = "event") -> pd.DataFrame:
    """root/<safe_class_name>/*.jpg  ->  manifest(image, x1..y2 = full image, label, source)."""
    rows = []
    for safe, label in UNSAFE.items():
        for p in glob.glob(os.path.join(root, safe, "*.jpg")) + glob.glob(os.path.join(root, safe, "*.png")):
            rows.append({"image": p, "x1": 0, "y1": 0, "x2": 1, "y2": 1, "label": label, "source": source})
    return pd.DataFrame(rows)


def manifest_from_ava(
    ava_csv: str,
    frames_root: str,
    frame_pattern: str,
    fps: float,
    dataset: str,
    id_to_name: Optional[Dict[int, str]] = None,
) -> pd.DataFrame:
    """AVA rows: video_id, timestamp, x1, y1, x2, y2 (normalised), action_id, person_id.

    frame_pattern example (CVB raw_frames): "{video_id}/{frame:05d}.jpg" with fps=30.
    Labels are mapped onto the 10 hackathon classes; unmapped labels are dropped.
    """
    df = pd.read_csv(ava_csv, header=None)
    df = df.iloc[:, :8]
    df.columns = ["video_id", "timestamp", "x1", "y1", "x2", "y2", "action_id", "person_id"]
    if id_to_name is None:
        id_to_name = CBVD5_ID_TO_NAME if dataset == "cbvd5" else {}
    table = CVB_TO_OURS if dataset == "cvb" else CBVD5_TO_OURS
    rows = []
    for r in df.itertuples():
        name = id_to_name.get(int(r.action_id), str(r.action_id)).lower()
        ours = table.get(name)
        if ours is None:
            continue
        if ours == "ruminating":  # CBVD-5 does not split rumination by posture
            ours = "ruminating (lying)" if (r.y2 - r.y1) < 0.6 * (r.x2 - r.x1) else "ruminating (standing)"
        img = os.path.join(frames_root, frame_pattern.format(video_id=r.video_id, frame=int(round(r.timestamp * fps))))
        rows.append({"image": img, "x1": r.x1, "y1": r.y1, "x2": r.x2, "y2": r.y2, "label": ours, "source": dataset})
    out = pd.DataFrame(rows)
    missing = (~out["image"].map(os.path.exists)).mean() if len(out) else 1.0
    if missing > 0.5:
        raise FileNotFoundError(
            f"{missing:.0%} of frames not found under {frames_root} with pattern '{frame_pattern}'. "
            "Check the dataset's frame naming and pass --frame-pattern accordingly."
        )
    return out[out["image"].map(os.path.exists)]


# ----------------------------------------------------------------------------- embedding
class Embedder:
    def __init__(self):
        from ..tracking.reidentification import CowAppearanceExtractor
        self.ex = CowAppearanceExtractor(model_name="dinov2_vits14")

    def crop_embed(self, img_bgr: np.ndarray, x1, y1, x2, y2, normalised: bool) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        if normalised:
            x1, x2, y1, y2 = x1 * w, x2 * w, y1 * h, y2 * h
        # 10% context pad so the head / water trough stays in view
        pw, ph = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
        box = (max(0, x1 - pw), max(0, y1 - ph), min(w, x2 + pw), min(h, y2 + ph))
        return self.ex.extract(img_bgr, box)


def embed_manifest(man: pd.DataFrame, emb: Embedder) -> np.ndarray:
    X = []
    for r in man.itertuples():
        img = cv2.imread(r.image)
        X.append(emb.crop_embed(img, r.x1, r.y1, r.x2, r.y2, normalised=True))
    return np.vstack(X)


def train_probe(man: pd.DataFrame, out_path: str, eval_source: str = "event") -> Dict:
    """Trains the probe; reports accuracy on held-out EVENT crops (the domain that matters)."""
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split

    emb = Embedder()
    X = embed_manifest(man, emb)
    y = man["label"].to_numpy()
    is_eval_dom = (man["source"] == eval_source).to_numpy()
    report = {}
    if is_eval_dom.sum() >= 30 and len(set(y[is_eval_dom])) > 1:
        idx = np.where(is_eval_dom)[0]
        tr_i, te_i = train_test_split(idx, test_size=0.3, random_state=0, stratify=y[idx] if min(pd.Series(y[idx]).value_counts()) > 1 else None)
        train_mask = np.ones(len(y), bool); train_mask[te_i] = False
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5).fit(X[train_mask], y[train_mask])
        report = classification_report(y[te_i], clf.predict(X[te_i]), output_dict=True, zero_division=0)
        print(classification_report(y[te_i], clf.predict(X[te_i]), zero_division=0))
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5).fit(X, y)
    joblib.dump({"clf": clf, "classes": list(clf.classes_)}, out_path)
    return report


# ----------------------------------------------------------------------------- event crops
def _read_frame(path: str, clip_sec: float) -> Optional[np.ndarray]:
    from .rumination import _open
    cap = _open(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, clip_sec * 1000.0)
    ok, fr = cap.read()
    cap.release()
    return fr if ok else None


def export_crops(rows: pd.DataFrame, timeline: pd.DataFrame, video_index: Dict[str, str], out_dir: str,
                 per_class: int = 40, seed: int = 0) -> int:
    """Writes crops pre-sorted by the current prediction for quick human correction."""
    from .rumination import resolve_video
    random.seed(seed)
    lab = timeline[["global_cow_id", "sec", "behavior"]]
    r = rows.merge(lab, on=["global_cow_id", "sec"], how="inner")
    n = 0
    for b, d in r.groupby("behavior"):
        d = d.sample(min(per_class, len(d)), random_state=seed)
        os.makedirs(os.path.join(out_dir, SAFE[b]), exist_ok=True)
        for x in d.itertuples():
            vp = resolve_video(getattr(x, "source_video", None), video_index)
            if vp is None:
                continue
            fr = _read_frame(vp, float(x.timestamp))
            if fr is None:
                continue
            crop = fr[int(max(0, x.bbox_y1)):int(x.bbox_y2), int(max(0, x.bbox_x1)):int(x.bbox_x2)]
            if crop.size:
                name = f"{x.camera_id}_{x.global_cow_id}_{x.sec}.jpg"
                cv2.imwrite(os.path.join(out_dir, SAFE[b], name), crop)
                n += 1
    return n


def predict_rows(model_path: str, rows: pd.DataFrame, video_index: Dict[str, str], every_sec: int = 10) -> pd.DataFrame:
    """Appearance probabilities for a subsample of (cow, camera, second) rows."""
    import joblib
    from .rumination import resolve_video
    m = joblib.load(model_path)
    emb = Embedder()
    sub = rows[(rows["sec"] % every_sec == 0) & rows["global_cow_id"].notna()]
    out = []
    for (vid, ts), d in sub.groupby(["source_video", "timestamp"]):
        vp = resolve_video(vid, video_index)
        fr = _read_frame(vp, float(ts)) if vp else None
        if fr is None:
            continue
        for x in d.itertuples():
            e = emb.crop_embed(fr, x.bbox_x1, x.bbox_y1, x.bbox_x2, x.bbox_y2, normalised=False)
            p = m["clf"].predict_proba(e[None])[0]
            rec = {"global_cow_id": x.global_cow_id, "sec": x.sec}
            rec.update({f"app::{c}": v for c, v in zip(m["classes"], p)})
            out.append(rec)
    return pd.DataFrame(out).groupby(["global_cow_id", "sec"]).mean().reset_index() if out else pd.DataFrame()
