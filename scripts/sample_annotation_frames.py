#!/usr/bin/env python3
"""Pick ~200 frames worth hand-labelling and pre-label them, ready for Roboflow / CVAT / Label Studio.

  python scripts/sample_annotation_frames.py --n 200 --out data/annotation

Selection
  * Stratified by camera (cam1 40% / cam2 25% / cam3 35%) and day vs night (IR).
  * Frames are spread over many clips; a clip goes entirely to train OR val (no leakage).
  * TRAIN frames: of K candidate times per clip, keep the HARDEST one, i.e. where the
    fine-tuned model and the COCO model disagree most / give many low-confidence boxes
    (bunk rear views, edge-cut cows, clusters, night). That is where labels pay off.
  * VAL frames: a random candidate, so the validation score stays representative.

Pre-labels (YOLO txt, class 0 = cow): union of both models at low confidence, NMS-merged.
You correct boxes instead of drawing them. Label EVERY cow you see (any pen); the pen ROI
decides membership downstream.

Output
  data/annotation/images/{train,val}/*.jpg
  data/annotation/labels/{train,val}/*.txt   (pre-labels; overwrite with your corrections)
  data/annotation/dataset.yaml, classes.txt, manifest.csv
"""
import argparse
import glob
import hashlib
import os
import random
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import pandas as pd

EVENT_RE = re.compile(r"Event(\d{14})")
CAM_RE = re.compile(r"Cam\s*0?(\d)", re.I)


def list_clips(root):
    rows = []
    for p in glob.glob(os.path.join(root, "**", "*.[Aa]vi"), recursive=True):
        m, c = EVENT_RE.search(os.path.basename(p)), CAM_RE.search(p)
        if not (m and c):
            continue
        ts = pd.to_datetime(m.group(1), format="%Y%m%d%H%M%S")
        hour = ts.hour + ts.minute / 60
        rows.append(dict(path=p, camera=f"cam{c.group(1)}", start=ts, clip=m.group(0),
                         night=not (7.0 <= hour < 17.5)))  # College Station, November daylight
    return pd.DataFrame(rows).sort_values(["camera", "start"]).reset_index(drop=True)


def grab(path, t, out_jpg):
    """Single-frame decode with ffmpeg (input seek; no full remux)."""
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{t:.2f}", "-vcodec", "hevc", "-i", path,
           "-frames:v", "1", "-q:v", "2", out_jpg]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    if r.returncode != 0 or not os.path.exists(out_jpg):  # container may not need the forced codec
        cmd = ["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{t:.2f}", "-i", path, "-frames:v", "1", "-q:v", "2", out_jpg]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    return cv2.imread(out_jpg) if os.path.exists(out_jpg) else None


def detect(model, img, classes, conf, imgsz):
    r = model(img, conf=conf, classes=classes, imgsz=imgsz, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 5), np.float32)
    return np.hstack([r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()[:, None]]).astype(np.float32)


def nms(boxes, iou=0.6):
    if len(boxes) == 0:
        return boxes
    keep = cv2.dnn.NMSBoxes([[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes],
                            [float(b[4]) for b in boxes], 0.0, iou)
    return boxes[np.array(keep).reshape(-1)] if len(keep) else boxes[:0]


def hardness(a, b, w, h):
    low = ((a[:, 4] < 0.4).sum() if len(a) else 0) + ((b[:, 4] < 0.4).sum() if len(b) else 0)
    allb = np.vstack([a, b]) if len(a) + len(b) else np.zeros((0, 5))
    edge = ((allb[:, 0] < 0.01 * w) | (allb[:, 2] > 0.99 * w) | (allb[:, 3] > 0.99 * h)).sum() if len(allb) else 0
    return abs(len(a) - len(b)) + 0.5 * low + 0.3 * edge


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="input")
    ap.add_argument("--out", default="data/annotation")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--camera-share", default="cam1=0.4,cam2=0.25,cam3=0.35")
    ap.add_argument("--candidates", type=int, default=3)
    ap.add_argument("--finetuned", default="runs/segment/runs/angus_train/angus_yolo26m-seg-4/weights/best.pt")
    ap.add_argument("--coco", default="yolo26x-seg.pt")
    ap.add_argument("--prelabel-conf", type=float, default=0.10)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--daylight-only", action="store_true", help="sample only daytime clips")
    ap.add_argument("--append", action="store_true",
                    help="add to an existing --out set (keeps existing frames, appends to manifest.csv)")
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed)

    from ultralytics import YOLO
    m_ft, m_coco = YOLO(a.finetuned), YOLO(a.coco)
    coco_cow = [k for k, v in m_coco.names.items() if v == "cow"]

    clips = list_clips(a.input)
    if a.daylight_only:
        clips = clips[~clips.night]
    if clips.empty:
        sys.exit(f"no Event*.Avi clips under {a.input}")
    share = {k: float(v) for k, v in (kv.split("=") for kv in a.camera_share.split(","))}
    share = {k: v for k, v in share.items() if k in set(clips.camera)}
    tot = sum(share.values())

    plan = []
    for cam, s in share.items():
        n_cam = int(round(a.n * s / tot))
        cc = clips[clips.camera == cam]
        strata = [g for _, g in cc.groupby("night")]
        per = [n_cam // len(strata) + (1 if i < n_cam % len(strata) else 0) for i in range(len(strata))]
        for g, k in zip(strata, per):
            idx = np.linspace(0, len(g) - 1, min(k, len(g))).round().astype(int)  # spread over time
            chosen = g.iloc[np.unique(idx)]
            extra = k - len(chosen)  # fewer clips than frames: reuse clips at other times
            if extra > 0:
                chosen = pd.concat([chosen, g.sample(extra, replace=True, random_state=a.seed)])
            plan.extend(chosen.to_dict("records"))

    # clip-level split, stratified: deterministic hash so re-runs keep the same split
    def is_val(clip):
        return int(hashlib.md5(clip.encode()).hexdigest(), 16) % 1000 < a.val_frac * 1000

    for sp in ("train", "val"):
        os.makedirs(os.path.join(a.out, "images", sp), exist_ok=True)
        os.makedirs(os.path.join(a.out, "labels", sp), exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="annot_")
    manifest = []
    for i, c in enumerate(plan):
        split = "val" if is_val(c["clip"]) else "train"
        best = None
        for j in range(a.candidates if split == "train" else 1):
            t = random.uniform(5, 290)
            img = grab(c["path"], t, os.path.join(tmp, f"c{i}_{j}.jpg"))
            if img is None:
                continue
            h, w = img.shape[:2]
            b1 = detect(m_ft, img, None, 0.05, a.imgsz)
            b2 = detect(m_coco, img, coco_cow, 0.05, a.imgsz)
            score = hardness(b1, b2, w, h)
            if best is None or score > best[0]:
                best = (score, t, img, b1, b2)
        if best is None:
            print(f"[skip] could not decode {c['path']}")
            continue
        score, t, img, b1, b2 = best
        h, w = img.shape[:2]
        boxes = nms(np.vstack([b1, b2]) if len(b1) + len(b2) else np.zeros((0, 5), np.float32))
        boxes = boxes[boxes[:, 4] >= a.prelabel_conf] if len(boxes) else boxes
        name = f"{c['camera']}_{c['clip']}_t{int(t):03d}"
        cv2.imwrite(os.path.join(a.out, "images", split, name + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(os.path.join(a.out, "labels", split, name + ".txt"), "w") as f:
            for x1, y1, x2, y2, _ in boxes:
                f.write(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")
        manifest.append(dict(image=f"images/{split}/{name}.jpg", split=split, camera=c["camera"], clip=c["clip"],
                             clip_start=str(c["start"]), t_sec=round(t, 1), night_by_clock=c["night"],
                             ir_frame=bool(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1].mean() < 12),
                             n_prelabels=len(boxes), n_finetuned=len(b1), n_coco=len(b2), hardness=round(score, 2)))
        print(f"[{i + 1}/{len(plan)}] {split:5s} {name}  prelabels={len(boxes)}  hardness={score:.1f}")

    mpath = os.path.join(a.out, "manifest.csv")
    new = pd.DataFrame(manifest)
    if a.append and os.path.exists(mpath):
        new = pd.concat([pd.read_csv(mpath), new], ignore_index=True).drop_duplicates("image", keep="last")
    new.to_csv(mpath, index=False)
    with open(os.path.join(a.out, "classes.txt"), "w") as f:
        f.write("cow\n")
    with open(os.path.join(a.out, "dataset.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(a.out)}\ntrain: images/train\nval: images/val\nnames:\n  0: cow\n")
    df = pd.DataFrame(manifest)
    if len(df):
        print(df.groupby(["split", "camera", "ir_frame"]).size().to_string())
    print(f"\nDone: {len(df)} frames in {a.out}. Correct the boxes, then train (see docs/ANNOTATION_GUIDE.md).")


if __name__ == "__main__":
    main()
