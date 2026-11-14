#!/usr/bin/env python3
"""Compare cow detectors on the annotation frames (never used for training).

  python scripts/compare_detectors.py \
      --model ft640=runs/segment/runs/angus_train/angus_yolo26m-seg-4/weights/best.pt@640 \
      --model ft1280=runs/segment/runs/angus_train/angus_yolo26m-seg-4/weights/best.pt@1280 \
      --model coco1280=yolo26x-seg.pt@1280 --split val

Reference boxes:
  * hand labels for frames listed in data/annotation/reviewed.txt (after Label Studio export)
    -> real precision / recall / F1 at IoU 0.5
  * otherwise a POOLED proxy: union of all compared models at conf >= 0.4 (NMS-merged).
    Proxy recall = share of boxes that at least one strong detector found. It is NOT
    ground truth; use it only to rank models until hand labels exist.
Results are broken down by camera and day/night (manifest.csv).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import pandas as pd


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match(pred, ref, thr=0.5):
    """Greedy one-to-one matching by IoU; returns (true positives, n_pred, n_ref)."""
    M = iou_matrix(pred, ref)
    tp = 0
    while M.size and M.max() >= thr:
        i, j = np.unravel_index(M.argmax(), M.shape)
        tp += 1
        M[i, :] = -1
        M[:, j] = -1
    return tp, len(pred), len(ref)


def nms(b, iou=0.6):
    if len(b) == 0:
        return b
    keep = cv2.dnn.NMSBoxes([[float(x[0]), float(x[1]), float(x[2] - x[0]), float(x[3] - x[1])] for x in b],
                            [float(x[4]) for x in b], 0.0, iou)
    return b[np.array(keep).reshape(-1)] if len(keep) else b[:0]


def yolo_txt(path, w, h):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return np.zeros((0, 4))
    a = np.loadtxt(path, ndmin=2)[:, 1:]
    return np.stack([(a[:, 0] - a[:, 2] / 2) * w, (a[:, 1] - a[:, 3] / 2) * h,
                     (a[:, 0] + a[:, 2] / 2) * w, (a[:, 1] + a[:, 3] / 2) * h], 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="name=weights@imgsz")
    ap.add_argument("--data", default="data/annotation")
    ap.add_argument("--split", default="val", choices=["train", "val", "all"])
    ap.add_argument("--hprc-split", choices=["train", "val", "test"],
                    help="use the clip-level split from hprc_split.csv (test = never used in training)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--proxy-conf", type=float, default=0.4)
    ap.add_argument("--out", default="outputs/detector_comparison.csv")
    a = ap.parse_args()

    from ultralytics import YOLO
    specs = []
    for m in a.model:
        name, rest = m.split("=", 1)
        w, sz = rest.rsplit("@", 1)
        y = YOLO(w)
        cows = [k for k, v in y.names.items() if "cow" in str(v).lower() or "cattle" in str(v).lower()]
        specs.append((name, y, int(sz), cows or None))

    man = pd.read_csv(os.path.join(a.data, "manifest.csv"))
    if a.hprc_split:
        hs = pd.read_csv(os.path.join(a.data, "hprc_split.csv"))
        man = man[man.image.isin(hs[hs.hprc_split == a.hprc_split].image)]
    elif a.split != "all":
        man = man[man.split == a.split]
    rpath = os.path.join(a.data, "reviewed.txt")
    reviewed = set(open(rpath).read().split()) if os.path.exists(rpath) else set()
    if reviewed:
        man = man[man.image.isin(reviewed)]
    mode = "HAND LABELS" if reviewed else "POOLED PROXY (not ground truth)"
    print(f"{len(man)} frames, reference = {mode}")

    rows = []
    for r in man.itertuples():
        img = cv2.imread(os.path.join(a.data, r.image))
        h, w = img.shape[:2]
        preds, times = {}, {}
        for name, y, sz, cls in specs:
            t0 = time.time()
            res = y(img, conf=min(a.conf, a.proxy_conf), imgsz=sz, classes=cls, verbose=False)[0]
            times[name] = time.time() - t0
            b = res.boxes
            preds[name] = (np.hstack([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None]])
                           if b is not None and len(b) else np.zeros((0, 5)))
        if reviewed:
            ref = yolo_txt(os.path.join(a.data, r.image.replace("images/", "labels/").rsplit(".", 1)[0] + ".txt"), w, h)
        else:
            pool = np.vstack([p[p[:, 4] >= a.proxy_conf] for p in preds.values()] or [np.zeros((0, 5))])
            ref = nms(pool)[:, :4] if len(pool) else np.zeros((0, 4))
        for name, p in preds.items():
            p = p[p[:, 4] >= a.conf]
            tp, n_pred, n_ref = match(p[:, :4], ref)
            rows.append(dict(model=name, image=r.image, camera=r.camera, night=bool(r.ir_frame),
                             tp=tp, n_pred=n_pred, n_ref=n_ref, sec=times[name]))
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)

    def summarize(g):
        tp, npred, nref = g.tp.sum(), g.n_pred.sum(), g.n_ref.sum()
        p, rc = tp / max(npred, 1), tp / max(nref, 1)
        return pd.Series(dict(recall=rc, precision=p, f1=2 * p * rc / max(p + rc, 1e-9), boxes_ref=nref,
                              ms_per_frame=1000 * g.sec.mean()))
    order = [s[0] for s in specs]
    print(f"\nOverall ({mode}, IoU 0.5, conf {a.conf}):")
    print(df.groupby("model").apply(summarize).reindex(order).round(3).to_string())
    print("\nRecall by camera / night:")
    t = df.groupby(["camera", "night", "model"]).apply(lambda g: g.tp.sum() / max(g.n_ref.sum(), 1)).unstack("model")
    print(t[order].round(3).to_string())
    print(f"\nper-frame results -> {a.out}")


if __name__ == "__main__":
    main()
