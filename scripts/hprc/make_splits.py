#!/usr/bin/env python3
"""Train / val / TEST split of the hand-labelled frames, for HPRC training.

* train = the reviewed `train` frames (unchanged).
* the reviewed `val` frames are split BY CLIP into
    val  -> used during training (early stopping, best.pt selection, choosing s/m/l)
    test -> touched only once at the end, so the reported recall is not tuned on.
  Clips are assigned alternately within each (camera, day/night) group so both halves
  cover every camera and lighting condition.

Writes into data/annotation/:
  hprc_train.txt / hprc_val.txt / hprc_test.txt   (paths './images/...', relative to the list)
  hprc_dataset.yaml                                 (path: filled in on the cluster by setup.sh)
  hprc_split.csv                                    (image -> split, for per-camera reports)
"""
import argparse
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/annotation")
    a = ap.parse_args()

    m = pd.read_csv(os.path.join(a.data, "manifest.csv"))
    reviewed = set(open(os.path.join(a.data, "reviewed.txt")).read().split())
    m = m[m.image.isin(reviewed)].copy()
    m["hprc_split"] = m.split

    val = m[m.split == "val"]
    for _, g in val.groupby(["camera", "ir_frame"]):
        clips = sorted(g["clip"].unique(), key=lambda c: (-int((g["clip"] == c).sum()), c))
        for i, c in enumerate(clips):
            m.loc[(m.split == "val") & (m["clip"] == c), "hprc_split"] = "val" if i % 2 == 0 else "test"

    for s in ("train", "val", "test"):
        with open(os.path.join(a.data, f"hprc_{s}.txt"), "w") as f:
            for img in sorted(m[m.hprc_split == s].image):
                f.write(f"./{img}\n")
    with open(os.path.join(a.data, "hprc_dataset.yaml"), "w") as f:
        f.write("path: __DATA_DIR__\ntrain: hprc_train.txt\nval: hprc_val.txt\ntest: hprc_test.txt\n"
                "names:\n  0: cow\n")
    m[["image", "camera", "clip", "ir_frame", "hprc_split"]].to_csv(
        os.path.join(a.data, "hprc_split.csv"), index=False)

    def boxes(img):
        p = os.path.join(a.data, img.replace("images/", "labels/").rsplit(".", 1)[0] + ".txt")
        return sum(1 for l in open(p) if l.strip()) if os.path.exists(p) else 0
    m["boxes"] = m.image.map(boxes)
    t = m.groupby(["hprc_split", "camera", "ir_frame"]).agg(frames=("image", "size"), boxes=("boxes", "sum"))
    print(t.unstack("ir_frame").fillna(0).astype(int).to_string())


if __name__ == "__main__":
    main()
