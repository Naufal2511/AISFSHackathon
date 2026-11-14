#!/usr/bin/env python3
"""Fast behavior appearance probe (DINOv2 ViT-S/14 frozen + logistic regression).

Step 1  export event crops pre-sorted by the geometric classifier (you fix wrong folders):
  python train_behavior_probe.py export-crops --records outputs/run_*/tracking_records.csv \
      --video-root input --out data/behavior_crops --per-class 40

Step 2  train on your labels (+ optional public datasets) and evaluate on held-out event crops:
  python train_behavior_probe.py train --event-crops data/behavior_crops \
      [--cvb-ava path/to/CVB_in_AVA_format/train.csv --cvb-frames path/to/raw_frames --cvb-pattern "{video_id}/{frame:05d}.jpg"] \
      [--cbvd-ava path/to/cbvd5_ava.csv --cbvd-frames path/to/frames --cbvd-pattern "{video_id}/{frame:06d}.jpg" --cbvd-fps 25] \
      --model-out models/behavior_probe.joblib
"""
import argparse, glob, os, sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export-crops")
    e.add_argument("--records", nargs="+", required=True)
    e.add_argument("--video-root", required=True)
    e.add_argument("--out", default="data/behavior_crops")
    e.add_argument("--per-class", type=int, default=40)
    t = sub.add_parser("train")
    t.add_argument("--event-crops", default=None)
    for ds in ("cvb", "cbvd"):
        t.add_argument(f"--{ds}-ava"); t.add_argument(f"--{ds}-frames"); t.add_argument(f"--{ds}-pattern")
    t.add_argument("--cvb-fps", type=float, default=30.0)
    t.add_argument("--cbvd-fps", type=float, default=25.0)
    t.add_argument("--max-per-class-external", type=int, default=400)
    t.add_argument("--model-out", default="models/behavior_probe.joblib")
    a = ap.parse_args()

    if a.cmd == "export-crops":
        from src.behavior.classifier import BehaviorClassifier
        from src.behavior.appearance import export_crops
        from src.behavior.rumination import index_videos
        recs = sorted({p for pat in a.records for p in glob.glob(pat)})
        clf = BehaviorClassifier(); tl = clf.run(recs)
        n = export_crops(clf.rows, tl, index_videos(a.video_root), a.out, a.per_class)
        print(f"Wrote {n} crops to {a.out}/<class>/ — move mislabeled crops to the correct folder, then run `train`.")
        return

    from src.behavior.appearance import manifest_from_ava, manifest_from_folders, train_probe
    parts = []
    if a.event_crops:
        parts.append(manifest_from_folders(a.event_crops, "event"))
    if a.cvb_ava:
        parts.append(manifest_from_ava(a.cvb_ava, a.cvb_frames, a.cvb_pattern, a.cvb_fps, "cvb"))
    if a.cbvd_ava:
        parts.append(manifest_from_ava(a.cbvd_ava, a.cbvd_frames, a.cbvd_pattern, a.cbvd_fps, "cbvd5"))
    man = pd.concat(parts, ignore_index=True)
    ext = man["source"] != "event"  # cap external data so it cannot swamp the event domain
    man = pd.concat([man[~ext], man[ext].groupby("label", group_keys=False).apply(
        lambda d: d.sample(min(len(d), a.max_per_class_external), random_state=0))])
    print(man.groupby(["source", "label"]).size().to_string())
    os.makedirs(os.path.dirname(a.model_out) or ".", exist_ok=True)
    train_probe(man, a.model_out)
    print(f"Saved probe to {a.model_out}")


if __name__ == "__main__":
    main()
