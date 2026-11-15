#!/usr/bin/env python3
"""Level-1 behavior classification for all cows from tracking_records.csv.

Examples
  # geometry-only (no video decoding, seconds to run)
  python classify_behavior.py --records outputs/tracking_records.csv --out outputs/behavior

  # several runs (clips / days) at once, expanded to every minute of both days
  python classify_behavior.py --records outputs/run_*/tracking_records.csv \
      --full-days 2026-11-07 2026-11-08 --out outputs/behavior

  # + measured rumination (decodes ~12 s snippets of long still bouts only)
  python classify_behavior.py --records ... --video-root input --probe-rumination

  # + learned appearance expert (see train_behavior_probe.py)
  python classify_behavior.py --records ... --video-root input --appearance-model models/behavior_probe.joblib
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.behavior.classifier import BehaviorClassifier
from src.behavior.export import write_all


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", nargs="+", default=["outputs/tracking_records.csv"])
    ap.add_argument("--out", default="outputs/behavior")
    ap.add_argument("--config", default="configs/behavior.yaml")
    ap.add_argument("--clip-start", default=None,
                    help="Wall-clock start (e.g. '2026-11-07 07:19:45') for CSVs without a source_video column")
    ap.add_argument("--full-days", nargs="*", default=None, help="Expand activity.csv to every minute of these dates")
    ap.add_argument("--video-root", default=None, help="Raw video root (needed for rumination probes / appearance)")
    ap.add_argument("--probe-rumination", action="store_true")
    ap.add_argument("--appearance-model", default=None)
    ap.add_argument("--appearance-every-sec", type=int, default=10)
    ap.add_argument("--emission", choices=["auto", "learned", "heuristic"], default="auto",
                    help="auto = fine-tuned classifier when enabled in config and --video-root is given")
    ap.add_argument("--weights", default=None, help="Override learned_model.weights")
    ap.add_argument("--no-1s", action="store_true", help="Skip the large per-second timeline CSV")
    a = ap.parse_args()

    records = sorted({p for pat in a.records for p in glob.glob(pat)})
    if not records:
        sys.exit(f"No record files match {a.records}")

    clf = BehaviorClassifier(a.config)
    video_index = None
    if a.video_root:
        from src.behavior.rumination import index_videos
        video_index = index_videos(a.video_root)
        print(f"[BEHAVIOR] indexed {len(video_index)} raw videos under {a.video_root}")

    appearance = None
    if a.appearance_model:
        from src.behavior.appearance import predict_rows
        clf.load(records, a.clip_start)
        appearance = predict_rows(a.appearance_model, clf.rows, video_index or {}, a.appearance_every_sec)
        print(f"[BEHAVIOR] appearance expert scored {len(appearance)} cow-seconds")

    lc = clf.cfg.get("learned_model", {})
    use_learned = a.emission == "learned" or (a.emission == "auto" and lc.get("enabled") and video_index)
    if a.emission == "learned" and not video_index:
        sys.exit("--emission learned needs --video-root (crops are read from the raw video)")
    model_name = "heuristic-fusion-v1"
    os.makedirs(a.out, exist_ok=True)
    if use_learned:
        if clf.fused is None:
            clf.load(records, a.clip_start)
        preds = clf.attach_learned(video_index, a.weights)
        if preds is not None:
            preds.to_csv(os.path.join(a.out, "box_predictions.csv"), index=False)
            model_name = clf.model.version
    with open(os.path.join(a.out, "model.txt"), "w") as f:  # read by scripts/export_contract.py
        f.write(model_name + "\n")
    print(f"[BEHAVIOR] emission model: {model_name}")

    tl = clf.run(records, a.clip_start, appearance, video_index, a.probe_rumination)
    paths = write_all(a.out, tl, clf.cfg, full_days=a.full_days, write_1s=not a.no_1s)
    print("[BEHAVIOR] behavior share:")
    print(tl["behavior"].value_counts(normalize=True).round(3).to_string())
    for k, v in paths.items():
        print(f"  {k:18s}: {v}")


if __name__ == "__main__":
    main()
