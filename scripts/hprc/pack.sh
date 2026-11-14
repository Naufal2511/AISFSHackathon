#!/usr/bin/env bash
# Run on the Mac. Bundles the hand-labelled dataset, COCO-pretrained weights and the HPRC
# scripts into one tarball to copy to the cluster.
#   ./scripts/hprc/pack.sh            -> outputs/hprc_bundle.tar.gz
set -euo pipefail
cd "$(dirname "$0")/../.."
.venv/bin/python scripts/hprc/make_splits.py >/dev/null
OUT=outputs/hprc_bundle.tar.gz
WEIGHTS=$(ls yolo26n.pt yolo26s.pt yolo26m.pt yolo26l.pt yolo26x.pt 2>/dev/null)  # n: AMP self-check
export COPYFILE_DISABLE=1   # no macOS ._ files in the tar
tar -czf "$OUT" --exclude='*.cache' --exclude='.DS_Store' \
    data/annotation/images data/annotation/labels data/annotation/manifest.csv \
    data/annotation/hprc_train.txt data/annotation/hprc_val.txt data/annotation/hprc_test.txt \
    data/annotation/hprc_dataset.yaml data/annotation/hprc_split.csv \
    scripts/hprc $WEIGHTS
ls -lh "$OUT"
echo "Next: scp $OUT <NetID>@grace.hprc.tamu.edu:/scratch/user/<NetID>/angus/"
