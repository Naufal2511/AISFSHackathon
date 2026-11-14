#!/usr/bin/env bash
# Run ONCE on an HPRC login node (it needs internet for pip):
#   cd /scratch/user/$USER/angus && tar -xzf hprc_bundle.tar.gz && bash scripts/hprc/setup.sh
# Creates the Python env, unpacks the dataset to a fixed path and makes the log/run folders.
set -euo pipefail
ROOT=${ANGUS_ROOT:-/scratch/user/$USER/angus}
cd "$ROOT"
source scripts/hprc/env.sh            # module loads (edit there if your cluster differs)

mkdir -p logs runs test_preds
chmod 700 "$ROOT"                      # event data: readable by you only

if [ ! -x venv/bin/python ]; then
  python -m venv venv
  venv/bin/pip install -q --upgrade pip
  venv/bin/pip install -q "ultralytics==8.4.212" pandas
fi
# dataset yaml with the absolute path on this cluster
sed "s#__DATA_DIR__#$ROOT/data/annotation#" data/annotation/hprc_dataset.yaml > data/annotation/dataset_hprc.yaml
# a folder of test images (symlinks) for annotated predictions
mkdir -p data/annotation/test_images
while read -r p; do ln -sf "$ROOT/data/annotation/${p#./}" data/annotation/test_images/; done < data/annotation/hprc_test.txt

# compute nodes may be offline: fetch the plot font now
venv/bin/python -c "from ultralytics.utils.checks import check_font; check_font('Arial.ttf')" || true
venv/bin/python -c "import torch, ultralytics; print('torch', torch.__version__, 'cuda build', torch.version.cuda); print('ultralytics', ultralytics.__version__)"
echo "train $(wc -l < data/annotation/hprc_train.txt)  val $(wc -l < data/annotation/hprc_val.txt)  test $(wc -l < data/annotation/hprc_test.txt) images"
echo "Setup done. Submit with:  sbatch scripts/hprc/train.slurm"
