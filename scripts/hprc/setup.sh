#!/usr/bin/env bash
# Run ONCE on an HPRC login node (it needs internet for pip), from the unpacked bundle:
#   cd /scratch/user/$USER/angus && tar -xzf hprc_bundle.tar.gz && bash scripts/hprc/setup.sh
# Creates the Python env, points the dataset yaml at this folder, makes log/run folders.
set -euo pipefail
ROOT=${ANGUS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}   # the bundle folder
cd "$ROOT"
echo "bundle folder: $ROOT"
[ -f data/annotation/hprc_dataset.yaml ] || { echo "ERROR: no data/annotation here - unpack hprc_bundle.tar.gz in $ROOT first"; exit 1; }
source scripts/hprc/env.sh            # module loads (edit there if your cluster differs)

# a Python >= 3.9 (the module one if the load worked, else whatever the system has)
PY=""
for c in python3.12 python3.11 python3.10 python3 python; do
  if command -v $c >/dev/null 2>&1 && $c -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
    PY=$c; break
  fi
done
[ -n "$PY" ] || { echo "ERROR: no Python >= 3.9. Try: module spider Python, then edit scripts/hprc/env.sh"; exit 1; }
echo "using $($PY --version) from $(command -v $PY)"
mkdir -p /scratch/user/$USER/angus_logs 2>/dev/null || true   # Slurm stdout of train.slurm goes here

mkdir -p logs runs test_preds
chmod 700 "$ROOT"                      # event data: readable by you only

if [ ! -x venv/bin/python ]; then
  $PY -m venv venv
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
echo "Setup done. Submit FROM THIS FOLDER:  cd $ROOT && sbatch scripts/hprc/train.slurm"
