#!/bin/bash
# Fine-tune COCO-pretrained YOLO26 detectors on the hand-labelled Pen 6 frames, one model per
# array task, each on one Tesla T4.
#   sbatch scripts/hprc/train.slurm                 # s, m, l
#   sbatch --array=1 scripts/hprc/train.slurm       # just m
#   sbatch --array=3 --time=08:00:00 scripts/hprc/train.slurm   # x (slow)
# Everything lives in the bundle folder you submit from (e.g. /scratch/user/$USER/angus):
#   logs/        per-attempt training logs (Slurm stdout: /scratch/user/$USER/angus_logs/)
#   runs/<name>/ checkpoints (last.pt every epoch, epochN.pt every 10), best.pt, plots, results.csv
#   runs/<name>_test/  metrics on the held-out TEST split (never used for training/selection)
#   test_preds/<name>/ test images with predicted boxes drawn
# Re-submitting (or a preemption/timeout requeue) RESUMES from last.pt.
#SBATCH --job-name=angus_yolo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=/scratch/user/%u/angus_logs/%x_%A_%a.out
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

set -uo pipefail
# The bundle folder = where you ran sbatch (Slurm runs a COPY of this script, so its own
# path is useless). Run sbatch from the folder that contains scripts/ data/ venv/.
ROOT=${ANGUS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$ROOT" || { echo "cannot cd to $ROOT"; exit 1; }
for need in scripts/hprc/env.sh venv/bin/activate data/annotation/dataset_hprc.yaml; do
  if [ ! -e "$need" ]; then
    echo "ERROR: $ROOT/$need missing. Run sbatch from the unpacked bundle folder, after"
    echo "       'bash scripts/hprc/setup.sh' finished there without errors."
    exit 1
  fi
done
mkdir -p logs runs test_preds
source scripts/hprc/env.sh
source venv/bin/activate
command -v yolo >/dev/null || { echo "ERROR: 'yolo' not in venv (setup.sh pip install failed?)"; exit 1; }
echo "== bundle folder: $ROOT"

MODELS=(yolo26s yolo26m yolo26l yolo26x)
BATCHES=(16 8 4 2)               # T4 16 GB at imgsz 1280 with AMP; halved automatically on OOM
EPOCHS=${EPOCHS:-120}
IMGSZ=${IMGSZ:-1280}
I=${SLURM_ARRAY_TASK_ID:-1}
MODEL=${MODELS[$I]}
NAME=${MODEL}_${IMGSZ}_hand
RUN=$ROOT/runs/$NAME
DATA=$ROOT/data/annotation/dataset_hprc.yaml

echo "== $(date) job ${SLURM_JOB_ID:-?} task $I model $MODEL host $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Requeue (and later resume) shortly before the time limit
PID=
trap 'echo "== $(date) time limit near: requeue"; [ -n "$PID" ] && kill -TERM $PID 2>/dev/null; wait; scontrol requeue ${SLURM_JOB_ID}; exit 0' USR1

run_bg() {  # run_bg LOGFILE cmd...: background + wait, so the USR1 trap can fire mid-training
  local log=$1; shift
  "$@" >> "$log" 2>&1 &
  PID=$!
  wait $PID
  local rc=$?
  tail -n 4 "$log"
  return $rc
}

if [ -f "$RUN/TRAIN_DONE" ]; then
  echo "training already finished -> evaluation only"
elif [ -f "$RUN/weights/last.pt" ]; then
  echo "resuming from $RUN/weights/last.pt"
  run_bg "$ROOT/logs/${NAME}_train_resume.log" yolo detect train resume=True model="$RUN/weights/last.pt"
  [ -f "$RUN/weights/best.pt" ] && touch "$RUN/TRAIN_DONE"
else
  B=${BATCHES[$I]}
  while [ "$B" -ge 1 ]; do
    echo "== training $MODEL batch $B"
    run_bg "$ROOT/logs/${NAME}_train_b${B}.log" yolo detect train model="$MODEL.pt" data="$DATA" imgsz=$IMGSZ epochs=$EPOCHS batch=$B \
        patience=30 close_mosaic=10 cos_lr=True save_period=10 cache=ram workers=8 \
        project="$ROOT/runs" name="$NAME" exist_ok=True plots=True
    if grep -qiE "out of memory|OutOfMemoryError" "$ROOT/logs/${NAME}_train_b${B}.log"; then
      echo "== OOM at batch $B, retrying with $((B / 2))"
      rm -rf "$RUN"
      B=$((B / 2))
      continue
    fi
    break
  done
  [ -f "$RUN/weights/best.pt" ] && touch "$RUN/TRAIN_DONE"
fi

if [ ! -f "$RUN/weights/best.pt" ]; then
  echo "no best.pt - training failed, see logs/${NAME}_train*.log"
  exit 1
fi

echo "== $(date) evaluating best.pt on the held-out TEST split"
yolo detect val model="$RUN/weights/best.pt" data="$DATA" split=test imgsz=$IMGSZ batch=4 \
    project="$ROOT/runs" name="${NAME}_test" exist_ok=True plots=True 2>&1 | tee "$ROOT/logs/${NAME}_test.log"

echo "== annotated test predictions -> test_preds/$NAME"
yolo detect predict model="$RUN/weights/best.pt" source="$ROOT/data/annotation/test_images" \
    imgsz=$IMGSZ conf=0.25 save=True line_width=4 project="$ROOT/test_preds" name="$NAME" exist_ok=True \
    2>&1 | tail -3

python - "$RUN" "$NAME" <<'EOF'
import json, sys, pandas as pd
run, name = sys.argv[1], sys.argv[2]
r = pd.read_csv(f"{run}/results.csv"); r.columns = [c.strip() for c in r.columns]
best = r.loc[r["metrics/mAP50-95(B)"].idxmax()]
out = {"model": name, "epochs_run": int(r["epoch"].max()), "best_epoch": int(best["epoch"]),
       "val_precision": round(best["metrics/precision(B)"], 4), "val_recall": round(best["metrics/recall(B)"], 4),
       "val_mAP50": round(best["metrics/mAP50(B)"], 4), "val_mAP50_95": round(best["metrics/mAP50-95(B)"], 4)}
json.dump(out, open(f"{run}/summary.json", "w"), indent=1)
print(out)
EOF
echo "== $(date) done. Copy back: runs/$NAME/weights/best.pt, runs/${NAME}_test, test_preds/$NAME"
