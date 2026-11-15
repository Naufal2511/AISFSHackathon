#!/bin/bash
# One pipeline chunk per array task, one GPU each. Submit with scripts/hprc/submit_pipeline.sh (it works out the
# array size and runs the smoke test first). Task 0 = the 5-minute smoke test; task N = chunk N of the share.
#SBATCH --job-name=pen6_pipe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/user/%u/pen6_logs/%x_%A_%a.out
#SBATCH --requeue

set -uo pipefail
ROOT=${PEN6_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$ROOT" || { echo "cannot cd to $ROOT"; exit 1; }
for need in venv/bin/python tools/ffmpeg/ffmpeg .env outputs/shares/plan.json; do
  [ -e "$need" ] || { echo "ERROR: $ROOT/$need missing: run scripts/hprc/pipeline_setup.sh and fetch_footage.sh first"; exit 1; }
done
source scripts/hprc/env.sh                       # modules + WebProxy (TAMU API from the compute node)
export PEN6_ID_PARALLEL=${PEN6_ID_PARALLEL:-3}   # identity-agent cases at once in THIS chunk
WHO=${PEN6_WHO:-hprc_day2}
VIDEOS=${PEN6_VIDEOS:-/scratch/user/$USER/pen6_videos}
CHUNK_H=${PEN6_CHUNK_H:-2}
I=${SLURM_ARRAY_TASK_ID:-0}
echo "== $(date) node $(hostname) GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) task $I"
COMMON=(--who "$WHO" --videos "$VIDEOS" --ffmpeg-dir "$ROOT/tools/ffmpeg" --chunk-hours "$CHUNK_H"
        --parallel 1 --stage2-workers "${PEN6_S2_WORKERS:-3}" --yes)
if [ "$I" = "0" ]; then
  venv/bin/python run_share.py "${COMMON[@]}" --smoke-only
else
  venv/bin/python run_share.py "${COMMON[@]}" --skip-checks --no-smoke --chunk "$I"
fi
rc=$?
echo "== $(date) task $I exit $rc"
exit $rc
