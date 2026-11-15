#!/usr/bin/env bash
# Submit the HPRC share: the 5-minute smoke test first, then every chunk as an array task that starts only if the
# smoke test passed. From the code folder on a login node:
#   bash scripts/hprc/submit_pipeline.sh                 # 2 h chunks, at most 8 running at once
#   PEN6_CHUNK_H=4 MAX_AT_ONCE=6 bash scripts/hprc/submit_pipeline.sh
#   GPU=a100 bash scripts/hprc/submit_pipeline.sh        # another GPU type (whatever your allocation has)
# Watch: squeue -u $USER      logs: /scratch/user/$USER/pen6_logs/      resubmitting resumes finished steps.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
WHO=${PEN6_WHO:-hprc_day2}
VIDEOS=${PEN6_VIDEOS:-/scratch/user/$USER/pen6_videos}
export PEN6_CHUNK_H=${PEN6_CHUNK_H:-2}
MAX=${MAX_AT_ONCE:-8}
GPU=${GPU:-t4}
source scripts/hprc/env.sh

echo "== setup check (login node: no GPU here, that warning is expected)"
venv/bin/python check_setup.py --who "$WHO" --videos "$VIDEOS" --ffmpeg-dir "$ROOT/tools/ffmpeg" || {
  echo "NOT READY: fix the FAIL lines above, then run this again"; exit 1; }

N=$(venv/bin/python run_share.py --who "$WHO" --videos "$VIDEOS" --chunk-hours "$PEN6_CHUNK_H" --parallel 1 \
      --dry-run --yes --skip-checks --no-smoke | grep -c "^  ${WHO}_c[0-9]")
[ "$N" -gt 0 ] || { echo "no chunks found for $WHO"; exit 1; }
echo "== $WHO: $N chunks of ${PEN6_CHUNK_H} h, at most $MAX at once, GPU $GPU"
SM=$(sbatch --parsable --array=0 --gres=gpu:$GPU:1 --export=ALL scripts/hprc/pipeline.slurm)
CH=$(sbatch --parsable --array=1-$N%$MAX --gres=gpu:$GPU:1 --dependency=afterok:$SM --export=ALL scripts/hprc/pipeline.slurm)
echo "smoke test job $SM; chunks job $CH (start only if the smoke test passes)"
echo "watch:  squeue -u $USER     tail -f /scratch/user/$USER/pen6_logs/pen6_pipe_${SM}_0.out"
