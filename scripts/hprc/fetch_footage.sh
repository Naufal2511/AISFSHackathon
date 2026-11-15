#!/usr/bin/env bash
# HPRC wrapper around fetch_clips.py (Google Drive -> /scratch with rclone, campus bandwidth). From the code folder:
#   bash scripts/hprc/fetch_footage.sh check     # can rclone see the organizers' folder? (configs/pipeline.yaml drive:)
#   bash scripts/hprc/fetch_footage.sh plan      # list every clip on the Drive, make outputs/shares/plan.json
#   bash scripts/hprc/fetch_footage.sh share     # download the hprc_day2 clips (+ the Day 1 clips around midnight)
#   bash scripts/hprc/fetch_footage.sh all       # everything (stitching + Level 2 here)
# One-time rclone login without a browser: docs/HPRC_PIPELINE.md step 3.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
VIDEOS=${PEN6_VIDEOS:-/scratch/user/$USER/pen6_videos}
mkdir -p "$VIDEOS" && chmod 700 "$VIDEOS"
PY="venv/bin/python fetch_clips.py --rclone $ROOT/tools/rclone"
case "${1:?usage: fetch_footage.sh check|plan|share|all}" in
  check) $PY --check ;;
  plan)  $PY --list && venv/bin/python run_share.py --plan-only --replan --videos outputs/shares/all_clips.txt
         echo "send outputs/shares/plan.json to the three laptops (they put it in outputs/shares/)" ;;
  share) $PY --list && $PY --who "${PEN6_WHO:-hprc_day2}" --dest "$VIDEOS" ;;   # Drive listing first: all of Day 2
  all)   $PY --all --dest "$VIDEOS" ;;
  *) echo "unknown command $1 (check|plan|share|all)"; exit 1 ;;
esac
echo "footage in $VIDEOS: $(find "$VIDEOS" -name 'Event*' | wc -l) clips"
