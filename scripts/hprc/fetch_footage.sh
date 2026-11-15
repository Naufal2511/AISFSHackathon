#!/usr/bin/env bash
# Google Drive -> HPRC /scratch with rclone, on the cluster (campus bandwidth, no laptop in between).
#
# One-time login (HPRC has no browser):
#   tools/rclone config        n (new) -> name: gdrive -> storage: drive -> client_id/secret: Enter -> scope: 2 (drive.readonly)
#                              -> service_account: Enter -> advanced: n -> "Use web browser?": n
#   it prints   rclone authorize "drive" "eyJ..."   : run THAT on your laptop (brew install rclone / rclone.org),
#   a browser opens, sign in with the account the footage is shared with, paste the long result back -> Shared Drive: n -> y -> q
#
# Then, from the code folder (/scratch/user/$USER/pen6):
#   bash scripts/hprc/fetch_footage.sh list  'gdrive:Pen 6 videos'   # list every clip -> outputs/shares/all_clips.txt
#   bash scripts/hprc/fetch_footage.sh plan                           # plan.json from that list (send it to the laptops)
#   bash scripts/hprc/fetch_footage.sh share 'gdrive:Pen 6 videos'   # download the hprc_day2 clips (+ Day 1 lead-in)
#   bash scripts/hprc/fetch_footage.sh all   'gdrive:Pen 6 videos'   # everything (stitching + Level 2 here)
#
# The folder was SHARED with you (not in your My Drive)?  export RCLONE_FLAGS="--drive-shared-with-me"
# Only have the folder link?  export RCLONE_FLAGS="--drive-root-folder-id <ID from .../folders/<ID>>" and use 'gdrive:'
# "downloadQuotaExceeded": Google throttles a shared file many people download. Copy it into your own Drive first
# (server side, nothing downloaded):  tools/rclone copy 'gdrive:Pen 6 videos' 'gdrive:pen6_copy' --drive-shared-with-me
# --drive-server-side-across-configs -P   then fetch from 'gdrive:pen6_copy' without --drive-shared-with-me.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
RC="$ROOT/tools/rclone"
VIDEOS=${PEN6_VIDEOS:-/scratch/user/$USER/pen6_videos}
CMD=${1:?usage: fetch_footage.sh list|plan|share|all ['remote:folder']}
REMOTE=${2:-}
FLAGS=${RCLONE_FLAGS:-}
mkdir -p outputs/shares "$VIDEOS"
chmod 700 "$VIDEOS"
case "$CMD" in
  list)
    [ -n "$REMOTE" ] || { echo "give the remote folder, e.g. 'gdrive:Pen 6 videos'"; exit 1; }
    $RC lsf -R --files-only --include "Event*" $FLAGS "$REMOTE" > outputs/shares/all_clips.txt
    echo "$(wc -l < outputs/shares/all_clips.txt) clips listed -> outputs/shares/all_clips.txt"
    head -3 outputs/shares/all_clips.txt ;;
  plan)
    venv/bin/python run_share.py --plan-only --videos outputs/shares/all_clips.txt --replan
    echo "send outputs/shares/plan.json to the three laptops (they put it in outputs/shares/)" ;;
  share|all)
    [ -n "$REMOTE" ] || { echo "give the remote folder, e.g. 'gdrive:Pen 6 videos'"; exit 1; }
    INC=outputs/shares/hprc_include.txt
    if [ "$CMD" = share ]; then
      venv/bin/python - <<'PY'
import json, sys
sys.path.insert(0, ".")
import run_share as rs
plan = json.load(open("outputs/shares/plan.json"))
sh = plan["shares"][rs.HPRC_SHARE]
start = rs.lead_in(sh["start"], plan["span"][0], 5)          # the clip(s) running into the first chunk's lead-in
names = rs.missing_for_window(plan, set(), start, sh["end"])  # everything that window needs
open("outputs/shares/hprc_include.txt", "w").write("".join(f"**/{n}\n" for n in names))
print(f"{len(names)} clips for {rs.HPRC_SHARE} ({start} -> {sh['end']})")
PY
    else
      echo "Event*" > "$INC"
    fi
    $RC copy $FLAGS "$REMOTE" "$VIDEOS" --include-from "$INC" --transfers 8 --checkers 16 \
        --drive-chunk-size 64M --retries 5 --low-level-retries 20 -P
    echo "footage in $VIDEOS: $(find "$VIDEOS" -name 'Event*' | wc -l) clips, $(du -sh "$VIDEOS" | cut -f1)" ;;
  *) echo "unknown command $CMD (list|plan|share|all)"; exit 1 ;;
esac
