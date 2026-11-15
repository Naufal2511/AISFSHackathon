#!/usr/bin/env bash
# Run on a laptop (Mac/Linux, or Git Bash on Windows) from the repo folder: copies the code AND the model weights
# (git-tracked files, LFS ones included as real files) to HPRC. No GitHub key or git-lfs needed on the cluster.
# Rerun it after `git pull` to update the cluster copy; only changed files are sent.
#   bash scripts/hprc/push_code.sh <NetID> [host]          host default: grace.hprc.tamu.edu
set -euo pipefail
cd "$(dirname "$0")/../.."
U=${1:?usage: push_code.sh <NetID> [host]}
H=${2:-grace.hprc.tamu.edu}
for w in models/angus_det_hprc_m1280.pt models/behavior/beh_finetuned.pt; do
  [ "$(wc -c < "$w")" -gt 1000000 ] || { echo "ERROR: $w is an LFS pointer here: run 'git lfs pull' first"; exit 1; }
done
ssh "$U@$H" "mkdir -p /scratch/user/$U/pen6 && chmod 700 /scratch/user/$U/pen6"
git ls-files -z | rsync -a --from0 --files-from=- --info=stats1 ./ "$U@$H:/scratch/user/$U/pen6/"
echo "code + weights at $H:/scratch/user/$U/pen6 (commit $(git rev-parse --short HEAD))"
echo "next, on the cluster:  cd /scratch/user/$U/pen6 && bash scripts/hprc/pipeline_setup.sh"
