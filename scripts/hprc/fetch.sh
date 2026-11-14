#!/usr/bin/env bash
# Run on the Mac: copy trained weights, metrics, test predictions and logs back.
#   ./scripts/hprc/fetch.sh <NetID> [host]      (host default grace.hprc.tamu.edu)
set -euo pipefail
cd "$(dirname "$0")/../.."
U=${1:?usage: fetch.sh <NetID> [host]}
H=${2:-grace.hprc.tamu.edu}
mkdir -p outputs/hprc
rsync -av --exclude 'epoch*.pt' "$U@$H:/scratch/user/$U/angus/runs" "$U@$H:/scratch/user/$U/angus/test_preds" \
      "$U@$H:/scratch/user/$U/angus/logs" outputs/hprc/
for f in outputs/hprc/runs/*/summary.json; do cat "$f"; echo; done
