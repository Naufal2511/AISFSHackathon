#!/usr/bin/env bash
# Run ONCE on an HPRC LOGIN node (it needs internet), inside the code folder pushed by push_code.sh:
#   cd /scratch/user/$USER/pen6 && bash scripts/hprc/pipeline_setup.sh
# Makes: venv/ (torch with CUDA, all requirements), tools/ffmpeg + tools/rclone (static builds), the DINOv2 cache,
# and checks the .env. Safe to rerun: finished parts are skipped.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
chmod 700 "$ROOT"                                    # event data and the TAMU key: readable by you only
source scripts/hprc/env.sh
mkdir -p tools /scratch/user/$USER/pen6_logs

# --- Python env
PY=""
for c in python3.12 python3.11 python3.10 python3; do
  if command -v $c >/dev/null 2>&1 && $c -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)' 2>/dev/null; then PY=$c; break; fi
done
[ -n "$PY" ] || { echo "ERROR: no Python 3.10-3.12. Run 'module spider Python' and edit scripts/hprc/env.sh"; exit 1; }
if [ ! -x venv/bin/python ]; then
  echo "== creating venv with $($PY --version)"
  $PY -m venv venv
  venv/bin/pip install -q --upgrade pip
fi
echo "== installing requirements (torch with CUDA from PyPI; a few minutes the first time)"
venv/bin/pip install -q -r requirements.txt -r brd_agent/requirements.txt
venv/bin/python -c "import torch; print('torch', torch.__version__, 'CUDA build', torch.version.cuda)"

# --- ffmpeg: a static build (decodes the cameras' HEVC); the module one may lack it
if [ ! -x tools/ffmpeg/ffmpeg ]; then
  echo "== downloading static ffmpeg"
  curl -fsSL -o tools/ffmpeg.tar.xz https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
  mkdir -p tools/ffmpeg && tar -xJf tools/ffmpeg.tar.xz -C tools/ffmpeg --strip-components=1 && rm tools/ffmpeg.tar.xz
fi
tools/ffmpeg/ffmpeg -version | head -1

# --- rclone: Google Drive -> /scratch without going through a laptop
if [ ! -x tools/rclone ]; then
  echo "== downloading rclone"
  curl -fsSL -o tools/rclone.zip https://downloads.rclone.org/rclone-current-linux-amd64.zip
  (cd tools && unzip -q -o rclone.zip && mv rclone-*-linux-amd64/rclone . && rm -rf rclone-*-linux-amd64 rclone.zip)
fi
tools/rclone version | head -1

# --- DINOv2 weights into the torch cache now (compute nodes reach the internet only through the proxy)
venv/bin/python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', verbose=False); print('DINOv2 cached')"

# --- TAMU key
if [ ! -s .env ] || ! grep -q '^TAMU_API_KEY=.\+' .env; then
  echo
  echo "MISSING: $ROOT/.env with your TAMU key. Create it with:"
  echo "   printf 'TAMU_API_KEY=%s\n' '<key>' > $ROOT/.env && chmod 600 $ROOT/.env"
else
  chmod 600 .env
  echo ".env present"
fi
echo
echo "Setup done. Next: footage (scripts/hprc/fetch_footage.sh), then:"
echo "   venv/bin/python check_setup.py --who hprc_day2 --videos /scratch/user/$USER/pen6_videos --ffmpeg-dir $ROOT/tools/ffmpeg"
