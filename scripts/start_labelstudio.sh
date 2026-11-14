#!/usr/bin/env bash
# Start a LOCAL Label Studio (data never leaves this machine) that can read images from this repo.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$ROOT"
echo "Label Studio -> http://localhost:8080   (local-files root: $ROOT)"
exec "$ROOT/.venv-labelstudio/bin/label-studio" start --port 8080 "$@"
