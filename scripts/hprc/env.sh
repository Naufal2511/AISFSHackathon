# Sourced by setup.sh and train.slurm. TAMU HPRC (Grace / FASTER) uses Lmod modules.
# If the Python version below is missing, run `module spider Python` and pick a 3.10-3.12 build.
if command -v module >/dev/null 2>&1; then
  module purge 2>/dev/null || true
  module load GCCcore/12.3.0 Python/3.11.3 2>/dev/null \
    || module load GCCcore/13.2.0 Python/3.11.5 2>/dev/null \
    || module load Python 2>/dev/null || true
  module load WebProxy 2>/dev/null || true   # internet from compute nodes on HPRC
fi
export PYTHONUNBUFFERED=1
