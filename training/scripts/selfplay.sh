#!/bin/bash
# Slurm job script for AlphaZero self-play
# USE THE SUBMIT WRAPPER: ./submit_selfplay.sh
#
# Expected environment variables (set by submit_selfplay.sh):
#   SELFPLAY_ARGS - arguments for the self-play binary

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/willzhao/alphazero}"
CONDA_ENV="${CONDA_ENV:-${PROJECT_DIR}/.conda/env}"
SELFPLAY_ARGS="${SELFPLAY_ARGS:-}"

cd "$PROJECT_DIR"

module load miniforge
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
python -m pip install -q -r "$PROJECT_DIR/requirements.txt"

TORCH_LIB="$(python - <<'PY'
from pathlib import Path
import torch
print(Path(torch.__file__).resolve().parent / "lib")
PY
)"

echo "=== AlphaZero Self-Play ==="
echo "  project dir: $PROJECT_DIR"
echo "  slurm job id: ${SLURM_JOB_ID:-manual}"
echo "  node: $(hostname)"
echo "  selfplay args: $SELFPLAY_ARGS"
echo "  cuda visible devices: ${CUDA_VISIBLE_DEVICES:-not set}"

# The self-play binary is a compiled Rust program
if [[ ! -x "$PROJECT_DIR/target/release/self-play" ]]; then
    echo "ERROR: missing $PROJECT_DIR/target/release/self-play" >&2
    echo "Build it first with: LIBTORCH_USE_PYTORCH=1 cargo build --release -p self-play" >&2
    exit 1
fi

export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"

"$PROJECT_DIR/target/release/self-play" $SELFPLAY_ARGS
