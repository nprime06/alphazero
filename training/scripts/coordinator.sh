#!/bin/bash
# Slurm job script for AlphaZero pipeline coordinator
# USE THE SUBMIT WRAPPER: ./submit_coordinator.sh
#
# Expected environment variables (set by submit_coordinator.sh):
#   RUN_DIR          - self-contained run directory
#   COORDINATOR_ARGS - arguments for orchestrator.coordinator

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/willzhao/alphazero}"
CONDA_ENV="${CONDA_ENV:-${PROJECT_DIR}/.conda/env}"
RUN_DIR="${RUN_DIR:?RUN_DIR must be set by submit_coordinator.sh}"
COORDINATOR_ARGS="${COORDINATOR_ARGS:-}"

cd "$PROJECT_DIR"

module load miniforge
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
python -m pip install -q -r "$PROJECT_DIR/requirements.txt"
python -m pip install -q \
    -e "$PROJECT_DIR/neural" \
    -e "$PROJECT_DIR/training" \
    -e "$PROJECT_DIR/orchestrator" \
    -e "$PROJECT_DIR/alphazero"

TORCH_LIB="$(python - <<'PY'
from pathlib import Path
import torch
print(Path(torch.__file__).resolve().parent / "lib")
PY
)"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR/neural:$PROJECT_DIR/training:$PROJECT_DIR/orchestrator:$PROJECT_DIR/alphazero:${PYTHONPATH:-}"
export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"

if [[ ! -x "$PROJECT_DIR/target/release/self-play" ]]; then
    echo "ERROR: missing $PROJECT_DIR/target/release/self-play" >&2
    echo "Build it first with: LIBTORCH_USE_PYTORCH=1 cargo build --release -p self-play" >&2
    exit 1
fi

if ! python - <<'PY'
import alphazero_py
PY
then
    echo "alphazero_py not installed; building editable PyO3 extension"
    python -m pip install -q maturin
    python -m maturin develop --manifest-path "$PROJECT_DIR/alphazero-py/Cargo.toml" --quiet
fi

echo "=== AlphaZero Pipeline Coordinator ==="
echo "  project dir: $PROJECT_DIR"
echo "  slurm job id: ${SLURM_JOB_ID:-manual}"
echo "  node: $(hostname)"
echo "  run dir: $RUN_DIR"
echo "  coordinator args: $COORDINATOR_ARGS"
echo "  cuda visible devices: ${CUDA_VISIBLE_DEVICES:-not set}"

python -m orchestrator.coordinator \
    --project-dir "$PROJECT_DIR" \
    --run-dir "$RUN_DIR" \
    $COORDINATOR_ARGS
