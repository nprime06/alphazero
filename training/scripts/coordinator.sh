#!/bin/bash
# Slurm job script for AlphaZero pipeline coordinator
# USE THE SUBMIT WRAPPER: ./submit_coordinator.sh
#
# Expected environment variables (set by submit_coordinator.sh):
#   RUN_DIR          - self-contained run directory
#   COORDINATOR_ARGS - arguments for orchestrator.coordinator
#   RESUME_DOCTOR   - if 1, validate RUN_DIR before running coordinator

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/willzhao/alphazero}"
CONDA_ENV="${CONDA_ENV:-${PROJECT_DIR}/.conda/env}"
RUN_DIR="${RUN_DIR:?RUN_DIR must be set by submit_coordinator.sh}"
COORDINATOR_ARGS="${COORDINATOR_ARGS:-}"
RESUME_DOCTOR="${RESUME_DOCTOR:-0}"

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
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$TORCH_LIB:${LD_LIBRARY_PATH:-}"

if [[ ! -x "$PROJECT_DIR/target/release/self-play" ]]; then
    echo "ERROR: missing $PROJECT_DIR/target/release/self-play" >&2
    echo "Build it first with: LIBTORCH_USE_PYTORCH=1 cargo build --release -p self-play" >&2
    exit 1
fi

check_rust_eval_loader() {
    python - <<'PY'
from orchestrator.evaluate import _load_alphazero_py

_load_alphazero_py()
PY
}

if ! check_rust_eval_loader
then
    echo "alphazero_py Rust evaluation loader failed; rebuilding editable PyO3 extension"
    python -m pip install -q maturin
    python -m maturin develop --manifest-path "$PROJECT_DIR/alphazero-py/Cargo.toml" --quiet
    check_rust_eval_loader
fi

echo "=== AlphaZero Pipeline Coordinator ==="
echo "  project dir: $PROJECT_DIR"
echo "  slurm job id: ${SLURM_JOB_ID:-manual}"
echo "  node: $(hostname)"
echo "  run dir: $RUN_DIR"
echo "  coordinator args: $COORDINATOR_ARGS"
echo "  resume doctor: $RESUME_DOCTOR"
echo "  cuda visible devices: ${CUDA_VISIBLE_DEVICES:-not set}"

if [[ "$RESUME_DOCTOR" == "1" && -f "$RUN_DIR/pipeline_state.yaml" ]]; then
    echo "=== AlphaZero Run Doctor ==="
    python -m orchestrator.doctor --run-dir "$RUN_DIR"
elif [[ "$RESUME_DOCTOR" == "1" ]]; then
    echo "resume doctor requested, but no existing pipeline_state.yaml was found"
    echo "treating $RUN_DIR as a new coordinator run"
fi

python -m orchestrator.coordinator \
    --project-dir "$PROJECT_DIR" \
    --run-dir "$RUN_DIR" \
    $COORDINATOR_ARGS
