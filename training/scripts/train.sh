#!/bin/bash
# Slurm job script for AlphaZero training
# USE THE SUBMIT WRAPPER: ./submit_train.sh
#
# Expected environment variables (set by submit_train.sh):
#   NUM_GPUS   - number of GPUs to use
#   RUN_DIR    - run directory for checkpoints, logs, TensorBoard
#   TRAIN_ARGS - arguments for training.train

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/willzhao/alphazero}"
CONDA_ENV="${CONDA_ENV:-${PROJECT_DIR}/.conda/env}"
NUM_GPUS="${NUM_GPUS:-1}"
RUN_DIR="${RUN_DIR:?RUN_DIR must be set by submit_train.sh}"
TRAIN_ARGS="${TRAIN_ARGS:-}"

cd "$PROJECT_DIR"

module load miniforge
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
python -m pip install -q -r "$PROJECT_DIR/requirements.txt"
python -m pip install -q -e "$PROJECT_DIR/neural" -e "$PROJECT_DIR/training"

if [[ "${NUM_GPUS:-1}" -ne 1 ]]; then
    echo "ERROR: training.train is currently single-process; DDP is not wired yet." >&2
    echo "Use NUM_GPUS=1 until distributed training is implemented." >&2
    exit 2
fi

TORCH_LIB="$(python - <<'PY'
from pathlib import Path
import torch
print(Path(torch.__file__).resolve().parent / "lib")
PY
)"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR/neural:$PROJECT_DIR/training:${PYTHONPATH:-}"
export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"

echo "=== AlphaZero Training ==="
echo "  project dir: $PROJECT_DIR"
echo "  num gpus: $NUM_GPUS"
echo "  run dir: $RUN_DIR"
echo "  train args: $TRAIN_ARGS"
echo "  slurm job id: ${SLURM_JOB_ID:-manual}"
echo "  node: $(hostname)"
echo "  cuda visible devices: ${CUDA_VISIBLE_DEVICES:-not set}"

torchrun --standalone --nproc_per_node=1 \
    -m training.train --run-dir "$RUN_DIR" $TRAIN_ARGS
