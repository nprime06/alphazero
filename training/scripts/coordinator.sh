#!/bin/bash
# Slurm job script for AlphaZero pipeline coordinator
# USE THE SUBMIT WRAPPER: ./submit_coordinator.sh
#
# Expected environment variables (set by submit_coordinator.sh):
#   RUN_DIR          - self-contained run directory
#   COORDINATOR_ARGS - arguments for orchestrator.coordinator

set -euo pipefail

cd /home/willzhao/alphazero

module load miniforge
eval "$(conda shell.bash hook)"
conda activate /home/willzhao/alphazero/.conda/env
pip install -q -r /home/willzhao/alphazero/requirements.txt

export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/willzhao/alphazero/neural:/home/willzhao/alphazero/training:/home/willzhao/alphazero/orchestrator:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/home/willzhao/alphazero/.conda/env/lib/python3.13/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

echo "=== AlphaZero Pipeline Coordinator ==="
echo "  slurm job id: $SLURM_JOB_ID"
echo "  node: $(hostname)"
echo "  run dir: $RUN_DIR"
echo "  coordinator args: ${COORDINATOR_ARGS:-}"
echo "  cuda visible devices: ${CUDA_VISIBLE_DEVICES:-not set}"

python -m orchestrator.coordinator --run-dir "$RUN_DIR" ${COORDINATOR_ARGS:-}
