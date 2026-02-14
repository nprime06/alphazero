#!/bin/bash
# Slurm job script for AlphaZero training
# USE THE SUBMIT WRAPPER: ./submit_train.sh
#
# Expected environment variables (set by submit_train.sh):
#   NUM_GPUS   - number of GPUs to use
#   RUN_DIR    - run directory for checkpoints, logs, TensorBoard
#   TRAIN_ARGS - arguments for training.train

set -euo pipefail

cd /home/willzhao/alphazero

module load miniforge
eval "$(conda shell.bash hook)"
conda activate /home/willzhao/alphazero/.conda/env
pip install -q -r /home/willzhao/alphazero/requirements.txt

export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/willzhao/alphazero/neural:/home/willzhao/alphazero/training:${PYTHONPATH:-}"

echo "=== AlphaZero Training ==="
echo "  num gpus: $NUM_GPUS"
echo "  run dir: $RUN_DIR"
echo "  train args: $TRAIN_ARGS"
echo "  slurm job id: $SLURM_JOB_ID"
echo "  node: $(hostname)"
echo "  cuda visible devices: ${CUDA_VISIBLE_DEVICES:-not set}"

torchrun --standalone --nproc_per_node=$NUM_GPUS \
    -m training.train --run-dir "$RUN_DIR" $TRAIN_ARGS
