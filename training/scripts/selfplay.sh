#!/bin/bash
# Slurm job script for AlphaZero self-play
# USE THE SUBMIT WRAPPER: ./submit_selfplay.sh
#
# Expected environment variables (set by submit_selfplay.sh):
#   SELFPLAY_ARGS - arguments for the self-play binary

set -euo pipefail

cd /home/willzhao/alphazero

module load miniforge
eval "$(conda shell.bash hook)"
conda activate /home/willzhao/alphazero/.conda/env
pip install -q -r /home/willzhao/alphazero/requirements.txt

echo "=== AlphaZero Self-Play ==="
echo "  slurm job id: $SLURM_JOB_ID"
echo "  node: $(hostname)"
echo "  selfplay args: $SELFPLAY_ARGS"
echo "  cuda visible devices: ${CUDA_VISIBLE_DEVICES:-not set}"

# The self-play binary is a compiled Rust program
export LD_LIBRARY_PATH="/home/willzhao/alphazero/.conda/env/lib/python3.13/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

./target/release/self-play $SELFPLAY_ARGS
