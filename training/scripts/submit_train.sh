#!/bin/bash
# Submit AlphaZero training job to Slurm
#
# USAGE:
#   ./submit_train.sh --data-dir /path/to/data
#   ./submit_train.sh --data-dir /path/to/data --gpus 2
#   ./submit_train.sh --data-dir /path/to/data --run-dir /path/to/run  # resume
#   ./submit_train.sh --dummy-data --network tiny --steps 1000          # smoke test
#
# OPTIONS:
#   --gpus N          Number of GPUs (default: 1)
#   --run-dir DIR     Run directory (created if omitted, with timestamp)
#   --time HH:MM:SS   Wall time limit (default: 12:00:00)
#   --cpus-per-gpu N  CPUs per GPU (default: 8)
#   --mem-per-gpu N   Memory per GPU in GB (default: 128)
#   All other args are passed through to training.train

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Defaults
NUM_GPUS=1
CPUS_PER_GPU=8
MEM_PER_GPU=128
TIME="6:00:00"
RUN_DIR=""

# Separate our args from train args
TRAIN_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --run-dir)
            RUN_DIR="$2"
            shift 2
            ;;
        --time)
            TIME="$2"
            shift 2
            ;;
        --cpus-per-gpu)
            CPUS_PER_GPU="$2"
            shift 2
            ;;
        --mem-per-gpu)
            MEM_PER_GPU="$2"
            shift 2
            ;;
        *)
            # Everything else is a train arg
            TRAIN_ARGS="$TRAIN_ARGS $1"
            shift
            ;;
    esac
done

NUM_CPUS=$((NUM_GPUS * CPUS_PER_GPU))
TOTAL_MEM=$((NUM_GPUS * MEM_PER_GPU))

# Create run directory if not resuming
RUNS_DIR="${PROJECT_DIR}/runs"
if [[ -z "$RUN_DIR" ]]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RUN_DIR="${RUNS_DIR}/train_${TIMESTAMP}"
fi
mkdir -p "$RUN_DIR"

echo "=== Submitting AlphaZero Training ==="
echo "  gpus: $NUM_GPUS"
echo "  run dir: $RUN_DIR"
echo "  train args: $TRAIN_ARGS"

sbatch \
    --job-name=az-train \
    --partition=mit_normal_gpu \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=$NUM_CPUS \
    --mem=${TOTAL_MEM}G \
    --gres=gpu:h200:$NUM_GPUS \
    --time=$TIME \
    --output="${RUN_DIR}/slurm-%j.log" \
    --error="${RUN_DIR}/slurm-%j.err" \
    --export=ALL,NUM_GPUS=$NUM_GPUS,RUN_DIR="$RUN_DIR",TRAIN_ARGS="$TRAIN_ARGS" \
    "${SCRIPT_DIR}/train.sh"

echo "submitted! logs in $RUN_DIR"
