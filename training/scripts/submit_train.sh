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
#   --partition NAME  Slurm partition (default: mit_normal_gpu)
#   --gpu-type NAME   Slurm GPU type for --gres (default: h200)
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
PARTITION="mit_normal_gpu"
GPU_TYPE="h200"
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
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --gpu-type)
            GPU_TYPE="$2"
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

if [[ "$NUM_GPUS" -ne 1 ]]; then
    echo "ERROR: training.train is currently single-process; DDP is not wired yet." >&2
    echo "Use --gpus 1 until distributed training is implemented." >&2
    exit 2
fi

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
echo "  partition: $PARTITION"
echo "  gpu type: $GPU_TYPE"
echo "  run dir: $RUN_DIR"
echo "  train args: $TRAIN_ARGS"

sbatch \
    --job-name=az-train \
    --partition="$PARTITION" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=$NUM_CPUS \
    --mem=${TOTAL_MEM}G \
    --gres=gpu:${GPU_TYPE}:$NUM_GPUS \
    --time=$TIME \
    --output="${RUN_DIR}/slurm-%j.log" \
    --error="${RUN_DIR}/slurm-%j.err" \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",NUM_GPUS=$NUM_GPUS,RUN_DIR="$RUN_DIR",TRAIN_ARGS="$TRAIN_ARGS" \
    "${SCRIPT_DIR}/train.sh"

echo "submitted! logs in $RUN_DIR"
