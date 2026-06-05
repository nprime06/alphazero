#!/bin/bash
# Submit AlphaZero pipeline coordinator to Slurm
#
# USAGE:
#   ./submit_coordinator.sh --config orchestrator/orchestrator/config.yaml
#   ./submit_coordinator.sh --config config.yaml --network tiny --iterations 5
#   ./submit_coordinator.sh --run-dir runs/coord_20250212_143000   # resume
#   ./submit_coordinator.sh --config config.yaml --time 48:00:00
#
# The coordinator runs the full self-play -> train -> evaluate -> promote
# loop. Each run is self-contained in its own directory under runs/.
#
# OPTIONS (for this wrapper):
#   --gpus N          Number of GPUs (default: 1)
#   --partition NAME  Slurm partition (default: mit_normal_gpu)
#   --gpu-type NAME   Slurm GPU type for --gres (default: h200)
#   --cpus N          Number of CPUs (default: 10)
#   --mem N           Memory in GB (default: 128)
#   --time HH:MM:SS   Wall time limit (default: 24:00:00)
#   --run-dir DIR     Resume an existing run directory
#   --skip-resume-doctor
#                    Do not run run-integrity checks before resuming
#   All other args are passed through to the coordinator

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Defaults
NUM_GPUS=1
PARTITION="mit_normal_gpu"
GPU_TYPE="h200"
NUM_CPUS=16
MEM=128
TIME="6:00:00"
RUN_DIR=""
COORDINATOR_ARGS=""
RESUME_DOCTOR=0

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
        --cpus)
            NUM_CPUS="$2"
            shift 2
            ;;
        --mem)
            MEM="$2"
            shift 2
            ;;
        --time)
            TIME="$2"
            shift 2
            ;;
        --run-dir)
            RUN_DIR="$2"
            RESUME_DOCTOR=1
            shift 2
            ;;
        --skip-resume-doctor)
            RESUME_DOCTOR=0
            shift
            ;;
        *)
            # Everything else is a coordinator arg
            COORDINATOR_ARGS="$COORDINATOR_ARGS $1"
            shift
            ;;
    esac
done

if [[ "$NUM_GPUS" -ne 1 ]]; then
    echo "ERROR: the coordinator currently runs single-process training; DDP is not wired yet." >&2
    echo "Use --gpus 1 until distributed training is implemented." >&2
    exit 2
fi

# Create run directory if not resuming
if [[ -z "$RUN_DIR" ]]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RUN_DIR="${PROJECT_DIR}/runs/coord_${TIMESTAMP}"
fi
mkdir -p "$RUN_DIR"

echo "=== Submitting AlphaZero Pipeline Coordinator ==="
echo "  run dir: $RUN_DIR"
echo "  gpus: $NUM_GPUS"
echo "  partition: $PARTITION"
echo "  gpu type: $GPU_TYPE"
echo "  time: $TIME"
echo "  resume doctor: $RESUME_DOCTOR"
echo "  coordinator args: $COORDINATOR_ARGS"

sbatch \
    --job-name=az-coord \
    --partition="$PARTITION" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=$NUM_CPUS \
    --mem=${MEM}G \
    --gres=gpu:${GPU_TYPE}:$NUM_GPUS \
    --time=$TIME \
    --output="${RUN_DIR}/slurm-%j.log" \
    --error="${RUN_DIR}/slurm-%j.err" \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",RUN_DIR="$RUN_DIR",COORDINATOR_ARGS="$COORDINATOR_ARGS",RESUME_DOCTOR="$RESUME_DOCTOR" \
    "${SCRIPT_DIR}/coordinator.sh"

echo "submitted! logs in $RUN_DIR"
