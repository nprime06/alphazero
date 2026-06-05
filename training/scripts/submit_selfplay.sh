#!/bin/bash
# Submit AlphaZero self-play job to Slurm
#
# USAGE:
#   ./submit_selfplay.sh --model /path/to/model.pt --games 1000 --output /path/to/data
#   ./submit_selfplay.sh --model /path/to/model.pt --games 1000 --output /path/to/data --gpus 1
#
# Self-play uses 1 GPU for inference + multiple CPU threads for MCTS.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

NUM_GPUS=1
PARTITION="mit_normal_gpu"
GPU_TYPE="h200"
NUM_CPUS=10
MEM=64
TIME="6:00:00"
SELFPLAY_ARGS=""
MAX_TIME_SECONDS=$((6 * 60 * 60))

time_to_seconds() {
    local value="$1"
    local hours minutes seconds

    if [[ "$value" =~ ^([0-9]+):([0-9]{2}):([0-9]{2})$ ]]; then
        hours="${BASH_REMATCH[1]}"
        minutes="${BASH_REMATCH[2]}"
        seconds="${BASH_REMATCH[3]}"
        echo $((10#$hours * 3600 + 10#$minutes * 60 + 10#$seconds))
        return 0
    fi

    echo "ERROR: unsupported --time format '$value'. Use HH:MM:SS." >&2
    return 1
}

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
        --time)
            TIME="$2"
            shift 2
            ;;
        *)
            SELFPLAY_ARGS="$SELFPLAY_ARGS $1"
            shift
            ;;
    esac
done

if [[ "$NUM_GPUS" -ne 1 ]]; then
    echo "ERROR: self-play currently uses one GPU for inference." >&2
    echo "Use --gpus 1 until multi-GPU self-play is implemented." >&2
    exit 2
fi

TIME_SECONDS="$(time_to_seconds "$TIME")"
if [[ "$TIME_SECONDS" -gt "$MAX_TIME_SECONDS" ]]; then
    echo "ERROR: self-play wall time must be <= 6:00:00 for this ORCD account." >&2
    exit 2
fi

RUN_DIR="${PROJECT_DIR}/runs/selfplay_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

echo "=== Submitting AlphaZero Self-Play ==="
echo "  run dir: $RUN_DIR"
echo "  gpus: $NUM_GPUS"
echo "  partition: $PARTITION"
echo "  gpu type: $GPU_TYPE"
echo "  time: $TIME"
echo "  selfplay args: $SELFPLAY_ARGS"

sbatch \
    --job-name=az-selfplay \
    --partition="$PARTITION" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=$NUM_CPUS \
    --mem=${MEM}G \
    --gres=gpu:${GPU_TYPE}:$NUM_GPUS \
    --time=$TIME \
    --output="${RUN_DIR}/slurm-%j.log" \
    --error="${RUN_DIR}/slurm-%j.err" \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",SELFPLAY_ARGS="$SELFPLAY_ARGS" \
    "${SCRIPT_DIR}/selfplay.sh"

echo "submitted! logs in $RUN_DIR"
