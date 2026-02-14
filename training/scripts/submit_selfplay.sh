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
NUM_CPUS=10
MEM=64
TIME="6:00:00"
SELFPLAY_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            NUM_GPUS="$2"
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

RUN_DIR="${PROJECT_DIR}/runs/selfplay_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

echo "=== Submitting AlphaZero Self-Play ==="
echo "  run dir: $RUN_DIR"
echo "  selfplay args: $SELFPLAY_ARGS"

sbatch \
    --job-name=az-selfplay \
    --partition=mit_normal_gpu \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=$NUM_CPUS \
    --mem=${MEM}G \
    --gres=gpu:h200:$NUM_GPUS \
    --time=$TIME \
    --output="${RUN_DIR}/slurm-%j.log" \
    --error="${RUN_DIR}/slurm-%j.err" \
    --export=ALL,SELFPLAY_ARGS="$SELFPLAY_ARGS" \
    "${SCRIPT_DIR}/selfplay.sh"

echo "submitted! logs in $RUN_DIR"
