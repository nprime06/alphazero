#!/bin/bash
# Submit a clean, bounded AlphaZero coordinator smoke to Slurm.
#
# This wrapper intentionally starts a fresh run directory. Use
# submit_coordinator.sh directly for explicit resume/repair work.
#
# USAGE:
#   bash training/scripts/submit_cluster_smoke.sh
#   bash training/scripts/submit_cluster_smoke.sh --time 1:00:00
#
# All supported resource overrides are passed to submit_coordinator.sh.
# Coordinator hyperparameters come from config_cluster_smoke.yaml.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="orchestrator/orchestrator/config_cluster_smoke.yaml"

ARGS=(
    --config "$CONFIG_PATH"
    --gpus 1
    --time 2:00:00
    --eval-backend rust
)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-dir|--skip-resume-doctor)
            echo "ERROR: submit_cluster_smoke.sh always starts a fresh run." >&2
            echo "Use submit_coordinator.sh directly for explicit resume/repair work." >&2
            exit 2
            ;;
        --gpus)
            if [[ "${2:-}" != "1" ]]; then
                echo "ERROR: cluster smoke is intentionally single-GPU." >&2
                exit 2
            fi
            ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

exec bash "$SCRIPT_DIR/submit_coordinator.sh" "${ARGS[@]}"
