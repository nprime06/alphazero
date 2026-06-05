#!/bin/bash
# Submit a clean, bounded AlphaZero coordinator smoke to Slurm.
#
# This wrapper intentionally starts a fresh run directory. Use
# submit_coordinator.sh directly for explicit resume/repair work.
#
# USAGE:
#   bash training/scripts/submit_cluster_smoke.sh
#   bash training/scripts/submit_cluster_smoke.sh --time 1:00:00
#   bash training/scripts/submit_cluster_smoke.sh --preflight-only
#
# All supported resource overrides are passed to submit_coordinator.sh.
# Coordinator hyperparameters come from config_cluster_smoke.yaml.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-${PROJECT_DIR}/.conda/env}"
CONFIG_PATH="orchestrator/orchestrator/config_cluster_smoke.yaml"
PREFLIGHT_ONLY=0
RUN_PREFLIGHT=1
PYTHON_BIN="${PYTHON_BIN:-}"

ARGS=(
    --config "$CONFIG_PATH"
    --gpus 1
    --time 2:00:00
    --eval-backend rust
)

find_python() {
    if [[ -n "$PYTHON_BIN" ]]; then
        echo "$PYTHON_BIN"
    elif [[ -x "$PROJECT_DIR/.conda/env/bin/python" ]]; then
        echo "$PROJECT_DIR/.conda/env/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        command -v python3
    elif command -v python >/dev/null 2>&1; then
        command -v python
    fi
}

configured_eval_backend() {
    local backend="rust"
    local i

    for ((i = 0; i < ${#ARGS[@]}; i++)); do
        case "${ARGS[$i]}" in
            --eval-backend)
                if [[ $((i + 1)) -lt ${#ARGS[@]} ]]; then
                    backend="${ARGS[$((i + 1))]}"
                fi
                ;;
            --eval-backend=*)
                backend="${ARGS[$i]#--eval-backend=}"
                ;;
        esac
    done

    echo "$backend"
}

check_rust_eval_loader() {
    local python_bin="$1"
    local torch_lib=""

    torch_lib="$("$python_bin" - <<'PY'
from pathlib import Path
import torch

print(Path(torch.__file__).resolve().parent / "lib")
PY
)"

    PYTHONPATH="$PROJECT_DIR/neural:$PROJECT_DIR/training:$PROJECT_DIR/orchestrator:$PROJECT_DIR/alphazero:${PYTHONPATH:-}" \
        LIBTORCH_USE_PYTORCH=1 \
        LD_LIBRARY_PATH="$CONDA_ENV/lib:$torch_lib:${LD_LIBRARY_PATH:-}" \
        "$python_bin" - <<'PY'
from orchestrator.evaluate import _load_alphazero_py

_load_alphazero_py()
PY
}

run_preflight() {
    echo "=== AlphaZero Cluster Smoke Preflight ==="
    echo "  project dir: $PROJECT_DIR"
    echo "  config: $CONFIG_PATH"

    if ! command -v sbatch >/dev/null 2>&1; then
        echo "ERROR: sbatch not found. Run this on an ORCD login node." >&2
        exit 2
    fi
    if ! command -v squeue >/dev/null 2>&1; then
        echo "ERROR: squeue not found. Run this on an ORCD login node." >&2
        exit 2
    fi
    if [[ ! -f "$PROJECT_DIR/$CONFIG_PATH" ]]; then
        echo "ERROR: missing smoke config: $PROJECT_DIR/$CONFIG_PATH" >&2
        exit 2
    fi
    if [[ ! -x "$PROJECT_DIR/target/release/self-play" ]]; then
        echo "ERROR: missing $PROJECT_DIR/target/release/self-play" >&2
        echo "Build it first with: LIBTORCH_USE_PYTORCH=1 cargo build --release -p self-play" >&2
        exit 2
    fi
    if [[ "$(configured_eval_backend)" == "rust" ]]; then
        python_bin="$(find_python)"
        if [[ -z "$python_bin" ]]; then
            echo "ERROR: no Python interpreter found for Rust evaluation preflight." >&2
            exit 2
        fi
        if ! check_rust_eval_loader "$python_bin"; then
            echo "ERROR: Rust evaluation loader failed. Rebuild with:" >&2
            echo "  LIBTORCH_USE_PYTORCH=1 $python_bin -m maturin develop --manifest-path alphazero-py/Cargo.toml" >&2
            exit 2
        fi
    fi

    existing_alpha_jobs="$(
        squeue --me -h -o '%i|%j|%T' 2>/dev/null \
            | awk -F'|' '$2 ~ /^az-(coord|selfplay|train)$/ {print}'
    )"
    if [[ -n "$existing_alpha_jobs" ]]; then
        echo "ERROR: existing AlphaZero Slurm jobs found:" >&2
        echo "$existing_alpha_jobs" >&2
        exit 2
    fi

    echo "preflight ok"
}

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
        --preflight-only)
            PREFLIGHT_ONLY=1
            shift
            ;;
        --skip-preflight)
            RUN_PREFLIGHT=0
            shift
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ "$RUN_PREFLIGHT" == "1" ]]; then
    run_preflight
fi
if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
    exit 0
fi

exec bash "$SCRIPT_DIR/submit_coordinator.sh" "${ARGS[@]}"
