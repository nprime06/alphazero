#!/bin/bash
# Print a concise ORCD AlphaZero campaign status snapshot.
#
# Safe for login nodes: this script only reads git state, Slurm status,
# run metadata, doctor output, artifact counts, and recent logs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
STATUS_FILE="${STATUS_FILE:-${PROJECT_DIR}/runs/slurm_setup/latest_orcd_jobs.txt}"
RUN_DIR=""
LOG_LINES=80
PYTHON_BIN="${PYTHON_BIN:-}"

usage() {
    cat <<'EOF'
usage: training/scripts/orcd_status.sh [--run-dir DIR] [--status-file FILE] [--log-lines N]

Print queue, latest coordinator submission, doctor, artifact, and log status.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-dir)
            RUN_DIR="$2"
            shift 2
            ;;
        --status-file)
            STATUS_FILE="$2"
            shift 2
            ;;
        --log-lines)
            LOG_LINES="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument '$1'" >&2
            usage >&2
            exit 2
            ;;
    esac
done

count_files() {
    local dir="$1"
    local pattern="$2"
    if [[ ! -d "$dir" ]]; then
        echo 0
        return
    fi
    find "$dir" -type f -name "$pattern" 2>/dev/null | wc -l | tr -d ' '
}

status_value() {
    local key="$1"
    local file="$2"
    if [[ ! -f "$file" ]]; then
        return 0
    fi
    awk -F= -v wanted="$key" '$1 == wanted {print substr($0, length($1) + 2)}' "$file" | tail -n 1
}

section() {
    printf '\n== %s ==\n' "$1"
}

cd "$PROJECT_DIR"

if [[ -z "$PYTHON_BIN" ]]; then
    if [[ -x "$PROJECT_DIR/.conda/env/bin/python" ]]; then
        PYTHON_BIN="$PROJECT_DIR/.conda/env/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python)"
    fi
fi
export PYTHONPATH="$PROJECT_DIR/neural:$PROJECT_DIR/training:$PROJECT_DIR/orchestrator:$PROJECT_DIR/alphazero:${PYTHONPATH:-}"

section "host"
hostname || true
date -Iseconds || date || true

section "git"
git rev-parse --short HEAD 2>/dev/null || echo "unknown"
git status --short 2>/dev/null || true

section "squeue"
if command -v squeue >/dev/null 2>&1; then
    squeue --me || true
else
    echo "squeue not found"
fi

section "latest submission"
if [[ -f "$STATUS_FILE" ]]; then
    cat "$STATUS_FILE"
else
    echo "missing: $STATUS_FILE"
fi

if [[ -z "$RUN_DIR" ]]; then
    RUN_DIR="$(status_value cluster_pilot_run_dir "$STATUS_FILE")"
fi
JOB_ID="$(status_value cluster_pilot_job "$STATUS_FILE")"

if [[ -n "${JOB_ID:-}" ]]; then
    section "sacct latest job"
    if command -v sacct >/dev/null 2>&1; then
        sacct -j "$JOB_ID" --format=JobID,JobName%24,Partition,State,ExitCode,Elapsed,Start,End -P || true
    else
        echo "sacct not found"
    fi
fi

if [[ -z "${RUN_DIR:-}" ]]; then
    section "run"
    echo "no run directory found in status file"
    exit 0
fi

section "run"
echo "$RUN_DIR"
if [[ ! -d "$RUN_DIR" ]]; then
    echo "missing run directory"
    exit 0
fi

section "doctor"
if [[ -z "$PYTHON_BIN" ]]; then
    echo "doctor skipped: no Python interpreter found"
elif "$PYTHON_BIN" -m orchestrator.doctor --run-dir "$RUN_DIR"; then
    echo "doctor_exit=0"
else
    echo "doctor_exit=$?"
fi

section "pipeline state"
if [[ -f "$RUN_DIR/pipeline_state.yaml" ]]; then
    cat "$RUN_DIR/pipeline_state.yaml"
else
    echo "missing pipeline_state.yaml"
fi

section "artifact counts"
echo "data_msgpack=$(count_files "$RUN_DIR/data" '*.msgpack')"
echo "weights=$(count_files "$RUN_DIR/weights" 'model_v*.pt')"
echo "checkpoints=$(count_files "$RUN_DIR/checkpoints" '*')"
if [[ -f "$RUN_DIR/promotion_ledger.jsonl" ]]; then
    echo "promotion_ledger_entries=$(grep -cve '^[[:space:]]*$' "$RUN_DIR/promotion_ledger.jsonl" || true)"
else
    echo "promotion_ledger_entries=0"
fi

section "recent logs"
recent_logs="$(ls -t "$RUN_DIR"/slurm-*.log "$RUN_DIR"/slurm-*.err 2>/dev/null | head -n 4 || true)"
if [[ -z "$recent_logs" ]]; then
    echo "no slurm logs found"
else
    while IFS= read -r log_file; do
        [[ -n "$log_file" ]] || continue
        echo "--- $log_file ---"
        tail -n "$LOG_LINES" "$log_file" || true
    done <<< "$recent_logs"
fi
