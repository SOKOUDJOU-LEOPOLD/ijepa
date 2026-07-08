#!/usr/bin/env bash
# Runs I-JEPA training to completion, auto-restarting on crash.
# Relies on meta.load_checkpoint: true in the config so each restart resumes
# from the last saved epoch (src/helper.py::load_checkpoint falls back to
# epoch 0 cleanly if no checkpoint exists yet).
set -uo pipefail

CONFIG="$1"
LOG_DIR="$2"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$REPO_DIR"
source .venv/bin/activate
mkdir -p "$LOG_DIR"

ATTEMPT=0
while true; do
    ATTEMPT=$((ATTEMPT + 1))
    echo "=== launch attempt $ATTEMPT: $(date -Is) ===" | tee -a "$LOG_DIR/supervisor.log"
    python main.py --fname "$CONFIG" --devices cuda:0 cuda:1 cuda:2 cuda:3 2>&1 | tee -a "$LOG_DIR/stdout.log"
    EXIT_CODE=${PIPESTATUS[0]}
    echo "=== attempt $ATTEMPT exited with code $EXIT_CODE: $(date -Is) ===" | tee -a "$LOG_DIR/supervisor.log"

    if [ "$EXIT_CODE" -eq 0 ]; then
        echo "training finished successfully, stopping retry loop" | tee -a "$LOG_DIR/supervisor.log"
        break
    fi

    echo "restarting in 30s (will resume from latest checkpoint)..." | tee -a "$LOG_DIR/supervisor.log"
    sleep 30
done
