#!/usr/bin/env bash
# Launch (or re-attach info for) an I-JEPA training run in a detached tmux
# session so it keeps running after you disconnect.
#
# Usage:
#   scripts/launch_training.sh [session-name] [config-path]
#
# Defaults to session "ijepa-vitb16" and configs/in1k_vitb16_ep600.yaml.
# Reattach anytime with: tmux attach -t <session-name>
set -uo pipefail

SESSION="${1:-ijepa-vitb16}"
CONFIG="${2:-configs/in1k_vitb16_ep600.yaml}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_DIR/experiments/$(basename "$CONFIG" .yaml)"

mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists. Attach with: tmux attach -t $SESSION"
    exit 0
fi

tmux new-session -d -s "$SESSION" -c "$REPO_DIR" \
    "bash '$REPO_DIR/scripts/_run_with_retry.sh' '$CONFIG' '$LOG_DIR'"

echo "started tmux session '$SESSION' running $CONFIG"
echo "attach with: tmux attach -t $SESSION"
echo "logs at: $LOG_DIR/stdout.log (and $LOG_DIR/supervisor.log for restarts)"
