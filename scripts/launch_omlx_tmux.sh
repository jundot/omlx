#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${OMLX_TMUX_SESSION:-omlx-tts}"
MODEL_DIR="${OMLX_MODEL_DIR:-$HOME/.omlx/models}"
HOST="${OMLX_HOST:-0.0.0.0}"
PORT="${OMLX_PORT:-8000}"
LOG_LEVEL="${OMLX_LOG_LEVEL:-info}"
API_KEY="${OMLX_API_KEY:-oMLX}"
LOG_FILE="${OMLX_LOG_FILE:-/tmp/omlx-tts.log}"
UV_RUN_ARGS="${OMLX_UV_RUN_ARGS:---extra audio}"

COMMAND="uv run ${UV_RUN_ARGS} omlx serve --model-dir ${MODEL_DIR} --host ${HOST} --port ${PORT} --log-level ${LOG_LEVEL} --api-key ${API_KEY} > ${LOG_FILE} 2>&1"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  tmux kill-session -t "$SESSION_NAME"
fi

tmux new-session -d -s "$SESSION_NAME" "$COMMAND"

echo "Started tmux session: $SESSION_NAME"
echo "Log file: $LOG_FILE"
echo "Command: $COMMAND"
