#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OMLX_BIN="$REPO_ROOT/.venv/bin/omlx"
BASE_PATH="${OMLX_BASE_PATH:-$HOME/.omlx}"

if [ ! -x "$OMLX_BIN" ]; then
    echo "error: missing executable $OMLX_BIN" >&2
    echo "Run the source install first: uv pip install --python .venv/bin/python -e ." >&2
    exit 1
fi

if [ ! -f "$BASE_PATH/settings.json" ]; then
    echo "error: missing app settings at $BASE_PATH/settings.json" >&2
    echo "Launch the app once or set OMLX_BASE_PATH to an existing oMLX data directory." >&2
    exit 1
fi

cd "$REPO_ROOT"

# macOS malloc's default large-allocation behavior can keep hundreds of MB of
# empty pages resident after repeated MLX model load/unload cycles, especially
# for bge-reranker-v2-m3. Space-efficient mode keeps those pages from
# accumulating while preserving normal model memory reclamation.
export MallocSpaceEfficient="${MallocSpaceEfficient:-1}"

exec "$OMLX_BIN" serve --base-path "$BASE_PATH" "$@"
