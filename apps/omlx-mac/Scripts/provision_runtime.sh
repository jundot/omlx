#!/usr/bin/env bash
# provision_runtime.sh — create the slim app's external Python runtime.
#
# The slim app does NOT embed Python or the ~1 GB MLX layer. This script
# provisions a pinned CPython + the project's locked dependencies into:
#
#   ~/Library/Application Support/oMLX/runtime/venv
#
# The app's PythonRuntime.resolve() prefers that venv. The app bundle still
# carries the small omlx/omlx_web source tree in Contents/Resources.
#
# Python source preference:
#   1. Homebrew python3.11 (/opt/homebrew or /usr/local)
#   2. a pinned uv-managed CPython (default 3.11.18)
#
# Env:
#   OMLX_RUNTIME_DIR    override the runtime directory
#   OMLX_PYTHON_VERSION uv fallback version (default 3.11.18)
#   OMLX_UV             uv executable (default: uv on PATH, then .venv/bin/uv)
#   OMLX_EXTRAS         uv extras (default: cluster)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUNTIME_DIR="${OMLX_RUNTIME_DIR:-$HOME/Library/Application Support/oMLX/runtime}"
VENV="$RUNTIME_DIR/venv"
PYTHON_VERSION="${OMLX_PYTHON_VERSION:-$(cat "$ROOT/.python-version" 2>/dev/null || echo 3.12.16)}"
EXTRAS="${OMLX_EXTRAS:-cluster}"

die() { echo "provision_runtime: $*" >&2; exit 1; }

UV="${OMLX_UV:-}"
if [ -z "$UV" ]; then
    UV="$(command -v uv || true)"
fi
if [ -z "$UV" ] && [ -x "$ROOT/.venv/bin/uv" ]; then
    UV="$ROOT/.venv/bin/uv"
fi
[ -n "$UV" ] || die "uv not found; install uv or pass OMLX_UV=/path/to/uv"

PYTHON=""
for candidate in \
    "/opt/homebrew/bin/python$PYTHON_VERSION" \
    "/usr/local/bin/python$PYTHON_VERSION" \
    "$(command -v "python$PYTHON_VERSION" || true)"
do
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
        PYTHON="$candidate"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "provision_runtime: no Homebrew python3.11; installing pinned $PYTHON_VERSION via uv" >&2
    "$UV" python install "$PYTHON_VERSION"
    PYTHON="$("$UV" python find "$PYTHON_VERSION")"
fi

echo "provision_runtime: python: $PYTHON"
mkdir -p "$RUNTIME_DIR"
"$UV" venv --clear --python "$PYTHON" "$VENV"

# The app ships omlx/omlx_web as source on PYTHONPATH; install the third-party
# dependencies from the project's lock so the runtime is reproducible.
echo "provision_runtime: installing locked dependencies ($EXTRAS)"
UV_PROJECT_ENVIRONMENT="$VENV" "$UV" sync --frozen --no-dev --no-install-project \
    --extra "$EXTRAS" --python "$PYTHON"

"$VENV/bin/python" - <<'PY'
import mlx
print("provision_runtime: dependency imports ok; mlx", getattr(mlx, "__version__", "?"))
PY

cat <<EOF
Runtime ready: $VENV
The app will use it automatically. If it does not, launch with:
  OMLX_PYTHON_OVERRIDE=$VENV/bin/python3
EOF
