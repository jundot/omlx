#!/usr/bin/env bash
# Build a release oMLX.app bundle with all optional native custom kernels.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEPLOYMENT_TARGET="${OMLX_CUSTOM_KERNEL_DEPLOYMENT_TARGET:-15.0}"
APP_BUILD_SCRIPT="$REPO_ROOT/apps/omlx-mac/Scripts/build.sh"
OUTPUT_DIR="${OMLX_NEXT_OUT:-$REPO_ROOT/apps/omlx-mac/build/Stage}"
STAGED_APP="$OUTPUT_DIR/oMLX.app"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if ! command -v uv >/dev/null 2>&1; then
        echo "error: uv is required to prepare the isolated Python 3.11 build environment." >&2
        exit 1
    fi

    BUILD_ENV="$REPO_ROOT/build/app-kernel-python"
    if [[ ! -x "$BUILD_ENV/bin/python" ]]; then
        echo "Creating isolated Python 3.11 kernel build environment..."
        uv venv --python 3.11 "$BUILD_ENV"
    fi
    uv pip install --quiet --python "$BUILD_ENV/bin/python" \
        "pip>=24" "cmake>=3.27" "nanobind==2.13.0" "setuptools>=61" wheel
    PYTHON_BIN="$BUILD_ENV/bin/python"
fi

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "error: oMLX custom kernels require macOS on Apple Silicon." >&2
    exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "error: Python was not found at $PYTHON_BIN" >&2
    echo "Set PYTHON_BIN to a Python 3.11 executable." >&2
    exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys

if sys.version_info[:2] != (3, 11):
    raise SystemExit(
        f"error: the app custom kernels must match its Python 3.11 runtime; "
        f"found {sys.version.split()[0]}"
    )
PY

if ! xcrun --find metal >/dev/null 2>&1; then
    echo "error: the Metal compiler was not found. Install full Xcode and select it with xcode-select." >&2
    exit 1
fi

if [[ ! -x "$APP_BUILD_SCRIPT" ]]; then
    echo "error: app build script is missing or not executable: $APP_BUILD_SCRIPT" >&2
    exit 1
fi

echo "Building oMLX.app with custom kernels using $PYTHON_BIN (macOS target $DEPLOYMENT_TARGET)..."
PATH="$(dirname "$PYTHON_BIN"):$PATH" \
PYTHON_BIN="$PYTHON_BIN" \
OMLX_CUSTOM_KERNEL_DEPLOYMENT_TARGET="$DEPLOYMENT_TARGET" \
    "$APP_BUILD_SCRIPT" release --with-custom-kernel "$@"

if [[ ! -d "$STAGED_APP" ]]; then
    echo "error: the build completed without producing $STAGED_APP" >&2
    exit 1
fi

echo "Verifying bundled native kernels..."
"$STAGED_APP/Contents/MacOS/omlx-cluster-python" - <<'PY'
from omlx.custom_kernels import native_kernel_status

status = native_kernel_status()
for name, result in status.items():
    state = "available" if result["available"] else "UNAVAILABLE"
    print(f"  {name}: {state}")
    if result["import_error"]:
        print(f"    {result['import_error']}")

missing = [name for name, result in status.items() if not result["available"]]
if missing:
    raise SystemExit(f"error: native kernel verification failed: {', '.join(missing)}")
PY

codesign --verify --deep --strict "$STAGED_APP"

# apps/omlx-mac/Scripts/build.sh intentionally builds the bundle against its
# Python 3.11 donor and cleans source-tree native artifacts first. Rebuild for
# the repository virtualenv afterwards so running `omlx` from this checkout
# does not see a 3.11-only extension and silently fall back to Python kernels.
DEV_PYTHON_BIN="${OMLX_DEV_PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
if [[ -x "$DEV_PYTHON_BIN" && "$DEV_PYTHON_BIN" != "$PYTHON_BIN" ]]; then
    echo "Rebuilding custom kernels for the source virtualenv ($DEV_PYTHON_BIN)..."
    PATH="$(dirname "$PYTHON_BIN"):$PATH" \
    OMLX_CUSTOM_KERNEL_DEPLOYMENT_TARGET="$DEPLOYMENT_TARGET" \
    MACOSX_DEPLOYMENT_TARGET="$DEPLOYMENT_TARGET" \
    CMAKE_ARGS="${CMAKE_ARGS:-} -DCMAKE_OSX_DEPLOYMENT_TARGET=$DEPLOYMENT_TARGET" \
        "$DEV_PYTHON_BIN" setup.py build_ext --inplace --force --with-custom-kernel
    "$DEV_PYTHON_BIN" - <<'PY'
from omlx.custom_kernels import native_kernel_status

status = native_kernel_status()
missing = [name for name, result in status.items() if not result["available"]]
if missing:
    raise SystemExit(
        "error: source-runtime native kernel verification failed: "
        + ", ".join(missing)
    )
print("Source-runtime native kernels verified.")
PY
fi

echo "oMLX.app and all custom kernels built successfully:"
echo "  $STAGED_APP"
