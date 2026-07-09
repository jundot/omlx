#!/usr/bin/env bash
# build.sh — produce a runnable oMLX.app for local manual testing.
#
# Side-by-side Swift bundle path: this builds `oMLX.app` alongside the
# legacy Python/PyObjC `oMLX.app` until the Swift app becomes the primary
# release channel.
# Pipeline:
#   1. xcodebuild with `-resolvePackageDependencies` so SPM deps pick up
#      any new minor/patch within the pin range each build
#   2. (auto) rebuild the venvstacks export if sources have drifted
#   3. stage the Swift `.app` and copy the venvstacks-produced Python
#      layers into Contents/Resources/Python/ verbatim
#   4. embed the omlx package from the worktree and ad-hoc sign
#
# venvstacks is the single source of truth for the bundle's Python
# environment — we no longer post-process its output with a uv-sync
# overlay. If a dep needs to move, edit packaging/venvstacks.toml (or
# bump pyproject.toml + uv.lock for the project itself) and let the
# fingerprint check trigger a fresh venvstacks rebuild.
#
# Donor source resolution (in order):
#   1. $OMLX_DONOR_APP — explicit override (e.g. /Applications/oMLX.app).
#                        Bypasses venvstacks rebuild; uses the override as-is.
#   2. packaging/_export/ — the venvstacks export tree. Default for dev
#                           builds. Rebuilt automatically when stale
#                           (fingerprint of pyproject.toml + venvstacks.toml
#                           + uv.lock differs from packaging/_export/.fingerprint).
#   3. /Applications/oMLX.app — last-resort fallback when --no-rebuild-donor
#                               is set and no local export exists.
#
# Usage:
#   apps/omlx-mac/Scripts/build.sh                    # Release, auto-rebuild donor when stale
#   apps/omlx-mac/Scripts/build.sh debug              # Debug build instead
#   apps/omlx-mac/Scripts/build.sh swift              # rebuild Swift app only; reuse existing _export/
#   apps/omlx-mac/Scripts/build.sh swift debug        # Debug Swift app rebuild; reuse existing _export/
#   apps/omlx-mac/Scripts/build.sh swift-fast         # like swift, but skip embedded native signing
#   apps/omlx-mac/Scripts/build.sh release --bare     # skip Python embed
#                                                       (no server, just the
#                                                       AppView shell)
#   apps/omlx-mac/Scripts/build.sh release --with-custom-kernel
#                                                     # build and bundle optional
#                                                     # native custom kernels
#   apps/omlx-mac/Scripts/build.sh --rebuild-donor    # force venvstacks rebuild
#   apps/omlx-mac/Scripts/build.sh --no-rebuild-donor # never rebuild; use
#                                                       existing donor even if stale
#
# Env overrides:
#   OMLX_DONOR_APP=/path/to/oMLX.app    # explicit donor (bypasses venvstacks)
#   OMLX_EXPORT_DIR=/path/to/_export    # override the venvstacks export tree
#                                       (release builds set this per macOS target)
#   OMLX_NEXT_OUT=/path/to/output_dir   # final stage location
#   PYTHON_BIN=/path/to/python3         # python used for venvstacks driver
#                                       (default: PATH lookup of python3)
#   OMLX_WITH_CUSTOM_KERNEL=1           # same as --with-custom-kernel
#   OMLX_CUSTOM_KERNEL_DEPLOYMENT_TARGET=15.0
#                                       # macOS min version for custom kernels

set -euo pipefail

SWIFT_REBUILD=0
SKIP_EMBEDDED_SIGN=0
case "$(echo "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    swift)
        SWIFT_REBUILD=1
        shift
        ;;
    swift-fast)
        SWIFT_REBUILD=1
        SKIP_EMBEDDED_SIGN=1
        shift
        ;;
esac

CONFIG=Release
if [ $# -gt 0 ]; then
    case "$(echo "$1" | tr '[:upper:]' '[:lower:]')" in
        debug)   CONFIG=Debug; shift ;;
        release) CONFIG=Release; shift ;;
        --*) ;;
        *)
            echo "error: unknown configuration '$1' (expected swift|swift-fast|debug|release)" >&2
            exit 2
            ;;
    esac
fi

BARE=0
WITH_CUSTOM_KERNEL="${OMLX_WITH_CUSTOM_KERNEL:-0}"
REBUILD_DONOR=auto    # auto | force | never
for arg in "$@"; do
    case "$arg" in
        --bare) BARE=1 ;;
        --with-custom-kernel) WITH_CUSTOM_KERNEL=1 ;;
        --rebuild-donor) REBUILD_DONOR=force ;;
        --no-rebuild-donor) REBUILD_DONOR=never ;;
        *) echo "error: unknown flag '$arg'" >&2; exit 2 ;;
    esac
done
case "$(echo "$WITH_CUSTOM_KERNEL" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) WITH_CUSTOM_KERNEL=1 ;;
    *) WITH_CUSTOM_KERNEL=0 ;;
esac
if [ "$SWIFT_REBUILD" -eq 1 ] && [ "$REBUILD_DONOR" = "force" ]; then
    echo "error: swift cannot be combined with --rebuild-donor; use release --rebuild-donor." >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"
PACKAGING_DIR="$REPO_ROOT/packaging"
CUSTOM_KERNEL_DIRS=(
    "$REPO_ROOT/omlx/custom_kernels/glm_moe_dsa"
    "$REPO_ROOT/omlx/custom_kernels/minimax_m3"
    "$REPO_ROOT/omlx/custom_kernels/qwen35_prefill"
)
# OMLX_EXPORT_DIR overrides the venvstacks export tree we copy Python
# layers from. Release builds use this to point at a per-target export
# copy with platform-specific mlx-metal wheels swapped in.
LOCAL_EXPORT="${OMLX_EXPORT_DIR:-$PACKAGING_DIR/_export}"

# OMLX_DONOR_APP is "explicit" only when the user set it; the default
# (/Applications/oMLX.app) is treated as a fallback, not an override.
OMLX_DONOR_APP_SET="${OMLX_DONOR_APP+1}"
OMLX_DONOR_APP="${OMLX_DONOR_APP:-/Applications/oMLX.app}"
OUTPUT_DIR="${OMLX_NEXT_OUT:-$PROJECT_DIR/build/Stage}"
BUILD_DIR="$PROJECT_DIR/build"

LIGHT_BLUE="\033[1;34m"
GREEN="\033[1;32m"
YELLOW="\033[1;33m"
RED="\033[1;31m"
RESET="\033[0m"

log()  { printf "${LIGHT_BLUE}[build.sh]${RESET} %s\n" "$*"; }
ok()   { printf "${GREEN}[build.sh]${RESET} %s\n" "$*"; }
warn() { printf "${YELLOW}[build.sh]${RESET} %s\n" "$*"; }
die()  { printf "${RED}[build.sh ERROR]${RESET} %s\n" "$*" >&2; exit 1; }

_is_mach_o_file() {
    local path="$1"
    local type
    type="$(file -b "$path" 2>/dev/null || true)"
    [[ "$type" == *"Mach-O"* ]]
}

_sign_embedded_mach_o_files() {
    local root="$1"
    local count=0

    while IFS= read -r -d '' path; do
        _is_mach_o_file "$path" || continue
        codesign --force --sign - "$path" >/dev/null 2>&1
        count=$((count + 1))
    done < <(
        find "$root" \
            \( -path "*/.dSYM/*" -o -path "*/__pycache__/*" \) -prune -o \
            -type f \( \
                -name "*.so" -o \
                -name "*.dylib" -o \
                -name "*.bundle" -o \
                -perm -100 -o \
                -perm -010 -o \
                -perm -001 \
            \) -print0
    )

    ok "  + signed $count embedded Mach-O files"
}

_app_python_layers_path() {
    local app="$1"
    for candidate in \
        "$app/Contents/Resources/Python" \
        "$app/Contents/Python" \
        "$app/Contents/Frameworks"
    do
        if [ -d "$candidate/cpython-3.11" ] && [ -d "$candidate/framework-mlx-base" ]; then
            printf "%s\n" "$candidate"
            return 0
        fi
    done
    return 1
}

if [ "$SWIFT_REBUILD" -eq 1 ] && [ "$BARE" -eq 0 ] && [ ! -d "$LOCAL_EXPORT" ]; then
    die "Swift rebuild requires existing venvstacks export at $LOCAL_EXPORT. Run build.sh release first."
fi

# --- Resolve donor: pick a layer source and (re)build via venvstacks if stale

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || true)}"

# Returns 0 if the local _export/ exists and its stored fingerprint matches
# the current pyproject/venvstacks/lockfile state.
_local_export_fresh() {
    [ -d "$LOCAL_EXPORT" ] || return 1
    [ -f "$LOCAL_EXPORT/.fingerprint" ] || return 1
    [ -n "$PYTHON_BIN" ] || return 1
    local current
    current="$("$PYTHON_BIN" "$PACKAGING_DIR/build.py" --print-fingerprint 2>/dev/null || true)"
    [ -n "$current" ] || return 1
    [ "$(cat "$LOCAL_EXPORT/.fingerprint")" = "$current" ]
}

_rebuild_venvstacks_export() {
    [ -n "$PYTHON_BIN" ] || die "python3 not found — install Python 3.11+ on PATH or set PYTHON_BIN."
    log "Rebuilding venvstacks export (this may take 5–10 minutes)…"
    "$PYTHON_BIN" "$PACKAGING_DIR/build.py" --venvstacks-only \
        || die "venvstacks rebuild failed; see output above."
    [ -d "$LOCAL_EXPORT" ] || die "venvstacks rebuild reported success but $LOCAL_EXPORT is missing."
    ok "Venvstacks export ready at $LOCAL_EXPORT"
}

_custom_kernel_deployment_target() {
    printf "%s\n" "${OMLX_CUSTOM_KERNEL_DEPLOYMENT_TARGET:-${MACOSX_DEPLOYMENT_TARGET:-15.0}}"
}

_custom_kernel_pythonpath() {
    [ -n "${DONOR_LAYERS:-}" ] || die "custom kernel build requires resolved donor layers."
    local mlx_site="$DONOR_LAYERS/framework-mlx-base/lib/python3.11/site-packages"
    [ -d "$mlx_site" ] || die "donor missing framework MLX site-packages: $mlx_site"
    if [ -n "${PYTHONPATH:-}" ]; then
        printf "%s:%s\n" "$mlx_site" "$PYTHONPATH"
    else
        printf "%s\n" "$mlx_site"
    fi
}

_clean_custom_kernel_build_artifacts() {
    local dir
    for dir in "${CUSTOM_KERNEL_DIRS[@]}"; do
        [ -d "$dir" ] || continue
        find "$dir" -maxdepth 1 -type f \( \
            -name "*.so" -o \
            -name "*.dylib" -o \
            -name "*.metallib" \
        \) -delete
    done

    if [ -d "$REPO_ROOT/build" ]; then
        for ext_name in \
            "omlx.custom_kernels.glm_moe_dsa._ext" \
            "omlx.custom_kernels.minimax_m3._ext" \
            "omlx.custom_kernels.qwen35_prefill._ext"; do
            find "$REPO_ROOT/build" \
                -type d \
                -name "$ext_name" \
                -prune \
                -exec rm -rf {} +
        done
    fi
}

_custom_kernel_minos() {
    otool -l "$1" 2>/dev/null | awk '
        /LC_BUILD_VERSION/ { in_build = 1 }
        in_build && /minos/ { print $2; exit }
    '
}

_validate_custom_kernel_deployment_target() {
    local expected="$1"
    local path
    local minos

    command -v otool >/dev/null 2>&1 || return 0

    local dir
    for dir in "${CUSTOM_KERNEL_DIRS[@]}"; do
        for path in "$dir"/_ext*.so "$dir"/lib*.dylib; do
            [ -e "$path" ] || continue
            minos="$(_custom_kernel_minos "$path")"
            [ -n "$minos" ] || die "could not read deployment target from $path."
            [ "$minos" = "$expected" ] \
                || die "custom kernel $path has macOS min $minos, expected $expected."
        done
    done
}

_check_custom_kernel_nanobind() {
    # setup.py build_ext runs without pip build isolation, so the pyproject
    # [build-system] nanobind pin is NOT enforced here — the kernels are
    # built with whatever nanobind $PYTHON_BIN resolves. A version that does
    # not match the one the bundled mlx wheel was built with isolates the
    # nanobind NB_DOMAIN: the extensions import and list symbols normally
    # but reject every mlx array at call time (issue #2139).
    local custom_kernel_pythonpath="$1"
    local expected actual
    expected="$(sed -n 's/^[[:space:]]*"nanobind==\([0-9][0-9.]*\)".*/\1/p' \
        "$REPO_ROOT/pyproject.toml" | head -1)"
    [ -n "$expected" ] || die "could not read the nanobind pin from pyproject.toml [build-system]."
    actual="$(PYTHONPATH="$custom_kernel_pythonpath" "$PYTHON_BIN" -c \
        'import nanobind; print(nanobind.__version__)' 2>/dev/null)" \
        || die "nanobind is not importable by $PYTHON_BIN — run: $PYTHON_BIN -m pip install nanobind==$expected"
    [ "$actual" = "$expected" ] \
        || die "nanobind $actual does not match the pyproject build pin $expected; the kernels would reject every mlx array at runtime (issue #2139). Run: $PYTHON_BIN -m pip install nanobind==$expected"
}

_check_custom_kernel_abi() {
    # Import each freshly built extension against the donor (bundled) mlx and
    # exercise the array type caster once. A metallib existence check cannot
    # catch a nanobind ABI mismatch — only an actual call does.
    local custom_kernel_pythonpath="$1"
    log "Verifying custom kernel ABI against the bundled MLX…"
    (
        cd "$REPO_ROOT"
        PYTHONPATH="$custom_kernel_pythonpath" "$PYTHON_BIN" - <<'PYEOF'
import importlib.util
import pathlib
import sys

import mlx.core as mx

failures = []
for name in ("glm_moe_dsa", "minimax_m3", "qwen35_prefill"):
    ext_dir = pathlib.Path("omlx/custom_kernels") / name
    so = next(ext_dir.glob("_ext.*.so"), None)
    if so is None:
        failures.append(f"{name}: _ext extension missing")
        continue
    # Extension modules must load under their compiled name (PyInit__ext);
    # CPython keys the extension cache by (name, path), so loading three
    # different .so files as "_ext" is fine.
    spec = importlib.util.spec_from_file_location("_ext", so)
    ext = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(ext)
    except Exception as exc:
        failures.append(f"{name}: import failed: {exc}")
        continue
    probe = getattr(ext, "abi_probe", None)
    if probe is None:
        failures.append(f"{name}: abi_probe symbol missing (stale build?)")
        continue
    try:
        probe(mx.zeros((1,)))
    except TypeError as exc:
        failures.append(f"{name}: mlx array rejected (nanobind ABI mismatch): {exc}")
for failure in failures:
    print(failure, file=sys.stderr)
sys.exit(1 if failures else 0)
PYEOF
    ) || die "custom kernel ABI check failed — the built extensions reject mlx arrays (issue #2139); see output above."
}

_build_custom_kernels() {
    [ -n "$PYTHON_BIN" ] || die "python3 not found — install Python 3.11+ on PATH or set PYTHON_BIN."
    local deployment_target
    local cmake_args
    local custom_kernel_pythonpath
    deployment_target="$(_custom_kernel_deployment_target)"
    cmake_args="${CMAKE_ARGS:-}"
    custom_kernel_pythonpath="$(_custom_kernel_pythonpath)"

    _check_custom_kernel_nanobind "$custom_kernel_pythonpath"
    log "Building optional native custom kernels (macOS deployment target $deployment_target)…"
    _clean_custom_kernel_build_artifacts
    (
        cd "$REPO_ROOT"
        export PYTHONPATH="$custom_kernel_pythonpath"
        export OMLX_CUSTOM_KERNEL_DEPLOYMENT_TARGET="$deployment_target"
        export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-$deployment_target}"
        if [[ "$cmake_args" != *"CMAKE_OSX_DEPLOYMENT_TARGET"* ]]; then
            export CMAKE_ARGS="${cmake_args:+$cmake_args }-DCMAKE_OSX_DEPLOYMENT_TARGET=$deployment_target"
        fi
        "$PYTHON_BIN" setup.py build_ext --inplace --force --with-custom-kernel
    ) || die "custom kernel build failed; see output above."
    [ -f "$REPO_ROOT/omlx/custom_kernels/glm_moe_dsa/omlx_glm_kernels.metallib" ] \
        || die "custom kernel build finished but GLM metallib is missing."
    [ -f "$REPO_ROOT/omlx/custom_kernels/minimax_m3/omlx_minimax_m3_kernels.metallib" ] \
        || die "custom kernel build finished but MiniMax M3 metallib is missing."
    [ -f "$REPO_ROOT/omlx/custom_kernels/qwen35_prefill/omlx_qwen35_prefill_kernels.metallib" ] \
        || die "custom kernel build finished but Qwen3.5 prefill metallib is missing."
    _validate_custom_kernel_deployment_target "$deployment_target"
    _check_custom_kernel_abi "$custom_kernel_pythonpath"
    ok "  + custom kernels ($deployment_target)"
}

resolve_donor_layers() {
    # Fast Swift rebuild mode reuses an already-built venvstacks export.
    # It intentionally refuses to rebuild or fall back to /Applications.
    if [ "$SWIFT_REBUILD" -eq 1 ]; then
        if [ ! -d "$LOCAL_EXPORT" ]; then
            die "Swift rebuild requires existing venvstacks export at $LOCAL_EXPORT. Run build.sh release first."
        fi
        DONOR_LAYERS="$LOCAL_EXPORT"
        DONOR_SOURCE="$LOCAL_EXPORT (swift rebuild, no venvstacks rebuild)"
        _local_export_fresh \
            || warn "Local export fingerprint mismatch; using stale layers (swift rebuild)."
        return
    fi

    # Explicit OMLX_DONOR_APP override → use it, skip rebuild logic entirely.
    if [ -n "$OMLX_DONOR_APP_SET" ]; then
        [ -d "$OMLX_DONOR_APP" ] || die "OMLX_DONOR_APP set but not found: $OMLX_DONOR_APP"
        DONOR_LAYERS="$(_app_python_layers_path "$OMLX_DONOR_APP")" \
            || die "OMLX_DONOR_APP missing bundled Python layers: $OMLX_DONOR_APP"
        DONOR_SOURCE="OMLX_DONOR_APP=$OMLX_DONOR_APP"
        return
    fi

    case "$REBUILD_DONOR" in
        force)
            _rebuild_venvstacks_export
            DONOR_LAYERS="$LOCAL_EXPORT"
            DONOR_SOURCE="$LOCAL_EXPORT (forced rebuild)"
            ;;
        never)
            if [ -d "$LOCAL_EXPORT" ]; then
                DONOR_LAYERS="$LOCAL_EXPORT"
                DONOR_SOURCE="$LOCAL_EXPORT (no rebuild)"
                _local_export_fresh \
                    || warn "Local export fingerprint mismatch; using stale layers (--no-rebuild-donor)."
            elif [ -d "$OMLX_DONOR_APP" ]; then
                DONOR_LAYERS="$(_app_python_layers_path "$OMLX_DONOR_APP")" \
                    || die "Fallback donor missing bundled Python layers: $OMLX_DONOR_APP"
                DONOR_SOURCE="$OMLX_DONOR_APP (fallback, --no-rebuild-donor)"
            else
                die "No donor available: $LOCAL_EXPORT and $OMLX_DONOR_APP both missing, --no-rebuild-donor prevents rebuild."
            fi
            ;;
        auto)
            if _local_export_fresh; then
                DONOR_LAYERS="$LOCAL_EXPORT"
                DONOR_SOURCE="$LOCAL_EXPORT (cached, fingerprint match)"
            else
                if [ -d "$LOCAL_EXPORT" ]; then
                    log "Local export is stale (fingerprint mismatch) — rebuilding."
                else
                    log "Local export missing — building."
                fi
                _rebuild_venvstacks_export
                DONOR_LAYERS="$LOCAL_EXPORT"
                DONOR_SOURCE="$LOCAL_EXPORT (rebuilt)"
            fi
            ;;
    esac
}

# --- Derive bundle version from omlx/_version.py --------------------------
#
# omlx/_version.py is the canonical source — pyproject.toml reads it via
# `[tool.setuptools.dynamic] version = {attr = ...}`. Mirror that into the
# Swift bundle so CFBundleShortVersionString (MARKETING_VERSION) tracks
# the Python package without a second hand-maintained string.
#
# CURRENT_PROJECT_VERSION uses `git rev-list --count HEAD` so each commit
# gives a monotonically-increasing CFBundleVersion — enough for the
# GitHub Releases updater's version comparisons.
#
# If either lookup fails the script bails rather than silently shipping
# stale numbers; falling back to the pbxproj placeholders would mask a
# real regression.

VERSION_FILE="$REPO_ROOT/omlx/_version.py"
[ -f "$VERSION_FILE" ] || die "missing $VERSION_FILE — cannot derive bundle version"
APP_VERSION=$(grep -oE '__version__[[:space:]]*=[[:space:]]*"[^"]+"' "$VERSION_FILE" | \
              sed -E 's/.*"([^"]+)".*/\1/')
[ -n "$APP_VERSION" ] || die "could not parse __version__ from $VERSION_FILE"
BUILD_NUMBER=$(git -C "$REPO_ROOT" rev-list --count HEAD 2>/dev/null || echo 1)
log "Bundle version: $APP_VERSION (build $BUILD_NUMBER) — from omlx/_version.py"

# --- xcodebuild -----------------------------------------------------------

log "Building oMLX ($CONFIG)…"
mkdir -p "$BUILD_DIR"

log "Resolving Swift package dependencies…"
xcodebuild -resolvePackageDependencies \
    -project "$PROJECT_DIR/oMLX.xcodeproj" \
    -scheme oMLX \
    >"$BUILD_DIR/spm-resolve.log" 2>&1 \
        || warn "SPM resolve emitted warnings; continuing with existing Package.resolved (see $BUILD_DIR/spm-resolve.log)."

xcodebuild \
    -project "$PROJECT_DIR/oMLX.xcodeproj" \
    -scheme oMLX \
    -configuration "$CONFIG" \
    -destination 'platform=macOS' \
    -derivedDataPath "$BUILD_DIR" \
    CODE_SIGN_IDENTITY="-" \
    CODE_SIGNING_REQUIRED=NO \
    CODE_SIGNING_ALLOWED=NO \
    MARKETING_VERSION="$APP_VERSION" \
    CURRENT_PROJECT_VERSION="$BUILD_NUMBER" \
    build >"$BUILD_DIR/xcodebuild.log" 2>&1 \
        || { tail -40 "$BUILD_DIR/xcodebuild.log" >&2; die "xcodebuild failed; full log: $BUILD_DIR/xcodebuild.log"; }

XCODE_APP="$BUILD_DIR/Build/Products/$CONFIG/oMLX.app"
[ -d "$XCODE_APP" ] || die "Expected $XCODE_APP — check build log."
ok "Built $XCODE_APP"

# --- Stage --------------------------------------------------------------

mkdir -p "$OUTPUT_DIR"
STAGED_APP="$OUTPUT_DIR/oMLX.app"

log "Staging bundle at $STAGED_APP"
rm -rf "$STAGED_APP"
ditto "$XCODE_APP" "$STAGED_APP"

if [ "$BARE" -eq 1 ]; then
    warn "--bare set: skipping Python embed. The server will fail to spawn."
    ok "Bundle ready: $STAGED_APP"
    exit 0
fi

RESOURCES_DIR="$STAGED_APP/Contents/Resources"
PYTHON_DIR="$RESOURCES_DIR/Python"
mkdir -p "$PYTHON_DIR" "$RESOURCES_DIR"

# --- Embed Python layers --------------------------------------------------

resolve_donor_layers
log "Using donor: $DONOR_SOURCE"
[ -d "$DONOR_LAYERS/cpython-3.11" ] || die "Donor missing cpython-3.11 at $DONOR_LAYERS"
[ -d "$DONOR_LAYERS/framework-mlx-base" ] || die "Donor missing framework-mlx-base at $DONOR_LAYERS"

log "Copying cpython-3.11 from donor…"
ditto "$DONOR_LAYERS/cpython-3.11" "$PYTHON_DIR/cpython-3.11"
ok "  + cpython-3.11"

log "Copying framework-mlx-base from donor (~1 GB)…"
ditto "$DONOR_LAYERS/framework-mlx-base" "$PYTHON_DIR/framework-mlx-base"
ok "  + framework-mlx-base"

if [ -d "$DONOR_LAYERS/__venvstacks__" ]; then
    ditto "$DONOR_LAYERS/__venvstacks__" "$PYTHON_DIR/__venvstacks__"
    ok "  + __venvstacks__ metadata"
fi

# --- Embed omlx package ---------------------------------------------------

if [ "$WITH_CUSTOM_KERNEL" = "1" ]; then
    _build_custom_kernels
fi

log "Copying omlx package from source tree…"
rm -rf "$RESOURCES_DIR/omlx"
mkdir -p "$RESOURCES_DIR/omlx"
# rsync gives us per-tree exclude semantics that ditto lacks.
RSYNC_EXCLUDES=(
    --exclude='__pycache__'
    --exclude='*.pyc'
    --exclude='tests'
    --exclude='.git'
    --exclude='custom_kernels/*/csrc'
)
if [ "$WITH_CUSTOM_KERNEL" != "1" ]; then
    RSYNC_EXCLUDES+=(
        --exclude='custom_kernels/*/*.so'
        --exclude='custom_kernels/*/*.dylib'
        --exclude='custom_kernels/*/*.metallib'
    )
fi
rsync -a \
    "${RSYNC_EXCLUDES[@]}" \
    "$REPO_ROOT/omlx/" "$RESOURCES_DIR/omlx/"
ok "  + omlx package"

log "Writing engine commit metadata..."
"$PYTHON_BIN" "$PACKAGING_DIR/build.py" --write-engine-commits "$RESOURCES_DIR/omlx" \
    || die "failed to write engine commit metadata."
ok "  + _engine_commits.json"

# --- Embed CLI wrapper ----------------------------------------------------

log "Writing app-bundle CLI wrapper..."
CLI_WRAPPER="$STAGED_APP/Contents/MacOS/omlx-cli"
cat > "$CLI_WRAPPER" <<'EOF'
#!/bin/sh
set -eu

REAL_PATH="$(realpath "$0")"
APP_ROOT="$(CDPATH= cd -- "$(dirname -- "$REAL_PATH")/.." && pwd)"
RESOURCES="$APP_ROOT/Resources"
PYROOT="$RESOURCES/Python"
CPYTHON="$PYROOT/cpython-3.11"
PYTHON="$CPYTHON/bin/python3"
MLX_SITE="$PYROOT/framework-mlx-base/lib/python3.11/site-packages"

export PYTHONHOME="$CPYTHON"
export PYTHONDONTWRITEBYTECODE=1
if [ -n "${PYTHONPATH:-}" ]; then
    export PYTHONPATH="$RESOURCES:$MLX_SITE:$PYTHONPATH"
else
    export PYTHONPATH="$RESOURCES:$MLX_SITE"
fi

exec "$PYTHON" -m omlx.cli "$@"
EOF
chmod 755 "$CLI_WRAPPER"
ok "  + omlx-cli"

# --- Compile AppIcon.icon (Tahoe Liquid Glass) ----------------------------
#
# Xcode 26.5's project build system does NOT route a standalone
# `Resources/AppIcon.icon` bundle into the actool invocation for macOS
# targets (despite folder.iconcomposer.icon being a declared input file
# type in the AssetCatalogCompiler.xcspec). The icon ends up missing from
# the bundle entirely.
#
# Workaround: invoke actool ourselves with both the asset catalog AND the
# .icon bundle as positional inputs. The result is a real AppIcon.icns +
# enriched Assets.car that macOS 26's Liquid Glass icon system picks up
# natively — no system tile wrap around our brand mark.
#
# We also patch the staged Info.plist with CFBundleIconName +
# CFBundleIconFile since the actool partial-info-plist isn't merged into
# the final plist by the time we're staging.

ICON_BUNDLE="$REPO_ROOT/apps/omlx-mac/Resources/AppIcon.icon"
XCASSETS_DIR="$REPO_ROOT/apps/omlx-mac/Resources/Assets.xcassets"
INFO_PLIST="$STAGED_APP/Contents/Info.plist"

if [ -d "$ICON_BUNDLE" ] && [ -d "$XCASSETS_DIR" ]; then
    log "Compiling AppIcon.icon via actool…"
    ACTOOL=/Applications/Xcode.app/Contents/Developer/usr/bin/actool
    ICON_TMP=$(mktemp -d)
    if "$ACTOOL" \
            "$XCASSETS_DIR" \
            "$ICON_BUNDLE" \
            --compile "$ICON_TMP" \
            --app-icon AppIcon \
            --target-device mac \
            --minimum-deployment-target 26.0 \
            --platform macosx \
            --bundle-identifier app.omlx \
            --output-format human-readable-text \
            --output-partial-info-plist "$ICON_TMP/icon.plist" \
            >/dev/null 2>&1; then
        cp "$ICON_TMP/AppIcon.icns" "$RESOURCES_DIR/AppIcon.icns"
        cp "$ICON_TMP/Assets.car"   "$RESOURCES_DIR/Assets.car"
        /usr/libexec/PlistBuddy \
            -c "Delete :CFBundleIconName" "$INFO_PLIST" 2>/dev/null || true
        /usr/libexec/PlistBuddy \
            -c "Add :CFBundleIconName string AppIcon" "$INFO_PLIST"
        /usr/libexec/PlistBuddy \
            -c "Delete :CFBundleIconFile" "$INFO_PLIST" 2>/dev/null || true
        /usr/libexec/PlistBuddy \
            -c "Add :CFBundleIconFile string AppIcon" "$INFO_PLIST"
        ok "  + AppIcon.icns + Assets.car (Tahoe icon-composer)"
    else
        warn "actool merge of AppIcon.icon failed; bundle ships legacy icon"
    fi
    rm -rf "$ICON_TMP"
fi

# --- Re-sign ad-hoc -------------------------------------------------------
#
# Even with CODE_SIGNING_ALLOWED=NO during xcodebuild, the staged bundle must
# have a coherent app-bundle signature. Sign embedded native Python/MLX
# libraries first, then seal the outer app bundle so TCC can match Full Disk
# Access grants against the app identity instead of a bare linker signature.

if [ "$SKIP_EMBEDDED_SIGN" -eq 1 ]; then
    warn "swift-fast set: skipping embedded native code signing."
else
    log "Ad-hoc signing embedded native code…"
    _sign_embedded_mach_o_files "$PYTHON_DIR"
    codesign --force --sign - "$CLI_WRAPPER" >/dev/null 2>&1
    ok "  + signed omlx-cli wrapper"
fi

log "Ad-hoc resigning app bundle…"
codesign --force --sign - "$STAGED_APP"

# Drop quarantine attributes so the bundle launches from anywhere.
xattr -dr com.apple.quarantine "$STAGED_APP" 2>/dev/null || true

# --- Done ----------------------------------------------------------------

ok "Done."
echo
echo "Bundle ready:"
echo "  $STAGED_APP"
echo
echo "To launch:"
echo "  open '$STAGED_APP'"
echo
echo "Server log will appear at:"
echo "  ~/Library/Application Support/oMLX/logs/server.log"
