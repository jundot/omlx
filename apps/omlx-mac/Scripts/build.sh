#!/usr/bin/env bash
# build.sh — produce a runnable oMLX-next.app for local manual testing.
#
# This is the side-by-side Swift bundle path called out in plan.md §5.
# It runs xcodebuild then "stages" the Python runtime (venvstacks layers +
# the omlx package) in-place. The canonical, codesigned + notarized release
# path will live in `packaging/build.py` once PR 12 lands; this script is
# just enough to run an end-to-end smoke test against the live admin API.
#
# Usage:
#   apps/omlx-mac/Scripts/build.sh                    # Release, default donor
#   apps/omlx-mac/Scripts/build.sh debug              # Debug build instead
#   apps/omlx-mac/Scripts/build.sh release --bare     # skip Python embed
#                                                       (no server, just the
#                                                       AppView shell)
#   apps/omlx-mac/Scripts/build.sh release --no-sync  # skip uv sync + overlay
#                                                       (use donor layer as-is)
#
# Env overrides:
#   OMLX_DONOR_APP=/path/to/oMLX.app    # provider of cpython + mlx layers
#   OMLX_NEXT_OUT=/path/to/output_dir   # final stage location
#   UV_BIN=/path/to/uv                  # explicit uv binary (default: PATH lookup)
#
# The donor app provides cpython-3.11 + framework-mlx-framework. The omlx
# package itself is copied from this checkout's `omlx/` so local Python
# changes are reflected in the bundle.
#
# Pyproject pins newer than the donor's frozen layer (e.g. mlx-vlm @ f96138e)
# are reconciled by syncing an isolated Python 3.11 venv at $BUILD_DIR/venv
# and overlaying selected packages onto framework-mlx-framework. See the
# OVERLAY_PKGS list below.

set -euo pipefail

CONFIG="${1:-Release}"
case "$(echo "$CONFIG" | tr '[:upper:]' '[:lower:]')" in
    debug)   CONFIG=Debug ;;
    release) CONFIG=Release ;;
    *) echo "error: unknown configuration '$CONFIG' (expected debug|release)" >&2; exit 2 ;;
esac

BARE=0
NO_SYNC=0
shift || true
for arg in "$@"; do
    case "$arg" in
        --bare) BARE=1 ;;
        --no-sync) NO_SYNC=1 ;;
        *) echo "error: unknown flag '$arg'" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"

DONOR_APP="${OMLX_DONOR_APP:-/Applications/oMLX.app}"
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

# --- xcodebuild -----------------------------------------------------------

log "Building oMLX-next ($CONFIG)…"
mkdir -p "$BUILD_DIR"

xcodebuild \
    -project "$PROJECT_DIR/oMLX.xcodeproj" \
    -scheme oMLX-next \
    -configuration "$CONFIG" \
    -destination 'platform=macOS' \
    -derivedDataPath "$BUILD_DIR" \
    CODE_SIGN_IDENTITY="-" \
    CODE_SIGNING_REQUIRED=NO \
    CODE_SIGNING_ALLOWED=NO \
    build >"$BUILD_DIR/xcodebuild.log" 2>&1 \
        || { tail -40 "$BUILD_DIR/xcodebuild.log" >&2; die "xcodebuild failed; full log: $BUILD_DIR/xcodebuild.log"; }

XCODE_APP="$BUILD_DIR/Build/Products/$CONFIG/oMLX-next.app"
[ -d "$XCODE_APP" ] || die "Expected $XCODE_APP — check build log."
ok "Built $XCODE_APP"

# --- Stage --------------------------------------------------------------

mkdir -p "$OUTPUT_DIR"
STAGED_APP="$OUTPUT_DIR/oMLX-next.app"

log "Staging bundle at $STAGED_APP"
rm -rf "$STAGED_APP"
ditto "$XCODE_APP" "$STAGED_APP"

if [ "$BARE" -eq 1 ]; then
    warn "--bare set: skipping Python embed. The server will fail to spawn."
    ok "Bundle ready: $STAGED_APP"
    exit 0
fi

FRAMEWORKS_DIR="$STAGED_APP/Contents/Frameworks"
RESOURCES_DIR="$STAGED_APP/Contents/Resources"
mkdir -p "$FRAMEWORKS_DIR" "$RESOURCES_DIR"

# --- Embed Python layers --------------------------------------------------

[ -d "$DONOR_APP" ] || die "Donor app not found: $DONOR_APP — install /Applications/oMLX.app or set OMLX_DONOR_APP."

DONOR_LAYERS="$DONOR_APP/Contents/Python"
if [ ! -d "$DONOR_LAYERS" ]; then
    DONOR_LAYERS="$DONOR_APP/Contents/Frameworks"
fi
[ -d "$DONOR_LAYERS/cpython-3.11" ] || die "Donor missing cpython-3.11 at $DONOR_LAYERS"
[ -d "$DONOR_LAYERS/framework-mlx-framework" ] || die "Donor missing framework-mlx-framework"

log "Copying cpython-3.11 from donor…"
ditto "$DONOR_LAYERS/cpython-3.11" "$FRAMEWORKS_DIR/cpython-3.11"
ok "  + cpython-3.11"

log "Copying framework-mlx-framework from donor (~1 GB)…"
ditto "$DONOR_LAYERS/framework-mlx-framework" "$FRAMEWORKS_DIR/framework-mlx-framework"
ok "  + framework-mlx-framework"

if [ -d "$DONOR_LAYERS/__venvstacks__" ]; then
    ditto "$DONOR_LAYERS/__venvstacks__" "$FRAMEWORKS_DIR/__venvstacks__"
    ok "  + __venvstacks__ metadata"
fi

# --- Overlay diverging packages from a fresh uv-synced 3.11 venv ---------
#
# The donor app's framework-mlx-framework is frozen to whatever was current
# at its build time. When pyproject.toml later moves a dependency forward
# (e.g. `mlx-vlm @ f96138e` for the new `mlx_vlm.speculative.utils`
# module), the donor's site-packages goes stale and the bundled server
# fails to import at startup.
#
# Resolve this by syncing an isolated Python 3.11 venv under $BUILD_DIR
# (kept separate from the worktree's own .venv, which may target a newer
# Python) and overlaying the listed packages into the mlx framework layer.
# Add package names to OVERLAY_PKGS as more pins drift from the donor.

if [ "$NO_SYNC" -eq 0 ]; then
    UV_BIN="${UV_BIN:-$(command -v uv || true)}"
    if [ -z "$UV_BIN" ]; then
        for candidate in /opt/homebrew/bin/uv "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
            if [ -x "$candidate" ]; then UV_BIN="$candidate"; break; fi
        done
    fi
    [ -n "$UV_BIN" ] || die "uv not found — install via Homebrew (brew install uv), set UV_BIN, or pass --no-sync."

    BUNDLE_VENV="$BUILD_DIR/venv"
    log "Syncing bundle venv at $BUNDLE_VENV (Python 3.11)…"
    UV_PROJECT_ENVIRONMENT="$BUNDLE_VENV" "$UV_BIN" sync \
        --python 3.11 \
        --project "$REPO_ROOT" \
        >"$BUILD_DIR/uv-sync.log" 2>&1 \
        || { tail -40 "$BUILD_DIR/uv-sync.log" >&2; die "uv sync failed; full log: $BUILD_DIR/uv-sync.log"; }

    VENV_SITE="$BUNDLE_VENV/lib/python3.11/site-packages"
    [ -d "$VENV_SITE" ] || die "Expected $VENV_SITE after uv sync — check $BUILD_DIR/uv-sync.log."

    OVERLAY_PKGS=("mlx_vlm")
    MLX_LAYER_SITE="$FRAMEWORKS_DIR/framework-mlx-framework/lib/python3.11/site-packages"

    for pkg in "${OVERLAY_PKGS[@]}"; do
        SRC="$VENV_SITE/$pkg"
        if [ ! -d "$SRC" ]; then
            warn "  skipped overlay: $pkg not present in $VENV_SITE"
            continue
        fi
        log "Overlaying $pkg from bundle venv → mlx framework layer…"
        rm -rf "$MLX_LAYER_SITE/$pkg"
        rsync -a --exclude='__pycache__' --exclude='*.pyc' "$SRC/" "$MLX_LAYER_SITE/$pkg/"
        # Drop stale donor dist-info for the package (both dash + underscore
        # variants) and copy the freshly synced one so pip metadata stays
        # consistent with the overlay.
        pkg_dash="${pkg//_/-}"
        find "$MLX_LAYER_SITE" -maxdepth 1 \
            \( -name "${pkg}-*.dist-info" -o -name "${pkg_dash}-*.dist-info" \) \
            -exec rm -rf {} + 2>/dev/null || true
        find "$VENV_SITE" -maxdepth 1 \
            \( -name "${pkg}-*.dist-info" -o -name "${pkg_dash}-*.dist-info" \) \
            -print | while read -r dist; do
                rsync -a "$dist/" "$MLX_LAYER_SITE/$(basename "$dist")/"
            done
        ok "  + $pkg (overlaid from bundle venv)"
    done
else
    warn "--no-sync set: donor framework-mlx-framework used as-is; newer pins won't apply."
fi

# --- Embed omlx package ---------------------------------------------------

log "Copying omlx package from source tree…"
rm -rf "$RESOURCES_DIR/omlx"
mkdir -p "$RESOURCES_DIR/omlx"
# rsync gives us per-tree exclude semantics that ditto lacks.
rsync -a \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='tests' \
    --exclude='.git' \
    "$REPO_ROOT/omlx/" "$RESOURCES_DIR/omlx/"
ok "  + omlx package"

# --- Re-sign ad-hoc -------------------------------------------------------
#
# Even with CODE_SIGNING_ALLOWED=NO during xcodebuild, we re-sign the staged
# bundle ad-hoc so Gatekeeper doesn't refuse to launch it from a non-derived
# location on first quarantine attribute. The Sparkle inner XPC services
# need to be signed before the umbrella .framework, which needs to be signed
# before the outer .app.

if [ -d "$FRAMEWORKS_DIR/Sparkle.framework" ]; then
    log "Ad-hoc resigning Sparkle.framework…"
    SPARKLE_BASE="$FRAMEWORKS_DIR/Sparkle.framework/Versions/B"
    for inner in \
        "$SPARKLE_BASE/XPCServices/Installer.xpc" \
        "$SPARKLE_BASE/XPCServices/Downloader.xpc" \
        "$SPARKLE_BASE/Autoupdate" \
        "$SPARKLE_BASE/Updater.app"; do
        [ -e "$inner" ] && codesign --force --sign - "$inner" >/dev/null 2>&1 || true
    done
    codesign --force --sign - "$FRAMEWORKS_DIR/Sparkle.framework" >/dev/null 2>&1 || true
fi

log "Ad-hoc resigning outer bundle…"
codesign --force --sign - "$STAGED_APP" >/dev/null 2>&1 || \
    warn "outer codesign emitted a warning; the app may still launch."

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
echo "  ~/Library/Application Support/oMLX-next/logs/server.log"
