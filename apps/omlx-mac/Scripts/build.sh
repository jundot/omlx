#!/usr/bin/env bash
# build.sh — produce a runnable oMLX-next.app for local manual testing.
#
# This is the side-by-side Swift bundle path called out in plan.md §5.
# Pipeline:
#   1. xcodebuild with `-resolvePackageDependencies` so SPM deps pick up
#      any new minor/patch within the pin range each build
#   2. (auto) rebuild the venvstacks export if sources have drifted
#   3. stage the Swift `.app` and copy the venvstacks-produced Python
#      layers into Contents/Frameworks/ verbatim
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
#   apps/omlx-mac/Scripts/build.sh release --bare     # skip Python embed
#                                                       (no server, just the
#                                                       AppView shell)
#   apps/omlx-mac/Scripts/build.sh --rebuild-donor    # force venvstacks rebuild
#   apps/omlx-mac/Scripts/build.sh --no-rebuild-donor # never rebuild; use
#                                                       existing donor even if stale
#
# Env overrides:
#   OMLX_DONOR_APP=/path/to/oMLX.app    # explicit donor (bypasses venvstacks)
#   OMLX_NEXT_OUT=/path/to/output_dir   # final stage location
#   PYTHON_BIN=/path/to/python3         # python used for venvstacks driver
#                                       (default: PATH lookup of python3)

set -euo pipefail

CONFIG="${1:-Release}"
case "$(echo "$CONFIG" | tr '[:upper:]' '[:lower:]')" in
    debug)   CONFIG=Debug ;;
    release) CONFIG=Release ;;
    *) echo "error: unknown configuration '$CONFIG' (expected debug|release)" >&2; exit 2 ;;
esac

BARE=0
REBUILD_DONOR=auto    # auto | force | never
shift || true
for arg in "$@"; do
    case "$arg" in
        --bare) BARE=1 ;;
        --rebuild-donor) REBUILD_DONOR=force ;;
        --no-rebuild-donor) REBUILD_DONOR=never ;;
        *) echo "error: unknown flag '$arg'" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"
PACKAGING_DIR="$REPO_ROOT/packaging"
LOCAL_EXPORT="$PACKAGING_DIR/_export"

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

resolve_donor_layers() {
    # Explicit OMLX_DONOR_APP override → use it, skip rebuild logic entirely.
    if [ -n "$OMLX_DONOR_APP_SET" ]; then
        [ -d "$OMLX_DONOR_APP" ] || die "OMLX_DONOR_APP set but not found: $OMLX_DONOR_APP"
        DONOR_LAYERS="$OMLX_DONOR_APP/Contents/Python"
        [ -d "$DONOR_LAYERS" ] || DONOR_LAYERS="$OMLX_DONOR_APP/Contents/Frameworks"
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
                DONOR_LAYERS="$OMLX_DONOR_APP/Contents/Python"
                [ -d "$DONOR_LAYERS" ] || DONOR_LAYERS="$OMLX_DONOR_APP/Contents/Frameworks"
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

# --- xcodebuild -----------------------------------------------------------

log "Building oMLX-next ($CONFIG)…"
mkdir -p "$BUILD_DIR"

log "Resolving Swift package dependencies…"
xcodebuild -resolvePackageDependencies \
    -project "$PROJECT_DIR/oMLX.xcodeproj" \
    -scheme oMLX-next \
    >"$BUILD_DIR/spm-resolve.log" 2>&1 \
        || warn "SPM resolve emitted warnings; continuing with existing Package.resolved (see $BUILD_DIR/spm-resolve.log)."

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

resolve_donor_layers
log "Using donor: $DONOR_SOURCE"
[ -d "$DONOR_LAYERS/cpython-3.11" ] || die "Donor missing cpython-3.11 at $DONOR_LAYERS"
[ -d "$DONOR_LAYERS/framework-mlx-framework" ] || die "Donor missing framework-mlx-framework at $DONOR_LAYERS"

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
