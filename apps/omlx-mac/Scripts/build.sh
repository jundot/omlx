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
#   OMLX_EXPORT_DIR=/path/to/_export    # override the venvstacks export tree
#                                       (release builds set this per macOS target)
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

FRAMEWORKS_DIR="$STAGED_APP/Contents/Frameworks"
RESOURCES_DIR="$STAGED_APP/Contents/Resources"
mkdir -p "$FRAMEWORKS_DIR" "$RESOURCES_DIR"

# --- Embed Python layers --------------------------------------------------

resolve_donor_layers
log "Using donor: $DONOR_SOURCE"
[ -d "$DONOR_LAYERS/cpython-3.11" ] || die "Donor missing cpython-3.11 at $DONOR_LAYERS"
[ -d "$DONOR_LAYERS/framework-mlx-base" ] || die "Donor missing framework-mlx-base at $DONOR_LAYERS"

log "Copying cpython-3.11 from donor…"
ditto "$DONOR_LAYERS/cpython-3.11" "$FRAMEWORKS_DIR/cpython-3.11"
ok "  + cpython-3.11"

log "Copying framework-mlx-base from donor (~1 GB)…"
ditto "$DONOR_LAYERS/framework-mlx-base" "$FRAMEWORKS_DIR/framework-mlx-base"
ok "  + framework-mlx-base"

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
# Even with CODE_SIGNING_ALLOWED=NO during xcodebuild, we re-sign the staged
# bundle ad-hoc so Gatekeeper doesn't refuse to launch it from a non-derived
# location on first quarantine attribute.

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
echo "  ~/Library/Application Support/oMLX/logs/server.log"
