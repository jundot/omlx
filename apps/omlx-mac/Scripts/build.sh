#!/usr/bin/env bash
# build.sh — produce a runnable Flyto MLX.app (thin menubar shell).
#
# The app does NOT embed a Python runtime. It shells out to the host
# project's virtualenv (~/Code/omlx/.venv/bin/python -m omlx.cli serve, see
# PythonRuntime.swift), so the bundle is a few MB and editing a worktree .py
# + restarting the server via the menubar takes effect immediately.
#
# Pipeline:
#   1. xcodebuild the Swift app target
#   2. stage the .app
#   3. compile the Tahoe Liquid Glass AppIcon via actool
#   4. ad-hoc re-sign + drop quarantine so it launches from anywhere
#
# Usage:
#   apps/omlx-mac/Scripts/build.sh            # Release (default)
#   apps/omlx-mac/Scripts/build.sh debug      # Debug build instead
#
# Env overrides:
#   OMLX_NEXT_OUT=/path/to/output_dir   # final stage location
#   DEVELOPER_DIR=/path/to/Xcode/.../Developer   # toolchain (auto-detected)

set -euo pipefail

CONFIG="${1:-Release}"
case "$(echo "$CONFIG" | tr '[:upper:]' '[:lower:]')" in
    debug)   CONFIG=Debug ;;
    release) CONFIG=Release ;;
    *) echo "error: unknown configuration '$CONFIG' (expected debug|release)" >&2; exit 2 ;;
esac

shift || true
for arg in "$@"; do
    case "$arg" in
        *) echo "error: unknown flag '$arg'" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"

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

# --- Toolchain: xcodebuild needs a full Xcode, not CommandLineTools --------

if ! xcodebuild -version >/dev/null 2>&1; then
    if [ -d /Applications/Xcode.app ]; then
        export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
    else
        die "xcodebuild unavailable. Install Xcode or run: sudo xcode-select -s /Applications/Xcode.app"
    fi
fi

# --- Derive bundle version from omlx/_version.py --------------------------
#
# omlx/_version.py is the canonical source — pyproject.toml reads it via
# `[tool.setuptools.dynamic]`. Mirror it into the Swift bundle so
# CFBundleShortVersionString (MARKETING_VERSION) tracks the Python package.
# CURRENT_PROJECT_VERSION uses `git rev-list --count HEAD` for a monotonic
# CFBundleVersion.

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

RESOURCES_DIR="$STAGED_APP/Contents/Resources"
mkdir -p "$RESOURCES_DIR"

# --- Compile AppIcon.icon (Tahoe Liquid Glass) ----------------------------
#
# Xcode 26's project build system does NOT route a standalone
# `Resources/AppIcon.icon` bundle into actool for macOS targets, so the icon
# ends up missing from the bundle. Workaround: invoke actool ourselves with
# both the asset catalog AND the .icon bundle as positional inputs, then
# patch the staged Info.plist with CFBundleIconName + CFBundleIconFile.

ICON_BUNDLE="$PROJECT_DIR/Resources/AppIcon.icon"
XCASSETS_DIR="$PROJECT_DIR/Resources/Assets.xcassets"
INFO_PLIST="$STAGED_APP/Contents/Info.plist"

if [ -d "$ICON_BUNDLE" ] && [ -d "$XCASSETS_DIR" ]; then
    log "Compiling AppIcon.icon via actool…"
    ACTOOL="$(xcrun --find actool 2>/dev/null || echo /Applications/Xcode.app/Contents/Developer/usr/bin/actool)"
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
# Re-sign the staged bundle ad-hoc so Gatekeeper doesn't refuse to launch it
# from a non-derived location on first quarantine attribute.

log "Ad-hoc resigning bundle…"
codesign --force --sign - "$STAGED_APP" >/dev/null 2>&1 || \
    warn "codesign emitted a warning; the app may still launch."

# Drop quarantine attributes so the bundle launches from anywhere.
xattr -dr com.apple.quarantine "$STAGED_APP" 2>/dev/null || true

# --- Done ----------------------------------------------------------------

ok "Done."
echo
echo "Bundle ready ($(du -sh "$STAGED_APP" | cut -f1)):"
echo "  $STAGED_APP"
echo
echo "The server is spawned from the host venv (~/Code/omlx/.venv); edit a"
echo "worktree .py and use the menubar's Force Restart to apply changes."
echo
echo "To launch:"
echo "  open '$STAGED_APP'"
echo
echo "Server log will appear at:"
echo "  ~/Library/Application Support/oMLX/logs/server.log"
