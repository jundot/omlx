#!/usr/bin/env bash
# Build oMLX-next.app locally without code signing.
#
# Usage:
#   apps/omlx-mac/Scripts/build.sh             # Debug
#   apps/omlx-mac/Scripts/build.sh release     # Release
#
# Outputs the built .app path to stdout. Codesign + DMG packaging lands in
# packaging/build.py at PR 2; this script is for fast local iteration.

set -euo pipefail

CONFIG="${1:-Debug}"
case "$(echo "$CONFIG" | tr '[:upper:]' '[:lower:]')" in
    debug)   CONFIG=Debug ;;
    release) CONFIG=Release ;;
    *) echo "error: unknown configuration '$CONFIG' (expected debug|release)" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DERIVED="$APP_DIR/build"

xcodebuild \
    -project "$APP_DIR/oMLX.xcodeproj" \
    -scheme oMLX-next \
    -configuration "$CONFIG" \
    -derivedDataPath "$DERIVED" \
    CODE_SIGN_IDENTITY="" \
    CODE_SIGNING_REQUIRED=NO \
    CODE_SIGNING_ALLOWED=NO \
    build

APP_PATH="$DERIVED/Build/Products/$CONFIG/oMLX-next.app"
if [[ ! -d "$APP_PATH" ]]; then
    echo "error: expected app at $APP_PATH but it doesn't exist" >&2
    exit 1
fi

echo
echo "Built: $APP_PATH"
