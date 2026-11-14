#!/usr/bin/env bash
# package_dmg.sh — wrap the staged oMLX.app in a direct-download DMG.
#
# The staged app is produced by `apps/omlx-mac/Scripts/build.sh release --slim`;
# this script does not rebuild the Python/runtime layers. Signing and
# notarization happen only when their credentials are supplied; without them it
# produces an unsigned DMG for local inspection and says so. When an identity is
# given, the helper is signed first, then the app, then the .dmg image itself
# (an unsigned container cannot be assessed by Gatekeeper even after the app
# inside has been notarized).
#
# Usage:
#   apps/omlx-mac/Scripts/package_dmg.sh
#   SIGN_IDENTITY="Developer ID Application: Andreas Kolb (YB894QRG4W)" \
#   NOTARY_PROFILE=omlx-notary apps/omlx-mac/Scripts/package_dmg.sh
#
# Env:
#   APP_PATH        staged app (default: apps/omlx-mac/build/Stage/oMLX.app)
#   OUT_DIR         output directory (default: build)
#   VERSION         DMG version suffix (default: Info.plist CFBundleShortVersionString)
#   SIGN_IDENTITY   Developer ID Application identity; empty = unsigned
#   NOTARY_PROFILE  notarytool keychain profile; empty = skip notarization

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
APP_PATH="${APP_PATH:-$ROOT/apps/omlx-mac/build/Stage/oMLX.app}"
OUT_DIR="${OUT_DIR:-$ROOT/build}"
SIGN_IDENTITY="${SIGN_IDENTITY:-}"
NOTARY_PROFILE="${NOTARY_PROFILE:-}"

die() { echo "package_dmg: $*" >&2; exit 1; }

[ -d "$APP_PATH" ] || die "staged app not found at $APP_PATH; run build.sh release first"
INFO="$APP_PATH/Contents/Info.plist"
VERSION="${VERSION:-$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$INFO" 2>/dev/null || true)}"
VERSION="${VERSION:-0.0.0}"
mkdir -p "$OUT_DIR"

if [ -n "$SIGN_IDENTITY" ]; then
    HELPER="$APP_PATH/Contents/Library/LaunchServices/app.omlx.helper"
    if [ -x "$HELPER" ]; then
        echo "signing the privileged helper"
        codesign --force --options runtime --timestamp \
            --identifier app.omlx.helper \
            --sign "$SIGN_IDENTITY" "$HELPER"
        codesign --verify --strict --verbose=2 "$HELPER"
    fi
    echo "signing $APP_PATH"
    codesign --force --deep --options runtime \
        --entitlements "$ROOT/apps/omlx-mac/Resources/oMLX.entitlements" \
        --sign "$SIGN_IDENTITY" "$APP_PATH"
    codesign --verify --deep --strict --verbose=2 "$APP_PATH"
else
    echo "package_dmg: SIGN_IDENTITY is empty; producing an UNSIGNED DMG" >&2
fi

STAGE="$(mktemp -d "${TMPDIR:-/tmp}/omlx-dmg.XXXXXX")"
cleanup() { rm -rf "$STAGE"; }
trap cleanup EXIT
ditto "$APP_PATH" "$STAGE/oMLX.app"
ln -s /Applications "$STAGE/Applications"

DMG="$OUT_DIR/oMLX-$VERSION.dmg"
hdiutil create -volname "oMLX" -srcfolder "$STAGE" -ov -format UDZO "$DMG"

# Sign the image itself. Notarization alone leaves the .dmg container unsigned,
# so `spctl --type open` cannot assess it ("no usable signature") even though the
# app inside is notarized. A signed image is what Gatekeeper check + distribution
# expect; the app and helper were already signed above.
if [ -n "$SIGN_IDENTITY" ]; then
    echo "signing $DMG"
    codesign --force --timestamp --sign "$SIGN_IDENTITY" "$DMG"
    codesign --verify --verbose=2 "$DMG"
fi

if [ -n "$NOTARY_PROFILE" ] && [ -n "$SIGN_IDENTITY" ]; then
    echo "notarizing $DMG"
    xcrun notarytool submit "$DMG" --keychain-profile "$NOTARY_PROFILE" --wait
    xcrun stapler staple "$DMG"
    xcrun stapler validate "$DMG"
    spctl --assess --type open --context context:primary-signature --verbose=2 "$DMG"
else
    echo "package_dmg: notarization skipped (SIGN_IDENTITY/NOTARY_PROFILE not both set)" >&2
fi

shasum -a 256 "$DMG"
echo "created $DMG"
