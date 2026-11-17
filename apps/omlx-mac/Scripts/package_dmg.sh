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
    # Sign nested Mach-O binaries inside-out first. `codesign --deep` on the
    # bundle does not reliably seal arbitrary Mach-O files under
    # Contents/Resources (the compiled custom kernels are exactly that), and an
    # unsigned nested binary makes notarization fail with "not signed with a
    # valid Developer ID certificate" + "signature does not include a secure
    # timestamp".
    # `Contents/MacOS` may hold our own launcher scripts (the slim runtime
    # wrappers `omlx-cli` and `omlx-cluster-python`). A bundle cannot be sealed
    # while a subcomponent there is unsigned: codesign reports "code object is
    # not signed at all" and names the script — which is exactly what blocked
    # the first slim DMG after the inside-out signing change. A non-Mach-O
    # executable takes a Developer ID signature in its extended attributes,
    # which the bundle seal and the ditto copy into the DMG both keep. They are
    # signed *before* the nested Mach-O pass, because signing the bundle's main
    # executable first validates the siblings and fails on an unsigned one.
    LAUNCHER_COUNT=0
    for launcher in "$APP_PATH/Contents/MacOS/"*; do
        [ -f "$launcher" ] || continue
        [ -x "$launcher" ] || continue
        case "$(file -b "$launcher")" in
            *Mach-O*) continue ;;
        esac
        codesign --force --options runtime --timestamp \
            --sign "$SIGN_IDENTITY" "$launcher"
        LAUNCHER_COUNT=$((LAUNCHER_COUNT + 1))
    done
    echo "  signed $LAUNCHER_COUNT MacOS launchers"

    echo "signing nested Mach-O binaries"
    NESTED_LIST="$(mktemp "${TMPDIR:-/tmp}/omlx-nested.XXXXXX")"
    find "$APP_PATH" -type f \
        \( -name "*.so" -o -name "*.dylib" -o -name "*.node" -o -perm -111 \) \
        -print0 > "$NESTED_LIST.0"
    tr '\0' '\n' < "$NESTED_LIST.0" > "$NESTED_LIST"
    rm -f "$NESTED_LIST.0"
    # The bundle's own main executable is sealed by the bundle signature below;
    # signing it here would validate the still-unsigned siblings first.
    MAIN_EXECUTABLE="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleExecutable' "$APP_PATH/Contents/Info.plist" 2>/dev/null || true)"
    NESTED_COUNT=0
    while IFS= read -r binary; do
        [ -n "$binary" ] || continue
        if [ -n "$MAIN_EXECUTABLE" ] && \
            [ "$binary" = "$APP_PATH/Contents/MacOS/$MAIN_EXECUTABLE" ]; then
            continue
        fi
        case "$(file -b "$binary")" in
            *Mach-O*)
                codesign --force --options runtime --timestamp \
                    --sign "$SIGN_IDENTITY" "$binary"
                NESTED_COUNT=$((NESTED_COUNT + 1))
                ;;
        esac
    done < "$NESTED_LIST"
    rm -f "$NESTED_LIST"
    echo "  signed $NESTED_COUNT nested binaries"

    echo "signing $APP_PATH"
    codesign --force --options runtime \
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
