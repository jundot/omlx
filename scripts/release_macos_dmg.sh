#!/usr/bin/env bash
# Build a signed, notarized, stapled arm64 DMG from an already-staged oMLX.app.
# Certificate import is intentionally owned by the caller; this script never
# reads a login keychain or exports signing material.

set -euo pipefail

die() {
    echo "release_macos_dmg.sh: error: $*" >&2
    exit 1
}

usage() {
    cat >&2 <<'EOF'
Usage: release_macos_dmg.sh \
  --app /path/to/oMLX.app \
  --output-dir /path/to/dist \
  --identity SIGNING_IDENTITY \
  --team-id TEAM_ID \
  --notary-key /path/to/AuthKey_KEYID.p8 \
  --notary-key-id KEY_ID \
  --notary-issuer ISSUER_UUID \
  [--version VERSION] [--entitlements /path/to/file.entitlements]
EOF
    exit 2
}

APP_PATH=""
OUTPUT_DIR=""
SIGNING_IDENTITY=""
APPLE_TEAM_ID=""
NOTARY_KEY_PATH=""
NOTARY_KEY_ID=""
NOTARY_ISSUER_ID=""
EXPECTED_VERSION=""
ENTITLEMENTS_PATH=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --app) APP_PATH="${2:-}"; shift 2 ;;
        --output-dir) OUTPUT_DIR="${2:-}"; shift 2 ;;
        --identity) SIGNING_IDENTITY="${2:-}"; shift 2 ;;
        --team-id) APPLE_TEAM_ID="${2:-}"; shift 2 ;;
        --notary-key) NOTARY_KEY_PATH="${2:-}"; shift 2 ;;
        --notary-key-id) NOTARY_KEY_ID="${2:-}"; shift 2 ;;
        --notary-issuer) NOTARY_ISSUER_ID="${2:-}"; shift 2 ;;
        --version) EXPECTED_VERSION="${2:-}"; shift 2 ;;
        --entitlements) ENTITLEMENTS_PATH="${2:-}"; shift 2 ;;
        -h|--help) usage ;;
        *) die "unknown argument: $1" ;;
    esac
done

[ -d "$APP_PATH" ] || die "app bundle not found: $APP_PATH"
[ -n "$OUTPUT_DIR" ] || usage
[ -n "$SIGNING_IDENTITY" ] || usage
[[ "$APPLE_TEAM_ID" =~ ^[A-Z0-9]{10}$ ]] || die "team ID must be 10 uppercase letters/digits"
[ -f "$NOTARY_KEY_PATH" ] || die "notary API key not found"
[[ "$NOTARY_KEY_ID" =~ ^[A-Z0-9]{10,}$ ]] || die "invalid App Store Connect API key ID"
[[ "$NOTARY_ISSUER_ID" =~ ^[0-9A-Fa-f-]{36}$ ]] || die "invalid App Store Connect issuer ID"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
if [ -z "$ENTITLEMENTS_PATH" ]; then
    ENTITLEMENTS_PATH="$REPO_ROOT/apps/omlx-mac/Resources/oMLX.entitlements"
fi
[ -f "$ENTITLEMENTS_PATH" ] || die "entitlements file not found: $ENTITLEMENTS_PATH"

APP_PATH="$(cd "$(dirname "$APP_PATH")" && pwd -P)/$(basename "$APP_PATH")"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd -P)"

BUNDLE_ID="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$APP_PATH/Contents/Info.plist")"
APP_VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$APP_PATH/Contents/Info.plist")"
[ "$BUNDLE_ID" = "app.omlx" ] || die "unexpected bundle identifier: $BUNDLE_ID"
[[ "$APP_VERSION" =~ ^[0-9A-Za-z.+-]+$ ]] || die "unsafe bundle version: $APP_VERSION"
if [ -n "$EXPECTED_VERSION" ] && [ "$APP_VERSION" != "$EXPECTED_VERSION" ]; then
    die "bundle version $APP_VERSION does not match requested version $EXPECTED_VERSION"
fi

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/omlx-release.XXXXXX")"
MOUNT_POINT=""
cleanup() {
    if [ -n "$MOUNT_POINT" ] && mount | grep -Fq " on $MOUNT_POINT "; then
        hdiutil detach "$MOUNT_POINT" >/dev/null 2>&1 || true
    fi
    rm -rf "$TMP_ROOT"
}
trap cleanup EXIT INT TERM

echo "Validating bundle structure and arm64 code…"
APP_PATH="$APP_PATH" /usr/bin/python3 - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["APP_PATH"]).resolve()
for link in root.rglob("*"):
    if not link.is_symlink():
        continue
    target = os.readlink(link)
    if os.path.isabs(target):
        raise SystemExit(f"absolute symlink is not allowed in release bundle: {link}")
    resolved = link.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError:
        raise SystemExit(f"symlink escapes release bundle: {link} -> {target}")
PY

MACH_O_PATHS=()
while IFS= read -r -d '' path; do
    file_type="$(file -b "$path" 2>/dev/null || true)"
    case "$file_type" in
        *Mach-O*)
            architectures="$(lipo -archs "$path" 2>/dev/null || true)"
            case " $architectures " in
                *" arm64 "*) ;;
                *) die "Mach-O file has no arm64 slice: $path ($architectures)" ;;
            esac
            MACH_O_PATHS[${#MACH_O_PATHS[@]}]="$path"
            ;;
    esac
done < <(
    find "$APP_PATH" \
        \( -path '*/.dSYM/*' -o -path '*/__pycache__/*' \) -prune -o \
        -type f \( \
            -name '*.so' -o -name '*.dylib' -o -name '*.bundle' -o \
            -perm -100 -o -perm -010 -o -perm -001 \
        \) -print0
)
[ "${#MACH_O_PATHS[@]}" -gt 0 ] || die "app contains no Mach-O code"

# Include the governing source license and the retained MLX MIT notice with
# the distributed app. The public-release checklist tracks the broader
# dependency/data notice audit separately.
LICENSES_DIR="$APP_PATH/Contents/Resources/Licenses"
mkdir -p "$LICENSES_DIR"
cp "$REPO_ROOT/LICENSE" "$LICENSES_DIR/oMLX-Apache-2.0.txt"
cp "$REPO_ROOT/omlx/custom_kernels/glm_moe_dsa/csrc/MLX_LICENSE.txt" \
    "$LICENSES_DIR/MLX-MIT.txt"

echo "Signing ${#MACH_O_PATHS[@]} Mach-O files with hardened runtime…"
for path in "${MACH_O_PATHS[@]}"; do
    codesign --force --timestamp --options runtime \
        --sign "$SIGNING_IDENTITY" "$path"
done

# build.sh ad-hoc signs the two shell entry points. Replace those signatures
# as well; on a disk image their extended-attribute signatures are preserved.
for path in "$APP_PATH/Contents/MacOS/omlx-cli" \
            "$APP_PATH/Contents/MacOS/omlx-cluster-python"; do
    [ -f "$path" ] || continue
    codesign --force --timestamp --options runtime \
        --sign "$SIGNING_IDENTITY" "$path"
done

# Seal nested code bundles from the inside out. `find -depth` emits children
# before parents; the outer app is sealed separately with its entitlements.
while IFS= read -r -d '' bundle; do
    [ "$bundle" = "$APP_PATH" ] && continue
    codesign --force --timestamp --options runtime \
        --sign "$SIGNING_IDENTITY" "$bundle"
done < <(
    find "$APP_PATH" -depth -type d \( \
        -name '*.app' -o -name '*.framework' -o -name '*.xpc' -o \
        -name '*.appex' -o -name '*.plugin' -o -name '*.bundle' \
    \) -print0
)

codesign --force --timestamp --options runtime \
    --entitlements "$ENTITLEMENTS_PATH" \
    --sign "$SIGNING_IDENTITY" "$APP_PATH"

codesign --verify --deep --strict --verbose=2 "$APP_PATH"
SIGNING_DETAILS="$(codesign --display --verbose=4 "$APP_PATH" 2>&1)"
ACTUAL_TEAM_ID="$(printf '%s\n' "$SIGNING_DETAILS" | sed -n 's/^TeamIdentifier=//p' | head -1)"
[ "$ACTUAL_TEAM_ID" = "$APPLE_TEAM_ID" ] \
    || die "signed app team ID does not match configured team"
printf '%s\n' "$SIGNING_DETAILS" | grep -q 'flags=.*runtime' \
    || die "outer app signature is missing hardened runtime"
printf '%s\n' "$SIGNING_DETAILS" | grep -q '^Timestamp=' \
    || die "outer app signature is missing a secure timestamp"

ENTITLEMENTS_DUMP="$TMP_ROOT/entitlements.plist"
codesign --display --entitlements "$ENTITLEMENTS_DUMP" "$APP_PATH" 2>/dev/null
if /usr/libexec/PlistBuddy -c 'Print :com.apple.security.get-task-allow' \
        "$ENTITLEMENTS_DUMP" >/dev/null 2>&1; then
    die "release signature contains forbidden get-task-allow entitlement"
fi

NOTARY_ARGS=(
    --key "$NOTARY_KEY_PATH"
    --key-id "$NOTARY_KEY_ID"
    --issuer "$NOTARY_ISSUER_ID"
)

notarize() {
    local artifact="$1"
    local label="$2"
    local response="$TMP_ROOT/notary-$label-response.json"
    local submission_id
    local status

    echo "Submitting $label to Apple's notary service…"
    if ! xcrun notarytool submit "$artifact" \
            "${NOTARY_ARGS[@]}" --wait --timeout 60m \
            --output-format json >"$response"; then
        cp "$response" "$OUTPUT_DIR/notary-$label-response.json" 2>/dev/null || true
        die "notarytool submission failed for $label"
    fi
    status="$(plutil -extract status raw -o - "$response" 2>/dev/null || true)"
    submission_id="$(plutil -extract id raw -o - "$response" 2>/dev/null || true)"
    if [ "$status" != "Accepted" ]; then
        if [ -n "$submission_id" ]; then
            xcrun notarytool log "$submission_id" "${NOTARY_ARGS[@]}" \
                "$OUTPUT_DIR/notary-$label-log.json" || true
        fi
        cp "$response" "$OUTPUT_DIR/notary-$label-response.json" 2>/dev/null || true
        die "Apple notarization did not accept $label (status: ${status:-unknown})"
    fi
}

# Notarize and staple the app before placing it in the DMG, so the updater's
# syspolicy_check remains valid after it copies the app out of the image.
APP_ZIP="$TMP_ROOT/oMLX-$APP_VERSION.zip"
ditto -c -k --keepParent --sequesterRsrc "$APP_PATH" "$APP_ZIP"
notarize "$APP_ZIP" app
xcrun stapler staple -v "$APP_PATH"
xcrun stapler validate -v "$APP_PATH"
/usr/bin/syspolicy_check distribution "$APP_PATH"

DMG_ROOT="$TMP_ROOT/dmg-root"
mkdir -p "$DMG_ROOT"
ditto "$APP_PATH" "$DMG_ROOT/oMLX.app"
ln -s /Applications "$DMG_ROOT/Applications"

DMG_NAME="oMLX-$APP_VERSION-macos15-26-arm64.dmg"
DMG_PATH="$OUTPUT_DIR/$DMG_NAME"
hdiutil create -volname "oMLX $APP_VERSION" -srcfolder "$DMG_ROOT" \
    -format UDZO -ov "$DMG_PATH"
codesign --force --timestamp --sign "$SIGNING_IDENTITY" "$DMG_PATH"
codesign --verify --verbose=2 "$DMG_PATH"

notarize "$DMG_PATH" dmg
xcrun stapler staple -v "$DMG_PATH"
xcrun stapler validate -v "$DMG_PATH"
/usr/sbin/spctl --assess --type open --verbose=2 \
    --context context:primary-signature "$DMG_PATH"

# Verify what users actually receive, not just the pre-image staging tree.
MOUNT_POINT="$TMP_ROOT/mounted"
mkdir -p "$MOUNT_POINT"
hdiutil attach -readonly -nobrowse -noautoopen \
    -mountpoint "$MOUNT_POINT" "$DMG_PATH" >/dev/null
codesign --verify --deep --strict --verbose=2 "$MOUNT_POINT/oMLX.app"
/usr/bin/syspolicy_check distribution "$MOUNT_POINT/oMLX.app"
hdiutil detach "$MOUNT_POINT" >/dev/null
MOUNT_POINT=""

(
    cd "$OUTPUT_DIR"
    shasum -a 256 "$DMG_NAME" >"$DMG_NAME.sha256"
)

echo "Release artifact ready: $DMG_PATH"
echo "Checksum: $DMG_PATH.sha256"
