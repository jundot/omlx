#!/usr/bin/env bash
# sparkle_loopback.sh — local appcast smoke test for the Sparkle wiring.
#
# Builds two oMLX bundles (v0.0.1 and v0.0.2), code-signs both with
# a self-signed certificate held in a *temporary* keychain, EdDSA-signs
# v0.0.2's zip, serves the appcast over http://localhost:8765/, and
# launches v0.0.1. Re-run after the bundle-id rename to confirm Sparkle's
# flow hasn't regressed.
#
# Steps the test exercises:
#   1. v0.0.1 fetches the local appcast.
#   2. `SparkleDriver.didFindValidUpdate` fires → UpdateController.state
#      flips to .available → AppView button reads "Install & Restart".
#   3. Sparkle's standard panel pops up; user clicks Install & Restart.
#   4. Sparkle verifies BOTH the EdDSA signature (we did `sign_update`) and
#      the macOS code signature designated-requirement match (both bundles
#      were signed by the same self-signed identity).
#   5. v0.0.2 re-checks the same feed and `updaterDidNotFindUpdate` fires
#      → UpdateController.state flips back to .idle.
#
# Usage:
#   apps/omlx-mac/Scripts/sparkle_loopback.sh
#
# Requirements:
#   - Sparkle SPM dep already resolved (any prior debug build).
#   - python3 + openssl in PATH.
#   - Port 8765 free.
#
# Reversibility:
#   • The EdDSA private key is stored in the user's Keychain under
#     "https://sparkle-project.org" (Sparkle's convention). It's tiny, named,
#     and easy to remove via Keychain Access if you ever want it gone.
#   • The code-signing self-signed cert lives in a TEMP keychain at
#     $TEST_DIR/loopback.keychain-db. The script removes that keychain from
#     the search list and deletes the file on exit (even Ctrl-C). Nothing is
#     ever written to your login keychain.
#
# Press Ctrl-C in this terminal to stop the HTTP server and clean up.

set -euo pipefail

PORT=8765
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN_DIR="$PROJECT_DIR/build/SourcePackages/artifacts/sparkle/Sparkle/bin"
TEST_DIR="$PROJECT_DIR/build/SparkleLoopback"
FEED_DIR="$TEST_DIR/feed"

LIGHT_BLUE="\033[1;34m"
GREEN="\033[1;32m"
YELLOW="\033[1;33m"
RED="\033[1;31m"
RESET="\033[0m"

log()  { printf "${LIGHT_BLUE}[loopback]${RESET} %s\n" "$*"; }
ok()   { printf "${GREEN}[loopback]${RESET} %s\n" "$*"; }
warn() { printf "${YELLOW}[loopback]${RESET} %s\n" "$*"; }
die()  { printf "${RED}[loopback ERROR]${RESET} %s\n" "$*" >&2; exit 1; }

# --- preflight ------------------------------------------------------------

[ -x "$BIN_DIR/generate_keys" ] || die "Sparkle CLI tools missing at $BIN_DIR.
Run any debug build of oMLX first to resolve the SPM artifact."

if lsof -i ":$PORT" >/dev/null 2>&1; then
    die "Port $PORT is already in use. Stop the listener and re-run, or edit PORT."
fi

command -v python3 >/dev/null || die "python3 is required for the static file server."
command -v openssl >/dev/null || die "openssl is required to mint the self-signed cert."

# --- workspace ------------------------------------------------------------

rm -rf "$TEST_DIR"
mkdir -p "$FEED_DIR"

# --- keypair --------------------------------------------------------------

# generate_keys behaviour:
#   • `-p` prints the existing public key on stdout, OR an error on stdout
#     with exit 1 if no keypair exists. (Yes, the error is on stdout too —
#     so the exit code is the only reliable signal here.)
#   • With no args, generates a new keypair and stores the private key in
#     the macOS Keychain under "https://sparkle-project.org". A Keychain
#     access dialog will appear — click "Always Allow" so subsequent
#     sign_update calls don't prompt again.

read_pub_key() {
    "$BIN_DIR/generate_keys" -p 2>/dev/null
}

if PUB_KEY="$(read_pub_key)" && [ "${#PUB_KEY}" -ge 40 ]; then
    :  # existing keypair, PUB_KEY looks like a base64 EdDSA pub key
else
    log "No Sparkle keypair in Keychain — generating one (Keychain dialog will appear)."
    log "Click 'Always Allow' so sign_update later doesn't prompt again."
    "$BIN_DIR/generate_keys" >"$TEST_DIR/generate_keys.log" 2>&1 \
        || die "generate_keys failed; see $TEST_DIR/generate_keys.log"
    if ! PUB_KEY="$(read_pub_key)" || [ "${#PUB_KEY}" -lt 40 ]; then
        die "generate_keys ran but the public key still isn't readable.
You may need to approve the Keychain prompt and re-run."
    fi
fi
ok "Public key: $PUB_KEY"

# --- temp keychain + self-signed code-signing identity --------------------
#
# Why we need this: xcodebuild's ad-hoc signing (`CODE_SIGN_IDENTITY="-"`)
# produces linker-signed bundles with Info.plist=not bound, which fail
# SecCodeCheckValidity (`errSecCSBadResource`). Sparkle rejects such
# updates with "the app is also signed with Code Signing, which is
# corrupted." Re-signing both bundles with a stable cert fixes both the
# resource sealing and the designated-requirement match.
#
# Everything below is fully reversible: the cert + private key live in a
# scratch keychain at $KEYCHAIN_PATH; the trap at exit removes that
# keychain from the user's search list and deletes the file.

KEYCHAIN_PATH="$TEST_DIR/loopback.keychain-db"
KEYCHAIN_PASSWORD="loopback"
SIGN_IDENTITY=""        # set by mint_signing_cert
ORIGINAL_KEYCHAINS=""   # captured by setup_keychain so teardown can restore

setup_keychain() {
    rm -f "$KEYCHAIN_PATH"
    security create-keychain -p "$KEYCHAIN_PASSWORD" "$KEYCHAIN_PATH" \
        || die "Could not create temp keychain at $KEYCHAIN_PATH"
    # Disable auto-lock so signing both bundles can't time out mid-script.
    security set-keychain-settings "$KEYCHAIN_PATH"
    security unlock-keychain -p "$KEYCHAIN_PASSWORD" "$KEYCHAIN_PATH"

    # Snapshot the current search list (filtered for any stale loopback
    # keychain from a previous run) and prepend our temp keychain so
    # codesign picks up the new identity without explicit --keychain.
    ORIGINAL_KEYCHAINS="$(security list-keychains -d user | tr -d '"' \
        | grep -v "loopback.keychain-db" | xargs)"
    security list-keychains -d user -s "$KEYCHAIN_PATH" $ORIGINAL_KEYCHAINS
}

teardown_keychain() {
    if [ -n "$ORIGINAL_KEYCHAINS" ]; then
        security list-keychains -d user -s $ORIGINAL_KEYCHAINS >/dev/null 2>&1 || true
    fi
    security delete-keychain "$KEYCHAIN_PATH" >/dev/null 2>&1 || true
}

mint_signing_cert() {
    local cn="oMLX Loopback ($(date +%H%M%S))"
    local conf="$TEST_DIR/openssl.cnf"
    local key="$TEST_DIR/cert.key"
    local pem="$TEST_DIR/cert.pem"
    local p12="$TEST_DIR/cert.p12"
    local p12_pw="loopback"

    cat > "$conf" <<EOF
[req]
distinguished_name = dn
x509_extensions = v3_ca
prompt = no

[dn]
CN = $cn

[v3_ca]
basicConstraints       = critical, CA:false
keyUsage               = critical, digitalSignature
extendedKeyUsage       = critical, codeSigning
subjectKeyIdentifier   = hash
EOF

    openssl req -newkey rsa:2048 -nodes -x509 -days 30 \
        -config "$conf" -extensions v3_ca \
        -keyout "$key" -out "$pem" >/dev/null 2>&1 \
        || die "openssl failed to generate the self-signed code-signing cert."

    openssl pkcs12 -export -in "$pem" -inkey "$key" -out "$p12" \
        -passout pass:"$p12_pw" -name "$cn" -legacy >/dev/null 2>&1 \
        || die "openssl failed to bundle cert+key into PKCS#12."

    # Import into the temp keychain and grant codesign + security access.
    security import "$p12" -k "$KEYCHAIN_PATH" -P "$p12_pw" \
        -T /usr/bin/codesign -T /usr/bin/security >/dev/null \
        || die "security import failed."

    # Allow codesign to read the private key without re-prompting for the
    # keychain password every call. -S apple-tool:,apple: matches both
    # codesign (apple-tool) and any apple-internal helpers.
    security set-key-partition-list -S apple-tool:,apple: \
        -s -k "$KEYCHAIN_PASSWORD" "$KEYCHAIN_PATH" >/dev/null 2>&1 || true

    SIGN_IDENTITY="$cn"

    # Clean up the on-disk PEM/key/p12 — the cert+key live in the keychain
    # now, no reason to keep them as plain files.
    rm -f "$key" "$pem" "$p12" "$conf"
}

resign_bundle() {
    local app="$1"
    local main_exec="$app/Contents/MacOS/$(basename "$app" .app)"

    # We deliberately DO NOT pass `--options runtime` here. Hardened runtime
    # turns on Library Validation, which requires a real Apple-issued Team ID
    # (or system-trusted anchor) for cross-binary code signing. Self-signed
    # certs have empty Team ID and dyld v4 rejects them with "mapping process
    # and mapped file (non-platform) have different Team IDs", even when
    # every Mach-O in the bundle is signed by the exact same self-signed cert.
    # Production builds (Developer ID) will set --options runtime via build.py.
    #
    # `codesign --deep` only recurses into Frameworks/ and PlugIns/, NOT into
    # loose Mach-O files in Contents/MacOS/. Xcode Debug builds emit sibling
    # dylibs there (oMLX.debug.dylib, __preview.dylib) that keep their
    # original linker-signed identity after `--deep`. Fix: sign every nested
    # Mach-O inside-out — dylibs first, then the outer bundle.

    local inner_files=()
    while IFS= read -r -d '' f; do
        inner_files+=("$f")
    done < <(find "$app/Contents" -type f -print0)

    for f in "${inner_files[@]}"; do
        [ "$f" = "$main_exec" ] && continue
        file -b "$f" 2>/dev/null | grep -q "Mach-O" || continue
        codesign --force \
            --sign "$SIGN_IDENTITY" --keychain "$KEYCHAIN_PATH" \
            "$f" >/dev/null 2>&1 \
            || die "codesign failed for inner Mach-O: $f"
    done

    codesign --force --deep \
        --sign "$SIGN_IDENTITY" --keychain "$KEYCHAIN_PATH" \
        "$app" >/dev/null 2>&1 \
        || die "codesign failed for $app"

    codesign --verify --deep --strict "$app" >/dev/null 2>&1 \
        || die "codesign --verify failed for $app (sealed-resource mismatch)."
}

log "Creating temporary keychain at $KEYCHAIN_PATH (cleaned up on exit)…"
setup_keychain

# Register cleanup IMMEDIATELY so any subsequent failure (xcodebuild,
# codesign, sign_update) still tears down the keychain. SERVER_PID is
# set later; the kill is `|| true` so the trap is safe before it exists.
SERVER_PID=""
trap '
    [ -n "$SERVER_PID" ] && kill $SERVER_PID 2>/dev/null || true
    teardown_keychain
    printf "\n[loopback] cleanup done — http server stopped + temp keychain removed.\n"
' EXIT

mint_signing_cert
ok "Self-signed identity: $SIGN_IDENTITY"

# Share the project's existing derivedDataPath so SwiftPM doesn't have to
# re-resolve Sparkle twice. Each invocation produces the .app at the same
# location, so we ditto it out after each build.
SHARED_BUILD="$PROJECT_DIR/build"

build_version() {
    local short="$1"
    local build="$2"
    local out_app="$3"
    local logfile="$TEST_DIR/build-$short.log"
    xcodebuild \
        -project "$PROJECT_DIR/oMLX.xcodeproj" \
        -scheme oMLX \
        -configuration Debug \
        -destination 'platform=macOS' \
        -derivedDataPath "$SHARED_BUILD" \
        MARKETING_VERSION="$short" \
        CURRENT_PROJECT_VERSION="$build" \
        CODE_SIGN_IDENTITY="-" \
        CODE_SIGNING_REQUIRED=NO \
        CODE_SIGNING_ALLOWED=NO \
        build >"$logfile" 2>&1 \
        || { tail -40 "$logfile" >&2; die "xcodebuild failed for v$short (log: $logfile)"; }
    local src="$SHARED_BUILD/Build/Products/Debug/oMLX.app"
    [ -d "$src" ] || die "xcodebuild reported success but $src is missing."
    rm -rf "$out_app"
    ditto "$src" "$out_app"
}

log "Building v0.0.1 (will run from here)…"
V1_APP="$TEST_DIR/v0.0.1/oMLX.app"
mkdir -p "$(dirname "$V1_APP")"
build_version "0.0.1" "1" "$V1_APP"

log "Building v0.0.2 (the update target)…"
V2_APP="$TEST_DIR/v0.0.2/oMLX.app"
mkdir -p "$(dirname "$V2_APP")"
build_version "0.0.2" "2" "$V2_APP"

# --- patch BOTH Info.plists (must happen BEFORE code-signing) ------------
#
# Both bundles get the same SUFeedURL + SUPublicEDKey so the relaunched
# v0.0.2 has a working Sparkle wired to our local loopback feed. Without
# this, v0.0.2 would log "Fatal updater error: The EdDSA public key is not
# valid" because the source Info.plist ships with an empty SUPublicEDKey.

log "Patching v0.0.1 + v0.0.2 Info.plists with localhost feed + EdDSA pub key…"
for app_plist in "$V1_APP/Contents/Info.plist" "$V2_APP/Contents/Info.plist"; do
    /usr/libexec/PlistBuddy -c "Set :SUFeedURL http://localhost:$PORT/appcast.xml" "$app_plist"
    /usr/libexec/PlistBuddy -c "Set :SUPublicEDKey $PUB_KEY" "$app_plist"
done

# --- code-sign both bundles ----------------------------------------------

log "Code-signing v0.0.1 with ${SIGN_IDENTITY}…"
resign_bundle "$V1_APP"

log "Code-signing v0.0.2 with ${SIGN_IDENTITY}…"
resign_bundle "$V2_APP"

# --- zip + EdDSA-sign v0.0.2 ---------------------------------------------

V2_ZIP="$FEED_DIR/oMLX-0.0.2.zip"
log "Zipping v0.0.2…"
ditto -ck --rsrc --sequesterRsrc --keepParent "$V2_APP" "$V2_ZIP"

log "EdDSA-signing v0.0.2 (Keychain prompt may appear)…"
SIGN_OUTPUT="$("$BIN_DIR/sign_update" "$V2_ZIP")"
EDSIG="$(echo "$SIGN_OUTPUT" | sed -nE 's/.*sparkle:edSignature="([^"]+)".*/\1/p')"
[ -n "$EDSIG" ] || die "sign_update did not return an edSignature.
Output was: $SIGN_OUTPUT"
LEN="$(stat -f%z "$V2_ZIP")"
ok "Signature: ${EDSIG:0:24}… (len=$LEN bytes)"

# --- write appcast.xml ----------------------------------------------------

PUBDATE="$(date -u +"%a, %d %b %Y %H:%M:%S +0000")"

cat > "$FEED_DIR/appcast.xml" <<XML
<?xml version="1.0" encoding="utf-8"?>
<rss xmlns:sparkle="http://www.andymatuschak.org/xml-namespaces/sparkle" version="2.0">
  <channel>
    <title>oMLX local-loopback feed</title>
    <item>
      <title>v0.0.2 (loopback test)</title>
      <pubDate>$PUBDATE</pubDate>
      <sparkle:channel>stable</sparkle:channel>
      <sparkle:version>2</sparkle:version>
      <sparkle:shortVersionString>0.0.2</sparkle:shortVersionString>
      <sparkle:minimumSystemVersion>15.0</sparkle:minimumSystemVersion>
      <enclosure url="http://localhost:$PORT/oMLX-0.0.2.zip"
                 length="$LEN"
                 type="application/octet-stream"
                 sparkle:edSignature="$EDSIG" />
    </item>
  </channel>
</rss>
XML
ok "Appcast written to $FEED_DIR/appcast.xml"

# --- serve ----------------------------------------------------------------

cd "$FEED_DIR"
python3 -m http.server "$PORT" >"$TEST_DIR/http.log" 2>&1 &
SERVER_PID=$!
cd - >/dev/null

# Tiny grace period so the server is accepting before we open the app.
sleep 1
curl -sf "http://localhost:$PORT/appcast.xml" >/dev/null \
    || die "Local HTTP server failed to start; see $TEST_DIR/http.log."

# --- go ------------------------------------------------------------------

echo
echo "============================================================"
echo "  Local appcast: http://localhost:$PORT/appcast.xml"
echo "  v0.0.1 app:    $V1_APP"
echo "  v0.0.2 zip:    $V2_ZIP"
echo
echo "  Launching v0.0.1 now. In the app:"
echo "    1. Menubar → Open Dashboard."
echo "    2. Status → scroll to Updates → click 'Check Now'."
echo "    3. Expect: AppView button flips to 'Install & Restart' and"
echo "       Sparkle's panel pops up showing v0.0.2."
echo "    4. Click 'Install & Restart' in Sparkle's panel."
echo "    5. App relaunches into v0.0.2 — confirm via About screen."
echo
echo "  Press Ctrl-C in this terminal to stop the HTTP server."
echo "============================================================"
echo

# Kill any pre-existing app.omlx instance (e.g. the prior Stage build
# from build.sh) so LaunchServices doesn't activate that one when we open
# the loopback path. `open -n` then forces a brand-new process from our
# explicit bundle path.
pkill -f "oMLX/Contents/MacOS/oMLX" 2>/dev/null || true
sleep 1
open -n "$V1_APP"

wait $SERVER_PID
