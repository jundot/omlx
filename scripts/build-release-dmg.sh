#!/bin/bash
# build-release-dmg.sh — build, sign, notarize, and LEAVE NO APP COPIES BEHIND.
#
# Every `build.sh` run stages app bundles (`build/Build/Products/*/oMLX.app`,
# `build/SlimStage/oMLX.app`, sometimes `build/Stage/...`). LaunchServices
# registers each one, so the user sees extra oMLX icons in Launchpad after a
# build — measured three times now. `package_dmg.sh` only wraps the app, so the
# cleanup belongs here: remove the staged bundles and unregister their paths,
# then rebuild the LaunchServices database and repaint the Dock.
#
# Usage: scripts/build-release-dmg.sh [--install]
#   --install   also install the DMG at this Mac (SSH/scp for peers is fine too)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

LSREGISTER=/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister
IDENTITY="Developer ID Application: Andreas Kolb (YB894QRG4W)"
STATAGE="$ROOT/apps/omlx-mac/build/SlimStage/oMLX.app"

clean_staged_apps() {
    # Deregister first, then delete: a record for a deleted path is what leaves
    # a dead icon behind.
    "$LSREGISTER" -dump 2>/dev/null \
        | grep -o "path:.*$ROOT[^ ]*\.app" | sed 's|path: *||' | sort -u \
        | while read -r bundle; do
            [ -n "$bundle" ] || continue
            "$LSREGISTER" -u "$bundle" >/dev/null 2>&1
        done
    find "$ROOT/apps/omlx-mac/build" -maxdepth 4 -name '*.app' -exec rm -rf {} + 2>/dev/null
    find "$ROOT/build" -maxdepth 3 -name '*.app' -exec rm -rf {} + 2>/dev/null
    "$LSREGISTER" -gc -f -r -domain user -domain local >/dev/null 2>&1
    killall Dock >/dev/null 2>&1
    echo "build-release-dmg: staged app copies removed, LaunchServices refreshed"
}

clean_staged_apps
OMLX_NEXT_OUT="$ROOT/apps/omlx-mac/build/SlimStage" \
    apps/omlx-mac/Scripts/build.sh release --slim || exit 1
APP_PATH="$STATAGE" OUT_DIR="$ROOT/build" \
    SIGN_IDENTITY="$IDENTITY" NOTARY_PROFILE=omlx-notary \
    apps/omlx-mac/Scripts/package_dmg.sh || exit 1
clean_staged_apps

DMG="$ROOT/build/oMLX-$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' \
    "$ROOT/apps/omlx-mac/Resources/Info.plist" 2>/dev/null || echo 0.0.0).dmg"
echo "build-release-dmg: $(shasum -a 256 "$DMG" | cut -d' ' -f1)  $DMG"
echo "build-release-dmg: apps on disk now: $(ls -d /Applications/oMLX*.app 2>/dev/null | wc -l | tr -d ' ')" \
     "| LS records: $("$LSREGISTER" -dump 2>/dev/null | grep -c 'oMLX\.app')"

if [ "${1:-}" = "--install" ] && [ -f "$DMG" ]; then
    cp "$DMG" /tmp/oMLX-install.dmg
    bash "$ROOT/scripts/../scripts/cleanup-cluster-macs.sh" >/dev/null 2>&1 || true
    hdiutil attach -nobrowse -readonly /tmp/oMLX-install.dmg >/dev/null 2>&1
    rm -rf /Applications/oMLX.app
    ditto /Volumes/oMLX/oMLX.app /Applications/oMLX.app
    hdiutil detach /Volumes/oMLX >/dev/null 2>&1
    rm -f /tmp/oMLX-install.dmg
    xattr -dr com.apple.quarantine /Applications/oMLX.app 2>/dev/null
    echo "build-release-dmg: installed $(shasum -a 256 /Applications/oMLX.app/Contents/MacOS/oMLX | cut -d' ' -f1)"
fi