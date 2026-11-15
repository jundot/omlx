#!/bin/bash
# cleanup-cluster-macs.sh — put oMLX Macs back to a true first-run state.
#
# AGENTS.md §11: cleanup means processes AND disk, on every Mac. This script does
# both in one pass and is idempotent.
#
# What goes:
#   * every oMLX process (app, omlx-server, rank wrappers, dns-sd, helper, test
#     runners) and port 8099
#   * ~/.omlx — settings, logs, model settings, measurements, operations,
#     tuning presets, pairing keys, deployments, incidents, benchmarks and the
#     runtime staging cache — EXCEPT cluster/identity.json, whose loss only
#     re-rolls the node id and invalidates every pairing
#   * every oMLX.app copy (/Applications, repo build dirs, DerivedData, Trash,
#     ~/omlx-pr, mounted images, temp DMGs) and the runtime venv
#   * the LaunchServices registrations of every path found above, so Launchpad
#     cannot keep an icon for an app that no longer exists
#
# What stays: the models (~/models, ~/Models), this repository, the notarized
# DMG under build/ and cluster/identity.json.
#
# Usage:
#   scripts/cleanup-cluster-macs.sh              # this Mac
#   scripts/cleanup-cluster-macs.sh --all        # this Mac + the three peers
#   scripts/cleanup-cluster-macs.sh --all --keep-settings   # keep settings.json
#                                                            (keeps model dirs,
#                                                            but skips Welcome)
set -uo pipefail

PEERS=(172.16.1.12 172.16.1.13 172.16.1.14)
LSREGISTER=/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister
ALL=0
KEEP_SETTINGS=0
WITH_HELPER=0
for arg in "$@"; do
    case "$arg" in
        --all) ALL=1 ;;
        --keep-settings) KEEP_SETTINGS=1 ;;
        --with-helper) WITH_HELPER=1 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

clean_here() {
    local host
    host="$(scutil --get LocalHostName 2>/dev/null || hostname)"
    echo "=== $host ==="

    # ---- processes -------------------------------------------------------
    pkill -f 'MacOS/oMLX' 2>/dev/null
    pkill -f 'omlx-server' 2>/dev/null
    pkill -f 'omlx serve' 2>/dev/null
    pkill -f 'omlx-cluster-python' 2>/dev/null
    pkill -f 'dns-sd -R oMLX' 2>/dev/null
    pkill -f 'app\.omlx\.helper' 2>/dev/null
    pkill -f 'XCTRunner' 2>/dev/null
    sleep 2
    if pgrep -f 'MacOS/oMLX|omlx-server|omlx serve|dns-sd -R oMLX' >/dev/null 2>&1; then
        pkill -9 -f 'MacOS/oMLX|omlx-server|omlx serve|dns-sd -R oMLX' 2>/dev/null
        sleep 1
    fi
    echo "  processes left: $(pgrep -fl 'MacOS/oMLX|omlx-server|omlx serve' | wc -l | tr -d ' ')" \
         "| port 8099: $(lsof -nP -iTCP:8099 -sTCP:LISTEN 2>/dev/null | wc -l | tr -d ' ')"

    # ---- oMLX state ------------------------------------------------------
    if [ -d "$HOME/.omlx" ]; then
        find "$HOME/.omlx" -mindepth 1 -maxdepth 1 \
            ! -name cluster ! -name 'backup-cluster-*' -exec rm -rf {} + 2>/dev/null
        if [ "$KEEP_SETTINGS" -eq 0 ]; then
            find "$HOME/.omlx" -mindepth 1 -maxdepth 1 -name 'backup-cluster-*' \
                -exec rm -rf {} + 2>/dev/null
        fi
        find "$HOME/.omlx/cluster" -mindepth 1 -maxdepth 1 \
            ! -name 'identity.json' -exec rm -rf {} + 2>/dev/null
    fi
    echo "  ~/.omlx: $(ls -A "$HOME/.omlx" 2>/dev/null | tr '\n' ' ')" \
         "| cluster: $(ls -A "$HOME/.omlx/cluster" 2>/dev/null | tr '\n' ' ')"

    # ---- copies, runtime, caches ----------------------------------------
    for path in \
        /Applications/oMLX.app \
        /Applications/oMLX-*.app \
        "$HOME/Library/Application Support/oMLX" \
        "$HOME/Library/Application Support/oMLX Machine" \
        "$HOME/Library/Caches/oMLX" \
        "$HOME/Library/Logs/oMLX" \
        "$HOME/Library/Saved Application State/omlx.oMLX.savedState" \
        "$HOME/Library/Saved Application State/app.omlx.oMLX.savedState" \
        "$HOME/omlx-pr" \
        "$HOME/omlx-distributed" ; do
        [ -e "$path" ] || continue
        rm -rf "$path" 2>/dev/null && echo "  removed $path"
    done
    find "$HOME/Library/Developer/Xcode/DerivedData" -maxdepth 1 -name 'oMLX-*' \
        -exec rm -rf {} + 2>/dev/null
    # Build products inside a checkout are apps too, and Launchpad shows them.
    for root in "$HOME/Documents/PycharmProjects/omlx-cluster-assistant" "$HOME/omlx-pr"; do
        [ -d "$root" ] || continue
        find "$root/apps/omlx-mac/build" -maxdepth 4 -name '*.app' \
            -exec rm -rf {} + 2>/dev/null
        find "$root/build" -maxdepth 3 -name '*.app' -exec rm -rf {} + 2>/dev/null
    done
    find "$HOME/.Trash" -maxdepth 1 -name '*oMLX*' -exec rm -rf {} + 2>/dev/null
    for vol in /Volumes/oMLX*; do
        [ -d "$vol" ] && hdiutil detach "$vol" >/dev/null 2>&1
    done
    find "${TMPDIR:-/tmp}" /tmp -maxdepth 1 \( -name 'oMLX-*.dmg' -o -name 'omlx-dmg.*' \) \
        -exec rm -rf {} + 2>/dev/null

    # ---- privileged helper (root-owned; needs the one admin dialog) -----
    # The helper lives in /Library and belongs to root. A cleanup that silently
    # left it behind means the next launch never prompts again — which reads as
    # "the helper was never removed" and hides whether the wired limit is still
    # forced. Report it, and remove it only when the caller asks for the admin
    # dialog.
    if [ -e /Library/PrivilegedHelperTools/app.omlx.helper ] || \
       [ -e /Library/LaunchDaemons/app.omlx.helper.plist ]; then
        if [ "$WITH_HELPER" -eq 1 ]; then
            sudo launchctl bootout system/app.omlx.helper >/dev/null 2>&1
            sudo rm -f /Library/PrivilegedHelperTools/app.omlx.helper \
                       /Library/LaunchDaemons/app.omlx.helper.plist
            # Never claim a removal that did not happen: sudo over a
            # non-interactive SSH session has no terminal to ask for the
            # password, and the old code printed "removed" anyway.
            if [ -e /Library/PrivilegedHelperTools/app.omlx.helper ] || \
               [ -e /Library/LaunchDaemons/app.omlx.helper.plist ]; then
                echo "  helper NOT removed (sudo failed — no terminal for the password)"
                echo "    run this at the Mac, or with a cached sudo:"
                echo "    sudo launchctl bootout system/app.omlx.helper; sudo rm -f /Library/PrivilegedHelperTools/app.omlx.helper /Library/LaunchDaemons/app.omlx.helper.plist"
            else
                echo "  removed the privileged helper"
            fi
        else
            echo "  helper STILL INSTALLED (root-owned): add --with-helper, or run"
            echo "    sudo launchctl bootout system/app.omlx.helper"
            echo "    sudo rm -f /Library/PrivilegedHelperTools/app.omlx.helper /Library/LaunchDaemons/app.omlx.helper.plist"
        fi
    fi

    # ---- LaunchServices / Launchpad -------------------------------------
    # `-gc` alone keeps records for paths that no longer exist, which is what
    # leaves dead icons behind. Unregister each stale path explicitly, then
    # rebuild the database and let the Dock repaint.
    "$LSREGISTER" -dump 2>/dev/null \
        | grep -o 'path:.*oMLX[^ ]*\.app' | sed 's|path: *||' | sort -u \
        | while read -r bundle; do
            [ -n "$bundle" ] || continue
            "$LSREGISTER" -u "$bundle" >/dev/null 2>&1
        done
    "$LSREGISTER" -gc -f -r -domain user -domain local >/dev/null 2>&1
    killall Dock >/dev/null 2>&1
    sleep 1

    # ---- verify ---------------------------------------------------------
    echo "  apps on disk: $(ls -d /Applications/oMLX*.app 2>/dev/null | wc -l | tr -d ' ')" \
         "| LS records: $("$LSREGISTER" -dump 2>/dev/null | grep -c 'oMLX\.app')"
}

if [ "$ALL" -eq 1 ]; then
    for host in "${PEERS[@]}"; do
        scp -q -o BatchMode=yes "$0" "andreaskolb@$host:/tmp/cleanup-cluster-macs.sh" \
            && ssh -o BatchMode=yes "andreaskolb@$host" \
                "bash /tmp/cleanup-cluster-macs.sh $([ "$KEEP_SETTINGS" -eq 1 ] && echo --keep-settings); rm -f /tmp/cleanup-cluster-macs.sh" &
    done
    clean_here
    wait
else
    clean_here
fi