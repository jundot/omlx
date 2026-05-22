#!/usr/bin/env bash
# build.sh — produce a runnable oMLX-next.app for local manual testing.
#
# This is the side-by-side Swift bundle path called out in plan.md §5.
# Pipeline:
#   1. (optional) rebuild venvstacks export if sources have drifted
#   2. xcodebuild with `-resolvePackageDependencies` so SPM deps pick up
#      any new minor/patch within the pin range each build
#   3. stage the Swift `.app` and embed Python layers from the donor
#   4. uv-sync an isolated bundle venv and overlay any packages that
#      have drifted from the donor (auto-detected via dist-info versions)
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
#   apps/omlx-mac/Scripts/build.sh release --no-sync  # skip uv sync + overlay
#                                                       (use donor layer as-is)
#   apps/omlx-mac/Scripts/build.sh --rebuild-donor    # force venvstacks rebuild
#   apps/omlx-mac/Scripts/build.sh --no-rebuild-donor # never rebuild; use
#                                                       existing donor even if stale
#
# Env overrides:
#   OMLX_DONOR_APP=/path/to/oMLX.app    # explicit donor (bypasses venvstacks)
#   OMLX_NEXT_OUT=/path/to/output_dir   # final stage location
#   UV_BIN=/path/to/uv                  # explicit uv binary (default: PATH lookup)
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
NO_SYNC=0
REBUILD_DONOR=auto    # auto | force | never
shift || true
for arg in "$@"; do
    case "$arg" in
        --bare) BARE=1 ;;
        --no-sync) NO_SYNC=1 ;;
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

# --- Overlay diverging packages from a fresh uv-synced 3.11 venv ---------
#
# Even when the donor is freshly rebuilt above, the bundle venv path is
# kept because:
#   1. With --no-rebuild-donor or an explicit OMLX_DONOR_APP, the donor
#      can still be stale relative to the worktree's pyproject.toml.
#   2. uv sync is fast (~10–30s) versus a full venvstacks rebuild
#      (~5–10 min), so it's the right tool for tight iteration on a
#      single dependency without re-locking the entire stack.
#
# Drift detection is automatic: every dist-info under the fresh venv is
# compared (by Version field) against the donor's dist-info; any package
# whose version has moved gets overlaid. No more manual OVERLAY_PKGS list.

_extract_metadata_field() {
    # $1: METADATA path, $2: field name (case-insensitive)
    grep -m1 -i "^$2: " "$1" 2>/dev/null | sed -E "s/^[^:]+:[[:space:]]*//"
}

_normalize_pkg_name() {
    # PEP 503 normalized name: lowercase, dashes/underscores/periods → dash
    echo "$1" | tr '[:upper:]' '[:lower:]' | tr '_.' '--'
}

_find_donor_dist_info() {
    # $1: layer site-packages, $2: PEP 503 normalized name
    # Returns the *.dist-info directory path or empty string. Matches
    # both dash and underscore variants and is case-insensitive so
    # original-cased dist-info dirs (e.g. Pillow-9.5.0.dist-info) match
    # the normalized lookup key.
    local site="$1" name="$2" name_us found
    name_us="$(echo "$name" | tr '-' '_')"
    found="$(find "$site" -maxdepth 1 -type d -iname "${name}-*.dist-info" -print -quit 2>/dev/null || true)"
    [ -z "$found" ] && \
        found="$(find "$site" -maxdepth 1 -type d -iname "${name_us}-*.dist-info" -print -quit 2>/dev/null || true)"
    echo "$found"
}

_overlay_one() {
    # Overlay a single drifted package from the fresh venv onto the donor
    # mlx layer. $1: PEP 503 name, $2: fresh dist-info, $3: donor site
    local name="$1" fresh_dist="$2" donor_site="$3"
    local fresh_site
    fresh_site="$(dirname "$fresh_dist")"

    # Determine top-level entries to copy. top_level.txt is best-effort;
    # not every wheel writes it. Fall back to RECORD parsing when absent.
    local top_levels=()
    if [ -f "$fresh_dist/top_level.txt" ]; then
        while IFS= read -r line; do
            [ -n "$line" ] && top_levels+=("$line")
        done < "$fresh_dist/top_level.txt"
    fi

    # Even if top_level.txt lists names, single-module packages may live
    # only as <name>.py at the site-packages root, so also try the
    # normalized name as a directory + .py fallback.
    if [ "${#top_levels[@]}" -eq 0 ]; then
        top_levels+=("$(echo "$name" | tr '-' '_')")
    fi

    # Drop stale donor dist-info (case-insensitive, both dash and underscore).
    local name_us
    name_us="$(echo "$name" | tr '-' '_')"
    find "$donor_site" -maxdepth 1 -type d \
        \( -iname "${name}-*.dist-info" -o -iname "${name_us}-*.dist-info" \) \
        -exec rm -rf {} + 2>/dev/null || true

    # Copy each top-level entry (directory or single-file .py)
    local copied=0
    for entry in "${top_levels[@]}"; do
        if [ -d "$fresh_site/$entry" ]; then
            rm -rf "$donor_site/$entry"
            rsync -a --exclude='__pycache__' --exclude='*.pyc' \
                "$fresh_site/$entry/" "$donor_site/$entry/"
            copied=1
        elif [ -f "$fresh_site/$entry.py" ]; then
            rm -f "$donor_site/$entry.py"
            cp -p "$fresh_site/$entry.py" "$donor_site/$entry.py"
            copied=1
        fi
    done

    # Copy the fresh dist-info verbatim (preserves RECORD, METADATA, etc.)
    rsync -a "$fresh_dist/" "$donor_site/$(basename "$fresh_dist")/"
    [ "$copied" -eq 1 ] && return 0
    return 1
}

if [ "$NO_SYNC" -eq 0 ]; then
    UV_BIN="${UV_BIN:-$(command -v uv || true)}"
    if [ -z "$UV_BIN" ]; then
        for candidate in /opt/homebrew/bin/uv "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
            if [ -x "$candidate" ]; then UV_BIN="$candidate"; break; fi
        done
    fi
    [ -n "$UV_BIN" ] || die "uv not found — install via Homebrew (brew install uv), set UV_BIN, or pass --no-sync."

    BUNDLE_VENV="$BUILD_DIR/venv"
    log "Syncing bundle venv at $BUNDLE_VENV (Python 3.11)…"
    UV_PROJECT_ENVIRONMENT="$BUNDLE_VENV" "$UV_BIN" sync \
        --python 3.11 \
        --project "$REPO_ROOT" \
        >"$BUILD_DIR/uv-sync.log" 2>&1 \
        || { tail -40 "$BUILD_DIR/uv-sync.log" >&2; die "uv sync failed; full log: $BUILD_DIR/uv-sync.log"; }

    VENV_SITE="$BUNDLE_VENV/lib/python3.11/site-packages"
    [ -d "$VENV_SITE" ] || die "Expected $VENV_SITE after uv sync — check $BUILD_DIR/uv-sync.log."

    MLX_LAYER_SITE="$FRAMEWORKS_DIR/framework-mlx-framework/lib/python3.11/site-packages"

    log "Detecting package drift between bundle venv and donor framework…"
    overlaid_count=0
    skipped_count=0
    skipped_missing=0
    for fresh_dist in "$VENV_SITE"/*.dist-info; do
        [ -d "$fresh_dist" ] || continue
        metadata="$fresh_dist/METADATA"
        [ -f "$metadata" ] || continue

        raw_name="$(_extract_metadata_field "$metadata" Name)"
        fresh_version="$(_extract_metadata_field "$metadata" Version)"
        [ -n "$raw_name" ] || continue
        [ -n "$fresh_version" ] || continue
        name="$(_normalize_pkg_name "$raw_name")"

        donor_dist="$(_find_donor_dist_info "$MLX_LAYER_SITE" "$name")"
        if [ -z "$donor_dist" ]; then
            # Package not present in donor. Skip rather than introduce
            # surprise additions — anything truly required for the
            # framework should be declared in venvstacks.toml.
            ((skipped_missing+=1)) || true
            continue
        fi
        donor_version="$(_extract_metadata_field "$donor_dist/METADATA" Version)"

        if [ "$fresh_version" = "$donor_version" ]; then
            ((skipped_count+=1)) || true
            continue
        fi

        log "  overlay: $raw_name $donor_version → $fresh_version"
        if _overlay_one "$name" "$fresh_dist" "$MLX_LAYER_SITE"; then
            ((overlaid_count+=1)) || true
        else
            warn "    no top-level files copied for $raw_name; check top_level.txt"
        fi
    done
    ok "Overlay: $overlaid_count drifted, $skipped_count in-sync, $skipped_missing not-in-donor (skipped)"
else
    warn "--no-sync set: donor framework-mlx-framework used as-is; newer pins won't apply."
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
