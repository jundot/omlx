#!/usr/bin/env bash
# Build/stage the DS4 support tree consumed by the macOS app bundle.
#
# The runtime app never builds or fetches DS4. Release builders run this script
# ahead of apps/omlx-mac/Scripts/build.sh so the bundle can copy the validated
# packaging/DS4Support tree into Contents/Resources/DS4Support.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/build-ds4-support.sh [options]

Build ds4-server from the pinned manifest source commit and stage the validated
runtime support tree for the macOS app bundle.

Options:
  --source VALUE   ds4 source checkout or git URL (default: manifest source_repo;
                   also OMLX_DS4_SOURCE_DIR)
  --commit SHA     source commit to fetch/verify (default: manifest source_commit;
                   also OMLX_DS4_COMMIT)
  --out DIR        Destination support tree (default: $OMLX_DS4_SUPPORT_OUT or
                   packaging/DS4Support)
  --manifest FILE  DS4 source manifest (default: bundled manifest.json)
  --skip-build     Do not run make; validate/copy an already-built ds4-server
                   from the source tree (also OMLX_DS4_SKIP_BUILD=1)
  -h, --help       Show this help

Environment:
  PYTHON_BIN       Python used to run omlx.ds4_support validation/copy helper
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

log() {
    printf '==> %s\n' "$*"
}

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"

SOURCE="${OMLX_DS4_SOURCE_DIR:-}"
COMMIT="${OMLX_DS4_COMMIT:-}"
OUT_DIR="${OMLX_DS4_SUPPORT_OUT:-$REPO_ROOT/packaging/DS4Support}"
MANIFEST_PATH=""
SKIP_BUILD="${OMLX_DS4_SKIP_BUILD:-0}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --source)
            [ "$#" -ge 2 ] || die "--source requires a path or git URL"
            SOURCE="$2"
            shift 2
            ;;
        --commit)
            [ "$#" -ge 2 ] || die "--commit requires a git commit"
            COMMIT="$2"
            shift 2
            ;;
        --out)
            [ "$#" -ge 2 ] || die "--out requires a directory"
            OUT_DIR="$2"
            shift 2
            ;;
        --manifest)
            [ "$#" -ge 2 ] || die "--manifest requires a file"
            MANIFEST_PATH="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

mkdir -p "$(dirname -- "$OUT_DIR")"
OUT_DIR="$(CDPATH= cd -- "$(dirname -- "$OUT_DIR")" && pwd)/$(basename -- "$OUT_DIR")"

if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
        PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
    else
        PYTHON_BIN="$(command -v python3)"
    fi
fi
[ -x "$PYTHON_BIN" ] || die "PYTHON_BIN is not executable: $PYTHON_BIN"

if [ "$SKIP_BUILD" = "1" ]; then
    log "Skipping DS4 build; using existing ds4-server from ${SOURCE:-manifest source}"
elif [ -n "$SOURCE" ]; then
    log "Building ds4-server from $SOURCE"
else
    log "Building ds4-server from pinned manifest source"
fi
rm -rf "$OUT_DIR"
PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" - "$OUT_DIR" "$SOURCE" "$COMMIT" "$MANIFEST_PATH" "$SKIP_BUILD" <<'PY'
import sys
from omlx.ds4_support import build_ds4_support_from_source

destination, source, commit, manifest, skip_build = sys.argv[1:6]
result = build_ds4_support_from_source(
    destination_dir=destination,
    source=source or None,
    commit=commit or None,
    manifest_path=manifest or None,
    skip_build=skip_build == "1",
    overwrite=True,
)
print(f"copied {len(result.copied_files)} DS4 support files")
PY

log "DS4 support tree ready: $OUT_DIR"
