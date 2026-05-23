#!/usr/bin/env bash
# Regenerate the glyph PNG embedded in the AppIcon.icon bundle from the
# canonical brand SVG.
#
# The .icon bundle (Tahoe-style icon-composer format) holds a single
# foreground glyph layer plus the icon.json manifest. The manifest is
# authored once via Xcode's Icon Composer GUI and committed as-is; this
# script only refreshes the rasterized glyph when the brand mark changes.
#
# We strip the SVG's background rect and drop-shadow before rasterizing
# so the glyph sits cleanly on whatever tile macOS 26 (or the .icon
# fill) renders behind it — the embedded rounded-rect background would
# stack with the system tile and look like an icon nested inside an icon.
#
# Source of truth: docs/images/icon-rounded-light.svg
# Output:          apps/omlx-mac/Resources/AppIcon.icon/Assets/omlx_glyph_1024.png
#
# Re-run after editing the brand SVG; commit the resulting PNG alongside.
# If the icon.json layer name/file-name change, also update the manifest.
#
# Requires: rsvg-convert (brew install librsvg).

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
REPO="$(cd "$ROOT/../.." && pwd)"

SRC="$REPO/docs/images/icon-rounded-light.svg"
OUT="$ROOT/Resources/AppIcon.icon/Assets/omlx_glyph_1024.png"

if ! command -v rsvg-convert >/dev/null 2>&1; then
    echo "error: rsvg-convert not found. Install with: brew install librsvg" >&2
    exit 1
fi
if [ ! -f "$SRC" ]; then
    echo "error: source SVG missing at $SRC" >&2
    exit 1
fi
if [ ! -d "$(dirname "$OUT")" ]; then
    echo "error: AppIcon.icon Assets directory missing at $(dirname "$OUT")" >&2
    exit 1
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# Strip the background rect (and its drop-shadow filter) so only the
# glyph paths remain. macOS 26's icon-composer system provides the tile.
sed -e '/<rect /d' \
    -e 's| filter="url(#shadow)"||' \
    "$SRC" > "$TMP/glyph.svg"

rsvg-convert -w 1024 -h 1024 "$TMP/glyph.svg" -o "$OUT"

echo "Regenerated $OUT"
