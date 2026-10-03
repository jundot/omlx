#!/usr/bin/env python3
"""Download vendored dependencies for offline admin panel.

All libraries use permissive licenses (MIT/ISC/BSD/OFL) that allow bundling.
Run this script to download/update all CDN dependencies to static/.

Usage:
    cd omlx/omlx/admin
    python vendor_deps.py
"""

import re
import ssl
import urllib.request
from pathlib import Path

STATIC = Path(__file__).parent / "static"

# SSL context for HTTPS downloads
SSL_CTX = ssl.create_default_context()


def _download(url: str, dest: Path, description: str = "", optional: bool = False) -> bool:
    """Download a file from URL to destination path.

    Args:
        optional: If True, silently skip 404 errors (some font variants don't exist).

    Returns:
        True if downloaded or already exists, False if skipped.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip] {dest.name} (already exists)")
        return True
    label = description or dest.name
    print(f"  [download] {label} <- {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, context=SSL_CTX) as resp:
            dest.write_bytes(resp.read())
        return True
    except urllib.error.HTTPError as e:
        if optional and e.code == 404:
            print(f"  [skip] {dest.name} (not available)")
            return False
        raise


# =========================================================================
# JavaScript dependencies
# =========================================================================
JS_DEPS = {
    # Alpine.js 3.14.8 (MIT)
    "js/alpine.min.js": "https://cdn.jsdelivr.net/npm/alpinejs@3.14.8/dist/cdn.min.js",
    # Lucide Icons 0.453.0 (ISC)
    "js/lucide.min.js": "https://unpkg.com/lucide@0.453.0/dist/umd/lucide.min.js",
    # Marked 12.0.0 (MIT)
    "js/marked.umd.js": "https://cdn.jsdelivr.net/npm/marked@12.0.0/lib/marked.umd.js",
    # marked-highlight 2.0.6 (MIT)
    "js/marked-highlight.umd.js": "https://cdn.jsdelivr.net/npm/marked-highlight@2.0.6/lib/index.umd.js",
    # Highlight.js 11.9.0 core (BSD-3-Clause)
    "js/highlight.min.js": "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js",
    # Highlight.js language packs
    "js/hljs-python.min.js": "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/python.min.js",
    "js/hljs-javascript.min.js": "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/javascript.min.js",
    "js/hljs-bash.min.js": "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/bash.min.js",
    "js/hljs-json.min.js": "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/json.min.js",
    # KaTeX 0.16.9 (MIT)
    "js/katex.min.js": "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js",
    "js/katex-auto-render.min.js": "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js",
    # GridStack 13.3.0 (MIT) - dashboard layout grid, bundle includes drag and drop
    "js/gridstack-all.js": "https://cdn.jsdelivr.net/npm/gridstack@13.3.0/dist/gridstack-all.js",
}

# =========================================================================
# CSS dependencies
# =========================================================================
CSS_DEPS = {
    # Highlight.js themes (BSD-3-Clause)
    "css/hljs-github.min.css": "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css",
    "css/hljs-github-dark.min.css": "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css",
    # KaTeX CSS (MIT) - references fonts/ relative path
    "css/katex.min.css": "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css",
    # GridStack CSS (MIT) - v12+ derives column widths from CSS variables
    "css/gridstack.min.css": "https://cdn.jsdelivr.net/npm/gridstack@13.3.0/dist/gridstack.min.css",
}


def download_js_css() -> None:
    """Download JavaScript and CSS dependencies."""
    print("\n=== JavaScript Dependencies ===")
    for dest_rel, url in JS_DEPS.items():
        _download(url, STATIC / dest_rel)

    print("\n=== CSS Dependencies ===")
    for dest_rel, url in CSS_DEPS.items():
        _download(url, STATIC / dest_rel)


# =========================================================================
# KaTeX fonts
# =========================================================================
KATEX_VERSION = "0.16.9"
KATEX_FONT_BASE = f"https://cdn.jsdelivr.net/npm/katex@{KATEX_VERSION}/dist/fonts"

# All KaTeX font files referenced in katex.min.css
KATEX_FONTS = [
    "KaTeX_AMS-Regular",
    "KaTeX_Caligraphic-Bold",
    "KaTeX_Caligraphic-Regular",
    "KaTeX_Fraktur-Bold",
    "KaTeX_Fraktur-Regular",
    "KaTeX_Main-Bold",
    "KaTeX_Main-BoldItalic",
    "KaTeX_Main-Italic",
    "KaTeX_Main-Regular",
    "KaTeX_Math-BoldItalic",
    "KaTeX_Math-Italic",
    "KaTeX_SansSerif-Bold",
    "KaTeX_SansSerif-Italic",
    "KaTeX_SansSerif-Regular",
    "KaTeX_Script-Regular",
    "KaTeX_Size1-Regular",
    "KaTeX_Size2-Regular",
    "KaTeX_Size3-Regular",
    "KaTeX_Size4-Regular",
    "KaTeX_Typewriter-Regular",
]


def download_katex_fonts() -> None:
    """Download KaTeX font files (woff2 + ttf fallback)."""
    print("\n=== KaTeX Fonts ===")
    # Place in css/fonts/ so katex.min.css relative path works (url(fonts/...))
    fonts_dir = STATIC / "css" / "fonts"
    for font_name in KATEX_FONTS:
        for ext in ("woff2", "ttf"):
            url = f"{KATEX_FONT_BASE}/{font_name}.{ext}"
            _download(url, fonts_dir / f"{font_name}.{ext}", optional=True)


# =========================================================================
# Fonts
# =========================================================================
# None: the console uses the system font stack (see tokens.json typography
# families), so no webfont is vendored. Apple platforms resolve Latin and CJK
# from -apple-system; other clients fall back to the families listed in the
# token stack.


def main() -> None:
    print(f"Vendor directory: {STATIC}")
    download_js_css()
    download_katex_fonts()
    print("\n=== Done! ===")

    # Summary
    total = 0
    for p in STATIC.rglob("*"):
        if p.is_file() and p.suffix != ".svg":
            total += p.stat().st_size
    print(f"Total vendored size: {total / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
