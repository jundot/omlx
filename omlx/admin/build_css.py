#!/usr/bin/env python3
"""Build script for Tailwind CSS compilation.

Downloads Tailwind v3 standalone CLI if needed and compiles CSS.
Requires no Node.js installation.

Usage:
    cd omlx/omlx/admin
    python build_css.py          # Build minified CSS
    python build_css.py --watch  # Watch mode for development
"""

import platform
import subprocess
import sys
import urllib.request
from pathlib import Path

TAILWIND_VERSION = "v3.4.17"
ADMIN_DIR = Path(__file__).parent


def get_binary_name() -> str:
    """Get platform-specific Tailwind CLI binary name."""
    machine = platform.machine().lower()
    system = platform.system().lower()
    if system == "darwin":
        arch = "arm64" if machine == "arm64" else "x64"
        return f"tailwindcss-macos-{arch}"
    elif system == "linux":
        arch = "arm64" if "aarch64" in machine else "x64"
        return f"tailwindcss-linux-{arch}"
    raise RuntimeError(f"Unsupported platform: {system} {machine}")


def ensure_binary() -> Path:
    """Download Tailwind standalone CLI if not present."""
    binary_name = get_binary_name()
    binary_path = ADMIN_DIR / binary_name

    if binary_path.exists():
        return binary_path

    url = (
        f"https://github.com/tailwindlabs/tailwindcss/releases/download/"
        f"{TAILWIND_VERSION}/{binary_name}"
    )
    print(f"Downloading Tailwind CSS {TAILWIND_VERSION}...")
    print(f"  {url}")
    urllib.request.urlretrieve(url, binary_path)
    binary_path.chmod(0o755)
    print(f"  Saved to {binary_path}")
    return binary_path


def main() -> None:
    binary = ensure_binary()

    input_css = ADMIN_DIR / "src" / "input.css"
    output_css = ADMIN_DIR / "static" / "css" / "tailwind.css"
    config = ADMIN_DIR / "tailwind.config.js"

    output_css.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(binary),
        "-i", str(input_css),
        "-o", str(output_css),
        "-c", str(config),
        "--minify",
    ]

    if "--watch" in sys.argv:
        cmd.append("--watch")
        print("Watching for changes... (Ctrl+C to stop)")

    print(f"Building: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode == 0 and "--watch" not in sys.argv:
        size = output_css.stat().st_size
        print(f"Output: {output_css} ({size:,} bytes)")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
