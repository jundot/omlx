#!/usr/bin/env python3
"""
Fix xgrammar installation in OMLX's libexec Python environment.

Run this after `brew reinstall jundot/omlx/omlx --with-grammar` if structured
output still fails with one of:
  - Cannot find library: libxgrammar_bindings.dylib
  - Library not loaded: @rpath/libtvm_ffi.dylib
  - Killed: 9 on import xgrammar

The fix:
  1. Creates the missing RECORD file in xgrammar-*.dist-info/
  2. Adds @loader_path rpaths to both dylibs so they can find each other
  3. Re-signs both dylibs (required on macOS SIP)
  4. Patches load_binding.py to prepend DYLD_LIBRARY_PATH fallback

NOTE: This fix lives in Homebrew's Cellar. After `brew upgrade omlx`, re-run
`brew reinstall jundot/omlx/omlx --with-grammar` (which applies the formula's
post_install hook). If that still fails, run this script again.
"""

from __future__ import annotations

import glob
import os
import pathlib
import subprocess
import sys


def find_omlx_libexec() -> pathlib.Path:
    """Locate the active OMLX libexec directory."""
    head_matches = sorted(glob.glob("/opt/homebrew/Cellar/omlx/HEAD-*"), reverse=True)
    if head_matches:
        return pathlib.Path(head_matches[0]) / "libexec"
    stable_matches = sorted(glob.glob("/opt/homebrew/Cellar/omlx/*/"), reverse=True)
    if stable_matches:
        versioned = [p for p in stable_matches if p.split("/")[-2] != "HEAD"]
        if versioned:
            return pathlib.Path(versioned[0]) / "libexec"
    raise RuntimeError(
        "Could not find OMLX Cellar directory. Is OMLX installed?\n"
        "  brew install jundot/omlx/omlx"
    )


def find_site_packages(libexec: pathlib.Path) -> pathlib.Path:
    site = libexec / "lib" / "python3.11" / "site-packages"
    if site.is_dir():
        return site
    raise RuntimeError(f"Site-packages not found at {site}")


def fix_record_file(site: pathlib.Path) -> None:
    xg_info_dirs = list(site.glob("xgrammar-*.dist-info"))
    if not xg_info_dirs:
        raise RuntimeError("xgrammar dist-info not found. Reinstall with --with-grammar")
    xg_info_dir = xg_info_dirs[0]
    record_path = xg_info_dir / "RECORD"
    if record_path.exists():
        print(f"  [skip] RECORD already exists")
        return
    print(f"  Creating RECORD at {record_path}")
    entries: list[str] = []
    xg_dir = site / "xgrammar"
    for fpath in sorted(xg_dir.rglob("*")):
        if fpath.is_file() and ".dSYM" not in str(fpath):
            size = fpath.stat().st_size
            rel = str(fpath.relative_to(site))
            entries.append(f"{rel},{size},")
    for fpath in sorted(xg_info_dir.rglob("*")):
        if fpath.is_file():
            size = fpath.stat().st_size
            rel = str(fpath.relative_to(site))
            entries.append(f"{rel},{size},")
    record_path.write_text("\n".join(entries) + "\n")
    print(f"  [ok] Created {len(entries)} RECORD entries")


def _add_rpath(dylib: pathlib.Path, rpath: str) -> None:
    result = subprocess.run(
        ["install_name_tool", "-add_rpath", rpath, str(dylib)],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"  [ok] Added rpath {rpath} -> {dylib.name}")
    else:
        check = subprocess.run(["otool", "-l", str(dylib)], capture_output=True, text=True)
        if rpath in check.stdout:
            print(f"  [skip] rpath {rpath} already in {dylib.name}")
        else:
            print(f"  [warn] install_name_tool: {result.stderr.strip()}")


def fix_rpaths(site: pathlib.Path) -> None:
    xg_dylib = site / "xgrammar" / "libxgrammar_bindings.dylib"
    tvm_dylib = site / "tvm_ffi" / "lib" / "libtvm_ffi.dylib"
    if not xg_dylib.exists():
        raise RuntimeError(f"libxgrammar_bindings.dylib not found: {xg_dylib}")
    if not tvm_dylib.exists():
        raise RuntimeError(f"libtvm_ffi.dylib not found: {tvm_dylib}")
    print("  libxgrammar_bindings.dylib:")
    _add_rpath(xg_dylib, "@loader_path/")
    _add_rpath(xg_dylib, "@loader_path/../tvm_ffi/lib/")
    print("  libtvm_ffi.dylib:")
    _add_rpath(tvm_dylib, "@loader_path/")


def re_sign_dylibs(site: pathlib.Path) -> None:
    for name in ["xgrammar/libxgrammar_bindings.dylib", "tvm_ffi/lib/libtvm_ffi.dylib"]:
        dylib = site / name
        if not dylib.exists():
            print(f"  [skip] {name} not found")
            continue
        result = subprocess.run(
            ["codesign", "-f", "-s", "-", str(dylib)],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"  [ok] Re-signed {pathlib.Path(name).name}")
        else:
            print(f"  [warn] codesign failed: {result.stderr.strip()}")


def patch_load_binding(site: pathlib.Path) -> None:
    load_binding = site / "xgrammar" / "load_binding.py"
    if not load_binding.exists():
        print(f"  [skip] load_binding.py not found")
        return
    content = load_binding.read_text()
    if "DYLD_LIBRARY_PATH" in content:
        print("  [skip] load_binding.py already patched")
        return
    patch = (
        "import os, pathlib as _pp\n"
        "_xg_root = _pp.Path(__file__).parent.parent\n"
        "_tvm_root = _xg_root / \"tvm_ffi\"\n"
        "_lib_dir = str(_tvm_root / \"lib\")\n"
        "_dylib_dir = str(_xg_root / \"xgrammar\")\n"
        "_existing = os.environ.get(\"DYLD_LIBRARY_PATH\", \"\")\n"
        "_parts = [p for p in [_lib_dir, _dylib_dir, _existing] if p]\n"
        "os.environ[\"DYLD_LIBRARY_PATH\"] = \":\".join(_parts)\n\n"
    )
    load_binding.write_text(patch + content)
    print(f"  [ok] Patched load_binding.py")


def verify(site: pathlib.Path) -> bool:
    python = site.parents[1] / "bin" / "python3.11"
    if not python.exists():
        python = pathlib.Path("/opt/homebrew/bin/python3.11")
    env = {
        **os.environ,
        "DYLD_LIBRARY_PATH": f"{site / 'xgrammar'}:{site / 'tvm_ffi' / 'lib'}",
    }
    result = subprocess.run(
        [str(python), "-c", "import xgrammar; print('xgrammar OK -- structured output ready')"],
        capture_output=True, text=True, env=env,
    )
    if result.returncode == 0:
        print(f"\nOK: {result.stdout.strip()}")
        return True
    print(f"\nFAIL: {result.stderr.strip()}")
    return False


def main() -> int:
    print("OMLX xgrammar fix script")
    print("=" * 50)
    try:
        libexec = find_omlx_libexec()
        print(f"[1/5] libexec: {libexec}")
        site = find_site_packages(libexec)
        print(f"[2/5] site-packages: {site}")
        print("[3/5] Creating RECORD file...")
        fix_record_file(site)
        print("[4/5] Fixing dylib rpaths...")
        fix_rpaths(site)
        print("[5/5] Re-signing dylibs...")
        re_sign_dylibs(site)
        print("[~ ] Patching load_binding.py...")
        patch_load_binding(site)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 1
    ok = verify(site)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
