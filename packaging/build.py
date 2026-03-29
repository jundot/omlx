#!/usr/bin/env python3
"""
Build script for oMLX macOS app.

This script:
1. Builds venvstacks layers (runtime + framework + app)
2. Creates macOS .app bundle
3. Packages into DMG

Usage:
    python build.py              # Build everything
    python build.py --skip-venv  # Skip venvstacks build (use existing)
    python build.py --dmg-only   # Only create DMG from existing build
"""

import argparse
import os
import platform
import plistlib
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import re

SCRIPT_DIR = Path(__file__).parent
BUILD_DIR = SCRIPT_DIR / "_build"
EXPORT_DIR = SCRIPT_DIR / "_export"
DIST_DIR = SCRIPT_DIR / "dist"
WHEELS_DIR = SCRIPT_DIR / "_wheels"
APP_NAME = "oMLX"
APP_BUNDLE = f"{APP_NAME}.app"


def _read_version() -> str:
    """Read version from omlx/_version.py (single source of truth)."""
    version_file = SCRIPT_DIR.parent / "omlx" / "_version.py"
    content = version_file.read_text()
    match = re.search(r'__version__\s*=\s*"([^"]+)"', content)
    if not match:
        raise RuntimeError(f"Cannot find __version__ in {version_file}")
    return match.group(1)


VERSION = _read_version()


def clean_all(preserve_venv: bool = False):
    """Remove build artifacts and caches for a clean build.

    Args:
        preserve_venv: When True, keep _build/, _export/, _wheels/ and
            requirements/ so that --skip-venv can reuse them.
    """
    print("\n[Clean] Removing build artifacts...")

    venv_dirs = {BUILD_DIR, EXPORT_DIR, WHEELS_DIR, SCRIPT_DIR / "requirements"}

    dirs_to_clean = [
        BUILD_DIR,      # _build/
        EXPORT_DIR,     # _export/
        WHEELS_DIR,     # _wheels/
        DIST_DIR,       # dist/
        SCRIPT_DIR / "requirements",  # venvstacks lock files
    ]

    files_to_clean = [
        SCRIPT_DIR / "_venvstacks_resolved.toml",
    ]

    def _rm_onerror(func, path, exc_info):
        """Handle .DS_Store, permission errors, and non-empty dirs during rmtree."""
        os.chmod(path, 0o777)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            func(path)

    for d in dirs_to_clean:
        if preserve_venv and d in venv_dirs:
            continue
        if d.exists():
            shutil.rmtree(d, onerror=_rm_onerror)
            print(f"  Removed {d.relative_to(SCRIPT_DIR)}/")

    for f in files_to_clean:
        if preserve_venv and f.name == "_venvstacks_resolved.toml":
            continue
        if f.exists():
            f.unlink()
            print(f"  Removed {f.relative_to(SCRIPT_DIR)}")

    print("  ✓ Clean complete\n")


def run_cmd(cmd: list, cwd: Path = None, check: bool = True):
    """Run a command and print output."""
    print(f"  → {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=False)
    if check and result.returncode != 0:
        print(f"  ✗ Command failed with code {result.returncode}")
        sys.exit(1)
    return result


def _resolve_mlx_version(toml_path: Path) -> str:
    """Resolve the mlx version that venvstacks locked.

    Reads the locked requirements file to find the exact mlx version,
    falling back to the latest version from PyPI if no lock file exists.
    """
    req_file = (
        SCRIPT_DIR
        / "requirements"
        / "framework-mlx-framework"
        / "requirements-framework-mlx-framework-macosx_arm64.txt"
    )
    if req_file.exists():
        import re as _re

        content = req_file.read_text()
        match = _re.search(r"^mlx==(\S+)", content, _re.MULTILINE)
        if match:
            return match.group(1)

    # No lock file yet — query PyPI for the latest version
    import json
    import urllib.request

    data = json.loads(
        urllib.request.urlopen("https://pypi.org/pypi/mlx/json").read()
    )
    return data["info"]["version"]


def swap_platform_wheels(
    export_dir: Path, macos_target: str, python_version: str = "3.11"
):
    """Replace mlx and mlx-metal in exported venvstacks with platform-specific wheels.

    Downloads the wheels for the given macOS target (e.g. "26.0") and replaces
    the existing packages in the framework layer's site-packages. This allows
    building on macOS 15 while targeting macOS 26 wheels that contain
    M5 Neural Accelerator matmul kernels.
    """
    import zipfile

    site_packages = (
        export_dir
        / "framework-mlx-framework"
        / "lib"
        / f"python{python_version}"
        / "site-packages"
    )
    if not site_packages.exists():
        print(f"  ✗ site-packages not found: {site_packages}")
        sys.exit(1)

    platform_tag = f"macosx_{macos_target.replace('.', '_')}_arm64"
    toml_path = SCRIPT_DIR / "venvstacks.toml"
    mlx_version = _resolve_mlx_version(toml_path)
    packages = ["mlx", "mlx-metal"]

    print(f"\n  Swapping mlx/mlx-metal to {platform_tag} (v{mlx_version})...")

    # Download platform-specific wheels
    wheels_tmp = SCRIPT_DIR / "_platform_wheels"
    if wheels_tmp.exists():
        shutil.rmtree(wheels_tmp)
    wheels_tmp.mkdir()

    for pkg in packages:
        run_cmd([
            sys.executable, "-m", "pip", "download",
            f"{pkg}=={mlx_version}",
            "--platform", platform_tag,
            f"--python-version={python_version}",
            "--only-binary", ":all:",
            "--no-deps",
            "-d", str(wheels_tmp),
        ])

    # Remove existing mlx/mlx-metal from site-packages
    for item in site_packages.iterdir():
        name = item.name.lower()
        if name in ("mlx", "mlx_metal") or name.startswith(
            ("mlx-", "mlx_metal-")
        ):
            if item.is_dir():
                shutil.rmtree(item)
                print(f"    Removed {item.name}")

    # Install downloaded wheels into site-packages
    for whl in wheels_tmp.glob("*.whl"):
        print(f"    Installing {whl.name}")
        with zipfile.ZipFile(whl) as zf:
            zf.extractall(site_packages)

    # Cleanup
    shutil.rmtree(wheels_tmp)
    print(f"  ✓ Swapped to {platform_tag}")



def _parse_git_requirements(toml_path: Path) -> list[tuple[str, str]]:
    """Extract git-based requirements from venvstacks.toml.

    Returns list of (full_requirement_string, git_url) tuples.
    e.g. ("mlx-lm @ git+https://...@sha", "git+https://...@sha")
    """
    content = toml_path.read_text()
    # Match lines like: "mlx-lm @ git+https://github.com/...@commit"
    pattern = r'"([^"]*\s*@\s*(git\+https://[^""]*))"'
    return re.findall(pattern, content)


def _wheel_version(whl_path: Path) -> str:
    """Extract version from wheel filename (e.g. mlx_lm-0.30.6-py3-none-any.whl -> 0.30.6)."""
    parts = whl_path.stem.split("-")
    if len(parts) >= 2:
        return parts[1]
    return "0.0.0"


def _wheel_pkg_name(whl_path: Path) -> str:
    """Extract normalized package name from wheel filename."""
    return whl_path.stem.split("-")[0].replace("_", "-").lower()


def build_local_wheels():
    """Pre-build wheels for git-pinned packages.

    venvstacks/uv disables source builds, so git-pinned packages must be
    pre-built as wheels. This function:
    1. Parses git URLs from venvstacks.toml
    2. Builds wheels via pip
    3. Returns a mapping of package_name -> version for toml rewriting
    """
    print("\n[0/4] Building wheels for git packages...")

    toml_path = SCRIPT_DIR / "venvstacks.toml"
    git_reqs = _parse_git_requirements(toml_path)

    if not git_reqs:
        print("  No git requirements found, skipping.")
        return {}

    # Clean and recreate wheels dir for fresh builds
    if WHEELS_DIR.exists():
        shutil.rmtree(WHEELS_DIR)
    WHEELS_DIR.mkdir(parents=True)

    for full_req, git_url in git_reqs:
        pkg_name = full_req.split("@")[0].strip()
        print(f"  Building wheel for {pkg_name} ...")
        run_cmd([
            sys.executable, "-m", "pip", "wheel",
            git_url,
            "--no-deps",
            "-w", str(WHEELS_DIR),
        ])

    # Build version mapping from built wheels
    version_map = {}
    for whl in WHEELS_DIR.glob("*.whl"):
        name = _wheel_pkg_name(whl)
        version = _wheel_version(whl)
        version_map[name] = version
        print(f"    {name} == {version}")

    print(f"  ✓ {len(version_map)} wheel(s) built in {WHEELS_DIR}")
    return version_map


def _find_wheel_for_package(pkg_name: str) -> Path | None:
    """Find the built wheel file for a package name."""
    normalized = pkg_name.lower().replace("-", "_")
    for whl in WHEELS_DIR.glob("*.whl"):
        whl_name = whl.stem.split("-")[0].lower()
        if whl_name == normalized:
            return whl
    return None


def _write_engine_commits(omlx_pkg_dir: Path):
    """Write _engine_commits.json to the omlx package for runtime SHA display.

    Extracts commit SHAs from venvstacks.toml git URLs and writes them
    so _get_engine_info() can show clickable commit links in the admin dashboard.
    """
    import json

    toml_path = SCRIPT_DIR / "venvstacks.toml"
    git_reqs = _parse_git_requirements(toml_path)

    repo_urls = {
        "mlx-lm": "https://github.com/ml-explore/mlx-lm",
        "mlx-vlm": "https://github.com/Blaizzy/mlx-vlm",
        "mlx-embeddings": "https://github.com/Blaizzy/mlx-embeddings",
    }

    commits = {}
    for full_req, git_url in git_reqs:
        pkg_name = full_req.split("@")[0].strip().lower()
        # git_url format: git+https://github.com/ml-explore/mlx-lm@bcf6306...
        if "@" in git_url:
            commit = git_url.rsplit("@", 1)[1]
            if pkg_name in repo_urls:
                commits[pkg_name] = {
                    "commit": commit,
                    "url": repo_urls[pkg_name],
                }

    if commits:
        commits_file = omlx_pkg_dir / "_engine_commits.json"
        commits_file.write_text(json.dumps(commits, indent=2) + "\n")
        print(f"  Generated _engine_commits.json: {list(commits.keys())}")


def _create_resolved_toml(version_map: dict[str, str]) -> Path:
    """Create a temporary venvstacks.toml with git URLs replaced by local file:// paths.

    Git-built wheels have different hashes than PyPI releases of the same version,
    so we must point directly to the local wheel files to avoid hash mismatches.
    """
    toml_path = SCRIPT_DIR / "venvstacks.toml"
    content = toml_path.read_text()

    for full_req, git_url in _parse_git_requirements(toml_path):
        pkg_name = full_req.split("@")[0].strip()
        whl = _find_wheel_for_package(pkg_name)
        if whl:
            whl_uri = whl.resolve().as_uri()
            old_line = f'"{full_req}"'
            new_line = f'"{pkg_name} @ {whl_uri}"'
            content = content.replace(old_line, new_line)
            print(f"    {pkg_name} @ git+... → {whl.name}")

    resolved_path = SCRIPT_DIR / "_venvstacks_resolved.toml"
    resolved_path.write_text(content)
    return resolved_path


def _check_git_commit_sync():
    """Verify git commit SHAs match between pyproject.toml and venvstacks.toml.

    Aborts the build if any git-pinned package has different commits
    in the two files, preventing accidental stale builds.
    """
    pyproject_path = SCRIPT_DIR.parent / "pyproject.toml"
    venvstacks_path = SCRIPT_DIR / "venvstacks.toml"

    pyproject_reqs = {
        r[0].split("@")[0].strip().lower(): r[1]
        for r in _parse_git_requirements(pyproject_path)
    }
    venvstacks_reqs = {
        r[0].split("@")[0].strip().lower(): r[1]
        for r in _parse_git_requirements(venvstacks_path)
    }

    mismatches = []
    for pkg in pyproject_reqs:
        if pkg in venvstacks_reqs and pyproject_reqs[pkg] != venvstacks_reqs[pkg]:
            mismatches.append(
                f"  {pkg}:\n"
                f"    pyproject.toml:    {pyproject_reqs[pkg]}\n"
                f"    venvstacks.toml:   {venvstacks_reqs[pkg]}"
            )

    if mismatches:
        print("\n✗ Git commit mismatch between pyproject.toml and venvstacks.toml:")
        for m in mismatches:
            print(m)
        print("\nUpdate both files to the same commit before building.")
        sys.exit(1)


def build_venvstacks():
    """Build venvstacks layers."""
    print("\n[1/4] Building venvstacks layers...")

    _check_git_commit_sync()

    # Step 1: Build wheels from git-pinned packages
    version_map = build_local_wheels()

    # Step 2: Create resolved toml (git URLs → version pins)
    if version_map:
        print("\n  Resolving git requirements to version pins...")
        resolved_toml = _create_resolved_toml(version_map)
    else:
        resolved_toml = SCRIPT_DIR / "venvstacks.toml"

    # Local wheels args
    local_wheels_args = []
    if WHEELS_DIR.exists() and any(WHEELS_DIR.glob("*.whl")):
        local_wheels_args = ["--local-wheels", str(WHEELS_DIR)]

    # Step 3: Lock environments (always re-lock to match current wheels)
    print("\n  Locking environments...")
    lock_cmd = [
        "pipx", "run", "venvstacks", "lock",
        str(resolved_toml),
    ] + local_wheels_args
    if version_map:
        # Force re-lock when git packages changed (hashes will differ)
        lock_cmd += ["--reset-lock", "*"]
    else:
        lock_cmd += ["--if-needed"]
    run_cmd(lock_cmd)

    # Step 4: Build environments
    print("\n  Building environments (this may take a while)...")
    run_cmd([
        "pipx", "run", "venvstacks", "build",
        str(resolved_toml),
        "--no-lock",
    ] + local_wheels_args)

    # Step 5: Export to local directory for app bundle
    print("\n  Exporting environments...")
    if EXPORT_DIR.exists():
        shutil.rmtree(EXPORT_DIR)

    run_cmd([
        "pipx", "run", "venvstacks", "local-export",
        str(resolved_toml),
        "--output-dir", str(EXPORT_DIR),
    ])

    # Cleanup temporary toml
    if version_map and resolved_toml.exists():
        resolved_toml.unlink()

    # Install mlx-audio with --no-deps to avoid mlx-lm version conflict
    # (mlx-audio pins mlx-lm==0.31.1 but omlx uses a newer commit)
    fw_python = (
        EXPORT_DIR / "framework-mlx-framework" / "bin" / "python3"
    )
    if fw_python.exists():
        print("\n  Installing mlx-audio (--no-deps)...")
        run_cmd([
            str(fw_python), "-m", "pip", "install",
            "--no-deps", "mlx-audio==0.4.1",
        ])

    return EXPORT_DIR


def _create_c_launcher(macos_dir: Path, app_name: str):
    """Compile a native Mach-O launcher binary for macOS menubar app startup.

    A compiled binary (not a bash script) is required as CFBundleExecutable
    so that macOS LaunchServices properly grants WindowServer GUI access
    to the process.

    On macOS Tahoe, exec-trampoline launchers (CFBundleExecutable -> launcher
    -> exec python3) can end up in a NotVisible state for status bar apps.
    To avoid this, the launcher initializes Python in-process via Py_BytesMain
    instead of replacing itself with exec().

    The launcher:
    - Detects both Python/ (release) and Frameworks/ (dev) directories
    - Sets PYTHONHOME, PYTHONPATH, PYTHONDONTWRITEBYTECODE
    - Loads bundled libpython3.11.dylib and calls Py_BytesMain("-m omlx_app")
    - Shows an error dialog via osascript if startup fails
    """
    launcher_c = macos_dir / "_launcher.c"
    launcher_c.write_text(r'''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>
#include <errno.h>
#include <dlfcn.h>
#include <mach-o/dyld.h>

typedef int (*py_bytes_main_fn)(int, char **);

static void show_error(const char *msg) {
    char cmd[2048];
    snprintf(cmd, sizeof(cmd),
        "osascript -e 'display dialog \"%s\" buttons {\"OK\"} "
        "default button 1 with icon stop with title \"oMLX\"'",
        msg);
    system(cmd);
}

int main(int argc, char *argv[]) {
    char exe_buf[PATH_MAX];
    char resolved[PATH_MAX];
    uint32_t size = sizeof(exe_buf);

    if (_NSGetExecutablePath(exe_buf, &size) != 0) {
        show_error("Failed to get executable path.");
        return 1;
    }
    if (!realpath(exe_buf, resolved)) {
        show_error("Failed to resolve executable path.");
        return 1;
    }

    /* Trim executable name to get MacOS/ directory */
    char *slash = strrchr(resolved, '/');
    if (!slash) { show_error("Invalid path."); return 1; }
    *slash = '\0';
    char macos_dir[PATH_MAX];
    strncpy(macos_dir, resolved, sizeof(macos_dir) - 1);

    /* Trim MacOS to get Contents/ directory */
    slash = strrchr(resolved, '/');
    if (!slash) { show_error("Invalid bundle structure."); return 1; }
    *slash = '\0';
    char contents_dir[PATH_MAX];
    strncpy(contents_dir, resolved, sizeof(contents_dir) - 1);

    /* Detect Python layer directory: Python/ (release) or Frameworks/ (dev) */
    char layers_dir[PATH_MAX];
    snprintf(layers_dir, sizeof(layers_dir), "%s/Python", contents_dir);
    if (access(layers_dir, F_OK) != 0) {
        snprintf(layers_dir, sizeof(layers_dir), "%s/Frameworks", contents_dir);
        if (access(layers_dir, F_OK) != 0) {
            show_error("Python runtime not found in app bundle.");
            return 1;
        }
    }

    /* Set PYTHONHOME */
    char pythonhome[PATH_MAX];
    snprintf(pythonhome, sizeof(pythonhome), "%s/cpython-3.11", layers_dir);
    setenv("PYTHONHOME", pythonhome, 1);

    /* Set PYTHONPATH */
    char pythonpath[PATH_MAX * 4];
    snprintf(pythonpath, sizeof(pythonpath),
        "%s/Resources:%s/app-omlx-app/lib/python3.11/site-packages:"
        "%s/framework-mlx-framework/lib/python3.11/site-packages",
        contents_dir, layers_dir, layers_dir);
    setenv("PYTHONPATH", pythonpath, 1);

    /* Prevent .pyc generation at runtime */
    setenv("PYTHONDONTWRITEBYTECODE", "1", 1);

    /* Ensure bundled python3 exists (used later by server subprocesses). */
    char python_bin[PATH_MAX];
    snprintf(python_bin, sizeof(python_bin), "%s/python3", macos_dir);
    if (access(python_bin, X_OK) != 0) {
        show_error("Python executable not found in app bundle.");
        return 1;
    }

    /* Load bundled libpython and run -m omlx_app in-process (no exec trampoline). */
    char libpython[PATH_MAX];
    snprintf(libpython, sizeof(libpython), "%s/lib/libpython3.11.dylib", contents_dir);
    void *py = dlopen(libpython, RTLD_NOW | RTLD_GLOBAL);
    if (!py) {
        char err[1024];
        snprintf(err, sizeof(err), "Failed to load libpython: %s", dlerror());
        show_error(err);
        return 1;
    }

    py_bytes_main_fn py_bytes_main = (py_bytes_main_fn)dlsym(py, "Py_BytesMain");
    if (!py_bytes_main) {
        char err[1024];
        snprintf(err, sizeof(err), "Failed to resolve Py_BytesMain: %s", dlerror());
        show_error(err);
        return 1;
    }

    char *py_argv[] = {"oMLX", "-m", "omlx_app", NULL};
    int rc = py_bytes_main(3, py_argv);
    return rc;
}
''')

    launcher_bin = macos_dir / app_name
    result = subprocess.run(
        ["cc", "-arch", "arm64", "-mmacosx-version-min=15.0", "-O2",
         "-o", str(launcher_bin), str(launcher_c)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  ✗ Launcher compilation failed: {result.stderr}")
        sys.exit(1)

    launcher_c.unlink()
    launcher_bin.chmod(0o755)


_MACOS_CODENAMES = {
    "14": "sonoma",
    "15": "sequoia",
    "26": "tahoe",
}


def _write_build_info(omlx_pkg_dir: Path, macos_target: str | None = None):
    """Write _build_info.py with build number for runtime display.

    Format: YYMMDDHHmmSS-macosNN-codename
    Example: 260313093001-macos15-sequoia
    """
    ts = datetime.now().strftime("%y%m%d%H%M%S")
    if macos_target:
        major = macos_target.split(".")[0]
    else:
        major = platform.mac_ver()[0].split(".")[0]
    codename = _MACOS_CODENAMES.get(major, "")
    tag = f"macos{major}-{codename}" if codename else f"macos{major}"
    build_number = f"{ts}-{tag}"
    build_info_file = omlx_pkg_dir / "_build_info.py"
    build_info_file.write_text(f'build_number = "{build_number}"\n')
    print(f"  Generated _build_info.py: {build_number}")


def create_app_bundle():
    """Create macOS .app bundle."""
    print("\n[2/4] Creating app bundle...")

    app_dir = DIST_DIR / APP_BUNDLE
    contents_dir = app_dir / "Contents"
    macos_dir = contents_dir / "MacOS"
    resources_dir = contents_dir / "Resources"
    frameworks_dir = contents_dir / "Frameworks"

    # Clean and create directories
    if app_dir.exists():
        shutil.rmtree(app_dir)

    macos_dir.mkdir(parents=True)
    resources_dir.mkdir(parents=True)
    frameworks_dir.mkdir(parents=True)

    # Copy venvstacks environments to Frameworks
    print("  Copying Python environment...")
    for layer in ["cpython-3.11", "framework-mlx-framework", "app-omlx-app"]:
        src = EXPORT_DIR / layer
        if src.exists():
            dst = frameworks_dir / layer
            shutil.copytree(src, dst, symlinks=True)
            print(f"    Copied {layer}")

    # Copy venvstacks metadata
    venvstacks_meta = EXPORT_DIR / "__venvstacks__"
    if venvstacks_meta.exists():
        shutil.copytree(venvstacks_meta, frameworks_dir / "__venvstacks__", symlinks=True)

    # Copy omlx_app to Resources
    print("  Copying omlx_app...")
    omlx_app_src = SCRIPT_DIR / "omlx_app"
    omlx_app_dst = resources_dir / "omlx_app"
    shutil.copytree(omlx_app_src, omlx_app_dst, ignore=shutil.ignore_patterns(
        "__pycache__", "*.pyc"
    ))

    # Copy omlx package to Resources
    print("  Copying omlx package...")
    omlx_src = SCRIPT_DIR.parent / "omlx"
    omlx_dst = resources_dir / "omlx"
    if omlx_src.exists():
        shutil.copytree(omlx_src, omlx_dst, ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", ".git", "tests", "examples"
        ))

    # Generate _engine_commits.json for engine SHA display in admin dashboard
    _write_engine_commits(omlx_dst)

    # Copy SVG logo files to Resources for menubar icons
    print("  Copying logo SVGs...")
    admin_static = SCRIPT_DIR.parent / "omlx" / "admin" / "static"
    svg_files = [
        "navbar-logo-dark.svg",
        "navbar-logo-light.svg",
        "menubar-outline.svg",
        "menubar-filled.svg",
    ]
    for svg_name in svg_files:
        svg_src = admin_static / svg_name
        if svg_src.exists():
            shutil.copy2(svg_src, resources_dir / svg_name)
            print(f"    Copied {svg_name}")

    # Copy Python binary into MacOS/ so macOS recognizes it as a bundle executable
    print("  Copying Python runtime into MacOS/...")
    src_python = frameworks_dir / "cpython-3.11" / "bin" / "python3"
    dst_python = macos_dir / "python3"
    shutil.copy2(src_python, dst_python)
    dst_python.chmod(0o755)

    # Python binary references @executable_path/../lib/libpython3.11.dylib
    # Create Contents/lib/ with symlink to the actual dylib in Frameworks
    lib_dir = contents_dir / "lib"
    lib_dir.mkdir(exist_ok=True)
    (lib_dir / "libpython3.11.dylib").symlink_to(
        "../Frameworks/cpython-3.11/lib/libpython3.11.dylib"
    )

    # Create compiled C launcher binary
    print("  Creating launcher...")
    _create_c_launcher(macos_dir, APP_NAME)

    # Create CLI launcher script (for terminal use: oMLX.app/Contents/MacOS/omlx-cli)
    # Named "omlx-cli" to avoid case-insensitive collision with "oMLX" on APFS.
    print("  Creating CLI launcher script...")
    cli_launcher = macos_dir / "omlx-cli"
    cli_launcher.write_text(
        '#!/bin/bash\n'
        'DIR="$(cd "$(dirname "$0")" && pwd)"\n'
        'CONTENTS="$(dirname "$DIR")"\n'
        'LAYERS="$CONTENTS/Frameworks"\n'
        '[ ! -d "$LAYERS" ] && LAYERS="$CONTENTS/Python"\n'
        'export PYTHONHOME="$LAYERS/cpython-3.11"\n'
        'export PYTHONPATH="$CONTENTS/Resources:$LAYERS/app-omlx-app/lib/python3.11/site-packages:$LAYERS/framework-mlx-framework/lib/python3.11/site-packages"\n'
        'export PYTHONDONTWRITEBYTECODE=1\n'
        'exec "$DIR/python3" -m omlx.cli "$@"\n'
    )
    cli_launcher.chmod(0o755)

    # Create Info.plist
    print("  Creating Info.plist...")
    info_plist = {
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": "com.omlx.app",
        "CFBundleVersion": VERSION,
        "CFBundleShortVersionString": VERSION,
        "CFBundleExecutable": APP_NAME,
        "CFBundlePackageType": "APPL",
        "CFBundleSignature": "????",
        "CFBundleIconFile": "AppIcon",
        "LSMinimumSystemVersion": "15.0",
        "LSUIElement": True,
        "NSHighResolutionCapable": True,
        "LSArchitecturePriority": ["arm64"],
        "NSHumanReadableCopyright": f"Copyright 2024 oMLX contributors. Version {VERSION}",
    }

    with open(contents_dir / "Info.plist", "wb") as f:
        plistlib.dump(info_plist, f)

    # Create placeholder icon
    create_placeholder_icon(resources_dir)

    print(f"  ✓ Created {app_dir}")
    return app_dir


def _create_composite_svg(dark_svg: Path) -> str:
    """Create a composite SVG: white rounded-rect background + black logo."""
    svg_content = dark_svg.read_text()
    # Extract the <g> element (contains transform + path)
    g_match = re.search(r"<g[^>]*>.*?</g>", svg_content, re.DOTALL)
    g_element = g_match.group(0) if g_match else ""
    # Change fill from white to black for white background
    g_element = g_element.replace('fill="#ffffff"', 'fill="#000000"')

    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" viewBox="0 0 1024 1024">
  <rect x="96" y="96" width="832" height="832" rx="186" ry="186" fill="#ffffff"/>
  <svg x="180" y="180" width="664" height="664" viewBox="0 0 497.000000 497.000000">
    {g_element}
  </svg>
</svg>'''


def create_placeholder_icon(resources_dir: Path):
    """Create app icon from SVG logo (dark logo on white background).

    Rendering priority:
    1. Exported venvstacks Python + AppKit (native SVG rendering)
    2. cairosvg (if installed in build env)
    3. Pillow placeholder (last resort)
    """
    icon_path = resources_dir / "AppIcon.icns"
    dark_svg = SCRIPT_DIR.parent / "omlx" / "admin" / "static" / "navbar-logo-dark.svg"

    if not dark_svg.exists():
        print("    Warning: navbar-logo-dark.svg not found, skipping icon")
        return

    # Create composite SVG (white bg + black penguin)
    composite_svg = _create_composite_svg(dark_svg)
    tmp_svg = resources_dir / "_icon_tmp.svg"
    tmp_png = resources_dir / "_icon_tmp.png"
    tmp_svg.write_text(composite_svg)

    try:
        # Method 1: Use exported runtime Python with AppKit (native macOS SVG rendering)
        runtime_python = EXPORT_DIR / "cpython-3.11" / "bin" / "python3"
        if runtime_python.exists() and _render_svg_with_appkit(runtime_python, tmp_svg, tmp_png):
            _png_to_icns(str(tmp_png), icon_path, resources_dir)
            print("    Created app icon from SVG (AppKit)")
        # Method 2: cairosvg
        elif _render_svg_with_cairosvg(composite_svg, tmp_png):
            _png_to_icns(str(tmp_png), icon_path, resources_dir)
            print("    Created app icon from SVG (cairosvg)")
        else:
            print("    Warning: Could not render SVG, no icon created")
    finally:
        tmp_svg.unlink(missing_ok=True)
        tmp_png.unlink(missing_ok=True)


def _render_svg_with_appkit(python_exe: Path, svg_path: Path, png_path: Path) -> bool:
    """Render SVG to PNG using AppKit's native NSImage (via subprocess).

    Uses the venvstacks runtime Python with PYTHONHOME + layer site-packages
    so that PyObjC (AppKit/Foundation) is available.
    """
    script = f'''
import sys
from Foundation import NSData
from AppKit import NSImage, NSBitmapImageRep, NSPNGFileType, NSMakeRect, NSCompositingOperationSourceOver
from AppKit import NSGraphicsContext, NSImageInterpolationHigh

svg_data = NSData.dataWithContentsOfFile_("{svg_path}")
if svg_data is None:
    sys.exit(1)

image = NSImage.alloc().initWithData_(svg_data)
if image is None:
    sys.exit(1)

size = 1024
out_image = NSImage.alloc().initWithSize_((size, size))
out_image.lockFocus()
ctx = NSGraphicsContext.currentContext()
ctx.setImageInterpolation_(NSImageInterpolationHigh)
image.drawInRect_fromRect_operation_fraction_(
    NSMakeRect(0, 0, size, size),
    NSMakeRect(0, 0, image.size().width, image.size().height),
    NSCompositingOperationSourceOver,
    1.0,
)
out_image.unlockFocus()

rep = NSBitmapImageRep.alloc().initWithData_(out_image.TIFFRepresentation())
png_data = rep.representationUsingType_properties_(NSPNGFileType, {{}})
png_data.writeToFile_atomically_("{png_path}", True)
'''
    runtime_dir = python_exe.parent.parent
    app_sp = EXPORT_DIR / "app-omlx-app" / "lib" / "python3.11" / "site-packages"
    fw_sp = EXPORT_DIR / "framework-mlx-framework" / "lib" / "python3.11" / "site-packages"

    env = os.environ.copy()
    env["PYTHONHOME"] = str(runtime_dir)
    env["PYTHONPATH"] = f"{app_sp}:{fw_sp}"

    try:
        result = subprocess.run(
            [str(python_exe), "-c", script],
            capture_output=True, timeout=30, env=env,
        )
        if result.returncode != 0:
            print(f"    AppKit stderr: {result.stderr.decode()[:200]}")
        return result.returncode == 0 and png_path.exists()
    except Exception as e:
        print(f"    AppKit rendering failed: {e}")
        return False


def _render_svg_with_cairosvg(svg_content: str, png_path: Path) -> bool:
    """Render SVG to PNG using cairosvg."""
    try:
        import cairosvg
        cairosvg.svg2png(
            bytestring=svg_content.encode(),
            write_to=str(png_path),
            output_width=1024, output_height=1024,
        )
        return png_path.exists()
    except ImportError:
        return False
    except Exception as e:
        print(f"    cairosvg rendering failed: {e}")
        return False


def _png_to_icns(png_path: str, icon_path: Path, resources_dir: Path):
    """Convert a 1024x1024 PNG to .icns via iconset using sips (macOS built-in)."""
    iconset_dir = resources_dir / "AppIcon.iconset"
    iconset_dir.mkdir(exist_ok=True)

    sizes = [
        (16, "icon_16x16.png"),
        (32, "icon_16x16@2x.png"),
        (32, "icon_32x32.png"),
        (64, "icon_32x32@2x.png"),
        (128, "icon_128x128.png"),
        (256, "icon_128x128@2x.png"),
        (256, "icon_256x256.png"),
        (512, "icon_256x256@2x.png"),
        (512, "icon_512x512.png"),
        (1024, "icon_512x512@2x.png"),
    ]

    for s, name in sizes:
        out = iconset_dir / name
        shutil.copy2(png_path, str(out))
        subprocess.run(
            ["sips", "-z", str(s), str(s), str(out)],
            capture_output=True,
        )

    subprocess.run(
        ["iconutil", "-c", "icns", str(iconset_dir), "-o", str(icon_path)],
        capture_output=True,
    )

    shutil.rmtree(iconset_dir)


def sign_app(app_dir: Path):
    """Ad-hoc sign the app bundle for development.

    Uses --deep to recursively sign subcomponents. This may fail on
    venvstacks layers (e.g. cpython-3.11 in Frameworks/) because codesign
    treats dotted directory names as framework bundles.

    If signing fails, the broken _CodeSignature is removed so the app
    can still run unsigned on the developer's machine. Release builds
    use build_release.py which relocates Frameworks/ to Python/ first.
    """
    print("\n[3/4] Signing app bundle...")

    result = subprocess.run(
        ["codesign", "--force", "--deep", "--sign", "-", str(app_dir)],
        capture_output=True,
    )

    if result.returncode != 0:
        # --deep signing failed (likely due to dotted dir names in Frameworks/).
        # Remove the broken _CodeSignature so macOS doesn't show "damaged" error.
        codesig = app_dir / "Contents" / "_CodeSignature"
        if codesig.exists():
            shutil.rmtree(codesig)
        print("  ⚠ Deep signing failed (expected for dev builds), running unsigned")
    else:
        print(f"  ✓ Signed {app_dir}")


def create_dmg(app_dir: Path):
    """Create DMG installer with Applications symlink for drag-and-drop."""
    print("\n[4/4] Creating DMG...")

    dmg_path = DIST_DIR / f"{APP_NAME}-{VERSION}.dmg"
    dmg_staging = DIST_DIR / "_dmg_staging"

    # Remove existing
    if dmg_path.exists():
        dmg_path.unlink()
    if dmg_staging.exists():
        shutil.rmtree(dmg_staging)

    # Create staging directory
    dmg_staging.mkdir(parents=True)

    # Copy app bundle to staging
    shutil.copytree(app_dir, dmg_staging / APP_BUNDLE, symlinks=True)

    # Create Applications symlink
    applications_link = dmg_staging / "Applications"
    applications_link.symlink_to("/Applications")

    print("  Creating DMG with Applications shortcut...")
    run_cmd([
        "hdiutil", "create",
        "-volname", APP_NAME,
        "-srcfolder", str(dmg_staging),
        "-ov", "-format", "UDZO",
        str(dmg_path)
    ])

    # Cleanup staging
    shutil.rmtree(dmg_staging)

    print(f"  ✓ Created {dmg_path}")
    return dmg_path


def main():
    parser = argparse.ArgumentParser(description="Build oMLX macOS app")
    parser.add_argument("--skip-venv", action="store_true",
                        help="Skip venvstacks build")
    parser.add_argument("--dmg-only", action="store_true",
                        help="Only create DMG from existing build")
    parser.add_argument("--macos-target",
                        help="Target macOS version for mlx/mlx-metal wheels "
                        "(e.g. 26.0). Downloads platform-specific wheels "
                        "with M5 Neural Accelerator support.")
    args = parser.parse_args()

    print(f"Building {APP_NAME} v{VERSION}")
    print("=" * 50)

    # Clean build artifacts before starting (unless dmg-only)
    if not args.dmg_only:
        clean_all(preserve_venv=args.skip_venv)

    DIST_DIR.mkdir(parents=True, exist_ok=True)

    if args.dmg_only:
        app_dir = DIST_DIR / APP_BUNDLE
        if not app_dir.exists():
            print(f"Error: {app_dir} not found. Run full build first.")
            sys.exit(1)
        create_dmg(app_dir)
    else:
        if not args.skip_venv:
            build_venvstacks()
        elif not EXPORT_DIR.exists():
            print("Warning: No existing envs found, building venvstacks...")
            build_venvstacks()

        # Swap mlx/mlx-metal wheels for target macOS version
        if args.macos_target:
            swap_platform_wheels(EXPORT_DIR, args.macos_target)

        app_dir = create_app_bundle()
        omlx_pkg_dir = app_dir / "Contents" / "Resources" / "omlx"
        _write_build_info(omlx_pkg_dir, args.macos_target)
        sign_app(app_dir)
        create_dmg(app_dir)

    print("\n" + "=" * 50)
    print("Build complete!")
    print(f"  App: {DIST_DIR / APP_BUNDLE}")
    print(f"  DMG: {DIST_DIR / f'{APP_NAME}-{VERSION}.dmg'}")


if __name__ == "__main__":
    main()
