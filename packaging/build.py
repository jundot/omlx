#!/usr/bin/env python3
"""
Build the venvstacks Python layers embedded inside the Swift macOS bundle.

The PyObjC menubar `.app` and DMG pipeline this script used to drive
retired with the Swift rewrite; the Swift bundle is now produced by
`apps/omlx-mac/Scripts/build.sh`, which invokes this script with
`--venvstacks-only` (or `--print-fingerprint`) to refresh the export.

Usage:
    python build.py --venvstacks-only
    python build.py --print-fingerprint
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
BUILD_DIR = SCRIPT_DIR / "_build"
EXPORT_DIR = SCRIPT_DIR / "_export"
DIST_DIR = SCRIPT_DIR / "dist"
WHEELS_DIR = SCRIPT_DIR / "_wheels"
APP_NAME = "oMLX"


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
        / "framework-mlx-base"
        / "requirements-framework-mlx-base-macosx_arm64.txt"
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
        / "framework-mlx-base"
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



def _find_target_python() -> str:
    """Find a Python interpreter matching the venvstacks target version.

    Sdist-only packages may compile C extensions, so the wheel must be built
    with the same Python version that venvstacks targets (e.g. 3.11).
    Falls back to sys.executable if no matching version is found.
    """
    toml_path = SCRIPT_DIR / "venvstacks.toml"
    content = toml_path.read_text()
    match = re.search(r'python_implementation\s*=\s*"cpython@(\d+\.\d+)', content)
    if not match:
        return sys.executable

    target_minor = match.group(1)  # e.g. "3.11"
    candidates = [
        shutil.which(f"python{target_minor}"),
        str(BUILD_DIR / f"cpython-{target_minor}" / "bin" / f"python{target_minor}"),
    ]
    for path in candidates:
        if not path or not Path(path).exists():
            continue
        # Skip interpreters without pip (e.g. venvstacks runtimes strip it)
        check = subprocess.run(
            [path, "-m", "pip", "--version"],
            capture_output=True,
        )
        if check.returncode == 0:
            return path

    print(f"  Warning: python{target_minor} not found, using {sys.executable}")
    return sys.executable


def _build_sdist_wheel(pkg_name: str) -> bool:
    """Build a wheel for a sdist-only package into _wheels/.

    Uses the target Python version so C extensions get the correct ABI tag.
    Returns True if the wheel was built successfully.
    """
    target_python = _find_target_python()
    print(f"  Building wheel for {pkg_name} (sdist-only, using {target_python})...")
    result = subprocess.run(
        [target_python, "-m", "pip", "wheel", pkg_name, "--no-deps",
         "-w", str(WHEELS_DIR)],
        capture_output=False,
    )
    return result.returncode == 0


def build_local_wheels(source_toml: Path | None = None):
    """Pre-build wheels for git-pinned packages.

    venvstacks/uv disables source builds (--only-binary :all:), so git-pinned
    packages must be pre-built as wheels. This function:
    1. Parses git URLs from `source_toml` (defaults to the resolved
       venvstacks.toml emitted by `_generate_venvstacks_toml`)
    2. Builds wheels via pip
    3. Returns a mapping of package_name -> version for toml rewriting

    Sdist-only dependencies (packages with no pre-built wheel on PyPI) are
    handled separately by _lock_with_sdist_retry() during the lock step.
    """
    print("\n[0/4] Building local wheels...")

    if source_toml is None:
        source_toml = SCRIPT_DIR / "_venvstacks_resolved.toml"
        if not source_toml.exists():
            source_toml = SCRIPT_DIR / "venvstacks.toml"
    git_reqs = _parse_git_requirements(source_toml)

    # Clean and recreate wheels dir for fresh builds
    if WHEELS_DIR.exists():
        shutil.rmtree(WHEELS_DIR)
    WHEELS_DIR.mkdir(parents=True)

    # Build wheels from git-pinned packages
    for full_req, git_url in git_reqs:
        pkg_name = full_req.split("@")[0].strip()
        print(f"  Building wheel for {pkg_name} ...")
        run_cmd([
            sys.executable, "-m", "pip", "wheel",
            git_url,
            "--no-deps",
            "-w", str(WHEELS_DIR),
        ])

    # Build version mapping from git-pinned wheels only
    # (used for rewriting venvstacks.toml git URLs to local file:// paths)
    git_pkg_names = {
        full_req.split("@")[0].strip().lower().replace("-", "_")
        for full_req, _ in git_reqs
    }
    version_map = {}
    for whl in WHEELS_DIR.glob("*.whl"):
        name = _wheel_pkg_name(whl)
        version = _wheel_version(whl)
        if name.replace("-", "_") in git_pkg_names:
            version_map[name] = version
        print(f"    {name} == {version}")

    total = len(list(WHEELS_DIR.glob("*.whl")))
    print(f"  ✓ {total} wheel(s) built in {WHEELS_DIR}")
    return version_map


def _lock_with_sdist_retry(lock_cmd: list, max_retries: int = 10):
    """Run venvstacks lock, auto-building wheels for sdist-only packages.

    When uv fails with "has no usable wheels", extract the package name,
    build a wheel locally into _wheels/, and retry. Repeats up to
    max_retries times to handle transitive sdist-only dependencies.
    """
    built = set()
    for attempt in range(max_retries):
        result = subprocess.run(
            lock_cmd, capture_output=True, text=True,
        )
        if result.returncode == 0:
            if result.stdout:
                print(result.stdout, end="")
            return

        stderr = result.stderr or ""
        stdout = result.stdout or ""
        combined = stderr + stdout

        # Pattern: "Because <pkg>==<ver> has no usable wheels"
        match = re.search(
            r"Because\s+(\S+)==\S+\s+has no usable wheels", combined
        )
        if not match:
            # Not a sdist-only failure — print output and abort
            if stdout:
                print(stdout, end="")
            if stderr:
                print(stderr, end="", file=sys.stderr)
            print(f"  ✗ Command failed with code {result.returncode}")
            sys.exit(1)

        pkg = match.group(1)
        if pkg in built:
            # Already tried this package, something else is wrong
            if stdout:
                print(stdout, end="")
            if stderr:
                print(stderr, end="", file=sys.stderr)
            print(f"  ✗ Already built {pkg} but lock still fails")
            sys.exit(1)

        print(f"  sdist-only dependency detected: {pkg}, building wheel...")
        if not _build_sdist_wheel(pkg):
            print(f"  ✗ Failed to build wheel for {pkg}")
            sys.exit(1)
        built.add(pkg)
        print(f"  Retrying lock (attempt {attempt + 2})...")

    print(f"  ✗ Lock still failing after {max_retries} sdist wheel builds")
    sys.exit(1)


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

    # Git pins now live in pyproject.toml (single source of truth); the
    # venvstacks.toml template no longer carries version-bearing entries.
    pyproject_path = SCRIPT_DIR.parent / "pyproject.toml"
    git_reqs = _parse_git_requirements(pyproject_path)

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


# Maps each venvstacks layer name to the pyproject.toml sections whose
# requirements feed into it. Entries:
#   "project" → [project] dependencies (PEP 621)
#   "<extra>" → [project.optional-dependencies].<extra>
# Layers not listed here are left empty (e.g. cpython-3.11 has no deps).
# When Jun's release path reintroduces a Python menubar application
# layer, add an entry like {"omlx-app": ["menubar"]} alongside.
# Later sources override earlier ones on a name collision (PEP 503
# normalized). [bundle] last means a bundle-specific [audio]-extra entry
# wins over [project]'s plain entry for the same package.
_LAYER_REQUIREMENTS_SOURCES = {
    "mlx-base": ["project", "bundle"],
}


def _read_pyproject_requirements() -> dict[str, list[str]]:
    """Read pyproject.toml and return {section: [req_string, ...]}.

    section is "project" for the main dependencies or the name of an
    entry under [project.optional-dependencies].
    """
    import tomllib

    pyproject = SCRIPT_DIR.parent / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())
    out: dict[str, list[str]] = {
        "project": list(data["project"]["dependencies"]),
    }
    for name, reqs in data["project"].get("optional-dependencies", {}).items():
        out[name] = list(reqs)
    return out


def _generate_venvstacks_toml() -> Path:
    """Render the venvstacks layer template + pyproject.toml deps into a
    resolved venvstacks.toml the rest of the build pipeline can consume.

    The committed packaging/venvstacks.toml carries only layer structure
    + dynlib_exclude + [tool.uv] settings; its `requirements = []` arrays
    are placeholders. This function fills them in from pyproject.toml so
    Python dep versions live in exactly one place.

    Stdlib-only implementation: tomllib to parse the template for layer
    discovery, then plain text substitution to inject populated arrays.
    Avoids a tomlkit dependency on the host Python that runs build.py.
    """
    import tomllib

    template_path = SCRIPT_DIR / "venvstacks.toml"
    out_path = SCRIPT_DIR / "_venvstacks_resolved.toml"
    pp_reqs = _read_pyproject_requirements()

    template_text = template_path.read_text()
    parsed = tomllib.loads(template_text)

    # Split the template on `[[...]]` table-array headers so we can find
    # and modify the right section without disturbing siblings.
    sections = re.split(r"(?m)(?=^\[\[)", template_text)

    populated = []
    for layer in parsed.get("frameworks", []) + parsed.get("applications", []):
        layer_name = str(layer["name"])
        sources = _LAYER_REQUIREMENTS_SOURCES.get(layer_name)
        if not sources:
            continue

        # Merge requirements from each pyproject section in order, deduping
        # by PEP 503 normalized name with extras stripped. Later sources
        # win — so [bundle]'s extras-bearing entries override [project]'s
        # plain entries for the same package (e.g. mistral-common[audio]).
        merged_map: dict[str, str] = {}
        for src in sources:
            if src not in pp_reqs:
                print(
                    f"  ✗ Layer {layer_name!r} references pyproject section "
                    f"{src!r} which is not declared. "
                    "Add it to pyproject.toml [project.optional-dependencies].",
                    file=sys.stderr,
                )
                sys.exit(1)
            for req in pp_reqs[src]:
                key = req.split("@", 1)[0].split(";", 1)[0]
                key = re.split(r"[<>=!~]", key, maxsplit=1)[0].strip()
                key = key.split("[", 1)[0].lower()  # strip PEP 508 extras
                merged_map[key] = req
        merged = list(merged_map.values())

        formatted_array = (
            "requirements = [\n"
            + "".join(f'    "{r}",\n' for r in merged)
            + "]"
        )

        # Find the section starting with [[frameworks]] or [[applications]]
        # whose `name = "X"` matches this layer, and replace its
        # `requirements = []` placeholder.
        layer_re = re.compile(
            rf'^name\s*=\s*"{re.escape(layer_name)}"\s*$',
            re.MULTILINE,
        )
        placeholder_re = re.compile(
            r"^requirements\s*=\s*\[\s*\]\s*$",
            re.MULTILINE,
        )
        section_re = re.compile(r"^\[\[(?:frameworks|applications)\]\]")

        injected = False
        for i, section in enumerate(sections):
            if not section_re.match(section):
                continue
            if not layer_re.search(section):
                continue
            new_section, n = placeholder_re.subn(
                formatted_array, section, count=1
            )
            if n == 0:
                continue
            sections[i] = new_section
            injected = True
            break

        if not injected:
            print(
                f"  ✗ Could not find `requirements = []` placeholder in layer "
                f"{layer_name!r}. The template at {template_path} must contain "
                "a stand-alone `requirements = []` line within the layer block.",
                file=sys.stderr,
            )
            sys.exit(1)

        populated.append(f"{layer_name} ({len(merged)} reqs)")

    out_path.write_text("".join(sections))
    print(f"  Resolved layer requirements from pyproject.toml: {', '.join(populated)}")
    return out_path


def _create_resolved_toml(version_map: dict[str, str], base_toml: Path) -> Path:
    """Rewrite git URLs in base_toml to local file:// wheel paths.

    Git-built wheels have different hashes than PyPI releases of the
    same version, so we must point directly to the local wheel files to
    avoid hash mismatches. The input toml comes from
    _generate_venvstacks_toml() and the output overwrites it in place.
    """
    content = base_toml.read_text()

    for full_req, git_url in _parse_git_requirements(base_toml):
        pkg_name = full_req.split("@")[0].strip()
        whl = _find_wheel_for_package(pkg_name)
        if whl:
            whl_uri = whl.resolve().as_uri()
            old_line = f'"{full_req}"'
            new_line = f'"{pkg_name} @ {whl_uri}"'
            content = content.replace(old_line, new_line)
            print(f"    {pkg_name} @ git+... → {whl.name}")

    base_toml.write_text(content)
    return base_toml


def _venvstacks_driver() -> list[str]:
    """Pick an available venvstacks driver as a command prefix.

    Resolution order (first that works wins):
      1. `<sys.executable> -m venvstacks` when venvstacks is importable
         from the Python running build.py.
      2. `uvx venvstacks` — uv's pipx-equivalent.
      3. `pipx run venvstacks` — historical default.
      4. `venvstacks` on PATH as a last resort.

    Why -m first: the PATH-installed `venvstacks` script's shebang may
    point at a dotted interpreter like `.../python3.11`. venvstacks 0.7
    computes `Path(sys.executable).suffix` to derive the runtime binary
    extension; on `python3.11` that returns ".11", producing a bogus
    `bin/python.11` path and breaking `pip --python`. Running via the
    current interpreter (typically `.../python3`, suffix "") sidesteps
    the bug. See https://github.com/lmstudio-ai/venvstacks/issues/ for
    the upstream report.
    """
    try:
        subprocess.run(
            [sys.executable, "-c", "import venvstacks"],
            check=True, capture_output=True,
        )
        return [sys.executable, "-m", "venvstacks"]
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    if shutil.which("uvx") is not None:
        return ["uvx", "venvstacks"]
    if shutil.which("pipx") is not None:
        return ["pipx", "run", "venvstacks"]
    if shutil.which("venvstacks") is not None:
        return ["venvstacks"]
    print(
        "  ✗ No venvstacks driver found. Install with one of:\n"
        "      pip install -e \".[dev]\"     (pip-managed venv)\n"
        "      uv sync --dev                (uv-managed venv)\n"
        "      pipx install venvstacks       (host-global tool)",
        file=sys.stderr,
    )
    sys.exit(1)


def build_venvstacks():
    """Build venvstacks layers."""
    print("\n[1/4] Building venvstacks layers...")

    venvstacks_cmd = _venvstacks_driver()
    print(f"  Using venvstacks driver: {' '.join(venvstacks_cmd)}")

    # Step 0: Render the resolved venvstacks.toml from pyproject.toml deps
    print("\n  Resolving layer requirements from pyproject.toml...")
    resolved_toml = _generate_venvstacks_toml()

    venvstacks_cmd = _venvstacks_driver()
    print(f"  Using venvstacks driver: {' '.join(venvstacks_cmd)}")

    # Step 1: Build wheels from git-pinned packages
    version_map = build_local_wheels(resolved_toml)

    # Step 2: Swap git URLs in the resolved toml for local wheel paths
    if version_map:
        print("\n  Resolving git requirements to local wheel paths...")
        _create_resolved_toml(version_map, resolved_toml)

    # Local wheels args
    local_wheels_args = []
    if WHEELS_DIR.exists() and any(WHEELS_DIR.glob("*.whl")):
        local_wheels_args = ["--local-wheels", str(WHEELS_DIR)]

    # Step 3: Lock environments (always re-lock to match current wheels)
    # If lock fails due to sdist-only packages (no pre-built wheel on PyPI),
    # _lock_with_sdist_retry() builds them locally and retries automatically.
    print("\n  Locking environments...")
    lock_cmd = venvstacks_cmd + [
        "lock",
        str(resolved_toml),
    ] + local_wheels_args
    if version_map:
        # Force re-lock when git packages changed (hashes will differ)
        lock_cmd += ["--reset-lock", "*"]
    else:
        lock_cmd += ["--if-needed"]
    _lock_with_sdist_retry(lock_cmd)

    # Step 4: Build environments
    print("\n  Building environments (this may take a while)...")
    run_cmd(venvstacks_cmd + [
        "build",
        str(resolved_toml),
        "--no-lock",
    ] + local_wheels_args)

    # Step 5: Export to local directory for app bundle
    print("\n  Exporting environments...")
    if EXPORT_DIR.exists():
        shutil.rmtree(EXPORT_DIR)

    run_cmd(venvstacks_cmd + [
        "local-export",
        str(resolved_toml),
        "--output-dir", str(EXPORT_DIR),
    ])

    # Cleanup the generated resolved toml — always temporary now that it's
    # produced fresh each build from pyproject.toml + venvstacks.toml.
    if resolved_toml.exists():
        resolved_toml.unlink()

    # Install mlx-audio separately: build wheel from git, install --no-deps.
    # mlx-audio pins mlx-lm==0.31.1 which conflicts with our git-pinned mlx-lm,
    # so it can't go through venvstacks' uv resolver.
    _install_mlx_audio(EXPORT_DIR)

    # Install paroquant --no-deps. The official [mlx] extra requires
    # torchvision which the mlx load path doesn't actually use; verified
    # end-to-end on 0.1.14. All real deps (mlx, mlx-lm, mlx-vlm, numpy,
    # huggingface_hub) are already in the framework layer.
    _install_paroquant(EXPORT_DIR)

    # Install xgrammar + apache-tvm-ffi --no-deps; oMLX uses the non-torch
    # paths only, and omlx/_torch_stub.py satisfies xgrammar's import-time
    # references to torch so the runtime torch dep is unnecessary.
    _install_xgrammar(EXPORT_DIR)

    # Bundle spacy language model for Kokoro TTS.
    # misaki's en.G2P tries spacy.cli.download() at runtime, which fails in
    # the code-signed app bundle (read-only site-packages).
    _install_spacy_model(EXPORT_DIR)

    # Strip large packages that are only needed for model conversion / data
    # loading, not inference. Saves ~780 MB in the app bundle.
    _strip_unused_packages(EXPORT_DIR)

    return EXPORT_DIR


# mlx-audio git commit — aligned with pyproject.toml [audio] extra
_MLX_AUDIO_GIT = "git+https://github.com/Blaizzy/mlx-audio@51753266e0a4f766fd5e6fbc46652224efc23981"


def _install_mlx_audio(export_dir: Path):
    """Build mlx-audio wheel from git and install into exported framework."""
    print("\n  Building mlx-audio from git...")
    audio_wheels = SCRIPT_DIR / "_audio_wheels"
    if audio_wheels.exists():
        shutil.rmtree(audio_wheels)
    audio_wheels.mkdir()

    # Build wheel
    run_cmd([
        sys.executable, "-m", "pip", "wheel",
        "--no-deps", "--wheel-dir", str(audio_wheels),
        _MLX_AUDIO_GIT,
    ])

    # Install into framework site-packages
    fw_site = (
        export_dir
        / "framework-mlx-base"
        / "lib"
        / "python3.11"
        / "site-packages"
    )
    if not fw_site.exists():
        print(f"  ✗ site-packages not found: {fw_site}")
        return

    import zipfile
    for whl in audio_wheels.glob("*.whl"):
        print(f"    Installing {whl.name} (--no-deps)")
        with zipfile.ZipFile(whl) as zf:
            zf.extractall(fw_site)

    shutil.rmtree(audio_wheels)
    print("  ✓ mlx-audio installed")


# paroquant version — keep in sync with pyproject.toml [paroquant] extra
_PAROQUANT_VERSION = "0.1.14"


def _install_paroquant(export_dir: Path):
    """Build paroquant wheel from PyPI and install --no-deps into framework."""
    print("\n  Building paroquant wheel...")
    paro_wheels = SCRIPT_DIR / "_paroquant_wheels"
    if paro_wheels.exists():
        shutil.rmtree(paro_wheels)
    paro_wheels.mkdir()

    run_cmd([
        sys.executable, "-m", "pip", "wheel",
        "--no-deps", "--wheel-dir", str(paro_wheels),
        f"paroquant=={_PAROQUANT_VERSION}",
    ])

    fw_site = (
        export_dir
        / "framework-mlx-base"
        / "lib"
        / "python3.11"
        / "site-packages"
    )
    if not fw_site.exists():
        print(f"  ✗ site-packages not found: {fw_site}")
        return

    import zipfile
    for whl in paro_wheels.glob("*.whl"):
        print(f"    Installing {whl.name} (--no-deps)")
        with zipfile.ZipFile(whl) as zf:
            zf.extractall(fw_site)

    shutil.rmtree(paro_wheels)
    print("  ✓ paroquant installed")


# xgrammar / tvm-ffi versions — single source of truth lives in
# omlx/_torch_stub.py (the stub MUST track the actually-installed versions
# or imports fail). Importing keeps the two files from drifting apart.
try:
    from omlx._torch_stub import (
        _TARGET_TVM_FFI_VERSIONS,
        _TARGET_XGRAMMAR_VERSIONS,
    )

    _XGRAMMAR_VERSION = _TARGET_XGRAMMAR_VERSIONS[0]
    _TVM_FFI_VERSION = _TARGET_TVM_FFI_VERSIONS[0]
except Exception:  # pragma: no cover — build runs may not have omlx on path yet
    _XGRAMMAR_VERSION = "0.2.0"
    _TVM_FFI_VERSION = "0.1.11"


def _install_xgrammar(export_dir: Path):
    """Install xgrammar + apache-tvm-ffi --no-deps into framework site-packages.

    xgrammar declares torch>=1.10.0 as a runtime dep, but oMLX only exercises
    its non-torch paths (numpy bitmasks + MLX kernel). Shipping torch would
    add ~500 MB to the bundle. omlx/_torch_stub.py satisfies xgrammar's
    import-time torch references so the package loads without real torch.

    Idempotent: a sentinel file is written after the last extract; if it's
    present we skip. Trusting both ``xgrammar/`` and ``tvm_ffi/`` to exist
    isn't enough — an interruption between the two extracts would otherwise
    leave a half-installed state the next run accepts.
    """
    fw_site = (
        export_dir
        / "framework-mlx-base"
        / "lib"
        / "python3.11"
        / "site-packages"
    )
    sentinel = fw_site / (
        f"_omlx_xgrammar_{_XGRAMMAR_VERSION}_tvmffi_{_TVM_FFI_VERSION}.installed"
    )
    if sentinel.exists():
        print("  ✓ xgrammar + apache-tvm-ffi already installed, skipping")
        return

    print("\n  Downloading xgrammar wheels...")
    xgr_wheels = SCRIPT_DIR / "_xgrammar_wheels"
    if xgr_wheels.exists():
        shutil.rmtree(xgr_wheels)
    xgr_wheels.mkdir()

    # Explicit platform tags so the build host's Python version doesn't matter.
    run_cmd([
        sys.executable, "-m", "pip", "download",
        "--no-deps", "--dest", str(xgr_wheels),
        "--python-version", "3.11",
        "--platform", "macosx_11_0_arm64",
        "--only-binary=:all:",
        f"xgrammar=={_XGRAMMAR_VERSION}",
        f"apache-tvm-ffi=={_TVM_FFI_VERSION}",
    ])

    if not fw_site.exists():
        print(f"  ✗ site-packages not found: {fw_site}")
        return

    import zipfile
    for whl in xgr_wheels.glob("*.whl"):
        print(f"    Installing {whl.name} (--no-deps)")
        with zipfile.ZipFile(whl) as zf:
            zf.extractall(fw_site)

    shutil.rmtree(xgr_wheels)

    # Integrity check before sentinel-write: zipfile.extractall is not
    # atomic per file, so a build-host interrupt (SIGKILL, ENOSPC,
    # inode exhaustion) mid-extract can leave truncated __init__.py
    # files on disk. ``sentinel.exists()`` would still accept the next
    # run, masking the partial install. Verify the package roots
    # exist with non-empty __init__.py before writing the sentinel.
    integrity_checks = (
        ("xgrammar", fw_site / "xgrammar" / "__init__.py"),
        ("tvm_ffi", fw_site / "tvm_ffi" / "__init__.py"),
    )
    for pkg_name, init_path in integrity_checks:
        if not init_path.exists() or init_path.stat().st_size == 0:
            print(
                f"  ✗ {pkg_name} install incomplete ({init_path}); refusing "
                "to write sentinel — next run will retry"
            )
            return

    # Atomic sentinel write: write to a tmp file in the same directory
    # then ``os.replace`` (POSIX-atomic on the same filesystem). A bare
    # ``Path.write_text`` is open+write+close and can itself be interrupted
    # mid-write, leaving a zero-length sentinel that ``sentinel.exists()``
    # would still accept — exactly the failure mode this sentinel is
    # supposed to guard against.
    sentinel_tmp = sentinel.with_suffix(sentinel.suffix + ".tmp")
    sentinel_tmp.write_text(
        f"xgrammar=={_XGRAMMAR_VERSION}\napache-tvm-ffi=={_TVM_FFI_VERSION}\n"
    )
    os.replace(sentinel_tmp, sentinel)
    print("  ✓ xgrammar + apache-tvm-ffi installed")


# spacy language model — required by misaki (Kokoro TTS G2P)
# Update version when spacy is bumped in venvstacks.toml
_SPACY_MODEL = "en_core_web_sm"
_SPACY_MODEL_VERSION = "3.8.0"
_SPACY_MODEL_URL = (
    "https://github.com/explosion/spacy-models/releases/download/"
    f"{_SPACY_MODEL}-{_SPACY_MODEL_VERSION}/"
    f"{_SPACY_MODEL}-{_SPACY_MODEL_VERSION}-py3-none-any.whl"
)


def _install_spacy_model(export_dir: Path):
    """Download and install spacy en_core_web_sm into exported framework."""
    import urllib.request
    import zipfile

    fw_site = (
        export_dir
        / "framework-mlx-base"
        / "lib"
        / "python3.11"
        / "site-packages"
    )
    if not fw_site.exists():
        print(f"  ✗ site-packages not found: {fw_site}")
        return

    # Skip if already installed
    if (fw_site / _SPACY_MODEL).exists():
        print(f"  ✓ {_SPACY_MODEL} already installed, skipping")
        return

    print(f"\n  Installing {_SPACY_MODEL}-{_SPACY_MODEL_VERSION}...")
    whl_path = SCRIPT_DIR / f"{_SPACY_MODEL}-{_SPACY_MODEL_VERSION}.whl"

    try:
        urllib.request.urlretrieve(_SPACY_MODEL_URL, whl_path)
        with zipfile.ZipFile(whl_path) as zf:
            zf.extractall(fw_site)
        print(f"  ✓ {_SPACY_MODEL} installed")
    finally:
        whl_path.unlink(missing_ok=True)


# Packages to strip from the app bundle. These are transitive dependencies
# pulled in by modelscope (datasets→pyarrow/pandas) and mlx-vlm (opencv)
# but are NOT needed for inference at runtime. torch/sympy kept as safety
# net in case any future dependency pulls them in transitively.
_STRIP_PACKAGES = [
    "torch",
    "sympy",           # torch dep (safety net)
    "cv2",             # opencv-python, mlx-vlm only uses it for image loading (Pillow suffices)
    "pyarrow",         # datasets dep
    "pandas",          # datasets dep
    "datasets",        # modelscope dep, not used at inference
    # dist-info dirs (matched by prefix)
]

# Prefixes for dist-info directories to remove alongside the packages above.
_STRIP_DIST_PREFIXES = [
    "torch-", "sympy-", "opencv_python-", "pyarrow-", "pandas-", "datasets-",
]


def _strip_unused_packages(export_dir: Path):
    """Remove large packages not needed for inference from exported framework."""
    fw_site = (
        export_dir
        / "framework-mlx-base"
        / "lib"
        / "python3.11"
        / "site-packages"
    )
    if not fw_site.exists():
        return

    print("\n  Stripping unused packages from app bundle...")
    saved = 0

    for item in sorted(fw_site.iterdir()):
        name = item.name
        should_strip = (
            name in _STRIP_PACKAGES
            or any(name.startswith(p) for p in _STRIP_DIST_PREFIXES)
        )
        if should_strip and item.exists():
            size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
            saved += size
            print(f"    Removed {name} ({size / 1024 / 1024:.0f} MB)")

    print(f"  ✓ Stripped {saved / 1024 / 1024:.0f} MB total")

    # Post-strip invariant: no torch artifact must survive. A partial torch
    # (some files but not enough for xgrammar) would be the worst possible
    # outcome — _torch_stub.find_spec("torch") would return a real spec,
    # install() would short-circuit, and xgrammar would import the broken
    # half-torch and fail at runtime with confusing errors. Fail fast.
    surviving_torch = [
        p.name for p in fw_site.iterdir()
        if p.name == "torch"
        or p.name.startswith("torch-")
        or p.name.startswith("torch_")
    ]
    if surviving_torch:
        raise RuntimeError(
            "Post-strip integrity check failed: torch artifacts survived "
            f"the strip step: {surviving_torch}. _torch_stub.find_spec would "
            "see a real torch and short-circuit install(), leaving xgrammar "
            "to fail at runtime against a half-installed torch. Add the "
            "leftover names to _STRIP_PACKAGES / _STRIP_DIST_PREFIXES."
        )




def _compute_donor_fingerprint() -> str:
    """Hash of the inputs that determine the venvstacks export shape.

    Used by `apps/omlx-mac/Scripts/build.sh` to decide whether the cached
    `_export/` is still in sync with the current sources, or needs a
    rebuild.
    """
    import hashlib

    h = hashlib.sha256()
    inputs = [
        SCRIPT_DIR.parent / "pyproject.toml",
        SCRIPT_DIR / "venvstacks.toml",
        SCRIPT_DIR.parent / "uv.lock",
    ]
    for path in inputs:
        if path.exists():
            h.update(path.read_bytes())
    return h.hexdigest()


def _write_export_fingerprint():
    """Write the current fingerprint into _export/ so callers can detect drift."""
    if not EXPORT_DIR.exists():
        return
    fingerprint = _compute_donor_fingerprint()
    (EXPORT_DIR / ".fingerprint").write_text(fingerprint + "\n")
    print(f"  Wrote _export/.fingerprint ({fingerprint[:12]}…)")


def main():
    parser = argparse.ArgumentParser(
        description="Build venvstacks Python layers consumed by the Swift "
                    "macOS bundle. The legacy PyObjC menubar `.app` and DMG "
                    "pipeline lived here too; both retired with the Swift "
                    "rewrite. Today this script is invoked by "
                    "`apps/omlx-mac/Scripts/build.sh` to produce / fingerprint "
                    "the export layers — there is no full-app path anymore."
    )
    parser.add_argument("--venvstacks-only", action="store_true",
                        help="Run venvstacks lock+build+export and stop. "
                             "Currently the only supported run mode.")
    parser.add_argument("--print-fingerprint", action="store_true",
                        help="Print the donor fingerprint and exit. "
                             "`build.sh` uses this to detect drift between "
                             "sources and the cached `_export/`.")
    parser.add_argument("--macos-target",
                        help="Target macOS version for mlx/mlx-metal wheels "
                             "(e.g. 26.0). Downloads platform-specific wheels "
                             "with M5 Neural Accelerator support.")
    args = parser.parse_args()

    if args.print_fingerprint:
        print(_compute_donor_fingerprint())
        return

    print(f"Building {APP_NAME} v{VERSION}")
    print("=" * 50)

    if not args.venvstacks_only:
        parser.error(
            "Pass --venvstacks-only. The Swift bundle is built by "
            "apps/omlx-mac/Scripts/build.sh; this script only produces "
            "the Python layers it embeds."
        )

    build_venvstacks()
    if args.macos_target:
        swap_platform_wheels(EXPORT_DIR, args.macos_target)
    _write_export_fingerprint()
    print("\n" + "=" * 50)
    print("venvstacks export ready at:")
    print(f"  {EXPORT_DIR}")


if __name__ == "__main__":
    main()
