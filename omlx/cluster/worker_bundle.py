# SPDX-License-Identifier: Apache-2.0
"""Pinned bootstrap program and source bundle for CUDA worker enrollment."""

from __future__ import annotations

import gzip
import hashlib
import io
import shlex
import tarfile
from functools import lru_cache
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_BOOTSTRAP_PATH = Path(__file__).with_name("cuda_worker_bootstrap.py")
_SOURCE_SUFFIXES = {".py", ".json", ".txt", ".metal", ".cpp", ".h"}
_EXCLUDED_PARTS = {"__pycache__"}


@lru_cache(maxsize=1)
def cuda_bootstrap_program() -> bytes:
    return _BOOTSTRAP_PATH.read_bytes()


def cuda_bootstrap_digest() -> str:
    return hashlib.sha256(cuda_bootstrap_program()).hexdigest()


def _worker_source_files(package_root: Path) -> tuple[Path, ...]:
    return tuple(
        path
        for path in sorted(package_root.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in _SOURCE_SUFFIXES
        and not (_EXCLUDED_PARTS & set(path.parts))
    )


@lru_cache(maxsize=1)
def worker_source_bundle() -> bytes:
    """Return a deterministic, native-binary-free snapshot of this oMLX build."""

    output = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for path in _worker_source_files(_PACKAGE_ROOT):
            relative = path.relative_to(_PACKAGE_ROOT.parent)
            data = path.read_bytes()
            info = tarfile.TarInfo(relative.as_posix())
            info.size = len(data)
            info.mode = 0o755 if path == _BOOTSTRAP_PATH else 0o644
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = "root"
            info.gname = "root"
            archive.addfile(info, io.BytesIO(data))
    return output.getvalue()


def worker_source_digest() -> str:
    return hashlib.sha256(worker_source_bundle()).hexdigest()


def build_cuda_join_command(
    *,
    controller_url: str,
    join_key: str,
    controller_key_fingerprint: str,
    source_digest: str,
) -> str:
    """Build one shell command with a pinned bootstrap and one-time secret."""

    bootstrap_url = controller_url.rstrip("/") + "/cluster/join/bootstrap.py"
    digest = cuda_bootstrap_digest()
    if len(source_digest) != 64 or any(
        character not in "0123456789abcdef" for character in source_digest
    ):
        raise ValueError("worker source digest must be lowercase SHA-256")
    return (
        '(omlx_join_tmp="$(mktemp)" && '
        'trap \'rm -f "$omlx_join_tmp"\' EXIT && '
        "(command -v curl >/dev/null 2>&1 "
        "&& command -v python3 >/dev/null 2>&1 "
        "|| (command -v apt-get >/dev/null 2>&1 "
        "&& sudo apt-get update "
        "&& sudo apt-get install -y ca-certificates curl python3)) && "
        f"curl --proto '=http,https' --max-filesize 1048576 -fsSL "
        f"{shlex.quote(bootstrap_url)} "
        '-o "$omlx_join_tmp" && '
        f"printf '%s  %s\\n' {shlex.quote(digest)} \"$omlx_join_tmp\" "
        "| sha256sum -c - && "
        'sudo python3 "$omlx_join_tmp" '
        f"--controller {shlex.quote(controller_url)} "
        f"--join-key {shlex.quote(join_key)} "
        "--controller-key-fingerprint "
        f"{shlex.quote(controller_key_fingerprint)} "
        f"--source-digest {source_digest})"
    )
