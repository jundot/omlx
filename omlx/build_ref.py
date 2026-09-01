# SPDX-License-Identifier: Apache-2.0
"""Resolve the source ref (branch or tag) this instance is running from.

Local development ships the same tree to several Macs at once, so the dashboard
has to say which branch a given node is actually on. Two delivery shapes exist
and both must work:

* the packaged Mac app has no ``.git`` directory, so the ref is baked in at
  deploy time via ``OMLX_BUILD_REF`` or a ``_build_ref.txt`` stamp file;
* a source checkout has ``.git``, so git itself is the authority.

Baked-in values win; git is the fallback. When nothing resolves the caller gets
``None`` and the dashboard simply omits the badge.
"""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path

_ENV_VAR = "OMLX_BUILD_REF"
_STAMP = Path(__file__).with_name("_build_ref.txt")
_REPO_ROOT = Path(__file__).resolve().parent.parent
_MAX_LEN = 64
_GIT_TIMEOUT = 2.0


def _clean(value: str | None) -> str | None:
    """Return the first line of ``value`` as a display-safe ref, else None.

    ``HEAD`` is rejected on purpose: ``rev-parse --abbrev-ref`` prints it for a
    detached checkout, where the short SHA is the useful answer instead.
    """

    if not value:
        return None
    lines = value.strip().splitlines()
    ref = lines[0].strip() if lines else ""
    if not ref or ref == "HEAD":
        return None
    if any(ord(char) < 32 for char in ref):
        return None
    return ref[:_MAX_LEN]


def _from_env() -> str | None:
    return _clean(os.environ.get(_ENV_VAR))


def _from_stamp(stamp: Path | None = None) -> str | None:
    path = _STAMP if stamp is None else stamp
    try:
        return _clean(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        return None


def _git(repo: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def _from_git(repo: Path | None = None) -> str | None:
    root = _REPO_ROOT if repo is None else repo
    # ``.git`` is a directory in a normal clone and a file in a worktree.
    if not (root / ".git").exists():
        return None
    branch = _clean(_git(root, "rev-parse", "--abbrev-ref", "HEAD"))
    if branch is not None:
        return branch
    return _clean(_git(root, "rev-parse", "--short", "HEAD"))


@lru_cache(maxsize=1)
def build_ref() -> str | None:
    """Deploy-time ref, else the checkout's git ref, else ``None``.

    Cached: the answer cannot change without restarting the process. Tests call
    ``build_ref.cache_clear()``.
    """

    return _from_env() or _from_stamp() or _from_git()
