# SPDX-License-Identifier: Apache-2.0
"""Publish an interpreter a peer coordinator can actually run (#2680).

``discover_remote_python_executable`` has always probed
``~/.omlx/bin/omlx-cluster-python`` first, but nothing ever wrote that file.
The packaged app ships only ``~/.omlx/bin/omlx``, and that is a ``-m omlx.cli``
launcher rather than an interpreter — it cannot answer ``-c 'import omlx'`` and
cannot serve as the ``python_executable`` the memory-ceiling and preflight
probes run their own scripts under.  Every candidate therefore failed on a Mac
that had oMLX installed, and the peer was reported as "worker runtime is not
installed" while the app sat in /Applications.

The server is the one process that already holds the answer: its own
``sys.executable`` plus the ``PYTHONHOME``/``PYTHONPATH`` its launcher
assembled.  Writing those into a tiny shell shim makes the peer discoverable
from any install mode — packaged app, pip, uv, or a source checkout — without
asking anybody to create a file by hand.
"""

from __future__ import annotations

import logging
import os
import shlex
import stat
import sys
from collections.abc import Mapping
from pathlib import Path

logger = logging.getLogger(__name__)

#: The exact candidate ``discover_remote_python_executable`` looks for.
CLUSTER_PYTHON_SHIM = "~/.omlx/bin/omlx-cluster-python"

_SHIM_DIRECTORY = (".omlx", "bin")
_SHIM_NAME = "omlx-cluster-python"
_SHIM_MODE = 0o755

# What the packaged launcher sets before the bundled interpreter can import
# omlx. Anything unset here is deliberately not mentioned in the shim: an
# exported empty PYTHONHOME breaks a pip or source install outright.
_INHERITED_VARIABLES = (
    "PYTHONHOME",
    "PYTHONPATH",
    "PYTHONDONTWRITEBYTECODE",
    "OMLX_BASE_PATH",
)


def _render(executable: str, environ: Mapping[str, str]) -> str:
    exports = "".join(
        f"export {name}={shlex.quote(value)}\n"
        for name in _INHERITED_VARIABLES
        if (value := (environ.get(name) or "").strip())
    )
    return (
        "#!/bin/sh\n"
        "# Written by oMLX so another Mac can run this node's interpreter over\n"
        "# SSH. Do not edit; it is rewritten on every server start.\n"
        f"{exports}"
        f"exec {shlex.quote(executable)} \"$@\"\n"
    )


def ensure_cluster_python_shim(
    *,
    home: Path | None = None,
    executable: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path | None:
    """Install the shim a coordinator discovers over SSH.

    Best effort by design: this runs during server start-up, and a read-only or
    unusual home directory must never keep the node from serving inference.
    Returns the shim path on success, ``None`` when it could not be written.
    """

    executable = executable or sys.executable
    environ = os.environ if environ is None else environ
    if not Path(executable).is_absolute():
        # A bare name resolves against the peer's PATH at SSH time, which is a
        # different PATH than this process has. Refuse rather than publish a
        # shim that works here and fails there.
        logger.debug("Not publishing a cluster shim for %r: not absolute", executable)
        return None

    directory = (home or Path.home()).joinpath(*_SHIM_DIRECTORY)
    target = directory / _SHIM_NAME
    script = _render(executable, environ)
    temporary = directory / f".{_SHIM_NAME}.new"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        # Rewriting in place would truncate the script out from under a peer
        # probe that is executing it right now, so swap it in atomically — and
        # skip the swap entirely when nothing changed.
        if (
            target.is_file()
            and target.read_text(encoding="utf-8") == script
            and target.stat().st_mode & stat.S_IXUSR
        ):
            return target
        temporary.write_text(script, encoding="utf-8")
        temporary.chmod(_SHIM_MODE)
        os.replace(temporary, target)
    except OSError as exc:
        logger.warning(
            "Could not publish %s for %s: %r", CLUSTER_PYTHON_SHIM, executable, exc
        )
        try:
            temporary.unlink(missing_ok=True)
        except OSError as cleanup_exc:  # pragma: no cover - nothing further to try
            logger.warning(
                "Could not remove the partial shim at %s: %r", temporary, cleanup_exc
            )
        return None
    logger.info("Published %s -> %s", CLUSTER_PYTHON_SHIM, executable)
    return target
