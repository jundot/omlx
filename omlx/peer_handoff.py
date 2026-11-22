# SPDX-License-Identifier: Apache-2.0
"""Free memory held by a resident peer service before oMLX loads a model.

oMLX can share one Mac with another service that keeps models resident on the
GPU (a distributed-inference provider, for example). The two compete for the
same unified memory: a load that would fit on an otherwise idle machine is
refused outright while the peer is holding memory, and the user has to stop
that peer by hand before retrying.

This hook runs inside the admission path. When a model about to be loaded does
not fit, it asks the peer to drain and stop, waits for it to actually exit, and
lets the normal admission code decide again with the memory freed.

The whole thing is best-effort: a missing binary, a timeout, or a broken
environment only logs and falls through to the usual error, so oMLX behaves
exactly as before when no peer is running.

Environment:

``OMLX_PEER_HANDOFF``
    Command used to stop the peer service, run through the shell. Defaults to
    ``darkbloom stop --force --timeout 20 --yes``. Set to an empty string to
    disable the handoff entirely.

``OMLX_PEER_STATUS``
    Command that exits 0 while the peer still holds memory. Defaults to
    ``darkbloom status``. The hook also accepts a status command whose output
    merely contains ``Daemon: running``.
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import time

logger = logging.getLogger(__name__)

DEFAULT_STATUS = "darkbloom status"
DEFAULT_STOP = "darkbloom stop --force --timeout 20 --yes"

# Where the Darkbloom installer leaves its CLI when it is not on PATH.
_PEER_CLI_FALLBACK = "~/.darkbloom/bin/darkbloom"

# Marker printed by the Darkbloom CLI while its provider is serving.
_RUNNING_MARKER = "Daemon: running"


def _status_command() -> str:
    return os.environ.get("OMLX_PEER_STATUS", DEFAULT_STATUS).strip()


def _stop_command() -> str:
    return os.environ.get("OMLX_PEER_STOP", DEFAULT_STOP).strip()


def _resolve(argv: list[str]) -> list[str]:
    """Make the command runnable when its binary is not on PATH."""
    if not argv:
        return argv
    head = argv[0]
    if os.path.dirname(head):
        return argv
    found = shutil.which(head)
    if found:
        argv[0] = found
        return argv
    fallback = os.path.expanduser(_PEER_CLI_FALLBACK)
    if os.path.exists(fallback):
        argv[0] = fallback
    return argv


def peer_running() -> bool:
    """True when the configured peer service is still holding memory."""
    command = _status_command()
    if not command:
        return False
    try:
        proc = subprocess.run(
            _resolve(shlex.split(command)), capture_output=True, text=True, timeout=15
        )
    except Exception:
        return False
    if proc.returncode == 0:
        return True
    return _RUNNING_MARKER in ((proc.stdout or "") + (proc.stderr or ""))


def release_peer_memory(model_id: str, needed_bytes: int = 0, wait_seconds: int = 90) -> bool:
    """Stop the peer service so ``model_id`` can be admitted.

    Returns True when the peer was running and has now exited, so the caller
    should re-run its admission check.
    """
    if not peer_running():
        return False
    stop = _stop_command()
    if not stop:
        return False

    needed = f"{needed_bytes / 1e9:.1f} GB" if needed_bytes else "more memory"
    logger.warning(
        "oMLX needs %s to load %r; stopping the resident peer service to free it",
        needed,
        model_id,
    )
    try:
        subprocess.run(
            _resolve(shlex.split(stop)),
            capture_output=True,
            text=True,
            timeout=wait_seconds + 30,
        )
    except Exception:
        logger.warning("Peer stop command did not complete", exc_info=True)
        return False

    # macOS keeps the exited process's pages around as inactive memory rather
    # than returning them to the free list, so waiting for a free-byte target
    # would burn the whole timeout on every handoff. Wait for the peer to
    # disappear instead, then give the kernel a moment to unmap its buffers.
    deadline = time.time() + min(wait_seconds, 90)
    while time.time() < deadline:
        if not peer_running():
            time.sleep(3)
            return True
        time.sleep(1)
    logger.warning("Peer stop returned but the peer is still running")
    return False
