# SPDX-License-Identifier: Apache-2.0
"""This Mac running another Mac's prefill helper, seen through its marker file.

The helper process writes the marker while it holds layers here, so this Mac's
server can show it, stop it, and explain a model load that no longer fits.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def active_marker_path() -> Path:
    return Path.home() / ".omlx" / "prefill-helper" / "active.json"


def _is_helper(pid: int) -> bool:
    """Whether ``pid`` is still a prefill helper (pids are reused)."""
    try:
        command = subprocess.run(
            ["/bin/ps", "-o", "command=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "omlx.prefill_helper.worker" in command


def active_lending() -> dict | None:
    """The helper running on this Mac, if its process is alive."""
    try:
        marker = json.loads(active_marker_path().read_text())
        pid = int(marker["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not _is_helper(pid):
        # A helper that died without clearing its marker.
        return None
    return marker


def lending_note() -> str:
    """Why a model load may not fit while a helper holds memory here."""
    marker = active_lending()
    if marker is None:
        return ""
    return (
        f" This Mac is lending {float(marker.get('active_gib') or 0):.0f} GB to "
        f"{marker.get('host') or 'another Mac'} as a Prefill Accelerator; press "
        "Stop on the dashboard banner to free it."
    )
