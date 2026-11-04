# SPDX-License-Identifier: Apache-2.0
"""This Mac running another Mac's prefill helper, seen through its marker file.

The helper process writes the marker while it holds layers here, so this Mac's
server can show it, stop it, and explain a model load that no longer fits. The
host's checks and helper starts also note here which Mac set this one up.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def active_marker_path() -> Path:
    return Path.home() / ".omlx" / "prefill-helper" / "active.json"


def host_marker_path() -> Path:
    return Path.home() / ".omlx" / "prefill-helper" / "host.json"


def record_host(node_id: str, name: str) -> None:
    """Remember the Mac that uses this one as its Prefill Accelerator."""
    path = host_marker_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"node_id": node_id, "name": name}))


def recorded_host() -> dict | None:
    try:
        host = json.loads(host_marker_path().read_text())
    except (OSError, ValueError):
        return None
    return host if isinstance(host, dict) and host.get("node_id") else None


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
