# SPDX-License-Identifier: Apache-2.0
"""Best-effort process title helpers."""

from __future__ import annotations

import sys

DEFAULT_PROCESS_TITLE = "omlx-server"


def set_process_title(title: str = DEFAULT_PROCESS_TITLE) -> bool:
    """Set the process title when optional platform support is available.

    Returns True when the native process title was updated. If the optional
    dependency is not installed, argv[0] is still updated for Python-level
    command-line displays and the function returns False.
    """
    if title:
        sys.argv[0] = title

    try:
        from setproctitle import setproctitle
    except ImportError:
        return False

    setproctitle(title)
    return True
