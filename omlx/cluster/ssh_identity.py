# SPDX-License-Identifier: Apache-2.0
"""Shared SSH identity constants used by every cluster SSH code path.

Kept import-free of the rest of the cluster package so ``ssh_policy``,
``ssh_keys``, ``provisioning`` and ``inventory`` can all reference the same
values without import cycles.
"""

from __future__ import annotations

import getpass
import os

_SSH_PORT = 22
_MANAGED_IDENTITY = "~/.ssh/omlx_cluster"


def default_ssh_user() -> str:
    """Return the current OS login name used as the SSH user by default."""

    try:
        return getpass.getuser()
    except OSError:
        return os.environ.get("USER") or os.environ.get("LOGNAME", "")
