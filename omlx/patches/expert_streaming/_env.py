# SPDX-License-Identifier: Apache-2.0
"""Tolerant env-knob parsing for module-level constants.

A malformed value (``OMLX_EXPERT_STREAMING_QD=abc``) must degrade to the
default, not raise ValueError at import and take the whole module down.
"""

from __future__ import annotations

import os


def env_int(name: str, default: int, lo: int | None = None) -> int:
    try:
        v = int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        v = int(default)
    return v if lo is None else max(lo, v)


def env_float(name: str, default: float) -> float:
    try:
        v = float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        v = float(default)
    return v
