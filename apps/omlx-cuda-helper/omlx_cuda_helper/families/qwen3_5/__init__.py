# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 / 3.6 / 3.8 dense (``qwen3_5``): planning without a GPU library."""

from __future__ import annotations

from .._qwen35_plan import (
    CONTRACT,
    FIRST_STAGE_MIN,
    check_layout,
    linear_layers,
    plan,
    resident,
    text_config,
)

__all__ = [
    "CONTRACT",
    "FIRST_STAGE_MIN",
    "check_layout",
    "linear_layers",
    "plan",
    "resident",
    "text_config",
]
