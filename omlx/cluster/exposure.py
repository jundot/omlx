# SPDX-License-Identifier: Apache-2.0
"""Feature exposure policy for experimental distributed inference."""

from __future__ import annotations

from typing import Any


def distributed_inference_enabled(settings: Any | None) -> bool:
    """Whether settings expose the distributed surface.

    On by default (user decision, 2026-11-16): a fresh install must reach
    the Cluster Assistant without hunting for this switch.
    """

    return bool(
        settings is not None
        and getattr(
            settings.server,
            "distributed_inference_enabled",
            True,
        )
    )
