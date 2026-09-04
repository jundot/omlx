# SPDX-License-Identifier: Apache-2.0
"""Feature exposure policy for experimental distributed inference."""

from __future__ import annotations

from typing import Any


def distributed_inference_enabled(settings: Any | None) -> bool:
    """Whether settings explicitly expose the distributed surface."""

    return bool(
        settings is not None
        and getattr(
            settings.server,
            "distributed_inference_enabled",
            False,
        )
    )
