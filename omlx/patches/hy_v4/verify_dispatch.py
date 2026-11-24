# SPDX-License-Identifier: Apache-2.0
"""Short, batched Hy4 draft verification on the validated Metal backend."""

from functools import cache
from threading import local

import mlx.core as mx

from . import exact

_state = local()


def set_armed(flag: bool) -> None:
    _state.armed = flag


def armed() -> bool:
    """Whether a regular Lightning MTP target verify forward is active."""
    return getattr(_state, "armed", False) and not exact.armed()


@cache
def _supported_gpu() -> bool:
    # The fast paths reproduce the reduction layout in this MLX version.
    return (
        mx.__version__ == "0.32.4"
        and mx.device_info().get("architecture") == "applegpu_g17s"
    )


def eligible(rows: int) -> bool:
    return (
        2 <= rows <= 6
        and armed()
        and mx.default_device() == mx.gpu
        and _supported_gpu()
    )
