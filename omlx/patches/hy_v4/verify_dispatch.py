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
    return mx.device_info().get("architecture") in ("applegpu_g17s", "applegpu_g17d")


def short_gpu_verify(rows: int) -> bool:
    """Regular short verification, independent of the native kernel version."""
    return 2 <= rows <= 6 and armed() and mx.default_device() == mx.gpu


def eligible(rows: int) -> bool:
    return short_gpu_verify(rows) and _supported_gpu()
