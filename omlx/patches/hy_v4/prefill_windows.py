# SPDX-License-Identifier: Apache-2.0
"""Preserve Hy4's arithmetic windows when one forward spans SSD boundaries."""

from contextlib import contextmanager
from contextvars import ContextVar

_width: ContextVar[int] = ContextVar("hy4_prefill_window_width", default=0)


def window_width() -> int:
    return _width.get()


@contextmanager
def prefill_windows(width: int):
    token = _width.set(width)
    try:
        yield
    finally:
        _width.reset(token)
