# SPDX-License-Identifier: Apache-2.0
"""Compatibility helpers for renamed dataclass fields."""

from __future__ import annotations

import functools


def deprecated_init_kwargs(**aliases: str):
    """Allow a dataclass constructor to accept renamed keyword arguments.

    Apply above ``@dataclass`` so ``Cls(legacy=...)`` and
    ``dataclasses.replace(cfg, legacy=...)`` keep working alongside a
    property alias. An explicitly passed legacy keyword (not ``None``)
    overrides the canonical keyword because ``replace()`` re-materializes
    every init field from the existing instance.
    """

    def decorator(cls):
        orig_init = cls.__init__

        @functools.wraps(orig_init)
        def wrapped_init(self, *args, **kwargs):
            for legacy, canonical in aliases.items():
                if legacy in kwargs:
                    value = kwargs.pop(legacy)
                    if value is not None:
                        kwargs[canonical] = value
            orig_init(self, *args, **kwargs)

        cls.__init__ = wrapped_init
        return cls

    return decorator
