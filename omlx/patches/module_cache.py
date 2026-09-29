# SPDX-License-Identifier: Apache-2.0
"""Per-module caches of static decode decisions.

One-token decode re-asks the same layout questions of every layer on every
step (quantization signatures, shapes, dtypes of weights that never change).
``cached_per_module`` answers them once per module and keeps the answer in
the module's ``__dict__``. The entry is keyed on the identity of every value
of the module and of its child modules down to ``depth`` levels (submodules,
weights, scales, biases), the module's training flag and caller ``flags``:
reassigning any of them rebuilds the entry on the next call. The entry keeps
the keyed objects alive, so an identity match cannot come from a new object
reusing a freed one's address. Plain configuration attributes of children
(bits, group_size, eps, ...) are construction-time constants and are not
re-checked.

Identities are compared with ``operator.is_`` rather than ``id()``: once an
audit hook is installed (mlx-vlm's imports install one), every ``id()`` call
raises an audit event and costs several times more.

Built values must not reference ``module`` itself (only its children and
tensors), so the entry creates no reference cycle through the module.
"""

from __future__ import annotations

from itertools import chain
from operator import is_
from typing import Any, Callable

import mlx.nn as nn


def _children(module: nn.Module, depth: int) -> list:
    children: list = []
    level = [module]
    for _ in range(depth):
        level = [
            value
            for parent in level
            for value in dict.values(parent)
            if isinstance(value, nn.Module)
        ]
        children.extend(level)
    return children


def cached_per_module(
    module: nn.Module,
    slot: str,
    build: Callable[[nn.Module], Any],
    *,
    depth: int = 1,
    flags: tuple = (),
) -> Any:
    """``build(module)``, cached in ``module.__dict__[slot]`` (see module docstring)."""
    state = (module._training, flags)
    entry = module.__dict__.get(slot)
    if entry is not None:
        children, refs, cached_state, value = entry
        if cached_state == state:
            current = tuple(chain(dict.values(module), *map(dict.values, children)))
            if len(current) == len(refs) and all(map(is_, current, refs)):
                return value
    value = build(module)
    children = _children(module, depth)
    refs = tuple(chain(dict.values(module), *map(dict.values, children)))
    module.__dict__[slot] = (children, refs, state, value)
    return value


__all__ = ["cached_per_module"]
