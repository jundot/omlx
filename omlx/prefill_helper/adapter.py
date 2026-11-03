# SPDX-License-Identifier: Apache-2.0
"""Per-family hooks that split one prefill chunk at layer ``k``.

The helper side keeps layers ``[0, k)`` and calls :meth:`StageAdapter.run_first`
for each chunk. This Mac calls :meth:`StageAdapter.run_rest` with what came
back and then :meth:`StageAdapter.apply_state`. Arrays cross the link as flat
``{name: mx.array}`` dicts so the wire format stays family independent.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import mlx.core as mx

Arrays = dict[str, mx.array]


class StageAdapter:
    """Split a model family's prefill forward between two Macs."""

    model_types: tuple[str, ...] = ()

    def language_model(self, model: Any) -> Any:
        """The module that owns ``layers`` (the scheduler holds an adapter)."""
        return getattr(model, "_language_model", None) or getattr(
            model, "language_model", model
        )

    def layer_count(self, lm: Any) -> int:
        raise NotImplementedError

    def max_split(self, lm: Any) -> int:
        """Largest ``k`` this family supports; the helper holds ``[0, k)``."""
        return self.layer_count(lm) - 1

    def keep_first(self, lm: Any, k: int) -> None:
        """Drop layers ``[k, n)`` before the helper loads weights."""
        raise NotImplementedError

    def load_first(self, path: str, k: int) -> Any:
        """Helper: load layers ``[0, k)`` of a checkpoint; returns the module
        ``run_first`` takes."""
        raise NotImplementedError

    def run_first(
        self, lm: Any, cache: list, tokens: mx.array, next_tokens: int
    ) -> tuple[Arrays, Arrays]:
        """Helper: run ``[0, k)`` on one chunk.

        Returns the tensors that cross the cut and the chunk's cache state
        for layers ``[0, k)``, both evaluated.
        """
        raise NotImplementedError

    def run_rest(
        self,
        prefill: Callable[..., Any],
        lm: Any,
        cache: list,
        tokens: mx.array,
        k: int,
        boundary: Arrays,
        state: Arrays,
        **kwargs: Any,
    ) -> Any:
        """This Mac: run ``[k, n)`` on the chunk the helper sent."""
        raise NotImplementedError

    def apply_state(self, cache: list, k: int, state: Arrays) -> None:
        """This Mac: bring layers ``[0, k)`` to the end of the chunk."""
        raise NotImplementedError

    def prefix_state(self, cache: list, k: int) -> Arrays:
        """This Mac: full state of layers ``[0, k)`` for a cached prefix."""
        raise NotImplementedError

    def load_prefix(self, cache: list, state: Arrays) -> None:
        """Helper: start from the prefix this Mac restored."""
        raise NotImplementedError
