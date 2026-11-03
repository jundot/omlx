# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4.1: split at any layer up to the CED midpoint.

Besides ``(h, pre)``, layers after the cut read the shared state of the last
KV source before it: its compressed KV and index keys (rebuilt here from this
Mac's cache plus the chunk's new rows), the chunk's top-k ids and the
candidate blocks.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import mlx.core as mx

from ...patches.deepseek_v41 import growth
from ...patches.deepseek_v41.loading import load
from ..adapter import Arrays, StageAdapter

_SLOTS = 7
# Compressed KV and index keys grow with the prompt on KV-source layers; the
# helper sends only the rows a chunk adds.
_GROWING = (2, 3)
_HISTORY = 6


def _key(layer: int, slot: int) -> str:
    return f"{layer}.{slot}"


def _rows(item: Any, slot: int) -> int:
    value = item[slot]
    return 0 if value is None else int(value.shape[1])


@contextmanager
def _stage_from(lm: Any, k: int):
    start = lm.start_idx
    lm.start_idx = k
    try:
        yield
    finally:
        lm.start_idx = start


class DeepseekV41Adapter(StageAdapter):
    model_types = ("deepseek_v41",)

    def layer_count(self, lm: Any) -> int:
        return lm._config.n_layers

    def max_split(self, lm: Any) -> int:
        # The CED midpoint holds the decoder replay input, so it stays here.
        return lm._config.n_layers // 2

    def keep_first(self, lm: Any, k: int) -> None:
        lm.start_idx, lm.end_idx = 0, k
        lm.layers = lm.layers[:k]

    def load_first(self, path: str, k: int) -> Any:
        # Engram tables stay memory-mapped so the layers get the GPU memory.
        model, _ = load(
            path,
            engram_ssd_offload=True,
            a8_prefill=True,
            stage=lambda lm: self.keep_first(lm, k),
        )
        return model.language_model

    def run_first(
        self, lm: Any, cache: list, tokens: mx.array, next_tokens: int
    ) -> tuple[Arrays, Arrays]:
        before = [{slot: _rows(item, slot) for slot in _GROWING} for item in cache]
        capture: dict[str, Any] = {}
        lm._omlx_prefill(
            tokens,
            cache=cache,
            next_prefill_tokens=next_tokens,
            skip_logits=True,
            _stage_capture=capture,
        )
        shared = capture["shared"]
        boundary = {"h": capture["h"], "pre": capture["pre"]}
        for name in ("idx", "candidates"):
            if name in shared:
                boundary[name] = shared[name]
        state = {}
        for i, item in enumerate(cache):
            for slot in range(_SLOTS):
                value = item[slot]
                if value is None:
                    continue
                if slot in _GROWING and item.compress_ratio:
                    value = value[:, before[i][slot] :]
                state[_key(i, slot)] = value
        mx.eval(list(boundary.values()), list(state.values()))
        return boundary, state

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
        shared = {
            name: boundary[name] for name in ("idx", "candidates") if name in boundary
        }
        sources = [i for i in lm._config.kv_source_layers if i < k]
        if sources:
            src = sources[-1]
            for slot, name in ((2, "kv"), (3, "index_k")):
                old, new = cache[src][slot], state[_key(src, slot)]
                shared[name] = new if old is None else mx.concatenate([old, new], 1)
        stage_input = {"h": boundary["h"], "pre": boundary["pre"], "shared": shared}
        # Engram hashes read the n-gram history from the stage's first layer;
        # the whole-model cache keeps it in layer 0.
        first = cache[k]
        history = first[_HISTORY]
        first[_HISTORY] = cache[0][_HISTORY]
        try:
            with _stage_from(lm, k):
                return prefill(
                    tokens, cache=cache[k:], _stage_input=stage_input, **kwargs
                )
        finally:
            first[_HISTORY] = (
                history if history is not None else mx.zeros((1, 0), mx.int64)
            )

    def apply_state(self, cache: list, k: int, state: Arrays) -> None:
        for i in range(k):
            item = cache[i]
            for slot in range(_SLOTS):
                value = state.get(_key(i, slot))
                if value is None:
                    continue
                if slot in _GROWING and item.compress_ratio:
                    previous = item[slot]
                    start = 0 if previous is None else int(previous.shape[1])
                    # Keep the in-place growth chain decode appends to.
                    item[slot] = growth.append(
                        item,
                        slot,
                        value if previous is None else previous,
                        value,
                        start,
                    )
                else:
                    item[slot] = value

    def prefix_state(self, cache: list, k: int) -> Arrays:
        state = {
            _key(i, slot): cache[i][slot]
            for i in range(k)
            for slot in range(_SLOTS)
            if cache[i][slot] is not None
        }
        mx.eval(list(state.values()))
        return state

    def load_prefix(self, cache: list, state: Arrays) -> None:
        for i, item in enumerate(cache):
            values = [state.get(_key(i, slot)) for slot in range(_SLOTS)]
            if any(value is not None for value in values):
                item.cache = values
