# SPDX-License-Identifier: Apache-2.0
"""MiniMax M3 / M3.1 (minimax_m3_vl): split at any layer.

Only the hidden state [B, L, hidden] crosses the cut. Each layer's cache gains
the chunk's K/V rows (and indexer key rows on sparse-index layers), which the
helper sends as they are. Every layer derives its RoPE offset from its own
cache, so both Macs build the mask from their first layer.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx_vlm.utils as vlm_utils
from mlx_vlm.models.base import create_attention_mask

from ...utils.model_loading import (
    expand_per_layer_quant_keys,
    maybe_apply_pre_load_patches,
)
from ..adapter import Arrays, HostArrays, StageAdapter

# Calls this short keep explicit positions, as the language model does.
_EXPLICIT_POSITIONS_MAX = 16


def _layers(lm: Any) -> list:
    return lm.model.layers


def _run_layers(lm: Any, h: mx.array, layers: list, cache: list) -> mx.array:
    mask = create_attention_mask(h, cache[0])
    length = h.shape[1]
    positions = None
    if length <= _EXPLICIT_POSITIONS_MAX:
        offset = cache[0].offset
        positions = mx.arange(offset, offset + length, dtype=mx.int32)[None]
    for layer, c in zip(layers, cache):
        h = layer(h, mask, c, position_ids=positions)
    return h


def _cache_state(i: int, item: Any, since: int) -> Arrays:
    """K/V (and indexer key) rows a layer gained after ``since``."""
    kv = getattr(item, "kv_cache", item)
    out = {
        f"{i}.k": kv.keys[..., since : kv.offset, :],
        f"{i}.v": kv.values[..., since : kv.offset, :],
    }
    if getattr(item, "index_keys", None) is not None:
        out[f"{i}.ik"] = item.index_keys[..., since : item.index_offset, :]
    return out


class MiniMaxM3Adapter(StageAdapter):
    model_types = ("minimax_m3_vl",)

    def layer_count(self, lm: Any) -> int:
        return len(_layers(lm))

    def resident(self, tensor: str) -> bool:
        # Vision and the draft head never load on the helper.
        return tensor.startswith("language_model.") and ".mtp." not in tensor

    def keep_first(self, lm: Any, k: int) -> None:
        lm.model.layers = _layers(lm)[:k]

    @contextmanager
    def use_first(self, lm: Any, k: int):
        layers = _layers(lm)
        lm.model.layers = layers[:k]
        try:
            yield
        finally:
            lm.model.layers = layers

    def load_first(self, path: str, k: int) -> Any:
        maybe_apply_pre_load_patches(path, None, for_vlm=True)
        # The server's MiniMax MoE sanitize wrapper, so packed and unpacked
        # shared experts load as they do there.
        engine = importlib.import_module("omlx.engine.vlm")
        load_config = vlm_utils.load_config

        def expanded(model_path: Any, **kwargs: Any) -> dict:
            return expand_per_layer_quant_keys(load_config(model_path, **kwargs))

        vlm_utils.load_config = expanded
        try:
            with engine._force_minimax_m3_moe_sanitize_on_load(Path(path)):
                model = vlm_utils.load_model(Path(path), lazy=True, strict=False)
        finally:
            vlm_utils.load_config = load_config
        lm = model.language_model
        self.keep_first(lm, k)
        mx.eval(lm.parameters())
        return lm

    def run_first(
        self,
        lm: Any,
        cache: list,
        tokens: mx.array,
        next_tokens: int,
        inputs: HostArrays | None = None,
    ) -> tuple[Arrays, Arrays]:
        before = [c.offset for c in cache]
        h = _run_layers(lm, lm.model.embed_tokens(tokens), _layers(lm), cache)
        state: Arrays = {}
        for i, c in enumerate(cache):
            state.update(_cache_state(i, c, before[i]))
        mx.eval(h, list(state.values()))
        return {"h": h}, state

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
        return _run_layers(lm, boundary["h"], _layers(lm)[k:], cache[k:])

    def apply_state(self, cache: list, k: int, state: Arrays) -> None:
        for i in range(k):
            item, keys = cache[i], state[f"{i}.k"]
            if not keys.shape[2]:
                continue
            item.update_and_fetch(keys, state[f"{i}.v"])
            if f"{i}.ik" in state:
                item.update_index_and_fetch(state[f"{i}.ik"])

    def prefix_state(self, cache: list, k: int) -> Arrays:
        state: Arrays = {}
        for i in range(k):
            state.update(_cache_state(i, cache[i], 0))
        mx.eval(list(state.values()))
        return state

    def load_prefix(self, cache: list, state: Arrays) -> None:
        self.apply_state(cache, len(cache), state)
