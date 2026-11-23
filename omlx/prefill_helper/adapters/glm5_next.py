# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 (glm5_next): split at any layer.

Only the hyper-connection streams ``h`` [B, L, hc_mult, D] cross the cut. Each
layer keeps its own state: a KDA layer a conv and a recurrent state (sent whole
each chunk), a sparse layer its latent KV rows and its pooled index keys (sent
as the rows the chunk added plus the unpooled tail). Masks come from the first
cache of each kind in the stage, so both Macs build them from their own layers.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx_vlm.utils as vlm_utils
from mlx_vlm.models.base import create_attention_mask, create_ssm_mask

from ...patches.mlx_lm_mtp import prompt_priming
from ...patches.moe_gate_up_fusion import apply_moe_gate_up_fusion
from ...utils.layer_pipeline import LayerPipeline
from ...utils.model_loading import (
    expand_per_layer_quant_keys,
    maybe_apply_pre_load_patches,
)
from ..adapter import Arrays, Capture, HostArrays, StageAdapter, language_tensor


def _layers(lm: Any) -> list:
    return lm.model.layers


def _run_layers(
    h: mx.array,
    layers: list,
    cache: list,
    first: int = 0,
    capture: Capture | None = None,
) -> mx.array:
    """``layers`` (global ids from ``first``) on the 4-D streams, with the
    vendored forward's masks and prefill memory policy. A drafter reads a
    layer with its streams averaged."""
    linear = next((c for layer, c in zip(layers, cache) if layer.is_linear), None)
    sparse = next((c for layer, c in zip(layers, cache) if not layer.is_linear), None)
    x = h[:, :, 0]
    fa_mask = create_attention_mask(
        x, sparse[0] if sparse is not None else None, return_array=True
    )
    ssm_mask = create_ssm_mask(x, linear)
    pipeline = (
        LayerPipeline(on_evaluated=mx.clear_cache, lazy_last=True)
        if h.shape[1] >= 256
        else None
    )
    for i, (layer, c) in enumerate(zip(layers, cache), first):
        h = layer(h, mask=ssm_mask if layer.is_linear else fa_mask, cache=c)
        extra = []
        if capture is not None and i in capture.layers:
            capture.take(i, h.mean(axis=2))
            extra.append(capture.out[i])
        if pipeline is not None:
            pipeline.push(h, *extra)
    if pipeline is not None:
        pipeline.drain()
    return h


def _sparse_rows(item: Any) -> tuple[int, int]:
    kv, pool = item[0], item[1]
    return kv.offset, pool.offset


def _sparse_state(i: int, item: Any, since: tuple[int, int]) -> Arrays:
    """Latent KV and pooled rows added after ``since``, plus the pool tail."""
    kv, pool = item[0], item[1]
    out = {}
    if kv.keys is not None:
        out[f"{i}.kv"] = kv.keys[..., since[0] : kv.offset, :]
    if pool.pooled is not None:
        out[f"{i}.pool"] = pool.pooled[:, since[1] :]
    if pool.remainder:
        out[f"{i}.tail_kv"] = pool.buf_kv[:, : pool.remainder]
        out[f"{i}.tail_gate"] = pool.buf_gate[:, : pool.remainder]
    return out


def _append_sparse(i: int, item: Any, state: Arrays) -> None:
    kv, pool = item[0], item[1]
    keys = state.get(f"{i}.kv")
    if keys is not None and keys.shape[2]:
        # The latent cache keeps zero-width values beside its keys.
        kv.update_and_fetch(keys, keys[..., :0])
    rows = state.get(f"{i}.pool")
    if rows is not None and rows.shape[1]:
        pool.update_and_fetch(rows)
    # The unpooled tail replaces the old one, as the state setter does.
    pool.remainder = 0
    pool.buf_kv = pool.buf_gate = None
    tail = state.get(f"{i}.tail_kv")
    if tail is not None:
        pool.accumulate_windows(tail, state[f"{i}.tail_gate"], 0)
    pool._undo = None
    pool._undo_chain = False


class Glm5NextAdapter(StageAdapter):
    model_types = ("glm5_next",)
    contract = 1
    captures_layers = True

    def layer_count(self, lm: Any) -> int:
        return len(_layers(lm))

    def resident(self, tensor: str) -> bool:
        # Vision and the Lightning MTP head never load on the helper.
        return language_tensor(tensor) and ".mtp." not in tensor

    def keep_first(self, lm: Any, k: int) -> None:
        lm.model.layers = _layers(lm)[:k]
        # The MTP head never runs here and is attached even without its files.
        if "mtp" in lm:
            del lm["mtp"]

    @contextmanager
    def use_first(self, lm: Any, k: int):
        layers = _layers(lm)
        lm.model.layers = layers[:k]
        try:
            yield
        finally:
            lm.model.layers = layers

    def load_first(self, path: str, k: int, a8_min_tokens: int | None = None) -> Any:
        maybe_apply_pre_load_patches(path, None, for_vlm=True)
        load_config = vlm_utils.load_config

        def expanded(model_path: Any, **kwargs: Any) -> dict:
            # The server's loader adds the same per-module quantization keys.
            return expand_per_layer_quant_keys(load_config(model_path, **kwargs))

        vlm_utils.load_config = expanded
        try:
            # Layers past the cut have no weights here; they are dropped
            # before anything is evaluated.
            model = vlm_utils.load_model(Path(path), lazy=True, strict=False)
        finally:
            vlm_utils.load_config = load_config
        lm = model.language_model
        self.keep_first(lm, k)
        apply_moe_gate_up_fusion(lm)
        mx.eval(lm.parameters())
        return lm

    def run_first(
        self,
        lm: Any,
        cache: list,
        tokens: mx.array,
        next_tokens: int,
        inputs: HostArrays | None = None,
        capture: Capture | None = None,
    ) -> tuple[Arrays, Arrays]:
        layers = _layers(lm)
        before = {
            i: _sparse_rows(c)
            for i, (layer, c) in enumerate(zip(layers, cache))
            if not layer.is_linear
        }
        h = lm.model.embed_tokens(tokens)
        h = mx.contiguous(
            mx.broadcast_to(
                h[:, :, None, :], (*h.shape[:2], lm.model.hc_mult, h.shape[-1])
            )
        )
        h = _run_layers(h, layers, cache, capture=capture)
        state: Arrays = {}
        for i, (layer, c) in enumerate(zip(layers, cache)):
            if layer.is_linear:
                state[f"{i}.conv"], state[f"{i}.rec"] = c[0], c[1]
            else:
                state.update(_sparse_state(i, c, before[i]))
        mx.eval(h, list(state.values()), list(capture.out.values()) if capture else [])
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
        capture: Capture | None = None,
        **kwargs: Any,
    ) -> Any:
        h = _run_layers(boundary["h"], _layers(lm)[k:], cache[k:], k, capture)
        if not prompt_priming.capture_eligible(lm, cache):
            return None
        # Lightning MTP primes its head from the normed hidden of every chunk.
        return lm.model.norm(h.mean(axis=2))

    def apply_state(self, cache: list, k: int, state: Arrays) -> None:
        for i in range(k):
            item = cache[i]
            if hasattr(item, "caches"):
                _append_sparse(i, item, state)
            else:
                item[0], item[1] = state[f"{i}.conv"], state[f"{i}.rec"]

    def finish_chunk(self, lm: Any, tokens: mx.array, output: Any, cache: list) -> None:
        if output is not None:
            prompt_priming.maybe_capture(lm, tokens, output, cache)

    def prefix_state(self, cache: list, k: int) -> Arrays:
        state: Arrays = {}
        for i in range(k):
            item = cache[i]
            if hasattr(item, "caches"):
                state.update(_sparse_state(i, item, (0, 0)))
            else:
                state[f"{i}.conv"], state[f"{i}.rec"] = item[0], item[1]
        mx.eval(list(state.values()))
        return state

    def load_prefix(self, cache: list, state: Arrays) -> None:
        self.apply_state(cache, len(cache), state)
