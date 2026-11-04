# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8 Flash-Next (qwen4_exp): split at any layer.

The residual streams cross the cut with the last layer's pending write applied
([B, L, hc_count * hidden]). A linear layer sends its conv and recurrent state
(and the PLE layer its conv window and token history) whole each chunk; a QSA
layer sends the K/V, indexer keys and positions the chunk appended. The PLE
table stays here: this Mac reads each chunk's rows and sends them, so the
helper holds no table. Masks and positions come from each side's own layers,
since the model-level forward indexes caches by absolute layer.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx_vlm.utils as vlm_utils
import numpy as np

from ...patches.mlx_lm_mtp import prompt_priming
from ...patches.mlx_vlm_qwen4_exp_compat import configure_qwen4_exp_runtime
from ...patches.moe_gate_up_fusion import apply_moe_gate_up_fusion
from ...patches.qwen35_fa256_attention import apply_qwen35_fa256_attention_patch
from ...patches.qwen35_gdn_chunked import apply_qwen35_gdn_prefill_patch
from ...patches.qwen35_gdn_prework import (
    apply_qwen35_gdn_prework_patch,
    configure_qwen4_decode,
)
from ...patches.qwen35_oq_a8 import apply_qwen35_oq_a8_patch
from ...patches.qwen35_q4_mlp import (
    apply_qwen35_q4_mlp_patch,
    apply_qwen35_q4_prefill_linear_patch,
)
from ...patches.sdpa256_attention import apply_sdpa256_attention_patch
from ...utils.model_loading import (
    expand_per_layer_quant_keys,
    maybe_apply_pre_load_patches,
)
from .. import wire
from ..adapter import Arrays, Capture, HostArrays, StageAdapter, language_tensor

_WARM = object()


def _language(lm: Any) -> Any:
    """The vendored language module (loaded only after the compat patch)."""
    return sys.modules[type(lm.model).__module__]


def _layers(lm: Any) -> list:
    return lm.model.layers


def _is_qsa(item: Any) -> bool:
    return hasattr(item, "update_indexer")


def _positions(item: Any, length: int) -> mx.array:
    # Text-only prefill: positions continue from the cache offset.
    return mx.arange(item.offset, item.offset + length, dtype=mx.int32)[None]


def _run_layers(
    lm: Any, h: mx.array, tokens: mx.array, layers: list, cache: list
) -> mx.array:
    """``layers`` on the residual streams; returns them with every write applied."""
    language = _language(lm)
    attention = next(
        (c for layer, c in zip(layers, cache) if not layer.is_linear), None
    )
    linear = next((c for layer, c in zip(layers, cache) if layer.is_linear), None)
    fa_mask = (
        language._create_qwen3_5_attention_mask(h, attention)
        if attention is not None
        else None
    )
    ssm_mask = (
        language.q35_language._create_qwen3_5_ssm_mask(h, linear)
        if linear is not None
        else None
    )
    positions = _positions(attention, h.shape[1]) if attention is not None else None
    defer = language.hc_fused.write_enabled()
    write = None
    for layer, c in zip(layers, cache):
        h = layer(
            h,
            tokens,
            mask=ssm_mask if layer.is_linear else fa_mask,
            cache=c,
            position_ids=positions,
            write=write,
            defer_write=defer,
        )
        if defer:
            h, write = h[0], h[1:]
    return h if write is None else language._hc_write(h, *write)


def _host_view(array: mx.array) -> tuple[np.ndarray, str]:
    """A zero-copy numpy view of an evaluated array and its wire dtype."""
    name = str(array.dtype).rsplit(".", 1)[-1]
    if array.dtype == mx.bfloat16:
        array = array.view(mx.uint16)
    mx.eval(array)
    return np.asarray(array), name


_SAFETENSORS_DTYPES = {"BF16": "bfloat16", "F16": "float16", "F32": "float32"}


def _ple_reader(table: Any, language: Any) -> Callable[[np.ndarray], tuple]:
    """Packed rows of this Mac's PLE table by global row, without the GPU.

    Returns ``read(rows) -> (arrays, settings)``; ``settings`` is
    [bits, group_size, cast_to_scale_dtype, fp8], as the helper decodes them.
    """
    if isinstance(table, language.DiskBackedShardedEmbedding):

        def read_disk(rows: np.ndarray) -> tuple:
            plan = table._plan(rows)
            if plan is None:
                raise RuntimeError("PLE shards with mixed formats are not supported")
            buffers = table._assemble(rows, plan)
            _, _, _, _, families, dtypes, bits, group_size = plan
            arrays = [
                (buffers[f], _SAFETENSORS_DTYPES.get(dtypes[f], dtypes[f].lower()))
                for f in families
            ]
            fp8 = int(str(dtypes[0]).upper().startswith("F8"))
            return arrays, [bits or 0, group_size or 0, 1, fp8]

        return read_disk
    fused = getattr(table, "fused", None)
    shards = [fused] if fused is not None else list(table.shards)
    offsets = np.asarray(
        [0, table.shard_offsets[-1]] if fused is not None else table.shard_offsets,
        dtype=np.int64,
    )
    quantized = isinstance(shards[0], nn.QuantizedEmbedding)
    families = ("weight", "scales", "biases") if quantized else ("weight",)
    views = [[_host_view(getattr(s, f)) for f in families] for s in shards]
    fp8 = int(views[0][0][1] == "uint8")
    settings = [
        shards[0].bits if quantized else 0,
        shards[0].group_size if quantized else 0,
        0,
        fp8,
    ]

    def read_resident(rows: np.ndarray) -> tuple:
        shard = np.searchsorted(offsets, rows, side="right") - 1
        local = rows - offsets[shard]
        out = [np.empty((rows.size, view.shape[1]), view.dtype) for view, _ in views[0]]
        for index in np.unique(shard):
            positions = np.flatnonzero(shard == index)
            for family, (view, _) in enumerate(views[index]):
                out[family][positions] = view[local[positions]]
        return [(o, name) for o, (_, name) in zip(out, views[0])], settings

    return read_resident


class _HostPLERows:
    """A helper's PLE table: this Mac reads each chunk's rows and sends them."""

    def __init__(self, dims: int, dtype: mx.Dtype):
        self.dims, self.dtype = dims, dtype
        self.rows: Any = None

    def put(self, inputs: HostArrays | None, layer: int) -> None:
        if inputs is None:
            self.rows = _WARM
            return
        name = f"p{layer}."
        if name + "ids" not in inputs:
            raise RuntimeError(f"No PLE rows for layer {layer}")
        data = [
            wire.from_host(*inputs[f"{name}{n}"])
            for n in range(3)
            if f"{name}{n}" in inputs
        ]
        self.rows = (
            inputs[name + "ids"][0],
            data,
            wire.from_host(*inputs[name + "ws"]),
            inputs[name + "q"][0].tolist(),
        )

    def __call__(self, indices: mx.array) -> mx.array:
        host = np.asarray(indices).astype(np.int64)
        rows, self.rows = self.rows, None
        if rows is _WARM:
            return mx.zeros((*host.shape, self.dims), self.dtype)
        if rows is None or not np.array_equal(rows[0], host):
            raise RuntimeError("PLE rows were sent for other ids")
        _, data, scale, (bits, group_size, cast, fp8) = rows
        values = data[0]
        if fp8:
            values = mx.from_fp8(values, dtype=mx.bfloat16)
        if bits:
            values = mx.dequantize(
                values,
                data[1],
                data[2],
                group_size=group_size,
                bits=bits,
                mode="affine",
            )
        if cast:
            values = values.astype(scale.dtype)
        return (values * scale).reshape(*host.shape, self.dims)


@contextmanager
def _no_ple_fuse(module: Any):
    """Keep ``Model.load_weights`` from joining PLE shards the helper lacks."""
    fuse = module.fuse_resident_ple_embeddings
    module.fuse_resident_ple_embeddings = lambda model, **kwargs: 0
    try:
        yield
    finally:
        module.fuse_resident_ple_embeddings = fuse


def _cache_state(i: int, item: Any, since: int | None) -> Arrays:
    """A layer's state after a chunk: QSA rows from ``since``, else every slot."""
    if not _is_qsa(item):
        return {f"{i}.{s}": v for s, v in enumerate(item.cache) if v is not None}
    start = since or 0
    return {
        f"{i}.k": item.keys[..., start : item.offset, :],
        f"{i}.v": item.values[..., start : item.offset, :],
        f"{i}.ik": item.index_keys[:, start:],
        f"{i}.ip": item.index_position_ids[..., start:],
    }


class Qwen4ExpAdapter(StageAdapter):
    model_types = ("qwen4_exp",)

    def layer_count(self, lm: Any) -> int:
        return len(_layers(lm))

    def resident(self, tensor: str) -> bool:
        # The PLE table stays on this Mac; vision and MTP never load.
        return (
            language_tensor(tensor)
            and ".mtp." not in tensor
            and ".ngram_embedding.shard" not in tensor
        )

    def keep_first(self, lm: Any, k: int) -> None:
        lm.model.layers = _layers(lm)[:k]
        for layer in lm.model.layers:
            if "ple" in layer:
                embedding = layer.ple.ple_embedding
                table = embedding.ngram_embedding
                embedding.ngram_embedding = _HostPLERows(
                    table.dims, table.weight_scale.dtype
                )

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
        # Resident tables open no PLE files; keep_first drops the table.
        configure_qwen4_exp_runtime(path, "resident")
        model_module = importlib.import_module("mlx_vlm.models.qwen4_exp.qwen4_exp")
        load_config = vlm_utils.load_config

        def expanded(model_path: Any, **kwargs: Any) -> dict:
            return expand_per_layer_quant_keys(load_config(model_path, **kwargs))

        vlm_utils.load_config = expanded
        try:
            with _no_ple_fuse(model_module):
                model = vlm_utils.load_model(Path(path), lazy=True, strict=False)
        finally:
            vlm_utils.load_config = load_config
        lm = model.language_model
        self.keep_first(lm, k)
        apply_moe_gate_up_fusion(lm)
        # The server's engine start installs these prefill kernels.
        apply_sdpa256_attention_patch()
        apply_qwen35_fa256_attention_patch()
        configure_qwen4_decode(model, wide_projections=False)
        apply_qwen35_gdn_prework_patch()
        apply_qwen35_gdn_prefill_patch()
        apply_qwen35_q4_mlp_patch()
        apply_qwen35_q4_prefill_linear_patch(model)
        if a8_min_tokens is not None:
            apply_qwen35_oq_a8_patch(model, min_tokens=a8_min_tokens)
        mx.eval(lm.parameters())
        return lm

    def prefetch(
        self, lm: Any, cache: list, current: Any, following: Any, start: int = 0
    ) -> None:
        # Only an SSD-backed PLE table on this Mac reads ahead.
        if any("ple" in layer for layer in _layers(lm)[start:]):
            lm.prefetch_ple(following, current)

    def job_feed(
        self, lm: Any, cache: list | None, tokens: np.ndarray, grid: list[int], k: int
    ) -> Callable[[int], HostArrays] | None:
        layers = _layers(lm)[:k]
        index = next((i for i, layer in enumerate(layers) if "ple" in layer), None)
        if index is None:
            return None
        embedding = layers[index].ple.ple_embedding
        table = embedding.ngram_embedding
        ids = mx.array(np.asarray(tokens, dtype=np.int64).reshape(1, -1))
        history = mx.concatenate(
            [embedding._previous_context(ids, cache[index] if cache else None), ids],
            axis=-1,
        )
        # Every chunk's rows at once, on this thread; reads run on the feeder.
        rows = np.asarray(embedding._ngram_indices(history, ids.shape[1]))
        read = _ple_reader(table, _language(lm))
        scale = wire.to_host(table.weight_scale)
        starts = np.cumsum([0, *grid]).tolist()
        name = f"p{index}."

        def feed(chunk: int) -> HostArrays:
            ids_chunk = rows[:, starts[chunk] : starts[chunk + 1]]
            arrays, settings = read(ids_chunk.reshape(-1))
            out: HostArrays = {
                name + "ids": (ids_chunk, "int64"),
                name + "q": (np.array(settings, dtype=np.int64), "int64"),
                name + "ws": scale,
            }
            for n, value in enumerate(arrays):
                out[f"{name}{n}"] = value
            return out

        return feed

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
        for i, layer in enumerate(layers):
            if "ple" in layer:
                layer.ple.ple_embedding.ngram_embedding.put(inputs, i)
        before = {i: c.offset for i, c in enumerate(cache) if _is_qsa(c)}
        h = lm.model.embed_tokens(tokens)
        h = mx.tile(h, (1, 1, lm.model.args.hc_count))
        h = _run_layers(lm, h, tokens, layers, cache)
        state: Arrays = {}
        for i, c in enumerate(cache):
            state.update(_cache_state(i, c, before.get(i)))
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
        capture: Capture | None = None,
        **kwargs: Any,
    ) -> Any:
        h = _run_layers(lm, boundary["h"], tokens, _layers(lm)[k:], cache[k:])
        host_ref = getattr(lm.model, "_omlx_mtp_prime_host", None)
        host = host_ref() if host_ref is not None else None
        if host is None or not prompt_priming.capture_eligible(host, cache):
            return None
        # Lightning MTP primes from the written residual of every chunk.
        return host, h

    def apply_state(self, cache: list, k: int, state: Arrays) -> None:
        for i in range(k):
            item = cache[i]
            if _is_qsa(item):
                keys = state[f"{i}.k"]
                if keys.shape[2]:
                    item.update_and_fetch(keys, state[f"{i}.v"])
                    item.update_indexer(state[f"{i}.ik"], state[f"{i}.ip"])
            else:
                for slot in range(len(item.cache)):
                    value = state.get(f"{i}.{slot}")
                    if value is not None:
                        item[slot] = value

    def finish_chunk(self, lm: Any, tokens: mx.array, output: Any, cache: list) -> None:
        if output is not None:
            host, h = output
            prompt_priming.maybe_capture(host, tokens, h, cache)

    def prefix_state(self, cache: list, k: int) -> Arrays:
        state: Arrays = {}
        for i in range(k):
            state.update(_cache_state(i, cache[i], 0))
        mx.eval(list(state.values()))
        return state

    def load_prefix(self, cache: list, state: Arrays) -> None:
        self.apply_state(cache, len(cache), state)
