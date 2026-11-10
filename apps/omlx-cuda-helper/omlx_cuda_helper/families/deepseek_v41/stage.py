# SPDX-License-Identifier: Apache-2.0
"""One GPU stage of DeepSeek V4.1: layers ``[first, last)``.

A boundary holds ``h`` [1, L, streams, dim], ``pre`` [1, L, streams] and, after
an index source, ``idx`` [1, L, K]. A layer's state is its seven cache slots; a
KV-source layer sends only the compressed rows and index keys the chunk added.
"""

from __future__ import annotations

import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from ... import checkpoint
from ...tensors import from_host, load, to_host
from ...wire import HostArray
from . import ngram, quant, resident, stage_start_allowed
from .config import Config, load_config
from .layers import COMPRESSED, INDEX_KEYS, PARTIAL_GATE, PARTIAL_KV, Block

Arrays = dict[str, HostArray]
_SLOTS = 7
_OFFSET, _HISTORY = 0, 6
# Freed buffers MLX may keep while a chunk runs.
_POOL_BYTES = 1 << 30
# Slots that grow with the prompt on KV-source layers.
_GROWING = (COMPRESSED, INDEX_KEYS)


class _Job:
    """Caches of one prompt: seven slots per layer and the next position."""

    def __init__(self, layers: int):
        self.caches: list[list] = [[None] * _SLOTS for _ in range(layers)]
        self.position = 0


class Stage:
    def __init__(self, model_dir: str | Path, first: int, last: int):
        config = self.config = load_config(model_dir)
        if not 0 <= first < last <= config.n_layers:
            raise ValueError(f"Invalid stage [{first}, {last})")
        self.first, self.last = first, last
        info = mx.device_info()
        self.device = str(info.get("device_name") or info.get("architecture") or "")
        index = checkpoint.tensors(model_dir)
        self.embed = None
        if first == 0:
            self.embed = load(model_dir, ["language_model.embed.weight"])[
                "language_model.embed.weight"
            ]
            mx.eval(self.embed)
        self.blocks = []
        for layer in range(first, last):
            names = [
                name
                for name in index
                if resident(name) and checkpoint.layer_of(name) == layer
            ]
            block = Block(load(model_dir, names), layer, config)
            mx.eval(block.parameters())
            self.blocks.append(block)
        self.dtype = self.blocks[0].attn_norm.dtype
        self.hasher = None
        if first == 0 and config.engram_layer_ids:
            # Layer 0 keeps the n-gram history even when it has no Engram.
            self.hasher = ngram.NgramHash(config, ngram.compressed_token_map(model_dir))

    def memory_bytes(self) -> int:
        return mx.get_active_memory()

    def new_job(self, layers: int) -> _Job:
        return _Job(layers)

    def load_prefix(self, job: _Job, state: Arrays) -> None:
        for name, value in state.items():
            layer, slot = (int(part) for part in name.split("."))
            if layer - self.first < len(job.caches):
                job.caches[layer - self.first][slot] = from_host(value)
                if slot == _OFFSET:
                    job.position = int(value[0][0])

    def warm(self, tokens: int) -> float:
        """Seconds of one dummy chunk once the kernels are compiled."""
        config, rng = self.config, np.random.default_rng(0)
        ids = rng.integers(0, config.vocab_size, tokens).astype(np.int32)
        boundary = None
        if self.first:
            shape = (1, tokens, config.hc_mult, config.dim)
            h = mx.random.normal(shape, key=mx.random.key(0)).astype(self.dtype)
            pre = mx.broadcast_to(
                (mx.arange(config.hc_mult) == 0).astype(mx.float32), shape[:-1]
            )
            boundary = {"h": h, "pre": pre}
        seconds = 0.0
        # The first pass compiles; the second gives the speed of a chunk.
        for _ in range(2):
            began = time.perf_counter()
            job = self.new_job(self.last - self.first)
            self._forward(job, ids, boundary, None)
            mx.synchronize()
            seconds = time.perf_counter() - began
        mx.clear_cache()
        return seconds

    def run(
        self,
        job: _Job,
        *,
        start: int,
        tokens: np.ndarray,
        boundary: Arrays | None,
        inputs: Arrays,
        following: int,
        capture: dict | None,
    ) -> tuple[Arrays, Arrays]:
        """One chunk. This family has no captured layers."""
        before = [
            [0 if cache[slot] is None else cache[slot].shape[1] for slot in _GROWING]
            for cache in job.caches
        ]
        arrays = None
        if boundary is not None:
            arrays = {name: from_host(value) for name, value in boundary.items()}
        out = self._forward(job, tokens, arrays, inputs)
        state = {}
        for j, cache in enumerate(job.caches):
            layer = self.first + j
            for slot in range(_SLOTS):
                value = cache[slot]
                if slot in _GROWING and self.config.cache_ratio(layer):
                    value = value[:, before[j][_GROWING.index(slot)] :]
                state[f"{layer}.{slot}"] = to_host(value)
        return {name: to_host(value) for name, value in out.items()}, state

    def _forward(self, job, tokens, boundary, inputs):
        """Boundary arrays after one chunk; ``inputs`` is None while warming."""
        # Unbounded, the pool keeps several times the memory a chunk needs.
        limit = mx.set_cache_limit(_POOL_BYTES)
        try:
            return self._layers(job, tokens, boundary, inputs)
        finally:
            mx.set_cache_limit(limit)

    def _layers(self, job, tokens, boundary, inputs):
        config: Config = self.config
        length, start = len(tokens), job.position
        if boundary is None:
            h = self.embed[mx.array(tokens)][None]
            h = mx.repeat(h[..., None, :], config.hc_mult, -2)
            pre = mx.broadcast_to(
                (mx.arange(config.hc_mult) == 0).astype(mx.float32), h.shape[:-1]
            )
            shared = {}
        else:
            h, pre = boundary["h"], boundary["pre"]
            # The rest is shared state of the layers before: the top-k rows.
            shared = {n: v for n, v in boundary.items() if n not in ("h", "pre")}
        hashes = history = None
        engram_ids = list(config.engram_layer_ids)
        if self.hasher is not None:
            history = job.caches[0][_HISTORY]
            history = np.asarray(history) if history is not None else None
            hashes, history = self.hasher(
                tokens, history if history is not None and history.size else None
            )
        for j, cache in enumerate(job.caches):
            block = self.blocks[j]
            if block.engram is not None:
                ids = hashes[:, :, engram_ids.index(block.layer)]
                unique, inverse = ngram.dedupe(ids)
                if inputs is None:
                    width = ids.shape[-1] * config.engram_head_dim
                    rows = mx.zeros((*unique.shape[:-1], width), mx.bfloat16)
                else:
                    rows = ngram.host_rows(inputs, block.layer, unique)
                h = block.add_engram(
                    h, rows, None if inverse is None else mx.array(inverse)
                )
            h, pre = block(h, pre, cache, shared, start)
            cache[_OFFSET] = mx.array([start + length], mx.int32)
            if history is not None and j == 0:
                cache[_HISTORY] = mx.array(history, mx.int64)
            self._fill_empty(cache, h.dtype)
            mx.eval(h, pre, cache)
        job.position = start + length
        out = {"h": h, "pre": pre}
        if "idx" in shared:
            out["idx"] = shared["idx"]
        mx.eval(out)
        return out

    def _fill_empty(self, cache: list, dtype: mx.Dtype) -> None:
        """Empty arrays for the slots a layer does not use, as the host has."""
        config = self.config
        empty = {
            COMPRESSED: lambda: quant.pack(
                mx.zeros((1, 0, config.head_dim)), 4, 16, True
            ),
            INDEX_KEYS: lambda: quant.pack(mx.zeros((1, 0, config.index_head_dim)), 4),
            PARTIAL_KV: lambda: mx.zeros((1, 0, config.head_dim), dtype),
            PARTIAL_GATE: lambda: mx.zeros((1, 0, config.head_dim), dtype),
            _HISTORY: lambda: mx.zeros((1, 0), mx.int64),
        }
        for slot, make in empty.items():
            if cache[slot] is None:
                cache[slot] = make()


def load_stage(model_dir: str | Path, first: int, last: int) -> Stage:
    if not stage_start_allowed(load_config(model_dir), first, last):
        raise ValueError(
            f"A DeepSeek V4.1 stage cannot start at layer {first}: a layer of "
            "the stage reads compressed KV or token history from before"
        )
    return Stage(model_dir, first, last)
