# SPDX-License-Identifier: Apache-2.0
"""One GPU stage of DeepSeek V4.1: layers ``[first, last)``.

A boundary holds ``h`` [1, L, streams, dim], ``pre`` [1, L, streams] and, after
an index source, ``idx`` [1, L, K]. A layer's state is its seven cache slots; a
KV-source layer sends only the compressed rows and index keys the chunk added.
A later stage rebuilds the compressed KV and index keys of the last KV source
before it from the state of the stages before, as the host does after a cut.
"""

from __future__ import annotations

import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from ... import checkpoint
from ...tensors import from_host, load, to_host
from ...wire import HostArray
from . import ngram, quant, resident
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
        # Compressed ids of the last tokens, for the Engram hashes.
        self.history: np.ndarray | None = None
        # Growing slots of the last KV source before the stage, by slot.
        self.upstream: dict[int, mx.array] = {}


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
            block.check_kernels()
            self.blocks.append(block)
        self.dtype = self.blocks[0].attn_norm.dtype
        self.hasher = None
        engram = any(first <= i < last for i in config.engram_layer_ids)
        if engram or (first == 0 and config.engram_layer_ids):
            # Layer 0 keeps the n-gram history even when it has no Engram.
            self.hasher = ngram.NgramHash(config, ngram.compressed_token_map(model_dir))
        # The KV source before the stage whose compressed rows the stage reads.
        self.source = None
        for layer in range(first, last):
            if layer in config.kv_source_layers or config.compress_ratios[layer]:
                if layer not in config.kv_source_layers:
                    self.source = config.kv_source(layer)
                break

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
                if layer == 0 and slot == _HISTORY:
                    job.history = np.asarray(value[0])

    def load_upstream(self, job: _Job, arrays: Arrays) -> None:
        """State of the layers before this stage: the job's prefix ("p:" names),
        then the state of each chunk ("s:" names) before the chunk runs."""
        for name, value in arrays.items():
            layer, slot = (int(part) for part in name[2:].split("."))
            if layer == self.source and slot in _GROWING:
                rows, old = from_host(value), job.upstream.get(slot)
                job.upstream[slot] = (
                    rows if old is None else mx.concatenate([old, rows], 1)
                )
            elif name[0] == "p" and layer == 0 and slot == _HISTORY:
                # The stage hashes each chunk itself from there on.
                job.history = np.asarray(value[0])

    def warm(self, tokens: int) -> float:
        """Seconds of one dummy chunk once the kernels are compiled."""
        config, rng = self.config, np.random.default_rng(0)
        ids = rng.integers(0, config.vocab_size, tokens).astype(np.int32)
        boundary, upstream = None, {}
        if self.first:
            shape = (1, tokens, config.hc_mult, config.dim)
            h = mx.random.normal(shape, key=mx.random.key(0)).astype(self.dtype)
            pre = mx.broadcast_to(
                (mx.arange(config.hc_mult) == 0).astype(mx.float32), shape[:-1]
            )
            boundary = {"h": h, "pre": pre}
        if self.source is not None:
            ratio = config.compress_ratios[self.source]
            rows = tokens // ratio
            kv = mx.random.normal((1, rows, config.head_dim), key=mx.random.key(1))
            upstream = {
                COMPRESSED: quant.pack(kv, 4, 16, True),
                INDEX_KEYS: quant.pack(mx.zeros((1, rows, config.index_head_dim)), 4),
            }
            # Each query's first visible rows, as the index of the stage before.
            top = mx.arange(min(config.index_topk, rows), dtype=mx.int32)
            visible = (mx.arange(tokens) + 1)[:, None] // ratio
            boundary["idx"] = mx.where(top < visible, top, -1)[None]
        seconds = 0.0
        # The first pass compiles; the second gives the speed of a chunk.
        for _ in range(2):
            began = time.perf_counter()
            job = self.new_job(self.last - self.first)
            job.upstream = dict(upstream)
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
        if self.source is not None:
            shared["kv"] = job.upstream[COMPRESSED]
            shared["index_k"] = job.upstream[INDEX_KEYS]
        hashes = None
        engram_ids = list(config.engram_layer_ids)
        if self.hasher is not None:
            history = job.history
            hashes, job.history = self.hasher(
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
            if self.hasher is not None and self.first + j == 0:
                cache[_HISTORY] = mx.array(job.history, mx.int64)
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
    return Stage(model_dir, first, last)
