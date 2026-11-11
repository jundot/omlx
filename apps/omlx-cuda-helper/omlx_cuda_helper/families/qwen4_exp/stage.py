# SPDX-License-Identifier: Apache-2.0
"""One GPU stage of Qwen3.8 Flash-Next: layers ``[first, last)``.

The boundary of every cut is the residual streams with all writes applied,
``h`` [1, L, streams * hidden]. A linear layer's state is its conv window and
recurrent state (the PLE layer adds its conv window and token history), sent
whole; a QSA layer's state is the rows the chunk appended.
"""

from __future__ import annotations

import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from ... import checkpoint
from ...tensors import from_host, load, to_host, unpack
from ...wire import HostArray
from . import (
    FIRST_STAGE_MIN,
    check_first_stage,
    kernels,
    layer_kinds,
    ple_layers,
    resident,
)
from . import text_config as read_text_config
from .attention import QSACache
from .layers import Layer, ple_rows
from .ops import Weights

Arrays = dict[str, HostArray]
_QSA_SLOTS = ("k", "v", "ik", "ip")


class Stage:
    def __init__(self, model_dir: str | Path, first: int, last: int):
        config = checkpoint.load_config(model_dir)
        text = read_text_config(config)
        check_first_stage(text)
        if 0 < first < FIRST_STAGE_MIN:
            raise ValueError(
                f"A later stage starts at layer {FIRST_STAGE_MIN} or above"
            )
        self.first, self.last = first, last
        info = mx.device_info()
        self.device = str(info.get("device_name") or info.get("architecture") or "")
        self.streams = text.get("hc_count", 4)
        self.hidden = text["hidden_size"]
        self.vocab = text["vocab_size"]

        index = checkpoint.tensors(model_dir)
        embedding = next(n for n in index if n.endswith("embed_tokens.weight"))
        root = embedding.removesuffix("embed_tokens.weight")
        names = [
            name
            for name in index
            if resident(name)
            and (
                first <= layer < last
                if (layer := checkpoint.layer_of(name)) is not None
                else first == 0
            )
        ]
        kinds, ple = layer_kinds(text), ple_layers(text)
        self.layers = []
        for i in range(first, last):
            wanted = [n for n in names if n.startswith(f"{root}layers.{i}.")]
            weights = Weights(load(model_dir, wanted), config, root)
            layer = Layer(weights.at(f"layers.{i}"), text, kinds[i], i in ple)
            mx.eval(layer.parameters())
            # The packed tensors of the unpacked linears are no longer needed.
            del weights
            mx.clear_cache()
            self.layers.append(layer)
        self.embedding = None
        if first == 0:
            wanted = [n for n in names if checkpoint.layer_of(n) is None]
            weights = Weights(load(model_dir, wanted), config, root)
            self.embedding = [
                weights[f"embed_tokens.{part}"]
                for part in ("weight", "scales", "biases")
                if f"embed_tokens.{part}" in weights
            ]
            self.embedding_quant = weights.quantization("embed_tokens")
            mx.eval(self.embedding)
            self.dtype = self._embed(mx.array([0])).dtype
        else:
            self.dtype = self.layers[0].attn_hc.down.weight.dtype

    def _embed(self, tokens: mx.array) -> mx.array:
        weight, *quantized = (part[tokens] for part in self.embedding)
        if not quantized:
            return weight
        return unpack(weight, *quantized, self.embedding_quant)

    def memory_bytes(self) -> int:
        return mx.get_active_memory()

    def new_job(self, layers: int) -> list:
        return [layer.new_cache() for layer in self.layers[:layers]]

    def load_prefix(self, job: list, state: Arrays) -> None:
        for j, cache in enumerate(job):
            name = f"{self.first + j}."
            if isinstance(cache, QSACache):
                if name + "k" in state:
                    cache.append(*(from_host(state[name + s]) for s in _QSA_SLOTS))
            else:
                for slot in range(len(cache)):
                    if f"{name}{slot}" in state:
                        cache[slot] = from_host(state[f"{name}{slot}"])
        held = [c.arrays() if isinstance(c, QSACache) else c for c in job]
        mx.eval([a for arrays in held for a in arrays if a is not None])

    def _forward(
        self,
        job: list,
        tokens: mx.array,
        h: mx.array | None,
        inputs: Arrays | None,
    ) -> tuple[mx.array, dict[str, mx.array]]:
        """Run the job's layers; ``inputs`` is None only for a warm-up chunk."""
        if h is None:
            h = mx.tile(self._embed(tokens)[None], (1, 1, self.streams))
        state: dict[str, mx.array] = {}
        for j, cache in enumerate(job):
            layer, index = self.layers[j], self.first + j
            rows = None
            if layer.ple is not None and inputs is not None:
                rows = ple_rows(inputs, f"p{index}.", tokens.shape[0])
            h = layer(h, tokens, rows, cache)
            if isinstance(cache, QSACache):
                state.update(
                    {f"{index}.{s}": a for s, a in zip(_QSA_SLOTS, cache.appended)}
                )
            else:
                state.update(
                    {f"{index}.{s}": a for s, a in enumerate(cache) if a is not None}
                )
            # Bound the lazy graph (and its temporaries) to one layer.
            mx.eval(h, *state.values(), *(cache.arrays() if layer.qsa else ()))
        return h, state

    def enable_kernels(self) -> None:
        """Take the CUDA kernels whose results match the plain-ops reference."""
        if kernels.available():
            for layer in self.layers:
                layer.enable_kernels()

    def warm(self, tokens: int) -> float:
        self.enable_kernels()
        ids = mx.arange(tokens, dtype=mx.int32) % (self.vocab - 3) + 3
        h = None
        if self.first:
            shape = (1, tokens, self.streams * self.hidden)
            h = mx.random.normal(shape).astype(self.dtype)
            mx.eval(h)
        seconds = 0.0
        # The first pass compiles kernels; the second one is the pace.
        for _ in range(2):
            began = time.perf_counter()
            self._forward(self.new_job(len(self.layers)), ids, h, None)
            mx.synchronize()
            seconds = time.perf_counter() - began
        mx.clear_cache()
        return seconds

    def run(
        self,
        job: list,
        *,
        start: int,
        tokens: np.ndarray,
        boundary: Arrays | None,
        inputs: Arrays,
        following: int,
        capture: dict | None,
    ) -> tuple[Arrays, Arrays]:
        # The host adapter of this family captures no layer outputs.
        h = None if boundary is None else from_host(boundary["h"])
        ids = mx.array(np.asarray(tokens, dtype=np.int32))
        h, state = self._forward(job, ids, h, inputs)
        return {"h": to_host(h)}, {name: to_host(a) for name, a in state.items()}


def load_stage(model_dir: str | Path, first: int, last: int) -> Stage:
    return Stage(model_dir, first, last)
