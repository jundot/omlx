# SPDX-License-Identifier: Apache-2.0
"""One GPU stage of Qwen3.5: layers ``[first, last)``.

The boundary of every cut is the hidden state ``h`` [1, L, hidden], with the
host's rotary frequencies ``rope`` when the job runs an attention layer on the
helper. A linear layer's state is its conv window and recurrent state, sent
whole; an attention layer's state is the K/V rows the chunk appended.
"""

from __future__ import annotations

import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from ... import checkpoint
from ...tensors import from_host, load, peak_bytes, to_host, unpack
from ...wire import HostArray
from . import check_layout, kernels, linear_layers, resident
from . import text_config as read_text_config
from .layers import KVCache, Layer, Weights, rotary_frequencies

Arrays = dict[str, HostArray]


class Stage:
    def __init__(self, model_dir: str | Path, first: int, last: int):
        config = checkpoint.load_config(model_dir)
        text = read_text_config(config)
        self.first, self.last = first, last
        info = mx.device_info()
        self.device = str(info.get("device_name") or info.get("architecture") or "")
        self.hidden = text["hidden_size"]
        self.vocab = text["vocab_size"]

        index = checkpoint.tensors(model_dir)
        check_layout(index)
        root = "language_model.model."
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
        linear = linear_layers(text)

        def load_layer(i: int) -> Layer:
            wanted = [n for n in names if n.startswith(f"{root}layers.{i}.")]
            weights = Weights(load(model_dir, wanted), config, root)
            layer = Layer(weights.at(f"layers.{i}"), text, linear[i])
            mx.eval(layer.parameters())
            # The packed tensors of the unpacked linears are no longer needed.
            del weights
            mx.clear_cache()
            return layer

        self._load_layer = load_layer
        self.layers = [load_layer(i) for i in range(first, last)]
        self.embedding = None
        if first == 0:
            wanted = [n for n in names if checkpoint.layer_of(n) is None]
            weights = Weights(load(model_dir, wanted), config, root)
            self.embedding = [
                weights.arrays[f"{root}embed_tokens.{part}"]
                for part in ("weight", "scales", "biases")
                if f"{root}embed_tokens.{part}" in weights.arrays
            ]
            self.embedding_quant = checkpoint.quantization(
                config, f"{root}embed_tokens"
            )
            mx.eval(self.embedding)
        self.dtype = self.layers[0].input_norm.dtype
        self.frequencies = rotary_frequencies(text)
        mx.eval(self.frequencies)

    def _embed(self, tokens: mx.array) -> mx.array:
        weight, *quantized = (part[tokens] for part in self.embedding)
        if not quantized:
            return weight
        return unpack(weight, *quantized, self.embedding_quant)

    def memory_bytes(self) -> int:
        return mx.get_active_memory()

    def peak_bytes(self) -> int:
        return peak_bytes()

    def drop_layer(self) -> None:
        """Free the weights of the last layer held."""
        self.layers.pop()
        self.last -= 1
        mx.clear_cache()

    def add_layer(self) -> None:
        """Load the layer after the last one held, as the stage loaded it."""
        layer = self._load_layer(self.last)
        if kernels.available():
            layer.enable_kernels()
        self.layers.append(layer)
        self.last += 1

    def new_job(self, layers: int) -> list:
        return [layer.new_cache() for layer in self.layers[:layers]]

    def trim(self, job: list, layers: int) -> None:
        del job[layers:]

    def load_prefix(self, job: list, state: Arrays) -> None:
        for j, cache in enumerate(job):
            name = f"{self.first + j}."
            if isinstance(cache, KVCache):
                if name + "k" in state:
                    cache.append(
                        from_host(state[name + "k"]), from_host(state[name + "v"])
                    )
            else:
                for slot in range(len(cache)):
                    if f"{name}{slot}" in state:
                        cache[slot] = from_host(state[f"{name}{slot}"])
        held = [c.arrays() if isinstance(c, KVCache) else c for c in job]
        mx.eval([a for arrays in held for a in arrays if a is not None])

    def _forward(
        self,
        job: list,
        h: mx.array,
        frequencies: mx.array | None,
        keep: dict[int, int] | None = None,
    ) -> tuple[mx.array, dict[str, mx.array], dict[int, mx.array]]:
        """Run the job's layers on ``h``.

        Returns the hidden state, the chunk's state and, for each layer in
        ``keep``, its output from that chunk row on.
        """
        state: dict[str, mx.array] = {}
        kept: dict[int, mx.array] = {}
        for j, cache in enumerate(job):
            layer, index = self.layers[j], self.first + j
            if not layer.linear and frequencies is None:
                raise RuntimeError("No rotary frequencies for this chunk (rope)")
            h = layer(h, cache, frequencies)
            if layer.linear:
                state[f"{index}.0"], state[f"{index}.1"] = cache
            else:
                state[f"{index}.k"], state[f"{index}.v"] = cache.appended
            if keep and index in keep:
                kept[index] = h[:, keep[index] :]
            # Bound the lazy graph (and its temporaries) to one layer.
            mx.eval(h, *state.values(), *(() if layer.linear else cache.arrays()))
        return h, state, kept

    def enable_kernels(self) -> None:
        """Take the CUDA kernels whose results match the plain-ops reference."""
        if kernels.available():
            for layer in self.layers:
                layer.enable_kernels()

    def warm(self, tokens: int) -> float:
        self.enable_kernels()
        ids = mx.arange(tokens, dtype=mx.int32) % (self.vocab - 3) + 3
        if self.first:
            h = mx.random.normal((1, tokens, self.hidden)).astype(self.dtype)
        else:
            h = self._embed(ids)[None]
        mx.eval(h)
        seconds = 0.0
        # The first pass compiles kernels; the second one is the pace.
        for _ in range(2):
            began = time.perf_counter()
            self._forward(self.new_job(len(self.layers)), h, self.frequencies)
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
        # The host's frequencies reach the first stage as an input and every
        # later stage with the hidden state.
        if boundary is None:
            h = self._embed(mx.array(np.asarray(tokens, dtype=np.int32)))[None]
            rope = inputs.get("rope")
        else:
            h = from_host(boundary["h"])
            rope = boundary.get("rope")
        frequencies = None if rope is None else from_host(rope)
        if rope is not None and frequencies.shape != self.frequencies.shape:
            raise ValueError("The rotary frequencies do not fit this model (rope)")
        keep = None
        if capture and start + h.shape[1] > capture["from"]:
            row = max(0, capture["from"] - start)
            keep = {i: row for i in capture["layers"]}
        h, state, kept = self._forward(job, h, frequencies, keep)
        out = {f"cap.{i}": to_host(rows) for i, rows in kept.items()}
        out["h"] = to_host(h)
        if rope is not None:
            out["rope"] = rope
        return out, {name: to_host(a) for name, a in state.items()}


def load_stage(model_dir: str | Path, first: int, last: int) -> Stage:
    return Stage(model_dir, first, last)
