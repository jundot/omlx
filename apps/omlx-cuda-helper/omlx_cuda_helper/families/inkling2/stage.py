# SPDX-License-Identifier: Apache-2.0
"""One GPU stage of Inkling 2: layers ``[first, last)``.

The boundary of every cut is the partial block sum ``h`` [1, L, hidden] and
``blk.<i>``, the partial sum that entered each block boundary layer ``i``
before the cut; a stage normalizes those into its completed blocks. A layer's
state is the K/V rows the chunk appended and its four conv windows.
"""

from __future__ import annotations

import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from ... import checkpoint
from ...tensors import from_host, load, peak_bytes, to_host, unpack
from ...wire import HostArray
from . import is_source, kernels, local_name, owner, text_config
from .layers import Cache, Config, Layer, source_layer, unit_rms

Arrays = dict[str, HostArray]


class Stage:
    def __init__(self, model_dir: str | Path, first: int, last: int):
        config = checkpoint.load_config(model_dir)
        self.cfg = Config.from_text(text_config(config))
        self.first, self.last = first, last
        info = mx.device_info()
        self.device = str(info.get("device_name") or info.get("architecture") or "")
        index = checkpoint.tensors(model_dir)
        source = is_source(index)
        names: dict[int | None, list[str]] = {}
        for name in index:
            layer = owner(name)
            if layer is not False:
                names.setdefault(layer, []).append(name)

        def load_layer(i: int) -> Layer:
            raw = {local_name(n): v for n, v in load(model_dir, names[i]).items()}
            if source:
                weights, path = source_layer(raw, i, self.cfg), ""
            else:
                prefix = f"layers.{i}."
                weights = {n[len(prefix) :]: v for n, v in raw.items()}
                path = f"model.layers.{i}."
            layer = Layer(weights, self.cfg, i, config, path)
            mx.eval(layer.parameters())
            # The packed tensors of unpacked linears are no longer needed.
            del raw, weights
            mx.clear_cache()
            return layer

        self._load_layer = load_layer
        self.layers = [load_layer(i) for i in range(first, last)]
        self.embedding = None
        if first == 0:
            raw = {local_name(n): v for n, v in load(model_dir, names[None]).items()}
            self.embedding = {
                part: raw[f"embed.{part}"]
                for part in ("weight", "scales", "biases")
                if f"embed.{part}" in raw
            }
            self.embedding_quant = checkpoint.quantization(config, "model.embed")
            self.embed_norm = raw["embed_norm.weight"]
            mx.eval(list(self.embedding.values()), self.embed_norm)
        self.dtype = self.layers[0].mlp_norm.dtype

    def _embed(self, tokens: mx.array) -> mx.array:
        parts = {name: part[tokens] for name, part in self.embedding.items()}
        rows = parts["weight"]
        if "scales" in parts:
            rows = unpack(
                rows, parts["scales"], parts.get("biases"), self.embedding_quant
            )
        return mx.fast.rms_norm(rows, self.embed_norm, self.cfg.eps)

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
        self.layers.append(self._load_layer(self.last))
        self.last += 1

    def new_job(self, layers: int) -> list[Cache]:
        return [Cache() for _ in range(layers)]

    def trim(self, job: list, layers: int) -> None:
        del job[layers:]

    def load_prefix(self, job: list, state: Arrays) -> None:
        for j, cache in enumerate(job):
            name = f"{self.first + j}."
            if name + "k" in state:
                cache.append(from_host(state[name + "k"]), from_host(state[name + "v"]))
                if self.layers[j].window:
                    cache.keep(self.layers[j].window)
            for slot in range(4):
                if f"{name}c{slot}" in state:
                    cache.conv[slot] = from_host(state[f"{name}c{slot}"])
        mx.eval([a for cache in job for a in cache.arrays()])

    def enable_kernels(self) -> None:
        """Take the CUDA kernels whose results match the plain-ops reference."""
        kernels.enable(self.cfg, self.dtype)

    def _forward(self, job, h, entered, keep):
        """Run the job's layers on ``h``. ``entered`` maps each boundary layer
        to the partial sum it took. Returns the partial sum, the chunk's state
        and, for each layer in ``keep``, the partial sum entering it from that
        chunk row on."""
        blocks = [unit_rms(entered[i]) for i in sorted(entered)]
        state: dict[str, mx.array] = {}
        kept: dict[int, mx.array] = {}
        for j, cache in enumerate(job):
            layer, index = self.layers[j], self.first + j
            if keep and index in keep:
                kept[index] = h[:, keep[index] :]
            if layer.boundary:
                entered[index] = h
            h = layer(h, blocks, cache)
            state[f"{index}.k"], state[f"{index}.v"] = cache.appended
            for slot, value in enumerate(cache.conv):
                state[f"{index}.c{slot}"] = value
            # Bound the lazy graph (and its temporaries) to one layer.
            mx.eval(h, blocks, *state.values(), *cache.arrays(), *kept.values())
        return h, state, kept

    def warm(self, tokens: int) -> float:
        self.enable_kernels()
        ids = mx.arange(tokens, dtype=mx.int32) % (self.cfg.vocab - 3) + 3
        if self.first:
            h = mx.random.normal((1, tokens, self.cfg.hidden)).astype(self.dtype)
            entered = {i: h for i in range(0, self.first, self.cfg.block)}
        else:
            h, entered = self._embed(ids)[None], {}
        mx.eval(h)
        seconds = 0.0
        # The first pass compiles kernels; the second one is the pace.
        for _ in range(2):
            began = time.perf_counter()
            self._forward(self.new_job(len(self.layers)), h, dict(entered), None)
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
        received: Arrays = {}
        if boundary is None:
            h = self._embed(mx.array(np.asarray(tokens, dtype=np.int32)))[None]
        else:
            h = from_host(boundary["h"])
            received = {n: v for n, v in boundary.items() if n.startswith("blk.")}
        entered = {int(n[4:]): from_host(v) for n, v in received.items()}
        keep = None
        if capture and start + h.shape[1] > capture["from"]:
            row = max(0, capture["from"] - start)
            keep = {i: row for i in capture["layers"]}
        h, state, kept = self._forward(job, h, entered, keep)
        out = {f"cap.{i}": to_host(rows) for i, rows in kept.items()}
        out["h"] = to_host(h)
        # The boundaries before this stage travel on as they came.
        out.update(received)
        for i, value in entered.items():
            if f"blk.{i}" not in received:
                out[f"blk.{i}"] = to_host(value)
        return out, {name: to_host(a) for name, a in state.items()}


def load_stage(model_dir: str | Path, first: int, last: int) -> Stage:
    return Stage(model_dir, first, last)
