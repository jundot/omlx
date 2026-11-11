# SPDX-License-Identifier: Apache-2.0
"""Layers [first, last) of GLM-5.3 on one GPU."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from ... import checkpoint, tensors
from ...wire import HostArray
from . import checks, kernels, root
from .model import Config, Layer

# Buffers MLX may keep cached after a chunk. Without a bound the cache grows
# past the memory the supervisor's plan leaves free on the GPU.
_CACHE_BYTES = 1 << 30
_CONV = re.compile(r"self_attn\.([qkv])_conv1d\.weight")
_EXPERT = re.compile(r"mlp\.experts\.(\d+)\.(\w+)\.(\w+)")
_FORGET_GATE = ("A_log", "dt_bias", "f_a_proj.", "f_b_proj.")


def _canonical(found: dict[str, mx.array], config: Config) -> dict[str, mx.array]:
    """Layer tensors under their MLX names, from either checkpoint layout."""
    out: dict[str, mx.array] = {}
    conv: dict[str, mx.array] = {}
    experts: dict[tuple[str, str], dict[int, mx.array]] = {}
    for name, value in found.items():
        name = name.replace("hc_attn_", "attn_hc.").replace("hc_ffn_", "ffn_hc.")
        if match := _CONV.fullmatch(name):
            conv[match.group(1)] = value
        elif match := _EXPERT.fullmatch(name):
            key = (match.group(2), match.group(3))
            experts.setdefault(key, {})[int(match.group(1))] = value
        else:
            attn, _, rest = name.partition(".")
            if attn == "self_attn" and rest.startswith(_FORGET_GATE):
                name = f"self_attn.forget_gate.{rest}"
            out[name] = value
    if conv:
        out["self_attn.conv1d.weight"] = mx.concatenate(
            [conv[part] for part in "qkv"], axis=0
        )
    for (proj, leaf), parts in experts.items():
        out[f"mlp.switch_mlp.{proj}.{leaf}"] = mx.stack(
            [parts[index] for index in sorted(parts)]
        )
    weight = out.get("self_attn.conv1d.weight")
    if weight is not None and weight.shape[-1] != 1:
        # Hugging Face stores [channels, 1, taps].
        out["self_attn.conv1d.weight"] = weight.moveaxis(2, 1)
    joint = out.pop("self_attn.kv_b_proj.weight", None)
    if joint is not None:
        # One matrix holds each head's key rows, then its value rows.
        joint = joint.reshape(config.heads, config.q_dim + config.v_dim, -1)
        out["self_attn.embed_q.weight"] = joint[:, : config.q_dim].swapaxes(-1, -2)
        out["self_attn.unembed_out.weight"] = joint[:, config.q_dim :]
    return out


class _Embedding:
    def __init__(self, found: dict[str, mx.array], quant: dict[str, Any] | None):
        self.weight = found["weight"]
        self.scales, self.biases = found.get("scales"), found.get("biases")
        self.quant = quant

    def parameters(self) -> list[mx.array]:
        return [a for a in (self.weight, self.scales, self.biases) if a is not None]

    def __call__(self, ids: mx.array) -> mx.array:
        if self.scales is None:
            return self.weight[ids]
        biases = None if self.biases is None else self.biases[ids]
        return tensors.unpack(self.weight[ids], self.scales[ids], biases, self.quant)


class Stage:
    def __init__(
        self,
        config: Config,
        layers: list[Layer],
        embedding: _Embedding | None,
        first: int,
    ):
        self.config, self.layers, self.embedding = config, layers, embedding
        self.first, self.last = first, first + len(layers)
        self.device = mx.device_info().get("device_name", "GPU")
        # A later stage makes its own input for the warm run in this dtype.
        self.dtype = layers[0].attn_norm.weight.dtype
        self._checked: set[str] = set()

    def warm(self, tokens: int) -> float:
        self.check_kernels()
        boundary = None
        if self.first:
            shape = (1, tokens, self.config.hc_mult, self.config.hidden)
            h = mx.random.normal(shape).astype(self.dtype)
            boundary = {"h": tensors.to_host(h)}
        ids = np.arange(tokens, dtype=np.int32) % 1000
        seconds = 0.0
        # The first run compiles the kernels; the second is the speed.
        for _ in range(2):
            began = time.perf_counter()
            self.run(
                self.new_job(len(self.layers)),
                start=0,
                tokens=ids,
                boundary=boundary,
                inputs={},
                following=0,
                capture=None,
            )
            seconds = time.perf_counter() - began
        mx.clear_cache()
        return seconds

    def check_kernels(self) -> None:
        """Compare each kernel with its op chain, once per layer kind."""
        if kernels.active():
            for layer in self.layers:
                kinds = (type(layer.attn).__name__, type(layer.mlp).__name__)
                if not self._checked.issuperset(kinds):
                    checks.layer(layer)
                    self._checked.update(kinds)

    def memory_bytes(self) -> int:
        return mx.get_active_memory()

    def new_job(self, layers: int) -> list[Any]:
        return [layer.attn.new_state() for layer in self.layers[:layers]]

    def load_prefix(self, job: list[Any], state: dict[str, HostArray]) -> None:
        slots: dict[int, dict[str, mx.array]] = {}
        for name, value in state.items():
            layer, _, slot = name.partition(".")
            slots.setdefault(int(layer) - self.first, {})[slot] = tensors.from_host(
                value
            )
        for index, layer_slots in slots.items():
            self.layers[index].attn.load_state(job[index], layer_slots)
        mx.eval([list(layer_slots.values()) for layer_slots in slots.values()])

    def run(
        self,
        job: list[Any],
        *,
        start: int,
        tokens: np.ndarray,
        boundary: dict[str, HostArray] | None,
        inputs: dict[str, HostArray],
        following: int,
        capture: dict[str, Any] | None,
    ) -> tuple[dict[str, HostArray], dict[str, HostArray]]:
        if boundary is None:
            h = self.embedding(mx.array(tokens)[None])
            streams = (*h.shape[:2], self.config.hc_mult, h.shape[-1])
            h = mx.contiguous(mx.broadcast_to(h[:, :, None, :], streams))
        else:
            h = tensors.from_host(boundary["h"])
        length = h.shape[1]
        wanted: list[int] = []
        if capture and start + length > capture["from"]:
            wanted = capture["layers"]
        out: dict[str, mx.array] = {}
        state: dict[str, mx.array] = {}
        for index, (layer, layer_state) in enumerate(zip(self.layers, job), self.first):
            h = layer(h, layer_state)
            arrays = layer.attn.state_arrays(index, layer_state)
            if index in wanted:
                # A drafter reads a layer with its streams averaged.
                rows = h.mean(axis=2)[:, max(0, capture["from"] - start) :]
                out[f"cap.{index}"] = rows
            mx.eval(h, list(arrays.values()), list(out.values()))
            state.update(arrays)
        out["h"] = h
        return (
            {name: tensors.to_host(value) for name, value in out.items()},
            {name: tensors.to_host(value) for name, value in state.items()},
        )


def _quantization(
    config: dict[str, Any], layer: int, module: str
) -> dict[str, Any] | None:
    """Settings of one layer module, keyed by any checkpoint layout."""
    table = config.get("quantization") or config.get("quantization_config") or {}
    paths = [
        f"language_model.model.layers.{layer}.{module}",
        f"model.language_model.layers.{layer}.{module}",
        f"model.layers.{layer}.{module}",
    ]
    path = next((p for p in paths if isinstance(table.get(p), dict)), paths[0])
    return checkpoint.quantization(config, path)


def load_stage(model_dir: str | Path, first: int, last: int) -> Stage:
    mx.set_cache_limit(_CACHE_BYTES)
    raw = checkpoint.load_config(model_dir)
    config = Config.from_dict(raw.get("text_config") or raw)
    names = list(checkpoint.tensors(model_dir))
    prefix = next(r for r in map(root, names) if r is not None)

    def tensors_under(scope: str) -> dict[str, mx.array]:
        scope = prefix + scope
        found = [n for n in names if n.startswith(scope)]
        loaded = tensors.load(model_dir, found)
        return {n[len(scope) :]: value for n, value in loaded.items()}

    layers = []
    for index in range(first, last):

        def quant(module: str, index: int = index) -> dict[str, Any] | None:
            return _quantization(raw, index, module)

        layer = Layer(
            _canonical(tensors_under(f"layers.{index}."), config), quant, config
        )
        # Evaluated layer by layer so the packed sources free as they go.
        mx.eval(layer.parameters())
        layers.append(layer)
    embedding = None
    if first == 0:
        table = raw.get("quantization") or raw.get("quantization_config") or {}
        paths = [prefix + "embed_tokens", "model.embed_tokens"]
        path = next((p for p in paths if isinstance(table.get(p), dict)), paths[0])
        embedding = _Embedding(
            tensors_under("embed_tokens."), checkpoint.quantization(raw, path)
        )
        mx.eval(embedding.parameters())
    return Stage(config, layers, embedding, first)
