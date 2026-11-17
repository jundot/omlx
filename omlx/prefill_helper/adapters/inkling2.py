# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 (inkling2): split at any layer.

The layers exchange a partial block sum and the completed attention-residual
blocks. The cut sends the partial sum ``h`` entering layer ``k`` and, as
``blk.<i>``, the partial sum entering each block boundary layer ``i`` below
``k``; each side normalizes those itself, as the boundary layer does. A
layer's state is the K/V rows the chunk appended and its four conv windows.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.utils import load_model

from ...patches.inkling2 import model as inkling2
from ...utils.model_loading import maybe_apply_pre_load_patches
from ..adapter import Arrays, Capture, HostArrays, StageAdapter

# Source names (model.llm.*) and the runtime names oQ writes (model.*).
_ROOTS = ("model.llm.", "model.")
_SHARED = ("embed.", "embed_norm.")
_LAYER = re.compile(r"^layers\.(\d+)\.")
# Source attention-residual queries: attn_queries[i] belongs to layer i + 1.
_QUERY = re.compile(r"^attn_res\.(attn|mlp)_queries\.(\d+)$")
_CONV_SLOTS = 4


def _layers(lm: Any) -> list:
    return lm.model.layers


def _local_name(tensor: str) -> str:
    for root in _ROOTS:
        if tensor.startswith(root):
            return tensor[len(root) :]
    return tensor


def _owner(tensor: str) -> int | None | bool:
    """Layer of a helper tensor, None for the embedding, False otherwise."""
    name = _local_name(tensor)
    if name.startswith(_SHARED):
        return None
    if match := _LAYER.match(name):
        return int(match.group(1))
    if match := _QUERY.match(name):
        return int(match.group(2)) + (match.group(1) == "attn")
    return False


def _unit_rms(x: mx.array) -> mx.array:
    return mx.fast.rms_norm(x.astype(mx.float32), None, 1e-6)


def _run_layers(
    h: mx.array,
    blocks: list,
    layers: list,
    cache: list,
    first: int = 0,
    capture: Capture | None = None,
) -> tuple[mx.array, dict[int, mx.array]]:
    """``layers`` (global ids from ``first``) on the partial sum ``h``.

    ``blocks`` gains a block at each boundary layer. Returns the partial sum
    and what entered each boundary layer.
    """
    entered = {}
    for i, (layer, c) in enumerate(zip(layers, cache), first):
        # The drafter reads the partial sum entering a layer.
        if capture is not None:
            capture.take(i, h)
        if layer.is_boundary:
            entered[i] = h
        h = layer(h, blocks, cache=c)
    return h, entered


def _cache_state(i: int, item: Any, since: int) -> Arrays:
    kv, conv = item[0], item[1]
    out = {
        f"{i}.k": kv.keys[..., since : kv.offset, :],
        f"{i}.v": kv.values[..., since : kv.offset, :],
    }
    for slot in range(_CONV_SLOTS):
        if conv[slot] is not None:
            out[f"{i}.c{slot}"] = conv[slot]
    return out


class Inkling2Adapter(StageAdapter):
    model_types = ("inkling2",)
    contract = 1
    captures_layers = True

    def serves(self, engine_type: str | None, model_path: str | Path | None) -> bool:
        # Text-only conversions run on the LLM engine, which has no helper.
        return engine_type == "vlm"

    def layer_count(self, lm: Any) -> int:
        return len(_layers(lm))

    def resident(self, tensor: str) -> bool:
        # The embedding and the layers; the head, the final mix and norm,
        # vision and the draft model stay on this Mac.
        return _owner(tensor) is not False

    def keep_first(self, lm: Any, k: int) -> None:
        lm.model.layers = _layers(lm)[:k]
        for name in ("lm_head", "vision_tower"):
            if name in lm:
                del lm[name]
        for name in ("final_input", "norm"):
            if name in lm.model:
                del lm.model[name]

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
        adapter, loaded = self, set()

        class First(inkling2.Model):
            def __init__(self, args: inkling2.ModelArgs):
                super().__init__(args)
                adapter.keep_first(self, k)

            def sanitize(self, weights: dict) -> dict:
                # A helper holds whole files, which also carry later layers.
                held = {}
                for name, value in weights.items():
                    layer = _owner(name)
                    if layer is None or (layer is not False and layer < k):
                        held[name] = value
                out = super().sanitize(held)
                loaded.update(out)
                return out

        lm, _ = load_model(
            Path(path),
            lazy=True,
            strict=False,
            get_model_classes=lambda config: (First, inkling2.ModelArgs),
        )
        missing = [n for n, _ in tree_flatten(lm.parameters()) if n not in loaded]
        if missing:
            raise ValueError(
                f"{len(missing)} parameters of the helper's layers are not in "
                f"the checkpoint, such as {missing[0]}"
            )
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
        before = [c[0].offset for c in cache]
        h = lm.model.embed_norm(lm.model.embed(tokens))
        h, entered = _run_layers(h, [], _layers(lm), cache, capture=capture)
        boundary = {"h": h, **{f"blk.{i}": v for i, v in entered.items()}}
        state: Arrays = {}
        for i, c in enumerate(cache):
            state.update(_cache_state(i, c, before[i]))
        mx.eval(
            list(boundary.values()),
            list(state.values()),
            list(capture.out.values()) if capture else [],
        )
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
        capture: Capture | None = None,
        **kwargs: Any,
    ) -> Any:
        layers = _layers(lm)
        below = sorted(int(n[4:]) for n in boundary if n.startswith("blk."))
        blocks = [_unit_rms(boundary[f"blk.{i}"]) for i in below]
        h, _ = _run_layers(boundary["h"], blocks, layers[k:], cache[k:], k, capture)
        end = len(layers)
        if capture is not None and end in capture.layers:
            # The last draft tap is the final-norm output, of the kept rows
            # only: the mix and the norm work row by row.
            rows = slice(capture.start, None)
            sources = [b[:, rows] for b in blocks]
            x = lm.model.final_input(sources, _unit_rms(h[:, rows]))[:, :, 0]
            capture.out[end] = lm.model.norm(x).astype(h.dtype)
        return None

    def apply_state(self, cache: list, k: int, state: Arrays) -> None:
        for i in range(k):
            kv, conv = cache[i][0], cache[i][1]
            keys = state[f"{i}.k"]
            if keys.shape[2]:
                kv.update_and_fetch(keys, state[f"{i}.v"])
            for slot in range(_CONV_SLOTS):
                value = state.get(f"{i}.c{slot}")
                if value is not None:
                    conv[slot] = value

    def prefix_state(self, cache: list, k: int) -> Arrays:
        state: Arrays = {}
        for i in range(k):
            state.update(_cache_state(i, cache[i], 0))
        mx.eval(list(state.values()))
        return state

    def load_prefix(self, cache: list, state: Arrays) -> None:
        self.apply_state(cache, len(cache), state)
