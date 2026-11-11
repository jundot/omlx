# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 / 3.6 / 3.8 dense (qwen3_5): split at any layer.

Only the hidden state [B, L, hidden] crosses the cut. A Gated DeltaNet layer
sends its conv window and recurrent state whole each chunk; a full-attention
layer sends the K/V rows the chunk appended. The model-level forward reads its
masks and positions from fixed layer indices, so each side builds them from
the first cache of each kind among its own layers. A helper that holds an
attention layer gets this Mac's rotary frequencies with every chunk and hands
them on with the hidden state.
"""

from __future__ import annotations

import json
import logging
import struct
import sys
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx_vlm.utils as vlm_utils
import numpy as np
from mlx.utils import tree_flatten
from mlx_vlm.models.cache import ArraysCache

from ...patches import qwen38_modelopt_mixed
from ...patches.mlx_lm_mtp import prompt_priming
from ...patches.qwen35_fa256_attention import apply_qwen35_fa256_attention_patch
from ...patches.qwen35_gdn_chunked import apply_qwen35_gdn_prefill_patch
from ...patches.qwen35_gdn_prework import apply_qwen35_gdn_prework_patch
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

logger = logging.getLogger(__name__)


def _layers(lm: Any) -> list:
    return lm.model.layers


def _language(lm: Any) -> Any:
    """The module of the model classes; patches rebind its functions."""
    return sys.modules[type(lm.model).__module__]


def _yarn(lm: Any) -> bool:
    rope = getattr(lm.args, "rope_parameters", None) or {}
    return (rope.get("type") or rope.get("rope_type")) == "yarn"


def _custom_loader(path: str | Path | None) -> bool:
    """Whether the server loads this checkpoint with a loader of its own
    (ParoQuant, ModelOpt mixed) instead of mlx-vlm's."""
    try:
        config = json.loads((Path(path).expanduser() / "config.json").read_text())
    except (OSError, TypeError, ValueError):
        return False
    quant = config.get("quantization_config")
    method = str(quant.get("quant_method") or "") if isinstance(quant, dict) else ""
    return method.lower() == "paroquant" or (
        method.lower() == "compressed-tensors"
        and qwen38_modelopt_mixed.is_supported_config(config)
    )


def _layout(tensor: str) -> str:
    """A tensor name without the differences of the MLX and Hugging Face
    layouts."""
    name = tensor.removeprefix("model.")
    return name.replace("language_model.model.", "language_model.", 1)


def _missing(lm: Any, path: str | Path, resident: Callable[[str], bool]) -> list[str]:
    """Parameters the helper computes with that the weight files do not name.

    A non-strict load leaves those at their random initial values. A helper
    Mac holds only the files of its layers, so the files present are read.
    """
    stored: set[str] = set()
    for file in Path(path).expanduser().glob("*.safetensors"):
        with open(file, "rb") as f:
            (length,) = struct.unpack("<Q", f.read(8))
            stored.update(_layout(name) for name in json.loads(f.read(length)))
    names = ("language_model." + name for name, _ in tree_flatten(lm.parameters()))
    return [n for n in names if resident(n) and _layout(n) not in stored]


def _run_layers(
    lm: Any,
    h: mx.array,
    layers: list,
    cache: list,
    first: int = 0,
    capture: Capture | None = None,
    frequencies: mx.array | None = None,
) -> mx.array:
    """``layers`` (global ids from ``first``) on the hidden state, with the
    masks and positions of the model-level forward. ``frequencies`` replace
    the rotary frequencies of the attention layers."""
    language = _language(lm)
    pairs = list(zip(layers, cache))
    if frequencies is not None:
        for layer in layers:
            if not layer.is_linear:
                layer.self_attn.rotary_emb._inv_freq = frequencies
    attention = next(((layer, c) for layer, c in pairs if not layer.is_linear), None)
    linear = next((c for layer, c in pairs if layer.is_linear), None)
    fa_mask = ssm_mask = positions = embeddings = None
    if linear is not None:
        ssm_mask = language._create_qwen3_5_ssm_mask(h, linear)
    if attention is not None:
        layer, item = attention
        fa_mask = language._create_qwen3_5_attention_mask(h, item)
        # Text-only prefill: positions continue from the cache offset.
        positions = mx.arange(item.offset, item.offset + h.shape[1], dtype=mx.int32)
        positions = mx.broadcast_to(positions[None, None], (3, *h.shape[:2]))
        rotary = layer.self_attn.rotary_emb
        if not rotary.fused_apply:
            embeddings = rotary(h, positions)
    for i, (layer, c) in enumerate(pairs, first):
        h = layer(
            h,
            mask=ssm_mask if layer.is_linear else fa_mask,
            cache=c,
            position_ids=positions,
            position_embeddings=embeddings,
        )
        if capture is not None:
            capture.take(i, h)
    return h


def _cache_state(i: int, item: Any, since: int) -> Arrays:
    """A layer's state after a chunk: K/V rows from ``since``, or both slots
    of a Gated DeltaNet layer."""
    if isinstance(item, ArraysCache):
        return {f"{i}.{s}": v for s, v in enumerate(item.cache) if v is not None}
    return {
        f"{i}.k": item.keys[..., since : item.offset, :],
        f"{i}.v": item.values[..., since : item.offset, :],
    }


class Qwen35Adapter(StageAdapter):
    model_types = ("qwen3_5",)
    contract = 1
    captures_layers = True

    def serves(self, engine_type: str | None, model_path: str | Path | None) -> bool:
        # Text-only conversions run on engines without a helper, and
        # load_first does not follow the server's own loaders.
        return engine_type == "vlm" and not _custom_loader(model_path)

    def layer_count(self, lm: Any) -> int:
        return len(_layers(lm))

    def resident(self, tensor: str) -> bool:
        # The embedding and the layers; the head, the final norm, the draft
        # head and vision stay on this Mac.
        return (
            language_tensor(tensor)
            and ".mtp." not in tensor
            and (".layers." in tensor or ".embed_tokens." in tensor)
        )

    def keep_first(self, lm: Any, k: int) -> None:
        lm.model.layers = _layers(lm)[:k]
        for name in ("lm_head", "mtp"):
            if name in lm:
                del lm[name]

    @contextmanager
    def use_first(self, lm: Any, k: int):
        layers = _layers(lm)
        lm.model.layers = layers[:k]
        try:
            yield
        finally:
            lm.model.layers = layers

    def load_first(self, path: str, k: int, a8_min_tokens: int | None = None) -> Any:
        if _custom_loader(path):
            raise ValueError(
                "this checkpoint needs the server's own loader (ParoQuant or "
                "ModelOpt mixed), which a prefill helper does not have"
            )
        maybe_apply_pre_load_patches(path, None, for_vlm=True)
        load_config = vlm_utils.load_config

        def expanded(model_path: Any, **kwargs: Any) -> dict:
            # The server's loader adds the same per-module quantization keys.
            return expand_per_layer_quant_keys(load_config(model_path, **kwargs))

        vlm_utils.load_config = expanded
        try:
            # Modules without weight files here are dropped before anything
            # is evaluated.
            model = vlm_utils.load_model(Path(path), lazy=True, strict=False)
        finally:
            vlm_utils.load_config = load_config
        lm = model.language_model
        self.keep_first(lm, k)
        missing = _missing(lm, path, self.resident)
        if missing:
            raise ValueError(
                f"{len(missing)} parameters of the helper's layers are not in "
                f"the checkpoint, such as {missing[0]}"
            )
        # The server's engine start installs these prefill kernels.
        apply_sdpa256_attention_patch()
        apply_qwen35_fa256_attention_patch()
        apply_qwen35_gdn_prework_patch()
        apply_qwen35_gdn_prefill_patch()
        apply_qwen35_q4_mlp_patch()
        apply_qwen35_q4_prefill_linear_patch(model)
        if a8_min_tokens is not None:
            apply_qwen35_oq_a8_patch(model, min_tokens=a8_min_tokens)
        mx.eval(lm.parameters())
        return lm

    def job_feed(
        self, lm: Any, cache: list | None, tokens: np.ndarray, grid: list[int], k: int
    ) -> Callable[[int], HostArrays] | None:
        attention = next((x for x in _layers(lm)[:k] if not x.is_linear), None)
        if attention is None:
            return None
        if _yarn(lm):
            # A context window past the native one changes the rotary
            # frequencies and scaling of this Mac's model only.
            raise RuntimeError("the helper's attention layers do not follow YaRN")
        # The rotary frequencies as this Mac's GPU rounded them. Another
        # backend's power function differs in the last bits, which turns
        # late positions by a visibly different angle.
        frequencies = np.asarray(attention.self_attn.rotary_emb.inv_freq)
        return lambda index: {"rope": (frequencies, "float32")}

    def run_first(
        self,
        lm: Any,
        cache: list,
        tokens: mx.array,
        next_tokens: int,
        inputs: HostArrays | None = None,
        capture: Capture | None = None,
    ) -> tuple[Arrays, Arrays]:
        before = [getattr(c, "offset", 0) for c in cache]
        # The host's frequencies rotate this Mac's attention layers too: its
        # GPU may round its own another way.
        rope = inputs.get("rope") if inputs else None
        frequencies = None if rope is None else wire.from_host(*rope)
        h = _run_layers(
            lm,
            lm.model.embed_tokens(tokens),
            _layers(lm),
            cache,
            capture=capture,
            frequencies=frequencies,
        )
        state: Arrays = {}
        for i, c in enumerate(cache):
            state.update(_cache_state(i, c, before[i]))
        boundary = {"h": h}
        if frequencies is not None:
            # They travel on with the hidden state for a helper's next stage.
            boundary["rope"] = frequencies
        mx.eval(h, list(state.values()), list(capture.out.values()) if capture else [])
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
        h = _run_layers(lm, boundary["h"], _layers(lm)[k:], cache[k:], k, capture)
        if not prompt_priming.capture_eligible(lm, cache):
            return None
        # Lightning MTP primes its head from the normed hidden of every chunk.
        return lm.model.norm(h)

    def apply_state(self, cache: list, k: int, state: Arrays) -> None:
        for i in range(k):
            item = cache[i]
            if isinstance(item, ArraysCache):
                for slot in range(len(item.cache)):
                    value = state.get(f"{i}.{slot}")
                    if value is not None:
                        item[slot] = value
            elif state[f"{i}.k"].shape[2]:
                item.update_and_fetch(state[f"{i}.k"], state[f"{i}.v"])

    def finish_chunk(self, lm: Any, tokens: mx.array, output: Any, cache: list) -> None:
        if output is None:
            return
        try:
            prompt_priming.maybe_capture(lm, tokens, output, cache)
        except Exception:
            # As in a local prefill: the head then starts unprimed.
            logger.debug("MTP prompt-priming capture failed", exc_info=True)

    def prefix_state(self, cache: list, k: int) -> Arrays:
        state: Arrays = {}
        for i in range(k):
            state.update(_cache_state(i, cache[i], 0))
        mx.eval(list(state.values()))
        return state

    def load_prefix(self, cache: list, state: Arrays) -> None:
        self.apply_state(cache, len(cache), state)
