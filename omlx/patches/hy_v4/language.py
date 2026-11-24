# SPDX-License-Identifier: Apache-2.0
"""Hy4 / Hy4-flash text decoder for oMLX's mlx-vlm runtime."""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import CacheList, KVCache, RotatingKVCache
from mlx_vlm.models.base import LanguageModelOutput, create_attention_mask

from ..deepseek_v4.switch_layers import QuantizedSwitchLinear, SwitchGLU
from ..deepseek_v41.mtp import AcceptanceDepthController
from ..mlx_lm_mtp.cache_rollback import _is_decode_consistent_armed, _is_undo_armed
from . import exact, verify_dispatch
from . import kernels as hc_kernels
from .config import TextConfig
from .flash import Engram
from .layers import (
    MLP,
    Attention,
    HyperConnection,
    HyperHead,
    MoE,
    hc_expand,
)
from .rope import Positions
from .verify_layers import LayerStack, layer_output


class DecoderLayer(nn.Module):
    def __init__(self, args: TextConfig, layer: int):
        super().__init__()
        self.self_attn = Attention(args, layer)
        if args.mlp_layer_types[layer] == "sparse":
            self.mlp = MoE(args)
        else:
            self.mlp = MLP(args, args.intermediate_size, post_norm=args.is_flash)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.hc_attn_layer = HyperConnection(args)
        self.hc_mlp_layer = HyperConnection(args)
        self._verify_moe_graph = None
        self._verify_tail_graph = None
        if args.engram_layer_ids and layer + 1 in args.engram_layer_ids:
            self.engram_layer = Engram(args, layer + 1)

    def _can_compile_moe(self, x):
        mlp = self.mlp
        if not (
            x.ndim == 3
            and x.shape[0] == 1
            and x.dtype == mx.bfloat16
            and type(mlp) is MoE
            and not mlp.training
            and verify_dispatch.eligible(x.shape[1])
        ):
            return False
        switch = mlp.switch_mlp
        return (
            type(switch) is SwitchGLU
            and "gate_up_proj" not in switch
            and all(
                type(proj) is QuantizedSwitchLinear
                and proj.mode == "affine"
                and proj.bits == 4
                and proj.group_size == 64
                for proj in switch.projections()
            )
        )

    def _mlp_forward(self, x):
        mlp = self.mlp
        if self._can_compile_moe(x):
            graph = self._verify_moe_graph
            if graph is None or graph[0] is not mlp:
                graph = (mlp, mx.compile(mlp.__call__, inputs=mlp.state))
                # Capture the MLP, leaving ownership with the decoder layer.
                object.__setattr__(self, "_verify_moe_graph", graph)
            return graph[1](x)
        return mlp(x)

    def _verify_tail(self, values, gate, streams, post):
        attention = self.self_attn
        pre = self.hc_mlp_layer.hc_pre
        norm = self.post_attention_layernorm
        mlp = self.mlp
        modules = (attention.o_proj, pre, norm, mlp)
        if "post_proj_norm" in attention:
            modules += (attention.post_proj_norm,)
        owners = (attention, *modules)
        graph = self._verify_tail_graph
        if (
            graph is None
            or len(graph[0]) != len(owners)
            or any(current is not saved for current, saved in zip(owners, graph[0]))
        ):

            def tail(values, gate, streams, post, transposed):
                return layer_output(
                    attention, pre, norm, mlp, values, gate, streams, post, transposed
                )

            graph = (owners, mx.compile(tail, inputs=[m.state for m in modules]))
            object.__setattr__(self, "_verify_tail_graph", graph)
        transposed = pre._fused.get(pre.hc_fn)
        return graph[1](values, gate, streams, post, transposed)

    def __call__(
        self,
        streams,
        mask=None,
        cache=None,
        prev_topk=None,
        engram_inputs=None,
        positions=None,
    ):
        if "engram_layer" in self and engram_inputs is not None:
            streams = streams + self.engram_layer(streams, *engram_inputs)
        x, post = self.hc_attn_layer(streams)
        x = self.input_layernorm(x)
        if (
            cache is not None
            and x.shape[1] > 1
            and self._can_compile_moe(x)
            and hc_kernels.available()
        ):
            (values, gate), topk = self.self_attn(
                x, mask, cache, prev_topk, positions, return_values=True
            )
            return self._verify_tail(values, gate, streams, post), topk

        x, topk = self.self_attn(x, mask, cache, prev_topk, positions)
        streams = hc_expand(x, streams, post)
        x, post = self.hc_mlp_layer(streams)
        x = self._mlp_forward(self.post_attention_layernorm(x))
        return hc_expand(x, streams, post), topk


class TextModel(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.hc_head = HyperHead(args)
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self._verify_stack = None
        kinds = args.attention_kinds
        self._first = {k: kinds.index(k) for k in ("sparse", "sliding") if k in kinds}

    def _masks(self, h, cache):
        masks = {}
        batched = any(
            c is not None and hasattr(_latent_cache(c), "left_padding") for c in cache
        )
        if h.shape[1] == 1 and not batched:
            return masks
        for kind, layer in self._first.items():
            c = _latent_cache(cache[layer])
            window = self.args.sliding_window if kind == "sliding" else None
            masks[kind] = create_attention_mask(
                h, c, window_size=window, return_array=True
            )
        return masks

    def __call__(
        self,
        inputs,
        cache=None,
        inputs_embeds=None,
        engram_inputs=None,
        positions=None,
    ):
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        if cache is None:
            cache = [None] * len(self.layers)
        masks = self._masks(h, cache)
        b, length, d = h.shape
        streams = mx.contiguous(
            mx.broadcast_to(h[:, :, None, :], (b, length, self.args.hc_mult, d))
        )
        if (
            b == 1
            and h.dtype == mx.bfloat16
            and not self.training
            and verify_dispatch.eligible(length)
            and hc_kernels.available()
            and _is_undo_armed()
            and not _is_decode_consistent_armed()
            and len(cache) == len(self.layers)
            and all(c is not None for c in cache)
        ):
            stack = self._verify_stack
            if stack is None or not stack.matches(self.layers):
                stack = LayerStack(self.layers, h)
                object.__setattr__(self, "_verify_stack", stack)
            if any(stack.ready):
                streams = stack(streams, cache, masks, engram_inputs, positions)
                return self.norm(self.hc_head(streams))
        prev_topk = None
        kinds = self.args.attention_kinds
        for layer, c, kind in zip(self.layers, cache, kinds):
            streams, topk = layer(
                streams, masks.get(kind), c, prev_topk, engram_inputs, positions
            )
            if kind == "sparse":
                prev_topk = topk
        return self.norm(self.hc_head(streams))


def _latent_cache(cache):
    if isinstance(cache, CacheList):
        return cache[0]
    return cache


def _slot_history(cache):
    """The slot-history cache entry (layer 0), if the model keeps one."""
    if cache is not None and isinstance(cache[0], CacheList):
        entries = cache[0].caches
        return entries[2] if len(entries) > 2 else None
    return None


def _history_planes(history) -> mx.array | None:
    """[3, B, slots] rotary position of every slot held by ``history``."""
    values = getattr(history, "values", None)
    if values is None or values.shape[-1] != 3:
        return None
    end = getattr(history, "_idx", None)
    end = history.offset if end is None else end
    return mx.moveaxis(values[:, 0, :end], -1, 0)


def _cache_extent(cache) -> tuple[int, int]:
    """(first, end) cache slots of a one-request latent KV cache.

    A batch cache that once held longer rows keeps left padding before
    ``first``; positions count from ``first``.
    """
    end = getattr(cache, "_idx", None)
    if end is None:
        return 0, int(cache.offset)
    left = getattr(cache, "left_padding", None)
    first = int(left.reshape(-1)[0].item()) if left is not None else 0
    return first, int(end)


def _history_values(history, planes, b: int, length: int) -> mx.array:
    """[b, 1, length, 3] rotary positions stored beside the token ids.

    Without explicit ``planes`` the call continues plain text from the cache's
    row offsets. A history restored from before positions were stored keeps its
    zero-width values.
    """
    if history.values is not None and history.values.shape[-1] == 0:
        return mx.zeros((b, 1, length, 0), dtype=mx.int32)
    if planes is None:
        start = mx.array(history.offset).reshape(-1, 1)
        planes = mx.broadcast_to((start + mx.arange(length))[None], (3, b, length))
    return planes.transpose(1, 2, 0)[:, None].astype(mx.int32)


def engram_inputs(
    ids: mx.array, args: TextConfig, media_token_ids, history=None, planes=None
) -> tuple[mx.array, mx.array]:
    """Engram n-grams [B, L, max_order] of ``ids`` and their keep mask.

    ``history`` is the slot-history cache of a decode; without it positions
    before the sequence read as padding. It keeps every slot's token id (keys)
    and its [3] rotary positions (values, from ``planes`` when given).
    """
    pad = args.engram_pad_id
    width = args.engram_max_ngram_size - 1
    b, length = ids.shape
    total = length
    parts = []
    if history is not None:
        tokens, _ = history.update_and_fetch(
            ids.astype(mx.int32)[:, None, :, None],
            _history_values(history, planes, b, length),
        )
        tokens = tokens[:, 0, :, 0]
        total = tokens.shape[1]
        lead = tokens[:, max(0, total - length - width) : total - length]
        if lead.shape[1]:
            parts.append(lead)
    missing = width - sum(p.shape[1] for p in parts)
    if missing:
        parts.insert(0, mx.full((b, missing), pad, dtype=mx.int32))
    parts.append(ids.astype(mx.int32))
    window = mx.concatenate(parts, axis=1)
    left = getattr(history, "left_padding", None)
    if left is not None:
        slots = mx.arange(total - length - width, total)[None]
        window = mx.where(slots < left[:, None], pad, window)
    ngrams = mx.stack(
        [window[:, width - j : width - j + length] for j in range(width + 1)],
        axis=-1,
    )
    image, video = media_token_ids
    keep = (ids != image) & (ids != video) & (ids != pad)
    return ngrams, keep


# Decode-exact verify gives greedy MTP output the bits of decoding without
# MTP, but its per-row work in the cache-reading parts costs Hy4-flash about
# 70 ms per six-row verify on M3 Ultra, more than drafting gains. It stays off
# until those parts get dedicated kernels.
EXACT_VERIFY = False


def set_dspark_verify_armed(flag: bool) -> None:
    """The MTP generator arms this around draft verify forwards."""
    exact.set_armed(flag and EXACT_VERIFY)
    verify_dispatch.set_armed(flag)


class LanguageModel(nn.Module):
    """Hy4 text model, also the embedded-drafter host for oMLX's MTP loop.

    With DFly2 armed the generator drives decoding through ``dspark_forward``
    (one draft block from the target's last committed row) and
    ``dspark_step_logits`` (one corrected draft slot after the other), and
    rolls the target caches back with ``mtp_partial_rollback``.
    """

    # Batched requests draft too. Each row verifies alone on the cache the
    # generator extracts for it, so the drafter reads that row's caches.
    _omlx_mtp_multi_request = True
    _omlx_mtp_independent_verify = True

    def __init__(self, config: TextConfig):
        super().__init__()
        self.args = config
        self.config = config
        self.model_type = config.model_type
        self.model = TextModel(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.media_token_ids = (120120, 120124)
        self._dfly_cache = None
        # Not an array attribute, which nn.Module would count as a parameter.
        self._position_delta = SimpleNamespace(value=None)
        self.configure_mtp(False)

    def configure_mtp(self, enabled: bool, depth: int = 5) -> None:
        """Arm DFly2 drafting (only when the drafter is attached)."""
        enabled = bool(enabled) and "dfly" in self
        self._omlx_mtp_decode_enabled = enabled
        self._omlx_dspark_decode_enabled = enabled
        self._omlx_mtp_chain = True
        self._omlx_mtp_head_clone = False
        self._omlx_mtp_rowwise_unsupported = True
        if "dfly" in self:
            # The generator caps its depth at dspark_block_size draft tokens.
            self.config.dspark_block_size = self.dfly.cfg.block_size - 1
            self._omlx_mtp_depth = max(1, min(depth, self.config.dspark_block_size))
            index_layers = set(self.dfly.cfg.index_layers)
            for i, layer in enumerate(self.layers):
                layer.self_attn._capture_topk = enabled and i in index_layers

    def get_mtp_module(self):
        return self.dfly if self._omlx_mtp_decode_enabled else None

    def make_mtp_cache(self):
        # DFly2 keeps no cache of its own.
        return []

    def make_mtp_depth_controller(self, depth):
        return AcceptanceDepthController(depth)

    def dspark_append_context(self, hidden_rows, cache, **_):
        """DFly2 reads the target's caches directly; there is nothing to append."""

    def dspark_prepare_block(self, cache, hidden, row, anchor, num_drafts, depth):
        """Prepare the inputs for a DFly2 block before the target cache rolls back."""
        width = num_drafts + 1
        if (
            cache is None
            or cache is not self._dfly_cache
            or hidden is None
            or hidden.shape[:2] != (1, width)
            or hidden.dtype != mx.bfloat16
            or not 2 <= width <= 6
            or not 1 <= depth <= 5
            or self.training
            or mx.default_device() != mx.gpu
            or not verify_dispatch._supported_gpu()
        ):
            return None

        extent = _latent_cache(cache[0])
        padded_end = getattr(extent, "_idx", None)
        left = getattr(extent, "left_padding", None)
        if padded_end is None:
            padded_end, left = extent.offset, None
        if not isinstance(padded_end, int) or padded_end - num_drafts <= 2048:
            return None
        # The accepted row leaves the cache at the same extent as a trim of
        # num_drafts - row. The retained keys themselves are unchanged.
        first = left.reshape(1) if left is not None else mx.zeros((1,), dtype=mx.int32)
        end = padded_end - num_drafts + row
        dfly = self.dfly
        histories = []
        for kv_layer, index_layer in zip(dfly.cfg.kv_layers, dfly.cfg.index_layers):
            kv = _latent_cache(cache[kv_layer])
            if kv.keys is None or kv.values is None or kv.keys.shape[2] < padded_end:
                return None
            topk = self.layers[index_layer].self_attn._captured_topk
            if isinstance(topk, list):
                if len(topk) != width or any(entry is None for entry in topk):
                    return None
                topk = mx.stack([entry[0, 0, 0] for entry in topk])
            elif topk is not None and topk.shape[:3] == (1, 1, width):
                topk = topk[0, 0]
            else:
                return None
            if topk.shape[-1] != 2048:
                return None
            index = mx.take(topk, row, axis=0).reshape(2048)
            valid = (index >= first) & (index < end)
            safe = mx.maximum(mx.minimum(index, end - 1), first)
            histories.append((kv.keys[:, :, safe], kv.values[:, :, safe], valid))

        start = end - first
        if self._position_delta.value is not None:
            start = start + self._position_delta.value
        return partial(
            dfly.block_hidden,
            self.model.embed_tokens,
            mx.take(hidden, row, axis=1),
            anchor,
            start,
            histories,
            depth + 1,
        )

    def dspark_forward(self, hidden_rows, anchor, cache=None, *, draft_length=None):
        """Final-normed hidden of one draft block.

        ``hidden_rows`` ends with the target's post-norm hidden of the row
        that produced ``anchor``; the target caches were already rolled back,
        so their length is the anchor's position.
        """
        target = self._dfly_cache
        if target is None:
            raise RuntimeError("DFly2 needs a target forward before drafting")
        dfly = self.dfly
        width = 1 + int(draft_length or dfly.cfg.block_size - 1)
        first, end = _cache_extent(_latent_cache(target[0]))
        row = hidden_rows.shape[1] - 1
        histories = []
        for kv_layer, index_layer in zip(dfly.cfg.kv_layers, dfly.cfg.index_layers):
            kv = _latent_cache(target[kv_layer])
            topk = self.layers[index_layer].self_attn._captured_topk
            pick = row
            if isinstance(topk, list):
                # An exact verify block keeps one selection per row.
                topk, pick = topk[row], 0
            if topk is None:
                # Shorter than the indexer's top-k: every committed slot.
                latent = kv.keys[..., first:end, :]
                histories.append((latent, kv.values[..., first:end, :], None))
                continue
            index = topk[0, 0, pick]
            valid = (index >= first) & (index < end)
            safe = mx.clip(index, first, end - 1)
            histories.append((kv.keys[:, :, safe], kv.values[:, :, safe], valid))
        start = end - first
        if self._position_delta.value is not None:
            # Image context: the anchor's position is its slot plus the delta.
            start = start + self._position_delta.value
        block = dfly.block_hidden(
            self.model.embed_tokens,
            hidden_rows[:, -1:],
            anchor.reshape(-1)[:1],
            start,
            histories,
            width,
        )
        return block, None

    def dspark_step_logits(self, block, index, previous):
        """Logits of draft slot ``index`` corrected with the previous token."""
        hidden = self.dfly.correct(
            self.model.embed_tokens, block[:, index + 1], previous
        )
        return self.project(hidden)

    def mtp_partial_rollback(self, cache, accepted: int, num_drafts: int) -> bool:
        """Drop the rejected drafts from every target cache.

        Returns False, touching nothing, when a cache cannot drop them (a
        rotated ring without the generator's undo log); the generator then
        falls back.
        """
        rejected = int(num_drafts) - int(accepted)
        if rejected <= 0:
            return True
        entries = []
        for c in cache:
            if c is not None:
                entries.extend(getattr(c, "caches", None) or (c,))
        if not all(e.is_trimmable() for e in entries):
            return False
        return all(e.trim(rejected) == rejected for e in entries)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for i, layer in enumerate(self.layers):
            attention = layer.self_attn
            if attention.kind == "sliding":
                caches.append(RotatingKVCache(max_size=attention.window, keep=0))
                continue
            entries = [KVCache()]
            if attention.index_module is not None:
                entries.append(KVCache())
            if i == 0 and self.config.engram_layer_ids:
                if len(entries) != 2:
                    raise ValueError("Engram history needs an indexed first layer")
                # Token ids for the Engram n-grams, trimmed with the KV caches.
                entries.append(KVCache())
            caches.append(CacheList(*entries))
        return caches

    def _engram_inputs(
        self, ids: mx.array, cache, positions=None
    ) -> tuple[mx.array, mx.array]:
        planes = None if positions is None else positions.planes
        return engram_inputs(
            ids, self.config, self.media_token_ids, _slot_history(cache), planes
        )

    def _positions(self, position_ids, inputs_embeds, length: int):
        """Rotary positions of this call, or None to follow the cache offsets.

        Merged image embeddings carry general [3, B, L] positions. Without
        them (decode, text chunks) the planes are consecutive text positions
        and only their per-row start matters.
        """
        if position_ids is None:
            return None
        planes = position_ids
        if planes.ndim == 2:
            planes = mx.broadcast_to(planes[None], (3,) + planes.shape)
        if planes.ndim != 3 or planes.shape[0] != 3 or planes.shape[-1] != length:
            raise ValueError(
                f"position_ids {tuple(position_ids.shape)} do not match "
                f"{length} input tokens"
            )
        return Positions(
            planes.astype(mx.int32),
            inputs_embeds is not None,
            self.config.mrope_section,
        )

    def _note_position_delta(self, positions, cache, length: int) -> None:
        """Remember position minus slot of the call's last token for the drafter."""
        delta = None
        if positions is not None and cache is not None:
            slot = mx.array(_latent_cache(cache[0]).offset).reshape(-1)[:1]
            delta = positions.planes[0, :1, -1] - (slot + length - 1)
        self._position_delta.value = delta

    def project(self, hidden: mx.array) -> mx.array:
        head = (
            self.model.embed_tokens.as_linear
            if self.config.tie_word_embeddings
            else self.lm_head
        )
        if self.config.enable_lm_head_fp32 and isinstance(head, nn.Linear):
            return exact.fp32_head(head, hidden)
        if self.config.enable_lm_head_fp32:
            hidden = hidden.astype(mx.float32)
        return exact.linear(head, hidden)

    def __call__(
        self,
        inputs: mx.array | None = None,
        cache: Any | None = None,
        inputs_embeds: mx.array | None = None,
        mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        **kwargs,
    ) -> LanguageModelOutput:
        if inputs is None:
            inputs = kwargs.get("input_ids")
        length = (inputs if inputs_embeds is None else inputs_embeds).shape[1]
        positions = self._positions(position_ids, inputs_embeds, length)
        if self._omlx_mtp_decode_enabled and cache is not None:
            # The drafter reads these caches after the generator's rollback.
            self._dfly_cache = cache
            self._note_position_delta(positions, cache, length)
        engram_inputs = None
        if self.config.engram_layer_ids:
            engram_inputs = self._engram_inputs(inputs, cache, positions)
            if positions is not None:
                positions.history = (
                    positions.planes
                    if cache is None
                    else _history_planes(_slot_history(cache))
                )
        with exact.scope(inputs if inputs is not None else inputs_embeds, cache):
            hidden = self.model(inputs, cache, inputs_embeds, engram_inputs, positions)
            logits = None if kwargs.get("skip_logits") else self.project(hidden)
        return LanguageModelOutput(
            logits=logits,
            hidden_states=[hidden] if kwargs.get("return_hidden") else None,
        )
