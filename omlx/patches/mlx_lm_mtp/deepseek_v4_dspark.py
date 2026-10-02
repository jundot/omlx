# SPDX-License-Identifier: Apache-2.0
"""Embedded DSpark support for DeepSeek-V4-Flash-0731.

The 0731 checkpoint keeps a three-stage DSpark drafter under ``mtp.0..2``
inside the target checkpoint.  Unlike the older DeepSeek-V4 Lightning MTP
head, DSpark predicts a block in parallel from target-layer hidden states and
then adds a lightweight left-to-right Markov bias while sampling the block.

This module only owns the model-side pieces.  The existing native-MTP
``GenerationBatch`` integration remains responsible for target verification,
exact rejection sampling, rollback, and output delivery.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

_PRIME_CTX_ATTR = "_omlx_mtp_prime_ctx"


def is_dspark_config(config: Any) -> bool:
    """Return whether *config* declares an embedded DeepSeek DSpark head."""
    block_size = int(getattr(config, "dspark_block_size", 0) or 0)
    target_ids = tuple(getattr(config, "dspark_target_layer_ids", ()) or ())
    return block_size > 0 and bool(target_ids)


def stage_count(config: Any) -> int:
    """Resolve and validate the number of embedded DSpark decoder stages."""
    target_ids = tuple(getattr(config, "dspark_target_layer_ids", ()) or ())
    explicit = int(getattr(config, "n_mtp_layers", 0) or 0)
    count = explicit or len(target_ids)
    if count <= 0:
        raise ValueError("DeepSeek DSpark requires at least one target layer")
    return count


class DSparkContextCache:
    """Physical-ring context K/V owned by one DSpark decoder stage.

    Draft-block K/V is deliberately not committed here.  Only hidden states
    from target tokens that survived verification are appended, so rollback is
    handled entirely by the target cache and this cache always stays on the
    committed timeline.  Once full, slots follow the checkpoint reference's
    physical ``absolute_position % max_size`` order.  DSpark attends those
    slots directly rather than rotating them back to chronological order.
    """

    def __init__(self, max_size: int):
        self.max_size = int(max_size)
        self.offset = 0
        self.keys: Optional[mx.array] = None

    def _chronological(self, value: mx.array) -> mx.array:
        """Temporarily expose a full physical ring from oldest to newest."""
        if value.shape[2] < self.max_size or self.offset <= self.max_size:
            return value
        cutoff = self.offset % self.max_size
        if cutoff == 0:
            return value
        return mx.concatenate(
            [value[:, :, cutoff:], value[:, :, :cutoff]],
            axis=2,
        )

    def _physical(self, value: mx.array, offset: int) -> mx.array:
        """Place chronological entries into absolute-position ring slots."""
        if value.shape[2] < self.max_size:
            return value
        cutoff = offset % self.max_size
        if cutoff == 0:
            return value
        split = self.max_size - cutoff
        return mx.concatenate(
            [value[:, :, split:], value[:, :, :split]],
            axis=2,
        )

    def append(
        self,
        keys: mx.array,
        *,
        start_offset: Optional[int] = None,
    ) -> None:
        length = int(keys.shape[2])
        if start_offset is not None:
            start = int(start_offset)
            if self.keys is None:
                self.offset = start
            elif self.offset != start:
                raise ValueError(
                    "DSpark context is not contiguous: "
                    f"cache={self.offset}, append={start}"
                )
        if self.keys is None:
            next_keys = keys
        else:
            next_keys = mx.concatenate([self._chronological(self.keys), keys], axis=2)
        next_offset = self.offset + length
        if next_keys.shape[2] > self.max_size:
            next_keys = next_keys[:, :, -self.max_size :]
        self.keys = self._physical(next_keys, next_offset)
        self.offset = next_offset

@dataclass
class _DSparkPrimeContext:
    caches: list[DSparkContextCache]
    expected_target_offset: int


def _target_cache_offset(cache: Optional[list[Any]]) -> Optional[int]:
    if not cache:
        return None

    def read(entry: Any) -> Optional[int]:
        offset = getattr(entry, "offset", None)
        if type(offset) is int:
            return offset
        if offset is not None and getattr(offset, "size", 0) == 1:
            try:
                return int(offset.reshape(()).item())
            except Exception:
                return None
        return None

    for entry in cache:
        got = read(entry)
        if got is not None:
            return got
        for child in getattr(entry, "caches", ()) or ():
            got = read(child)
            if got is not None:
                return got
    return None


def drop_primed(host: Any) -> None:
    try:
        delattr(host, _PRIME_CTX_ATTR)
    except AttributeError:
        pass


def capture_prompt(
    host: Any,
    inputs: mx.array,
    aux_hidden: mx.array,
    target_cache: Optional[list[Any]],
) -> None:
    """Stream target-layer hidden states into the DSpark context cache."""
    if target_cache is None or inputs.ndim != 2 or inputs.shape[0] != 1:
        return
    offset_after = _target_cache_offset(target_cache)
    if offset_after is None:
        return
    seq_len = int(inputs.shape[1])
    start_offset = offset_after - seq_len
    ctx = getattr(host, _PRIME_CTX_ATTR, None)
    if not isinstance(ctx, _DSparkPrimeContext) or (
        ctx.expected_target_offset != start_offset
    ):
        drop_primed(host)
        ctx = _DSparkPrimeContext(
            caches=host.make_mtp_cache(),
            expected_target_offset=start_offset,
        )
        setattr(host, _PRIME_CTX_ATTR, ctx)
    try:
        host.dspark_append_context(
            aux_hidden,
            ctx.caches,
            start_offset=start_offset,
        )
    except Exception:
        drop_primed(host)
        logger.debug("DeepSeek DSpark prompt capture failed", exc_info=True)
        return
    ctx.expected_target_offset = offset_after
    arrays = []
    for cache in ctx.caches:
        if cache.keys is not None:
            arrays.append(cache.keys)
    if arrays:
        mx.async_eval(arrays)


def take_primed(
    host: Any,
    target_cache: Optional[list[Any]],
    _main_token: Any,
) -> Optional[tuple[list[DSparkContextCache], int]]:
    """Finish the prompt-to-decode seam for the native MTP batch loop."""
    ctx = getattr(host, _PRIME_CTX_ATTR, None)
    drop_primed(host)
    if not isinstance(ctx, _DSparkPrimeContext):
        return None
    target_offset = _target_cache_offset(target_cache)
    # Activation has already forwarded one target token with capture disabled.
    if target_offset is None or ctx.expected_target_offset != target_offset - 1:
        return None
    history_offset = min((cache.offset for cache in ctx.caches), default=0)
    return ctx.caches, history_offset


def register(dsv4: Any) -> None:
    """Register DSpark modules on the injected ``mlx_lm.deepseek_v4`` module."""
    if hasattr(dsv4, "DSparkBlock"):
        return

    HyperConnection = dsv4.HyperConnection
    HyperHead = dsv4.HyperHead
    DeepseekV4MoE = dsv4.DeepseekV4MoE
    LimitedSwiGLU = dsv4.LimitedSwiGLU
    hc_expand = dsv4.hc_expand
    scaled_dot_product_attention = dsv4.scaled_dot_product_attention

    class DSparkAttention(dsv4.LocalAttention):
        """Non-causal block attention over committed context plus draft block."""

        def append_context(
            self,
            main_x: mx.array,
            cache: DSparkContextCache,
            *,
            start_offset: Optional[int] = None,
        ) -> None:
            offset = cache.offset if start_offset is None else int(start_offset)
            batch, length, _ = main_x.shape
            kv = self.kv_norm(self.wkv(main_x)).reshape(batch, 1, length, self.head_dim)
            kv = self.rope(kv, offset)
            cache.append(kv, start_offset=start_offset)

        def __call__(
            self,
            x: mx.array,
            cache: DSparkContextCache,
            query_width: Optional[int] = None,
        ) -> mx.array:
            batch, block_width, _ = x.shape
            query_width = min(int(query_width or block_width), block_width)
            offset = cache.offset

            q = self.wq_b(self.q_norm(self.wq_a(x[:, :query_width])))
            q = q.reshape(batch, query_width, self.n_heads, self.head_dim)
            q = mx.fast.rms_norm(q, None, self.config.rms_norm_eps)
            q = self.rope(q.transpose(0, 2, 1, 3), offset)

            block_kv = self.kv_norm(self.wkv(x)).reshape(
                batch, 1, block_width, self.head_dim
            )
            block_kv = self.rope(block_kv, offset)
            if cache.keys is None:
                kv = block_kv
            else:
                kv = mx.concatenate([cache.keys, block_kv], axis=2)

            out = scaled_dot_product_attention(
                q,
                kv,
                kv,
                cache=None,
                scale=self.scale,
                mask=None,
                sinks=self.attn_sink.astype(q.dtype),
            )
            out = self.rope(out, offset, inverse=True)
            out = out.reshape(
                batch,
                self.o_groups,
                -1,
                query_width,
                self.head_dim,
            )
            out = out.transpose(0, 1, 3, 2, 4).flatten(-2)
            out = self.wo_a(out)
            out = out.transpose(0, 2, 1, 3).flatten(-2)
            out = self.wo_b(out)
            if self.sharding_group is not None:
                out = mx.distributed.all_sum(out, group=self.sharding_group)
            return out

    class DSparkMarkovHead(nn.Module):
        def __init__(self, config: Any):
            super().__init__()
            rank = int(config.dspark_markov_rank)
            self.markov_w1 = nn.Embedding(config.vocab_size, rank)
            self.markov_w2 = nn.Linear(rank, config.vocab_size, bias=False)

    class DSparkConfidenceHead(nn.Module):
        def __init__(self, config: Any):
            super().__init__()
            width = config.hidden_size + int(config.dspark_markov_rank)
            self.proj = nn.Linear(width, 1, bias=False)

    class DSparkBlock(nn.Module):
        """One checkpoint stage stored directly under ``mtp.<stage>``."""

        def __init__(self, config: Any, layer_idx: int, stage_idx: int, stages: int):
            super().__init__()
            self.attn = DSparkAttention(config, layer_idx)
            self.ffn = DeepseekV4MoE(config, layer_idx)
            self.ffn.switch_mlp.activation = LimitedSwiGLU(
                config.swiglu_limit,
                fp32=True,
            )
            self.ffn.shared_experts.fp32_swiglu = True
            self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.attn_hc = HyperConnection(config)
            self.ffn_hc = HyperConnection(config)
            if stage_idx == 0:
                fused_width = config.hidden_size * len(config.dspark_target_layer_ids)
                self.main_proj = nn.Linear(fused_width, config.hidden_size, bias=False)
                self.main_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if stage_idx == stages - 1:
                self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                self.hc_head = HyperHead(config)
                self.markov_head = DSparkMarkovHead(config)
                self.confidence_head = DSparkConfidenceHead(config)

        def __call__(
            self,
            hidden: mx.array,
            input_ids: mx.array,
            cache: DSparkContextCache,
            output_width: Optional[int] = None,
        ) -> mx.array:
            residual = hidden
            x, post, comb = self.attn_hc(hidden)
            x = self.attn(
                self.attn_norm(x),
                cache,
                query_width=output_width,
            )
            if output_width is not None and output_width < hidden.shape[1]:
                residual = residual[:, :output_width]
                post = post[:, :output_width]
                comb = comb[:, :output_width]
                input_ids = input_ids[:, :output_width]
            hidden = hc_expand(x, residual, post, comb)

            residual = hidden
            x, post, comb = self.ffn_hc(hidden)
            x = self.ffn(self.ffn_norm(x), input_ids)
            return hc_expand(x, residual, post, comb)

    dsv4.DSparkContextCache = DSparkContextCache
    dsv4.DSparkBlock = DSparkBlock


__all__ = [
    "DSparkContextCache",
    "capture_prompt",
    "is_dspark_config",
    "register",
    "stage_count",
    "take_primed",
]
