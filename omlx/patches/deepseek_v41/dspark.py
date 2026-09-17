# SPDX-License-Identifier: MIT
"""V4.1 DSpark stages, following the published single-pass mHC forward.

The physical context ring is shared with the V4 implementation. Draft keys
never enter that ring. Server acceptance/rollback is a separate integration.
"""

from dataclasses import replace

import mlx.core as mx
import mlx.nn as nn

from ..mlx_lm_mtp.deepseek_v4_dspark import DSparkContextCache
from .head import project_logits
from .language import Attention, Block, RMSNorm, hc_mixes, hc_post, hc_pre, rope
from .quantization import quantize_activation


class DSparkAttention(Attention):
    def append_context(self, main_x, cache, *, start_offset=None):
        c = self._config
        start = cache.offset if start_offset is None else start_offset
        positions = mx.arange(start, start + main_x.shape[1])
        kv = quantize_activation(
            rope(self.kv_norm(self.wkv(main_x)), positions, c, False)
        )
        cache.append(kv[:, None], start_offset=start_offset)

    def __call__(self, x, cache):
        c = self._config
        batch, length, _ = x.shape
        positions = mx.arange(cache.offset, cache.offset + length)
        q = rope(
            self.wq_b(self.q_norm(self.wq_a(x))).reshape(
                batch, length, c.n_heads, c.head_dim
            ),
            positions,
            c,
            False,
        )
        draft_kv = quantize_activation(
            rope(self.kv_norm(self.wkv(x)), positions, c, False)
        )
        kv = mx.concatenate([cache.keys[:, 0], draft_kv], axis=1)
        # V4.1 does not apply V4's per-head query RMS normalization.
        # The checkpoint's sink and attention accumulation are FP32. SDPA
        # requires a matching sink dtype, so promote this small draft block
        # instead of rounding the learned sink down to BF16.
        out = (
            mx.fast.scaled_dot_product_attention(
                q.astype(mx.float32).transpose(0, 2, 1, 3),
                kv[:, None].astype(mx.float32),
                kv[:, None].astype(mx.float32),
                scale=c.head_dim**-0.5,
                sinks=self.attn_sink.astype(mx.float32),
            )
            .transpose(0, 2, 1, 3)
            .astype(q.dtype)
        )
        out = rope(out, positions, c, False, inverse=True)
        grouped = out.reshape(batch, length, c.o_groups, -1)
        weight = self.wo_a.weight.reshape(c.o_groups, c.o_lora_rank, -1)
        return self.wo_b(mx.einsum("bsgd,grd->bsgr", grouped, weight).flatten(-2))


class MarkovHead(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.embed = nn.Embedding(c.vocab_size, c.dspark_markov_rank)
        self.head = nn.Linear(c.dspark_markov_rank, c.vocab_size, bias=False)

    def __call__(self, ids):
        embedding = self.embed(ids)
        return (
            project_logits(embedding, self.head.weight),
            embedding,
        )


class ConfidenceHead(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.proj = nn.Linear(c.dim + c.dspark_markov_rank, 1, bias=False)

    def __call__(self, h, markov):
        joined = mx.concatenate([h, markov], axis=-1).astype(mx.float32)
        return (joined @ self.proj.weight.astype(mx.float32).T).squeeze(-1)


class DSparkBlock(Block):
    def __init__(self, c, stage):
        experts, active = c.get_moe_config(c.n_layers + stage)
        draft = replace(
            c,
            n_routed_experts=experts,
            n_activated_experts=active,
            engram_layer_ids=(),
            engram_num_embeddings=(),
        )
        super().__init__(draft, c.n_layers + stage)
        self.attn = DSparkAttention(draft, c.n_layers + stage)
        if stage == 0:
            self.main_proj = nn.Linear(
                c.dim * len(c.dspark_target_layer_ids), c.dim, bias=False
            )
            self.main_norm = RMSNorm(c.dim, c.norm_eps)
        if stage == c.n_mtp_layers - 1:
            self.norm = RMSNorm(c.dim, c.norm_eps)
            self.markov_head = MarkovHead(c)
            self.confidence_head = ConfidenceHead(c)

    def __call__(self, h, pre, cache):
        ap, ao, ac = hc_mixes(
            h, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base, self._config
        )
        h = hc_post(self.attn(self.attn_norm(hc_pre(h, pre)), cache), h, ao, ac)
        fp, fo, fc = hc_mixes(
            h, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base, self._config
        )
        h = hc_post(self.ffn(self.ffn_norm(hc_pre(h, ap)), None), h, fo, fc)
        return h, fp


def make_stages(c):
    if c.dspark_block_size <= 0 or c.n_mtp_layers <= 0 or not c.dspark_target_layer_ids:
        raise ValueError("MTP preservation requires a complete DSpark configuration")
    if any(i < 0 or i >= c.n_layers for i in c.dspark_target_layer_ids):
        raise ValueError("Invalid DSpark target layer")
    if len(c.compress_ratios) < c.n_layers + c.n_mtp_layers or any(
        c.compress_ratios[c.n_layers : c.n_layers + c.n_mtp_layers]
    ):
        raise ValueError("DSpark stages require uncompressed local attention")
    return [DSparkBlock(c, i) for i in range(c.n_mtp_layers)]


def make_cache(c):
    return [DSparkContextCache(c.window_size) for _ in range(c.n_mtp_layers)]


def forward_spec(model, input_ids, main_hidden, cache, *, temperature=0.0):
    """Seed context on the first call; subsequently propose a full draft block.

    main_hidden must contain only newly committed target positions. Calls after
    priming append exactly one position, matching the published DSpark contract.
    Returned proposals still require verification by the target model.
    """
    c = model._config
    if not c.preserve_mtp:
        raise ValueError("This checkpoint was loaded without DSpark weights")
    if len(cache) != len(model.mtp) or len({item.offset for item in cache}) != 1:
        raise ValueError("DSpark stage cache offsets differ")
    if input_ids.ndim != 2 or input_ids.shape[1] != 1:
        raise ValueError("DSpark requires one initial token per batch row")
    if (
        main_hidden.ndim != 3
        or main_hidden.shape[0] != input_ids.shape[0]
        or main_hidden.shape[-1] != c.dim * len(c.dspark_target_layer_ids)
    ):
        raise ValueError("Invalid DSpark target hidden shape")
    initial = cache[0].offset == 0
    if main_hidden.shape[1] < 1 or (not initial and main_hidden.shape[1] != 1):
        raise ValueError("DSpark decode appends exactly one committed target position")
    first = model.mtp[0]
    main_x = first.main_norm(first.main_proj(main_hidden))
    for stage, item in zip(model.mtp, cache):
        stage.attn.append_context(main_x, item)
    if initial:
        return None
    logits, h = proposal_forward(model, input_ids, cache)
    final = model.mtp[-1]
    proposed, biases, embeddings = [input_ids[:, 0]], [], []
    for i in range(c.dspark_block_size):
        bias, embedding = final.markov_head(proposed[-1])
        step = logits[:, i] + bias
        token = (
            mx.argmax(step, axis=-1)
            if temperature == 0
            else mx.random.categorical(step / temperature)
        )
        proposed.append(token.astype(input_ids.dtype))
        biases.append(step)
        embeddings.append(embedding)
    return (
        mx.stack(proposed, axis=1),
        mx.stack(biases, axis=1),
        final.confidence_head(h, mx.stack(embeddings, axis=1)),
    )


def proposal_forward(model, input_ids, cache, draft_length=None):
    """Compute parallel draft logits; the shared loop applies Markov biases."""
    c = model._config
    width = max(1, min(draft_length or c.dspark_block_size, c.dspark_block_size))
    input_ids = input_ids.reshape(input_ids.shape[0], -1)[:, -1:]
    ids = mx.concatenate(
        [
            input_ids,
            mx.full(
                (input_ids.shape[0], width - 1),
                c.dspark_noise_token_id,
                dtype=input_ids.dtype,
            ),
        ],
        axis=1,
    )
    h = mx.repeat(model.embed(ids)[..., None, :], c.hc_mult, -2)
    pre = mx.broadcast_to((mx.arange(c.hc_mult) == 0).astype(mx.float32), h.shape[:-1])
    for stage, item in zip(model.mtp, cache):
        h, pre = stage(h, pre, item)
    final = model.mtp[-1]
    h = hc_pre(h, pre)
    logits = project_logits(final.norm(h), model.head.weight)
    return logits, h
