# SPDX-License-Identifier: Apache-2.0
"""DFly2, the block drafter shipped inside Hy4 and Hy4-flash checkpoints.

A draft block is ``[bonus, MASK x N]`` at positions ``a .. a+N``, where ``a``
is the bonus token's position. Slot 0 fuses the bonus embedding with the
target's final hidden state of the last committed row (``mtp_fusion``). Five
single-stream pre-norm layers run over the block; layer ``i`` attends to

* the latent KV cache of target layer ``dfly_target_kv_layers[i]`` at the
  slots that target layer's indexer selected for the last committed row
  (every committed slot while the context is shorter than the indexer's
  top-k), and
* the block's own keys, causally.

The drafter keeps no KV cache of its own. Each draft slot ``k + 1`` is then
corrected with the previous token's embedding (``hidden_correction``) and
projected by the target LM head, so drafts are produced left to right.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.mla import MultiLinear

from .config import ModelConfig, TextConfig


@dataclass
class DFlyConfig:
    layers: int
    block_size: int
    mask_token_id: int
    kv_layers: tuple[int, ...]
    index_layers: tuple[int, ...]
    ffn_width: int
    post_proj_norm: bool

    @classmethod
    def from_model(cls, config: ModelConfig, ffn_width: int, post_proj_norm: bool):
        n = int(config.dfly_num_hidden_layers or 0)
        kv = tuple(config.dfly_target_kv_layers or ())
        index = tuple(config.dfly_target_index_layer or kv)
        if not n or len(kv) != n or len(index) != n:
            raise ValueError("DFly needs one target KV and index layer per layer")
        if config.dfly_context_local_window:
            raise ValueError("DFly local-window context is not supported")
        if not config.dfly_attn_causal or config.dfly_enable_ihc:
            raise ValueError("Only the causal, single-stream DFly layout is supported")
        speculative = config.speculative_config or {}
        # Hy4-flash's config carries no block size; its serving default is 6.
        block = config.dfly_block_size or (
            int(speculative.get("num_speculative_tokens", 5)) + 1
        )
        return cls(
            layers=n,
            block_size=int(block),
            mask_token_id=int(config.dfly_mask_token_id),
            kv_layers=kv,
            index_layers=index,
            ffn_width=ffn_width,
            post_proj_norm=post_proj_norm,
        )


class DFlyAttention(nn.Module):
    """Gated MLA with sinks over shared target KV plus the block's own keys."""

    def __init__(self, args: TextConfig, post_proj_norm: bool):
        super().__init__()
        heads = args.num_attention_heads
        nope, rope, value = (
            args.qk_nope_head_dim,
            args.qk_rope_head_dim,
            args.v_head_dim,
        )
        d = args.hidden_size
        self.num_heads = heads
        self.nope = nope
        self.rope_dim = rope
        self.v_head_dim = value
        self.rank = args.kv_lora_rank
        self.scale = (nope + rope) ** -0.5
        self.q_a_proj = nn.Linear(d, args.q_lora_rank, bias=False)
        self.q_a_layernorm = nn.RMSNorm(args.q_lora_rank, eps=args.rms_norm_eps)
        self.q_b_proj = nn.Linear(args.q_lora_rank, heads * (nope + rope), bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(d, self.rank + rope, bias=False)
        self.kv_a_layernorm = nn.RMSNorm(self.rank, eps=args.rms_norm_eps)
        self.embed_q = MultiLinear(nope, self.rank, heads)
        self.unembed_out = MultiLinear(self.rank, value, heads)
        self.linear_gate = nn.Linear(d, heads * value, bias=False)
        self.o_proj = nn.Linear(heads * value, d, bias=False)
        self.learnable_sink_param = mx.zeros((heads,), dtype=mx.float32)
        if post_proj_norm:
            self.post_proj_norm = nn.RMSNorm(d, eps=args.rms_norm_eps)
        self.rope = nn.RoPE(rope, traditional=True, base=args.rope_theta("sparse"))

    def __call__(self, x, start, history):
        """``history`` is (latent [1, 1, S, rank], k_pe [1, 1, S, rope], valid).

        ``valid`` is an [S] mask of usable history slots, or None for all.
        """
        b, width, _ = x.shape
        gate = mx.sigmoid(self.linear_gate(x)).reshape(
            b, width, self.num_heads, self.v_head_dim
        )
        qr = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(qr).reshape(b, width, self.num_heads, -1).swapaxes(1, 2)
        q_nope, q_pe = mx.split(q, [self.nope], axis=-1)
        latent, k_pe = mx.split(self.kv_a_proj_with_mqa(x), [self.rank], axis=-1)
        latent = self.kv_a_layernorm(latent)[:, None]
        k_pe = self.rope(k_pe.reshape(b, 1, width, self.rope_dim), start)
        q_pe = self.rope(q_pe, start)
        hist_latent, hist_pe, hist_valid = history
        latent = mx.concatenate([hist_latent.astype(latent.dtype), latent], axis=2)
        k_pe = mx.concatenate([hist_pe.astype(k_pe.dtype), k_pe], axis=2)
        past = hist_latent.shape[2]
        rows = mx.arange(width)[:, None]
        cols = mx.arange(past + width)[None]
        allowed = (cols < past) | (cols - past <= rows)
        if hist_valid is not None:
            allowed = allowed & mx.concatenate(
                [hist_valid, mx.ones((width,), dtype=mx.bool_)]
            )
        scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        scores = mx.where(
            allowed, scores, mx.array(mx.finfo(scores.dtype).min, scores.dtype)
        )
        out = mx.fast.scaled_dot_product_attention(
            self.embed_q(q_nope),
            latent,
            latent,
            scale=self.scale,
            mask=scores,
            sinks=self.learnable_sink_param.astype(x.dtype),
        )
        out = self.unembed_out(out).swapaxes(1, 2) * gate
        out = self.o_proj(out.reshape(b, width, -1))
        if "post_proj_norm" in self:
            out = self.post_proj_norm(out)
        return out


class DFlyMLP(nn.Module):
    def __init__(self, hidden: int, width: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, width, bias=False)
        self.up_proj = nn.Linear(hidden, width, bias=False)
        self.down_proj = nn.Linear(width, hidden, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DFlyLayer(nn.Module):
    def __init__(self, args: TextConfig, cfg: DFlyConfig):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.self_attn = DFlyAttention(args, cfg.post_proj_norm)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.mlp = DFlyMLP(args.hidden_size, cfg.ffn_width)

    def __call__(self, x, start, history):
        x = x + self.self_attn(self.input_layernorm(x), start, history)
        return x + self.mlp(self.post_attention_layernorm(x))


class DFlyFusion(nn.Module):
    """Slot 0: ``E[bonus] + gate * eh_proj([enorm(E[bonus]), hnorm(h)])``."""

    def __init__(self, args: TextConfig):
        super().__init__()
        d = args.hidden_size
        self.enorm = nn.RMSNorm(d, eps=args.rms_norm_eps)
        self.hnorm = nn.RMSNorm(d, eps=args.rms_norm_eps)
        self.eh_proj = nn.Linear(2 * d, d, bias=False)
        self.gate = mx.zeros((d,))

    def __call__(self, anchor, hidden):
        fused = self.eh_proj(
            mx.concatenate([self.enorm(anchor), self.hnorm(hidden)], axis=-1)
        )
        return anchor + self.gate * fused


class DFlyCorrection(nn.Module):
    """``h + down(silu(gate(z)) * up(z))`` with ``z = [hnorm(h), enorm(E[prev])]``."""

    def __init__(self, args: TextConfig, width: int):
        super().__init__()
        d = args.hidden_size
        self.hidden_norm = nn.RMSNorm(d, eps=args.rms_norm_eps)
        self.embed_norm = nn.RMSNorm(d, eps=args.rms_norm_eps)
        self.gate_proj = nn.Linear(2 * d, width, bias=False)
        self.up_proj = nn.Linear(2 * d, width, bias=False)
        self.down_proj = nn.Linear(width, d, bias=False)

    def __call__(self, hidden, previous):
        z = mx.concatenate(
            [self.hidden_norm(hidden), self.embed_norm(previous)], axis=-1
        )
        return hidden + self.down_proj(nn.silu(self.gate_proj(z)) * self.up_proj(z))


class DFlyModel(nn.Module):
    def __init__(self, args: TextConfig, cfg: DFlyConfig, correction_width: int):
        super().__init__()
        self.cfg = cfg
        self.layers = [DFlyLayer(args, cfg) for _ in range(cfg.layers)]
        self.final_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.mtp_fusion = DFlyFusion(args)
        self.hidden_correction = DFlyCorrection(args, correction_width)

    def block_hidden(self, embed, target_hidden, bonus, start, histories, width):
        """Final-normed hidden of the block ``[bonus, MASK x (width - 1)]``.

        ``target_hidden`` is the target's [1, 1, D] post-norm hidden of the
        last committed row and ``start`` the bonus token's position.
        """
        ids = mx.concatenate(
            [
                bonus.reshape(1, 1).astype(mx.int32),
                mx.full((1, width - 1), self.cfg.mask_token_id, dtype=mx.int32),
            ],
            axis=1,
        )
        x = embed(ids)
        first = self.mtp_fusion(x[:, :1], target_hidden.astype(x.dtype))
        x = mx.concatenate([first, x[:, 1:]], axis=1)
        for layer, history in zip(self.layers, histories):
            x = layer(x, start, history)
        return self.final_norm(x)

    def correct(self, embed, hidden, previous):
        """Hidden of one draft slot corrected with its previous token."""
        prev = embed(previous.reshape(-1, 1).astype(mx.int32))[:, 0]
        return self.hidden_correction(hidden, prev.astype(hidden.dtype))
