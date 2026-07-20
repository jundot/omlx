# SPDX-License-Identifier: Apache-2.0
"""Bonsai DSpark drafter in MLX.

Architecture: 6-layer cross-attention backbone identical to mlx-dspark's
DSparkDrafter (qwen3 style: 2-norm layers, silu MLP, standard GQA attention)
plus a Prism extension: log-SNR sinusoidal conditioning injected into the
fused target hidden state before the backbone.

Log-SNR conditioning:
  given scalar γ (log signal-to-noise ratio), produce a 128-dim sinusoidal
  embedding, then pass through a 2-layer MLP (log_snr_fc1, log_snr_fc2) to
  obtain a 5120-dim bias added to the fused target hidden state. At inference
  we use a fixed γ = config.log_snr_inference (default 10.0 = very clean signal).

Weight sharing:
  token_embd.weight in the GGUF is dtype=42 (Prism ternary) and is shared with
  the target model's embedding. Call bind_target_embedding(target_model) after
  loading both models to wire the target's embedding into the drafter.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .config import BonsaiDSparkConfig


# ---------------------------------------------------------------------------
# Sub-modules mirroring mlx-dspark's DSparkDrafter components
# ---------------------------------------------------------------------------


class _MLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class _CtxCache:
    """Append-only per-layer context K/V cache (identical to mlx-dspark CtxCache)."""

    __slots__ = ("k", "v")

    def __init__(self):
        self.k = None
        self.v = None

    def append(self, k: mx.array, v: mx.array) -> None:
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = mx.concatenate([self.k, k], axis=2)
            self.v = mx.concatenate([self.v, v], axis=2)

    def trim_to(self, length: int) -> None:
        if self.k is not None and length < self.k.shape[2]:
            self.k = self.k[:, :, :length, :]
            self.v = self.v[:, :, :length, :]

    @property
    def length(self) -> int:
        return 0 if self.k is None else self.k.shape[2]


class _CrossAttention(nn.Module):
    """Cross-attention: Q from draft block, K/V from [context, block]."""

    def __init__(self, config: BonsaiDSparkConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = config.scaling

        h = config.hidden_size
        self.q_proj = nn.Linear(h, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(h, self.n_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(h, self.n_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, h, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=config.rope_theta)

    def _kv(self, x: mx.array):
        B, S, _ = x.shape
        k = self.k_proj(x).reshape(B, S, self.n_kv, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, S, self.n_kv, self.head_dim).transpose(0, 2, 1, 3)
        return self.k_norm(k), v

    def update_ctx(self, fused_new: mx.array, ctx_offset: int, cache: _CtxCache) -> None:
        k, v = self._kv(fused_new)
        cache.append(self.rope(k, offset=ctx_offset), v)

    def attend(self, hidden: mx.array, block_offset: int, cache: _CtxCache) -> mx.array:
        B, q_len, _ = hidden.shape
        q = self.q_proj(hidden).reshape(B, q_len, self.n_heads, self.head_dim)
        q = self.rope(self.q_norm(q).transpose(0, 2, 1, 3), offset=block_offset)

        k_blk, v_blk = self._kv(hidden)
        k_blk = self.rope(k_blk, offset=block_offset)
        k = mx.concatenate([cache.k, k_blk], axis=2)
        v = mx.concatenate([cache.v, v_blk], axis=2)

        # GQA: repeat KV heads to match Q heads
        if self.n_heads != self.n_kv:
            repeats = self.n_heads // self.n_kv
            k = mx.repeat(k, repeats, axis=1)
            v = mx.repeat(v, repeats, axis=1)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, q_len, -1))


class _DecoderLayer(nn.Module):
    """Qwen3/Llama-style 2-norm layer."""

    def __init__(self, config: BonsaiDSparkConfig):
        super().__init__()
        self.self_attn = _CrossAttention(config)
        self.mlp = _MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, hidden: mx.array, block_offset: int, cache: _CtxCache) -> mx.array:
        h = hidden + self.self_attn.attend(self.input_layernorm(hidden), block_offset, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class _VanillaMarkov(nn.Module):
    """Rank-256 previous-token correction: logits += markov_w2(markov_w1[prev_token])."""

    def __init__(self, vocab: int, rank: int):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab, rank)
        # markov_w2 is vocab-scale; keep quantized same as lm_head
        self.markov_w2 = nn.QuantizedLinear(rank, vocab, bias=False, group_size=32, bits=4)

    def prev_embeddings(self, token_ids: mx.array) -> mx.array:
        return self.markov_w1(token_ids)

    def step_bias(self, token_ids: mx.array) -> mx.array:
        return self.markov_w2(self.markov_w1(token_ids))


class _ConfidenceHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, 1)

    def __call__(self, features: mx.array) -> mx.array:
        return self.proj(features).squeeze(-1)


# ---------------------------------------------------------------------------
# Log-SNR sinusoidal embedding (diffusion-style timestep embedding)
# ---------------------------------------------------------------------------


def _log_snr_sinusoidal(log_snr: float, dim: int) -> mx.array:
    """Produce a ``[1, 1, dim]`` sinusoidal embedding of ``log_snr``.

    Uses the standard diffusion timestep embedding formula with half-dim
    sin frequencies and half-dim cos frequencies, scaled by the log-SNR scalar.
    """
    half = dim // 2
    # freqs: 10000 ^ (-2i / half_dim) for i in 0..half-1
    freqs = mx.exp(-math.log(10000.0) * mx.arange(half, dtype=mx.float32) / half)
    args = mx.array([log_snr], dtype=mx.float32) * freqs  # [half]
    emb = mx.concatenate([mx.sin(args), mx.cos(args)])    # [dim]
    return emb.reshape(1, 1, dim)


# ---------------------------------------------------------------------------
# Main drafter
# ---------------------------------------------------------------------------


class BonsaiDSparkDrafter(nn.Module):
    """Bonsai 27B DSpark drafter with log-SNR conditioning.

    Usage
    -----
    After loading::

        drafter = BonsaiDSparkDrafter(config)
        drafter.load_weights("drafter.safetensors")
        drafter.bind_target_embedding(target_model)   # share token embedding

    Then use with :func:`~omlx.custom_kernels.bonsai.dspark.generate.speculative_generate`.
    """

    def __init__(self, config: BonsaiDSparkConfig):
        super().__init__()
        self.config = config
        self.block_size = config.block_size
        self.mask_token_id = config.mask_token_id

        h = config.hidden_size
        v = config.vocab_size

        # Token embedding — will be replaced by bind_target_embedding() if weights
        # are unavailable (the GGUF token_embd.weight is dtype=42 / unreadable).
        self.embed_tokens = nn.Embedding(v, h)

        # Fused target hidden projection: [n_tap * h] → [h]
        n_tap = len(config.target_layer_ids)
        self.fc = nn.Linear(n_tap * h, h, bias=False)
        self.hidden_norm = nn.RMSNorm(h, eps=config.rms_norm_eps)

        # Log-SNR conditioning MLP: [log_snr_dim] → [h] → [h]
        self.log_snr_fc1 = nn.Linear(config.log_snr_dim, h, bias=True)
        self.log_snr_fc2 = nn.Linear(h, h, bias=True)

        # Cross-attention backbone
        self.layers = [_DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(h, eps=config.rms_norm_eps)

        # Output head — stored as QuantizedLinear (MLX Q4 group_size=32 bits=4)
        # to avoid loading a ~5 GB fp32 weight matrix.  The safetensors file
        # written by convert_gguf stores the packed uint32 + scales + biases.
        self.lm_head = nn.QuantizedLinear(h, v, bias=False, group_size=32, bits=4)

        # Markov head — markov_w2 is also vocab-scale; keep quantized
        self.markov_head = _VanillaMarkov(v, config.markov_rank)

        # Confidence head
        self.confidence_head = None
        if config.enable_confidence_head:
            in_dim = h + (config.markov_rank if config.confidence_head_with_markov else 0)
            self.confidence_head = _ConfidenceHead(in_dim)

        # Cached log-SNR embedding (recomputed only when log_snr changes)
        self._log_snr_cache: tuple[float, mx.array] | None = None

    # -----------------------------------------------------------------------
    # Target embedding binding
    # -----------------------------------------------------------------------

    def bind_target_embedding(self, target_model) -> None:
        """Replace embed_tokens with the target model's embedding layer.

        The Bonsai drafter shares the vocabulary with the target (Qwen3.5 VLM).
        The GGUF token_embd.weight is dtype=42 (Prism ternary) and cannot be
        dequantized; instead we borrow the already-loaded target embedding.

        Supports both mlx-lm (model.model.embed_tokens) and mlx-vlm
        (model.language_model.model.embed_tokens) target layouts.
        """
        if hasattr(target_model, "language_model"):
            self.embed_tokens = target_model.language_model.model.embed_tokens
        elif hasattr(target_model, "model") and hasattr(target_model.model, "embed_tokens"):
            self.embed_tokens = target_model.model.embed_tokens
        else:
            raise ValueError(
                "Cannot find embed_tokens on target_model; "
                "expected .language_model.model.embed_tokens or .model.embed_tokens"
            )

    # -----------------------------------------------------------------------
    # Cache helpers
    # -----------------------------------------------------------------------

    def make_ctx_cache(self) -> list[_CtxCache]:
        return [_CtxCache() for _ in self.layers]

    # -----------------------------------------------------------------------
    # Log-SNR conditioning
    # -----------------------------------------------------------------------

    def _log_snr_bias(self, log_snr: float) -> mx.array:
        """Return ``[1, 1, hidden_size]`` conditioning bias for ``log_snr``."""
        if self._log_snr_cache is not None and self._log_snr_cache[0] == log_snr:
            return self._log_snr_cache[1]
        emb = _log_snr_sinusoidal(log_snr, self.config.log_snr_dim)
        h1 = nn.gelu(self.log_snr_fc1(emb))
        bias = self.log_snr_fc2(h1)
        self._log_snr_cache = (log_snr, bias)
        return bias

    # -----------------------------------------------------------------------
    # Context update (called once per committed token group during prefill / verify)
    # -----------------------------------------------------------------------

    def fuse_target(self, target_hidden_cat: mx.array, log_snr: float | None = None) -> mx.array:
        """Project concatenated tap states [B, L, n_tap*H] → [B, L, H]
        and optionally add log-SNR conditioning bias."""
        fused = self.hidden_norm(self.fc(target_hidden_cat))
        if log_snr is not None:
            fused = fused + self._log_snr_bias(log_snr)
        return fused

    def update_context(self, target_hidden_cat: mx.array, ctx_offset: int,
                       ctx_caches: list[_CtxCache],
                       log_snr: float | None = None) -> None:
        fused = self.fuse_target(target_hidden_cat, log_snr)
        for layer, cache in zip(self.layers, ctx_caches):
            layer.self_attn.update_ctx(fused, ctx_offset, cache)

    # -----------------------------------------------------------------------
    # Backbone forward
    # -----------------------------------------------------------------------

    def backbone(self, noise_embedding: mx.array, block_offset: int,
                 ctx_caches: list[_CtxCache]) -> mx.array:
        h = noise_embedding
        for layer, cache in zip(self.layers, ctx_caches):
            h = layer(h, block_offset, cache)
        return self.norm(h)

    def compute_logits(self, hidden: mx.array) -> mx.array:
        return self.lm_head(hidden)

    # -----------------------------------------------------------------------
    # Block drafting
    # -----------------------------------------------------------------------

    def _embed(self, ids: mx.array) -> mx.array:
        return self.embed_tokens(ids)

    def _draft_logits(self, block_ids: list[int], ctx_caches: list[_CtxCache],
                      cap: int, block_offset: int) -> mx.array:
        """Run backbone over a block, return logits for the first ``cap`` positions."""
        emb = self._embed(mx.array([block_ids]))          # [1, k, H]
        h = self.backbone(emb, block_offset, ctx_caches)  # [1, k, H]
        return self.compute_logits(h[0, :cap])             # [cap, V]

    def sample_block(self, base_logits: mx.array, first_prev_token: int) -> mx.array:
        """Greedy sequential draft with Markov correction."""
        k = base_logits.shape[0]
        tokens = []
        prev = mx.array([first_prev_token])
        for i in range(k):
            step = base_logits[i] + self.markov_head.step_bias(prev)[0]
            nxt = mx.argmax(step, axis=-1, keepdims=True)
            tokens.append(nxt)
            prev = nxt
        return mx.concatenate(tokens)

    def sample_block_probs(self, base_logits: mx.array, first_prev_token: int,
                           temperature: float, top_p: float = 1.0,
                           top_k: int = 0) -> tuple[mx.array, mx.array]:
        """Temperature draft (for speculative sampling)."""
        from mlx_dspark.sampling import sample_probs, truncate_probs

        k = base_logits.shape[0]
        inv_t = 1.0 / temperature
        tokens, probs = [], []
        prev = mx.array([first_prev_token])
        for i in range(k):
            logits = base_logits[i]
            logits = logits + self.markov_head.step_bias(prev)[0]
            q = truncate_probs(mx.softmax(logits * inv_t, axis=-1), top_p, top_k)
            probs.append(q)
            nxt = sample_probs(q).reshape(1)
            tokens.append(nxt)
            prev = nxt
        return mx.concatenate(tokens), mx.stack(probs, axis=0)

    def confidence_logits(self, block_hidden: mx.array,
                          prev_token_ids: mx.array) -> mx.array | None:
        if self.confidence_head is None:
            return None
        if self.config.confidence_head_with_markov:
            feats = mx.concatenate(
                [block_hidden, self.markov_head.prev_embeddings(prev_token_ids)], axis=-1
            )
        else:
            feats = block_hidden
        return self.confidence_head(feats)
