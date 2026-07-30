# SPDX-License-Identifier: Apache-2.0
"""Decomposed small-L attention for gemma4 speculative verify forwards.

MLX's fused SDPA only covers head_dim {64, 80, 128} for multi-token
queries; gemma4 (head_dim 256 sliding / 512 global) therefore drops to the
unfused path for any L >= 2 forward, whose cost grows with context — the
L=1 -> L=2 step measured +88% at 16k ctx on the 26B MoE. That step is what
makes MTP verify cycles lose to plain decoding on low-accept content.

This patch keeps the QKV projections batched over the L tokens but runs
the attention itself one token at a time, so every call takes the fused
L=1 vector kernel (which does support 256/512) and the sliding-window
ring cache stays on its in-place single-token path (no unrotation).
Causality comes for free: token i attends the cache right after its own
K/V were appended, exactly like decode.

Measured on gemma-4-26B-A4B oQ4e (M3 Ultra): L=2 @16k 25.9 -> 18.3 ms,
L=3 27.7 -> 25.9 ms; from L=4 up the per-token full-KV reads lose to the
stock multi-token pass, so the route is gated to L in {2, 3}. Logit
parity vs stock is argmax-stable with max|dlogit| ~2-3.5 post-softcap
(kernel accumulation-order noise, same class as the verify-qmm route).

Scope gates (everything else passes through unchanged):
- 2 <= L <= 3, cache present, no ``shared_kv`` input;
- backbones without KV sharing only (``num_kv_shared_layers == 0`` —
  26B/31B): on E2B/E4B a decomposed donor would hand downstream shared
  layers a final-token KV view they cannot causally slice on a rotated
  ring;
- zero left padding (memoized per cache object; the singleton MTP path
  guarantees compact caches).
"""

from __future__ import annotations

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)

_MIN_L = 2
_MAX_L = 3


def apply() -> bool:
    """Wrap ``mlx_vlm.models.gemma4.language.Attention.__call__``. Idempotent."""
    try:
        from mlx_vlm.models.gemma4 import language as g4_lang
    except Exception as e:
        logger.debug(f"mlx_vlm.gemma4 not importable for verify attention: {e}")
        return False

    cls = g4_lang.Attention
    if getattr(cls, "_omlx_verify_attn_patched", False):
        return True

    original_call = cls.__call__
    sdpa = g4_lang.scaled_dot_product_attention

    def _zero_left_padding(cache) -> bool:
        left = getattr(cache, "left_padding", None)
        if left is None:
            return True
        cached = getattr(cache, "_omlx_zero_left_pad", None)
        if cached is None:
            try:
                cached = max(int(v) for v in left.tolist()) == 0
            except Exception:
                cached = False
            cache._omlx_zero_left_pad = cached
        return cached

    def __call__(self, x, mask=None, cache=None, shared_kv=None, offset=None):
        B, L, _ = x.shape
        if (
            shared_kv is not None
            or cache is None
            or not (_MIN_L <= L <= _MAX_L)
            or getattr(self.config, "num_kv_shared_layers", 0)
            or not _zero_left_padding(cache)
        ):
            return original_call(
                self, x, mask, cache, shared_kv=shared_kv, offset=offset
            )

        queries = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        queries = self.q_norm(queries)
        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
        if self.use_k_eq_v:
            values = keys
        else:
            values = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)

        off = mx.array(cache.offset)
        keys = self.k_norm(keys)
        keys = keys.transpose(0, 2, 1, 3)
        keys = self.rope(keys, offset=off)
        values = self.v_norm(values)
        values = values.transpose(0, 2, 1, 3)
        queries = queries.transpose(0, 2, 1, 3)
        queries = self.rope(queries, offset=off)

        outs = []
        kv_last = None
        for i in range(L):
            big_k, big_v = cache.update_and_fetch(
                keys[..., i : i + 1, :], values[..., i : i + 1, :]
            )
            kv_last = (big_k, big_v)
            outs.append(
                sdpa(
                    queries[..., i : i + 1, :],
                    big_k,
                    big_v,
                    cache=cache,
                    scale=self.scale,
                    mask=None,
                )
            )

        output = mx.concatenate(outs, axis=2)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), kv_last, off

    cls.__call__ = __call__
    cls._omlx_verify_attn_patched = True
    logger.info("gemma4 decomposed small-L verify attention patch applied")
    return True
