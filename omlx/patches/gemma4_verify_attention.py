# SPDX-License-Identifier: Apache-2.0
"""Small-L attention route for gemma4 speculative verify forwards.

Gemma4 uses head_dim 256 for sliding layers and 512 for global layers. MLX
0.32.2 now handles small multi-row head-dim-256 calls with its native vector
kernel, so this patch explicitly leaves those shapes alone. Head-dim 512 still
has no fused MLX route; for global layers that unfused cost grows with context
and is what makes MTP verify cycles lose to plain decoding on low-accept
content.

Two routes, picked per layer:

- Global (full-attention, head_dim 512) layers: the fused multi-row
  vector kernel (``gemma4_verify_kernel``) scans the linear KV once per
  row pair for the whole block. Measured single-layer at 16k ctx
  (32q/4kv, bf16): L=2 792us vs 3805 stock, L=4 1349us vs 3796, L=8
  2586us vs 3834. Routed for L in [2, 12]; stock batched wins past ~12.
  If the kernel is unavailable the per-token fallback below covers
  L <= 3 (one full-KV scan per row — the pre-kernel route).
- Sliding layers: QKV stays batched but attention runs one token at a
  time on the fused L=1 vector kernel (L in {2, 3}; stock handles L >= 4
  where the window-capped ring keeps the unfused pass cheap). Routing
  sliding through the stock bf16 unfused pass instead was measured to
  cost story-content draft acceptance (65 -> 53% at 4k ctx on the 26B):
  its bf16 score rounding across 50 layers flips near-tie verify rows.
  The fp32-score per-token path keeps verify noise at the shipped level.

The routes ignore the mask argument (verify blocks are strictly
end-aligned causal, and decode-step masks carry no bidirectional image
overlay — non-square masks bypass it upstream). Logit parity vs stock:
near-tie rows (top-2 gap under ~0.2) can flip — same class as the
verify-qmm route — while stable rows agree; max|dlogit| ~1-2.5
post-softcap.

Scope gates (everything else passes through unchanged):
- cache present, no ``shared_kv`` input;
- backbones without KV sharing only (``num_kv_shared_layers == 0`` —
  26B/31B): on E2B/E4B a decomposed donor would hand downstream shared
  layers a final-token KV view they cannot causally slice on a rotated
  ring;
- zero left padding (memoized per cache object; the singleton MTP path
  guarantees compact caches);
- the kernel route additionally requires a lane-splittable head_dim and a
  plain 4-D KV buffer of the activations dtype (quantized caches fall
  back to per-token).
"""

from __future__ import annotations

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)

_MIN_L = 2
_MAX_L = 3  # per-token route ceiling (sliding layers / kernel fallback)
_KERNEL_MAX_L = 12  # fused kernel ceiling; stock batched wins past ~12

_MLX0322_FULL_HEAD_DIMS = frozenset({64, 72, 80, 96, 128})
_MLX0322_VECTOR_HEAD_DIMS = frozenset({64, 96, 128, 256})


def _mlx0322_default_fused(head_dim: int, query_len: int, gqa_factor: int) -> bool:
    if query_len <= 8:
        return (
            head_dim in _MLX0322_VECTOR_HEAD_DIMS
            and query_len * gqa_factor <= 32
        )
    return head_dim in _MLX0322_FULL_HEAD_DIMS


def apply() -> bool:
    """Wrap ``mlx_vlm.models.gemma4.language.Attention.__call__``. Idempotent."""
    try:
        from mlx_vlm.models.gemma4 import language as g4_lang
    except Exception as e:
        logger.debug(f"mlx_vlm.gemma4 not importable for verify attention: {e}")
        return False

    from . import gemma4_verify_kernel

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

    def _kernel_cache_ok(cache, dtype) -> bool:
        keys = getattr(cache, "keys", None)
        values = getattr(cache, "values", None)
        return (
            isinstance(keys, mx.array)
            and isinstance(values, mx.array)
            and keys.ndim == 4
            and keys.dtype == dtype
            and values.dtype == dtype
        )

    def __call__(self, x, mask=None, cache=None, shared_kv=None, offset=None):
        B, L, _ = x.shape
        gqa_factor = self.n_heads // self.n_kv_heads
        if (
            shared_kv is not None
            or cache is None
            or L < _MIN_L
            or getattr(self.config, "num_kv_shared_layers", 0)
            or not _zero_left_padding(cache)
        ):
            return original_call(
                self, x, mask, cache, shared_kv=shared_kv, offset=offset
            )

        # Do not replace a shape MLX 0.32.2 already sends to a native fused
        # kernel. This is the common Gemma4 sliding-layer verify path at
        # head_dim=256 and L<=4.
        if _mlx0322_default_fused(self.head_dim, L, gqa_factor):
            return original_call(
                self, x, mask, cache, shared_kv=shared_kv, offset=offset
            )

        use_kernel = (
            not self.is_sliding
            and L <= _KERNEL_MAX_L
            and self.head_dim % 32 == 0  # kernel splits head_dim across lanes
            and _kernel_cache_ok(cache, x.dtype)
            and gemma4_verify_kernel.is_available()
            # Device threadgroup budgets are pipeline-dependent (virtualized
            # GPUs cap below 32 * gqa threads); infeasible geometry falls
            # back before the batched cache update commits us to the route.
            and gemma4_verify_kernel.kernel_max_rows(
                gqa_factor, x.dtype
            )
            > 0
        )
        if not use_kernel and L > _MAX_L:
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

        if use_kernel:
            big_k, big_v = cache.update_and_fetch(keys, values)
            kv_last = (big_k, big_v)
            output = gemma4_verify_kernel.fused_verify_sdpa(
                queries, cache.keys, cache.values, big_k.shape[2], self.scale
            )
        else:
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
    logger.info("gemma4 small-L verify attention patch applied (kernel + per-token)")
    return True
