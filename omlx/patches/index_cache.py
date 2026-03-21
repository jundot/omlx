# SPDX-License-Identifier: Apache-2.0
"""IndexCache: skip redundant indexer computation in DSA layers.

Based on the IndexCache paper (arXiv:2603.12201) by THUDM/Z.ai.
Adjacent layers in DeepSeek Sparse Attention share 70-100% of selected
tokens.  By reusing topk indices from a "Full" layer in subsequent
"Shared" layers we skip the expensive Q*K attention + argpartition in
the Indexer while keeping the indexer KV cache up to date.

Supported model types: deepseek_v32, glm_moe_dsa
"""

from __future__ import annotations

import logging
from typing import Any, Optional

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

logger = logging.getLogger(__name__)

# Sentinel for supported model types
_SUPPORTED_MODEL_TYPES = {"deepseek_v32", "glm_moe_dsa"}

# Track whether the class-level patch has been applied
_class_patch_applied = False


def _get_model_type(model: Any) -> str | None:
    """Extract model_type string from a loaded mlx-lm model."""
    for attr in ("model_type", "args"):
        obj = getattr(model, attr, None)
        if obj is None:
            continue
        if isinstance(obj, str):
            return obj
        mt = getattr(obj, "model_type", None)
        if isinstance(mt, str):
            return mt
    return None


def _build_layer_pattern(num_layers: int, freq: int) -> list[bool]:
    """Build per-layer Full/Shared pattern.

    Returns a list of booleans where True = Full (compute indexer),
    False = Shared (reuse cached indices).
    Layer 0 is always Full.
    """
    pattern = []
    for i in range(num_layers):
        pattern.append(i % freq == 0)
    return pattern


def _update_indexer_cache_only(indexer: Any, x: Any, cache: Any) -> None:
    """Update the indexer KV cache without computing attention scores.

    This runs only the K projection path of the Indexer so future Full
    layers have correct keys.  Q projection, Q*K attention, and
    argpartition are all skipped.
    """
    b, s, _ = x.shape
    k = indexer.wk(x)
    k = indexer.k_norm(k)
    k = mx.reshape(k, (b, 1, s, indexer.head_dim))

    offset = cache.offset if cache is not None else 0
    k_pe, k_nope = mx.split(k, [indexer.rope_head_dim], axis=-1)
    k_pe = indexer.rope(k_pe, offset=offset)
    k = mx.concatenate([k_pe, k_nope], axis=-1)

    if cache is not None:
        cache.update_and_fetch(k, mx.zeros([b, 1, s, 0]))


def _make_patched_attention_call(original_call):
    """Create a patched __call__ for DeepseekV32Attention.

    The patched version checks for _ic_is_full flag:
    - If absent or True: run the original indexer (Full layer)
    - If False: skip indexer, reuse cached topk_indices (Shared layer)
    """
    from mlx_lm.models.base import scaled_dot_product_attention

    def patched_call(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        # If this instance has no IndexCache flags, run original
        if not hasattr(self, "_ic_is_full"):
            return original_call(self, x, mask, cache)

        B, L, D = x.shape

        qr = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(qr)

        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)
        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(
            compressed_kv, [self.kv_lora_rank], axis=-1
        )
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(
            0, 2, 1, 3
        )
        kv_latent = self.kv_a_layernorm(compressed_kv)

        offset = cache[0].offset if cache is not None else 0
        q_pe = self.rope(q_pe, offset)
        k_pe = self.rope(k_pe, offset)

        kv_latent = mx.expand_dims(kv_latent, axis=1)

        if cache is not None:
            kv_latent, k_pe = cache[0].update_and_fetch(kv_latent, k_pe)
        else:
            cache = [None] * 2

        # --- IndexCache: conditionally skip indexer ---
        if self._ic_is_full:
            topk_indices = self.indexer(x, qr, mask, cache=cache[1])
            self._ic_model_ref._index_cache_state[
                "last_topk_indices"
            ] = topk_indices
        else:
            _update_indexer_cache_only(self.indexer, x, cache[1])
            topk_indices = self._ic_model_ref._index_cache_state[
                "last_topk_indices"
            ]
        # --- end IndexCache ---

        if topk_indices is not None:
            if L == 1:
                idx = topk_indices[:, :, 0, :, None]
                kv_latent = mx.take_along_axis(
                    kv_latent,
                    mx.broadcast_to(
                        idx, idx.shape[:-1] + (kv_latent.shape[-1],)
                    ),
                    axis=2,
                )
                k_pe = mx.take_along_axis(
                    k_pe,
                    mx.broadcast_to(idx, idx.shape[:-1] + (k_pe.shape[-1],)),
                    axis=2,
                )
                mask = None
            else:
                shape = list(topk_indices.shape)
                shape[-1] = kv_latent.shape[2]
                sparse_mask = mx.zeros(shape, dtype=mx.bool_)
                sparse_mask = mx.put_along_axis(
                    sparse_mask, topk_indices, mx.array(True), axis=-1
                )
                if mask is not None:
                    sparse_mask = sparse_mask & mask
                mask = sparse_mask

        # Ensure the indexer cache is evaluated even if unused
        if cache is not None and cache[0] is not None:
            cache[0].keys = mx.depends(
                cache[0].keys, (cache[1].keys, cache[1].values)
            )

        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(
                mask,
                pe_scores,
                mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
            )

        if L == 1:
            q_nope = self.embed_q(q_nope)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)

        output = scaled_dot_product_attention(
            q_nope, k, v, cache=cache, scale=self.scale, mask=pe_scores
        )
        if L == 1:
            output = self.unembed_out(output)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)

    return patched_call


def apply_index_cache(model: Any, index_cache_freq: int) -> bool:
    """Apply IndexCache monkey-patch to a loaded DSA model.

    Args:
        model: A loaded mlx-lm model instance.
        index_cache_freq: Every Nth layer keeps its indexer (Full).
            For example, freq=4 means 75% of indexers are skipped.

    Returns:
        True if the patch was applied, False if the model is not supported.
    """
    global _class_patch_applied

    model_type = _get_model_type(model)
    if model_type not in _SUPPORTED_MODEL_TYPES:
        logger.debug(
            f"IndexCache: model_type '{model_type}' not supported, skipping"
        )
        return False

    if index_cache_freq < 2:
        logger.warning(
            f"IndexCache: freq={index_cache_freq} < 2, no layers to skip"
        )
        return False

    # Get the inner model (model.model is DeepseekV32Model)
    inner_model = model.model

    # Determine layer count
    layers = inner_model.layers
    num_layers = len(layers)
    pattern = _build_layer_pattern(num_layers, index_cache_freq)

    full_count = sum(pattern)
    shared_count = num_layers - full_count
    logger.info(
        f"IndexCache: {num_layers} layers, freq={index_cache_freq}, "
        f"Full={full_count}, Shared={shared_count}"
    )

    # Attach shared state to the inner model
    inner_model._index_cache_state = {"last_topk_indices": None}

    # Set per-layer flags on each attention module
    for i, layer in enumerate(layers):
        if layer is None:
            continue
        attn = layer.self_attn
        attn._ic_is_full = pattern[i]
        attn._ic_model_ref = inner_model

    # Apply class-level monkey-patch (only once)
    if not _class_patch_applied:
        from mlx_lm.models.deepseek_v32 import (
            DeepseekV32Attention,
            DeepseekV32Model,
        )

        # Patch Attention.__call__
        original_attn_call = DeepseekV32Attention.__call__
        DeepseekV32Attention.__call__ = _make_patched_attention_call(
            original_attn_call
        )

        # Patch Model.__call__ to reset shared state each forward pass
        original_model_call = DeepseekV32Model.__call__

        def _patched_model_call(self, x, cache=None):
            if hasattr(self, "_index_cache_state"):
                self._index_cache_state["last_topk_indices"] = None
            return original_model_call(self, x, cache=cache)

        DeepseekV32Model.__call__ = _patched_model_call

        _class_patch_applied = True
        logger.info("IndexCache: class-level patches applied")

    return True
