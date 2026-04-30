# SPDX-License-Identifier: Apache-2.0
"""
VLM (Vision-Language Model) adapter for BatchGenerator integration.

This module provides VLMModelAdapter, a wrapper around mlx-vlm's model
that presents a standard model interface compatible with mlx-lm's
BatchGenerator. The adapter handles vision embedding injection during
prefill while allowing standard token-ID-based decode.

Architecture:
    VLMModelAdapter wraps the VLM's language_model, intercepting calls
    during prefill to substitute token IDs with pre-computed vision+text
    embeddings. After prefill, the adapter becomes transparent, passing
    token IDs directly to language_model for autoregressive decode.

    The vision encoder runs ONCE before BatchGenerator.insert(), and the
    resulting embeddings are registered via set_pending_embeddings().
    During chunked prefill, the adapter slices embeddings to match the
    chunk size requested by BatchGenerator.
"""

import logging
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from ..patches.qwen3_5_attention import force_text_only_rope

logger = logging.getLogger(__name__)


class VLMModelAdapter(nn.Module):
    """
    Adapter wrapping a VLM's language_model for BatchGenerator compatibility.

    The BatchGenerator calls self.model(input_ids, cache=...) during prefill
    and decode. For VLM requests with images, this adapter substitutes
    pre-computed input_embeds during prefill. After prefill completes,
    decode uses standard token IDs (vision context is in KV cache).

    Thread safety:
        The pending embeddings are set by Scheduler._schedule_waiting()
        and consumed by BatchGenerator._process_prompts(), both running
        in the same thread (single ThreadPoolExecutor worker).

    Attributes:
        _vlm_model: The full VLM model (vision_tower + language_model + projector)
        _pending_embeds: Pre-computed embeddings for the next prefill
        _pending_kwargs: Extra model-specific kwargs (e.g., position_ids)
        _embed_offset: Current chunk offset during chunked prefill
    """

    def __init__(self, vlm_model: nn.Module):
        super().__init__()
        self._vlm_model = vlm_model
        self._language_model = vlm_model.language_model
        self._uses_mrope = self._detect_mrope(vlm_model)

        # Pending vision embeddings state (set before prefill, cleared after)
        self._pending_embeds: Optional[mx.array] = None
        self._pending_kwargs: Dict[str, Any] = {}
        self._embed_offset: int = 0

        # Per-request mRoPE state: UID → rope_delta mapping.
        # Populated by scheduler after VLM prefill, consumed during decode.
        # The _patched_generation_batch_step builds _batch_rope_deltas
        # from this dict + current batch UIDs before each step.
        self._uid_rope_deltas: Dict[int, float] = {}
        self._batch_rope_deltas: Optional[mx.array] = None

    @property
    def layers(self):
        """Expose language model layers for cache creation."""
        return self._language_model.model.layers

    @property
    def model_type(self) -> str:
        """Expose model_type for config access."""
        if hasattr(self._vlm_model, "config") and hasattr(self._vlm_model.config, "model_type"):
            return self._vlm_model.config.model_type
        return "vlm"

    @property
    def config(self):
        """Expose model config."""
        return self._vlm_model.config

    @property
    def args(self):
        """Expose model args (alias for config, used by some mlx-lm code)."""
        if hasattr(self._language_model, "args"):
            return self._language_model.args
        return self.config

    def make_cache(self) -> List[Any]:
        """
        Create KV cache using the language model's make_cache().

        Returns the same cache types (KVCache, RotatingKVCache, ArraysCache, etc.)
        as if the language model were used directly. Falls back to default KVCache
        per layer if the language model doesn't define make_cache().
        """
        if hasattr(self._language_model, "make_cache"):
            return self._language_model.make_cache()
        from mlx_lm.models.cache import KVCache
        return [KVCache() for _ in range(len(self.layers))]

    def set_pending_embeddings(
        self,
        inputs_embeds: mx.array,
        extra_kwargs: Optional[Dict[str, Any]] = None,
        start_offset: int = 0,
    ) -> None:
        """
        Register pre-computed vision+text embeddings for the next prefill.

        Must be called before BatchGenerator.insert() for VLM requests.
        The embeddings will be consumed during the subsequent prefill
        and automatically cleared when prefill completes.

        Args:
            inputs_embeds: Merged vision+text embeddings, shape (1, seq_len, hidden_dim)
            extra_kwargs: Model-specific kwargs (e.g., for Gemma3: attention_mask_4d)
            start_offset: Initial offset into embeddings (for cache-hit requests
                          where the first ``start_offset`` tokens are already cached)
        """
        self._pending_embeds = inputs_embeds
        self._pending_kwargs = extra_kwargs or {}
        self._embed_offset = start_offset

    def clear_pending_embeddings(self) -> None:
        """Explicitly clear pending embeddings (called after prefill or on abort)."""
        self._pending_embeds = None
        self._pending_kwargs = {}
        self._embed_offset = 0

    @staticmethod
    def _detect_mrope(vlm_model) -> bool:
        """Check if VLM model uses multi-dimensional RoPE (mRoPE)."""
        config = getattr(vlm_model, "config", None)
        if config is None:
            return False
        text_config = getattr(config, "text_config", None)
        if text_config is None:
            return False
        rope_cfg = getattr(text_config, "rope_scaling", None) or getattr(
            text_config, "rope_parameters", None
        )
        if isinstance(rope_cfg, dict):
            return "mrope_section" in rope_cfg
        return False

    def clear_vlm_position_state(self) -> None:
        """Clear stale mRoPE position state from previous VLM requests.

        Must be called before text-only request prefill to prevent
        position contamination from prior VLM requests. The language model
        stores ``_position_ids`` and ``_rope_deltas`` as instance variables
        during ``get_input_embeddings()``; these persist across requests
        and cause wrong position computation for text-only prompts.

        Always sets both attributes unconditionally because they may not
        exist yet (only created on first ``get_input_embeddings()`` call),
        but the LanguageModel.__call__() accesses them without hasattr.
        """
        self._language_model._position_ids = None
        self._language_model._rope_deltas = None
        self._batch_rope_deltas = None

    def register_rope_delta(self, uid: int, delta: float) -> None:
        """Register rope_delta for a UID after VLM prefill."""
        self._uid_rope_deltas[uid] = delta

    def unregister_rope_delta(self, uid: int) -> None:
        """Remove rope_delta for a finished/aborted UID."""
        self._uid_rope_deltas.pop(uid, None)

    def set_batch_rope_deltas(self, deltas: mx.array) -> None:
        """Set per-request rope_deltas for the current decode batch.

        Called by _patched_generation_batch_step before each step with
        an array of rope_deltas aligned to the batch slot order.
        """
        self._batch_rope_deltas = deltas

    def get_last_rope_deltas(self) -> float:
        """Extract rope_deltas from language model after VLM prefill.

        Should be called after get_input_embeddings() which sets
        ``_rope_deltas`` on the language model.
        """
        rd = getattr(self._language_model, "_rope_deltas", None)
        if rd is None:
            return 0.0
        if hasattr(rd, "item"):
            return float(rd.item())
        return float(rd)

    @property
    def has_pending_embeddings(self) -> bool:
        """Check if there are pending embeddings for prefill."""
        return self._pending_embeds is not None

    def __call__(
        self,
        input_ids: mx.array,
        cache: Optional[List[Any]] = None,
        **kwargs,
    ) -> Any:
        """
        Forward pass, dispatching between VLM prefill and standard decode.

        Supports three paths:
        1. Batched VLM: ``inputs_embeds`` kwarg from _process_prompts()
        2. Legacy single VLM: ``_pending_embeds`` set via set_pending_embeddings()
        3. Standard decode: token IDs only

        Args:
            input_ids: Token IDs, shape (batch, seq_len)
            cache: KV cache list
            **kwargs: Additional kwargs from BatchGenerator.
                inputs_embeds: Pre-computed embeddings for batched VLM prefill
                vlm_extra_kwargs: Model-specific kwargs (e.g., position_ids)

        Returns:
            Model output (logits as mx.array)
        """
        inputs_embeds = kwargs.pop("inputs_embeds", None)
        vlm_extra = kwargs.pop("vlm_extra_kwargs", None) or {}
        vlm_extra.pop("_captured_rope_deltas", None)

        if inputs_embeds is not None:
            result = self._language_model(
                input_ids,
                inputs_embeds=inputs_embeds,
                cache=cache,
                **vlm_extra,
                **kwargs,
            )
        elif self._pending_embeds is not None:
            result = self._forward_with_embeddings(input_ids, cache, **kwargs)
        else:
            if self._uses_mrope and self._batch_rope_deltas is not None and cache is not None:
                offsets = None
                for c in cache:
                    if hasattr(c, "offset"):
                        offsets = c.offset
                        break
                B, L = input_ids.shape
                deltas = self._batch_rope_deltas
                if (offsets is not None and isinstance(offsets, mx.array)
                        and deltas.size == B):
                    positions = offsets + deltas
                    position_ids = mx.broadcast_to(
                        positions[None, :, None], (3, B, L)
                    )
                    # Decode never adds new image tokens; the broadcast
                    # collapses the 3 mRoPE sections to identical values,
                    # so the patched Qwen3_5Attention can skip its per-layer
                    # .item() probes and take the plain-RoPE branch directly.
                    with force_text_only_rope():
                        result = self._language_model(
                            input_ids, cache=cache, position_ids=position_ids, **kwargs
                        )
                else:
                    result = self._language_model(
                        input_ids, cache=cache, **kwargs
                    )
            elif self._uses_mrope and cache is not None:
                offsets = None
                for c in cache:
                    if hasattr(c, "offset") and isinstance(c.offset, mx.array) and c.offset.ndim > 0:
                        offsets = c.offset
                        break
                if offsets is not None:
                    B, L = input_ids.shape
                    position_ids = mx.broadcast_to(
                        offsets[None, :, None], (3, B, L)
                    )
                    with force_text_only_rope():
                        result = self._language_model(
                            input_ids, cache=cache, position_ids=position_ids, **kwargs
                        )
                else:
                    result = self._language_model(
                        input_ids, cache=cache, **kwargs
                    )
            else:
                if hasattr(self._vlm_model, "_set_position_state"):
                    self._vlm_model._set_position_state(input_ids)
                result = self._language_model(input_ids, cache=cache, **kwargs)

        if hasattr(result, "logits"):
            return result.logits
        return result

    def _forward_with_embeddings(
        self,
        input_ids: mx.array,
        cache: Optional[List[Any]] = None,
        **kwargs,
    ) -> Any:
        """Forward pass with pre-computed vision embeddings (prefill phase)."""
        chunk_len = input_ids.shape[1]
        total_len = self._pending_embeds.shape[1]

        end_offset = min(self._embed_offset + chunk_len, total_len)
        chunk_embeds = self._pending_embeds[:, self._embed_offset:end_offset, :]

        result = self._language_model(
            input_ids,
            inputs_embeds=chunk_embeds,
            cache=cache,
            **self._pending_kwargs,
            **kwargs,
        )

        self._embed_offset = end_offset

        if self._embed_offset >= total_len - 1:
            self.clear_pending_embeddings()

        return result

    def get_input_embeddings(self, input_ids: mx.array, pixel_values: Optional[mx.array] = None, **kwargs) -> Any:
        """
        Compute vision+text merged embeddings.

        Delegates to the VLM model's get_input_embeddings(), which runs
        the vision encoder and merges image features with text embeddings.

        Args:
            input_ids: Token IDs with image placeholders
            pixel_values: Preprocessed image tensors
            **kwargs: Model-specific kwargs (e.g., image_grid_thw)

        Returns:
            InputEmbeddingsFeatures with inputs_embeds and optional extra data
        """
        return self._vlm_model.get_input_embeddings(input_ids, pixel_values, **kwargs)
