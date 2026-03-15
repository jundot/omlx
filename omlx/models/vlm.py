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

logger = logging.getLogger(__name__)


class _IntOffsetCacheProxy:
    """Proxy that converts mx.array cache.offset to Python int.

    mlx-lm's BatchKVCache stores ``offset`` as ``mx.array`` for efficient
    batched updates, but mlx-vlm models (e.g., Qwen3.5) use ``cache.offset``
    as a slice index which requires a Python int.  This proxy transparently
    converts on read while leaving internal cache operations (which access
    ``self.offset`` on the real object) unaffected.
    """

    __slots__ = ("_cache",)

    def __init__(self, cache: Any):
        object.__setattr__(self, "_cache", cache)

    @property
    def offset(self):
        raw = self._cache.offset
        if isinstance(raw, mx.array):
            # BatchKVCache: return _idx (buffer write index) instead of
            # per-element offset[0].  _idx matches the value used by
            # make_mask() to compute the attention mask's kv dimension.
            # Using offset[0] causes mask truncation when left_padding[0] > 0
            # because offset[i] = _idx - left_padding[i].
            if hasattr(self._cache, "_idx"):
                return self._cache._idx
            if raw.ndim == 0:
                return int(raw.item())
            return int(raw.reshape(-1)[0].item())
        return raw

    def __getattr__(self, name: str):
        return getattr(self._cache, name)

    def __setattr__(self, name: str, value: Any):
        if name == "_cache":
            object.__setattr__(self, name, value)
        else:
            setattr(self._cache, name, value)

    def __getitem__(self, key):
        return self._cache[key]

    def __setitem__(self, key, value):
        self._cache[key] = value

    def __bool__(self):
        return True

    def __iter__(self):
        return iter(self._cache)


def _wrap_caches(cache_list: Optional[List[Any]]) -> Optional[List[Any]]:
    """Wrap batch cache objects with int-offset proxies for mlx-vlm compatibility."""
    if cache_list is None:
        return None
    return [_IntOffsetCacheProxy(c) for c in cache_list]


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
        """
        Initialize the adapter.

        Args:
            vlm_model: The full VLM model loaded by mlx_vlm.utils.load()
        """
        super().__init__()
        self._vlm_model = vlm_model
        self._language_model = vlm_model.language_model

        # Pending vision embeddings state (set before prefill, cleared after)
        self._pending_embeds: Optional[mx.array] = None
        self._pending_kwargs: Dict[str, Any] = {}
        self._embed_offset: int = 0

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
        # Fallback: default KVCache for each layer (matches mlx-lm's make_prompt_cache)
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
        wrapped_cache = _wrap_caches(cache)
        inputs_embeds = kwargs.pop("inputs_embeds", None)
        vlm_extra = kwargs.pop("vlm_extra_kwargs", None) or {}

        if inputs_embeds is not None:
            # Batched VLM path: embeddings from _process_prompts
            result = self._language_model(
                input_ids,
                inputs_embeds=inputs_embeds,
                cache=wrapped_cache,
                **vlm_extra,
                **kwargs,
            )
        elif self._pending_embeds is not None:
            # Legacy single-request path
            result = self._forward_with_embeddings(input_ids, wrapped_cache, **kwargs)
        else:
            # Standard decode path: token IDs only.
            # VLM models that reuse another architecture's LanguageModel
            # (e.g., MiniCPM-o using qwen3_vl) rely on position state being
            # initialized by the full Model.__call__(). Since omlx calls the
            # language model directly, replicate that initialization here.
            if hasattr(self._vlm_model, "_set_position_state"):
                self._vlm_model._set_position_state(input_ids)
            result = self._language_model(input_ids, cache=wrapped_cache, **kwargs)

        # mlx-vlm models return LanguageModelOutput(logits=...) but
        # mlx-lm's BatchGenerator expects raw mx.array logits.
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

        # Slice embeddings for this chunk
        end_offset = min(self._embed_offset + chunk_len, total_len)
        chunk_embeds = self._pending_embeds[:, self._embed_offset:end_offset, :]

        # If chunk_embeds is shorter than input_ids (last chunk edge case),
        # the language model handles the size mismatch via inputs_embeds taking priority
        result = self._language_model(
            input_ids,
            inputs_embeds=chunk_embeds,
            cache=cache,
            **self._pending_kwargs,
            **kwargs,
        )

        self._embed_offset = end_offset

        # Check if prefill is complete (only 1 token remaining = the last token
        # that gets processed in _step, not in _process_prompts)
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
