"""Request admission for verified cold text prefill capabilities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .capabilities import inspect_prefill_model


@dataclass(frozen=True)
class PrefillEligibility:
    """Batching support and a stable reason for fallback diagnostics."""

    eligible: bool
    reason: str


def prefill_eligibility(
    model: Any,
    request: Any,
    caches: Sequence[Any] | None,
    *,
    turboquant_enabled: bool = False,
    speculative_enabled: bool = False,
    distributed_enabled: bool = False,
    snapshots_required: bool = False,
) -> PrefillEligibility:
    """Check the actual cache after prefix lookup, without changing any state.

    Capabilities describe model execution; request features and scheduler-owned
    snapshot requirements can still veto it. Unsupported combinations retain
    the normal single-request path rather than disabling caching or media.
    """
    from mlx_lm.models.cache import KVCache

    capabilities, reason = inspect_prefill_model(model)
    if capabilities is None:
        return PrefillEligibility(False, reason)
    if distributed_enabled:
        return PrefillEligibility(False, "distributed_execution")
    if getattr(request, "rope_deltas", 0.0) != 0:
        return PrefillEligibility(False, "unsupported_position_state")
    if (
        getattr(request, "vlm_inputs_embeds", None) is not None
        or getattr(request, "vlm_extra_kwargs", None)
        or getattr(request, "images", None)
        or getattr(request, "videos", None)
    ):
        return PrefillEligibility(False, "multimodal_request")
    if (
        speculative_enabled
        or getattr(request, "specprefill_indices", None) is not None
        or getattr(request, "_specprefill_enabled", False)
    ):
        return PrefillEligibility(False, "speculative_prefill")
    if turboquant_enabled:
        return PrefillEligibility(False, "quantized_cache")
    if getattr(request, "cached_tokens", 0) != 0:
        return PrefillEligibility(False, "prefix_cache")
    if snapshots_required:
        return PrefillEligibility(False, "boundary_snapshots")
    remaining_tokens = getattr(request, "remaining_tokens", None)
    if remaining_tokens is None:
        remaining_tokens = getattr(request, "prompt_token_ids", None)
    if remaining_tokens is None or len(remaining_tokens) <= 1:
        return PrefillEligibility(False, "no_prefill_tokens")
    if not caches or len(caches) != len(capabilities.cache_types):
        return PrefillEligibility(False, "unsupported_cache")
    for cache, cache_type in zip(caches, capabilities.cache_types):
        if type(cache) is not cache_type:
            return PrefillEligibility(False, "unsupported_cache")
        if cache_type is KVCache and (
            cache.offset != 0 or cache.keys is not None or cache.values is not None
        ):
            return PrefillEligibility(False, "prefix_cache")
    return PrefillEligibility(True, "supported")
