# SPDX-License-Identifier: Apache-2.0
"""Target-model prefill workflow for SpecPrefill."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from ..request import Request
from .planning import SpecPrefillTargetPlan


@dataclass(frozen=True)
class SpecPrefillTargetPrefillResult:
    """Target cache and generation-kickoff token returned on success."""

    prompt_cache: list[Any]
    tokens_to_process: Sequence[int]
    static_prefix_cached_tokens: int = 0


class ExactPrefixCache(Protocol):
    """Exact target-prefix operations provided by the tiered prefix cache."""

    def restore_exact_prefix(
        self,
        request_id: str,
        tokens: list[int],
        *,
        promote_to_hot_cache: bool,
    ) -> list[Any] | None: ...

    def store_exact_prefix(
        self,
        request_id: str,
        tokens: list[int],
        cache_data: list[dict[str, Any]],
        model_cache_config: Any = ...,
    ) -> Any: ...


CheckAbort = Callable[[int], None]
ReportProgress = Callable[[int, int], None]
SyncAndClearCache = Callable[[], None]
# Issue #2177: helper signature mirroring Scheduler._extract_cache_states.
# Returns the per-layer extracted state dicts and an optional ModelCacheConfig.
ExtractCacheStates = Callable[[list[Any]], tuple[list[dict[str, Any]], Any | None]]


def run_specprefill_target_prefill(
    *,
    target_model: Any,
    request: Request,
    plan: SpecPrefillTargetPlan,
    all_tokens: Sequence[int],
    selected_indices: mx.array,
    prefill_step_size: int,
    stream: Any,
    check_abort: CheckAbort,
    report_system_progress: ReportProgress,
    report_sparse_progress: ReportProgress,
    sync_and_clear_cache: SyncAndClearCache,
    log: logging.Logger,
    # Issue #2177: opt-in static system/tool prefix reuse. When the cache,
    # tokens, and extraction callback are provided, the first request stores
    # the prefetched system state in the tiered cache; later requests restore
    # it and skip the target-model system-prefill loop.
    extract_cache_states: ExtractCacheStates | None = None,
    exact_prefix_cache: ExactPrefixCache | None = None,
    static_prefix_tokens: Sequence[int] | None = None,
    promote_static_prefix_to_hot_cache: bool = True,
) -> SpecPrefillTargetPrefillResult:
    """Prefill system and selected conversation tokens for one request."""
    prompt_cache = None
    sys_arr = None
    conversation_tokens = None
    selected = None
    selected_indices_list = None

    try:
        from ..patches.specprefill import (
            _find_attention_layers,
            _get_attn_module,
            _OffsetAdjustedRoPE,
            cleanup_rope,
            sparse_prefill,
        )

        # A stale _OffsetAdjustedRoPE can survive if a prior specprefill
        # request never reached its cleanup (#766). Restore the genuine rope
        # before any prefill so system KV is written at true positions.
        cleanup_rope(target_model)

        system_token_count = plan.system_token_count
        conversation_tokens = plan.conversation_tokens
        conversation_token_count = plan.conversation_token_count
        generation_kickoff_index = plan.generation_kickoff_index
        prefill_started_at = time.monotonic()
        prompt_cache = (
            request.prompt_cache
            if request.cached_tokens > 0 and request.prompt_cache is not None
            else None
        )

        # Issue #2177: try to restore the stable system/tool prefix from the
        # tiered prefix cache before prefilling. This mirrors how the
        # draft workflow (``draft.run_specprefill_draft_scoring``) uses
        # ``draft_prefix_cache.fetch_cache``/``reconstruct_cache`` to reuse the
        # draft-model prefix instead of re-scoring from scratch.
        static_prefix_cached_tokens = 0
        if exact_prefix_cache is not None and static_prefix_tokens:
            full_static_prefix_tokens = list(static_prefix_tokens)
            restored_cache = exact_prefix_cache.restore_exact_prefix(
                f"{request.request_id}:specprefill-static-restore",
                full_static_prefix_tokens,
                promote_to_hot_cache=promote_static_prefix_to_hot_cache,
            )
            if restored_cache is not None:
                prompt_cache = restored_cache
                static_prefix_cached_tokens = len(full_static_prefix_tokens)
                report_system_progress(system_token_count, system_token_count)
                log.info(
                    f"SpecPrefill: restored {static_prefix_cached_tokens} "
                    "static system-prefix tokens from tiered cache"
                )
        if prompt_cache is None:
            prompt_cache = make_prompt_cache(target_model)

        if system_token_count > 0 and static_prefix_cached_tokens == 0:
            sys_arr = mx.array(all_tokens[:system_token_count])
            system_processed = 0
            while sys_arr.size > prefill_step_size:
                check_abort(system_processed)
                report_system_progress(system_processed, system_token_count)
                with mx.stream(stream):
                    target_model(sys_arr[:prefill_step_size][None], cache=prompt_cache)
                    mx.eval([cache_layer.state for cache_layer in prompt_cache])
                    # Keep the next chunk view on the target-model stream.
                    sys_arr = sys_arr[prefill_step_size:]
                system_processed += prefill_step_size
                check_abort(system_processed)
                report_system_progress(system_processed, system_token_count)
                # Drain before clear to avoid the stream/cache race in #557.
                sync_and_clear_cache()
            if sys_arr.size > 0:
                check_abort(system_processed)
                final_system_token_count = int(sys_arr.size)
                report_system_progress(system_processed, system_token_count)
                with mx.stream(stream):
                    target_model(sys_arr[None], cache=prompt_cache)
                    mx.eval([cache_layer.state for cache_layer in prompt_cache])
                system_processed += final_system_token_count
                check_abort(system_processed)
                report_system_progress(system_processed, system_token_count)
            log.info(
                f"SpecPrefill: system prompt {system_token_count} tokens full prefill"
            )

            # Issue #2177: capture the just-computed system-prefix KV states so
            # the next request with the same system prefix can restore them
            # without re-running the target model. The store path mirrors the
            # draft workflow's ``store_cache`` call after scoring.
            if (
                exact_prefix_cache is not None
                and static_prefix_tokens
                and extract_cache_states
            ):
                try:
                    extracted_states, model_cache_config = extract_cache_states(
                        prompt_cache
                    )
                    if extracted_states:
                        exact_prefix_cache.store_exact_prefix(
                            f"{request.request_id}:specprefill-static-store",
                            list(static_prefix_tokens),
                            extracted_states,
                            model_cache_config=model_cache_config,
                        )
                except Exception as error:
                    log.debug(
                        f"SpecPrefill: static prefix tiered-cache store failed: {error}"
                    )
        selected = selected_indices
        # BatchGenerator processes the generation-kickoff token separately.
        if plan.remove_kickoff_index:
            selected_indices_list = selected.tolist()
            selected_indices_list.remove(generation_kickoff_index)
            selected = mx.array(sorted(selected_indices_list))

        with mx.stream(stream):
            sparse_prefill(
                target_model,
                conversation_tokens,
                selected,
                prompt_cache,
                step_size=prefill_step_size,
                position_offset=plan.position_offset,
                progress_callback=report_sparse_progress,
            )

        # sparse_prefill computes adjustment for selected conversation tokens.
        # Decrement to reserve BatchGenerator's separately processed kickoff position.
        for _, layer in _find_attention_layers(target_model):
            attention_module = _get_attn_module(layer)
            if (
                attention_module
                and hasattr(attention_module, "rope")
                and isinstance(attention_module.rope, _OffsetAdjustedRoPE)
            ):
                attention_module.rope._adjustment -= 1

        selected_token_count = int(selected.shape[0])
        prefill_seconds = time.monotonic() - prefill_started_at
        system_cache_summary = (
            f"{static_prefix_cached_tokens} static-cached"
            if static_prefix_cached_tokens > 0
            else f"{system_token_count} full"
        )
        log.info(
            f"SpecPrefill: sparse prefill {selected_token_count}/"
            f"{conversation_token_count} conv tokens in {prefill_seconds:.1f}s "
            f"(total {request.num_prompt_tokens}, cached {request.cached_tokens}, "
            f"system {system_cache_summary}, conv {conversation_token_count} sparse)"
        )
        return SpecPrefillTargetPrefillResult(
            prompt_cache=prompt_cache,
            tokens_to_process=all_tokens[-1:],
            static_prefix_cached_tokens=static_prefix_cached_tokens,
        )
    except Exception:
        prompt_cache = None
        sys_arr = None
        conversation_tokens = None
        selected_indices = None
        selected_indices_list = None
        selected = None
        raise
