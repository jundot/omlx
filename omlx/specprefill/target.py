# SPDX-License-Identifier: Apache-2.0
"""Target-model prefill workflow for SpecPrefill."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from ..request import Request
from .planning import SpecPrefillTargetPlan


@dataclass(frozen=True)
class SpecPrefillTargetPrefillResult:
    """Target cache and generation-kickoff token returned on success."""

    prompt_cache: list[Any]
    tokens_to_process: Sequence[int]


CheckAbort = Callable[[int], None]
ReportProgress = Callable[[int, int], None]
SyncAndClearCache = Callable[[], None]


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
        prompt_cache = make_prompt_cache(target_model)

        if system_token_count > 0:
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
        log.info(
            f"SpecPrefill: sparse prefill {selected_token_count}/"
            f"{conversation_token_count} conv tokens in {prefill_seconds:.1f}s "
            f"(total {request.num_prompt_tokens}, cached {request.cached_tokens}, "
            f"system {system_token_count} full, conv {conversation_token_count} sparse)"
        )
        return SpecPrefillTargetPrefillResult(
            prompt_cache=prompt_cache,
            tokens_to_process=all_tokens[-1:],
        )
    except Exception:
        prompt_cache = None
        sys_arr = None
        conversation_tokens = None
        selected_indices = None
        selected_indices_list = None
        selected = None
        raise
