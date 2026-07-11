# SPDX-License-Identifier: Apache-2.0
"""Draft-model scoring workflow for SpecPrefill."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import mlx.core as mx

from ..prefill_progress import get_prefill_tracker
from ..request import Request
from .policy import SpecPrefillScoringPlan


class DraftPrefixCache(Protocol):
    """Draft-cache operations used by the scoring workflow."""

    def fetch_cache(
        self, request_id: str, tokens: Sequence[int]
    ) -> tuple[Any, list[int]]: ...

    def preload_blocks(self, block_table: Any) -> int: ...

    def reconstruct_cache(self, block_table: Any) -> Any: ...

    def store_cache(
        self,
        request_id: str,
        tokens: Sequence[int],
        cache_data: list[Any],
        model_cache_config: Any = ...,
    ) -> Any: ...


ExtractCacheStates = Callable[[list[Any]], tuple[list[dict[str, Any]], Any | None]]
SyncAndClearCache = Callable[[], None]


def run_specprefill_draft_scoring(
    *,
    request: Request,
    plan: SpecPrefillScoringPlan,
    draft_model: Any,
    draft_prefix_cache: DraftPrefixCache | None,
    model_id: str,
    prefill_step_size: int,
    stream: Any,
    extract_cache_states: ExtractCacheStates,
    sync_and_clear_cache: SyncAndClearCache,
    log: logging.Logger,
) -> None:
    """Score draft tokens, persist reusable cache state, and update ``request``."""
    tokens_to_score = plan.tokens_to_score
    n_to_score = plan.n_to_score
    tracker = get_prefill_tracker()

    try:
        # Resolve patch helpers at call time so engine-specific patches apply.
        from ..patches.specprefill import score_tokens, select_chunks

        draft_cache = None
        draft_cached_tokens = 0
        if draft_prefix_cache is not None:
            try:
                block_table, _draft_remaining = draft_prefix_cache.fetch_cache(
                    request.request_id, tokens_to_score
                )
                if block_table and block_table.num_tokens > 0:
                    draft_prefix_cache.preload_blocks(block_table)
                    reconstructed_cache = draft_prefix_cache.reconstruct_cache(
                        block_table
                    )
                    if reconstructed_cache:
                        draft_cache = reconstructed_cache
                        draft_cached_tokens = block_table.num_tokens
            except Exception as error:
                log.debug(f"SpecPrefill: draft cache fetch failed: {error}")

        tracker_extra = {
            "prompt_tokens": request.num_prompt_tokens,
            "system_tokens": request.specprefill_system_end,
            "conversation_tokens": request.num_prompt_tokens
            - request.specprefill_system_end,
            "cached_tokens": request.cached_tokens,
        }

        def report_score_progress(
            processed: int, total: int, phase: str
        ) -> None:
            tracker.update(
                request.request_id,
                min(processed, total - 1),
                total,
                model_id,
                phase=f"specprefill_{phase}",
                detail="scoring draft tokens",
                extra=tracker_extra,
            )

        tracker.update(
            request.request_id,
            0,
            n_to_score,
            model_id,
            phase="specprefill_scoring",
            detail="scoring draft tokens",
            extra=tracker_extra,
        )
        scoring_started_at = time.monotonic()
        with mx.stream(stream):
            importance, used_cache = score_tokens(
                draft_model,
                tokens_to_score,
                prefill_step_size=prefill_step_size,
                existing_cache=draft_cache,
                progress_callback=report_score_progress,
            )
            # Keep the lazy selection ops on the scoring stream (#2183, #2197).
            selected_indices = select_chunks(importance, keep_pct=plan.keep_pct)
        scoring_seconds = time.monotonic() - scoring_started_at

        selected_token_count = selected_indices.shape[0]
        request.specprefill_indices = selected_indices
        request.specprefill_total_tokens = n_to_score
        request.specprefill_position_offset = (
            request.cached_tokens + plan.effective_system
        )
        # Scheduler-owned dynamic field.
        request._specprefill_system_tokens = plan.effective_system  # type: ignore[attr-defined]

        log_extra = []
        if draft_cached_tokens > 0:
            log_extra.append(f"draft cache hit {draft_cached_tokens}")
        log_extra.append(
            f"prompt {request.num_prompt_tokens} = system "
            f"{request.specprefill_system_end} + conv "
            f"{request.num_prompt_tokens - request.specprefill_system_end}, "
            f"cached {request.cached_tokens}"
        )
        tracker.update(
            request.request_id,
            n_to_score - 1,
            n_to_score,
            model_id,
            phase="specprefill_selected",
            detail="selected sparse tokens",
            extra={
                **tracker_extra,
                "scored_tokens": n_to_score,
                "selected_tokens": selected_token_count,
                "keep_percent": round(selected_token_count / n_to_score * 100),
            },
        )
        log.info(
            f"SpecPrefill: scored {n_to_score} tokens in {scoring_seconds:.1f}s, "
            f"selected {selected_token_count}/{n_to_score} "
            f"(keep={selected_token_count / n_to_score * 100:.0f}%, "
            f"{', '.join(log_extra)})"
        )

        if draft_prefix_cache is not None and used_cache is not None:
            try:
                extracted_cache, model_cache_config = extract_cache_states(used_cache)
                if extracted_cache:
                    draft_prefix_cache.store_cache(
                        request.request_id,
                        tokens_to_score,
                        extracted_cache,
                        model_cache_config=model_cache_config,
                    )
            except Exception as error:
                log.debug(f"SpecPrefill: draft cache store failed: {error}")

        # Drain the stream before releasing Metal buffers to avoid #557.
        del used_cache
        sync_and_clear_cache()
        tracker.update(request.request_id, n_to_score, n_to_score, model_id)
    except Exception as error:
        log.error(f"SpecPrefill scoring failed, falling back to normal path: {error}")
        request.specprefill_indices = None
        tracker.remove(request.request_id)
