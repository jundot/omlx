# SPDX-License-Identifier: Apache-2.0
"""Tests for the SpecPrefill draft-scoring workflow."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import mlx.core as mx

import omlx.specprefill.draft as draft_workflow
from omlx.request import Request, SamplingParams
from omlx.specprefill.policy import plan_specprefill_scoring


class _Logger:
    def __init__(self) -> None:
        self.debug_messages: list[str] = []
        self.info_messages: list[str] = []
        self.error_messages: list[str] = []

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.debug_messages.append(message)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.info_messages.append(message)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.error_messages.append(message)


class _Tracker:
    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []
        self.removed: list[str] = []

    def update(
        self,
        request_id: str,
        processed: int,
        total: int,
        model_id: str,
        phase: str = "prefill",
        detail: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.updates.append(
            {
                "request_id": request_id,
                "processed": processed,
                "total": total,
                "model_id": model_id,
                "phase": phase,
                "detail": detail,
                "extra": extra,
            }
        )

    def remove(self, request_id: str) -> None:
        self.removed.append(request_id)


class _DraftCache:
    def __init__(
        self,
        block_table: Any = None,
        reconstructed_cache: Any = None,
        fetch_error: Exception | None = None,
    ) -> None:
        self.block_table = block_table
        self.reconstructed_cache = reconstructed_cache
        self.fetch_error = fetch_error
        self.fetches: list[tuple[str, list[int]]] = []
        self.preloads: list[Any] = []
        self.reconstructions: list[Any] = []
        self.stores: list[tuple[str, list[int], list[Any], Any]] = []

    def fetch_cache(self, request_id: str, tokens: list[int]) -> tuple[Any, list[int]]:
        self.fetches.append((request_id, list(tokens)))
        if self.fetch_error is not None:
            raise self.fetch_error
        return self.block_table, []

    def preload_blocks(self, block_table: Any) -> int:
        self.preloads.append(block_table)
        return block_table.num_tokens

    def reconstruct_cache(self, block_table: Any) -> Any:
        self.reconstructions.append(block_table)
        return self.reconstructed_cache

    def store_cache(
        self,
        request_id: str,
        tokens: list[int],
        cache_data: list[Any],
        model_cache_config: Any = None,
    ) -> None:
        self.stores.append(
            (request_id, list(tokens), cache_data, model_cache_config)
        )


def _request_and_plan() -> tuple[Request, Any]:
    request = Request(
        request_id="request-1",
        prompt=list(range(20)),
        sampling_params=SamplingParams(),
    )
    request.prompt_token_ids = list(range(20))
    request.num_prompt_tokens = 20
    request.remaining_tokens = request.prompt_token_ids
    request.specprefill_system_end = 4
    request.cached_tokens = 0
    plan = plan_specprefill_scoring(
        remaining_tokens=request.remaining_tokens,
        system_prompt_end=request.specprefill_system_end,
        cached_tokens=request.cached_tokens,
        requested_threshold=None,
        requested_keep_pct=None,
        default_threshold=8,
        default_keep_pct=0.2,
    )
    assert plan is not None
    return request, plan


def _run(
    request: Request,
    plan: Any,
    *,
    draft_cache: _DraftCache | None = None,
    score_tokens: Callable[..., Any] | None = None,
    extract_cache_states: Callable[[list[Any]], tuple[list[dict[str, Any]], Any]] | None = None,
) -> tuple[_Tracker, _Logger, dict[str, Any]]:
    tracker = _Tracker()
    logger = _Logger()
    selected_indices = mx.arange(3)
    stream = object()
    trace: dict[str, Any] = {"streams": [], "syncs": [], "score_calls": []}

    def default_score_tokens(
        model: Any, tokens: list[int], **kwargs: Any
    ) -> tuple[Any, list[str]]:
        trace["score_calls"].append(kwargs)
        return mx.zeros(plan.n_to_score), ["draft-cache"]

    def select_chunks(importance: Any, keep_pct: float) -> Any:
        return selected_indices

    def use_stream(selected_stream: Any):
        trace["streams"].append(selected_stream)
        return nullcontext()

    with (
        patch.object(draft_workflow, "get_prefill_tracker", return_value=tracker),
        patch(
            "omlx.patches.specprefill.score_tokens",
            side_effect=score_tokens or default_score_tokens,
        ),
        patch("omlx.patches.specprefill.select_chunks", side_effect=select_chunks),
        patch.object(draft_workflow.mx, "stream", side_effect=use_stream),
    ):
        draft_workflow.run_specprefill_draft_scoring(
            request=request,
            plan=plan,
            draft_model=object(),
            draft_prefix_cache=draft_cache,
            model_id="model-id",
            prefill_step_size=4,
            stream=stream,
            extract_cache_states=extract_cache_states or (lambda cache: ([], None)),
            sync_and_clear_cache=lambda: trace["syncs"].append(stream),
            log=logger,
        )
    trace["selected_indices"] = selected_indices
    trace["stream"] = stream
    return tracker, logger, trace


def test_success_updates_request_tracker_logger_and_stream():
    request, plan = _request_and_plan()

    tracker, logger, trace = _run(request, plan)

    assert request.specprefill_indices is trace["selected_indices"]
    assert request.specprefill_total_tokens == plan.n_to_score
    assert request.specprefill_position_offset == plan.effective_system
    assert request._specprefill_system_tokens == plan.effective_system
    assert [update["phase"] for update in tracker.updates] == [
        "specprefill_scoring",
        "specprefill_selected",
        "prefill",
    ]
    assert tracker.updates[-1]["processed"] == plan.n_to_score
    assert tracker.removed == []
    assert trace["streams"] == [trace["stream"]]
    assert trace["syncs"] == [trace["stream"]]
    assert logger.info_messages[0].startswith("SpecPrefill: scored")


def test_reconstructed_cache_is_scored_and_stored():
    request, plan = _request_and_plan()
    block_table = SimpleNamespace(num_tokens=3)
    reconstructed_cache = ["reconstructed"]
    draft_cache = _DraftCache(block_table, reconstructed_cache)
    model_cache_config = object()

    def extract_cache_states(cache: list[Any]) -> tuple[list[dict[str, Any]], Any]:
        assert cache == ["draft-cache"]
        return [{"state": "value"}], model_cache_config

    _, _, trace = _run(
        request,
        plan,
        draft_cache=draft_cache,
        extract_cache_states=extract_cache_states,
    )

    assert trace["score_calls"][0]["existing_cache"] is reconstructed_cache
    assert draft_cache.fetches == [(request.request_id, list(plan.tokens_to_score))]
    assert draft_cache.preloads == [block_table]
    assert draft_cache.reconstructions == [block_table]
    assert draft_cache.stores == [
        (
            request.request_id,
            list(plan.tokens_to_score),
            [{"state": "value"}],
            model_cache_config,
        )
    ]


def test_cache_fetch_error_falls_back_to_uncached_scoring():
    request, plan = _request_and_plan()
    draft_cache = _DraftCache(fetch_error=RuntimeError("disk gone"))

    _, logger, trace = _run(request, plan, draft_cache=draft_cache)

    assert trace["score_calls"][0]["existing_cache"] is None
    assert any("draft cache fetch failed: disk gone" in message for message in logger.debug_messages)


def test_scoring_error_clears_request_and_tracker():
    request, plan = _request_and_plan()

    def fail_scoring(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("boom")

    tracker, logger, _ = _run(request, plan, score_tokens=fail_scoring)

    assert request.specprefill_indices is None
    assert tracker.removed == [request.request_id]
    assert logger.error_messages == [
        "SpecPrefill scoring failed, falling back to normal path: boom"
    ]
