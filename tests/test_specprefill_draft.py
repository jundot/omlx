# SPDX-License-Identifier: Apache-2.0
"""Tests for the SpecPrefill draft-scoring workflow."""

from __future__ import annotations

import gc
import weakref
from collections.abc import Callable
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import mlx.core as mx
import pytest

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
        block_size: int = 1024,
    ) -> None:
        self.block_table = block_table
        self.reconstructed_cache = reconstructed_cache
        self.fetch_error = fetch_error
        self.block_size = block_size
        self.fetches: list[tuple[str, list[int]]] = []
        self.preloads: list[Any] = []
        self.reconstructions: list[Any] = []
        self.stores: list[tuple[str, list[int], list[Any], Any]] = []
        self.store_boundary_snapshots: list[Any] = []

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
        boundary_snapshots: dict[int, list[Any]] | None = None,
    ) -> None:
        self.stores.append((request_id, list(tokens), cache_data, model_cache_config))
        self.store_boundary_snapshots.append(boundary_snapshots)


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
    draft_model: Any = None,
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
            draft_model=draft_model if draft_model is not None else object(),
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
    # The lookup leaves the last token out; the store still covers all of it.
    assert draft_cache.fetches == [
        (request.request_id, list(plan.tokens_to_score[:-1]))
    ]
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

    assert any("draft cache fetch failed: disk gone" in message for message in logger.debug_messages)
    # A fetch failure leaves nothing to restore. The empty cache allocated in
    # its place is what score_tokens would have made anyway; what matters is
    # that scoring is not handed the failed fetch's remains.
    assert trace["score_calls"][0]["existing_cache"] in (None, [])


def test_a_miss_still_allocates_the_cache_here():
    """The allocation has to happen here, or there is no boundary to capture.

    score_tokens hands its own cache back only after scoring. This test
    pins the succeeding branch: the earlier tests all passed a draft_model
    that make_prompt_cache rejects, so they exercised only the failure.
    """
    request, plan = _request_and_plan()
    allocated = [_RecurrentLayer()]

    with patch.object(
        draft_workflow, "make_prompt_cache", return_value=allocated
    ) as made:
        _, _, trace = _run(request, plan, draft_cache=_DraftCache())

    assert made.called
    assert trace["score_calls"][0]["existing_cache"] is allocated


def test_allocation_failure_is_survivable():
    """A model make_prompt_cache cannot read must not fail the request.

    Losing the allocation costs this request its boundary capture, which is
    the behaviour before this change, not a failure.
    """
    request, plan = _request_and_plan()

    with patch.object(
        draft_workflow, "make_prompt_cache", side_effect=RuntimeError("no layers")
    ):
        _, logger, trace = _run(request, plan, draft_cache=_DraftCache())

    assert trace["score_calls"][0]["existing_cache"] is None
    assert any(
        "draft cache preallocation failed: no layers" in message
        for message in logger.debug_messages
    )


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


class _RecurrentLayer:
    """Stands in for a non-sliceable (GDN/ArraysCache) layer."""


class KVCache:  # noqa: N801 - the class name is the sliceability signal
    """Stands in for a sliceable attention layer, matched by class name."""


def test_last_reachable_boundary_picks_the_final_reported_chunk():
    # _prefill_draft reports cached_len + j*step, then m-1, then m.
    assert draft_workflow._last_reachable_boundary(0, 13423, 2048, 1024) == 12288
    assert draft_workflow._last_reachable_boundary(4096, 13423, 2048, 1024) == 12288


def test_last_reachable_boundary_is_none_when_suffix_fits_one_chunk():
    # 14336 restored, 16368 to score: the single 2032-token chunk crosses no
    # reported boundary, so nothing new is publishable at this step.
    assert draft_workflow._last_reachable_boundary(14336, 16368, 2048, 1024) is None
    # Chunking on the block size makes the next full block reachable.
    assert draft_workflow._last_reachable_boundary(14336, 16368, 1024, 1024) == 15360


def test_an_aligned_total_publishes_the_boundary_below_it():
    """A boundary at the prompt end is unusable when the same prompt returns.

    The draft lookup asks for all but the last token, so it can never match
    a block ending at n_to_score. 16384 in 2048-token chunks reports 2048,
    ..., 14336, then 16383 and 16384; the last aligned position below the end
    is 14336, a full chunk back rather than one block back, because the
    prefill never stands anywhere in between.
    """
    assert draft_workflow._last_reachable_boundary(0, 16384, 2048, 1024) == 14336
    assert draft_workflow._last_reachable_boundary(0, 8192, 2048, 256) == 6144
    # Chunking finer than a block makes the block below the end reachable.
    assert draft_workflow._last_reachable_boundary(0, 8192, 256, 256) == 7936


def test_boundary_snapshot_is_captured_once_and_passed_to_store():
    """One extraction per scoring, at the only boundary worth publishing.

    Extracting at every block boundary and keeping the last one cost a full
    state extraction per block and showed up directly as scoring wall time.
    """
    request, plan = _request_and_plan()
    block_table = SimpleNamespace(num_tokens=4)
    recurrent, sliceable = _RecurrentLayer(), KVCache()
    reconstructed_cache = [recurrent, sliceable]
    draft_cache = _DraftCache(block_table, reconstructed_cache, block_size=4)
    seen_snapshot_caches: list[list[Any]] = []

    def extract_cache_states(cache: list[Any]) -> tuple[list[dict[str, Any]], Any]:
        seen_snapshot_caches.append(list(cache))
        return [{"state": "recurrent"}, None], None

    def score_tokens(model: Any, tokens: list[int], **kwargs: Any) -> tuple[Any, Any]:
        report = kwargs["progress_callback"]
        # cached_len 4, n_to_score 16, step 4: reported positions are
        # 4, 8, 12 (chunks), 15 (the clipped last chunk), then 16. The
        # target is 12: 16 is aligned but is the prompt end.
        for processed in (4, 8, 8, 12, 12, 15, 15, 16):
            report(processed, plan.n_to_score, "scoring")
        report(plan.n_to_score, plan.n_to_score, "lookahead")
        return mx.zeros(plan.n_to_score), reconstructed_cache

    _run(
        request,
        plan,
        draft_cache=draft_cache,
        score_tokens=score_tokens,
        extract_cache_states=extract_cache_states,
    )

    # Boundary captures null the sliceable layers, so they are the calls
    # carrying a None; the remaining call is the pre-existing store-time
    # extraction of the whole cache. Exactly one boundary capture.
    boundary_calls = [call for call in seen_snapshot_caches if None in call]
    assert boundary_calls == [[recurrent, None]]
    assert seen_snapshot_caches[-1] == [recurrent, sliceable]

    passed = draft_cache.store_boundary_snapshots[-1]
    assert passed is not None
    assert list(passed) == [12]
    assert passed[12] == [{"state": "recurrent"}, None]


def test_boundary_capture_ignores_unreported_positions():
    request, plan = _request_and_plan()
    block_table = SimpleNamespace(num_tokens=3)
    reconstructed_cache = [_RecurrentLayer()]
    draft_cache = _DraftCache(block_table, reconstructed_cache, block_size=4)
    extractions = 0

    def extract_cache_states(cache: list[Any]) -> tuple[list[dict[str, Any]], Any]:
        nonlocal extractions
        if None in cache:  # a boundary capture, not the store-time extraction
            extractions += 1
        return [{"state": "recurrent"}], None

    def score_tokens(model: Any, tokens: list[int], **kwargs: Any) -> tuple[Any, Any]:
        report = kwargs["progress_callback"]
        # Starting at 4 the target is 12; these are aligned but stop short.
        for processed in (4, 8):
            report(processed, plan.n_to_score, "scoring")
        return mx.zeros(plan.n_to_score), reconstructed_cache

    _run(
        request,
        plan,
        draft_cache=draft_cache,
        score_tokens=score_tokens,
        extract_cache_states=extract_cache_states,
    )

    assert extractions == 0
    assert draft_cache.store_boundary_snapshots[-1] is None


class _CountingCache:
    """Minimal cache whose ``state`` is real, so mx.eval in _prefill_draft works."""

    def __init__(self) -> None:
        self.offset = 0
        self.state = mx.zeros((1,), dtype=mx.float32)


class _CountingModel:
    """Advances its caches by the chunk length, as a real model's layers do."""

    def __init__(self, caches: list[_CountingCache]) -> None:
        self._caches = caches

    def __call__(self, prompt: mx.array, cache: list[Any] | None = None) -> mx.array:
        for entry in self._caches:
            entry.offset += prompt.shape[1]
        return mx.zeros((1, prompt.shape[1], 2), dtype=mx.float32)


def _replay_prefill(n: int, step: int) -> tuple[list[int], list[tuple[int, int]]]:
    """Run the real _prefill_draft and record what it reports and holds."""
    from omlx.patches.specprefill import _prefill_draft

    cache = [_CountingCache()]
    model = _CountingModel(cache)
    seen: list[tuple[int, int]] = []
    _prefill_draft(
        model,
        list(range(n)),
        cache,
        step_size=step,
        progress_callback=lambda processed, total: seen.append(
            (processed, cache[0].offset)
        ),
    )
    return [processed for processed, _ in seen], seen


def test_prefill_draft_schedule_is_exactly_this():
    """Pin the reported sequence, because the boundary math is derived from it.

    _last_reachable_boundary models this loop: it has to, since the state for
    a boundary exists only while the prefill is standing on it. Asserting the
    prediction alone is too weak -- several plausible changes to the loop
    leave the largest aligned position unmoved. So pin the sequence itself.
    A change here is not necessarily a bug; it means someone must re-derive
    the boundary math against the new schedule.
    """
    # 33 tokens in chunks of 8: the final chunk is truncated to leave one
    # token for the call that produces the logits.
    assert _replay_prefill(33, 8)[0] == [0, 8, 8, 16, 16, 24, 24, 32, 33]
    # An exact multiple still leaves that one token over, so the last full
    # chunk stops at 31 and the logits call carries it to 32.
    assert _replay_prefill(32, 8)[0] == [0, 8, 8, 16, 16, 24, 24, 31, 32]
    # A suffix inside one chunk reports only its endpoints.
    assert _replay_prefill(5, 8)[0] == [0, 4, 5]


def _schedule_oracle(n: int, step: int) -> list[int]:
    """What ``_prefill_draft`` reports for an *n*-token suffix, without MLX.

    A transcription of the loop, so the exhaustive sweep below can run in
    milliseconds. It is only trusted because
    ``test_the_schedule_oracle_is_the_real_loop`` checks it against the real
    ``_prefill_draft`` on the cases where a transcription is easiest to get
    wrong, and ``test_prefill_draft_schedule_is_exactly_this`` pins the real
    loop itself.
    """
    reported: list[int] = []
    processed = 0
    while n - processed > 1:
        reported.append(processed)
        processed += min(step, n - processed - 1)
        reported.append(processed)
    reported.append(n)
    return reported


def _reachable(reported: list[int], cached: int, n: int, block: int) -> int | None:
    aligned = [
        cached + p
        for p in reported
        if p > 0 and cached + p < n and (cached + p) % block == 0
    ]
    return max(aligned) if aligned else None


# (n, step, block): the real loop runs only on these. Each family is one a
# plausible transcription or closed form gets wrong.
_ADVERSARIAL = [
    (64, 8, 8),  # block == step, prompt end aligned
    (64, 8, 4),  # block divides step
    (100, 16, 8),  # block divides step, end unaligned
    (50, 8, 3),  # block does not divide step
    (61, 8, 5),
    (97, 12, 9),
    (40, 6, 4),
    (33, 8, 8),  # final chunk truncated to leave the logits token
    (33, 8, 4),
    (10, 2, 3),  # only the truncated chunk lands on a boundary
    (32, 8, 8),  # exact block-aligned prompt end: the end is never the answer
    (48, 16, 8),
    (16, 4, 4),
    (2, 2, 2),  # the smallest suffix that still runs the loop
    (5, 8, 4),  # a suffix inside one chunk
]


@pytest.mark.parametrize("n,step,block", _ADVERSARIAL)
def test_the_schedule_oracle_is_the_real_loop(n, step, block):
    reported, _ = _replay_prefill(n, step)
    assert reported == _schedule_oracle(n, step)


@pytest.mark.parametrize("n,step,block", _ADVERSARIAL)
def test_boundary_prediction_matches_the_real_loop(n, step, block):
    reported, _ = _replay_prefill(n, step)
    predicted = draft_workflow._last_reachable_boundary(0, n, step, block)
    assert predicted != n
    assert predicted == _reachable(reported, 0, n, block), predicted


def test_boundary_prediction_matches_the_schedule_everywhere():
    """Sweep, because the interesting cases are where block does not divide step.

    An earlier version of this test used six hand-picked combinations, all of
    which happened to have ``block_size`` dividing ``step``. That is the one
    family where a closed-form guess at the schedule is right; over this
    sweep such a guess is wrong thousands of times, each one a published
    boundary missed and a block of reuse lost. The sweep runs against the
    oracle; the real loop runs on ``_ADVERSARIAL``.
    """
    for step in range(2, 20):
        for block in range(2, 20):
            for n in range(2, 200):
                reported = _schedule_oracle(n, step)
                predicted = draft_workflow._last_reachable_boundary(0, n, step, block)
                assert predicted != n, (n, step, block)
                assert predicted == _reachable(reported, 0, n, block), (
                    n,
                    step,
                    block,
                    predicted,
                )


def test_boundary_prediction_survives_a_nonzero_cached_length():
    for cached in (16, 1024, 4096):
        for step in (8, 2048):
            for extra in (1, 7, 64, 3073):
                n = cached + extra
                reported, _ = _replay_prefill(extra, step)
                aligned = [cached + p for p in reported if cached < cached + p < n]
                aligned = [p for p in aligned if p % 1024 == 0]
                predicted = draft_workflow._last_reachable_boundary(
                    cached, n, step, 1024
                )
                assert predicted == (max(aligned) if aligned else None), (
                    cached,
                    step,
                    extra,
                    predicted,
                )


def test_prefill_draft_holds_exactly_the_tokens_it_reports():
    """Every reported position is the number of tokens the cache then holds.

    The snapshot is taken inside that callback, so if this ever stopped
    holding, a checkpoint would be stored under the wrong token count and a
    later restore would silently resume from the wrong place.
    """
    for n, step in ((64, 8), (100, 16), (33, 8)):
        _, seen = _replay_prefill(n, step)
        assert all(processed == offset for processed, offset in seen), (n, step, seen)
        assert seen[-1][0] == n


def test_sliceable_cache_types_track_the_scheduler():
    """The duplicated set must not drift from the one it mirrors.

    draft.py cannot import the scheduler's copy without a cycle, and the
    store that consumes these snapshots re-slices exactly the types the
    scheduler calls sliceable. Tests have no such cycle, so assert the equality here
    rather than leaving two lists to drift apart silently.
    """
    from omlx.scheduler import _KNOWN_SLICEABLE_CACHE_TYPES

    assert draft_workflow._SLICEABLE_CACHE_TYPES == _KNOWN_SLICEABLE_CACHE_TYPES


class _TrackedCache(list):
    """A cache list that can be weakly referenced, so its death is observable."""


class _ReconstructingCache(_DraftCache):
    """Hands out a restored cache without keeping a reference to it."""

    def __init__(self, layers: int = 2, **kwargs: Any) -> None:
        super().__init__(block_table=SimpleNamespace(num_tokens=16), **kwargs)
        self.layers = layers
        self.reconstruct_calls = 0

    def reconstruct_cache(self, block_table: Any) -> Any:
        self.reconstruct_calls += 1
        return _TrackedCache(_RecurrentLayer() for _ in range(self.layers))

    def store_cache(self, request_id: str, tokens: Any, cache_data: Any, **kw: Any):
        # Record that a store happened, not what was stored: keeping the
        # payload here would be this fake holding the cache alive, and the
        # test would then be measuring itself.
        self.stores.append((request_id, len(list(tokens)), len(cache_data), None))
        self.store_boundary_snapshots.append(kw.get("boundary_snapshots"))
        return None


def test_restored_cache_is_released_before_the_clear():
    """A cache-hit request must not still name the restored KV at the clear.

    sync_and_clear_cache is the only point that returns those buffers. Any
    local still holding the restored cache -- the reconstruct result, the
    cache score_tokens hands back, the snapshot, the extracted payload --
    keeps hundreds of MB of draft KV alive through the call meant to reclaim
    it, and makes the reclaim accounting measure a delta it did not get.
    """
    request, plan = _request_and_plan()
    draft_cache = _ReconstructingCache()
    tracker = _Logger(), _Tracker()
    logger, tracker = tracker[0], tracker[1]
    observed: dict[str, Any] = {}
    cache_ref: list[Any] = []

    def score_tokens(model: Any, tokens: list[int], **kwargs: Any) -> tuple[Any, Any]:
        existing = kwargs["existing_cache"]
        # Watch a layer, not the list: the extracted payload holds the layers'
        # arrays without holding the list, so a weak reference to the list
        # alone would miss that alias.
        observed["hit"] = existing is not None
        cache_ref.append(weakref.ref(existing[0]))
        return mx.zeros(plan.n_to_score), existing

    def on_clear() -> None:
        gc.collect()
        observed["alive_at_clear"] = cache_ref[0]() is not None

    # `new=` rather than `side_effect=`: a MagicMock records its call
    # arguments, which would itself keep the cache alive and make this test
    # measure the mock instead of the code.
    with (
        patch.object(draft_workflow, "get_prefill_tracker", new=lambda: tracker),
        patch("omlx.patches.specprefill.score_tokens", new=score_tokens),
        patch(
            "omlx.patches.specprefill.select_chunks",
            new=lambda importance, keep_pct: mx.arange(3),
        ),
        patch.object(draft_workflow.mx, "stream", new=lambda s: nullcontext()),
    ):
        draft_workflow.run_specprefill_draft_scoring(
            request=request,
            plan=plan,
            draft_model=object(),
            draft_prefix_cache=draft_cache,
            model_id="model-id",
            prefill_step_size=4,
            stream=object(),
            # As the real one does, the extracted payload references what it
            # was extracted from.
            extract_cache_states=lambda cache: (
                [{"state": layer} for layer in cache],
                None,
            ),
            sync_and_clear_cache=on_clear,
            log=logger,
        )

    assert draft_cache.reconstruct_calls == 1
    assert observed["hit"] is True, "the restored cache must reach score_tokens"
    assert (
        observed["alive_at_clear"] is False
    ), "something still names the restored draft cache at the clear point"
    # The hit is still reported and the store still happens.
    assert draft_cache.stores, "a cache hit must not skip the store"


class TestRepeatedBlockAlignedPrompt:
    """A block-aligned prompt scored twice restores a snapshot below its end.

    Real pieces throughout: a tiny random-weight Qwen3.5 (GDN recurrent and
    attention layers), BlockAwarePrefixCache over a PagedSSDCacheManager on
    disk, and the scheduler's own state extraction. The second request is
    served by a fresh prefix-cache instance, so the restore comes from disk.
    """

    BLOCK = 16
    STEP = 32
    N = 128  # block-aligned

    @staticmethod
    def _model():
        from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

        args = TextModelArgs.from_dict(
            {
                "model_type": "qwen3_5",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 256,
                "linear_num_value_heads": 2,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 16,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 3,
                "full_attention_interval": 2,
                "tie_word_embeddings": True,
                "rms_norm_eps": 1e-5,
                "head_dim": 32,
                "rope_theta": 1000.0,
                "partial_rotary_factor": 0.5,
                "max_position_embeddings": 1024,
            }
        )
        mx.random.seed(0)
        model = TextModel(args)
        mx.eval(model.parameters())
        return model

    def _prefix_cache(self, model, cache_dir):
        from mlx_lm.models.cache import make_prompt_cache

        from omlx.cache.hybrid_cache import ModelCacheConfig
        from omlx.cache.paged_cache import PagedCacheManager
        from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
        from omlx.cache.prefix_cache import BlockAwarePrefixCache

        types = ModelCacheConfig.from_cache_list(
            make_prompt_cache(model), model_name="tiny-draft"
        ).get_type_names()
        paged = PagedCacheManager(
            block_size=self.BLOCK, max_blocks=256, model_name="tiny-draft"
        )
        ssd = PagedSSDCacheManager(
            cache_dir=cache_dir,
            max_size_bytes=256 * 1024**2,
            expected_model_name="tiny-draft",
            expected_num_layers=len(model.layers),
            expected_block_size=self.BLOCK,
            expected_layer_cache_types=types,
        )
        paged.set_paged_ssd_cache_manager(ssd)
        prefix = BlockAwarePrefixCache(
            model=model, paged_cache_manager=paged, paged_ssd_cache_manager=ssd
        )
        return prefix, ssd

    def _score(self, model, tokens, prefix_cache):
        from omlx.scheduler import Scheduler

        request = Request(
            request_id=f"r-{id(prefix_cache)}",
            prompt=list(tokens),
            sampling_params=SamplingParams(),
        )
        request.prompt_token_ids = list(tokens)
        request.num_prompt_tokens = len(tokens)
        request.remaining_tokens = request.prompt_token_ids
        request.specprefill_system_end = 0
        request.cached_tokens = 0
        plan = plan_specprefill_scoring(
            remaining_tokens=request.remaining_tokens,
            system_prompt_end=0,
            cached_tokens=0,
            requested_threshold=None,
            requested_keep_pct=None,
            default_threshold=8,
            default_keep_pct=0.25,
        )
        assert plan is not None and plan.n_to_score == len(tokens)

        import omlx.patches.specprefill as sp

        real_score_tokens = sp.score_tokens
        seen: dict[str, Any] = {}

        def score_tokens(m, toks, **kwargs):
            existing = kwargs.get("existing_cache")
            seen["restored"] = (
                0 if existing is None else sp._logical_cache_offset(m, existing)
            )
            kwargs["temp"] = 0.0
            importance, cache = real_score_tokens(m, toks, **kwargs)
            seen["importance"] = importance
            seen["cache"] = cache
            return importance, cache

        extract_self = SimpleNamespace(model_name="tiny-draft")
        with patch.object(sp, "score_tokens", side_effect=score_tokens):
            draft_workflow.run_specprefill_draft_scoring(
                request=request,
                plan=plan,
                draft_model=model,
                draft_prefix_cache=prefix_cache,
                model_id="m",
                prefill_step_size=self.STEP,
                stream=mx.default_stream(mx.default_device()),
                extract_cache_states=lambda cache: Scheduler._extract_cache_states(
                    extract_self, cache
                ),
                sync_and_clear_cache=lambda: None,
                log=_Logger(),
            )
        assert request.specprefill_indices is not None
        seen["selection"] = request.specprefill_indices.tolist()
        return seen

    @staticmethod
    def _prompt_state(model, tokens, step):
        from mlx_lm.models.cache import make_prompt_cache

        from omlx.patches.specprefill import _prefill_draft

        cache = make_prompt_cache(model)
        _prefill_draft(model, tokens, cache, step_size=step)
        return cache

    @staticmethod
    def _assert_same_state(got, want):
        for g, w in zip(got, want):
            if hasattr(w, "keys"):
                assert g.offset == w.offset
                assert mx.array_equal(
                    g.keys[..., : g.offset, :], w.keys[..., : w.offset, :]
                ).item()
            else:
                for a, b in zip(g.cache, w.cache):
                    assert mx.array_equal(a, b).item()

    def test_second_request_restores_below_the_end_and_scores_like_cold(
        self, tmp_path
    ):
        model = self._model()
        tokens = [(i * 37 + 11) % 256 for i in range(self.N)]

        cold = self._score(model, tokens, None)

        writer, writer_ssd = self._prefix_cache(model, tmp_path)
        first = self._score(model, tokens, writer)
        writer_ssd.close()
        assert first["restored"] == 0

        reader, reader_ssd = self._prefix_cache(model, tmp_path)
        second = self._score(model, tokens, reader)
        reader_ssd.close()

        # _prefill_draft reports 32, 64, 96, 127, 128 on the first request,
        # so the last aligned boundary below 128 is 96. The N-1 lookup
        # matches seven blocks (112) and walks back to that snapshot.
        assert second["restored"] == 96
        # Restored at a point the cold prefill also stands on, the suffix
        # runs the same chunks, so the result is exact, not merely close.
        assert mx.array_equal(cold["importance"], second["importance"]).item()
        assert second["selection"] == cold["selection"]
        self._assert_same_state(
            second["cache"], self._prompt_state(model, tokens, self.STEP)
        )
