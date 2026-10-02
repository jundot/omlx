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
    extract_cache_states: (
        Callable[[list[Any]], tuple[list[dict[str, Any]], Any]] | None
    ) = None,
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

    assert any(
        "draft cache fetch failed: disk gone" in message
        for message in logger.debug_messages
    )
    assert trace["score_calls"][0]["existing_cache"] in (None, [])


def test_allocation_failure_is_survivable():
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
    pass


@pytest.mark.parametrize(
    "cached,n,step,block,expected",
    [
        (0, 32, 8, 8, 24),
        (0, 33, 8, 8, 32),
        (0, 50, 8, 3, 48),
        (0, 10, 2, 3, 9),
        (0, 2, 2, 2, None),
        (16, 21, 8, 4, 20),
        (16, 17, 8, 4, None),
        (4096, 7169, 2048, 1024, 7168),
        (14336, 16368, 2048, 1024, None),
    ],
)
def test_boundary_matches_prefill_progress(cached, n, step, block, expected):
    from omlx.patches.specprefill import _prefill_draft

    cache = SimpleNamespace(offset=cached, state=mx.zeros((1,)))
    reported = []

    def model(prompt, cache):
        cache[0].offset += prompt.shape[1]
        return mx.zeros((1, prompt.shape[1], 2))

    def progress(processed, total):
        assert cache.offset == cached + processed
        reported.append(cache.offset)

    _prefill_draft(
        model,
        list(range(n - cached)),
        [cache],
        step_size=step,
        progress_callback=progress,
    )
    aligned = [p for p in reported if cached < p < n and p % block == 0]
    assert (max(aligned) if aligned else None) == expected
    assert draft_workflow._last_reachable_boundary(cached, n, step, block) == expected


def test_sliceable_cache_types_track_the_scheduler():
    from omlx.scheduler import _KNOWN_SLICEABLE_CACHE_TYPES

    assert draft_workflow._SLICEABLE_CACHE_TYPES == _KNOWN_SLICEABLE_CACHE_TYPES


class _ReconstructingCache(_DraftCache):
    def __init__(self) -> None:
        super().__init__(block_table=SimpleNamespace(num_tokens=16))
        self.reconstruct_calls = 0

    def reconstruct_cache(self, block_table: Any) -> Any:
        self.reconstruct_calls += 1
        return [_RecurrentLayer() for _ in range(2)]

    def store_cache(self, request_id: str, tokens: Any, cache_data: Any, **kw: Any):
        self.stores.append((request_id, len(list(tokens)), len(cache_data), None))
        return None


def test_restored_cache_is_released_before_the_clear():
    request, plan = _request_and_plan()
    draft_cache = _ReconstructingCache()
    logger, tracker = _Logger(), _Tracker()
    observed: dict[str, Any] = {}
    cache_ref: list[Any] = []

    def score_tokens(model: Any, tokens: list[int], **kwargs: Any) -> tuple[Any, Any]:
        existing = kwargs["existing_cache"]
        observed["hit"] = existing is not None
        cache_ref.append(weakref.ref(existing[0]))
        return mx.zeros(plan.n_to_score), existing

    def on_clear() -> None:
        gc.collect()
        observed["alive_at_clear"] = cache_ref[0]() is not None

    # Mock call recording would retain cache arguments through the clear.
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
    assert draft_cache.stores, "a cache hit must not skip the store"


class TestDraftCacheReuse:
    BLOCK = 128
    STEP = 256

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
                "max_position_embeddings": 4096,
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
            hot_cache_max_bytes=0,
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
            default_keep_pct=0.75,
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
        with patch.object(sp, "score_tokens", new=score_tokens):
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
        if prefix_cache is not None:
            prefix_cache.release_cache(request.request_id)
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
                for attr in ("keys", "values"):
                    assert mx.array_equal(
                        getattr(g, attr)[..., : g.offset, :],
                        getattr(w, attr)[..., : w.offset, :],
                    ).item()
            else:
                for a, b in zip(g.cache, w.cache):
                    assert mx.array_equal(a, b).item()

    @pytest.mark.parametrize(
        "stored,n,restored", [(1024, 1024, 768), (1154, 1216, 1024)]
    )
    def test_disk_reuse_matches_cold_scoring(self, tmp_path, stored, n, restored):
        model = self._model()
        tokens = [(i * 37 + 11) % 256 for i in range(max(stored, n))]
        cold = self._score(model, tokens[:n], None)

        writer, writer_ssd = self._prefix_cache(model, tmp_path)
        try:
            first = self._score(model, tokens[:stored], writer)
            assert first["restored"] == 0
            self._assert_same_state(
                first["cache"], self._prompt_state(model, tokens[:stored], self.STEP)
            )
        finally:
            writer_ssd.close()

        reader, reader_ssd = self._prefix_cache(model, tmp_path)
        try:
            second = self._score(model, tokens[:n], reader)
        finally:
            reader_ssd.close()

        assert second["restored"] == restored
        assert mx.array_equal(cold["importance"], second["importance"]).item()
        assert second["selection"] == cold["selection"]
        # Exercise ranked selection beyond the mandatory 512-token tail.
        assert 512 < len(second["selection"]) < n
        self._assert_same_state(
            second["cache"], self._prompt_state(model, tokens[:n], self.STEP)
        )

    @pytest.mark.parametrize("cached", [64, 80])
    def test_scoring_rejects_cache_at_or_past_prompt_end(self, cached):
        from omlx.patches.specprefill import score_tokens

        model = self._model()
        tokens = list(range(80))
        cache = self._prompt_state(model, tokens[:cached], self.STEP)
        with pytest.raises(
            ValueError, match="leave at least the last prompt token uncached"
        ):
            score_tokens(model, tokens[:64], existing_cache=cache)
