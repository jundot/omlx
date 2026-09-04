# SPDX-License-Identifier: Apache-2.0
"""Tests for the SpecPrefill target-prefill workflow."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import mlx.core as mx
import pytest

import omlx.specprefill.target as target_workflow
from omlx.patches.specprefill import _OffsetAdjustedRoPE
from omlx.specprefill.planning import plan_specprefill_target


class _Logger:
    def __init__(self) -> None:
        self.info_messages: list[str] = []

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.info_messages.append(message)


class _AbortError(Exception):
    pass


class _Model:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, Any]] = []

    def __call__(self, tokens: Any, *, cache: Any) -> Any:
        self.calls.append((tokens, cache))
        return tokens


class _CacheLayer:
    """Mock cache layer that supports the ``.state`` property setter.

    The real mlx-lm cache types (KVCache, RotatingKVCache, ArraysCache) expose
    a ``state`` property with a setter that stores the KV tensor tuple. The
    static-prefix KV cache (#2177) restores states by assigning
    ``layer.state = state``. This mock stores the assigned value so the restore
    path can be exercised without real MLX tensors.
    """

    def __init__(self) -> None:
        self._state = (object(),)

    @property
    def state(self) -> Any:
        return self._state

    @state.setter
    def state(self, value: Any) -> None:
        self._state = value


class _TieredExactPrefixCache:
    def __init__(self) -> None:
        self.tokens: list[int] | None = None
        self.layer_states: list[dict[str, Any]] | None = None
        self.restore_promotions: list[bool] = []

    def restore_exact_prefix(
        self,
        request_id: str,
        tokens: list[int],
        *,
        promote_to_hot_cache: bool,
    ) -> list[Any] | None:
        del request_id
        self.restore_promotions.append(promote_to_hot_cache)
        if tokens != self.tokens or self.layer_states is None:
            return None
        restored_layers = [_CacheLayer() for _ in self.layer_states]
        for restored_layer, layer_state in zip(
            restored_layers, self.layer_states, strict=True
        ):
            restored_layer.state = layer_state["state"]
        return restored_layers

    def store_exact_prefix(
        self,
        request_id: str,
        tokens: list[int],
        cache_data: list[dict[str, Any]],
        model_cache_config: Any = None,
    ) -> object:
        del request_id, model_cache_config
        self.tokens = list(tokens)
        self.layer_states = cache_data
        return object()


def _extract_cache_states(
    cache: list[Any],
) -> tuple[list[dict[str, Any]], Any]:
    return [
        {
            "state": layer.state,
            "meta_state": (),
            "class_name": "_CacheLayer",
            "cache_type": "test",
        }
        for layer in cache
    ], None


def _all_tokens(
    system_token_count: int,
    conversation_token_count: int,
    conversation_start: int = 1_000,
) -> list[int]:
    return list(range(system_token_count)) + list(
        range(conversation_start, conversation_start + conversation_token_count)
    )


def _run(
    *,
    system_token_count: int,
    conversation_token_count: int,
    selected_indices: list[int],
    cached_tokens: int = 0,
    request_prompt_cache: list[Any] | None = None,
    conversation_start: int = 1_000,
    extract_cache_states: target_workflow.ExtractCacheStates | None = None,
    abort_error: _AbortError | None = None,
    abort_at: int | None = None,
    sparse_abort_error: _AbortError | None = None,
    exact_prefix_cache: _TieredExactPrefixCache | None = None,
    static_prefix_tokens: list[int] | None = None,
    promote_static_prefix_to_hot_cache: bool = True,
) -> tuple[Any, _Logger, dict[str, Any]]:
    all_tokens = _all_tokens(
        system_token_count,
        conversation_token_count,
        conversation_start,
    )
    plan = plan_specprefill_target(
        all_tokens=all_tokens,
        system_token_count=system_token_count,
        selected_indices=selected_indices,
        position_offset=system_token_count,
    )
    model = _Model()
    prompt_cache = [_CacheLayer()]
    selected_array = mx.array(selected_indices)
    original_rope = object()
    attention_module = SimpleNamespace(rope=original_rope)
    attention_layer = SimpleNamespace(self_attn=attention_module)
    model.layers = [attention_layer]
    logger = _Logger()
    stream = object()
    trace: dict[str, Any] = {
        "abort_points": [],
        "evaluations": [],
        "sparse_calls": [],
        "sparse_progress": [],
        "streams": [],
        "syncs": [],
        "system_progress": [],
    }

    def check_abort(processed: int) -> None:
        trace["abort_points"].append(processed)
        if abort_error is not None and processed == abort_at:
            raise abort_error

    def report_system_progress(processed: int, total: int) -> None:
        trace["system_progress"].append((processed, total))

    def report_sparse_progress(processed: int, total: int) -> None:
        trace["sparse_progress"].append((processed, total))
        if sparse_abort_error is not None:
            raise sparse_abort_error

    def sparse_prefill(
        target_model: Any,
        tokens: Any,
        selected: Any,
        cache: Any,
        **kwargs: Any,
    ) -> None:
        trace["sparse_calls"].append(
            {
                "cache": cache,
                "model": target_model,
                "position_offset": kwargs["position_offset"],
                "selected": selected,
                "step_size": kwargs["step_size"],
                "tokens": list(tokens),
            }
        )
        rope = _OffsetAdjustedRoPE(attention_module.rope, adjustment=10)
        attention_module.rope = rope
        trace["rope"] = rope
        kwargs["progress_callback"](0, len(tokens))

    def use_stream(selected_stream: Any):
        assert selected_stream is stream
        trace["streams"].append(selected_stream)
        return nullcontext()

    with (
        patch.object(target_workflow, "make_prompt_cache", return_value=prompt_cache),
        patch.object(
            target_workflow.mx, "eval", side_effect=trace["evaluations"].append
        ),
        patch.object(target_workflow.mx, "stream", side_effect=use_stream),
        patch(
            "omlx.patches.specprefill._find_attention_layers",
            return_value=[(0, attention_layer)],
        ),
        patch(
            "omlx.patches.specprefill._get_attn_module",
            return_value=attention_module,
        ),
        patch("omlx.patches.specprefill.sparse_prefill", side_effect=sparse_prefill),
    ):
        result = target_workflow.run_specprefill_target_prefill(
            target_model=model,
            request=SimpleNamespace(
                request_id="target-request",
                cached_tokens=cached_tokens,
                num_prompt_tokens=cached_tokens + len(all_tokens),
                prompt_cache=request_prompt_cache,
            ),
            plan=plan,
            all_tokens=all_tokens,
            selected_indices=selected_array,
            prefill_step_size=4,
            stream=stream,
            check_abort=check_abort,
            report_system_progress=report_system_progress,
            report_sparse_progress=report_sparse_progress,
            sync_and_clear_cache=lambda: trace["syncs"].append(stream),
            log=logger,
            extract_cache_states=extract_cache_states,
            exact_prefix_cache=exact_prefix_cache,
            static_prefix_tokens=static_prefix_tokens,
            promote_static_prefix_to_hot_cache=promote_static_prefix_to_hot_cache,
        )
    trace.update(
        {
            "all_tokens": all_tokens,
            "model": model,
            "prompt_cache": prompt_cache,
            "selected_indices": selected_array,
            "stream": stream,
        }
    )
    return result, logger, trace


def test_system_prefill_chunks_reports_checks_abort_and_uses_stream():
    _, _, trace = _run(
        system_token_count=13,
        conversation_token_count=8,
        selected_indices=[0, 2, 6],
    )

    assert [int(tokens.shape[1]) for tokens, _ in trace["model"].calls] == [4, 4, 4, 1]
    assert all(cache is trace["prompt_cache"] for _, cache in trace["model"].calls)
    assert trace["system_progress"] == [
        (0, 13),
        (4, 13),
        (4, 13),
        (8, 13),
        (8, 13),
        (12, 13),
        (12, 13),
        (13, 13),
    ]
    assert trace["abort_points"] == [0, 4, 4, 8, 8, 12, 12, 13]
    assert len(trace["evaluations"]) == 4
    assert trace["streams"] == [trace["stream"]] * 5
    assert trace["syncs"] == [trace["stream"]] * 3


@pytest.mark.parametrize(
    ("selected_indices", "expected_selected", "keeps_original"),
    [
        ([0, 5, 10], [0, 5, 10], True),
        ([10, 11, 0], [0, 10], False),
        ([11, 1, 11, 5], [1, 5, 11], False),
    ],
)
def test_sparse_prefill_preserves_sparse_inputs(
    selected_indices: list[int], expected_selected: list[int], keeps_original: bool
):
    _, _, trace = _run(
        system_token_count=5,
        conversation_token_count=12,
        selected_indices=selected_indices,
    )

    sparse_call = trace["sparse_calls"][0]
    assert sparse_call["model"] is trace["model"]
    assert sparse_call["cache"] is trace["prompt_cache"]
    assert sparse_call["tokens"] == trace["all_tokens"][5:]
    assert sparse_call["step_size"] == 4
    assert sparse_call["position_offset"] == 5
    assert sparse_call["selected"].tolist() == expected_selected
    assert (sparse_call["selected"] is trace["selected_indices"]) is keeps_original


def test_runtime_patch_helpers_adjust_rope_log_and_handoff_result():
    with patch.object(target_workflow.time, "monotonic", side_effect=[10.0, 11.2]):
        result, logger, trace = _run(
            system_token_count=5,
            conversation_token_count=10,
            selected_indices=[0, 5, 9],
        )

    assert result.prompt_cache is trace["prompt_cache"]
    assert result.tokens_to_process == trace["all_tokens"][-1:]
    assert trace["rope"]._adjustment == 9
    assert logger.info_messages == [
        "SpecPrefill: system prompt 5 tokens full prefill",
        "SpecPrefill: sparse prefill 2/10 conv tokens in 1.2s "
        "(total 15, cached 0, system 5 full, conv 10 sparse)",
    ]


def test_target_prefill_extends_an_existing_partial_prefix_cache():
    restored_prefix_cache = [_CacheLayer()]

    _, _, trace = _run(
        system_token_count=5,
        conversation_token_count=8,
        selected_indices=[0, 2, 6],
        cached_tokens=4,
        request_prompt_cache=restored_prefix_cache,
    )

    assert all(cache is restored_prefix_cache for _, cache in trace["model"].calls)
    assert trace["sparse_calls"][0]["cache"] is restored_prefix_cache


def test_github_2177_restores_static_prefix_from_tiered_cache():
    exact_prefix_cache = _TieredExactPrefixCache()
    static_prefix_tokens = list(range(5))
    common_args = {
        "system_token_count": 5,
        "conversation_token_count": 12,
        "selected_indices": [0, 5, 10],
        "exact_prefix_cache": exact_prefix_cache,
        "static_prefix_tokens": static_prefix_tokens,
        "extract_cache_states": _extract_cache_states,
    }

    _, _, cold_trace = _run(**common_args)
    warm_result, warm_logger, warm_trace = _run(
        **common_args,
        conversation_start=2_000,
        promote_static_prefix_to_hot_cache=False,
    )

    assert len(cold_trace["model"].calls) == 2
    assert warm_trace["model"].calls == []
    assert warm_result.static_prefix_cached_tokens == len(static_prefix_tokens)
    assert exact_prefix_cache.restore_promotions == [True, False]
    assert "system 5 static-cached" in warm_logger.info_messages[-1]


def test_static_prefix_hit_supersedes_a_shorter_block_cache_hit():
    exact_prefix_cache = _TieredExactPrefixCache()
    static_prefix_tokens = list(range(5))
    _run(
        system_token_count=5,
        conversation_token_count=8,
        selected_indices=[0, 2, 6],
        exact_prefix_cache=exact_prefix_cache,
        static_prefix_tokens=static_prefix_tokens,
        extract_cache_states=_extract_cache_states,
    )
    shorter_block_cache = [_CacheLayer()]

    result, _, warm_trace = _run(
        system_token_count=3,
        conversation_token_count=8,
        selected_indices=[0, 2, 6],
        cached_tokens=2,
        request_prompt_cache=shorter_block_cache,
        exact_prefix_cache=exact_prefix_cache,
        static_prefix_tokens=static_prefix_tokens,
        extract_cache_states=_extract_cache_states,
    )

    assert result.static_prefix_cached_tokens == 5
    assert result.prompt_cache is not shorter_block_cache
    assert warm_trace["model"].calls == []


def test_scheduler_abort_error_propagates_unchanged():
    abort_error = _AbortError("abort")

    with pytest.raises(_AbortError) as exception_info:
        _run(
            system_token_count=13,
            conversation_token_count=8,
            selected_indices=[0, 2, 6],
            abort_error=abort_error,
            abort_at=4,
        )

    assert exception_info.value is abort_error


def test_abort_releases_target_locals_before_propagating():
    abort_error = _AbortError("abort during sparse prefill")

    with pytest.raises(_AbortError) as exception_info:
        _run(
            system_token_count=5,
            conversation_token_count=8,
            selected_indices=[0, 2, 7],
            sparse_abort_error=abort_error,
        )

    assert exception_info.value is abort_error
    target_traceback = exception_info.tb
    while (
        target_traceback is not None
        and target_traceback.tb_frame.f_code
        is not target_workflow.run_specprefill_target_prefill.__code__
    ):
        target_traceback = target_traceback.tb_next
    assert target_traceback is not None
    target_locals = target_traceback.tb_frame.f_locals
    assert target_locals["prompt_cache"] is None
    assert target_locals["sys_arr"] is None
    assert target_locals["conversation_tokens"] is None
    assert target_locals["selected_indices"] is None
    assert target_locals["selected_indices_list"] is None
    assert target_locals["selected"] is None
