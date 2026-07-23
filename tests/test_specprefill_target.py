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
    def __init__(self) -> None:
        self.state = object()


def _all_tokens(system_token_count: int, conversation_token_count: int) -> list[int]:
    return list(range(system_token_count)) + list(
        range(1_000, 1_000 + conversation_token_count)
    )


def _run(
    *,
    system_token_count: int,
    conversation_token_count: int,
    selected_indices: list[int],
    abort_error: _AbortError | None = None,
    abort_at: int | None = None,
    sparse_abort_error: _AbortError | None = None,
) -> tuple[Any, _Logger, dict[str, Any]]:
    all_tokens = _all_tokens(system_token_count, conversation_token_count)
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
        patch.object(target_workflow.mx, "eval", side_effect=trace["evaluations"].append),
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
                cached_tokens=0,
                num_prompt_tokens=len(all_tokens),
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
    [([0, 5, 10], [0, 5, 10], True), ([10, 11, 0], [0, 10], False), ([11, 1, 11, 5], [1, 5, 11], False)],
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
