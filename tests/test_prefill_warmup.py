# SPDX-License-Identifier: Apache-2.0
"""Tests for shared, cache-safe model shape warmups."""

from types import SimpleNamespace

import pytest

from omlx.utils.prefill_warmup import (
    planned_local_prefill_shape_warmup_tokens,
    run_prefill_shape_warmup,
)


@pytest.mark.parametrize("model_type", ["deepseek_v4", "deepseek_v4_mtp"])
def test_local_ds4_shape_warmup_is_enabled_by_default(model_type):
    assert planned_local_prefill_shape_warmup_tokens(model_type, environ={}) == 1024


def test_local_shape_warmup_is_gated_and_operator_can_disable_it():
    assert planned_local_prefill_shape_warmup_tokens("llama", environ={}) == 0
    assert (
        planned_local_prefill_shape_warmup_tokens(
            "deepseek_v4", environ={"OMLX_PREFILL_SHAPE_WARMUP": "0"}
        )
        == 0
    )


def test_shape_warmup_evaluates_cache_then_releases_scratch():
    calls = []

    class FakeMX:
        int32 = "int32"

        @staticmethod
        def zeros(shape, *, dtype):
            calls.append(("zeros", shape, dtype))
            return "token-batch"

        @staticmethod
        def eval(*values):
            calls.append(("eval", values))

        @staticmethod
        def synchronize():
            calls.append(("synchronize",))

        @staticmethod
        def clear_cache():
            calls.append(("clear_cache",))

    class Model:
        def __call__(self, tokens, **kwargs):
            calls.append(("model", tokens, kwargs))
            return "hidden-output"

    caches = [SimpleNamespace(state="cache-0"), SimpleNamespace(state="cache-1")]

    def cache_factory(model, *, max_kv_size):
        calls.append(("cache_factory", model, max_kv_size))
        return caches

    ticks = iter((10.0, 10.25))
    model = Model()
    report = run_prefill_shape_warmup(
        FakeMX,
        model,
        tokens=1024,
        max_kv_size=32768,
        cache_factory=cache_factory,
        clock=lambda: next(ticks),
    )

    assert report == {
        "active": True,
        "tokens": 1024,
        "elapsed_seconds": 0.25,
    }
    assert ("cache_factory", model, 32768) in calls
    assert ("zeros", (1, 1024), "int32") in calls
    assert (
        "model",
        "token-batch",
        {"cache": caches, "skip_lm_head": True},
    ) in calls
    assert ("eval", ("hidden-output", ["cache-0", "cache-1"])) in calls
    assert calls[-1] == ("clear_cache",)


def test_shape_warmup_rejects_unbounded_token_count():
    with pytest.raises(ValueError, match="out of bounds"):
        run_prefill_shape_warmup(
            SimpleNamespace(),
            object(),
            tokens=4097,
            max_kv_size=None,
        )
