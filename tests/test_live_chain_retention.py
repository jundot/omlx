# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the CacheList live-chain retention path."""

from types import SimpleNamespace

import pytest

from omlx.scheduler import Scheduler

try:
    import mlx.core as mx

    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

    HAS_MLX = True
except ImportError:
    HAS_MLX = False


pytestmark = [
    pytest.mark.skipif(not HAS_MLX, reason="MLX not available"),
    pytest.mark.skipif(
        not hasattr(Scheduler, "_extract_prefill_snapshot_states"),
        reason="eager boundary snapshots not available",
    ),
]


def test_deepseek_cachelist_prompt_boundary_round_trip(monkeypatch):
    """Retain/adopt preserves every PoolingCache state element."""
    apply_deepseek_v4_patch()
    from mlx_lm.models.cache import CacheList, PoolingCache, RotatingKVCache

    monkeypatch.setenv("OMLX_RETAIN_LIVE_CHAIN", "1")

    rotating = RotatingKVCache(max_size=16)
    keys = mx.arange(1 * 2 * 6 * 4, dtype=mx.float32).reshape(1, 2, 6, 4)
    rotating.update_and_fetch(keys, keys + 1000)

    pooling = PoolingCache(4)
    pooling.update_and_fetch(mx.arange(1 * 3 * 8, dtype=mx.float32).reshape(1, 3, 8))
    layer = CacheList(rotating, pooling)

    scheduler = Scheduler.__new__(Scheduler)
    scheduler._stream = mx.default_stream(mx.default_device())
    scheduler.model_name = "synthetic-deepseek-v4"
    scheduler._retention_slot = None

    first = SimpleNamespace(
        request_id="r1",
        prompt_token_ids=[10, 11, 12, 13, 14, 15, 16],
        needs_think_prefix=True,
        vlm_extra_keys_for_cache=None,
        _retention_prompt_snapshot=None,
        # Finish-time extraction is what gates retention here. The retained
        # payload itself must come from the eager insert-time snapshot because
        # PoolingCache cannot be sliced back from final decode state.
        _extracted_cache=[{"state": (mx.zeros((1,)), mx.zeros((1,)))}],
        _model_cache_config=None,
    )

    scheduler._capture_prompt_boundary_state(first, [layer])
    assert first._retention_prompt_snapshot is not None
    assert first._retention_prompt_snapshot["offset"] == 6
    assert first._retention_prompt_snapshot["cache"][0]["class_name"] == "CacheList"

    scheduler._retain_live_chain("r1", first)
    assert first._retention_chain_retained is True
    assert scheduler._retention_slot is not None

    second = SimpleNamespace(
        request_id="r2",
        prompt_token_ids=[10, 11, 12, 13, 14, 15, 99, 100],
        vlm_extra_keys_for_cache=None,
    )
    assert scheduler._try_adopt_retained_chain(second) is True
    assert second.cached_tokens == 6
    assert second.remaining_tokens == [99, 100]

    rebuilt = second.prompt_cache[0]
    assert type(rebuilt).__name__ == "CacheList"
    assert [type(c).__name__ for c in rebuilt.caches] == [
        "PrefillReadyRotatingKVCache",
        "PoolingCache",
    ]
    pooling_state = rebuilt.caches[1].state
    assert len(pooling_state) == 5
    assert tuple(pooling_state[2].shape) == (1, 3, 8)

    # An adopted request normally prefills only the short new-turn suffix via
    # Scheduler._schedule_waiting(), then inserts directly. Simulate its one
    # processed token and verify the renewed prompt boundary can feed turn 3.
    rebuilt.caches[0].update_and_fetch(
        mx.zeros((1, 2, 1, 4), dtype=mx.float32),
        mx.zeros((1, 2, 1, 4), dtype=mx.float32),
    )
    rebuilt.caches[1].update_and_fetch(mx.zeros((1, 1, 8), dtype=mx.float32))
    second.needs_think_prefix = True
    second._retention_prompt_snapshot = None
    second._extracted_cache = [{"state": (mx.zeros((1,)), mx.zeros((1,)))}]
    second._model_cache_config = None

    scheduler._capture_prompt_boundary_state(second, second.prompt_cache)
    assert second._retention_prompt_snapshot["offset"] == 7
    scheduler._retain_live_chain("r2", second)
    assert scheduler._retention_slot["tokens"] == second.prompt_token_ids[:7]

    third = SimpleNamespace(
        request_id="r3",
        prompt_token_ids=second.prompt_token_ids[:7] + [101, 102],
        vlm_extra_keys_for_cache=None,
    )
    assert scheduler._try_adopt_retained_chain(third) is True
    assert third.cached_tokens == 7


def test_adoption_sets_native_admission_cached_tokens(monkeypatch):
    """Admission reads the cached_tokens set by prepare-time adoption."""
    monkeypatch.setenv("OMLX_RETAIN_LIVE_CHAIN", "1")
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._retention_slot = {
        "tokens": [1, 2, 3],
        "cache": [],
        "config": None,
    }
    request = SimpleNamespace(
        request_id="r",
        prompt_token_ids=[1, 2, 3, 4, 5],
        vlm_extra_keys_for_cache=None,
    )

    # Empty cache payload is rejected before adoption, so exercise the exact
    # request fields through a minimal construct-layer stub.
    scheduler._retention_slot["cache"] = [{"class_name": "dummy"}]
    scheduler._construct_retained_layer = lambda _layer: object()
    assert scheduler._try_adopt_retained_chain(request) is True
    assert request.cached_tokens == 3
    assert request.remaining_tokens == [4, 5]
