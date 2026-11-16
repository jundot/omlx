# SPDX-License-Identifier: Apache-2.0
"""A prefill chunk that passes the tail keeps the Gated DeltaNet state there."""

import types

import mlx.core as mx
import mlx.nn as nn
import pytest

lang = pytest.importorskip("mlx_vlm.models.qwen3_5.language")
vcache = pytest.importorskip("mlx_vlm.models.cache")

from omlx.patches import qwen35_gdn_tail_capture as tail_capture  # noqa: E402
from omlx.scheduler import Scheduler  # noqa: E402


def layer_cfg():
    return types.SimpleNamespace(
        hidden_size=256,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
    )


@pytest.fixture(scope="module")
def layer():
    assert tail_capture.apply_qwen35_gdn_tail_capture_patch()
    cfg = layer_cfg()
    mx.random.seed(0)
    gdn = lang.Qwen3_5GatedDeltaNet(cfg)
    gdn.update(
        nn.utils.tree_map(
            lambda p: p.astype(mx.bfloat16) if p.ndim > 1 else p, gdn.parameters()
        )
    )
    mx.eval(gdn.parameters())
    return gdn


def _cache(gdn, history):
    cache = vcache.ArraysCache(size=2)
    if history is not None:
        gdn(history, cache=cache)
    return cache


def _close(a, b, atol):
    return mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item() <= atol


@pytest.mark.parametrize("prefix,length,offset", [(0, 32, 26), (100, 33, 26), (500, 40, 1), (500, 40, 39)])
def test_capture_matches_the_stock_forward(layer, prefix, length, offset):
    history = mx.random.normal((1, prefix, 256)).astype(mx.bfloat16) if prefix else None
    x = mx.random.normal((1, length, 256)).astype(mx.bfloat16)

    stock = _cache(layer, history)
    out_stock = layer(x, cache=stock)
    head = _cache(layer, history)
    layer(x[:, :offset], cache=head)

    captured = _cache(layer, history)
    assert tail_capture.request_tail_capture([captured], offset)
    out = layer(x, cache=captured)
    conv_at, state_at = captured._omlx_captured
    mx.eval(out, out_stock, conv_at, state_at)

    # The whole chunk's output and final state are the stock forward's.
    assert _close(out, out_stock, 4e-3)
    assert mx.array_equal(captured[0], stock[0]).item()
    assert _close(captured[1], stock[1], 1e-3)
    # The state at the offset is what a chunk ending there leaves.
    assert _close(conv_at, head[0], 4e-3)
    assert _close(state_at, head[1], 1e-3)
    # The request is one-shot.
    assert captured._omlx_capture_at is None


def test_offsets_outside_the_chunk_run_the_stock_forward(layer):
    x = mx.random.normal((1, 8, 256)).astype(mx.bfloat16)
    cache = _cache(layer, None)
    tail_capture.request_tail_capture([cache], 8)
    layer(x, cache=cache)
    assert cache._omlx_captured is None and cache._omlx_capture_at is None


def test_only_arrays_caches_take_a_capture(layer):
    assert not tail_capture.request_tail_capture([vcache.KVCache()], 4)
    assert not tail_capture.request_tail_capture([vcache.ArraysCache(size=2)], 0)
    assert not tail_capture.request_tail_capture([], 4)


def test_snapshot_cache_holds_the_captured_arrays(layer):
    x = mx.random.normal((1, 16, 256)).astype(mx.bfloat16)
    cache = _cache(layer, None)
    other = object()
    tail_capture.request_tail_capture([cache], 10)
    layer(x, cache=cache)
    conv_at, state_at = cache._omlx_captured
    snapshot = tail_capture.captured_snapshot_cache([other, cache])
    assert snapshot[0] is other
    assert snapshot[1] is not cache and snapshot[1].cache[0] is conv_at and snapshot[1].cache[1] is state_at
    assert cache._omlx_captured is None
    # Nothing captured: no snapshot.
    tail_capture.request_tail_capture([cache], 10)
    assert tail_capture.captured_snapshot_cache([cache]) is None


class _Model(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers


def test_capture_layers_are_found_behind_the_vlm_adapter(layer):
    class Adapter(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self._language_model = inner

    class Adapter2(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self._vlm_model = _Model([inner])
            self._language_model = inner

    other = lang.Qwen3_5GatedDeltaNet(layer_cfg())
    assert tail_capture.model_capture_layers(Adapter(_Model([layer, other]))) == 2
    # The same layer reached twice counts once.
    assert tail_capture.model_capture_layers(Adapter2(_Model([layer]))) == 1


def test_scheduler_keeps_the_stop_for_other_recurrent_layers(layer):
    sched = Scheduler.__new__(Scheduler)
    sched.model = _Model([nn.Linear(4, 4)])
    cache = vcache.ArraysCache(size=2)
    request = types.SimpleNamespace(request_id="r1")
    assert sched._plan_tail_capture(
        request, [cache], cache_tokens=1000, n_tokens=32, tail_at=1026, system_at=None
    ) is None
    assert getattr(cache, "_omlx_capture_at", None) is None


def test_scheduler_plans_a_capture_only_inside_the_chunk(layer, monkeypatch):
    sched = Scheduler.__new__(Scheduler)
    sched.model = _Model([layer])
    request = types.SimpleNamespace(request_id="r1")
    cache = vcache.ArraysCache(size=2)
    kw = dict(cache_tokens=1000, n_tokens=32, system_at=None)

    assert sched._plan_tail_capture(request, [cache], tail_at=1026, **kw) == 1026
    assert cache._omlx_capture_at == 26
    assert sched._plan_tail_capture(request, [cache], tail_at=1032, **kw) is None
    assert sched._plan_tail_capture(request, [cache], tail_at=None, **kw) is None
    assert sched._plan_tail_capture(
        request, [cache], tail_at=1026, cache_tokens=1000, n_tokens=32, system_at=1010
    ) is None
    # MTP prompt priming tracks this request: keep the exact stop.
    ctx = types.SimpleNamespace(valid=True, request_id="r1")
    monkeypatch.setattr(
        "omlx.scheduler._mtp_priming._find_ctx", lambda model: ctx
    )
    assert sched._plan_tail_capture(request, [cache], tail_at=1026, **kw) is None
