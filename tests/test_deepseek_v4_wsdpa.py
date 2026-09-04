# SPDX-License-Identifier: Apache-2.0
"""Tests for DeepSeek-V4 fused windowed + pooled prefill attention."""

import math

import mlx.core as mx
import pytest

requires_metal = pytest.mark.skipif(
    not mx.metal.is_available(), reason="Metal is required"
)


def _max_abs(a, b):
    return mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()


def _reference_attention(
    q,
    kv,
    pooled,
    sinks,
    scale,
    offset,
    window,
    ratio,
    topk=None,
):
    """Explicit fp32 reference for the exact rows visited by the kernels.

    ``kv`` may be a trimmed RotatingKVCache buffer holding only the last
    ``kv.shape[2]`` rows; buffer row 0 then maps to absolute position
    ``offset + q_len - kv_len``.
    """
    base = offset + q.shape[2] - kv.shape[2]
    head_outputs = []
    for head in range(q.shape[1]):
        row_outputs = []
        for row in range(q.shape[2]):
            position = offset + row
            local_start = max(base, position - window + 1) - base
            local_end = position - base
            key_parts = [kv[0, 0, local_start : local_end + 1].astype(mx.float32)]

            pooled_length = 0 if pooled is None else pooled.shape[1]
            visible_pool = min((position + 1) // ratio, pooled_length)
            if topk is None:
                if visible_pool:
                    key_parts.append(pooled[0, :visible_pool].astype(mx.float32))
            else:
                indices = []
                for index in topk[0, row].tolist():
                    if index >= visible_pool:
                        break
                    indices.append(index)
                if indices:
                    key_parts.append(pooled[0, indices].astype(mx.float32))

            keys = mx.concatenate(key_parts, axis=0)
            scores = (keys @ q[0, head, row].astype(mx.float32)) * scale
            normalizer = mx.logsumexp(scores, axis=-1)
            normalizer = mx.logaddexp(normalizer, sinks[head].astype(mx.float32))
            weights = mx.exp(scores - normalizer)
            row_outputs.append((weights[:, None] * keys).sum(axis=0))
        head_outputs.append(mx.stack(row_outputs))
    return mx.stack(head_outputs)[None].astype(mx.bfloat16)


def _inputs(q_len, offset, window, pooled_len, ratio, trim=0):
    mx.random.seed(7)
    kv_len = offset + q_len - trim
    q = (mx.random.normal((1, 64, q_len, 512)) * 0.25).astype(mx.bfloat16)
    kv = (mx.random.normal((1, 1, kv_len, 512)) * 0.25).astype(mx.bfloat16)
    pooled = (mx.random.normal((1, pooled_len, 512)) * 0.25).astype(mx.bfloat16)
    sinks = (mx.random.normal((64,)) * 0.1).astype(mx.bfloat16)
    scale = 1.0 / math.sqrt(512)
    mx.eval(q, kv, pooled, sinks)
    return q, kv, pooled, sinks, scale, offset, window, ratio


def _reset_wsdpa(monkeypatch):
    from omlx.patches.deepseek_v4 import wsdpa_attention as wsdpa

    monkeypatch.setattr(wsdpa, "_ENABLED", True)
    monkeypatch.setattr(wsdpa, "_TOPK_ENABLED", True)
    monkeypatch.setattr(wsdpa, "_broken", False)
    monkeypatch.setattr(wsdpa, "_ready", False, raising=False)
    monkeypatch.setattr(wsdpa, "_topk_ready", False, raising=False)
    return wsdpa


def test_wsdpa_prefill_route_activates_only_after_output_evaluates(monkeypatch):
    wsdpa = _reset_wsdpa(monkeypatch)
    q = mx.zeros((1, 64, 2, 512), dtype=mx.bfloat16)
    kv = mx.zeros((1, 1, 2, 512), dtype=mx.bfloat16)
    sinks = mx.zeros((64,), dtype=mx.bfloat16)

    assert not wsdpa.wsdpa_prefill_route_active()

    monkeypatch.setattr(
        wsdpa,
        "_get_kernel",
        lambda: lambda **kwargs: [mx.zeros((64, 2, 512), dtype=mx.bfloat16)],
    )
    out = wsdpa.wsdpa_prefill(q, kv, None, sinks, 1.0, 0, 128, 1)

    assert out is not None
    assert wsdpa.wsdpa_prefill_route_active()


def test_wsdpa_dispatch_failure_keeps_route_inactive(monkeypatch):
    wsdpa = _reset_wsdpa(monkeypatch)
    q = mx.zeros((1, 64, 2, 512), dtype=mx.bfloat16)
    kv = mx.zeros((1, 1, 2, 512), dtype=mx.bfloat16)
    sinks = mx.zeros((64,), dtype=mx.bfloat16)

    def fail(**kwargs):
        raise RuntimeError("synthetic dispatch failure")

    monkeypatch.setattr(wsdpa, "_get_kernel", lambda: fail)

    assert wsdpa.wsdpa_prefill(q, kv, None, sinks, 1.0, 0, 128, 1) is None
    assert wsdpa._broken
    assert not wsdpa.wsdpa_prefill_route_active()


def test_wsdpa_first_evaluation_failure_keeps_route_inactive(monkeypatch):
    wsdpa = _reset_wsdpa(monkeypatch)
    q = mx.zeros((1, 64, 2, 512), dtype=mx.bfloat16)
    kv = mx.zeros((1, 1, 2, 512), dtype=mx.bfloat16)
    sinks = mx.zeros((64,), dtype=mx.bfloat16)
    monkeypatch.setattr(
        wsdpa,
        "_get_kernel",
        lambda: lambda **kwargs: [mx.zeros((64, 2, 512), dtype=mx.bfloat16)],
    )

    def fail_eval(*args):
        raise RuntimeError("synthetic evaluation failure")

    monkeypatch.setattr(wsdpa.mx, "eval", fail_eval)

    assert wsdpa.wsdpa_prefill(q, kv, None, sinks, 1.0, 0, 128, 1) is None
    assert wsdpa._broken
    assert not wsdpa.wsdpa_prefill_route_active()


def test_wsdpa_topk_route_activates_only_after_output_evaluates(monkeypatch):
    wsdpa = _reset_wsdpa(monkeypatch)
    q = mx.zeros((1, 64, 5, 512), dtype=mx.bfloat16)
    kv = mx.zeros((1, 1, 5, 512), dtype=mx.bfloat16)
    pooled = mx.zeros((1, 3, 512), dtype=mx.bfloat16)
    topk = mx.zeros((1, 5, 2), dtype=mx.uint32)
    sinks = mx.zeros((64,), dtype=mx.bfloat16)

    assert not wsdpa.wsdpa_prefill_route_active(topk=True)

    monkeypatch.setattr(
        wsdpa,
        "_get_topk_kernel",
        lambda: lambda **kwargs: [mx.zeros((64, 5, 512), dtype=mx.bfloat16)],
    )
    out = wsdpa.wsdpa_topk_prefill(q, kv, pooled, topk, sinks, 1.0, 0, 128, 4)

    assert out is not None
    assert wsdpa.wsdpa_prefill_route_active(topk=True)


def test_wsdpa_topk_first_evaluation_failure_keeps_route_inactive(monkeypatch):
    wsdpa = _reset_wsdpa(monkeypatch)
    q = mx.zeros((1, 64, 5, 512), dtype=mx.bfloat16)
    kv = mx.zeros((1, 1, 9, 512), dtype=mx.bfloat16)
    pooled = mx.zeros((1, 3, 512), dtype=mx.bfloat16)
    topk = mx.zeros((1, 5, 2), dtype=mx.uint32)
    sinks = mx.zeros((64,), dtype=mx.float32)
    monkeypatch.setattr(
        wsdpa,
        "_get_topk_kernel",
        lambda: lambda **kwargs: [mx.zeros((64, 5, 512), dtype=mx.bfloat16)],
    )
    monkeypatch.setattr(
        wsdpa.mx,
        "eval",
        lambda *args: (_ for _ in ()).throw(RuntimeError("top-k eval failed")),
    )

    out = wsdpa.wsdpa_topk_prefill(q, kv, pooled, topk, sinks, 1.0, 0, 4, 4)

    assert out is None
    assert wsdpa._broken
    assert not wsdpa.wsdpa_prefill_route_active()
    assert not wsdpa.wsdpa_prefill_route_active(topk=True)


def test_wsdpa_route_state_respects_disable_and_failure(monkeypatch):
    wsdpa = _reset_wsdpa(monkeypatch)
    monkeypatch.setattr(wsdpa, "_ready", True)
    monkeypatch.setattr(wsdpa, "_topk_ready", True)

    assert wsdpa.wsdpa_prefill_route_active()
    assert wsdpa.wsdpa_prefill_route_active(topk=True)

    monkeypatch.setattr(wsdpa, "_TOPK_ENABLED", False)
    assert wsdpa.wsdpa_prefill_route_active()
    assert not wsdpa.wsdpa_prefill_route_active(topk=True)

    monkeypatch.setattr(wsdpa, "_ENABLED", False)
    assert not wsdpa.wsdpa_prefill_route_active()

    monkeypatch.setattr(wsdpa, "_ENABLED", True)
    monkeypatch.setattr(wsdpa, "_broken", True)
    assert not wsdpa.wsdpa_prefill_route_active()
    assert not wsdpa.wsdpa_prefill_route_active(topk=True)


def test_wsdpa_import_does_not_register_head_dim_512_globally():
    from omlx import memory_monitor
    from omlx.patches.deepseek_v4 import wsdpa_attention  # noqa: F401

    assert 512 not in memory_monitor._SDPA_TILED_PREFILL_HEAD_DIMS


@requires_metal
def test_wsdpa_prefill_matches_explicit_reference(monkeypatch):
    wsdpa = _reset_wsdpa(monkeypatch)
    args = _inputs(q_len=5, offset=7, window=4, pooled_len=3, ratio=4)

    out = wsdpa.wsdpa_prefill(*args)
    ref = _reference_attention(*args)

    assert out is not None
    mx.eval(out, ref)
    assert out.shape == ref.shape
    assert out.dtype == mx.bfloat16
    assert _max_abs(out, ref) < 8e-3


@requires_metal
def test_wsdpa_topk_prefill_matches_explicit_reference(monkeypatch):
    wsdpa = _reset_wsdpa(monkeypatch)
    q, kv, pooled, sinks, scale, offset, window, ratio = _inputs(
        q_len=6,
        offset=11,
        window=5,
        pooled_len=5,
        ratio=4,
    )
    topk = mx.array(
        [
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 3],
            [1, 2, 3],
            [1, 2, 4],
        ],
        dtype=mx.uint32,
    )[None]

    out = wsdpa.wsdpa_topk_prefill(
        q, kv, pooled, topk, sinks, scale, offset, window, ratio
    )
    ref = _reference_attention(
        q, kv, pooled, sinks, scale, offset, window, ratio, topk=topk
    )

    assert out is not None
    mx.eval(out, ref)
    assert out.shape == ref.shape
    assert out.dtype == mx.bfloat16
    assert _max_abs(out, ref) < 8e-3


def test_wsdpa_prefill_rejects_non_deepseek_v4_head_count(monkeypatch):
    wsdpa = _reset_wsdpa(monkeypatch)
    q = mx.zeros((1, 16, 4, 512), dtype=mx.bfloat16)
    kv = mx.zeros((1, 1, 4, 512), dtype=mx.bfloat16)
    sinks = mx.zeros((16,), dtype=mx.bfloat16)

    assert wsdpa.wsdpa_prefill(q, kv, None, sinks, 1.0, 0, 128, 1) is None


@requires_metal
def test_wsdpa_prefill_matches_reference_with_trimmed_rotating_cache(monkeypatch):
    """RotatingKVCache trims the local buffer to the last W + L - 1 rows, so
    during later prefill chunks buffer row 0 is at absolute position
    base = offset + L - S > 0. The kernel must translate window bounds."""
    wsdpa = _reset_wsdpa(monkeypatch)
    args = _inputs(q_len=8, offset=15, window=6, pooled_len=4, ratio=4, trim=5)

    out = wsdpa.wsdpa_prefill(*args)
    ref = _reference_attention(*args)

    assert out is not None
    mx.eval(out, ref)
    assert out.shape == ref.shape
    assert _max_abs(out, ref) < 8e-3


@requires_metal
def test_wsdpa_topk_prefill_matches_reference_with_trimmed_rotating_cache(
    monkeypatch,
):
    wsdpa = _reset_wsdpa(monkeypatch)
    q, kv, pooled, sinks, scale, offset, window, ratio = _inputs(
        q_len=6,
        offset=11,
        window=5,
        pooled_len=5,
        ratio=4,
        trim=3,
    )
    topk = mx.array(
        [
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 3],
            [1, 2, 3],
            [1, 2, 4],
        ],
        dtype=mx.uint32,
    )[None]

    out = wsdpa.wsdpa_topk_prefill(
        q, kv, pooled, topk, sinks, scale, offset, window, ratio
    )
    ref = _reference_attention(
        q, kv, pooled, sinks, scale, offset, window, ratio, topk=topk
    )

    assert out is not None
    mx.eval(out, ref)
    assert out.shape == ref.shape
    assert _max_abs(out, ref) < 8e-3
