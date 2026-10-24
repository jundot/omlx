# SPDX-License-Identifier: Apache-2.0
"""NAX DeepSeek V4 window + top-k attention against ``dsv4_wsdpa_topk``.

Synthetic bf16 tensors at the V4 Flash attention geometry (64 heads x 512,
128-row sliding window, index_topk = 512 pooled rows at ratio 4).
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.deepseek_v4 import sparse_attention_nax as nax_attn
from omlx.patches.deepseek_v4 import wsdpa_attention as wsdpa

HEADS, DIM = 64, 512


def _nax_hardware() -> bool:
    try:
        from omlx.custom_kernels.nax import is_nax_available

        return bool(is_nax_available())
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _nax_hardware(), reason="needs M5 tensor units")


@pytest.fixture
def nax(monkeypatch):
    # Upstream #4396 removed the OMLX_DSV4_WSDPA env switches and their
    # module flags; only reset what the module still has.
    for flag in ("_ENABLED", "_TOPK_ENABLED"):
        if hasattr(wsdpa, flag):
            monkeypatch.setattr(wsdpa, flag, True)
    monkeypatch.setattr(wsdpa, "_broken", False)

    def set_enabled(enabled: bool, min_rows: int | None = None):
        monkeypatch.setattr(nax_attn, "_ENABLED", enabled)
        monkeypatch.setattr(nax_attn, "_BROKEN", False)
        if min_rows is not None:
            monkeypatch.setattr(nax_attn, "MIN_ROWS", min_rows)
        nax_attn.nax_sparse_attention_available.cache_clear()

    yield set_enabled
    nax_attn.nax_sparse_attention_available.cache_clear()


def _inputs(length, offset, window, pooled_len, ratio, topk_count, trim=0, seed=7):
    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    kv_len = offset + length - trim
    q = (mx.random.normal((1, HEADS, length, DIM)) * 0.25).astype(mx.bfloat16)
    kv = (mx.random.normal((1, 1, kv_len, DIM)) * 0.25).astype(mx.bfloat16)
    pooled = (mx.random.normal((1, pooled_len, DIM)) * 0.25).astype(mx.bfloat16)
    sinks = (mx.random.normal((HEADS,)) * 0.1).astype(mx.bfloat16)
    # Temporally sorted distinct pooled rows per query, as the Indexer returns;
    # rows past the query's visible prefix sort last.
    rows = np.stack(
        [np.sort(rng.choice(pooled_len, topk_count, replace=False)) for _ in range(length)]
    )
    topk = mx.array(rows[None].astype(np.uint32))
    mx.eval(q, kv, pooled, sinks, topk)
    return q, kv, pooled, topk, sinks, 1.0 / math.sqrt(DIM), offset, window, ratio


def _exact(q, kv, pooled, topk, sinks, scale, offset, window, ratio):
    """float64 reference over exactly the rows ``dsv4_wsdpa_topk`` visits."""
    qn = np.array(q.astype(mx.float32))[0].astype(np.float64)
    kvn = np.array(kv.astype(mx.float32))[0, 0].astype(np.float64)
    pn = np.array(pooled.astype(mx.float32))[0].astype(np.float64)
    tk = np.array(topk)[0]
    sink = np.array(sinks.astype(mx.float32)).astype(np.float64)
    length = qn.shape[1]
    base = offset + length - kvn.shape[0]
    out = np.zeros_like(qn)
    for row in range(length):
        position = offset + row
        start = max(base, position - window + 1) - base
        keys = [kvn[start : position - base + 1]]
        visible = min((position + 1) // ratio, pn.shape[0])
        chosen = []
        for index in tk[row]:
            if index >= visible:
                break
            chosen.append(index)
        if chosen:
            keys.append(pn[chosen])
        keys = np.concatenate(keys)
        scores = qn[:, row] @ keys.T * scale
        top = np.maximum(scores.max(-1), sink)
        weights = np.exp(scores - top[:, None])
        denominator = weights.sum(-1) + np.exp(sink - top)
        out[:, row] = (weights @ keys) / denominator[:, None]
    return out


def _both(nax, args, min_rows=None):
    nax(False)
    native = wsdpa.wsdpa_topk_prefill(*args)
    nax(True, min_rows)
    before = nax_attn.nax_dispatch_counts()["nax"]
    got = wsdpa.wsdpa_topk_prefill(*args)
    assert native is not None and got is not None
    mx.eval(native, got)
    assert nax_attn.nax_dispatch_counts()["nax"] == before + 1
    assert got.shape == native.shape and got.dtype == mx.bfloat16
    return np.array(native.astype(mx.float32))[0], np.array(got.astype(mx.float32))[0]


def _assert_close(got, native, exact):
    """Within one bf16 ulp of native at each output row's scale, and no
    further from the float64 reference than native is."""
    row = np.maximum(np.abs(got), np.abs(native)).max(-1, keepdims=True)
    ulp = np.exp2(np.floor(np.log2(np.maximum(row, 2.0**-126))) - 7)
    diff = np.abs(got - native)
    assert (diff <= ulp).all(), float((diff / ulp).max())
    rms = lambda value: float(np.sqrt(np.mean((value - exact) ** 2)))  # noqa: E731
    assert rms(got) <= rms(native) * 1.25 + 1e-7


@pytest.mark.parametrize(
    "length,offset,pooled_len,topk_count",
    [
        (16, 2400, 604, 512),  # every top-k row visible to the last queries
        (48, 0, 600, 512),  # chunk start: short windows, few or no pooled rows
        (64, 1000, 600, 512),  # part of each top-k list still invisible
        (33, 4095, 1032, 512),
        (40, 3000, 760, 100),  # top-k count not a multiple of the tile
    ],
)
def test_matches_native_and_reference(nax, length, offset, pooled_len, topk_count):
    args = _inputs(length, offset, 128, pooled_len, 4, topk_count, seed=length)
    native, got = _both(nax, args)
    _assert_close(got, native, _exact(*args))


@pytest.mark.parametrize("window,topk_count", [(5, 3), (100, 37), (200, 512)])
def test_other_window_and_topk_sizes(nax, window, topk_count):
    """Windows shorter and longer than one 128-slot tile, ragged tails."""
    args = _inputs(24, 900, window, 600, 4, topk_count, seed=window)
    native, got = _both(nax, args)
    _assert_close(got, native, _exact(*args))


def test_trimmed_local_buffer(nax):
    """A rotating local KV buffer that no longer holds the oldest rows."""
    args = _inputs(32, 2000, 128, 600, 4, 512, trim=1900, seed=5)
    native, got = _both(nax, args)
    _assert_close(got, native, _exact(*args))


def test_small_case_from_wsdpa_suite(nax):
    args = list(_inputs(6, 11, 5, 5, 4, 3, seed=7))
    args[3] = mx.array(
        [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 3], [1, 2, 3], [1, 2, 4]],
        dtype=mx.uint32,
    )[None]
    native, got = _both(nax, tuple(args), min_rows=1)
    _assert_close(got, native, _exact(*args))


def test_deterministic(nax):
    args = _inputs(48, 3000, 128, 800, 4, 512, seed=9)
    nax(True)
    first = nax_attn.sparse_topk_attention_nax(*args)
    second = nax_attn.sparse_topk_attention_nax(*args)
    mx.eval(first, second)
    assert mx.array_equal(first.view(mx.uint16), second.view(mx.uint16)).item()


def test_unsupported_inputs_stay_native(nax):
    nax(True)
    q, kv, pooled, topk, sinks, scale, offset, window, ratio = _inputs(
        32, 1000, 128, 600, 4, 512
    )
    call = nax_attn.sparse_topk_attention_nax
    assert call(q[:, :, :4], kv, pooled, topk[:, :4], sinks, scale, offset, window, ratio) is None
    assert call(q[:, :48], kv, pooled, topk, sinks[:48], scale, offset, window, ratio) is None
    assert call(q, kv, pooled, topk.astype(mx.int32), sinks, scale, offset, window, ratio) is None
    assert (
        call(q.astype(mx.float16), kv, pooled, topk, sinks, scale, offset, window, ratio)
        is None
    )
    nax(False)
    assert call(q, kv, pooled, topk, sinks, scale, offset, window, ratio) is None


def test_disabled_by_default(nax):
    nax(False)
    assert not nax_attn.nax_sparse_attention_available()
