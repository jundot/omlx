# SPDX-License-Identifier: Apache-2.0
"""Routing tests for the MMA verify attention patch (mlx_lm sdpa hook)."""

from __future__ import annotations

import sys

import mlx.core as mx
import pytest
from mlx_lm.models import base as mlx_base
from mlx_lm.models.cache import KVCache, QuantizedKVCache

from omlx.patches import qwen35_verify_qmm as vq
from omlx.patches import verify_mma_attention as vma
from omlx.patches import verify_mma_kernel as vmk

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="requires Metal"
)

D = 128


def _save_hooks():
    return {
        name: (mod, mod.scaled_dot_product_attention)
        for name, mod in list(sys.modules.items())
        if name.startswith("mlx_lm.models.")
        and getattr(mod, "scaled_dot_product_attention", None) is not None
    }


@pytest.fixture
def patched(monkeypatch):
    """Patched mlx_lm sdpa with the kernel replaced by a call-counting spy."""
    # apply() re-points every mlx_lm module that imported the hook by name;
    # put them all back afterwards.
    saved = _save_hooks()
    calls = []
    real = vmk.mma_verify_sdpa

    def spy(*a, **k):
        calls.append(a[0].shape)
        return real(*a, **k)

    monkeypatch.setattr(vmk, "mma_verify_sdpa", spy)
    assert vma.apply() is True
    calls.clear()  # drop apply()'s probe call
    vq.set_verify_qmm_armed(True)  # the MTP verify forward arms this flag
    yield calls
    vq.set_verify_qmm_armed(False)
    for mod, original in saved.values():
        mod.scaled_dot_product_attention = original


def _filled_cache(prefix=300, hkv=8, dtype=mx.bfloat16, cls=KVCache):
    mx.random.seed(3)
    cache = cls()
    k = (mx.random.normal((1, hkv, prefix, D)) * 0.5).astype(dtype)
    cache.update_and_fetch(k, k)
    return cache


def _verify_block(cache, T, hq=32, hkv=8, dtype=mx.bfloat16):
    q = (mx.random.normal((1, hq, T, D)) * 0.5).astype(dtype)
    k = (mx.random.normal((1, hkv, T, D)) * 0.5).astype(dtype)
    v = (mx.random.normal((1, hkv, T, D)) * 0.5).astype(dtype)
    bk, bv = cache.update_and_fetch(k, v)
    return q, bk, bv


@pytest.mark.parametrize("T", [3, 4, 8])
def test_verify_block_routes_to_kernel_and_matches_stock(patched, T):
    cache = _filled_cache()
    q, bk, bv = _verify_block(cache, T)
    got = mlx_base.scaled_dot_product_attention(q, bk, bv, cache, D**-0.5, "causal")
    want = mx.fast.scaled_dot_product_attention(q, bk, bv, scale=D**-0.5, mask="causal")
    mx.eval(got, want)
    assert len(patched) == 1
    assert mx.abs(got.astype(mx.float32) - want.astype(mx.float32)).max().item() < 2e-2


@pytest.mark.parametrize("state", ["unarmed", "row_exact"])
def test_routes_only_while_the_verify_flag_is_armed(patched, state):
    # Outside a verify forward (prefill chunks, plain decode) and in row-exact
    # verify (output must equal one-row decode) the stock kernel serves it.
    if state == "unarmed":
        vq.set_verify_qmm_armed(False)
    else:
        vq.set_verify_qmm_armed(True, row_exact=True)
    cache = _filled_cache()
    q, bk, bv = _verify_block(cache, 4)
    got = mlx_base.scaled_dot_product_attention(q, bk, bv, cache, D**-0.5, "causal")
    want = mx.fast.scaled_dot_product_attention(q, bk, bv, scale=D**-0.5, mask="causal")
    mx.eval(got, want)
    assert patched == []
    assert mx.array_equal(got, want).item()


def test_modules_that_imported_the_hook_by_name_are_rerouted(patched):
    from mlx_lm.models import qwen3

    assert qwen3.scaled_dot_product_attention is mlx_base.scaled_dot_product_attention


@pytest.mark.parametrize(
    "why",
    ["one_row", "two_rows", "nine_rows", "array_mask", "float32", "quantized_cache",
     "wide_gqa", "head_dim_64"],
)
def test_everything_else_stays_on_stock(patched, why):
    T, hq, hkv, dtype, mask, cls, d = 4, 32, 8, mx.bfloat16, "causal", KVCache, D
    if why == "one_row":
        T = 1
    elif why == "two_rows":
        T = 2
    elif why == "nine_rows":
        T = 9
    elif why == "array_mask":
        mask = mx.zeros((T, 304 + 0)).astype(mx.bfloat16)
    elif why == "float32":
        dtype = mx.float32
    elif why == "quantized_cache":
        cls = QuantizedKVCache
    elif why == "wide_gqa":
        hq, hkv, T = 64, 2, 4  # 32 * 4 rows > 32
    elif why == "head_dim_64":
        d = 64
    if cls is QuantizedKVCache:
        cache = _filled_cache(cls=QuantizedKVCache)
        q = mx.zeros((1, hq, T, D), dtype=mx.bfloat16)
        assert not vma.route_ok(q, q, q, cache, mask, None)
        return
    if d != D:
        q = mx.zeros((1, hq, T, d), dtype=dtype)
        k = mx.zeros((1, hkv, 300, d), dtype=dtype)
        assert not vma.route_ok(q, k, k, KVCache(), mask, None)
        return
    cache = _filled_cache(hkv=hkv, dtype=dtype)
    q, bk, bv = _verify_block(cache, T, hq=hq, hkv=hkv, dtype=dtype)
    if why == "array_mask":
        mask = mx.zeros((T, bk.shape[2]), dtype=dtype)
    mlx_base.scaled_dot_product_attention(q, bk, bv, cache, D**-0.5, mask)
    assert patched == []


def test_mtp_patch_installs_the_hook():
    saved = _save_hooks()
    try:
        from omlx.patches.mlx_lm_mtp import apply_mlx_lm_mtp_patch

        assert apply_mlx_lm_mtp_patch() is True
        assert getattr(mlx_base.scaled_dot_product_attention, "_omlx_verify_mma", False)
    finally:
        for mod, original in saved.values():
            mod.scaled_dot_product_attention = original


def test_apply_leaves_the_hook_off_when_the_kernel_cannot_run(monkeypatch):
    # metal_kernel compiles and launches lazily, at mx.eval: a try/except around
    # the call in the hook cannot see a Metal failure. apply() probes once.
    saved = _save_hooks()
    # Another test in the same process may have left the hook installed (apply()
    # is idempotent and then skips the probe); start from a plain function.
    stock = lambda *args, **kwargs: None  # noqa: E731
    mlx_base.scaled_dot_product_attention = stock
    monkeypatch.setattr(vma, "_probe_ok", lambda: False)
    try:
        assert vma.apply() is False
        assert mlx_base.scaled_dot_product_attention is stock
    finally:
        for mod, original in saved.values():
            mod.scaled_dot_product_attention = original


def test_probe_runs_the_kernel_for_real():
    assert vma._probe_ok() is True


def test_engaged_counter_and_log_prove_the_route(patched, caplog):
    import logging

    before = vma.engaged_count()
    cache = _filled_cache()
    q, bk, bv = _verify_block(cache, 4)
    with caplog.at_level(logging.INFO, logger=vma.logger.name):
        mlx_base.scaled_dot_product_attention(q, bk, bv, cache, D**-0.5, "causal")
    assert vma.engaged_count() == before + 1
    # a call that stays on stock does not count
    q1, bk1, bv1 = _verify_block(cache, 1)
    mlx_base.scaled_dot_product_attention(q1, bk1, bv1, cache, D**-0.5, "causal")
    assert vma.engaged_count() == before + 1


def test_failed_probe_is_false_and_not_rerun(monkeypatch):
    calls = []

    def boom(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("metal compile failed")

    monkeypatch.setattr(vmk, "mma_verify_sdpa", boom)
    monkeypatch.setattr(vma, "_probe_result", None)  # forget any earlier probe
    assert vma._probe_ok() is False
    assert vma._probe_ok() is False
    assert len(calls) == 1


def test_keys_slice_shorter_than_the_cache_offset_stays_on_stock(patched):
    cache = _filled_cache()
    q, bk, bv = _verify_block(cache, 4)
    assert vma.route_ok(q, bk, bv, cache, "causal", None)  # control: full slice routes
    assert not vma.route_ok(q, bk[:, :, :-1], bv[:, :, :-1], cache, "causal", None)


def test_buffer_dtype_must_match_the_queries(patched):
    cache = _filled_cache(dtype=mx.float16)
    q, bk, bv = _verify_block(cache, 4, dtype=mx.float16)
    assert vma.route_ok(q, bk, bv, cache, "causal", None)  # control: same dtype routes
    assert not vma.route_ok(q.astype(mx.bfloat16), bk, bv, cache, "causal", None)


@pytest.mark.parametrize("hq,rows,routes", [(32, 4, True), (44, 3, False)])
def test_gqa_times_rows_limit_is_32(patched, hq, rows, routes):
    # one MMA group serves at most 32 rows: gqa 8 x 4 = 32 routes, 11 x 3 = 33 does not
    cache = _filled_cache(hkv=4)
    q, bk, bv = _verify_block(cache, rows, hq=hq, hkv=4)
    assert vma.route_ok(q, bk, bv, cache, "causal", None) is routes
