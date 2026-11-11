# SPDX-License-Identifier: Apache-2.0
"""Tests for the Qwen3.5/3.6 FA-256 steel attention patch."""

from __future__ import annotations

import math
import sys
import types
from unittest.mock import MagicMock

import mlx.core as mx
import pytest

import omlx.patches.qwen35_fa256_attention as patch

_needs_metal = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
# The real gate, kept before the fixture below replaces it on the module.
_m1_gate = patch._m1_sdpa_enabled


@pytest.fixture(autouse=True)
def _steel_route_only(monkeypatch):
    # These tests exercise the steel-kernel route; on M1 hosts the chunked
    # MLX SDPA route (M1 section at the end of this file) would intercept first.
    monkeypatch.setattr(patch, "_m1_sdpa_enabled", lambda: False)


def _qkv(q_len=128, kv_len=None, dtype=mx.bfloat16):
    kv_len = q_len if kv_len is None else kv_len
    mx.random.seed(3)
    q = mx.random.normal((1, 24, q_len, 256)).astype(dtype)
    k = mx.random.normal((1, 4, kv_len, 256)).astype(dtype)
    v = mx.random.normal((1, 4, kv_len, 256)).astype(dtype)
    mx.eval(q, k, v)
    return q, k, v


def _install_fake_vlm_base(monkeypatch):
    root = types.ModuleType("mlx_vlm")
    models = types.ModuleType("mlx_vlm.models")
    base = types.ModuleType("mlx_vlm.models.base")
    language = types.ModuleType("mlx_vlm.models.qwen3_5.language")

    def original(q, k, v, cache, scale, mask=None, sinks=None):
        return "original"

    base.scaled_dot_product_attention = original
    language.scaled_dot_product_attention = original
    root.models = models
    models.base = base

    for name, module in {
        "mlx_vlm": root,
        "mlx_vlm.models": models,
        "mlx_vlm.models.base": base,
        "mlx_vlm.models.qwen3_5.language": language,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return base, language


@pytest.fixture(autouse=True)
def _fresh_fa256_patch(monkeypatch):
    import omlx.patches.qwen35_fa256_attention as patch
    from omlx import memory_monitor

    monkeypatch.setattr(patch, "_PATCHED", False, raising=False)
    memory_monitor._SDPA_TILED_PREFILL_HEAD_DIMS.pop(256, None)
    # Pin the NAX auto-gate off so apply/route behavior stays identical on
    # M5-family test machines; the NAX gating tests override this locally.
    monkeypatch.setattr(patch, "is_nax_available", lambda: False)
    # Pin the auto budget so unit tests stay off the GPU and deterministic;
    # the calibration itself is covered by its own Metal-gated test.
    monkeypatch.setattr(
        patch,
        "_auto_dispatch_budget",
        lambda *a, **k: patch._DEFAULT_DISPATCH_BUDGET,
    )
    yield
    monkeypatch.setattr(patch, "_PATCHED", False, raising=False)
    memory_monitor._SDPA_TILED_PREFILL_HEAD_DIMS.pop(256, None)


def test_route_gate_is_qwen_fa256_only():
    import omlx.patches.qwen35_fa256_attention as patch

    q, k, _ = _qkv(128, 2048)
    assert patch._should_route(q, k, None, "causal", None, min_kv_len=2048)
    # E4: mask=None means "no mask" in MLX's own semantics, but this kernel
    # is always invoked with causal=True -- routing on None would silently
    # causal-ize a caller that intended unrestricted attention. Only the
    # literal "causal" string routes.
    # See docs/qwen35-hardening-and-optimization.md E4.
    assert not patch._should_route(q, k, None, None, None, min_kv_len=2048)
    # any q%kv==0 GQA layout routes (issue #2155): the MoE 16/2 layout is
    # the case the relaxation exists for
    assert patch._should_route(q[:, :12], k, None, "causal", None, 2048)
    assert patch._should_route(q, k[:, :2], None, "causal", None, 2048)
    assert patch._should_route(q[:, :16], k[:, :2], None, "causal", None, 2048)
    # non-divisible head counts stay on the stock path
    assert not patch._should_route(q[:, :10], k, None, "causal", None, 2048)
    assert not patch._should_route(q[:, :14], k[:, :3], None, "causal", None, 2048)
    assert not patch._should_route(q[:, :, :1], k, None, "causal", None, 2048)
    # decode-shaped multi-row (MTP verify, qL = 1 + depth <= 9) -> stock path;
    # the steel prefill kernel is 3-16x slower at tiny q_len (issue #2127)
    for q_len in (2, 4, 9, 15):
        qv, kv, _ = _qkv(q_len, 16384)
        assert not patch._should_route(qv, kv, None, "causal", None, 2048)
    qv, kv, _ = _qkv(16, 16384)
    assert patch._should_route(qv, kv, None, "causal", None, 2048)
    assert not patch._should_route(q, k, None, mx.zeros((128, 2048)), None, 2048)
    assert not patch._should_route(q, k, None, "causal", mx.zeros((4,)), 2048)

    class _QuantCache:
        bits = 4

    assert not patch._should_route(q, k, _QuantCache(), "causal", None, 2048)


def test_vlm_patch_routes_and_passes_through(monkeypatch):
    import omlx.patches.qwen35_fa256_attention as patch

    base, language = _install_fake_vlm_base(monkeypatch)
    calls = []

    def fake_kernel(
        q, k, v, scale, causal=True, q_block=32, k_block=8, dispatch_budget=0
    ):
        calls.append(
            (q.shape, k.shape, scale, causal, q_block, k_block, dispatch_budget)
        )
        return "steel"

    monkeypatch.setattr(patch, "_native_kernel", lambda: fake_kernel)
    monkeypatch.setattr(
        patch._attention_fast, "fa256_supports_dispatch_budget", lambda: True
    )
    monkeypatch.setattr(patch.mx.metal, "is_available", lambda: True)

    assert patch.apply_qwen35_fa256_attention_patch(min_kv_len=16)
    q, k, v = _qkv(32, 32)
    scale = 1.0 / math.sqrt(256)
    assert base.scaled_dot_product_attention(q, k, v, None, scale, "causal") == "steel"
    assert language.scaled_dot_product_attention is base.scaled_dot_product_attention
    assert calls == [
        (
            (1, 24, 32, 256),
            (1, 4, 32, 256),
            scale,
            True,
            32,
            8,
            patch._DEFAULT_DISPATCH_BUDGET,
        )
    ]

    from omlx import memory_monitor

    routes = memory_monitor._SDPA_TILED_PREFILL_HEAD_DIMS[256]
    assert any(route.min_query_len == 16 and route.min_kv_len == 16 for route in routes)

    q_decode, _, _ = _qkv(1, 32)
    assert (
        base.scaled_dot_product_attention(q_decode, k, v, None, scale, "causal")
        == "original"
    )


def test_kernel_failure_keeps_registered_bounded_route(monkeypatch):
    import omlx.patches.qwen35_fa256_attention as patch

    base, _ = _install_fake_vlm_base(monkeypatch)
    monkeypatch.setattr(
        patch, "_native_kernel", lambda: MagicMock(side_effect=RuntimeError("boom"))
    )
    monkeypatch.setattr(
        patch._attention_fast, "fa256_supports_dispatch_budget", lambda: True
    )
    monkeypatch.setattr(patch.mx.metal, "is_available", lambda: True)
    bounded = MagicMock(return_value="bounded")
    monkeypatch.setattr(patch, "_bounded_sdpa_fallback", bounded)

    assert patch.apply_qwen35_fa256_attention_patch(min_kv_len=16)
    q, k, v = _qkv(32, 32)
    scale = 1.0 / math.sqrt(256)
    assert (
        base.scaled_dot_product_attention(q, k, v, None, scale, "causal") == "bounded"
    )
    bounded.assert_called_once_with(q, k, v, scale, "causal", None)


def test_dispatch_budget_uses_auto_calibration(monkeypatch):
    import omlx.patches.qwen35_fa256_attention as patch

    base, _ = _install_fake_vlm_base(monkeypatch)
    calls = []

    def fake_kernel(
        q, k, v, scale, causal=True, q_block=32, k_block=8, dispatch_budget=0
    ):
        calls.append(dispatch_budget)
        return "steel"

    monkeypatch.setattr(patch, "_native_kernel", lambda: fake_kernel)
    monkeypatch.setattr(
        patch._attention_fast, "fa256_supports_dispatch_budget", lambda: True
    )
    monkeypatch.setattr(patch, "_auto_dispatch_budget", lambda *a, **k: 777)
    monkeypatch.setattr(patch.mx.metal, "is_available", lambda: True)

    assert patch.apply_qwen35_fa256_attention_patch(min_kv_len=16)
    q, k, v = _qkv(32, 32)
    base.scaled_dot_product_attention(q, k, v, None, 0.0625, "causal")
    assert calls == [777]


def test_dispatch_budget_zeroed_on_old_extension(monkeypatch):
    # An extension built before the chunked-dispatch fix rejects the kwarg;
    # the patch must fall back to the single-dispatch behavior instead of
    # failing every routed call into the stock path (issue #2225).
    import omlx.patches.qwen35_fa256_attention as patch

    base, _ = _install_fake_vlm_base(monkeypatch)
    calls = []

    def fake_kernel(
        q, k, v, scale, causal=True, q_block=32, k_block=8, dispatch_budget=0
    ):
        calls.append(dispatch_budget)
        return "steel"

    monkeypatch.setattr(patch, "_native_kernel", lambda: fake_kernel)
    monkeypatch.setattr(
        patch._attention_fast, "fa256_supports_dispatch_budget", lambda: False
    )
    monkeypatch.setattr(patch.mx.metal, "is_available", lambda: True)

    assert patch.apply_qwen35_fa256_attention_patch(min_kv_len=16)
    q, k, v = _qkv(32, 32)
    base.scaled_dot_product_attention(q, k, v, None, 0.0625, "causal")
    assert calls == [0]


def test_apply_skips_on_nax_gpu(monkeypatch):
    # MLX 0.32.2 has a native NAX split-D fused path for head-dim-256 causal
    # prefill, so the patch must not replace it with the pre-NAX kernel.
    import omlx.patches.qwen35_fa256_attention as patch

    monkeypatch.setattr(patch, "is_nax_available", lambda: True)
    assert patch.apply_qwen35_fa256_attention_patch() is False


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal is required")
def test_native_fa256_matches_mlx_reference_small():
    from omlx.custom_kernels.attention import fast

    if not fast.has_symbol("fa256_attention"):
        pytest.skip("native fa256_attention is unavailable")

    q, k, v = _qkv(128)
    scale = 1.0 / math.sqrt(256)
    out = fast.fa256_attention(q, k, v, scale, causal=True)
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
    mx.eval(out, ref)

    err = mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))).item()
    rel = (
        mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)))
        / (mx.max(mx.abs(ref.astype(mx.float32))) + 1e-9)
    ).item()
    assert err < 2e-2
    assert rel < 1e-2


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal is required")
def test_auto_dispatch_budget_calibrates_within_clamp():
    import omlx.patches.qwen35_fa256_attention as patch
    from omlx.custom_kernels.attention import fast

    if not fast.has_symbol("fa256_attention"):
        pytest.skip("native fa256_attention is unavailable")
    if not fast.fa256_supports_dispatch_budget():
        pytest.skip("extension predates chunked dispatch")

    budget = patch._auto_dispatch_budget(fast.fa256_attention, 32, 8)
    assert patch._MIN_AUTO_BUDGET <= budget <= patch._MAX_AUTO_BUDGET


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal is required")
@pytest.mark.parametrize(
    "q_len,kv_len",
    [
        (2048, 8192),  # chunked-prefill shape: kL >> qL
        (4096, 4096),  # square: later chunks causally dead for early rows
        (2048, 8001),  # unaligned kL -> align_K variant on the last chunk
    ],
)
def test_native_fa256_chunked_matches_single_dispatch(q_len, kv_len):
    # The dispatch budget splits the key axis into separately dispatched
    # chunks combined by logsumexp weights (issue #2225); the result must
    # match the single-dispatch kernel up to combine rounding.
    from omlx.custom_kernels.attention import fast

    if not fast.has_symbol("fa256_attention"):
        pytest.skip("native fa256_attention is unavailable")
    if not fast.fa256_supports_dispatch_budget():
        pytest.skip("extension predates chunked dispatch")

    q, k, v = _qkv(q_len, kv_len)
    scale = 1.0 / math.sqrt(256)
    single = fast.fa256_attention(q, k, v, scale, causal=True, dispatch_budget=0)
    # Budget forcing ~8 chunks for these shapes.
    budget = (24 * q_len * kv_len) // 8
    chunked = fast.fa256_attention(
        q, k, v, scale, causal=True, dispatch_budget=budget
    )
    mx.eval(single, chunked)

    diff = mx.max(mx.abs(single.astype(mx.float32) - chunked.astype(mx.float32))).item()
    assert not mx.isnan(chunked.astype(mx.float32)).any().item()
    assert diff < 5e-3


# --- M1 route: query-chunked MLX SDPA ---------------------------------------


def _reference(q, k, v, scale):
    """Dense end-aligned causal attention in fp32 (GQA by head repeat)."""
    q32, k32, v32 = (t.astype(mx.float32) for t in (q, k, v))
    rep = q.shape[1] // k.shape[1]
    k32 = mx.repeat(k32, rep, axis=1)
    v32 = mx.repeat(v32, rep, axis=1)
    scores = (q32 * scale) @ k32.transpose(0, 1, 3, 2)
    q_len, kv_len = q.shape[2], k.shape[2]
    rows = mx.arange(q_len)[:, None] + (kv_len - q_len)
    cols = mx.arange(kv_len)[None, :]
    scores = mx.where(cols <= rows, scores, -mx.inf)
    return mx.softmax(scores, axis=-1) @ v32


@_needs_metal
@pytest.mark.parametrize(
    "q_len,kv_len,rows",
    [(256, 1024, 64), (200, 777, 64), (128, 128, 32), (96, 2048, 96)],
)
@pytest.mark.parametrize("heads", [(8, 2), (6, 1)])
def test_chunked_sdpa_matches_dense_causal(q_len, kv_len, rows, heads):
    hq, hkv = heads
    mx.random.seed(0)
    q = mx.random.normal((1, hq, q_len, 256)).astype(mx.float16)
    k = mx.random.normal((1, hkv, kv_len, 256)).astype(mx.float16)
    v = mx.random.normal((1, hkv, kv_len, 256)).astype(mx.float16)
    scale = 256**-0.5
    out = patch._m1_chunked_sdpa(q, k, v, scale, rows)
    ref = _reference(q, k, v, scale)
    assert out.shape == q.shape
    assert mx.max(mx.abs(out.astype(mx.float32) - ref)).item() < 2e-2


def test_query_rows_respect_work_budget_and_floor():
    # 2048-row chunk, 24 q heads: shrinks with depth, never below 64 rows
    assert patch._m1_query_rows(24, 2048, 16384) == 256
    assert patch._m1_query_rows(24, 2048, 32768) == 128
    assert patch._m1_query_rows(24, 2048, 65536) == 64
    assert patch._m1_query_rows(24, 2048, 98304) == 64
    # ~128k keys: even 64 rows exceed the max dispatch work -> steel kernel
    assert patch._m1_query_rows(24, 2048, 131072) == 0
    # 16-head MoE layout keeps the M1 route a little deeper
    assert patch._m1_query_rows(16, 2048, 131072) == 64
    # short calls run in one dispatch
    assert patch._m1_query_rows(24, 512, 4096) == 512


def test_route_follows_the_gpu_generation(monkeypatch):
    from omlx.utils import hardware

    gate = _m1_gate.__wrapped__
    monkeypatch.setattr(hardware, "get_apple_gpu_generation", lambda: 13)
    assert gate()
    monkeypatch.setattr(hardware, "get_apple_gpu_generation", lambda: 15)
    assert not gate()


@_needs_metal
def test_d512_route_shape_gates(monkeypatch):
    monkeypatch.setattr(patch, "_m1_sdpa_enabled", lambda: True)
    q = mx.zeros((1, 16, 64, 512), dtype=mx.float16)
    short = mx.zeros((1, 2, 1024, 512), dtype=mx.float16)
    # below the key threshold, wrong head_dim, array mask, sinks: not routed
    assert patch._m1_d512_attention(q, short, short, None, 0.04, "causal", None) is None
    q256 = mx.zeros((1, 16, 64, 256), dtype=mx.float16)
    k256 = mx.zeros((1, 2, 40000, 256), dtype=mx.float16)
    assert (
        patch._m1_d512_attention(q256, k256, k256, None, 0.04, "causal", None) is None
    )
    monkeypatch.setattr(patch, "_M1_D512_MIN_KV", 256)
    k = mx.zeros((1, 2, 300, 512), dtype=mx.float16)
    assert (
        patch._m1_d512_attention(q, k, k, None, 0.04, mx.zeros((64, 300)), None) is None
    )
    assert (
        patch._m1_d512_attention(q, k, k, None, 0.04, "causal", mx.zeros((16,))) is None
    )
    monkeypatch.setattr(patch, "_m1_sdpa_enabled", lambda: False)
    assert patch._m1_d512_attention(q, k, k, None, 0.04, "causal", None) is None


@_needs_metal
def test_d512_route_matches_dense_causal(monkeypatch):
    monkeypatch.setattr(patch, "_m1_sdpa_enabled", lambda: True)
    monkeypatch.setattr(patch, "_M1_D512_MIN_KV", 256)
    monkeypatch.setattr(patch, "_M1_SDPA_WORK", 16 * 32 * 300)  # force several blocks
    mx.random.seed(3)
    q = mx.random.normal((1, 16, 96, 512)).astype(mx.float16)
    k = mx.random.normal((1, 2, 300, 512)).astype(mx.float16)
    out = patch._m1_d512_attention(q, k, k, None, 512**-0.5, "causal", None)
    assert out is not None and out.shape == q.shape
    ref = _reference(q, k, k, 512**-0.5)
    assert mx.max(mx.abs(out.astype(mx.float32) - ref)).item() < 2e-2


@_needs_metal
def test_chunked_sdpa_blocks_share_one_scratch_size(monkeypatch):
    # Blocks of different sizes leave MLX's buffer cache holding GBs of
    # scratch it cannot reuse; every block must see the same key length.
    seen = []
    real = mx.fast.scaled_dot_product_attention

    def spy(q, k, v, **kwargs):
        seen.append((k.shape, kwargs["mask"].shape if hasattr(kwargs["mask"], "shape") else None))
        return real(q, k, v, **kwargs)

    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", spy)
    q = mx.random.normal((1, 8, 256, 64)).astype(mx.float16)
    k = mx.random.normal((1, 2, 4096, 64)).astype(mx.float16)
    patch._m1_chunked_sdpa(q, k, k, 0.125, 64)
    assert len(seen) == 4
    assert len(set(seen)) == 1
