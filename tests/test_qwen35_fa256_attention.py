# SPDX-License-Identifier: Apache-2.0
"""Tests for the Qwen3.5/3.6 FA-256 steel attention patch."""

from __future__ import annotations

import math
import sys
import types

import mlx.core as mx
import pytest


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

    monkeypatch.setattr(patch, "_PATCHED", False, raising=False)
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
    monkeypatch.delenv("OMLX_FA256_STEEL", raising=False)
    monkeypatch.delenv("OMLX_FA256_MIN_KV_LEN", raising=False)
    monkeypatch.delenv("OMLX_FA256_Q_BLOCK", raising=False)
    monkeypatch.delenv("OMLX_FA256_K_BLOCK", raising=False)
    monkeypatch.delenv("OMLX_FA256_DEBUG", raising=False)
    monkeypatch.delenv("OMLX_FA256_DISPATCH_BUDGET", raising=False)
    yield
    monkeypatch.setattr(patch, "_PATCHED", False, raising=False)


def test_route_gate_is_qwen_fa256_only():
    import omlx.patches.qwen35_fa256_attention as patch

    q, k, _ = _qkv(128, 2048)
    assert patch._should_route(q, k, None, "causal", None, min_kv_len=2048)
    assert patch._should_route(q, k, None, None, None, min_kv_len=2048)
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
        patch._fa256_fast, "fa256_supports_dispatch_budget", lambda: True
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

    q_decode, _, _ = _qkv(1, 32)
    assert (
        base.scaled_dot_product_attention(q_decode, k, v, None, scale, "causal")
        == "original"
    )


def test_dispatch_budget_env_and_capability_gate(monkeypatch):
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
        patch._fa256_fast, "fa256_supports_dispatch_budget", lambda: True
    )
    monkeypatch.setattr(patch.mx.metal, "is_available", lambda: True)
    monkeypatch.setenv("OMLX_FA256_DISPATCH_BUDGET", "12345")

    assert patch.apply_qwen35_fa256_attention_patch(min_kv_len=16)
    q, k, v = _qkv(32, 32)
    base.scaled_dot_product_attention(q, k, v, None, 0.0625, "causal")
    assert calls == [12345]


def test_dispatch_budget_auto_calibration_used_when_env_unset(monkeypatch):
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
        patch._fa256_fast, "fa256_supports_dispatch_budget", lambda: True
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
        patch._fa256_fast, "fa256_supports_dispatch_budget", lambda: False
    )
    monkeypatch.setattr(patch.mx.metal, "is_available", lambda: True)

    assert patch.apply_qwen35_fa256_attention_patch(min_kv_len=16)
    q, k, v = _qkv(32, 32)
    base.scaled_dot_product_attention(q, k, v, None, 0.0625, "causal")
    assert calls == [0]


def test_apply_skips_on_nax_gpu(monkeypatch):
    # On NAX GPUs stock SDPA's unfused head_dim-256 prefill runs its matmuls
    # on the tensor units and beats the pre-NAX steel kernel (M5 Max report:
    # 4k pp 828 -> 400 tok/s), so the auto mode must not install the patch.
    import omlx.patches.qwen35_fa256_attention as patch

    monkeypatch.setattr(patch, "is_nax_available", lambda: True)
    assert patch.apply_qwen35_fa256_attention_patch() is False


def test_apply_env_forces_steel_on_nax_gpu(monkeypatch):
    import omlx.patches.qwen35_fa256_attention as patch

    _install_fake_vlm_base(monkeypatch)
    monkeypatch.setattr(patch, "is_nax_available", lambda: True)
    monkeypatch.setattr(patch, "_native_kernel", lambda: lambda *a, **k: "steel")
    monkeypatch.setenv("OMLX_FA256_STEEL", "1")
    assert patch.apply_qwen35_fa256_attention_patch() is True


def test_apply_env_kill_switch_wins(monkeypatch):
    import omlx.patches.qwen35_fa256_attention as patch

    monkeypatch.setattr(patch, "is_nax_available", lambda: False)
    monkeypatch.setenv("OMLX_FA256_STEEL", "0")
    assert patch.apply_qwen35_fa256_attention_patch() is False


def test_qwen_native_symbols_are_not_registered_on_glm_extension():
    from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast

    assert not glm_fast.has_symbol("qwen35_fa256_attention")
    assert not glm_fast.has_symbol("qwen35_q4_affine_qmm_t")
    assert not glm_fast.has_symbol("qwen35_q5_affine_qmm_t")
    assert not glm_fast.has_symbol("qwen35_q6_affine_qmm_t")
    assert not glm_fast.has_symbol("qwen35_q8_affine_qmm_t")
    assert not glm_fast.has_symbol("qwen35_moe_weighted_sum")


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal is required")
def test_native_fa256_matches_mlx_reference_small():
    from omlx.custom_kernels.qwen35_prefill import fast

    if not fast.has_symbol("qwen35_fa256_attention"):
        pytest.skip("native qwen35_fa256_attention is unavailable")

    q, k, v = _qkv(128)
    scale = 1.0 / math.sqrt(256)
    out = fast.qwen35_fa256_attention(q, k, v, scale, causal=True)
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
    from omlx.custom_kernels.qwen35_prefill import fast

    if not fast.has_symbol("qwen35_fa256_attention"):
        pytest.skip("native qwen35_fa256_attention is unavailable")
    if not fast.fa256_supports_dispatch_budget():
        pytest.skip("extension predates chunked dispatch")

    budget = patch._auto_dispatch_budget(fast.qwen35_fa256_attention, 32, 8)
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
    from omlx.custom_kernels.qwen35_prefill import fast

    if not fast.has_symbol("qwen35_fa256_attention"):
        pytest.skip("native qwen35_fa256_attention is unavailable")
    if not fast.fa256_supports_dispatch_budget():
        pytest.skip("extension predates chunked dispatch")

    q, k, v = _qkv(q_len, kv_len)
    scale = 1.0 / math.sqrt(256)
    single = fast.qwen35_fa256_attention(q, k, v, scale, causal=True, dispatch_budget=0)
    # Budget forcing ~8 chunks for these shapes.
    budget = (24 * q_len * kv_len) // 8
    chunked = fast.qwen35_fa256_attention(
        q, k, v, scale, causal=True, dispatch_budget=budget
    )
    mx.eval(single, chunked)

    diff = mx.max(mx.abs(single.astype(mx.float32) - chunked.astype(mx.float32))).item()
    assert not mx.isnan(chunked.astype(mx.float32)).any().item()
    assert diff < 5e-3
