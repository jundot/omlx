# SPDX-License-Identifier: Apache-2.0
"""Tests for the DeepSeek-V4 ANE prefill patch."""

import sys
from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn

import omlx.patches.deepseek_v4.ane_prefill as ane_patch
from omlx.custom_kernels.qwen35_prefill import fast
from omlx.patches.deepseek_v4 import decode_consistency

SEQ = 2048


def _mxfp8_linear(in_features, out_features):
    return nn.QuantizedLinear(
        in_features, out_features, bias=False, group_size=32, bits=8, mode="mxfp8"
    )


def _attach(owner, prep, model0=None, model1=None):
    state, _, _ = prep
    owner._omlx_ane_config = ane_patch._AneConfig(SEQ)
    owner._omlx_ane_state = replace(
        state, model=model0 or object(), model1=model1 or object()
    )


def test_prepare_linear_splits_mxfp8_rows():
    linear = _mxfp8_linear(256, 512)

    prep = ane_patch._prepare_linear(linear, 0.5)

    assert prep is not None
    state, dense0, dense1 = prep
    assert state.ane_outputs == 256
    assert state.gpu_outputs == 256
    assert dense0.shape == (128, 256)
    assert dense1.shape == (128, 256)
    assert dense0.dtype == mx.float32
    # The suffix is an affine q8 requant of the trailing rows.
    assert state.weight.dtype == mx.uint32
    assert state.weight.shape == (256, 64)
    assert state.scales.shape == (256, 256 // 64)
    reference = mx.dequantize(
        linear.weight, linear.scales, group_size=32, bits=8, mode="mxfp8"
    )
    suffix = mx.dequantize(
        state.weight, state.scales, state.biases, group_size=64, bits=8
    )
    assert float(mx.abs(suffix - reference[256:]).max()) < 0.05


def test_prepare_linear_adds_fp16_cpu_middle_rows():
    linear = _mxfp8_linear(256, 1024)

    prep = ane_patch._prepare_linear(linear, 0.5, 0.125)

    assert prep is not None
    state, _, _ = prep
    assert state.ane_outputs == 512
    assert state.cpu_outputs == 128
    assert state.gpu_outputs == 384
    assert state.cpu_weight is not None
    assert state.cpu_weight.shape == (128, 256)
    assert state.cpu_weight.dtype == mx.float16
    reference = mx.dequantize(
        linear.weight, linear.scales, group_size=32, bits=8, mode="mxfp8"
    )
    assert (
        float(
            mx.abs(
                state.cpu_weight.astype(mx.float32)
                - reference[512:640].astype(mx.float32)
            ).max()
        )
        < 0.01
    )


def test_prepare_rejects_affine_and_undersized_linears():
    affine = nn.QuantizedLinear(256, 512, bias=False, group_size=64, bits=4)
    assert ane_patch._prepare_linear(affine, 0.5) is None
    # A 0.4 fraction of 512 rows cannot reach the 128-row instance minimum.
    small = _mxfp8_linear(256, 512)
    assert ane_patch._prepare_linear(small, 0.4) is None


def test_linear_backend_guards_and_dispatch(monkeypatch):
    linear = _mxfp8_linear(256, 512)
    prep = ane_patch._prepare_linear(linear, 0.5)
    _attach(linear, prep)
    marker = mx.zeros((1, SEQ, 512), dtype=mx.bfloat16)
    calls = []

    def fake_dual(x, weight, scales, biases, model0, model1, bits, variant, gs):
        calls.append((bits, gs, int(weight.shape[0])))
        return marker

    monkeypatch.setattr(fast, "qwen35_ane_dual_affine_qmm_t", fake_dual)
    x = mx.zeros((1, SEQ, 256), dtype=mx.bfloat16)

    out = ane_patch._linear_backend(linear, x)

    assert out is marker
    assert calls == [(8, 64, 256)]
    # DSpark verify arming and decode shapes fall through to the GPU path.
    decode_consistency.set_armed(True)
    try:
        assert ane_patch._linear_backend(linear, x) is None
    finally:
        decode_consistency.set_armed(False)
    assert ane_patch._linear_backend(linear, x[:, :4]) is None
    assert (
        ane_patch._linear_backend(linear, x.astype(mx.float32)) is None
    )


def test_linear_backend_dispatches_shared_cpu_middle(monkeypatch):
    linear = _mxfp8_linear(256, 1024)
    prep = ane_patch._prepare_linear(linear, 0.5, 0.125)
    _attach(linear, prep)
    marker = mx.zeros((1, SEQ, 1024), dtype=mx.bfloat16)
    calls = []

    def fake_hybrid(*args):
        calls.append(args)
        return marker

    monkeypatch.setattr(
        fast, "qwen35_ane_dual_cpu_fp16_affine_qmm_t", fake_hybrid
    )
    x = mx.zeros((1, SEQ, 256), dtype=mx.bfloat16)

    assert ane_patch._linear_backend(linear, x) is marker
    assert len(calls) == 1
    assert calls[0][1].shape == (128, 256)
    assert calls[0][-2:] == (12, True)


def test_mlp_backend_reassembles_gate_and_up(monkeypatch):
    mlp = SimpleNamespace(
        gate_proj=_mxfp8_linear(256, 512),
        up_proj=_mxfp8_linear(256, 512),
        down_proj=lambda hidden: hidden,
        swiglu_limit=10.0,
        fp32_swiglu=False,
    )
    prep = ane_patch._prepare_mlp(mlp)
    assert prep is not None
    _attach(mlp, prep)
    state = mlp._omlx_ane_state
    half = state.ane_outputs // 2
    total = 2 * state.ane_outputs + 2 * state.gpu_outputs
    # Column-coded combined output so the reassembly order is observable.
    # Stays float32: bfloat16 cannot represent the larger column codes.
    combined = mx.broadcast_to(mx.arange(total, dtype=mx.float32), (1, SEQ, total))
    monkeypatch.setattr(
        fast, "qwen35_ane_dual_affine_qmm_t", lambda *args: combined
    )
    captured = {}

    def fake_swiglu(gate, up, limit):
        captured["gate"] = gate[0, 0]
        captured["up"] = up[0, 0]
        captured["limit"] = limit
        return gate

    monkeypatch.setitem(
        sys.modules,
        "mlx_lm.models.deepseek_v4",
        SimpleNamespace(_limited_swiglu=fake_swiglu),
    )
    x = mx.zeros((1, SEQ, 256), dtype=mx.bfloat16)

    out = ane_patch._mlp_backend(mlp, x)

    assert out is not None
    gpu = state.gpu_outputs
    expected_gate = (
        list(range(0, half))
        + list(range(2 * half, 3 * half))
        + list(range(4 * half, 4 * half + gpu))
    )
    expected_up = (
        list(range(half, 2 * half))
        + list(range(3 * half, 4 * half))
        + list(range(4 * half + gpu, total))
    )
    assert [int(v) for v in captured["gate"].tolist()] == expected_gate
    assert [int(v) for v in captured["up"].tolist()] == expected_up
    assert captured["limit"] == 10.0


def test_prepare_stacked_spans_both_linears():
    attn_linear = _mxfp8_linear(256, 512)
    indexer_linear = _mxfp8_linear(256, 256)

    prep = ane_patch._prepare_stacked(attn_linear, indexer_linear, 0.5)

    assert prep is not None
    state, dense0, dense1 = prep
    assert state.ane_outputs == 256
    assert state.gpu_outputs == 512
    assert state.split == 512
    attn_ref = mx.dequantize(
        attn_linear.weight, attn_linear.scales, group_size=32, bits=8, mode="mxfp8"
    )
    indexer_ref = mx.dequantize(
        indexer_linear.weight,
        indexer_linear.scales,
        group_size=32,
        bits=8,
        mode="mxfp8",
    )
    assert float(mx.abs(dense0 - attn_ref[:128].astype(mx.float32)).max()) < 1e-3
    # The requantized suffix crosses the attn/indexer boundary.
    suffix = mx.dequantize(
        state.weight, state.scales, state.biases, group_size=64, bits=8
    )
    stacked_ref = mx.concatenate((attn_ref[256:], indexer_ref))
    assert float(mx.abs(suffix - stacked_ref).max()) < 0.05


def test_prepare_stacked_cpu_rows_preserve_combined_order():
    attn_linear = _mxfp8_linear(256, 1024)
    indexer_linear = _mxfp8_linear(256, 1024)

    prep = ane_patch._prepare_stacked(
        attn_linear, indexer_linear, 0.5, 0.125
    )

    assert prep is not None
    state, _, _ = prep
    assert state.ane_outputs == 1024
    assert state.cpu_outputs == 256
    assert state.gpu_outputs == 768
    assert state.cpu_weight is not None
    stacked_ref = mx.concatenate(
        (
            mx.dequantize(
                attn_linear.weight,
                attn_linear.scales,
                group_size=32,
                bits=8,
                mode="mxfp8",
            ),
            mx.dequantize(
                indexer_linear.weight,
                indexer_linear.scales,
                group_size=32,
                bits=8,
                mode="mxfp8",
            ),
        )
    )
    assert (
        float(
            mx.abs(
                state.cpu_weight.astype(mx.float32)
                - stacked_ref[1024:1280].astype(mx.float32)
            ).max()
        )
        < 0.01
    )


def test_stacked_backend_splits_and_plain_backend_ignores_it(monkeypatch):
    attn_linear = _mxfp8_linear(256, 512)
    indexer_linear = _mxfp8_linear(256, 256)
    prep = ane_patch._prepare_stacked(attn_linear, indexer_linear, 0.5)
    _attach(attn_linear, prep)
    total = 768
    combined = mx.broadcast_to(mx.arange(total, dtype=mx.float32), (1, SEQ, total))
    monkeypatch.setattr(
        fast, "qwen35_ane_dual_affine_qmm_t", lambda *args: combined
    )
    x = mx.zeros((1, SEQ, 256), dtype=mx.bfloat16)

    split = ane_patch._stacked_backend(attn_linear, x)

    assert split is not None
    attn_q, indexer_q = split
    assert attn_q.shape[-1] == 512
    assert indexer_q.shape[-1] == 256
    assert int(indexer_q[0, 0, 0]) == 512
    # A stacked state must not satisfy the plain linear backend, or the
    # fallback path would return 768 columns where 512 are expected.
    assert ane_patch._linear_backend(attn_linear, x) is None


def _fake_layer():
    return SimpleNamespace(
        attn=SimpleNamespace(
            wq_b=_mxfp8_linear(256, 512),
            wo_b=_mxfp8_linear(256, 768),
        ),
        ffn=SimpleNamespace(
            shared_experts=SimpleNamespace(
                gate_proj=_mxfp8_linear(256, 512),
                up_proj=_mxfp8_linear(256, 512),
            )
        ),
    )


def test_enable_compiles_banks_and_registers_backends(monkeypatch):
    monkeypatch.delenv("OMLX_QWEN35_ANE_PREFILL", raising=False)
    monkeypatch.setattr(ane_patch, "is_nax_available", lambda: False)
    monkeypatch.setattr(fast, "qwen35_ane_available", lambda: True)
    monkeypatch.setattr(fast, "has_symbol", lambda name: True)
    monkeypatch.setattr(
        fast, "qwen35_cpu_shared_resource_available", lambda: True
    )
    compiled = {}

    def fake_banks(weights0, weights1, sequence_length):
        compiled["count"] = len(weights0)
        compiled["sequence_length"] = sequence_length
        return (
            [object() for _ in weights0],
            [object() for _ in weights1],
            2,
        )

    monkeypatch.setattr(ane_patch, "_compile_dual_banks", fake_banks)
    registered = {}
    monkeypatch.setitem(
        sys.modules,
        "mlx_lm.models.deepseek_v4",
        SimpleNamespace(
            register_ane_linear_backend=(
                lambda fn: registered.__setitem__("linear", fn)
            ),
            register_ane_mlp_backend=(
                lambda fn: registered.__setitem__("mlp", fn)
            ),
        ),
    )
    layer = _fake_layer()
    model = SimpleNamespace(model=SimpleNamespace(layers=[layer]))

    count = ane_patch.enable_deepseek_v4_ane_prefill(model, sequence_length=SEQ)

    # Shared expert plus plain wq_b; wo_b is deliberately not accelerated.
    assert count == 2
    assert compiled == {"count": 2, "sequence_length": SEQ}
    assert registered["linear"] is ane_patch._linear_backend
    assert registered["mlp"] is ane_patch._mlp_backend
    assert layer.attn.wq_b._omlx_ane_state.model is not None
    assert not hasattr(layer.attn.wo_b, "_omlx_ane_state")
    assert layer.ffn.shared_experts._omlx_ane_state.ane_outputs == 256
    assert layer.ffn.shared_experts._omlx_ane_state.cpu_weight is None
    assert layer.attn.wq_b._omlx_ane_state.cpu_weight is not None
    assert model._omlx_ane_procedure_count == 2
    assert model._omlx_ane_cpu_prefill_count == 1
    assert model._omlx_ane_mlp_prefill_count == 1
    assert model._omlx_ane_resident_program_count == 2


def test_enable_stacks_indexer_wq_b_when_present(monkeypatch):
    monkeypatch.delenv("OMLX_QWEN35_ANE_PREFILL", raising=False)
    monkeypatch.setattr(ane_patch, "is_nax_available", lambda: False)
    monkeypatch.setattr(fast, "qwen35_ane_available", lambda: True)
    monkeypatch.setattr(fast, "has_symbol", lambda name: True)
    monkeypatch.setattr(
        fast, "qwen35_cpu_shared_resource_available", lambda: True
    )
    monkeypatch.setattr(
        ane_patch,
        "_compile_dual_banks",
        lambda w0, w1, seq: ([object() for _ in w0], [object() for _ in w1], 2),
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_lm.models.deepseek_v4",
        SimpleNamespace(
            register_ane_linear_backend=lambda fn: None,
            register_ane_mlp_backend=lambda fn: None,
            register_ane_stacked_q_backend=lambda fn: None,
        ),
    )
    layer = _fake_layer()
    layer.attn.indexer = SimpleNamespace(wq_b=_mxfp8_linear(256, 256))
    model = SimpleNamespace(model=SimpleNamespace(layers=[layer]))

    count = ane_patch.enable_deepseek_v4_ane_prefill(model, sequence_length=SEQ)

    # Default targets: shared expert + stacked wq_b (wo_b excluded).
    assert count == 2
    state = layer.attn.wq_b._omlx_ane_state
    assert isinstance(state, ane_patch._StackedState)
    assert state.split == 512
    assert state.cpu_weight is not None
    assert not hasattr(layer.attn.wo_b, "_omlx_ane_state")


def test_enable_disables_cpu_share_without_shared_scheduler(monkeypatch):
    monkeypatch.delenv("OMLX_QWEN35_ANE_PREFILL", raising=False)
    monkeypatch.setattr(ane_patch, "is_nax_available", lambda: False)
    monkeypatch.setattr(fast, "qwen35_ane_available", lambda: True)
    monkeypatch.setattr(fast, "has_symbol", lambda name: True)
    monkeypatch.setattr(
        fast, "qwen35_cpu_shared_resource_available", lambda: False
    )
    monkeypatch.setattr(
        ane_patch,
        "_compile_dual_banks",
        lambda w0, w1, seq: ([object() for _ in w0], [object() for _ in w1], 2),
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_lm.models.deepseek_v4",
        SimpleNamespace(
            register_ane_linear_backend=lambda fn: None,
            register_ane_mlp_backend=lambda fn: None,
        ),
    )
    layer = _fake_layer()
    model = SimpleNamespace(model=SimpleNamespace(layers=[layer]))

    assert ane_patch.enable_deepseek_v4_ane_prefill(model, sequence_length=SEQ) == 2
    assert layer.attn.wq_b._omlx_ane_state.cpu_weight is None
    assert model._omlx_ane_cpu_prefill_count == 0


def test_enable_skips_on_nax_gpu(monkeypatch):
    monkeypatch.delenv("OMLX_QWEN35_ANE_PREFILL", raising=False)
    monkeypatch.setattr(ane_patch, "is_nax_available", lambda: True)
    model = SimpleNamespace(model=SimpleNamespace(layers=[_fake_layer()]))

    assert ane_patch.enable_deepseek_v4_ane_prefill(model, sequence_length=SEQ) == 0
    assert not hasattr(model, "_omlx_ane_procedure_count")
