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

SEQ = 4096


def _mxfp8_linear(in_features, out_features):
    return nn.QuantizedLinear(
        in_features, out_features, bias=False, group_size=32, bits=8, mode="mxfp8"
    )


def _affine_q8_linear(in_features, out_features):
    return nn.QuantizedLinear(
        in_features, out_features, bias=False, group_size=64, bits=8, mode="affine"
    )


def _mxfp8_grouped_linear(groups, in_features, out_features):
    dense = mx.random.normal((groups, out_features, in_features)).astype(mx.bfloat16)
    weight, scales = mx.quantize(dense, group_size=32, bits=8, mode="mxfp8")
    return SimpleNamespace(
        weight=weight,
        scales=scales,
        group_size=32,
        bits=8,
        mode="mxfp8",
        bias=None,
    )


def _attach(owner, prep, model0=None, model1=None):
    state, _, _ = prep
    owner._omlx_ane_config = ane_patch._AneConfig(SEQ)
    owner._omlx_ane_profile_category = (
        ane_patch._PROFILE_MLP
        if isinstance(state, ane_patch._MlpState)
        else ane_patch._PROFILE_ATTENTION_INPUT
        if isinstance(state, ane_patch._AttentionInputState)
        else ane_patch._PROFILE_WO_A
        if isinstance(state, ane_patch._GroupedLinearState)
        else ane_patch._PROFILE_QUERY
    )
    owner._omlx_ane_state = replace(
        state, model=model0 or object(), model1=model1 or object()
    )


def test_profile_snapshot_merges_expanded_native_and_python_metrics(monkeypatch):
    width = len(fast._ANE_PROFILE_NATIVE_KEYS)
    values = [0.0] * (len(fast._ANE_PROFILE_CATEGORIES) * width)
    query_offset = fast._ANE_PROFILE_CATEGORY_IDS["deepseek_query"] * width
    values[query_offset] = 3.0
    fake_ext = SimpleNamespace(
        qwen35_ane_profile_set_enabled=lambda enabled: None,
        qwen35_ane_profile_reset=lambda: None,
        qwen35_ane_profile_snapshot=lambda: values,
    )
    monkeypatch.setattr(fast, "_ext", fake_ext)
    monkeypatch.setattr(fast, "_ane_profile_aux_enabled", False)

    assert fast.qwen35_ane_profile_set_enabled(True)
    fast.qwen35_ane_profile_reset()
    fast.qwen35_ane_profile_record(
        "deepseek_query", exact_operations=2, logical_tokens=8192
    )
    snapshot = fast.qwen35_ane_profile_snapshot()

    assert snapshot["deepseek_query"]["operations"] == 3
    assert snapshot["deepseek_query"]["exact_operations"] == 2
    assert snapshot["deepseek_query"]["logical_tokens"] == 8192
    assert snapshot["mlp"]["operations"] == 0


def test_profile_category_falls_back_safely_for_legacy_extension(monkeypatch):
    monkeypatch.setattr(fast, "_ext", SimpleNamespace())
    assert fast.qwen35_ane_profile_category_id("deepseek_query") == 1

    monkeypatch.setattr(
        fast,
        "_ext",
        SimpleNamespace(qwen35_ane_profile_category_count=lambda: 7),
    )
    assert fast.qwen35_ane_profile_category_id("deepseek_query") == 5


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

    def fake_dual(
        x, weight, scales, biases, model0, model1, bits, variant, gs, category
    ):
        calls.append((bits, gs, int(weight.shape[0]), category))
        return marker

    monkeypatch.setattr(fast, "qwen35_ane_dual_affine_qmm_t", fake_dual)
    x = mx.zeros((1, SEQ, 256), dtype=mx.bfloat16)

    out = ane_patch._linear_backend(linear, x)

    assert out is marker
    assert calls == [
        (8, 64, 256, fast.qwen35_ane_profile_category_id("deepseek_query"))
    ]
    # DSpark verify arming and decode shapes fall through to the GPU path.
    decode_consistency.set_armed(True)
    try:
        assert ane_patch._linear_backend(linear, x) is None
    finally:
        decode_consistency.set_armed(False)
    assert ane_patch._linear_backend(linear, x[:, :4]) is None
    assert ane_patch._linear_backend(linear, x.astype(mx.float32)) is None


def test_linear_backend_dispatches_shared_cpu_middle(monkeypatch):
    linear = _mxfp8_linear(256, 1024)
    prep = ane_patch._prepare_linear(linear, 0.5, 0.125)
    _attach(linear, prep)
    marker = mx.zeros((1, SEQ, 1024), dtype=mx.bfloat16)
    calls = []

    def fake_hybrid(*args):
        calls.append(args)
        return marker

    monkeypatch.setattr(fast, "qwen35_ane_dual_cpu_fp16_affine_qmm_t", fake_hybrid)
    x = mx.zeros((1, SEQ, 256), dtype=mx.bfloat16)

    assert ane_patch._linear_backend(linear, x) is marker
    assert len(calls) == 1
    assert calls[0][1].shape == (128, 256)
    assert calls[0][-2:] == (12, True)


def test_linear_backend_pads_profitable_short_tile_and_slices(monkeypatch):
    linear = _mxfp8_linear(256, 512)
    prep = ane_patch._prepare_linear(linear, 0.5)
    _attach(linear, prep)
    linear._omlx_ane_config = ane_patch._AneConfig(
        SEQ, tail_padding_min_tokens=96
    )
    seen = []
    records = []

    def fake_dual(x, *args):
        seen.append(x)
        return mx.full((1, SEQ, 512), 7, dtype=x.dtype)

    monkeypatch.setattr(fast, "qwen35_ane_dual_affine_qmm_t", fake_dual)
    monkeypatch.setattr(
        fast,
        "qwen35_ane_profile_record",
        lambda category, **values: records.append((category, values)),
    )
    x = mx.ones((1, 100, 256), dtype=mx.bfloat16)

    result = ane_patch._linear_backend(linear, x)
    assert result is not None
    mx.eval(result, *seen)

    assert result.shape == (1, 100, 512)
    assert seen[0].shape == (1, SEQ, 256)
    assert bool(mx.all(seen[0][:, :100] == 1))
    assert bool(mx.all(seen[0][:, 100:] == 0))
    assert bool(mx.all(result == 7))
    assert records == [
        (
            "deepseek_query",
            {
                "padded_operations": 1,
                "logical_tokens": 100,
                "padded_tokens": SEQ - 100,
            },
        )
    ]


def test_tail_padding_never_intercepts_native_dspark_verify(monkeypatch):
    linear = _mxfp8_linear(256, 512)
    prep = ane_patch._prepare_linear(linear, 0.5)
    _attach(linear, prep)
    linear._omlx_ane_config = ane_patch._AneConfig(
        SEQ, tail_padding_min_tokens=2
    )
    calls = []
    records = []
    monkeypatch.setattr(
        fast,
        "qwen35_ane_dual_affine_qmm_t",
        lambda *args: calls.append(args),
    )
    monkeypatch.setattr(
        fast,
        "qwen35_ane_profile_record",
        lambda category, **values: records.append((category, values)),
    )

    decode_consistency.set_armed(True)
    try:
        assert (
            ane_patch._linear_backend(
                linear, mx.ones((1, 6, 256), dtype=mx.bfloat16)
            )
            is None
        )
    finally:
        decode_consistency.set_armed(False)
    assert calls == []
    assert records == [
        (
            "deepseek_query",
            {"fallback_operations": 1, "dspark_bypasses": 1},
        )
    ]


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
    monkeypatch.setattr(fast, "qwen35_ane_dual_affine_qmm_t", lambda *args: combined)
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


def test_mlp_backend_routes_activation_through_prepared_down(monkeypatch):
    mlp = SimpleNamespace(
        gate_proj=_mxfp8_linear(256, 512),
        up_proj=_mxfp8_linear(256, 512),
        down_proj=_mxfp8_linear(512, 512),
        swiglu_limit=10.0,
        fp32_swiglu=False,
    )
    prep = ane_patch._prepare_mlp(mlp)
    assert prep is not None
    _attach(mlp, prep)
    state = mlp._omlx_ane_state
    total = 2 * state.ane_outputs + 2 * state.gpu_outputs
    combined = mx.ones((1, SEQ, total), dtype=mx.bfloat16)
    down_marker = mx.zeros((1, SEQ, 512), dtype=mx.bfloat16)
    monkeypatch.setattr(fast, "qwen35_ane_dual_affine_qmm_t", lambda *args: combined)
    monkeypatch.setattr(ane_patch, "_linear_backend", lambda linear, x: down_marker)
    monkeypatch.setitem(
        sys.modules,
        "mlx_lm.models.deepseek_v4",
        SimpleNamespace(_limited_swiglu=lambda gate, up, limit: gate),
    )

    assert (
        ane_patch._mlp_backend(mlp, mx.zeros((1, SEQ, 256), dtype=mx.bfloat16))
        is down_marker
    )


def test_prepare_grouped_linear_keeps_group_blocks_compact():
    linear = _mxfp8_grouped_linear(2, 256, 512)

    prep = ane_patch._prepare_grouped_linear(linear, 0.5)

    assert prep is not None
    state, dense0, dense1 = prep
    assert state.groups == 2
    assert state.ane_outputs == 256
    assert state.gpu_outputs == 256
    assert dense0.shape == dense1.shape == (256, 256)
    # The suffix stores only each group's real K-wide block, not a sparse
    # block-diagonal [groups*K] expansion.
    assert state.weight.shape == (2 * 256, 256 // 4)


def test_grouped_backend_dispatches_and_guards_shape(monkeypatch):
    linear = _mxfp8_grouped_linear(2, 256, 512)
    prep = ane_patch._prepare_grouped_linear(linear, 0.5)
    assert prep is not None
    _attach(linear, prep)
    marker = mx.zeros((1, 2, SEQ, 512), dtype=mx.bfloat16)
    calls = []

    def fake_grouped(*args):
        calls.append(args)
        return marker

    monkeypatch.setattr(fast, "qwen35_ane_dual_grouped_affine_qmm_t", fake_grouped)
    x = mx.zeros((1, 2, SEQ, 256), dtype=mx.bfloat16)

    assert ane_patch._grouped_backend(linear, x) is marker
    assert calls[0][6:] == (
        2,
        8,
        8,
        64,
        fast.qwen35_ane_profile_category_id("deepseek_wo_a"),
    )
    assert ane_patch._grouped_backend(linear, x[:, :, :4]) is None
    decode_consistency.set_armed(True)
    try:
        assert ane_patch._grouped_backend(linear, x) is None
    finally:
        decode_consistency.set_armed(False)


def test_grouped_backend_pads_sequence_axis_only(monkeypatch):
    linear = _mxfp8_grouped_linear(2, 256, 512)
    prep = ane_patch._prepare_grouped_linear(linear, 0.5)
    assert prep is not None
    _attach(linear, prep)
    linear._omlx_ane_config = ane_patch._AneConfig(
        SEQ, tail_padding_min_tokens=96
    )
    seen = []

    def fake_grouped(x, *args):
        seen.append(x)
        return mx.zeros((1, 2, SEQ, 512), dtype=x.dtype)

    monkeypatch.setattr(fast, "qwen35_ane_dual_grouped_affine_qmm_t", fake_grouped)
    result = ane_patch._grouped_backend(
        linear, mx.ones((1, 2, 100, 256), dtype=mx.bfloat16)
    )
    assert result is not None
    mx.eval(result, *seen)

    assert seen[0].shape == (1, 2, SEQ, 256)
    assert result.shape == (1, 2, 100, 512)


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

    prep = ane_patch._prepare_stacked(attn_linear, indexer_linear, 0.5, 0.125)

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
    monkeypatch.setattr(fast, "qwen35_ane_dual_affine_qmm_t", lambda *args: combined)
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


def _fake_attention_inputs(*, sparse=False):
    attn = SimpleNamespace(
        wq_a=_mxfp8_linear(256, 512),
        wkv=_mxfp8_linear(256, 256),
    )
    if sparse:
        attn.compressor = SimpleNamespace(
            wkv=_affine_q8_linear(256, 256),
            wgate=_affine_q8_linear(256, 256),
        )
        attn.indexer = SimpleNamespace(
            compressor=SimpleNamespace(
                wkv=_affine_q8_linear(256, 128),
                wgate=_affine_q8_linear(256, 128),
            ),
            weights_proj=_affine_q8_linear(256, 64),
        )
    return attn


def test_prepare_attention_input_stacks_mixed_quantized_linears():
    attn = _fake_attention_inputs(sparse=True)

    prep = ane_patch._prepare_attention_input(attn)

    assert prep is not None
    state, dense0, dense1 = prep
    assert isinstance(state, ane_patch._AttentionInputState)
    assert state.segments == (
        ("wq_a", 512),
        ("wkv", 256),
        ("compressor_wkv", 256),
        ("compressor_wgate", 256),
        ("indexer_compressor_wkv", 128),
        ("indexer_compressor_wgate", 128),
        ("indexer_weights", 64),
    )
    assert state.ane_outputs == 768
    assert state.gpu_outputs == 832
    assert dense0.shape == (384, 256)
    assert dense1.shape == (384, 256)
    assert dense0.dtype == dense1.dtype == mx.float32
    assert state.cpu_weight is None

    reference = mx.concatenate(
        [
            ane_patch._dequant_quantized_rows(linear, 0, size)
            for (_, linear), (_, size) in zip(
                ane_patch._attention_input_linears(attn), state.segments
            )
        ]
    )
    suffix = mx.dequantize(
        state.weight,
        state.scales,
        state.biases,
        group_size=64,
        bits=8,
    )
    assert float(mx.abs(dense0 - reference[:384].astype(mx.float32)).max()) < 1e-3
    assert float(mx.abs(suffix - reference[768:]).max()) < 0.05


def test_attention_input_backend_restores_named_segments(monkeypatch):
    attn = _fake_attention_inputs()
    prep = ane_patch._prepare_attention_input(attn)
    assert prep is not None
    _attach(attn, prep)
    total = sum(size for _, size in attn._omlx_ane_state.segments)
    combined = mx.broadcast_to(mx.arange(total, dtype=mx.float32), (1, SEQ, total))
    monkeypatch.setattr(fast, "qwen35_ane_dual_affine_qmm_t", lambda *args: combined)
    x = mx.zeros((1, SEQ, 256), dtype=mx.bfloat16)

    outputs = ane_patch._attention_input_backend(attn, x)

    assert outputs is not None
    assert tuple(outputs) == ("wq_a", "wkv")
    assert outputs["wq_a"].shape[-1] == 512
    assert outputs["wkv"].shape[-1] == 256
    assert int(outputs["wkv"][0, 0, 0]) == 512
    assert ane_patch._attention_input_backend(attn, x[:, :4]) is None


def _fake_layer():
    attn = _fake_attention_inputs()
    attn.wq_b = _mxfp8_linear(256, 512)
    attn.wo_b = _mxfp8_linear(256, 768)
    return SimpleNamespace(
        attn=attn,
        ffn=SimpleNamespace(
            shared_experts=SimpleNamespace(
                gate_proj=_mxfp8_linear(256, 512),
                up_proj=_mxfp8_linear(256, 512),
                down_proj=_mxfp8_linear(512, 512),
            )
        ),
    )


def test_enable_compiles_banks_and_registers_backends(monkeypatch):
    monkeypatch.delenv("OMLX_QWEN35_ANE_PREFILL", raising=False)
    monkeypatch.setattr(ane_patch, "is_nax_available", lambda: False)
    monkeypatch.setattr(fast, "qwen35_ane_available", lambda: True)
    monkeypatch.setattr(fast, "has_symbol", lambda name: True)
    monkeypatch.setattr(fast, "qwen35_cpu_shared_resource_available", lambda: True)
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
            register_ane_mlp_backend=(lambda fn: registered.__setitem__("mlp", fn)),
            register_ane_attention_input_backend=(
                lambda fn: registered.__setitem__("attention_input", fn)
            ),
        ),
    )
    layer = _fake_layer()
    model = SimpleNamespace(model=SimpleNamespace(layers=[layer]))

    count = ane_patch.enable_deepseek_v4_ane_prefill(
        model,
        sequence_length=SEQ,
        tail_padding_min_tokens=3000,
    )

    # Shared gate/up, shared down, attention-input stack, and plain wq_b;
    # wo_b is excluded.
    assert count == 4
    assert compiled == {"count": 4, "sequence_length": SEQ}
    assert registered["linear"] is ane_patch._linear_backend
    assert registered["mlp"] is ane_patch._mlp_backend
    assert registered["attention_input"] is ane_patch._attention_input_backend
    assert isinstance(layer.attn._omlx_ane_state, ane_patch._AttentionInputState)
    assert layer.attn.wq_b._omlx_ane_state.model is not None
    assert not hasattr(layer.attn.wo_b, "_omlx_ane_state")
    assert layer.ffn.shared_experts._omlx_ane_state.ane_outputs == 256
    assert layer.ffn.shared_experts.down_proj._omlx_ane_state.ane_outputs == 256
    assert layer.ffn.shared_experts._omlx_ane_state.cpu_weight is None
    assert layer.attn.wq_b._omlx_ane_state.cpu_weight is not None
    assert model._omlx_ane_procedure_count == 4
    assert model._omlx_ane_cpu_prefill_count == 1
    assert model._omlx_ane_mlp_prefill_count == 1
    assert model._omlx_ane_attention_input_prefill_count == 1
    assert model._omlx_ane_down_prefill_count == 1
    assert model._omlx_ane_wo_a_prefill_count == 0
    assert model._omlx_ane_query_prefill_count == 1
    assert layer.ffn.shared_experts._omlx_ane_profile_category == "deepseek_mlp"
    assert (
        layer.ffn.shared_experts.down_proj._omlx_ane_profile_category
        == "deepseek_down"
    )
    assert layer.attn._omlx_ane_profile_category == "deepseek_attention_input"
    assert layer.attn.wq_b._omlx_ane_profile_category == "deepseek_query"
    assert model._omlx_ane_tail_padding_min_tokens == 3000
    assert model._omlx_ane_dspark_native_compatible is True
    assert layer.attn.wq_b._omlx_ane_config.tail_padding_min_tokens == 3000
    assert model._omlx_ane_resident_program_count == 2


def test_enable_compiles_grouped_wo_a_bank_and_registers_backend(monkeypatch):
    monkeypatch.delenv("OMLX_QWEN35_ANE_PREFILL", raising=False)
    monkeypatch.setattr(ane_patch, "is_nax_available", lambda: False)
    monkeypatch.setattr(fast, "qwen35_ane_available", lambda: True)
    monkeypatch.setattr(fast, "has_symbol", lambda name: True)
    monkeypatch.setattr(fast, "qwen35_cpu_shared_resource_available", lambda: True)
    monkeypatch.setattr(
        ane_patch,
        "_compile_dual_banks",
        lambda w0, w1, seq: ([object() for _ in w0], [object() for _ in w1], 2),
    )
    grouped_compile = {}

    def fake_grouped(w0, w1, seq, groups):
        grouped_compile.update(count=len(w0), sequence_length=seq, groups=groups)
        return [object() for _ in w0], [object() for _ in w1], 2

    monkeypatch.setattr(ane_patch, "_compile_grouped_dual_banks", fake_grouped)
    registered = {}
    monkeypatch.setitem(
        sys.modules,
        "mlx_lm.models.deepseek_v4",
        SimpleNamespace(
            register_ane_linear_backend=lambda fn: None,
            register_ane_mlp_backend=lambda fn: None,
            register_ane_grouped_linear_backend=(
                lambda fn: registered.__setitem__("grouped", fn)
            ),
        ),
    )
    layer = _fake_layer()
    layer.attn.wo_a = _mxfp8_grouped_linear(2, 256, 512)
    model = SimpleNamespace(model=SimpleNamespace(layers=[layer]))

    count = ane_patch.enable_deepseek_v4_ane_prefill(model, sequence_length=SEQ)

    assert count == 5
    assert grouped_compile == {"count": 1, "sequence_length": SEQ, "groups": 2}
    assert registered["grouped"] is ane_patch._grouped_backend
    assert isinstance(layer.attn.wo_a._omlx_ane_state, ane_patch._GroupedLinearState)
    assert model._omlx_ane_wo_a_prefill_count == 1
    assert model._omlx_ane_resident_program_count == 4


def test_enable_can_disable_down_and_grouped_wo_a_independently(monkeypatch):
    monkeypatch.delenv("OMLX_QWEN35_ANE_PREFILL", raising=False)
    monkeypatch.setattr(ane_patch, "is_nax_available", lambda: False)
    monkeypatch.setattr(fast, "qwen35_ane_available", lambda: True)
    monkeypatch.setattr(fast, "has_symbol", lambda name: True)
    monkeypatch.setattr(fast, "qwen35_cpu_shared_resource_available", lambda: True)
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
            register_ane_attention_input_backend=lambda fn: None,
        ),
    )
    layer = _fake_layer()
    layer.attn.wo_a = _mxfp8_grouped_linear(2, 256, 512)
    model = SimpleNamespace(model=SimpleNamespace(layers=[layer]))

    count = ane_patch.enable_deepseek_v4_ane_prefill(
        model,
        sequence_length=SEQ,
        down_enabled=False,
        wo_a_enabled=False,
    )

    assert count == 3
    assert model._omlx_ane_down_prefill_count == 0
    assert model._omlx_ane_wo_a_prefill_count == 0
    assert not hasattr(layer.ffn.shared_experts.down_proj, "_omlx_ane_state")
    assert not hasattr(layer.attn.wo_a, "_omlx_ane_state")


def test_enable_applies_tuned_down_and_grouped_wo_a_fractions(monkeypatch):
    monkeypatch.delenv("OMLX_QWEN35_ANE_PREFILL", raising=False)
    monkeypatch.setattr(ane_patch, "is_nax_available", lambda: False)
    monkeypatch.setattr(fast, "qwen35_ane_available", lambda: True)
    monkeypatch.setattr(fast, "has_symbol", lambda name: True)
    monkeypatch.setattr(fast, "qwen35_cpu_shared_resource_available", lambda: True)
    monkeypatch.setattr(
        ane_patch,
        "_compile_dual_banks",
        lambda w0, w1, seq: ([object() for _ in w0], [object() for _ in w1], 2),
    )
    monkeypatch.setattr(
        ane_patch,
        "_compile_grouped_dual_banks",
        lambda w0, w1, seq, groups: (
            [object() for _ in w0],
            [object() for _ in w1],
            2,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_lm.models.deepseek_v4",
        SimpleNamespace(
            register_ane_linear_backend=lambda fn: None,
            register_ane_mlp_backend=lambda fn: None,
            register_ane_attention_input_backend=lambda fn: None,
            register_ane_grouped_linear_backend=lambda fn: None,
        ),
    )
    layer = _fake_layer()
    layer.ffn.shared_experts.down_proj = _mxfp8_linear(512, 1024)
    layer.attn.wo_a = _mxfp8_grouped_linear(2, 256, 1024)
    model = SimpleNamespace(model=SimpleNamespace(layers=[layer]))

    ane_patch.enable_deepseek_v4_ane_prefill(
        model,
        sequence_length=SEQ,
        down_fraction=0.75,
        wo_a_fraction=0.75,
    )

    assert layer.ffn.shared_experts.down_proj._omlx_ane_state.ane_outputs == 768
    assert layer.attn.wo_a._omlx_ane_state.ane_outputs == 768


def test_enable_stacks_indexer_wq_b_when_present(monkeypatch):
    monkeypatch.delenv("OMLX_QWEN35_ANE_PREFILL", raising=False)
    monkeypatch.setattr(ane_patch, "is_nax_available", lambda: False)
    monkeypatch.setattr(fast, "qwen35_ane_available", lambda: True)
    monkeypatch.setattr(fast, "has_symbol", lambda name: True)
    monkeypatch.setattr(fast, "qwen35_cpu_shared_resource_available", lambda: True)
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

    # Default targets: shared gate/up + down + attention input + stacked wq_b.
    assert count == 4
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
    monkeypatch.setattr(fast, "qwen35_cpu_shared_resource_available", lambda: False)
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

    assert ane_patch.enable_deepseek_v4_ane_prefill(model, sequence_length=SEQ) == 4
    assert layer.attn.wq_b._omlx_ane_state.cpu_weight is None
    assert model._omlx_ane_cpu_prefill_count == 0


def test_enable_disables_cpu_share_below_measured_tile(monkeypatch):
    monkeypatch.delenv("OMLX_QWEN35_ANE_PREFILL", raising=False)
    monkeypatch.setattr(ane_patch, "is_nax_available", lambda: False)
    monkeypatch.setattr(fast, "qwen35_ane_available", lambda: True)
    monkeypatch.setattr(fast, "has_symbol", lambda name: True)
    monkeypatch.setattr(fast, "qwen35_cpu_shared_resource_available", lambda: True)
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

    assert (
        ane_patch.enable_deepseek_v4_ane_prefill(
            model,
            sequence_length=2048,
            cpu_fraction=0.125,
        )
        == 4
    )
    assert layer.attn.wq_b._omlx_ane_state.cpu_weight is None
    assert model._omlx_ane_cpu_prefill_count == 0


def test_enable_skips_on_nax_gpu(monkeypatch):
    monkeypatch.delenv("OMLX_QWEN35_ANE_PREFILL", raising=False)
    monkeypatch.setattr(ane_patch, "is_nax_available", lambda: True)
    model = SimpleNamespace(model=SimpleNamespace(layers=[_fake_layer()]))

    assert ane_patch.enable_deepseek_v4_ane_prefill(model, sequence_length=SEQ) == 0
    assert not hasattr(model, "_omlx_ane_procedure_count")
