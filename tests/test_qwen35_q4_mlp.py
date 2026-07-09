from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest


def _require_q4_kernel():
    from omlx.custom_kernels.qwen35_prefill import fast

    if not fast.has_symbol("qwen35_q4_affine_qmm_t"):
        pytest.skip("qwen35_q4_affine_qmm_t native kernel unavailable")
    return fast


def _require_qmm_kernels(bits):
    from omlx.custom_kernels.qwen35_prefill import fast

    for bit in bits:
        name = f"qwen35_q{bit}_affine_qmm_t"
        if not fast.has_symbol(name):
            pytest.skip(f"{name} native kernel unavailable")
    return fast


def _quantized_bf16(linear, bits=4):
    qlinear = nn.QuantizedLinear.from_linear(
        linear, group_size=64, bits=bits, mode="affine"
    )
    qlinear.scales = qlinear.scales.astype(mx.bfloat16)
    if qlinear.biases is not None:
        qlinear.biases = qlinear.biases.astype(mx.bfloat16)
    return qlinear


@pytest.mark.parametrize("bits", [4, 5, 6, 8])
def test_qwen35_q_affine_qmm_matches_mlx_quantized_matmul(bits):
    fast = _require_qmm_kernels((bits,))
    x = mx.random.normal((1, 32, 256)).astype(mx.bfloat16)
    w_full = mx.random.normal((128, 256)).astype(mx.float32)
    weight, scales, biases = mx.quantize(
        w_full, group_size=64, bits=bits, mode="affine"
    )
    scales = scales.astype(x.dtype)
    biases = biases.astype(x.dtype)
    ref = mx.quantized_matmul(
        x,
        weight,
        scales=scales,
        biases=biases,
        transpose=True,
        group_size=64,
        bits=bits,
        mode="affine",
    )
    got = getattr(fast, f"qwen35_q{bits}_affine_qmm_t")(
        x, weight, scales, biases, 8
    )
    mx.eval(ref, got)

    diff = mx.abs(got.astype(mx.float32) - ref.astype(mx.float32))
    mx.eval(diff)
    max_abs = float(mx.max(diff).item())
    rel = float(
        (mx.max(diff) / (mx.max(mx.abs(ref.astype(mx.float32))) + 1e-9)).item()
    )
    assert max_abs <= 1.0
    assert rel <= 0.05


def test_qwen35_q4_mlp_patch_routes_prefill_and_skips_decode(monkeypatch):
    fast = _require_q4_kernel()
    import mlx_lm.models.qwen3_5 as qwen35

    from omlx.patches.qwen35_q4_mlp import apply_qwen35_q4_mlp_patch

    monkeypatch.setenv("OMLX_QWEN35_Q4_MLP", "1")
    monkeypatch.setenv("OMLX_QWEN35_Q4_MLP_MIN_TOKENS", "16")

    mlp = qwen35.MLP(256, 512)
    for name in ("gate_proj", "up_proj", "down_proj"):
        setattr(mlp, name, _quantized_bf16(getattr(mlp, name)))

    x = mx.random.normal((1, 32, 256)).astype(mx.bfloat16)
    y_ref = qwen35.MLP.__call__(mlp, x)
    mx.eval(y_ref)

    calls = {"count": 0}
    orig_qmm = fast.qwen35_q4_affine_qmm_t

    def spy(*args, **kwargs):
        calls["count"] += 1
        return orig_qmm(*args, **kwargs)

    monkeypatch.setattr(fast, "qwen35_q4_affine_qmm_t", spy)
    assert apply_qwen35_q4_mlp_patch() is True
    y = mlp(x)
    mx.eval(y)
    assert calls["count"] == 3
    assert (
        mx.max(mx.abs(y.astype(mx.float32) - y_ref.astype(mx.float32))).item()
        <= 1.0
    )

    calls["count"] = 0
    y_decode = mlp(x[:, :1, :])
    mx.eval(y_decode)
    assert calls["count"] == 0


def test_qwen35_mixed_bit_mlp_patch_routes_5_bit_down_proj(monkeypatch):
    fast = _require_qmm_kernels((4, 5))
    import mlx_lm.models.qwen3_5 as qwen35

    from omlx.patches.qwen35_q4_mlp import apply_qwen35_q4_mlp_patch

    monkeypatch.setenv("OMLX_QWEN35_Q4_MLP", "1")
    monkeypatch.setenv("OMLX_QWEN35_Q4_MLP_MIN_TOKENS", "16")

    mlp = qwen35.MLP(256, 512)
    mlp.gate_proj = _quantized_bf16(mlp.gate_proj, bits=4)
    mlp.up_proj = _quantized_bf16(mlp.up_proj, bits=4)
    mlp.down_proj = _quantized_bf16(mlp.down_proj, bits=5)

    x = mx.random.normal((1, 32, 256)).astype(mx.bfloat16)
    orig_call = getattr(qwen35.MLP, "_omlx_q4_mlp_original_call", qwen35.MLP.__call__)
    y_ref = orig_call(mlp, x)
    mx.eval(y_ref)

    calls = {4: 0, 5: 0}
    orig_q4 = fast.qwen35_q4_affine_qmm_t
    orig_q5 = fast.qwen35_q5_affine_qmm_t

    def spy_q4(*args, **kwargs):
        calls[4] += 1
        return orig_q4(*args, **kwargs)

    def spy_q5(*args, **kwargs):
        calls[5] += 1
        return orig_q5(*args, **kwargs)

    monkeypatch.setattr(fast, "qwen35_q4_affine_qmm_t", spy_q4)
    monkeypatch.setattr(fast, "qwen35_q5_affine_qmm_t", spy_q5)
    assert apply_qwen35_q4_mlp_patch() is True

    y = mlp(x)
    mx.eval(y)
    assert calls == {4: 2, 5: 1}
    assert (
        mx.max(mx.abs(y.astype(mx.float32) - y_ref.astype(mx.float32))).item()
        <= 1.0
    )


def test_qwen35_q8_route_uses_bit_specific_min_tokens():
    _require_qmm_kernels((4, 8))

    import omlx.patches.qwen35_q4_mlp as q4patch

    q4_linear = nn.QuantizedLinear(
        256,
        128,
        bias=False,
        group_size=64,
        bits=4,
    )
    q8_linear = nn.QuantizedLinear(
        256,
        128,
        bias=False,
        group_size=64,
        bits=8,
    )
    for linear in (q4_linear, q8_linear):
        linear.scales = linear.scales.astype(mx.bfloat16)
        if linear.biases is not None:
            linear.biases = linear.biases.astype(mx.bfloat16)

    x = mx.random.normal((1, 32, 256)).astype(mx.bfloat16)

    assert q4patch._can_route_affine_linear(
        q4_linear,
        x,
        min_tokens=16,
        q8_min_tokens=64,
    )
    assert not q4patch._can_route_affine_linear(
        q8_linear,
        x,
        min_tokens=16,
        q8_min_tokens=64,
    )
    assert q4patch._can_route_affine_linear(
        q8_linear,
        x,
        min_tokens=16,
        q8_min_tokens=16,
    )


def test_qwen35_q4_mlp_patch_prechecks_down_proj_before_gate_up(monkeypatch):
    fast = _require_q4_kernel()
    import mlx_lm.models.qwen3_5 as qwen35

    from omlx.patches.qwen35_q4_mlp import apply_qwen35_q4_mlp_patch

    monkeypatch.setenv("OMLX_QWEN35_Q4_MLP", "1")
    monkeypatch.setenv("OMLX_QWEN35_Q4_MLP_MIN_TOKENS", "16")

    mlp = qwen35.MLP(256, 512)
    mlp.gate_proj = _quantized_bf16(mlp.gate_proj)
    mlp.up_proj = _quantized_bf16(mlp.up_proj)

    # oQ4e models can keep gate/up as supported q4 while down_proj is not
    # supported by the native q4 tile. The patch must not compute gate/up with
    # native qmm and then throw that work away by falling back to the stock MLP.
    unsupported_down = nn.QuantizedLinear(
        512,
        48,
        bias=False,
        group_size=64,
        bits=4,
    )
    unsupported_down.scales = unsupported_down.scales.astype(mx.bfloat16)
    if unsupported_down.biases is not None:
        unsupported_down.biases = unsupported_down.biases.astype(mx.bfloat16)
    mlp.down_proj = unsupported_down

    x = mx.random.normal((1, 32, 256)).astype(mx.bfloat16)
    calls = {"count": 0}
    orig_qmm = fast.qwen35_q4_affine_qmm_t

    def spy(*args, **kwargs):
        calls["count"] += 1
        return orig_qmm(*args, **kwargs)

    monkeypatch.setattr(fast, "qwen35_q4_affine_qmm_t", spy)
    assert apply_qwen35_q4_mlp_patch() is True
    y = mlp(x)
    mx.eval(y)
    assert calls["count"] == 0


def test_qwen35_q4_prefill_linear_patch_routes_supported_only(monkeypatch):
    fast = _require_q4_kernel()
    import mlx_vlm.models.qwen3_5.language as qwen35_lang

    from omlx.patches.qwen35_q4_mlp import apply_qwen35_q4_prefill_linear_patch

    monkeypatch.setenv("OMLX_QWEN35_Q4_LINEAR", "1")
    monkeypatch.setenv("OMLX_QWEN35_Q4_LINEAR_MIN_TOKENS", "16")

    supported = nn.QuantizedLinear(256, 128, bias=False, group_size=64, bits=4)
    unsupported = nn.QuantizedLinear(256, 48, bias=False, group_size=64, bits=4)
    for linear in (supported, unsupported):
        linear.scales = linear.scales.astype(mx.bfloat16)
        if linear.biases is not None:
            linear.biases = linear.biases.astype(mx.bfloat16)

    x = mx.random.normal((1, 32, 256)).astype(mx.bfloat16)
    calls = {"count": 0}
    orig_qmm = fast.qwen35_q4_affine_qmm_t

    def spy(*args, **kwargs):
        calls["count"] += 1
        return orig_qmm(*args, **kwargs)

    monkeypatch.setattr(fast, "qwen35_q4_affine_qmm_t", spy)
    assert apply_qwen35_q4_prefill_linear_patch() is True
    out0, out1 = qwen35_lang._target_verify_linears(
        (supported, unsupported), x, False
    )
    mx.eval(out0, out1)
    assert calls["count"] == 1

    calls["count"] = 0
    decode = qwen35_lang._target_verify_linear(supported, x[:, :1, :], False)
    mx.eval(decode)
    assert calls["count"] == 0


def test_qwen35_q4_lm_prefill_linear_patch_routes_attention_and_gdn(
    monkeypatch,
):
    fast = _require_q4_kernel()
    import mlx_lm.models.qwen3_5 as qwen35

    import omlx.patches.qwen35_q4_mlp as q4patch

    monkeypatch.setenv("OMLX_QWEN35_Q4_LM_LINEAR", "1")
    monkeypatch.setenv("OMLX_QWEN35_Q4_LINEAR_MIN_TOKENS", "16")

    args = qwen35.TextModelArgs(
        model_type="qwen3_5",
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        attention_bias=False,
        rms_norm_eps=1e-6,
        max_position_embeddings=4096,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        rope_parameters={
            "type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
        },
    )

    attn = qwen35.Attention(args)
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        setattr(attn, name, _quantized_bf16(getattr(attn, name)))

    gdn = qwen35.GatedDeltaNet(args)
    for name in (
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "out_proj",
    ):
        setattr(gdn, name, _quantized_bf16(getattr(gdn, name)))

    orig_attn_call = qwen35.Attention.__call__
    orig_gdn_call = qwen35.GatedDeltaNet.__call__
    orig_lm_patched = q4patch._LM_LINEAR_PATCHED

    saved_attrs = {}
    for cls, attrs in (
        (
            qwen35.Attention,
            (
                "_omlx_q4_lm_attention_patched",
                "_omlx_q4_lm_attention_original_call",
            ),
        ),
        (
            qwen35.GatedDeltaNet,
            (
                "_omlx_q4_lm_gdn_patched",
                "_omlx_q4_lm_gdn_original_call",
            ),
        ),
    ):
        for attr in attrs:
            saved_attrs[(cls, attr)] = (
                getattr(cls, attr) if hasattr(cls, attr) else None,
                hasattr(cls, attr),
            )
            if hasattr(cls, attr):
                delattr(cls, attr)

    x = mx.random.normal((1, 32, 256)).astype(mx.bfloat16)
    y_attn_ref = orig_attn_call(attn, x)
    y_gdn_ref = orig_gdn_call(gdn, x)
    mx.eval(y_attn_ref, y_gdn_ref)

    calls = {"count": 0}
    orig_qmm = fast.qwen35_q4_affine_qmm_t

    def spy(*args, **kwargs):
        calls["count"] += 1
        return orig_qmm(*args, **kwargs)

    try:
        monkeypatch.setattr(q4patch, "_LM_LINEAR_PATCHED", False)
        monkeypatch.setattr(fast, "qwen35_q4_affine_qmm_t", spy)
        assert q4patch.apply_qwen35_q4_lm_prefill_linear_patch() is True

        y_attn = attn(x)
        mx.eval(y_attn)
        assert calls["count"] == 3
        assert (
            mx.max(
                mx.abs(y_attn.astype(mx.float32) - y_attn_ref.astype(mx.float32))
            ).item()
            <= 1.0
        )

        calls["count"] = 0
        y_gdn = gdn(x)
        mx.eval(y_gdn)
        assert calls["count"] == 2
        assert (
            mx.max(
                mx.abs(y_gdn.astype(mx.float32) - y_gdn_ref.astype(mx.float32))
            ).item()
            <= 1.0
        )

        calls["count"] = 0
        y_attn_decode = attn(x[:, :1, :])
        y_gdn_decode = gdn(x[:, :1, :])
        mx.eval(y_attn_decode, y_gdn_decode)
        assert calls["count"] == 0
    finally:
        qwen35.Attention.__call__ = orig_attn_call
        qwen35.GatedDeltaNet.__call__ = orig_gdn_call
        q4patch._LM_LINEAR_PATCHED = orig_lm_patched
        for (cls, attr), (value, existed) in saved_attrs.items():
            if existed:
                setattr(cls, attr, value)
            elif hasattr(cls, attr):
                delattr(cls, attr)
