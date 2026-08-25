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
    got = getattr(fast, f"qwen35_q{bits}_affine_qmm_t")(x, weight, scales, biases, 8)
    mx.eval(ref, got)

    diff = mx.abs(got.astype(mx.float32) - ref.astype(mx.float32))
    mx.eval(diff)
    max_abs = float(mx.max(diff).item())
    rel = float((mx.max(diff) / (mx.max(mx.abs(ref.astype(mx.float32))) + 1e-9)).item())
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
    assert mx.max(mx.abs(y.astype(mx.float32) - y_ref.astype(mx.float32))).item() <= 1.0

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
    assert mx.max(mx.abs(y.astype(mx.float32) - y_ref.astype(mx.float32))).item() <= 1.0


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


def test_post_ane_qmm_or_linear_routes_q8_through_env_threshold(monkeypatch):
    import omlx.patches.qwen35_q4_mlp as q4patch

    routed = []
    monkeypatch.setattr(
        q4patch,
        "_linear_qmm",
        lambda linear, x, variant: routed.append((linear, variant)) or x,
    )

    class _Stock:
        def __init__(self, bits=None):
            if bits is not None:
                self.bits = bits
            self.called = 0

        def __call__(self, x):
            self.called += 1
            return x

    x = mx.zeros((1, 2048, 64), dtype=mx.bfloat16)

    q8 = _Stock(bits=8)
    assert q4patch._post_ane_qmm_or_linear(q8, x, 8) is x
    assert q8.called == 1
    assert routed == []

    monkeypatch.setenv("OMLX_QWEN35_Q8_LINEAR_MIN_TOKENS", "2048")
    q8_low = _Stock(bits=8)
    q4patch._post_ane_qmm_or_linear(q8_low, x, 8)
    assert q8_low.called == 0
    assert routed == [(q8_low, 8)]

    q5 = _Stock(bits=5)
    q4patch._post_ane_qmm_or_linear(q5, x, 8)
    assert q5.called == 0
    assert routed[-1] == (q5, 8)


def test_qwen35_q8_gdn_backend_has_first_refusal_before_gpu_threshold(
    monkeypatch,
):
    import mlx_lm.models.qwen3_5 as qwen35

    import omlx.patches.qwen35_q4_mlp as q4patch

    class BackendCalledError(Exception):
        pass

    monkeypatch.setattr(q4patch, "_has_native_qmm", lambda: True)
    monkeypatch.setenv("OMLX_QWEN35_Q4_LM_LINEAR", "1")
    monkeypatch.setenv("OMLX_QWEN35_Q4_LINEAR_MIN_TOKENS", "16")
    monkeypatch.setenv("OMLX_QWEN35_Q8_LINEAR_MIN_TOKENS", "16384")

    class FakeGDN:
        sharding_group = None
        in_proj_qkv = object()
        in_proj_z = object()
        in_proj_b = object()
        in_proj_a = object()

    gdn = FakeGDN()
    x = mx.zeros((1, 32, 1), dtype=mx.bfloat16)

    def gdn_backend(module, inputs, target_verify=False):
        assert module is gdn
        assert inputs is x
        assert target_verify is False
        raise BackendCalledError

    def original_call(module, inputs, mask=None, cache=None):
        return inputs

    orig_gdn_call = qwen35.GatedDeltaNet.__call__
    orig_lm_patched = q4patch._LM_LINEAR_PATCHED
    orig_gdn_backend = q4patch._LM_GDN_PREFILL_BACKEND
    saved_attrs = {}
    for attr in (
        "_omlx_q4_lm_gdn_patched",
        "_omlx_q4_lm_gdn_original_call",
        "_omlx_q4_lm_gdn_wrapper",
    ):
        saved_attrs[attr] = (
            getattr(qwen35.GatedDeltaNet, attr)
            if hasattr(qwen35.GatedDeltaNet, attr)
            else None,
            hasattr(qwen35.GatedDeltaNet, attr),
        )
        if hasattr(qwen35.GatedDeltaNet, attr):
            delattr(qwen35.GatedDeltaNet, attr)

    try:
        qwen35.GatedDeltaNet.__call__ = original_call
        q4patch._LM_LINEAR_PATCHED = False
        q4patch.register_qwen35_lm_gdn_prefill_backend(gdn_backend)
        assert q4patch.apply_qwen35_q4_lm_prefill_linear_patch() is True

        with pytest.raises(BackendCalledError):
            qwen35.GatedDeltaNet.__call__(gdn, x)
        monkeypatch.setenv("OMLX_QWEN35_Q4_LM_LINEAR", "0")
        assert qwen35.GatedDeltaNet.__call__(gdn, x) is x
    finally:
        qwen35.GatedDeltaNet.__call__ = orig_gdn_call
        q4patch._LM_LINEAR_PATCHED = orig_lm_patched
        q4patch._LM_GDN_PREFILL_BACKEND = orig_gdn_backend
        for attr, (value, existed) in saved_attrs.items():
            if existed:
                setattr(qwen35.GatedDeltaNet, attr, value)
            elif hasattr(qwen35.GatedDeltaNet, attr):
                delattr(qwen35.GatedDeltaNet, attr)


@pytest.mark.parametrize(
    (
        "group_size",
        "nax_available",
        "nax_qmm_kernels_built",
        "allow_gs128",
        "expected",
    ),
    [
        (64, True, True, False, True),
        (128, False, False, False, True),
        (128, False, True, False, True),
        (128, True, False, False, False),
        (128, True, True, False, False),
        (128, True, False, True, True),
        (128, True, True, True, True),
    ],
)
def test_qwen35_qmm_routing_uses_stock_nax_availability(
    monkeypatch,
    group_size,
    nax_available,
    nax_qmm_kernels_built,
    allow_gs128,
    expected,
):
    import omlx.patches.qwen35_q4_mlp as q4patch
    from omlx.custom_kernels.qwen35_prefill import fast

    linear = nn.QuantizedLinear(
        256,
        128,
        bias=False,
        group_size=group_size,
        bits=4,
    )
    linear.scales = linear.scales.astype(mx.bfloat16)
    linear.biases = linear.biases.astype(mx.bfloat16)

    monkeypatch.setattr(q4patch, "_qmm_supports_group_size", lambda _gs: True)
    monkeypatch.setattr(q4patch, "_native_qmm_for_bits", lambda _bits: object())
    monkeypatch.setattr(q4patch, "is_nax_available", lambda: nax_available)
    monkeypatch.setattr(
        fast,
        "nax_qmm_kernels_built",
        lambda: nax_qmm_kernels_built,
    )
    if allow_gs128:
        monkeypatch.setenv("OMLX_QWEN35_Q4_MLP_ALLOW_GS128", "1")
    else:
        monkeypatch.delenv("OMLX_QWEN35_Q4_MLP_ALLOW_GS128", raising=False)

    assert (
        q4patch._is_supported_affine_linear_shape(
            linear,
            mx.bfloat16,
            ndim=3,
            seq_len=2048,
            input_dim=256,
        )
        is expected
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
    out0, out1 = qwen35_lang._target_verify_linears((supported, unsupported), x, False)
    mx.eval(out0, out1)
    assert calls["count"] == 1

    calls["count"] = 0
    decode = qwen35_lang._target_verify_linear(supported, x[:, :1, :], False)
    mx.eval(decode)
    assert calls["count"] == 0


def test_qwen35_q4_lm_attention_uses_sdpa_installed_after_the_patch(monkeypatch):
    """The patch must not freeze the SDPA it saw at install time (issue #2372).

    TurboQuant installs its own dispatcher when a TQ-enabled model loads, which
    happens after this patch whenever any earlier load ran without TurboQuant.
    A frozen reference kept routing TurboQuant caches into the plain mlx-lm SDPA,
    which raised 'TurboQuantKVCache' object has no attribute 'group_size'.
    """
    _require_q4_kernel()
    import importlib

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

    orig_attn_call = qwen35.Attention.__call__
    orig_lm_patched = q4patch._LM_LINEAR_PATCHED
    saved_attrs = {}
    for attr in (
        "_omlx_q4_lm_attention_patched",
        "_omlx_q4_lm_attention_original_call",
    ):
        existed = hasattr(qwen35.Attention, attr)
        saved_attrs[attr] = (
            getattr(qwen35.Attention, attr) if existed else None,
            existed,
        )
        if hasattr(qwen35.Attention, attr):
            delattr(qwen35.Attention, attr)

    x = mx.random.normal((1, 32, 256)).astype(mx.bfloat16)
    calls = {"count": 0}

    def sentinel_sdpa(queries, keys, values, cache=None, **kwargs):
        calls["count"] += 1
        return mx.zeros_like(queries)

    try:
        q4patch._LM_LINEAR_PATCHED = False
        assert q4patch.apply_qwen35_q4_lm_prefill_linear_patch() is True

        # Install the replacement dispatcher only after the patch is in place,
        # the way a later TurboQuant-enabled model load does.
        attn_module = importlib.import_module(qwen35.Attention.__module__)
        monkeypatch.setattr(attn_module, "scaled_dot_product_attention", sentinel_sdpa)

        y = attn(x)
        mx.eval(y)
        assert calls["count"] == 1
        assert y.shape == x.shape
    finally:
        qwen35.Attention.__call__ = orig_attn_call
        q4patch._LM_LINEAR_PATCHED = orig_lm_patched
        for attr, (value, existed) in saved_attrs.items():
            if existed:
                setattr(qwen35.Attention, attr, value)
            elif hasattr(qwen35.Attention, attr):
                delattr(qwen35.Attention, attr)


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
    for name in ("in_proj_qkv", "in_proj_z", "out_proj"):
        setattr(gdn, name, _quantized_bf16(getattr(gdn, name)))
    for name in ("in_proj_b", "in_proj_a"):
        setattr(gdn, name, _quantized_bf16(getattr(gdn, name), bits=8))

    gdn_q8 = qwen35.GatedDeltaNet(args)
    for name in (
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "out_proj",
    ):
        setattr(gdn_q8, name, _quantized_bf16(getattr(gdn_q8, name), bits=8))

    orig_attn_call = qwen35.Attention.__call__
    orig_gdn_call = qwen35.GatedDeltaNet.__call__
    orig_lm_patched = q4patch._LM_LINEAR_PATCHED
    orig_gdn_backend = q4patch._LM_GDN_PREFILL_BACKEND

    saved_attrs = {}
    for cls, attrs in (
        (
            qwen35.Attention,
            (
                "_omlx_q4_lm_attention_patched",
                "_omlx_q4_lm_attention_original_call",
                "_omlx_q4_lm_attention_wrapper",
            ),
        ),
        (
            qwen35.GatedDeltaNet,
            (
                "_omlx_q4_lm_gdn_patched",
                "_omlx_q4_lm_gdn_original_call",
                "_omlx_q4_lm_gdn_wrapper",
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

        backend_calls = []

        def gdn_backend(module, inputs, target_verify=False):
            backend_calls.append((module, inputs.shape, target_verify))
            return (
                module.in_proj_qkv(inputs),
                module.in_proj_z(inputs),
                module.in_proj_b(inputs),
                module.in_proj_a(inputs),
            )

        q4patch.register_qwen35_lm_gdn_prefill_backend(gdn_backend)
        y_gdn_backend = gdn(x)
        mx.eval(y_gdn_backend)
        assert backend_calls == [(gdn, x.shape, False)]
        assert (
            mx.max(
                mx.abs(y_gdn_backend.astype(mx.float32) - y_gdn_ref.astype(mx.float32))
            ).item()
            <= 1.0
        )

        # The q8 standalone GPU tile is intentionally disabled below 16K,
        # but that threshold must not prevent the independent 2K ANE backend
        # from receiving the GDN projections.
        backend_calls.clear()
        y_gdn_q8_backend = gdn_q8(x)
        mx.eval(y_gdn_q8_backend)
        assert backend_calls == [(gdn_q8, x.shape, False)]
        assert y_gdn_q8_backend.shape == x.shape

        # Simulate the MTP lifecycle restoring GDN.__call__ while leaving the
        # process-wide patch flag and class metadata behind. A subsequent
        # model load must validate the live callable and reinstall the hook.
        qwen35.GatedDeltaNet.__call__ = orig_gdn_call
        assert q4patch._LM_LINEAR_PATCHED is True
        assert q4patch.apply_qwen35_q4_lm_prefill_linear_patch() is True
        assert (
            qwen35.GatedDeltaNet.__call__
            is qwen35.GatedDeltaNet._omlx_q4_lm_gdn_wrapper
        )
        backend_calls.clear()
        y_gdn_reloaded = gdn(x)
        mx.eval(y_gdn_reloaded)
        assert backend_calls == [(gdn, x.shape, False)]

        calls["count"] = 0
        y_attn_decode = attn(x[:, :1, :])
        y_gdn_decode = gdn(x[:, :1, :])
        mx.eval(y_attn_decode, y_gdn_decode)
        assert calls["count"] == 0
    finally:
        qwen35.Attention.__call__ = orig_attn_call
        qwen35.GatedDeltaNet.__call__ = orig_gdn_call
        q4patch._LM_LINEAR_PATCHED = orig_lm_patched
        q4patch._LM_GDN_PREFILL_BACKEND = orig_gdn_backend
        for (cls, attr), (value, existed) in saved_attrs.items():
            if existed:
                setattr(cls, attr, value)
            elif hasattr(cls, attr):
                delattr(cls, attr)


def _muse_applied():
    from omlx.patches.mlx_vlm_muse_glimmer_compat import (
        apply_mlx_vlm_muse_glimmer_compat_patch,
    )
    from omlx.patches.qwen35_q4_mlp import apply_muse_glimmer_q4_prefill_patch

    apply_mlx_vlm_muse_glimmer_compat_patch()
    if not apply_muse_glimmer_q4_prefill_patch():
        pytest.skip("muse q4 prefill patch unavailable (native kernel missing)")


def _tiny_muse_text_config():
    from mlx_vlm.models.muse_glimmer.config import TextConfig

    return TextConfig(
        vocab_size=64,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=4096,
        sliding_window=64,
        layer_types=["sliding_attention", "full_attention"],
        layer_rope_theta=[10000.0, 0],
    )


def _quantize_module_linears(module, names, bits=4):
    for name in names:
        linear = getattr(module, name)
        setattr(
            module,
            name,
            nn.QuantizedLinear.from_linear(
                linear, group_size=64, bits=bits, mode="affine"
            ),
        )


def test_muse_glimmer_q4_attention_wrapper_matches_bf16_reference(monkeypatch):
    from omlx.patches.mlx_vlm_muse_glimmer_compat import (
        apply_mlx_vlm_muse_glimmer_compat_patch,
    )

    apply_mlx_vlm_muse_glimmer_compat_patch()
    from mlx_vlm.models.muse_glimmer.language import Attention

    import omlx.patches.qwen35_q4_mlp as q4patch

    monkeypatch.setattr(
        q4patch, "_can_route_affine_linear", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(
        q4patch, "_can_route_affine_linear_shape", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(
        q4patch,
        "_linear_qmm",
        lambda linear, inputs, variant: linear(inputs),
    )

    mx.random.seed(0)
    attn = Attention(_tiny_muse_text_config(), 1)
    attn.set_dtype(mx.bfloat16)
    _quantize_module_linears(
        attn, ("q_proj", "k_proj", "v_proj", "gate_proj", "o_proj")
    )
    original_call = getattr(
        Attention,
        "_omlx_q4_muse_attn_original_call",
        Attention.__call__,
    )
    patched_call = q4patch._make_patched_muse_attention(
        original_call,
        variant=8,
        min_tokens=1,
        q8_min_tokens=1,
    )

    inputs = mx.random.normal((1, 64, 128)).astype(mx.bfloat16)
    patched = patched_call(attn, inputs, mask=None, cache=None)
    reference = original_call(attn, inputs, mask=None, cache=None)
    mx.eval(patched, reference)

    assert bool(mx.array_equal(patched, reference))


def _assert_muse_qmm_close(actual, expected):
    # The native tile and mx.quantized_matmul use different BF16 reduction
    # orders. Real Muse oQ checkpoints store BF16 scales and biases, so allow
    # the observed one-ULP projection drift while keeping a tight end-to-end
    # bound on the mirrored MLP/attention bodies.
    max_diff = mx.max(mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32)))
    assert float(max_diff.item()) <= 0.02


def _install_muse_qmm_spy(monkeypatch):
    import omlx.patches.qwen35_q4_mlp as q4patch

    calls = {"count": 0}
    original_qmm = q4patch._linear_qmm

    def spy(*args, **kwargs):
        calls["count"] += 1
        return original_qmm(*args, **kwargs)

    monkeypatch.setattr(q4patch, "_linear_qmm", spy)
    return calls


def test_muse_glimmer_q4_mlp_patch_matches_bf16_reference(monkeypatch):
    _muse_applied()
    from mlx_vlm.models.muse_glimmer.language import MLP

    calls = _install_muse_qmm_spy(monkeypatch)

    mx.random.seed(0)
    mlp = MLP(_tiny_muse_text_config())
    mlp.set_dtype(mx.bfloat16)
    _quantize_module_linears(mlp, ("gate_proj", "up_proj", "down_proj"))
    orig_call = type(mlp)._omlx_q4_mlp_original_call

    prefill = mx.random.normal((1, 2048, 128)).astype(mx.bfloat16)
    decode = mx.random.normal((1, 1, 128)).astype(mx.bfloat16)

    patched_out = mlp(prefill)
    orig_out = orig_call(mlp, prefill)
    mx.eval(patched_out, orig_out)
    assert calls["count"] == 3
    _assert_muse_qmm_close(patched_out, orig_out)

    calls["count"] = 0
    patched_out = mlp(decode)
    orig_out = orig_call(mlp, decode)
    mx.eval(patched_out, orig_out)
    assert calls["count"] == 0
    assert bool(mx.array_equal(patched_out, orig_out))


def test_muse_glimmer_q4_attention_patch_matches_bf16_reference(monkeypatch):
    _muse_applied()
    from mlx_vlm.models.muse_glimmer.language import Attention

    calls = _install_muse_qmm_spy(monkeypatch)

    mx.random.seed(0)
    config = _tiny_muse_text_config()
    for layer_idx in (0, 1):  # sliding+rope and full+NoPE
        attn = Attention(config, layer_idx)
        attn.set_dtype(mx.bfloat16)
        _quantize_module_linears(
            attn, ("q_proj", "k_proj", "v_proj", "gate_proj", "o_proj")
        )
        orig_call = type(attn)._omlx_q4_muse_attn_original_call

        prefill = mx.random.normal((1, 2048, 128)).astype(mx.bfloat16)
        decode = mx.random.normal((1, 1, 128)).astype(mx.bfloat16)

        calls["count"] = 0
        patched_out = attn(prefill, mask=None, cache=None)
        orig_out = orig_call(attn, prefill, mask=None, cache=None)
        mx.eval(patched_out, orig_out)
        assert calls["count"] == 5
        _assert_muse_qmm_close(patched_out, orig_out)

        calls["count"] = 0
        patched_out = attn(decode, mask=None, cache=None)
        orig_out = orig_call(attn, decode, mask=None, cache=None)
        mx.eval(patched_out, orig_out)
        assert calls["count"] == 0
        assert bool(mx.array_equal(patched_out, orig_out))


def test_muse_glimmer_q4_attention_patch_with_cache_and_mask(monkeypatch):
    _muse_applied()
    from mlx_lm.models.base import create_attention_mask
    from mlx_vlm.models.cache import RotatingKVCache
    from mlx_vlm.models.muse_glimmer.language import Attention

    calls = _install_muse_qmm_spy(monkeypatch)

    mx.random.seed(0)
    config = _tiny_muse_text_config()
    attn = Attention(config, 0)  # sliding layer
    attn.set_dtype(mx.bfloat16)
    _quantize_module_linears(
        attn, ("q_proj", "k_proj", "v_proj", "gate_proj", "o_proj")
    )
    orig_call = type(attn)._omlx_q4_muse_attn_original_call

    x = mx.random.normal((1, 2048, 128)).astype(mx.bfloat16)
    cache_a = RotatingKVCache(max_size=64)
    cache_b = RotatingKVCache(max_size=64)
    mask = create_attention_mask(x, cache_a, window_size=64)

    patched_out = attn(x, mask=mask, cache=cache_a)
    orig_out = orig_call(attn, x, mask=mask, cache=cache_b)
    mx.eval(patched_out, orig_out)
    assert calls["count"] == 5
    _assert_muse_qmm_close(patched_out, orig_out)
    assert cache_a.offset == cache_b.offset
