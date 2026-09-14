import mlx.core as mx
import pytest

from omlx.patches.deepseek_v4 import switch_layers as sl


def require_native(*symbols):
    missing = [symbol for symbol in symbols if not sl.glm_fast.has_symbol(symbol)]
    if missing:
        pytest.skip(f"native kernel unavailable: {', '.join(missing)}")


def mixed_layer(dims=64, experts=4):
    layer = sl.SwitchGLU(dims, dims, experts, bias=False)
    layer.up_proj = layer.up_proj.to_quantized(64, 2, mode="affine")
    layer.gate_proj = layer.gate_proj.to_quantized(64, 2, mode="affine")
    layer.down_proj = layer.down_proj.to_quantized(32, 4, mode="mxfp4")
    for p in (layer.up_proj, layer.gate_proj):
        p.scales = p.scales.astype(mx.float16)
        p.biases = p.biases.astype(mx.float16)
    return layer


def indices(routes, experts=4):
    return mx.arange(routes, dtype=mx.int32).reshape(1, 1, routes) % experts


def trace_native(monkeypatch):
    calls = []
    for kind, name in (
        ("affine", "deepseek_affine_gather_qmm_blocks"),
        ("mxfp4", "deepseek_mxfp4_gather_qmm_blocks"),
    ):
        original = getattr(sl.glm_fast, name)

        def wrapper(*args, _kind=kind, _original=original, **kwargs):
            calls.append((_kind, args[0].dtype))
            return _original(*args, **kwargs)

        monkeypatch.setattr(sl.glm_fast, name, wrapper)
    return calls


def test_production_mixed_triplet_retains_native(monkeypatch):
    require_native(
        "deepseek_affine_gather_qmm_blocks",
        "deepseek_mxfp4_gather_qmm_blocks",
    )
    layer = mixed_layer()
    calls = trace_native(monkeypatch)
    x = mx.random.normal((1, 11, 64), dtype=mx.bfloat16)
    y = layer(x, indices(66))
    mx.eval(y)
    assert y.dtype == mx.bfloat16
    assert calls == [
        ("affine", mx.float16),
        ("affine", mx.float16),
        ("mxfp4", mx.float16),
    ]


@pytest.mark.parametrize("routes, native", [(63, False), (64, True)])
def test_threshold_boundary(routes, native, monkeypatch):
    require_native(
        "deepseek_affine_gather_qmm_blocks",
        "deepseek_mxfp4_gather_qmm_blocks",
    )
    layer = mixed_layer()
    calls = trace_native(monkeypatch)
    y = layer(mx.ones((1, 1, 64), dtype=mx.bfloat16), indices(routes))
    mx.eval(y)
    assert y.dtype == mx.bfloat16
    assert bool(calls) is native


def test_mxfp4_shape_and_sorted_guards():
    require_native("deepseek_mxfp4_gather_qmm_blocks")
    p = mixed_layer().down_proj
    assert p._can_use_mxfp4_blocks(mx.ones((64, 1, 64), mx.float16), True)
    assert not p._can_use_mxfp4_blocks(mx.ones((64, 2, 64), mx.float16), True)
    assert not p._can_use_mxfp4_blocks(mx.ones((64, 1, 64), mx.float16), False)


def test_unknown_mode_cannot_authorize_cast():
    layer = mixed_layer()
    layer.down_proj.mode = "future_quant"
    assert not layer.down_proj._is_fp16_compatible()
    assert [p._is_fp16_compatible() for p in (layer.up_proj, layer.gate_proj, layer.down_proj)] == [True, True, False]


def test_malformed_mxfp4_cannot_authorize_cast():
    p = mixed_layer().down_proj
    p.scales = p.scales[..., :-1]
    assert not p._is_fp16_compatible()


def test_mxfp4_float32_uses_stock(monkeypatch):
    p = mixed_layer().down_proj
    native_called = False
    stock_called = False
    native = sl.glm_fast.deepseek_mxfp4_gather_qmm_blocks
    stock = mx.gather_qmm

    def native_spy(*args, **kwargs):
        nonlocal native_called
        native_called = True
        return native(*args, **kwargs)

    def stock_spy(*args, **kwargs):
        nonlocal stock_called
        stock_called = True
        return stock(*args, **kwargs)

    monkeypatch.setattr(sl.glm_fast, "deepseek_mxfp4_gather_qmm_blocks", native_spy)
    monkeypatch.setattr(mx, "gather_qmm", stock_spy)
    x = mx.ones((64, 1, 64), dtype=mx.float32)
    y = p(x, mx.arange(64, dtype=mx.int32) % 4, sorted_indices=True)
    mx.eval(y)
    assert y.dtype == mx.float32
    assert stock_called and not native_called


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_mxfp4_narrow_fast_paths(dtype, monkeypatch):
    require_native("deepseek_mxfp4_gather_qmm_blocks")
    p = mixed_layer().down_proj
    calls = trace_native(monkeypatch)
    y = p(mx.ones((64, 1, 64), dtype=dtype), mx.arange(64, dtype=mx.int32) % 4, sorted_indices=True)
    mx.eval(y)
    assert y.dtype == dtype
    assert calls == [("mxfp4", dtype)]


def test_all_affine_rescue_unchanged():
    layer = sl.SwitchGLU(64, 64, 4, bias=False)
    for name in ("up_proj", "gate_proj", "down_proj"):
        p = getattr(layer, name).to_quantized(64, 2, mode="affine")
        p.scales = p.scales.astype(mx.float16)
        p.biases = p.biases.astype(mx.float16)
        setattr(layer, name, p)
    y = layer(mx.ones((1, 7, 64), dtype=mx.bfloat16), indices(42))
    mx.eval(y)
    assert y.dtype == mx.bfloat16


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_all_mxfp4_shared_plan_unchanged(dtype, monkeypatch):
    require_native("deepseek_mxfp4_gather_qmm_blocks")
    layer = sl.SwitchGLU(64, 64, 4, bias=False)
    for name in ("up_proj", "gate_proj", "down_proj"):
        setattr(layer, name, getattr(layer, name).to_quantized(32, 4, mode="mxfp4"))
    original_has_symbol = sl.glm_fast.has_symbol
    monkeypatch.setattr(
        sl.glm_fast,
        "has_symbol",
        lambda name: False
        if "mxfp4_gather_qmm_pair" in name
        else original_has_symbol(name),
    )
    calls = trace_native(monkeypatch)
    y = layer(mx.ones((1, 11, 64), dtype=dtype), indices(66))
    mx.eval(y)
    assert y.dtype == dtype
    assert calls == [("mxfp4", dtype)] * 3


def test_single_fp16_affine_projection_enables_mixed_rescue(monkeypatch):
    require_native(
        "deepseek_affine_gather_qmm_blocks",
        "deepseek_mxfp4_gather_qmm_blocks",
    )
    layer = sl.SwitchGLU(64, 64, 4, bias=False)
    affine = layer.up_proj.to_quantized(64, 2, mode="affine")
    affine.scales = affine.scales.astype(mx.float16)
    affine.biases = affine.biases.astype(mx.float16)
    layer.up_proj = affine
    layer.gate_proj = layer.gate_proj.to_quantized(32, 4, mode="mxfp4")
    layer.down_proj = layer.down_proj.to_quantized(32, 4, mode="mxfp4")
    original_has_symbol = sl.glm_fast.has_symbol
    monkeypatch.setattr(
        sl.glm_fast,
        "has_symbol",
        lambda name: False
        if "mxfp4_gather_qmm_pair" in name
        else original_has_symbol(name),
    )
    calls = trace_native(monkeypatch)
    y = layer(mx.ones((1, 11, 64), dtype=mx.bfloat16), indices(66))
    mx.eval(y)
    assert y.dtype == mx.bfloat16
    assert calls == [
        ("affine", mx.float16),
        ("mxfp4", mx.float16),
        ("mxfp4", mx.float16),
    ]


def test_numerical_parity_with_deliberate_stock(monkeypatch):
    require_native(
        "deepseek_affine_gather_qmm_blocks",
        "deepseek_mxfp4_gather_qmm_blocks",
    )
    mx.random.seed(540)
    layer = mixed_layer(dims=256, experts=8)
    x = mx.random.normal((1, 16, 256), dtype=mx.bfloat16)
    idx = indices(96, experts=8)
    native = layer(x, idx)
    mx.eval(native)
    monkeypatch.setattr(sl.QuantizedSwitchLinear, "_native_block_kind", lambda *a, **k: None)
    stock = layer(x, idx)
    mx.eval(stock)
    delta = mx.abs(native.astype(mx.float32) - stock.astype(mx.float32))
    assert bool(mx.all(mx.isfinite(native)).item())
    assert native.dtype == stock.dtype == mx.bfloat16
    assert float(mx.max(delta).item()) <= 0.02
    print("PARITY", "max", float(mx.max(delta).item()), "mean", float(mx.mean(delta).item()))


def test_native_requirement_skips_without_symbol(monkeypatch):
    monkeypatch.setattr(sl.glm_fast, "has_symbol", lambda _name: False)
    with pytest.raises(pytest.skip.Exception):
        require_native("deepseek_mxfp4_gather_qmm_blocks")
