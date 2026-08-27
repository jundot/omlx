import logging
import re
import weakref
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

import omlx.patches.qwen35_ane_prefill as ane_patch
from omlx.custom_kernels.qwen35_prefill import fast


def test_ane_compile_bindings_release_the_python_gil():
    bindings = (
        Path(__file__).resolve().parents[1]
        / "omlx/custom_kernels/qwen35_prefill/csrc/bindings.cpp"
    ).read_text(encoding="utf-8")
    blocks = bindings.split("  m.def(")
    guard = "nb::call_guard<nb::gil_scoped_release>()"

    for name in (
        "qwen35_ane_compile_linear",
        "qwen35_ane_compile_linear_bank",
        "qwen35_ane_compile_fp16_linear",
        "qwen35_ane_compile_swiglu_down",
    ):
        block = next(part for part in blocks if f'"{name}"' in part)
        assert guard in block


def test_dual_ane_ticket_acquisition_rolls_back_the_first_ticket():
    source = (
        Path(__file__).resolve().parents[1]
        / "omlx/custom_kernels/qwen35_prefill/csrc/qwen35_ane.mm"
    ).read_text(encoding="utf-8")
    helper = source.split("static AneTicketPair begin_ane_ticket_pair(", 1)[1]
    helper = helper.split("bool qwen35_ane_available()", 1)[0]

    assert "first->cancel_ticket(first_ticket);" in helper
    assert source.count("begin_ane_ticket_pair(") == 3


def test_ane_dispatch_guard_transfers_each_ticket_after_thread_spawn():
    """A thread-constructor failure must leave every unowned ticket guarded."""
    source = (
        Path(__file__).resolve().parents[1]
        / "omlx/custom_kernels/qwen35_prefill/csrc/qwen35_ane.mm"
    ).read_text(encoding="utf-8")
    dual = source.split("class DualAneHybridPrimitive", 1)[1]
    dual = dual.split("class AneHybridQ4SwiGLUDownPrimitive", 1)[0]
    fused = source.split("class AneHybridQ4SwiGLUDownPrimitive", 1)[1]

    assert "void transfer_ticket0() noexcept" in source
    assert "void transfer_ticket1() noexcept" in source
    assert dual.index("std::thread([model0") < dual.index(
        "ane_guard.transfer_ticket0();"
    )
    assert dual.index("std::thread([model1") < dual.index(
        "ane_guard.transfer_ticket1();"
    )
    assert fused.index("std::thread([model =") < fused.index(
        "ane_guard.transfer_ticket0();"
    )
    assert fused.index("std::thread([model1 =") < fused.index(
        "ane_guard.transfer_ticket1();"
    )
    assert source.count("ane_guard.transfer_ticket0();") == 2
    assert source.count("ane_guard.transfer_ticket1();") == 2


def test_hybrid_merge_waits_for_gpu_suffix_completion():
    """ANE completion alone must not let merge race an in-flight GPU qmm."""
    source = (
        Path(__file__).resolve().parents[1]
        / "omlx/custom_kernels/qwen35_prefill/csrc/qwen35_ane.mm"
    ).read_text(encoding="utf-8")
    single = source.split("class AneHybridQ4Primitive", 1)[1]
    single = single.split("class DualAneHybridPrimitive", 1)[0]
    dual = source.split("class DualAneHybridPrimitive", 1)[1]
    dual = dual.split("class AneHybridQ4SwiGLUDownPrimitive", 1)[0]

    for block, ane_wait in (
        (single, "model_->wait(ticket)"),
        (dual, "model1_->wait(ticket1)"),
    ):
        assert "[qmm_buffer retain];" in block
        assert block.count("[qmm_buffer waitUntilCompleted];") == 2
        assert block.index(ane_wait) < block.rindex(
            "[qmm_buffer waitUntilCompleted];"
        )
        assert block.rindex("[qmm_buffer waitUntilCompleted];") < block.index(
            "auto merge ="
        )
        assert block.rindex("[qmm_buffer release];") < block.index("auto merge =")


@pytest.fixture(autouse=True)
def _restore_lm_gdn_backend():
    import omlx.patches.qwen35_q4_mlp as q4_patch

    previous = q4_patch._LM_GDN_PREFILL_BACKEND
    try:
        yield
    finally:
        q4_patch.register_qwen35_lm_gdn_prefill_backend(previous)


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.QuantizedLinear(
            128, 256, bias=False, group_size=128, bits=4
        )
        self.up_proj = nn.QuantizedLinear(128, 256, bias=False, group_size=128, bits=4)
        self.down_proj = nn.QuantizedLinear(
            256, 128, bias=False, group_size=128, bits=4
        )


def _no_bank_builder(sequence_length):
    """Force the pre-builder staging path regardless of the built extension."""
    raise RuntimeError("bank builder disabled for this test")


class _Model(nn.Module):
    def __init__(self, count):
        super().__init__()
        self.layers = [_MLP() for _ in range(count)]


class _GDN(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj_qkv = nn.QuantizedLinear(
            128, 256, bias=False, group_size=64, bits=5
        )
        self.in_proj_z = nn.QuantizedLinear(128, 128, bias=False, group_size=64, bits=5)
        self.in_proj_b = nn.QuantizedLinear(128, 48, bias=False, group_size=64, bits=5)
        self.in_proj_a = nn.QuantizedLinear(128, 48, bias=False, group_size=64, bits=5)


class _Q6MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.QuantizedLinear(
            128, 256, bias=False, group_size=64, bits=6
        )
        self.up_proj = nn.QuantizedLinear(
            128, 256, bias=False, group_size=64, bits=6
        )
        self.down_proj = nn.QuantizedLinear(
            256, 128, bias=False, group_size=64, bits=6
        )
        for linear in (self.gate_proj, self.up_proj, self.down_proj):
            linear.scales = linear.scales.astype(mx.bfloat16)
            linear.biases = linear.biases.astype(mx.bfloat16)


class _Q6GDN(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj_qkv = nn.QuantizedLinear(
            128, 256, bias=False, group_size=64, bits=6
        )
        self.in_proj_z = nn.QuantizedLinear(
            128, 128, bias=False, group_size=64, bits=6
        )
        self.in_proj_b = nn.QuantizedLinear(
            128, 48, bias=False, group_size=64, bits=6
        )
        self.in_proj_a = nn.QuantizedLinear(
            128, 48, bias=False, group_size=64, bits=6
        )
        for linear in (
            self.in_proj_qkv,
            self.in_proj_z,
            self.in_proj_b,
            self.in_proj_a,
        ):
            linear.scales = linear.scales.astype(mx.bfloat16)
            linear.biases = linear.biases.astype(mx.bfloat16)


def test_q6_mlp_and_gdn_are_eligible_for_ane_hybrid_prefill():
    assert ane_patch._eligible_pair(_Q6MLP())
    assert ane_patch._eligible_gdn(_Q6GDN())


@pytest.mark.parametrize(
    ("bits", "symbol"),
    [
        (4, "qwen35_ane_dual_cpu_fp16_q4_swiglu_t"),
        (5, "qwen35_ane_dual_cpu_fp16_swiglu_t"),
        (6, "qwen35_ane_dual_cpu_fp16_swiglu_t"),
        (8, "qwen35_ane_dual_cpu_fp16_swiglu_t"),
    ],
)
def test_cpu_gate_kernel_keeps_q4_specialized(bits, symbol):
    assert ane_patch._cpu_gate_kernel_symbol(bits) == symbol


@pytest.mark.parametrize(
    ("bits", "symbol"),
    [
        (4, "qwen35_ane_cpu_fp16_q4_swiglu_t"),
        (5, "qwen35_ane_cpu_fp16_swiglu_t"),
        (6, "qwen35_ane_cpu_fp16_swiglu_t"),
        (8, "qwen35_ane_cpu_fp16_swiglu_t"),
    ],
)
def test_cpu_gate_kernel_selects_single_ane_symbols(bits, symbol):
    assert ane_patch._cpu_gate_kernel_symbol(bits, dual=False) == symbol


class _OQ4eMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.QuantizedLinear(128, 256, bias=False, group_size=64, bits=4)
        self.up_proj = nn.QuantizedLinear(128, 256, bias=False, group_size=64, bits=4)
        self.down_proj = nn.QuantizedLinear(256, 128, bias=False, group_size=64, bits=5)


class _OQ4eGDN(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj_qkv = nn.QuantizedLinear(
            128, 256, bias=False, group_size=64, bits=4
        )
        self.in_proj_z = nn.QuantizedLinear(128, 128, bias=False, group_size=64, bits=5)
        self.in_proj_b = nn.QuantizedLinear(128, 48, bias=False, group_size=64, bits=4)
        self.in_proj_a = nn.QuantizedLinear(128, 48, bias=False, group_size=64, bits=4)


class _OQ8MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.QuantizedLinear(128, 256, bias=False, group_size=64, bits=8)
        self.up_proj = nn.QuantizedLinear(128, 256, bias=False, group_size=64, bits=8)
        self.down_proj = nn.QuantizedLinear(256, 128, bias=False, group_size=64, bits=8)


class _OQ8GDN(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj_qkv = nn.QuantizedLinear(
            128, 256, bias=False, group_size=64, bits=8
        )
        self.in_proj_z = nn.QuantizedLinear(128, 128, bias=False, group_size=64, bits=8)
        self.in_proj_b = nn.QuantizedLinear(128, 48, bias=False, group_size=64, bits=8)
        self.in_proj_a = nn.QuantizedLinear(128, 48, bias=False, group_size=64, bits=8)


def _affine_linear(input_dim, output_dim, bits, group_size):
    linear = nn.QuantizedLinear(
        input_dim, output_dim, bias=False, group_size=group_size, bits=bits
    )
    linear.scales = linear.scales.astype(mx.bfloat16)
    linear.biases = linear.biases.astype(mx.bfloat16)
    return linear


def _make_affine_mlp(bits, group_size):
    return SimpleNamespace(
        gate_proj=_affine_linear(128, 256, bits, group_size),
        up_proj=_affine_linear(128, 256, bits, group_size),
        down_proj=_affine_linear(256, 128, bits, group_size),
    )


def _make_affine_gdn(bits, group_size):
    return SimpleNamespace(
        in_proj_qkv=_affine_linear(128, 256, bits, group_size),
        in_proj_z=_affine_linear(128, 128, bits, group_size),
        in_proj_b=_affine_linear(128, 48, bits, group_size),
        in_proj_a=_affine_linear(128, 48, bits, group_size),
    )


@pytest.mark.parametrize("sequence_length", [2048, 4096])
def test_configure_scheduler_preserves_wide_prompt_chunks(sequence_length):
    scheduler = SimpleNamespace(
        config=SimpleNamespace(prefill_step_size=2048),
        _qwen35_prefill_floor=4096,
    )

    configured = ane_patch.configure_qwen35_ane_prefill_scheduler(
        scheduler,
        sequence_length,
    )

    assert configured is True
    assert scheduler.config.prefill_step_size == 2048
    assert scheduler._qwen35_prefill_floor == 4096


def test_configure_scheduler_warns_when_shape_exceeds_delivered_width(caplog):
    scheduler = SimpleNamespace(
        config=SimpleNamespace(prefill_step_size=2048, paged_cache_block_size=2048),
        _qwen35_prefill_floor=4096,
        block_aware_cache=object(),
    )

    # Boundary snapshots cap delivered chunks at the 2048 block edge, so a
    # 4096 shape can never receive a full tile and must warn loudly.
    with caplog.at_level(logging.WARNING, logger="omlx.patches.qwen35_ane_prefill"):
        assert ane_patch.configure_qwen35_ane_prefill_scheduler(scheduler, 4096)
    assert "never execute" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="omlx.patches.qwen35_ane_prefill"):
        assert ane_patch.configure_qwen35_ane_prefill_scheduler(scheduler, 2048)
    assert "never execute" not in caplog.text

    caplog.clear()
    no_boundary = SimpleNamespace(
        config=SimpleNamespace(prefill_step_size=2048),
        _qwen35_prefill_floor=4096,
    )
    with caplog.at_level(logging.WARNING, logger="omlx.patches.qwen35_ane_prefill"):
        assert ane_patch.configure_qwen35_ane_prefill_scheduler(no_boundary, 4096)
    assert "never execute" not in caplog.text


def test_short_chunks_exit_before_the_tiling_planner(monkeypatch):
    monkeypatch.setattr(
        ane_patch,
        "_tiled_input_plan",
        lambda *args: pytest.fail("planner must not run for short chunks"),
    )
    mlp = SimpleNamespace(
        _omlx_ane_prefill_config=ane_patch._AnePrefillConfig(2048, 0.5, 8)
    )
    assert ane_patch._backend(mlp, mx.zeros((1, 64, 8), dtype=mx.float16)) is None
    gdn = SimpleNamespace(_omlx_ane_gdn_config=ane_patch._AneGDNConfig(2048, 0.5, 8))
    assert ane_patch._gdn_backend(gdn, mx.zeros((1, 64, 8), dtype=mx.float16)) is None


def test_wide_tile_tail_routes_native_qmm_from_min_tokens(monkeypatch):
    import omlx.patches.qwen35_q4_mlp as q4_patch

    routed = []
    monkeypatch.setattr(
        q4_patch,
        "_linear_qmm",
        lambda linear, value, variant: routed.append(int(value.shape[-2])) or value,
    )
    monkeypatch.setattr(
        ane_patch, "_backend_exact", lambda _mlp, block, _tv=False: block
    )
    monkeypatch.setattr(ane_patch, "swiglu", lambda gate, up: gate + up)
    mlp = SimpleNamespace(
        gate_proj=lambda value: value,
        up_proj=lambda value: value,
        down_proj=lambda value: value,
        _omlx_ane_prefill_config=ane_patch._AnePrefillConfig(4096, 0.5, 8),
    )

    result = ane_patch._backend(
        mlp, mx.zeros((1, 4096 + 2048, 8), dtype=mx.float16)
    )

    assert result is not None
    mx.eval(result)
    # The 2048-row tail sits at the min-tokens boundary, so gate, up, and
    # down all take the native qmm route instead of stock MLX.
    assert routed == [2048, 2048, 2048]


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        (2047, None),
        (2048, (1, 0)),
        (4095, (1, 2047)),
        (4096, (2, 0)),
        (8191, (3, 2047)),
    ],
)
def test_wide_tile_plan_uses_every_complete_block(rows, expected):
    x = mx.zeros((1, rows, 8), dtype=mx.float16)
    assert ane_patch._tiled_input_plan(x, 2048) == expected


def test_mlp_wide_call_tiles_full_blocks_and_keeps_gpu_tail(monkeypatch):
    calls = []

    def exact(_mlp, block, _target_verify=False):
        calls.append(("ane", int(block.shape[-2])))
        return mx.full(block.shape, 7, dtype=block.dtype)

    def linear(label, offset):
        def run(value):
            calls.append((label, int(value.shape[-2])))
            return value + offset

        return run

    monkeypatch.setattr(ane_patch, "_backend_exact", exact)
    monkeypatch.setattr(ane_patch, "swiglu", lambda gate, up: gate + up)
    mlp = SimpleNamespace(
        gate_proj=linear("gate", 10),
        up_proj=linear("up", 20),
        down_proj=linear("down", 0),
        _omlx_ane_prefill_config=ane_patch._AnePrefillConfig(2048, 0.53, 8),
    )

    result = ane_patch._backend(
        mlp, mx.zeros((1, 4095, 8), dtype=mx.float16)
    )
    mx.eval(result)

    assert result.shape == (1, 4095, 8)
    assert calls == [
        ("ane", 2048),
        ("gate", 2047),
        ("up", 2047),
        ("down", 2047),
    ]
    assert result[:, :2048].tolist()[0][0][0] == 7
    assert result[:, 2048:].tolist()[0][0][0] == 30


def test_mlp_profitable_tail_is_padded_and_sliced(monkeypatch):
    calls = []

    def exact(_mlp, block, _target_verify=False):
        calls.append((int(block.shape[-2]), float(mx.sum(block).item())))
        return mx.full(block.shape, 7, dtype=block.dtype)

    monkeypatch.setattr(ane_patch, "_backend_exact", exact)
    mlp = SimpleNamespace(
        _omlx_ane_prefill_config=ane_patch._AnePrefillConfig(
            2048, 0.53, 8, tail_padding_min_tokens=1358
        )
    )
    x = mx.ones((1, 4095, 8), dtype=mx.float16)

    result = ane_patch._backend(mlp, x)
    assert result is not None
    mx.eval(result)

    assert result.shape == (1, 4095, 8)
    assert calls == [(2048, 16384.0), (2048, 16376.0)]
    assert bool(mx.all(result == 7))


def test_profitable_short_prefill_uses_one_padded_tile(monkeypatch):
    seen = []

    def exact(_mlp, block, _target_verify=False):
        seen.append(block)
        return block + 2

    monkeypatch.setattr(ane_patch, "_backend_exact", exact)
    mlp = SimpleNamespace(
        _omlx_ane_prefill_config=ane_patch._AnePrefillConfig(
            2048, 0.53, 8, tail_padding_min_tokens=1358
        )
    )
    x = mx.ones((1, 1400, 8), dtype=mx.float16)

    result = ane_patch._backend(mlp, x)
    assert result is not None
    mx.eval(result, *seen)

    assert result.shape == x.shape
    assert seen[0].shape == (1, 2048, 8)
    assert bool(mx.all(seen[0][:, :1400] == 1))
    assert bool(mx.all(seen[0][:, 1400:] == 0))
    assert bool(mx.all(result == 3))


def test_low_fraction_wide_mlp_still_dispatches_complete_tile(monkeypatch):
    calls = []

    def exact(_mlp, block, _target_verify=False):
        calls.append(int(block.shape[-2]))
        return block

    monkeypatch.setattr(ane_patch, "_backend_exact", exact)
    monkeypatch.setattr(ane_patch, "swiglu", lambda gate, up: gate + up)
    mlp = SimpleNamespace(
        gate_proj=lambda value: value,
        up_proj=lambda value: value,
        down_proj=lambda value: value,
        _omlx_ane_prefill_config=ane_patch._AnePrefillConfig(2048, 0.25, 8),
    )
    result = ane_patch._backend(
        mlp, mx.zeros((1, 4095, 8), dtype=mx.float16)
    )
    assert result is not None
    mx.eval(result)
    assert calls == [2048]
    assert result.shape == (1, 4095, 8)


def test_gdn_wide_call_tiles_only_tokenwise_projections(monkeypatch):
    calls = []
    scheduled = []

    def exact(_gdn, block, _target_verify=False):
        calls.append(("ane", int(block.shape[-2])))
        return tuple(
            mx.full((1, block.shape[-2], 1), value, dtype=block.dtype)
            for value in (1, 2, 3, 4)
        )

    class Linear:
        def __init__(self, value):
            self.value = value

        def __call__(self, block):
            calls.append((self.value, int(block.shape[-2])))
            return mx.full(
                (1, block.shape[-2], 1), self.value, dtype=block.dtype
            )

    monkeypatch.setattr(ane_patch, "_gdn_backend_exact", exact)
    monkeypatch.setattr(mx, "async_eval", lambda *values: scheduled.append(values))
    linears = [Linear(value) for value in (10, 20, 30, 40)]
    gdn = SimpleNamespace(
        in_proj_qkv=linears[0],
        in_proj_z=linears[1],
        in_proj_b=linears[2],
        in_proj_a=linears[3],
        _omlx_ane_gdn_config=ane_patch._AneGDNConfig(2048, 0.50, 8),
    )

    result = ane_patch._gdn_backend(
        gdn, mx.zeros((1, 4095, 8), dtype=mx.float16)
    )
    assert result is not None
    assert len(scheduled) == 1
    assert all(
        actual is expected for actual, expected in zip(scheduled[0], result)
    )
    mx.eval(*result)

    assert [part.shape for part in result] == [(1, 4095, 1)] * 4
    assert calls == [
        ("ane", 2048),
        (10, 2047),
        (20, 2047),
        (30, 2047),
        (40, 2047),
    ]
    assert [part[0, 0, 0].item() for part in result] == [1, 2, 3, 4]
    assert [part[0, -1, 0].item() for part in result] == [10, 20, 30, 40]


def test_gdn_profitable_tail_is_padded_before_recurrence(monkeypatch):
    calls = []

    def exact(_gdn, block, _target_verify=False):
        calls.append((int(block.shape[-2]), float(mx.sum(block).item())))
        return tuple(
            mx.full((1, block.shape[-2], 1), value, dtype=block.dtype)
            for value in (1, 2, 3, 4)
        )

    monkeypatch.setattr(ane_patch, "_gdn_backend_exact", exact)
    gdn = SimpleNamespace(
        _omlx_ane_gdn_config=ane_patch._AneGDNConfig(
            2048, 0.50, 8, tail_padding_min_tokens=1358
        )
    )
    x = mx.ones((1, 4095, 8), dtype=mx.float16)

    result = ane_patch._gdn_backend(gdn, x)
    assert result is not None
    mx.eval(*result)

    assert [part.shape for part in result] == [(1, 4095, 1)] * 4
    assert calls == [(2048, 16384.0), (2048, 16376.0)]
    assert [part[0, -1, 0].item() for part in result] == [1, 2, 3, 4]


def test_install_dispatch_adds_gdn_projection_compatibility_hook(monkeypatch):
    fallback = object()
    accelerated = object()

    def target_linears(linears, x, target_verify=False):
        return fallback

    vlm = SimpleNamespace(
        Qwen3_5MLP=None,
        register_qwen3_5_mlp_prefill_backend=lambda backend: None,
        _target_verify_linears=target_linears,
    )
    lm = SimpleNamespace(MLP=None)

    def import_module(name):
        if name == "mlx_vlm.models.qwen3_5.language":
            return vlm
        if name == "mlx_lm.models.qwen3_5":
            return lm
        raise ImportError(name)

    monkeypatch.setattr(ane_patch.importlib, "import_module", import_module)
    monkeypatch.setattr(ane_patch, "_VLM_HOOK_INSTALLED", False)
    monkeypatch.setattr(ane_patch, "_VLM_GDN_HOOK_INSTALLED", False)
    monkeypatch.setattr(ane_patch, "_GDN_MODULES", weakref.WeakValueDictionary())
    monkeypatch.setattr(
        ane_patch, "_gdn_backend", lambda gdn, x, target_verify=False: accelerated
    )

    gdn = _GDN()
    ane_patch._register_gdn_module(gdn)

    assert ane_patch._install_dispatch()
    assert (
        vlm._target_verify_linears(
            (gdn.in_proj_qkv, gdn.in_proj_z, gdn.in_proj_b, gdn.in_proj_a),
            mx.zeros((1, 1, 128)),
        )
        is accelerated
    )
    assert vlm._target_verify_linears((object(),), mx.zeros((1, 1, 128))) is fallback


def test_install_dispatch_registers_mlx_lm_gdn_backend(monkeypatch):
    import omlx.patches.qwen35_q4_mlp as q4patch

    registrations = []
    vlm = SimpleNamespace(Qwen3_5MLP=None)
    lm = SimpleNamespace(MLP=None)

    def import_module(name):
        if name == "mlx_vlm.models.qwen3_5.language":
            return vlm
        if name == "mlx_lm.models.qwen3_5":
            return lm
        raise ImportError(name)

    monkeypatch.setattr(ane_patch.importlib, "import_module", import_module)
    monkeypatch.setattr(
        q4patch,
        "register_qwen35_lm_gdn_prefill_backend",
        registrations.append,
    )

    assert ane_patch._install_dispatch()
    assert registrations == [ane_patch._gdn_backend]


def test_enable_marks_only_requested_number_of_loaded_mlps(monkeypatch):
    monkeypatch.setattr(fast, "qwen35_ane_available", lambda: True)
    monkeypatch.setattr(fast, "has_symbol", lambda name: False)
    monkeypatch.setattr(ane_patch, "_install_dispatch", lambda: True)
    monkeypatch.setattr(ane_patch, "_eligible_pair", lambda mlp: True)
    compiled = []

    def compile_pair(mlp, config):
        compiled.append((mlp, config))
        return object()

    monkeypatch.setattr(ane_patch, "_compile_pair", compile_pair)
    model = _Model(4)

    count = ane_patch.enable_qwen35_ane_prefill(
        model,
        sequence_length=2048,
        fraction=0.4,
        max_layers=2,
    )

    assert count == 2
    assert len(compiled) == 2
    marked = [hasattr(layer, "_omlx_ane_prefill_config") for layer in model.layers]
    assert sum(marked) == 2
    assert all(
        hasattr(layer, "_omlx_ane_prefill_state")
        for layer in model.layers
        if hasattr(layer, "_omlx_ane_prefill_config")
    )


@pytest.mark.parametrize("available", [False, True])
def test_cpu_shared_resource_scheduler_is_capability_guarded(
    monkeypatch, available
):
    monkeypatch.setattr(fast, "qwen35_ane_available", lambda: True)
    monkeypatch.setattr(
        fast, "qwen35_cpu_shared_resource_available", lambda: available
    )
    monkeypatch.setattr(ane_patch, "_install_dispatch", lambda: True)
    monkeypatch.setattr(ane_patch, "_eligible_pair", lambda mlp: True)
    captured = []

    def enable_banks(model, candidates, config, **kwargs):
        captured.append(config)
        return (1, 1, 0, 2)

    monkeypatch.setattr(ane_patch, "_enable_dual_procedure_banks", enable_banks)
    model = _Model(1)
    model.layers[0].gate_proj.scales = model.layers[0].gate_proj.scales.astype(
        mx.float16
    )

    count = ane_patch.enable_qwen35_ane_prefill(
        model,
        sequence_length=2048,
        cpu_fraction=0.125,
        cpu_threads=8,
        cpu_shared_resource=True,
    )

    assert count == 1
    assert captured[0].cpu_threads == 8
    assert captured[0].cpu_shared_resource is available


def test_down_projection_cpu_share_is_prepared_and_dispatched(monkeypatch):
    monkeypatch.setattr(fast, "has_symbol", lambda name: True)
    mlp = _MLP()
    linear = mlp.down_proj
    linear.scales = linear.scales.astype(mx.float16)
    linear.biases = linear.biases.astype(mx.float16)
    state = ane_patch._prepare_cpu_linear(linear, 0.5)

    assert state is not None
    assert state.weight.shape == (64, 256)
    assert state.gpu_weight.shape[0] == 64

    captured = []

    def hybrid(*args):
        captured.append(args)
        return mx.zeros((1, 1, 128), dtype=mx.float16)

    monkeypatch.setattr(fast, "qwen35_cpu_fp16_affine_qmm_t", hybrid)
    result = ane_patch._post_ane_linear(
        linear,
        mx.zeros((1, 1, 256), dtype=mx.float16),
        8,
        q8_threshold_env="OMLX_TEST_Q8_THRESHOLD",
        cpu_state=state,
        cpu_threads=8,
        cpu_shared_resource=True,
    )

    assert result.shape == (1, 1, 128)
    assert captured[0][-2:] == (8, True)


def test_enable_caps_dual_layers_at_resident_program_budget(monkeypatch):
    monkeypatch.setattr(fast, "qwen35_ane_available", lambda: True)
    monkeypatch.setattr(fast, "has_symbol", lambda name: False)
    monkeypatch.setattr(ane_patch, "_install_dispatch", lambda: True)
    monkeypatch.setattr(ane_patch, "_eligible_pair", lambda mlp: True)
    monkeypatch.setattr(ane_patch, "_ANE_RESIDENT_PROGRAM_LIMIT", 4)
    monkeypatch.setattr(
        ane_patch,
        "_compile_pair",
        lambda mlp, config: SimpleNamespace(model1=object()),
    )
    model = _Model(4)

    count = ane_patch.enable_qwen35_ane_prefill(
        model,
        sequence_length=2048,
        fraction=0.5,
        max_layers=4,
        dual_ane=True,
    )

    assert count == 2
    assert model._omlx_ane_dual_prefill_count == 2
    assert model._omlx_ane_resident_program_count == 4


def test_enable_logs_gdn_starvation_when_budget_exhausted(monkeypatch, caplog):
    monkeypatch.setattr(fast, "qwen35_ane_available", lambda: True)
    monkeypatch.setattr(fast, "has_symbol", lambda name: False)
    monkeypatch.setattr(ane_patch, "_install_dispatch", lambda: True)
    monkeypatch.setattr(ane_patch, "_eligible_pair", lambda mlp: True)
    monkeypatch.setattr(ane_patch, "_ANE_RESIDENT_PROGRAM_LIMIT", 4)
    monkeypatch.setattr(
        ane_patch,
        "_compile_pair",
        lambda mlp, config: SimpleNamespace(model1=object()),
    )
    model = _Model(2)

    with caplog.at_level(logging.INFO):
        count = ane_patch.enable_qwen35_ane_prefill(
            model,
            sequence_length=2048,
            fraction=0.5,
            max_layers=2,
            dual_ane=True,
            gdn=True,
        )

    assert count == 2
    assert model._omlx_ane_gdn_prefill_count == 0
    assert model._omlx_ane_procedure_count == 2
    messages = [record.getMessage() for record in caplog.records]
    assert any("budget exhausted before GDN" in message for message in messages)
    assert any("Stopped eager ANE preparation" in message for message in messages)


def test_enable_packs_all_dual_layers_into_two_procedure_banks(monkeypatch):
    monkeypatch.setattr(fast, "qwen35_ane_available", lambda: True)
    monkeypatch.setattr(fast, "has_symbol", lambda name: True)
    monkeypatch.setattr(
        fast, "qwen35_ane_linear_bank_builder", _no_bank_builder
    )
    monkeypatch.setattr(ane_patch, "_install_dispatch", lambda: True)
    monkeypatch.setattr(ane_patch, "_eligible_pair", lambda mlp: True)
    compiled = []

    def compile_bank(weights, sequence_length, ane_instance):
        models = [object() for _ in weights]
        compiled.append((len(weights), sequence_length, ane_instance, models))
        return models

    monkeypatch.setattr(fast, "qwen35_ane_compile_linear_bank", compile_bank)
    model = _Model(4)

    count = ane_patch.enable_qwen35_ane_prefill(
        model,
        sequence_length=2048,
        fraction=0.5,
        max_layers=4,
        dual_ane=True,
    )

    assert count == 4
    assert [(n, sequence, instance) for n, sequence, instance, _ in compiled] == [
        (4, 2048, 1),
        (4, 2048, 2),
    ]
    assert model._omlx_ane_dual_prefill_count == 4
    assert model._omlx_ane_resident_program_count == 2
    assert model._omlx_ane_procedure_count == 4
    assert {id(layer._omlx_ane_prefill_state.model) for layer in model.layers} == {
        id(value) for value in compiled[0][3]
    }
    assert {id(layer._omlx_ane_prefill_state.model1) for layer in model.layers} == {
        id(value) for value in compiled[1][3]
    }


def test_bank_chunk_spans_respects_byte_cap():
    weights = [mx.zeros((4, 4), dtype=mx.int8) for _ in range(4)]

    spans = ane_patch._bank_chunk_spans(weights, 2 * 16)

    assert spans == [(0, 2), (2, 4)]
    assert ane_patch._bank_chunk_spans(weights, 1) == [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert ane_patch._bank_chunk_spans(weights, 1 << 30) == [(0, 4)]


def test_ane_bank_memory_headroom_ok_math(monkeypatch):
    """Regression for a live jetsam kill: process phys_footprint climbed to
    48.3GB on a 48GB machine across a bounded 4-attempt retry ladder, each
    attempt individually dropping its own references but racing the ANE
    driver's asynchronous device-mapping release. The gate compares against
    total system memory (the same ledger jetsam uses), not oMLX's own
    configured ceiling, since it has to work even before any ceiling has
    propagated."""
    import omlx.utils.hardware as hardware
    import omlx.utils.proc_memory as proc_memory

    gib = 1024**3
    monkeypatch.setattr(hardware, "get_total_memory_bytes", lambda: 48 * gib)

    monkeypatch.setattr(proc_memory, "get_phys_footprint", lambda: int(30 * gib))
    assert ane_patch._ane_bank_memory_headroom_ok() is True  # 62.5% < 70%

    monkeypatch.setattr(proc_memory, "get_phys_footprint", lambda: int(34 * gib))
    assert ane_patch._ane_bank_memory_headroom_ok() is False  # ~70.8% >= 70%

    # total==0 (measurement unavailable): default to allowing the attempt.
    monkeypatch.setattr(hardware, "get_total_memory_bytes", lambda: 0)
    assert ane_patch._ane_bank_memory_headroom_ok() is True


def test_ane_bank_memory_headroom_ok_defaults_true_on_error(monkeypatch):
    import omlx.utils.hardware as hardware

    def boom():
        raise RuntimeError("sysctl unavailable")

    monkeypatch.setattr(hardware, "get_total_memory_bytes", boom)
    assert ane_patch._ane_bank_memory_headroom_ok() is True


def test_ane_bank_memory_footprint_snapshot_reports_measured_bytes(monkeypatch):
    """The skip-warning log needs the actual measured numbers, not just the
    fixed threshold constant, so a future "why is ANE slow on this box" is
    answerable from one log line."""
    import omlx.utils.hardware as hardware
    import omlx.utils.proc_memory as proc_memory

    gib = 1024**3
    monkeypatch.setattr(proc_memory, "get_phys_footprint", lambda: 34 * gib)
    monkeypatch.setattr(hardware, "get_total_memory_bytes", lambda: 48 * gib)

    assert ane_patch._ane_bank_memory_footprint_snapshot() == (34 * gib, 48 * gib)


def test_ane_bank_memory_footprint_snapshot_defaults_zero_on_error(monkeypatch):
    import omlx.utils.hardware as hardware

    def boom():
        raise RuntimeError("sysctl unavailable")

    monkeypatch.setattr(hardware, "get_total_memory_bytes", boom)
    assert ane_patch._ane_bank_memory_footprint_snapshot() == (0, 0)


def test_bank_split_ladder_stops_retrying_when_headroom_runs_out(monkeypatch):
    """The circuit breaker must stop issuing further compile attempts (not
    just refuse to start) once memory tightens mid-ladder -- this is what
    actually prevents a jetsam kill, since a machine can have headroom for
    the first attempt but not the later, smaller-but-still-compounding
    ones."""
    monkeypatch.setattr(ane_patch.time, "sleep", lambda *_a: None)
    calls = []

    def compile_span(start, stop):
        calls.append((start, stop))
        raise RuntimeError("ANE procedure bank load failed: 0x20004")

    headroom_calls = []

    def headroom_ok():
        headroom_calls.append(True)
        return len(headroom_calls) == 1  # only the first attempt has headroom

    monkeypatch.setattr(ane_patch, "_ane_bank_memory_headroom_ok", headroom_ok)
    monkeypatch.setattr(
        ane_patch, "_ane_bank_memory_footprint_snapshot", lambda: (0, 0)
    )

    result = ane_patch._bank_split_ladder([4, 4, 4, 4], compile_span)

    assert result is None
    assert calls == [(0, 4)]  # only the first (monolithic) attempt ran
    assert len(headroom_calls) == 2  # gate checked before attempt 1 and 2


def test_bank_split_ladder_settles_between_failed_attempts(monkeypatch):
    """A failed attempt must give the ANE driver's asynchronous device
    -mapping release a moment before the next attempt's headroom check
    measures phys_footprint -- otherwise the check races the driver and can
    under- or over-react to memory that hasn't actually been freed yet."""
    monkeypatch.setattr(ane_patch, "_ane_bank_memory_headroom_ok", lambda: True)
    sleeps = []
    monkeypatch.setattr(ane_patch.time, "sleep", lambda s: sleeps.append(s))
    gc_calls = []
    monkeypatch.setattr(ane_patch.gc, "collect", lambda: gc_calls.append(True))

    def compile_span(start, stop):
        raise RuntimeError("ANE procedure bank load failed: 0x20004")

    ane_patch._bank_split_ladder([4, 4, 4, 4], compile_span)

    assert len(gc_calls) >= 1
    assert sleeps and all(s == ane_patch._ANE_BANK_RETRY_SETTLE_SECONDS for s in sleeps)


def test_compile_single_banks_stops_retrying_when_headroom_runs_out(monkeypatch):
    """Same circuit breaker, applied to the not-yet-refactored single-ANE
    calibration path (which has its own inline copy of the retry ladder)."""
    monkeypatch.setattr(ane_patch.time, "sleep", lambda *_a: None)
    calls = []

    def compile_bank(weights, sequence_length, ane_instance):
        calls.append(len(weights))
        raise RuntimeError("ANE procedure bank load failed: 0x20004")

    monkeypatch.setattr(fast, "qwen35_ane_compile_linear_bank", compile_bank)

    headroom_calls = []

    def headroom_ok():
        headroom_calls.append(True)
        return len(headroom_calls) == 1

    monkeypatch.setattr(ane_patch, "_ane_bank_memory_headroom_ok", headroom_ok)
    monkeypatch.setattr(
        ane_patch, "_ane_bank_memory_footprint_snapshot", lambda: (0, 0)
    )

    weights = [mx.zeros((4, 4), dtype=mx.int8) for _ in range(4)]
    result = ane_patch._compile_single_banks(weights, 2048)

    assert result is None
    assert calls == [4]
    assert len(headroom_calls) == 2


def test_compile_single_bank_targets_unpinned_instance(monkeypatch):
    weights = [mx.zeros((4, 4), dtype=mx.int8) for _ in range(3)]
    calls = []

    def compile_bank(values, sequence_length, ane_instance):
        calls.append((len(values), sequence_length, ane_instance))
        return [object() for _ in values]

    monkeypatch.delenv("OMLX_QWEN35_ANE_BANK_MAX_BYTES", raising=False)
    monkeypatch.setattr(fast, "qwen35_ane_compile_linear_bank", compile_bank)
    monkeypatch.setattr(ane_patch, "_ane_bank_memory_headroom_ok", lambda: True)

    result = ane_patch._compile_single_banks(weights, 2048)

    assert result is not None
    models, resident = result
    assert len(models) == 3
    assert resident == 1
    assert calls == [(3, 2048, 0)]


def test_enable_splits_banks_when_monolithic_load_fails(monkeypatch):
    monkeypatch.delenv("OMLX_QWEN35_ANE_BANK_MAX_BYTES", raising=False)
    monkeypatch.setattr(fast, "qwen35_ane_available", lambda: True)
    monkeypatch.setattr(fast, "has_symbol", lambda name: True)
    monkeypatch.setattr(
        fast, "qwen35_ane_linear_bank_builder", _no_bank_builder
    )
    monkeypatch.setattr(ane_patch, "_install_dispatch", lambda: True)
    monkeypatch.setattr(ane_patch, "_eligible_pair", lambda mlp: True)
    monkeypatch.setattr(ane_patch, "_ane_bank_memory_headroom_ok", lambda: True)
    monkeypatch.setattr(ane_patch.time, "sleep", lambda *_a: None)
    compiled = []

    def compile_bank(weights, sequence_length, ane_instance):
        compiled.append((len(weights), ane_instance))
        if len(weights) > 2:
            raise RuntimeError("ANE procedure bank load failed: 0x20004")
        return [object() for _ in weights]

    monkeypatch.setattr(fast, "qwen35_ane_compile_linear_bank", compile_bank)
    model = _Model(4)

    count = ane_patch.enable_qwen35_ane_prefill(
        model,
        sequence_length=2048,
        fraction=0.5,
        max_layers=4,
        dual_ane=True,
    )

    # Ladder: monolithic fails, the near-half retry (3 procedures with the
    # slack term) fails too, and the halved cap lands on single-procedure
    # banks that succeed.
    assert count == 4
    assert compiled == [
        (4, 1),
        (3, 1),
        (1, 1),
        (1, 2),
        (1, 1),
        (1, 2),
        (1, 1),
        (1, 2),
        (1, 1),
        (1, 2),
    ]
    assert model._omlx_ane_resident_program_count == 8
    assert model._omlx_ane_procedure_count == 4
    assert all(
        layer._omlx_ane_prefill_state.model1 is not None for layer in model.layers
    )


def test_enable_first_retry_is_a_near_half_split(monkeypatch):
    monkeypatch.delenv("OMLX_QWEN35_ANE_BANK_MAX_BYTES", raising=False)
    monkeypatch.setattr(fast, "qwen35_ane_available", lambda: True)
    monkeypatch.setattr(fast, "has_symbol", lambda name: True)
    monkeypatch.setattr(
        fast, "qwen35_ane_linear_bank_builder", _no_bank_builder
    )
    monkeypatch.setattr(ane_patch, "_install_dispatch", lambda: True)
    monkeypatch.setattr(ane_patch, "_eligible_pair", lambda mlp: True)
    monkeypatch.setattr(ane_patch, "_ane_bank_memory_headroom_ok", lambda: True)
    monkeypatch.setattr(ane_patch.time, "sleep", lambda *_a: None)
    compiled = []

    def compile_bank(weights, sequence_length, ane_instance):
        compiled.append((len(weights), ane_instance))
        if len(weights) == 4:
            raise RuntimeError("ANE procedure bank load failed: 0x20004")
        return [object() for _ in weights]

    monkeypatch.setattr(fast, "qwen35_ane_compile_linear_bank", compile_bank)
    model = _Model(4)

    count = ane_patch.enable_qwen35_ane_prefill(
        model,
        sequence_length=2048,
        fraction=0.5,
        max_layers=4,
        dual_ane=True,
    )

    assert count == 4
    assert compiled == [(4, 1), (3, 1), (3, 2), (1, 1), (1, 2)]
    assert model._omlx_ane_resident_program_count == 4


def test_enable_env_cap_forces_split_banks(monkeypatch):
    monkeypatch.setenv("OMLX_QWEN35_ANE_BANK_MAX_BYTES", "1")
    monkeypatch.setattr(fast, "qwen35_ane_available", lambda: True)
    monkeypatch.setattr(fast, "has_symbol", lambda name: True)
    monkeypatch.setattr(
        fast, "qwen35_ane_linear_bank_builder", _no_bank_builder
    )
    monkeypatch.setattr(ane_patch, "_install_dispatch", lambda: True)
    monkeypatch.setattr(ane_patch, "_eligible_pair", lambda mlp: True)
    monkeypatch.setattr(ane_patch, "_ane_bank_memory_headroom_ok", lambda: True)
    monkeypatch.setattr(ane_patch.time, "sleep", lambda *_a: None)
    compiled = []

    def compile_bank(weights, sequence_length, ane_instance):
        compiled.append((len(weights), ane_instance))
        return [object() for _ in weights]

    monkeypatch.setattr(fast, "qwen35_ane_compile_linear_bank", compile_bank)
    model = _Model(4)

    count = ane_patch.enable_qwen35_ane_prefill(
        model,
        sequence_length=2048,
        fraction=0.5,
        max_layers=4,
        dual_ane=True,
    )

    assert count == 4
    assert all(size == 1 for size, _ in compiled)
    assert len(compiled) == 8
    assert model._omlx_ane_resident_program_count == 8


def test_enable_falls_back_to_per_layer_when_split_banks_fail(monkeypatch):
    monkeypatch.delenv("OMLX_QWEN35_ANE_BANK_MAX_BYTES", raising=False)
    monkeypatch.setattr(fast, "qwen35_ane_available", lambda: True)
    monkeypatch.setattr(fast, "has_symbol", lambda name: True)
    monkeypatch.setattr(
        fast, "qwen35_ane_linear_bank_builder", _no_bank_builder
    )
    monkeypatch.setattr(ane_patch, "_install_dispatch", lambda: True)
    monkeypatch.setattr(ane_patch, "_eligible_pair", lambda mlp: True)
    monkeypatch.setattr(ane_patch, "_ane_bank_memory_headroom_ok", lambda: True)
    monkeypatch.setattr(ane_patch.time, "sleep", lambda *_a: None)

    def compile_bank(weights, sequence_length, ane_instance):
        raise RuntimeError("ANE procedure bank load failed: 0x20004")

    monkeypatch.setattr(fast, "qwen35_ane_compile_linear_bank", compile_bank)
    per_layer = []

    def compile_pair(mlp, config):
        per_layer.append(mlp)
        return SimpleNamespace(model1=object())

    monkeypatch.setattr(ane_patch, "_compile_pair", compile_pair)
    model = _Model(4)

    count = ane_patch.enable_qwen35_ane_prefill(
        model,
        sequence_length=2048,
        fraction=0.5,
        max_layers=4,
        dual_ane=True,
    )

    assert count == 4
    assert len(per_layer) == 4
    assert model._omlx_ane_dual_prefill_count == 4


def test_compile_pair_builds_one_combined_ane_program(monkeypatch):
    mlp = _MLP()
    for linear in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
        linear.scales = linear.scales.astype(mx.bfloat16)
        linear.biases = linear.biases.astype(mx.bfloat16)
    config = ane_patch._AnePrefillConfig(2048, 0.5, 8)
    compiled = []

    def compile_linear(weight, sequence_length):
        mx.eval(weight)
        compiled.append((weight.shape, weight.dtype, sequence_length))
        return object()

    monkeypatch.setattr(fast, "qwen35_ane_compile_linear", compile_linear)
    monkeypatch.setattr(
        fast,
        "has_symbol",
        lambda name: False,
    )

    state = ane_patch._compile_pair(mlp, config)

    assert state is not None
    assert compiled == [((256, 128), mx.float32, 2048)]
    assert state.ane_outputs == 128
    assert state.gpu_outputs == 128
    assert state.weight.shape == (256, 16)
    assert state.scales.shape == (256, 1)
    assert state.biases.shape == (256, 1)


def test_compile_pair_splits_one_prompt_across_two_ane_instances(monkeypatch):
    mlp = _MLP()
    for linear in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
        linear.scales = linear.scales.astype(mx.bfloat16)
        linear.biases = linear.biases.astype(mx.bfloat16)
    compiled = []

    def compile_linear(weight, sequence_length, ane_instance=0):
        mx.eval(weight)
        model = object()
        compiled.append((weight.shape, sequence_length, ane_instance, model))
        return model

    monkeypatch.setattr(fast, "qwen35_ane_compile_linear", compile_linear)
    monkeypatch.setattr(
        fast,
        "has_symbol",
        lambda name: (
            name
            in {
                "qwen35_ane_dual_affine_qmm_t",
                "qwen35_ane_dual_q4_swiglu_t",
            }
        ),
    )

    state = ane_patch._compile_pair(
        mlp, ane_patch._AnePrefillConfig(2048, 0.5, 8, dual_ane=True)
    )

    assert state is not None
    assert [
        (shape, sequence, instance) for shape, sequence, instance, _ in compiled
    ] == [
        ((128, 128), 2048, 1),
        ((128, 128), 2048, 2),
    ]
    assert state.model is compiled[0][3]
    assert state.model1 is compiled[1][3]
    assert state.ane_outputs == 128
    assert state.gpu_outputs == 128


def test_prepare_pair_accepts_oq4e_group64_with_q5_down():
    mlp = _OQ4eMLP()
    for linear in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
        linear.scales = linear.scales.astype(mx.bfloat16)
        linear.biases = linear.biases.astype(mx.bfloat16)

    prepared = ane_patch._prepare_pair_for_bank(
        mlp,
        ane_patch._AnePrefillConfig(2048, 0.5, 8, dual_ane=True),
    )

    assert prepared is not None
    state, dense0, dense1 = prepared
    assert state.group_size == 64
    assert state.weight.shape == (256, 16)
    assert state.scales.shape == (256, 2)
    assert dense0.shape == (128, 128)
    assert dense1.shape == (128, 128)


def test_prepare_pair_single_ane_keeps_one_full_prefix():
    mlp = _MLP()
    for linear in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
        linear.scales = linear.scales.astype(mx.bfloat16)
        linear.biases = linear.biases.astype(mx.bfloat16)

    prepared = ane_patch._prepare_pair_for_bank(
        mlp,
        ane_patch._AnePrefillConfig(2048, 0.5, 8, dual_ane=False),
    )

    assert prepared is not None
    state, dense0, dense1 = prepared
    assert state.ane_outputs == 128
    assert dense0.shape == (256, 128)
    assert dense1 is None


def test_eligible_pair_preserves_q4_and_accepts_affine_q8():
    mlp = _MLP()
    for linear in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
        linear.scales = linear.scales.astype(mx.bfloat16)
        linear.biases = linear.biases.astype(mx.bfloat16)
    q8_mlp = _OQ8MLP()
    for linear in (q8_mlp.gate_proj, q8_mlp.up_proj, q8_mlp.down_proj):
        linear.scales = linear.scales.astype(mx.bfloat16)
        linear.biases = linear.biases.astype(mx.bfloat16)

    assert ane_patch._eligible_pair(mlp)
    assert ane_patch._eligible_pair(q8_mlp)


@pytest.mark.parametrize("dual", [False, True])
def test_q6_mlp_is_eligible_and_uses_generic_fused_swiglu(monkeypatch, dual):
    assert ane_patch._eligible_pair(_make_affine_mlp(6, 64))
    generic_name = ane_patch._fused_swiglu_symbol(6, dual=dual)
    assert generic_name == (
        "qwen35_ane_dual_affine_swiglu_t"
        if dual
        else "qwen35_ane_affine_swiglu_t"
    )
    q4_name = "qwen35_ane_dual_q4_swiglu_t" if dual else "qwen35_ane_q4_swiglu_t"
    captured = {}
    activation = mx.zeros((1, 1, 4), dtype=mx.bfloat16)

    monkeypatch.setattr(fast, "has_symbol", lambda name: name == generic_name)

    def fused(*args):
        captured["args"] = args
        return activation

    monkeypatch.setattr(fast, generic_name, fused, raising=False)
    monkeypatch.setattr(
        fast,
        q4_name,
        lambda *args: pytest.fail("Q6 must use the generic fused SwiGLU"),
    )

    import omlx.patches.qwen35_q4_mlp as q4_patch

    monkeypatch.setattr(q4_patch, "_linear_qmm", lambda linear, value, variant: value)
    model0 = object()
    model1 = object() if dual else None
    state = ane_patch._CombinedMLPState(
        model=model0,
        model1=model1,
        weight=mx.zeros((4, 3), dtype=mx.uint32),
        scales=mx.zeros((4, 1), dtype=mx.bfloat16),
        biases=mx.zeros((4, 1), dtype=mx.bfloat16),
        ane_outputs=2,
        gpu_outputs=2,
        bits=6,
        group_size=64,
    )
    mlp = SimpleNamespace(
        down_proj=object(),
        _omlx_ane_prefill_config=ane_patch._AnePrefillConfig(
            1, 0.5, 8, dual_ane=dual
        ),
        _omlx_ane_prefill_state=state,
    )
    x = mx.zeros((1, 1, 16), dtype=mx.bfloat16)

    result = ane_patch._backend(mlp, x)
    mx.eval(result)

    expected = (
        x,
        state.weight,
        state.scales,
        state.biases,
        model0,
        *(() if not dual else (model1,)),
        6,
        8,
        64,
    )
    assert captured["args"] == expected


def test_compile_pair_cache_identity_includes_bits_and_group_size(monkeypatch):
    mlp = _make_affine_mlp(bits=6, group_size=64)
    compiled = []

    monkeypatch.setattr(
        fast,
        "has_symbol",
        lambda name: name == "qwen35_ane_affine_swiglu_t",
    )

    def compile_linear(weight, sequence_length):
        compiled.append((weight.shape, sequence_length))
        return object()

    monkeypatch.setattr(fast, "qwen35_ane_compile_linear", compile_linear)
    config = ane_patch._AnePrefillConfig(2048, 0.5, 8)

    first = ane_patch._compile_pair(mlp, config)
    replacement = _make_affine_mlp(bits=8, group_size=128)
    mlp.gate_proj = replacement.gate_proj
    mlp.up_proj = replacement.up_proj
    mlp.down_proj = replacement.down_proj
    second = ane_patch._compile_pair(mlp, config)

    assert first is not None and second is not None
    assert (first.bits, first.group_size) == (6, 64)
    assert (second.bits, second.group_size) == (8, 128)
    assert first is not second
    assert len(compiled) == 2
    assert len(mlp._omlx_ane_prefill_cache) == 2


def test_prepare_pair_tracks_q8_bits_and_packed_shape(monkeypatch):
    mlp = _OQ8MLP()
    for linear in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
        linear.scales = linear.scales.astype(mx.bfloat16)
        linear.biases = linear.biases.astype(mx.bfloat16)
    monkeypatch.setattr(
        fast,
        "has_symbol",
        lambda name: name == "qwen35_ane_dual_affine_swiglu_t",
    )

    prepared = ane_patch._prepare_pair_for_bank(
        mlp,
        ane_patch._AnePrefillConfig(2048, 0.5, 8, dual_ane=True),
    )

    assert prepared is not None
    state, dense0, dense1 = prepared
    assert state.bits == 8
    assert state.weight.shape == (256, 32)
    assert state.scales.shape == (256, 2)
    assert dense0.shape == (128, 128)
    assert dense1.shape == (128, 128)


def test_compile_pair_skips_q8_without_generic_fused_swiglu(monkeypatch):
    mlp = _OQ8MLP()
    for linear in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
        linear.scales = linear.scales.astype(mx.bfloat16)
        linear.biases = linear.biases.astype(mx.bfloat16)

    monkeypatch.setattr(fast, "has_symbol", lambda name: False)
    monkeypatch.setattr(
        fast,
        "qwen35_ane_compile_linear",
        lambda *args: pytest.fail("Q8 must not prepare an unfused ANE path"),
    )

    assert (
        ane_patch._compile_pair(
            mlp,
            ane_patch._AnePrefillConfig(2048, 0.5, 8),
        )
        is None
    )


def test_compile_gdn_accepts_q8_and_propagates_bits(monkeypatch):
    gdn = _OQ8GDN()
    for linear in (
        gdn.in_proj_qkv,
        gdn.in_proj_z,
        gdn.in_proj_b,
        gdn.in_proj_a,
    ):
        linear.scales = linear.scales.astype(mx.bfloat16)
        linear.biases = linear.biases.astype(mx.bfloat16)
    compiled = []

    def compile_linear(weight, sequence_length):
        mx.eval(weight)
        compiled.append((weight.shape, weight.dtype, sequence_length))
        return object()

    monkeypatch.setattr(fast, "qwen35_ane_compile_linear", compile_linear)
    monkeypatch.setattr(
        fast,
        "has_symbol",
        lambda name: name == "qwen35_ane_affine_qmm_t",
    )

    state = ane_patch._compile_gdn(gdn, ane_patch._AneGDNConfig(2048, 0.5, 8))

    assert state is not None
    assert state.bits == 8
    assert state.group_size == 64
    assert state.weight.shape == (256, 32)
    assert state.scales.shape == (256, 2)
    assert compiled == [((128, 128), mx.float32, 2048)]


@pytest.mark.parametrize("group_size", [64, 128])
def test_q6_gdn_packs_suffix_and_extracts_b_a(group_size, monkeypatch):
    gdn = _make_affine_gdn(6, group_size)
    compiled = []

    monkeypatch.setattr(
        fast,
        "qwen35_ane_compile_linear",
        lambda weight, sequence_length: compiled.append(weight.shape) or object(),
    )
    monkeypatch.setattr(
        fast,
        "has_symbol",
        lambda name: name == "qwen35_ane_affine_qmm_t",
    )

    config = ane_patch._AneGDNConfig(1, 0.5, 8)
    state = ane_patch._compile_gdn(gdn, config)

    assert ane_patch._eligible_gdn(gdn)
    assert state is not None
    assert state.bits == 6
    assert state.group_size == group_size
    assert state.weight.shape == (384, 24)
    assert state.scales.shape == (384, 128 // group_size)
    assert state.biases.shape == state.scales.shape
    assert state.b_outputs == 48
    assert state.a_outputs == 48
    total_outputs = (
        state.z_outputs + state.qkv_outputs + state.b_outputs + state.a_outputs
    )
    combined = mx.arange(total_outputs).reshape(1, 1, total_outputs).astype(
        mx.bfloat16
    )
    monkeypatch.setattr(
        fast,
        "qwen35_ane_affine_qmm_t",
        lambda *args: combined,
    )

    import omlx.patches.qwen35_q4_mlp as q4_patch

    monkeypatch.setattr(
        q4_patch,
        "_linear_qmm",
        lambda *args: pytest.fail("packed Q6 GDN must not launch b/a qmm"),
    )
    gdn._omlx_ane_gdn_config = config
    gdn._omlx_ane_gdn_state = state

    x = mx.zeros((1, 1, 128), dtype=mx.bfloat16)
    mixed_qkv, z, b, a = ane_patch._gdn_backend(gdn, x)
    mx.eval(mixed_qkv, z, b, a)

    assert compiled == [(128, 128)]
    assert z.shape == (1, 1, state.z_outputs)
    assert mixed_qkv.shape == (1, 1, state.qkv_outputs)
    assert b.shape == (1, 1, state.b_outputs)
    assert a.shape == (1, 1, state.a_outputs)
    assert b[0, 0, 0].item() == state.z_outputs + state.qkv_outputs
    assert a[0, 0, 0].item() == total_outputs - state.a_outputs


def test_backend_dispatches_single_q8_swiglu_with_bits(monkeypatch):
    activation = mx.zeros((1, 1, 4), dtype=mx.bfloat16)
    captured = {}

    monkeypatch.setattr(
        fast,
        "has_symbol",
        lambda name: name == "qwen35_ane_affine_swiglu_t",
    )

    def fused(*args):
        captured["args"] = args
        return activation

    monkeypatch.setattr(fast, "qwen35_ane_affine_swiglu_t", fused, raising=False)

    import omlx.patches.qwen35_q4_mlp as q4_patch

    monkeypatch.setattr(q4_patch, "_linear_qmm", lambda linear, value, variant: value)
    state = ane_patch._CombinedMLPState(
        model=object(),
        weight=mx.zeros((4, 4), dtype=mx.uint32),
        scales=mx.zeros((4, 2), dtype=mx.bfloat16),
        biases=mx.zeros((4, 2), dtype=mx.bfloat16),
        ane_outputs=2,
        gpu_outputs=2,
        bits=8,
    )
    mlp = SimpleNamespace(
        down_proj=object(),
        _omlx_ane_prefill_config=ane_patch._AnePrefillConfig(1, 0.5, 8),
        _omlx_ane_prefill_state=state,
    )
    x = mx.zeros((1, 1, 16), dtype=mx.bfloat16)

    result = ane_patch._backend(mlp, x)
    mx.eval(result)

    assert captured["args"] == (
        x,
        state.weight,
        state.scales,
        state.biases,
        state.model,
        8,
        8,
        128,
    )


def test_backend_dispatches_dual_q8_swiglu_with_bits(monkeypatch):
    activation = mx.zeros((1, 1, 4), dtype=mx.bfloat16)
    captured = {}

    monkeypatch.setattr(
        fast,
        "has_symbol",
        lambda name: name == "qwen35_ane_dual_affine_swiglu_t",
    )

    def fused(*args):
        captured["args"] = args
        return activation

    monkeypatch.setattr(
        fast,
        "qwen35_ane_dual_affine_swiglu_t",
        fused,
        raising=False,
    )

    import omlx.patches.qwen35_q4_mlp as q4_patch

    monkeypatch.setattr(q4_patch, "_linear_qmm", lambda linear, value, variant: value)
    model0, model1 = object(), object()
    state = ane_patch._CombinedMLPState(
        model=model0,
        model1=model1,
        weight=mx.zeros((4, 4), dtype=mx.uint32),
        scales=mx.zeros((4, 2), dtype=mx.bfloat16),
        biases=mx.zeros((4, 2), dtype=mx.bfloat16),
        ane_outputs=2,
        gpu_outputs=2,
        bits=8,
    )
    mlp = SimpleNamespace(
        down_proj=object(),
        _omlx_ane_prefill_config=ane_patch._AnePrefillConfig(1, 0.5, 8, dual_ane=True),
        _omlx_ane_prefill_state=state,
    )
    x = mx.zeros((1, 1, 16), dtype=mx.bfloat16)

    result = ane_patch._backend(mlp, x)
    mx.eval(result)

    assert captured["args"] == (
        x,
        state.weight,
        state.scales,
        state.biases,
        model0,
        model1,
        8,
        8,
        128,
    )
@pytest.mark.parametrize("bits", [5, 6, 8])
def test_prepare_pair_enables_cpu_gate_share_for_q5_plus(monkeypatch, bits):
    mlp = SimpleNamespace(
        gate_proj=nn.QuantizedLinear(
            128, 256, bias=False, group_size=64, bits=bits
        ),
        up_proj=nn.QuantizedLinear(
            128, 256, bias=False, group_size=64, bits=bits
        ),
        down_proj=nn.QuantizedLinear(
            256, 128, bias=False, group_size=64, bits=bits
        ),
    )
    for linear in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
        linear.scales = linear.scales.astype(mx.float16)
        linear.biases = linear.biases.astype(mx.float16)
    monkeypatch.setattr(
        fast,
        "has_symbol",
        lambda name: name == "qwen35_ane_dual_cpu_fp16_swiglu_t",
    )

    state = ane_patch._prepare_pair_runtime_state(
        mlp,
        ane_patch._AnePrefillConfig(
            2048, 0.5, 8, dual_ane=True, cpu_fraction=0.25
        ),
        object(),
        object(),
    )

    assert state is not None
    assert state.bits == bits
    assert state.cpu_outputs == 64
    assert state.cpu_weight is not None
    assert state.cpu_weight.shape == (128, 128)
    assert state.gpu_outputs == 64
    assert state.weight.shape == (128, 4 * bits)


@pytest.mark.parametrize("bits", [4, 5, 6, 8])
def test_prepare_pair_enables_cpu_gate_share_for_single_ane(monkeypatch, bits):
    mlp = SimpleNamespace(
        gate_proj=nn.QuantizedLinear(
            128, 256, bias=False, group_size=64, bits=bits
        ),
        up_proj=nn.QuantizedLinear(
            128, 256, bias=False, group_size=64, bits=bits
        ),
        down_proj=nn.QuantizedLinear(
            256, 128, bias=False, group_size=64, bits=bits
        ),
    )
    for linear in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
        linear.scales = linear.scales.astype(mx.float16)
        linear.biases = linear.biases.astype(mx.float16)
    expected_symbol = ane_patch._cpu_gate_kernel_symbol(bits, dual=False)
    monkeypatch.setattr(fast, "has_symbol", lambda name: name == expected_symbol)

    state = ane_patch._prepare_pair_runtime_state(
        mlp,
        ane_patch._AnePrefillConfig(
            2048, 0.5, 8, dual_ane=False, cpu_fraction=0.25
        ),
        object(),
        None,
    )

    assert state is not None
    assert state.model1 is None
    assert state.cpu_outputs == 64
    assert state.cpu_weight is not None
    assert state.cpu_weight.shape == (128, 128)
    assert state.gpu_outputs == 64


def test_compile_gdn_combines_z_then_qkv_and_keeps_q5_suffix(monkeypatch):
    gdn = _GDN()
    for linear in (
        gdn.in_proj_qkv,
        gdn.in_proj_z,
        gdn.in_proj_b,
        gdn.in_proj_a,
    ):
        linear.scales = linear.scales.astype(mx.bfloat16)
        linear.biases = linear.biases.astype(mx.bfloat16)
    compiled = []

    def compile_linear(weight, sequence_length):
        mx.eval(weight)
        compiled.append((weight.shape, weight.dtype, sequence_length))
        return object()

    monkeypatch.setattr(fast, "qwen35_ane_compile_linear", compile_linear)
    monkeypatch.setattr(
        fast,
        "has_symbol",
        lambda name: name == "qwen35_ane_affine_qmm_t",
    )

    state = ane_patch._compile_gdn(gdn, ane_patch._AneGDNConfig(2048, 0.5, 8))

    assert state is not None
    assert compiled == [((128, 128), mx.float32, 2048)]
    assert state.z_outputs == 128
    assert state.qkv_outputs == 256
    assert state.weight.shape == (256, 20)
    assert state.scales.shape == (256, 2)
    assert state.bits == 5
    assert state.group_size == 64


def test_prepare_gdn_accepts_oq4e_mixed_q4_q5_quantization():
    gdn = _OQ4eGDN()
    for linear in (
        gdn.in_proj_qkv,
        gdn.in_proj_z,
        gdn.in_proj_b,
        gdn.in_proj_a,
    ):
        linear.scales = linear.scales.astype(mx.bfloat16)
        linear.biases = linear.biases.astype(mx.bfloat16)

    prepared = ane_patch._prepare_gdn_for_bank(
        gdn,
        ane_patch._AneGDNConfig(2048, 0.75, 8, dual_ane=True),
    )

    assert prepared is not None
    state, dense0, dense1 = prepared
    assert state.bits == 4
    assert state.group_size == 64
    assert state.weight.shape == (256, 16)
    assert state.scales.shape == (256, 2)
    assert dense0.shape == (64, 128)
    assert dense1.shape == (64, 128)


def test_prepare_gdn_single_ane_keeps_one_full_prefix():
    gdn = _OQ4eGDN()
    for linear in (
        gdn.in_proj_qkv,
        gdn.in_proj_z,
        gdn.in_proj_b,
        gdn.in_proj_a,
    ):
        linear.scales = linear.scales.astype(mx.bfloat16)
        linear.biases = linear.biases.astype(mx.bfloat16)

    prepared = ane_patch._prepare_gdn_for_bank(
        gdn,
        ane_patch._AneGDNConfig(2048, 0.75, 8, dual_ane=False),
    )

    assert prepared is not None
    state, dense0, dense1 = prepared
    assert state.z_outputs == 128
    assert state.qkv_outputs == 256
    assert dense0.shape == (128, 128)
    assert dense1 is None


def test_prepare_gdn_splits_residual_qkv_across_cpu_and_gpu(monkeypatch):
    gdn = _OQ4eGDN()
    for linear in (
        gdn.in_proj_qkv,
        gdn.in_proj_z,
        gdn.in_proj_b,
        gdn.in_proj_a,
    ):
        linear.scales = linear.scales.astype(mx.float16)
        linear.biases = linear.biases.astype(mx.float16)
    monkeypatch.setattr(
        fast,
        "has_symbol",
        lambda name: name == "qwen35_ane_dual_cpu_fp16_affine_qmm_t",
    )

    prepared = ane_patch._prepare_gdn_for_bank(
        gdn,
        ane_patch._AneGDNConfig(
            2048, 0.5, 8, dual_ane=True, cpu_fraction=0.20
        ),
    )

    assert prepared is not None
    state, dense0, dense1 = prepared
    assert dense0.shape == (64, 128)
    assert dense1.shape == (64, 128)
    assert state.cpu_outputs == 64
    assert state.cpu_weight is not None
    assert state.cpu_weight.shape == (64, 128)
    assert state.weight.shape == (192, 16)
    assert state.scales.shape == (192, 2)


def test_prepare_gdn_splits_cpu_work_with_single_ane(monkeypatch):
    gdn = _OQ4eGDN()
    for linear in (
        gdn.in_proj_qkv,
        gdn.in_proj_z,
        gdn.in_proj_b,
        gdn.in_proj_a,
    ):
        linear.scales = linear.scales.astype(mx.float16)
        linear.biases = linear.biases.astype(mx.float16)
    monkeypatch.setattr(
        fast,
        "has_symbol",
        lambda name: name == "qwen35_ane_cpu_fp16_affine_qmm_t",
    )

    prepared = ane_patch._prepare_gdn_for_bank(
        gdn,
        ane_patch._AneGDNConfig(
            2048, 0.5, 8, dual_ane=False, cpu_fraction=0.20
        ),
    )

    assert prepared is not None
    state, dense0, dense1 = prepared
    assert dense0.shape == (128, 128)
    assert dense1 is None
    assert state.cpu_outputs == 64
    assert state.cpu_weight is not None
    assert state.cpu_weight.shape == (64, 128)
    assert state.weight.shape == (192, 16)


def test_gdn_backend_routes_cpu_split_through_three_way_native_merge(monkeypatch):
    combined = mx.array([[[1, 2, 10, 20, 30, 40]]], dtype=mx.float16)
    calls = []

    def hybrid(*args):
        calls.append(args)
        return combined

    monkeypatch.setattr(fast, "qwen35_ane_dual_cpu_fp16_affine_qmm_t", hybrid)
    import omlx.patches.qwen35_q4_mlp as q4_patch

    monkeypatch.setattr(
        q4_patch,
        "_linear_qmm",
        lambda linear, x, variant: mx.zeros((*x.shape[:-1], 1), dtype=x.dtype),
    )
    state = ane_patch._CombinedGDNState(
        model=object(),
        model1=object(),
        weight=mx.zeros((2, 16), dtype=mx.uint32),
        scales=mx.zeros((2, 2), dtype=mx.float16),
        biases=mx.zeros((2, 2), dtype=mx.float16),
        qkv_outputs=4,
        z_outputs=2,
        bits=4,
        group_size=64,
        cpu_weight=mx.zeros((1, 128), dtype=mx.float16),
        cpu_outputs=1,
    )
    gdn = SimpleNamespace(
        in_proj_qkv=object(),
        in_proj_z=object(),
        in_proj_b=object(),
        in_proj_a=object(),
        _omlx_ane_gdn_config=ane_patch._AneGDNConfig(
            1, 0.4, 8, True, 0.1, 6, True
        ),
        _omlx_ane_gdn_state=state,
    )
    x = mx.zeros((1, 1, 128), dtype=mx.float16)

    mixed_qkv, z, _, _ = ane_patch._gdn_backend(gdn, x)
    mx.eval(mixed_qkv, z)

    assert len(calls) == 1
    assert calls[0][-2:] == (6, True)
    assert z.tolist() == [[[1.0, 2.0]]]
    assert mixed_qkv.tolist() == [[[10.0, 20.0, 30.0, 40.0]]]


def test_gdn_backend_routes_single_ane_cpu_split(monkeypatch):
    combined = mx.array([[[1, 2, 10, 20, 30, 40]]], dtype=mx.float16)
    calls = []

    def hybrid(*args):
        calls.append(args)
        return combined

    monkeypatch.setattr(fast, "qwen35_ane_cpu_fp16_affine_qmm_t", hybrid)
    import omlx.patches.qwen35_q4_mlp as q4_patch

    monkeypatch.setattr(
        q4_patch,
        "_linear_qmm",
        lambda linear, x, variant: mx.zeros((*x.shape[:-1], 1), dtype=x.dtype),
    )
    state = ane_patch._CombinedGDNState(
        model=object(),
        weight=mx.zeros((2, 16), dtype=mx.uint32),
        scales=mx.zeros((2, 2), dtype=mx.float16),
        biases=mx.zeros((2, 2), dtype=mx.float16),
        qkv_outputs=4,
        z_outputs=2,
        bits=4,
        group_size=64,
        cpu_weight=mx.zeros((1, 128), dtype=mx.float16),
        cpu_outputs=1,
    )
    gdn = SimpleNamespace(
        in_proj_qkv=object(),
        in_proj_z=object(),
        in_proj_b=object(),
        in_proj_a=object(),
        _omlx_ane_gdn_config=ane_patch._AneGDNConfig(
            1, 0.4, 8, False, 0.1, 6, True
        ),
        _omlx_ane_gdn_state=state,
    )
    x = mx.zeros((1, 1, 128), dtype=mx.float16)

    mixed_qkv, z, _, _ = ane_patch._gdn_backend(gdn, x)
    mx.eval(mixed_qkv, z)

    assert len(calls) == 1
    assert state.model in calls[0]
    assert calls[0][-2:] == (6, True)
    assert z.tolist() == [[[1.0, 2.0]]]
    assert mixed_qkv.tolist() == [[[10.0, 20.0, 30.0, 40.0]]]


def test_gdn_backend_restores_projection_order_and_keeps_b_a_exact(monkeypatch):
    combined = mx.array([[[1, 2, 10, 20, 30, 40]]], dtype=mx.bfloat16)
    captured = []
    monkeypatch.setattr(fast, "qwen35_ane_affine_qmm_t", lambda *args: combined)

    import omlx.patches.qwen35_q4_mlp as q4_patch

    def exact(linear, x, variant):
        captured.append((linear, variant))
        return mx.full((*x.shape[:-1], 1), len(captured), dtype=x.dtype)

    monkeypatch.setattr(q4_patch, "_linear_qmm", exact)
    b_proj, a_proj = object(), object()
    state = ane_patch._CombinedGDNState(
        model=object(),
        weight=mx.zeros((4, 10), dtype=mx.uint32),
        scales=mx.zeros((4, 1), dtype=mx.bfloat16),
        biases=mx.zeros((4, 1), dtype=mx.bfloat16),
        qkv_outputs=4,
        z_outputs=2,
        bits=5,
        group_size=64,
    )
    gdn = SimpleNamespace(
        in_proj_qkv=object(),
        in_proj_z=object(),
        in_proj_b=b_proj,
        in_proj_a=a_proj,
        _omlx_ane_gdn_config=ane_patch._AneGDNConfig(1, 0.4, 8),
        _omlx_ane_gdn_state=state,
    )
    x = mx.zeros((1, 1, 64), dtype=mx.bfloat16)

    mixed_qkv, z, b, a = ane_patch._gdn_backend(gdn, x)
    mx.eval(mixed_qkv, z, b, a)

    assert z.tolist() == [[[1, 2]]]
    assert mixed_qkv.tolist() == [[[10, 20, 30, 40]]]
    assert captured == [(b_proj, 8), (a_proj, 8)]
    assert b.tolist() == [[[1]]]
    assert a.tolist() == [[[2]]]


def test_backend_reassembles_combined_gate_and_up_outputs(monkeypatch):
    combined = mx.array(
        [
            [
                [1.0, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0],
            ]
        ],
        dtype=mx.bfloat16,
    )
    captured = {}

    def hybrid(*args, **kwargs):
        return combined

    def capture_swiglu(gate, up):
        captured["gate"] = gate
        captured["up"] = up
        return gate

    monkeypatch.setattr(fast, "qwen35_ane_q4_affine_qmm_t", hybrid)
    monkeypatch.setattr(fast, "has_symbol", lambda name: False)
    monkeypatch.setattr(ane_patch, "swiglu", capture_swiglu)

    import omlx.patches.qwen35_q4_mlp as q4_patch

    monkeypatch.setattr(q4_patch, "_linear_qmm", lambda linear, x, variant: x)
    state = ane_patch._CombinedMLPState(
        model=object(),
        weight=mx.zeros((4, 1), dtype=mx.uint32),
        scales=mx.zeros((4, 1), dtype=mx.bfloat16),
        biases=mx.zeros((4, 1), dtype=mx.bfloat16),
        ane_outputs=2,
        gpu_outputs=2,
    )
    mlp = SimpleNamespace(
        down_proj=object(),
        _omlx_ane_prefill_config=ane_patch._AnePrefillConfig(1, 0.5, 8),
        _omlx_ane_prefill_state=state,
    )

    result = ane_patch._backend(mlp, mx.zeros((1, 1, 8), dtype=mx.bfloat16))
    mx.eval(result, captured["gate"], captured["up"])

    assert captured["gate"].tolist() == [[[1.0, 2.0, 3.0, 4.0]]]
    assert captured["up"].tolist() == [[[10.0, 20.0, 30.0, 40.0]]]
    assert result.tolist() == captured["gate"].tolist()


def test_backend_uses_fused_merge_swiglu_when_available(monkeypatch):
    activation = mx.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=mx.bfloat16)
    captured = {}

    monkeypatch.setattr(
        fast,
        "has_symbol",
        lambda name: name == "qwen35_ane_q4_swiglu_t",
    )
    monkeypatch.setattr(
        fast,
        "qwen35_ane_q4_affine_qmm_t",
        lambda *args: pytest.fail("raw merge path should not run"),
    )

    def fused(*args):
        captured["fused_args"] = args
        return activation

    monkeypatch.setattr(fast, "qwen35_ane_q4_swiglu_t", fused)

    import omlx.patches.qwen35_q4_mlp as q4_patch

    def down(linear, value, variant):
        captured["down"] = (linear, value, variant)
        return value

    monkeypatch.setattr(q4_patch, "_linear_qmm", down)
    state = ane_patch._CombinedMLPState(
        model=object(),
        weight=mx.zeros((4, 1), dtype=mx.uint32),
        scales=mx.zeros((4, 1), dtype=mx.bfloat16),
        biases=mx.zeros((4, 1), dtype=mx.bfloat16),
        ane_outputs=2,
        gpu_outputs=2,
    )
    down_proj = object()
    mlp = SimpleNamespace(
        down_proj=down_proj,
        _omlx_ane_prefill_config=ane_patch._AnePrefillConfig(1, 0.5, 8),
        _omlx_ane_prefill_state=state,
    )
    x = mx.zeros((1, 1, 8), dtype=mx.bfloat16)

    result = ane_patch._backend(mlp, x)
    mx.eval(result)

    assert captured["fused_args"] == (
        x,
        state.weight,
        state.scales,
        state.biases,
        state.model,
        8,
        128,
    )
    assert captured["down"] == (down_proj, activation, 8)
    assert result.tolist() == activation.tolist()


def test_backend_uses_both_ane_models_for_one_prompt(monkeypatch):
    activation = mx.zeros((1, 1, 4), dtype=mx.bfloat16)
    captured = {}

    def dual(*args):
        captured["dual_args"] = args
        return activation

    monkeypatch.setattr(fast, "qwen35_ane_dual_q4_swiglu_t", dual)

    import omlx.patches.qwen35_q4_mlp as q4_patch

    monkeypatch.setattr(q4_patch, "_linear_qmm", lambda linear, value, variant: value)
    model0, model1 = object(), object()
    state = ane_patch._CombinedMLPState(
        model=model0,
        model1=model1,
        weight=mx.zeros((4, 1), dtype=mx.uint32),
        scales=mx.zeros((4, 1), dtype=mx.bfloat16),
        biases=mx.zeros((4, 1), dtype=mx.bfloat16),
        ane_outputs=2,
        gpu_outputs=2,
    )
    mlp = SimpleNamespace(
        down_proj=object(),
        _omlx_ane_prefill_config=ane_patch._AnePrefillConfig(1, 0.5, 8, dual_ane=True),
        _omlx_ane_prefill_state=state,
    )
    x = mx.zeros((1, 1, 8), dtype=mx.bfloat16)

    result = ane_patch._backend(mlp, x)
    mx.eval(result)

    assert captured["dual_args"] == (
        x,
        state.weight,
        state.scales,
        state.biases,
        model0,
        model1,
        8,
        128,
    )


def test_fused_down_backend_runs_compatible_cpu_hidden_branch(monkeypatch):
    output = mx.zeros((1, 1, 8), dtype=mx.float16)
    captured = {}

    def fused(*args):
        captured["args"] = args
        return output

    monkeypatch.setattr(
        fast, "qwen35_ane_dual_cpu_fp16_q4_swiglu_down_t", fused
    )
    model0, model1 = object(), object()
    state = ane_patch._FusedDownMLPState(
        model=model0,
        model1=model1,
        gate_up_weight=mx.zeros((4, 1), dtype=mx.uint32),
        gate_up_scales=mx.zeros((4, 1), dtype=mx.float16),
        gate_up_biases=mx.zeros((4, 1), dtype=mx.float16),
        down_weight=mx.zeros((8, 1), dtype=mx.uint32),
        down_scales=mx.zeros((8, 1), dtype=mx.float16),
        down_biases=mx.zeros((8, 1), dtype=mx.float16),
        cpu_gate_up_weight=mx.zeros((4, 8), dtype=mx.float16),
        cpu_down_weight=mx.zeros((8, 2), dtype=mx.float16),
    )
    config = ane_patch._AnePrefillConfig(
        1,
        0.5,
        8,
        dual_ane=True,
        cpu_fraction=0.14,
        cpu_threads=12,
        cpu_shared_resource=True,
        ane_down_fraction=0.19,
        fused_down=True,
    )
    mlp = SimpleNamespace(
        _omlx_ane_prefill_config=config,
        _omlx_ane_fused_down_state=state,
    )
    x = mx.zeros((1, 1, 8), dtype=mx.float16)

    result = ane_patch._backend(mlp, x)
    mx.eval(result)

    assert result is output
    assert captured["args"] == (
        x,
        state.cpu_gate_up_weight,
        state.cpu_down_weight,
        state.gate_up_weight,
        state.gate_up_scales,
        state.gate_up_biases,
        state.down_weight,
        state.down_scales,
        state.down_biases,
        model0,
        model1,
        8,
        128,
        12,
        True,
    )


@pytest.mark.parametrize(
    ("bits", "expected_kernel"),
    [(4, "q4"), (5, "generic"), (6, "generic"), (8, "generic")],
)
def test_cpu_gate_uses_bit_appropriate_fused_swiglu(
    monkeypatch, bits, expected_kernel
):
    activation = mx.zeros((1, 1, 4), dtype=mx.float16)
    captured = {}

    def generic_fused(*args):
        captured["kernel"] = "generic"
        captured["args"] = args
        return activation

    def q4_fused(*args):
        captured["kernel"] = "q4"
        captured["args"] = args
        return activation

    monkeypatch.setattr(
        fast, "qwen35_ane_dual_cpu_fp16_q4_swiglu_t", q4_fused
    )
    monkeypatch.setattr(
        fast, "qwen35_ane_dual_cpu_fp16_swiglu_t", generic_fused
    )
    monkeypatch.setattr(
        ane_patch,
        "_post_ane_linear",
        lambda linear, value, *args, **kwargs: value,
    )
    model0, model1 = object(), object()
    state = ane_patch._CombinedMLPState(
        model=model0,
        model1=model1,
        weight=mx.zeros((4, 4 * bits), dtype=mx.uint32),
        scales=mx.zeros((4, 2), dtype=mx.float16),
        biases=mx.zeros((4, 2), dtype=mx.float16),
        ane_outputs=2,
        gpu_outputs=2,
        group_size=64,
        bits=bits,
        cpu_weight=mx.zeros((4, 128), dtype=mx.float16),
        cpu_outputs=2,
    )
    config = ane_patch._AnePrefillConfig(
        1,
        0.5,
        8,
        dual_ane=True,
        cpu_fraction=0.25,
        cpu_threads=12,
        cpu_shared_resource=True,
    )
    mlp = SimpleNamespace(
        down_proj=object(),
        _omlx_ane_prefill_config=config,
        _omlx_ane_prefill_state=state,
    )
    x = mx.zeros((1, 1, 128), dtype=mx.float16)

    result = ane_patch._backend(mlp, x)
    mx.eval(result)

    expected_args = (
        x,
        state.cpu_weight,
        state.weight,
        state.scales,
        state.biases,
        model0,
        model1,
    )
    if bits != 4:
        expected_args += (bits,)
    expected_args += (
        8,
        64,
        12,
        True,
    )
    assert captured == {"kernel": expected_kernel, "args": expected_args}


@pytest.mark.parametrize(
    ("bits", "expected_kernel"),
    [(4, "q4"), (5, "generic"), (6, "generic"), (8, "generic")],
)
def test_single_ane_cpu_gate_uses_bit_appropriate_fused_swiglu(
    monkeypatch, bits, expected_kernel
):
    activation = mx.zeros((1, 1, 4), dtype=mx.float16)
    captured = {}

    def generic_fused(*args):
        captured["kernel"] = "generic"
        captured["args"] = args
        return activation

    def q4_fused(*args):
        captured["kernel"] = "q4"
        captured["args"] = args
        return activation

    monkeypatch.setattr(fast, "qwen35_ane_cpu_fp16_q4_swiglu_t", q4_fused)
    monkeypatch.setattr(fast, "qwen35_ane_cpu_fp16_swiglu_t", generic_fused)
    monkeypatch.setattr(
        ane_patch,
        "_post_ane_linear",
        lambda linear, value, *args, **kwargs: value,
    )
    model = object()
    state = ane_patch._CombinedMLPState(
        model=model,
        weight=mx.zeros((4, 4 * bits), dtype=mx.uint32),
        scales=mx.zeros((4, 2), dtype=mx.float16),
        biases=mx.zeros((4, 2), dtype=mx.float16),
        ane_outputs=2,
        gpu_outputs=2,
        group_size=64,
        bits=bits,
        cpu_weight=mx.zeros((4, 128), dtype=mx.float16),
        cpu_outputs=2,
    )
    config = ane_patch._AnePrefillConfig(
        1,
        0.5,
        8,
        dual_ane=False,
        cpu_fraction=0.25,
        cpu_threads=12,
        cpu_shared_resource=True,
    )
    mlp = SimpleNamespace(
        down_proj=object(),
        _omlx_ane_prefill_config=config,
        _omlx_ane_prefill_state=state,
    )
    x = mx.zeros((1, 1, 128), dtype=mx.float16)

    result = ane_patch._backend(mlp, x)
    mx.eval(result)

    expected_args = (
        x,
        state.cpu_weight,
        state.weight,
        state.scales,
        state.biases,
        model,
    )
    if bits != 4:
        expected_args += (bits,)
    expected_args += (8, 64, 12, True)
    assert captured == {"kernel": expected_kernel, "args": expected_args}


def test_install_dispatch_wraps_outer_q4_mlp_dispatch(monkeypatch):
    class PatchedMLP:
        _omlx_q4_mlp_patched = True

        def __call__(self, x):
            return x

    registrations = []
    gdn_registrations = []
    vlm = SimpleNamespace(
        Qwen3_5MLP=PatchedMLP,
        register_qwen3_5_mlp_prefill_backend=registrations.append,
        register_qwen3_5_gdn_prefill_backend=gdn_registrations.append,
    )

    def import_module(name):
        if name == "mlx_vlm.models.qwen3_5.language":
            return vlm
        raise ImportError(name)

    monkeypatch.setattr(ane_patch.importlib, "import_module", import_module)
    monkeypatch.setattr(ane_patch, "_PATCHED_CLASSES", set())
    monkeypatch.setattr(ane_patch, "_VLM_HOOK_INSTALLED", False)
    monkeypatch.setattr(ane_patch, "_VLM_GDN_HOOK_INSTALLED", False)

    assert ane_patch._install_dispatch()
    assert PatchedMLP in ane_patch._PATCHED_CLASSES
    assert registrations == []
    assert gdn_registrations == [ane_patch._gdn_backend]


@pytest.mark.parametrize(
    ("sequence_length", "fraction", "max_layers"),
    [(512, 0.4, 1), (2048, 0.01, 1), (2048, 0.4, 0)],
)
def test_enable_rejects_unsafe_fixed_shape_settings(
    sequence_length, fraction, max_layers
):
    with pytest.raises(ValueError):
        ane_patch.enable_qwen35_ane_prefill(
            _Model(1),
            sequence_length=sequence_length,
            fraction=fraction,
            max_layers=max_layers,
        )


def test_enable_uses_ane_on_nax_gpu_when_model_setting_enabled(monkeypatch):
    monkeypatch.delenv("OMLX_QWEN35_ANE_PREFILL", raising=False)
    monkeypatch.setattr(fast, "qwen35_ane_available", lambda: True)
    monkeypatch.setattr(fast, "has_symbol", lambda name: False)
    monkeypatch.setattr(ane_patch, "_install_dispatch", lambda: True)
    monkeypatch.setattr(ane_patch, "_eligible_pair", lambda mlp: True)
    monkeypatch.setattr(ane_patch, "_compile_pair", lambda mlp, config: object())
    model = _Model(2)

    count = ane_patch.enable_qwen35_ane_prefill(model, sequence_length=2048)

    assert count == 2


def test_enable_env_forces_ane_on_nax_gpu(monkeypatch):
    monkeypatch.setenv("OMLX_QWEN35_ANE_PREFILL", "1")
    monkeypatch.setattr(fast, "qwen35_ane_available", lambda: True)
    monkeypatch.setattr(fast, "has_symbol", lambda name: False)
    monkeypatch.setattr(ane_patch, "_install_dispatch", lambda: True)
    monkeypatch.setattr(ane_patch, "_eligible_pair", lambda mlp: True)
    monkeypatch.setattr(ane_patch, "_compile_pair", lambda mlp, config: object())
    model = _Model(2)

    count = ane_patch.enable_qwen35_ane_prefill(
        model, sequence_length=2048, max_layers=2
    )

    assert count == 2


def test_enable_env_kill_switch_wins(monkeypatch):
    monkeypatch.setenv("OMLX_QWEN35_ANE_PREFILL", "0")
    installed = []
    monkeypatch.setattr(
        ane_patch, "_install_dispatch", lambda: installed.append(True) or True
    )

    count = ane_patch.enable_qwen35_ane_prefill(_Model(1), sequence_length=2048)

    assert count == 0
    assert installed == []


def test_prefill_status_reports_configured_layers():
    model = SimpleNamespace(
        _omlx_ane_mlp_prefill_count=12,
        _omlx_ane_gdn_prefill_count=4,
        _omlx_ane_dual_prefill_count=8,
        _omlx_ane_resident_program_count=24,
    )
    assert ane_patch.qwen35_ane_prefill_status(model) == {
        "attempted": True,
        "configured": True,
        "shed": False,
        "mlp_layers": 12,
        "gdn_layers": 4,
        "dual_ane_layers": 8,
        "resident_programs": 24,
        "tail_padding_min_tokens": 0,
    }


def test_prefill_status_flags_attempted_but_empty():
    model = SimpleNamespace(
        _omlx_ane_mlp_prefill_count=0,
        _omlx_ane_gdn_prefill_count=0,
        _omlx_ane_dual_prefill_count=0,
        _omlx_ane_resident_program_count=0,
    )
    status = ane_patch.qwen35_ane_prefill_status(model)
    assert status["attempted"] is True
    assert status["configured"] is False


def test_prefill_status_safe_on_untouched_model():
    status = ane_patch.qwen35_ane_prefill_status(SimpleNamespace())
    assert status["attempted"] is False
    assert status["configured"] is False
    assert status["mlp_layers"] == 0


def test_enable_warns_when_no_eligible_layers(monkeypatch, caplog):
    monkeypatch.setattr(fast, "qwen35_ane_available", lambda: True)
    monkeypatch.setattr(ane_patch, "_install_dispatch", lambda: True)
    monkeypatch.setattr(
        ane_patch, "_enable_dual_procedure_banks", lambda *args, **kwargs: None
    )
    monkeypatch.delenv("OMLX_QWEN35_ANE_PREFILL", raising=False)

    model = SimpleNamespace(modules=lambda: [])
    with caplog.at_level(logging.WARNING, logger="omlx.patches.qwen35_ane_prefill"):
        count = ane_patch.enable_qwen35_ane_prefill(model)

    assert count == 0
    assert "no eligible MLP layers found" in caplog.text
    assert ane_patch.qwen35_ane_prefill_status(model)["attempted"] is True


def test_bank_prepare_keeps_packed_q6_suffix():
    gdn = _make_affine_gdn(6, 128)
    config = ane_patch._AneGDNConfig(2048, 0.5, 8, True)
    packed = ane_patch._pack_affine_gdn_suffix(
        gdn.in_proj_qkv, gdn.in_proj_b, gdn.in_proj_a, 0, (6, 128)
    )
    assert packed is not None
    expected_weight, expected_scales, expected_biases, b_outputs, a_outputs = packed

    prepared = ane_patch._prepare_gdn_for_bank(gdn, config)
    assert prepared is not None
    state, _dense0, dense1 = prepared
    assert (state.b_outputs, state.a_outputs) == (b_outputs, a_outputs)
    assert state.weight.shape == expected_weight.shape
    assert bool(mx.array_equal(state.weight, expected_weight))
    assert bool(mx.array_equal(state.scales, expected_scales))
    assert bool(mx.array_equal(state.biases, expected_biases))
    assert dense1 is not None


def test_enable_survives_warmup_failure(monkeypatch, caplog):
    monkeypatch.setattr(fast, "qwen35_ane_available", lambda: True)
    monkeypatch.setattr(fast, "has_symbol", lambda name: True)
    monkeypatch.setattr(
        fast, "qwen35_ane_linear_bank_builder", _no_bank_builder
    )
    monkeypatch.setattr(ane_patch, "_install_dispatch", lambda: True)
    monkeypatch.setattr(ane_patch, "_eligible_pair", lambda mlp: True)

    class _FailingModel:
        def warmup(self):
            raise RuntimeError("ANE evaluation failed")

    monkeypatch.setattr(
        fast,
        "qwen35_ane_compile_linear_bank",
        lambda weights, sequence_length, ane_instance: [
            _FailingModel() for _ in weights
        ],
    )
    model = _Model(2)
    with caplog.at_level(logging.WARNING, logger="omlx.patches.qwen35_ane_prefill"):
        count = ane_patch.enable_qwen35_ane_prefill(
            model,
            sequence_length=2048,
            fraction=0.5,
            max_layers=2,
            dual_ane=True,
        )

    assert count == 2
    assert "ANE warmup failed" in caplog.text


def test_prepare_cpu_linear_needs_the_native_symbol(monkeypatch):
    linear = nn.QuantizedLinear(128, 256, bias=False, group_size=64, bits=4)
    linear.scales = linear.scales.astype(mx.float16)
    linear.biases = linear.biases.astype(mx.float16)

    monkeypatch.setattr(fast, "has_symbol", lambda name: False)
    assert ane_patch._prepare_cpu_linear(linear, 0.25) is None

    monkeypatch.setattr(
        fast, "has_symbol", lambda name: name == "qwen35_cpu_fp16_affine_qmm_t"
    )
    assert ane_patch._prepare_cpu_linear(linear, 0.25) is not None


def test_enable_warms_the_cpu_sharing_path(monkeypatch):
    """CPU-shared modules get one dummy dispatch at load, not at first use."""
    monkeypatch.setattr(fast, "qwen35_ane_available", lambda: True)
    monkeypatch.setattr(fast, "has_symbol", lambda name: True)
    monkeypatch.setattr(ane_patch, "_install_dispatch", lambda: True)
    monkeypatch.setattr(ane_patch, "_eligible_pair", lambda mlp: True)
    monkeypatch.setattr(
        fast,
        "qwen35_ane_compile_linear_bank",
        lambda weights, sequence_length, ane_instance: [object() for _ in weights],
    )
    warmed = []
    monkeypatch.setattr(
        ane_patch, "_backend", lambda module, x: warmed.append(module) or mx.zeros(1)
    )

    model = _Model(2)
    for layer in model.layers:
        for linear in (layer.gate_proj, layer.up_proj, layer.down_proj):
            linear.scales = linear.scales.astype(mx.float16)
            linear.biases = linear.biases.astype(mx.float16)

    count = ane_patch.enable_qwen35_ane_prefill(
        model,
        sequence_length=2048,
        fraction=0.5,
        max_layers=2,
        dual_ane=True,
        cpu_fraction=0.25,
    )

    assert count == 2
    assert {id(m) for m in warmed} == {id(layer) for layer in model.layers}

    # Without a CPU share the dummy dispatch must not run.
    warmed.clear()
    plain = _Model(2)
    count = ane_patch.enable_qwen35_ane_prefill(
        plain,
        sequence_length=2048,
        fraction=0.5,
        max_layers=2,
        dual_ane=True,
    )
    assert count == 2
    assert warmed == []


# --- ANE prefill transient memory estimate (issue #2841) ---


def test_ane_prefill_transient_bytes_reads_live_model_dims():
    """The estimate sums (input + output) * seq * 2 over the compiled models."""
    seq = 2048

    def _ane_model(input_dim, output_dim):
        return SimpleNamespace(
            input_dim=input_dim, output_dim=output_dim, sequence_length=seq
        )

    mlp = SimpleNamespace(
        _omlx_ane_prefill_state=SimpleNamespace(
            model=_ane_model(5120, 4608), model1=_ane_model(5120, 4608)
        )
    )
    gdn = SimpleNamespace(
        _omlx_ane_gdn_state=SimpleNamespace(
            model=_ane_model(5120, 768), model1=None
        )
    )
    model = SimpleNamespace(modules=lambda: [mlp, gdn, SimpleNamespace()])

    expected = 2 * (5120 + 4608) * seq * 2 + (5120 + 768) * seq * 2
    assert ane_patch.ane_prefill_transient_bytes(model) == expected


def test_ane_prefill_transient_bytes_zero_without_ane():
    """A model with no compiled ANE slice reserves nothing."""
    assert ane_patch.ane_prefill_transient_bytes(SimpleNamespace()) == 0
    plain = SimpleNamespace(modules=lambda: [SimpleNamespace()])
    assert ane_patch.ane_prefill_transient_bytes(plain) == 0


def test_ane_prefill_transient_bytes_prices_fused_and_down_states():
    """Fused SwiGLU/down banks and ANE down slices hold IOSurfaces too."""
    seq = 2048

    def _ane_model(input_dim, output_dim):
        return SimpleNamespace(
            input_dim=input_dim, output_dim=output_dim, sequence_length=seq
        )

    fused = SimpleNamespace(
        _omlx_ane_fused_down_state=SimpleNamespace(
            model=_ane_model(5120, 5120), model1=_ane_model(5120, 5120)
        )
    )
    down = SimpleNamespace(
        _omlx_ane_prefill_state=SimpleNamespace(
            model=_ane_model(5120, 4608),
            model1=None,
            down_ane=SimpleNamespace(
                model=_ane_model(4608, 5120), model1=_ane_model(4608, 5120)
            ),
        )
    )
    model = SimpleNamespace(modules=lambda: [fused, down])

    expected = (
        2 * (5120 + 5120) * seq * 2
        + (5120 + 4608) * seq * 2
        + 2 * (4608 + 5120) * seq * 2
    )
    assert ane_patch.ane_prefill_transient_bytes(model) == expected


# --- incremental bank builder staging (issue #2781) ---


class _RecorderBankBuilder:
    """Stands in for the native AneLinearBankBuilder."""

    def __init__(self, fail_full_span_once=False):
        self.added = []
        self.compiled_spans = []
        self._fail_full_span_once = fail_full_span_once

    def add(self, weight):
        self.added.append(weight)

    @property
    def size(self):
        return len(self.added)

    def compile(self, ane_instance, start, stop):
        if self._fail_full_span_once and stop - start == len(self.added):
            self._fail_full_span_once = False
            raise RuntimeError("bank load failed (0x20004)")
        self.compiled_spans.append((ane_instance, start, stop))
        return [object() for _ in range(stop - start)]


def _builder_test_setup(monkeypatch, builders):
    monkeypatch.setattr(fast, "qwen35_ane_available", lambda: True)
    monkeypatch.setattr(fast, "has_symbol", lambda name: True)
    monkeypatch.setattr(ane_patch, "_install_dispatch", lambda: True)
    monkeypatch.setattr(ane_patch, "_eligible_pair", lambda mlp: True)
    monkeypatch.setattr(ane_patch, "_ane_bank_memory_headroom_ok", lambda: True)
    monkeypatch.setattr(ane_patch.time, "sleep", lambda *_a: None)
    monkeypatch.setattr(
        fast, "qwen35_ane_linear_bank_builder", lambda seq: builders.pop(0)
    )
    monkeypatch.setattr(
        fast,
        "qwen35_ane_compile_linear_bank",
        lambda *a, **k: pytest.fail(
            "the builder path must not stage all weights at once"
        ),
    )


def test_enable_streams_layers_through_the_bank_builder(monkeypatch):
    """Each layer is handed to the builder as prepared, not held until compile."""
    builder0 = _RecorderBankBuilder()
    builder1 = _RecorderBankBuilder()
    _builder_test_setup(monkeypatch, [builder0, builder1])

    model = _Model(4)
    count = ane_patch.enable_qwen35_ane_prefill(
        model,
        sequence_length=2048,
        fraction=0.5,
        max_layers=4,
        dual_ane=True,
    )

    assert count == 4
    assert len(builder0.added) == 4
    assert len(builder1.added) == 4
    assert builder0.compiled_spans == [(1, 0, 4)]
    assert builder1.compiled_spans == [(2, 0, 4)]
    assert model._omlx_ane_resident_program_count == 2
    states = [layer._omlx_ane_prefill_state for layer in model.layers]
    assert all(s.model is not None and s.model1 is not None for s in states)


def test_builder_split_ladder_retries_without_restaging(monkeypatch):
    """A monolithic load failure retries in spans from the stored chunks."""
    builder0 = _RecorderBankBuilder(fail_full_span_once=True)
    builder1 = _RecorderBankBuilder()
    _builder_test_setup(monkeypatch, [builder0, builder1])

    model = _Model(4)
    count = ane_patch.enable_qwen35_ane_prefill(
        model,
        sequence_length=2048,
        fraction=0.5,
        max_layers=4,
        dual_ane=True,
    )

    assert count == 4
    # staged exactly once despite the retry
    assert len(builder0.added) == 4
    # first attempt failed on the monolithic span, retry split into two banks
    assert builder0.compiled_spans[0][1:] != (0, 4) or len(
        builder0.compiled_spans
    ) > 1
    assert model._omlx_ane_resident_program_count == 4


def test_warmup_failure_latches_only_the_owning_module(monkeypatch, caplog):
    """One broken procedure disables its module; the rest keep warming (#2940)."""
    monkeypatch.setattr(fast, "qwen35_ane_available", lambda: True)
    monkeypatch.setattr(fast, "has_symbol", lambda name: True)
    monkeypatch.setattr(
        fast, "qwen35_ane_linear_bank_builder", _no_bank_builder
    )
    monkeypatch.setattr(ane_patch, "_install_dispatch", lambda: True)
    monkeypatch.setattr(ane_patch, "_eligible_pair", lambda mlp: True)

    warm_calls = []

    class _WarmModel:
        def __init__(self, index):
            self.index = index

        def warmup(self):
            if self.index == 1:
                raise RuntimeError("ANE evaluation failed")
            warm_calls.append(self.index)

    def compile_bank(weights, sequence_length, ane_instance):
        return [_WarmModel(i) for i in range(len(weights))]

    monkeypatch.setattr(fast, "qwen35_ane_compile_linear_bank", compile_bank)
    model = _Model(3)

    with caplog.at_level(logging.WARNING, logger="omlx.patches.qwen35_ane_prefill"):
        count = ane_patch.enable_qwen35_ane_prefill(
            model,
            sequence_length=2048,
            fraction=0.5,
            max_layers=3,
            dual_ane=True,
        )

    assert count == 3
    assert getattr(model.layers[1], "_omlx_ane_prefill_failed", False)
    assert not getattr(model.layers[0], "_omlx_ane_prefill_failed", False)
    assert not getattr(model.layers[2], "_omlx_ane_prefill_failed", False)
    assert sorted(warm_calls) == [0, 0, 2, 2]
    assert "disabling ANE" in caplog.text


# --- fused bank streaming + CPU warm helper (post-#2935 follow-up) ---


def _dual_ane_down_mlp():
    return SimpleNamespace(
        gate_proj=nn.QuantizedLinear(128, 1024, bias=False, group_size=128, bits=4),
        up_proj=nn.QuantizedLinear(128, 1024, bias=False, group_size=128, bits=4),
        down_proj=nn.QuantizedLinear(1024, 128, bias=False, group_size=128, bits=4),
    )


def _dual_ane_down_config():
    # hidden=1024 (gate/up out_features); per_ane=128, total_ane=256,
    # cpu_hidden=128, gpu_start=384 -- all multiples of 128, and
    # hidden - gpu_start = 640 is also a multiple of 128, so
    # _prepare_fused_down_for_bank's validity gate passes.
    return ane_patch._AnePrefillConfig(
        1,
        0.5,
        8,
        dual_ane=True,
        cpu_fraction=0.125,
        ane_down_fraction=0.125,
        fused_down=True,
    )


def test_prepare_fused_down_for_bank_only_dequantizes_consumed_down_columns(
    monkeypatch,
):
    """C3: dense_down used to dequantize down_proj's ENTIRE packed weight,
    even though only columns [0:gpu_start] are ever consumed below
    (down0/down1/cpu_down_weight) -- the [gpu_start:hidden] suffix was
    computed and immediately discarded, a pure-waste ~0.5GB/layer fp32
    transient. The down dequantize call must now receive only the
    gpu_start-wide packed slice, not the full matrix.
    See docs/qwen35-hardening-and-optimization.md C3."""
    mlp = _dual_ane_down_mlp()
    config = _dual_ane_down_config()
    real_dequantize = mx.dequantize
    captured_shapes = []

    def spy_dequantize(w, *args, **kwargs):
        captured_shapes.append(tuple(w.shape))
        return real_dequantize(w, *args, **kwargs)

    monkeypatch.setattr(ane_patch.mx, "dequantize", spy_dequantize)

    result = ane_patch._prepare_fused_down_for_bank(mlp, config)
    assert result is not None

    # down's dequantize call is issued first, before any gate/up dense_rows
    # calls. It must only see the gpu_start-wide packed slice (384 // 8 =
    # 48 columns), not the full 1024 // 8 = 128 columns of down_proj's
    # packed weight.
    full_packed_width = mlp.down_proj.weight.shape[1]
    assert full_packed_width == 128
    assert captured_shapes[0][1] == 48
    assert captured_shapes[0][1] < full_packed_width


def test_prepare_fused_down_for_bank_sliced_dequant_matches_full_dequant():
    """Correctness check for the C3 optimization: dequantizing only the
    consumed prefix of the down matrix must be bit-identical to the old
    dequantize-the-whole-thing-then-slice behavior, not merely faster."""
    mlp = _dual_ane_down_mlp()
    config = _dual_ane_down_config()

    result = ane_patch._prepare_fused_down_for_bank(mlp, config)
    assert result is not None
    state, (gate0, up0, down0, gate1, up1, down1) = result

    down = mlp.down_proj
    reference = mx.dequantize(
        down.weight, down.scales, down.biases, group_size=128, bits=4
    ).astype(mx.float32)
    per_ane, total_ane, gpu_start = 128, 256, 384
    assert bool(mx.array_equal(down0, mx.contiguous(reference[:, :per_ane])))
    assert bool(
        mx.array_equal(down1, mx.contiguous(reference[:, per_ane:total_ane]))
    )
    assert bool(
        mx.array_equal(
            state.cpu_down_weight,
            mx.contiguous(reference[:, total_ane:gpu_start]).astype(mx.float16),
        )
    )


class _RecorderFusedBankBuilder:
    """Stands in for the native AneFusedBankBuilder."""

    def __init__(self):
        self.added = []
        self.compiled_spans = []

    def add(self, gate, up, down):
        self.added.append((gate, up, down))

    @property
    def size(self):
        return len(self.added)

    def compile(self, ane_instance, start, stop):
        self.compiled_spans.append((ane_instance, start, stop))
        return [object() for _ in range(stop - start)]


def _fused_bank_state():
    return ane_patch._FusedDownMLPState(
        model=None,
        model1=None,
        gate_up_weight=mx.zeros((4, 1), dtype=mx.uint32),
        gate_up_scales=mx.zeros((4, 1), dtype=mx.float16),
        gate_up_biases=mx.zeros((4, 1), dtype=mx.float16),
        down_weight=mx.zeros((8, 1), dtype=mx.uint32),
        down_scales=mx.zeros((8, 1), dtype=mx.float16),
        down_biases=mx.zeros((8, 1), dtype=mx.float16),
    )


def test_fused_bank_staging_streams_through_builder(monkeypatch):
    """Fused enable stages one gate/up/down triple at a time (issue #2781)."""
    builders = []

    def _make_builder(sequence_length):
        assert sequence_length == 2048
        builder = _RecorderFusedBankBuilder()
        builders.append(builder)
        return builder

    monkeypatch.setattr(
        fast, "qwen35_ane_fused_bank_builder", _make_builder, raising=False
    )
    monkeypatch.setattr(
        ane_patch,
        "_prepare_fused_down_for_bank",
        lambda module, config: (
            _fused_bank_state(),
            tuple(mx.zeros((2, 2)) for _ in range(6)),
        ),
    )
    monkeypatch.setattr(ane_patch, "_warm_ane_models", lambda models: None)

    config = ane_patch._AnePrefillConfig(
        2048, 0.3, 8, dual_ane=True, fused_down=True
    )
    model = SimpleNamespace()
    modules = [SimpleNamespace() for _ in range(3)]

    result = ane_patch._enable_fused_down_banks(model, modules, config)

    assert result == (3, 2)
    assert len(builders) == 2
    assert [len(builder.added) for builder in builders] == [3, 3]
    assert builders[0].compiled_spans == [(1, 0, 3)]
    assert builders[1].compiled_spans == [(2, 0, 3)]
    for module in modules:
        assert module._omlx_ane_fused_down_state.model is not None
        assert module._omlx_ane_fused_down_state.model1 is not None
    assert model._omlx_ane_down_prefill_count == 3


def test_fused_bank_staging_falls_back_without_builder(monkeypatch):
    """Old extensions without the fused builder keep the one-shot path."""
    monkeypatch.delattr(fast, "qwen35_ane_fused_bank_builder", raising=False)
    compiled = []

    def _compile_bank(gates, ups, downs, sequence_length, ane_instance):
        compiled.append((len(gates), sequence_length, ane_instance))
        return [object() for _ in gates]

    monkeypatch.setattr(
        fast, "qwen35_ane_compile_swiglu_down_bank", _compile_bank
    )
    monkeypatch.setattr(
        ane_patch,
        "_prepare_fused_down_for_bank",
        lambda module, config: (
            _fused_bank_state(),
            tuple(mx.zeros((2, 2)) for _ in range(6)),
        ),
    )
    monkeypatch.setattr(ane_patch, "_warm_ane_models", lambda models: None)

    config = ane_patch._AnePrefillConfig(
        2048, 0.3, 8, dual_ane=True, fused_down=True
    )
    model = SimpleNamespace()
    modules = [SimpleNamespace() for _ in range(2)]

    result = ane_patch._enable_fused_down_banks(model, modules, config)

    assert result == (2, 2)
    assert compiled == [(2, 2048, 1), (2, 2048, 2)]
    for module in modules:
        assert module._omlx_ane_fused_down_state.model is not None


def test_warm_cpu_sharing_path_dispatches_mlp_and_gdn(monkeypatch):
    """The shared helper warms MLP and GDN modules with exact-shape zeros."""
    calls = []

    def _mlp_backend(module, x):
        calls.append(("mlp", x.shape))
        return mx.zeros((1,))

    def _gdn_backend(module, x):
        calls.append(("gdn", x.shape))
        return (mx.zeros((1,)),)

    monkeypatch.setattr(ane_patch, "_backend", _mlp_backend)
    monkeypatch.setattr(ane_patch, "_gdn_backend", _gdn_backend)

    linear = SimpleNamespace(
        weight=mx.zeros((4, 4), dtype=mx.uint32),
        bits=4,
        scales=mx.zeros((1,), dtype=mx.float16),
    )
    mlp = SimpleNamespace(gate_proj=linear)
    gdn = SimpleNamespace(in_proj_qkv=linear)

    ane_patch._warm_cpu_sharing_path(64, [mlp], [gdn])

    assert calls == [("mlp", (1, 64, 32)), ("gdn", (1, 64, 32))]

# --- opt-in reuse of Apple ANE compiled programs ---


_ANE_MM_PATH = (
    Path(__file__).resolve().parents[1]
    / "omlx/custom_kernels/qwen35_prefill/csrc/qwen35_ane.mm"
)


@pytest.fixture(scope="module")
def ane_mm() -> str:
    return _ANE_MM_PATH.read_text(encoding="utf-8")


def test_compile_cache_native_gate_is_exact_opt_in(ane_mm):
    gate = re.search(r"bool ane_compile_cache_enabled\(\) \{.*?\n\}", ane_mm, re.S)
    assert gate, "ane_compile_cache_enabled() is absent from qwen35_ane.mm"
    assert 'getenv("OMLX_QWEN35_ANE_COMPILE_CACHE")' in gate.group()
    assert 'strcmp(value, "1") == 0' in gate.group()


def test_compile_cache_covers_all_four_native_compile_sites(ane_mm):
    """Individual linear, single fused SwiGLU/down, linear banks, and fused
    banks use one content-hash cache/fallback implementation."""
    assert ane_mm.count("load_or_compile_ane_model(") == 5
    assert ane_mm.count("model, identifier, ane_instance") == 4
    assert ane_mm.count("@selector(compileWithQoS:options:error:)") == 1


def test_compile_cache_cleanup_keeps_the_entry_lock_file_stable(ane_mm):
    """Never unlink a lock path while waiters may hold its old inode open.

    Recreating the pathname would establish a second lock domain and allow
    two processes to mutate the same staging directory concurrently.
    """
    assert "NSString *ane_compile_cache_lock_path(NSString *entry_directory)" in ane_mm
    assert ane_mm.count("ane_compile_cache_lock_path(") == 2
    assert "unlink(ane_compile_cache_lock_path" not in ane_mm
    assert "Never unlink this path" in ane_mm


def test_compile_cache_keeps_historical_delete_on_unload(ane_mm):
    """Apple owns the compiled AOT cache; oMLX staging files remain temporary."""
    assert "persistent_" not in ane_mm
    assert ane_mm.count(
        "remove_ane_staging_directory(directory_, cache_lock_entry_)"
    ) == 2
    assert "ScopedAneCacheLock cache_lock(cache_lock_entry);" in ane_mm
    assert "ANE compile cache cleanup deferred" in ane_mm
    assert "NSTemporaryDirectory()" in ane_mm


def test_compile_cache_hit_restores_without_recompiling(ane_mm):
    """The framework-derived URL and descriptor hash must remain authoritative.

    macOS 27 verifies the in-memory model's per-file hashes and rejects the
    caller-assigned staging URL that rc3 introduced in both cache modes.
    """
    assert "setModelURL:" not in ane_mm
    assert "fileURLWithPath:" not in ane_mm
    assert "@selector(compiledModelExists)" in ane_mm
    assert re.search(r"if \(!restored\) \{\s*compile_fresh\(\);\s*\}", ane_mm)
    assert re.search(
        r"return \{\s*staged \? directory : nil,\s*"
        r"staged \? cache_lock_entry : nil,\s*\};",
        ane_mm,
    )


def test_compile_cache_hit_load_failure_invalidates_then_compiles_once(ane_mm):
    assert "ANE compile cache fallback" in ane_mm
    assert "@selector(purgeCompiledModel)" in ane_mm
    assert ane_mm.count("compile_fresh();") == 2
    assert ane_mm.count("@selector(loadWithQoS:options:error:)") == 2
    fallback = ane_mm.index('NSLog(@"oMLX: ANE compile cache fallback')
    purge = ane_mm.index("@selector(purgeCompiledModel)", fallback)
    recompile = ane_mm.index("compile_fresh();", purge)
    assert fallback < purge < recompile


def test_compile_cache_lock_key_matches_native_content_hash(ane_mm):
    assert "NSCachesDirectory" in ane_mm
    assert 'stringByAppendingPathComponent:@"v1"' in ane_mm
    assert "operatingSystemVersionString" in ane_mm
    assert "ane_compile_cache_lock_entry(NSString *identifier)" in ane_mm
    assert '.i%d"' not in ane_mm
    assert "stringByAppendingPathComponent:identifier" in ane_mm


def test_compile_cache_serializes_cross_process_writers(ane_mm):
    assert "#include <sys/file.h>" in ane_mm
    assert re.search(r"flock\(.*LOCK_EX", ane_mm)
    assert "Hold the descriptor-hash lock from probe through load" in ane_mm


def test_compile_cache_fails_open_when_root_or_lock_is_unavailable(ane_mm):
    assert "ANE compile cache unavailable" in ane_mm
    assert "temporary" in ane_mm


def test_compile_cache_telemetry_uses_native_log_prefix(ane_mm):
    for event in ("hit", "miss", "fallback"):
        assert f'@"oMLX: ANE compile cache {event}' in ane_mm


def test_compile_cache_lock_acquisition_is_bounded(ane_mm):
    """A suspended holder must not hang later loads: non-blocking flock with a
    deadline, failing open to the temp path on timeout."""
    assert "LOCK_EX | LOCK_NB" in ane_mm
    assert "kAneCacheLockTimeout" in ane_mm
    assert "ANE compile cache lock acquisition timed out" in ane_mm

class _ReleasableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Linear(4, 4)
        self.gdn = nn.Linear(4, 4)


class _NativeHandle:
    """Weakref-able stand-in for a compiled AneLinearModel."""


def test_release_latches_modules_drops_states_and_zeroes_counters():
    import gc
    model = _ReleasableModel()
    handle = _NativeHandle()
    ref = weakref.ref(handle)
    model.mlp._omlx_ane_prefill_state = SimpleNamespace(model=handle)
    model.gdn._omlx_ane_gdn_state = SimpleNamespace(model=_NativeHandle())
    model._omlx_ane_mlp_prefill_count = 1
    model._omlx_ane_gdn_prefill_count = 1
    model._omlx_ane_dual_prefill_count = 1
    model._omlx_ane_resident_program_count = 2

    released, programs = ane_patch.release_qwen35_ane_prefill(model)
    del handle
    gc.collect()

    assert (released, programs) == (2, 2)
    assert model.mlp._omlx_ane_prefill_state is None
    assert model.gdn._omlx_ane_gdn_state is None
    # The latch is what stops the dispatch sites from lazily recompiling a
    # missing state -- without it the release would be a slow no-op.
    assert model.mlp._omlx_ane_prefill_failed is True
    assert model.gdn._omlx_ane_gdn_failed is True
    # The dropped state held the last reference: the native handle dies with
    # it, which is what actually returns the mapped bank memory.
    assert ref() is None
    status = ane_patch.qwen35_ane_prefill_status(model)
    assert status["attempted"] is True
    assert status["configured"] is False
    assert status["shed"] is True
    assert status["resident_programs"] == 0


def test_release_is_idempotent_and_noop_without_slices():
    model = _ReleasableModel()
    assert ane_patch.release_qwen35_ane_prefill(model) == (0, 0)
    model.mlp._omlx_ane_prefill_state = SimpleNamespace(model=_NativeHandle())
    ane_patch.release_qwen35_ane_prefill(model)
    assert ane_patch.release_qwen35_ane_prefill(model) == (0, 0)


# --- dashboard toggle for the compile cache ---


def test_compile_cache_setting_round_trips_and_defaults_off():
    """The advanced-settings toggle must persist in settings.json and stay
    off for installs that predate it."""
    from omlx.settings import CacheSettings

    cache = CacheSettings(ane_compile_cache=True)
    assert CacheSettings.from_dict(cache.to_dict()).ane_compile_cache is True

    assert CacheSettings.from_dict({}).ane_compile_cache is False
    assert CacheSettings().ane_compile_cache is False


def test_compile_cache_setting_exports_the_native_env_gate(monkeypatch):
    """serve exports the env var the native gate reads once at the first
    compile; an explicit env override wins."""
    import os

    source = (
        Path(__file__).resolve().parents[1] / "omlx/cli.py"
    ).read_text(encoding="utf-8")
    assert 'os.environ.setdefault("OMLX_QWEN35_ANE_COMPILE_CACHE", "1")' in source

    monkeypatch.delenv("OMLX_QWEN35_ANE_COMPILE_CACHE", raising=False)
    os.environ.setdefault("OMLX_QWEN35_ANE_COMPILE_CACHE", "1")
    assert os.environ["OMLX_QWEN35_ANE_COMPILE_CACHE"] == "1"

    monkeypatch.setenv("OMLX_QWEN35_ANE_COMPILE_CACHE", "0")
    os.environ.setdefault("OMLX_QWEN35_ANE_COMPILE_CACHE", "1")
    assert os.environ["OMLX_QWEN35_ANE_COMPILE_CACHE"] == "0"


# --- below-floor GDN fraction is explained, not silent (#2899, #2905) ---


def _floor_gdn(z_outputs: int, qkv_outputs: int):
    return SimpleNamespace(
        in_proj_z=SimpleNamespace(weight=SimpleNamespace(shape=(z_outputs, 1))),
        in_proj_qkv=SimpleNamespace(weight=SimpleNamespace(shape=(qkv_outputs, 1))),
    )


def test_recurrent_safe_gdn_slice_caps_ane_at_z():
    """A wider requested slice must not move recurrent qkv rows onto ANE."""
    assert (
        ane_patch._recurrent_safe_gdn_ane_outputs(6144, 10240, 0.50, 128)
        == 6144
    )
    assert (
        ane_patch._recurrent_safe_gdn_ane_outputs(6144, 10240, 0.375, 128)
        == 6144
    )
    assert ane_patch._recurrent_safe_gdn_ane_outputs(6144, 10240, 0.35, 128) == 0


def test_recurrent_safe_gdn_slice_rejects_unaligned_z():
    assert ane_patch._recurrent_safe_gdn_ane_outputs(320, 1728, 0.50, 128) == 0
    assert ane_patch._recurrent_safe_gdn_ane_outputs(320, 1728, 0.50, 64) == 320


def test_recurrent_safe_gdn_cap_is_reported(caplog):
    gdn = _floor_gdn(512, 1536)
    gdn._omlx_ane_gdn_state = object()
    model = SimpleNamespace(modules=lambda: [gdn])

    with caplog.at_level(logging.INFO):
        ane_patch._log_gdn_recurrent_safe_cap(model, 0.50, 1, True)

    assert "requested 0.500 to 0.250" in caplog.text
    assert "recurrent qkv" in caplog.text


def test_min_viable_gdn_fraction_tracks_the_alignment():
    """Single-ANE slices align to 64, dual to 128, so the same model has a
    different floor in each mode."""
    gdn = _floor_gdn(512, 1536)

    assert ane_patch._min_viable_gdn_fraction(gdn, 128) == 0.25
    total = 2048
    assert (int(total * 0.25) // 128) * 128 >= 512
    assert (int(total * 0.15) // 128) * 128 < 512

    # Exact z must satisfy the active ANE alignment. A wider aligned prefix
    # would enter recurrent qkv rows and is intentionally rejected.
    assert ane_patch._min_viable_gdn_fraction(_floor_gdn(320, 1728), 64) == 0.15625
    assert ane_patch._min_viable_gdn_fraction(_floor_gdn(320, 1728), 128) is None

    # z alone larger than the whole projection can never engage
    assert ane_patch._min_viable_gdn_fraction(_floor_gdn(2050, 10), 128) is None


def test_warn_gdn_below_floor_names_the_floor(monkeypatch, caplog):
    model = SimpleNamespace(modules=lambda: [_floor_gdn(512, 1536)])
    monkeypatch.setattr(ane_patch, "_eligible_gdn", lambda module: True)

    with caplog.at_level(logging.WARNING):
        ane_patch._warn_gdn_below_floor(model, True, 0, 0.15, True)

    assert "0.150" in caplog.text and "0.250" in caplog.text


def test_warn_gdn_below_floor_stays_quiet_when_the_fraction_is_viable(
    monkeypatch, caplog
):
    """Only a below-floor fraction gets the warning: a 0 count from a compile
    failure or from GDN being off has its own reporting."""
    model = SimpleNamespace(modules=lambda: [_floor_gdn(512, 1536)])
    monkeypatch.setattr(ane_patch, "_eligible_gdn", lambda module: True)

    with caplog.at_level(logging.WARNING):
        ane_patch._warn_gdn_below_floor(model, True, 0, 0.45, True)
        ane_patch._warn_gdn_below_floor(model, False, 0, 0.15, True)
        ane_patch._warn_gdn_below_floor(model, True, 12, 0.15, True)

    assert "floor" not in caplog.text


def test_tuner_floor_delegates_to_the_patch_rule():
    """One implementation of the bank rule, so the tuner grid clamp and the
    enable-path warning cannot disagree."""
    from omlx.admin import ane_tuning
    from omlx.patches import qwen35_ane_prefill as patch

    gdn = _floor_gdn(512, 1536)
    assert ane_tuning._min_viable_gdn_fraction(
        patch, gdn, 128
    ) == patch._min_viable_gdn_fraction(gdn, 128)
