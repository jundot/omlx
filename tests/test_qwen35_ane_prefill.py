import logging
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


def test_enable_splits_banks_when_monolithic_load_fails(monkeypatch):
    monkeypatch.delenv("OMLX_QWEN35_ANE_BANK_MAX_BYTES", raising=False)
    monkeypatch.setattr(fast, "qwen35_ane_available", lambda: True)
    monkeypatch.setattr(fast, "has_symbol", lambda name: True)
    monkeypatch.setattr(ane_patch, "_install_dispatch", lambda: True)
    monkeypatch.setattr(ane_patch, "_eligible_pair", lambda mlp: True)
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
    monkeypatch.setattr(ane_patch, "_install_dispatch", lambda: True)
    monkeypatch.setattr(ane_patch, "_eligible_pair", lambda mlp: True)
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
    monkeypatch.setattr(ane_patch, "_install_dispatch", lambda: True)
    monkeypatch.setattr(ane_patch, "_eligible_pair", lambda mlp: True)
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
    monkeypatch.setattr(ane_patch, "_install_dispatch", lambda: True)
    monkeypatch.setattr(ane_patch, "_eligible_pair", lambda mlp: True)

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


@pytest.mark.parametrize("bits", [5, 6])
@pytest.mark.parametrize("dual", [False, True])
def test_q5_q6_mlp_is_eligible_and_uses_generic_fused_swiglu(
    monkeypatch, bits, dual
):
    assert ane_patch._eligible_pair(_make_affine_mlp(bits, 64))
    generic_name = ane_patch._fused_swiglu_symbol(bits, dual=dual)
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
        lambda *args: pytest.fail("Q5/Q6 must use the generic fused SwiGLU"),
    )

    import omlx.patches.qwen35_q4_mlp as q4_patch

    monkeypatch.setattr(q4_patch, "_linear_qmm", lambda linear, value, variant: value)
    model0 = object()
    model1 = object() if dual else None
    state = ane_patch._CombinedMLPState(
        model=model0,
        model1=model1,
        weight=mx.zeros((4, bits), dtype=mx.uint32),
        scales=mx.zeros((4, 1), dtype=mx.bfloat16),
        biases=mx.zeros((4, 1), dtype=mx.bfloat16),
        ane_outputs=2,
        gpu_outputs=2,
        bits=bits,
        group_size=64,
    )
    mlp = SimpleNamespace(
        down_proj=object(),
        _omlx_ane_prefill_config=ane_patch._AnePrefillConfig(
            1, 0.5, 8, dual_ane=dual
        ),
        _omlx_ane_prefill_state=state,
    )
    x = mx.zeros((1, 1, 32), dtype=mx.bfloat16)

    result = ane_patch._backend(mlp, x)
    mx.eval(result)

    expected = (
        x,
        state.weight,
        state.scales,
        state.biases,
        model0,
        *(() if not dual else (model1,)),
        bits,
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
    assert state.weight.shape == (192, 32)
    assert state.scales.shape == (192, 2)
    assert compiled == [((192, 128), mx.float32, 2048)]


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

    assert compiled == [(192, 128)]
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
    assert compiled == [((192, 128), mx.float32, 2048)]
    assert state.z_outputs == 128
    assert state.qkv_outputs == 256
    assert state.weight.shape == (192, 20)
    assert state.scales.shape == (192, 2)
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
    assert state.weight.shape == (128, 16)
    assert state.scales.shape == (128, 2)
    assert dense0.shape == (128, 128)
    assert dense1.shape == (128, 128)


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
