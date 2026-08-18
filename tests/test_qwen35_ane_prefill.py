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


@pytest.mark.parametrize("sequence_length", [2048, 4096])
def test_configure_scheduler_uses_the_compiled_ane_shape(sequence_length):
    scheduler = SimpleNamespace(
        config=SimpleNamespace(prefill_step_size=2048),
        _qwen35_prefill_floor=4096,
    )

    configured = ane_patch.configure_qwen35_ane_prefill_scheduler(
        scheduler,
        sequence_length,
    )

    assert configured is True
    assert scheduler.config.prefill_step_size == sequence_length
    assert scheduler._qwen35_prefill_floor == 0


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
