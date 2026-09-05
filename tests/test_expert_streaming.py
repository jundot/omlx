# SPDX-License-Identifier: Apache-2.0
"""Tests for MoE expert streaming (SSD) — residency, settings, and forced logic."""

import json
import struct
import tempfile
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from omlx.admin import routes as admin_routes
from omlx.engine_pool import EngineEntry, EnginePool
from omlx.model_settings import ModelSettings
from omlx.patches.expert_streaming.residency import expert_streaming_estimate


def _failed_pool() -> tuple[EnginePool, EngineEntry]:
    pool = EnginePool()
    entry = EngineEntry(
        model_id="glm-moe",
        model_path="/tmp/glm-moe",
        model_type="llm",
        engine_type="batched",
        estimated_size=1,
        load_failed=True,
        load_failure_message="failed",
        load_failure_at=1.0,
    )
    pool._entries[entry.model_id] = entry
    return pool, entry


async def _update_settings(pool, settings, request):
    manager = MagicMock()
    manager.get_settings.return_value = settings
    state = MagicMock()
    with (
        patch("omlx.admin.routes._get_engine_pool", return_value=pool),
        patch("omlx.admin.routes._get_settings_manager", return_value=manager),
        patch("omlx.admin.routes._get_server_state", return_value=state),
    ):
        result = await admin_routes.update_model_settings("glm-moe", request, is_admin=True)
    manager.set_settings.assert_called_once()
    return result


def _write_fake_glm_checkpoint(tmp: Path, num_layers=4, experts=8, hidden=32, moe_hidden=16):
    """Create a minimal fake GLM MoE DSA checkpoint with headers only (no real weights)."""
    config = {
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": num_layers,
        "n_routed_experts": experts,
        "hidden_size": hidden,
        "moe_intermediate_size": moe_hidden,
        "first_k_dense_replace": 0,
        "moe_layer_freq": 1,
    }
    (tmp / "config.json").write_text(json.dumps(config))
    # Build a single safetensors file with stacked expert banks per layer
    # We write valid safetensors header + dummy data.
    import numpy as np

    # helper to write one safetensors file with a set of tensors
    # Use safetensors library if available, otherwise manual header
    tensors = {}
    for layer in range(num_layers):
        for proj, shape in [
            ("gate_proj", (experts, moe_hidden, hidden)),
            ("up_proj", (experts, moe_hidden, hidden)),
            ("down_proj", (experts, hidden, moe_hidden)),
        ]:
            key = f"model.layers.{layer}.mlp.switch_mlp.{proj}.weight"
            # BF16: 2 bytes per element
            size = int(np.prod(shape)) * 2
            tensors[key] = (shape, "BF16", size)
        # shared experts etc not needed

    # Also add a dense tensor to have dense_bytes
    tensors["model.embed_tokens.weight"] = ((32000, hidden), "BF16", 32000 * hidden * 2)
    tensors["lm_head.weight"] = ((32000, hidden), "BF16", 32000 * hidden * 2)

    # Build header
    header = {}
    offset = 0
    for k, (shape, dtype, size) in tensors.items():
        header[k] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    header_bytes = json.dumps(header).encode()
    # Write file
    fname = tmp / "model.safetensors"
    with fname.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(b"\x00" * offset)
    # Also write index
    weight_map = {k: fname.name for k in tensors}
    (tmp / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


def test_residency_supported_for_glm():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _write_fake_glm_checkpoint(tmp, num_layers=4, experts=8)
        est = expert_streaming_estimate(tmp)
        assert est.supported is True
        assert est.num_moe_layers == 4
        assert est.experts_per_layer == 8
        assert est.per_expert_bytes > 0
        assert est.expert_bytes > 0
        assert est.dense_bytes > 0
        assert est.resident_bytes > est.streaming_bytes
        # budget 2 GiB should give some slots
        assert est.slots_for_budget(2 * 1024**3) == 8  # all experts fit


def test_residency_unsupported_for_dense_model():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "config.json").write_text(json.dumps({"model_type": "llama", "num_hidden_layers": 2}))
        # minimal safetensors with no expert keys
        header = {"model.embed_tokens.weight": {"dtype": "BF16", "shape": [100, 32], "data_offsets": [0, 6400]}}
        hb = json.dumps(header).encode()
        with (tmp / "model.safetensors").open("wb") as f:
            f.write(struct.pack("<Q", len(hb)))
            f.write(hb)
            f.write(b"\x00" * 6400)
        est = expert_streaming_estimate(tmp)
        assert est.supported is False


# Widened into the allowlist (Fase B3): all four expose the same stacked
# ``layers.N.mlp.switch_mlp.{gate,up,down}_proj`` layout and the
# ``moe_intermediate_size`` key that _resolve_moe_dims reads, so they already
# passed the structural estimate — they were only missing from the list.
# (experts_key varies: the Qwen families call it num_experts, DeepSeek/GLM-4
# call it n_routed_experts.)
_WIDENED_MOE_TYPES = [
    ("qwen3_moe", "num_experts"),
    ("qwen2_moe", "num_experts"),
    ("deepseek_v3", "n_routed_experts"),
    ("glm4_moe", "n_routed_experts"),
]


def _write_fake_switch_mlp_checkpoint(
    tmp: Path,
    model_type: str,
    experts_key: str,
    num_layers=4,
    experts=8,
    hidden=32,
    moe_hidden=16,
):
    """Minimal stacked-switch_mlp checkpoint (headers only) for one model type."""
    config = {
        "model_type": model_type,
        "num_hidden_layers": num_layers,
        experts_key: experts,
        "hidden_size": hidden,
        "moe_intermediate_size": moe_hidden,
    }
    (tmp / "config.json").write_text(json.dumps(config))
    tensors = {}
    for layer in range(num_layers):
        for proj, shape in [
            ("gate_proj", (experts, moe_hidden, hidden)),
            ("up_proj", (experts, moe_hidden, hidden)),
            ("down_proj", (experts, hidden, moe_hidden)),
        ]:
            key = f"model.layers.{layer}.mlp.switch_mlp.{proj}.weight"
            tensors[key] = (shape, "BF16", int(np.prod(shape)) * 2)
    tensors["model.embed_tokens.weight"] = ((32000, hidden), "BF16", 32000 * hidden * 2)

    header = {}
    offset = 0
    for k, (shape, dtype, size) in tensors.items():
        header[k] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    header_bytes = json.dumps(header).encode()
    with (tmp / "model.safetensors").open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(b"\x00" * offset)
    (tmp / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in tensors}})
    )
    return config


@pytest.mark.parametrize("model_type,experts_key", _WIDENED_MOE_TYPES)
def test_widened_moe_type_estimate_supported(model_type, experts_key):
    """The four widened types produce a usable streaming estimate."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _write_fake_switch_mlp_checkpoint(
            tmp, model_type, experts_key, num_layers=4, experts=8
        )
        est = expert_streaming_estimate(tmp)
        assert est.supported is True, est.reason
        assert est.num_moe_layers == 4
        assert est.experts_per_layer == 8
        assert est.per_expert_bytes > 0
        assert est.expert_bytes > 0
        assert est.dense_bytes > 0
        assert est.resident_bytes > est.streaming_bytes


@pytest.mark.parametrize("model_type,experts_key", _WIDENED_MOE_TYPES)
def test_widened_moe_type_dims_not_defaulted(model_type, experts_key):
    """Dims regression: the resolved moe_hidden must come from the config's
    moe_intermediate_size, never from the 1407 fallback in _resolve_moe_dims.

    1407 is a GLM-4.5-Air-specific constant. A silent fallthrough would slice
    every expert bank at the wrong offset — corrupt outputs with no error.
    """
    from omlx.patches.expert_streaming import _resolve_moe_dims

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        config = _write_fake_switch_mlp_checkpoint(
            tmp, model_type, experts_key, num_layers=2, experts=4, hidden=32, moe_hidden=16
        )
        hidden, moe_hidden = _resolve_moe_dims([config])
        assert moe_hidden == 16, f"{model_type}: fell back to the 1407 default"
        assert hidden == 32


def _write_no_layer_index_checkpoint(
    tmp: Path,
    model_type: str,
    experts_key: str,
    num_layers=4,
    experts=8,
    hidden=32,
    moe_hidden=16,
    extra_config: dict | None = None,
):
    """Checkpoint whose expert keys carry no ``layers.N.`` index.

    Forces the estimate down the config-only fallback path (the header scan
    finds no layer indices), which is what _CONFIG_DERIVABLE_MOE_LAYERS and
    _ALL_LAYERS_MOE exist for.
    """
    config = {
        "model_type": model_type,
        "num_hidden_layers": num_layers,
        experts_key: experts,
        "hidden_size": hidden,
        "moe_intermediate_size": moe_hidden,
    }
    if extra_config:
        config.update(extra_config)
    (tmp / "config.json").write_text(json.dumps(config))
    tensors = {}
    for proj, shape in [
        ("gate_proj", (experts, moe_hidden, hidden)),
        ("up_proj", (experts, moe_hidden, hidden)),
        ("down_proj", (experts, hidden, moe_hidden)),
    ]:
        key = f"model.mlp.switch_mlp.{proj}.weight"
        tensors[key] = (shape, "BF16", int(np.prod(shape)) * 2)
    tensors["model.embed_tokens.weight"] = ((32000, hidden), "BF16", 32000 * hidden * 2)

    header = {}
    offset = 0
    for k, (shape, dtype, size) in tensors.items():
        header[k] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    header_bytes = json.dumps(header).encode()
    with (tmp / "model.safetensors").open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(b"\x00" * offset)
    (tmp / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in tensors}})
    )
    return config


def test_config_derivable_first_k_fallback():
    """deepseek_v3 / glm4_moe resolve MoE layers from first_k_dense_replace."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _write_no_layer_index_checkpoint(
            tmp, "deepseek_v3", "n_routed_experts",
            num_layers=8, experts=8, extra_config={"first_k_dense_replace": 3},
        )
        est = expert_streaming_estimate(tmp)
        assert est.supported is True, est.reason
        assert est.num_moe_layers == 5, est.num_moe_layers  # layers 3..7
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _write_no_layer_index_checkpoint(
            tmp, "glm4_moe", "n_routed_experts",
            num_layers=4, experts=8, extra_config={"first_k_dense_replace": 1},
        )
        est = expert_streaming_estimate(tmp)
        assert est.supported is True, est.reason
        assert est.num_moe_layers == 3, est.num_moe_layers  # layers 1..3


def test_config_derivable_qwen3_sparse_step():
    """qwen3_moe mirrors mlx: (i+1) % step == 0 minus mlp_only_layers."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _write_no_layer_index_checkpoint(
            tmp, "qwen3_moe", "num_experts", num_layers=6, experts=8,
            extra_config={"decoder_sparse_step": 2, "mlp_only_layers": [0]},
        )
        est = expert_streaming_estimate(tmp)
        assert est.supported is True, est.reason
        assert est.num_moe_layers == 3, est.num_moe_layers  # i=1,3,5
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _write_no_layer_index_checkpoint(
            tmp, "qwen3_moe", "num_experts", num_layers=4, experts=8,
        )
        est = expert_streaming_estimate(tmp)
        assert est.supported is True, est.reason
        assert est.num_moe_layers == 4, est.num_moe_layers  # defaults: all MoE
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _write_no_layer_index_checkpoint(
            tmp, "qwen3_moe", "num_experts", num_layers=4, experts=8,
            extra_config={"decoder_sparse_step": 0},
        )
        est = expert_streaming_estimate(tmp)
        assert est.supported is True, est.reason
        assert est.num_moe_layers == 4, est.num_moe_layers  # invalid step -> 1


def test_all_layers_moe_qwen2():
    """qwen2_moe's decoder is unconditionally all-MoE."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _write_no_layer_index_checkpoint(
            tmp, "qwen2_moe", "num_experts", num_layers=5, experts=8,
        )
        est = expert_streaming_estimate(tmp)
        assert est.supported is True, est.reason
        assert est.num_moe_layers == 5, est.num_moe_layers


class _FakeMLPLayer:
    def __init__(self, e, o, i):
        self.mlp = _FakeMoE(e, o, i)


@pytest.mark.parametrize("model_type,experts_key", _WIDENED_MOE_TYPES)
def test_convert_walk_mlp_layout_widened(model_type, experts_key):
    """convert_model_to_streaming reaches the ``.mlp.switch_mlp`` banks of
    the four widened types — the runtime half of the allowlist widening.
    Header-only estimates cannot prove the layer walk finds the banks."""
    from omlx.patches.expert_streaming import convert_model_to_streaming
    from omlx.patches.expert_streaming.streaming_switch import StreamingSwitchGLU

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        num_layers, experts, hidden, moe_hidden = 2, 4, 32, 16
        _write_fake_switch_mlp_checkpoint(
            tmp, model_type, experts_key,
            num_layers=num_layers, experts=experts,
            hidden=hidden, moe_hidden=moe_hidden,
        )
        layers = [_FakeMLPLayer(experts, moe_hidden, hidden) for _ in range(num_layers)]
        model = _FakeTextModel(
            layers,
            [],
            {
                "model_type": model_type,
                experts_key: experts,
                "hidden_size": hidden,
                "moe_intermediate_size": moe_hidden,
                "num_hidden_layers": num_layers,
            },
        )
        out_model, _backing = convert_model_to_streaming(
            model, str(tmp), None, use_file_backing=False
        )
        assert out_model is model
        for layer in layers:
            assert isinstance(layer.mlp.switch_mlp, StreamingSwitchGLU), model_type


class _FakeQuantProj:
    def __init__(self, shape, scales_shape, group_size, bits, mode="affine"):
        import mlx.core as mx

        self.weight = mx.ones(shape)
        self.scales = mx.ones(scales_shape)
        self.group_size = group_size
        self.bits = bits
        self.mode = mode


class _FakeQuantSwitchMLP:
    def __init__(self, e, o, i):
        # JANGQ mixed precision: a 2-bit gate with 3-bit up/down.
        self.gate_proj = _FakeQuantProj((e, o, i), (e, o, 1), 32, 2)
        self.up_proj = _FakeQuantProj((e, o, i), (e, o, 1), 64, 3)
        self.down_proj = _FakeQuantProj((e, i, o), (e, 1, 1), 64, 3)


class _FakeQuantMoE:
    def __init__(self, e, o, i):
        self.switch_mlp = _FakeQuantSwitchMLP(e, o, i)


class _FakeQuantMLPLayer:
    def __init__(self, e, o, i):
        self.mlp = _FakeQuantMoE(e, o, i)


def test_convert_walk_mixed_precision_jangq():
    """Each streaming linear keeps its own source projection packing.

    JANGQ checkpoints mix precisions inside one layer; a layer-level
    single packing would silently requantize two of the three banks."""
    from omlx.patches.expert_streaming import convert_model_to_streaming
    from omlx.patches.expert_streaming.streaming_switch import (
        StreamingQuantizedSwitchLinear,
        StreamingSwitchGLU,
    )

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        num_layers, experts, hidden, moe_hidden = 2, 4, 32, 16
        _write_fake_switch_mlp_checkpoint(
            tmp, "qwen3_moe", "num_experts",
            num_layers=num_layers, experts=experts,
            hidden=hidden, moe_hidden=moe_hidden,
        )
        layers = [_FakeQuantMLPLayer(experts, moe_hidden, hidden) for _ in range(num_layers)]
        model = _FakeTextModel(
            layers,
            [],
            {
                "model_type": "qwen3_moe",
                "num_experts": experts,
                "hidden_size": hidden,
                "moe_intermediate_size": moe_hidden,
                "num_hidden_layers": num_layers,
            },
        )
        out_model, _backing = convert_model_to_streaming(
            model, str(tmp), None, use_file_backing=False
        )
        assert out_model is model
        for layer in layers:
            sm = layer.mlp.switch_mlp
            assert isinstance(sm, StreamingSwitchGLU)
            for proj in (sm.gate_proj, sm.up_proj, sm.down_proj):
                assert isinstance(proj, StreamingQuantizedSwitchLinear)
            assert (sm.gate_proj.bits, sm.gate_proj.group_size) == (2, 32)
            assert (sm.up_proj.bits, sm.up_proj.group_size) == (3, 64)
            assert (sm.down_proj.bits, sm.down_proj.group_size) == (3, 64)


def test_topk_inapplicable_is_soft_noop():
    """An approximate threshold on a type without a hook warns and stays exact."""
    from omlx.patches.expert_streaming import adaptive_topk

    try:
        for model_type in ("qwen3_moe", "qwen2_moe", "deepseek_v3", "glm4_moe"):
            assert not adaptive_topk.is_topk_applicable(model_type)
            assert (
                adaptive_topk.configure_from_settings(
                    ModelSettings(expert_streaming_topk_threshold=0.5),
                    model_type=model_type,
                )
                is None
            )
        # Applicable types still resolve; exact stays exact.
        for model_type in ("qwen4_exp", "glm5_next", "qwen4_exp_text", "glm5_next_text"):
            assert adaptive_topk.is_topk_applicable(model_type)
        assert (
            adaptive_topk.configure_from_settings(
                ModelSettings(expert_streaming_topk_threshold=0.5),
                model_type="qwen4_exp",
            )
            == 0.5
        )
        assert (
            adaptive_topk.configure_from_settings(ModelSettings(), model_type="qwen3_moe")
            is None
        )
    finally:
        adaptive_topk.configure(None)


def test_unlisted_switch_mlp_type_is_not_supported():
    """The OOM guard. A checkpoint with stacked switch_mlp banks whose
    model_type is outside the allowlist must NOT report as streamable.

    Structural detection alone would accept it — and EnginePool forces
    streaming from that flag, while the converter only reaches the banks on a
    lazy-loaded model. So an unlisted type would be forced into streaming
    without the lazy load and mlx_lm would materialize the full
    multi-hundred-GB banks: OOM / SIGKILL. This is the regression that
    motivated making the allowlist a hard gate.
    """
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # Same layout as a supported type, only the model_type differs.
        _write_fake_switch_mlp_checkpoint(
            tmp, "hypothetical_future_moe", "n_routed_experts", num_layers=4, experts=8
        )
        est = expert_streaming_estimate(tmp)
        assert est.supported is False
        assert est.reason is not None and "allowlist" in est.reason


def test_overhead_factor_is_single_sourced():
    """P3: the 1.05 overhead lives in residency.py; engine_pool and the
    Qwen PLE estimate import it instead of carrying literals."""
    import omlx.patches.expert_streaming.residency as res
    import omlx.patches.mlx_vlm_qwen4_exp_compat.residency as qres

    assert qres._MODEL_OVERHEAD_FACTOR == res._MODEL_OVERHEAD_FACTOR
    import inspect

    import omlx.engine_pool as ep

    src = inspect.getsource(ep.EnginePool._entry_runtime_resident_size)
    assert "overhead = 1.05" not in src
    assert "_MODEL_OVERHEAD_FACTOR" in src or "_OVERHEAD" in src


def test_allowlist_is_single_source_of_truth():
    """The package root must re-export residency.py's set, not keep a copy —
    a second literal is exactly how the gates drifted apart."""
    from omlx.patches.expert_streaming import SUPPORTED_TYPES
    from omlx.patches.expert_streaming import residency

    assert SUPPORTED_TYPES is residency.SUPPORTED_TYPES


def test_residency_force_streaming():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _write_fake_glm_checkpoint(tmp, num_layers=4, experts=16, hidden=64, moe_hidden=32)
        est = expert_streaming_estimate(tmp)
        assert est.supported is True
        # streaming should be smaller than resident
        assert est.resident_bytes > est.streaming_bytes
        # P0: the decision carries a margin + transient, so the ceiling
        # must clear streaming*margin + transient, not just streaming.
        ceiling = int(est.streaming_bytes * 1.10) + est.per_layer_expert_bytes
        assert est.force_streaming(ceiling) is True
        assert est.force_streaming(est.resident_bytes * 2) is False
        assert est.force_streaming(max(0, est.streaming_bytes - 1)) is False


def test_residency_force_streaming_margin_and_transient():
    """P0: a load exactly at the ceiling is refused, never admitted; the
    per-layer transient is charged on the streaming side."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # Wide experts so resident and streaming differ well beyond the
        # 10% decision margin (small fixtures overlap and must refuse).
        _write_fake_glm_checkpoint(tmp, num_layers=4, experts=32, hidden=256, moe_hidden=256)
        est = expert_streaming_estimate(tmp)
        assert est.supported is True
        assert est.resident_bytes > int(est.streaming_bytes * 1.10)
        # Exactly at resident: margin pushes resident over -> forced.
        assert est.force_streaming(est.resident_bytes) is True
        # Exactly at streaming: margin pushes streaming over -> not forced.
        assert est.force_streaming(est.streaming_bytes) is False
        # A ceiling that fits streaming_bytes but not its transient refuses.
        assert est.force_streaming(est.streaming_bytes, est.per_layer_expert_bytes) is False
        assert (
            est.force_streaming(
                int(est.streaming_bytes * 1.10) + est.per_layer_expert_bytes
            )
            is True
        )
        # Backwards compat: transient defaults to 0, ceiling 0 never forces.
        assert est.force_streaming(0) is False
        assert est.force_streaming(-1) is False


def _write_fake_cold_tier(model_tmp: Path, scale: float = 0.5):
    """P1 helper: expert_cold/ mirror with the same keys at ~scale bytes."""
    import numpy as np

    cold = model_tmp / "expert_cold"
    cold.mkdir(exist_ok=True)
    header = {}
    offset = 0
    index = json.loads((model_tmp / "model.safetensors.index.json").read_text())["weight_map"]
    src_hdr = {}
    for f in model_tmp.glob("*.safetensors"):
        raw = f.read_bytes()
        hsize = struct.unpack("<Q", raw[:8])[0]
        src_hdr.update(json.loads(raw[8:8 + hsize]))
    for k in sorted(index):
        if "switch_mlp" not in k:
            continue
        entry = src_hdr[k]
        size = max(8, int((entry["data_offsets"][1] - entry["data_offsets"][0]) * scale))
        header[k] = {"dtype": entry["dtype"], "shape": entry["shape"], "data_offsets": [offset, offset + size]}
        offset += size
    header["__metadata__"] = {"omlx_cold_bits": "2", "omlx_cold_group_size": "64"}
    hb = json.dumps(header).encode()
    with (cold / "cold-00001.safetensors").open("wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(b"\x00" * offset)
    return cold


def test_estimate_cold_tier_effective_bytes():
    """P1: with an active cold tier the estimate reports what decode reads."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _write_fake_glm_checkpoint(tmp, num_layers=4, experts=16, hidden=64, moe_hidden=64)
        base = expert_streaming_estimate(tmp)
        assert base.supported is True
        assert base.tier == "none"
        assert base.expert_bytes_effective == base.expert_bytes
        cold = _write_fake_cold_tier(tmp, scale=0.5)
        eff = expert_streaming_estimate(tmp, cold_root=cold)
        assert eff.supported is True
        assert eff.tier == "uniform"
        assert 0 < eff.expert_bytes_effective < base.expert_bytes
        assert 0 < eff.per_expert_bytes_effective < base.per_expert_bytes
        # HOBBIT blend sits between uniform-cold and source.
        hob = expert_streaming_estimate(tmp, cold_root=cold, hot_fraction=0.5)
        assert hob.tier == "hobbit"
        assert eff.expert_bytes_effective < hob.expert_bytes_effective < base.expert_bytes


def test_cache_slots_reconciled_for_fused_layout():
    """P0: a fused (2-proj) model must not over-commit the budget 1.5x.

    The convert-time reconciliation resizes capacity * per_slot <= budget
    for the majority projection layout. Unit-level: the invariant the
    reconciler enforces holds for both layouts.
    """
    from omlx.patches.expert_streaming.streaming_switch import ExpertLRUCache

    per_expert = 12 * 1024 * 1024  # 12 MiB across all projections
    budget = 6 * 1024**3
    for n_proj, name in ((3, "split"), (2, "fused")):
        per_slot = max(1, per_expert // n_proj)
        cache = ExpertLRUCache(budget, per_slot, num_layers=48)
        assert cache.capacity * cache.per_expert_bytes <= budget, name
        # The old //3 sizing on a fused model violates the invariant.
        if n_proj == 2:
            naive = ExpertLRUCache(budget, max(1, per_expert // 3), num_layers=48)
            assert naive.capacity * naive.per_expert_bytes <= budget  # global cap holds
            assert naive.capacity > cache.capacity  # ...but promises 1.5x more slots


def test_pilot_staging_equivalence_and_fused_wiring():
    """P1: staged bundles equal sync loads; fused projs are wired."""
    import numpy as np

    from omlx.patches.expert_streaming.prefetch import ExpertPrefetcher
    from omlx.patches.expert_streaming.streaming_switch import (
        ExpertLRUCache,
        StreamingQuantizedSwitchLinear,
    )

    class _FakeBacking:
        def load_expert_slice(self, key, eid):
            return np.full((4,), float(eid), dtype=np.float32)

    backing = _FakeBacking()
    cache = ExpertLRUCache(0, 1, num_layers=1)  # page-cache-only: staging only
    pf = ExpertPrefetcher(cache, workers=1)
    pf.start()
    try:
        lin = StreamingQuantizedSwitchLinear(
            layer_idx=0,
            proj_name="gate_up_proj",
            stacked_weight_key="w",
            stacked_scales_key="s",
            stacked_biases_key=None,
            num_experts=8,
            input_dims=16,
            output_dims=32,
            backing=backing,
            cache=cache,
        )
        lin._prefetcher = pf  # wiring path under test (fused proj included)
        assert hasattr(lin, "_load_expert_np")
        pf.submit(lin, [3, 5])
        deadline = __import__("time").time() + 10.0
        got = None
        while __import__("time").time() < deadline:
            got = pf.take(lin.bundle_key(3))
            if got is not None:
                break
            __import__("time").sleep(0.02)
        assert got is not None
        assert float(np.asarray(got[0]).ravel()[0]) == 3.0
        # direct load matches staged bytes
        direct = lin._load_expert_np(5)
        deadline = __import__("time").time() + 10.0
        staged5 = None
        while __import__("time").time() < deadline:
            staged5 = pf.take(lin.bundle_key(5))
            if staged5 is not None:
                break
            __import__("time").sleep(0.02)
        assert staged5 is not None
        assert np.array_equal(np.asarray(staged5[0]), np.asarray(direct[0]))
    finally:
        pf.stop()


def test_transition_table_predicts_and_overfetches():
    """FU1: temporal transitions accumulate (EWMA) and predict_next
    returns the top candidate excluding the demand set."""
    from omlx.patches.expert_streaming.streaming_switch import SpeculationState

    st = SpeculationState()
    # Cold table: no prediction.
    assert st.predict_next(0, [1, 2]) == []
    # Temporal credit prev -> now; repeated 2->3 wins over 2->1.
    st.record_prev(0, [1, 2])
    st.record_prev(0, [2, 3])
    st.record_prev(0, [2, 3])  # repeat to strengthen 2->3 over 2->2
    preds = st.predict_next(0, [2])
    assert 3 in preds  # strongest transition from expert 2
    assert 2 not in preds  # demand set excluded
    assert len(preds) <= 1  # default k=1
    # Other layers unaffected.
    assert st.predict_next(7, [2]) == []
    assert st.trans_updates == 2  # first record has no prev to credit
    # Bounded rows.
    st2 = SpeculationState()
    st2.record_prev(0, [0])
    st2.record_prev(0, list(range(20)))
    assert len(st2.trans.get((0, 0), {})) <= 8


def test_stash_ring_fifo_bound_and_hits():
    """Gap §7.3: the speculation stash ring is FIFO-bounded and hit-counted."""
    from omlx.patches.expert_streaming.streaming_switch import (
        _STASH_MAX_ENTRIES,
        SpeculationState,
    )

    st = SpeculationState()
    assert st.stash_get((0, 1, "w")) is None
    assert st.stats["stash_misses"] == 0  # get is pure; misses counted by caller
    for eid in range(_STASH_MAX_ENTRIES + 10):
        assert st.stash_insert((0, eid, "w"), ("w", "s", None)) is True
    assert len(st.stash) <= _STASH_MAX_ENTRIES
    assert st.stats["stash_evictions"] >= 10
    # Fresh insert hits; evicted old entry misses.
    assert st.stash_get((0, _STASH_MAX_ENTRIES + 9, "w")) is not None
    assert st.stats["stash_hits"] == 1
    assert st.stash_get((0, 0, "w")) is None
    # Closed state refuses inserts.
    st.close()
    assert st.stash_insert((0, 999, "w"), ("w", "s", None)) is False


def test_layer_context_rolling_union_equivalence():
    """Gap §7.3: rolling and union modes resolve identical bundles."""
    import numpy as np

    from omlx.patches.expert_streaming.streaming_switch import (
        _LayerLoadContext,
        ExpertLRUCache,
    )

    class _FakeLinear:
        def __init__(self, layer_idx, name):
            self.layer_idx = layer_idx
            self.proj_name = name
            self._io_pool_override = None

        def bundle_key(self, eid):
            return (self.layer_idx, int(eid), self.proj_name)

        def _tier_bank_bytes_for(self, ids):
            return len(ids) * 16

        def _load_expert_bank_np(self, ids):
            return [(np.full((4,), float(e), dtype=np.float32), None, None) for e in ids]

        def _load_expert_bank_np_full(self, ids):
            rows = self._load_expert_bank_np(ids)
            return ([(list(ids), [np.empty((len(ids), 16), dtype=np.uint8)])], rows)

    def _run(mode):
        cache = ExpertLRUCache(1 << 20, 4096, num_layers=1)
        lins = [_FakeLinear(0, f"p{i}") for i in range(3)]
        ctx = _LayerLoadContext(lins, cache, mode=mode)
        for lin in lins:
            ctx.ensure(lin, [2, 5, 7])
        return {
            lin.proj_name: {e: float(np.asarray(b[0]).ravel()[0]) for e, b in ctx.bundles[id(lin)].items()}
            for lin in lins
        }, dict(ctx.hits), dict(ctx.misses)

    rolling, rh, rm = _run("rolling")
    union, uh, um = _run("union")
    assert rolling == union
    for lin_bundles in rolling.values():
        assert lin_bundles == {2: 2.0, 5: 5.0, 7: 7.0}


def test_transition_profile_round_trip_with_fingerprint(tmp_path=None):
    """Transition payload survives save/load; mismatch is ignored."""
    import json

    from omlx.patches.expert_streaming import load_transition_profile
    from omlx.patches.expert_streaming.streaming_switch import SpeculationState

    st = SpeculationState()
    st.record_prev(0, [1, 2])
    st.record_prev(0, [2, 3])
    payload = st.to_payload()
    assert payload["updates"] == 1
    st2 = SpeculationState()
    assert st2.load_payload(payload) >= 1
    assert 3 in st2.predict_next(0, [2])
    # Fingerprint mismatch path: wrong model name ignored.
    import tempfile
    from pathlib import Path

    from omlx.patches.expert_streaming import save_transition_profile

    with tempfile.TemporaryDirectory() as td:
        model = Path(td) / "m"
        (model / ".omlx").mkdir(parents=True)
        (model / "config.json").write_text("{}")

        class _B:
            model_path = model
            spec_state = st

        save_transition_profile(_B())
        raw = json.loads((model / ".omlx" / "expert_transition.json").read_text())
        assert raw["model"] == "m"
        st3 = SpeculationState()
        assert load_transition_profile(_B(), st3) >= 1
        # Tamper the model name -> ignored.
        raw["model"] = "other"
        (model / ".omlx" / "expert_transition.json").write_text(json.dumps(raw))
        assert load_transition_profile(_B(), SpeculationState()) == 0


def test_residency_governor_hysteresis_and_bounds():
    """Dynamic residency: shrink/clear/grow from free memory, with floors."""
    import omlx.patches.expert_streaming.governor as gov_mod
    from omlx.patches.expert_streaming.governor import ExpertResidencyGovernor
    from omlx.patches.expert_streaming.streaming_switch import ExpertLRUCache

    G = 1024**3
    cache = ExpertLRUCache(int(2 * G), 696888, num_layers=48)
    g = ExpertResidencyGovernor(
        cache, 696888, 48, int(6 * G),
        low_free_bytes=6 * G, target_free_bytes=12 * G, high_free_bytes=24 * G,
        cooldown_s=0.0, min_cap=32,
    )
    start_cap = cache.capacity
    assert start_cap > 0

    # Fill the cache beyond a shrink target to see evictions on _apply.
    for l in range(48):
        for e in range(8):
            cache.put((l, e, "g"), ("w", "s", None))
    # Pressure: free between LOW and TGT -> halve, evict down to new cap.
    gov_mod._free_bytes = lambda: 8 * G
    act = g.observe(force=True)
    assert "shrink" in act and cache.capacity < start_cap
    assert len(cache._store) <= cache.capacity
    total = sum(cache._layer_counts.values())
    assert total == len(cache._store)
    # Critical: free below LOW -> clear.
    gov_mod._free_bytes = lambda: 4 * G
    act = g.observe(force=True)
    assert "clear" in act and len(cache._store) == 0
    # Abundance: free above HIGH -> grow, ceiling max_budget.
    gov_mod._free_bytes = lambda: 40 * G
    caps = []
    for _ in range(6):
        g.observe(force=True)
        caps.append(cache.capacity)
    assert caps == sorted(caps)  # monotone growth
    # Ceiling: capacity * per_slot <= max_budget (+1 slot rounding).
    assert cache.capacity * g.per_slot <= int(6 * G) + g.per_slot
    # Cooldown suppresses action.
    g2 = ExpertResidencyGovernor(
        cache, 696888, 48, int(6 * G), cooldown_s=3600.0,
    )
    gov_mod._free_bytes = lambda: 8 * G
    assert g2.observe() == ""  # inside cooldown
    assert g2.observe(force=True) != ""  # forced bypasses cooldown
    # Stable band: free between TGT and HIGH must not act (hysteresis).
    gov_mod._free_bytes = lambda: 8 * G
    g.observe(force=True)
    mid = (g.target_free_bytes + g.high_free_bytes) // 2
    gov_mod._free_bytes = lambda: mid
    cap_before = cache.capacity
    assert g.observe(force=True) == ""
    assert cache.capacity == cap_before

    # Summary carries governor state.
    s = g.summary()
    assert s["actions"] >= 1 and "capacity" in s


def test_s3fifo_cache_equivalence_and_scan_resistance():
    """P2: S3-FIFO honors the same interface; one-hit scans don't evict hot."""
    from omlx.patches.expert_streaming.streaming_switch import (
        ExpertLRUCache,
        S3FIFOExpertCache,
        make_expert_cache,
    )

    assert make_expert_cache(1 << 20, 4096, num_layers=1).policy == "lru"
    fifo = S3FIFOExpertCache(10 * 4096, 4096, num_layers=1)
    assert fifo.policy == "s3fifo"
    # Interface parity: put/get/contains/size/retain_hot/clear. A get
    # promotes small -> main, so referenced entries survive the small FIFO.
    for eid in range(5):
        fifo.put((0, eid, "w"), ("w", "s", None))
        assert fifo.get((0, eid, "w")) is not None
    assert fifo.size == 5
    assert (0, 0, "w") in fifo
    assert fifo.get((0, 99, "w")) is None
    assert fifo.stats.hits == 5 and fifo.stats.misses == 1
    assert fifo.retain_hot({(0, 0)}) == 4
    assert fifo.size == 1
    fifo.clear()
    assert fifo.size == 0
    # Scan resistance: hot set re-referenced, then a one-hit scan larger
    # than capacity. LRU loses hot; S3-FIFO keeps it in main.
    lru = ExpertLRUCache(5 * 4096, 4096, num_layers=1)
    s3 = S3FIFOExpertCache(5 * 4096, 4096, num_layers=1)
    # Interleaved put+get builds the hot set (decode-like reuse); a
    # burst put-then-get would be scan-like and correctly filtered.
    for c in (lru, s3):
        for eid in range(3):
            c.put((0, eid, "w"), ("w", "s", None))
            assert c.get((0, eid, "w")) is not None
    for eid in range(10, 20):
        lru.put((0, eid, "w"), ("w", "s", None))
        s3.put((0, eid, "w"), ("w", "s", None))
    lru_hot = sum(1 for eid in range(3) if lru.get((0, eid, "w")) is not None)
    s3_hot = sum(1 for eid in range(3) if s3.get((0, eid, "w")) is not None)
    assert s3_hot == 3
    assert lru_hot < 3


def test_expert_streaming_summary_aggregates_counters():
    """P1: the one-line summary reflects LRU + prefetch + stash + fallbacks."""
    from omlx.patches.expert_streaming import expert_streaming_summary
    from omlx.patches.expert_streaming.streaming_switch import ExpertLRUCache

    cache = ExpertLRUCache(1 << 20, 4096, num_layers=1)
    cache.put((0, 1, "w"), ("w", "s", None))
    assert cache.get((0, 1, "w")) is not None
    assert cache.get((0, 9, "w")) is None
    cache._count_ctx_fallback("read_failure")

    class _Backing:
        _expert_prefetcher = None

    s = expert_streaming_summary(cache, _Backing())
    assert s["lru_hits"] == 1
    assert s["lru_misses"] == 1
    assert s["lru_hit_rate"] == 0.5
    assert s["ctx_fallbacks"] == {"read_failure": 1}
    # Never raises on empty/None inputs.
    assert expert_streaming_summary(None, None)["ctx_fallbacks"] == {}


def test_shutdown_expert_streaming_is_idempotent_and_safe():
    """P0: shutdown handles None/dict/partial backings without raising."""
    from omlx.patches.expert_streaming import shutdown_expert_streaming

    shutdown_expert_streaming(None)
    shutdown_expert_streaming({})

    class _Partial:
        pass

    shutdown_expert_streaming(_Partial())

    stopped = []
    closed = []

    class _Full:
        class _Pref:
            def stop(self):
                stopped.append(True)

        _expert_prefetcher = _Pref()

        def close(self):
            closed.append(True)

    shutdown_expert_streaming(_Full())
    assert stopped == [True]
    assert closed == [True]


def test_reload_does_not_leak_prefetch_threads():
    """P0: PILOT threads are owned by the backing and stopped on shutdown."""
    import threading

    from omlx.patches.expert_streaming import shutdown_expert_streaming
    from omlx.patches.expert_streaming.prefetch import ExpertPrefetcher

    # Thread names repeat per prefetcher (expert-prefetch-{i}), so track
    # thread identities, not names.
    before = {t.ident for t in threading.enumerate()}
    prefs = [ExpertPrefetcher(cache=None, workers=2) for _ in range(3)]
    for p in prefs:
        p.start()
    assert all(w.is_alive() for p in prefs for w in p._workers)
    for p in prefs:

        class _Backing:
            def close(self):
                pass

        b = _Backing()
        b._expert_prefetcher = p  # type: ignore[attr-defined]
        shutdown_expert_streaming(b)
    for p in prefs:
        for w in p._workers:
            w.join(timeout=5.0)
    leaked = [
        t for t in threading.enumerate()
        if t.ident not in before and t.is_alive() and (t.name or "").startswith("expert-prefetch-")
    ]
    assert not leaked


def test_model_settings_round_trip():
    s = ModelSettings(expert_streaming_enabled=True, expert_streaming_budget_gib=3.5)
    d = s.to_dict()
    assert d["expert_streaming_enabled"] is True
    assert d["expert_streaming_budget_gib"] == 3.5
    s2 = ModelSettings.from_dict(d)
    assert s2.expert_streaming_enabled is True
    assert s2.expert_streaming_budget_gib == 3.5


def test_model_settings_default_not_streaming():
    s = ModelSettings()
    assert s.expert_streaming_enabled is False
    assert s.expert_streaming_budget_gib is None
    assert s.expert_streaming_budget_auto is True  # default-on


@pytest.mark.asyncio
async def test_expert_streaming_persists_via_api():
    pool, entry = _failed_pool()
    entry.config_model_type = "glm_moe_dsa"
    settings = ModelSettings()
    await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(expert_streaming_enabled=True, expert_streaming_budget_gib=2.0),
    )
    assert settings.expert_streaming_enabled is True
    assert settings.expert_streaming_budget_gib == 2.0


@pytest.mark.asyncio
async def test_expert_streaming_budget_validation():
    pool, _ = _failed_pool()
    settings = ModelSettings()
    with pytest.raises(admin_routes.HTTPException, match="between 0 and 64"):
        await _update_settings(
            pool,
            settings,
            admin_routes.ModelSettingsRequest(expert_streaming_budget_gib=100),
        )


@pytest.mark.asyncio
async def test_expert_streaming_budget_cleared_with_null():
    pool, _ = _failed_pool()
    settings = ModelSettings(expert_streaming_budget_gib=4.0)
    await _update_settings(pool, settings, admin_routes.ModelSettingsRequest(expert_streaming_budget_gib=None))
    assert settings.expert_streaming_budget_gib is None


@pytest.mark.asyncio
async def test_expert_streaming_change_triggers_reload_signature():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _write_fake_glm_checkpoint(tmp, num_layers=2, experts=4, hidden=32, moe_hidden=16)
        pool, entry = _failed_pool()
        entry.model_path = str(tmp)
        entry.config_model_type = "glm_moe_dsa"
        entry.engine = MagicMock()
        entry.load_failed = False
        pool._unload_engine = AsyncMock()
        result = await _update_settings(pool, ModelSettings(), admin_routes.ModelSettingsRequest(expert_streaming_enabled=True))
        assert result["requires_reload"] is True
        assert result["auto_unloaded"] is True


@pytest.mark.asyncio
async def test_expert_streaming_budget_auto_persists_via_api():
    pool, _ = _failed_pool()
    settings = ModelSettings()
    await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(expert_streaming_budget_auto=True),
    )
    assert settings.expert_streaming_budget_auto is True
    await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(expert_streaming_budget_auto=None),
    )
    assert settings.expert_streaming_budget_auto is None


def test_expert_streaming_excluded_from_profiles():
    from omlx.model_profiles import EXCLUDED_FROM_PROFILES

    assert "expert_streaming_enabled" in EXCLUDED_FROM_PROFILES
    assert "expert_streaming_budget_gib" in EXCLUDED_FROM_PROFILES
    assert "expert_streaming_budget_auto" in EXCLUDED_FROM_PROFILES


def test_engine_pool_forced_streaming_status(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _write_fake_glm_checkpoint(tmp, num_layers=4, experts=8, hidden=64, moe_hidden=32)
        pool = EnginePool()
        entry = EngineEntry(
            model_id="m",
            model_path=str(tmp),
            model_type="llm",
            engine_type="batched",
            estimated_size=1,
            config_model_type="glm_moe_dsa",
        )
        pool._entries["m"] = entry

        est = expert_streaming_estimate(tmp)
        assert est.supported is True
        assert est.resident_bytes > est.streaming_bytes
        # P0: the forcing window is [streaming*margin + transient,
        # resident*margin); the old midpoint may sit below it on small
        # fixtures, so target the window explicitly.
        lo = int(est.streaming_bytes * 1.10) + est.per_layer_expert_bytes
        hi = int(est.resident_bytes * 1.10)
        assert lo < hi
        ceiling = (lo + hi) // 2
        monkeypatch.setattr(pool, "_fallback_admission_ceiling", lambda: ceiling)
        monkeypatch.setattr(pool, "_current_ceiling", lambda: ceiling)
        enabled, forced, _ = pool._expert_streaming_status(entry, ModelSettings())
        assert forced is True
        assert enabled is True
        eff = pool._effective_expert_streaming_settings(entry, ModelSettings())
        assert eff.expert_streaming_enabled is True


def test_engine_pool_dflash_blocked_by_expert_streaming(monkeypatch):
    """DFlash selection must yield to expert streaming (requested or forced):
    DFlash's pipeline loads the target fully resident and never applies the
    streaming patches, so the two settings are mutually exclusive."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _write_fake_glm_checkpoint(tmp, num_layers=4, experts=8, hidden=64, moe_hidden=32)
        pool = EnginePool()
        entry = EngineEntry(
            model_id="m",
            model_path=str(tmp),
            model_type="llm",
            engine_type="batched",
            estimated_size=1,
            config_model_type="glm_moe_dsa",
        )

        # Requested streaming: the explicit setting wins over DFlash.
        dflash_settings = ModelSettings(
            dflash_enabled=True, dflash_draft_model="/tmp/draft"
        )
        assert pool._dflash_blocked_by_expert_streaming(entry, dflash_settings) is False
        dflash_settings.expert_streaming_enabled = True
        assert pool._dflash_blocked_by_expert_streaming(entry, dflash_settings) is True

        # Forced streaming: memory ceiling forces it without an explicit setting.
        est = expert_streaming_estimate(tmp)
        _lo = int(est.streaming_bytes * 1.10) + est.per_layer_expert_bytes
        _hi = int(est.resident_bytes * 1.10)
        assert _lo < _hi
        ceiling = (_lo + _hi) // 2
        monkeypatch.setattr(pool, "_fallback_admission_ceiling", lambda: ceiling)
        monkeypatch.setattr(pool, "_current_ceiling", lambda: ceiling)
        unrequested = ModelSettings(
            dflash_enabled=True, dflash_draft_model="/tmp/draft"
        )
        assert pool._dflash_blocked_by_expert_streaming(entry, unrequested) is True

        # Streaming that cannot fit streamed (streaming_bytes > ceiling) does
        # not force, so DFlash stays eligible.
        tiny_ceiling = est.streaming_bytes // 2
        monkeypatch.setattr(pool, "_fallback_admission_ceiling", lambda: tiny_ceiling)
        monkeypatch.setattr(pool, "_current_ceiling", lambda: tiny_ceiling)
        assert pool._dflash_blocked_by_expert_streaming(entry, unrequested) is False


# ---------------------------------------------------------------------------
# DeepSeek V4 (ffn-nested MoE + MTP/DSpark stages)
# ---------------------------------------------------------------------------


def _write_fake_dsv4_checkpoint(
    tmp: Path,
    num_layers=2,
    experts=8,
    hidden=32,
    moe_hidden=16,
    mtp_stages=2,
):
    """Fake DeepSeek V4 checkpoint: ffn-nested switch_mlp + mtp.<stage> banks."""
    import numpy as np

    config = {
        "model_type": "deepseek_v4",
        "num_hidden_layers": num_layers,
        "n_routed_experts": experts,
        "hidden_size": hidden,
        "moe_intermediate_size": moe_hidden,
    }
    (tmp / "config.json").write_text(json.dumps(config))

    tensors = {}

    def _add_bank(prefix: str) -> None:
        for proj, shape in [
            ("gate_proj", (experts, moe_hidden, hidden)),
            ("up_proj", (experts, moe_hidden, hidden)),
            ("down_proj", (experts, hidden, moe_hidden)),
        ]:
            key = f"{prefix}.{proj}.weight"
            tensors[key] = (shape, "BF16", int(np.prod(shape)) * 2)

    for layer in range(num_layers):
        _add_bank(f"model.layers.{layer}.ffn.switch_mlp")
    for stage in range(mtp_stages):
        _add_bank(f"mtp.{stage}.ffn.switch_mlp")

    tensors["model.embed_tokens.weight"] = ((32000, hidden), "BF16", 32000 * hidden * 2)
    tensors["lm_head.weight"] = ((32000, hidden), "BF16", 32000 * hidden * 2)

    header = {}
    offset = 0
    for k, (shape, dtype, size) in tensors.items():
        header[k] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    header_bytes = json.dumps(header).encode()
    fname = tmp / "model.safetensors"
    with fname.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(b"\x00" * offset)
    weight_map = {k: fname.name for k in tensors}
    (tmp / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


def test_residency_supported_for_deepseek_v4_with_mtp():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        num_layers, experts, stages = 3, 8, 2
        _write_fake_dsv4_checkpoint(tmp, num_layers=num_layers, experts=experts, mtp_stages=stages)
        est = expert_streaming_estimate(tmp)
        assert est.supported is True
        # MTP/DSpark stages count as streamable MoE layers
        assert est.num_moe_layers == num_layers + stages
        assert est.experts_per_layer == experts
        assert est.per_expert_bytes > 0
        assert est.expert_bytes > 0
        assert est.resident_bytes > est.streaming_bytes
        assert est.slots_for_budget(64 * 1024**3) == experts  # all fit


def test_residency_mtp_excluded_when_absent():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _write_fake_dsv4_checkpoint(tmp, num_layers=2, experts=8, mtp_stages=0)
        est = expert_streaming_estimate(tmp)
        assert est.supported is True
        assert est.num_moe_layers == 2


def test_stacked_key_candidates_cover_ffn_and_mtp():
    from omlx.patches.expert_streaming import (
        _candidate_stacked_keys,
        _mtp_candidate_stacked_keys,
        _resolve_stacked_key,
    )

    cands = _candidate_stacked_keys(5, "gate_proj", "weight")
    assert cands[0] == "model.layers.5.mlp.switch_mlp.gate_proj.weight"
    assert "model.layers.5.ffn.switch_mlp.gate_proj.weight" in cands

    mtp_cands = _mtp_candidate_stacked_keys(2, "up_proj", "scales")
    assert mtp_cands[0] == "mtp.2.ffn.switch_mlp.up_proj.scales"
    assert mtp_cands[1] == "mtp.2.block.ffn.switch_mlp.up_proj.scales"

    class _MapFFN:
        _weight_map = {"model.layers.5.ffn.switch_mlp.gate_proj.weight": "a.safetensors"}

    assert (
        _resolve_stacked_key(cands, "gate_proj", "weight", _MapFFN(), "layers.5.")
        == "model.layers.5.ffn.switch_mlp.gate_proj.weight"
    )

    class _MapScan:
        _weight_map = {"model.layers.5.ffn.switch_mlp.up_proj.weight": "b.safetensors"}

    assert (
        _resolve_stacked_key(
            _candidate_stacked_keys(5, "up_proj", "weight"),
            "up_proj",
            "weight",
            _MapScan(),
            "layers.5.",
        )
        == "model.layers.5.ffn.switch_mlp.up_proj.weight"
    )

    # no weight map (RAM dict / single-file) -> legacy mlp default
    assert (
        _resolve_stacked_key(cands, "gate_proj", "weight", None, "layers.5.")
        == "model.layers.5.mlp.switch_mlp.gate_proj.weight"
    )


class _FakeProj:
    def __init__(self, shape):
        import mlx.core as mx

        self.weight = mx.ones(shape)


class _FakeSwitchGLU:
    def __init__(self, e, o, i, activation=None):
        self.gate_proj = _FakeProj((e, o, i))
        self.up_proj = _FakeProj((e, o, i))
        self.down_proj = _FakeProj((e, i, o))
        if activation is not None:
            self.activation = activation


class _FakeMoE:
    def __init__(self, e, o, i, activation=None):
        self.switch_mlp = _FakeSwitchGLU(e, o, i, activation)


class _FakeLayer:
    def __init__(self, e, o, i):
        self.ffn = _FakeMoE(e, o, i)


class _FakeDSparkStage:
    def __init__(self, e, o, i):
        self.ffn = _FakeMoE(e, o, i)


class _FakeTextModel:
    def __init__(self, layers, mtp, config):
        self.model = type("InnerModel", (), {})()
        self.model.layers = layers
        self.model.mtp = mtp
        self.config = config


def test_convert_walk_ffn_and_mtp():
    from omlx.patches.expert_streaming import convert_model_to_streaming
    from omlx.patches.expert_streaming.streaming_switch import (
        StreamingSwitchGLU,
        StreamingSwitchLinear,
    )

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        num_layers, experts, hidden, moe_hidden, stages = 2, 4, 8, 4, 2
        _write_fake_dsv4_checkpoint(
            tmp,
            num_layers=num_layers,
            experts=experts,
            hidden=hidden,
            moe_hidden=moe_hidden,
            mtp_stages=stages,
        )
        marker = object()
        layers = [_FakeLayer(experts, moe_hidden, hidden) for _ in range(num_layers)]
        layers[0].ffn.switch_mlp.activation = marker
        mtp = [_FakeDSparkStage(experts, moe_hidden, hidden) for _ in range(stages)]
        model = _FakeTextModel(
            layers,
            mtp,
            {
                "model_type": "deepseek_v4",
                "hidden_size": hidden,
                "moe_intermediate_size": moe_hidden,
            },
        )

        out_model, backing = convert_model_to_streaming(
            model, str(tmp), None, use_file_backing=False
        )
        assert out_model is model
        # ram-dict backing is not returned (existing contract)
        assert backing is None

        for layer in layers:
            sm = layer.ffn.switch_mlp
            assert isinstance(sm, StreamingSwitchGLU)
            assert isinstance(sm.gate_proj, StreamingSwitchLinear)
            assert isinstance(sm.down_proj, StreamingSwitchLinear)
        # activation copied from the original switch_mlp
        assert layers[0].ffn.switch_mlp._activation is marker
        assert layers[1].ffn.switch_mlp._activation is None
        # MTP stages converted with distinct cache layer ids
        for stage in mtp:
            assert isinstance(stage.ffn.switch_mlp, StreamingSwitchGLU)


def test_convert_walk_mtpblock_layout():
    from omlx.patches.expert_streaming import convert_model_to_streaming
    from omlx.patches.expert_streaming.streaming_switch import StreamingSwitchGLU

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        num_layers, experts, hidden, moe_hidden = 2, 4, 8, 4
        _write_fake_dsv4_checkpoint(
            tmp,
            num_layers=num_layers,
            experts=experts,
            hidden=hidden,
            moe_hidden=moe_hidden,
            mtp_stages=0,
        )
        layers = [_FakeLayer(experts, moe_hidden, hidden) for _ in range(num_layers)]

        class _Block:
            def __init__(self, e, o, i):
                self.ffn = _FakeMoE(e, o, i)

        class _MTPBlockStage:
            def __init__(self, e, o, i):
                self.block = _Block(e, o, i)

        mtp = [_MTPBlockStage(experts, moe_hidden, hidden)]
        model = _FakeTextModel(
            layers,
            mtp,
            {
                "model_type": "deepseek_v4",
                "hidden_size": hidden,
                "moe_intermediate_size": moe_hidden,
            },
        )
        convert_model_to_streaming(model, str(tmp), None, use_file_backing=False)
        assert isinstance(layers[0].ffn.switch_mlp, StreamingSwitchGLU)
        assert isinstance(mtp[0].block.ffn.switch_mlp, StreamingSwitchGLU)


def test_streaming_glu_uses_custom_activation():
    import mlx.core as mx

    from omlx.patches.expert_streaming.streaming_switch import (
        ExpertLRUCache,
        StreamingSwitchGLU,
        StreamingSwitchLinear,
    )

    experts, moe_hidden, hidden = 4, 6, 8
    calls = []

    def act(up, gate):
        calls.append((tuple(up.shape), tuple(gate.shape)))
        return up + gate

    cache = ExpertLRUCache(1 << 24, 4096, num_layers=1)
    backing = {
        (0, "gate_proj"): mx.arange(experts * moe_hidden * hidden, dtype=mx.float32).reshape(
            experts, moe_hidden, hidden
        )
        / 100.0,
        (0, "up_proj"): mx.arange(experts * moe_hidden * hidden, dtype=mx.float32).reshape(
            experts, moe_hidden, hidden
        )
        / 200.0,
        (0, "down_proj"): mx.arange(experts * hidden * moe_hidden, dtype=mx.float32).reshape(
            experts, hidden, moe_hidden
        )
        / 300.0,
    }
    glu = StreamingSwitchGLU(
        input_dims=hidden,
        hidden_dims=moe_hidden,
        num_experts=experts,
        layer_idx=0,
        backing=backing,
        cache=cache,
        quantized=False,
        activation=act,
    )
    for name, out_dim, in_dim in [
        ("gate_proj", moe_hidden, hidden),
        ("up_proj", moe_hidden, hidden),
        ("down_proj", hidden, moe_hidden),
    ]:
        setattr(
            glu,
            name,
            StreamingSwitchLinear(
                layer_idx=0,
                proj_name=name,
                stacked_key=f"k.{name}",
                num_experts=experts,
                input_dims=in_dim,
                output_dims=out_dim,
                backing=backing,
                cache=cache,
            ),
        )

    x = mx.random.uniform(shape=(1, 3, hidden))
    idx = mx.array([[[0, 1], [2, 3], [0, 2]]])  # (1, 3, 2) — below sort threshold
    out = glu(x, idx)
    # custom activation used with the original call order (up, gate)
    assert len(calls) == 1
    assert glu._activation is act
    assert out.shape == (1, 3, 2, hidden)


def _write_shard(path: Path, tensors: dict[str, tuple[tuple[int, ...], str]]) -> None:
    """Write a minimal safetensors file: valid headers + zero data."""
    import numpy as np

    _DTY_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "U32": 4, "U8": 1}
    header: dict = {}
    offset = 0
    data: dict[str, int] = {}
    for key, (shape, dtype) in tensors.items():
        nbytes = int(np.prod(shape)) * _DTY_BYTES[dtype]
        header[key] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
        data[key] = nbytes
    hb = json.dumps(header).encode()
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(b"\x00" * offset)


def test_backing_store_multi_root_resolution():
    from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        primary = root / "model"
        extra = root / "stripe"
        extra.mkdir()
        primary.mkdir()

        # Primary shard: dense tensor only. Expert bank lives on the stripe root.
        _write_shard(primary / "model.safetensors", {"model.embed_tokens.weight": ((64, 32), "BF16")})
        _write_shard(extra / "experts.safetensors", {"model.layers.0.mlp.switch_mlp.gate_proj.weight": ((4, 16, 32), "BF16")})
        (primary / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "model.embed_tokens.weight": "model.safetensors",
                        "model.layers.0.mlp.switch_mlp.gate_proj.weight": "experts.safetensors",
                    }
                }
            )
        )

        backing = ExpertBackingStore(primary, extra_roots=[extra])
        assert backing._resolve_file("experts.safetensors") == (extra / "experts.safetensors").resolve()
        slc = backing.load_expert_slice("model.layers.0.mlp.switch_mlp.gate_proj.weight", 2)
        assert slc.shape == (16, 32)

        # Without the extra root the expert key cannot resolve.
        from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore as BS

        primary_only = BS(primary)
        with pytest.raises((KeyError, FileNotFoundError)):
            primary_only.load_expert_slice("model.layers.0.mlp.switch_mlp.gate_proj.weight", 0)


def test_backing_store_extra_root_wins_for_mirrored_shards():
    from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        primary = root / "model"
        extra = root / "stripe"
        primary.mkdir()
        extra.mkdir()
        _write_shard(primary / "model.safetensors", {"model.embed_tokens.weight": ((64, 32), "BF16")})
        _write_shard(extra / "model.safetensors", {"model.embed_tokens.weight": ((64, 32), "BF16")})
        (primary / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"model.embed_tokens.weight": "model.safetensors"}})
        )
        backing = ExpertBackingStore(primary, extra_roots=[extra])
        # Mirrored shard is served from the stripe root (that copy is the
        # striped one the user placed there deliberately).
        assert backing._resolve_file("model.safetensors") == (extra / "model.safetensors").resolve()


def test_np_to_mx_bf16_preserves_bits():
    """BF16 promotion must reinterpret raw bits (like mx.load) — subnormals
    included. The old shift->f32->astype path flushed them to zero (FTZ)."""
    import mlx.core as mx
    import numpy as np

    from omlx.patches.expert_streaming.shard_bank import _np_to_mx

    bits = np.array([0x0002, 0x0032, 0x0070, 0x3F80, 0xC000, 0x7F80], dtype=np.uint16)
    out = _np_to_mx("k", bits, "BF16")
    assert out.dtype == mx.bfloat16
    assert np.array_equal(np.array(out.view(mx.uint16)), bits)


def test_streaming_glu_matches_reference():
    """Streaming SwitchGLU (shared routing plan across gate/up/down) must be
    bit-exact against mlx-lm's reference SwitchGLU."""
    import mlx.core as mx
    import numpy as np

    from mlx_lm.models.switch_layers import SwitchGLU

    from omlx.patches.expert_streaming.streaming_switch import (
        ExpertLRUCache,
        StreamingSwitchGLU,
        StreamingSwitchLinear,
    )

    E, H, D = 8, 32, 16
    ref = SwitchGLU(D, H, E)  # default SwiGLU activation
    mx.eval(ref.parameters())

    cache = ExpertLRUCache(1 << 26, 1 << 18, num_layers=1)
    backing: dict = {}
    glu = StreamingSwitchGLU(
        input_dims=D,
        hidden_dims=H,
        num_experts=E,
        layer_idx=0,
        backing=backing,
        cache=cache,
        quantized=False,
        activation=None,
    )
    for name in ("gate_proj", "up_proj", "down_proj"):
        ref_lin = getattr(ref, name)
        backing[(0, name)] = ref_lin.weight
        setattr(
            glu,
            name,
            StreamingSwitchLinear(
                layer_idx=0,
                proj_name=name,
                stacked_key=f"k.{name}",
                num_experts=E,
                input_dims=ref_lin.input_dims,
                output_dims=ref_lin.output_dims,
                backing=backing,
                cache=cache,
            ),
        )

    rng = np.random.default_rng(42)
    for trial in range(3):
        T = 40 if trial < 2 else 80  # second size crosses the sort threshold
        x = mx.array(rng.standard_normal((1, T, D)).astype(np.float32))
        idx = mx.array(rng.integers(0, E, size=(1, T, 1)).astype(np.int32))
        out_ref = ref(x, idx)
        out_s = glu(x, idx)
        assert out_s.shape == out_ref.shape
        assert mx.array_equal(out_ref, out_s), f"mismatch at trial {trial}"


def _build_streaming_glu(ref, E, H, D, cache, backing, quantized_linears):
    """P1 helper: mirror a reference SwitchGLU with streaming linears.

    ``quantized_linears`` maps proj name -> (group_size, bits, mode) for
    projections built as StreamingQuantizedSwitchLinear; the rest become
    StreamingSwitchLinear. Returns the wired StreamingSwitchGLU.
    """
    import mlx.core as mx

    from omlx.patches.expert_streaming.streaming_switch import (
        StreamingQuantizedSwitchLinear,
        StreamingSwitchGLU,
        StreamingSwitchLinear,
    )

    glu = StreamingSwitchGLU(
        input_dims=D,
        hidden_dims=H,
        num_experts=E,
        layer_idx=0,
        backing=backing,
        cache=cache,
        quantized=bool(quantized_linears),
        activation=None,
    )
    for name in ("gate_proj", "up_proj", "down_proj"):
        ref_lin = getattr(ref, name)
        if name in quantized_linears:
            gs, bits, mode = quantized_linears[name]
            backing[(0, name, "weight")] = ref_lin.weight
            backing[(0, name, "scales")] = ref_lin.scales
            backing[(0, name, "biases")] = ref_lin.biases
            setattr(
                glu,
                name,
                StreamingQuantizedSwitchLinear(
                    layer_idx=0,
                    proj_name=name,
                    stacked_weight_key=f"k.{name}.weight",
                    stacked_scales_key=f"k.{name}.scales",
                    stacked_biases_key=f"k.{name}.biases",
                    num_experts=E,
                    input_dims=ref_lin.input_dims,
                    output_dims=ref_lin.output_dims,
                    backing=backing,
                    cache=cache,
                    group_size=gs,
                    bits=bits,
                    mode=mode,
                ),
            )
        else:
            backing[(0, name)] = ref_lin.weight
            setattr(
                glu,
                name,
                StreamingSwitchLinear(
                    layer_idx=0,
                    proj_name=name,
                    stacked_key=f"k.{name}",
                    num_experts=E,
                    input_dims=ref_lin.input_dims,
                    output_dims=ref_lin.output_dims,
                    backing=backing,
                    cache=cache,
                ),
            )
    mx.eval(ref.parameters())
    return glu


@pytest.mark.parametrize(
    "quantized_linears",
    [
        {},  # bf16 everywhere
        {"down_proj": (32, 4, "affine")},  # f32 gate/up + qmm down
        {
            "gate_proj": (32, 4, "affine"),
            "up_proj": (32, 4, "affine"),
            "down_proj": (32, 4, "affine"),
        },  # all-quantized q4
        {
            "gate_proj": (64, 8, "affine"),
            "up_proj": (64, 8, "affine"),
            "down_proj": (32, 8, "affine"),
        },  # all-quantized q8
        {
            "gate_proj": (32, 2, "affine"),  # JANGQ-style mixed packing
            "up_proj": (32, 4, "affine"),
            "down_proj": (32, 4, "affine"),
        },
    ],
    ids=["bf16", "q4-down", "q4-all", "q8-all", "mixed-jangq"],
)
def test_streaming_glu_quant_grade_matches_reference(quantized_linears):
    """P1: streaming-vs-resident bit-identical across the quant grade.

    The contract a new quantization must satisfy: same bytes, same
    layout, same output — adding a quant means adding a row here.
    """
    import mlx.core as mx
    import numpy as np

    from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchGLU

    from omlx.patches.expert_streaming.streaming_switch import ExpertLRUCache

    # D=64: gate/up last dim 64 (groups 32/64 valid), down last dim H=32.
    E, H, D = 8, 32, 64
    ref = SwitchGLU(D, H, E)
    for name, (gs, bits, _mode) in quantized_linears.items():
        proj = getattr(ref, name)
        ref.__dict__[name] = QuantizedSwitchLinear(
            proj.input_dims, proj.output_dims, E, bias=False,
            group_size=gs, bits=bits,
        )
    cache = ExpertLRUCache(1 << 26, 1 << 18, num_layers=1)
    backing: dict = {}
    glu = _build_streaming_glu(ref, E, H, D, cache, backing, quantized_linears)

    rng = np.random.default_rng(7)
    for trial in range(2):
        T = 40 if trial == 0 else 80  # second crosses the sort threshold
        x = mx.array(rng.standard_normal((1, T, D)).astype(np.float32))
        idx = mx.array(rng.integers(0, E, size=(1, T, 1)).astype(np.int32))
        out_ref = ref(x, idx)
        out_s = glu(x, idx)
        assert out_s.shape == out_ref.shape
        assert mx.array_equal(out_ref, out_s), f"mismatch {quantized_linears} trial {trial}"


def test_streaming_quantized_glu_matches_reference():
    """Quantized streaming GLU (f32 gate/up + qmm down, shared plan, sort
    path) must be bit-exact against the reference composition."""
    import mlx.core as mx
    import numpy as np

    from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchGLU

    from omlx.patches.expert_streaming.streaming_switch import (
        ExpertLRUCache,
        StreamingQuantizedSwitchLinear,
        StreamingSwitchGLU,
        StreamingSwitchLinear,
    )

    E, H, D = 8, 32, 16
    ref = SwitchGLU(D, H, E)
    ref.down_proj = QuantizedSwitchLinear(H, D, E, bias=False, group_size=32, bits=4)
    mx.eval(ref.parameters())

    cache = ExpertLRUCache(1 << 26, 1 << 18, num_layers=1)
    backing: dict = {}
    glu = StreamingSwitchGLU(
        input_dims=D,
        hidden_dims=H,
        num_experts=E,
        layer_idx=0,
        backing=backing,
        cache=cache,
        quantized=True,
        activation=None,
    )
    for name in ("gate_proj", "up_proj"):
        ref_lin = getattr(ref, name)
        backing[(0, name)] = ref_lin.weight
        setattr(
            glu,
            name,
            StreamingSwitchLinear(
                layer_idx=0,
                proj_name=name,
                stacked_key=f"k.{name}",
                num_experts=E,
                input_dims=ref_lin.input_dims,
                output_dims=ref_lin.output_dims,
                backing=backing,
                cache=cache,
            ),
        )
    ref_down = ref.down_proj
    backing[(0, "down_proj", "weight")] = ref_down.weight
    backing[(0, "down_proj", "scales")] = ref_down.scales
    backing[(0, "down_proj", "biases")] = ref_down.biases
    setattr(
        glu,
        "down_proj",
        StreamingQuantizedSwitchLinear(
            layer_idx=0,
            proj_name="down_proj",
            stacked_weight_key="k.down_proj.weight",
            stacked_scales_key="k.down_proj.scales",
            stacked_biases_key="k.down_proj.biases",
            num_experts=E,
            input_dims=ref_down.input_dims,
            output_dims=ref_down.output_dims,
            backing=backing,
            cache=cache,
            group_size=32,
            bits=4,
            mode="affine",
        ),
    )

    rng = np.random.default_rng(7)
    for trial in range(3):
        T = 40 if trial < 2 else 80
        x = mx.array(rng.standard_normal((1, T, D)).astype(np.float32))
        idx = mx.array(rng.integers(0, E, size=(1, T, 1)).astype(np.int32))
        out_ref = ref(x, idx)
        out_s = glu(x, idx)
        assert out_s.shape == out_ref.shape
        assert mx.array_equal(out_ref, out_s), f"mismatch at trial {trial}"


def test_budget_zero_means_page_cache_only():
    """Explicit budget 0 = page-cache only: no LRU slots, admission charges
    dense only (file-cache pages are clean/evictable, not committed)."""
    import mlx.core as mx

    from omlx.patches.expert_streaming import _get_budget_bytes
    from omlx.patches.expert_streaming.streaming_switch import ExpertLRUCache

    assert _get_budget_bytes(ModelSettings(expert_streaming_budget_gib=0.0), None) == 0
    assert _get_budget_bytes(ModelSettings(expert_streaming_budget_gib=None), None) == 0
    assert _get_budget_bytes(ModelSettings(expert_streaming_budget_gib=2.0), None) == 2 << 30

    cache = ExpertLRUCache(0, 1 << 20, num_layers=4)
    assert cache.capacity == 0
    cache.put((0, 0, "k"), mx.zeros(4))
    assert cache.size == 0
    assert cache.get((0, 0, "k")) is None


def test_estimate_scan_is_cached(caplog):
    """Repeat scans hit the lru_cache (identity) and emit the timed debug line."""
    import logging

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _write_fake_glm_checkpoint(tmp, num_layers=2, experts=4, hidden=32, moe_hidden=16)
        with caplog.at_level(logging.DEBUG, logger="omlx.patches.expert_streaming.residency"):
            est1 = expert_streaming_estimate(str(tmp))
            est2 = expert_streaming_estimate(str(tmp))
        assert est1 is est2
        assert any(
            "Expert streaming scan" in rec.message and "cache-hit" in rec.message
            for rec in caplog.records
        )


def test_detect_per_expert_jangq_layout():
    """Per-expert JANGQ banks count as routed experts.

    layers.N.ffn.experts.{i}.w{1,2,3}.* carry no stacked switch_mlp
    key; without this branch the estimate reports 'no expert tensors
    found', lazy load never arms, and the eager eval OOMs (dsv4-jang).
    Shared experts are dense-resident and must stay out.
    """
    from omlx.patches.expert_streaming.residency import _detect_expert_keys

    wm = {}
    for e in range(4):
        for src in ("w1", "w2", "w3"):
            for suf in ("weight", "scales", "biases"):
                wm[f"layers.0.ffn.experts.{e}.{src}.{suf}"] = "m.safetensors"
    for src in ("w1", "w2", "w3"):
        wm[f"layers.0.ffn.shared_experts.{src}.weight"] = "m.safetensors"
    cfg = {"model_type": "deepseek_v4", "num_hidden_layers": 1, "n_routed_experts": 4}

    keys, n_layers, n_routed = _detect_expert_keys(wm, {}, cfg)

    assert n_layers == 1
    assert n_routed == 4
    assert len(keys) == 4 * 3 * 3
    assert not any("shared_experts" in k for k in keys)


def test_engine_pool_budget_zero_streaming_bytes():
    from omlx.patches.expert_streaming.residency import expert_streaming_estimate

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _write_fake_glm_checkpoint(tmp, num_layers=2, experts=4, hidden=32, moe_hidden=16)
        est = expert_streaming_estimate(str(tmp))
        assert est.supported
        # 0 budget -> dense-only resident estimate (no cache term)
        assert est.streaming_bytes_for_budget(0) == int(est.dense_bytes * 1.05)
        assert est.slots_for_budget(0) == 0


def test_engine_pool_streaming_budget_helper():
    from omlx.engine_pool import EnginePool

    assert EnginePool._streaming_budget_bytes(None) == 0
    assert EnginePool._streaming_budget_bytes(ModelSettings()) == 0
    assert EnginePool._streaming_budget_bytes(ModelSettings(expert_streaming_budget_gib=0.0)) == 0
    assert EnginePool._streaming_budget_bytes(ModelSettings(expert_streaming_budget_gib=4)) == 4 << 30


class TestAutoStreamingBudget:
    GIB = 1024**3

    def test_manual_wins(self):
        from omlx.engine_pool import EnginePool

        assert (
            EnginePool._auto_streaming_budget_bytes(4.0, 10 * self.GIB, 64 * self.GIB)
            is None
        )
        # Explicit 0 (= page-cache only) also wins over auto.
        assert (
            EnginePool._auto_streaming_budget_bytes(0.0, 10 * self.GIB, 64 * self.GIB)
            is None
        )
        # Only an unset budget lets auto size the cache.
        assert (
            EnginePool._auto_streaming_budget_bytes(None, 10 * self.GIB, 64 * self.GIB)
            == 8 * self.GIB
        )

    def test_no_ceiling_or_no_headroom_declines(self):
        from omlx.engine_pool import EnginePool

        assert EnginePool._auto_streaming_budget_bytes(None, 10 * self.GIB, 0) is None
        # ceiling covers streaming + reserve exactly -> no headroom
        assert (
            EnginePool._auto_streaming_budget_bytes(None, 10 * self.GIB, 12 * self.GIB)
            is None
        )

    def test_half_headroom_capped_at_8gib(self):
        from omlx.engine_pool import EnginePool

        # headroom 52 GiB -> half 26, capped at 8 GiB default
        got = EnginePool._auto_streaming_budget_bytes(None, 10 * self.GIB, 64 * self.GIB)
        assert got == 8 * self.GIB
        # small headroom 6 GiB -> half 3 GiB, whole MiB
        got = EnginePool._auto_streaming_budget_bytes(None, 10 * self.GIB, 18 * self.GIB)
        assert got == 3 * self.GIB

    def test_knee_caps_below_default(self):
        from omlx.engine_pool import EnginePool

        got = EnginePool._auto_streaming_budget_bytes(
            None, 10 * self.GIB, 64 * self.GIB, knee_bytes=2 * self.GIB
        )
        assert got == 2 * self.GIB

    def test_knee_zero_means_page_cache(self):
        from omlx.engine_pool import EnginePool

        # bench measured the LRU as adding nothing -> cap 0 -> decline (None)
        assert (
            EnginePool._auto_streaming_budget_bytes(
                None, 10 * self.GIB, 64 * self.GIB, knee_bytes=0
            )
            is None
        )

    def test_effective_settings_resolves_auto(self):
        from types import SimpleNamespace

        from omlx.engine_pool import EnginePool

        pool = EnginePool()
        pool._get_final_ceiling = lambda: 0
        pool._fallback_admission_ceiling = lambda: 64 * self.GIB
        est = SimpleNamespace(supported=True, streaming_bytes=10 * self.GIB)
        entry = SimpleNamespace(model_id="m", model_path="/nonexistent")
        pool._expert_streaming_status = lambda _e, _s: (True, False, est)
        # Default settings (budget_auto True, no explicit budget) resolve.
        settings = ModelSettings(expert_streaming_enabled=True)
        with tempfile.TemporaryDirectory() as td:
            entry.model_path = td  # no knee file -> default 8 GiB cap
            out = pool._effective_expert_streaming_settings(entry, settings)
        assert out is not settings  # copied, input untouched
        assert settings.expert_streaming_budget_gib is None
        assert out.expert_streaming_budget_gib == 8.0

    def test_effective_settings_opt_out_declines(self):
        from types import SimpleNamespace

        from omlx.engine_pool import EnginePool

        pool = EnginePool()
        pool._get_final_ceiling = lambda: 0
        pool._fallback_admission_ceiling = lambda: 64 * self.GIB
        est = SimpleNamespace(supported=True, streaming_bytes=10 * self.GIB)
        entry = SimpleNamespace(model_id="m", model_path="/nonexistent")
        pool._expert_streaming_status = lambda _e, _s: (True, False, est)
        # Explicit False opts out: settings pass through untouched.
        settings = ModelSettings(
            expert_streaming_enabled=True, expert_streaming_budget_auto=False
        )
        out = pool._effective_expert_streaming_settings(entry, settings)
        assert out is settings
        assert out.expert_streaming_budget_gib is None
        # Explicit 0 (= page-cache only) also wins over auto: preserved
        # as-is, never overwritten with an auto budget.
        settings = ModelSettings(
            expert_streaming_enabled=True, expert_streaming_budget_gib=0.0
        )
        out = pool._effective_expert_streaming_settings(entry, settings)
        assert out.expert_streaming_budget_gib == 0.0

    def test_effective_settings_manual_wins_over_auto(self):
        from types import SimpleNamespace

        from omlx.engine_pool import EnginePool

        pool = EnginePool()
        pool._get_final_ceiling = lambda: 0
        pool._fallback_admission_ceiling = lambda: 64 * self.GIB
        est = SimpleNamespace(supported=True, streaming_bytes=10 * self.GIB)
        entry = SimpleNamespace(model_id="m", model_path="/nonexistent")
        settings = ModelSettings(
            expert_streaming_enabled=True,
            expert_streaming_budget_auto=True,
            expert_streaming_budget_gib=1.0,
        )
        out = pool._resolve_auto_budget_bytes(settings, est, entry)
        assert out is None


@pytest.mark.asyncio
async def test_expert_streaming_budget_zero_persists():
    pool, _ = _failed_pool()
    settings = ModelSettings()
    await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(expert_streaming_budget_gib=0),
    )
    assert settings.expert_streaming_budget_gib == 0.0


def test_adaptive_topk_configure_semantics():
    from omlx.patches.expert_streaming import adaptive_topk

    adaptive_topk.configure(None)
    assert adaptive_topk.current_threshold() is None
    adaptive_topk.configure(1.0)  # >= 1.0 normalizes to exact
    assert adaptive_topk.current_threshold() is None
    adaptive_topk.configure(0.85)
    assert adaptive_topk.current_threshold() == 0.85
    with pytest.raises(ValueError):
        adaptive_topk.configure(0.01)
    adaptive_topk.configure(None)
    assert adaptive_topk.configure_from_settings(ModelSettings()) is None
    assert (
        adaptive_topk.configure_from_settings(
            ModelSettings(expert_streaming_topk_threshold=0.9)
        )
        == 0.9
    )
    adaptive_topk.configure(None)


def test_adaptive_topk_truncate_math():
    import mlx.core as mx

    from omlx.patches.expert_streaming.adaptive_topk import truncate_topk_mass

    # sorted input, threshold 0.7: keep 0.5 + 0.3 (cum_before 0.8 crosses at 3rd)
    inds = mx.array([[7, 3, 5]], dtype=mx.uint32)
    scores = mx.array([[0.5, 0.3, 0.2]], dtype=mx.float32)
    i2, s2, keeps = truncate_topk_mass(inds, scores, 0.7, return_keeps=True)
    assert keeps == 2.0
    assert i2.tolist() == [[7, 3, 7]]  # dropped slot padded with top expert
    assert s2.shape == scores.shape
    # renormalized to the ORIGINAL total mass (1.0)
    total = mx.sum(s2, axis=-1)
    assert abs(float(total) - 1.0) < 1e-6
    assert s2[0][0].item() == pytest.approx(0.625, abs=1e-6)
    assert s2[0][1].item() == pytest.approx(0.375, abs=1e-6)
    assert s2[0][2].item() == 0.0

    # unsorted input: same kept SET, scores reordered descending
    inds3 = mx.array([[5, 7, 3]], dtype=mx.uint32)
    scores3 = mx.array([[0.2, 0.5, 0.3]], dtype=mx.float32)
    i3, s3 = truncate_topk_mass(inds3, scores3, 0.7)
    assert i3.tolist() == [[7, 3, 7]]

    # scale invariance (GLM routed_scaling_factor): x10 scores -> same split
    i4, s4 = truncate_topk_mass(inds, scores * 10.0, 0.7)
    assert i4.tolist() == [[7, 3, 7]]
    assert s4[0][0].item() == pytest.approx(6.25, abs=1e-5)

    # threshold >= total mass keeps everything (order changes only)
    i5, s5 = truncate_topk_mass(inds3, scores3, 0.999)
    assert sorted(i5[0].tolist()) == [3, 5, 7]
    assert abs(float(mx.sum(s5, axis=-1)) - 1.0) < 1e-6


def test_adaptive_topk_qwen_block_patch():
    """The patched Qwen3_5MoeSparseMoeBlock must bypass at exact threshold
    and produce finite, same-shape output when truncation engages."""
    import mlx.core as mx
    from types import SimpleNamespace

    pytest.importorskip("mlx_vlm")
    from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock
    from omlx.patches.expert_streaming import adaptive_topk

    cfg = SimpleNamespace(
        hidden_size=32,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=32,
        num_experts=8,
        num_experts_per_tok=2,
    )
    block = Qwen3_5MoeSparseMoeBlock(cfg)
    mx.eval(block.parameters())

    applied = adaptive_topk.apply_qwen35_moe_topk_patch()
    if not applied:
        pytest.skip("qwen3_5_moe language module not patchable here")

    rng = __import__("numpy").random.default_rng(3)
    x = mx.array(rng.standard_normal((1, 4, 32)).astype("float32"))

    adaptive_topk.configure(None)
    out_exact = block(x)
    mx.eval(out_exact)

    adaptive_topk.configure(0.6)
    out_trunc = block(x)
    mx.eval(out_trunc)

    assert out_trunc.shape == out_exact.shape
    assert bool(mx.isfinite(out_trunc).all())
    adaptive_topk.configure(None)


def test_shard_bank_pin_expert_mlock():
    """pin_expert locks the page-aligned file range and dedupes repeats AND
    shared pages (Fase L: pinned_bytes counts UNIQUE locked pages)."""
    import mlx.core as mx  # noqa: F401

    from omlx.patches.expert_streaming.shard_bank import (
        ExpertBackingStore,
        _PAGE_SIZE,
    )

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _write_shard(
            tmp / "model.safetensors",
            {"model.layers.0.mlp.switch_mlp.gate_proj.weight": ((4, 8, 16), "BF16")},
        )
        (tmp / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"model.layers.0.mlp.switch_mlp.gate_proj.weight": "model.safetensors"}})
        )
        backing = ExpertBackingStore(tmp)
        key = "model.layers.0.mlp.switch_mlp.gate_proj.weight"
        locked = backing.pin_expert(key, 0)
        assert locked > 0
        # Unique-page accounting: at least one full page wired, never below
        # the slice bytes.
        assert backing.pinned_bytes >= locked
        assert backing.pinned_bytes % _PAGE_SIZE == 0
        assert backing.pinned_count == 1
        # duplicate pin skipped
        assert backing.pin_expert(key, 0) == 0
        assert backing.pinned_count == 1
        # Fase L dedupe: expert 1 shares expert 0's page — the unique-page
        # tally must NOT grow (both live inside the first page).
        first = backing.pinned_bytes
        assert backing.pin_expert(key, 1) > 0
        assert backing.pinned_count == 2
        assert backing.pinned_bytes == first, "shared page must not double-charge"
        # range math sanity: page-rounded >= slice bytes
        off, end = backing._reader_for_key(key).expert_byte_range(key, 1)
        assert end - off == 8 * 16 * 2


def test_page_cache_warmer_flow():
    from omlx.patches.expert_streaming import warmer
    from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore
    from omlx.patches.expert_streaming.streaming_switch import (
        ExpertLRUCache,
        StreamingQuantizedSwitchLinear,
    )

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        tensors = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            tensors[f"model.layers.0.mlp.switch_mlp.{proj}.weight"] = ((4, 8, 16), "BF16")
            tensors[f"model.layers.0.mlp.switch_mlp.{proj}.scales"] = ((4, 8, 1), "U32")
            tensors[f"model.layers.0.mlp.switch_mlp.{proj}.biases"] = ((4, 8, 1), "U32")
        tensors["model.layers.1.mlp.switch_mlp.gate_proj.weight"] = ((4, 8, 16), "BF16")
        tensors["model.layers.1.mlp.switch_mlp.gate_proj.scales"] = ((4, 8, 1), "U32")
        tensors["model.layers.1.mlp.switch_mlp.gate_proj.biases"] = ((4, 8, 1), "U32")
        _write_shard(tmp / "model.safetensors", tensors)
        wm = {k: "model.safetensors" for k in tensors}
        (tmp / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))

        backing = ExpertBackingStore(tmp)
        cache = ExpertLRUCache(0, 1 << 12, num_layers=2)

        def mk_lin(layer, proj):
            return StreamingQuantizedSwitchLinear(
                layer_idx=layer,
                proj_name=proj,
                stacked_weight_key=f"model.layers.{layer}.mlp.switch_mlp.{proj}.weight",
                stacked_scales_key=f"model.layers.{layer}.mlp.switch_mlp.{proj}.scales",
                stacked_biases_key=f"model.layers.{layer}.mlp.switch_mlp.{proj}.biases",
                num_experts=4,
                input_dims=16,
                output_dims=8,
                backing=backing,
                cache=cache,
            )

        linears = {0: [mk_lin(0, p) for p in ("gate_proj", "up_proj", "down_proj")]}
        linears[1] = [mk_lin(1, "gate_proj")]
        w = warmer.PageCacheWarmer(linears)
        hook = warmer.WarmPinHook(w, None)
        # token 1: layer 0 records its set; layer 1 records its own
        hook.on_layer_start(0, 8)
        hook.on_layer_plan(0, [1, 2, 3], 8)
        hook.on_layer_start(1, 8)
        hook.on_layer_plan(1, [2], 8)
        # token 2: layer 0 fires warm for layer 1's token-1 set
        hook.on_layer_start(0, 8)
        hook.on_layer_plan(0, [0, 1], 8)
        # warm pool is async; give it a moment and confirm no crash + reads done
        import time

        deadline = time.time() + 5
        while time.time() < deadline and w.warmed == 0:
            time.sleep(0.05)
        assert w.warmed > 0
        # big-prefetch guard: positions > _MAX_WARM_ROWS records empty
        hook.on_layer_start(0, 4096)
        hook.on_layer_plan(0, [0, 1, 2, 3], 4096)
        assert w.last_uniq[0] == []


def _ra_warmer_setup(budget_bytes: int = 0):
    from omlx.patches.expert_streaming import warmer
    from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore
    from omlx.patches.expert_streaming.streaming_switch import (
        ExpertLRUCache,
        StreamingQuantizedSwitchLinear,
    )

    tmp = tempfile.mkdtemp()
    tmpp = Path(tmp)
    tensors = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        tensors[f"model.layers.0.mlp.switch_mlp.{proj}.weight"] = ((4, 8, 16), "BF16")
        tensors[f"model.layers.0.mlp.switch_mlp.{proj}.scales"] = ((4, 8, 1), "U32")
        tensors[f"model.layers.0.mlp.switch_mlp.{proj}.biases"] = ((4, 8, 1), "U32")
    tensors["model.layers.1.mlp.switch_mlp.gate_proj.weight"] = ((4, 8, 16), "BF16")
    tensors["model.layers.1.mlp.switch_mlp.gate_proj.scales"] = ((4, 8, 1), "U32")
    tensors["model.layers.1.mlp.switch_mlp.gate_proj.biases"] = ((4, 8, 1), "U32")
    _write_shard(tmpp / "model.safetensors", tensors)
    wm = {k: "model.safetensors" for k in tensors}
    (tmpp / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))

    backing = ExpertBackingStore(tmpp)
    cache = ExpertLRUCache(budget_bytes, 1 << 12, num_layers=2)

    def mk_lin(layer, proj):
        return StreamingQuantizedSwitchLinear(
            layer_idx=layer,
            proj_name=proj,
            stacked_weight_key=f"model.layers.{layer}.mlp.switch_mlp.{proj}.weight",
            stacked_scales_key=f"model.layers.{layer}.mlp.switch_mlp.{proj}.scales",
            stacked_biases_key=f"model.layers.{layer}.mlp.switch_mlp.{proj}.biases",
            num_experts=4,
            input_dims=16,
            output_dims=8,
            backing=backing,
            cache=cache,
        )

    linears = {0: [mk_lin(0, p) for p in ("gate_proj", "up_proj", "down_proj")]}
    linears[1] = [mk_lin(1, "gate_proj")]
    return tmp, warmer, backing, linears


def test_readahead_warmer_advises_contiguous_runs():
    """advise-only warmer: contiguous expert ids collapse into one F_RDADVISE
    run per projection key, and no discarded reads happen (Fase G)."""
    from unittest.mock import patch

    from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore

    tmp, warmer, backing, linears = _ra_warmer_setup()
    try:
        advised = []
        w = warmer.PageCacheWarmer(linears, advise_only=True)
        hook = warmer.WarmPinHook(w, None)
        hook.on_layer_plan(1, [2], 8)
        hook.on_layer_plan(0, [1, 2, 3], 8)

        import time

        # The submit is async: the spy must be in place before the trigger.
        with patch.object(
            ExpertBackingStore,
            "advise_expert_run",
            side_effect=lambda key, first, count: advised.append((key, first, count)) or True,
        ):
            hook.on_layer_start(0, 8)  # fires advises for layer 1's [2]
            deadline = time.time() + 5
            while time.time() < deadline and not advised:
                time.sleep(0.05)

        assert advised, "advise jobs were not submitted"
        # layer 1 has one linear (gate_proj family: weight+scales+biases keys),
        # ids [2] -> single (2, 1) run per key.
        assert all(count == 1 and first == 2 for _, first, count in advised)
        assert all("model.layers.1." in key for key, _, _ in advised)
        assert w.warmed == 0  # advise-only: no reads
        assert w.advised == len(advised)
    finally:
        backing.close()


def test_backing_advise_expert_run_real_range():
    """Real F_RDADVISE against a temp shard: True on macOS, and a missing key
    degrades to False instead of raising."""
    import sys

    from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore

    tmp, warmer, backing, linears = _ra_warmer_setup()
    try:
        key = "model.layers.0.mlp.switch_mlp.gate_proj.weight"
        got = backing.advise_expert_run(key, 1, 2)  # experts 1..2 contiguous
        if sys.platform == "darwin":
            ok, adv_bytes, adv_segs = got
            assert ok is True and adv_bytes > 0 and adv_segs == 1
        else:
            assert got == (False, 0, 0)
        assert backing.advise_expert_run("nope.missing.key", 0, 1) == (False, 0, 0)
    finally:
        backing.close()


def test_expert_lru_retain_hot():
    """retain_hot keeps only the (layer, expert) pairs given, rebuilding
    per-layer counts — the hotness seeder's cache-swap primitive."""
    from omlx.patches.expert_streaming.streaming_switch import ExpertLRUCache

    cache = ExpertLRUCache(1 << 20, 4096, num_layers=2)
    for eid in range(6):
        cache.put((0, eid, "w"), ("w", "s", None))
        cache.put((1, eid, "w"), ("w", "s", None))
    evicted = cache.retain_hot({(0, 1), (0, 2), (1, 5)})
    assert evicted == 9
    assert cache.size == 3
    assert (0, 1, "w") in cache
    assert (0, 2, "w") in cache
    assert (1, 5, "w") in cache
    assert cache._layer_counts == {0: 2, 1: 1}
    # Budget-0 caches are no-ops.
    empty = ExpertLRUCache(0, 4096, num_layers=2)
    assert empty.retain_hot({(0, 0)}) == 0


def test_prefill_hotness_recorder_seeds_lru():
    """Budget>0: the recorder swaps the cache to the prompt's hot set —
    non-hot entries evicted, missing hot bundles loaded from the backing."""
    tmp, warmer, backing, linears = _ra_warmer_setup(budget_bytes=4 << 20)
    try:
        cache = linears[0][0].cache
        wkey0 = linears[0][0].stacked_weight_key
        wkey1 = linears[1][0].stacked_weight_key
        # Pre-fill with non-hot entries (what the last prefill chunks left).
        cache.put((0, 0, wkey0), ("w", "s", None))
        cache.put((1, 0, wkey1), ("w", "s", None))

        rec = warmer.PrefillHotnessRecorder(linears, backing, cache, per_expert_bytes=0)
        rec.on_layer_plan(0, [1, 2, 0, 3], 4096)  # prefill-sized: all experts seen
        rec.on_layer_plan(1, [3, 2], 4096)
        assert rec.saw_prefill is True
        assert rec.seeded is False

        # Decode-sized call triggers the one-shot seed.
        rec.maybe_seed(0, 8)
        assert rec.seeded is True
        assert rec.seeded_experts >= 0
        # The main-lineage seed warms the LRU on a worker thread; wait for
        # it before asserting cache contents (seed_done is the contract).
        assert rec.seed_done.wait(timeout=10)
        assert rec.seeded_experts > 0

        # Non-hot entries evicted (layer 0 expert 0 was seen but 1,2 rank higher
        # only if counts differ — with equal counts most_common is stable, so
        # assert the hot bundles are present and junk-only layers are gone).
        assert (1, 0, wkey1) not in cache
        assert (1, 3, wkey1) in cache
        assert cache.get((1, 3, wkey1)) is not None

        # One-shot: a second decode call does not re-seed.
        seeded_experts = rec.seeded_experts
        rec.maybe_seed(1, 8)
        assert rec.seeded_experts == seeded_experts
    finally:
        backing.close()


def test_prefill_hotness_recorder_budget0_page_cache_seed():
    """Budget-0: the seed issues bounded discarded reads (page-cache only),
    async on the warm pool, and never touches the LRU."""
    from unittest.mock import patch

    from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore

    tmp, warmer, backing, linears = _ra_warmer_setup(budget_bytes=0)
    try:
        cache = linears[0][0].cache
        assert cache.capacity == 0
        rec = warmer.PrefillHotnessRecorder(linears, backing, cache, per_expert_bytes=0)
        rec.on_layer_plan(0, [0, 1, 2, 3], 4096)
        rec.on_layer_plan(1, [1, 2], 4096)

        reads = []
        import time

        with patch.object(
            ExpertBackingStore,
            "load_expert_slice",
            side_effect=lambda key, eid: reads.append((key, eid)) or b"\0" * 8,
        ), patch.object(
            ExpertBackingStore,
            "load_expert_run",
            side_effect=lambda key, first, count: reads.extend(
                (key, e) for e in range(first, first + count)
            )
            or [b"\0" * 8] * count,
        ):
            rec.maybe_seed(0, 8)
            deadline = time.time() + 5
            while time.time() < deadline and not reads:
                time.sleep(0.05)

        assert reads, "page-cache seed burst was not submitted"
        assert rec.seeded is True
        assert cache.size == 0  # budget 0: nothing enters the LRU
    finally:
        backing.close()


def test_prefill_hotness_recorder_ignores_decode_only():
    """Without any prefill-sized call, the seed never fires (cheap no-op on
    every decode call — saw_prefill stays False)."""
    tmp, warmer, backing, linears = _ra_warmer_setup(budget_bytes=0)
    try:
        rec = warmer.PrefillHotnessRecorder(linears, backing, None)
        rec.on_layer_plan(0, [0, 1], 8)  # decode-sized rows
        rec.maybe_seed(0, 8)
        rec.maybe_seed(1, 8)
        assert rec.saw_prefill is False
        assert rec.seeded is False
        assert rec.seeded_experts == 0
    finally:
        backing.close()


def test_pin_controller_pins_within_budget():
    from omlx.patches.expert_streaming import warmer
    from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        tensors = {}
        for layer in range(2):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                tensors[f"model.layers.{layer}.mlp.switch_mlp.{proj}.weight"] = ((4, 8, 16), "BF16")
                tensors[f"model.layers.{layer}.mlp.switch_mlp.{proj}.scales"] = ((4, 8, 1), "U32")
                tensors[f"model.layers.{layer}.mlp.switch_mlp.{proj}.biases"] = ((4, 8, 1), "U32")
        _write_shard(tmp / "model.safetensors", tensors)
        (tmp / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {k: "model.safetensors" for k in tensors}})
        )
        backing = ExpertBackingStore(tmp)

        def _stub_lin(layer, proj):
            from types import SimpleNamespace

            return SimpleNamespace(
                stacked_weight_key=f"model.layers.{layer}.mlp.switch_mlp.{proj}.weight",
                stacked_scales_key=f"model.layers.{layer}.mlp.switch_mlp.{proj}.scales",
                stacked_biases_key=f"model.layers.{layer}.mlp.switch_mlp.{proj}.biases",
                backing=backing,
            )

        pc = warmer.PinController(
            {0: [_stub_lin(0, p) for p in ("gate_proj", "up_proj", "down_proj")],
             1: [_stub_lin(1, p) for p in ("gate_proj", "up_proj", "down_proj")]},
            backing,
            budget_bytes=1 << 20,
            observe_calls=2,
            per_expert_bytes=(8 * 16 * 2) + (8 * 4) + (8 * 4),
        )
        for token in range(4):
            for layer in range(2):
                pc.on_layer_plan(layer, [token % 4, (token + 1) % 4], 8)
        # pin pass scheduled after 2*2 layer reports
        import time

        deadline = time.time() + 5
        while time.time() < deadline and not pc.pinned:
            time.sleep(0.05)
        assert pc.pinned
        deadline = time.time() + 5
        while time.time() < deadline and backing.pinned_count == 0:
            time.sleep(0.05)
        assert backing.pinned_count > 0
        assert 0 < backing.pinned_bytes <= (1 << 20) + (1 << 12)


class _GuardBackingStub:
    """Minimal backing exposing only streaming_guard_info for the scheduler."""

    def __init__(self, num_moe_layers, experts_per_layer, per_expert_bytes):
        self.streaming_guard_info = {
            "num_moe_layers": num_moe_layers,
            "experts_per_layer": experts_per_layer,
            "per_expert_bytes": per_expert_bytes,
        }


def _guard_scheduler(model):
    """A Scheduler via __new__ with only the transient-model attrs wired."""
    from omlx.scheduler import Scheduler

    sched = Scheduler.__new__(Scheduler)
    sched.model = model
    sched._prefill_transient_tracker = None
    sched.memory_monitor = None
    return sched


def test_streaming_bank_bytes_bound_and_saturation():
    back = _GuardBackingStub(48, 512, 2_500_000)
    model = MagicMock()
    model._expert_streaming_backing = back
    sched = _guard_scheduler(model)
    ratio = sched._STREAMING_BANK_TOKEN_RATIO

    # Small chunk: banks scale with tokens (uniq ~ ratio * n per layer).
    small = sched._streaming_bank_bytes(100)
    assert small == 48 * int(ratio * 100) * 2_500_000
    # Saturation at experts_per_layer.
    big = sched._streaming_bank_bytes(10_000)
    assert big == 48 * 512 * 2_500_000
    # Monotone in between.
    assert small < sched._streaming_bank_bytes(413) < big


def test_streaming_bank_bytes_absent_without_backing():
    model = MagicMock()
    del model._expert_streaming_backing  # MagicMock: raise AttributeError
    sched = _guard_scheduler(model)
    assert sched._streaming_bank_bytes(413) == 0
    assert sched._predicted_chunk_transient(413, 0) == 0.0


def test_predicted_chunk_transient_includes_bank_term():
    back = _GuardBackingStub(48, 512, 2_500_000)
    model = MagicMock()
    model._expert_streaming_backing = back
    sched = _guard_scheduler(model)

    pred = sched._predicted_chunk_transient(413, 0)
    uniq = min(512, int(sched._STREAMING_BANK_TOKEN_RATIO * 413))
    expected = 48 * uniq * 2_500_000 * sched._PREFILL_TRANSIENT_SAFETY
    assert pred == pytest.approx(expected)


def test_admission_floor_scales_down_for_streaming_chunks():
    """A big chunk's measured transient must not floor-limit smaller chunks
    when streaming banks dominate (they scale ~linearly with tokens)."""
    from types import SimpleNamespace

    from omlx.scheduler import Scheduler
    from omlx.prefill_transient_tracker import PrefillTransientTracker

    back = _GuardBackingStub(48, 512, 2_500_000)
    model = MagicMock()
    model._expert_streaming_backing = back
    sched = _guard_scheduler(model)
    tracker = PrefillTransientTracker()
    tracker.update(1897, int(17_385_500))  # measured 17.4MB/token at 1.9k chunk
    tracker._observed_max_bytes = float(1897 * 17_385_500)  # ~32GB
    tracker._last_n_tokens = 1897
    sched._prefill_transient_tracker = tracker

    n_small = 512
    bound = Scheduler._admission_transient_bound(sched, n_small, 0)
    # The floor is discounted linearly (never the raw ~32GB)...
    floor = tracker.observed_max_bytes * n_small / 1897
    assert floor < tracker.observed_max_bytes
    # ...so the bound is the bank-aware predicted term, which itself stays
    # below the raw observed max.
    uniq = min(512, int(sched._STREAMING_BANK_TOKEN_RATIO * n_small))
    expected_pred = 48 * uniq * 2_500_000 * sched._PREFILL_TRANSIENT_SAFETY
    assert bound == pytest.approx(max(expected_pred, floor))
    assert bound < tracker.observed_max_bytes


def test_admission_floor_kept_without_streaming():
    """Non-streaming keeps the conservative size-invariant observed_max floor."""
    from omlx.scheduler import Scheduler
    from omlx.prefill_transient_tracker import PrefillTransientTracker

    sched = _guard_scheduler(MagicMock())
    del sched.model._expert_streaming_backing
    tracker = PrefillTransientTracker()
    tracker.update(1897, int(17_385_500))
    tracker._observed_max_bytes = 32.0 * 1024**3
    tracker._last_n_tokens = 1897
    sched._prefill_transient_tracker = tracker

    bound = Scheduler._admission_transient_bound(sched, 512, 0)
    assert bound == pytest.approx(32.0 * 1024**3)


# ---------------------------------------------------------------------------
# Fase H: per-model IO overrides (autotune)
# ---------------------------------------------------------------------------


def test_io_overrides_resolution_and_clamping():
    """Settings resolution: unset keys stay None, depth clamps to 1..64."""
    from omlx.model_settings import ModelSettings
    from omlx.patches.expert_streaming import _io_overrides

    # All defaults → every override None
    ov = _io_overrides(ModelSettings())
    assert ov == {
        "expert_streaming_io_depth": None,
        "expert_streaming_coalesce": None,
        "expert_streaming_readahead": None,
        "expert_streaming_seed": None,
        "expert_streaming_pilot": None,
        "expert_streaming_per_layer_eval": None,
        "expert_streaming_pins": None,
        "expert_streaming_hot_fraction": None,
        "expert_streaming_pin_gib": None,
        "expert_streaming_pin_sync": None,
        "expert_streaming_pin_regime": None,
        "expert_streaming_cache_policy": None,
        "expert_streaming_dynamic": None,
        "expert_streaming_dynamic_max_gib": None,
    }

    # Depth clamping and boolean pass-through; the new UX knobs normalize:
    # cache policy lowercases, governor ceiling clamps to the 0..64 window.
    ov = _io_overrides(
        ModelSettings(
            expert_streaming_io_depth=999,
            expert_streaming_coalesce=False,
            expert_streaming_readahead=False,
            expert_streaming_seed=False,
            expert_streaming_pilot=True,
            expert_streaming_per_layer_eval=False,
            expert_streaming_pins=True,
            expert_streaming_pin_gib=3.0,
            expert_streaming_cache_policy="S3FIFO",
            expert_streaming_dynamic=True,
            expert_streaming_dynamic_max_gib=8.0,
        )
    )
    assert ov["expert_streaming_io_depth"] == 64
    assert ov["expert_streaming_coalesce"] is False
    assert ov["expert_streaming_readahead"] is False
    assert ov["expert_streaming_seed"] is False
    assert ov["expert_streaming_pilot"] is True
    assert ov["expert_streaming_per_layer_eval"] is False
    assert ov["expert_streaming_pins"] is True
    assert ov["expert_streaming_pin_gib"] == 3.0
    assert ov["expert_streaming_cache_policy"] == "s3fifo"
    assert ov["expert_streaming_dynamic"] is True
    assert ov["expert_streaming_dynamic_max_gib"] == 8.0

    # Invalid depth → None (env default); None settings object → all None.
    # Invalid policy and out-of-range ceiling fall back the same way.
    ov = _io_overrides(ModelSettings(expert_streaming_io_depth=0))
    assert ov["expert_streaming_io_depth"] is None
    assert _io_overrides(None)["expert_streaming_io_depth"] is None
    ov = _io_overrides(
        ModelSettings(
            expert_streaming_cache_policy="arc",
            expert_streaming_dynamic_max_gib=100.0,
        )
    )
    assert ov["expert_streaming_cache_policy"] is None
    assert ov["expert_streaming_dynamic_max_gib"] is None


def test_convert_applies_io_overrides():
    """convert_model_to_streaming wires per-model io_depth/coalesce to linears."""
    from omlx.model_settings import ModelSettings  # noqa: F811
    from omlx.patches.expert_streaming import convert_model_to_streaming
    from omlx.patches.expert_streaming.streaming_switch import (
        StreamingSwitchLinear,
        io_pool_for,
    )

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        num_layers, experts, hidden, moe_hidden, stages = 2, 4, 8, 4, 1
        _write_fake_dsv4_checkpoint(
            tmp,
            num_layers=num_layers,
            experts=experts,
            hidden=hidden,
            moe_hidden=moe_hidden,
            mtp_stages=stages,
        )
        layers = [_FakeLayer(experts, moe_hidden, hidden) for _ in range(num_layers)]
        mtp = [_FakeDSparkStage(experts, moe_hidden, hidden) for _ in range(stages)]
        model = _FakeTextModel(
            layers,
            mtp,
            {
                "model_type": "deepseek_v4",
                "hidden_size": hidden,
                "moe_intermediate_size": moe_hidden,
            },
        )
        convert_model_to_streaming(
            model,
            str(tmp),
            ModelSettings(expert_streaming_io_depth=8, expert_streaming_coalesce=False),
            use_file_backing=False,
        )

        pool = io_pool_for(8)
        assert pool._max_workers == 8
        # Registry dedup: same depth → same executor instance
        assert io_pool_for(8) is pool
        assert io_pool_for(None)._max_workers >= 1

        for lyr in layers:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                lin = getattr(lyr.ffn.switch_mlp, proj)
                assert isinstance(lin, StreamingSwitchLinear)
                assert lin._io_pool_override is pool
                assert lin._coalesce_override is False
        for stage in mtp:
            lin = stage.ffn.switch_mlp.down_proj
            assert lin._io_pool_override is pool
            assert lin._coalesce_override is False


def test_convert_without_io_overrides_keeps_defaults():
    """No settings overrides → linears keep the env-default IO behavior."""
    from omlx.model_settings import ModelSettings
    from omlx.patches.expert_streaming import convert_model_to_streaming

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _write_fake_dsv4_checkpoint(
            tmp, num_layers=2, experts=4, hidden=8, moe_hidden=4, mtp_stages=0
        )
        layers = [_FakeLayer(4, 4, 8) for _ in range(2)]
        model = _FakeTextModel(
            layers,
            [],
            {
                "model_type": "deepseek_v4",
                "hidden_size": 8,
                "moe_intermediate_size": 4,
            },
        )
        convert_model_to_streaming(model, str(tmp), ModelSettings(), use_file_backing=False)
        for lyr in layers:
            lin = lyr.ffn.switch_mlp.down_proj
            assert lin._io_pool_override is None
            assert lin._coalesce_override is None


def test_io_pool_for_clamps_and_caches():
    """io_pool_for clamps invalid depths and caches one executor per depth."""
    from omlx.patches.expert_streaming.streaming_switch import io_pool_for

    fallback = io_pool_for(None)
    assert io_pool_for("bogus") is fallback
    assert io_pool_for(-3) is fallback  # invalid depth → env-default pool
    pool = io_pool_for(3)
    assert pool._max_workers == 3
    assert io_pool_for(3) is pool
    assert io_pool_for("5") is io_pool_for(5)
    assert io_pool_for(999)._max_workers == 64


# --- Qwen per-layer eval boundary (G4 / qwen35_stream_eval) -----------------


def _stream_eval_recorder(monkeypatch):
    """Swap the module's mx for a recorder so the wrapper's eval/clear calls
    are observable without touching the real allocator."""
    import types

    from omlx.patches.expert_streaming import qwen35_stream_eval as qse

    calls = {"eval": 0, "clear": 0}

    def _bump(key):
        def _f(*args, **kwargs):
            calls[key] += 1

        return _f

    monkeypatch.setattr(
        qse, "mx", types.SimpleNamespace(eval=_bump("eval"), clear_cache=_bump("clear"))
    )
    # The clear routes through omlx.utils.metal_sync (Fase J Etapa C/D — a
    # pool clear must never run unsynchronized), so that helper is patched
    # too. Whichever route the wrapper takes is counted exactly once.
    monkeypatch.setattr(
        "omlx.utils.metal_sync._sync_and_clear_cache", _bump("clear")
    )
    return qse, calls


def test_qwen_stream_eval_fires_on_prefill(monkeypatch):
    import numpy as np

    qse, calls = _stream_eval_recorder(monkeypatch)
    assert qse.configure_from_settings(True) is True

    class Layer:
        _stream_eval = True

    wrapped = qse._wrap_call(lambda self, x, *a, **k: x + 1)
    out = wrapped(Layer(), np.zeros((1, 8, 4), dtype=np.float32))
    assert calls == {"eval": 1, "clear": 1}
    # The passthrough result is untouched.
    assert out.shape == (1, 8, 4)


def test_qwen_stream_eval_skips_decode_verify_and_gating(monkeypatch):
    import numpy as np

    qse, calls = _stream_eval_recorder(monkeypatch)

    class Layer:
        _stream_eval = True

    wrapped = qse._wrap_call(lambda self, x, *a, **k: x + 1)
    qse.configure_from_settings(True)

    # Decode shape [1, 1, H]: forced syncs/token would erode the QD16 win.
    wrapped(Layer(), np.zeros((1, 1, 4), dtype=np.float32))
    assert calls == {"eval": 0, "clear": 0}

    # MTP verify passes stay lazy for the same reason.
    wrapped(Layer(), np.zeros((1, 6, 4), dtype=np.float32), target_verify=True)
    assert calls == {"eval": 0, "clear": 0}

    # A layer without the converter's _stream_eval flag is untouched
    # (non-streaming loads never see the boundary).
    class Plain:
        pass

    wrapped(Plain(), np.zeros((1, 6, 4), dtype=np.float32))
    assert calls == {"eval": 0, "clear": 0}

    # Knob off disables the boundary entirely.
    assert qse.configure_from_settings(False) is False
    wrapped(Layer(), np.zeros((1, 6, 4), dtype=np.float32))
    assert calls == {"eval": 0, "clear": 0}


def test_qwen_stream_eval_configure_none_restores_env_default(monkeypatch):
    from omlx.patches.expert_streaming import qwen35_stream_eval as qse

    # The env default is captured at import; simulate both captured values and
    # confirm None defers to them while explicit values override.
    monkeypatch.setattr(qse, "_PER_LAYER_EVAL_DEFAULT", False)
    assert qse.configure_from_settings(None) is False
    monkeypatch.setattr(qse, "_PER_LAYER_EVAL_DEFAULT", True)
    assert qse.configure_from_settings(None) is True
    assert qse.configure_from_settings(False) is False
    assert qse.configure_from_settings(True) is True
    qse.configure_from_settings(None)


def test_qwen_stream_eval_patch_idempotent():
    try:
        from mlx_vlm.models.qwen3_5_moe import language as q35
    except ImportError:
        pytest.skip("mlx_vlm not installed")
    from omlx.patches.expert_streaming import qwen35_stream_eval as qse

    cls = q35.Qwen3_5MoeDecoderLayer
    orig_call = cls.__call__
    try:
        assert qse.apply_qwen35_moe_stream_eval() is True
        wrapped_once = cls.__call__
        # Second apply must not stack a second wrapper.
        assert qse.apply_qwen35_moe_stream_eval() is True
        assert cls.__call__ is wrapped_once
        assert getattr(cls, "_omlx_stream_eval_wrapped", False) is True
    finally:
        cls.__call__ = orig_call
        if hasattr(cls, "_omlx_stream_eval_wrapped"):
            del cls._omlx_stream_eval_wrapped


def test_expert_streaming_per_layer_eval_round_trip():
    s = ModelSettings(expert_streaming_per_layer_eval=False)
    d = s.to_dict()
    assert d["expert_streaming_per_layer_eval"] is False
    assert ModelSettings.from_dict(d).expert_streaming_per_layer_eval is False
    assert ModelSettings().expert_streaming_per_layer_eval is None


def test_expert_streaming_per_layer_eval_excluded_from_profiles():
    from omlx.model_profiles import EXCLUDED_FROM_PROFILES

    assert "expert_streaming_per_layer_eval" in EXCLUDED_FROM_PROFILES


@pytest.mark.asyncio
async def test_expert_streaming_per_layer_eval_persists_via_api():
    pool, entry = _failed_pool()
    entry.config_model_type = "qwen4_exp"
    settings = ModelSettings()
    await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(expert_streaming_per_layer_eval=False),
    )
    assert settings.expert_streaming_per_layer_eval is False


# --- Learned pin store server integration (I2) -------------------------------


def test_pin_controller_profile_round_trip(tmp_path):
    """An explicitly wired per-model profile path persists observed frequencies
    and a fresh controller wires the learned hot set from token 1."""
    from collections import Counter
    from unittest.mock import MagicMock

    from omlx.patches.expert_streaming import warmer as warmer_mod

    profile = str(tmp_path / ".omlx" / "expert_pin_profile.json")
    linears = {0: [MagicMock()], 1: [MagicMock()]}
    ctl = warmer_mod.PinController(
        linears,
        MagicMock(),
        per_expert_bytes=1024,
        profile_path=profile,
    )
    ctl.freq = {0: Counter({3: 9, 5: 2}), 1: Counter({7: 4})}
    ctl.save_profile()

    import json
    import os

    assert os.path.exists(profile)
    data = json.loads(open(profile).read())
    assert data["freq"]["0"] == [[3, 9], [5, 2]]

    ctl2 = warmer_mod.PinController(
        linears,
        MagicMock(),
        per_expert_bytes=1024,
        profile_path=profile,
    )
    assert ctl2.freq == {0: Counter({3: 9, 5: 2}), 1: Counter({7: 4})}
    # Learned hot set available: pinned immediately, no observation window.
    assert ctl2.pinned is True
    # Env path (explicit opt-in) wins over the per-model derived path.
    assert warmer_mod.PinController(
        linears, MagicMock(), profile_path=profile
    ).profile_path == profile or warmer_mod.PIN_PROFILE_PATH == ""


def test_expert_streaming_pins_io_overrides():
    from omlx.model_settings import ModelSettings
    from omlx.patches.expert_streaming import _io_overrides

    ov = _io_overrides(
        ModelSettings(expert_streaming_pins=True, expert_streaming_pin_gib=3.0)
    )
    assert ov["expert_streaming_pins"] is True
    assert ov["expert_streaming_pin_gib"] == 3.0
    ov = _io_overrides(ModelSettings())
    assert ov["expert_streaming_pins"] is None
    assert ov["expert_streaming_pin_gib"] is None


@pytest.mark.asyncio
async def test_expert_streaming_pins_persist_via_api():
    pool, entry = _failed_pool()
    entry.config_model_type = "qwen4_exp"
    settings = ModelSettings()
    await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(
            expert_streaming_pins=True, expert_streaming_pin_gib=2.5
        ),
    )
    assert settings.expert_streaming_pins is True
    assert settings.expert_streaming_pin_gib == 2.5


@pytest.mark.asyncio
async def test_expert_streaming_pin_sync_regime_persist_via_api():
    """Fase M1: the two pin knobs were reachable only through the bench
    harness — the patch model did not declare them, so the webUI could not
    send them at all."""
    pool, entry = _failed_pool()
    entry.config_model_type = "qwen4_exp"
    settings = ModelSettings()
    await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(
            expert_streaming_pin_sync=True, expert_streaming_pin_regime="prefill"
        ),
    )
    assert settings.expert_streaming_pin_sync is True
    assert settings.expert_streaming_pin_regime == "prefill"

    # Clearing sends explicit nulls, not absent keys.
    await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(
            expert_streaming_pin_sync=None, expert_streaming_pin_regime=None
        ),
    )
    assert settings.expert_streaming_pin_sync is None
    assert settings.expert_streaming_pin_regime is None


@pytest.mark.asyncio
async def test_expert_streaming_pin_regime_validation():
    pool, _ = _failed_pool()
    settings = ModelSettings()
    with pytest.raises(admin_routes.HTTPException, match="'decode' or 'prefill'"):
        await _update_settings(
            pool,
            settings,
            admin_routes.ModelSettingsRequest(expert_streaming_pin_regime="nope"),
        )


@pytest.mark.asyncio
async def test_expert_streaming_pin_gib_validation():
    pool, _ = _failed_pool()
    settings = ModelSettings()
    with pytest.raises(admin_routes.HTTPException, match="between 0 and 64"):
        await _update_settings(
            pool,
            settings,
            admin_routes.ModelSettingsRequest(expert_streaming_pin_gib=99),
        )


def test_expert_streaming_pins_excluded_from_profiles():
    from omlx.model_profiles import EXCLUDED_FROM_PROFILES

    assert "expert_streaming_pins" in EXCLUDED_FROM_PROFILES
    assert "expert_streaming_pin_gib" in EXCLUDED_FROM_PROFILES
    # Fase M1 knobs. Left unclassified these leak through profiles and
    # templates, and they also break the completeness test in
    # tests/test_admin_profiles_api.py (every ModelSettings field must be
    # classified in exactly one of the three sets).
    assert "expert_streaming_pin_sync" in EXCLUDED_FROM_PROFILES
    assert "expert_streaming_pin_regime" in EXCLUDED_FROM_PROFILES


def test_save_expert_pin_profile_hook():
    """The engine stop() helper saves via the backing's attached controller
    and never raises, even when nothing is attached."""
    from omlx.patches.expert_streaming import save_expert_pin_profile

    engine = MagicMock()
    backing = MagicMock()
    engine._expert_streaming_backing = backing
    save_expert_pin_profile(engine)  # no controller attached -> no-op
    backing._pin_controller = None
    save_expert_pin_profile(engine)
    ctl = MagicMock()
    backing._pin_controller = ctl
    save_expert_pin_profile(engine)
    ctl.save_profile.assert_called_once()

    # Nothing attached anywhere: silent no-op (and the model-side holder path).
    bare = MagicMock(spec=object)
    save_expert_pin_profile(bare)

class _TieredDictBacking:
    """HOBBIT test backing: `load_expert` serves the source (4-bit) bank for
    hot experts and a requantized (3-bit) bank for the rest, keyed by the
    same (layer, proj) layout the dict fallback reads."""

    def __init__(self, e: int, hot: set, out_d: int, in_d: int, seed: int = 3):
        import mlx.core as mx
        import numpy as np
        mx.random.seed(seed)
        dense = mx.random.normal((e, out_d, in_d))
        w4, s4, b4 = mx.quantize(dense, group_size=32, bits=4)
        w3, s3, b3 = mx.quantize(dense, group_size=32, bits=3)
        self.banks = {
            "weight": {"hot": w4, "cold": w3},
            "scales": {"hot": s4, "cold": s3},
            "biases": {"hot": b4, "cold": b3},
        }
        self.hot = {int(h) for h in hot}

    def load_expert(self, key: str, expert_id: int):
        part = key.rsplit(".", 1)[1] if "." in key else key
        tier = "hot" if int(expert_id) in self.hot else "cold"
        bank = self.banks[part][tier]
        w = bank[int(expert_id)]
        return w


def test_hobbit_split_matches_tiered_reference():
    """Dual-tier gather (I6): the split output must equal composing each
    position's expert from ITS OWN tier's packing — hot positions at 4-bit
    source values, cold positions at the 3-bit tier values."""
    import mlx.core as mx
    import numpy as np

    from omlx.patches.expert_streaming.streaming_switch import (
        ExpertLRUCache,
        StreamingQuantizedSwitchLinear,
    )

    E, D_in, D_out = 8, 64, 32
    hot = {1, 3}
    backing = _TieredDictBacking(E, hot, D_out, D_in)

    lin = StreamingQuantizedSwitchLinear(
        layer_idx=0,
        proj_name="down_proj",
        stacked_weight_key="k.down_proj.weight",
        stacked_scales_key="k.down_proj.scales",
        stacked_biases_key="k.down_proj.biases",
        num_experts=E,
        input_dims=D_in,
        output_dims=D_out,
        backing=backing,
        cache=ExpertLRUCache(1 << 26, 1 << 18, num_layers=1),
        group_size=32,
        bits=4,  # source (hot) packing
        mode="affine",
    )
    lin.set_hobbit_split(hot, cold_bits=3, cold_gs=32)

    rng = np.random.default_rng(11)
    T = 40
    # Linear contract (post _gather_sort): one x row per route, rhs flat ids.
    # x_exp is [N, 1, D] where N = len(indices.flat); rhs is [N].
    x_rows = rng.standard_normal((T, D_in)).astype(np.float32)
    x_rows_mx = mx.array(x_rows)
    flat_ids = rng.integers(0, E, size=T).astype(np.int32)
    x = mx.array(x_rows[:, None, :])  # [T, 1, D_in]
    idx = mx.array(flat_ids)  # [T]

    out = lin(x, idx)
    assert out.shape == (T, 1, D_out), out.shape

    # Reference: per position, dequantize that expert's tier and matmul.
    banks = backing.banks
    outs = np.zeros((T, 1, D_out), dtype=np.float32)
    for t in range(T):
        e = int(flat_ids[t])
        tier = "hot" if e in hot else "cold"
        w = banks["weight"][tier][e]
        s = banks["scales"][tier][e]
        b = banks["biases"][tier][e]
        dense = mx.dequantize(w, s, b, group_size=32, bits=4 if tier == "hot" else 3)
        r = x_rows_mx[t] @ dense.T
        mx.eval(r)
        outs[t, 0] = np.array(r).astype(np.float32)
    diff = np.abs(np.array(out.astype(mx.float32)) - outs).max()
    assert diff < 1e-3, f"split output mismatch: {diff}"

    # All-hot and all-cold degenerate arms must equal the uniform banks.
    # Each arm gets its OWN backing tier set — the runtime contract is that
    # the backing and the linear share one hot set (installed by the convert).
    all_hot_backing = _TieredDictBacking(E, set(range(E)), D_out, D_in)
    lin_all_hot = StreamingQuantizedSwitchLinear(
        layer_idx=1, proj_name="down_proj",
        stacked_weight_key="k.down_proj.weight",
        stacked_scales_key="k.down_proj.scales",
        stacked_biases_key="k.down_proj.biases",
        num_experts=E, input_dims=D_in, output_dims=D_out,
        backing=all_hot_backing, cache=ExpertLRUCache(1 << 26, 1 << 18, num_layers=1),
        group_size=32, bits=4, mode="affine",
    )
    lin_all_hot.set_hobbit_split(set(range(E)), cold_bits=3, cold_gs=32)
    out_all_hot = lin_all_hot(x, idx)
    ref4 = mx.gather_qmm(
        x, all_hot_backing.banks["weight"]["hot"], all_hot_backing.banks["scales"]["hot"], all_hot_backing.banks["biases"]["hot"],
        rhs_indices=idx.astype(mx.uint32), transpose=True, group_size=32,
        bits=4, mode="affine", sorted_indices=False,
    )
    assert mx.allclose(out_all_hot, ref4, atol=1e-3).item()

    # Uniform-cold arm (I5 contract): an empty hot set means split INACTIVE
    # and the converter overrides bits to the tier packing — here modeled by
    # a bits=3 linear reading the all-cold backing.
    all_cold_backing = _TieredDictBacking(E, set(), D_out, D_in)
    lin_all_cold = StreamingQuantizedSwitchLinear(
        layer_idx=2, proj_name="down_proj",
        stacked_weight_key="k.down_proj.weight",
        stacked_scales_key="k.down_proj.scales",
        stacked_biases_key="k.down_proj.biases",
        num_experts=E, input_dims=D_in, output_dims=D_out,
        backing=all_cold_backing, cache=ExpertLRUCache(1 << 26, 1 << 18, num_layers=1),
        group_size=32, bits=3, mode="affine",
    )
    out_all_cold = lin_all_cold(x, idx)
    ref3 = mx.gather_qmm(
        x, all_cold_backing.banks["weight"]["cold"], all_cold_backing.banks["scales"]["cold"], all_cold_backing.banks["biases"]["cold"],
        rhs_indices=idx.astype(mx.uint32), transpose=True, group_size=32,
        bits=3, mode="affine", sorted_indices=False,
    )
    assert mx.allclose(out_all_cold, ref3, atol=1e-3).item()


# --- Fase I6: hotness signal, profile cap, and fraction denominator ------------


def _i6_tmp_profile(pairs, layer="0"):
    import json
    import os
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump({"freq": {layer: pairs}}, f)
    return path


def test_pin_controller_counts_per_token_usage():
    """Fase I6: the per-token bincount payload makes the learned profile rank
    experts by USE — id 1 (5 of 6 routed positions) dominates; the old
    presence signal counted each uniq expert exactly once and flattened
    every frequency (the 42x64 all-count-4 profile)."""
    from collections import Counter

    import numpy as np

    from omlx.patches.expert_streaming import warmer

    flat = np.array([1, 1, 4, 1, 1, 1], dtype=np.int32)
    pc = warmer.PinController({0: []}, None, num_experts=8)
    pc.on_layer_plan(0, [1, 4], int(flat.size), np.bincount(flat, minlength=8))
    assert pc.freq[0] == Counter({1: 5, 4: 1})
    # Fase L: prefill-sized calls accumulate under the PREFILL regime only
    # (they never pin) — the decode regime stays untouched.
    pc.on_layer_plan(0, [1, 4], 4096, np.bincount(flat, minlength=8))
    assert pc.freq[0] == Counter({1: 5, 4: 1})
    assert pc.regimes["prefill"][0] == Counter({1: 5, 4: 1})
    assert pc.pinned is False

    # dict and list payloads coerce identically; oversized lists clamp to
    # the controller's expert width.
    pc2 = warmer.PinController({0: []}, None, num_experts=4)
    pc2.on_layer_plan(0, [0, 2], 8, {0: 3, 2: 1})
    assert pc2.freq[0] == Counter({0: 3, 2: 1})
    pc2.on_layer_plan(0, [1], 8, [5, 7, 0, 0, 9])
    # list clamped to the expert width (first 4): experts 0/1 gain 5/7.
    assert pc2.freq[0] == Counter({0: 8, 1: 7, 2: 1})
    # Unusable payloads fall back to the presence signal.
    pc3 = warmer.PinController({0: []}, None, num_experts=8)
    pc3.on_layer_plan(0, [2, 6], 8, "not-a-payload")
    assert pc3.freq[0] == Counter({2: 1, 6: 1})


def test_pin_controller_three_arg_fallback_compat():
    """Old wiring calls on_layer_plan(layer, uniq, positions) with no counts:
    the presence-based fallback must keep the pre-I6 behavior unchanged."""
    from collections import Counter

    from omlx.patches.expert_streaming import warmer

    pc = warmer.PinController({0: []}, None)
    pc.on_layer_plan(0, [3, 7], 8)
    pc.on_layer_plan(0, [3], 8)
    assert pc.freq[0] == Counter({3: 2, 7: 1})
    pc.on_layer_plan(0, [3, 7], 4096)  # prefill-sized: presence, no pin
    # Fase L: the prefill row lands in the prefill regime, not the decode
    # freq — regimes never overwrite each other.
    assert pc.freq[0] == Counter({3: 2, 7: 1})
    assert pc.regimes["prefill"][0] == Counter({3: 1, 7: 1})
    assert pc.pinned is False


def test_warm_pin_hook_forwards_counts_only_to_pin_and_recorder():
    """WarmPinHook repasse: counts reach the pinner and the recorder, never
    the readahead warmer (its contiguous-run F_RDADVISE grouping is
    set-based); a warmer-only hook asks for no histogram at all."""
    from unittest.mock import MagicMock

    from omlx.patches.expert_streaming import warmer

    counts = [1, 2, 0, 0]
    w = MagicMock(spec=["on_layer_plan", "on_layer_start"])
    p = MagicMock(spec=["on_layer_plan"])
    rec = MagicMock(spec=["on_layer_plan", "maybe_seed"])
    hook = warmer.WarmPinHook(w, p, rec)
    assert hook.wants_usage_counts is True
    hook.on_layer_plan(0, [0, 1], 8, counts)
    p.on_layer_plan.assert_called_once_with(0, [0, 1], 8, counts)
    rec.on_layer_plan.assert_called_once_with(0, [0, 1], 8, counts)
    # Warmer got the 3-arg contract: NO counts payload.
    w.on_layer_plan.assert_called_once_with(0, [0, 1], 8)
    hook2 = warmer.WarmPinHook(w, None, None)
    assert hook2.wants_usage_counts is False


def test_pin_profile_keep_cap_env_default_and_truncation(tmp_path, monkeypatch):
    """_PIN_PROFILE_KEEP defaults to 512 (OMLX_EXPERT_STREAMING_PIN_KEEP) and
    caps the per-layer record list in the saved JSON without changing the
    format — a 288-expert model must not be truncated to 64 entries."""
    import json

    from omlx.patches.expert_streaming import warmer as warmer_mod

    assert warmer_mod._PIN_PROFILE_KEEP == 512
    monkeypatch.setenv("OMLX_EXPERT_STREAMING_PIN_KEEP", "300")
    import importlib
    from collections import Counter

    warmer2 = importlib.reload(warmer_mod)
    try:
        assert warmer2._PIN_PROFILE_KEEP == 300
        profile = str(tmp_path / "expert_pin_profile.json")
        linears = {0: [MagicMock()], 1: [MagicMock()]}
        ctl = warmer2.PinController(
            linears,
            MagicMock(),
            per_expert_bytes=1024,
            profile_path=profile,
        )
        # 400 distinct experts observed: only the top 300 survive the cap.
        # Counts are 400-e (expert 0 hottest), so the top 300 are 0..299.
        ctl.freq = {0: Counter({e: 400 - e for e in range(400)})}
        ctl.save_profile()
        data = json.loads(open(profile).read())
        assert len(data["freq"]["0"]) == 300
        assert {e for e, _ in data["freq"]["0"]} == set(range(300))
    finally:
        monkeypatch.delenv("OMLX_EXPERT_STREAMING_PIN_KEEP")
        importlib.reload(warmer_mod)


def test_hot_set_loader_real_denominator():
    """Fase I6 fraction semantics: ceil(fraction * num_experts) hot experts
    (clamped to available records); without num_experts the record count
    stays the denominator so legacy 2-arg calls are unchanged."""
    from omlx.patches.expert_streaming.shard_bank import load_hot_set_from_profile

    # Full-width profile on a 288-expert model: the fraction's denominator
    # is the model width, not the record count.
    pairs = [[i, 288 - i] for i in range(288)]  # descending counts
    p = _i6_tmp_profile(pairs)
    p2 = _i6_tmp_profile([[5, 9], [2, 4]])
    try:
        hot = load_hot_set_from_profile(p, 0.25, num_experts=288)
        assert len(hot["layer_0"]) == 72  # ceil(0.25 * 288)
        assert hot["layer_0"] == {e for e, _ in pairs[:72]}
        # Legacy 2-arg (no num_experts): denominator is the record count.
        hot_legacy = load_hot_set_from_profile(p, 0.25)
        assert len(hot_legacy["layer_0"]) == 72  # ceil(0.25 * 288 records)
        # Truncated profile (64 records, the pre-I6 keep cap) on a wide
        # model: the real denominator would ask 72 but only 64 exist.
        p_trunc = _i6_tmp_profile(pairs[:64])
        hot_trunc = load_hot_set_from_profile(p_trunc, 0.25, num_experts=288)
        assert len(hot_trunc["layer_0"]) == 64  # clamped to available
        # Fraction beyond the records clamps to what was observed.
        hot_full = load_hot_set_from_profile(p, 0.99, num_experts=288)
        assert len(hot_full["layer_0"]) == 286  # ceil(0.99 * 288)
        # Sparse profile on a wide model: cannot elect unseen experts.
        hot_sparse = load_hot_set_from_profile(p2, 0.5, num_experts=288)
        assert hot_sparse["layer_0"] == {5, 2}
        import os

        os.unlink(p_trunc)
    finally:
        import os

        os.unlink(p)
        os.unlink(p2)



# ---------------------------------------------------------------------------
# Fase K F1/F2/F3 — O2 advisor key fix, speculation guard, stash ring
# ---------------------------------------------------------------------------

class _AdviseRecorderBacking:
    """Backing double for the O2 advisor: records advise_expert_run calls
    and serves fake runs/slices for the stash ring."""

    def __init__(self, per_expert_bytes: int = 64, num_experts: int = 64):
        self.advised: list[tuple[str, int, int]] = []
        self.read_runs: list[tuple[str, int, int]] = []
        self.per_expert_bytes = per_expert_bytes
        self.num_experts = num_experts
        self.reader = _FakeReaderShim(self)

    def advise_expert_run(self, key: str, first_id: int, count: int):
        self.advised.append((key, first_id, count))
        return (True, self.per_expert_bytes * count, 1)

    def _reader_for_key(self, key: str, expert_id: int | None = None):
        return self.reader

    def load_expert_run(self, key: str, first_id: int, count: int) -> list:
        import numpy as np

        self.read_runs.append((key, first_id, count))
        return [
            np.arange(self.per_expert_bytes, dtype=np.uint8).reshape(
                (self.per_expert_bytes,)
            )
            + (first_id + i)
            for i in range(count)
        ]

    def load_expert_slice(self, key: str, expert_id: int):
        import numpy as np

        return np.arange(self.per_expert_bytes, dtype=np.uint8)

    def tensor_dtype(self, key: str):
        return None


class _FakeReaderShim:
    def __init__(self, backing):
        self.backing = backing
        self.path = "/fake/shard.safetensors"

    def _rp_for(self, key: str):
        return None  # unused by the advisor


def _make_advise_linear(layer_idx: int, proj: str, backing, cache):
    from omlx.patches.expert_streaming.streaming_switch import (
        StreamingQuantizedSwitchLinear,
    )

    return StreamingQuantizedSwitchLinear(
        layer_idx=layer_idx,
        proj_name=proj,
        stacked_weight_key=f"model.layers.{layer_idx}.mlp.switch_mlp.{proj}.weight",
        stacked_scales_key=f"model.layers.{layer_idx}.mlp.switch_mlp.{proj}.scales",
        stacked_biases_key=None,
        num_experts=backing.num_experts,
        input_dims=32,
        output_dims=32,
        backing=backing,
        cache=cache,
    )


def _attach_spec(cache):
    """Fase K K1: attach a fresh per-conversion SpeculationState to a cache.

    Mirrors what convert_model_to_streaming does: one instance per
    conversion, hung off the cache (and the backing when it is an object).
    """
    from omlx.patches.expert_streaming import streaming_switch as ss

    cache.spec_state = ss.SpeculationState()
    return cache.spec_state


class TestFaseKO2Advisor:
    def test_advise_uses_next_layer_key(self):
        """Fase K F1: the advisor must F_RDADVISE the NEXT layer's banks.

        K1: the whole speculation state is per-conversion — the test
        attaches its own state and never touches module globals.
        """
        from omlx.patches.expert_streaming import streaming_switch as ss

        cache = ss.ExpertLRUCache(0, 4096, num_layers=2)
        spec = _attach_spec(cache)
        backing = _AdviseRecorderBacking()
        lin0 = _make_advise_linear(0, "gate_proj", backing, cache)
        lin1_up = _make_advise_linear(1, "up_proj", backing, cache)
        lin1_down = _make_advise_linear(1, "down_proj", backing, cache)
        spec.register_linears(1, [lin1_up, lin1_down])
        spec.prev_uniq_by_layer[1] = [3, 4, 9]
        with patch.object(ss, "_RA_ENV", True), patch.object(ss, "_STASH_ENV", False):
            plan = ss._RemapPlan()
            lin0._advise_next_layer_prev_token(plan)
        keys = {k for k, _, _ in backing.advised}
        assert keys == {
            "model.layers.1.mlp.switch_mlp.up_proj.weight",
            "model.layers.1.mlp.switch_mlp.down_proj.weight",
        }, f"advised wrong banks: {keys}"
        # 2 targets x 3 experts (runs (3,2) + (9,1))
        assert spec.stats["advised"] == 6
        # Fase 2 telemetry: runs/expert/bytes per advised run.
        assert spec.stats["advised_runs"] == 4, spec.stats
        assert spec.stats["advised_experts"] == 6
        assert spec.stats["advised_bytes"] == 2 * (64 * 2 + 64), spec.stats
        assert spec.stats["advice_tier_segments"] == 4
        assert spec.stats["advice_failures"] == 0

    def test_advise_guard_skips_prefill_shaped_sets(self):
        """Fase K F2: > _MAX_ADVISE_ROWS experts is prefill-shaped, skip."""
        from omlx.patches.expert_streaming import streaming_switch as ss

        cache = ss.ExpertLRUCache(0, 4096, num_layers=2)
        spec = _attach_spec(cache)
        backing = _AdviseRecorderBacking(num_experts=512)
        lin0 = _make_advise_linear(0, "gate_proj", backing, cache)
        lin1 = _make_advise_linear(1, "up_proj", backing, cache)
        spec.register_linears(1, [lin1])
        spec.prev_uniq_by_layer[1] = list(range(200))
        with patch.object(ss, "_RA_ENV", True):
            plan = ss._RemapPlan()
            lin0._advise_next_layer_prev_token(plan)
        assert backing.advised == [], "prefill-shaped set must not be advised"
        assert spec.stats["advised_runs"] == 0, "no advice stats for skipped plans"

    def test_advise_dedupes_runs_per_layer_call(self):
        """Fase K F2: the 3 projections share one plan -> no double advise."""
        from omlx.patches.expert_streaming import streaming_switch as ss

        cache = ss.ExpertLRUCache(0, 4096, num_layers=2)
        spec = _attach_spec(cache)
        backing = _AdviseRecorderBacking()
        lin0 = _make_advise_linear(0, "gate_proj", backing, cache)
        lin1 = _make_advise_linear(1, "up_proj", backing, cache)
        spec.register_linears(1, [lin1])
        spec.prev_uniq_by_layer[1] = [3, 4]
        with patch.object(ss, "_RA_ENV", True):
            plan = ss._RemapPlan()
            lin0._advise_next_layer_prev_token(plan)
            lin0._advise_next_layer_prev_token(plan)
        assert len(backing.advised) == 1, f"expected 1 run, got {backing.advised}"


class TestFaseKO2StashRing:
    def test_stash_populate_serves_demand(self):
        """Fase K F3: speculated runs land in the ring under tier-aware
        bundle keys and a later demand get() hits them."""
        import time

        import numpy as np

        from omlx.patches.expert_streaming import streaming_switch as ss

        cache = ss.ExpertLRUCache(0, 4096, num_layers=2)
        spec = _attach_spec(cache)
        backing = _AdviseRecorderBacking()
        lin0 = _make_advise_linear(0, "gate_proj", backing, cache)
        lin1 = _make_advise_linear(1, "up_proj", backing, cache)
        spec.register_linears(1, [lin1])
        spec.prev_uniq_by_layer[1] = [3, 4]
        with patch.object(ss, "_RA_ENV", True), patch.object(ss, "_STASH_ENV", True):
            plan = ss._RemapPlan()
            lin0._advise_next_layer_prev_token(plan)
            deadline = time.time() + 5.0
            while spec.stats["stash_inserts"] < 2 and time.time() < deadline:
                time.sleep(0.02)
        assert spec.stats["stash_inserts"] == 2, "stash reads must land"
        for eid in (3, 4):
            key = lin1.bundle_key(eid)
            assert key in spec.stash, f"stash missing {key}"
            w, s, b = spec.stash[key]
            assert w.shape == (64,)
        # A demand get() against the LRU resolution path must hit the ring.
        with patch.object(ss, "_RA_ENV", True), patch.object(ss, "_STASH_ENV", True):
            got = lin1._bundle_cached_or_staged(3)
        assert got is not None and got[0].shape == (64,)
        # FIFO ring: inserting beyond _STASH_MAX_ENTRIES evicts the oldest.
        for eid in range(10, 10 + ss._STASH_MAX_ENTRIES):
            key = (1, eid, lin1.stacked_weight_key)
            spec.stash_insert(key, (np.zeros(4, np.uint8), None, None))
        assert len(spec.stash) <= ss._STASH_MAX_ENTRIES
        # Dedupe: re-inserting an existing key is a no-op that never evicts.
        existing = list(spec.stash_order)[0]
        before = len(spec.stash)
        assert spec.stash_insert(existing, (np.zeros(4, np.uint8), None, None)) is False
        assert len(spec.stash) == before


    def test_stash_sparse_ids_exact_keys(self):
        """Fase K K2: sparse speculation reads EXACT keys — [3,5,9] must
        never read or store experts 4, 6, 7, 8 (reader-only segmentation
        produced (3,2) jobs that spanned the hole and stored the wrong
        experts)."""
        import time

        from omlx.patches.expert_streaming import streaming_switch as ss

        cache = ss.ExpertLRUCache(0, 4096, num_layers=2)
        spec = _attach_spec(cache)
        backing = _AdviseRecorderBacking()
        lin = _make_advise_linear(1, "up_proj", backing, cache)
        spec.register_linears(1, [lin])
        with patch.object(ss, "_RA_ENV", True), patch.object(ss, "_STASH_ENV", True):
            lin._stash_populate([3, 5, 9])
        deadline = time.time() + 5.0
        while spec.stats["stash_inserts"] < 3 and time.time() < deadline:
            time.sleep(0.02)
        assert spec.stats["stash_inserts"] == 3, "all three targets must land"
        assert spec.stats["stash_targets"] == 3
        assert spec.stats["stash_inserts"] <= spec.stats["stash_targets"], "coverage"
        for eid in (3, 5, 9):
            assert lin.bundle_key(eid) in spec.stash, f"missing {eid}"
        for eid in (4, 6, 7, 8):
            assert lin.bundle_key(eid) not in spec.stash, f"hole {eid} stored!"
        # The reads themselves were single-expert runs (no 3,4 span).
        assert (lin.stacked_weight_key, 3, 1) in backing.read_runs, "run (3,1) expected"
        assert not any(run[1:] == (3, 2) for run in backing.read_runs), "no span (3,2)"

    def test_stash_off_by_default_no_stash_reads(self):
        """Fase K F3: STASH=0 (default) issues advisory hints only."""
        from omlx.patches.expert_streaming import streaming_switch as ss

        cache = ss.ExpertLRUCache(0, 4096, num_layers=2)
        spec = _attach_spec(cache)
        backing = _AdviseRecorderBacking()
        lin0 = _make_advise_linear(0, "gate_proj", backing, cache)
        lin1 = _make_advise_linear(1, "up_proj", backing, cache)
        spec.register_linears(1, [lin1])
        spec.prev_uniq_by_layer[1] = [3, 4]
        with patch.object(ss, "_RA_ENV", True), patch.object(ss, "_STASH_ENV", False):
            plan = ss._RemapPlan()
            lin0._advise_next_layer_prev_token(plan)
        assert backing.read_runs == [], "stash disabled: no speculative reads"
        assert len(backing.advised) == 1, "F_RDADVISE still fires"


class TestFaseKSpecStateLifecycle:
    """Fase K corrections K1/K7: per-conversion isolation and drain."""

    def test_stash_isolated_per_backing(self):
        """K1: two conversions with the same keys never share ring bytes."""
        import time

        from omlx.patches.expert_streaming import streaming_switch as ss

        def build() -> tuple:
            cache = ss.ExpertLRUCache(0, 4096, num_layers=2)
            spec = _attach_spec(cache)
            backing = _AdviseRecorderBacking()
            lin = _make_advise_linear(1, "up_proj", backing, cache)
            spec.register_linears(1, [lin])
            return cache, backing, spec, lin

        ca, ba, sa, la = build()
        cb, bb, sb, lb = build()
        with patch.object(ss, "_RA_ENV", True), patch.object(ss, "_STASH_ENV", True):
            la._stash_populate([3])
        deadline = time.time() + 5.0
        while sa.stats["stash_inserts"] < 1 and time.time() < deadline:
            time.sleep(0.02)
        assert sa.stats["stash_inserts"] == 1, "A's speculation must land"
        key = la.bundle_key(3)
        assert key in sa.stash
        # B's ring never saw A's bytes; its stats are untouched.
        assert sb.stats["stash_inserts"] == 0
        assert lb.bundle_key(3) not in sb.stash
        # Closing A drains and clears the ring; a closed state accepts nothing.
        sa.close()
        assert sa.stash == {} and sa.stash_order == []
        assert sa.is_closed()
        with patch.object(ss, "_RA_ENV", True), patch.object(ss, "_STASH_ENV", True):
            assert la._spec_state() is sa
            la._stash_populate([5])  # closed: silently dropped
        assert sa.stats["stash_inserts"] == 1, "no inserts after close"
        # B keeps working independently after A closed.
        with patch.object(ss, "_RA_ENV", True), patch.object(ss, "_STASH_ENV", True):
            lb._stash_populate([7])
        deadline = time.time() + 5.0
        while sb.stats["stash_inserts"] < 1 and time.time() < deadline:
            time.sleep(0.02)
        assert sb.stats["stash_inserts"] == 1

    def test_stash_close_drains_inflight_reads(self):
        """K1: close() during in-flight speculation accepts no late writes."""
        import time

        from omlx.patches.expert_streaming import streaming_switch as ss

        class _SlowBacking(_AdviseRecorderBacking):
            def load_expert_run(self, key, first_id, count):
                time.sleep(0.05)
                return super().load_expert_run(key, first_id, count)

        cache = ss.ExpertLRUCache(0, 4096, num_layers=2)
        spec = _attach_spec(cache)
        backing = _SlowBacking()
        lin = _make_advise_linear(1, "up_proj", backing, cache)
        spec.register_linears(1, [lin])
        stock = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        with patch.object(ss, "_RA_ENV", True), patch.object(ss, "_STASH_ENV", True):
            lin._stash_populate(stock)
        deadline = time.time() + 5.0
        while spec.pending and time.time() < deadline:
            time.sleep(0.01)
        assert spec.stats["stash_inserts"] > 0, "reads started before close"
        with patch.object(ss, "_RA_ENV", True), patch.object(ss, "_STASH_ENV", True):
            lin._stash_populate([13, 14, 15])
        before = spec.stats["stash_inserts"]
        spec.close()
        deadline = time.time() + 5.0
        while backing.read_runs and time.time() < deadline:
            time.sleep(0.02)
        time.sleep(0.3)  # let any straggler worker finish its disk read
        assert spec.stash == {}, "close must leave the ring empty"
        assert spec.stats["stash_inserts"] == before, "no inserts after close"
        with spec.lock:
            assert spec.pending == set(), "pending futures dropped on close"

    def test_stash_concurrent_writers_keep_invariants(self):
        """K7: N threads over overlapping keys keep the ring consistent.

        Invariants: bounded size, FIFO without duplicates, locked stats,
        bounded pending queue, zero exceptions even under a submit storm.
        """
        import time

        import numpy as np

        from omlx.patches.expert_streaming import streaming_switch as ss

        spec = ss.SpeculationState()
        errors: list = []

        def writer(offset: int):
            try:
                for i in range(400):
                    key = (1, (offset + i * 37) % 300, "k")
                    spec.stash_insert(key, (np.zeros(1, np.uint8), None, None))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(o,)) for o in range(8)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        assert not errors, f"writer exceptions: {errors}"
        assert len(spec.stash) <= ss._STASH_MAX_ENTRIES
        assert len(spec.stash_order) == len(set(spec.stash_order)), "FIFO has dups"
        inserts = spec.stats["stash_inserts"]
        evictions = spec.stats["stash_evictions"]
        assert inserts == len(spec.stash) + evictions, "stats must reconcile"
        # Pending cap: a storm of slow reads is dropped, never unbounded.
        cache = ss.ExpertLRUCache(0, 4096, num_layers=2)
        spec2 = _attach_spec(cache)
        accepted = 0
        for _ in range(ss._STASH_MAX_PENDING + 200):
            if spec2.submit(lambda: time.sleep(0.001)):
                accepted += 1
        with spec2.lock:
            assert len(spec2.pending) <= ss._STASH_MAX_PENDING
            assert spec2._pending_reserved == 0, "no leaked reservations"
        assert accepted <= ss._STASH_MAX_PENDING + 200
        spec2.close()
        with spec2.lock:
            assert spec2.pending == set()
            assert spec2._pending_reserved == 0
    def test_pending_converges_after_worker_and_pool_failure(self):
        """Fase L6A: reservations and pending converge after a raising
        worker and after an executor that rejects the submission itself."""
        import time

        from omlx.patches.expert_streaming import streaming_switch as ss

        class _DeadPool:
            def submit(self, fn):
                raise RuntimeError("pool dead")

        # Worker exception: the future completes exceptionally, the done
        # callback discards it; pending and reservations converge without
        # close().
        cache = ss.ExpertLRUCache(0, 4096, num_layers=2)
        spec = _attach_spec(cache)

        def _boom():
            raise RuntimeError("spec worker failed")

        for _ in range(4):
            assert spec.submit(_boom)
        deadline = time.time() + 5.0
        while spec.pending and time.time() < deadline:
            time.sleep(0.01)
        with spec.lock:
            assert spec.pending == set(), "worker failure must drain pending"
            assert spec._pending_reserved == 0
        spec.close()
        with spec.lock:
            assert spec.pending == set() and spec._pending_reserved == 0

        # Executor hand-off failure: submit() itself raises, the atomic
        # reservation must be released on that path.
        cache2 = ss.ExpertLRUCache(0, 4096, num_layers=2)
        spec2 = _attach_spec(cache2)
        old_pool = ss._EXPERT_IO_POOL
        try:
            ss._EXPERT_IO_POOL = _DeadPool()
            assert spec2.submit(lambda: None) is False
            with spec2.lock:
                assert spec2._pending_reserved == 0, "executor failure leaked a slot"
                assert spec2.pending == set()
        finally:
            ss._EXPERT_IO_POOL = old_pool
        spec2.close()



class TestFaseKAdmissionFilter:
    def test_admission_engages_at_large_budget(self):
        """Fase K F9: the baggy capacity < 4096 cap is gone — the filter
        must engage at a 6 GiB budget (measured capacity ~6847)."""
        from omlx.patches.expert_streaming import streaming_switch as ss

        cache = ss.ExpertLRUCache(6 * 1024**3, 920_000, num_layers=48)
        assert cache.capacity > 4096, "premise: 6 GiB budget exceeds the old cap"
        with patch.object(ss, "_ADMISSION_ENV", True):
            cache2 = ss.ExpertLRUCache(6 * 1024**3, 920_000, num_layers=48)
            assert cache2._admission_enabled is True
        with patch.object(ss, "_ADMISSION_ENV", False):
            cache3 = ss.ExpertLRUCache(6 * 1024**3, 920_000, num_layers=48)
            assert cache3._admission_enabled is False


class TestFaseKPriorPath:
    def test_prior_round_trip_via_model_path(self, tmp_path):
        """Fase K F10: prior persists next to the real model dir; load clamps
        the sample count to 1 so the first measured chunk dominates."""
        from unittest.mock import MagicMock

        from omlx import prefill_transient_tracker as pt

        model_dir = tmp_path / "models" / "Qwen3.8-Flash-Next-oQ4e-mtp"
        model_dir.mkdir(parents=True)
        with patch.object(
            pt, "_PRIOR_DIR", tmp_path / "cache"
        ), patch.object(
            pt, "logger", MagicMock()
        ):
            t = pt.PrefillTransientTracker("qwen", model_path=model_dir)
            t.update(1000, 200_000)
            for _ in range(10):
                t.update(1000, 200_000)
            assert t.samples == 11
            t.save_prior()
            prior_file = model_dir / ".omlx" / "prefill_transient_prior.json"
            assert prior_file.exists(), "prior must live next to the model"
            t2 = pt.PrefillTransientTracker("qwen", model_path=model_dir)
            assert t2.load_prior() is True
            assert t2.bytes_per_token == 200.0
            assert t2.samples == 0, "samples must be clamped to 0 on load"
            assert t2.last_delta_bytes == 0, "prior delta is not measurement (K3)"
            assert t2.last_n_tokens == 0, "prior token count is not measurement (K3)"
            t2.update(1000, 400_000)
            assert t2.bytes_per_token == 400.0, "first measured chunk replaces prior"
            assert t2.samples == 1

    def test_prior_no_hardcoded_roots(self, tmp_path):
        """Fase K F10: without a model path the prior falls back to the
        env/cache dir — never a hardcoded device root."""
        from unittest.mock import MagicMock

        from omlx import prefill_transient_tracker as pt

        cache_dir = tmp_path / "omlx-cache"
        cache_dir.mkdir()
        with patch.object(pt, "_PRIOR_DIR", cache_dir), patch.object(
            pt, "logger", MagicMock()
        ):
            t = pt.PrefillTransientTracker("m1")
            t.update(100, 50_000)
            t.save_prior()
            assert (cache_dir / "prefill_prior_m1.json").exists()
            t2 = pt.PrefillTransientTracker("m1")
            assert t2.load_prior() is True

# ---------------------------------------------------------------------------
# Fase K F6 — read_expert_into, bank promotion (A1/A1b), rolling layer
# context (Etapa B), memtrace (ports of the faseJ pipeline tests)
# ---------------------------------------------------------------------------


def _write_shard_filled(path: Path, tensors: dict, seed: int = 1234) -> None:
    """Minimal safetensors file with deterministic non-zero payloads.

    _write_shard writes all-zero data, which would make any bit-exactness
    comparison vacuous — every byte pattern would match trivially.

    BF16 tensors get values from a small positive range rather than random
    bit patterns: random bf16 words decode to exponents around 1e30, the
    dequantized products overflow to inf/NaN, and then array_equal fails
    on NaN != NaN for outputs whose bit patterns are in fact identical.
    """
    import mlx.core as mx

    _DTY_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "U32": 4, "U8": 1}
    rng = np.random.default_rng(seed)
    header: dict = {}
    offset = 0
    blobs = []
    for key, (shape, dtype) in tensors.items():
        nbytes = int(np.prod(shape)) * _DTY_BYTES[dtype]
        header[key] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + nbytes],
        }
        if dtype == "BF16":
            vals = rng.uniform(0.5, 2.0, size=int(np.prod(shape))).astype(np.float32)
            words = np.asarray(
                mx.array(vals).astype(mx.bfloat16).view(mx.uint16), dtype="<u2"
            )
            blobs.append(words.tobytes())
        else:
            blobs.append(
                rng.integers(1, 256, size=nbytes, dtype=np.uint8).tobytes()
            )
        offset += nbytes
    hb = json.dumps(header).encode()
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for blob in blobs:
            f.write(blob)


def _quant_linear(tmp: Path, n_experts: int = 4):
    """A StreamingQuantizedSwitchLinear backed by a real shard store."""
    from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore
    from omlx.patches.expert_streaming.streaming_switch import (
        ExpertLRUCache,
        StreamingQuantizedSwitchLinear,
    )

    w_key = "model.layers.0.mlp.switch_mlp.gate_proj.weight"
    s_key = "model.layers.0.mlp.switch_mlp.gate_proj.scales"
    b_key = "model.layers.0.mlp.switch_mlp.gate_proj.biases"
    # 4-bit packed in uint32 (8 values/word): 128 in / 16 out, group_size 64.
    # MLX requires scales and biases to share a shape, so biases are
    # per-group too, matching what a real affine-quantized checkpoint stores.
    _write_shard_filled(
        tmp / "model.safetensors",
        {
            w_key: ((n_experts, 16, 16), "U32"),
            s_key: ((n_experts, 16, 2), "BF16"),
            b_key: ((n_experts, 16, 2), "BF16"),
        },
    )
    (tmp / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {k: "model.safetensors" for k in (w_key, s_key, b_key)}}
        )
    )
    backing = ExpertBackingStore(tmp)
    cache = ExpertLRUCache(64 * 1024, 1024, num_layers=1)
    linear = StreamingQuantizedSwitchLinear(
        layer_idx=0,
        proj_name="gate_proj",
        stacked_weight_key=w_key,
        stacked_scales_key=s_key,
        stacked_biases_key=b_key,
        num_experts=n_experts,
        input_dims=128,
        output_dims=16,
        backing=backing,
        cache=cache,
        group_size=64,
        bits=4,
        mode="affine",
        has_bias=False,
    )
    return linear


def test_backing_read_expert_into_matches_load_expert_slice():
    """C2: the coalesced read_expert_into must be byte-identical to the
    per-expert load_expert_slice path (correctness contract for the miss
    path consolidation in the linear's bank reader).

    The payload is deterministic NON-zero (filled shard), so a reversed
    run->buffer pairing cannot slip through undetected.
    """
    import numpy as np

    from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        key = "model.layers.0.mlp.switch_mlp.gate_proj.weight"
        per = 16 * 32 * 2
        _write_shard_filled(tmp / "model.safetensors", {key: ((4, 16, 32), "BF16")})
        (tmp / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {key: "model.safetensors"}})
        )
        backing = ExpertBackingStore(tmp)
        out1 = np.empty((1, per), dtype=np.uint8)
        assert backing.read_expert_into([(key, [0])], [out1])
        assert out1.any(), "fixture payload is all zeros; this gate proves nothing"
        assert np.array_equal(
            out1[0], backing.load_expert_slice(key, 0).view(np.uint8).reshape(-1)
        )
        # Multi-expert, out of order: exercises the multi-run parallel branch.
        eids = [3, 0, 2]
        out3 = np.empty((3, per), dtype=np.uint8)
        assert backing.read_expert_into([(key, eids)], [out3])
        for slot, eid in enumerate(eids):
            assert np.array_equal(
                out3[slot], backing.load_expert_slice(key, eid).view(np.uint8).reshape(-1)
            )
        # Wrong buffer shape -> False (safe fallback signal).
        bad = np.empty((2, per + 1), dtype=np.uint8)
        assert not backing.read_expert_into([(key, [0, 1])], [bad])
        empty = np.empty((0, per), dtype=np.uint8)
        assert backing.read_expert_into([(key, [])], [empty])




def test_read_expert_into_bridges_gap_but_scatters_only_demand():
    """Fase K K5: with merge_gap=2 a hole is read as ONE run, yet out
    receives ONLY the demanded rows — byte-identical to the unbridged
    path, and gap bytes can never leak into the output."""
    import numpy as np

    from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        key = "model.layers.0.mlp.switch_mlp.gate_proj.weight"
        per = 4 * 4 * 2  # BF16 (4,4) -> 32 B/expert
        _write_shard_filled(tmp / "model.safetensors", {key: ((16, 4, 4), "BF16")})
        (tmp / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {key: "model.safetensors"}})
        )
        backing = ExpertBackingStore(tmp)
        eids = [3, 4, 7, 8]  # hole ids 5,6
        reads: list[int] = []
        reader = backing._reader_for_key(key)
        orig = reader._read_into

        def spy(off, buf):
            reads.append(len(buf))
            return orig(off, buf)

        reader._read_into = spy
        out = np.empty((4, per), dtype=np.uint8)
        assert backing.read_expert_into([(key, eids)], [out], merge_gap=2)
        # ONE preadv covering rows 3..8 inclusive: 6 experts x per bytes.
        assert reads == [6 * per], f"expected one 6-row read, got {reads}"
        for slot, eid in enumerate(eids):
            assert np.array_equal(
                out[slot], backing.load_expert_slice(key, eid).view(np.uint8).reshape(-1)
            ), f"demanded row {eid} corrupted"
        # Unbridged run gives the SAME output bytes (gap rows never appear).
        reads.clear()
        out2 = np.empty((4, per), dtype=np.uint8)
        assert backing.read_expert_into([(key, eids)], [out2])
        assert len(reads) == 2, "unbridged: 2 separate reads"
        assert np.array_equal(out, out2), "bridge must not change the output"


def test_read_expert_banks_no_bridge_without_split():
    """Fase K K5 + re-gate: the C2 bank reader bridges ONLY while the
    HOBBIT split is active — without a split, merge_gap stays 0 and the
    rows returned are exactly the demanded ids (no hole rows)."""
    with tempfile.TemporaryDirectory() as td:
        lin = _quant_linear(Path(td), n_experts=16)
        got = lin._read_expert_banks([3, 4, 7, 8])
        assert got is not None
        segments, rows = got
        assert len(rows) == 4, f"exactly the demanded rows: {len(rows)}"
        assert segments[0][0] == [3, 4, 7, 8], segments[0][0]



def test_prefill_pool_used_on_rolling_path():
    """Fase K K4: the rolling prefetch must route its submissions through
    the regime pool — prefill-shaped demand sets use the PREFILL_QD pool,
    decode demand keeps the process-wide singleton."""
    from concurrent.futures import ThreadPoolExecutor

    from omlx.patches.expert_streaming import streaming_switch as ss

    class _RecordingPool(ThreadPoolExecutor):
        def __init__(self, *a, tag, **k):
            super().__init__(*a, **k)
            self.tag = tag
            self.submitted: list = []

        def submit(self, fn, *a, **k):
            self.submitted.append(fn)
            return super().submit(fn, *a, **k)

    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "m").mkdir()
        lin0 = _quant_linear(Path(td) / "m", n_experts=16)
        cache = lin0.cache
        backing = lin0.backing
        backing.num_experts = 16  # fixture needs for _make_advise_linear
        lin1 = _make_advise_linear(0, "up_proj", backing, cache)
        lin1.cache = cache
        prefill_pool = _RecordingPool(2, tag="prefill")
        decode_pool = _RecordingPool(2, tag="decode")
        try:
            with patch.object(ss, "_PREFILL_IO_POOL", prefill_pool), patch.object(
                ss, "_EXPERT_IO_POOL", decode_pool
            ), patch.object(ss, "_PREFILL_REGIME_MIN_POSITIONS", 8):
                ctx = ss._LayerLoadContext([lin0, lin1], cache)
                # Prefill-shaped (16 experts > 8): prefetch must go to the
                # PREFILL_QD pool.
                ctx.ensure(lin0, list(range(16)))
                assert ctx.misses[id(lin1)] == 16
                assert prefill_pool.submitted, "rolling prefetch missed the prefill pool"
                assert not decode_pool.submitted
                # Drain the in-flight prefetch before the next scenario.
                for fut in list(ctx._futures.values()):
                    fut.result()
                # Decode-shaped (3 experts <= 8): prefetch keeps the singleton.
                ctx2 = ss._LayerLoadContext([lin0, lin1], cache)
                ctx2.ensure(lin0, [1, 2, 3])
                assert not prefill_pool.submitted[1:], "decode must not touch the prefill pool"
                assert decode_pool.submitted, "decode prefetch uses the singleton"
        finally:
            prefill_pool.shutdown(wait=False, cancel_futures=True)
            decode_pool.shutdown(wait=False, cancel_futures=True)



def test_bank_bytes_respect_tier_widths():
    """Fase K K6: under the HOBBIT split the byte math uses the TIER's
    own width (hot = source packing). The cold-first estimate under-counts
    every hot expert, so an oversize bank could slip past the prefetch and
    bank caps; the tier-aware sum cannot."""
    from types import SimpleNamespace

    from omlx.patches.expert_streaming import streaming_switch as ss

    class _WidthBacking(_AdviseRecorderBacking):
        def __init__(self):
            super().__init__(per_expert_bytes=64, num_experts=64)
            self.hot_w = SimpleNamespace(expert_bytes=2 * 1024 * 1024)
            self.cold_w = SimpleNamespace(expert_bytes=1 * 1024 * 1024)

        def _reader_for_key(self, key, expert_id=None):
            width = (
                self.hot_w
                if expert_id is not None and expert_id < 8
                else self.cold_w
            )
            return SimpleNamespace(_rp_for=lambda k: width, path="/fake")

    cache = ss.ExpertLRUCache(0, 4096, num_layers=1)
    backing = _WidthBacking()
    lin = _make_advise_linear(0, "gate_proj", backing, cache)
    lin.set_hobbit_split(set(range(8)), cold_bits=2, cold_gs=32)
    assert lin._is_split_active()
    ids = list(range(16))  # 8 hot + 8 cold, 2 stacked keys per expert
    hot = 8 * 2 * 2 * 1024 * 1024
    cold = 8 * 2 * 1 * 1024 * 1024
    assert lin._tier_bank_bytes_for(ids) == hot + cold
    assert lin._tier_bank_bytes_for(ids) > lin._bank_bytes_for(16), "under-counts hot"
    # No split: the tier-aware sum reduces to the plain estimate.
    lin2 = _make_advise_linear(0, "gate_proj", _WidthBacking(), cache)
    assert lin2._tier_bank_bytes_for([0, 1]) == lin2._bank_bytes_for(2)



def _quant_linear_same_store(tmp: Path, backing, cache, n_experts: int = 16):
    """Second StreamingQuantizedSwitchLinear over the SAME shard store.

    The per-layer load context runs several projections per layer; the
    piece helpers must share one backing/cache so the read path and the
    tier resolution stay realistic.
    """
    from omlx.patches.expert_streaming.streaming_switch import (
        StreamingQuantizedSwitchLinear,
    )

    return StreamingQuantizedSwitchLinear(
        layer_idx=0,
        proj_name="gate_proj",
        stacked_weight_key="model.layers.0.mlp.switch_mlp.gate_proj.weight",
        stacked_scales_key="model.layers.0.mlp.switch_mlp.gate_proj.scales",
        stacked_biases_key="model.layers.0.mlp.switch_mlp.gate_proj.biases",
        num_experts=n_experts,
        input_dims=128,
        output_dims=16,
        backing=backing,
        cache=cache,
        group_size=64,
        bits=4,
        mode="affine",
        has_bias=False,
    )


def test_ctx_mode_is_union_for_decode_positions():
    """Fase 1: <=64 routed rows resolve through union; prefill keeps
    rolling; max_rows=0 disables the hybrid (rolling everywhere)."""
    from omlx.patches.expert_streaming import streaming_switch as ss

    with patch.object(ss, "_DECODE_UNION_MAX_ROWS", 64):
        for pos in (1, 10, 64):
            assert ss._layer_ctx_mode(pos, quantized=True, barrier=True) == "union", pos
        for pos in (65, 512, 4096):
            assert ss._layer_ctx_mode(pos, quantized=True, barrier=True) == "rolling", pos
    assert ss._layer_ctx_mode(1, quantized=False, barrier=True) is None
    assert ss._layer_ctx_mode(1, quantized=True, barrier=False) is None
    with patch.object(ss, "_DECODE_UNION_MAX_ROWS", 0):
        assert ss._layer_ctx_mode(1, quantized=True, barrier=True) == "rolling"
    with patch.object(ss, "_CTX_ROLLING_ENV", False):
        ctx = ss._LayerLoadContext([], None)  # type: ignore[arg-type]
        assert ctx.mode == "union", "global kill switch forces union"


def test_ctx_hybrid_matches_explicit_union_and_rolling_bundles():
    """Fase 1: the hybrid union path and the rolling path resolve the same
    decode demand to byte-identical bundles (the mode switch only changes
    read scheduling, never the gather bytes)."""
    import numpy as np

    from omlx.patches.expert_streaming import streaming_switch as ss

    with tempfile.TemporaryDirectory() as td:
        store = Path(td) / "m"
        store.mkdir()
        lin0 = _quant_linear(store, n_experts=16)
        backing = lin0.backing
        backing.num_experts = 16
        lin1 = _quant_linear_same_store(store, backing, lin0.cache)
        demand = [1, 3, 5, 8]
        ctx_r = ss._LayerLoadContext([lin0, lin1], lin0.cache, mode="rolling")
        ctx_u = ss._LayerLoadContext([lin0, lin1], lin0.cache, mode="union")
        ctx_r.ensure(lin0, demand)
        ctx_u.ensure(lin0, demand)
        assert not ctx_r.failed and not ctx_u.failed
        b_r = ctx_r.bundles.get(id(lin0), {})
        b_u = ctx_u.bundles.get(id(lin0), {})
        assert set(b_r) == set(b_u) == set(demand)
        for eid in demand:
            for a, b in zip(b_r[eid], b_u[eid]):
                assert np.array_equal(a, b), f"bundle {eid} diverged"


def test_ctx_hybrid_preserves_hobbit_keys():
    """Fase 1: with the HOBBIT split on, both modes keep the tier-suffixed
    bundle_key contract — hot ids never alias cold copies, and the union
    path resolves the same keys as rolling."""
    import numpy as np

    from omlx.patches.expert_streaming import streaming_switch as ss

    with tempfile.TemporaryDirectory() as td:
        store = Path(td) / "m"
        store.mkdir()
        lin0 = _quant_linear(store, n_experts=16)
        lin0.set_hobbit_split(set(range(8)), cold_bits=2, cold_gs=32)
        assert lin0._is_split_active()
        lin1 = _quant_linear_same_store(store, lin0.backing, lin0.cache)
        lin1.set_hobbit_split(set(range(8)), cold_bits=2, cold_gs=32)
        demand = [0, 3, 5]  # all hot: resolves on the source packing
        ctx_r = ss._LayerLoadContext([lin0, lin1], lin0.cache, mode="rolling")
        ctx_u = ss._LayerLoadContext([lin0, lin1], lin0.cache, mode="union")
        ctx_r.ensure(lin0, demand)
        ctx_u.ensure(lin0, demand)
        assert not ctx_r.failed and not ctx_u.failed
        for eid in demand:
            k = lin0.bundle_key(eid)
            assert k[2] == lin0.stacked_weight_key, f"hot key must stay unsuffixed: {k}"
        for eid in (8, 12):
            k = lin0.bundle_key(eid)
            assert k[2].endswith("#c"), f"cold key must be tier-suffixed: {k}"
        b_u = ctx_u.bundles.get(id(lin0), {})
        assert set(b_u) == set(demand)



def test_advisor_stats_count_bytes_per_tier_segment():
    """Fase 2: advise_expert_run reports the REAL byte coverage and the
    tier segments it needed — a run straddling hot/cold reports two
    segments and per-tier bytes (the old bool hid both)."""
    from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore

    class _FakeReader:
        def __init__(self, width):
            self.width = width

        def expert_byte_range(self, key, eid):
            return eid * self.width, (eid + 1) * self.width

        def advise_range(self, off, length):
            return True

    class _SegBacking(ExpertBackingStore):
        def __init__(self):
            self.hot = _FakeReader(2 * 1024 * 1024)
            self.cold = _FakeReader(1 * 1024 * 1024)

        def _reader_for_key(self, key, expert_id=None):
            return self.hot if (expert_id or 0) < 8 else self.cold

    b = _SegBacking()
    ok, adv_bytes, segs = b.advise_expert_run("k", 0, 16)  # 8 hot + 8 cold
    assert ok is True and segs == 2, (ok, segs)
    assert adv_bytes == 8 * 2 * 1024 * 1024 + 8 * 1 * 1024 * 1024, adv_bytes
    ok2, bytes2, segs2 = b.advise_expert_run("k", 0, 4)
    assert ok2 is True and segs2 == 1 and bytes2 == 4 * 2 * 1024 * 1024


def test_read_stats_collected_under_profile():
    """Fase M3: read_expert_into accumulates PER-BACKING telemetry (calls/
    runs/bytes/component_e2e); a disabled backing stays inert and module-
    level read_stats() without a backing returns None (no global state)."""
    import numpy as np

    from omlx.patches.expert_streaming import shard_bank as sb_mod
    from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        key = "model.layers.0.mlp.switch_mlp.gate_proj.weight"
        per = 4 * 4 * 2
        _write_shard_filled(tmp / "model.safetensors", {key: ((16, 4, 4), "BF16")})
        (tmp / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {key: "model.safetensors"}})
        )
        backing = ExpertBackingStore(tmp)
        out = np.empty((4, per), dtype=np.uint8)
        assert sb_mod.read_stats() is None, "no backing: no global state"
        backing.read_telemetry.enabled = False
        assert backing.read_expert_into([(key, [3, 4, 7, 8])], [out])
        assert sb_mod.read_stats(backing) is None, "disabled: inert"
        backing.read_telemetry.enabled = True
        assert backing.read_expert_into([(key, [3, 4, 7, 8])], [out])
        snap = sb_mod.read_stats(backing)
        assert snap is not None
        assert snap["lifetime"]["calls"] >= 1
        assert snap["lifetime"]["runs"] == 2, snap
        assert snap["lifetime"]["bytes"] == 4 * per, snap
        e2e = snap["lifetime"]["stages_us"]["component_e2e_us"]
        assert e2e["count"] >= 1 and e2e["p95"] is not None, e2e
        assert snap["lifetime"]["requested_inflight_peak"] >= 1
        assert snap["profiling_enabled"] is True
        assert snap["sample_capacity"] >= 64
        assert "prefill" not in snap or snap["prefill"]["calls"] >= 1, snap



def test_run_window_completion_vs_order_byte_identical():
    """Fase 5b: the completion-order window scatters the same bytes as
    the submission-order window even when reads complete out of order
    (a slow early run must not block the rest, and must not corrupt
    anyone's rows)."""
    import time

    import numpy as np

    from omlx.patches.expert_streaming import shard_bank as sb_mod
    from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        key = "model.layers.0.mlp.switch_mlp.gate_proj.weight"
        per = 4 * 4 * 2
        _write_shard_filled(tmp / "model.safetensors", {key: ((64, 4, 4), "BF16")})
        (tmp / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {key: "model.safetensors"}})
        )
        backing = ExpertBackingStore(tmp)
        eids = list(range(0, 32, 2))  # 16 single-expert runs
        reader = backing._reader_for_key(key)
        orig = reader._read_into

        def out_of_order(off, buf):
            # Earlier rows take MUCH longer: completions come reversed.
            time.sleep(0.015 if off < 10 * per else 0.001)
            return orig(off, buf)

        reader._read_into = out_of_order

        def run(mode, out):
            with patch.object(sb_mod, "_RUN_WINDOW_COMPLETION", mode):
                assert backing.read_expert_into([(key, eids)], [out])

        out_order = np.empty((len(eids), per), dtype=np.uint8)
        out_comp = np.empty((len(eids), per), dtype=np.uint8)
        run(False, out_order)
        run(True, out_comp)
        assert np.array_equal(out_order, out_comp), "windows must scatter identical bytes"
        for slot, eid in enumerate(eids):
            expect = backing.load_expert_slice(key, eid).view(np.uint8).reshape(-1)
            assert np.array_equal(out_comp[slot], expect), f"row {eid} corrupted"

def test_memtrace_disabled_by_default_is_noop():
    """With the env var unset the singleton is a null tracer: recording is
    a no-op and enabled is False, so the hot path stays free."""
    from omlx.patches.expert_streaming import streaming_switch as ss

    assert ss.memtrace.enabled is False
    assert ss.memtrace.record("linear.resolve", layer=0) is None


def test_memtracer_records_rows_and_tracks_peaks():
    """An armed tracer records one row per call with the memory counters
    and keeps a running per-metric peak."""
    import mlx.core as mx

    from omlx.patches.expert_streaming.memtrace import MemTracer

    tracer = MemTracer(path=None)
    tracer.record("linear.stack", layer=3, proj="up_proj", bank_bytes=1234)
    tracer.record("linear.stack", layer=3, proj="down_proj", bank_bytes=7)

    rows = tracer.rows()
    assert len(rows) == 2
    first = rows[0]
    for key in ("seq", "t", "event", "layer", "proj", "active", "cache", "peak",
                "footprint", "rss", "bank_bytes"):
        assert key in first, f"missing field {key}"
    assert first["event"] == "linear.stack"
    assert first["layer"] == 3
    assert first["proj"] == "up_proj"
    assert rows[1]["seq"] == 2

    peaks = tracer.peaks()
    assert peaks["bank_bytes"] == 1234
    assert peaks["active"] >= 0 and isinstance(peaks["active"], int)
    assert int(mx.get_active_memory()) >= 0


def test_memtracer_writes_jsonl():
    """A path-armed tracer appends valid JSONL rows flush-on-write."""
    import json

    from omlx.patches.expert_streaming.memtrace import MemTracer

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "trace.jsonl"
        tracer = MemTracer(path=str(path))
        tracer.record("glu.enter", layer=1, uniq=8)
        tracer.record("glu.exit", layer=1, uniq=8)
        tracer.flush()
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        assert len(lines) == 2
        rows = [json.loads(ln) for ln in lines]
        assert [r["event"] for r in rows] == ["glu.enter", "glu.exit"]
        assert tracer.rows() == []


def test_memtracer_scope_emits_enter_exit():
    """scope brackets a block with .enter/.exit rows, even on exception."""
    from omlx.patches.expert_streaming.memtrace import MemTracer

    tracer = MemTracer(path=None)
    with tracer.scope("ctx.ensure", layer=2):
        pass
    assert [r["event"] for r in tracer.rows()] == ["ctx.ensure.enter", "ctx.ensure.exit"]

    tracer.reset()
    try:
        with tracer.scope("ctx.ensure", layer=2):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert [r["event"] for r in tracer.rows()] == ["ctx.ensure.enter", "ctx.ensure.exit"]


def test_memtracer_summary_reports_gib_peaks():
    """summary() exposes both raw bytes and GiB-normalized peaks."""
    from omlx.patches.expert_streaming.memtrace import MemTracer

    tracer = MemTracer(path=None)
    tracer.record("linear.stack", bank_bytes=2 << 30)
    s = tracer.summary()
    assert s["enabled"] is True
    assert s["peaks_bytes"]["bank_bytes"] == 2 << 30
    assert s["peak_bank_bytes_gib"] == 2.0


def _ctx_modes_setup():
    """Build (tmp, linears, cache) for the _LayerLoadContext mode tests."""
    from omlx.patches.expert_streaming.streaming_switch import ExpertLRUCache

    tmp, _warmer, _backing, linears = _ra_warmer_setup()
    # budget 0 -> every expert misses, so all three projections must load.
    return tmp, linears, ExpertLRUCache(0, 1 << 12, num_layers=2)


def test_ctx_union_mode_traces_union_bank_bytes():
    """Legacy union mode (rolling=0) holds every projection's NumPy bank at
    once; the trace must report that union, not one projection."""
    import shutil

    from omlx.patches.expert_streaming import streaming_switch as ss
    from omlx.patches.expert_streaming.memtrace import MemTracer

    tmp, linears, cache = _ctx_modes_setup()
    tracer = MemTracer(path=None)
    original_trace, original_mode = ss.memtrace, ss._CTX_ROLLING_ENV
    ss.memtrace = tracer
    ss._CTX_ROLLING_ENV = False
    try:
        ctx = ss._LayerLoadContext(linears[0], cache)
        ctx.ensure(linears[0][0], [0, 1, 2, 3])

        events = [r["event"] for r in tracer.rows()]
        assert "ctx.ensure.enter" in events
        assert "ctx.ensure.exit" in events

        exit_row = next(r for r in tracer.rows() if r["event"] == "ctx.ensure.exit")
        assert exit_row["n_proj"] == 3
        assert exit_row["uniq"] == 4
        assert exit_row["n_loaded"] == 3
        assert exit_row["miss_per_proj"] == [4, 4, 4]
        expected = sum(lin._bank_bytes_for(4) for lin in linears[0])
        assert expected > 0
        assert exit_row["bank_bytes"] == expected
        assert exit_row["bank_bytes"] == 3 * linears[0][0]._bank_bytes_for(4)
        assert ctx.failed is False
        for lin in linears[0]:
            assert len(ctx.bundles[id(lin)]) == 4
    finally:
        ss.memtrace = original_trace
        ss._CTX_ROLLING_ENV = original_mode
        shutil.rmtree(tmp, ignore_errors=True)


def test_ctx_rolling_mode_resolves_one_projection_at_a_time():
    """Rolling mode (default): each call resolves only the asking projection
    and keeps at most _CTX_PREFETCH_AHEAD banks in flight — never the union."""
    import shutil

    from omlx.patches.expert_streaming import streaming_switch as ss
    from omlx.patches.expert_streaming.memtrace import MemTracer

    tmp, linears, cache = _ctx_modes_setup()
    tracer = MemTracer(path=None)
    original_trace, original_mode = ss.memtrace, ss._CTX_ROLLING_ENV
    ss.memtrace = tracer
    ss._CTX_ROLLING_ENV = True
    try:
        ctx = ss._LayerLoadContext(linears[0], cache)

        ctx.ensure(linears[0][0], [0, 1, 2, 3])
        assert len(ctx.bundles[id(linears[0][0])]) == 4
        assert len(ctx.bundles[id(linears[0][1])]) == 0
        assert len(ctx._futures) <= max(1, ss._CTX_PREFETCH_AHEAD)
        first_row = next(r for r in tracer.rows() if r["event"] == "ctx.ensure.exit")
        assert first_row["proj"] == linears[0][0].proj_name
        assert first_row["miss"] == 4
        assert first_row["bank_bytes"] == linears[0][0]._bank_bytes_for(4)
        assert first_row["bank_bytes"] * 3 == sum(
            lin._bank_bytes_for(4) for lin in linears[0]
        )

        ctx.ensure(linears[0][1], [0, 1, 2, 3])
        ctx.ensure(linears[0][2], [0, 1, 2, 3])
        for lin in linears[0]:
            assert len(ctx.bundles[id(lin)]) == 4
            assert ctx.hits[id(lin)] == 0
            assert ctx.misses[id(lin)] == 4
        assert ctx.failed is False
        assert ctx._futures == {}
        assert ctx._inflight_bytes == 0

        exits = [r for r in tracer.rows() if r["event"] == "ctx.ensure.exit"]
        assert [r["proj"] for r in exits] == [lin.proj_name for lin in linears[0]]
        before = len(tracer.rows())
        ctx.ensure(linears[0][0], [0, 1, 2, 3])
        assert len(tracer.rows()) == before
    finally:
        ss.memtrace = original_trace
        ss._CTX_ROLLING_ENV = original_mode
        shutil.rmtree(tmp, ignore_errors=True)


def test_ctx_rolling_mode_prefetch_ahead_zero_reads_on_demand():
    """With the prefetch window closed, each projection reads synchronously
    and no future is ever held in flight (still correct, just no overlap)."""
    import shutil

    from omlx.patches.expert_streaming import streaming_switch as ss

    tmp, linears, cache = _ctx_modes_setup()
    original_ahead, original_mode = ss._CTX_PREFETCH_AHEAD, ss._CTX_ROLLING_ENV
    ss._CTX_ROLLING_ENV = True
    ss._CTX_PREFETCH_AHEAD = 0
    try:
        ctx = ss._LayerLoadContext(linears[0], cache)
        for lin in linears[0]:
            ctx.ensure(lin, [0, 1, 2, 3])
            assert ctx._futures == {}
        assert ctx.failed is False
        for lin in linears[0]:
            assert len(ctx.bundles[id(lin)]) == 4
    finally:
        ss._CTX_PREFETCH_AHEAD = original_ahead
        ss._CTX_ROLLING_ENV = original_mode
        shutil.rmtree(tmp, ignore_errors=True)


def test_ctx_prefetch_ahead_default_keeps_io_depth_above_one():
    """The rolling path must keep more than one projection's read in flight.
    read_expert_into issues its preadv calls one at a time, so the prefetch
    window is the only thing giving the layer call an I/O queue depth above
    1. Pins the default so a revert to 1 fails here rather than as an
    unexplained throughput regression."""
    from omlx.patches.expert_streaming import streaming_switch as ss

    assert ss._CTX_PREFETCH_AHEAD >= 2


def test_ctx_rolling_mode_matches_union_mode_results():
    """Both modes must produce identical bundles — the rolling path is a
    scheduling change, not a data change. Bit-exactness of the loaded rows
    is what protects the token-ID gate downstream."""
    import shutil

    import numpy as np

    from omlx.patches.expert_streaming import streaming_switch as ss
    from omlx.patches.expert_streaming.streaming_switch import ExpertLRUCache

    tmp, _warmer, backing, linears = _ra_warmer_setup()
    original_mode = ss._CTX_ROLLING_ENV
    try:
        results = {}
        for mode in (False, True):
            cache = ExpertLRUCache(0, 1 << 12, num_layers=2)
            ss._CTX_ROLLING_ENV = mode
            ctx = ss._LayerLoadContext(linears[0], cache)
            for lin in linears[0]:
                ctx.ensure(lin, [3, 1, 0, 2])  # deliberately unsorted
            per_proj = {}
            for lin in linears[0]:
                bundles = ctx.bundles[id(lin)]
                per_proj[lin.proj_name] = {
                    eid: tuple(
                        np.asarray(part) if part is not None else None
                        for part in bundle
                    )
                    for eid, bundle in sorted(bundles.items())
                }
            results[mode] = per_proj
        assert set(results[False]) == set(results[True])
        for proj in results[False]:
            for eid in results[False][proj]:
                a, b = results[False][proj][eid], results[True][proj][eid]
                assert len(a) == len(b)
                for pa, pb in zip(a, b):
                    if pa is None or pb is None:
                        assert pa is pb
                    else:
                        assert np.array_equal(pa, pb)
    finally:
        ss._CTX_ROLLING_ENV = original_mode
        shutil.rmtree(tmp, ignore_errors=True)


def test_bank_bytes_for_scales_linearly_and_zeroes():
    """_bank_bytes_for is the tile-sizing primitive; it must be linear in
    the expert count and return 0 for a non-positive count."""
    import shutil

    from omlx.patches.expert_streaming import streaming_switch as ss

    tmp, linears, cache = _ctx_modes_setup()
    try:
        lin = linears[0][0]
        assert lin._bank_bytes_for(0) == 0
        one = lin._bank_bytes_for(1)
        assert one > 0
        assert lin._bank_bytes_for(3) == 3 * one
    finally:
        ss.memtrace = getattr(ss, "memtrace", None)
        shutil.rmtree(tmp, ignore_errors=True)

def test_bank_promote_is_bit_identical_to_per_expert_stack(tmp_path):
    """Etapa A1: promoting the whole demand bank in one mx.array must yield
    byte-for-byte the same tensors as promoting U per-expert slices and
    stacking them — so gather_qmm sees an identical bank."""
    import mlx.core as mx

    linear = _quant_linear(tmp_path, n_experts=5)

    ids = [4, 1, 3]  # deliberately out of order: row order must be preserved
    got = linear._load_expert_bank_mx(ids)
    assert got is not None
    segments, rows = got
    # Non-split linear: exactly one segment holding every asked expert.
    assert len(segments) == 1
    seg_ids, (w_fast, s_fast, b_fast) = segments[0]
    assert seg_ids == ids

    dt = linear._slice_dtypes_lazy()
    mini_w, mini_s, mini_b = [], [], []
    for w, s, b in rows:
        mini_w.append(linear._promote_np(w))
        mini_s.append(linear._promote_np(s, dt[0]))
        mini_b.append(linear._promote_np(b, dt[1]))

    assert w_fast.dtype == mx.stack(mini_w, axis=0).dtype
    assert mx.array_equal(w_fast, mx.stack(mini_w, axis=0))
    assert s_fast.dtype == mx.stack(mini_s, axis=0).dtype
    assert mx.array_equal(s_fast, mx.stack(mini_s, axis=0))
    assert mx.array_equal(b_fast, mx.stack(mini_b, axis=0))

    # Row k of the bank is expert ids[k], not file order.
    assert mx.array_equal(w_fast[0], mini_w[0])
    assert mx.array_equal(w_fast[2], mini_w[2])
    assert not mx.array_equal(w_fast[0], mini_w[1])


def test_quantized_call_identical_with_and_without_bank_promote(monkeypatch, tmp_path):
    """Etapa A1 end-to-end: the layer output must be bit-identical whether
    the single-promotion fast path is enabled or not. This is the gate that
    makes A1 lossless rather than near-lossless."""
    import mlx.core as mx

    from omlx.patches.expert_streaming import streaming_switch as ss

    x = mx.random.normal((6, 128)).astype(mx.float32)
    indices = mx.array([2, 0, 1, 2, 0, 1], dtype=mx.int32)

    def run(enabled):
        # A fresh linear per arm: reusing one would leave the LRU populated,
        # turning the second arm into a hit-only run and making the comparison
        # vacuous (neither arm would exercise a miss).
        linear = _quant_linear(tmp_path, n_experts=3)
        engaged: list[bool] = []
        real = linear._load_expert_bank_mx

        def spy(ids):
            out = real(ids)
            engaged.append(out is not None)
            return out

        monkeypatch.setattr(ss, "_BANK_PROMOTE_ENV", enabled)
        monkeypatch.setattr(linear, "_load_expert_bank_mx", spy)
        out = linear(x, indices)
        mx.eval(out)
        return out, engaged

    out_on, engaged_on = run(True)
    out_off, engaged_off = run(False)

    assert engaged_on == [True], engaged_on
    assert engaged_off == [], engaged_off

    assert out_on.dtype == out_off.dtype
    bits_on = np.ascontiguousarray(out_on).view(np.uint32).reshape(-1)
    bits_off = np.ascontiguousarray(out_off).view(np.uint32).reshape(-1)
    assert np.array_equal(bits_on, bits_off)
    assert bool(mx.all(mx.isfinite(out_on)))


def test_bank_promote_returns_none_without_bank_backing(tmp_path):
    """A dict backing cannot serve contiguous banks, so A1 must decline and
    let the legacy per-expert path run."""
    import mlx.core as mx

    linear = _quant_linear(tmp_path, n_experts=3)
    linear.backing = {(0, "gate_proj", "weight"): mx.zeros((3, 16, 16))}
    assert linear._load_expert_bank_mx([0, 1]) is None


def test_read_expert_banks_matches_legacy_bank_np(tmp_path):
    """The _read_expert_banks refactor must not change what
    _load_expert_bank_np returns to its callers."""
    linear = _quant_linear(tmp_path, n_experts=5)
    ids = [4, 1, 3]

    rows = linear._load_expert_bank_np(ids)
    got = linear._read_expert_banks(ids)
    assert rows is not None and got is not None
    segments, rows2 = got

    assert len(segments) == 1
    seg_ids, banks = segments[0]
    assert seg_ids == ids
    assert banks[0].shape == (len(ids), linear._slice_bytes(linear.stacked_weight_key))
    assert len(rows) == len(rows2) == len(ids)
    for (w1, s1, b1), (w2, s2, b2) in zip(rows, rows2):
        assert np.array_equal(w1, w2)
        assert np.array_equal(s1, s2)
        assert np.array_equal(b1, b2)


# ---------------------------------------------------------------------------
# Fase J — Etapa A1b: single-promotion bank build on the layer-context path
# ---------------------------------------------------------------------------


def _quant_glu(tmp: Path, n_experts: int = 3):
    """A quantized StreamingSwitchGLU whose projections sit on real shards.

    Built with quantized=True so StreamingSwitchGLU installs a
    _LayerLoadContext — the Etapa B path that runs by default and the one
    A1b targets. A1 itself never fires here: the context pre-resolves every
    expert, so missing is empty by the time the linear asks. That gap is
    exactly why A1b exists.
    """
    from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore
    from omlx.patches.expert_streaming.streaming_switch import (
        ExpertLRUCache,
        StreamingQuantizedSwitchLinear,
        StreamingSwitchGLU,
    )

    hidden, moe_hidden = 128, 64
    specs = (
        ("gate_proj", moe_hidden, hidden),
        ("up_proj", moe_hidden, hidden),
        ("down_proj", hidden, moe_hidden),
    )
    tensors = {}
    for proj, o, i in specs:
        base = f"model.layers.0.mlp.switch_mlp.{proj}"
        tensors[f"{base}.weight"] = ((n_experts, o, i // 8), "U32")
        tensors[f"{base}.scales"] = ((n_experts, o, i // 64), "BF16")
        tensors[f"{base}.biases"] = ((n_experts, o, i // 64), "BF16")
    _write_shard_filled(tmp / "model.safetensors", tensors)
    (tmp / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in tensors}})
    )

    backing = ExpertBackingStore(tmp)
    cache = ExpertLRUCache(64 * 1024, 1024, num_layers=1)
    glu = StreamingSwitchGLU(
        input_dims=hidden,
        hidden_dims=moe_hidden,
        num_experts=n_experts,
        layer_idx=0,
        backing=backing,
        cache=cache,
        fused_gate_up=False,
        inverse_scatter=False,
        quantized=True,
        group_size=64,
        bits=4,
        mode="affine",
    )
    for proj, o, i in specs:
        base = f"model.layers.0.mlp.switch_mlp.{proj}"
        setattr(
            glu,
            proj,
            StreamingQuantizedSwitchLinear(
                layer_idx=0,
                proj_name=proj,
                stacked_weight_key=f"{base}.weight",
                stacked_scales_key=f"{base}.scales",
                stacked_biases_key=f"{base}.biases",
                num_experts=n_experts,
                input_dims=i,
                output_dims=o,
                backing=backing,
                cache=cache,
                group_size=64,
                bits=4,
                mode="affine",
                has_bias=False,
            ),
        )
    return glu


def test_ctx_bank_promote_is_bit_identical_on_default_path(monkeypatch, tmp_path):
    """Etapa A1b: with the Etapa B barrier on (the default), the GLU output
    must be bit-identical whether single-promotion is enabled or not.

    This is the gate that makes A1b lossless rather than near-lossless. It
    also proves A1b engages where A1 does not: A1 measured zero invocations
    on this path, and would leave the default configuration with no
    bank-promotion win.
    """
    import mlx.core as mx

    from omlx.patches.expert_streaming import streaming_switch as ss

    assert ss._LAYER_BARRIER_ENV  # the default path A1b is scoped to

    # Fase 1 hybrid: A1b engages on the ROLLING mode, which decode-shaped
    # calls no longer use (positions <= 64 -> union). Use prefill-shaped
    # positions so this gate keeps exercising the rolling default path.
    x = mx.random.normal((1, 130, 128)).astype(mx.float32)
    indices = mx.array([i % 3 for i in range(130)], dtype=mx.int32)

    def run(enabled):
        monkeypatch.setattr(ss, "_BANK_PROMOTE_CTX_ENV", enabled)
        glu = _quant_glu(tmp_path, n_experts=3)
        engaged: list[bool] = []
        real = ss.StreamingQuantizedSwitchLinear._promote_banks

        def spy(self, segments):
            out = real(self, segments)
            engaged.append(out is not None)
            return out

        monkeypatch.setattr(ss.StreamingQuantizedSwitchLinear, "_promote_banks", spy)
        out = glu(x, indices)
        mx.eval(out)
        return out, engaged

    out_on, engaged_on = run(True)
    out_off, engaged_off = run(False)

    # One call per projection (gate/up/down).
    assert engaged_on == [True, True, True], engaged_on
    assert engaged_off == [], engaged_off

    assert out_on.dtype == out_off.dtype
    bits_on = np.ascontiguousarray(out_on).view(np.uint32).reshape(-1)
    bits_off = np.ascontiguousarray(out_off).view(np.uint32).reshape(-1)
    assert np.array_equal(bits_on, bits_off)
    assert bool(mx.all(mx.isfinite(out_on)))


def test_ctx_bank_promote_declined_on_partial_demand(monkeypatch, tmp_path):
    """A bank covering only part of the demand set must not be promoted.

    The missing experts do live in one contiguous bank, but the cache hits
    would have to be promoted separately and concatenated — a different
    layout contract that A1b deliberately leaves on the legacy path.
    """
    import mlx.core as mx

    from omlx.patches.expert_streaming import streaming_switch as ss

    monkeypatch.setattr(ss, "_BANK_PROMOTE_CTX_ENV", True)
    glu = _quant_glu(tmp_path, n_experts=3)

    # Seed one expert so gate_proj's demand set is a partial miss.
    proj = glu.gate_proj
    rows = proj._load_expert_bank_np([0])
    assert rows is not None
    glu._cache.put((0, 0, proj.stacked_weight_key), rows[0])

    recorded: dict = {}
    real_ensure = ss._LayerLoadContext.ensure

    def spy(ctx, linear, expert_ids):
        real_ensure(ctx, linear, expert_ids)
        recorded[linear.proj_name] = (
            ctx.bank_raw.get(id(linear)),
            ctx.bank_ids.get(id(linear)),
            ctx.misses.get(id(linear)),
        )

    monkeypatch.setattr(ss._LayerLoadContext, "ensure", spy)

    # Fase 1 hybrid: rolling (single-promotion) only engages on
    # prefill-shaped calls; pin the partial-demand decline there.
    x = mx.random.normal((1, 130, 128)).astype(mx.float32)
    indices = mx.array([i % 3 for i in range(130)], dtype=mx.int32)
    out = glu(x, indices)
    mx.eval(out)

    assert recorded["gate_proj"][2] == 2, recorded["gate_proj"]
    assert recorded["gate_proj"][0] is None
    assert recorded["gate_proj"][1] is None
    assert recorded["up_proj"][0] is not None
    assert recorded["up_proj"][1] == [0, 1, 2]
    assert bool(mx.all(mx.isfinite(out)))


class TestFaseKRunMerge:
    def test_group_runs_bridges_small_gap_within_tier(self):
        """Fase K F7: a run may bridge a <=2 gap of non-demanded ids (the
        gap bytes are read but never promoted); tier boundaries never merge.
        Bridging is active only while the HOBBIT split is on — single-tier
        demand is already contiguous, and bridging costs ~8% of 2k TTFT there.

        """
        from omlx.patches.expert_streaming import streaming_switch as ss

        lin = _make_advise_linear(0, "gate_proj", _AdviseRecorderBacking(), ss.ExpertLRUCache(0, 4096))
        # No split: bridging must NOT engage regardless of the env value.
        with patch.object(ss, "_RUN_MERGE_GAP", 2):
            runs = lin._group_runs([3, 4, 7, 8])
            assert runs == [(3, 2), (7, 2)], runs
            runs = lin._group_runs([3, 4, 8, 9])
            assert runs == [(3, 2), (8, 2)], runs
        # Split active with the DEFAULT env: bridge is OFF (measured net
        # loss in both regimes, split4 artifacts) — plain grouping.
        lin.set_hobbit_split({3, 4, 5, 6, 7, 8, 9}, cold_bits=2, cold_gs=32)
        with patch.object(ss, "_RUN_MERGE_GAP", 0):
            runs = lin._group_runs([3, 4, 7, 8])
            assert runs == [(3, 2), (7, 2)], runs
        # Env-opt-in bridging still engages under the split (covers 3..8).
        with patch.object(ss, "_RUN_MERGE_GAP", 2):
            runs = lin._group_runs([3, 4, 7, 8])
            assert runs == [(3, 6)], runs
            # Gap of 3 (ids 3,4 then 8,9) is beyond the bridge budget.
            runs = lin._group_runs([3, 4, 8, 9])
            assert runs == [(3, 2), (8, 2)], runs
            # Bridge disabled by env.
            with patch.object(ss, "_RUN_MERGE_GAP", 0):
                runs = lin._group_runs([3, 4, 7, 8])
                assert runs == [(3, 2), (7, 2)], runs
            # max_run still bounds the bridged run (bridge clamps to the cap).
            with patch.object(ss, "_RUN_MAX", 4):
                runs = lin._group_runs([3, 4, 7, 8])
                assert runs == [(3, 4), (7, 2)], runs

    def test_group_runs_bridge_never_crosses_tier(self):
        """Under the HOBBIT split the bridge stays inside the first tier."""
        from omlx.patches.expert_streaming import streaming_switch as ss

        backing = _AdviseRecorderBacking()
        lin = _make_advise_linear(0, "gate_proj", backing, ss.ExpertLRUCache(0, 4096))
        lin.set_hobbit_split({1, 2, 9}, cold_bits=2, cold_gs=32)  # hot: 1,2,9
        ids = [1, 2, 4, 5, 9]  # hot(1,2) gap cold(4,5): must NOT bridge to cold
        with patch.object(ss, "_RUN_MERGE_GAP", 2):
            runs = lin._group_runs(ids)
        assert runs[0] == (1, 2), f"hot run must stay intact: {runs}"
        assert (4, 2) in runs or (4, 1) in runs




class TestFaseM3ReadTelemetry:
    """Fase M3: telemetry is per-backing, phase-scoped, bounded and
thread-safe; ctx fallback counters are per-cache."""

    @staticmethod
    def _backing(tmp: Path, enabled: bool = True):
        import numpy as np

        from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore

        key = "model.layers.0.mlp.switch_mlp.gate_proj.weight"
        per = 4 * 4 * 2
        _write_shard_filled(tmp / "model.safetensors", {key: ((16, 4, 4), "BF16")})
        (tmp / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {key: "model.safetensors"}})
        )
        backing = ExpertBackingStore(tmp)
        backing.read_telemetry.enabled = enabled
        return backing, key, per

    def test_read_telemetry_isolated_per_backing(self, tmp_path):
        """Two backings never share counters; disabling one leaves it
        inert while the other keeps counting."""
        with tempfile.TemporaryDirectory() as td:
            backing, key, per = self._backing(Path(td), enabled=True)
            (Path(td) / "b2").mkdir()
            backing2, key2, _per2 = self._backing(Path(td) / "b2", enabled=False)
            out = np.empty((4, per), dtype=np.uint8)
            out2 = np.empty((4, per), dtype=np.uint8)
            assert backing.read_expert_into([(key, [3, 4, 7, 8])], [out])
            assert backing2.read_expert_into([(key2, [3, 4, 7, 8])], [out2])
            s1 = backing.read_telemetry.summary()
            s2 = backing2.read_telemetry.summary()
            assert s1["lifetime"]["calls"] == 1, s1
            assert s2["lifetime"]["calls"] == 0, "disabled backing stays inert"

    def test_prefill_and_decode_summaries_do_not_mix(self, tmp_path):
        """begin/end scopes split one request: prefill metrics land ONLY
        under prefill, decode under decode."""
        with tempfile.TemporaryDirectory() as td:
            backing, key, per = self._backing(Path(td))
            tel = backing.read_telemetry
            out = np.empty((4, per), dtype=np.uint8)
            tel.begin_phase("prefill", request_id="req-1", engine_id="e1")
            assert backing.read_expert_into([(key, [0, 1, 2, 3])], [out])
            tel.end_phase()
            tel.begin_phase("decode", request_id="req-1", engine_id="e1")
            assert backing.read_expert_into([(key, [4, 5, 6, 7])], [out])
            tel.end_phase()
            snap = tel.summary()
            assert snap["prefill"]["calls"] == 1
            assert snap["decode"]["calls"] == 1
            assert snap["lifetime"]["calls"] == 2
            assert "req-1/prefill" in snap["requests"]
            p_pre = snap["prefill"]["stages_us"]["component_e2e_us"]["count"]
            p_dec = snap["decode"]["stages_us"]["component_e2e_us"]["count"]
            assert p_pre == 1 and p_dec == 1

    def test_two_requests_do_not_merge_phase_stats(self, tmp_path):
        """Per-request keys stay distinct; the merged prefill/decode views
        are the ANALYTIC merge of both requests, never a mix across
        phases."""
        with tempfile.TemporaryDirectory() as td:
            backing, key, per = self._backing(Path(td))
            tel = backing.read_telemetry
            out = np.empty((4, per), dtype=np.uint8)
            for rid in ("r1", "r2"):
                tel.begin_phase("prefill", request_id=rid)
                backing.read_expert_into([(key, [0, 1, 2, 3])], [out])
                tel.end_phase()
                tel.begin_phase("decode", request_id=rid)
                backing.read_expert_into([(key, [4, 5, 6, 7])], [out])
                tel.end_phase()
            snap = tel.summary()
            assert set(snap["requests"]) == {"r1/prefill", "r1/decode", "r2/prefill", "r2/decode"}
            assert snap["prefill"]["calls"] == 2
            assert snap["decode"]["calls"] == 2
            assert snap["lifetime"]["calls"] == 4

    def test_reservoir_or_histogram_is_bounded(self):
        """Percentile reservoirs cap at sample_capacity and report the
        drop count — long runs cannot grow memory linearly."""
        from omlx.patches.expert_streaming.shard_bank import ReadTelemetry

        tel = ReadTelemetry(enabled=True, sample_capacity=128)
        for _ in range(20000):
            tel.record_call(
                runs=1, bytes_=16, requested_inflight=1,
                timings={"component_e2e_us": [1234], "read_duration_us": [100]},
            )
        snap = tel.summary()
        assert snap["sample_capacity"] == 128
        e2e = snap["lifetime"]["stages_us"]["component_e2e_us"]
        assert e2e["count"] == 20000
        assert snap["dropped_samples"] >= 20000 - 128, snap
        assert e2e["p50"] == 1234 and e2e["max"] == 1234

    def test_concurrent_read_telemetry_reconciles_counts(self, tmp_path):
        """Threads hammering one backing: the per-call lock must reconcile
        calls/runs exactly (worker threads never take the lock)."""
        import threading

        with tempfile.TemporaryDirectory() as td:
            backing, key, per = self._backing(Path(td))
            out = np.empty((4, per), dtype=np.uint8)
            errors = []

            def worker():
                try:
                    for _ in range(25):
                        if not backing.read_expert_into([(key, [3, 4, 7, 8])], [out]):
                            errors.append("read failed")
                except Exception as e:
                    errors.append(repr(e))

            threads = [threading.Thread(target=worker) for _ in range(4)]
            for th in threads:
                th.start()
            for th in threads:
                th.join()
            assert not errors, errors
            snap = backing.read_telemetry.summary()
            assert snap["lifetime"]["calls"] == 100, snap
            assert snap["lifetime"]["runs"] == 200, snap

    def test_ctx_fallback_stats_reset_on_engine_unload(self):
        """The cache-owned counter resets cleanly (engine unload path);
        a fresh cache starts zeroed — nothing survives in module state."""
        from omlx.patches.expert_streaming import streaming_switch as ss

        cache = ss.ExpertLRUCache(0, 4096, num_layers=2)
        cache._count_ctx_fallback("read_failure")
        cache._count_ctx_fallback("bank_too_large")
        assert cache.ctx_fallback_stats() == {"read_failure": 1, "bank_too_large": 1}
        cache._reset_ctx_fallback_stats()
        assert cache.ctx_fallback_stats() == {}
        assert ss.ExpertLRUCache(0, 4096).ctx_fallback_stats() == {}



class TestFaseM2M4StageAndPoolTelemetry:
    """Fase M2/M4: stage-split read timers (pool/SSD/planner/scatter) and
    observed run-pool concurrency — requested QD is not effective depth."""

    @staticmethod
    def _backing(tmp: Path, enabled: bool = True):
        import numpy as np

        from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore

        key = "model.layers.0.mlp.switch_mlp.gate_proj.weight"
        per = 4 * 4 * 2
        _write_shard_filled(tmp / "model.safetensors", {key: ((16, 4, 4), "BF16")})
        (tmp / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {key: "model.safetensors"}})
        )
        backing = ExpertBackingStore(tmp)
        backing.read_telemetry.enabled = enabled
        return backing, key, per

    def test_read_telemetry_splits_stages_with_a3_vocabulary(self, tmp_path):
        """Fase A3: the honest window vocabulary — read_duration and
        worker_start_delay are SEPARATE per-run buckets; window_wait
        covers the caller's blocks; last_future_wait isolates the final
        run's tail; the old ambiguous names are gone."""
        import time

        with tempfile.TemporaryDirectory() as td:
            backing, key, per = self._backing(Path(td))
            reader = backing._reader_for_key(key)
            orig = reader._read_into

            def slow(off, buf):
                time.sleep(0.002)
                return orig(off, buf)

            reader._read_into = slow
            out = np.empty((4, per), dtype=np.uint8)
            assert backing.read_expert_into([(key, [3, 4, 7, 8])], [out])
            snap = backing.read_telemetry.summary()
            st = snap["lifetime"]["stages_us"]
            assert st["read_duration_us"]["count"] == 2, st
            assert st["read_duration_us"]["p50"] >= 1900, st  # ~2 ms per run
            assert st["worker_start_delay_us"]["count"] == 2
            assert st["plan_us"]["count"] == 1
            assert st["reader_resolve_us"]["count"] == 1
            assert st["scatter_us"]["count"] == 2, st
            assert st["component_e2e_us"]["count"] == 1
            ww = st["window_wait_us"]
            assert ww["count"] == 2, ww  # one pop in the main loop + one drain
            tail = st["last_future_wait_us"]
            assert tail["count"] == 1 and tail["p50"] is not None, tail
            for gone in ("queue_wait_us", "preadv_us", "future_tail_us"):
                assert gone not in st, gone

    def test_component_e2e_covers_all_stages(self, tmp_path):
        """e2e >= the sequential host stages (resolve+plan+scatter+reads)"""
        import time

        with tempfile.TemporaryDirectory() as td:
            backing, key, per = self._backing(Path(td))
            reader = backing._reader_for_key(key)
            orig = reader._read_into

            def slow(off, buf):
                time.sleep(0.001)
                return orig(off, buf)

            reader._read_into = slow
            out = np.empty((4, per), dtype=np.uint8)
            assert backing.read_expert_into([(key, [0, 1, 2, 3])], [out])
            st = backing.read_telemetry.summary()["lifetime"]["stages_us"]
            e2e = st["component_e2e_us"]["p50"] or 0
            host = 0
            for m in ("reader_resolve_us", "plan_us", "read_duration_us", "scatter_us"):
                host += st[m]["sum"]
            assert e2e >= host, (e2e, host)

    def test_run_timing_recorded_when_worker_raises(self, tmp_path):
        """A failing worker records the component as failed with a
        fallback_us duration — the caller's legacy fallback is visible."""
        from omlx.patches.expert_streaming import shard_bank as sb_mod

        with tempfile.TemporaryDirectory() as td:
            backing, key, per = self._backing(Path(td))
            reader = backing._reader_for_key(key)
            orig = reader._read_into

            def boom(off, buf):
                raise OSError("disk gone")

            reader._read_into = boom
            out = np.empty((4, per), dtype=np.uint8)
            assert not backing.read_expert_into([(key, [3, 4, 7, 8])], [out])
            st = backing.read_telemetry.summary()["lifetime"]["stages_us"]
            assert st["fallback_us"]["count"] >= 1, st
            assert st["component_e2e_us"]["count"] >= 1
            assert backing.read_telemetry.summary()["lifetime"]["failed_calls"] >= 1

    def test_telemetry_does_not_touch_mlx_in_worker(self, tmp_path):
        """The profiled worker path (submit wrapper) must never create MLX
        arrays — patch mx.array to raise and run a profiled multi-run read."""
        from unittest.mock import patch

        from omlx.patches.expert_streaming import shard_bank as sb_mod

        with tempfile.TemporaryDirectory() as td:
            backing, key, per = self._backing(Path(td))
            out = np.empty((4, per), dtype=np.uint8)
            with patch.object(sb_mod.mx, "array", side_effect=AssertionError("MLX in worker")):
                assert backing.read_expert_into([(key, [3, 4, 7, 8])], [out])
            assert backing.read_telemetry.summary()["lifetime"]["calls"] == 1

    def test_run_pool_active_peak_less_than_requested_when_workers_busy(
        self, tmp_path, monkeypatch
    ):
        """Fase M4: with ONE worker, four runs request in-flight=4 but the
        observed pool active peak is 1 — requested QD is a window, not
        effective depth."""
        import threading
        import time

        from concurrent.futures import ThreadPoolExecutor

        from omlx.patches.expert_streaming import shard_bank as sb_mod

        with tempfile.TemporaryDirectory() as td:
            backing, key, per = self._backing(Path(td))
            reader = backing._reader_for_key(key)
            orig = reader._read_into

            def slow(off, buf):
                time.sleep(0.002)
                return orig(off, buf)

            reader._read_into = slow
            out = np.empty((4, per), dtype=np.uint8)
            old_singleton = sb_mod._RUN_IO_POOL_SINGLETON
            old_qd = sb_mod._RUN_IO_QD
            one = ThreadPoolExecutor(max_workers=1, thread_name_prefix="one-run")
            one.telemetry = sb_mod.RunPoolTelemetry()
            try:
                sb_mod._RUN_IO_POOL_SINGLETON = one
                monkeypatch.setattr(sb_mod, "_RUN_IO_QD", 4)
                assert backing.read_expert_into([(key, [0, 2, 4, 6])], [out])
            finally:
                sb_mod._RUN_IO_POOL_SINGLETON = old_singleton
                one.shutdown(wait=True, cancel_futures=True)
            snap = one.telemetry.snapshot()
            assert snap["submitted"] == 4
            assert snap["started"] == 4 and snap["completed"] == 4
            assert snap["active_peak"] == 1, snap
            assert snap["queue_delay_us_max"] > 0, "later runs queued"
            req = backing.read_telemetry.summary()["lifetime"]["requested_inflight_peak"]
            assert req == 4 and snap["active_peak"] == 1 < req

    def test_pool_telemetry_balances_after_success_failure_cancel(self):
        """submitted == started + queued holds across success, failure and
        cancellation (cancelled tasks never start, so they stay queued)."""
        import time

        from concurrent.futures import ThreadPoolExecutor

        from omlx.patches.expert_streaming import shard_bank as sb_mod

        pool = ThreadPoolExecutor(max_workers=1)
        tel = sb_mod.RunPoolTelemetry()
        try:
            slow_ts = time.perf_counter_ns()
            slow = tel.wrap(slow_ts, lambda: time.sleep(0.05))
            tel.submit_notice()
            f_slow = pool.submit(slow)
            for _ in range(3):
                ts = time.perf_counter_ns()
                fn = tel.wrap(ts, lambda: 1)
                tel.submit_notice()
                pool.submit(fn)
            # Two more wrapped jobs: one success, one failure.
            okt = time.perf_counter_ns()
            okf = tel.wrap(okt, lambda: 2)
            tel.submit_notice()
            assert pool.submit(okf).result() == 2
            bad = tel.wrap(time.perf_counter_ns(), lambda: 1 / 0)
            tel.submit_notice()
            try:
                pool.submit(bad).result()
                assert False, "must raise"
            except ZeroDivisionError:
                pass
            f_slow.result()
            snap = tel.snapshot()
            assert snap["completed"] == 5 and snap["failed"] == 1, snap
            assert snap["started"] == snap["completed"] + snap["failed"]
            # Balance invariant: every submission is either started or still
            # queued; cancelled tasks never start, so they stay queued.
            assert snap["submitted"] == snap["started"] + snap["queued"], snap
            assert snap["active"] == 0
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

    def test_run_pool_telemetry_does_not_change_read_bytes(self, tmp_path):
        """Pool wrapping is measurement-only: outputs stay byte-identical.
        """
        import time

        from concurrent.futures import ThreadPoolExecutor

        from omlx.patches.expert_streaming import shard_bank as sb_mod

        with tempfile.TemporaryDirectory() as td:
            backing, key, per = self._backing(Path(td))
            old = sb_mod._RUN_IO_POOL_SINGLETON
            pools = []
            results = []
            try:
                for on in (False, True):
                    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="p2")
                    pool.telemetry = sb_mod.RunPoolTelemetry() if on else None
                    pools.append(pool)
                    monkeypatch_pool = pool
                    sb_mod._RUN_IO_POOL_SINGLETON = pool
                    out = np.empty((4, per), dtype=np.uint8)
                    assert backing.read_expert_into([(key, [3, 4, 7, 8])], [out])
                    results.append(np.copy(out))
            finally:
                sb_mod._RUN_IO_POOL_SINGLETON = old
                for p in pools:
                    p.shutdown(wait=True, cancel_futures=True)
            assert np.array_equal(results[0], results[1]), "wrapper changed bytes"



class TestFaseM6MemtraceContext:
    """Fase M6: ambient phase/request context and per-(layer, proj) event
    sequence on memtrace rows."""

    def test_context_fields_land_on_every_row(self):
        from omlx.patches.expert_streaming.memtrace import MemTracer

        tr = MemTracer(path=None, sample_every=1)
        tr.set_context(phase="prefill", request_id="r1", engine_id="e1")
        tr.record("dual_tier.enter", layer=0, proj="gate_proj")
        tr.set_context(phase="decode", request_id="r1", engine_id="e1")
        tr.record("dual_tier.enter", layer=1, proj="up_proj")
        tr.clear_context()
        tr.record("ctx.ensure.exit", layer=2, proj="down_proj")
        rows = tr.rows()
        assert rows[0]["phase"] == "prefill" and rows[0]["request_id"] == "r1"
        assert rows[1]["phase"] == "decode"
        assert "phase" not in rows[2], "cleared context must vanish"

    def test_event_seq_monotone_per_layer_proj(self):
        from omlx.patches.expert_streaming.memtrace import MemTracer

        tr = MemTracer(path=None, sample_every=1)
        tr.record("dual_tier.enter", layer=0, proj="gate_proj")
        tr.record("dual_tier.hot.bank_ready", layer=0, proj="gate_proj")
        tr.record("dual_tier.enter", layer=0, proj="up_proj")
        tr.record("dual_tier.layer_exit", layer=0, proj="gate_proj")
        rows = tr.rows()
        seq_g = [r["event_seq"] for r in rows if r.get("proj") == "gate_proj"]
        seq_u = [r["event_seq"] for r in rows if r.get("proj") == "up_proj"]
        assert seq_g == [1, 2, 3], seq_g
        assert seq_u == [1], seq_u


class TestFaseM1PinWiring:
    """Fase M1: pins reach the PinController as explicit configuration
    before the first request — settings win over env, the bench wires them
    ahead of engine load, and the JSON proves the effective values."""

    def _regime_settings(self, regime: str = "prefill"):
        from bench.bench_expert_streaming import _bench_settings

        return _bench_settings(
            pins=True,
            pin_gib=0.5,
            pin_regime=regime,
            budget=0.0,
            topk=None,
            prior=None,
            cold_tier=None,
            hot_fraction=None,
            mtp=False,
            mtp_block=None,
            ane=False,
            specprefill_draft=None,
            specprefill_keep=None,
        )

    def test_bench_settings_wire_regime_and_sync(self):
        """The bench builds ModelSettings with the pin knobs explicitly,
        so the controller inside get_engine sees them from token 1."""
        s = self._regime_settings("prefill")
        assert s.expert_streaming_pins is True
        assert s.expert_streaming_pin_regime == "prefill"
        assert s.expert_streaming_pin_sync is True
        assert s.expert_streaming_pin_gib == 0.5
        s_off = self._regime_settings()
        s_off.expert_streaming_pins = None
        s_off.expert_streaming_pin_regime = None
        s_off.expert_streaming_pin_sync = None
        s_off.expert_streaming_pin_gib = None

    def test_pin_settings_override_env_defaults(self, tmp_path, monkeypatch):
        """Explicit constructor args win over the env constants; when None,
        the env fallbacks still apply (server compatibility)."""
        from collections import Counter

        from omlx.patches.expert_streaming import warmer as warmer_mod

        monkeypatch.setattr(warmer_mod, "PIN_REGIME", "decode")
        monkeypatch.setattr(warmer_mod, "PIN_SYNC_ENABLED", False)
        linears = {0: [self._stub_lin()]}
        backing = self._pin_backing()
        # Explicit: prefill regime + sync on.
        ctl = warmer_mod.PinController(
            linears, backing, per_expert_bytes=1024,
            pin_regime="prefill", pin_sync=True,
        )
        assert ctl.pin_regime == "prefill"
        assert ctl.profile_regime == "prefill"
        assert ctl.pin_sync is True
        # Unset: env fallback constants apply.
        ctl2 = warmer_mod.PinController(linears, backing, per_expert_bytes=1024)
        assert ctl2.pin_regime == "decode"
        assert ctl2.pin_sync is False

    def test_pin_regime_invalid_falls_back_to_decode(self):
        from omlx.patches.expert_streaming import warmer as warmer_mod

        linears = {0: [self._stub_lin()]}
        backing = self._pin_backing()
        ctl = warmer_mod.PinController(
            linears, backing, per_expert_bytes=1024, pin_regime="bogus",
        )
        assert ctl.pin_regime == "decode"

    def _stub_lin(self):
        from types import SimpleNamespace

        base = "model.layers.0.mlp.switch_mlp.gate_proj"
        return SimpleNamespace(
            stacked_weight_key=f"{base}.weight",
            stacked_scales_key=f"{base}.scales",
            stacked_biases_key=None,
            backing=None,
        )

    def _pin_backing(self, per_expert: int = 1024):
        class _Reader:
            def __init__(self, name, width):
                self.path = f"stub-{name}"
                self.width = width

            def expert_byte_range(self, key, eid):
                return int(eid) * self.width, (int(eid) + 1) * self.width

        class _B:
            def __init__(self):
                self.path = "stub"
                self.pinned_calls: list = []
                self.pinned_bytes = 0
                self._pinned: set = set()

            def _reader_for_key(self, key, expert_id=None):
                return _Reader("hot", per_expert)

            def pin_expert(self, key, expert_id):
                self.pinned_calls.append((key, int(expert_id)))
                self._pinned.add((key, int(expert_id)))
                self.pinned_bytes += per_expert
                return per_expert

            @property
            def pinned_count(self):
                return len(self._pinned)

        return _B()

    def test_sync_pin_completes_before_first_request(self, tmp_path):
        """pin_sync=True with a learned profile: the mlock pass finishes
        inside the controller construction — proven by pins_applied_at_load
        and the backing's wired bytes."""
        import json

        from collections import Counter

        from omlx.patches.expert_streaming import warmer as warmer_mod

        profile = str(tmp_path / "expert_pin_profile.json")
        linears = {0: [self._stub_lin()]}
        backing = self._pin_backing()
        ctl = warmer_mod.PinController(
            linears, backing, per_expert_bytes=1024, profile_path=profile,
        )
        ctl.freq = {0: Counter({3: 9})}
        ctl.save_profile()
        ctl2 = warmer_mod.PinController(
            linears, backing, per_expert_bytes=1024, profile_path=profile,
            pin_sync=True,
        )
        assert ctl2.pins_applied_at_load is True
        assert ctl2.pin_sync is True
        assert backing.pinned_count > 0
        assert ctl2.pin_load_time_ms > 0.0


class TestFaseL1CtxObservability:
    """Fase L1: the hybrid decode fast path is observable — explicit ctx
    memtrace fields, per-reason fallback counters, and bit-exactness when
    the fast path degrades to the legacy resolution."""

    def test_ctx_fallback_stats_api(self):
        """Fase M3: fallback counters live per CACHE (per engine), not in
        module globals — two caches never mix reasons."""
        from omlx.patches.expert_streaming import streaming_switch as ss

        cache = ss.ExpertLRUCache(0, 4096, num_layers=2)
        assert cache.ctx_fallback_stats() == {}
        cache._count_ctx_fallback("read_failure")
        cache._count_ctx_fallback("read_failure")
        cache._count_ctx_fallback("bank_too_large")
        assert cache.ctx_fallback_stats() == {
            "read_failure": 2,
            "bank_too_large": 1,
        }
        other = ss.ExpertLRUCache(0, 4096, num_layers=2)
        assert other.ctx_fallback_stats() == {}, "independent per engine"
        cache._reset_ctx_fallback_stats()
        assert cache.ctx_fallback_stats() == {}

    def test_ctx_fallback_read_failure_counted_on_rolling(self, monkeypatch, tmp_path):
        """A ctx read that returns nothing usable must bump read_failure and
        fall back to the legacy per-expert resolution; output stays finite."""
        import mlx.core as mx

        from omlx.patches.expert_streaming import streaming_switch as ss

        glu = _quant_glu(tmp_path, n_experts=3)
        glu._cache._reset_ctx_fallback_stats()
        monkeypatch.setattr(
            ss.StreamingQuantizedSwitchLinear,
            "_load_expert_bank_np",
            lambda self, ids: None,
        )
        monkeypatch.setattr(
            ss.StreamingQuantizedSwitchLinear,
            "_load_expert_bank_np_full",
            lambda self, ids: None,
        )
        x = mx.random.normal((1, 130, 128)).astype(mx.float32)
        indices = mx.array([i % 3 for i in range(130)], dtype=mx.int32)
        out = glu(x, indices)
        mx.eval(out)
        stats = glu._cache.ctx_fallback_stats()
        assert stats.get("read_failure", 0) >= 1, stats
        assert bool(mx.all(mx.isfinite(out)))

    def test_ctx_fallback_union_declines_bank_too_large(self, monkeypatch, tmp_path):
        """Union declines a decode call whose bank set exceeds the cap; the
        linears fall back per expert and count bank_too_large."""
        import mlx.core as mx

        from omlx.patches.expert_streaming import streaming_switch as ss

        glu = _quant_glu(tmp_path, n_experts=3)
        glu._cache._reset_ctx_fallback_stats()
        monkeypatch.setattr(ss, "_CTX_UNION_MAX_BYTES", 1)
        x = mx.random.normal((1, 8, 128)).astype(mx.float32)
        indices = mx.array([0, 1, 2, 0, 1, 2, 0, 1], dtype=mx.int32)
        out = glu(x, indices)
        mx.eval(out)
        stats = glu._cache.ctx_fallback_stats()
        assert stats.get("bank_too_large", 0) >= 3, stats
        assert bool(mx.all(mx.isfinite(out)))

    def test_ctx_fallback_dict_backing_counted(self, monkeypatch, tmp_path):
        """A projection without a bank reader means no context is built;
        that degradation is counted once per GLU call."""
        import mlx.core as mx

        from omlx.patches.expert_streaming import streaming_switch as ss

        # Remove the bank reader from the CLASS: hasattr() reads True for a
        # None value, so a real removal is what the GLU check observes.
        monkeypatch.delattr(
            ss.StreamingQuantizedSwitchLinear, "_load_expert_bank_np"
        )
        glu = _quant_glu(tmp_path, n_experts=3)
        glu._cache._reset_ctx_fallback_stats()
        x = mx.random.normal((1, 8, 128)).astype(mx.float32)
        indices = mx.array([0, 1, 2, 0, 1, 2, 0, 1], dtype=mx.int32)
        out = glu(x, indices)
        mx.eval(out)
        stats = glu._cache.ctx_fallback_stats()
        assert stats.get("dict_backing", 0) == 1, stats
        assert bool(mx.all(mx.isfinite(out)))

    def test_ctx_fallback_preserves_bit_exactness(self, monkeypatch, tmp_path):
        """Fase L1 gate: a forced ctx read failure must fall back to the
        legacy resolution with byte-identical output — the token-ID gate
        survives even when the fast path cannot serve."""
        import mlx.core as mx

        from omlx.patches.expert_streaming import streaming_switch as ss

        x = mx.random.normal((1, 130, 128)).astype(mx.float32)
        indices = mx.array([i % 3 for i in range(130)], dtype=mx.int32)

        def run():
            glu = _quant_glu(tmp_path, n_experts=3)
            out = glu(x, indices)
            mx.eval(out)
            return out

        out_ok = run()
        bits_ok = np.ascontiguousarray(out_ok).view(np.uint32).reshape(-1)
        with monkeypatch.context() as m:
            m.setattr(
                ss.StreamingQuantizedSwitchLinear,
                "_load_expert_bank_np",
                lambda self, ids: None,
            )
            m.setattr(
                ss.StreamingQuantizedSwitchLinear,
                "_load_expert_bank_np_full",
                lambda self, ids: None,
            )
            out_fb = run()
        bits_fb = np.ascontiguousarray(out_fb).view(np.uint32).reshape(-1)
        assert np.array_equal(bits_ok, bits_fb), "fallback changed output bits"

    def test_ctx_memtrace_records_explicit_fields(self, monkeypatch, tmp_path):
        """Both ctx modes record the Fase L1 explicit frame: ctx_mode,
        positions, ctx_bank_bytes, ctx_inflight_bytes, ctx_prefetch_count."""
        import mlx.core as mx

        from omlx.patches.expert_streaming import streaming_switch as ss

        class _FakeTrace:
            enabled = True

            def __init__(self):
                self.rows = []

            def record(self, event, **fields):
                self.rows.append((event, fields))

        fake = _FakeTrace()
        monkeypatch.setattr(ss, "memtrace", fake)
        glu = _quant_glu(tmp_path, n_experts=3)

        def call(n_positions, cycled):
            x = mx.random.normal((1, n_positions, 128)).astype(mx.float32)
            if cycled:
                indices = mx.array([i % 3 for i in range(n_positions)], dtype=mx.int32)
            else:
                indices = mx.array([i % 3 for i in range(n_positions)], dtype=mx.int32)
            out = glu(x, indices)
            mx.eval(out)

        call(130, True)  # rolling
        call(8, False)  # union
        modes_seen = set()
        for event, fields in fake.rows:
            if event != "ctx.ensure.exit":
                continue
            for key in (
                "ctx_mode",
                "positions",
                "ctx_bank_bytes",
                "ctx_inflight_bytes",
                "ctx_prefetch_count",
            ):
                assert key in fields, (event, key, fields)
            modes_seen.add(fields["ctx_mode"])
        assert modes_seen == {"rolling", "union"}, modes_seen

    def test_memtrace_event_aggregates(self):
        """MemTracer aggregates tracked numeric fields per event, so a run
        reports mean/max positions, bank bytes and prefetch counts without
        JSONL post-processing."""
        from omlx.patches.expert_streaming.memtrace import MemTracer

        tr = MemTracer(path=None, sample_every=1)
        tr.record(
            "ctx.ensure.exit",
            layer=0,
            positions=8,
            ctx_bank_bytes=1000,
            ctx_prefetch_count=2,
            ctx_inflight_bytes=700,
        )
        tr.record(
            "ctx.ensure.exit",
            layer=0,
            positions=64,
            ctx_bank_bytes=5000,
            ctx_prefetch_count=0,
            ctx_inflight_bytes=0,
        )
        s = tr.summary()
        agg = s["event_aggregates"]["ctx.ensure.exit"]
        assert agg["positions"]["mean"] == 36.0, agg
        assert agg["positions"]["max"] == 64.0, agg
        assert agg["ctx_prefetch_count"]["max"] == 2.0, agg
        assert agg["ctx_bank_bytes"]["mean"] == 3000.0, agg

class TestFaseL2RegimePins:
    """Fase L2: regime-split pin profiles, fingerprint gating, proportional
    per-layer budgets with page dedupe, and tier-aware pin resolution."""

    @staticmethod
    def _pin_backing(per_expert: int = 1024):
        """Fake backing with reader identity per expert and pin recording."""
        class _Reader:
            def __init__(self, name, width):
                self.path = f"stub-{name}"
                self.width = width

            def expert_byte_range(self, key, eid):
                return int(eid) * self.width, (int(eid) + 1) * self.width

        class _B:
            def __init__(self):
                self.hot = _Reader("hot", per_expert)
                self.cold = _Reader("cold", per_expert)
                self.pinned_calls: list[tuple] = []
                self.pinned_bytes = 0
                self._pinned: set = set()

            def _reader_for_key(self, key, expert_id=None):
                return self.hot if (expert_id or 0) < 8 else self.cold

            def pin_expert(self, key, expert_id):
                r = self._reader_for_key(key, expert_id)
                self.pinned_calls.append((r.path, key, int(expert_id)))
                self._pinned.add((r.path, key, int(expert_id)))
                self.pinned_bytes += r.width
                return r.width

            @property
            def pinned_count(self):
                return len(self._pinned)

        return _B()

    @staticmethod
    def _lin(layer: int, proj: str):
        from types import SimpleNamespace

        base = f"model.layers.{layer}.mlp.switch_mlp.{proj}"
        return SimpleNamespace(
            stacked_weight_key=f"{base}.weight",
            stacked_scales_key=f"{base}.scales",
            stacked_biases_key=None,
            backing=None,
        )

    def test_pin_profile_v2_fingerprint_gating(self, tmp_path):
        """A v2 profile applies only when the model fingerprint matches;
        mismatch logs and ignores the profile (never a silent apply)."""
        from collections import Counter

        from omlx.patches.expert_streaming import warmer as warmer_mod

        profile = str(tmp_path / "expert_pin_profile.json")
        linears = {0: [self._lin(0, "gate_proj"), self._lin(0, "down_proj")]}
        backing = self._pin_backing()
        ctl = warmer_mod.PinController(
            linears, backing, per_expert_bytes=1024, profile_path=profile,
            model_fingerprint={"model": "A", "profile_format": 2},
            packing="oQ4e4 b-gs64",
        )
        ctl.freq = {0: Counter({3: 9, 5: 2})}
        ctl.save_profile()
        import json

        data = json.loads(open(profile).read())
        assert data["version"] == 2
        assert data["regimes"]["decode"]["freq"]["0"] == [[3, 9], [5, 2]]
        assert data["regimes"]["prefill"] == {"freq": {}}

        # Matching fingerprint: loads and latches the pin pass.
        ok = warmer_mod.PinController(
            linears, backing, per_expert_bytes=1024, profile_path=profile,
            model_fingerprint={"model": "A", "profile_format": 2},
        )
        assert ok.freq[0] == Counter({3: 9, 5: 2})
        assert ok.fingerprint_match is True
        assert ok.pinned is True

        # Mismatched fingerprint: profile ignored, no pin, no freq.
        bad = warmer_mod.PinController(
            linears, backing, per_expert_bytes=1024, profile_path=profile,
            model_fingerprint={"model": "B", "profile_format": 2},
        )
        assert bad.freq == {}
        assert bad.fingerprint_match is False
        assert bad.pinned is False

    def test_pin_regime_selects_prefill_hot_set(self, tmp_path):
        """Arm E: the PREfill regime drives the pin selection — the pinner
        wires the prefill-learned experts, not the decode ones."""
        from collections import Counter

        from omlx.patches.expert_streaming import warmer as warmer_mod

        linears = {0: [self._lin(0, "gate_proj"), self._lin(0, "down_proj")]}
        backing = self._pin_backing()
        ctl = warmer_mod.PinController(
            linears, backing, per_expert_bytes=1024, profile_path=None,
            pin_regime="prefill",
        )
        ctl.on_layer_plan(0, [2, 3], 4096)  # prefill rows only
        ctl.on_layer_plan(0, [6], 8)  # decode row: expert 6 hot in decode
        ctl._pin_all(sync=True)
        pinned = {e for _r, _k, e in backing.pinned_calls}
        assert pinned == {2, 3}, pinned  # prefill hot set, not decode 6

    def test_pin_budget_respected_after_page_dedupe(self):
        """The unique page set of the chosen experts must fit the budget —
        page-aligned neighbors share pages and never double-charge."""
        from collections import Counter

        from omlx.patches.expert_streaming import warmer as warmer_mod
        from omlx.patches.expert_streaming.shard_bank import _PAGE_SIZE

        linears = {0: [self._lin(0, "gate_proj"), self._lin(0, "down_proj")]}
        backing = self._pin_backing(per_expert=1024)
        budget = 2 * _PAGE_SIZE  # 32 KiB on 16 KiB pages
        ctl = warmer_mod.PinController(
            linears, backing, per_expert_bytes=1024, budget_bytes=budget,
        )
        # 30 hot experts per layer -> selection far above the budget; the
        # trim pass must drop the least-hot until the unique pages fit.
        ctl.freq = {
            0: Counter({e: 1000 - e for e in range(30)}),
        }
        ctl._pin_all(sync=True)
        assert ctl.pinned_pages_estimate * _PAGE_SIZE <= budget, (
            ctl.pinned_pages_estimate,
            budget,
        )
        assert backing.pinned_bytes <= budget, backing.pinned_bytes
        assert ctl.pin_jobs > 0
        assert ctl.pin_load_time_ms >= 0.0

    def test_pin_expert_tier_aware_reader(self):
        """Cold experts pin through the COLD reader; hot experts through the
        source reader — the HOBBIT split never crosses files inside one pin."""
        from collections import Counter

        from omlx.patches.expert_streaming import warmer as warmer_mod

        linears = {0: [self._lin(0, "gate_proj")]}
        backing = self._pin_backing()
        ctl = warmer_mod.PinController(
            linears, backing, per_expert_bytes=1024,
        )
        ctl.freq = {0: Counter({3: 9, 9: 9})}  # 3 hot (source), 9 cold (tier)
        ctl._pin_all(sync=True)
        by_eid = {e: r for r, _k, e in backing.pinned_calls}
        assert by_eid[3] == "stub-hot", by_eid
        assert by_eid[9] == "stub-cold", by_eid

    def test_pin_sync_applies_before_first_decode(self, tmp_path, monkeypatch):
        """PIN_SYNC=1: the mlock pass completes inside engine load — after
        construction the backing already reports the wired set."""
        from collections import Counter

        from omlx.patches.expert_streaming import warmer as warmer_mod

        profile = str(tmp_path / "expert_pin_profile.json")
        linears = {0: [self._lin(0, "gate_proj")]}
        backing = self._pin_backing()
        ctl = warmer_mod.PinController(
            linears, backing, per_expert_bytes=1024, profile_path=profile,
        )
        ctl.freq = {0: Counter({3: 9})}
        ctl.save_profile()
        monkeypatch.setattr(warmer_mod, "PIN_SYNC_ENABLED", True)
        ctl2 = warmer_mod.PinController(
            linears, backing, per_expert_bytes=1024, profile_path=profile,
        )
        assert ctl2.pinned is True
        assert backing.pinned_count > 0
        assert ctl2.pin_load_time_ms > 0.0

    def test_hot_set_loader_reads_v2_decode_regime(self, tmp_path):
        """HOBBIT reads the DECODE regime of a v2 profile; v1 legacy freq
        keeps working, and a decode-less v2 profile yields no hot set."""
        import json

        from omlx.patches.expert_streaming.shard_bank import load_hot_set_from_profile

        p = tmp_path / "expert_pin_profile.json"
        p.write_text(
            json.dumps(
                {
                    "version": 2,
                    "regimes": {
                        "decode": {"freq": {"0": [[5, 9], [2, 4]]}},
                        "prefill": {"freq": {"0": [[7, 9]]}},
                    },
                    "freq": {"0": [[5, 9], [2, 4]]},
                }
            )
        )
        hot = load_hot_set_from_profile(str(p), 0.5, num_experts=4)
        assert hot["layer_0"] == {5, 2}, hot  # decode regime: 2 of 4 experts
        hot2 = load_hot_set_from_profile(str(p), 0.5)
        # Legacy 2-arg: denominator is the record count (2) -> 1 hot expert.
        assert hot2["layer_0"] == {5}
        # v1 fallback: legacy top-level freq.
        p.write_text(json.dumps({"freq": {"0": [[9, 9]]}}))
        hot3 = load_hot_set_from_profile(str(p), 1.0, num_experts=4)
        assert hot3["layer_0"] == {9}, hot3


# ---------------------------------------------------------------------------
# Fase A — hygiene post-review: phase attribution, fail-high gates, honest
# window vocabulary, pool owners, overhead probe. Tests are WRITTEN here;
# execution happens in Fase B when the machine frees up.
# ---------------------------------------------------------------------------


class TestFaseA1LegacyPhaseAttribution:
    """Fase A1: the bench's phase helpers must attribute the FIRST chat of
    the legacy path to prefill and only the second chat to decode, with
    memtrace agreeing at the same boundary."""

    def test_legacy_path_attributes_prefill_to_prefill(self):
        """The legacy flow is EXACTLY the bench's else-branch sequence:
        open prefill, first chat, switch to decode, second chat, close.
        The first chat must observe prefill; the second, decode."""
        from bench.bench_expert_streaming import (
            close_phase,
            open_phase,
            switch_phase,
        )

        tel = _FakeRelayTel()
        mt6 = _FakeRelayMt6()
        phase_during_chat = []

        def first_chat():
            phase_during_chat.append(tel.active_phase)

        def second_chat():
            phase_during_chat.append(tel.active_phase)

        open_phase(tel, mt6, "prefill", "bench-entry")
        first_chat()
        switch_phase(tel, mt6, "decode", "bench-entry")
        second_chat()
        close_phase(tel)

        assert phase_during_chat == ["prefill", "decode"], phase_during_chat
        assert tel.events == [
            ("begin", "prefill"),
            ("end", "prefill"),
            ("begin", "decode"),
            ("end", "decode"),
        ], tel.events
        assert mt6.contexts == ["prefill", "decode"], mt6.contexts

    def test_single_request_switches_phase_at_first_token(self):
        from bench.bench_expert_streaming import (
            close_phase,
            open_phase,
            switch_phase,
        )

        tel = _FakeRelayTel()
        mt6 = _FakeRelayMt6()
        open_phase(tel, mt6, "prefill", "bench-entry")
        assert tel.active_phase == "prefill"
        switch_phase(tel, mt6, "decode", "bench-entry")
        assert tel.active_phase == "decode"
        assert mt6.last_context == "decode"
        close_phase(tel)
        assert tel.active_phase is None

    def test_disabled_telemetry_does_not_disturb_the_flow(self):
        from bench.bench_expert_streaming import close_phase, open_phase

        tel = _FakeRelayTel()
        tel.enabled = False
        open_phase(tel, None, "prefill", "e")
        close_phase(tel)
        assert tel.active_phase is None and tel.events == []


class _FakeRelayTel:
    """Fake read telemetry that records the active phase + event order."""

    enabled = True

    def __init__(self):
        self.active_phase = None
        self.events = []

    def begin_phase(self, phase, request_id=None, engine_id=None):
        self.active_phase = phase
        self.events.append(("begin", phase))

    def end_phase(self):
        self.events.append(("end", self.active_phase))
        self.active_phase = None
        return {}


class _FakeRelayMt6:
    """Fake memtracer recording every set_context call."""

    def __init__(self):
        self.contexts = []
        self.last_context = None

    def set_context(self, phase, request_id=None, engine_id=None):
        self.contexts.append(phase)
        self.last_context = phase


class TestFaseA2EffectiveConfigGate:
    """Fase A2: fail-high — a null or incomplete effective_config aborts
    under --gate-tokens BEFORE an artifact is written; outside gate mode
    it warns loudly instead."""

    @staticmethod
    def _complete_cfg():
        return {
            "git_sha": "abc123",
            "single_request": True,
            "decode_tokens": 48,
            "chunk_schedule": {"reference_step": 1024},
            "budget_gib": 0.0,
            "ctx_mode_policy": "hybrid",
            "decode_union_rows": 64,
            "ctx_ahead": 3,
            "expert_qd": 16,
            "run_qd": 16,
            "prefill_qd": 0,
            "run_merge_gap": 0,
            "ra_enabled": True,
            "stash_enabled": False,
            "pins_enabled": False,
            "profile_enabled": False,
            "memtrace_enabled": False,
            "read_sampling_mode": "off",
            "cache_cool_protocol": "warm-page-cache",
            "active_engines": 1,
        }

    def test_bench_gate_tokens_fails_without_effective_config(self):
        from bench.bench_expert_streaming import assert_effective_config_complete

        with pytest.raises(SystemExit):
            assert_effective_config_complete(None, gate=True)
        with pytest.raises(SystemExit):
            assert_effective_config_complete({"git_sha": "x"}, gate=True)

    def test_result_contains_effective_config_fails_when_incomplete_in_gate_mode(
        self,
    ):
        """A6.7: gated results must carry the COMPLETE block — dropping one
        critical field is as fatal as a null block."""
        from bench.bench_expert_streaming import assert_effective_config_complete

        cfg = self._complete_cfg()
        assert_effective_config_complete(cfg, gate=True)  # must not raise
        del cfg["chunk_schedule"]
        with pytest.raises(SystemExit):
            assert_effective_config_complete(cfg, gate=True)

    def test_non_gate_mode_warns_but_continues(self, capsys):
        from bench.bench_expert_streaming import assert_effective_config_complete

        assert_effective_config_complete(None, gate=False)  # must not raise
        captured = capsys.readouterr()
        assert "WARNING" in captured.out


class TestFaseA3WindowMetricSplit:
    """Fase A3: the renamed vocabulary decomposes the window honestly —
    worker_start_delay and read_duration split each run; window_wait
    covers the caller's blocks; last_future_wait isolates the tail."""

    @staticmethod
    def _backing(tmp: Path):
        from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore

        key = "model.layers.0.mlp.switch_mlp.gate_proj.weight"
        per = 4 * 4 * 2
        _write_shard_filled(tmp / "model.safetensors", {key: ((16, 4, 4), "BF16")})
        (tmp / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {key: "model.safetensors"}})
        )
        backing = ExpertBackingStore(tmp)
        backing.read_telemetry.enabled = True
        return backing, key, per

    def test_worker_start_delay_vs_read_duration_split(self, tmp_path):
        """A fake reader with a known sleep: the sleep lands in
        read_duration_us (SSD/kernel time) while worker_start_delay_us
        stays small — the two never merge."""
        import time

        with tempfile.TemporaryDirectory() as td:
            backing, key, per = self._backing(Path(td))
            reader = backing._reader_for_key(key)
            orig = reader._read_into

            def slow(off, buf):
                time.sleep(0.02)
                return orig(off, buf)

            reader._read_into = slow
            out = np.empty((4, per), dtype=np.uint8)
            assert backing.read_expert_into([(key, [3, 4, 7, 8])], [out])
            st = backing.read_telemetry.summary()["lifetime"]["stages_us"]
            rd = st["read_duration_us"]
            ws = st["worker_start_delay_us"]
            assert rd["count"] == 2 and rd["p50"] >= 18000, rd  # ~20 ms per run
            assert ws["count"] == 2, ws
            assert ws["p50"] is None or ws["p50"] < rd["p50"], (ws, rd)

    def test_last_future_wait_isolates_tail(self, tmp_path, monkeypatch):
        """A slow FIRST run inflates window_wait_us but NOT the tail: the
        final run's wait is the true tail of the burst, not the span.

        The read pool is pinned to a single worker deliberately. This asserts
        ATTRIBUTION (which stage a caller-side block is charged to), not
        concurrency. On the process-wide 16-worker singleton all four runs
        execute in parallel, so the slow run is the one still in flight when
        the window shrinks to one — at which point last_future_wait_us
        legitimately absorbs it. The assertion then passed or failed depending
        on how warm the singleton happened to be: green alone, red inside a
        full-suite run. Serializing the reads makes the documented scenario
        real — the first run blocks, the last one is fast.
        """
        import time
        from concurrent.futures import ThreadPoolExecutor

        from omlx.patches.expert_streaming import shard_bank

        serial = ThreadPoolExecutor(max_workers=1)
        monkeypatch.setattr(shard_bank, "_run_io_pool", lambda: serial)
        try:
            with tempfile.TemporaryDirectory() as td:
                backing, key, per = self._backing(Path(td))
                reader = backing._reader_for_key(key)
                orig = reader._read_into
                state = {"n": 0}

                def slow_first(off, buf):
                    state["n"] += 1
                    if state["n"] == 1:
                        time.sleep(0.05)
                    return orig(off, buf)

                reader._read_into = slow_first
                out = np.empty((8, per), dtype=np.uint8)
                # 4 separate runs. Ids must stay inside the fixture's
                # 16-expert bank — read_expert_into rejects out-of-range ids
                # outright (shard_bank bounds check), which would mask the
                # tail/window split this test is actually about.
                eids = [0, 1, 4, 5, 8, 9, 12, 13]
                assert backing.read_expert_into([(key, eids)], [out])
                st = backing.read_telemetry.summary()["lifetime"]["stages_us"]
                tail = st["last_future_wait_us"]
                ww = st["window_wait_us"]
                assert tail["count"] == 1 and tail["p50"] is not None, tail
                assert tail["p50"] < 40000, tail  # the FINAL run is fast
                assert ww["count"] >= 1 and ww["sum"] >= 40000, ww  # the slow run
                assert tail["sum"] < ww["sum"], (tail, ww)
        finally:
            serial.shutdown(wait=False)

    def test_read_stats_stage_keys_frozen(self):
        """A6: the canonical stage-key set of read_stats is a snapshot —
        accidental renames outside a reviewed transition are caught here
        (the comparator refuses mixed vocabularies)."""
        from omlx.patches.expert_streaming.shard_bank import _READ_METRICS

        assert tuple(_READ_METRICS) == (
            "component_e2e_us",
            "reader_resolve_us",
            "plan_us",
            "buffer_alloc_us",
            "worker_start_delay_us",
            "read_duration_us",
            "window_wait_us",
            "last_future_wait_us",
            "scatter_us",
            "fallback_us",
        )


class TestFaseA4PoolOwnerAttribution:
    """Fase A4: the process-wide pool telemetry can attribute activity to
    one owner; per-owner counters reconcile with the process totals."""

    def test_pool_telemetry_per_owner(self):
        import time

        from omlx.patches.expert_streaming.shard_bank import RunPoolTelemetry

        ptel = RunPoolTelemetry()
        ts = time.perf_counter_ns()
        for owner in ("A", "B"):
            fn = ptel.wrap(ts, lambda: 1, owner=owner)
            ptel.submit_notice(owner=owner)
            assert fn() == 1
        bad = ptel.wrap(ts, lambda: 1 / 0, owner="A")
        ptel.submit_notice(owner="A")
        with pytest.raises(ZeroDivisionError):
            bad()

        snap = ptel.snapshot()
        snap_a = ptel.snapshot(owner="A")
        snap_b = ptel.snapshot(owner="B")
        assert snap["submitted"] == 3 and snap["completed"] == 2, snap
        assert snap_a["submitted"] == 2 and snap_a["completed"] == 1
        assert snap_a["failed"] == 1
        assert snap_b["submitted"] == 1 and snap_b["completed"] == 1
        # Reconciliation invariant: per-owner counters sum to the pool.
        assert snap_a["submitted"] + snap_b["submitted"] == snap["submitted"]
        assert snap_a["completed"] + snap_b["completed"] == snap["completed"]
        assert snap_a["failed"] + snap_b["failed"] == snap["failed"]

        # Deltas filter by owner.
        before_a = snap_a
        fn = ptel.wrap(ts, lambda: 2, owner="A")
        ptel.submit_notice(owner="A")
        assert fn() == 2
        d_a = ptel.delta(before_a, owner="A")
        assert d_a["submitted"] == 1 and d_a["completed"] == 1 and d_a["failed"] == 0
        d_all = ptel.delta(
            {"submitted": 3, "started": 3, "completed": 2, "failed": 1},
            owner=None,
        )
        assert d_all["submitted"] == 1 and d_all["failed"] == 0
        assert d_all["completed"] == 1

    def test_read_expert_into_attributes_pool_tasks_to_owner(self, tmp_path):
        """End-to-end: read_expert_into tags its runs with id(backing), so
        the owner-filtered snapshot reflects ONLY this backing."""
        from concurrent.futures import ThreadPoolExecutor

        from omlx.patches.expert_streaming import shard_bank as sb_mod

        with tempfile.TemporaryDirectory() as td:
            backing, key, per = TestFaseA3WindowMetricSplit._backing(Path(td))
            old = sb_mod._RUN_IO_POOL_SINGLETON
            pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="a4-owner")
            pool.telemetry = sb_mod.RunPoolTelemetry()
            try:
                sb_mod._RUN_IO_POOL_SINGLETON = pool
                out = np.empty((4, per), dtype=np.uint8)
                assert backing.read_expert_into([(key, [3, 4, 7, 8])], [out])
            finally:
                sb_mod._RUN_IO_POOL_SINGLETON = old
                pool.shutdown(wait=True, cancel_futures=True)
            snap_all = pool.telemetry.snapshot()
            snap_owner = pool.telemetry.snapshot(owner=id(backing))
            assert snap_all["submitted"] == 2 and snap_all["completed"] == 2
            assert snap_owner["submitted"] == 2, snap_owner
            assert snap_owner["completed"] == 2


class TestFaseA5OverheadProbe:
    """Fase A5: the synthetic probe bounds the per-call instrumentation
    cost without a model; the PROFILE A/B later fills the result field."""

    def test_probe_reports_sane_per_call_costs(self):
        from bench.overhead_probe import _measure

        rep = _measure(300)
        assert rep["calls"] == 300
        assert rep["stage_metrics"] >= 8
        for key in (
            "record_call_us_per_call",
            "summary_us_per_snapshot",
            "stages_dict_build_us_per_call",
            "pool_wrap_pair_us_per_run",
            "estimated_read_path_overhead_us_per_component",
        ):
            assert rep[key] > 0.0, key


class TestStreamingWeightedSumFastPath:
    """The MoE weighted-sum must reach the 0.6.4 native kernel when present
    (it used to import a ``.kernels.fast`` package that never existed, so the
    fast path silently never engaged) and fall back to scatter-unsort when
    the kernel raises or is missing."""

    @staticmethod
    def _make_split_glu():
        import mlx.core as mx

        from omlx.patches.expert_streaming.streaming_switch import (
            ExpertLRUCache,
            StreamingSwitchGLU,
            StreamingSwitchLinear,
        )

        experts, moe_hidden, hidden = 4, 6, 8
        cache = ExpertLRUCache(1 << 24, 4096, num_layers=1)
        backing = {
            (0, "gate_proj"): mx.arange(experts * moe_hidden * hidden, dtype=mx.float32).reshape(
                experts, moe_hidden, hidden
            )
            / 100.0,
            (0, "up_proj"): mx.arange(experts * moe_hidden * hidden, dtype=mx.float32).reshape(
                experts, moe_hidden, hidden
            )
            / 200.0,
            (0, "down_proj"): mx.arange(experts * hidden * moe_hidden, dtype=mx.float32).reshape(
                experts, hidden, moe_hidden
            )
            / 300.0,
        }
        glu = StreamingSwitchGLU(
            input_dims=hidden,
            hidden_dims=moe_hidden,
            num_experts=experts,
            layer_idx=0,
            backing=backing,
            cache=cache,
            quantized=False,
            activation=lambda up, gate: up + gate,
        )
        for name, out_dim, in_dim in [
            ("gate_proj", moe_hidden, hidden),
            ("up_proj", moe_hidden, hidden),
            ("down_proj", hidden, moe_hidden),
        ]:
            setattr(
                glu,
                name,
                StreamingSwitchLinear(
                    layer_idx=0,
                    proj_name=name,
                    stacked_key=f"k.{name}",
                    num_experts=experts,
                    input_dims=in_dim,
                    output_dims=out_dim,
                    backing=backing,
                    cache=cache,
                ),
            )
        return glu

    def test_weighted_sum_routes_to_native_kernel(self, monkeypatch):
        import mlx.core as mx

        import omlx.custom_kernels.glm_moe_dsa.fast as glm_fast

        assert hasattr(glm_fast, "glm_moe_weighted_sum")
        sentinel = object()
        seen = {}

        def spy(x_sorted, inv_order, scores):
            seen["shapes"] = (tuple(x_sorted.shape), tuple(scores.shape))
            return sentinel

        monkeypatch.setattr(glm_fast, "glm_moe_weighted_sum", spy)
        glu = self._make_split_glu()
        # >= 64 routed rows so the gather-sort path (do_sort) engages.
        x = mx.random.uniform(shape=(1, 32, 8))
        idx = mx.broadcast_to(mx.array([0, 1, 2, 3]), (1, 32, 4))
        scores = mx.ones((1, 32, 4), dtype=mx.float32) / 4.0
        out = glu(x, idx, scores=scores, weighted_sum=True)
        assert out is sentinel
        assert seen["shapes"][1] == (1, 32, 4)

    def test_weighted_sum_falls_back_when_kernel_raises(self, monkeypatch):
        import mlx.core as mx

        import omlx.custom_kernels.glm_moe_dsa.fast as glm_fast

        def boom(x_sorted, inv_order, scores):
            raise RuntimeError("native weighted-sum unavailable")

        monkeypatch.setattr(glm_fast, "glm_moe_weighted_sum", boom)
        glu = self._make_split_glu()
        x = mx.random.uniform(shape=(1, 32, 8))
        idx = mx.broadcast_to(mx.array([0, 1, 2, 3]), (1, 32, 4))
        scores = mx.ones((1, 32, 4), dtype=mx.float32) / 4.0
        out = glu(x, idx, scores=scores, weighted_sum=True)
        mx.eval(out)
        # Fallback keeps the top-k axis (scatter-unsort + squeeze(-2)).
        assert tuple(out.shape) == (1, 32, 4, 8)


class TestProposalCounters:
    """Fase 0: cross-layer proposal plumbing (predicted vs observed).

    The Fase 1 predictor records proposals for layer l+1 while layer l
    runs; the demand path records observations. report() must turn the
    two sets into per-layer recall/precision so a bench can gate the
    predictor on measured accuracy (>=85% bar)."""

    @staticmethod
    def _acc():
        from omlx.patches.expert_streaming.streaming_switch import (
            ProfileAccumulator,
        )

        return ProfileAccumulator(enabled=True)

    def test_cross_layer_recall_and_precision(self):
        acc = self._acc()
        # Asymmetric sizes pin the denominators: 4 proposed, 3 observed,
        # 2 in common -> recall 2/3, precision 2/4 (a recall/precision
        # swap would fail).
        acc.record_predicted(6, [10, 11, 12, 13])
        acc.record_observed(6, [11, 12, 14])
        rep = acc.report()
        per = rep["prediction"]["6"]
        assert per["predicted"] == 4
        assert per["observed"] == 3
        assert per["hit"] == 2
        assert per["recall"] == 2 / 3
        assert per["precision"] == 2 / 4
        tots = rep["prediction_totals"]
        assert tots["predicted"] == 4
        assert tots["observed"] == 3
        assert tots["hit"] == 2
        assert tots["recall"] == 2 / 3

    def test_empty_proposal_side_is_zero_not_nan(self):
        acc = self._acc()
        acc.record_observed(2, [1, 2])
        rep = acc.report()
        assert rep["prediction"]["2"]["recall"] == 0.0
        assert rep["prediction"]["2"]["precision"] == 0.0

    def test_disabled_accumulator_is_zero_cost(self):
        from omlx.patches.expert_streaming.streaming_switch import (
            ProfileAccumulator,
        )

        acc = ProfileAccumulator(enabled=False)
        acc.record_predicted(6, [10, 11])
        acc.record_observed(6, [11])
        assert acc.predicted == {}
        assert acc.observed == {}
        assert acc.report()["prediction_totals"]["hit"] == 0


class TestCachePrior:
    """Fase 3: cache-conditional routing (approximate, opt-in).

    A logit bonus for LRU-resident experts before top-k. Bonus <= 0 or
    an empty/unknown residency must leave routing bit-identical."""

    def test_rerank_promotes_resident(self):
        import mlx.core as mx

        from omlx.patches.expert_streaming.adaptive_topk import (
            rerank_cache_prior,
        )

        # Ascending probs: stock top-2 is {2, 3}; a big bonus on expert 0
        # must flip it into the top-2.
        gates = mx.array([[0.1, 0.2, 0.3, 0.4]], dtype=mx.float32)
        out = rerank_cache_prior(gates, {0}, 5.0)
        mx.eval(out)
        top2 = set(int(v) for v in mx.argpartition(out, kth=-2, axis=-1)[..., -2:].reshape(-1).tolist())
        assert top2 == {0, 3}
        # Still a distribution.
        assert abs(float(mx.sum(out).item()) - 1.0) < 1e-5

    def test_rerank_identity_when_off(self):
        import mlx.core as mx

        from omlx.patches.expert_streaming.adaptive_topk import (
            rerank_cache_prior,
        )

        gates = mx.array([[0.1, 0.2, 0.3, 0.4]], dtype=mx.float32)
        mx.eval(gates)
        assert rerank_cache_prior(gates, {0}, 0.0) is gates
        assert rerank_cache_prior(gates, set(), 5.0) is gates
        assert rerank_cache_prior(gates, {99}, 5.0) is gates

    def test_resident_experts_intersection_and_closed(self):
        from omlx.patches.expert_streaming.adaptive_topk import (
            resident_experts,
        )

        class _Lin:
            def __init__(self, keys):
                self._keys = keys

            def bundle_key(self, e):
                return ("L", e, self._keys)

        class _GLU:
            _num_experts = 4
            gate_proj = _Lin("g")
            up_proj = _Lin("u")
            down_proj = None
            _cache = type("C", (), {"_store": {
                ("L", 1, "g"), ("L", 2, "g"),
                ("L", 1, "u"), ("L", 2, "u"), ("L", 3, "u"),
            }})()

        # Intersection over projections: only {1, 2} are fully resident.
        assert resident_experts(_GLU()) == {1, 2}
        # Fail closed on anything unexpected.
        assert resident_experts(object()) == set()
        assert resident_experts(None) == set()

    def test_safe_float_env(self, monkeypatch):
        import omlx.patches.expert_streaming.adaptive_topk as tk

        monkeypatch.setenv("OMLX_EXPERT_STREAMING_CACHE_PRIOR", "banana")
        assert tk._safe_float_env("OMLX_EXPERT_STREAMING_CACHE_PRIOR", 0.0) == 0.0
        monkeypatch.setenv("OMLX_EXPERT_STREAMING_CACHE_PRIOR", "")
        assert tk._safe_float_env("OMLX_EXPERT_STREAMING_CACHE_PRIOR", 0.0) == 0.0
        monkeypatch.setenv("OMLX_EXPERT_STREAMING_CACHE_PRIOR", "1.5")
        assert tk._safe_float_env("OMLX_EXPERT_STREAMING_CACHE_PRIOR", 0.0) == 1.5

    def test_configure_cache_prior_settings_and_env(self, monkeypatch):
        from omlx.model_settings import ModelSettings
        from omlx.patches.expert_streaming import adaptive_topk as tk

        prev = tk.cache_prior_bonus()
        try:
            # Explicit settings win.
            assert tk.cache_prior_from_settings(ModelSettings(expert_streaming_cache_prior=2.0)) == 2.0
            assert tk.cache_prior_bonus() == 2.0
            # None falls back to env; garbage fails closed to exact.
            monkeypatch.setenv("OMLX_EXPERT_STREAMING_CACHE_PRIOR", "1.5")
            assert tk.cache_prior_from_settings(ModelSettings()) == 1.5
            assert tk.cache_prior_from_settings(ModelSettings(expert_streaming_cache_prior="nope")) == 0.0
            assert tk.cache_prior_from_settings(None) == 1.5
        finally:
            monkeypatch.delenv("OMLX_EXPERT_STREAMING_CACHE_PRIOR", raising=False)
            tk.configure_cache_prior(prev if prev > 0 else None)

    def test_boost_logits_matches_numpy_oracle(self):
        import mlx.core as mx
        import numpy as np

        from omlx.patches.expert_streaming.adaptive_topk import (
            apply_cache_prior_to_logits,
        )

        vals = np.array([[0.5, -1.0, 2.0, 0.0]], dtype=np.float32)
        logits = mx.array(vals)
        out = apply_cache_prior_to_logits(logits, {1, 2}, 3.0)
        mx.eval(out)
        expect = vals.copy()
        expect[0, [1, 2]] += 3.0
        assert np.allclose(np.array(out), expect, atol=1e-6)
        # Identity paths return the same object.
        assert apply_cache_prior_to_logits(logits, {1}, 0.0) is logits
        assert apply_cache_prior_to_logits(logits, set(), 3.0) is logits
        assert apply_cache_prior_to_logits(logits, {99}, 3.0) is logits



def test_slot_bank_bit_exact_and_multitoken_invariant(monkeypatch):
    """Slot-bank spike (docs/spike-slot-bank.md).

    1. Bit-exactness: with the arena enabled, outputs equal the legacy
       path for identical inputs (single- and multi-position calls).
    2. #27861 invariant: a multi-position batch whose positions route to
       DISJOINT expert sets must never alias two uncached experts to one
       slot — outputs equal the per-position reference.
    3. Counters: full-hit calls take the fast path (stack=0 evidence is
       in the profile: load=stack=0 on the second identical call).
    """
    import mlx.core as mx
    import numpy as np

    import omlx.patches.expert_streaming.streaming_switch as ss
    monkeypatch.setattr(ss, "_SLOT_BANK_ENV", True)
    monkeypatch.setattr(ss, "_SLOT_BANK_SLOTS", 8)
    monkeypatch.setattr(ss, "_SLOT_BANK_MAX_POSITIONS", 64)

    E, D_in, D_out = 8, 64, 32
    mx.random.seed(3)
    dense = mx.random.normal((E, D_out, D_in))
    w4, s4, b4 = mx.quantize(dense, group_size=32, bits=4)

    class _RP:
        def __init__(self, shape):
            self.per_shape = tuple(shape)

    class _Reader:
        def __init__(self, shapes):
            self._shapes = shapes

        def _rp_for(self, key):
            return _RP(self._shapes[key])

    class _ReaderBacking:
        """Dict backing WITH the reader protocol the slot arena needs.

        mx.quantize returns (w, s, b) as separate arrays — store each
        component per stacked key, exactly like a real backing reader.
        """

        def __init__(self):
            self.bank = {
                "k.down_proj.weight": w4,
                "k.down_proj.scales": s4,
                "k.down_proj.biases": b4,
            }
            self.shapes = {
                "k.down_proj.weight": tuple(w4.shape[1:]),
                "k.down_proj.scales": tuple(s4.shape[1:]),
                "k.down_proj.biases": tuple(b4.shape[1:]),
            }
            self.dtypes = {
                "k.down_proj.weight": w4.dtype,
                "k.down_proj.scales": s4.dtype,
                "k.down_proj.biases": b4.dtype,
            }

        def _reader_for_key(self, key, expert_id=None):
            return _Reader(self.shapes)

        def load_expert(self, key, expert_id):
            return self.bank[key][int(expert_id)]

    def _make():
        return ss.StreamingQuantizedSwitchLinear(
            layer_idx=0,
            proj_name="down_proj",
            stacked_weight_key="k.down_proj.weight",
            stacked_scales_key="k.down_proj.scales",
            stacked_biases_key="k.down_proj.biases",
            num_experts=E,
            input_dims=D_in,
            output_dims=D_out,
            backing=_ReaderBacking(),
            cache=ss.ExpertLRUCache(1 << 26, 1 << 18, num_layers=1),
            group_size=32,
            bits=4,
            mode="affine",
        )

    rng = np.random.default_rng(11)
    # --- 1: single-position calls, bit-exact vs arena-off twin ---
    lin_on = _make()
    lin_off = _make()
    for trial in range(3):
        T = 4 if trial < 2 else 40
        x = mx.array(rng.standard_normal((T, 1, D_in)).astype(np.float32))
        idx = mx.array(rng.integers(0, E, size=T).astype(np.int32))
        out_ref = lin_off(x, idx)
        out_on = lin_on(x, idx)
        assert out_on.shape == out_ref.shape
        assert mx.array_equal(out_ref, out_on), f"trial {trial} mismatch"

    # --- 2: #27861 invariant: disjoint expert sets across positions ---
    # 8 experts, 8 slots: positions alternate between two disjoint halves.
    # After the first call populates slots, a second call with the OTHER
    # half must still be exact (evictions + inserts under load).
    x2 = mx.array(rng.standard_normal((8, 1, D_in)).astype(np.float32))
    idx_a = mx.array([0, 2, 4, 6, 0, 2, 4, 6], dtype=mx.int32)  # even
    idx_b = mx.array([1, 3, 5, 7, 1, 3, 5, 7], dtype=mx.int32)  # odd
    lin_i = _make()
    out_a1 = lin_i(x2, idx_a)
    out_b = lin_i(x2, idx_b)  # disjoint: all misses, evictions happen
    out_a2 = lin_i(x2, idx_a)  # evicted even set must reload exactly
    ref_a = lin_off(x2, idx_a)
    ref_b = lin_off(x2, idx_b)
    assert mx.array_equal(out_a1, ref_a)
    assert mx.array_equal(out_b, ref_b)
    assert mx.array_equal(out_a2, ref_a)

    # --- 3: repeated-slot gather is exact (same expert twice in batch) ---
    idx_r = mx.array([5, 5, 5, 5, 2, 2, 5, 2], dtype=mx.int32)
    out_r = lin_i(x2, idx_r)
    ref_r = lin_off(x2, idx_r)
    assert mx.array_equal(out_r, ref_r)

    # --- 4: fast-path counters: identical call twice -> full hit ---
    # Use a demand set that FITS the 8 slots alongside what is resident
    # (a 4-expert set): no eviction pressure, second call must be full-hit.
    idx_fit = mx.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=mx.int32)
    lin_f = _make()
    lin_f(x2, idx_fit)
    sb = lin_f._slot_bank()
    assert sb is not None and sb.slots == 8
    h0, m0 = sb.hits, sb.misses
    lin_f(x2, idx_fit)
    assert sb.misses == m0, "identical demand set should be a full hit"
    assert sb.hits > h0

    # --- 5: prefill-shaped call bypasses the arena ---
    xp = mx.array(rng.standard_normal((100, 1, D_in)).astype(np.float32))
    idx_p = mx.array(rng.integers(0, E, size=100).astype(np.int32))
    out_p = lin_i(xp, idx_p)
    ref_p = lin_off(xp, idx_p)
    assert mx.array_equal(out_p, ref_p)
