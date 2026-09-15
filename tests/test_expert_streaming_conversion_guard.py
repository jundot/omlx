"""Conversion guard, cold-tier label validation, prefill pin ordering."""

import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest


def _write_checkpoint(tmp: Path, *, fused: bool = False) -> None:
    config = {
        "model_type": "qwen4_exp",
        "num_hidden_layers": 2,
        "num_experts": 4,
        "hidden_size": 32,
        "moe_intermediate_size": 16,
    }
    (tmp / "config.json").write_text(json.dumps(config))
    import numpy as np

    tensors = {}
    for layer in range(2):
        if fused:
            projs = [
                ("gate_up_proj", (4, 32, 32)),
                ("down_proj", (4, 32, 16)),
            ]
        else:
            projs = [
                ("gate_proj", (4, 16, 32)),
                ("up_proj", (4, 16, 32)),
                ("down_proj", (4, 32, 16)),
            ]
        for proj, shape in projs:
            key = f"model.layers.{layer}.mlp.switch_mlp.{proj}.weight"
            tensors[key] = (shape, "BF16", int(np.prod(shape)) * 2)
    header = {}
    offset = 0
    for k, (shape, dtype, size) in tensors.items():
        header[k] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    hb = json.dumps(header).encode()
    with (tmp / "model.safetensors").open("wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(b"\x00" * offset)
    (tmp / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in tensors}})
    )


def _write_cold_tier(tmp: Path, keys: list[str], *, bits: str) -> None:
    cold = tmp / "expert_cold"
    cold.mkdir()
    header = {"__metadata__": {"omlx_cold_bits": bits, "omlx_cold_group_size": "64"}}
    offset = 0
    for k in keys:
        header[k] = {"dtype": "BF16", "shape": [4], "data_offsets": [offset, offset + 8]}
        offset += 8
    hb = json.dumps(header).encode()
    with (cold / "cold.safetensors").open("wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(b"\x00" * offset)


class TestNoConversionReturnsNoBacking:
    """A backing with zero converted layers must never leave convert."""

    def test_missing_layers(self, tmp_path):
        from omlx.patches.expert_streaming import convert_model_to_streaming

        _write_checkpoint(tmp_path)
        model = SimpleNamespace()  # no layers anywhere
        out_model, backing = convert_model_to_streaming(
            model, str(tmp_path), None, use_file_backing=True
        )
        assert out_model is model
        assert backing is None

    def test_layers_without_moe(self, tmp_path):
        from omlx.patches.expert_streaming import convert_model_to_streaming

        _write_checkpoint(tmp_path)
        model = SimpleNamespace(
            model=SimpleNamespace(layers=[SimpleNamespace(), SimpleNamespace()])
        )
        _, backing = convert_model_to_streaming(
            model, str(tmp_path), None, use_file_backing=True
        )
        assert backing is None


class TestEnsureStreamingBackingOrRaise:
    def test_raises_when_requested_supported_and_nothing_converted(self, tmp_path):
        from omlx.patches.expert_streaming import ensure_streaming_backing_or_raise

        _write_checkpoint(tmp_path)
        with pytest.raises(RuntimeError, match="no MoE layer"):
            ensure_streaming_backing_or_raise(
                SimpleNamespace(), None, requested=True, model_name=str(tmp_path)
            )

    def test_passes_with_real_conversion(self, tmp_path):
        from omlx.patches.expert_streaming import ensure_streaming_backing_or_raise

        _write_checkpoint(tmp_path)
        backing = SimpleNamespace(streaming_converted=2)
        ensure_streaming_backing_or_raise(
            SimpleNamespace(), backing, requested=True, model_name=str(tmp_path)
        )

    def test_passes_when_legacy_wrapped(self, tmp_path):
        from omlx.patches.expert_streaming import ensure_streaming_backing_or_raise

        _write_checkpoint(tmp_path)
        OffloadSwitchGLU = type("OffloadSwitchGLU", (), {})
        model = SimpleNamespace()
        model.named_modules = lambda: [("mlp", OffloadSwitchGLU())]
        ensure_streaming_backing_or_raise(
            model, None, requested=True, model_name=str(tmp_path)
        )

    def test_noop_when_not_requested(self, tmp_path):
        from omlx.patches.expert_streaming import ensure_streaming_backing_or_raise

        _write_checkpoint(tmp_path)
        ensure_streaming_backing_or_raise(
            SimpleNamespace(), None, requested=False, model_name=str(tmp_path)
        )

    def test_noop_when_unsupported(self, tmp_path):
        from omlx.patches.expert_streaming import ensure_streaming_backing_or_raise

        (tmp_path / "config.json").write_text(json.dumps({"model_type": "dense"}))
        ensure_streaming_backing_or_raise(
            SimpleNamespace(), None, requested=True, model_name=str(tmp_path)
        )


class TestColdTierLabelValidation:
    """The omlx_cold_bits check must actually run."""

    def _spy_backing(self, monkeypatch):
        from omlx.patches.expert_streaming import shard_bank

        captured = {}
        orig = shard_bank.ExpertBackingStore

        def _spy(*a, **k):
            captured.update(k)
            return orig(*a, **k)

        monkeypatch.setattr(shard_bank, "ExpertBackingStore", _spy)
        return captured

    def test_mismatched_bits_label_rejected(self, tmp_path, monkeypatch):
        from omlx.patches.expert_streaming import convert_model_to_streaming

        _write_checkpoint(tmp_path)
        keys = [
            f"model.layers.{i}.mlp.switch_mlp.{p}.weight"
            for i in range(2)
            for p in ("gate_proj", "up_proj", "down_proj")
        ]
        _write_cold_tier(tmp_path, keys, bits="3")
        captured = self._spy_backing(monkeypatch)
        _, backing = convert_model_to_streaming(
            SimpleNamespace(),
            str(tmp_path),
            SimpleNamespace(expert_streaming_cold_tier="2"),
            use_file_backing=True,
        )
        try:
            # A 3-bit tier must not serve a "2" request — validation ran.
            assert captured.get("cold_root") is None
        finally:
            if backing is not None:
                backing.close()

    def test_matching_bits_label_accepted(self, tmp_path, monkeypatch):
        from omlx.patches.expert_streaming import convert_model_to_streaming

        _write_checkpoint(tmp_path)
        keys = [
            f"model.layers.{i}.mlp.switch_mlp.{p}.weight"
            for i in range(2)
            for p in ("gate_proj", "up_proj", "down_proj")
        ]
        _write_cold_tier(tmp_path, keys, bits="3")
        captured = self._spy_backing(monkeypatch)
        _, backing = convert_model_to_streaming(
            SimpleNamespace(),
            str(tmp_path),
            SimpleNamespace(expert_streaming_cold_tier="3"),
            use_file_backing=True,
        )
        try:
            assert captured.get("cold_root") is not None
        finally:
            if backing is not None:
                backing.close()


class TestPrefillPinOrdering:
    """The GiB->slots pin must use the reconciled per_expert_bytes."""

    def test_fused_model_pins_reconciled_slots(self, tmp_path):
        import mlx.core as mx

        from omlx.patches.expert_streaming import convert_model_to_streaming

        _write_checkpoint(tmp_path, fused=True)
        layers = []
        for _ in range(2):
            glu = SimpleNamespace(
                gate_up_proj=SimpleNamespace(weight=mx.zeros((4, 32, 32))),
                down_proj=SimpleNamespace(weight=mx.zeros((4, 32, 16))),
            )
            layers.append(SimpleNamespace(mlp=SimpleNamespace(switch_mlp=glu)))
        model = SimpleNamespace(model=SimpleNamespace(layers=layers))
        _, backing = convert_model_to_streaming(
            model,
            str(tmp_path),
            SimpleNamespace(expert_streaming_prefill_budget_gib=0.01),
            use_file_backing=True,
        )
        assert backing is not None
        try:
            cache = backing._streaming_cache
            # Reconciliation must have run (fused: 2 projections, not 3).
            assert cache.per_expert_bytes > 0
            expected = int(0.01 * 1024**3) // cache.per_expert_bytes
            assert cache._prefill_global_cap == expected
        finally:
            backing.close()


class TestCanonicalKillSwitch:
    """OMLX_MOE_EXPERT_OFFLOAD=0 gates the canonical path too."""

    def test_predicate_honors_env(self, monkeypatch):
        from omlx.model_settings import moe_offload_requested

        monkeypatch.setenv("OMLX_MOE_EXPERT_OFFLOAD", "0")
        assert moe_offload_requested({"expert_streaming_enabled": True}) is False
        assert moe_offload_requested({"moe_expert_offload_enabled": True}) is False
        monkeypatch.setenv("OMLX_MOE_EXPERT_OFFLOAD", "1")
        assert moe_offload_requested({"expert_streaming_enabled": True}) is True

    def test_converter_noops_under_env(self, tmp_path, monkeypatch):
        from omlx.patches.expert_streaming import convert_model_to_streaming

        _write_checkpoint(tmp_path)
        monkeypatch.setenv("OMLX_MOE_EXPERT_OFFLOAD", "0")
        model = SimpleNamespace()
        out_model, backing = convert_model_to_streaming(
            model,
            str(tmp_path),
            SimpleNamespace(expert_streaming_enabled=True),
            use_file_backing=True,
        )
        assert out_model is model
        assert backing is None


class TestSpeculativeExclusivity:
    """The canonical key shares the exclusivity contract, discriminated
    by backend — DFlash/VLM-MTP always reject, MTP is allowed where the
    streaming stack actually supports it."""

    def test_canonical_key_dflash_rejected(self):
        from omlx.model_settings import validate_moe_expert_offload

        with pytest.raises(ValueError, match="DFlash"):
            validate_moe_expert_offload(
                {"expert_streaming_enabled": True, "dflash_enabled": True},
                model_type="qwen4_exp",
            )

    def test_canonical_key_vlm_mtp_rejected(self):
        from omlx.model_settings import validate_moe_expert_offload

        with pytest.raises(ValueError, match="VLM"):
            validate_moe_expert_offload(
                {"expert_streaming_enabled": True, "vlm_mtp_enabled": True},
                model_type="qwen4_exp",
            )

    def test_mtp_allowed_on_unified_types(self):
        from omlx.model_settings import validate_moe_expert_offload

        # Unified streaming converts MTP-stage banks too — the combo works.
        validate_moe_expert_offload(
            {"expert_streaming_enabled": True, "mtp_enabled": True},
            model_type="qwen4_exp",
        )

    def test_mtp_allowed_on_deepseek_v41(self):
        from omlx.model_settings import validate_moe_expert_offload

        # Native DSpark verify runs under frozen residency (verify_scope).
        validate_moe_expert_offload(
            {"moe_expert_offload_enabled": True, "mtp_enabled": True},
            model_type="deepseek_v41",
        )

    def test_mtp_rejected_on_legacy_only_types(self):
        from omlx.model_settings import validate_moe_expert_offload

        with pytest.raises(ValueError, match="Lightning MTP"):
            validate_moe_expert_offload(
                {"moe_expert_offload_enabled": True, "mtp_enabled": True},
                model_type="gemma4",
            )



class TestCompatAllowlistUnified:
    """Streaming-owned types are gated by the converter's own structural
    estimate, not a second allowlist."""

    def test_streaming_type_approved_by_estimate(self, tmp_path):
        from omlx.patches.moe_offload_compat import moe_offload_compatibility

        _write_checkpoint(tmp_path)  # qwen4_exp, BF16 switch_mlp banks
        supported, reason = moe_offload_compatibility(str(tmp_path))
        assert supported, reason

    def test_streaming_type_estimate_failure_carries_reason(self, tmp_path):
        from omlx.patches.moe_offload_compat import moe_offload_compatibility

        # qwen3_moe is streaming-owned but NOT in the legacy list: the
        # estimate's reason must surface instead of a bare type rejection.
        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "qwen3_moe"})
        )
        supported, reason = moe_offload_compatibility(str(tmp_path))
        assert not supported
        assert reason  # structural reason, not "not supported for this model type"

    def test_unknown_type_still_rejected(self, tmp_path):
        from omlx.patches.moe_offload_compat import moe_offload_compatibility

        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "totally_dense"})
        )
        supported, reason = moe_offload_compatibility(str(tmp_path))
        assert not supported
        assert "not supported" in reason


class TestStackedKeyValidation:
    """A real weight map turns a missing required key into a
    conversion-time failure instead of a first-fetch failure."""

    def _backing(self, keys):
        return SimpleNamespace(_weight_map={k: "shard.safetensors" for k in keys})


    def test_optional_bias_falls_back(self):
        from omlx.patches.expert_streaming import _resolve_stacked_key

        backing = self._backing({"model.layers.0.mlp.switch_mlp.gate_proj.weight"})
        key = _resolve_stacked_key(
            ["model.layers.0.mlp.switch_mlp.gate_proj.biases"],
            "gate_proj",
            "biases",
            backing,
            "layers.0.",
            required=False,
        )
        assert key == "model.layers.0.mlp.switch_mlp.gate_proj.biases"

    def test_empty_map_keeps_fallback(self):
        from omlx.patches.expert_streaming import _resolve_stacked_key

        key = _resolve_stacked_key(
            ["model.layers.0.mlp.switch_mlp.gate_proj.weight"],
            "gate_proj",
            "weight",
            self._backing({}),
            "layers.0.",
        )
        assert key == "model.layers.0.mlp.switch_mlp.gate_proj.weight"

    def test_exact_candidate_wins(self):
        from omlx.patches.expert_streaming import _resolve_stacked_key

        keys = [
            "model.layers.0.mlp.switch_mlp.gate_proj.weight",
            "other.layers.0.mlp.switch_mlp.gate_proj.weight",
        ]
        key = _resolve_stacked_key(
            ["model.layers.0.mlp.switch_mlp.gate_proj.weight"],
            "gate_proj",
            "weight",
            self._backing(keys),
            "layers.0.",
        )
        assert key == keys[0]


class TestSchedulerSerializer:
    """One wrapper-chain walk feeds both request serialization and the
    prefill guard."""

    def test_backing_found_through_vlm_wrapper(self):
        from omlx.scheduler import _model_uses_expert_streaming, _streaming_backing_of

        backing = SimpleNamespace()
        inner = SimpleNamespace(_expert_streaming_backing=backing)
        wrapper = SimpleNamespace(language_model=None, _vlm_model=inner)
        assert _streaming_backing_of(wrapper) is backing
        assert _model_uses_expert_streaming(wrapper) is True

    def test_no_backing_anywhere(self):
        from omlx.scheduler import _model_uses_expert_streaming

        model = SimpleNamespace(model=SimpleNamespace(layers=[]))
        assert _model_uses_expert_streaming(model) is False

    def test_self_loop_bounded(self):
        from omlx.scheduler import _streaming_backing_of

        node = SimpleNamespace()
        node.model = node  # adapter property returning itself
        assert _streaming_backing_of(node) is None
