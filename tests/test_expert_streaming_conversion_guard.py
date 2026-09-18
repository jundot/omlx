"""Conversion guard, cold-tier label validation, prefill pin ordering."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from streaming_fixtures import closer, write_moe_checkpoint, write_safetensors


def _write_cold_tier(tmp: Path, keys: list[str], *, bits: str) -> None:
    cold = tmp / "expert_cold"
    cold.mkdir()
    write_safetensors(
        cold / "cold.safetensors",
        {k: ([4], "BF16", 8) for k in keys},
        metadata={"omlx_cold_bits": bits, "omlx_cold_group_size": "64"},
    )


class TestNoConversionReturnsNoBacking:
    """A backing with zero converted layers must never leave convert."""

    def test_missing_layers(self, tmp_path):
        from omlx.patches.expert_streaming import convert_model_to_streaming

        write_moe_checkpoint(tmp_path)
        model = SimpleNamespace()  # no layers anywhere
        out_model, backing = convert_model_to_streaming(
            model, str(tmp_path), None, use_file_backing=True
        )
        assert out_model is model
        assert backing is None

    def test_layers_without_moe(self, tmp_path):
        from omlx.patches.expert_streaming import convert_model_to_streaming

        write_moe_checkpoint(tmp_path)
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

        write_moe_checkpoint(tmp_path)
        with pytest.raises(RuntimeError, match="no MoE layer"):
            ensure_streaming_backing_or_raise(
                SimpleNamespace(), None, requested=True, model_name=str(tmp_path)
            )

    def test_passes_with_real_conversion(self, tmp_path):
        from omlx.patches.expert_streaming import ensure_streaming_backing_or_raise

        write_moe_checkpoint(tmp_path)
        backing = SimpleNamespace(streaming_converted=2)
        ensure_streaming_backing_or_raise(
            SimpleNamespace(), backing, requested=True, model_name=str(tmp_path)
        )

    def test_passes_when_legacy_wrapped(self, tmp_path):
        from omlx.patches.expert_streaming import ensure_streaming_backing_or_raise

        write_moe_checkpoint(tmp_path)
        OffloadSwitchGLU = type("OffloadSwitchGLU", (), {})
        model = SimpleNamespace()
        model.named_modules = lambda: [("mlp", OffloadSwitchGLU())]
        ensure_streaming_backing_or_raise(
            model, None, requested=True, model_name=str(tmp_path)
        )

    def test_noop_when_not_requested(self, tmp_path):
        from omlx.patches.expert_streaming import ensure_streaming_backing_or_raise

        write_moe_checkpoint(tmp_path)
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

    @pytest.mark.parametrize(
        ("tier", "accepted"),
        [("2", False), ("3", True)],
        ids=["mismatched_bits_rejected", "matching_bits_accepted"],
    )
    def test_bits_label_validated(self, tmp_path, monkeypatch, tier, accepted, closer):
        from omlx.patches.expert_streaming import convert_model_to_streaming

        write_moe_checkpoint(tmp_path)
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
            SimpleNamespace(expert_streaming_cold_tier=tier),
            use_file_backing=True,
        )
        if backing is not None:
            closer(backing)
        # A 3-bit tier must not serve a "2" request — validation ran.
        assert (captured.get("cold_root") is not None) is accepted


class TestPrefillPinOrdering:
    """The GiB->slots pin must use the reconciled per_expert_bytes."""

    def test_fused_model_pins_reconciled_slots(self, tmp_path, closer):
        import mlx.core as mx

        from omlx.patches.expert_streaming import convert_model_to_streaming

        write_moe_checkpoint(tmp_path, fused=True)
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
        closer(backing)
        cache = backing._streaming_cache
        # Reconciliation must have run (fused: 2 projections, not 3).
        assert cache.per_expert_bytes > 0
        expected = int(0.01 * 1024**3) // cache.per_expert_bytes
        assert cache._prefill_global_cap == expected


class TestReadaheadStamp:
    """The per-model readahead flag rides spec_state to the spec-state
    advisor (the sole F_RDADVISE predictor); the warm hook carries no
    warmer leg anymore."""

    def _model(self):
        import mlx.core as mx

        layers = []
        for _ in range(2):
            glu = SimpleNamespace(
                gate_up_proj=SimpleNamespace(weight=mx.zeros((4, 32, 32))),
                down_proj=SimpleNamespace(weight=mx.zeros((4, 32, 16))),
            )
            layers.append(SimpleNamespace(mlp=SimpleNamespace(switch_mlp=glu)))
        return SimpleNamespace(model=SimpleNamespace(layers=layers))

    def test_readahead_setting_stamps_spec_state(self, tmp_path, closer):
        from omlx.patches.expert_streaming import convert_model_to_streaming

        write_moe_checkpoint(tmp_path, fused=True)
        model = self._model()
        _, backing = convert_model_to_streaming(
            model,
            str(tmp_path),
            SimpleNamespace(expert_streaming_readahead=False),
            use_file_backing=True,
        )
        assert backing is not None
        closer(backing)
        # The advisor reads spec_state.readahead_enabled (None = env
        # default) — the resolved per-model flag must be stamped.
        assert backing.spec_state.readahead_enabled is False
        # Seed defaults on, so the warm/pin hook still attaches —
        # with pin + recorder legs only, never a warmer leg.
        sm = model.model.layers[0].mlp.switch_mlp
        hook = getattr(sm, "_warm_pins", None)
        assert hook is not None
        assert not hasattr(hook, "warmer")
        assert hook.pinner is None
        assert hook.recorder is not None

    def test_readahead_unset_defaults_on(self, tmp_path, closer):
        from omlx.patches.expert_streaming import convert_model_to_streaming
        from omlx.patches.expert_streaming import warmer as _warmer_mod

        write_moe_checkpoint(tmp_path, fused=True)
        _, backing = convert_model_to_streaming(
            self._model(), str(tmp_path), None, use_file_backing=True
        )
        assert backing is not None
        closer(backing)
        # Setting unset -> the env default (RA_ENABLED) is stamped.
        assert backing.spec_state.readahead_enabled is _warmer_mod.RA_ENABLED


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

        write_moe_checkpoint(tmp_path)
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

    @pytest.mark.parametrize(
        ("settings", "model_type", "match"),
        [
            pytest.param(
                {"expert_streaming_enabled": True, "dflash_enabled": True},
                "qwen4_exp",
                "DFlash",
                id="canonical_key_dflash_rejected",
            ),
            pytest.param(
                {"expert_streaming_enabled": True, "vlm_mtp_enabled": True},
                "qwen4_exp",
                "VLM",
                id="canonical_key_vlm_mtp_rejected",
            ),
            # Unified streaming converts MTP-stage banks too — allowed.
            pytest.param(
                {"expert_streaming_enabled": True, "mtp_enabled": True},
                "qwen4_exp",
                None,
                id="mtp_allowed_on_unified_types",
            ),
            # Native DSpark verify runs under frozen residency (verify_scope).
            pytest.param(
                {"moe_expert_offload_enabled": True, "mtp_enabled": True},
                "deepseek_v41",
                None,
                id="mtp_allowed_on_deepseek_v41",
            ),
            pytest.param(
                {"moe_expert_offload_enabled": True, "mtp_enabled": True},
                "gemma4",
                "Lightning MTP",
                id="mtp_rejected_on_legacy_only_types",
            ),
        ],
    )
    def test_speculative_exclusivity(self, settings, model_type, match):
        from omlx.model_settings import validate_moe_expert_offload

        if match is None:
            validate_moe_expert_offload(settings, model_type=model_type)
        else:
            with pytest.raises(ValueError, match=match):
                validate_moe_expert_offload(settings, model_type=model_type)


class TestCompatAllowlistUnified:
    """Streaming-owned types are gated by the converter's own structural
    estimate, not a second allowlist."""

    def test_streaming_type_approved_by_estimate(self, tmp_path):
        from omlx.patches.moe_offload_compat import moe_offload_compatibility

        write_moe_checkpoint(tmp_path)  # qwen4_exp, BF16 switch_mlp banks
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
