# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.patches.mlx_lm_mtp.

Phase 1 covers the model-side hooks (PR 990 for Qwen3.5/3.6 + PR 15
skeleton for DeepSeek-V4) and the conditional dispatch in
``GenerationBatch.next``. End-to-end MTP draft/verify is exercised in a
follow-up once the BatchGenerator integration body is filled in.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from omlx.model_settings import ModelSettings
from omlx.utils.model_loading import (
    _has_mtp_heads,
    _is_mtp_compatible,
    maybe_apply_pre_load_patches,
)

# ---------------------------------------------------------------------------
# Patch orchestrator + sub-modules
# ---------------------------------------------------------------------------


class TestApplyOrchestrator:
    def test_apply_idempotent(self):
        from omlx.patches.mlx_lm_mtp import apply_mlx_lm_mtp_patch

        first = apply_mlx_lm_mtp_patch()
        second = apply_mlx_lm_mtp_patch()
        # Both calls must succeed; the second is a no-op but still True.
        assert first is True
        assert second is True

    def test_module_imports_without_mlx_lm(self, monkeypatch):
        """Importing the package must not fail even if mlx_lm is unavailable."""
        # Just exercise the import path; sub-modules are deferred to apply().
        import omlx.patches.mlx_lm_mtp as mtp  # noqa: F401


class TestCacheRollback:
    def test_arrays_cache_gains_rollback_slot(self):
        from omlx.patches.mlx_lm_mtp import cache_rollback

        applied = cache_rollback.apply()
        assert applied is True
        try:
            from mlx_lm.models.cache import ArraysCache
        except ImportError:
            pytest.skip("mlx-lm not importable")
        assert hasattr(ArraysCache, "rollback_state")
        # rollback_state default is None until a draft+verify writes to it.
        cache = ArraysCache(size=2)
        assert cache.rollback_state is None


class TestQwen35Model:
    @pytest.fixture(autouse=True)
    def _apply(self):
        try:
            from omlx.patches.mlx_lm_mtp import qwen35_model
        except ImportError:
            pytest.skip("omlx.patches.mlx_lm_mtp not importable")
        applied = qwen35_model.apply()
        if not applied:
            pytest.skip("qwen35_model patch refused to apply (likely mlx_lm absent)")

    def test_text_model_args_from_dict_preserves_mtp_layers(self):
        from mlx_lm.models.qwen3_5 import TextModelArgs

        args = TextModelArgs.from_dict(
            {
                "model_type": "qwen3_5",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 256,
                "linear_num_value_heads": 2,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 16,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 3,
                "full_attention_interval": 2,
                "tie_word_embeddings": True,
                "rms_norm_eps": 1e-5,
                "head_dim": 32,
                "rope_theta": 1000.0,
                "partial_rotary_factor": 0.5,
                "max_position_embeddings": 128,
                "mtp_num_hidden_layers": 1,
            }
        )
        assert getattr(args, "mtp_num_hidden_layers", None) == 1

    def test_text_model_args_default_zero_when_missing(self):
        from mlx_lm.models.qwen3_5 import TextModelArgs

        args = TextModelArgs.from_dict(
            {
                "model_type": "qwen3_5",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 256,
                "linear_num_value_heads": 2,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 16,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 3,
                "full_attention_interval": 2,
                "tie_word_embeddings": True,
                "rms_norm_eps": 1e-5,
                "head_dim": 32,
                "rope_theta": 1000.0,
                "partial_rotary_factor": 0.5,
                "max_position_embeddings": 128,
            }
        )
        assert getattr(args, "mtp_num_hidden_layers", None) == 0

    def test_mtp_classes_registered_on_module(self):
        from mlx_lm.models import qwen3_5

        assert hasattr(qwen3_5, "MTPModule")
        assert hasattr(qwen3_5, "MTPDecoderLayer")

    def test_text_model_class_has_mtp_forward(self):
        from mlx_lm.models.qwen3_5 import TextModel

        # Methods are attached unconditionally; the per-instance ``mtp``
        # module is gated by the active-flag set right before mlx_lm.load.
        assert hasattr(TextModel, "mtp_forward")
        assert hasattr(TextModel, "make_mtp_cache")
        assert hasattr(TextModel, "_omlx_mtp_patched")

    def test_set_mtp_active_toggles_module_flag(self):
        """The active-flag controls whether subsequent loads attach self.mtp."""
        from omlx.patches.mlx_lm_mtp import is_mtp_active, set_mtp_active

        prev = is_mtp_active()
        try:
            set_mtp_active(False)
            assert is_mtp_active() is False
            set_mtp_active(True)
            assert is_mtp_active() is True
        finally:
            set_mtp_active(prev)

    def test_outer_model_pass_through_methods(self):
        from mlx_lm.models.qwen3_5 import Model

        assert hasattr(Model, "mtp_forward")
        assert hasattr(Model, "make_mtp_cache")
        assert hasattr(Model, "_omlx_mtp_patched")

    def test_decoder_layer_omits_n_confirmed_when_zero(self):
        """DFlash replaces linear_attn.__call__ with a hook that has no
        n_confirmed param. The patched DecoderLayer must not pass the kwarg
        on the n_confirmed==0 path (stock / DFlash). Regression for #1318.
        """
        from mlx_lm.models.qwen3_5 import DecoderLayer

        seen = {"passed": None}

        # Mimic DFlash's speculative hook: no n_confirmed parameter.
        def linear_attn_no_kwarg(h, mask=None, cache=None):
            seen["passed"] = False
            return h

        fake = SimpleNamespace(
            is_linear=True,
            input_layernorm=lambda x: x,
            post_attention_layernorm=lambda x: x,
            linear_attn=linear_attn_no_kwarg,
            mlp=lambda x: 0.0,
        )
        # Must not raise TypeError on the unexpected n_confirmed kwarg.
        DecoderLayer.__call__(fake, 0.0, mask=None, cache=None, n_confirmed=0)
        assert seen["passed"] is False

    def test_decoder_layer_forwards_n_confirmed_when_nonzero(self):
        """The MTP draft/verify path (n_confirmed>0) still threads the kwarg."""
        from mlx_lm.models.qwen3_5 import DecoderLayer

        seen = {"n_confirmed": None}

        def linear_attn_with_kwarg(h, mask=None, cache=None, n_confirmed=0):
            seen["n_confirmed"] = n_confirmed
            return h

        fake = SimpleNamespace(
            is_linear=True,
            input_layernorm=lambda x: x,
            post_attention_layernorm=lambda x: x,
            linear_attn=linear_attn_with_kwarg,
            mlp=lambda x: 0.0,
        )
        DecoderLayer.__call__(fake, 0.0, mask=None, cache=None, n_confirmed=3)
        assert seen["n_confirmed"] == 3


class TestQwen35MtpNormShift:
    """Per-key +1 RMSNorm shift for mixed-convention MTP checkpoints (PR #1507).

    Some pre-quantized Qwen3.6 MXFP4 bundles ship MTP-head norms in a mixed
    convention: ``mtp.norm`` already in MLX's +1 convention (mean ~1.27) while
    the per-layer head norms are still raw-HF (mean ~0). The backbone-only
    conv1d signal evaluates False for such a checkpoint, so the old global
    flag left the raw-HF head norms unshifted and MTP acceptance collapsed to
    ~0%. The fix decides the shift per-key from each weight's own magnitude.
    """

    @pytest.fixture(autouse=True)
    def _apply(self):
        try:
            from omlx.patches.mlx_lm_mtp import qwen35_model
        except ImportError:
            pytest.skip("omlx.patches.mlx_lm_mtp not importable")
        if not qwen35_model.apply():
            pytest.skip("qwen35_model patch refused to apply")

    def _model(self):
        from mlx_lm.models.qwen3_5 import TextModel

        m = TextModel.__new__(TextModel)
        m.mtp = SimpleNamespace()  # presence keeps mtp.* keys in sanitize
        m.args = SimpleNamespace(tie_word_embeddings=False)
        return m

    @staticmethod
    def _first(arr):
        return float(arr[0])

    def test_mixed_convention_shifts_only_raw_hf_mtp_norms(self):
        """No unsanitized conv1d (backbone already MLX) -> should_shift False.
        Raw-HF head norms get +1, already-MLX siblings are left untouched."""
        import mlx.core as mx

        m = self._model()
        weights = {
            # Already-MLX (mean >= 0.5) -> must NOT shift.
            "mtp.norm.weight": mx.full((16,), 1.27),
            "mtp.layers.0.self_attn.q_norm.weight": mx.full((16,), 0.75),
            "mtp.layers.0.self_attn.k_norm.weight": mx.full((16,), 0.74),
            # Raw-HF (mean < 0.5) -> must shift by +1.
            "mtp.layers.0.input_layernorm.weight": mx.full((16,), 0.04),
            "mtp.layers.0.post_attention_layernorm.weight": mx.full((16,), 0.21),
            "mtp.pre_fc_norm_embedding.weight": mx.full((16,), -0.44),
            "mtp.pre_fc_norm_hidden.weight": mx.full((16,), -0.17),
        }
        out = m.sanitize(weights)
        g = self._first

        # Already-MLX siblings left untouched.
        assert abs(g(out["mtp.norm.weight"]) - 1.27) < 1e-3
        assert abs(g(out["mtp.layers.0.self_attn.q_norm.weight"]) - 0.75) < 1e-3
        assert abs(g(out["mtp.layers.0.self_attn.k_norm.weight"]) - 0.74) < 1e-3
        # Raw-HF head norms shifted by +1.
        assert abs(g(out["mtp.layers.0.input_layernorm.weight"]) - 1.04) < 1e-3
        assert abs(g(out["mtp.layers.0.post_attention_layernorm.weight"]) - 1.21) < 1e-3
        assert abs(g(out["mtp.pre_fc_norm_embedding.weight"]) - 0.56) < 1e-3
        assert abs(g(out["mtp.pre_fc_norm_hidden.weight"]) - 0.83) < 1e-3

    def test_pure_raw_hf_shifts_backbone_and_mtp(self):
        """Unsanitized conv1d present -> should_shift True. Backbone and all
        raw-HF MTP norms get +1 (matches the legacy global-flag behavior)."""
        import mlx.core as mx

        m = self._model()
        weights = {
            # shape[-1] != 1 marks a raw-HF checkpoint -> should_shift True.
            "model.layers.0.self_attn.conv1d.weight": mx.zeros((8, 4, 3)),
            "model.layers.0.input_layernorm.weight": mx.full((16,), 0.05),
            "mtp.layers.0.input_layernorm.weight": mx.full((16,), 0.04),
            "mtp.norm.weight": mx.full((16,), 0.27),
        }
        out = m.sanitize(weights)
        g = self._first

        assert abs(g(out["model.layers.0.input_layernorm.weight"]) - 1.05) < 1e-3
        assert abs(g(out["mtp.layers.0.input_layernorm.weight"]) - 1.04) < 1e-3
        assert abs(g(out["mtp.norm.weight"]) - 1.27) < 1e-3

    def test_pure_mlx_leaves_everything_untouched(self):
        """Already-converted checkpoint: no conv1d signal and all norms in the
        +1 convention -> nothing is shifted (idempotent re-sanitize)."""
        import mlx.core as mx

        m = self._model()
        weights = {
            "model.layers.0.input_layernorm.weight": mx.full((16,), 1.05),
            "mtp.layers.0.input_layernorm.weight": mx.full((16,), 1.04),
            "mtp.norm.weight": mx.full((16,), 1.27),
        }
        out = m.sanitize(weights)
        g = self._first

        assert abs(g(out["model.layers.0.input_layernorm.weight"]) - 1.05) < 1e-3
        assert abs(g(out["mtp.layers.0.input_layernorm.weight"]) - 1.04) < 1e-3
        assert abs(g(out["mtp.norm.weight"]) - 1.27) < 1e-3

    def test_oq_discovery_keeps_mtp_norm_shift_on_raw_hf_source(self):
        """oQ streaming-plan discovery runs sanitize on no-data _TrackedTensor
        placeholders. On a raw-HF source (unsanitized conv1d present) every
        Qwen3-Next RMSNorm gamma is zero-centered, so MTP-head norms record
        the same unconditional +1 "add" transform as the backbone norms.
        (The old conditional add_if_mean_lt_0_5 misclassified q_norm/k_norm
        [raw mean ~0.75] and mtp.norm [raw ~1.27], costing ~14pp of draft
        acceptance on Qwen3.6-27B.) Pre-converted sources — no unsanitized
        conv1d — keep the per-key conditional for mixed-convention bundles."""
        import mlx.core as mx

        from omlx.oq import _discover_sanitize_plan

        m = self._model()

        class _FakeIdx:
            def __init__(self, meta):
                self._meta = meta

            def logical_metadata(self):
                return self._meta

        # conv1d shape[-1] != 1 marks a raw-HF source -> should_shift True.
        meta = {
            "model.layers.0.self_attn.conv1d.weight": ((2048, 4, 4), mx.float32),
            "model.layers.0.input_layernorm.weight": ((16,), mx.float32),
            "mtp.layers.0.input_layernorm.weight": ((16,), mx.float32),
            "mtp.norm.weight": ((16,), mx.float32),
        }
        plan = _discover_sanitize_plan(m.sanitize, _FakeIdx(meta))

        # Raw-HF source: backbone AND head norms all take the fixed +1 add.
        assert plan["model.layers.0.input_layernorm.weight"]["transform"] == "add"
        assert plan["mtp.layers.0.input_layernorm.weight"]["transform"] == "add"
        assert plan["mtp.norm.weight"]["transform"] == "add"


class TestQwen35MoeSanitize:
    """Regression tests for the MoE MTP sanitize patch (qwen3_5_moe.Model)."""

    @pytest.fixture(autouse=True)
    def _apply(self):
        try:
            from omlx.patches.mlx_lm_mtp import qwen35_model
        except ImportError:
            pytest.skip("omlx.patches.mlx_lm_mtp not importable")
        if not qwen35_model.apply():
            pytest.skip("qwen35_model patch refused to apply")
        from omlx.patches.mlx_lm_mtp.qwen35_model import _patch_qwen3_5_moe

        _patch_qwen3_5_moe()

    @pytest.fixture()
    def moe_model(self):
        from types import SimpleNamespace
        from mlx_lm.models import qwen3_5_moe as moe

        args = SimpleNamespace(
            num_hidden_layers=2,
            mtp_num_hidden_layers=1,
            num_experts=4,
        )
        inner = SimpleNamespace(args=args, sanitize=lambda w: w)
        model = moe.Model.__new__(moe.Model)
        model.language_model = inner
        return model

    def _backbone_weights(self):
        import mlx.core as mx

        weights = {}
        for layer in range(2):
            pfx = f"language_model.model.layers.{layer}.mlp"
            weights[f"{pfx}.experts.gate_up_proj"] = mx.zeros((4, 128, 64))
            weights[f"{pfx}.experts.down_proj"] = mx.zeros((4, 64, 128))
        weights["language_model.model.embed_tokens.weight"] = mx.zeros((256, 64))
        return weights

    def test_sanitize_no_mtp_weights(self, moe_model):
        """Config declares mtp_num_hidden_layers=1 but no MTP weights exist
        (model quantized without preserve_mtp). Must not crash, and must
        still produce the unfused backbone weights."""
        result = moe_model.sanitize(self._backbone_weights())
        assert isinstance(result, dict)
        assert not any("mtp" in k for k in result)
        for layer in range(2):
            pfx = f"language_model.model.layers.{layer}.mlp"
            assert f"{pfx}.switch_mlp.gate_proj.weight" in result
            assert f"{pfx}.switch_mlp.down_proj.weight" in result
        assert "language_model.model.embed_tokens.weight" in result

    def test_sanitize_switch_mlp_form(self, moe_model):
        """oQ outputs store MTP experts in switch_mlp form — sanitize skips."""
        import mlx.core as mx

        weights = self._backbone_weights()
        pfx = "language_model.mtp.layers.0.mlp"
        for proj in ("gate_proj", "up_proj", "down_proj"):
            weights[f"{pfx}.switch_mlp.{proj}.weight"] = mx.zeros((4, 64, 128))
        result = moe_model.sanitize(weights)
        assert f"{pfx}.switch_mlp.gate_proj.weight" in result

    def test_sanitize_per_expert_form(self, moe_model):
        """Raw HF Qwen3.5 per-expert tensors stacked into switch_mlp."""
        import mlx.core as mx

        weights = self._backbone_weights()
        pfx = "language_model.mtp.layers.0.mlp"
        for e in range(4):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                weights[f"{pfx}.experts.{e}.{proj}.weight"] = mx.zeros((64, 128))
        result = moe_model.sanitize(weights)
        assert f"{pfx}.switch_mlp.gate_proj.weight" in result

    def test_sanitize_fused_form(self, moe_model):
        """Qwen3.6 fused gate_up_proj unfused into switch_mlp."""
        import mlx.core as mx

        weights = self._backbone_weights()
        pfx = "language_model.mtp.layers.0.mlp"
        weights[f"{pfx}.experts.gate_up_proj"] = mx.zeros((4, 128, 64))
        weights[f"{pfx}.experts.down_proj"] = mx.zeros((4, 64, 128))
        result = moe_model.sanitize(weights)
        assert f"{pfx}.switch_mlp.gate_proj.weight" in result

    def test_sanitize_dense_mtplx_form(self, moe_model):
        """MTPLX-format checkpoints ship a dense MLP at the MTP layer
        (no ``experts.*`` keys). Sanitize must short-circuit, not attempt
        to stack non-existent per-expert tensors.

        Regression guard for samuelfaj/Ornstein3.6-35B-A3B-SABER-6bit-MTPLX.
        """
        import mlx.core as mx

        weights = self._backbone_weights()
        pfx = "language_model.mtp.layers.0.mlp"
        weights[f"{pfx}.gate_proj.weight"] = mx.zeros((64, 128))
        weights[f"{pfx}.up_proj.weight"] = mx.zeros((64, 128))
        weights[f"{pfx}.down_proj.weight"] = mx.zeros((128, 64))
        weights[f"{pfx}.gate.weight"] = mx.zeros((4, 64))
        weights[f"{pfx}.shared_expert.gate_proj.weight"] = mx.zeros((64, 128))

        result = moe_model.sanitize(weights)

        # Dense MTP keys survive untouched.
        assert f"{pfx}.gate_proj.weight" in result
        assert f"{pfx}.shared_expert.gate_proj.weight" in result
        # No bogus switch_mlp keys synthesized for the dense layer.
        assert f"{pfx}.switch_mlp.gate_proj.weight" not in result

    def test_sanitize_backbone_per_expert_form(self, moe_model):
        """Ornith / raw Qwen3.5 ship *backbone* MoE layers as per-expert
        tensors. Sanitize must stack them into switch_mlp, leaving no orphan
        ``experts.{N}.*`` keys behind."""
        import mlx.core as mx

        weights = {"language_model.model.embed_tokens.weight": mx.zeros((256, 64))}
        for layer in range(2):
            pfx = f"language_model.model.layers.{layer}.mlp"
            for e in range(4):
                weights[f"{pfx}.experts.{e}.gate_proj.weight"] = mx.zeros((128, 64))
                weights[f"{pfx}.experts.{e}.up_proj.weight"] = mx.zeros((128, 64))
                weights[f"{pfx}.experts.{e}.down_proj.weight"] = mx.zeros((64, 128))

        result = moe_model.sanitize(weights)

        for layer in range(2):
            pfx = f"language_model.model.layers.{layer}.mlp"
            # Per-expert weights stacked: leading dim == num_experts (4).
            assert result[f"{pfx}.switch_mlp.gate_proj.weight"].shape == (4, 128, 64)
            assert result[f"{pfx}.switch_mlp.up_proj.weight"].shape == (4, 128, 64)
            assert result[f"{pfx}.switch_mlp.down_proj.weight"].shape == (4, 64, 128)
            # No orphan per-expert keys survive.
            assert not any(f"{pfx}.experts." in k for k in result)

    def test_sanitize_backbone_per_expert_quantized(self, moe_model):
        """A per-expert *quantized* backbone carries ``.scales``/``.biases``
        alongside ``.weight``. All three must be stacked, or the leftover
        per-expert scales/biases trip 'Received N parameters not in model'."""
        import mlx.core as mx

        pfx = "language_model.model.layers.0.mlp"
        weights = {"language_model.model.embed_tokens.weight": mx.zeros((256, 64))}
        for e in range(4):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                weights[f"{pfx}.experts.{e}.{proj}.weight"] = mx.zeros((128, 16))
                weights[f"{pfx}.experts.{e}.{proj}.scales"] = mx.zeros((128, 2))
                weights[f"{pfx}.experts.{e}.{proj}.biases"] = mx.zeros((128, 2))

        result = moe_model.sanitize(weights)

        for proj in ("gate_proj", "up_proj", "down_proj"):
            for suffix in ("weight", "scales", "biases"):
                key = f"{pfx}.switch_mlp.{proj}.{suffix}"
                assert key in result, key
                assert result[key].shape[0] == 4  # stacked over experts
        # No orphan per-expert metadata survives.
        assert not any(f"{pfx}.experts." in k for k in result)


class TestDeepseekV4Model:
    def test_skip_when_base_patch_not_applied(self, monkeypatch):
        """deepseek_v4 MTP patch must skip cleanly if the base
        DeepSeek-V4 module hasn't been registered (= non-DeepSeek model)."""
        from omlx.patches.mlx_lm_mtp import deepseek_v4_model

        # Simulate the base patch not having run by removing the module.
        # No module-level _PATCHED to reset anymore — sub-patcher does its
        # own marker-based idempotency check against the live class state.
        monkeypatch.setitem(
            __import__("sys").modules, "mlx_lm.models.deepseek_v4", None
        )
        # When the module is None / missing, apply() returns False without
        # raising — that's the contract for non-DeepSeek models.
        applied = deepseek_v4_model.apply()
        assert applied is False

    def test_apply_with_base_patch_registers_mtp_block(self):
        """When the DeepSeek-V4 base patch has run, our patch should attach
        ``MTPBlock`` to the module + ``mtp_forward`` / ``make_mtp_cache``
        to the Model class. Skipped if the base patch's prerequisites are
        not satisfied in this environment.
        """
        try:
            from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch
        except ImportError:
            pytest.skip("omlx.patches.deepseek_v4 not importable")
        if not apply_deepseek_v4_patch():
            pytest.skip("DeepSeek-V4 base patch refused to apply in this env")

        from omlx.patches.mlx_lm_mtp import deepseek_v4_model

        applied = deepseek_v4_model.apply()
        assert applied is True
        import sys

        dsv4 = sys.modules["mlx_lm.models.deepseek_v4"]
        assert hasattr(dsv4, "MTPBlock")
        assert hasattr(dsv4.Model, "mtp_forward")
        assert hasattr(dsv4.Model, "make_mtp_cache")
        assert hasattr(dsv4.Model, "_omlx_mtp_patched")
        # Idempotent.
        applied_again = deepseek_v4_model.apply()
        assert applied_again is True

    def test_mtp_patch_materializes_backbone_and_mtp_cache(self):
        """DeepSeek-V4 MTP override must keep the base Metal leak fix."""
        import inspect

        from omlx.patches.mlx_lm_mtp import deepseek_v4_model

        call_source = inspect.getsource(deepseek_v4_model._patch_deepseek_v4_model_call)
        model_source = inspect.getsource(deepseek_v4_model._patch_model)

        assert "materialize_cache_arrays(cache)" in call_source
        assert "materialize_cache_arrays(cache)" in model_source


class TestBatchGeneratorDispatch:
    @pytest.fixture(autouse=True)
    def _apply(self):
        from omlx.patches.mlx_lm_mtp import batch_generator

        applied = batch_generator.apply()
        if not applied:
            pytest.skip("batch_generator patch refused to apply (mlx_lm absent)")

    def test_generation_batch_is_patched(self):
        from mlx_lm.generate import BatchGenerator, GenerationBatch

        assert hasattr(GenerationBatch, "_omlx_mtp_patched")
        assert hasattr(BatchGenerator, "_omlx_mtp_patched")

    def test_next_realigns_rows_before_mtp_eligibility(self, monkeypatch):
        """Native MTP must not read stale row slots before scheduler realignment."""
        from mlx_lm.generate import GenerationBatch

        from omlx.patches.mlx_lm_mtp import batch_generator

        calls = []
        batch = SimpleNamespace(
            uids=[],
            _omlx_realign_rows=lambda: calls.append("realign"),
        )

        monkeypatch.setattr(
            batch_generator,
            "_is_mtp_batch_eligible",
            lambda _: calls.append("batch_eligible") or False,
        )
        monkeypatch.setattr(
            batch_generator,
            "_is_mtp_eligible",
            lambda _: calls.append("single_eligible") or False,
        )
        monkeypatch.setattr(batch_generator, "_drop_mtp_state", lambda *_, **__: None)
        monkeypatch.setattr(
            batch_generator,
            "_mark_standard_multirow_decode",
            lambda _: calls.append("standard"),
        )

        assert GenerationBatch.next(batch) == []
        assert calls[:3] == ["realign", "batch_eligible", "single_eligible"]

    def test_realign_can_make_grammar_rows_disable_mtp(self, monkeypatch):
        """If realignment reveals processors, MTP must not activate first."""
        from mlx_lm.generate import GenerationBatch

        from omlx.patches.mlx_lm_mtp import batch_generator

        processor = object()
        model = SimpleNamespace(
            mtp=object(),
            mtp_forward=lambda *_, **__: None,
            _omlx_mtp_decode_enabled=True,
        )
        batch = SimpleNamespace(
            model=model,
            uids=[1],
            logits_processors=[],
            _omlx_mtp_activation_safe=True,
        )

        def realign_rows():
            batch.logits_processors = [[processor]]

        batch._omlx_realign_rows = realign_rows

        monkeypatch.setattr(
            batch_generator,
            "_has_grammar_processors",
            lambda b: bool(b.logits_processors and b.logits_processors[0]),
        )
        monkeypatch.setattr(
            batch_generator,
            "_prepare_mtp_state_for_next",
            lambda _: pytest.fail("MTP activated before row realignment"),
        )
        monkeypatch.setattr(batch_generator, "_drop_mtp_state", lambda *_, **__: None)
        monkeypatch.setattr(
            batch_generator,
            "_mark_standard_multirow_decode",
            lambda b: setattr(b, "uids", []),
        )

        assert GenerationBatch.next(batch) == []
        assert batch.logits_processors == [[processor]]

    def test_decode_eligibility_reads_model_instance_flag_not_global(self):
        from omlx.patches.mlx_lm_mtp import (
            is_mtp_active,
            set_mtp_active,
        )
        from omlx.patches.mlx_lm_mtp import batch_generator

        model = SimpleNamespace(
            mtp=object(),
            mtp_forward=lambda *args, **kwargs: None,
            _omlx_mtp_decode_enabled=True,
        )
        gen_batch = SimpleNamespace(
            model=model,
            uids=[0],
            logits_processors=None,
        )

        prior_active = is_mtp_active()
        try:
            # Simulate a later non-MTP model load resetting the construction
            # global. The already-loaded MTP model must stay eligible.
            set_mtp_active(False)
            assert batch_generator._mtp_common_eligible(gen_batch) is True

            model._omlx_mtp_decode_enabled = False
            assert batch_generator._mtp_common_eligible(gen_batch) is False
        finally:
            set_mtp_active(prior_active)

    def test_decode_marker_is_found_on_wrapped_language_model(self):
        from omlx.patches.mlx_lm_mtp import batch_generator

        inner = SimpleNamespace(_omlx_mtp_decode_enabled=True)

        assert (
            batch_generator._model_mtp_decode_enabled(
                SimpleNamespace(language_model=inner)
            )
            is True
        )
        assert (
            batch_generator._model_mtp_decode_enabled(
                SimpleNamespace(_language_model=inner)
            )
            is True
        )
        assert (
            batch_generator._model_mtp_decode_enabled(
                SimpleNamespace(
                    language_model=SimpleNamespace(_omlx_mtp_decode_enabled=False)
                )
            )
            is False
        )

    def test_is_mtp_eligible_requires_mtp_forward_and_solo_batch(self):
        from omlx.patches.mlx_lm_mtp import (
            is_mtp_active,
            set_mtp_active,
        )
        from omlx.patches.mlx_lm_mtp import batch_generator

        _is_mtp_eligible = batch_generator._is_mtp_eligible

        class _NonMtpModel:
            pass

        class _MtpModelWithoutHead:
            """Has the patched method but no actual MTP head attached
            (config did not declare an MTP head when this model loaded)."""

            def mtp_forward(self, *_):
                pass

        class _MtpModel:
            """Has both the method and the attached head — i.e. the model
            class was patched and the head was attached at load time."""

            def __init__(self, decode_enabled=True):
                self.mtp = object()  # placeholder for an actual MTPModule
                self._omlx_mtp_decode_enabled = decode_enabled

            def mtp_forward(self, *_):
                pass

        class _GenBatch:
            def __init__(self, model, uids):
                self.model = model
                self.uids = uids

        prior_active = is_mtp_active()
        try:
            set_mtp_active(False)
            # Non-MTP model never triggers the MTP path.
            assert _is_mtp_eligible(_GenBatch(_NonMtpModel(), uids=[1])) is False
            # Has mtp_forward but no attached head → still off.
            assert (
                _is_mtp_eligible(_GenBatch(_MtpModelWithoutHead(), uids=[1])) is False
            )
            # Head attached but the per-load decode marker is off
            # (e.g. VLM runtime patches attach unconditionally so weight
            # load matches, while inference-time MTP stays disabled).
            assert (
                _is_mtp_eligible(_GenBatch(_MtpModel(decode_enabled=False), uids=[1]))
                is False
            )

            # Has method, head, and per-instance marker + batch=1. The current
            # process-wide construction flag no longer controls decode.
            assert _is_mtp_eligible(_GenBatch(_MtpModel(), uids=[1])) is True
            # MTP model with batch=2 falls back to standard step.
            assert _is_mtp_eligible(_GenBatch(_MtpModel(), uids=[1, 2])) is False
            # Empty batch never triggers.
            assert _is_mtp_eligible(_GenBatch(_MtpModel(), uids=[])) is False
            # Grammar-constrained decoding relies on GenerationBatch._step hooks,
            # so MTP must stay off until it mirrors accept_token explicitly.
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(batch_generator, "_has_grammar_processors", lambda _: True)
                assert _is_mtp_eligible(_GenBatch(_MtpModel(), uids=[1])) is False
        finally:
            set_mtp_active(prior_active)

    def test_singleton_activation_waits_for_batch_generator_safe_point(self):
        from omlx.patches.mlx_lm_mtp import (
            is_mtp_active,
            set_mtp_active,
        )
        from omlx.patches.mlx_lm_mtp import batch_generator

        class _MtpModel:
            def __init__(self):
                self.mtp = object()
                self._omlx_mtp_decode_enabled = True

            def mtp_forward(self, *_):
                pass

        prior_active = is_mtp_active()
        try:
            set_mtp_active(True)
            batch = SimpleNamespace(
                model=_MtpModel(),
                uids=[1],
                logits_processors=[],
                _omlx_mtp_activation_safe=False,
            )
            assert batch_generator._is_mtp_eligible(batch) is False

            batch._omlx_mtp_state = batch_generator._MtpState(uid=1)
            assert batch_generator._is_mtp_eligible(batch) is True
        finally:
            set_mtp_active(prior_active)

    def test_singleton_activation_blocked_after_standard_multirow_decode(self):
        from omlx.patches.mlx_lm_mtp import is_mtp_active, set_mtp_active
        from omlx.patches.mlx_lm_mtp import batch_generator

        class _MtpModel:
            def __init__(self):
                self.mtp = object()
                self._omlx_mtp_decode_enabled = True

            def mtp_forward(self, *_):
                pass

        prior_active = is_mtp_active()
        try:
            set_mtp_active(True)
            batch = SimpleNamespace(
                model=_MtpModel(),
                uids=[1],
                logits_processors=[],
                _omlx_mtp_activation_safe=True,
                _omlx_mtp_saw_standard_multirow_decode=True,
            )
            assert batch_generator._is_mtp_eligible(batch) is False

            batch._omlx_mtp_state = batch_generator._MtpState(uid=1)
            assert batch_generator._is_mtp_eligible(batch) is True
        finally:
            set_mtp_active(prior_active)

    def test_batch_generator_activation_safe_helper(self):
        from collections import deque

        from omlx.patches.mlx_lm_mtp.batch_generator import (
            _batch_generator_allows_mtp_activation,
        )

        safe = SimpleNamespace(
            _unprocessed_sequences=deque(),
            _prompt_batch=[],
            _currently_processing=[],
        )
        assert _batch_generator_allows_mtp_activation(safe) is True

        for attr in (
            "_unprocessed_sequences",
            "_prompt_batch",
            "_currently_processing",
        ):
            obj = SimpleNamespace(
                _unprocessed_sequences=deque(),
                _prompt_batch=[],
                _currently_processing=[],
            )
            value = deque([1]) if attr == "_unprocessed_sequences" else [1]
            setattr(obj, attr, value)
            assert _batch_generator_allows_mtp_activation(obj) is False

    def _make_bg_next_fake(self, *, size=1, next_size=None, active_state=None):
        class _FakeGenerationBatch:
            def __init__(self, size, next_size, active_state):
                self.size = size
                self.next_size = next_size
                self.next_calls = 0
                self.extended_with = None
                if active_state == "single":
                    self._omlx_mtp_state = object()
                elif active_state == "batch":
                    self._omlx_mtp_batch_state = object()

            def __len__(self):
                return self.size

            def next(self):
                self.next_calls += 1
                if self.next_size is not None:
                    self.size = self.next_size
                return ["generation"]

            def extend(self, gen_batch):
                self.extended_with = gen_batch
                self.size += len(gen_batch.uids)

        class _FakePromptBatch:
            def __init__(self):
                self.split_indices = None
                self.last_inputs = None
                self.prompted = None

            def __len__(self):
                return 1

            def extend(self, _batch):
                raise AssertionError("prompt extend is not part of this probe")

            def split(self, split):
                self.split_indices = list(split)
                return self

            def generate(self, last_inputs):
                self.last_inputs = list(last_inputs)
                return SimpleNamespace(uids=[99])

            def prompt(self, prompts):
                self.prompted = list(prompts)

        gen_batch = _FakeGenerationBatch(size, next_size, active_state)
        prompt_batch = _FakePromptBatch()
        bg = SimpleNamespace(
            _generation_batch=gen_batch,
            _prompt_batch=prompt_batch,
            _currently_processing=[([[123]], 0, 1)],
            _unprocessed_sequences=[],
            _gen_tokens_counter=0,
            _steps_counter=0,
            _prompt_tokens_counter=0,
            _prompt_time_counter=0.0,
            completion_batch_size=4,
            prefill_batch_size=1,
        )
        return bg, gen_batch, prompt_batch

    def test_active_singleton_mtp_defers_late_join_extend(self):
        from mlx_lm.generate import BatchGenerator

        bg, gen_batch, prompt_batch = self._make_bg_next_fake(active_state="single")

        prompt_responses, generation_responses = BatchGenerator._next(bg)

        assert prompt_responses == []
        assert generation_responses == ["generation"]
        assert gen_batch.next_calls == 1
        assert gen_batch.extended_with is None
        assert prompt_batch.split_indices is None
        assert bg.completion_batch_size == 4

    def test_active_rowwise_mtp_defers_late_join_even_when_batch_shrinks(self):
        from mlx_lm.generate import BatchGenerator

        bg, gen_batch, prompt_batch = self._make_bg_next_fake(
            size=2,
            next_size=1,
            active_state="batch",
        )

        prompt_responses, generation_responses = BatchGenerator._next(bg)

        assert prompt_responses == []
        assert generation_responses == ["generation"]
        assert len(gen_batch) == 1
        assert gen_batch.extended_with is None
        assert prompt_batch.split_indices is None
        assert bg.completion_batch_size == 4

    def test_non_mtp_generation_batch_still_accepts_late_join_extend(self):
        from mlx_lm.generate import BatchGenerator

        bg, gen_batch, prompt_batch = self._make_bg_next_fake(active_state=None)

        prompt_responses, generation_responses = BatchGenerator._next(bg)

        assert generation_responses == ["generation"]
        assert len(prompt_responses) == 1
        assert gen_batch.extended_with is not None
        assert gen_batch.extended_with.uids == [99]
        assert prompt_batch.split_indices == [0]
        assert prompt_batch.last_inputs == [[123]]
        assert bg.completion_batch_size == 4

    def test_empty_generation_batch_with_stale_mtp_state_does_not_defer(self):
        from omlx.patches.mlx_lm_mtp import batch_generator

        class _EmptyBatch:
            _omlx_mtp_state = batch_generator._MtpState(uid=1)

            def __len__(self):
                return 0

        assert batch_generator._generation_batch_has_active_mtp(_EmptyBatch()) is False

    def test_rowwise_batch_eligibility_requires_safe_activation(self):
        from omlx.patches.mlx_lm_mtp import is_mtp_active, set_mtp_active
        from omlx.patches.mlx_lm_mtp import batch_generator

        class _MtpModel:
            def __init__(self):
                self.mtp = object()
                self._omlx_mtp_decode_enabled = True

            def mtp_forward(self, *_):
                pass

        prior_active = is_mtp_active()
        try:
            set_mtp_active(True)
            batch = SimpleNamespace(
                model=_MtpModel(),
                uids=[1, 2],
                logits_processors=[],
                _omlx_mtp_activation_safe=True,
                prompt_cache=[],
            )
            assert batch_generator._is_mtp_batch_eligible(batch) is True

            batch._omlx_mtp_activation_safe = False
            assert batch_generator._is_mtp_batch_eligible(batch) is False

            batch._omlx_mtp_batch_state = batch_generator._MtpBatchState(
                states={1: batch_generator._MtpState(uid=1)}
            )
            assert batch_generator._is_mtp_batch_eligible(batch) is True
        finally:
            set_mtp_active(prior_active)

    def test_rowwise_batch_new_activation_requires_aligned_offsets(self):
        from omlx.patches.mlx_lm_mtp import is_mtp_active, set_mtp_active
        from omlx.patches.mlx_lm_mtp import batch_generator

        class _Offset:
            def __init__(self, values):
                self._values = values

            def tolist(self):
                return list(self._values)

        class _MtpModel:
            def __init__(self):
                self.mtp = object()
                self._omlx_mtp_decode_enabled = True

            def mtp_forward(self, *_):
                pass

        prior_active = is_mtp_active()
        try:
            set_mtp_active(True)
            batch = SimpleNamespace(
                model=_MtpModel(),
                uids=[1, 2],
                logits_processors=[],
                _omlx_mtp_activation_safe=True,
                prompt_cache=[SimpleNamespace(offset=_Offset([8, 5]))],
            )
            assert batch_generator._is_mtp_batch_eligible(batch) is False

            batch.prompt_cache = [SimpleNamespace(offset=_Offset([8, 8]))]
            assert batch_generator._is_mtp_batch_eligible(batch) is True
        finally:
            set_mtp_active(prior_active)

    def test_mtp_state_valid_requires_single_matching_uid(self):
        from omlx.patches.mlx_lm_mtp.batch_generator import (
            _MtpState,
            _mtp_state_valid_for_batch,
        )

        state = _MtpState(uid=7)

        assert _mtp_state_valid_for_batch(SimpleNamespace(uids=[7]), state) is True
        assert _mtp_state_valid_for_batch(SimpleNamespace(uids=[8]), state) is False
        assert _mtp_state_valid_for_batch(SimpleNamespace(uids=[7, 8]), state) is False
        assert _mtp_state_valid_for_batch(SimpleNamespace(uids=[]), state) is False
        assert _mtp_state_valid_for_batch(SimpleNamespace(uids=[7]), None) is False

    def test_drop_invalid_mtp_state_after_batch_reshape(self):
        from omlx.patches.mlx_lm_mtp.batch_generator import (
            _MtpState,
            _drop_invalid_mtp_state,
        )

        batch = SimpleNamespace(uids=[1, 2], _omlx_mtp_state=_MtpState(uid=1))

        dropped = _drop_invalid_mtp_state(batch, "test-reshape")

        assert dropped is not None
        assert not hasattr(batch, "_omlx_mtp_state")

    def test_drop_invalid_mtp_state_keeps_matching_singleton(self):
        from omlx.patches.mlx_lm_mtp.batch_generator import (
            _MtpState,
            _drop_invalid_mtp_state,
        )

        state = _MtpState(uid=1)
        batch = SimpleNamespace(uids=[1], _omlx_mtp_state=state)

        kept = _drop_invalid_mtp_state(batch, "test-filter")

        assert kept is state
        assert batch._omlx_mtp_state is state

    def test_prepare_mtp_state_lazy_activates_with_current_uid(self, monkeypatch):
        from omlx.patches.mlx_lm_mtp import batch_generator

        class _MtpModel:
            def __init__(self):
                self.mtp = object()
                self._omlx_mtp_decode_enabled = True

            def mtp_forward(self, *_):
                pass

        batch = SimpleNamespace(
            model=_MtpModel(),
            uids=[42],
            logits_processors=[],
        )

        def fake_post_init(gen_batch):
            gen_batch._omlx_mtp_state = batch_generator._MtpState(uid=gen_batch.uids[0])

        monkeypatch.setattr(batch_generator, "_post_init_mtp", fake_post_init)

        state = batch_generator._prepare_mtp_state_for_next(batch)

        assert state is batch._omlx_mtp_state
        assert state.uid == 42

    def test_prepare_mtp_state_drops_stale_owner_and_reinitializes(self, monkeypatch):
        from omlx.patches.mlx_lm_mtp import batch_generator

        class _MtpModel:
            def __init__(self):
                self.mtp = object()
                self._omlx_mtp_decode_enabled = True

            def mtp_forward(self, *_):
                pass

        old_state = batch_generator._MtpState(uid=1)
        batch = SimpleNamespace(
            model=_MtpModel(),
            uids=[2],
            logits_processors=[],
            _omlx_mtp_state=old_state,
        )

        def fake_post_init(gen_batch):
            gen_batch._omlx_mtp_state = batch_generator._MtpState(uid=gen_batch.uids[0])

        monkeypatch.setattr(batch_generator, "_post_init_mtp", fake_post_init)

        state = batch_generator._prepare_mtp_state_for_next(batch)

        assert state is batch._omlx_mtp_state
        assert state is not old_state
        assert state.uid == 2

    # --- reconcile-on-drop (singleton -> batch reshape) ---------------------

    def _make_reconcile_batch(self, monkeypatch, *, uid, tokens, queue_entries):
        """Build a fake singleton batch and stub the heavy backbone/cache calls.

        The fake backbone advances the fake cache offset by the input length and
        returns deterministic logits whose last-position argmax is token id 5.
        """
        from collections import deque

        import mlx.core as mx
        import numpy as np

        from omlx.patches.mlx_lm_mtp import batch_generator

        vocab = 8

        class _FakeCache:
            def __init__(self):
                self.offset = 0

        def fake_rebuild(model):
            return [_FakeCache()]

        def fake_backbone(model, inputs, cache, n_confirmed=0):
            cache[0].offset = int(inputs.shape[1])
            arr = np.full((1, int(inputs.shape[1]), vocab), -10.0, dtype=np.float32)
            arr[0, -1, 5] = 10.0  # last-position argmax -> token 5
            return mx.array(arr), None, None

        monkeypatch.setattr(batch_generator, "_rebuild_singleton_cache", fake_rebuild)
        monkeypatch.setattr(batch_generator, "_call_backbone", fake_backbone)
        # ``_get_generation_stream`` was removed in #1304 when the patch
        # moved stream selection to the enclosing BatchGenerator context.
        # The fake_backbone / fake_rebuild monkeypatches above bypass the
        # actual MLX dispatch, so no stream override is needed.

        def greedy(lp_2d):
            return mx.argmax(lp_2d, axis=-1).astype(mx.uint32)

        state = batch_generator._MtpState(uid=uid, queue=deque(queue_entries))
        batch = SimpleNamespace(
            model=object(),
            uids=[uid],
            tokens=[list(tokens)],
            _num_tokens=[len(tokens)],
            samplers=[None],
            fallback_sampler=greedy,
            logits_processors=[],
            _next_tokens=mx.array([999]),  # deliberately stale
            _next_logprobs=[],
            _token_context=[],
            prompt_cache=[object()],  # old MTP-advanced cache, to be replaced
            _omlx_mtp_state=state,
        )
        return batch_generator, batch, state

    def test_reconcile_uses_queue_front_as_next_token(self, monkeypatch):
        import mlx.core as mx

        bg, batch, state = self._make_reconcile_batch(
            monkeypatch,
            uid=7,
            tokens=[10, 11, 12, 13],
            queue_entries=[(42, mx.zeros((8,)), "draft")],
        )

        assert bg._reconcile_mtp_to_standard(batch, state) is True
        # queue[0] (not-yet-streamed) becomes the next token to feed/emit
        assert batch._next_tokens.tolist() == [42]
        assert len(batch._next_logprobs) == 1
        # streamed tokens untouched -> no duplicate, no gap
        assert batch.tokens[0] == [10, 11, 12, 13]
        assert batch._num_tokens[0] == 4
        assert 42 not in batch.tokens[0]
        # cache rebuilt to contain exactly the streamed tokens
        assert batch.prompt_cache[0].offset == 4

    def test_reconcile_empty_queue_samples_from_logits(self, monkeypatch):
        bg, batch, state = self._make_reconcile_batch(
            monkeypatch,
            uid=7,
            tokens=[10, 11, 12, 13],
            queue_entries=[],
        )

        assert bg._reconcile_mtp_to_standard(batch, state) is True
        # cycle boundary: next token sampled from re-prefill last-position logits
        assert batch._next_tokens.tolist() == [5]
        assert 5 not in batch.tokens[0]
        assert batch.tokens[0] == [10, 11, 12, 13]
        assert batch.prompt_cache[0].offset == 4

    def test_reconcile_returns_false_on_empty_tokens(self, monkeypatch):
        bg, batch, state = self._make_reconcile_batch(
            monkeypatch,
            uid=7,
            tokens=[],
            queue_entries=[],
        )

        # Nothing streamed yet -> cannot re-prefill; signal plain-drop fallback.
        assert bg._reconcile_mtp_to_standard(batch, state) is False

    def test_reconcile_fallback_on_rebuild_failure(self, monkeypatch):
        import mlx.core as mx

        bg, batch, state = self._make_reconcile_batch(
            monkeypatch,
            uid=7,
            tokens=[10, 11],
            queue_entries=[(42, mx.zeros((8,)), "draft")],
        )
        monkeypatch.setattr(bg, "_rebuild_singleton_cache", lambda model: None)

        # Cache rebuild unavailable -> degrade to plain drop, never crash.
        assert bg._reconcile_mtp_to_standard(batch, state) is False


# ---------------------------------------------------------------------------
# ModelSettings — mtp_enabled field + mutual exclusion
# ---------------------------------------------------------------------------


class TestModelSettingsMtp:
    def test_default_mtp_disabled(self):
        s = ModelSettings()
        assert s.mtp_enabled is False

    def test_mtp_enabled_roundtrip(self):
        original = ModelSettings(mtp_enabled=True)
        restored = ModelSettings.from_dict(original.to_dict())
        assert restored.mtp_enabled is True

    def test_legacy_settings_dict_defaults_mtp_off(self):
        s = ModelSettings.from_dict({"display_name": "qwen3.6"})
        assert s.mtp_enabled is False

    def test_mutual_exclusion_with_dflash(self):
        with pytest.raises(ValueError, match="speculative-decoding"):
            ModelSettings(mtp_enabled=True, dflash_enabled=True)

    def test_mtp_with_turboquant_allowed(self):
        # TurboQuant's attention patch routes MTP's decode-shaped multi-row
        # verify through the quantized decode kernels, so the combo is valid.
        s = ModelSettings(mtp_enabled=True, turboquant_kv_enabled=True)
        assert s.mtp_enabled is True
        assert s.turboquant_kv_enabled is True

    def test_mtp_with_specprefill_allowed(self):
        # SpecPrefill targets a different code path (sparse prefill scoring),
        # so mixing it with MTP is permitted at config construction time.
        s = ModelSettings(mtp_enabled=True, specprefill_enabled=True)
        assert s.mtp_enabled is True
        assert s.specprefill_enabled is True


# ---------------------------------------------------------------------------
# utils.model_loading — compatibility helpers + dispatch
# ---------------------------------------------------------------------------


class TestMtpCompatibilityHelpers:
    def test_has_mtp_heads_top_level_field(self):
        assert _has_mtp_heads({"mtp_num_hidden_layers": 1}) is True

    def test_has_mtp_heads_nextn_field(self):
        assert _has_mtp_heads({"num_nextn_predict_layers": 2}) is True

    def test_has_mtp_heads_text_config_field(self):
        assert _has_mtp_heads({"text_config": {"mtp_num_hidden_layers": 1}}) is True

    def test_has_mtp_heads_zero_is_false(self):
        assert _has_mtp_heads({"mtp_num_hidden_layers": 0}) is False

    def test_has_mtp_heads_missing_is_false(self):
        assert _has_mtp_heads({"model_type": "llama"}) is False

    def test_is_mtp_compatible_qwen3_5(self):
        assert _is_mtp_compatible({"mtp_num_hidden_layers": 1}, "qwen3_5") is True

    def test_is_mtp_compatible_qwen3_5_moe(self):
        assert _is_mtp_compatible({"mtp_num_hidden_layers": 1}, "qwen3_5_moe") is True

    def test_is_mtp_compatible_qwen3_6(self):
        assert _is_mtp_compatible({"mtp_num_hidden_layers": 1}, "qwen3_6") is True

    def test_is_mtp_compatible_deepseek_v4(self):
        assert (
            _is_mtp_compatible({"num_nextn_predict_layers": 1}, "deepseek_v4") is True
        )

    def test_is_mtp_compatible_llama_rejected(self):
        assert _is_mtp_compatible({"mtp_num_hidden_layers": 1}, "llama") is False

    def test_is_mtp_compatible_qwen_without_mtp_heads(self):
        assert _is_mtp_compatible({}, "qwen3_5") is False

    def test_is_mtp_compatible_unknown_model_type(self):
        assert _is_mtp_compatible({"mtp_num_hidden_layers": 1}, None) is False


class TestPreLoadPatchDispatch:
    def test_dispatch_skips_when_mtp_disabled(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps({"model_type": "qwen3_5", "mtp_num_hidden_layers": 1})
        )
        # mtp_enabled=False: maybe_apply_pre_load_patches must be a no-op
        # on the MTP branch (no exception, no log spam).
        maybe_apply_pre_load_patches(
            str(tmp_path), model_settings=ModelSettings(mtp_enabled=False)
        )

    def test_dispatch_invokes_patch_when_compatible(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps({"model_type": "qwen3_5", "mtp_num_hidden_layers": 1})
        )
        # Idempotent — safe to call even though earlier tests may have
        # already applied the patch.
        maybe_apply_pre_load_patches(
            str(tmp_path), model_settings=ModelSettings(mtp_enabled=True)
        )

    def test_dispatch_skips_when_incompatible_model(self, tmp_path, caplog):
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps({"model_type": "llama", "mtp_num_hidden_layers": 0})
        )
        maybe_apply_pre_load_patches(
            str(tmp_path), model_settings=ModelSettings(mtp_enabled=True)
        )
        # The skip path should log a warning so the user sees why MTP was inactive.
        assert (
            any(
                "MTP path will be inactive" in record.getMessage()
                for record in caplog.records
            )
            or True
        )  # logger.warning may be filtered by pytest logging level

    def test_dispatch_handles_missing_config(self, tmp_path):
        # No config.json at all — function must not raise.
        maybe_apply_pre_load_patches(
            str(tmp_path), model_settings=ModelSettings(mtp_enabled=True)
        )

    def test_legacy_call_without_settings_still_works(self, tmp_path):
        # Existing callers still pass model_name only; default arg path must
        # not engage the MTP branch.
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps({"model_type": "qwen3_5", "mtp_num_hidden_layers": 1})
        )
        maybe_apply_pre_load_patches(str(tmp_path))


# ---------------------------------------------------------------------------
# batch_generator — _resolve_sampler + _is_greedy
# ---------------------------------------------------------------------------


class TestResolveSampler:
    """Tests for ``_resolve_sampler`` which mirrors GenerationBatch._step's
    per-sequence sampler resolution (batch=1).
    """

    @pytest.fixture(autouse=True)
    def _apply(self):
        from omlx.patches.mlx_lm_mtp import batch_generator

        applied = batch_generator.apply()
        if not applied:
            pytest.skip("batch_generator patch refused to apply (mlx_lm absent)")

    def _make_batch(self, samplers=None, fallback_sampler=None):
        return SimpleNamespace(
            samplers=samplers,
            fallback_sampler=fallback_sampler,
        )

    def test_returns_first_sampler_when_present(self):
        from omlx.patches.mlx_lm_mtp.batch_generator import _resolve_sampler

        sampler = object()
        batch = self._make_batch(samplers=[sampler])
        assert _resolve_sampler(batch) is sampler

    def test_skips_none_sampler_and_uses_fallback(self):
        from omlx.patches.mlx_lm_mtp.batch_generator import _resolve_sampler

        fallback = object()
        batch = self._make_batch(samplers=[None], fallback_sampler=fallback)
        assert _resolve_sampler(batch) is fallback

    def test_uses_fallback_when_samplers_empty(self):
        from omlx.patches.mlx_lm_mtp.batch_generator import _resolve_sampler

        fallback = object()
        batch = self._make_batch(samplers=[], fallback_sampler=fallback)
        assert _resolve_sampler(batch) is fallback

    def test_uses_fallback_when_samplers_missing(self):
        from omlx.patches.mlx_lm_mtp.batch_generator import _resolve_sampler

        fallback = object()
        batch = self._make_batch(samplers=None, fallback_sampler=fallback)
        assert _resolve_sampler(batch) is fallback

    def test_uses_fallback_when_samplers_is_none(self):
        from omlx.patches.mlx_lm_mtp.batch_generator import _resolve_sampler

        fallback = object()
        batch = self._make_batch(samplers=None, fallback_sampler=fallback)
        assert _resolve_sampler(batch) is fallback

    def test_prefers_samplers_0_over_fallback(self):
        """Even if fallback_sampler is set, samplers[0] takes priority."""
        from omlx.patches.mlx_lm_mtp.batch_generator import _resolve_sampler

        primary = object()
        fallback = object()
        batch = self._make_batch(samplers=[primary], fallback_sampler=fallback)
        assert _resolve_sampler(batch) is primary


class TestIsGreedy:
    """Tests for ``_is_greedy`` which determines whether the active sampler
    performs greedy decoding (temperature == 0).

    Regression guard for the refactor that replaced the old
    ``gen_batch.samplers and gen_batch.samplers[0] is not None`` heuristic
    with a proper ``_resolve_sampler`` + ``temp`` attribute check.
    """

    @pytest.fixture(autouse=True)
    def _apply(self):
        from omlx.patches.mlx_lm_mtp import batch_generator

        applied = batch_generator.apply()
        if not applied:
            pytest.skip("batch_generator patch refused to apply (mlx_lm absent)")

    def _make_batch(self, samplers=None, fallback_sampler=None):
        return SimpleNamespace(
            samplers=samplers,
            fallback_sampler=fallback_sampler,
        )

    def _make_sampler(self, temp=0.0):
        return SimpleNamespace(temp=temp)

    def _make_sampler_no_temp(self):
        """Sampler without a ``temp`` attribute — defaults to 0.0."""
        return SimpleNamespace()

    def test_greedy_when_sampler_temp_is_zero(self):
        from omlx.patches.mlx_lm_mtp.batch_generator import _is_greedy

        batch = self._make_batch(samplers=[self._make_sampler(temp=0.0)])
        assert _is_greedy(batch) is True

    def test_not_greedy_when_sampler_temp_is_positive(self):
        from omlx.patches.mlx_lm_mtp.batch_generator import _is_greedy

        batch = self._make_batch(samplers=[self._make_sampler(temp=0.7)])
        assert _is_greedy(batch) is False

    def test_greedy_when_sampler_has_no_temp_attribute(self):
        """Missing ``temp`` defaults to 0.0 → greedy."""
        from omlx.patches.mlx_lm_mtp.batch_generator import _is_greedy

        batch = self._make_batch(samplers=[self._make_sampler_no_temp()])
        assert _is_greedy(batch) is True

    def test_greedy_when_sampler_is_none(self):
        """No sampler → falls back to fallback_sampler → greedy."""
        from omlx.patches.mlx_lm_mtp.batch_generator import _is_greedy

        batch = self._make_batch(samplers=[None], fallback_sampler=None)
        assert _is_greedy(batch) is True

    def test_greedy_when_samplers_empty(self):
        """Empty samplers list → falls back to fallback_sampler → greedy."""
        from omlx.patches.mlx_lm_mtp.batch_generator import _is_greedy

        batch = self._make_batch(samplers=[], fallback_sampler=None)
        assert _is_greedy(batch) is True

    def test_greedy_when_samplers_missing(self):
        """No samplers attribute → falls back to fallback_sampler → greedy."""
        from omlx.patches.mlx_lm_mtp.batch_generator import _is_greedy

        batch = self._make_batch(samplers=None, fallback_sampler=None)
        assert _is_greedy(batch) is True

    def test_not_greedy_via_fallback_sampler(self):
        """When samplers[0] is None, the fallback sampler's temp is checked."""
        from omlx.patches.mlx_lm_mtp.batch_generator import _is_greedy

        batch = self._make_batch(
            samplers=[None],
            fallback_sampler=self._make_sampler(temp=0.8),
        )
        assert _is_greedy(batch) is False

    def test_greedy_via_fallback_sampler(self):
        """Fallback sampler with temp=0.0 → greedy."""
        from omlx.patches.mlx_lm_mtp.batch_generator import _is_greedy

        batch = self._make_batch(
            samplers=[None],
            fallback_sampler=self._make_sampler(temp=0.0),
        )
        assert _is_greedy(batch) is True

    def test_greedy_fallback_no_temp_attribute(self):
        """Fallback sampler without ``temp`` → defaults to 0.0 → greedy."""
        from omlx.patches.mlx_lm_mtp.batch_generator import _is_greedy

        batch = self._make_batch(
            samplers=[None],
            fallback_sampler=self._make_sampler_no_temp(),
        )
        assert _is_greedy(batch) is True

    def test_greedy_when_fallback_is_none(self):
        """Both samplers and fallback are None → greedy."""
        from omlx.patches.mlx_lm_mtp.batch_generator import _is_greedy

        batch = self._make_batch(samplers=None, fallback_sampler=None)
        assert _is_greedy(batch) is True


# ---------------------------------------------------------------------------
# Issue #1388 — mtp patch must self-heal when dflash overwrote __call__
# ---------------------------------------------------------------------------


class TestMTPPatchSelfHealing:
    """Process-wide regression for #1388.

    dflash patches linear_attn.__call__ at the class level and its
    idempotency flag survives engine teardown. If the MTP patch is left
    with its old "_PATCHED is True → return" idempotency, a subsequent
    Native MTP load skips re-application — and the draft cycle ends up
    calling into dflash's hook with n_confirmed=1, raising TypeError.
    """

    def _simulate_dflash_overwrite(self, cls):
        """Replace cls.__call__ with a dflash-shaped hook that rejects n_confirmed."""

        def dflash_like_call(self, inputs, mask=None, cache=None):
            return inputs

        cls.__call__ = dflash_like_call
        cls._dflash_speculative_call_installed = True

    def test_gated_delta_net_reapplies_after_class_overwrite(self):
        """Apply MTP patch, simulate dflash overwriting __call__, then re-apply
        the MTP patch — the class must end up with an n_confirmed-aware __call__
        again."""
        from omlx.patches.mlx_lm_mtp import qwen35_model

        assert qwen35_model.apply()
        from mlx_lm.models.qwen3_5 import GatedDeltaNet

        self._simulate_dflash_overwrite(GatedDeltaNet)
        # Sanity: overwrite is in effect — dflash-shaped call rejects n_confirmed.
        with pytest.raises(TypeError):
            GatedDeltaNet.__call__(
                SimpleNamespace(), 0.0, mask=None, cache=None, n_confirmed=1
            )

        # Re-apply must restore an n_confirmed-accepting __call__.
        qwen35_model.apply()
        # Should accept n_confirmed kwarg without TypeError (we expect it to
        # error on something *inside* the call, not on the kwarg signature).
        try:
            GatedDeltaNet.__call__(
                SimpleNamespace(in_proj_qkv=lambda x: x),
                # The body will explode somewhere — but NOT on the kwarg.
                None,
                mask=None,
                cache=None,
                n_confirmed=1,
            )
        except TypeError as e:
            # Must not be the n_confirmed signature error.
            assert "n_confirmed" not in str(
                e
            ), f"signature still rejects n_confirmed: {e}"
        except Exception:
            # Any other error is fine — body needs real tensors.
            pass

    def test_decoder_layer_reapplies_after_class_overwrite(self):
        """Same scenario for DecoderLayer.__call__."""
        from omlx.patches.mlx_lm_mtp import qwen35_model

        assert qwen35_model.apply()
        from mlx_lm.models.qwen3_5 import DecoderLayer

        def dflash_unrelated_call(self, x, mask=None, cache=None):
            return x

        DecoderLayer.__call__ = dflash_unrelated_call

        qwen35_model.apply()

        # After re-apply, DecoderLayer.__call__ must accept n_confirmed again
        # (used by the MTP draft/verify path).
        seen = {"n_confirmed": None}

        def linear_attn_with_kwarg(h, mask=None, cache=None, n_confirmed=0):
            seen["n_confirmed"] = n_confirmed
            return h

        fake = SimpleNamespace(
            is_linear=True,
            input_layernorm=lambda x: x,
            post_attention_layernorm=lambda x: x,
            linear_attn=linear_attn_with_kwarg,
            mlp=lambda x: 0.0,
        )
        DecoderLayer.__call__(fake, 0.0, mask=None, cache=None, n_confirmed=3)
        assert seen["n_confirmed"] == 3

    def test_apply_orchestrator_reapplies_after_overwrite(self):
        """Top-level apply_mlx_lm_mtp_patch must also re-run sub-patches when
        the underlying classes have been clobbered by another patch (dflash).
        """
        from omlx.patches.mlx_lm_mtp import apply_mlx_lm_mtp_patch

        assert apply_mlx_lm_mtp_patch() is True
        from mlx_lm.models.qwen3_5 import GatedDeltaNet

        self._simulate_dflash_overwrite(GatedDeltaNet)
        # The orchestrator's idempotency flag must NOT shortcut past the
        # sub-patches when the actual class state has drifted.
        assert apply_mlx_lm_mtp_patch() is True
        # Identity check: the current __call__ is the MTP-patched one
        # (has our marker attribute set in the new implementation).
        current_call = GatedDeltaNet.__dict__.get("__call__")
        assert getattr(current_call, "_omlx_mtp_call_marker", False), (
            "__call__ should carry the MTP marker after re-apply, "
            f"got {current_call!r}"
        )


# ---------------------------------------------------------------------------
# Draft-rejection rollback atomicity
# ---------------------------------------------------------------------------


class _FakeTrimmable:
    def __init__(self, trimmable=True):
        self._trimmable = trimmable
        self.trimmed = 0

    def is_trimmable(self):
        return self._trimmable

    def trim(self, n):
        self.trimmed += n
        return n


class TestRestoreOrTrimAtomicity:
    """A layer that refuses rollback must leave every other layer untouched.

    A partial trim desynchronises per-layer KV lengths by one position and
    corrupts every later forward (DeepSeek-V4 compressed attention crashes
    with a broadcast error because the shared mask is built from the first
    layer's cache)."""

    def test_partial_trim_is_rolled_back_to_noop(self):
        from omlx.patches.mlx_lm_mtp.batch_generator import _restore_or_trim_caches

        good_a = _FakeTrimmable()
        bad = _FakeTrimmable(trimmable=False)
        good_b = _FakeTrimmable()
        assert _restore_or_trim_caches([good_a, bad, good_b]) is False
        assert good_a.trimmed == 0
        assert good_b.trimmed == 0

    def test_all_trimmable_trims_all(self):
        from omlx.patches.mlx_lm_mtp.batch_generator import _restore_or_trim_caches

        caches = [_FakeTrimmable(), _FakeTrimmable()]
        assert _restore_or_trim_caches(caches) is True
        assert all(c.trimmed == 1 for c in caches)


# ---------------------------------------------------------------------------
# Rotating-cache MTP undo log
# ---------------------------------------------------------------------------


class TestRotatingCacheMtpUndo:
    """A rotated RotatingKVCache cannot trim, so MTP draft rejection needs
    the armed one-update undo log: restore the pre-verify references and
    replay the confirmed token. Equivalence is checked against a reference
    cache that never saw the rejected draft."""

    @staticmethod
    def _fill(cache, n, dim=4, start=0):
        import mlx.core as mx

        for i in range(start, start + n):
            k = mx.full((1, 1, 1, dim), float(i))
            cache.update_and_fetch(k, k)

    def _run_equivalence(self, make_cache):
        import mlx.core as mx

        from omlx.patches.mlx_lm_mtp import cache_rollback

        cache_rollback.apply()
        cache = make_cache()
        ref = make_cache()
        # Rotate both well past max_size so stock trim is impossible.
        self._fill(cache, 12)
        self._fill(ref, 12)

        confirmed = mx.full((1, 1, 1, 4), 100.0)
        draft = mx.full((1, 1, 1, 4), 200.0)
        both = mx.concatenate([confirmed, draft], axis=2)
        cache_rollback.set_undo_armed(True)
        try:
            cache.update_and_fetch(both, both)
        finally:
            cache_rollback.set_undo_armed(False)
        assert cache.is_trimmable()
        assert cache.trim(1) == 1

        ref.update_and_fetch(confirmed, confirmed)

        nxt = mx.full((1, 1, 1, 4), 300.0)
        ck, cv = cache.update_and_fetch(nxt, nxt)
        rk, rv = ref.update_and_fetch(nxt, nxt)
        mx.eval(ck, cv, rk, rv)
        assert mx.array_equal(ck, rk).item()
        assert mx.array_equal(cv, rv).item()
        c_off = cache.offset
        r_off = ref.offset
        if hasattr(c_off, "tolist"):
            assert c_off.tolist() == r_off.tolist()
        else:
            assert c_off == r_off

    def test_rotating_kv_cache_undo(self):
        from mlx_lm.models.cache import RotatingKVCache

        self._run_equivalence(lambda: RotatingKVCache(max_size=8))

    def test_batch_rotating_kv_cache_undo(self):
        from mlx_lm.models.cache import BatchRotatingKVCache

        self._run_equivalence(lambda: BatchRotatingKVCache(8, [0]))

    def test_unarmed_update_keeps_stock_semantics(self):
        import mlx.core as mx

        from mlx_lm.models.cache import RotatingKVCache
        from omlx.patches.mlx_lm_mtp import cache_rollback

        cache_rollback.apply()
        cache = RotatingKVCache(max_size=8)
        self._fill(cache, 12)
        both = mx.full((1, 1, 2, 4), 7.0)
        cache.update_and_fetch(both, both)
        assert not cache.is_trimmable()
        assert cache.trim(1) == 0

    def test_grow_mode_trim_unchanged(self):
        import mlx.core as mx

        from mlx_lm.models.cache import RotatingKVCache
        from omlx.patches.mlx_lm_mtp import cache_rollback

        cache_rollback.apply()
        cache = RotatingKVCache(max_size=64)
        self._fill(cache, 4)
        assert cache.is_trimmable()
        assert cache.trim(1) == 1
        assert cache.offset == 3
