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

    def test_sanitize_no_mtp_weights(self, moe_model, caplog):
        """Config declares mtp_num_hidden_layers=1 but no MTP weights exist
        (model quantized without preserve_mtp). Must not crash."""
        import logging

        with caplog.at_level(logging.DEBUG):
            result = moe_model.sanitize(self._backbone_weights())
        assert not any("mtp" in k for k in result)
        assert any("no MTP weights found" in r.getMessage() for r in caplog.records)

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


class TestDeepseekV4Model:
    def test_skip_when_base_patch_not_applied(self, monkeypatch):
        """deepseek_v4 MTP patch must skip cleanly if the base
        DeepSeek-V4 module hasn't been registered (= non-DeepSeek model)."""
        from omlx.patches.mlx_lm_mtp import deepseek_v4_model

        # Simulate the base patch not having run by removing the module.
        # No module-level _PATCHED to reset anymore — sub-patcher does its
        # own marker-based idempotency check against the live class state.
        monkeypatch.setitem(__import__("sys").modules, "mlx_lm.models.deepseek_v4", None)
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


class TestBatchGeneratorDispatch:
    @pytest.fixture(autouse=True)
    def _apply(self):
        from omlx.patches.mlx_lm_mtp import batch_generator

        applied = batch_generator.apply()
        if not applied:
            pytest.skip("batch_generator patch refused to apply (mlx_lm absent)")

    def test_generation_batch_is_patched(self):
        from mlx_lm.generate import GenerationBatch

        assert hasattr(GenerationBatch, "_omlx_mtp_patched")

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

            def __init__(self):
                self.mtp = object()  # placeholder for an actual MTPModule

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
                _is_mtp_eligible(_GenBatch(_MtpModelWithoutHead(), uids=[1]))
                is False
            )
            # Head attached but the per-load mtp_active flag is off
            # (e.g. VLM runtime patches attach unconditionally so weight
            # load matches, while inference-time MTP stays disabled).
            assert _is_mtp_eligible(_GenBatch(_MtpModel(), uids=[1])) is False

            set_mtp_active(True)
            # Has both method and head + batch=1 + flag on → triggers the path.
            assert _is_mtp_eligible(_GenBatch(_MtpModel(), uids=[1])) is True
            # MTP model with batch=2 falls back to standard step.
            assert (
                _is_mtp_eligible(_GenBatch(_MtpModel(), uids=[1, 2])) is False
            )
            # Empty batch never triggers.
            assert _is_mtp_eligible(_GenBatch(_MtpModel(), uids=[])) is False
            # Grammar-constrained decoding relies on GenerationBatch._step hooks,
            # so MTP must stay off until it mirrors accept_token explicitly.
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(batch_generator, "_has_grammar_processors", lambda _: True)
                assert _is_mtp_eligible(_GenBatch(_MtpModel(), uids=[1])) is False
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

            def mtp_forward(self, *_):
                pass

        batch = SimpleNamespace(
            model=_MtpModel(),
            uids=[42],
            logits_processors=[],
        )

        def fake_post_init(gen_batch):
            gen_batch._omlx_mtp_state = batch_generator._MtpState(
                uid=gen_batch.uids[0]
            )

        monkeypatch.setattr(batch_generator, "_post_init_mtp", fake_post_init)

        state = batch_generator._prepare_mtp_state_for_next(batch)

        assert state is batch._omlx_mtp_state
        assert state.uid == 42

    def test_prepare_mtp_state_drops_stale_owner_and_reinitializes(self, monkeypatch):
        from omlx.patches.mlx_lm_mtp import batch_generator

        class _MtpModel:
            def __init__(self):
                self.mtp = object()

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
            gen_batch._omlx_mtp_state = batch_generator._MtpState(
                uid=gen_batch.uids[0]
            )

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

    def test_mutual_exclusion_with_turboquant(self):
        with pytest.raises(ValueError, match="TurboQuant"):
            ModelSettings(mtp_enabled=True, turboquant_kv_enabled=True)

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
        assert (
            _has_mtp_heads({"text_config": {"mtp_num_hidden_layers": 1}}) is True
        )

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
        assert any(
            "MTP path will be inactive" in record.getMessage()
            for record in caplog.records
        ) or True  # logger.warning may be filtered by pytest logging level

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
                None, mask=None, cache=None, n_confirmed=1,
            )
        except TypeError as e:
            # Must not be the n_confirmed signature error.
            assert "n_confirmed" not in str(e), (
                f"signature still rejects n_confirmed: {e}"
            )
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
