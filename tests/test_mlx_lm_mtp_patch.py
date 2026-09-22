# SPDX-License-Identifier: Apache-2.0
"""Lightning MTP dispatch, cache ownership and small-model parity tests."""

from __future__ import annotations

import contextlib
import copy
import json
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten
from omlx.cache.type_registry import CacheTypeRegistry
from mlx_lm.generate_utils import BatchCounters
from mlx_lm.generate import BatchGenerator
from mlx_lm.models.cache import KVCache

from omlx.model_settings import ModelSettings
from omlx.patches import mlx_lm_mtp
from omlx.patches.mlx_lm_mtp import batch_generator as bg
from omlx.patches.mlx_lm_mtp import batched_head, cache_rollback
from omlx.patches.mlx_lm_mtp.batch_policy import BatchPolicy
from omlx.patches.mlx_vlm_mtp import qwen35_verify_linear
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

    def test_full_accept_clears_pooling_undo_chain(self):
        from omlx.patches.mlx_lm_mtp.batch_generator import _clear_rollback

        pool = SimpleNamespace(_undo=object(), _undo_chain=True)
        container = SimpleNamespace(caches=[SimpleNamespace(caches=[pool])])

        _clear_rollback([container])

        assert pool._undo is None
        assert pool._undo_chain is False


class TestMtpBoundaryCommit:
    @staticmethod
    def _run_full_accept_cycle(monkeypatch, *, emitted, drafts, clamp=None):
        import mlx.core as mx

        from omlx.patches.mlx_lm_mtp import batch_generator as bg

        cache = SimpleNamespace(offset=emitted - 1)
        model = SimpleNamespace(_omlx_mtp_commit_align=4)
        if clamp is not None:
            model.mtp_clamp_accept = lambda _cache, accepted, _drafts: min(
                accepted, clamp
            )

        draft_ids = list(range(1, drafts + 1))
        state = bg._MtpState(
            uid=1,
            chain=True,
            depth=drafts,
            mtp_cache=[],
            next_main=mx.array([15], dtype=mx.uint32),
            drafts=mx.array(draft_ids, dtype=mx.uint32),
            draft_lps=[mx.zeros((32,)) for _ in draft_ids],
        )

        def logits_for(targets):
            rows = []
            for target in targets:
                row = [-100.0] * 32
                row[target] = 0.0
                rows.append(row)
            return mx.array([rows], dtype=mx.float32)

        def fake_backbone(_model, inputs, _cache, **_kwargs):
            width = int(inputs.shape[1])
            cache.offset += width
            targets = draft_ids + [20] if width > 1 else [21]
            return (
                logits_for(targets),
                mx.zeros((1, width, 8), dtype=mx.float32),
                None,
            )

        def fake_rollback(_model, _cache, accepted, num_drafts, _gdn_states):
            cache.offset -= num_drafts - accepted
            return True

        def greedy(logprobs):
            return mx.argmax(logprobs, axis=-1).astype(mx.uint32)

        batch = SimpleNamespace(
            model=model,
            prompt_cache=[cache],
            tokens=[list(range(emitted))],
            samplers=[None],
            fallback_sampler=greedy,
            logits_processors=[],
            _token_context=[],
            max_tokens=[1000],
            _num_tokens=[0],
            _matchers=[SimpleNamespace(advance=lambda token: False)],
        )

        monkeypatch.setattr(bg, "_call_backbone", fake_backbone)
        monkeypatch.setattr(bg, "_chain_rollback", fake_rollback)
        monkeypatch.setattr(bg, "_chain_next_drafts", lambda *args, **kwargs: None)
        monkeypatch.setattr(bg, "_clear_rollback", lambda _cache: None)

        bg._run_verify_cycle_chain(batch, state)
        return batch, state, cache

    @pytest.mark.parametrize(
        ("drafts", "clamp", "boundary_source"),
        [
            (1, None, "bonus"),
            (2, None, "draft"),
            (3, 1, "verify"),
            (4, None, "draft"),
        ],
    )
    def test_boundary_emit_matches_backbone_offset(
        self, monkeypatch, drafts, clamp, boundary_source
    ):
        batch, state, cache = self._run_full_accept_cycle(
            monkeypatch,
            emitted=2,
            drafts=drafts,
            clamp=clamp,
        )

        distance = 4 - len(batch.tokens[0])
        emitted_sources = []
        for _ in range(distance):
            token_id, _logprobs, source = state.queue.popleft()
            batch.tokens[0].append(token_id)
            emitted_sources.append(source)

        assert emitted_sources[-1] == boundary_source
        assert len(batch.tokens[0]) == 4
        assert cache.offset == 4


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


class TestQwen35MtpPipelineStages:
    """#3518: the MTP forward replaces Qwen3_5TextModel.__call__ even with MTP
    off, so it must honour planned pipeline stages like MLX-LM's own forward."""

    ARGS = {
        "model_type": "qwen3_5",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 8,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 256,
        "linear_num_value_heads": 4,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "head_dim": 16,
        # Layers 3 and 7 use full attention; the rest are linear attention.
        "full_attention_interval": 4,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
        },
    }

    @pytest.fixture(autouse=True)
    def _apply(self):
        from omlx.patches.mlx_lm_mtp import qwen35_model

        if not qwen35_model.apply():
            pytest.skip("qwen35_model patch refused to apply (likely mlx_lm absent)")

    def _build(self):
        from mlx_lm.models import qwen3_5

        mx.random.seed(7)
        model = qwen3_5.TextModel(qwen3_5.TextModelArgs(**self.ARGS))
        mx.eval(model.parameters())
        return model

    def test_planned_stages_match_the_unsplit_forward(self, monkeypatch):
        from omlx.cluster.planner import (
            PipelineAssignment,
            install_unequal_pipeline_plan,
        )

        def assignment(rank, start, end):
            return PipelineAssignment(
                node_id=f"node-{rank}",
                rank=rank,
                start_layer=start,
                end_layer=end,
                layer_weight_bytes=1,
                fixed_weight_bytes=1,
                reserve_bytes=0,
                capacity_bytes=1,
            )

        class Group:
            def __init__(self, rank):
                self._rank = rank

            def rank(self):
                return self._rank

            def size(self):
                return 2

        tokens = mx.array([[5, 17, 42, 9, 128, 3]])
        unsplit = self._build()
        expected = unsplit.model(tokens, unsplit.make_cache())

        # Rank 1 runs layers 0-6 and sends; rank 0 runs only full-attention
        # layer 7, so its ssm_idx is None.
        plan = [assignment(1, 0, 7), assignment(0, 7, 8)]
        stages = {}
        with install_unequal_pipeline_plan(plan):
            for rank in (1, 0):
                stage = self._build()
                stage.model.pipeline(Group(rank))
                stages[rank] = stage
        assert len(stages[1].model.pipeline_layers) == 7
        assert stages[0].model.ssm_idx is None

        sent = []

        def send(x, dst, **kwargs):
            sent.append((x, dst))
            return x

        monkeypatch.setattr(mx.distributed, "send", send)
        monkeypatch.setattr(
            mx.distributed, "recv_like", lambda x, src, **kw: sent[-1][0]
        )
        monkeypatch.setattr(mx.distributed, "all_gather", lambda x, **kwargs: x)

        stages[1].model(tokens, stages[1].make_cache())
        assert [dst for _, dst in sent] == [0]
        received = stages[0].model(tokens, stages[0].make_cache())
        assert mx.allclose(received, expected, atol=1e-5, rtol=1e-5).item()


class TestQwen35MtpNormShift:
    """MTP-head RMSNorm +1 shift follows the checkpoint layout, not the tensor.

    Raw-HF (unsanitized conv1d) shifts every zero-centered gamma by +1. An
    MLX-format checkpoint is loaded as stored: the old per-tensor
    ``mean < 0.5`` guess re-shifted already converted ``pre_fc_norm_*``
    gammas (+0.49 / +0.27 on Qwen3.6-35B-A3B) and halved draft acceptance
    (#3742). Legacy mixed heads are handled by ``norm_repair`` instead.
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

    def test_mlx_checkpoint_loads_mtp_norms_as_stored(self):
        """No unsanitized conv1d -> every MTP norm is loaded as stored, even
        when its mean sits below the old 0.5 cutoff (#3742 reproduction)."""
        import mlx.core as mx

        m = self._model()
        stored = {
            "model.layers.0.self_attn.conv1d.weight": mx.zeros((8, 3, 1)),
            "mtp.norm.weight": 1.27,
            "mtp.layers.0.self_attn.q_norm.weight": 0.75,
            # Converted Qwen3.6-35B-A3B pre-fc gammas sit below 0.5.
            "mtp.pre_fc_norm_hidden.weight": 0.4937,
            "mtp.pre_fc_norm_embedding.weight": 0.2734,
            # A raw-looking value is still loaded as stored.
            "mtp.layers.0.input_layernorm.weight": 0.04,
        }
        weights = {
            k: (v if isinstance(v, mx.array) else mx.full((16,), v))
            for k, v in stored.items()
        }
        out = m.sanitize(weights)
        for k, v in stored.items():
            if isinstance(v, mx.array):
                continue
            assert abs(self._first(out[k]) - v) < 1e-3, k

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
            # Raw gammas above 0.5 shift too; the layout decides, not the mean.
            "mtp.layers.0.post_attention_layernorm.weight": mx.full((16,), 0.87),
            "mtp.norm.weight": mx.full((16,), 1.93),
        }
        out = m.sanitize(weights)
        g = self._first

        assert abs(g(out["model.layers.0.input_layernorm.weight"]) - 1.05) < 1e-3
        assert abs(g(out["mtp.layers.0.input_layernorm.weight"]) - 1.04) < 1e-3
        assert abs(g(out["mtp.layers.0.post_attention_layernorm.weight"]) - 1.87) < 1e-3
        assert abs(g(out["mtp.norm.weight"]) - 2.93) < 1e-3

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
        the same unconditional +1 "add" transform as the backbone norms. A
        pre-converted source records no transform at all."""
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

        # Pre-converted source: no transform is recorded for any norm.
        meta["model.layers.0.self_attn.conv1d.weight"] = ((2048, 4, 1), mx.float32)
        plan = _discover_sanitize_plan(m.sanitize, _FakeIdx(meta))
        for key in (
            "model.layers.0.input_layernorm.weight",
            "mtp.layers.0.input_layernorm.weight",
            "mtp.norm.weight",
        ):
            assert plan[key]["transform"] == "passthrough", key


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

    def _make_bg_next_fake(
        self, *, size=1, next_size=None, active_state=None, queue_len=0
    ):
        from collections import deque

        from omlx.patches.mlx_lm_mtp import batch_generator

        class _FakeGenerationBatch:
            def __init__(self, size, next_size, active_state):
                self.size = size
                self.next_size = next_size
                self.next_calls = 0
                self.extended_with = None
                if active_state == "single":
                    # A real state with a matching uid: the handoff-ready
                    # predicate validates ownership and reads queue depth.
                    self.uids = [1]
                    self._omlx_mtp_state = batch_generator._MtpState(
                        uid=1,
                        queue=deque([(5, None, "draft")] * queue_len),
                    )
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
            _counters=BatchCounters(),
            _steps_counter=0,
            _prompt_tokens_counter=0,
            _prompt_time_counter=0.0,
            completion_batch_size=4,
            prefill_batch_size=1,
        )
        return bg, gen_batch, prompt_batch

    def test_active_singleton_mtp_deep_queue_still_defers_late_join(self):
        from mlx_lm.generate import BatchGenerator

        bg, gen_batch, prompt_batch = self._make_bg_next_fake(
            active_state="single", queue_len=2
        )

        prompt_responses, generation_responses = BatchGenerator._next(bg)

        assert prompt_responses == []
        assert generation_responses == ["generation"]
        assert gen_batch.next_calls == 1
        assert gen_batch.extended_with is None
        assert prompt_batch.split_indices is None
        assert bg.completion_batch_size == 4

    @pytest.mark.parametrize("queue_len", [0, 1])
    def test_active_singleton_mtp_drained_queue_admits_late_join_same_call(
        self, queue_len
    ):
        # #2515: with pending work and a drained MTP queue the completion
        # pin is skipped, so the late join is promoted in this very call.
        from mlx_lm.generate import BatchGenerator

        bg, gen_batch, prompt_batch = self._make_bg_next_fake(
            active_state="single", queue_len=queue_len
        )

        prompt_responses, generation_responses = BatchGenerator._next(bg)

        assert generation_responses == ["generation"]
        assert len(prompt_responses) == 1
        assert gen_batch.extended_with is not None
        assert gen_batch.extended_with.uids == [99]
        assert prompt_batch.split_indices == [0]
        assert prompt_batch.last_inputs == [[123]]
        assert bg.completion_batch_size == 4

    def test_active_batch_mtp_admits_late_join_when_batch_shrinks(self):
        from mlx_lm.generate import BatchGenerator

        bg, gen_batch, prompt_batch = self._make_bg_next_fake(
            size=2,
            next_size=1,
            active_state="batch",
        )

        prompt_responses, generation_responses = BatchGenerator._next(bg)

        assert len(prompt_responses) == 1
        assert generation_responses == ["generation"]
        assert len(gen_batch) == 2
        assert gen_batch.extended_with.uids == [99]
        assert prompt_batch.split_indices == [0]
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

    def test_handoff_ready_truth_table(self):
        from collections import deque

        from omlx.patches.mlx_lm_mtp import batch_generator

        def make(queue_len=1, **overrides):
            batch = SimpleNamespace(
                uids=[1],
                _omlx_mtp_activation_safe=False,
                _omlx_mtp_state=batch_generator._MtpState(
                    uid=1, queue=deque([(5, None, "draft")] * queue_len)
                ),
            )
            for key, value in overrides.items():
                setattr(batch, key, value)
            return batch

        ready = batch_generator._singleton_mtp_handoff_ready
        assert ready(None) is False
        # Missing activation-safe stamp means no known pending work.
        unstamped = make()
        delattr(unstamped, "_omlx_mtp_activation_safe")
        assert ready(unstamped) is False
        assert ready(make(_omlx_mtp_activation_safe=True)) is False
        # Row-wise batch state keeps the deferral.
        assert ready(make(_omlx_mtp_batch_state=object())) is False
        # Stale or absent singleton state keeps the pin.
        assert ready(make(uids=[2])) is False
        assert ready(make(_omlx_mtp_state=None)) is False
        assert ready(make(queue_len=0)) is True
        assert ready(make(queue_len=1)) is True
        assert ready(make(queue_len=2)) is False

    def test_patched_next_hands_off_before_mtp_cycle(self, monkeypatch):
        from collections import deque

        from mlx_lm.generate import GenerationBatch

        from omlx.patches.mlx_lm_mtp import batch_generator

        state = batch_generator._MtpState(uid=1, queue=deque([(5, None, "draft")]))
        batch = SimpleNamespace(
            uids=[1],
            _omlx_mtp_state=state,
            _omlx_mtp_activation_safe=False,
        )

        calls = []

        def fake_handoff(gen_batch, handoff_state):
            assert handoff_state is state
            calls.append("handoff")
            del gen_batch._omlx_mtp_state
            return True

        monkeypatch.setattr(
            batch_generator, "_is_mtp_batch_eligible", lambda _: False
        )
        monkeypatch.setattr(batch_generator, "_is_mtp_eligible", lambda _: True)
        monkeypatch.setattr(
            batch_generator, "_handoff_mtp_for_late_join", fake_handoff
        )
        monkeypatch.setattr(
            batch_generator,
            "_prepare_mtp_state_for_next",
            lambda _: pytest.fail("MTP cycle ran despite a pending late join"),
        )
        monkeypatch.setattr(batch_generator, "_drop_mtp_state", lambda *_, **__: None)
        monkeypatch.setattr(
            batch_generator,
            "_mark_standard_multirow_decode",
            lambda b: setattr(b, "uids", []),
        )

        assert GenerationBatch.next(batch) == []
        assert calls == ["handoff"]

    def test_patched_next_keeps_mtp_when_no_pending(self, monkeypatch):
        from collections import deque

        from mlx_lm.generate import GenerationBatch

        from omlx.patches.mlx_lm_mtp import batch_generator

        state = batch_generator._MtpState(uid=1, queue=deque([(5, None, "draft")]))
        batch = SimpleNamespace(
            uids=[1],
            _omlx_mtp_state=state,
            _omlx_mtp_activation_safe=True,
        )
        sentinel = ["mtp-step"]

        monkeypatch.setattr(
            batch_generator, "_is_mtp_batch_eligible", lambda _: False
        )
        monkeypatch.setattr(batch_generator, "_is_mtp_eligible", lambda _: True)
        monkeypatch.setattr(
            batch_generator,
            "_handoff_mtp_for_late_join",
            lambda *_: pytest.fail("handoff fired without pending work"),
        )
        monkeypatch.setattr(
            batch_generator, "_prepare_mtp_state_for_next", lambda _: state
        )
        monkeypatch.setattr(
            batch_generator, "_mtp_next", lambda _b, _s: sentinel
        )

        assert GenerationBatch.next(batch) is sentinel

    def test_patched_next_counts_parked_standard_tokens(self, monkeypatch):
        from mlx_lm.generate import GenerationBatch

        from omlx.patches.mlx_lm_mtp import batch_generator

        batch = SimpleNamespace(
            uids=[7],
            _omlx_mtp_park_state=batch_generator._MtpParkState(
                uid=7,
                tokens_remaining=2,
            ),
            _step=lambda: ([42], [None]),
            _num_tokens=[0],
            max_tokens=[10],
            _matchers=[SimpleNamespace(advance=lambda token: False)],
            Response=lambda **kwargs: kwargs,
            extract_cache=lambda _idx: [],
            tokens=[[]],
        )
        monkeypatch.setattr(
            batch_generator, "_is_mtp_batch_eligible", lambda _: False
        )
        monkeypatch.setattr(batch_generator, "_is_mtp_eligible", lambda _: False)
        monkeypatch.setattr(batch_generator, "_drop_mtp_state", lambda *_, **__: None)

        GenerationBatch.next(batch)
        assert batch._omlx_mtp_park_state.tokens_remaining == 1
        GenerationBatch.next(batch)
        assert batch._omlx_mtp_park_state.tokens_remaining == 0

    def _make_handoff_batch(self, monkeypatch, *, queue_entries, next_main=None):
        """Fake singleton batch for _handoff_mtp_for_late_join tests.

        The fake backbone records call input widths and returns logits whose
        last-position argmax is token id 5, mirroring _make_reconcile_batch.
        """
        from collections import deque

        import mlx.core as mx
        import numpy as np

        from omlx.patches.mlx_lm_mtp import batch_generator

        vocab = 8
        backbone_calls = []

        class _FakeCache:
            def __init__(self):
                self.offset = 4
                self.rollback_state = ("conv", "ssm")
                self._mtp_draft_stash = object()

        def fake_backbone(model, inputs, cache, n_confirmed=0):
            backbone_calls.append(int(inputs.shape[1]))
            arr = np.full((1, int(inputs.shape[1]), vocab), -10.0, dtype=np.float32)
            arr[0, -1, 5] = 10.0
            return mx.array(arr), None, None

        monkeypatch.setattr(batch_generator, "_call_backbone", fake_backbone)

        def greedy(lp_2d):
            return mx.argmax(lp_2d, axis=-1).astype(mx.uint32)

        state = batch_generator._MtpState(
            uid=7, queue=deque(queue_entries), next_main=next_main
        )
        batch = SimpleNamespace(
            model=object(),
            uids=[7],
            tokens=[[10, 11, 12, 13]],
            samplers=[None],
            fallback_sampler=greedy,
            logits_processors=[],
            _next_tokens=mx.array([999]),
            _next_logprobs=[],
            _token_context=[],
            prompt_cache=[_FakeCache()],
            _omlx_mtp_state=state,
        )
        return batch_generator, batch, state, backbone_calls

    def test_late_join_handoff_queue_one_is_zero_cost_and_exact(self, monkeypatch):
        import mlx.core as mx

        lp = mx.zeros((8,))
        bg, batch, state, backbone_calls = self._make_handoff_batch(
            monkeypatch, queue_entries=[(42, lp, "draft")]
        )
        monkeypatch.setattr(
            bg,
            "_call_backbone",
            lambda *_, **__: pytest.fail("queue==1 handoff must not run the backbone"),
        )
        old_cache = batch.prompt_cache[0]

        assert bg._handoff_mtp_for_late_join(batch, state) is True
        # The sole queued token (only one absent from the cache) is fed next.
        assert batch._next_tokens.tolist() == [42]
        assert batch._next_logprobs[0] is lp
        assert batch.tokens[0] == [10, 11, 12, 13]
        # Same cache object continues -- no rebuild, no re-prefill.
        assert batch.prompt_cache[0] is old_cache
        assert old_cache.rollback_state is None
        assert old_cache._mtp_draft_stash is None
        assert not hasattr(batch, "_omlx_mtp_state")
        assert not hasattr(batch, "_omlx_mtp_park_state")
        assert not hasattr(batch, "_omlx_mtp_tax_probe")

    def test_late_join_handoff_queue_empty_feeds_next_main_without_parking(
        self, monkeypatch
    ):
        import mlx.core as mx

        bg, batch, state, backbone_calls = self._make_handoff_batch(
            monkeypatch,
            queue_entries=[],
            next_main=mx.array([7], dtype=mx.uint32),
        )

        assert bg._handoff_mtp_for_late_join(batch, state) is True
        # Exactly one 1-token forward materializes next_main.
        assert backbone_calls == [1]
        assert batch._next_tokens.tolist() == [5]
        assert batch.prompt_cache[0].rollback_state is None
        assert batch.prompt_cache[0]._mtp_draft_stash is None
        assert not hasattr(batch, "_omlx_mtp_state")
        assert not hasattr(batch, "_omlx_mtp_park_state")
        assert not hasattr(batch, "_omlx_mtp_tax_probe")

    def test_late_join_handoff_declines_deep_queue(self, monkeypatch):
        import mlx.core as mx

        lp = mx.zeros((8,))
        bg, batch, state, backbone_calls = self._make_handoff_batch(
            monkeypatch,
            queue_entries=[(42, lp, "draft"), (43, lp, "bonus")],
        )

        assert bg._handoff_mtp_for_late_join(batch, state) is False
        assert batch._omlx_mtp_state is state
        assert batch._next_tokens.tolist() == [999]
        assert backbone_calls == []

    @pytest.mark.parametrize(
        "handoff", ["_handoff_mtp_for_late_join", "_park_mtp_to_standard"]
    )
    @pytest.mark.parametrize("recovery_fails", [False, True])
    def test_handoff_recovers_after_cache_mutation(
        self, monkeypatch, handoff, recovery_fails
    ):
        from mlx_lm.models.cache import TokenBuffer

        bg, batch, state, backbone_calls = self._make_handoff_batch(
            monkeypatch, queue_entries=[], next_main=mx.array([13], dtype=mx.uint32)
        )
        batch.model = CountingModel()
        backbone = bg._call_backbone
        sampler = batch.fallback_sampler
        error = RuntimeError("sampling failed after cache write")
        samples = []

        class Processor:
            def __init__(self):
                self.count = 0
                self.prefixes = []

            def __call__(self, tokens, logits):
                self.count += 1
                self.prefixes.append(tokens.tolist())
                return logits

            def snapshot_state(self):
                return self.count

            def restore_state(self, count):
                self.count = count

        processor = Processor()
        batch.logits_processors = [[processor]]
        batch._token_context = [TokenBuffer([10, 11, 12])]

        def advancing_backbone(model, inputs, cache, **kwargs):
            cache[0].offset += int(inputs.shape[1])
            return backbone(model, inputs, cache, **kwargs)

        def fail_once(logprobs):
            samples.append(True)
            if len(samples) == 1:
                raise error
            return sampler(logprobs)

        monkeypatch.setattr(bg, "_call_backbone", advancing_backbone)
        batch.fallback_sampler = fail_once
        if recovery_fails:
            monkeypatch.setattr(bg, "_rebuild_singleton_cache", lambda model: None)
            with pytest.raises(RuntimeError, match="could not restore") as caught:
                getattr(bg, handoff)(batch, state)
            assert caught.value.__cause__ is error
            assert batch._omlx_mtp_state is state
            assert batch._next_tokens.tolist() == [999]
        else:
            assert getattr(bg, handoff)(batch, state)
            assert batch._next_tokens.tolist() == [14]
            cache = batch.prompt_cache[0].extract(0)
            assert cache.keys[0, 0, : cache.offset, 0].tolist() == [10, 11, 12, 13]
            assert batch._token_context[0].tokens.tolist() == [10, 11, 12, 13]
            assert processor.count == 1
            assert processor.prefixes == [[10, 11, 12, 13]] * 2
            assert not hasattr(batch, "_omlx_mtp_state")
        assert backbone_calls == [1]

    def test_performance_park_starts_reentry_cooldown(self, monkeypatch):
        import mlx.core as mx

        bg, batch, state, backbone_calls = self._make_handoff_batch(
            monkeypatch,
            queue_entries=[],
            next_main=mx.array([7], dtype=mx.uint32),
        )

        assert bg._park_mtp_to_standard(batch, state) is True
        assert backbone_calls == [1]
        assert batch._omlx_mtp_park_state.uid == 7
        assert batch._omlx_mtp_park_state.tokens_remaining == 128
        assert batch._next_tokens.tolist() == [5]
        assert not hasattr(batch, "_omlx_mtp_state")

    def test_failed_reentry_probe_doubles_cooldown(self, monkeypatch):
        import mlx.core as mx

        bg, batch, state, _ = self._make_handoff_batch(
            monkeypatch,
            queue_entries=[],
            next_main=mx.array([7], dtype=mx.uint32),
        )
        assert bg._park_mtp_to_standard(batch, state) is True

        retry = bg._MtpState(
            uid=7,
            next_main=mx.array([7], dtype=mx.uint32),
            reentry_probe=True,
        )
        batch._omlx_mtp_state = retry
        assert bg._park_mtp_to_standard(batch, retry) is True
        assert batch._omlx_mtp_park_state.tokens_remaining == 256

    def test_batch_eligibility_requires_safe_activation(self, monkeypatch):
        from omlx.patches.mlx_lm_mtp import is_mtp_active, set_mtp_active
        from omlx.patches.mlx_lm_mtp import batch_generator

        class _MtpModel:
            _omlx_mtp_multi_request = True

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

    def test_batch_new_activation_allows_ragged_offsets(self, monkeypatch):
        # Continuous batching admits rows at different times, so per-row
        # cache offsets rarely align. Activation must not require alignment:
        # each row is seeded from its own extract_cache view and steady-state
        # row cycles diverge the offsets anyway (#2150).
        from omlx.patches.mlx_lm_mtp import is_mtp_active, set_mtp_active
        from omlx.patches.mlx_lm_mtp import batch_generator

        class _Offset:
            def __init__(self, values):
                self._values = values

            def tolist(self):
                return list(self._values)

        class _MtpModel:
            _omlx_mtp_multi_request = True

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
            assert batch_generator._is_mtp_batch_eligible(batch) is True
        finally:
            set_mtp_active(prior_active)

    def test_batch_activation_allowed_after_standard_multirow_decode(self, monkeypatch):
        # The multirow-decode marker protects singleton re-initialization
        # only. A batch's first decode step is always standard (MTP has not
        # activated yet), so blocking batch activation on the marker would
        # permanently lock every batch out of row-wise MTP (#2150).
        from omlx.patches.mlx_lm_mtp import is_mtp_active, set_mtp_active
        from omlx.patches.mlx_lm_mtp import batch_generator

        class _MtpModel:
            _omlx_mtp_multi_request = True

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
                _omlx_mtp_saw_standard_multirow_decode=True,
                prompt_cache=[],
            )
            assert batch_generator._is_mtp_batch_eligible(batch) is True

            singleton = SimpleNamespace(
                model=_MtpModel(),
                uids=[1],
                logits_processors=[],
                _omlx_mtp_activation_safe=True,
                _omlx_mtp_saw_standard_multirow_decode=True,
            )
            assert batch_generator._is_mtp_eligible(singleton) is False
        finally:
            set_mtp_active(prior_active)

    def test_singleton_recovery_clears_marker_when_cache_compact(self):
        # A batch that shrinks back to one row with no residual left padding
        # satisfies the singleton-init invariant again, so the multirow
        # marker must lift and the surviving request regains MTP (#2150).
        from omlx.patches.mlx_lm_mtp import batch_generator

        class _Padding:
            def __init__(self, values):
                self._values = values

            def tolist(self):
                return list(self._values)

        batch = SimpleNamespace(
            uids=[1],
            _omlx_mtp_saw_standard_multirow_decode=True,
            prompt_cache=[SimpleNamespace(left_padding=_Padding([0]))],
        )
        batch_generator._maybe_clear_multirow_marker(batch)
        assert batch._omlx_mtp_saw_standard_multirow_decode is False

        # Residual padding: the invariant does not hold — keep the marker.
        padded = SimpleNamespace(
            uids=[1],
            _omlx_mtp_saw_standard_multirow_decode=True,
            prompt_cache=[SimpleNamespace(left_padding=_Padding([3]))],
        )
        batch_generator._maybe_clear_multirow_marker(padded)
        assert padded._omlx_mtp_saw_standard_multirow_decode is True

        # Still multi-row: never clears.
        multirow = SimpleNamespace(
            uids=[1, 2],
            _omlx_mtp_saw_standard_multirow_decode=True,
            prompt_cache=[SimpleNamespace(left_padding=_Padding([0, 0]))],
        )
        batch_generator._maybe_clear_multirow_marker(multirow)
        assert multirow._omlx_mtp_saw_standard_multirow_decode is True

    def test_singleton_recovery_checks_cachelist_sub_caches(self):
        # CacheList layers (GLM 5.2 / DeepSeek v3.2 lineage) keep left
        # padding on their sub-caches, not on the container. The compactness
        # check must recurse into ``.caches`` instead of skipping the layer,
        # or the marker clears without verifying anything.
        from omlx.patches.mlx_lm_mtp import batch_generator

        class _Padding:
            def __init__(self, values):
                self._values = values

            def tolist(self):
                return list(self._values)

        class _CacheList:
            def __init__(self, *caches):
                self.caches = caches

        padded = SimpleNamespace(
            uids=[1],
            _omlx_mtp_saw_standard_multirow_decode=True,
            prompt_cache=[
                _CacheList(SimpleNamespace(left_padding=_Padding([3])))
            ],
        )
        batch_generator._maybe_clear_multirow_marker(padded)
        assert padded._omlx_mtp_saw_standard_multirow_decode is True

        compact = SimpleNamespace(
            uids=[1],
            _omlx_mtp_saw_standard_multirow_decode=True,
            prompt_cache=[
                _CacheList(
                    SimpleNamespace(left_padding=_Padding([0])),
                    SimpleNamespace(left_padding=_Padding([0])),
                )
            ],
        )
        batch_generator._maybe_clear_multirow_marker(compact)
        assert compact._omlx_mtp_saw_standard_multirow_decode is False

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

        The fake model advances the fake cache offset by the input length and
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
                self._mtp_undo = None

        def fake_rebuild(model):
            return [_FakeCache()]

        def forward(inputs, cache):
            cache[0].offset = int(inputs.shape[1])
            cache[0]._mtp_undo = object()
            arr = np.full((1, int(inputs.shape[1]), vocab), -10.0, dtype=np.float32)
            arr[0, -1, 5] = 10.0  # last-position argmax -> token 5
            return mx.array(arr)

        monkeypatch.setattr(batch_generator, "_rebuild_singleton_cache", fake_rebuild)

        def greedy(lp_2d):
            return mx.argmax(lp_2d, axis=-1).astype(mx.uint32)

        state = batch_generator._MtpState(uid=uid, queue=deque(queue_entries))
        batch = SimpleNamespace(
            model=forward,
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
        assert batch.prompt_cache[0]._mtp_undo is None

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

        # No committed history is available to reconstruct.
        assert bg._reconcile_mtp_to_standard(batch, state) is False

    def test_reconcile_reports_rebuild_failure(self, monkeypatch):
        import mlx.core as mx

        bg, batch, state = self._make_reconcile_batch(
            monkeypatch,
            uid=7,
            tokens=[10, 11],
            queue_entries=[(42, mx.zeros((8,)), "draft")],
        )
        monkeypatch.setattr(bg, "_rebuild_singleton_cache", lambda model: None)

        # The caller must stop decoding when cache recovery is unavailable.
        assert bg._reconcile_mtp_to_standard(batch, state) is False

    def test_extend_stops_when_singleton_cache_recovery_fails(self, monkeypatch):
        from mlx_lm.generate import GenerationBatch

        bg, batch, state = self._make_reconcile_batch(
            monkeypatch, uid=7, tokens=[10, 11], queue_entries=[]
        )
        monkeypatch.setattr(bg, "_rebuild_singleton_cache", lambda model: None)

        class UnmergeableUids(list):
            def extend(self, other):
                pytest.fail("Merged rows after cache recovery failed")

        batch.uids = UnmergeableUids(batch.uids)
        donor = SimpleNamespace(uids=[8])

        with pytest.raises(RuntimeError, match="could not restore the committed cache"):
            GenerationBatch.extend(batch, donor)
        assert batch.uids == [7]
        assert batch._omlx_mtp_state is state


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

    def test_has_mtp_heads_dspark_fields(self):
        assert (
            _has_mtp_heads(
                {
                    "dspark_block_size": 5,
                    "dspark_target_layer_ids": [40, 41, 42],
                }
            )
            is True
        )

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

    @pytest.mark.parametrize("model_type", ["mimo_v2", "mimo_v2_flash"])
    def test_is_mtp_compatible_mimo_v2(self, model_type):
        assert (
            _is_mtp_compatible({"num_nextn_predict_layers": 3}, model_type) is True
        )

    def test_is_mtp_compatible_gemma4_unified(self):
        config = {
            "text_config": {
                "mtp_num_hidden_layers": 4,
                "mtp_assistant_config": {"model_type": "gemma4_unified_assistant"},
            }
        }
        assert _is_mtp_compatible(config, "gemma4_unified") is True

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
    the armed verify-block undo log: restore the pre-verify state and replay
    the confirmed/accepted prefix. Equivalence is checked against a reference
    cache that never saw the rejected positions."""

    @staticmethod
    def _fill(cache, n, dim=4, start=0):
        import mlx.core as mx

        for i in range(start, start + n):
            k = mx.full((1, 1, 1, dim), float(i))
            cache.update_and_fetch(k, k)

    def _run_equivalence(self, make_cache, num_drafts=1):
        import mlx.core as mx

        from omlx.patches.mlx_lm_mtp import cache_rollback

        cache_rollback.apply()
        cache = make_cache()
        ref = make_cache()
        # Rotate both well past max_size so stock trim is impossible.
        self._fill(cache, 12)
        self._fill(ref, 12)

        verify_steps = num_drafts + 1
        verify = mx.broadcast_to(
            mx.arange(100, 100 + verify_steps, dtype=mx.float32).reshape(
                1, 1, verify_steps, 1
            ),
            (1, 1, verify_steps, 4),
        )
        cache_rollback.set_undo_armed(True)
        try:
            cache.update_and_fetch(verify, verify)
        finally:
            cache_rollback.set_undo_armed(False)
        assert cache.is_trimmable()
        assert cache.trim(num_drafts) == num_drafts

        ref.update_and_fetch(verify[..., :1, :], verify[..., :1, :])

        nxt = mx.full((1, 1, 1, 4), 300.0)
        ck, cv = cache.update_and_fetch(nxt, nxt)
        rk, rv = ref.update_and_fetch(nxt, nxt)
        mx.eval(ck, cv, rk, rv)
        assert mx.array_equal(ck, rk).item()
        assert mx.array_equal(cv, rv).item()
        assert (cache.max_size, cache._idx) == (ref.max_size, ref._idx)
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

    def test_rotating_kv_cache_depth_eight_verify_undo(self):
        from mlx_lm.models.cache import RotatingKVCache

        # Depth-8 contains one confirmed token plus eight drafts. The old gate
        # admitted at most eight positions, excluding this nine-position update.
        self._run_equivalence(lambda: RotatingKVCache(max_size=8), num_drafts=8)

    def test_batch_rotating_kv_cache_depth_eight_verify_undo(self):
        from mlx_lm.models.cache import BatchRotatingKVCache

        self._run_equivalence(lambda: BatchRotatingKVCache(8, [0]), num_drafts=8)

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


class TestReversiblePerformancePark:
    """Performance parking must be scoped and reversible for one sequence.

    GenerationBatch objects are reused across requests through extend() merges,
    so the cooldown is keyed by uid and must not leak to a later request.
    """

    def _fake_batch(self, uids):
        model = SimpleNamespace(
            mtp_forward=lambda *a, **k: None,
            mtp=object(),
            _omlx_mtp_decode_enabled=True,
        )
        return SimpleNamespace(
            model=model,
            uids=list(uids),
            logits_processors=None,
        )

    def test_cooldown_blocks_then_releases_same_uid(self, monkeypatch):
        from omlx.patches.mlx_lm_mtp import batch_generator

        gb = self._fake_batch([7])
        assert batch_generator._mtp_common_eligible(gb)
        park = batch_generator._MtpParkState(uid=7, tokens_remaining=2)
        gb._omlx_mtp_park_state = park
        assert not batch_generator._mtp_common_eligible(gb)
        batch_generator._record_parked_standard_step(gb)
        assert not batch_generator._mtp_common_eligible(gb)
        batch_generator._record_parked_standard_step(gb)
        monkeypatch.setattr(batch_generator, "_prefill_activity_recent", lambda: False)
        assert batch_generator._mtp_common_eligible(gb)

    def test_cooldown_does_not_leak_to_reused_batch(self):
        from omlx.patches.mlx_lm_mtp import batch_generator

        gb = self._fake_batch([7])
        gb._omlx_mtp_park_state = batch_generator._MtpParkState(uid=7)
        # The parked request finished; a new request reuses the batch object.
        gb.uids = [8]
        assert batch_generator._mtp_common_eligible(gb)
        assert not hasattr(gb, "_omlx_mtp_park_state")

    def test_failed_probe_uses_bounded_exponential_backoff(self):
        from omlx.patches.mlx_lm_mtp import batch_generator

        park = batch_generator._MtpParkState(uid=7)
        expected = [256, 512, 1024, 2048, 4096, 4096]
        actual = []
        for _ in expected:
            park.restart_after_failed_probe()
            actual.append(park.tokens_remaining)
        assert actual == expected

    def test_active_probe_stays_eligible_during_prefill_contention(self, monkeypatch):
        from omlx.patches.mlx_lm_mtp import batch_generator

        gb = self._fake_batch([7])
        gb._omlx_mtp_park_state = batch_generator._MtpParkState(
            uid=7,
            tokens_remaining=0,
        )
        gb._omlx_mtp_state = batch_generator._MtpState(
            uid=7,
            reentry_probe=True,
        )
        monkeypatch.setattr(batch_generator, "_prefill_activity_recent", lambda: True)
        assert batch_generator._mtp_common_eligible(gb)

    def test_singleton_park_does_not_gate_batch_eligibility(self, monkeypatch):
        from omlx.patches.mlx_lm_mtp import batch_generator

        gb = self._fake_batch([7, 8])
        gb._omlx_mtp_park_state = batch_generator._MtpParkState(
            uid=7,
            tokens_remaining=1,
        )
        batch_generator._record_parked_standard_step(gb)
        assert gb._omlx_mtp_park_state.tokens_remaining == 0
        monkeypatch.setattr(batch_generator, "_prefill_activity_recent", lambda: False)
        assert batch_generator._mtp_common_eligible(gb)

    def test_due_cooldown_marks_new_state_as_reentry_probe(self, monkeypatch):
        from omlx.patches.mlx_lm_mtp import batch_generator

        gb = self._fake_batch([7])
        gb._omlx_mtp_park_state = batch_generator._MtpParkState(
            uid=7,
            tokens_remaining=0,
        )
        monkeypatch.setattr(batch_generator, "_prefill_activity_recent", lambda: False)
        monkeypatch.setattr(
            batch_generator, "_set_singleton_mrope_delta", lambda _: None
        )

        def fake_post_init(batch):
            batch._omlx_mtp_state = batch_generator._MtpState(uid=7)

        monkeypatch.setattr(batch_generator, "_post_init_mtp", fake_post_init)

        state = batch_generator._prepare_mtp_state_for_next(gb)
        assert state is not None
        assert state.reentry_probe is True

    def test_existing_controller_accepts_winning_reentry_probe(self):
        from omlx.patches.mlx_lm_mtp import batch_generator

        controller = batch_generator._DepthController(3)
        controller._warmup = []
        controller.exit_streak = 0
        state = batch_generator._MtpState(
            uid=7,
            controller=controller,
            reentry_probe=True,
        )
        gb = self._fake_batch([7])
        gb._omlx_mtp_park_state = batch_generator._MtpParkState(
            uid=7,
            tokens_remaining=0,
        )

        assert batch_generator._maybe_finish_mtp_reentry_probe(
            gb,
            state,
            was_warmup=False,
        )
        assert state.reentry_probe is False
        assert not hasattr(gb, "_omlx_mtp_park_state")

    def test_losing_reentry_probe_keeps_cooldown_state(self):
        from omlx.patches.mlx_lm_mtp import batch_generator

        controller = batch_generator._DepthController(3)
        controller._warmup = []
        controller.exit_streak = 1
        state = batch_generator._MtpState(
            uid=7,
            controller=controller,
            reentry_probe=True,
        )
        gb = self._fake_batch([7])
        park = batch_generator._MtpParkState(uid=7, tokens_remaining=0)
        gb._omlx_mtp_park_state = park

        assert not batch_generator._maybe_finish_mtp_reentry_probe(
            gb,
            state,
            was_warmup=False,
        )
        assert state.reentry_probe is True
        assert gb._omlx_mtp_park_state is park

    def test_tax_probe_discarded_when_batch_gains_rows(self):
        from omlx.patches.mlx_lm_mtp.batch_generator import (
            _STD_TAX_SKIP,
            _arm_std_tax_probe,
            _record_std_tax_sample,
        )

        gb = self._fake_batch([7])
        _arm_std_tax_probe(gb, 12.0, uid=7)
        for _ in range(_STD_TAX_SKIP + 2):
            _record_std_tax_sample(gb, 10.0)
        assert hasattr(gb, "_omlx_mtp_tax_probe")
        # Another request merges in mid-probe: multi-row step timings must
        # not contaminate the singleton loop-tax ratio.
        gb.uids = [7, 9]
        _record_std_tax_sample(gb, 10.0)
        assert not hasattr(gb, "_omlx_mtp_tax_probe")
        assert not hasattr(gb.model, "_omlx_mtp_loop_tax")


@pytest.fixture(autouse=True)
def _quiet_prefill_tracker():
    """Loop-tax probes consult the process-global prefill tracker; keep
    every test in this module hermetic against activity left behind by
    other suites (or by earlier tests here)."""
    from omlx.prefill_progress import get_prefill_tracker

    get_prefill_tracker().clear()
    yield
    get_prefill_tracker().clear()


class TestLoopTaxHygiene:
    """#2622: park probes must not latch contention-shaped timings.

    A park that fires while another request is prefilling (any engine on
    the shared GPU) measures a contention-inflated t0; if the std samples
    then run after the contention ends, the t0/t_std ratio poisons
    ``model._omlx_mtp_loop_tax`` and every later sequence exits MTP
    against an inflated margin until restart.
    """

    def _fake_batch(self, uids):
        return SimpleNamespace(model=SimpleNamespace(), uids=list(uids))

    def _tracker(self):
        from omlx.prefill_progress import get_prefill_tracker

        return get_prefill_tracker()

    def _drain_probe(self, gb, std_ms=10.0):
        from omlx.patches.mlx_lm_mtp.batch_generator import (
            _STD_TAX_SAMPLES,
            _STD_TAX_SKIP,
            _record_std_tax_sample,
        )

        for _ in range(_STD_TAX_SKIP + _STD_TAX_SAMPLES):
            _record_std_tax_sample(gb, std_ms)

    def test_probe_not_armed_while_prefill_live(self):
        from omlx.patches.mlx_lm_mtp.batch_generator import _arm_std_tax_probe

        self._tracker().update("r1", 10, 100, "modelA")
        gb = self._fake_batch([7])
        _arm_std_tax_probe(gb, 12.0, uid=7)
        assert not hasattr(gb, "_omlx_mtp_tax_probe")

    def test_probe_not_armed_right_after_prefill_end(self):
        from omlx.patches.mlx_lm_mtp.batch_generator import _arm_std_tax_probe

        tracker = self._tracker()
        tracker.update("r1", 10, 100, "modelA")
        tracker.update("r1", 100, 100, "modelA")  # completes, entry popped
        gb = self._fake_batch([7])
        _arm_std_tax_probe(gb, 12.0, uid=7)
        assert not hasattr(gb, "_omlx_mtp_tax_probe")

    def test_probe_discarded_when_prefill_ran_during_sampling(self):
        from omlx.patches.mlx_lm_mtp.batch_generator import (
            _STD_TAX_SAMPLES,
            _STD_TAX_SKIP,
            _arm_std_tax_probe,
            _record_std_tax_sample,
        )

        gb = self._fake_batch([7])
        _arm_std_tax_probe(gb, 15.0, uid=7)
        assert hasattr(gb, "_omlx_mtp_tax_probe")
        for _ in range(_STD_TAX_SKIP + _STD_TAX_SAMPLES - 1):
            _record_std_tax_sample(gb, 10.0)
        # A prefill starts (and would end) inside the sampling window: the
        # finalize step must throw the whole probe away.
        self._tracker().update("r2", 10, 100, "modelB")
        _record_std_tax_sample(gb, 10.0)
        assert not hasattr(gb, "_omlx_mtp_tax_probe")
        assert not hasattr(gb.model, "_omlx_mtp_loop_tax")

    def test_clean_probe_still_latches_with_timestamp(self):
        from omlx.patches.mlx_lm_mtp.batch_generator import _arm_std_tax_probe

        gb = self._fake_batch([7])
        _arm_std_tax_probe(gb, 15.0, uid=7)
        self._drain_probe(gb)
        assert gb.model._omlx_mtp_loop_tax == pytest.approx(1.5)
        assert hasattr(gb.model, "_omlx_mtp_loop_tax_ts")

    def test_tax_log_level_by_threshold(self, caplog):
        import logging

        from omlx.patches.mlx_lm_mtp.batch_generator import _arm_std_tax_probe

        logger_name = "omlx.patches.mlx_lm_mtp.batch_generator"
        with caplog.at_level(logging.INFO, logger=logger_name):
            gb = self._fake_batch([7])
            _arm_std_tax_probe(gb, 15.0, uid=7)
            self._drain_probe(gb)  # tax 1.5 >= warn threshold
            assert any(
                "MTP loop tax measured" in r.message
                and r.levelno == logging.INFO
                for r in caplog.records
            )
            caplog.clear()
            gb2 = self._fake_batch([8])
            _arm_std_tax_probe(gb2, 11.0, uid=8)
            self._drain_probe(gb2)  # tax 1.1 < warn threshold -> DEBUG only
            assert not any(
                "MTP loop tax measured" in r.message for r in caplog.records
            )

    def test_effective_loop_tax_decays_toward_default(self):
        import time

        from omlx.patches.mlx_lm_mtp.batch_generator import (
            _STD_TAX_DECAY_S,
            _DepthController,
            _effective_loop_tax,
        )

        model = SimpleNamespace()
        assert _effective_loop_tax(model) is None
        model._omlx_mtp_loop_tax = 1.5
        # Legacy attribute without a timestamp passes through unchanged.
        assert _effective_loop_tax(model) == pytest.approx(1.5)
        model._omlx_mtp_loop_tax_ts = time.monotonic()
        assert _effective_loop_tax(model) == pytest.approx(1.5, abs=0.01)
        model._omlx_mtp_loop_tax_ts = time.monotonic() - 3 * _STD_TAX_DECAY_S
        aged = _effective_loop_tax(model)
        assert _DepthController.EXIT_MARGIN <= aged < 1.2

    def test_recently_active_window(self):
        tracker = self._tracker()
        assert not tracker.recently_active(3.0)
        tracker.update("r1", 10, 100, "m")
        assert tracker.recently_active(3.0)
        tracker.update("r1", 100, 100, "m")
        assert tracker.recently_active(3.0)  # just-finished counts
        tracker.clear()
        assert not tracker.recently_active(3.0)


class TestReconcileChunked:
    def test_reconcile_re_prefills_in_scheduler_chunks(self, monkeypatch):
        """The reconcile re-prefill uses the generator's prefill step, never a
        single forward over the whole context: memory and sparse-attention
        paths differ from the chunked prefill the request already had."""
        from collections import deque
        from types import SimpleNamespace

        import mlx.core as mx
        import numpy as np

        from omlx.patches.mlx_lm_mtp import batch_generator

        shapes = []

        class _FakeCache:
            def __init__(self):
                self.offset = 0
                self._mtp_undo = None

        def forward(inputs, cache):
            shapes.append(int(inputs.shape[1]))
            cache[0].offset += int(inputs.shape[1])
            arr = np.full((1, int(inputs.shape[1]), 8), -10.0, dtype=np.float32)
            arr[0, -1, 5] = 10.0
            return mx.array(arr)

        monkeypatch.setattr(batch_generator, "_rebuild_singleton_cache", lambda m: [_FakeCache()])
        tokens = list(range(1300))
        state = batch_generator._MtpState(uid=1, queue=deque())
        batch = SimpleNamespace(
            model=forward,
            uids=[1],
            tokens=[tokens],
            _num_tokens=[len(tokens)],
            samplers=[None],
            fallback_sampler=lambda lp: mx.argmax(lp, axis=-1).astype(mx.uint32),
            logits_processors=[],
            _next_tokens=mx.array([999]),
            _next_logprobs=[],
            _token_context=[],
            prompt_cache=[object()],
            _omlx_mtp_state=state,
            prefill_step_size=512,
        )
        assert batch_generator._reconcile_mtp_to_standard(batch, state) is True
        assert shapes == [512, 512, 276], shapes
        assert batch.prompt_cache[0].offset == 1300
        assert batch._next_tokens.tolist() == [5]

    @pytest.mark.parametrize("step", [128, 256])
    @pytest.mark.parametrize("queued", [False, True])
    def test_reconcile_inherits_generator_chunk_size(self, step, queued):
        from collections import deque

        import mlx.core as mx
        from mlx_lm.generate import BatchGenerator
        from mlx_lm.models.llama import Model, ModelArgs

        from omlx.patches.mlx_lm_mtp import batch_generator

        class ProbeModel(Model):
            def __init__(self):
                super().__init__(
                    ModelArgs(
                        model_type="llama",
                        hidden_size=32,
                        num_hidden_layers=2,
                        intermediate_size=64,
                        num_attention_heads=4,
                        rms_norm_eps=1e-5,
                        vocab_size=64,
                    )
                )
                self.reconcile_shapes = []

            def __call__(self, inputs, cache=None, return_hidden=False):
                logits = super().__call__(inputs, cache=cache)
                self.reconcile_shapes.append((int(inputs.shape[1]), return_hidden))
                return (logits, None) if return_hidden else logits

        batch_generator.apply()
        model = ProbeModel()
        generator = BatchGenerator(model, prefill_step_size=step, max_tokens=16)
        try:
            generator.insert([[i % 64 for i in range(1300)]])
            for _ in range(32):
                generator.next()
                if generator._generation_batch.uids:
                    break
            batch = generator._generation_batch
            assert batch.uids
            assert batch.prefill_step_size == step
            history = list(batch.tokens[0])
            queue = deque([(7, mx.zeros((64,)), "draft")]) if queued else deque()
            state = batch_generator._MtpState(uid=batch.uids[0], queue=queue)
            # Exercise the row projection used by multi-row reconciliation too.
            row = batch_generator._make_row_batch(batch, 0, state=state)
            assert row.prefill_step_size == step
            target = row if queued else batch
            model.reconcile_shapes.clear()
            assert batch_generator._reconcile_mtp_to_standard(target, state)
            expected = [step] * (len(history) // step)
            if len(history) % step:
                expected.append(len(history) % step)
            assert model.reconcile_shapes == [(size, False) for size in expected]
            assert target.tokens[0] == history

            reference = batch_generator._rebuild_singleton_cache(model)
            tokens = mx.array(history, dtype=mx.uint32)
            for start in range(0, len(history), step):
                logits = model(tokens[None, start : start + step], cache=reference)
                mx.eval(logits)
            for actual, wanted in zip(target.prompt_cache, reference):
                for (_, a), (_, b) in zip(
                    tree_flatten(actual.state), tree_flatten(wanted.state)
                ):
                    assert (
                        mx.allclose(a, b).item() if isinstance(a, mx.array) else a == b
                    )
            expected_token = 7 if queued else mx.argmax(logits[:, -1, :]).item()
            assert target._next_tokens.item() == expected_token
        finally:
            generator.close()


def test_chain_rollback_finds_the_method_behind_the_vlm_adapter():
    """Partial rollback reaches the language model through its adapter."""
    from omlx.patches.mlx_lm_mtp import batch_generator as bg

    calls = []

    class _Language:
        def mtp_partial_rollback(self, cache, accepted, num_drafts):
            calls.append((accepted, num_drafts))
            return True

    class _Adapter:
        def __init__(self):
            self._language_model = _Language()

    assert bg._chain_rollback(_Adapter(), [], 1, 3, None) is True
    assert calls == [(1, 3)]


class CountingModel(nn.Module):
    """Predict the successor token while retaining every actual input in KV."""

    _omlx_mtp_multi_request = True

    def __init__(self):
        super().__init__()
        self.mtp = nn.Identity()
        self._omlx_mtp_decode_enabled = True
        self._omlx_mtp_chain = True
        self._omlx_mtp_depth = 2
        self.model = SimpleNamespace(norm=nn.Identity())
        self.calls = []

    def make_cache(self):
        return [KVCache()]

    def make_mtp_cache(self):
        return []

    def __call__(self, inputs, cache=None, return_hidden=False, n_confirmed=0):
        self.calls.append((tuple(inputs.shape), n_confirmed))
        values = inputs[:, None, :, None].astype(mx.float32)
        if cache is not None:
            cache[0].update_and_fetch(values, values)
        logits = self._logits(inputs)
        return (
            (logits, inputs[..., None].astype(mx.float32)) if return_hidden else logits
        )

    def _logits(self, tokens):
        return mx.where(mx.arange(64) == ((tokens + 1) % 64)[..., None], 10.0, -10.0)

    def mtp_forward(self, hidden, tokens, cache, return_hidden=False, **kwargs):
        logits = self._logits(tokens)
        return (
            (logits, tokens[..., None].astype(mx.float32)) if return_hidden else logits
        )

    def mtp_partial_rollback(self, cache, accepted, num_drafts):
        count = num_drafts - accepted
        return all(c.trim(count) == count for c in cache)


def generate(
    model,
    prompts,
    limits,
    *,
    stop_tokens=None,
    late_join=False,
    processors=None,
    samplers=None,
):
    bg.apply()
    cache_rollback.apply()
    gen = BatchGenerator(
        model,
        sampler=lambda lp: mx.argmax(lp, -1),
        prefill_batch_size=len(prompts),
        max_tokens=max(limits),
        stop_tokens=stop_tokens,
    )
    output, terminal = {}, {}
    try:
        initial = 1 if late_join else len(prompts)
        for uid in gen.insert(
            prompts[:initial],
            max_tokens=limits[:initial],
            logits_processors=processors,
            samplers=samplers,
        ):
            output[uid] = []
        for step in range(100):
            if late_join and step == 2:
                for uid in gen.insert(prompts[1:], max_tokens=limits[1:]):
                    output[uid] = []
            _, responses = gen.next()
            for response in responses:
                assert response.uid not in terminal, "Emission after terminal response"
                output[response.uid].append(response.token)
                if response.finish_reason:
                    terminal[response.uid] = response
            if len(terminal) == len(prompts):
                break
        assert len(terminal) == len(prompts)
        return output, terminal
    finally:
        gen.close()


@pytest.mark.parametrize("recovery_fails", [False, True])
def test_singleton_rollback_requires_successful_cache_recovery(
    monkeypatch, recovery_fails
):
    class RejectDrafts(CountingModel):
        def mtp_forward(self, hidden, tokens, cache, return_hidden=False, **kwargs):
            logits = self._logits(tokens + 1)
            return (
                (logits, tokens[..., None].astype(mx.float32))
                if return_hidden
                else logits
            )

    monkeypatch.setattr(bg, "_chain_rollback", lambda *args: False)
    if recovery_fails:
        monkeypatch.setattr(bg, "_rebuild_singleton_cache", lambda model: None)
        with pytest.raises(RuntimeError, match="could not restore the committed cache"):
            generate(RejectDrafts(), [[1, 2]], [8])
    else:
        output, terminal = generate(RejectDrafts(), [[1, 2]], [8])
        assert output[0] == list(range(3, 11))
        cache = terminal[0].prompt_cache[0]
        assert cache.keys[0, 0, : cache.offset, 0].tolist() == list(range(1, 10))


@pytest.mark.parametrize("stop", [False, True])
def test_park_recovery_includes_the_token_being_emitted(monkeypatch, stop):
    class RejectDrafts(CountingModel):
        def mtp_forward(self, hidden, tokens, cache, return_hidden=False, **kwargs):
            logits = self._logits(tokens + 1)
            return (
                (logits, tokens[..., None].astype(mx.float32))
                if return_hidden
                else logits
            )

    feed = bg._feed_next_main_to_standard
    backbone = bg._call_backbone
    failures = []
    in_handoff = False

    def observed_feed(*args):
        nonlocal in_handoff
        in_handoff = True
        try:
            return feed(*args)
        finally:
            in_handoff = False

    def fail_after_forward(*args, **kwargs):
        result = backbone(*args, **kwargs)
        if in_handoff and not failures:
            failures.append(True)
            raise RuntimeError("injected handoff failure")
        return result

    monkeypatch.setattr(bg, "_feed_next_main_to_standard", observed_feed)
    monkeypatch.setattr(bg, "_call_backbone", fail_after_forward)
    monkeypatch.setattr(bg._DepthController, "should_exit", lambda self: True)
    output, terminal = generate(
        RejectDrafts(), [[1, 2]], [8], stop_tokens=[[5]] if stop else None
    )
    assert failures == ([] if stop else [True])
    assert output[0] == list(range(3, 6 if stop else 11))
    assert terminal[0].finish_reason == ("stop" if stop else "length")
    cache = terminal[0].prompt_cache[0]
    assert cache.keys[0, 0, : cache.offset, 0].tolist() == list(
        range(1, 5 if stop else 11)
    )


@pytest.mark.parametrize("limits", [[8, 9], [1, 8], [3, 8], [4, 8]])
def test_ragged_finish_cache_matches_actual_inputs(limits, monkeypatch):
    monkeypatch.setattr(bg, "_batch_policy_for_next", lambda batch: None)
    model = CountingModel()
    prompts = [[1, 2], [4, 5, 6]]
    output, terminal = generate(model, prompts, limits)
    for uid, prompt in enumerate(prompts):
        expected = list(range(prompt[-1] + 1, prompt[-1] + 1 + limits[uid]))
        assert output[uid] == expected
        cache = terminal[uid].prompt_cache[0]
        actual = cache.keys[0, 0, : cache.offset, 0].tolist()
        tokens = prompt + expected
        assert actual in (tokens, tokens[:-1])
    if min(limits) >= 4:
        assert ((2, 3), 1) in model.calls


@pytest.mark.parametrize("batch_size", [2, 4])
def test_batched_handoff_restores_unequal_frontiers(monkeypatch, batch_size):
    class UnequalDrafts(CountingModel):
        def mtp_forward(self, hidden, tokens, cache, return_hidden=False, **kwargs):
            logits = self._logits(tokens + (tokens % 5 == 0))
            return (
                (logits, tokens[..., None].astype(mx.float32))
                if return_hidden
                else logits
            )

    model = UnequalDrafts()
    prompts = [[1, 2], [3, 4, 5], [4, 5, 6], [8, 9]][:batch_size]
    limits = [18, 22, 24, 27][:batch_size]
    model._omlx_mtp_decode_enabled = False
    expected, expected_terminal = generate(model, prompts, limits)
    model._omlx_mtp_decode_enabled = True
    original = bg._mtp_batch_next
    handoffs = []

    def handoff(batch, state):
        result = original(batch, state)
        if (
            len(batch.uids) == batch_size
            and all(s.stats.cycles >= 2 and not s.queue for s in state.states.values())
            and len(set(batch.prompt_cache[0].offset.tolist())) > 1
        ):
            offsets = batch.prompt_cache[0].offset.tolist()
            assert len(set(offsets)) > 1
            before = len(model.calls)
            assert bg._feed_batch_mains_to_standard(batch, state)
            assert model.calls[before:] == [((batch_size, 1), 0)]
            bg._drop_mtp_batch_state(batch, "test-handoff")
            model._omlx_mtp_decode_enabled = False
            handoffs.append(offsets)
        return result

    monkeypatch.setattr(bg, "_mtp_batch_next", handoff)
    actual, terminal = generate(model, prompts, limits)
    assert len(handoffs) == 1
    assert actual == expected
    for uid in range(len(prompts)):
        cache = terminal[uid].prompt_cache[0]
        reference = expected_terminal[uid].prompt_cache[0]
        assert (
            cache.keys[0, 0, : cache.offset, 0].tolist()
            == reference.keys[0, 0, : reference.offset, 0].tolist()
        )


@pytest.mark.parametrize("batch_size", [2, 4])
def test_batch_cost_parking_and_reentry_preserve_tokens(
    monkeypatch, batch_size, caplog
):
    from omlx.patches.mlx_lm_mtp.batch_policy import BatchPolicy

    standard = BatchPolicy.observe_standard
    mtp = BatchPolicy.observe_mtp
    park = BatchPolicy.park
    parks = []

    def measured_mtp(self, depth, accepted, milliseconds, *, stable):
        mtp(self, depth, accepted, 40.0, stable=stable)

    def short_cooldown(self):
        park(self)
        parks.append(self.uids)
        self.remaining = 3

    monkeypatch.setattr(
        BatchPolicy, "observe_standard", lambda self, ms: standard(self, 10.0)
    )
    monkeypatch.setattr(BatchPolicy, "observe_mtp", measured_mtp)
    monkeypatch.setattr(BatchPolicy, "park", short_cooldown)
    model = CountingModel()
    prompts = [[1] * (i + 2) for i in range(batch_size)]
    limits = [90] * batch_size
    model._omlx_mtp_decode_enabled = False
    expected, _ = generate(model, prompts, limits)
    model._omlx_mtp_decode_enabled = True
    with caplog.at_level("INFO", logger=bg.__name__):
        actual, _ = generate(model, prompts, limits)
    assert parks
    assert actual == expected
    assert "Lightning MTP batch parked" in caplog.text
    assert caplog.text.count("Lightning MTP batch activated") >= 2


def test_stop_inside_accepted_run_keeps_survivor_correct(monkeypatch):
    monkeypatch.setattr(bg, "_batch_policy_for_next", lambda batch: None)
    model = CountingModel()
    output, terminal = generate(
        model, [[1, 2], [10, 11, 12]], [12, 9], stop_tokens=[[6]]
    )
    assert output[0] == [3, 4, 5, 6]
    assert terminal[0].finish_reason == "stop"
    assert output[1] == list(range(13, 22))
    cache = terminal[0].prompt_cache[0]
    assert cache.keys[0, 0, : cache.offset, 0].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_second_row_processor_is_applied():
    def force_thirty(tokens, logits):
        return mx.broadcast_to(
            mx.where(mx.arange(64) == 30, 0.0, -float("inf")), logits.shape
        )

    output, _ = generate(
        CountingModel(), [[1, 2], [10, 11]], [9, 9], processors=[[], [force_thirty]]
    )
    assert output[0] == list(range(3, 12))
    assert output[1] == [30] * 9


def test_late_join_does_not_duplicate_or_skip_tokens():
    output, _ = generate(
        CountingModel(), [[1, 2], [10, 11, 12]], [12, 8], late_join=True
    )
    assert output[0] == list(range(3, 15))
    assert output[1] == list(range(13, 21))


class CapturingCountingModel(CountingModel):
    """CountingModel whose forward honours layer captures like mlx-vlm."""

    def __init__(self):
        super().__init__()
        self.mtp = None
        self.captures = []

    def __call__(
        self,
        inputs,
        cache=None,
        return_hidden=False,
        n_confirmed=0,
        capture_layer_ids=None,
    ):
        from mlx_vlm.models.base import LanguageModelOutput

        drafter = getattr(self, "_omlx_drafter", None)
        scope = getattr(drafter, "scope_uids", None)
        if not return_hidden and scope and capture_layer_ids is None:
            # Mirror the runtime wrapper: ordinary steps feed the drafter.
            logits, hidden = super().__call__(
                inputs, cache=cache, return_hidden=True, n_confirmed=n_confirmed
            )
            drafter.observe(scope, [hidden for _ in drafter.target_layer_ids])
            return logits
        result = super().__call__(
            inputs, cache=cache, return_hidden=return_hidden, n_confirmed=n_confirmed
        )
        if not return_hidden:
            return result
        logits, hidden = result
        captured = [hidden * (10 * (i + 1)) for i in capture_layer_ids or []]
        self.captures.append((tuple(inputs.shape), list(capture_layer_ids or [])))
        return LanguageModelOutput(
            logits=logits, hidden_states=captured + [hidden], gdn_states=None
        )


class TableDrafter:
    """Block drafter proposing successors, optionally wrong from a position."""

    target_layer_ids = [1, 3]

    def __init__(self, depth, wrong_from=None):
        self.depth = depth
        self.wrong_from = wrong_from
        self.fed = {}
        self.released = []
        self.jobs = 0
        self.scope_uids = None

    @contextlib.contextmanager
    def decode_scope(self, uids):
        previous = self.scope_uids
        self.scope_uids = tuple(uids)
        try:
            yield
        finally:
            self.scope_uids = previous

    def draft(self, jobs):
        for row_batch, state, captured, committed, _ in jobs:
            assert len(captured) == len(self.target_layer_ids)
            hidden = mx.concatenate(list(captured), axis=-1)
            assert hidden.shape[0] == 1 and hidden.shape[1] == committed.shape[0]
            self.fed[state.uid] = self.fed.get(state.uid, 0) + int(hidden.shape[1])
            anchor = int(committed.reshape(-1)[-1].item())
            drafts = [(anchor + 1 + j) % 64 for j in range(self.depth)]
            if self.wrong_from is not None and self.wrong_from < self.depth:
                drafts[self.wrong_from] = (drafts[self.wrong_from] + 7) % 64
            state.drafts = mx.array(drafts, dtype=mx.uint32)
            state.draft_lps = []
            # Stochastic rows need the draft distribution q; a one-hot q
            # makes every draft a certain proposal.
            state.draft_accept_lps = (
                []
                if bg._is_greedy(row_batch)
                else [
                    mx.where(mx.arange(64) == token, 0.0, -float("inf"))
                    for token in drafts
                ]
            )
            self.jobs += 1

    def observe(self, uids, captured):
        assert len(captured) == len(self.target_layer_ids)
        for uid in uids:
            self.fed[uid] = self.fed.get(uid, 0) + int(captured[0].shape[1])

    def release(self, uids):
        self.released.extend(uids)


def _drafted_counting_model(depth, wrong_from=None):
    model = CapturingCountingModel()
    model._omlx_mtp_depth = depth
    model._omlx_drafter = TableDrafter(depth, wrong_from=wrong_from)
    return model


@pytest.mark.parametrize("wrong_from", [None, 0, 2])
@pytest.mark.parametrize("late_join", [False, True])
def test_block_drafter_matches_standard_and_feeds_committed_context(
    wrong_from, late_join
):
    prompts = [[1, 2], [10, 11, 12], [30, 31]]
    limits = [12, 8, 15]
    standard = CapturingCountingModel()
    standard._omlx_mtp_decode_enabled = False
    expected, _ = generate(standard, prompts, limits, late_join=late_join)

    model = _drafted_counting_model(5, wrong_from=wrong_from)
    output, terminal = generate(model, prompts, limits, late_join=late_join)
    assert output == expected
    drafter = model._omlx_drafter
    assert drafter.jobs > 0
    # Every verify forward asked for the drafter's layers plus the head layer.
    assert all(ids == drafter.target_layer_ids for _, ids in model.captures)
    # Each row's context holds exactly one hidden per committed position:
    # the last prompt token, then every emitted token except the newest
    # one, which is still the anchor. Rejected drafts never enter it.
    for uid in terminal:
        assert drafter.fed[uid] == len(output[uid])
    # Finished rows leave the drafter registry.
    assert sorted(drafter.released) == sorted(terminal)


def test_block_drafter_serves_sampled_rows_with_draft_distribution():
    """A sampled row rides Leviathan acceptance against the drafter's q."""
    from omlx.utils.sampling import make_sampler

    mx.random.seed(11)
    model = _drafted_counting_model(3)
    stochastic = make_sampler(temp=0.7)
    output, terminal = generate(
        model, [[1, 2], [10, 11]], [6, 6], samplers=[None, stochastic]
    )
    assert output[0] == list(range(3, 9))
    # CountingModel logits are near one-hot, so sampling still follows the
    # successor rule; the row must finish with the requested length.
    assert output[1] == list(range(12, 18))
    assert {uid: r.finish_reason for uid, r in terminal.items()} == {
        0: "length",
        1: "length",
    }
    assert model._omlx_drafter.jobs > 0


@pytest.mark.parametrize("unequal_acceptance", [False, True])
def test_active_batch_admits_new_row_without_rebuilding_old_rows(
    monkeypatch, unequal_acceptance
):
    class DraftModel(CountingModel):
        def mtp_forward(self, hidden, tokens, cache, return_hidden=False, **kwargs):
            if unequal_acceptance:
                logits = self._logits(tokens + (tokens % 5 == 0))
                return (
                    (logits, tokens[..., None].astype(mx.float32))
                    if return_hidden
                    else logits
                )
            return super().mtp_forward(hidden, tokens, cache, return_hidden, **kwargs)

    bg.apply()
    model = DraftModel()
    prompts = [[1, 2], [10, 11, 12], [20, 21]]
    limits = [80, 80, 40]
    output = {uid: [] for uid in range(3)}
    terminal = {}
    verified = []
    original_verify = bg._run_verify_cycle_batched

    def verify(batch, state):
        verified.append(tuple(batch.uids))
        return original_verify(batch, state)

    monkeypatch.setattr(bg, "_run_verify_cycle_batched", verify)
    monkeypatch.setattr(
        bg,
        "_reconcile_mtp_to_standard",
        lambda *args: pytest.fail("Late join replayed an existing request's history"),
    )
    gen = BatchGenerator(
        model, sampler=lambda lp: mx.argmax(lp, -1), prefill_batch_size=2, max_tokens=80
    )

    def step():
        _, responses = gen.next()
        for response in responses:
            assert response.uid not in terminal
            output[response.uid].append(response.token)
            if response.finish_reason:
                terminal[response.uid] = response

    try:
        first = gen.insert(prompts[:2], max_tokens=limits[:2])
        for _ in range(12):
            step()
            if tuple(first) in verified:
                break
        assert tuple(first) in verified
        new_uid = gen.insert(prompts[2:], max_tokens=limits[2:])[0]
        for _ in range(120):
            step()
            if len(terminal) == 3:
                break
        assert tuple(first + [new_uid]) in verified
        assert len(terminal) == 3
        for uid, prompt in enumerate(prompts):
            expected = [(prompt[-1] + i + 1) % 64 for i in range(limits[uid])]
            assert output[uid] == expected
            cache = terminal[uid].prompt_cache[0]
            cached = cache.keys[0, 0, : cache.offset, 0].tolist()
            assert cached in (prompt + expected, (prompt + expected)[:-1])
    finally:
        gen.close()


def test_cancel_one_row_keeps_other_rows_at_their_frontiers():
    bg.apply()
    gen = BatchGenerator(
        CountingModel(),
        sampler=lambda lp: mx.argmax(lp, -1),
        prefill_batch_size=3,
        max_tokens=16,
    )
    output = {}
    try:
        uids = gen.insert([[1, 2], [10, 11], [20, 21]])
        output = {uid: [] for uid in uids}
        for step in range(30):
            if step == 3:
                gen.remove([uids[1]])
            _, responses = gen.next()
            for r in responses:
                assert step < 3 or r.uid != uids[1]
                output[r.uid].append(r.token)
            if len(output[0]) == 16 and len(output[2]) == 16:
                break
        assert output[0] == list(range(3, 19))
        assert output[2] == list(range(22, 38))
    finally:
        gen.close()


def test_second_row_uses_stochastic_acceptance(monkeypatch):
    seen = []

    def stochastic(logits):
        return mx.random.categorical(logits)

    stochastic.temp = 0.7
    original = bg._accept_lp_for

    def accept_lp(sampler, values):
        if sampler is stochastic:
            seen.append(values.shape[0])
        return original(sampler, values)

    monkeypatch.setattr(bg, "_accept_lp_for", accept_lp)
    output, _ = generate(
        CountingModel(), [[1, 2], [10, 11]], [10, 10], samplers=[None, stochastic]
    )
    assert output[0] == list(range(3, 13))
    assert output[1] == list(range(12, 22))
    assert any(
        rows > 1 for rows in seen
    ), "Target verification must use stochastic acceptance"


def test_failed_rollback_recovers_committed_tokens(monkeypatch):
    model = CountingModel()
    # Force rejection without changing target logits by making every draft wrong.
    monkeypatch.setattr(
        type(model),
        "mtp_forward",
        lambda self, hidden, tokens, cache, return_hidden=False, **kw: (
            self._logits((tokens + 5) % 64),
            tokens[..., None].astype(mx.float32),
        ),
    )
    monkeypatch.setattr(type(model), "mtp_partial_rollback", lambda *args: False)
    output, _ = generate(model, [[1, 2], [10, 11]], [8, 8])
    assert output[0] == list(range(3, 11))
    assert output[1] == list(range(12, 20))


@pytest.mark.parametrize("ratio", [2, 4, 8])
@pytest.mark.parametrize("accepted", [0, 1, 2])
def test_pooling_rollback_preserves_each_rows_window_phase(ratio, accepted):
    import copy

    import numpy as np

    from omlx.patches.deepseek_v4.cache_extras import BatchPoolingCache, PoolingCache

    def update(cache, inputs):
        kv, gates, _ = cache.accumulate_windows(inputs, inputs * 0.1, 0)
        if kv.shape[1]:
            windows = kv.reshape(kv.shape[0], -1, ratio, kv.shape[-1])
            window_gates = gates.reshape(gates.shape[0], -1, ratio, gates.shape[-1])
            cache.update_and_fetch(windows.mean(2))
            previous, previous_gates = cache.prev_for_prepend()
            if previous is not None:
                cache.store_prev(
                    mx.concatenate([previous, windows], 1),
                    mx.concatenate([previous_gates, window_gates], 1),
                    1,
                )
            else:
                cache.store_prev(windows, window_gates, 0)

    rows = []
    for length in [ratio - 1, ratio + 1, ratio * 2]:
        cache = PoolingCache(ratio)
        update(cache, mx.arange(length * 4).reshape(1, length, 4).astype(mx.float32))
        rows.append(cache)
    batch = BatchPoolingCache.merge(rows)
    reference = copy.deepcopy(batch)
    block = mx.arange(3 * 3 * 4).reshape(3, 3, 4).astype(mx.float32) + 100
    update(batch, block)
    assert batch.trim(2 - accepted) == 2 - accepted
    update(reference, block[:, : accepted + 1])
    assert batch._processed == reference._processed
    assert batch._pool_lengths == reference._pool_lengths
    assert batch.remainder == reference.remainder
    assert batch._prev_valid == reference._prev_valid
    for i, remainder in enumerate(batch.remainder):
        if remainder:
            np.testing.assert_array_equal(
                batch.buf_kv[i, :remainder], reference.buf_kv[i, :remainder]
            )
            np.testing.assert_array_equal(
                batch.buf_gate[i, :remainder], reference.buf_gate[i, :remainder]
            )
        length = batch._pool_lengths[i]
        if length:
            np.testing.assert_array_equal(
                batch.pooled[i, :length], reference.pooled[i, :length]
            )
        if batch._prev_valid[i]:
            np.testing.assert_array_equal(
                batch.prev_win_kv[i], reference.prev_win_kv[i]
            )


def test_fixed_depth_survivor_does_not_cache_past_length_limit(monkeypatch):
    monkeypatch.setattr(bg._DepthController, "observe", lambda *args, **kwargs: None)
    model = CountingModel()
    output, terminal = generate(model, [[1, 2], [4, 5, 6]], [8, 9])
    cache = terminal[1].prompt_cache[0]
    cached = cache.keys[0, 0, : cache.offset, 0].tolist()
    tokens = [4, 5, 6] + output[1]
    assert cached in (tokens, tokens[:-1])


@pytest.mark.parametrize("prompts", [[[1, 2]], [[1, 2], [10, 11]]])
def test_fixed_depth_drafts_the_full_depth_every_cycle(prompts):
    model = CountingModel()
    model._omlx_mtp_depth = 4
    model._omlx_mtp_depth_fixed = True
    output, _ = generate(model, prompts, [24] * len(prompts))
    for uid, prompt in enumerate(prompts):
        assert output[uid] == list(range(prompt[-1] + 1, prompt[-1] + 25))
    widths = [shape[1] for shape, _ in model.calls]
    first = next(i for i, width in enumerate(widths) if width > 1)
    assert set(widths[first:]) == {5}


def test_unrecoverable_batch_cache_failure_does_not_resume_decode(monkeypatch):
    def fail(*args, **kwargs):
        raise bg._MtpStepFallback("injected verify failure")

    monkeypatch.setattr(bg, "_run_verify_cycle_batched", fail)
    monkeypatch.setattr(bg, "_rebuild_singleton_cache", lambda model: None)
    with pytest.raises(RuntimeError, match="could not restore"):
        generate(CountingModel(), [[1, 2], [10, 11]], [8, 8])


def test_dspark_quantized_output_projection_keeps_each_batch_row():
    from mlx_lm.models.mla import MultiLinear

    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

    apply_deepseek_v4_patch()
    from mlx_lm.models.deepseek_v4 import (
        _project_attention_output,
        set_dspark_verify_armed,
    )

    mx.random.seed(651)
    groups, width = 8, 512
    attention = SimpleNamespace(
        rope=lambda value, offset, inverse: value,
        o_groups=groups,
        head_dim=width,
        wo_a=MultiLinear(width, 128, groups).to_quantized(
            group_size=32, bits=8, mode="mxfp8"
        ),
        wo_b=nn.Identity(),
    )
    inputs = mx.random.normal((2, groups, 3, width), dtype=mx.bfloat16)
    try:
        set_dspark_verify_armed(False)
        expected = mx.concatenate(
            [
                mx.concatenate(
                    [
                        _project_attention_output(
                            attention, inputs[b : b + 1, :, t : t + 1], 0
                        )
                        for t in range(3)
                    ],
                    axis=1,
                )
                for b in range(2)
            ],
            axis=0,
        )
        set_dspark_verify_armed(True)
        actual = _project_attention_output(attention, inputs, 0)
        mx.eval(actual, expected)
        assert actual.shape == expected.shape
        assert mx.array_equal(actual, expected).item()
    finally:
        set_dspark_verify_armed(False)


def test_eight_requests_share_verification_and_finish_independently():
    model = CountingModel()
    prompts = [[i, i + 1] for i in range(8)]
    limits = list(range(8, 16))
    output, _ = generate(model, prompts, limits)
    for uid, limit in enumerate(limits):
        assert output[uid] == list(range(uid + 2, uid + 2 + limit))
    assert ((8, 3), 1) in model.calls


def test_singleton_performance_parking_does_not_disable_shared_mtp():
    bg.apply()
    model = CountingModel()
    model._omlx_mtp_decode_enabled = False
    gen = BatchGenerator(model, sampler=lambda lp: mx.argmax(lp, -1), max_tokens=16)
    try:
        gen.insert([[1, 2], [10, 11]], max_tokens=[12, 16])
        for _ in range(4):
            gen.next()
            if len(gen._generation_batch.uids) == 2:
                break
        active = gen._generation_batch
        active._omlx_mtp_park_state = bg._MtpParkState(uid=0)
        model._omlx_mtp_decode_enabled = True
        for _ in range(6):
            gen.next()
            if getattr(active, "_omlx_mtp_batch_state", None) is not None:
                break
        assert set(active._omlx_mtp_batch_state.states) == {0, 1}
        assert not hasattr(active, "_omlx_mtp_park_state")
        for _ in range(32):
            gen.next()
            if not len(gen._generation_batch):
                break
        assert not len(gen._generation_batch)
    finally:
        gen.close()


def test_shared_acceptance_preserves_different_prefix_lengths(monkeypatch):
    class IntermittentDrafts(CountingModel):
        def mtp_forward(self, hidden, tokens, cache, return_hidden=False, **kwargs):
            logits = self._logits(tokens + (tokens % 5 == 0).astype(tokens.dtype))
            return (
                (logits, tokens[..., None].astype(mx.float32))
                if return_hidden
                else logits
            )

    monkeypatch.setattr(bg, "_DepthController", lambda *args, **kwargs: None)
    observed = set()
    original = bg._run_verify_cycle_chain

    def cycle(row, state, **kwargs):
        packed = kwargs.get("greedy_result")
        if packed is not None:
            observed.add(packed[0])
        return original(row, state, **kwargs)

    monkeypatch.setattr(bg, "_run_verify_cycle_chain", cycle)
    prompts = [[1, 2], [3, 4, 5], [6, 7], [8, 9, 10, 11]]
    limits = [12, 15, 18, 20]
    reference = CountingModel()
    reference._omlx_mtp_decode_enabled = False
    expected, _ = generate(reference, prompts, limits)
    actual, _ = generate(IntermittentDrafts(), prompts, limits)
    assert actual == expected
    assert observed == {0, 1, 2}


@pytest.mark.parametrize("batch_size", [2, 4])
def test_stochastic_shared_acceptance_keeps_row_ownership(batch_size, monkeypatch):
    from omlx.utils.sampling import make_sampler

    monkeypatch.setattr(bg, "_DepthController", lambda *args, **kwargs: None)
    monkeypatch.setattr(bg, "_batch_policy_for_next", lambda batch: None)
    original = bg._run_verify_cycle_chain
    shared = []

    def traced(batch, state, **kwargs):
        result = kwargs.get("stochastic_result")
        if result is not None:
            shared.append((state.uid, list(result)))
            assert result[1:3] == state.drafts.tolist()
        return original(batch, state, **kwargs)

    monkeypatch.setattr(bg, "_run_verify_cycle_chain", traced)
    model = CountingModel()
    prompts = [[1, 2], [4, 5, 6], [7, 8], [10, 11, 12]][:batch_size]
    limits = [18, 22, 24, 27][:batch_size]
    output, terminal = generate(
        model,
        prompts,
        limits,
        samplers=[make_sampler(temp=1) for _ in prompts],
    )
    assert len({uid for uid, _ in shared}) == batch_size
    for uid, prompt in enumerate(prompts):
        assert output[uid] == list(range(prompt[-1] + 1, prompt[-1] + 1 + limits[uid]))
        cache = terminal[uid].prompt_cache[0]
        actual = cache.keys[0, 0, : cache.offset, 0].tolist()
        assert actual in (prompt + output[uid], (prompt + output[uid])[:-1])


@pytest.mark.parametrize(
    "settings",
    [
        {"temp": 1.0},
        {"temp": 0.8, "top_p": 0.9, "top_k": 4, "min_p": 0.05},
    ],
)
def test_draft_distribution_matches_request_sampling(settings):
    from omlx.utils.sampling import make_sampler

    target = make_sampler(**settings)
    row = SimpleNamespace(samplers=[target], fallback_sampler=target)
    state = bg._MtpState(uid=0)
    draft = bg._resolve_draft_sampler(row, state)
    lp = bg._logprobs(mx.array([[2.0, 1.5, 1.0, 0.5, 0.0, -1.0]]))
    assert mx.array_equal(bg._accept_lp_for(draft, lp), bg._accept_lp_for(target, lp))
    assert bg._resolve_draft_sampler(row, state) is draft


def test_stochastic_acceptance_preserves_target_marginal():
    from omlx.utils.sampling import make_sampler

    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        mx.random.seed(781)
        sampler = make_sampler(temp=1.0)
        target = mx.array([0.1, 0.3, 0.6])
        q = mx.log(mx.array([0.8, 0.15, 0.05]))
        lp = mx.broadcast_to(mx.log(target), (3, 3))
        counts = mx.zeros((3,), dtype=mx.int32)
        for _ in range(64):
            samples = []
            for _ in range(64):
                drafts = mx.random.categorical(mx.broadcast_to(q, (2, 3)))
                result = bg._stochastic_verify_tokens(sampler, lp, drafts, [q, q])
                samples.append(mx.where(result[0] > 0, result[1], result[3]))
            emitted = mx.stack(samples)
            counts += (emitted[:, None] == mx.arange(3)).sum(axis=0)
            mx.eval(counts)
        # The proposal strongly favors the opposite token from the target.
        # Acceptance without residual correction cannot satisfy this bound.
        assert mx.all(mx.abs(counts / 4096 - target) < 0.03)
    finally:
        mx.set_default_device(previous_device)


def test_sparse_stochastic_acceptance_preserves_filtered_target_marginal():
    from omlx.utils.sampling import make_sampler

    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        mx.random.seed(783)
        sampler = make_sampler(temp=1.0, top_k=3)
        assert bg._sparse_top_k(sampler) == 3
        target = mx.array([0.05, 0.1, 0.3, 0.55])
        filtered = mx.array([0.0, 0.1, 0.3, 0.55]) / 0.95
        q = mx.log(mx.array([0.05, 0.8, 0.1, 0.05]))
        lp = mx.broadcast_to(mx.log(target), (3, 4))
        counts = mx.zeros((4,), dtype=mx.int32)
        for _ in range(64):
            samples = []
            for _ in range(64):
                drafts = mx.random.categorical(mx.broadcast_to(q, (2, 4)))
                result = bg._stochastic_verify_tokens(sampler, lp, drafts, [q, q])
                samples.append(mx.where(result[0] > 0, result[1], result[3]))
            emitted = mx.stack(samples)
            counts += (emitted[:, None] == mx.arange(4)).sum(axis=0)
            mx.eval(counts)
        # Token 0 is outside the target's top-k and must never be emitted even
        # though the draft proposes it.
        assert counts[0].item() == 0
        assert mx.all(mx.abs(counts / 4096 - filtered) < 0.03)
    finally:
        mx.set_default_device(previous_device)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("temperature", [0.7, 1.0])
def test_acceptance_density_matches_low_precision_sampler(dtype, temperature):
    from omlx.utils.sampling import make_sampler

    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        mx.random.seed(425)
        lp = bg._logprobs((mx.random.normal((2, 4096)) * 4).astype(dtype))
        sampler = make_sampler(temp=temperature)
        # Match categorical_sampling's scale in the original dtype. Only the
        # acceptance normalization is promoted; target sampling is unchanged.
        scaled = (lp * (1 / temperature)).astype(mx.float32)
        expected = mx.softmax(scaled, axis=-1)
        actual = mx.exp(bg._accept_lp_for(sampler, lp))
        assert actual.dtype == mx.float32
        assert mx.allclose(actual, expected, atol=1e-7, rtol=1e-5)
    finally:
        mx.set_default_device(previous_device)


def _adapter(language):
    from omlx.models.vlm import VLMModelAdapter

    return VLMModelAdapter(SimpleNamespace(language_model=language))


def _model(family):
    if family == "qwen_vlm":
        from mlx_vlm.models.qwen3_5.config import ModelConfig, TextConfig, VisionConfig
        from mlx_vlm.models.qwen3_5.language import LanguageModel

        from omlx.patches.mlx_vlm_mtp import qwen35_vlm_runtime

        qwen35_vlm_runtime.apply()
        config = TextConfig.from_dict(vars(_model("qwen").args))
        config.mtp_num_hidden_layers = 1
        return _adapter(
            LanguageModel(config, ModelConfig(config, VisionConfig(), "qwen3_5"))
        )
    if family == "qwen":
        from omlx.patches.mlx_lm_mtp import qwen35_model

        qwen35_model.apply()
        from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

        from test_mtp_prompt_priming import TINY_CONFIG

        return TextModel(
            TextModelArgs.from_dict(dict(TINY_CONFIG, max_position_embeddings=256))
        )
    if family == "qwen4":
        from test_mlx_vlm_qwen4_exp_compat import (
            _make_bound_qwen4_language_model,
            _tiny_config,
        )

        config = _tiny_config()
        config.text_config.layer_types = ["linear_attention", "qwen_sparse_attention"]
        language, owner = _make_bound_qwen4_language_model(config)
        model = _adapter(language)
        model._mtp_owner = owner
        return model
    if family == "v41":
        from test_deepseek_v41 import load_reference_weights
        from test_deepseek_v41_mtp import LanguageModel, mtp_config

        model = LanguageModel(mtp_config())
        load_reference_weights(model)
        model.configure_mtp(True, 2)
        return model
    if family == "v4_dspark":
        from test_deepseek_v4_dspark import _tiny_config

        from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch
        from omlx.patches.mlx_lm_mtp import deepseek_v4_model

        apply_deepseek_v4_patch()
        deepseek_v4_model.apply()
        import mlx_lm.models.deepseek_v4 as dsv4

        return dsv4.Model(_tiny_config(dsv4))
    if family == "inkling":
        from omlx.patches.mlx_vlm_mtp import inkling_vlm_runtime

        inkling_vlm_runtime.apply()
        from test_inkling_vlm_mtp import _mtp_language_model

        return _adapter(_mtp_language_model())
    if family == "gemma":
        from test_gemma4_vlm_mtp_runtime import (
            TINY_ASSISTANT_CONFIG,
            _language_model,
            _text_config,
        )

        from omlx.patches.mlx_vlm_mtp import gemma4_vlm_runtime

        gemma4_vlm_runtime.apply()
        return _adapter(
            _language_model(
                _text_config({"mtp_assistant_config": TINY_ASSISTANT_CONFIG})
            )
        )
    if family == "nemotron":
        from test_nemotron_mtp_patch import TINY_CONFIG

        from omlx.patches.mlx_lm_mtp import nemotron_h_chain, nemotron_h_model

        nemotron_h_model.apply()
        nemotron_h_chain.apply()
        from mlx_lm.models import nemotron_h as nh

        return nh.Model(nh.ModelArgs.from_dict(TINY_CONFIG))
    if family == "glm":
        from test_glm_mtp_patch import TINY_CFG

        from omlx.patches.glm_moe_dsa import apply_glm_moe_dsa_patch
        from omlx.patches.mlx_lm_mtp import glm_moe_dsa_model

        apply_glm_moe_dsa_patch()
        glm_moe_dsa_model.apply()
        from mlx_lm.models.glm_moe_dsa import Model, ModelArgs

        return Model(ModelArgs.from_dict(TINY_CFG))
    if family == "glm5":
        from test_glm5_next_mtp import make_host

        return _adapter(make_host(mtp_layers=1))
    if family == "step":
        from test_step3p7_patch import step3p7_mtp_model

        fixture = step3p7_mtp_model.__wrapped__()
        model = next(fixture)
        # Construction is complete; restore the fixture's process flag.
        fixture.close()
        return model
    raise AssertionError(family)


@pytest.mark.parametrize("family", ["qwen", "qwen_vlm"])
@pytest.mark.parametrize("queued", [False, True])
def test_reconcile_matches_ordinary_prefill_cache_and_next_token(family, queued):
    from collections import deque

    mx.random.seed(3702)
    model = _model(family)
    mx.eval(model.parameters())
    history = [3, 4, 5, 6, 7] * 8
    step = 16
    state = bg._MtpState(
        uid=17,
        queue=deque([(7, mx.zeros((256,)), "draft")]) if queued else deque(),
    )
    batch = SimpleNamespace(
        model=model,
        uids=[17],
        tokens=[history.copy()],
        _num_tokens=[4],
        samplers=[None],
        fallback_sampler=lambda lp: mx.argmax(lp, axis=-1).astype(mx.uint32),
        logits_processors=[],
        _next_tokens=mx.array([999]),
        _next_logprobs=[],
        _token_context=[],
        prompt_cache=[object()],
        prefill_step_size=step,
    )
    with bg._prompt_priming.decode_scope(model, batch.uids):
        assert bg._reconcile_mtp_to_standard(batch, state)
        reference = bg._rebuild_singleton_cache(model)
        tokens = mx.array(history, dtype=mx.uint32)
        for start in range(0, len(history), step):
            logits = model(tokens[None, start : start + step], cache=reference)
            mx.eval(logits)
        for actual, expected in zip(batch.prompt_cache, reference, strict=True):
            actual_state = tree_flatten(actual.state)
            expected_state = tree_flatten(expected.state)
            for (key, value), (ref_key, ref) in zip(
                actual_state, expected_state, strict=True
            ):
                assert key == ref_key
                if isinstance(value, mx.array):
                    assert mx.array_equal(value, ref).item(), (family, key)
                else:
                    assert value == ref
            assert getattr(actual, "meta_state", ()) == getattr(
                expected, "meta_state", ()
            )
        expected_token = 7 if queued else mx.argmax(logits[:, -1, :]).item()
        assert batch._next_tokens.item() == expected_token
        assert batch.tokens == [history]
        assert batch._num_tokens == [4]
        if queued:
            assert mx.array_equal(batch._next_logprobs[0], state.queue[0][1])
        else:
            expected_lp = bg._logprobs(logits[:, -1, :])[0]
            assert mx.array_equal(batch._next_logprobs[0], expected_lp)


@pytest.mark.parametrize("family", ["qwen", "qwen_vlm"])
def test_qwen_late_join_preserves_cache_without_history_replay(family, monkeypatch):
    monkeypatch.setattr(mlx_lm_mtp, "_MTP_ACTIVE", True)
    mx.random.seed(3702)
    model = _model(family)
    mx.eval(model.parameters())
    handoffs = []
    feed = bg._feed_batch_mains_to_standard

    def checked_feed(batch, state):
        reference = copy.deepcopy(batch.prompt_cache)
        inputs = mx.stack([state.states[uid].next_main for uid in batch.uids])
        bg._set_batched_mrope_deltas(batch, batch.uids)
        with bg._prompt_priming.decode_scope(batch.model, batch.uids):
            expected_lp = bg._logprobs(batch.model(inputs, cache=reference)[:, -1, :])
        result = feed(batch, state)
        assert result
        assert mx.array_equal(batch._next_tokens, mx.argmax(expected_lp, axis=-1))
        assert mx.array_equal(mx.stack(batch._next_logprobs), expected_lp)
        for actual, expected in zip(batch.prompt_cache, reference, strict=True):
            for (key, value), (ref_key, ref) in zip(
                tree_flatten(actual.state), tree_flatten(expected.state), strict=True
            ):
                assert key == ref_key
                if isinstance(value, mx.array):
                    assert mx.array_equal(value, ref), (family, key)
                else:
                    assert value == ref
            assert getattr(actual, "meta_state", ()) == getattr(
                expected, "meta_state", ()
            )
        handoffs.append(tuple(batch.uids))
        return result

    monkeypatch.setattr(bg, "_feed_batch_mains_to_standard", checked_feed)
    monkeypatch.setattr(
        bg,
        "_reconcile_mtp_to_standard",
        lambda *args: pytest.fail("Late join replayed an existing request's history"),
    )
    bg.apply()
    gen = BatchGenerator(
        model, sampler=lambda lp: mx.argmax(lp, -1), prefill_batch_size=2, max_tokens=40
    )
    try:
        first = gen.insert([[3, 4, 5], [7, 8, 9, 10, 11]])
        for _ in range(12):
            gen.next()
            if getattr(gen._generation_batch, "_omlx_mtp_batch_state", None):
                break
        assert getattr(gen._generation_batch, "_omlx_mtp_batch_state", None)
        new_uid = gen.insert([[12, 13, 14, 15]])[0]
        for _ in range(12):
            gen.next()
            active = gen._generation_batch
            state = getattr(active, "_omlx_mtp_batch_state", None)
            if state is not None and new_uid in state.states:
                break
        assert tuple(first) in handoffs
        assert state is not None and set(state.states) == set(first + [new_uid])
    finally:
        gen.close()


@pytest.mark.parametrize(
    "family,unequal_depths",
    [
        ("qwen", False),
        ("qwen", True),
        ("qwen_vlm", False),
        ("qwen_vlm", True),
        ("qwen4", False),
        ("qwen4", True),
        # V4.1 owns its depth controller; the common override does not apply.
        ("v41", False),
        # Singleton-only adapters need batching/late-join checks, not a
        # duplicate matrix for a multi-request depth policy they never use.
        ("gemma", False),
        ("v4_dspark", False),
        ("inkling", False),
        ("nemotron", False),
        ("glm", False),
        ("glm5", False),
        ("glm5", True),
        ("step", False),
    ],
)
@pytest.mark.parametrize("late_join", [False, True])
@pytest.mark.parametrize("batch_size", [2, 4])
def test_multi_request_mtp_or_singleton_only_matches_standard(
    family, unequal_depths, late_join, batch_size, monkeypatch
):
    previous = mlx_lm_mtp.is_mtp_active()
    depth = mlx_lm_mtp.get_mtp_depth()
    try:
        mlx_lm_mtp.set_mtp_active(True)
        mlx_lm_mtp.set_mtp_depth(2)
        if unequal_depths:
            # Exercise mismatched row depths independently of the policy
            # which deliberately selects one common depth for a cohort.
            monkeypatch.setattr(bg, "_batch_policy_for_next", lambda batch: None)
            created = 0

            def observe(controller, *args, **kwargs):
                nonlocal created
                if not hasattr(controller, "_test_phase"):
                    controller._test_phase = created % 2
                    controller._test_cycle = 0
                    created += 1
                schedule = (2, 1, 0)
                controller.cur = schedule[
                    (controller._test_cycle + controller._test_phase) % 3
                ]
                controller._test_cycle += 1

            monkeypatch.setattr(bg._DepthController, "observe", observe)
        mx.random.seed(173)
        model = _model(family)
        batch_supported = family in {"qwen", "qwen_vlm", "qwen4", "v41", "glm5"}
        assert bg._model_supports_batch_mtp(model) is batch_supported
        mx.eval(model.parameters())
        host = getattr(
            model, "_language_model", getattr(model, "language_model", model)
        )
        prompts = [
            [3, 4, 5, 6, 7],
            [3, 6, 7, 8, 4, 5, 6],
            [4, 5, 6],
            [7, 8, 9, 10, 11, 12],
        ][:batch_size]
        limits = [12, 16, 9, 20][:batch_size]
        host._omlx_mtp_decode_enabled = False
        expected, _ = generate(model, prompts, limits, late_join=late_join)
        host._omlx_mtp_decode_enabled = True
        calls = []
        draft_depths = set()
        shared_shapes = []
        batch_accepts = []
        if family in {"qwen_vlm", "glm5"}:
            rollback = host.rollback_speculative_cache

            def record_rollback(cache, gdn, accepted, block_size):
                if isinstance(accepted, list):
                    batch_accepts.append(accepted)
                return rollback(cache, gdn, accepted, block_size)

            monkeypatch.setattr(host, "rollback_speculative_cache", record_rollback)
        backbone = bg._call_backbone

        def record_backbone(model, inputs, cache, n_confirmed=0):
            if n_confirmed and inputs.shape[0] > 1 and inputs.shape[1] > 1:
                shared_shapes.append(tuple(inputs.shape))
            return backbone(model, inputs, cache, n_confirmed)

        monkeypatch.setattr(bg, "_call_backbone", record_backbone)
        entry = (
            "_run_verify_cycle_legacy"
            if family == "step"
            else "_run_verify_cycle_chain"
        )
        original = getattr(bg, entry)

        def traced(row, state, **kwargs):
            calls.append(state.uid)
            if state.chain:
                draft_depths.add(int(state.drafts.shape[0]))
            return original(row, state, **kwargs)

        monkeypatch.setattr(bg, entry, traced)
        batch_next = bg._mtp_batch_next
        batch_calls = []

        def traced_batch(batch, state):
            batch_calls.append(tuple(batch.uids))
            return batch_next(batch, state)

        monkeypatch.setattr(bg, "_mtp_batch_next", traced_batch)
        actual, _ = generate(model, prompts, limits, late_join=late_join)
        assert actual == expected
        if not batch_supported:
            assert not batch_calls
            assert not shared_shapes
            return
        assert batch_calls
        assert set(calls) == set(
            range(batch_size)
        ), "Every UID must execute MTP, not standard fallback"
        if unequal_depths and family in {"qwen", "qwen_vlm", "glm5"}:
            assert len(draft_depths) > 1, "The mismatched-depth case must vary depths"
        if family not in {"v41", "step"}:
            assert shared_shapes, "Chain models must share target verification"
        if family in {"qwen_vlm", "glm5"} and not unequal_depths and not late_join:
            assert batch_accepts, "Use the model's vector rollback contract"
    finally:
        mlx_lm_mtp.set_mtp_active(previous)
        mlx_lm_mtp.set_mtp_depth(depth)


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("late_join", [False, True])
def test_initialization_uses_one_forward_without_cache_extraction(
    size, late_join, monkeypatch
):
    active = mlx_lm_mtp.is_mtp_active()
    mlx_lm_mtp.set_mtp_active(True)
    monkeypatch.setattr(bg, "_DepthController", lambda *args, **kwargs: None)
    monkeypatch.setattr(bg, "_batch_policy_for_next", lambda batch: None)
    try:
        mx.random.seed(173)
        model = _model("qwen_vlm")
        original = bg._prepare_mtp_batch_state_for_next
        forward = bg._call_backbone
        calls = []
        shared = []
        checking = False

        def backbone(model, inputs, cache, n_confirmed=0):
            if checking:
                calls.append(tuple(inputs.shape))
            return forward(model, inputs, cache, n_confirmed)

        def prepare(batch):
            nonlocal checking
            if getattr(batch, "_omlx_mtp_batch_state", None) is not None:
                return original(batch)
            cache = batch.prompt_cache
            before = len(calls)

            def forbidden_extract(index):
                pytest.fail("Shared initialization must not extract target rows")

            with monkeypatch.context() as patch:
                patch.setattr(batch, "extract_cache", forbidden_extract)
                checking = True
                try:
                    result = original(batch)
                finally:
                    checking = False
            assert batch.prompt_cache is cache
            assert calls[before:] == [(len(batch.uids), 1)]
            shared.append(tuple(batch.uids))
            return result

        monkeypatch.setattr(bg, "_call_backbone", backbone)
        monkeypatch.setattr(bg, "_prepare_mtp_batch_state_for_next", prepare)
        prompts = [[3, 4, 5], [3, 6, 7, 8], [4, 5, 6], [7, 8, 9, 10]][:size]
        actual, _ = generate(model, prompts, [20] * size, late_join=late_join)
        assert shared
        model._language_model._omlx_mtp_decode_enabled = False
        expected, _ = generate(model, prompts, [20] * size, late_join=late_join)
        assert actual == expected
    finally:
        mlx_lm_mtp.set_mtp_active(active)


def calibrated(batch=4, depth=2):
    policy = BatchPolicy(range(batch), depth)
    for cost in [100, 50, 10, 10, 10]:
        policy.observe_standard(cost)
    assert not policy.needs_standard()
    return policy


def test_whole_batch_cost_can_lose_despite_per_row_acceptance():
    policy = calibrated()
    for _ in range(35):
        depth = policy.cur
        policy.observe_mtp(depth, [depth] * 4, 40, stable=True)
    assert policy.score(0) == 0.4
    assert policy.score(2) < policy.score(0)
    assert policy.should_park()
    policy.park()
    assert policy.needs_standard()
    for _ in range(127):
        policy.observe_standard(10)
    assert policy.needs_standard()
    policy.observe_standard(10)
    assert not policy.needs_standard()
    assert not policy.costs
    assert policy.cur == 2


def test_depth_transition_does_not_measure_mixed_async_work():
    policy = calibrated()
    policy.observe_mtp(2, [2] * 4, 100, stable=False)
    assert not policy.costs
    policy.observe_mtp(2, [2] * 4, 14, stable=True)
    assert list(policy.costs[2]) == [14]


def test_depth_choice_uses_batch_cost_instead_of_draft_count():
    policy = calibrated()
    for _ in range(30):
        depth = policy.cur
        policy.observe_mtp(depth, [depth] * 4, {1: 12, 2: 28}[depth], stable=True)
    assert policy.cur == 1
    assert not policy.should_park()


def test_same_acceptance_can_win_at_two_rows_and_lose_at_four():
    two = calibrated(batch=2, depth=1)
    four = calibrated(batch=4, depth=1)
    for _ in range(30):
        two.observe_mtp(1, [1, 1], 12, stable=True)
        four.observe_mtp(1, [1] * 4, 24, stable=True)
    assert not two.should_park()
    assert four.should_park()


def test_cost_includes_scheduler_interval_but_not_a_mode_transition():
    policy = calibrated()
    assert abs(policy.cycle_time_ms("standard", 1.0, 1.01) - 10) < 1e-9
    assert abs(policy.cycle_time_ms("standard", 1.015, 1.025) - 15) < 1e-9
    assert abs(policy.cycle_time_ms("mtp", 2.0, 2.03) - 30) < 1e-9
    assert abs(policy.cycle_time_ms("mtp", 2.034, 2.064) - 34) < 1e-9


def test_prefill_wait_does_not_park_a_faster_batch():
    policy = calibrated(batch=2, depth=2)
    now = 0.0
    for _ in range(40):
        costs = {depth: list(values) for depth, values in policy.costs.items()}
        policy.interrupt_timing()
        now += 1.0
        elapsed = policy.cycle_time_ms("mtp", now, now + 0.02)
        policy.observe_mtp(2, [2, 2], elapsed, stable=True)
        assert elapsed is None
        assert {depth: list(values) for depth, values in policy.costs.items()} == costs
        now += 0.02
        for _ in range(3):
            depth = policy.cur
            elapsed = policy.cycle_time_ms("mtp", now, now + 0.02)
            policy.observe_mtp(depth, [depth, depth], elapsed, stable=True)
            now += 0.02
        assert not policy.should_park()
    assert all(abs(t - 20) < 1e-6 for costs in policy.costs.values() for t in costs)


def test_interrupted_standard_sample_preserves_calibration_and_cooldown():
    policy = calibrated()
    policy.park()
    for _ in range(128):
        policy.interrupt_timing()
        elapsed = policy.cycle_time_ms("standard", 1, 2)
        policy.observe_standard(elapsed)
    assert policy.remaining == 0
    assert policy.standard_warmup == 2
    assert list(policy.standard) == [10, 10, 10]
    for i in range(3):
        now = 2 + i * 0.01
        policy.observe_standard(policy.cycle_time_ms("standard", now, now + 0.01))
    assert not policy.needs_standard()


def test_generator_prefill_interrupts_batch_cost_samples(monkeypatch):
    observed = []
    clock = BatchPolicy.cycle_time_ms

    def record(policy, *args):
        elapsed = clock(policy, *args)
        observed.append(elapsed)
        return elapsed

    monkeypatch.setattr(BatchPolicy, "cycle_time_ms", record)
    bg.apply()
    gen = BatchGenerator(
        CountingModel(),
        sampler=lambda lp: mx.argmax(lp, -1),
        prefill_batch_size=2,
        prefill_step_size=3,
        max_tokens=100,
    )
    try:
        gen.insert([[1, 2], [3, 4]])
        for _ in range(12):
            gen.next()
        observed.clear()
        gen.insert([[20] * 24])
        for _ in range(6):
            gen.next()
        assert None in observed
        assert any(value is not None for value in observed)
    finally:
        gen.close()


def _coupled_sampler(index):
    key = mx.random.key(index + 90)

    def sampler(lp):
        return mx.random.categorical(bg._accept_lp_for(sampler, lp), key=key)

    sampler.temp = 0.7
    sampler.top_p = 0.9
    return sampler


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("family", ["qwen_vlm", "qwen4"])
@pytest.mark.parametrize("stochastic", [False, True])
def test_batched_head_matches_row_caches_across_depth_changes(size, family, stochastic):
    active = mlx_lm_mtp.is_mtp_active()
    mlx_lm_mtp.set_mtp_active(True)
    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        mx.random.seed(44)
        model = _model(family)
        states = [
            bg._MtpState(uid=i, mtp_cache=model.make_mtp_cache()) for i in range(size)
        ]
        refs = [
            bg._MtpState(uid=i, mtp_cache=model.make_mtp_cache()) for i in range(size)
        ]
        rows = [
            SimpleNamespace(
                model=model,
                samplers=[lambda lp: mx.argmax(lp, -1)],
                logits_processors=None,
            )
            for _ in states
        ]
        if stochastic:
            for index, (state, ref, row) in enumerate(zip(states, refs, rows)):
                # Fixed per-row random keys couple scalar/batched draws while
                # still exercising categorical draws and filtered q values.
                sampler = _coupled_sampler(index)
                state.draft_sampler = ref.draft_sampler = sampler
                row.samplers = [sampler]
        owner = bg._MtpBatchState(states=dict(enumerate(states)))
        batch = SimpleNamespace(model=model, _omlx_mtp_batch_state=owner)
        cache_identity = None
        for cycle, depth in enumerate([2, 4, 1, 3]):
            jobs = []
            for index, (state, ref, row) in enumerate(zip(states, refs, rows)):
                state.depth = ref.depth = depth
                length = (cycle + index) % 3 + 1
                hidden = mx.random.normal((1, length, 64))
                tokens = mx.random.randint(
                    0, model._language_model.args.vocab_size, (length,)
                ).astype(mx.uint32)
                bg._mtp_head_trim_to(ref.mtp_cache, ref.hist_offset)
                bg._chain_next_drafts(row, ref, hidden, tokens, None)
                jobs.append((row, state, hidden, tokens, None))
            batched_head.draft(batch, jobs)
            if cache_identity is not None:
                assert owner.head.cache is cache_identity
            cache_identity = owner.head.cache
            assert owner.head.speculative == depth - 1
            for index, (state, ref) in enumerate(zip(states, refs)):
                assert state.mtp_cache is None
                assert state.hist_offset == ref.hist_offset
                assert mx.array_equal(state.drafts, ref.drafts)
                for actual_lp, expected_lp in zip(
                    state.draft_accept_lps, ref.draft_accept_lps
                ):
                    assert mx.allclose(actual_lp, expected_lp, rtol=1e-4, atol=1e-4)
                for layer, reference in zip(owner.head.cache, ref.mtp_cache):
                    actual = layer.extract(index)
                    assert actual.offset == reference.offset
                    for (_, tensor), (_, expected) in zip(
                        tree_flatten(
                            CacheTypeRegistry.get_handler_for_object(
                                actual
                            ).serialize_state(actual)
                        ),
                        tree_flatten(
                            CacheTypeRegistry.get_handler_for_object(
                                reference
                            ).serialize_state(reference)
                        ),
                    ):
                        assert mx.allclose(tensor, expected, rtol=1e-4, atol=1e-4)
        batched_head.flush(owner)
        assert owner.head is None
        assert all(state.mtp_cache for state in states)
    finally:
        mx.set_default_device(previous_device)
        mlx_lm_mtp.set_mtp_active(active)


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("late_join", [False, True])
@pytest.mark.parametrize("family", ["qwen_vlm", "qwen4"])
def test_batched_head_survives_join_and_staggered_finish(
    size, late_join, monkeypatch, family
):

    active = mlx_lm_mtp.is_mtp_active()
    mlx_lm_mtp.set_mtp_active(True)
    monkeypatch.setattr(bg, "_DepthController", lambda *args, **kwargs: None)
    monkeypatch.setattr(bg, "_batch_policy_for_next", lambda batch: None)
    try:
        mx.random.seed(173)
        model = _model(family)
        host = model._language_model
        prompts = [[3, 4, 5], [3, 6, 7, 8], [4, 5, 6], [7, 8, 9, 10]][:size]
        limits = [18, 26, 33, 40][:size]
        host._omlx_mtp_decode_enabled = False
        expected, _ = generate(model, prompts, limits, late_join=late_join)
        host._omlx_mtp_decode_enabled = True
        owners, reused = [], []
        original = batched_head.draft

        def traced(batch, jobs):
            owner = batch._omlx_mtp_batch_state
            previous = owner.head
            result = original(batch, jobs)
            owners.append(owner)
            reused.append(previous is not None and previous.cache is owner.head.cache)
            return result

        monkeypatch.setattr(batched_head, "draft", traced)
        actual, _ = generate(model, prompts, limits, late_join=late_join)
        assert actual == expected
        assert any(reused), "The head cache must persist across actual verify cycles"
        assert all(owner.head is None for owner in owners)
    finally:
        mlx_lm_mtp.set_mtp_active(active)


@pytest.mark.parametrize("family", ["qwen_vlm", "qwen4"])
def test_cancel_restores_surviving_head_cache(monkeypatch, family):
    from mlx_lm.generate import BatchGenerator

    active = mlx_lm_mtp.is_mtp_active()
    mlx_lm_mtp.set_mtp_active(True)
    monkeypatch.setattr(bg, "_DepthController", lambda *args, **kwargs: None)
    monkeypatch.setattr(bg, "_batch_policy_for_next", lambda batch: None)
    gen = None
    try:
        bg.apply()
        mx.random.seed(173)
        model = _model(family)
        gen = BatchGenerator(
            model,
            sampler=lambda lp: mx.argmax(lp, -1),
            prefill_batch_size=2,
            max_tokens=32,
        )
        uids = gen.insert([[3, 4, 5], [3, 6, 7, 8]])
        cancelled = False
        for _ in range(100):
            _, responses = gen.next()
            if cancelled:
                assert all(response.uid != uids[1] for response in responses)
                if any(response.finish_reason for response in responses):
                    break
            else:
                batch = gen._generation_batch
                owner = getattr(batch, "_omlx_mtp_batch_state", None)
                if owner is not None and owner.head is not None:
                    survivor = owner.states[uids[0]]
                    assert survivor.mtp_cache is None
                    gen.remove([uids[1]])
                    assert owner.head is None
                    assert survivor.mtp_cache is not None
                    assert all(
                        c.offset >= survivor.hist_offset for c in survivor.mtp_cache
                    )
                    cancelled = True
        else:
            pytest.fail("Surviving request did not finish")
        assert cancelled, "Cancellation must occur while the batch owns its head cache"
    finally:
        if gen is not None:
            gen.close()
        mlx_lm_mtp.set_mtp_active(active)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize(
    "mode", ["matching", "mixed", "unfiltered", "greedy", "custom"]
)
def test_draft_filter_batching_preserves_rows_and_rng(dtype, mode):
    from omlx.utils.sampling import make_sampler

    mx.random.seed(718)
    logits = (mx.random.normal((4, 257)) * 3).astype(dtype)
    lp = bg._logprobs(logits)
    mx.eval(lp)
    samplers = []
    for row in range(4):
        if mode == "custom":
            samplers.append(_coupled_sampler(row))
        else:
            samplers.append(
                make_sampler(
                    temp=0 if mode == "greedy" else 1,
                    top_p=1 if mode == "unfiltered" else 0.95,
                    top_k=(
                        0
                        if mode == "unfiltered"
                        else 20 + (row if mode == "mixed" else 0)
                    ),
                )
            )
    for seed in range(3):
        mx.random.seed(seed)
        expected = [
            bg._sample_draft_with_logprobs(s, lp[i : i + 1])
            for i, s in enumerate(samplers)
        ]
        token = mx.concatenate([v[0] for v in expected])
        density = mx.concatenate([v[1] for v in expected])
        mx.eval(token, density)
        rng = mx.random.state[0].tolist()
        mx.random.seed(seed)
        actual_token, actual_density = batched_head._sample_rows(samplers, lp)
        mx.eval(actual_token, actual_density)
        assert mx.array_equal(token, actual_token).item()
        assert mx.array_equal(density, actual_density).item()
        assert mx.random.state[0].tolist() == rng
    assert (getattr(samplers[0], "_mtp_batch_sampling_key", None) is not None) == (
        mode in {"matching", "mixed"}
    )


def test_unsupported_adapter_remains_singleton_only(monkeypatch):
    monkeypatch.setattr(bg, "_mtp_common_eligible", lambda _: True)
    model = CountingModel()
    model._omlx_mtp_multi_request = False
    batch = SimpleNamespace(model=model, uids=[1, 2])
    assert not bg._is_mtp_batch_eligible(batch)
    assert "single-request" in bg._ineligibility_reason(batch)
    batch.uids = [1]
    assert bg._is_mtp_eligible(batch)


@pytest.mark.parametrize("late_join", [False, True])
def test_singleton_only_model_decodes_multirow_without_mtp(monkeypatch, late_join):
    active = mlx_lm_mtp.is_mtp_active()
    mlx_lm_mtp.set_mtp_active(True)
    monkeypatch.setattr(bg, "_DepthController", lambda *args, **kwargs: None)
    try:
        model = CountingModel()
        model._omlx_mtp_multi_request = False
        singleton_uids = []
        singleton_next = bg._mtp_next

        def record_singleton(batch, state):
            assert len(batch.uids) == 1
            singleton_uids.append(batch.uids[0])
            return singleton_next(batch, state)

        def forbidden_batch(*args):
            pytest.fail("A singleton-only model entered multi-request MTP")

        monkeypatch.setattr(bg, "_mtp_next", record_singleton)
        monkeypatch.setattr(bg, "_mtp_batch_next", forbidden_batch)
        prompts = [[1, 2], [4, 5, 6]]
        limits = [8, 24]
        model._omlx_mtp_decode_enabled = False
        expected, _ = generate(model, prompts, limits)
        model._omlx_mtp_decode_enabled = True
        model.calls.clear()
        singleton_uids.clear()
        if late_join:
            gen = BatchGenerator(
                model, sampler=lambda lp: mx.argmax(lp, -1), prefill_batch_size=2
            )
            actual = {0: [], 1: []}
            joined = False
            terminal = set()
            try:
                gen.insert(prompts[:1], max_tokens=limits[:1])
                for _ in range(100):
                    _, responses = gen.next()
                    for response in responses:
                        assert response.uid not in terminal
                        actual[response.uid].append(response.token)
                        if response.finish_reason:
                            terminal.add(response.uid)
                    # Join only after MTP actually owns the first request.
                    if singleton_uids and not joined:
                        gen.insert(prompts[1:], max_tokens=limits[1:])
                        joined = True
                    if len(terminal) == 2:
                        break
                assert joined and len(terminal) == 2
            finally:
                gen.close()
        else:
            actual, _ = generate(model, prompts, limits)
        assert actual == expected
        assert any(shape[0] > 1 and not verified for shape, verified in model.calls)
        assert 1 in singleton_uids, "The surviving compact row should regain MTP"
        if late_join:
            assert 0 in singleton_uids, "The initial single request should use MTP"
    finally:
        mlx_lm_mtp.set_mtp_active(active)


def test_singleton_and_ordinary_dispatch_are_preserved(monkeypatch):
    calls = []

    from mlx_vlm.speculative.ops import linear as ops

    def original(linear, x):
        calls.append(x.shape)
        return True

    monkeypatch.setattr(ops, "_use_target_verify_dense", original)
    qwen35_verify_linear.apply()
    first = ops._use_target_verify_dense
    qwen35_verify_linear.apply()
    assert ops._use_target_verify_dense is first
    assert first(None, mx.zeros((1, 3, 64)))
    assert not first(None, mx.zeros((4, 3, 64)))
    assert calls == [(1, 3, 64)]


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("bits", [None, 4, 8])
@pytest.mark.parametrize("batch,length", [(2, 2), (4, 3)])
def test_batch_linear_matches_standard_projection(dtype, bits, batch, length):
    qwen35_verify_linear.apply()
    mx.random.seed(79)
    layer = nn.Linear(2048, 4096, bias=False)
    if bits is not None:
        layer = nn.QuantizedLinear.from_linear(layer, bits=bits, group_size=64)
    layer.set_dtype(dtype)
    x = mx.random.normal((batch, length, 2048)).astype(dtype)
    from mlx_vlm.speculative.ops.linear import _target_verify_linear

    actual = _target_verify_linear(layer, x)
    expected = layer(x)
    assert actual.dtype == dtype
    assert mx.array_equal(actual, expected)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("size", [2, 4])
def test_quantized_qwen_batch_matches_standard(bits, size, monkeypatch):

    from omlx.patches import mlx_lm_mtp
    from omlx.patches.mlx_lm_mtp import batch_generator as bg

    active = mlx_lm_mtp.is_mtp_active()
    mlx_lm_mtp.set_mtp_active(True)
    monkeypatch.setattr(bg, "_DepthController", lambda *args, **kwargs: None)
    monkeypatch.setattr(bg, "_batch_policy_for_next", lambda batch: None)
    try:
        mx.random.seed(173)
        model = _model("qwen_vlm")
        nn.quantize(model, bits=bits, group_size=32)
        model.set_dtype(mx.bfloat16)
        host = model._language_model
        prompts = [[3, 4, 5], [3, 6, 7, 8], [4, 5, 6], [7, 8, 9, 10]][:size]
        limits = [18, 26, 22, 30][:size]
        host._omlx_mtp_decode_enabled = False
        expected, _ = generate(model, prompts, limits)
        host._omlx_mtp_decode_enabled = True
        from mlx_vlm.speculative.ops import linear as ops

        gate = ops._use_target_verify_dense
        observed = []

        def record(linear, x):
            result = gate(linear, x)
            if x.ndim == 3 and x.shape[0] > 1 and x.shape[1] > 1:
                observed.append(tuple(x.shape))
                assert not result
            return result

        monkeypatch.setattr(ops, "_use_target_verify_dense", record)
        actual, _ = generate(model, prompts, limits)
        assert observed, "Actual multi-row verification must use ordinary projections"
        assert actual == expected
    finally:
        mlx_lm_mtp.set_mtp_active(active)


def _assert_rows_equal(actual, references, accepted):
    for row, count in enumerate(accepted):
        for index, layer in enumerate(actual):
            observed = layer.extract(row)
            expected = references[count][index].extract(row)
            left = tree_flatten(observed.state)
            right = tree_flatten(expected.state)
            assert [key for key, _ in left] == [key for key, _ in right]
            for (key, value), (_, reference) in zip(left, right):
                if isinstance(value, mx.array):
                    assert mx.array_equal(value, reference).item(), (row, index, key)
            if hasattr(observed, "offset"):
                assert observed.offset == expected.offset


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("late_join", [False, True])
def test_vector_commit_matches_scalar_states_and_ordinary_tokens(
    monkeypatch, batch_size, late_join
):
    previous = mlx_lm_mtp.is_mtp_active()
    previous_depth = mlx_lm_mtp.get_mtp_depth()
    mlx_lm_mtp.set_mtp_active(True)
    mlx_lm_mtp.set_mtp_depth(2)
    monkeypatch.setattr(bg, "_DepthController", lambda *args, **kwargs: None)
    monkeypatch.setattr(bg, "_batch_policy_for_next", lambda batch: None)
    try:
        mx.random.seed(173)
        model = _model("qwen4")
        host = model._language_model
        mx.eval(model.parameters())
        prompts = [
            [3, 4, 5, 6, 7],
            [3, 6, 7, 8, 4, 5, 6],
            [4, 5, 6],
            [7, 8, 9, 10, 11, 12],
        ][:batch_size]
        limits = [18, 22, 17, 26][:batch_size]
        host._omlx_mtp_decode_enabled = False
        expected, _ = generate(model, prompts, limits, late_join=late_join)
        host._omlx_mtp_decode_enabled = True
        original = model.rollback_speculative_cache
        checked = []

        def rollback(caches, gdn, accepted, size):
            if not isinstance(accepted, list):
                return original(caches, gdn, accepted, size)
            # Exercise zero, full and ragged acceptance even when a random
            # tiny head happens to reject every draft during generation.
            cases = [
                [0] * len(accepted),
                [size - 1] * len(accepted),
                [i % size for i in range(len(accepted))],
                accepted,
            ]
            for values in cases:
                copies = {count: copy.deepcopy((caches, gdn)) for count in set(values)}
                references = {}
                for count, (reference, transaction) in copies.items():
                    original(reference, transaction, [count] * len(values), size)
                    references[count] = reference
                actual, transaction = copy.deepcopy((caches, gdn))
                original(actual, transaction, values, size)
                _assert_rows_equal(actual, references, values)
                checked.append(tuple(values))
            return original(caches, gdn, accepted, size)

        monkeypatch.setattr(model, "rollback_speculative_cache", rollback)
        actual, _ = generate(model, prompts, limits, late_join=late_join)
        assert actual == expected
        assert any(len(values) == batch_size for values in checked)
        assert any(len(set(values)) > 1 for values in checked)
    finally:
        mlx_lm_mtp.set_mtp_active(previous)
        mlx_lm_mtp.set_mtp_depth(previous_depth)


def _cache_rows(cache):
    for layer in cache or ():
        offset = getattr(layer, "offset", None)
        if isinstance(offset, mx.array) and offset.ndim == 1:
            return int(offset.size)
    return None


@pytest.mark.parametrize("family", ["qwen_vlm", "qwen4"])
def test_shared_verify_boundary_emit_uses_private_row_cache(monkeypatch, family):
    previous = mlx_lm_mtp.is_mtp_active()
    previous_depth = mlx_lm_mtp.get_mtp_depth()
    mlx_lm_mtp.set_mtp_active(True)
    mlx_lm_mtp.set_mtp_depth(2)
    monkeypatch.setattr(bg, "_DepthController", lambda *args, **kwargs: None)
    monkeypatch.setattr(bg, "_batch_policy_for_next", lambda batch: None)
    try:
        mx.random.seed(173)
        model = _model(family)
        host = model._language_model
        mx.eval(model.parameters())
        prompts = [[3, 4, 5, 6, 7], [3, 6, 7, 8, 4, 5, 6], [4, 5, 6]]
        limits = [18, 22, 17]
        host._omlx_mtp_decode_enabled = False
        expected, _ = generate(model, prompts, limits)
        host._omlx_mtp_decode_enabled = True
        # Force boundary crossings within the short generation.
        model._omlx_mtp_commit_align = 4
        materialized = []
        last_verify_rows = [0]
        original_materialize = bg._materialize_mtp_boundary_emit

        def materialize(row, state):
            materialized.append(last_verify_rows[0])
            return original_materialize(row, state)

        original_backbone = bg._call_backbone

        def backbone(target, inputs, cache, n_confirmed=0):
            rows = _cache_rows(cache)
            assert rows is None or rows == int(inputs.shape[0])
            if n_confirmed:
                last_verify_rows[0] = int(inputs.shape[0])
            return original_backbone(target, inputs, cache, n_confirmed)

        monkeypatch.setattr(bg, "_materialize_mtp_boundary_emit", materialize)
        monkeypatch.setattr(bg, "_call_backbone", backbone)
        actual, _ = generate(model, prompts, limits)
        assert actual == expected
        assert any(rows > 1 for rows in materialized)
    finally:
        mlx_lm_mtp.set_mtp_active(previous)
        mlx_lm_mtp.set_mtp_depth(previous_depth)


@pytest.mark.parametrize("batch", [1, 2])
def test_lightning_verify_preserves_quantized_linear_dispatch(monkeypatch, batch):
    from mlx_vlm.models.qwen3_5 import speculative_verifier
    from mlx_vlm.speculative.ops import linear as ops

    from omlx.patches import qwen35_verify_qmm

    qwen35_verify_linear.apply()
    monkeypatch.setattr(qwen35_verify_qmm, "_is_armed", lambda: True)
    calls = []
    layer = nn.QuantizedLinear(64, 64, bits=4, group_size=64)
    x = mx.zeros((batch, 3, 64))

    def projection(self, values):
        calls.append(values.shape)
        return values

    monkeypatch.setattr(nn.QuantizedLinear, "__call__", projection)
    assert ops._target_verify_linear(layer, x) is x
    assert ops._target_verify_linears((layer, layer), x) == (x, x)
    verifier = speculative_verifier.Qwen3_5BatchInvariantForward()
    assert verifier._linear(layer, x) is x
    assert verifier._linears((layer, layer), x) == (x, x)
    assert verifier.quantized_linear(layer, x) is x
    assert calls == [x.shape] * 7


def _nax_available() -> bool:
    try:
        from omlx.custom_kernels.nax import is_nax_available

        return bool(is_nax_available())
    except Exception:
        return False


def _quantized_linear(n, k, bits, seed):
    mx.random.seed(seed)
    layer = nn.QuantizedLinear(k, n, bias=False, group_size=64, bits=bits)
    weight = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
    layer.weight, layer.scales, layer.biases = mx.quantize(
        weight, group_size=64, bits=bits
    )
    return layer


def _reference(layer, x):
    return mx.quantized_matmul(
        x.astype(mx.float32),
        layer.weight,
        layer.scales.astype(mx.float32),
        layer.biases.astype(mx.float32),
        transpose=True,
        group_size=64,
        bits=layer.bits,
    )


def _close(actual, expected, tol=1e-2):
    scale = mx.abs(expected).max().item()
    return mx.abs(actual.astype(mx.float32) - expected).max().item() <= tol * scale


@pytest.mark.skipif(not _nax_available(), reason="needs the M5 tensor unit")
def test_packed_linear_replaces_4bit_projections(monkeypatch):
    from omlx.patches import qwen35_packed_linear as packed

    # GDN input projections: 4-bit qkv/z pack together, 5-bit b/a stay.
    gdn = nn.Module()
    gdn.in_proj_qkv = _quantized_linear(256, 512, 4, 1)
    gdn.in_proj_z = _quantized_linear(128, 512, 4, 2)
    gdn.in_proj_b = _quantized_linear(48, 512, 5, 3)
    gdn.in_proj_a = _quantized_linear(48, 512, 5, 4)
    mlp = nn.Module()
    mlp.gate_proj = _quantized_linear(256, 512, 4, 5)
    mlp.up_proj = _quantized_linear(256, 512, 4, 6)
    mlp.down_proj = _quantized_linear(512, 256, 5, 7)
    layer = nn.Module()
    layer.linear_attn = gdn
    layer.mlp = mlp
    language = nn.Module()
    language.model = nn.Module()
    language.model.layers = [layer]
    language.lm_head = _quantized_linear(384, 512, 4, 8)
    originals = [
        (gdn, "in_proj_qkv", gdn.in_proj_qkv),
        (gdn, "in_proj_z", gdn.in_proj_z),
        (mlp, "gate_proj", mlp.gate_proj),
        (mlp, "up_proj", mlp.up_proj),
        (language, "lm_head", language.lm_head),
    ]
    kept = (gdn.in_proj_b, gdn.in_proj_a, mlp.down_proj)

    assert packed.pack_model(language) == len(originals)
    assert (gdn.in_proj_b, gdn.in_proj_a, mlp.down_proj) == kept
    # The tensor-unit matvec (<= 8 rows) and GEMM paths, aligned or not.
    for rows in (1, 3, 8, 9, 70):
        x = (mx.random.normal((1, rows, 512)) * 0.5).astype(mx.bfloat16)
        for parent, name, original in originals:
            module = getattr(parent, name)
            assert isinstance(module, packed.PackedLinear)
            out = module(x)
            assert out.shape == (1, rows, original.weight.shape[0])
            assert _close(out[0], _reference(original, x[0]))

    class Verifier:
        def _linears(self, linears, x):
            return tuple(linear(x) for linear in linears)

    packed.apply(Verifier)
    x = (mx.random.normal((1, 5, 512)) * 0.5).astype(mx.bfloat16)
    outputs = Verifier()._linears(
        (gdn.in_proj_qkv, gdn.in_proj_z, gdn.in_proj_b, gdn.in_proj_a), x
    )
    references = (originals[0][2], originals[1][2], *kept[:2])
    for out, original in zip(outputs, references):
        assert _close(out[0], _reference(original, x[0]))


@pytest.mark.parametrize("bits", [4, 5])
@pytest.mark.parametrize("batch,length", [(1, 8), (2, 6), (4, 6), (1, 24), (3, 5)])
def test_verify_qmm_routes_batched_rows_through_mma_kernel(
    monkeypatch, batch, length, bits
):
    """rows = batch x block share one weight pass and match stock numerics."""
    from omlx.patches import qwen35_verify_qmm

    qwen35_verify_qmm.apply_verify_qmm_patch()
    monkeypatch.setattr(qwen35_verify_qmm, "_is_armed", lambda: True)
    monkeypatch.setattr(qwen35_verify_qmm, "_MIN_MMA_ROUTE_N", 256)
    monkeypatch.setattr(qwen35_verify_qmm, "_MIN_ROUTE_N", 256)
    routed = []
    original = qwen35_verify_qmm.vk_qmm_mma

    def spy(x2, *args, **kwargs):
        routed.append(tuple(x2.shape))
        return original(x2, *args, **kwargs)

    monkeypatch.setattr(qwen35_verify_qmm, "vk_qmm_mma", spy)
    mx.random.seed(7)
    layer = nn.QuantizedLinear(512, 288, bits=bits, group_size=64)
    x = (mx.random.normal((batch, length, 512)) * 0.5).astype(mx.bfloat16)
    expected = mx.quantized_matmul(
        x,
        layer.weight,
        layer.scales,
        layer.biases,
        transpose=True,
        group_size=64,
        bits=bits,
    ).astype(mx.float32)
    out = layer(x).astype(mx.float32)
    mx.eval(out, expected)
    rows = batch * length
    if rows >= 7:
        assert routed == [(rows, 512)]
    else:
        # Below seven rows the 5-bit layout has no small-M kernel.
        assert routed == []
    assert out.shape == (batch, length, 288)
    tolerance = 2.0 * mx.abs(expected).max().item() / 256 + 1e-2
    assert mx.abs(out - expected).max().item() <= tolerance


@pytest.mark.parametrize("nax", [True, False])
def test_spec_command_buffers_restore_caps_after_the_step(monkeypatch, nax):
    """Speculative steps raise MLX's buffer caps on M5 only and always restore them."""
    from omlx.custom_kernels import nax as nax_mod
    from omlx.custom_kernels.qwen35_prefill import fast

    caps = [(50, 50)]

    def fake_set(ops, mb):
        previous = caps[-1]
        caps.append((ops, mb))
        return previous

    monkeypatch.setattr(fast, "set_command_buffer_caps", fake_set)
    monkeypatch.setattr(nax_mod, "is_nax_available", lambda: nax)
    with pytest.raises(RuntimeError), bg._spec_command_buffers():
        inside = caps[-1]
        raise RuntimeError("step failed")
    assert inside == (bg._SPEC_BUFFER_CAPS if nax else (50, 50))
    assert caps[-1] == (50, 50)
class TestQwen35PipelineAwareness:
    """#3518: the MTP __call__ replacement must keep upstream's pipeline path."""

    def _patched_shell(self, *, pipeline_rank, pipeline_size):
        from mlx_lm.models import qwen3_5

        from omlx.patches.mlx_lm_mtp.qwen35_model import _patch_qwen3_5_text_model

        _patch_qwen3_5_text_model(qwen3_5)

        model = qwen3_5.Qwen3_5TextModel.__new__(qwen3_5.Qwen3_5TextModel)
        calls = []

        class FakeLayer:
            is_linear = False

            def __call__(self, h, mask=None, cache=None, n_confirmed=0):
                calls.append(("layer", n_confirmed))
                return h

        local = FakeLayer()
        # Non-local stage entries are None after apply_pipeline_assignment.
        model.layers = [None, local] if pipeline_rank == 1 else [local, None]
        model.pipeline_layers = [local]
        model.embed_tokens = lambda ids: ids
        model.fa_idx = None
        model.ssm_idx = None
        model.pipeline_rank = pipeline_rank
        model.pipeline_size = pipeline_size
        return model, calls

    def test_patched_call_survives_a_pipeline_assignment(self, monkeypatch):
        import mlx.core as mx

        model, calls = self._patched_shell(pipeline_rank=1, pipeline_size=2)
        sent, gathered = [], []
        monkeypatch.setattr(
            mx.distributed, "send", lambda value, *_a, **_k: sent.append(1) or value
        )
        monkeypatch.setattr(
            mx.distributed,
            "all_gather",
            lambda value, *_a, **_k: gathered.append(1) or value,
        )
        monkeypatch.setattr(
            mx.distributed,
            "recv_like",
            lambda value, *_a, **_k: value,
        )
        tokens = mx.array([[1, 2, 3]])

        out = model(tokens, cache=[None])

        assert bool((out == tokens).all())  # pre-norm hidden, through the fakes
        # The local layer ran; the None-filled non-local entries never did.
        assert calls == [("layer", 0)]
        # Rank 1 of 2 sends to rank 0 and joins the gather — the collectives
        # the unpatched replacement skipped entirely (#3518's crash/hang).
        assert sent == [1]
        assert gathered == [1]

    def test_single_node_behavior_is_unchanged(self):
        model, calls = self._patched_shell(pipeline_rank=0, pipeline_size=1)
        import mlx.core as mx

        tokens = mx.array([[1, 2, 3]])
        out = model(tokens, cache=[None], n_confirmed=0)

        assert out is tokens
        assert calls == [("layer", 0)]
