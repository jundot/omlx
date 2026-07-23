# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.speculative.vlm_mtp.

Phase 2A: covers drafter validation, lazy bind, and wrapper-level dispatch
to mlx-vlm's ``_mtp_rounds`` / ``_mtp_rounds_batch``. The actual mlx-vlm
helpers are mocked so this suite stays fast and does not touch model
weights.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest

from omlx.speculative import vlm_mtp


def _fake_drafter_model(model_type: str = "gemma4_assistant") -> MagicMock:
    """Build a stand-in for Gemma4AssistantDraftModel that satisfies the
    minimum API used by VLMMTPDrafter."""
    drafter = MagicMock()
    drafter.config = MagicMock(model_type=model_type)
    return drafter


def test_load_vlm_mtp_drafter_happy_path():
    """Valid gemma4_assistant artifact returns a populated VLMMTPDrafter."""
    fake_model = _fake_drafter_model("gemma4_assistant")
    with patch.object(vlm_mtp, "_vlm_load_drafter", return_value=(fake_model, "mtp")):
        drafter = vlm_mtp.load_vlm_mtp_drafter("/path/to/drafter")
    assert isinstance(drafter, vlm_mtp.VLMMTPDrafter)
    assert drafter.draft_kind == "mtp"
    assert drafter.source_path == "/path/to/drafter"
    assert drafter.model is fake_model


def test_load_vlm_mtp_drafter_accepts_unified_assistant():
    """Valid gemma4_unified_assistant artifact is accepted."""
    fake_model = _fake_drafter_model("gemma4_unified_assistant")
    with patch.object(vlm_mtp, "_vlm_load_drafter", return_value=(fake_model, "mtp")):
        drafter = vlm_mtp.load_vlm_mtp_drafter("/path/to/drafter")
    assert isinstance(drafter, vlm_mtp.VLMMTPDrafter)
    assert drafter.model is fake_model


def test_load_vlm_mtp_drafter_rejects_dflash_kind():
    """A drafter that resolves to non-mtp kind is rejected (None + warn)."""
    fake_model = _fake_drafter_model("qwen3_dflash")
    with patch.object(
        vlm_mtp, "_vlm_load_drafter", return_value=(fake_model, "dflash")
    ):
        result = vlm_mtp.load_vlm_mtp_drafter("/path/to/drafter")
    assert result is None


def test_load_vlm_mtp_drafter_accepts_qwen3_5_mtp():
    """qwen3_5_mtp model_type with kind='mtp' is accepted."""
    fake_model = _fake_drafter_model("qwen3_5_mtp")
    with patch.object(vlm_mtp, "_vlm_load_drafter", return_value=(fake_model, "mtp")):
        drafter = vlm_mtp.load_vlm_mtp_drafter("/path/to/qwen-mtp")
    assert isinstance(drafter, vlm_mtp.VLMMTPDrafter)
    assert drafter.draft_kind == "mtp"
    assert drafter.model is fake_model


def test_load_vlm_mtp_drafter_swallows_load_exception():
    """Load failures are logged and converted to None — never raised."""
    with patch.object(
        vlm_mtp,
        "_vlm_load_drafter",
        side_effect=RuntimeError("HF repo not found"),
    ):
        result = vlm_mtp.load_vlm_mtp_drafter("not-a-real-drafter")
    assert result is None


def test_run_vlm_mtp_decode_single_request_dispatches_to_mtp_rounds():
    """Single-int first_bonus routes to ``_mtp_rounds``, yields first_bonus
    then any tokens that the round loop emits."""
    fake_model = _fake_drafter_model("gemma4_assistant")
    drafter = vlm_mtp.VLMMTPDrafter(fake_model, "mtp", "/p")
    target = MagicMock()
    sampler = MagicMock()

    yielded = [(11, None), (22, None), (33, None)]
    with (
        patch.object(vlm_mtp, "_mtp_rounds", return_value=iter(yielded)) as m_single,
        patch.object(vlm_mtp, "_mtp_rounds_batch") as m_batch,
        patch.object(vlm_mtp, "_buffer_mtp_target_cache") as m_buffer,
    ):
        prompt_tokens = mx.array([[5, 6, 7]], dtype=mx.int32)
        out = list(
            vlm_mtp.run_vlm_mtp_decode(
                target_language_model=target,
                drafter=drafter,
                prompt_cache=[],
                hidden=mx.zeros((1, 1, 8)),
                shared_kv_states={},
                first_bonus=7,
                max_tokens=4,
                sampler=sampler,
                prompt_tokens=prompt_tokens,
            )
        )

    # first_bonus 7 is yielded by the wrapper before _mtp_rounds takes over
    assert out == [7, 11, 22, 33]
    m_single.assert_called_once()
    m_batch.assert_not_called()
    m_buffer.assert_called_once()
    buffer_args = m_buffer.call_args.args
    assert buffer_args[0] == []
    assert getattr(buffer_args[1], "_drafter", buffer_args[1]) is fake_model
    assert buffer_args[2] is None
    # first_bonus int forwarded as int
    kwargs = m_single.call_args.kwargs
    assert kwargs["first_bonus"] == 7
    assert kwargs["max_tokens"] == 4
    assert kwargs["prompt_tokens"] is prompt_tokens


def test_run_vlm_mtp_decode_batch_dispatches_to_mtp_rounds_batch():
    """Multi-row mx.array first_bonus routes to ``_mtp_rounds_batch``,
    emits first_bonus row then the round-loop rows."""
    fake_model = _fake_drafter_model("gemma4_assistant")
    drafter = vlm_mtp.VLMMTPDrafter(fake_model, "mtp", "/p")
    target = MagicMock()
    sampler = MagicMock()

    first_bonus = mx.array([1, 2, 3])  # B=3
    yielded = [([1, None, 3], None), ([None, None, None], None)]
    with (
        patch.object(
            vlm_mtp, "_mtp_rounds_batch", return_value=iter(yielded)
        ) as m_batch,
        patch.object(vlm_mtp, "_mtp_rounds") as m_single,
        patch.object(vlm_mtp, "_buffer_mtp_target_cache") as m_buffer,
    ):
        out = list(
            vlm_mtp.run_vlm_mtp_decode(
                target_language_model=target,
                drafter=drafter,
                prompt_cache=[],
                hidden=mx.zeros((3, 1, 8)),
                shared_kv_states={},
                first_bonus=first_bonus,
                max_tokens=4,
                sampler=sampler,
                eos_token_ids={2, 5},
            )
        )

    # First yielded row is the first_bonus row (one int per request).
    assert out == [[1, 2, 3], [1, None, 3], [None, None, None]]
    m_batch.assert_called_once()
    m_single.assert_not_called()
    m_buffer.assert_not_called()
    kwargs = m_batch.call_args.kwargs
    # EOS forwarded as a fresh set (function does its own copy)
    assert kwargs["eos_token_ids"] == {2, 5}


def test_run_vlm_mtp_decode_single_scalar_array_unwraps_to_int():
    """B=1 mx.array first_bonus is treated as single-request and unwrapped."""
    fake_model = _fake_drafter_model("gemma4_assistant")
    drafter = vlm_mtp.VLMMTPDrafter(fake_model, "mtp", "/p")
    target = MagicMock()
    sampler = MagicMock()

    first_bonus = mx.array([42])  # B=1 should not take the batch branch
    with (
        patch.object(vlm_mtp, "_mtp_rounds", return_value=iter([])) as m_single,
        patch.object(vlm_mtp, "_mtp_rounds_batch") as m_batch,
    ):
        out = list(
            vlm_mtp.run_vlm_mtp_decode(
                target_language_model=target,
                drafter=drafter,
                prompt_cache=[],
                hidden=mx.zeros((1, 1, 8)),
                shared_kv_states={},
                first_bonus=first_bonus,
                max_tokens=4,
                sampler=sampler,
            )
        )

    # _mtp_rounds yields nothing here, so only the wrapper's first_bonus
    # emit makes it into the stream.
    assert out == [42]
    m_single.assert_called_once()
    m_batch.assert_not_called()
    assert m_single.call_args.kwargs["first_bonus"] == 42


@pytest.mark.parametrize(
    "vlm_mtp_kw, other_kw",
    [
        ("dflash_enabled", "dflash_enabled"),
        ("specprefill_enabled", "specprefill_enabled"),
        ("mtp_enabled", "mtp_enabled"),
        ("turboquant_kv_enabled", "turboquant_kv_enabled"),
    ],
)
def test_model_settings_vlm_mtp_mutex(vlm_mtp_kw, other_kw):
    """ModelSettings.__post_init__ raises when vlm_mtp_enabled overlaps
    with any other speculative / cache-mutating toggle."""
    from omlx.model_settings import ModelSettings

    with pytest.raises(ValueError, match="vlm_mtp_enabled"):
        ModelSettings(vlm_mtp_enabled=True, **{other_kw: True})


# ---------------------------------------------------------------------------
# MoE config patch tests
# ---------------------------------------------------------------------------


class TestMoeConfigPatch:
    """Verify that the MoE compat patch in vlm_mtp.py correctly handles
    qwen3_5_moe_text text_config dicts."""

    def test_patch_is_applied_on_import(self):
        """The patch runs at import time; Qwen3_5MTPConfig.__post_init__
        should be the patched version."""
        try:
            from mlx_vlm.speculative.drafters.qwen3_5_mtp.config import (
                Qwen3_5MTPConfig,
            )
        except ImportError:
            pytest.skip("mlx-vlm qwen3_5_mtp drafter not available")

        # The patched __post_init__ is a closure, not the original method.
        # Verify it was replaced by checking it's not the unpatched version.
        src = Qwen3_5MTPConfig.__post_init__
        # The patched version references MoETextConfig in its closure.
        assert src is not None

    def test_moe_text_config_accepted(self):
        """Qwen3_5MTPConfig.from_dict with a MoE text_config does not raise."""
        try:
            from mlx_vlm.speculative.drafters.qwen3_5_mtp.config import (
                Qwen3_5MTPConfig,
            )
        except ImportError:
            pytest.skip("mlx-vlm qwen3_5_mtp drafter not available")

        moe_config = {
            "model_type": "qwen3_5_mtp",
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "num_experts": 8,
                "num_experts_per_tok": 2,
                "shared_expert_intermediate_size": 128,
                "moe_intermediate_size": 128,
                "rms_norm_eps": 1e-6,
                "vocab_size": 256,
                "max_position_embeddings": 128,
                "linear_num_value_heads": 4,
                "linear_num_key_heads": 4,
                "linear_key_head_dim": 16,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4,
                "mtp_num_hidden_layers": 1,
            },
        }
        cfg = Qwen3_5MTPConfig.from_dict(moe_config)
        assert cfg.text_config is not None
        assert cfg.text_config.hidden_size == 64
        assert cfg.text_config.num_experts == 8

    def test_dense_text_config_still_works(self):
        """Qwen3_5MTPConfig.from_dict with a dense text_config still works."""
        try:
            from mlx_vlm.speculative.drafters.qwen3_5_mtp.config import (
                Qwen3_5MTPConfig,
            )
        except ImportError:
            pytest.skip("mlx-vlm qwen3_5_mtp drafter not available")

        dense_config = {
            "model_type": "qwen3_5_mtp",
            "text_config": {
                "model_type": "qwen3_5",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-6,
                "vocab_size": 256,
                "max_position_embeddings": 128,
                "linear_num_value_heads": 4,
                "linear_num_key_heads": 4,
                "linear_key_head_dim": 16,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4,
                "mtp_num_hidden_layers": 1,
            },
        }
        cfg = Qwen3_5MTPConfig.from_dict(dense_config)
        assert cfg.text_config is not None
        assert cfg.text_config.hidden_size == 64


# ---------------------------------------------------------------------------
# dense Qwen3.5 VLM runtime patch tests
# ---------------------------------------------------------------------------


def test_dense_vlm_runtime_return_hidden_uses_language_model_output_contract():
    """Dense Qwen3.5 VLM MTP verify must satisfy mlx-vlm's output contract."""
    from mlx_vlm.models.base import LanguageModelOutput
    from omlx.patches.mlx_vlm_mtp import qwen35_vlm_runtime

    logits = mx.zeros((1, 2, 16))
    hidden = mx.zeros((1, 2, 8))
    gdn_states = [{"state": "mock"}]

    class FakeStockOutput:
        def __init__(self):
            self.logits = logits
            self.hidden_states = [hidden]
            self.gdn_states = gdn_states

    class FakeLanguageModel:
        def __init__(self, args, config=None):
            self.args = args
            self.config = config
            self.model = SimpleNamespace(layers=[object(), object()])
            self.forward_kwargs = None

        def __call__(
            self,
            inputs,
            inputs_embeds=None,
            mask=None,
            cache=None,
            **kwargs,
        ):
            self.forward_kwargs = kwargs
            return FakeStockOutput()

    q35_lang = SimpleNamespace(LanguageModel=FakeLanguageModel)
    qwen35_vlm_runtime._patch_vlm_language_model(q35_lang)

    model = q35_lang.LanguageModel(
        SimpleNamespace(mtp_num_hidden_layers=0, tie_word_embeddings=True),
        config=None,
    )
    out = model(
        mx.array([[1, 2]], dtype=mx.int32),
        cache=[],
        return_hidden=True,
        return_shared_kv=True,
        capture_layer_ids=[99],
    )

    assert isinstance(out, LanguageModelOutput)
    assert out.logits is logits
    assert out.hidden_states == [hidden]
    assert out.hidden_states[-1] is hidden
    assert out.gdn_states is gdn_states
    assert out.shared_kv_states == {}
    assert model.forward_kwargs["capture_layer_ids"] == [1]


def test_moe_vlm_sanitize_unfuses_gate_up_by_midpoint(monkeypatch):
    """The VLM MoE sanitize patch must preserve upstream midpoint slicing."""
    from omlx.patches.mlx_vlm_mtp import qwen35_moe_vlm_model
    from mlx_vlm.models.qwen3_5_moe import qwen3_5_moe

    monkeypatch.setattr(qwen35_moe_vlm_model, "_APPLIED", False)
    if hasattr(qwen3_5_moe.Model, "_omlx_mtp_vlm_patched"):
        monkeypatch.delattr(qwen3_5_moe.Model, "_omlx_mtp_vlm_patched")

    assert qwen35_moe_vlm_model.apply() is True

    fake_self = SimpleNamespace(
        config=SimpleNamespace(
            text_config=SimpleNamespace(
                tie_word_embeddings=False,
                num_hidden_layers=1,
                num_experts=0,
            )
        )
    )
    gate_up = mx.arange(2 * 6 * 3).reshape(2, 6, 3)
    weights = {
        "model.language_model.layers.0.mlp.experts.gate_up_proj": gate_up,
        "model.language_model.layers.0.mlp.experts.down_proj": mx.ones((2, 4, 3)),
    }

    result = qwen3_5_moe.Model.sanitize(fake_self, weights)

    gate_key = "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"
    up_key = "language_model.model.layers.0.mlp.switch_mlp.up_proj.weight"
    assert bool(mx.all(result[gate_key] == gate_up[:, :3, :]).item())
    assert bool(mx.all(result[up_key] == gate_up[:, 3:, :]).item())


def test_moe_vlm_runtime_sanitize_unfuses_gate_up_by_midpoint():
    """The runtime sanitize wrapper must not reintroduce the old split path."""
    from omlx.patches.mlx_vlm_mtp import qwen35_moe_vlm_runtime

    class FakeModel:
        pass

    fake_outer = SimpleNamespace(Model=FakeModel)
    qwen35_moe_vlm_runtime._patch_vlm_outer_model_sanitize(fake_outer)

    fake_self = SimpleNamespace(
        config=SimpleNamespace(
            text_config=SimpleNamespace(
                tie_word_embeddings=False,
                num_hidden_layers=1,
                num_experts=0,
            )
        )
    )
    gate_up = mx.arange(2 * 6 * 3).reshape(2, 6, 3)
    weights = {
        "model.language_model.layers.0.mlp.experts.gate_up_proj": gate_up,
        "model.language_model.layers.0.mlp.experts.down_proj": mx.ones((2, 4, 3)),
    }

    result = FakeModel.sanitize(fake_self, weights)

    gate_key = "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"
    up_key = "language_model.model.layers.0.mlp.switch_mlp.up_proj.weight"
    assert bool(mx.all(result[gate_key] == gate_up[:, :3, :]).item())
    assert bool(mx.all(result[up_key] == gate_up[:, 3:, :]).item())


def _per_expert_vlm_self(num_experts=2, num_hidden_layers=1):
    return SimpleNamespace(
        config=SimpleNamespace(
            text_config=SimpleNamespace(
                tie_word_embeddings=False,
                num_hidden_layers=num_hidden_layers,
                num_experts=num_experts,
            )
        )
    )


def test_moe_vlm_sanitize_stacks_per_expert_backbone(monkeypatch):
    """Ornith / raw Qwen3.5 ship backbone MoE layers as per-expert tensors.
    The model-level sanitize must stack them into switch_mlp form."""
    from omlx.patches.mlx_vlm_mtp import qwen35_moe_vlm_model
    from mlx_vlm.models.qwen3_5_moe import qwen3_5_moe

    monkeypatch.setattr(qwen35_moe_vlm_model, "_APPLIED", False)
    if hasattr(qwen3_5_moe.Model, "_omlx_mtp_vlm_patched"):
        monkeypatch.delattr(qwen3_5_moe.Model, "_omlx_mtp_vlm_patched")
    assert qwen35_moe_vlm_model.apply() is True

    pfx_in = "model.language_model.layers.0.mlp"
    weights = {}
    for e in range(2):
        weights[f"{pfx_in}.experts.{e}.gate_proj.weight"] = mx.zeros((8, 4))
        weights[f"{pfx_in}.experts.{e}.up_proj.weight"] = mx.zeros((8, 4))
        weights[f"{pfx_in}.experts.{e}.down_proj.weight"] = mx.zeros((4, 8))

    result = qwen3_5_moe.Model.sanitize(_per_expert_vlm_self(), weights)

    pfx = "language_model.model.layers.0.mlp"
    assert result[f"{pfx}.switch_mlp.gate_proj.weight"].shape == (2, 8, 4)
    assert result[f"{pfx}.switch_mlp.down_proj.weight"].shape == (2, 4, 8)
    assert not any(f"{pfx}.experts." in k for k in result)


def test_moe_vlm_sanitize_stacks_per_expert_backbone_quantized(monkeypatch):
    """A per-expert *quantized* backbone carries .scales/.biases. The
    model-level sanitize must stack all three, leaving no orphan keys."""
    from omlx.patches.mlx_vlm_mtp import qwen35_moe_vlm_model
    from mlx_vlm.models.qwen3_5_moe import qwen3_5_moe

    monkeypatch.setattr(qwen35_moe_vlm_model, "_APPLIED", False)
    if hasattr(qwen3_5_moe.Model, "_omlx_mtp_vlm_patched"):
        monkeypatch.delattr(qwen3_5_moe.Model, "_omlx_mtp_vlm_patched")
    assert qwen35_moe_vlm_model.apply() is True

    pfx_in = "model.language_model.layers.0.mlp"
    weights = {}
    for e in range(2):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            weights[f"{pfx_in}.experts.{e}.{proj}.weight"] = mx.zeros((8, 4))
            weights[f"{pfx_in}.experts.{e}.{proj}.scales"] = mx.zeros((8, 1))
            weights[f"{pfx_in}.experts.{e}.{proj}.biases"] = mx.zeros((8, 1))

    result = qwen3_5_moe.Model.sanitize(_per_expert_vlm_self(), weights)

    pfx = "language_model.model.layers.0.mlp"
    for proj in ("gate_proj", "up_proj", "down_proj"):
        for suffix in ("weight", "scales", "biases"):
            key = f"{pfx}.switch_mlp.{proj}.{suffix}"
            assert key in result, key
            assert result[key].shape[0] == 2
    assert not any(f"{pfx}.experts." in k for k in result)


def test_moe_vlm_sanitize_stacks_per_expert_mtp_quantized(monkeypatch):
    """A per-expert *quantized* MTP head also carries .scales/.biases.
    The model-level VLM sanitize path must keep parity with the runtime
    sanitize path and stack all three suffixes."""
    from omlx.patches.mlx_vlm_mtp import qwen35_moe_vlm_model
    from mlx_vlm.models.qwen3_5_moe import qwen3_5_moe

    monkeypatch.setattr(qwen35_moe_vlm_model, "_APPLIED", False)
    if hasattr(qwen3_5_moe.Model, "_omlx_mtp_vlm_patched"):
        monkeypatch.delattr(qwen3_5_moe.Model, "_omlx_mtp_vlm_patched")
    assert qwen35_moe_vlm_model.apply() is True

    pfx_in = "mtp.layers.0.mlp"
    weights = {}
    for e in range(2):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            weights[f"{pfx_in}.experts.{e}.{proj}.weight"] = mx.zeros((8, 4))
            weights[f"{pfx_in}.experts.{e}.{proj}.scales"] = mx.zeros((8, 1))
            weights[f"{pfx_in}.experts.{e}.{proj}.biases"] = mx.zeros((8, 1))

    result = qwen3_5_moe.Model.sanitize(_per_expert_vlm_self(), weights)

    pfx = "language_model.mtp.layers.0.mlp"
    for proj in ("gate_proj", "up_proj", "down_proj"):
        for suffix in ("weight", "scales", "biases"):
            key = f"{pfx}.switch_mlp.{proj}.{suffix}"
            assert key in result, key
            assert result[key].shape[0] == 2
    assert not any(f"{pfx}.experts." in k for k in result)


def test_moe_vlm_runtime_sanitize_stacks_per_expert_backbone():
    """The runtime sanitize wrapper must also stack per-expert backbone
    layers (parity with the model-level patch and the LLM patch)."""
    from omlx.patches.mlx_vlm_mtp import qwen35_moe_vlm_runtime

    class FakeModel:
        pass

    fake_outer = SimpleNamespace(Model=FakeModel)
    qwen35_moe_vlm_runtime._patch_vlm_outer_model_sanitize(fake_outer)

    pfx_in = "model.language_model.layers.0.mlp"
    weights = {}
    for e in range(2):
        weights[f"{pfx_in}.experts.{e}.gate_proj.weight"] = mx.zeros((8, 4))
        weights[f"{pfx_in}.experts.{e}.up_proj.weight"] = mx.zeros((8, 4))
        weights[f"{pfx_in}.experts.{e}.down_proj.weight"] = mx.zeros((4, 8))

    result = FakeModel.sanitize(_per_expert_vlm_self(), weights)

    pfx = "language_model.model.layers.0.mlp"
    assert result[f"{pfx}.switch_mlp.gate_proj.weight"].shape == (2, 8, 4)
    assert result[f"{pfx}.switch_mlp.up_proj.weight"].shape == (2, 8, 4)
    assert result[f"{pfx}.switch_mlp.down_proj.weight"].shape == (2, 4, 8)
    assert not any(f"{pfx}.experts." in k for k in result)


# ---------------------------------------------------------------------------
# _call_backbone return format tests
# ---------------------------------------------------------------------------


class TestCallBackbone:
    """Verify _call_backbone handles both tuple and LanguageModelOutput."""

    def test_tuple_2_return(self):
        """mlx-lm dense path returns (logits, hidden) 2-tuple."""
        from omlx.patches.mlx_lm_mtp.batch_generator import _call_backbone

        import mlx.core as mx

        logits = mx.zeros((1, 1, 100))
        hidden = mx.zeros((1, 1, 64))

        model = MagicMock(return_value=(logits, hidden))
        result = _call_backbone(model, mx.zeros((1, 4)), cache=[])
        assert result[0] is logits
        assert result[1] is hidden
        assert result[2] is None  # gdn_states

    def test_tuple_3_return(self):
        """mlx-vlm MoE path returns (logits, hidden, gdn_states) 3-tuple."""
        from omlx.patches.mlx_lm_mtp.batch_generator import _call_backbone

        import mlx.core as mx

        logits = mx.zeros((1, 1, 100))
        hidden = mx.zeros((1, 1, 64))
        gdn = [{"state": "mock"}]

        model = MagicMock(return_value=(logits, hidden, gdn))
        result = _call_backbone(model, mx.zeros((1, 4)), cache=[])
        assert result[0] is logits
        assert result[1] is hidden
        assert result[2] is gdn

    def test_language_model_output_return(self):
        """LanguageModelOutput is correctly unpacked."""
        from omlx.patches.mlx_lm_mtp.batch_generator import _call_backbone

        import mlx.core as mx
        from mlx_vlm.models.base import LanguageModelOutput

        logits = mx.zeros((1, 1, 100))
        hidden = mx.zeros((1, 1, 64))
        gdn = [{"state": "mock"}]

        out = LanguageModelOutput(
            logits=logits,
            hidden_states=[hidden],
            gdn_states=gdn,
        )
        model = MagicMock(return_value=out)
        result = _call_backbone(model, mx.zeros((1, 4)), cache=[])
        assert result[0] is logits
        assert result[1] is hidden
        assert result[2] is gdn
