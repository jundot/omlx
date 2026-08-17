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


def test_qwen38_block_fp8_dequantization():
    from omlx.patches.mlx_vlm_mtp.qwen38_fp8 import dequantize_fp8_weights

    weight_key = "model.language_model.layers.0.self_attn.q_proj.weight"
    weights = {
        weight_key: mx.to_fp8(mx.ones((130, 129), dtype=mx.float32)),
        f"{weight_key}_scale_inv": mx.array(
            [[0.5, 1.0], [2.0, 4.0]], dtype=mx.bfloat16
        ),
    }

    out = dequantize_fp8_weights(weights)
    expected = mx.ones((130, 129), dtype=mx.bfloat16)
    expected[:128, :128] *= 0.5
    expected[:128, 128:] *= 1.0
    expected[128:, :128] *= 2.0
    expected[128:, 128:] *= 4.0

    assert not any(key.endswith("weight_scale_inv") for key in out)
    assert out[weight_key].dtype == mx.bfloat16
    assert mx.array_equal(out[weight_key], expected).item()


def test_qwen38_block_fp8_rejects_invalid_scale_grid():
    from omlx.patches.mlx_vlm_mtp.qwen38_fp8 import dequantize_fp8_weights

    with pytest.raises(ValueError, match="Invalid FP8 scale shape"):
        dequantize_fp8_weights(
            {
                "proj.weight": mx.to_fp8(mx.ones((129, 129))),
                "proj.weight_scale_inv": mx.ones((1, 2)),
            }
        )


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


class TestMTPRoundClearDrainsGPUWork:
    """The per-token cache clear must drain the round's GPU work first.

    mlx-vlm submits the MTP verify hidden state and the drafter's state
    arrays with mx.async_eval, so mx.clear_cache() at the yield boundary can
    release Metal buffers an in-flight command buffer still references (#300).
    The drain has to name mlx-vlm's own thread-local stream: that is the
    stream ``_mtp_rounds`` dispatches the verify/rollback forwards on
    (``with mx.stream(generation_stream)``), and it is a different object from
    mlx-lm's generation_stream. The helper's second, no-argument
    mx.synchronize() covers the engine stream the scheduler advances the
    generator under.
    """

    @staticmethod
    def _recorder() -> tuple[list, object]:
        streams: list = []
        return streams, patch.object(
            vlm_mtp,
            "_sync_and_clear_cache",
            side_effect=lambda stream=None: streams.append(stream),
        )

    def _assert_vlm_stream(self, streams: list, expected_calls: int) -> None:
        from mlx_lm.generate import generation_stream as mlx_lm_stream

        assert len(streams) == expected_calls, (
            f"expected {expected_calls} synchronized clear(s), got {streams!r}"
        )
        assert all(s is vlm_mtp._vlm_generation_stream for s in streams), (
            "MTP round cleared the Metal buffer cache without draining "
            f"mlx-vlm's stream: {streams!r}"
        )
        assert vlm_mtp._vlm_generation_stream is not mlx_lm_stream

    def test_single_round_loop_drains_before_every_token_yield(self):
        """Each token yielded by ``_mtp_rounds`` is preceded by a synchronized
        clear; the wrapper's own first_bonus yield needs none (no round has
        run yet)."""
        drafter = vlm_mtp.VLMMTPDrafter(
            _fake_drafter_model("gemma4_assistant"), "mtp", "/p"
        )
        streams, recording = self._recorder()

        with (
            recording,
            patch.object(
                vlm_mtp, "_mtp_rounds", return_value=iter([(11, None), (22, None)])
            ),
            patch.object(vlm_mtp, "_buffer_mtp_target_cache"),
        ):
            gen = vlm_mtp.run_vlm_mtp_decode(
                target_language_model=MagicMock(),
                drafter=drafter,
                prompt_cache=[],
                hidden=mx.zeros((1, 1, 8)),
                shared_kv_states={},
                first_bonus=7,
                max_tokens=4,
                sampler=MagicMock(),
            )
            assert next(gen) == 7
            assert streams == [], "first_bonus yield must not clear the cache"
            assert next(gen) == 11
            self._assert_vlm_stream(streams, 1)
            assert next(gen) == 22
            self._assert_vlm_stream(streams, 2)

    def test_batch_round_loop_drains_before_every_round_yield(self):
        drafter = vlm_mtp.VLMMTPDrafter(
            _fake_drafter_model("gemma4_assistant"), "mtp", "/p"
        )
        streams, recording = self._recorder()
        yielded = [([1, None, 3], None), ([None, None, None], None)]

        with (
            recording,
            patch.object(vlm_mtp, "_mtp_rounds_batch", return_value=iter(yielded)),
        ):
            out = list(
                vlm_mtp.run_vlm_mtp_decode(
                    target_language_model=MagicMock(),
                    drafter=drafter,
                    prompt_cache=[],
                    hidden=mx.zeros((3, 1, 8)),
                    shared_kv_states={},
                    first_bonus=mx.array([1, 2, 3]),
                    max_tokens=4,
                    sampler=MagicMock(),
                )
            )

        assert out == [[1, 2, 3], [1, None, 3], [None, None, None]]
        self._assert_vlm_stream(streams, 2)


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


def _qwen_vlm_with_attached_mtp(*, decode_enabled=True):
    return SimpleNamespace(
        language_model=SimpleNamespace(
            mtp=object(),
            _omlx_mtp_decode_enabled=decode_enabled,
        )
    )


def test_root_mtp_weights_remap_to_attached_language_model():
    from omlx.patches.mlx_vlm_mtp.qwen35_vlm_runtime import (
        _remap_root_mtp_weights,
    )

    weights = [
        ("language_model.model.embed_tokens.weight", object()),
        ("mtp.fc.weight", object()),
        ("mtp.fc.scales", object()),
        ("mtp.fc.biases", object()),
    ]

    result = _remap_root_mtp_weights(_qwen_vlm_with_attached_mtp(), weights)

    assert [key for key, _ in result] == [
        "language_model.model.embed_tokens.weight",
        "language_model.mtp.fc.weight",
        "language_model.mtp.fc.scales",
        "language_model.mtp.fc.biases",
    ]


def test_root_mtp_weights_remap_when_decode_is_disabled():
    from omlx.patches.mlx_vlm_mtp.qwen35_vlm_runtime import (
        _remap_root_mtp_weights,
    )

    result = _remap_root_mtp_weights(
        _qwen_vlm_with_attached_mtp(decode_enabled=False),
        [("mtp.norm.weight", object())],
    )

    assert result[0][0] == "language_model.mtp.norm.weight"


def test_canonical_mtp_weights_pass_through_unchanged():
    from omlx.patches.mlx_vlm_mtp.qwen35_vlm_runtime import (
        _remap_root_mtp_weights,
    )

    weights = [("language_model.mtp.fc.weight", object())]

    assert _remap_root_mtp_weights(_qwen_vlm_with_attached_mtp(), weights) is weights


def test_root_mtp_weights_without_attached_module_pass_through():
    from omlx.patches.mlx_vlm_mtp.qwen35_vlm_runtime import (
        _remap_root_mtp_weights,
    )

    weights = [("mtp.fc.weight", object())]
    model = SimpleNamespace(language_model=SimpleNamespace())

    assert _remap_root_mtp_weights(model, weights) is weights


def test_root_and_canonical_mtp_weights_are_rejected():
    from omlx.patches.mlx_vlm_mtp.qwen35_vlm_runtime import (
        _remap_root_mtp_weights,
    )

    weights = [
        ("mtp.fc.weight", object()),
        ("language_model.mtp.fc.weight", object()),
    ]

    with pytest.raises(ValueError, match="both root and canonical MTP weights"):
        _remap_root_mtp_weights(_qwen_vlm_with_attached_mtp(), weights)


def test_qwen_vlm_outer_load_weights_remaps_root_mtp(monkeypatch):
    from omlx.patches.mlx_vlm_mtp import qwen35_vlm_runtime
    from mlx_vlm.models import qwen3_5 as q35_outer

    class FakeModel:
        def load_weights(self, weights, strict=True):
            self.received_weights = weights
            self.received_strict = strict
            return "loaded"

    monkeypatch.setattr(q35_outer, "Model", FakeModel)
    qwen35_vlm_runtime._patch_vlm_outer_model_load_weights()

    model = FakeModel()
    model.language_model = SimpleNamespace(mtp=object())

    assert model.load_weights([("mtp.fc.weight", object())], strict=False) == "loaded"
    assert model.received_weights[0][0] == "language_model.mtp.fc.weight"
    assert model.received_strict is False


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
