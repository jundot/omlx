# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.patches.mlx_vlm_mtp.gemma4_vlm_runtime.

Covers assistant-config retention through ``TextConfig.from_dict``, head
attach gating on ``LanguageModel.__init__``, and the Lightning
``mtp_forward`` adapter bookkeeping (query-position source, stale-bind
refresh, rejected-tail slicing) with a stubbed drafter — no weights.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx
import pytest

pytest.importorskip("mlx_vlm.models.gemma4")
pytest.importorskip("mlx_vlm.models.gemma4_unified")

from omlx.patches import mlx_lm_mtp as lm_mtp
from omlx.patches.mlx_vlm_mtp import gemma4_vlm_runtime, set_mtp_attach_enabled

TINY_ASSISTANT_CONFIG = {
    "model_type": "gemma4_assistant",
    "backbone_hidden_size": 24,
    "tie_word_embeddings": True,
    "use_ordered_embeddings": False,
    "block_size": 4,
    "text_config": {
        "model_type": "gemma4_text",
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "intermediate_size": 32,
        "num_attention_heads": 2,
        "head_dim": 8,
        "global_head_dim": 8,
        "num_key_value_heads": 2,
        "num_global_key_value_heads": 1,
        "num_kv_shared_layers": 0,
        "vocab_size": 64,
        "sliding_window": 8,
        "sliding_window_pattern": 2,
        "attention_k_eq_v": True,
        "hidden_size_per_layer_input": 0,
        "use_double_wide_mlp": False,
    },
}

TINY_BACKBONE_CONFIG = {
    "model_type": "gemma4_text",
    "hidden_size": 24,
    "num_hidden_layers": 2,
    "intermediate_size": 32,
    "num_attention_heads": 2,
    "head_dim": 8,
    "global_head_dim": 8,
    "num_key_value_heads": 2,
    "num_global_key_value_heads": 1,
    "num_kv_shared_layers": 0,
    "vocab_size": 64,
    "sliding_window": 8,
    "sliding_window_pattern": 2,
    "attention_k_eq_v": True,
    "hidden_size_per_layer_input": 0,
    "use_double_wide_mlp": False,
}


@pytest.fixture(autouse=True)
def _applied_patch():
    assert gemma4_vlm_runtime.apply()
    set_mtp_attach_enabled(True)
    lm_mtp.set_mtp_active(False)
    yield
    lm_mtp.set_mtp_active(False)
    set_mtp_attach_enabled(True)


def _text_config(extra: dict | None = None):
    from mlx_vlm.models.gemma4.config import TextConfig

    params = dict(TINY_BACKBONE_CONFIG)
    if extra:
        params.update(extra)
    return TextConfig.from_dict(params)


def _language_model(config):
    from mlx_vlm.models.gemma4.language import LanguageModel

    return LanguageModel(config)


def _unified_text_config(extra: dict | None = None):
    from mlx_vlm.models.gemma4_unified.config import TextConfig

    params = dict(TINY_BACKBONE_CONFIG, model_type="gemma4_unified_text")
    if extra:
        params.update(extra)
    return TextConfig.from_dict(params)


def test_apply_is_idempotent():
    assert gemma4_vlm_runtime.apply()
    assert gemma4_vlm_runtime.apply()


def test_text_config_retains_assistant_config():
    cfg = _text_config({"mtp_assistant_config": TINY_ASSISTANT_CONFIG})
    assert cfg.mtp_assistant_config == TINY_ASSISTANT_CONFIG
    assert _text_config().mtp_assistant_config is None


def test_unified_text_config_retains_assistant_config():
    assistant = dict(TINY_ASSISTANT_CONFIG, model_type="gemma4_unified_assistant")
    cfg = _unified_text_config({"mtp_assistant_config": assistant})
    assert cfg.mtp_assistant_config == assistant
    assert _unified_text_config().mtp_assistant_config is None


def test_no_attach_without_assistant_config():
    lm_mtp.set_mtp_active(True)
    lm = _language_model(_text_config())
    assert getattr(lm, "mtp", None) is None
    assert lm._omlx_mtp_decode_enabled is False
    assert lm.make_mtp_cache() == []


def test_attach_without_decode_when_mtp_inactive():
    # mtp_enabled=False load: the head still attaches so persisted
    # language_model.mtp.* weights bind, but decode stays off.
    lm = _language_model(_text_config({"mtp_assistant_config": TINY_ASSISTANT_CONFIG}))
    assert lm.mtp is not None
    assert lm._omlx_mtp_decode_enabled is False
    assert not getattr(lm, "_omlx_mtp_chain", False)


def test_attach_skipped_when_attach_gate_off():
    set_mtp_attach_enabled(False)
    lm_mtp.set_mtp_active(True)
    lm = _language_model(_text_config({"mtp_assistant_config": TINY_ASSISTANT_CONFIG}))
    assert getattr(lm, "mtp", None) is None
    assert lm._omlx_mtp_decode_enabled is False


def test_attach_and_chain_flags_when_active():
    lm_mtp.set_mtp_active(True)
    lm_mtp.set_mtp_depth(3)
    lm = _language_model(_text_config({"mtp_assistant_config": TINY_ASSISTANT_CONFIG}))
    assert lm.mtp is not None
    assert lm._omlx_mtp_decode_enabled is True
    assert lm._omlx_mtp_chain is True
    assert lm._omlx_mtp_depth == 3
    assert lm.make_mtp_cache() == []
    # The drafter forces KV sharing across all of its layers.
    assert (
        lm.mtp.config.text_config.num_kv_shared_layers
        == lm.mtp.config.text_config.num_hidden_layers
    )


def _stubbed_mtp_lm(cache_entries):
    """LanguageModel with an attached stub drafter and fake cache stash."""
    lm_mtp.set_mtp_active(True)
    lm = _language_model(_text_config({"mtp_assistant_config": TINY_ASSISTANT_CONFIG}))

    drafter = MagicMock()
    drafter._input_embed = lambda ids: mx.zeros((1, 1, 24), dtype=mx.float32)
    drafter._input_embed_scale = 1.0
    drafter.return_value = (
        mx.zeros((1, 1, 24), dtype=mx.float32),
        mx.zeros((1, 1, 64), dtype=mx.float32),
    )
    lm.mtp = drafter
    lm._omlx_mtp_cache_ref = cache_entries
    return lm, drafter


def test_mtp_forward_position_prefers_rotating_absolute_offset():
    # BatchRotatingKVCache._offset is the absolute committed length; its
    # _idx is a ring index and must NOT be used.
    lm, drafter = _stubbed_mtp_lm([SimpleNamespace(_offset=5, _idx=99, offset="na")])
    lm._omlx_mtp_shared_kv = {
        "full_attention": (mx.zeros((1, 1, 7, 8)), mx.zeros((1, 1, 7, 8)))
    }
    lm._omlx_mtp_kv_offset = 7

    hidden = mx.zeros((1, 3, 24), dtype=mx.float32)
    ids = mx.zeros((1, 3), dtype=mx.uint32)
    logits, head_hidden = lm.mtp_forward(hidden, ids, [], return_hidden=True)

    assert drafter._kv_valid_len == 5
    inputs_embeds, shared_kv, position_ids = drafter.call_args.args
    # Only the last (hidden, token) pair is consumed; fused input is
    # [tok_embed(24), hidden(24)].
    assert inputs_embeds.shape == (1, 1, 48)
    # Query position = last committed slot (valid_len - 1).
    assert position_ids.tolist() == [[4]]
    # Rejected tail (7 captured - 5 committed) sliced off the stash.
    assert shared_kv["full_attention"][0].shape[-2] == 5
    assert logits.shape == (1, 1, 64)
    assert head_hidden.shape == (1, 1, 24)


def test_mtp_forward_uses_plain_int_offset_and_batch_idx():
    lm, drafter = _stubbed_mtp_lm([SimpleNamespace(offset=6)])
    lm._omlx_mtp_shared_kv = {
        "full_attention": (mx.zeros((1, 1, 6, 8)), mx.zeros((1, 1, 6, 8)))
    }
    lm._omlx_mtp_kv_offset = 6
    lm.mtp_forward(mx.zeros((1, 1, 24)), mx.zeros((1, 1), dtype=mx.uint32), [])
    assert drafter._kv_valid_len == 6

    lm._omlx_mtp_cache_ref = [SimpleNamespace(_idx=4)]
    lm._omlx_mtp_kv_offset = 4
    lm.mtp_forward(mx.zeros((1, 1, 24)), mx.zeros((1, 1), dtype=mx.uint32), [])
    assert drafter._kv_valid_len == 4


def test_mtp_forward_rebinds_stale_input_embed():
    # nn.quantize() swaps the backbone embed_tokens module after the
    # __init__-time bind; mtp_forward must re-bind so the drafter never
    # embeds through a stale (random-init) module.
    lm, drafter = _stubbed_mtp_lm([SimpleNamespace(offset=3)])
    lm._omlx_mtp_shared_kv = {
        "full_attention": (mx.zeros((1, 1, 3, 8)), mx.zeros((1, 1, 3, 8)))
    }
    lm._omlx_mtp_kv_offset = 3
    lm.mtp_forward(mx.zeros((1, 1, 24)), mx.zeros((1, 1), dtype=mx.uint32), [])
    drafter.bind.assert_called_once_with(lm)


def test_mtp_forward_requires_shared_kv_stash():
    lm, _ = _stubbed_mtp_lm([SimpleNamespace(offset=3)])
    lm._omlx_mtp_shared_kv = None
    with pytest.raises(RuntimeError, match="shared K/V stash"):
        lm.mtp_forward(mx.zeros((1, 1, 24)), mx.zeros((1, 1), dtype=mx.uint32), [])
