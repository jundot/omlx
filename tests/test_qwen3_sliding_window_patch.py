# SPDX-License-Identifier: Apache-2.0
"""Tests for the Qwen3 sliding-window compatibility patch."""

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.models import qwen3 as upstream_qwen3

from omlx.patches import qwen3_sliding_window as patch_module
from omlx.patches.qwen3_sliding_window import qwen3_model


def _model_args(args_class, **overrides):
    values = {
        "model_type": "qwen3",
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "rms_norm_eps": 1e-6,
        "vocab_size": 32,
        "num_key_value_heads": 1,
        "max_position_embeddings": 32,
        "rope_theta": 10000.0,
        "head_dim": 4,
        "tie_word_embeddings": True,
    }
    values.update(overrides)
    return args_class(**values)


def test_full_attention_fallback_matches_upstream_qwen3():
    """Configs without layer_types must preserve stock Qwen3 output."""
    upstream = upstream_qwen3.Qwen3Model(_model_args(upstream_qwen3.ModelArgs))
    patched = qwen3_model.Qwen3Model(_model_args(qwen3_model.ModelArgs))

    mx.eval(upstream.parameters())
    patched.load_weights(list(tree_flatten(upstream.parameters())), strict=True)

    inputs = mx.array([[1, 2, 3, 4]])
    expected = upstream(inputs)
    actual = patched(inputs)
    mx.eval(expected, actual)

    assert np.array_equal(np.array(actual), np.array(expected))


def test_sliding_layers_run_chunked_and_match_the_window_mask(monkeypatch):
    """Sliding layers must skip the full window mask and keep its output."""
    calls = []
    create_attention_mask = qwen3_model.create_attention_mask

    def _recording_mask(h, cache=None, window_size=None):
        calls.append(window_size)
        return create_attention_mask(h, cache, window_size=window_size)

    monkeypatch.setattr(qwen3_model, "create_attention_mask", _recording_mask)
    args = _model_args(
        qwen3_model.ModelArgs,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=3,
    )
    model = qwen3_model.Qwen3Model(args)
    mx.eval(model.parameters())
    inputs = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])

    actual = model(inputs)
    h = model.embed_tokens(inputs)
    window_mask = create_attention_mask(h, None, window_size=3)
    for is_sliding, layer in zip(model.is_sliding, model.layers):
        h = layer(h, mask=window_mask if is_sliding else "causal")
    expected = model.norm(h)
    mx.eval(actual, expected)

    assert model.is_sliding == [True, False]
    assert calls == [None]
    assert np.allclose(np.array(actual), np.array(expected), atol=1e-5)


def test_patch_install_is_idempotent_and_updates_live_module(monkeypatch):
    """mlx-lm class lookup must see the patched classes exactly once."""
    original_args = upstream_qwen3.ModelArgs
    original_model = upstream_qwen3.Qwen3Model
    monkeypatch.setattr(upstream_qwen3, "ModelArgs", original_args)
    monkeypatch.setattr(upstream_qwen3, "Qwen3Model", original_model)
    monkeypatch.setattr(patch_module, "_APPLIED", False)

    assert patch_module.apply_qwen3_sliding_window_patch() is True
    assert upstream_qwen3.ModelArgs is qwen3_model.ModelArgs
    assert upstream_qwen3.Qwen3Model is qwen3_model.Qwen3Model
    assert patch_module.apply_qwen3_sliding_window_patch() is False
