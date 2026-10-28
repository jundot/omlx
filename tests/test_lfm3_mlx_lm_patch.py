# SPDX-License-Identifier: Apache-2.0
"""LFM3 has to resolve through mlx-lm, or LiquidAI's checkpoint cannot load.

Pinned mlx-lm stops at LFM2. ``mlx_lm.utils._get_classes`` therefore raises
``Model type lfm3 not supported`` before a single weight is read, and the
engine reports the model as unloadable. The vendored module is Apple's
implementation from mlx-lm PR #2016; these tests pin the failure path, not just
the registration.
"""

from __future__ import annotations

import importlib
import sys

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from omlx.patches import lfm3_mlx_lm  # noqa: E402
from omlx.patches.lfm3_mlx_lm import (  # noqa: E402
    apply_lfm3_mlx_lm_patch,
    is_lfm3,
)

# Shaped from LiquidAI/LFM3-1.7B-MLX-8bit: conv, sliding-attention and
# full-attention layers alternate, and RoPE settings live per layer type.
CONFIG = {
    "model_type": "lfm3",
    "architectures": ["Lfm3ForCausalLM"],
    "layer_types": ["conv", "conv", "full_attention", "sliding_attention"],
    "rope_parameters": {
        "full_attention": {
            "rope_theta": 1000000.0,
            "rope_type": "default",
            "partial_rotary_factor": 0.5,
        },
        "sliding_attention": {"rope_theta": 10000, "rope_type": "default"},
    },
}

ARGS = dict(
    vocab_size=256,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    max_position_embeddings=128,
    norm_eps=1e-5,
    conv_bias=False,
    conv_L_cache=3,
    sliding_window=8,
    use_sandwich_norm=True,
    tie_word_embeddings=True,
    **{k: CONFIG[k] for k in ("model_type", "layer_types", "rope_parameters")},
)


@pytest.fixture(autouse=True)
def _fresh_patch_state():
    """``_APPLIED`` is process-wide; leaking it would hide a broken import."""
    lfm3_mlx_lm._APPLIED = False
    sys.modules.pop("mlx_lm.models.lfm3", None)
    import mlx_lm.models

    mlx_lm.models.__dict__.pop("lfm3", None)
    yield
    sys.modules.pop("mlx_lm.models.lfm3", None)
    mlx_lm.models.__dict__.pop("lfm3", None)


def test_the_patch_reports_which_models_it_is_for():
    assert is_lfm3({"model_type": "lfm3"})
    assert not is_lfm3({"model_type": "lfm2"})
    assert not is_lfm3({})
    assert not is_lfm3(None)


def test_the_architecture_resolves_through_mlx_lm():
    """This is the exact call that failed with ``Model type lfm3 not supported``."""
    from mlx_lm.utils import _get_classes

    with pytest.raises(ValueError, match="lfm3 not supported"):
        _get_classes(CONFIG)

    assert apply_lfm3_mlx_lm_patch()

    model_cls, args_cls = _get_classes(CONFIG)
    assert model_cls is lfm3_mlx_lm.model.Model
    assert args_cls is lfm3_mlx_lm.model.ModelArgs
    assert importlib.import_module("mlx_lm.models.lfm3") is lfm3_mlx_lm.model

    # The loader imports by name; the package attribute must agree.
    from mlx_lm.models import lfm3

    assert lfm3 is lfm3_mlx_lm.model


def test_applying_twice_is_a_no_op():
    assert apply_lfm3_mlx_lm_patch()
    assert not apply_lfm3_mlx_lm_patch()
    assert lfm3_mlx_lm.is_applied()


def test_a_tiny_checkpoint_runs_a_forward_pass():
    assert apply_lfm3_mlx_lm_patch()
    model_cls = lfm3_mlx_lm.model.Model
    args_cls = lfm3_mlx_lm.model.ModelArgs
    model = model_cls(args_cls(**ARGS))
    tokens = mx.array([[1, 2, 3]])
    logits = model(tokens, cache=model.make_cache())
    assert logits.shape == (1, 3, ARGS["vocab_size"])


def test_cache_kind_follows_the_layer_type():
    from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

    model = lfm3_mlx_lm.model.Model(lfm3_mlx_lm.model.ModelArgs(**ARGS))
    caches = model.make_cache()
    assert len(caches) == ARGS["num_hidden_layers"]
    assert isinstance(caches[0], ArraysCache)
    assert isinstance(caches[2], KVCache)
    assert isinstance(caches[3], RotatingKVCache)
    assert caches[3].max_size == ARGS["sliding_window"]


def test_sanitize_strips_the_vl_prefix_and_fixes_conv_layout():
    model = lfm3_mlx_lm.model.Model(lfm3_mlx_lm.model.ModelArgs(**ARGS))
    # Converted checkpoints keep the VL naming; MLX Conv1d wants (out, L, in).
    conv = mx.zeros((32, 1, 3))
    out = model.sanitize(
        {
            "language_model.model.layers.0.conv.conv.weight": conv,
            "language_model.model.embed_tokens.weight": mx.zeros((256, 32)),
        }
    )
    assert set(out) == {
        "model.layers.0.conv.conv.weight",
        "model.embed_tokens.weight",
    }
    assert out["model.layers.0.conv.conv.weight"].shape == (32, 3, 1)
