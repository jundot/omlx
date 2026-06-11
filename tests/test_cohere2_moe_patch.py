# SPDX-License-Identifier: Apache-2.0
"""Tests for the Cohere2 MoE mlx-lm compatibility patch."""

import importlib
import sys
from unittest.mock import MagicMock

import mlx.core as mx
import mlx.nn as nn

from omlx.utils import model_loading
from omlx.utils.model_loading import maybe_apply_pre_load_patches


def _tiny_args_config(**overrides):
    cfg = {
        "model_type": "cohere2_moe",
        "hidden_size": 16,
        "head_dim": 4,
        "num_hidden_layers": 2,
        "intermediate_size": 8,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "num_shared_experts": 0,
        "vocab_size": 32,
        "sliding_window": 8,
        "sliding_window_pattern": 2,
        "first_k_dense_replace": 1,
        "prefix_dense_intermediate_size": 12,
        "layer_types": ["full_attention", "sliding_attention"],
        "norm_topk_prob": False,
    }
    cfg.update(overrides)
    return cfg


def test_pre_load_dispatch_applies_cohere2_moe_patch(tmp_path, monkeypatch):
    monkeypatch.setattr(model_loading, "_patch_mlx_lm_load_config", lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "omlx.patches.mlx_lm_mtp",
        MagicMock(set_mtp_active=MagicMock()),
    )
    apply_mock = MagicMock(return_value=True)
    monkeypatch.setitem(
        sys.modules,
        "omlx.patches.cohere2_moe",
        MagicMock(apply_cohere2_moe_patch=apply_mock),
    )
    (tmp_path / "config.json").write_text('{"model_type": "cohere2_moe"}')

    maybe_apply_pre_load_patches(str(tmp_path))

    apply_mock.assert_called_once_with()


def test_apply_registers_cohere2_moe_module():
    from omlx.patches.cohere2_moe import apply_cohere2_moe_patch, is_applied

    first = apply_cohere2_moe_patch()
    second = apply_cohere2_moe_patch()

    assert is_applied() is True
    assert first in (True, False)
    assert second is False
    assert "mlx_lm.models.cohere2_moe" in sys.modules

    import mlx_lm.models as models_pkg

    module = importlib.import_module("mlx_lm.models.cohere2_moe")
    assert models_pkg.cohere2_moe is module
    assert module.Model.__name__ == "Model"
    assert module.ModelArgs.__name__ == "ModelArgs"


def test_get_classes_resolves_cohere2_moe_and_vision_remap():
    from mlx_lm.utils import _get_classes

    from omlx.patches.cohere2_moe import apply_cohere2_moe_patch

    apply_cohere2_moe_patch()

    model_cls, args_cls = _get_classes({"model_type": "cohere2_moe"})
    assert model_cls.__name__ == "Model"
    assert args_cls.__name__ == "ModelArgs"

    model_cls, args_cls = _get_classes({"model_type": "cohere2_vision"})
    assert model_cls.__name__ == "Model"
    assert args_cls.__name__ == "ModelArgs"


def test_tiny_model_forward_and_cache_types():
    from mlx_lm.models.cache import KVCache, RotatingKVCache

    from omlx.patches.cohere2_moe import apply_cohere2_moe_patch

    apply_cohere2_moe_patch()
    from mlx_lm.models import cohere2_moe

    args = cohere2_moe.ModelArgs(**_tiny_args_config())
    model = cohere2_moe.Model(args)

    logits = model(mx.array([[1, 2, 3]], dtype=mx.int32))
    mx.eval(logits)

    assert logits.shape == (1, 3, 32)
    cache = model.make_cache()
    assert isinstance(cache[0], KVCache)
    assert isinstance(cache[1], RotatingKVCache)
    assert model.layers is model.model.layers


def test_tiny_model_forward_with_attention_width_larger_than_hidden_size():
    from omlx.patches.cohere2_moe import apply_cohere2_moe_patch

    apply_cohere2_moe_patch()
    from mlx_lm.models import cohere2_moe

    args = cohere2_moe.ModelArgs(
        **_tiny_args_config(
            hidden_size=8,
            head_dim=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=8,
        )
    )
    model = cohere2_moe.Model(args)

    logits = model(mx.array([[1, 2, 3]], dtype=mx.int32))
    mx.eval(logits)

    assert logits.shape == (1, 3, 32)
    assert model.model.layers[0].self_attn.q_proj.weight.shape == (16, 8)
    assert model.model.layers[0].self_attn.o_proj.weight.shape == (8, 16)


def test_rms_norm_and_prefix_dense_force_rope_config():
    from omlx.patches.cohere2_moe import apply_cohere2_moe_patch

    apply_cohere2_moe_patch()
    from mlx_lm.models import cohere2_moe

    args = cohere2_moe.ModelArgs(
        **_tiny_args_config(
            rms_norm_eps=1e-6,
            layer_norm_bias=False,
            prefix_dense_sliding_window_pattern=1,
            mlp_layer_types=["dense", "sparse"],
            layer_types=["full_attention", "sliding_attention"],
        )
    )
    model = cohere2_moe.Model(args)

    assert isinstance(model.model.norm, nn.RMSNorm)
    assert isinstance(model.model.layers[0].input_layernorm, nn.RMSNorm)
    assert model.model.layers[0].self_attn.use_sliding_window is False
    assert model.model.layers[0].self_attn.force_rope is True
    assert model.model.layers[1].self_attn.use_sliding_window is True
    assert model.model.layers[1].self_attn.force_rope is False


def test_sanitize_accepts_north_prefixed_switch_mlp_weights():
    from omlx.patches.cohere2_moe import apply_cohere2_moe_patch

    apply_cohere2_moe_patch()
    from mlx_lm.models import cohere2_moe

    args = cohere2_moe.ModelArgs(**_tiny_args_config())
    model = cohere2_moe.Model(args)

    weights = {
        "vision_tower.patch_embed.weight": mx.zeros((1,)),
        "language_model.model.embed_tokens.weight": mx.zeros((32, 16)),
        "language_model.model.layers.1.mlp.switch_mlp.up_proj.weight": mx.zeros(
            (2, 8, 16)
        ),
        "language_model.model.layers.1.mlp.switch_mlp.gate_proj.weight": mx.zeros(
            (2, 8, 16)
        ),
        "language_model.model.layers.1.mlp.switch_mlp.down_proj.weight": mx.zeros(
            (2, 16, 8)
        ),
        "lm_head.weight": mx.zeros((32, 16)),
    }

    out = model.sanitize(weights)

    assert "vision_tower.patch_embed.weight" not in out
    assert "lm_head.weight" not in out
    assert "model.model.embed_tokens.weight" not in out
    assert "model.embed_tokens.weight" in out
    assert "model.layers.1.mlp.switch_mlp.up_proj.weight" in out


def test_sanitize_stacks_raw_expert_weights_for_hf_layout():
    from omlx.patches.cohere2_moe import apply_cohere2_moe_patch

    apply_cohere2_moe_patch()
    from mlx_lm.models import cohere2_moe

    args = cohere2_moe.ModelArgs(**_tiny_args_config())
    model = cohere2_moe.Model(args)
    weights = {}
    for expert in range(args.num_experts):
        weights[f"model.layers.1.mlp.experts.{expert}.up_proj.weight"] = mx.full(
            (8, 16), expert + 1
        )
        weights[f"model.layers.1.mlp.experts.{expert}.gate_proj.weight"] = mx.full(
            (8, 16), expert + 2
        )
        weights[f"model.layers.1.mlp.experts.{expert}.down_proj.weight"] = mx.full(
            (16, 8), expert + 3
        )

    out = model.sanitize(weights)

    up = out["model.layers.1.mlp.switch_mlp.up_proj.weight"]
    gate = out["model.layers.1.mlp.switch_mlp.gate_proj.weight"]
    down = out["model.layers.1.mlp.switch_mlp.down_proj.weight"]
    assert up.shape == (2, 8, 16)
    assert gate.shape == (2, 8, 16)
    assert down.shape == (2, 16, 8)
    assert not any(".experts." in key for key in out)
