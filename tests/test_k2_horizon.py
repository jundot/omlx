# SPDX-License-Identifier: Apache-2.0
"""K2 model arithmetic, quantization, and cache checks."""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.cache import make_prompt_cache

from omlx.patches.k2_horizon import apply_k2_horizon_patch
from omlx.patches.k2_horizon.k2_horizon_model import GroupedRMSNorm, Model, ModelArgs


def small_config(**overrides):
    return dict(
        dict(
            model_type="k2_horizon",
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=128,
            rms_norm_eps=1e-5,
            layernorm_num_groups=2,
            mlp_only_layers=[],
            num_experts=0,
            num_experts_per_tok=0,
            moe_intermediate_size=0,
            num_shared_experts=0,
            moe_gate_bias=True,
            norm_topk_prob=True,
            router_score_func="sigmoid",
            router_scaling_factor=1,
            query_key_norm=False,
            rope_parameters={"rope_type": "default", "rope_theta": 10000},
        ),
        **overrides,
    )


@pytest.mark.parametrize("kind", ["dense", "moe", "mova", "yarn", "partial"])
def test_model_cache_and_quantization(kind):
    apply_k2_horizon_patch()
    config = small_config()
    if kind in ("moe", "mova"):
        config.update(
            num_experts=4,
            num_experts_per_tok=2,
            num_shared_experts=1,
            moe_intermediate_size=64,
        )
    if kind == "mova":
        config.update(
            mova_num_experts=4,
            mova_num_experts_per_tok=2,
            attention_gate_func="softplus",
        )
    if kind == "yarn":
        config["rope_parameters"].update(
            rope_type="yarn",
            factor=4,
            original_max_position_embeddings=128,
            beta_fast=32,
            beta_slow=1,
            attention_factor=1,
        )
    if kind == "partial":
        config["rope_head_dim"] = 8
    model = Model(ModelArgs.from_dict(config))
    model.set_dtype(mx.bfloat16)
    ids = mx.array([[1, 2, 3, 4]])
    full = model(ids)
    cache = make_prompt_cache(model)
    model(ids[:, :3], cache=cache)
    tail = model(ids[:, 3:], cache=cache)
    assert mx.allclose(full[:, -1].astype(mx.float32), tail[:, -1], atol=0.04).item()
    from mlx_lm.utils import quantize_model

    model, _ = quantize_model(model, config, group_size=32, bits=4)
    assert mx.all(mx.isfinite(model(ids))).item()
    if kind in ("moe", "mova"):
        assert not isinstance(model.layers[0].mlp.gate, nn.QuantizedLinear)


@pytest.mark.parametrize("groups", [1, 2, 4])
@pytest.mark.parametrize("length", [1, 8, 512])
def test_grouped_norm_and_oq_protection(groups, length):
    from omlx.oq import universal_quant_predicate

    x = mx.arange(length * 64).reshape(1, length, 64).astype(mx.bfloat16)
    actual = GroupedRMSNorm(64, groups, 1e-5)(x)
    grouped = x.astype(mx.float32).reshape(1, length, groups, 64 // groups)
    expected = (
        grouped * mx.rsqrt(mx.mean(grouped**2, -1, keepdims=True) + 1e-5)
    ).reshape(x.shape)
    assert mx.allclose(actual.astype(mx.float32), expected, atol=0.01).item()
    config = {"model_type": "k2_horizon"}
    layer = nn.Linear(64, 64)
    assert (
        universal_quant_predicate("model.layers.0.self_attn.v_router", layer, config)
        is False
    )
    assert (
        universal_quant_predicate("model.layers.0.self_attn.q_proj", layer, config)[
            "bits"
        ]
        == 8
    )


@pytest.mark.parametrize("quantized", [False, True])
def test_indexed_checkpoint_roundtrip(tmp_path, quantized):
    import json

    from mlx.utils import tree_flatten
    from mlx_lm import utils

    apply_k2_horizon_patch()
    config = small_config()
    model = Model(ModelArgs.from_dict(config))
    if quantized:
        model, config = utils.quantize_model(model, config, group_size=32, bits=4)
    weights = dict(tree_flatten(model.parameters()))
    shard = "pytorch_model-00001-of-00001.safetensors"
    mx.save_safetensors(str(tmp_path / shard), weights)
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: shard for key in weights}})
    )
    restored, _ = utils.load_model(tmp_path)
    ids = mx.array([[1, 2, 3]])
    assert mx.array_equal(model(ids), restored(ids)).item()
    (tmp_path / shard).unlink()
    with pytest.raises(FileNotFoundError, match="Missing K2 checkpoint shard"):
        utils.load_model(tmp_path)


def test_mova_router_preserves_source_partition_rounding():
    from omlx.patches.k2_horizon.k2_horizon_model import router_logits

    x = mx.ones((1, 4), mx.bfloat16)
    weights = mx.array([[1, 1 / 256, -1, 0], [0, 0, 1 / 512, 0]], mx.bfloat16)
    partial = router_logits(x, weights, partitions=2)
    full = router_logits(x, weights, partitions=1)
    assert partial.tolist() == [[0, 1 / 512]]
    assert full.tolist() == [[1 / 256, 1 / 512]]
    assert mx.argmax(partial).item() == 1
    assert mx.argmax(full).item() == 0


def test_yarn_rotation_uses_each_batch_offset():
    from omlx.patches.k2_horizon.k2_horizon_model import YarnRoPE

    rope = YarnRoPE(
        SimpleNamespace(
            rope_head_dim=16,
            rope_theta=10000,
            rope_parameters=dict(
                attention_factor=1.0,
                original_max_position_embeddings=128,
                beta_fast=32,
                beta_slow=1,
                factor=4,
            ),
        )
    )
    x = mx.arange(2 * 4 * 8 * 16).reshape(2, 4, 8, 16).astype(mx.bfloat16)
    for length in (1, 8):
        values = x[:, :, :length]
        actual = rope(values, offset=mx.array([4096, 8192]))
        expected = mx.concatenate(
            [
                rope(values[i : i + 1], offset=offset)
                for i, offset in enumerate((4096, 8192))
            ]
        )
        assert mx.array_equal(actual, expected).item()


def test_router_bias_only_changes_selection():
    from omlx.patches.k2_horizon.k2_horizon_model import route

    x = mx.ones((2, 1, 4), mx.bfloat16)
    weight = mx.zeros((3, 4), mx.bfloat16)
    bias = mx.array([0.0, 0.1, 0.2])
    for top_k, scale in [(1, 1.0), (2, 2.5), (1, 3.0)]:
        indices, weights = route(x, weight, bias, top_k, scale)
        expected = mx.broadcast_to(mx.arange(3 - top_k, 3), indices.shape)
        assert mx.array_equal(mx.sort(indices), expected).item()
        assert mx.all(weights == scale / top_k).item()
