"""K2 uses standard oQ levels and preserves output-head calibration."""

import json

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from test_k2_horizon import small_config

from omlx.oq import (
    OQ_LEVELS,
    OQImatrixCollector,
    _collect_k2_horizon_lm_head_imatrix,
    quantize_oq_streaming,
    universal_quant_predicate,
)
from omlx.patches.k2_horizon import apply_k2_horizon_patch
from omlx.patches.k2_horizon.k2_horizon_model import Model, ModelArgs


@pytest.mark.parametrize("level", sorted(OQ_LEVELS))
@pytest.mark.parametrize("budgeted", [False, True])
def test_k2_uses_standard_oq_policy(level, budgeted):
    config = small_config(num_hidden_layers=16, num_experts=4)
    if budgeted:
        config.update(
            _oq_use_budget_plan=True,
            _oq_boost_map={
                "model.layers.8.mlp.experts.down_proj": {
                    "bits": 6,
                    "group_size": 64,
                    "mode": "affine",
                }
            },
        )
    generic = {**config, "model_type": "generic"}
    for path in (
        "model.embed_tokens",
        "lm_head",
        "model.layers.8.self_attn.q_proj",
        "model.layers.8.self_attn.k_proj",
        "model.layers.8.self_attn.v_proj",
        "model.layers.8.self_attn.o_proj",
        "model.layers.8.self_attn.gate_proj",
        "model.layers.8.self_attn.v_experts",
        "model.layers.8.self_attn.v_router",
        "model.layers.8.mlp.gate_proj",
        "model.layers.8.mlp.up_proj",
        "model.layers.8.mlp.down_proj",
        "model.layers.8.mlp.gate",
        "model.layers.8.mlp.experts.gate_proj",
        "model.layers.8.mlp.experts.up_proj",
        "model.layers.8.mlp.experts.down_proj",
        "model.layers.8.mlp.shared_experts.down_proj",
    ):
        assert universal_quant_predicate(path, None, config, level) == (
            universal_quant_predicate(path, None, generic, level)
        ), path


@pytest.fixture(params=["dense", "moe", "mova"])
def k2_checkpoint(tmp_path, request):
    from mlx.utils import tree_flatten

    config = small_config(num_hidden_layers=4, mlp_only_layers=[0])
    if request.param != "dense":
        config.update(
            num_experts=4,
            num_experts_per_tok=2,
            num_shared_experts=1,
            moe_intermediate_size=64,
        )
    if request.param == "mova":
        config.update(
            mova_num_experts=4,
            mova_num_experts_per_tok=2,
            attention_gate_func="softplus",
        )
    mx.random.seed(320)
    model = Model(ModelArgs.from_dict(config))
    model.set_dtype(mx.bfloat16)
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(
        str(source / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    return source


@pytest.mark.parametrize("level", sorted(OQ_LEVELS))
def test_every_oq_level_converts_and_reloads_k2(k2_checkpoint, tmp_path, level):
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load_model

    apply_k2_horizon_patch()
    source = k2_checkpoint
    output = tmp_path / f"oQ{level}"
    quantize_oq_streaming(
        str(source),
        str(output),
        level,
        enhanced=False,
        sensitivity_map_override={i: i + 1.0 for i in range(4)},
    )
    model, _ = load_model(output)
    bits = [
        module.bits for _, module in model.named_modules() if hasattr(module, "bits")
    ]
    assert bits
    assert min(bits) < 8 if level < 8 else set(bits) == {8}
    for layer in model.layers:
        if hasattr(layer.mlp, "gate"):
            assert isinstance(layer.mlp.gate, nn.Linear)
        if hasattr(layer.self_attn, "v_router"):
            assert isinstance(layer.self_attn.v_router, nn.Linear)
    inputs = mx.array([[1, 2, 3, 4]])
    full = model(inputs)
    cache = make_prompt_cache(model)
    prefix = model(inputs[:, :3], cache=cache)
    mx.eval(prefix)
    tail = model(inputs[:, 3:], cache=cache)
    assert bool(mx.all(mx.isfinite(full)) & mx.all(mx.isfinite(tail)))
    assert bool(mx.allclose(full[:, -1].astype(mx.float32), tail[:, -1], atol=0.15))


@pytest.mark.parametrize("head", ["untied", "tied", "missing"])
@pytest.mark.parametrize("four_dimensional", [False, True])
def test_head_calibration_without_logits(monkeypatch, head, four_dimensional):
    mx.random.seed(212)
    model = Model(ModelArgs.from_dict(small_config(tie_word_embeddings=head == "tied")))
    if head == "missing":
        del model.lm_head
    hidden = mx.random.normal((1, 2, 3, 64) if four_dimensional else (1, 2, 64))
    expected = model.model.norm(hidden.mean(axis=2) if four_dimensional else hidden)
    mx.eval(expected)

    def no_logits(*args, **kwargs):
        raise AssertionError("Calibration must not invoke a vocabulary projection")

    monkeypatch.setattr(nn.Linear, "__call__", no_logits)
    collector = OQImatrixCollector()
    collector.install(model)
    try:
        assert _collect_k2_horizon_lm_head_imatrix(model, hidden, collector) == (
            head == "untied"
        )
        if head == "untied":
            entry = collector.entries["lm_head"]
            np.testing.assert_allclose(
                entry.in_sum2,
                np.square(np.asarray(expected)).sum(axis=(0, 1)),
                rtol=1e-6,
            )
            assert entry.counts.tolist() == [2]
        else:
            assert collector.entries == {}
    finally:
        collector.restore(model)
