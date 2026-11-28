# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 Pro (``kimi_k31``) on the pinned mlx-lm.

Kimi K3 with sparse attention on its full-attention layers. Layers 4 and 6
(1-based) are sparse here and select 8 keys, so short prompts exercise it.
"""

from __future__ import annotations

import json

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.models import kimi_k3
from mlx_lm.models.base import create_causal_mask

from omlx.cluster.pipeline_compat import pipeline_assignment_is_honored
from omlx.cluster.planner import _supports_pipeline, _supports_tensor_parallel
from omlx.patches.kimi_k31 import model as kimi_k31
from omlx.utils.model_loading import maybe_apply_pre_load_patches
from tests.test_deepseek_v41 import run_two_ranks

DSA = {
    "dsa_layers": [4, 6],
    "index_topk": 8,
    "index_head_dim": 16,
    "index_num_heads": 4,
    "index_rope_dim": 0,
    "index_k_norm_type": "layernorm",
    "index_topk_freq": 1,
}
TEXT = {
    "model_type": "kimi_linear",
    "vocab_size": 256,
    "hidden_size": 64,
    "num_hidden_layers": 6,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "intermediate_size": 96,
    "rms_norm_eps": 1e-5,
    "activation_situ_beta": 4.0,
    "activation_situ_linear_beta": 25.0,
    "linear_attn_config": {
        "kda_layers": [1, 2, 3, 5],
        "full_attn_layers": [4, 6],
        "num_heads": 2,
        "head_dim": 32,
        "short_conv_kernel_size": 4,
        "gate_lower_bound": -5.0,
        "use_full_rank_gate": True,
    },
    "num_experts": 8,
    "moe_intermediate_size": 32,
    "q_lora_rank": 32,
    "kv_lora_rank": 32,
    "qk_nope_head_dim": 16,
    "qk_rope_head_dim": 8,
    "v_head_dim": 16,
    "mla_use_nope": True,
    "mla_use_output_gate": True,
    "num_experts_per_token": 2,
    "num_shared_experts": 1,
    "first_k_dense_replace": 1,
    "routed_expert_hidden_size": 32,
    "latent_moe_use_norm": True,
    "attn_res_block_size": 2,
}
CONFIG = {"model_type": "kimi_k31", "text_config": {**TEXT, "dsa_config": DSA}}


@pytest.fixture(autouse=True)
def _cpu():
    # Tiny random models: GPU fp32 rounding is larger than the tolerances.
    device = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.random.seed(0)
    yield
    mx.set_default_device(device)


def _model(**text):
    config = {**CONFIG, "text_config": {**CONFIG["text_config"], **text}}
    model = kimi_k31.Model(kimi_k31.ModelArgs.from_dict(config))
    mx.eval(model.parameters())
    return model


def _run(model, tokens, chunk):
    cache = model.make_cache()
    outs = [
        model(tokens[:, start : start + chunk], cache=cache)
        for start in range(0, tokens.shape[1], chunk)
    ]
    return mx.concatenate(outs, axis=1)


def test_a_context_within_the_key_budget_gives_kimi_k3_logits():
    model = _model()
    k3 = kimi_k3.Model(
        kimi_k3.ModelArgs.from_dict({"model_type": "kimi_k3", "text_config": TEXT})
    )
    shared = [
        (name, value)
        for name, value in tree_flatten(model.parameters())
        if ".indexer." not in name
    ]
    k3.update(tree_unflatten(shared))
    tokens = mx.random.randint(0, 256, (1, DSA["index_topk"]))

    assert [index for index, layer in enumerate(model.layers) if layer.is_dsa] == [3, 5]
    assert mx.array_equal(_run(model, tokens, 4), _run(k3, tokens, 4)).item()


def test_selected_keys_give_the_same_logits_in_chunks_and_in_decode():
    model = _model()
    tokens = mx.random.randint(0, 256, (1, 48))
    whole = _run(model, tokens, 48)
    every_key = _run(_model_with(model, index_topk=4096), tokens, 48)

    assert mx.abs(whole - every_key).max().item() > 1e-2
    assert mx.allclose(_run(model, tokens, 16), whole, atol=1e-4, rtol=1e-4).item()
    prefill_then_decode = mx.concatenate(
        [_run(model, tokens[:, :24], 24), _run_decode(model, tokens, 24)], axis=1
    )
    assert mx.allclose(prefill_then_decode, whole, atol=1e-4, rtol=1e-4).item()


def _model_with(model, **dsa):
    other = _model(dsa_config={**DSA, **dsa})
    other.update(model.parameters())
    return other


def _run_decode(model, tokens, start):
    cache = model.make_cache()
    model(tokens[:, :start], cache=cache)
    steps = [
        model(tokens[:, index : index + 1], cache=cache)
        for index in range(start, tokens.shape[1])
    ]
    return mx.concatenate(steps, axis=1)


def test_the_indexer_scores_keys_without_rope():
    indexer = _model().layers[3].self_attn.indexer
    rows, topk = 32, indexer.index_topk
    x = mx.random.normal((1, rows, 64))
    qr = mx.random.normal((1, rows, 32))
    mask = create_causal_mask(rows)

    q = indexer.wq_b(qr).reshape(rows, indexer.n_heads, indexer.head_dim)
    k = indexer.k_norm(indexer.wk(x))[0]
    weights = indexer.weights_proj(x)[0]
    scores = (mx.maximum(mx.einsum("shd,td->sht", q, k), 0) * weights[..., None]).sum(1)
    scores = mx.where(mask, scores, -mx.inf)
    expected = mx.sort(mx.argpartition(scores, kth=-topk, axis=-1)[:, -topk:], -1)

    selected = mx.sort(indexer(x, qr, mask)[0, 0], -1)
    # A row with fewer causal keys than the budget also selects masked keys.
    assert mx.array_equal(selected[topk - 1 :], expected[topk - 1 :]).item()


@pytest.mark.parametrize(
    "text",
    [
        {"dsa_config": {**DSA, "index_rope_dim": 64}},
        {"dsa_config": {**DSA, "index_topk_freq": 4}},
        {"dsa_config": {**DSA, "index_k_norm_type": "rmsnorm"}},
        {"dsa_config": {**DSA, "dsa_layers": [3, 6]}},
        {"dsa_config": None},
        {"q_lora_rank": None},
    ],
)
def test_unsupported_sparse_attention_settings_are_rejected(text):
    with pytest.raises(ValueError):
        _model(**text)


def test_a_checkpoint_with_a_vision_tower_is_planned_as_a_text_model(tmp_path):
    config = {**CONFIG, "vision_config": {"hidden_size": 8}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    maybe_apply_pre_load_patches(str(tmp_path))

    assert pipeline_assignment_is_honored(tmp_path)
    assert _supports_pipeline(config)
    assert _supports_tensor_parallel(config)


# One rank of the test below. Rank 1 ends on a sparse layer, so the stage
# hand-off goes through that layer's cache.
_RANK = """
import json
import sys
import mlx.core as mx
group = mx.distributed.init(backend="ring", strict=True)
mx.set_default_device(mx.cpu)
from omlx.cluster.planner import PipelineAssignment, install_unequal_pipeline_plan
from omlx.cluster.tensor_strategies import apply_tensor_strategy
from omlx.patches.kimi_k31 import apply_kimi_k31_patch
assert apply_kimi_k31_patch()
from mlx_lm.models import kimi_k31

args = kimi_k31.ModelArgs.from_dict(json.loads(sys.argv[1]))
mx.random.seed(7)
prompt = mx.random.randint(0, 256, (1, 28))
steps = [prompt[:, :12], prompt[:, 12:24]]
steps += [prompt[:, index : index + 1] for index in range(24, 28)]

def build():
    mx.random.seed(11)
    model = kimi_k31.Model(args)
    mx.eval(model.parameters())
    return model

def run(model):
    cache = model.make_cache()
    outs = []
    for chunk in steps:
        outs.append(model(chunk, cache=cache))
        mx.eval(outs[-1])
    return outs

reference = run(build())
plan = (
    PipelineAssignment("a", 0, 4, 6, 1, 0, 0, 1),
    PipelineAssignment("b", 1, 0, 4, 1, 0, 0, 1),
)
pipelined = build()
with install_unequal_pipeline_plan(plan):
    pipelined.model.pipeline(group)
sharded = build()
assert apply_tensor_strategy(sharded, group, mx_module=mx) == "kimi_k31"
for model in (pipelined, sharded):
    for expected, got in zip(reference, run(model)):
        assert mx.allclose(expected, got, atol=1e-4, rtol=1e-4).item()
print(json.dumps({"rank": group.rank(), "ok": True}), flush=True)
"""


@pytest.mark.slow
def test_pipeline_stages_and_tensor_ranks_match_one_process(tmp_path):
    run_two_ranks(tmp_path, _RANK, json.dumps(CONFIG))
