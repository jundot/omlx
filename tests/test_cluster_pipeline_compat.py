# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from omlx.cluster.pipeline_compat import (
    install_pipeline_compatibility,
    pipeline_assignment_is_honored,
)
from omlx.cluster.planner import PipelineAssignment
from tests.test_deepseek_v41 import run_two_ranks


def _assignment():
    return (
        PipelineAssignment(
            node_id="local",
            rank=0,
            start_layer=0,
            end_layer=2,
            layer_weight_bytes=2,
            fixed_weight_bytes=1,
            reserve_bytes=1,
            capacity_bytes=8,
        ),
    )


def _model_config(tmp_path, model_type):
    model = tmp_path / model_type
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": model_type}))
    return model


def test_standard_pipeline_mixin_has_an_explicit_assignment_contract(tmp_path):
    model = _model_config(tmp_path, "deepseek_v3")

    assert not pipeline_assignment_is_honored(model)
    with install_pipeline_compatibility(_assignment()):
        assert pipeline_assignment_is_honored(model)


def test_thin_qwen_moe_wrapper_inherits_the_pipeline_contract(tmp_path):
    model = _model_config(tmp_path, "qwen3_5_moe")

    assert not pipeline_assignment_is_honored(model)
    with install_pipeline_compatibility(_assignment()):
        assert pipeline_assignment_is_honored(model)


def test_nemotron_compatibility_has_an_explicit_assignment_contract(tmp_path):
    model = _model_config(tmp_path, "nemotron_h")

    assert not pipeline_assignment_is_honored(model)
    with install_pipeline_compatibility(_assignment()):
        assert pipeline_assignment_is_honored(model)


def test_deepseek_v32_inherits_the_pipeline_contract(tmp_path):
    model = _model_config(tmp_path, "deepseek_v32")

    with install_pipeline_compatibility(_assignment()):
        assert pipeline_assignment_is_honored(model)


def test_minimax_declares_its_wrapped_assigned_stage_contract(tmp_path):
    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    model = _model_config(tmp_path, "minimax_m3_vl")
    maybe_apply_pre_load_patches(str(model))

    assert pipeline_assignment_is_honored(model)


@pytest.mark.parametrize("model_type", ["glm5_next", "deepseek_v41"])
def test_cluster_adapters_inherit_the_pipeline_contract(tmp_path, model_type):
    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    model = _model_config(tmp_path, model_type)
    maybe_apply_pre_load_patches(str(model))

    assert not pipeline_assignment_is_honored(model)
    with install_pipeline_compatibility(_assignment()):
        assert pipeline_assignment_is_honored(model)


_GLM5_NEXT_TEXT = dict(
    model_type="glm5_next_text",
    hidden_size=256,
    intermediate_size=256,
    moe_intermediate_size=128,
    num_hidden_layers=5,
    num_attention_heads=4,
    num_key_value_heads=4,
    vocab_size=1024,
    layer_types=["linear_attention"] * 3
    + ["deepseek_sparse_attention", "linear_attention"],
    mlp_layer_types=["dense"] * 3 + ["sparse", "sparse"],
    indexer_types=["full"] * 5,
    first_k_dense_replace=3,
    n_routed_experts=8,
    n_shared_experts=1,
    num_experts_per_tok=4,
    kv_lora_rank=64,
    q_lora_rank=128,
    qk_head_dim=256,
    qk_nope_head_dim=64,
    qk_rope_head_dim=0,
    v_head_dim=64,
    index_head_dim=64,
    index_n_heads=4,
    index_topk=64,
    index_kpool=4,
    linear_num_heads=4,
    linear_head_dim=64,
    linear_conv_kernel_dim=4,
    hc_mult=4,
    num_nextn_predict_layers=0,
    routed_scaling_factor=2.5,
    max_position_embeddings=4096,
    rms_norm_eps=1e-5,
    linear_attn_config=dict(
        gate_lower_bound=-5.0, head_dim=64, num_heads=4, short_conv_kernel_size=4
    ),
)


def test_glm5_next_sanitize_maps_hf_and_mlx_names_to_the_text_model():
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from omlx.patches.glm5_next_mlx_lm import apply_glm5_next_mlx_lm_patch

    assert apply_glm5_next_mlx_lm_patch()
    from mlx_lm.models import glm5_next

    model = glm5_next.Model(
        glm5_next.ModelArgs.from_dict(
            {"model_type": "glm5_next", "text_config": _GLM5_NEXT_TEXT}
        )
    )
    norm, head, vision = mx.ones((256,)), mx.ones((1024, 256)), mx.ones((4,))
    hf = model.sanitize(
        {
            "model.language_model.norm.weight": norm,
            "lm_head.weight": head,
            "model.visual.patch_embed.proj.weight": vision,
        }
    )
    converted = model.sanitize(
        {
            "language_model.model.norm.weight": norm,
            "language_model.lm_head.weight": head,
            "vision_model.patch_embed.proj.weight": vision,
        }
    )

    # The rank loads with strict=False, so a wrong name would stay random.
    expected = {"language_model.model.norm.weight", "language_model.lm_head.weight"}
    assert set(hf) == set(converted) == expected
    assert expected <= {name for name, _ in tree_flatten(model.parameters())}


# One rank of the GLM-5.3 test below: a tiny random model against the vendored
# single-node forward, as two pipeline stages and as two tensor-parallel ranks.
_GLM5_NEXT_RANK = """
import json
import sys
import mlx.core as mx
import mlx.nn as nn
group = mx.distributed.init(backend="ring", strict=True)
from omlx.cluster.planner import PipelineAssignment, install_unequal_pipeline_plan
from omlx.cluster.tensor_strategies import apply_tensor_strategy
from omlx.patches.glm5_next_mlx_lm import apply_glm5_next_mlx_lm_patch
assert apply_glm5_next_mlx_lm_patch()
from mlx_lm.models import glm5_next
from mlx_vlm.models.glm5_next import language

text = json.loads(sys.argv[1])
args = glm5_next.ModelArgs.from_dict({"model_type": "glm5_next", "text_config": text})
mx.random.seed(7)
prompt = mx.random.randint(0, 1024, (1, 265))
# LayerPipeline prefill, a short block, then two deferred one-token decodes.
steps = [prompt[:, :260], prompt[:, 260:263], prompt[:, 263:264], prompt[:, 264:]]

def build(embed_dtype=mx.float32, bits=None):
    mx.random.seed(11)
    model = glm5_next.Model(args)
    embed = model.language_model.model.embed_tokens
    embed.weight = embed.weight.astype(embed_dtype)
    if bits:
        nn.quantize(
            model,
            64,
            bits,
            class_predicate=lambda path, module: hasattr(module, "to_quantized")
            and module.weight.shape[-1] % 64 == 0,
        )
    mx.eval(model.parameters())
    return model

def run(forward, make_cache):
    # One evaluation per step, as in serving, keeps collectives in order.
    cache = make_cache()
    outs = []
    for chunk in steps:
        outs.append(forward(chunk, cache))
        mx.eval(outs[-1])
    return outs

def vendored(model):
    lm = model.language_model
    lm.model.__class__ = language.Glm5NextModel
    outs = run(lambda x, c: lm(x, cache=c).logits, lm.make_cache)
    lm.model.__class__ = glm5_next.PipelinedTextModel
    return outs

# Rank 1 [0, 3) -> rank 0 [3, 5). The layers return fp32 while recv_like
# takes the embedding dtype, so a bf16 embedding is checked too.
plan = (
    PipelineAssignment("a", 0, 3, 5, 1, 0, 0, 1),
    PipelineAssignment("b", 1, 0, 3, 1, 0, 0, 1),
)
for embed_dtype in (mx.float32, mx.bfloat16):
    model = build(embed_dtype)
    reference = vendored(model)
    with install_unequal_pipeline_plan(plan):
        model.model.pipeline(group)
    got = run(lambda x, c: model(x, cache=c), model.make_cache)
    if embed_dtype == mx.float32:
        assert all(mx.array_equal(r, g).item() for r, g in zip(reference, got))
    assert all(mx.isfinite(g).all().item() for g in got)
    assert all(
        mx.array_equal(mx.argmax(r[:, -1], -1), mx.argmax(g[:, -1], -1)).item()
        for r, g in zip(reference, got)
    )

for bits in (None, 4):
    reference = vendored(build(bits=bits))
    model = build(bits=bits)
    apply_tensor_strategy(model, group, mx_module=mx)
    got = run(lambda x, c: model(x, cache=c), model.make_cache)
    for r, g in zip(reference, got):
        assert mx.allclose(r, g, atol=1e-4, rtol=1e-4).item(), mx.abs(r - g).max()
print(json.dumps({"rank": group.rank(), "ok": True}), flush=True)
"""


def test_glm5_next_cluster_forwards_match_single_process(tmp_path):
    run_two_ranks(tmp_path, _GLM5_NEXT_RANK, json.dumps(_GLM5_NEXT_TEXT))


# One rank of a two-stage GLM-5.3 deployment decoding with Lightning MTP: rank
# zero drafts and decides, rank one replays its decisions. A random head drafts
# nothing acceptable, so the oracle mode drafts the expected tokens with every
# third block corrupted, covering full and partial accepts. Greedy output must
# equal plain single-process decoding on both ranks.
_GLM5_NEXT_MTP_RANK = """
import json
import sys
import mlx.core as mx
group = mx.distributed.init(backend="ring", strict=True)
from mlx_lm.generate import BatchGenerator
from omlx.cluster.control_plane import RankControlPlane
from omlx.cluster.mtp_sync import ClusterMtpSync
from omlx.cluster.performance import execution_profile
from omlx.cluster.planner import PipelineAssignment, install_unequal_pipeline_plan
from omlx.cluster.runtime_optimizations import (
    install_runtime_optimizations,
    wait_for_pipeline_peers_on_host,
)
from omlx.cluster.tensor_strategies import apply_tensor_strategy
from omlx.patches import mlx_lm_mtp
from omlx.patches.glm5_next_mlx_lm import apply_glm5_next_mlx_lm_patch
from omlx.patches.mlx_lm_mtp import batch_generator as bg
from omlx.patches.mlx_lm_mtp import prompt_priming
assert apply_glm5_next_mlx_lm_patch() and mlx_lm_mtp.apply_mlx_lm_mtp_patch()
from mlx_lm.models import glm5_next
from mlx_vlm.models.glm5_next import language

text, port, depth = json.loads(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
oracle, split = sys.argv[4] == "oracle", sys.argv[5]
pipeline = split.startswith("pipeline")
# Without rank-zero sampling every rank gathers rank zero's output.
rank_zero = split == "pipeline"
rank = group.rank()
mlx_lm_mtp.set_mtp_active(True)
mlx_lm_mtp.set_mtp_depth(depth, fixed=True)
args = glm5_next.ModelArgs.from_dict({"model_type": "glm5_next", "text_config": text})

def build():
    mx.random.seed(11)
    model = glm5_next.Model(args)
    mx.eval(model.parameters())
    lm = model.language_model
    assert (getattr(lm, "mtp", None) is not None) == (rank == 0)
    assert lm._omlx_mtp_decode_enabled and lm._omlx_mtp_chain
    return model

mx.random.seed(7)
prompt = mx.random.randint(0, 1024, (40,)).tolist()
limit = 24

# Plain greedy decoding through the vendored single-process forward.
lm = build().language_model
lm.model.__class__ = language.Glm5NextModel
with prompt_priming.suppress_capture():
    cache = lm.make_cache()
    logits = lm(mx.array([prompt]), cache=cache).logits
    reference = []
    for _ in range(limit):
        token = mx.argmax(logits[:, -1], -1)
        reference.append(int(token.item()))
        logits = lm(token[None], cache=cache).logits

# A fresh copy: forwards cache fused projections that sharding would not cut.
model = build()

plan = (
    PipelineAssignment("a", 0, 3, 5, 1, 0, 0, 1),
    PipelineAssignment("b", 1, 0, 3, 1, 0, 0, 1),
)
if pipeline:
    with install_unequal_pipeline_plan(plan):
        model.model.pipeline(group)
else:
    # The head stays whole on rank zero; only the layers are sharded.
    apply_tensor_strategy(model, group, mx_module=mx)

cycles = []
original = bg._run_verify_cycle_chain
def counted(*a, **k):
    cycles.append(1)
    return original(*a, **k)
bg._run_verify_cycle_chain = counted
accepted = []
share_result = ClusterMtpSync.share_cycle_result
def recorded(self, *a):
    result = share_result(self, *a)
    accepted.append(result[0])
    return result
ClusterMtpSync.share_cycle_result = recorded

def oracle_drafts(gen_batch, state, *_args, **_kwargs):
    # state.next_main is the last committed token, so drafts start after it.
    done = len(gen_batch.tokens[0]) - len(prompt) + len(state.queue)
    ids = (reference[done : done + depth] + [0] * depth)[:depth]
    if done % 3 == 0:
        ids[-1] = (ids[-1] + 1) % 1024
    state.drafts = mx.array(ids, dtype=mx.uint32)
    state.draft_lps, state.draft_accept_lps = [], []
if oracle and rank == 0:
    bg._chain_next_drafts = oracle_drafts

# As a serving rank runs: rank zero samples, the others skip the vocabulary.
with (
    RankControlPlane(
        rank=rank, world_size=2, host="127.0.0.1", port=port, token="glm-mtp-test"
    ) as plane,
    install_runtime_optimizations(
        model,
        group,
        execution_profile("balanced", sampling_rank_only=rank_zero),
        batchable=True,
        pipeline_parallel=pipeline,
    ) as optimizations,
    wait_for_pipeline_peers_on_host(model),
):
    assert optimizations["rank_zero_logits"]["active"] == rank_zero
    model._omlx_cluster_mtp = ClusterMtpSync(rank=rank, control_plane=plane)
    gen = BatchGenerator(model, sampler=lambda lp: mx.argmax(lp, -1), max_tokens=limit)
    tokens = []
    try:
        gen.insert([prompt], max_tokens=[limit])
        for _ in range(4 * limit):
            _, responses = gen.next()
            tokens.extend(response.token for response in responses)
            if any(response.finish_reason for response in responses):
                break
    finally:
        gen.close()
assert cycles, "Lightning MTP never ran a verify cycle"
assert tokens == reference, (tokens, reference)
assert len(accepted) == len(cycles)
if oracle:
    assert max(accepted) == depth and 0 < min(accepted) < depth, accepted
print(json.dumps({"rank": rank, "ok": True, "cycles": len(cycles)}), flush=True)
"""


@pytest.mark.parametrize(
    ("depth", "drafts", "split"),
    [
        (3, "head", "pipeline"),
        (3, "oracle", "pipeline"),
        (3, "oracle", "pipeline-gather"),
        (3, "oracle", "tensor"),
    ],
)
def test_glm5_next_cluster_lightning_mtp_matches_greedy_decoding(
    tmp_path, depth, drafts, split
):
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    text = dict(_GLM5_NEXT_TEXT, num_nextn_predict_layers=1)
    run_two_ranks(
        tmp_path, _GLM5_NEXT_MTP_RANK, json.dumps(text), port, depth, drafts, split
    )
