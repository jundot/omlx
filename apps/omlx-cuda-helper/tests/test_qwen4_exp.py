# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8 Flash-Next stages must reproduce the oMLX contract fixture."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from omlx_cuda_helper.families import qwen4_exp  # noqa: E402
from omlx_cuda_helper.families.qwen4_exp import kernels  # noqa: E402
from omlx_cuda_helper.families.qwen4_exp.attention import Attention  # noqa: E402
from omlx_cuda_helper.families.qwen4_exp.layers import (  # noqa: E402
    GatedDeltaNet,
    HyperConnection,
    MoE,
)
from omlx_cuda_helper.families.qwen4_exp.ops import Weights  # noqa: E402
from omlx_cuda_helper.families.qwen4_exp.stage import load_stage  # noqa: E402

# Recorded by tests/fixtures/record_prefill_contract.py in the oMLX tree.
FIXTURE = Path(__file__).parents[3] / "tests/fixtures/prefill_contract/qwen4_exp"
needs_fixture = pytest.mark.skipif(
    not FIXTURE.exists(), reason="the oMLX contract fixture is not next to this tree"
)


def _case():
    meta = json.loads((FIXTURE / "meta.json").read_text())
    data = np.load(FIXTURE / "case.npz")
    arrays = {name: (data[name], dtype) for name, dtype, _, _ in meta["manifest"]}
    return meta, arrays


def _group(arrays, tag):
    return {n[len(tag) :]: v for n, v in arrays.items() if n.startswith(tag)}


@needs_fixture
@pytest.mark.parametrize(
    "bounds",
    [(0, 3), (0, 2, 3), (0, 4)],
    ids=["one-stage", "two-stages", "fewer-layers-than-held"],
)
def test_stages_reproduce_the_contract_fixture(bounds):
    meta, arrays = _case()
    k, sizes, skip = meta["layers"], meta["sizes"], meta["prefix_chunks"]
    assert meta["contract"] == qwen4_exp.CONTRACT
    assert all(a >= qwen4_exp.FIRST_STAGE_MIN for a in bounds[1:-1])
    stages = [
        load_stage(FIXTURE / "model", first, last)
        for first, last in zip(bounds, bounds[1:])
    ]
    for stage in stages:
        # A layer loaded again after a long prompt runs as at the first load.
        stage.drop_layer()
        stage.add_layer()
    jobs = [s.new_job(min(s.last, k) - s.first) for s in stages]
    for stage in stages:
        # On CUDA the replay then runs through the self-checked kernels.
        stage.enable_kernels()
    for stage, job in zip(stages, jobs):
        stage.load_prefix(job, _group(arrays, "p:"))

    tokens = np.asarray(meta["ids"], dtype=np.int32)[sum(sizes[:skip]) :]
    start = 0
    for index, size in enumerate(sizes[skip:]):
        boundary, state = None, {}
        for stage, job in zip(stages, jobs):
            boundary, own = stage.run(
                job,
                start=start,
                tokens=tokens[start : start + size],
                boundary=boundary,
                inputs=_group(arrays, f"in{index}:"),
                following=0,
                capture=None,
            )
            assert all(stage.first <= int(n.split(".")[0]) < stage.last for n in own)
            state.update(own)
        start += size
        for tag, got in ((f"b{index}:", boundary), (f"s{index}:", state)):
            want = _group(arrays, tag)
            assert sorted(got) == sorted(want), tag
            for name, (data, dtype) in got.items():
                expected, expected_dtype = want[name]
                assert (dtype, data.dtype, data.shape) == (
                    expected_dtype,
                    expected.dtype,
                    expected.shape,
                ), (
                    tag + name
                )
                if data.dtype.kind == "f":
                    np.testing.assert_allclose(
                        data,
                        expected,
                        rtol=meta["rtol"],
                        atol=meta["atol"],
                        err_msg=tag + name,
                    )
                else:
                    np.testing.assert_array_equal(data, expected, err_msg=tag + name)


@needs_fixture
def test_plan_counts_only_what_a_stage_loads():
    shared, layers = qwen4_exp.plan(FIXTURE / "model")
    assert len(layers) == 4 and min(layers) > 0
    # The embedding is shared; the output head, the mixer and the PLE table
    # stay on the host.
    assert shared == 64 * 32 * 4
    assert not qwen4_exp.resident(
        "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.0.weight"
    )


@pytest.mark.skipif(not kernels.available(), reason="CUDA kernels")
def test_kernels_pass_their_self_checks_on_the_checkpoint_geometry():
    # Qwen3.8 Flash-Next shapes with random bf16 weights: a kernel that fails
    # its check would only log a warning and leave the slow path in place.
    text = {
        "hidden_size": 2560,
        "hc_count": 4,
        "rms_norm_eps": 1e-6,
        "num_attention_heads": 24,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "indexer_n_heads": 4,
        "indexer_head_dim": 128,
        "indexer_budget": 2048,
        "indexer_compress_ratio": 4,
        "rope_parameters": {"partial_rotary_factor": 0.25, "rope_theta": 1e7},
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "output_gate_type": "sigmoid",
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 640,
    }
    shapes = {
        "self_attn.q_proj.weight": (12288, 2560),
        "self_attn.k_proj.weight": (512, 2560),
        "self_attn.v_proj.weight": (512, 2560),
        "self_attn.o_proj.weight": (2560, 6144),
        "self_attn.q_norm.weight": (256,),
        "self_attn.k_norm.weight": (256,),
        "self_attn.indexer.index_qk_proj.weight": (640, 2560),
        "self_attn.indexer.q_layernorm.weight": (128,),
        "self_attn.indexer.k_layernorm.weight": (128,),
        "hc.hc_norm.weight": (10240,),
        "hc.input_mix_weight_down.weight": (320, 10240),
        "hc.input_mix_weight_up.weight": (10240, 320),
        "hc.block_inject_weight.weight": (4, 10240),
        "gdn.in_proj_qkv.weight": (10240, 2560),
        "gdn.in_proj_z.weight": (6144, 2560),
        "gdn.in_proj_a.weight": (48, 2560),
        "gdn.in_proj_b.weight": (48, 2560),
        "gdn.out_proj.weight": (2560, 6144),
        "gdn.conv1d.weight": (10240, 4, 1),
        "gdn.A_log": (48,),
        "gdn.dt_bias": (48,),
        "gdn.norm.weight": (128,),
    }
    mx.random.seed(0)
    arrays = {
        name: (0.02 * mx.random.normal(shape)).astype(mx.bfloat16)
        for name, shape in shapes.items()
    }
    weights = Weights(arrays, {}, "")
    attention = Attention(weights.at("self_attn"), text)
    attention.enable_kernels()
    connection = HyperConnection(weights.at("hc"), text)
    connection.enable_kernels()
    linear = GatedDeltaNet(weights.at("gdn"), text)
    linear.enable_kernels()
    # Eight experts of the checkpoint's widths, 4-bit in groups of 64.
    bank = {"router": (8, 2560), "shared_expert_gate": (1, 2560)}
    bank.update({f"shared_expert.{n}_proj": (640, 2560) for n in ("gate", "up")})
    bank["shared_expert.down_proj"] = (2560, 640)
    for name, shape in (("gate", (8, 640, 2560)), ("up", (8, 640, 2560))):
        bank[f"switch_mlp.{name}_proj"] = shape
    bank["switch_mlp.down_proj"] = (8, 2560, 640)
    arrays = {}
    for name, shape in bank.items():
        dense = (0.02 * mx.random.normal(shape)).astype(mx.bfloat16)
        if name.startswith("switch_mlp"):
            packed = mx.quantize(dense, group_size=64, bits=4)
            for part, value in zip(("weight", "scales", "biases"), packed):
                arrays[f"{name}.{part}"] = value
        else:
            arrays[f"{name}.weight"] = dense
    arrays["gate.weight"] = arrays.pop("router.weight")
    config = {"quantization": {"group_size": 64, "bits": 4, "mode": "affine"}}
    moe = MoE(Weights(arrays, config, ""), text)
    moe.enable_kernels()
    assert attention.fused_prep and attention.selection is not None
    assert attention.kernel is not None
    assert connection.fused and connection.write is kernels.hc_write
    assert linear.fused and linear.recurrence is kernels.gated_delta
    assert moe.expert_kernel
