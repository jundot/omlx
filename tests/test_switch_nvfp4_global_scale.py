# SPDX-License-Identifier: Apache-2.0
"""ModelOpt NVFP4 routed experts with a per-expert ``weight_scale_2``."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_vlm.quantization.nvfp4 import replace_scaled_quantized_linears

import omlx.patches.m5_gather_qmm as m5
from omlx.patches.deepseek_v4.switch_layers import (
    NVFP4_GLOBAL_SCALE,
    SwitchGLU,
    install_nvfp4_global_scales,
)
from omlx.patches.mlx_vlm_glm5_next_compat import (
    apply_mlx_vlm_glm5_next_compat_patch,
)
from omlx.patches.mlx_vlm_mtp import glm5_next_vlm_runtime
from omlx.patches.moe_gate_up_fusion import (
    apply_switch_glu_gate_up_fusion,
    can_fuse,
)

requires_metal = pytest.mark.skipif(
    not mx.metal.is_available(), reason="gather_qmm global_scale runs on the GPU"
)

E, N, K = 4, 64, 128


@pytest.fixture(autouse=True)
def _fresh_reroute():
    """Leave ``mx.gather_qmm`` as this test found it."""
    was_installed = getattr(mx.gather_qmm, "_omlx_m5_reroute", False)
    raw = m5._original_gather_qmm if was_installed else mx.gather_qmm
    yield
    mx.gather_qmm = m5._gather_qmm_rerouted if was_installed else raw
    m5._original_gather_qmm = raw


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.switch_mlp = SwitchGLU(K, N, E)


def _modelopt_nvfp4(seed, out_dims=N, in_dims=K):
    """Per-expert NVFP4 codes, block scales, ``weight_scale_2`` and f32 weights."""
    rng = np.random.default_rng(seed)
    spread = np.array([0.01, 0.1, 1.0, 3.0], np.float32)[:, None, None]
    w = mx.array(
        rng.standard_normal((E, out_dims, in_dims)).astype(np.float32) * spread
    )
    codes, scales, ws2, dense = [], [], [], []
    for e in range(E):
        amax = mx.abs(w[e]).max().astype(mx.float32)
        q, s = mx.quantize(w[e], mode="nvfp4", global_scale=amax)
        codes.append(q)
        scales.append(s)
        ws2.append(amax / NVFP4_GLOBAL_SCALE)
        dense.append(
            mx.dequantize(q, s, mode="nvfp4", global_scale=amax, dtype=mx.float32)
        )
    return mx.stack(codes), mx.stack(scales), mx.stack(ws2), mx.stack(dense)


def _load_gate(block, seed=0):
    codes, scales, ws2, dense = _modelopt_nvfp4(seed)
    mapped = install_nvfp4_global_scales(
        block, {"switch_mlp.gate_proj.weight_scale_2": ws2}
    )
    proj = block.switch_mlp.gate_proj.to_quantized(group_size=16, bits=4, mode="nvfp4")
    proj.update(
        {
            "weight": codes,
            "scales": scales,
            "global_scale": mapped["switch_mlp.gate_proj.global_scale"],
        }
    )
    return proj, dense


def _reference(x, indices, dense):
    return mx.stack([x[i, 0] @ dense[int(indices[i])].T for i in range(x.shape[0])])[
        :, None
    ]


def _rel(a, b):
    return float(mx.abs(a - b).max() / mx.abs(b).max())


def test_install_ignores_other_scales_and_stand_ins():
    weights = {"vision.mlp.fc1.weight_scale_2": mx.array(0.5)}
    assert install_nvfp4_global_scales(_Block(), weights) == weights
    weights["switch_mlp.gate_proj.weight_scale_2"] = mx.ones((E,))
    assert install_nvfp4_global_scales(object(), weights) is weights


@requires_metal
def test_decode_applies_global_scale():
    proj, dense = _load_gate(_Block())
    x = mx.random.normal((6, 1, K), key=mx.random.key(1))
    indices = mx.array([0, 1, 2, 3, 2, 1], dtype=mx.uint32)
    y = proj(x, indices)
    assert _rel(y, _reference(x, indices, dense)) < 5e-3


@requires_metal
def test_sorted_prefill_keeps_global_scale_on_the_reroute(monkeypatch):
    m5.apply_m5_gather_qmm_workaround()

    def no_nax(*args, **kwargs):
        raise AssertionError("the NAX reroute has no global scale input")

    monkeypatch.setattr(m5, "_nax_sorted_gather_qmm", no_nax)
    proj, dense = _load_gate(_Block())
    indices = mx.sort(mx.array(np.random.default_rng(2).integers(0, E, 64), mx.uint32))
    x = mx.random.normal((64, 1, K), key=mx.random.key(3))
    y = proj(x, indices, sorted_indices=True)
    assert _rel(y, _reference(x, indices, dense)) < 5e-3


def _nvfp4_switch_glu():
    """Gate/up share tensor scales but not packed weights, like ModelOpt."""
    block = _Block()
    sw = block.switch_mlp
    for seed, name in enumerate(("gate_proj", "up_proj", "down_proj")):
        out_dims, in_dims = (K, N) if name == "down_proj" else (N, K)
        codes, scales, ws2, _ = _modelopt_nvfp4(seed, out_dims, in_dims)
        if name == "gate_proj":
            gate_scale = ws2
        elif name == "up_proj":
            ws2 = gate_scale
        mapped = install_nvfp4_global_scales(
            block, {f"switch_mlp.{name}.weight_scale_2": ws2}
        )
        q = sw[name].to_quantized(group_size=16, bits=4, mode="nvfp4")
        q.update(
            {
                "weight": codes,
                "scales": scales,
                "global_scale": mapped[f"switch_mlp.{name}.global_scale"],
            }
        )
        sw[name] = q
    return sw


def test_gate_up_fusion_needs_matching_global_scales():
    sw = _nvfp4_switch_glu()
    assert can_fuse(sw)
    sw.up_proj.global_scale = sw.up_proj.global_scale * 2
    assert not can_fuse(sw)


@requires_metal
@pytest.mark.parametrize("tokens", [1, 96])
def test_fused_nvfp4_matches_separate_projections(tokens):
    sw = _nvfp4_switch_glu()
    x = (mx.random.normal((1, tokens, K), key=mx.random.key(5)) * 0.1).astype(
        mx.bfloat16
    )
    indices = mx.array(
        np.random.default_rng(6).integers(0, E, (1, tokens, 2)), mx.uint32
    )
    expected = sw(x, indices)
    mx.eval(expected)
    holder = type("Model", (_Block,), {"__module__": "mlx_vlm.models.glm5_next"})()
    holder.switch_mlp = sw
    assert apply_switch_glu_gate_up_fusion(holder) == 1
    actual = sw(x, indices)
    assert _rel(actual, expected) < 1e-2


@pytest.mark.parametrize("mtp_naming", ["converted", "nextn"])
@pytest.mark.parametrize("with_scale", [False, True])
def test_glm_sanitize_quantize_and_mtp_preserve_scales(mtp_naming, with_scale):
    apply_mlx_vlm_glm5_next_compat_patch()
    assert glm5_next_vlm_runtime.apply()
    from mlx_vlm.models.glm5_next import Model, ModelConfig, TextConfig

    config = TextConfig(
        model_type="glm5_next_text",
        vocab_size=128,
        hidden_size=K,
        intermediate_size=128,
        moe_intermediate_size=N,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        n_shared_experts=1,
        n_routed_experts=E,
        routed_scaling_factor=1.0,
        kv_lora_rank=32,
        q_lora_rank=32,
        qk_rope_head_dim=0,
        v_head_dim=32,
        qk_nope_head_dim=32,
        num_experts_per_tok=2,
        first_k_dense_replace=0,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        index_topk=4,
        index_head_dim=32,
        index_n_heads=2,
        layer_types=["deepseek_sparse_attention"],
        mlp_layer_types=["sparse"],
        linear_attn_config={"num_heads": 2, "head_dim": 32},
        num_nextn_predict_layers=1,
    )
    model = Model(ModelConfig(text_config=config, model_type="glm5_next"))
    ws2 = mx.array([0.01, 0.1, 1.0, 3.0], mx.float32)
    backbone = "language_model.model.layers.0.mlp.switch_mlp.gate_proj"
    mtp = "language_model.mtp.0.block.mlp.switch_mlp.gate_proj"
    mtp_input = (
        mtp
        if mtp_naming == "converted"
        else "language_model.model.layers.1.mlp.switch_mlp.gate_proj"
    )
    before = dict(tree_flatten(model.parameters()))
    suffix = "weight_scale_2" if with_scale else "weight"
    value = ws2 if with_scale else mx.ones((E, N, K))
    weights = model.sanitize(
        {f"{backbone}.{suffix}": value, f"{mtp_input}.{suffix}": value}
    )
    paths = [backbone, mtp]
    if not with_scale:
        assert set(weights) == {f"{path}.weight" for path in paths}
        after = dict(tree_flatten(model.parameters()))
        assert before.keys() == after.keys()
        assert all(after[key] is value for key, value in before.items())
        return
    assert set(weights) == {f"{path}.global_scale" for path in paths}
    nn.quantize(
        model,
        group_size=16,
        bits=4,
        mode="nvfp4",
        class_predicate=lambda path, module: path in paths,
    )
    # mlx-vlm's NVFP4 wrapper step: it must leave the renamed scales to oMLX.
    replace_scaled_quantized_linears(model, weights)
    model.load_weights(list(weights.items()), strict=False)
    modules = dict(model.named_modules())
    for path in paths:
        assert mx.array_equal(modules[path].global_scale, ws2 * 2688.0)
        assert modules[path].global_scale.dtype == mx.float32
