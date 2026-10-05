# SPDX-License-Identifier: Apache-2.0
"""Post-load gate/up fusion for oMLX's own SwitchGLU variants.

(omlx/patches/moe_gate_up_fusion.py)

MiMo V2 (GLM DSA SwitchGLU) and GLM-5.3 (DeepSeek V4 SwitchGLU) run the
routed gate and up projections as one gather_qmm over the concatenated
[gate; up] expert weights. Every output column is the same K-reduction as
in the separate calls, so the fused module must match the unfused one bit
for bit on every path: unsorted decode and verify blocks, sorted prefill,
with and without the native weighted-sum combine.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches import moe_gate_up_fusion as fusion
from omlx.patches.deepseek_v4 import switch_layers as v4
from omlx.patches.glm_moe_dsa import switch_layers as dsa

E, D, INTER, TOP_K = 16, 128, 64, 8


def _clamped_swiglu(x_up, x_gate):
    # GLM-5.3's activation (Glm5NextClampedSwiGLU with limit 10).
    x_gate = mx.clip(x_gate, a_min=None, a_max=10.0)
    x_up = mx.clip(x_up, a_min=-10.0, a_max=10.0)
    return nn.silu(x_gate) * x_up


class _Holder(nn.Module):
    """A loaded-model stand-in; its class module marks the model family."""

    def __init__(self, glu):
        super().__init__()
        self.layers = [glu]


def _holder_class(family: str):
    return type("Model", (_Holder,), {"__module__": f"mlx_lm.models.{family}"})


def _make_glu(module, quant, seed=0, activation=None):
    mx.random.seed(seed)
    kwargs = {} if activation is None else {"activation": activation}
    glu = module.SwitchGLU(D, INTER, E, **kwargs)
    for name in ("gate_proj", "up_proj", "down_proj"):
        glu[name].weight = glu[name].weight.astype(mx.bfloat16)
    group_size, bits, mode = quant
    nn.quantize(glu, group_size=group_size, bits=bits, mode=mode)
    mx.eval(glu.parameters())
    return glu


def _twin(module, glu, quant, activation=None):
    twin = _make_glu(module, quant, seed=99, activation=activation)
    for name in ("gate_proj", "up_proj", "down_proj"):
        for field in ("weight", "scales", "biases"):
            if glu[name].get(field) is not None:
                setattr(twin[name], field, mx.array(glu[name][field]))
    mx.eval(twin.parameters())
    return twin


def _routes(tokens, seed=1):
    key = mx.random.key(seed)
    scores = mx.random.uniform(shape=(1, tokens, E), key=key)
    return mx.argpartition(-scores, kth=TOP_K - 1, axis=-1)[..., :TOP_K]


def _scores(indices):
    s = mx.random.uniform(shape=indices.shape)
    return (s / s.sum(axis=-1, keepdims=True)).astype(mx.float32)


def _fused(module, glu, quant, family, activation=None):
    fused = _twin(module, glu, quant, activation=activation)
    model = _holder_class(family)(fused)
    assert fusion.apply_switch_glu_gate_up_fusion(model) == 1
    assert "gate_up_proj" in fused
    assert "gate_proj" not in fused and "up_proj" not in fused
    assert fused.gate_up_proj.weight.shape[1] == 2 * INTER
    return fused


def _assert_bit_exact(reference, fused, tokens):
    x = (mx.random.normal((1, tokens, D)) * 0.5).astype(mx.bfloat16)
    i = _routes(tokens)
    s = _scores(i)
    for weighted in (False, True):
        ref = reference(x, i, scores=s, weighted_sum=weighted)
        got = fused(x, i, scores=s, weighted_sum=weighted)
        mx.eval(ref, got)
        assert ref.shape == got.shape and ref.dtype == got.dtype
        assert bool(mx.array_equal(ref, got)), (tokens, weighted)


GLM_DSA_QUANTS = [(32, 4, "mxfp4"), (64, 4, "affine")]
V4_QUANTS = [(64, 4, "affine"), (64, 8, "affine"), (32, 4, "affine")]
# decode (8 routes, unsorted), verify block (24 routes), sorted prefill
# (768 routes) and a NAX-sized prefill (2048 routes, stock gather_qmm).
TOKENS = [1, 3, 96, 256]


@pytest.mark.parametrize("quant", GLM_DSA_QUANTS)
@pytest.mark.parametrize("tokens", TOKENS)
def test_glm_dsa_switch_glu_fused_is_bit_exact(quant, tokens):
    reference = _make_glu(dsa, quant)
    fused = _fused(dsa, reference, quant, "mimo_v2")
    _assert_bit_exact(reference, fused, tokens)


@pytest.mark.parametrize("quant", V4_QUANTS)
@pytest.mark.parametrize("tokens", TOKENS)
def test_deepseek_v4_switch_glu_fused_is_bit_exact(quant, tokens):
    reference = _make_glu(v4, quant, activation=_clamped_swiglu)
    fused = _fused(v4, reference, quant, "glm5_next", activation=_clamped_swiglu)
    _assert_bit_exact(reference, fused, tokens)


@pytest.mark.parametrize("quant", [(32, 4, "mxfp4"), (64, 2, "affine"), (64, 3, "affine")])
def test_deepseek_v4_keeps_native_pair_kernel_formats(quant):
    glu = _make_glu(v4, quant)
    assert not fusion.can_fuse(glu)
    model = _holder_class("glm5_next")(glu)
    assert fusion.apply_switch_glu_gate_up_fusion(model) == 0
    assert "gate_proj" in glu and "gate_up_proj" not in glu


def test_mismatched_gate_up_formats_stay_separate():
    holder = _holder_class("mimo_v2")(dsa.SwitchGLU(D, INTER, E))
    nn.quantize(
        holder,
        group_size=64,
        bits=4,
        class_predicate=lambda p, m: (
            {"group_size": 64, "bits": 8} if p.endswith("up_proj") else hasattr(m, "to_quantized")
        ),
    )
    glu = holder.layers[0]
    assert glu.up_proj.bits == 8 and glu.gate_proj.bits == 4
    assert not fusion.can_fuse(glu)
    assert fusion.apply_switch_glu_gate_up_fusion(holder) == 0


def test_unquantized_and_already_fused_modules_are_skipped():
    plain = dsa.SwitchGLU(D, INTER, E)
    assert not fusion.can_fuse(plain)
    fused = dsa.SwitchGLU(D, INTER, E, fused_gate_up=True)
    nn.quantize(fused, group_size=64, bits=4)
    assert not fusion.can_fuse(fused)


def test_only_mimo_and_glm5_next_families_and_kill_switch(monkeypatch):
    quant = (64, 4, "affine")
    other = _holder_class("deepseek_v4")(_make_glu(v4, quant))
    assert fusion.apply_switch_glu_gate_up_fusion(other) == 0
    assert fusion.apply_switch_glu_gate_up_fusion(object()) == 0

    monkeypatch.setenv("OMLX_MOE_GATE_UP_FUSION", "0")
    glm = _holder_class("glm5_next")(_make_glu(v4, quant))
    assert fusion.apply_switch_glu_gate_up_fusion(glm) == 0
    monkeypatch.delenv("OMLX_MOE_GATE_UP_FUSION")
    assert fusion.apply_switch_glu_gate_up_fusion(glm) == 1


def _mimo_moe(T):
    import importlib

    from omlx.patches.mimo_v2 import apply_mimo_v2_patch

    apply_mimo_v2_patch()
    mimo = importlib.import_module("mlx_lm.models.mimo_v2")
    if mimo._FusedSwitchGLU is None:
        pytest.skip("oMLX GLM MoE kernels unavailable")
    cfg = mimo.ModelArgs.from_dict(
        {
            "model_type": "mimo_v2",
            "vocab_size": 1000,
            "hidden_size": D,
            "intermediate_size": 256,
            "moe_intermediate_size": INTER,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "v_head_dim": 24,
            "rope_theta": 1000.0,
            "swa_num_attention_heads": 4,
            "swa_num_key_value_heads": 2,
            "swa_head_dim": 32,
            "swa_v_head_dim": 24,
            "swa_rope_theta": 1000.0,
            "sliding_window_size": 32,
            "add_full_attention_sink_bias": False,
            "add_swa_attention_sink_bias": True,
            "hybrid_layer_pattern": [0, 1],
            "moe_layer_freq": [0, 1],
            "n_routed_experts": E,
            "num_experts_per_tok": TOP_K,
            "n_group": 1,
            "topk_group": 1,
            "norm_topk_prob": True,
            "topk_method": "noaux_tc",
            "partial_rotary_factor": 0.5,
            "attention_bias": False,
            "layernorm_epsilon": 1e-5,
            "max_position_embeddings": 1000,
            "attention_value_scale": 0.707,
        }
    )
    moe = mimo.MoE(cfg)
    mx.random.seed(0)
    moe.gate.weight = mx.random.normal(moe.gate.weight.shape) * 0.1
    moe.gate.e_score_correction_bias = mx.zeros_like(moe.gate.e_score_correction_bias)
    for name in ("gate_proj", "up_proj", "down_proj"):
        lin = moe.switch_mlp[name]
        lin.weight = lin.weight.astype(mx.bfloat16)
    nn.quantize(moe.switch_mlp, group_size=32, bits=4, mode="mxfp4")
    x = mx.random.normal((1, T, D)).astype(mx.bfloat16)
    mx.eval(moe.parameters(), x)
    return moe, x


@pytest.mark.parametrize("T", [1, 4, 96, 300])
def test_mimo_moe_block_fused_matches_unfused(T):
    moe, x = _mimo_moe(T)
    ref = moe(x)
    mx.eval(ref)
    # MiMo's MoE module lives in mlx_lm.models.mimo_v2: a supported family.
    assert fusion.apply_switch_glu_gate_up_fusion(moe) == 1
    assert "gate_up_proj" in moe.switch_mlp
    got = moe(x)
    mx.eval(got)
    assert got.shape == ref.shape and got.dtype == ref.dtype
    assert bool(mx.array_equal(ref, got))


def _glm5_language():
    from omlx.patches import mlx_vlm_glm5_next_compat as compat

    compat.apply_mlx_vlm_glm5_next_compat_patch()
    import importlib

    return importlib.import_module("mlx_vlm.models.glm5_next.language")


def _eager_clamped_swiglu(x_up, x_gate, limit):
    x_gate = mx.clip(x_gate, a_min=None, a_max=limit)
    x_up = mx.clip(x_up, a_min=-limit, a_max=limit)
    return nn.silu(x_gate) * x_up


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
@pytest.mark.parametrize("rows", [1, 8, 3000])
def test_glm5_next_compiled_clamped_swiglu_is_bit_exact(dtype, rows):
    lang = _glm5_language()
    act = lang.Glm5NextClampedSwiGLU(10.0)
    # The fused gate/up output is split into strided halves.
    gate_up = (mx.random.normal((rows, 1, 2 * INTER)) * 8).astype(dtype)
    x_gate, x_up = mx.split(gate_up, 2, axis=-1)
    got = act(x_up, x_gate)
    ref = _eager_clamped_swiglu(x_up, x_gate, 10.0)
    mx.eval(got, ref)
    assert got.dtype == ref.dtype and bool(mx.array_equal(got, ref))
    # Another limit recompiles instead of reusing the first constant.
    got7 = lang.Glm5NextClampedSwiGLU(7.0)(x_up, x_gate)
    assert bool(mx.array_equal(got7, _eager_clamped_swiglu(x_up, x_gate, 7.0)))
    unclamped = lang.Glm5NextClampedSwiGLU(None)(x_up, x_gate)
    assert bool(mx.array_equal(unclamped, nn.silu(x_gate) * x_up))
