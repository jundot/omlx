# SPDX-License-Identifier: Apache-2.0
"""Tiny, offline NumPy oracles for the HF HY-V4-Flash text MLPs.

Reference: transformers c2ec850510a37978ea83df226589623d5f2be776,
models/hy_v4_flash/modeling_hy_v4_flash.py.
"""

from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from mlx.utils import tree_unflatten  # noqa: E402

from omlx.patches.hy_v4_flash.experts import build_mlp, rotate_activation  # noqa: E402

_BANKS = ("experts.gate_up_proj", "experts.down_proj")
_PREFIX = "model.layers.1.mlp."
_SPEC = {"bits": 8, "block": 256, "group_size": 128}


def _args(**overrides):
    fields = dict(
        mlp_layer_types=["dense", "sparse"],
        hidden_size=16,
        intermediate_size=32,
        n_routed_experts=4,
        n_shared_experts=2,
        num_experts_per_tok=3,
        n_group=1,
        topk_group=1,
        routed_expert_hidden_size=512,
        moe_intermediate_size=256,
        norm_topk_prob=True,
        routed_scaling_factor=3.998,
        hidden_act="silu",
        swiglu_limit=0.625,
        rms_norm_eps=1e-5,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _bf(x):
    """Round to nearest even BF16, represented in FP32 for NumPy matmul."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) & np.uint32(0xFFFF0000)).view(np.float32)


def _source(args, block):
    rng = np.random.default_rng(462)
    h, e, i = args.hidden_size, args.n_routed_experts, args.moe_intermediate_size
    weights = {}

    def linear(name, shape):
        weights[name] = _bf(rng.normal(size=shape) / np.sqrt(shape[-1]))

    def dense(prefix, mid):
        for name in ("gate_proj", "up_proj"):
            linear(prefix + name + ".weight", (mid, h))
        linear(prefix + "down_proj.weight", (h, mid))

    if args.mlp_layer_types[block] == "dense":
        dense("", args.intermediate_size)
        norm = "post_fc2_norm.weight"
    else:
        latent = args.routed_expert_hidden_size
        if latent is not None:
            linear("routed_expert_down_proj.weight", (latent, h))
            linear("routed_expert_up_proj.weight", (h, latent))
        else:
            latent = h
        linear("gate.weight", (e, h))
        weights["gate.e_score_correction_bias"] = rng.normal(0, 0.2, e).astype(
            np.float32
        )
        # Source leaves are bare bank names, not per-expert linears.
        linear(_BANKS[0], (e, 2 * i, latent))
        linear(_BANKS[1], (e, latent, i))
        dense("shared_experts.", i * args.n_shared_experts)
        norm = "combine_output_norm.weight"
    weights[norm] = _bf(rng.uniform(0.5, 1.5, h))
    return weights, _bf(rng.normal(size=(2, 3, h)))


def _reference(args, w, x):
    # NumPy translation of HF HYV4Flash{MLP,TopkRouter,Experts,MoE}:
    # individual expert matvecs, without MLX gather, packing or Hadamards.
    def fc(a, weight):
        return _bf(a @ weight.T)

    def swiglu(gate, up):
        return _bf(_bf(gate / (1 + np.exp(-gate))) * up)

    def dense(a, prefix):
        gate = fc(a, w[prefix + "gate_proj.weight"])
        up = fc(a, w[prefix + "up_proj.weight"])
        return fc(swiglu(gate, up), w[prefix + "down_proj.weight"])

    def norm(a, weight):
        unit = a / np.sqrt(np.mean(a * a, axis=-1, keepdims=True) + args.rms_norm_eps)
        return _bf(_bf(unit) * weight)

    if "post_fc2_norm.weight" in w:
        return norm(dense(x, ""), w["post_fc2_norm.weight"])
    shape = x.shape
    x = x.reshape(-1, x.shape[-1])
    scores = 1 / (1 + np.exp(-(x @ w["gate.weight"].T)))
    choice = scores + w["gate.e_score_correction_bias"]
    idx = np.argsort(choice, axis=-1)[:, -args.num_experts_per_tok :]
    selected = np.take_along_axis(scores, idx, axis=-1)
    if args.norm_topk_prob:
        selected /= selected.sum(-1, keepdims=True) + 1e-20
    selected *= args.routed_scaling_factor
    latent = (
        fc(x, w["routed_expert_down_proj.weight"])
        if args.routed_expert_hidden_size is not None
        else x
    )
    out = np.zeros_like(latent)
    for e in range(args.n_routed_experts):
        rows, slots = np.nonzero(idx == e)
        gate, up = np.split(fc(latent[rows], w[_BANKS[0]][e]), 2, axis=-1)
        limit = args.swiglu_limit
        act = swiglu(np.minimum(gate, limit), np.clip(up, -limit, limit))
        term = _bf(fc(act, w[_BANKS[1]][e]) * selected[rows, slots, None])
        out[rows] = _bf(out[rows] + term)
    if args.routed_expert_hidden_size is not None:
        out = fc(out, w["routed_expert_up_proj.weight"])
    out = _bf(out + dense(x, "shared_experts."))
    return norm(out, w["combine_output_norm.weight"]).reshape(shape)


def _hadamard(x):
    # Independent normalized block transform, not the runtime rotation helper.
    shape = x.shape
    y = x.reshape(-1, 256)
    for stride in (1, 2, 4, 8, 16, 32, 64, 128):
        a, b = np.split(y.reshape(-1, 2 * stride), 2, axis=-1)
        y = np.concatenate((a + b, a - b), axis=-1)
    return y.reshape(shape) / 16


def _checkpoint(source, packed, bits=8):
    rng = np.random.default_rng(17)
    result = {}
    for name, w in source.items():
        if name in packed:
            signs = rng.choice(np.array([-1, 1], np.float32), size=w.shape[-1])
            rotated = mx.array(_hadamard(w * signs)).astype(mx.bfloat16)
            q, s, b = mx.quantize(rotated, group_size=128, bits=bits)
            for suffix, tensor in (
                ("weight", q),
                ("oqh_scales", s),
                ("oqh_biases", b),
                ("oqh_signs", mx.array(signs.reshape(-1, 256)).astype(mx.bfloat16)),
            ):
                result[f"{name}.{suffix}"] = tensor
        else:
            key = name + ".weight" if name in _BANKS else name
            result[key] = mx.array(w).astype(
                mx.float32 if name.endswith("_bias") else mx.bfloat16
            )
    return result


def _run(args, block, packed=(), bits=8):
    spec = {**_SPEC, "bits": bits}
    args.omlx_oqh = (
        {"version": 1, "tensors": {_PREFIX + k: spec for k in packed}}
        if packed
        else None
    )
    w, x = _source(args, block)
    model = build_mlp(args, block)
    # The standalone runtime assigns tensors to its existing expert banks.
    # The server's process-wide load_weights hook would replace those banks.
    model.update(
        tree_unflatten(list(_checkpoint(w, packed, bits).items())), strict=True
    )
    got = model(mx.array(x).astype(mx.bfloat16))
    assert got.dtype == mx.bfloat16
    ref = _reference(args, w, x)
    got = np.asarray(got.astype(mx.float32))
    assert got.shape == ref.shape
    return np.linalg.norm(got - ref) / np.linalg.norm(ref)


def test_dense_unclamped_swiglu_and_sandwich_norm():
    assert _run(_args(), 0) < 0.007


@pytest.mark.parametrize(
    "packed",
    [
        pytest.param((), id="bf16"),
        pytest.param(_BANKS[:1], id="packed-gate-up"),
        pytest.param(_BANKS[1:], id="packed-down"),
        pytest.param(_BANKS, id="packed-both"),
    ],
)
def test_latent_and_shared_experts_match_bf16_source(packed):
    err = _run(_args(), 1, packed)
    assert err < (0.025 if packed else 0.007)


@pytest.mark.parametrize("bits, tol", [(4, 0.18), (5, 0.09), (6, 0.055)])
def test_lower_bit_packed_experts_match_bf16_source(bits, tol):
    assert _run(_args(), 1, _BANKS, bits) < tol


def test_optional_latent_bridges_absent():
    assert (
        _run(_args(routed_expert_hidden_size=None, moe_intermediate_size=24), 1) < 0.007
    )


@pytest.mark.parametrize("normalize", [False, True])
def test_corrected_group_selection_uses_top_two_but_unbiased_mixing_scores(normalize):
    args = _args(
        hidden_size=1,
        n_routed_experts=6,
        n_group=2,
        topk_group=1,
        num_experts_per_tok=2,
        norm_topk_prob=normalize,
    )
    # Group 0 contains the highest single adjusted score, but group 1 wins
    # by the sum of its best two. The correction selects 5 instead of 4.
    scores = np.array([0.99, 0.35, 0.01, 0.60, 0.55, 0.10], dtype=np.float32)
    logits = np.log(scores / (1 - scores))
    gate = build_mlp(args, 1).gate
    gate.weight = mx.array(logits[:, None])
    gate.e_score_correction_bias = mx.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.8])
    observed, weights, indices = gate(mx.ones((1, 1, 1), mx.bfloat16))
    np.testing.assert_allclose(observed, logits[None], atol=1e-6)
    idx = np.asarray(indices)[0]
    assert set(idx.tolist()) == {3, 5}
    expect = scores[idx]
    if normalize:
        expect /= expect.sum()
    np.testing.assert_allclose(
        weights, (expect * args.routed_scaling_factor)[None], rtol=1e-6
    )


def test_signed_rotation_is_orthonormal_per_256_not_full_width():
    rng = np.random.default_rng(331)
    x = _bf(rng.normal(size=(2, 3, 512)))
    signs = rng.choice(np.array([-1, 1], np.float32), 512)
    y = rotate_activation(
        mx.array(x).astype(mx.bfloat16),
        mx.array(signs.reshape(2, 256)).astype(mx.bfloat16),
    )
    np.testing.assert_allclose(
        y.astype(mx.float32), _bf(_hadamard(x * signs)), rtol=0.008, atol=1e-4
    )
    with pytest.raises(ValueError, match="signs"):
        rotate_activation(mx.array(x), mx.ones((1, 256)))


@pytest.mark.parametrize(
    "meta",
    [
        {"version": 2, "tensors": {}},
        {"version": True, "tensors": {}},
        {"version": 1, "tensors": []},
        {"version": 1, "tensors": {_PREFIX + _BANKS[0]: {**_SPEC, "bits": 3}}},
        {"version": 1, "tensors": {_PREFIX + _BANKS[0]: {**_SPEC, "block": 128}}},
        {"version": 1, "tensors": {_PREFIX + _BANKS[0]: {**_SPEC, "group_size": 64}}},
        {"version": 1, "tensors": {_PREFIX + _BANKS[0] + ".weight": _SPEC}},
        {"version": 1, "tensors": {"model.layers.0.mlp." + _BANKS[1]: _SPEC}},
    ],
)
def test_reject_unsupported_format_instead_of_silently_using_bf16(meta):
    with pytest.raises(ValueError, match="oQh|omlx_oqh"):
        build_mlp(_args(omlx_oqh=meta), 1)


@pytest.mark.parametrize(
    "name,dim",
    [(_BANKS[0], "routed_expert_hidden_size"), (_BANKS[1], "moe_intermediate_size")],
)
def test_check_selected_bank_input_width(name, dim):
    args = _args(
        omlx_oqh={"version": 1, "tensors": {_PREFIX + name: _SPEC}}, **{dim: 384}
    )
    with pytest.raises(ValueError, match="width 384.*multiple of 256"):
        build_mlp(args, 1)
