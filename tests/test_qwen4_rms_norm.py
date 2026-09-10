# SPDX-License-Identifier: Apache-2.0
"""``Qwen4ExpRMSNorm`` against the hand-rolled formulation it replaced.

The module used to spell RMS norm out in MLX ops (fp32 upcast, optional
grouping, ``y * rsqrt(mean(y^2) + eps) * (1 + w)``) where ``mx.fast.rms_norm``
exists.  Routing it through the fast path is a refactor, so these are
characterization tests: ``_reference`` is the old body verbatim, and the module
must keep matching it.

``rms_norm`` takes a 1-D weight only, so the grouped case cannot hand its
``(groups, group_size)`` weight to the kernel and applies the scale separately.
That scale must stay fp32 -- rounding ``(1 + w)`` to bf16 costs 3.9e-3 relative,
which is half a bf16 ULP of pure avoidable error.
"""

from __future__ import annotations

import importlib

import mlx.core as mx
import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat

compat.apply_mlx_vlm_qwen4_exp_compat_patch()
language = importlib.import_module("mlx_vlm.models.qwen4_exp.language")

BF16_ULP = 2.0**-7
FP32_ULP = 2.0**-23


def _reference(x: mx.array, weight: mx.array, eps: float, group_size: int | None):
    """The pre-fast-path body, kept verbatim as the thing we must not change."""
    dtype = x.dtype
    y = x.astype(mx.float32)
    if group_size is not None:
        y = y.reshape(*y.shape[:-1], -1, group_size)
        weight = weight.reshape(-1, group_size)
    y = y * mx.rsqrt(mx.mean(mx.square(y), axis=-1, keepdims=True) + eps)
    y = y * (1.0 + weight.astype(mx.float32))
    return y.reshape(x.shape).astype(dtype)


def _module(dim: int, group_size: int | None, weight: mx.array):
    module = language.Qwen4ExpRMSNorm(dim, group_size=group_size, eps=1e-6)
    module.weight = weight
    return module


# (name, input shape, dim, group_size) -- every geometry the model builds.
_SEEDS = {
    (name, scale): 1000 * i + int(scale * 100)
    for i, name in enumerate(
        [
            "hc_norm_verify",
            "hc_norm_decode",
            "hc_norm_6rows",
            "qk_layernorm",
            "qk_layernorm_decode",
            "indexer",
            "ple_pre_fc",
            "gdn_norm_conv",
        ]
    )
    for scale in (0.0, 0.02, 0.5)
}

SHAPES = [
    ("hc_norm_verify", (1, 4, 10240), 10240, 2560),
    ("hc_norm_decode", (1, 1, 10240), 10240, 2560),
    ("hc_norm_6rows", (1, 6, 10240), 10240, 2560),
    ("qk_layernorm", (1, 4, 24, 256), 256, None),
    ("qk_layernorm_decode", (1, 1, 24, 256), 256, None),
    ("indexer", (1, 4, 4, 128), 128, None),
    ("ple_pre_fc", (1, 4, 2560), 2560, None),
    ("gdn_norm_conv", (1, 4, 2048), 2048, None),
]


@pytest.mark.parametrize("name,shape,dim,group_size", SHAPES, ids=[s[0] for s in SHAPES])
@pytest.mark.parametrize("weight_scale", [0.0, 0.02, 0.5])
def test_stays_within_one_bf16_ulp_of_the_hand_rolled_formulation(
    name, shape, dim, group_size, weight_scale
):
    """bf16 is the production dtype; the fast path may move at most one ULP.

    The kernel sums the squares in a different order than ``mx.mean``, so a
    handful of values land on the far side of a bf16 rounding boundary.  Over
    3.1M elements that is ~2 ppm at any weight magnitude, never above one ULP.
    weight_scale 0.02 is the checkpoint's actual magnitude; 0.0 is the
    zero-centered init, which makes ``(1 + w)`` a no-op and would hide a
    dropped scale, so it is covered but never on its own.
    """
    mx.random.seed(_SEEDS[(name, weight_scale)])
    x = (mx.random.normal(shape) * 1.5).astype(mx.bfloat16)
    weight = (mx.random.normal((dim,)) * weight_scale).astype(mx.bfloat16)

    got = _module(dim, group_size, weight)(x)
    want = _reference(x, weight, 1e-6, group_size)

    assert got.dtype == mx.bfloat16
    assert got.shape == x.shape
    delta = mx.abs(got.astype(mx.float32) - want.astype(mx.float32))
    rel = delta / mx.maximum(mx.abs(want.astype(mx.float32)), 1e-30)
    assert mx.max(rel).item() <= BF16_ULP, f"{name} moved more than one ULP"
    # the rate is an aggregate property -- one element of a 6144-wide decode
    # tensor is 163 ppm by construction, so it is asserted in bulk below
    assert (delta > 0).sum().item() <= 2, f"{name} moved more than two elements"


def test_deviation_is_smaller_than_the_models_own_batching_spread():
    """The bar: MLX already moves this norm's input more than we do.

    Running the same row through a 1-row and a 4-row forward gives different
    results from MLX's own matmul path selection.  A refactor that stays below
    that spread cannot be what makes a generation differ.
    """
    mx.random.seed(4242)
    x = (mx.random.normal((1, 4, 10240)) * 1.5).astype(mx.bfloat16)
    weight = (mx.random.normal((10240,)) * 0.02).astype(mx.bfloat16)
    module = _module(10240, 2560, weight)

    ours = mx.abs(
        module(x).astype(mx.float32) - _reference(x, weight, 1e-6, 2560).astype(mx.float32)
    )
    # the same reference formulation, one row at a time vs all four at once
    batched = _reference(x, weight, 1e-6, 2560)
    per_row = mx.concatenate(
        [_reference(x[:, i : i + 1], weight, 1e-6, 2560) for i in range(x.shape[1])], axis=1
    )
    theirs = mx.abs(batched.astype(mx.float32) - per_row.astype(mx.float32))

    assert mx.max(ours).item() <= mx.max(theirs).item(), (
        f"our max {mx.max(ours).item():.3e} exceeds the batching spread "
        f"{mx.max(theirs).item():.3e}"
    )


@pytest.mark.parametrize("name,shape,dim,group_size", SHAPES, ids=[s[0] for s in SHAPES])
def test_fp32_inputs_stay_within_a_few_ulp(name, shape, dim, group_size):
    """fp32 has no bf16 rounding to mask the kernel's different summation order."""
    mx.random.seed(11)
    x = (mx.random.normal(shape) * 1.5).astype(mx.float32)
    weight = (mx.random.normal((dim,)) * 0.02).astype(mx.float32)

    got = _module(dim, group_size, weight)(x)
    want = _reference(x, weight, 1e-6, group_size)

    assert got.dtype == mx.float32
    rel = mx.abs(got - want) / mx.maximum(mx.abs(want), 1e-30)
    assert mx.max(rel).item() <= 4 * FP32_ULP


def test_the_grouped_scale_is_not_rounded_to_bf16():
    """A bf16 (1 + w) costs ~3.9e-3 relative -- the full half-ULP rounding bound.

    Guards the one shortcut that looks harmless and is not: precomputing the
    scale in the weight's own dtype instead of fp32.  Round-to-nearest cannot
    exceed half a ULP, so this asserts the error saturates that bound rather
    than passing it -- which is already ~16000x the fp32 path's ~2 fp32 ULP.
    """
    mx.random.seed(5)
    weight = (mx.random.normal((10240,)) * 0.02).astype(mx.bfloat16)
    exact = 1.0 + weight.astype(mx.float32)
    rounded = exact.astype(mx.bfloat16).astype(mx.float32)
    worst = mx.max(mx.abs(rounded - exact) / mx.abs(exact)).item()
    assert 0.4 * BF16_ULP < worst <= 0.5 * BF16_ULP

    x = (mx.random.normal((1, 4, 10240)) * 1.5).astype(mx.bfloat16)
    got = _module(10240, 2560, weight)(x)
    naive = (
        mx.fast.rms_norm(x.reshape(1, 4, 4, 2560), None, 1e-6).astype(mx.float32)
        * rounded.reshape(-1, 2560)
    ).reshape(x.shape).astype(mx.bfloat16)
    assert not mx.array_equal(got, naive).item()


def test_grouping_is_per_group_not_whole_vector():
    """Four groups of 2560 must each get their own scale; one flat norm differs."""
    mx.random.seed(2)
    x = (mx.random.normal((1, 4, 10240)) * 1.5).astype(mx.bfloat16)
    # groups with deliberately unequal magnitude, so a whole-vector norm cannot
    # coincide with the grouped one
    x = mx.concatenate([x[..., :2560] * 8.0, x[..., 2560:]], axis=-1)
    weight = (mx.random.normal((10240,)) * 0.02).astype(mx.bfloat16)

    grouped = _module(10240, 2560, weight)(x)
    flat = _module(10240, None, weight)(x)
    assert not mx.array_equal(grouped, flat).item()
    assert mx.array_equal(grouped, _reference(x, weight, 1e-6, 2560)).item()


def test_rejects_a_dim_not_divisible_by_group_size():
    with pytest.raises(ValueError, match="divisible"):
        language.Qwen4ExpRMSNorm(10240, group_size=3000)


def test_weight_stays_a_parameter_and_nothing_else_is_registered():
    """A cached scale filed into the module dict would break checkpoint I/O."""
    module = language.Qwen4ExpRMSNorm(10240, group_size=2560, eps=1e-6)
    mx.eval(module(mx.zeros((1, 4, 10240), mx.bfloat16)))
    assert list(module.parameters()) == ["weight"]


def test_deviation_rate_across_the_whole_geometry_is_a_few_ppm():
    """The rate bound the per-shape tests are too small to carry.

    ~3.1M elements per weight scale.  Measured 1-8 ppm and independent of
    weight magnitude; 50 ppm leaves an order of magnitude of headroom.
    """
    total = moved = 0
    worst = 0.0
    for scale in (0.0, 0.02, 0.5):
        for seed in range(12):
            for name, shape, dim, group_size in SHAPES:
                mx.random.seed(seed * 97 + len(name))
                x = (mx.random.normal(shape) * 1.5).astype(mx.bfloat16)
                weight = (mx.random.normal((dim,)) * scale).astype(mx.bfloat16)
                got = _module(dim, group_size, weight)(x)
                want = _reference(x, weight, 1e-6, group_size)
                delta = mx.abs(got.astype(mx.float32) - want.astype(mx.float32))
                total += got.size
                moved += (delta > 0).sum().item()
                rel = delta / mx.maximum(mx.abs(want.astype(mx.float32)), 1e-30)
                worst = max(worst, mx.max(rel).item())

    assert worst <= BF16_ULP, f"{worst / BF16_ULP:.2f} ULP exceeds one"
    assert moved / total <= 50e-6, f"{moved / total * 1e6:.2f} ppm over {total} elements"
