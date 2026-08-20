# SPDX-License-Identifier: Apache-2.0
"""GLM SwitchGLU's block-list MoE dispatch must match stock gather_qmm.

The dispatch (shared with DeepSeek-V4) replaces stock ``mx.gather_qmm``
with the native block-bucketed kernels on pre-NAX hardware — V4's own
comments document the stock path as a ~2x kernel-level regression there.
These tests pin numerical agreement for both GLU layouts (separate
gate/up and fused gate_up) on mxfp4 weights, and the kill switch.
"""

import pytest

mx = pytest.importorskip("mlx.core")

from omlx.patches.glm_moe_dsa import switch_layers as sl  # noqa: E402

pytestmark = pytest.mark.skipif(
    not sl._BLOCK_DISPATCH
    or not sl.glm_fast.has("deepseek_mxfp4_gather_qmm_blocks"),
    reason="native block kernels unavailable",
)

TOKENS, TOP_K, EXPERTS, DIM, HIDDEN = 96, 2, 8, 64, 64


def _quantize_glu(glu, fused, group_size=32, bits=4, mode="mxfp4", dtype=None):
    names = ("gate_up_proj", "down_proj") if fused else (
        "gate_proj", "up_proj", "down_proj"
    )
    for name in names:
        layer = getattr(glu, name)
        if dtype is not None and mode == "affine":
            # The affine block gate requires scales/biases in the
            # activation dtype; quantizing a same-dtype weight yields that.
            layer.weight = layer.weight.astype(dtype)
        setattr(
            glu, name,
            layer.to_quantized(group_size=group_size, bits=bits, mode=mode),
        )
    return glu


def _run_pair(fused, monkeypatch, dtype=mx.bfloat16, expect_kind=None, **quant):
    mx.random.seed(7)
    glu = sl.SwitchGLU(DIM, HIDDEN, EXPERTS, fused_gate_up=fused)
    _quantize_glu(glu, fused, dtype=dtype, **quant)
    if expect_kind is not None:
        # Guard against a vacuous pass: the gate must actually open.
        probe = mx.zeros((TOKENS * TOP_K, 1, HIDDEN), dtype=dtype)
        assert glu.down_proj._native_block_kind(probe, True) == expect_kind
    x = mx.random.normal((1, TOKENS, DIM)).astype(dtype)
    indices = mx.random.randint(0, EXPERTS, (1, TOKENS, TOP_K), dtype=mx.uint32)

    out_blocks = glu(x, indices)
    mx.eval(out_blocks)
    monkeypatch.setattr(sl, "_BLOCK_DISPATCH", False)
    out_stock = glu(x, indices)
    mx.eval(out_stock)
    return out_blocks, out_stock


@pytest.mark.parametrize("fused", [False, True])
def test_block_dispatch_matches_stock(fused, monkeypatch):
    out_blocks, out_stock = _run_pair(fused, monkeypatch, expect_kind="mxfp4")
    assert out_blocks.shape == out_stock.shape
    diff = mx.max(
        mx.abs(out_blocks.astype(mx.float32) - out_stock.astype(mx.float32))
    )
    scale = float(mx.max(mx.abs(out_stock.astype(mx.float32))).item()) or 1.0
    assert float(diff.item()) <= 2e-2 * scale, float(diff.item())


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_affine_wide_bits_match_stock(bits, dtype, monkeypatch):
    """4/8-bit affine (the common community exports) joined 2/3-bit —
    scales/biases must match the activation dtype for the gate to open."""
    if not sl.glm_fast.has("deepseek_affine_gather_qmm_blocks"):
        pytest.skip("affine block kernels unavailable")
    out_blocks, out_stock = _run_pair(
        False, monkeypatch, dtype=dtype, expect_kind="affine",
        group_size=64, bits=bits, mode="affine",
    )
    assert out_blocks.shape == out_stock.shape
    diff = mx.max(
        mx.abs(out_blocks.astype(mx.float32) - out_stock.astype(mx.float32))
    )
    scale = float(mx.max(mx.abs(out_stock.astype(mx.float32))).item()) or 1.0
    assert float(diff.item()) <= 2e-2 * scale, float(diff.item())


def test_inverse_scatter_sorting_matches_stock(monkeypatch):
    """GLM's production SwitchGLU sorts with the put_along_axis inverse
    permutation (inverse_scatter=True); the block dispatch sits between
    sort and unsort and must be permutation-transparent. (The fused
    weighted_sum combine consumes the same inv_order either way and its
    kernel rejects test-sized shapes, so it is exercised elsewhere.)"""
    mx.random.seed(11)
    glu = sl.SwitchGLU(DIM, HIDDEN, EXPERTS, inverse_scatter=True)
    _quantize_glu(glu, fused=False)
    x = mx.random.normal((1, TOKENS, DIM)).astype(mx.bfloat16)
    indices = mx.random.randint(0, EXPERTS, (1, TOKENS, TOP_K), dtype=mx.uint32)

    out_blocks = glu(x, indices)
    mx.eval(out_blocks)
    monkeypatch.setattr(sl, "_BLOCK_DISPATCH", False)
    out_stock = glu(x, indices)
    mx.eval(out_stock)
    assert out_blocks.shape == out_stock.shape
    diff = mx.max(
        mx.abs(out_blocks.astype(mx.float32) - out_stock.astype(mx.float32))
    )
    scale = float(mx.max(mx.abs(out_stock.astype(mx.float32))).item()) or 1.0
    assert float(diff.item()) <= 2e-2 * scale, float(diff.item())


def test_kill_switch_forces_stock(monkeypatch):
    monkeypatch.setattr(sl, "_BLOCK_DISPATCH", False)
    glu = _quantize_glu(sl.SwitchGLU(DIM, HIDDEN, EXPERTS), fused=False)
    x = mx.random.normal((1, TOKENS, DIM)).astype(mx.bfloat16)
    indices = mx.random.randint(0, EXPERTS, (1, TOKENS, TOP_K), dtype=mx.uint32)
    assert glu.down_proj._native_block_kind(
        mx.zeros((TOKENS * TOP_K, 1, HIDDEN), dtype=mx.bfloat16), True
    ) is None
    out = glu(x, indices)
    mx.eval(out)
    assert out.shape == (1, TOKENS, TOP_K, DIM)


def test_affine_native_rejects_k_not_divisible_by_group_size():
    """K=96 at group_size=64 truncates to K//gs == 1, so a 1-wide scales
    tensor used to slip past the shape validation and the kernel read out
    of bounds; the native op must throw instead of launching."""
    if not sl.glm_fast.has("deepseek_affine_gather_qmm_blocks"):
        pytest.skip("affine block kernels unavailable")
    rows, n_out, k_in, gs, bits = 64, 32, 96, 64, 4
    x = mx.zeros((rows, 1, k_in), dtype=mx.bfloat16)
    weight = mx.zeros((EXPERTS, n_out, k_in * bits // 32), dtype=mx.uint32)
    scales = mx.zeros((EXPERTS, n_out, k_in // gs), dtype=mx.bfloat16)
    biases = mx.zeros((EXPERTS, n_out, k_in // gs), dtype=mx.bfloat16)
    indices = mx.zeros((rows,), dtype=mx.uint32)
    block_bm, variant = sl._block_config(rows, "affine")
    block_meta, block_count = sl._build_mxfp4_blocks(indices, EXPERTS, block_bm)
    with pytest.raises(Exception, match="[Gg]roup|[Ss]cales|[Ss]hape"):
        mx.eval(
            sl.glm_fast.deepseek_affine_gather_qmm_blocks(
                x, weight, scales, biases,
                block_meta, block_count, gs, bits, variant,
            )
        )


def test_small_decode_batches_stay_on_stock():
    # indices.size < 64 skips the sort, and unsorted routes never qualify
    # for the block kernels — decode-sized calls keep today's exact path.
    glu = _quantize_glu(sl.SwitchGLU(DIM, HIDDEN, EXPERTS), fused=False)
    x = mx.random.normal((1, 4, DIM)).astype(mx.bfloat16)
    indices = mx.random.randint(0, EXPERTS, (1, 4, TOP_K), dtype=mx.uint32)
    out = glu(x, indices)
    mx.eval(out)
    assert out.shape == (1, 4, TOP_K, DIM)
