# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 row-exact verify experts in expert order: each (row, expert) pair
keeps the bits ``gather_qmm`` gives it in the row's one-row decode step."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.inkling2 import model as inkling2
from omlx.patches.inkling2 import verify_experts
from omlx.patches.qwen35_verify_qmm import set_verify_qmm_armed

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

_EXPERTS, _TOP_K = 10, 4


def _experts(hidden, inter, fmt="nvfp4", experts=_EXPERTS):
    args = inkling2.TextArgs(
        hidden_size=hidden,
        intermediate_size=inter,
        n_routed_experts=experts,
        num_experts_per_tok=_TOP_K,
    )
    spec = inkling2.expert_quant_spec(fmt)
    mod = inkling2.Inkling2Experts(args, spec, spec)
    rng = np.random.default_rng(0)
    for proj, out_dims, in_dims in (
        (mod.w13, 2 * inter, hidden),
        (mod.w2, hidden, inter),
    ):
        codes, scales, amaxes = [], [], []
        for _ in range(experts):
            w = mx.array(
                (0.05 * rng.standard_normal((out_dims, in_dims))).astype(np.float32)
            )
            if fmt == "nvfp4":
                amax = mx.abs(w).max()
                q, s = mx.quantize(w, mode="nvfp4", global_scale=amax)
                amaxes.append(amax)
            else:
                q, s = mx.quantize(w, mode=fmt)
            codes.append(q)
            scales.append(s)
        proj.weight = mx.stack(codes)
        proj.scales = mx.stack(scales)
        if amaxes:
            proj.global_scale = mx.stack(amaxes).astype(mx.float32)
    return mod


def _block(rows, hidden, seed, dtype=mx.bfloat16, experts=_EXPERTS):
    """Activations, routing and weights of a verify block whose rows share
    experts: each row keeps half of the previous row's experts."""
    rng = np.random.default_rng(seed)
    mx.random.seed(seed)
    x = (mx.random.normal((rows, hidden)) * 0.7).astype(dtype)
    ids = [rng.permutation(experts)[:_TOP_K]]
    for _ in range(rows - 1):
        keep = rng.permutation(ids[-1])[: _TOP_K // 2]
        rest = [e for e in rng.permutation(experts) if e not in keep]
        ids.append(rng.permutation(np.concatenate([keep, rest[: _TOP_K - keep.size]])))
    inds = mx.array(np.stack(ids).astype(np.uint32))
    weights = mx.random.uniform(shape=(rows, _TOP_K)).astype(mx.float32)
    return x, inds, weights


def _armed(mod, x, inds, weights):
    set_verify_qmm_armed(True, row_exact=True)
    try:
        z, w = mod.unreduced(x, inds, weights)
        mx.eval(z, w)
    finally:
        set_verify_qmm_armed(False)
    return z, w


def _count_gathers(monkeypatch):
    calls = []
    gather = verify_experts.gather

    def counted(*args, **kwargs):
        calls.append(1)
        return gather(*args, **kwargs)

    monkeypatch.setattr(verify_experts, "gather", counted)
    return calls


# K = 304 leaves 48 values past the last full 256-value block (lanes 0 to 5),
# K = 144 and 80 have no full block, 2304 and 1152 are Inkling 2 Mini's.
@pytest.mark.parametrize(
    "hidden, inter", [(304, 144), (272, 80), (2304, 1152)], ids=str
)
@pytest.mark.parametrize("rows", [2, 3, 5, 16])
# float16 applies the NVFP4 tensor scale inside the product.
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16], ids=str)
def test_verify_rows_keep_their_decode_step_bits(
    hidden, inter, rows, dtype, monkeypatch
):
    mod = _experts(hidden, inter)
    calls = _count_gathers(monkeypatch)
    for seed in range(2):
        x, inds, weights = _block(rows, hidden, seed, dtype)
        z, w = _armed(mod, x, inds, weights)
        # Each row alone, as its decode step runs it, and the block as
        # ``gather_qmm`` runs it (unarmed, 64 pairs and more are sorted by
        # expert, which changes their arithmetic).
        alone = mx.concatenate(
            [
                mod.unreduced(x[r : r + 1], inds[r : r + 1], weights[r : r + 1])[0]
                for r in range(rows)
            ]
        )
        assert mx.array_equal(z, alone)
        if inds.size < 64:
            with monkeypatch.context() as m:
                m.setattr(verify_experts, "PAIRS_PER_EXPERT", 0)
                ref_z, ref_w = mod.unreduced(x, inds, weights)
            assert mx.array_equal(z, ref_z)
            assert mx.array_equal(w, ref_w)
    # w13 and w2 for the block and for each row alone, twice.
    assert len(calls) == 2 * 2 * (1 + rows)


@pytest.mark.parametrize("case", ["unarmed", "wide", "mxfp8", "fast_k"])
def test_other_blocks_keep_gather_qmm(case, monkeypatch):
    # MXFP8 groups are 32 wide; K = 512 is MLX's fast one-row kernel.
    hidden, inter = {"fast_k": (512, 144), "mxfp8": (320, 160)}.get(case, (304, 144))
    mod = _experts(hidden, inter, "mxfp8" if case == "mxfp8" else "nvfp4")
    calls = _count_gathers(monkeypatch)
    # Unarmed, from four pairs per expert on (gather_qmm's batched kernel).
    rows = {
        "wide": verify_experts.MAX_ROWS + 1,
        "unarmed": verify_experts.PAIRS_PER_EXPERT * _EXPERTS // _TOP_K,
    }.get(case, 3)
    x, inds, weights = _block(rows, hidden, 5)
    if case == "unarmed":
        z, _ = mod.unreduced(x, inds, weights)
    else:
        z, _ = _armed(mod, x, inds, weights)
    mx.eval(z)
    assert not calls
    assert z.shape == (rows, _TOP_K, hidden)


# 40 experts, 4 per row: 3 rows (12 pairs) run unsorted, 20 and 39 rows (80
# and 156 pairs, under four per expert) sorted, all one pair at a time.
@pytest.mark.parametrize("rows", [3, 20, 39])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16], ids=str)
def test_short_blocks_take_the_kernel_with_gather_qmm_bits(rows, dtype, monkeypatch):
    # A short unarmed prompt chunk, such as the suffix after a prefix cache
    # hit, below gather_qmm's batched kernel.
    experts = 40
    mod = _experts(304, 144, experts=experts)
    x, inds, weights = _block(rows, 304, 11, dtype, experts=experts)
    # gather_qmm's own path: no block counts as short.
    monkeypatch.setattr(verify_experts, "PAIRS_PER_EXPERT", 0)
    ref_z, ref_w = mod.unreduced(x, inds, weights)
    mx.eval(ref_z, ref_w)
    monkeypatch.undo()
    calls = _count_gathers(monkeypatch)
    z, w = mod.unreduced(x, inds, weights)
    mx.eval(z, w)
    assert len(calls) == 2
    assert mx.array_equal(z, ref_z)
    assert mx.array_equal(w, ref_w)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16], ids=str)
def test_decode_step_takes_the_kernel_with_gather_qmm_bits(dtype, monkeypatch):
    # A one-row step (decode, armed or not) runs its 8 pairs in expert order
    # too; its bits are gather_qmm's, so serial decode does not change.
    mod = _experts(2304, 1152)
    x, inds, weights = _block(1, 2304, 7, dtype)
    xe = mx.expand_dims(x, (-2, -3))
    ref = []
    for proj, inp in ((mod.w13, xe), (mod.w2, None)):
        if inp is None:
            y = ref[0]
            s13 = mod.w13.post_scale(inds, dtype)
            if s13 is not None:
                s13 = s13[..., None, None].astype(y.dtype)
            inp = inkling2._softcapped_swiglu(
                y[..., 0::2], y[..., 1::2], mod._gcap, mod._ucap, s13
            )
        ref.append(proj(inp, inds, False))
    calls = _count_gathers(monkeypatch)
    z, _ = mod.unreduced(x, inds, weights)
    mx.eval(z)
    assert len(calls) == 2
    assert mx.array_equal(z, ref[1].squeeze(-2))


@pytest.mark.parametrize("table", ["threadgroup", "constant"])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16], ids=str)
@pytest.mark.parametrize("rows", [1, 3])
def test_both_code_tables_keep_gather_qmm_bits(table, dtype, rows, monkeypatch):
    # M1/M2 read the codes through a threadgroup byte table, later GPUs
    # through a constant nibble table; both must give gather_qmm's bits.
    monkeypatch.setattr(
        verify_experts, "_threadgroup_table", lambda: table == "threadgroup"
    )
    verify_experts._kernel.cache_clear()
    try:
        mod = _experts(2304, 1152)
        x, inds, weights = _block(rows, 2304, 13, dtype)
        z, _ = _armed(mod, x, inds, weights)
        set_verify_qmm_armed(False)
        monkeypatch.setattr(verify_experts, "supported", lambda proj, dt: False)
        ref, _ = mod.unreduced(x, inds, weights)
        assert mx.array_equal(z, ref)
    finally:
        verify_experts._kernel.cache_clear()
