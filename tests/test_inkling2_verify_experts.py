# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 row-exact verify experts in expert order: each (row, expert) pair
keeps the bits ``gather_qmm`` gives it in the row's one-row decode step."""

from __future__ import annotations

from functools import cache

import mlx.core as mx
import numpy as np
import pytest

from omlx.custom_kernels.nax import is_nax_available
from omlx.patches.inkling2 import expert_mma, verify_experts
from omlx.patches.inkling2 import model as inkling2
from omlx.patches.qwen35_verify_qmm import set_verify_qmm_armed

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

_EXPERTS, _TOP_K = 10, 4


@cache
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


def _count_gathers(monkeypatch, kernel=verify_experts):
    calls = []
    gather = kernel.gather

    def counted(*args, **kwargs):
        calls.append(1)
        return gather(*args, **kwargs)

    monkeypatch.setattr(kernel, "gather", counted)
    return calls


# K = 304 leaves 48 values past the last full 256-value block (lanes 0 to 5),
# K = 144 and 80 have no full block, 2304 and 1152 are Inkling 2 Mini's.
@pytest.mark.parametrize(
    "hidden, inter, rows, dtype",
    [
        (304, 144, 3, mx.bfloat16),
        (272, 80, 16, mx.float16),
        (2304, 1152, 16, mx.bfloat16),
        (2304, 1152, 3, mx.float16),
    ],
    ids=str,
)
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
    # Unarmed from four pairs per expert on (gather_qmm's batched kernel).
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


@pytest.mark.parametrize("table", [True, False], ids=["threadgroup", "plain"])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16], ids=str)
def test_both_code_decodes_keep_gather_qmm_bits(table, dtype, monkeypatch):
    # M1/M2 decode the codes through a threadgroup table, later GPUs one value
    # at a time; both must keep gather_qmm's bits on every host.
    monkeypatch.setattr(verify_experts, "_threadgroup_table", lambda: table)
    verify_experts._kernel.cache_clear()
    try:
        mod = _experts(2304, 1152)
        x, inds, weights = _block(3, 2304, 13, dtype)
        z, _ = _armed(mod, x, inds, weights)
        monkeypatch.setattr(verify_experts, "supported", lambda proj, dt: False)
        ref, _ = mod.unreduced(x, inds, weights)
        assert mx.array_equal(z, ref)
    finally:
        verify_experts._kernel.cache_clear()


# 40 experts, 4 per row: 3 rows run unsorted, 39 rows sorted, both one pair
# at a time in gather_qmm (under four pairs per expert).
@pytest.mark.parametrize("rows", [3, 39])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16], ids=str)
def test_short_blocks_take_the_kernel_with_gather_qmm_bits(rows, dtype, monkeypatch):
    mod = _experts(304, 144, experts=40)
    x, inds, weights = _block(rows, 304, 11, dtype, experts=40)
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


def _affine_experts(bits, dtype):
    args = inkling2.TextArgs(
        hidden_size=2304,
        intermediate_size=1152,
        n_routed_experts=_EXPERTS,
        num_experts_per_tok=_TOP_K,
    )
    spec = {"mode": "affine", "bits": bits, "group_size": 64}
    mod = inkling2.Inkling2Experts(args, spec, spec)
    mx.random.seed(0)
    for proj in (mod.w13, mod.w2):
        e, n, _ = proj.weight.shape
        w = mx.random.normal((e, n, proj.scales.shape[-1] * 64)) * 0.05
        q, s, b = mx.quantize(w, group_size=64, bits=bits)
        proj.weight, proj.scales, proj.biases = q, s.astype(dtype), b.astype(dtype)
    return mod


def _prefill_block(dtype):
    x, inds, weights = _block(70, 2304, 21, dtype, experts=8)
    return x, mx.array([0, 1, 2, 4, 5, 6, 8, 9], dtype=mx.uint32)[inds], weights


@pytest.mark.skipif(is_nax_available(), reason="NAX hosts keep gather_qmm")
def test_prefill_chunks_take_the_q3_kernel_with_gather_qmm_bits(monkeypatch):
    mod = _affine_experts(3, mx.float16)
    x, inds, weights = _prefill_block(mx.float16)
    calls = _count_gathers(monkeypatch, expert_mma)
    z, _ = mod.unreduced(x, inds, weights)
    monkeypatch.setattr(expert_mma, "supported", lambda proj, dtype: False)
    ref, _ = mod.unreduced(x, inds, weights)
    assert len(calls) == 2
    assert mx.array_equal(z, ref)


@pytest.mark.parametrize(
    "bits, dtype, nax",
    [(3, mx.bfloat16, False), (4, mx.float16, False), (3, mx.float16, True)],
    ids=["bfloat16", "bits4", "nax"],
)
def test_other_prefill_chunks_keep_gather_qmm(bits, dtype, nax, monkeypatch):
    monkeypatch.setattr(expert_mma, "is_nax_available", lambda: nax)
    mod = _affine_experts(bits, dtype)
    x, inds, weights = _prefill_block(dtype)
    calls = _count_gathers(monkeypatch, expert_mma)
    mx.eval(mod.unreduced(x, inds, weights)[0])
    assert not calls
