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


def _experts(hidden, inter, fmt="nvfp4"):
    args = inkling2.TextArgs(
        hidden_size=hidden,
        intermediate_size=inter,
        n_routed_experts=_EXPERTS,
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
        for _ in range(_EXPERTS):
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


def _block(rows, hidden, seed, dtype=mx.bfloat16):
    """Activations, routing and weights of a verify block whose rows share
    experts: each row keeps half of the previous row's experts."""
    rng = np.random.default_rng(seed)
    mx.random.seed(seed)
    x = (mx.random.normal((rows, hidden)) * 0.7).astype(dtype)
    ids = [rng.permutation(_EXPERTS)[:_TOP_K]]
    for _ in range(rows - 1):
        keep = rng.permutation(ids[-1])[: _TOP_K // 2]
        rest = [e for e in rng.permutation(_EXPERTS) if e not in keep]
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
def test_verify_rows_keep_their_decode_step_bits(hidden, inter, rows, monkeypatch):
    mod = _experts(hidden, inter)
    calls = _count_gathers(monkeypatch)
    for seed in range(2):
        x, inds, weights = _block(rows, hidden, seed)
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
            ref_z, ref_w = mod.unreduced(x, inds, weights)
            assert mx.array_equal(z, ref_z)
            assert mx.array_equal(w, ref_w)
    assert len(calls) == 4  # w13 and w2, twice


@pytest.mark.parametrize(
    "case", ["unarmed", "one_row", "wide", "float16", "mxfp8", "fast_k"]
)
def test_other_blocks_keep_gather_qmm(case, monkeypatch):
    # MXFP8 groups are 32 wide; K = 512 is MLX's fast one-row kernel.
    hidden, inter = {"fast_k": (512, 144), "mxfp8": (320, 160)}.get(case, (304, 144))
    mod = _experts(hidden, inter, "mxfp8" if case == "mxfp8" else "nvfp4")
    calls = _count_gathers(monkeypatch)
    rows = {"one_row": 1, "wide": verify_experts.MAX_ROWS + 1}.get(case, 3)
    dtype = mx.float16 if case == "float16" else mx.bfloat16
    x, inds, weights = _block(rows, hidden, 5, dtype)
    if case == "unarmed":
        z, _ = mod.unreduced(x, inds, weights)
    else:
        z, _ = _armed(mod, x, inds, weights)
    mx.eval(z)
    assert not calls
    assert z.shape == (rows, _TOP_K, hidden)
