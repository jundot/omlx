# SPDX-License-Identifier: Apache-2.0
"""Tests for the runtime-compiled NAX sorted gather_qmm (m5_gather_qmm_nax).

Covers the plain sorted gather, the gate/up variant with the SwiGLU in its
epilogue (``sorted_gather_qmm_swiglu``, routed through
``m5_gather_qmm.fused_gate_up_activation``) and its row-mapped form that
reads token rows in place (``moe_routes.sort_routes``). Every variant must be
bit-identical to the path it replaces and fall back to it when unsupported.
"""

from __future__ import annotations

import importlib
import io

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import omlx.patches.m5_gather_qmm as patch_mod
import omlx.patches.m5_gather_qmm_nax as nax
from omlx.patches import moe_gate_up_fusion as fusion
from omlx.patches.m5_gather_qmm import apply_m5_gather_qmm_workaround
from omlx.patches.moe_routes import sort_routes


def _on_nax() -> bool:
    if not mx.metal.is_available():
        return False
    try:
        from omlx.custom_kernels.nax import is_nax_available

        return bool(is_nax_available())
    except Exception:  # noqa: BLE001
        return False


needs_nax = pytest.mark.skipif(not _on_nax(), reason="needs an M5 (NAX) GPU")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("OMLX_M5_GATHER_QMM_NAX", raising=False)
    monkeypatch.delenv("OMLX_M5_GATHER_QMM_NAX_PLAN", raising=False)


def _stock():
    fn = mx.gather_qmm
    if getattr(fn, "_omlx_m5_reroute", False):
        fn = patch_mod._original_gather_qmm
    return fn


def _quantized(E, N, K, mode, bits, gs, dtype, seed=0):
    w = (mx.random.normal((E, N, K), key=mx.random.key(seed)) * 0.05).astype(dtype)
    if mode == "affine":
        wq, scales, biases = mx.quantize(w, group_size=gs, bits=bits)
        wd = mx.dequantize(wq, scales, biases, group_size=gs, bits=bits)
    else:
        wq, scales = mx.quantize(w, group_size=gs, bits=bits, mode=mode)
        biases = None
        wd = mx.dequantize(wq, scales, group_size=gs, bits=bits, mode=mode)
    return wq, scales, biases, wd


def _rows(counts, K, dtype, seed=1):
    idx = mx.array(np.repeat(np.arange(len(counts)), counts).astype(np.uint32))
    x = (mx.random.normal((int(idx.shape[0]), 1, K), key=mx.random.key(seed)) * 0.5).astype(
        dtype
    )
    return x, idx


def _routed_rows(tokens, top_k, E, K, dtype, skew, seed=2, x_scale=0.5):
    """SwitchGLU-style sorted rows: top-k routing, flattened, sorted."""
    key = mx.random.key(seed)
    if skew:
        p = 1.0 / (mx.arange(E) + 5.0) ** skew
        scores = mx.log(p)[None, :] + mx.random.gumbel(shape=(tokens, E), key=key)
    else:
        scores = mx.random.uniform(shape=(tokens, E), key=key)
    inds = mx.argpartition(-scores, kth=top_k - 1, axis=-1)[:, :top_k].astype(mx.uint32)
    flat = inds.flatten()
    order = mx.argsort(flat)
    x = (mx.random.normal((tokens, K), key=mx.random.key(seed + 1)) * x_scale).astype(
        dtype
    )
    return x[order // top_k][:, None, :], flat[order]


def _nax(x, wq, scales, biases, idx, mode, bits, gs, plan=None):
    out = nax.sorted_gather_qmm(
        x, wq, scales, biases, idx, group_size=gs, bits=bits, mode=mode, plan=plan
    )
    assert out is not None
    return out


def _stock_sorted(x, wq, scales, biases, idx, mode, bits, gs):
    return _stock()(
        x,
        wq,
        scales,
        biases,
        rhs_indices=idx,
        transpose=True,
        group_size=gs,
        bits=bits,
        mode=mode,
        sorted_indices=True,
    )


def _fp32_ref(x, wd, idx):
    return x.astype(mx.float32) @ wd[idx].swapaxes(-1, -2).astype(mx.float32)


# ---------------------------------------------------------------------------
# Support gating (no GPU work)
# ---------------------------------------------------------------------------


def test_supports_gating():
    E, N, K, M = 4, 64, 128, 16
    x = mx.zeros((M, 1, K), dtype=mx.bfloat16)
    idx = mx.zeros((M,), dtype=mx.uint32)
    w4 = mx.zeros((E, N, K // 8), dtype=mx.uint32)
    s64 = mx.zeros((E, N, K // 64), dtype=mx.bfloat16)
    s32u8 = mx.zeros((E, N, K // 32), dtype=mx.uint8)

    assert nax.supports(x, w4, s64, s64, idx, 64, 4, "affine")
    assert nax.supports(x.astype(mx.float16), w4, s64.astype(mx.float16),
                        s64.astype(mx.float16), idx, 64, 4, "affine")
    assert nax.supports(x, w4, s32u8, None, idx, 32, 4, "mxfp4")
    w8 = mx.zeros((E, N, K // 4), dtype=mx.uint32)
    assert nax.supports(x, w8, s64, s64, idx, 64, 8, "affine")

    # fp32 activations, scale dtype mismatch, missing biases.
    assert not nax.supports(x.astype(mx.float32), w4, s64, s64, idx, 64, 4, "affine")
    assert not nax.supports(x, w4, s64.astype(mx.float16), s64, idx, 64, 4, "affine")
    assert not nax.supports(x, w4, s64, None, idx, 64, 4, "affine")
    # Unsupported bit widths / modes / group sizes.
    w3 = mx.zeros((E, N, K * 3 // 32), dtype=mx.uint32)
    assert not nax.supports(x, w3, s64, s64, idx, 64, 3, "affine")
    assert not nax.supports(x, w4, s32u8, None, idx, 32, 4, "nvfp4")
    s16 = mx.zeros((E, N, K // 16), dtype=mx.uint8)
    assert not nax.supports(x, w4, s16, None, idx, 16, 4, "mxfp4")
    # Layout: 2-D rows, rows per index != 1, int32 indices, too few rows.
    assert not nax.supports(x.reshape(M, K), w4, s64, s64, idx, 64, 4, "affine")
    x2 = mx.zeros((M // 2, 2, K), dtype=mx.bfloat16)
    assert not nax.supports(x2, w4, s64, s64, idx[: M // 2], 64, 4, "affine")
    assert not nax.supports(x, w4, s64, s64, idx.astype(mx.int32), 64, 4, "affine")
    assert not nax.supports(x[:4], w4, s64, s64, idx[:4], 64, 4, "affine")
    # Shape mismatch between w and K.
    assert not nax.supports(x, w8, s64, s64, idx, 64, 4, "affine")
    # Ragged N: only N % 64 == 32 is canaried.
    w96 = mx.zeros((E, 96, K // 8), dtype=mx.uint32)
    s96 = mx.zeros((E, 96, K // 64), dtype=mx.bfloat16)
    assert nax.supports(x, w96, s96, s96, idx, 64, 4, "affine")
    w100 = mx.zeros((E, 100, K // 8), dtype=mx.uint32)
    s100 = mx.zeros((E, 100, K // 64), dtype=mx.bfloat16)
    assert not nax.supports(x, w100, s100, s100, idx, 64, 4, "affine")


def test_kill_switch(monkeypatch):
    monkeypatch.setenv("OMLX_M5_GATHER_QMM_NAX", "0")
    x = mx.zeros((16, 1, 128), dtype=mx.bfloat16)
    w = mx.zeros((4, 64, 16), dtype=mx.uint32)
    s = mx.zeros((4, 64, 2), dtype=mx.bfloat16)
    idx = mx.zeros((16,), dtype=mx.uint32)
    assert not nax.enabled()
    assert nax.sorted_gather_qmm(x, w, s, s, idx, group_size=64, bits=4) is None


# ---------------------------------------------------------------------------
# Exactness on NAX hardware
# ---------------------------------------------------------------------------

# Empty experts, runs spanning several 64-row tiles, partial tiles of all
# sizes (1..63 rows), a single-row run.
_COUNTS = (70, 0, 5, 33, 64, 17, 1, 130, 0, 11)

_ALIGNED = [
    ("affine", 4, 64, mx.bfloat16),
    ("affine", 4, 32, mx.bfloat16),
    ("affine", 4, 128, mx.bfloat16),
    ("affine", 8, 64, mx.bfloat16),
    ("affine", 8, 32, mx.float16),
    ("affine", 4, 64, mx.float16),
    ("mxfp4", 4, 32, mx.bfloat16),
    ("mxfp4", 4, 32, mx.float16),
]


P = nax.Plan
SEG, DB = nax._SCHED_SEG, nax._SCHED_DB

# Every configuration _plan picks, plus the plain layouts of each schedule
# and tile height, a 64-deep seg with 128-row tiles and a small x group.
_PLANS = [
    P(SEG, 64, 64, 0, 0),
    P(DB, 64, 64, 0, 0),
    P(DB, 64, 64, 32, 0),
    P(DB, 96, 64, 32, 0),
    P(SEG, 96, 128, 32, 0),
    P(SEG, 128, 128, 32, 8192),
    P(SEG, 128, 64, 0, 0),
    P(DB, 128, 64, 0, 0),
    P(SEG, 96, 64, 3, 0),
]


def _plan_id(plan):
    return plan.describe().replace(" ", "-")


@pytest.fixture(params=_PLANS, ids=_plan_id)
def forced_plan(request, monkeypatch):
    """Run a test with every configuration pinned through the env override."""
    plan = request.param
    sched = "seg" if plan.sched == SEG else "db"
    monkeypatch.setenv(
        "OMLX_M5_GATHER_QMM_NAX_PLAN",
        f"{sched},{plan.bm},{plan.bk},{plan.gx},{plan.pad}",
    )
    return plan


@needs_nax
@pytest.mark.parametrize("mode,bits,gs,dtype", _ALIGNED)
@pytest.mark.parametrize("plan", _PLANS, ids=_plan_id)
@pytest.mark.parametrize("K", [256, 384])
def test_bit_identical_to_stock_sorted_kernel(mode, bits, gs, dtype, plan, K):
    """K % 64 == 0: same dequantization and tensor-op order as mlx's kernel.

    K = 384 leaves a 64-deep tail after the 128-deep K steps.
    """
    E, N = len(_COUNTS), 128
    wq, scales, biases, _ = _quantized(E, N, K, mode, bits, gs, dtype)
    x, idx = _rows(_COUNTS, K, dtype)
    out = _nax(x, wq, scales, biases, idx, mode, bits, gs, plan)
    ref = _stock_sorted(x, wq, scales, biases, idx, mode, bits, gs)
    assert out.shape == ref.shape and out.dtype == ref.dtype
    assert mx.array_equal(out, ref).item()


@needs_nax
@pytest.mark.parametrize("mode,gs", [("affine", 64), ("mxfp4", 32)])
@pytest.mark.parametrize("skew", [0.0, 1.2])
def test_routed_rows_match_stock(mode, gs, skew):
    """SwitchGLU routing (uniform and skewed, empty experts) at MoE shapes."""
    E, N, K = 64, 192, 1152
    wq, scales, biases, _ = _quantized(E, N, K, mode, 4, gs, mx.bfloat16, seed=5)
    for tokens in (160, 600, 1200):  # 20, 75 and 150 rows per expert
        x, idx = _routed_rows(tokens, 8, E, K, mx.bfloat16, skew)
        ref = _stock_sorted(x, wq, scales, biases, idx, mode, 4, gs)
        for plan in [None] + _PLANS:
            out = _nax(x, wq, scales, biases, idx, mode, 4, gs, plan)
            assert mx.array_equal(out, ref).item(), f"tokens={tokens} plan={plan}"


@needs_nax
@pytest.mark.parametrize(
    "mode,bits,dtype",
    [("affine", 4, mx.bfloat16), ("affine", 8, mx.float16), ("mxfp4", 4, mx.bfloat16)],
)
@pytest.mark.parametrize("K", [32, 96, 544])
def test_ragged_k_matches_fp32_reference(mode, bits, dtype, K, forced_plan):
    """K % 64 == 32 (group 32): the stock kernel's tail is wrong here."""
    E, N = len(_COUNTS), 128
    wq, scales, biases, wd = _quantized(E, N, K, mode, bits, 32, dtype)
    x, idx = _rows(_COUNTS, K, dtype)
    out = _nax(x, wq, scales, biases, idx, mode, bits, 32)
    ref = _fp32_ref(x, wd, idx)
    err = mx.abs(out.astype(mx.float32) - ref)
    # bf16/fp16 output rounding of an fp32-accumulated dot product.
    tol = mx.abs(ref) * (2.0**-7 if dtype == mx.bfloat16 else 2.0**-10) + 1e-3
    assert mx.all(err <= tol).item(), f"max err {err.max().item()}"


@needs_nax
def test_ragged_k_tail_never_reads_past_the_row():
    """The K tail zero-fills the weight tile instead of dequantizing past K.

    The scales/biases are views whose buffer continues with NaN right after
    the last expert's last row: a kernel that dequantizes the tail block
    past K (as mlx's fixed kernel does) turns that into NaN * 0 = NaN.
    """
    E, N, K = 4, 64, 96
    wq, scales, biases, wd = _quantized(E, N, K, "affine", 4, 32, mx.bfloat16)
    nan = mx.full((64,), float("nan"), dtype=mx.bfloat16)
    s_view = mx.concatenate([scales.reshape(-1), nan])[: scales.size].reshape(scales.shape)
    b_view = mx.concatenate([biases.reshape(-1), nan])[: biases.size].reshape(biases.shape)
    x, idx = _rows((0, 0, 0, 40), K, mx.bfloat16)
    out = _nax(x, wq, s_view, b_view, idx, "affine", 4, 32)
    assert not mx.any(mx.isnan(out)).item()
    ref = _fp32_ref(x, wd, idx)
    assert mx.abs(out.astype(mx.float32) - ref).max().item() < 0.05


@needs_nax
@pytest.mark.parametrize("mode,gs", [("affine", 64), ("mxfp4", 32)])
@pytest.mark.parametrize("K", [256, 384])
def test_ragged_n_matches_stock(mode, gs, K, forced_plan):
    E, N = len(_COUNTS), 96
    wq, scales, biases, _ = _quantized(E, N, K, mode, 4, gs, mx.bfloat16)
    x, idx = _rows(_COUNTS, K, mx.bfloat16)
    out = _nax(x, wq, scales, biases, idx, mode, 4, gs)
    ref = _stock_sorted(x, wq, scales, biases, idx, mode, 4, gs)
    assert mx.array_equal(out, ref).item()


@needs_nax
@pytest.mark.parametrize("mode,gs", [("affine", 64), ("mxfp4", 32)])
def test_more_than_32768_rows(mode, gs):
    """One call past the stock kernel's int16 row-offset limit.

    Every row is independent, so the stock kernel run on <= 32768-row
    slices (where it is correct) is an exact reference.
    """
    E, N, K = 16, 64, 128
    counts = tuple([2304] * 15 + [4000])  # 38560 rows, heavy last expert
    wq, scales, biases, _ = _quantized(E, N, K, mode, 4, gs, mx.bfloat16)
    x, idx = _rows(counts, K, mx.bfloat16)
    rows = int(idx.shape[0])
    assert rows > 32768
    half = rows // 2
    ref = mx.concatenate(
        [
            _stock_sorted(x[:half], wq, scales, biases, idx[:half], mode, 4, gs),
            _stock_sorted(x[half:], wq, scales, biases, idx[half:], mode, 4, gs),
        ]
    )
    for plan in [None] + _PLANS:
        out = _nax(x, wq, scales, biases, idx, mode, 4, gs, plan)
        assert mx.array_equal(out, ref).item(), f"plan={plan}"


@needs_nax
def test_single_expert_and_all_experts_empty_but_one():
    E, N, K = 32, 128, 128
    wq, scales, biases, _ = _quantized(E, N, K, "affine", 4, 64, mx.bfloat16)
    # >= 4 rows per expert overall, so mlx picks its sorted rhs kernel too.
    for counts in ((0,) * 31 + (300,), (129,) + (0,) * 31):
        x, idx = _rows(counts, K, mx.bfloat16)
        ref = _stock_sorted(x, wq, scales, biases, idx, "affine", 4, 64)
        for plan in [None] + _PLANS:
            out = _nax(x, wq, scales, biases, idx, "affine", 4, 64, plan)
            assert mx.array_equal(out, ref).item(), f"plan={plan}"


@needs_nax
@pytest.mark.parametrize("bm", [64, 96, 128])
def test_tile_scan_descriptors(bm):
    """(row_start, expert, rows) tiles, expert-major, <= bm rows each."""
    counts = (70, 0, 5, 33, 64, 17, 1, 130, 0, 11)
    idx = mx.array(np.repeat(np.arange(len(counts)), counts).astype(np.uint32))
    M, E = int(idx.shape[0]), len(counts)
    max_tiles = (M + bm - 1) // bm + E
    tiles, count = nax._get_kernel("scan")(
        inputs=[idx, mx.array([M, E, max_tiles], dtype=mx.int32)],
        template=[("BM", bm), ("MAXE", nax._MAX_EXPERTS)],
        grid=(1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(max_tiles * 4,), (1,)],
        output_dtypes=[mx.uint32, mx.uint32],
    )
    expect = []
    start = 0
    for e, n in enumerate(counts):
        for r in range(start, start + n, bm):
            expect.append((r, e, min(bm, start + n - r), 0))
        start += n
    n_tiles = int(count[0].item())
    assert n_tiles == len(expect)
    got = np.array(tiles).reshape(-1, 4)[:n_tiles]
    assert [tuple(int(v) for v in t) for t in got] == expect


@needs_nax
@pytest.mark.parametrize("bm", [64, 96])
def test_tile_scan_randomized(bm):
    """Row counts with every M % 4, clustered and sparse expert use."""
    rng = np.random.default_rng(7)
    for trial in range(40):
        E = int(rng.integers(1, 600))
        M = int(rng.integers(8, 3000))
        pool = rng.integers(0, E, max(1, E // 5)) if trial % 2 else np.arange(E)
        idx_np = np.sort(rng.choice(pool, M)).astype(np.uint32)
        max_tiles = (M + bm - 1) // bm + min(E, M)
        tiles, count = nax._get_kernel("scan")(
            inputs=[mx.array(idx_np), mx.array([M, E, max_tiles], dtype=mx.int32)],
            template=[("BM", bm), ("MAXE", nax._MAX_EXPERTS)],
            grid=(1024, 1, 1),
            threadgroup=(1024, 1, 1),
            output_shapes=[(max_tiles * 4,), (1,)],
            output_dtypes=[mx.uint32, mx.uint32],
        )
        expect = []
        for e in range(E):
            lo = int(np.searchsorted(idx_np, e, "left"))
            hi = int(np.searchsorted(idx_np, e, "right"))
            expect += [(r, e, min(bm, hi - r), 0) for r in range(lo, hi, bm)]
        n_tiles = int(count[0].item())
        got = np.array(tiles).reshape(-1, 4)[:n_tiles]
        assert [tuple(int(v) for v in t) for t in got] == expect, (E, M)


@needs_nax
def test_tile_scan_bounded_on_unsorted_indices():
    """Unsorted input breaks the contract but must not overrun the buffer."""
    idx = mx.array(np.tile(np.arange(8, dtype=np.uint32), 50))
    M, E = int(idx.shape[0]), 8
    max_tiles = (M + 63) // 64 + E
    _, count = nax._get_kernel("scan")(
        inputs=[idx, mx.array([M, E, max_tiles], dtype=mx.int32)],
        template=[("BM", 64), ("MAXE", nax._MAX_EXPERTS)],
        grid=(1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(max_tiles * 4,), (1,)],
        output_dtypes=[mx.uint32, mx.uint32],
    )
    assert int(count[0].item()) <= max_tiles


# ---------------------------------------------------------------------------
# Wrapper routing
# ---------------------------------------------------------------------------


@pytest.fixture
def installed(monkeypatch):
    """The m5 reroute wrapper on mx.gather_qmm (restored afterwards)."""
    was_installed = getattr(mx.gather_qmm, "_omlx_m5_reroute", False)
    raw = patch_mod._original_gather_qmm if was_installed else mx.gather_qmm
    mx.gather_qmm = raw
    monkeypatch.delenv("OMLX_M5_GATHER_QMM_FIX", raising=False)
    assert apply_m5_gather_qmm_workaround()
    yield raw
    mx.gather_qmm = raw
    patch_mod._original_gather_qmm = raw
    if was_installed:
        mx.gather_qmm = patch_mod._gather_qmm_rerouted


@needs_nax
def test_wrapper_routes_sorted_calls_to_nax(installed, monkeypatch):
    calls = []
    real = nax.sorted_gather_qmm

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        calls.append(out is not None)
        return out

    monkeypatch.setattr(nax, "sorted_gather_qmm", spy)
    E, N, K = 8, 128, 96  # ragged K: the stock sorted kernel is wrong here
    wq, scales, biases, wd = _quantized(E, N, K, "affine", 4, 32, mx.bfloat16)
    x, idx = _rows((20, 0, 50, 7, 64, 1, 30, 9), K, mx.bfloat16)
    out = mx.gather_qmm(
        x, wq, scales, biases, rhs_indices=idx, transpose=True, group_size=32, bits=4,
        sorted_indices=True,
    )
    assert calls == [True]
    ref = _fp32_ref(x, wd, idx)
    assert mx.abs(out.astype(mx.float32) - ref).max().item() < 0.05

    # Positional scales/biases and default group size / bits (affine 64/4).
    wq, scales, biases, _ = _quantized(E, N, 128, "affine", 4, 64, mx.bfloat16)
    x, idx = _rows((20, 0, 50, 7, 64, 1, 30, 9), 128, mx.bfloat16)
    out = mx.gather_qmm(x, wq, scales, biases, None, idx, sorted_indices=True)
    ref = _stock_sorted(x, wq, scales, biases, idx, "affine", 4, 64)
    assert calls == [True, True]
    assert mx.array_equal(out, ref).item()

    # Fewer than 4 rows per expert: mlx's qmv path, not the NAX route.
    xs, idxs = _rows((3, 0, 5, 7, 2, 1, 4, 9), 128, mx.bfloat16)
    mx.gather_qmm(xs, wq, scales, biases, rhs_indices=idxs, sorted_indices=True)
    assert calls == [True, True]

    # Unsorted calls and lhs gathers never take the route.
    mx.gather_qmm(x, wq, scales, biases, rhs_indices=idx, group_size=64, bits=4)
    mx.gather_qmm(
        x, wq, scales, biases, lhs_indices=mx.arange(x.shape[0]).astype(mx.uint32),
        rhs_indices=idx, sorted_indices=True,
    )
    assert calls == [True, True]


@needs_nax
def test_wrapper_kill_switch_keeps_stock_path(installed, monkeypatch):
    monkeypatch.setenv("OMLX_M5_GATHER_QMM_NAX", "0")
    seen = []
    monkeypatch.setattr(
        nax, "_launch", lambda *a, **k: seen.append(1) or None
    )
    E, N, K = 8, 128, 128
    wq, scales, biases, _ = _quantized(E, N, K, "affine", 4, 64, mx.bfloat16)
    x, idx = _rows((20, 0, 50, 7, 64, 1, 30, 9), K, mx.bfloat16)
    out = mx.gather_qmm(
        x, wq, scales, biases, rhs_indices=idx, group_size=64, bits=4, sorted_indices=True
    )
    ref = _stock_sorted(x, wq, scales, biases, idx, "affine", 4, 64)
    assert not seen
    assert mx.array_equal(out, ref).item()


@needs_nax
def test_failed_self_test_falls_back(installed, monkeypatch):
    monkeypatch.setattr(nax, "_verified", {})
    monkeypatch.setattr(nax, "_self_test", lambda key: False)
    E, N, K = 8, 128, 128
    wq, scales, biases, _ = _quantized(E, N, K, "affine", 4, 64, mx.bfloat16)
    x, idx = _rows((20, 0, 50, 7, 64, 1, 30, 9), K, mx.bfloat16)
    assert (
        nax.sorted_gather_qmm(x, wq, scales, biases, idx, group_size=64, bits=4) is None
    )
    out = mx.gather_qmm(
        x, wq, scales, biases, rhs_indices=idx, group_size=64, bits=4, sorted_indices=True
    )
    ref = _stock_sorted(x, wq, scales, biases, idx, "affine", 4, 64)
    assert mx.array_equal(out, ref).item()


@needs_nax
@pytest.mark.parametrize("plan", _PLANS, ids=_plan_id)
def test_self_test_passes_for_supported_instantiations(plan):
    for key in [
        (mx.bfloat16, "affine", 4, 64, True, True),
        (mx.bfloat16, "affine", 4, 32, True, False),
        (mx.float16, "affine", 8, 64, False, True),
        (mx.bfloat16, "mxfp4", 4, 32, True, True),
        (mx.bfloat16, "mxfp4", 4, 32, False, False),
    ]:
        if plan.sched == DB and not (key[4] and key[5]):
            continue  # db runs aligned shapes only (seg covers the rest)
        dtype, mode, bits, gs, align_n, align_k = key
        full = (dtype, mode, bits, gs, plan, align_n, align_k)
        assert nax._self_test(full) is True, full


# ---------------------------------------------------------------------------
# Gate/up SwiGLU epilogue (sorted_gather_qmm_swiglu)
# ---------------------------------------------------------------------------

_EPI_FORMATS = [
    ("affine", 4, 64, mx.bfloat16),  # GLM-5.3, Qwen3.8
    ("mxfp4", 4, 32, mx.bfloat16),  # MiMo-V2.6
    ("affine", 4, 32, mx.bfloat16),
    ("affine", 8, 64, mx.float16),
]
# Every pinned plan on the main format, the other formats on the automatic
# plan. Each runtime instantiation also runs its own canary.
_EPI_CASES = [_EPI_FORMATS[0] + (plan,) for plan in _PLANS] + [
    fmt + (None,) for fmt in _EPI_FORMATS[1:]
]
# plain SwiGLU, GLM-5.3's clamp, and a bound that is not a bf16 value
_LIMITS = [None, 10.0, 7.3]


def _case_id(case):
    mode, bits, gs, dtype, plan = case
    name = f"{mode}{bits}-g{gs}-{'bf16' if dtype == mx.bfloat16 else 'fp16'}"
    return f"{name}-{_plan_id(plan) if plan else 'auto'}"


def _wide_rows(counts, K, dtype, seed=1):
    """Sorted rows whose scale spans 0.1x-16x, so the projections reach
    sigmoid's saturated tails and GLM-5.3's clip bounds."""
    idx = mx.array(np.repeat(np.arange(len(counts)), counts).astype(np.uint32))
    M = int(idx.shape[0])
    scale = mx.power(10.0, mx.linspace(-1.0, 1.2, M)).reshape(M, 1, 1)
    x = (mx.random.normal((M, 1, K), key=mx.random.key(seed)) * scale).astype(dtype)
    return x, idx


def _unfused(x, wq, scales, biases, idx, mode, bits, gs, limit, plan=None):
    gate_up = nax.sorted_gather_qmm(
        x, wq, scales, biases, idx, group_size=gs, bits=bits, mode=mode, plan=plan
    )
    assert gate_up is not None
    x_gate, x_up = mx.split(gate_up, 2, axis=-1)
    return nax.reference_activation(x_up, x_gate, limit)


def _fused(x, wq, scales, biases, idx, mode, bits, gs, limit, plan=None):
    out = nax.sorted_gather_qmm_swiglu(
        x, wq, scales, biases, idx, group_size=gs, bits=bits, mode=mode,
        limit=limit, plan=plan,
    )
    assert out is not None
    return out


def _assert_bitwise(got, ref, msg=""):
    assert got.shape == ref.shape and got.dtype == ref.dtype, msg
    view = {2: mx.uint16, 4: mx.uint32}[got.dtype.size]
    assert mx.array_equal(got.view(view), ref.view(view)).item(), msg


@pytest.fixture
def epilogue_calls(monkeypatch):
    """Records whether each sorted_gather_qmm_swiglu call produced output."""
    calls = []
    real = nax.sorted_gather_qmm_swiglu

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        calls.append(out is not None)
        return out

    monkeypatch.setattr(nax, "sorted_gather_qmm_swiglu", spy)
    return calls


@pytest.fixture
def launches(monkeypatch):
    """Records, per epilogue launch, whether it read rows through a map."""
    calls = []
    real = nax._launch

    def spy(*args, **kwargs):
        if kwargs.get("epi"):
            calls.append(kwargs.get("row_map") is not None)
        return real(*args, **kwargs)

    monkeypatch.setattr(nax, "_launch", spy)
    return calls


def _epilogue_off(monkeypatch):
    monkeypatch.setattr(nax, "sorted_gather_qmm_swiglu", lambda *a, **k: None)


def _row_map_off(monkeypatch):
    real = nax.sorted_gather_qmm_swiglu

    def declined(*args, row_map=None, **kwargs):
        return None if row_map is not None else real(*args, **kwargs)

    monkeypatch.setattr(nax, "sorted_gather_qmm_swiglu", declined)


@needs_nax
@pytest.mark.parametrize("case", _EPI_CASES, ids=_case_id)
def test_epilogue_bit_identical_to_unfused(case):
    """K with and without a 64-deep tail after the 128-deep steps, each
    activation; n = 128 (four output column tiles)."""
    mode, bits, gs, dtype, plan = case
    for K in (256, 384):
        wq, scales, biases, _ = _quantized(len(_COUNTS), 256, K, mode, bits, gs, dtype)
        x, idx = _wide_rows(_COUNTS, K, dtype)
        for limit in _LIMITS:
            got = _fused(x, wq, scales, biases, idx, mode, bits, gs, limit, plan)
            ref = _unfused(x, wq, scales, biases, idx, mode, bits, gs, limit, plan)
            assert got.shape == (x.shape[0], 1, 128)
            _assert_bitwise(got, ref, f"K={K} limit={limit}")


@needs_nax
@pytest.mark.parametrize(
    "mode,gs,limit", [("affine", 64, 10.0), ("affine", 64, None), ("mxfp4", 32, None)]
)
@pytest.mark.parametrize("skew", [0.0, 1.2])
def test_epilogue_routed_rows_bit_identical(mode, gs, limit, skew):
    """SwitchGLU routing (uniform and skewed: ragged runs, empty experts)
    at 20, 75 and 150 rows per expert, automatic and every pinned plan."""
    E, n, K = 64, 96, 1152
    wq, scales, biases, _ = _quantized(E, 2 * n, K, mode, 4, gs, mx.bfloat16, seed=5)
    for tokens in (160, 600, 1200):
        x, idx = _routed_rows(tokens, 8, E, K, mx.bfloat16, skew, x_scale=4.0)
        ref = _unfused(x, wq, scales, biases, idx, mode, 4, gs, limit)
        for plan in [None] + _PLANS:
            got = _fused(x, wq, scales, biases, idx, mode, 4, gs, limit, plan)
            _assert_bitwise(got, ref, f"tokens={tokens} plan={plan}")


@needs_nax
@pytest.mark.parametrize("mode,gs", [("affine", 32), ("mxfp4", 32)])
@pytest.mark.parametrize("K", [96, 544])
def test_epilogue_ragged_k_bit_identical(mode, gs, K):
    """K % 64 == 32: the epilogue kernel runs the same tail sub-steps."""
    wq, scales, biases, _ = _quantized(len(_COUNTS), 128, K, mode, 4, gs, mx.bfloat16)
    x, idx = _wide_rows(_COUNTS, K, mx.bfloat16)
    for plan in [None] + _PLANS:
        for limit in (None, 10.0):
            got = _fused(x, wq, scales, biases, idx, mode, 4, gs, limit, plan)
            ref = _unfused(x, wq, scales, biases, idx, mode, 4, gs, limit, plan)
            _assert_bitwise(got, ref, f"plan={plan} limit={limit}")


@needs_nax
@pytest.mark.parametrize("mode,gs,limit", [("affine", 64, 10.0), ("mxfp4", 32, None)])
def test_epilogue_more_than_32768_rows(mode, gs, limit):
    """One call past 32768 sorted rows (32-bit row offsets in the epilogue)."""
    E, n, K = 16, 64, 128
    counts = tuple([2304] * 15 + [4000])  # 38560 rows, heavy last expert
    wq, scales, biases, _ = _quantized(E, 2 * n, K, mode, 4, gs, mx.bfloat16)
    x, idx = _wide_rows(counts, K, mx.bfloat16)
    assert int(idx.shape[0]) > 32768
    ref = _unfused(x, wq, scales, biases, idx, mode, 4, gs, limit)
    for plan in [None] + _PLANS:
        got = _fused(x, wq, scales, biases, idx, mode, 4, gs, limit, plan)
        _assert_bitwise(got, ref, f"plan={plan}")


@needs_nax
@pytest.mark.parametrize("limit", [None, 10.0])
def test_epilogue_non_finite_projections(limit):
    """Rows that overflow the accumulators (inf, inf - inf = NaN) take the
    same activation ops as in the unfused kernel."""
    K = 256
    wq, scales, biases, _ = _quantized(
        len(_COUNTS), 128, K, "affine", 4, 64, mx.bfloat16
    )
    x, idx = _wide_rows(_COUNTS, K, mx.bfloat16)
    big = mx.array([3.0e38, -3.0e38, float("inf"), float("-inf")], dtype=mx.bfloat16)
    x = mx.concatenate([x[:40], mx.tile(big, (32, 1, K // 4)), x[72:]], axis=0)
    assert x.shape[0] == idx.shape[0]
    got = _fused(x, wq, scales, biases, idx, "affine", 4, 64, limit)
    ref = _unfused(x, wq, scales, biases, idx, "affine", 4, 64, limit)
    assert not mx.all(mx.isfinite(ref)).item()
    _assert_bitwise(got, ref)


def _glm5_language():
    from omlx.patches import mlx_vlm_glm5_next_compat as compat

    compat.apply_mlx_vlm_glm5_next_compat_patch()
    return importlib.import_module("mlx_vlm.models.glm5_next.language")


def test_reference_activation_matches_the_model_activations():
    """The self-test reference is the models' activation, bit for bit."""
    from mlx_lm.models.activations import swiglu as lm_swiglu
    from mlx_lm.models.switch_layers import SwiGLU as LmSwiGLU
    from mlx_vlm.models.switch_layers import SwiGLU as VlmSwiGLU

    from omlx.patches.deepseek_v4.switch_layers import SwiGLU as V4SwiGLU
    from omlx.patches.glm_moe_dsa.switch_layers import SwiGLU as DsaSwiGLU

    lang = _glm5_language()
    gate_up = (mx.random.normal((3000, 1, 256), key=mx.random.key(3)) * 8).astype(
        mx.bfloat16
    )
    x_gate, x_up = mx.split(gate_up, 2, axis=-1)
    plain = nax.reference_activation(x_up, x_gate)
    for act in (LmSwiGLU(), VlmSwiGLU(), V4SwiGLU(), DsaSwiGLU()):
        _assert_bitwise(act(x_up, x_gate), plain, type(act).__module__)
    _assert_bitwise(lm_swiglu(x_gate, x_up), plain)
    _assert_bitwise(lang.Glm5NextClampedSwiGLU(None)(x_up, x_gate), plain)
    for limit in (10.0, 7.3):
        _assert_bitwise(
            lang.Glm5NextClampedSwiGLU(limit)(x_up, x_gate),
            nax.reference_activation(x_up, x_gate, limit),
            f"limit={limit}",
        )


def test_epilogue_unsupported_calls_return_none(monkeypatch):
    E, K, M = 4, 128, 64
    x = mx.zeros((M, 1, K), dtype=mx.bfloat16)
    idx = mx.zeros((M,), dtype=mx.uint32)

    def call(n, **kw):
        w = mx.zeros((E, 2 * n, K // 8), dtype=mx.uint32)
        s = mx.zeros((E, 2 * n, K // 64), dtype=mx.bfloat16)
        args = dict(group_size=64, bits=4, verify=False)
        args.update(kw)
        xx = args.pop("x", x)
        return nax.sorted_gather_qmm_swiglu(xx, w, s, s, idx, **args)

    # [gate; up] must fill whole 64-column tiles: n % 32 == 0.
    assert call(48) is None
    assert call(16) is None
    # non-finite clip bound, fp32 activations
    assert call(64, limit=float("inf")) is None
    assert call(64, limit=float("nan")) is None
    assert call(64, x=x.astype(mx.float32)) is None
    monkeypatch.setenv("OMLX_M5_GATHER_QMM_NAX", "0")
    assert call(64) is None


@needs_nax
def test_epilogue_failed_self_tests_fall_back(monkeypatch):
    E, n, K = len(_COUNTS), 64, 128
    wq, scales, biases, _ = _quantized(E, 2 * n, K, "affine", 4, 64, mx.bfloat16)
    x, idx = _wide_rows(_COUNTS, K, mx.bfloat16)

    def call():
        return nax.sorted_gather_qmm_swiglu(
            x, wq, scales, biases, idx, group_size=64, bits=4, limit=10.0
        )

    monkeypatch.setattr(nax, "_verified", {})
    monkeypatch.setattr(nax, "_self_test_act", lambda key: False)
    assert call() is None
    # The plain instantiation (the fallback's kernel) failing blocks it too.
    monkeypatch.setattr(nax, "_verified", {})
    monkeypatch.setattr(nax, "_self_test_act", lambda key: True)
    monkeypatch.setattr(nax, "_self_test", lambda key: False)
    assert call() is None
    # Undecided (e.g. first call inside a function transformation): retried.
    monkeypatch.setattr(nax, "_verified", {})
    monkeypatch.setattr(nax, "_self_test", lambda key: True)
    monkeypatch.setattr(nax, "_self_test_act", lambda key: None)
    assert call() is None
    assert not any(len(k) > 7 for k in nax._verified)
    monkeypatch.setattr(nax, "_self_test_act", lambda key: True)
    assert call() is not None


def test_activation_classes():
    from mlx_lm.models.switch_layers import SwiGLU as LmSwiGLU
    from mlx_vlm.models.switch_layers import SwiGLU as VlmSwiGLU

    from omlx.patches.deepseek_v4.switch_layers import SwiGLU as V4SwiGLU
    from omlx.patches.glm_moe_dsa.switch_layers import SwiGLU as DsaSwiGLU

    lang = _glm5_language()
    for act in (LmSwiGLU(), VlmSwiGLU(), V4SwiGLU(), DsaSwiGLU()):
        assert patch_mod._swiglu_limit(act) is None
    assert patch_mod._swiglu_limit(lang.Glm5NextClampedSwiGLU(10)) == 10.0
    assert patch_mod._swiglu_limit(lang.Glm5NextClampedSwiGLU(None)) is None

    class SubSwiGLU(DsaSwiGLU):  # a subclass may compute something else
        pass

    for act in (nn.GELU(), SubSwiGLU(), lambda up, gate: up * gate):
        assert patch_mod._swiglu_limit(act) is patch_mod._UNSUPPORTED


def _dsa_gate_up(mode="affine", gs=64, E=16, n=64, K=128, bias=False):
    from omlx.patches.glm_moe_dsa import switch_layers as dsa

    lin = dsa.SwitchLinear(K, 2 * n, E, bias=bias)
    w = mx.random.normal(lin.weight.shape, key=mx.random.key(9)) * 0.05
    lin.weight = w.astype(mx.bfloat16)
    if bias:
        lin.bias = mx.random.normal(lin.bias.shape).astype(mx.bfloat16)
    q = lin.to_quantized(group_size=gs, bits=4, mode=mode)
    mx.eval(q.parameters())
    return q


@needs_nax
def test_fused_gate_up_activation_routing(installed, epilogue_calls, monkeypatch):
    from omlx.patches.glm_moe_dsa.switch_layers import SwiGLU

    proj = _dsa_gate_up()
    x, idx = _routed_rows(96, 8, 16, 128, mx.bfloat16, 0.5, x_scale=4.0)
    act = SwiGLU()
    got = patch_mod.fused_gate_up_activation(proj, x, idx, act)
    assert epilogue_calls == [True]
    x_gate, x_up = mx.split(proj(x, idx, sorted_indices=True), 2, axis=-1)
    _assert_bitwise(got, act(x_up, x_gate))

    # Not taken: unknown activation, too few rows per expert, per-expert
    # bias, the switch, or the reroute wrapper not installed.
    assert patch_mod.fused_gate_up_activation(proj, x, idx, nn.GELU()) is None
    few_x, few_idx = _routed_rows(6, 8, 16, 128, mx.bfloat16, 0.0)  # 48 rows / 16
    assert patch_mod.fused_gate_up_activation(proj, few_x, few_idx, act) is None
    biased = _dsa_gate_up(bias=True)
    assert patch_mod.fused_gate_up_activation(biased, x, idx, act) is None
    monkeypatch.setenv("OMLX_M5_GATHER_QMM_NAX", "0")
    assert patch_mod.fused_gate_up_activation(proj, x, idx, act) is None
    monkeypatch.delenv("OMLX_M5_GATHER_QMM_NAX")
    mx.gather_qmm = installed
    assert patch_mod.fused_gate_up_activation(proj, x, idx, act) is None
    assert epilogue_calls == [True]


# ---------------------------------------------------------------------------
# Row map: token rows read in place (sorted_gather_qmm_swiglu(row_map=...))
# ---------------------------------------------------------------------------


def _token_rows(T, K, dtype, seed=3):
    """Token rows whose scale spans 0.1x-16x (sigmoid tails, clip bounds)."""
    scale = mx.power(10.0, mx.linspace(-1.0, 1.2, T)).reshape(T, 1, 1)
    return (mx.random.normal((T, 1, K), key=mx.random.key(seed)) * scale).astype(dtype)


def _row_map(M, T):
    """A scrambled, repeating sorted row -> token row map touching every
    token row, the last one included."""
    return ((mx.arange(M, dtype=mx.uint32) * 7 + 3) % T).astype(mx.uint32)


@needs_nax
@pytest.mark.parametrize("case", _EPI_CASES, ids=_case_id)
def test_row_map_bit_identical_to_materialised_rows(case):
    """Rows read through the map == the copied rows, and == the unfused
    path (plain kernel + split + activation) on the copy."""
    mode, bits, gs, dtype, plan = case
    idx = mx.array(np.repeat(np.arange(len(_COUNTS)), _COUNTS).astype(np.uint32))
    M = int(idx.shape[0])
    T = M // 3 + 1
    for K in (256, 384):
        wq, scales, biases, _ = _quantized(len(_COUNTS), 256, K, mode, bits, gs, dtype)
        x_tok = _token_rows(T, K, dtype)
        row_map = _row_map(M, T)
        x = x_tok[row_map]
        kw = dict(group_size=gs, bits=bits, mode=mode, plan=plan)
        for limit in (None, 10.0):
            got = nax.sorted_gather_qmm_swiglu(
                x_tok, wq, scales, biases, idx, limit=limit, row_map=row_map, **kw
            )
            assert got is not None and got.shape == (M, 1, 128)
            ref = nax.sorted_gather_qmm_swiglu(
                x, wq, scales, biases, idx, limit=limit, **kw
            )
            _assert_bitwise(got, ref, f"K={K} limit={limit}")
            gate_up = nax.sorted_gather_qmm(x, wq, scales, biases, idx, **kw)
            x_gate, x_up = mx.split(gate_up, 2, axis=-1)
            _assert_bitwise(got, nax.reference_activation(x_up, x_gate, limit))


@needs_nax
@pytest.mark.parametrize("mode,gs,limit", [("affine", 64, 10.0), ("mxfp4", 32, None)])
@pytest.mark.parametrize("tokens,top_k,E", [(300, 8, 16), (1100, 10, 64)])
def test_row_map_of_routed_tokens(mode, gs, limit, tokens, top_k, E):
    """SwitchGLU routing through sort_routes, default plans."""
    K, n = 256, 64
    wq, scales, biases, _ = _quantized(E, 2 * n, K, mode, 4, gs, mx.bfloat16)
    scores = mx.random.uniform(shape=(1, tokens, E), key=mx.random.key(tokens))
    inds = mx.argpartition(-scores, kth=top_k - 1, axis=-1)[..., :top_k]
    h = (mx.random.normal((1, tokens, K), key=mx.random.key(1)) * 4.0).astype(
        mx.bfloat16
    )
    x_tok, row_map, idx, _ = sort_routes(mx.expand_dims(h, (-2, -3)), inds)
    kw = dict(group_size=gs, bits=4, mode=mode, limit=limit)
    got = nax.sorted_gather_qmm_swiglu(
        x_tok, wq, scales, biases, idx, row_map=row_map, **kw
    )
    ref = nax.sorted_gather_qmm_swiglu(x_tok[row_map], wq, scales, biases, idx, **kw)
    assert got is not None and ref is not None
    _assert_bitwise(got, ref)


def test_sort_routes_matches_mlx_lm_gather_sort():
    from mlx_lm.models.switch_layers import _gather_sort

    h = mx.random.normal((2, 50, 1, 1, 32)).astype(mx.bfloat16)
    inds = mx.argpartition(
        -mx.random.uniform(shape=(2, 50, 16)), kth=5, axis=-1
    )[..., :6]
    x_tok, row_map, idx, inv = sort_routes(h, inds)
    x_ref, idx_ref, inv_ref = _gather_sort(h, inds)
    assert x_tok.shape == (100, 1, 32) and row_map.dtype == mx.uint32
    _assert_bitwise(x_tok[row_map], x_ref)
    assert mx.array_equal(idx, idx_ref).item()
    assert mx.array_equal(inv, inv_ref).item()


def test_row_map_unsupported_calls_return_none(monkeypatch):
    E, K, n, T = 4, 128, 64, 40
    idx = mx.zeros((64,), dtype=mx.uint32)
    x_tok = mx.zeros((T, 1, K), dtype=mx.bfloat16)
    w = mx.zeros((E, 2 * n, K // 8), dtype=mx.uint32)
    s = mx.zeros((E, 2 * n, K // 64), dtype=mx.bfloat16)
    rmap = _row_map(64, T)

    def call(row_map, x=x_tok):
        return nax.sorted_gather_qmm_swiglu(
            x, w, s, s, idx, group_size=64, bits=4, verify=False, row_map=row_map
        )

    assert call(rmap.astype(mx.int32)) is None  # map must be uint32
    assert call(rmap[:-1]) is None  # one entry per sorted row
    assert call(rmap.reshape(8, 8)) is None
    assert call(rmap, x=x_tok.reshape(T, K)) is None  # token rows [T, 1, K]
    assert not nax.supports(x_tok, w, s, s, idx, 64, 4, "affine")  # no map
    assert nax.supports(x_tok, w, s, s, idx, 64, 4, "affine", rmap)
    monkeypatch.setenv("OMLX_M5_GATHER_QMM_NAX", "0")
    assert call(rmap) is None


def _dsa_problem(tokens=96, E=16, n=64, K=128, top_k=8):
    proj = _dsa_gate_up(E=E, n=n, K=K)
    scores = mx.random.uniform(shape=(1, tokens, E), key=mx.random.key(2))
    inds = mx.argpartition(-scores, kth=top_k - 1, axis=-1)[..., :top_k]
    h = (mx.random.normal((1, tokens, K), key=mx.random.key(3)) * 4.0).astype(
        mx.bfloat16
    )
    mx.eval(inds, h)
    return proj, h, inds


@needs_nax
def test_failed_row_map_self_test_falls_back(monkeypatch, installed, launches):
    from omlx.patches.glm_moe_dsa.switch_layers import SwiGLU

    proj, h, inds = _dsa_problem()
    x_tok, row_map, idx, _ = sort_routes(mx.expand_dims(h, (-2, -3)), inds)
    x = x_tok[row_map]
    ref = patch_mod.fused_gate_up_activation(proj, x, idx, SwiGLU())
    monkeypatch.setattr(nax, "_verified", {})
    monkeypatch.setattr(nax, "_self_test_act_map", lambda key: False)
    del launches[:]
    got = patch_mod.fused_gate_up_activation(
        proj, x, idx, SwiGLU(), token_rows=(x_tok, row_map)
    )
    # declined map -> the copy (earlier launches: the unmapped canaries,
    # re-run after the verdict cache reset)
    assert launches[-1] is False and True not in launches
    _assert_bitwise(got, ref)


@needs_nax
def test_fused_gate_up_activation_reads_token_rows(installed, launches, monkeypatch):
    from omlx.patches.glm_moe_dsa.switch_layers import SwiGLU

    proj, h, inds = _dsa_problem()
    x_tok, row_map, idx, _ = sort_routes(mx.expand_dims(h, (-2, -3)), inds)
    x = x_tok[row_map]
    act = SwiGLU()
    got = patch_mod.fused_gate_up_activation(
        proj, x, idx, act, token_rows=(x_tok, row_map)
    )
    ref = patch_mod.fused_gate_up_activation(proj, x, idx, act)
    assert launches == [True, False]
    _assert_bitwise(got, ref)
    x_gate, x_up = mx.split(proj(x, idx, sorted_indices=True), 2, axis=-1)
    _assert_bitwise(got, act(x_up, x_gate))
    # A declined map, or one the kernel does not take: the copy, same result.
    with monkeypatch.context() as m:
        _row_map_off(m)
        off = patch_mod.fused_gate_up_activation(
            proj, x, idx, act, token_rows=(x_tok, row_map)
        )
    bad = patch_mod.fused_gate_up_activation(
        proj, x, idx, act, token_rows=(x_tok, row_map.astype(mx.int32))
    )
    assert launches == [True, False, False, False]
    _assert_bitwise(off, ref)
    _assert_bitwise(bad, ref)


# ---------------------------------------------------------------------------
# Model forwards: epilogue and row map vs the unfused path
# ---------------------------------------------------------------------------

D, INTER, N_EXPERTS, TOP_K = 128, 64, 16, 8


class _Holder(nn.Module):
    def __init__(self, glu):
        super().__init__()
        self.layers = [glu]


def _holder_class(family: str):
    return type("Model", (_Holder,), {"__module__": f"mlx_lm.models.{family}"})


def _make_glu(module, quant, seed=0, activation=None):
    mx.random.seed(seed)
    kwargs = {} if activation is None else {"activation": activation}
    glu = module.SwitchGLU(D, INTER, N_EXPERTS, **kwargs)
    for name in ("gate_proj", "up_proj", "down_proj"):
        glu[name].weight = glu[name].weight.astype(mx.bfloat16)
    group_size, bits, mode = quant
    nn.quantize(glu, group_size=group_size, bits=bits, mode=mode)
    mx.eval(glu.parameters())
    return glu.eval()  # loaded models run in eval mode


def _fused_glu(module, quant, family, activation=None, seed=0):
    glu = _make_glu(module, quant, seed=seed, activation=activation)
    assert fusion.apply_switch_glu_gate_up_fusion(_holder_class(family)(glu)) == 1
    return glu


def _fused_twin(module, glu, quant, family, activation=None):
    twin = _make_glu(module, quant, seed=99, activation=activation)
    for name in ("gate_proj", "up_proj", "down_proj"):
        for field in ("weight", "scales", "biases"):
            if glu[name].get(field) is not None:
                setattr(twin[name], field, mx.array(glu[name][field]))
    mx.eval(twin.parameters())
    assert fusion.apply_switch_glu_gate_up_fusion(_holder_class(family)(twin)) == 1
    return twin


def _routes(tokens, seed=1):
    scores = mx.random.uniform(shape=(1, tokens, N_EXPERTS), key=mx.random.key(seed))
    return mx.argpartition(-scores, kth=TOP_K - 1, axis=-1)[..., :TOP_K]


def _inputs(tokens):
    x = (mx.random.normal((1, tokens, D), key=mx.random.key(tokens)) * 2.0).astype(
        mx.bfloat16
    )
    inds = _routes(tokens)
    w = mx.random.uniform(shape=inds.shape, key=mx.random.key(5))
    scores = (w / w.sum(axis=-1, keepdims=True)).astype(mx.float32)
    mx.eval(x, inds, scores)
    return x, inds, scores


def _check_glu(reference, fused, epilogue_calls, monkeypatch, tokens_list):
    for tokens in tokens_list:
        x, i, s = _inputs(tokens)
        for weighted in (False, True):
            del epilogue_calls[:]
            ref = reference(x, i, scores=s, weighted_sum=weighted)
            got = fused(x, i, scores=s, weighted_sum=weighted)
            mx.eval(ref, got)
            assert epilogue_calls == [True], (tokens, weighted)
            _assert_bitwise(got, ref, f"tokens={tokens} weighted={weighted}")
            # the same module with the epilogue declined
            with monkeypatch.context() as m:
                _epilogue_off(m)
                off = fused(x, i, scores=s, weighted_sum=weighted)
            _assert_bitwise(got, off)


def _gathers(out) -> int:
    f = io.StringIO()
    mx.export_to_dot(f, out)
    return f.getvalue().count('label ="Gather"')


def _check_row_map_on_off(run, launches, monkeypatch, n_launch=1):
    """``run()`` with the row map (default) and without: bitwise equal, one
    Gather fewer in the output graph, and only mapped epilogue launches."""
    del launches[:]
    on = run()
    gathers_on = _gathers(on)
    mx.eval(on)
    assert launches == [True] * n_launch
    with monkeypatch.context() as m:
        _row_map_off(m)
        off = run()
        gathers_off = _gathers(off)
        mx.eval(off)
    assert launches == [True] * n_launch + [False] * n_launch
    assert gathers_on == gathers_off - n_launch
    _assert_bitwise(on, off)


@needs_nax
@pytest.mark.parametrize("quant", [(32, 4, "mxfp4"), (64, 4, "affine")])
def test_glm_dsa_switch_glu_epilogue_bit_exact(
    installed, epilogue_calls, monkeypatch, quant
):
    """MiMo V2's experts (GLM DSA SwitchGLU): fused gate/up + epilogue vs
    the original separate gate and up projections."""
    from omlx.patches.glm_moe_dsa import switch_layers as dsa

    reference = _make_glu(dsa, quant)
    fused = _fused_twin(dsa, reference, quant, "mimo_v2")
    _check_glu(reference, fused, epilogue_calls, monkeypatch, (96, 300))


@needs_nax
@pytest.mark.parametrize("quant", [(64, 4, "affine"), (64, 8, "affine")])
def test_deepseek_v4_switch_glu_clamped_epilogue_bit_exact(
    installed, epilogue_calls, monkeypatch, quant
):
    """GLM-5.3's experts (DeepSeek V4 SwitchGLU, Glm5NextClampedSwiGLU)."""
    from omlx.patches.deepseek_v4 import switch_layers as v4

    act = _glm5_language().Glm5NextClampedSwiGLU(10.0)
    reference = _make_glu(v4, quant, activation=act)
    fused = _fused_twin(v4, reference, quant, "glm5_next", activation=act)
    _check_glu(reference, fused, epilogue_calls, monkeypatch, (160, 300))


@needs_nax
def test_decode_and_short_blocks_keep_the_unfused_path(installed, epilogue_calls):
    from omlx.patches.glm_moe_dsa import switch_layers as dsa

    quant = (32, 4, "mxfp4")
    reference = _make_glu(dsa, quant)
    fused = _fused_twin(dsa, reference, quant, "mimo_v2")
    for tokens in (1, 3, 7):  # unsorted, or < 4 rows per expert
        x = mx.random.normal((1, tokens, D)).astype(mx.bfloat16)
        i = _routes(tokens)
        _assert_bitwise(fused(x, i), reference(x, i))
    assert not any(epilogue_calls)
    # Training mode keeps the differentiable path.
    x = mx.random.normal((1, 96, D)).astype(mx.bfloat16)
    i = _routes(96)
    ref = reference(x, i)
    _assert_bitwise(fused.train()(x, i), ref)
    assert epilogue_calls == []
    _assert_bitwise(fused.eval()(x, i), ref)
    assert epilogue_calls == [True]


def _mimo_moe(T):
    from omlx.patches.mimo_v2 import apply_mimo_v2_patch

    apply_mimo_v2_patch()
    mimo = importlib.import_module("mlx_lm.models.mimo_v2")
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
            "n_routed_experts": N_EXPERTS,
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
    x = (mx.random.normal((1, T, D)) * 2.0).astype(mx.bfloat16)
    mx.eval(moe.parameters(), x)
    return moe.eval(), x


@needs_nax
@pytest.mark.parametrize("T", [96, 300])
def test_mimo_moe_block_epilogue_matches_unfused(installed, epilogue_calls, T):
    moe, x = _mimo_moe(T)
    ref = moe(x)
    mx.eval(ref)
    assert fusion.apply_switch_glu_gate_up_fusion(moe) == 1
    got = moe(x)
    mx.eval(got)
    assert epilogue_calls == [True]
    _assert_bitwise(got, ref)


@needs_nax
@pytest.mark.parametrize("quant", [(32, 4, "mxfp4"), (64, 4, "affine")])
def test_glm_dsa_switch_glu_row_map(installed, launches, monkeypatch, quant):
    """MiMo V2's experts (GLM DSA SwitchGLU), both inverse-order variants."""
    from omlx.patches.glm_moe_dsa import switch_layers as dsa

    glu = _fused_glu(dsa, quant, "mimo_v2")
    for tokens in (96, 300):
        x, inds, scores = _inputs(tokens)
        for inverse_scatter in (False, True):
            glu.inverse_scatter = inverse_scatter
            for weighted in (False, True):
                _check_row_map_on_off(
                    lambda: glu(x, inds, scores=scores, weighted_sum=weighted),
                    launches,
                    monkeypatch,
                )


@needs_nax
def test_deepseek_v4_switch_glu_row_map(installed, launches, monkeypatch):
    """GLM-5.3's experts (DeepSeek V4 SwitchGLU, clamped SwiGLU)."""
    from omlx.patches.deepseek_v4 import switch_layers as v4

    act = _glm5_language().Glm5NextClampedSwiGLU(10.0)
    glu = _fused_glu(v4, (64, 4, "affine"), "glm5_next", activation=act)
    for tokens in (160, 300):
        x, inds, scores = _inputs(tokens)
        for weighted in (False, True):
            _check_row_map_on_off(
                lambda: glu(x, inds, scores=scores, weighted_sum=weighted),
                launches,
                monkeypatch,
            )


# Qwen: mlx-vlm SwitchGLU regrouped by qwen35_moe_gate_up (Qwen3.8 runs the
# weighted-sum prefill path from 1024 tokens, the patched call below that).


def _qwen_glu(seed=7):
    from mlx_vlm.models.switch_layers import SwitchGLU as VLMSwitchGLU

    mx.random.seed(seed)
    glu = VLMSwitchGLU(D, INTER, N_EXPERTS)
    for name in ("gate_proj", "up_proj", "down_proj"):
        glu[name].weight = glu[name].weight.astype(mx.bfloat16)
    nn.quantize(glu, group_size=64, bits=4)
    mx.eval(glu.parameters())
    return glu.eval()


@pytest.fixture
def qwen_regroup(monkeypatch):
    """qwen35_moe_gate_up with its SwitchGLU call patch restored afterwards."""
    from mlx_lm.models.switch_layers import SwitchGLU
    from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward
    from mlx_vlm.models.switch_layers import SwitchGLU as VLMSwitchGLU

    import omlx.patches.qwen35_moe_gate_up as gate_up

    monkeypatch.delenv("OMLX_QWEN35_MOE_GATE_UP", raising=False)
    verifier = Qwen3_5BatchInvariantForward
    monkeypatch.setattr(verifier, "_switch_glu", verifier._switch_glu)
    saved = {cls: cls.__dict__.get("__call__") for cls in (SwitchGLU, VLMSwitchGLU)}
    flags = ("_omlx_gate_up_fused_call", "_omlx_gate_up_original_call")
    yield gate_up
    for cls, call in saved.items():
        original = cls.__dict__.get("_omlx_gate_up_original_call", call)
        cls.__call__ = original if original is not None else call
        for attr in flags:
            if attr in cls.__dict__:
                delattr(cls, attr)
    gate_up._CALL_PATCHED = False


def _regrouped(gate_up_mod, glu):
    class _FakeQwen4Model:
        pass

    _FakeQwen4Model.__module__ = "mlx_vlm.models.qwen4_exp.qwen4_exp"
    model = _FakeQwen4Model()
    model.named_modules = lambda: [("blocks.0", glu)]
    assert gate_up_mod.apply_qwen35_moe_gate_up_fusion(model) == 1
    return glu


def _qwen_weighted_sum_kernel():
    from omlx.custom_kernels.qwen35_prefill import fast

    if not fast.has_symbol("qwen35_moe_weighted_sum"):
        pytest.skip("qwen35_moe_weighted_sum native kernel unavailable")
    return fast.qwen35_moe_weighted_sum


@needs_nax
def test_qwen_weighted_sum_prefill_epilogue_bit_exact(
    installed, epilogue_calls, qwen_regroup
):
    import omlx.patches.qwen35_moe_weighted_sum as ws

    kernel = _qwen_weighted_sum_kernel()
    reference = _qwen_glu()
    fused = _regrouped(qwen_regroup, _qwen_glu())
    for tokens in (96, 300):
        x, inds, scores = _inputs(tokens)
        del epilogue_calls[:]
        ref = ws._native_switch_weighted_sum(reference, x, inds, scores, kernel)
        got = ws._native_switch_weighted_sum(fused, x, inds, scores, kernel)
        mx.eval(ref, got)
        assert epilogue_calls == [True]
        _assert_bitwise(got, ref, f"tokens={tokens}")


@needs_nax
def test_qwen_regrouped_call_epilogue_bit_exact(
    installed, epilogue_calls, qwen_regroup
):
    reference = _qwen_glu()
    fused = _regrouped(qwen_regroup, _qwen_glu())
    for tokens in (96, 300):
        x, inds, _ = _inputs(tokens)
        del epilogue_calls[:]
        ref = reference(x, inds)
        got = fused(x, inds)
        mx.eval(ref, got)
        assert epilogue_calls == [True]
        _assert_bitwise(got, ref, f"tokens={tokens}")


@needs_nax
def test_qwen_weighted_sum_prefill_row_map(
    installed, launches, monkeypatch, qwen_regroup
):
    import omlx.patches.qwen35_moe_weighted_sum as ws

    kernel = _qwen_weighted_sum_kernel()
    glu = _regrouped(qwen_regroup, _qwen_glu())
    for tokens in (96, 300):
        x, inds, scores = _inputs(tokens)
        _check_row_map_on_off(
            lambda: ws._native_switch_weighted_sum(glu, x, inds, scores, kernel),
            launches,
            monkeypatch,
        )


@needs_nax
def test_qwen_regrouped_call_row_map(installed, launches, monkeypatch, qwen_regroup):
    """Qwen's sorted SwitchGLU call below the weighted-sum threshold."""
    glu = _regrouped(qwen_regroup, _qwen_glu())
    for tokens in (96, 300):
        x, inds, _ = _inputs(tokens)
        _check_row_map_on_off(lambda: glu(x, inds), launches, monkeypatch)
