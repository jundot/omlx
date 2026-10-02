# SPDX-License-Identifier: Apache-2.0
"""Reroute sorted gather_qmm around defective M5 NAX kernels (issue #2267).

On M5-generation GPUs mlx dispatches ``sorted_indices=True`` quantized
gather matmuls (the MoE sorted-prefill path) to the NAX
``*_gather_qmm_rhs_nax`` kernels. Two independent defects corrupt their
output through mlx 0.32.2:

- K remainder: the ``align_K=false`` tail bounds the activation tile
  load by ``BK`` instead of the K remainder
  (``quantized_nax.h::affine_gather_qmm_rhs_nax``), so whenever
  ``K % 64 != 0`` the tail multiplies stale threadgroup weights with
  out-of-bounds activation reads. The result is deterministically wrong
  output plus occasional recycled-buffer garbage (~1e36). This is what
  issue #2267's bit-exactness test caught: with ``inter=32`` the test's
  ``down_proj`` runs at K=32. All dtypes are affected (bf16/fp16/fp32),
  and the mxfp4 variant carries the same tail bug.
- Row offsets: the int16 fix that landed in mlx 0.32.0 still misses this
  kernel, so sorted row counts above 32768 overflow the row offset
  (ml-explore/mlx#3856, fixed upstream by ml-explore/mlx#3922 after the
  0.32.2 release). Reachable in production: a 4097+ token prefill chunk
  of a top-8 MoE crosses the boundary, and a top-10 MoE (Qwen4-Exp)
  crosses it at 3277 tokens.

``sorted_indices`` is a pure performance hint, so the wrapper can always
fall back to dropping it. The two defects get different treatment:

- ``K % 64 != 0`` drops the flag: the unsorted gather path guards
  ``K % 64 == 0`` before entering NAX and falls back to the verified
  steel kernels, at some prefill-throughput cost for the affected
  shapes only.
- Row overflow first goes to oMLX's native NAX gather kernel
  (``custom_kernels/qwen35_prefill``), a copy of the mlx rhs kernel with
  the row bound clamped before it narrows to int16. One dispatch covers
  every row, and each output row is bit-identical to the sliced result.
  Without the native extension, or for layouts it does not cover, the
  call keeps the flag and is split instead: balanced ``<= 32768``-row
  slices of the (already sorted) activations and indices whose partial
  outputs are concatenated. Each slice stays on the NAX rhs kernel, which
  keeps wide prefill chunks (4096+ tokens on a top-8/top-10 MoE) on the
  tensor units instead of the per-row gather kernel. Slices are balanced
  rather than cut at the cap so every slice still clears mlx's
  ``rows / experts >= 4`` gate for the batched rhs path.

The wrapper self-arms: the first matching call runs a tiny canary
against an fp32 dequantized reference and only intervenes when the
corruption is actually present on this machine/mlx build. Healthy
setups keep the fast path untouched, and the patch retires itself once
mlx ships a kernel fix. Kill switch: ``OMLX_M5_GATHER_QMM_FIX=0``; the
native kernel alone can be disabled with ``OMLX_M5_GATHER_QMM_NATIVE=0``.

On NAX hosts every supported sorted call (``transpose=True`` rhs gather
of ``[M, 1, K]`` rows, bf16/fp16 activations, affine 4/8-bit or MXFP4
weights, at least 4 rows per expert) goes to the runtime-compiled kernel
in ``m5_gather_qmm_nax`` before any of the above: segmented tile
scheduling (one expert per 64- to 128-row tile) on the tensor units,
correct for any K and row count in one dispatch, bit-identical to mlx's
sorted kernel wherever that kernel is correct. Each kernel instantiation self-tests
once; anything unsupported or failing keeps the stock handling above.
``OMLX_M5_GATHER_QMM_NAX=0`` disables only this route.

``fused_gate_up_activation`` lets a SwitchGLU forward whose fused ``[gate;
up]`` sorted projection would take that route run its SwiGLU in the NAX
kernel's epilogue instead of a separate elementwise pass (bit-identical).
Given the token rows and the sorted row map (``moe_routes.sort_routes``),
that kernel reads each routed token's row in place instead of from the
``[T * k, 1, K]`` copy that the sort would otherwise gather (bit-identical).
"""

from __future__ import annotations

import logging
import os

import mlx.core as mx

from omlx.custom_kernels.nax import is_nax_available

from . import m5_gather_qmm_nax as _nax

logger = logging.getLogger(__name__)

# ml-explore/mlx#3856: sorted row offsets overflow int16 past this count.
_MAX_SORTED_ROWS = 32768

_original_gather_qmm = None
_defective: bool | None = None
# Native NAX gather op, resolved on first oversized call (False: unavailable).
_native_gather = None
_nax_host: bool | None = None

# Positional parameters of mx.gather_qmm after (x, w).
_POSITIONAL = (
    "scales",
    "biases",
    "lhs_indices",
    "rhs_indices",
    "transpose",
    "group_size",
    "bits",
    "mode",
)
_DEFAULT_GROUP_SIZE = {"affine": 64, "mxfp4": 32}


def _sorted_gather_qmm_defective() -> bool:
    """Run the K=96 canary once; True when the NAX rhs kernel corrupts."""
    global _defective
    if _defective is not None:
        return _defective
    if not mx.metal.is_available():
        _defective = False
        return False
    n, e, out_dim, k = 64, 8, 64, 96
    keys = mx.random.split(mx.random.key(0x2267), 3)
    w = mx.random.normal((e, out_dim, k), key=keys[0]).astype(mx.bfloat16)
    wq, scales, biases = mx.quantize(w, group_size=32, bits=4)
    x = (mx.random.normal((n, 1, k), key=keys[1]) * 0.5).astype(mx.bfloat16)
    idx = mx.sort(mx.random.randint(0, e, (n,), key=keys[2]).astype(mx.uint32))
    wd = mx.dequantize(wq, scales, biases, group_size=32, bits=4)
    ref = x.astype(mx.float32) @ wd[idx].swapaxes(-1, -2).astype(mx.float32)
    out = _original_gather_qmm(
        x,
        wq,
        scales,
        biases,
        rhs_indices=idx,
        transpose=True,
        group_size=32,
        bits=4,
        sorted_indices=True,
    )
    err = mx.abs(out.astype(mx.float32) - ref).max().item()
    # Corruption sits at output magnitude (median row error ~7 at this
    # size); bf16 rounding stays below ~0.1. NaN also counts as corrupt.
    _defective = not (err < 1.0)
    if _defective:
        logger.warning(
            "sorted gather_qmm corrupts on this machine (canary max err "
            "%.3g); rerouting K %% 64 != 0 sorted calls to the unsorted "
            "path and segmenting >%d-row sorted calls (issue #2267)",
            err,
            _MAX_SORTED_ROWS,
        )
    return _defective


def _rhs_indices(args, kwargs):
    return args[3] if len(args) > 3 else kwargs.get("rhs_indices")


def _needs_reroute(x, args, kwargs) -> bool:
    """True when this call would select the defective NAX rhs kernel."""
    if not kwargs.get("sorted_indices"):
        return False
    # Positional layout after (x, w): scales, biases, lhs_indices,
    # rhs_indices, transpose, group_size, bits, mode. sorted_indices is
    # keyword-only.
    lhs = args[2] if len(args) > 2 else kwargs.get("lhs_indices")
    rhs = _rhs_indices(args, kwargs)
    transpose = args[4] if len(args) > 4 else kwargs.get("transpose", True)
    # The rhs kernel is only selected for the rhs-indices-only sorted
    # path with transposed weights (x @ w.T).
    if lhs is not None or rhs is None or not transpose:
        return False
    if x.shape[-1] % 64:
        return True
    return rhs.size * x.shape[-2] > _MAX_SORTED_ROWS


def _segment_bounds(rows: int) -> list[tuple[int, int]]:
    """Balanced ``[start, stop)`` slices with at most ``_MAX_SORTED_ROWS`` each."""
    count = -(-rows // _MAX_SORTED_ROWS)
    size = -(-rows // count)
    return [(start, min(start + size, rows)) for start in range(0, rows, size)]


def _segmented_sorted_gather_qmm(x, w, args, kwargs):
    """Issue an oversized sorted rhs call as ``<= 32768``-row slices.

    Only the layout the MoE sorted path produces is handled: a 3-D
    ``[rows, 1, K]`` activation with a flat sorted ``rhs_indices`` of the
    same row count. Anything else returns None and the caller falls back
    to dropping ``sorted_indices``.
    """
    rhs = _rhs_indices(args, kwargs)
    if x.ndim != 3 or x.shape[1] != 1 or rhs.ndim != 1 or rhs.shape[0] != x.shape[0]:
        return None
    rows = int(x.shape[0])
    outputs = []
    for start, stop in _segment_bounds(rows):
        seg_args = list(args)
        seg_kwargs = kwargs
        if len(args) > 3:
            seg_args[3] = rhs[start:stop]
        else:
            seg_kwargs = dict(kwargs, rhs_indices=rhs[start:stop])
        outputs.append(_original_gather_qmm(x[start:stop], w, *seg_args, **seg_kwargs))
    return mx.concatenate(outputs, axis=0)


def _resolve_native_gather():
    """The native NAX gather op, or None when this build/machine lacks it."""
    global _native_gather
    if _native_gather is None:
        _native_gather = False
        if os.environ.get("OMLX_M5_GATHER_QMM_NATIVE", "1") != "0":
            try:
                from ..custom_kernels.qwen35_prefill import fast

                if fast.gather_qmm_rhs_available():
                    _native_gather = fast.qwen35_gather_qmm_rhs_t
            except Exception:
                logger.debug("native NAX gather_qmm unavailable", exc_info=True)
    return _native_gather or None


def _native_sorted_gather_qmm(x, w, args, kwargs):
    """Issue an oversized sorted rhs call as one native NAX dispatch.

    Covers the MoE sorted-prefill layout with affine weights; returns None
    for anything else so the caller can slice instead.
    """
    native = _resolve_native_gather()
    if native is None:
        return None

    def arg(position, name, default=None):
        return args[position] if len(args) > position else kwargs.get(name, default)

    scales = arg(0, "scales")
    biases = arg(1, "biases")
    rhs = _rhs_indices(args, kwargs)
    group_size = arg(5, "group_size")
    bits = arg(6, "bits")
    if (
        arg(7, "mode", "affine") != "affine"
        or scales is None
        or biases is None
        or group_size is None
        or bits is None
        or kwargs.get("stream") is not None
        or x.ndim != 3
        or x.shape[1] != 1
        or rhs.ndim != 1
        or rhs.shape[0] != x.shape[0]
    ):
        return None
    try:
        return native(
            mx.contiguous(x),
            w,
            scales,
            biases,
            mx.contiguous(rhs.astype(mx.uint32)),
            int(bits),
            int(group_size),
        )
    except ValueError:
        # Layout outside the native kernel's validated envelope.
        return None


def _on_nax_host() -> bool:
    global _nax_host
    if _nax_host is None:
        try:
            _nax_host = bool(is_nax_available())
        except Exception:  # noqa: BLE001
            _nax_host = False
    return _nax_host


def _nax_rows_ok(x, w) -> bool:
    """Same gate as mlx's own choice of the sorted rhs kernel (GatherQMM:
    B >= 16 rows and B / E >= 4); fewer rows per expert run the per-row
    qmv kernel, which is correct and cheaper there."""
    rows = x.shape[0] if x.ndim == 3 else 0
    return rows >= 16 and w.ndim == 3 and rows // max(int(w.shape[0]), 1) >= 4


def _nax_sorted_gather_qmm(x, w, args, kwargs):
    """Route a sorted rhs gather to the NAX kernel; None keeps mlx's path."""
    params = dict(zip(_POSITIONAL, args))
    for name in _POSITIONAL:
        if name in kwargs:
            params[name] = kwargs[name]
    rhs = params.get("rhs_indices")
    if (
        not isinstance(x, mx.array)
        or not isinstance(w, mx.array)
        or not isinstance(rhs, mx.array)
        or params.get("lhs_indices") is not None
        or not params.get("transpose", True)
    ):
        return None
    if not _nax_rows_ok(x, w):
        return None
    mode = params.get("mode") or "affine"
    group_size = params.get("group_size")
    if group_size is None:
        group_size = _DEFAULT_GROUP_SIZE.get(mode)
    bits = params.get("bits")
    if bits is None:
        bits = 4
    if group_size is None or "scales" not in params:
        return None
    return _nax.sorted_gather_qmm(
        x,
        w,
        params["scales"],
        params.get("biases"),
        rhs,
        group_size=int(group_size),
        bits=int(bits),
        mode=mode,
        stream=kwargs.get("stream"),
    )


def _gather_qmm_rerouted(x, w, *args, **kwargs):
    if kwargs.get("sorted_indices") and _nax.enabled() and _on_nax_host():
        out = _nax_sorted_gather_qmm(x, w, args, kwargs)
        if out is not None:
            return out
    if _needs_reroute(x, args, kwargs) and _sorted_gather_qmm_defective():
        if x.shape[-1] % 64 == 0:
            out = _native_sorted_gather_qmm(x, w, args, kwargs)
            if out is None:
                out = _segmented_sorted_gather_qmm(x, w, args, kwargs)
            if out is not None:
                return out
        kwargs = dict(kwargs, sorted_indices=False)
    return _original_gather_qmm(x, w, *args, **kwargs)


_gather_qmm_rerouted._omlx_m5_reroute = True


# SwitchGLU activations the gate/up epilogue reproduces, by exact class:
# ``__call__(x_up, x_gate)`` is mlx-lm / mlx-vlm's compiled
# ``swiglu(x_gate, x_up) = nn.silu(x_gate) * x_up`` ...
_SWIGLU_ACTIVATIONS = frozenset(
    {
        ("mlx_lm.models.switch_layers", "SwiGLU"),
        ("mlx_vlm.models.switch_layers", "SwiGLU"),
        ("omlx.patches.glm_moe_dsa.switch_layers", "SwiGLU"),
        ("omlx.patches.deepseek_v4.switch_layers", "SwiGLU"),
    }
)
# ... or GLM-5.3's clamped SwiGLU (``limit`` attribute; None: plain).
_CLAMPED_SWIGLU_ACTIVATIONS = frozenset(
    {("mlx_vlm.models.glm5_next.language", "Glm5NextClampedSwiGLU")}
)
_UNSUPPORTED = object()


def _swiglu_limit(activation):
    """None (plain SwiGLU), the clip limit, or _UNSUPPORTED."""
    cls = type(activation)
    key = (cls.__module__, cls.__qualname__)
    if key in _SWIGLU_ACTIVATIONS:
        return None
    if key in _CLAMPED_SWIGLU_ACTIVATIONS:
        limit = getattr(activation, "limit", None)
        return None if limit is None else float(limit)
    return _UNSUPPORTED


def fused_gate_up_activation(proj, x, indices, activation, token_rows=None):
    """``activation(x_up, x_gate)`` of a fused ``[gate; up]`` projection in
    one kernel, or None.

    ``proj`` is a quantized switch linear whose expert rows are the gate
    rows followed by the up rows, ``x`` / ``indices`` the sorted routed rows
    (``[M, 1, K]`` / ``[M]``) of what would be a ``sorted_indices=True``
    call. Where that call would run on the NAX sorted gather kernel (this
    wrapper installed on an M5 host, >= 16 rows and >= 4 rows per expert, a
    layout ``m5_gather_qmm_nax`` supports, no per-expert bias) and the
    activation is a SwiGLU this module knows, the activation runs in the
    kernel's epilogue (``m5_gather_qmm_nax.sorted_gather_qmm_swiglu``,
    self-tested bit-identical to the unfused path) and only the ``[M, 1,
    N / 2]`` result is written. None otherwise: the caller runs the
    projection, the split and the activation itself.

    ``token_rows`` = ``(x_tok, row_map)`` with ``x == x_tok[row_map]``
    (``moe_routes.sort_routes``) lets the kernel read the sorted rows from
    the token rows in place, so a lazy ``x`` is never materialised
    (bit-identical). If the row-mapped kernel declines, ``x`` is used as
    before.
    """
    limit = _swiglu_limit(activation)
    if limit is _UNSUPPORTED:
        return None
    if not getattr(mx.gather_qmm, "_omlx_m5_reroute", False):
        return None
    if not (_nax.enabled() and _on_nax_host()):
        return None
    if "bias" in proj or not all(hasattr(proj, a) for a in ("group_size", "bits")):
        return None
    w = proj.get("weight")
    scales = proj.get("scales")
    if not isinstance(w, mx.array) or not isinstance(scales, mx.array):
        return None
    if not isinstance(x, mx.array) or not isinstance(indices, mx.array):
        return None
    if not _nax_rows_ok(x, w):
        return None
    kw = dict(
        group_size=int(proj.group_size),
        bits=int(proj.bits),
        mode=getattr(proj, "mode", None) or "affine",
        limit=limit,
    )
    if token_rows is not None:
        x_tok, row_map = token_rows
        if isinstance(x_tok, mx.array) and isinstance(row_map, mx.array):
            out = _nax.sorted_gather_qmm_swiglu(
                x_tok, w, scales, proj.get("biases"), indices, row_map=row_map, **kw
            )
            if out is not None:
                return out
    return _nax.sorted_gather_qmm_swiglu(
        x, w, scales, proj.get("biases"), indices, **kw
    )


def apply_m5_gather_qmm_workaround() -> bool:
    """Install the reroute wrapper on ``mx.gather_qmm``.

    Idempotent; returns True when the wrapper was installed by this
    call. Disabled entirely via ``OMLX_M5_GATHER_QMM_FIX=0``.
    """
    global _original_gather_qmm
    if os.environ.get("OMLX_M5_GATHER_QMM_FIX", "1") == "0":
        return False
    if getattr(mx.gather_qmm, "_omlx_m5_reroute", False):
        return False
    _original_gather_qmm = mx.gather_qmm
    mx.gather_qmm = _gather_qmm_rerouted
    logger.debug("m5 sorted gather_qmm reroute installed")
    return True
