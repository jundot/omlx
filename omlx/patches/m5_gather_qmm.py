# SPDX-License-Identifier: Apache-2.0
"""Reroute sorted gather_qmm around a defective M5 NAX kernel (issue #2267).

On M5-generation GPUs mlx dispatches ``sorted_indices=True`` quantized
gather matmuls (the MoE sorted-prefill path) to the NAX
``*_gather_qmm_rhs_nax`` kernels. Whenever ``K % 64 != 0`` their tail reads
weights and scales past the expert's K extent
(``quantized_nax.h::affine_gather_qmm_rhs_nax``; the mxfp4 variant too).
Through mlx 0.32.2 the activation tile load was unbounded as well, so the
output was deterministically wrong. mlx 0.32.3 bounds the activations
(ml-explore/mlx#4009), but the over-read stays: the output turns NaN when
the memory past an expert holds a NaN or Inf. The int16 row-offset overflow
past 32768 sorted rows (ml-explore/mlx#3856) is fixed in mlx 0.32.3
(ml-explore/mlx#3922).

Dropping ``sorted_indices`` is always safe (the unsorted path computes the
same product), so affected calls go to the unsorted path, which guards
``K % 64 == 0`` before entering NAX and falls back to the verified steel
kernels, at some prefill-throughput cost for those shapes only.

The wrapper self-arms: the first matching call runs a tiny canary whose last
expert is followed by NaN in the same buffer, against an fp32 dequantized
reference, and only intervenes when the defect is present on this
machine/mlx build. Healthy setups keep the fast path untouched, and the patch
retires itself once mlx ships a kernel fix.

On NAX hosts every supported sorted call (``transpose=True`` rhs gather
of ``[M, 1, K]`` rows, bf16/fp16 activations, affine 4/8-bit or MXFP4
weights, at least 4 rows per expert) goes to the runtime-compiled kernel
in ``m5_gather_qmm_nax`` before any of the above: segmented tile
scheduling (one expert per 64- to 128-row tile) on the tensor units,
correct for any K and row count in one dispatch, bit-identical to mlx's
sorted kernel wherever that kernel is correct. Each kernel instantiation self-tests
once; anything unsupported or failing keeps the stock handling above.

``fused_gate_up_activation`` lets a SwitchGLU forward whose fused ``[gate;
up]`` sorted projection would take that route run its SwiGLU in the NAX
kernel's epilogue instead of a separate elementwise pass (bit-identical).
Given the token rows and the sorted row map (``moe_routes.sort_routes``),
that kernel reads each routed token's row in place instead of from the
``[T * k, 1, K]`` copy that the sort would otherwise gather (bit-identical).
"""

from __future__ import annotations

import logging

import mlx.core as mx

from omlx.custom_kernels.nax import is_nax_available

from . import m5_gather_qmm_nax as _nax

logger = logging.getLogger(__name__)

_original_gather_qmm = None
_defective: bool | None = None
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


def _poisoned_tail(a: mx.array, value) -> mx.array:
    """``a`` followed in the same buffer by one more expert filled with ``value``."""
    pad = mx.full((1,) + a.shape[1:], value, dtype=a.dtype)
    return mx.concatenate([a, pad])[: a.shape[0]]


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
        _poisoned_tail(wq, 0xFFFFFFFF),
        _poisoned_tail(scales, float("nan")),
        _poisoned_tail(biases, float("nan")),
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
            "path (issue #2267)",
            err,
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
    return x.shape[-1] % 64 != 0


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
    if kwargs.get("sorted_indices") and _on_nax_host():
        out = _nax_sorted_gather_qmm(x, w, args, kwargs)
        if out is not None:
            return out
    if _needs_reroute(x, args, kwargs) and _sorted_gather_qmm_defective():
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
    if not _on_nax_host():
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
    call.
    """
    global _original_gather_qmm
    if getattr(mx.gather_qmm, "_omlx_m5_reroute", False):
        return False
    _original_gather_qmm = mx.gather_qmm
    mx.gather_qmm = _gather_qmm_rerouted
    logger.debug("m5 sorted gather_qmm reroute installed")
    return True
