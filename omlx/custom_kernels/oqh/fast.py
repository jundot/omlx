"""oQh kernels.

A weight row stores uniform codes in MLX affine packing and one scale,
``w = scale * (q - (2**bits - 1) / 2)``, in a basis rotated by a signed block
Walsh-Hadamard on the input axis. Producer ops (norms, activations, the gated
norm) write their output in that basis, so the matrix products need no extra
pass. Decode uses the native GEMVs; many rows go through the stock quantized
matmul with unit scales, so NAX is used where the hardware has it.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from .._loader import load_extension, native_stream_kwargs

_ext, _IMPORT_ERROR = load_extension(__package__, Path(__file__).parent)


def has_native() -> bool:
    return _ext is not None


def is_native_available() -> bool:
    return has_native()


def import_error() -> Exception | None:
    return _IMPORT_ERROR


def _require():
    if _ext is None:
        raise RuntimeError(f"oQh native extension unavailable: {_IMPORT_ERROR}")
    return _ext


def qmv(x: mx.array, w: mx.array, rscale: mx.array, bits: int, stream=None) -> mx.array:
    """One matrix-vector product per row of x. w: [N, K * bits / 32] uint32, rscale: [N]."""
    return _require().qmv(x, w, rscale, bits, **native_stream_kwargs(stream))


def gather_qmv(
    x: mx.array,
    w: mx.array,
    rscale: mx.array,
    indices: mx.array,
    bits: int,
    stream=None,
) -> mx.array:
    """Routed experts: x has 1 row (shared) or one row per index. Returns [P, N]."""
    return _require().gather_qmv(
        x, w, rscale, indices, bits, **native_stream_kwargs(stream)
    )


_UNIT_VIEWS: dict = {}


def _unit_view(w: mx.array, bits: int, dtype) -> tuple[mx.array, mx.array, int]:
    """All-ones scales and -center biases, shared by every weight of one shape and bit width."""
    k = w.shape[-1] * 32 // bits
    gs = next(g for g in (128, 64, 32) if k % g == 0)
    key = (tuple(w.shape[:-1]), k, gs, bits, dtype)
    view = _UNIT_VIEWS.get(key)
    if view is None:
        center = float((1 << bits) - 1) / 2.0
        shape = (*w.shape[:-1], k // gs)
        view = (mx.ones(shape, dtype), mx.full(shape, -center, dtype))
        _UNIT_VIEWS[key] = view
    return view[0], view[1], gs


# Up to this width the row scale rides in the stock kernel's group scale and
# bias. The bias -center * rscale rounds to the activation dtype; above 5 bits
# that error grows past a few percent of the rounding step, so wider codes keep
# unit scales and multiply the output.
_FOLD_MAX_BITS = 5


def _row_view(w, rscale, bits: int, dtype):
    """Group scales and biases that carry each row's scale."""
    k = w.shape[-1] * 32 // bits
    gs = next(g for g in (128, 64, 32) if k % g == 0)
    center = float((1 << bits) - 1) / 2.0
    scales = mx.broadcast_to(
        mx.expand_dims(rscale.astype(dtype), -1), (*rscale.shape, k // gs)
    )
    return scales, scales * -center, gs


def qmm(x: mx.array, w: mx.array, rscale: mx.array, bits: int, stream=None) -> mx.array:
    """Many rows of x (prefill) through the stock quantized_matmul."""
    if bits <= _FOLD_MAX_BITS:
        s, b, gs = _row_view(w, rscale, bits, x.dtype)
        return mx.quantized_matmul(
            x, w, s, b, transpose=True, group_size=gs, bits=bits, stream=stream
        )
    s1, b1, gs = _unit_view(w, bits, x.dtype)
    y = mx.quantized_matmul(
        x, w, s1, b1, transpose=True, group_size=gs, bits=bits, stream=stream
    )
    return y * rscale


def gather_qmm(
    x: mx.array,
    w: mx.array,
    rscale: mx.array,
    indices: mx.array,
    bits: int,
    sorted_indices: bool = False,
    stream=None,
) -> mx.array:
    """Routed experts for many rows through the stock gather_qmm. w: [E, N, ...], rscale: [E, N]."""
    if bits <= _FOLD_MAX_BITS:
        s, b, gs = _row_view(w, rscale, bits, x.dtype)
        return mx.gather_qmm(
            x,
            w,
            s,
            b,
            rhs_indices=indices,
            transpose=True,
            group_size=gs,
            bits=bits,
            sorted_indices=sorted_indices,
            stream=stream,
        )
    s1, b1, gs = _unit_view(w, bits, x.dtype)
    y = mx.gather_qmm(
        x,
        w,
        s1,
        b1,
        rhs_indices=indices,
        transpose=True,
        group_size=gs,
        bits=bits,
        sorted_indices=sorted_indices,
        stream=stream,
    )
    return y * mx.expand_dims(rscale[indices], -2)


def rmsnorm_had(
    x, weight, signs, eps: float, block: int, dual: bool = False, stream=None
):
    """Rotated RMSNorm(x) * weight, or (rotated, plain) when dual."""
    out = _require().rmsnorm_had(
        x, weight, signs, eps, block, dual, **native_stream_kwargs(stream)
    )
    return (out[0], out[1]) if dual else out[0]


def add_rmsnorm_had(
    x, r, weight, signs, eps: float, block: int, dual: bool = False, stream=None
):
    """h = x + r, then the rotated norm of h. Returns (h, rotated) or (h, rotated, plain)."""
    out = _require().add_rmsnorm_had(
        x, r, weight, signs, eps, block, dual, **native_stream_kwargs(stream)
    )
    return tuple(out)


def act_had(a, b, signs, mode: int, block: int, stream=None) -> mx.array:
    """mode 0: silu(a) * b, 1: a * sigmoid(b), 3: a; then the rotation."""
    return _require().act_had(a, b, signs, mode, block, **native_stream_kwargs(stream))


def act_split_had(y, signs, block: int, limit=None, stream=None) -> mx.array:
    """y holds gate | up halves of a fused projection: rotated silu(gate) * up.

    With ``limit``, the clamped SwiGLU silu(min(gate, limit)) * clamp(up, -limit, limit).
    """
    return _require().act_split_had(
        y, signs, block, limit, **native_stream_kwargs(stream)
    )


def gnorm_had(
    x, z, weight, signs, eps: float, block: int, dual: bool = False, stream=None
):
    """Gated RMSNorm over 128-wide heads, then the rotation of each row.

    Returns the rotated output, or (rotated, plain) when dual.
    """
    out = _require().gnorm_had(
        x, z, weight, signs, eps, block, dual, **native_stream_kwargs(stream)
    )
    return (out[0], out[1]) if dual else out[0]


def rotate(x, signs, block: int, stream=None) -> mx.array:
    """The input rotation alone, for callers that pass an unrotated activation."""
    return _require().act_had(x, None, signs, 3, block, **native_stream_kwargs(stream))
