"""oQh quantization of one tensor against its input Gram.

The weight is rotated into the oQh basis, rounded to uniform levels with one
scale per row, its codes re-picked by the oQe Hessian refinement, and the row
scale refit by least squares under the same Hessian. quantize_affine keeps
group scales and biases instead: the oQe quantizer runs on the rotated weight,
weighted by the rotated Gram's diagonal, before the same refinement.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from .. import oq
from . import format as fmt

# Row-scale candidates around the RMS, as multiples of the Gaussian-optimal step.
_RTN_FACTORS = np.linspace(0.6, 1.4, 41)
_REFIT_ROW_ELEMS = 1 << 25
_GROUP = 64

_LEVEL_STEPS: dict[int, float] = {}


def rotate(x: mx.array, signs: mx.array, block: int) -> mx.array:
    """Signed block Hadamard of the last axis in float32, for offline use."""
    shape = x.shape
    y = x.astype(mx.float32) * signs.astype(mx.float32)
    y = y.reshape(*shape[:-1], shape[-1] // block, block)
    return mx.hadamard_transform(y).reshape(shape)


def rotate_gram(gram: mx.array, signs: mx.array, block: int) -> mx.array:
    """The Gram of rotated inputs: R G R^T for R = H_b diag(signs)."""
    s = signs.astype(mx.float32)
    g = gram.astype(mx.float32) * s[:, None] * s[None, :]
    g = rotate(
        mx.swapaxes(rotate(g, mx.ones_like(s), block), -1, -2), mx.ones_like(s), block
    )
    return (g + mx.swapaxes(g, -1, -2)) * 0.5


def level_step(bits: int) -> float:
    """MSE-optimal step of 2**bits symmetric uniform levels for unit-variance values."""
    step = _LEVEL_STEPS.get(bits)
    if step is None:
        x = np.random.default_rng(0).standard_normal(1_000_000)
        base = np.arange(1 << bits) - ((1 << bits) - 1) / 2
        best = (math.inf, 1.0)
        for d in np.linspace(0.05, 1.5, 1451):
            u = base * d
            edges = (u[1:] + u[:-1]) / 2
            err = float(np.mean((x - u[np.searchsorted(edges, x)]) ** 2))
            if err < best[0]:
                best = (err, float(d))
        step = _LEVEL_STEPS[bits] = best[1]
    return step


def _bf16(x: mx.array) -> mx.array:
    return x.astype(mx.bfloat16).astype(mx.float32)


def _round_rows(wr: mx.array, bits: int):
    """Nearest-level codes and the bf16 row scale with the lowest squared error."""
    qmax = (1 << bits) - 1
    center = qmax / 2.0
    rms = mx.sqrt(mx.mean(wr * wr, axis=-1))
    step = level_step(bits)
    best_err = best_s = best_q = None
    for f in _RTN_FACTORS:
        s = _bf16(rms * float(f * step))
        safe = mx.where(s > 0, s, mx.ones_like(s))
        q = mx.clip(mx.round(wr / safe[..., None] + center), 0, qmax)
        err = mx.sum((s[..., None] * (q - center) - wr) ** 2, axis=-1)
        if best_err is None:
            best_err, best_s, best_q = err, s, q
        else:
            take = err < best_err
            best_err = mx.where(take, err, best_err)
            best_s = mx.where(take, s, best_s)
            best_q = mx.where(take[..., None], q, best_q)
        mx.eval(best_err, best_s, best_q)
    return best_q, best_s


def _refit_scales(
    wr: mx.array, codes: mx.array, scale: mx.array, hr: mx.array, bits: int
):
    """argmin_s (W - s C) H (W - s C)^T per row, rounded to bf16."""
    center = ((1 << bits) - 1) / 2.0
    n_exp, n_rows, n_in = wr.shape
    rows = max(1, _REFIT_ROW_ELEMS // (n_exp * n_in))
    out = []
    for r0 in range(0, n_rows, rows):
        c = codes[:, r0 : r0 + rows] - center
        hc = c @ hr
        num = mx.sum(hc * wr[:, r0 : r0 + rows], axis=-1)
        den = mx.sum(hc * c, axis=-1)
        s = scale[:, r0 : r0 + rows]
        fit = mx.where(den > 0, num / mx.where(den > 0, den, mx.ones_like(den)), s)
        out.append(_bf16(mx.where(fit > 0, fit, s)))
        mx.eval(out[-1])
    return mx.concatenate(out, axis=1)


def _objective(wr, codes, scale, hr, bits) -> float:
    center = ((1 << bits) - 1) / 2.0
    err = scale[..., None] * (codes - center) - wr
    return float(mx.sum((err @ hr) * err).item())


def quantize(weight: mx.array, gram: mx.array, bits: int, signs: mx.array, block: int):
    """oQh codes and row scales of one (rows, in) or (experts, rows, in) weight.

    gram: the matching _shrunk_gram output (1 or experts, in, in) in the plain basis.
    Returns (packed uint32, rscale bfloat16, objective before, objective after).
    """
    if bits not in fmt.SUPPORTED_BITS:
        raise ValueError(f"oQh bits must be one of {fmt.SUPPORTED_BITS}, got {bits}")
    lead = weight.shape[:-1]
    n_in = int(weight.shape[-1])
    w3 = weight.reshape(-1, *weight.shape[-2:]) if weight.ndim == 3 else weight[None]
    n_exp, n_rows = int(w3.shape[0]), int(w3.shape[1])
    wr = rotate(w3, signs, block)
    hr = rotate_gram(gram, signs, block)
    mx.eval(wr, hr)
    codes, scale = _round_rows(wr, bits)
    before = _objective(wr, codes, scale, hr, bits)

    center = ((1 << bits) - 1) / 2.0
    gs = _GROUP if n_in % _GROUP == 0 else 32
    group_scale = mx.repeat(scale[..., None], n_in // gs, axis=-1)
    one = mx.ones((1, 1, 1), mx.float32)
    packed = oq._pack_affine_codes(
        codes.reshape(-1, n_in), one, mx.zeros_like(one), gs, bits
    )
    refined = oq._refine_affine_codes(
        wr, packed, group_scale, -center * group_scale, hr, bits, gs
    )
    if refined is not None:
        packed = refined[0].reshape(-1, packed.shape[-1])
        codes = oq._unpack_affine_codes(packed, bits, gs, n_in).reshape(
            n_exp, n_rows, n_in
        )
    scale = _refit_scales(wr, codes, scale, hr, bits)
    after = _objective(wr, codes, scale, hr, bits)
    packed = packed.reshape(*lead, -1)
    rscale = scale.reshape(*lead).astype(mx.bfloat16)
    mx.eval(packed, rscale)
    return packed, rscale, before, after


def quantize_affine(
    weight: mx.array,
    gram: mx.array,
    bits: int,
    signs: mx.array,
    block: int,
    group_size: int,
):
    """Rotated codes with group scales and biases of one weight, as quantize().

    Returns (packed uint32, scales, biases, objective before, objective after).
    """
    if bits not in fmt.SUPPORTED_BITS:
        raise ValueError(f"oQh bits must be one of {fmt.SUPPORTED_BITS}, got {bits}")
    lead = weight.shape[:-1]
    w3 = weight.reshape(-1, *weight.shape[-2:]) if weight.ndim == 3 else weight[None]
    wr = rotate(w3, signs, block)
    hr = rotate_gram(gram, signs, block)
    mx.eval(wr, hr)
    importance = mx.diagonal(hr, axis1=-2, axis2=-1)
    if w3.shape[0] == 1:
        wr, importance = wr[0], importance[0]
    packed, scales, biases = oq._quantize_chunked(
        wr, group_size, bits, "affine", importance=importance
    )
    refined = oq._refine_affine_codes(wr, packed, scales, biases, hr, bits, group_size)
    if refined is None:
        deq = mx.dequantize(packed, scales, biases, group_size=group_size, bits=bits)
        err = deq - wr
        err = err.astype(mx.float32).reshape(-1, *w3.shape[-2:])
        before = after = mx.sum((err @ hr) * err).item()
    else:
        packed, before, after = refined
    mx.eval(packed)
    return (
        packed.reshape(*lead, -1),
        scales.reshape(*lead, -1),
        biases.reshape(*lead, -1),
        before,
        after,
    )


def dequantize_affine(
    packed: mx.array,
    scales: mx.array,
    biases: mx.array,
    bits: int,
    signs: mx.array,
    block: int,
):
    """The float32 weight a group-scale oQh tensor stands for, in the plain basis."""
    n_in = int(packed.shape[-1]) * 32 // bits
    group_size = n_in // int(scales.shape[-1])
    wr = mx.dequantize(packed, scales, biases, group_size=group_size, bits=bits)
    wr = wr.astype(mx.float32)
    shape = wr.shape
    w = mx.hadamard_transform(wr.reshape(*shape[:-1], n_in // block, block))
    return w.reshape(shape) * signs.astype(mx.float32)


def dequantize(
    packed: mx.array, rscale: mx.array, bits: int, signs: mx.array, block: int
):
    """The float32 weight an oQh tensor stands for, in the plain basis."""
    n_in = int(packed.shape[-1]) * 32 // bits
    gs = _GROUP if n_in % _GROUP == 0 else 32
    center = ((1 << bits) - 1) / 2.0
    codes = oq._unpack_affine_codes(packed, bits, gs, n_in)
    wr = rscale.astype(mx.float32)[..., None] * (codes - center)
    # The rotation is orthogonal and symmetric per block: R^T = diag(signs) H_b.
    shape = wr.shape
    w = mx.hadamard_transform(wr.reshape(*shape[:-1], n_in // block, block)).reshape(
        shape
    )
    return w * signs.astype(mx.float32)
