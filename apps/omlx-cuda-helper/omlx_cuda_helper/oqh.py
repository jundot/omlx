# SPDX-License-Identifier: Apache-2.0
"""oQh linears: rotated codes with a row scale or with affine groups.

An oQh linear ``<p>`` stores ``<p>.weight`` (codes in MLX affine packing),
``<p>.oqh_signs`` [K / b, b] and either ``<p>.rscale`` [..., N], for
w = rscale * (q - (2^bits - 1) / 2), or ``<p>.oqh_scales`` / ``<p>.oqh_biases``
(affine groups). The weights live in the basis of a signed block
Walsh-Hadamard transform of the input axis, so each input is rotated the same
way: every block of b values times its signs, transformed, times 1 / sqrt(b).
oMLX writes the format and records its version in ``config["omlx_oqh"]``.

The rotation kernels follow the host's butterfly order and round once, so a
rotated bf16 input has the bits of the host's.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx
import numpy as np

from .checkpoint import OQH_SIGNS as SIGNS

logger = logging.getLogger(__name__)

# Widest code the host folds the row scale into the affine group scale; wider
# codes keep unit scales and multiply the output by the row scale.
FOLD_MAX_BITS = 5


def is_oqh(weights: dict[str, Any], prefix: str) -> bool:
    return f"{prefix}.{SIGNS}" in weights


def bits_of(weights: dict[str, Any], prefix: str) -> int:
    signs = weights[f"{prefix}.{SIGNS}"]
    return int(weights[f"{prefix}.weight"].shape[-1]) * 32 // int(np.prod(signs.shape))


def affine_view(
    weights: dict[str, mx.array], prefix: str
) -> tuple[mx.array, mx.array, dict[str, Any], mx.array | None]:
    """(scales, biases, quantization, output scale) of an oQh tensor as MLX affine.

    The row scale rides in group scales and biases as the host's prefill
    matmul builds them: the bias -center * scale rounds to the scale dtype.
    Above FOLD_MAX_BITS the groups hold unit scales and the output scale is
    the row scale.
    """
    weight = weights[f"{prefix}.weight"]
    bits = bits_of(weights, prefix)
    k = int(weight.shape[-1]) * 32 // bits
    scales = weights.get(f"{prefix}.oqh_scales")
    if scales is not None:
        biases = weights[f"{prefix}.oqh_biases"]
        quant = {"group_size": k // int(scales.shape[-1]), "bits": bits}
        return scales, biases, quant, None
    rscale = weights[f"{prefix}.rscale"]
    gs = next(g for g in (128, 64, 32) if k % g == 0)
    shape = (*rscale.shape, k // gs)
    center = float((1 << bits) - 1) / 2.0
    quant = {"group_size": gs, "bits": bits}
    if bits <= FOLD_MAX_BITS:
        scales = mx.broadcast_to(mx.expand_dims(rscale, -1), shape)
        return scales, scales * -center, quant, None
    ones = mx.ones(shape, rscale.dtype)
    return ones, mx.full(shape, -center, rscale.dtype), quant, rscale


def load_bank(weights: dict[str, mx.array], prefix: str) -> tuple | None:
    """(weight, scales, biases, quantization, rotation) of an oQh expert bank,
    None for another bank. A row-scale bank has biases None and group size 0."""
    signs = weights.get(f"{prefix}.{SIGNS}")
    if signs is None:
        return None
    weight = weights[f"{prefix}.weight"]
    bits = int(weight.shape[-1]) * 32 // signs.size
    scales = weights.get(f"{prefix}.oqh_scales")
    if scales is None:
        scales, biases, group = weights[f"{prefix}.rscale"], None, 0
    else:
        biases = weights[f"{prefix}.oqh_biases"]
        group = signs.size // int(scales.shape[-1])
    quant = {"bits": bits, "group_size": group, "mode": "affine", "oqh": True}
    return weight, scales, biases, quant, (signs, int(signs.shape[-1]))


def bank_dense(bank: tuple, experts: mx.array | None = None) -> mx.array:
    """The bf16 weights [E, N, K] of an oQh expert bank (weight, scales, biases,
    quantization), rebuilt as the expert kernel rebuilds them; ``experts``
    picks some of them.

    A row-scale bank (group size 0) keeps scales [E, N]; they ride in affine
    groups as ``affine_view`` builds them.
    """
    weight, scales, biases, quant = bank
    if experts is not None:
        weight, scales = weight[experts], scales[experts]
        biases = None if biases is None else biases[experts]
    bits, post = quant["bits"], None
    k = int(weight.shape[-1]) * 32 // bits
    gs = quant["group_size"] or next(g for g in (128, 64, 32) if k % g == 0)
    if not quant["group_size"]:
        shape = (*scales.shape, k // gs)
        center = float((1 << bits) - 1) / 2.0
        if bits <= FOLD_MAX_BITS:
            scales = mx.broadcast_to(mx.expand_dims(scales, -1), shape)
            biases = scales * -center
        else:
            post = scales
            scales = mx.ones(shape, post.dtype)
            biases = mx.full(shape, -center, post.dtype)
    dense = mx.dequantize(
        weight,
        scales.astype(mx.float32),
        biases.astype(mx.float32),
        group_size=gs,
        bits=bits,
    )
    if post is not None:
        # The host rounds the product before the row scale multiplies it;
        # this rebuild folds both, so it differs by one rounding there.
        dense = dense * mx.expand_dims(post.astype(mx.float32), -1)
    return dense.astype(mx.bfloat16)


def bank_gather(x: mx.array, bank: tuple, experts: mx.array) -> mx.array:
    """Rows [n, 1, K] through each row's expert of an oQh bank; ``experts`` is
    sorted. Plain ops for banks the expert kernel does not read: CUDA's
    gather_qmm has no 2-bit form, so the experts in use are rebuilt dense.
    """
    used, local = np.unique(np.asarray(experts), return_inverse=True)
    dense = bank_dense(bank, mx.array(used.astype(np.uint32)))
    return mx.gather_mm(
        x,
        dense.swapaxes(-1, -2),
        rhs_indices=mx.array(local.astype(np.uint32)),
        sorted_indices=True,
    )


_HEADER = """
__device__ inline float oqh_in(float x) { return x; }
__device__ inline float oqh_in(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ inline float oqh_in(__half x) { return __half2float(x); }
__device__ inline void oqh_out(float x, __nv_bfloat16* y) { *y = __float2bfloat16(x); }
__device__ inline void oqh_out(float x, __half* y) { *y = __float2half(x); }
__device__ inline void oqh_out(float x, float* y) { *y = x; }
"""

# One warp rotates one BLK-wide block; lane l holds values l * V .. l * V + V - 1.
# The butterflies run from bit 0 up, as the host kernels run them, so the
# float32 sums are the host's. MODE 0 rotates a, 1 rotates silu(a) * b,
# 2 the clamped SwiGLU silu(min(a, lim)) * clamp(b, -lim, lim), 3 the
# gated norm output a * silu(b) and 4 a times its row's b, rounded to TO first.
_ROTATE = """
  constexpr int V = BLK / 32;
  const int L = a_shape[1];
  const size_t w = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (w >= size_t(a_shape[0]) * (L / BLK)) return;
  const size_t row = w / (L / BLK);
  const int col = int(w % (L / BLK)) * BLK + lane * V;
  const size_t at = row * L + col;
  float v[V];
  #pragma unroll
  for (int i = 0; i < V; ++i) {
    const float av = oqh_in(a[at + i]);
    float f;
    if (MODE == 0) {
      f = av;
    } else if (MODE == 1) {
      f = (av / (1.0f + expf(-av))) * oqh_in(b[at + i]);
    } else if (MODE == 2) {
      const float lim = limit[0], g = fminf(av, lim);
      f = (g / (1.0f + expf(-g))) * fminf(fmaxf(oqh_in(b[at + i]), -lim), lim);
    } else if (MODE == 3) {
      const float z = oqh_in(b[at + i]);
      f = av * (z / (1.0f + expf(-z)));
    } else {
      TO r;
      oqh_out(av * oqh_in(b[row]), &r);
      f = oqh_in(r);
    }
    v[i] = f * oqh_in(signs[col + i]);
  }
  #pragma unroll
  for (int h = 1; h < V; h <<= 1) {
    #pragma unroll
    for (int i = 0; i < V; ++i) {
      if ((i & h) == 0) {
        const float p = v[i], q = v[i + h];
        v[i] = p + q;
        v[i + h] = p - q;
      }
    }
  }
  #pragma unroll
  for (int s = 1; s < 32; s <<= 1) {
    const bool up = (lane & s) != 0;
    #pragma unroll
    for (int i = 0; i < V; ++i) {
      const float p = __shfl_xor_sync(0xffffffffu, v[i], s);
      v[i] = up ? (p - v[i]) : (v[i] + p);
    }
  }
  #pragma unroll
  for (int i = 0; i < V; ++i) oqh_out(v[i] * SCALE, out + at + i);
"""

# One block a row of D = 8 x threads values: RMS norm as the host's oqh_norm
# rounds it, w * T(x * inv) in T, then the 256-wide rotation, one warp a block.
_NORM_ROTATE = """
  const int D = x_shape[1];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const size_t base = size_t(blockIdx.x) * D + tid * 8;
  const uint4 raw = *reinterpret_cast<const uint4*>(x + base);
  const T* xv = reinterpret_cast<const T*>(&raw);
  float v[8], ss = 0.0f;
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    v[i] = oqh_in(xv[i]);
    ss += v[i] * v[i];
  }
  __shared__ float red[32];
  #pragma unroll
  for (int s = 16; s > 0; s >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, s);
  if (lane == 0) red[warp] = ss;
  __syncthreads();
  float total = lane < int(blockDim.x >> 5) ? red[lane] : 0.0f;
  #pragma unroll
  for (int s = 16; s > 0; s >>= 1) total += __shfl_xor_sync(0xffffffffu, total, s);
  const float inv = 1.0f / sqrtf(total / float(D) + eps[0]);
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int e = tid * 8 + i;
    T n, p;
    oqh_out(v[i] * inv, &n);
    oqh_out(oqh_in(w[e]) * oqh_in(n), &p);
    v[i] = oqh_in(p) * oqh_in(signs[e]);
  }
  #pragma unroll
  for (int h = 1; h < 8; h <<= 1) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      if ((i & h) == 0) {
        const float a = v[i], b = v[i + h];
        v[i] = a + b;
        v[i + h] = a - b;
      }
    }
  }
  #pragma unroll
  for (int s = 1; s < 32; s <<= 1) {
    const bool up = (lane & s) != 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float q = __shfl_xor_sync(0xffffffffu, v[i], s);
      v[i] = up ? (q - v[i]) : (v[i] + q);
    }
  }
  uint4 packed;
  T* ov = reinterpret_cast<T*>(&packed);
  #pragma unroll
  for (int i = 0; i < 8; ++i) oqh_out(v[i] * 0.0625f, ov + i);
  *reinterpret_cast<uint4*>(out + base) = packed;
"""

_KERNELS: dict = {}


def _kernel(block: int):
    kernel = _KERNELS.get(block)
    if kernel is None:
        # The host normalizes by the float32 rsqrt of the block width.
        scale = repr(float(np.float32(1.0 / np.sqrt(block))))
        kernel = _KERNELS[block] = mx.fast.cuda_kernel(
            name=f"oqh_rotate_{block}",
            input_names=["a", "b", "signs", "limit"],
            output_names=["out"],
            source=_ROTATE.replace("SCALE", f"{scale}f"),
            header=_HEADER,
        )
    return kernel


def _rotate(a, b, signs, block, mode, limit, dtype):
    if block not in (128, 256) or a.shape[-1] % block:
        raise ValueError(
            f"oQh rotation block {block} does not tile width {a.shape[-1]}"
        )
    shape = a.shape
    a2 = a.reshape(-1, shape[-1])
    # MODE 4 reads one b a row.
    b2 = a2 if b is None else (b.reshape(-1) if mode == 4 else b.reshape(-1, shape[-1]))
    lim = mx.array([0.0 if limit is None else limit], mx.float32)
    warps = a2.shape[0] * (shape[-1] // block)
    out = _kernel(block)(
        inputs=[a2, b2, signs.reshape(-1), lim],
        # MLX names an instance by its template, so the dtypes must be in it.
        template=[
            ("BLK", block),
            ("MODE", mode),
            ("TA", a2.dtype),
            ("TO", dtype),
        ],
        grid=(-(-warps // 8) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[a2.shape],
        output_dtypes=[dtype],
    )[0]
    return out.reshape(shape)


_MATCHES: list[bool] = []


def _kernel_matches() -> bool:
    """Whether the rotation kernel gives the bits of ``reference_rotate`` here.

    Checked once a process, at the first rotation.
    """
    if not _MATCHES:
        key = mx.random.key(0)
        x = (mx.random.normal((64, 512), key=key) * 3).astype(mx.bfloat16)
        signs = mx.where(mx.random.uniform(shape=(512,), key=key) < 0.5, -1.0, 1.0)
        signs = signs.astype(mx.bfloat16)
        ok = all(
            mx.array_equal(
                _rotate(x, None, signs, block, 0, None, mx.bfloat16),
                reference_rotate(x, signs, block, mx.bfloat16),
            ).item()
            for block in (128, 256)
        )
        if not ok:
            logger.warning("oQh rotation kernel differs; using the reference")
        _MATCHES.append(ok)
    return _MATCHES[0]


def rotate(x: mx.array, signs: mx.array, block: int, dtype=None) -> mx.array:
    """x rotated into an oQh basis; a float32 x rounds once, to ``dtype``."""
    dtype = dtype or (signs.dtype if x.dtype == mx.float32 else x.dtype)
    if not _kernel_matches():
        return reference_rotate(x, signs, block, dtype)
    return _rotate(x, None, signs, block, 0, None, dtype)


def act_rotate(
    gate: mx.array, up: mx.array, signs: mx.array, block: int, limit=None
) -> mx.array:
    """silu(gate) * up in float32, rotated, rounded once (clamped with ``limit``)."""
    if not _kernel_matches():
        g, u = gate.astype(mx.float32), up.astype(mx.float32)
        if limit is not None:
            g, u = mx.minimum(g, limit), mx.clip(u, -limit, limit)
        return reference_rotate((g / (1.0 + mx.exp(-g))) * u, signs, block, gate.dtype)
    mode = 1 if limit is None else 2
    return _rotate(gate, up, signs, block, mode, limit, gate.dtype)


def gate_rotate(normed: mx.array, z: mx.array, signs: mx.array, block: int) -> mx.array:
    """normed * silu(z) in float32, rotated, rounded once: the host's gated norm."""
    if not _kernel_matches():
        zf = z.astype(mx.float32)
        gated = normed.astype(mx.float32) * (zf / (1.0 + mx.exp(-zf)))
        return reference_rotate(gated, signs, block, normed.dtype)
    return _rotate(normed, z, signs, block, 3, None, normed.dtype)


def scale_rotate(
    x: mx.array, row_scales: mx.array, signs: mx.array, block: int, dtype
) -> mx.array:
    """x times its row's scale, rounded to ``dtype``, then rotated: the input of
    a down projection whose host rounds the weighted activation first."""
    if not _kernel_matches():
        y = (x.astype(mx.float32) * row_scales.astype(mx.float32)[:, None]).astype(
            dtype
        )
        return reference_rotate(y, signs, block, dtype)
    return _rotate(x, row_scales, signs, block, 4, None, dtype)


def norm_fits(width: int, block: int) -> bool:
    """Whether ``norm_rotate`` takes rows of this width: one thread a 8 values."""
    return block == 256 and width % 256 == 0 and width <= 8192


def _norm_kernel(x, weight, signs, eps):
    if "norm" not in _KERNELS:
        _KERNELS["norm"] = mx.fast.cuda_kernel(
            name="oqh_norm_rotate",
            input_names=["x", "w", "signs", "eps"],
            output_names=["out"],
            source=_NORM_ROTATE,
            header=_HEADER,
        )
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    out = _KERNELS["norm"](
        inputs=[x2, weight, signs.reshape(-1), mx.array([eps], mx.float32)],
        template=[("T", x2.dtype)],
        grid=(x2.shape[0] * shape[-1] // 8, 1, 1),
        threadgroup=(shape[-1] // 8, 1, 1),
        output_shapes=[x2.shape],
        output_dtypes=[x2.dtype],
    )[0]
    return out.reshape(shape)


def _reference_norm(x, weight, eps):
    """The host's oqh_norm on plain ops: w * T(x * inv), rounded in T."""
    xf = x.astype(mx.float32)
    inv = mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return weight * (xf * inv).astype(x.dtype)


_NORM_MATCHES: list[bool] = []


def norm_rotate(x, weight, signs, block: int, eps: float) -> mx.array:
    """RMSNorm(x) * weight rotated for its oQh readers (see ``norm_fits``)."""
    if not _NORM_MATCHES:
        key = mx.random.key(1)
        t = (mx.random.normal((32, 1024), key=key) * 2).astype(mx.bfloat16)
        w = (1 + 0.1 * mx.random.normal((1024,), key=key)).astype(mx.bfloat16)
        s = mx.where(mx.random.uniform(shape=(1024,), key=key) < 0.5, -1.0, 1.0)
        s = s.astype(mx.bfloat16)
        got = _norm_kernel(t, w, s, 1e-6)
        want = reference_rotate(_reference_norm(t, w, 1e-6), s, 256, mx.bfloat16)
        a, b = got.astype(mx.float32), want.astype(mx.float32)
        error = (mx.linalg.norm(a - b) / mx.linalg.norm(b)).item()
        # The row sums take another order; a few outputs round the other way.
        ok = error < 1e-3 and _kernel_matches()
        if not ok:
            logger.warning("oQh norm kernel differs (%.2e); using the reference", error)
        _NORM_MATCHES.append(ok)
    if not _NORM_MATCHES[0]:
        return reference_rotate(_reference_norm(x, weight, eps), signs, block, x.dtype)
    return _norm_kernel(x, weight, signs, eps)


def check_shared(*linears: Any) -> None:
    """Refuse oQh readers of one input with different rotations: they share
    one rotated copy."""
    rotations = [lin.rotation for lin in linears if lin.rotation is not None]
    for signs, block in rotations[1:]:
        first, size = rotations[0]
        if block != size or not mx.array_equal(signs, first).item():
            raise ValueError("oQh linears that read one input rotate it differently")


def reference_rotate(x: mx.array, signs: mx.array, block: int, dtype) -> mx.array:
    """``rotate`` on plain ops, in the same butterfly order."""
    shape = x.shape
    v = x.astype(mx.float32).reshape(-1, shape[-1] // block, block)
    v = v * signs.astype(mx.float32).reshape(-1, block)
    h = 1
    while h < block:
        v = v.reshape(*v.shape[:-1], block // (2 * h), 2, h)
        lo, hi = v[..., 0, :], v[..., 1, :]
        v = mx.stack([lo + hi, lo - hi], axis=-2).reshape(*v.shape[:-3], block)
        h *= 2
    scale = float(np.float32(1.0 / np.sqrt(block)))
    return (v * scale).astype(dtype).reshape(shape)
