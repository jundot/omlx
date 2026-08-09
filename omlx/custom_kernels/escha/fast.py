
"""Escha-MLX trellis kernels: decode-on-the-fly EXL3 qgemm via Metal.

Pure-Python path (no compiled extension): the MSL is JIT-compiled by
``mx.fast.metal_kernel`` at first use, exactly like higgs does through the
mlx C API. ``eschamoe_gather_qgemm`` computes, for token rows whose expert id
is ``eids[m]`` (sorted ascending), the fused decode+GEMM::

    dst[m, :] = xh[m, :] @ Wtilde[ eids[m] ]

where ``Wtilde`` is the raw EXL3 trellis decode of ``code[e]`` (no Hadamard,
no scales - the caller applies the blockwise Hadamard and rin/rout on the
activations, see omlx.patches.escha_trellis).

Verification: the kernel is bit-identical to the reference runtime; the CPU
reference ``ref_trellis`` here matches the kernel to one f16 rounding step
(the MSL stages decoded weights as f16, matching the escha runtime).
"""
from __future__ import annotations

import math
import re

import mlx.core as mx

from .msl import MSL

# Codebook 1 (MCG) constants -- exllamav3 decode_3inst<1>.
MCG_MULT = 0xCBAC1FED
MCG_ADD = 0
MCG_MASK = 0x8FFF8FFF
MCG_XOR = 0x3B603B60

_QMV_MSL = """constexpr int WORDS = 8 * K;
constexpr int IN = TK * 16;
constexpr int OUT = TN * 16;

threadgroup float x_sh[TK * 16];

uint row = threadgroup_position_in_grid.y;
uint tid = thread_index_in_threadgroup;
uint sg = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;

// Stage the transformed activation row of this expert.
for (uint i = tid; i < uint(IN); i += 128u) {
    x_sh[i] = xh[row * uint(IN) + i];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

uint o = threadgroup_position_in_grid.x * 4u + sg;
if (o >= uint(OUT)) {
    return;
}

// Split the output index into a tile column and a slot column.
uint tn = o >> 4;
uint c = o & 15u;
uint cb2 = (c >> 3) & 1u;
uint c7 = c & 7u;

const device short* base =
    code + ulong(eids[row]) * ulong(TK) * ulong(TN) * ulong(16 * K);

// Each lane owns one code pair. The pair index inverts the closed form
// of tile_perm for a fixed column. One pair gives two adjacent rows.
uint q = lane & 3u;
uint rh = (lane >> 2) & 1u;
uint t = 4u * (4u * c7 + q) + 2u * cb2 + rh;
uint r0 = 8u * rh + 2u * q;

// The bit offsets copy unpack_tile. The wrap term 256 * K comes before
// the term -16. Thus the unsigned value stays 0 or more.
uint b0 = 2u * t * uint(K) + uint(K) + 256u * uint(K) - 16u;
uint b2 = b0 + uint(K) + 16u;
uint i0 = (b0 / 32u) % uint(WORDS);
uint i1w = (b2 - 1u) / 32u;
uint s1 = (i1w + 1u) * 32u - b2;
uint i1 = i1w % uint(WORDS);

float acc = 0.0f;
for (uint tk = lane >> 3; tk < uint(TK); tk += 4u) {
    const device short* tile = base + (tk * uint(TN) + tn) * uint(16 * K);
    uint w0 = uint(ushort(tile[2u * i0])) | (uint(ushort(tile[2u * i0 + 1u])) << 16);
    uint wb = uint(ushort(tile[2u * i1])) | (uint(ushort(tile[2u * i1 + 1u])) << 16);

    // The 64-bit funnel makes the shift safe when s1 is 0.
    ulong pair = (ulong(w0) << 32) | ulong(wb);
    uint w1 = uint(pair >> s1);

    // The codebook hash. Refer to the tile decode kernel. The half cast
    // repeats the f16 round of the CPU decode.
    uint x0 = ((w1 >> uint(K)) & 0xFFFFu) * cb[0] + cb[1];
    x0 = (x0 & cb[2]) ^ cb[3];
    uint x1 = (w1 & 0xFFFFu) * cb[0] + cb[1];
    x1 = (x1 & cb[2]) ^ cb[3];
    half2 h0 = as_type<half2>(x0);
    half2 h1 = as_type<half2>(x1);
    float v0 = float(h0.x) + float(h0.y);
    float v1 = float(h1.x) + float(h1.y);

    acc = fma(x_sh[tk * 16u + r0], v0, acc);
    acc = fma(x_sh[tk * 16u + r0 + 1u], v1, acc);
}

acc = simd_sum(acc);
if (lane == 0u) {
    dst[row * uint(OUT) + o] = acc;
}"""
_kernel_cache: dict[tuple[int, int, int], object] = {}
_fused_cache: dict = {}
_qmv_cache: dict[tuple[int, int, int], object] = {}
_IMPORT_ERROR = None


def is_native_available() -> bool:
    """Metal custom kernels require mlx built with the fast path (always on
    Apple silicon wheels)."""
    return _IMPORT_ERROR is None and hasattr(mx.fast, "metal_kernel")


def import_error():
    return _IMPORT_ERROR


def has_symbol(name: str) -> bool:
    return name in native_symbols()


def native_symbols() -> tuple[str, ...]:
    if not is_native_available():
        return ()
    return ("eschamoe_gather_qgemm", "eschamoe_gather_qmv")


def missing_symbols(required: tuple[str, ...]) -> list[str]:
    return [name for name in required if not has_symbol(name)]


def eschamoe_kernel(K: int, TK: int, TN: int):
    """Compile (and cache) the trellis GEMM kernel for one (K, TK, TN)."""
    key = (int(K), int(TK), int(TN))
    kern = _kernel_cache.get(key)
    if kern is not None:
        return kern
    body = re.sub(r"\bK\b", str(K), MSL)
    body = re.sub(r"\bTK\b", str(TK), body)
    body = re.sub(r"\bTN\b", str(TN), body)
    header = "#include <metal_stdlib>\nusing namespace metal;\n"
    kern = mx.fast.metal_kernel(
        name="escha_qgemm",
        input_names=["xh", "code", "eids", "cb"],
        output_names=["dst"],
        header=header,
        source=body,
    )
    _kernel_cache[key] = kern
    return kern


def eschamoe_gather_qgemm(
    xh: mx.array,
    code: mx.array,
    eids: mx.array,
    K: int,
) -> mx.array:
    """Fused trellis decode + GEMM for gathered tokens.

    Args:
        xh: float32 [rows, TK*16] -- activations pre-transformed (had(x*rin)).
        code: int16 [E, TK, TN, 16*K] packed codes for all experts.
        eids: uint32/int32 [rows] sorted ascending; rows if sorted by expert.
        K: bits per weight (2 or 3).
    Returns:
        float32 [rows, TN*16] = xh @ Wtilde[expert].
    """
    if K not in (2, 3):
        raise ValueError(f"eschamoe trellis K must be 2 or 3, got {K}")
    rows, IN = xh.shape
    TK = code.shape[1]
    TN = code.shape[2]
    kernel = eschamoe_kernel(K, TK, TN)
    cb = mx.array(
        [MCG_MULT, MCG_ADD, MCG_MASK, MCG_XOR, rows], mx.uint32
    )
    (out,) = kernel(
        inputs=[xh, code, eids, cb],
        grid=(math.ceil(TN * 16 / 128) * 128, (rows + 31) // 32, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(rows, TN * 16)],
        output_dtypes=[mx.float32],
    )
    return out



def eschamoe_gather_qmv(xh, code, eids, K):
    """Small-batch variant: one threadgroup per row. Efficient for decode
    (one token -> 8 routed rows); same decode contract as the GEMM kernel."""
    if K not in (2, 3):
        raise ValueError(f"eschamoe trellis K must be 2 or 3, got {K}")
    rows, IN = xh.shape
    TK, TN = code.shape[1], code.shape[2]
    groups_x = (TN * 16 + 3) // 4
    cb = mx.array([MCG_MULT, MCG_ADD, MCG_MASK, MCG_XOR], mx.uint32)
    kern = _qmv_cache.get((K, TK, TN))
    if kern is None:
        body = re.sub(r"\bK\b", str(K), _QMV_MSL)
        body = re.sub(r"\bTK\b", str(TK), body)
        body = re.sub(r"\bTN\b", str(TN), body)
        header = "#include <metal_stdlib>\nusing namespace metal;\n"
        kern = mx.fast.metal_kernel(
            name="escha_qmv",
            input_names=["xh", "code", "eids", "cb"],
            output_names=["dst"],
            header=header,
            source=body,
        )
        _qmv_cache[(K, TK, TN)] = kern
    (out,) = kern(
        inputs=[xh, code, eids, cb],
        grid=(groups_x * 128, rows, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(rows, TN * 16)],
        output_dtypes=[mx.float32],
    )
    return out



_FUSED_HDR = '\n\ninline void had_shared(threadgroup float* sh, uint off, uint nb, uint lane, uint grp) {\n    uint b = grp % nb;                       // uniform barrier participation\n    uint base = off + b * 128u;\n    for (uint s = 1u; s < 128u; s <<= 1u) {\n        threadgroup_barrier(mem_flags::mem_threadgroup);\n        float v = sh[base + lane];\n        float o = sh[base + (lane ^ s)];\n        threadgroup_barrier(mem_flags::mem_threadgroup);\n        sh[base + lane] = (lane & s) ? o - v : v + o;\n    }\n}\n\ninline float2 pair_vals(const device short* tile, uint t, uint Kk, constant uint* cb) {\n    uint b0 = 2u * t * Kk + Kk + 256u * Kk - 16u;\n    uint b2 = b0 + Kk + 16u;\n    uint i0 = (b0 / 32u) % uint(8 * Kk);\n    uint i1w = (b2 - 1u) / 32u;\n    uint s1 = (i1w + 1u) * 32u - b2;\n    uint i1 = i1w % uint(8 * Kk);\n    uint w0 = uint(ushort(tile[2u * i0])) | (uint(ushort(tile[2u * i0 + 1u])) << 16);\n    uint wb = uint(ushort(tile[2u * i1])) | (uint(ushort(tile[2u * i1 + 1u])) << 16);\n    ulong pair = (ulong(w0) << 32) | ulong(wb);\n    uint w1 = uint(pair >> s1);\n    uint x0 = ((w1 >> Kk) & 0xFFFFu) * cb[0] + cb[1];\n    x0 = (x0 & cb[2]) ^ cb[3];\n    uint x1 = (w1 & 0xFFFFu) * cb[0] + cb[1];\n    x1 = (x1 & cb[2]) ^ cb[3];\n    half2 h0 = as_type<half2>(x0);\n    half2 h1 = as_type<half2>(x1);\n    return float2(float(h0.x) + float(h0.y), float(h1.x) + float(h1.y));\n}\n'
_FUSED_BODY = '\nthreadgroup float x_sh[2048];\nthreadgroup float gu_sh[1024];\nthreadgroup float a_sh[512];\nconstexpr float INV_SQRT128 = 0.08838834764;\n\nuint row = thread_position_in_grid.y;\nuint tid = thread_position_in_grid.x;\nuint lane = tid & 127u;\nuint grp = tid >> 7u;\nuint e = eids[row];\n\nfor (uint i = tid; i < 2048u; i += 1024u) x_sh[i] = xh1[row * 2048u + i];\nthreadgroup_barrier(mem_flags::mem_threadgroup);\n\nconst device short* base1 = code1 + (ulong)e * 128ul * 64ul * 32ul;\nuint c = tid;                                   // 1024 threads -> one gu col each\nuint tn = c >> 4u;\nuint cs = c & 15u;\nuint cb2 = (cs >> 3) & 1u;\nuint c7 = cs & 7u;\nfloat acc = 0.0f;\nfor (uint tk = 0u; tk < 128u; ++tk) {\n    const device short* tile = base1 + (tk * 64u + tn) * 32u;\n    for (uint q = 0u; q < 4u; ++q) {\n        for (uint rh = 0u; rh < 2u; ++rh) {\n            uint t = 4u * (4u * c7 + q) + 2u * cb2 + rh;\n            float2 vv = pair_vals(tile, t, 2u, cb);\n            uint r0 = 8u * rh + 2u * q;\n            uint k0 = tk * 16u + r0;\n            acc = fma(x_sh[k0], vv.x, acc);\n            acc = fma(x_sh[k0 + 1u], vv.y, acc);\n        }\n    }\n}\ngu_sh[c] = acc;\nthreadgroup_barrier(mem_flags::mem_threadgroup);\nhad_shared(gu_sh, 0u, 8u, lane, grp);\nthreadgroup_barrier(mem_flags::mem_threadgroup);\ngu_sh[tid] = gu_sh[tid] * INV_SQRT128 * rout1[row * 1024u + tid];\nthreadgroup_barrier(mem_flags::mem_threadgroup);\nif (tid < 512u) {\n    float gate = gu_sh[tid];\n    float up = gu_sh[tid + 512u];\n    float sg = gate / (1.0f + metal::exp(-gate));\n    a_sh[tid] = sg * up * rin2[row * 512u + tid];\n}\nthreadgroup_barrier(mem_flags::mem_threadgroup);\nhad_shared(a_sh, 0u, 4u, lane, grp);\nthreadgroup_barrier(mem_flags::mem_threadgroup);\nif (tid < 512u) a_sh[tid] *= INV_SQRT128;\nthreadgroup_barrier(mem_flags::mem_threadgroup);\n\nconst device short* base2 = code2 + (ulong)e * 32ul * 128ul * 48ul;\nfor (uint cc = 0u; cc < 2u; ++cc) {\n    uint c2 = tid + cc * 1024u;\n    uint tn2 = c2 >> 4u;\n    uint cs2 = c2 & 15u;\n    uint cb22 = (cs2 >> 3) & 1u;\n    uint c72 = cs2 & 7u;\n    float acc2 = 0.0f;\n    for (uint tk = 0u; tk < 32u; ++tk) {\n        const device short* tile = base2 + (tk * 128u + tn2) * 48u;\n        for (uint q = 0u; q < 4u; ++q) {\n            for (uint rh = 0u; rh < 2u; ++rh) {\n                uint t = 4u * (4u * c72 + q) + 2u * cb22 + rh;\n                float2 vv = pair_vals(tile, t, 3u, cb);\n                uint r0 = 8u * rh + 2u * q;\n                uint k0 = tk * 16u + r0;\n                acc2 = fma(a_sh[k0], vv.x, acc2);\n                acc2 = fma(a_sh[k0 + 1u], vv.y, acc2);\n            }\n        }\n    }\n    dst[row * 2048u + c2] = acc2;\n}\n'


def eschamoe_fused_layer(xh1, code1, code2, eids, rout1, rin2, rout2):
    """One kernel per MoE layer: decode gate_up (2-bit) -> in-group had ->
    SwiGLU -> decode down (3-bit) -> GEMMs. Returns y_pre [rows, 2048]; the
    host applies the final had128(* rout2). 8 inputs, 1 output, 512 threads
    per row."""
    kern = _fused_cache.get(None)
    if kern is None:
        kern = mx.fast.metal_kernel(
            name="escha_fused_layer",
            input_names=["xh1", "code1", "code2", "eids", "rout1", "rin2", "rout2", "cb"],
            output_names=["dst"],
            header=_FUSED_HDR,
            source=_FUSED_BODY,
        )
        _fused_cache[None] = kern
    rows = xh1.shape[0]
    cb = mx.array([MCG_MULT, MCG_ADD, MCG_MASK, MCG_XOR], mx.uint32)
    (dst,) = kern(
        inputs=[xh1, code1, code2, eids, rout1, rin2, rout2, cb],
        grid=(1024, rows, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(rows, 2048)],
        output_dtypes=[mx.float32],
    )
    return dst


# --------------------------------------------------------------------------
# Pure-MLX batch decode: convert packed codes to dense baked weights.
# (No custom kernel -- use for decode-on-first-use caches and tests.)
# --------------------------------------------------------------------------

def decode_3inst_mx(codes):
    x = codes.astype(mx.uint32) * mx.array(0xCBAC1FED, mx.uint32)
    x = (x & mx.array(0x8FFF8FFF, mx.uint32)) ^ mx.array(0x3B603B60, mx.uint32)
    lo = x & mx.array(0xFFFF, mx.uint32)
    hi = (x >> mx.array(16, mx.uint32)) & mx.array(0xFFFF, mx.uint32)
    vals = []
    for b in (lo, hi):
        bf = b.astype(mx.float32)
        s = mx.floor(bf / 32768.0)
        e = mx.floor((bf % 32768.0) / 1024.0)
        m = bf % 1024.0
        mant = m + mx.where(e > 0, 1024.0, 0.0)
        exp2 = mx.where(e == 0, -24.0, e - 25.0)
        vals.append((1.0 - 2.0 * s) * mant * mx.power(2.0, exp2))
    return vals[0] + vals[1]


def _had_axis(w, axis):
    """H128 blocks along `axis` of [N, a, b]; pure mlx ops."""
    w = mx.moveaxis(w, axis, -1)
    lead = w.shape[:-1]
    n = w.shape[-1]
    y = w.reshape(*lead, n // 128, 128)
    y = mx.matmul(y.reshape(-1, n // 128, 128), _h128_constant())
    return mx.moveaxis(y.reshape(*lead, n), -1, axis)


_H128C = None


def _h128_constant():
    global _H128C
    if _H128C is None:
        h = mx.array([[1.0]])
        while h.shape[0] < 128:
            h = mx.concatenate([mx.concatenate([h, h], 1), mx.concatenate([h, -h], 1)], 0)
        _H128C = h * (1.0 / (128.0 ** 0.5))
    return _H128C


def decode_experts_dense(code, rin, rout, K):
    """Batch decode + bake routed experts.

    Args:
        code: int16 [n, TK, TN, 16K] packed codes.
        rin: f16 [n, in]; rout: f16 [n, out].
    Returns:
        bf16 [n, out, in] dense expert matrices
        (W = (H Wt H * rin * rout).T, matching the reference runtime).
    """
    import numpy as _np
    n = code.shape[0]
    codes = mx.array(_unpack_trellis(_np.asarray(code), K))     # uint16
    vals = decode_3inst_mx(codes.astype(mx.uint32))            # [n,tk,tn,256]
    perm = mx.array(_PERM_INV, mx.int32)
    idx = mx.broadcast_to(perm[None, None, None, :], vals.shape)
    tiles = mx.take_along_axis(vals, idx, axis=-1)
    tk, tn = code.shape[1], code.shape[2]
    wt = tiles.reshape(n, tk, tn, 16, 16).transpose(0, 1, 3, 2, 4).reshape(n, tk * 16, tn * 16)
    wt = _had_axis(_had_axis(wt, 1), 2)
    wt = wt * rin[:, :, None].astype(mx.float32) * rout[:, None, :].astype(mx.float32)
    return mx.transpose(wt, (0, 2, 1)).astype(mx.bfloat16)


# --------------------------------------------------------------------------
# CPU reference decode (bit-exact vs the escha wheel when stored as f16)
# --------------------------------------------------------------------------

def _tensor_core_perm():
    perm = [0] * 256
    for t in range(32):
        r0 = (t % 4) * 2
        c0 = t // 4
        rows = (r0, r0 + 1, r0 + 8, r0 + 9)
        for j, c in enumerate((c0, c0 + 8)):
            for i, r in enumerate(rows):
                perm[t * 8 + j * 4 + i] = r * 16 + c
    return perm


_PERM = _tensor_core_perm()


def _perm_inverse():
    inv = [0] * 256
    for i, j in enumerate(_PERM):
        inv[j] = i
    return inv


_PERM_INV = _perm_inverse()


def _unpack_trellis(packed, k):
    import numpy as np
    lead = packed.shape[:-1]
    u32 = packed.reshape(-1, 16 * k).view(np.uint32)
    n_words = k * 256 // 32
    t = np.arange(128)
    b0 = t * 2 * k + k - 16 + 256 * k
    b2 = b0 + k + 16
    s1 = ((b2 - 1) // 32 + 1) * 32 - b2
    i0 = (b0 // 32) % n_words
    i1 = ((b2 - 1) // 32) % n_words
    a = u32[:, i0].astype(np.uint64)
    b = u32[:, i1].astype(np.uint64)
    w1 = (((a << np.uint64(32)) | b) >> s1.astype(np.uint64)).astype(np.uint32)
    w0 = (w1 >> np.uint32(k)) & np.uint32(0xFFFF)
    w1 = w1 & np.uint32(0xFFFF)
    codes = np.empty((u32.shape[0], 256), dtype=np.uint16)
    codes[:, 0::2] = w0
    codes[:, 1::2] = w1
    return codes.reshape(*lead, 256)


def _decode_3inst(codes):
    import numpy as np
    x = codes.astype(np.uint32) * np.uint32(MCG_MULT)
    x = (x & np.uint32(MCG_MASK)) ^ np.uint32(MCG_XOR)
    halves = x.view(np.uint16).reshape(*x.shape, 2).astype(np.uint16)
    return halves.view(np.float16).astype(np.float32).sum(axis=-1)


def ref_trellis(code, K):
    """Decode one expert: code [TK, TN, 16*K] int16 -> [TK*16, TN*16] f32
    (the raw Wtilde, matching the Metal kernel to f16 rounding)."""
    import numpy as np
    tk, tn = code.shape[0], code.shape[1]
    vals = _decode_3inst(_unpack_trellis(code, K))
    tiles = np.empty_like(vals)
    tiles[..., _PERM] = vals
    return tiles.reshape(tk, tn, 16, 16).transpose(0, 2, 1, 3).reshape(tk * 16, tn * 16)
