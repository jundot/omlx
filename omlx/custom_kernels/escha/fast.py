
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

import re

import mlx.core as mx

from .msl import MSL

# Codebook 1 (MCG) constants -- exllamav3 decode_3inst<1>.
MCG_MULT = 0xCBAC1FED
MCG_ADD = 0
MCG_MASK = 0x8FFF8FFF
MCG_XOR = 0x3B603B60

_IMPORT_ERROR = None
try:
    _kernel_cache: dict[tuple[int, int, int], object] = {}
except Exception as exc:  # pragma: no cover
    _IMPORT_ERROR = exc


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
        grid=(TN * 128, (rows + 31) // 32, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(rows, TN * 16)],
        output_dtypes=[mx.float32],
    )
    return out


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
