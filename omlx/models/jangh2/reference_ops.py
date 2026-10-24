"""Independent, chunked JANGH2 arithmetic shared by NumPy tests and MLX.

This is a correctness-first fallback, not a replacement for fused Metal kernels.
It only expands selected experts for a bounded number of token/expert routes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np


def pack_codes(codes: Any, bits: int, xp=np):
    """Test/fixture utility: pack an LSB-first row bitstream, including 3-bit spills."""
    q = np.asarray(codes)
    if bits not in (2, 3, 4) or q.shape[-1] % 32:
        raise ValueError("Need 2/3/4-bit codes and a row width divisible by 32")
    if np.any(q < 0) or np.any(q >= (1 << bits)):
        raise ValueError("Code outside the codebook")
    out = np.zeros((*q.shape[:-1], q.shape[-1] * bits // 32), dtype=np.uint32)
    for i in range(q.shape[-1]):
        word, shift = divmod(i * bits, 32)
        v = q[..., i].astype(np.uint32)
        out[..., word] |= v << np.uint32(shift)
        if shift + bits > 32:
            out[..., word + 1] |= v >> np.uint32(32 - shift)
    return xp.array(out, dtype=xp.uint32)


def unpack_codes(packed: Any, bits: int, width: int, xp=np):
    if bits not in (2, 3, 4) or width % 32 or packed.shape[-1] != width * bits // 32:
        raise ValueError("Invalid packed shape or code width")
    positions = xp.arange(width, dtype=xp.uint32) * bits
    word, shift = positions // 32, positions % 32
    a = xp.take(packed, word, axis=-1) >> shift
    # Clamp the otherwise-unused final read. Shifts are always 0..31 even
    # where a spill is masked out, avoiding undefined native shifts by 32.
    b = xp.take(packed, xp.minimum(word + 1, packed.shape[-1] - 1), axis=-1)
    spill = xp.where(
        shift + bits > 32, b << ((32 - shift) % 32), xp.array(0, dtype=xp.uint32)
    )
    return (a | spill) & ((1 << bits) - 1)


def hadamard32(x: Any, xp=np, *, optimized: bool = False):
    if x.shape[-1] % 32:
        raise ValueError("Hadamard-32 requires a last dimension divisible by 32")
    shape = x.shape
    a = x.astype(xp.float32).reshape(-1, 32)
    # MLX exposes a fused Walsh-Hadamard primitive. Keeping NumPy's
    # explicit butterfly fallback makes tests independent of Metal.
    if optimized and callable(getattr(xp, "hadamard_transform", None)):
        return xp.hadamard_transform(a).reshape(shape)
    step = 1
    while step < 32:
        b = a.reshape(-1, 32 // (2 * step), 2, step)
        left, right = b[:, :, 0, :], b[:, :, 1, :]
        a = xp.concatenate((left + right, left - right), axis=-1).reshape(-1, 32)
        step *= 2
    return (a * (1.0 / math.sqrt(32))).reshape(shape)


def _project(x, ids, spec, xp, *, input_rotated=False, optimized=False):
    packed, scales, bits, levels, rotation = spec
    if rotation == "hadamard32":
        if not input_rotated:
            x = hadamard32(x, xp, optimized=optimized)
    elif rotation != "none":
        raise ValueError(f"Unknown rotation {rotation!r}")
    chosen = xp.take(packed, ids, axis=0)
    codes = unpack_codes(chosen, bits, x.shape[-1], xp)
    w = (
        xp.take(levels, codes, axis=0)
        * xp.take(scales, ids, axis=0).astype(xp.float32)[..., None]
    )
    return xp.matmul(w, x.astype(xp.float32)[..., None])[..., 0]


def expert_forward(
    x: Any,
    indices: Any,
    projections: Mapping[str, Any],
    xp=np,
    *,
    chunk_size: int = 1,
    evaluation: str = "legacy",
):
    """Return unweighted per-route SwiGLU output shaped (..., top_k, model_dim)."""
    if x.ndim < 2 or indices.ndim != x.ndim or x.shape[:-1] != indices.shape[:-1]:
        raise ValueError("Token dimensions of x and expert indices must match")
    if evaluation not in ("legacy", "per_chunk"):
        raise ValueError("Unsupported evaluation strategy")
    if chunk_size < 1 or chunk_size > 64 or indices.shape[-1] < 1:
        raise ValueError("Invalid route chunk size or top-k")
    if set(projections) != {"gate_proj", "up_proj", "down_proj"}:
        raise ValueError("Expected a full gate/up/down projection triplet")
    if indices.size == 0:
        return xp.zeros((*indices.shape, x.shape[-1]), dtype=x.dtype)
    flat_ids = indices.reshape(-1)
    if "int" not in str(indices.dtype):
        raise ValueError("Expert indices must be integers")
    min_id, max_id = xp.min(flat_ids), xp.max(flat_ids)
    evaluate = getattr(xp, "eval", None)
    if evaluate is not None:
        evaluate(min_id, max_id)
    e = projections["gate_proj"][0].shape[0]
    if int(min_id.item()) < 0 or int(max_id.item()) >= e:
        raise ValueError("Expert index outside checkpoint range")
    xf = x.reshape(-1, x.shape[-1])
    top_k = indices.shape[-1]
    chunks = []
    for start in range(0, flat_ids.size, chunk_size):
        end = min(start + chunk_size, flat_ids.size)
        ids = flat_ids[start:end]
        rows = xp.arange(start, end, dtype=xp.int32) // top_k
        xx = xp.take(xf, rows, axis=0)
        # Gate and up use the same rotated activation in JANGH2. Compute
        # it once per route batch instead of twice. Do not assume equal
        # rotations for other checkpoints.
        optimized = evaluation != "legacy"
        common_rotation = (
            optimized
            and projections["gate_proj"][4] == "hadamard32"
            and projections["up_proj"][4] == "hadamard32"
        )
        shared_x = hadamard32(xx, xp, optimized=True) if common_rotation else xx
        gate = _project(
            shared_x,
            ids,
            projections["gate_proj"],
            xp,
            input_rotated=common_rotation,
            optimized=optimized,
        )
        up = _project(
            shared_x,
            ids,
            projections["up_proj"],
            xp,
            input_rotated=common_rotation,
            optimized=optimized,
        )
        # Stable sigmoid rather than dropping the rotation or replacing the
        # expert layer with ordinary affine quantization.
        exp_neg_abs = xp.exp(-xp.abs(gate))
        sigmoid = xp.where(
            gate >= 0, 1 / (1 + exp_neg_abs), exp_neg_abs / (1 + exp_neg_abs)
        )
        hidden = gate * sigmoid * up
        # Intermediate evaluation introduces a GPU dispatch/synchronization
        # boundary inside every routed expert. The fast path only evaluates
        # the finished route batch, keeping memory bounded.
        if evaluation == "legacy" and evaluate is not None:
            evaluate(hidden)
        del gate, up, xx, shared_x
        result = _project(
            hidden, ids, projections["down_proj"], xp, optimized=optimized
        ).astype(x.dtype)
        if evaluate is not None:
            evaluate(result)
        chunks.append(result)
        del hidden
    return xp.concatenate(chunks, axis=0).reshape(*indices.shape, x.shape[-1])
