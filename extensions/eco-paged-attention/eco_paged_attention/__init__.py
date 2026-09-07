"""ECO paged MHA/GQA decode extension for MLX 0.32.2."""

import math

import mlx.core as mx

from . import _ext

HEAD_DIMS = frozenset((64, 128, 256))
GQA_FACTORS = frozenset((1, 2, 3, 4, 5, 6, 8, 12, 16))


def supports_geometry(query_heads, kv_heads, head_dim):
    return (
        kv_heads > 0
        and query_heads > 0
        and query_heads % kv_heads == 0
        and query_heads // kv_heads in GQA_FACTORS
        and head_dim in HEAD_DIMS
    )


def _paged_attention(q, k, v, tables, lengths, *, scale, stream=None, partitions=0):
    """Internal entry point for cache-owned, validated page metadata."""
    if not all(isinstance(x, mx.array) for x in (q, k, v, tables, lengths)):
        raise TypeError("Inputs must be MLX arrays")
    if not math.isfinite(scale):
        raise ValueError("Scale must be finite")
    if partitions not in (0, 32, 64, 128, 256, 512):
        raise ValueError("Partitions must be 0 (auto) or a power of two from 32 to 512")
    if q.ndim != 4 or q.shape[2] != 1 or q.shape[3] not in HEAD_DIMS:
        raise ValueError("Expected queries [batch, heads, 1, D], D in 64/128/256")
    if k.ndim != 4 or k.shape != v.shape or k.shape[-1] != q.shape[-1]:
        raise ValueError("Expected matching KV pages [pages, kv_heads, page_size, D]")
    if not supports_geometry(q.shape[1], k.shape[1], q.shape[-1]):
        raise ValueError("Unsupported query/KV head ratio or head dimension")
    if (
        q.dtype not in (mx.float32, mx.float16, mx.bfloat16)
        or k.dtype != q.dtype
        or v.dtype != q.dtype
    ):
        raise ValueError("Expected matching float32, float16 or bfloat16 inputs")
    if (
        tables.ndim != 2
        or tables.shape[0] != q.shape[0]
        or lengths.shape != (q.shape[0],)
    ):
        raise ValueError("Invalid page metadata shape")
    if tables.dtype != mx.uint32 or lengths.dtype != mx.uint32:
        raise ValueError("Page metadata must use uint32")
    if k.shape[0] == 0 or k.shape[2] == 0 or tables.shape[1] == 0 or q.shape[0] == 0:
        raise ValueError("Empty page pools and batches are unsupported")
    stream = mx.default_stream(mx.gpu) if stream is None else stream
    if not isinstance(stream, mx.Stream) or stream.device != mx.gpu:
        raise ValueError("Expected a GPU stream")
    out = mx.zeros(q.shape, dtype=q.dtype)
    _ext.paged_attention(q, k, v, tables, lengths, scale, out, stream, partitions)
    return out


def paged_attention(q, k, v, tables, lengths, *, scale, stream=None):
    """Decode over paged KV. Metadata validation synchronizes with the CPU."""
    # Validate shapes and dtypes before reading metadata. Evaluation remains lazy.
    out = _paged_attention(q, k, v, tables, lengths, scale=scale, stream=stream)
    rows = tables.tolist()
    for row, length in zip(rows, lengths.tolist()):
        if not 0 < length <= len(row) * k.shape[2]:
            raise ValueError("Context lengths must fit the page table and be positive")
        if any(
            page >= k.shape[0]
            for page in row[: (length + k.shape[2] - 1) // k.shape[2]]
        ):
            raise ValueError("Physical page index is outside the pool")
    return out
