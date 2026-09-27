"""Zero-copy shard descriptors for flat mlx-serve n-gram tables.

This adapts the storage metadata only; oMLX's existing disk reader, prefetch,
affine dequantization and lifecycle remain responsible for serving rows.
"""

import json
import struct
from pathlib import Path


def flat_ngram_views(path, prefix, shard_sizes, dims, config):
    path = Path(path)
    file_size = path.stat().st_size
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("Truncated n-gram header")
        size = struct.unpack("<Q", raw)[0]
        if size > min(16 * 1024**2, file_size - 8):
            raise ValueError("Invalid n-gram header size")
        header = json.loads(f.read(size))
    meta = header.get("__metadata__", {})
    if meta.get("format") != "mlx-serve-ngram":
        raise ValueError("Unsupported flat n-gram format")
    bits, group = int(meta.get("bits", 0)), int(meta.get("group_size", 0))
    if bits not in (2, 3, 4, 5, 6, 8) or group not in (32, 64, 128):
        raise ValueError("Unsupported flat n-gram quantization")
    if bits != config.get("bits") or group != config.get("group_size"):
        raise ValueError("N-gram configuration differs from table metadata")
    if dims % group or dims * bits % 32:
        raise ValueError("Invalid flat n-gram embedding width")
    if not shard_sizes or any(type(n) is not int or n <= 0 for n in shard_sizes):
        raise ValueError("Invalid n-gram shard sizes")
    rows = sum(shard_sizes)
    columns = {
        "weight": dims * bits // 32,
        "scales": dims // group,
        "biases": dims // group,
    }
    regions = []
    views = {}
    for part, width in columns.items():
        entry = header[part]
        dtype = entry["dtype"]
        if (part == "weight" and dtype != "U32") or (
            part != "weight" and dtype not in ("F16", "BF16", "F32")
        ):
            raise ValueError("Invalid flat n-gram dtype")
        itemsize = 4 if dtype in ("U32", "F32") else 2
        if entry["shape"] != [rows, width]:
            raise ValueError("Invalid flat n-gram tensor shape")
        offsets = entry["data_offsets"]
        if len(offsets) != 2 or any(type(n) is not int for n in offsets):
            raise ValueError("Invalid flat n-gram tensor offsets")
        start, end = offsets
        if (
            start < 0
            or end - start != rows * width * itemsize
            or 8 + size + end > file_size
        ):
            raise ValueError("Truncated or invalid flat n-gram tensor region")
        if any(
            start < other_end and other_start < end
            for other_start, other_end in regions
        ):
            raise ValueError("Overlapping flat n-gram tensor regions")
        regions.append((start, end))
        row_start = 0
        for shard, count in enumerate(shard_sizes):
            key = f"{prefix}.shard_{shard}.{part}"
            byte_start = start + row_start * width * itemsize
            views[key] = {
                "dtype": dtype,
                "shape": [count, width],
                "data_offsets": [byte_start, byte_start + count * width * itemsize],
            }
            row_start += count
    return views
