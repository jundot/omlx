# SPDX-License-Identifier: Apache-2.0
"""Shared safetensors container primitives.

Three small pieces every shard-scanning patch used to grow privately:

- ``read_safetensors_header`` — the 8-byte little-endian length prefix +
  JSON header parse.
- ``SAFETENSORS_NUMPY_DTYPES`` — dtype tag -> numpy transport dtype.
  BF16 and the FP8 tags have no numpy equivalent; they travel as raw
  bits (uint16 / uint8) and are reinterpreted on the MLX side.
- ``file_signature`` — the ``(size, mtime_ns)`` change-detection pair
  the residency scanners use as a cheap "did this file change" key.

Leaf module: imports nothing from omlx so every patch can share it.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import BinaryIO, Union

import numpy as np

# Safetensors payloads are little-endian by spec; multi-byte dtypes pin
# "<" explicitly so the table is correct regardless of host order.
SAFETENSORS_NUMPY_DTYPES: dict[str, np.dtype] = {
    "BOOL": np.dtype("?"),
    "U8": np.dtype("u1"),
    "I8": np.dtype("i1"),
    "U16": np.dtype("<u2"),
    "I16": np.dtype("<i2"),
    "F16": np.dtype("<f2"),
    # Raw bits — reinterpret as mx.bfloat16 on the MLX side (a
    # shift->f32->astype roundtrip flushes subnormals via Metal FTZ).
    "BF16": np.dtype("<u2"),
    "U32": np.dtype("<u4"),
    "I32": np.dtype("<i4"),
    "F32": np.dtype("<f4"),
    "U64": np.dtype("<u8"),
    "I64": np.dtype("<i8"),
    "F64": np.dtype("<f8"),
    # No numpy fp8 — raw bytes; decode via mx.from_fp8.
    "F8_E4M3": np.dtype("u1"),
    "F8_E5M2": np.dtype("u1"),
}


def read_safetensors_header(source: str | Path | BinaryIO) -> dict:
    """Parse a safetensors JSON header (8-byte LE length + JSON dict).

    ``source`` is a filesystem path (opened read-only for the call) or an
    already-open binary file positioned at offset 0 — the latter is left
    positioned at the data start so the caller keeps reading tensor bytes
    off the same descriptor.
    """
    if hasattr(source, "read"):
        return _parse_header(source)
    with open(source, "rb") as f:
        return _parse_header(f)


def _parse_header(f: BinaryIO) -> dict:
    hsize = struct.unpack("<Q", f.read(8))[0]
    return json.loads(f.read(hsize))


def file_signature(path: str | Path) -> tuple[int, int]:
    """``(size, mtime_ns)`` change signature for *path*.

    Same semantic as residency's ``_index_sig_of``: a rewrite with an
    identical size still differs by ``st_mtime_ns``. Raises OSError when
    the file is gone — callers that tolerate disappearance guard with
    ``is_file()`` first.
    """
    st = os.stat(path)
    return (int(st.st_size), int(st.st_mtime_ns))
