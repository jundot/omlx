# SPDX-License-Identifier: Apache-2.0
"""Bounded little-endian safetensors I/O for Hy4 Flash (no model or mmap).

The header reader is also usable by the runtime. It returns absolute byte
ranges in payload order; it never materializes a tensor or imports MLX.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import struct
import sys
from dataclasses import dataclass

MAX_HEADER = 16 << 20
ITEM = {"BF16": 2, "F32": 4, "I32": 4, "U32": 4}
BITS, BLOCK, GROUP = 8, 256, 128
# Routed-bank widths the converter writes and the runtime loads. 8 is the
# reference export; lower widths trade quality for memory (4 fits 128 GB).
SUPPORTED_BITS = (4, 5, 6, 8)
# Optional plain MLX affine quantization of the dense BF16 trunk linears
# (``--trunk-bits``). Decode at batch size 1 is bandwidth-bound, and these
# weights are read on every token. kv_b_proj (raw weight in the absorbed MLA
# path), the router, the sparse indexer, Engram, iHC and embeddings stay exact.
TRUNK_GROUP = 64
SUPPORTED_TRUNK_BITS = (8,)
TRUNK_LINEAR = re.compile(
    r"(?:model\.layers\.\d+\.(?:self_attn\.(?:q_a_proj|q_b_proj|kv_a_proj_with_mqa"
    r"|linear_gate|o_proj)|mlp\.(?:(?:shared_experts\.)?(?:gate|up|down)_proj"
    r"|routed_expert_(?:down|up)_proj))|lm_head)\.weight"
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def unique(pairs):
    result = {}
    for k, v in pairs:
        require(k not in result, f"duplicate JSON key: {k}")
        result[k] = v
    return result


@dataclass(frozen=True)
class Spec:
    dtype: str
    shape: tuple[int, ...]

    def __post_init__(self):
        require(
            isinstance(self.dtype, str)
            and self.dtype in ITEM
            and len(self.shape) <= 32
            and all(type(x) is int and 0 <= x < 1 << 63 for x in self.shape),
            f"unsupported tensor dtype/shape: {self}",
        )

    @property
    def nbytes(self):
        return math.prod(self.shape) * ITEM[self.dtype]


@dataclass(frozen=True)
class View:
    start: int  # Absolute file offset, including length prefix and padded header.
    spec: Spec


def read_header(f, *, detached=False) -> dict[str, View]:
    """Check the header AND file length. A detached file must end after the header."""
    st = os.fstat(f.fileno())
    require(stat.S_ISREG(st.st_mode), "expected regular safetensors file")
    f.seek(0)
    prefix = f.read(8)
    require(len(prefix) == 8, "truncated safetensors prefix")
    length = struct.unpack("<Q", prefix)[0]
    require(2 <= length <= MAX_HEADER, f"invalid safetensors header length: {length}")
    raw = f.read(length)
    require(
        len(raw) == length and raw.startswith(b"{"),
        "truncated/invalid safetensors header",
    )
    entries = json.loads(raw, object_pairs_hook=unique)
    meta = entries.pop("__metadata__", {})
    require(
        isinstance(meta, dict) and all(isinstance(v, str) for v in meta.values()),
        "bad metadata",
    )
    base = 8 + length
    views = {}
    for k, e in entries.items():
        require(
            k
            and isinstance(e, dict)
            and isinstance(e.get("shape"), list)
            and isinstance(e.get("dtype"), str)
            and isinstance(e.get("data_offsets"), list)
            and len(e["data_offsets"]) == 2,
            f"invalid tensor: {k}",
        )
        spec = Spec(e["dtype"], tuple(e["shape"]))
        a, b = e["data_offsets"]
        require(
            type(a) is int and type(b) is int and a >= 0 and b - a == spec.nbytes,
            f"invalid extent: {k}",
        )
        views[k] = View(base + a, spec)
    ordered = dict(
        sorted(views.items(), key=lambda kv: (kv[1].start, kv[1].spec.nbytes))
    )
    end = base
    for k, v in ordered.items():
        require(v.start == end, f"hole/overlap before {k}")
        end += v.spec.nbytes
    require(
        st.st_size == (base if detached else end),
        f"wrong safetensors file length: {f.name}",
    )
    return ordered


def chunks(f, length, limit):
    """Exactly length bytes starting at the current file position."""
    require(length >= 0 and limit > 0, "invalid chunk extent")
    while length:
        n = min(length, limit)
        data = f.read(n)
        require(len(data) == n, f"short read: {f.name}")
        yield data
        length -= n


class Writer:
    """Append bounded chunks into each tensor's final non-overlapping slot."""

    def __init__(self, f, specs):
        self.f = f
        entries = {"__metadata__": {"format": "mlx"}}
        offset, self.views = 0, {}
        # Match safetensors' element alignment and deterministic header order.
        for k, spec in sorted(
            specs.items(), key=lambda kv: (-ITEM[kv[1].dtype], kv[0])
        ):
            require(k and k != "__metadata__", "reserved output key")
            entries[k] = {
                "dtype": spec.dtype,
                "shape": spec.shape,
                "data_offsets": [offset, offset + spec.nbytes],
            }
            self.views[k] = View(offset, spec)
            offset += spec.nbytes
        raw = json.dumps(entries, separators=(",", ":")).encode()
        raw += b" " * (-len(raw) % 8)
        require(len(raw) <= MAX_HEADER and f.tell() == 0, "invalid output header")
        header = struct.pack("<Q", len(raw)) + raw
        require(f.write(header) == len(header), "short header write")
        self.views = {
            k: View(len(header) + v.start, v.spec) for k, v in self.views.items()
        }
        self.cursor = {k: v.start for k, v in self.views.items()}
        self.hashes = {k: hashlib.sha256() for k in specs}

    def append(self, key, data):
        buf = memoryview(data).cast("B")
        pos, v = self.cursor[key], self.views[key]
        require(pos + len(buf) <= v.start + v.spec.nbytes, f"write overflow: {key}")
        self.f.seek(pos)
        require(self.f.write(buf) == len(buf), f"short write: {key}")
        self.hashes[key].update(buf)
        self.cursor[key] += len(buf)

    def finish(self, budget):
        """Read back every output extent; return verified whole-shard integrity record."""
        f = self.f
        require(
            all(
                self.cursor[k] == v.start + v.spec.nbytes for k, v in self.views.items()
            ),
            "unfinished shard",
        )
        f.flush()
        os.fsync(f.fileno())
        require(read_header(f) == self.views, "output header differs from plan")
        header_size = f.tell()
        f.seek(0)
        whole = hashlib.sha256()
        for data in chunks(f, header_size, budget):
            whole.update(data)
        for k, v in self.views.items():
            tensor = hashlib.sha256()
            for data in chunks(f, v.spec.nbytes, budget):
                whole.update(data)
                tensor.update(data)
            require(
                tensor.digest() == self.hashes[k].digest(),
                f"output readback mismatch: {k}",
            )
        return {"size": f.tell(), "sha256": whole.hexdigest()}


def bank_specs(key, spec, bits=BITS):
    require(
        spec.dtype == "BF16"
        and len(spec.shape) == 3
        and all(spec.shape)
        and spec.shape[-1] % BLOCK == 0,
        f"expected BF16 [expert,out,in] bank with contiguous {BLOCK}-wide WHT blocks: {key}",
    )
    e, n, k = spec.shape
    return {
        key + ".weight": Spec("U32", (e, n, k * bits // 32)),
        key + ".oqh_scales": Spec("BF16", (e, n, k // GROUP)),
        key + ".oqh_biases": Spec("BF16", (e, n, k // GROUP)),
        key + ".oqh_signs": Spec("BF16", (k // BLOCK, BLOCK)),
    }


def quantize_rows(raw, signs, bits=BITS):
    """RTN in the signed WHT basis. No calibration samples, Gram or refinement.

    Read complete BF16 rows, multiply in F32 by the producer's signs, apply
    the normalized 256-point WHT on the input axis, then call MLX's F32
    group-128 affine quantizer at ``bits`` (default 8). Codes are packed low byte first; scale
    and bias are rounded to BF16 **after** picking codes. No transpose.
    """
    import mlx.core as mx
    import numpy as np

    require(sys.byteorder == "little", "conversion requires little-endian MLX")
    k = signs.size
    w = mx.array(np.frombuffer(raw, dtype="<u2")).view(mx.bfloat16).reshape(-1, k)
    blocks = (w.astype(mx.float32) * signs).reshape(-1, BLOCK)
    rotated = mx.hadamard_transform(blocks).reshape(-1, k)
    packed, s, b = mx.quantize(rotated, group_size=GROUP, bits=bits, mode="affine")
    s, b = s.astype(mx.bfloat16).view(mx.uint16), b.astype(mx.bfloat16).view(mx.uint16)
    mx.eval(packed, s, b)
    require(packed.dtype == mx.uint32, "unexpected packed weight dtype")
    return packed, s, b


def write_bank(key, spec, data_chunks, writer, bits=BITS):
    import mlx.core as mx

    from ...oqh import format as fmt

    # The fused gate/up projection is one producer consumer, not two sign
    # fields. Down consumes a different activation; it has its own signs.
    signs = mx.array(fmt.make_signs(key, spec.shape[-1])).astype(mx.float32)
    stored = signs.astype(mx.bfloat16).view(mx.uint16)
    mx.eval(stored)
    writer.append(key + ".oqh_signs", stored)
    for data in data_chunks:
        p, s, b = quantize_rows(data, signs, bits)
        writer.append(key + ".weight", p)
        writer.append(key + ".oqh_scales", s)
        writer.append(key + ".oqh_biases", b)


def trunk_specs(key, spec, bits):
    require(
        spec.dtype == "BF16"
        and len(spec.shape) == 2
        and all(spec.shape)
        and spec.shape[-1] % TRUNK_GROUP == 0,
        f"expected BF16 [out,in] trunk linear with {TRUNK_GROUP}-wide groups: {key}",
    )
    n, k = spec.shape
    base = key.removesuffix(".weight")
    return {
        key: Spec("U32", (n, k * bits // 32)),
        base + ".scales": Spec("BF16", (n, k // TRUNK_GROUP)),
        base + ".biases": Spec("BF16", (n, k // TRUNK_GROUP)),
    }


def write_trunk(key, spec, data_chunks, writer, bits):
    """Row-chunked ``mx.quantize`` of a BF16 linear, identical to nn.QuantizedLinear."""
    import mlx.core as mx
    import numpy as np

    require(sys.byteorder == "little", "conversion requires little-endian MLX")
    base, k = key.removesuffix(".weight"), spec.shape[-1]
    for data in data_chunks:
        w = mx.array(np.frombuffer(data, dtype="<u2")).view(mx.bfloat16).reshape(-1, k)
        p, s, b = mx.quantize(w, group_size=TRUNK_GROUP, bits=bits, mode="affine")
        s = s.astype(mx.bfloat16).view(mx.uint16)
        b = b.astype(mx.bfloat16).view(mx.uint16)
        mx.eval(p, s, b)
        require(p.dtype == mx.uint32, "unexpected packed weight dtype")
        writer.append(key, p)
        writer.append(base + ".scales", s)
        writer.append(base + ".biases", b)
