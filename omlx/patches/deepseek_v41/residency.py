# SPDX-License-Identifier: Apache-2.0
"""Original and converted V4.1 residency estimates from tensor headers."""

import json
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class EngramResidencyEstimate:
    supported: bool
    resident_bytes: int
    mmap_bytes: int
    engram_bytes: int

    def force_ssd_offload(self, memory_ceiling):
        return (
            self.supported
            and memory_ceiling > 0
            and self.resident_bytes > memory_ceiling >= self.mmap_bytes
        )


@lru_cache(maxsize=128)
def header_with_offset(filename, size, mtime_ns):
    """A shard's safetensors header and the file offset of its tensor data."""
    with open(filename, "rb") as file:
        raw = file.read(8)
        if len(raw) != 8:
            raise ValueError("Truncated safetensors header")
        length = struct.unpack("<Q", raw)[0]
        if length > size - 8:
            raise ValueError("Invalid safetensors header length")
        return json.loads(file.read(length)), 8 + length


def _header(filename, size, mtime_ns):
    return header_with_offset(filename, size, mtime_ns)[0]


def checkpoint_signature(model_path):
    """Sizes and mtimes of every file a residency estimate depends on."""
    path = Path(model_path).expanduser().resolve()
    files = [path / "config.json", path / "model.safetensors.index.json"]
    files.extend(path.glob("*.safetensors"))
    files.extend((path / "engram").glob("*.safetensors"))
    return tuple(
        (str(f), st.st_size, st.st_mtime_ns)
        for f in sorted(files)
        for st in (f.stat(),)
    )


def deepseek_v41_residency_estimate(model_path):
    path = Path(model_path).expanduser().resolve()
    return _estimate(str(path), checkpoint_signature(path))


@lru_cache(maxsize=128)
def _estimate(model_path, signature):
    path = Path(model_path)
    config = json.loads((path / "config.json").read_text())
    spec = config.get("omlx_deepseek_v41")
    if spec is not None and spec.get("version") != 1:
        return EngramResidencyEstimate(False, 0, 0, 0)
    mapping = json.loads((path / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    headers = {}
    stats = {name: (size, mtime) for name, size, mtime in signature}

    def entry(filename, key):
        if filename not in headers:
            full = str(path / filename)
            headers[filename] = _header(full, *stats[full])
        return headers[filename][key]

    def tensor_bytes(filename, key):
        start, end = entry(filename, key)["data_offsets"]
        if start < 0 or end < start:
            raise ValueError("Invalid Engram tensor offsets")
        return end - start

    base, engram = 0, 0
    if spec is not None:
        tables = spec.get("engram_tables", {})
        if not tables:
            return EngramResidencyEstimate(False, 0, 0, 0)
        engram_keys = {
            table[key]
            for table in tables.values()
            for key in ("weight_key", "scale_key", "bias_key")
            if table.get(key)
        }
        base = sum(
            tensor_bytes(filename, key)
            for key, filename in mapping.items()
            if key not in engram_keys
        )
        for table in tables.values():
            engram += tensor_bytes(table["weight_file"], table["weight_key"])
            if table.get("scale_key"):
                engram += tensor_bytes(
                    table.get("scale_file") or table["weight_file"], table["scale_key"]
                )
            if table.get("bias_key"):
                engram += tensor_bytes(
                    table.get("scale_file") or table["weight_file"], table["bias_key"]
                )
    else:
        if config.get("model_type") != "deepseek_v41":
            return EngramResidencyEstimate(False, 0, 0, 0)
        for key, filename in mapping.items():
            if key.startswith("mtp."):
                continue
            size = tensor_bytes(filename, key)
            if ".engram.embed." in key:
                engram += size
                continue
            weight_key = (
                key.removesuffix(".scale") + ".weight"
                if key.endswith(".scale")
                else key
            )
            dense = weight_key.endswith("wo_a.weight") or weight_key in (
                "head.weight",
                "embed.weight",
            )
            weight = entry(mapping[weight_key], weight_key)
            if weight["dtype"].startswith("F8_E4M3"):
                if key.endswith(".scale"):
                    # MLX MXFP8 uses one scale per row and 32 channels.
                    size = (
                        0
                        if dense
                        else weight["shape"][0] * entry(filename, key)["shape"][1]
                    )
                elif dense:
                    size *= 2
            base += size
    return EngramResidencyEstimate(
        engram > 0,
        int((base + engram) * 1.05),
        int(base * 1.05) + 32 * 1024 * 1024,
        engram,
    )
