# SPDX-License-Identifier: Apache-2.0
"""FRIDA memory admission from safetensors headers, without weight allocation."""

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FridaMemoryEstimate:
    resident_bytes: int
    loading_bytes: int
    source_bytes: int


def estimate_frida_memory(
    path: str | Path, precision: str = "fp32"
) -> FridaMemoryEstimate:
    if precision not in ("fp32", "bf16"):
        raise ValueError("frida_precision must be fp32 or bf16")
    source, resident = 0, 0
    for filename in ("model.safetensors", "head.safetensors"):
        with (Path(path) / filename).open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            if length > 512 * 2**20:
                raise ValueError("Invalid safetensors header length")
            header = json.loads(stream.read(length))
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            start, end = tensor["data_offsets"]
            source += end - start
            # Upstream drops the tied embedding alias before conversion.
            if (
                filename == "model.safetensors"
                and name == "encoder.embed_tokens.weight"
                and "shared.weight" in header
            ):
                continue
            elements = math.prod(tensor["shape"])
            resident += elements * (
                4 if filename == "head.safetensors" or precision == "fp32" else 2
            )
    return FridaMemoryEstimate(
        math.ceil(resident * 1.05), math.ceil((resident + source) * 1.05), source
    )
