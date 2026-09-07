# SPDX-License-Identifier: Apache-2.0
"""Qwen4-Exp PLE storage estimates shared by runtime and settings UIs."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_MODEL_OVERHEAD_FACTOR = 1.05
_NGRAM_EMBEDDING_MARKER = ".ngram_embedding."


@dataclass(frozen=True)
class Qwen4ExpResidencyEstimate:
    """Estimated resident bytes with and without the PLE embedding table."""

    supported: bool
    checkpoint_bytes: int
    ple_bytes: int
    resident_bytes: int
    mmap_bytes: int

    def force_ssd_offload(self, memory_ceiling: int) -> bool:
        """Return True only when mmap turns an otherwise impossible load viable."""

        return (
            self.supported
            and memory_ceiling > 0
            and self.resident_bytes > memory_ceiling
            and self.mmap_bytes <= memory_ceiling
        )


def _safetensors_header(path: Path) -> dict:
    with path.open("rb") as file:
        raw_size = file.read(8)
        if len(raw_size) != 8:
            return {}
        header_size = struct.unpack("<Q", raw_size)[0]
        return json.loads(file.read(header_size))


def _ple_tensor_bytes(ple_path: Path) -> tuple[int, bool]:
    index_path = ple_path / "model.safetensors.index.json"
    if not index_path.is_file():
        return 0, False
    try:
        weight_map = json.loads(index_path.read_text()).get("weight_map") or {}
    except (OSError, ValueError):
        return 0, False

    selected: dict[str, list[str]] = {}
    for key, filename in weight_map.items():
        if _NGRAM_EMBEDDING_MARKER in key:
            selected.setdefault(filename, []).append(key)
    if not selected:
        return 0, False

    total = 0
    try:
        for filename, keys in selected.items():
            header = _safetensors_header(ple_path / filename)
            for key in keys:
                start, end = header[key]["data_offsets"]
                total += int(end) - int(start)
    except (KeyError, OSError, TypeError, ValueError):
        return 0, False
    return total, total > 0


@lru_cache(maxsize=128)
def _cached_residency_estimate(
    ple_path_string: str,
    checkpoint_signature: tuple[tuple[str, int, int], ...],
    _index_signature: tuple[int, int] | None,
) -> Qwen4ExpResidencyEstimate:
    ple_path = Path(ple_path_string)
    checkpoint_bytes = sum(size for _, size, _ in checkpoint_signature)
    ple_bytes, supported = _ple_tensor_bytes(ple_path)
    resident_bytes = int(checkpoint_bytes * _MODEL_OVERHEAD_FACTOR)
    mmap_bytes = int(max(0, checkpoint_bytes - ple_bytes) * _MODEL_OVERHEAD_FACTOR)
    return Qwen4ExpResidencyEstimate(
        supported=supported,
        checkpoint_bytes=checkpoint_bytes,
        ple_bytes=ple_bytes,
        resident_bytes=resident_bytes,
        mmap_bytes=mmap_bytes,
    )


def qwen4_exp_residency_estimate(
    model_path: str | Path,
) -> Qwen4ExpResidencyEstimate:
    """Inspect Qwen4 checkpoint headers without materializing tensor data."""

    ple_path = Path(model_path).expanduser().resolve()
    checkpoint_files = {path.resolve() for path in ple_path.glob("*.safetensors")}
    signature = tuple(
        sorted(
            (str(path), stat.st_size, stat.st_mtime_ns)
            for path in checkpoint_files
            for stat in (path.stat(),)
        )
    )
    index_path = ple_path / "model.safetensors.index.json"
    index_stat = index_path.stat() if index_path.is_file() else None
    index_signature = (
        (index_stat.st_size, index_stat.st_mtime_ns)
        if index_stat is not None
        else None
    )
    return _cached_residency_estimate(str(ple_path), signature, index_signature)
