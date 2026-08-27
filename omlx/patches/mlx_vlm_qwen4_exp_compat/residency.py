# SPDX-License-Identifier: Apache-2.0
"""Qwen4-Exp PLE storage estimates shared by runtime and settings UIs."""

from __future__ import annotations

import json
import os
import struct
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_MODEL_OVERHEAD_FACTOR = 1.05
_NGRAM_EMBEDDING_MARKER = ".ngram_embedding."
_PREFAULT_CHUNK_BYTES = 16 * 1024 * 1024


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


def _resolve_ple_path(model_path: Path) -> Path:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        return model_path
    try:
        config = json.loads(config_path.read_text())
    except (OSError, ValueError):
        return model_path
    artifact = config.get("qwen4_exp_artifact") or {}
    relative_ple = artifact.get("ple_artifact")
    if relative_ple is None:
        return model_path
    relative_ple = Path(relative_ple)
    if relative_ple.is_absolute():
        return model_path
    candidate = (model_path / relative_ple).resolve()
    artifact_root = model_path.parent.resolve()
    if candidate != artifact_root and artifact_root not in candidate.parents:
        return model_path
    return candidate


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

    compute_path = Path(model_path).expanduser().resolve()
    ple_path = _resolve_ple_path(compute_path)
    roots = {compute_path, ple_path}
    checkpoint_files = {
        path.resolve() for root in roots for path in root.glob("*.safetensors")
    }
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
        (index_stat.st_size, index_stat.st_mtime_ns) if index_stat is not None else None
    )
    return _cached_residency_estimate(str(ple_path), signature, index_signature)


def prefault_qwen4_exp_checkpoint(
    model_path: str | Path,
    *,
    chunk_bytes: int = _PREFAULT_CHUNK_BYTES,
    include_ple: bool = True,
) -> tuple[int, float]:
    """Sequentially fault resident Qwen4 checkpoint pages into the file cache.

    MLX keeps safetensor storage file-backed. Evaluating the arrays makes the
    graph concrete but does not touch every weight page, so a 60+ GB MoE can
    otherwise spend minutes demand-faulting random expert and PLE pages during
    its first decode. A sequential pass is dramatically cheaper on NVMe and
    makes that cost part of model loading instead of the first request.

    Set ``OMLX_QWEN4_PREFAULT=0`` to disable the pass. The returned tuple is
    ``(bytes_read, elapsed_seconds)``.
    """

    requested = os.environ.get("OMLX_QWEN4_PREFAULT", "auto").strip().lower()
    if requested in {"0", "false", "off", "no"}:
        return 0, 0.0
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")

    compute_path = Path(model_path).expanduser().resolve()
    ple_path = _resolve_ple_path(compute_path)
    checkpoint_files = sorted(
        {
            path.resolve()
            for root in {compute_path, ple_path}
            for path in root.glob("*.safetensors")
        }
    )
    if not checkpoint_files:
        return 0, 0.0

    buffer = bytearray(chunk_bytes)
    view = memoryview(buffer)
    total = 0
    started = time.perf_counter()
    for path in checkpoint_files:
        with path.open("rb", buffering=0) as file:
            if include_ple:
                ranges = [(0, path.stat().st_size)]
            else:
                raw_size = file.read(8)
                if len(raw_size) != 8:
                    continue
                header_size = struct.unpack("<Q", raw_size)[0]
                header = json.loads(file.read(header_size))
                data_start = 8 + header_size
                ranges = [(0, data_start)]
                ranges.extend(
                    (data_start + int(start), data_start + int(end))
                    for key, entry in header.items()
                    if key != "__metadata__" and _NGRAM_EMBEDDING_MARKER not in key
                    for start, end in (entry["data_offsets"],)
                )
                ranges.sort()
                merged: list[tuple[int, int]] = []
                for start, end in ranges:
                    if merged and start <= merged[-1][1]:
                        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                    else:
                        merged.append((start, end))
                ranges = merged

            for start, end in ranges:
                file.seek(start)
                remaining = end - start
                while remaining > 0:
                    count = file.readinto(view[: min(remaining, chunk_bytes)])
                    if not count:
                        break
                    # Keep the read observable without retaining a second copy
                    # of the checkpoint in Python-managed memory.
                    total += count
                    remaining -= count
                    _ = view[count - 1]
    return total, time.perf_counter() - started
