#!/usr/bin/env python3
"""Prepare a DeepSeek-V4 oQ checkpoint for direct SSD expert streaming.

The mixed 2.4-bit checkpoint already stores routed expert weights in MLX's
runtime-native uint32-packed shape. Its affine scales and biases are BF16,
however, while the DeepSeek-V4 fast MoE path consumes FP16 metadata. Loading
those BF16 bytes directly into FP16 expert banks corrupts values; converting
them during every SSD publication adds a staging and Metal compute barrier.

This tool converts only routed ``switch_mlp`` scales and biases from BF16 to
FP16. Tensor names, shapes, byte sizes, data offsets, and packed expert weights
remain unchanged. Each shard is copied, converted, fsynced, and atomically
replaced, so interruption leaves the currently active shard untouched. Shards
already converted to F16 are skipped, making the operation resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import time
from pathlib import Path

import numpy as np


def _read_header(path: Path) -> tuple[int, dict]:
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
    return header_size, header


def _is_target(name: str, metadata: dict) -> bool:
    return (
        ".ffn.switch_mlp." in name
        and name.endswith((".scales", ".biases"))
        and metadata.get("dtype") == "BF16"
    )


def _header_bytes(header: dict, size: int) -> bytes:
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(encoded) > size:
        raise ValueError(
            f"Updated safetensors header grew from {size} to {len(encoded)} bytes"
        )
    return encoded + b" " * (size - len(encoded))


def _convert_region(
    mapped: np.memmap,
    start: int,
    stop: int,
    *,
    chunk_bytes: int,
) -> None:
    if (stop - start) % 2:
        raise ValueError("BF16 tensor payload does not contain whole uint16 values")
    chunk_elements = max(1, chunk_bytes // 2)
    source = mapped[start:stop].view(np.uint16)
    for offset in range(0, source.size, chunk_elements):
        target = source[offset : offset + chunk_elements]
        fp32 = (target.astype(np.uint32) << np.uint32(16)).view(np.float32)
        with np.errstate(over="ignore", invalid="ignore"):
            fp16 = fp32.astype(np.float16)
        target[:] = fp16.view(np.uint16)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _convert_shard(path: Path, *, chunk_bytes: int) -> dict:
    header_size, header = _read_header(path)
    targets = [
        (name, metadata)
        for name, metadata in header.items()
        if name != "__metadata__" and _is_target(name, metadata)
    ]
    if not targets:
        return {"shard": path.name, "converted": False, "tensors": 0, "bytes": 0}

    temporary = path.with_name(f".{path.name}.streaming-convert.tmp")
    if temporary.exists():
        temporary.unlink()
    started = time.perf_counter()
    shutil.copyfile(path, temporary)
    mapped = np.memmap(temporary, dtype=np.uint8, mode="r+")
    data_start = 8 + header_size
    converted_bytes = 0
    try:
        for _name, metadata in targets:
            start, stop = (int(value) for value in metadata["data_offsets"])
            _convert_region(
                mapped,
                data_start + start,
                data_start + stop,
                chunk_bytes=chunk_bytes,
            )
            metadata["dtype"] = "F16"
            converted_bytes += stop - start
        mapped.flush()
    finally:
        del mapped

    with temporary.open("r+b", buffering=0) as handle:
        handle.seek(8)
        handle.write(_header_bytes(header, header_size))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {
        "shard": path.name,
        "converted": True,
        "tensors": len(targets),
        "bytes": converted_bytes,
        "seconds": round(time.perf_counter() - started, 3),
        "sha256": _sha256(path),
    }


def _inventory(root: Path) -> list[dict]:
    inventory = []
    for path in sorted(root.glob("*.safetensors")):
        _, header = _read_header(path)
        targets = [
            metadata
            for name, metadata in header.items()
            if name != "__metadata__" and _is_target(name, metadata)
        ]
        inventory.append(
            {
                "shard": path.name,
                "tensors": len(targets),
                "bytes": sum(
                    int(item["data_offsets"][1]) - int(item["data_offsets"][0])
                    for item in targets
                ),
            }
        )
    return inventory


def _write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform the conversion; otherwise print a dry-run inventory",
    )
    parser.add_argument("--chunk-mib", type=int, default=64)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="conversion journal (default: MODEL/streaming-layout-conversion.json)",
    )
    args = parser.parse_args()
    if args.chunk_mib <= 0:
        parser.error("--chunk-mib must be positive")

    root = args.model.expanduser().resolve()
    if not (root / "config.json").is_file():
        parser.error(f"not a local model directory: {root}")
    config = json.loads((root / "config.json").read_text())
    if not str(config.get("model_type", "")).startswith("deepseek_v4"):
        parser.error("checkpoint is not DeepSeek-V4")
    inventory = _inventory(root)
    pending_tensors = sum(item["tensors"] for item in inventory)
    pending_bytes = sum(item["bytes"] for item in inventory)
    print(
        json.dumps(
            {
                "model": str(root),
                "pending_tensors": pending_tensors,
                "pending_bytes": pending_bytes,
                "pending_gib": round(pending_bytes / 1024**3, 3),
                "shards": inventory,
            },
            indent=2,
        ),
        flush=True,
    )
    if not args.apply:
        return

    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else root / "streaming-layout-conversion.json"
    )
    results = []
    manifest = {
        "format": "omlx.deepseek_v4_streaming_layout.v1",
        "model": str(root),
        "transforms": [
            {
                "match": "*.ffn.switch_mlp.*.(scales|biases)",
                "source_dtype": "BF16",
                "target_dtype": "F16",
                "payload": "numeric conversion",
            },
            {
                "match": "*.ffn.switch_mlp.*.weight",
                "source_dtype": "U32",
                "target_dtype": "U32",
                "payload": "unchanged; already runtime-native packed layout",
            },
        ],
        "converted_tensors": 0,
        "converted_bytes": 0,
        "shards": results,
        "complete": False,
    }
    for item in inventory:
        if not item["tensors"]:
            continue
        result = _convert_shard(
            root / item["shard"], chunk_bytes=args.chunk_mib * 1024**2
        )
        results.append(result)
        manifest["converted_tensors"] = sum(
            entry["tensors"] for entry in results
        )
        manifest["converted_bytes"] = sum(entry["bytes"] for entry in results)
        _write_manifest(manifest_path, manifest)

    remaining = _inventory(root)
    remaining_tensors = sum(item["tensors"] for item in remaining)
    manifest["complete"] = remaining_tensors == 0
    manifest["remaining_tensors"] = remaining_tensors
    _write_manifest(manifest_path, manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    if remaining_tensors:
        raise SystemExit(f"conversion incomplete: {remaining_tensors} tensors remain")


if __name__ == "__main__":
    main()
