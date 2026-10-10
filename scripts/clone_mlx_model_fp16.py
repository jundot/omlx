#!/usr/bin/env python3
"""Create an FP16 clone of an MLX quantized model without changing its weights.

Packed integer weight tensors are copied unchanged. Floating-point checkpoint
tensors are converted to FP16 one safetensors shard at a time, and the cloned
config advertises FP16. The source directory is always treated as read-only.

This is intended for the optional Qwen3.5/3.8 ANE+CPU prefill path. That path
can let BNNS consume the model's FP16 activations directly while the existing
q4 packed weights remain available to the GPU suffix.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx
from safetensors import safe_open

_FP16_MAX = 65504.0


def _clone_config(source: Path, destination: Path) -> None:
    config = json.loads(source.read_text())
    if isinstance(config.get("text_config"), dict):
        config["text_config"]["dtype"] = "float16"
    if "dtype" in config:
        config["dtype"] = "float16"
    destination.write_text(json.dumps(config, indent=2) + "\n")


def _conversion_issues(shard: Path, tensors: dict[str, mx.array]) -> list[str]:
    issues: list[str] = []
    for name, value in tensors.items():
        if not mx.issubdtype(value.dtype, mx.floating):
            continue
        finite = mx.isfinite(value)
        non_finite = int(mx.sum(~finite).item())
        if non_finite:
            issues.append(f"{shard.name}:{name}: {non_finite} NaN or infinite value(s)")
        if value.dtype == mx.bfloat16:
            finite_abs = mx.where(
                finite,
                mx.abs(value).astype(mx.float32),
                mx.array(0.0, dtype=mx.float32),
            )
            maximum = float(mx.max(finite_abs).item()) if value.size else 0.0
            if maximum > _FP16_MAX:
                issues.append(
                    f"{shard.name}:{name}: maximum absolute value {maximum:g} "
                    f"exceeds the FP16 limit {_FP16_MAX:g}"
                )
    return issues


def _validate_conversion(shards: list[Path]) -> None:
    issues: list[str] = []
    for index, shard in enumerate(shards, start=1):
        tensors = mx.load(str(shard))
        issues.extend(_conversion_issues(shard, tensors))
        del tensors
        mx.clear_cache()
        print(f"[{index}/{len(shards)}] validated {shard.name}", flush=True)
    if issues:
        report = "\n".join(f"- {issue}" for issue in issues)
        raise ValueError(
            "FP16 clone validation failed; no checkpoint files were written:\n" + report
        )


def clone_model(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination = destination.resolve()
    if source == destination:
        raise ValueError("The destination must differ from the source model")
    if not source.is_dir():
        raise ValueError(f"Source model directory does not exist: {source}")
    if destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        raise ValueError(f"Destination already exists and is not empty: {destination}")

    shards = sorted(source.glob("*.safetensors"))
    if not shards:
        raise ValueError(f"No safetensors shards found in {source}")
    _validate_conversion(shards)

    destination.mkdir(parents=True, exist_ok=True)

    for item in source.iterdir():
        if item.suffix == ".safetensors":
            continue
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        elif item.name == "config.json":
            _clone_config(item, target)
        else:
            shutil.copy2(item, target)

    for index, shard in enumerate(shards, start=1):
        target = destination / shard.name
        temporary = destination / f".{shard.name}.partial.safetensors"
        with safe_open(shard, framework="np") as handle:
            metadata = handle.metadata() or {}
        tensors = mx.load(str(shard))
        converted = {
            name: value.astype(mx.float16) if value.dtype == mx.bfloat16 else value
            for name, value in tensors.items()
        }
        mx.save_safetensors(str(temporary), converted, metadata=metadata)
        temporary.replace(target)
        del converted, tensors
        mx.clear_cache()
        print(f"[{index}/{len(shards)}] converted {shard.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    clone_model(args.source, args.destination)


if __name__ == "__main__":
    main()
