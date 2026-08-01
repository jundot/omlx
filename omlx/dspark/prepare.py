"""Create immutable, directly loadable low-bit dSpark checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from .compat import model_fingerprint, probe_drafter


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _nbytes(array: Any) -> int:
    value = getattr(array, "nbytes", None)
    return int(value() if callable(value) else value or 0)


def _module_inventory(config_path: Path) -> tuple[set[str], set[str]]:
    import mlx.nn as nn

    from .native_config import DSparkConfig
    from .native_model import DSparkDrafter

    config = DSparkConfig.from_json(str(config_path))
    model = DSparkDrafter(config)
    quantized: set[str] = set()
    norms: set[str] = set()
    for path, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Embedding)) and not path.startswith(
            "log_snr_embed"
        ):
            quantized.add(path)
        if isinstance(module, nn.RMSNorm):
            norms.add(path)
    del model
    return quantized, norms


def prepare_checkpoint(
    source: str | Path,
    destination: str | Path,
    *,
    target: str | Path,
    bits: int = 2,
    group_size: int = 64,
    source_revision: str | None = None,
) -> dict[str, Any]:
    """Quantize a supported dSpark checkpoint tensor-by-tensor and publish atomically."""
    if bits not in {2, 4, 8}:
        raise ValueError("bits must be 2, 4, or 8")
    if group_size not in {32, 64, 128}:
        raise ValueError("group_size must be 32, 64, or 128")
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    probe = probe_drafter(source, "auto")
    source_files = sorted(source.glob("*.safetensors"))
    if not source_files:
        raise ValueError(f"no source tensors in {source}")
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")

    tmp = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    tmp.mkdir(parents=True, exist_ok=False)
    try:
        shutil.copy2(source / "README.md", tmp / "README.md") if (
            source / "README.md"
        ).exists() else None
        config = json.loads((source / "config.json").read_text(encoding="utf-8"))
        config["quantization"] = {
            "group_size": group_size,
            "bits": bits,
            "mode": "affine",
        }
        (tmp / "config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        quantized_paths, norm_paths = _module_inventory(source / "config.json")

        import mlx.core as mx

        shard_index = 0
        output_hashes: dict[str, str] = {}
        output_tensor_count = 0
        for source_file in source_files:
            tensors = mx.load(str(source_file))
            for key in sorted(tensors):
                value = tensors[key]
                module_path, _, leaf = key.rpartition(".")
                output: dict[str, Any]
                if leaf == "weight" and module_path in quantized_paths:
                    packed, scales, biases = mx.quantize(
                        value, group_size=group_size, bits=bits
                    )
                    output = {
                        key: packed,
                        f"{module_path}.scales": scales,
                        f"{module_path}.biases": biases,
                    }
                else:
                    if (
                        leaf == "weight"
                        and module_path in norm_paths
                        and bool(config.get("offset_rms_norm"))
                    ):
                        value = value + 1.0
                    output = {key: value}
                mx.eval(output)
                shard_index += 1
                shard = tmp / f"model-{shard_index:05d}.safetensors"
                mx.save_safetensors(str(shard), output)
                output_hashes[shard.name] = _sha256(shard)
                output_tensor_count += len(output)
                del output, value
                mx.clear_cache()
            del tensors

        source_hashes = {path.name: _sha256(path) for path in source_files}
        manifest = {
            "schema_version": 1,
            "format": probe.format,
            "source_path": str(source),
            "source_revision": source_revision,
            "source_sha256": source_hashes,
            "target_fingerprint": model_fingerprint(target),
            "target_num_hidden_layers": json.loads(
                (Path(target) / "config.json").read_text(encoding="utf-8")
            )
            .get("text_config", {})
            .get("num_hidden_layers"),
            "owns_embedding": probe.owns_embedding,
            "owns_output_head": probe.owns_output_head,
            "source_tensor_count": probe.tensor_count,
            "tensor_count": output_tensor_count,
            "weight_sha256": output_hashes,
            "quantization": {
                "status": "ready",
                "bits": bits,
                "group_size": group_size,
                "mode": "affine",
                "runtime_conversion": False,
                "offset_rms_norm_materialized": bool(config.get("offset_rms_norm")),
            },
            "prepared_at": int(time.time()),
        }
        (tmp / "dspark_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, destination)
        return manifest
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a low-bit dSpark checkpoint")
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--target", required=True)
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--source-revision")
    args = parser.parse_args()
    manifest = prepare_checkpoint(
        args.source,
        args.destination,
        target=args.target,
        bits=args.bits,
        group_size=args.group_size,
        source_revision=args.source_revision,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
