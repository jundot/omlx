"""Direct loader support for prepared low-bit DeepSpec checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def is_prepared_checkpoint(path: str | Path) -> bool:
    manifest = Path(path) / "dspark_manifest.json"
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        try:
            data = json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        quant = data.get("quantization") or data.get("quantization_config") or {}
        return (
            int(quant.get("bits", 0) or 0) > 0
            and int(quant.get("group_size", 0) or 0) > 0
        )
    quant = data.get("quantization") or {}
    return quant.get("status") == "ready" and int(quant.get("bits", 0)) > 0


def checkpoint_quantization(path: str | Path) -> tuple[int, int]:
    """Return the immutable quantization encoded by a prepared checkpoint."""
    manifest_path = Path(path, "dspark_manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        quant = manifest.get("quantization") or {}
        if quant.get("status") != "ready":
            raise ValueError(f"prepared dSpark checkpoint is not ready: {path}")
    else:
        config = json.loads(Path(path, "config.json").read_text(encoding="utf-8"))
        quant = config.get("quantization") or config.get("quantization_config") or {}
    return int(quant.get("bits", 0)), int(quant.get("group_size", 0))


def load_prequantized_drafter(
    repo_or_path: str | Path,
    *,
    strict: bool = True,
) -> tuple[Any, Any]:
    """Load packed tensors using checkpoint metadata, never runtime settings."""
    import glob
    import os

    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from .native_config import DSparkConfig
    from .native_model import DSparkDrafter

    path = str(Path(repo_or_path).expanduser().resolve())
    manifest_path = Path(path, "dspark_manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        quant = manifest.get("quantization") or {}
    else:
        config_data = json.loads(Path(path, "config.json").read_text(encoding="utf-8"))
        quant = (
            config_data.get("quantization")
            or config_data.get("quantization_config")
            or {}
        )
    prepared_bits = int(quant.get("bits", 0))
    prepared_group = int(quant.get("group_size", 0))
    if manifest_path.exists() and quant.get("status") != "ready":
        raise ValueError(f"prepared dSpark checkpoint is not ready: {repo_or_path}")
    if prepared_bits <= 0 or prepared_group <= 0:
        raise ValueError("prepared dSpark checkpoint has invalid quantization metadata")

    config = DSparkConfig.from_json(os.path.join(path, "config.json"))
    drafter = DSparkDrafter(config)
    nn.quantize(
        drafter,
        group_size=prepared_group,
        bits=prepared_bits,
        class_predicate=lambda module_path, module: (
            isinstance(module, (nn.Linear, nn.Embedding))
            and not module_path.startswith("log_snr_embed")
        ),
    )
    weights: dict[str, mx.array] = {}
    for safetensor in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        weights.update(mx.load(safetensor))
    model_keys = {key for key, _ in tree_flatten(drafter.parameters())}
    checkpoint_keys = set(weights)
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    if strict and (missing or unexpected):
        raise ValueError(
            "prepared dSpark tensor names do not match quantized model: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    drafter.load_weights(list(weights.items()), strict=not (missing or unexpected))
    # RMSNorm offsets are materialized by prepare_checkpoint before packing.
    mx.eval(drafter.parameters())
    return drafter, config
