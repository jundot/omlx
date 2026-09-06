# SPDX-License-Identifier: Apache-2.0
"""Match local Uno adapters to compatible K2 bases."""

import json
from dataclasses import dataclass
from pathlib import Path

from .patches.k2_horizon.checkpoint import checkpoint_files

_BASES = {
    (1536, 28, 5120, 64256): ("IFM/K2-Horizon-0.9B", 131072),
    (4096, 36, 12288, 250624): ("IFM/K2-Horizon-7B", 262144),
}


def uno_base_id(path: str | Path) -> str | None:
    path = Path(path)
    try:
        adapter = path / "adapter_config.json"
        if adapter.is_file():
            config = json.loads(adapter.read_text())
            base = config.get("base_model_name_or_path")
            return (
                base
                if config.get("peft_type") == "LORA"
                and base in {value[0] for value in _BASES.values()}
                else None
            )
        config = json.loads((path / "config.json").read_text())
        if config.get("model_type") != "k2_horizon" or config.get("num_experts", 0):
            return None
        shape = tuple(
            config.get(key)
            for key in (
                "hidden_size",
                "num_hidden_layers",
                "intermediate_size",
                "vocab_size",
            )
        )
        return _BASES.get(shape, (None, 0))[0]
    except (OSError, ValueError, TypeError, AttributeError):
        return None


@dataclass(frozen=True)
class UnoBundle:
    base_path: Path
    adapter_path: Path
    base_model_id: str
    context_length: int
    config: dict
    block_size: int = 8


def resolve_uno_bundle(base_path: str | Path, adapter_path: str | Path) -> UnoBundle:
    base_path, adapter_path = Path(base_path), Path(adapter_path)
    base_id = uno_base_id(base_path)
    if base_id is None or uno_base_id(adapter_path) != base_id:
        raise ValueError("Select a Uno adapter that matches this K2 base.")
    if not (adapter_path / "adapter_model.safetensors").is_file():
        raise ValueError("The selected Uno adapter has no adapter_model.safetensors.")
    config = json.loads((base_path / "config.json").read_text())
    if config.get("attention_gate_func") or config.get("model_file"):
        raise ValueError("Uno requires a native, ungated K2 base.")
    checkpoint_files(base_path)
    limit = next(limit for name, limit in _BASES.values() if name == base_id)
    return UnoBundle(
        base_path,
        adapter_path,
        base_id,
        min(config["max_position_embeddings"], limit),
        config,
    )
