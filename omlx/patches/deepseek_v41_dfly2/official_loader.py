"""Strict metadata and safetensors loader for the official DFly2 checkpoint.

The loader is deliberately independent from oMLX. It validates the HF index
before reading a shard, and allows selected-prefix loading so a local staging
forward never silently substitutes random tensors for official weights.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class LoadPlan:
    shards: tuple[str, ...]
    tensor_names: tuple[str, ...]
    missing_prefixes: tuple[str, ...]


@dataclass(frozen=True)
class LoadedOfficialWeights:
    """Selected safetensor arrays exactly as supplied by the official shard."""

    tensors: dict[str, object]
    shards: tuple[Path, ...]


def _shard_sort_key(name: str) -> tuple[int, str]:
    try:
        return (int(name.split("-")[1]), name)
    except (IndexError, ValueError):
        return (0, name)


def required_prefixes(*, layer_ids: tuple[int, ...], include_fusion: bool) -> tuple[str, ...]:
    """Prefixes for the checkpoint's own layer numbering.

    The official DFly2 checkpoint numbers its draft layers from zero, while
    ``dfly_config.target_kv_layers`` names the target layers they shadow. Passing
    target ids here (28/32/36) asks for prefixes the checkpoint never has; the
    caller must translate with ``target_layer_map`` first.
    """
    prefixes = tuple(f"model.layers.{layer}." for layer in layer_ids)
    if include_fusion:
        prefixes += (("model.mtp_fusion.", "model.hidden_correction.", "model.final_norm."))
    return prefixes


def target_layer_map(config: dict) -> dict[int, int]:
    """Draft layer index in the checkpoint -> target layer it shadows."""
    dfly = config.get("dfly_config") or {}
    targets = dfly.get("target_kv_layers") or []
    return {index: int(target) for index, target in enumerate(targets)}


def draft_layer_ids(config: dict) -> tuple[int, ...]:
    return tuple(sorted(target_layer_map(config)))


@dataclass(frozen=True)
class OfficialDFly2Checkpoint:
    config: dict
    weight_map: dict[str, str]

    @classmethod
    def from_metadata(cls, config_path: Path, index_path: Path) -> "OfficialDFly2Checkpoint":
        config = json.loads(config_path.read_text())
        index = json.loads(index_path.read_text())
        if config.get("model_type") != "deepseek_v41" or not isinstance(config.get("dfly_config"), dict):
            raise ValueError("expected an official DeepSeek-V4.1 DFly2 config")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("safetensors index has no weight_map")
        return cls(config=config, weight_map=weight_map)

    def plan_for_tensor_names(self, tensor_names: Iterable[str]) -> LoadPlan:
        names = tuple(sorted(set(tensor_names)))
        missing = tuple(name for name in names if name not in self.weight_map)
        if missing:
            raise KeyError(f"required official DFly2 tensor not present: {', '.join(missing)}")
        shards = tuple(sorted({self.weight_map[name] for name in names}, key=_shard_sort_key))
        return LoadPlan(shards=shards, tensor_names=names, missing_prefixes=())

    def plan_for_prefixes(self, prefixes: Iterable[str]) -> LoadPlan:
        names: list[str] = []
        missing: list[str] = []
        for prefix in prefixes:
            matched = [name for name in self.weight_map if name.startswith(prefix)]
            if not matched:
                missing.append(prefix)
            names.extend(matched)
        if missing:
            raise KeyError(f"required official DFly2 prefix not present: {', '.join(missing)}")
        names = sorted(set(names))
        shards = tuple(sorted({self.weight_map[name] for name in names}, key=_shard_sort_key))
        return LoadPlan(shards=shards, tensor_names=tuple(names), missing_prefixes=tuple(missing))

    def load_available_shards(self, weights_dir: Path, plan: LoadPlan) -> LoadedOfficialWeights:
        paths = tuple(weights_dir / shard for shard in plan.shards)
        absent = [str(path) for path in paths if not path.is_file()]
        if absent:
            raise FileNotFoundError(f"missing official shard(s): {', '.join(absent)}")
        # Import here: metadata-only validation works even if safetensors is absent.
        from safetensors import safe_open

        # The official checkpoint is bfloat16, which numpy cannot represent and
        # which safetensors' mlx backend still refuses (it converts through
        # numpy). MLX reads bfloat16 natively, so the mlx path loads a whole
        # shard at a time; numpy remains the fallback for metadata-only use.
        try:
            import mlx.core as mx
        except ImportError:  # pragma: no cover - numpy fallback
            mx = None

        tensors: dict[str, object] = {}
        selected = set(plan.tensor_names)
        for path in paths:
            expected = {name for name in selected if self.weight_map[name] == path.name}
            if mx is not None:
                shard = mx.load(str(path))
                missing = expected - set(shard)
                if missing:
                    raise ValueError(f"{path.name} is missing indexed tensor(s): {sorted(missing)}")
                for name in sorted(expected):
                    tensors[name] = shard[name]
                del shard
                continue
            from safetensors import safe_open

            with safe_open(str(path), framework="numpy") as handle:
                names = set(handle.keys())
                missing = expected - names
                if missing:
                    raise ValueError(f"{path.name} is missing indexed tensor(s): {sorted(missing)}")
                for name in sorted(expected):
                    tensors[name] = handle.get_tensor(name)
        if set(tensors) != selected:
            raise AssertionError("loader returned a tensor set different from the requested plan")
        return LoadedOfficialWeights(tensors=tensors, shards=paths)
