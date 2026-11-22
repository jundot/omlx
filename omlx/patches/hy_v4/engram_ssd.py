# SPDX-License-Identifier: Apache-2.0
"""Read Hy4 Flash Engram tables from their Safetensors on demand."""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from ..deepseek_v41.residency import header_with_offset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Table:
    weight_file: Path
    weight_key: str
    scale_file: Path | None = None
    scale_key: str | None = None
    bias_key: str | None = None
    bits: int | None = None
    group_size: int = 32


@dataclass(frozen=True)
class Plan:
    path: Path
    tables: dict[int, Table]
    skipped: dict[Path, frozenset[str]]
    table_bytes: int


def plan_engram_ssd(model_path: str | Path) -> Plan:
    """Locate every Engram table without reading its payload."""
    path = Path(model_path).resolve()
    config = json.loads((path / "config.json").read_text())
    layers = (config.get("text_config") or {}).get("engram_layer_ids") or []
    if config.get("model_type") != "hy_v4_flash" or not layers:
        raise ValueError("This checkpoint has no Hy4 Flash Engram tables")
    quant = config.get("quantization") or config.get("quantization_config") or {}
    index = json.loads((path / "model.safetensors.index.json").read_text())
    mapping = index["weight_map"]
    skipped: dict[Path, set[str]] = {}
    table_bytes = 0

    def locate(key, dtype):
        nonlocal table_bytes
        file = path / mapping[key]
        info = file.stat()
        entries, _ = header_with_offset(str(file), info.st_size, info.st_mtime_ns)
        if entries[key]["dtype"] != dtype:
            raise ValueError(f"Unsupported Hy4 Engram tensor: {key}")
        start, end = entries[key]["data_offsets"]
        table_bytes += end - start
        skipped.setdefault(file, set()).add(key)
        return file

    tables = {}
    for layer in layers:
        name = f"model.layers.{layer - 1}.engram_layer.multi_head_embedding.embedding"
        prefix = next(
            (p for p in (name, f"language_model.{name}") if f"{p}.weight" in mapping),
            None,
        )
        if prefix is None:
            raise ValueError(f"Missing Hy4 Engram table for layer {layer}")
        weight_key = f"{prefix}.weight"
        if f"{prefix}.scales" not in mapping:
            tables[layer] = Table(locate(weight_key, "BF16"), weight_key)
            continue
        recipe = quant.get(prefix, quant)
        scale_file = locate(f"{prefix}.scales", "BF16")
        if locate(f"{prefix}.biases", "BF16") != scale_file:
            raise ValueError(
                f"Hy4 Engram scales and biases must share a shard: {prefix}"
            )
        tables[layer] = Table(
            locate(weight_key, "U32"),
            weight_key,
            scale_file,
            f"{prefix}.scales",
            f"{prefix}.biases",
            recipe["bits"],
            recipe["group_size"],
        )
    return Plan(
        path,
        tables,
        {file: frozenset(keys) for file, keys in skipped.items()},
        table_bytes,
    )


class _Load:
    def __init__(self, plan):
        self.plan = plan
        self.readers = {}

    def close(self):
        for reader in self.readers.values():
            reader.close()
        self.readers.clear()


_current_load: ContextVar[_Load | None] = ContextVar(
    "hy4_engram_ssd_load", default=None
)
_hook_lock = Lock()
_hooks_installed = False


def open_embedding(layer):
    """Called by the Hy4 model constructor before allocating resident weights."""
    state = _current_load.get()
    if state is None:
        return None
    from ..deepseek_v41.storage import DiskEngramEmbedding

    table = state.plan.tables[layer]
    reader = DiskEngramEmbedding(
        table.weight_file,
        table.weight_key,
        table.scale_key,
        table.scale_file,
        bias_key=table.bias_key,
        bits=table.bits,
        group_size=table.group_size,
    )
    state.readers[layer] = reader
    return reader


def _install_load_hooks(vlm_utils):
    """Drop the tables from mlx-vlm's normal load so they are never read."""
    global _hooks_installed
    with _hook_lock:
        if _hooks_installed:
            return
        read = vlm_utils._load_safetensors
        load_weights = vlm_utils.load_model_weights

        def read_shard(path):
            state = _current_load.get()
            skipped = state.plan.skipped.get(Path(path).absolute()) if state else None
            if not skipped:
                return read(path)
            return {k: v for k, v in read(path).items() if k not in skipped}

        def load_model_weights(model, config, weights, model_path, **kwargs):
            state = _current_load.get()
            if state is not None and Path(model_path).resolve() == state.plan.path:
                prefixes = set()
                for table in state.plan.tables.values():
                    bare = table.weight_key.removesuffix(".weight")
                    bare = bare.removeprefix("language_model.")
                    prefixes.update((bare, f"language_model.{bare}"))
                config = dict(config)
                for key in ("quantization", "quantization_config"):
                    if isinstance(config.get(key), dict):
                        config[key] = {
                            k: v for k, v in config[key].items() if k not in prefixes
                        }
            return load_weights(model, config, weights, model_path, **kwargs)

        vlm_utils._load_safetensors = read_shard
        vlm_utils.load_model_weights = load_model_weights
        _hooks_installed = True


def load(model_path, *, trust_remote_code=False, lazy=False):
    """Load Hy4 Flash with its Engram tables read from SSD."""
    from mlx_vlm import utils

    state = _Load(plan_engram_ssd(model_path))
    _install_load_hooks(utils)
    token = _current_load.set(state)
    try:
        model, processor = utils.load(
            state.plan.path, trust_remote_code=trust_remote_code, lazy=lazy
        )
        if state.readers.keys() != state.plan.tables.keys():
            raise ValueError("Hy4 SSD load did not attach every Engram table")
    except BaseException:
        state.close()
        raise
    finally:
        _current_load.reset(token)
    model._omlx_engram_ssd = state
    logger.info(
        "Hy4 Engram SSD offload: %d tables, %.2f GiB",
        len(state.readers),
        state.plan.table_bytes / 1024**3,
    )
    return model, processor
