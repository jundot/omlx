# SPDX-License-Identifier: Apache-2.0
"""Load Hy4-family Engram tables from their Safetensors on demand."""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from ..deepseek_v41.residency import header_with_offset

logger = logging.getLogger(__name__)

MODEL_TYPES = frozenset({"hy_v4_flash", "hy_v4_vl"})
_TABLE_SUFFIX = ".engram_layer.multi_head_embedding.embedding"
_PACKED_DTYPES = {"BF16", "F16", "F32"}


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
    table_only_shards: frozenset[Path]
    table_bytes: int


def plan_engram_ssd(model_path: str | Path) -> Plan:
    """Resolve storage and validate every table without reading its payload."""
    path = Path(model_path).resolve()
    config = json.loads((path / "config.json").read_text())
    text = config.get("text_config") or {}
    layers = text.get("engram_layer_ids") or []
    if config.get("model_type") not in MODEL_TYPES or not layers:
        raise ValueError("This checkpoint has no supported Hy4 Engram tables")
    orders = text["engram_layer_head_vocab_sizes"]
    width = text["engram_n_embed_per_ngram"] // text["engram_n_head_per_ngram"]
    quant = config.get("quantization") or config.get("quantization_config") or {}

    def header(file):
        info = file.stat()
        entries, payload = header_with_offset(str(file), info.st_size, info.st_mtime_ns)
        return entries, info.st_size - payload

    index = path / "model.safetensors.index.json"
    if index.is_file():
        mapping = json.loads(index.read_text())["weight_map"]
    else:
        mapping = {}
        for shard in path.glob("*.safetensors"):
            for key in header(shard)[0].keys() - {"__metadata__"}:
                if key in mapping:
                    raise ValueError(f"Duplicate Safetensors key: {key}")
                mapping[key] = shard.name

    skipped: dict[Path, set[str]] = {}
    table_bytes = 0

    def tensor(key, dtype_widths, shape):
        nonlocal table_bytes
        shard = mapping[key]
        if Path(shard).name != shard:
            raise ValueError(f"Invalid Hy4 Engram shard name: {shard}")
        file = path / shard
        entries, payload_size = header(file)
        desc = entries[key]
        start, end = desc["data_offsets"]
        size = dtype_widths.get(desc["dtype"])
        if (
            not size
            or desc["shape"] != list(shape)
            or not 0 <= start < end <= payload_size
            or end - start != shape[0] * shape[1] * size
        ):
            raise ValueError(f"Invalid Hy4 Engram tensor: {key}")
        skipped.setdefault(file, set()).add(key)
        table_bytes += end - start
        return file, desc["dtype"]

    tables = {}
    for layer in layers:
        module = f"model.layers.{layer - 1}{_TABLE_SUFFIX}"
        candidates = [
            p for p in (module, f"language_model.{module}") if f"{p}.weight" in mapping
        ]
        if len(candidates) != 1:
            raise ValueError(f"Hy4 Engram layer {layer} must have exactly one table")
        prefix = candidates[0]
        rows = sum(sum(order) for order in orders[str(layer)])
        weight_key, scale_key, bias_key = (
            f"{prefix}.{part}" for part in ("weight", "scales", "biases")
        )
        if scale_key not in mapping and bias_key not in mapping:
            file, _ = tensor(weight_key, {"BF16": 2}, (rows, width))
            tables[layer] = Table(file, weight_key)
            continue
        if scale_key not in mapping or bias_key not in mapping:
            raise ValueError(f"Incomplete affine Hy4 Engram table: {prefix}")
        recipe = quant.get(prefix, quant)
        bits, group = recipe.get("bits"), recipe.get("group_size")
        if (
            recipe.get("mode", "affine") != "affine"
            or bits not in (2, 3, 4, 5, 6, 8)
            or group not in (32, 64, 128)
            or width % group
            or width * bits % 32
        ):
            raise ValueError(f"Unsupported Hy4 Engram quantization: {prefix}")
        file, _ = tensor(weight_key, {"U32": 4}, (rows, width * bits // 32))
        types = {dtype: 4 if dtype == "F32" else 2 for dtype in _PACKED_DTYPES}
        scales, scales_dtype = tensor(scale_key, types, (rows, width // group))
        biases, biases_dtype = tensor(bias_key, types, (rows, width // group))
        if scales != biases or scales_dtype != biases_dtype:
            raise ValueError(
                f"Hy4 Engram scales and biases must share a shard and dtype: {prefix}"
            )
        tables[layer] = Table(
            file, weight_key, scales, scale_key, bias_key, bits, group
        )

    return Plan(
        path,
        tables,
        {file: frozenset(keys) for file, keys in skipped.items()},
        frozenset(
            file
            for file, keys in skipped.items()
            if header(file)[0].keys() - {"__metadata__"} == keys
        ),
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
    if layer in state.readers:
        raise ValueError(f"Duplicate Hy4 Engram module for layer {layer}")
    reader = DiskEngramEmbedding(
        table.weight_file,
        table.weight_key,
        table.scale_key,
        table.scale_file,
        bias_key=table.bias_key,
        bits=table.bits,
        group_size=table.group_size,
        preserve_dtype=True,
    )
    state.readers[layer] = reader
    return reader


def _install_load_hooks(vlm_utils):
    """Keep mlx-vlm's regular FP8, oQ and oQh load paths for the backbone."""
    global _hooks_installed
    with _hook_lock:
        if _hooks_installed:
            return
        read = vlm_utils._load_safetensors
        load_weights = vlm_utils.load_model_weights

        def read_shard(path):
            state = _current_load.get()
            file = Path(path).absolute() if state is not None else None
            if state is None or file not in state.plan.skipped:
                return read(path)
            if file in state.plan.table_only_shards:
                return {}
            # MX arrays load lazily; discard table arrays in mixed shards before
            # they can be sanitized, quantized or evaluated.
            return {
                k: v for k, v in read(path).items() if k not in state.plan.skipped[file]
            }

        def load_model_weights(model, config, weights, model_path, **kwargs):
            state = _current_load.get()
            if state is not None and Path(model_path).resolve() == state.plan.path:
                prefixes = set()
                for table in state.plan.tables.values():
                    name = table.weight_key.removesuffix(".weight")
                    bare = name.removeprefix("language_model.")
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
    """Load a Hy4 VLM using SSD tables and the normal checkpoint handling."""
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
