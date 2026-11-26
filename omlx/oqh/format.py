"""oQh checkpoint format.

A quantized linear ``<path>`` stores:

- ``<path>.weight``: uint32 codes in MLX affine packing, ``[..., N, K * bits / 32]``.
- ``<path>.rscale``: one scale per output row, ``[..., N]``.
- ``<path>.oqh_signs``: +-1 signs of the input rotation, ``[K / b, b]`` for
  block ``b``, in the activation dtype so kernels read them as is.

The real weight is ``rscale * (q - (2**bits - 1) / 2)`` in the basis
``H_b diag(signs)`` of the input axis, where ``H_b`` is the normalized
Walsh-Hadamard transform applied to each block of ``b`` input channels. Every
linear that reads the same producer output shares its signs and block, so the
producer can write the rotated activation once.

From AFFINE_LEVEL up, families that allow it keep group scales instead of the
row scale: ``<path>.oqh_scales`` and ``<path>.oqh_biases`` hold MLX affine
scales and biases of AFFINE_GROUP_SIZE inputs in the rotated basis. The names
differ from ``.scales`` so loaders without oQh refuse the tensors.

The tensors describe themselves: bits come from the packed width, the block
from the signs and the group from the scales. ``config["omlx_oqh"]`` records
the format version and each source path with its bits, block and, for group
scales, group size. The ``quantization`` entries stay those of the affine
tensors, so the stock loaders build every other module as usual.

This module needs no MLX: the cluster planner reads it.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np

CONFIG_KEY = "omlx_oqh"
FORMAT_VERSION = 1
# 256 inputs fit one simdgroup in the producer kernels, and wider blocks
# measured no better on model KL.
MAX_BLOCK = 256
# Blocks of linears whose input axis is split across tensor-parallel ranks
# must fit in one rank's share at this size.
TP_BLOCK_RANKS = 2
SUPPORTED_BITS = (2, 3, 4, 5, 6, 8)
# At 4 bits and up, group scales and biases on the rotated weights measured
# 8-10% lower KL than a row scale at the same size; below, the row scale wins.
AFFINE_LEVEL = 4
AFFINE_GROUP_SIZE = 128
# Families whose tensor-parallel strategy splits oQh tensors in place. The
# others replace linears with MLX's sharded layer types, which oQh has none of.
TENSOR_PARALLEL_TYPES = frozenset({"glm5_next", "deepseek_v41"})


def block_for(in_dim: int, row_parallel: bool = False) -> int:
    """Largest power-of-two block dividing the input width (and one rank's share)."""
    width = in_dim // TP_BLOCK_RANKS if row_parallel else in_dim
    if width <= 0 or (row_parallel and in_dim % TP_BLOCK_RANKS):
        width = in_dim
    return min(width & -width, MAX_BLOCK)


def make_signs(key: str, n: int) -> np.ndarray:
    """Deterministic +-1 signs for one producer, as int8."""
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=n)


def config_entry(bits: int, block: int, group_size: int | None = None) -> dict:
    entry = {"bits": int(bits), "block": int(block)}
    if group_size:
        entry["group_size"] = int(group_size)
    return entry


def oqh_paths(config: dict) -> dict[str, dict]:
    """{path: {"bits", "block"[, "group_size"]}} of the oQh tensors, or {}."""
    block = config.get(CONFIG_KEY) or {}
    if int(block.get("version", 0)) != FORMAT_VERSION:
        return {}
    return dict(block.get("tensors") or {})


def is_oqh_config(config: dict) -> bool:
    """Whether a checkpoint holds oQh tensors, in any format version."""
    return CONFIG_KEY in config


def checkpoint_version(model_path) -> int | None:
    """The oQh format version of a checkpoint folder, None without oQh tensors."""
    try:
        config = json.loads((Path(model_path) / "config.json").read_text())
    except (OSError, TypeError, ValueError):
        return None
    if not isinstance(config, dict) or not is_oqh_config(config):
        return None
    block = config[CONFIG_KEY]
    return int(block.get("version", 0)) if isinstance(block, dict) else 0


def is_oqh_checkpoint(model_path) -> bool:
    try:
        config = json.loads((Path(model_path) / "config.json").read_text())
    except (OSError, TypeError, ValueError):
        return False
    return isinstance(config, dict) and is_oqh_config(config)
