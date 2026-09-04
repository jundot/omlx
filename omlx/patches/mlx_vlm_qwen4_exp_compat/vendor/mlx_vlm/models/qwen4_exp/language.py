from __future__ import annotations

import json
import math
import mmap
import os
import struct
import weakref
from bisect import bisect_right
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .cache import ArraysCache, BatchKVCache, KVCache, QuantizedKVCache, dynamic_roll
from ..qwen3_5.language import LanguageModel as Qwen3_5LanguageModel
from ..qwen3_5.language import (
    Qwen3_5Attention,
    Qwen3_5GatedDeltaNet,
    _create_qwen3_5_attention_mask,
    _create_qwen3_5_ssm_mask,
    _target_verify_linear,
    _target_verify_linears,
)
from ..qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock
from .config import ModelConfig, TextConfig
from .qsa_fast import (
    contiguous_causal_gathered_qsa,
    contiguous_causal_gathered_qsa_decode,
    pool_completed_index_keys,
)

_PLE_RUNTIME_MODEL_PATH: Path | None = None
_PLE_RUNTIME_MODE = "resident"
_HYPER_SPLIT_INDICES: dict[tuple[int, int], tuple[mx.array, mx.array]] = {}


@dataclass(frozen=True)
class Qwen4ExpMTPRuntime:
    """Lightning MTP construction decision for the next model load."""

    enabled: bool = False
    checkpoint_prefix: str | None = None


_MTP_RUNTIME = Qwen4ExpMTPRuntime()


@dataclass(frozen=True)
class _PLESpeculativeState:
    """PLE inputs needed to restore a partially accepted verify window."""

    history: mx.array
    input_ids: mx.array
    conv_state: mx.array
    conv_inputs: mx.array


def configure_mtp_runtime(
    model_path: str | Path,
    *,
    enabled: bool,
) -> Qwen4ExpMTPRuntime:
    """Detect and bind an embedded Qwen4 Lightning MTP checkpoint head."""
    global _MTP_RUNTIME

    checkpoint_prefix = None
    if enabled:
        from omlx.utils.model_loading import _checkpoint_qwen4_mtp_weight_prefix

        checkpoint_prefix = _checkpoint_qwen4_mtp_weight_prefix(model_path)

    _MTP_RUNTIME = Qwen4ExpMTPRuntime(
        enabled=bool(enabled and checkpoint_prefix),
        checkpoint_prefix=checkpoint_prefix,
    )
    return _MTP_RUNTIME


def get_mtp_runtime() -> Qwen4ExpMTPRuntime:
    return _MTP_RUNTIME


def resolve_ple_runtime_mode(
    requested: str, *, checkpoint_bytes: int, physical_memory: int
) -> str:
    requested = requested.strip().lower()
    if requested == "ssd_mmap":
        requested = "mmap"
    if requested not in {"auto", "resident", "mmap"}:
        raise ValueError("OMLX_QWEN4_PLE_MODE must be auto, resident, or mmap")
    if requested != "auto":
        return requested
    return "mmap" if checkpoint_bytes > physical_memory * 0.70 else "resident"


def configure_ple_runtime(model_path: str | Path, mode: str | None = None) -> str:
    """Bind same-directory PLE storage before Qwen4 model construction."""
    global _PLE_RUNTIME_MODEL_PATH, _PLE_RUNTIME_MODE

    compute_path = Path(model_path).expanduser().resolve()
    requested = mode or os.environ.get("OMLX_QWEN4_PLE_MODE")
    if requested is None:
        requested = "auto"
    checkpoint_bytes = sum(
        path.stat().st_size
        for path in compute_path.glob("*.safetensors")
    )
    physical_memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    _PLE_RUNTIME_MODE = resolve_ple_runtime_mode(
        requested,
        checkpoint_bytes=checkpoint_bytes,
        physical_memory=physical_memory,
    )
    _PLE_RUNTIME_MODEL_PATH = compute_path
    return _PLE_RUNTIME_MODE


def get_ple_runtime_mode() -> str:
    return _PLE_RUNTIME_MODE


def _append_indexer_positions(
    cached: Optional[mx.array], position_ids: mx.array
) -> mx.array:
    if cached is None:
        return position_ids
    if cached.ndim == 3 and position_ids.ndim == 2:
        position_ids = mx.broadcast_to(
            position_ids[None],
            (cached.shape[0], *position_ids.shape),
        )
    elif cached.ndim == 2 and position_ids.ndim == 3:
        cached = mx.broadcast_to(cached[None], (position_ids.shape[0], *cached.shape))
    elif cached.ndim != position_ids.ndim:
        raise ValueError(
            "QSA position IDs must be 2-D text positions or 3-D MRoPE positions, "
            f"got cached={cached.shape} and current={position_ids.shape}."
        )
    return mx.concatenate([cached, position_ids], axis=-1)


class _QSAIndexerCache:
    """Capacity-backed raw and completed-block state shared by QSA caches.

    Raw keys and positions remain part of the public/persisted cache state.
    The normalized, RoPE-rotated block bank is deliberately ephemeral: state
    restore, row extraction/filtering, and rollback can rebuild it exactly from
    those raw tensors without changing the serialized cache schema.
    """

    index_step = 8192

    def _init_indexer_cache(self):
        self._index_keys = None
        self._index_position_ids = None
        self._index_offset = 0
        self._index_capacity_managed = True
        self._invalidate_pooled_indexer()

    @property
    def index_keys(self):
        if self._index_keys is None:
            return None
        return self._index_keys[:, : self._index_offset]

    @index_keys.setter
    def index_keys(self, value):
        self._index_keys = value
        self._index_offset = 0 if value is None else int(value.shape[1])
        self._index_capacity_managed = False
        self._invalidate_pooled_indexer()

    @property
    def index_position_ids(self):
        if self._index_position_ids is None:
            return None
        return self._index_position_ids[..., : self._index_offset]

    @index_position_ids.setter
    def index_position_ids(self, value):
        self._index_position_ids = value
        self._invalidate_pooled_indexer()

    def _restore_indexer_state(self, keys, position_ids):
        if (keys is None) != (position_ids is None):
            raise ValueError("QSA raw keys and positions must be restored together")
        if keys is not None and (
            keys.ndim != 3
            or position_ids.ndim not in {2, 3}
            or keys.shape[1] != position_ids.shape[-1]
        ):
            raise ValueError("Restored QSA raw keys and positions are misaligned")
        self._index_keys = keys
        self._index_position_ids = position_ids
        self._index_offset = 0 if keys is None else int(keys.shape[1])
        self._index_capacity_managed = False
        self._invalidate_pooled_indexer()

    @staticmethod
    def _growth_capacity(current: int, needed: int, step: int) -> int:
        stepped = ((needed + step - 1) // step) * step
        return max(stepped, 2 * current if current else step)

    def _ensure_indexer_capacity(
        self,
        sample_keys: mx.array,
        sample_positions: mx.array,
        needed: int,
    ) -> None:
        if self._index_keys is None:
            current = 0
        else:
            current = min(
                int(self._index_keys.shape[1]),
                int(self._index_position_ids.shape[-1]),
            )
        if needed <= current:
            return
        capacity = ((needed + self.index_step - 1) // self.index_step) * self.index_step
        if current and self._index_capacity_managed:
            capacity = max(capacity, 2 * current)
        new_keys = mx.zeros(
            (sample_keys.shape[0], capacity, sample_keys.shape[-1]),
            dtype=sample_keys.dtype,
        )
        if sample_positions.ndim == 3:
            position_shape = (
                sample_positions.shape[0],
                sample_positions.shape[1],
                capacity,
            )
        else:
            position_shape = (sample_positions.shape[0], capacity)
        new_positions = mx.zeros(position_shape, dtype=sample_positions.dtype)
        if self._index_keys is not None and self._index_offset:
            new_keys[:, : self._index_offset] = self._index_keys[
                :, : self._index_offset
            ]
            new_positions[..., : self._index_offset] = self._index_position_ids[
                ..., : self._index_offset
            ]
        self._index_keys = new_keys
        self._index_position_ids = new_positions
        self._index_capacity_managed = True

    def update_indexer(self, keys: mx.array, position_ids: mx.array):
        if keys.ndim != 3 or position_ids.ndim not in {2, 3}:
            raise ValueError("QSA index updates require [B,S,D] keys and positions")
        length = int(keys.shape[1])
        if position_ids.shape[-1] != length:
            raise ValueError("QSA index update keys and positions are misaligned")

        if self._index_position_ids is not None:
            if self._index_position_ids.ndim == 3 and position_ids.ndim == 2:
                position_ids = mx.broadcast_to(
                    position_ids[None],
                    (self._index_position_ids.shape[0], *position_ids.shape),
                )
            elif self._index_position_ids.ndim == 2 and position_ids.ndim == 3:
                # MRoPE promotion is rare. Collapse the old position backing to
                # its logical prefix so the capacity grow below creates a
                # writable three-coordinate allocation.
                old_positions = self._index_position_ids[
                    ..., : self._index_offset
                ]
                self._index_position_ids = mx.broadcast_to(
                    old_positions[None],
                    (position_ids.shape[0], *old_positions.shape),
                )
                self._index_capacity_managed = False
            elif self._index_position_ids.ndim != position_ids.ndim:
                raise ValueError("QSA position rank changed incompatibly")

        end = self._index_offset + length
        self._ensure_indexer_capacity(keys, position_ids, end)
        self._index_keys[:, self._index_offset : end] = keys
        self._index_position_ids[..., self._index_offset : end] = position_ids
        self._index_offset = end
        return self.index_keys, self.index_position_ids

    def _invalidate_pooled_indexer(self):
        self._pooled_index_keys = None
        self._pooled_index_offset = 0
        self._pooled_index_ratio = None
        self._pooled_index_tag = None

    def pooled_indexer_keys(
        self,
        compress_ratio: int,
        index_key_norm,
        apply_index_rope,
        *,
        cache_tag=None,
    ) -> mx.array:
        """Return the completed block bank, computing only its new suffix."""

        if self._index_keys is None or self._index_position_ids is None:
            raise ValueError("QSA pooled keys require raw indexer state")
        complete_blocks = self._index_offset // compress_ratio
        if (
            self._pooled_index_ratio != compress_ratio
            or self._pooled_index_tag is not cache_tag
            or self._pooled_index_offset > complete_blocks
        ):
            self._invalidate_pooled_indexer()
            self._pooled_index_ratio = compress_ratio
            self._pooled_index_tag = cache_tag

        start_block = self._pooled_index_offset
        if start_block < complete_blocks:
            new_pooled = pool_completed_index_keys(
                self.index_keys,
                self.index_position_ids,
                compress_ratio=compress_ratio,
                index_key_norm=index_key_norm,
                apply_index_rope=apply_index_rope,
                start_block=start_block,
                stop_block=complete_blocks,
            )
            current_capacity = (
                0
                if self._pooled_index_keys is None
                else int(self._pooled_index_keys.shape[1])
            )
            if complete_blocks > current_capacity:
                block_step = max(1, self.index_step // compress_ratio)
                capacity = self._growth_capacity(
                    current_capacity,
                    complete_blocks,
                    block_step,
                )
                new_buffer = mx.zeros(
                    (new_pooled.shape[0], capacity, new_pooled.shape[-1]),
                    dtype=new_pooled.dtype,
                )
                if self._pooled_index_keys is not None and start_block:
                    new_buffer[:, :start_block] = self._pooled_index_keys[
                        :, :start_block
                    ]
                self._pooled_index_keys = new_buffer
            self._pooled_index_keys[:, start_block:complete_blocks] = new_pooled
            self._pooled_index_offset = complete_blocks

        if self._pooled_index_keys is None:
            return mx.zeros(
                (self._index_keys.shape[0], 0, self._index_keys.shape[-1]),
                dtype=self._index_keys.dtype,
            )
        return self._pooled_index_keys[:, : self._pooled_index_offset]

    def _trim_indexer(self, length: int):
        self._index_offset = min(self._index_offset, max(0, int(length)))
        self._invalidate_pooled_indexer()

    @property
    def indexer_nbytes(self):
        size = 0
        for array in (
            self._index_keys,
            self._index_position_ids,
            self._pooled_index_keys,
        ):
            if array is not None:
                size += array.nbytes
        return size


class QSAKVCache(_QSAIndexerCache, KVCache):
    """KV cache with the raw indexer keys and multimodal positions used by QSA."""

    # Hybrid/TurboQuant caches do not currently expose a way to carry the
    # indexer's unprojected keys. Uniform quantization uses the specialized
    # QSAQuantizedKVCache below; other schemes leave this cache in float.
    preserve_auxiliary_kv_state = True
    step = 8192
    geometric_growth = True

    def __init__(self):
        super().__init__()
        self._init_indexer_cache()

    @property
    def state(self):
        if self.keys is None:
            return None, None, self.index_keys, self.index_position_ids
        return (
            self.keys[..., : self.offset, :],
            self.values[..., : self.offset, :],
            self.index_keys,
            self.index_position_ids,
        )

    @state.setter
    def state(self, value):
        self.keys, self.values, index_keys, index_position_ids = value
        self.offset = 0 if self.keys is None else self.keys.shape[2]
        self._geometric_capacity_managed = False
        self._restore_indexer_state(index_keys, index_position_ids)

    def trim(self, n):
        n = min(self.offset, n)
        super().trim(n)
        self._trim_indexer(self.offset)
        return n

    def extract(self, idx):
        cache = QSAKVCache()
        if self.keys is not None:
            cache.keys = mx.contiguous(
                self.keys[idx : idx + 1, :, : self.offset, :]
            )
            cache.values = mx.contiguous(
                self.values[idx : idx + 1, :, : self.offset, :]
            )
            cache.offset = self.offset
            cache._geometric_capacity_managed = False
        if self.index_keys is not None:
            index_keys = mx.contiguous(self.index_keys[idx : idx + 1])
            if self.index_position_ids.ndim == 3:
                index_position_ids = mx.contiguous(
                    self.index_position_ids[:, idx : idx + 1]
                )
            else:
                index_position_ids = mx.contiguous(
                    self.index_position_ids[idx : idx + 1]
                )
            cache._restore_indexer_state(index_keys, index_position_ids)
        return cache

    def filter(self, batch_indices):
        if self.keys is not None:
            self.keys = self.keys[batch_indices]
            self.values = self.values[batch_indices]
        if self.index_keys is not None:
            self.index_keys = self.index_keys[batch_indices]
            if self.index_position_ids.ndim == 3:
                self.index_position_ids = self.index_position_ids[:, batch_indices]
            else:
                self.index_position_ids = self.index_position_ids[batch_indices]

    def to_batch(self, left_padding):
        """Convert a singleton QSA cache without dropping indexer state."""

        batch = BatchQSAKVCache(left_padding)
        padding = mx.array(left_padding)
        if self.empty() and self.index_keys is None:
            return batch
        if padding.size != 1:
            raise ValueError(
                "A warm QSA cache can only seed one batch row, got "
                f"left_padding={padding.tolist()}"
            )
        pad = int(padding.item())
        if not self.empty():
            keys, values = self.state[:2]
            if pad:
                keys = mx.pad(keys, [(0, 0), (0, 0), (pad, 0), (0, 0)])
                values = mx.pad(values, [(0, 0), (0, 0), (pad, 0), (0, 0)])
            batch.kv_cache.state = (
                keys,
                values,
                mx.array([self.offset], dtype=mx.int32),
                padding.astype(mx.int32),
            )
        if self.index_keys is not None:
            index_keys = self.index_keys[:, : self.offset]
            positions = self.index_position_ids[..., : self.offset]
            if pad:
                index_keys = mx.pad(index_keys, [(0, 0), (pad, 0), (0, 0)])
                positions = mx.pad(
                    positions,
                    (
                        [(0, 0), (0, 0), (pad, 0)]
                        if positions.ndim == 3
                        else [(0, 0), (pad, 0)]
                    ),
                )
            batch.index_keys = index_keys
            batch.index_position_ids = positions
            batch.index_offset = index_keys.shape[1]
        return batch

    @classmethod
    def merge(cls, caches):
        return BatchQSAKVCache.merge(caches)

    def to_quantized(self, group_size: int = 64, bits: int = 4):
        base = super().to_quantized(group_size=group_size, bits=bits)
        cache = QSAQuantizedKVCache(group_size=group_size, bits=bits)
        cache.keys = base.keys
        cache.values = base.values
        cache.offset = base.offset
        cache.index_keys = self.index_keys
        cache.index_position_ids = self.index_position_ids
        return cache

    @property
    def nbytes(self):
        return super().nbytes + self.indexer_nbytes


class BatchQSAKVCache:
    """Batch KV cache that keeps QSA raw keys and text/MRoPE positions aligned."""

    def __init__(self, left_padding):
        self.kv_cache = BatchKVCache(left_padding)
        self.index_keys = None
        self.index_position_ids = None
        self.index_offset = 0

    @property
    def offset(self):
        return self.kv_cache.offset

    @property
    def left_padding(self):
        return self.kv_cache.left_padding

    def update_and_fetch(self, keys, values):
        return self.kv_cache.update_and_fetch(keys, values)

    def update_indexer(self, keys: mx.array, position_ids: mx.array):
        if self.index_keys is None:
            self.index_keys = keys
            self.index_position_ids = position_ids
        else:
            self.index_keys = mx.concatenate([self.index_keys, keys], axis=1)
            self.index_position_ids = _append_indexer_positions(
                self.index_position_ids, position_ids
            )
        self.index_offset = self.index_keys.shape[1]
        return self.index_keys, self.index_position_ids

    def prepare(self, **kwargs):
        self.kv_cache.prepare(**kwargs)

    def finalize(self):
        right_padding = getattr(self.kv_cache, "_right_padding", None)
        self.kv_cache.finalize()
        if right_padding is None or self.index_keys is None:
            return
        self.index_keys = dynamic_roll(self.index_keys, right_padding, axis=1)
        if self.index_position_ids.ndim == 3:
            self.index_position_ids = dynamic_roll(
                self.index_position_ids, right_padding[None], axis=2
            )
        else:
            self.index_position_ids = dynamic_roll(
                self.index_position_ids, right_padding, axis=1
            )

    def make_mask(self, *args, **kwargs):
        return self.kv_cache.make_mask(*args, **kwargs)

    def filter(self, batch_indices):
        min_left = int(self.left_padding[batch_indices].min().item())
        self.kv_cache.filter(batch_indices)
        if self.index_keys is None:
            return
        self.index_keys = self.index_keys[batch_indices]
        if self.index_position_ids.ndim == 3:
            self.index_position_ids = self.index_position_ids[:, batch_indices]
        else:
            self.index_position_ids = self.index_position_ids[batch_indices]
        if min_left > 0:
            self.index_keys = self.index_keys[:, min_left:]
            self.index_position_ids = self.index_position_ids[..., min_left:]
            self.index_offset -= min_left

    @staticmethod
    def _pad_index(cache, target, sample_keys, sample_positions):
        length = (
            0
            if cache.index_keys is None
            else getattr(cache, "index_offset", cache.index_keys.shape[1])
        )
        if isinstance(length, mx.array):
            if length.size != 1:
                raise ValueError(
                    "QSA index length must be scalar after row normalization"
                )
            length = int(length.item())
        else:
            length = int(length)
        left = target - length
        if cache.index_keys is None:
            offset = cache.offset
            batch_size = offset.shape[0] if isinstance(offset, mx.array) else 1
            keys = mx.zeros(
                (batch_size, 0, sample_keys.shape[-1]),
                dtype=sample_keys.dtype,
            )
            if sample_positions.ndim == 3:
                positions = mx.zeros(
                    (sample_positions.shape[0], batch_size, 0),
                    dtype=sample_positions.dtype,
                )
            else:
                positions = mx.zeros(
                    (batch_size, 0), dtype=sample_positions.dtype
                )
        else:
            keys = cache.index_keys[:, :length]
            positions = cache.index_position_ids[..., :length]
            # Widen 2-D text positions to the join's widest rank before
            # padding (#3294 item 2): the runtime update path already
            # broadcasts text up to MRoPE in _append_indexer_positions; joins
            # must apply the same rule or the concatenate below sees ranks 2
            # and 3. Replicating across MRoPE channels matches runtime.
            if sample_positions.ndim == 3 and positions.ndim == 2:
                positions = mx.broadcast_to(
                    positions[None],
                    (sample_positions.shape[0], *positions.shape),
                )
        if left:
            keys = mx.pad(keys, [(0, 0), (left, 0), (0, 0)])
            positions = mx.pad(
                positions,
                (
                    [(0, 0), (0, 0), (left, 0)]
                    if positions.ndim == 3
                    else [(0, 0), (left, 0)]
                ),
            )
        return keys, positions

    def extend(self, other):
        if not isinstance(other, BatchQSAKVCache):
            raise TypeError(f"Cannot extend BatchQSAKVCache with {type(other)}")

        for cache in (self, other):
            if (cache.index_keys is None) != (cache.index_position_ids is None):
                raise ValueError("QSA raw keys and positions must be extended together")
            if cache.index_keys is None:
                if cache.kv_cache.size():
                    raise ValueError("Cannot extend QSA KV state without indexer state")
            elif cache.index_offset != cache.kv_cache.size():
                raise ValueError(
                    "QSA extend requires aligned KV and indexer widths, got "
                    f"kv={cache.kv_cache.size()} and indexer={cache.index_offset}"
                )

        sample_keys = (
            self.index_keys if self.index_keys is not None else other.index_keys
        )
        # Prefer the WIDEST position rank over "first non-None" (#3294 item
        # 2): promotion only widens, so a 2-D sample would strand a 3-D row
        # with nothing to promote to, and position_axis would be picked from
        # the wrong rank. Order-sensitive defect, so pick from both sides.
        self_positions = self.index_position_ids
        other_positions = other.index_position_ids
        if (
            self_positions is not None
            and other_positions is not None
            and self_positions.ndim != other_positions.ndim
        ):
            sample_positions = (
                other_positions if self_positions.ndim == 2 else self_positions
            )
        elif self_positions is not None:
            sample_positions = self_positions
        else:
            sample_positions = other_positions
        if sample_keys is None or sample_positions is None:
            self.kv_cache.extend(other.kv_cache)
            return
        target = max(self.index_offset, other.index_offset)
        left = self._pad_index(self, target, sample_keys, sample_positions)
        right = self._pad_index(other, target, sample_keys, sample_positions)
        index_keys = mx.concatenate([left[0], right[0]], axis=0)
        position_axis = 1 if sample_positions.ndim == 3 else 0
        index_position_ids = mx.concatenate(
            [left[1], right[1]], axis=position_axis
        )

        self.kv_cache.extend(other.kv_cache)
        self.index_keys = index_keys
        self.index_position_ids = index_position_ids
        self.index_offset = target

    def extract(self, idx):
        cache = QSAKVCache()
        base = self.kv_cache.extract(idx)
        cache.keys, cache.values, cache.offset = base.keys, base.values, base.offset
        if self.index_keys is not None:
            padding = int(self.left_padding[idx].item())
            cache.index_keys = mx.contiguous(
                self.index_keys[idx : idx + 1, padding : self.index_offset]
            )
            if self.index_position_ids.ndim == 3:
                cache.index_position_ids = mx.contiguous(
                    self.index_position_ids[
                        :, idx : idx + 1, padding : self.index_offset
                    ]
                )
            else:
                cache.index_position_ids = mx.contiguous(
                    self.index_position_ids[idx : idx + 1, padding : self.index_offset]
                )
        return cache

    @classmethod
    def merge(cls, caches):
        rows = []
        for cache in caches:
            if isinstance(cache, cls):
                batch_size = int(cache.offset.shape[0])
                if cache.kv_cache.keys is None:
                    if cache.index_keys is not None:
                        raise ValueError(
                            "Cannot merge a QSA batch with indexer state but no KV state"
                        )
                    rows.extend(QSAKVCache() for _ in range(batch_size))
                else:
                    rows.extend(cache.extract(idx) for idx in range(batch_size))
            elif isinstance(cache, QSAKVCache):
                rows.append(cache)
            else:
                raise TypeError(f"Cannot merge QSA cache with {type(cache)}")

        out = cls([0] * len(rows))
        if not rows:
            return out

        lengths = []
        for row in rows:
            kv_length = int(row.offset)
            if row.keys is None:
                if kv_length:
                    raise ValueError("QSA cache has a non-zero offset without KV state")
            elif kv_length > row.keys.shape[2]:
                raise ValueError("QSA cache offset exceeds its KV storage")

            if (row.index_keys is None) != (row.index_position_ids is None):
                raise ValueError("QSA raw keys and positions must be merged together")
            index_length = 0 if row.index_keys is None else row.index_keys.shape[1]
            if row.index_position_ids is not None and (
                row.index_position_ids.ndim not in {2, 3}
                or row.index_position_ids.shape[-1] != index_length
            ):
                raise ValueError("QSA raw keys and positions are misaligned")
            if index_length != kv_length:
                raise ValueError(
                    "QSA merge requires aligned KV and indexer lengths, got "
                    f"kv={kv_length} and indexer={index_length}"
                )
            lengths.append(index_length)

        out.kv_cache = BatchKVCache.merge(rows)
        sample = next((row for row in rows if row.index_keys is not None), None)
        if sample is None:
            return out
        # Pick the widest position rank across every cache, not the first
        # non-None sample (#3294 item 2): promotion only widens, so a 2-D
        # first sample would strand a 3-D row at concatenate time.
        widest_positions = sample.index_position_ids
        for row in rows:
            pos = row.index_position_ids
            if pos is not None and pos.ndim > widest_positions.ndim:
                widest_positions = pos
        target = out.kv_cache.size()
        if target != max(lengths):
            raise ValueError(
                "QSA merge produced different KV and indexer widths, got "
                f"kv={target} and indexer={max(lengths)}"
            )
        padded_rows = [
            cls._pad_index(
                row,
                target,
                sample.index_keys,
                widest_positions,
            )
            for row in rows
        ]
        out.index_keys = mx.concatenate([row[0] for row in padded_rows], axis=0)
        position_axis = 1 if widest_positions.ndim == 3 else 0
        out.index_position_ids = mx.concatenate(
            [row[1] for row in padded_rows], axis=position_axis
        )
        out.index_offset = target
        return out

    def size(self):
        return self.kv_cache.size()

    def empty(self):
        return self.kv_cache.empty()

    def is_trimmable(self):
        return self.kv_cache.is_trimmable()

    def trim(self, n):
        trimmed = self.kv_cache.trim(n)
        self.index_offset = max(0, self.index_offset - trimmed)
        # Slice the physical arrays like the singleton trim does:
        # update_indexer concatenates onto them and re-derives index_offset
        # from shape[1], so stale draft columns would otherwise fossilize
        # and desync the indexer from the KV by the trimmed amount.
        if trimmed and self.index_keys is not None:
            self.index_keys = self.index_keys[:, : self.index_offset]
            self.index_position_ids = self.index_position_ids[
                ..., : self.index_offset
            ]
        return trimmed

    @property
    def state(self):
        return (
            self.kv_cache.state,
            (
                None
                if self.index_keys is None
                else self.index_keys[:, : self.index_offset]
            ),
            (
                None
                if self.index_position_ids is None
                else self.index_position_ids[..., : self.index_offset]
            ),
        )

    @state.setter
    def state(self, value):
        kv_state, self.index_keys, self.index_position_ids = value
        self.kv_cache.state = kv_state
        self.index_offset = 0 if self.index_keys is None else self.index_keys.shape[1]

    @property
    def nbytes(self):
        extra = 0
        if self.index_keys is not None:
            extra = self.index_keys.nbytes + self.index_position_ids.nbytes
        return self.kv_cache.nbytes + extra


class QSAQuantizedKVCache(_QSAIndexerCache, QuantizedKVCache):
    """Uniformly quantized QSA cache that retains float indexer state."""

    preserve_auxiliary_kv_state = True
    step = 8192
    geometric_growth = True

    def __init__(self, group_size: int = 64, bits: int = 8):
        super().__init__(group_size=group_size, bits=bits)
        self._init_indexer_cache()

    @property
    def state(self):
        if self.keys is None:
            keys, values = None, None
        else:
            keys, values = super().state
        return keys, values, self.index_keys, self.index_position_ids

    @state.setter
    def state(self, value):
        self.keys, self.values, index_keys, index_position_ids = value
        self.offset = 0 if self.keys is None else self.keys[0].shape[2]
        self._geometric_capacity_managed = False
        self._restore_indexer_state(index_keys, index_position_ids)

    def trim(self, n):
        n = min(self.offset, n)
        super().trim(n)
        self._trim_indexer(self.offset)
        return n

    def extract(self, idx):
        cache = QSAQuantizedKVCache(self.group_size, self.bits)
        if self.keys is not None:
            cache.keys = tuple(
                mx.contiguous(x[idx : idx + 1, :, : self.offset, :])
                for x in self.keys
            )
            cache.values = tuple(
                mx.contiguous(x[idx : idx + 1, :, : self.offset, :])
                for x in self.values
            )
            cache.offset = self.offset
            cache._geometric_capacity_managed = False
        if self.index_keys is not None:
            index_keys = mx.contiguous(self.index_keys[idx : idx + 1])
            if self.index_position_ids.ndim == 3:
                index_position_ids = mx.contiguous(
                    self.index_position_ids[:, idx : idx + 1]
                )
            else:
                index_position_ids = mx.contiguous(
                    self.index_position_ids[idx : idx + 1]
                )
            cache._restore_indexer_state(index_keys, index_position_ids)
        return cache

    def filter(self, batch_indices):
        if self.keys is not None:
            self.keys = tuple(x[batch_indices] for x in self.keys)
            self.values = tuple(x[batch_indices] for x in self.values)
        if self.index_keys is not None:
            self.index_keys = self.index_keys[batch_indices]
            if self.index_position_ids.ndim == 3:
                self.index_position_ids = self.index_position_ids[:, batch_indices]
            else:
                self.index_position_ids = self.index_position_ids[batch_indices]

    @property
    def nbytes(self):
        size = 0 if self.keys is None else super().nbytes
        return size + self.indexer_nbytes


class Qwen4ExpRMSNorm(nn.Module):
    """Qwen4 RMSNorm, whose checkpoint weights are centered at zero."""

    def __init__(self, dim: int, group_size: int | None = None, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.group_size = group_size
        if group_size is not None and dim % group_size:
            raise ValueError(f"{dim=} must be divisible by {group_size=}")
        self.weight = mx.zeros(dim)

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        y = x.astype(mx.float32)
        if self.group_size is not None:
            y = y.reshape(*y.shape[:-1], -1, self.group_size)
            weight = self.weight.reshape(-1, self.group_size)
        else:
            weight = self.weight
        y = y * mx.rsqrt(mx.mean(mx.square(y), axis=-1, keepdims=True) + self.eps)
        y = y * (1.0 + weight.astype(mx.float32))
        return y.reshape(x.shape).astype(dtype)


class Qwen4ExpRMSNormGated(nn.Module):
    def __init__(self, dim: int, eps: float, activation: str):
        super().__init__()
        self.eps = eps
        self.activation = activation
        self.weight = mx.ones(dim)

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        dtype = x.dtype
        y = mx.fast.rms_norm(x, self.weight, self.eps).astype(mx.float32)
        gate = gate.astype(mx.float32)
        if self.activation == "sigmoid":
            gate = mx.sigmoid(gate)
        else:
            gate = nn.silu(gate)
        return (y * gate).astype(dtype)


class Qwen4ExpGatedDeltaNet(Qwen3_5GatedDeltaNet):
    def __init__(self, config: TextConfig):
        super().__init__(config)
        self.norm = Qwen4ExpRMSNormGated(
            self.head_v_dim,
            eps=config.rms_norm_eps,
            activation=config.output_gate_type or config.hidden_act,
        )

    def _normalize_qk(self, q: mx.array, k: mx.array):
        # Transformers/FLA uses L2 normalization (epsilon after the sum),
        # followed by the usual 1/sqrt(head_dim) query scaling.
        scale = q.shape[-1] ** -0.5
        q = q * mx.rsqrt(mx.sum(mx.square(q), axis=-1, keepdims=True) + 1e-6)
        k = k * mx.rsqrt(mx.sum(mx.square(k), axis=-1, keepdims=True) + 1e-6)
        return q * scale, k


class Qwen4ExpQSAIndexer(nn.Module):
    """Select compressed key blocks using Qwen Sparse Attention scores."""

    def __init__(self, config: TextConfig, rotary_emb):
        super().__init__()
        self.n_heads = config.indexer_n_heads
        self.kv_heads = config.indexer_kv_heads
        self.head_dim = config.indexer_head_dim
        self.token_budget = config.indexer_budget
        self.compress_ratio = config.indexer_compress_ratio
        self.block_topk = self.token_budget // self.compress_ratio
        self.rotary_emb = rotary_emb
        self.index_qk_proj = nn.Linear(
            config.hidden_size,
            (self.n_heads + self.kv_heads) * self.head_dim,
            bias=False,
        )
        self.q_layernorm = Qwen4ExpRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_layernorm = Qwen4ExpRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    @staticmethod
    def _default_position_ids(batch: int, start: int, length: int):
        positions = mx.arange(start, start + length, dtype=mx.int32)
        return mx.broadcast_to(positions[None], (batch, length))

    def _apply_rope(self, x: mx.array, position_ids: mx.array) -> mx.array:
        # MRoPE's helper applies the same partial rotary transform to both
        # operands, so use a throwaway second operand for indexer-only states.
        rotated, _ = self.rotary_emb.apply_rotary(x, x, position_ids, unsqueeze_dim=1)
        return rotated

    def __call__(
        self,
        hidden_states: mx.array,
        cache: Optional[QSAKVCache],
        position_ids: Optional[mx.array],
        target_verify: bool = False,
    ) -> Optional[mx.array]:
        projected = _target_verify_linear(
            self.index_qk_proj, hidden_states, target_verify
        )
        return self.from_projected(projected, cache, position_ids)

    def from_projected(
        self,
        qk: mx.array,
        cache: Optional[QSAKVCache],
        position_ids: Optional[mx.array],
    ) -> Optional[mx.array]:
        batch, seq_len, _ = qk.shape
        past_len = cache.offset if cache is not None else 0
        if position_ids is None:
            position_ids = self._default_position_ids(batch, past_len, seq_len)

        qk = qk.reshape(batch, seq_len, self.n_heads + self.kv_heads, self.head_dim)
        query = qk[:, :, : self.n_heads]
        raw_keys = qk[:, :, self.n_heads :].squeeze(2)
        query = self.q_layernorm(query).transpose(0, 2, 1, 3)

        if cache is not None:
            raw_keys, full_position_ids = cache.update_indexer(raw_keys, position_ids)
        else:
            full_position_ids = position_ids

        key_len = raw_keys.shape[1]
        if isinstance(past_len, mx.array):
            # Batched rows carry per-row KV offsets, but the indexer cache is
            # left-padded to one width; the masks below need aligned-column
            # scalars or the (batch,) offsets broadcast into the seq axis.
            past_len = key_len - seq_len
        max_complete_blocks = key_len // self.compress_ratio
        if max_complete_blocks <= self.block_topk:
            return None

        query = self._apply_rope(query, position_ids)
        complete_key_len = max_complete_blocks * self.compress_ratio
        if cache is not None and hasattr(cache, "pooled_indexer_keys"):
            pooled_keys = cache.pooled_indexer_keys(
                self.compress_ratio,
                self.k_layernorm,
                self._apply_rope,
                cache_tag=self,
            )
        else:
            pooled_keys = pool_completed_index_keys(
                raw_keys,
                full_position_ids,
                compress_ratio=self.compress_ratio,
                index_key_norm=self.k_layernorm,
                apply_index_rope=self._apply_rope,
            )
        pooled_keys = mx.expand_dims(pooled_keys, axis=1)

        # Score in float32, as the reference does: which blocks win is a discrete
        # choice, and rounding the products flips the ones near the cut-off.
        scores = query.astype(mx.float32) @ pooled_keys.astype(mx.float32).transpose(
            0, 1, 3, 2
        )
        scores = mx.sum(mx.maximum(scores, 0), axis=1)
        scores = scores / math.sqrt(self.head_dim)

        query_ends = past_len + mx.arange(seq_len) + 1
        complete_counts = query_ends // self.compress_ratio
        valid_blocks = (
            mx.arange(max_complete_blocks)[None, None, :]
            < complete_counts[None, :, None]
        )
        scores = mx.where(valid_blocks, scores, -mx.inf)
        selected_blocks = mx.argpartition(scores, kth=-self.block_topk, axis=-1)[
            ..., -self.block_topk :
        ]

        # Mark the winners on the block axis and widen that to tokens. Comparing
        # every token against every pick costs seq_len * key_len * block_topk
        # bytes per prefill step -- 12 GB per sparse layer at a 12k prompt --
        # against seq_len * key_len here.
        block_hits = mx.put_along_axis(
            mx.zeros((batch, seq_len, max_complete_blocks), dtype=mx.bool_),
            selected_blocks,
            mx.array(True),
            axis=-1,
        )
        selected_tokens = mx.repeat(block_hits, self.compress_ratio, axis=-1)
        if complete_key_len < key_len:
            selected_tokens = mx.concatenate(
                [
                    selected_tokens,
                    mx.zeros(
                        (batch, seq_len, key_len - complete_key_len), dtype=mx.bool_
                    ),
                ],
                axis=-1,
            )

        token_indices = mx.arange(key_len)
        tail_starts = complete_counts * self.compress_ratio
        tail = (token_indices[None, None, :] >= tail_starts[None, :, None]) & (
            token_indices[None, None, :] < query_ends[None, :, None]
        )
        causal = token_indices[None, None, :] < query_ends[None, :, None]
        use_sparse = complete_counts > self.block_topk
        selected_tokens = mx.where(
            use_sparse[None, :, None], selected_tokens | tail, causal
        )
        return selected_tokens[:, None]


class Qwen4ExpAttention(Qwen3_5Attention):
    def __init__(self, config: TextConfig):
        super().__init__(config)
        self.q_norm = Qwen4ExpRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen4ExpRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.indexer = Qwen4ExpQSAIndexer(config, self.rotary_emb)

    @staticmethod
    def _batch_one_text_position_ids(
        position_ids: Optional[mx.array],
        length: int,
    ) -> bool:
        """Accept absent or shape-matched 2-D text positions, never MRoPE."""

        return position_ids is None or bool(
            isinstance(position_ids, mx.array)
            and position_ids.ndim == 2
            and position_ids.shape == (1, length)
        )

    def _gathered_text_prefill_eligible(
        self,
        x: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Any],
        position_ids: Optional[mx.array],
        position_embeddings: Optional[tuple[mx.array, mx.array]],
        target_verify: bool,
    ) -> bool:
        """Fail closed outside the proven contiguous batch-one text shape."""

        causal_mask = mask is None or (isinstance(mask, str) and mask == "causal")
        if not (
            x.ndim == 3
            and x.shape[0] == 1
            and x.shape[1] > 1
            and causal_mask
            and type(cache) is QSAKVCache
            and isinstance(cache.offset, int)
            and position_embeddings is None
            and not target_verify
            and self._batch_one_text_position_ids(position_ids, x.shape[1])
        ):
            return False
        return bool(
            # Below the QSA budget the official path attends the complete
            # prefix directly and is faster than building gathered blocks.
            # Switch only after sparse selection can reduce actual work.
            cache.offset + x.shape[1] > self.indexer.token_budget
        )

    def _gathered_text_decode_eligible(
        self,
        x: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Any],
        position_ids: Optional[mx.array],
        position_embeddings: Optional[tuple[mx.array, mx.array]],
        target_verify: bool,
    ) -> bool:
        """Fail closed outside scalar-offset batch-one text decode."""

        causal_mask = mask is None or (isinstance(mask, str) and mask == "causal")
        if not (
            x.ndim == 3
            and x.shape[:2] == (1, 1)
            and causal_mask
            and type(cache) is QSAKVCache
            and isinstance(cache.offset, int)
            and position_embeddings is None
            and not target_verify
            and self._batch_one_text_position_ids(position_ids, 1)
        ):
            return False

        # A restored/foreign cache without aligned auxiliary indexer state must
        # stay on the official path.  The gathered arm cannot safely discover
        # that mismatch after it has appended the new main K/V row.
        if cache.offset:
            if cache.index_keys is None or cache.index_position_ids is None:
                return False
            if (
                cache.index_keys.shape[1] != cache.offset
                or cache.index_position_ids.shape[-1] != cache.offset
            ):
                return False

        prospective_blocks = (cache.offset + 1) // self.indexer.compress_ratio
        return prospective_blocks > self.indexer.block_topk

    def _gathered_text_prefill(
        self,
        x: mx.array,
        cache: QSAKVCache,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        """Project once, append both caches, and attend only to selected K/V."""

        batch, length, _ = x.shape
        q_proj_output, keys, values = _target_verify_linears(
            (self.q_proj, self.k_proj, self.v_proj),
            x,
            False,
        )
        queries, gate = mx.split(
            q_proj_output.reshape(batch, length, self.num_attention_heads, -1),
            2,
            axis=-1,
        )
        gate = gate.reshape(batch, length, -1)
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(
            keys.reshape(batch, length, self.num_key_value_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        values = values.reshape(
            batch, length, self.num_key_value_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        past_len = cache.offset
        if position_ids is None:
            text_position_ids = mx.arange(
                past_len, past_len + length, dtype=mx.int32
            )[None]
            rotary_position_ids = mx.broadcast_to(
                text_position_ids,
                (3, batch, length),
            )
        else:
            text_position_ids = position_ids
            rotary_position_ids = position_ids
        queries, keys = self.rotary_emb.apply_rotary(
            queries,
            keys,
            rotary_position_ids,
            unsqueeze_dim=1,
        )
        keys, values = cache.update_and_fetch(keys, values)

        projected = self.indexer.index_qk_proj(x).reshape(
            batch,
            length,
            self.indexer.n_heads + self.indexer.kv_heads,
            self.indexer.head_dim,
        )
        index_queries = self.indexer.q_layernorm(
            projected[:, :, : self.indexer.n_heads]
        ).transpose(0, 2, 1, 3)
        raw_index_keys = projected[:, :, self.indexer.n_heads :].squeeze(2)
        raw_index_keys, full_position_ids = cache.update_indexer(
            raw_index_keys,
            text_position_ids,
        )
        pooled_index_keys = cache.pooled_indexer_keys(
            self.indexer.compress_ratio,
            self.indexer.k_layernorm,
            self.indexer._apply_rope,
            cache_tag=self.indexer,
        )
        index_queries = self.indexer._apply_rope(
            index_queries,
            text_position_ids,
        ).transpose(0, 2, 1, 3)

        output = contiguous_causal_gathered_qsa(
            queries,
            keys,
            values,
            index_queries,
            raw_index_keys,
            full_position_ids,
            num_query_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            indexer_head_dim=self.indexer.head_dim,
            compress_ratio=self.indexer.compress_ratio,
            token_budget=self.indexer.token_budget,
            index_key_norm=self.indexer.k_layernorm,
            apply_index_rope=self.indexer._apply_rope,
            pooled_index_keys=pooled_index_keys,
        )
        output = output.reshape(batch, length, -1)
        return self.o_proj(output * mx.sigmoid(gate))

    def _gathered_text_decode(
        self,
        x: mx.array,
        cache: QSAKVCache,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        """Append one token and attend only to QSA-selected cached K/V rows."""

        batch, length, _ = x.shape
        q_proj_output, new_keys, new_values = _target_verify_linears(
            (self.q_proj, self.k_proj, self.v_proj),
            x,
            False,
        )
        queries, gate = mx.split(
            q_proj_output.reshape(batch, length, self.num_attention_heads, -1),
            2,
            axis=-1,
        )
        gate = gate.reshape(batch, length, -1)
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        new_keys = self.k_norm(
            new_keys.reshape(
                batch,
                length,
                self.num_key_value_heads,
                self.head_dim,
            )
        ).transpose(0, 2, 1, 3)
        new_values = new_values.reshape(
            batch,
            length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(0, 2, 1, 3)

        past_len = cache.offset
        if position_ids is None:
            text_position_ids = mx.arange(
                past_len,
                past_len + 1,
                dtype=mx.int32,
            )[None]
            rotary_position_ids = mx.broadcast_to(
                text_position_ids,
                (3, batch, length),
            )
        else:
            text_position_ids = position_ids
            rotary_position_ids = position_ids
        queries, new_keys = self.rotary_emb.apply_rotary(
            queries,
            new_keys,
            rotary_position_ids,
            unsqueeze_dim=1,
        )
        keys, values = cache.update_and_fetch(new_keys, new_values)

        projected = self.indexer.index_qk_proj(x).reshape(
            batch,
            length,
            self.indexer.n_heads + self.indexer.kv_heads,
            self.indexer.head_dim,
        )
        index_queries = self.indexer.q_layernorm(
            projected[:, :, : self.indexer.n_heads]
        ).transpose(0, 2, 1, 3)
        raw_index_keys = projected[:, :, self.indexer.n_heads :].squeeze(2)
        cache.update_indexer(raw_index_keys, text_position_ids)
        pooled_index_keys = cache.pooled_indexer_keys(
            self.indexer.compress_ratio,
            self.indexer.k_layernorm,
            self.indexer._apply_rope,
            cache_tag=self.indexer,
        )
        index_queries = self.indexer._apply_rope(
            index_queries,
            text_position_ids,
        ).transpose(0, 2, 1, 3)

        output = contiguous_causal_gathered_qsa_decode(
            queries,
            keys,
            values,
            index_queries,
            pooled_index_keys,
            num_query_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            indexer_head_dim=self.indexer.head_dim,
            compress_ratio=self.indexer.compress_ratio,
            token_budget=self.indexer.token_budget,
        )
        output = output.reshape(batch, length, -1)
        return self.o_proj(output * mx.sigmoid(gate))

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_ids: Optional[mx.array] = None,
        position_embeddings: Optional[tuple[mx.array, mx.array]] = None,
        target_verify: bool = False,
    ) -> mx.array:
        if self._gathered_text_decode_eligible(
            x,
            mask,
            cache,
            position_ids,
            position_embeddings,
            target_verify,
        ):
            return self._gathered_text_decode(x, cache, position_ids)

        if self._gathered_text_prefill_eligible(
            x,
            mask,
            cache,
            position_ids,
            position_embeddings,
            target_verify,
        ):
            return self._gathered_text_prefill(x, cache, position_ids)

        qsa_mask = self.indexer(
            x,
            cache,
            position_ids,
            target_verify=target_verify,
        )
        if qsa_mask is not None:
            if mask is None or (isinstance(mask, str) and mask == "causal"):
                mask = qsa_mask
            elif isinstance(mask, mx.array):
                if mask.dtype == mx.bool_:
                    mask = mask & qsa_mask
                else:
                    sparse_bias = mx.where(qsa_mask, 0.0, -mx.inf).astype(mask.dtype)
                    mask = mask + sparse_bias
            # The specialized left-padded decode path remains dense. It is
            # uncommon, and preserving its row-specific cache semantics is
            # preferable to applying an incorrectly aligned sparse mask.
        return super().__call__(
            x,
            mask=mask,
            cache=cache,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            target_verify=target_verify,
        )


class Qwen4ExpGatedResidual(nn.Module):
    def __init__(self, config: TextConfig, use_combine: bool = True):
        super().__init__()
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.hc_lowrank = config.hc_lowrank
        hc_hidden_size = self.hc_count * self.hidden_size
        self.hc_norm = Qwen4ExpRMSNorm(
            hc_hidden_size,
            group_size=self.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.input_mix_weight_down = nn.Linear(
            hc_hidden_size, config.hc_lowrank, bias=False
        )
        self.input_mix_weight_up = nn.Linear(
            config.hc_lowrank, hc_hidden_size, bias=False
        )
        if use_combine:
            self.block_inject_weight = nn.Linear(
                hc_hidden_size, self.hc_count, bias=False
            )

    def __call__(self, hyper_input: mx.array, target_verify: bool = False):
        compiled_forward = getattr(self, "_compiled_forward", None)
        if (
            compiled_forward is not None
            and not target_verify
            and hyper_input.ndim == 3
            and hyper_input.shape[:2] == (1, 1)
            and hyper_input.dtype == mx.bfloat16
            and not get_mtp_runtime().enabled
        ):
            return compiled_forward(hyper_input)
        return self._forward(hyper_input, target_verify=target_verify)

    def _forward(self, hyper_input: mx.array, target_verify: bool = False):
        normed = self.hc_norm(hyper_input)
        hybrid = None
        if (
            getattr(self, "_omlx_exact_hybrid_projection", False)
            and not target_verify
            and hyper_input.shape == (1, 1, 10240)
            and hyper_input.dtype == mx.bfloat16
            and not get_mtp_runtime().enabled
        ):
            from .hc_projection import hybrid_projection

            hybrid = hybrid_projection(
                normed,
                self.input_mix_weight_down,
                self.block_inject_weight,
            )
        input_inject_weight = getattr(self, "input_inject_weight", None)
        if hybrid is not None:
            mix = hybrid[..., : self.hc_lowrank]
            block_injection = hybrid[
                ..., self.hc_lowrank : self.hc_lowrank + self.hc_count
            ]
        elif input_inject_weight is None:
            mix = _target_verify_linear(
                self.input_mix_weight_down, normed, target_verify
            )
            block_injection = (
                _target_verify_linear(
                    self.block_inject_weight, normed, target_verify
                )
                if "block_inject_weight" in self
                else None
            )
        else:
            combined = _target_verify_linear(
                input_inject_weight,
                normed,
                target_verify,
            )
            indices = _HYPER_SPLIT_INDICES.get((self.hc_lowrank, self.hc_count))
            if indices is None:
                indices = (
                    mx.arange(self.hc_lowrank, dtype=mx.int32),
                    mx.arange(
                        self.hc_lowrank,
                        self.hc_lowrank + self.hc_count,
                        dtype=mx.int32,
                    ),
                )
                _HYPER_SPLIT_INDICES[(self.hc_lowrank, self.hc_count)] = indices
            mix = mx.take(combined, indices[0], axis=-1)
            block_injection = mx.take(combined, indices[1], axis=-1)

        mix = nn.silu(mix / self.hc_count)
        mix = mx.sigmoid(
            _target_verify_linear(
                self.input_mix_weight_up,
                mix,
                target_verify,
            )
        )
        mix = mix.reshape(*mix.shape[:-1], self.hc_count, self.hidden_size)
        streams = normed.reshape(*normed.shape[:-1], self.hc_count, self.hidden_size)
        mixed_input = mx.mean(mix * streams, axis=-2)
        if block_injection is None:
            return mixed_input
        injection_weights = 2 * mx.sigmoid(
            block_injection / self.hc_count
        )
        return mixed_input, hyper_input, injection_weights


def _matching_projection_tensor(left: nn.Module, right: nn.Module, name: str) -> bool:
    left_has = hasattr(left, name)
    right_has = hasattr(right, name)
    if left_has != right_has:
        return False
    if not left_has:
        return True

    left_value = getattr(left, name)
    right_value = getattr(right, name)
    if left_value is None or right_value is None:
        return left_value is None and right_value is None
    return (
        isinstance(left_value, mx.array)
        and isinstance(right_value, mx.array)
        and left_value.ndim > 0
        and left_value.ndim == right_value.ndim
        and left_value.shape[1:] == right_value.shape[1:]
        and left_value.dtype == right_value.dtype
    )


def _can_fuse_hyper_connection(module: Qwen4ExpGatedResidual) -> bool:
    """Fail closed unless both projections have identical MLX semantics."""
    if type(module) is not Qwen4ExpGatedResidual:
        return False
    if (
        hasattr(module, "input_inject_weight")
        or hasattr(module, "_compiled_forward")
        or getattr(module, "_omlx_exact_hybrid_projection", False)
    ):
        return False

    down = getattr(module, "input_mix_weight_down", None)
    injection = getattr(module, "block_inject_weight", None)
    if down is None or injection is None or type(down) is not type(injection):
        return False
    if type(down) not in (nn.Linear, nn.QuantizedLinear):
        return False
    if not _matching_projection_tensor(down, injection, "weight"):
        return False
    if (
        down.weight.shape[0] != module.hc_lowrank
        or injection.weight.shape[0] != module.hc_count
    ):
        return False
    for attribute in ("group_size", "bits", "mode"):
        if getattr(down, attribute, None) != getattr(injection, attribute, None):
            return False
    return all(
        _matching_projection_tensor(down, injection, name)
        for name in ("scales", "biases", "bias")
    )


def _can_prepare_exact_hybrid(module: Qwen4ExpGatedResidual) -> bool:
    if not (
        module.hc_count == 4
        and module.hidden_size == 2560
        and module.hc_lowrank == 320
    ):
        return False
    from .hc_projection import compatible_projections

    return compatible_projections(
        module.input_mix_weight_down,
        module.block_inject_weight,
    )


def _unique_hyper_connections(model: nn.Module):
    modules = [model]
    modules.extend(module for _, module in model.named_modules() if module is not model)
    seen = set()
    for module in modules:
        if id(module) in seen:
            continue
        seen.add(id(module))
        if type(module) is Qwen4ExpGatedResidual:
            yield module


def fuse_hyper_connection_projections(model: nn.Module) -> int:
    """Prepare the lossless one-dispatch Qwen4 decode projection.

    The tempting 324-row concatenation was rejected because MLX changes its
    QMV traversal at that width and can change raw BF16 outputs.  The raw
    320-row low-rank and four-row injection banks deliberately remain separate
    so every fallback retains the checkpoint's canonical two projections.
    """
    targets = [
        module
        for module in _unique_hyper_connections(model)
        if _can_fuse_hyper_connection(module)
        and _can_prepare_exact_hybrid(module)
    ]
    for module in targets:
        module._omlx_exact_hybrid_projection = True
    return len(targets)


def compile_hyper_connections(model: nn.Module) -> int:
    """Compile each hyper-connection's strict single-token decode path once."""
    compiled = 0
    for module in _unique_hyper_connections(model):
        if hasattr(module, "_compiled_forward"):
            continue
        module._compiled_forward = mx.compile(module._forward)
        compiled += 1
    return compiled


_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _build_layer_multipliers(
    unigram_vocab_size: int, ngram_size: int, ple_layer_index: int, seed: int
):
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    multipliers = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        multipliers.append(2 * (_splitmix64(value) % half_bound) + 1)
    return mx.array(multipliers, dtype=mx.int64)


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _find_nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


class _SafeTensorMMap:
    """Read selected dense or affine-packed rows without resident weights."""

    def __init__(self, path: Path):
        self.path = path
        self._file = path.open("rb")
        header_size = struct.unpack("<Q", self._file.read(8))[0]
        self._header = json.loads(self._file.read(header_size))
        self._data_start = 8 + header_size
        self._mapping = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            self._mapping.madvise(mmap.MADV_RANDOM)
        except (AttributeError, OSError):
            pass

    def tensor_shape(self, key: str) -> tuple[int, ...]:
        return tuple(self._header[key]["shape"])

    def tensor_dtype(self, key: str) -> str:
        return str(self._header[key]["dtype"])

    def rows(self, key: str, rows: list[int]) -> mx.array:
        entry = self._header[key]
        shape = tuple(entry["shape"])
        start, end = entry["data_offsets"]
        dtype = entry["dtype"]
        dtype_info = {
            "BF16": (np.dtype("<u2"), 2),
            "F16": (np.dtype("<f2"), 2),
            "F32": (np.dtype("<f4"), 4),
            "U32": (np.dtype("<u4"), 4),
            "F8_E4M3": (np.dtype("u1"), 1),
        }.get(dtype)
        if dtype_info is None:
            raise TypeError(f"SSD-backed Qwen4 PLE does not support {dtype}")
        np_dtype, item_size = dtype_info
        if len(shape) != 2 or end - start != math.prod(shape) * item_size:
            raise ValueError(f"Invalid sparse PLE tensor layout for {key}")
        view = np.ndarray(
            shape,
            dtype=np_dtype,
            buffer=self._mapping,
            offset=self._data_start + start,
        )
        copied = np.array(view[np.asarray(rows, dtype=np.intp)], copy=True)
        if dtype == "BF16":
            values = (copied.astype(np.uint32) << np.uint32(16)).view(np.float32)
            return mx.array(values).astype(mx.bfloat16)
        if dtype == "F8_E4M3":
            return mx.from_fp8(mx.array(copied), dtype=mx.bfloat16)
        return mx.array(copied)

    def close(self):
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None
        if self._file is not None:
            self._file.close()
            self._file = None


class DiskBackedShardedEmbedding(nn.Module):
    """The 128-way dense or oQ-affine PLE table, gathered from SSD mmap."""

    def __init__(
        self,
        model_path: str | Path,
        prefix: str,
        num_embeddings: int,
        dims: int,
        num_shards: int,
    ):
        super().__init__()
        base, remainder = divmod(num_embeddings, num_shards)
        self.shard_sizes = tuple(
            base + (1 if index < remainder else 0) for index in range(num_shards)
        )
        offsets = [0]
        for size in self.shard_sizes:
            offsets.append(offsets[-1] + size)
        self.shard_offsets = tuple(offsets)
        self.dims = dims
        self.weight_scale = mx.ones((1,), dtype=mx.bfloat16)
        self._prefix = prefix
        self.rows_read = 0
        self.last_touched_shards: tuple[int, ...] = ()
        self._readers: dict[str, _SafeTensorMMap] = {}
        self._tensor_readers: dict[str, _SafeTensorMMap] = {}
        self._shard_specs: dict[
            int, tuple[str, str | None, str | None, int | None, int | None]
        ] = {}

        model_path = Path(model_path)
        index_path = model_path / "model.safetensors.index.json"
        weight_map = json.loads(index_path.read_text()).get("weight_map", {})

        def register_reader(key: str) -> _SafeTensorMMap:
            filename = weight_map[key]
            reader = self._readers.get(filename)
            if reader is None:
                reader = _SafeTensorMMap(model_path / filename)
                self._readers[filename] = reader
            self._tensor_readers[key] = reader
            return reader

        runtime_prefix = prefix
        if runtime_prefix.startswith("model.language_model."):
            runtime_prefix = (
                "language_model.model." + runtime_prefix[len("model.language_model.") :]
            )
        for shard_index, shard_size in enumerate(self.shard_sizes):
            bases = (
                f"{prefix}.shard_{shard_index}",
                f"{prefix}.shards.{shard_index}",
                f"{runtime_prefix}.shard_{shard_index}",
                f"{runtime_prefix}.shards.{shard_index}",
            )
            base = next(
                (
                    candidate
                    for candidate in bases
                    if f"{candidate}.weight" in weight_map
                ),
                None,
            )
            if base is None:
                raise KeyError(
                    f"SSD-backed PLE shard {shard_index} is absent; "
                    f"checked {', '.join(bases)}"
                )

            weight_key = f"{base}.weight"
            scales_key = f"{base}.scales"
            biases_key = f"{base}.biases"
            weight_reader = register_reader(weight_key)
            weight_shape = weight_reader.tensor_shape(weight_key)
            weight_dtype = weight_reader.tensor_dtype(weight_key)
            if len(weight_shape) != 2 or weight_shape[0] != shard_size:
                raise ValueError(f"Unexpected shape for {weight_key}: {weight_shape}")

            if scales_key not in weight_map and biases_key not in weight_map:
                if weight_shape[1] != dims:
                    raise ValueError(
                        f"Unexpected dense PLE width for {weight_key}: "
                        f"expected {dims}, got {weight_shape[1]}"
                    )
                if weight_dtype not in {"BF16", "F16", "F32", "F8_E4M3"}:
                    raise TypeError(
                        f"SSD-backed dense Qwen4 PLE does not support {weight_dtype}"
                    )
                self._shard_specs[shard_index] = (
                    weight_key,
                    None,
                    None,
                    None,
                    None,
                )
                continue

            if scales_key not in weight_map or biases_key not in weight_map:
                raise ValueError(
                    f"Incomplete affine PLE tensors for {base}: both scales and "
                    "biases are required"
                )
            if weight_dtype != "U32":
                raise TypeError(
                    f"Affine Qwen4 PLE weight must be U32, got {weight_dtype} "
                    f"for {weight_key}"
                )

            scales_reader = register_reader(scales_key)
            biases_reader = register_reader(biases_key)
            scales_shape = scales_reader.tensor_shape(scales_key)
            biases_shape = biases_reader.tensor_shape(biases_key)
            if scales_shape != biases_shape or len(scales_shape) != 2:
                raise ValueError(
                    f"Invalid affine PLE parameter shapes for {base}: "
                    f"scales={scales_shape}, biases={biases_shape}"
                )
            if scales_shape[0] != shard_size or scales_shape[1] <= 0:
                raise ValueError(
                    f"Unexpected affine PLE scale shape for {base}: {scales_shape}"
                )
            if dims % scales_shape[1] != 0:
                raise ValueError(
                    f"Cannot infer affine PLE group size for {base}: "
                    f"dims={dims}, scales={scales_shape}"
                )
            group_size = dims // scales_shape[1]
            packed_bits = weight_shape[1] * 32
            if packed_bits % dims != 0:
                raise ValueError(
                    f"Cannot infer affine PLE bits for {base}: "
                    f"dims={dims}, packed_shape={weight_shape}"
                )
            bits = packed_bits // dims
            if bits not in {2, 3, 4, 5, 6, 8} or group_size not in {32, 64, 128}:
                raise ValueError(
                    f"Unsupported affine PLE layout for {base}: "
                    f"bits={bits}, group_size={group_size}"
                )
            if dims % group_size or weight_shape[1] != dims * bits // 32:
                raise ValueError(
                    f"Inconsistent affine PLE layout for {base}: "
                    f"weight={weight_shape}, scales={scales_shape}, dims={dims}"
                )
            self._shard_specs[shard_index] = (
                weight_key,
                scales_key,
                biases_key,
                bits,
                group_size,
            )

    def __call__(self, indices: mx.array) -> mx.array:
        shape = indices.shape
        flat = indices.reshape(-1)
        mx.eval(flat)
        host_indices = [int(index) for index in flat.tolist()]
        if any(index < 0 or index >= self.shard_offsets[-1] for index in host_indices):
            raise IndexError("embedding index is outside the sharded vocabulary")
        shard_indices = [
            bisect_right(self.shard_offsets, index) - 1 for index in host_indices
        ]
        touched = tuple(sorted(set(shard_indices)))
        self.last_touched_shards = touched
        self.rows_read = 0
        result = mx.zeros((len(host_indices), self.dims), dtype=mx.bfloat16)
        for shard_index in touched:
            positions = [
                i
                for i, current_shard in enumerate(shard_indices)
                if current_shard == shard_index
            ]
            local = [
                host_indices[i] - self.shard_offsets[shard_index] for i in positions
            ]
            weight_key, scales_key, biases_key, bits, group_size = self._shard_specs[
                shard_index
            ]
            values = self._tensor_readers[weight_key].rows(weight_key, local)
            if bits is not None:
                assert scales_key is not None
                assert biases_key is not None
                assert group_size is not None
                scales = self._tensor_readers[scales_key].rows(scales_key, local)
                biases = self._tensor_readers[biases_key].rows(biases_key, local)
                values = mx.dequantize(
                    values,
                    scales,
                    biases,
                    group_size=group_size,
                    bits=bits,
                    mode="affine",
                )
            values = values.astype(mx.bfloat16) * self.weight_scale
            self.rows_read += len(local)
            result = result.at[mx.array(positions, dtype=mx.int32)].add(values)
        return result.reshape(*shape, self.dims)

    def close(self):
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()
        self._tensor_readers.clear()
        self._shard_specs.clear()

    @property
    def _prefix(self):
        return self.__prefix

    @_prefix.setter
    def _prefix(self, value):
        self.__prefix = value


class ShardedEmbedding(nn.Module):
    """Embedding kept in checkpoint-sized row shards to avoid a 100 GB join."""

    def __init__(self, num_embeddings: int, dims: int, num_shards: int):
        super().__init__()
        if num_shards <= 0 or num_shards > num_embeddings:
            raise ValueError("num_shards must be in [1, num_embeddings]")
        base, remainder = divmod(num_embeddings, num_shards)
        self.shard_sizes = tuple(
            base + (1 if index < remainder else 0) for index in range(num_shards)
        )
        self.shards = [nn.Embedding(size, dims) for size in self.shard_sizes]
        # Official FP8 checkpoints keep one shared scale for every PLE shard.
        # Decode only the selected rows so the 100 GB table stays compact.
        self.weight_scale = mx.ones((1,), dtype=mx.bfloat16)
        offsets = [0]
        for size in self.shard_sizes:
            offsets.append(offsets[-1] + size)
        self.shard_offsets = tuple(offsets)
        self.dims = dims

    def __call__(self, indices: mx.array) -> mx.array:
        fused = getattr(self, "fused", None)
        if fused is not None:
            return fused(indices) * self.weight_scale

        flat = indices.reshape(-1)
        # One tiny host sync avoids scheduling gathers against all 128 giant
        # PLE shards for every token.
        mx.eval(flat)
        host_indices = [int(index) for index in flat.tolist()]
        if not host_indices:
            return self.shards[0](flat).reshape(*indices.shape, self.dims)
        if any(index < 0 or index >= self.shard_offsets[-1] for index in host_indices):
            raise IndexError("embedding index is outside the sharded vocabulary")

        shard_indices = [
            bisect_right(self.shard_offsets, index) - 1 for index in host_indices
        ]
        result = None
        for shard_index in sorted(set(shard_indices)):
            positions_list = [
                position
                for position, current_shard in enumerate(shard_indices)
                if current_shard == shard_index
            ]
            local_indices = [
                host_indices[position] - self.shard_offsets[shard_index]
                for position in positions_list
            ]
            positions = mx.array(positions_list, dtype=mx.int32)
            values = self.shards[shard_index](mx.array(local_indices, dtype=mx.int32))
            if values.dtype == mx.uint8:
                values = mx.from_fp8(values, dtype=mx.bfloat16)
            values = values * self.weight_scale
            if result is None:
                result = mx.zeros((len(host_indices), self.dims), dtype=values.dtype)
            result = result.at[positions].add(values)
        return result.reshape(*indices.shape, self.dims)

    def fuse_quantized_shards(self) -> bool:
        """Join compatible packed shards without dequantizing the PLE table.

        Resident Qwen4 PLE otherwise synchronizes token IDs to the host before
        every lookup so it can choose among 128 enormous shard buffers.  A
        single packed embedding keeps exactly the same affine rows while making
        the lookup a normal device-side gather.  The caller owns the temporary
        peak-memory admission check required while old and joined buffers
        coexist.
        """

        if getattr(self, "fused", None) is not None:
            return False
        shards = list(self.shards)
        if not shards or not all(
            type(shard) is nn.QuantizedEmbedding for shard in shards
        ):
            return False
        first = shards[0]
        if not all(
            shard.dims == first.dims
            and shard.group_size == first.group_size
            and shard.bits == first.bits
            and shard.mode == first.mode
            and shard.weight.dtype == first.weight.dtype
            and shard.scales.dtype == first.scales.dtype
            and (shard.biases is None) == (first.biases is None)
            and (shard.biases is None or shard.biases.dtype == first.biases.dtype)
            for shard in shards
        ):
            return False

        total_rows = sum(int(shard.weight.shape[0]) for shard in shards)
        if total_rows != self.shard_offsets[-1] or first.dims != self.dims:
            return False

        fused = nn.QuantizedEmbedding(
            1,
            self.dims,
            group_size=first.group_size,
            bits=first.bits,
            mode=first.mode,
        )
        fused.weight = mx.concatenate([shard.weight for shard in shards], axis=0)
        fused.scales = mx.concatenate([shard.scales for shard in shards], axis=0)
        if first.biases is None:
            fused.biases = None
        else:
            fused.biases = mx.concatenate([shard.biases for shard in shards], axis=0)
        fused.num_embeddings = total_rows
        arrays = [fused.weight, fused.scales]
        if fused.biases is not None:
            arrays.append(fused.biases)
        mx.eval(*arrays)
        self.fused = fused
        self.shards = []
        return True


def fuse_resident_ple_embeddings(
    model: nn.Module,
    *,
    minimum_physical_memory: int = 192 * 1024**3,
) -> int:
    """Fuse resident affine PLE shards only where the temporary peak is safe."""

    if get_ple_runtime_mode() != "resident":
        return 0
    physical_memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    if physical_memory < minimum_physical_memory:
        return 0

    fused = 0
    language_model = getattr(model, "language_model", None)
    layers = getattr(getattr(language_model, "model", None), "layers", ())
    for layer in layers:
        ple = getattr(layer, "ple", None)
        embedding = getattr(
            getattr(ple, "ple_embedding", None),
            "ngram_embedding",
            None,
        )
        if type(embedding) is ShardedEmbedding and embedding.fuse_quantized_shards():
            fused += 1
    return fused


class Qwen4ExpNGramEmbedding(nn.Module):
    def __init__(
        self,
        config: TextConfig,
        embedding_dim: int,
        layer_idx: int,
        ple_layer_index: int,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.ngram_size = config.ngram_size
        self.context_len = self.ngram_size - 1
        self.heads_per_ngram = config.heads_per_ngram
        self.ngram_heads = self.context_len * self.heads_per_ngram
        self.ple_layer_index = ple_layer_index
        self.unigram_vocab_size = config.vocab_size
        self.seed = config.seed
        eos = config.eos_token_id
        self.eos_token_id = eos[0] if isinstance(eos, list) else eos

        head_vocab_sizes = []
        head_offsets = []
        total_vocab_size = 0
        for head_idx in range(self.ngram_heads):
            global_head_idx = ple_layer_index * self.ngram_heads + head_idx
            size = _find_nth_prime_after(
                config.ngram_vocab_size_base - 1, global_head_idx + 1
            )
            head_vocab_sizes.append(size)
            head_offsets.append(total_vocab_size)
            total_vocab_size += size

        self.layer_multipliers = _build_layer_multipliers(
            self.unigram_vocab_size,
            self.ngram_size,
            ple_layer_index,
            self.seed,
        )
        self.ngram_heads_vocab_sizes = mx.array(head_vocab_sizes, dtype=mx.int64)
        self.ngram_heads_offsets = mx.array(head_offsets, dtype=mx.int64)
        divisor = config.make_ngram_vocab_size_divisible_by
        padded_vocab_size = math.ceil(total_vocab_size / divisor) * divisor
        embedding_args = (
            padded_vocab_size,
            embedding_dim // self.ngram_heads,
            config.split_ngram_parts,
        )
        if _PLE_RUNTIME_MODE == "mmap":
            if _PLE_RUNTIME_MODEL_PATH is None:
                raise RuntimeError("SSD-backed PLE has no configured model path")
            prefix = (
                f"model.language_model.layers.{layer_idx}.ple.ple_embedding."
                "ngram_embedding"
            )
            self.ngram_embedding = DiskBackedShardedEmbedding(
                _PLE_RUNTIME_MODEL_PATH,
                prefix,
                *embedding_args,
            )
        else:
            self.ngram_embedding = ShardedEmbedding(*embedding_args)

    def _shift_right_ignore_eos(self, token_ids: mx.array, shift: int):
        if shift == 0:
            return token_ids
        batch, seq_len = token_ids.shape
        positions = mx.arange(seq_len, dtype=mx.int64)
        eos_positions = mx.where(token_ids == self.eos_token_id, positions, -1)
        previous_eos_inclusive = mx.cummax(eos_positions, axis=1)
        previous_eos = mx.concatenate(
            [mx.full((batch, 1), -1, dtype=mx.int64), previous_eos_inclusive[:, :-1]],
            axis=1,
        )
        segment_start = previous_eos + 1
        position_in_segment = positions[None] - segment_start
        source_positions = positions - shift
        gather_positions = mx.broadcast_to(
            mx.maximum(source_positions, 0)[None], (batch, seq_len)
        )
        shifted = mx.take_along_axis(token_ids, gather_positions, axis=1)
        valid = (position_in_segment >= shift) & (source_positions[None] >= 0)
        return mx.where(valid, shifted, self.eos_token_id)

    def _previous_context(
        self, input_ids: mx.array, cache: ArraysCache | None
    ) -> mx.array:
        batch = input_ids.shape[0]
        if cache is not None and cache[3] is not None:
            return cache[3]
        return mx.full(
            (batch, self.context_len), self.eos_token_id, dtype=mx.int64
        )

    def __call__(self, input_ids: mx.array, cache: Optional[ArraysCache]):
        input_ids = input_ids.astype(mx.int64)
        previous_context = self._previous_context(input_ids, cache)

        token_history = mx.concatenate([previous_context, input_ids], axis=-1)
        if cache is not None:
            cache[3] = mx.contiguous(token_history[:, -self.context_len :])

        shifted_tokens = [
            self._shift_right_ignore_eos(token_history, shift)
            for shift in range(self.ngram_size)
        ]
        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed_ids = shifted_tokens[0] * self.layer_multipliers[0]
            for position in range(1, ngram):
                mixed_ids = mx.bitwise_xor(
                    mixed_ids,
                    shifted_tokens[position] * self.layer_multipliers[position],
                )
            sizes = self.ngram_heads_vocab_sizes[start:end]
            offsets = self.ngram_heads_offsets[start:end]
            ngram_ids = mixed_ids[..., None] % sizes[None, None]
            blocks.append(ngram_ids + offsets[None, None])

        ngram_ids = mx.concatenate(blocks, axis=-1)[:, -input_ids.shape[1] :]
        embeddings = self.ngram_embedding(ngram_ids)
        return embeddings.reshape(*embeddings.shape[:-2], -1)


class Qwen4ExpPLELayer(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int, ple_layer_index: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count
        hc_hidden_size = self.hidden_size * self.hc_count
        self.ple_embedding = Qwen4ExpNGramEmbedding(
            config, config.ple_embed_dim, layer_idx, ple_layer_index
        )
        self.key_proj = nn.Linear(config.ple_embed_dim, hc_hidden_size, bias=False)
        self.value_proj = nn.Linear(config.ple_embed_dim, self.hidden_size, bias=False)
        self.norm_key = Qwen4ExpRMSNorm(
            hc_hidden_size,
            group_size=self.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.norm_query = Qwen4ExpRMSNorm(
            hc_hidden_size,
            group_size=self.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.norm_conv = Qwen4ExpRMSNorm(
            hc_hidden_size,
            group_size=self.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.conv_dilation = config.ngram_size
        self.short_conv_state_len = (
            config.ple_conv_kernel_size - 1
        ) * self.conv_dilation
        self.conv1d = nn.Conv1d(
            hc_hidden_size,
            hc_hidden_size,
            kernel_size=config.ple_conv_kernel_size,
            dilation=self.conv_dilation,
            groups=hc_hidden_size,
            bias=False,
        )

    def _short_conv(self, x: mx.array, cache: Optional[ArraysCache]):
        batch = x.shape[0]
        if cache is not None and cache[2] is not None:
            state = cache[2]
        else:
            state = mx.zeros(
                (batch, self.short_conv_state_len, x.shape[-1]), dtype=x.dtype
            )
        conv_input = mx.concatenate([state, x], axis=1)
        if cache is not None:
            cache[2] = mx.contiguous(conv_input[:, -self.short_conv_state_len :])
        return nn.silu(self.conv1d(conv_input)), state

    def __call__(
        self,
        hidden_states: mx.array,
        input_ids: mx.array,
        cache: Optional[ArraysCache],
        mask: Optional[mx.array],
        target_verify: bool = False,
    ):
        capture_speculative_state = target_verify and input_ids.shape[1] > 1
        if cache is not None:
            cache._qwen4_exp_ple_speculative_state = None
        history = None
        if capture_speculative_state and cache is not None:
            history = self.ple_embedding._previous_context(
                input_ids.astype(mx.int64), cache
            )
        embeddings = self.ple_embedding(input_ids, cache)
        keys = self.norm_key(
            _target_verify_linear(self.key_proj, embeddings, target_verify)
        ).reshape(*hidden_states.shape[:-1], self.hc_count, self.hidden_size)
        values = _target_verify_linear(self.value_proj, embeddings, target_verify)
        queries = self.norm_query(hidden_states).reshape(
            *hidden_states.shape[:-1], self.hc_count, self.hidden_size
        )
        gate = mx.sum(keys * queries, axis=-1, keepdims=True) / math.sqrt(
            self.hidden_size
        )
        gate = mx.sign(gate) * mx.sqrt(mx.maximum(mx.abs(gate), 1e-6))
        gated_values = mx.sigmoid(gate) * values[..., None, :]
        gated_values = gated_values.reshape(*hidden_states.shape)
        normed = self.norm_conv(gated_values)
        if mask is not None and isinstance(mask, mx.array) and mask.ndim == 2:
            gated_values = mx.where(mask[..., None], gated_values, 0)
            normed = mx.where(mask[..., None], normed, 0)
        conv_output, conv_state = self._short_conv(normed, cache)
        if history is not None and cache is not None:
            cache._qwen4_exp_ple_speculative_state = _PLESpeculativeState(
                history=history,
                input_ids=input_ids.astype(mx.int64),
                conv_state=conv_state,
                conv_inputs=normed,
            )
        return gated_values + conv_output


class Qwen4ExpDecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.is_linear = config.layer_types[layer_idx] == "linear_attention"
        if self.is_linear:
            self.linear_attn = Qwen4ExpGatedDeltaNet(config)
        else:
            self.self_attn = Qwen4ExpAttention(config)
        self.mlp = Qwen3_5MoeSparseMoeBlock(config)
        ple_index = (
            config.ple_layer_ids.index(layer_idx + 1)
            if layer_idx + 1 in config.ple_layer_ids
            else None
        )
        if ple_index is not None:
            self.ple = Qwen4ExpPLELayer(config, layer_idx, ple_index)
        self.attn_hyper_connection = Qwen4ExpGatedResidual(config)
        self.mlp_hyper_connection = Qwen4ExpGatedResidual(config)

    def __call__(
        self,
        hidden_states: mx.array,
        input_ids: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Any],
        position_ids: Optional[mx.array],
        gdn_sink=None,
        target_verify: bool = False,
    ):
        if "ple" in self:
            hidden_states = hidden_states + self.ple(
                hidden_states,
                input_ids,
                cache,
                mask,
                target_verify=target_verify,
            )

        mixed, hyper_input, injection_weights = self.attn_hyper_connection(
            hidden_states,
            target_verify=target_verify,
        )
        if self.is_linear:
            branch = self.linear_attn(
                mixed,
                mask=mask,
                cache=cache,
                gdn_sink=gdn_sink,
                target_verify=target_verify,
            )
        else:
            branch = self.self_attn(
                mixed,
                mask=mask,
                cache=cache,
                position_ids=position_ids,
                target_verify=target_verify,
            )
        injection = branch[..., None, :] * injection_weights[..., None]
        hidden_states = hyper_input + injection.reshape(*hyper_input.shape)

        mixed, hyper_input, injection_weights = self.mlp_hyper_connection(
            hidden_states,
            target_verify=target_verify,
        )
        branch = self.mlp(mixed, target_verify=target_verify)
        injection = branch[..., None, :] * injection_weights[..., None]
        return hyper_input + injection.reshape(*hyper_input.shape)


class Qwen4ExpModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.args = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Qwen4ExpDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.hyper_connection_mixer = Qwen4ExpGatedResidual(config, use_combine=False)
        self.ssm_idx = next(
            (i for i, layer in enumerate(self.layers) if layer.is_linear), 0
        )
        self.fa_idx = next(
            (i for i, layer in enumerate(self.layers) if not layer.is_linear), 0
        )

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        position_ids: Optional[mx.array] = None,
        capture_layer_ids=None,
        hidden_sink=None,
        gdn_sink=None,
        **kwargs,
    ):
        del kwargs
        hidden_states = (
            self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        )
        hidden_states = mx.tile(hidden_states, (1, 1, self.args.hc_count))
        if cache is None:
            cache = [None] * len(self.layers)

        fa_mask = _create_qwen3_5_attention_mask(hidden_states, cache[self.fa_idx])
        ssm_mask = _create_qwen3_5_ssm_mask(hidden_states, cache[self.ssm_idx])
        if mask is not None and isinstance(mask, mx.array) and mask.ndim == 2:
            ssm_mask = mask

        capture = set(capture_layer_ids or [])
        for index, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
            layer_mask = ssm_mask if layer.is_linear else fa_mask
            hidden_states = layer(
                hidden_states,
                inputs,
                mask=layer_mask,
                cache=layer_cache,
                position_ids=position_ids,
                gdn_sink=gdn_sink,
                target_verify=gdn_sink is not None,
            )
            if hidden_sink is not None and index in capture:
                hidden_sink.append(
                    self.hyper_connection_mixer(
                        hidden_states,
                        target_verify=gdn_sink is not None,
                    )
                )

        if inputs_embeds is None and gdn_sink is None:
            host_ref = getattr(self, "_omlx_mtp_prime_host", None)
            host = host_ref() if host_ref is not None else None
            if host is not None:
                from omlx.patches.mlx_lm_mtp import prompt_priming

                if prompt_priming.capture_eligible(host, cache):
                    prompt_priming.maybe_capture(
                        host,
                        inputs,
                        hidden_states,
                        cache,
                    )

        if hidden_sink is not None and capture_layer_ids == []:
            # Lightning MTP consumes all residual streams before the final
            # mixer. Ordinary layer captures retain their mixed representation.
            hidden_sink.append(hidden_states)

        return self.hyper_connection_mixer(
            hidden_states,
            target_verify=gdn_sink is not None,
        )


class Qwen4ExpMTPModule(nn.Module):
    """Embedded one-layer draft head for Qwen4 Lightning MTP."""

    def __init__(self, args: TextConfig):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.hc_count = args.hc_count
        hc_hidden_size = self.hc_count * self.hidden_size
        self.pre_fc_norm_embedding = Qwen4ExpRMSNorm(
            self.hidden_size,
            eps=args.rms_norm_eps,
        )
        self.pre_fc_norm_hidden = Qwen4ExpRMSNorm(
            hc_hidden_size,
            eps=args.rms_norm_eps,
        )
        self.fc_embedding = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
        )
        self.fc_hidden = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
        )

        layer_config = replace(
            args,
            num_hidden_layers=1,
            layer_types=["qwen_sparse_attention"],
            full_attention_interval=1,
            ple_layer_ids=[],
        )
        self.layers = [Qwen4ExpDecoderLayer(layer_config, layer_idx=0)]
        self.hyper_connection_mixer = Qwen4ExpGatedResidual(
            layer_config,
            use_combine=False,
        )

    def fuse_inputs(
        self,
        token_embeddings: mx.array,
        hidden_states: mx.array,
    ) -> mx.array:
        expected_width = self.hc_count * self.hidden_size
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.reshape(
                *hidden_states.shape[:-2],
                expected_width,
            )
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != expected_width:
            raise ValueError(
                "Qwen4 Lightning MTP expects hidden shape "
                "[batch, tokens, hc_count * hidden_size]."
            )

        projected_embedding = self.fc_embedding(
            self.pre_fc_norm_embedding(token_embeddings)
        )
        hidden_streams = self.pre_fc_norm_hidden(hidden_states).reshape(
            *hidden_states.shape[:-1],
            self.hc_count,
            self.hidden_size,
        )
        projected_hidden = self.fc_hidden(hidden_streams)
        return (projected_embedding[..., None, :] + projected_hidden).reshape(
            hidden_states.shape
        )

    def __call__(
        self,
        hidden_states: mx.array,
        next_token_ids: mx.array,
        embed_tokens,
        cache=None,
    ) -> tuple[mx.array, mx.array]:
        hidden_states = self.fuse_inputs(
            embed_tokens(next_token_ids),
            hidden_states,
        )
        if cache is None:
            cache = [None] * len(self.layers)
        mask = _create_qwen3_5_attention_mask(
            hidden_states,
            cache[0] if cache else None,
        )
        for layer, layer_cache in zip(self.layers, cache):
            hidden_states = layer(
                hidden_states,
                next_token_ids,
                mask=mask,
                cache=layer_cache,
                position_ids=None,
            )
        return self.hyper_connection_mixer(hidden_states), hidden_states


class LanguageModel(Qwen3_5LanguageModel):
    def __init__(self, args: TextConfig, config: ModelConfig = None):
        nn.Module.__init__(self)
        self.args = args
        self.config = config
        self.model_type = args.model_type
        self.model = Qwen4ExpModel(args)
        self.model._omlx_mtp_prime_host = weakref.ref(self)
        self._position_ids = None
        self._rope_deltas = None
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def bind_mtp_owner(self, owner) -> None:
        """Reference a root-level Lightning MTP head without double registration."""
        self._omlx_qwen4_mtp_owner = weakref.ref(owner)
        self._enable_mtp_decode_markers()

    def _enable_mtp_decode_markers(self) -> None:
        from omlx.patches.mlx_lm_mtp import get_mtp_depth

        self._omlx_mtp_decode_enabled = True
        self._omlx_mtp_chain = True
        self._omlx_mtp_depth = get_mtp_depth()
        self._omlx_mtp_head_prenorm = True

    def get_mtp_module(self):
        module = getattr(self, "mtp", None)
        if module is not None:
            return module
        owner_ref = getattr(self, "_omlx_qwen4_mtp_owner", None)
        owner = owner_ref() if owner_ref is not None else None
        return getattr(owner, "mtp", None) if owner is not None else None

    def __call__(self, inputs, inputs_embeds=None, mask=None, cache=None, **kwargs):
        return_hidden = bool(kwargs.get("return_hidden", False))
        mtp_capture = return_hidden and kwargs.get("capture_layer_ids") is None
        if mtp_capture:
            kwargs["capture_layer_ids"] = []
        output = super().__call__(inputs, inputs_embeds, mask, cache, **kwargs)
        if mtp_capture and output.hidden_states:
            output.hidden_states = [output.hidden_states[0]]
        return output

    def mtp_forward(
        self,
        hidden_states,
        next_token_ids,
        mtp_cache,
        return_hidden: bool = False,
        logits_keep: int = 0,
    ):
        mtp = self.get_mtp_module()
        if mtp is None:
            raise RuntimeError(
                "Qwen4 Lightning MTP forward called without an attached head."
            )
        mtp_output, hc_hidden = mtp(
            hidden_states,
            next_token_ids,
            self.model.embed_tokens,
            mtp_cache,
        )
        logits_source = mtp_output
        if logits_keep and logits_source.shape[1] > logits_keep:
            logits_source = logits_source[:, -logits_keep:, :]
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(logits_source)
        else:
            logits = self.lm_head(logits_source)
        if return_hidden:
            return logits, hc_hidden
        return logits

    def make_mtp_cache(self):
        mtp = self.get_mtp_module()
        return [QSAKVCache() for _ in mtp.layers] if mtp is not None else []

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.is_linear:
                caches.append(ArraysCache(size=4 if "ple" in layer else 2))
            else:
                caches.append(QSAKVCache())
        return caches

    @staticmethod
    def _normalize_accepted_counts(accepted):
        if isinstance(accepted, int):
            return [accepted]
        if isinstance(accepted, mx.array):
            return [int(value) for value in accepted.reshape(-1).tolist()]
        return [int(value) for value in accepted]

    @staticmethod
    def _discard_ple_snapshots(caches):
        for cache in caches:
            if getattr(cache, "_qwen4_exp_ple_speculative_state", None) is not None:
                cache._qwen4_exp_ple_speculative_state = None

    @staticmethod
    def _validate_ple_snapshot(snapshot, accepted_values):
        batch, window = snapshot.input_ids.shape
        if len(accepted_values) not in (1, batch):
            raise ValueError(
                "PLE speculative rollback accepted count does not match batch size"
            )
        if any(value < 0 or value >= window for value in accepted_values):
            raise ValueError(
                "PLE speculative rollback accepted count is outside the verify window"
            )

    @staticmethod
    def _restore_ple_state(cache, snapshot, accepted_values):
        batch = snapshot.input_ids.shape[0]
        values = accepted_values * batch if len(accepted_values) == 1 else accepted_values
        accepted_array = mx.array(values, dtype=mx.int32)
        retained = accepted_array + 1

        history_len = snapshot.history.shape[1]
        history = mx.concatenate([snapshot.history, snapshot.input_ids], axis=1)
        history_positions = retained[:, None] + mx.arange(
            history_len, dtype=mx.int32
        )[None, :]
        cache[3] = mx.contiguous(
            mx.take_along_axis(history, history_positions, axis=1)
        )

        state_len = snapshot.conv_state.shape[1]
        if state_len:
            conv_input = mx.concatenate(
                [snapshot.conv_state, snapshot.conv_inputs], axis=1
            )
            conv_positions = retained[:, None] + mx.arange(
                state_len, dtype=mx.int32
            )[None, :]
            conv_positions = mx.broadcast_to(
                conv_positions[..., None],
                (batch, state_len, snapshot.conv_state.shape[-1]),
            )
            cache[2] = mx.contiguous(
                mx.take_along_axis(conv_input, conv_positions, axis=1)
            )

    def rollback_speculative_cache(self, caches, gdn_states, accepted, block_size):
        """Restore PLE state to the accepted prefix after inherited rollback."""
        accepted_values = self._normalize_accepted_counts(accepted)

        pending = []
        try:
            for cache in caches:
                snapshot = getattr(cache, "_qwen4_exp_ple_speculative_state", None)
                if snapshot is None:
                    continue
                self._validate_ple_snapshot(snapshot, accepted_values)
                pending.append((cache, snapshot))
            result = super().rollback_speculative_cache(
                caches, gdn_states, accepted, block_size
            )
            for cache, snapshot in pending:
                self._restore_ple_state(cache, snapshot, accepted_values)
            return result
        finally:
            self._discard_ple_snapshots(caches)
