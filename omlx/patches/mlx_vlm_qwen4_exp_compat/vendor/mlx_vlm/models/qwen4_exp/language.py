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
from types import SimpleNamespace
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
)
from ..qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock
from .config import ModelConfig, TextConfig

_PLE_RUNTIME_MODEL_PATH: Path | None = None
_PLE_RUNTIME_MODE = "resident"


@dataclass(frozen=True)
class Qwen4ExpMTPRuntime:
    """Lightning MTP construction decision for the next model load."""

    enabled: bool = False
    checkpoint_prefix: str | None = None


_MTP_RUNTIME = Qwen4ExpMTPRuntime()


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
    """Bind the external PLE artifact before Qwen4 model construction."""
    global _PLE_RUNTIME_MODEL_PATH, _PLE_RUNTIME_MODE

    compute_path = Path(model_path).expanduser().resolve()
    ple_path = compute_path
    artifact = {}
    config_path = compute_path / "config.json"
    if config_path.is_file():
        artifact = json.loads(config_path.read_text()).get("qwen4_exp_artifact") or {}
        relative_ple = artifact.get("ple_artifact")
        if relative_ple is not None:
            relative_ple = Path(relative_ple)
            if relative_ple.is_absolute():
                raise ValueError("Qwen4-Exp PLE artifact path must be relative")
            ple_path = (compute_path / relative_ple).resolve()
            artifact_root = compute_path.parent.resolve()
            if ple_path != artifact_root and artifact_root not in ple_path.parents:
                raise ValueError("Qwen4-Exp PLE artifact escapes its artifact root")

    requested = mode or os.environ.get("OMLX_QWEN4_PLE_MODE")
    if requested is None:
        requested = artifact.get("ple_residency", "auto")
    checkpoint_bytes = sum(
        path.stat().st_size
        for root in {compute_path, ple_path}
        for path in root.glob("*.safetensors")
    )
    physical_memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    _PLE_RUNTIME_MODE = resolve_ple_runtime_mode(
        requested,
        checkpoint_bytes=checkpoint_bytes,
        physical_memory=physical_memory,
    )
    _PLE_RUNTIME_MODEL_PATH = ple_path
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


class QSAKVCache(KVCache):
    """KV cache with the raw indexer keys and multimodal positions used by QSA."""

    # Hybrid/TurboQuant caches do not currently expose a way to carry the
    # indexer's unprojected keys. Uniform quantization uses the specialized
    # QSAQuantizedKVCache below; other schemes leave this cache in float.
    preserve_auxiliary_kv_state = True

    def __init__(self):
        super().__init__()
        self.index_keys = None
        self.index_position_ids = None

    def update_indexer(self, keys: mx.array, position_ids: mx.array):
        if self.index_keys is None:
            self.index_keys = keys
            self.index_position_ids = position_ids
        else:
            self.index_keys = mx.concatenate([self.index_keys, keys], axis=1)
            self.index_position_ids = _append_indexer_positions(
                self.index_position_ids, position_ids
            )
        return self.index_keys, self.index_position_ids

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
        self.keys, self.values, self.index_keys, self.index_position_ids = value
        self.offset = 0 if self.keys is None else self.keys.shape[2]

    def trim(self, n):
        n = min(self.offset, n)
        super().trim(n)
        if self.index_keys is not None:
            self.index_keys = self.index_keys[:, : self.offset]
            self.index_position_ids = self.index_position_ids[..., : self.offset]
        return n

    def extract(self, idx):
        cache = QSAKVCache()
        if self.keys is not None:
            cache.keys = mx.contiguous(self.keys[idx : idx + 1])
            cache.values = mx.contiguous(self.values[idx : idx + 1])
            cache.offset = self.offset
        if self.index_keys is not None:
            cache.index_keys = mx.contiguous(self.index_keys[idx : idx + 1])
            if self.index_position_ids.ndim == 3:
                cache.index_position_ids = mx.contiguous(
                    self.index_position_ids[:, idx : idx + 1]
                )
            else:
                cache.index_position_ids = mx.contiguous(
                    self.index_position_ids[idx : idx + 1]
                )
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
        size = super().nbytes
        if self.index_keys is not None:
            size += self.index_keys.nbytes + self.index_position_ids.nbytes
        return size


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
        length = 0 if cache.index_keys is None else cache.index_offset
        left = target - length
        if cache.index_keys is None:
            keys = mx.zeros(
                (cache.offset.shape[0], 0, sample_keys.shape[-1]),
                dtype=sample_keys.dtype,
            )
            if sample_positions.ndim == 3:
                positions = mx.zeros(
                    (sample_positions.shape[0], cache.offset.shape[0], 0),
                    dtype=sample_positions.dtype,
                )
            else:
                positions = mx.zeros(
                    (cache.offset.shape[0], 0), dtype=sample_positions.dtype
                )
        else:
            keys = cache.index_keys[:, :length]
            positions = cache.index_position_ids[..., :length]
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
        self.kv_cache.extend(other.kv_cache)
        sample_keys = (
            self.index_keys if self.index_keys is not None else other.index_keys
        )
        sample_positions = (
            self.index_position_ids
            if self.index_position_ids is not None
            else other.index_position_ids
        )
        if sample_keys is None:
            return
        target = max(self.index_offset, other.index_offset)
        left = self._pad_index(self, target, sample_keys, sample_positions)
        right = self._pad_index(other, target, sample_keys, sample_positions)
        self.index_keys = mx.concatenate([left[0], right[0]], axis=0)
        position_axis = 1 if sample_positions.ndim == 3 else 0
        self.index_position_ids = mx.concatenate(
            [left[1], right[1]], axis=position_axis
        )
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
        caches = list(caches)
        out = cls([0] * len(caches))
        if not caches:
            return out
        out.kv_cache = BatchKVCache.merge(caches)
        sample = next((cache for cache in caches if cache.index_keys is not None), None)
        if sample is None:
            return out
        target = max(cache.offset for cache in caches)
        rows = [
            cls._pad_index(
                SimpleNamespace(
                    index_keys=cache.index_keys,
                    index_position_ids=cache.index_position_ids,
                    index_offset=cache.offset,
                    offset=mx.array([cache.offset]),
                ),
                target,
                sample.index_keys,
                sample.index_position_ids,
            )
            for cache in caches
        ]
        out.index_keys = mx.concatenate([row[0] for row in rows], axis=0)
        position_axis = 1 if sample.index_position_ids.ndim == 3 else 0
        out.index_position_ids = mx.concatenate(
            [row[1] for row in rows], axis=position_axis
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


class QSAQuantizedKVCache(QuantizedKVCache):
    """Uniformly quantized QSA cache that retains float indexer state."""

    preserve_auxiliary_kv_state = True

    def __init__(self, group_size: int = 64, bits: int = 8):
        super().__init__(group_size=group_size, bits=bits)
        self.index_keys = None
        self.index_position_ids = None

    def update_indexer(self, keys: mx.array, position_ids: mx.array):
        if self.index_keys is None:
            self.index_keys = keys
            self.index_position_ids = position_ids
        else:
            self.index_keys = mx.concatenate([self.index_keys, keys], axis=1)
            self.index_position_ids = _append_indexer_positions(
                self.index_position_ids, position_ids
            )
        return self.index_keys, self.index_position_ids

    @property
    def state(self):
        if self.keys is None:
            keys, values = None, None
        else:
            keys, values = super().state
        return keys, values, self.index_keys, self.index_position_ids

    @state.setter
    def state(self, value):
        self.keys, self.values, self.index_keys, self.index_position_ids = value
        self.offset = 0 if self.keys is None else self.keys[0].shape[2]

    def trim(self, n):
        n = min(self.offset, n)
        super().trim(n)
        if self.index_keys is not None:
            self.index_keys = self.index_keys[:, : self.offset]
            self.index_position_ids = self.index_position_ids[..., : self.offset]
        return n

    def extract(self, idx):
        cache = QSAQuantizedKVCache(self.group_size, self.bits)
        if self.keys is not None:
            cache.keys = tuple(mx.contiguous(x[idx : idx + 1]) for x in self.keys)
            cache.values = tuple(mx.contiguous(x[idx : idx + 1]) for x in self.values)
            cache.offset = self.offset
        if self.index_keys is not None:
            cache.index_keys = mx.contiguous(self.index_keys[idx : idx + 1])
            if self.index_position_ids.ndim == 3:
                cache.index_position_ids = mx.contiguous(
                    self.index_position_ids[:, idx : idx + 1]
                )
            else:
                cache.index_position_ids = mx.contiguous(
                    self.index_position_ids[idx : idx + 1]
                )
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
        if self.index_keys is not None:
            size += self.index_keys.nbytes + self.index_position_ids.nbytes
        return size


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
        max_complete_blocks = key_len // self.compress_ratio
        if max_complete_blocks <= self.block_topk:
            return None

        query = self._apply_rope(query, position_ids)
        complete_key_len = max_complete_blocks * self.compress_ratio
        pooled_keys = raw_keys[:, :complete_key_len].reshape(
            batch, max_complete_blocks, self.compress_ratio, self.head_dim
        )
        pooled_keys = mx.expand_dims(
            self.k_layernorm(
                mx.mean(pooled_keys.astype(mx.float32), axis=2).astype(raw_keys.dtype)
            ),
            axis=1,
        )

        block_starts = mx.arange(max_complete_blocks) * self.compress_ratio
        block_position_ids = full_position_ids[..., block_starts]
        pooled_keys = self._apply_rope(pooled_keys, block_position_ids)

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

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_ids: Optional[mx.array] = None,
        position_embeddings: Optional[tuple[mx.array, mx.array]] = None,
        target_verify: bool = False,
    ) -> mx.array:
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
        normed = self.hc_norm(hyper_input)
        mix = nn.silu(
            _target_verify_linear(
                self.input_mix_weight_down,
                normed,
                target_verify,
            )
            / self.hc_count
        )
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
        if "block_inject_weight" not in self:
            return mixed_input
        injection_weights = 2 * mx.sigmoid(
            _target_verify_linear(
                self.block_inject_weight,
                normed,
                target_verify,
            )
            / self.hc_count
        )
        return mixed_input, hyper_input, injection_weights


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

    def __call__(self, input_ids: mx.array, cache: Optional[ArraysCache]):
        input_ids = input_ids.astype(mx.int64)
        batch = input_ids.shape[0]
        if cache is not None and cache[3] is not None:
            previous_context = cache[3]
        else:
            previous_context = mx.full(
                (batch, self.context_len), self.eos_token_id, dtype=mx.int64
            )

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
        return nn.silu(self.conv1d(conv_input))

    def __call__(
        self,
        hidden_states: mx.array,
        input_ids: mx.array,
        cache: Optional[ArraysCache],
        mask: Optional[mx.array],
        target_verify: bool = False,
    ):
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
        return gated_values + self._short_conv(normed, cache)


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
