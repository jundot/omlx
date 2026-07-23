# SPDX-License-Identifier: Apache-2.0
"""
Paged SSD Cache Manager for oMLX KV cache.

This module implements SSD-based storage for paged KV cache blocks,
enabling larger effective cache sizes than GPU memory allows.

Key features:
- Block-level safetensors serialization (compatible with mlx-lm)
- Hash-based subdirectory structure for scalability
- LRU-based paged SSD cache size management
- Startup scan to reuse existing cache files

Reference: mlx-lm/mlx_lm/models/cache.py (save_prompt_cache, load_prompt_cache)
"""

from __future__ import annotations

import contextlib
import errno
import json
import logging
import os
import queue
import shutil
import struct
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from omlx.utils.formatting import format_bytes

from .interface import CacheManager
from .stats import PagedSSDCacheStats

logger = logging.getLogger(__name__)

# Check for MLX
try:
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None


# --- Async I/O constants ---
# Fraction of host RAM the pending-write queue targets at saturation.
# The queue holds raw-byte copies of KV blocks that the background
# writer hasn't drained yet (see ``_extract_tensor_bytes`` in
# ``save_block``). The hard fraction below bounds the soft floor so
# large-block workloads cannot silently reserve an unsafe amount of RAM.
_PENDING_WRITES_TARGET_RAM_FRACTION = 0.10
_PENDING_WRITES_HARD_RAM_FRACTION = 0.30
_PENDING_WRITES_SOFT_FLOOR = 32
_PENDING_WRITES_CEILING = 256
_PENDING_WRITE_PUT_TIMEOUT_SECONDS = 1.0

# Conservative defaults for the per-block cost estimator. The actual
# bytes-per-block depends on the model (num_layers × num_kv_heads ×
# head_dim × dtype_size × block_size_tokens × 2). At construction time
# the PagedSSDCacheManager doesn't always know these — see __init__'s
# ``expected_kv_bytes_per_token`` parameter — so the module-level
# default targets a 35B-class bf16 model whose per-token KV is ≈200 KB
# spread across all layers. Smaller models will be over-conservative
# (fine), larger models or larger blocks should pass an explicit value.
_DEFAULT_BLOCK_SIZE_TOKENS = 256
_DEFAULT_KV_BYTES_PER_TOKEN = 200_000


def _compute_max_pending_writes(
    block_size_tokens: int = _DEFAULT_BLOCK_SIZE_TOKENS,
    kv_bytes_per_token: int = _DEFAULT_KV_BYTES_PER_TOKEN,
    target_fraction: float = _PENDING_WRITES_TARGET_RAM_FRACTION,
    hard_fraction: float = _PENDING_WRITES_HARD_RAM_FRACTION,
) -> int:
    """Compute max pending writes queue depth.

    Scales by *block bytes* so the target pending pool stays near
    ``target_fraction`` of host RAM regardless of how big each block is,
    while ``hard_fraction`` bounds the soft floor:

        worst_case_bytes = cap × block_size_tokens × kv_bytes_per_token
        cap = (total_ram × target_fraction) / (block_size × kv_bytes_per_token)

    Bounded by a soft floor, a byte hard cap, and a ceiling:
      - Soft floor at 32 so even small systems with large blocks retain
        burst headroom for a few in-flight writes — dropping to zero
        means every save serializes against the disk and the writer
        thread becomes a hard bottleneck on the inference loop.
      - Hard cap at 30% of host RAM so the soft floor cannot turn very
        large blocks into an unsafe memory reservation.
      - Ceiling at 256 so 512 GB+ systems don't pin gigabytes against
        a writer that's already keeping up at lower caps.

    Workload sizing: long-context coding workloads snapshot ~73 blocks
    per turn at 150 K tokens (block_size=2048), and at the default
    block size of 256 tokens that's ~586 blocks per snapshot. The
    queue is a *burst ceiling*, not steady state — a healthy writer
    drains it continuously and the cap only matters when the writer
    is fighting memory pressure or slow disk. Sustained saturation now
    falls back to inline writes, so the cap controls when save latency
    moves onto the request path.

    Defaults target a 35B-class bf16 model at the default
    ``paged_cache_block_size=256``; pass an explicit
    ``kv_bytes_per_token`` for larger models or quantized configs.
    """
    try:
        total_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        block_bytes = max(1, block_size_tokens * kv_bytes_per_token)
        target = int(total_bytes * target_fraction / block_bytes)
        hard_cap = max(1, int(total_bytes * hard_fraction / block_bytes))
        soft_target = max(_PENDING_WRITES_SOFT_FLOOR, target)
        return max(1, min(_PENDING_WRITES_CEILING, soft_target, hard_cap))
    except (ValueError, OSError):
        return 64  # Safe default


# Module-level constant for static callers that don't have model
# config. The PagedSSDCacheManager recomputes per-instance from its
# ``__init__`` parameters so a non-default block size or model
# generation can plumb through.
_MAX_PENDING_WRITES = _compute_max_pending_writes()

# Cap on the number of LRU blocks ``_enforce_size_limit_for_new_block`` is
# allowed to unlink in one inline burst. Eviction normally returns ~1
# entry; the cap exists for the ENOSPC-recovery path where the disk-usage
# cache invalidates and the next ``_get_effective_max_size`` call can
# shrink sharply — ``evict_until_size`` would then return hundreds of
# entries at once and stall the inference thread on a syscall storm.
# Deferred-but-not-unlinked entries are reinserted into the index so
# subsequent saves drain the remainder; bounds per-call latency at the
# cost of taking multiple saves to fully reconverge.
_MAX_INLINE_UNLINKS_PER_SAVE = 32


# Cache format version. Bump when on-disk layout or RotatingKVCache meta_state
# semantics change in a way that older blocks become unsafe to load.
#
# Version "2": added with the mlx-lm 0.31.3 contract fix (issues #934 / #903).
# Version "1" / unset: pre-fix blocks. RotatingKVCache layers may have been
#   zero-padded to max_size, which after the fix would leak zero positions
#   into attention. Treat such blocks as a cache miss instead of migrating.
_CACHE_FORMAT_VERSION = "3"

# Versions whose blocks the current code can read. V3 polyfills V2 blocks
# whose layer data was stored as the legacy 2-tuple `(keys, values)` —
# they are upgraded to N-tuple markers on read so the rest of omlx core
# sees a uniform shape. New writes always use V3.
_READABLE_CACHE_FORMAT_VERSIONS = frozenset({"2", "3"})


# Layer cache type names whose meta_state should be clamped on save so the
# rotating buffer's _idx never exceeds the actual buffer length. Restoring a
# cache where _idx > keys.shape[2] makes BatchRotatingKVCache.merge() either
# overshoot the RHS or (when omlx pads) leak zero positions into attention.
_ROTATING_CACHE_TYPES = (
    "RotatingKVCache",
    "BatchRotatingKVCache",
    "BufferedRotatingKVCache",
)

_STORAGE_CLASS_NAME_ALIASES = {
    # mlx-vlm MTP uses this transient target-cache wrapper for rollback
    # slack. Persist the canonical rotating-cache type so blocks remain
    # compatible with the model.make_cache() signature.
    "BufferedRotatingKVCache": "RotatingKVCache",
}


def _storage_layer_cache_types(
    layer_cache_types: list[str] | tuple[str, ...] | None,
) -> list[str] | None:
    """Return type names to persist in new cache block metadata."""
    if layer_cache_types is None:
        return None
    return [
        _STORAGE_CLASS_NAME_ALIASES.get(cache_type, cache_type)
        for cache_type in layer_cache_types
    ]


def _canonicalize_layer_cache_types(
    layer_cache_types: list[str] | tuple[str, ...] | None,
) -> list[str] | None:
    """Normalize wrapper class names for metadata compatibility checks.

    Wrapper classes that keep the same tensor representation compare equal.
    Types that change tensor representation (e.g., ``TurboQuantKVCache`` vs
    ``KVCache``) are NOT collapsed -- that mismatch is real and the block must
    be invalidated.
    """
    if layer_cache_types is None:
        return None
    wrapper_to_canonical = {
        "SizedArraysCache": "ArraysCache",
        "PrefillReadyRotatingKVCache": "RotatingKVCache",
        # Batch and single-request TurboQuant caches persist the same packed
        # per-request state (the save path records whichever class name it
        # extracted; the restore path rebuilds a TurboQuantKVCache from
        # either). Collapsing them keeps the predicted layout from
        # refresh_ssd_layer_signature — which always says
        # "TurboQuantKVCache" — from sweeping valid batch-form blocks.
        "BatchTurboQuantKVCache": "TurboQuantKVCache",
    }
    return [
        wrapper_to_canonical.get(cache_type, cache_type)
        for cache_type in layer_cache_types
    ]


def _cache_compat_signature(
    *,
    model_name: str = "",
    num_layers: int = 0,
    block_size: int = 0,
    layer_cache_types: list[str] | None = None,
    turboquant_kv_bits: float | None = None,
) -> str:
    """Return a stable compatibility signature for a persisted cache block."""
    payload = {
        "model_name": model_name or "",
        "num_layers": int(num_layers or 0),
        "block_size": int(block_size or 0),
        "layer_cache_types": list(layer_cache_types or []),
    }
    # TurboQuant packed state width depends on the bit depth
    # (packed_width = ceil(head_dim * bits / 32)), so blocks written at
    # different bit depths are shape-incompatible (#2045). Only stamped
    # when TurboQuant is active so non-TurboQuant signatures stay
    # byte-identical to the previous format.
    if turboquant_kv_bits is not None:
        payload["turboquant_kv_bits"] = float(turboquant_kv_bits)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _signature_turboquant_bits(cache_signature: str) -> float | None:
    """Extract ``turboquant_kv_bits`` from a stored signature, or None."""
    if not cache_signature:
        return None
    try:
        payload = json.loads(cache_signature)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        # Corrupted/foreign signature that parses as a JSON scalar or list.
        # Report "no recorded depth" instead of raising: an AttributeError
        # here would abort the whole stale-signature sweep.
        return None
    bits = payload.get("turboquant_kv_bits")
    if bits is None:
        return None
    try:
        return float(bits)
    except (TypeError, ValueError):
        return None


def _block_turboquant_bits(
    layer_cache_types: list[str] | None,
    layer_meta_states: list[tuple] | None,
) -> float | None:
    """Read the bit depth a block's own TurboQuant layers were packed at.

    TurboQuant meta_state is ``(offset, bits, seed, ...)`` — the same tuple
    the restore path reads back at reconstruction. Deriving the signature
    stamp from the block itself keeps it truthful even when the manager's
    expectation is stale or not yet learned.
    """
    if not layer_cache_types or not layer_meta_states:
        return None
    for i, cache_type in enumerate(layer_cache_types):
        if cache_type not in ("TurboQuantKVCache", "BatchTurboQuantKVCache"):
            continue
        if i >= len(layer_meta_states):
            continue
        meta_state = layer_meta_states[i]
        if isinstance(meta_state, (list, tuple)) and len(meta_state) >= 3:
            try:
                return float(meta_state[1])
            except (TypeError, ValueError):
                return None
    return None


def _clamp_rotating_meta_states(
    cache_data: list[Any],
    layer_cache_types: list[str] | None,
    layer_meta_states: list[tuple] | None,
) -> list[tuple] | None:
    """Clamp ``_idx`` to ``keys.shape[2]`` for RotatingKVCache layers.

    RotatingKVCache.meta_state is ``(keep, max_size, offset, _idx)``. When
    we save a snapshot, ``_idx`` must reflect the actual buffer length so
    the restored cache lands in case 1 of ``_temporal_order``. Older code
    paths could leave ``_idx == max_size`` after zero-padding the buffer;
    by clamping at write time we ensure newer blocks are always safe to
    restore.
    """
    if not layer_meta_states or not layer_cache_types:
        return layer_meta_states

    clamped: list[tuple] = []
    for i, meta in enumerate(layer_meta_states):
        if (
            i < len(layer_cache_types)
            and layer_cache_types[i] in _ROTATING_CACHE_TYPES
            and meta
            and len(meta) >= 4
            and i < len(cache_data)
        ):
            layer_data = cache_data[i]
            seq_len: int | None = None
            if (
                isinstance(layer_data, tuple)
                and len(layer_data) == 2
                and not (
                    isinstance(layer_data[0], str) and layer_data[0].startswith("__")
                )
            ):
                keys = layer_data[0]
                if hasattr(keys, "shape") and len(keys.shape) >= 3:
                    seq_len = int(keys.shape[2])
            if seq_len is not None:
                try:
                    keep, max_size, offset, idx = meta[:4]
                    idx_int = int(idx)
                    if idx_int > seq_len:
                        clamped.append((keep, max_size, offset, str(seq_len)))
                        continue
                except (TypeError, ValueError):
                    pass
        clamped.append(meta)
    return clamped


def _has_zero_dim(tensor: Any) -> bool:
    """Check if a tensor has any zero-dimension axis (unsupported by safetensors)."""
    return hasattr(tensor, "shape") and any(d == 0 for d in tensor.shape)


def _encode_shape(shape) -> str:
    """Encode tensor shape as comma-separated string for safetensors metadata."""
    return ",".join(str(d) for d in shape)


def _decode_shape(shape_str: str) -> tuple:
    """Decode shape string back to tuple of ints."""
    return tuple(int(d) for d in shape_str.split(","))


def _store_nstate_elements_flat(
    arrays: dict[str, Any],
    cache_list_meta: dict[str, str],
    prefix: str,
    elements,
) -> None:
    """Write N elements as ``{prefix}_state_{k}`` keys with a
    ``{prefix}_state_count`` count marker. Zero-dim shapes are
    preserved via ``{prefix}_state_{k}_zero_dim``. Composite
    elements (a bare tuple/list or a nested ``__nstate__``
    marker) recurse under a ``{elem_key}`` sub-prefix and record
    a ``{elem_key}_nested`` marker; the flat ``{elem_key}``
    tensor is deliberately omitted so an older reader hits its
    ``Missing {elem_key} in arrays`` path and skips the block.

    Module-level on purpose — same recursive-closure cycle bug as
    ``_load_nstate_flat`` (see that docstring).
    """
    cache_list_meta[f"{prefix}_state_count"] = str(len(elements))
    for k, elem in enumerate(elements):
        elem_key = f"{prefix}_state_{k}"
        if elem is None:
            # None placeholder — store an empty marker tensor
            # and a sentinel zero_dim entry so the loader can
            # restore None instead of materializing zeros.
            arrays[elem_key] = mx.zeros((1,))
            cache_list_meta[f"{elem_key}_none"] = "1"
        elif _has_zero_dim(elem):
            arrays[elem_key] = mx.zeros((1,))
            cache_list_meta[f"{elem_key}_zero_dim"] = _encode_shape(elem.shape)
        elif (
            isinstance(elem, tuple)
            and len(elem) >= 2
            and isinstance(elem[0], str)
            and elem[0] == "__nstate__"
        ):
            # Nested ``('__nstate__', class_name, [sub...])``
            # marker — recurse, no flat tensor written.
            cache_list_meta[f"{elem_key}_nested"] = "nstate"
            sub_class = elem[1] if len(elem) >= 2 else None
            sub_elements = elem[2] if len(elem) >= 3 else []
            if sub_class:
                cache_list_meta[f"{elem_key}_state_class_name"] = sub_class
            _store_nstate_elements_flat(arrays, cache_list_meta, elem_key, sub_elements)
        elif isinstance(elem, (tuple, list)):
            # Bare tuple/list of sub-elements — recurse, no flat
            # tensor written.
            cache_list_meta[f"{elem_key}_nested"] = "tuple"
            _store_nstate_elements_flat(arrays, cache_list_meta, elem_key, list(elem))
        else:
            if not isinstance(elem, mx.array):
                raise TypeError(
                    f"unsupported non-array nstate element "
                    f"{elem_key}: {type(elem).__name__}"
                )
            arrays[elem_key] = elem


def _load_nstate_flat(
    arrays: dict[str, Any],
    file_metadata: dict[str, str],
    prefix: str,
    fallback_class: str | None,
) -> tuple | None:
    """Read either V3 ``state_count`` keys or V2 ``keys``/``values``
    polyfill at ``prefix``. Returns ``('__nstate__', class_name, elements)``
    on success or None on missing tensors.

    Module-level on purpose: the previous nested-closure version formed a
    self-referential cycle (recursive closure) that captured ``arrays`` —
    hundreds of MB of KV tensors — and only gen-2 gc could free it.
    """
    count_key = f"{prefix}_state_count"
    class_name = None
    if file_metadata:
        class_name = file_metadata.get(f"{prefix}_state_class_name")
    if class_name is None:
        class_name = fallback_class

    elements: list[Any] = []
    if file_metadata and count_key in file_metadata:
        # V3 path
        try:
            count = int(file_metadata[count_key])
        except (ValueError, TypeError):
            return None
        for k in range(count):
            elem_key = f"{prefix}_state_{k}"
            none_marker = f"{elem_key}_none"
            zd_marker = f"{elem_key}_zero_dim"
            nested_marker = f"{elem_key}_nested"
            if file_metadata and none_marker in file_metadata:
                elements.append(None)
                continue
            if file_metadata and nested_marker in file_metadata:
                # Composite element — recurse, then restore the same
                # shape it had on save (bare tuple vs __nstate__).
                sub = _load_nstate_flat(arrays, file_metadata, elem_key, None)
                if sub is None:
                    return None
                if file_metadata[nested_marker] == "tuple":
                    elements.append(tuple(sub[2]))
                elif file_metadata[nested_marker] == "nstate":
                    # Explicit marker on write — preserve the full
                    # ('__nstate__', class_name, elements) as-is;
                    # never unwrap (would drop the marker/class_name).
                    elements.append(sub)
                else:
                    # Corrupt/unknown nested marker — fail closed.
                    return None
                continue
            if elem_key not in arrays:
                logger.error(f"Missing {elem_key} in arrays")
                return None
            if file_metadata and zd_marker in file_metadata:
                elements.append(mx.zeros(_decode_shape(file_metadata[zd_marker])))
            else:
                elements.append(arrays[elem_key])
    else:
        # V2 polyfill: legacy ``{prefix}_keys`` / ``{prefix}_values``.
        keys_key = f"{prefix}_keys"
        values_key = f"{prefix}_values"
        if keys_key not in arrays or values_key not in arrays:
            return None
        k_zd = f"{prefix}_keys_zero_dim"
        v_zd = f"{prefix}_values_zero_dim"
        if file_metadata and k_zd in file_metadata:
            elements.append(mx.zeros(_decode_shape(file_metadata[k_zd])))
        else:
            elements.append(arrays[keys_key])
        if file_metadata and v_zd in file_metadata:
            elements.append(mx.zeros(_decode_shape(file_metadata[v_zd])))
        else:
            elements.append(arrays[values_key])
    return ("__nstate__", class_name, elements)


# --- Safetensors dtype mapping for background-thread-safe serialization ---
# These mappings enable writing safetensors files without any mx/Metal API,
# bypassing the bfloat16 limitation that blocked PR #16 v2 (numpy doesn't
# support bfloat16, but safetensors format natively does via "BF16" dtype).

_MX_TO_ST_DTYPE: dict[Any, str] = {}
_ST_TO_MX_DTYPE: dict[str, Any] = {}
_ST_DTYPE_TO_NP: dict[str, Any] = {}

if HAS_MLX:
    _MX_TO_ST_DTYPE = {
        mx.float16: "F16",
        mx.float32: "F32",
        mx.bfloat16: "BF16",
        mx.int8: "I8",
        mx.int16: "I16",
        mx.int32: "I32",
        mx.int64: "I64",
        mx.uint8: "U8",
        mx.uint16: "U16",
        mx.uint32: "U32",
        mx.uint64: "U64",
        mx.bool_: "BOOL",
    }
    _ST_TO_MX_DTYPE = {v: k for k, v in _MX_TO_ST_DTYPE.items()}

_ST_DTYPE_TO_NP = {
    "F16": np.float16,
    "F32": np.float32,
    "BF16": np.uint16,  # bfloat16 handled via uint16 view
    "I8": np.int8,
    "I16": np.int16,
    "I32": np.int32,
    "I64": np.int64,
    "U8": np.uint8,
    "U16": np.uint16,
    "U32": np.uint32,
    "U64": np.uint64,
    "BOOL": np.bool_,
}


def _extract_tensor_bytes(arr: mx.array) -> tuple[bytes, str, list[int]]:
    """Extract raw bytes from an mx.array.

    Materialize the array at this last-mile boundary before touching the
    Python buffer protocol. ``store_cache`` may create lazy block slices,
    clones, or placeholder arrays after scheduler-side pre-eval collection,
    and ``memoryview(arr)`` would otherwise trigger an implicit eval from the
    background cache-store worker thread.

    For bfloat16 arrays, uses view(uint16) since the buffer protocol does
    not support bfloat16 directly. Materialize the view as well so the raw
    buffer read never becomes an implicit MLX eval.

    Args:
        arr: MLX array to serialize.

    Returns:
        Tuple of (raw_bytes, safetensors_dtype_string, shape_list).
    """
    mx.eval(arr)
    dtype_str = _MX_TO_ST_DTYPE[arr.dtype]
    shape = list(arr.shape)
    if arr.dtype == mx.bfloat16:
        u16 = arr.view(mx.uint16)
        mx.eval(u16)
        raw = bytes(memoryview(u16))
    else:
        raw = bytes(memoryview(arr))
    return raw, dtype_str, shape


def _restore_tensor_from_bytes(
    raw: bytes, dtype_str: str, shape: list[int]
) -> mx.array:
    """Restore an mx.array from raw bytes extracted by _extract_tensor_bytes.

    No Metal API required — uses numpy as intermediary.

    Args:
        raw: Raw tensor bytes.
        dtype_str: Safetensors dtype string (e.g., "F16", "BF16").
        shape: Tensor shape as list of ints.

    Returns:
        Restored mx.array with correct dtype and shape.
    """
    np_dtype = _ST_DTYPE_TO_NP[dtype_str]
    np_arr = np.frombuffer(raw, dtype=np_dtype)
    arr = mx.array(np_arr)
    if dtype_str == "BF16":
        arr = arr.view(mx.bfloat16)
    return arr.reshape(shape)


def _write_safetensors_no_mx(
    path: str,
    tensors_raw: dict[str, tuple[bytes, str, list[int]]],
    metadata: dict[str, str] | None = None,
) -> int:
    """Write a safetensors file without any mx/Metal API calls.

    Safe to call from background threads. Produces files fully compatible
    with mx.load(path, return_metadata=True).

    The safetensors binary format:
      [8 bytes: header_size as little-endian uint64]
      [header_size bytes: JSON header]
      [remaining bytes: concatenated tensor data]

    Args:
        path: Output file path (must include .safetensors extension).
        tensors_raw: Dict of {name: (raw_bytes, dtype_str, shape)}.
        metadata: Optional string-to-string metadata dict.

    Returns:
        Total file size in bytes.
    """
    offset = 0
    header_tensors = {}
    all_data = []

    for name, (raw, dtype_str, shape) in tensors_raw.items():
        header_tensors[name] = {
            "dtype": dtype_str,
            "shape": shape,
            "data_offsets": [offset, offset + len(raw)],
        }
        all_data.append(raw)
        offset += len(raw)

    header_dict = dict(header_tensors)
    if metadata:
        header_dict["__metadata__"] = metadata

    header_json = json.dumps(header_dict, separators=(",", ":")).encode("utf-8")
    # Safetensors spec: header must be 8-byte aligned
    pad = (8 - len(header_json) % 8) % 8
    header_json += b" " * pad

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        for d in all_data:
            f.write(d)

    return 8 + len(header_json) + offset


def parse_size(size_str: str) -> int:
    """
    Parse a human-readable size string to bytes.

    Args:
        size_str: Size string like "100GB", "50MB", "1TB"

    Returns:
        Size in bytes.
    """
    size_str = size_str.strip().upper()

    units = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
    }

    for unit, multiplier in units.items():
        if size_str.endswith(unit):
            try:
                value = float(size_str[: -len(unit)])
                return int(value * multiplier)
            except ValueError:
                pass

    # Try parsing as plain number (bytes)
    try:
        return int(size_str)
    except ValueError:
        raise ValueError(f"Invalid size string: {size_str}")


@dataclass
class PagedSSDBlockMetadata:
    """
    Metadata for a block stored on SSD.

    Attributes:
        block_hash: Content hash (SHA256) for identification
        file_path: Full path to safetensors file
        file_size: Size in bytes
        token_count: Number of tokens in this block
        created_at: Timestamp when saved
        last_access: Last access time for LRU tracking
        num_layers: Number of model layers
        model_name: Model name for cache isolation between different models
        block_size: Paged cache block size that created this block
        cache_signature: Compatibility signature for the saved cache layout
        layer_cache_types: Per-layer cache type names (e.g., ["KVCache", "ArraysCache"])
        layer_meta_states: Per-layer meta_state tuples for reconstruction
    """

    block_hash: bytes
    file_path: Path
    file_size: int
    token_count: int
    created_at: float
    last_access: float
    num_layers: int
    model_name: str = ""
    block_size: int = 0
    cache_signature: str = ""
    layer_cache_types: list[str] | None = None
    layer_meta_states: list[tuple] | None = None

    def touch(self) -> None:
        """Update last access time."""
        self.last_access = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "block_hash": self.block_hash.hex(),
            "file_path": str(self.file_path),
            "file_size": self.file_size,
            "token_count": self.token_count,
            "created_at": self.created_at,
            "last_access": self.last_access,
            "num_layers": self.num_layers,
            "model_name": self.model_name,
            "block_size": self.block_size,
            "cache_signature": self.cache_signature,
        }
        if self.layer_cache_types:
            result["layer_cache_types"] = self.layer_cache_types
        if self.layer_meta_states:
            # Convert tuples to lists for JSON serialization
            result["layer_meta_states"] = [list(m) for m in self.layer_meta_states]
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PagedSSDBlockMetadata:
        """Create from dictionary."""
        # Parse layer_meta_states back to tuples
        layer_meta_states = None
        if "layer_meta_states" in data and data["layer_meta_states"]:
            layer_meta_states = [tuple(m) for m in data["layer_meta_states"]]

        return cls(
            block_hash=bytes.fromhex(data["block_hash"]),
            file_path=Path(data["file_path"]),
            file_size=data["file_size"],
            token_count=data["token_count"],
            created_at=data["created_at"],
            last_access=data["last_access"],
            num_layers=data["num_layers"],
            model_name=data.get("model_name", ""),
            block_size=data.get("block_size", 0),
            cache_signature=data.get("cache_signature", ""),
            layer_cache_types=data.get("layer_cache_types"),
            layer_meta_states=layer_meta_states,
        )


class PagedSSDCacheIndex:
    """
    In-memory index of SSD cache files.

    Provides O(1) lookup by block_hash and LRU tracking for size management.
    Thread-safe for concurrent access.
    """

    def __init__(self, max_size_bytes: int):
        """
        Initialize the SSD cache index.

        Args:
            max_size_bytes: Maximum total size of SSD cache files.
        """
        self._index: dict[bytes, PagedSSDBlockMetadata] = {}
        self._lru: OrderedDict[bytes, float] = OrderedDict()
        self._total_size: int = 0
        self._max_size: int = max_size_bytes
        self._lock = threading.RLock()

    def add(self, metadata: PagedSSDBlockMetadata) -> None:
        """
        Add a block to the index.

        Args:
            metadata: Block metadata to add.
        """
        with self._lock:
            # Remove existing entry if present
            if metadata.block_hash in self._index:
                old_meta = self._index[metadata.block_hash]
                self._total_size -= old_meta.file_size
                del self._lru[metadata.block_hash]

            self._index[metadata.block_hash] = metadata
            self._lru[metadata.block_hash] = metadata.last_access
            self._total_size += metadata.file_size

    def sort_lru_by_last_access(self) -> None:
        """Restore LRU ordering from each entry's last access timestamp."""
        with self._lock:
            self._lru = OrderedDict(
                sorted(
                    (
                        (block_hash, self._index[block_hash].last_access)
                        for block_hash in self._lru
                        if block_hash in self._index
                    ),
                    key=lambda item: item[1],
                )
            )

    def get(self, block_hash: bytes) -> PagedSSDBlockMetadata | None:
        """
        Get block metadata by hash.

        Args:
            block_hash: Block content hash.

        Returns:
            PagedSSDBlockMetadata if found, None otherwise.
        """
        with self._lock:
            return self._index.get(block_hash)

    def remove(self, block_hash: bytes) -> PagedSSDBlockMetadata | None:
        """
        Remove a block from the index.

        Args:
            block_hash: Block content hash.

        Returns:
            Removed metadata if found, None otherwise.
        """
        with self._lock:
            if block_hash not in self._index:
                return None

            metadata = self._index.pop(block_hash)
            del self._lru[block_hash]
            self._total_size -= metadata.file_size
            return metadata

    def touch(self, block_hash: bytes) -> None:
        """
        Update last access time (move to end of LRU).

        Args:
            block_hash: Block content hash.
        """
        with self._lock:
            if block_hash in self._index:
                self._index[block_hash].touch()
                self._lru.move_to_end(block_hash)
                self._lru[block_hash] = self._index[block_hash].last_access

    def get_lru_entries(self, count: int) -> list[PagedSSDBlockMetadata]:
        """
        Get least recently used entries.

        Args:
            count: Maximum number of entries to return.

        Returns:
            List of LRU metadata entries.
        """
        with self._lock:
            result = []
            for block_hash in list(self._lru.keys())[:count]:
                if block_hash in self._index:
                    result.append(self._index[block_hash])
            return result

    def evict_until_size(
        self,
        target_size: int,
        max_count: int | None = None,
    ) -> list[PagedSSDBlockMetadata]:
        """
        Evict LRU entries until total size is below target.

        Args:
            target_size: Target total size in bytes.
            max_count: Optional cap on the number of entries removed in
                one call. When the cap is hit before ``total_size`` drops
                below ``target_size`` the call returns the partial slice
                and leaves the remaining LRU entries in the index; the
                caller is expected to retry on the next save. The cap is
                pushed down here (rather than the caller popping a
                surplus and reinserting it) so the index never exposes a
                transient "evicted but not yet unlinked" gap that a
                concurrent writer's ``contains()`` check could observe
                as a deleted block.

        Returns:
            List of evicted metadata (files need to be deleted by caller).

        Note:
            Loop termination depends on ``remove()`` decrementing
            ``_total_size`` for every popped entry. If a future refactor
            moves the decrement to "after the on-disk unlink succeeds",
            this loop must also gain a "skip entries already pulled this
            pass" guard or it can spin forever when unlinks fail.
        """
        with self._lock:
            evicted = []
            while self._total_size > target_size and self._lru:
                if max_count is not None and len(evicted) >= max_count:
                    break
                # Get LRU entry (first in OrderedDict)
                block_hash = next(iter(self._lru))
                metadata = self.remove(block_hash)
                if metadata:
                    evicted.append(metadata)
            return evicted

    def contains(self, block_hash: bytes) -> bool:
        """Check if block exists in index."""
        with self._lock:
            return block_hash in self._index

    @property
    def total_size(self) -> int:
        """Get total size of indexed files."""
        with self._lock:
            return self._total_size

    @property
    def max_size(self) -> int:
        """Get maximum allowed size."""
        return self._max_size

    @property
    def count(self) -> int:
        """Get number of indexed blocks."""
        with self._lock:
            return len(self._index)

    def update_file_size(self, block_hash: bytes, actual_size: int) -> None:
        """Update file size for a block after background write completes.

        Args:
            block_hash: Block content hash.
            actual_size: Actual file size in bytes.
        """
        with self._lock:
            entry = self._index.get(block_hash)
            if entry is not None:
                self._total_size += actual_size - entry.file_size
                entry.file_size = actual_size

    def get_all_hashes(self) -> list[bytes]:
        """Get all indexed block hashes."""
        with self._lock:
            return list(self._index.keys())

    def get_all_metadata(self) -> list[PagedSSDBlockMetadata]:
        """Get a snapshot of all indexed block metadata."""
        with self._lock:
            return list(self._index.values())


@dataclass
class _HotCacheBudgetEntry:
    owner: Any
    block_hash: bytes
    size_bytes: int


class SharedHotCacheBudget:
    """Process-wide byte budget for hot cache entries across cache managers."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max(0, int(max_bytes))
        self._entries: OrderedDict[tuple[int, bytes], _HotCacheBudgetEntry] = (
            OrderedDict()
        )
        self._total_bytes = 0
        self._lock = threading.RLock()

    @staticmethod
    def _key(owner: Any, block_hash: bytes) -> tuple[int, bytes]:
        return (id(owner), block_hash)

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return self._total_bytes

    @property
    def remaining_bytes(self) -> int:
        with self._lock:
            return max(0, self.max_bytes - self._total_bytes)

    def touch(self, owner: Any, block_hash: bytes) -> None:
        """Mark an entry as recently used in the global LRU order."""
        with self._lock:
            key = self._key(owner, block_hash)
            if key in self._entries:
                self._entries.move_to_end(key)

    def forget(self, owner: Any, block_hash: bytes) -> None:
        """Remove one entry from budget accounting if present."""
        with self._lock:
            key = self._key(owner, block_hash)
            entry = self._entries.pop(key, None)
            if entry is not None:
                self._total_bytes = max(0, self._total_bytes - entry.size_bytes)

    def forget_owner(self, owner: Any) -> None:
        """Remove all entries owned by a cache manager."""
        owner_id = id(owner)
        with self._lock:
            keys = [key for key in self._entries if key[0] == owner_id]
            for key in keys:
                entry = self._entries.pop(key)
                self._total_bytes = max(0, self._total_bytes - entry.size_bytes)

    def clear_all_owners(self) -> int:
        """Clear the hot cache of every manager the budget still references.

        The budget keeps a strong reference to each owning manager, so a
        manager orphaned by an abnormal teardown stays reachable here even
        when it is no longer attached to a loaded scheduler. Snapshot the
        owners under the lock, then clear outside it (clear_hot_cache calls
        forget_owner, which re-takes the lock).
        """
        with self._lock:
            owners = []
            seen = set()
            for entry in self._entries.values():
                if id(entry.owner) not in seen:
                    seen.add(id(entry.owner))
                    owners.append(entry.owner)
        cleared = 0
        for owner in owners:
            fn = getattr(owner, "clear_hot_cache", None)
            if callable(fn):
                try:
                    cleared += fn()
                except Exception:
                    # Keep going for the other owners, but do not hide the
                    # failure: a silently-swallowed error makes admin recovery
                    # look successful while memory is still pinned.
                    logger.warning(
                        "clear_hot_cache failed for an orphaned owner",
                        exc_info=True,
                    )
        return cleared

    def shrink_to(
        self,
        target_bytes: int,
        protected_hashes: set[bytes] | None = None,
    ) -> int:
        """Shrink the shared hot cache to ``target_bytes`` by global LRU order.

        Returns the budgeted bytes removed from hot-cache ownership. Dirty
        evictions are handed back to each owner so the existing SSD write-through
        path can preserve them.
        """
        target_bytes = max(0, int(target_bytes))
        protected_hashes = protected_hashes or set()
        victims: list[tuple[Any, bytes, int]] = []

        with self._lock:
            while self._total_bytes > target_bytes and self._entries:
                victim_key = None
                victim = None
                for key, candidate in self._entries.items():
                    if candidate.block_hash not in protected_hashes:
                        victim_key = key
                        victim = candidate
                        break
                if victim_key is None or victim is None:
                    break

                self._entries.pop(victim_key)
                self._total_bytes = max(0, self._total_bytes - victim.size_bytes)
                victims.append((victim.owner, victim.block_hash, victim.size_bytes))

        freed = 0
        for owner, block_hash, size_bytes in victims:
            evicted = owner._hot_cache_remove(block_hash, update_budget=False)
            if evicted is not None:
                freed += size_bytes
                owner._handle_hot_cache_eviction(block_hash, evicted)
        return freed

    def put(
        self, owner: Any, block_hash: bytes, size_bytes: int
    ) -> list[tuple[Any, bytes]]:
        """Account an entry and return globally-evicted owners/block hashes."""
        victims: list[tuple[Any, bytes]] = []
        size_bytes = max(0, int(size_bytes))
        with self._lock:
            key = self._key(owner, block_hash)
            old = self._entries.pop(key, None)
            if old is not None:
                self._total_bytes = max(0, self._total_bytes - old.size_bytes)

            self._entries[key] = _HotCacheBudgetEntry(
                owner=owner,
                block_hash=block_hash,
                size_bytes=size_bytes,
            )
            self._total_bytes += size_bytes

            while self._total_bytes > self.max_bytes and self._entries:
                victim_key, victim = self._entries.popitem(last=False)
                if victim_key == key and not self._entries:
                    self._entries[victim_key] = victim
                    break
                self._total_bytes = max(0, self._total_bytes - victim.size_bytes)
                victims.append((victim.owner, victim.block_hash))

        return victims


class PagedSSDCacheManager(CacheManager):
    """
    Manages SSD storage for KV cache blocks.

    Features:
    - Block-level safetensors serialization
    - Hash-based subdirectory structure (single level: /a/, /b/, etc.)
    - LRU-based SSD cache size management

    Implements the CacheManager ABC interface for consistency with other
    cache implementations in oMLX.

    Example:
        >>> manager = PagedSSDCacheManager(
        ...     cache_dir=Path("/tmp/ssd_cache"),
        ...     max_size_bytes=100 * 1024**3,  # 100GB
        ... )
        >>> manager.save_block(block_hash, cache_data, token_count=64)
        >>> loaded = manager.load_block(block_hash)
    """

    # Subdirectory prefixes (hash first char)
    SUBDIR_CHARS = "0123456789abcdef"

    def __init__(
        self,
        cache_dir: Path | None,
        max_size_bytes: int,
        hot_cache_max_bytes: int = 0,
        hot_cache_only: bool = False,
        hot_cache_budget: SharedHotCacheBudget | None = None,
        expected_model_name: str = "",
        expected_num_layers: int = 0,
        expected_block_size: int = 0,
        expected_block_size_tokens: int = _DEFAULT_BLOCK_SIZE_TOKENS,
        expected_kv_bytes_per_token: int = _DEFAULT_KV_BYTES_PER_TOKEN,
        expected_layer_cache_types: list[str] | None = None,
    ):
        """
        Initialize the SSD cache manager.

        Args:
            cache_dir: Directory for SSD cache files.
            max_size_bytes: Maximum total size of SSD cache.
            hot_cache_max_bytes: Maximum in-memory hot cache size in bytes.
                0 means disabled (default).
            hot_cache_only: When True, skip directory init and writer thread.
                All data is stored exclusively in the hot cache (RAM only).
                No SSD I/O is performed.
            hot_cache_budget: Optional process-wide hot cache budget shared
                by all loaded model cache managers.
            expected_model_name: Current model name. Blocks saved for a
                different model name are skipped at startup. Empty string
                disables this check (backwards compatible).
            expected_num_layers: Current cache-layer count. Blocks saved with
                a different num_layers are skipped at startup. 0 disables this
                check (backwards compatible). Catches stale blocks left over
                after a model upgrade changes its effective layer count (e.g.,
                #1404 attaching MTPModule changed 30 -> 40 layers).
            expected_block_size: Current paged cache block size. Blocks saved
                with another block size are skipped at startup. 0 disables this
                check for backwards compatibility.
            expected_block_size_tokens: Paged-cache block size in tokens used to
                size the pending-writes queue. Separate from ``expected_block_size``
                so the writer-queue formula keeps a real value (default 256)
                even when the cache-compat check is disabled (0). Passing a
                larger value shrinks the cap so small Macs with large blocks
                don't pin gigabytes at saturation; passing a smaller value lets
                the cap grow to give workloads with many tiny blocks enough
                burst headroom.
            expected_kv_bytes_per_token: Per-token KV byte estimate (all
                layers, K + V, dtype). Together with ``expected_block_size_tokens``
                this drives the bytes-aware queue cap. Defaults to a
                35B-class bf16 estimate; pass an explicit value for
                quantized models or unusually wide/narrow architectures.
            expected_layer_cache_types: Optional current cache layout. When
                provided, blocks with a different per-layer type list are
                skipped at startup.
        """
        self._cache_dir = cache_dir
        self._max_size = max_size_bytes
        self._index = PagedSSDCacheIndex(max_size_bytes)
        self._incompatible_index = PagedSSDCacheIndex(max_size_bytes)
        self._hot_cache_only = hot_cache_only
        self._expected_model_name = expected_model_name
        self._expected_num_layers = expected_num_layers
        self._expected_block_size = expected_block_size
        self._expected_layer_cache_types = expected_layer_cache_types
        # TurboQuant bit depth requests will quantize at; learned together
        # with the layer signature via ``set_expected_layer_signature``
        # (the depth is only known once the engine has applied the model's
        # TurboQuant settings, after this manager is constructed).
        self._expected_turboquant_kv_bits: float | None = None
        # Set once we have swept stale-signature blocks for the current
        # ``_expected_layer_cache_types`` / ``_expected_turboquant_kv_bits``.
        # Re-assigning the signature (e.g., via
        # ``adopt_layer_signature_if_unset``) resets this so the new
        # signature triggers its own one-shot sweep.
        self._signature_sweep_completed = False
        self._lock = threading.RLock()

        # Disk usage cache for dynamic effective max size (30s TTL)
        self._disk_usage_cache = None  # type: shutil._ntuple_diskusage | None
        self._disk_usage_cache_time: float = 0.0
        self._last_disk_pressure_warn: float = 0.0

        # Statistics
        self._stats = {
            "saves": 0,
            "saves_persisted": 0,
            "loads": 0,
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "evict_unlink_failures": 0,
            "errors": 0,
            "hot_cache_hits": 0,
            "hot_cache_evictions": 0,
            "hot_cache_promotions": 0,
            "preload_calls": 0,
            "preload_blocks_loaded": 0,
            "preload_time_ms": 0.0,
            "ssd_write_drops": 0,
            "ssd_inline_write_fallbacks": 0,
        }

        # --- Hot cache (in-memory raw-bytes tier) ---
        self._hot_cache_budget = hot_cache_budget
        self._hot_cache_max_bytes = (
            hot_cache_budget.max_bytes
            if hot_cache_budget is not None
            else hot_cache_max_bytes
        )
        self._hot_cache_enabled = self._hot_cache_max_bytes > 0
        self._hot_cache: OrderedDict[bytes, dict] = OrderedDict()
        self._hot_cache_total_bytes: int = 0
        self._hot_cache_lock = threading.Lock()

        # Initialize directory structure and scan existing files
        # Skip in hot_cache_only mode: no SSD I/O, so no directories needed.
        if self._cache_dir and not self._hot_cache_only:
            self._init_directories()
            self._scan_existing_files()

        # --- Background writer for non-blocking saves ---
        # Recompute the pending-writes cap from THIS cache's block/model
        # parameters so a non-default block size shrinks (or grows) the
        # cap appropriately. Falls back to the module-level constant
        # when no override is supplied.
        #
        # Stash the inputs the constructor was called with so callers
        # (and the plumbing-regression test) can verify what reached
        # the manager without depending on the cap math landing in a
        # particular floor/ceiling band on the test host.
        self._expected_block_size_tokens = expected_block_size_tokens
        self._expected_kv_bytes_per_token = expected_kv_bytes_per_token
        self._max_pending_writes = _compute_max_pending_writes(
            block_size_tokens=expected_block_size_tokens,
            kv_bytes_per_token=expected_kv_bytes_per_token,
        )
        self._write_queue: queue.Queue = queue.Queue(maxsize=self._max_pending_writes)
        # Track which block hashes are queued for background write
        self._pending_write_hashes: set = set()
        self._pending_write_hashes_lock = threading.Lock()
        # Lock ordering invariant: _hot_cache_lock -> _pending_write_hashes_lock.
        # Never acquire in reverse. Load path: _hot_cache_get (holds _hot_cache_lock,
        # releases), then _pending_write_buffer_get (holds _pending_write_hashes_lock).
        # Eviction path: _hot_cache_put (holds _hot_cache_lock, releases), then
        # _enqueue_ssd_write (holds _pending_write_hashes_lock).
        self._pending_write_buffers: dict[bytes, dict] = {}
        self._writer_shutdown = threading.Event()
        # Writer thread is only needed when writing to SSD.
        self._writer_thread = None
        if not self._hot_cache_only:
            self._writer_thread = threading.Thread(
                target=self._writer_loop,
                name="ssd-cache-writer",
                daemon=True,
            )
            self._writer_thread.start()

        hot_info = ""
        if self._hot_cache_enabled:
            hot_info = f", hot_cache={format_bytes(hot_cache_max_bytes)}"
        # Log initialization with disk space info
        disk_info = ""
        if self._cache_dir:
            try:
                du = shutil.disk_usage(self._cache_dir)
                disk_info = (
                    f", disk_free={format_bytes(du.free)}, "
                    f"cache_used={format_bytes(self._tracked_ssd_size())}, "
                    f"incompatible_files={self._incompatible_index.count}"
                )
            except OSError:
                pass
        logger.info(
            f"PagedSSDCacheManager initialized: dir={self._cache_dir}, "
            f"max_size={format_bytes(max_size_bytes)}{hot_info}, "
            f"existing_files={self._index.count}{disk_info}"
        )

    # --- Hot cache helpers ---

    @staticmethod
    def _hot_cache_entry_size(entry: dict) -> int:
        """Calculate memory footprint of a hot cache entry.

        Entries from save_block() use 'tensors_raw' (raw bytes).
        Entries from _promote_to_hot_cache() may use 'arrays' (mx.array objects
        loaded from SSD, not from active inference — safe to retain).
        """
        if "arrays" in entry:
            return sum(arr.nbytes for arr in entry["arrays"].values())
        if "tensors_raw" in entry:
            return sum(len(raw) for raw, _, _ in entry["tensors_raw"].values())
        return 0

    def _effective_hot_cache_max_bytes(self) -> int:
        if self._hot_cache_budget is not None:
            return self._hot_cache_budget.max_bytes
        return self._hot_cache_max_bytes

    def _hot_cache_available_bytes(self) -> int:
        if self._hot_cache_budget is not None:
            return self._hot_cache_budget.remaining_bytes
        return max(0, self._hot_cache_max_bytes - self._hot_cache_total_bytes)

    def _handle_hot_cache_eviction(self, block_hash: bytes, entry: dict) -> None:
        self._stats["hot_cache_evictions"] += 1
        if not entry.get("dirty", True):
            logger.debug(
                "Evicted clean hot cache block %s; SSD copy already exists",
                block_hash.hex()[:16],
            )
            return
        self._enqueue_ssd_write(block_hash, entry)

    def _hot_cache_put(self, block_hash: bytes, entry: dict) -> None:
        """Add entry to hot cache, evicting LRU entries if capacity exceeded.

        Evicted entries are flushed to SSD via the background writer thread.
        """
        entry_size = self._hot_cache_entry_size(entry)
        evicted_entries: list = []

        if self._hot_cache_budget is not None:
            with self._hot_cache_lock:
                old = self._hot_cache.pop(block_hash, None)
                if old is not None:
                    self._hot_cache_total_bytes -= self._hot_cache_entry_size(old)
                self._hot_cache[block_hash] = entry
                self._hot_cache_total_bytes += entry_size

            victims = self._hot_cache_budget.put(self, block_hash, entry_size)
            for owner, victim_hash in victims:
                evicted = owner._hot_cache_remove(victim_hash, update_budget=False)
                if evicted is not None:
                    owner._handle_hot_cache_eviction(victim_hash, evicted)
            return

        with self._hot_cache_lock:
            # Remove old entry if updating
            if block_hash in self._hot_cache:
                old = self._hot_cache.pop(block_hash)
                self._hot_cache_total_bytes -= self._hot_cache_entry_size(old)

            # Evict LRU entries until we have room
            while (
                self._hot_cache_total_bytes + entry_size > self._hot_cache_max_bytes
                and self._hot_cache
            ):
                evicted_hash, evicted = self._hot_cache.popitem(last=False)
                self._hot_cache_total_bytes -= self._hot_cache_entry_size(evicted)
                evicted_entries.append((evicted_hash, evicted))

            self._hot_cache[block_hash] = entry
            self._hot_cache_total_bytes += entry_size

        # Flush evicted entries to SSD outside the hot cache lock
        for evicted_hash, evicted in evicted_entries:
            self._handle_hot_cache_eviction(evicted_hash, evicted)

    def _enqueue_ssd_write(
        self,
        block_hash: bytes,
        entry: dict,
        *,
        blocking: bool = False,
    ) -> bool:
        """Enqueue a hot cache entry for SSD background write.

        Used when evicting from hot cache or flushing on shutdown.
        Adds block to SSD index before enqueueing write.

        All callers wait briefly for queue space. If saturation persists, the
        caller writes inline so dirty hot-cache blocks are never dropped just
        because the background writer is behind.
        """
        if self._hot_cache_only:
            return False
        if not entry.get("dirty", True):
            return True

        blk_meta = entry.get("block_metadata")
        if blk_meta is None:
            return False
        file_path = blk_meta.file_path
        tensors_raw = entry.get("tensors_raw", {})
        if not tensors_raw:
            return False
        metadata = entry["file_metadata"]

        # 1. Buffer first — instant read-back for concurrent loads (CPD K1).
        #    Must precede _index.add so load_block never sees an index hit
        #    for a block that has no file and no buffer entry yet.
        with self._pending_write_hashes_lock:
            if block_hash in self._pending_write_buffers:
                return True
            self._pending_write_buffers[block_hash] = entry
            self._pending_write_hashes.add(block_hash)

        # 2. Index second — makes the block discoverable in has_block/contains.
        if not self._index.contains(block_hash):
            self._enforce_size_limit_for_new_block(blk_meta.file_size)
            self._incompatible_index.remove(block_hash)
            self._index.add(blk_meta)

        # 3. Queue third — enqueue for background writer.
        try:
            item = (block_hash, tensors_raw, metadata, file_path)
            # Non-blocking callers (hot-cache LRU spill) also wait so a
            # transient writer backlog doesn't silently drop blocks. Blocking
            # callers (shutdown flush) use the same bounded wait.
            self._write_queue.put(item, timeout=_PENDING_WRITE_PUT_TIMEOUT_SECONDS)
            logger.debug(
                f"Evicted hot cache block to SSD write queue: "
                f"{block_hash.hex()[:16]}..."
            )
            return True
        except queue.Full:
            self._stats["ssd_inline_write_fallbacks"] += 1
            logger.warning(
                f"SSD write queue saturated (cap={self._max_pending_writes}); "
                f"writing evicted block {block_hash.hex()[:16]} inline"
            )
            ok = self._write_block_file(
                block_hash,
                tensors_raw,
                metadata,
                file_path,
                source="inline-fallback",
            )
            self._clear_pending_write(block_hash)
            return ok

    def _hot_cache_get(self, block_hash: bytes) -> dict | None:
        """Get entry from hot cache, updating LRU order. Returns None on miss."""
        with self._hot_cache_lock:
            if block_hash in self._hot_cache:
                self._hot_cache.move_to_end(block_hash)
                entry = self._hot_cache[block_hash]
            else:
                return None
        if self._hot_cache_budget is not None:
            self._hot_cache_budget.touch(self, block_hash)
        return entry

    def _pending_write_buffer_get(self, block_hash: bytes) -> dict | None:
        """Get entry from pending-write buffer. Returns None on miss."""
        with self._pending_write_hashes_lock:
            return self._pending_write_buffers.get(block_hash)

    def _hot_cache_remove(
        self, block_hash: bytes, *, update_budget: bool = True
    ) -> dict | None:
        """Remove entry from hot cache if present."""
        with self._hot_cache_lock:
            old = self._hot_cache.pop(block_hash, None)
            if old:
                self._hot_cache_total_bytes -= self._hot_cache_entry_size(old)
        if old is not None and update_budget and self._hot_cache_budget is not None:
            self._hot_cache_budget.forget(self, block_hash)
        return old

    def _promote_to_hot_cache(
        self,
        block_hash: bytes,
        arrays: dict[str, Any],
        file_metadata: Any,
        metadata: PagedSSDBlockMetadata,
    ) -> None:
        """Promote a block loaded from SSD into the hot cache."""
        try:
            promoted_raw = {}
            for name, arr in arrays.items():
                promoted_raw[name] = _extract_tensor_bytes(arr)
            entry = {
                "tensors_raw": promoted_raw,
                "file_metadata": (
                    file_metadata if isinstance(file_metadata, dict) else {}
                ),
                "num_layers": metadata.num_layers,
                "layer_cache_types": metadata.layer_cache_types,
                "block_metadata": metadata,
                "dirty": False,
            }
            self._hot_cache_put(block_hash, entry)
            self._stats["hot_cache_promotions"] += 1
        except Exception:
            pass  # Promotion failure is non-critical

    def _init_directories(self) -> None:
        """Create cache directory structure."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for first hex character
        for char in self.SUBDIR_CHARS:
            subdir = self._cache_dir / char
            subdir.mkdir(exist_ok=True)

    def _get_file_path(self, block_hash: bytes) -> Path:
        """
        Get file path for a block hash.

        Uses first character of hex hash as subdirectory.

        Args:
            block_hash: Block content hash.

        Returns:
            Path to the safetensors file.
        """
        hash_hex = block_hash.hex()
        subdir = hash_hex[0]  # First character
        filename = f"{hash_hex}.safetensors"
        return self._cache_dir / subdir / filename

    def _tracked_ssd_size(self) -> int:
        """Return all SSD cache bytes tracked for this shared cache directory."""
        return self._index.total_size + self._incompatible_index.total_size

    def _tracked_ssd_count(self) -> int:
        """Return compatible plus incompatible tracked SSD cache file count."""
        return self._index.count + self._incompatible_index.count

    def _scan_existing_files(self) -> None:
        """Scan cache directory for existing files and build the compatible index.

        Only blocks compatible with the currently loaded model/layout are
        indexed. Incompatible blocks are left on disk so a shared SSD cache
        directory can safely serve multiple loaded models without one model's
        startup scan deleting another model's cache.
        """
        logger.info(f"Scanning SSD cache directory: {self._cache_dir}")

        scanned = 0
        indexed = 0
        skipped_incompatible = 0
        skipped_incompatible_bytes = 0
        errors = 0

        for subdir in self.SUBDIR_CHARS:
            subdir_path = self._cache_dir / subdir
            if not subdir_path.exists():
                continue

            for file_path in subdir_path.glob("*.safetensors"):
                scanned += 1
                try:
                    metadata = self._read_file_metadata(file_path)
                    if metadata is None:
                        continue
                    if not self._is_compatible_block(metadata):
                        skipped_incompatible += 1
                        skipped_incompatible_bytes += metadata.file_size
                        self._incompatible_index.add(metadata)
                        continue
                    self._index.add(metadata)
                    indexed += 1
                except Exception as e:
                    logger.warning(f"Failed to read {file_path}: {e}")
                    errors += 1

        self._index.sort_lru_by_last_access()
        self._incompatible_index.sort_lru_by_last_access()

        log_msg = (
            f"SSD cache scan complete: scanned={scanned}, indexed={indexed}, "
            f"errors={errors}, total_size={format_bytes(self._index.total_size)}"
        )
        if skipped_incompatible > 0:
            log_msg += (
                f", skipped_incompatible={skipped_incompatible} blocks "
                f"({format_bytes(skipped_incompatible_bytes)})"
            )
        logger.info(log_msg)

        # Startup can find a cache directory that already exceeds the shared
        # SSD budget. Converge immediately before serving requests.
        tracked_size = self._tracked_ssd_size()
        if tracked_size > 0 and tracked_size > self._get_effective_max_size():
            self._enforce_size_limit_for_new_block(0, unbounded=True)

    def _is_compatible_block(self, metadata: PagedSSDBlockMetadata) -> bool:
        """Return True when a block can be indexed for this manager."""
        if self._expected_model_name and metadata.model_name:
            if metadata.model_name != self._expected_model_name:
                return False
        if self._expected_num_layers > 0 and metadata.num_layers > 0:
            if metadata.num_layers != self._expected_num_layers:
                return False
        if self._expected_block_size > 0:
            if metadata.block_size <= 0:
                return False
            if metadata.block_size != self._expected_block_size:
                return False
        if self._expected_layer_cache_types is not None:
            if _canonicalize_layer_cache_types(
                metadata.layer_cache_types
            ) != _canonicalize_layer_cache_types(self._expected_layer_cache_types):
                return False
        if (
            self._expected_layer_cache_types is not None
            and not self._is_compatible_cache_signature(metadata)
        ):
            return False
        return True

    def _is_compatible_cache_signature(self, metadata: PagedSSDBlockMetadata) -> bool:
        """Return True when a saved cache_signature matches enabled checks."""
        if not metadata.cache_signature:
            return self._signature_bits_match("")

        try:
            payload = json.loads(metadata.cache_signature)
        except (TypeError, ValueError):
            expected_signature = (
                self._expected_cache_signature()
                if self._expected_layer_cache_types is not None
                else ""
            )
            return (
                not expected_signature or metadata.cache_signature == expected_signature
            )

        if self._expected_model_name:
            if payload.get("model_name", "") != self._expected_model_name:
                return False

        if self._expected_num_layers > 0:
            try:
                num_layers = int(payload.get("num_layers", 0) or 0)
            except (TypeError, ValueError):
                num_layers = 0
            if num_layers > 0 and num_layers != self._expected_num_layers:
                return False

        if self._expected_block_size > 0:
            try:
                block_size = int(payload.get("block_size", 0) or 0)
            except (TypeError, ValueError):
                block_size = 0
            if block_size > 0 and block_size != self._expected_block_size:
                return False

        if self._expected_layer_cache_types is not None:
            layer_cache_types = payload.get("layer_cache_types")
            if not isinstance(layer_cache_types, (list, tuple)):
                return False
            if _canonicalize_layer_cache_types(
                layer_cache_types
            ) != _canonicalize_layer_cache_types(self._expected_layer_cache_types):
                return False

        if not self._signature_bits_match(metadata.cache_signature):
            return False

        return True

    def _signature_bits_match(self, cache_signature: str) -> bool:
        """True when a block's recorded TurboQuant depth satisfies expectations.

        With no expected depth every block passes. With one, the block must
        PROVE a matching depth: the packed state width is
        ``ceil(head_dim * bits / 32)``, so a block written at another depth —
        or one with no recorded depth (pre-depth-stamping saves) — has an
        incompatible or unverifiable width, and restoring it poisons batch
        concatenation (#2045).
        """
        if self._expected_turboquant_kv_bits is None:
            return True
        return (
            _signature_turboquant_bits(cache_signature)
            == self._expected_turboquant_kv_bits
        )

    def _expected_cache_signature(self) -> str:
        if (
            not self._expected_model_name
            and self._expected_num_layers <= 0
            and self._expected_block_size <= 0
            and self._expected_layer_cache_types is None
        ):
            return ""
        return _cache_compat_signature(
            model_name=self._expected_model_name,
            num_layers=self._expected_num_layers,
            block_size=self._expected_block_size,
            layer_cache_types=self._expected_layer_cache_types,
            turboquant_kv_bits=self._expected_turboquant_kv_bits,
        )

    def _read_file_metadata(self, file_path: Path) -> PagedSSDBlockMetadata | None:
        """
        Read metadata from an existing cache file.

        Args:
            file_path: Path to safetensors file.

        Returns:
            PagedSSDBlockMetadata if valid, None otherwise.
        """
        if not HAS_MLX:
            return None

        try:
            # Load just the metadata without loading tensors
            _, metadata = mx.load(str(file_path), return_metadata=True)

            block_hash_hex = metadata.get("block_hash", "")
            if not block_hash_hex:
                return None

            # Reject pre-fix blocks. RotatingKVCache layers in those files
            # may have been zero-padded to max_size, which the new merge
            # contract would treat as real attention keys. See #934 / #903
            # and the _CACHE_FORMAT_VERSION docstring for context.
            #
            # V3 polyfills V2 blocks at read time so already-stored caches
            # stay valid after the N-tuple state refactor. Versions outside
            # _READABLE_CACHE_FORMAT_VERSIONS are still rejected.
            cache_version = metadata.get("omlx_cache_format_version")
            if cache_version not in _READABLE_CACHE_FORMAT_VERSIONS:
                logger.debug(
                    "Skipping cache block with unsupported format version "
                    "%r (readable %r): %s",
                    cache_version,
                    sorted(_READABLE_CACHE_FORMAT_VERSIONS),
                    file_path,
                )
                return None

            file_stat = file_path.stat()

            # Parse cache type information if present
            layer_cache_types = None
            layer_meta_states = None

            # A present-but-unparseable field is corruption (torn write,
            # damaged file), not a legacy block: indexing the block with the
            # field silently dropped would make reconstruction guess layer
            # types or per-layer meta for its tensors. Treat the block as
            # unusable; the hash then misses and the next request re-stores
            # it. Absent fields (legacy blocks) still pass through.
            if "layer_cache_types" in metadata and metadata["layer_cache_types"]:
                try:
                    layer_cache_types = json.loads(metadata["layer_cache_types"])
                except (json.JSONDecodeError, TypeError):
                    logger.warning(
                        "Corrupt layer_cache_types metadata JSON in cache "
                        "file %s; treating the block as unusable.",
                        file_path,
                    )
                    return None

            if "layer_meta_states" in metadata and metadata["layer_meta_states"]:
                try:
                    raw_meta_states = json.loads(metadata["layer_meta_states"])
                    layer_meta_states = [tuple(m) if m else () for m in raw_meta_states]
                except (json.JSONDecodeError, TypeError):
                    logger.warning(
                        "Corrupt layer_meta_states metadata JSON in cache "
                        "file %s; treating the block as unusable.",
                        file_path,
                    )
                    return None

            return PagedSSDBlockMetadata(
                block_hash=bytes.fromhex(block_hash_hex),
                file_path=file_path,
                file_size=file_stat.st_size,
                token_count=int(metadata.get("token_count", 0)),
                created_at=file_stat.st_ctime,
                last_access=file_stat.st_mtime,
                num_layers=int(metadata.get("num_layers", 0)),
                model_name=metadata.get("model_name", ""),
                block_size=int(metadata.get("block_size", 0)),
                cache_signature=metadata.get("cache_signature", ""),
                layer_cache_types=layer_cache_types,
                layer_meta_states=layer_meta_states,
            )
        except Exception as e:
            logger.debug(f"Failed to read metadata from {file_path}: {e}")
            return None

    def _write_block_file(
        self,
        block_hash: bytes,
        tensors_raw: dict[str, Any],
        metadata: dict[str, str],
        file_path: Path,
        *,
        source: str,
    ) -> bool:
        """Write one serialized block to disk from raw tensor bytes."""
        temp_path = None
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = file_path.with_name(file_path.stem + "_tmp.safetensors")
            actual_size = _write_safetensors_no_mx(
                str(temp_path), tensors_raw, metadata
            )

            os.rename(str(temp_path), str(file_path))

            # The block is now durable on disk; bump the persist counter
            # before any cleanup so ``saves_persisted`` reflects rename
            # success even if the post-rename eviction check below unlinks it.
            self._stats["saves_persisted"] += 1
            self._index.update_file_size(block_hash, actual_size)

            # Check if block was evicted while write was pending.
            if not self._index.contains(block_hash):
                logger.debug(
                    "Block %s evicted during %s write, cleaning up file",
                    block_hash.hex()[:16],
                    source,
                )
                with contextlib.suppress(Exception):
                    file_path.unlink()
            return True
        except Exception as e:
            if isinstance(e, OSError) and e.errno in (
                errno.ENOSPC,
                errno.EDQUOT,
            ):
                # Background writes may fail after save_block already returned
                # True, while inline fallbacks can still report False to the
                # caller. In both cases, surface disk pressure at ERROR level
                # and force the next save to recompute available space.
                logger.error(
                    "SSD cache disk full, cannot write block %s via %s: %s "
                    "(subsequent saves will recompute disk pressure)",
                    block_hash.hex()[:16],
                    source,
                    e,
                )
                # Invalidate the 30s disk-usage snapshot so the next
                # save sees the true (now-critical) free space and evicts
                # aggressively rather than trusting a stale inflated limit.
                # In-flight saves that already passed
                # _enforce_size_limit_for_new_block are still queued and may
                # ENOSPC again; invalidation only protects the next round of
                # save_block calls.
                with self._lock:
                    self._disk_usage_cache = None
            else:
                logger.error(
                    "SSD cache %s write failed for %s: %s",
                    source,
                    block_hash.hex()[:16],
                    e,
                )
            self._stats["errors"] += 1
            self._index.remove(block_hash)
            for p in (temp_path, file_path):
                with contextlib.suppress(Exception):
                    if p is not None and isinstance(p, Path) and p.exists():
                        p.unlink()
            return False

    def _clear_pending_write(
        self, block_hash: bytes, *, remove_hot_cache: bool = False
    ) -> None:
        """Clear pending-write bookkeeping after a queued or inline write."""
        with self._pending_write_hashes_lock:
            self._pending_write_hashes.discard(block_hash)
            self._pending_write_buffers.pop(block_hash, None)
        if remove_hot_cache:
            self._hot_cache_remove(block_hash)

    def _writer_loop(self) -> None:
        """Background writer that drains the write queue.

        Runs in a dedicated daemon thread. Writes full safetensors files
        using pure Python I/O (no mx/Metal API calls), then atomically
        renames temp files to their final paths.

        This is safe because save_block() extracts tensor data as raw bytes
        on the inference thread (Metal-safe), and this thread only performs
        standard file I/O operations.
        """
        while True:
            item = None
            try:
                item = self._write_queue.get(timeout=1.0)
            except queue.Empty:
                # Exit if shutdown was requested and queue is empty
                if self._writer_shutdown.is_set():
                    break
                continue

            if item is None:  # Sentinel for shutdown
                break

            block_hash, tensors_raw, metadata, file_path = item
            try:
                self._write_block_file(
                    block_hash, tensors_raw, metadata, file_path, source="background"
                )
                self._clear_pending_write(
                    block_hash, remove_hot_cache=not self._hot_cache_enabled
                )
            finally:
                # Avoid pinning the last raw tensor-byte batch while the
                # writer thread blocks waiting for more work.
                item = None
                block_hash = tensors_raw = metadata = file_path = None

    def save_block(
        self,
        block_hash: bytes,
        cache_data: list[Any],
        token_count: int,
        model_name: str = "",
        layer_cache_types: list[str] | None = None,
        layer_meta_states: list[tuple] | None = None,
        hot_cache_write_back: bool = True,
    ) -> bool:
        """
        Save a KV cache block to SSD storage (non-blocking).

        Data is enqueued for background writing. The block is immediately
        available for reads via the in-memory pending-writes buffer.

        Args:
            block_hash: Content hash for the block.
            cache_data: List of per-layer data. Each element is either:
                - (keys, values) tuple for standard caches (KVCache, etc.)
                - ('__cache_list__', sub_tensors) marker tuple for CacheList layers,
                  where sub_tensors is List[Tuple[keys, values]] per sub-cache.
            token_count: Number of tokens in the block.
            model_name: Model name for cache isolation between different models.
            layer_cache_types: Optional list of cache type names per layer
                (e.g., ["KVCache", "ArraysCache", "KVCache", "CacheList"]).
            layer_meta_states: Optional list of meta_state tuples per layer
                for reconstruction (e.g., [(offset,), (keep, max_size, offset, _idx)]).
            hot_cache_write_back: When False in SSD-backed hot-cache mode, enqueue
                through the SSD writer path instead of retaining a hot-cache copy.

        Returns:
            True if enqueued successfully, False otherwise.
        """
        if not HAS_MLX:
            logger.error("MLX not available, cannot save block")
            return False

        layer_cache_types = _storage_layer_cache_types(layer_cache_types)

        # First save call after a model load is the canonical source for
        # the live layer-cache signature (post-TurboQuant / post-MTP). If
        # the manager wasn't told the signature at construction, adopt it
        # now and sweep any index entries left over from a prior config.
        if self.adopt_layer_signature_if_unset(layer_cache_types):
            try:
                self.invalidate_stale_layer_signature()
            except Exception as e:
                logger.warning("Stale-signature sweep failed: %s", e)

        # Check if already exists in index (thread-safe)
        if self._index.contains(block_hash):
            self._index.touch(block_hash)
            self._stats["hits"] += 1
            return True

        # Also check hot cache / pending writes buffer
        hot_entry = None
        with self._hot_cache_lock:
            if block_hash in self._hot_cache:
                hot_entry = self._hot_cache[block_hash]

        if hot_entry is not None:
            if hot_cache_write_back or self._hot_cache_only:
                self._stats["hits"] += 1
                return True
            if not hot_entry.get("dirty", True):
                self._stats["hits"] += 1
                return True
            # Pressure write-through for an already-dirty hot-cache entry:
            # use the same pending-buffer / SSD-writer path as hot-cache
            # eviction, then drop the long-lived hot-cache reference.
            if self._enqueue_ssd_write(block_hash, hot_entry):
                self._hot_cache_remove(block_hash)
                self._stats["hits"] += 1
                return True
            return False

        file_path = self._get_file_path(block_hash)

        try:

            # Prepare arrays for safetensors. Three layer_data shapes are
            # accepted:
            # - ``('__nstate__', class_name, [elem0, elem1, ...])`` — V3
            #   N-tuple state from a handler-driven serialize_state path.
            # - ``('__cache_list__', sub_tensors)`` — composite layer; each
            #   sub_tensor may itself be a 2-tuple ``(keys, values)`` (V2
            #   legacy from prefix_cache) or an ``__nstate__`` marker.
            # - ``('__turboquant__'/'__turboquant_v2__', ...)`` — bespoke
            #   TurboQuant payload, unchanged.
            # - ``(keys, values)`` 2-tuple — V2 legacy. Promoted to V3 by
            #   storing as a length-2 ``__nstate__`` so the on-disk shape
            #   is uniform regardless of whether the producer (prefix_cache,
            #   etc.) has been migrated to emit ``__nstate__`` markers yet.
            arrays = {}
            cache_list_meta = (
                {}
            )  # Per-layer sidecar metadata (sub_count, state_count, etc.)

            # Shim; module-level to avoid a recursive-closure refcount
            # cycle pinning `arrays` — see _store_nstate_elements_flat.
            def _store_nstate_elements(prefix: str, elements):
                _store_nstate_elements_flat(arrays, cache_list_meta, prefix, elements)

            for i, layer_data in enumerate(cache_data):
                if (
                    isinstance(layer_data, tuple)
                    and len(layer_data) >= 2
                    and isinstance(layer_data[0], str)
                    and layer_data[0] == "__nstate__"
                ):
                    # ('__nstate__', class_name, [elements]) — V3 native
                    class_name = layer_data[1] if len(layer_data) >= 2 else None
                    elements = layer_data[2] if len(layer_data) >= 3 else []
                    if class_name:
                        cache_list_meta[f"layer_{i}_state_class_name"] = class_name
                    _store_nstate_elements(f"layer_{i}", elements)
                elif (
                    isinstance(layer_data, tuple)
                    and len(layer_data) == 2
                    and isinstance(layer_data[0], str)
                    and layer_data[0] == "__cache_list__"
                ):
                    # CacheList: sub-indexed tensors. Each sub_tensor may be
                    # a 2-tuple (legacy) or an ``__nstate__`` marker.
                    sub_tensors = layer_data[1]
                    cache_list_meta[f"layer_{i}_sub_count"] = str(len(sub_tensors))
                    for j, sub_tensor in enumerate(sub_tensors):
                        sub_prefix = f"layer_{i}_sub_{j}"
                        if (
                            isinstance(sub_tensor, tuple)
                            and len(sub_tensor) >= 2
                            and isinstance(sub_tensor[0], str)
                            and sub_tensor[0] == "__nstate__"
                        ):
                            sub_class_name = (
                                sub_tensor[1] if len(sub_tensor) >= 2 else None
                            )
                            sub_elements = sub_tensor[2] if len(sub_tensor) >= 3 else []
                            if sub_class_name:
                                cache_list_meta[f"{sub_prefix}_state_class_name"] = (
                                    sub_class_name
                                )
                            _store_nstate_elements(sub_prefix, sub_elements)
                        elif (
                            isinstance(sub_tensor, (list, tuple))
                            and len(sub_tensor) >= 2
                        ):
                            # V2 legacy: treat as N-tuple with no class name.
                            _store_nstate_elements(sub_prefix, list(sub_tensor))
                        else:
                            logger.error(
                                f"Unsupported sub_tensor format at layer {i} "
                                f"sub {j}: {type(sub_tensor).__name__}"
                            )
                            return False
                elif (
                    isinstance(layer_data, tuple)
                    and len(layer_data) == 2
                    and isinstance(layer_data[0], str)
                    and layer_data[0] in ("__turboquant__", "__turboquant_v2__")
                ):
                    # TurboQuant v2: NamedTuple states (ks, vs)
                    ks, vs = layer_data[1]
                    # Flatten NamedTuple fields into individual tensors
                    tq_tensor_idx = 0
                    for prefix, state in [("k", ks), ("v", vs)]:
                        for field_name in state._fields:
                            val = getattr(state, field_name)
                            if isinstance(val, mx.array):
                                arrays[f"layer_{i}_tq_{prefix}_{field_name}"] = val
                                tq_tensor_idx += 1
                    cache_list_meta[f"layer_{i}_turboquant_v2"] = "1"
                    cache_list_meta[f"layer_{i}_tq_key_type"] = type(ks).__name__
                    cache_list_meta[f"layer_{i}_tq_value_type"] = type(vs).__name__
                    cache_list_meta[f"layer_{i}_tq_key_fields"] = ",".join(ks._fields)
                    cache_list_meta[f"layer_{i}_tq_value_fields"] = ",".join(vs._fields)
                else:
                    # V2 legacy: 2-tuple (keys, values). Upgrade to V3
                    # __nstate__ on disk so all readers see a uniform shape.
                    if not (
                        isinstance(layer_data, (list, tuple)) and len(layer_data) >= 2
                    ):
                        logger.error(
                            f"Unsupported layer_data format at layer {i}: "
                            f"{type(layer_data).__name__}"
                        )
                        return False
                    _store_nstate_elements(f"layer_{i}", list(layer_data))

            block_size = self._expected_block_size or token_count
            # Stamp the depth the block's own TurboQuant layers were packed
            # at (observation), falling back to the manager's expectation
            # only when the block carries no meta_state. A signature must
            # never vouch for a width the payload does not have.
            block_bits = _block_turboquant_bits(layer_cache_types, layer_meta_states)
            cache_signature = _cache_compat_signature(
                model_name=model_name,
                num_layers=len(cache_data),
                block_size=block_size,
                layer_cache_types=layer_cache_types,
                turboquant_kv_bits=(
                    block_bits
                    if block_bits is not None
                    else self._expected_turboquant_kv_bits
                ),
            )

            # Prepare metadata
            metadata = {
                "omlx_cache_format_version": _CACHE_FORMAT_VERSION,
                "block_hash": block_hash.hex(),
                "token_count": str(token_count),
                "num_layers": str(len(cache_data)),
                "model_name": model_name,
                "block_size": str(block_size),
                "cache_signature": cache_signature,
                "created_at": str(time.time()),
            }

            # Add cache type information if provided
            if layer_cache_types:
                metadata["layer_cache_types"] = json.dumps(layer_cache_types)
            if layer_meta_states:
                clamped_meta_states = _clamp_rotating_meta_states(
                    cache_data, layer_cache_types, layer_meta_states
                )
                metadata["layer_meta_states"] = json.dumps(
                    [list(m) if m else [] for m in clamped_meta_states]
                )

            # Merge CacheList sub_count metadata
            metadata.update(cache_list_meta)

            # Last-mile materialization happens in _extract_tensor_bytes.
            # scheduler._cleanup_finished still pre-dispatches real KV arrays,
            # but store_cache creates additional lazy slices, clones, and
            # placeholders here after that collection step. Evaluate those
            # derived arrays before memoryview() so the buffer protocol never
            # becomes the first MLX eval site on the store-cache worker thread.
            # Race history: #978/#1040/#1106/#1437/#1558.
            tensors_raw = {}
            for name, arr in arrays.items():
                tensors_raw[name] = _extract_tensor_bytes(arr)

            # Estimate file size: raw tensor bytes + safetensors header.
            # The header is JSON-encoded per tensor (name + dtype + shape +
            # data_offsets, typically ~85 bytes) plus an 8-byte length prefix
            # and the user metadata block. Compute the metadata-JSON length
            # exactly (large `layer_meta_states` JSON on deep-layer models
            # can exceed a fixed 1 KiB constant) and keep 128 B/tensor as an
            # upper bound on the per-tensor header. The 256 B margin covers
            # the JSON separators / `__metadata__` key envelope safetensors
            # adds at write time.
            try:
                metadata_json_len = len(json.dumps(metadata).encode("utf-8"))
            except (TypeError, ValueError):
                metadata_json_len = 1024
            header_overhead = metadata_json_len + 256 + 128 * len(tensors_raw)
            estimated_size = (
                sum(len(raw) for raw, _, _ in tensors_raw.values()) + header_overhead
            )

            now = time.time()
            block_metadata = PagedSSDBlockMetadata(
                block_hash=block_hash,
                file_path=file_path,
                file_size=estimated_size,
                token_count=token_count,
                created_at=now,
                last_access=now,
                num_layers=len(cache_data),
                model_name=model_name,
                block_size=block_size,
                cache_signature=cache_signature,
                layer_cache_types=layer_cache_types,
                layer_meta_states=layer_meta_states,
            )

            # Store in hot cache (or temporary buffer) for immediate read-back.
            # Uses raw bytes (not mx.array objects) so Metal GPU memory can be
            # released as soon as the inference thread is done with the arrays.
            # NOTE: _promote_to_hot_cache() stores mx.array objects directly
            # because those are freshly loaded from SSD (not active inference),
            # so they don't tie up Metal allocations from the inference pipeline.
            # Storing live inference arrays here would accumulate GPU memory
            # under a large hot cache and cause kernel panics (IOGPUMemory underflow).
            cache_entry = {
                "tensors_raw": tensors_raw,
                "file_metadata": metadata,
                "num_layers": len(cache_data),
                "layer_cache_types": layer_cache_types,
                "block_metadata": block_metadata,
                "dirty": True,
            }

            if self._hot_cache_enabled and (
                hot_cache_write_back or self._hot_cache_only
            ):
                # Write-back mode: store only in hot cache, no SSD index entry.
                # SSD index entry is created later when block is evicted or
                # flushed to SSD (in _enqueue_ssd_write).
                self._hot_cache_put(block_hash, cache_entry)
                self._stats["saves"] += 1
                return True

            if self._hot_cache_only:
                # Hot cache disabled but hot_cache_only set: block is not retained.
                return False

            if self._hot_cache_enabled and not hot_cache_write_back:
                # Pressure write-through: keep the dirty-block durability path
                # but avoid retaining this block as a hot-cache entry.
                ok = self._enqueue_ssd_write(block_hash, cache_entry)
                if ok:
                    self._stats["saves"] += 1
                    logger.debug(
                        f"Enqueued block for SSD write-through: "
                        f"{block_hash.hex()[:16]}..., "
                        f"size={format_bytes(estimated_size)}"
                    )
                return ok

            # Evict LRU blocks to make room for the new block. Done here
            # (post-tensor-build) so the actual block size is known and the
            # cache doesn't oscillate around the configured limit.
            self._enforce_size_limit_for_new_block(estimated_size)

            # SSD path: add to index for SSD file tracking
            self._incompatible_index.remove(block_hash)
            self._index.add(block_metadata)

            # Hot cache disabled: use temporary buffer + immediate SSD write
            with self._hot_cache_lock:
                self._hot_cache[block_hash] = cache_entry

            # Track pending write
            with self._pending_write_hashes_lock:
                self._pending_write_hashes.add(block_hash)

            # Enqueue full file write for background thread. Wait on Full so
            # transient bursts (faster than the writer can drain) don't
            # immediately punch holes in the cache chain.
            try:
                self._write_queue.put(
                    (block_hash, tensors_raw, metadata, file_path),
                    timeout=_PENDING_WRITE_PUT_TIMEOUT_SECONDS,
                )
            except queue.Full:
                self._stats["ssd_inline_write_fallbacks"] += 1
                logger.warning(
                    f"SSD cache write queue saturated (cap={self._max_pending_writes}); "
                    f"writing {block_hash.hex()[:16]} inline"
                )
                ok = self._write_block_file(
                    block_hash,
                    tensors_raw,
                    metadata,
                    file_path,
                    source="inline-fallback",
                )
                self._clear_pending_write(block_hash, remove_hot_cache=True)
                if not ok:
                    return False
                self._stats["saves"] += 1
                return True

            self._stats["saves"] += 1
            logger.debug(
                f"Enqueued block for SSD cache write: {block_hash.hex()[:16]}..., "
                f"size={format_bytes(estimated_size)}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to prepare block for SSD cache: {e}")
            self._stats["errors"] += 1
            return False

    def _reconstruct_cache_data(
        self,
        arrays: dict[str, Any],
        file_metadata: dict[str, str],
        num_layers: int,
        layer_cache_types: list[str] | None = None,
    ) -> list[Any] | None:
        """Reconstruct cache_data list from flattened arrays and metadata.

        Shared helper for load_block(), load_block_with_metadata(), and
        pending-writes read path to avoid code duplication.

        Returns layer_data as one of:
        - ``('__nstate__', class_name, [elem0, elem1, ...])`` — V3 N-tuple.
        - ``('__cache_list__', sub_tensors)`` where each sub_tensor is an
          ``__nstate__`` marker — composite layer.
        - ``('__turboquant_v2__', (ks, vs))`` — TurboQuant payload (unchanged).

        V2 blocks (`layer_{i}_keys` / `layer_{i}_values` keys, no
        ``state_count`` metadata) are read via a polyfill that converts
        them to ``__nstate__`` markers with two elements, so downstream
        code paths see a uniform shape.

        Args:
            arrays: Flattened tensor dict.
            file_metadata: Safetensors metadata dict (string values).
            num_layers: Number of model layers.
            layer_cache_types: Per-layer cache type names.

        Returns:
            Reconstructed cache_data list, or None on error.
        """
        cache_data: list[Any] = []

        # When the on-disk state has exactly two elements (which covers all
        # legacy 2-tuple caches: KVCache, RotatingKVCache, ConcatenateKVCache,
        # ChunkedKVCache, QuantizedKVCache when stored as keys/values), the
        # reconstructed layer is unwrapped to a plain ``(keys, values)``
        # 2-tuple so existing callers (prefix_cache, scheduler, tests) see
        # no change. Real N-tuple caches (PoolingCache, BatchKVCache, ...)
        # surface as ``('__nstate__', class_name, elements)`` markers that
        # downstream code must dispatch on.
        def _maybe_unwrap_legacy(marker: tuple) -> Any:
            _, _, elements = marker
            # Only the legacy plain ``(keys, values)`` shape unwraps. If
            # either element is itself composite (a recursed sub-state), the
            # ``__nstate__`` wrapper must survive so the shape matches save.
            if len(elements) == 2 and not any(
                isinstance(e, (tuple, list)) for e in elements
            ):
                return (elements[0], elements[1])
            return marker

        # Shim; module-level to avoid a recursive-closure refcount
        # cycle pinning `arrays` — see _load_nstate_flat.
        def _load_nstate(prefix: str, fallback_class: str | None) -> tuple | None:
            return _load_nstate_flat(arrays, file_metadata, prefix, fallback_class)

        for i in range(num_layers):
            cache_type = (
                layer_cache_types[i]
                if layer_cache_types and i < len(layer_cache_types)
                else None
            )

            if cache_type == "CacheList":
                sub_count_key = f"layer_{i}_sub_count"
                sub_count = 0
                if file_metadata and sub_count_key in file_metadata:
                    try:
                        sub_count = int(file_metadata[sub_count_key])
                    except (ValueError, TypeError):
                        pass

                if sub_count > 0:
                    sub_tensors: list[Any] = []
                    for j in range(sub_count):
                        sub_marker = _load_nstate(
                            f"layer_{i}_sub_{j}", fallback_class=None
                        )
                        if sub_marker is None:
                            logger.error(
                                f"Missing sub-cache {j} for CacheList layer {i}"
                            )
                            return None
                        # Length-2 sub-states unwrap to (keys, values); longer
                        # N-tuples surface as ``__nstate__`` markers downstream.
                        sub_tensors.append(_maybe_unwrap_legacy(sub_marker))
                    # Preserve the legacy list shape — callers (prefix_cache,
                    # tests) expect ``cache_data[i]`` to be a list of
                    # sub-cache states for CacheList layers, not a wrapper
                    # marker.
                    cache_data.append(sub_tensors)
                else:
                    layer_marker = _load_nstate(f"layer_{i}", fallback_class=cache_type)
                    if layer_marker is None:
                        logger.error(f"Missing N-tuple state for layer {i}")
                        return None
                    cache_data.append(_maybe_unwrap_legacy(layer_marker))
            elif file_metadata and f"layer_{i}_turboquant_v2" in file_metadata:
                # TurboQuant v2: reconstruct NamedTuple states from flattened tensors
                from ..turboquant_kv import (
                    TurboQuantMSEState,
                    TurboQuantPolarProdState,
                    TurboQuantPolarState,
                    TurboQuantProdState,
                    TurboQuantSplitState,
                )

                key_type = file_metadata.get(f"layer_{i}_tq_key_type", "")
                value_type = file_metadata.get(f"layer_{i}_tq_value_type", "")
                key_fields = file_metadata.get(f"layer_{i}_tq_key_fields", "").split(
                    ","
                )
                value_fields = file_metadata.get(
                    f"layer_{i}_tq_value_fields", ""
                ).split(",")
                _type_map = {
                    "TurboQuantMSEState": TurboQuantMSEState,
                    "TurboQuantProdState": TurboQuantProdState,
                    "TurboQuantPolarState": TurboQuantPolarState,
                    "TurboQuantPolarProdState": TurboQuantPolarProdState,
                    "TurboQuantSplitState": TurboQuantSplitState,
                }
                try:
                    k_cls = _type_map[key_type]
                    v_cls = _type_map[value_type]
                    k_tensors = [arrays[f"layer_{i}_tq_k_{f}"] for f in key_fields]
                    v_tensors = [arrays[f"layer_{i}_tq_v_{f}"] for f in value_fields]
                    ks = k_cls(*k_tensors)
                    vs = v_cls(*v_tensors)
                    cache_data.append(("__turboquant_v2__", (ks, vs)))
                except (KeyError, TypeError) as e:
                    logger.error(f"TurboQuant v2 layer {i}: reconstruction failed: {e}")
                    return None
            else:
                # Standard cache layer (KVCache, RotatingKVCache,
                # PoolingCache, ...). V3 stores all state elements as
                # ``layer_{i}_state_{k}``; V2 polyfill reads the legacy
                # ``layer_{i}_keys`` / ``layer_{i}_values`` 2-tuple shape.
                # Length-2 markers unwrap to ``(keys, values)`` for legacy
                # caller compatibility; longer N-tuples (PoolingCache etc.)
                # propagate as ``__nstate__`` markers.
                layer_marker = _load_nstate(f"layer_{i}", fallback_class=cache_type)
                if layer_marker is None:
                    logger.error(f"Missing N-tuple state for layer {i}")
                    return None
                cache_data.append(_maybe_unwrap_legacy(layer_marker))

        return cache_data

    @staticmethod
    def _arrays_from_tensors_raw(
        tensors_raw: dict[str, tuple[bytes, str, list[int]]],
    ) -> dict[str, mx.array]:
        """Convert raw bytes dict back to mx.array dict for _reconstruct_cache_data.

        Args:
            tensors_raw: Dict of {name: (raw_bytes, dtype_str, shape)}.

        Returns:
            Dict of {name: mx.array} with correct dtypes and shapes.
        """
        arrays = {}
        for name, (raw, dtype_str, shape) in tensors_raw.items():
            arrays[name] = _restore_tensor_from_bytes(raw, dtype_str, shape)
        return arrays

    def load_block(self, block_hash: bytes) -> list[Any] | None:
        """
        Load a KV cache block from SSD storage.

        Checks pending writes first (in-memory, no I/O), then falls back to disk
        read with a timeout to prevent inference deadlocks.

        Args:
            block_hash: Content hash for the block.

        Returns:
            List of per-layer data, or None if not found/timed out.
            Each element is either:
            - (keys, values) tuple for standard caches
            - List[Tuple[keys, values]] for CacheList layers
        """
        if not HAS_MLX:
            logger.error("MLX not available, cannot load block")
            return None

        # Check hot cache first (in-memory, no I/O)
        entry = self._hot_cache_get(block_hash)
        if entry is not None:
            # Entries from _promote_to_hot_cache() store mx.array objects directly
            # (safe — they come from SSD loads, not active inference).
            # Entries from save_block() use tensors_raw (raw bytes).
            arrays = entry.get("arrays") or self._arrays_from_tensors_raw(
                entry["tensors_raw"]
            )
            cache_data = self._reconstruct_cache_data(
                arrays,
                entry["file_metadata"],
                entry["num_layers"],
                entry["layer_cache_types"],
            )
            if cache_data is not None:
                self._index.touch(block_hash)
                self._stats["loads"] += 1
                self._stats["hits"] += 1
                self._stats["hot_cache_hits"] += 1
                logger.debug(f"Loaded block from hot cache: {block_hash.hex()[:16]}...")
            return cache_data

        # Check pending-write buffer (evicted from hot cache, SSD write in progress)
        entry = self._pending_write_buffer_get(block_hash)
        if entry is not None:
            arrays = entry.get("arrays") or self._arrays_from_tensors_raw(
                entry["tensors_raw"]
            )
            cache_data = self._reconstruct_cache_data(
                arrays,
                entry["file_metadata"],
                entry["num_layers"],
                entry["layer_cache_types"],
            )
            if cache_data is not None:
                self._index.touch(block_hash)
                self._stats["loads"] += 1
                self._stats["hits"] += 1
                self._stats["hot_cache_hits"] += 1
                logger.debug(
                    f"Loaded block from pending write buffer: "
                    f"{block_hash.hex()[:16]}..."
                )
            return cache_data

        # Check index
        metadata = self._index.get(block_hash)
        if metadata is None:
            self._stats["misses"] += 1
            return None

        file_path = metadata.file_path

        if not file_path.exists():
            logger.warning(f"SSD cache file missing: {file_path}")
            self._index.remove(block_hash)
            self._stats["misses"] += 1
            return None

        try:
            # Load directly on the inference thread (Metal-safe).
            # SSD read for a ~10MB block takes ~2ms @ 5GB/s — negligible.
            # Previous executor-based approach caused deadlocks when
            # mx.load() in a worker thread contested Metal GPU resources
            # with the main inference thread.
            try:
                arrays, file_metadata = mx.load(str(file_path), return_metadata=True)
            except FileNotFoundError:
                # Concurrent evictor unlinked the file between the
                # exists() check above and this load. Treat as a miss
                # and prune the stale index entry.
                self._index.remove(block_hash)
                self._stats["misses"] += 1
                return None

            # Defensive: even if the index is stale (e.g. from a previous
            # run that pre-dates the format version field), reject blocks
            # without a readable version marker before they can poison
            # the hot cache or downstream merge logic.
            if (
                file_metadata
                and file_metadata.get("omlx_cache_format_version")
                not in _READABLE_CACHE_FORMAT_VERSIONS
            ):
                self._index.remove(block_hash)
                self._stats["misses"] += 1
                return None

            # Get layer_cache_types for CacheList detection
            layer_cache_types = metadata.layer_cache_types
            if (
                not layer_cache_types
                and file_metadata
                and "layer_cache_types" in file_metadata
            ):
                try:
                    layer_cache_types = json.loads(file_metadata["layer_cache_types"])
                except (json.JSONDecodeError, TypeError):
                    layer_cache_types = None

            cache_data = self._reconstruct_cache_data(
                arrays,
                file_metadata,
                metadata.num_layers,
                layer_cache_types,
            )
            if cache_data is None:
                return None

            # Update access time
            self._index.touch(block_hash)
            self._stats["loads"] += 1
            self._stats["hits"] += 1

            # Promote to hot cache for faster access next time
            if self._hot_cache_enabled:
                self._promote_to_hot_cache(block_hash, arrays, file_metadata, metadata)

            logger.debug(f"Loaded block from SSD cache: {block_hash.hex()[:16]}...")
            return cache_data

        except Exception as e:
            logger.error(f"Failed to load block from SSD cache: {e}")
            self._stats["errors"] += 1
            # Remove corrupted entry
            self._index.remove(block_hash)
            try:
                file_path.unlink()
            except Exception:
                pass
            return None

    def load_block_with_metadata(
        self,
        block_hash: bytes,
        promote_to_hot_cache: bool = True,
    ) -> tuple[list[Any] | None, dict[str, Any] | None]:
        """
        Load a KV cache block with its metadata from SSD storage.

        Checks pending writes first (zero I/O), then falls back to disk
        read with a timeout to prevent inference deadlocks.

        Args:
            block_hash: Content hash for the block.
            promote_to_hot_cache: When False, do not retain SSD-loaded data in
                the hot cache after reconstructing it for the active request.

        Returns:
            Tuple of (cache_data, metadata_dict) where:
            - cache_data: List of per-layer data, or None.
              Each element is either (keys, values) or List[Tuple[keys, values]]
              for CacheList layers.
            - metadata_dict: Dictionary with cache type info, or None
              {
                  "layer_cache_types": List[str],  # per-layer type names
                  "layer_meta_states": List[Tuple],  # per-layer meta states
                  "num_layers": int,
                  "token_count": int,
              }
        """
        if not HAS_MLX:
            logger.error("MLX not available, cannot load block")
            return None, None

        # Check hot cache first (in-memory, no I/O)
        entry = self._hot_cache_get(block_hash)
        if entry is not None:
            blk_meta = entry["block_metadata"]
            arrays = entry.get("arrays") or self._arrays_from_tensors_raw(
                entry["tensors_raw"]
            )
            cache_data = self._reconstruct_cache_data(
                arrays,
                entry["file_metadata"],
                entry["num_layers"],
                entry["layer_cache_types"],
            )
            if cache_data is None:
                return None, None

            metadata_dict = {
                "num_layers": entry["num_layers"],
                "token_count": blk_meta.token_count,
                "model_name": blk_meta.model_name,
                "block_size": blk_meta.block_size,
                "cache_signature": blk_meta.cache_signature,
                "layer_cache_types": entry["layer_cache_types"],
                "layer_meta_states": blk_meta.layer_meta_states,
            }

            self._index.touch(block_hash)
            self._stats["loads"] += 1
            self._stats["hits"] += 1
            self._stats["hot_cache_hits"] += 1
            logger.debug(
                f"Loaded block with metadata from hot cache: "
                f"{block_hash.hex()[:16]}..."
            )
            return cache_data, metadata_dict

        # Check pending-write buffer (evicted from hot cache, SSD write in progress)
        entry = self._pending_write_buffer_get(block_hash)
        if entry is not None:
            blk_meta = entry["block_metadata"]
            arrays = entry.get("arrays") or self._arrays_from_tensors_raw(
                entry["tensors_raw"]
            )
            cache_data = self._reconstruct_cache_data(
                arrays,
                entry["file_metadata"],
                entry["num_layers"],
                entry["layer_cache_types"],
            )
            if cache_data is None:
                return None, None

            metadata_dict = {
                "num_layers": entry["num_layers"],
                "token_count": blk_meta.token_count,
                "model_name": blk_meta.model_name,
                "block_size": blk_meta.block_size,
                "cache_signature": blk_meta.cache_signature,
                "layer_cache_types": entry["layer_cache_types"],
                "layer_meta_states": blk_meta.layer_meta_states,
            }

            self._index.touch(block_hash)
            self._stats["loads"] += 1
            self._stats["hits"] += 1
            self._stats["hot_cache_hits"] += 1
            logger.debug(
                f"Loaded block with metadata from pending write buffer: "
                f"{block_hash.hex()[:16]}..."
            )
            return cache_data, metadata_dict

        # Check index
        block_metadata = self._index.get(block_hash)
        if block_metadata is None:
            self._stats["misses"] += 1
            return None, None

        file_path = block_metadata.file_path

        if not file_path.exists():
            logger.warning(f"SSD cache file missing: {file_path}")
            self._index.remove(block_hash)
            self._stats["misses"] += 1
            return None, None

        try:
            # Load directly on the inference thread (Metal-safe).
            # See load_block() for rationale on removing the executor.
            arrays, file_metadata = mx.load(str(file_path), return_metadata=True)

            # Defensive version check, mirrors load_block().
            if (
                file_metadata
                and file_metadata.get("omlx_cache_format_version")
                not in _READABLE_CACHE_FORMAT_VERSIONS
            ):
                self._index.remove(block_hash)
                self._stats["misses"] += 1
                return None, None

            # Parse layer_cache_types early for CacheList detection
            layer_cache_types = block_metadata.layer_cache_types
            if (
                not layer_cache_types
                and file_metadata
                and "layer_cache_types" in file_metadata
            ):
                try:
                    layer_cache_types = json.loads(file_metadata["layer_cache_types"])
                except (json.JSONDecodeError, TypeError):
                    layer_cache_types = None

            cache_data = self._reconstruct_cache_data(
                arrays,
                file_metadata,
                block_metadata.num_layers,
                layer_cache_types,
            )
            if cache_data is None:
                return None, None

            # Build metadata dict for reconstruction
            metadata_dict = {
                "num_layers": block_metadata.num_layers,
                "token_count": block_metadata.token_count,
                "model_name": block_metadata.model_name,
                "block_size": block_metadata.block_size,
                "cache_signature": block_metadata.cache_signature,
                "layer_cache_types": layer_cache_types,
                "layer_meta_states": block_metadata.layer_meta_states,
            }

            if not metadata_dict["layer_meta_states"] and file_metadata:
                if "layer_meta_states" in file_metadata:
                    try:
                        raw = json.loads(file_metadata["layer_meta_states"])
                        metadata_dict["layer_meta_states"] = [
                            tuple(m) if m else () for m in raw
                        ]
                    except (json.JSONDecodeError, TypeError):
                        pass

            # Update access time
            self._index.touch(block_hash)
            self._stats["loads"] += 1
            self._stats["hits"] += 1

            # Promote to hot cache for faster access next time
            if self._hot_cache_enabled and promote_to_hot_cache:
                self._promote_to_hot_cache(
                    block_hash, arrays, file_metadata, block_metadata
                )

            logger.debug(
                f"Loaded block with metadata from SSD cache: {block_hash.hex()[:16]}..."
            )
            return cache_data, metadata_dict

        except Exception as e:
            logger.error(f"Failed to load block from SSD cache: {e}")
            self._stats["errors"] += 1
            # Remove corrupted entry
            self._index.remove(block_hash)
            try:
                file_path.unlink()
            except Exception:
                pass
            return None, None

    def get_block_metadata(self, block_hash: bytes) -> PagedSSDBlockMetadata | None:
        """
        Get metadata for a block without loading the data.

        Args:
            block_hash: Content hash for the block.

        Returns:
            PagedSSDBlockMetadata if found, None otherwise.
        """
        return self._index.get(block_hash)

    def has_block(self, block_hash: bytes) -> bool:
        """
        Check if a block exists in cache (hot cache, pending writes, or SSD storage).

        Args:
            block_hash: Content hash for the block.

        Returns:
            True if block exists in hot cache, pending write buffer, or SSD index.
        """
        if self._index.contains(block_hash):
            return True
        # Block may have been evicted from SSD index but still in hot cache
        with self._hot_cache_lock:
            if block_hash in self._hot_cache:
                return True
        # Block may be evicted from hot cache and awaiting SSD write
        with self._pending_write_hashes_lock:
            if block_hash in self._pending_write_buffers:
                return True
        return False

    def preload_matched_blocks(self, block_hashes: list[bytes]) -> int:
        """
        Parallel-load matched blocks from SSD into hot cache.

        For cold-start optimization: loads blocks that exist on SSD but not
        in hot cache, using parallel I/O. After preload, subsequent
        load_block() / load_block_with_metadata() calls hit hot cache (~0ms)
        instead of SSD (~2ms per block).

        Individual block failures are non-fatal (logged and skipped).

        Args:
            block_hashes: Block hashes confirmed as cache hits.

        Returns:
            Number of blocks successfully loaded into hot cache.
        """
        if not self._hot_cache_enabled:
            return 0

        if not HAS_MLX:
            return 0

        # Filter to blocks that need loading: in SSD index but not hot cache
        to_load = []
        for bh in block_hashes:
            metadata = self._index.get(bh)
            if metadata is None:
                continue
            if self._hot_cache_get(bh) is not None:
                continue
            to_load.append((bh, metadata))

        if len(to_load) < 4:
            return 0

        # Guard: don't preload more than available hot cache capacity.
        # If we preload N blocks but hot cache can only hold M < N,
        # blocks evict each other and reconstruct_cache falls back to SSD.
        # CPD-accepted (GLM L1).
        available = self._hot_cache_available_bytes()
        if available <= 0:
            return 0
        capped_to_load: list[tuple[bytes, PagedSSDBlockMetadata]] = []
        selected_bytes = 0
        for bh, metadata in to_load:
            try:
                block_bytes = max(0, int(getattr(metadata, "file_size", 0) or 0))
            except (TypeError, ValueError):
                block_bytes = 0
            if block_bytes <= 0:
                try:
                    block_bytes = metadata.file_path.stat().st_size
                except OSError:
                    block_bytes = 0
            if selected_bytes + block_bytes > available:
                break
            capped_to_load.append((bh, metadata))
            selected_bytes += block_bytes
        to_load = capped_to_load
        if len(to_load) < 4:
            return 0

        # Cap workers to limit peak memory (each load allocates ~122-275MB).
        # 8 workers ≈ 1.4GB peak, vs 2.8GB at 16. CPD-accepted (G1/Q3).
        start = time.perf_counter()
        loaded_count = 0
        max_workers = min(8, len(to_load))

        def _load_one(block_hash: bytes, metadata: PagedSSDBlockMetadata) -> bool:
            file_path = metadata.file_path
            if not file_path.exists():
                return False
            try:
                arrays, file_metadata = mx.load(str(file_path), return_metadata=True)
                if (
                    file_metadata
                    and file_metadata.get("omlx_cache_format_version")
                    not in _READABLE_CACHE_FORMAT_VERSIONS
                ):
                    return False
                self._promote_to_hot_cache(block_hash, arrays, file_metadata, metadata)
                return True
            except Exception as e:
                logger.warning(f"Preload failed for block {block_hash.hex()[:16]}: {e}")
                return False

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_load_one, bh, meta): bh for bh, meta in to_load}
            for future in as_completed(futures):
                try:
                    if future.result():
                        loaded_count += 1
                except Exception:
                    pass

        elapsed_ms = (time.perf_counter() - start) * 1000
        self._stats["preload_calls"] += 1
        self._stats["preload_blocks_loaded"] += loaded_count
        self._stats["preload_time_ms"] += elapsed_ms

        if loaded_count > 0:
            logger.info(
                f"Preloaded {loaded_count}/{len(to_load)} blocks into hot cache "
                f"(workers={max_workers}, time={elapsed_ms:.1f}ms)"
            )
        return loaded_count

    def adopt_layer_signature_if_unset(
        self, layer_cache_types: list[str] | None
    ) -> bool:
        """Adopt ``layer_cache_types`` as the expected signature if none was set.

        The scheduler may not be able to derive the post-patch cache layout
        before constructing this manager (TurboQuant / MTP / dtype changes
        happen at model-load time). Save sites pass the live signature on
        every call, so the manager can adopt it the first time it sees one.

        Returns True when adoption actually happened (caller can use this
        to trigger the one-shot sweep). Returns False when the manager
        already had a signature or when ``layer_cache_types`` is empty.
        """
        if not layer_cache_types:
            return False
        if self._expected_layer_cache_types is not None:
            return False
        canonical = _canonicalize_layer_cache_types(layer_cache_types)
        with self._lock:
            if self._expected_layer_cache_types is not None:
                return False  # raced with another adopter
            self._expected_layer_cache_types = list(layer_cache_types)
            self._signature_sweep_completed = False
        logger.info(
            "PagedSSDCacheManager adopted layer cache signature "
            "(%d layers, %d unique types)",
            len(layer_cache_types),
            len(set(canonical or ())),
        )
        return True

    def set_expected_layer_signature(
        self,
        layer_cache_types: list[str] | None,
        *,
        turboquant_kv_bits: float | None = None,
    ) -> bool:
        """Set the live layer-cache signature, replacing stale expectations.

        Unlike ``adopt_layer_signature_if_unset``, this is used by callers that
        learn the final cache layout after manager construction (for example
        TurboQuant settings applied by the engine after the scheduler starts).

        ``turboquant_kv_bits`` is the live TurboQuant bit depth (None when
        TurboQuant is inactive). A bit-depth change alone also triggers the
        sweep: blocks written at another depth have a different packed state
        width and would crash batch concatenation if mixed (#2045).

        Returns True when the canonical signature changed and a stale-signature
        sweep should run. Returns False for empty input or a canonical no-op.
        """
        if not layer_cache_types:
            return False

        new_signature = list(layer_cache_types)
        new_canonical = _canonicalize_layer_cache_types(new_signature)
        new_bits = (
            float(turboquant_kv_bits) if turboquant_kv_bits is not None else None
        )

        with self._lock:
            old_signature = self._expected_layer_cache_types
            old_canonical = _canonicalize_layer_cache_types(old_signature)
            bits_changed = new_bits != self._expected_turboquant_kv_bits
            if old_canonical == new_canonical and not bits_changed:
                if old_signature != new_signature:
                    self._expected_layer_cache_types = new_signature
                return False

            self._expected_layer_cache_types = new_signature
            self._expected_turboquant_kv_bits = new_bits
            self._signature_sweep_completed = False

        logger.info(
            "PagedSSDCacheManager updated layer cache signature "
            "(%d layers, %d unique types, turboquant_kv_bits=%s)",
            len(new_signature),
            len(set(new_canonical or ())),
            new_bits,
        )
        return True

    def invalidate_stale_layer_signature(self) -> int:
        """Drop in-memory index entries whose layer_cache_types — or, when a
        TurboQuant depth is expected, whose recorded bit depth — disagree
        with the current expected signature.

        Scoped to the current ``_expected_model_name``: blocks belonging to
        other models share the SSD directory and remain valid for them, so
        we leave them alone. Legacy blocks without a recorded ``model_name``
        are also skipped — we cannot safely attribute them.

        The SSD files are unlinked from the index (and any in-memory hot
        copy is dropped), but the on-disk file is left for LRU to reclaim
        later. Returns the number of blocks dropped from the index.

        Idempotent: a second call after a clean sweep returns 0.
        """
        if self._expected_layer_cache_types is None:
            return 0
        if not self._expected_model_name:
            # Without an owning model_name we cannot scope safely; refuse
            # rather than risk evicting another model's blocks.
            return 0
        if self._signature_sweep_completed:
            return 0

        expected = _canonicalize_layer_cache_types(self._expected_layer_cache_types)
        expects_bits = self._expected_turboquant_kv_bits is not None

        with self._index._lock:
            stale: list[bytes] = []
            for h, meta in self._index._index.items():
                if not meta.model_name or meta.model_name != self._expected_model_name:
                    continue
                got = _canonicalize_layer_cache_types(meta.layer_cache_types)
                if got is None:
                    # Pre-signature blocks lack the metadata to judge the
                    # layout. Without a depth expectation, skip rather than
                    # guess — newer saves will replace them. With one, the
                    # block can no more prove its packed width than its
                    # layout, so it is unsafe to keep (see
                    # _signature_bits_match).
                    if expects_bits:
                        stale.append(h)
                    continue
                if got != expected:
                    stale.append(h)
                    continue
                if not self._signature_bits_match(meta.cache_signature):
                    stale.append(h)

        for h in stale:
            self.forget_block(h)

        self._signature_sweep_completed = True

        if stale:
            logger.info(
                "Invalidated %d SSD index entries with stale layer "
                "cache signature for model %r (kept %d)",
                len(stale),
                self._expected_model_name,
                len(self._index._index),
            )
        return len(stale)

    def forget_block(self, block_hash: bytes) -> bool:
        """
        Remove a block from this manager's in-memory indexes without deleting
        its SSD file.

        Used when a prefix entry points at a block that is incompatible with
        the current model/layout. The file may still be valid for another
        model sharing the same cache directory.
        """
        with self._lock:
            removed = self._hot_cache_remove(block_hash) is not None

            with self._pending_write_hashes_lock:
                if block_hash in self._pending_write_buffers:
                    removed = True
                self._pending_write_buffers.pop(block_hash, None)
                self._pending_write_hashes.discard(block_hash)

            metadata = self._index.remove(block_hash)
            if metadata is not None:
                self._incompatible_index.add(metadata)
                removed = True

            return removed

    def delete_block(self, block_hash: bytes) -> bool:
        """
        Delete a block from SSD storage.

        Args:
            block_hash: Content hash for the block.

        Returns:
            True if deleted successfully.
        """
        with self._lock:
            # Also remove from hot cache
            self._hot_cache_remove(block_hash)

            # Also remove from pending write buffer
            with self._pending_write_hashes_lock:
                self._pending_write_buffers.pop(block_hash, None)
                self._pending_write_hashes.discard(block_hash)

            metadata = self._index.remove(block_hash)
            incompatible_metadata = self._incompatible_index.remove(block_hash)
            if metadata is None:
                metadata = incompatible_metadata
            if metadata is None:
                return False

            try:
                if metadata.file_path.exists():
                    metadata.file_path.unlink()
                    logger.debug(f"Deleted SSD cache file: {metadata.file_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete SSD cache file: {e}")
                return False

    # Use at most 99% of available disk space to avoid filling disk completely
    _DISK_SAFE_RATIO = 0.99

    def _get_effective_max_size(self) -> int:
        """Get effective max size considering actual disk free space.

        Returns the minimum of configured max_size and 99% of disk space
        available for cache (current cache size + disk free). This ensures
        eviction triggers before the disk fills up even when other processes
        consume disk space after the server started.

        Uses a 30-second TTL cache for shutil.disk_usage() results.
        """
        if self._cache_dir is None:
            return self._max_size

        # Take the lock so a concurrent writer-thread invalidation
        # (sets _disk_usage_cache=None on ENOSPC) can't interleave with
        # this read-check-write and let one save see a fresh value paired
        # with a stale timestamp (or vice versa).
        now = time.monotonic()
        with self._lock:
            if (
                self._disk_usage_cache is None
                or now - self._disk_usage_cache_time > 30.0
            ):
                try:
                    self._disk_usage_cache = shutil.disk_usage(self._cache_dir)
                except OSError as e:
                    logger.warning(
                        f"Failed to check disk usage for SSD cache dir "
                        f"{self._cache_dir}: {e}"
                    )
                    return self._max_size
                self._disk_usage_cache_time = now
            disk_free = self._disk_usage_cache.free

        disk_available = self._tracked_ssd_size() + disk_free
        disk_limit = int(disk_available * self._DISK_SAFE_RATIO)
        return min(self._max_size, disk_limit)

    def _evict_tracked_until_size(
        self,
        target_size: int,
        max_count: int | None = None,
    ) -> list[tuple[PagedSSDCacheIndex, PagedSSDBlockMetadata]]:
        """Remove oldest tracked SSD entries from their indexes until target."""
        evicted: list[tuple[PagedSSDCacheIndex, PagedSSDBlockMetadata]] = []

        while self._tracked_ssd_size() > target_size:
            if max_count is not None and len(evicted) >= max_count:
                break

            compatible = self._index.get_lru_entries(1)
            incompatible = self._incompatible_index.get_lru_entries(1)
            if not compatible and not incompatible:
                break

            if compatible and incompatible:
                if incompatible[0].last_access <= compatible[0].last_access:
                    source_index = self._incompatible_index
                    candidate = incompatible[0]
                else:
                    source_index = self._index
                    candidate = compatible[0]
            elif incompatible:
                source_index = self._incompatible_index
                candidate = incompatible[0]
            else:
                source_index = self._index
                candidate = compatible[0]

            metadata = source_index.remove(candidate.block_hash)
            if metadata is not None:
                evicted.append((source_index, metadata))

        return evicted

    def _enforce_size_limit_for_new_block(
        self,
        estimated_new_size: int = 1 * 1024 * 1024,
        *,
        max_unlinks: int | None = None,
        unbounded: bool = False,
    ) -> None:
        """Enforce size limit before adding a new block.

        ``estimated_new_size`` should be the actual byte size of the block
        about to be inserted. The 1 MiB default is for callers that don't
        yet know the size at the time eviction is needed; passing the
        actual size avoids cache oscillation around the configured limit.
        """
        effective_max = self._get_effective_max_size()

        # Warn when disk pressure shrinks effective limit well below configured
        # (throttled to once per 60s to avoid log spam)
        if effective_max < self._max_size * 0.1:
            now = time.monotonic()
            if now - self._last_disk_pressure_warn > 60.0:
                self._last_disk_pressure_warn = now
                logger.warning(
                    f"SSD cache disk pressure: effective limit "
                    f"{format_bytes(effective_max)} "
                    f"(configured {format_bytes(self._max_size)}), "
                    f"disk nearly full"
                )
        target_size = effective_max - estimated_new_size
        if target_size < 0:
            target_size = int(effective_max * 0.9)

        max_count = None if unbounded else max_unlinks
        if max_count is None and not unbounded:
            max_count = _MAX_INLINE_UNLINKS_PER_SAVE

        if self._tracked_ssd_size() > target_size:
            evicted = self._evict_tracked_until_size(
                target_size,
                max_count=max_count,
            )
            # Inline unlinks on the calling thread. Eviction typically returns
            # a single entry per save because the tracked LRU walk stops as
            # soon as the shared SSD budget is back under target. Inline
            # removes bounded-queue contention entirely. Hot cache is NOT
            # touched here — ``delete_block()`` is the only path that clears
            # both tiers.
            #
            # Bounded inline burst. The ENOSPC-recovery path invalidates the
            # 30 s disk-usage cache, which can shrink the next
            # ``_get_effective_max_size`` call sharply. Cap the burst at
            # ``_MAX_INLINE_UNLINKS_PER_SAVE`` and leave remaining LRU
            # entries in their indexes so subsequent saves drain the rest.
            # Bounds per-call latency at the cost of taking multiple saves
            # to fully reconverge.
            for source_index, metadata in evicted:
                self._unlink_evicted(metadata, source_index)
            if max_count is not None and len(evicted) >= max_count:
                logger.debug(
                    f"Inline eviction capped at {max_count} entries; "
                    f"{self._tracked_ssd_size() - target_size} bytes remain "
                    f"above target for subsequent saves to drain"
                )

    def enforce_size_limit(self) -> int:
        """
        Enforce SSD cache size limit by evicting LRU files.

        Returns:
            Number of bytes freed.
        """
        # Decide what to evict under the lock, but perform unlinks outside
        # it: a single unlink on a slow disk (NFS / encrypted FS / ENOSPC
        # retry path) can block tens to hundreds of ms, and every
        # _get_effective_max_size() / writer-thread cache-invalidation
        # contends on self._lock. The index has its own internal lock
        # protecting the LRU/size accounting.
        with self._lock:
            initial_size = self._tracked_ssd_size()
            effective_max = self._get_effective_max_size()

            if initial_size <= effective_max:
                return 0

            target_size = int(effective_max * 0.9)  # 90% of effective max
            evicted = self._evict_tracked_until_size(target_size)

        # Do NOT remove from hot cache — see _enforce_size_limit_for_new_block
        for source_index, metadata in evicted:
            self._unlink_evicted(metadata, source_index)

        freed = initial_size - self._tracked_ssd_size()
        logger.info(
            f"SSD cache size enforcement: freed {format_bytes(freed)}, "
            f"evicted {len(evicted)} files"
        )
        return freed

    def _unlink_evicted(
        self,
        metadata: PagedSSDBlockMetadata,
        source_index: PagedSSDCacheIndex | None = None,
    ) -> None:
        """Delete an evicted block file from disk.

        On unlink failure other than FileNotFoundError, re-add the
        metadata to the index so ``total_size`` keeps reflecting actual
        on-disk bytes; without this, accumulated failures would let the
        cache silently exceed ``max_size`` (the index would report free
        space that does not exist on disk).
        """
        try:
            metadata.file_path.unlink(missing_ok=True)
            self._stats["evictions"] += 1
        except OSError as e:
            restore_index = source_index or self._index
            # Restore the index entry so total_size matches disk reality.
            # The re-added entry lands at the LRU tail (most-recently
            # touched), which deprioritises immediate re-eviction.
            restore_index.add(metadata)
            self._stats["evict_unlink_failures"] += 1
            logger.exception(
                "Failed to delete evicted SSD cache file %s: %s",
                metadata.file_path,
                e,
            )

    def clear_hot_cache(self) -> int:
        """Clear all in-memory (hot) cache entries.

        Returns:
            Number of entries cleared.
        """
        with self._hot_cache_lock:
            count = len(self._hot_cache)
            self._hot_cache.clear()
            self._hot_cache_total_bytes = 0
        if self._hot_cache_budget is not None:
            self._hot_cache_budget.forget_owner(self)
        if count:
            logger.info("Cleared %d hot cache entries", count)
        return count

    def shrink_hot_cache_to(
        self,
        target_bytes: int,
        protected_hashes: set[bytes] | None = None,
    ) -> int:
        """Shrink this manager's hot cache to ``target_bytes`` by local LRU."""
        target_bytes = max(0, int(target_bytes))
        protected_hashes = protected_hashes or set()

        if self._hot_cache_budget is not None:
            return self._hot_cache_budget.shrink_to(
                target_bytes, protected_hashes=protected_hashes
            )

        evicted_entries: list[tuple[bytes, dict, int]] = []
        with self._hot_cache_lock:
            while self._hot_cache_total_bytes > target_bytes and self._hot_cache:
                victim_hash = None
                for block_hash in self._hot_cache:
                    if block_hash not in protected_hashes:
                        victim_hash = block_hash
                        break
                if victim_hash is None:
                    break

                evicted = self._hot_cache.pop(victim_hash)
                size = self._hot_cache_entry_size(evicted)
                self._hot_cache_total_bytes = max(0, self._hot_cache_total_bytes - size)
                evicted_entries.append((victim_hash, evicted, size))

        freed = 0
        for block_hash, evicted, size in evicted_entries:
            freed += size
            self._handle_hot_cache_eviction(block_hash, evicted)

        if freed and self._hot_cache_only:
            logger.warning(
                "Shrank hot-cache-only tier by %s; evicted chains are not "
                "persisted to SSD",
                format_bytes(freed),
            )
        elif freed:
            logger.info("Shrank hot cache by %s", format_bytes(freed))
        return freed

    def clear(self) -> int:
        """
        Clear all SSD cache files.

        Returns:
            Number of files deleted.
        """
        with self._lock:
            count = 0
            block_hashes = (
                self._index.get_all_hashes() + self._incompatible_index.get_all_hashes()
            )
            for block_hash in dict.fromkeys(block_hashes):
                if self.delete_block(block_hash):
                    count += 1

            logger.info(f"Cleared SSD cache: deleted {count} files")
            return count

    def get_stats(self) -> PagedSSDCacheStats:
        """
        Get SSD cache statistics.

        Returns:
            PagedSSDCacheStats with cache metrics.
        """
        with self._lock:
            with self._hot_cache_lock:
                hot_entries = len(self._hot_cache)
                hot_size = self._hot_cache_total_bytes
            return PagedSSDCacheStats(
                hits=self._stats["hits"],
                misses=self._stats["misses"],
                evictions=self._stats["evictions"],
                saves=self._stats["saves"],
                saves_persisted=self._stats["saves_persisted"],
                loads=self._stats["loads"],
                errors=self._stats["errors"],
                evict_unlink_failures=self._stats["evict_unlink_failures"],
                total_size_bytes=self._tracked_ssd_size(),
                max_size_bytes=self._get_effective_max_size(),
                configured_max_size_bytes=self._max_size,
                num_files=self._tracked_ssd_count(),
                hot_cache_entries=hot_entries,
                hot_cache_size_bytes=hot_size,
                hot_cache_max_bytes=self._effective_hot_cache_max_bytes(),
                hot_cache_hits=self._stats["hot_cache_hits"],
                hot_cache_evictions=self._stats["hot_cache_evictions"],
                hot_cache_promotions=self._stats["hot_cache_promotions"],
                ssd_write_drops=self._stats["ssd_write_drops"],
                ssd_inline_write_fallbacks=self._stats["ssd_inline_write_fallbacks"],
            )

    def get_stats_for_model(self, model_name: str) -> PagedSSDCacheStats:
        """Get model-scoped SSD cache statistics.

        The SSD cache directory can be shared across multiple loaded models, so
        dashboard per-model rows must be filtered by block metadata rather than
        reusing the global cache totals.
        """
        normalized_name = model_name.rstrip("/")
        basename = os.path.basename(normalized_name) if normalized_name else ""

        def _matches(candidate: str) -> bool:
            candidate = candidate.rstrip("/")
            if not candidate:
                return False
            if candidate == normalized_name:
                return True
            if basename and os.path.basename(candidate) == basename:
                return True
            return False

        with self._lock:
            indexed_entries = [
                metadata
                for metadata in self._index.get_all_metadata()
                if _matches(metadata.model_name)
            ]
            indexed_size = sum(metadata.file_size for metadata in indexed_entries)
            indexed_count = len(indexed_entries)

            with self._hot_cache_lock:
                hot_entries = []
                hot_size = 0
                for entry in self._hot_cache.values():
                    blk_meta = entry.get("block_metadata")
                    if blk_meta is None or not _matches(blk_meta.model_name):
                        continue
                    hot_entries.append(entry)
                    hot_size += self._hot_cache_entry_size(entry)

            return PagedSSDCacheStats(
                hits=self._stats["hits"],
                misses=self._stats["misses"],
                evictions=self._stats["evictions"],
                saves=self._stats["saves"],
                saves_persisted=self._stats["saves_persisted"],
                loads=self._stats["loads"],
                errors=self._stats["errors"],
                evict_unlink_failures=self._stats["evict_unlink_failures"],
                total_size_bytes=indexed_size,
                max_size_bytes=self._get_effective_max_size(),
                configured_max_size_bytes=self._max_size,
                num_files=indexed_count,
                hot_cache_entries=len(hot_entries),
                hot_cache_size_bytes=hot_size,
                hot_cache_max_bytes=self._effective_hot_cache_max_bytes(),
                hot_cache_hits=self._stats["hot_cache_hits"],
                hot_cache_evictions=self._stats["hot_cache_evictions"],
                hot_cache_promotions=self._stats["hot_cache_promotions"],
                ssd_write_drops=self._stats["ssd_write_drops"],
                ssd_inline_write_fallbacks=self._stats["ssd_inline_write_fallbacks"],
            )

    def get_stats_dict(self) -> dict[str, Any]:
        """
        Get SSD cache statistics as a dictionary.

        This method provides the legacy dictionary format for compatibility.

        Returns:
            Dictionary with cache statistics.
        """
        with self._lock:
            with self._hot_cache_lock:
                hot_entries = len(self._hot_cache)
                hot_size = self._hot_cache_total_bytes
            effective_max = self._get_effective_max_size()
            return {
                "cache_dir": str(self._cache_dir) if self._cache_dir else "None",
                "max_size": effective_max,
                "max_size_formatted": format_bytes(effective_max),
                "configured_max_size": self._max_size,
                "configured_max_size_formatted": format_bytes(self._max_size),
                "total_size": self._tracked_ssd_size(),
                "total_size_formatted": format_bytes(self._tracked_ssd_size()),
                "utilization": (
                    self._tracked_ssd_size() / effective_max
                    if effective_max > 0
                    else 0.0
                ),
                "num_files": self._tracked_ssd_count(),
                "hot_cache_entries": hot_entries,
                "hot_cache_size_bytes": hot_size,
                "hot_cache_max_bytes": self._effective_hot_cache_max_bytes(),
                "hot_cache_size_formatted": format_bytes(hot_size),
                "hot_cache_max_formatted": format_bytes(
                    self._effective_hot_cache_max_bytes()
                ),
                **self._stats,
            }

    def close(self) -> None:
        """Close the SSD cache manager, flushing hot cache and pending writes."""
        logger.info("Shutting down PagedSSDCacheManager...")

        # Flush hot cache entries to SSD before shutdown.
        # Dirty blocks wait for queue space first; sustained saturation falls
        # back to an inline write on this thread.
        if self._hot_cache_enabled:
            with self._hot_cache_lock:
                entries_to_flush = list(self._hot_cache.items())
            flushed = 0
            failed = 0
            for block_hash, entry in entries_to_flush:
                if self._writer_thread and not self._writer_thread.is_alive():
                    logger.warning(
                        "Writer thread died during shutdown flush, "
                        f"aborting ({flushed} flushed, "
                        f"{len(entries_to_flush) - flushed - failed} remaining)"
                    )
                    break
                blk_meta = entry.get("block_metadata")
                if not entry.get("dirty", True):
                    continue
                if blk_meta and blk_meta.file_path.exists():
                    continue
                if self._enqueue_ssd_write(block_hash, entry, blocking=True):
                    flushed += 1
                else:
                    failed += 1
            if flushed:
                logger.info(f"Flushed {flushed} hot cache blocks to SSD")
            if failed:
                logger.warning(f"Failed to flush {failed} hot cache blocks")

        # Signal writer thread to stop (after processing remaining queue)
        if self._writer_thread:
            self._writer_shutdown.set()

            # Send sentinel to unblock the writer if it's waiting on the queue
            try:
                self._write_queue.put_nowait(None)
            except queue.Full:
                pass  # Writer will check shutdown flag on next iteration

            # Wait for writer to finish — longer timeout to allow flush
            timeout = 120 if self._hot_cache_enabled else 60
            self._writer_thread.join(timeout=timeout)
            if self._writer_thread.is_alive():
                logger.warning(
                    f"SSD cache writer thread did not stop within {timeout}s"
                )

        # Clear hot cache and pending write buffer
        with self._hot_cache_lock:
            self._hot_cache.clear()
            self._hot_cache_total_bytes = 0
        if self._hot_cache_budget is not None:
            self._hot_cache_budget.forget_owner(self)
        with self._pending_write_hashes_lock:
            self._pending_write_buffers.clear()
            self._pending_write_hashes.clear()

        logger.debug("PagedSSDCacheManager closed")

    def __repr__(self) -> str:
        return (
            f"PagedSSDCacheManager(dir={self._cache_dir}, "
            f"size={format_bytes(self._tracked_ssd_size())}/"
            f"{format_bytes(self._max_size)}, "
            f"files={self._tracked_ssd_count()})"
        )

    # =========================================================================
    # CacheManager ABC Interface Implementation
    # =========================================================================

    def fetch(self, key: Any) -> tuple[Any | None, bool]:
        """
        Fetch a cached block from SSD storage.

        Args:
            key: Block hash (bytes) to look up.

        Returns:
            Tuple of (cache_data, True) if found, (None, False) otherwise.
        """
        if not isinstance(key, bytes):
            return None, False

        cache_data = self.load_block(key)
        if cache_data is not None:
            return cache_data, True
        return None, False

    def store(self, key: Any, value: Any) -> bool:
        """
        Store a block in SSD cache.

        Args:
            key: Block hash (bytes).
            value: Tuple of (cache_data, token_count) or just cache_data.

        Returns:
            True if stored successfully.
        """
        if not isinstance(key, bytes):
            return False

        if isinstance(value, tuple) and len(value) >= 2:
            cache_data, token_count = value[0], value[1]
            model_name = value[2] if len(value) > 2 else ""
        else:
            cache_data = value
            token_count = 0
            model_name = ""

        return self.save_block(key, cache_data, token_count, model_name)

    def evict(self, key: Any) -> bool:
        """
        Evict a specific block from SSD cache.

        Args:
            key: Block hash (bytes) to evict.

        Returns:
            True if evicted, False if not found.
        """
        if not isinstance(key, bytes):
            return False

        return self.delete_block(key)

    @property
    def size(self) -> int:
        """
        Get the current number of cached blocks.

        Returns:
            Number of cached blocks.
        """
        return self._index.count

    @property
    def max_size(self) -> int:
        """
        Get the effective maximum cache size in bytes.

        This accounts for actual disk free space, returning the minimum of
        the configured max size and 99% of available disk space for cache.

        Returns:
            Effective maximum cache size in bytes.
        """
        return self._get_effective_max_size()

    @property
    def configured_max_size(self) -> int:
        """
        Get the originally configured maximum cache size in bytes.

        Returns:
            Configured maximum cache size in bytes.
        """
        return self._max_size
