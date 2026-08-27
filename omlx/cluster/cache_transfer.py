# SPDX-License-Identifier: Apache-2.0
"""Lossless prompt-cache handoff between disaggregated MLX workers.

The wire schema is the in-memory equivalent of MLX-LM's prompt-cache
safetensors contract: cache class names, ``meta_state`` strings, and a flat
ordered list of tensor leaves.  Only the transport differs.  Tensor leaves go
directly through ``mx.distributed.send``/``recv`` so a JACCL group keeps the KV
handoff on RDMA and never stages it through Python bytes, HTTP, or SSD.

This module deliberately knows nothing about a particular model.  A cache
type is admitted only when MLX-LM exposes it as a cache class with
``from_state`` and every state leaf is an MLX array.  Unsupported future cache
types fail closed before the first tensor is sent.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

CACHE_TRANSFER_SCHEMA_VERSION = 1
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_CACHE_ARRAYS = 4096
MAX_ARRAY_DIMS = 16
_DTYPE_NAMES = (
    "bool_",
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "float16",
    "bfloat16",
    "float32",
    "float64",
    "complex64",
)


def _transfer_window() -> int:
    """Number of ordered point-to-point ops submitted per Metal eval."""

    try:
        value = int(os.environ.get("OMLX_CLUSTER_CACHE_TRANSFER_WINDOW", "8"))
    except ValueError:
        value = 8
    return max(1, min(16, value))


@dataclass(frozen=True)
class PreparedCacheTransfer:
    manifest: dict[str, Any]
    arrays: tuple[Any, ...]
    nbytes: int


@dataclass(frozen=True)
class CacheTransferStats:
    array_count: int
    tensor_bytes: int
    manifest_bytes: int
    elapsed_seconds: float

    @property
    def bytes_per_second(self) -> float:
        return (
            self.tensor_bytes / self.elapsed_seconds
            if self.elapsed_seconds > 0
            else 0.0
        )


def _dtype_name(dtype: Any) -> str:
    name = str(dtype).rsplit(".", 1)[-1]
    if name not in _DTYPE_NAMES:
        raise TypeError(f"cache transfer does not support dtype {dtype}")
    return name


def _dtype_from_name(mx: Any, name: str) -> Any:
    if name not in _DTYPE_NAMES:
        raise ValueError(f"cache transfer manifest has unsupported dtype {name!r}")
    return getattr(mx, name)


def _canonical_manifest_bytes(manifest: dict[str, Any]) -> bytes:
    payload = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if not payload or len(payload) > MAX_MANIFEST_BYTES:
        raise ValueError("cache transfer manifest is outside the bounded size")
    return payload


def prepare_cache_transfer(
    cache: Sequence[Any],
    *,
    model_identity: str,
    prompt_tokens: int,
) -> PreparedCacheTransfer:
    """Freeze the schema of one completed-prefill cache without copying it."""

    import mlx.core as mx
    from mlx.utils import tree_flatten

    if not cache:
        raise ValueError("cache transfer requires at least one cache member")
    if not isinstance(model_identity, str) or not model_identity:
        raise ValueError("cache transfer requires a model identity")
    if prompt_tokens < 1:
        raise ValueError("cache transfer requires a positive prompt length")

    state_items = list(tree_flatten([entry.state for entry in cache]))
    if not state_items or len(state_items) > MAX_CACHE_ARRAYS:
        raise ValueError("cache transfer has an invalid tensor-leaf count")

    arrays: list[Any] = []
    tensor_entries: list[dict[str, Any]] = []
    total_bytes = 0
    for key, value in state_items:
        if not isinstance(key, str) or not isinstance(value, mx.array):
            raise TypeError("cache transfer state must contain only MLX arrays")
        shape = tuple(int(dimension) for dimension in value.shape)
        if len(shape) > MAX_ARRAY_DIMS or any(dimension < 0 for dimension in shape):
            raise ValueError("cache transfer encountered an invalid tensor shape")
        nbytes = int(value.nbytes)
        total_bytes += nbytes
        tensor_entries.append(
            {
                "key": key,
                "shape": list(shape),
                "dtype": _dtype_name(value.dtype),
                "nbytes": nbytes,
            }
        )
        arrays.append(value)

    metadata_items = list(
        tree_flatten(
            [
                [entry.meta_state for entry in cache],
                {},
                [type(entry).__name__ for entry in cache],
            ]
        )
    )
    if any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in metadata_items
    ):
        raise TypeError("cache transfer metadata must contain only strings")

    manifest = {
        "schema_version": CACHE_TRANSFER_SCHEMA_VERSION,
        "model_identity": model_identity,
        "prompt_tokens": int(prompt_tokens),
        "tensor_bytes": total_bytes,
        "tensors": tensor_entries,
        "metadata": [[key, value] for key, value in metadata_items],
    }
    _canonical_manifest_bytes(manifest)
    return PreparedCacheTransfer(manifest, tuple(arrays), total_bytes)


def _validate_manifest(
    manifest: dict[str, Any],
    *,
    expected_model_identity: str | None,
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("unsupported cache transfer schema")
    identity = manifest.get("model_identity")
    if not isinstance(identity, str) or not identity:
        raise ValueError("cache transfer manifest has no model identity")
    if expected_model_identity is not None and identity != expected_model_identity:
        raise ValueError("cache transfer model identity does not match decoder")
    if (
        not isinstance(manifest.get("prompt_tokens"), int)
        or manifest["prompt_tokens"] < 1
    ):
        raise ValueError("cache transfer manifest has an invalid prompt length")
    tensors = manifest.get("tensors")
    metadata = manifest.get("metadata")
    if (
        not isinstance(tensors, list)
        or not tensors
        or len(tensors) > MAX_CACHE_ARRAYS
        or not isinstance(metadata, list)
    ):
        raise ValueError("cache transfer manifest has invalid tensor metadata")
    normalized_metadata: list[tuple[str, str]] = []
    for item in metadata:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not all(isinstance(value, str) for value in item)
        ):
            raise ValueError("cache transfer manifest has invalid cache metadata")
        normalized_metadata.append((item[0], item[1]))
    total_bytes = 0
    for entry in tensors:
        if not isinstance(entry, dict):
            raise ValueError("cache transfer manifest has an invalid tensor entry")
        if not isinstance(entry.get("key"), str):
            raise ValueError("cache transfer tensor has no tree key")
        shape = entry.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) > MAX_ARRAY_DIMS
            or any(not isinstance(value, int) or value < 0 for value in shape)
        ):
            raise ValueError("cache transfer tensor has an invalid shape")
        _ = entry.get("dtype")
        if not isinstance(_, str) or _ not in _DTYPE_NAMES:
            raise ValueError("cache transfer tensor has an invalid dtype")
        nbytes = entry.get("nbytes")
        if not isinstance(nbytes, int) or nbytes < 0:
            raise ValueError("cache transfer tensor has an invalid byte count")
        total_bytes += nbytes
    if manifest.get("tensor_bytes") != total_bytes:
        raise ValueError("cache transfer tensor byte ledger does not balance")
    return tensors, normalized_metadata


def restore_cache_transfer(
    manifest: dict[str, Any],
    arrays: Sequence[Any],
    *,
    expected_model_identity: str | None = None,
) -> list[Any]:
    """Reconstruct MLX-LM cache objects from an already-received manifest."""

    import mlx.core as mx
    import mlx_lm.models.cache as cache_module
    from mlx.utils import tree_unflatten

    tensors, metadata_items = _validate_manifest(
        manifest,
        expected_model_identity=expected_model_identity,
    )
    if len(arrays) != len(tensors):
        raise ValueError("cache transfer received the wrong tensor count")
    state_items: list[tuple[str, Any]] = []
    for entry, value in zip(tensors, arrays):
        if not isinstance(value, mx.array):
            raise TypeError("cache transfer received a non-MLX tensor")
        if tuple(value.shape) != tuple(entry["shape"]):
            raise ValueError("cache transfer received a tensor with the wrong shape")
        if _dtype_name(value.dtype) != entry["dtype"]:
            raise ValueError("cache transfer received a tensor with the wrong dtype")
        if int(value.nbytes) != entry["nbytes"]:
            raise ValueError("cache transfer received a tensor with the wrong size")
        state_items.append((entry["key"], value))

    states = tree_unflatten(state_items)
    cache_info, _user_metadata, classes = tree_unflatten(metadata_items)
    if not (
        isinstance(states, list)
        and isinstance(cache_info, list)
        and isinstance(classes, list)
        and len(states) == len(cache_info) == len(classes)
    ):
        raise ValueError("cache transfer metadata does not describe one cache list")

    restored = []
    for class_name, state, meta_state in zip(classes, states, cache_info):
        cache_class = getattr(cache_module, class_name, None)
        if (
            cache_class is None
            or class_name.startswith("_")
            or not callable(getattr(cache_class, "from_state", None))
        ):
            raise ValueError(f"cache transfer does not admit class {class_name!r}")
        restored.append(cache_class.from_state(state, meta_state))
    return restored


def send_cache_transfer(
    mx: Any,
    prepared: PreparedCacheTransfer,
    *,
    dst: int,
    group: Any = None,
) -> CacheTransferStats:
    """Send one prepared cache over the group's point-to-point data plane."""

    payload = _canonical_manifest_bytes(prepared.manifest)
    started = time.perf_counter()
    length = mx.array([len(payload)], dtype=mx.uint32)
    manifest_array = mx.array(list(payload), dtype=mx.uint8)
    mx.eval(mx.distributed.send(length, dst, group=group))
    mx.eval(mx.distributed.send(manifest_array, dst, group=group))
    window = _transfer_window()
    for start in range(0, len(prepared.arrays), window):
        operations = []
        for value in prepared.arrays[start : start + window]:
            # Python ``mlx.core.array`` intentionally does not expose the C++
            # ``flags()`` API. Always materialize a dense wire view; this is a
            # no-op for already-contiguous cache tensors and preserves the same
            # storage contract safetensors would reconstruct on the decoder.
            wire_value = mx.contiguous(value)
            operations.append(mx.distributed.send(wire_value, dst, group=group))
        mx.eval(*operations)
    mx.synchronize()
    return CacheTransferStats(
        array_count=len(prepared.arrays),
        tensor_bytes=prepared.nbytes,
        manifest_bytes=len(payload),
        elapsed_seconds=time.perf_counter() - started,
    )


def recv_cache_transfer(
    mx: Any,
    *,
    src: int,
    group: Any = None,
    expected_model_identity: str | None = None,
) -> tuple[list[Any], dict[str, Any], CacheTransferStats]:
    """Receive and reconstruct one cache from a point-to-point data plane."""

    started = time.perf_counter()
    length_array = mx.distributed.recv((1,), mx.uint32, src, group=group)
    mx.eval(length_array)
    manifest_length = int(length_array.item())
    if not 1 <= manifest_length <= MAX_MANIFEST_BYTES:
        raise ValueError("cache transfer received an invalid manifest length")
    payload_array = mx.distributed.recv((manifest_length,), mx.uint8, src, group=group)
    mx.eval(payload_array)
    try:
        manifest = json.loads(bytes(payload_array.tolist()))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("cache transfer manifest is not valid JSON") from exc
    tensors, _ = _validate_manifest(
        manifest,
        expected_model_identity=expected_model_identity,
    )
    arrays = []
    window = _transfer_window()
    for start in range(0, len(tensors), window):
        received = [
            mx.distributed.recv(
                tuple(entry["shape"]),
                _dtype_from_name(mx, entry["dtype"]),
                src,
                group=group,
            )
            for entry in tensors[start : start + window]
        ]
        mx.eval(*received)
        arrays.extend(received)
    mx.synchronize()
    cache = restore_cache_transfer(
        manifest,
        arrays,
        expected_model_identity=expected_model_identity,
    )
    return (
        cache,
        manifest,
        CacheTransferStats(
            array_count=len(arrays),
            tensor_bytes=int(manifest["tensor_bytes"]),
            manifest_bytes=manifest_length,
            elapsed_seconds=time.perf_counter() - started,
        ),
    )


__all__ = [
    "CACHE_TRANSFER_SCHEMA_VERSION",
    "CacheTransferStats",
    "PreparedCacheTransfer",
    "prepare_cache_transfer",
    "recv_cache_transfer",
    "restore_cache_transfer",
    "send_cache_transfer",
]
