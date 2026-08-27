# SPDX-License-Identifier: Apache-2.0
"""Fail-closed local safetensors ranges for shard-native TP loading.

This is the file-format primitive for a future structure-first tensor loader.
It deliberately does not accept URLs or alter the production loader yet: a
rank may read a strict subset of a *complete local* safetensors checkpoint,
while model sync/Hugging Face continue using their atomic full-file caches.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .staging import model_identity_digest

_MAX_HEADER_BYTES = 64 * 1024 * 1024
_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}
_MLX_DTYPE_NAMES = {
    "BOOL": "bool_",
    "U8": "uint8",
    "I8": "int8",
    "U16": "uint16",
    "I16": "int16",
    "F16": "float16",
    "BF16": "bfloat16",
    "U32": "uint32",
    "I32": "int32",
    "F32": "float32",
    "U64": "uint64",
    "I64": "int64",
    "F64": "float64",
}
_SAFE_NAME = re.compile(r"^[^/\\\x00]+\.safetensors$")


@dataclass(frozen=True)
class TensorDescriptor:
    name: str
    filename: str
    dtype: str
    shape: tuple[int, ...]
    data_start: int
    data_stop: int
    payload_start: int


@dataclass(frozen=True)
class TensorPartition:
    """One exact rank-local slice.

    ``axis=None`` means replicated. ``segments>1`` keeps the same weighted
    interval from every fused segment (DS4 wq_b/attention-sink use eight).
    ``boundary_multiple`` is expressed in elements of the sharded axis and
    lets quantized adapters prove that no rank boundary cuts a storage group.
    """

    axis: int | None = None
    segments: int = 1
    rank: int = 0
    weights: tuple[int, ...] = (1,)
    boundary_multiple: int = 1

    def normalized(self, shape: tuple[int, ...]) -> TensorPartition:
        if self.axis is None:
            if self.segments != 1 or self.rank != 0 or self.weights != (1,):
                raise ValueError("replicated partition cannot carry rank slicing")
            return self
        if not shape:
            raise ValueError("a scalar tensor cannot be tensor-parallel sharded")
        axis = self.axis % len(shape)
        if self.segments < 1 or shape[axis] % self.segments:
            raise ValueError("tensor dimension is not divisible by fused segments")
        if not self.weights or any(value <= 0 for value in self.weights):
            raise ValueError("partition weights must be positive")
        if not 0 <= self.rank < len(self.weights):
            raise ValueError("partition rank is outside its weight vector")
        if self.boundary_multiple < 1:
            raise ValueError("partition boundary multiple must be positive")
        segment = shape[axis] // self.segments
        total = sum(self.weights)
        before = sum(self.weights[: self.rank])
        after = before + self.weights[self.rank]
        if segment * before % total or segment * after % total:
            raise ValueError("weighted TP boundary is not an integer tensor offset")
        start = segment * before // total
        stop = segment * after // total
        if start % self.boundary_multiple or stop % self.boundary_multiple:
            raise ValueError("weighted TP boundary cuts a declared quantization group")
        return TensorPartition(
            axis=axis,
            segments=self.segments,
            rank=self.rank,
            weights=self.weights,
            boundary_multiple=self.boundary_multiple,
        )

    def local_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        normalized = self.normalized(shape)
        if normalized.axis is None:
            return shape
        result = list(shape)
        total = sum(normalized.weights)
        result[normalized.axis] = (
            shape[normalized.axis] * normalized.weights[normalized.rank] // total
        )
        return tuple(result)


@dataclass(frozen=True)
class PartitionManifestEntry:
    descriptor: TensorDescriptor
    partition: TensorPartition
    local_shape: tuple[int, ...]
    local_sha256: str


@dataclass(frozen=True)
class PartitionManifest:
    model_identity: str
    entries: tuple[PartitionManifestEntry, ...]


_REPLICATED = TensorPartition()


class LocalSafetensors:
    """Header-validated, local-file-only checkpoint view."""

    def __init__(self, model_path: str | Path) -> None:
        if isinstance(model_path, str) and "://" in model_path:
            raise ValueError("shard-native ranges require a complete local checkpoint")
        self.root = Path(model_path).expanduser().resolve()
        if not self.root.is_dir():
            raise ValueError(f"model path is not a directory: {self.root}")
        self.model_identity = model_identity_digest(self.root)
        self._headers: dict[str, dict[str, TensorDescriptor]] = {}
        # One read-only byte mapping per shard avoids tens of thousands of
        # open/mmap/close cycles on expert-heavy checkpoints (DS4 has 72,317
        # indexed tensors).  Pages are still faulted only for the exact local
        # ranges copied below; mapping a file does not make its full payload
        # resident.
        self._payloads: dict[str, np.memmap] = {}
        self._weight_map = self._read_weight_map()

    def _read_weight_map(self) -> dict[str, str]:
        index = self.root / "model.safetensors.index.json"
        if index.is_file():
            payload = json.loads(index.read_text())
            raw = payload.get("weight_map")
            if not isinstance(raw, dict) or not raw:
                raise ValueError("safetensors index has no weight_map")
            result: dict[str, str] = {}
            for name, filename in raw.items():
                if not isinstance(name, str) or not isinstance(filename, str):
                    raise ValueError("safetensors index entries must be strings")
                if not _SAFE_NAME.fullmatch(filename):
                    raise ValueError(f"unsafe safetensors filename: {filename!r}")
                result[name] = filename
            return result

        result = {}
        files = sorted(self.root.glob("model*.safetensors"))
        if not files:
            raise ValueError("no local model safetensors files were found")
        for path in files:
            for name in self._header(path.name):
                if name in result:
                    raise ValueError(f"duplicate safetensors tensor {name!r}")
                result[name] = path.name
        return result

    @property
    def tensor_names(self) -> tuple[str, ...]:
        """Names in the checkpoint's signed index order.

        The structure-first loader uses this without asking ``mx.load`` to
        construct a lazy array for every full tensor first.
        """

        return tuple(self._weight_map)

    def validate_complete(self) -> None:
        """Prove the local checkpoint contains exactly its indexed tensors.

        A Hugging Face identifier or a partially staged pipeline directory is
        intentionally not eligible for range loading.  Production may only
        select this path after every referenced shard and header has been
        checked; the ordinary MLX-LM downloader remains the atomic fallback.
        """

        names_by_file: dict[str, set[str]] = {}
        for name, filename in self._weight_map.items():
            names_by_file.setdefault(filename, set()).add(name)
        for filename, indexed in names_by_file.items():
            header = self._header(filename)
            actual = set(header)
            if actual != indexed:
                missing = sorted(indexed - actual)
                extra = sorted(actual - indexed)
                detail = []
                if missing:
                    detail.append(f"missing={missing[:3]!r}")
                if extra:
                    detail.append(f"extra={extra[:3]!r}")
                raise ValueError(
                    f"{filename}: safetensors index/header disagreement "
                    + " ".join(detail)
                )

        indexed_files = set(names_by_file)
        local_files = {path.name for path in self.root.glob("model*.safetensors")}
        if local_files != indexed_files:
            raise ValueError(
                "local safetensors shard set does not exactly match the index"
            )

    def _header(self, filename: str) -> dict[str, TensorDescriptor]:
        cached = self._headers.get(filename)
        if cached is not None:
            return cached
        if not _SAFE_NAME.fullmatch(filename):
            raise ValueError(f"unsafe safetensors filename: {filename!r}")
        path = self.root / filename
        if not path.is_file() or path.is_symlink() and not path.resolve().is_file():
            raise ValueError(f"missing local safetensors shard: {filename}")
        with path.open("rb") as stream:
            raw_length = stream.read(8)
            if len(raw_length) != 8:
                raise ValueError(f"{filename}: truncated header length")
            header_length = struct.unpack("<Q", raw_length)[0]
            if not 0 < header_length <= _MAX_HEADER_BYTES:
                raise ValueError(f"{filename}: invalid header length {header_length}")
            raw_header = stream.read(header_length)
            if len(raw_header) != header_length:
                raise ValueError(f"{filename}: truncated header")
        header = json.loads(raw_header)
        if not isinstance(header, dict):
            raise ValueError(f"{filename}: safetensors header is not an object")
        payload_start = 8 + header_length
        payload_bytes = path.stat().st_size - payload_start
        descriptors: dict[str, TensorDescriptor] = {}
        spans: list[tuple[int, int, str]] = []
        for name, raw in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(name, str) or not isinstance(raw, dict):
                raise ValueError(f"{filename}: invalid tensor metadata")
            dtype = raw.get("dtype")
            shape = raw.get("shape")
            offsets = raw.get("data_offsets")
            if dtype not in _DTYPE_BYTES:
                raise ValueError(f"{filename}:{name}: unsupported dtype {dtype!r}")
            if (
                not isinstance(shape, list)
                or any(not isinstance(value, int) or value < 0 for value in shape)
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or any(not isinstance(value, int) for value in offsets)
            ):
                raise ValueError(f"{filename}:{name}: invalid shape/offset metadata")
            start, stop = offsets
            elements = 1
            for dimension in shape:
                elements *= dimension
            expected = elements * _DTYPE_BYTES[dtype]
            if (
                start < 0
                or stop < start
                or stop > payload_bytes
                or stop - start != expected
            ):
                raise ValueError(f"{filename}:{name}: offsets do not match dtype/shape")
            # Safetensors does not require either the JSON payload start or a
            # tensor's relative start to be naturally aligned (real MLX files
            # pack U8 and U32 tensors without padding). ``stop - start`` must
            # still be an exact whole number of dtype elements; the
            # shape/length equality above is the format's alignment check.
            descriptor = TensorDescriptor(
                name=name,
                filename=filename,
                dtype=dtype,
                shape=tuple(shape),
                data_start=start,
                data_stop=stop,
                payload_start=payload_start,
            )
            descriptors[name] = descriptor
            spans.append((start, stop, name))
        spans.sort()
        cursor = 0
        for start, stop, name in spans:
            if start != cursor:
                raise ValueError(f"{filename}:{name}: overlapping or gapped payload")
            cursor = stop
        if cursor != payload_bytes:
            raise ValueError(f"{filename}: trailing or missing safetensors payload")
        self._headers[filename] = descriptors
        return descriptors

    def descriptor(self, name: str) -> TensorDescriptor:
        filename = self._weight_map.get(name)
        if filename is None:
            raise KeyError(name)
        descriptor = self._header(filename).get(name)
        if descriptor is None:
            raise ValueError(f"index/header disagreement for tensor {name!r}")
        return descriptor

    def _numpy_partition(
        self,
        path: Path,
        descriptor: TensorDescriptor,
        partition: TensorPartition,
        *,
        use_mapping_cache: bool = False,
    ) -> np.ndarray:
        itemsize = _DTYPE_BYTES[descriptor.dtype]
        storage_dtype = np.dtype(f"<u{itemsize}")
        mapped = self._payloads.get(descriptor.filename) if use_mapping_cache else None
        if mapped is None:
            mapped = np.memmap(path, dtype=np.uint8, mode="r")
            if use_mapping_cache:
                self._payloads[descriptor.filename] = mapped
        start = descriptor.payload_start + descriptor.data_start
        stop = descriptor.payload_start + descriptor.data_stop
        source = mapped[start:stop].view(storage_dtype).reshape(descriptor.shape)
        normalized = partition.normalized(descriptor.shape)
        if normalized.axis is None:
            return np.array(source, copy=True, order="C")
        axis = normalized.axis
        segment_size = descriptor.shape[axis] // normalized.segments
        total = sum(normalized.weights)
        before = sum(normalized.weights[: normalized.rank])
        after = before + normalized.weights[normalized.rank]
        start = segment_size * before // total
        stop = segment_size * after // total
        pieces = []
        for segment in range(normalized.segments):
            index = [slice(None)] * len(descriptor.shape)
            base = segment * segment_size
            index[axis] = slice(base + start, base + stop)
            pieces.append(source[tuple(index)])
        if len(pieces) == 1:
            return np.array(pieces[0], copy=True, order="C")
        return np.concatenate(pieces, axis=axis)

    def partition_bytes(
        self,
        name: str,
        partition: TensorPartition = _REPLICATED,
        *,
        use_mapping_cache: bool = False,
    ) -> tuple[TensorDescriptor, np.ndarray]:
        descriptor = self.descriptor(name)
        values = self._numpy_partition(
            self.root / descriptor.filename,
            descriptor,
            partition,
            use_mapping_cache=use_mapping_cache,
        )
        if tuple(values.shape) != partition.local_shape(descriptor.shape):
            raise RuntimeError("rank-local safetensors slice has the wrong shape")
        return descriptor, values

    def manifest_entry(
        self,
        name: str,
        partition: TensorPartition = _REPLICATED,
    ) -> PartitionManifestEntry:
        descriptor, values = self.partition_bytes(name, partition)
        digest = hashlib.sha256(values.view(np.uint8)).hexdigest()
        return PartitionManifestEntry(
            descriptor=descriptor,
            partition=partition.normalized(descriptor.shape),
            local_shape=tuple(values.shape),
            local_sha256=digest,
        )

    def manifest(self, entries: dict[str, TensorPartition]) -> PartitionManifest:
        return PartitionManifest(
            model_identity=self.model_identity,
            entries=tuple(
                self.manifest_entry(name, partition)
                for name, partition in entries.items()
            ),
        )

    def verify_manifest(self, manifest: PartitionManifest) -> None:
        """Pin a rank plan to the same model revision and tensor metadata."""

        if manifest.model_identity != self.model_identity:
            raise ValueError(
                "rank-local manifest belongs to a different model identity"
            )
        if not manifest.entries:
            raise ValueError("rank-local manifest is empty")
        seen: set[str] = set()
        for entry in manifest.entries:
            name = entry.descriptor.name
            if name in seen:
                raise ValueError(f"duplicate rank-local manifest tensor {name!r}")
            seen.add(name)
            if self.descriptor(name) != entry.descriptor:
                raise ValueError(f"safetensors metadata changed for {name!r}")

    def load_entry(self, entry: PartitionManifestEntry, *, mx_module: Any) -> Any:
        current = self.descriptor(entry.descriptor.name)
        if current != entry.descriptor:
            raise ValueError("safetensors metadata changed after manifest creation")
        descriptor, values = self.partition_bytes(current.name, entry.partition)
        if tuple(values.shape) != entry.local_shape:
            raise ValueError("rank-local tensor shape changed after qualification")
        if hashlib.sha256(values.view(np.uint8)).hexdigest() != entry.local_sha256:
            raise ValueError("rank-local tensor checksum mismatch")
        storage = mx_module.array(values)
        logical_dtype = getattr(mx_module, _MLX_DTYPE_NAMES[descriptor.dtype])
        if descriptor.dtype == "BOOL":
            result = storage.astype(logical_dtype)
        elif storage.dtype == logical_dtype:
            result = storage
        else:
            result = storage.view(logical_dtype)
        mx_module.eval(result)
        return result

    def load_partition(
        self,
        name: str,
        partition: TensorPartition = _REPLICATED,
        *,
        mx_module: Any,
        expected_descriptor: TensorDescriptor | None = None,
        evaluate: bool = False,
    ) -> Any:
        """Read one already-qualified local slice without a full-tensor graph.

        ``expected_descriptor`` pins the read to the metadata inspected by the
        adapter before it mutates a model structure.  Byte checksummed
        :meth:`load_entry` remains available for persisted/cross-process
        manifests; this one-shot loader instead validates the checkpoint file
        snapshot before and after the complete load so it does not read every
        60--100 GiB rank shard twice.
        """

        current = self.descriptor(name)
        if expected_descriptor is not None and current != expected_descriptor:
            raise ValueError("safetensors metadata changed after qualification")
        descriptor, values = self.partition_bytes(
            name,
            partition,
            use_mapping_cache=True,
        )
        storage = mx_module.array(values)
        logical_dtype = getattr(mx_module, _MLX_DTYPE_NAMES[descriptor.dtype])
        if descriptor.dtype == "BOOL":
            result = storage.astype(logical_dtype)
        elif storage.dtype == logical_dtype:
            result = storage
        else:
            result = storage.view(logical_dtype)
        if evaluate:
            mx_module.eval(result)
        return result


def validate_quantized_partition(
    weight: TensorDescriptor,
    scales: TensorDescriptor,
    partition: TensorPartition,
    *,
    bits: int,
    group_size: int,
    biases: TensorDescriptor | None = None,
) -> None:
    """Prove packed weights and quantization metadata select identical values."""

    if bits not in (2, 3, 4, 6, 8) or 32 % bits:
        raise ValueError("unsupported packed quantization bit width")
    if group_size <= 0 or not weight.shape or len(scales.shape) != len(weight.shape):
        raise ValueError("invalid quantized tensor metadata")
    if weight.shape[:-1] != scales.shape[:-1]:
        raise ValueError("quantized weight/scale output dimensions differ")
    logical_weight_inputs = weight.shape[-1] * 32 // bits
    logical_scale_inputs = scales.shape[-1] * group_size
    if logical_weight_inputs != logical_scale_inputs:
        raise ValueError("packed weight and scale groups cover different inputs")
    if biases is not None and (
        biases.shape != scales.shape or biases.dtype != scales.dtype
    ):
        raise ValueError("quantization biases do not match scales")
    local_weight = partition.local_shape(weight.shape)
    local_scales = partition.local_shape(scales.shape)
    if (
        partition.axis is not None
        and partition.axis % len(weight.shape) == len(weight.shape) - 1
    ):
        if local_weight[-1] * 32 // bits != local_scales[-1] * group_size:
            raise ValueError("rank-local packed weights cut a quantization group")
    elif local_weight[:-1] != local_scales[:-1]:
        raise ValueError("rank-local quantization rows do not match")


_DS4_LAYER_OR_MTP = r"(?:model\.)?(?:layers\.\d+|mtp\.\d+(?:\.block)?)"


def deepseek_v4_partition(
    name: str,
    *,
    rank: int,
    shard_weights: tuple[int, ...],
    world_size: int,
    quant_boundary: int = 1,
    o_groups: int = 8,
) -> TensorPartition:
    """Declare DS4's raw-checkpoint ownership without guessing other models.

    Unknown/fixed tensors stay replicated. The caller supplies the already
    signed TP weights; vocabulary projections deliberately remain equal, as
    required by the all-gather protocol.
    """

    if world_size != len(shard_weights) or not 0 <= rank < world_size:
        raise ValueError("DS4 shard vector does not match the TP group")
    if o_groups < 1:
        raise ValueError("DS4 output-group count must be positive")
    raw = name.removeprefix("model.")
    weighted = dict(rank=rank, weights=shard_weights)
    suffix = r"(?:weight|scales|biases|bias)"
    if re.fullmatch(
        rf"{_DS4_LAYER_OR_MTP}\.attn\.wq_b\.{suffix}", name
    ) or re.fullmatch(rf"{_DS4_LAYER_OR_MTP}\.attn\.attn_sink", name):
        return TensorPartition(axis=0, segments=o_groups, **weighted)
    if re.fullmatch(
        rf"{_DS4_LAYER_OR_MTP}\.attn\.wo_a\.(?:weight|scales|biases)", name
    ):
        return TensorPartition(axis=-1, **weighted)
    if re.fullmatch(
        rf"{_DS4_LAYER_OR_MTP}\.ffn\.(?:shared_experts\.(?:w1|w3|gate_proj|up_proj)|experts\.\d+\.(?:w1|w3)|switch_mlp\.(?:gate_proj|up_proj))\.{suffix}",
        name,
    ):
        return TensorPartition(
            axis=0,
            boundary_multiple=quant_boundary,
            **weighted,
        )
    if re.fullmatch(
        rf"{_DS4_LAYER_OR_MTP}\.ffn\.(?:shared_experts\.(?:w2|down_proj)|experts\.\d+\.w2|switch_mlp\.down_proj)\.(?:weight|scales|biases)",
        name,
    ):
        return TensorPartition(axis=-1, **weighted)
    if raw in {
        "head.weight",
        "head.scales",
        "head.biases",
        "head.bias",
        "lm_head.weight",
        "lm_head.scales",
        "lm_head.biases",
        "lm_head.bias",
    } or re.fullmatch(
        r"mtp\.\d+\.markov_head\.markov_w2\.(?:weight|scales|biases|bias)",
        raw,
    ):
        return TensorPartition(axis=0, rank=rank, weights=(1,) * world_size)
    return TensorPartition()


__all__ = [
    "LocalSafetensors",
    "PartitionManifest",
    "PartitionManifestEntry",
    "TensorDescriptor",
    "TensorPartition",
    "deepseek_v4_partition",
    "validate_quantized_partition",
]
