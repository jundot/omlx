# SPDX-License-Identifier: Apache-2.0
"""Qwen4-Exp language blocks implemented with MLX."""

from __future__ import annotations

import json
import math
import mmap
import os
import struct
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.cache import ArraysCache, KVCache

from ..qwen3_5.language import LanguageModel as Qwen3_5LanguageModel
from ..qwen3_5.language import Qwen3_5Attention, Qwen3_5GatedDeltaNet
from ..qwen3_5.language import (
    _create_qwen3_5_attention_mask,
    _create_qwen3_5_ssm_mask,
)
from ..qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock

from .config import TextConfig

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007
_PLE_RUNTIME_MODEL_PATH: Path | None = None
_PLE_RUNTIME_MODE = "resident"
_HYPER_SPLIT_INDICES: dict[tuple[int, int], tuple[mx.array, mx.array]] = {}


def resolve_ple_runtime_mode(
    requested: str,
    *,
    checkpoint_bytes: int,
    physical_memory: int,
) -> str:
    """Resolve ``auto`` while retaining enough RAM for KV cache and Metal."""
    requested = requested.strip().lower()
    if requested not in {"auto", "resident", "mmap"}:
        raise ValueError("OMLX_QWEN4_PLE_MODE must be one of: auto, resident, mmap")
    if requested != "auto":
        return requested
    return "mmap" if checkpoint_bytes > physical_memory * 0.70 else "resident"


def configure_ple_runtime(
    model_path: str | Path,
    mode: str | None = None,
) -> str:
    """Configure PLE construction for the next Qwen4-Exp model instance."""
    global _PLE_RUNTIME_MODEL_PATH, _PLE_RUNTIME_MODE

    model_path = Path(model_path)
    requested = mode or os.environ.get("OMLX_QWEN4_PLE_MODE", "auto")
    checkpoint_bytes = sum(
        path.stat().st_size for path in model_path.glob("*.safetensors")
    )
    physical_memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    resolved = resolve_ple_runtime_mode(
        requested,
        checkpoint_bytes=checkpoint_bytes,
        physical_memory=physical_memory,
    )
    _PLE_RUNTIME_MODEL_PATH = model_path
    _PLE_RUNTIME_MODE = resolved
    return resolved


def _hyper_split_indices(lowrank: int, hc_count: int) -> tuple[mx.array, mx.array]:
    key = (lowrank, hc_count)
    indices = _HYPER_SPLIT_INDICES.get(key)
    if indices is None:
        indices = (
            mx.arange(lowrank, dtype=mx.int32),
            mx.arange(lowrank, lowrank + hc_count, dtype=mx.int32),
        )
        _HYPER_SPLIT_INDICES[key] = indices
    return indices


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    return all(value % divisor for divisor in range(3, math.isqrt(value) + 1, 2))


def _find_nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


def _build_layer_multipliers(
    unigram_vocab_size: int,
    ngram_size: int,
    ple_layer_index: int,
    seed: int,
) -> list[int]:
    multiplier_max = ((1 << 63) - 1) // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    return [
        2
        * (
            _splitmix64((base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64)
            % half_bound
        )
        + 1
        for index in range(ngram_size)
    ]


class NGramHasher:
    """Exact SplitMix64 n-gram ids used by the Transformers reference."""

    def __init__(self, args: TextConfig, ple_layer_index: int = 0):
        self.ngram_size = args.ngram_size
        self.context_len = self.ngram_size - 1
        self.heads_per_ngram = args.heads_per_ngram
        self.ngram_heads = self.context_len * self.heads_per_ngram
        self.eos_token_id = (
            args.eos_token_id[0]
            if isinstance(args.eos_token_id, list)
            else args.eos_token_id
        )

        self.head_vocab_sizes = [
            _find_nth_prime_after(
                args.ngram_vocab_size_base - 1,
                ple_layer_index * self.ngram_heads + head_index + 1,
            )
            for head_index in range(self.ngram_heads)
        ]
        self.head_offsets = []
        self.total_vocab_size = 0
        for size in self.head_vocab_sizes:
            self.head_offsets.append(self.total_vocab_size)
            self.total_vocab_size += size

        divisor = args.make_ngram_vocab_size_divisible_by
        self.padded_vocab_size = math.ceil(self.total_vocab_size / divisor) * divisor
        if self.padded_vocab_size % args.split_ngram_parts:
            raise ValueError("padded n-gram vocabulary must divide evenly into shards")
        self.shard_rows = self.padded_vocab_size // args.split_ngram_parts
        self.layer_multipliers = _build_layer_multipliers(
            args.vocab_size,
            self.ngram_size,
            ple_layer_index,
            args.seed,
        )

    def _shift_right_ignore_eos(self, token_ids: mx.array, shift: int) -> mx.array:
        if shift == 0:
            return token_ids
        batch_size, seq_len = token_ids.shape
        positions = mx.arange(seq_len, dtype=mx.int64)
        eos_positions = mx.where(token_ids == self.eos_token_id, positions, -1)
        previous_eos_inclusive = mx.cummax(eos_positions, axis=1)
        previous_eos = mx.concatenate(
            [
                mx.full((batch_size, 1), -1, dtype=mx.int64),
                previous_eos_inclusive[:, :-1],
            ],
            axis=1,
        )
        segment_start = previous_eos + 1
        source_positions = positions - shift
        gather_positions = mx.broadcast_to(
            mx.maximum(source_positions, 0)[None, :], (batch_size, seq_len)
        )
        shifted = mx.take_along_axis(token_ids, gather_positions, axis=1)
        valid = (positions[None, :] - segment_start >= shift) & (
            source_positions[None, :] >= 0
        )
        return mx.where(valid, shifted, self.eos_token_id)

    def compute_ids(
        self,
        token_history: mx.array,
        *,
        current_length: int | None = None,
        layer_multipliers: mx.array | None = None,
        head_vocab_sizes: mx.array | None = None,
        head_offsets: mx.array | None = None,
    ) -> mx.array:
        token_history = token_history.astype(mx.int64)
        multipliers = (
            self.layer_multipliers if layer_multipliers is None else layer_multipliers
        )
        all_vocab_sizes = (
            self.head_vocab_sizes if head_vocab_sizes is None else head_vocab_sizes
        )
        all_offsets = self.head_offsets if head_offsets is None else head_offsets
        shifted_tokens = [
            self._shift_right_ignore_eos(token_history, shift)
            for shift in range(self.ngram_size)
        ]
        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed_ids = shifted_tokens[0] * multipliers[0]
            for position in range(1, ngram):
                mixed_ids = mx.bitwise_xor(
                    mixed_ids,
                    shifted_tokens[position] * multipliers[position],
                )
            vocab_sizes = mx.array(all_vocab_sizes[start:end], dtype=mx.int64)
            offsets = mx.array(all_offsets[start:end], dtype=mx.int64)
            blocks.append(mixed_ids[..., None] % vocab_sizes + offsets)

        ids = mx.concatenate(blocks, axis=-1)
        if current_length is not None:
            ids = ids[:, -current_length:, :]
        return ids


class QuantizedEmbeddingShard(nn.Module):
    """Shape-only quantized embedding whose tensors are replaced at load time."""

    def __init__(
        self,
        num_embeddings: int,
        dims: int,
        group_size: int,
        bits: int,
        mode: str = "affine",
    ):
        super().__init__()
        if mode != "affine":
            raise ValueError("Qwen4-Exp PLE currently supports affine shards")
        if dims % group_size:
            raise ValueError(
                "embedding dimensions must divide into quantization groups"
            )
        self.num_embeddings = num_embeddings
        self.dims = dims
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        self.weight = mx.zeros((num_embeddings, dims * bits // 32), dtype=mx.uint32)
        self.scales = mx.zeros((num_embeddings, dims // group_size), dtype=mx.bfloat16)
        self.biases = mx.zeros((num_embeddings, dims // group_size), dtype=mx.bfloat16)
        self.freeze()

    def __call__(self, ids: mx.array) -> mx.array:
        return mx.dequantize(
            self.weight[ids],
            scales=self.scales[ids],
            biases=self.biases[ids],
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )

    def to_quantized(
        self,
        group_size: int | None = None,
        bits: int | None = None,
        mode: str = "affine",
        **_kwargs,
    ):
        """Tell ``nn.quantize`` this checkpoint-native module is already done."""
        group_size = self.group_size if group_size is None else group_size
        bits = self.bits if bits is None else bits
        if (group_size, bits, mode) != (self.group_size, self.bits, self.mode):
            raise ValueError("PLE shard quantization does not match the checkpoint")
        return self


class ShardedQuantizedEmbedding(nn.Module):
    """Contiguous row-sharded embedding with sparse, mmap-friendly gathers."""

    def __init__(
        self,
        num_embeddings: int,
        dims: int,
        num_shards: int,
        group_size: int = 32,
        bits: int = 4,
        mode: str = "affine",
    ):
        super().__init__()
        if num_embeddings % num_shards:
            raise ValueError("embedding rows must divide evenly into shards")
        self.num_embeddings = num_embeddings
        self.dims = dims
        self.num_shards = num_shards
        self.shard_rows = num_embeddings // num_shards
        self.last_touched_shards: tuple[int, ...] = ()
        for shard_index in range(num_shards):
            setattr(
                self,
                f"shard_{shard_index}",
                QuantizedEmbeddingShard(
                    self.shard_rows,
                    dims,
                    group_size=group_size,
                    bits=bits,
                    mode=mode,
                ),
            )

    def __call__(self, ids: mx.array) -> mx.array:
        original_shape = ids.shape
        flat_ids = ids.reshape(-1).astype(mx.int64)
        mx.eval(flat_ids)
        host_ids = [int(value) for value in flat_ids.tolist()]
        if any(value < 0 or value >= self.num_embeddings for value in host_ids):
            raise IndexError("n-gram embedding id is outside the padded vocabulary")

        touched = tuple(sorted({value // self.shard_rows for value in host_ids}))
        self.last_touched_shards = touched
        first_shard = self.shard_0
        output = mx.zeros((len(host_ids), self.dims), dtype=first_shard.scales.dtype)
        for shard_index in touched:
            positions_list = [
                index
                for index, value in enumerate(host_ids)
                if value // self.shard_rows == shard_index
            ]
            local_ids_list = [
                host_ids[index] % self.shard_rows for index in positions_list
            ]
            positions = mx.array(positions_list, dtype=mx.int32)
            local_ids = mx.array(local_ids_list, dtype=mx.int64)
            values = getattr(self, f"shard_{shard_index}")(local_ids)
            output = output.at[positions].add(values)
        return output.reshape(*original_shape, self.dims)


_SAFETENSORS_NUMPY_DTYPES = {
    "U32": np.dtype("<u4"),
    "I32": np.dtype("<i4"),
    "I64": np.dtype("<i8"),
    "F16": np.dtype("<f2"),
    "F32": np.dtype("<f4"),
    # NumPy has no portable bfloat16 dtype. Keep the raw 16-bit payload and
    # expand it exactly when copying the requested rows out of the mmap.
    "BF16": np.dtype("<u2"),
}


class _SafeTensorMMap:
    """Minimal read-only safetensors reader optimized for sparse row access."""

    def __init__(self, path: Path):
        self.path = path
        self._file = path.open("rb")
        raw_header_length = self._file.read(8)
        if len(raw_header_length) != 8:
            self.close()
            raise ValueError(f"Invalid safetensors header in {path}")
        header_length = struct.unpack("<Q", raw_header_length)[0]
        raw_header = self._file.read(header_length)
        if len(raw_header) != header_length:
            self.close()
            raise ValueError(f"Truncated safetensors header in {path}")
        self._header = json.loads(raw_header)
        self._data_start = 8 + header_length
        self._mapping = mmap.mmap(
            self._file.fileno(), length=0, access=mmap.ACCESS_READ
        )
        try:
            self._mapping.madvise(mmap.MADV_RANDOM)
        except (AttributeError, OSError):
            pass

    def tensor_shape(self, key: str) -> tuple[int, ...]:
        try:
            return tuple(self._header[key]["shape"])
        except KeyError as exc:
            raise KeyError(f"Tensor {key!r} is missing from {self.path}") from exc

    def rows(self, key: str, row_indices: list[int]) -> tuple[np.ndarray, str]:
        try:
            entry = self._header[key]
        except KeyError as exc:
            raise KeyError(f"Tensor {key!r} is missing from {self.path}") from exc
        dtype_name = entry["dtype"]
        try:
            dtype = _SAFETENSORS_NUMPY_DTYPES[dtype_name]
        except KeyError as exc:
            raise TypeError(
                f"Unsupported safetensors dtype {dtype_name!r} for {key}"
            ) from exc
        shape = tuple(entry["shape"])
        if len(shape) != 2:
            raise ValueError(f"Sparse PLE tensor {key!r} must be two-dimensional")
        start, end = entry["data_offsets"]
        expected_bytes = math.prod(shape) * dtype.itemsize
        if end - start != expected_bytes:
            raise ValueError(f"Invalid byte range for safetensors tensor {key!r}")

        view = np.ndarray(
            shape,
            dtype=dtype,
            buffer=self._mapping,
            offset=self._data_start + start,
        )
        copied = np.array(view[np.asarray(row_indices, dtype=np.intp)], copy=True)
        if dtype_name == "BF16":
            copied = (copied.astype(np.uint32) << np.uint32(16)).view(np.float32)
        return copied, dtype_name

    def close(self) -> None:
        mapping = getattr(self, "_mapping", None)
        if mapping is not None:
            mapping.close()
            self._mapping = None
        file_object = getattr(self, "_file", None)
        if file_object is not None:
            file_object.close()
            self._file = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class DiskBackedQuantizedEmbedding(nn.Module):
    """Quantized embedding that copies only requested rows from SSD-backed mmap."""

    def __init__(
        self,
        model_path: str | Path,
        prefix: str,
        num_embeddings: int,
        dims: int,
        num_shards: int,
        group_size: int = 32,
        bits: int = 4,
        mode: str = "affine",
    ):
        super().__init__()
        if mode != "affine":
            raise ValueError("Qwen4-Exp PLE currently supports affine shards")
        if num_embeddings % num_shards:
            raise ValueError("embedding rows must divide evenly into shards")
        if dims % group_size:
            raise ValueError(
                "embedding dimensions must divide into quantization groups"
            )

        self.num_embeddings = num_embeddings
        self.dims = dims
        self.num_shards = num_shards
        self.shard_rows = num_embeddings // num_shards
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        self.last_touched_shards: tuple[int, ...] = ()
        self.rows_read = 0
        self._prefix = prefix
        self._readers: dict[str, _SafeTensorMMap] = {}
        self._tensor_readers: dict[str, _SafeTensorMMap] = {}

        model_path = Path(model_path)
        index_path = model_path / "model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"SSD-backed PLE requires a safetensors index: {index_path}"
            )
        weight_map = json.loads(index_path.read_text()).get("weight_map", {})
        expected_shapes = {
            "weight": (self.shard_rows, dims * bits // 32),
            "scales": (self.shard_rows, dims // group_size),
            "biases": (self.shard_rows, dims // group_size),
        }
        for shard_index in range(num_shards):
            for suffix, expected_shape in expected_shapes.items():
                key = f"{prefix}.shard_{shard_index}.{suffix}"
                try:
                    filename = weight_map[key]
                except KeyError as exc:
                    raise KeyError(
                        f"SSD-backed PLE tensor {key!r} is absent from the index"
                    ) from exc
                reader = self._readers.get(filename)
                if reader is None:
                    reader = _SafeTensorMMap(model_path / filename)
                    self._readers[filename] = reader
                if reader.tensor_shape(key) != expected_shape:
                    raise ValueError(
                        f"Unexpected shape for {key}: "
                        f"{reader.tensor_shape(key)} != {expected_shape}"
                    )
                self._tensor_readers[key] = reader

    def _read_rows(self, key: str, row_indices: list[int]) -> mx.array:
        array, dtype_name = self._tensor_readers[key].rows(key, row_indices)
        self.rows_read += len(row_indices)
        result = mx.array(array)
        if dtype_name == "BF16":
            result = result.astype(mx.bfloat16)
        return result

    def __call__(self, ids: mx.array) -> mx.array:
        original_shape = ids.shape
        flat_ids = ids.reshape(-1).astype(mx.int64)
        mx.eval(flat_ids)
        host_ids = [int(value) for value in flat_ids.tolist()]
        if any(value < 0 or value >= self.num_embeddings for value in host_ids):
            raise IndexError("n-gram embedding id is outside the padded vocabulary")

        touched = tuple(sorted({value // self.shard_rows for value in host_ids}))
        self.last_touched_shards = touched
        self.rows_read = 0
        output = None
        for shard_index in touched:
            positions_list = [
                index
                for index, value in enumerate(host_ids)
                if value // self.shard_rows == shard_index
            ]
            local_ids_list = [
                host_ids[index] % self.shard_rows for index in positions_list
            ]
            base = f"{self._prefix}.shard_{shard_index}"
            weight = self._read_rows(f"{base}.weight", local_ids_list)
            scales = self._read_rows(f"{base}.scales", local_ids_list)
            biases = self._read_rows(f"{base}.biases", local_ids_list)
            values = mx.dequantize(
                weight,
                scales=scales,
                biases=biases,
                group_size=self.group_size,
                bits=self.bits,
                mode=self.mode,
            )
            if output is None:
                output = mx.zeros((len(host_ids), self.dims), dtype=values.dtype)
            positions = mx.array(positions_list, dtype=mx.int32)
            output = output.at[positions].add(values)

        if output is None:
            output = mx.zeros((0, self.dims), dtype=mx.bfloat16)
        return output.reshape(*original_shape, self.dims)


class NGramEmbedding(nn.Module):
    """Qwen4 PLE hashing plus the checkpoint's 128 quantized row shards."""

    def __init__(
        self,
        args: TextConfig,
        layer_idx: int,
        ple_layer_index: int = 0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hasher = NGramHasher(args, ple_layer_index=ple_layer_index)
        self.context_len = self.hasher.context_len
        self.eos_token_id = self.hasher.eos_token_id
        self.layer_multipliers = mx.array(self.hasher.layer_multipliers, dtype=mx.int64)
        self.ngram_heads_vocab_sizes = mx.array(
            self.hasher.head_vocab_sizes, dtype=mx.int64
        )
        self.ngram_heads_offsets = mx.array(self.hasher.head_offsets, dtype=mx.int64)
        head_dims = args.ple_embed_dim // self.hasher.ngram_heads
        embedding_args = {
            "num_embeddings": self.hasher.padded_vocab_size,
            "dims": head_dims,
            "num_shards": args.split_ngram_parts,
            "group_size": 32,
            "bits": 4,
            "mode": "affine",
        }
        if _PLE_RUNTIME_MODE == "mmap":
            if _PLE_RUNTIME_MODEL_PATH is None:
                raise RuntimeError("SSD-backed PLE has no configured model path")
            prefix = (
                f"language_model.model.layers.{layer_idx}.ple.ple_embedding."
                "ngram_embedding"
            )
            self.ngram_embedding = DiskBackedQuantizedEmbedding(
                model_path=_PLE_RUNTIME_MODEL_PATH,
                prefix=prefix,
                **embedding_args,
            )
        else:
            self.ngram_embedding = ShardedQuantizedEmbedding(**embedding_args)

    def __call__(self, input_ids: mx.array, cache=None) -> mx.array:
        input_ids = input_ids.astype(mx.int64)
        batch_size, seq_len = input_ids.shape
        previous = None if cache is None else cache[3]
        if previous is None or previous.shape[0] != batch_size:
            previous = mx.full(
                (batch_size, self.context_len),
                self.eos_token_id,
                dtype=mx.int64,
            )
        token_history = mx.concatenate([previous, input_ids], axis=1)
        if cache is not None:
            cache[3] = token_history[:, -self.context_len :]
        ngram_ids = self.hasher.compute_ids(
            token_history,
            current_length=seq_len,
            layer_multipliers=self.layer_multipliers,
            head_vocab_sizes=self.ngram_heads_vocab_sizes,
            head_offsets=self.ngram_heads_offsets,
        )
        embeddings = self.ngram_embedding(ngram_ids)
        return embeddings.reshape(*embeddings.shape[:-2], -1)


class PLEDepthwiseConv1d(nn.Module):
    """Depthwise dilated convolution using the checkpoint's PyTorch layout."""

    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.weight = mx.zeros((channels, 1, kernel_size))

    def __call__(self, x: mx.array) -> mx.array:
        weight = mx.moveaxis(self.weight, 2, 1)
        return mx.conv1d(
            x,
            weight,
            stride=1,
            padding=0,
            dilation=self.dilation,
            groups=self.channels,
        )


class PLELayer(nn.Module):
    """Per-layer hashed lexical embedding injection."""

    def __init__(self, args: TextConfig, layer_idx: int, ple_layer_index: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = args.hidden_size
        self.hc_count = args.hc_count
        hc_hidden_size = self.hidden_size * self.hc_count
        self.ple_embedding = NGramEmbedding(
            args,
            layer_idx=layer_idx,
            ple_layer_index=ple_layer_index,
        )
        self.key_proj = nn.Linear(args.ple_embed_dim, hc_hidden_size, bias=False)
        self.value_proj = nn.Linear(args.ple_embed_dim, self.hidden_size, bias=False)
        self.norm_key = GroupedRMSNorm(
            hc_hidden_size,
            group_size=self.hidden_size,
            eps=args.rms_norm_eps,
        )
        self.norm_query = GroupedRMSNorm(
            hc_hidden_size,
            group_size=self.hidden_size,
            eps=args.rms_norm_eps,
        )
        self.norm_conv = GroupedRMSNorm(
            hc_hidden_size,
            group_size=self.hidden_size,
            eps=args.rms_norm_eps,
        )
        self.conv_dilation = args.ngram_size
        self.short_conv_state_len = (args.ple_conv_kernel_size - 1) * self.conv_dilation
        self.conv1d = PLEDepthwiseConv1d(
            hc_hidden_size,
            kernel_size=args.ple_conv_kernel_size,
            dilation=self.conv_dilation,
        )

    @staticmethod
    def _apply_mask(hidden_states: mx.array, mask: mx.array | None) -> mx.array:
        if mask is None:
            return hidden_states
        if mask.ndim > 2:
            mask = mask.reshape(mask.shape[0], -1)[:, -hidden_states.shape[1] :]
        return mx.where(mask[..., None], hidden_states, 0)

    def _short_conv(self, hidden_states: mx.array, cache=None) -> mx.array:
        batch_size, seq_len, channels = hidden_states.shape
        previous = None if cache is None else cache[2]
        if previous is None or previous.shape[0] != batch_size:
            previous = mx.zeros(
                (batch_size, self.short_conv_state_len, channels),
                dtype=hidden_states.dtype,
            )
        conv_input = mx.concatenate([previous, hidden_states], axis=1)
        if cache is not None:
            cache[2] = mx.contiguous(conv_input[:, -self.short_conv_state_len :, :])
        conv_input = conv_input[:, -(self.short_conv_state_len + seq_len) :, :]
        return nn.silu(self.conv1d(conv_input))

    def __call__(
        self,
        hidden_states: mx.array,
        input_ids: mx.array,
        cache=None,
        conv_mask: mx.array | None = None,
    ) -> mx.array:
        embeddings = self.ple_embedding(input_ids, cache)
        key = self.norm_key(self.key_proj(embeddings)).reshape(
            *hidden_states.shape[:-1], self.hc_count, self.hidden_size
        )
        value = self.value_proj(embeddings)
        query = self.norm_query(hidden_states).reshape(
            *hidden_states.shape[:-1], self.hc_count, self.hidden_size
        )
        gate = mx.sum(key * query, axis=-1, keepdims=True) / math.sqrt(self.hidden_size)
        gate = mx.sqrt(mx.maximum(mx.abs(gate), 1e-6)) * mx.sign(gate)
        gated_value = mx.sigmoid(gate) * value[..., None, :]
        gated_value = gated_value.reshape(*hidden_states.shape)
        normalized = self.norm_conv(gated_value)
        gated_value = self._apply_mask(gated_value, conv_mask)
        normalized = self._apply_mask(normalized, conv_mask)
        return gated_value + self._short_conv(normalized, cache)


class QSAKVCache(KVCache):
    """KV cache with the raw single-head keys required by QSA."""

    def __init__(self):
        super().__init__()
        self.indexer_keys = None
        self.indexer_offset = 0

    def update_indexer(self, raw_keys: mx.array) -> mx.array:
        previous = self.indexer_offset
        length = raw_keys.shape[1]
        if self.indexer_keys is None or previous + length > self.indexer_keys.shape[1]:
            batch_size, _, dims = raw_keys.shape
            steps = (self.step + length - 1) // self.step
            extension = mx.zeros(
                (batch_size, steps * self.step, dims), dtype=raw_keys.dtype
            )
            if self.indexer_keys is None:
                self.indexer_keys = extension
            else:
                self.indexer_keys = mx.concatenate(
                    [self.indexer_keys[:, :previous, :], extension], axis=1
                )
        self.indexer_offset += length
        self.indexer_keys[:, previous : self.indexer_offset, :] = raw_keys
        return self.indexer_keys[:, : self.indexer_offset, :]

    def trim(self, count):
        trimmed = super().trim(count)
        self.indexer_offset = max(0, self.indexer_offset - trimmed)
        return trimmed

    @property
    def nbytes(self):
        base = super().nbytes
        if self.indexer_keys is None:
            return base
        return base + self.indexer_keys.nbytes


class QSAIndexer(nn.Module):
    """Qwen Sparse Attention block selector."""

    def __init__(self, args: TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.index_n_heads = args.indexer_n_heads
        self.index_kv_heads = args.indexer_kv_heads
        self.index_head_dim = args.indexer_head_dim
        self.token_budget = args.indexer_budget
        self.compress_ratio = args.indexer_compress_ratio
        self.block_topk = self.token_budget // self.compress_ratio
        self.index_qk_proj = nn.Linear(
            args.hidden_size,
            (self.index_n_heads + self.index_kv_heads) * self.index_head_dim,
            bias=False,
        )
        self.q_layernorm = GroupedRMSNorm(self.index_head_dim, eps=args.rms_norm_eps)
        self.k_layernorm = GroupedRMSNorm(self.index_head_dim, eps=args.rms_norm_eps)
        self.rotary_dim = int(
            args.head_dim * args.rope_parameters.get("partial_rotary_factor", 1.0)
        )
        base = args.rope_parameters.get("rope_theta", 10_000)
        self._inv_freq = (
            1.0
            / (
                base
                ** (
                    mx.arange(0, self.rotary_dim, 2, dtype=mx.float32)
                    / max(self.rotary_dim, 1)
                )
            )
            if self.rotary_dim
            else mx.zeros((0,), dtype=mx.float32)
        )

    @property
    def inv_freq(self):
        return self._inv_freq

    @staticmethod
    def _rotate_half(x: mx.array) -> mx.array:
        half = x.shape[-1] // 2
        return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)

    def _apply_rope(self, x: mx.array, positions: mx.array | int) -> mx.array:
        if self.rotary_dim == 0:
            return x
        positions = mx.array(positions, dtype=mx.float32)
        frequencies = positions[..., None] * self.inv_freq
        angles = mx.concatenate([frequencies, frequencies], axis=-1)
        rotary, passthrough = x[..., : self.rotary_dim], x[..., self.rotary_dim :]
        rotary = rotary * mx.cos(angles) + self._rotate_half(rotary) * mx.sin(angles)
        return mx.concatenate([rotary, passthrough], axis=-1)

    def select_token_indices(
        self,
        query: mx.array,
        raw_keys: mx.array,
        *,
        visible_length: int,
        query_position: int,
    ) -> mx.array:
        visible_length = min(int(visible_length), int(raw_keys.shape[0]))
        complete_blocks = visible_length // self.compress_ratio
        if complete_blocks:
            block_tokens = mx.arange(
                complete_blocks * self.compress_ratio, dtype=mx.int32
            ).reshape(complete_blocks, self.compress_ratio)
            pooled_keys = mx.mean(
                raw_keys[: complete_blocks * self.compress_ratio]
                .reshape(complete_blocks, self.compress_ratio, self.index_head_dim)
                .astype(mx.float32),
                axis=1,
            ).astype(raw_keys.dtype)
            pooled_keys = self.k_layernorm(pooled_keys)
            pooled_keys = self._apply_rope(pooled_keys, block_tokens[:, 0])
            query = self.q_layernorm(query)
            query = self._apply_rope(query, query_position)
            scores = mx.sum(
                query[:, None, :].astype(mx.float32)
                * pooled_keys[None, :, :].astype(mx.float32),
                axis=-1,
            )
            scores = mx.sum(mx.maximum(scores, 0), axis=0) / math.sqrt(
                self.index_head_dim
            )
            count = min(self.block_topk, complete_blocks)
            selected_blocks = mx.argpartition(scores, kth=-count)[-count:]
            selected = block_tokens[selected_blocks].reshape(-1)
        else:
            selected = mx.zeros((0,), dtype=mx.int32)

        tail = mx.arange(
            complete_blocks * self.compress_ratio,
            visible_length,
            dtype=mx.int32,
        )
        return mx.concatenate([selected, tail])

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        cache: QSAKVCache | None = None,
        position_ids: mx.array | None = None,
    ) -> mx.array:
        batch_size, seq_length, _ = hidden_states.shape
        projected = self.index_qk_proj(hidden_states)
        query_dims = self.index_n_heads * self.index_head_dim
        queries, raw_keys = mx.split(projected, [query_dims], axis=-1)
        queries = queries.reshape(
            batch_size, seq_length, self.index_n_heads, self.index_head_dim
        )
        raw_keys = raw_keys.reshape(batch_size, seq_length, self.index_head_dim)
        if cache is not None:
            all_raw_keys = cache.update_indexer(raw_keys)
        else:
            all_raw_keys = raw_keys
        previous_length = all_raw_keys.shape[1] - seq_length

        key_length = all_raw_keys.shape[1]
        selected_mask = mx.zeros(
            (batch_size, 1, seq_length, key_length), dtype=mx.bool_
        )
        text_positions = position_ids
        if text_positions is not None and text_positions.ndim == 3:
            text_positions = text_positions[0]
        for batch_index in range(batch_size):
            for query_index in range(seq_length):
                visible_length = previous_length + query_index + 1
                if text_positions is None:
                    query_position = visible_length - 1
                else:
                    query_position = int(
                        text_positions[batch_index, query_index].item()
                    )
                selected = self.select_token_indices(
                    queries[batch_index, query_index],
                    all_raw_keys[batch_index],
                    visible_length=visible_length,
                    query_position=query_position,
                )
                selected_mask = selected_mask.at[
                    batch_index, 0, query_index, selected
                ].add(mx.ones(selected.shape, dtype=mx.bool_))

        minimum = mx.array(-1e9, dtype=hidden_states.dtype)
        return mx.where(selected_mask, mx.array(0, hidden_states.dtype), minimum)


class Qwen4ExpRMSNormGated(nn.Module):
    """Gated RMSNorm with Qwen4's selectable sigmoid output gate."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        activation: str = "sigmoid",
    ):
        super().__init__()
        self.eps = eps
        self.activation = activation
        self.weight = mx.ones((hidden_size,))

    def __call__(self, hidden_states: mx.array, gate: mx.array | None = None):
        normalized = mx.fast.rms_norm(hidden_states, self.weight, self.eps)
        if gate is None:
            return normalized.astype(hidden_states.dtype)
        gate = gate.astype(mx.float32)
        if self.activation == "sigmoid":
            gate = mx.sigmoid(gate)
        elif self.activation == "silu":
            gate = nn.silu(gate)
        else:
            raise ValueError(f"Unsupported Qwen4 output gate: {self.activation}")
        return (gate * normalized.astype(mx.float32)).astype(hidden_states.dtype)


class Qwen4ExpGatedDeltaNet(Qwen3_5GatedDeltaNet):
    def __init__(self, args: TextConfig):
        super().__init__(args)
        self.norm = Qwen4ExpRMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            activation=args.output_gate_type or "silu",
        )


class Qwen4ExpAttention(Qwen3_5Attention):
    def __init__(self, args: TextConfig, layer_idx: int):
        super().__init__(args)
        self.q_norm = GroupedRMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = GroupedRMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.indexer = QSAIndexer(args, layer_idx=layer_idx)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: QSAKVCache | None = None,
        position_ids: mx.array | None = None,
        position_embeddings=None,
        target_verify: bool = False,
    ) -> mx.array:
        qsa_mask = self.indexer(
            x,
            cache=cache,
            position_ids=position_ids,
        )
        if isinstance(mask, mx.array):
            if mask.dtype == mx.bool_:
                mask = mx.where(mask, mx.array(0, x.dtype), mx.array(-1e9, x.dtype))
            mask = mask + qsa_mask
        else:
            mask = qsa_mask
        return super().__call__(
            x,
            mask=mask,
            cache=cache,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            target_verify=target_verify,
        )


class Qwen4ExpDecoderLayer(nn.Module):
    def __init__(self, args: TextConfig, layer_idx: int):
        super().__init__()
        self.layer_type = args.layer_types[layer_idx]
        self.is_linear = self.layer_type == "linear_attention"
        if self.is_linear:
            self.linear_attn = Qwen4ExpGatedDeltaNet(args)
        else:
            self.self_attn = Qwen4ExpAttention(args, layer_idx=layer_idx)
        self.mlp = Qwen3_5MoeSparseMoeBlock(args)
        ple_index = (
            args.ple_layer_ids.index(layer_idx + 1)
            if layer_idx + 1 in args.ple_layer_ids
            else None
        )
        self.ple = (
            PLELayer(args, layer_idx=layer_idx, ple_layer_index=ple_index)
            if ple_index is not None
            else None
        )
        self.attn_hyper_connection = GatedResidual(args)
        self.mlp_hyper_connection = GatedResidual(args)

    @staticmethod
    def _inject(
        block_output: mx.array,
        hyper_input: mx.array,
        injection_weights: mx.array,
    ) -> mx.array:
        injection = block_output[..., None, :] * injection_weights[..., :, None]
        return hyper_input + injection.reshape(*hyper_input.shape)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        attention_mask=None,
        conv_mask=None,
        cache=None,
        position_ids=None,
        ple_input_ids=None,
        target_verify: bool = False,
    ) -> mx.array:
        if self.ple is not None:
            hidden_states = hidden_states + self.ple(
                hidden_states,
                ple_input_ids,
                cache=cache,
                conv_mask=conv_mask,
            )

        mixed, hyper_input, injection_weights = self.attn_hyper_connection(
            hidden_states
        )
        if self.is_linear:
            block_output = self.linear_attn(
                mixed,
                mask=conv_mask,
                cache=cache,
                target_verify=target_verify,
            )
        else:
            block_output = self.self_attn(
                mixed,
                mask=attention_mask,
                cache=cache,
                position_ids=position_ids,
                target_verify=target_verify,
            )
        hidden_states = self._inject(block_output, hyper_input, injection_weights)

        mixed, hyper_input, injection_weights = self.mlp_hyper_connection(hidden_states)
        block_output = self.mlp(mixed, target_verify=target_verify)
        return self._inject(block_output, hyper_input, injection_weights)


class Qwen4ExpModel(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Qwen4ExpDecoderLayer(args, layer_idx)
            for layer_idx in range(args.num_hidden_layers)
        ]
        self.hyper_connection_mixer = GatedResidual(args, use_combine=False)
        self.ssm_idx = next(
            index for index, layer in enumerate(self.layers) if layer.is_linear
        )
        self.fa_idx = next(
            index for index, layer in enumerate(self.layers) if not layer.is_linear
        )

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: mx.array | None = None,
        mask=None,
        cache=None,
        position_ids=None,
        capture_layer_ids=None,
        hidden_sink=None,
        gdn_sink=None,
    ) -> mx.array:
        del mask, capture_layer_ids, hidden_sink
        hidden_states = (
            self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        )
        if cache is None:
            cache = [None] * len(self.layers)
        attention_mask = _create_qwen3_5_attention_mask(
            hidden_states, cache[self.fa_idx]
        )
        conv_mask = _create_qwen3_5_ssm_mask(hidden_states, cache[self.ssm_idx])
        hidden_states = mx.tile(hidden_states, (1, 1, self.args.hc_count))
        for layer, layer_cache in zip(self.layers, cache):
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                conv_mask=conv_mask,
                cache=layer_cache,
                position_ids=position_ids,
                ple_input_ids=inputs,
                target_verify=gdn_sink is not None,
            )
        return self.hyper_connection_mixer(hidden_states)


class LanguageModel(Qwen3_5LanguageModel):
    def __init__(self, args: TextConfig, config=None):
        nn.Module.__init__(self)
        self.args = args
        self.config = config
        self.model_type = args.model_type
        self.model = Qwen4ExpModel(args)
        self._rope_deltas = None
        self._position_ids = None
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def make_cache(self):
        return [
            ArraysCache(size=4) if layer.is_linear else QSAKVCache()
            for layer in self.layers
        ]


class GroupedRMSNorm(nn.Module):
    """RMS-normalize each residual stream while keeping one flat weight."""

    def __init__(self, dims: int, group_size: int | None = None, eps: float = 1e-6):
        super().__init__()
        if group_size is not None and dims % group_size:
            raise ValueError(
                f"dims ({dims}) must be divisible by group_size ({group_size})"
            )
        self.dims = dims
        self.group_size = group_size
        self.eps = eps
        self.weight = mx.ones((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        original_shape = x.shape
        if self.group_size is not None:
            x = x.reshape(*x.shape[:-1], -1, self.group_size)
        x = x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        if self.group_size is not None:
            x = x.reshape(original_shape)
        return x * self.weight


class GatedResidual(nn.Module):
    """Qwen4's learned hyper-connection mixer and block injector."""

    def __init__(self, args: TextConfig, use_combine: bool = True):
        super().__init__()
        self.hc_count = args.hc_count
        self.hidden_size = args.hidden_size
        self.hc_lowrank = args.hc_lowrank
        hc_hidden_size = self.hc_count * self.hidden_size
        self.hc_norm = GroupedRMSNorm(
            hc_hidden_size,
            group_size=self.hidden_size,
            eps=args.rms_norm_eps,
        )
        self.input_mix_weight_down = nn.Linear(
            hc_hidden_size, args.hc_lowrank, bias=False
        )
        self.input_mix_weight_up = nn.Linear(
            args.hc_lowrank, hc_hidden_size, bias=False
        )
        self.block_inject_weight = (
            nn.Linear(hc_hidden_size, self.hc_count, bias=False)
            if use_combine
            else None
        )

    def __call__(self, hyper_input: mx.array):
        expected = self.hc_count * self.hidden_size
        if hyper_input.shape[-1] != expected:
            raise ValueError(
                f"Expected {expected} hyper-connection features, got {hyper_input.shape[-1]}"
            )
        compiled_forward = getattr(self, "_compiled_forward", None)
        if compiled_forward is not None and hyper_input.shape[-2] == 1:
            return compiled_forward(hyper_input)
        return self._forward(hyper_input)

    def _forward(self, hyper_input: mx.array):
        normalized = self.hc_norm(hyper_input)
        input_inject_weight = getattr(self, "input_inject_weight", None)
        if input_inject_weight is None:
            input_mix = self.input_mix_weight_down(normalized)
            block_injection = (
                self.block_inject_weight(normalized)
                if self.block_inject_weight is not None
                else None
            )
        else:
            combined = input_inject_weight(normalized)
            input_indices, injection_indices = _hyper_split_indices(
                self.hc_lowrank, self.hc_count
            )
            input_mix = mx.take(combined, input_indices, axis=-1)
            block_injection = mx.take(combined, injection_indices, axis=-1)
        input_mix = nn.silu(input_mix / self.hc_count)
        input_mix = mx.sigmoid(self.input_mix_weight_up(input_mix))
        input_mix = input_mix.reshape(
            *input_mix.shape[:-1], self.hc_count, self.hidden_size
        )
        streams = normalized.reshape(
            *normalized.shape[:-1], self.hc_count, self.hidden_size
        )
        mixed = mx.mean(input_mix * streams, axis=-2)

        if block_injection is None:
            return mixed
        injection = 2 * mx.sigmoid(block_injection / self.hc_count)
        return mixed, hyper_input, injection


def _can_fuse_hyper_connection(module: GatedResidual) -> bool:
    if hasattr(module, "input_inject_weight"):
        return False
    injection = getattr(module, "block_inject_weight", None)
    down = getattr(module, "input_mix_weight_down", None)
    if injection is None or down is None or type(down) is not type(injection):
        return False
    if down.weight.shape[1:] != injection.weight.shape[1:]:
        return False
    if down.weight.dtype != injection.weight.dtype:
        return False
    for attribute in ("group_size", "bits", "mode"):
        if getattr(down, attribute, None) != getattr(injection, attribute, None):
            return False
    for tensor_name in ("scales", "biases", "bias"):
        if hasattr(down, tensor_name) != hasattr(injection, tensor_name):
            return False
    return True


def fuse_hyper_connection_projections(model: nn.Module) -> int:
    """Fuse same-input low-rank and injection projections row-wise."""
    modules = [model]
    modules.extend(module for _, module in model.named_modules() if module is not model)
    targets = []
    seen = set()
    for module in modules:
        if id(module) in seen:
            continue
        seen.add(id(module))
        if isinstance(module, GatedResidual) and _can_fuse_hyper_connection(module):
            targets.append(module)

    for module in targets:
        down = module.input_mix_weight_down
        injection = module.block_inject_weight
        fused = {"weight": mx.concatenate([down.weight, injection.weight], axis=0)}
        for tensor_name in ("scales", "biases", "bias"):
            if hasattr(down, tensor_name):
                fused[tensor_name] = mx.concatenate(
                    [getattr(down, tensor_name), getattr(injection, tensor_name)],
                    axis=0,
                )
        mx.eval(*fused.values())
        for tensor_name, value in fused.items():
            setattr(down, tensor_name, value)
        module.input_inject_weight = down
        del module.input_mix_weight_down
        del module.block_inject_weight
    return len(targets)


def compile_hyper_connections(model: nn.Module) -> int:
    """Compile each hyper-connection's single-token decode path once."""
    modules = [model]
    modules.extend(module for _, module in model.named_modules() if module is not model)
    compiled = 0
    seen = set()
    for module in modules:
        if id(module) in seen:
            continue
        seen.add(id(module))
        if not isinstance(module, GatedResidual) or hasattr(
            module, "_compiled_forward"
        ):
            continue
        module._compiled_forward = mx.compile(module._forward)
        compiled += 1
    return compiled


Qwen4ExpTextRMSNorm = GroupedRMSNorm
Qwen4ExpTextGatedResidual = GatedResidual

__all__ = [
    "GatedResidual",
    "GroupedRMSNorm",
    "Qwen4ExpTextGatedResidual",
    "Qwen4ExpTextRMSNorm",
]
