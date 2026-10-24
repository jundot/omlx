"""Read-only, row-addressable PLE storage for stock MLX-VLM Qwen4.

No model-sized MLX arrays are allocated by this module. Safetensors headers are
validated before mapping. Only requested rows are copied to MLX and dequantized.
The stock constructor interception is temporary and local to the load context.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import functools
import inspect
import logging
import math
import mmap
import os
from pathlib import Path
import struct
import threading
from typing import Any

import numpy as np

from .row_cache import PackedRowCache
from .checkpoint import Checkpoint, read_header
from .metadata import canonical, normalize_quantization, ple_layer_indices, safe_shard_path

log = logging.getLogger('jangh2_bridge')
_DTYPES = {'U32': np.dtype('<u4'), 'F16': np.dtype('<f2'),
           'BF16': np.dtype('<u2'), 'F32': np.dtype('<f4')}


# A v3-compatible, single-file replacement. No dependency / OS-limit changes.
PLE_STORAGE_VERSION = 4


@dataclass(frozen=True)
class _TensorSlice:
    """Validated metadata only; does not retain a file or a mapped array."""
    path: Path
    dtype: np.dtype
    shape: tuple[int, ...]
    offset: int
    file_size: int


class _FileMappingCache:
    """Bounded, read-only mappings. Mapped views never escape a locked read.

    Python 3.11 mmap can retain a descriptor after the source file is closed.
    Keep a configurable number of physical files mapped, not one per tensor.
    Evict BEFORE opening a replacement so even the opening peak is bounded.
    The open() descriptor is temporary; only mmap's descriptor persists.
    """
    def __init__(self, max_open_files: int = 8):
        if type(max_open_files) is not int or max_open_files < 1:
            raise ValueError('max_open_files must be a positive integer')
        self._limit = max_open_files
        self._maps: OrderedDict[Path, mmap.mmap] = OrderedDict()
        self._lock = threading.RLock()
        self._closed = False
        self._peak = 0
        self._opens = 0
        self._evictions = 0

    def read_rows(self, tensor: _TensorSlice, ids: np.ndarray) -> np.ndarray:
        """Return independent row copies, safe after eviction / close.

        No file-wide bytes read or array copy occurs. The temporary ndarray is
        only a view over the mmap; integer-array indexing copies requested rows.
        """
        ids = np.asarray(ids)
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
            raise TypeError('Mapping row indices must be a one-dimensional integer array')
        if ids.size and (np.any(ids < 0) or np.any(ids >= tensor.shape[0])):
            raise IndexError('Mapping row index outside tensor')
        with self._lock:
            if self._closed:
                raise RuntimeError('PLE mapping cache has been closed')
            mm = self._maps.get(tensor.path)
            if mm is None:
                if len(self._maps) >= self._limit:
                    _, old = self._maps.popitem(last=False)
                    old.close()
                    self._evictions += 1
                with tensor.path.open('rb') as stream:
                    actual_size = os.fstat(stream.fileno()).st_size
                    if actual_size != tensor.file_size:
                        raise ValueError(f'PLE shard size changed after validation: {tensor.path}')
                    end = tensor.offset + math.prod(tensor.shape) * tensor.dtype.itemsize
                    if tensor.offset < 0 or end > actual_size:
                        raise ValueError(f'PLE tensor exceeds mapped file: {tensor.path}')
                    mm = mmap.mmap(stream.fileno(), actual_size, access=mmap.ACCESS_READ)
                self._maps[tensor.path] = mm
                self._opens += 1
                self._peak = max(self._peak, len(self._maps))
            else:
                self._maps.move_to_end(tensor.path)
            # Shape/offset descriptors for other tensors in a cached file are
            # validated too, without reading the payload into resident memory.
            end = tensor.offset + math.prod(tensor.shape) * tensor.dtype.itemsize
            if tensor.offset < 0 or end > len(mm):
                raise ValueError(f'PLE tensor exceeds mapped file: {tensor.path}')
            view = np.ndarray(tensor.shape, dtype=tensor.dtype, buffer=mm,
                              offset=tensor.offset, order='C')
            try:
                # Advanced integer indexing produces a copy. Explicitly delete
                # the view even on failure so a retained traceback cannot keep
                # an exported buffer alive when close() or eviction runs.
                return view[ids]
            finally:
                del view

    def stats(self) -> dict[str, int | bool]:
        with self._lock:
            return {'limit': self._limit, 'open_files': len(self._maps),
                    'peak_open_files': self._peak, 'file_opens': self._opens,
                    'evictions': self._evictions, 'read_only': True,
                    'closed': self._closed}

    def close(self) -> None:
        with self._lock:
            self._closed = True
            while self._maps:
                _, mm = self._maps.popitem(last=False)
                mm.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class IndexedMMapEmbedding:
    """Plain callable (not an nn.Module): the table is not a model parameter.

    Supports explicitly validated affine Q4/Q8 shards and F16/BF16/F32 scale /
    bias arrays. Names, shapes, dtypes and quantization must all agree. Mapping
    reserves virtual address ranges, not a copy of the complete embedding table.
    """
    def __init__(self, checkpoint: Checkpoint, layer_idx: int,
                 num_embeddings: int, dims: int, num_shards: int, *,
                 max_open_files: int = 8, row_cache_bytes: int = 0,
                 device_cache_bytes: int = 0):
        self._lock = threading.RLock()
        self._shards: list[tuple[int, int, dict[str, _TensorSlice]]] = []
        self._mapping_cache = _FileMappingCache(max_open_files=max_open_files)
        if type(row_cache_bytes) is not int or row_cache_bytes < 0:
            raise ValueError("row_cache_bytes must be a nonnegative integer")
        self._row_cache = None
        if type(device_cache_bytes) is not int or device_cache_bytes < 0:
            raise ValueError('device_cache_bytes must be nonnegative')
        self._device_cache_bytes = device_cache_bytes
        self._device_cache = None
        self.disk_rows_read = 0
        self._tensor_count = 0
        self._unique_files = 0
        self._closed = False
        self.rows_read = 0
        self.row_count, self.row_width = int(num_embeddings), int(dims)
        self.dims = self.row_width
        self.bits = self.group_size = 0
        self.aux_dtype = ''
        if (num_embeddings <= 0 or dims <= 0 or num_shards <= 0
                or num_shards > num_embeddings):
            raise ValueError('Invalid PLE dimensions / shard count')
        prefix = f'language_model.model.layers.{layer_idx}.ple.ple_embedding.ngram_embedding.shards.'
        selected: dict[str, str] = {}
        for raw_key in checkpoint.weight_map:
            key = canonical(raw_key)
            if not key.startswith(prefix):
                continue
            if key in selected:
                raise ValueError(f'Ambiguous PLE tensor aliases: {selected[key]!r}, {raw_key!r}')
            selected[key] = raw_key
        expected = {f'{prefix}{i}.{name}' for i in range(num_shards)
                    for name in ('weight', 'scales', 'biases')}
        if set(selected) != expected:
            raise ValueError('PLE index does not match the configured shards: '
                             f'missing={sorted(expected-set(selected))[:6]}, '
                             f'extra={sorted(set(selected)-expected)[:6]}')
        quant = normalize_quantization(checkpoint.config)
        headers: dict[str, tuple[int, dict, int]] = {}
        base_rows, remainder = divmod(num_embeddings, num_shards)
        row_start = 0
        try:
            for index in range(num_shards):
                key_prefix = f'{prefix}{index}'
                spec = {k: quant.get(k) for k in ('bits', 'group_size')}
                spec['mode'] = quant.get('mode', 'affine')
                override = quant.get(key_prefix)
                if override is not None:
                    if not isinstance(override, dict):
                        raise ValueError(f'Invalid PLE quantization override: {key_prefix}')
                    spec.update(override)
                if (set(spec) != {'bits', 'group_size', 'mode'}
                        or spec['mode'] != 'affine'
                        or type(spec['bits']) is not int or spec['bits'] not in (4, 8)
                        or type(spec['group_size']) is not int or spec['group_size'] not in (32, 64, 128)):
                    raise ValueError(f'Unsupported PLE quantization at {key_prefix}: {spec}')
                bits, group = spec['bits'], spec['group_size']
                if dims % group or dims * bits % 32:
                    raise ValueError(f'PLE width {dims} is incompatible with {spec}')
                if index == 0:
                    self.bits, self.group_size = bits, group
                elif (bits, group) != (self.bits, self.group_size):
                    raise ValueError('Mixed PLE shard quantization is not supported')
                rows = base_rows + (index < remainder)
                arrays: dict[str, _TensorSlice] = {}
                # Add the container before filling so partial failures close it.
                self._shards.append((row_start, row_start + rows, arrays))
                for name in ('weight', 'scales', 'biases'):
                    raw_key = selected[key_prefix + '.' + name]
                    filename = checkpoint.weight_map[raw_key]
                    path = safe_shard_path(checkpoint.root, filename)
                    if filename not in headers:
                        header = read_header(path)
                        with path.open('rb') as stream:
                            payload_start = 8 + struct.unpack('<Q', stream.read(8))[0]
                        headers[filename] = payload_start, header, path.stat().st_size
                    payload_start, header, file_size = headers[filename]
                    if raw_key not in header:
                        raise ValueError(f'Indexed PLE tensor is missing: {raw_key}')
                    item = header[raw_key]
                    dtype = item['dtype']
                    shape = tuple(item['shape'])
                    wanted = (rows, dims * bits // 32 if name == 'weight' else dims // group)
                    allowed_dtype = ('U32',) if name == 'weight' else ('F16', 'BF16', 'F32')
                    if shape != wanted or dtype not in allowed_dtype:
                        raise ValueError(f'Invalid PLE {name}: {raw_key}, shape={shape}, '
                                         f'dtype={dtype}; expected shape={wanted}, dtype in {allowed_dtype}')
                    if name != 'weight':
                        if not self.aux_dtype:
                            self.aux_dtype = dtype
                        elif dtype != self.aux_dtype:
                            raise ValueError('PLE scales and biases must share one floating dtype')
                    arrays[name] = _TensorSlice(
                        path=path, dtype=_DTYPES[dtype], shape=shape,
                        offset=payload_start + item['data_offsets'][0], file_size=file_size)
                    self._tensor_count += 1
                row_start += rows
            if row_start != num_embeddings:
                raise ValueError('PLE row count does not match model configuration')
            self._unique_files = len({item.path for _, _, items in self._shards for item in items.values()})
            self._shard_ends = np.array([end for _,end,_ in self._shards],dtype=np.int64)
            self._row_cache = PackedRowCache(self.row_width*self.bits//32,
                                             self.row_width//self.group_size,row_cache_bytes)
            log.info('JANGH2: PLE layer %d: indexed-mmap, %d shards, %d x %d, '
                     'affine Q%d/group%d/%s; no resident table', layer_idx, num_shards,
                     num_embeddings, dims, self.bits, self.group_size, self.aux_dtype)
            log.info('JANGH2: PLE storage v4: %d tensors in %d physical files; '
                     'lazy read-only mmap cache, max_open_files=%d per reader; '
                     'no OS-limit changes', self._tensor_count, self._unique_files,max_open_files)
            if self._row_cache.capacity:
                log.info('JANGH2: packed PLE row cache: %.1f MiB payload budget, capacity=%d rows; '
                         'binary shard lookup; no resident table', row_cache_bytes/(1024*1024),
                         self._row_cache.capacity)
        except BaseException:
            self.close()
            raise

    def mapping_stats(self) -> dict[str, int | bool]:
        """Diagnostics without exporting live buffers or opening any files."""
        return {**self._mapping_cache.stats(), 'tensor_count': self._tensor_count,
                'unique_files': self._unique_files, 'disk_rows_read': self.disk_rows_read,
                **(self._row_cache.stats() if self._row_cache is not None else {}),
                **(self._device_cache.stats() if self._device_cache is not None else {})}

    def _ids(self, ids: Any) -> np.ndarray:
        ids = np.asarray(ids)
        if not np.issubdtype(ids.dtype, np.integer):
            raise TypeError('PLE row indices must be integers')
        if ids.size and (np.any(ids < 0) or np.any(ids >= self.row_count)):
            raise IndexError('PLE row index is outside the embedding vocabulary')
        return ids.astype(np.int64, copy=False)

    def _aux_float32(self, value: np.ndarray) -> np.ndarray:
        if self.aux_dtype == 'BF16':
            return (value.astype(np.uint32) << 16).view(np.float32)
        return value.astype(np.float32)

    def _empty_rows(self, count):
        return (np.empty((count,self.row_width*self.bits//32),np.uint32),
                np.empty((count,self.row_width//self.group_size),np.float32),
                np.empty((count,self.row_width//self.group_size),np.float32))

    def _read_rows_uncached(self, ids):
        packed,scales,biases = self._empty_rows(ids.size)
        # Binary search once, then visit only shards actually requested. The
        # former implementation scanned all 128 shards for every decode token.
        shard_ids = np.searchsorted(self._shard_ends,ids,side='right')
        for shard_index in np.unique(shard_ids):
            start,end,arrays = self._shards[int(shard_index)]
            positions = np.flatnonzero(shard_ids == shard_index)
            local = ids[positions]-start
            packed[positions] = self._mapping_cache.read_rows(arrays['weight'],local)
            scales[positions] = self._aux_float32(self._mapping_cache.read_rows(arrays['scales'],local))
            biases[positions] = self._aux_float32(self._mapping_cache.read_rows(arrays['biases'],local))
        self.disk_rows_read += ids.size
        return packed,scales,biases

    def _read_unique_rows(self, ids):
        """Read already-unique validated IDs while the reader lock is held."""
        if self._row_cache.capacity == 0:
            return self._read_rows_uncached(ids)
        rows = self._empty_rows(ids.size)
        hit_positions, hits, miss_positions = self._row_cache.get_many(ids)
        if hit_positions.size:
            for target, value in zip(rows, hits):
                target[hit_positions] = value
        if miss_positions.size:
            missing = self._read_rows_uncached(ids[miss_positions])
            for target, value in zip(rows, missing):
                target[miss_positions] = value
            for j, pos in enumerate(miss_positions):
                self._row_cache.put(int(ids[pos]), *(value[j] for value in missing))
        return rows

    def read_rows(self, row_ids: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Independent exact packed-row copies, preserving caller ordering."""
        with self._lock:
            if self._closed:
                raise RuntimeError('PLE storage has been closed')
            ids = self._ids(row_ids).reshape(-1)
            if self._row_cache.capacity == 0:
                result = self._read_rows_uncached(ids)
            else:
                unique, inverse = np.unique(ids, return_inverse=True)
                rows = self._read_unique_rows(unique)
                result = tuple(value[inverse] for value in rows)
            self.rows_read += ids.size
            return result

    def __call__(self, indices):
        if not self._device_cache_bytes:
            return self._lookup_cpu(indices)
        import mlx.core as mx
        with self._lock:
            if self._closed:
                raise RuntimeError('PLE storage has been closed')
            if self._device_cache is None:
                from .device_rows import DeviceRowCache
                dtype={'F16':mx.float16,'BF16':mx.bfloat16,'F32':mx.float32}[self.aux_dtype]
                self._device_cache=DeviceRowCache(mx,self.row_width,dtype,self.row_count,
                                                  byte_limit=self._device_cache_bytes)
            return self._device_cache.lookup(indices,self._lookup_cpu)

    def _lookup_cpu(self, indices):
        import mlx.core as mx
        with self._lock:
            if self._closed:
                raise RuntimeError('PLE storage has been closed')
            ids = self._ids(indices)
            dtype = {'F16': mx.float16, 'BF16': mx.bfloat16, 'F32': mx.float32}[self.aux_dtype]
            if ids.size == 0:
                return mx.zeros((*ids.shape, self.row_width), dtype=dtype)
            unique, inverse = np.unique(ids.reshape(-1), return_inverse=True)
            packed, scales, biases = self._read_unique_rows(unique)
            self.rows_read += unique.size
        values = mx.dequantize(mx.array(packed, dtype=mx.uint32),
                               mx.array(scales).astype(dtype), mx.array(biases).astype(dtype),
                               group_size=self.group_size, bits=self.bits, mode='affine')
        values = values[mx.array(inverse, dtype=mx.int64)]
        return values.reshape(*ids.shape, self.row_width)

    lookup = __call__

    def close(self):
        with self._lock:
            self._closed = True
            self._shards.clear()
            if self._row_cache is not None:
                self._row_cache.close()
            if self._device_cache is not None:
                self._device_cache.close()
            self._mapping_cache.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


@dataclass
class PLEState:
    kind: str
    readers: list[IndexedMMapEmbedding] = field(default_factory=list)
    created_layers: list[int] = field(default_factory=list)
    expected_layers: list[int] = field(default_factory=list)

    def validate_complete(self):
        if self.kind == 'stock-indexed-mmap' and sorted(self.created_layers) != sorted(self.expected_layers):
            raise RuntimeError(f'Not all PLE tables were mapped: '
                               f'created={self.created_layers}, expected={self.expected_layers}')

    def close(self):
        for reader in self.readers:
            reader.close()


@contextmanager
def stock_ple_context(language, checkpoint: Checkpoint, *,
                      max_open_files: int = 8, row_cache_bytes: int = 0,
                      device_cache_bytes: int = 0):
    """Intercept only the intended stock PLE constructors, before allocation.

    The original n-gram hashing, prime vocabulary layout and forward function
    are retained. The resident embedding constructor is NEVER used for this
    context. Other threads retain their original constructor behaviour.
    """
    ngram = getattr(language, 'Qwen4ExpNGramEmbedding', None)
    original_sharded = getattr(language, 'ShardedEmbedding', None)
    if ngram is None or original_sharded is None:
        raise RuntimeError('Unrecognised Qwen4 PLE API: no vendor runtime hook and '
                           'no supported stock Qwen4ExpNGramEmbedding/ShardedEmbedding')
    original_init = ngram.__init__
    init_signature = inspect.signature(original_init)
    sharded_signature = inspect.signature(original_sharded)
    required = {'self', 'config', 'embedding_dim', 'layer_idx', 'ple_layer_index'}
    if (set(init_signature.parameters) != required
            or set(sharded_signature.parameters) != {'num_embeddings', 'dims', 'num_shards'}
            or getattr(original_init, '__globals__', {}).get('ShardedEmbedding') is not original_sharded
            or getattr(original_init, '_jangh2_stock_ple', False)):
        raise RuntimeError('Unrecognised stock PLE constructor contract; refusing resident allocation')
    text_config = checkpoint.config.get('text_config', {})
    configured_ids = text_config.get('ple_layer_ids', [])
    # Qwen4ExpDecoderLayer selects PLE using layer_idx + 1. Its constructor
    # layer_idx and checkpoint tensor paths are both already zero-based.
    expected = ple_layer_indices(configured_ids, text_config.get('num_hidden_layers'))
    state = PLEState('stock-indexed-mmap', expected_layers=list(expected))
    log.info('JANGH2: PLE layer mapping: config IDs (one-based)=%s -> '
             'tensor layers (zero-based)=%s; configured ordinals preserved',
             list(configured_ids), list(expected))
    active = ContextVar('jangh2_ple_active', default=False)
    layer = ContextVar('jangh2_ple_layer', default=None)

    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        if not active.get():
            return original_init(self, *args, **kwargs)
        bound = init_signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        config = bound.arguments['config']
        layer_idx = bound.arguments['layer_idx']
        if type(layer_idx) is not int or layer_idx not in state.expected_layers:
            raise ValueError(f'Unexpected zero-based PLE layer {layer_idx!r}; '
                             f'expected {state.expected_layers} from one-based IDs {configured_ids}')
        ordinal = bound.arguments['ple_layer_index']
        expected_ordinal = state.expected_layers.index(layer_idx)
        if type(ordinal) is not int or ordinal != expected_ordinal:
            raise ValueError(f'PLE ordinal mismatch for zero-based layer {layer_idx}: '
                             f'got {ordinal!r}, expected {expected_ordinal}; '
                             'refusing to change the n-gram hashing/table association')
        if getattr(config, 'ple_storage', None):
            raise ValueError('Existing external PLE manifest cannot be mixed with indexed-mmap bridge')
        if layer_idx in state.created_layers:
            raise RuntimeError(f'Duplicate PLE construction for layer {layer_idx}')
        token = layer.set(layer_idx)
        try:
            result = original_init(*bound.args, **bound.kwargs)
            if not isinstance(getattr(self, 'ngram_embedding', None), IndexedMMapEmbedding):
                raise RuntimeError('Stock PLE constructor did not install indexed-mmap storage')
            return result
        finally:
            layer.reset(token)

    def sharded(num_embeddings, dims, num_shards):
        if not active.get():
            return original_sharded(num_embeddings, dims, num_shards)
        layer_idx = layer.get()
        if layer_idx is None:
            raise RuntimeError('PLE construction outside a known n-gram layer; refusing resident allocation')
        reader = IndexedMMapEmbedding(checkpoint, layer_idx, num_embeddings, dims, num_shards,
                                      max_open_files=max_open_files,row_cache_bytes=row_cache_bytes,
                                      device_cache_bytes=device_cache_bytes)
        state.readers.append(reader)
        state.created_layers.append(layer_idx)
        return reader

    init._jangh2_stock_ple = True
    token = active.set(True)
    language.ShardedEmbedding = sharded
    ngram.__init__ = init
    try:
        yield state
    except BaseException:
        state.close()
        raise
    finally:
        ngram.__init__ = original_init
        language.ShardedEmbedding = original_sharded
        active.reset(token)
