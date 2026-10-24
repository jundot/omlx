"""Read-only safetensors/index preflight. No model-sized arrays are allocated."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any

from .metadata import (canonical, canonical_map, expert_specs, is_mtp,
                       is_ple_storage, normalize_quantization, safe_shard_path,
                       validate_config)

@dataclass(frozen=True)
class TensorInfo:
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    filename: str
    raw_key: str


def read_header(path: Path) -> dict[str, dict]:
    size = path.stat().st_size
    with path.open('rb') as f:
        start = f.read(8)
        if len(start) != 8:
            raise ValueError(f'Truncated safetensors file: {path}')
        length = struct.unpack('<Q', start)[0]
        if length < 2 or length > min(64 * 1024**2, size - 8):
            raise ValueError(f'Invalid safetensors header length in {path}: {length}')
        header = json.loads(f.read(length))
    if not isinstance(header, dict):
        raise ValueError(f'Invalid safetensors header in {path}')
    payload_size = size - 8 - length
    for key, item in header.items():
        if key == '__metadata__':
            continue
        if not isinstance(item, dict):
            raise ValueError(f'Invalid tensor descriptor: {path.name}:{key}')
        shape = item.get('shape')
        offsets = item.get('data_offsets')
        if (not isinstance(shape, list) or any(type(x) is not int or x < 0 for x in shape)
            or not isinstance(offsets, list) or len(offsets) != 2
            or any(type(x) is not int for x in offsets)
            or not 0 <= offsets[0] <= offsets[1] <= payload_size):
            raise ValueError(f'Invalid tensor shape/offsets: {path.name}:{key}')
        widths = {'U32':4,'F16':2,'BF16':2,'F32':4,'F64':8,'I32':4,'I64':8,'U8':1,'I8':1,'BOOL':1}
        dtype = item.get('dtype')
        if dtype in widths and math.prod(shape) * widths[dtype] != offsets[1] - offsets[0]:
            raise ValueError(f'Tensor byte count disagrees with shape/dtype: {path.name}:{key}')
    return header


@dataclass
class Checkpoint:
    root: Path
    config: dict[str, Any]
    weight_map: dict[str, str]
    tensors: dict[str, TensorInfo]
    config_sha256: str

    def report(self) -> dict[str, Any]:
        resident = sum(v.nbytes for k,v in self.tensors.items() if not is_ple_storage(k) and not is_mtp(k))
        ple = sum(v.nbytes for k,v in self.tensors.items() if is_ple_storage(k))
        mtp = sum(v.nbytes for k,v in self.tensors.items() if is_mtp(k))
        return {'model':str(self.root), 'config_sha256':self.config_sha256,
                'indexed_tensors':len(self.tensors), 'shard_files':len(set(self.weight_map.values())),
                'expert_blocks':len(expert_specs(normalize_quantization(self.config))),
                'resident_weight_bytes':resident, 'ssd_ple_weight_bytes':ple, 'disabled_mtp_weight_bytes':mtp,
                'note':'Weight storage only; excludes KV cache, activations, temporary arrays and allocator overhead.'}


def inspect_checkpoint(root: str | Path) -> Checkpoint:
    root = Path(root).expanduser().resolve()
    config_data = (root/'config.json').read_bytes()
    config = json.loads(config_data)
    validate_config(config)
    index = json.loads((root/'model.safetensors.index.json').read_text())
    weight_map = index.get('weight_map')
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError('Missing or empty model.safetensors.index.json weight_map')
    raw_tensors: dict[str, TensorInfo] = {}
    by_file: dict[str, list[str]] = {}
    for key, filename in weight_map.items():
        if not isinstance(key,str) or not isinstance(filename,str):
            raise ValueError('Invalid index key or filename')
        safe_shard_path(root, filename)
        by_file.setdefault(filename, []).append(key)
    for filename, keys in by_file.items():
        header = read_header(safe_shard_path(root, filename))
        for key in keys:
            if key not in header:
                raise ValueError(f'Index references missing tensor {key!r} in {filename!r}')
            item = header[key]
            begin,end = item['data_offsets']
            raw_tensors[key] = TensorInfo(tuple(item['shape']),item['dtype'],end-begin,filename,key)
    tensors = canonical_map(raw_tensors)
    text = config.get('text_config', config)
    e,d,i = (int(text.get(k,0)) for k in ('num_experts','hidden_size','moe_intermediate_size'))
    if min(e,d,i) < 1 or d % 32 or i % 32:
        raise ValueError(f'Unsupported expert dimensions: E={e}, D={d}, I={i}')
    for parent, group in expert_specs(normalize_quantization(config)).items():
        if is_mtp(parent):
            continue
        for projection, spec in group.items():
            n,k = (d,i) if projection == 'down_proj' else (i,d)
            p,s = parent+'.'+projection+'.tq2_packed', parent+'.'+projection+'.tq2_scales'
            if p not in tensors or s not in tensors:
                raise ValueError(f'Missing packed expert tensor pair: {parent}.{projection}')
            if tensors[p].shape != (e,n,k*spec['bits']//32) or tensors[p].dtype != 'U32':
                raise ValueError(f'Wrong packed expert descriptor: {p}: {tensors[p]}')
            if tensors[s].shape != (e,n) or tensors[s].dtype != 'F16':
                raise ValueError(f'Wrong per-row scale descriptor: {s}: {tensors[s]}')
    return Checkpoint(root,config,weight_map,tensors,hashlib.sha256(config_data).hexdigest())
