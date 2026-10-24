"""Real MLX expert modules and MLX-VLM-aware installation.

The native option imports upstream jang_tools.jangh. The reference option is
implemented here and does not need a newer jang package or custom Metal build.
"""
from __future__ import annotations
import logging
from typing import Any, Mapping

import mlx.core as mx
import mlx.nn as nn

from .metadata import (PROJECTIONS, canonical, check_tensor_pair, expert_specs,
                       normalize_quantization, validate_config, is_mtp)
from .reference_ops import expert_forward

log = logging.getLogger('jangh2_bridge')

class ReferenceSwitchLinear(nn.Module):
    def __init__(self, in_dims: int, out_dims: int, experts: int, spec: dict,
                 levels: tuple[float, ...]):
        super().__init__()
        self.input_dims, self.output_dims, self.num_experts = in_dims, out_dims, experts
        self.bits, self.rotation = int(spec['bits']), spec['rotation']
        self.tq2_packed = mx.zeros((experts,out_dims,in_dims*self.bits//32),dtype=mx.uint32)
        self.tq2_scales = mx.zeros((experts,out_dims),dtype=mx.float16)
        self._levels = mx.array(levels,dtype=mx.float32)

    def to_quantized(self, *, mode='jangtq2', bits=None, rotation=None, group_size=None):
        # Accept only the matching format, never throw away a rotation field.
        if mode != 'jangtq2' or bits not in (None,self.bits) or rotation not in (None,self.rotation):
            raise ValueError('JANGH2 module metadata does not match its checkpoint')
        return self

    def spec(self):
        return self.tq2_packed,self.tq2_scales,self.bits,self._levels,self.rotation

class ReferenceSwitchGLU(nn.Module):
    def __init__(self, d: int, hidden: int, experts: int, specs: Mapping[str,dict],
                 levels: Mapping[int,tuple[float,...]], *, chunk_size: int=1, fast: bool=False):
        super().__init__()
        self.chunk_size = chunk_size
        self.fast = fast
        self.is_jangtq2 = True
        for name in PROJECTIONS:
            n,k = (d,hidden) if name == 'down_proj' else (hidden,d)
            setattr(self,name,ReferenceSwitchLinear(k,n,experts,specs[name],levels[specs[name]['bits']]))

    def __call__(self,x,indices):
        return expert_forward(x,indices,{p:getattr(self,p).spec() for p in PROJECTIONS},mx,
                              chunk_size=self.chunk_size,
                              evaluation='per_chunk' if self.fast else 'legacy')


class MetalSwitchGLU(nn.Module):
    """Opt-in direct packed-weight Metal execution; same checkpoint parameters."""
    def __init__(self, d: int, hidden: int, experts: int, specs: Mapping[str,dict],
                 levels: Mapping[int,tuple[float,...]]):
        super().__init__()
        from .metal_ops import KernelSpec
        from .performance import PerformanceOptions
        options=PerformanceOptions.from_environment(fast_defaults=True)
        self._kernel_spec=KernelSpec(d,hidden,experts,
            tuple(int(specs[p]['bits']) for p in PROJECTIONS),
            tuple(specs[p]['rotation'] for p in PROJECTIONS),
            tuple(tuple(levels[specs[p]['bits']]) for p in PROJECTIONS),
            options.gate_rows,options.down_rows,options.sort_threshold)
        self._performance=options
        self.is_jangtq2=True
        for name in PROJECTIONS:
            n,k=(d,hidden) if name=='down_proj' else (hidden,d)
            setattr(self,name,ReferenceSwitchLinear(k,n,experts,specs[name],levels[specs[name]['bits']]))

    def __call__(self,x,indices):
        from .metal_ops import forward
        projections={p:(getattr(self,p).tq2_packed,getattr(self,p).tq2_scales)
                     for p in PROJECTIONS}
        # Keep the proven single-token/decode path byte-identical in metal_ops.
        if (self._performance.prefill == 'grouped' and x.shape[-2] >= self._performance.prefill_min_tokens):
            from .grouped_ops import forward as grouped_forward
            # Grouped geometry is qualified/tuned separately from decode geometry.
            from dataclasses import replace
            grouped_spec=replace(self._kernel_spec,gate_rows=4,down_rows=4)
            return grouped_forward(x,indices,projections,grouped_spec,mx,
                                   block=self._performance.prefill_tile)
        return forward(x,indices,projections,self._kernel_spec,mx)


def known_switch_classes() -> tuple[type, ...]:
    from mlx_vlm.models.switch_layers import SwitchGLU as VLMSwitchGLU
    from mlx_lm.models.switch_layers import SwitchGLU as LMSwitchGLU
    return VLMSwitchGLU,LMSwitchGLU


def make_switch(d: int, hidden: int, experts: int, specs: Mapping[str,dict],
                levels: Mapping[int,tuple[float,...]], *, backend: str, chunk_size: int=1):
    if backend in ('reference', 'reference-fast'):
        return ReferenceSwitchGLU(d,hidden,experts,specs,levels,chunk_size=chunk_size,
                                  fast=backend=='reference-fast')
    if backend == 'metal':
        return MetalSwitchGLU(d,hidden,experts,specs,levels)
    if backend != 'native':
        raise ValueError(f'Unknown JANGH2 backend: {backend}')
    try:
        from jang_tools.jangh.switch import TQSwitchGLU
        from jang_tools.jangh.format import codebook
    except ImportError as exc:
        raise ImportError('Native backend needs jang_tools.jangh (not the legacy jang_tools.loader). '
                          'Use --backend reference without changing dependencies, or see README native setup.') from exc
    import numpy as np
    for bits, values in levels.items():
        if not np.allclose(codebook(bits),np.asarray(values,np.float32),rtol=0,atol=2e-6):
            raise ValueError(f'Installed native backend codebook differs for {bits} bits')
    g,u,dn = (specs[p] for p in PROJECTIONS)
    result = TQSwitchGLU(d,hidden,experts,g['bits'],dn['bits'],swiglu_limit=0.0,
                         rotation_gate_up=g['rotation'],rotation_down=dn['rotation'],bits_up=u['bits'])
    result.is_jangtq2 = True
    return result


def replace_experts(model: nn.Module, config: dict, *, backend: str,
                    chunk_size: int=1) -> list[str]:
    levels = validate_config(config)
    wanted = {p:s for p,s in expert_specs(normalize_quantization(config)).items() if not is_mtp(p)}
    text = config.get('text_config',config)
    if text.get('hidden_act','silu') not in ('silu','swiglu') or float(text.get('swiglu_limit',0) or 0) != 0:
        raise ValueError('This bridge currently supports unclipped SiLU-gated Qwen4 experts only')
    known = known_switch_classes()
    seen: set[str] = set()
    replacements = []
    for path,parent in list(model.named_modules()):
        switch = getattr(parent,'switch_mlp',None)
        if switch is None:
            continue
        key = canonical(path+'.switch_mlp')
        if is_mtp(key):
            continue
        if key not in wanted:
            if isinstance(switch,known):
                raise ValueError(f'Routed SwitchGLU has no JANGH2 spec: {key}')
            continue
        if not isinstance(switch,known):
            raise TypeError(f'Unexpected expert module at {key}: {type(switch)}')
        activation = getattr(switch,'activation',None)
        if activation is not None and type(activation).__name__ != 'SwiGLU' and getattr(activation,'__name__',None) != 'silu':
            raise ValueError(f'Unsupported activation at {key}: {type(activation)}')
        shapes = {}
        for projection in PROJECTIONS:
            layer = getattr(switch,projection)
            if getattr(layer,'bias',None) is not None:
                raise ValueError(f'Expert bias is unsupported: {key}.{projection}')
            shapes[projection] = tuple(layer.weight.shape)
        e,h,d = shapes['gate_proj']
        if shapes['up_proj'] != (e,h,d) or shapes['down_proj'] != (e,d,h):
            raise ValueError(f'Unexpected SwiGLU dimensions at {key}: {shapes}')
        replacement = make_switch(d,h,e,wanted[key],levels,backend=backend,chunk_size=chunk_size)
        replacements.append((parent,replacement,key))
        seen.add(key)
    if seen != set(wanted):
        raise ValueError(f'JANGH2 specs without matching modules: {sorted(set(wanted)-seen)[:8]}')
    # Validate the complete replacement plan before changing the architecture.
    for parent,replacement,key in replacements:
        parent.switch_mlp = replacement
    log.info('JANGH2: installed %d routed expert blocks (%s backend)',len(seen),backend)
    return sorted(seen)


def validate_loaded_experts(model: nn.Module, weights: Mapping[str,Any]) -> None:
    for path,module in model.named_modules():
        if not getattr(module,'is_jangtq2',False):
            continue
        for projection in PROJECTIONS:
            layer = getattr(module,projection)
            prefix = path+'.'+projection
            keys = [prefix+'.tq2_packed',prefix+'.tq2_scales']
            if any(key not in weights for key in keys):
                raise ValueError(f'Packed weights absent at runtime path {prefix}')
            check_tensor_pair(weights[keys[0]],weights[keys[1]],layer.num_experts,
                              layer.output_dims,layer.input_dims,layer.bits)


def quantize_affine(model: nn.Module, config: dict, weights: Mapping[str,Any]) -> None:
    q = normalize_quantization(config)
    used: set[str] = set()
    def predicate(path,module):
        spec = q.get(canonical(path))
        if isinstance(spec,dict) and spec.get('mode') == 'jangtq2':
            if not hasattr(module,'tq2_packed') or not hasattr(module,'tq2_scales'):
                raise ValueError(f'JANGH2 projection was not replaced before quantization: {path}')
            module.to_quantized(**spec)  # validates bits/rotation; returns itself
            return False
        if path+'.scales' not in weights:
            return False
        if not hasattr(module,'to_quantized'):
            raise ValueError(f'Affine scales found for a non-quantizable module: {path}')
        values = {'bits':q['bits'],'group_size':q['group_size'],'mode':q.get('mode','affine')}
        if isinstance(spec,dict):
            values.update(spec)
        if values['mode'] != 'affine' or set(values) != {'bits','group_size','mode'}:
            raise ValueError(f'Unsupported non-expert quantization at {path}: {values}')
        used.add(path)
        return values
    nn.quantize(model,group_size=q['group_size'],bits=q['bits'],mode=q.get('mode','affine'),class_predicate=predicate)
    expected = {key[:-len('.scales')] for key in weights if key.endswith('.scales')}
    if expected != used:
        raise ValueError(f'Affine weights not covered by model quantization: {sorted(expected-used)[:12]}')
    log.info('JANGH2: retained per-module affine quantization for %d modules',len(used))
