"""Small synthetic or selected real-checkpoint expert fixtures for local tests."""
from __future__ import annotations
import struct
import numpy as np
from .metadata import CUBIC, PROJECTIONS


def synthetic(*, d=2560, hidden=640, experts=8, bits=(2,2,2),
              rotations=('hadamard32',)*3, seed=1211):
    rng=np.random.default_rng(seed)
    levels={}
    for b in set(bits):
        a,c=CUBIC[b];u=np.arange(1<<b,dtype=np.float64)-((1<<b)-1)/2
        levels[b]=tuple((u*(a+c*u*u)).astype(np.float32).tolist())
    specs={p:{'bits':b,'mode':'jangtq2','rotation':r}
           for p,b,r in zip(PROJECTIONS,bits,rotations)}
    weights={}
    for p,n,k in [('gate_proj',hidden,d),('up_proj',hidden,d),('down_proj',d,hidden)]:
        b=specs[p]['bits']
        # Any uint32 bit pattern is a valid 2/3/4-bit packed row when K % 32 == 0.
        packed=rng.integers(0,2**32,(experts,n,k*b//32),dtype=np.uint32)
        scales=rng.uniform(.008,.025,(experts,n)).astype(np.float16)
        weights[p]=(packed,scales)
    return {'d':d,'hidden':hidden,'experts':experts,'specs':specs,'levels':levels,
            'weights':weights,'source':'synthetic','seed':seed}


def from_checkpoint(model_path, *, layer=0, experts=8):
    """Read only selected experts from one block; never load the whole model."""
    from .checkpoint import inspect_checkpoint,read_header
    from .metadata import expert_specs,normalize_quantization,validate_config,safe_shard_path
    from .ple import _TensorSlice,_FileMappingCache
    cp=inspect_checkpoint(model_path)
    groups=expert_specs(normalize_quantization(cp.config))
    keys=[k for k in groups if k.endswith(f'layers.{layer}.mlp.switch_mlp')]
    if len(keys)!=1:
        raise ValueError(f'No unique JANGH2 expert triplet for layer {layer}: {keys}')
    prefix=keys[0];text=cp.config['text_config'];e=int(text['num_experts'])
    if not 1<=experts<=min(2048,e):
        raise ValueError(f'experts must be 1..{min(2048,e)}')
    d=int(text['hidden_size']);h=int(text['moe_intermediate_size'])
    selected=np.linspace(0,e-1,experts,dtype=np.int64)
    cache=_FileMappingCache(max_open_files=4);headers={};weights={}
    try:
        for p in PROJECTIONS:
            pair=[]
            for suffix,dt in [('tq2_packed','<u4'),('tq2_scales','<f2')]:
                info=cp.tensors[prefix+'.'+p+'.'+suffix]
                path=safe_shard_path(cp.root,info.filename)
                if path not in headers:
                    header=read_header(path)
                    with path.open('rb') as f: start=8+struct.unpack('<Q',f.read(8))[0]
                    headers[path]=(header,start,path.stat().st_size)
                header,start,size=headers[path];item=header[info.raw_key]
                view=_TensorSlice(path,np.dtype(dt),tuple(item['shape']),start+item['data_offsets'][0],size)
                pair.append(cache.read_rows(view,selected))
            weights[p]=tuple(pair)
    finally:
        cache.close()
    return {'d':d,'hidden':h,'experts':experts,'specs':groups[prefix],
            'levels':validate_config(cp.config),'weights':weights,'source':'checkpoint',
            'model':str(cp.root),'layer':layer,'selected_experts':selected.tolist(),
            'config_sha256':cp.config_sha256}


def mlx_models(fixture, backends, *, tuning=None):
    import mlx.core as mx
    from dataclasses import replace
    from .experts import make_switch
    weights={p:(mx.array(v[0],dtype=mx.uint32),mx.array(v[1],dtype=mx.float16))
             for p,v in fixture['weights'].items()}
    models={}
    for label,backend in backends.items():
        model=make_switch(fixture['d'],fixture['hidden'],fixture['experts'],
                          fixture['specs'],fixture['levels'],backend=backend,
                          chunk_size=8 if backend=='reference-fast' else 1)
        if backend=='metal':
            # Benchmarks must not silently inherit an exported serving profile.
            from .performance import PerformanceOptions
            model._kernel_spec=replace(model._kernel_spec,gate_rows=4,down_rows=4,sort_threshold=64)
            model._performance=PerformanceOptions(ple_cache_mib=16,ple_open_files=16)
        if backend=='metal' and tuning and label in tuning:
            kernel_keys={'gate_rows','down_rows','sort_threshold'}
            model._kernel_spec=replace(model._kernel_spec,**{k:v for k,v in tuning[label].items() if k in kernel_keys})
            option_keys={'prefill','prefill_tile','prefill_min_tokens'}
            model._performance=replace(model._performance,**{k:v for k,v in tuning[label].items() if k in option_keys})
        entries=[]
        for p,pair in weights.items():
            entries.extend([(p+'.tq2_packed',pair[0]),(p+'.tq2_scales',pair[1])])
        model.load_weights(entries,strict=True);model.eval()
        models[label]=model
    mx.eval(*[array for pair in weights.values() for array in pair])
    return models


def dense_oracle(x,indices,fixture):
    """Independent row extraction / dense matmul reference, for small fixtures."""
    h32=np.ones((1,1),np.float32)
    for _ in range(5):h32=np.block([[h32,h32],[h32,-h32]])
    h32/=np.sqrt(32.)
    def rotate(row,projection):
        return ((row.reshape(-1,32)@h32.T).reshape(row.shape)
                if fixture['specs'][projection]['rotation']=='hadamard32' else row)
    matrices={}
    for p,(packed,scales) in fixture['weights'].items():
        bits=fixture['specs'][p]['bits'];width=fixture['hidden'] if p=='down_proj' else fixture['d']
        # Read the literal bitstream one input column at a time. This does not
        # call the production unpacker or Metal helpers.
        codes=[]
        for col in range(width):
            word,shift=divmod(col*bits,32)
            code=packed[...,word]>>shift
            if shift+bits>32:code=code|(packed[...,word+1]<<(32-shift))
            codes.append(code & ((1<<bits)-1))
        codes=np.stack(codes,axis=-1)
        matrices[p]=np.asarray(fixture['levels'][bits],np.float32)[codes]*scales.astype(np.float32)[...,None]
    xf=x.reshape(-1,x.shape[-1]);ids=indices.reshape(-1,indices.shape[-1])
    output=np.empty((*ids.shape,fixture['d']),np.float32)
    for t,row in enumerate(xf):
        for j,e in enumerate(ids[t]):
            g=matrices['gate_proj'][e]@rotate(row,'gate_proj')
            u=matrices['up_proj'][e]@rotate(row,'up_proj')
            z=np.exp(-np.abs(g));sig=np.where(g>=0,1/(1+z),z/(1+z))
            hidden=g*sig*u
            output[t,j]=matrices['down_proj'][e]@rotate(hidden,'down_proj')
    return output.reshape(*indices.shape,fixture['d'])


def compare(actual,expected,*,rtol=1e-4,peak_rtol=.01,atol=2e-6):
    actual=np.asarray(actual,dtype=np.float32);expected=np.asarray(expected,dtype=np.float32)
    if actual.shape!=expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError('Non-finite output or wrong output shape')
    delta=actual-expected
    relative=float(np.linalg.norm(delta.ravel())/(np.linalg.norm(expected.ravel())+1e-20))
    peak=float(np.max(np.abs(delta))) if delta.size else 0.
    allowed=peak_rtol*(float(np.max(np.abs(expected))) if expected.size else 0)+atol
    if relative>rtol or peak>allowed:
        raise AssertionError(f'Numerical disagreement: relative L2={relative:.6g} (limit {rtol}), max={peak:.6g} (limit {allowed:.6g})')
    return {'relative_l2':relative,'max_abs':peak,'relative_l2_limit':rtol,'max_abs_limit':allowed}
