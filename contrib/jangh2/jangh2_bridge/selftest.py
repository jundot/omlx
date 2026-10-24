"""Run on the user's Mac: real MLX execution, no downloaded model required."""
from __future__ import annotations
import math
import numpy as np
from .metadata import CUBIC, PROJECTIONS
from .reference_ops import pack_codes


def test_config(bits=2):
    alpha,beta=CUBIC[bits]
    u=np.arange(1<<bits,dtype=np.float32)-((1<<bits)-1)/2
    levels=(u*(alpha+beta*u*u)).astype(np.float32).tolist()
    return {'model_type':'qwen4_exp','text_config':{'hidden_act':'silu'},
            'jangtq':{'version':2,'packing':'lsb-bitstream','scale_dtype':'float16',
                      'rotation':'hadamard32','codebook_family':'odd-cubic',
                      'codebooks':{str(bits):{'levels':levels}}},
            'quantization':{'bits':8,'group_size':64,**{
                'model.layers.0.mlp.switch_mlp.'+p:{'mode':'jangtq2','bits':bits,'rotation':'hadamard32'}
                for p in PROJECTIONS}}}


def _dense_reference(x,idx,matrices):
    # Deliberately independent of reference_ops.hadamard32/expert_forward.
    h=np.ones((1,1),np.float32)
    for _ in range(5): h=np.block([[h,h],[h,-h]])
    h=h/np.sqrt(32.)
    def rotate(a): return (a.reshape(-1,32)@h.T).reshape(a.shape)
    flat=x.reshape(-1,x.shape[-1]); ids=idx.reshape(-1,idx.shape[-1])
    result=np.empty((*ids.shape,x.shape[-1]),np.float32)
    for row in range(len(flat)):
        for route,expert in enumerate(ids[row]):
            rotated=rotate(flat[row])
            g=matrices['gate_proj'][expert]@rotated
            u=matrices['up_proj'][expert]@rotated
            # x scale in the fixture makes overflow impossible.
            hidden=(g/(1+np.exp(-g)))*u
            result[row,route]=matrices['down_proj'][expert]@rotate(hidden)
    return result.reshape(*idx.shape,x.shape[-1])


def run(backend='reference', *, full_dimensions=True):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_vlm.models.switch_layers import SwitchGLU
    from .experts import replace_experts,quantize_affine,validate_loaded_experts
    from .loader import compare_parameters
    from .metadata import canonical_map
    if not mx.metal.is_available():
        raise RuntimeError('This test needs the Apple Metal backend; Linux/CPU is not a substitute')
    mx.set_default_device(mx.gpu)
    class Node(nn.Module):
        pass
    def make_model(d,i,e):
        model=Node(); model.language_model=Node(); model.language_model.model=Node()
        layer=Node(); layer.mlp=Node(); layer.mlp.switch_mlp=SwitchGLU(d,i,e)
        model.language_model.model.layers=[layer]
        return model
    rng=np.random.default_rng(7643)
    scenarios=([(1024,512,2),(1024,512,3),(1024,512,4)] if backend=='native'
               else [(64,32,2),(96,64,3),(128,64,4)])
    if full_dimensions:
        scenarios.append((2560,640,2))
    cases=0
    for d,hidden,bits in scenarios:
        e=4; config=test_config(bits); model=make_model(d,hidden,e)
        levels=np.array(config['jangtq']['codebooks'][str(bits)]['levels'],np.float32)
        replace_experts(model,config,backend=backend,chunk_size=8 if backend=='reference-fast' else 1)
        weights={}; dense={}
        for p,n,k in [('gate_proj',hidden,d),('up_proj',hidden,d),('down_proj',d,hidden)]:
            q=rng.integers(0,1<<bits,(e,n,k),dtype=np.uint8)
            scales=rng.uniform(.02,.06,(e,n)).astype(np.float16)
            dense[p]=levels[q]*scales.astype(np.float32)[...,None]
            prefix='language_model.layers.0.mlp.switch_mlp.'+p
            weights[prefix+'.tq2_packed']=mx.array(pack_codes(q,bits),dtype=mx.uint32)
            weights[prefix+'.tq2_scales']=mx.array(scales,dtype=mx.float16)
        weights=canonical_map(weights)
        validate_loaded_experts(model,weights)
        quantize_affine(model,config,weights)
        compare_parameters(dict(tree_flatten(model.parameters())),weights)
        model.load_weights(list(weights.items()),strict=True); model.eval()
        mx.eval(model.parameters())
        switch=model.language_model.model.layers[0].mlp.switch_mlp
        for tokens in (1,33):  # 2 routes decode; 66 routes exercise native sorted prefill
            for dtype in (mx.float32,mx.bfloat16):
                x=mx.array(rng.normal(0,.2,(1,tokens,d)).astype(np.float32),dtype=dtype)
                ids_np=rng.integers(0,e,(1,tokens,2),dtype=np.uint32)
                ids=mx.array(ids_np,dtype=mx.uint32)
                expected=_dense_reference(np.array(x.astype(mx.float32)),ids_np,dense)
                actual=switch(x,ids)
                mx.eval(actual)
                actual_np=np.array(actual.astype(mx.float32))
                if actual_np.shape != expected.shape or not np.isfinite(actual_np).all():
                    raise AssertionError(f'Bad shape or nonfinite output: {actual_np.shape}')
                difference=actual_np-expected
                relative=float(np.linalg.norm(difference)/(np.linalg.norm(expected)+1e-12))
                peak=float(np.max(np.abs(difference)))
                limit=0.025 if dtype==mx.bfloat16 or backend=='native' else 0.0001
                if relative>limit or peak > .04*float(np.max(np.abs(expected)))+1e-4:
                    raise AssertionError(f'{backend} mismatch D={d},I={hidden},bits={bits},T={tokens},dtype={dtype}: '
                                         f'relative L2={relative:.6g}, peak={peak:.6g}')
                print(f'PASS {backend}: D={d}, I={hidden}, bits={bits}, tokens={tokens}, '
                      f'dtype={dtype}, relative_L2={relative:.3g}, max_abs={peak:.3g}',flush=True)
                cases+=1
        del model,switch,weights,dense
        mx.synchronize(); mx.clear_cache()
    print(f'MLX self-test passed: {cases} numerical cases; strict loading and VLM expert replacement passed.',flush=True)
    return cases
