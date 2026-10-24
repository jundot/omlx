"""Opt-in expert-grouped prompt kernels. No host routing reads or dense weights.

A GPU sort and per-expert boundaries group 2/4 routed tokens. Each SIMD lane
unpacks a weight once and reuses it for every token in the tile. Decode continues
through the unchanged baseline kernels. Compile/qualify on Apple Metal before enabling.
"""
from __future__ import annotations
from dataclasses import dataclass
import functools
import math
from .metal_ops import (KernelSpec, KernelDefinition, _definition, _READ_HEADER,
                        _table_function, _kernel_for, PROJECTIONS, route_plan)

@dataclass
class Schedule:
    order: object
    ids: object
    starts: object
    tile_ends: object
    inverse: object
    max_tiles: int


def schedule(flat_ids, experts: int, block: int, xp) -> Schedule:
    if type(experts) is not int or experts<1 or block not in (2,4):
        raise ValueError('Positive expert count and a 2/4-token tile are required')
    dtype=str(flat_ids.dtype).split('.')[-1]
    if dtype not in ('int32','uint32','int64','uint64'):
        raise TypeError('Routes must use integer IDs')
    # Invalid IDs share an extra sentinel bucket. No truncation of large uint64.
    ids=xp.where((flat_ids>=0)&(flat_ids<experts),flat_ids,experts).reshape(-1).astype(xp.uint32)
    n=ids.size
    order=xp.argsort(ids); ids=ids[order]
    starts=xp.searchsorted(ids,xp.arange(experts+2,dtype=xp.uint32)).astype(xp.uint32)
    counts=starts[1:]-starts[:-1]
    tile_ends=xp.cumsum((counts+block-1)//block).astype(xp.uint32)
    # sum(ceil(count/B)) <= ceil(R/B) + nonempty_buckets - 1, and <= R.
    max_tiles=min(n,math.ceil(n/block)+experts) if n else 0
    return Schedule(order,ids,starts,tile_ends,xp.argsort(order),max_tiles)


def tile_ranges(s: Schedule, block: int):
    """CPU-only oracle/debug helper; never called in the inference path."""
    for expert in range(len(s.starts)-1):
        for start in range(int(s.starts[expert]),int(s.starts[expert+1]),block):
            yield expert,start,min(start+block,int(s.starts[expert+1]))


def _preamble(spec,block):
    return f'''
    const uint lane = thread_position_in_threadgroup.x & 31u;
    const uint sg = thread_position_in_threadgroup.x >> 5;
    const uint tile_id = threadgroup_position_in_grid.y;
    // Uniform for the entire threadgroup: safe before any shared-memory barrier.
    if (tile_id >= tile_ends[{spec.experts}u]) return;
    uint lo = 0u, hi = {spec.experts+1}u;
    while (lo < hi) {{
        uint mid = lo + (hi-lo)/2u;
        if (tile_ends[mid] <= tile_id) lo = mid+1u; else hi = mid;
    }}
    const uint expert = lo;
    const uint prior_tiles = expert ? tile_ends[expert-1u] : 0u;
    const uint first_route = starts[expert] + (tile_id-prior_tiles)*{block}u;
    const uint route_end = starts[expert+1u];
    const bool invalid = expert >= {spec.experts}u;
    const uint safe_expert = invalid ? 0u : expert;
    const float NAN_VALUE = as_type<float>(0x7fc00000u);
'''

@functools.lru_cache(maxsize=256)
def build_grouped_gate(spec: KernelSpec, top_k: int, block: int=4) -> KernelDefinition:
    if block not in (2,4) or type(top_k) is not int or top_k<1:
        raise ValueError('Invalid grouped gate tile/top-k')
    d,h=spec.input_dims,spec.hidden_dims; bg,bu,_=spec.bits;rps=spec.gate_rows
    rotated=spec.rotations[2]=='hadamard32';groups=32//rps if rotated else 2;tile=groups*rps
    store=(f'if (lane==0u) hidden_block[b*32u + sg*{rps}u + r] = value;' if rotated else
           f'if (lane==0u && route<route_end && row<{h}u) out[route*{h}u+row]=value;')
    epilogue=''
    if rotated:
        epilogue=f'''
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg==0u) {{
        for (uint b=0u;b<{block}u;++b) {{
            float value=hidden_block[b*32u+lane];
            for (ushort step=1;step<32;step<<=1) {{
                float other=simd_shuffle_xor(value,step);
                value=(lane & step) ? other-value : value+other;
            }}
            uint route=first_route+b, row=tile_base+lane;
            if (route<route_end && row<{h}u)
                out[route*{h}u+row]=value*0.1767766952966369f;
        }}
    }}
'''
    source=_preamble(spec,block)+f'''
    const uint tile_base=threadgroup_position_in_grid.x*{tile}u;
    {'threadgroup float hidden_block['+str(block*32)+'];' if rotated else ''}
    float ag[{block}][{rps}]={{{{0.0f}}}}, au[{block}][{rps}]={{{{0.0f}}}};
    for (uint input_block=0u; input_block<{d}u; input_block+=128u) {{
        uint col=input_block+lane*4u;
        for (uint r=0u;r<{rps}u;++r) {{
            uint row=tile_base+sg*{rps}u+r;
            if (row<{h}u && col<{d}u) {{
                const device uint* pg=wg+(ulong(safe_expert)*{h}u+row)*{d*bg//32}u;
                const device uint* pu=wu+(ulong(safe_expert)*{h}u+row)*{d*bu//32}u;
                for (uint j=0u;j<4u;++j) {{
                    if (col+j<{d}u) {{
                        // Decode once, reuse across all routed tokens in this tile.
                        float vg=jangh_level_g(jangh_read<{bg}>(pg,col+j))*float(sg_weights[ulong(safe_expert)*{h}u+row]);
                        float vu=jangh_level_u(jangh_read<{bu}>(pu,col+j))*float(su_weights[ulong(safe_expert)*{h}u+row]);
                        #pragma clang loop unroll(full)
                        for (uint b = 0; b < {block}u; ++b) {{
                            uint route=first_route+b;
                            if (route<route_end) {{
                                uint token=token_rows[route];
                                ag[b][r]=fma(vg,xg[token*{d}u+col+j],ag[b][r]);
                                au[b][r]=fma(vu,xu[token*{d}u+col+j],au[b][r]);
                            }}
                        }}
                    }}
                }}
            }}
        }}
    }}
    for (uint b = 0; b < {block}u; ++b) {{
        uint route=first_route+b;
        for (uint r=0u;r<{rps}u;++r) {{
            uint row=tile_base+sg*{rps}u+r;
            // SIMD reduction executes in every lane, even for tail routes/rows.
            float g=ag[b][r],u=au[b][r],value=0.0f;
            for (ushort step=16;step>0;step>>=1) {{
                g+=simd_shuffle_down(g,step);
                u+=simd_shuffle_down(u,step);
            }}
            if (route<route_end && row<{h}u) {{
                float ex=metal::precise::exp(-abs(g));
                float sig=g>=0.0f ? 1.0f/(1.0f+ex) : ex/(1.0f+ex);
                value=invalid ? NAN_VALUE : (g*sig)*u;
            }}
            {store}
        }}
    }}
    {epilogue}
'''
    header=_READ_HEADER+_table_function('jangh_level_g',spec.levels[0])+_table_function('jangh_level_u',spec.levels[1])
    return _definition('grouped_gate',('xg','xu','wg','sg_weights','wu','su_weights','starts','tile_ends','token_rows'),source,header)

@functools.lru_cache(maxsize=128)
def build_grouped_down(spec: KernelSpec, block: int=4) -> KernelDefinition:
    if block not in (2,4):raise ValueError('Invalid grouped down tile')
    n,k=spec.input_dims,spec.hidden_dims;b=spec.bits[2];rps=spec.down_rows
    source=_preamble(spec,block)+f'''
    const uint tile_base=threadgroup_position_in_grid.x*{2*rps}u;
    float acc[{block}][{rps}]={{{{0.0f}}}};
    for (uint input_block=0u;input_block<{k}u;input_block+=128u) {{
        uint col=input_block+lane*4u;
        for (uint r=0u;r<{rps}u;++r) {{
            uint row=tile_base+sg*{rps}u+r;
            if (row<{n}u && col<{k}u) {{
                const device uint* p=wd+(ulong(safe_expert)*{n}u+row)*{k*b//32}u;
                for (uint j=0u;j<4u;++j) {{
                    if (col+j<{k}u) {{
                        float w=jangh_level_d(jangh_read<{b}>(p,col+j))*float(sd_weights[ulong(safe_expert)*{n}u+row]);
                        #pragma clang loop unroll(full)
                        for (uint b = 0; b < {block}u; ++b) {{
                            uint route=first_route+b;
                            if (route<route_end) acc[b][r]=fma(w,hidden[route*{k}u+col+j],acc[b][r]);
                        }}
                    }}
                }}
            }}
        }}
    }}
    for (uint b = 0; b < {block}u; ++b) {{
        uint route=first_route+b;
        for (uint r=0u;r<{rps}u;++r) {{
            uint row=tile_base+sg*{rps}u+r;
            float value=acc[b][r];
            for (ushort step=16;step>0;step>>=1)
                value+=simd_shuffle_down(value,step);
            if (lane==0u && route<route_end && row<{n}u) {{
                out[route*{n}u+row]=OutType(invalid ? NAN_VALUE : value);
            }}
        }}
    }}
'''
    return _definition('grouped_down',('hidden','wd','sd_weights','starts','tile_ends'),source,
                       _READ_HEADER+_table_function('jangh_level_d',spec.levels[2]))


def forward(x,indices,projections,spec:KernelSpec,mx,*,block=4):
    plan=route_plan(x.shape,indices.shape,spec)
    if x.dtype not in (mx.float16,mx.bfloat16,mx.float32):raise TypeError('Unsupported activation dtype')
    if set(projections)!=set(PROJECTIONS):raise ValueError('Complete expert triplet required')
    for name,b,n,k in zip(PROJECTIONS,spec.bits,(spec.hidden_dims,spec.hidden_dims,spec.input_dims),
                          (spec.input_dims,spec.input_dims,spec.hidden_dims)):
        p,s=projections[name]
        if (tuple(p.shape)!=(spec.experts,n,k*b//32) or p.dtype!=mx.uint32 or
            tuple(s.shape)!=(spec.experts,n) or s.dtype!=mx.float16):
            raise ValueError(f'Packed shape/dtype mismatch: {name}')
    sched=schedule(indices,spec.experts,block,mx)
    if not plan.routes:return mx.zeros(plan.output_shape,dtype=x.dtype)
    xf=x.reshape(-1,spec.input_dims).astype(mx.float32)
    rot=mx.hadamard_transform(xf.reshape(-1,32)).reshape(xf.shape) if 'hadamard32' in spec.rotations[:2] else None
    xg=rot if spec.rotations[0]=='hadamard32' else xf
    xu=rot if spec.rotations[1]=='hadamard32' else xf
    tokens=(sched.order//plan.top_k).astype(mx.uint32)
    pg,sg=projections['gate_proj'];pu,su=projections['up_proj'];pd,sd=projections['down_proj']
    gu=_kernel_for(mx,build_grouped_gate(spec,plan.top_k,block))
    hidden=gu(inputs=[xg,xu,pg,sg,pu,su,sched.starts,sched.tile_ends,tokens],
              output_shapes=[(plan.routes,spec.hidden_dims)],output_dtypes=[mx.float32],
              grid=(plan.gate_grid[0],sched.max_tiles,1),threadgroup=plan.gate_threads)[0]
    down=_kernel_for(mx,build_grouped_down(spec,block))
    result=down(inputs=[hidden,pd,sd,sched.starts,sched.tile_ends],template=[('OutType',x.dtype)],
                output_shapes=[(plan.routes,spec.input_dims)],output_dtypes=[x.dtype],
                grid=(plan.down_grid[0],sched.max_tiles,1),threadgroup=plan.down_threads)[0]
    return result[sched.inverse].reshape(plan.output_shape)
