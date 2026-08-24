# SPDX-License-Identifier: Apache-2.0
"""Research candidate: serial-exact scaled Q4/Q5 group64 cross-row QMV."""
from __future__ import annotations
_CACHE={}
NSG=4
MIN_N_Q4=1
MIN_N_Q5=1


def _pieces(m,bits):
    if bits==4:
        setup="ushort packed[4][4];"
        wload="const device uchar* w=reinterpret_cast<const device uchar*>(w_q)+o*(K/2)+k/2+int(lane)*8;const device ushort* ws=reinterpret_cast<const device ushort*>(w);for(int i=0;i<4;i++)packed[r][i]=ws[i];"
        sums="const device T* z=x+row*K+k+int(lane)*16;for(int i=0;i<16;i+=4)sums[row]+=z[i]+z[i+1]+z[i+2]+z[i+3];"
        math="""for(int i=0;i<4;i++){VF a0,a1,a2,a3;for(int row=0;row<M;row++){const device T* z=x+row*K+k+int(lane)*16+4*i;a0[row]=float(z[0]);a1[row]=float(z[1]);a2[row]=float(z[2]);a3[row]=float(z[3]);}for(int r=0;r<4;r++){ushort p=packed[r][i];part[r]+=a0*float(p&15)+a1*float((p>>4)&15)+a2*float((p>>8)&15)+a3*float((p>>12)&15);}}"""
    else:
        setup="uchar packed[4][10];"
        wload="const device uchar* w=reinterpret_cast<const device uchar*>(w_q)+o*(K*5/8)+k*5/8+int(lane)*10;*reinterpret_cast<thread packed_uchar4*>(&packed[r][0])=*reinterpret_cast<const device packed_uchar4*>(w);*reinterpret_cast<thread packed_uchar4*>(&packed[r][4])=*reinterpret_cast<const device packed_uchar4*>(w+4);*reinterpret_cast<thread packed_uchar2*>(&packed[r][8])=*reinterpret_cast<const device packed_uchar2*>(w+8);"
        sums="const device T* z=x+row*K+k+int(lane)*16;sums[row]+=z[0]+z[1]+z[2]+z[3]+z[4]+z[5]+z[6]+z[7];sums[row]+=z[8]+z[9]+z[10]+z[11]+z[12]+z[13]+z[14]+z[15];"
        math="""for(int pack=0;pack<2;pack++){VF a0,a1,a2,a3,a4,a5,a6,a7;for(int row=0;row<M;row++){const device T* z=x+row*K+k+int(lane)*16+pack*8;a0[row]=float(z[0]);a1[row]=float(z[1])/32.0f;a2[row]=float(z[2])/4.0f;a3[row]=float(z[3])/128.0f;a4[row]=float(z[4])/16.0f;a5[row]=float(z[5])/2.0f;a6[row]=float(z[6])/64.0f;a7[row]=float(z[7])/8.0f;}int b=pack*5;for(int r=0;r<4;r++){part[r]+=(packed[r][b]&0x1f)*a0;part[r]+=(packed[r][b]&0xe0)*a1;part[r]+=(packed[r][b+1]&0x03)*(a1*256.0f);part[r]+=(packed[r][b+1]&0x7c)*a2;part[r]+=(packed[r][b+1]&0x80)*a3;part[r]+=(packed[r][b+2]&0x0f)*(a3*256.0f);part[r]+=(packed[r][b+2]&0xf0)*a4;part[r]+=(packed[r][b+3]&0x01)*(a4*256.0f);part[r]+=(packed[r][b+3]&0x3e)*a5;part[r]+=(packed[r][b+3]&0xc0)*a6;part[r]+=(packed[r][b+4]&0x07)*(a6*256.0f);part[r]+=(packed[r][b+4]&0xf8)*a7;}}"""
    return setup,wload,sums,math


def _kernel(m,bits,dtype,nsg):
    import mlx.core as mx
    key=(m,bits,dtype,nsg)
    if key in _CACHE:return _CACHE[key]
    setup,wload,sums,math=_pieces(m,bits)
    src=f"""using namespace metal;constexpr int M={m};typedef vec<float,M> VF;uint lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;uint tile=threadgroup_position_in_grid.y;int K=int(K_size),N=int(N_size),out_row=int(tile)*{4*nsg}+int(sg)*4;VF acc[4]={{VF(0),VF(0),VF(0),VF(0)}};for(int k=0;k<K;k+=512){{{setup}float sc[4],bi[4];for(int r=0;r<4;r++){{int o=out_row+r;{wload}int gi=o*(K/64)+k/64+int(lane)/4;sc[r]=float(scales[gi]);bi[r]=float(biases[gi]);}}VF sums=VF(0),part[4]={{VF(0),VF(0),VF(0),VF(0)}};for(int row=0;row<M;row++){{{sums}}}{math}for(int r=0;r<4;r++)acc[r]+=sc[r]*part[r]+sums*bi[r];}}for(int r=0;r<4;r++)for(int row=0;row<M;row++){{float v=simd_sum(acc[r][row]);if(lane==0)y[row*N+out_row+r]=T(v);}}"""
    for loop in (
        "for(int r=0;r<4;r++)",
        "for(int row=0;row<M;row++)",
        "for(int pack=0;pack<2;pack++)",
        "for(int i=0;i<4;i++)",
        "for(int i=0;i<10;i++)",
    ):
        src=src.replace(loop, '_Pragma("unroll")\n'+loop)
    tag={mx.bfloat16:'bf16',mx.float16:'fp16'}.get(dtype,'unk');k=mx.fast.metal_kernel(name=f'omlx_q35_exact_q{bits}_m{m}_nsg{nsg}_{tag}',input_names=['x','w_q','scales','biases','K_size','N_size'],output_names=['y'],source=src);_CACHE[key]=k;return k


def exact_crossrow(linear, x):
    import mlx.core as mx

    meta = getattr(linear, "_omlx_exact_crossrow_meta", None)
    if meta is None:
        bits = int(getattr(linear, "bits", 0))
        n = int(linear.weight.shape[0])
        nsg = 1 if n < 1024 else (8 if n >= 100000 else NSG)
        min_n = MIN_N_Q4 if bits == 4 else MIN_N_Q5
        biases = getattr(linear, "biases", None)
        scale_dtype = linear.scales.dtype
        supported = (
            bits in (4, 5)
            and linear.group_size == 64
            and linear.mode == "affine"
            and biases is not None
            and biases.dtype == scale_dtype
            and scale_dtype in (mx.bfloat16, mx.float16)
            and n >= min_n
            and n % (4 * nsg) == 0
        )
        input_dim = int(linear.weight.shape[1]) * 32 // max(bits, 1)
        meta = (supported, bits, n, nsg, scale_dtype, input_dim, {})
        linear._omlx_exact_crossrow_meta = meta
    supported, bits, n, nsg, scale_dtype, input_dim, launches = meta
    m = int(x.shape[1]) if x.ndim == 3 else 0
    if not (
        supported
        and m in (2, 3, 4)
        and x.dtype == scale_dtype
        and x.shape[-1] == input_dim
        and input_dim % 512 == 0
    ):
        return None
    launch = launches.get(m)
    if launch is None:
        launch = (
            _kernel(m, bits, x.dtype, nsg),
            [("T", x.dtype)],
            (32 * nsg, n // (4 * nsg), 1),
            (32 * nsg, 1, 1),
            [(m, n)],
            [x.dtype],
        )
        launches[m] = launch
    kernel, template, grid, threadgroup, output_shapes, output_dtypes = launch
    (y,) = kernel(
        inputs=[x[0], linear.weight, linear.scales, linear.biases, input_dim, n],
        template=template,
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
    )
    return (y + linear.bias if "bias" in linear else y)[None]
