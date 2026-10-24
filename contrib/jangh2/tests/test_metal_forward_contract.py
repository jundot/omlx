"""Host-side routing, shape and launch contracts with NumPy kernel doubles."""
from types import SimpleNamespace
import unittest
import numpy as np
from test_metal_backend import make_spec
from jangh2_bridge.metal_ops import forward
from jangh2_bridge.reference_ops import pack_codes,unpack_codes,expert_forward,hadamard32

class FakeMLX:
    float32=np.float32;float16=np.float16;bfloat16='BF16';uint32=np.uint32
    def __init__(self,spec):
        self.spec=spec;self.calls=[];self.fast=SimpleNamespace(metal_kernel=self.factory)
    def __getattr__(self,name):
        if name in ('eval','synchronize','async_eval'): raise AssertionError('Unexpected synchronization')
        return getattr(np,name)
    def hadamard_transform(self,x): return hadamard32(x,np)
    def factory(self,**definition):
        s=self.spec;source=definition['source'];kind='gate_up' if 'gate_up' in definition['name'] else 'down'
        def kernel(**kw):
            self.calls.append((kind,kw));ins=kw['inputs'];out_shape=kw['output_shapes'][0]
            out=np.empty(out_shape,np.float32)
            if kind=='gate_up':
                xg,xu,pg,sg,pu,su,ids,tokenrows=ins
                top_k=int(source.split('route / ')[1].split('u')[0]) if 'route / ' in source else None
                wg=np.asarray(s.levels[0],np.float32)[unpack_codes(pg,s.bits[0],s.input_dims)]*sg.astype(np.float32)[...,None]
                wu=np.asarray(s.levels[1],np.float32)[unpack_codes(pu,s.bits[1],s.input_dims)]*su.astype(np.float32)[...,None]
                for r,e in enumerate(ids):
                    if e>=s.experts: out[r]=np.nan;continue
                    t=int(tokenrows[r]) if top_k is None else r//top_k
                    g=wg[e]@xg[t];u=wu[e]@xu[t]
                    ex=np.exp(-np.abs(g));sig=np.where(g>=0,1/(1+ex),ex/(1+ex))
                    out[r]=g*sig*u
                if 'threadgroup_barrier' in source: out=hadamard32(out,np)
            else:
                hidden,pd,sd,ids=ins
                wd=np.asarray(s.levels[2],np.float32)[unpack_codes(pd,s.bits[2],s.hidden_dims)]*sd.astype(np.float32)[...,None]
                for r,e in enumerate(ids): out[r]=wd[e]@hidden[r] if e<s.experts else np.nan
            return [out.astype(kw['output_dtypes'][0])]
        return kernel

def fixture(spec,tokens,top_k=3):
    rng=np.random.default_rng(162);p={};old={}
    for name,b,n,k,lv,rot in zip(('gate_proj','up_proj','down_proj'),spec.bits,
        (spec.hidden_dims,spec.hidden_dims,spec.input_dims),
        (spec.input_dims,spec.input_dims,spec.hidden_dims),spec.levels,spec.rotations):
        q=rng.integers(0,1<<b,(spec.experts,n,k),dtype=np.uint32)
        sc=rng.uniform(.01,.05,(spec.experts,n)).astype(np.float16)
        packed=pack_codes(q,b);p[name]=(packed,sc);old[name]=(packed,sc,b,np.array(lv,np.float32),rot)
    x=rng.normal(0,.2,(2,tokens,spec.input_dims*2)).astype(np.float32)[...,::2]
    ids=rng.integers(0,spec.experts,(2,tokens,top_k),dtype=np.uint32)
    return x,ids,p,old

class ForwardContractTests(unittest.TestCase):
    def test_sorted_unsorted_noncontiguous_repeated_routes_and_mixed_metadata(self):
        for bits,rot in [((2,2,2),('hadamard32',)*3),((2,3,4),('none','hadamard32','none'))]:
            for tokens in (1,15):
                with self.subTest(bits=bits,tokens=tokens):
                    s=make_spec(bits,rot,d=96,h=64,e=5);xp=FakeMLX(s)
                    x,ids,p,old=fixture(s,tokens)
                    actual=forward(x,ids,p,s,xp)
                    expected=expert_forward(x,ids,old,np,chunk_size=8,evaluation='per_chunk')
                    np.testing.assert_allclose(actual,expected,rtol=3e-5,atol=1e-6)
                    self.assertEqual([k for k,_ in xp.calls],['gate_up','down'])
    def test_invalid_ids_emit_nan_without_an_out_of_bounds_read(self):
        s=make_spec(d=64,h=32,e=4);x,ids,p,_=fixture(s,1)
        for dtype,bad in [(np.int32,-1),(np.uint32,999),(np.uint64,2**32),(np.int64,-2**32)]:
            with self.subTest(dtype=dtype,bad=bad):
                v=ids.astype(dtype);v[0,0,0]=bad
                out=forward(x,v,p,s,FakeMLX(s))
                self.assertTrue(np.isnan(out[0,0,0]).all())
                self.assertTrue(np.isfinite(out[0,0,1:]).all())
    def test_shape_and_dtype_checks_remain_active(self):
        s=make_spec(d=64,h=32,e=4);x,ids,p,_=fixture(s,1)
        with self.assertRaises(TypeError): forward(x,ids.astype(np.float32),p,s,FakeMLX(s))
        with self.assertRaises(TypeError): forward(x.astype(np.float64),ids,p,s,FakeMLX(s))
        pp=dict(p);pp['up_proj']=(p['up_proj'][0],p['up_proj'][1].astype(np.float32))
        with self.assertRaises(ValueError): forward(x,ids,pp,s,FakeMLX(s))
    def test_empty_routes_have_no_dispatch(self):
        s=make_spec(d=64,h=32,e=4);x,ids,p,_=fixture(s,1);xp=FakeMLX(s)
        out=forward(x[:,:0],ids[:,:0],p,s,xp)
        self.assertEqual(out.shape,(2,0,3,64));self.assertEqual(xp.calls,[])

if __name__=='__main__':unittest.main()
