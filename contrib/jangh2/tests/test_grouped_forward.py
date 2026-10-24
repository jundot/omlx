"""NumPy kernel doubles verify grouped dispatch/order, not Metal compilation."""
from types import SimpleNamespace
import unittest
import numpy as np
from jangh2_bridge.grouped_ops import forward
from jangh2_bridge.reference_ops import unpack_codes,expert_forward,hadamard32
from test_metal_backend import make_spec
from test_metal_forward_contract import fixture

class GroupedDouble:
    float16=np.float16;float32=np.float32;bfloat16='bf16';uint32=np.uint32
    def __init__(self,spec):
        self.spec=spec;self.calls=[];self.fast=SimpleNamespace(metal_kernel=self.factory)
    def __getattr__(self,name):
        if name in ('eval','synchronize','async_eval'):raise AssertionError('Host synchronisation in grouped path')
        return getattr(np,name)
    def hadamard_transform(self,x):return hadamard32(x,np)
    def factory(self,**definition):
        kind='gate' if 'grouped_gate' in definition['name'] else 'down';s=self.spec
        def kernel(**kw):
            self.calls.append((kind,kw));out=np.empty(kw['output_shapes'][0],np.float32)
            if kind=='gate':
                xg,xu,pg,sg,pu,su,starts,tiles,tokens=kw['inputs']
                wg=np.asarray(s.levels[0],np.float32)[unpack_codes(pg,s.bits[0],s.input_dims)]*sg[...,None]
                wu=np.asarray(s.levels[1],np.float32)[unpack_codes(pu,s.bits[1],s.input_dims)]*su[...,None]
                for e in range(s.experts+1):
                    for r in range(int(starts[e]),int(starts[e+1])):
                        if e==s.experts:out[r]=np.nan;continue
                        g=wg[e]@xg[tokens[r]];u=wu[e]@xu[tokens[r]]
                        ex=np.exp(-np.abs(g));sig=np.where(g>=0,1/(1+ex),ex/(1+ex))
                        out[r]=g*sig*u
                if s.rotations[2]=='hadamard32':out=hadamard32(out,np)
            else:
                hidden,pd,sd,starts,tiles=kw['inputs']
                wd=np.asarray(s.levels[2],np.float32)[unpack_codes(pd,s.bits[2],s.hidden_dims)]*sd[...,None]
                for e in range(s.experts+1):
                    for r in range(int(starts[e]),int(starts[e+1])):
                        out[r]=wd[e]@hidden[r] if e<s.experts else np.nan
            return [out.astype(kw['output_dtypes'][0])]
        return kernel

class GroupedForwardTests(unittest.TestCase):
    def test_sorted_restoration_and_exact_metadata_across_tile_sizes(self):
        for bits,rot in [((2,2,2),('hadamard32',)*3),((2,3,4),('none','hadamard32','none'))]:
            s=make_spec(bits,rot,d=96,h=64,e=5)
            for tokens in (1,17):
                x,ids,p,old=fixture(s,tokens,3)
                expect=expert_forward(x,ids,old,np,chunk_size=8,evaluation='per_chunk')
                for tile in (2,4):
                    xp=GroupedDouble(s);out=forward(x,ids,p,s,xp,block=tile)
                    np.testing.assert_allclose(out,expect,rtol=3e-5,atol=1e-6)
                    self.assertEqual([k for k,_ in xp.calls],['gate','down'])
    def test_empty_and_invalid_routes(self):
        s=make_spec(d=64,h=32,e=4);x,ids,p,_=fixture(s,3)
        xp=GroupedDouble(s);out=forward(x[:,:0],ids[:,:0],p,s,xp)
        self.assertEqual(out.shape,(2,0,3,64));self.assertEqual(xp.calls,[])
        ids=ids.astype(np.int64);ids[0,0,0]=2**32;ids[1,0,0]=-1
        out=forward(x,ids,p,s,xp)
        self.assertTrue(np.isnan(out[:,0,0]).all());self.assertTrue(np.isfinite(out[:,:,1:]).all())
    def test_shape_dtype_contracts_not_relaxed(self):
        s=make_spec(d=64,h=32,e=4);x,ids,p,_=fixture(s,3)
        with self.assertRaises(TypeError):forward(x,ids.astype(np.float32),p,s,GroupedDouble(s))
        with self.assertRaises(ValueError):forward(x,ids,{'gate_proj':p['gate_proj']},s,GroupedDouble(s))
