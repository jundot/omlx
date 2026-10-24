import copy
import unittest
from pathlib import Path
import tempfile
import json
import numpy as np

from jangh2_bridge.metadata import (
    canonical, canonical_map, validate_config, expert_specs, normalize_quantization,
    is_ple_storage, choose_ple_prefix, check_tensor_pair, safe_shard_path,
)
from jangh2_bridge.reference_ops import pack_codes, unpack_codes, hadamard32, expert_forward

LEVELS = [-1.5104438066482544, -0.4528312385082245, 0.4528312385082245, 1.5104438066482544]

def config():
    return {'model_type': 'qwen4_exp', 'jangtq': {'version': 2, 'packing': 'lsb-bitstream',
        'scale_dtype': 'float16', 'rotation': 'hadamard32', 'codebook_family': 'odd-cubic',
        'codebooks': {'2': {'levels': list(LEVELS), 'alpha': .893, 'beta': .05065}}},
        'quantization': {'bits': 8, 'group_size': 64, **{
            f'model.layers.0.mlp.switch_mlp.{p}': {'mode': 'jangtq2', 'bits': 2, 'rotation': 'hadamard32'}
            for p in ('gate_proj','up_proj','down_proj')}}}

class MetadataTests(unittest.TestCase):
    def test_aliases(self):
        suffix = 'layers.1.mlp.switch_mlp.gate_proj.tq2_packed'
        for pre in ['model.language_model.', 'language_model.model.', 'language_model.', 'model.']:
            self.assertEqual(canonical(pre+suffix), 'language_model.model.'+suffix)
        self.assertEqual(canonical('visual.blocks.0.weight'), 'vision_tower.blocks.0.weight')
        self.assertEqual(canonical('lm_head.weight'), 'language_model.lm_head.weight')
    def test_ple_alias(self):
        self.assertEqual(canonical('language_model.layers.1.ple.ngram_embedding.shard_3.weight'),
            'language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.3.weight')
    def test_no_broad_rewrites(self):
        self.assertEqual(canonical('foo.model.layers.0.weight'), 'foo.model.layers.0.weight')
    def test_collision_rejected(self):
        with self.assertRaises(ValueError): canonical_map({'lm_head.weight': object(), 'language_model.lm_head.weight': object()})
    def test_shared_object_alias_allowed(self):
        x = object()
        self.assertEqual(len(canonical_map({'lm_head.weight': x, 'language_model.lm_head.weight': x})), 1)
    def test_metadata(self):
        self.assertEqual(validate_config(config())[2], tuple(LEVELS))
    def test_wrong_version(self):
        c=config(); c['jangtq']['version']=1
        with self.assertRaises(ValueError): validate_config(c)
    def test_wrong_packing(self):
        c=config(); c['jangtq']['packing']='unknown'
        with self.assertRaises(ValueError): validate_config(c)
    def test_nonfinite_levels(self):
        c=config(); c['jangtq']['codebooks']['2']['levels'][0]=float('nan')
        with self.assertRaises(ValueError): validate_config(c)
    def test_custom_levels_rejected(self):
        c=config(); c['jangtq']['codebooks']['2']['levels'][0]=-10
        with self.assertRaises(ValueError): validate_config(c)
    def test_quant_aliases(self):
        c=config(); q=normalize_quantization(c)
        self.assertEqual(q['language_model.model.layers.0.mlp.switch_mlp.gate_proj']['rotation'],'hadamard32')
        self.assertEqual(len(expert_specs(q)),1)
    def test_missing_projection(self):
        c=config(); del c['quantization']['model.layers.0.mlp.switch_mlp.down_proj']
        with self.assertRaises(ValueError): expert_specs(normalize_quantization(c))
    def test_alias_config_conflict(self):
        c=config(); c['quantization']['language_model.layers.0.mlp.switch_mlp.gate_proj']={'bits':4,'mode':'jangtq2','rotation':'hadamard32'}
        with self.assertRaises(ValueError): normalize_quantization(c)
    def test_ple_only_recognition(self):
        self.assertTrue(is_ple_storage('language_model.layers.1.ple.ngram_embedding.shards.0.scales'))
        self.assertFalse(is_ple_storage('language_model.layers.1.ple.key_proj.weight'))
        self.assertFalse(is_ple_storage('language_model.layers.1.ple.ngram_embedding.weight_scale'))
    def test_ple_prefix_resolver(self):
        raw='language_model.layers.1.ple.ngram_embedding'
        keys={raw+'.shards.0.weight':'a.safetensors'}
        self.assertEqual(choose_ple_prefix('model.language_model.layers.1.ple.ple_embedding.ngram_embedding', keys), raw)
    def test_ple_missing_rejected(self):
        with self.assertRaises(ValueError): choose_ple_prefix('model.language_model.layers.1.ple.ple_embedding.ngram_embedding',{})
    def test_shape_checks(self):
        check_tensor_pair(np.zeros((4,64,8),np.uint32),np.ones((4,64),np.float16),4,64,128,2)
        with self.assertRaises(ValueError): check_tensor_pair(np.zeros((4,64,8),np.uint32),np.ones((4,64,1),np.float16),4,64,128,2)
    def test_dtype_checks(self):
        with self.assertRaises(ValueError): check_tensor_pair(np.zeros((4,64,8),np.float32),np.ones((4,64),np.float16),4,64,128,2)
    def test_shard_traversal(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError): safe_shard_path(Path(d), '../secret')

class ArithmeticTests(unittest.TestCase):
    def test_known_lsb_word(self):
        q=np.tile(np.arange(4,dtype=np.uint8),8)[None]
        packed=pack_codes(q,2,np)
        np.testing.assert_array_equal(packed,np.array([[0xe4e4e4e4,0xe4e4e4e4]],np.uint32))
    def test_pack_roundtrips(self):
        rng=np.random.default_rng(4)
        for b in [2,3,4]:
            q=rng.integers(0,1<<b,(2,3,96),dtype=np.uint8)
            np.testing.assert_array_equal(unpack_codes(pack_codes(q,b,np),b,96,np),q)
    def test_hadamard_explicit(self):
        H=np.ones((1,1),np.float32)
        for _ in range(5): H=np.block([[H,H],[H,-H]])
        H/=np.sqrt(32)
        x=np.random.default_rng(2).normal(size=(3,64)).astype(np.float32)
        expected=(x.reshape(3,2,32)@H.T).reshape(3,64)
        np.testing.assert_allclose(hadamard32(x,np),expected,atol=2e-6)
    def test_hadamard_inverse(self):
        x=np.random.default_rng(3).normal(size=(2,96)).astype(np.float32)
        np.testing.assert_allclose(hadamard32(hadamard32(x,np),np),x,atol=2e-6)
    def test_bad_width(self):
        with self.assertRaises(ValueError): hadamard32(np.zeros((2,33),np.float32),np)
    def test_expert_forward_against_dense_reference(self):
        rng=np.random.default_rng(9); E,D,I,T,K=4,64,32,5,2
        data={}; dense={}; levels=np.asarray(LEVELS,np.float32)
        for p,n,k in [('gate_proj',I,D),('up_proj',I,D),('down_proj',D,I)]:
            q=rng.integers(0,4,(E,n,k),dtype=np.uint8)
            scales=rng.uniform(.01,.08,(E,n)).astype(np.float16)
            data[p]=(pack_codes(q,2,np),scales,2,levels,'hadamard32')
            dense[p]=levels[q]*scales.astype(np.float32)[...,None]
        x=rng.normal(size=(T,D)).astype(np.float32)
        idx=np.array([[0,1],[2,3],[1,2],[3,0],[0,2]],np.uint32)
        H=np.ones((1,1),np.float32)
        for _ in range(5): H=np.block([[H,H],[H,-H]])
        H/=np.sqrt(32)
        def rotate(v): return (v.reshape(-1,32)@H.T).reshape(v.shape)
        expected=np.empty((T,K,D),np.float32)
        for t in range(T):
            for j in range(K):
                e=int(idx[t,j]); xx=rotate(x[t]); g=dense['gate_proj'][e]@xx; u=dense['up_proj'][e]@xx
                h=(g/(1+np.exp(-g)))*u
                expected[t,j]=dense['down_proj'][e]@rotate(h)
        for chunk in [1,3,20]:
            actual=expert_forward(x,idx,data,np,chunk_size=chunk)
            np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-6)
    def test_invalid_indices(self):
        with self.assertRaises(ValueError): expert_forward(np.zeros((2,32)), np.zeros((3,2),np.uint32), {},np)


class ExtraMetadataTests(unittest.TestCase):
    def test_quantized_module_root_aliases(self):
        self.assertEqual(canonical('language_model.embed_tokens'), 'language_model.model.embed_tokens')
        self.assertEqual(canonical('lm_head'), 'language_model.lm_head')
        self.assertEqual(canonical('model.norm'), 'language_model.model.norm')

class CheckpointTests(unittest.TestCase):
    def test_read_safe_header_and_index(self):
        from jangh2_bridge.checkpoint import inspect_checkpoint
        import struct
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); c=config(); c['text_config']={'hidden_size':64, 'moe_intermediate_size':32, 'num_experts':4}
            tensors={}; raw=bytearray(); weights={}
            for proj,n,k in [('gate_proj',32,64),('up_proj',32,64),('down_proj',64,32)]:
                for suffix,array in [('tq2_packed',np.zeros((4,n,k*2//32),np.uint32)),('tq2_scales',np.ones((4,n),np.float16))]:
                    key=f'language_model.layers.0.mlp.switch_mlp.{proj}.{suffix}'
                    value=array.tobytes(); start=len(raw); raw.extend(value)
                    tensors[key]={'shape':list(array.shape),'dtype':'U32' if suffix=='tq2_packed' else 'F16','data_offsets':[start,len(raw)]}
                    weights[key]='model.safetensors'
            header=json.dumps(tensors).encode(); (root/'model.safetensors').write_bytes(struct.pack('<Q',len(header))+header+raw)
            (root/'config.json').write_text(json.dumps(c)); (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':weights}))
            cp=inspect_checkpoint(root)
            self.assertEqual(cp.report()['expert_blocks'],1)
            self.assertEqual(cp.report()['resident_weight_bytes'],len(raw))
            self.assertEqual(len(cp.tensors),6)
            # Tampering with an indexed tensor name must fail, not leave random init.
            weights['model.bad.weight']='model.safetensors'
            (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':weights}))
            with self.assertRaises(ValueError): inspect_checkpoint(root)
    def test_header_size_bounds(self):
        from jangh2_bridge.checkpoint import read_header
        import struct
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.safetensors'; p.write_bytes(struct.pack('<Q',2**40))
            with self.assertRaises(ValueError): read_header(p)

if __name__=='__main__': unittest.main()
