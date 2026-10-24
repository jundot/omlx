"""Read-only SSD lookup and stock-Qwen4 constructor regression tests."""
import json
from pathlib import Path
import struct
from types import ModuleType, SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from jangh2_bridge.checkpoint import Checkpoint
from jangh2_bridge.metadata import canonical


def fixture(root, *, bits=8, dtype='F16', shards=2, layer_idx=1, code_offset=0):
    """Small real safetensors files, no safetensors/MLX dependency."""
    root=Path(root)
    prefix=f'language_model.layers.{layer_idx}.ple.ngram_embedding.shards.'
    dims, group, sizes=32, 32, [3,2][:shards]
    weight_map={}; expected=[]; quant={'bits':4,'group_size':32,'mode':'affine'}
    for i,rows in enumerate(sizes):
        codes=(np.arange(rows*dims,dtype=np.uint32).reshape(rows,dims)+i*7+code_offset) % (1<<bits)
        packed=np.zeros((rows,dims*bits//32),np.uint32)
        for col in range(dims):
            packed[:,col*bits//32] |= codes[:,col] << ((col*bits)%32)
        scales=np.full((rows,1),0.5,np.float32)
        biases=np.full((rows,1),-1.0,np.float32)
        def aux(x):
            if dtype=='BF16': return (x.view(np.uint32)>>16).astype('<u2')
            return x.astype({'F16':'<f2','F32':'<f4'}[dtype])
        arrays={'weight':('U32',packed),'scales':(dtype,aux(scales)),'biases':(dtype,aux(biases))}
        header={}; payload=b''
        for name,(dt,arr) in arrays.items():
            key=prefix+str(i)+'.'+name; data=arr.tobytes()
            header[key]={'dtype':dt,'shape':list(arr.shape),'data_offsets':[len(payload),len(payload)+len(data)]}
            payload+=data
            weight_map[key]=f'part-{i}.safetensors'
        encoded=json.dumps(header).encode()
        (root/f'part-{i}.safetensors').write_bytes(struct.pack('<Q',len(encoded))+encoded+payload)
        quant[prefix+str(i)]={'bits':bits,'group_size':group,'mode':'affine'}
        expected.append(codes.astype(np.float32)*scales+biases)
    config={'model_type':'qwen4_exp','text_config':{'ple_layer_ids':[layer_idx+1]},'quantization':quant}
    ckpt=Checkpoint(root,config,weight_map,{},'fixture')
    return ckpt,np.concatenate(expected),sum(sizes),dims,len(sizes)


def fake_language():
    """Separate module globals model how upstream resolves ShardedEmbedding."""
    m=ModuleType('fixture_qwen4_language')
    m.__file__='fixture_qwen4_language.py'
    exec('''
class ShardedEmbedding:
    def __init__(self, num_embeddings, dims, num_shards):
        raise AssertionError("Resident PLE constructor must never be called")
class Qwen4ExpNGramEmbedding:
    def __init__(self, config, embedding_dim, layer_idx, ple_layer_index):
        self.layer_idx = layer_idx
        self.ple_layer_index = ple_layer_index
        self.ngram_embedding = ShardedEmbedding(config.num_embeddings, embedding_dim, config.num_shards)
''',m.__dict__)
    return m


class MMapPLETests(unittest.TestCase):
    def test_q4_q8_read_actual_indexed_rows(self):
        from jangh2_bridge.ple import IndexedMMapEmbedding
        for bits in (4,8):
            for dtype in ('F16','BF16','F32'):
                with self.subTest(bits=bits,dtype=dtype), tempfile.TemporaryDirectory() as d:
                    ckpt,w,n,dim,s=fixture(d,bits=bits,dtype=dtype)
                    reader=IndexedMMapEmbedding(ckpt,1,n,dim,s)
                    ids=np.array([4,0,3,2,4],np.int64)
                    packed,scale,bias=reader.read_rows(ids)
                    # An independent, simple affine decoder for the fixture.
                    values=np.stack([((packed[:,j*bits//32] >> ((j*bits)%32)) & ((1<<bits)-1)) for j in range(dim)],axis=1)
                    np.testing.assert_array_equal(values*scale+bias,w[ids])
                    self.assertEqual(reader.row_count,n)
                    self.assertEqual(reader.row_width,dim)
                    self.assertEqual(reader.bits,bits)
                    self.assertEqual(reader.rows_read,len(ids))
                    reader.close()
                    with self.assertRaises(RuntimeError): reader.read_rows(ids)

    def test_no_payload_read_during_setup_and_readonly_mapping(self):
        from jangh2_bridge.ple import IndexedMMapEmbedding
        with tempfile.TemporaryDirectory() as d:
            ckpt,_,n,dim,s=fixture(d)
            before={p.name:p.read_bytes() for p in Path(d).iterdir()}
            reader=IndexedMMapEmbedding(ckpt,1,n,dim,s)
            self.assertEqual(reader.rows_read,0)
            self.assertTrue(reader.mapping_stats()['read_only'])
            self.assertEqual(reader.mapping_stats()['open_files'], 0)
            reader.read_rows(np.array([0]))
            reader.close()
            self.assertEqual(before,{p.name:p.read_bytes() for p in Path(d).iterdir()})

    def test_bounds_integer_and_closed_validation(self):
        from jangh2_bridge.ple import IndexedMMapEmbedding
        with tempfile.TemporaryDirectory() as d:
            ckpt,_,n,dim,s=fixture(d)
            reader=IndexedMMapEmbedding(ckpt,1,n,dim,s)
            for ids in (np.array([-1]),np.array([n])):
                with self.assertRaises(IndexError): reader.read_rows(ids)
            with self.assertRaises(TypeError): reader.read_rows(np.array([0.5]))
            self.assertEqual(reader.read_rows(np.array([],dtype=np.int64))[0].shape,(0,dim//4))
            reader.close(); reader.close()

    def test_wrong_shape_shard_count_and_unsupported_quantization_fail(self):
        from jangh2_bridge.ple import IndexedMMapEmbedding
        with tempfile.TemporaryDirectory() as d:
            ckpt,_,n,dim,s=fixture(d)
            for args in ((n+1,dim,s),(n,dim*2,s),(n,dim,s+1)):
                with self.assertRaises(ValueError): IndexedMMapEmbedding(ckpt,1,*args)
            for spec in ckpt.config['quantization'].values():
                if isinstance(spec,dict): spec['mode']='jangtq2'
            with self.assertRaises(ValueError): IndexedMMapEmbedding(ckpt,1,n,dim,s)

    def test_missing_aux_tensor_and_alias_collision_fail(self):
        from jangh2_bridge.ple import IndexedMMapEmbedding
        with tempfile.TemporaryDirectory() as d:
            ckpt,_,n,dim,s=fixture(d)
            key=next(k for k in ckpt.weight_map if k.endswith('.biases'))
            filename=ckpt.weight_map.pop(key)
            with self.assertRaises(ValueError): IndexedMMapEmbedding(ckpt,1,n,dim,s)
            ckpt.weight_map[key]=filename
            ckpt.weight_map[canonical(key)]=filename
            with self.assertRaises(ValueError): IndexedMMapEmbedding(ckpt,1,n,dim,s)

    def test_constructor_interception_and_restoration(self):
        from jangh2_bridge.ple import stock_ple_context
        with tempfile.TemporaryDirectory() as d:
            ckpt,_,n,dim,s=fixture(d)
            lang=fake_language(); original=lang.ShardedEmbedding; init=lang.Qwen4ExpNGramEmbedding.__init__
            config=SimpleNamespace(num_embeddings=n,num_shards=s,ple_storage=None)
            with stock_ple_context(lang,ckpt) as state:
                obj=lang.Qwen4ExpNGramEmbedding(config,dim,1,0)
                self.assertEqual(state.created_layers,[1])
                self.assertEqual(obj.ngram_embedding.row_count,n)
            self.assertIs(lang.ShardedEmbedding,original)
            self.assertIs(lang.Qwen4ExpNGramEmbedding.__init__,init)
            # Reader lives with the constructed model after hooks are removed.
            obj.ngram_embedding.read_rows(np.array([0]))
            state.close()

    def test_restoration_and_reader_cleanup_after_error(self):
        from jangh2_bridge.ple import stock_ple_context
        with tempfile.TemporaryDirectory() as d:
            ckpt,_,n,dim,s=fixture(d)
            lang=fake_language(); original=lang.ShardedEmbedding; init=lang.Qwen4ExpNGramEmbedding.__init__
            config=SimpleNamespace(num_embeddings=n,num_shards=s,ple_storage=None)
            with self.assertRaisesRegex(RuntimeError,'simulated'):
                with stock_ple_context(lang,ckpt) as state:
                    obj=lang.Qwen4ExpNGramEmbedding(config,dim,1,0)
                    raise RuntimeError('simulated')
            self.assertIs(lang.ShardedEmbedding,original)
            self.assertIs(lang.Qwen4ExpNGramEmbedding.__init__,init)
            with self.assertRaises(RuntimeError): obj.ngram_embedding.read_rows(np.array([0]))

    def test_stock_unknown_external_manifest_fails_without_resident_fallback(self):
        from jangh2_bridge.ple import stock_ple_context
        with tempfile.TemporaryDirectory() as d:
            ckpt,_,n,dim,s=fixture(d); lang=fake_language()
            config=SimpleNamespace(num_embeddings=n,num_shards=s,ple_storage={'manifest':'other.json'})
            with self.assertRaises(ValueError):
                with stock_ple_context(lang,ckpt):
                    lang.Qwen4ExpNGramEmbedding(config,dim,1,0)

    def test_stock_signature_mismatch_rejected_before_construction(self):
        from jangh2_bridge.ple import stock_ple_context
        with tempfile.TemporaryDirectory() as d:
            ckpt,_,_,_,_=fixture(d); lang=fake_language()
            lang.Qwen4ExpNGramEmbedding.__init__=lambda self,unknown:None
            with self.assertRaises(RuntimeError):
                with stock_ple_context(lang,ckpt): pass

    def test_runtime_selector_uses_stock_when_vendor_hook_absent(self):
        from jangh2_bridge.loader import ple_runtime_context
        with tempfile.TemporaryDirectory() as d:
            ckpt,_,n,dim,s=fixture(d); lang=fake_language()
            config=SimpleNamespace(num_embeddings=n,num_shards=s,ple_storage=None)
            with ple_runtime_context(lang,ckpt) as state:
                obj=lang.Qwen4ExpNGramEmbedding(config,dim,1,0)
                self.assertEqual(state.kind,'stock-indexed-mmap')
            obj.ngram_embedding.close()


class RuntimeSafetyTests(unittest.TestCase):
    def test_vendor_api_preserved_and_restored(self):
        from jangh2_bridge.loader import ple_runtime_context
        class Disk:
            def __init__(self, model_path, prefix, num_embeddings, dims, num_shards):
                self.prefix=prefix
        lang=SimpleNamespace(DiskBackedShardedEmbedding=Disk,
                             _PLE_RUNTIME_MODE='resident',_PLE_RUNTIME_MODEL_PATH='original',
                             _MTP_RUNTIME=SimpleNamespace(enabled=True))
        def configure(root,mode):
            lang._PLE_RUNTIME_MODE=mode; lang._PLE_RUNTIME_MODEL_PATH=root
            return mode
        def mtp(root,enabled): lang._MTP_RUNTIME=SimpleNamespace(enabled=enabled)
        lang.configure_ple_runtime=configure; lang.configure_mtp_runtime=mtp
        lang.get_ple_runtime_mode=lambda:lang._PLE_RUNTIME_MODE
        original_init=Disk.__init__; previous_mtp=lang._MTP_RUNTIME
        with tempfile.TemporaryDirectory() as d:
            ckpt,_,n,dim,s=fixture(d)
            with ple_runtime_context(lang,ckpt) as state:
                self.assertEqual(state.kind,'vendor-mmap')
                self.assertEqual(lang._PLE_RUNTIME_MODE,'mmap')
                self.assertFalse(lang._MTP_RUNTIME.enabled)
                obj=Disk(d,'model.language_model.layers.1.ple.ple_embedding.ngram_embedding',n,dim,s)
                self.assertEqual(obj.prefix,'language_model.layers.1.ple.ngram_embedding')
            self.assertIs(Disk.__init__,original_init)
            self.assertEqual(lang._PLE_RUNTIME_MODE,'resident')
            self.assertEqual(lang._PLE_RUNTIME_MODEL_PATH,'original')
            self.assertIs(lang._MTP_RUNTIME,previous_mtp)

    def test_vendor_cannot_silently_return_resident(self):
        from jangh2_bridge.loader import ple_runtime_context
        with tempfile.TemporaryDirectory() as d:
            ckpt,_,_,_,_=fixture(d)
            lang=SimpleNamespace(configure_ple_runtime=lambda root,mode:'resident')
            with self.assertRaisesRegex(RuntimeError,'refused mmap'):
                with ple_runtime_context(lang,ckpt): pass

    def test_missing_all_storage_interfaces_fails_closed(self):
        from jangh2_bridge.loader import ple_runtime_context
        with tempfile.TemporaryDirectory() as d:
            ckpt,_,_,_,_=fixture(d)
            with self.assertRaisesRegex(RuntimeError,'Unrecognised Qwen4 PLE API'):
                with ple_runtime_context(SimpleNamespace(),ckpt): pass

    def test_other_thread_does_not_use_checkpoint_context(self):
        from jangh2_bridge.ple import stock_ple_context
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.TemporaryDirectory() as d:
            ckpt,_,n,dim,s=fixture(d); lang=fake_language()
            cfg=SimpleNamespace(num_embeddings=n,num_shards=s,ple_storage=None)
            with stock_ple_context(lang,ckpt) as state:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future=executor.submit(lang.Qwen4ExpNGramEmbedding,cfg,dim,1,0)
                    # Original fixture constructor intentionally raises. No bridge
                    # mapping must be installed in the unrelated worker context.
                    with self.assertRaisesRegex(AssertionError,'Resident PLE'):
                        future.result()
                self.assertEqual(state.created_layers,[])
                lang.Qwen4ExpNGramEmbedding(cfg,dim,1,0)
                state.validate_complete()
            state.close()

    def test_incomplete_layer_coverage_rejected(self):
        from jangh2_bridge.ple import stock_ple_context
        with tempfile.TemporaryDirectory() as d:
            ckpt,_,_,_,_=fixture(d); lang=fake_language()
            with self.assertRaisesRegex(RuntimeError,'Not all PLE'):
                with stock_ple_context(lang,ckpt) as state:
                    state.validate_complete()

    def test_real_mmap_lookup_through_numpy_backend_double(self):
        from jangh2_bridge.ple import IndexedMMapEmbedding
        import sys
        fake=ModuleType('mlx.core'); calls=[]
        for name in ('array','uint32','int64','float16','float32','zeros'):
            setattr(fake,name,getattr(np,name))
        fake.bfloat16=np.float32
        def dequantize(packed,scales,biases,*,group_size,bits,mode):
            calls.append((packed.shape,group_size,bits,mode))
            dims=packed.shape[-1]*32//bits
            q=np.stack([(packed[:,j*bits//32]>>((j*bits)%32))&((1<<bits)-1) for j in range(dims)],axis=1)
            return (q*np.repeat(scales,group_size,axis=1)+np.repeat(biases,group_size,axis=1)).astype(scales.dtype)
        fake.dequantize=dequantize
        with tempfile.TemporaryDirectory() as d:
            ckpt,expected,n,dim,s=fixture(d)
            reader=IndexedMMapEmbedding(ckpt,1,n,dim,s)
            with patch.dict(sys.modules,{'mlx':ModuleType('mlx'),'mlx.core':fake}):
                ids=np.array([[4,0,4],[2,3,2]],np.int64)
                out=reader(ids)
                np.testing.assert_array_equal(out,expected[ids])
                self.assertEqual(calls,[((4,8),32,8,'affine')])
                self.assertEqual(reader.rows_read,4)  # deduplication is intentional
                self.assertEqual(reader(np.empty((0,2),np.int64)).shape,(0,2,dim))
                np.testing.assert_array_equal(reader(np.array(0,np.int64)),expected[0])
            reader.close()

if __name__=='__main__': unittest.main()
