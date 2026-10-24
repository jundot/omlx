import unittest
from types import SimpleNamespace
import tempfile
from pathlib import Path
import numpy as np

class LoaderUtilityTests(unittest.TestCase):
    def test_ple_constructor_patch_restores_after_error(self):
        from jangh2_bridge.loader import ple_prefix_context
        seen=[]
        class Disk:
            def __init__(self,model_path,prefix,num_embeddings,dims,num_shards):
                seen.append(prefix)
        language=SimpleNamespace(DiskBackedShardedEmbedding=Disk)
        original=Disk.__init__
        requested='model.language_model.layers.1.ple.ple_embedding.ngram_embedding'
        actual='language_model.layers.1.ple.ngram_embedding'
        with self.assertRaises(RuntimeError):
            with ple_prefix_context(language,{actual+'.shards.0.weight':'a.safetensors'}):
                Disk(Path('/x'),requested,128,32,4)
                raise RuntimeError('constructor failed')
        self.assertEqual(seen,[actual])
        self.assertIs(Disk.__init__,original)
    def test_strict_parameter_coverage(self):
        from jangh2_bridge.loader import compare_parameters
        w={'a':np.zeros((3,4))}
        compare_parameters(w,dict(w))
        with self.assertRaises(ValueError): compare_parameters(w,{})
        with self.assertRaises(ValueError): compare_parameters(w,{'a':np.zeros((4,3))})
        with self.assertRaises(ValueError): compare_parameters(w,{'a':w['a'],'b':w['a']})
    def test_no_numeric_change_in_non_norm_parameters(self):
        from jangh2_bridge.loader import recenter_residual_norms
        class Residual: pass
        class Ordinary: pass
        model=SimpleNamespace(named_modules=lambda:[('a',Residual()),('b',Ordinary())])
        w={'a.weight':np.array([1.,.25,1.5],np.float32), 'b.weight':np.array([1.],np.float32)}
        unchanged=w['b.weight']
        recenter_residual_norms(model,w,Residual,np)
        np.testing.assert_array_equal(w['a.weight'],[0.,-.75,.5])
        self.assertIs(w['b.weight'],unchanged)
    def test_vendor_global_state_restored(self):
        from jangh2_bridge.loader import vendor_runtime_context
        language=SimpleNamespace(_PLE_RUNTIME_MODEL_PATH='old',_PLE_RUNTIME_MODE='resident',_MTP_RUNTIME=False)
        with self.assertRaises(RuntimeError):
            with vendor_runtime_context(language):
                language._PLE_RUNTIME_MODEL_PATH='new'; language._PLE_RUNTIME_MODE='mmap'; language._MTP_RUNTIME=True
                raise RuntimeError('failed load')
        self.assertEqual(vars(language),{'_PLE_RUNTIME_MODEL_PATH':'old','_PLE_RUNTIME_MODE':'resident','_MTP_RUNTIME':False})
