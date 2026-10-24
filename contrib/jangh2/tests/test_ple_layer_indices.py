"""One-based config IDs versus zero-based decoder/tensor indices.

The reference boundary is MLX-VLM v0.7.8 Qwen4ExpDecoderLayer:
  ple_index = config.ple_layer_ids.index(layer_idx + 1)
  self.ple = Qwen4ExpPLELayer(config, layer_idx, ple_index)
See docs/PLE-v3-PLAN.md and verification/SOURCES-v3.md.

CPU tests exercise real indexed safetensors readers. MLX is doubled only in
one check_ple command test; that test is not a Metal numerical validation.
"""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from test_ple_storage import fixture, fake_language
from jangh2_bridge.checkpoint import Checkpoint
from jangh2_bridge.ple import stock_ple_context


def multi_fixture(root, layer_indices):
    root = Path(root)
    weights, quant, expected = {}, {'bits':4, 'group_size':32, 'mode':'affine'}, {}
    for layer_idx in layer_indices:
        sub = root / f'layer-{layer_idx}'
        sub.mkdir()
        ckpt, dense, n, dims, shards = fixture(sub, layer_idx=layer_idx,
                                               code_offset=layer_idx * 11)
        weights.update({key: f'{sub.name}/{value}' for key, value in ckpt.weight_map.items()})
        quant.update(ckpt.config['quantization'])
        expected[layer_idx] = dense
    config = {'model_type':'qwen4_exp', 'text_config':{
        'ple_layer_ids':[i+1 for i in layer_indices], 'num_hidden_layers':8},
        'quantization':quant}
    return Checkpoint(root,config,weights,{},'fixture'), expected, n,dims,shards


def args_config(n, shards):
    return SimpleNamespace(num_embeddings=n, num_shards=shards, ple_storage=None)


class LayerIndexTests(unittest.TestCase):
    def test_one_based_ids_map_once_and_preserve_order(self):
        from jangh2_bridge.metadata import ple_layer_indices
        self.assertEqual(ple_layer_indices([2]), (1,))
        self.assertEqual(ple_layer_indices([1,2,4], 4), (0,1,3))
        self.assertEqual(ple_layer_indices([4,2], 4), (3,1))
        self.assertEqual(ple_layer_indices((2,)), (1,))

    def test_invalid_or_duplicate_config_ids_rejected(self):
        from jangh2_bridge.metadata import ple_layer_indices
        for ids in ([], None, '2', [0], [-1], [True], [1.0], [2,'3'], [2,2]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                ple_layer_indices(ids)
        with self.assertRaises(ValueError):
            ple_layer_indices([5], 4)
        for count in (0, -1, True, 4.0):
            with self.subTest(count=count), self.assertRaises(ValueError):
                ple_layer_indices([2], count)

    def test_config_id_two_uses_real_tensor_layer_one(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt, dense, n,dims,shards = fixture(d)
            lang = fake_language()
            with stock_ple_context(lang,ckpt) as state:
                obj = lang.Qwen4ExpNGramEmbedding(args_config(n,shards),dims,1,0)
                self.assertEqual(state.expected_layers,[1])
                self.assertEqual(state.created_layers,[1])
                self.assertEqual(obj.layer_idx,1)
                self.assertEqual(obj.ple_layer_index,0)
                state.validate_complete()
            packed,scales,biases = obj.ngram_embedding.read_rows([0])
            decoded = np.stack([(packed[:,j//4] >> (8*(j%4))) & 255
                                for j in range(dims)], axis=1)
            np.testing.assert_array_equal(decoded*scales+biases, dense[[0]])
            state.close()

    def test_config_id_one_uses_tensor_layer_zero(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt,_,n,dims,s = fixture(d,layer_idx=0)
            lang=fake_language()
            with stock_ple_context(lang,ckpt) as state:
                obj=lang.Qwen4ExpNGramEmbedding(args_config(n,s),dims,0,0)
                self.assertEqual(obj.layer_idx,0)
                state.validate_complete()
            state.close()

    def test_multiple_non_sorted_layers_keep_their_original_ordinals(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt,_,n,dims,s=multi_fixture(d,[3,1])
            lang=fake_language()
            with stock_ple_context(lang,ckpt) as state:
                # Model construction goes in decoder order, not config order.
                first=lang.Qwen4ExpNGramEmbedding(args_config(n,s),dims,1,1)
                second=lang.Qwen4ExpNGramEmbedding(args_config(n,s),dims,3,0)
                self.assertEqual((first.layer_idx, first.ple_layer_index),(1,1))
                self.assertEqual((second.layer_idx, second.ple_layer_index),(3,0))
                self.assertEqual(state.created_layers,[1,3])
                state.validate_complete()
            state.close()

    def test_adjacent_layer_with_wrong_ordinal_is_not_silently_used(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt,_,n,dims,s=multi_fixture(d,[1,2])
            lang=fake_language()
            with stock_ple_context(lang,ckpt) as state:
                # Layer 2 exists, but belongs to ordinal 1, not ordinal 0.
                with self.assertRaisesRegex(ValueError,'ordinal'):
                    lang.Qwen4ExpNGramEmbedding(args_config(n,s),dims,2,0)
                self.assertEqual(state.created_layers,[])
            state.close()

    def test_wrong_one_based_argument_is_rejected_before_reader_creation(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt,_,n,dims,s=fixture(d)
            lang=fake_language()
            with stock_ple_context(lang,ckpt) as state:
                with self.assertRaisesRegex(ValueError,'zero-based'):
                    lang.Qwen4ExpNGramEmbedding(args_config(n,s),dims,2,0)
                self.assertEqual(state.created_layers,[])
            state.close()

    def test_boolean_constructor_indices_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt,_,n,dims,s=fixture(d)
            lang=fake_language()
            with stock_ple_context(lang,ckpt) as state:
                for layer,ordinal in ((True,0),(1,False)):
                    with self.subTest(layer=layer,ordinal=ordinal), self.assertRaises(ValueError):
                        lang.Qwen4ExpNGramEmbedding(args_config(n,s),dims,layer,ordinal)
            state.close()

    def test_duplicate_config_ids_fail_before_patching(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt,_,n,dims,s=fixture(d)
            ckpt.config['text_config']['ple_layer_ids']=[2,2]
            lang=fake_language(); original=lang.ShardedEmbedding
            with self.assertRaises(ValueError):
                with stock_ple_context(lang,ckpt):
                    pass
            self.assertIs(lang.ShardedEmbedding,original)

    def test_preflight_main_uses_zero_based_layer_and_real_reader(self):
        import check_ple
        import jangh2_bridge.ple_selftest
        with tempfile.TemporaryDirectory() as d:
            ckpt,_,n,dims,s=fixture(d)
            lang=fake_language()
            cfg=args_config(n,s)
            cfg.ple_layer_ids=[2]; cfg.num_hidden_layers=4
            cfg.ple_embed_dim=dims; cfg.split_ngram_parts=s
            architecture=ModuleType('mlx_vlm.models.qwen4_exp')
            architecture.ModelConfig=SimpleNamespace(from_dict=lambda data:SimpleNamespace(text_config=cfg))
            utils=ModuleType('mlx_vlm.utils')
            utils.load_config=lambda root:dict(ckpt.config)
            utils.update_module_configs=lambda config,*unused:config
            core=ModuleType('mlx.core')
            for name in ('array','uint32','int64','float16','float32','zeros','all','isfinite'):
                setattr(core,name,getattr(np,name))
            core.bfloat16=np.float32
            core.eval=lambda *args:None
            def dequantize(packed,scales,biases,*,group_size,bits,mode):
                cols=packed.shape[-1]*32//bits
                codes=np.stack([(packed[:,j*bits//32] >> ((j*bits)%32)) & ((1<<bits)-1)
                                for j in range(cols)],axis=1)
                return (codes*np.repeat(scales,group_size,axis=1)+
                        np.repeat(biases,group_size,axis=1)).astype(scales.dtype)
            core.dequantize=dequantize
            mlx=ModuleType('mlx'); mlx.core=core
            vlm=ModuleType('mlx_vlm'); vlm.utils=utils
            modules={'mlx':mlx,'mlx.core':core,'mlx_vlm':vlm,'mlx_vlm.utils':utils,
                     'mlx_vlm.models.qwen4_exp':architecture,
                     'mlx_vlm.models.qwen4_exp.language':lang}
            output=io.StringIO()
            with patch.dict(sys.modules,modules), \
                 patch('jangh2_bridge.checkpoint.inspect_checkpoint',return_value=ckpt), \
                 patch('jangh2_bridge.ple_selftest.run'), \
                 patch.object(sys,'argv',['check_ple.py','--model',d]), \
                 contextlib.redirect_stdout(output):
                check_ple.main()
            self.assertIn('PLE preflight passed.',output.getvalue())
            self.assertIn('layers.1',output.getvalue())


if __name__=='__main__':
    unittest.main()
