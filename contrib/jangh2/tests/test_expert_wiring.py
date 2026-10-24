"""Boundary-contract tests using small NumPy-backed doubles, NOT GPU tests.

Real MLX dispatch and numerical parity are covered separately by test_on_mac.py.
"""
import importlib
import sys
from types import ModuleType
import unittest
from unittest.mock import patch
import numpy as np
from test_bridge import config

class Module:
    def named_modules(self):
        def walk(name,value):
            if isinstance(value,Module):
                yield name,value
                for key,item in vars(value).items():
                    if not key.startswith('_'):
                        yield from walk((name+'.' if name else '')+key,item)
            elif isinstance(value,list):
                for i,item in enumerate(value): yield from walk(name+'.'+str(i),item)
        return list(walk('',self))

class SwiGLU(Module): pass
class Linear(Module):
    def __init__(self,d,n,e): self.weight=np.zeros((e,n,d),np.float32)
    def to_quantized(self,group_size=64,bits=4,mode='affine'):
        self.quantization=(group_size,bits,mode)
        return self
class VlmSwitch(Module):
    def __init__(self,d,h,e):
        self.gate_proj=Linear(d,h,e); self.up_proj=Linear(d,h,e); self.down_proj=Linear(h,d,e)
        self.activation=SwiGLU()
class LmSwitch(VlmSwitch): pass

def make_model(cls=VlmSwitch):
    m=Module(); m.language_model=Module(); m.language_model.model=Module()
    layer=Module(); layer.mlp=Module(); layer.mlp.switch_mlp=cls(64,32,4)
    m.language_model.model.layers=[layer]
    return m

def quantize(model,group_size,bits,mode,class_predicate):
    for path,module in model.named_modules():
        if hasattr(module,'to_quantized'):
            params=class_predicate(path,module)
            if isinstance(params,dict): module.to_quantized(**params)
            elif params: module.to_quantized(group_size,bits,mode)

class ExpertWiringTests(unittest.TestCase):
    def setUp(self):
        fake={name:ModuleType(name) for name in ('mlx','mlx.core','mlx.nn','mlx_vlm','mlx_vlm.models',
             'mlx_vlm.models.switch_layers','mlx_lm','mlx_lm.models','mlx_lm.models.switch_layers')}
        for name in dir(np):
            if not name.startswith('__'): setattr(fake['mlx.core'],name,getattr(np,name))
        fake['mlx.nn'].Module=Module; fake['mlx.nn'].quantize=quantize
        fake['mlx_vlm.models.switch_layers'].SwitchGLU=VlmSwitch
        fake['mlx_lm.models.switch_layers'].SwitchGLU=LmSwitch
        self.patcher=patch.dict(sys.modules,fake); self.patcher.start()
        sys.modules.pop('jangh2_bridge.experts',None)
        self.ex=importlib.import_module('jangh2_bridge.experts')
    def tearDown(self):
        sys.modules.pop('jangh2_bridge.experts',None)
        self.patcher.stop()
    def test_original_signature_reproduces_rotation_error(self):
        with self.assertRaisesRegex(TypeError,'rotation'):
            Linear(64,32,4).to_quantized(mode='jangtq2',bits=2,rotation='hadamard32')
    def test_optimized_reference_switch_preserves_packed_layout(self):
        m = make_model()
        replaced = self.ex.replace_experts(
            m, config(), backend='reference-fast', chunk_size=8)
        self.assertEqual(len(replaced), 1)
        switch = m.language_model.model.layers[0].mlp.switch_mlp
        self.assertIsInstance(switch, self.ex.ReferenceSwitchGLU)
        self.assertTrue(switch.fast)
        self.assertEqual(switch.chunk_size, 8)
        for projection in ('gate_proj', 'up_proj', 'down_proj'):
            part = getattr(switch, projection)
            self.assertEqual(part.rotation, 'hadamard32')
            self.assertEqual(part.bits, 2)

    def test_metal_backend_uses_identical_strict_parameter_names(self):
        m=make_model()
        self.ex.replace_experts(m,config(),backend='metal')
        switch=m.language_model.model.layers[0].mlp.switch_mlp
        self.assertIsInstance(switch,self.ex.MetalSwitchGLU)
        self.assertTrue(switch.is_jangtq2)
        self.ex.quantize_affine(m,config(),{})
        for projection in ('gate_proj','up_proj','down_proj'):
            part=getattr(switch,projection)
            self.assertEqual(part.bits,2)
            self.assertEqual(part.tq2_packed.dtype,np.uint32)
            self.assertEqual(part.tq2_scales.dtype,np.float16)

    def test_grouped_geometry_is_independent_of_tuned_decode_geometry(self):
        import os
        with patch.dict(os.environ, {'OMLX_JANGH_PREFILL':'grouped',
                'OMLX_JANGH_PREFILL_MIN_TOKENS':'2','OMLX_JANGH_METAL_GATE_ROWS':'8',
                'OMLX_JANGH_METAL_DOWN_ROWS':'2'}):
            m=make_model();self.ex.replace_experts(m,config(),backend='metal')
        switch=m.language_model.model.layers[0].mlp.switch_mlp
        with patch('jangh2_bridge.grouped_ops.forward',return_value='grouped') as grouped, \
             patch('jangh2_bridge.metal_ops.forward',return_value='decode') as decode:
            self.assertEqual(switch(np.zeros((1,4,64)),np.zeros((1,4,3),np.uint32)), 'grouped')
            spec=grouped.call_args.args[3]
            self.assertEqual((spec.gate_rows,spec.down_rows),(4,4))
            self.assertEqual(switch(np.zeros((1,1,64)),np.zeros((1,1,3),np.uint32)), 'decode')
            spec=decode.call_args.args[3]
            self.assertEqual((spec.gate_rows,spec.down_rows),(8,2))

    def test_vlm_and_lm_switches_replaced(self):
        for cls in (VlmSwitch,LmSwitch):
            m=make_model(cls)
            replaced=self.ex.replace_experts(m,config(),backend='reference')
            self.assertEqual(len(replaced),1)
            sw=m.language_model.model.layers[0].mlp.switch_mlp
            self.assertIsInstance(sw,self.ex.ReferenceSwitchGLU)
            for proj in ('gate_proj','up_proj','down_proj'):
                layer=getattr(sw,proj)
                self.assertIs(layer.to_quantized(mode='jangtq2',bits=2,rotation='hadamard32'),layer)
                with self.assertRaises(ValueError): layer.to_quantized(mode='affine',bits=2)
    def test_quantizer_skips_already_installed_jangh(self):
        m=make_model(); self.ex.replace_experts(m,config(),backend='reference')
        self.ex.quantize_affine(m,config(),{})  # no TypeError; no affine replacement
    def test_unreplaced_projection_rejected(self):
        with self.assertRaisesRegex(ValueError,'not replaced'):
            self.ex.quantize_affine(make_model(),config(),{})
    def test_missing_target_rejected_without_mutation(self):
        m=make_model(); original=m.language_model.model.layers[0].mlp.switch_mlp
        c=config()
        for key,value in list(c['quantization'].items()):
            if isinstance(value,dict): c['quantization'][key.replace('layers.0','layers.1')]=value
        with self.assertRaises(ValueError): self.ex.replace_experts(m,c,backend='reference')
        self.assertIs(m.language_model.model.layers[0].mlp.switch_mlp,original)
    def test_expert_bias_rejected(self):
        m=make_model(); m.language_model.model.layers[0].mlp.switch_mlp.gate_proj.bias=np.zeros((4,32))
        with self.assertRaisesRegex(ValueError,'bias'): self.ex.replace_experts(m,config(),backend='reference')
    def test_affine_bits_preserved(self):
        m=make_model(); m.language_model.model.embed_tokens=Linear(64,32,1)
        c=config(); c['quantization']['language_model.embed_tokens']={'bits':4,'group_size':32}
        self.ex.replace_experts(m,c,backend='reference')
        self.ex.quantize_affine(m,c,{'language_model.model.embed_tokens.scales':np.ones((32,2))})
        self.assertEqual(m.language_model.model.embed_tokens.quantization,(32,4,'affine'))
    def test_unknown_scale_path_rejected(self):
        m=make_model(); self.ex.replace_experts(m,config(),backend='reference')
        with self.assertRaises(ValueError): self.ex.quantize_affine(m,config(),{'unknown.scales':np.ones(1)})
