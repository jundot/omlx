"""CPU contracts only. Actual Metal compile/arithmetic is check_performance.py."""
import ast
import inspect
import unittest
import numpy as np
from jangh2_bridge.metal_ops import KernelSpec, build_gate_up, build_down, route_plan
from jangh2_bridge.metadata import CUBIC


def make_spec(bits=(2,2,2), rotations=('hadamard32',)*3, d=2560, h=640, e=16,
              gate_rows=4, down_rows=4):
    levels=[]
    for b in bits:
        a,c=CUBIC.get(b,(1.0,0.0)); u=np.arange(1<<b,dtype=np.float64)-((1<<b)-1)/2
        levels.append(tuple((u*(a+c*u*u)).astype(np.float32).tolist()))
    return KernelSpec(d,h,e,bits,rotations,tuple(levels),gate_rows,down_rows)

class MetalPlanTests(unittest.TestCase):
    def test_actual_allosaurus_shapes_and_small_tail(self):
        s=make_spec()
        p=route_plan((1,1,2560),(1,1,8),s)
        self.assertEqual(p.routes,8)
        self.assertEqual(p.output_shape,(1,1,8,2560))
        self.assertFalse(p.sorted)
        self.assertEqual(p.gate_grid,(20*256,8,1))
        self.assertEqual(p.down_grid,(320*64,8,1))
        self.assertEqual(p.intermediate_elements,8*640)
        # Temporary storage scales with routes * hidden, not routes * D * hidden.
        self.assertLess(p.intermediate_elements*4, 100_000)

    def test_prefill_batch_and_sorting(self):
        p=route_plan((2,17,2560),(2,17,8),make_spec())
        self.assertEqual(p.routes,272)
        self.assertTrue(p.sorted)
        self.assertEqual(p.output_shape,(2,17,8,2560))

    def test_empty_shape_and_invalid_contracts(self):
        s=make_spec()
        self.assertEqual(route_plan((0,2560),(0,8),s).routes,0)
        for xs,ids in [((1,2560),(2,8)),((1,128),(1,8)),((2560,),(8,)),((1,2560),(1,0))]:
            with self.subTest(xs=xs,ids=ids),self.assertRaises(ValueError): route_plan(xs,ids,s)
        for kw in ({'bits':(1,2,2)}, {'rotations':('none','random','none')},
                   {'d':65}, {'h':0}, {'e':0}, {'gate_rows':3}, {'down_rows':3}):
            with self.subTest(kw=kw),self.assertRaises(ValueError): make_spec(**kw)

    def test_mixed_bits_and_independent_rotations(self):
        s=make_spec(bits=(2,3,4),rotations=('none','hadamard32','none'),d=96,h=64)
        g=build_gate_up(s,top_k=3,sorted_routes=False)
        d=build_down(s)
        self.assertIn('jangh_read<2>',g.source)
        self.assertIn('jangh_read<3>',g.source)
        self.assertIn('jangh_read<4>',d.source)
        self.assertNotIn('threadgroup_barrier',g.source)

    def test_fused_rotated_hidden_uses_uniform_barrier_and_bounds(self):
        source=build_gate_up(make_spec(),8,False).source
        self.assertIn('threadgroup_barrier',source)
        self.assertIn('simd_shuffle_xor',source)
        self.assertIn('expert >=',source)
        self.assertIn('NAN',source)
        self.assertNotIn('return;',source.split('threadgroup_barrier')[0])

    def test_precision_and_codebooks_are_not_requantized(self):
        for b in (2,3,4):
            kernel=build_gate_up(make_spec(bits=(b,b,b)),8,False)
            self.assertIn('float accg',kernel.source)
            self.assertIn('simd_shuffle_down',kernel.source)
            self.assertIn('jangh_level_g',kernel.header)
            self.assertNotIn('affine',kernel.source)

    def test_no_cpu_sync_in_metal_forward(self):
        from jangh2_bridge import metal_ops
        tree=ast.parse(inspect.getsource(metal_ops.forward))
        forbidden={'eval','async_eval','synchronize','item','tolist','numpy','array'}
        calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertFalse(set(calls)&forbidden, calls)

    def test_kernel_factories_do_not_cache_tensors_across_threads(self):
        from jangh2_bridge.metal_ops import _kernel_for
        from types import SimpleNamespace
        from concurrent.futures import ThreadPoolExecutor
        made=[]
        def factory(**kw):
            obj=object(); made.append(obj); return obj
        xp=SimpleNamespace(fast=SimpleNamespace(metal_kernel=factory))
        definition=build_down(make_spec())
        first=_kernel_for(xp,definition)
        self.assertIs(_kernel_for(xp,definition),first)
        with ThreadPoolExecutor(max_workers=1) as worker:
            second=worker.submit(_kernel_for,xp,definition).result()
            self.assertIs(worker.submit(_kernel_for,xp,definition).result(),second)
        self.assertIsNot(first,second)
        self.assertEqual(len(made),2)

if __name__=='__main__': unittest.main()
