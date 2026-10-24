import unittest
import numpy as np
from jangh2_bridge.grouped_ops import schedule, tile_ranges, build_grouped_gate, build_grouped_down
from test_metal_backend import make_spec

class GroupedTests(unittest.TestCase):
    def test_every_route_once_even_empty_experts_and_tails(self):
        for experts in (1,4,512):
            for count in (0,1,3,7,32,1024):
                for block in (2,4):
                    rng=np.random.default_rng(count)
                    ids=rng.integers(0,experts,count,dtype=np.uint32)
                    s=schedule(ids,experts,block,np)
                    covered=[]
                    for e,start,end in tile_ranges(s,block):
                        self.assertTrue(np.all(s.ids[start:end]==e))
                        self.assertLessEqual(end-start,block)
                        covered.extend(range(start,end))
                    self.assertEqual(covered,list(range(count)))
                    self.assertEqual(len(set(s.order.tolist())),count)
                    np.testing.assert_array_equal(s.ids[s.inverse],ids)
                    self.assertLessEqual(len(list(tile_ranges(s,block))),s.max_tiles)
    def test_invalid_signed_and_large_ids_live_in_safe_sentinel_bucket(self):
        for ids in [np.array([-1,0,2,2**32],np.int64),np.array([2**32,0,2],np.uint64)]:
            s=schedule(ids,3,4,np)
            expect=np.where((ids>=0)&(ids<3),ids,3).astype(np.uint32)
            np.testing.assert_array_equal(s.ids[s.inverse],expect)
    def test_duplicate_routes_restore_original_order(self):
        ids=np.array([2,2,1,2,0,0,1,2],np.uint32)
        s=schedule(ids,4,4,np)
        np.testing.assert_array_equal(s.order[s.inverse],np.arange(len(ids)))
    def test_reject_bad_contract(self):
        for args in [(np.array([0.5]),4,4),(np.array([0]),0,4),(np.array([0]),4,3)]:
            with self.assertRaises((TypeError,ValueError)):schedule(*args,np)
    def test_kernel_has_shared_weight_inner_token_loop_and_uniform_exit(self):
        for bits in ((2,2,2),(2,3,4)):
            for block in (2,4):
                s=make_spec(bits=bits,d=96,h=64,e=5)
                for k in (build_grouped_gate(s,3,block),build_grouped_down(s,block)):
                    self.assertIn('tile_ends',k.inputs)
                    self.assertIn('for (uint b = 0;',k.source)
                    self.assertIn('jangh_read',k.source)
                    self.assertIn('grouped',k.name)
                    self.assertNotIn('atomic_',k.source)
    def test_grouped_forward_contains_no_host_route_reads(self):
        import ast,inspect
        from jangh2_bridge.grouped_ops import forward
        tree=ast.parse(inspect.getsource(forward))
        banned={'eval','item','tolist','synchronize','asarray'}
        self.assertFalse([n for n in ast.walk(tree) if isinstance(n,ast.Attribute) and n.attr in banned])
