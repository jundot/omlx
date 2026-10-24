import unittest
import numpy as np
from jangh2_bridge.row_cache import PackedRowCache

class BulkRowTests(unittest.TestCase):
    def test_bulk_matches_scalar_and_copies_out(self):
        c=PackedRowCache(4,2,1024)
        for i in range(6):c.put(i,np.full(4,i,np.uint32),np.full(2,i+.5),np.full(2,-i))
        pos,values,miss=c.get_many(np.array([4,1,99,4],np.int64))
        np.testing.assert_array_equal(pos,[0,1,3]);np.testing.assert_array_equal(miss,[2])
        np.testing.assert_array_equal(values[0][:,0],[4,1,4])
        values[0].fill(900)
        self.assertEqual(c.get(4)[0][0],4)
        self.assertEqual(c.stats()['row_cache_hits'],4)
    def test_bulk_preserves_lru_access_order(self):
        c=PackedRowCache(1,1,36)
        for i in range(3):c.put(i,[i],[i],[i])
        c.get_many(np.array([0,1,0]))
        c.put(3,[3],[3],[3])
        self.assertIsNone(c.get(2))
        self.assertIsNotNone(c.get(0))
    def test_empty_and_all_misses(self):
        c=PackedRowCache(4,2,0)
        for ids in (np.array([],np.int64),np.array([1,2],np.int64)):
            p,v,m=c.get_many(ids)
            self.assertEqual(len(p),0);self.assertEqual(len(m),len(ids))
            self.assertEqual(v[0].shape,(0,4))
    def test_bulk_validates_integer_vector(self):
        c=PackedRowCache(1,1,12)
        for ids in (np.array([1.5]),np.array([[1]])):
            with self.assertRaises((TypeError,ValueError)):c.get_many(ids)
    def test_bulk_zero_slot_is_hit(self):
        c=PackedRowCache(1,1,12);c.put(7,[4],[2],[1])
        p,v,m=c.get_many(np.array([7]))
        self.assertEqual(p.tolist(),[0]);self.assertEqual(m.tolist(),[])
        self.assertEqual(v[0][0,0],4)

if __name__=='__main__':unittest.main()
