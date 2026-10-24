import threading
import unittest
import numpy as np
from jangh2_bridge.device_rows import DeviceRowCache, capacity_for, row_slots

class XP:
    def __getattr__(self,key):return getattr(np,key)
    def eval(self,*args):pass

class DeviceRowsTests(unittest.TestCase):
    def setUp(self):
        self.xp=XP(); self.calls=[]
        self.table=(np.arange(64*4).reshape(64,4)*.01).astype(np.float16)
        self.c=DeviceRowCache(self.xp,4,np.float16,64,byte_limit=8*(8+8))
    def fetch(self,ids):
        self.calls.append(ids.copy());return self.table[ids].copy()
    def test_repeated_rows_are_hits_no_second_fallback(self):
        ids=np.array([[2,1,2]],np.int64)
        first=self.c.lookup(ids,self.fetch);second=self.c.lookup(ids,self.fetch)
        np.testing.assert_array_equal(first,self.table[ids]);np.testing.assert_array_equal(second,first)
        self.assertEqual(len(self.calls),1);self.assertEqual(self.c.stats()['device_all_hit_calls'],1)
    def test_misses_fetch_only_uncached_unique_rows(self):
        self.c.lookup(np.array([1,2],np.int64),self.fetch)
        out=self.c.lookup(np.array([2,3,3],np.int64),self.fetch)
        np.testing.assert_array_equal(self.calls[-1],[3]);np.testing.assert_array_equal(out,self.table[[2,3,3]])
    def test_collisions_do_not_return_another_rows_values(self):
        ids=np.arange(40,dtype=np.int64)
        old=self.c.lookup(ids,self.fetch)
        for i in range(0,40,5):
            x=ids[i:i+5];np.testing.assert_array_equal(self.c.lookup(x,self.fetch),self.table[x])
        np.testing.assert_array_equal(old,self.table[ids])
        self.assertLessEqual(self.c.stats()['device_payload_bytes'],128)
    def test_shape_dtype_empty_close_and_invalid(self):
        out=self.c.lookup(np.empty((1,0),np.int64),self.fetch)
        self.assertEqual(out.shape,(1,0,4));self.assertEqual(out.dtype,np.float16)
        for ids in (np.array([-1]),np.array([64]),np.array([1.2])):
            with self.assertRaises((IndexError,TypeError)): self.c.lookup(ids,self.fetch)
        self.c.close()
        with self.assertRaises(RuntimeError):self.c.lookup(np.array([1]),self.fetch)
    def test_owner_thread_enforced(self):
        self.c.lookup(np.array([1]),self.fetch);errors=[]
        def run():
            try:self.c.lookup(np.array([1]),self.fetch)
            except Exception as exc:errors.append(exc)
        t=threading.Thread(target=run);t.start();t.join()
        self.assertIsInstance(errors[0],RuntimeError)
    def test_budget_and_hash_contract(self):
        for budget in (1,127,128,1024):
            n=capacity_for(4,2,budget)
            self.assertLessEqual(n*16,budget)
            if n:self.assertEqual(n&(n-1),0)
        ids=np.array([0,2**32,2**40,123],np.int64)
        slots=row_slots(ids,8,np)
        self.assertTrue(np.all((slots>=0)&(slots<8)))

    def test_reject_unsupported_cache_dtype_and_nonpositive_vocabulary(self):
        with self.assertRaises(TypeError):DeviceRowCache(self.xp,4,np.float64,64,byte_limit=128)
        with self.assertRaises(ValueError):DeviceRowCache(self.xp,4,np.float16,0,byte_limit=128)
