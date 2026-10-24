"""Read real small safetensors files; ensure cache optimisations preserve values."""
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from test_ple_storage import fixture
from jangh2_bridge.ple import IndexedMMapEmbedding

class PLEPerformanceTests(unittest.TestCase):
    def test_repeated_rows_hit_without_new_disk_reads(self):
        with tempfile.TemporaryDirectory() as d:
            ck,_,n,w,s=fixture(d)
            r=IndexedMMapEmbedding(ck,1,n,w,s,row_cache_bytes=1024,max_open_files=16)
            with patch.object(r._mapping_cache,'read_rows',wraps=r._mapping_cache.read_rows) as read:
                a=r.read_rows(np.array([4,0,3,2,4]))
                first=read.call_count
                b=r.read_rows(np.array([0,4,0]))
                self.assertEqual(read.call_count,first)
                for x,y in zip(a,b): np.testing.assert_array_equal(y,x[[1,0,1]])
            self.assertGreater(r.mapping_stats()['row_cache_hits'],0)
            self.assertLessEqual(r.mapping_stats()['row_cache_payload_bytes'],1024)
            r.close()

    def test_eviction_preserves_rows_and_closed_state(self):
        with tempfile.TemporaryDirectory() as d:
            ck,_,n,w,s=fixture(d)
            r=IndexedMMapEmbedding(ck,1,n,w,s,row_cache_bytes=80,max_open_files=1)
            old=IndexedMMapEmbedding(ck,1,n,w,s)
            for ids in ([4,0,1,4,2],[1,0,3],[2,4,0,4],[0]):
                a=r.read_rows(np.array(ids)); b=old.read_rows(np.array(ids))
                for x,y in zip(a,b): np.testing.assert_array_equal(x,y)
            self.assertGreater(r.mapping_stats()['row_cache_evictions'],0)
            self.assertLessEqual(r.mapping_stats()['row_cache_payload_bytes'],80)
            self.assertLessEqual(r.mapping_stats()['peak_open_files'],1)
            r.close();old.close()
            self.assertEqual(r.mapping_stats()['row_cache_payload_bytes'],0)
            with self.assertRaises(RuntimeError): r.read_rows(np.array([0]))

    def test_cache_preserves_aux_all_formats_and_is_copy_out(self):
        for dtype in ('F16','BF16','F32'):
            for bits in (4,8):
                with self.subTest(dtype=dtype,bits=bits),tempfile.TemporaryDirectory() as d:
                    ck,_,n,w,s=fixture(d,bits=bits,dtype=dtype)
                    with_reader=IndexedMMapEmbedding(ck,1,n,w,s,row_cache_bytes=1024)
                    arrays=with_reader.read_rows(np.array([4,0,3])); saved=tuple(x.copy() for x in arrays)
                    for x in arrays: x.fill(0)
                    again=with_reader.read_rows(np.array([4,0,3]))
                    for x,y in zip(saved,again): np.testing.assert_array_equal(x,y)
                    with_reader.close()

    def test_zero_cache_keeps_legacy_semantics_and_invalid_limits_fail(self):
        with tempfile.TemporaryDirectory() as d:
            ck,_,n,w,s=fixture(d)
            r=IndexedMMapEmbedding(ck,1,n,w,s)
            self.assertEqual(r.mapping_stats()['limit'],8)
            self.assertEqual(r.mapping_stats()['row_cache_capacity'],0)
            r.close()
            for kw in ({'row_cache_bytes':-1},{'row_cache_bytes':True},{'max_open_files':0}):
                with self.assertRaises(ValueError): IndexedMMapEmbedding(ck,1,n,w,s,**kw)

if __name__=='__main__': unittest.main()
