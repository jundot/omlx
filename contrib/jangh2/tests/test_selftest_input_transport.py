"""Reproduce Metal array-import limits without claiming to execute Metal."""
import unittest
import numpy as np
from jangh2_bridge import device_selftest
from test_device_scatter_regression import ScatterCheckedXP

class ImportCheckedXP(ScatterCheckedXP):
    def array(self, value, *args, **kwargs):
        if isinstance(value, np.ndarray) and any(s < 0 for s in value.strides):
            raise ValueError('Cannot convert DLPack arrays with negative strides to mlx array.')
        if isinstance(value, (tuple, list)):
            if any(isinstance(v, int) and v > 2**63-1 for v in value):
                raise RuntimeError('std::bad_cast')
        return super().array(value, *args, **kwargs)

class SelftestInputTransportTests(unittest.TestCase):
    def test_reversed_fixture_keeps_order(self):
        values = np.arange(70, dtype=np.int64)[::-1]
        self.assertLess(values.strides[0], 0)
        actual = device_selftest._index_array(ImportCheckedXP(), values, np.int64)
        np.testing.assert_array_equal(actual, values)

    def test_uint64_max_does_not_take_python_signed_import_path(self):
        actual = device_selftest._index_array(ImportCheckedXP(), [2**64-1], np.uint64)
        self.assertEqual(actual.dtype, np.dtype('uint64'))
        self.assertEqual(int(actual[0]), 2**64-1)

    def test_all_index_dtypes_preserve_values(self):
        for dtype in (np.int32,np.uint32,np.int64,np.uint64):
            with self.subTest(dtype=dtype):
                a=device_selftest._index_array(ImportCheckedXP(),[7,0,7],dtype)
                self.assertEqual(a.dtype,np.dtype(dtype))
                np.testing.assert_array_equal(a,[7,0,7])

    def test_negative_signed_fixture_not_wrapped(self):
        a=device_selftest._index_array(ImportCheckedXP(),[-1],np.int64)
        self.assertEqual(int(a[0]),-1)

    def test_complete_diagnostic_passes_both_import_restrictions(self):
        cases=device_selftest.run(ImportCheckedXP(),verbose=False)
        self.assertEqual(len(cases),4) # NumPy has F16/F32, not MLX BF16.

    def test_invalid_ids_still_reach_cache_validation(self):
        from unittest.mock import patch
        from jangh2_bridge.device_rows import DeviceRowCache
        old=DeviceRowCache.lookup
        seen=[]
        def record(cache,ids,fetch):
            seen.extend(int(x) for x in np.asarray(ids).reshape(-1))
            return old(cache,ids,fetch)
        with patch.object(DeviceRowCache,'lookup',record):
            device_selftest.run(ImportCheckedXP(),verbose=False)
        for value in (-1,128,2**32+5,2**64-1):
            self.assertIn(value,seen)
