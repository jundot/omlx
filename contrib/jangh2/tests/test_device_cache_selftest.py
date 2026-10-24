import importlib
from pathlib import Path
import subprocess
import sys
import unittest
import numpy as np
from test_device_scatter_regression import ScatterCheckedXP

ROOT = Path(__file__).resolve().parents[1]


class DeviceCacheSelftestTests(unittest.TestCase):
    def test_shared_checks_cover_hot_misses_collisions_and_wide_ids(self):
        module = importlib.import_module('jangh2_bridge.device_selftest')
        xp = ScatterCheckedXP()
        cases = module.run(xp, verbose=False)
        # NumPy has F16 and F32 here, not BF16. Actual MLX covers all three.
        self.assertEqual(len(cases), 4)
        self.assertEqual({c['case'] for c in cases}, {'rows-and-invalid-ids', 'wide-row-tags'})
        for case in cases:
            self.assertEqual(case['tag_dtype'], 'uint32')
            self.assertEqual(case['tag_words'], 2)
            self.assertGreaterEqual(case['device_all_hit_calls'], 1)
            self.assertLessEqual(case['device_payload_bytes'], case['byte_limit'])
        self.assertTrue(xp.scatter_log)
        self.assertTrue(all(d.itemsize <= 4 and u.itemsize <= 4
                            for d, u in xp.scatter_log))

    def test_selftest_does_not_hide_bad_row_results(self):
        from unittest.mock import patch
        from jangh2_bridge.device_rows import DeviceRowCache
        module = importlib.import_module('jangh2_bridge.device_selftest')
        original = DeviceRowCache.lookup
        def corrupt(cache, ids, fetch):
            return original(cache, ids, fetch) + 1
        with patch.object(DeviceRowCache, 'lookup', corrupt):
            with self.assertRaises(AssertionError):
                module.run(ScatterCheckedXP(), verbose=False)

    def test_selftest_does_not_accept_64bit_tag_storage(self):
        from unittest.mock import patch
        from jangh2_bridge.device_rows import DeviceRowCache
        module = importlib.import_module('jangh2_bridge.device_selftest')
        original = DeviceRowCache.lookup
        def corrupt(cache, ids, fetch):
            out = original(cache, ids, fetch)
            if cache._tags is not None:
                cache._tags = cache._tags.astype(np.int64)
            return out
        with patch.object(DeviceRowCache, 'lookup', corrupt):
            with self.assertRaises((AssertionError, ValueError)):
                module.run(ScatterCheckedXP(), verbose=False)

    def test_cache_only_help_without_mlx_or_model(self):
        run = subprocess.run([sys.executable, str(ROOT / 'check_device_cache.py'), '--help'],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn('No model weights', run.stdout)


if __name__ == '__main__':
    unittest.main()
