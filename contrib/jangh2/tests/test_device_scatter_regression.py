"""CPU regression double for Metal's 64-bit scatter-value restriction.

This does not simulate Apple Metal execution. It enforces the documented
unsupported operation so NumPy's broader dtype support cannot hide this bug.
"""
import unittest
import numpy as np
from jangh2_bridge.device_rows import DeviceRowCache, row_slots


class ScatterCheckedArray(np.ndarray):
    def __array_finalize__(self, parent):
        self.scatter_log = getattr(parent, 'scatter_log', None)

    def __setitem__(self, index, value):
        if self.dtype.kind in 'iu' and self.dtype.itemsize == 8:
            raise ValueError(
                '[scatter] GPU scatter does not yet support '
                f'{self.dtype} for the input or updates.'
            )
        if self.scatter_log is not None:
            self.scatter_log.append((self.dtype, np.asarray(value).dtype))
        super().__setitem__(index, value)


class ScatterCheckedXP:
    def __init__(self):
        self.scatter_log = []
        self.evaluations = 0

    def __getattr__(self, name):
        return getattr(np, name)

    def wrap(self, array):
        out = np.asarray(array).view(ScatterCheckedArray)
        out.scatter_log = self.scatter_log
        return out

    def array(self, *args, **kwargs):
        return self.wrap(np.array(*args, **kwargs))

    def full(self, *args, **kwargs):
        return self.wrap(np.full(*args, **kwargs))

    def zeros(self, *args, **kwargs):
        return self.wrap(np.zeros(*args, **kwargs))

    def stack(self, *args, **kwargs):
        return self.wrap(np.stack(*args, **kwargs))

    def eval(self, *arrays):
        self.evaluations += 1


def rows_for(ids, dtype=np.float32):
    """Encode the complete tested row IDs without allocating a giant table."""
    ids = np.asarray(ids, dtype=np.uint64)
    # Four 16-bit limbs are exactly representable in float32.
    return np.stack([(ids >> (16*i)) & 65535 for i in range(4)], axis=-1).astype(dtype)


class DeviceScatterRegressionTests(unittest.TestCase):
    def setUp(self):
        self.xp = ScatterCheckedXP()
        self.calls = []

    def cache(self, row_count=2**48):
        return DeviceRowCache(self.xp, 4, np.float32, row_count,
                              byte_limit=8 * (8 + 4*4))

    def fetch(self, ids):
        self.calls.append(ids.copy())
        return self.xp.array(rows_for(ids))

    def test_first_miss_and_second_hit_use_supported_scatter_values(self):
        cache = self.cache()
        ids = self.xp.array([[2, 1, 2]], dtype=np.int64)
        actual = cache.lookup(ids, self.fetch)
        np.testing.assert_array_equal(actual, rows_for(ids))
        np.testing.assert_array_equal(cache.lookup(ids, self.fetch), actual)
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(self.xp.scatter_log)
        for destination, update in self.xp.scatter_log:
            self.assertFalse(destination.kind in 'iu' and destination.itemsize == 8)
            self.assertFalse(update.kind in 'iu' and update.itemsize == 8)
        cache.close()

    def test_tag_storage_keeps_full_64bit_identity_at_original_budget(self):
        cache = self.cache()
        cache.lookup(self.xp.array([2**40+7], dtype=np.int64), self.fetch)
        self.assertEqual(cache._tags.dtype, np.dtype('uint32'))
        self.assertEqual(cache._tags.shape, (cache.capacity, 2))
        self.assertEqual(cache._tags.nbytes, cache.capacity * 8)
        self.assertEqual(cache._tags.nbytes + cache._values.nbytes,
                         cache.stats()['device_payload_bytes'])
        cache.close()

    def test_same_low_word_same_slot_different_high_word_is_never_false_hit(self):
        ids = np.array([7, 2**40+7, 2**41+7], np.int64)
        cache = self.cache()
        slots = row_slots(ids, cache.capacity, np)
        self.assertTrue(np.all(slots == slots[0]))
        held = None
        for rid in [*ids, *ids[::-1], *ids]:
            x = self.xp.array([rid, rid], dtype=np.int64)
            result = cache.lookup(x, self.fetch)
            np.testing.assert_array_equal(result, rows_for(x))
            if held is None:
                held = result
        np.testing.assert_array_equal(held, rows_for([ids[0], ids[0]]))
        self.assertGreater(len(self.calls), 3)
        cache.close()

    def test_real_checkpoint_upper_row_id_is_exact(self):
        cache = self.cache(row_count=320001536)
        ids = self.xp.array([0, 320001535, 320001535], dtype=np.int64)
        np.testing.assert_array_equal(cache.lookup(ids, self.fetch), rows_for(ids))
        np.testing.assert_array_equal(cache.lookup(ids, self.fetch), rows_for(ids))
        self.assertEqual(len(self.calls), 1)
        cache.close()

    def test_all_supported_index_dtypes_keep_exact_values(self):
        for dtype in [np.int32, np.uint32, np.int64, np.uint64]:
            with self.subTest(dtype=dtype):
                cache = self.cache()
                ids = self.xp.array([0, 1234567, 7, 7], dtype=dtype)
                np.testing.assert_array_equal(cache.lookup(ids, self.fetch), rows_for(ids))
                np.testing.assert_array_equal(cache.lookup(ids, self.fetch), rows_for(ids))
                cache.close()

    def test_invalid_rows_cannot_match_uninitialised_or_truncated_tags(self):
        cache = self.cache(row_count=16)
        cache.lookup(self.xp.array([0, 7], dtype=np.int64), self.fetch)
        before = len(self.calls)
        for rid, dtype in [(-1, np.int64), (16, np.int64),
                           (2**32+7, np.uint64), (2**64-1, np.uint64)]:
            with self.subTest(rid=rid):
                with self.assertRaises(IndexError):
                    cache.lookup(self.xp.array([rid], dtype=dtype), self.fetch)
        self.assertEqual(len(self.calls), before)
        np.testing.assert_array_equal(cache.lookup(self.xp.array([0, 7], dtype=np.int64),
                                                  self.fetch), rows_for([0, 7]))
        cache.close()

    def test_uint32_boundary_and_signed64_high_bits_do_not_wrap(self):
        cache = self.cache(row_count=np.iinfo(np.int64).max)
        ids = self.xp.array([2**31-1, 2**31, 2**32-1, 2**32, 2**62+7], dtype=np.int64)
        for _ in range(2):
            np.testing.assert_array_equal(cache.lookup(ids, self.fetch), rows_for(ids))
        cache.close()

    def test_many_colliding_inserts_return_exact_rows_and_keep_old_outputs(self):
        cache = self.cache()
        rng = np.random.default_rng(14101)
        outputs = []
        for _ in range(100):
            ids = rng.integers(0, 2**47, size=(2, 9), dtype=np.int64)
            ids[1, :3] = ids[0, :3]
            out = cache.lookup(self.xp.array(ids), self.fetch)
            outputs.append((out, rows_for(ids)))
        for actual, expected in outputs:
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(cache.stats()['device_payload_bytes'], 192)
        cache.close()

    def test_empty_and_disabled_cache_do_not_create_tag_scatter(self):
        cache = DeviceRowCache(self.xp, 4, np.float32, 64, byte_limit=0)
        empty = cache.lookup(self.xp.array([], dtype=np.int64), self.fetch)
        self.assertEqual(empty.shape, (0, 4))
        ids = self.xp.array([3, 2], dtype=np.int64)
        np.testing.assert_array_equal(cache.lookup(ids, self.fetch), rows_for(ids))
        self.assertEqual(self.xp.scatter_log, [])
        cache.close()

    def test_rejected_fallback_does_not_commit_wrong_tag(self):
        cache = self.cache()
        ids = self.xp.array([7], dtype=np.int64)
        with self.assertRaises(ValueError):
            cache.lookup(ids, lambda i: self.xp.zeros((len(i), 3), dtype=np.float32))
        np.testing.assert_array_equal(cache.lookup(ids, self.fetch), rows_for(ids))
        self.assertEqual(len(self.calls), 1)
        cache.close()


if __name__ == '__main__':
    unittest.main()
