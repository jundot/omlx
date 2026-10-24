"""Shared exact-row-cache checks for CPU regression tests and Mac qualification.

Passing with a NumPy test double is not Apple Metal qualification. The standalone
Mac entry point and qualify.py both call this on oMLX's actual GPU worker.
"""
from __future__ import annotations
import numpy as np
from .device_rows import DeviceRowCache, row_slots



def _index_array(xp, values, dtype):
    """Upload exact, contiguous integer fixtures without Python-list coercion.

    MLX 0.32.4 validates Python-list ints through a signed int64 conversion
    even when uint64 was requested. A typed NumPy buffer preserves UINT64_MAX
    and also avoids negative-stride DLPack inputs. Only test inputs use this
    helper; the production cache and GPU kernels are unchanged.
    """
    for device_dtype, host_dtype in ((xp.int32, np.int32),
                                     (xp.uint32, np.uint32),
                                     (xp.int64, np.int64),
                                     (xp.uint64, np.uint64)):
        if dtype == device_dtype:
            # Apply the explicit dtype before inference can choose float64
            # for a list mixing small integers and UINT64_MAX.
            host = np.ascontiguousarray(values, dtype=host_dtype)
            return xp.array(host, dtype=dtype)
    raise TypeError('Unsupported index fixture dtype')


def _equal(xp, actual, expected):
    xp.eval(actual, expected)
    if actual.dtype != expected.dtype or tuple(actual.shape) != tuple(expected.shape):
        raise AssertionError('Device PLE result changed shape/dtype')
    if not bool(xp.all(actual == expected).item()):
        raise AssertionError('Device PLE rows differ')


def _stats(cache, xp, byte_limit, case):
    stats = cache.stats()
    if stats['device_all_hit_calls'] < 1 or stats['device_payload_bytes'] > byte_limit:
        raise AssertionError('Device cache hit/budget contract failed')
    if cache._tags.dtype != xp.uint32 or tuple(cache._tags.shape) != (cache.capacity, 2):
        raise AssertionError('Cache must store exact row tags as two uint32 words')
    return {'case': case, 'dtype': str(cache.dtype), 'tag_dtype': 'uint32',
            'tag_words': 2, 'byte_limit': byte_limit, **stats}


def _wide_rows(ids):
    """Eight exact bytes per ID, all representable in F16/BF16/F32."""
    words = np.asarray(ids, dtype=np.uint64)
    return np.stack([(words >> (8*i)) & 255 for i in range(8)], axis=-1).astype(np.float32)


def run(xp, *, verbose=True):
    """Exercise actual cache inserts/hits/evictions, not just a dtype allocation."""
    rng = np.random.default_rng(14101)
    dtypes = [xp.float16]
    if getattr(xp, 'bfloat16', None) is not None:
        dtypes.append(xp.bfloat16)
    dtypes.append(xp.float32)
    cases = []
    for dtype in dtypes:
        table = xp.array(rng.normal(size=(128, 64)).astype(np.float32), dtype=dtype)
        cache = DeviceRowCache(xp, 64, dtype, 128, byte_limit=4096)
        fetches = []
        def fetch(ids):
            fetches.append(ids.tolist())
            return table[_index_array(xp, ids, xp.int64)]
        try:
            for ids_np in ([2, 1, 2], [2, 1, 2], np.arange(70),
                           np.arange(70)[::-1], [5, 5], [5, 5]):
                # The reversed fixture is a negative-stride NumPy view.
                # MLX cannot import it through DLPack; materialise the same
                # row order before upload, without weakening the cache test.
                ids = _index_array(xp, ids_np, xp.int64)
                _equal(xp, cache.lookup(ids, fetch), table[ids])
            for index_dtype in (xp.int32, xp.uint32, xp.int64, xp.uint64):
                ids = _index_array(xp, [5, 5], index_dtype)
                _equal(xp, cache.lookup(ids, fetch), table[ids])
            calls_before = len(fetches)
            for value, index_dtype in ((-1, xp.int64), (128, xp.int64),
                                        (2**32+5, xp.uint64), (2**64-1, xp.uint64)):
                # Construct and verify the input OUTSIDE the expected-error
                # block: an importer failure is not a cache rejection.
                bad_ids = _index_array(xp, [value], index_dtype)
                xp.eval(bad_ids)
                if bad_ids.dtype != index_dtype or int(bad_ids[0].item()) != value:
                    raise AssertionError(f'Invalid-ID fixture changed value/dtype: {value}')
                try:
                    bad = cache.lookup(bad_ids, fetch)
                    xp.eval(bad)
                except IndexError:
                    pass
                else:
                    raise AssertionError(f'Out-of-vocabulary row was not rejected: {value}')
            if len(fetches) != calls_before:
                raise AssertionError('Invalid rows reached the backing reader')
            cases.append(_stats(cache, xp, 4096, 'rows-and-invalid-ids'))
        finally:
            cache.close()
        if verbose:
            print(f'PASS device row cache: {dtype}; misses, hits, collisions and invalid IDs', flush=True)

        # No enormous table: synthesise tiny row payloads from large IDs.
        itemsize = 4 if dtype == xp.float32 else 2
        budget = 8 * (8 + 8*itemsize)
        wide = DeviceRowCache(xp, 8, dtype, 2**41+8, byte_limit=budget)
        ids_np = np.array([7, 2**40+7, 2**41+7], dtype=np.int64)
        slots = row_slots(ids_np, wide.capacity, np)
        if not np.all(slots == slots[0]):
            raise AssertionError('Wide-ID fixture must deliberately collide')
        held = None
        def wide_fetch(ids):
            return xp.array(_wide_rows(ids), dtype=dtype)
        try:
            for rid in [*ids_np, *ids_np[::-1], 7, 7]:
                ids = _index_array(xp, [int(rid), int(rid)], xp.int64)
                result = wide.lookup(ids, wide_fetch)
                _equal(xp, result, xp.array(_wide_rows([rid, rid]), dtype=dtype))
                if held is None:
                    held = result
            _equal(xp, held, xp.array(_wide_rows([7, 7]), dtype=dtype))
            cases.append(_stats(wide, xp, budget, 'wide-row-tags'))
        finally:
            wide.close()
        if verbose:
            print(f'PASS full row-ID identity: {dtype}; same low word and slot, different high word', flush=True)
    return cases
