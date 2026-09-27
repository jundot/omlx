"""Bounded ABBA comparison of legacy and Sushi 1.0.5 EXL3 readers.

Run as python -m benchmarks.exl3_readers on a Metal host. Synthetic expert
banks use the Qwen3.8 routed projection dimensions, without loading a model.
These timings are kernel measurements, not end-to-end response speed.
"""
import json
import statistics
import time

import mlx.core as mx
import numpy as np

from omlx.quantization import exl3


def main():
    rng = np.random.default_rng(105)
    results = []
    for k, n in [(2560, 640), (640, 2560)]:
        packed = mx.array(rng.integers(0, 65536, (16, k // 16, n // 16, 42), dtype=np.uint16))
        mx.eval(packed)
        for rows in [10, 128, 512]:
            x = mx.array(rng.normal(size=(rows, k)).astype(np.float16))
            ids = mx.array(rng.integers(0, 16, rows, dtype=np.uint32))
            mx.eval(x, ids)
            outputs = []
            samples = {False: [], True: []}
            for fast in [False, True]:
                exl3._FAST_READERS = fast
                y = exl3._inner(x, ids, packed, exl3.Exl3Spec(42, 15), n)
                mx.eval(y)
                outputs.append(np.asarray(y).copy())
            if not np.array_equal(*outputs):
                raise RuntimeError('Reader outputs differ')
            for fast in [False, True, True, False] * 5:
                exl3._FAST_READERS = fast
                start = time.perf_counter()
                for _ in range(10):
                    y = exl3._inner(x, ids, packed, exl3.Exl3Spec(42, 15), n)
                    mx.eval(y)
                samples[fast].append((time.perf_counter() - start) / 10)
            old, new = [statistics.median(samples[f]) for f in [False, True]]
            result = {'input': k, 'output': n, 'rows': rows,
                      'legacy_ms': round(old * 1000, 3), 'fast_ms': round(new * 1000, 3),
                      'speedup': round(old / new, 3), 'bit_identical': True}
            results.append(result)
            print(json.dumps(result), flush=True)
        del packed
        mx.clear_cache()
    print(json.dumps({'device': mx.device_info(), 'results': results}, default=str), flush=True)


if __name__ == '__main__':
    main()
