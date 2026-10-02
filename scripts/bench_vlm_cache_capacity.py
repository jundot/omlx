# SPDX-License-Identifier: Apache-2.0
"""Bounded cache-only reproduction; never loads weights or contacts the server."""

import argparse
import gc
import json
import time

import mlx.core as mx
from mlx_lm.models import cache as lm_cache
from mlx_vlm.models import cache as vlm_cache
from mlx_vlm.speculative.cache_state import start_speculative_cache

import omlx.scheduler  # noqa: F401 (activate serving cache patches)
from omlx.utils.proc_memory import get_phys_footprint

p = argparse.ArgumentParser()
p.add_argument("mode", choices=["append", "remerge", "ragged", "singleton"])
p.add_argument("--backend", choices=["vlm", "lm"], default="vlm")
a = p.parse_args()
BatchKVCache = (vlm_cache if a.backend == "vlm" else lm_cache).BatchKVCache
mx.set_memory_limit(8 * 1024**3)
mx.set_cache_limit(4 * 1024**3)
rows = 1 if a.mode == "singleton" else 2
layers, tokens, steps = 4, 8192, 32
caches = [BatchKVCache([0] * rows) for _ in range(layers)]


def evaluate():
    mx.eval([c.state for c in caches])
    mx.synchronize()


def sample(stage, step):
    return dict(
        stage=stage,
        step=step,
        active=mx.get_active_memory(),
        pool=mx.get_cache_memory(),
        peak_active=mx.get_peak_memory(),
        phys_footprint=get_phys_footprint(),
        logical_bytes=sum(c._idx * rows * 4 * 256 * 2 * 2 for c in caches),
    )


for c in caches:
    x = mx.ones((rows, 4, tokens, 256), mx.bfloat16)
    c.update_and_fetch(x, x * 2)
del x, c
evaluate()
mx.clear_cache()
mx.reset_peak_memory()
records = [sample("initial", 0)]
start = time.monotonic()
for step in range(1, steps + 1):
    transaction = start_speculative_cache(caches, 5) if a.mode == "ragged" else None
    if a.mode == "singleton":
        # Qwen3_5Model's singleton BatchKVCache -> KVCache -> merge path.
        row_caches = [c.extract(0) for c in caches]
        for c in row_caches:
            x = mx.full((1, 4, 5, 256), step, mx.bfloat16)
            c.update_and_fetch(x, x * 2)
        caches = [BatchKVCache.merge([c]) for c in row_caches]
        del row_caches
    else:
        for c in caches:
            x = mx.full((rows, 4, 5, 256), step, mx.bfloat16)
            c.update_and_fetch(x, x * 2)
        if a.mode == "ragged":
            transaction.commit([2, 5])
            del transaction
        elif a.mode == "remerge":
            caches = [
                BatchKVCache.merge([c.extract(i) for i in range(rows)]) for c in caches
            ]
    del x, c
    evaluate()
    records.append(sample("step", step))
    assert mx.get_active_memory() + mx.get_cache_memory() < 6 * 1024**3

elapsed = time.monotonic() - start
expected = [tokens + steps * n for n in ([2, 5] if a.mode == "ragged" else [5] * rows)]
for c in caches:
    assert c.offset.tolist() == expected, (c.offset.tolist(), expected)
    for row in range(rows):
        assert bool(mx.all(c.keys[row, :, c._idx - 1, :] == steps).item())
del c
gc.collect()
mx.synchronize()
mx.clear_cache()
records.append(sample("cleared", steps))
print(
    json.dumps(
        dict(
            mode=a.mode,
            backend=a.backend,
            rows=rows,
            layers=layers,
            tokens=tokens,
            steps=steps,
            elapsed=elapsed,
            offsets=expected,
            correctness="passed",
            measurements=records,
        ),
        indent=2,
    )
)
