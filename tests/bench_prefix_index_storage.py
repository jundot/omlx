# SPDX-License-Identifier: Apache-2.0
"""Compare production prefix-index builds against an unchanged git baseline.

Run: python tests/bench_prefix_index_storage.py --baseline <main-revision>
This measures CPU metadata only, not model inference or GPU allocations.
"""

import argparse
import ast
import gc
import json
import statistics
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from types import SimpleNamespace

import omlx.cache.prefix_cache as prefix_module
from omlx.cache.paged_cache import PagedCacheManager, compute_block_hash
from omlx.cache.prefix_cache import BlockAwarePrefixCache


def original_method(revision):
    source = subprocess.check_output(
        ["git", "show", f"{revision}:omlx/cache/prefix_cache.py"], text=True
    )
    cls = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "BlockAwarePrefixCache"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_update_prefix_index"
    )
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {"compute_block_hash": compute_block_hash, "Any": object}
    exec(compile(module, "<unchanged-baseline-method>", "exec"), namespace)
    return namespace[method.name]


def measure(count, baseline, prehashed=False):
    block_size = 256
    tokens = [i % 100000 for i in range(count * block_size)]
    paged = PagedCacheManager(
        block_size=block_size,
        max_blocks=count + 1,
        initial_blocks=count + 1,
        model_name="synthetic-prefix-index-benchmark",
    )
    cache = BlockAwarePrefixCache(SimpleNamespace(layers=[]), paged)
    blocks = paged.get_new_blocks(count)
    ids = [block.block_id for block in blocks]
    methods = {"main": baseline, "shared": BlockAwarePrefixCache._update_prefix_index}

    def reset():
        cache._prefix_index.clear()
        if not prehashed:
            for block in blocks:
                block.block_hash = None
        gc.collect()

    def retained():
        snapshots = {id(entry[1]): entry[1] for entry in cache._prefix_index.values()}
        return sum(sys.getsizeof(value) for value in snapshots.values())

    for method in methods.values():
        reset()
        method(cache, tokens, ids)
    rows = []
    for arm in ("main", "shared", "shared", "main") * 3:
        reset()
        start = time.perf_counter()
        methods[arm](cache, tokens, ids)
        elapsed = time.perf_counter() - start
        rows.append({"arm": arm, "seconds": elapsed, "id_tuple_bytes": retained()})
    heap = {}
    for arm, method in methods.items():
        reset()
        tracemalloc.start()
        method(cache, tokens, ids)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        heap[arm] = {"retained_python_bytes": current, "peak_python_bytes": peak}
    return {
        "production_module": str(Path(prefix_module.__file__).resolve()),
        "blocks": count,
        "tokens": len(tokens),
        "block_size": block_size,
        "prehashed": prehashed,
        "timing_conditions": (
            "one discarded warmup/arm; three ABBA blocks; cold index; "
            + ("precomputed block hashes" if prehashed else "cold block hashes")
            + "; GC outside timing; tracemalloc separate"
        ),
        "rows": rows,
        "median_seconds": {
            arm: statistics.median(row["seconds"] for row in rows if row["arm"] == arm)
            for arm in methods
        },
        "heap": heap,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--blocks", nargs="+", type=int, default=[512, 2048, 4096])
    parser.add_argument(
        "--prehashed", action="store_true", help="reuse hashes assigned before indexing"
    )
    args = parser.parse_args()
    assert (
        Path(prefix_module.__file__)
        .resolve()
        .is_relative_to(Path(__file__).resolve().parents[1])
    ), "benchmark imported a different checkout"
    baseline = original_method(args.baseline)
    for count in args.blocks:
        print(json.dumps(measure(count, baseline, args.prehashed)), flush=True)
