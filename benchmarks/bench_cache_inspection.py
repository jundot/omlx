# SPDX-License-Identifier: Apache-2.0
"""Compare optional inspection overhead with synthetic paged SSD blocks.

Run from the repo root: python benchmarks/bench_cache_inspection.py
No model download, server, or original prompt/media data is needed.
"""

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx  # noqa: E402
from tokenizers import Tokenizer, models  # noqa: E402

from omlx.cache.inspection import BlockInspection, InspectionRenderer  # noqa: E402
from omlx.cache.paged_cache import compute_block_hash  # noqa: E402
from omlx.cache.paged_ssd_cache import PagedSSDCacheManager  # noqa: E402


def measure(blocks: int, enabled: bool) -> dict:
    tokenizer = Tokenizer(models.WordLevel({f"word{i}": i for i in range(4096)}))
    renderer = InspectionRenderer(tokenizer, "synthetic") if enabled else None
    kv = [(mx.zeros((1, 4, 256, 64)), mx.ones((1, 4, 256, 64))) for _ in range(2)]
    mx.eval(*(array for pair in kv for array in pair))
    with tempfile.TemporaryDirectory(prefix="omlx-inspection-bench-") as directory:
        manager = PagedSSDCacheManager(
            Path(directory), 2**30, inspection_renderer=renderer
        )
        started = time.perf_counter()
        parent = None
        try:
            for block in range(blocks):
                ids = [(block * 256 + index) % 4096 for index in range(256)]
                block_hash = compute_block_hash(parent, ids, model_name="synthetic")
                inspection = (
                    BlockInspection(
                        tuple(ids), block * 256, parent.hex() if parent else None
                    )
                    if enabled
                    else None
                )
                assert manager.save_block(
                    block_hash, kv, 256, "synthetic", inspection=inspection
                )
                parent = block_hash
            producer_ms = (time.perf_counter() - started) * 1000
        finally:
            manager.close()
        total_ms = (time.perf_counter() - started) * 1000
        stats = manager.get_stats_dict()
        assert stats["errors"] == 0 and stats["inspection_errors"] == 0
        return {
            "producer_ms": producer_ms,
            "total_ms": total_ms,
            "bytes": sum(
                p.stat().st_size for p in Path(directory).glob("*/*") if p.is_file()
            ),
            "inspection_writes": stats["inspection_writes"],
            "inline_fallbacks": stats["ssd_inline_write_fallbacks"],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.blocks < 1 or args.repeats < 1:
        parser.error("blocks and repeats must be positive")
    measure(2, False)  # Warm up MLX and filesystem helpers.
    runs = {"disabled": [], "enabled": []}
    for repeat in range(args.repeats):
        # Alternate ordering to reduce warm-cache and thermal bias.
        for enabled in ((False, True) if repeat % 2 == 0 else (True, False)):
            runs["enabled" if enabled else "disabled"].append(
                measure(args.blocks, enabled)
            )
    summary = {
        name: {key: statistics.median(run[key] for run in values) for key in values[0]}
        for name, values in runs.items()
    }
    print(
        json.dumps(
            {
                "blocks": args.blocks,
                "repeats": args.repeats,
                "medians": summary,
                "runs": runs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
