# SPDX-License-Identifier: Apache-2.0
"""Bench: Qwen4-Exp PLE N-gram gather cost — resident RAM vs SSD mmap.

Isolates the only code path that differs between PLE residency modes
(`qwen4_ple_ssd_offload`): the row gather. Everything else in the model is
identical between modes, so the delta measured here is the end-to-end delta.

Arms (same packed U32 rows, same dequant math, same shapes):
  resident    shards loaded into RAM as mx arrays; GPU gather + dequant.
  mmap_warm   DiskBackedShardedEmbedding on the real safetensors; rows that
              live in the OS page cache.
  mmap_cold   fresh rows outside anything the process touched (SSD reads).
              Uses `purge` when passwordless sudo is available, else falls
              back to shard files the resident arm never opened.

Row shape from the Qwen3.8-Flash-Next-oQ4e-mtp checkpoint: 128 shards x
2,500,012 rows x 160 dims @ 4-bit affine (group 32) -> ~100 B/row, 29.8 GiB
table. Per decode step the model gathers ngram_heads = 16 rows from ONE PLE
layer; a 512-token prefill gathers 8192 rows.

Usage:
    uv run bench/bench_ple_residency.py [--model PATH] [--resident-shards 16]
        [--out bench/results/ple_residency.json]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_MODEL = "/Volumes/SSD 4TB/AI Models/Qwen3.8-Flash-Next-oQ4e-mtp"
PLE_PREFIX = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding"
NGRAM_HEADS = 16  # context_len (ngram_size-1) * heads_per_ngram
DIMS = 160  # ple_embed_dim // ngram_heads
NUM_SHARDS = 128
ROWS_PER_SHARD = 2_500_012

# (name, tokens): decode gathers 16 rows/token, prefill gathers 16/token too.
SHAPES = [
    ("decode", 1),
    ("prefill_512", 512),
    ("prefill_4096", 4096),
]
ITERS = {"decode": 300, "prefill_512": 30, "prefill_4096": 8}
WARMUP = {"decode": 20, "prefill_512": 3, "prefill_4096": 2}


def _purge_available() -> bool:
    if shutil.which("purge") is None:
        return False
    try:
        subprocess.run(
            ["sudo", "-n", "purge"], check=True, capture_output=True, timeout=120
        )
        return True
    except Exception:
        return False


def load_resident_shards(model_path: Path, num_shards: int) -> dict:
    """Load the first `num_shards` PLE shards' packed tensors into RAM."""
    import mlx.core as mx

    from omlx.patches.mlx_vlm_qwen4_exp_compat import (
        apply_mlx_vlm_qwen4_exp_compat_patch,
    )

    apply_mlx_vlm_qwen4_exp_compat_patch()
    weight_map = json.loads(
        (model_path / "model.safetensors.index.json").read_text()
    )["weight_map"]

    shards = []
    t0 = time.perf_counter()
    for shard_index in range(num_shards):
        for sub in (f"shard_{shard_index}", f"shards.{shard_index}"):
            base = f"{PLE_PREFIX}.{sub}"
            weight_file = weight_map.get(f"{base}.weight")
            if weight_file is None:
                continue
            arrays = mx.load(str(model_path / weight_file))
            shards.append(
                (
                    arrays[f"{base}.weight"],
                    arrays[f"{base}.scales"],
                    arrays[f"{base}.biases"],
                )
            )
            break
        else:
            raise FileNotFoundError(f"PLE shard {shard_index} not found in {model_path}")
    mx.eval(*[a for tup in shards for a in tup])
    took = time.perf_counter() - t0
    return {"shards": shards, "load_s": took}


def resident_gather(resident: dict, ids: np.ndarray) -> "mx.array":
    """Same structure as DiskBackedShardedEmbedding.__call__, RAM-backed."""
    import bisect

    import mlx.core as mx

    shard_sizes = [ROWS_PER_SHARD] * NUM_SHARDS
    offsets = [0]
    for size in shard_sizes:
        offsets.append(offsets[-1] + size)
    flat = ids.reshape(-1)
    shard_indices = [bisect.bisect_right(offsets, int(i)) - 1 for i in flat]
    result = mx.zeros((len(flat), DIMS), dtype=mx.bfloat16)
    for shard_index in sorted(set(shard_indices)):
        positions = [
            i for i, s in enumerate(shard_indices) if s == shard_index
        ]
        local = [
            int(flat[i]) - offsets[shard_index] for i in positions
        ]
        if shard_index >= len(resident["shards"]):
            raise IndexError(
                f"shard {shard_index} outside the resident subset "
                f"({len(resident['shards'])} shards)"
            )
        weight, scales, biases = resident["shards"][shard_index]
        local_arr = mx.array(local, dtype=mx.int32)
        values = mx.dequantize(
            weight[local_arr],
            scales[local_arr],
            biases[local_arr],
            group_size=32,
            bits=4,
            mode="affine",
        )
        result = result.at[mx.array(positions, dtype=mx.int32)].add(values)
    return result.reshape(*ids.shape, DIMS)


def time_arm(fn, shape_name: str, tokens: int, fresh_rng=None, id_range=None) -> dict:
    """Time fn(ids) over ITERS; returns ms stats. fresh_rng = new ids/call."""
    import mlx.core as mx

    rows = tokens * NGRAM_HEADS
    iters = ITERS[shape_name]
    rng = np.random.default_rng(0)
    if id_range is None:
        id_range = ROWS_PER_SHARD * 16

    def make_ids():
        if fresh_rng is not None:
            return fresh_rng(rows)
        if not hasattr(make_ids, "_cache"):
            make_ids._cache = {}
        if rows not in make_ids._cache:
            make_ids._cache[rows] = rng.integers(
                0, id_range, size=(1, tokens, NGRAM_HEADS)
            )
        return make_ids._cache[rows]

    for _ in range(WARMUP[shape_name] if fresh_rng is None else 0):
        mx.eval(fn(make_ids()))
        if fresh_rng is None:
            mx.clear_cache()
    samples = []
    for _ in range(iters):
        ids = make_ids()
        t0 = time.perf_counter()
        out = fn(ids)
        mx.eval(out)
        samples.append((time.perf_counter() - t0) * 1000)
        mx.clear_cache()
    arr = np.array(samples)
    return {
        "iters": iters,
        "rows_per_call": rows,
        "min_ms": round(float(arr.min()), 4),
        "mean_ms": round(float(arr.mean()), 4),
        "median_ms": round(float(np.median(arr)), 4),
        "p95_ms": round(float(np.percentile(arr, 95)), 4),
        "us_per_row": round(float(arr.mean()) * 1000 / rows, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--resident-shards", type=int, default=16)
    parser.add_argument("--out", default="bench/results/ple_residency.json")
    args = parser.parse_args()

    import mlx.core as mx

    from omlx.patches.mlx_vlm_qwen4_exp_compat import (
        apply_mlx_vlm_qwen4_exp_compat_patch,
    )

    apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.language import DiskBackedShardedEmbedding

    model_path = Path(args.model).expanduser().resolve()
    results = {
        "model": model_path.name,
        "ngram_heads": NGRAM_HEADS,
        "dims_per_head": DIMS,
        "num_shards": NUM_SHARDS,
        "rows_per_shard": ROWS_PER_SHARD,
        "resident_shards": args.resident_shards,
        "bytes_per_row_approx": 100,
    }

    # ---- resident arm -------------------------------------------------
    print(f"loading {args.resident_shards} PLE shards into RAM ...", flush=True)
    resident = load_resident_shards(model_path, args.resident_shards)
    gib = sum(
        w.nbytes + s.nbytes + b.nbytes for w, s, b in resident["shards"]
    ) / 1024**3
    print(f"resident load {resident['load_s']:.1f}s ({gib:.2f} GiB)", flush=True)
    results["resident_load_s"] = round(resident["load_s"], 2)
    results["resident_gib"] = round(gib, 2)

    rng = np.random.default_rng(7)
    # shared ids live in the shards the resident arm holds, so both arms
    # gather the exact same rows
    id_range = ROWS_PER_SHARD * args.resident_shards

    def resident_fn(ids):
        return resident_gather(resident, ids)

    for name, tokens in SHAPES:
        results[f"resident_{name}"] = time_arm(
            resident_fn, name, tokens, id_range=id_range
        )
        print(
            f"resident    {name:<12} {results[f'resident_{name}']['median_ms']:.3f} ms "
            f"({results[f'resident_{name}']['us_per_row']} us/row)",
            flush=True,
        )

    # ---- mmap arm (warm) ----------------------------------------------
    # ids restricted to the shards the resident load already pulled through
    # the page cache, so both arms see the same rows.
    mmap = DiskBackedShardedEmbedding(
        model_path,
        PLE_PREFIX,
        ROWS_PER_SHARD * NUM_SHARDS,
        DIMS,
        NUM_SHARDS,
    )

    def mmap_fn(ids):
        return mmap(mx.array(ids))

    for name, tokens in SHAPES:
        results[f"mmap_warm_{name}"] = time_arm(
            mmap_fn, name, tokens, id_range=id_range
        )
        print(
            f"mmap warm   {name:<12} {results[f'mmap_warm_{name}']['median_ms']:.3f} ms "
            f"({results[f'mmap_warm_{name}']['us_per_row']} us/row)",
            flush=True,
        )

    # ---- mmap arm (cold: fresh rows each call) -------------------------
    cold_row_base = ROWS_PER_SHARD * 100  # shard 100+, untouched above

    def fresh_cold(rows: int):
        return rng.integers(
            cold_row_base,
            cold_row_base + ROWS_PER_SHARD * 16,
            size=(1, rows // NGRAM_HEADS, NGRAM_HEADS),
        )

    purged = _purge_available()
    results["purged"] = purged
    for name, tokens in SHAPES:
        # fresh rows every call, no warmup: every call pays the read.
        iters = ITERS[name]
        rows = tokens * NGRAM_HEADS
        samples = []
        for _ in range(iters):
            ids = fresh_cold(rows)
            t0 = time.perf_counter()
            out = mmap_fn(ids)
            mx.eval(out)
            samples.append((time.perf_counter() - t0) * 1000)
            mx.clear_cache()
        arr = np.array(samples)
        results[f"mmap_cold_{name}"] = {
            "iters": iters,
            "rows_per_call": rows,
            "min_ms": round(float(arr.min()), 4),
            "mean_ms": round(float(arr.mean()), 4),
            "median_ms": round(float(np.median(arr)), 4),
            "p95_ms": round(float(np.percentile(arr, 95)), 4),
            "us_per_row": round(float(arr.mean()) * 1000 / rows, 3),
        }
        print(
            f"mmap cold   {name:<12} {results[f'mmap_cold_{name}']['median_ms']:.3f} ms "
            f"({results[f'mmap_cold_{name}']['us_per_row']} us/row)",
            flush=True,
        )

    # ---- summary -------------------------------------------------------
    print("\n=== per decode step (16 rows, 1 PLE layer) ===")
    r = results["resident_decode"]["median_ms"]
    w = results["mmap_warm_decode"]["median_ms"]
    c = results["mmap_cold_decode"]["median_ms"]
    print(f"resident {r:.3f} ms | mmap warm {w:.3f} ms | mmap cold {c:.3f} ms")
    print(f"\nresults -> {args.out}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
