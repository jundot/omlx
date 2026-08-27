#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bit-exact ABBA gate for the DS4 M=1024 O-A -> BF16 -> O-B chain.

The native candidate deliberately keeps the BF16 O-A frontier.  It changes
only the O-A epilogue address calculation so that the mandatory intermediate
is born in O-B's token-major layout, then launches an otherwise unchanged
MLX-style MXFP8 O-B reduction in the same primitive.  No production dispatch
imports this benchmark.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx

try:
    from benchmarks.bench_ds4_projection_campaign import (
        O_GROUPS,
        O_RANK,
        TOKENS,
        _load_attention_layer,
        _qmm,
        rank_shape,
    )
except ModuleNotFoundError:  # Direct ``python benchmarks/this_file.py`` use.
    from bench_ds4_projection_campaign import (  # type: ignore[no-redef]
        O_GROUPS,
        O_RANK,
        TOKENS,
        _load_attention_layer,
        _qmm,
        rank_shape,
    )
from omlx.custom_kernels.glm_moe_dsa import fast

VARIANTS = {
    0: {"o_a_bm": 32, "o_b_bm": 32},
    1: {"o_a_bm": 64, "o_b_bm": 64},
    2: {"o_a_bm": 64, "o_b_bm": 32},
    3: {"o_a_bm": 32, "o_b_bm": 64},
}


def _load_single_attention_layer(model: Path, layer: int):
    """Load the unsliced H64 O-A/O-B banks for the single-node gate."""

    index = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    prefix = f"layers.{layer}.attn."
    shards = sorted(
        {filename for key, filename in index.items() if key.startswith(prefix)}
    )
    loaded = {}
    for shard in shards:
        loaded.update(mx.load(str(model / shard)))

    def get(name: str, suffix: str = "weight"):
        return loaded[f"layers.{layer}.attn.{name}.{suffix}"]

    tensors = {
        "o_a_weight": mx.contiguous(get("wo_a").reshape(O_GROUPS, O_RANK, -1)),
        "o_a_scales": mx.contiguous(
            get("wo_a", "scales").reshape(O_GROUPS, O_RANK, -1)
        ),
        "o_b_weight": get("wo_b"),
        "o_b_scales": get("wo_b", "scales"),
    }
    _evaluate(tuple(tensors.values()))
    return tensors, shards, SimpleNamespace(
        local_heads=64,
        o_a_input=4096,
    )


def _evaluate(value: Any) -> None:
    values = value if isinstance(value, (tuple, list)) else (value,)
    mx.eval(*values)
    mx.synchronize()


def _summary(samples: list[float]) -> dict[str, Any]:
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def _abba(
    reference: Callable[[], Any],
    candidate: Callable[[], Any],
    *,
    warmup: int,
    cycles: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        _evaluate(reference())
        _evaluate(candidate())
    samples = {"stock": [], "native": []}
    functions = {"stock": reference, "native": candidate}
    for _ in range(cycles):
        # A-B-B-A balances both short-term drift and command-order effects.
        for name in ("stock", "native", "native", "stock"):
            started = time.perf_counter_ns()
            _evaluate(functions[name]())
            samples[name].append((time.perf_counter_ns() - started) / 1e6)
    report = {name: _summary(values) for name, values in samples.items()}
    report["speedup"] = report["stock"]["median_ms"] / report["native"]["median_ms"]
    return report


def _exact(left: mx.array, right: mx.array) -> dict[str, Any]:
    _evaluate((left, right))
    delta = mx.abs(left.astype(mx.float32) - right.astype(mx.float32))
    return {
        "array_equal": bool(mx.array_equal(left, right).item()),
        "max_abs": float(mx.max(delta).item()),
    }


def run_rank(
    model: Path,
    layer: int,
    rank: int,
    *,
    warmup: int,
    cycles: int,
    parity_seeds: int,
) -> dict[str, Any]:
    if rank == -1:
        tensors, shards, shape = _load_single_attention_layer(model, layer)
    else:
        tensors, shards = _load_attention_layer(model, layer, rank)
        shape = rank_shape(rank)

    def stock_mid(value: mx.array) -> mx.array:
        projected = _qmm(mx, value, tensors["o_a_weight"], tensors["o_a_scales"])
        return mx.contiguous(
            projected.transpose(0, 2, 1, 3).reshape(1, TOKENS, O_GROUPS * O_RANK)
        )

    def stock_chain(value: mx.array) -> mx.array:
        projected = _qmm(mx, value, tensors["o_a_weight"], tensors["o_a_scales"])
        projected = projected.transpose(0, 2, 1, 3).reshape(
            1, TOKENS, O_GROUPS * O_RANK
        )
        return _qmm(mx, projected, tensors["o_b_weight"], tensors["o_b_scales"])

    parity: dict[str, list[dict[str, Any]]] = {
        f"variant_{variant}": [] for variant in VARIANTS
    }
    timing_input = None
    for seed_index in range(parity_seeds):
        mx.random.seed(63_000 + layer * 100 + rank * 10 + seed_index)
        value = mx.random.normal((1, O_GROUPS, TOKENS, shape.o_a_input)).astype(
            mx.bfloat16
        )
        _evaluate(value)
        if timing_input is None:
            timing_input = value

        reference_mid = stock_mid(value)
        reference_final = _qmm(
            mx,
            reference_mid,
            tensors["o_b_weight"],
            tensors["o_b_scales"],
        )
        _evaluate((reference_mid, reference_final))
        for variant, tile in VARIANTS.items():
            native_mid = fast.ds4_output_oa_interleaved(
                value,
                tensors["o_a_weight"],
                tensors["o_a_scales"],
                0 if tile["o_a_bm"] == 32 else 1,
            )
            final_from_native_mid = _qmm(
                mx,
                native_mid,
                tensors["o_b_weight"],
                tensors["o_b_scales"],
            )
            native_final = fast.ds4_output_projection_chain(
                value,
                tensors["o_a_weight"],
                tensors["o_a_scales"],
                tensors["o_b_weight"],
                tensors["o_b_scales"],
                variant,
            )
            parity[f"variant_{variant}"].append(
                {
                    "seed": seed_index,
                    "o_a_bf16_boundary": _exact(reference_mid, native_mid),
                    "o_b_from_native_boundary": _exact(
                        reference_final, final_from_native_mid
                    ),
                    "native_chain_final": _exact(reference_final, native_final),
                }
            )

    assert timing_input is not None
    timings = {}
    for variant, tile in VARIANTS.items():
        mid_variant = 0 if tile["o_a_bm"] == 32 else 1
        timings[f"variant_{variant}"] = {
            "tile": tile,
            "o_a_plus_layout": _abba(
                lambda: stock_mid(timing_input),
                lambda v=mid_variant: fast.ds4_output_oa_interleaved(
                    timing_input,
                    tensors["o_a_weight"],
                    tensors["o_a_scales"],
                    v,
                ),
                warmup=warmup,
                cycles=cycles,
            ),
            "full_chain": _abba(
                lambda: stock_chain(timing_input),
                lambda v=variant: fast.ds4_output_projection_chain(
                    timing_input,
                    tensors["o_a_weight"],
                    tensors["o_a_scales"],
                    tensors["o_b_weight"],
                    tensors["o_b_scales"],
                    v,
                ),
                warmup=warmup,
                cycles=cycles,
            ),
        }

    all_exact = all(
        boundary["array_equal"]
        for samples in parity.values()
        for sample in samples
        for boundary in (
            sample["o_a_bf16_boundary"],
            sample["o_b_from_native_boundary"],
            sample["native_chain_final"],
        )
    )
    best_variant = max(
        VARIANTS,
        key=lambda variant: timings[f"variant_{variant}"]["full_chain"]["speedup"],
    )
    return {
        "layer": layer,
        "rank": rank,
        "rank_shape": {
            "local_heads": shape.local_heads,
            "o_a_input": shape.o_a_input,
        },
        "checkpoint_shards": shards,
        "storage_contract": {
            "activation_dtype": str(timing_input.dtype),
            "o_a_weight": {
                "shape": list(tensors["o_a_weight"].shape),
                "dtype": str(tensors["o_a_weight"].dtype),
            },
            "o_a_scales": {
                "shape": list(tensors["o_a_scales"].shape),
                "dtype": str(tensors["o_a_scales"].dtype),
            },
            "mandatory_bf16_boundary": [1, TOKENS, O_GROUPS * O_RANK],
            "mandatory_boundary_bytes": TOKENS * O_GROUPS * O_RANK * 2,
            "stock_pre_layout": [1, O_GROUPS, TOKENS, O_RANK],
            "o_b_weight": {
                "shape": list(tensors["o_b_weight"].shape),
                "dtype": str(tensors["o_b_weight"].dtype),
            },
            "o_b_scales": {
                "shape": list(tensors["o_b_scales"].shape),
                "dtype": str(tensors["o_b_scales"].dtype),
            },
        },
        "parity": parity,
        "all_boundaries_exact": all_exact,
        "timings": timings,
        "best_variant": best_variant,
        "best_full_chain_speedup": timings[f"variant_{best_variant}"]["full_chain"][
            "speedup"
        ],
    }


def main() -> int:
    global TOKENS
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[2])
    parser.add_argument(
        "--ranks", type=int, choices=(-1, 0, 1), nargs="+", default=[0, 1]
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--cycles", type=int, default=12)
    parser.add_argument("--parity-seeds", type=int, default=3)
    parser.add_argument("--tokens", type=int, choices=(1024, 2048), default=1024)
    parser.add_argument(
        "--shard-weights",
        default="3,5",
        help="Two-rank DS4 head-unit split; use 4,4 for canonical TP2",
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    TOKENS = args.tokens
    shard_weights = tuple(
        int(item.strip()) for item in args.shard_weights.split(",")
    )
    if len(shard_weights) != 2 or any(weight < 1 for weight in shard_weights):
        raise ValueError("--shard-weights requires two positive integers")
    if sum(shard_weights) != 8:
        raise ValueError("DS4 shard weights must sum to eight head units")
    # The imported loader and shape function intentionally share the source
    # module globals. Override their benchmark-only partition before loading;
    # no production model class or environment variable is touched.
    previous_shard_weights = rank_shape.__globals__["SHARD_WEIGHTS"]
    try:
        rank_shape.__globals__["SHARD_WEIGHTS"] = shard_weights
        if not fast.has_symbol(
            "ds4_output_projection_chain"
        ) or not fast.has_symbol("ds4_output_oa_interleaved"):
            raise RuntimeError(
                "rebuilt DS4 output-chain native symbols are required"
            )
        reports = [
            run_rank(
                args.model.expanduser(),
                layer,
                rank,
                warmup=args.warmup,
                cycles=args.cycles,
                parity_seeds=args.parity_seeds,
            )
            for layer in args.layers
            for rank in args.ranks
        ]
    finally:
        rank_shape.__globals__["SHARD_WEIGHTS"] = previous_shard_weights
    payload = {
        "scope": "isolated_real_weight_ds4_m1024_output_projection_chain",
        "production_dispatch": False,
        "arithmetic_contract": {
            "o_a": "MLX MXFP8 dequantization + BK32 FP32 accumulation",
            "intermediate": "mandatory BF16 round, token-major physical store",
            "o_b": "MLX MXFP8 dequantization + BK32 FP32 accumulation",
            "parity_gate": "mx.array_equal at O-A BF16 and O-B output",
        },
        "reports": reports,
        "all_boundaries_exact": all(row["all_boundaries_exact"] for row in reports),
        "both_ranks_benefit": all(
            row["best_full_chain_speedup"] > 1.0 for row in reports
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.strict and not (
        payload["all_boundaries_exact"] and payload["both_ranks_benefit"]
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
