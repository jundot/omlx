#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real-weight DS4 3:5 M1024 projection attribution and grouping probe.

This benchmark-only harness extends ``bench_ds4_tp_stage_profile`` with
per-projection brackets, then tests exact stock operations that could reduce
dispatches or input rereads. It never installs production dispatch and uses
identity collectives inside its isolated rank-local process.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

try:
    from benchmarks.bench_ds4_tp_stage_profile import (
        DS4TPShape,
        PROJECTION_DETAIL_CATEGORIES,
        _load_real_layers,
        identity_tp_collectives,
        modeled_collective_ms,
        normalize_to_observed_wall,
        profile_real_layers,
    )
except ModuleNotFoundError:  # Direct ``python benchmarks/this_file.py`` use.
    from bench_ds4_tp_stage_profile import (  # type: ignore[no-redef]
        DS4TPShape,
        PROJECTION_DETAIL_CATEGORIES,
        _load_real_layers,
        identity_tp_collectives,
        modeled_collective_ms,
        normalize_to_observed_wall,
        profile_real_layers,
    )


def _summary(values: Iterable[float]) -> dict[str, float]:
    samples = list(values)
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _balanced_timings(
    evaluate: Callable[[Any], None],
    functions: Mapping[str, Callable[[], Any]],
    *,
    warmup: int,
    cycles: int,
) -> dict[str, dict[str, float]]:
    names = tuple(functions)
    for _ in range(warmup):
        for name in names:
            evaluate(functions[name]())
    samples = {name: [] for name in names}
    orders = (
        names,
        tuple(reversed(names)),
        names[1:] + names[:1],
        tuple(reversed(names[1:] + names[:1])),
    )
    for _ in range(cycles):
        for order in orders:
            for name in order:
                started = time.perf_counter_ns()
                evaluate(functions[name]())
                samples[name].append((time.perf_counter_ns() - started) / 1e6)
    return {name: _summary(values) for name, values in samples.items()}


def _flatten_arrays(value: Any) -> list[Any]:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    if isinstance(value, mx.array):
        return [value]
    return [
        item
        for _path, item in tree_flatten(value)
        if isinstance(item, mx.array)
    ]


def _split_rows(value: Any, modules: Sequence[Any]) -> tuple[Any, ...]:
    cursor = 0
    result = []
    for module in modules:
        rows = int(module.weight.shape[0])
        result.append(value[..., cursor : cursor + rows])
        cursor += rows
    return tuple(result)


def _parity(reference: Any, candidate: Any) -> dict[str, Any]:
    import mlx.core as mx

    expected = _flatten_arrays(reference)
    actual = _flatten_arrays(candidate)
    if len(expected) != len(actual):
        return {"array_equal": False, "slices": []}
    slices = [
        bool(mx.array_equal(left, right).item())
        for left, right in zip(expected, actual)
    ]
    return {"array_equal": all(slices), "slices": slices}


def _projected_tps(
    wall_fraction: float,
    local_speedup: float,
    baseline_tps: float,
) -> float:
    return baseline_tps / (1.0 - wall_fraction + wall_fraction / local_speedup)


def _candidate_result(
    name: str,
    reference_name: str,
    candidate_name: str,
    timings: Mapping[str, Mapping[str, float]],
    parity: Mapping[str, Any],
    affected: Sequence[str],
    projection_fractions: Mapping[str, float],
    *,
    baseline_tps: float,
    min_speedup: float = 1.02,
) -> dict[str, Any]:
    reference_ms = float(timings[reference_name]["median_ms"])
    candidate_ms = float(timings[candidate_name]["median_ms"])
    local_speedup = reference_ms / candidate_ms
    wall_fraction = sum(projection_fractions.get(item, 0.0) for item in affected)
    exact = bool(parity["array_equal"])
    return {
        "candidate": name,
        "reference": reference_name,
        "implementation": candidate_name,
        "affected_projections": list(affected),
        "affected_observed_wall_fraction": wall_fraction,
        "parity": dict(parity),
        "timings": {
            reference_name: dict(timings[reference_name]),
            candidate_name: dict(timings[candidate_name]),
        },
        "local_speedup": local_speedup,
        "minimum_evidence_speedup": min_speedup,
        "projected_tps": (
            _projected_tps(wall_fraction, local_speedup, baseline_tps)
            if exact
            else None
        ),
        "eligible": exact and local_speedup >= min_speedup,
    }


def run_grouping_probe(
    model_dir: Path,
    *,
    shape: DS4TPShape,
    layer_index: int,
    warmup: int,
    cycles: int,
    projection_fractions: Mapping[str, float],
    baseline_tps: float,
    min_speedup: float,
) -> dict[str, Any]:
    """Benchmark exact grouping/concat/compile candidates on one ratio-4 layer."""

    import mlx.core as mx

    model, layers, ratios = _load_real_layers(
        model_dir,
        (layer_index,),
        shape,
    )
    del model
    layer = layers[0]
    if ratios[layer_index] != 4:
        raise ValueError("projection grouping probe requires a ratio-4 layer")
    attn = layer.attn
    indexer = attn.indexer

    mx.random.seed(41_002)
    x = mx.random.normal((shape.tokens, shape.hidden)).astype(mx.bfloat16)
    q_residual = attn.q_norm(attn.wq_a(x))
    o_a_input = mx.random.normal(
        (
            1,
            shape.o_groups,
            shape.tokens,
            shape.local_units * shape.head_dim,
        )
    ).astype(mx.bfloat16)
    mx.eval(x, q_residual, o_a_input)

    direct_quant = (attn.wq_a, attn.wkv)
    q_residual_quant = (attn.wq_b, indexer.wq_b)
    main_compressor = (attn.compressor.wkv, attn.compressor.wgate)
    indexer_dense = (
        indexer.compressor.wkv,
        indexer.compressor.wgate,
        indexer.weights_proj,
    )
    all_dense = (*main_compressor, *indexer_dense)

    # Build concatenated constants before timing. The candidate question is
    # dispatch/reduction geometry, not one-time model preparation.
    def prepare_quant(modules: Sequence[Any]):
        weight = mx.concatenate([module.weight for module in modules], axis=0)
        scales = mx.concatenate([module.scales for module in modules], axis=0)
        mx.eval(weight, scales)
        return weight, scales

    def prepared_quant_call(modules: Sequence[Any], prepared: Any, value: Any):
        first = modules[0]
        packed = mx.quantized_matmul(
            value,
            prepared[0],
            scales=prepared[1],
            biases=None,
            transpose=True,
            group_size=first.group_size,
            bits=first.bits,
            mode=first.mode,
        )
        return _split_rows(packed, modules)

    def prepare_dense(modules: Sequence[Any]):
        weight = mx.concatenate([module.weight for module in modules], axis=0)
        mx.eval(weight)
        return weight

    def prepared_dense_call(modules: Sequence[Any], weight: Any, value: Any):
        return _split_rows(value @ weight.T, modules)

    direct_quant_weights = prepare_quant(direct_quant)
    q_residual_weights = prepare_quant(q_residual_quant)
    main_compressor_weight = prepare_dense(main_compressor)
    indexer_dense_weight = prepare_dense(indexer_dense)
    all_dense_weight = prepare_dense(all_dense)

    def evaluate(value: Any) -> None:
        arrays = _flatten_arrays(value)
        if arrays:
            mx.eval(*arrays)
        mx.synchronize()

    def direct_quant_separate():
        return tuple(module(x) for module in direct_quant)

    def direct_quant_concat():
        return prepared_quant_call(direct_quant, direct_quant_weights, x)

    def q_residual_separate():
        return tuple(module(q_residual) for module in q_residual_quant)

    def q_residual_concat():
        return prepared_quant_call(q_residual_quant, q_residual_weights, q_residual)

    def main_compressor_separate():
        return tuple(module(x) for module in main_compressor)

    def main_compressor_concat():
        return prepared_dense_call(main_compressor, main_compressor_weight, x)

    def indexer_dense_separate():
        return tuple(module(x) for module in indexer_dense)

    def indexer_dense_concat():
        return prepared_dense_call(indexer_dense, indexer_dense_weight, x)

    def all_dense_separate():
        return tuple(module(x) for module in all_dense)

    def all_dense_concat():
        return prepared_dense_call(all_dense, all_dense_weight, x)

    def q_chain_separate(value: Any):
        q_a = attn.wq_a(value)
        raw_kv = attn.wkv(value)
        q_res = attn.q_norm(q_a)
        return attn.wq_b(q_res), indexer.wq_b(q_res), raw_kv

    def q_chain_concat(value: Any):
        q_a, raw_kv = prepared_quant_call(
            direct_quant,
            direct_quant_weights,
            value,
        )
        q_res = attn.q_norm(q_a)
        q_b, index_q = prepared_quant_call(
            q_residual_quant,
            q_residual_weights,
            q_res,
        )
        return q_b, index_q, raw_kv

    compiled_q_chain = mx.compile(q_chain_concat)

    def output_chain(value: Any):
        projected = attn.wo_a(value)
        projected = projected.transpose(0, 2, 1, 3).flatten(-2)
        return attn.wo_b(projected)

    compiled_output_chain = mx.compile(output_chain)

    functions = {
        "direct_quant_separate": direct_quant_separate,
        "direct_quant_concat": direct_quant_concat,
        "q_residual_separate": q_residual_separate,
        "q_residual_concat": q_residual_concat,
        "main_compressor_separate": main_compressor_separate,
        "main_compressor_concat": main_compressor_concat,
        "indexer_dense_separate": indexer_dense_separate,
        "indexer_dense_concat": indexer_dense_concat,
        "all_dense_separate": all_dense_separate,
        "all_dense_concat": all_dense_concat,
        "q_chain_separate": lambda: q_chain_separate(x),
        "q_chain_concat": lambda: q_chain_concat(x),
        "q_chain_concat_compiled": lambda: compiled_q_chain(x),
        "output_chain": lambda: output_chain(o_a_input),
        "output_chain_compiled": lambda: compiled_output_chain(o_a_input),
    }

    with identity_tp_collectives(len(shape.shard_weights)):
        values = {name: function() for name, function in functions.items()}
        for value in values.values():
            evaluate(value)
        parity = {
            "direct_quant_concat": _parity(
                values["direct_quant_separate"], values["direct_quant_concat"]
            ),
            "q_residual_concat": _parity(
                values["q_residual_separate"], values["q_residual_concat"]
            ),
            "main_compressor_concat": _parity(
                values["main_compressor_separate"],
                values["main_compressor_concat"],
            ),
            "indexer_dense_concat": _parity(
                values["indexer_dense_separate"], values["indexer_dense_concat"]
            ),
            "all_dense_concat": _parity(
                values["all_dense_separate"], values["all_dense_concat"]
            ),
            "q_chain_concat": _parity(
                values["q_chain_separate"], values["q_chain_concat"]
            ),
            "q_chain_concat_compiled": _parity(
                values["q_chain_separate"],
                values["q_chain_concat_compiled"],
            ),
            "output_chain_compiled": _parity(
                values["output_chain"], values["output_chain_compiled"]
            ),
        }
        timings = _balanced_timings(
            evaluate,
            functions,
            warmup=warmup,
            cycles=cycles,
        )

    candidates = [
        _candidate_result(
            "concat_q_a_and_raw_wkv",
            "direct_quant_separate",
            "direct_quant_concat",
            timings,
            parity["direct_quant_concat"],
            ("q_a", "raw_wkv"),
            projection_fractions,
            baseline_tps=baseline_tps,
            min_speedup=min_speedup,
        ),
        _candidate_result(
            "concat_q_b_and_indexer_q",
            "q_residual_separate",
            "q_residual_concat",
            timings,
            parity["q_residual_concat"],
            ("q_b", "indexer_q"),
            projection_fractions,
            baseline_tps=baseline_tps,
            min_speedup=min_speedup,
        ),
        _candidate_result(
            "concat_main_compressor_pair",
            "main_compressor_separate",
            "main_compressor_concat",
            timings,
            parity["main_compressor_concat"],
            ("compressor_wkv", "compressor_gate"),
            projection_fractions,
            baseline_tps=baseline_tps,
            min_speedup=min_speedup,
        ),
        _candidate_result(
            "concat_indexer_dense_triplet",
            "indexer_dense_separate",
            "indexer_dense_concat",
            timings,
            parity["indexer_dense_concat"],
            (
                "indexer_compressor_wkv",
                "indexer_compressor_gate",
                "indexer_weights",
            ),
            projection_fractions,
            baseline_tps=baseline_tps,
            min_speedup=min_speedup,
        ),
        _candidate_result(
            "concat_all_dense_direct_x",
            "all_dense_separate",
            "all_dense_concat",
            timings,
            parity["all_dense_concat"],
            (
                "compressor_wkv",
                "compressor_gate",
                "indexer_compressor_wkv",
                "indexer_compressor_gate",
                "indexer_weights",
            ),
            projection_fractions,
            baseline_tps=baseline_tps,
            min_speedup=min_speedup,
        ),
        _candidate_result(
            "concat_q_input_chain",
            "q_chain_separate",
            "q_chain_concat",
            timings,
            parity["q_chain_concat"],
            ("q_a", "raw_wkv", "q_b", "indexer_q"),
            projection_fractions,
            baseline_tps=baseline_tps,
            min_speedup=min_speedup,
        ),
        _candidate_result(
            "compile_concat_q_input_chain",
            "q_chain_separate",
            "q_chain_concat_compiled",
            timings,
            parity["q_chain_concat_compiled"],
            ("q_a", "raw_wkv", "q_b", "indexer_q"),
            projection_fractions,
            baseline_tps=baseline_tps,
            min_speedup=min_speedup,
        ),
        _candidate_result(
            "compile_output_chain",
            "output_chain",
            "output_chain_compiled",
            timings,
            parity["output_chain_compiled"],
            ("o_a", "o_b"),
            projection_fractions,
            baseline_tps=baseline_tps,
            min_speedup=min_speedup,
        ),
    ]
    candidates.sort(
        key=lambda item: (
            item["eligible"],
            item["projected_tps"] or 0.0,
        ),
        reverse=True,
    )
    return {
        "device": mx.device_info(),
        "layer": layer_index,
        "ratio": 4,
        "shape": shape.to_dict(),
        "timings": timings,
        "parity": parity,
        "candidates_ranked": candidates,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    model_dir = args.model.expanduser().resolve()
    weights = tuple(int(item.strip()) for item in args.shard_weights.split(","))
    shape = DS4TPShape(
        rank=args.rank,
        shard_weights=weights,
        tokens=args.tokens,
        prefix_tokens=args.prefix_tokens,
    )
    representatives = tuple(args.layers or (0, 2, 3))
    profile = profile_real_layers(
        model_dir,
        shape=shape,
        layers=representatives,
        warmup=args.warmup,
        cycles=args.cycles,
        projection_detail=True,
    )
    collectives = modeled_collective_ms(
        shape,
        bandwidth_gbps=args.collective_bandwidth_gbps,
        latency_us=args.collective_latency_us,
    )
    observed = normalize_to_observed_wall(
        profile["representative_stage_compute_ms"],
        collective_ms=collectives["total_ms"],
        tokens=shape.tokens,
        baseline_tps=args.baseline_tps,
        target_tps=args.target_tps,
    )
    scale = observed["isolated_to_observed_scale"]
    wall_ms = observed["observed_wall_ms"]
    projection_attributed_ms = {
        name: profile["representative_stage_projection_ms"][name] * scale
        for name in PROJECTION_DETAIL_CATEGORIES
    }
    projection_fractions = {
        name: value / wall_ms for name, value in projection_attributed_ms.items()
    }
    individual_rank = sorted(
        (
            {
                "projection": name,
                "isolated_stage_ms": profile[
                    "representative_stage_projection_ms"
                ][name],
                "attributed_ms": projection_attributed_ms[name],
                "observed_wall_fraction": projection_fractions[name],
                "infinite_ceiling_tps": args.baseline_tps
                / (1.0 - projection_fractions[name]),
            }
            for name in PROJECTION_DETAIL_CATEGORIES
        ),
        key=lambda item: item["observed_wall_fraction"],
        reverse=True,
    )
    grouping = run_grouping_probe(
        model_dir,
        shape=shape,
        layer_index=args.grouping_layer,
        warmup=max(2, args.warmup),
        cycles=args.grouping_cycles,
        projection_fractions=projection_fractions,
        baseline_tps=args.baseline_tps,
        min_speedup=args.min_speedup,
    )

    def future_candidate(
        name: str,
        affected: Sequence[str],
    ) -> dict[str, Any]:
        fraction = sum(projection_fractions[item] for item in affected)
        target_time = args.baseline_tps / args.target_tps
        denominator = target_time - (1.0 - fraction)
        return {
            "candidate": name,
            "affected_projections": list(affected),
            "observed_wall_fraction": fraction,
            "tps_at_2x": _projected_tps(fraction, 2.0, args.baseline_tps),
            "infinite_ceiling_tps": args.baseline_tps / (1.0 - fraction),
            "required_speedup_for_target_alone": (
                fraction / denominator if denominator > 0 else None
            ),
        }

    future_candidates = [
        future_candidate("native_or_sharded_o_a_o_b_chain", ("o_a", "o_b")),
        future_candidate(
            "native_q_input_chain",
            ("q_a", "raw_wkv", "q_b", "indexer_q"),
        ),
        future_candidate(
            "native_compressor_projection_bank",
            (
                "compressor_wkv",
                "compressor_gate",
                "indexer_compressor_wkv",
                "indexer_compressor_gate",
                "indexer_weights",
            ),
        ),
        future_candidate(
            "all_projection_work",
            PROJECTION_DETAIL_CATEGORIES,
        ),
    ]
    future_candidates.sort(key=lambda item: item["tps_at_2x"], reverse=True)
    projection_total = sum(projection_fractions.values())
    return {
        "schema_version": 1,
        "scope": "isolated_real_weight_rank_local_projection_profile",
        "profile": profile,
        "collective_model": collectives,
        "observed_attribution": observed,
        "projection_attributed_ms": projection_attributed_ms,
        "projection_fractions": projection_fractions,
        "individual_projections_ranked": individual_rank,
        "grouping_probe": grouping,
        "future_candidates_ranked": future_candidates,
        "decision": {
            "lead_candidate": "native_or_sharded_o_a_o_b_chain",
            "output_chain_share_of_projection_bucket": (
                (projection_fractions["o_a"] + projection_fractions["o_b"])
                / projection_total
            ),
            "stock_compile_promoted": False,
            "reason": (
                "O-A/O-B dominate the bucket, while stock compile is below the "
                "2% evidence gate; the next implementation must change the "
                "output-chain kernel or distributed ownership, not only wrap it."
            ),
        },
        "claims": {
            "production_dispatch_changed": False,
            "server_started": False,
            "remote_host_touched": False,
            "grouping_requires_array_equal": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=(0, 1), default=0)
    parser.add_argument("--shard-weights", default="3,5")
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--prefix-tokens", type=int, default=8192)
    parser.add_argument("--layers", type=int, nargs="*")
    parser.add_argument("--grouping-layer", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--grouping-cycles", type=int, default=5)
    parser.add_argument("--min-speedup", type=float, default=1.02)
    parser.add_argument("--baseline-tps", type=float, default=628.76)
    parser.add_argument("--target-tps", type=float, default=1000.0)
    parser.add_argument("--collective-bandwidth-gbps", type=float, default=6.2)
    parser.add_argument("--collective-latency-us", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
