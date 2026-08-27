#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bounded-tile, deterministic DS4 TP prefill MoE phase-B contract.

The default path is CPU-only: it reports scratch, dispatch, and logical-byte
ceilings for the frozen M=1024 TP4/4 shape.  Once a later native agent exposes
the isolated candidate symbol, ``--model`` compares each supertile against the
current expert-major down + unsort + BF16 score/reduction path.

This harness does not wire production dispatch or compile native artifacts.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

try:
    from benchmarks.bench_ds4_tp_prefill_moe_campaign import (
        MXFP4_BYTES_PER_WEIGHT,
        bf16,
        bf16_bits,
        fp16,
        synthetic_routes,
    )
except ModuleNotFoundError:  # Direct ``python benchmarks/<script>.py`` execution.
    from bench_ds4_tp_prefill_moe_campaign import (  # type: ignore[no-redef]
        MXFP4_BYTES_PER_WEIGHT,
        bf16,
        bf16_bits,
        fp16,
        synthetic_routes,
    )


CONTRACT_VERSION = 1
PHASE_B_SYMBOL = "deepseek_mxfp4_down_top6_tiled"
SUPERTILE_CANDIDATES = (128, 256, 512, 1024, 2048, 4096)
PHASE_B_ABI = {
    "inputs": (
        "activated_f16",
        "down_weight_u32",
        "down_scales_u8",
        "block_meta_i32",
        "block_count_i32",
        "inverse_order_u32",
        "scores_f32",
        "supertile_i32",
    ),
    "output": "local_output_bf16",
}


@dataclass(frozen=True)
class Shape:
    tokens: int = 1024
    topk: int = 6
    experts: int = 256
    hidden: int = 4096
    local_intermediate: int = 1024
    layers: int = 43
    bm: int = 32
    bn: int = 32
    element_bytes: int = 2

    @property
    def routes(self) -> int:
        return self.tokens * self.topk

    @property
    def deterministic_blocks(self) -> int:
        # The frozen fixture has exactly 24 routes/expert, hence one BM32 block.
        return self.experts


@dataclass(frozen=True)
class TilePlan:
    supertile: int
    scratch_bytes: int
    supertiles: int
    down_dispatches: int
    reduction_dispatches: int
    total_dispatches: int
    all_layer_dispatches: int
    down_workgroups: int
    down_weight_read_amplification: float


def tile_plan(shape: Shape, supertile: int) -> TilePlan:
    if supertile <= 0 or supertile > shape.hidden:
        raise ValueError("supertile must be in (0, hidden]")
    if supertile % shape.bn != 0 or shape.hidden % supertile != 0:
        raise ValueError("supertile must divide hidden and be a BN multiple")

    supertiles = shape.hidden // supertile
    down_dispatches = supertiles
    reduction_dispatches = supertiles
    down_workgroups = (
        shape.deterministic_blocks * (supertile // shape.bn) * supertiles
    )
    expected_workgroups = shape.deterministic_blocks * (shape.hidden // shape.bn)
    if down_workgroups != expected_workgroups:
        raise AssertionError("tiling changed total down output-tile coverage")
    return TilePlan(
        supertile=supertile,
        scratch_bytes=shape.routes * supertile * shape.element_bytes,
        supertiles=supertiles,
        down_dispatches=down_dispatches,
        reduction_dispatches=reduction_dispatches,
        total_dispatches=down_dispatches + reduction_dispatches,
        all_layer_dispatches=(down_dispatches + reduction_dispatches) * shape.layers,
        down_workgroups=down_workgroups,
        down_weight_read_amplification=1.0,
    )


def output_partitions(shape: Shape, supertile: int) -> tuple[range, ...]:
    tile_plan(shape, supertile)
    return tuple(
        range(start, start + supertile)
        for start in range(0, shape.hidden, supertile)
    )


def validate_output_partitions(shape: Shape, supertile: int) -> None:
    columns = [
        column
        for part in output_partitions(shape, supertile)
        for column in part
    ]
    if columns != list(range(shape.hidden)):
        raise ValueError("output supertiles overlap or leave a gap")


def _mib(value: float) -> float:
    return value / (1024 * 1024)


def _amdahl(fraction: float, component_speedup: float) -> float:
    return 1.0 / ((1.0 - fraction) + fraction / component_speedup)


def byte_model(shape: Shape, moe_runtime_fraction: float = 0.5) -> dict[str, float]:
    """Logical payload ceilings; metadata reads are reported separately.

    The current payload counts explicit graph boundaries conservatively:
    down FP16 store, inverse gather, BF16 cast, BF16 score multiply, and BF16
    top-6 sum. The candidate writes/reads one bounded FP16 scratch tile and
    writes the local BF16 output. Actual cache traffic must be measured.
    """

    if not 0 < moe_runtime_fraction < 1:
        raise ValueError("moe_runtime_fraction must be between zero and one")
    r = shape.routes
    h = shape.hidden
    i = shape.local_intermediate
    b = shape.element_bytes

    route_down = r * h * b
    local_output = shape.tokens * h * b
    down_weights = shape.experts * h * i * MXFP4_BYTES_PER_WEIGHT
    down_activation_reads = route_down * (i / h) * (h // shape.bn)

    # Down result store plus four explicit tail stages:
    # unsort (read/write), cast (read/write), weight (read/write), sum (read/out).
    current_tail_payload = (
        route_down
        + 2 * route_down
        + 2 * route_down
        + 2 * route_down
        + route_down
        + local_output
    )
    candidate_tail_payload = 2 * route_down + local_output
    inverse_metadata_upper = shape.tokens * h * shape.topk * 4
    score_metadata_upper = shape.tokens * h * shape.topk * 4

    current_down_and_tail = down_weights + down_activation_reads + current_tail_payload
    candidate_down_and_tail = (
        down_weights + down_activation_reads + candidate_tail_payload
    )
    down_tail_speedup = current_down_and_tail / candidate_down_and_tail

    # Gate/up byte models are inherited from the phase-A campaign.
    route_x = r * h * b
    route_mid = r * i * b
    output_tiles_gate = i // shape.bn
    one_projection_weights = down_weights
    current_gate_up = (
        2 * one_projection_weights
        + 2 * route_x * output_tiles_gate
        + 5 * route_mid  # pair write/read (4x) plus activated write (1x)
    )
    phase_a_gate_up = (
        2 * one_projection_weights + route_x * output_tiles_gate + route_mid
    )

    current_routed = current_gate_up + current_down_and_tail
    phase_b_routed = current_gate_up + candidate_down_and_tail
    phase_ab_routed = phase_a_gate_up + candidate_down_and_tail
    phase_b_routed_speedup = current_routed / phase_b_routed
    phase_ab_routed_speedup = current_routed / phase_ab_routed

    return {
        "full_route_tensor_mib": _mib(route_down),
        "local_output_mib": _mib(local_output),
        "down_weight_mib_per_rank_layer": _mib(down_weights),
        "down_activation_logical_reads_mib": _mib(down_activation_reads),
        "current_tail_payload_mib": _mib(current_tail_payload),
        "candidate_tail_payload_mib": _mib(candidate_tail_payload),
        "tail_payload_speedup_ceiling": current_tail_payload
        / candidate_tail_payload,
        "inverse_metadata_logical_upper_mib": _mib(inverse_metadata_upper),
        "score_metadata_logical_upper_mib": _mib(score_metadata_upper),
        "current_down_and_tail_mib": _mib(current_down_and_tail),
        "candidate_down_and_tail_mib": _mib(candidate_down_and_tail),
        "down_and_tail_speedup_ceiling": down_tail_speedup,
        "phase_b_routed_speedup_ceiling": phase_b_routed_speedup,
        "phase_b_shape_e2e_ceiling": _amdahl(
            moe_runtime_fraction, phase_b_routed_speedup
        ),
        "phase_ab_routed_speedup_ceiling": phase_ab_routed_speedup,
        "phase_ab_shape_e2e_ceiling": _amdahl(
            moe_runtime_fraction, phase_ab_routed_speedup
        ),
    }


def reference_reduce(
    sorted_down: Sequence[Sequence[float]],
    inverse_order: Sequence[int],
    scores: Sequence[Sequence[float]],
    *,
    tokens: int,
    hidden: int,
) -> tuple[tuple[int, ...], ...]:
    """Return exact BF16 bits for the current post-down operation order."""

    if len(sorted_down) != tokens * 6 or len(inverse_order) != tokens * 6:
        raise ValueError("down rows and inverse order must have tokens*6 rows")
    if len(scores) != tokens or any(len(row) != 6 for row in scores):
        raise ValueError("scores must be [tokens, 6]")
    if any(len(row) != hidden for row in sorted_down):
        raise ValueError("sorted down rows have the wrong hidden width")

    output: list[tuple[int, ...]] = []
    for token in range(tokens):
        row_bits = []
        for column in range(hidden):
            total = bf16(0.0)
            for slot in range(6):
                route_id = token * 6 + slot
                sorted_row = inverse_order[route_id]
                route_value = bf16(fp16(sorted_down[sorted_row][column]))
                route_score = bf16(scores[token][slot])
                weighted = bf16(route_value * route_score)
                total = bf16(total + weighted)
            row_bits.append(bf16_bits(total))
        output.append(tuple(row_bits))
    return tuple(output)


def tiled_reduce(
    sorted_down: Sequence[Sequence[float]],
    inverse_order: Sequence[int],
    scores: Sequence[Sequence[float]],
    *,
    tokens: int,
    hidden: int,
    supertile: int,
) -> tuple[tuple[int, ...], ...]:
    """Model down-tile scratch consumption and fixed slot-order reduction."""

    shape = Shape(tokens=tokens, hidden=hidden, bn=1)
    if hidden % supertile != 0:
        raise ValueError("test supertile must divide hidden")
    output = [[0] * hidden for _ in range(tokens)]
    for columns in output_partitions(shape, supertile):
        for token in range(tokens):
            for column in columns:
                total = bf16(0.0)
                for slot in range(6):
                    route_id = token * 6 + slot
                    sorted_row = inverse_order[route_id]
                    route_value = bf16(fp16(sorted_down[sorted_row][column]))
                    route_score = bf16(scores[token][slot])
                    weighted = bf16(route_value * route_score)
                    total = bf16(total + weighted)
                output[token][column] = bf16_bits(total)
    return tuple(tuple(row) for row in output)


def analysis_report(shape: Shape, moe_runtime_fraction: float) -> dict[str, object]:
    plans = []
    for supertile in SUPERTILE_CANDIDATES:
        plan = tile_plan(shape, supertile)
        plans.append(
            {
                **asdict(plan),
                "scratch_mib": _mib(plan.scratch_bytes),
                "scratch_fraction_of_full_route": (
                    plan.scratch_bytes
                    / (shape.routes * shape.hidden * shape.element_bytes)
                ),
            }
        )
    return {
        "contract_version": CONTRACT_VERSION,
        "candidate_symbol": PHASE_B_SYMBOL,
        "abi": PHASE_B_ABI,
        "shape": asdict(shape),
        "exactness": {
            "down_store_dtype": "float16",
            "route_value_cast": "float16 -> bfloat16",
            "score_cast": "float32 -> bfloat16",
            "multiply_dtype": "bfloat16",
            "sum_dtype": "bfloat16",
            "sum_order": [0, 1, 2, 3, 4, 5],
            "atomics_allowed": False,
        },
        "plans": plans,
        "bytes": byte_model(shape, moe_runtime_fraction),
        "claims": {
            "down_weight_read_amplification": 1.0,
            "speed_claimed_without_gpu_gate": False,
        },
    }


def _evaluate(mx, value) -> None:
    mx.eval(value)
    mx.synchronize()


def _abba(
    mx,
    candidate: Callable[[], object],
    baseline: Callable[[], object],
    warmup: int,
    cycles: int,
) -> dict[str, list[float]]:
    for _ in range(warmup):
        _evaluate(mx, candidate())
        _evaluate(mx, baseline())
    timings = {"candidate": [], "baseline": []}
    for _ in range(cycles):
        for name, function in (
            ("candidate", candidate),
            ("baseline", baseline),
            ("baseline", baseline),
            ("candidate", candidate),
        ):
            started = time.perf_counter_ns()
            _evaluate(mx, function())
            timings[name].append((time.perf_counter_ns() - started) / 1e6)
    return timings


def _summary(values: Iterable[float]) -> dict[str, float]:
    values = list(values)
    return {
        "median_ms": statistics.median(values),
        "mean_ms": statistics.mean(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def run_gpu_gate(args: argparse.Namespace, shape: Shape) -> dict[str, object]:
    import mlx.core as mx

    from bench_ds4_tp_prefill_moe import load_tp_layer
    from omlx.custom_kernels.glm_moe_dsa import fast
    from omlx.patches.deepseek_v4.switch_layers import (
        _block_config,
        _build_mxfp4_blocks,
    )

    candidate_op = getattr(fast, PHASE_B_SYMBOL, None)
    if candidate_op is None or not fast.has_symbol(PHASE_B_SYMBOL):
        raise RuntimeError(
            f"isolated phase-B native prototype {PHASE_B_SYMBOL!r} is unavailable"
        )
    if shape.tokens != 1024 or shape.topk != 6:
        raise ValueError("phase-B promotion is frozen to M=1024/top-6")

    (
        up_weight,
        up_scales,
        gate_weight,
        gate_scales,
        down_weight,
        down_scales,
        shards,
    ) = load_tp_layer(args.model, args.layer, args.rank)

    mx.random.seed(1000 + shape.tokens)
    x = mx.random.normal((1, shape.tokens, shape.hidden)).astype(mx.bfloat16)
    routes = mx.array(synthetic_routes(shape), dtype=mx.uint32).reshape(
        1, shape.tokens, shape.topk
    )
    scores = mx.softmax(
        mx.random.normal((1, shape.tokens, shape.topk)).astype(mx.float32),
        axis=-1,
    )
    flat_routes = routes.flatten()
    order = mx.argsort(flat_routes)
    inverse = mx.argsort(order).astype(mx.uint32)
    sorted_ids = flat_routes[order]
    sorted_x = mx.expand_dims(x, (-2, -3)).flatten(0, -3)[order // shape.topk]
    sorted_x = sorted_x.astype(mx.float16)
    bm, variant = _block_config(shape.routes, "mxfp4")
    block_meta, block_count = _build_mxfp4_blocks(
        sorted_ids, shape.experts, bm
    )

    pair = fast.deepseek_mxfp4_gather_qmm_pair_concat_blocks(
        sorted_x,
        up_weight,
        up_scales,
        gate_weight,
        gate_scales,
        block_meta,
        block_count,
        variant,
    )
    up = pair[..., : shape.local_intermediate]
    gate = mx.minimum(pair[..., shape.local_intermediate :], args.activation_limit)
    activated = (gate * mx.sigmoid(gate)) * mx.clip(
        up, -args.activation_limit, args.activation_limit
    )
    mx.eval(activated, inverse, scores, block_meta, block_count)
    mx.synchronize()

    def baseline():
        down = fast.deepseek_mxfp4_gather_qmm_blocks(
            activated,
            down_weight,
            down_scales,
            block_meta,
            block_count,
            variant,
        )
        down = down[inverse].reshape(
            1, shape.tokens, shape.topk, 1, shape.hidden
        )
        down = down.squeeze(-2).astype(mx.bfloat16)
        return (down * scores[..., None].astype(down.dtype)).sum(-2)

    baseline_value = baseline()
    mx.eval(baseline_value)
    mx.synchronize()
    results = []
    for supertile in args.supertiles:
        plan = tile_plan(shape, supertile)

        def candidate(tile=supertile):
            return candidate_op(
                activated,
                down_weight,
                down_scales,
                block_meta,
                block_count,
                inverse,
                scores,
                tile,
            )

        candidate_value = candidate()
        mx.eval(candidate_value)
        mx.synchronize()
        exact = bool(mx.array_equal(candidate_value, baseline_value).item())
        max_abs = float(
            mx.max(
                mx.abs(
                    candidate_value.astype(mx.float32)
                    - baseline_value.astype(mx.float32)
                )
            ).item()
        )
        timings = _abba(mx, candidate, baseline, args.warmup, args.cycles)
        candidate_stats = _summary(timings["candidate"])
        baseline_stats = _summary(timings["baseline"])
        speedup = baseline_stats["median_ms"] / candidate_stats["median_ms"]
        passed = exact and speedup >= args.min_speedup
        results.append(
            {
                "supertile": supertile,
                "plan": asdict(plan),
                "parity": {"array_equal": exact, "max_abs": max_abs},
                "candidate": candidate_stats,
                "baseline": baseline_stats,
                "speedup": speedup,
                "passed": passed,
            }
        )
        if args.strict and not passed:
            raise SystemExit(2)

    return {
        "contract_version": CONTRACT_VERSION,
        "device_label": args.device_label,
        "model": str(args.model),
        "layer": args.layer,
        "rank": args.rank,
        "shape": asdict(shape),
        "shards": shards,
        "block": {"bm": bm, "variant": variant},
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument(
        "--device-label", choices=("m3-ultra", "m5-max"), help="required with --model"
    )
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--rank", type=int, choices=(0, 1), default=0)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument(
        "--supertiles",
        type=int,
        nargs="+",
        default=SUPERTILE_CANDIDATES,
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--activation-limit", type=float, default=10.0)
    parser.add_argument("--min-speedup", type=float, default=1.0)
    parser.add_argument("--moe-runtime-fraction", type=float, default=0.5)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shape = Shape(tokens=args.tokens)
    report: dict[str, object] = {
        "analysis": analysis_report(shape, args.moe_runtime_fraction)
    }
    if args.model is not None:
        if args.device_label is None:
            raise SystemExit("--device-label is required with --model")
        report["gpu_gate"] = run_gpu_gate(args, shape)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
