#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Contract and benchmark harness for the DS4 TP prefill MoE campaign.

The analysis path is CPU-only and describes the exact M=1024, TP4/4 shape.
When the isolated phase-A native prototype is available, ``--model`` runs a
balanced three-way comparison:

* the current MXFP4 pair-concat kernel plus LimitedSwiGLU;
* stock ``mx.gather_qmm`` plus LimitedSwiGLU; and
* the proposed shared-X pair kernel with a LimitedSwiGLU epilogue.

Production dispatch is intentionally outside this harness.  The candidate ABI
is frozen here before implementation so M3 Ultra and M5 Max can be measured
independently without silently changing the comparison.
"""

from __future__ import annotations

import argparse
import json
import statistics
import struct
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


CONTRACT_VERSION = 1
PHASE_A_SYMBOL = "deepseek_mxfp4_gather_qmm_pair_swiglu_blocks"
MXFP4_BYTES_PER_WEIGHT = 17 / 32


@dataclass(frozen=True)
class Shape:
    """The first promotion shape: DS4 equal TP2, called TP4/4 in the UI."""

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


@dataclass(frozen=True)
class ExpertBlock:
    """A deterministic slice of the stable expert-sorted route list."""

    sorted_start: int
    expert: int
    rows: int


def synthetic_routes(shape: Shape) -> tuple[int, ...]:
    """Return the real benchmark's deterministic, distinct top-6 pattern."""

    offsets = (0, 37, 79, 131, 181, 233)
    if shape.topk != len(offsets):
        raise ValueError("the DS4 campaign contract requires fixed top-6 routing")
    return tuple(
        (token * 73 + offset) % shape.experts
        for token in range(shape.tokens)
        for offset in offsets
    )


def deterministic_expert_map(
    routes: Sequence[int], experts: int, bm: int
) -> tuple[tuple[int, ...], tuple[ExpertBlock, ...]]:
    """Build a stable expert-major order and prefix-addressed block list.

    The secondary route-id key is part of the contract.  A future GPU builder
    may use a histogram and exclusive scan, but atomics may not decide the
    order of blocks or equal-expert routes.
    """

    if experts <= 0 or bm <= 0:
        raise ValueError("experts and bm must be positive")
    if any(expert < 0 or expert >= experts for expert in routes):
        raise ValueError("route expert is outside the local expert table")

    order = tuple(sorted(range(len(routes)), key=lambda rid: (routes[rid], rid)))
    blocks: list[ExpertBlock] = []
    cursor = 0
    for expert in range(experts):
        start = cursor
        while cursor < len(order) and routes[order[cursor]] == expert:
            cursor += 1
        for row in range(start, cursor, bm):
            blocks.append(ExpertBlock(row, expert, min(bm, cursor - row)))
    if cursor != len(order):
        raise AssertionError("expert map did not consume every route")
    return order, tuple(blocks)


def validate_expert_map(
    routes: Sequence[int],
    order: Sequence[int],
    blocks: Sequence[ExpertBlock],
    *,
    experts: int,
    bm: int,
) -> None:
    """Raise if a proposed map violates the phase-A routing contract."""

    if tuple(sorted(order)) != tuple(range(len(routes))):
        raise ValueError("expert order is not a route-id permutation")
    expected_order, expected_blocks = deterministic_expert_map(routes, experts, bm)
    if tuple(order) != expected_order:
        raise ValueError("expert order is not stable by (expert, route_id)")
    if tuple(blocks) != expected_blocks:
        raise ValueError("block list is not the deterministic prefix plan")


def _f32(value: float) -> float:
    return struct.unpack(">f", struct.pack(">f", value))[0]


def fp16(value: float) -> float:
    """Round once to IEEE binary16, returning the rounded Python float."""

    return struct.unpack(">e", struct.pack(">e", value))[0]


def bf16_bits(value: float) -> int:
    """Round a finite value to bfloat16 using round-to-nearest-even."""

    bits = struct.unpack(">I", struct.pack(">f", _f32(value)))[0]
    exponent = bits & 0x7F800000
    mantissa = bits & 0x007FFFFF
    if exponent == 0x7F800000:
        upper = bits >> 16
        if mantissa and not (upper & 0x007F):
            upper |= 1
        return upper
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def bf16(value: float) -> float:
    return struct.unpack(">f", struct.pack(">I", bf16_bits(value) << 16))[0]


def reference_top6_scalar(
    down_accumulators: Sequence[float], scores: Sequence[float]
) -> float:
    """Model the current DS4 post-down cast, weight, and reduction boundary.

    Each down accumulator is first stored as FP16 by the current block GEMM,
    then cast to the original BF16 dtype.  Scores are cast from FP32 to BF16,
    multiplication is BF16, and MLX's six-row small reduction accumulates in
    slot order in BF16.  Phase B must reproduce this operation sequence.
    """

    if len(down_accumulators) != 6 or len(scores) != 6:
        raise ValueError("DS4 Flash uses exactly six routed slots")
    total = bf16(0.0)
    for value, score in zip(down_accumulators, scores):
        route_value = bf16(fp16(value))
        weighted = bf16(route_value * bf16(score))
        total = bf16(total + weighted)
    return total


def sorted_route_top6_scalar(
    sorted_route_ids: Sequence[int],
    down_by_route: Sequence[float],
    scores: Sequence[float],
) -> float:
    """Reduce expert-major results by original slot, never completion order."""

    if tuple(sorted(sorted_route_ids)) != tuple(range(6)):
        raise ValueError("one-token route ids must be a permutation of 0..5")
    restored = [0.0] * 6
    for route_id in sorted_route_ids:
        restored[route_id] = down_by_route[route_id]
    return reference_top6_scalar(restored, scores)


def _mib(value: float) -> float:
    return value / (1024 * 1024)


def materialization_ledger(shape: Shape) -> dict[str, object]:
    r = shape.routes
    h = shape.hidden
    i = shape.local_intermediate
    b = shape.element_bytes
    current = {
        "sorted_route_input": r * h * b,
        "gate_up_pair": r * 2 * i * b,
        "activated_mid": r * i * b,
        "sorted_down_routes": r * h * b,
        "unsorted_down_routes": r * h * b,
        "weighted_routes": r * h * b,
        "local_output": shape.tokens * h * b,
    }
    phase_a = dict(current)
    phase_a.pop("gate_up_pair")
    phase_ab = {
        "bounded_down_tile_scratch": r * shape.bn * b,
        "local_output": shape.tokens * h * b,
    }
    core_eliminated = (
        current["gate_up_pair"]
        + current["activated_mid"]
        + current["sorted_down_routes"]
    )
    return {
        "current_mib": {key: _mib(value) for key, value in current.items()},
        "phase_a_mib": {key: _mib(value) for key, value in phase_a.items()},
        "phase_ab_target_mib": {
            key: _mib(value) for key, value in phase_ab.items()
        },
        "phase_a_removed_persistent_mib": _mib(current["gate_up_pair"]),
        "phase_ab_removed_core_persistent_mib": _mib(core_eliminated),
        "optional_indirect_x_additional_mib": _mib(current["sorted_route_input"]),
    }


def _amdahl(fraction: float, component_speedup: float) -> float:
    if not 0 <= fraction < 1:
        raise ValueError("fraction must be in [0, 1)")
    if component_speedup <= 0:
        raise ValueError("component speedup must be positive")
    return 1.0 / ((1.0 - fraction) + fraction / component_speedup)


def roofline(shape: Shape, moe_runtime_fraction: float = 0.5) -> dict[str, float]:
    """Return FLOP, logical-byte, and Amdahl ceilings for the first campaign."""

    if not 0 < moe_runtime_fraction < 1:
        raise ValueError("moe_runtime_fraction must be between zero and one")
    r = shape.routes
    h = shape.hidden
    i = shape.local_intermediate
    b = shape.element_bytes

    one_projection_flops = 2 * r * h * i
    gate_up_flops = 2 * one_projection_flops
    down_flops = one_projection_flops
    routed_flops = gate_up_flops + down_flops

    one_projection_weights = shape.experts * h * i * MXFP4_BYTES_PER_WEIGHT
    output_tiles_gate = i // shape.bn
    output_tiles_down = h // shape.bn
    route_x = r * h * b
    route_mid = r * i * b
    route_down = r * h * b

    current_gate_up = (
        2 * one_projection_weights
        + 2 * route_x * output_tiles_gate
        + 2 * route_mid  # pair store, then activation read
        + route_mid  # activated write
    )
    phase_a_gate_up = (
        2 * one_projection_weights
        + route_x * output_tiles_gate
        + route_mid
    )
    current_down = (
        one_projection_weights
        + route_mid * output_tiles_down
        + route_down
    )
    current_routed = current_gate_up + current_down
    phase_a_routed = phase_a_gate_up + current_down

    # The A+B byte ceiling assumes the activated expert block remains on chip
    # through down projection and only the final local token row is persisted.
    # It is an upper bound, not a claim that a BM32 block fits that schedule.
    phase_ab_routed = (
        phase_a_gate_up
        - route_mid
        + one_projection_weights
        + shape.tokens * h * b
    )

    phase_a_logical_moe_speedup = current_routed / phase_a_routed
    phase_ab_logical_moe_speedup = current_routed / phase_ab_routed
    phase_a_total_fraction = moe_runtime_fraction * (2 / 3)

    return {
        "routed_flops_per_layer_chunk": routed_flops,
        "routed_flops_per_token_per_rank_all_layers": (
            routed_flops * shape.layers / shape.tokens
        ),
        "required_tflops_per_rank_at_1000_tok_s": (
            routed_flops * shape.layers / shape.tokens * 1000 / 1e12
        ),
        "mxfp4_weight_mib_per_projection_per_rank_layer": _mib(
            one_projection_weights
        ),
        "current_logical_routed_mib_per_layer": _mib(current_routed),
        "phase_a_logical_routed_mib_per_layer": _mib(phase_a_routed),
        "phase_ab_logical_routed_mib_per_layer": _mib(phase_ab_routed),
        "phase_a_logical_moe_speedup_ceiling": phase_a_logical_moe_speedup,
        "phase_ab_logical_moe_speedup_ceiling": phase_ab_logical_moe_speedup,
        "phase_a_shape_e2e_ceiling": _amdahl(
            moe_runtime_fraction, phase_a_logical_moe_speedup
        ),
        "phase_ab_shape_e2e_ceiling": _amdahl(
            moe_runtime_fraction, phase_ab_logical_moe_speedup
        ),
        "phase_a_infinite_e2e_ceiling": 1 / (1 - phase_a_total_fraction),
        "phase_ab_infinite_e2e_ceiling": 1 / (1 - moe_runtime_fraction),
    }


def analysis_report(shape: Shape, moe_runtime_fraction: float) -> dict[str, object]:
    routes = synthetic_routes(shape)
    order, blocks = deterministic_expert_map(routes, shape.experts, shape.bm)
    counts = [0] * shape.experts
    for expert in routes:
        counts[expert] += 1
    active_per_slot = []
    for slot in range(shape.topk):
        active_per_slot.append(
            len(set(routes[slot :: shape.topk]))
        )
    active_experts = sum(count > 0 for count in counts)
    return {
        "contract_version": CONTRACT_VERSION,
        "shape": asdict(shape),
        "routes": shape.routes,
        "deterministic_map": {
            "blocks": len(blocks),
            "active_experts": active_experts,
            "min_routes_per_active_expert": min(count for count in counts if count),
            "max_routes_per_expert": max(counts),
            "order_checksum": sum(
                (position + 1) * route for position, route in enumerate(order)
            ),
        },
        "materializations": materialization_ledger(shape),
        "roofline": roofline(shape, moe_runtime_fraction),
        "phase_b_contract": {
            "atomics_allowed": False,
            "route_score_stage": "after_fp16_down_store_and_bf16_cast",
            "reduction_slot_order": [0, 1, 2, 3, 4, 5],
            "active_experts_per_slot": active_per_slot,
            "per_slot_down_weight_read_amplification": (
                sum(active_per_slot) / active_experts
            ),
            "accepted_schedules": [
                "six sequential per-slot output passes",
                "expert-major down plus deterministic fixed-slot second reduction",
                "bounded output-tile scratch plus deterministic fixed-slot reduction",
            ],
        },
    }


def _evaluate(mx, value) -> None:
    mx.eval(value)
    mx.synchronize()


def _balanced_timings(mx, functions: dict[str, object], warmup: int, cycles: int):
    names = tuple(functions)
    for _ in range(warmup):
        for name in names:
            _evaluate(mx, functions[name]())
    timings = {name: [] for name in names}
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
                _evaluate(mx, functions[name]())
                timings[name].append((time.perf_counter_ns() - started) / 1e6)
    return timings


def _timing_summary(values: Iterable[float]) -> dict[str, float]:
    values = list(values)
    return {
        "median_ms": statistics.median(values),
        "mean_ms": statistics.mean(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def run_phase_a(args: argparse.Namespace, shape: Shape) -> dict[str, object]:
    import mlx.core as mx

    from bench_ds4_tp_prefill_moe import load_tp_layer
    from omlx.custom_kernels.glm_moe_dsa import fast
    from omlx.patches.deepseek_v4.switch_layers import (
        _block_config,
        _build_mxfp4_blocks,
    )

    if shape.tokens != 1024 or shape.topk != 6:
        raise ValueError("phase-A promotion is frozen to M=1024/top-6")
    candidate = getattr(fast, PHASE_A_SYMBOL, None)
    if candidate is None or not fast.has_symbol(PHASE_A_SYMBOL):
        raise RuntimeError(
            f"isolated phase-A native prototype {PHASE_A_SYMBOL!r} is unavailable"
        )

    (
        up_weight,
        up_scales,
        gate_weight,
        gate_scales,
        _down_weight,
        _down_scales,
        shards,
    ) = load_tp_layer(args.model, args.layer, args.rank)

    mx.random.seed(1000 + shape.tokens)
    x = mx.random.normal((1, shape.tokens, shape.hidden)).astype(mx.bfloat16)
    route_values = synthetic_routes(shape)
    routes = mx.array(route_values, dtype=mx.uint32).reshape(
        1, shape.tokens, shape.topk
    )
    flat_routes = routes.flatten()
    order = mx.argsort(flat_routes)
    sorted_ids = flat_routes[order]
    sorted_x = mx.expand_dims(x, (-2, -3)).flatten(0, -3)[order // shape.topk]
    sorted_x = sorted_x.astype(mx.float16)
    bm, variant = _block_config(shape.routes, "mxfp4")
    block_meta, block_count = _build_mxfp4_blocks(
        sorted_ids, shape.experts, bm
    )
    mx.eval(sorted_x, sorted_ids, block_meta, block_count)
    mx.synchronize()

    stock_kwargs = {
        "transpose": True,
        "group_size": 32,
        "bits": 4,
        "mode": "mxfp4",
        "sorted_indices": True,
    }

    def activate(up, gate):
        gate = mx.minimum(gate, args.activation_limit)
        return (gate * mx.sigmoid(gate)) * mx.clip(
            up, -args.activation_limit, args.activation_limit
        )

    def current():
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
        return activate(
            pair[..., : shape.local_intermediate],
            pair[..., shape.local_intermediate :],
        )

    def stock():
        up = mx.gather_qmm(
            sorted_x,
            up_weight,
            up_scales,
            None,
            rhs_indices=sorted_ids,
            **stock_kwargs,
        )
        gate = mx.gather_qmm(
            sorted_x,
            gate_weight,
            gate_scales,
            None,
            rhs_indices=sorted_ids,
            **stock_kwargs,
        )
        return activate(up, gate)

    def shared_x():
        return candidate(
            sorted_x,
            up_weight,
            up_scales,
            gate_weight,
            gate_scales,
            block_meta,
            block_count,
            float(args.activation_limit),
            variant,
        )

    current_value = current()
    stock_value = stock()
    candidate_value = shared_x()
    mx.eval(current_value, stock_value, candidate_value)
    mx.synchronize()
    exact_current = bool(mx.array_equal(candidate_value, current_value).item())
    exact_stock = bool(mx.array_equal(candidate_value, stock_value).item())
    current_stock_exact = bool(mx.array_equal(current_value, stock_value).item())
    max_abs_current = float(
        mx.max(
            mx.abs(
                candidate_value.astype(mx.float32)
                - current_value.astype(mx.float32)
            )
        ).item()
    )
    max_abs_stock = float(
        mx.max(
            mx.abs(
                candidate_value.astype(mx.float32)
                - stock_value.astype(mx.float32)
            )
        ).item()
    )

    timings = _balanced_timings(
        mx,
        {"shared_x": shared_x, "pair_concat": current, "stock": stock},
        args.warmup,
        args.cycles,
    )
    summaries = {name: _timing_summary(values) for name, values in timings.items()}
    candidate_ms = summaries["shared_x"]["median_ms"]
    current_ms = summaries["pair_concat"]["median_ms"]
    stock_ms = summaries["stock"]["median_ms"]
    best_baseline_ms = min(current_ms, stock_ms)
    passed = (
        exact_current
        and exact_stock
        and current_stock_exact
        and best_baseline_ms / candidate_ms >= args.min_speedup
    )
    result = {
        "contract_version": CONTRACT_VERSION,
        "device_label": args.device_label,
        "model": str(args.model),
        "layer": args.layer,
        "rank": args.rank,
        "shape": asdict(shape),
        "shards": shards,
        "block": {"bm": bm, "variant": variant},
        "post_limited_swiglu_parity": {
            "candidate_vs_pair_concat_exact": exact_current,
            "candidate_vs_pair_concat_max_abs": max_abs_current,
            "candidate_vs_stock_exact": exact_stock,
            "candidate_vs_stock_max_abs": max_abs_stock,
            "pair_concat_vs_stock_exact": current_stock_exact,
        },
        "timings": summaries,
        "speedup_vs_pair_concat": current_ms / candidate_ms,
        "speedup_vs_stock": stock_ms / candidate_ms,
        "speedup_vs_best_current": best_baseline_ms / candidate_ms,
        "min_speedup": args.min_speedup,
        "passed": passed,
    }
    if args.strict and not passed:
        raise SystemExit(2)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument(
        "--device-label", choices=("m3-ultra", "m5-max"), help="required with --model"
    )
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--rank", type=int, choices=(0, 1), default=0)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--activation-limit", type=float, default=10.0)
    parser.add_argument("--min-speedup", type=float, default=1.05)
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
        report["phase_a"] = run_phase_a(args, shape)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
