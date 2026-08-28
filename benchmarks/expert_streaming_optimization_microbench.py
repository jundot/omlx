#!/usr/bin/env python3
"""Bounded microbenchmarks for proposed SSD expert-streaming optimizations.

This intentionally uses a synthetic one-shard checkpoint.  It exercises the
real safetensor reader, cache, quantized expert banks, and speculative executor
without loading a production model.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import numpy as np
from mlx_lm.models.switch_layers import SwiGLU

from omlx.expert_streaming.execution import SpeculativeExecution
from omlx.expert_streaming.pool import StreamingSwitchGLU
from omlx.expert_streaming.safetensors import ExpertReader, SafetensorExpertIndex

PROJECTIONS = (
    ("gate_proj", "intermediate", "hidden"),
    ("up_proj", "intermediate", "hidden"),
    ("down_proj", "hidden", "intermediate"),
)


def _checkpoint(
    path: Path,
    *,
    experts: int,
    hidden: int,
    intermediate: int,
) -> None:
    tensors: dict[str, mx.array] = {}
    dimensions = {"hidden": hidden, "intermediate": intermediate}
    for projection, output_name, input_name in PROJECTIONS:
        dense = (
            mx.random.normal((experts, dimensions[output_name], dimensions[input_name]))
            / dimensions[input_name] ** 0.5
        )
        dense = dense.astype(mx.float16)
        weight, scales, biases = mx.quantize(
            dense, group_size=32, bits=4, mode="affine"
        )
        prefix = f"language_model.model.layers.0.mlp.switch_mlp.{projection}"
        tensors[f"{prefix}.weight"] = weight
        tensors[f"{prefix}.scales"] = scales
        tensors[f"{prefix}.biases"] = biases
    shard = "model.safetensors"
    mx.save_safetensors(str(path / shard), tensors, metadata={"format": "mlx"})
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: shard for key in tensors}})
    )


def _pool(
    index: SafetensorExpertIndex,
    reader: ExpertReader,
    *,
    experts: int,
    top_k: int,
    cache_slots: int,
) -> StreamingSwitchGLU:
    return StreamingSwitchGLU(
        layer=0,
        num_experts=experts,
        top_k=top_k,
        pinned_experts=(),
        cache_slots=cache_slots,
        locations=index.layer(0),
        projection_metadata={
            name: {"group_size": 32, "bits": 4, "mode": "affine"}
            for name in ("gate_proj", "up_proj", "down_proj")
        },
        activation=SwiGLU(),
        reader=reader,
    )


def _timed(call: Callable[[], Any]) -> tuple[Any, float, int, int]:
    mx.reset_peak_memory()
    active = int(mx.get_active_memory())
    started = time.perf_counter()
    result = call()
    arrays = result if isinstance(result, (list, tuple)) else [result]
    mx.eval(*arrays)
    mx.synchronize()
    elapsed = time.perf_counter() - started
    peak = int(mx.get_peak_memory())
    return result, elapsed, active, peak


def _preload(pool: StreamingSwitchGLU, count: int) -> None:
    pool.preload_hotlist([(expert, count - expert) for expert in range(count)])


def _eager_promote(pool: StreamingSwitchGLU, values: tuple[int, ...]) -> None:
    """Reproduce the former eager bank commit for the comparison baseline."""
    pool.promote(values)
    arrays = [
        pool._array(projection, part)
        for projection, _, _ in PROJECTIONS
        for part in pool.projection_metadata[projection]["parts"]
    ]
    mx.eval(*arrays, pool._slot_map, pool._resident_mask)


def _promotion_benchmark(
    index: SafetensorExpertIndex,
    *,
    experts: int,
    hidden: int,
    top_k: int,
) -> dict[str, Any]:
    del hidden
    readers = [ExpertReader(index), ExpertReader(index)]
    eager = _pool(index, readers[0], experts=experts, top_k=top_k, cache_slots=128)
    lazy = _pool(index, readers[1], experts=experts, top_k=top_k, cache_slots=128)
    try:
        _preload(eager, 128)
        _preload(lazy, 128)
        x = mx.random.normal((1, 1, eager.gate_proj.input_dims)).astype(mx.float16)
        scores = mx.full((1, 1, top_k), 1.0 / top_k, dtype=mx.float16)

        def one(
            pool: StreamingSwitchGLU,
            promote: Callable[[StreamingSwitchGLU, tuple[int, ...]], None],
            group: tuple[int, ...],
        ) -> mx.array:
            promote(pool, group)
            indices = mx.array([[group]], dtype=mx.int32)
            return pool(x, indices, scores=scores, weighted_sum=True)

        # Compile the lazy-scatter and retry shapes before measurement.
        warm_group = tuple(range(138, 138 + top_k))
        mx.eval(one(eager, _eager_promote, warm_group))
        mx.eval(one(lazy, lambda pool, values: pool.promote(values), warm_group))

        groups = [
            tuple(range(128, 128 + top_k)),
            tuple(range(118, 118 + top_k)),
        ] * 10

        def run(
            pool: StreamingSwitchGLU,
            promote: Callable[[StreamingSwitchGLU, tuple[int, ...]], None],
        ) -> list[mx.array]:
            outputs = []
            for group in groups:
                output = one(pool, promote, group)
                mx.eval(output)
                outputs.append(output)
            return outputs

        eager_outputs, eager_s, eager_active, eager_peak = _timed(
            lambda: run(eager, _eager_promote)
        )
        lazy_outputs, lazy_s, lazy_active, lazy_peak = _timed(
            lambda: run(lazy, lambda pool, values: pool.promote(values))
        )
        exact = all(
            bool(mx.all(left == right).item())
            for left, right in zip(eager_outputs, lazy_outputs, strict=True)
        )
        return {
            "exact": exact,
            "iterations": len(groups),
            "eager_ms_per_miss": eager_s * 1000 / len(groups),
            "lazy_ms_per_miss": lazy_s * 1000 / len(groups),
            "speedup": eager_s / lazy_s,
            "eager_peak_delta_mib": (eager_peak - eager_active) / 1024**2,
            "lazy_peak_delta_mib": (lazy_peak - lazy_active) / 1024**2,
        }
    finally:
        for reader in readers:
            reader.close()


def _capped_forward(
    pool: StreamingSwitchGLU,
    x: mx.array,
    indices: mx.array,
    scores: mx.array,
    *,
    group_size: int,
) -> mx.array:
    values = pool._flatten_indices(indices)
    configured = pool.cache_slots
    try:
        pool.cache_slots = group_size
        output = pool._forward_expert_major(x, indices, values)
    finally:
        pool.cache_slots = configured
    return (output * scores[..., None].astype(output.dtype)).sum(-2)


def _prefill_benchmark(
    index: SafetensorExpertIndex,
    *,
    experts: int,
    hidden: int,
    top_k: int,
    tokens: int,
    layers: int,
    repeats: int,
) -> dict[str, Any]:
    readers = [ExpertReader(index) for _ in range(4)]
    pools = {
        "cache_group": _pool(
            index, readers[0], experts=experts, top_k=top_k, cache_slots=128
        ),
        "capped_64": _pool(
            index, readers[1], experts=experts, top_k=top_k, cache_slots=128
        ),
        "capped_96": _pool(
            index, readers[2], experts=experts, top_k=top_k, cache_slots=128
        ),
        "capped_32": _pool(
            index, readers[3], experts=experts, top_k=top_k, cache_slots=128
        ),
    }
    try:
        for pool in pools.values():
            _preload(pool, 128)
        indices_np = np.fromfunction(
            lambda _batch, token, route: (token * top_k + route * 17) % 128,
            (1, tokens, top_k),
            dtype=int,
        ).astype(np.int32)
        indices = mx.array(indices_np)
        scores = mx.full((1, tokens, top_k), 1.0 / top_k, dtype=mx.float16)
        initial = mx.random.normal((1, tokens, hidden)).astype(mx.float16)

        def stack(pool: StreamingSwitchGLU, *, cap: int | None) -> mx.array:
            value = initial
            for _ in range(layers):
                value = (
                    pool(value, indices, scores=scores, weighted_sum=True)
                    if cap is None
                    else _capped_forward(pool, value, indices, scores, group_size=cap)
                )
            return value

        mx.eval(stack(pools["cache_group"], cap=None))
        mx.eval(stack(pools["capped_96"], cap=96))
        mx.eval(stack(pools["capped_64"], cap=64))
        mx.eval(stack(pools["capped_32"], cap=32))

        def repeated(pool: StreamingSwitchGLU, cap: int | None) -> list[mx.array]:
            outputs = []
            for _ in range(repeats):
                output = stack(pool, cap=cap)
                mx.eval(output)
                outputs.append(output)
            return outputs

        measurements = {
            "cache_group": _timed(lambda: repeated(pools["cache_group"], None)),
            "capped_96": _timed(lambda: repeated(pools["capped_96"], 96)),
            "capped_64": _timed(lambda: repeated(pools["capped_64"], 64)),
            "capped_32": _timed(lambda: repeated(pools["capped_32"], 32)),
        }
        wide_outputs = measurements["cache_group"][0]
        exact = {
            name: all(
                bool(mx.all(left == right).item())
                for left, right in zip(wide_outputs, measurement[0], strict=True)
            )
            for name, measurement in measurements.items()
            if name != "cache_group"
        }
        work_tokens = tokens * repeats
        return {
            "exact": exact,
            "tokens": tokens,
            "layers": layers,
            "repeats": repeats,
            "variants": {
                name: {
                    "tokens_per_second": work_tokens / measurement[1],
                    "relative_throughput": (
                        measurements["cache_group"][1] / measurement[1]
                    ),
                    "peak_delta_mib": (measurement[3] - measurement[2]) / 1024**2,
                }
                for name, measurement in measurements.items()
            },
        }
    finally:
        for reader in readers:
            reader.close()


class _Cache:
    def __init__(self):
        self.offset = 0


class _DecodeModel:
    def __init__(
        self,
        pools: list[StreamingSwitchGLU],
        top_k: int,
        *,
        route_groups: int = 4,
    ):
        self.pools = pools
        self.top_k = top_k
        self.route_groups = route_groups

    def __call__(self, input_ids: mx.array, cache: _Cache | None = None):
        token = int(input_ids.item())
        start = (token % self.route_groups) * self.top_k
        indices = mx.array(
            [[[start + route for route in range(self.top_k)]]], dtype=mx.int32
        )
        scores = mx.full((1, 1, self.top_k), 1.0 / self.top_k, dtype=mx.float16)
        value = mx.full(
            (1, 1, self.pools[0].gate_proj.input_dims),
            (token + 1) / 32,
            dtype=mx.float16,
        )
        if cache is not None:
            cache.offset += 1
        for pool in self.pools:
            value = pool(value, indices, scores=scores, weighted_sum=True)
        return SimpleNamespace(logits=value)


class _UngatedSpeculativeExecution(SpeculativeExecution):
    """Former behavior retained only as a microbenchmark baseline."""

    def _cache_ready_for_speculation(self) -> bool:
        return True


class _WorkingSetSpeculativeExecution(SpeculativeExecution):
    """Prototype readiness based on observed resident-route coverage."""

    def __init__(
        self,
        runtime: Any,
        *,
        policy: str,
        min_observed_tokens: int = 4,
        minimum_coverage: float = 0.95,
    ):
        super().__init__(runtime, policy=policy)
        self.min_observed_tokens = min_observed_tokens
        self.minimum_coverage = minimum_coverage

    def _cache_ready_for_speculation(self) -> bool:
        if self._cache_fill_ready:
            return True
        for pool in self.runtime.pools:
            if pool._route_tokens < self.min_observed_tokens:
                return False
            total = int(np.sum(pool._route_hotness, dtype=np.uint64))
            if total <= 0:
                return False
            resident = int(
                np.sum(
                    pool._route_hotness[pool._resident_mask_np],
                    dtype=np.uint64,
                )
            )
            if resident / total < self.minimum_coverage:
                return False
        self._cache_fill_ready = True
        return True


def _reset_pool(pool: StreamingSwitchGLU) -> None:
    pool._expert_to_slot.clear()
    pool._dynamic_lru.clear()
    pool._resident_mask_np[:] = False
    pool._slot_map_np[:] = 0
    pool._resident_mask = mx.array(pool._resident_mask_np)
    pool._slot_map = mx.array(pool._slot_map_np)
    pool._free_slots = list(range(pool.pinned_count, pool.pool_size))
    pool._route_hotness[:] = 0
    pool._route_counts[:] = 0
    pool._last_used[:] = 0
    pool._route_tokens = 0
    pool._next_hotness_decay = pool._hotness_decay_interval
    pool._access_clock = 0
    pool._last_indices = None
    pool._last_slots = None
    pool.stats = type(pool.stats)()


def _decode_variant(
    model: _DecodeModel,
    execution: SpeculativeExecution,
    *,
    tokens: int,
) -> tuple[list[mx.array], float, int, int, dict[str, int]]:
    cache = _Cache()
    # Compile the model and speculative wrapper, then return to a cold cache.
    mx.eval(model(mx.array([[3]], dtype=mx.int32), cache=cache).logits)
    for pool in model.pools:
        _reset_pool(pool)
    execution._has_checked_pass = False
    execution._cache_fill_ready = False
    execution.stats = type(execution.stats)()
    cache.offset = 0

    def run() -> list[mx.array]:
        outputs = []
        for token in range(tokens):
            result = model(mx.array([[token]], dtype=mx.int32), cache=cache)
            mx.eval(result.logits)
            outputs.append(result.logits)
        return outputs

    outputs, elapsed, active, peak = _timed(run)
    assert cache.offset == tokens
    return outputs, elapsed, active, peak, execution.stats.as_dict()


def _fill_gated_speculation_benchmark(
    index: SafetensorExpertIndex,
    *,
    experts: int,
    top_k: int,
    tokens: int,
) -> dict[str, Any]:
    readers = [ExpertReader(index), ExpertReader(index)]
    pool_sets = [
        [
            _pool(index, reader, experts=experts, top_k=top_k, cache_slots=40)
            for _ in range(4)
        ]
        for reader in readers
    ]
    runtimes = [SimpleNamespace(pools=pools) for pools in pool_sets]
    models = [_DecodeModel(pools, top_k) for pools in pool_sets]
    executions = [
        _UngatedSpeculativeExecution(runtimes[0], policy="speculative"),
        SpeculativeExecution(runtimes[1], policy="speculative"),
    ]
    for execution, model in zip(executions, models, strict=True):
        execution.attach(model)
    try:
        ungated = _decode_variant(models[0], executions[0], tokens=tokens)
        gated = _decode_variant(models[1], executions[1], tokens=tokens)
        ungated_outputs, ungated_s, ungated_active, ungated_peak, ungated_stats = (
            ungated
        )
        gated_outputs, gated_s, gated_active, gated_peak, gated_stats = gated
        exact = all(
            bool(mx.all(left == right).item())
            for left, right in zip(ungated_outputs, gated_outputs, strict=True)
        )
        return {
            "exact": exact,
            "tokens": tokens,
            "ungated_tokens_per_second": tokens / ungated_s,
            "fill_gated_tokens_per_second": tokens / gated_s,
            "speedup": ungated_s / gated_s,
            "ungated_peak_delta_mib": (ungated_peak - ungated_active) / 1024**2,
            "fill_gated_peak_delta_mib": (gated_peak - gated_active) / 1024**2,
            "ungated_execution": ungated_stats,
            "fill_gated_execution": gated_stats,
        }
    finally:
        for execution in executions:
            execution.close()
        for reader in readers:
            reader.close()


def _capacity_scaling_benchmark(
    index: SafetensorExpertIndex,
    *,
    experts: int,
    top_k: int,
    tokens: int,
    prefill_tokens: int,
) -> dict[str, Any]:
    """Isolate hot-bank width and capacity-independent speculation readiness."""
    readers = [ExpertReader(index) for _ in range(4)]
    hot_pool_sets = [
        [
            _pool(
                index,
                readers[variant],
                experts=experts,
                top_k=top_k,
                cache_slots=capacity,
            )
            for _ in range(4)
        ]
        for variant, capacity in enumerate((32, 128))
    ]
    gate_pool_sets = [
        [
            _pool(
                index,
                readers[variant + 2],
                experts=experts,
                top_k=top_k,
                cache_slots=128,
            )
            for _ in range(4)
        ]
        for variant in range(2)
    ]
    hot_models = [_DecodeModel(pools, top_k, route_groups=3) for pools in hot_pool_sets]
    gate_models = [_DecodeModel(pools, top_k) for pools in gate_pool_sets]
    gate_runtimes = [SimpleNamespace(pools=pools) for pools in gate_pool_sets]
    gate_executions = [
        SpeculativeExecution(gate_runtimes[0], policy="speculative"),
        _WorkingSetSpeculativeExecution(
            gate_runtimes[1],
            policy="speculative",
            min_observed_tokens=4,
        ),
    ]
    for execution, model in zip(gate_executions, gate_models, strict=True):
        execution.attach(model)

    try:
        hot_count = top_k * 3
        for pools in hot_pool_sets:
            for pool in pools:
                _preload(pool, hot_count)

        def hot_run(model: _DecodeModel, run_tokens: int = tokens) -> list[mx.array]:
            cache = _Cache()
            outputs = []
            for token in range(run_tokens):
                result = model(mx.array([[token]], dtype=mx.int32), cache=cache)
                mx.eval(result.logits)
                outputs.append(result.logits)
            assert cache.offset == run_tokens
            return outputs

        # Compile both fixed bank shapes before measuring identical hot routes.
        mx.eval(hot_models[0](mx.array([[0]], dtype=mx.int32)).logits)
        mx.eval(hot_models[1](mx.array([[0]], dtype=mx.int32)).logits)
        hot_32 = _timed(lambda: hot_run(hot_models[0]))
        hot_128 = _timed(lambda: hot_run(hot_models[1]))
        hot_exact = all(
            bool(mx.all(left == right).item())
            for left, right in zip(hot_32[0], hot_128[0], strict=True)
        )

        prefill_indices_np = np.fromfunction(
            lambda _batch, token, route: (token * top_k + route * 17) % 100,
            (1, prefill_tokens, top_k),
            dtype=int,
        ).astype(np.int32)
        prefill_indices = mx.array(prefill_indices_np)
        prefill_scores = mx.full(
            (1, prefill_tokens, top_k),
            1.0 / top_k,
            dtype=mx.float16,
        )
        prefill_input = mx.random.normal(
            (1, prefill_tokens, hot_pool_sets[0][0].gate_proj.input_dims)
        ).astype(mx.float16)

        def prefill_stack(pool: StreamingSwitchGLU) -> mx.array:
            value = prefill_input
            for _ in range(4):
                value = pool(
                    value,
                    prefill_indices,
                    scores=prefill_scores,
                    weighted_sum=True,
                )
            return value

        mx.eval(prefill_stack(hot_pool_sets[0][0]))
        mx.eval(prefill_stack(hot_pool_sets[1][0]))
        prefill_32 = _timed(lambda: prefill_stack(hot_pool_sets[0][0]))
        prefill_128 = _timed(lambda: prefill_stack(hot_pool_sets[1][0]))
        prefill_exact = bool(mx.all(prefill_32[0] == prefill_128[0]).item())

        full_gate = _decode_variant(gate_models[0], gate_executions[0], tokens=tokens)
        working_set_gate = _decode_variant(
            gate_models[1], gate_executions[1], tokens=tokens
        )
        gate_exact = all(
            bool(mx.all(left == right).item())
            for left, right in zip(full_gate[0], working_set_gate[0], strict=True)
        )

        expanding_sparse = _DecodeModel(gate_pool_sets[0], top_k, route_groups=12)
        expanding_full = _DecodeModel(hot_pool_sets[1], top_k, route_groups=12)
        expansion_tokens = 12
        sparse_bytes_before = readers[2].file_cache_bytes_read
        sparse_loads_before = sum(pool.stats.loads for pool in gate_pool_sets[0])
        sparse_expansion = _timed(lambda: hot_run(expanding_sparse, expansion_tokens))
        sparse_bytes = readers[2].file_cache_bytes_read - sparse_bytes_before
        sparse_loads = (
            sum(pool.stats.loads for pool in gate_pool_sets[0]) - sparse_loads_before
        )
        full_bytes_before = readers[1].file_cache_bytes_read
        full_loads_before = sum(pool.stats.loads for pool in hot_pool_sets[1])
        full_expansion = _timed(lambda: hot_run(expanding_full, expansion_tokens))
        full_bytes = readers[1].file_cache_bytes_read - full_bytes_before
        full_loads = (
            sum(pool.stats.loads for pool in hot_pool_sets[1]) - full_loads_before
        )
        expansion_exact = all(
            bool(mx.all(left == right).item())
            for left, right in zip(sparse_expansion[0], full_expansion[0], strict=True)
        )

        return {
            "exact": {
                "hot_capacity_32_vs_128": hot_exact,
                "prefill_capacity_32_vs_128": prefill_exact,
                "full_vs_working_set_gate": gate_exact,
                "sparse_vs_optimistic_fill": expansion_exact,
            },
            "tokens": tokens,
            "hot_decode": {
                "capacity_32_tokens_per_second": tokens / hot_32[1],
                "capacity_128_tokens_per_second": tokens / hot_128[1],
                "capacity_128_relative_throughput": hot_32[1] / hot_128[1],
                "capacity_32_peak_delta_mib": (hot_32[3] - hot_32[2]) / 1024**2,
                "capacity_128_peak_delta_mib": (hot_128[3] - hot_128[2]) / 1024**2,
            },
            "prefill": {
                "tokens": prefill_tokens,
                "capacity_32_tokens_per_second": prefill_tokens / prefill_32[1],
                "capacity_128_tokens_per_second": prefill_tokens / prefill_128[1],
                "capacity_128_relative_throughput": prefill_32[1] / prefill_128[1],
                "capacity_32_peak_delta_mib": (prefill_32[3] - prefill_32[2]) / 1024**2,
                "capacity_128_peak_delta_mib": (prefill_128[3] - prefill_128[2])
                / 1024**2,
            },
            "speculation_readiness_at_128": {
                "full_gate_tokens_per_second": tokens / full_gate[1],
                "working_set_gate_tokens_per_second": tokens / working_set_gate[1],
                "working_set_speedup": full_gate[1] / working_set_gate[1],
                "full_gate_execution": full_gate[4],
                "working_set_execution": working_set_gate[4],
            },
            "optimistic_fill_at_128": {
                "tokens": expansion_tokens,
                "sparse_tokens_per_second": expansion_tokens / sparse_expansion[1],
                "full_tokens_per_second": expansion_tokens / full_expansion[1],
                "speedup": sparse_expansion[1] / full_expansion[1],
                "sparse_cold_loads": sparse_loads,
                "full_cold_loads": full_loads,
                "sparse_ssd_bytes": sparse_bytes,
                "full_ssd_bytes": full_bytes,
            },
        }
    finally:
        for execution in gate_executions:
            execution.close()
        for reader in readers:
            reader.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=160)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--intermediate", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--prefill-tokens", type=int, default=256)
    parser.add_argument("--prefill-layers", type=int, default=12)
    parser.add_argument("--prefill-repeats", type=int, default=2)
    parser.add_argument("--decode-tokens", type=int, default=12)
    parser.add_argument("--capacity-decode-tokens", type=int, default=128)
    parser.add_argument("--capacity-prefill-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.experts < 160:
        parser.error("--experts must be at least 160")
    if args.top_k > 10:
        parser.error("--top-k must be at most 10 for the default route fixtures")

    mx.random.seed(0)

    with tempfile.TemporaryDirectory(prefix="omlx-expert-microbench-") as temporary:
        path = Path(temporary)
        _checkpoint(
            path,
            experts=args.experts,
            hidden=args.hidden,
            intermediate=args.intermediate,
        )
        index = SafetensorExpertIndex(path)
        started = time.perf_counter()
        report = {
            "geometry": {
                "experts": args.experts,
                "hidden": args.hidden,
                "intermediate": args.intermediate,
                "top_k": args.top_k,
            },
            "lazy_promotion": _promotion_benchmark(
                index,
                experts=args.experts,
                hidden=args.hidden,
                top_k=args.top_k,
            ),
            "prefill_group_cap": _prefill_benchmark(
                index,
                experts=args.experts,
                hidden=args.hidden,
                top_k=args.top_k,
                tokens=args.prefill_tokens,
                layers=args.prefill_layers,
                repeats=args.prefill_repeats,
            ),
            "fill_gated_speculation": _fill_gated_speculation_benchmark(
                index,
                experts=args.experts,
                top_k=args.top_k,
                tokens=args.decode_tokens,
            ),
            "capacity_scaling": _capacity_scaling_benchmark(
                index,
                experts=args.experts,
                top_k=args.top_k,
                tokens=args.capacity_decode_tokens,
                prefill_tokens=args.capacity_prefill_tokens,
            ),
        }
        report["wall_seconds"] = time.perf_counter() - started
        report["peak_mib"] = mx.get_peak_memory() / 1024**2
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
