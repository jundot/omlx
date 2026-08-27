#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Isolated rank-local DeepSeek-V4 TP prefill/verify stage profiler.

This harness is deliberately absent from the serving path.  It lazy-loads one
real checkpoint layer for each DS4 attention schedule, slices those layers to
an explicit TP vector (3:5 by default), and runs full block forwards with
identity transport.  Bounded synchronization brackets attribute GPU wall to:

* attention and compressor projections;
* routed-MoE gate/up pair work;
* routed-MoE down work;
* sparse indexer work; and
* everything else in the block.

The real two-rank collective cost is kept separate and modeled from measured
JACCL bandwidth/latency.  Finally, the representative-layer mix is normalized
to an observed end-to-end prefill rate and rendered as an Amdahl table.  No
production class is patched until ``main`` explicitly enters the isolated
profiling context, and every patch is restored before the process exits.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Mapping, Sequence


CATEGORIES = (
    "attention_projections",
    "routed_moe_pair",
    "routed_moe_down",
    "indexer",
    "collectives",
    "misc",
)
COMPUTE_CATEGORIES = tuple(item for item in CATEGORIES if item != "collectives")
ATTENTION_PROJECTION_DETAILS = (
    "q_a",
    "q_b",
    "raw_wkv",
    "compressor_wkv",
    "compressor_gate",
    "o_a",
    "o_b",
)
INDEXER_PROJECTION_DETAILS = (
    "indexer_q",
    "indexer_weights",
    "indexer_compressor_wkv",
    "indexer_compressor_gate",
)
PROJECTION_DETAIL_CATEGORIES = (
    *ATTENTION_PROJECTION_DETAILS,
    *INDEXER_PROJECTION_DETAILS,
)
FINE_DETAIL_CATEGORIES = (
    "router",
    "shared_moe",
    "hyperconnection",
    "norms",
    "attention_qkv_bank",
    "attention_q_b",
    "attention_core",
    "attention_output",
)
PROFILE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DS4TPShape:
    """The exact local geometry for one rank of an 8-unit DS4 TP split."""

    rank: int
    shard_weights: tuple[int, ...] = (3, 5)
    tokens: int = 1024
    prefix_tokens: int = 8192
    hidden: int = 4096
    heads: int = 64
    head_dim: int = 512
    o_groups: int = 8
    intermediate: int = 2048
    topk: int = 6
    index_topk: int = 512
    layers: int = 43
    ratio4_layers: int = 21
    element_bytes: int = 2

    def __post_init__(self) -> None:
        if not self.shard_weights or any(weight < 1 for weight in self.shard_weights):
            raise ValueError("shard weights must be positive")
        if sum(self.shard_weights) != self.heads // self.o_groups:
            raise ValueError("DS4 shard weights must sum to eight head units")
        if not 0 <= self.rank < len(self.shard_weights):
            raise ValueError("rank is outside the TP vector")
        if self.tokens < 1 or self.prefix_tokens < 0:
            raise ValueError("token counts must be non-negative and tokens positive")

    @property
    def local_units(self) -> int:
        return self.shard_weights[self.rank]

    @property
    def local_heads(self) -> int:
        return self.o_groups * self.local_units

    @property
    def local_intermediate(self) -> int:
        return self.intermediate * self.local_units // sum(self.shard_weights)

    @property
    def route_rows(self) -> int:
        return self.tokens * self.topk

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            {
                "local_units": self.local_units,
                "local_heads": self.local_heads,
                "local_intermediate": self.local_intermediate,
                "route_rows": self.route_rows,
            }
        )
        return result


def modeled_collective_ms(
    shape: DS4TPShape,
    *,
    bandwidth_gbps: float = 6.2,
    latency_us: float = 30.0,
    include_indexer_gathers: bool = True,
) -> dict[str, float]:
    """Model one TP2 chunk's mandatory traffic from measured link figures."""

    if bandwidth_gbps <= 0 or latency_us < 0:
        raise ValueError(
            "collective bandwidth must be positive and latency non-negative"
        )
    if len(shape.shard_weights) != 2:
        raise ValueError("the current collective model is calibrated for TP2")

    activation_bytes = shape.tokens * shape.hidden * shape.element_bytes
    one_activation_ms = (
        latency_us / 1000.0 + activation_bytes / (bandwidth_gbps * 1e9) * 1000.0
    )
    activation_calls = 2 * shape.layers
    activation_ms = activation_calls * one_activation_ms

    # Ratio-4 indexer row sharding activates once the existing pooled prefix
    # reaches 2,048 rows (8,192 source tokens). Each rank contributes half the
    # query rows, each carrying exact uint32 top-k indices.
    indexer_calls = 0
    indexer_payload_bytes = 0
    if include_indexer_gathers and shape.prefix_tokens // 4 >= 2048:
        indexer_calls = shape.ratio4_layers
        local_rows = (shape.tokens + len(shape.shard_weights) - 1) // len(
            shape.shard_weights
        )
        indexer_payload_bytes = local_rows * shape.index_topk * 4
    one_indexer_ms = (
        latency_us / 1000.0
        + indexer_payload_bytes / (bandwidth_gbps * 1e9) * 1000.0
        if indexer_calls
        else 0.0
    )
    indexer_ms = indexer_calls * one_indexer_ms
    return {
        "activation_all_sum_calls": float(activation_calls),
        "activation_payload_bytes": float(activation_bytes),
        "activation_ms": activation_ms,
        "indexer_all_gather_calls": float(indexer_calls),
        "indexer_payload_bytes_per_rank": float(indexer_payload_bytes),
        "indexer_ms": indexer_ms,
        "total_ms": activation_ms + indexer_ms,
    }


def _group_speedup_row(
    name: str,
    categories: Sequence[str],
    fractions: Mapping[str, float],
    *,
    baseline_tps: float,
    target_tps: float,
) -> dict[str, Any]:
    fraction = sum(float(fractions.get(category, 0.0)) for category in categories)
    target_time = baseline_tps / target_tps
    remaining = 1.0 - fraction
    denominator = target_time - remaining
    required = fraction / denominator if denominator > 0 else None
    doubled_time = remaining + fraction / 2.0
    infinite_time = remaining
    return {
        "component": name,
        "categories": list(categories),
        "wall_fraction": fraction,
        "required_speedup_for_target": required,
        "target_possible_alone": required is not None,
        "tps_at_2x": baseline_tps / doubled_time,
        "infinite_ceiling_tps": (
            baseline_tps / infinite_time if infinite_time > 0 else None
        ),
    }


def amdahl_table(
    fractions: Mapping[str, float],
    *,
    baseline_tps: float,
    target_tps: float,
) -> list[dict[str, Any]]:
    """Return single-component and useful combined Amdahl interventions."""

    if baseline_tps <= 0 or target_tps <= baseline_tps:
        raise ValueError("target TPS must be greater than a positive baseline")
    if set(fractions) != set(CATEGORIES):
        raise ValueError(
            "category fractions must contain exactly the profiler schema"
        )
    total = sum(float(fractions.get(category, 0.0)) for category in CATEGORIES)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"category fractions must sum to one, got {total}")

    rows = [
        _group_speedup_row(
            category,
            (category,),
            fractions,
            baseline_tps=baseline_tps,
            target_tps=target_tps,
        )
        for category in CATEGORIES
    ]
    rows.extend(
        [
            _group_speedup_row(
                "routed_moe_total",
                ("routed_moe_pair", "routed_moe_down"),
                fractions,
                baseline_tps=baseline_tps,
                target_tps=target_tps,
            ),
            _group_speedup_row(
                "kernel_hotset",
                (
                    "attention_projections",
                    "routed_moe_pair",
                    "routed_moe_down",
                    "indexer",
                ),
                fractions,
                baseline_tps=baseline_tps,
                target_tps=target_tps,
            ),
            _group_speedup_row(
                "kernel_hotset_plus_collectives",
                (
                    "attention_projections",
                    "routed_moe_pair",
                    "routed_moe_down",
                    "indexer",
                    "collectives",
                ),
                fractions,
                baseline_tps=baseline_tps,
                target_tps=target_tps,
            ),
        ]
    )
    return rows


def projected_tps(
    fractions: Mapping[str, float],
    speedups: Mapping[str, float],
    *,
    baseline_tps: float,
) -> float:
    """Project throughput for mutually exclusive category speedups."""

    import math

    if baseline_tps <= 0 or set(fractions) != set(CATEGORIES):
        raise ValueError("projection requires a positive baseline and full schema")
    if not set(speedups).issubset(CATEGORIES):
        raise ValueError("speedup map contains an unknown category")
    relative_wall = 0.0
    for category in CATEGORIES:
        speedup = float(speedups.get(category, 1.0))
        if speedup <= 0:
            raise ValueError("component speedups must be positive")
        if math.isinf(speedup):
            continue
        relative_wall += float(fractions[category]) / speedup
    return baseline_tps / relative_wall


def required_group_speedup(
    fractions: Mapping[str, float],
    categories: Sequence[str],
    *,
    baseline_tps: float,
    target_tps: float,
    fixed_speedups: Mapping[str, float] | None = None,
) -> float | None:
    """Solve one shared group factor after applying fixed category changes."""

    import math

    selected = set(categories)
    if not selected or not selected.issubset(CATEGORIES):
        raise ValueError("speedup group contains an unknown category")
    fixed_speedups = fixed_speedups or {}
    fixed_wall = 0.0
    group_wall = 0.0
    for category in CATEGORIES:
        fraction = float(fractions[category])
        if category in selected:
            group_wall += fraction
            continue
        speedup = float(fixed_speedups.get(category, 1.0))
        if speedup <= 0:
            raise ValueError("component speedups must be positive")
        if not math.isinf(speedup):
            fixed_wall += fraction / speedup
    budget = baseline_tps / target_tps - fixed_wall
    return group_wall / budget if budget > 0 else None


def optimization_scenarios(
    fractions: Mapping[str, float],
    *,
    baseline_tps: float,
    target_tps: float,
) -> list[dict[str, Any]]:
    """Concrete joint interventions implied by the measured wall split."""

    import math

    moe = ("routed_moe_pair", "routed_moe_down")
    attention = ("attention_projections",)
    two_each = {
        "attention_projections": 2.0,
        "routed_moe_pair": 2.0,
        "routed_moe_down": 2.0,
    }
    return [
        {
            "scenario": "attention_2x_and_routed_moe_2x",
            "speedups": two_each,
            "projected_tps": projected_tps(
                fractions,
                two_each,
                baseline_tps=baseline_tps,
            ),
        },
        {
            "scenario": "attention_2x_then_required_routed_moe",
            "fixed_speedups": {"attention_projections": 2.0},
            "required_group_speedup": required_group_speedup(
                fractions,
                moe,
                baseline_tps=baseline_tps,
                target_tps=target_tps,
                fixed_speedups={"attention_projections": 2.0},
            ),
        },
        {
            "scenario": "routed_moe_2x_then_required_attention",
            "fixed_speedups": {
                "routed_moe_pair": 2.0,
                "routed_moe_down": 2.0,
            },
            "required_group_speedup": required_group_speedup(
                fractions,
                attention,
                baseline_tps=baseline_tps,
                target_tps=target_tps,
                fixed_speedups={
                    "routed_moe_pair": 2.0,
                    "routed_moe_down": 2.0,
                },
            ),
        },
        {
            "scenario": "hide_collectives_then_required_attention_and_moe",
            "fixed_speedups": {"collectives": "infinite"},
            "required_group_speedup": required_group_speedup(
                fractions,
                (*attention, *moe),
                baseline_tps=baseline_tps,
                target_tps=target_tps,
                fixed_speedups={"collectives": math.inf},
            ),
        },
    ]


def normalize_to_observed_wall(
    compute_ms: Mapping[str, float],
    *,
    collective_ms: float,
    tokens: int,
    baseline_tps: float,
    target_tps: float,
) -> dict[str, Any]:
    """Scale isolated compute proportions onto an observed distributed wall."""

    if tokens < 1 or baseline_tps <= 0 or collective_ms < 0:
        raise ValueError("normalization inputs are invalid")
    wall_ms = tokens / baseline_tps * 1000.0
    if collective_ms >= wall_ms:
        raise ValueError("modeled collectives consume the entire observed wall")
    isolated = {
        category: max(0.0, float(compute_ms.get(category, 0.0)))
        for category in COMPUTE_CATEGORIES
    }
    isolated_total = sum(isolated.values())
    if isolated_total <= 0:
        raise ValueError("isolated compute profile is empty")
    compute_budget = wall_ms - collective_ms
    scale = compute_budget / isolated_total
    attributed_ms = {
        category: isolated[category] * scale for category in COMPUTE_CATEGORIES
    }
    attributed_ms["collectives"] = collective_ms
    fractions = {
        category: attributed_ms[category] / wall_ms for category in CATEGORIES
    }
    return {
        "baseline_tps": baseline_tps,
        "target_tps": target_tps,
        "tokens": tokens,
        "observed_wall_ms": wall_ms,
        "isolated_compute_ms": isolated,
        "isolated_to_observed_scale": scale,
        "attributed_ms": attributed_ms,
        "fractions": fractions,
        "amdahl": amdahl_table(
            fractions,
            baseline_tps=baseline_tps,
            target_tps=target_tps,
        ),
        "optimization_scenarios": optimization_scenarios(
            fractions,
            baseline_tps=baseline_tps,
            target_tps=target_tps,
        ),
    }


class _LocalGroup:
    def __init__(self, rank: int, size: int) -> None:
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


def _array_leaves(value: Any) -> list[Any]:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    if isinstance(value, mx.array):
        return [value]
    try:
        return [
            item
            for _path, item in tree_flatten(value)
            if isinstance(item, mx.array)
        ]
    except (TypeError, ValueError):
        return []


def _evaluate_array_leaves(value: Any) -> None:
    import mlx.core as mx

    leaves = _array_leaves(value)
    if leaves:
        mx.eval(*leaves)


class TimingRecorder:
    """Exclusive GPU-wall brackets used only inside this benchmark process."""

    def __init__(
        self,
        *,
        synchronize: Callable[[], None],
        evaluate: Callable[[Any], None],
        clock_ns: Callable[[], int] = time.perf_counter_ns,
        barrier_overhead_ns: int = 0,
    ) -> None:
        self.synchronize = synchronize
        self.evaluate = evaluate
        self.clock_ns = clock_ns
        self.barrier_overhead_ns = max(0, int(barrier_overhead_ns))
        self.active = False
        self.depth = 0
        self.nanoseconds: defaultdict[str, int] = defaultdict(int)
        self.calls: defaultdict[str, int] = defaultdict(int)

    def reset(self) -> None:
        self.nanoseconds.clear()
        self.calls.clear()

    def call(
        self,
        category: str,
        function: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ):
        if not self.active or self.depth:
            return function(*args, **kwargs)
        self.synchronize()
        started = self.clock_ns()
        self.depth += 1
        try:
            result = function(*args, **kwargs)
            self.evaluate(result)
            self.synchronize()
        finally:
            self.depth -= 1
        elapsed = max(0, self.clock_ns() - started - self.barrier_overhead_ns)
        self.nanoseconds[category] += elapsed
        self.calls[category] += 1
        return result

    def milliseconds(self) -> dict[str, float]:
        return {
            category: value / 1e6 for category, value in self.nanoseconds.items()
        }


class DS4LayerInstrumentation:
    """Temporarily bracket selected real layer operations by semantic class."""

    _PAIR_SYMBOLS = (
        "deepseek_mxfp4_gather_qmm_pair_blocks",
        "deepseek_mxfp4_gather_qmm_pair_concat_blocks",
        "deepseek_mxfp4_gather_qmm_pair_swiglu_blocks_tail8",
        "deepseek_affine_gather_qmm_pair_concat_blocks",
    )
    _DIRECT_DOWN_SYMBOLS = ("deepseek_mxfp4_gather_qmm_blocks_tail8",)

    def __init__(
        self,
        layer: Any,
        recorder: TimingRecorder,
        *,
        projection_detail: bool = False,
        fine_detail: bool = False,
    ) -> None:
        self.layer = layer
        self.recorder = recorder
        self.projection_detail = projection_detail
        self.fine_detail = fine_detail
        self._stack = ExitStack()
        self._module_categories: dict[int, str] = {}
        self._patched_classes: set[type] = set()

    def _register(self, module: Any, category: str) -> None:
        if module is None:
            return
        self._module_categories[id(module)] = category
        cls = type(module)
        if cls in self._patched_classes:
            return
        original = cls.__call__
        categories = self._module_categories
        recorder = self.recorder

        def profiled(instance: Any, *args: Any, **kwargs: Any):
            selected = categories.get(id(instance))
            if selected is None:
                return original(instance, *args, **kwargs)
            return recorder.call(selected, original, instance, *args, **kwargs)

        setattr(cls, "__call__", profiled)
        self._stack.callback(setattr, cls, "__call__", original)
        self._patched_classes.add(cls)

    def _patch_symbol(self, owner: Any, name: str, category: str) -> None:
        original = getattr(owner, name, None)
        if not callable(original):
            return
        recorder = self.recorder

        def profiled(*args: Any, **kwargs: Any):
            return recorder.call(category, original, *args, **kwargs)

        setattr(owner, name, profiled)
        self._stack.callback(setattr, owner, name, original)

    def __enter__(self):
        import mlx_lm.models.deepseek_v4 as dsv4

        from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast

        attn = self.layer.attn
        attention_labels = (
            ("wq_a", "q_a"),
            ("wq_b", "q_b"),
            ("wkv", "raw_wkv"),
            ("wo_a", "o_a"),
            ("wo_b", "o_b"),
        )
        for name, detail in attention_labels:
            self._register(
                getattr(attn, name, None),
                detail if self.projection_detail else "attention_projections",
            )
        compressor = getattr(attn, "compressor", None)
        if compressor is not None:
            self._register(
                getattr(compressor, "wkv", None),
                "compressor_wkv"
                if self.projection_detail
                else "attention_projections",
            )
            self._register(
                getattr(compressor, "wgate", None),
                "compressor_gate"
                if self.projection_detail
                else "attention_projections",
            )

        indexer = getattr(attn, "indexer", None)
        if indexer is not None:
            # The outer bracket owns its compressor projections, score kernel,
            # deterministic top-k, and compact gather without double counting.
            if not self.projection_detail:
                self._register(indexer, "indexer")
            indexer_labels = (
                (getattr(indexer, "wq_b", None), "indexer_q"),
                (getattr(indexer, "weights_proj", None), "indexer_weights"),
                (
                    getattr(getattr(indexer, "compressor", None), "wkv", None),
                    "indexer_compressor_wkv",
                ),
                (
                    getattr(getattr(indexer, "compressor", None), "wgate", None),
                    "indexer_compressor_gate",
                ),
            )
            for module, detail in indexer_labels:
                self._register(
                    module,
                    detail if self.projection_detail else "indexer",
                )

        switch = self.layer.ffn.switch_mlp
        self._register(switch.up_proj, "routed_moe_pair")
        self._register(switch.gate_proj, "routed_moe_pair")
        self._register(switch.down_proj, "routed_moe_down")
        for symbol in self._PAIR_SYMBOLS:
            self._patch_symbol(glm_fast, symbol, "routed_moe_pair")
        for symbol in self._DIRECT_DOWN_SYMBOLS:
            self._patch_symbol(glm_fast, symbol, "routed_moe_down")
        if self.fine_detail:
            self._register(self.layer.ffn.gate, "router")
            self._register(self.layer.ffn.shared_experts, "shared_moe")
            self._register(self.layer.attn_hc, "hyperconnection")
            self._register(self.layer.ffn_hc, "hyperconnection")
            self._register(self.layer.attn_norm, "norms")
            self._register(self.layer.ffn_norm, "norms")
            for symbol in ("hc_expand", "hc_residual_branch", "hc_merge_branch"):
                self._patch_symbol(dsv4, symbol, "hyperconnection")
            self._patch_symbol(
                dsv4,
                "_verify_q_a_kv_bank",
                "attention_qkv_bank",
            )
            self._patch_symbol(
                dsv4,
                "_project_verify_q_b",
                "attention_q_b",
            )
            self._patch_symbol(
                dsv4,
                "_batched_m1_attention",
                "attention_core",
            )
            for symbol in (
                "scaled_dot_product_attention",
                "wsdpa_prefill",
                "wsdpa_topk_prefill",
            ):
                self._patch_symbol(dsv4, symbol, "attention_core")
            self._patch_symbol(
                dsv4,
                "_project_attention_output",
                "attention_output",
            )
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stack.close()


@contextmanager
def identity_tp_collectives(size: int) -> Iterator[None]:
    """Keep the rank-local forward shape-valid without touching a peer."""

    import mlx.core as mx

    original_sum = mx.distributed.all_sum
    original_gather = mx.distributed.all_gather

    def all_sum(value: Any, *args: Any, **kwargs: Any):
        return value

    def all_gather(value: Any, *args: Any, **kwargs: Any):
        return mx.concatenate([value] * size, axis=0)

    mx.distributed.all_sum = all_sum
    mx.distributed.all_gather = all_gather
    try:
        yield
    finally:
        mx.distributed.all_sum = original_sum
        mx.distributed.all_gather = original_gather


def _measure_barrier_overhead(cycles: int = 25) -> int:
    import mlx.core as mx

    samples = []
    for _ in range(max(3, cycles)):
        mx.synchronize()
        started = time.perf_counter_ns()
        mx.synchronize()
        samples.append(time.perf_counter_ns() - started)
    return int(statistics.median(samples))


def _load_real_layers(
    model_dir: Path,
    layers: Sequence[int],
    shape: DS4TPShape,
) -> tuple[Any, list[Any], dict[int, int]]:
    """Lazy-load then progressively materialize only selected local shards."""

    import mlx.core as mx
    from mlx_lm import utils as mlx_utils

    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    os.environ["OMLX_TP_SHARD_WEIGHTS"] = ",".join(
        str(weight) for weight in shape.shard_weights
    )
    maybe_apply_pre_load_patches(
        model_dir,
        model_settings=SimpleNamespace(mtp_enabled=False, mtp_num_draft_tokens=0),
    )
    model, _config = mlx_utils.load_model(model_dir, lazy=True, strict=False)
    all_layers = list(model.model.layers)
    if any(not 0 <= index < len(all_layers) for index in layers):
        raise ValueError("profile layer is outside the checkpoint")
    selected = [all_layers[index] for index in layers]
    group = _LocalGroup(shape.rank, len(shape.shard_weights))
    for layer in selected:
        model.model.layers = [layer]
        model.shard(group)
        mx.eval(layer.parameters())
        mx.synchronize()
        gc.collect()
        mx.clear_cache()
    model.model.layers = selected
    ratios = {
        index: int(getattr(layer.attn, "compress_ratio", 0))
        for index, layer in zip(layers, selected)
    }
    del all_layers
    gc.collect()
    mx.clear_cache()
    return model, selected, ratios


def _new_layer_cache(model: Any, position: int) -> Any:
    return model.make_cache()[position]


def _layer_call(
    layer: Any,
    cache: Any,
    h: Any,
    input_ids: Any,
    *,
    standard_mask: bool = True,
):
    from mlx_lm.models.base import create_attention_mask

    first = cache[0] if hasattr(cache, "caches") else cache
    mask = create_attention_mask(
        h[:, :, 0, :],
        first,
        window_size=layer.attn.config.sliding_window,
        return_array=True,
    )
    return layer(
        h,
        mask,
        cache,
        input_ids,
        _standard_mask=standard_mask,
    )


def _warm_layer_cache(
    layer: Any,
    cache: Any,
    shape: DS4TPShape,
    *,
    seed: int,
) -> None:
    import mlx.core as mx

    remaining = shape.prefix_tokens
    # Reproduce the serving scheduler's canonical prefill frontier regardless
    # of the measured call width. Using ``shape.tokens`` made a B=1 decode
    # profile warm an 8K cache through 8,192 one-token forwards, which was both
    # prohibitively slow and unlike production's 1K prompt chunks.
    step = min(1024, max(1, remaining))
    iteration = 0
    while remaining > 0:
        width = min(step, remaining)
        mx.random.seed(seed + iteration)
        h = mx.random.normal((1, width, 4, shape.hidden)).astype(mx.bfloat16)
        ids = mx.zeros((1, width), dtype=mx.int32)
        result = _layer_call(layer, cache, h, ids)
        mx.eval(result, cache.state)
        mx.synchronize()
        remaining -= width
        iteration += 1


def profile_real_layers(
    model_dir: Path,
    *,
    shape: DS4TPShape,
    layers: Sequence[int],
    warmup: int,
    cycles: int,
    projection_detail: bool = False,
    dspark_verify: bool = False,
    fine_detail: bool = False,
) -> dict[str, Any]:
    """Run full real-block forwards and return per-layer category medians."""

    import mlx.core as mx

    if warmup < 0 or cycles < 1:
        raise ValueError("warmup must be non-negative and cycles positive")
    model, selected, ratios = _load_real_layers(model_dir, layers, shape)
    if dspark_verify:
        for layer in selected:
            layer.attn._omlx_decode_consistent = True
    barrier_overhead = _measure_barrier_overhead()
    layer_reports: dict[str, Any] = {}

    with identity_tp_collectives(len(shape.shard_weights)):
        for position, (layer_index, layer) in enumerate(zip(layers, selected)):
            samples = []
            recorder = TimingRecorder(
                synchronize=mx.synchronize,
                evaluate=_evaluate_array_leaves,
                barrier_overhead_ns=barrier_overhead,
            )
            with DS4LayerInstrumentation(
                layer,
                recorder,
                projection_detail=projection_detail,
                fine_detail=fine_detail,
            ):
                for cycle in range(warmup + cycles):
                    cache = _new_layer_cache(model, position)
                    _warm_layer_cache(
                        layer,
                        cache,
                        shape,
                        seed=10_000 + layer_index * 100 + cycle,
                    )
                    mx.random.seed(20_000 + layer_index * 100 + cycle)
                    h = mx.random.normal(
                        (1, shape.tokens, 4, shape.hidden)
                    ).astype(mx.bfloat16)
                    ids = mx.zeros((1, shape.tokens), dtype=mx.int32)
                    mx.eval(h, ids)
                    mx.synchronize()
                    recorder.reset()
                    started = time.perf_counter_ns()
                    recorder.active = True
                    try:
                        if dspark_verify:
                            from mlx_lm.models.deepseek_v4 import (
                                set_dspark_verify_armed,
                            )
                            from omlx.patches.mlx_lm_mtp import cache_rollback

                            cache_rollback.set_undo_armed(True)
                            set_dspark_verify_armed(True)
                        output = _layer_call(layer, cache, h, ids)
                        mx.eval(output, cache.state)
                        mx.synchronize()
                    finally:
                        if dspark_verify:
                            set_dspark_verify_armed(False)
                            cache_rollback.set_undo_armed(False)
                        recorder.active = False
                    total_ms = max(
                        0.0,
                        (
                            time.perf_counter_ns()
                            - started
                            # Each bracket has an un-timed leading synchronize
                            # and a timed trailing synchronize. The category
                            # subtracts the latter; remove both empty-barrier
                            # costs from the full wall before deriving misc.
                            - 2 * sum(recorder.calls.values()) * barrier_overhead
                        )
                        / 1e6,
                    )
                    raw_components = recorder.milliseconds()
                    details = {
                        category: raw_components.get(category, 0.0)
                        for category in PROJECTION_DETAIL_CATEGORIES
                    }
                    fine_details = {
                        category: raw_components.get(category, 0.0)
                        for category in FINE_DETAIL_CATEGORIES
                    }
                    if projection_detail:
                        components = {
                            "attention_projections": sum(
                                details[category]
                                for category in ATTENTION_PROJECTION_DETAILS
                            ),
                            "indexer": sum(
                                details[category]
                                for category in INDEXER_PROJECTION_DETAILS
                            ),
                            "routed_moe_pair": raw_components.get(
                                "routed_moe_pair", 0.0
                            ),
                            "routed_moe_down": raw_components.get(
                                "routed_moe_down", 0.0
                            ),
                        }
                    else:
                        components = {
                            category: raw_components.get(category, 0.0)
                            for category in COMPUTE_CATEGORIES
                            if category != "misc"
                        }
                    measured = sum(components.values())
                    components["misc"] = max(0.0, total_ms - measured)
                    if cycle >= warmup:
                        samples.append(
                            {
                                "total_ms": total_ms,
                                "components_ms": components,
                                "projection_details_ms": details,
                                "fine_details_ms": fine_details,
                                "calls": dict(recorder.calls),
                            }
                        )
                    del cache, output
                    mx.clear_cache()

            layer_reports[str(layer_index)] = {
                "compress_ratio": ratios[layer_index],
                "samples": samples,
                "median_total_ms": statistics.median(
                    sample["total_ms"] for sample in samples
                ),
                "median_components_ms": {
                    category: statistics.median(
                        sample["components_ms"].get(category, 0.0)
                        for sample in samples
                    )
                    for category in COMPUTE_CATEGORIES
                },
                "median_projection_details_ms": {
                    category: statistics.median(
                        sample["projection_details_ms"].get(category, 0.0)
                        for sample in samples
                    )
                    for category in PROJECTION_DETAIL_CATEGORIES
                },
                "median_fine_details_ms": {
                    category: statistics.median(
                        sample["fine_details_ms"].get(category, 0.0)
                        for sample in samples
                    )
                    for category in FINE_DETAIL_CATEGORIES
                },
            }

    ratio_counts: defaultdict[int, int] = defaultdict(int)
    config = json.loads((model_dir / "config.json").read_text())
    for ratio in config.get("compress_ratios", ())[: shape.layers]:
        ratio_counts[int(ratio)] += 1
    by_ratio: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for report in layer_reports.values():
        by_ratio[int(report["compress_ratio"])].append(report)
    missing = sorted(set(ratio_counts) - set(by_ratio))
    if missing:
        raise ValueError(
            f"profile has no representative for compression ratios {missing}"
        )

    representative_stage_ms = {category: 0.0 for category in COMPUTE_CATEGORIES}
    representative_projection_ms = {
        category: 0.0 for category in PROJECTION_DETAIL_CATEGORIES
    }
    representative_fine_ms = {
        category: 0.0 for category in FINE_DETAIL_CATEGORIES
    }
    for ratio, count in ratio_counts.items():
        reports = by_ratio[ratio]
        for category in COMPUTE_CATEGORIES:
            representative_stage_ms[category] += count * statistics.mean(
                report["median_components_ms"][category] for report in reports
            )
        for category in PROJECTION_DETAIL_CATEGORIES:
            representative_projection_ms[category] += count * statistics.mean(
                report["median_projection_details_ms"][category]
                for report in reports
            )
        for category in FINE_DETAIL_CATEGORIES:
            representative_fine_ms[category] += count * statistics.mean(
                report["median_fine_details_ms"][category] for report in reports
            )
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "scope": "isolated_rank_local_real_layer_forward",
        "model": str(model_dir),
        "device": dict(mx.device_info()),
        "shape": shape.to_dict(),
        "barrier_overhead_ns": barrier_overhead,
        "layer_reports": layer_reports,
        "ratio_counts": {
            str(key): value for key, value in sorted(ratio_counts.items())
        },
        "representative_stage_compute_ms": representative_stage_ms,
        "representative_stage_projection_ms": representative_projection_ms,
        "representative_stage_fine_ms": representative_fine_ms,
        "projection_detail": projection_detail,
        "dspark_verify": dspark_verify,
        "fine_detail": fine_detail,
    }


def _default_representative_layers(model_dir: Path) -> tuple[int, ...]:
    config = json.loads((model_dir / "config.json").read_text())
    ratios = config.get("compress_ratios", ())
    first: dict[int, int] = {}
    for index, ratio in enumerate(ratios[:43]):
        first.setdefault(int(ratio), index)
    if set(first) != {0, 4, 128}:
        raise ValueError("checkpoint does not expose the DS4 0/4/128 schedule")
    return tuple(first[ratio] for ratio in (0, 4, 128))


def _markdown_amdahl(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| component | wall | required for target | TPS at 2x | infinite ceiling |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        required = row["required_speedup_for_target"]
        ceiling = row["infinite_ceiling_tps"]
        lines.append(
            (
                "| {component} | {fraction:.1%} | {required} | "
                "{double:.1f} | {ceiling} |"
            ).format(
                component=row["component"],
                fraction=row["wall_fraction"],
                required=(f"{required:.2f}x" if required is not None else "impossible"),
                double=row["tps_at_2x"],
                ceiling=(f"{ceiling:.1f}" if ceiling is not None else "unbounded"),
            )
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=(0, 1), default=1)
    parser.add_argument("--shard-weights", default="3,5")
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--prefix-tokens", type=int, default=8192)
    parser.add_argument("--layers", type=int, nargs="*")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--projection-detail", action="store_true")
    parser.add_argument("--fine-detail", action="store_true")
    parser.add_argument(
        "--dspark-verify",
        action="store_true",
        help="arm the exact DSpark target-verification path for the measured call",
    )
    parser.add_argument("--baseline-tps", type=float, default=628.76)
    parser.add_argument("--target-tps", type=float, default=1000.0)
    parser.add_argument("--collective-bandwidth-gbps", type=float, default=6.2)
    parser.add_argument("--collective-latency-us", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = tuple(int(item.strip()) for item in args.shard_weights.split(","))
    shape = DS4TPShape(
        rank=args.rank,
        shard_weights=weights,
        tokens=args.tokens,
        prefix_tokens=args.prefix_tokens,
    )
    layers = tuple(args.layers or _default_representative_layers(args.model))
    profile = profile_real_layers(
        args.model.expanduser().resolve(),
        shape=shape,
        layers=layers,
        warmup=args.warmup,
        cycles=args.cycles,
        projection_detail=args.projection_detail,
        dspark_verify=args.dspark_verify,
        fine_detail=args.fine_detail,
    )
    collectives = modeled_collective_ms(
        shape,
        bandwidth_gbps=args.collective_bandwidth_gbps,
        latency_us=args.collective_latency_us,
    )
    attribution = normalize_to_observed_wall(
        profile["representative_stage_compute_ms"],
        collective_ms=collectives["total_ms"],
        tokens=shape.tokens,
        baseline_tps=args.baseline_tps,
        target_tps=args.target_tps,
    )
    report = profile | {
        "collective_model": collectives,
        "observed_attribution": attribution,
        "claims": {
            "production_instrumentation_installed": False,
            "remote_host_touched": False,
            "collective_time_is_modeled": True,
            "rank_local_compute_is_measured": True,
            "distributed_critical_path_directly_measured": False,
            "interpretation": (
                "Use the bottleneck rank's category mix; do not add concurrent "
                "rank walls. Re-run this same harness on each physical rank to "
                "locate the distributed critical path."
            ),
        },
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(encoded + "\n")
    print(encoded)
    print()
    print(_markdown_amdahl(attribution["amdahl"]))


if __name__ == "__main__":
    main()
