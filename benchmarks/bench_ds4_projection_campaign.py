#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exact M=1024 DS4 3:5 projection campaign and portable M5 concat probe.

The default mode is CPU-only.  ``--model`` loads only the requested attention
weights, reproduces rank-local 24/40-head geometry, and profiles every major
projection.  ``--moe-gate-up`` additionally runs the rank-1 width-1280
gate/up-concatenation control requested for M5.  No production model dispatch
imports this benchmark.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

TOKENS = 1024
HIDDEN = 4096
Q_RANK = 1024
HEAD_DIM = 512
HEADS = 64
O_GROUPS = 8
O_RANK = 1024
INDEX_HEADS = 64
INDEX_DIM = 128
SHARD_WEIGHTS = (3, 5)
REPRESENTATIVE_LAYERS = (0, 2, 3)
LAYER_RATIOS = {0: 0, 2: 4, 3: 128}


@dataclass(frozen=True)
class RankShape:
    rank: int
    units: int
    local_heads: int
    q_b_rows: int
    o_a_input: int
    moe_intermediate: int


def rank_shape(rank: int) -> RankShape:
    if rank not in (0, 1):
        raise ValueError("DS4 3:5 TP has exactly two ranks")
    units = SHARD_WEIGHTS[rank]
    return RankShape(
        rank=rank,
        units=units,
        local_heads=O_GROUPS * units,
        q_b_rows=O_GROUPS * units * HEAD_DIM,
        o_a_input=units * HEAD_DIM,
        moe_intermediate=256 * units,
    )


def projection_schema(ratio: int, shape: RankShape) -> dict[str, dict[str, Any]]:
    if ratio not in (0, 4, 128):
        raise ValueError("unsupported DS4 compression ratio")
    result: dict[str, dict[str, Any]] = {
        "q_a": {"m": TOKENS, "k": HIDDEN, "n": Q_RANK, "storage": "mxfp8"},
        "q_b": {
            "m": TOKENS,
            "k": Q_RANK,
            "n": shape.q_b_rows,
            "storage": "mxfp8",
        },
        "raw_kv": {
            "m": TOKENS,
            "k": HIDDEN,
            "n": HEAD_DIM,
            "storage": "mxfp8",
        },
        "o_a": {
            "batches": O_GROUPS,
            "m": TOKENS,
            "k": shape.o_a_input,
            "n": O_RANK,
            "storage": "mxfp8",
        },
        "o_b": {
            "m": TOKENS,
            "k": O_GROUPS * O_RANK,
            "n": HIDDEN,
            "storage": "mxfp8",
        },
    }
    if ratio:
        compressor_rows = 1024 if ratio == 4 else 512
        result["compressor_kv"] = {
            "m": TOKENS,
            "k": HIDDEN,
            "n": compressor_rows,
            "storage": "bf16",
        }
        result["compressor_gate"] = dict(result["compressor_kv"])
    if ratio == 4:
        for name in ("index_compressor_kv", "index_compressor_gate"):
            result[name] = {
                "m": TOKENS,
                "k": HIDDEN,
                "n": 256,
                "storage": "bf16",
            }
        # Row TP is balanced in production; only half the query rows are local.
        result["index_q_b"] = {
            "m": TOKENS // 2,
            "k": Q_RANK,
            "n": INDEX_HEADS * INDEX_DIM,
            "storage": "mxfp8",
        }
        result["index_weights"] = {
            "m": TOKENS // 2,
            "k": HIDDEN,
            "n": INDEX_HEADS,
            "storage": "bf16",
        }
    return result


def _projection_flops(spec: dict[str, Any]) -> int:
    return 2 * spec.get("batches", 1) * spec["m"] * spec["n"] * spec["k"]


def analysis_report() -> dict[str, Any]:
    shapes = {str(rank): rank_shape(rank) for rank in (0, 1)}
    schemas = {
        str(rank): {
            str(ratio): projection_schema(ratio, shape) for ratio in (0, 4, 128)
        }
        for rank, shape in shapes.items()
    }
    for rank_schemas in schemas.values():
        for schema in rank_schemas.values():
            for spec in schema.values():
                spec["flops"] = _projection_flops(spec)
    return {
        "shape": {
            "tokens": TOKENS,
            "hidden": HIDDEN,
            "shard_weights": SHARD_WEIGHTS,
            "ranks": {key: asdict(value) for key, value in shapes.items()},
        },
        "representative_layers": {
            str(layer): LAYER_RATIOS[layer] for layer in REPRESENTATIVE_LAYERS
        },
        "projections": schemas,
        "gates": {
            "projection_bucket_min_speedup": 1.30,
            "all_projection_boundaries": "mx.array_equal",
            "production_dispatch": False,
        },
        "candidate_order": [
            "exact grouped input banks",
            "tuned MXFP8 Q-B/O-A/O-B schedule",
            "exact Q/KV norm+RoPE finalizer",
            "dependent-chain fusion only after individual boundaries pass",
        ],
        "m5_gate_up_concat": {
            "rank": 1,
            "local_intermediate": 1280,
            "combined_rows": 2560,
            "canonical_backing": (
                "construct one concatenated weight/scale bank during load, bind "
                "two views for fallback, then release the original banks"
            ),
            "duplicate_steady_state_bytes": 0,
        },
    }


def _tensor_key(layer: int, name: str, suffix: str = "weight") -> str:
    return f"layers.{layer}.attn.{name}.{suffix}"


def _load_attention_layer(model: Path, layer: int, rank: int):
    import mlx.core as mx

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
        return loaded[_tensor_key(layer, name, suffix)]

    shape = rank_shape(rank)
    unit_row = HEAD_DIM

    def shard_q_b(value):
        pieces = []
        group_rows = value.shape[0] // O_GROUPS
        start = sum(SHARD_WEIGHTS[:rank]) * unit_row
        stop = start + shape.units * unit_row
        for group in mx.split(value, O_GROUPS, axis=0):
            if group.shape[0] != group_rows:
                raise AssertionError("Q-B segment split changed shape")
            pieces.append(group[start:stop])
        return mx.contiguous(mx.concatenate(pieces, axis=0))

    q_b_weight = shard_q_b(get("wq_b"))
    q_b_scales = shard_q_b(get("wq_b", "scales"))

    # Checkpoint wo_a is flattened [8*1024, K/4]. Restore its group axis,
    # then slice the matching 3/8 or 5/8 input columns and scale groups.
    o_a_weight = get("wo_a").reshape(O_GROUPS, O_RANK, -1)
    o_a_scales = get("wo_a", "scales").reshape(O_GROUPS, O_RANK, -1)
    packed_start = sum(SHARD_WEIGHTS[:rank]) * (HEAD_DIM // 4)
    packed_stop = packed_start + shape.units * (HEAD_DIM // 4)
    scale_start = sum(SHARD_WEIGHTS[:rank]) * (HEAD_DIM // 32)
    scale_stop = scale_start + shape.units * (HEAD_DIM // 32)

    tensors: dict[str, Any] = {
        "q_a_weight": get("wq_a"),
        "q_a_scales": get("wq_a", "scales"),
        "q_b_weight": q_b_weight,
        "q_b_scales": q_b_scales,
        "q_norm": get("q_norm"),
        "raw_kv_weight": get("wkv"),
        "raw_kv_scales": get("wkv", "scales"),
        "kv_norm": get("kv_norm"),
        "o_a_weight": mx.contiguous(o_a_weight[..., packed_start:packed_stop]),
        "o_a_scales": mx.contiguous(o_a_scales[..., scale_start:scale_stop]),
        "o_b_weight": get("wo_b"),
        "o_b_scales": get("wo_b", "scales"),
    }
    ratio = LAYER_RATIOS[layer]
    if ratio:
        tensors.update(
            {
                "compressor_kv": get("compressor.wkv"),
                "compressor_gate": get("compressor.wgate"),
            }
        )
    if ratio == 4:
        tensors.update(
            {
                "index_compressor_kv": get("indexer.compressor.wkv"),
                "index_compressor_gate": get("indexer.compressor.wgate"),
                "index_q_b_weight": get("indexer.wq_b"),
                "index_q_b_scales": get("indexer.wq_b", "scales"),
                "index_weights": get("indexer.weights_proj"),
            }
        )
    mx.eval(*tensors.values())
    mx.synchronize()
    return tensors, shards


def _qmm(mx, x, weight, scales):
    return mx.quantized_matmul(
        x,
        weight,
        scales=scales,
        biases=None,
        transpose=True,
        group_size=32,
        bits=8,
        mode="mxfp8",
    )


def _evaluate(mx, value) -> None:
    values = value if isinstance(value, (tuple, list)) else (value,)
    mx.eval(*values)
    mx.synchronize()


def _summary(values: Iterable[float]) -> dict[str, float]:
    samples = list(values)
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _balanced(
    mx,
    functions: dict[str, Callable[[], Any]],
    warmup: int,
    cycles: int,
) -> dict[str, dict[str, float]]:
    names = tuple(functions)
    for _ in range(warmup):
        for name in names:
            _evaluate(mx, functions[name]())
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
                _evaluate(mx, functions[name]())
                samples[name].append((time.perf_counter_ns() - started) / 1e6)
    return {name: _summary(values) for name, values in samples.items()}


def _array_equal_tuple(mx, left: Sequence[Any], right: Sequence[Any]) -> list[bool]:
    _evaluate(mx, (*left, *right))
    return [bool(mx.array_equal(a, b).item()) for a, b in zip(left, right)]


def run_layer_probe(
    model: Path,
    layer: int,
    rank: int,
    warmup: int,
    cycles: int,
    min_speedup: float,
    tile_sweep: bool,
) -> dict[str, Any]:
    import mlx.core as mx

    ratio = LAYER_RATIOS[layer]
    shape = rank_shape(rank)
    tensors, shards = _load_attention_layer(model, layer, rank)
    mx.random.seed(31_000 + layer * 10 + rank)
    x = mx.random.normal((1, TOKENS, HIDDEN)).astype(mx.bfloat16)
    o_input = mx.random.normal((1, O_GROUPS, TOKENS, shape.o_a_input)).astype(
        mx.bfloat16
    )
    mx.eval(x, o_input)

    def q_a():
        return _qmm(mx, x, tensors["q_a_weight"], tensors["q_a_scales"])

    q_a_value = q_a()
    q_residual = mx.fast.rms_norm(q_a_value, tensors["q_norm"], 1e-6)

    def q_b():
        return _qmm(mx, q_residual, tensors["q_b_weight"], tensors["q_b_scales"])

    def raw_kv():
        return _qmm(mx, x, tensors["raw_kv_weight"], tensors["raw_kv_scales"])

    def o_a():
        return _qmm(mx, o_input, tensors["o_a_weight"], tensors["o_a_scales"])

    o_a_value = o_a()
    o_mid = o_a_value.transpose(0, 2, 1, 3).reshape(1, TOKENS, O_GROUPS * O_RANK)

    def o_b():
        return _qmm(mx, o_mid, tensors["o_b_weight"], tensors["o_b_scales"])

    stage_functions: dict[str, Callable[[], Any]] = {
        "q_a": q_a,
        "q_b": q_b,
        "raw_kv": raw_kv,
        "o_a": o_a,
        "o_b": o_b,
    }
    if ratio:
        stage_functions.update(
            {
                "compressor_kv": lambda: x @ tensors["compressor_kv"].T,
                "compressor_gate": lambda: x @ tensors["compressor_gate"].T,
            }
        )
    if ratio == 4:
        row_x = x[:, : TOKENS // 2]
        row_q = q_residual[:, : TOKENS // 2]
        stage_functions.update(
            {
                "index_compressor_kv": lambda: x @ tensors["index_compressor_kv"].T,
                "index_compressor_gate": lambda: x @ tensors["index_compressor_gate"].T,
                "index_q_b": lambda: _qmm(
                    mx,
                    row_q,
                    tensors["index_q_b_weight"],
                    tensors["index_q_b_scales"],
                ),
                "index_weights": lambda: row_x @ tensors["index_weights"].T,
            }
        )

    # Safe stock grouping controls. Q-A/raw-KV have identical MXFP8 K and
    # reduction geometry. Ratio-4 BF16 banks group only equal-output pairs;
    # ratio-128 keeps its two banks separate because concat is non-exact.
    input_weight = mx.concatenate(
        (tensors["q_a_weight"], tensors["raw_kv_weight"]), axis=0
    )
    input_scales = mx.concatenate(
        (tensors["q_a_scales"], tensors["raw_kv_scales"]), axis=0
    )
    dense_groups: list[tuple[str, Any]] = []
    if ratio == 4:
        dense_groups = [
            (
                "compressor",
                mx.concatenate(
                    (tensors["compressor_kv"], tensors["compressor_gate"]),
                    axis=0,
                ),
            ),
            (
                "index_compressor",
                mx.concatenate(
                    (
                        tensors["index_compressor_kv"],
                        tensors["index_compressor_gate"],
                    ),
                    axis=0,
                ),
            ),
        ]
    elif ratio == 128:
        dense_groups = [
            ("compressor_kv", tensors["compressor_kv"]),
            ("compressor_gate", tensors["compressor_gate"]),
        ]
    mx.eval(input_weight, input_scales, *(weight for _, weight in dense_groups))

    def input_reference():
        values = [q_a(), raw_kv()]
        if ratio:
            values.extend(
                (x @ tensors["compressor_kv"].T, x @ tensors["compressor_gate"].T)
            )
        if ratio == 4:
            values.extend(
                (
                    x @ tensors["index_compressor_kv"].T,
                    x @ tensors["index_compressor_gate"].T,
                )
            )
        return tuple(values)

    def input_grouped():
        packed_q = _qmm(mx, x, input_weight, input_scales)
        values = [packed_q[..., :Q_RANK], packed_q[..., Q_RANK:]]
        if ratio == 4:
            packed = [x @ weight.T for _, weight in dense_groups]
            values.extend((packed[0][..., :1024], packed[0][..., 1024:]))
            values.extend((packed[1][..., :256], packed[1][..., 256:]))
        elif ratio == 128:
            values.extend(x @ weight.T for _, weight in dense_groups)
        return tuple(values)

    reference_value = input_reference()
    grouped_value = input_grouped()
    grouped_parity = _array_equal_tuple(mx, reference_value, grouped_value)
    input_timings = _balanced(
        mx,
        {"separate": input_reference, "grouped": input_grouped},
        warmup,
        cycles,
    )

    # Full projection bucket excludes norm/RoPE and attention itself, matching
    # the stage profiler's module brackets. Indexer row-local projections stay
    # in their separately reported bucket and are not counted here.
    def projection_bucket_separate():
        qa = q_a()
        qres = mx.fast.rms_norm(qa, tensors["q_norm"], 1e-6)
        values = [
            _qmm(mx, qres, tensors["q_b_weight"], tensors["q_b_scales"]),
            raw_kv(),
        ]
        if ratio:
            values.extend(
                (x @ tensors["compressor_kv"].T, x @ tensors["compressor_gate"].T)
            )
        if ratio == 4:
            values.extend(
                (
                    x @ tensors["index_compressor_kv"].T,
                    x @ tensors["index_compressor_gate"].T,
                )
            )
        oa = o_a()
        omid = oa.transpose(0, 2, 1, 3).reshape(1, TOKENS, O_GROUPS * O_RANK)
        values.append(_qmm(mx, omid, tensors["o_b_weight"], tensors["o_b_scales"]))
        return tuple(values)

    def projection_bucket_grouped():
        grouped_inputs = input_grouped()
        qa = grouped_inputs[0]
        qres = mx.fast.rms_norm(qa, tensors["q_norm"], 1e-6)
        values = [
            _qmm(mx, qres, tensors["q_b_weight"], tensors["q_b_scales"]),
            grouped_inputs[1],
            *grouped_inputs[2:],
        ]
        oa = o_a()
        omid = oa.transpose(0, 2, 1, 3).reshape(1, TOKENS, O_GROUPS * O_RANK)
        values.append(_qmm(mx, omid, tensors["o_b_weight"], tensors["o_b_scales"]))
        return tuple(values)

    bucket_reference = projection_bucket_separate()
    bucket_grouped = projection_bucket_grouped()
    bucket_parity = _array_equal_tuple(mx, bucket_reference, bucket_grouped)
    bucket_timings = _balanced(
        mx,
        {
            "separate": projection_bucket_separate,
            "grouped_input": projection_bucket_grouped,
        },
        warmup,
        cycles,
    )
    bucket_speedup = (
        bucket_timings["separate"]["median_ms"]
        / bucket_timings["grouped_input"]["median_ms"]
    )
    passed = all(bucket_parity) and bucket_speedup >= min_speedup
    stage_timings = _balanced(mx, stage_functions, warmup, cycles)

    tile_sweeps: dict[str, Any] = {}
    if tile_sweep:
        from omlx.custom_kernels.glm_moe_dsa import fast

        if not fast.has_symbol("ds4_projection_mxfp8_qmm"):
            raise RuntimeError("isolated DS4 projection QMM symbol is unavailable")
        sweep_inputs = {
            "q_b": (
                mx.contiguous(q_residual),
                tensors["q_b_weight"],
                tensors["q_b_scales"],
                q_b,
            ),
            "o_b": (
                mx.contiguous(o_mid),
                tensors["o_b_weight"],
                tensors["o_b_scales"],
                o_b,
            ),
            "o_a": (
                mx.contiguous(o_input),
                tensors["o_a_weight"],
                tensors["o_a_scales"],
                o_a,
            ),
        }
        mx.eval(*(value for values in sweep_inputs.values() for value in values[:3]))
        use_nax = bool(
            fast.ds4_projection_nax_kernels_built()
            and fast.ds4_projection_nax_device_available()
        )
        for name, (candidate_x, weight, scales, stock) in sweep_inputs.items():
            functions = {"stock": stock}
            for variant in range(10):
                functions[f"classic_{variant}"] = (
                    lambda v=variant, x_=candidate_x, w_=weight, s_=scales: fast.ds4_projection_mxfp8_qmm(
                        x_, w_, s_, v
                    )
                )
            if use_nax:
                for variant in range(10):
                    functions[f"nax_{variant}"] = (
                        lambda v=variant, x_=candidate_x, w_=weight, s_=scales: fast.ds4_projection_mxfp8_qmm(
                            x_,
                            w_,
                            s_,
                            0,
                            use_nax=True,
                            nax_variant=v,
                        )
                    )
            stock_value = stock()
            candidates = {key: function() for key, function in functions.items()}
            _evaluate(mx, tuple(candidates.values()))
            exact = {
                key: bool(mx.array_equal(value, stock_value).item())
                for key, value in candidates.items()
            }
            timings = _balanced(mx, functions, warmup, cycles)
            exact_candidates = [
                key for key in functions if key != "stock" and exact[key]
            ]
            best = min(
                exact_candidates,
                key=lambda key: timings[key]["median_ms"],
                default=None,
            )
            tile_sweeps[name] = {
                "nax_active": use_nax,
                "array_equal": exact,
                "timings": timings,
                "best_exact": best,
                "best_speedup": (
                    timings["stock"]["median_ms"] / timings[best]["median_ms"]
                    if best is not None
                    else 0.0
                ),
            }

        projected_ms = bucket_timings["separate"]["median_ms"]
        for name in ("q_b", "o_a", "o_b"):
            sweep = tile_sweeps[name]
            best = sweep["best_exact"]
            if best is not None:
                projected_ms -= sweep["timings"]["stock"]["median_ms"]
                projected_ms += sweep["timings"][best]["median_ms"]
        tile_sweeps["projected_bucket"] = {
            "baseline_ms": bucket_timings["separate"]["median_ms"],
            "candidate_ms": projected_ms,
            "speedup": bucket_timings["separate"]["median_ms"] / projected_ms,
            "minimum_speedup": min_speedup,
        }
    return {
        "layer": layer,
        "ratio": ratio,
        "rank": rank,
        "rank_shape": asdict(shape),
        "shards": shards,
        "stage_timings": stage_timings,
        "input_grouping": {
            "array_equal": grouped_parity,
            "all_exact": all(grouped_parity),
            "timings": input_timings,
            "speedup": input_timings["separate"]["median_ms"]
            / input_timings["grouped"]["median_ms"],
        },
        "projection_bucket": {
            "array_equal": bucket_parity,
            "all_exact": all(bucket_parity),
            "timings": bucket_timings,
            "speedup": bucket_speedup,
            "minimum_speedup": min_speedup,
            "passed": passed,
        },
        "tile_sweeps": tile_sweeps,
    }


def run_m5_gate_up_concat(
    model: Path,
    layer: int,
    warmup: int,
    cycles: int,
) -> dict[str, Any]:
    import mlx.core as mx

    from benchmarks.bench_ds4_tp_prefill_moe_asymmetric import load_tp_layer

    tensors = load_tp_layer(model, layer, rank=1)
    width = tensors["width"]
    if width != 1280:
        raise ValueError("M5 concat control requires the 5/8 width-1280 slice")
    mx.random.seed(51_000 + layer)
    x = mx.random.normal((1, TOKENS, HIDDEN)).astype(mx.bfloat16)
    offsets = (0, 37, 79, 131, 181, 233)
    routes = mx.array(
        [
            [
                [(token * 73 + offset) % 256 for offset in offsets]
                for token in range(TOKENS)
            ]
        ],
        dtype=mx.uint32,
    )
    flat_routes = routes.flatten()
    order = mx.argsort(flat_routes)
    sorted_ids = flat_routes[order]
    sorted_x = mx.expand_dims(x, (-2, -3)).flatten(0, -3)[order // 6]
    sorted_x = sorted_x.astype(mx.float16)

    # Canonical candidate order is up then gate, matching LimitedSwiGLU.
    combined_weight = mx.concatenate(
        (tensors["up_weight"], tensors["gate_weight"]), axis=1
    )
    combined_scales = mx.concatenate(
        (tensors["up_scales"], tensors["gate_scales"]), axis=1
    )
    mx.eval(
        sorted_x,
        sorted_ids,
        combined_weight,
        combined_scales,
        tensors["up_weight"],
        tensors["up_scales"],
        tensors["gate_weight"],
        tensors["gate_scales"],
    )

    kwargs = {
        "transpose": True,
        "group_size": 32,
        "bits": 4,
        "mode": "mxfp4",
        "rhs_indices": sorted_ids,
        "sorted_indices": True,
    }

    def activate(up, gate):
        gate = mx.minimum(gate, 10.0)
        return (gate * mx.sigmoid(gate)) * mx.clip(up, -10.0, 10.0)

    def separate():
        up = mx.gather_qmm(
            sorted_x,
            tensors["up_weight"],
            tensors["up_scales"],
            None,
            **kwargs,
        )
        gate = mx.gather_qmm(
            sorted_x,
            tensors["gate_weight"],
            tensors["gate_scales"],
            None,
            **kwargs,
        )
        return activate(up, gate)

    def combined():
        pair = mx.gather_qmm(
            sorted_x,
            combined_weight,
            combined_scales,
            None,
            **kwargs,
        )
        return activate(pair[..., :width], pair[..., width:])

    separate_value, combined_value = separate(), combined()
    _evaluate(mx, (separate_value, combined_value))
    exact = bool(mx.array_equal(separate_value, combined_value).item())
    max_abs = float(
        mx.max(
            mx.abs(
                separate_value.astype(mx.float32) - combined_value.astype(mx.float32)
            )
        ).item()
    )
    timings = _balanced(
        mx, {"separate": separate, "combined": combined}, warmup, cycles
    )
    return {
        "portable_host": dict(mx.device_info()),
        "layer": layer,
        "rank": 1,
        "local_intermediate": width,
        "combined_rows": 2 * width,
        "array_equal_after_limited_swiglu": exact,
        "max_abs": max_abs,
        "timings": timings,
        "speedup": timings["separate"]["median_ms"] / timings["combined"]["median_ms"],
        "memory_contract": {
            "benchmark_has_transient_duplicate": True,
            "production_design": (
                "combined weight/scales become canonical backing; release the "
                "two source banks and expose fallback slices as views"
            ),
            "steady_state_duplicate_bytes": 0,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--rank", type=int, choices=(0, 1), default=0)
    parser.add_argument("--layers", type=int, nargs="*", default=REPRESENTATIVE_LAYERS)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--min-speedup", type=float, default=1.30)
    parser.add_argument("--moe-gate-up", action="store_true")
    parser.add_argument("--tile-sweep", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report: dict[str, Any] = {"analysis": analysis_report()}
    if args.model is not None:
        unknown = sorted(set(args.layers) - set(REPRESENTATIVE_LAYERS))
        if unknown:
            raise SystemExit(f"unsupported representative layers: {unknown}")
        report["layer_probes"] = [
            run_layer_probe(
                args.model,
                layer,
                args.rank,
                args.warmup,
                args.cycles,
                args.min_speedup,
                args.tile_sweep,
            )
            for layer in args.layers
        ]
        if args.moe_gate_up:
            report["m5_gate_up_concat"] = run_m5_gate_up_concat(
                args.model, 20, args.warmup, args.cycles
            )
        if args.strict and not all(
            row["projection_bucket"]["passed"] for row in report["layer_probes"]
        ):
            print(json.dumps(report, indent=2, sort_keys=True))
            raise SystemExit(2)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
