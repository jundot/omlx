#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Contract, checkpoint audit, and ABBA probe for the DS4 projection bundle.

This file does not participate in model dispatch.  Its default path is CPU-only;
``--model`` additionally measures stock MLX groupings which bound the first
native implementation.  The native ABI is deliberately frozen at ratio-4,
B=M=1 before a production implementation exists.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


CONTRACT_VERSION = 1
NATIVE_B1_SYMBOL = "deepseek_v4_qkv_compressor_bundle_b1"
M1024_ROLLBACK_ENV = "OMLX_DSV4_QKV_BUNDLE_PREFILL"
HIDDEN = 4096
Q_RANK = 1024
KV_DIM = 512
LAYERS = 43
LAYER_RATIOS = (0, 0) + tuple(4 if layer % 2 == 0 else 128 for layer in range(2, 43))


@dataclass(frozen=True)
class Projection:
    name: str
    rows: int
    storage: str

    @property
    def checkpoint_bytes(self) -> int:
        value = self.rows * HIDDEN * (1 if self.storage == "mxfp8" else 2)
        if self.storage == "mxfp8":
            # This released MLX checkpoint is already packed MXFP8: uint32
            # codes hold four values and one uint8 E8M0 scale is stored per
            # output row/input group-32. There is no compact 128x128 F8 scale
            # grid to expand at runtime.
            value += self.rows * (HIDDEN // 32)
        return value

    @property
    def runtime_bytes(self) -> int:
        value = self.rows * HIDDEN * (1 if self.storage == "mxfp8" else 2)
        if self.storage == "mxfp8":
            value += self.rows * (HIDDEN // 32)
        return value

    @property
    def output_bytes(self) -> int:
        return self.rows * 2


def projections(ratio: int) -> tuple[Projection, ...]:
    result = [
        Projection("wq_a", Q_RANK, "mxfp8"),
        Projection("wkv", KV_DIM, "mxfp8"),
    ]
    if ratio:
        width = 1024 if ratio == 4 else 512
        result.extend(
            (
                Projection("compressor.wkv", width, "bf16"),
                Projection("compressor.wgate", width, "bf16"),
            )
        )
    if ratio == 4:
        result.extend(
            (
                Projection("indexer.compressor.wkv", 256, "bf16"),
                Projection("indexer.compressor.wgate", 256, "bf16"),
            )
        )
    if ratio not in (0, 4, 128):
        raise ValueError(f"unsupported DS4 compression ratio {ratio}")
    return tuple(result)


def packed_slices(ratio: int) -> dict[str, tuple[int, int]]:
    cursor = 0
    result = {}
    for projection in projections(ratio):
        result[projection.name] = (cursor, cursor + projection.rows)
        cursor += projection.rows
    return result


def dispatch_ledger() -> dict[str, Any]:
    counts = {ratio: LAYER_RATIOS.count(ratio) for ratio in (0, 4, 128)}
    current = sum(counts[ratio] * len(projections(ratio)) for ratio in counts)
    full_bundle = sum(counts.values())
    # A source-compatible staging point: one quantized bank plus one dense
    # bank on compressed layers, and one quantized bank on local layers.
    two_bank = counts[0] + 2 * (counts[4] + counts[128])
    return {
        "layers": counts,
        "current_projection_dispatches_per_rank": current,
        "two_bank_dispatches_per_rank": two_bank,
        "full_bundle_dispatches_per_rank": full_bundle,
        "full_bundle_dispatches_saved_per_rank": current - full_bundle,
        "tp2_current_aggregate_dispatches": current * 2,
        "tp2_full_bundle_aggregate_dispatches": full_bundle * 2,
        "collectives_changed": 0,
    }


def byte_ledger() -> dict[str, Any]:
    layers = {}
    for ratio in (0, 4, 128):
        specs = projections(ratio)
        layers[str(ratio)] = {
            "checkpoint_bytes": sum(spec.checkpoint_bytes for spec in specs),
            "runtime_bytes": sum(spec.runtime_bytes for spec in specs),
            "b1_output_bytes": sum(spec.output_bytes for spec in specs),
        }
    checkpoint = sum(
        sum(spec.checkpoint_bytes for spec in projections(ratio))
        for ratio in LAYER_RATIOS
    )
    runtime = sum(
        sum(spec.runtime_bytes for spec in projections(ratio))
        for ratio in LAYER_RATIOS
    )
    return {
        "per_layer": layers,
        "all_layers_checkpoint_bytes": checkpoint,
        "all_layers_checkpoint_mib": checkpoint / 2**20,
        "all_layers_runtime_bytes": runtime,
        "all_layers_runtime_mib": runtime / 2**20,
        "ape_bytes_excluded_from_projection_bundle": 5_672_960,
    }


def promotion_contract() -> dict[str, Any]:
    return {
        "first_native_symbol": NATIVE_B1_SYMBOL,
        "shape": {"batch": 1, "rows": 1, "hidden": HIDDEN, "ratio": 4},
        "inputs": [
            "x_bf16[1,4096]",
            "wq_a_u32[1024,1024]",
            "wq_a_scale_u8[1024,128]",
            "wkv_u32[512,1024]",
            "wkv_scale_u8[512,128]",
            "compressor_wkv_bf16[1024,4096]",
            "compressor_wgate_bf16[1024,4096]",
            "index_compressor_wkv_bf16[256,4096]",
            "index_compressor_wgate_bf16[256,4096]",
        ],
        "output": "packed_bf16[1,4096]",
        "packed_slices": packed_slices(4),
        "forbidden_inputs": ["ape", "cache", "position", "distributed_group"],
        "rounding_boundaries": {
            "mxfp8": "independent fp32 reduction per row, then one bf16 store",
            "compressors": "independent fp32 reduction per row, then one bf16 store",
            "cache": "existing Compressor.consume receives the six bf16 views",
            "ape_ratio4": "existing path casts ape to bf16 before gate addition",
            "ape_ratio128": "existing path promotes bf16 gate to fp32 before ape addition",
        },
        "parity": {
            "projection_boundary": "mx.array_equal for every packed slice",
            "decode_positions": [0, 1, 2, 3],
            "state": "array-equal remainder, previous-window, pooled cache, and logits",
            "tp": "both ranks; no new collective and identical replicated outputs",
        },
        "performance": {
            "order": ["reference", "candidate", "candidate", "reference"],
            "machines": ["m3-ultra", "m5"],
            "b1_min_speedup_each_machine": 1.05,
            "m1024_min_speedup_each_machine": 1.05,
            "comparison": "faster of stock separate projections and safe grouped baseline",
        },
        "m1024_production_candidate": {
            "shape": {"batch": 1, "rows": 1024, "hidden": HIDDEN, "ratio": 4},
            "storage": {
                "q_a_raw_kv": "original packed MXFP8 U32 + original U8 scales",
                "compressors": "original BF16 rows",
                "requantization": False,
                "steady_state_duplicate_weight_bytes": 0,
            },
            "dispatches": {"Apple M3 Ultra": 3, "Apple M5 Max": 4},
            "packed_groups": [
                ["wq_a", "wkv"],
                ["compressor.wkv", "compressor.wgate"],
                [
                    "indexer.compressor.wkv",
                    "indexer.compressor.wgate",
                ],
            ],
            "m5_exception": (
                "main BF16 compressor KV/gate remain separate because the "
                "N=2048 grouping changes M5 reduction geometry"
            ),
            "m5_qkv_tile": "NAX BM128/BK64/BN64/WM4/WN2 (variant 5)",
            "hardware": ["Apple M3 Ultra", "Apple M5 Max"],
            "single_node": True,
            "tp2_rank_local": True,
            "collectives_changed": 0,
            "rollback_env": M1024_ROLLBACK_ENV,
            "default_enabled": False,
        },
    }


def analysis_report() -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "layer_ratios": list(LAYER_RATIOS),
        "bytes": byte_ledger(),
        "dispatches": dispatch_ledger(),
        "promotion": promotion_contract(),
        "tp_contract": (
            "wq_a, raw wkv, both compressor pairs, their caches, and q_residual "
            "are replicated; only downstream wq_b/query heads are sharded"
        ),
    }


def _tensor_key(layer: int, name: str, suffix: str = "weight") -> str:
    return f"layers.{layer}.attn.{name}.{suffix}"


def audit_checkpoint(model: Path) -> dict[str, Any]:
    """Validate all bundle tensor headers without materializing weight data."""

    from safetensors import safe_open

    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    handles: dict[str, Any] = {}
    errors = []
    checked = 0
    for layer, ratio in enumerate(LAYER_RATIOS):
        for spec in projections(ratio):
            entries = [
                (
                    "weight",
                    (
                        (spec.rows, HIDDEN // 4)
                        if spec.storage == "mxfp8"
                        else (spec.rows, HIDDEN)
                    ),
                    "U32" if spec.storage == "mxfp8" else "BF16",
                )
            ]
            if spec.storage == "mxfp8":
                entries.append(
                    ("scales", (spec.rows, HIDDEN // 32), "U8")
                )
            for suffix, shape, dtype in entries:
                key = _tensor_key(layer, spec.name, suffix)
                filename = index.get(key)
                if filename is None:
                    errors.append(f"missing {key}")
                    continue
                if filename not in handles:
                    handles[filename] = safe_open(model / filename, framework="np")
                view = handles[filename].get_slice(key)
                got_shape = tuple(view.get_shape())
                got_dtype = view.get_dtype()
                if got_shape != shape or got_dtype != dtype:
                    errors.append(
                        f"{key}: expected {dtype}{shape}, got {got_dtype}{got_shape}"
                    )
                checked += 1
    return {"checked_tensor_headers": checked, "errors": errors}


def _summary(values: Iterable[float]) -> dict[str, float]:
    values = list(values)
    return {
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _abba(
    evaluate: Callable[[Any], None],
    reference: Callable[[], Any],
    candidate: Callable[[], Any],
    cycles: int,
) -> dict[str, Any]:
    for _ in range(3):
        for fn in (reference, candidate, candidate, reference):
            evaluate(fn())
    samples = {"reference": [], "candidate": []}
    for _ in range(cycles):
        for name, fn in (
            ("reference", reference),
            ("candidate", candidate),
            ("candidate", candidate),
            ("reference", reference),
        ):
            start = time.perf_counter_ns()
            evaluate(fn())
            samples[name].append((time.perf_counter_ns() - start) / 1e6)
    result = {name: _summary(values) for name, values in samples.items()}
    result["speedup"] = (
        result["reference"]["median_ms"] / result["candidate"]["median_ms"]
    )
    return result


def _balanced_timings(
    evaluate: Callable[[Any], None],
    functions: dict[str, Callable[[], Any]],
    cycles: int,
) -> dict[str, dict[str, float]]:
    names = tuple(functions)
    for _ in range(3):
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
                start = time.perf_counter_ns()
                evaluate(functions[name]())
                samples[name].append((time.perf_counter_ns() - start) / 1e6)
    return {name: _summary(values) for name, values in samples.items()}


def run_stock_probe(model: Path, layer: int, rows: int, cycles: int) -> dict[str, Any]:
    """Measure only already-available MLX groupings; not the native candidate."""

    import mlx.core as mx

    ratio = LAYER_RATIOS[layer]
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    files: dict[str, Any] = {}

    def load(key: str):
        filename = index[key]
        if filename not in files:
            files[filename] = mx.load(str(model / filename))
        return files[filename][key]

    def load_quant(name: str):
        weight = load(_tensor_key(layer, name)).view(mx.uint32)
        scale = load(_tensor_key(layer, name, "scales"))
        return weight, scale

    q_a = load_quant("wq_a")
    raw_kv = load_quant("wkv")
    dense = [
        load(_tensor_key(layer, spec.name))
        for spec in projections(ratio)
        if spec.storage == "bf16"
    ]
    q_pair = (
        mx.concatenate((q_a[0], raw_kv[0]), axis=0),
        mx.concatenate((q_a[1], raw_kv[1]), axis=0),
    )
    # B1 GEMV keeps exact arithmetic when all dense banks are concatenated.
    # At M=1024, ratio-4's 1024-row and 256-row pairs must remain separate;
    # ratio-128's 512->1024 concat changes MLX's GEMM reduction geometry.
    if rows == 1:
        dense_members = [dense] if dense else []
    elif ratio == 4:
        dense_members = (
            [dense[:2], dense[2:]]
            if rows == 1024
            else [dense[:2], [dense[2]], [dense[3]]]
        )
    else:
        dense_members = [[weight] for weight in dense]
    dense_groups = [
        mx.concatenate(group, axis=0) if len(group) > 1 else group[0]
        for group in dense_members
    ]
    mx.eval(*q_a, *raw_kv, *q_pair, *dense, *dense_groups)
    x = mx.random.normal((rows, HIDDEN)).astype(mx.bfloat16)
    mx.eval(x)

    def qmm(pair):
        return mx.quantized_matmul(
            x,
            pair[0],
            scales=pair[1],
            biases=None,
            transpose=True,
            group_size=32,
            bits=8,
            mode="mxfp8",
        )

    def reference():
        return (qmm(q_a), qmm(raw_kv), *(x @ weight.T for weight in dense))

    def candidate():
        q = qmm(q_pair)
        outputs = [q[:, :Q_RANK], q[:, Q_RANK:]]
        if rows == 1 and dense:
            packed = x @ dense_groups[0].T
            cursor = 0
            for weight in dense:
                outputs.append(packed[:, cursor : cursor + weight.shape[0]])
                cursor += weight.shape[0]
        elif ratio == 4:
            for packed, group in zip(
                (x @ value.T for value in dense_groups), dense_members
            ):
                cursor = 0
                for weight in group:
                    outputs.append(packed[:, cursor : cursor + weight.shape[0]])
                    cursor += weight.shape[0]
        else:
            outputs.extend(x @ weight.T for weight in dense_groups)
        return tuple(outputs)

    def evaluate(values):
        mx.eval(*values)
        mx.synchronize()

    expected = reference()
    actual = candidate()
    evaluate(expected)
    evaluate(actual)
    parity = [bool(mx.array_equal(a, b).item()) for a, b in zip(expected, actual)]
    return {
        "device": mx.device_info(),
        "layer": layer,
        "ratio": ratio,
        "rows": rows,
        "reference_dispatches": len(projections(ratio)),
        "grouped_dispatches": 1 + len(dense_groups),
        "projection_array_equal": parity,
        "all_array_equal": all(parity),
        "timing": _abba(evaluate, reference, candidate, cycles),
        "note": "grouped stock bound only; the native one-dispatch ABI is not called",
    }


def run_native_b1_probe(
    model: Path,
    layer: int,
    cycles: int,
    min_speedup: float,
) -> dict[str, Any]:
    """Run the fixed ratio-4 native bundle against both exact baselines."""

    import mlx.core as mx

    from omlx.custom_kernels.glm_moe_dsa import fast

    if LAYER_RATIOS[layer] != 4:
        raise ValueError("native B1 probe requires a ratio-4 layer")
    if not fast.has_symbol(NATIVE_B1_SYMBOL):
        raise RuntimeError(f"native bundle symbol {NATIVE_B1_SYMBOL!r} unavailable")

    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    files: dict[str, Any] = {}

    def load(key: str):
        filename = index[key]
        if filename not in files:
            files[filename] = mx.load(str(model / filename))
        return files[filename][key]

    def load_quant(name: str):
        return (
            load(_tensor_key(layer, name)).view(mx.uint32),
            load(_tensor_key(layer, name, "scales")),
        )

    q_a = load_quant("wq_a")
    raw_kv = load_quant("wkv")
    dense = [
        load(_tensor_key(layer, spec.name))
        for spec in projections(4)
        if spec.storage == "bf16"
    ]
    q_group = (
        mx.concatenate((q_a[0], raw_kv[0]), axis=0),
        mx.concatenate((q_a[1], raw_kv[1]), axis=0),
    )
    dense_group = mx.concatenate(dense, axis=0)
    mx.eval(*q_a, *raw_kv, *dense, *q_group, dense_group)
    mx.random.seed(42002)
    x = mx.random.normal((1, HIDDEN)).astype(mx.bfloat16)
    mx.eval(x)

    qmm_kwargs = dict(
        transpose=True,
        group_size=32,
        bits=8,
        mode="mxfp8",
    )

    def qmm(pair):
        return mx.quantized_matmul(
            x,
            pair[0],
            scales=pair[1],
            biases=None,
            **qmm_kwargs,
        )

    def separate():
        return (qmm(q_a), qmm(raw_kv), *(x @ weight.T for weight in dense))

    def grouped():
        quant = qmm(q_group)
        packed_dense = x @ dense_group.T
        result = [quant[:, :Q_RANK], quant[:, Q_RANK:]]
        cursor = 0
        for weight in dense:
            result.append(packed_dense[:, cursor : cursor + weight.shape[0]])
            cursor += weight.shape[0]
        return tuple(result)

    slices = packed_slices(4)

    def native():
        packed = fast.deepseek_v4_qkv_compressor_bundle_b1(
            x,
            q_a[0],
            q_a[1],
            raw_kv[0],
            raw_kv[1],
            *dense,
        )
        return tuple(packed[:, start:stop] for start, stop in slices.values())

    def evaluate(values):
        mx.eval(*values)
        mx.synchronize()

    separate_value = separate()
    grouped_value = grouped()
    native_value = native()
    evaluate(separate_value)
    evaluate(grouped_value)
    evaluate(native_value)
    native_vs_separate = [
        bool(mx.array_equal(actual, expected).item())
        for actual, expected in zip(native_value, separate_value)
    ]
    native_vs_grouped = [
        bool(mx.array_equal(actual, expected).item())
        for actual, expected in zip(native_value, grouped_value)
    ]
    grouped_vs_separate = [
        bool(mx.array_equal(actual, expected).item())
        for actual, expected in zip(grouped_value, separate_value)
    ]

    timings = _balanced_timings(
        evaluate,
        {"separate": separate, "grouped": grouped, "native": native},
        cycles,
    )
    faster_baseline = min(
        timings["separate"]["median_ms"], timings["grouped"]["median_ms"]
    )
    speedup = faster_baseline / timings["native"]["median_ms"]
    passed = (
        all(native_vs_separate)
        and all(native_vs_grouped)
        and all(grouped_vs_separate)
        and speedup >= min_speedup
    )
    return {
        "device": mx.device_info(),
        "layer": layer,
        "ratio": 4,
        "rows": 1,
        "packed_slices": slices,
        "dispatches": {
            "separate": 6,
            "grouped": 2,
            "native": fast.deepseek_v4_qkv_compressor_bundle_b1_dispatches(),
        },
        "projection_array_equal": {
            "native_vs_separate": native_vs_separate,
            "native_vs_grouped": native_vs_grouped,
            "grouped_vs_separate": grouped_vs_separate,
        },
        "timings": timings,
        "speedup_vs_faster_baseline": speedup,
        "min_speedup": min_speedup,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--rows", default="1,1024")
    parser.add_argument("--cycles", type=int, default=12)
    parser.add_argument("--min-speedup", type=float, default=1.05)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    report = analysis_report()
    if args.model:
        model = args.model.expanduser()
        report["checkpoint_audit"] = audit_checkpoint(model)
        report["stock_probes"] = [
            run_stock_probe(model, args.layer, int(rows), args.cycles)
            for rows in args.rows.split(",")
        ]
        try:
            report["native_b1_probe"] = run_native_b1_probe(
                model, args.layer, args.cycles, args.min_speedup
            )
        except RuntimeError as exc:
            report["native_b1_probe"] = {"available": False, "error": str(exc)}
        if args.strict and not report["native_b1_probe"].get("passed", False):
            print(json.dumps(report, indent=2, sort_keys=True))
            raise SystemExit(2)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
