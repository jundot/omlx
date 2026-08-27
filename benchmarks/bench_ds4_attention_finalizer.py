#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exact DS4 M=1024 BF16 Q/KV RMSNorm+RoPE finalizer gate.

The default mode is CPU-only. Passing ``--model`` loads one real KV norm
weight and the model's explicit FP32 RoPE frequencies, checks normalized and
rotated boundaries with ``mx.array_equal``, and runs balanced M3 timings. The
candidate symbols are isolated and have no production dispatch.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

TOKENS = 1024
HEAD_DIM = 512
SUPPORTED_HEADS = (24, 32, 40, 64)


def byte_ledger(heads: int) -> dict[str, float]:
    if heads not in SUPPORTED_HEADS:
        raise ValueError("unsupported DS4 local head count")
    q_mib = TOKENS * heads * HEAD_DIM * 2 / 2**20
    kv_mib = TOKENS * HEAD_DIM * 2 / 2**20
    return {
        "q_normalized_intermediate_mib": q_mib,
        "kv_normalized_intermediate_mib": kv_mib,
        "current_norm_write_plus_rope_read_mib": 2 * (q_mib + kv_mib),
        "candidate_removed_mib": 2 * (q_mib + kv_mib),
    }


def analysis_report() -> dict[str, Any]:
    return {
        "shape": {
            "tokens": TOKENS,
            "head_dim": HEAD_DIM,
            "heads": SUPPORTED_HEADS,
            "q_input": "[1,1024,H,512] BF16",
            "q_output": "[1,H,1024,512] BF16",
            "kv_input": "[1,1024,512] BF16",
            "kv_output": "[1,1,1024,512] BF16",
            "freqs": "[256] FP32 including 224 infinity no-RoPE pairs",
        },
        "dispatches": {"current": 4, "candidate": 2},
        "bytes": {str(heads): byte_ledger(heads) for heads in SUPPORTED_HEADS},
        "rounding_contract": [
            "four BF16 reads per lane converted to FP32",
            "two-level simd_sum identical to MLX rms_single_row",
            "metal::precise::rsqrt(acc/512 + eps)",
            "BF16 normalization before optional BF16 KV weight multiply",
            "explicit 1/freq, fast::cos/sin, traditional adjacent pairs",
            "BF16 rotated store",
        ],
        "gate": {
            "normalized_array_equal": True,
            "rotated_array_equal": True,
            "minimum_combined_speedup": 1.10,
            "production_dispatch": False,
        },
    }


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


def _bit_equal(mx, left, right) -> bool:
    """Compare BF16 storage bits, including NaN payloads and signed zero."""

    return bool(mx.array_equal(left.view(mx.uint16), right.view(mx.uint16)).item())


def _abba(
    mx,
    candidate: Callable[[], Any],
    reference: Callable[[], Any],
    warmup: int,
    cycles: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        _evaluate(mx, reference())
        _evaluate(mx, candidate())
    samples = {"reference": [], "candidate": []}
    for _ in range(cycles):
        for name, function in (
            ("reference", reference),
            ("candidate", candidate),
            ("candidate", candidate),
            ("reference", reference),
        ):
            started = time.perf_counter_ns()
            _evaluate(mx, function())
            samples[name].append((time.perf_counter_ns() - started) / 1e6)
    result = {name: _summary(values) for name, values in samples.items()}
    result["speedup"] = (
        result["reference"]["median_ms"] / result["candidate"]["median_ms"]
    )
    result["saved_ms"] = (
        result["reference"]["median_ms"] - result["candidate"]["median_ms"]
    )
    return result


def _load_contract(model: Path, layer: int):
    import mlx.core as mx

    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

    apply_deepseek_v4_patch()
    dm = sys.modules["mlx_lm.models.deepseek_v4"]
    config = json.loads((model / "config.json").read_text())
    ropes = {
        "local": dm.DeepseekV4RoPE(
            config["qk_rope_head_dim"],
            config["rope_theta"],
            None,
            config["max_position_embeddings"],
        ),
        "compressed": dm.DeepseekV4RoPE(
            config["qk_rope_head_dim"],
            config["compress_rope_theta"],
            config["rope_scaling"],
            config["max_position_embeddings"],
        ),
    }
    frequency_sets = {
        name: rope._get_freqs(HEAD_DIM, False) for name, rope in ropes.items()
    }
    index = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    key = f"layers.{layer}.attn.kv_norm.weight"
    weight = mx.load(str(model / index[key]))[key]
    mx.eval(*frequency_sets.values(), weight)
    mx.synchronize()
    return config, ropes, frequency_sets, weight


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx

    from omlx.custom_kernels.glm_moe_dsa import fast

    required = ("ds4_q_head_rms_rope", "ds4_kv_rms_rope")
    if not fast.is_native_available() or not all(
        fast.has_symbol(symbol) for symbol in required
    ):
        raise RuntimeError("isolated DS4 attention finalizer symbols unavailable")

    config, ropes, frequency_sets, kv_weight = _load_contract(args.model, args.layer)
    rope = ropes[args.rope_family]
    freqs = frequency_sets[args.rope_family]
    eps = float(config["rms_norm_eps"])
    results = []
    for heads in args.heads:
        if heads not in SUPPORTED_HEADS:
            raise ValueError(f"unsupported local head count {heads}")
        mx.random.seed(61_000 + heads)
        q = mx.random.normal((1, TOKENS, heads, HEAD_DIM)).astype(mx.bfloat16)
        kv = mx.random.normal((1, TOKENS, HEAD_DIM)).astype(mx.bfloat16)
        mx.eval(q, kv)

        def q_norm_reference(q_=q):
            return mx.fast.rms_norm(q_, None, eps)

        def q_reference():
            return rope(q_norm_reference().transpose(0, 2, 1, 3), args.offset)

        def q_candidate(q_=q):
            return fast.ds4_q_head_rms_rope(q_, freqs, args.offset, eps)

        def kv_norm_reference(kv_=kv):
            return mx.fast.rms_norm(kv_, kv_weight, eps)

        def kv_reference():
            return rope(
                kv_norm_reference().reshape(1, 1, TOKENS, HEAD_DIM),
                args.offset,
            )

        def kv_candidate(kv_=kv):
            return fast.ds4_kv_rms_rope(kv_, kv_weight, freqs, args.offset, eps)

        q_norm = q_norm_reference()
        q_norm_native = fast.ds4_q_head_rms_rope(
            q, freqs, args.offset, eps, return_normalized=True
        )
        q_rotated = q_reference()
        q_rotated_native = q_candidate()
        kv_norm = kv_norm_reference()
        kv_norm_native = fast.ds4_kv_rms_rope(
            kv,
            kv_weight,
            freqs,
            args.offset,
            eps,
            return_normalized=True,
        )
        kv_rotated = kv_reference()
        kv_rotated_native = kv_candidate()
        _evaluate(
            mx,
            (
                q_norm,
                q_norm_native,
                q_rotated,
                q_rotated_native,
                kv_norm,
                kv_norm_native,
                kv_rotated,
                kv_rotated_native,
            ),
        )
        parity = {
            "q_normalized": _bit_equal(mx, q_norm, q_norm_native),
            "q_rotated": _bit_equal(mx, q_rotated, q_rotated_native),
            "kv_normalized": _bit_equal(mx, kv_norm, kv_norm_native),
            "kv_rotated": _bit_equal(mx, kv_rotated, kv_rotated_native),
        }

        parity_matrix = []
        for family, case_rope in ropes.items():
            case_freqs = frequency_sets[family]
            for case_offset in args.parity_offsets:
                q_case_reference = case_rope(q_norm.transpose(0, 2, 1, 3), case_offset)
                q_case_candidate = fast.ds4_q_head_rms_rope(
                    q, case_freqs, case_offset, eps
                )
                kv_case_reference = case_rope(
                    kv_norm.reshape(1, 1, TOKENS, HEAD_DIM), case_offset
                )
                kv_case_candidate = fast.ds4_kv_rms_rope(
                    kv, kv_weight, case_freqs, case_offset, eps
                )
                _evaluate(
                    mx,
                    (
                        q_case_reference,
                        q_case_candidate,
                        kv_case_reference,
                        kv_case_candidate,
                    ),
                )
                parity_matrix.append(
                    {
                        "family": family,
                        "offset": case_offset,
                        "q_rotated_bits_equal": _bit_equal(
                            mx, q_case_reference, q_case_candidate
                        ),
                        "kv_rotated_bits_equal": _bit_equal(
                            mx, kv_case_reference, kv_case_candidate
                        ),
                    }
                )

        q_timing = _abba(mx, q_candidate, q_reference, args.warmup, args.cycles)
        kv_timing = _abba(mx, kv_candidate, kv_reference, args.warmup, args.cycles)
        combined_timing = _abba(
            mx,
            lambda: (q_candidate(), kv_candidate()),
            lambda: (q_reference(), kv_reference()),
            args.warmup,
            args.cycles,
        )
        passed = (
            all(parity.values())
            and all(
                row["q_rotated_bits_equal"] and row["kv_rotated_bits_equal"]
                for row in parity_matrix
            )
            and combined_timing["speedup"] >= args.min_speedup
        )
        results.append(
            {
                "heads": heads,
                "parity": parity,
                "parity_matrix": parity_matrix,
                "q": q_timing,
                "kv": kv_timing,
                "combined": combined_timing,
                "bytes": byte_ledger(heads),
                "minimum_speedup": args.min_speedup,
                "passed": passed,
            }
        )
    if args.strict and not all(result["passed"] for result in results):
        raise SystemExit(2)
    return {
        "device": dict(mx.device_info()),
        "model": str(args.model),
        "layer": args.layer,
        "offset": args.offset,
        "rope_family": args.rope_family,
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--heads", type=int, nargs="+", default=SUPPORTED_HEADS)
    parser.add_argument("--tokens", type=int, choices=(1024, 2048), default=1024)
    parser.add_argument("--offset", type=int, default=8192)
    parser.add_argument(
        "--rope-family", choices=("local", "compressed"), default="compressed"
    )
    parser.add_argument("--parity-offsets", type=int, nargs="+", default=(0, 8192))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--min-speedup", type=float, default=1.10)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    global TOKENS
    args = parse_args()
    TOKENS = args.tokens
    report: dict[str, Any] = {"analysis": analysis_report()}
    if args.model is not None:
        report["gpu_gate"] = run_gate(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
