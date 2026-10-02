#!/usr/bin/env python3
"""Probe the single-submission fused ANE SwiGLU/down branch on real weights."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm.models.activations import swiglu


def _cosine(a: mx.array, b: mx.array) -> float:
    af = a.astype(mx.float32)
    bf = b.astype(mx.float32)
    value = mx.sum(af * bf) / (
        mx.sqrt(mx.sum(mx.square(af))) * mx.sqrt(mx.sum(mx.square(bf)))
    )
    mx.eval(value)
    return float(value.item())


def _measure(call, repeats: int) -> tuple[float, list[float]]:
    value = call()
    mx.eval(value)
    mx.synchronize()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        value = call()
        mx.eval(value)
        mx.synchronize()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples), samples


def _first_mlp(model: Any) -> Any:
    for module in model.modules():
        if all(
            hasattr(module, name)
            for name in ("gate_proj", "up_proj", "down_proj")
        ):
            return module
    raise RuntimeError("No dense Qwen MLP was found")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output-fraction",
        type=float,
        default=1.0,
        help="Fraction of final output rows included in the fused projection",
    )
    parser.add_argument("--zero-ane-down", action="store_true")
    parser.add_argument("--zero-gpu", action="store_true")
    parser.add_argument(
        "--dual",
        action="store_true",
        help="Treat each requested fraction as the share assigned to each ANE",
    )
    parser.add_argument("--bank-copies", type=int, default=0)
    parser.add_argument(
        "--cpu-fraction",
        type=float,
        default=0.0,
        help="Hidden-channel share assigned to the fused fp16 CPU branch",
    )
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=(0.05, 0.075, 0.10, 0.125, 0.15),
    )
    args = parser.parse_args()

    from omlx.custom_kernels.qwen35_prefill import fast
    from omlx.patches.qwen35_q4_mlp import _linear_qmm
    from omlx.utils.model_loading import load_text_model

    if not fast.qwen35_ane_swiglu_down_available():
        raise RuntimeError("The private ANE fused SwiGLU/down path is unavailable")

    print(f"Loading {args.model}", flush=True)
    model, _ = load_text_model(str(args.model))
    mlp = _first_mlp(model)
    gate = mlp.gate_proj
    up = mlp.up_proj
    down = mlp.down_proj
    if any(int(linear.bits) != 4 for linear in (gate, up, down)):
        raise RuntimeError("The legacy fused branch requires q4 weights")
    group_size = int(gate.group_size)
    if group_size != 128 or int(down.group_size) != group_size:
        raise RuntimeError("The legacy fused branch requires group size 128")

    hidden_dim = int(gate.weight.shape[0])
    model_dim = int(gate.weight.shape[1]) * 8
    fused_outputs = (int(model_dim * args.output_fraction) // 64) * 64
    if fused_outputs <= 0 or fused_outputs > model_dim:
        raise ValueError("--output-fraction must produce 1..model_dim aligned rows")
    mx.random.seed(0)
    x = mx.random.normal((1, args.tokens, model_dim)).astype(mx.float16)

    def gpu_call():
        gate_value = _linear_qmm(gate, x, 8)
        up_value = _linear_qmm(up, x, 8)
        return _linear_qmm(down, swiglu(gate_value, up_value), 8)

    full_reference = gpu_call()
    mx.eval(full_reference)
    reference = mx.contiguous(full_reference[..., :fused_outputs])
    gpu_seconds, gpu_samples = _measure(gpu_call, args.repeats)
    dense_down = mx.dequantize(
        down.weight,
        down.scales,
        down.biases,
        group_size=group_size,
        bits=4,
    ).astype(mx.float32)

    results = []
    for fraction in args.fractions:
        per_ane_hidden = (int(hidden_dim * fraction) // 128) * 128
        ane_hidden = per_ane_hidden * (2 if args.dual else 1)
        cpu_hidden = (int(hidden_dim * args.cpu_fraction) // 128) * 128
        gpu_start = ane_hidden + cpu_hidden
        gpu_hidden = hidden_dim - gpu_start
        if per_ane_hidden <= 0 or gpu_hidden <= 0 or gpu_hidden % 128:
            print(f"Skipping invalid fraction {fraction:.4f}", flush=True)
            continue

        def dense_rows(linear, start=0, end=per_ane_hidden):
            return mx.contiguous(
                mx.dequantize(
                    linear.weight[start:end],
                    linear.scales[start:end],
                    linear.biases[start:end],
                    group_size=group_size,
                    bits=4,
                ).astype(mx.float32)
            )

        gate_dense = dense_rows(gate)
        up_dense = dense_rows(up)
        down_dense = mx.contiguous(
            dense_down[:fused_outputs, :per_ane_hidden]
        )
        gate_dense1 = None
        up_dense1 = None
        down_dense1 = None
        if args.dual:
            gate_dense1 = dense_rows(
                gate, per_ane_hidden, 2 * per_ane_hidden
            )
            up_dense1 = dense_rows(up, per_ane_hidden, 2 * per_ane_hidden)
            down_dense1 = mx.contiguous(
                dense_down[
                    :fused_outputs, per_ane_hidden : 2 * per_ane_hidden
                ]
            )
        compiled_down = (
            mx.zeros_like(down_dense) if args.zero_ane_down else down_dense
        )
        cpu_gate_up_weight = None
        cpu_down_weight = None
        if cpu_hidden:
            cpu_gate_up_weight = mx.contiguous(
                mx.concatenate(
                    (
                        dense_rows(gate, ane_hidden, gpu_start),
                        dense_rows(up, ane_hidden, gpu_start),
                    ),
                    axis=0,
                ).astype(mx.float16)
            )
            cpu_down_weight = mx.contiguous(
                dense_down[:fused_outputs, ane_hidden:gpu_start].astype(
                    mx.float16
                )
            )
        packed_start = gpu_start // 8
        scale_start = gpu_start // group_size
        gpu_gate_up_weight = mx.contiguous(
            mx.concatenate((gate.weight[gpu_start:], up.weight[gpu_start:]), axis=0)
        )
        gpu_gate_up_scales = mx.contiguous(
            mx.concatenate((gate.scales[gpu_start:], up.scales[gpu_start:]), axis=0)
        )
        gpu_gate_up_biases = mx.contiguous(
            mx.concatenate((gate.biases[gpu_start:], up.biases[gpu_start:]), axis=0)
        )
        gpu_down_weight = mx.contiguous(
            down.weight[:fused_outputs, packed_start:]
        )
        gpu_down_scales = mx.contiguous(
            down.scales[:fused_outputs, scale_start:]
        )
        gpu_down_biases = mx.contiguous(
            down.biases[:fused_outputs, scale_start:]
        )
        if args.zero_gpu:
            gpu_gate_up_scales = mx.zeros_like(gpu_gate_up_scales)
            gpu_down_scales = mx.zeros_like(gpu_down_scales)
        mx.eval(
            gate_dense,
            up_dense,
            down_dense,
            compiled_down,
            gpu_gate_up_weight,
            gpu_gate_up_scales,
            gpu_gate_up_biases,
            gpu_down_weight,
            gpu_down_scales,
            gpu_down_biases,
        )
        if cpu_gate_up_weight is not None and cpu_down_weight is not None:
            mx.eval(cpu_gate_up_weight, cpu_down_weight)
        if gate_dense1 is not None:
            mx.eval(gate_dense1, up_dense1, down_dense1)
        gate_suffix_dense = mx.dequantize(
            gate.weight[gpu_start:],
            gate.scales[gpu_start:],
            gate.biases[gpu_start:],
            group_size=group_size,
            bits=4,
        ).astype(mx.float16)
        up_suffix_dense = mx.dequantize(
            up.weight[gpu_start:],
            up.scales[gpu_start:],
            up.biases[gpu_start:],
            group_size=group_size,
            bits=4,
        ).astype(mx.float16)
        prefix_activation = swiglu(
            mx.matmul(x, gate_dense.astype(mx.float16).T),
            mx.matmul(x, up_dense.astype(mx.float16).T),
        )
        prefix_activation1 = None
        if gate_dense1 is not None:
            prefix_activation1 = swiglu(
                mx.matmul(x, gate_dense1.astype(mx.float16).T),
                mx.matmul(x, up_dense1.astype(mx.float16).T),
            )
        suffix_activation = swiglu(
            mx.matmul(x, gate_suffix_dense.T),
            mx.matmul(x, up_suffix_dense.T),
        )
        suffix_reference = mx.matmul(
            suffix_activation,
            dense_down[:fused_outputs, gpu_start:].astype(mx.float16).T,
        )
        native_gate_up = fast.qwen35_q4_affine_qmm_t(
            x,
            gpu_gate_up_weight,
            gpu_gate_up_scales,
            gpu_gate_up_biases,
            8,
            group_size,
        )
        native_suffix_activation = swiglu(
            native_gate_up[..., :gpu_hidden],
            native_gate_up[..., gpu_hidden:],
        )
        native_suffix = fast.qwen35_q4_affine_qmm_t(
            native_suffix_activation,
            gpu_down_weight,
            gpu_down_scales,
            gpu_down_biases,
            8,
            group_size,
        )
        mx.eval(native_suffix)
        prefix_reference = mx.matmul(
            prefix_activation, down_dense.astype(mx.float16).T
        )
        if prefix_activation1 is not None:
            prefix_reference = prefix_reference + mx.matmul(
                prefix_activation1, down_dense1.astype(mx.float16).T
            )
        if cpu_gate_up_weight is not None and cpu_down_weight is not None:
            cpu_activation = swiglu(
                mx.matmul(x, cpu_gate_up_weight[:cpu_hidden].T),
                mx.matmul(x, cpu_gate_up_weight[cpu_hidden:].T),
            )
            prefix_reference = prefix_reference + mx.matmul(
                cpu_activation, cpu_down_weight.T
            )
        split_reference = prefix_reference + suffix_reference
        mx.eval(split_reference)

        started = time.perf_counter()
        try:
            if args.bank_copies > 0:
                ane_model = fast.qwen35_ane_compile_swiglu_down_bank(
                    [gate_dense] * args.bank_copies,
                    [up_dense] * args.bank_copies,
                    [compiled_down] * args.bank_copies,
                    args.tokens,
                    1 if args.dual else 0,
                )[0]
            else:
                ane_model = fast.qwen35_ane_compile_swiglu_down(
                    gate_dense,
                    up_dense,
                    compiled_down,
                    args.tokens,
                    1 if args.dual else 0,
                )
            ane_model1 = None
            if args.dual:
                if args.bank_copies > 0:
                    ane_model1 = fast.qwen35_ane_compile_swiglu_down_bank(
                        [gate_dense1] * args.bank_copies,
                        [up_dense1] * args.bank_copies,
                        [down_dense1] * args.bank_copies,
                        args.tokens,
                        2,
                    )[0]
                else:
                    ane_model1 = fast.qwen35_ane_compile_swiglu_down(
                        gate_dense1,
                        up_dense1,
                        down_dense1,
                        args.tokens,
                        2,
                    )
        except Exception as exc:
            result = {
                "requested_fraction": fraction,
                "realized_fraction": ane_hidden / hidden_dim,
                "error": f"{type(exc).__name__}: {exc}",
                "compile_seconds": time.perf_counter() - started,
            }
            results.append(result)
            print("CANDIDATE " + json.dumps(result, sort_keys=True), flush=True)
            continue
        compile_seconds = time.perf_counter() - started

        def candidate_call(
            gpu_gate_up_weight=gpu_gate_up_weight,
            gpu_gate_up_scales=gpu_gate_up_scales,
            gpu_gate_up_biases=gpu_gate_up_biases,
            gpu_down_weight=gpu_down_weight,
            gpu_down_scales=gpu_down_scales,
            gpu_down_biases=gpu_down_biases,
            ane_model=ane_model,
            ane_model1=ane_model1,
            cpu_gate_up_weight=cpu_gate_up_weight,
            cpu_down_weight=cpu_down_weight,
        ):
            if cpu_gate_up_weight is not None:
                return fast.qwen35_ane_dual_cpu_fp16_q4_swiglu_down_t(
                    x,
                    cpu_gate_up_weight,
                    cpu_down_weight,
                    gpu_gate_up_weight,
                    gpu_gate_up_scales,
                    gpu_gate_up_biases,
                    gpu_down_weight,
                    gpu_down_scales,
                    gpu_down_biases,
                    ane_model,
                    ane_model1,
                    8,
                    group_size,
                    args.cpu_threads,
                    True,
                )
            call = (
                fast.qwen35_ane_dual_q4_swiglu_down_t
                if ane_model1 is not None
                else fast.qwen35_ane_q4_swiglu_down_t
            )
            models = (
                (ane_model, ane_model1)
                if ane_model1 is not None
                else (ane_model,)
            )
            return call(
                x,
                gpu_gate_up_weight,
                gpu_gate_up_scales,
                gpu_gate_up_biases,
                gpu_down_weight,
                gpu_down_scales,
                gpu_down_biases,
                *models,
                8,
                group_size,
            )

        seconds, samples = _measure(candidate_call, args.repeats)
        candidate = candidate_call()
        mx.eval(candidate)
        difference = candidate.astype(mx.float32) - reference.astype(mx.float32)
        mx.eval(difference)
        result = {
            "requested_fraction": fraction,
            "realized_fraction": ane_hidden / hidden_dim,
            "ane_hidden": ane_hidden,
            "per_ane_hidden": per_ane_hidden,
            "cpu_hidden": cpu_hidden,
            "cpu_fraction": cpu_hidden / hidden_dim,
            "cpu_threads": args.cpu_threads,
            "dual": args.dual,
            "compile_seconds": compile_seconds,
            "median_ms": seconds * 1000,
            "samples_ms": [sample * 1000 for sample in samples],
            "speedup_vs_gpu": gpu_seconds / seconds,
            "cosine": _cosine(reference, candidate),
            "split_reference_cosine": _cosine(reference, split_reference),
            "ane_vs_split_cosine": _cosine(split_reference, candidate),
            "ane_vs_suffix_cosine": _cosine(suffix_reference, candidate),
            "ane_vs_prefix_cosine": _cosine(prefix_reference, candidate),
            "native_suffix_cosine": _cosine(suffix_reference, native_suffix),
            "rmse": float(mx.sqrt(mx.mean(mx.square(difference))).item()),
            "max_abs": float(mx.max(mx.abs(difference)).item()),
            "candidate_nan_count": int(mx.sum(mx.isnan(candidate)).item()),
            "candidate_inf_count": int(mx.sum(mx.isinf(candidate)).item()),
            "prefix_activation_max": float(mx.max(mx.abs(prefix_activation)).item()),
            "split_reference_max": float(mx.max(mx.abs(split_reference)).item()),
        }
        results.append(result)
        print("CANDIDATE " + json.dumps(result, sort_keys=True), flush=True)

    print(
        "RESULT "
        + json.dumps(
            {
                "model": str(args.model),
                "tokens": args.tokens,
                "layer": type(mlp).__name__,
                "model_dim": model_dim,
                "hidden_dim": hidden_dim,
                "fused_outputs": fused_outputs,
                "gpu_median_ms": gpu_seconds * 1000,
                "gpu_samples_ms": [sample * 1000 for sample in gpu_samples],
                "candidates": results,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
