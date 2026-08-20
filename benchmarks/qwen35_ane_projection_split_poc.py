#!/usr/bin/env python3
"""Probe dual-ANE/GPU output-row splits for remaining Qwen3.5 projections.

This intentionally stays outside the production dispatcher.  It answers two
bounded questions with real model weights:

* can the GDN output projection be split by output rows; and
* can the full-attention Q/K/V projections be treated as one output-row bank?
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import mlx.core as mx


def _measure(call: Callable[[], mx.array], repeats: int) -> tuple[float, list[float]]:
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


def _cosine(a: mx.array, b: mx.array) -> float:
    af = a.astype(mx.float32)
    bf = b.astype(mx.float32)
    value = mx.sum(af * bf) / (
        mx.sqrt(mx.sum(mx.square(af))) * mx.sqrt(mx.sum(mx.square(bf)))
    )
    mx.eval(value)
    return float(value.item())


def _find_gdn(model: Any) -> Any:
    for module in model.modules():
        if hasattr(module, "in_proj_qkv") and hasattr(module, "out_proj"):
            return module
    raise RuntimeError("No Qwen GDN layer was found")


def _find_full_attention(model: Any) -> Any:
    for module in model.modules():
        if all(hasattr(module, name) for name in ("q_proj", "k_proj", "v_proj", "o_proj")):
            return module
    raise RuntimeError("No Qwen full-attention layer was found")


def _quant_spec(linears: Sequence[Any]) -> tuple[int, int, int, Any]:
    bits = int(linears[0].bits)
    group_size = int(linears[0].group_size)
    input_dim = int(linears[0].weight.shape[1]) * 32 // bits
    dtype = linears[0].scales.dtype
    for linear in linears[1:]:
        candidate = (
            int(linear.bits),
            int(linear.group_size),
            int(linear.weight.shape[1]) * 32 // int(linear.bits),
            linear.scales.dtype,
        )
        if candidate != (bits, group_size, input_dim, dtype):
            raise RuntimeError(f"Projection bank has mixed quantization: {candidate}")
    return bits, group_size, input_dim, dtype


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--target",
        choices=("gdn-output", "attention-qkv"),
        required=True,
    )
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--production-dispatch",
        action="store_true",
        help="Compile one layer through the production dispatcher and verify it",
    )
    parser.add_argument(
        "--fp16-ane",
        action="store_true",
        help="Use FP16 rather than ANE INT8 weights for the isolated split",
    )
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=(0.25, 0.35, 0.45),
        help="Total output fraction assigned equally across ANE0 and ANE1",
    )
    args = parser.parse_args()

    from omlx.custom_kernels.qwen35_prefill import fast
    from omlx.patches.qwen35_q4_mlp import _linear_qmm
    from omlx.utils.model_loading import load_text_model

    if not fast.qwen35_ane_available():
        raise RuntimeError("The private ANE runtime is unavailable")
    for symbol in (
        "qwen35_ane_compile_linear_bank",
        "qwen35_ane_dual_affine_qmm_t",
    ):
        if not fast.has_symbol(symbol):
            raise RuntimeError(f"Required native symbol is unavailable: {symbol}")

    print(f"Loading {args.model}", flush=True)
    model, _ = load_text_model(str(args.model))
    if args.target == "gdn-output":
        owner = _find_gdn(model)
        linears = (owner.out_proj,)
        labels = ("out",)
        profile_category = 1
    else:
        owner = _find_full_attention(model)
        linears = (owner.q_proj, owner.k_proj, owner.v_proj)
        labels = ("q", "k", "v")
        profile_category = 2

    bits, group_size, input_dim, dtype = _quant_spec(linears)
    output_dims = tuple(int(linear.weight.shape[0]) for linear in linears)
    output_dim = sum(output_dims)

    # Concatenating rows is lossless for independent affine projections.  It
    # also makes the GPU suffix a single qmm, matching the intended production
    # shape rather than measuring three Python/native dispatches.
    weight = mx.contiguous(mx.concatenate([linear.weight for linear in linears]))
    scales = mx.contiguous(mx.concatenate([linear.scales for linear in linears]))
    biases = mx.contiguous(mx.concatenate([linear.biases for linear in linears]))
    mx.eval(weight, scales, biases)

    mx.random.seed(0)
    x = mx.random.normal((1, args.tokens, input_dim)).astype(dtype)
    mx.eval(x)

    def gpu_call() -> mx.array:
        return mx.concatenate([_linear_qmm(linear, x, 8) for linear in linears], axis=-1)

    reference = gpu_call()
    mx.eval(reference)
    gpu_seconds, gpu_samples = _measure(gpu_call, args.repeats)

    if args.production_dispatch:
        from omlx.patches import qwen35_ane_prefill as ane_prefill

        enabled = ane_prefill.enable_qwen35_ane_prefill(
            model,
            sequence_length=args.tokens,
            fraction=0.50,
            max_layers=1,
            gdn=True,
            gdn_fraction=0.50,
            gdn_max_layers=1,
            gdn_output=args.target == "gdn-output",
            gdn_output_fraction=0.25,
            attention=args.target == "attention-qkv",
            attention_fraction=0.35,
        )
        if args.target == "gdn-output":
            production_call = lambda: ane_prefill._gdn_output_backend(owner, x)
        else:
            production_call = lambda: mx.concatenate(
                ane_prefill._attention_backend(owner, x), axis=-1
            )
        smoke = production_call()
        if smoke is None:
            raise RuntimeError("Production projection dispatcher declined the input")
        mx.eval(smoke)
        seconds, samples = _measure(production_call, args.repeats)
        # The native hybrid operator reuses internal staging buffers, so keep
        # the accuracy sample as the final invocation instead of retaining it
        # across subsequent timed calls.
        candidate = production_call()
        mx.eval(candidate)
        difference = candidate.astype(mx.float32) - reference.astype(mx.float32)
        state = owner._omlx_ane_projection_state
        ane_outputs = output_dim - int(state.weight.shape[0])
        mx.eval(difference)
        print(
            "PRODUCTION "
            + json.dumps(
                {
                    "enabled_mlp_layers": enabled,
                    "status": ane_prefill.qwen35_ane_prefill_status(model),
                    "median_ms": seconds * 1000,
                    "samples_ms": [sample * 1000 for sample in samples],
                    "speedup_vs_gpu": gpu_seconds / seconds,
                    "cosine": _cosine(reference, candidate),
                    "ane_prefix_cosine": _cosine(
                        reference[..., :ane_outputs], candidate[..., :ane_outputs]
                    ),
                    "gpu_suffix_cosine": _cosine(
                        reference[..., ane_outputs:], candidate[..., ane_outputs:]
                    ),
                    "ane_outputs": ane_outputs,
                    "ane_model": repr(state.model),
                    "ane_model1": repr(state.model1),
                    "rmse": float(mx.sqrt(mx.mean(mx.square(difference))).item()),
                    "max_abs": float(mx.max(mx.abs(difference)).item()),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return

    fused_qmm = getattr(fast, f"qwen35_q{bits}_affine_qmm_t")

    def fused_gpu_call() -> mx.array:
        return fused_qmm(x, weight, scales, biases, 8, group_size)

    fused_gpu = fused_gpu_call()
    mx.eval(fused_gpu)
    fused_gpu_seconds, fused_gpu_samples = _measure(
        fused_gpu_call, args.repeats
    )

    prepared = []
    weights0 = []
    weights1 = []
    for fraction in args.fractions:
        ane_outputs = (int(output_dim * fraction) // 128) * 128
        split = ane_outputs // 2
        gpu_outputs = output_dim - ane_outputs
        if ane_outputs <= 0 or split % 64 or gpu_outputs <= 0 or gpu_outputs % 64:
            print(f"Skipping invalid fraction {fraction:.4f}", flush=True)
            continue
        dense0 = mx.contiguous(
            mx.dequantize(
                weight[:split],
                scales[:split],
                biases[:split],
                group_size=group_size,
                bits=bits,
            ).astype(mx.float32)
        )
        dense1 = mx.contiguous(
            mx.dequantize(
                weight[split:ane_outputs],
                scales[split:ane_outputs],
                biases[split:ane_outputs],
                group_size=group_size,
                bits=bits,
            ).astype(mx.float32)
        )
        gpu_weight = mx.contiguous(weight[ane_outputs:])
        gpu_scales = mx.contiguous(scales[ane_outputs:])
        gpu_biases = mx.contiguous(biases[ane_outputs:])
        mx.eval(dense0, dense1, gpu_weight, gpu_scales, gpu_biases)
        weights0.append(dense0)
        weights1.append(dense1)
        prepared.append(
            (fraction, ane_outputs, gpu_weight, gpu_scales, gpu_biases)
        )

    compile_started = time.perf_counter()
    if args.fp16_ane:
        models0 = fast.qwen35_ane_compile_fp16_linear_bank(
            weights0, args.tokens, 1
        )
        models1 = fast.qwen35_ane_compile_fp16_linear_bank(
            weights1, args.tokens, 2
        )
    else:
        models0 = fast.qwen35_ane_compile_linear_bank(weights0, args.tokens, 1)
        models1 = fast.qwen35_ane_compile_linear_bank(weights1, args.tokens, 2)
    compile_seconds = time.perf_counter() - compile_started
    del weights0, weights1

    results = []
    for index, entry in enumerate(prepared):
        fraction, ane_outputs, gpu_weight, gpu_scales, gpu_biases = entry

        def candidate_call(
            gpu_weight=gpu_weight,
            gpu_scales=gpu_scales,
            gpu_biases=gpu_biases,
            model0=models0[index],
            model1=models1[index],
        ) -> mx.array:
            return fast.qwen35_ane_dual_affine_qmm_t(
                x,
                gpu_weight,
                gpu_scales,
                gpu_biases,
                model0,
                model1,
                bits,
                8,
                group_size,
                profile_category,
            )

        seconds, samples = _measure(candidate_call, args.repeats)
        candidate = candidate_call()
        mx.eval(candidate)
        difference = candidate.astype(mx.float32) - reference.astype(mx.float32)
        mx.eval(difference)
        result = {
            "requested_fraction": fraction,
            "realized_fraction": ane_outputs / output_dim,
            "ane_outputs": ane_outputs,
            "median_ms": seconds * 1000,
            "samples_ms": [sample * 1000 for sample in samples],
            "speedup_vs_gpu": gpu_seconds / seconds,
            "speedup_vs_fused_gpu": fused_gpu_seconds / seconds,
            "cosine": _cosine(reference, candidate),
            "rmse": float(mx.sqrt(mx.mean(mx.square(difference))).item()),
            "max_abs": float(mx.max(mx.abs(difference)).item()),
        }
        results.append(result)
        print("CANDIDATE " + json.dumps(result, sort_keys=True), flush=True)

    print(
        "RESULT "
        + json.dumps(
            {
                "model": str(args.model),
                "target": args.target,
                "owner": type(owner).__name__,
                "labels": labels,
                "tokens": args.tokens,
                "input_dim": input_dim,
                "output_dims": output_dims,
                "bits": bits,
                "group_size": group_size,
                "compile_seconds": compile_seconds,
                "gpu_median_ms": gpu_seconds * 1000,
                "gpu_samples_ms": [sample * 1000 for sample in gpu_samples],
                "fused_gpu_median_ms": fused_gpu_seconds * 1000,
                "fused_gpu_samples_ms": [
                    sample * 1000 for sample in fused_gpu_samples
                ],
                "fused_gpu_cosine": _cosine(reference, fused_gpu),
                "candidates": results,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
