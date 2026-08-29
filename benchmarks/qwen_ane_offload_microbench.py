#!/usr/bin/env python3
"""Microbenchmark Qwen3.5+/Qwen4 ANE offload on real checkpoint projections.

Only one shared-expert MLP and one GDN projection bundle are loaded. This
keeps the benchmark focused on heterogeneous offload viability without loading
Qwen4's large routed-expert and PLE tensors.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu

from omlx.custom_kernels.qwen35_prefill import fast
from omlx.patches import qwen35_ane_prefill as patch


def _cosine(left: mx.array, right: mx.array) -> float:
    left = left.astype(mx.float32).reshape(-1)
    right = right.astype(mx.float32).reshape(-1)
    value = mx.sum(left * right) / (
        mx.sqrt(mx.sum(left * left)) * mx.sqrt(mx.sum(right * right))
    )
    mx.eval(value)
    return float(value.item())


def _compare(reference: Any, candidate: Any) -> dict[str, float]:
    reference_values = reference if isinstance(reference, tuple) else (reference,)
    candidate_values = candidate if isinstance(candidate, tuple) else (candidate,)
    cosines = []
    max_errors = []
    for expected, actual in zip(reference_values, candidate_values, strict=True):
        difference = actual.astype(mx.float32) - expected.astype(mx.float32)
        mx.eval(difference)
        cosines.append(_cosine(expected, actual))
        max_errors.append(float(mx.max(mx.abs(difference)).item()))
    return {
        "minimum_cosine": min(cosines),
        "maximum_absolute_error": max(max_errors),
    }


def _measure(factory: Callable[[], Any], repeats: int) -> tuple[float, Any]:
    output = factory()
    mx.eval(*(output if isinstance(output, tuple) else (output,)))
    mx.synchronize()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        output = factory()
        mx.eval(*(output if isinstance(output, tuple) else (output,)))
        mx.synchronize()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples), output


class _Checkpoint:
    def __init__(self, root: Path):
        self.root = root
        self.config = json.loads((root / "config.json").read_text())
        index = json.loads((root / "model.safetensors.index.json").read_text())
        self.weight_map = index["weight_map"]
        self.files: dict[str, dict[str, mx.array]] = {}

    def tensor(self, key: str) -> mx.array:
        filename = self.weight_map[key]
        if filename not in self.files:
            self.files[filename] = mx.load(str(self.root / filename))
        return self.files[filename][key]

    def linear(self, prefix: str) -> nn.QuantizedLinear:
        quantization = self.config.get("quantization_config") or {}
        spec = {
            key: quantization[key]
            for key in ("bits", "group_size", "mode")
            if key in quantization
        }
        override = quantization.get(prefix)
        if isinstance(override, dict):
            spec.update(override)
        if "bits" not in spec or "group_size" not in spec:
            raise ValueError(f"{prefix} is not an affine quantized projection")
        bits = int(spec["bits"])
        group_size = int(spec["group_size"])
        weight = self.tensor(f"{prefix}.weight")
        output_dim = int(weight.shape[0])
        input_dim = int(weight.shape[1]) * 32 // bits
        linear = nn.QuantizedLinear(
            input_dim,
            output_dim,
            bias=False,
            group_size=group_size,
            bits=bits,
            mode=str(spec.get("mode", "affine")),
        )
        linear.weight = weight
        linear.scales = self.tensor(f"{prefix}.scales")
        linear.biases = self.tensor(f"{prefix}.biases")
        return linear


def _compile_mlp(
    checkpoint: _Checkpoint,
    layer: int,
    sequence_length: int,
    fraction: float,
) -> tuple[Any, patch._AnePrefillConfig, patch._CombinedMLPState]:
    prefix = f"language_model.model.layers.{layer}.mlp.shared_expert"
    mlp = SimpleNamespace(
        gate_proj=checkpoint.linear(f"{prefix}.gate_proj"),
        up_proj=checkpoint.linear(f"{prefix}.up_proj"),
        down_proj=checkpoint.linear(f"{prefix}.down_proj"),
    )
    config = patch._AnePrefillConfig(
        sequence_length=sequence_length,
        fraction=fraction,
        variant=8,
        dual_ane=True,
    )
    prepared = patch._prepare_pair_for_bank(mlp, config)
    if prepared is None:
        raise RuntimeError("The selected Qwen shared expert is not ANE eligible")
    state, dense0, dense1 = prepared
    if dense1 is None:
        raise RuntimeError("Dual-ANE shared-expert preparation was incomplete")
    banked = patch._compile_dual_banks([dense0], [dense1], sequence_length)
    if banked is None:
        raise RuntimeError("The shared-expert ANE bank could not be compiled")
    models0, models1, _ = banked
    state = patch.replace(state, model=models0[0], model1=models1[0])
    mlp._omlx_ane_prefill_config = config
    mlp._omlx_ane_prefill_state = state
    return mlp, config, state


def _compile_gdn(
    checkpoint: _Checkpoint,
    layer: int,
    sequence_length: int,
    fraction: float,
) -> tuple[Any, patch._AneGDNConfig, patch._CombinedGDNState]:
    prefix = f"language_model.model.layers.{layer}.linear_attn"
    gdn = SimpleNamespace(
        in_proj_qkv=checkpoint.linear(f"{prefix}.in_proj_qkv"),
        in_proj_z=checkpoint.linear(f"{prefix}.in_proj_z"),
        in_proj_b=checkpoint.linear(f"{prefix}.in_proj_b"),
        in_proj_a=checkpoint.linear(f"{prefix}.in_proj_a"),
    )
    floor = patch._min_viable_gdn_fraction(gdn, 128)
    if floor is None:
        raise RuntimeError("The selected Qwen GDN is not ANE eligible")
    fraction = max(fraction, floor)
    config = patch._AneGDNConfig(
        sequence_length=sequence_length,
        fraction=fraction,
        variant=8,
        dual_ane=True,
    )
    prepared = patch._prepare_gdn_for_bank(gdn, config)
    if prepared is None:
        raise RuntimeError("The selected Qwen GDN could not be prepared")
    state, dense0, dense1 = prepared
    if dense1 is None:
        raise RuntimeError("Dual-ANE GDN preparation was incomplete")
    banked = patch._compile_dual_banks([dense0], [dense1], sequence_length)
    if banked is None:
        raise RuntimeError("The GDN ANE bank could not be compiled")
    models0, models1, _ = banked
    state = patch.replace(state, model=models0[0], model1=models1[0])
    gdn._omlx_ane_gdn_config = config
    gdn._omlx_ane_gdn_state = state
    return gdn, config, state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--mlp-fraction", type=float, default=0.40)
    parser.add_argument("--gdn-fraction", type=float, default=0.40)
    parser.add_argument(
        "--components", nargs="+", choices=("mlp", "gdn"), default=("mlp", "gdn")
    )
    args = parser.parse_args()
    if not fast.qwen35_ane_available():
        raise RuntimeError("The private ANE runtime is unavailable")
    if not fast.qwen35_ane_bank_compiler_available():
        raise RuntimeError("The ANE procedure-bank compiler is unavailable")

    checkpoint = _Checkpoint(args.model)
    model_type = str(checkpoint.config.get("model_type") or "")
    if not patch.qwen_ane_model_type_supported(model_type):
        raise ValueError(f"Unsupported Qwen ANE architecture: {model_type}")

    results: dict[str, Any] = {
        "model": str(args.model),
        "model_type": model_type,
        "layer": args.layer,
        "tokens": args.tokens,
        "repeats": args.repeats,
    }

    if "mlp" in args.components:
        started = time.perf_counter()
        mlp, config, state = _compile_mlp(
            checkpoint, args.layer, args.tokens, args.mlp_fraction
        )
        compile_seconds = time.perf_counter() - started
        input_dim = int(mlp.gate_proj.weight.shape[1]) * 32 // int(mlp.gate_proj.bits)
        x = mx.random.normal((1, args.tokens, input_dim)).astype(
            mlp.gate_proj.scales.dtype
        )
        def baseline():
            return mlp.down_proj(swiglu(mlp.gate_proj(x), mlp.up_proj(x)))

        def hybrid():
            return patch._backend_exact(mlp, x, False)

        gpu_seconds, gpu_output = _measure(baseline, args.repeats)
        ane_seconds, ane_output = _measure(hybrid, args.repeats)
        results["mlp"] = {
            "bits": int(mlp.gate_proj.bits),
            "group_size": int(mlp.gate_proj.group_size),
            "source_dtype": str(mlp.gate_proj.scales.dtype),
            "ane_fraction_effective": state.ane_outputs
            / int(mlp.gate_proj.weight.shape[0]),
            "gpu_median_ms": gpu_seconds * 1e3,
            "ane_gpu_median_ms": ane_seconds * 1e3,
            "speedup": gpu_seconds / ane_seconds,
            "compile_seconds": compile_seconds,
            "accuracy": _compare(gpu_output, ane_output),
            "cpu_offload_eligible": mlp.gate_proj.scales.dtype == mx.float16,
        }
        del config

    if "gdn" in args.components:
        started = time.perf_counter()
        gdn, config, state = _compile_gdn(
            checkpoint, args.layer, args.tokens, args.gdn_fraction
        )
        compile_seconds = time.perf_counter() - started
        qkv = gdn.in_proj_qkv
        input_dim = int(qkv.weight.shape[1]) * 32 // int(qkv.bits)
        x = mx.random.normal((1, args.tokens, input_dim)).astype(qkv.scales.dtype)
        def baseline():
            return tuple(
                patch._tail_qmm_or_linear(linear, x, 8)
                for linear in (
                    gdn.in_proj_qkv,
                    gdn.in_proj_z,
                    gdn.in_proj_b,
                    gdn.in_proj_a,
                )
            )

        def hybrid():
            return patch._gdn_backend_exact(gdn, x, False)

        gpu_seconds, gpu_output = _measure(baseline, args.repeats)
        ane_seconds, ane_output = _measure(hybrid, args.repeats)
        total_outputs = state.z_outputs + state.qkv_outputs
        results["gdn"] = {
            "bits": int(qkv.bits),
            "group_size": int(qkv.group_size),
            "source_dtype": str(qkv.scales.dtype),
            "ane_fraction_effective": state.z_outputs / total_outputs,
            "gpu_median_ms": gpu_seconds * 1e3,
            "ane_gpu_median_ms": ane_seconds * 1e3,
            "speedup": gpu_seconds / ane_seconds,
            "compile_seconds": compile_seconds,
            "accuracy": _compare(gpu_output, ane_output),
            "cpu_offload_eligible": qkv.scales.dtype == mx.float16,
        }

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
