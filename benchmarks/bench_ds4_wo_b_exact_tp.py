#!/usr/bin/env python3
"""Gate an exact TP2 row-sharded DS4 ``wo_b`` decode schedule.

The currently shipped attention output path is, for two tensor ranks::

    current = qmv(W, latent_rank0) + qmv(W, latent_rank1)

Both QMV results are rounded to BF16 before JACCL's BF16 sum.  The earlier
``wo_b`` sharding probe instead summed the two 8192-wide latents first.  That
changes the MXFP8 reduction/rounding boundary and is therefore not lossless.

This benchmark measures a different schedule::

    gathered = stack(latent_rank0, latent_rank1)
    local = exact_m2_r2_fused_sum(W[local_output_rows], gathered)
    output = all_gather(local)

The first exact probe used four output rows per simdgroup.  Sharding 4096
output rows in half therefore reduced the launch from 512 to 256 threadgroups,
while M=2 doubled each thread's accumulator bank.  That recovered only 1.7%
on M3 Ultra and 11.1% on M5 Max -- less than the extra collective costs.

``exact_m2_r2_fused_sum`` is the final rescue gate.  It keeps the existing
DSpark kernel's exact 32-lane K reduction, but assigns two rather than four
output rows to each simdgroup.  A 2048-row shard therefore launches the same
512 threadgroups / 1024 simdgroups as the current 4096-row M=1 QMV, with four
rather than eight FP32 accumulators per thread.  Its epilogue explicitly
rounds both independent QMV contributions to BF16 before adding and rounding
again, matching the current QMV + JACCL BF16-sum boundary without a separate
GPU add dispatch.

The schedule halves the ``wo_b`` bytes read per rank and replaces one 8 KiB
all-sum per layer with a 16 KiB latent all-gather plus a 4 KiB output
all-gather.  The script does not initialize distributed MLX; it isolates the
real-weight compute/parity gate on one Mac and prices the two collective
schedules from measured link constants.

Run only while the public model is unloaded, for example::

    python benchmarks/bench_ds4_wo_b_exact_tp.py \
      ~/.lmstudio/models/Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from contextlib import ExitStack
from functools import lru_cache
from pathlib import Path

import mlx.core as mx
from safetensors import safe_open

from omlx.patches.deepseek_v4 import verify_qmv


LAYERS = 43
HIDDEN_DIMS = 4096
LORA_DIMS = 8192
RESULTS_PER_SIMDGROUP = 2
SIMDGROUPS_PER_THREADGROUP = 2
PROMOTION_THRESHOLDS_MS = {
    "Apple M3 Ultra": 9.946,
    "Apple M5 Max": 9.366,
}


# This is the M=2/R=2 specialization of verify_qmv._SOURCE.  The inner loop,
# MXFP8 conversion, scale placement, FP32 accumulation, and 32-lane simd_sum
# are deliberately unchanged.  Only the independent output-row tile narrows
# from four to two and the two BF16 contribution stores collapse into the
# explicitly double-rounded sum in the epilogue.
_EXACT_M2_R2_FUSED_SUM_SOURCE = r"""
    constexpr int M = 2;
    constexpr int VALUES_PER_THREAD = 8;
    constexpr int BLOCK_SIZE = VALUES_PER_THREAD * 32;
    constexpr int RESULTS_PER_SIMDGROUP = 2;
    constexpr int OUTPUTS_PER_THREADGROUP = 4;

    const uint simd_group = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int output_base =
        int(threadgroup_position_in_grid.y) * OUTPUTS_PER_THREADGROUP
        + int(simd_group) * RESULTS_PER_SIMDGROUP;
    if (output_base >= N) {
        return;
    }

    const device uint8_t* weight_ptr = (const device uint8_t*)weight
        + output_base * K + int(lane) * VALUES_PER_THREAD;
    const device T* input_ptr = (const device T*)input
        + int(lane) * VALUES_PER_THREAD;
    device T* output_ptr = (device T*)output + output_base;
    const device uint8_t* scale_ptr = (const device uint8_t*)scales
        + output_base * (K / GS)
        + int(lane) / (GS / VALUES_PER_THREAD);

    float accum[M][RESULTS_PER_SIMDGROUP] = {{0}};
    for (int k = 0; k < K; k += BLOCK_SIZE) {
        float input_values[M][VALUES_PER_THREAD];
        for (int row = 0; row < M; ++row) {
            for (int idx = 0; idx < VALUES_PER_THREAD; ++idx) {
                input_values[row][idx] = float(input_ptr[row * K + idx]);
            }
        }
        for (int result = 0; result < RESULTS_PER_SIMDGROUP; ++result) {
            const device uint8_t* row_weight = weight_ptr + result * K;
            const device uint8_t* row_scale =
                scale_ptr + result * (K / GS);
            const float scale = omlx_fp8_scale(row_scale[0]);
            for (int row = 0; row < M; ++row) {
                accum[row][result] += omlx_mxfp8_dot<VALUES_PER_THREAD>(
                    row_weight, input_values[row], scale);
            }
        }
        weight_ptr += BLOCK_SIZE;
        scale_ptr += BLOCK_SIZE / GS;
        input_ptr += BLOCK_SIZE;
    }

    for (int result = 0; result < RESULTS_PER_SIMDGROUP; ++result) {
        // Each simd_sum has exactly the same 32-lane inputs and reduction
        // order as two independent MLX M=1 qmv_fast calls.
        const float value_rank0 = simd_sum(accum[0][result]);
        const float value_rank1 = simd_sum(accum[1][result]);
        if (lane == 0) {
            // Preserve both rounding points from current distributed decode:
            // QMV FP32 -> BF16 on each rank, then JACCL BF16 add -> BF16.
            const T rounded_rank0 = T(value_rank0);
            const T rounded_rank1 = T(value_rank1);
            output_ptr[result] = T(
                float(rounded_rank0) + float(rounded_rank1));
        }
    }
"""


@lru_cache(maxsize=1)
def _exact_m2_r2_fused_sum_kernel():
    return mx.fast.metal_kernel(
        name="omlx_bench_ds4_wo_b_exact_m2_r2_fused_sum",
        input_names=["input", "weight", "scales"],
        output_names=["output"],
        header=verify_qmv._HEADER,
        source=_EXACT_M2_R2_FUSED_SUM_SOURCE,
        ensure_row_contiguous=True,
    )


class _MXFP8Shard:
    """Minimum QuantizedLinear protocol consumed by ``exact_verify_qmv``."""

    bits = 8
    group_size = 32
    mode = "mxfp8"

    def __init__(self, weight: mx.array, scales: mx.array):
        self.weight = weight
        self.scales = scales

    def get(self, name: str, default=None):
        return default

    def __contains__(self, name: str) -> bool:
        return False


def _qmm(value: mx.array, weight: mx.array, scales: mx.array) -> mx.array:
    return mx.quantized_matmul(
        value,
        weight,
        scales=scales,
        biases=None,
        transpose=True,
        group_size=32,
        bits=8,
        mode="mxfp8",
    )


def _exact_m2_r2_fused_sum(
    module: _MXFP8Shard,
    gathered_latents: mx.array,
) -> mx.array:
    """Project two rank contributions with exact M=1 reductions and BF16 sum."""

    if gathered_latents.ndim != 2 or tuple(gathered_latents.shape) != (
        2,
        LORA_DIMS,
    ):
        raise ValueError("rescue gate requires exactly two 8192-wide decode rows")
    if gathered_latents.dtype not in (mx.bfloat16, mx.float16):
        raise ValueError("rescue gate requires BF16 or FP16 decode rows")
    output_dims = int(module.scales.shape[0])
    if (
        output_dims != HIDDEN_DIMS // 2
        or int(module.bits) != 8
        or int(module.group_size) != 32
        or module.mode != "mxfp8"
    ):
        raise ValueError("rescue gate requires the DS4 TP2 MXFP8 wo_b shard")

    flat = mx.contiguous(gathered_latents)
    (output,) = _exact_m2_r2_fused_sum_kernel()(
        inputs=[flat, module.weight, module.scales],
        template=[
            ("T", flat.dtype),
            ("K", LORA_DIMS),
            ("N", output_dims),
            ("GS", int(module.group_size)),
        ],
        # mx.fast grid dimensions are threads, not threadgroups. With a
        # (32, 2, 1) threadgroup and four output rows per threadgroup this is
        # 512 threadgroups / 1024 simdgroups for the 2048-row TP2 shard.
        grid=(32, output_dims // RESULTS_PER_SIMDGROUP, 1),
        threadgroup=(32, SIMDGROUPS_PER_THREADGROUP, 1),
        output_shapes=[(1, output_dims)],
        output_dtypes=[flat.dtype],
    )
    return output


def _load_wo_b(model: Path):
    index = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    stack = ExitStack()
    files = {}
    modules = []
    for layer in range(LAYERS):
        weight_key = f"layers.{layer}.attn.wo_b.weight"
        scales_key = f"layers.{layer}.attn.wo_b.scales"
        filename = index[weight_key]
        if filename not in files:
            files[filename] = stack.enter_context(
                safe_open(model / filename, framework="numpy")
            )
        source = files[filename]
        weight = mx.array(source.get_tensor(weight_key))
        scales = mx.array(source.get_tensor(scales_key))
        split = int(scales.shape[0]) // 2
        modules.append(
            (
                weight,
                scales,
                _MXFP8Shard(weight[:split], scales[:split]),
                _MXFP8Shard(weight[split:], scales[split:]),
            )
        )
    mx.eval(
        [
            value
            for weight, scales, _, _ in modules
            for value in (weight, scales)
        ]
    )
    return stack, modules


def _run_current(modules, local_latent: mx.array) -> None:
    """One rank's current full-output QMV contribution."""

    for weight, scales, _, _ in modules:
        mx.eval(_qmm(local_latent, weight, scales))


def _run_candidate(
    modules,
    gathered_latents: mx.array,
    *,
    shard: int = 0,
) -> None:
    """One rank's fused exact-M=2, half-output rescue kernel."""

    for _, _, left, right in modules:
        projected = _exact_m2_r2_fused_sum(
            left if shard == 0 else right,
            gathered_latents,
        )
        mx.eval(projected)


def _time_abba(modules, left, gathered, *, cycles: int):
    # Compile and fault both shapes before timing, then balance allocator/cache
    # inheritance by measuring A-B-B-A in every cycle.
    _run_current(modules, left)
    _run_candidate(modules, gathered)
    mx.synchronize()
    current = []
    candidate = []
    for _ in range(cycles):
        for target, fn in (
            (current, lambda: _run_current(modules, left)),
            (candidate, lambda: _run_candidate(modules, gathered)),
            (candidate, lambda: _run_candidate(modules, gathered)),
            (current, lambda: _run_current(modules, left)),
        ):
            mx.synchronize()
            started = time.perf_counter()
            fn()
            mx.synchronize()
            target.append(time.perf_counter() - started)
    return current, candidate


def _parity(modules, left, right):
    exact = 0
    elements = 0
    max_abs = 0.0
    sum_abs = 0.0
    sum_squared = 0.0
    reference_squared = 0.0
    gathered = mx.concatenate([left, right], axis=0)
    for weight, scales, first, second in modules:
        # This is the shipped two-rank rounding boundary: each rank materializes
        # a BF16 QMV result, then the two BF16 tensors are added.
        reference = _qmm(left, weight, scales) + _qmm(right, weight, scales)
        first_rows = _exact_m2_r2_fused_sum(first, gathered)
        second_rows = _exact_m2_r2_fused_sum(second, gathered)
        candidate = mx.concatenate(
            [first_rows, second_rows],
            axis=-1,
        )
        mx.eval(reference, candidate)
        delta = mx.abs(reference.astype(mx.float32) - candidate.astype(mx.float32))
        exact += int(mx.sum(reference == candidate).item())
        elements += int(reference.size)
        max_abs = max(max_abs, float(mx.max(delta).item()))
        sum_abs += float(mx.sum(delta).item())
        sum_squared += float(mx.sum(delta * delta).item())
        reference_squared += float(
            mx.sum(reference.astype(mx.float32) ** 2).item()
        )
    return {
        "exact_elements": exact,
        "elements": elements,
        "exact_fraction": exact / elements if elements else 1.0,
        "max_abs": max_abs,
        "mean_abs": sum_abs / elements if elements else 0.0,
        "rmse": math.sqrt(sum_squared / elements) if elements else 0.0,
        "relative_l2": (
            math.sqrt(sum_squared / reference_squared)
            if reference_squared > 0
            else 0.0
        ),
    }


def _collective_price(*, latency_s: float, bandwidth: float):
    bf16 = 2
    current_payload = HIDDEN_DIMS * bf16
    latent_payload = LORA_DIMS * bf16
    local_output_payload = (HIDDEN_DIMS // 2) * bf16
    current = LAYERS * (latency_s + current_payload / bandwidth)
    candidate = LAYERS * (
        2 * latency_s
        + (latent_payload + local_output_payload) / bandwidth
    )
    return {
        "current_seconds_43_layers": current,
        "candidate_seconds_43_layers": candidate,
        "added_seconds_43_layers": candidate - current,
        "current_collectives_per_layer": 1,
        "candidate_collectives_per_layer": 2,
        "current_payload_bytes_per_layer": current_payload,
        "candidate_payload_bytes_per_layer": (
            latent_payload + local_output_payload
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--collective-latency-us", type=float, default=31.15)
    parser.add_argument("--collective-bandwidth-gbps", type=float, default=6.178)
    parser.add_argument(
        "--promotion-threshold-ms",
        type=float,
        help=(
            "override the device-specific strict compute threshold "
            "(M3 Ultra 9.946 ms; M5 Max 9.366 ms)"
        ),
    )
    parser.add_argument("--skip-parity", action="store_true")
    args = parser.parse_args()

    stack, modules = _load_wo_b(args.model.expanduser())
    try:
        mx.random.seed(17)
        left = mx.random.normal((1, LORA_DIMS)).astype(mx.bfloat16)
        right = mx.random.normal((1, LORA_DIMS)).astype(mx.bfloat16)
        gathered = mx.concatenate([left, right], axis=0)
        mx.eval(left, right, gathered)
        current, candidate = _time_abba(
            modules,
            left,
            gathered,
            cycles=args.cycles,
        )
        current_median = statistics.median(current)
        candidate_median = statistics.median(candidate)
        device_name = str(mx.device_info().get("device_name", "unknown"))
        promotion_threshold_ms = (
            args.promotion_threshold_ms
            if args.promotion_threshold_ms is not None
            else PROMOTION_THRESHOLDS_MS.get(device_name)
        )
        collective = _collective_price(
            latency_s=args.collective_latency_us * 1e-6,
            bandwidth=args.collective_bandwidth_gbps * 1e9,
        )
        full_bytes = sum(
            weight.nbytes + scales.nbytes
            for weight, scales, _, _ in modules
        )
        local_bytes = sum(
            first.weight.nbytes + first.scales.nbytes
            for _, _, first, _ in modules
        )
        parity = None if args.skip_parity else _parity(modules, left, right)
        compute_pass = (
            None
            if promotion_threshold_ms is None
            else candidate_median * 1000 < promotion_threshold_ms
        )
        parity_pass = (
            None
            if parity is None
            else parity["exact_fraction"] == 1.0 and parity["max_abs"] == 0.0
        )
        current_total = current_median + collective["current_seconds_43_layers"]
        rescue_total = candidate_median + collective["candidate_seconds_43_layers"]
        priced_end_to_end_pass = rescue_total < current_total
        result = {
            "device_name": device_name,
            "candidate_variant": "exact_m2_r2_fused_bf16_sum",
            "layers": len(modules),
            "kernel_geometry": {
                "current_output_rows": HIDDEN_DIMS,
                "current_results_per_simdgroup": 4,
                "current_threadgroups_per_layer": HIDDEN_DIMS // 8,
                "current_simdgroups_per_layer": HIDDEN_DIMS // 4,
                "rescue_output_rows": HIDDEN_DIMS // 2,
                "rescue_input_rows": 2,
                "rescue_results_per_simdgroup": RESULTS_PER_SIMDGROUP,
                "rescue_threadgroups_per_layer": (
                    (HIDDEN_DIMS // 2)
                    // (RESULTS_PER_SIMDGROUP * SIMDGROUPS_PER_THREADGROUP)
                ),
                "rescue_simdgroups_per_layer": (
                    (HIDDEN_DIMS // 2) // RESULTS_PER_SIMDGROUP
                ),
                "reduction_lanes": 32,
                "fused_bf16_contribution_sum": True,
            },
            "full_wo_b_bytes_per_rank_current": full_bytes,
            "local_wo_b_bytes_per_rank_candidate": local_bytes,
            "weight_bytes_saved_per_rank_per_token": full_bytes - local_bytes,
            "current_compute_seconds_43_layers": current_median,
            "rescue_compute_seconds_43_layers": candidate_median,
            "candidate_compute_seconds_43_layers": candidate_median,
            "rescue_isolated_compute_speedup": current_median / candidate_median,
            "collectives": collective,
            "current_compute_plus_collective_seconds": current_total,
            "rescue_compute_plus_collective_seconds": rescue_total,
            "candidate_compute_plus_collective_seconds": rescue_total,
            "promotion_gate": {
                "strict_compute_threshold_ms": promotion_threshold_ms,
                "compute_pass": compute_pass,
                "priced_end_to_end_pass": priced_end_to_end_pass,
                "requires_exact_fraction": 1.0,
                "parity_pass": parity_pass,
                "promote": bool(
                    compute_pass and parity_pass and priced_end_to_end_pass
                ),
                "known_thresholds_ms": PROMOTION_THRESHOLDS_MS,
            },
            "parity": parity,
            "current_samples_ms": [value * 1000 for value in current],
            "rescue_samples_ms": [value * 1000 for value in candidate],
            "candidate_samples_ms": [value * 1000 for value in candidate],
        }
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        stack.close()


if __name__ == "__main__":
    main()
