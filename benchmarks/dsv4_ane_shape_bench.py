# SPDX-License-Identifier: Apache-2.0
"""Per-shape ANE/CPU/GPU microbench for DeepSeek-V4-Flash prefill.

Times each dense projection candidate at a fixed-shape prefill width:
the current GPU path (stock mxfp8 quantized matmul) against a hybrid split
that runs an INT8 channel prefix on both ANEs, an optional FP16 middle on
the CPU, and an affine-q8 suffix on the GPU through the shared Qwen runtime.
The synthetic shapes need less than 1 GiB. A real-model smoke benchmark is
intentionally separate and opt-in because the full checkpoint needs more
than 96 GiB of unified memory.

Run with OMLX_ANE_PROFILE=1 to collect per-op ANE phase counters.

Usage:
    python benchmarks/dsv4_ane_shape_bench.py --sequence-length 4096 \
        --ane-fractions 0.4,0.5,0.6 --cpu-fractions 0,0.05,0.1,0.15
"""

import argparse
import statistics
import time

import mlx.core as mx

from omlx.custom_kernels.qwen35_prefill import fast

# [name, combined out_features, in_features] of the offload candidates.
# shared_gate_up is the stacked gate/up pair. stacked_wq_b includes the
# attention (32768) and indexer (8192) projections that share q_residual.
# The attention_input variants stack every projection that consumes the
# residual stream before attention; their widths correspond to local,
# ratio-128 compressed, and ratio-4 sparse layers respectively.
SHAPES = (
    ("shared_gate_up", 4096, 4096),
    ("shared_down", 4096, 2048),
    ("wq_b", 32768, 1024),
    ("stacked_wq_b", 40960, 1024),
    ("attention_input_local", 1536, 4096),
    ("attention_input_ratio128", 2560, 4096),
    ("attention_input_ratio4", 4160, 4096),
    ("wo_b", 4096, 8192),
)


def median_ms(fn, warmup=3, iters=15):
    for _ in range(warmup):
        mx.eval(fn())
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        mx.eval(fn())
        times.append((time.perf_counter() - start) * 1e3)
    return statistics.median(times)


def bench_shape(
    name,
    out_features,
    in_features,
    sequence_length,
    ane_fractions,
    cpu_fractions,
    cpu_threads,
    cpu_shared_resource,
):
    weight = mx.random.normal((out_features, in_features)).astype(mx.bfloat16)
    x = mx.random.normal((1, sequence_length, in_features)).astype(mx.bfloat16)
    mx.eval(weight, x)

    w8, s8 = mx.quantize(weight, group_size=32, bits=8, mode="mxfp8")
    mx.eval(w8, s8)
    gpu_ms = median_ms(
        lambda: mx.quantized_matmul(
            x, w8, s8, None, transpose=True, group_size=32, bits=8, mode="mxfp8"
        )
    )
    print(
        f"[{name}] out={out_features} in={in_features} seq={sequence_length} "
        f"gpu_mxfp8_full={gpu_ms:.3f}ms"
    )

    # Suffix-format cost preview on a representative half of the rows.
    half_rows = out_features // 2
    for label, group_size, bits, mode in (
        ("q4_gs64", 64, 4, "affine"),
        ("q8_gs64", 64, 8, "affine"),
        ("mxfp8_gs32", 32, 8, "mxfp8"),
    ):
        quantized = mx.quantize(
            weight[:half_rows], group_size=group_size, bits=bits, mode=mode
        )
        mx.eval(*quantized)
        biases = quantized[2] if len(quantized) == 3 else None
        half_ms = median_ms(
            lambda q=quantized, b=biases, gs=group_size, bt=bits, md=mode: (
                mx.quantized_matmul(
                    x, q[0], q[1], b, transpose=True, group_size=gs, bits=bt, mode=md
                )
            )
        )
        print(f"[{name}]   gpu_half_{label}={half_ms:.3f}ms")

    for ane_fraction in ane_fractions:
        per_instance = int(out_features * ane_fraction / 2 // 128) * 128
        ane_rows = 2 * per_instance
        gpu_rows = out_features - ane_rows
        if per_instance < 128 or gpu_rows < 64 or gpu_rows % 64:
            print(f"[{name}]   ane_fraction={ane_fraction}: skipped (alignment)")
            continue
        dense = weight.astype(mx.float32)
        dense0 = mx.contiguous(dense[:per_instance])
        dense1 = mx.contiguous(dense[per_instance : 2 * per_instance])
        # The native compiler reads the buffers directly; lazy inputs crash.
        mx.eval(dense0, dense1)
        compile_start = time.perf_counter()
        ane0 = fast.qwen35_ane_compile_linear(dense0, sequence_length, 1)
        ane1 = fast.qwen35_ane_compile_linear(dense1, sequence_length, 2)
        compile_s = time.perf_counter() - compile_start

        for cpu_fraction in cpu_fractions:
            cpu_rows = int(out_features * cpu_fraction // 64) * 64
            gpu_start = ane_rows + cpu_rows
            gpu_rows = out_features - gpu_start
            if gpu_rows < 64 or gpu_rows % 64:
                print(
                    f"[{name}]   ane_fraction={ane_fraction} "
                    f"cpu_fraction={cpu_fraction}: skipped (alignment)"
                )
                continue
            cpu_weight = None
            if cpu_rows:
                cpu_weight = mx.contiguous(
                    weight[ane_rows:gpu_start].astype(mx.float16)
                )
            sw, ss, sb = mx.quantize(
                mx.contiguous(weight[gpu_start:]), group_size=64, bits=8
            )
            sw, ss, sb = (
                mx.contiguous(sw),
                mx.contiguous(ss.astype(x.dtype)),
                mx.contiguous(sb.astype(x.dtype)),
            )
            arrays = [sw, ss, sb]
            if cpu_weight is not None:
                arrays.append(cpu_weight)
            mx.eval(*arrays)

            def hybrid():
                if cpu_weight is None:
                    return fast.qwen35_ane_dual_affine_qmm_t(
                        x, sw, ss, sb, ane0, ane1, 8, 8, 64
                    )
                return fast.qwen35_ane_dual_cpu_fp16_affine_qmm_t(
                    x,
                    cpu_weight,
                    sw,
                    ss,
                    sb,
                    ane0,
                    ane1,
                    8,
                    8,
                    64,
                    1,
                    cpu_threads,
                    cpu_shared_resource,
                )

            fast.qwen35_ane_profile_set_enabled(True)
            fast.qwen35_ane_profile_reset()
            hybrid_ms = median_ms(hybrid)
            snapshot = fast.qwen35_ane_profile_snapshot()
            metrics = snapshot["gdn"]
            ops = max(1, int(metrics["operations"]))
            ane_eval = metrics["ane0_eval_ns"] / ops / 1e6
            pack = metrics["pack_ns"] / ops / 1e6
            gpu_qmm = metrics["gpu_qmm_ns"] / ops / 1e6
            cpu_mm = metrics["cpu_matmul_ns"] / ops / 1e6
            fast.qwen35_ane_profile_set_enabled(False)
            speedup = gpu_ms / hybrid_ms if hybrid_ms else 0.0
            print(
                f"[{name}]   ane_fraction={ane_fraction} "
                f"cpu_fraction={cpu_fraction} ane_rows={ane_rows} "
                f"cpu_rows={cpu_rows} gpu_rows={gpu_rows} "
                f"hybrid={hybrid_ms:.3f}ms (vs gpu {speedup:.2f}x) "
                f"compile={compile_s:.1f}s ane_eval/op={ane_eval:.3f}ms "
                f"cpu_mm/op={cpu_mm:.3f}ms pack/op={pack:.3f}ms "
                f"gpu_qmm/op={gpu_qmm:.3f}ms"
            )


def bench_wo_a_grouped(sequence_length):
    """Correctness and timing for the 8-group wo_a projection on ANE."""
    groups, out_per_group, in_per_group = 8, 1024, 4096
    out_total = groups * out_per_group
    in_total = groups * in_per_group
    w = mx.random.normal((groups, out_per_group, in_per_group)).astype(mx.bfloat16)
    x = mx.random.normal((1, sequence_length, in_total)).astype(mx.bfloat16)
    mx.eval(w, x)
    x_groups = [
        mx.contiguous(x[..., g * in_per_group : (g + 1) * in_per_group])
        for g in range(groups)
    ]
    x_grouped = mx.stack(x_groups, axis=1)
    mx.eval(*x_groups, x_grouped)

    quantized = [
        mx.quantize(w[g], group_size=32, bits=8, mode="mxfp8") for g in range(groups)
    ]
    for pair in quantized:
        mx.eval(*pair)

    def gpu_all():
        outs = [
            mx.quantized_matmul(
                x_groups[g],
                quantized[g][0],
                quantized[g][1],
                None,
                transpose=True,
                group_size=32,
                bits=8,
                mode="mxfp8",
            )
            for g in range(groups)
        ]
        return mx.concatenate(outs, axis=-1)

    gpu_ms = median_ms(gpu_all)
    print(
        f"[wo_a_grouped] groups={groups} out={out_total} in_total={in_total} "
        f"gpu_mxfp8_full={gpu_ms:.3f}ms"
    )

    # Correctness: one full-output grouped program against an fp32 reference.
    flat = mx.contiguous(w.reshape(out_total, in_per_group).astype(mx.float32))
    mx.eval(flat)
    model = fast.qwen35_ane_compile_linear_grouped(flat, sequence_length, 1, groups)
    junk = mx.random.normal((64, in_total)).astype(mx.bfloat16)
    sw, ss, sb = mx.quantize(junk, group_size=64, bits=4)
    sw, ss, sb = (
        mx.contiguous(sw),
        mx.contiguous(ss.astype(x.dtype)),
        mx.contiguous(sb.astype(x.dtype)),
    )
    mx.eval(sw, ss, sb)
    combined = fast.qwen35_ane_affine_qmm_t(x, sw, ss, sb, model, 4, 8, 64)
    ane_out = combined[..., :out_total].astype(mx.float32)
    mx.eval(ane_out)
    for g in range(groups):
        reference = x_groups[g].astype(mx.float32) @ w[g].astype(mx.float32).T
        block = ane_out[..., g * out_per_group : (g + 1) * out_per_group]
        cosine = mx.sum(reference * block) / (
            mx.linalg.norm(reference) * mx.linalg.norm(block)
        )
        print(f"[wo_a_grouped]   group{g} cosine={float(cosine):.5f}")

    # Timing: the production dual-grouped primitive keeps the compact GPU
    # suffix grouped and overlaps it with both ANE instances.
    for rows_per_group_per_instance in (256, 128):
        prefix0 = mx.contiguous(
            mx.concatenate(
                [w[g][:rows_per_group_per_instance] for g in range(groups)]
            ).astype(mx.float32)
        )
        prefix1 = mx.contiguous(
            mx.concatenate(
                [
                    w[g][rows_per_group_per_instance : 2 * rows_per_group_per_instance]
                    for g in range(groups)
                ]
            ).astype(mx.float32)
        )
        mx.eval(prefix0, prefix1)
        model0 = fast.qwen35_ane_compile_linear_grouped(
            prefix0, sequence_length, 1, groups
        )
        model1 = fast.qwen35_ane_compile_linear_grouped(
            prefix1, sequence_length, 2, groups
        )
        ane_rows = 2 * rows_per_group_per_instance
        suffix = mx.contiguous(
            mx.concatenate([w[g][ane_rows:] for g in range(groups)])
        )
        sw, ss, sb = mx.quantize(suffix, group_size=64, bits=8)
        sw, ss, sb = (
            mx.contiguous(sw),
            mx.contiguous(ss.astype(x.dtype)),
            mx.contiguous(sb.astype(x.dtype)),
        )
        mx.eval(sw, ss, sb)
        fast.qwen35_ane_profile_set_enabled(True)
        fast.qwen35_ane_profile_reset()
        hybrid_ms = median_ms(
            lambda m0=model0, m1=model1, qw=sw, qs=ss, qb=sb: (
                fast.qwen35_ane_dual_grouped_affine_qmm_t(
                    x_grouped, qw, qs, qb, m0, m1, groups, 8, 8, 64
                )
            )
        )
        snapshot = fast.qwen35_ane_profile_snapshot()
        ops = max(1, int(snapshot["gdn"]["operations"]))
        ane_eval = snapshot["gdn"]["ane0_eval_ns"] / ops / 1e6
        pack = snapshot["gdn"]["pack_ns"] / ops / 1e6
        fast.qwen35_ane_profile_set_enabled(False)
        fraction = 2 * rows_per_group_per_instance / out_per_group
        speedup = gpu_ms / hybrid_ms if hybrid_ms else 0.0
        print(
            f"[wo_a_grouped]   fraction={fraction:.2f} "
            f"ane_rows/inst={groups * rows_per_group_per_instance} "
            f"hybrid={hybrid_ms:.3f}ms (vs gpu {speedup:.2f}x) "
            f"ane_eval/op={ane_eval:.3f}ms pack/op={pack:.3f}ms"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument(
        "--ane-fractions", "--fractions", dest="ane_fractions", default="0.4,0.5,0.6"
    )
    parser.add_argument("--cpu-fractions", default="0,0.05,0.1,0.15")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument(
        "--disable-cpu-shared-resource",
        action="store_true",
        help="Use ordinary CPU scheduling instead of the shared-cluster policy",
    )
    parser.add_argument(
        "--shapes",
        default="shared_gate_up,wq_b,stacked_wq_b,wo_b",
        help="Comma-separated synthetic shapes to run",
    )
    parser.add_argument("--wo-a", action="store_true", help="grouped wo_a only")
    args = parser.parse_args()
    ane_fractions = [float(value) for value in args.ane_fractions.split(",")]
    cpu_fractions = [float(value) for value in args.cpu_fractions.split(",")]

    if not fast.qwen35_ane_available():
        raise SystemExit("Private ANE runtime unavailable")
    if args.wo_a:
        bench_wo_a_grouped(args.sequence_length)
        return
    selected = {value.strip() for value in args.shapes.split(",") if value.strip()}
    unknown = selected - {shape[0] for shape in SHAPES}
    if unknown:
        raise SystemExit(f"Unknown shapes: {', '.join(sorted(unknown))}")
    for name, out_features, in_features in SHAPES:
        if name in selected:
            bench_shape(
                name,
                out_features,
                in_features,
                args.sequence_length,
                ane_fractions,
                cpu_fractions,
                args.cpu_threads,
                not args.disable_cpu_shared_resource,
            )


if __name__ == "__main__":
    main()
