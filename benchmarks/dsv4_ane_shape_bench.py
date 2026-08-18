# SPDX-License-Identifier: Apache-2.0
"""Per-shape ANE/GPU feasibility microbench for DeepSeek-V4-Flash prefill.

Times each dense projection candidate at a fixed 2048-token prefill shape:
the current GPU path (stock mxfp8 quantized matmul) against a hybrid split
that runs an INT8 channel prefix on both ANEs while the GPU computes an
affine suffix through the existing dual merge primitive. The suffix uses a
q4 stand-in requantized from the same weights because the primitive does
not accept an mxfp8 suffix yet; q4/q8/mxfp8 suffix-half GPU timings are
reported separately so the Phase B suffix format cost is visible.

Run with OMLX_ANE_PROFILE=1 to collect per-op ANE phase counters.

Usage:
    python benchmarks/dsv4_ane_shape_bench.py [--fractions 0.4,0.5,0.6]
"""

import argparse
import statistics
import time

import mlx.core as mx

from omlx.custom_kernels.qwen35_prefill import fast

SEQ = 2048
# [name, out_features, in_features] of the v1 offload candidates.
SHAPES = (
    ("shared_gate_up", 4096, 4096),
    ("wq_b", 32768, 1024),
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


def bench_shape(name, out_features, in_features, fractions):
    weight = mx.random.normal((out_features, in_features)).astype(mx.bfloat16)
    x = mx.random.normal((1, SEQ, in_features)).astype(mx.bfloat16)
    mx.eval(weight, x)

    w8, s8 = mx.quantize(weight, group_size=32, bits=8, mode="mxfp8")
    mx.eval(w8, s8)
    gpu_ms = median_ms(
        lambda: mx.quantized_matmul(
            x, w8, s8, None, transpose=True, group_size=32, bits=8, mode="mxfp8"
        )
    )
    print(
        f"[{name}] out={out_features} in={in_features} seq={SEQ} "
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

    for fraction in fractions:
        per_instance = int(out_features * fraction / 2 // 128) * 128
        gpu_rows = out_features - 2 * per_instance
        if per_instance < 128 or gpu_rows < 64 or gpu_rows % 64:
            print(f"[{name}]   fraction={fraction}: skipped (alignment)")
            continue
        dense = weight.astype(mx.float32)
        dense0 = mx.contiguous(dense[:per_instance])
        dense1 = mx.contiguous(dense[per_instance : 2 * per_instance])
        # The native compiler reads the buffers directly; lazy inputs crash.
        mx.eval(dense0, dense1)
        compile_start = time.perf_counter()
        ane0 = fast.qwen35_ane_compile_linear(dense0, SEQ, 1)
        ane1 = fast.qwen35_ane_compile_linear(dense1, SEQ, 2)
        compile_s = time.perf_counter() - compile_start

        sw, ss, sb = mx.quantize(
            mx.contiguous(weight[2 * per_instance :]), group_size=64, bits=4
        )
        sw, ss, sb = (
            mx.contiguous(sw),
            mx.contiguous(ss.astype(x.dtype)),
            mx.contiguous(sb.astype(x.dtype)),
        )
        mx.eval(sw, ss, sb)

        fast.qwen35_ane_profile_set_enabled(True)
        fast.qwen35_ane_profile_reset()
        hybrid_ms = median_ms(
            lambda w=sw, s=ss, b=sb, m0=ane0, m1=ane1: (
                fast.qwen35_ane_dual_affine_qmm_t(x, w, s, b, m0, m1, 4, 8, 64)
            )
        )
        snapshot = fast.qwen35_ane_profile_snapshot()
        ops = max(1, int(snapshot["gdn"]["operations"]))
        ane_eval = snapshot["gdn"]["ane0_eval_ns"] / ops / 1e6
        pack = snapshot["gdn"]["pack_ns"] / ops / 1e6
        gpu_qmm = snapshot["gdn"]["gpu_qmm_ns"] / ops / 1e6
        fast.qwen35_ane_profile_set_enabled(False)
        speedup = gpu_ms / hybrid_ms if hybrid_ms else 0.0
        print(
            f"[{name}]   fraction={fraction} ane_rows={2 * per_instance} "
            f"gpu_rows={gpu_rows} hybrid={hybrid_ms:.3f}ms "
            f"(vs gpu {speedup:.2f}x) compile={compile_s:.1f}s "
            f"ane_eval/op={ane_eval:.3f}ms pack/op={pack:.3f}ms "
            f"gpu_qmm/op={gpu_qmm:.3f}ms"
        )


def bench_wo_a_grouped():
    """Correctness and timing for the 8-group wo_a projection on ANE."""
    groups, out_per_group, in_per_group = 8, 1024, 4096
    out_total = groups * out_per_group
    in_total = groups * in_per_group
    w = mx.random.normal((groups, out_per_group, in_per_group)).astype(mx.bfloat16)
    x = mx.random.normal((1, SEQ, in_total)).astype(mx.bfloat16)
    mx.eval(w, x)
    x_groups = [
        mx.contiguous(x[..., g * in_per_group : (g + 1) * in_per_group])
        for g in range(groups)
    ]
    mx.eval(*x_groups)

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
    model = fast.qwen35_ane_compile_linear_grouped(flat, SEQ, 1, groups)
    junk = mx.random.normal((64, in_total)).astype(mx.bfloat16)
    sw, ss, sb = mx.quantize(junk, group_size=64, bits=4)
    sw, ss, sb = (
        mx.contiguous(sw),
        mx.contiguous(ss.astype(x.dtype)),
        mx.contiguous(sb.astype(x.dtype)),
    )
    mx.eval(sw, ss, sb)
    combined = fast.qwen35_ane_affine_qmm_t(
        x, sw, ss, sb, model, 4, 8, 64
    )
    ane_out = combined[..., :out_total].astype(mx.float32)
    mx.eval(ane_out)
    for g in range(groups):
        reference = x_groups[g].astype(mx.float32) @ w[g].astype(mx.float32).T
        block = ane_out[..., g * out_per_group : (g + 1) * out_per_group]
        cosine = mx.sum(reference * block) / (
            mx.linalg.norm(reference) * mx.linalg.norm(block)
        )
        print(f"[wo_a_grouped]   group{g} cosine={float(cosine):.5f}")

    # Timing: dual grouped prefix with a tiny junk suffix; read the ANE
    # counters since the suffix here is not fraction-realistic.
    for rows_per_group_per_instance in (256, 128):
        prefix0 = mx.contiguous(
            mx.concatenate(
                [w[g][:rows_per_group_per_instance] for g in range(groups)]
            ).astype(mx.float32)
        )
        prefix1 = mx.contiguous(
            mx.concatenate(
                [
                    w[g][
                        rows_per_group_per_instance : 2
                        * rows_per_group_per_instance
                    ]
                    for g in range(groups)
                ]
            ).astype(mx.float32)
        )
        mx.eval(prefix0, prefix1)
        model0 = fast.qwen35_ane_compile_linear_grouped(prefix0, SEQ, 1, groups)
        model1 = fast.qwen35_ane_compile_linear_grouped(prefix1, SEQ, 2, groups)
        fast.qwen35_ane_profile_set_enabled(True)
        fast.qwen35_ane_profile_reset()
        hybrid_ms = median_ms(
            lambda m0=model0, m1=model1: (
                fast.qwen35_ane_dual_affine_qmm_t(x, sw, ss, sb, m0, m1, 4, 8, 64)
            )
        )
        snapshot = fast.qwen35_ane_profile_snapshot()
        ops = max(1, int(snapshot["gdn"]["operations"]))
        ane_eval = snapshot["gdn"]["ane0_eval_ns"] / ops / 1e6
        pack = snapshot["gdn"]["pack_ns"] / ops / 1e6
        fast.qwen35_ane_profile_set_enabled(False)
        fraction = 2 * rows_per_group_per_instance / out_per_group
        print(
            f"[wo_a_grouped]   fraction={fraction:.2f} "
            f"ane_rows/inst={groups * rows_per_group_per_instance} "
            f"wall_with_junk_suffix={hybrid_ms:.3f}ms "
            f"ane_eval/op={ane_eval:.3f}ms pack/op={pack:.3f}ms"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fractions", default="0.4,0.5,0.6")
    parser.add_argument("--wo-a", action="store_true", help="grouped wo_a only")
    args = parser.parse_args()
    fractions = [float(value) for value in args.fractions.split(",")]

    if not fast.qwen35_ane_available():
        raise SystemExit("Private ANE runtime unavailable")
    if args.wo_a:
        bench_wo_a_grouped()
        return
    for name, out_features, in_features in SHAPES:
        bench_shape(name, out_features, in_features, fractions)


if __name__ == "__main__":
    main()
