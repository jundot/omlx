#!/usr/bin/env python3
"""Benchmark Qwen4 direct-index sparse GQA against the gathered MLX path."""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx
import numpy as np

from omlx.custom_kernels.glm_moe_dsa import fast
from omlx.patches import mlx_vlm_qwen4_exp_compat as compat


compat.apply_mlx_vlm_qwen4_exp_compat_patch()
from mlx_vlm.models.qwen4_exp import qsa_fast  # noqa: E402


def _time(call, repetitions: int):
    samples = []
    output = None
    for _ in range(repetitions):
        start = time.perf_counter()
        output = call()
        mx.eval(output)
        mx.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples, output


def _portable(queries, keys, values, selected, selected_valid):
    query_tokens = queries.shape[2]
    selected_keys = qsa_fast._batch_gather_tokens(
        keys.transpose(0, 2, 1, 3), selected
    ).transpose(0, 1, 3, 2, 4)
    selected_values = qsa_fast._batch_gather_tokens(
        values.transpose(0, 2, 1, 3), selected
    ).transpose(0, 1, 3, 2, 4)
    grouped_queries = queries.transpose(0, 2, 1, 3).reshape(
        1, query_tokens, 2, 12, 256
    )
    scores = (
        grouped_queries.astype(mx.float32)
        @ selected_keys.astype(mx.float32).swapaxes(-1, -2)
    ) / (256**0.5)
    scores = mx.where(
        selected_valid[:, :, None, None, :],
        scores,
        mx.finfo(scores.dtype).min,
    )
    probabilities = mx.softmax(scores, axis=-1).astype(queries.dtype)
    return (probabilities @ selected_values).reshape(
        1, query_tokens, 24, 256
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key-tokens", type=int, default=50_000)
    parser.add_argument("--query-tokens", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=927)
    args = parser.parse_args()
    if not fast.is_native_available() or not fast.has_symbol(
        "qwen4_qsa_sparse_gqa_attention"
    ):
        raise SystemExit("rebuild glm_moe_dsa with the Qwen4 sparse GQA ABI")
    if args.key_tokens - args.query_tokens < 2048:
        raise SystemExit("benchmark needs at least 2,048 visible prefix tokens")

    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    q_offset = args.key_tokens - args.query_tokens
    queries = mx.random.normal((1, 24, args.query_tokens, 256)).astype(mx.bfloat16)
    keys = mx.random.normal((1, 2, args.key_tokens, 256)).astype(mx.bfloat16)
    values = mx.random.normal((1, 2, args.key_tokens, 256)).astype(mx.bfloat16)
    blocks = []
    expanded = []
    expanded_valid = []
    for row in range(args.query_tokens):
        complete = (q_offset + row + 1) // 4
        chosen = np.sort(rng.choice(complete, size=512, replace=False)).astype(
            np.uint32
        )
        blocks.append(chosen)
        tokens = (chosen[:, None] * 4 + np.arange(4, dtype=np.uint32)).reshape(-1)
        tail_start = complete * 4
        tail = np.arange(tail_start, q_offset + row + 1, dtype=np.uint32)
        expanded.append(np.pad(np.concatenate((tokens, tail)), (0, 3 - len(tail))))
        expanded_valid.append(
            np.concatenate(
                (
                    np.ones(2048 + len(tail), dtype=np.bool_),
                    np.zeros(3 - len(tail), dtype=np.bool_),
                )
            )
        )
    selected_blocks = mx.array(np.stack(blocks)[None])
    selected_tokens = mx.array(np.stack(expanded)[None])
    valid = mx.array(np.stack(expanded_valid)[None])
    selected_tokens = mx.where(valid, selected_tokens, 0)
    mx.eval(queries, keys, values, selected_blocks, selected_tokens)

    reference = _portable(queries, keys, values, selected_tokens, valid)
    mx.eval(reference)
    for key_tile, dimension_tile in ((128, 32), (64, 64)):
        def call(key_tile=key_tile, dimension_tile=dimension_tile):
            return fast.qwen4_qsa_sparse_gqa_attention(
                queries,
                keys,
                values,
                selected_blocks[:, None],
                256**-0.5,
                q_offset,
                key_tile=key_tile,
                dimension_tile=dimension_tile,
            )

        mx.eval(call())
        samples, native = _time(call, args.repetitions)
        native_rows = native.transpose(0, 2, 1, 3)
        error = mx.abs(native_rows.astype(mx.float32) - reference.astype(mx.float32))
        mx.eval(error)
        print(
            f"native BK={key_tile} DC={dimension_tile}: "
            f"median={statistics.median(samples):.3f} ms "
            f"min={min(samples):.3f} max={max(samples):.3f} "
            f"max_error={float(mx.max(error).item()):.7f}"
        )

    samples, _ = _time(
        lambda: _portable(queries, keys, values, selected_tokens, valid),
        args.repetitions,
    )
    print(
        f"portable gathered: median={statistics.median(samples):.3f} ms "
        f"min={min(samples):.3f} max={max(samples):.3f}"
    )


if __name__ == "__main__":
    main()
