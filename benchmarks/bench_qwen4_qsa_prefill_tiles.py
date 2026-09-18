# SPDX-License-Identifier: Apache-2.0
"""Compare complete QSA prefill at 256-query and automatic native tile widths.

Run from the repository root with PYTHONPATH=. and the native glm_moe_dsa
extension built against the installed MLX version; no model weights are needed.
This measures one attention call, not whole-model prompt processing or decode.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
import time

import mlx.core as mx

from omlx.custom_kernels.glm_moe_dsa import fast
from omlx.patches import mlx_vlm_qwen4_exp_compat as compat

compat.apply_mlx_vlm_qwen4_exp_compat_patch()
from mlx_vlm.models.qwen4_exp import qsa_fast  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-tokens", type=int, default=65536)
    parser.add_argument("--query-tokens", type=int, default=2048)
    parser.add_argument("--repetitions", type=int, default=12)
    args = parser.parse_args()
    if args.query_tokens < 1024 or args.key_tokens <= args.query_tokens + 2048:
        parser.error(
            "use at least 1024 queries and more than 2048 cached prefix tokens"
        )
    if args.repetitions < 2:
        parser.error("at least two repetitions are required")
    if not fast.is_native_available() or not all(
        fast.has_symbol(name)
        for name in (
            "qwen4_qsa_indexer_scores",
            "qwen4_qsa_topk_indices",
            "qwen4_qsa_sparse_gqa_attention",
        )
    ):
        parser.error("build the native glm_moe_dsa extension first")
    mx.random.seed(81)
    rows, tokens = args.query_tokens, args.key_tokens
    inputs = (
        (mx.random.normal((1, 24, rows, 256)) * 0.1).astype(mx.bfloat16),
        (mx.random.normal((1, 2, tokens, 256)) * 0.1).astype(mx.bfloat16),
        (mx.random.normal((1, 2, tokens, 256)) * 0.1).astype(mx.bfloat16),
        (mx.random.normal((1, rows, 4, 128)) * 0.1).astype(mx.bfloat16),
        (mx.random.normal((1, tokens, 128)) * 0.1).astype(mx.bfloat16),
        mx.arange(tokens)[None],
    )
    kwargs = dict(
        num_query_heads=24,
        num_key_value_heads=2,
        head_dim=256,
        indexer_head_dim=128,
        compress_ratio=4,
        token_budget=2048,
        index_key_norm=lambda x: x,
        apply_index_rope=lambda x, p: x,
    )
    kwargs["pooled_index_keys"] = qsa_fast.pool_completed_index_keys(
        inputs[4],
        inputs[5],
        compress_ratio=4,
        index_key_norm=kwargs["index_key_norm"],
        apply_index_rope=kwargs["apply_index_rope"],
    )
    mx.eval(inputs, kwargs["pooled_index_keys"])

    def call(mode):
        return qsa_fast.contiguous_causal_gathered_qsa(
            *inputs, query_chunk=256 if mode == "control" else None, **kwargs
        )

    expected = call("control")
    mx.eval(expected)
    assert all(
        getattr(qsa_fast, f"_NATIVE_QSA_{stage}_PROVEN")
        for stage in ("MAIN", "SCORE", "TOPK")
    )
    widths = []
    original = qsa_fast._native_sparse_gqa_attention

    def record(q, *a, **kw):
        widths.append(q.shape[2])
        return original(q, *a, **kw)

    try:
        qsa_fast._native_sparse_gqa_attention = record
        actual = call("candidate")
        mx.eval(actual)
    finally:
        qsa_fast._native_sparse_gqa_attention = original
    assert max(widths) > 256, "automatic wider tiles did not engage"
    assert mx.array_equal(expected.view(mx.uint8), actual.view(mx.uint8)).item()
    for _ in range(3):
        mx.eval(call("control"), call("candidate"))
    samples = {"control": [], "candidate": []}
    for index in range(args.repetitions):
        order = ("control", "candidate") if index % 2 == 0 else ("candidate", "control")
        for mode in order:
            mx.synchronize()
            start = time.perf_counter()
            mx.eval(call(mode))
            samples[mode].append((time.perf_counter() - start) * 1000)
    medians = {mode: statistics.median(times) for mode, times in samples.items()}
    print(
        json.dumps(
            {
                "device": mx.device_info()["device_name"],
                "mlx": importlib.metadata.version("mlx"),
                "key_tokens": tokens,
                "query_tokens": rows,
                "repetitions_per_arm": args.repetitions,
                "candidate_tiles": widths,
                "byte_identical": True,
                "median_ms": medians,
                "samples_ms": samples,
                "latency_reduction_percent": 100
                * (1 - medians["candidate"] / medians["control"]),
                "scope": "Synthetic QSA attention component with precomputed pooled keys; not whole-model PP or decode",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
