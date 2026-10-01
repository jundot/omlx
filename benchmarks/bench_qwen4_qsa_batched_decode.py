#!/usr/bin/env python3
"""Benchmark batched Qwen4 QSA decode and verify steps, attention layers only.

Builds the served model's QSA attention layers with random weights, gives each a
BatchQSAKVCache merged from rows of fabricated history (lengths base + i * stagger)
and times one batched step through all of them, per arm:

  dense   fresh per-row indexer caches, dense masked SDPA (the path on main)
  banks   persistent per-row pooled banks, dense masked SDPA
  gather  persistent banks, gathered attention over each query's selection

  python benchmarks/bench_qwen4_qsa_batched_decode.py --bits 4 --batch 1,2,4,8
  python benchmarks/bench_qwen4_qsa_batched_decode.py --bits 4 --batch 2,8 \\
      --verify 4 --rollback --arms dense,gather
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import random
import statistics
import time

import mlx.core as mx
import mlx.nn as nn

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat

compat.apply_mlx_vlm_qwen4_exp_compat_patch()
import mlx_vlm.models.qwen4_exp.language as language  # noqa: E402
from mlx_vlm.models.qwen4_exp import TextConfig  # noqa: E402

ARMS = {"dense": (False, False), "banks": (True, False), "gather": (True, True)}
WARMUP = 4


def _config():
    # Qwen3.8-Flash-Next QSA shapes; the MoE and linear-attention parts are unused.
    return TextConfig(
        model_type="qwen4_exp_text",
        hidden_size=2560,
        num_hidden_layers=2,
        num_attention_heads=24,
        linear_num_value_heads=48,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=64,
        moe_intermediate_size=64,
        rms_norm_eps=1e-6,
        vocab_size=256,
        num_key_value_heads=2,
        max_position_embeddings=262144,
        hc_count=2,
        hc_lowrank=8,
        head_dim=256,
        layer_types=["linear_attention", "qwen_sparse_attention"],
        ple_layer_ids=[],
        ple_embed_dim=32,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=4,
        split_ngram_parts=4,
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=2048,
        indexer_compress_ratio=4,
        eos_token_id=1,
        rope_parameters={
            "rope_type": "default",
            "mrope_section": [11, 11, 10],
            "mrope_interleaved": True,
            "rope_theta": 10_000_000,
            "partial_rotary_factor": 0.25,
        },
    )


def _fabricate(tokens: int):
    cache = language.QSAKVCache()
    keys = (mx.random.normal((1, 2, tokens, 256)) * 0.5).astype(mx.bfloat16)
    values = mx.random.normal((1, 2, tokens, 256)).astype(mx.bfloat16)
    cache.update_and_fetch(keys, values)
    index_keys = mx.random.normal((1, tokens, 128)).astype(mx.bfloat16)
    positions = mx.broadcast_to(mx.arange(tokens)[None, None], (3, 1, tokens))
    cache.update_indexer(index_keys, positions)
    mx.eval(cache.keys, cache.values, cache.index_keys, cache.index_position_ids)
    return cache


def _rollback(caches, query_rows: int, rng: random.Random):
    # Ragged Lightning MTP accept: each row keeps 1..query_rows of the window.
    retained = [rng.randint(1, query_rows) for _ in range(caches[0].offset.shape[0])]
    keep = max(retained)
    for cache in caches:
        if query_rows > keep:
            cache.trim(query_rows - keep)
        padding = [keep - r for r in retained]
        if any(padding):
            cache.prepare(right_padding=padding)
            cache.finalize()
    mx.eval(
        [c.keys for c in caches],
        [c.values for c in caches],
        [c.index_keys for c in caches],
        [c.offset for c in caches],
    )


def _run(args, layers, arm: str, batch: int, gathered_calls: list[int]):
    banks, gather = ARMS[arm]
    language._BATCH_ROW_BANKS_ENABLED = banks
    language._GATHERED_BATCH_DISABLED = not gather
    lengths = [args.base + i * args.stagger for i in range(batch)]
    caches = []
    for _ in layers:
        rows = [_fabricate(n) for n in lengths]
        caches.append(language.BatchQSAKVCache.merge(rows) if batch > 1 else rows[0])
    mx.eval([c.keys for c in caches])

    query_rows = max(1, args.verify)
    rng = random.Random(1)
    build, step, rollback = [], [], []
    gathered_calls[0] = 0
    for i in range(args.steps):
        x = (mx.random.normal((batch, query_rows, args.hidden)) * 0.1).astype(
            mx.bfloat16
        )
        mx.eval(x)
        start = time.perf_counter()
        first = caches[0]
        mask = language._create_qwen3_5_attention_mask(x, first) if batch > 1 else None
        offset = first.offset if batch > 1 else mx.array([first.offset])
        positions = mx.broadcast_to(
            offset[None, :, None] + mx.arange(query_rows)[None, None],
            (3, batch, query_rows),
        )
        hidden = x
        for layer, cache in zip(layers, caches):
            hidden = layer(
                hidden,
                mask=mask,
                cache=cache,
                position_ids=positions,
                target_verify=bool(args.verify),
            )
        built = time.perf_counter()
        mx.eval(hidden, *[c.keys for c in caches])
        done = time.perf_counter()
        rolled = done
        if args.verify:
            if args.rollback and batch > 1:
                _rollback(caches, query_rows, rng)
                rolled = time.perf_counter()
            else:
                for cache in caches:
                    cache.trim(query_rows - 1)
        if i >= WARMUP:
            build.append(built - start)
            step.append(done - start)
            rollback.append(rolled - done)
    return (
        statistics.median(build) * 1e3,
        statistics.median(step) * 1e3,
        statistics.median(rollback) * 1e3,
        gathered_calls[0],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default="dense,banks,gather")
    parser.add_argument("--batch", default="1,2,4,8")
    parser.add_argument("--base", type=int, default=100_000)
    parser.add_argument("--stagger", type=int, default=2_000)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument(
        "--verify", type=int, default=0, help="query rows per step (0 = decode)"
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="verify: ragged accept via trim + prepare + finalize",
    )
    parser.add_argument(
        "--bits", type=int, default=0, help="quantize the projections (served: 4)"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.steps <= WARMUP:
        raise SystemExit(f"--steps must exceed the {WARMUP} warm-up steps")
    arms = args.arms.split(",")
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise SystemExit(f"unknown arms: {', '.join(sorted(unknown))}")

    config = _config()
    args.hidden = config.hidden_size
    mx.random.seed(args.seed)
    layers = []
    for _ in range(args.layers):
        layer = language.Qwen4ExpAttention(config)
        layer.set_dtype(mx.bfloat16)
        if args.bits:
            nn.quantize(layer, group_size=64, bits=args.bits)
        mx.eval(layer.parameters())
        layers.append(layer)

    # Count gathered-arm calls so a run shows whether the changed path ran.
    gathered_calls = [0]
    gathered = language.Qwen4ExpAttention._gathered_batch

    def counted(self, *call_args, **kwargs):
        gathered_calls[0] += 1
        return gathered(self, *call_args, **kwargs)

    language.Qwen4ExpAttention._gathered_batch = counted

    for batch in (int(b) for b in args.batch.split(",")):
        # One row never batches, so every arm takes the same path.
        for arm in arms[:1] if batch == 1 else arms:
            build, step, rollback, calls = _run(
                args, layers, arm, batch, gathered_calls
            )
            line = (
                f"B={batch} {arm:6s} step={step:7.2f} ms "
                f"per-row={step / batch:6.2f} ms build={build:6.2f} ms"
            )
            if args.rollback:
                line += f" rollback={rollback:7.2f} ms"
            print(f"{line} gathered-calls={calls}", flush=True)
            mx.clear_cache()


if __name__ == "__main__":
    main()
