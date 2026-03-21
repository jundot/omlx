# SPDX-License-Identifier: Apache-2.0
"""Benchmark: sequential vs batched grammar bitmask filling.

Compares three approaches for filling bitmasks across a batch of
grammar-constrained requests:

1. **Per-request**: fill + apply per-request (Stage 1 baseline)
2. **Sequential-shared**: sequential fill into shared buffer, one batched apply
3. **BatchGrammarMatcher**: C++ thread pool fill, one batched apply

Context: bitmask fill is microseconds; model forward is 10-50ms.
The ``mx.async_eval`` overlap with model forward is the primary
latency win, not the fill strategy itself.

Run with:
    python -m pytest tests/bench_grammar_bitmask.py -v -s
"""

import time

import mlx.core as mx
import numpy as np
import pytest

pytestmark = pytest.mark.slow


def _make_matchers(compiler, n, schema):
    import xgrammar as xgr
    compiled = compiler.compile_json_schema(schema)
    return [xgr.GrammarMatcher(compiled) for _ in range(n)]


@pytest.fixture(scope="module")
def compiler():
    xgr = pytest.importorskip("xgrammar")
    vocab_size = 32000
    vocab = [f"<tok_{i}>" for i in range(vocab_size)]
    vocab[0] = "<unk>"
    vocab[1] = "<s>"
    vocab[2] = "</s>"
    for c in '{}[]":, \nabcdefghijklmnopqrstuvwxyz0123456789':
        idx = ord(c) if ord(c) < vocab_size else hash(c) % (vocab_size - 10) + 10
        vocab[idx] = c
    ti = xgr.TokenizerInfo(vocab)
    return xgr.GrammarCompiler(ti), vocab_size


SCHEMA = '{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]}'

BATCH_SIZES = [1, 2, 4, 8, 16, 32]
WARMUP = 5
ITERS = 200


class TestBitmaskFillBenchmark:
    """Benchmark bitmask fill strategies (CPU-only, no model forward)."""

    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_fill_strategies(self, compiler, batch_size):
        xgr = pytest.importorskip("xgrammar")
        from xgrammar.kernels.apply_token_bitmask_mlx import apply_token_bitmask_mlx

        comp, vocab_size = compiler
        bitmask_width = (vocab_size + 31) // 32
        matchers = _make_matchers(comp, batch_size, SCHEMA)
        bitmask = np.full((batch_size, bitmask_width), -1, dtype=np.int32)
        logits = mx.zeros((batch_size, vocab_size))

        # --- 1. Per-request fill + per-request apply (Stage 1 baseline) ---
        per_req_bm = np.full((1, bitmask_width), -1, dtype=np.int32)
        for _ in range(WARMUP):
            for m in matchers:
                per_req_bm[:] = -1
                m.fill_next_token_bitmask(per_req_bm)
                apply_token_bitmask_mlx(mx.array(per_req_bm), logits[0:1], vocab_size)

        t0 = time.perf_counter()
        for _ in range(ITERS):
            for i, m in enumerate(matchers):
                per_req_bm[:] = -1
                m.fill_next_token_bitmask(per_req_bm)
                apply_token_bitmask_mlx(mx.array(per_req_bm), logits[i : i + 1], vocab_size)
        per_req_us = (time.perf_counter() - t0) / ITERS * 1e6

        # --- 2. Sequential fill into shared buffer + batched apply ---
        for _ in range(WARMUP):
            bitmask[:] = -1
            for i, m in enumerate(matchers):
                m.fill_next_token_bitmask(bitmask, i)
            apply_token_bitmask_mlx(mx.array(bitmask), logits, vocab_size)

        t0 = time.perf_counter()
        for _ in range(ITERS):
            bitmask[:] = -1
            for i, m in enumerate(matchers):
                m.fill_next_token_bitmask(bitmask, i)
            apply_token_bitmask_mlx(mx.array(bitmask), logits, vocab_size)
        seq_shared_us = (time.perf_counter() - t0) / ITERS * 1e6

        # --- 3. BatchGrammarMatcher + batched apply ---
        batch_matcher = xgr.BatchGrammarMatcher()
        for _ in range(WARMUP):
            bitmask[:] = -1
            batch_matcher.batch_fill_next_token_bitmask(matchers, bitmask)
            apply_token_bitmask_mlx(mx.array(bitmask), logits, vocab_size)

        t0 = time.perf_counter()
        for _ in range(ITERS):
            bitmask[:] = -1
            batch_matcher.batch_fill_next_token_bitmask(matchers, bitmask)
            apply_token_bitmask_mlx(mx.array(bitmask), logits, vocab_size)
        batch_us = (time.perf_counter() - t0) / ITERS * 1e6

        print(
            f"\n  batch_size={batch_size:>2}  "
            f"per-request={per_req_us:>8.1f} µs  "
            f"seq-shared={seq_shared_us:>8.1f} µs  "
            f"batch-matcher={batch_us:>8.1f} µs  "
            f"(seq-shared vs per-req: {per_req_us / seq_shared_us:.2f}x)"
        )
