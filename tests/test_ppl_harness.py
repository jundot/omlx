# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the perplexity harness math (bench/ppl_expert_streaming.py)."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

from ppl_expert_streaming import iter_windows, window_nll  # noqa: E402


def test_iter_windows_disjoint_and_capped():
    tokens = list(range(20))
    windows = list(iter_windows(tokens, ctx=5, max_windows=None))
    # step = ctx-1 = 4; starts 0, 4, 8, 12 (16+5 > 20 stops)
    assert [start for start, _ in windows] == [0, 4, 8, 12]
    assert windows[0][1] == [0, 1, 2, 3, 4]
    assert windows[1][1] == [4, 5, 6, 7, 8]

    assert list(iter_windows(tokens, ctx=5, max_windows=2)) == windows[:2]
    # Corpus shorter than one window yields nothing.
    assert list(iter_windows(list(range(4)), ctx=5, max_windows=None)) == []


def test_window_nll_uniform_logits_is_log_vocab():
    # Uniform logits over a vocab of 10: every target costs log(10) nats.
    ctx = 5
    logits = np.zeros((ctx, 10), dtype=np.float32)
    targets = np.arange(ctx)
    total, n = window_nll(logits, targets)
    assert n == ctx - 1  # first position is context-only
    assert abs(total / n - math.log(10.0)) < 1e-5


def test_window_nll_picks_correct_column():
    # One-hot-ish logits: the correct target gets ~0 cost, others get ~V.
    ctx = 4
    logits = np.full((ctx, 6), -20.0, dtype=np.float32)
    targets = np.array([1, 0, 3, 5])
    # logits[:-1] (rows 0..2) predict targets[1:] = [0, 3, 5].
    for i, t in enumerate(targets[1:]):
        logits[i, t] = 20.0
    total, n = window_nll(logits, targets)
    assert n == 3
    assert total / n < 1e-3
