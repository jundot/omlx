# SPDX-License-Identifier: Apache-2.0
"""Candidate-block prefilter exactness tests (Hierarchical Sparse Indexer L1)."""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest


def _select_candidate_blocks_np(
    logits: np.ndarray,
    compress_lens,
    topk_blocks: int,
    block_size: int,
) -> np.ndarray:
    """NumPy reference matching inference/model.py:select_candidate_blocks."""
    width = logits.shape[-1]
    pad = (-width) % block_size
    if pad:
        logits_p = np.pad(logits, [(0, 0)] * (logits.ndim - 1) + [(0, pad)], constant_values=-np.inf)
    else:
        logits_p = logits
    scores = logits_p.reshape(*logits.shape[:-1], -1, block_size).max(axis=-1)
    num_blocks = scores.shape[-1]
    last = (np.asarray(compress_lens) - 1) // block_size
    # pin newest block
    ar = np.arange(num_blocks)
    if np.ndim(last) == 0:
        scores = scores.copy()
        scores[..., ar == int(last)] = np.inf
    else:
        scores = scores.copy()
        # broadcast last against scores leading dims — keep simple for unit tests
        for idx in np.ndindex(scores.shape[:-1]):
            scores[idx][ar == int(np.asarray(last)[idx])] = np.inf
    k = min(topk_blocks, num_blocks)
    # topk
    part = np.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
    top_vals = np.take_along_axis(scores, part, axis=-1)
    keep = np.zeros_like(scores, dtype=bool)
    # scatter
    for idx in np.ndindex(keep.shape[:-1]):
        for j, (bi, val) in enumerate(zip(part[idx], top_vals[idx])):
            if val > -np.inf:
                keep[idx][bi] = True
    mask = np.repeat(keep, block_size, axis=-1)[..., :width]
    return mask


@pytest.fixture
def select_fn():
    from omlx.patches.deepseek_v41.select_candidates import select_candidate_blocks

    return select_candidate_blocks


def test_newest_block_always_kept(select_fn):
    # 3 blocks of size 8; only first 20 positions reachable
    width = 20
    logits = np.full((1, 1, width), -1.0, dtype=np.float32)
    # make block 0 huge, block 1 tiny, block 2 (newest, partial) tiny
    logits[..., 0:8] = 10.0
    logits[..., 8:16] = -5.0
    logits[..., 16:20] = -5.0
    compress_lens = 20
    mask = select_fn(mx.array(logits), compress_lens, topk_blocks=2, block_size=8)
    m = np.array(mask)
    # newest block positions 16..19 must be True despite low score
    assert m[..., 16:20].all()
    # block 0 kept as high score
    assert m[..., 0:8].all()


def test_mask_width_matches_logits(select_fn):
    logits = mx.full((2, 3, 17), -1.0)
    mask = select_fn(logits, compress_lens=17, topk_blocks=4, block_size=8)
    assert mask.shape == logits.shape
