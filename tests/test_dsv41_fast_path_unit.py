# SPDX-License-Identifier: Apache-2.0
"""Unit tests for DSV4.1 fast-path helpers (no full weight load)."""

from __future__ import annotations

import mlx.core as mx

from omlx.patches.deepseek_v41.predicates import is_deepseek_v4, is_deepseek_v41
from omlx.patches.deepseek_v41.select_candidates import select_candidate_blocks
from omlx.patches.deepseek_v41.shared_runtime import SharedAttentionRuntime


def test_predicates_v4_vs_v41():
    assert is_deepseek_v4("deepseek_v4")
    assert is_deepseek_v4("deepseek_v4_flash")
    assert not is_deepseek_v4("deepseek_v41")
    assert is_deepseek_v41("deepseek_v41")
    assert is_deepseek_v41("deepseek_v41_text")
    assert not is_deepseek_v41("deepseek_v4")


def test_shared_runtime_pool_register_identity_within_pass():
    rt = SharedAttentionRuntime()
    a = object()
    b = object()
    assert rt.register_kv_pool(2, a) is a
    assert rt.register_kv_pool(2, b) is a
    rt.kv_pool_by_source.clear()
    assert rt.register_kv_pool(2, b) is b


def test_select_candidate_blocks_pins_last_block():
    logits = mx.array([[1.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 2.0]], dtype=mx.float32)
    mask = select_candidate_blocks(logits, compress_lens=8, topk_blocks=1, block_size=4)
    mx.eval(mask)
    assert mask.shape == logits.shape
    assert bool(mask[0, 4:8].any())


def test_fast_path_is_default():
    from omlx.patches.deepseek_v41 import fast_path as fp

    assert fp.fast_path_enabled() is True
