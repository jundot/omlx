"""Exact layout gate for batched DS4 O-A target-verification preparation."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

apply_deepseek_v4_patch()
dm = sys.modules["mlx_lm.models.deepseek_v4"]


@pytest.mark.parametrize("heads", (24, 32, 40, 64))
@pytest.mark.parametrize("rows", (2, 3, 4, 5, 6))
def test_batched_prepare_equals_decode_row_concatenation(heads, rows):
    attn = SimpleNamespace(o_groups=8, head_dim=512)
    mx.random.seed(20260824 + heads + rows)
    output = mx.random.normal((1, heads, rows, 512)).astype(mx.bfloat16)

    batched = dm._prepare_attention_output(attn, output)
    rowwise = mx.concatenate(
        [
            dm._prepare_attention_output(attn, output[:, :, index : index + 1])
            for index in range(rows)
        ],
        axis=2,
    )
    mx.eval(batched, rowwise)

    assert batched.shape == (1, 8, rows, heads // 8 * 512)
    assert mx.array_equal(batched, rowwise).item()
