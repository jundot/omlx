"""Exactness and scheduling gates for DS4 HyperConnection overlap."""

from __future__ import annotations

import inspect
import sys
from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

apply_deepseek_v4_patch()
dm = sys.modules["mlx_lm.models.deepseek_v4"]
hc = sys.modules["mlx_lm.models.hyper_connection"]


class _Tensor:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self.ndim = len(shape)


class _Group:
    def __init__(self, size=2):
        self._size = size

    def size(self):
        return self._size


def _block(*, training=False, size=2):
    group = _Group(size)
    return SimpleNamespace(
        training=training,
        attn=SimpleNamespace(sharding_group=group),
        ffn=SimpleNamespace(sharding_group=group),
    )


def test_overlap_gate_is_exact_m1024_tp2_only(monkeypatch):
    monkeypatch.setattr(dm, "_DEEPSEEK_V4_HC_RESIDUAL_OVERLAP", True)
    monkeypatch.setattr(dm, "is_dspark_verify_armed", lambda: False)
    h = _Tensor((1, 1024, 2, 4096), mx.bfloat16)
    assert dm._can_overlap_hc_residual(_block(), h)
    assert not dm._can_overlap_hc_residual(_block(size=1), h)
    assert not dm._can_overlap_hc_residual(_block(training=True), h)
    assert not dm._can_overlap_hc_residual(
        _block(), _Tensor((1, 512, 2, 4096), mx.bfloat16)
    )
    monkeypatch.setattr(dm, "is_dspark_verify_armed", lambda: True)
    assert not dm._can_overlap_hc_residual(_block(), h)


def test_split_residual_branch_is_bit_exact_to_stock_hc_expand():
    mx.random.seed(20260822)
    x = mx.random.normal((1, 4, 8)).astype(mx.bfloat16)
    residual = mx.random.normal((1, 4, 2, 8)).astype(mx.bfloat16)
    post = mx.random.normal((1, 4, 2)).astype(mx.float32)
    comb = mx.random.normal((1, 4, 2, 2)).astype(mx.float32)
    stock = hc.hc_expand(x, residual, post, comb)
    branch = hc.hc_residual_branch(residual, comb)
    split = hc.hc_merge_branch(x, post, branch)
    mx.eval(stock, branch, split)
    assert branch.dtype == mx.float32
    assert mx.array_equal(stock, split).item()


def test_residual_graph_is_created_before_collective_modules():
    source = inspect.getsource(dm.DeepseekV4Block.__call__)
    attn_branch = source.index("residual_branch = hc_residual_branch")
    attention = source.index("x = self.attn(", attn_branch)
    attn_merge = source.index("hc_merge_branch", attention)
    ffn_branch = source.index("residual_branch = hc_residual_branch", attn_merge)
    ffn = source.index("x = self.ffn(", ffn_branch)
    ffn_merge = source.index("hc_merge_branch", ffn)
    assert attn_branch < attention < attn_merge < ffn_branch < ffn < ffn_merge
