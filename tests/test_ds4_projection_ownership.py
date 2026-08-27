# SPDX-License-Identifier: Apache-2.0
"""Contracts for exact owner-computed DS4 projection banks."""

from types import SimpleNamespace

import mlx.core as mx

from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

apply_deepseek_v4_patch()

import mlx_lm.models.deepseek_v4 as dsv4  # noqa: E402


class _Group:
    def __init__(self, rank):
        self._rank = rank

    def rank(self):
        return self._rank

    def size(self):
        return 2


class _Projection:
    def __init__(self, value):
        self.value = value
        self.weight = mx.zeros((value.shape[-1], 1))

    def __call__(self, _x):
        return self.value


def _enable(monkeypatch):
    monkeypatch.setenv("OMLX_TP_SHARD_WEIGHTS", "3,5")
    monkeypatch.setenv("OMLX_DSV4_PROJECTION_OWNER_RANK", "0")
    monkeypatch.setattr(dsv4, "_PROJECTION_OWNER_LOGGED", False)


def test_projection_owner_keeps_exact_views_from_symmetric_sum(monkeypatch):
    _enable(monkeypatch)
    reduced = []

    def all_sum(value, *, group):
        reduced.append((value, group.rank()))
        return value

    monkeypatch.setattr(mx.distributed, "all_sum", all_sum)
    x = mx.zeros((1, 1, 4), dtype=mx.bfloat16)
    first = mx.array([[[1, 2]]], dtype=mx.bfloat16)
    second = mx.array([[[3, 4, 5]]], dtype=mx.bfloat16)

    outputs = dsv4._owned_projection_bank(
        x,
        (_Projection(first), _Projection(second)),
        _Group(0),
    )
    mx.eval(*outputs)

    assert reduced[0][1] == 0
    assert tuple(reduced[0][0].shape) == (1, 1, 5)
    assert bool(mx.array_equal(outputs[0], first).item())
    assert bool(mx.array_equal(outputs[1], second).item())


def test_projection_peer_splits_received_storage_without_computation(monkeypatch):
    _enable(monkeypatch)
    packed = mx.array([[[1, 2, 3, 4, 5]]], dtype=mx.bfloat16)
    reduced = []

    def all_sum(value, *, group):
        reduced.append((tuple(value.shape), group.rank()))
        return packed

    monkeypatch.setattr(mx.distributed, "all_sum", all_sum)
    def fail(_x):
        raise AssertionError("peer computed projection")

    modules = (
        SimpleNamespace(weight=mx.zeros((2, 1)), __call__=fail),
        SimpleNamespace(weight=mx.zeros((3, 1)), __call__=fail),
    )
    x = mx.zeros((1, 1, 4), dtype=mx.bfloat16)

    outputs = dsv4._owned_projection_bank(x, modules, _Group(1))
    mx.eval(*outputs)

    assert reduced == [((1, 1, 5), 1)]
    assert outputs[0].tolist() == [[[1.0, 2.0]]]
    assert outputs[1].tolist() == [[[3.0, 4.0, 5.0]]]


def test_projection_owner_is_inert_without_signed_3x5_plan(monkeypatch):
    monkeypatch.setenv("OMLX_TP_SHARD_WEIGHTS", "4,4")
    monkeypatch.setenv("OMLX_DSV4_PROJECTION_OWNER_RANK", "0")
    x = mx.zeros((1, 1, 4), dtype=mx.bfloat16)
    module = _Projection(mx.zeros((1, 1, 2), dtype=mx.bfloat16))

    assert dsv4._owned_projection_bank(x, (module,), _Group(0)) is None
