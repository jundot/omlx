# SPDX-License-Identifier: Apache-2.0
"""Admission rules for heterogeneous CUDA supernodes."""

from __future__ import annotations

import argparse

import pytest

from benchmarks.heterogeneous_pool_probe import (
    _parse_rank_set,
    _supernode_failures,
)


def _topology(*accelerators: str, nccl: bool = True):
    return {
        "world_size": len(accelerators),
        "ranks": [
            {
                "rank": rank,
                "accelerator": accelerator,
                "nccl_available": nccl and accelerator == "cuda",
            }
            for rank, accelerator in enumerate(accelerators)
        ],
    }


def test_adjacent_cuda_pair_is_a_valid_supernode():
    topology = _topology("metal", "metal", "cuda", "cuda")

    assert _supernode_failures(((2, 3),), topology) == []


def test_supernode_refuses_slow_ring_placement_or_non_cuda_member():
    topology = _topology("metal", "cuda", "metal", "cuda")

    failures = _supernode_failures(((0, 2),), topology)

    assert any("non-CUDA rank" in failure for failure in failures)
    assert any("adjacent in the outer Ring" in failure for failure in failures)


def test_supernode_requires_nccl_on_every_cuda_member():
    topology = _topology("metal", "cuda", "cuda", nccl=False)

    failures = _supernode_failures(((1, 2),), topology)

    assert failures == ["CUDA supernode 1 lacks NCCL on rank(s): 1, 2"]


def test_supernode_rank_parser_rejects_duplicates():
    with pytest.raises(argparse.ArgumentTypeError, match="unique"):
        _parse_rank_set("2,2")
