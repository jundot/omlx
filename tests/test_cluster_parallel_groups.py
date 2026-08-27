# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from omlx.cluster.parallel_groups import (
    hybrid_group_split_supported,
    install_pipeline_group_routing,
    split_parallel_groups,
)


class FakeSubgroup:
    def __init__(self, size, rank):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


class FakeWorld(FakeSubgroup):
    def __init__(self, size, rank):
        super().__init__(size, rank)
        self.splits = []

    def split(self, color, key=-1):
        self.splits.append((color, key))
        # TP split is called first for the tested TP2 x PP2 rank map.
        if len(self.splits) == 1:
            return FakeSubgroup(2, key)
        return FakeSubgroup(2, key)


@pytest.mark.parametrize(
    "rank, expected",
    [
        (0, (0, 0)),
        (1, (0, 1)),
        (2, (1, 0)),
        (3, (1, 1)),
    ],
)
def test_hybrid_groups_follow_signed_rank_coordinates(rank, expected):
    world = FakeWorld(4, rank)
    groups = split_parallel_groups(world, 2)

    assert (groups.pipeline_stage, groups.tensor_rank) == expected
    assert groups.hybrid is True
    assert groups.tensor.size() == groups.pipeline.size() == 2
    assert world.splits == [(expected[0], expected[1]), (expected[1], expected[0])]


def test_pure_parallelism_reuses_world_without_splitting():
    tensor_world = FakeWorld(2, 1)
    tensor = split_parallel_groups(tensor_world, 2)
    assert tensor.tensor is tensor_world
    assert tensor.pipeline is None
    assert tensor_world.splits == []

    pipeline_world = FakeWorld(3, 2)
    pipeline = split_parallel_groups(pipeline_world, 1)
    assert pipeline.pipeline is pipeline_world
    assert pipeline.tensor is None
    assert pipeline_world.splits == []


def test_hybrid_capability_fails_closed_on_stock_mlx_api():
    unsupported = SimpleNamespace(distributed=SimpleNamespace())
    supported = SimpleNamespace(
        distributed=SimpleNamespace(
            is_group_split_available=lambda backend: backend == "jaccl"
        )
    )

    assert hybrid_group_split_supported("jaccl", mx_module=unsupported) is False
    assert hybrid_group_split_supported("ring", mx_module=supported) is False
    assert hybrid_group_split_supported("jaccl-ring", mx_module=supported) is True


def test_pipeline_collectives_are_scoped_to_model_forward_only():
    calls = []

    class Distributed:
        def send(self, value, peer, **kwargs):
            calls.append(("send", kwargs.get("group")))
            return value

        def recv(self, *args, **kwargs):
            calls.append(("recv", kwargs.get("group")))
            return args[0]

        def recv_like(self, value, peer, **kwargs):
            calls.append(("recv_like", kwargs.get("group")))
            return value

        def all_gather(self, value, **kwargs):
            calls.append(("all_gather", kwargs.get("group")))
            return value

    distributed = Distributed()
    mx = SimpleNamespace(distributed=distributed)
    pipeline_group = FakeSubgroup(2, 0)
    explicit_group = FakeSubgroup(2, 1)

    class PipelineModel:
        def __call__(self, value):
            distributed.recv_like(value, 1)
            distributed.send(value, 1)
            distributed.all_gather(value)
            distributed.all_gather(value, group=explicit_group)
            return value

    model = SimpleNamespace(model=PipelineModel())
    original_send = distributed.send

    with install_pipeline_group_routing(
        model,
        pipeline_group,
        mx_module=mx,
    ):
        model.model("x")
        distributed.send("outside", 1)

    assert calls == [
        ("recv_like", pipeline_group),
        ("send", pipeline_group),
        ("all_gather", pipeline_group),
        ("all_gather", explicit_group),
        ("send", None),
    ]
    assert distributed.send == original_send
