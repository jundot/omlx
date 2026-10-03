# SPDX-License-Identifier: Apache-2.0
"""Lightning MTP across the ranks of a pipeline-parallel Qwen3.5-family deployment.

1. MTP was refused for every deployment although every rank can run the same
   draft/verify cycle: it is now a deployment property (``execution.mtp``);
2. the adaptive MTP depth controller reads each rank's own wall clock, so ranks
   pick different verify windows and the group deadlocks.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from omlx.cluster.deployment import ClusterDeployment, ClusterHost
from omlx.cluster.launch import build_mlx_launch_argv
from omlx.cluster.performance import ExecutionSettings, execution_profile
from omlx.cluster.planner import PipelineAssignment

# -- 3. MTP as a deployment property -------------------------------------------------


def _deployment(execution: ExecutionSettings) -> ClusterDeployment:
    return ClusterDeployment(
        deployment_id="mtp-test",
        model="org/model",
        backend="ring",
        hosts=(
            ClusterHost("local", "127.0.0.1", ("10.0.0.1",)),
            ClusterHost("peer", "peer.local", ("10.0.0.2",)),
        ),
        assignments=(
            PipelineAssignment("local", 0, 2, 4, 2, 0, 0, 4),
            PipelineAssignment("peer", 1, 0, 2, 2, 0, 0, 4),
        ),
        plan_hash="e" * 64,
        execution=execution,
    )


def test_mtp_excludes_the_rank_zero_sampler():
    base = execution_profile("interactive")
    with pytest.raises(ValueError, match="lockstep sampler"):
        replace(base, mtp=True)  # sampling_rank_only defaults to True
    ok = replace(base, mtp=True, sampling_rank_only=False, async_overlap=False)
    assert ok.mtp is True


def test_mtp_round_trips_through_the_stored_deployment():
    ex = replace(
        execution_profile("interactive"),
        mtp=True,
        sampling_rank_only=False,
        async_overlap=False,
    )
    assert ExecutionSettings.from_dict(ex.to_dict()).mtp is True
    assert (
        ExecutionSettings.from_dict(execution_profile("balanced").to_dict()).mtp
        is False
    )


def test_launcher_passes_mtp_to_every_rank(tmp_path):
    from omlx.cluster.inference_worker import build_parser

    def argv_for(execution):
        argv = build_mlx_launch_argv(
            _deployment(execution),
            hostfile=(tmp_path / "hosts.json").resolve(),
            api_port=32100,
            collective_port=32120,
            python_executable="/opt/omlx/bin/python",
        )
        return argv[argv.index("--") + 1 :]

    plain = argv_for(replace(execution_profile("interactive")))
    assert "--mtp" not in plain
    with_mtp = argv_for(
        replace(
            execution_profile("interactive"),
            mtp=True,
            sampling_rank_only=False,
            async_overlap=False,
        )
    )
    assert "--mtp" in with_mtp
    assert build_parser().parse_args(with_mtp[3:]).mtp is True


def test_distributed_engine_accepts_mtp_only_when_the_deployment_carries_it():
    from omlx.engine.distributed import DistributedBatchedEngine

    settings = SimpleNamespace(mtp_enabled=True)
    plain = DistributedBatchedEngine(
        _deployment(execution_profile("interactive")), model_settings=settings
    )
    with pytest.raises(ValueError, match="mtp_enabled"):
        plain._validate_model_settings()

    carried = replace(
        execution_profile("interactive"),
        mtp=True,
        sampling_rank_only=False,
        async_overlap=False,
    )
    DistributedBatchedEngine(
        _deployment(carried), model_settings=settings
    )._validate_model_settings()
    # everything else stays refused
    with pytest.raises(ValueError, match="turboquant_kv_enabled"):
        DistributedBatchedEngine(
            _deployment(carried),
            model_settings=SimpleNamespace(
                mtp_enabled=True, turboquant_kv_enabled=True
            ),
        )._validate_model_settings()


# -- 4. adaptive depth must be decided once, by rank 0 -------------------------------


def test_depth_controller_adopts_rank_zero_decision(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from omlx.patches.mlx_lm_mtp import batch_generator, distributed_sync

    controller = batch_generator._DepthController
    # restores the unpatched ``observe`` when the test ends
    monkeypatch.setattr(controller, "observe", controller.observe)
    assert distributed_sync.install() is True
    assert distributed_sync.install() is True  # idempotent: no double wrapping

    class Group:
        def __init__(self, rank, size):
            self._rank, self._size = rank, size

        def rank(self):
            return self._rank

        def size(self):
            return self._size

    sent = []

    def fake_all_sum(local, group=None):
        sent.append(local.tolist())
        return mx.array([2, 5], dtype=mx.int32)  # what rank 0 decided

    monkeypatch.setattr(mx.distributed, "all_sum", fake_all_sum)

    # a non-zero rank contributes zeros and adopts rank 0's depth and exit streak
    monkeypatch.setattr(mx.distributed, "init", lambda *a, **k: Group(1, 2))
    follower = controller(4)
    follower.observe(used=4, accepted=1, cycle_ms=95.0)
    assert (follower.cur, follower.exit_streak) == (2, 5)
    assert sent[-1] == [0, 0]

    # rank 0 contributes its own decision
    monkeypatch.setattr(mx.distributed, "init", lambda *a, **k: Group(0, 2))
    leader = controller(4)
    leader.observe(used=4, accepted=4, cycle_ms=20.0)
    assert sent[-1] == [leader_local := sent[-1][0], sent[-1][1]]
    assert leader_local >= 1

    # a single process never issues a collective
    sent.clear()
    monkeypatch.setattr(mx.distributed, "init", lambda *a, **k: Group(0, 1))
    solo = controller(4)
    solo.observe(used=4, accepted=4, cycle_ms=20.0)
    assert sent == []
