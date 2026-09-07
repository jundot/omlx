# SPDX-License-Identifier: Apache-2.0
"""MTP + pure tensor-parallel clustering (test/cluster-run-from-source).

Covers the pieces that let TP=N clustering run native MTP on top of the
existing "identical logits/decisions on every rank" invariant: the
distributed-engine validator carving mtp_enabled out of the incompatibility
list for pure-TP plans only, ClusterDeployment/launch.py plumbing that
threads mtp_enabled/mtp_num_draft_tokens onto the worker's launch argv, and
the rank-0-owned depth/park broadcast + desync checksum in
omlx.patches.mlx_lm_mtp.batch_generator (the piece that keeps independent
per-rank _DepthController instances from picking different depths on the
same cycle -- a shape mismatch in the next TP collective that hangs rather
than crashes).

Worker-side ordering/wiring (maybe_apply_pre_load_patches before
provider.load_default(), configure_distributed_mtp's coordinator kwarg) is
covered in tests/test_cluster_inference_worker.py, reusing that file's
_run_rank harness. Weight-byte accounting for the replicated MTP head is
covered in tests/test_cluster_planner.py.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from omlx.cluster.deployment import ClusterDeployment, ClusterHost
from omlx.cluster.launch import build_mlx_launch_argv
from omlx.cluster.planner import PipelineAssignment
from omlx.engine.distributed import DistributedBatchedEngine
from omlx.patches.mlx_lm_mtp import batch_generator as bg


def _deployment(*, tensor_parallel_size: int = 1) -> ClusterDeployment:
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
        plan_hash="d" * 64,
        tensor_parallel_size=tensor_parallel_size,
    )


# ---------------------------------------------------------------------------
# DistributedBatchedEngine._validate_model_settings
# ---------------------------------------------------------------------------


def test_pure_tp_admits_mtp_enabled():
    engine = DistributedBatchedEngine(
        _deployment(tensor_parallel_size=2),
        model_settings=SimpleNamespace(mtp_enabled=True),
    )
    engine._validate_model_settings()  # must not raise


def test_pipeline_parallel_still_rejects_mtp_enabled():
    engine = DistributedBatchedEngine(
        _deployment(tensor_parallel_size=1),  # world_size=2, pipeline-parallel
        model_settings=SimpleNamespace(mtp_enabled=True),
    )
    with pytest.raises(ValueError, match="mtp_enabled"):
        engine._validate_model_settings()


def test_pipeline_parallel_error_explains_the_stack_boundary():
    """distributed.py:244's bare incompatibility list cost real investigation
    time; the message must say why, not just what."""

    engine = DistributedBatchedEngine(
        _deployment(tensor_parallel_size=1),
        model_settings=SimpleNamespace(dflash_enabled=True),
    )
    with pytest.raises(ValueError, match="never imports that stack"):
        engine._validate_model_settings()


@pytest.mark.parametrize(
    "name",
    [
        "dflash_enabled",
        "specprefill_enabled",
        "vlm_mtp_enabled",
        "turboquant_kv_enabled",
        # thinking_budget_enabled is deliberately absent: #2731 (already on
        # main) removed it from the incompatible list, since thinking-budget
        # behavior was aligned across engines and no longer needs gating.
    ],
)
def test_the_other_four_settings_stay_rejected_even_for_pure_tp(name):
    engine = DistributedBatchedEngine(
        _deployment(tensor_parallel_size=2),
        model_settings=SimpleNamespace(**{name: True}),
    )
    with pytest.raises(ValueError, match=name):
        engine._validate_model_settings()


def test_no_model_settings_is_a_noop():
    engine = DistributedBatchedEngine(_deployment(tensor_parallel_size=1))
    engine._validate_model_settings()  # must not raise


# ---------------------------------------------------------------------------
# ClusterDeployment: mtp_enabled / mtp_num_draft_tokens
# ---------------------------------------------------------------------------


def test_deployment_rejects_mtp_enabled_for_pipeline_parallel():
    with pytest.raises(ValueError, match="pure tensor-parallel"):
        replace(_deployment(tensor_parallel_size=1), mtp_enabled=True)


def test_deployment_accepts_mtp_enabled_for_pure_tp():
    deployment = replace(
        _deployment(tensor_parallel_size=2),
        mtp_enabled=True,
        mtp_num_draft_tokens=4,
    )
    assert deployment.mtp_enabled is True
    assert deployment.mtp_num_draft_tokens == 4


def test_deployment_rejects_a_non_int_draft_token_count():
    with pytest.raises(ValueError, match="mtp_num_draft_tokens"):
        replace(
            _deployment(tensor_parallel_size=2),
            mtp_enabled=True,
            mtp_num_draft_tokens="5",
        )


def test_out_of_range_draft_token_counts_are_not_rejected_here():
    """Bounds clamping is set_mtp_depth's job (max(1, min(8, depth))), applied
    on the worker; ClusterDeployment matches the single-node contract by only
    validating type, not range, so a value that clamps fine locally does not
    suddenly fail deployment construction."""

    deployment = replace(
        _deployment(tensor_parallel_size=2),
        mtp_enabled=True,
        mtp_num_draft_tokens=99,
    )
    assert deployment.mtp_num_draft_tokens == 99


def test_deployment_round_trips_mtp_fields_through_to_dict_from_dict():
    deployment = replace(
        _deployment(tensor_parallel_size=2),
        mtp_enabled=True,
        mtp_num_draft_tokens=3,
    )
    decoded = ClusterDeployment.from_dict(deployment.to_dict())
    assert decoded.mtp_enabled is True
    assert decoded.mtp_num_draft_tokens == 3


def test_deployment_defaults_mtp_off_when_absent_from_payload():
    """from_dict must tolerate a persisted deployment written before this
    feature existed -- no mtp_enabled/mtp_num_draft_tokens keys at all."""

    payload = _deployment(tensor_parallel_size=2).to_dict()
    del payload["mtp_enabled"]
    del payload["mtp_num_draft_tokens"]
    decoded = ClusterDeployment.from_dict(payload)
    assert decoded.mtp_enabled is False
    assert decoded.mtp_num_draft_tokens is None


# ---------------------------------------------------------------------------
# launch.py: build_mlx_launch_argv threads the two new CLI flags
# ---------------------------------------------------------------------------


def test_launch_argv_carries_mtp_flags_when_enabled(tmp_path):
    deployment = replace(
        _deployment(tensor_parallel_size=2),
        mtp_enabled=True,
        mtp_num_draft_tokens=4,
    )
    argv = build_mlx_launch_argv(
        deployment,
        hostfile=(tmp_path / "hosts.json").resolve(),
        api_port=32100,
        collective_port=32120,
        python_executable="/opt/omlx/bin/python",
        cwd=Path("/opt/omlx"),
    )
    assert "--mtp-enabled" in argv
    assert argv[argv.index("--mtp-num-draft-tokens") + 1] == "4"


def test_launch_argv_omits_draft_tokens_flag_when_unset(tmp_path):
    deployment = replace(_deployment(tensor_parallel_size=2), mtp_enabled=True)
    argv = build_mlx_launch_argv(
        deployment,
        hostfile=(tmp_path / "hosts.json").resolve(),
        api_port=32100,
        collective_port=32120,
        python_executable="/opt/omlx/bin/python",
        cwd=Path("/opt/omlx"),
    )
    assert "--mtp-enabled" in argv
    assert "--mtp-num-draft-tokens" not in argv


def test_launch_argv_omits_mtp_enabled_flag_when_off(tmp_path):
    deployment = _deployment(tensor_parallel_size=2)
    argv = build_mlx_launch_argv(
        deployment,
        hostfile=(tmp_path / "hosts.json").resolve(),
        api_port=32100,
        collective_port=32120,
        python_executable="/opt/omlx/bin/python",
        cwd=Path("/opt/omlx"),
    )
    assert "--mtp-enabled" not in argv
    assert "--mtp-num-draft-tokens" not in argv


# ---------------------------------------------------------------------------
# batch_generator.py: rank-0-owned depth/park broadcast + desync checksum
# ---------------------------------------------------------------------------


class _FakeGroup:
    def __init__(self, size: int) -> None:
        self._size = size

    def size(self) -> int:
        return self._size


@pytest.fixture(autouse=True)
def _reset_distributed_mtp():
    bg.configure_distributed_mtp(group=None)
    yield
    bg.configure_distributed_mtp(group=None)


def test_configure_distributed_mtp_disabled_by_default():
    assert bg._DISTRIBUTED_MTP is None
    assert bg._mtp_depth_controller_allowed() is True  # single-node: always allowed


def test_only_the_coordinator_may_construct_a_depth_controller():
    bg.configure_distributed_mtp(group=_FakeGroup(2), coordinator=True)
    assert bg._mtp_depth_controller_allowed() is True

    bg.configure_distributed_mtp(group=_FakeGroup(2), coordinator=False)
    assert bg._mtp_depth_controller_allowed() is False


def _two_rank_all_sum(coordinator_payload, worker_payload):
    """Simulate mx.distributed.all_sum(group=size-2) without a real group."""

    def summed(payload, group):
        return tuple(a + b for a, b in zip(payload, worker_payload))

    return summed


def test_depth_and_park_agree_across_ranks_without_a_real_collective():
    bg.configure_distributed_mtp(group=_FakeGroup(2), coordinator=True, checksum=True)
    state = bg._MtpState(uid=1)
    state._omlx_dist_sync = True
    state.controller = bg._DepthController(3)
    state.controller.cur = 2  # simulate a mid-run decision
    committed = [10, 11, 12]
    worker_hash = bg._mtp_cycle_hash(2, 1, 0, committed)

    bg._sync_distributed_mtp_cycle(
        state,
        2,
        1,
        committed,
        all_sum=_two_rank_all_sum(None, (0, 0, worker_hash)),
    )

    assert state.depth == 2  # rank 0's decision, not the worker's
    assert state._dist_should_exit is False
    assert bg._mtp_should_exit(state) is False


def test_park_decision_propagates_from_coordinator_to_workers():
    bg.configure_distributed_mtp(group=_FakeGroup(2), coordinator=True, checksum=True)
    state = bg._MtpState(uid=1)
    state._omlx_dist_sync = True
    state.controller = bg._DepthController(3)
    state.controller.exit_streak = state.controller.EXIT_STREAK  # force should_exit()
    committed = [7]
    worker_hash = bg._mtp_cycle_hash(1, 0, 0, committed)

    bg._sync_distributed_mtp_cycle(
        state,
        1,
        0,
        committed,
        all_sum=_two_rank_all_sum(None, (0, 0, worker_hash)),
    )

    assert state._dist_should_exit is True
    assert bg._mtp_should_exit(state) is True


def test_worker_rank_has_no_controller_and_reads_the_broadcast():
    """The worker's own _sync_distributed_mtp_cycle call: it contributes
    zeros (no local controller/decision), and still ends up with rank 0's
    agreed depth/park via the same collective."""

    bg.configure_distributed_mtp(group=_FakeGroup(2), coordinator=False, checksum=True)
    state = bg._MtpState(uid=1)
    state._omlx_dist_sync = True
    assert state.controller is None
    committed = [1, 2]

    coordinator_payload = (3, 1, bg._mtp_cycle_hash(2, 2, 0, committed))

    def fake_all_sum(payload, group):
        # payload here is the worker's own zero contribution + its checksum.
        return tuple(a + b for a, b in zip(coordinator_payload, payload))

    bg._sync_distributed_mtp_cycle(
        state, 2, 2, committed, all_sum=fake_all_sum
    )

    assert state.depth == 3
    assert state._dist_should_exit is True


def test_checksum_mismatch_raises_instead_of_silently_hanging_later():
    bg.configure_distributed_mtp(group=_FakeGroup(2), coordinator=True, checksum=True)
    state = bg._MtpState(uid=1)
    state._omlx_dist_sync = True
    state.controller = bg._DepthController(3)
    committed = [1, 2, 3]

    def mismatched_all_sum(payload, group):
        # A worker whose committed tokens (or accept/depth decision) diverged
        # contributes a checksum that does not match.
        return tuple(a + b for a, b in zip(payload, (0, 0, 424242)))

    with pytest.raises(RuntimeError, match="desync"):
        bg._sync_distributed_mtp_cycle(
            state, 2, 1, committed, all_sum=mismatched_all_sum
        )


def test_checksum_disabled_lets_a_mismatch_through():
    """The debug detector must be disable-able (OMLX_MTP_DISTRIBUTED_CHECKSUM)
    once trust is established, without changing the payload shape -- only
    whether a mismatch raises."""

    bg.configure_distributed_mtp(group=_FakeGroup(2), coordinator=True, checksum=False)
    state = bg._MtpState(uid=1)
    state._omlx_dist_sync = True
    state.controller = bg._DepthController(3)
    committed = [1, 2, 3]

    def mismatched_all_sum(payload, group):
        return tuple(a + b for a, b in zip(payload, (0, 0, 424242)))

    bg._sync_distributed_mtp_cycle(
        state, 2, 1, committed, all_sum=mismatched_all_sum
    )  # must not raise


def test_sync_is_a_noop_when_the_sequence_is_not_under_distributed_sync():
    """Single-node (or depth<=1 / non-chain) sequences must never reach the
    collective at all -- _omlx_dist_sync stays False."""

    bg.configure_distributed_mtp(group=_FakeGroup(2), coordinator=True)
    state = bg._MtpState(uid=1)
    assert state._omlx_dist_sync is False

    def exploding_all_sum(payload, group):
        raise AssertionError("must not be called when _omlx_dist_sync is False")

    bg._sync_distributed_mtp_cycle(state, 1, 1, [1], all_sum=exploding_all_sum)
    assert state.depth == 1  # untouched


def test_single_node_should_exit_is_unchanged_by_the_distributed_glue():
    """No _DISTRIBUTED_MTP configured: _mtp_should_exit must fall back to the
    original controller.should_exit() read, byte-for-byte the pre-existing
    single-node behavior."""

    state = bg._MtpState(uid=1)
    state.controller = bg._DepthController(3)
    assert bg._mtp_should_exit(state) is False
    state.controller.exit_streak = state.controller.EXIT_STREAK
    assert bg._mtp_should_exit(state) is True
