# SPDX-License-Identifier: Apache-2.0
"""State-machine tests for the hotjoin scaffold (no effects, all pure)."""

import pytest

from omlx.cluster.hotjoin import (
    HotJoinController,
    InvalidMembershipTransition,
    MembershipEvent,
    MembershipPhase,
)


@pytest.fixture()
def controller():
    return HotJoinController(deployment_id="cluster-test")


# ---------------------------------------------------------------------------
# Legal paths
# ---------------------------------------------------------------------------


def test_join_happy_path_walks_the_whole_machine(controller):
    assert controller.phase is MembershipPhase.STABLE

    controller.apply(MembershipEvent.JOIN_REQUESTED, node_id="mini")
    assert controller.phase is MembershipPhase.JOIN_EVALUATING
    assert controller.pending_node == "mini"

    controller.apply(MembershipEvent.JOIN_COMPATIBLE)
    controller.apply(MembershipEvent.DRAIN_COMPLETE)
    controller.apply(MembershipEvent.TEARDOWN_VERIFIED)
    controller.apply(MembershipEvent.PLAN_READY)
    controller.apply(MembershipEvent.PROVISIONED)
    controller.apply(MembershipEvent.RELOAD_READY)
    assert controller.phase is MembershipPhase.VERIFYING

    controller.apply(MembershipEvent.CANARY_PASSED)
    assert controller.phase is MembershipPhase.STABLE
    assert controller.pending_node is None
    assert len(controller.history) == 8


def test_leave_request_skips_evaluation_and_provisioning_is_for_joins(controller):
    controller.apply(MembershipEvent.LEAVE_REQUESTED, node_id="mini")
    assert controller.phase is MembershipPhase.DRAINING
    controller.apply(MembershipEvent.DRAIN_COMPLETE)
    controller.apply(MembershipEvent.TEARDOWN_VERIFIED)
    controller.apply(MembershipEvent.PLAN_READY)
    assert controller.phase is MembershipPhase.PROVISIONING
    # Leaving still stages the coordinator's new stage layout, so the
    # provisioning phase applies to joins and leaves alike.


def test_node_lost_fails_closed_into_full_drain(controller):
    controller.apply(MembershipEvent.NODE_LOST, node_id="studio")

    assert controller.phase is MembershipPhase.DRAINING


def test_drain_timeout_keeps_the_old_world(controller):
    controller.apply(MembershipEvent.LEAVE_REQUESTED, node_id="mini")
    controller.apply(MembershipEvent.DRAIN_TIMEOUT)

    assert controller.phase is MembershipPhase.STABLE


@pytest.mark.parametrize(
    "failure_event,phase_before",
    [
        (MembershipEvent.PLAN_FAILED, MembershipPhase.REPLANNING),
        (MembershipEvent.PROVISION_FAILED, MembershipPhase.PROVISIONING),
        (MembershipEvent.RELOAD_FAILED, MembershipPhase.RELOADING),
        (MembershipEvent.CANARY_FAILED, MembershipPhase.VERIFYING),
    ],
)
def test_post_teardown_failures_roll_back(controller, failure_event, phase_before):
    controller.apply(MembershipEvent.LEAVE_REQUESTED, node_id="mini")
    controller.apply(MembershipEvent.DRAIN_COMPLETE)
    controller.apply(MembershipEvent.TEARDOWN_VERIFIED)
    while controller.phase is not phase_before:
        controller.apply(
            {
                MembershipPhase.REPLANNING: MembershipEvent.PLAN_READY,
                MembershipPhase.PROVISIONING: MembershipEvent.PROVISIONED,
                MembershipPhase.RELOADING: MembershipEvent.RELOAD_READY,
            }[controller.phase]
        )

    controller.apply(failure_event)
    assert controller.phase is MembershipPhase.ROLLING_BACK

    controller.apply(MembershipEvent.ROLLBACK_COMPLETE)
    assert controller.phase is MembershipPhase.STABLE


def test_teardown_failure_is_terminal_until_operator_reset(controller):
    controller.apply(MembershipEvent.LEAVE_REQUESTED, node_id="mini")
    controller.apply(MembershipEvent.DRAIN_COMPLETE)
    controller.apply(MembershipEvent.TEARDOWN_FAILED)

    assert controller.phase is MembershipPhase.FAILED
    # Invariant 1: no automatic path out of an unverified teardown.
    for event in MembershipEvent:
        if event is MembershipEvent.OPERATOR_RESET:
            continue
        assert not controller.can(event), event

    controller.apply(MembershipEvent.OPERATOR_RESET)
    assert controller.phase is MembershipPhase.STABLE


def test_rollback_failure_strands_in_failed(controller):
    controller.apply(MembershipEvent.LEAVE_REQUESTED, node_id="mini")
    controller.apply(MembershipEvent.DRAIN_COMPLETE)
    controller.apply(MembershipEvent.TEARDOWN_VERIFIED)
    controller.apply(MembershipEvent.PLAN_FAILED)
    controller.apply(MembershipEvent.ROLLBACK_FAILED)

    assert controller.phase is MembershipPhase.FAILED


def test_incompatible_candidate_returns_to_stable(controller):
    controller.apply(MembershipEvent.JOIN_REQUESTED, node_id="mini")
    controller.apply(MembershipEvent.JOIN_INCOMPATIBLE)

    assert controller.phase is MembershipPhase.STABLE
    assert controller.pending_node is None


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------


def test_illegal_transition_raises_and_preserves_phase(controller):
    with pytest.raises(InvalidMembershipTransition, match="not valid while stable"):
        controller.apply(MembershipEvent.DRAIN_COMPLETE)

    assert controller.phase is MembershipPhase.STABLE
    assert controller.history == []


def test_member_events_require_node_id(controller):
    with pytest.raises(ValueError, match="requires node_id"):
        controller.apply(MembershipEvent.JOIN_REQUESTED)


def test_event_type_is_checked(controller):
    with pytest.raises(TypeError):
        controller.apply("join_requested")


def test_can_reports_legality_without_mutating(controller):
    assert controller.can(MembershipEvent.JOIN_REQUESTED)
    assert not controller.can(MembershipEvent.CANARY_PASSED)
    assert controller.phase is MembershipPhase.STABLE
    assert controller.history == []


def test_history_is_an_auditable_trail(controller):
    controller.apply(MembershipEvent.JOIN_REQUESTED, node_id="mini", at=1000.0)
    controller.apply(MembershipEvent.JOIN_INCOMPATIBLE, at=1001.0)

    payload = controller.to_dict()
    assert payload["phase"] == "stable"
    assert payload["history"] == [
        {
            "from": "stable",
            "event": "join_requested",
            "to": "join_evaluating",
            "node_id": "mini",
            "at": 1000.0,
        },
        {
            "from": "join_evaluating",
            "event": "join_incompatible",
            "to": "stable",
            "node_id": None,
            "at": 1001.0,
        },
    ]


# ---------------------------------------------------------------------------
# NotImplemented seam
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "verb,args",
    [
        ("evaluate_candidate", ("mini",)),
        ("drain", ()),
        ("teardown", ()),
        ("replan", ()),
        ("provision", ()),
        ("reload", ()),
        ("canary", ()),
    ],
)
def test_effect_verbs_are_scaffolded(controller, verb, args):
    with pytest.raises(NotImplementedError, match="scaffold"):
        getattr(controller, verb)(*args)


def test_scaffold_errors_point_at_the_supported_replan_path(controller):
    try:
        controller.teardown()
    except NotImplementedError as exc:
        assert "POST /admin/api/cluster/replan" in str(exc)
