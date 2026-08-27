# SPDX-License-Identifier: Apache-2.0
"""SCAFFOLD — hot cluster membership (join/leave mid-service).

**Status: documented seam, not an implementation.** Nothing in this module
launches, kills, or probes anything. The state machine below is the designed
contract; every verb that would touch the engine pool, the launcher, or the
network raises :class:`NotImplementedError`. Until this lands, a membership
change is a full deactivate → re-plan → reload, which ``POST
/admin/api/cluster/replan`` (``omlx/cluster/routes.py``) already performs as
one action.

## Why this exists

Today world size, rank→layer assignment, and the plan hash are frozen in the
persisted ``ClusterDeployment`` and re-validated by every rank before loading
— a membership change is defined as a new deployment. The medium-term design
(``docs/heterogeneous-cluster.md``, "ConnectX CUDA pairs and the future
supernode gateway") keeps the *outer* MLX ring stable while *inner* groups
(composite stages behind a gateway process) join and leave, so adding a
supernode would not restart the whole world. That requires the gateway runner
the doc specifies; until it exists, "hot" membership means automating the
restart dance safely and observably — which is what this state machine
describes.

## The state machine

Phases (``MembershipPhase``):

```
                 JOIN_REQUESTED                 LEAVE_REQUESTED / NODE_LOST
   STABLE ──────────────────────► JOIN_EVALUATING ─┐
     ▲                            │                │
     │ JOIN_INCOMPATIBLE          │ JOIN_COMPATIBLE│
     └────────────────────────────┘                ▼
     │                                       DRAINING  ◄── in-flight requests
     │ DRAIN_TIMEOUT (stay on old world)       │        quiesce (engine-pool gate)
     └─────────────────────────────────────────┤
                                               ▼ DRAIN_COMPLETE
                                        TEARING_DOWN ── verified teardown is
                                               │        the memory barrier
                              TEARDOWN_FAILED  ▼ TEARDOWN_VERIFIED
                                               REPLANNING
                                               │ PLAN_READY (PLAN_FAILED ↓)
                                               ▼
                                        PROVISIONING ── shard staging for a
                                               │ PROVISIONED   joining node
                                               ▼             (PROVISION_FAILED ↓)
                                        RELOADING ── RELOAD_READY / RELOAD_FAILED ↓
                                               ▼
                                        VERIFYING ── CANARY_PASSED → STABLE
                                               │ CANARY_FAILED
                                               ▼
                                        ROLLING_BACK ── ROLLBACK_COMPLETE → STABLE
                                               │ ROLLBACK_FAILED
                                               ▼
                                            FAILED ── OPERATOR_RESET → STABLE
```

Invariants the machine enforces (and the tests pin):

1. **Fail-closed on teardown doubt.** ``TEARDOWN_FAILED`` goes to ``FAILED``,
   never to ``REPLANNING``: admitting a new world while an old rank's unified
   memory may still be resident is the orphan-GPU-memory hole (audit G1/G2).
2. **A refused drain keeps the old world.** ``DRAIN_TIMEOUT`` returns to
   ``STABLE`` with the previous deployment untouched.
3. **Every post-teardown failure rolls back** to the last verified-serving
   deployment; only a rollback failure strands the cluster in ``FAILED``.
4. **Loss of any member is total.** ``NODE_LOST`` drains the whole world —
   a pipeline/TP group cannot degrade (matches the gateway doc: loss of
   either supernode member removes the whole supernode and invalidates the
   approved plan).
5. The machine is a pure function of ``(phase, event)``; all effects hang off
   verbs that are not implemented yet.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MembershipPhase(Enum):
    STABLE = "stable"
    JOIN_EVALUATING = "join_evaluating"
    DRAINING = "draining"
    TEARING_DOWN = "tearing_down"
    REPLANNING = "replanning"
    PROVISIONING = "provisioning"
    RELOADING = "reloading"
    VERIFYING = "verifying"
    ROLLING_BACK = "rolling_back"
    FAILED = "failed"


class MembershipEvent(Enum):
    JOIN_REQUESTED = "join_requested"
    JOIN_COMPATIBLE = "join_compatible"
    JOIN_INCOMPATIBLE = "join_incompatible"
    LEAVE_REQUESTED = "leave_requested"
    NODE_LOST = "node_lost"
    DRAIN_COMPLETE = "drain_complete"
    DRAIN_TIMEOUT = "drain_timeout"
    TEARDOWN_VERIFIED = "teardown_verified"
    TEARDOWN_FAILED = "teardown_failed"
    PLAN_READY = "plan_ready"
    PLAN_FAILED = "plan_failed"
    PROVISIONED = "provisioned"
    PROVISION_FAILED = "provision_failed"
    RELOAD_READY = "reload_ready"
    RELOAD_FAILED = "reload_failed"
    CANARY_PASSED = "canary_passed"
    CANARY_FAILED = "canary_failed"
    ROLLBACK_COMPLETE = "rollback_complete"
    ROLLBACK_FAILED = "rollback_failed"
    OPERATOR_RESET = "operator_reset"


_TRANSITIONS: dict[tuple[MembershipPhase, MembershipEvent], MembershipPhase] = {
    (MembershipPhase.STABLE, MembershipEvent.JOIN_REQUESTED): (
        MembershipPhase.JOIN_EVALUATING
    ),
    (MembershipPhase.JOIN_EVALUATING, MembershipEvent.JOIN_COMPATIBLE): (
        MembershipPhase.DRAINING
    ),
    (MembershipPhase.JOIN_EVALUATING, MembershipEvent.JOIN_INCOMPATIBLE): (
        MembershipPhase.STABLE
    ),
    (MembershipPhase.STABLE, MembershipEvent.LEAVE_REQUESTED): (
        MembershipPhase.DRAINING
    ),
    # A dead member cannot be negotiated with: the collective is already
    # broken, so the world drains (fail-closed), it does not degrade.
    (MembershipPhase.STABLE, MembershipEvent.NODE_LOST): MembershipPhase.DRAINING,
    (MembershipPhase.DRAINING, MembershipEvent.DRAIN_COMPLETE): (
        MembershipPhase.TEARING_DOWN
    ),
    (MembershipPhase.DRAINING, MembershipEvent.DRAIN_TIMEOUT): (
        MembershipPhase.STABLE
    ),
    (MembershipPhase.TEARING_DOWN, MembershipEvent.TEARDOWN_VERIFIED): (
        MembershipPhase.REPLANNING
    ),
    # Invariant 1: unverified teardown is terminal for the automatic path.
    (MembershipPhase.TEARING_DOWN, MembershipEvent.TEARDOWN_FAILED): (
        MembershipPhase.FAILED
    ),
    (MembershipPhase.REPLANNING, MembershipEvent.PLAN_READY): (
        MembershipPhase.PROVISIONING
    ),
    (MembershipPhase.REPLANNING, MembershipEvent.PLAN_FAILED): (
        MembershipPhase.ROLLING_BACK
    ),
    (MembershipPhase.PROVISIONING, MembershipEvent.PROVISIONED): (
        MembershipPhase.RELOADING
    ),
    (MembershipPhase.PROVISIONING, MembershipEvent.PROVISION_FAILED): (
        MembershipPhase.ROLLING_BACK
    ),
    (MembershipPhase.RELOADING, MembershipEvent.RELOAD_READY): (
        MembershipPhase.VERIFYING
    ),
    (MembershipPhase.RELOADING, MembershipEvent.RELOAD_FAILED): (
        MembershipPhase.ROLLING_BACK
    ),
    (MembershipPhase.VERIFYING, MembershipEvent.CANARY_PASSED): (
        MembershipPhase.STABLE
    ),
    (MembershipPhase.VERIFYING, MembershipEvent.CANARY_FAILED): (
        MembershipPhase.ROLLING_BACK
    ),
    (MembershipPhase.ROLLING_BACK, MembershipEvent.ROLLBACK_COMPLETE): (
        MembershipPhase.STABLE
    ),
    (MembershipPhase.ROLLING_BACK, MembershipEvent.ROLLBACK_FAILED): (
        MembershipPhase.FAILED
    ),
    (MembershipPhase.FAILED, MembershipEvent.OPERATOR_RESET): (
        MembershipPhase.STABLE
    ),
}

#: Events that carry a member's node_id (the candidate joining, the member
#: leaving, or the member that vanished).
_MEMBER_EVENTS = frozenset(
    {
        MembershipEvent.JOIN_REQUESTED,
        MembershipEvent.LEAVE_REQUESTED,
        MembershipEvent.NODE_LOST,
    }
)


class InvalidMembershipTransition(ValueError):
    """An event arrived that the current phase does not allow."""


@dataclass(frozen=True)
class MembershipTransition:
    """One recorded step, for the audit trail the cluster event log will own."""

    phase_from: MembershipPhase
    event: MembershipEvent
    phase_to: MembershipPhase
    node_id: str | None
    at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "from": self.phase_from.value,
            "event": self.event.value,
            "to": self.phase_to.value,
            "node_id": self.node_id,
            "at": self.at,
        }


@dataclass
class HotJoinController:
    """Pure membership state machine for one clustered model.

    Tracks the phase of a membership change and the audit trail. All real
    effects (draining the engine pool, verified teardown, planning, staging,
    reload, canary) are the ``*_unsupported`` verbs below, which raise
    :class:`NotImplementedError` pointing at the machinery that will back
    them. ``apply`` is pure and fully testable.
    """

    deployment_id: str | None = None
    phase: MembershipPhase = MembershipPhase.STABLE
    pending_node: str | None = None
    history: list[MembershipTransition] = field(default_factory=list)

    def can(self, event: MembershipEvent) -> bool:
        """Whether *event* is legal in the current phase."""

        return (self.phase, event) in _TRANSITIONS

    def apply(
        self,
        event: MembershipEvent,
        *,
        node_id: str | None = None,
        at: float | None = None,
    ) -> MembershipPhase:
        """Advance the machine, recording the transition. Pure."""

        if not isinstance(event, MembershipEvent):
            raise TypeError("event must be a MembershipEvent")
        if event in _MEMBER_EVENTS and not node_id:
            raise ValueError(f"{event.value} requires node_id")
        target = _TRANSITIONS.get((self.phase, event))
        if target is None:
            raise InvalidMembershipTransition(
                f"{event.value} is not valid while {self.phase.value} "
                f"(deployment {self.deployment_id or 'none'})"
            )
        transition = MembershipTransition(
            phase_from=self.phase,
            event=event,
            phase_to=target,
            node_id=node_id,
            at=time.time() if at is None else at,
        )
        self.history.append(transition)
        self.phase = target
        if event is MembershipEvent.JOIN_REQUESTED:
            self.pending_node = node_id
        if target in (MembershipPhase.STABLE, MembershipPhase.FAILED):
            self.pending_node = None
        return target

    # -- Effect verbs: the seam. Each names the machinery that will back it.

    def _unsupported(self, verb: str, backing: str) -> NotImplementedError:
        return NotImplementedError(
            f"hot membership '{verb}' is a scaffold: {backing}. Until this "
            "lands, a membership change is one POST /admin/api/cluster/replan "
            "(deactivate → re-plan → reload) with no mid-service continuity."
        )

    def evaluate_candidate(self, node_id: str) -> None:
        """Probe the candidate (capabilities, model presence, rdma_ctl)."""

        raise self._unsupported(
            "evaluate_candidate",
            "will compose omlx/cluster/probe.py, peer-probe and the "
            "backend-selection rule in omlx/cluster/backends.py",
        )

    def drain(self) -> None:
        """Quiesce in-flight requests without interrupting them."""

        raise self._unsupported(
            "drain",
            "will drive EnginePool.prepare_cluster_reload's quiescence gate "
            "and rank-side drain evidence (engine_pool._entry_is_quiescent)",
        )

    def teardown(self) -> None:
        """Verified teardown of the current world."""

        raise self._unsupported(
            "teardown",
            "will drive DistributedJobSupervisor._terminate and must treat "
            "DistributedTeardownError as TEARDOWN_FAILED (fail-closed)",
        )

    def replan(self) -> None:
        """Build the N-node plan for the new membership."""

        raise self._unsupported(
            "replan",
            "will call plan_unequal_pipeline / plan_proportional_pipeline "
            "and sign the placement the way /admin/api/cluster/replan does",
        )

    def provision(self) -> None:
        """Stage this node's shard files (joining members only)."""

        raise self._unsupported(
            "provision",
            "will drive omlx/cluster/staging.py against the new plan",
        )

    def reload(self) -> None:
        """Launch the new world and wait for every rank's readiness."""

        raise self._unsupported(
            "reload",
            "will drive the engine pool reload behind the same launch "
            "manifest + readiness contract activation uses",
        )

    def canary(self) -> None:
        """Prove the new world serves before declaring STABLE."""

        raise self._unsupported(
            "canary",
            "will issue the readiness generation activation performs",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "deployment_id": self.deployment_id,
            "phase": self.phase.value,
            "pending_node": self.pending_node,
            "history": [transition.to_dict() for transition in self.history],
        }
