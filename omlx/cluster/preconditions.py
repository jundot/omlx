# SPDX-License-Identifier: Apache-2.0
"""B4: the "Why can't I start?" precondition rows.

One row per thing that must be true before the cluster can start — SSH
reachable, fabric link up, model staged everywhere, memory budget fits,
parallelism strategy compatible — each with live evidence, the evidence's
age, and a fix affordance. The GUI panel and ``omlx cluster status
--explain`` both render exactly these rows, so they can never tell
different stories.

Everything here is **pure over already-collected evidence**: no SSH, no
file I/O, no subprocess. The endpoint layer (``routes.cluster_readiness``)
gathers the evidence with bounded timeouts and hands it in; each derivation
below is therefore testable without two Macs.

**Parallelism authority (inherited from B5, see
``routes._strategy_support``):** the ``strategy`` row consumes the
plan-time flags computed from ``ModelLayout.supports_tensor_parallel`` /
``supports_pipeline`` plus the divisor arithmetic. It deliberately does NOT
consult ``pipeline_compat.pipeline_assignment_is_honored()`` — that answers
a different, narrower question (does the installed runtime hook consume an
unequal layer assignment) and stays the load-time enforcement inside the
worker path. This row is UX; the ``PlanningError`` paths remain the safety
boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .models import TransportState
from .readiness import (
    is_fabric_verified,
    ladder_copy,
    ladder_rank,
    link_ladder_state,
)

# Row states are presentation; readiness is the boolean conjunction from
# design B.2: all(ssh_ok, fabric >= reachable, model_staged,
# budget_fits_live, strategy_compatible). A WARN row can still satisfy its
# conjunct (a verified Ethernet ring starts fine) or not (no model chosen
# yet: nothing is broken, but Start cannot proceed).
PASS = "pass"
WARN = "warn"
FAIL = "fail"

ROW_IDS: tuple[str, ...] = ("ssh", "fabric", "staging", "budget", "strategy")

_GIB = 1024**3


@dataclass(frozen=True)
class PreconditionRow:
    """One precondition with live evidence and the action that fixes it."""

    id: str
    state: str  # "pass" | "warn" | "fail"
    evidence: str
    evidence_age_s: float = 0.0
    fix: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "state": self.state,
            "evidence": self.evidence,
            "evidence_age_s": self.evidence_age_s,
            "fix": dict(self.fix) if self.fix else None,
        }


@dataclass(frozen=True)
class ReadinessReport:
    """The five rows plus the cluster-level conjunction."""

    ready: bool
    rows: tuple[PreconditionRow, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "rows": [row.to_dict() for row in self.rows],
        }


def _get(item: Any, name: str, default: Any = None) -> Any:
    """Read a field from a mapping or an object (PeerHealth, dict, …)."""

    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _gib(value: Any) -> str:
    return f"{int(value or 0) / _GIB:.1f} GiB"


def _aged(evidence: str, age: float | None) -> str:
    """Evidence always carries its age in the string as well as the field."""

    if age is None:
        return evidence
    return f"{evidence} · {int(age)} s ago"


def _ssh_row(
    peers: Sequence[Any],
    age: float | None,
) -> tuple[PreconditionRow, bool]:
    if not peers:
        return (
            PreconditionRow(
                id="ssh",
                state=PASS,
                evidence=_aged("this Mac only — no remote workers to reach", age),
                evidence_age_s=float(age or 0.0),
                fix=None,
            ),
            True,
        )
    unreachable = [peer for peer in peers if not _get(peer, "reachable", False)]
    if unreachable:
        parts = []
        for peer in unreachable:
            host = str(_get(peer, "host", "") or _get(peer, "node_id", "peer"))
            detail = str(_get(peer, "detail", "") or "did not answer")
            parts.append(f"{host} ✗ {detail}")
        evidence = "; ".join(parts)
        state = FAIL
        ok = False
    else:
        evidence = " · ".join(
            f"{_get(peer, 'host', _get(peer, 'node_id', 'peer'))} ✓ answered"
            for peer in peers
        )
        state = PASS
        ok = True
    return (
        PreconditionRow(
            id="ssh",
            state=state,
            evidence=_aged(evidence, age),
            evidence_age_s=float(age or 0.0),
            fix={"kind": "reverify"},
        ),
        ok,
    )


def _fabric_row(
    *,
    fabric_required: bool,
    link_status: Any,
    shared_link: Any,
    fabric_ok: bool | None,
    fabric_detail: str,
    age: float | None,
) -> tuple[PreconditionRow, bool]:
    if not fabric_required:
        return (
            PreconditionRow(
                id="fabric",
                state=PASS,
                evidence=_aged("single Mac — no fabric link needed", age),
                evidence_age_s=float(age or 0.0),
                fix=None,
            ),
            True,
        )
    rung = link_ladder_state(link_status, shared_link)
    reason, remedy = ladder_copy(rung)
    # The gate the Start button already answers to: every pair resolved a
    # shared, verified address (``_resolve_fabric``'s ``ok``). Where the
    # caller has no such verdict, the B3 ladder decides: REACHABLE or above
    # is two-ended proof.
    ok = (
        bool(fabric_ok)
        if fabric_ok is not None
        else ladder_rank(rung) >= ladder_rank(TransportState.REACHABLE)
    )
    if not ok:
        state = FAIL
        evidence = fabric_detail or reason
        if remedy:
            evidence = f"{evidence} — {remedy}"
    elif (
        is_fabric_verified(rung)
        or ladder_rank(rung) >= ladder_rank(TransportState.REACHABLE)
    ):
        state = PASS
        evidence = fabric_detail or reason
    else:
        # The route works (fabric_ok said so) but the evidence supports no
        # RDMA rung — a verified Ethernet TCP ring. Startable, but say so.
        state = WARN
        evidence = fabric_detail or reason
        evidence = f"{evidence} (TCP ring — no RDMA fabric)"
    return (
        PreconditionRow(
            id="fabric",
            state=state,
            evidence=_aged(evidence, age),
            evidence_age_s=float(age or 0.0),
            fix={"kind": "doctor"},
        ),
        ok,
    )


def _staging_row(
    *,
    staging: Mapping[str, Any] | None,
    staging_job_id: str,
    required_bytes: int,
    model_error: str,
    age: float | None,
) -> tuple[PreconditionRow, bool]:
    fix: dict[str, Any] | None = None
    if model_error:
        state, ok = FAIL, False
        evidence = f"the model could not be read: {model_error}"
    elif staging is None:
        if required_bytes > 0:
            # A model is selected but no manifest could be computed yet.
            state, ok = WARN, False
            evidence = "staging has not been checked yet"
        else:
            state, ok = WARN, False
            evidence = "choose a model to check what must be copied"
    elif staging.get("error"):
        state, ok = FAIL, False
        evidence = str(staging["error"])
    elif staging.get("ready"):
        nodes = staging.get("nodes") or ()
        state, ok = PASS, True
        evidence = (
            f"every node already holds its files ({len(nodes)} checked)"
            if nodes
            else "every node already holds its files"
        )
    else:
        state, ok = FAIL, False
        missing = [
            node
            for node in (staging.get("nodes") or ())
            if not node.get("ready")
        ]
        parts = [
            f"⟳ {node.get('node_id', 'node')} missing "
            f"{_gib(int(node.get('missing_bytes') or 0) + int(node.get('missing_sidecar_bytes') or 0))}"
            for node in missing
        ] or [f"missing {_gib(staging.get('total_missing_bytes'))} in total"]
        evidence = " · ".join(parts)
        fix = {"kind": "stage_details", "job_id": staging_job_id or None}
    return (
        PreconditionRow(
            id="staging",
            state=state,
            evidence=_aged(evidence, age),
            evidence_age_s=float(age or 0.0),
            fix=fix,
        ),
        ok,
    )


def _budget_row(
    *,
    budgets: Sequence[Mapping[str, Any]],
    required_bytes: int,
    plan_error: str,
    model_error: str,
    age: float | None,
) -> tuple[PreconditionRow, bool]:
    def _binding(budget: Mapping[str, Any]) -> str:
        # /node-budgets carries ``binding`` inside B5's ``breakdown``; pure
        # test fixtures may put it at the top level. Accept both.
        return str(
            budget.get("binding")
            or (budget.get("breakdown") or {}).get("binding")
            or ""
        )

    def _worst_node() -> Mapping[str, Any] | None:
        if not budgets:
            return None
        # The node whose constraint binds hardest: prefer a role reserve
        # (B5's ``binding``), then the largest reserve — that is the node
        # whose role change gains the most usable memory.
        reserved = [
            budget
            for budget in budgets
            if _binding(budget) == "role_reserve"
        ]
        pool = reserved or list(budgets)
        return max(pool, key=lambda budget: int(budget.get("reserve_bytes") or 0))

    fix: dict[str, Any] | None = None
    if model_error:
        state, ok = FAIL, False
        evidence = f"the model could not be read: {model_error}"
    elif required_bytes <= 0:
        state, ok = WARN, False
        evidence = "choose a model to compare against each Mac's live budget"
    elif not budgets:
        state, ok = FAIL, False
        evidence = "no Mac reported a usable memory budget"
    elif plan_error:
        state, ok = FAIL, False
        worst = _worst_node()
        evidence = plan_error
        if worst is not None:
            evidence += (
                f" — {worst.get('node_id', 'node')} is the "
                f"{worst.get('role', 'headless')} role, reserving "
                f"{_gib(worst.get('reserve_bytes'))}"
            )
            fix = {"kind": "role_editor", "node_id": worst.get("node_id")}
    else:
        usable = sum(int(budget.get("usable_bytes") or 0) for budget in budgets)
        if usable >= required_bytes:
            state, ok = PASS, True
            evidence = (
                f"model needs {_gib(required_bytes)} · "
                f"{_gib(usable)} usable across {len(budgets)} node(s)"
            )
        else:
            state, ok = FAIL, False
            worst = _worst_node()
            evidence = (
                f"model needs {_gib(required_bytes)} but only "
                f"{_gib(usable)} is usable across {len(budgets)} node(s)"
            )
            if worst is not None:
                evidence += (
                    f" — {worst.get('node_id', 'node')} is the "
                    f"{worst.get('role', 'headless')} role, reserving "
                    f"{_gib(worst.get('reserve_bytes'))} "
                    f"(binding: {_binding(worst) or 'role_reserve'})"
                )
                fix = {"kind": "role_editor", "node_id": worst.get("node_id")}
    return (
        PreconditionRow(
            id="budget",
            state=state,
            evidence=_aged(evidence, age),
            evidence_age_s=float(age or 0.0),
            fix=fix,
        ),
        ok,
    )


def _strategy_row(
    *,
    strategies: Mapping[str, Any] | None,
    model_error: str,
    age: float | None,
) -> tuple[PreconditionRow, bool]:
    if model_error:
        state, ok = FAIL, False
        evidence = f"the model could not be read: {model_error}"
    elif not strategies:
        state, ok = WARN, False
        evidence = "choose a model to evaluate parallelism strategies"
    else:
        tensor = dict(strategies.get("tensor") or {})
        pipeline = dict(strategies.get("pipeline") or {})
        supported = [
            name
            for name, mode in (("tensor", tensor), ("pipeline", pipeline))
            if mode.get("supported")
        ]
        if supported:
            state, ok = PASS, True
            unsupported = [
                f"{name}: {str(mode.get('reason') or 'not supported')}"
                for name, mode in (("tensor", tensor), ("pipeline", pipeline))
                if not mode.get("supported")
            ]
            evidence = f"{' and '.join(supported)} parallelism supported"
            if unsupported:
                evidence += f" ({'; '.join(unsupported)})"
        else:
            state, ok = FAIL, False
            reasons = [
                str(mode.get("reason") or f"{name} parallelism is not possible")
                for name, mode in (("tensor", tensor), ("pipeline", pipeline))
            ]
            evidence = " ".join(reasons)
    return (
        PreconditionRow(
            id="strategy",
            state=state,
            evidence=_aged(evidence, age),
            evidence_age_s=float(age or 0.0),
            fix=None,
        ),
        ok,
    )


def readiness_rows(
    *,
    peers: Sequence[Any] = (),
    fabric_required: bool = True,
    link_status: Any = None,
    shared_link: Any = None,
    fabric_ok: bool | None = None,
    fabric_detail: str = "",
    staging: Mapping[str, Any] | None = None,
    staging_job_id: str = "",
    budgets: Sequence[Mapping[str, Any]] = (),
    required_bytes: int = 0,
    plan_error: str = "",
    strategies: Mapping[str, Any] | None = None,
    model_error: str = "",
    ages: Mapping[str, float] | None = None,
) -> ReadinessReport:
    """Compose the five precondition rows from already-collected evidence.

    ``peers`` are per-remote-host reachability records (mappings or
    ``PeerHealth``-likes with ``host``/``reachable``/``detail``);
    ``link_status``/``shared_link``/``fabric_ok`` are the B3 ladder evidence
    and ``_resolve_fabric``'s verdict; ``staging`` is a ``stage_manifest``
    result; ``budgets`` are B5's per-node ``/node-budgets`` payloads
    (``usable_bytes``/``reserve_bytes``/``role``/``binding``);
    ``strategies`` is B5's ``{tensor, pipeline}`` gating. ``ages`` maps row
    id → evidence age in seconds.
    """

    ages = ages or {}
    ssh_row, ssh_ok = _ssh_row(peers, ages.get("ssh"))
    fabric_row, fabric_ok_flag = _fabric_row(
        fabric_required=fabric_required,
        link_status=link_status,
        shared_link=shared_link,
        fabric_ok=fabric_ok,
        fabric_detail=fabric_detail,
        age=ages.get("fabric"),
    )
    staging_row, staging_ok = _staging_row(
        staging=staging,
        staging_job_id=staging_job_id,
        required_bytes=required_bytes,
        model_error=model_error,
        age=ages.get("staging"),
    )
    budget_row, budget_ok = _budget_row(
        budgets=budgets,
        required_bytes=required_bytes,
        plan_error=plan_error,
        model_error=model_error,
        age=ages.get("budget"),
    )
    strategy_row, strategy_ok = _strategy_row(
        strategies=strategies,
        model_error=model_error,
        age=ages.get("strategy"),
    )
    return ReadinessReport(
        ready=all((ssh_ok, fabric_ok_flag, staging_ok, budget_ok, strategy_ok)),
        rows=(ssh_row, fabric_row, staging_row, budget_row, strategy_row),
    )
