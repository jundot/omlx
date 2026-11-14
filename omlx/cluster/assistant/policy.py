# SPDX-License-Identifier: Apache-2.0
"""The resource policy the Cluster Assistant plans a model with.

The thin layer's planning defect was not a missing arithmetic rule: oMLX's own
planner (``plan_hybrid`` / ``plan_unequal_pipeline``) already models Engram/PLE
tables on SSD, routed-expert offload, the memory-guard tier, the context and the
microbatch. What was missing was the *decision* around it. The assistant's
election built every stage as if the whole checkpoint had to stay resident, at
one fixed context, under one fixed tier, and it treated every measured Mac as
one cluster. A model that the coordinator would have launched with its tables on
SSD was therefore refused before the coordinator was ever asked.

This module is that decision, and nothing else. It owns:

* ``ResourcePolicy`` — one resolved description of the levers a plan uses. It is
  carried through measure → recommend → plan → apply → start and is part of the
  plan signature, so a policy that drifts is a ``plan_mismatch`` rather than a
  silent re-plan.
* ``SavedPolicy`` — the saved settings a policy starts from. A user's explicit
  setting is the default; the solver only proposes a change, and the UI shows it
  and applies it on Start setup.
* ``solve_cluster_plan`` — the "fewest Macs that fit, then the best set and the
  least-invasive lever" solver. It is a loop around ``election.elect``, not a
  second planner: every feasibility verdict still comes from ``elect`` and
  ``plan_hybrid``. A subset whose summed admission cannot hold the model under
  any lever is refused before the context ladder is enumerated, and identical
  subproblems are memoized for the duration of one call — never across
  measurements. Both are cost cuts only: the selected plan, its levers and its
  signature are the ones the full enumeration would choose.
* ``context_ladder`` / ``largest_context_for`` — the context dimension of the
  search, shared with ``recommendation.py``.
* ``diverse_candidates`` — the best feasible plan for each cluster size
  (2..4 Macs), ranked by the election's own scoring, for the measurement
  search loop. It reuses the same solver machinery as ``solve_cluster_plan``
  and never changes what that solver elects: it only keeps every size's best
  instead of stopping at the fewest Macs that fit.

The solver is pure and deterministic: no HTTP, no SSH, no model loading, no
settings writes. The one filesystem access is ``moe_offload_layer_bytes``
reading a checkpoint's own headers with oMLX's existing helper; it is cached by
that helper and never estimates a number.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import combinations
from typing import Any

from ...patches.deepseek_v41.moe_offload import offload_layer_bytes
from ..catalogue import _CONTEXT_LADDER
from ..performance import ExecutionProfileName
from ..planner import ModelLayout, _held_layer_bytes
from .election import (
    MAX_CANDIDATE_NODES,
    ElectionResult,
    NodeFacts,
    NoFeasiblePlanError,
    PlanCandidate,
    budget_for,
    elect,
    rank_candidates,
)

GIB = 1024**3

# The smallest cluster ``ClusterDeployment`` accepts. A one-Mac plan is still a
# legal *answer* (the model fits here) but it is not applicable; apply refuses it
# with ``single_node_plan``. The solver starts at two when two Macs were measured.
MIN_CLUSTER_NODES = 2

# Context presets shared by the catalogue, recommendation and this solver.
MIN_CONTEXT_TOKENS = _CONTEXT_LADDER[0]

TIER_ORDER = {"safe": 0, "balanced": 1, "aggressive": 2, "custom": 1}
VALID_TIERS = frozenset(TIER_ORDER)

TABLE_OFFLOAD_ENGRAM = "engram"
TABLE_OFFLOAD_PLE = "ple"
TABLE_OFFLOAD_NONE = "none"
VALID_TABLE_OFFLOAD = frozenset(
    {TABLE_OFFLOAD_ENGRAM, TABLE_OFFLOAD_PLE, TABLE_OFFLOAD_NONE}
)

ADMISSION_P05 = "p05"
ADMISSION_P50 = "p50"
ADMISSION_P95 = "p95"
VALID_ADMISSION = frozenset({ADMISSION_P05, ADMISSION_P50, ADMISSION_P95})

NODE_SELECTION_MINIMAL = "minimal"
NODE_SELECTION_ALL = "all"
VALID_NODE_SELECTION = frozenset({NODE_SELECTION_MINIMAL, NODE_SELECTION_ALL})

# The MoE resident fractions the solver proposes, least-invasive first. A lower
# fraction frees more memory; the last one is the aggressive end of the ladder.
MOE_RESIDENT_LADDER = (0.75, 0.5, 0.25)

# A plan at this context or above without routed-expert offload is preferred over
# a larger-context plan that has to offload experts (decision 2026-11-14): losing
# context is a smaller cost than pushing routed experts to SSD, and 32k is the
# documented default target. Only when nothing at >= this floor fits without
# expert offload does the solver descend below it or allow expert offload.
PREFERRED_CONTEXT_FLOOR = 32768

# One warning per lever the plan applied, so Details can explain why the plan
# differs from the saved settings and from the full measured cluster.
WARNING_TABLE_OFFLOAD_ENABLED = "table_offload_enabled"
WARNING_MOE_OFFLOAD_ENABLED = "moe_offload_enabled"
WARNING_CONTEXT_REDUCED = "context_reduced"
WARNING_TIER_RAISED = "tier_raised"
WARNING_CHUNK_REDUCED = "chunk_reduced"
WARNING_MACS_NOT_NEEDED = "macs_not_needed"
WARNING_ADMISSION_OPTIMISTIC = "optimistic_admission"


class PolicyError(ValueError):
    """A policy that cannot be built from the settings and the model."""


@dataclass(frozen=True)
class SavedPolicy:
    """The saved settings a resource policy starts from.

    ``memory_guard_tier`` is ``None`` when the caller did not choose one; each
    selected Mac then plans against its own measured tier. The two table flags
    are kept separate because this checkout still stores DeepSeek Engram and
    Qwen4 PLE under different keys.
    """

    memory_guard_tier: str | None = None
    engram_ssd_offload: bool = False
    ple_ssd_offload: bool = False
    moe_expert_offload_enabled: bool = False
    moe_expert_offload_resident_fraction: float = 0.25
    max_context_window: int | None = None

    def __post_init__(self) -> None:
        if self.memory_guard_tier is not None:
            tier = self.memory_guard_tier.strip().lower()
            if tier not in VALID_TIERS:
                raise ValueError(f"unknown memory guard tier: {self.memory_guard_tier!r}")
            object.__setattr__(self, "memory_guard_tier", tier)
        if not 0 < self.moe_expert_offload_resident_fraction <= 1:
            raise ValueError("moe resident fraction must be in (0, 1]")
        if self.max_context_window is not None and self.max_context_window <= 0:
            raise ValueError("max_context_window must be positive when set")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> SavedPolicy:
        """Read the saved settings the measurement recorded.

        Unknown or malformed values fall back to the documented defaults rather
        than making a plan impossible: the settings file is user-editable, and a
        typo there must not look like "this model does not fit".
        """

        data = payload or {}
        tier = data.get("memory_guard_tier")
        if isinstance(tier, str) and tier.strip().lower() in VALID_TIERS:
            tier = tier.strip().lower()
        else:
            tier = None
        fraction = _fraction(
            data.get("moe_expert_offload_resident_fraction"), default=0.25
        )
        context = data.get("max_context_window")
        if not isinstance(context, int) or isinstance(context, bool) or context <= 0:
            context = None
        return cls(
            memory_guard_tier=tier,
            engram_ssd_offload=bool(data.get("deepseek_v41_engram_ssd_offload")),
            ple_ssd_offload=bool(data.get("qwen4_ple_ssd_offload")),
            moe_expert_offload_enabled=bool(data.get("moe_expert_offload_enabled")),
            moe_expert_offload_resident_fraction=fraction,
            max_context_window=context,
        )

    def table_offload_for(self, model: ModelLayout) -> str:
        """The table switch this checkpoint supports, if it was saved on."""

        kind = model.table_offload_kind
        if kind == TABLE_OFFLOAD_ENGRAM and self.engram_ssd_offload:
            return TABLE_OFFLOAD_ENGRAM
        if kind == TABLE_OFFLOAD_PLE and self.ple_ssd_offload:
            return TABLE_OFFLOAD_PLE
        return TABLE_OFFLOAD_NONE

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_guard_tier": self.memory_guard_tier,
            "engram_ssd_offload": self.engram_ssd_offload,
            "ple_ssd_offload": self.ple_ssd_offload,
            "moe_expert_offload_enabled": self.moe_expert_offload_enabled,
            "moe_expert_offload_resident_fraction": (
                self.moe_expert_offload_resident_fraction
            ),
            "max_context_window": self.max_context_window,
        }


@dataclass(frozen=True)
class ResourcePolicy:
    """One resolved plan policy: the levers, not a second planner.

    ``memory_guard_tier`` is ``None`` for "each selected Mac's own saved tier".
    ``context_tokens`` is the context the plan was actually built at, so a
    reduced context is visible in the signature and in the plan summary.
    """

    context_tokens: int
    table_offload: str = TABLE_OFFLOAD_NONE
    #: The checkpoint's own table kind (engram/ple/""). It lets the apply step
    #: write an explicit "keep it resident" when the plan chose no offload.
    table_kind: str = ""
    moe_expert_offload: bool = False
    moe_resident_fraction: float = 1.0
    memory_guard_tier: str | None = None
    microbatch_size: int = 1
    admission: str = ADMISSION_P50
    target_context_tokens: int = 0

    def __post_init__(self) -> None:
        if self.context_tokens <= 0:
            raise ValueError("context_tokens must be positive")
        if self.table_offload not in VALID_TABLE_OFFLOAD:
            raise ValueError(f"unknown table offload: {self.table_offload!r}")
        if self.table_kind not in ("", TABLE_OFFLOAD_ENGRAM, TABLE_OFFLOAD_PLE):
            raise ValueError(f"unknown table kind: {self.table_kind!r}")
        if self.memory_guard_tier is not None:
            tier = self.memory_guard_tier.strip().lower()
            if tier not in VALID_TIERS:
                raise ValueError(f"unknown memory guard tier: {self.memory_guard_tier!r}")
            object.__setattr__(self, "memory_guard_tier", tier)
        if not 0 < self.moe_resident_fraction <= 1:
            raise ValueError("moe resident fraction must be in (0, 1]")
        if self.microbatch_size < 1:
            raise ValueError("microbatch_size must be positive")
        if self.admission not in VALID_ADMISSION:
            raise ValueError(f"unknown admission percentile: {self.admission!r}")
        if self.target_context_tokens < 0:
            raise ValueError("target_context_tokens must be non-negative")

    @property
    def context_reduced(self) -> bool:
        return (
            self.target_context_tokens > 0
            and self.context_tokens < self.target_context_tokens
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_tokens": self.context_tokens,
            "target_context_tokens": self.target_context_tokens,
            "table_offload": self.table_offload,
            "table_kind": self.table_kind,
            "moe_expert_offload": self.moe_expert_offload,
            "moe_resident_fraction": self.moe_resident_fraction,
            "memory_guard_tier": self.memory_guard_tier,
            "microbatch_size": self.microbatch_size,
            "admission": self.admission,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> ResourcePolicy | None:
        if not payload:
            return None
        return cls(
            context_tokens=int(payload["context_tokens"]),
            target_context_tokens=int(payload.get("target_context_tokens") or 0),
            table_offload=str(payload.get("table_offload") or TABLE_OFFLOAD_NONE),
            table_kind=str(payload.get("table_kind") or ""),
            moe_expert_offload=bool(payload.get("moe_expert_offload", False)),
            moe_resident_fraction=_fraction(
                payload.get("moe_resident_fraction"), default=1.0
            ),
            memory_guard_tier=(
                None
                if payload.get("memory_guard_tier") in (None, "")
                else str(payload["memory_guard_tier"])
            ),
            microbatch_size=int(payload.get("microbatch_size") or 1),
            admission=str(payload.get("admission") or ADMISSION_P50),
        )


@dataclass(frozen=True)
class PolicyWarning:
    """One lever the plan applied, or one Mac it left out."""

    code: str
    message: str
    node_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "node_id": self.node_id, "message": self.message}


@dataclass(frozen=True)
class ExcludedNode:
    """A measured Mac the plan does not use, and why."""

    node_id: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"node_id": self.node_id, "reason": self.reason}


@dataclass(frozen=True)
class SizeRefusal:
    """Why a cluster of this size could not run the model."""

    node_count: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"node_count": self.node_count, "reason": self.reason}


@dataclass(frozen=True)
class ClusterPlanSelection:
    """The elected plan plus the policy and subset that produced it."""

    election: ElectionResult
    policy: ResourcePolicy
    excluded_nodes: tuple[ExcludedNode, ...] = ()
    warnings: tuple[PolicyWarning, ...] = ()
    size_refusals: tuple[SizeRefusal, ...] = ()

    @property
    def feasible(self) -> bool:
        return self.election.feasible

    @property
    def selected_node_ids(self) -> tuple[str, ...]:
        return self.election.rank_order

    def to_dict(self) -> dict[str, Any]:
        return {
            "resource_policy": self.policy.to_dict(),
            "excluded_nodes": [item.to_dict() for item in self.excluded_nodes],
            "warnings": [item.to_dict() for item in self.warnings],
            "size_refusals": [item.to_dict() for item in self.size_refusals],
        }


@dataclass(frozen=True)
class _Lever:
    """One policy variant before the context is chosen."""

    table_offload: str
    moe_expert_offload: bool
    moe_resident_fraction: float
    memory_guard_tier: str | None
    admission: str
    intervention_rank: int


class _SolverMemo:
    """Per-call caches for one ``solve_cluster_plan`` invocation.

    The election is a pure function of its inputs, so an identical subproblem
    may be answered from its first evaluation. The caches live and die with
    one solver call: nothing leaks across measurements, and a new measurement
    always re-elects from its own facts. Within one call the node IDs uniquely
    identify the measured facts (the solver validates that), and the caller
    holds the model object throughout, so ``id(model)`` is a safe identity key.
    """

    def __init__(self) -> None:
        self._moe: dict[tuple[int, float, bool], tuple[int, ...]] = {}
        self._elections: dict[
            tuple[Any, ...], ElectionResult | NoFeasiblePlanError
        ] = {}

    def moe_bytes(
        self, model: ModelLayout, fraction: float, *, mtp: bool
    ) -> tuple[int, ...]:
        """The checkpoint's own offload bytes, read once per (fraction, mtp).

        The helper resolves the checkpoint's headers on every call; the memo
        turns the repeated reads of one solve into one. The module-level name
        is resolved at call time so a test fake still applies.
        """

        key = (id(model), fraction, mtp)
        if key not in self._moe:
            self._moe[key] = moe_offload_layer_bytes(model, fraction, mtp=mtp)
        return self._moe[key]

    def election(
        self, key: tuple[Any, ...], compute: Callable[[], ElectionResult]
    ) -> ElectionResult:
        """The cached verdict for one subproblem, or the first evaluation.

        A refusal is cached too: the election is deterministic, so re-asking
        an identical subproblem would raise the identical error.
        """

        cached = self._elections.get(key)
        if cached is None:
            try:
                cached = compute()
            except NoFeasiblePlanError as exc:
                cached = exc
            self._elections[key] = cached
        if isinstance(cached, NoFeasiblePlanError):
            raise cached
        return cached


def context_ladder(
    *, target: int | None, declared: int | None, floor: int = MIN_CONTEXT_TOKENS
) -> list[int]:
    """The contexts this question may answer with, ascending.

    The target, the model's own declared limit and the documented floor all cap
    the answer. A non-preset limit is added so the answer can be the model's own
    number rather than the next preset above it.
    """

    ceiling: int | None = None
    for value in (target, declared):
        if value is not None and value > 0:
            ceiling = value if ceiling is None else min(ceiling, value)
    ladder = [
        tokens
        for tokens in _CONTEXT_LADDER
        if tokens >= floor and (ceiling is None or tokens <= ceiling)
    ]
    if ceiling is not None:
        ladder.append(min(ceiling, _CONTEXT_LADDER[-1]))
    ladder = sorted(set(ladder))
    if not ladder:
        ladder = [max(1, ceiling or floor)]
    return ladder


def moe_offload_layer_bytes(
    model: ModelLayout,
    fraction: float,
    *,
    mtp: bool = False,
) -> tuple[int, ...]:
    """Per-layer routed-expert bytes oMLX would read from SSD.

    Uses oMLX's own helper, so the election and the coordinator's plan cannot
    disagree about what the loader will hold. Empty means "this checkpoint is
    not an offload-capable MoE" — never an estimate of zero.
    """

    if not 0 < fraction <= 1:
        raise ValueError("moe resident fraction must be in (0, 1]")
    source = (model.source or "").strip()
    if not source or source == "synthetic":
        return ()
    try:
        from ...patches.moe_offload_compat import moe_offload_compatibility

        supported, _reason = moe_offload_compatibility(source)
        if not supported:
            return ()
        import json
        from pathlib import Path

        config = json.loads((Path(source) / "config.json").read_text())
        if str(config.get("model_type", "")).lower() != "deepseek_v41":
            return ()
        layer_bytes = offload_layer_bytes(source, float(fraction), mtp_resident=mtp)
    except Exception:
        # A checkpoint whose expert layout cannot be read is not offloadable.
        # The solver then keeps the resident policy and the near miss says the
        # model does not fit; it never invents a saving.
        return ()
    if len(layer_bytes) != model.layer_count:
        return ()
    return tuple(layer_bytes)


def model_supports_moe_offload(model: ModelLayout) -> bool:
    return bool(moe_offload_layer_bytes(model, 0.25))


def policy_model_settings(policy: ResourcePolicy) -> dict[str, Any]:
    """The per-model settings write that makes this policy the launched one.

    Only levers the policy enables are written. A saved ``false`` is not
    overwritten with another ``false``, and a switch the checkpoint does not
    have is never touched. The coordinator reads these values back, so the plan
    it re-derives is the plan the election approved.
    """

    patch: dict[str, Any] = {}
    if policy.table_offload == TABLE_OFFLOAD_ENGRAM:
        patch["deepseek_v41_engram_ssd_offload"] = True
    elif policy.table_offload == TABLE_OFFLOAD_PLE:
        patch["qwen4_ple_ssd_offload"] = True
    elif policy.table_kind == TABLE_OFFLOAD_ENGRAM:
        # The plan explicitly keeps this checkpoint's table resident. Writing
        # the switch is what makes the launched process match the plan; it is
        # shown in the plan and confirmed by Start setup, never silent.
        patch["deepseek_v41_engram_ssd_offload"] = False
    elif policy.table_kind == TABLE_OFFLOAD_PLE:
        patch["qwen4_ple_ssd_offload"] = False
    if policy.moe_expert_offload:
        patch["moe_expert_offload_enabled"] = True
        patch["moe_expert_offload_resident_fraction"] = (
            policy.moe_resident_fraction
        )
    return patch


def policy_warnings(
    policy: ResourcePolicy,
    saved: SavedPolicy,
    *,
    saved_table: str,
    selected: Sequence[str],
    measured: Sequence[str],
    target_context_tokens: int,
    saved_microbatch_size: int,
) -> tuple[PolicyWarning, ...]:
    """One warning per lever the plan applied, plus ``macs_not_needed``."""

    warnings: list[PolicyWarning] = []
    if policy.table_offload != TABLE_OFFLOAD_NONE and policy.table_offload != saved_table:
        label = "Engram" if policy.table_offload == TABLE_OFFLOAD_ENGRAM else "PLE"
        warnings.append(
            PolicyWarning(
                code=WARNING_TABLE_OFFLOAD_ENABLED,
                message=(
                    f"the {label} table is kept on SSD for this plan; the saved "
                    "setting was off. Start setup writes it before staging"
                ),
            )
        )
    if policy.moe_expert_offload and not saved.moe_expert_offload_enabled:
        warnings.append(
            PolicyWarning(
                code=WARNING_MOE_OFFLOAD_ENABLED,
                message=(
                    "routed experts are streamed from SSD with "
                    f"{policy.moe_resident_fraction:.0%} resident; the saved "
                    "setting was off. Start setup writes it before staging"
                ),
            )
        )
    if policy.context_tokens < target_context_tokens:
        warnings.append(
            PolicyWarning(
                code=WARNING_CONTEXT_REDUCED,
                message=(
                    f"the context is {policy.context_tokens:,} tokens, reduced "
                    f"from {target_context_tokens:,} so the model fits these Macs"
                ),
            )
        )
    saved_tier = (saved.memory_guard_tier or "balanced").lower()
    chosen_tier = (policy.memory_guard_tier or saved_tier).lower()
    if TIER_ORDER.get(chosen_tier, 1) > TIER_ORDER.get(saved_tier, 1):
        warnings.append(
            PolicyWarning(
                code=WARNING_TIER_RAISED,
                message=(
                    f"the memory tier is {chosen_tier}, raised from {saved_tier}; "
                    "Start setup launches with it"
                ),
            )
        )
    if policy.microbatch_size < saved_microbatch_size:
        warnings.append(
            PolicyWarning(
                code=WARNING_CHUNK_REDUCED,
                message=(
                    f"the pipeline microbatch is {policy.microbatch_size}, "
                    f"reduced from {saved_microbatch_size}"
                ),
            )
        )
    if policy.admission == ADMISSION_P95:
        warnings.append(
            PolicyWarning(
                code=WARNING_ADMISSION_OPTIMISTIC,
                message=(
                    "this plan is sized against the p95 admission ceiling (the "
                    "optimistic end of what these Macs measured); the load can "
                    "still be refused under real pressure"
                ),
            )
        )
    extra = [node_id for node_id in measured if node_id not in selected]
    if extra:
        warnings.append(
            PolicyWarning(
                code=WARNING_MACS_NOT_NEEDED,
                node_id=",".join(extra),
                message=(
                    "not needed: the selected "
                    f"{len(selected)} Macs fit this model at "
                    f"{policy.context_tokens:,} tokens"
                ),
            )
        )
    return tuple(warnings)


def solve_cluster_plan(
    nodes: Sequence[NodeFacts],
    model: ModelLayout,
    *,
    target_context_tokens: int,
    rank0_extra_bytes: int = 0,
    mtp: bool = False,
    execution_profile: ExecutionProfileName = "balanced",
    microbatch_size: int = 1,
    saved: SavedPolicy | None = None,
    policy: ResourcePolicy | None = None,
    saved_baseline: SavedPolicy | None = None,
    forced_node_ids: Sequence[str] | None = None,
    node_selection: str = NODE_SELECTION_MINIMAL,
    allow_optimistic_admission: bool = False,
    origin_node: str | None = None,
    fabric_links: Sequence[Sequence[str]] | None = None,
    max_size: int | None = None,
    two_mac_fallback: bool = False,
    exclude_expert_offload: bool = False,
) -> ClusterPlanSelection:
    """Bring this model up on the fewest Macs that fit.

    ``policy`` set means "re-prove exactly this policy" (apply/start);
    ``forced_node_ids`` restricts the Macs to the approved subset. Otherwise the
    solver enumerates subset sizes from two upward, then the lever ladder, and
    stops at the first size that fits. Extra Macs are reported as
    ``excluded_nodes``; they are never added "because they are there".

    Two cost cuts keep the enumeration cheap without changing its answer: a
    subset whose summed admission cannot hold the model under any lever is
    refused before its ladder is descended (with the same refusal the descent
    would end on), and identical subproblems are memoized for this call only.

    ``fabric_links`` is the measured line: when given, a rank order is only
    electable if every consecutive pair has a link. ``max_size`` caps the
    minimal search (the offered 2-Mac fallback uses it). ``two_mac_fallback``
    records the user's explicit consent in the plan signature.
    """

    node_list = list(nodes)
    if not node_list:
        raise ValueError("at least one node is required")
    node_ids = [facts.node_id for facts in node_list]
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("node IDs must be unique")
    if target_context_tokens <= 0:
        raise ValueError("target_context_tokens must be positive")
    if microbatch_size < 1:
        raise ValueError("microbatch_size must be positive")
    saved = saved or SavedPolicy()
    baseline = saved_baseline or saved
    if node_selection not in VALID_NODE_SELECTION:
        raise ValueError(f"unknown node selection: {node_selection!r}")

    facts_by_id = {facts.node_id: facts for facts in node_list}

    # The MVP plans pipeline parallelism only. mlx-lm cannot split an
    # architecture that does not implement the pipeline forward path, and the
    # coordinator's own ``/plan`` refuses it — so the assistant must refuse it
    # here instead of returning a feasible plan that can never stage. A
    # synthetic pre-download layout has no architecture flag and is exempt, as
    # it is in the coordinator.
    if (
        len(node_list) > 1
        and (model.source or "") != "synthetic"
        and not model.supports_pipeline
    ):
        raise NoFeasiblePlanError(
            f"{model.source or 'this checkpoint'} cannot be split into pipeline "
            "stages: mlx-lm does not implement the pipeline forward path for "
            "this architecture, and the Cluster Assistant plans pipeline "
            "parallelism only. Choose a pipeline-capable model",
            _empty_result(node_list, origin_node),
            action="choose_pipeline_model",
        )

    if policy is not None:
        selection_nodes = _selected_facts(node_list, forced_node_ids)
        result = _elect_one(
            selection_nodes,
            model,
            policy=policy,
            rank0_extra_bytes=rank0_extra_bytes,
            mtp=mtp,
            execution_profile=execution_profile,
            origin_node=origin_node,
            fabric_links=fabric_links,
            two_mac_fallback=two_mac_fallback,
        )
        excluded = _excluded(selection_nodes, node_list, policy, result)
        warnings = policy_warnings(
            policy,
            baseline,
            saved_table=baseline.table_offload_for(model),
            selected=result.rank_order,
            measured=node_ids,
            target_context_tokens=policy.target_context_tokens
            or target_context_tokens,
            saved_microbatch_size=microbatch_size,
        )
        return ClusterPlanSelection(
            election=result,
            policy=policy,
            excluded_nodes=excluded,
            warnings=warnings,
        )

    candidates = (
        _selected_facts(node_list, forced_node_ids)
        if forced_node_ids
        else node_list
    )
    if node_selection == NODE_SELECTION_ALL or forced_node_ids:
        sizes = [len(candidates)] if len(candidates) >= MIN_CLUSTER_NODES else [1]
    else:
        largest = len(candidates)
        if max_size is not None:
            largest = min(largest, max_size)
        sizes = list(range(MIN_CLUSTER_NODES, largest + 1)) or [1]

    memo = _SolverMemo()
    levers = _levers(
        saved,
        model,
        allow_optimistic_admission,
        memo=memo,
        exclude_expert_offload=exclude_expert_offload,
    )
    ladder = context_ladder(
        target=target_context_tokens,
        declared=_declared_context(model, saved),
    )
    refusals: list[SizeRefusal] = []
    last_error: NoFeasiblePlanError | None = None

    for size in sizes:
        best, size_error = _best_for_size(
            candidates,
            size,
            model,
            ladder=ladder,
            levers=levers,
            rank0_extra_bytes=rank0_extra_bytes,
            mtp=mtp,
            execution_profile=execution_profile,
            origin_node=origin_node,
            fabric_links=fabric_links,
            two_mac_fallback=two_mac_fallback,
            microbatch_size=microbatch_size,
            memo=memo,
        )
        if size_error is not None:
            last_error = size_error
        if best is not None:
            chosen = best[3]
            selected = tuple(best[2].rank_order)
            return ClusterPlanSelection(
                election=best[2],
                policy=chosen,
                excluded_nodes=_excluded(
                    [facts_by_id[node_id] for node_id in selected],
                    node_list,
                    chosen,
                    best[2],
                ),
                warnings=policy_warnings(
                    chosen,
                    baseline,
                    saved_table=baseline.table_offload_for(model),
                    selected=selected,
                    measured=node_ids,
                    target_context_tokens=target_context_tokens,
                    saved_microbatch_size=microbatch_size,
                ),
                size_refusals=tuple(refusals),
            )
        refusals.append(
            SizeRefusal(
                node_count=size,
                reason=_refusal_reason(size_error),
            )
        )

    if last_error is not None:
        raise NoFeasiblePlanError(
            _combined_refusal(refusals, last_error),
            last_error.result,
        ) from last_error
    # No subset produced an election at all (one measured Mac that cannot hold
    # the model, for instance). Ask the election for its own refusal so the
    # caller gets the same plan-shaped answer as everywhere else.
    fallback_context = max(ladder)
    fallback = _elect_one(
        candidates,
        model,
        policy=_default_policy(
            model,
            saved,
            fallback_context,
            microbatch_size=microbatch_size,
        ),
        rank0_extra_bytes=rank0_extra_bytes,
        mtp=mtp,
        execution_profile=execution_profile,
        origin_node=origin_node,
        fabric_links=fabric_links,
        two_mac_fallback=two_mac_fallback,
    )
    return ClusterPlanSelection(
        election=fallback,
        policy=_default_policy(
            model, saved, fallback_context, microbatch_size=microbatch_size
        ),
        excluded_nodes=(),
    )


def diverse_candidates(
    nodes: Sequence[NodeFacts],
    model: ModelLayout,
    *,
    target_context_tokens: int,
    rank0_extra_bytes: int = 0,
    mtp: bool = False,
    execution_profile: ExecutionProfileName = "balanced",
    microbatch_size: int = 1,
    saved: SavedPolicy | None = None,
    fabric_links: Sequence[Sequence[str]] | None = None,
    allow_optimistic_admission: bool = False,
    max_count: int = 3,
) -> list[PlanCandidate]:
    """The best deployable plan per cluster size, for the measurement search.

    The search loop needs DIVERSE constellations to measure: the best feasible
    plan for 2 Macs, for 3, for 4 — a different node count is the fewer-nodes
    vs more-context tradeoff. ``solve_cluster_plan`` stops at the fewest Macs
    that fit and returns only that one; this asks the same solver machinery for
    every size from ``MIN_CLUSTER_NODES`` up to ``min(len(nodes), 4)`` and keeps
    each size's best, so no size is silently skipped just because a smaller one
    already fit.

    Pure and additive: it reuses ``_best_for_size`` / ``_plans_for_subset`` /
    ``elect`` / ``plan_hybrid`` (the planner still owns every layer split) and
    changes nothing about what ``solve_cluster_plan`` or ``elect`` return. Each
    returned candidate carries its ``shard_plan`` and ``plan_hash`` so a caller
    can deploy it. Sizes with no feasible plan are skipped; a one-Mac "plan" is
    never a cluster and is never returned. The result is ranked best-first by
    the election's own scoring (``rank_candidates``) and capped at ``max_count``.

    ``fabric_links`` is respected exactly as in the election: a rank order is
    only feasible when every consecutive pair has a measured link. An
    infeasible model (or one mlx-lm cannot pipeline) yields an empty list, not
    an error — the search simply has nothing worth measuring.
    """

    node_list = list(nodes)
    if not node_list:
        raise ValueError("at least one node is required")
    node_ids = [facts.node_id for facts in node_list]
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("node IDs must be unique")
    if target_context_tokens <= 0:
        raise ValueError("target_context_tokens must be positive")
    if microbatch_size < 1:
        raise ValueError("microbatch_size must be positive")
    if max_count <= 0:
        return []
    saved = saved or SavedPolicy()

    # A checkpoint mlx-lm cannot pipeline never yields a deployable multi-Mac
    # candidate (the solver refuses it with ``choose_pipeline_model``); the
    # search gets an empty list rather than a plan that can never stage.
    if (
        len(node_list) > 1
        and (model.source or "") != "synthetic"
        and not model.supports_pipeline
    ):
        return []

    largest = min(len(node_list), MAX_CANDIDATE_NODES)
    if largest < MIN_CLUSTER_NODES:
        return []

    memo = _SolverMemo()
    levers = _levers(
        saved,
        model,
        allow_optimistic_admission,
        memo=memo,
    )
    ladder = context_ladder(
        target=target_context_tokens,
        declared=_declared_context(model, saved),
    )

    bests: list[PlanCandidate] = []
    for size in range(MIN_CLUSTER_NODES, largest + 1):
        best, _size_error = _best_for_size(
            node_list,
            size,
            model,
            ladder=ladder,
            levers=levers,
            rank0_extra_bytes=rank0_extra_bytes,
            mtp=mtp,
            execution_profile=execution_profile,
            origin_node=None,
            fabric_links=fabric_links,
            two_mac_fallback=False,
            microbatch_size=microbatch_size,
            memo=memo,
        )
        if best is None:
            continue
        candidate = best[2].selected
        if candidate is not None and candidate.feasible:
            # Carry the resolved policy so the search loop's plan_apply
            # re-elects the identical signature (best[3] is the ResourcePolicy).
            bests.append(replace(candidate, resource_policy=best[3]))

    return list(rank_candidates(bests, node_list)[:max_count])


def _best_for_size(
    candidates: Sequence[NodeFacts],
    size: int,
    model: ModelLayout,
    *,
    ladder: Sequence[int],
    levers: Sequence[_Lever],
    rank0_extra_bytes: int,
    mtp: bool,
    execution_profile: ExecutionProfileName,
    origin_node: str | None,
    fabric_links: Sequence[Sequence[str]] | None,
    two_mac_fallback: bool,
    microbatch_size: int,
    memo: _SolverMemo,
) -> tuple[
    tuple[int, int, ElectionResult, ResourcePolicy] | None,
    NoFeasiblePlanError | None,
]:
    """The best plan for exactly ``size`` Macs, plus the last refusal if none.

    Enumerates every subset of this size and keeps the best option under the
    solver's context/lever/election key (``_best_option``). ``solve_cluster_plan``
    stops at the first size that returns a plan (fewest Macs that fit);
    ``diverse_candidates`` asks every size and keeps each best. The feasibility
    verdicts still come from ``_plans_for_subset`` / ``elect`` / ``plan_hybrid``;
    this only enumerates subsets and never re-derives a split. The returned
    error is the last ``NoFeasiblePlanError`` any subset raised, so the caller
    can reproduce the solver's combined refusal unchanged.
    """

    options: list[tuple[int, int, ElectionResult, ResourcePolicy]] = []
    size_error: NoFeasiblePlanError | None = None
    for subset in combinations(candidates, size):
        subset = tuple(sorted(subset, key=lambda facts: facts.node_id))
        try:
            for option in _plans_for_subset(
                subset,
                model,
                ladder=ladder,
                levers=levers,
                rank0_extra_bytes=rank0_extra_bytes,
                mtp=mtp,
                execution_profile=execution_profile,
                origin_node=origin_node,
                fabric_links=fabric_links,
                two_mac_fallback=two_mac_fallback,
                microbatch_size=microbatch_size,
                memo=memo,
            ):
                options.append(option)
        except NoFeasiblePlanError as exc:
            size_error = exc
            continue
    if options:
        return _best_option(options), size_error
    return None, size_error


def _plans_for_subset(
    subset: Sequence[NodeFacts],
    model: ModelLayout,
    *,
    ladder: Sequence[int],
    levers: Sequence[_Lever],
    rank0_extra_bytes: int,
    mtp: bool,
    execution_profile: ExecutionProfileName,
    origin_node: str | None,
    fabric_links: Sequence[Sequence[str]] | None = None,
    two_mac_fallback: bool = False,
    microbatch_size: int = 1,
    memo: _SolverMemo | None = None,
) -> list[tuple[int, int, ElectionResult, ResourcePolicy]]:
    """Every feasible plan for this subset, best context first.

    Context is quality: a plan that keeps the requested context is better than
    one that drops to 8k, even if the latter needs a more invasive lever. Within
    one context the least-invasive lever wins, then the election's own scoring.

    The ladder is descended linearly, largest first, stopping at the first
    context where any lever fits. Before that descent, a subset that cannot
    hold the model under *any* lever even at the smallest context is refused
    early (``_subset_cannot_fit``); the refusal is the election's own verdict
    at the descent's last evaluation point, so the message and the per-Mac
    payload are exactly what the full descent would end on.
    """

    if memo is not None and _subset_cannot_fit(
        subset,
        model,
        levers=levers,
        ladder=ladder,
        rank0_extra_bytes=rank0_extra_bytes,
        mtp=mtp,
        memo=memo,
    ):
        # The linear descent evaluates contexts largest-first and levers in
        # ladder order, so its final evaluation — the error it would raise —
        # is the smallest context with the last lever. Ask exactly that
        # question and raise its verdict; skipping the other evaluations
        # cannot change the refusal.
        canonical = _policy_for(
            model,
            levers[-1],
            min(ladder),
            target=0,
            microbatch_size=microbatch_size,
        )
        try:
            _elect_one(
                subset,
                model,
                policy=canonical,
                rank0_extra_bytes=rank0_extra_bytes,
                mtp=mtp,
                execution_profile=execution_profile,
                origin_node=origin_node,
                fabric_links=fabric_links,
                two_mac_fallback=two_mac_fallback,
                memo=memo,
            )
        except NoFeasiblePlanError as exc:
            raise exc
        # The planner found a plan the bound called impossible: trust the
        # planner, never the bound, and run the full descent.

    plans: list[tuple[int, int, ElectionResult, ResourcePolicy]] = []
    last_error: NoFeasiblePlanError | None = None
    descending = sorted(ladder, reverse=True)

    def descend(
        contexts: Sequence[int], lever_set: Sequence[_Lever]
    ) -> list[tuple[int, int, ElectionResult, ResourcePolicy]]:
        nonlocal last_error
        found: list[tuple[int, int, ElectionResult, ResourcePolicy]] = []
        for context_tokens in contexts:
            for lever in lever_set:
                policy = _policy_for(
                    model,
                    lever,
                    context_tokens,
                    target=0,
                    microbatch_size=microbatch_size,
                )
                try:
                    result = _elect_one(
                        subset,
                        model,
                        policy=policy,
                        rank0_extra_bytes=rank0_extra_bytes,
                        mtp=mtp,
                        execution_profile=execution_profile,
                        origin_node=origin_node,
                        fabric_links=fabric_links,
                        two_mac_fallback=two_mac_fallback,
                        memo=memo,
                    )
                except NoFeasiblePlanError as exc:
                    last_error = exc
                    continue
                found.append(
                    (
                        context_tokens,
                        lever.intervention_rank,
                        result,
                        replace(
                            policy,
                            target_context_tokens=max(ladder),
                        ),
                    )
                )
            if found:
                # A larger context that fits makes every smaller context worse
                # on the quality-first key; skip the rest of this pass.
                break
        return found

    # Quality first (decision 2026-11-14): keep the largest fitting context, even
    # if that needs routed-expert offload. Reducing context silently would trade
    # quality for speed — that is the user's choice, surfaced as an A/B option in
    # the plan, never made here. Lookup tables are already offloaded by default
    # in `_levers` (no speed cost, frees RAM for experts).
    plans = descend(descending, levers)
    if not plans:
        if last_error is not None:
            raise last_error
        raise NoFeasiblePlanError(
            f"no legal plan on {len(subset)} Macs at any context from "
            f"{min(ladder):,} to {max(ladder):,} tokens",
            _empty_result(subset, origin_node),
        )
    return plans


def _subset_cannot_fit(
    subset: Sequence[NodeFacts],
    model: ModelLayout,
    *,
    levers: Sequence[_Lever],
    ladder: Sequence[int],
    rank0_extra_bytes: int,
    mtp: bool,
    memo: _SolverMemo,
) -> bool:
    """Whether no lever can fit this subset at any ladder context.

    A necessary condition, never a feasibility verdict: every pipeline plan
    holds the replicated fixed weights on each Mac, the resident layer weights,
    the KV cache for the planned context, and the rank-0 head — so their sum
    must fit the subset's summed admission. Each lever is checked against its
    own best admission (``budget_for`` at rank 1, which only drops the rank-0
    margin) and its own resident bytes (the planner's ``_held_layer_bytes``,
    so the arithmetic cannot drift from the planner's), at the smallest
    context, where KV is cheapest. A lever whose sum still exceeds the summed
    admission cannot plan at any larger context or in any rank order; when
    every lever is ruled out this way, the subset is skipped. Per-node limits,
    contiguity and stage starts are deliberately *not* checked here — ignoring
    them only makes the bound weaker, never wrong. The refusal itself still
    comes from ``elect``.
    """

    min_context = min(ladder)
    kv_total = (
        model.kv_bytes_per_token_per_layer * model.layer_count * min_context
        if model.kv_bytes_per_token_per_layer > 0
        else 0
    )
    extra = rank0_extra_bytes + (model.mtp_weight_bytes if mtp else 0)
    fixed_total = len(subset) * model.fixed_weight_bytes
    for lever in levers:
        moe_bytes = (
            memo.moe_bytes(model, lever.moe_resident_fraction, mtp=mtp)
            if lever.moe_expert_offload
            else ()
        )
        held = _held_layer_bytes(
            model,
            lever.table_offload != TABLE_OFFLOAD_NONE,
            moe_offload_layer_bytes=moe_bytes,
        )
        admission_total = sum(
            budget_for(
                facts,
                rank=1,
                tier=lever.memory_guard_tier,
                admission=lever.admission,
            ).usable_bytes
            for facts in subset
        )
        if fixed_total + sum(held) + kv_total + extra <= admission_total:
            return False
    return True


def _best_option(
    options: Sequence[tuple[int, int, ElectionResult, ResourcePolicy]],
) -> tuple[int, int, ElectionResult, ResourcePolicy]:
    """Best at the minimal subset size: context, lever, then the election score.

    The election already ordered its own candidates; ``considered`` is best
    first. Adding the context and the intervention rank ahead of it keeps
    "fewest changes to what the user saved" true without inventing a speed
    ranking when no compute was measured.
    """

    def key(item: tuple[int, int, ElectionResult, ResourcePolicy]) -> tuple[Any, ...]:
        context_tokens, lever_rank, result, _policy = item
        selected = result.selected
        critical = (
            float("inf")
            if selected is None or selected.critical_stage_seconds is None
            else selected.critical_stage_seconds
        )
        headroom = 0 if selected is None else selected.min_headroom_bytes
        return (
            -context_tokens,
            lever_rank,
            critical,
            -headroom,
            len(result.rank_order),
            result.rank_order,
        )

    return min(options, key=key)


def _elect_one(
    nodes: Sequence[NodeFacts],
    model: ModelLayout,
    *,
    policy: ResourcePolicy,
    rank0_extra_bytes: int,
    mtp: bool,
    execution_profile: ExecutionProfileName,
    origin_node: str | None,
    fabric_links: Sequence[Sequence[str]] | None = None,
    two_mac_fallback: bool = False,
    memo: _SolverMemo | None = None,
) -> ElectionResult:
    """One subset, one policy: still the election's verdict, never a copy."""

    moe_bytes: tuple[int, ...] = ()
    if policy.moe_expert_offload:
        moe_bytes = (
            memo.moe_bytes(model, policy.moe_resident_fraction, mtp=mtp)
            if memo is not None
            else moe_offload_layer_bytes(
                model, policy.moe_resident_fraction, mtp=mtp
            )
        )
        if not moe_bytes:
            raise NoFeasiblePlanError(
                "this checkpoint does not support routed-expert offload; the "
                "plan cannot use it",
                _empty_result(nodes, origin_node),
            )

    def evaluate() -> ElectionResult:
        return elect(
            nodes,
            model,
            context_tokens=policy.context_tokens,
            rank0_extra_bytes=rank0_extra_bytes,
            mtp=mtp,
            workload_profile=execution_profile,
            microbatch_size=policy.microbatch_size,
            tensor_parallel_size=1,
            origin_node=origin_node,
            memory_guard_tier=policy.memory_guard_tier,
            admission=policy.admission,
            table_offload=policy.table_offload,
            moe_offload_layer_bytes=moe_bytes,
            moe_expert_offload_resident_fraction=(
                policy.moe_resident_fraction if policy.moe_expert_offload else 0.0
            ),
            fabric_links=fabric_links,
            two_mac_fallback=two_mac_fallback,
        )

    if memo is None:
        return evaluate()
    # Every input the election reads, in one key. The node order matters (it
    # orders the candidates), the policy is frozen and hashable, and within
    # one solver call a node ID identifies exactly one measured facts object.
    key = (
        tuple(facts.node_id for facts in nodes),
        id(model),
        policy,
        rank0_extra_bytes,
        mtp,
        execution_profile,
        origin_node,
        None
        if fabric_links is None
        else tuple(tuple(pair) for pair in fabric_links),
        two_mac_fallback,
    )
    return memo.election(key, evaluate)


def _levers(
    saved: SavedPolicy,
    model: ModelLayout,
    allow_optimistic: bool,
    memo: _SolverMemo | None = None,
    *,
    exclude_expert_offload: bool = False,
) -> tuple[_Lever, ...]:
    """The lever ladder, least-invasive first.

    Saved settings → the table switch the checkpoint supports → routed-expert
    offload at 75/50/25 % resident → a higher memory tier → the optimistic p95
    admission (only with explicit consent). Context is handled by the caller,
    ahead of the lever rank, because losing context is a quality loss.

    ``exclude_expert_offload`` drops the routed-expert rungs, so a caller can ask
    "what is the best plan that keeps experts resident?" — the speed side of the
    context-vs-experts tradeoff the user chooses between.
    """

    # One support probe per solve: the answer is a property of the checkpoint,
    # and the memo keeps the ladder from re-reading its headers per rung.
    supports_moe = (
        bool(memo.moe_bytes(model, 0.25, mtp=False))
        if memo is not None
        else model_supports_moe_offload(model)
    )
    saved_table = saved.table_offload_for(model)
    # Decision 2026-11-14: offload a checkpoint's lookup table (Engram/PLE) by
    # default. Keeping that table on SSD frees RAM for routed experts, so the
    # solver does not have to push experts to SSD; the plan shows the lever and
    # Start writes it, so it is never silent. A checkpoint without such a table
    # is unaffected (kind "" -> none). This is the saved choice when it is
    # already on, otherwise it default-offloads the table the model supports.
    default_table = model.table_offload_kind or saved_table
    levers: list[_Lever] = [
        _Lever(
            table_offload=default_table,
            moe_expert_offload=saved.moe_expert_offload_enabled and supports_moe,
            moe_resident_fraction=saved.moe_expert_offload_resident_fraction,
            memory_guard_tier=saved.memory_guard_tier,
            admission=ADMISSION_P50,
            intervention_rank=0,
        )
    ]
    if supports_moe and not exclude_expert_offload:
        fractions = list(MOE_RESIDENT_LADDER)
        if (
            saved.moe_expert_offload_enabled
            and saved.moe_expert_offload_resident_fraction not in fractions
        ):
            fractions.insert(0, saved.moe_expert_offload_resident_fraction)
        for index, fraction in enumerate(fractions):
            if (
                saved.moe_expert_offload_enabled
                and saved.moe_expert_offload_resident_fraction == fraction
            ):
                continue
            levers.append(
                _Lever(
                    table_offload=saved_table,
                    moe_expert_offload=True,
                    moe_resident_fraction=fraction,
                    memory_guard_tier=saved.memory_guard_tier,
                    admission=ADMISSION_P50,
                    intervention_rank=2 + index,
                )
            )
    base_rank = 2 + len(MOE_RESIDENT_LADDER)
    if (saved.memory_guard_tier or "balanced") != "aggressive":
        levers.append(
            _Lever(
                table_offload=saved_table,
                moe_expert_offload=saved.moe_expert_offload_enabled and supports_moe,
                moe_resident_fraction=saved.moe_expert_offload_resident_fraction,
                memory_guard_tier="aggressive",
                admission=ADMISSION_P50,
                intervention_rank=base_rank,
            )
        )
    if allow_optimistic:
        levers.append(
            _Lever(
                table_offload=saved_table,
                moe_expert_offload=saved.moe_expert_offload_enabled and supports_moe,
                moe_resident_fraction=saved.moe_expert_offload_resident_fraction,
                memory_guard_tier=saved.memory_guard_tier,
                admission=ADMISSION_P95,
                intervention_rank=base_rank + 1,
            )
        )
    # De-duplicate while preserving order: a saved offload at 0.25 must not be
    # proposed again by the ladder.
    unique: list[_Lever] = []
    seen: set[tuple[Any, ...]] = set()
    for lever in levers:
        key = (
            lever.table_offload,
            lever.moe_expert_offload,
            round(lever.moe_resident_fraction, 4),
            lever.memory_guard_tier,
            lever.admission,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(lever)
    return tuple(unique)


def _policy_for(
    model: ModelLayout,
    lever: _Lever,
    context_tokens: int,
    *,
    target: int,
    microbatch_size: int = 1,
) -> ResourcePolicy:
    return ResourcePolicy(
        context_tokens=context_tokens,
        target_context_tokens=target or context_tokens,
        table_offload=lever.table_offload,
        table_kind=model.table_offload_kind,
        moe_expert_offload=lever.moe_expert_offload,
        moe_resident_fraction=(
            lever.moe_resident_fraction if lever.moe_expert_offload else 1.0
        ),
        memory_guard_tier=lever.memory_guard_tier,
        microbatch_size=microbatch_size,
        admission=lever.admission,
    )


def _default_policy(
    model: ModelLayout,
    saved: SavedPolicy,
    context_tokens: int,
    *,
    microbatch_size: int,
) -> ResourcePolicy:
    return ResourcePolicy(
        context_tokens=context_tokens,
        target_context_tokens=context_tokens,
        table_offload=saved.table_offload_for(model),
        table_kind=model.table_offload_kind,
        moe_expert_offload=saved.moe_expert_offload_enabled
        and model_supports_moe_offload(model),
        moe_resident_fraction=(
            saved.moe_expert_offload_resident_fraction
            if saved.moe_expert_offload_enabled
            else 1.0
        ),
        memory_guard_tier=saved.memory_guard_tier,
        microbatch_size=microbatch_size,
        admission=ADMISSION_P50,
    )


def _declared_context(model: ModelLayout, saved: SavedPolicy) -> int | None:
    limits = [
        value
        for value in (model.max_context_tokens, saved.max_context_window)
        if value is not None and value > 0
    ]
    return min(limits) if limits else None


def _selected_facts(
    nodes: Sequence[NodeFacts], forced_node_ids: Sequence[str] | None
) -> list[NodeFacts]:
    if not forced_node_ids:
        return list(nodes)
    by_id = {facts.node_id: facts for facts in nodes}
    missing = [node_id for node_id in forced_node_ids if node_id not in by_id]
    if missing:
        raise ValueError(
            "the approved node set is not measured: " + ", ".join(sorted(missing))
        )
    return [by_id[node_id] for node_id in forced_node_ids]


def _excluded(
    selected: Sequence[NodeFacts],
    measured: Sequence[NodeFacts],
    policy: ResourcePolicy,
    result: ElectionResult,
) -> tuple[ExcludedNode, ...]:
    chosen = set(result.rank_order or [facts.node_id for facts in selected])
    return tuple(
        ExcludedNode(
            node_id=facts.node_id,
            reason=(
                "not needed: the selected set fits at "
                f"{policy.context_tokens:,} tokens with "
                f"{_policy_label(policy)}"
            ),
        )
        for facts in measured
        if facts.node_id not in chosen
    )


def _policy_label(policy: ResourcePolicy) -> str:
    parts = []
    if policy.table_offload != TABLE_OFFLOAD_NONE:
        parts.append(f"{policy.table_offload} tables on SSD")
    if policy.moe_expert_offload:
        parts.append(f"{policy.moe_resident_fraction:.0%} experts resident")
    if policy.memory_guard_tier:
        parts.append(policy.memory_guard_tier)
    return ", ".join(parts) if parts else "resident memory"


def _empty_result(
    nodes: Sequence[NodeFacts], origin_node: str | None
) -> ElectionResult:
    """A plan-shaped refusal for a policy the model cannot express."""

    from .election import ElectionPolicy

    return ElectionResult(
        feasible=False,
        coordinator_node="",
        rank_order=(),
        endpoint=None,
        selected=None,
        considered=(),
        rejected=(),
        policy=ElectionPolicy(),
        origin_node=origin_node or "",
        plan_signature="",
        reasons=("the requested resource policy cannot be applied to this model",),
        tensor_parallel_size=1,
    )


def _refusal_reason(error: NoFeasiblePlanError | None) -> str:
    if error is None:
        return "no legal pipeline split for this set"
    return str(error)


def _combined_refusal(
    refusals: Sequence[SizeRefusal], last_error: NoFeasiblePlanError
) -> str:
    if not refusals:
        return str(last_error)
    by_size = "; ".join(
        f"{item.node_count} Mac{'s' if item.node_count != 1 else ''}: {item.reason}"
        for item in refusals
    )
    return f"no set of Macs can run this model. Tried {by_size}"


def _fraction(value: Any, *, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    number = float(value)
    if not 0 < number <= 1:
        return default
    return number
