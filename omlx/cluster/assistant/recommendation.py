# SPDX-License-Identifier: Apache-2.0
"""Which model should this measured cluster run?

``plan`` answers "given this model, how do we split it?". This module answers the
question that comes first (the project design notes/§13.3a), and the two are
deliberately different contracts:

* recommendation reads only the **hardware** half of a stored measurement —
  admission, compute, links, endpoints, disk, and whether a Mac already holds a
  checkpoint — and may evaluate **any** candidate model;
* ``plan`` keeps the strict Chapter 3 match, so the model the user picks is
  measured again (model-bound) and planned from that measurement.

Three rules are deliberate.

1. **Feasibility has one authority.** Nothing here re-derives an admission or
   layer-cut rule; every candidate is handed to ``election.elect()`` and the
   verdict read back. ``elect`` already owns ``plan_hybrid``, the rank-0
   tail/head, admission and the no-fallback rule. What this module adds is only
   the question — one model, one context — and the ranking of the answers.
2. **PP only.** A model mlx-lm cannot split into pipeline stages is refused
   with one action, never offered as tensor parallelism (§8.1). ``elect`` alone
   would happily split its weights, which is how a plan gets approved for a
   model that then fails at load, so the architecture's own capability flag is
   checked before the election is asked.
3. **Unknown is not fast and not zero.** Without measured compute the result
   says "unknown" — no stage seconds, no token rate, no "fast" (§17.2).
   Likewise unknown disk free space stays ``None``; a Mac that cannot be proven
   short of room is not called full.

The context answer comes from descending a fixed ladder through ``elect``
(largest first, stopping at the first context that still plans): KV cache
grows with the prompt, so "fits" without a context length is close to
meaningless, and the largest context that still plans is the only number the
cluster can honour. The ladder is imported from ``catalogue.py`` rather than
copied: the model shelf and the recommendation must report the same presets.

Pure: no HTTP, no SSH, no settings, no writes, no model loading. The one
filesystem access is ``load_model_candidate`` reading a checkpoint's headers,
and it is separated from ``recommend`` so the decision itself is a pure function
of measured facts and layouts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

# The ladder belongs to ``catalogue.py``; two ladders would slowly disagree
# about which contexts the product reports. Imported, not copied.
from ..performance import ExecutionProfileName
from ..planner import ModelLayout, PlanningError, complete_model_layout
from .election import (
    ElectionError,
    NodeFacts,
    NoFeasiblePlanError,
    StagePlan,
    budget_for,
)
from .policy import (
    ClusterPlanSelection,
    ResourcePolicy,
    SavedPolicy,
    context_ladder,
    solve_cluster_plan,
)

GIB = 1024**3

# A recommendation is a shortlist, not a shelf dump. The brief's model shelf is
# an advanced surface; this keeps one request bounded and the answer readable.
MAX_RECOMMENDATION_CANDIDATES = 32

# The one next step a caller can act on, per refusal kind (§8.6). Exactly one is
# ever attached to a candidate, and exactly one to a result with no winner.
ACTION_CHOOSE_SMALLER_MODEL = "choose_smaller_model"
ACTION_CHOOSE_PIPELINE_MODEL = "choose_pipeline_model"
ACTION_CHECK_MODEL_PATH = "check_model_path"
ACTION_FREE_DISK_SPACE = "free_disk_space"

# Why a candidate was not recommended, machine-readable (§13.3a). ``memory`` is
# the ordinary "does not fit"; the other two need a different fix, and the
# dashboard must not render all three as "too large".
FAILURE_MEMORY = "memory"
FAILURE_PIPELINE_UNSUPPORTED = "pipeline_unsupported"
FAILURE_MODEL_UNREADABLE = "model_unreadable"

WARNING_SPEED_UNKNOWN = "speed_unknown"
WARNING_CONTEXT_BELOW_TARGET = "context_below_target"
WARNING_PRESENCE_UNMEASURED = "model_presence_unmeasured"
WARNING_STAGING_REQUIRED = "staging_required"
WARNING_DISK_TOO_SMALL = "insufficient_disk"
WARNING_MULTI_NODE = "multi_node_required"


@dataclass(frozen=True)
class RecommendationWarning:
    """One caveat about a candidate, in English, with an optional action.

    ``action`` is empty when nothing has to be done: a copy that is needed is
    the normal path, not a fault. A warning that *does* need a fix carries
    exactly one action, like every other refusal in the assistant.
    """

    code: str
    message: str
    node_id: str = ""
    action: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "node_id": self.node_id,
            "message": self.message,
            "action": self.action,
        }


@dataclass(frozen=True)
class ModelCandidate:
    """One model the user might run, and what is known about it.

    ``layout``/``layout_error`` are filled by ``load_model_candidate``; exactly
    one of them is set. ``present_on`` is the set of Macs the measurement proved
    hold *this* checkpoint. ``None`` means it was never measured for this model
    — which is not the same as "no Mac has it", so every Mac is then reported as
    needing the copy instead of the answer claiming a presence nobody checked.
    """

    model_path: str
    friendly_name: str = ""
    declared_context_tokens: int | None = None
    layout: ModelLayout | None = None
    layout_error: str = ""
    present_on: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        path = self.model_path.strip()
        if not path:
            raise ValueError("model_path is required")
        object.__setattr__(self, "model_path", path)
        if self.declared_context_tokens is not None and (
            self.declared_context_tokens <= 0
        ):
            raise ValueError("declared_context_tokens must be positive")
        if self.layout is None and not self.layout_error:
            raise ValueError(
                "a candidate needs a layout or the reason one could not be read"
            )
        if self.layout is not None and self.layout_error:
            raise ValueError("a candidate cannot have both a layout and a read error")

    @property
    def model_id(self) -> str:
        """The stable key of this candidate: its path, as the caller wrote it."""

        return self.model_path

    @property
    def label(self) -> str:
        """What to show a person: the friendly name, else the folder name."""

        name = self.friendly_name.strip()
        if name:
            return name
        return Path(self.model_path).name or self.model_path

    @property
    def weight_bytes(self) -> int:
        """Checkpoint weight bytes, or 0 when the layout could not be read."""

        return 0 if self.layout is None else self.layout.total_weight_bytes


@dataclass(frozen=True)
class CandidateNode:
    """What one Mac needs before it can hold this candidate's stage."""

    node_id: str
    model_present: bool
    #: Bytes that must be copied to this Mac before it can load its stage.
    stage_bytes: int
    #: ``None`` is "unknown", never 0.
    disk_free_bytes: int | None
    can_stage: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "model_present": self.model_present,
            "stage_bytes": self.stage_bytes,
            "disk_free_bytes": self.disk_free_bytes,
            "can_stage": self.can_stage,
        }


@dataclass(frozen=True)
class ModelRecommendation:
    """One candidate's verdict: whether, how, how far, and why not.

    A feasible recommendation is the plan for the *largest context that still
    plans*, so ``max_context_tokens`` is a number the cluster can honour rather
    than a hope. ``headroom_bytes`` is the tightest stage's headroom, which is
    what decides whether the split survives a passing workload.
    """

    model_id: str
    label: str
    model_path: str
    feasible: bool
    pp_supported: bool
    pipeline_stages: int
    nodes_required: int
    weight_bytes: int
    download_bytes: int
    max_context_tokens: int
    declared_context_tokens: int | None
    context_limited: bool
    headroom_bytes: int
    rank0_node: str
    rank_order: tuple[str, ...]
    stages: tuple[StagePlan, ...]
    nodes: tuple[CandidateNode, ...]
    critical_stage_rank: int | None
    critical_stage_seconds: float | None
    measured_speed: bool
    plan_signature: str
    failure_kind: str
    reasons: tuple[str, ...]
    warnings: tuple[RecommendationWarning, ...]
    action: str
    #: The levers this recommendation would use, and the Macs it leaves out.
    resource_policy: ResourcePolicy | None = None
    excluded_nodes: tuple[str, ...] = ()
    #: "single" (the local engine on one Mac, no pipeline) or "cluster" (a
    #: pipeline across 2+ Macs). Single-node is preferred when the model fits on
    #: one Mac at the target context (E6, the live-run notes): no pipeline overhead.
    mode: str = "cluster"

    @property
    def headroom_gib(self) -> float:
        return self.headroom_bytes / GIB

    @property
    def download_gib(self) -> float:
        return self.download_bytes / GIB

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "label": self.label,
            "model_path": self.model_path,
            "feasible": self.feasible,
            "pp_supported": self.pp_supported,
            "pipeline_stages": self.pipeline_stages,
            "nodes_required": self.nodes_required,
            "weight_bytes": self.weight_bytes,
            "download_bytes": self.download_bytes,
            "download_gib": self.download_gib,
            "max_context_tokens": self.max_context_tokens,
            "declared_context_tokens": self.declared_context_tokens,
            "context_limited": self.context_limited,
            "headroom_bytes": self.headroom_bytes,
            "headroom_gib": self.headroom_gib,
            "rank0_node": self.rank0_node,
            "rank_order": list(self.rank_order),
            "stages": [stage.to_dict() for stage in self.stages],
            "nodes": [node.to_dict() for node in self.nodes],
            "critical_stage_rank": self.critical_stage_rank,
            "critical_stage_seconds": self.critical_stage_seconds,
            "measured_speed": self.measured_speed,
            "plan_signature": self.plan_signature,
            "failure_kind": self.failure_kind,
            "reasons": list(self.reasons),
            "warnings": [warning.to_dict() for warning in self.warnings],
            "action": self.action,
            "resource_policy": (
                None if self.resource_policy is None else self.resource_policy.to_dict()
            ),
            "excluded_nodes": list(self.excluded_nodes),
            "mode": self.mode,
        }


@dataclass(frozen=True)
class RecommendationResult:
    """The ranked shortlist for one measured cluster.

    The first feasible candidate is the recommendation; the rest are what the
    user may pick instead. Feasible candidates are ranked by usable context
    first, then by the measured critical stage, then by smaller size — never
    by raw checkpoint size alone. When nothing is feasible the near misses
    are ordered smallest first, so the next thing to try is the first thing
    shown, and the result carries exactly one action.
    """

    candidates: tuple[ModelRecommendation, ...]
    target_context_tokens: int | None
    mtp: bool
    execution_profile: ExecutionProfileName
    reasons: tuple[str, ...]
    #: The single-node option for the selected model (E6): when the recommended
    #: model also fits on one Mac at the target context, the local engine is
    #: faster than a pipeline (the live-run notes). ``None`` when it does not fit
    #: on one Mac. The cluster recommendation above stays the compatible answer;
    #: this is the preferred faster option the caller may choose instead.
    single_node: ModelRecommendation | None = None

    @property
    def feasible(self) -> bool:
        return self.selected is not None

    @property
    def selected(self) -> ModelRecommendation | None:
        return next(
            (item for item in self.candidates if item.feasible),
            None,
        )

    @property
    def action(self) -> str:
        """The one next step. Empty when a model was recommended.

        Candidates arrive "near misses smallest first", so the first entry is
        usually the smallest model — which is often simply not installed on this
        Mac and therefore ``model_unreadable``. Returning *its* action told a
        user whose real blocker was memory to check a model path. The action
        therefore comes from the first near miss whose failure is about the
        cluster rather than about a missing checkpoint; a group where every
        candidate is unreadable still answers ``check_model_path``.
        """

        selected = self.selected
        if selected is not None:
            return ""
        if not self.candidates:
            return ACTION_CHOOSE_SMALLER_MODEL
        for candidate in self.candidates:
            if candidate.failure_kind != FAILURE_MODEL_UNREADABLE:
                return candidate.action or ACTION_CHOOSE_SMALLER_MODEL
        return self.candidates[0].action or ACTION_CHOOSE_SMALLER_MODEL


def load_model_candidate(
    model_path: str,
    *,
    friendly_name: str = "",
    declared_context_tokens: int | None = None,
) -> ModelCandidate:
    """Read one candidate's layout from its own headers.

    This is the only filesystem access in the module, and it is deliberately
    outside ``recommend``: a caller that already has layouts can rank them
    without touching a disk, and a model that cannot be read becomes a candidate
    with a reason instead of disappearing from the list. The readers are oMLX's
    own — ``complete_model_layout`` refuses a stage fragment, and
    ``_read_model_context_length`` is where the real context limit lives.
    """

    path = (model_path or "").strip()
    if not path:
        raise ValueError("model_path is required")
    declared = declared_context_tokens
    try:
        layout = complete_model_layout(path)
    except (OSError, PlanningError, RuntimeError, ValueError) as exc:
        return ModelCandidate(
            model_path=path,
            friendly_name=friendly_name,
            declared_context_tokens=declared,
            layout_error=f"could not read the model: {_short(exc)}",
        )
    if declared is None:
        declared = _declared_context(path)
    return ModelCandidate(
        model_path=path,
        friendly_name=friendly_name,
        declared_context_tokens=declared,
        layout=layout,
    )


def recommend(
    nodes: Sequence[NodeFacts],
    candidates: Sequence[ModelCandidate],
    *,
    context_tokens: int | None = None,
    execution_profile: ExecutionProfileName = "balanced",
    mtp: bool = False,
    rank0_extra_bytes: int = 0,
    microbatch_size: int = 1,
    saved_settings: Mapping[str, Any] | None = None,
) -> RecommendationResult:
    """Rank these candidates for these measured Macs. Pure and idempotent.

    ``context_tokens`` is the caller's target, not a promise: each candidate is
    reported at the largest ladder context that still plans *and* does not
    exceed the target or the model's own declared limit. Feasible candidates
    are ranked by usable context first, then by the measured critical stage,
    then by smaller size; near misses follow, smallest first. ``mtp`` is the
    Lightning setting the measurement was taken with, because the model chosen
    here will be measured and planned again with that same setting.
    """

    node_list = list(nodes)
    if not node_list:
        raise ValueError("at least one node is required")
    node_ids = [facts.node_id for facts in node_list]
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("node IDs must be unique")
    if len(candidates) > MAX_RECOMMENDATION_CANDIDATES:
        raise ValueError(
            f"a recommendation covers at most {MAX_RECOMMENDATION_CANDIDATES} "
            f"candidates; got {len(candidates)}"
        )
    if not candidates:
        raise ValueError("at least one candidate model is required")
    if context_tokens is not None and context_tokens <= 0:
        raise ValueError("context_tokens must be positive")
    if rank0_extra_bytes < 0:
        raise ValueError("rank0_extra_bytes must be non-negative")
    if microbatch_size < 1:
        raise ValueError("microbatch_size must be positive")

    verdicts = [
        _recommend_one(
            node_list,
            replace(candidate, present_on=_presence(node_list, candidate)),
            target=context_tokens,
            execution_profile=execution_profile,
            mtp=mtp,
            rank0_extra_bytes=rank0_extra_bytes,
            microbatch_size=microbatch_size,
            saved=SavedPolicy.from_mapping(saved_settings),
        )
        for candidate in candidates
    ]
    ordered = _ordered(verdicts)
    return RecommendationResult(
        candidates=ordered,
        target_context_tokens=context_tokens,
        mtp=mtp,
        execution_profile=execution_profile,
        reasons=_result_reasons(ordered, target=context_tokens),
        single_node=_single_node_option(
            node_list,
            candidates,
            ordered,
            target=context_tokens,
            mtp=mtp,
            rank0_extra_bytes=rank0_extra_bytes,
        ),
    )


def _single_node_option(
    nodes: Sequence[NodeFacts],
    candidates: Sequence[ModelCandidate],
    ordered: Sequence[ModelRecommendation],
    *,
    target: int | None,
    mtp: bool,
    rank0_extra_bytes: int,
) -> ModelRecommendation | None:
    """The single-node option for the selected model, when it fits on one Mac.

    Additive to the cluster recommendation (E6): the recommended model may also
    run on a single Mac at the target context, which is faster than a pipeline
    (the live-run notes). Returns ``None`` when no model is recommended or the
    selected model does not fit on one Mac at the target context.
    """

    selected = next((item for item in ordered if item.feasible), None)
    if selected is None:
        return None
    candidate = next(
        (item for item in candidates if item.model_id == selected.model_id), None
    )
    if candidate is None or candidate.layout is None:
        return None
    candidate = replace(candidate, present_on=_presence(nodes, candidate))
    declared = candidate.declared_context_tokens
    target_tokens = target or (declared or 32768)
    fit = _single_node_fit(
        nodes,
        candidate.layout,
        target=target_tokens,
        declared=declared,
        mtp=mtp,
        rank0_extra_bytes=rank0_extra_bytes,
    )
    layout = candidate.layout
    if fit is None or fit[1] < target_tokens or layout is None:
        return None
    download_bytes = candidate.weight_bytes + (layout.mtp_weight_bytes if mtp else 0)
    return _single_node_recommendation(
        candidate,
        nodes,
        fit,
        download_bytes=download_bytes,
        declared=declared,
        target=target,
    )


def _presence(
    nodes: Sequence[NodeFacts], candidate: ModelCandidate
) -> tuple[str, ...] | None:
    """Which of these Macs listed this checkpoint, or ``None`` when unknown.

    The measurement records each Mac's own catalogue, so "already there" is a
    fact for any candidate — not only for the checkpoint that happened to be
    measured. ``None`` still means "not measured" and keeps the honest
    ``staging_required`` warning.
    """

    covered = [facts for facts in nodes if facts.model_paths]
    if not covered:
        return candidate.present_on
    wanted = _path_key(candidate.model_path)
    return tuple(
        facts.node_id
        for facts in covered
        if wanted in {_path_key(path) for path in facts.model_paths}
    )


def _path_key(value: str) -> str:
    text = str(value or "").strip().rstrip("/")
    return str(Path(text).expanduser()) if text else ""


def _recommend_one(
    nodes: Sequence[NodeFacts],
    candidate: ModelCandidate,
    *,
    target: int | None,
    execution_profile: ExecutionProfileName,
    mtp: bool,
    rank0_extra_bytes: int,
    microbatch_size: int,
    saved: SavedPolicy,
) -> ModelRecommendation:
    """One candidate, asked the same question of the same authority."""

    layout = candidate.layout

    if layout is None:
        return _refusal(
            candidate,
            nodes,
            download_bytes=0,
            declared=candidate.declared_context_tokens,
            failure_kind=FAILURE_MODEL_UNREADABLE,
            pp_supported=False,
            action=ACTION_CHECK_MODEL_PATH,
            reasons=(candidate.layout_error,),
        )

    # The checkpoint size that has to be copied, draft head included: a Mac
    # missing the model needs room for every byte, not just the decoder layers.
    download_bytes = candidate.weight_bytes + (layout.mtp_weight_bytes if mtp else 0)
    declared = candidate.declared_context_tokens

    if len(nodes) > 1 and not layout.supports_pipeline:
        # mlx-lm cannot split this architecture into stages, so a multi-Mac
        # plan would stage the weights and then fail at load (§8.1).
        if layout.supports_tensor_parallel:
            reason = (
                f"{candidate.label} cannot be split into pipeline stages; it "
                f"would need tensor parallelism across {len(nodes)} Macs, and "
                "the Cluster Assistant plans pipeline parallelism only"
            )
        else:
            reason = (
                f"mlx-lm cannot split {candidate.label} at all, so it cannot "
                f"run across {len(nodes)} Macs"
            )
        return _refusal(
            candidate,
            nodes,
            download_bytes=download_bytes,
            declared=declared,
            failure_kind=FAILURE_PIPELINE_UNSUPPORTED,
            pp_supported=False,
            action=ACTION_CHOOSE_PIPELINE_MODEL,
            reasons=(reason,),
        )

    selection, failure = _solve_one(
        nodes,
        layout,
        target=target,
        declared=declared,
        execution_profile=execution_profile,
        mtp=mtp,
        rank0_extra_bytes=rank0_extra_bytes,
        microbatch_size=microbatch_size,
        saved=saved,
    )
    if selection is None:
        return _refusal(
            candidate,
            nodes,
            download_bytes=download_bytes,
            declared=declared,
            failure_kind=FAILURE_MEMORY,
            pp_supported=True,
            action=ACTION_CHOOSE_SMALLER_MODEL,
            reasons=(failure,),
        )

    election = selection.election
    selected = election.selected
    if selected is None:  # pragma: no cover - elect() cannot be feasible without it
        raise ElectionError("a feasible election returned no selected plan")
    tokens = selection.policy.context_tokens

    critical = selected.critical_stage
    measured_speed = selected.critical_stage_seconds is not None
    # Staging warnings are about the Macs this plan will actually write.
    # Excluded Macs are never staged; naming them made a 2-Mac DeepSeek plan
    # read as if it had to copy 357 GiB to all four Macs.
    selected_ids = set(selected.rank_order)
    staging = tuple(
        row
        for row in _staging_nodes(nodes, candidate, download_bytes)
        if row.node_id in selected_ids
    )
    warnings = _feasible_warnings(
        candidate,
        staging,
        download_bytes=download_bytes,
        context_tokens=tokens,
        target=target,
        measured_speed=measured_speed,
        stage_count=len(selected.stages),
    )
    reasons = [
        f"{candidate.label} fits {len(selected.stages)} "
        f"{'Mac' if len(selected.stages) == 1 else 'Macs'} in "
        f"{len(selected.stages)} pipeline "
        f"{'stage' if len(selected.stages) == 1 else 'stages'} at "
        f"{tokens:,} tokens of context",
        f"rank 0 is {selected.coordinator_node}, with "
        f"{_gib(selected.rank0_headroom_bytes)} GiB headroom inside its "
        f"{_gib(selected.rank0_admission_bytes)} GiB admission",
    ]
    if selected.critical_stage_seconds is not None and critical is not None:
        reasons.append(
            f"slowest stage: rank {critical.rank} on {critical.node_id} "
            f"({selected.critical_stage_seconds:.3f} s)"
        )

    return ModelRecommendation(
        model_id=candidate.model_id,
        label=candidate.label,
        model_path=candidate.model_path,
        feasible=True,
        pp_supported=True,
        pipeline_stages=len(selected.stages),
        nodes_required=len(selected.stages),
        weight_bytes=candidate.weight_bytes,
        download_bytes=download_bytes,
        max_context_tokens=tokens,
        declared_context_tokens=declared,
        context_limited=declared is not None and tokens < declared,
        headroom_bytes=selected.min_headroom_bytes,
        rank0_node=selected.coordinator_node,
        rank_order=selected.rank_order,
        stages=selected.stages,
        nodes=_staging_nodes(nodes, candidate, download_bytes),
        critical_stage_rank=None if critical is None else critical.rank,
        critical_stage_seconds=selected.critical_stage_seconds,
        measured_speed=measured_speed,
        plan_signature=selected.plan_hash,
        failure_kind="",
        reasons=tuple(reasons),
        warnings=warnings,
        action="",
        resource_policy=selection.policy,
        excluded_nodes=tuple(item.node_id for item in selection.excluded_nodes),
    )


def _solve_one(
    nodes: Sequence[NodeFacts],
    layout: ModelLayout,
    *,
    target: int | None,
    declared: int | None,
    execution_profile: ExecutionProfileName,
    mtp: bool,
    rank0_extra_bytes: int,
    microbatch_size: int,
    saved: SavedPolicy,
) -> tuple[ClusterPlanSelection | None, str]:
    """The resource-policy solver's best plan for this candidate.

    A recommendation is not a second planner: every candidate is handed to the
    same ``solve_cluster_plan`` that ``plan`` uses, so the recommendation and
    the later plan cannot disagree about the Mac set, the context or the levers.
    ``declared`` is folded into the saved settings so a checkpoint's own context
    limit caps the ladder even when the model layout did not carry it.
    """

    effective = saved
    if declared is not None and declared > 0:
        current = saved.max_context_window
        effective = replace(
            saved,
            max_context_window=(
                declared if current is None else min(current, declared)
            ),
        )
    target_tokens = target or (declared or 32768)
    try:
        selection = solve_cluster_plan(
            nodes,
            layout,
            target_context_tokens=target_tokens,
            rank0_extra_bytes=rank0_extra_bytes,
            mtp=mtp,
            execution_profile=execution_profile,
            microbatch_size=microbatch_size,
            saved=effective,
        )
    except NoFeasiblePlanError as exc:
        return None, str(exc)
    except (ElectionError, PlanningError, ValueError) as exc:
        return None, str(exc)
    return selection, ""


def _single_node_fit(
    nodes: Sequence[NodeFacts],
    layout: ModelLayout,
    *,
    target: int,
    declared: int | None,
    mtp: bool,
    rank0_extra_bytes: int,
) -> tuple[NodeFacts, int, int, int, Any] | None:
    """The best single Mac + largest ladder context that fits on it, or None.

    Single-node inference (E6): the whole model on one Mac, no pipeline. Reuses
    ``budget_for`` (the one admission authority): resident weights (with the
    checkpoint's Engram/PLE table on SSD, which the plan writes and the engine
    applies) + MTP head + KV for the context + rank0 overhead must fit the Mac's
    rank-0 admission. The best Mac is the one with the most headroom at that
    context. Returns ``(facts, context, headroom, required, budget)``.
    """

    ladder = context_ladder(target=target, declared=declared)
    # The plan (and the engine) keeps a checkpoint's lookup table — DeepSeek
    # Engram or Qwen4 PLE — on SSD by default: it costs no speed, and charging
    # it to RAM here would refuse a local run the engine can serve. The engine
    # prices exactly these resident layer bytes (`tensor_parallel_layer_bytes`)
    # and forces the switch when residency would leave no prompt room, so this
    # is the same question asked of the same numbers, not a second estimate.
    weights = (
        layout.total_weight_bytes
        - sum(layout.tensor_parallel_ssd_layer_bytes)
        + (layout.mtp_weight_bytes if mtp else 0)
    )
    kv_per_token = layout.kv_bytes_per_token_per_layer * layout.layer_count
    budgets = [(facts, budget_for(facts, rank=0)) for facts in nodes]
    for ctx in sorted(ladder, reverse=True):
        required = weights + kv_per_token * ctx + rank0_extra_bytes
        fitting = [
            (facts, budget.usable_bytes - required, budget)
            for facts, budget in budgets
            if required <= budget.usable_bytes
        ]
        if fitting:
            facts, headroom, budget = max(fitting, key=lambda item: item[1])
            return facts, ctx, headroom, required, budget
    return None


def _single_node_recommendation(
    candidate: ModelCandidate,
    nodes: Sequence[NodeFacts],
    fit: tuple[NodeFacts, int, int, int, Any],
    *,
    download_bytes: int,
    declared: int | None,
    target: int | None,
) -> ModelRecommendation:
    """A single-node recommendation: the local engine on one Mac, no pipeline."""

    facts, ctx, headroom, required, budget = fit
    layout = candidate.layout
    assert layout is not None  # a fit is only computed from a measured layout
    stage = StagePlan(
        rank=0,
        node_id=facts.node_id,
        first_layer=0,
        last_layer=layout.layer_count,
        layer_count=layout.layer_count,
        resident_bytes=required,
        admission_bytes=budget.usable_bytes,
        headroom_bytes=headroom,
        predicted_seconds=None,
        critical=True,
    )
    staging = _staging_nodes(nodes, candidate, download_bytes)
    warnings = _feasible_warnings(
        candidate,
        staging,
        download_bytes=download_bytes,
        context_tokens=ctx,
        target=target,
        measured_speed=False,
        stage_count=1,
    )
    reasons = (
        f"{candidate.label} fits on one Mac ({facts.node_id}) at {ctx:,} tokens "
        "of context — single-node inference, no pipeline overhead",
        f"{_gib(headroom)} GiB headroom inside {_gib(budget.usable_bytes)} GiB "
        "admission",
    )
    return ModelRecommendation(
        model_id=candidate.model_id,
        label=candidate.label,
        model_path=candidate.model_path,
        feasible=True,
        pp_supported=True,
        pipeline_stages=1,
        nodes_required=1,
        weight_bytes=candidate.weight_bytes,
        download_bytes=download_bytes,
        max_context_tokens=ctx,
        declared_context_tokens=declared,
        context_limited=declared is not None and ctx < declared,
        headroom_bytes=headroom,
        rank0_node=facts.node_id,
        rank_order=(facts.node_id,),
        stages=(stage,),
        nodes=staging,
        critical_stage_rank=0,
        critical_stage_seconds=None,
        measured_speed=False,
        plan_signature="",
        failure_kind="",
        reasons=reasons,
        warnings=warnings,
        action="",
        resource_policy=None,
        excluded_nodes=tuple(
            facts.node_id for facts in nodes if facts.node_id != fit[0].node_id
        ),
        mode="single",
    )


def _staging_nodes(
    nodes: Sequence[NodeFacts],
    candidate: ModelCandidate,
    download_bytes: int,
) -> tuple[CandidateNode, ...]:
    """What each Mac needs before it can load its stage of this candidate."""

    present_on = candidate.present_on
    rows: list[CandidateNode] = []
    for facts in nodes:
        present = present_on is not None and facts.node_id in present_on
        stage_bytes = 0 if present else download_bytes
        rows.append(
            CandidateNode(
                node_id=facts.node_id,
                model_present=present,
                stage_bytes=stage_bytes,
                disk_free_bytes=facts.disk_free_bytes,
                # The election's own predicate, asked about *this* candidate:
                # ``NodeFacts.model_present`` describes the checkpoint the
                # measurement covered, not the one being recommended here.
                can_stage=replace(facts, model_present=present).can_stage(stage_bytes),
            )
        )
    return tuple(rows)


def _feasible_warnings(
    candidate: ModelCandidate,
    staging: Sequence[CandidateNode],
    *,
    download_bytes: int,
    context_tokens: int,
    target: int | None,
    measured_speed: bool,
    stage_count: int,
) -> tuple[RecommendationWarning, ...]:
    """Everything a person should know before choosing this model."""

    warnings: list[RecommendationWarning] = []

    if candidate.present_on is None:
        warnings.append(
            RecommendationWarning(
                code=WARNING_PRESENCE_UNMEASURED,
                message=(
                    "the measurement covered a different model, so whether these "
                    f"Macs already hold {candidate.label} was not measured; "
                    f"{_gib(download_bytes)} GiB is assumed to be copied"
                ),
            )
        )

    missing = [row for row in staging if not row.model_present]
    if missing:
        warnings.append(
            RecommendationWarning(
                code=WARNING_STAGING_REQUIRED,
                message=(
                    f"{_gib(download_bytes)} GiB has to be copied to "
                    + ", ".join(row.node_id for row in missing)
                ),
            )
        )

    for row in staging:
        if row.model_present or row.can_stage:
            continue
        warnings.append(
            RecommendationWarning(
                code=WARNING_DISK_TOO_SMALL,
                node_id=row.node_id,
                message=(
                    f"{row.node_id} has {_gib(row.disk_free_bytes or 0)} GiB free "
                    f"but needs {_gib(download_bytes)} GiB for this model"
                ),
                action=ACTION_FREE_DISK_SPACE,
            )
        )

    if not measured_speed:
        warnings.append(
            RecommendationWarning(
                code=WARNING_SPEED_UNKNOWN,
                message=(
                    "no compute was measured on these Macs, so no speed is "
                    "predicted for this model"
                ),
            )
        )

    if target is not None and context_tokens < target:
        warnings.append(
            RecommendationWarning(
                code=WARNING_CONTEXT_BELOW_TARGET,
                message=(
                    f"the largest context that fits is {context_tokens:,} tokens, "
                    f"below the requested {target:,}"
                ),
            )
        )

    if stage_count > 1:
        warnings.append(
            RecommendationWarning(
                code=WARNING_MULTI_NODE,
                message=(
                    f"needs {stage_count} Macs; the model cannot run if one goes away"
                ),
            )
        )

    return tuple(warnings)


def _refusal(
    candidate: ModelCandidate,
    nodes: Sequence[NodeFacts],
    *,
    download_bytes: int,
    declared: int | None,
    failure_kind: str,
    pp_supported: bool,
    action: str,
    reasons: Sequence[str],
) -> ModelRecommendation:
    """A candidate that cannot be chosen, with its reason and one next step."""

    return ModelRecommendation(
        model_id=candidate.model_id,
        label=candidate.label,
        model_path=candidate.model_path,
        feasible=False,
        pp_supported=pp_supported,
        pipeline_stages=0,
        nodes_required=0,
        weight_bytes=candidate.weight_bytes,
        download_bytes=download_bytes,
        max_context_tokens=0,
        declared_context_tokens=declared,
        context_limited=False,
        headroom_bytes=0,
        rank0_node="",
        rank_order=(),
        stages=(),
        nodes=_staging_nodes(nodes, candidate, download_bytes),
        critical_stage_rank=None,
        critical_stage_seconds=None,
        measured_speed=False,
        plan_signature="",
        failure_kind=failure_kind,
        reasons=tuple(reason for reason in reasons if reason),
        warnings=(),
        action=action,
    )


def _ranking_key(item: ModelRecommendation) -> tuple[Any, ...]:
    """Ascending sort key for feasible candidates: context, speed, then size.

    Usable context is quality, so the largest context that still plans ranks
    first — a heavier checkpoint capped at a tiny context is the worse answer
    (B11). Equal contexts are broken by the measured critical stage, the same
    predicted seconds the election scores; ``None`` is worst-case, because
    unknown is not fast (§17.2) and no rate is ever estimated. Smaller size is
    the last resort, and the model ID keeps the answer stable for one input.
    """

    critical = (
        float("inf")
        if item.critical_stage_seconds is None
        else item.critical_stage_seconds
    )
    return -item.max_context_tokens, critical, item.weight_bytes, item.model_id


def _ordered(
    candidates: Sequence[ModelRecommendation],
) -> tuple[ModelRecommendation, ...]:
    """Feasible first by usable context, then near misses smallest first.

    Feasible candidates are ranked by ``_ranking_key``: the largest usable
    context first, then the faster measured critical stage, then the smaller
    checkpoint. This deliberately departs from ``catalogue_for_cluster``'s
    largest-first order: raw size placed a 253.9 GiB checkpoint capped at 4k
    above a 117.2 GiB one that honours 32k (B11), and the assistant then
    recommended the worse model. Near misses keep the catalogue's order —
    smallest first — so the next thing worth trying is the first thing shown.
    """

    runnable = sorted(
        (item for item in candidates if item.feasible),
        key=_ranking_key,
    )
    rejected = sorted(
        (item for item in candidates if not item.feasible),
        key=lambda item: (item.weight_bytes, item.model_id),
    )
    return tuple(runnable) + tuple(rejected)


def _result_reasons(
    candidates: Sequence[ModelRecommendation],
    *,
    target: int | None,
) -> tuple[str, ...]:
    selected = next((item for item in candidates if item.feasible), None)
    wanted = "" if target is None else f" at up to {target:,} tokens"
    if selected is not None:
        return (
            f"{selected.label} is the recommended model for these Macs{wanted}: "
            f"{selected.pipeline_stages} pipeline "
            f"{'stage' if selected.pipeline_stages == 1 else 'stages'} at "
            f"{selected.max_context_tokens:,} tokens, rank 0 on "
            f"{selected.rank0_node}",
        )
    if not candidates:
        return ("no candidate models were supplied",)
    return (
        f"none of the {len(candidates)} candidate models fits these Macs{wanted}",
        *(
            f"{item.label}: {item.reasons[0] if item.reasons else 'does not fit'}"
            for item in candidates
        ),
    )


def _declared_context(model_path: str) -> int | None:
    """The model's own context limit, when oMLX can read it without loading.

    ``catalogue.assess_model_path`` uses the same reader; a model whose config
    does not declare one keeps ``None`` — unknown is not a number.
    """

    try:
        from omlx.model_discovery import _read_model_context_length

        value = _read_model_context_length(Path(model_path))
    except Exception:  # a missing or odd config is not a reason to refuse
        return None
    return value if isinstance(value, int) and value > 0 else None


def _gib(value: float) -> str:
    return f"{value / GIB:.1f}"


def _short(exc: BaseException) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    return text[-300:]
