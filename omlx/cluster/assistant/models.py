# SPDX-License-Identifier: Apache-2.0
"""Request and response schemas for the Cluster Assistant admin API.

The contract is the one decided in PROJECT-BRIEF §13.2/§13.3 and §8.5:

* ``measure`` starts a job and returns a ``measurement_id`` immediately;
* ``status`` reports that job;
* ``plan`` is synchronous, pure, and takes the facts from a persisted
  measurement selected by ``measurement_id`` — never from the request body and
  never from a probe it runs itself.

The plan object this module serialises is::

    measurement_id, measurement_timestamp, plan_signature, feasible,
    coordinator_node, rank_order, groups, stages, reasons, warnings, policy,
    origin_node, tensor_parallel_size, rejected, model, context_tokens, mtp,
    predicted, observed, endpoint

They live here rather than in ``election.py`` on purpose. The election is a
decision library with a frozen contract and a frozen scoring order; the API
shape around it will move (a session block, a recommendation, an applied plan),
and serialisation details must not drag the decision with them. Everything this
module adds to an ``ElectionResult`` is *derived* and non-scoring: warnings about
unmeasured inputs, the predicted-stage block, and the endpoint taken from the
measurement record.

``predicted`` deliberately contains no tokens/s. PROJECT-BRIEF §17.2 forbids
promising absolute throughput before it has been measured, and oMLX's own plan
objects report predicted stage seconds, not model throughput — so this aligns
with them and leaves the rate fields to the post-smoke measurement, which will
fill ``observed``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..planner import PlanningError
from .election import ElectionResult
from .operations import (
    OperationEndpoint,
    OperationObserved,
    OperationProgress,
    OperationRecord,
    OperationWarning,
)
from .policy import ClusterPlanSelection, ResourcePolicy
from .provisioning import PlanInputs
from .recommendation import (
    MAX_RECOMMENDATION_CANDIDATES,
    ModelRecommendation,
    RecommendationResult,
)
from .recommendation import RecommendationWarning as RecommendationWarningData
from .state import MeasurementRecord, ModelIdentity
from .topology import missing_line_links

# The MVP supports 2–4 Macs; ``election.MAX_CANDIDATE_NODES`` is the authority,
# and a request may carry one Mac because a single-Mac plan is a legal answer
# ("this Mac can run the model alone").
MAX_REQUEST_NODES = 4
MAX_CONTEXT_TOKENS = 1_048_576
MAX_LAYER_COUNT = 2048
MAX_MICROBATCH = 256
MAX_TENSOR_PARALLEL = 64

WARNING_UNMEASURED_PLAN = "unmeasured_plan"
WARNING_ENDPOINT_UNRESOLVED = "endpoint_unresolved"

# The one next step when the elected Mac cannot name a dialable endpoint.
ACTION_CONFIGURE_ENDPOINT = "configure_endpoint"
# The one next step when no Mac can host rank 0 for these inputs (§8.6).
ACTION_ADJUST_PLAN = "adjust_model_or_context"


def request_model_identity(
    *,
    model_path: str | None,
    model_size_bytes: int | None,
    layer_count: int,
    mtp: bool,
) -> ModelIdentity:
    """The checkpoint a request names, in the one type the store compares.

    Exactly one identity, mirroring ``cluster.routes``: a real checkpoint on
    disk, or the pre-download size plus layer count. ``PlanningError`` is the
    same error the planner raises, so callers have one failure to handle.
    """

    path = (model_path or "").strip()
    if path and model_size_bytes is not None:
        raise PlanningError("provide exactly one of model_path or model_size_bytes")
    if not path and model_size_bytes is None:
        raise PlanningError("provide exactly one of model_path or model_size_bytes")
    try:
        return ModelIdentity(
            model_path=path,
            model_size_bytes=model_size_bytes,
            layer_count=layer_count,
            mtp=mtp,
        )
    except ValueError as exc:
        raise PlanningError(str(exc)) from exc


class AssistantMeasureNode(BaseModel):
    """One Mac the measurement job may probe over SSH.

    The same shape as ``ClusterNodeBudgetHostRequest``. It is re-declared here
    so the assistant API does not import the whole cluster admin route module
    for a schema.
    """

    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1, max_length=128)
    ssh: str = Field(min_length=1, max_length=255)
    python_executable: str | None = Field(default=None, max_length=4096)


class AssistantMeasureRequest(BaseModel):
    """Start measuring these Macs for this model. Read-only for the cluster."""

    model_config = ConfigDict(extra="forbid")

    nodes: list[AssistantMeasureNode] = Field(
        min_length=1, max_length=MAX_REQUEST_NODES
    )
    # node_id → role key (``headless`` / ``workstation``), as in node-budgets.
    roles: dict[str, str] = Field(default_factory=dict)
    model_path: str | None = Field(default=None, max_length=4096)
    model_size_bytes: int | None = Field(default=None, gt=0, le=1 << 60)
    layer_count: int = Field(default=80, gt=0, le=MAX_LAYER_COUNT)
    # Lightning MTP changes rank-0's weight, so it is part of what is measured
    # and a plan that asks for a different setting is refused.
    mtp: bool = False
    # The distributed performance probe boots a launched job on every Mac. It
    # is opt-in, and only runs when the server injected an implementation.
    include_performance_probe: bool = False

    def model_identity(self) -> ModelIdentity:
        """The identity this measurement covers.

        No model path and no size is a **hardware-only** measurement: it
        answers "what hardware is here?" before the user chooses a model. The
        chosen model is measured again before ``plan``.
        """

        if self.model_path is None and self.model_size_bytes is None:
            if self.mtp:
                raise PlanningError(
                    "a hardware-only measurement cannot reserve the MTP head"
                )
            return ModelIdentity(hardware_only=True)
        return request_model_identity(
            model_path=self.model_path,
            model_size_bytes=self.model_size_bytes,
            layer_count=self.layer_count,
            mtp=self.mtp,
        )


class AssistantMeasureResponse(BaseModel):
    """What a started measurement returns: a handle, not a result."""

    measurement_id: str
    state: Literal["running"]
    poll: str


class AssistantRunLocalRequest(BaseModel):
    """Run the measured model on this Mac alone, without a cluster.

    The measured Macs stay what they are; this question is only "does this
    checkpoint fit on *this* Mac at this context, and may its own engine load
    it". It is deliberately not a plan request: ``plan`` stays the PP-only
    cluster contract and a single-Mac plan stays unapplicable
    (``ClusterDeployment`` needs 2+ hosts).

    ``layer_count`` is optional and only a consistency check against the
    measured layout — a measurement identity keyed by a path does not carry
    layers, so the caller may repeat what it saw. ``None`` means "not asserted".
    """

    model_config = ConfigDict(extra="forbid")

    measurement_id: str = Field(min_length=1, max_length=128)
    model_path: str = Field(min_length=1, max_length=4096)
    layer_count: int | None = Field(default=None, gt=0, le=MAX_LAYER_COUNT)
    context_tokens: int = Field(default=8192, ge=1, le=MAX_CONTEXT_TOKENS)
    mtp: bool = False


class AssistantPlanRequest(BaseModel):
    """One planning question about one already-taken measurement.

    The Macs, their admission, compute and endpoints all come from the stored
    measurement. The remaining fields are plan-time inputs:

    * ``context_tokens`` and ``rank0_extra_bytes`` change the split, not what was
      measured, so they are deliberately not part of the measurement's identity;
    * ``mtp`` and the model identity *are*, because they change which weights
      rank 0 holds — a mismatched plan is refused instead of answered.
    """

    model_config = ConfigDict(extra="forbid")

    measurement_id: str = Field(min_length=1, max_length=128)
    # Optional explicit node set. When omitted the measurement's own Macs are
    # used; when given it must match exactly, because a plan may not silently
    # cover a different cluster than the one that was measured.
    node_ids: list[str] | None = Field(
        default=None, min_length=1, max_length=MAX_REQUEST_NODES
    )
    model_path: str | None = Field(default=None, max_length=4096)
    model_size_bytes: int | None = Field(default=None, gt=0, le=1 << 60)
    layer_count: int = Field(default=80, gt=0, le=MAX_LAYER_COUNT)
    context_tokens: int = Field(default=8192, ge=1, le=MAX_CONTEXT_TOKENS)
    mtp: bool = False
    rank0_extra_bytes: int = Field(default=0, ge=0, le=1 << 60)
    execution_profile: Literal["interactive", "balanced", "throughput"] = "balanced"
    microbatch_size: int | None = Field(default=None, ge=1, le=MAX_MICROBATCH)
    # Where the user opened the app. Recorded, never used to pick rank 0.
    origin_node: str | None = Field(default=None, max_length=255)
    tensor_parallel_size: int = Field(default=1, ge=1, le=MAX_TENSOR_PARALLEL)
    # Resource policy overrides. ``None`` keeps the saved setting; the solver
    # only proposes a change and the plan shows it.
    memory_guard_tier: Literal["safe", "balanced", "aggressive", "custom"] | None = (
        None
    )
    table_offload: Literal["engram", "ple", "none"] | None = None
    moe_expert_offload: bool | None = None
    moe_resident_fraction: float | None = Field(default=None, gt=0, le=1)
    # p05 is the floor, p50/live is the default, p95 needs explicit consent.
    admission: Literal["p05", "p50", "p95"] = "p50"
    allow_optimistic_admission: bool = False
    # Fewest Macs that fit is the default; "all" is the advanced override.
    node_selection: Literal["minimal", "all"] = "minimal"
    # Explicit consent to the offered 2-Mac plan when the selected fabric is
    # incomplete. It salts the signature, so apply/start must repeat it.
    confirm_two_mac_fallback: bool = False

    def model_identity(self) -> ModelIdentity:
        return request_model_identity(
            model_path=self.model_path,
            model_size_bytes=self.model_size_bytes,
            layer_count=self.layer_count,
            mtp=self.mtp,
        )


class PlanStage(BaseModel):
    """One pipeline stage, as oMLX's planner produced it."""

    rank: int
    node_id: str
    first_layer: int
    last_layer: int
    layer_count: int
    resident_bytes: int
    resident_gib: float
    admission_bytes: int
    headroom_bytes: int
    headroom_gib: float
    predicted_seconds: float | None
    critical: bool


class PlanStageSeconds(BaseModel):
    """The predicted time of one stage, or ``None`` when it was not measured."""

    rank: int
    node_id: str
    seconds: float | None


class PlanPrediction(BaseModel):
    """What the measurements support predicting — and nothing more.

    There is no ``decode_tok_per_s`` here: an unmeasured rate is not zero and
    not a guess (PROJECT-BRIEF §17.2). ``measured`` false means the whole block
    is worst case, which is exactly how the election scored it.
    """

    measured: bool
    critical_stage_rank: int | None
    critical_stage_seconds: float | None
    stages: list[PlanStageSeconds]


class PlanObserved(BaseModel):
    """Measured throughput, filled only after a smoke request.

    Always ``None`` in this phase. The shape is declared so the plan object's
    ``observed`` is a field rather than a promise, and so token rates have one
    place to arrive later instead of leaking into ``predicted``.
    """

    prefill_tok_per_s: float | None = None
    decode_tok_per_s: float | None = None
    ttft_ms: float | None = None
    measured_at: str = ""


class PlanEndpoint(BaseModel):
    """The endpoint the elected Mac owns.

    ``port`` is the elected Mac's own configured port, already resolved by the
    measurement. It is never 0: an unresolved endpoint is ``None`` on the plan,
    with one action, because a clickable URL nobody answers is worse than none.
    ``ready`` stays false until a health check after start.
    """

    node: str
    host: str
    port: int = Field(ge=1, le=65535)
    url: str
    ready: bool = False


class PlanTopology(BaseModel):
    """The measured fabric, as the pipeline line this plan needs.

    ``measured`` false means the transport probe did not answer; the plan then
    uses the nominal ring and ``complete`` is unknown. When it is measured,
    ``missing_links`` names the exact consecutive rank pairs without a cable.
    """

    kind: Literal["line"] = "line"
    measured: bool = False
    complete: bool | None = None
    missing_links: list[list[str]] = Field(default_factory=list)
    fallback_used: bool = False


class PlanWarning(BaseModel):
    """Something the caller should know before trusting the plan."""

    code: str
    node_id: str = ""
    message: str
    # A machine-readable next step, empty when nothing has to be done.
    action: str = ""


class PlanPolicy(BaseModel):
    """The election policy that was applied, recorded with the result."""

    name: str
    free_admission_preference: bool
    stop_on_infeasible: bool


class RejectedNode(BaseModel):
    """A Mac that could not host rank 0, in the election's own words."""

    node_id: str
    reason: str
    attempts: int
    details: list[str]


class PlanExcludedNode(BaseModel):
    """A measured Mac the plan does not use, and why."""

    node_id: str
    reason: str


class PlanResourcePolicy(BaseModel):
    """The levers this plan was built with. Shown and applied on Start setup."""

    context_tokens: int
    target_context_tokens: int = 0
    table_offload: Literal["engram", "ple", "none"] = "none"
    table_kind: Literal["engram", "ple", ""] = ""
    moe_expert_offload: bool = False
    moe_resident_fraction: float = 1.0
    memory_guard_tier: str | None = None
    microbatch_size: int = 1
    admission: Literal["p05", "p50", "p95"] = "p50"


class PlanTradeoff(BaseModel):
    """One alternative the user may choose instead of the returned plan.

    Decision 2026-11-14: reducing context costs quality but keeps speed; routing
    experts to SSD keeps quality but costs speed. That is the user's call, so
    when the returned (quality-first) plan has to offload experts, the assistant
    also returns the best plan that keeps experts resident. Applying it is the
    normal ``apply`` with this ``resource_policy`` and ``plan_signature``.
    """

    label: Literal["speed", "quality"] = "speed"
    context_tokens: int
    plan_signature: str
    coordinator_node: str
    rank_order: list[str] = Field(default_factory=list)
    resource_policy: PlanResourcePolicy
    predicted_critical_stage_seconds: float | None = None
    #: The quality-first plan this is the alternative to (same measurement).
    against_context_tokens: int = 0
    against_moe_resident_fraction: float = 1.0


class MeasurementError(BaseModel):
    """Why a measurement job ended, and the one thing to do about it."""

    code: str
    message: str
    action: str


class MeasuredModelSummary(BaseModel):
    """One model the measured Mac's own catalogue listed.

    The free-space affordance names it by ``model_id`` and shows its size; 0
    means that Mac reported none. ``model_path`` is where that Mac holds it,
    which is the folder name its own delete route resolves.
    """

    model_id: str
    model_path: str
    size_bytes: int = 0


class MeasuredNodeSummary(BaseModel):
    """The plan-relevant part of one measured Mac, for the status poll.

    ``models`` is the model list that Mac's own catalogue reported while it was
    measured. The free-space affordance lists one Mac the user picked from these
    entries; a ``size_bytes`` of 0 means that Mac reported no size, and an empty
    list means no inventory was measured (never "this Mac holds nothing").
    """

    node_id: str
    admission_p05_gib: float
    endpoint_host: str = ""
    endpoint_port: int | None = None
    models: list[MeasuredModelSummary] = Field(default_factory=list)


class AssistantMeasurementStatus(BaseModel):
    """A minimal, read-only snapshot of one measurement job.

    No deployments, no health, no SSE: the UI polls this until ``state`` is
    ``ready`` and then plans. A failed job carries exactly one ``action``.
    """

    measurement_id: str
    state: str
    created_at: str
    updated_at: str
    measured_at: str
    model: str
    node_count: int
    nodes: list[MeasuredNodeSummary]
    warnings: list[PlanWarning]
    error: MeasurementError | None = None


class AssistantPlanResponse(BaseModel):
    """The plan object from PROJECT-BRIEF §13.3.

    An infeasible answer carries the same shape with ``feasible`` false, an
    empty ``coordinator_node``, no ``endpoint`` and no stages: nothing
    downstream can read a coordinator out of "there is no plan".
    """

    measurement_id: str
    measurement_timestamp: str
    plan_signature: str
    feasible: bool
    coordinator_node: str
    rank_order: list[str]
    groups: list[int]
    stages: list[PlanStage]
    reasons: list[str]
    warnings: list[PlanWarning]
    policy: PlanPolicy
    origin_node: str
    tensor_parallel_size: int
    rejected: list[RejectedNode]
    model: str
    context_tokens: int
    mtp: bool
    predicted: PlanPrediction
    observed: PlanObserved | None = None
    endpoint: PlanEndpoint | None = None
    #: The Macs this plan binds to, and the measured Macs it leaves out.
    selected_node_ids: list[str] = Field(default_factory=list)
    excluded_nodes: list[PlanExcludedNode] = Field(default_factory=list)
    #: The resolved levers, including a reduced context or a raised tier.
    resource_policy: PlanResourcePolicy | None = None
    #: The context-vs-experts tradeoff the user may choose instead. Empty when
    #: the returned plan already keeps experts resident (nothing to trade).
    tradeoffs: list[PlanTradeoff] = Field(default_factory=list)
    #: The measured line and whether this rank order can actually use it.
    topology: PlanTopology | None = None


def _plan_tradeoffs(
    selection: ClusterPlanSelection | None,
    alternatives: Sequence[ClusterPlanSelection],
) -> list[PlanTradeoff]:
    """The user-choosable alternatives to the returned plan (usually the speed one).

    Only an alternative that keeps routed experts resident is a real tradeoff; one
    that still offloads experts offers no speed win, so it is dropped. Empty means
    the returned plan already keeps experts resident and there is nothing to trade.
    """

    if selection is None:
        return []
    tradeoffs: list[PlanTradeoff] = []
    for alternative in alternatives:
        if alternative.policy.moe_expert_offload:
            continue
        policy = alternative.policy
        selected = alternative.election.selected
        tradeoffs.append(
            PlanTradeoff(
                label="speed",
                context_tokens=policy.context_tokens,
                plan_signature=alternative.election.plan_signature,
                coordinator_node=alternative.election.coordinator_node,
                rank_order=list(alternative.election.rank_order),
                resource_policy=PlanResourcePolicy(
                    context_tokens=policy.context_tokens,
                    target_context_tokens=policy.target_context_tokens,
                    table_offload=policy.table_offload,
                    table_kind=policy.table_kind,
                    moe_expert_offload=policy.moe_expert_offload,
                    moe_resident_fraction=policy.moe_resident_fraction,
                    memory_guard_tier=policy.memory_guard_tier,
                    microbatch_size=policy.microbatch_size,
                    admission=policy.admission,
                ),
                predicted_critical_stage_seconds=(
                    None if selected is None else selected.critical_stage_seconds
                ),
                against_context_tokens=selection.policy.context_tokens,
                against_moe_resident_fraction=selection.policy.moe_resident_fraction,
            )
        )
    return tradeoffs


def build_plan_response(
    result: ElectionResult,
    *,
    request: AssistantPlanRequest,
    measurement: MeasurementRecord,
    selection: ClusterPlanSelection | None = None,
    alternatives: Sequence[ClusterPlanSelection] = (),
) -> AssistantPlanResponse:
    """The §13.3 plan object for one election result.

    Pure: it reads the election result and the persisted measurement and adds
    only derived, non-scoring fields. The election's own numbers — order, cuts,
    headroom, signature — are passed through unchanged, and the endpoint's host
    and port come from the measurement rather than from any settings read here.
    """

    selected = result.selected
    endpoint = _endpoint(result, measurement)
    stages = (
        []
        if selected is None
        else [
            PlanStage(
                rank=stage.rank,
                node_id=stage.node_id,
                first_layer=stage.first_layer,
                last_layer=stage.last_layer,
                layer_count=stage.layer_count,
                resident_bytes=stage.resident_bytes,
                resident_gib=stage.resident_gib,
                admission_bytes=stage.admission_bytes,
                headroom_bytes=stage.headroom_bytes,
                headroom_gib=stage.headroom_gib,
                predicted_seconds=stage.predicted_seconds,
                critical=stage.critical,
            )
            for stage in selected.stages
        ]
    )
    critical = None if selected is None else selected.critical_stage
    critical_seconds = None if selected is None else selected.critical_stage_seconds
    return AssistantPlanResponse(
        measurement_id=measurement.measurement_id,
        # An empty timestamp says "unknown". Filling it with the time of this
        # request would date measurements that were taken at some other time.
        measurement_timestamp=measurement.measured_at,
        plan_signature=result.plan_signature,
        feasible=result.feasible,
        coordinator_node=result.coordinator_node,
        rank_order=list(result.rank_order),
        groups=[] if selected is None else list(selected.groups),
        stages=stages,
        reasons=list(result.reasons),
        warnings=_warnings(
            result,
            measurement,
            endpoint,
            selection,
            fallback=request.confirm_two_mac_fallback,
        ),
        policy=PlanPolicy(
            name=result.policy.name,
            free_admission_preference=result.policy.free_admission_preference,
            stop_on_infeasible=result.policy.stop_on_infeasible,
        ),
        origin_node=result.origin_node,
        tensor_parallel_size=result.tensor_parallel_size,
        rejected=[
            RejectedNode(
                node_id=item.node_id,
                reason=item.reason,
                attempts=item.attempts,
                details=list(item.details),
            )
            for item in result.rejected
        ],
        model=measurement.model.label,
        context_tokens=(
            selection.policy.context_tokens
            if selection is not None
            else request.context_tokens
        ),
        mtp=request.mtp,
        predicted=PlanPrediction(
            measured=critical_seconds is not None,
            critical_stage_rank=None if critical is None else critical.rank,
            critical_stage_seconds=critical_seconds,
            stages=[
                PlanStageSeconds(
                    rank=stage.rank,
                    node_id=stage.node_id,
                    seconds=stage.predicted_seconds,
                )
                for stage in stages
            ],
        ),
        # Filled by the smoke test in a later phase, never assumed here.
        observed=None,
        endpoint=endpoint,
        selected_node_ids=list(result.rank_order),
        excluded_nodes=(
            []
            if selection is None
            else [
                PlanExcludedNode(node_id=item.node_id, reason=item.reason)
                for item in selection.excluded_nodes
            ]
        ),
        topology=_plan_topology(
            result,
            measurement,
            fallback_used=request.confirm_two_mac_fallback,
        ),
        resource_policy=(
            None
            if selection is None
            else PlanResourcePolicy(
                context_tokens=selection.policy.context_tokens,
                target_context_tokens=selection.policy.target_context_tokens,
                table_offload=selection.policy.table_offload,
                table_kind=selection.policy.table_kind,
                moe_expert_offload=selection.policy.moe_expert_offload,
                moe_resident_fraction=selection.policy.moe_resident_fraction,
                memory_guard_tier=selection.policy.memory_guard_tier,
                microbatch_size=selection.policy.microbatch_size,
                admission=selection.policy.admission,
            )
        ),
        tradeoffs=_plan_tradeoffs(selection, alternatives),
    )


def measurement_status(measurement: MeasurementRecord) -> AssistantMeasurementStatus:
    """The status payload for one stored measurement. Reads only."""

    nodes = measurement.measurement_set().nodes
    summaries: list[MeasuredNodeSummary] = []
    for node in nodes:
        endpoint = measurement.endpoint_for(node.node_id)
        summaries.append(
            MeasuredNodeSummary(
                node_id=node.node_id,
                admission_p05_gib=node.admission_p05_gib,
                endpoint_host=node.endpoint_host,
                endpoint_port=None if endpoint is None else endpoint.port or None,
                models=[
                    MeasuredModelSummary(
                        model_id=str(item.get("model_id") or ""),
                        model_path=str(item.get("model_path") or ""),
                        size_bytes=int(item.get("size_bytes") or 0),
                    )
                    for item in measurement.inventories.get(node.node_id, ())
                ],
            )
        )
    return AssistantMeasurementStatus(
        measurement_id=measurement.measurement_id,
        state=measurement.state,
        created_at=measurement.created_at,
        updated_at=measurement.updated_at,
        measured_at=measurement.measured_at,
        model=measurement.model.label,
        node_count=measurement.node_count,
        nodes=summaries,
        warnings=[
            PlanWarning(
                code=warning.code,
                node_id=warning.node_id,
                message=warning.message,
            )
            for warning in measurement.warnings
        ],
        error=(
            None
            if measurement.error is None
            else MeasurementError(
                code=measurement.error.code,
                message=measurement.error.message,
                action=measurement.error.action,
            )
        ),
    )


def _endpoint(
    result: ElectionResult, measurement: MeasurementRecord
) -> PlanEndpoint | None:
    """The elected Mac's endpoint, or ``None`` when it cannot be dialled.

    The host and port are the measurement's, not the election's: the election
    names the owner, and only the measurement knows what that Mac was
    reachable on. A partial answer (a host without a port, a loopback address)
    is not upgraded to a guess.
    """

    if not result.feasible or not result.coordinator_node:
        return None
    endpoint = measurement.endpoint_for(result.coordinator_node)
    if endpoint is None or not endpoint.resolved:
        return None
    return PlanEndpoint(
        node=result.coordinator_node,
        host=endpoint.host,
        port=endpoint.port,
        url=f"http://{endpoint.host}:{endpoint.port}",
        # Only a health check after start may say otherwise.
        ready=False,
    )


def _warnings(
    result: ElectionResult,
    measurement: MeasurementRecord,
    endpoint: PlanEndpoint | None,
    selection: ClusterPlanSelection | None = None,
    *,
    fallback: bool = False,
) -> list[PlanWarning]:
    """Derived caveats, in English, about what the plan could not measure.

    The measurement's own warnings are carried through unchanged: they are the
    only record of which Mac lacked a probe or a ceiling, and dropping them
    here would let the plan look better measured than it is.
    """

    warnings = [
        PlanWarning(code=warning.code, node_id=warning.node_id, message=warning.message)
        for warning in measurement.warnings
    ]
    if selection is not None:
        warnings.extend(
            PlanWarning(code=item.code, node_id=item.node_id, message=item.message)
            for item in selection.warnings
        )
    if fallback:
        warnings.append(
            PlanWarning(
                code="two_mac_fallback",
                message=(
                    "the measured fabric is incomplete for the requested Mac "
                    "set; this plan runs on two Macs, as you confirmed"
                ),
            )
        )
    selected = result.selected
    if selected is not None and selected.critical_stage_seconds is None:
        warnings.append(
            PlanWarning(
                code=WARNING_UNMEASURED_PLAN,
                node_id=result.coordinator_node,
                message=(
                    "no stage was measured, so no speed is predicted for this plan"
                ),
            )
        )
    if result.feasible and endpoint is None:
        warnings.append(
            PlanWarning(
                code=WARNING_ENDPOINT_UNRESOLVED,
                node_id=result.coordinator_node,
                message=(
                    "the elected Mac reported no reachable address and port; bind "
                    "oMLX to this cluster's private interface and measure again"
                ),
                action=ACTION_CONFIGURE_ENDPOINT,
            )
        )
    return warnings


def _plan_topology(
    result: ElectionResult,
    measurement: MeasurementRecord,
    *,
    fallback_used: bool,
) -> PlanTopology:
    """The measured line for this rank order, or the unknown-fabric state."""

    if not measurement.fabric_measured:
        return PlanTopology(measured=False, complete=None, fallback_used=fallback_used)
    missing = missing_line_links(result.rank_order, measurement.fabric_links)
    return PlanTopology(
        measured=True,
        complete=not missing,
        missing_links=[[first, second] for first, second in missing],
        fallback_used=fallback_used,
    )


def plan_error_payload(code: str, message: str, action: str) -> dict[str, Any]:
    """A machine-readable refusal body with exactly one next step.

    Used for both 409s and the 404, so a client distinguishes "measure again"
    from "no plan exists" from "unknown measurement" without parsing prose.
    """

    return {"code": code, "message": message, "action": action}


# ---------------------------------------------------------------------------
# recommendation (PROJECT-BRIEF §11.1a/§13.3a)
# ---------------------------------------------------------------------------

# No candidate model fits these Macs. Distinct from ``no_feasible_plan``: this
# one is answered by another model, not by another context for the same one.
NO_FEASIBLE_CANDIDATE_CODE = "no_feasible_candidate"


class AssistantRecommendCandidate(BaseModel):
    """One model the user might run on the measured Macs.

    The same shape the model shelf uses: a path this Mac can read plus the
    friendly name and declared context the UI already knows. A candidate whose
    layout cannot be read is reported with a reason — it is never dropped.
    """

    model_config = ConfigDict(extra="forbid")

    model_path: str = Field(min_length=1, max_length=4096)
    friendly_name: str | None = Field(default=None, max_length=256)
    declared_context_tokens: int | None = Field(
        default=None, ge=1, le=MAX_CONTEXT_TOKENS
    )


class AssistantRecommendRequest(BaseModel):
    """Which model should these already-measured Macs run?

    Only the hardware half of the measurement is read (§11.1a), so the candidate
    models do not have to include the checkpoint the measurement covered. The
    model the user picks is measured again (model-bound) before ``plan``.
    """

    model_config = ConfigDict(extra="forbid")

    measurement_id: str = Field(min_length=1, max_length=128)
    candidates: list[AssistantRecommendCandidate] = Field(
        min_length=1, max_length=MAX_RECOMMENDATION_CANDIDATES
    )
    # The context the user wants. Recommendation answers at the largest context
    # it can honour at or below this, never above it.
    context_tokens: int | None = Field(default=None, ge=1, le=MAX_CONTEXT_TOKENS)
    execution_profile: Literal["interactive", "balanced", "throughput"] = "balanced"


class RecommendedNode(BaseModel):
    """What one Mac needs before it can load this candidate's stage."""

    node_id: str
    model_present: bool
    stage_bytes: int
    #: ``None`` is unknown free space, not zero.
    disk_free_bytes: int | None
    can_stage: bool


class RecommendedStage(BaseModel):
    """One pipeline stage of the recommended split."""

    rank: int
    node_id: str
    first_layer: int
    last_layer: int
    layer_count: int
    resident_bytes: int
    resident_gib: float
    admission_bytes: int
    headroom_bytes: int
    headroom_gib: float
    predicted_seconds: float | None
    critical: bool


class RecommendationWarning(BaseModel):
    """Something the caller should know before choosing this model."""

    code: str
    node_id: str = ""
    message: str
    # Empty when nothing has to be done; the copy path is not a fault.
    action: str = ""


class RecommendedModel(BaseModel):
    """One candidate's verdict, in the shape §13.3a asks for.

    There is no token rate anywhere: ``critical_stage_seconds`` is ``None`` when
    no compute was measured, and ``measured_speed`` says which of the two this
    is. A recommendation is a model selection, not a launch promise.
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
    download_gib: float
    max_context_tokens: int
    declared_context_tokens: int | None
    context_limited: bool
    headroom_bytes: int
    headroom_gib: float
    rank0_node: str
    rank_order: list[str]
    stages: list[RecommendedStage]
    nodes: list[RecommendedNode]
    critical_stage_rank: int | None
    critical_stage_seconds: float | None
    measured_speed: bool
    plan_signature: str
    failure_kind: str
    reasons: list[str]
    warnings: list[RecommendationWarning]
    # One next step when this candidate cannot be chosen, else "".
    action: str
    #: The levers this candidate would use, and the measured Macs it leaves out.
    resource_policy: PlanResourcePolicy | None = None
    excluded_nodes: list[str] = Field(default_factory=list)
    #: "single" (local engine on one Mac) or "cluster" (pipeline across 2+).
    mode: str = "cluster"


class AssistantRecommendResponse(BaseModel):
    """The §13.3a recommendation object for one measured cluster.

    ``selected_model`` names the one model the primary path should offer; the
    ordered ``candidates`` list is what an advanced shelf shows. No feasible
    candidate is a 409 carrying this same body plus ``code``/``action``/
    ``message``, so the near misses come with the refusal.
    """

    measurement_id: str
    measurement_timestamp: str
    measurement_model: str
    # No recommendation may require a model identity, but ``measure`` still
    # needs one (Chapter 3), so a model-less measurement cannot exist yet.
    hardware_only: bool
    target_context_tokens: int | None
    mtp: bool
    execution_profile: str
    feasible: bool
    selected_model: str
    selected_context_tokens: int | None
    candidates: list[RecommendedModel]
    reasons: list[str]
    warnings: list[RecommendationWarning]
    #: The single-node option for the selected model (E6): when it fits on one
    #: Mac at the target context, the local engine is faster than a pipeline.
    #: ``None`` when it does not fit on one Mac. Additive to the cluster answer.
    single_node: RecommendedModel | None = None


def build_recommend_response(
    result: RecommendationResult,
    *,
    request: AssistantRecommendRequest,
    measurement: MeasurementRecord,
) -> AssistantRecommendResponse:
    """The recommendation object for one pure recommendation result."""

    selected = result.selected
    return AssistantRecommendResponse(
        measurement_id=measurement.measurement_id,
        measurement_timestamp=measurement.measured_at,
        measurement_model=measurement.model.label,
        hardware_only=measurement.model.hardware_only,
        target_context_tokens=result.target_context_tokens,
        mtp=result.mtp,
        execution_profile=result.execution_profile,
        feasible=result.feasible,
        selected_model="" if selected is None else selected.model_id,
        selected_context_tokens=None
        if selected is None
        else selected.max_context_tokens,
        candidates=[_recommended_model(item) for item in result.candidates],
        reasons=list(result.reasons),
        # The caveats that apply to the recommendation itself, i.e. to the model
        # it names. Per-candidate warnings stay on each candidate.
        warnings=(
            [] if selected is None else [_warning(item) for item in selected.warnings]
        ),
        single_node=(
            None if result.single_node is None else _recommended_model(result.single_node)
        ),
    )


def _recommended_model(item: ModelRecommendation) -> RecommendedModel:
    return RecommendedModel(
        model_id=item.model_id,
        label=item.label,
        model_path=item.model_path,
        feasible=item.feasible,
        pp_supported=item.pp_supported,
        pipeline_stages=item.pipeline_stages,
        nodes_required=item.nodes_required,
        weight_bytes=item.weight_bytes,
        download_bytes=item.download_bytes,
        download_gib=item.download_gib,
        max_context_tokens=item.max_context_tokens,
        declared_context_tokens=item.declared_context_tokens,
        context_limited=item.context_limited,
        headroom_bytes=item.headroom_bytes,
        headroom_gib=item.headroom_gib,
        rank0_node=item.rank0_node,
        rank_order=list(item.rank_order),
        stages=[
            RecommendedStage(
                rank=stage.rank,
                node_id=stage.node_id,
                first_layer=stage.first_layer,
                last_layer=stage.last_layer,
                layer_count=stage.layer_count,
                resident_bytes=stage.resident_bytes,
                resident_gib=stage.resident_gib,
                admission_bytes=stage.admission_bytes,
                headroom_bytes=stage.headroom_bytes,
                headroom_gib=stage.headroom_gib,
                predicted_seconds=stage.predicted_seconds,
                critical=stage.critical,
            )
            for stage in item.stages
        ],
        nodes=[
            RecommendedNode(
                node_id=node.node_id,
                model_present=node.model_present,
                stage_bytes=node.stage_bytes,
                disk_free_bytes=node.disk_free_bytes,
                can_stage=node.can_stage,
            )
            for node in item.nodes
        ],
        critical_stage_rank=item.critical_stage_rank,
        critical_stage_seconds=item.critical_stage_seconds,
        measured_speed=item.measured_speed,
        plan_signature=item.plan_signature,
        failure_kind=item.failure_kind,
        reasons=list(item.reasons),
        warnings=[_warning(warning) for warning in item.warnings],
        action=item.action,
        resource_policy=(
            None
            if item.resource_policy is None
            else PlanResourcePolicy(
                context_tokens=item.resource_policy.context_tokens,
                target_context_tokens=item.resource_policy.target_context_tokens,
                table_offload=item.resource_policy.table_offload,
                table_kind=item.resource_policy.table_kind,
                moe_expert_offload=item.resource_policy.moe_expert_offload,
                moe_resident_fraction=item.resource_policy.moe_resident_fraction,
                memory_guard_tier=item.resource_policy.memory_guard_tier,
                microbatch_size=item.resource_policy.microbatch_size,
                admission=item.resource_policy.admission,
            )
        ),
        excluded_nodes=list(item.excluded_nodes),
        mode=item.mode,
    )


def _warning(item: RecommendationWarningData) -> RecommendationWarning:
    return RecommendationWarning(
        code=item.code,
        node_id=item.node_id,
        message=item.message,
        action=item.action,
    )


# ---------------------------------------------------------------------------
# apply / start / operation status (PROJECT-BRIEF §13.2, CHAPTER-5.md §4)
# ---------------------------------------------------------------------------

OPERATIONS_PATH = "/admin/api/cluster/assistant/operations"


class AssistantApplyRequest(BaseModel):
    """Approve one elected plan for staging. Starts no inference.

    Everything measured — the Macs, admission, compute, links, endpoints, model
    and MTP setting — comes from the stored measurement. The fields here are the
    plan-time inputs that change the election without changing what was
    measured, plus the signature the caller approved: ``apply`` re-elects and
    refuses anything that does not produce the same signature, so the request is
    a claim to be checked, not a stage table to be obeyed.
    """

    model_config = ConfigDict(extra="forbid")

    measurement_id: str = Field(min_length=1, max_length=128)
    # The election's plan hash from ``/plan``.
    plan_signature: str = Field(min_length=16, max_length=64)
    model_path: str | None = Field(default=None, max_length=4096)
    model_size_bytes: int | None = Field(default=None, gt=0, le=1 << 60)
    layer_count: int = Field(default=80, gt=0, le=MAX_LAYER_COUNT)
    context_tokens: int = Field(default=8192, ge=1, le=MAX_CONTEXT_TOKENS)
    rank0_extra_bytes: int = Field(default=0, ge=0, le=1 << 60)
    mtp: bool = False
    execution_profile: Literal["interactive", "balanced", "throughput"] = "balanced"
    microbatch_size: int | None = Field(default=None, ge=1, le=MAX_MICROBATCH)
    # The plan's own answer, carried for validation only: a request that names a
    # different coordinator or order than the election produces is refused.
    coordinator_node: str = Field(min_length=1, max_length=128)
    rank_order: list[str] = Field(min_length=1, max_length=MAX_REQUEST_NODES)
    # Where the user was sitting. Recorded for diagnostics; never a fallback.
    origin_node: str | None = Field(default=None, max_length=255)
    # The resource policy the plan approved. It is repeated on apply so the
    # election re-proves the same levers; a drift is ``plan_mismatch``.
    target_context_tokens: int | None = Field(default=None, ge=1, le=MAX_CONTEXT_TOKENS)
    table_offload: Literal["engram", "ple", "none"] = "none"
    table_kind: Literal["engram", "ple", ""] = ""
    moe_expert_offload: bool = False
    moe_resident_fraction: float = Field(default=1.0, gt=0, le=1)
    memory_guard_tier: Literal["safe", "balanced", "aggressive", "custom"] | None = (
        None
    )
    admission: Literal["p05", "p50", "p95"] = "p50"
    #: Carried from the plan inputs for compatibility; the approved rank order,
    #: not this field, decides the subset on apply.
    node_selection: Literal["minimal", "all"] = "minimal"
    #: The user's explicit consent to the 2-Mac fallback; it is part of the
    #: signature that apply re-proves.
    confirm_two_mac_fallback: bool = False

    def model_identity(self) -> ModelIdentity:
        return request_model_identity(
            model_path=self.model_path,
            model_size_bytes=self.model_size_bytes,
            layer_count=self.layer_count,
            mtp=self.mtp,
        )

    def resource_policy(self) -> ResourcePolicy:
        """The approved policy, in the one type plan/apply/start share."""

        return ResourcePolicy(
            context_tokens=self.context_tokens,
            target_context_tokens=self.target_context_tokens or self.context_tokens,
            table_offload=self.table_offload,
            table_kind=self.table_kind,
            moe_expert_offload=self.moe_expert_offload,
            moe_resident_fraction=self.moe_resident_fraction,
            memory_guard_tier=self.memory_guard_tier,
            microbatch_size=self.microbatch_size or 1,
            admission=self.admission,
        )

    def plan_inputs(self) -> PlanInputs:
        """The pure input bundle ``provisioning`` re-elects from."""

        return PlanInputs(
            measurement_id=self.measurement_id,
            plan_signature=self.plan_signature,
            model=self.model_identity(),
            context_tokens=self.context_tokens,
            execution_profile=self.execution_profile,
            rank0_extra_bytes=self.rank0_extra_bytes,
            mtp=self.mtp,
            coordinator_node=self.coordinator_node,
            rank_order=tuple(self.rank_order),
            microbatch_size=self.microbatch_size or 1,
            origin_node=self.origin_node or "",
            target_context_tokens=self.target_context_tokens,
            resource_policy=self.resource_policy(),
            two_mac_fallback=self.confirm_two_mac_fallback,
        )


class AssistantStartRequest(BaseModel):
    """Start the deployment ``apply`` staged. Never stages anything itself."""

    model_config = ConfigDict(extra="forbid")

    measurement_id: str = Field(min_length=1, max_length=128)
    plan_signature: str = Field(min_length=16, max_length=64)
    coordinator_node: str = Field(min_length=1, max_length=128)
    # The apply operation whose staged deployment this starts. When omitted the
    # one applied operation for this measurement and signature is used, and a
    # missing one is a 409 with one action.
    operation_id: str | None = Field(default=None, max_length=128)


class AssistantPerformanceProbeRequest(BaseModel):
    """The in-app switch for the post-ready performance measurement."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool


class AssistantStopRequest(BaseModel):
    """Stop the deployment one ready start operation launched.

    The operation is the only input: it carries the elected coordinator, the
    port and the deployment ID the coordinator assigned, so the route never has
    to guess a target or re-elect from a possibly stale measurement. Stopping is
    idempotent.
    """

    model_config = ConfigDict(extra="forbid")

    operation_id: str = Field(min_length=1, max_length=128)


class AssistantStopResponse(BaseModel):
    """The one deployment a stop asked the coordinator to unload."""

    operation_id: str
    deployment_id: str
    stopped: bool
    warnings: list[OperationWarningView]


class AssistantOperationHandle(BaseModel):
    """What a started apply/start returns: a handle, not a result."""

    operation_id: str
    state: str
    poll: str


class AssistantSessionResponse(BaseModel):
    """A reloaded card's last deployment, with enough inputs to start again.

    No key, SSH target or host is stored: only the approved plan inputs and the
    coordinator/deployment identity the operation record already holds.
    """

    operation_id: str
    kind: str
    state: str
    stopped: bool = False
    measurement_id: str
    plan_signature: str
    model_path: str = ""
    context_tokens: int = 0
    target_context_tokens: int = 0
    rank0_extra_bytes: int = 0
    mtp: bool = False
    execution_profile: str = "balanced"
    microbatch_size: int = 1
    coordinator_node: str
    rank_order: list[str]
    #: The explicit 2-Mac fallback consent, so Start again re-proves the same
    #: signature after a reload.
    two_mac_fallback: bool = False
    resource_policy: PlanResourcePolicy | None = None
    deployment_id: str = ""
    endpoint: OperationEndpointView | None = None
    updated_at: str


class AssistantDiagnosticsResponse(BaseModel):
    """One redacted support bundle. Read-only; a missing record is not an error."""

    generated_at: str
    app: dict[str, str]
    measurement: dict[str, Any] | None = None
    operation: dict[str, Any] | None = None
    plan: dict[str, Any] | None = None
    topology: dict[str, Any] | None = None
    health: dict[str, Any] | None = None
    events: list[dict[str, Any]] = Field(default_factory=list)
    redactions: list[str] = Field(default_factory=list)


class NetworkPreflightHost(BaseModel):
    """One host's read-only addressing evidence."""

    host: str
    bridges: list[str] = Field(default_factory=list)
    thunderbolt_interfaces: list[str] = Field(default_factory=list)
    rdma_interfaces: list[str] = Field(default_factory=list)
    addresses: list[dict[str, Any]] = Field(default_factory=list)
    self_assigned_interfaces: list[str] = Field(default_factory=list)
    link_local_addresses: dict[str, str] = Field(default_factory=dict)
    media_speeds: dict[str, int] = Field(default_factory=dict)


class NetworkPreflightResponse(BaseModel):
    """A read-only bridge plan; ``executed`` is always false."""

    generated_at: str
    hosts: list[NetworkPreflightHost]
    commands: list[str]
    warnings: list[str]
    confirmation_required: bool = True
    executed: bool = False


class OperationProgressView(BaseModel):
    """Progress a UI can render without inventing anything."""

    phase: str
    message: str
    nodes_ready: int
    nodes_total: int


class OperationEndpointView(BaseModel):
    """The coordinator's endpoint. ``ready`` is false until health and smoke."""

    node: str
    host: str
    port: int = Field(ge=1, le=65535)
    url: str
    ready: bool


class OperationObservedView(BaseModel):
    """The one measured rate block, or ``None`` on the operation.

    Filled only after ``ready`` by a real completion the coordinator served.
    A client must render it as measured and timestamped; it is never a promise
    and never fills ``predicted``.
    """

    prefill_tok_per_s: float | None = None
    decode_tok_per_s: float | None = None
    ttft_ms: float | None = None
    measured_at: str = ""


class OperationWarningView(BaseModel):
    code: str
    node_id: str = ""
    message: str
    action: str = ""


class OperationErrorView(BaseModel):
    """Why an operation failed, and the one thing to do about it."""

    code: str
    message: str
    action: str


class AssistantOperationStatus(BaseModel):
    """The operation status object (CHAPTER-5.md §4.3).

    Deliberately separate from the measurement status: a measurement job can be
    ready while no deployment exists, and an operation can be ready while its
    measurement has since expired. There is no token-rate field anywhere: the
    operation reports readiness, never throughput.
    """

    operation_id: str
    kind: str
    state: str
    created_at: str
    updated_at: str
    measurement_id: str
    plan_signature: str
    placement_signature: str
    coordinator_node: str
    #: Whether the coordinator was this Mac; the stop route uses this record.
    coordinator_local: bool = False
    rank_order: list[str]
    context_tokens: int
    rank0_extra_bytes: int
    microbatch_size: int
    mtp: bool
    execution_profile: str
    #: The approved resource policy (engram/ple, MoE offload, tier, context).
    resource_policy: PlanResourcePolicy | None = None
    progress: OperationProgressView
    endpoint: OperationEndpointView | None = None
    #: The coordinator's deployment ID, known after a start reports ready.
    deployment_id: str = ""
    #: True after a successful stop; ``state`` remains ``ready``.
    stopped: bool = False
    #: Measured after ``ready``, or ``None`` when no probe ran/finished.
    observed: OperationObservedView | None = None
    warnings: list[OperationWarningView]
    error: OperationErrorView | None = None


def operation_handle(record: OperationRecord) -> AssistantOperationHandle:
    """The handle an apply/start request returns immediately."""

    return AssistantOperationHandle(
        operation_id=record.operation_id,
        state=record.state,
        poll=f"{OPERATIONS_PATH}/{record.operation_id}",
    )


def operation_status(record: OperationRecord) -> AssistantOperationStatus:
    """The status payload for one persisted operation. Reads only."""

    return AssistantOperationStatus(
        operation_id=record.operation_id,
        kind=record.kind,
        state=record.state,
        created_at=record.created_at,
        updated_at=record.updated_at,
        measurement_id=record.measurement_id,
        plan_signature=record.plan_signature,
        placement_signature=record.placement_signature,
        coordinator_node=record.coordinator_node,
        coordinator_local=record.coordinator_local,
        rank_order=list(record.rank_order),
        context_tokens=record.context_tokens,
        rank0_extra_bytes=record.rank0_extra_bytes,
        microbatch_size=record.microbatch_size,
        mtp=record.mtp,
        execution_profile=record.execution_profile,
        resource_policy=_policy_view(record.policy),
        progress=_progress_view(record.progress),
        endpoint=_endpoint_view(record.endpoint),
        deployment_id=record.deployment_id,
        stopped=record.stopped,
        observed=_observed_view(record.observed),
        warnings=[_operation_warning(item) for item in record.warnings],
        error=(
            None
            if record.error is None
            else OperationErrorView(
                code=record.error.code,
                message=record.error.message,
                action=record.error.action,
            )
        ),
    )


def _progress_view(progress: OperationProgress) -> OperationProgressView:
    return OperationProgressView(
        phase=progress.phase,
        message=progress.message,
        nodes_ready=progress.nodes_ready,
        nodes_total=progress.nodes_total,
    )


def _policy_view(policy: Any) -> PlanResourcePolicy | None:
    if not policy:
        return None
    return PlanResourcePolicy(
        context_tokens=int(policy.get("context_tokens") or 0),
        target_context_tokens=int(policy.get("target_context_tokens") or 0),
        table_offload=str(policy.get("table_offload") or "none"),
        table_kind=str(policy.get("table_kind") or ""),
        moe_expert_offload=bool(policy.get("moe_expert_offload", False)),
        moe_resident_fraction=float(policy.get("moe_resident_fraction") or 1.0),
        memory_guard_tier=policy.get("memory_guard_tier"),
        microbatch_size=int(policy.get("microbatch_size") or 1),
        admission=str(policy.get("admission") or "p50"),
    )


def session_view(record: OperationRecord) -> AssistantSessionResponse:
    """The last deployment a reloaded card can show, with no re-election."""

    return AssistantSessionResponse(
        operation_id=record.operation_id,
        kind=record.kind,
        state=record.state,
        stopped=record.stopped,
        measurement_id=record.measurement_id,
        plan_signature=record.plan_signature,
        model_path=record.model_path,
        context_tokens=record.context_tokens,
        target_context_tokens=record.target_context_tokens or record.context_tokens,
        rank0_extra_bytes=record.rank0_extra_bytes,
        mtp=record.mtp,
        execution_profile=record.execution_profile,
        microbatch_size=record.microbatch_size,
        coordinator_node=record.coordinator_node,
        rank_order=list(record.rank_order),
        two_mac_fallback=record.two_mac_fallback,
        resource_policy=_policy_view(record.policy),
        deployment_id=record.deployment_id,
        endpoint=_endpoint_view(record.endpoint),
        updated_at=record.updated_at,
    )


def _observed_view(observed: OperationObserved | None) -> OperationObservedView | None:
    if observed is None:
        return None
    return OperationObservedView(
        prefill_tok_per_s=observed.prefill_tok_per_s,
        decode_tok_per_s=observed.decode_tok_per_s,
        ttft_ms=observed.ttft_ms,
        measured_at=observed.measured_at,
    )


def _endpoint_view(endpoint: OperationEndpoint | None) -> OperationEndpointView | None:
    if endpoint is None:
        return None
    return OperationEndpointView(
        node=endpoint.node,
        host=endpoint.host,
        port=endpoint.port,
        url=endpoint.url,
        ready=endpoint.ready,
    )


def _operation_warning(item: OperationWarning) -> OperationWarningView:
    return OperationWarningView(
        code=item.code,
        node_id=item.node_id,
        message=item.message,
        action=item.action,
    )
