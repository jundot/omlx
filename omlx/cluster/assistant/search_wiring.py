# SPDX-License-Identifier: Apache-2.0
"""Wire the search loop's evaluate seam to the real apply/start/observe seams.

``evaluate_constellation`` is the loop body: it deploys ONE candidate
constellation through the existing oMLX path and measures it. It reuses
``plan_apply`` (oMLX's planner still owns the layer split), ``run_apply_job``,
``run_start_job`` and ``observe_performance`` — nothing here re-implements
staging, activation or the split. The candidate's rank order is forced through
``PlanInputs.rank_order``; ``elect_for_apply`` re-proves it and a drifted plan is
a ``plan_mismatch`` (recorded infeasible, the loop continues).

The result is a ``ConstellationResult`` carrying the REAL measured throughput
(oMLX usage fields), which ``evaluation.select_best`` ranks. A deployment that
fails to apply/start is infeasible; one that starts but reports no timing is
feasible-but-unmeasured (never preferred over a measured one).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping
from typing import Any

from .election import NoFeasiblePlanError, PlanCandidate
from .evaluation import ConstellationResult
from .operations import (
    KIND_APPLY,
    KIND_START,
    STATE_APPLIED,
    STATE_READY,
    OperationStore,
)
from .provisioning import PlanInputs, ProvisioningError, plan_apply
from .start import observe_performance, run_apply_job, run_start_job
from .start import _serving_model_id


async def evaluate_constellation(
    candidate: PlanCandidate,
    *,
    record: Any,
    store: OperationStore,
    seams: Any,
    ops_for: Callable[[Any], Any],
    mtp: bool = False,
    execution_profile: str = "balanced",
    rank0_extra_bytes: int = 0,
    microbatch_size: int | None = None,
    resource_policy: Any | None = None,
    performance_store: Any | None = None,
    model_settings_patch: Mapping[str, Any] | None = None,
    probe_enabled: bool = False,
    sleep: Callable[[float], None] = time.sleep,
) -> ConstellationResult:
    """Apply + start + measure one candidate; return its measured result.

    Never raises for a deployment refusal: a plan/apply/start failure becomes an
    infeasible (or unmeasured) result so the search loop can continue. Only an
    unexpected error propagates (the loop records it).
    """

    base = dict(
        rank_order=candidate.rank_order,
        model_path=record.model.model_path,
        context_tokens=candidate.context_tokens,
        plan_signature=candidate.plan_hash,
    )
    inputs = PlanInputs(
        measurement_id=record.measurement_id,
        plan_signature=candidate.plan_hash,
        model=record.model,
        context_tokens=candidate.context_tokens,
        execution_profile=execution_profile,
        rank0_extra_bytes=rank0_extra_bytes,
        mtp=mtp,
        coordinator_node=candidate.rank_order[0] if candidate.rank_order else "",
        rank_order=candidate.rank_order,
        microbatch_size=microbatch_size,
        target_context_tokens=candidate.context_tokens,
        resource_policy=(
            candidate.resource_policy
            if candidate.resource_policy is not None
            else resource_policy
        ),
    )
    try:
        plan = plan_apply(record, inputs, seams=seams)
    except (ProvisioningError, NoFeasiblePlanError):
        return ConstellationResult(feasible=False, **base)

    ops = ops_for(plan)
    if model_settings_patch:
        # A redeploy tuning config is a per-model setting (the MTP depth), so it
        # must be on the coordinator *before* the launch this block starts. A
        # failed write would measure the wrong config, so the row is dropped
        # rather than silently measuring the baseline.
        applied_patch = await _write_model_settings(
            ops, record.model.model_path, model_settings_patch
        )
        if not applied_patch:
            return ConstellationResult(feasible=False, **base)
    apply_op = store.start(
        kind=KIND_APPLY,
        measurement_id=record.measurement_id,
        plan_signature=candidate.plan_hash,
        coordinator_node=plan.coordinator.node_id,
        coordinator_local=plan.coordinator.local,
        rank_order=plan.rank_order,
        context_tokens=candidate.context_tokens,
        target_context_tokens=candidate.context_tokens,
        model_path=record.model.model_path,
        rank0_extra_bytes=rank0_extra_bytes,
        microbatch_size=microbatch_size or 1,
        mtp=mtp,
        execution_profile=execution_profile,
        policy=(resource_policy.to_dict() if resource_policy is not None else None),
        endpoint=plan.endpoint,
    )
    await asyncio.to_thread(
        run_apply_job,
        store,
        apply_op.operation_id,
        plan,
        ops,
        sleep=sleep,
        performance_store=performance_store,
    )
    applied = store.get(apply_op.operation_id)
    if applied is None or applied.state != STATE_APPLIED:
        return ConstellationResult(feasible=False, **base)

    start_op = store.start(
        kind=KIND_START,
        measurement_id=record.measurement_id,
        plan_signature=applied.plan_signature,
        coordinator_node=applied.coordinator_node,
        coordinator_local=applied.coordinator_local,
        rank_order=applied.rank_order,
        context_tokens=applied.context_tokens,
        target_context_tokens=applied.target_context_tokens,
        model_path=applied.model_path,
        rank0_extra_bytes=applied.rank0_extra_bytes,
        microbatch_size=applied.microbatch_size,
        mtp=applied.mtp,
        execution_profile=applied.execution_profile,
        two_mac_fallback=applied.two_mac_fallback,
        policy=dict(applied.policy),
        endpoint=plan.endpoint,
    )
    await run_start_job(
        store,
        start_op.operation_id,
        plan.request,
        ops,
        expected_placement=applied.placement_signature,
        probe_enabled=probe_enabled,
    )
    ready = store.get(start_op.operation_id)
    if ready is None or ready.state != STATE_READY:
        # Deployed but not ready: feasible, yet nothing measured to rank by.
        return ConstellationResult(feasible=True, **base)

    observed = await observe_performance(
        store, start_op.operation_id, ops, record.model.model_path
    )
    return ConstellationResult(
        feasible=True,
        decode_tok_per_s=observed.decode_tok_per_s if observed else None,
        prefill_tok_per_s=observed.prefill_tok_per_s if observed else None,
        ttft_ms=observed.ttft_ms if observed else None,
        deployment_id=ready.deployment_id,
        coordinator_node=ready.coordinator_node,
        operation_id=start_op.operation_id,
        **base,
    )

async def _write_model_settings(
    ops: Any, model_path: str, patch: Mapping[str, Any]
) -> bool:
    """Write one config's model settings onto the coordinator before launch.

    A redeploy tuning config is a per-model setting (the MTP adaptive depth), so
    it has to be in place before the deployment starts. Returns whether the
    write landed; a config whose write fails must not be measured as if it were
    the baseline.
    """

    if ops is None or ops.models is None or ops.write_model_settings is None:
        return False
    try:
        listing = await asyncio.to_thread(ops.models)
        model_id = _serving_model_id(listing, model_path)
        if not model_id:
            return False
        await asyncio.to_thread(ops.write_model_settings, model_id, dict(patch))
    except Exception:
        return False
    return True
