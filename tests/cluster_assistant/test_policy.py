# SPDX-License-Identifier: Apache-2.0
"""Phase A: the resource policy, the subset solver and the levers.

These tests are fake-driven on purpose. The decision is a pure function of
measured facts and a checkpoint shape, so a Mac, a model and SSH would only add
noise. The live evidence for the same levers belongs in ``PHASE-2B-RUN.md``.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from omlx.cluster.assistant import election as election_module
from omlx.cluster.assistant import policy as policy_module
from omlx.cluster.assistant.election import NodeFacts, NoFeasiblePlanError
from omlx.cluster.assistant.policy import (
    ADMISSION_P50,
    ADMISSION_P95,
    ResourcePolicy,
    SavedPolicy,
    policy_model_settings,
    solve_cluster_plan,
)
from omlx.cluster.planner import ModelLayout

GIB = 1024**3


def facts(
    node_id: str,
    *,
    capacity_gib: float = 48.0,
    admission_gib: float = 20.5,
    p50_gib: float | None = None,
    p95_gib: float | None = None,
    tier: str = "balanced",
    host: str = "",
) -> NodeFacts:
    return NodeFacts(
        node_id=node_id,
        capacity_gib=capacity_gib,
        admission_p05_gib=admission_gib,
        admission_p50_gib=p50_gib,
        admission_p95_gib=p95_gib,
        memory_guard_tier=tier,
        endpoint_host=host,
    )


def layout(
    *,
    layers: int = 20,
    layer_gib: float = 2.0,
    fixed_gib: float = 1.0,
    table_gib: float = 0.0,
    table_kind: str = "",
    kv_bytes_per_token_per_layer: int = 0,
    max_context_tokens: int = 0,
    path: str = "/models/fake",
    supports_pipeline: bool = True,
) -> ModelLayout:
    return ModelLayout(
        source=path,
        fixed_weight_bytes=int(fixed_gib * GIB),
        layer_weight_bytes=(int(layer_gib * GIB),) * layers,
        supports_pipeline=supports_pipeline,
        tensor_parallel_ssd_layer_bytes=(
            (int(table_gib * GIB),) * layers if table_gib else ()
        ),
        table_offload_kind=table_kind,
        kv_bytes_per_token_per_layer=kv_bytes_per_token_per_layer,
        max_context_tokens=max_context_tokens,
    )


def solve(nodes, model, **overrides: Any):
    values: dict[str, Any] = {
        "target_context_tokens": 8192,
        "execution_profile": "balanced",
    }
    values.update(overrides)
    return solve_cluster_plan(nodes, model, **values)


# ---------------------------------------------------------------------------
# Every lever can turn a refusal into a plan
# ---------------------------------------------------------------------------


def test_keeping_tables_on_ssd_turns_a_refusal_into_a_plan() -> None:
    model = layout(table_gib=1.0, table_kind="engram")
    nodes = [facts("a"), facts("b")]

    with pytest.raises(NoFeasiblePlanError):
        solve(
            nodes,
            model,
            policy=ResourcePolicy(context_tokens=8192, table_offload="none"),
            forced_node_ids=["a", "b"],
        )

    offloaded = solve(
        nodes,
        model,
        policy=ResourcePolicy(context_tokens=8192, table_offload="engram"),
        forced_node_ids=["a", "b"],
    )
    assert offloaded.policy.table_offload == "engram"
    assert offloaded.election.plan_signature


def test_routed_expert_offload_turns_a_refusal_into_a_plan(monkeypatch) -> None:
    # A fake DeepSeek checkpoint: every layer has 1 GiB of routed experts oMLX
    # can stream from SSD. The arithmetic still comes from the policy module's
    # one helper, which is what the coordinator uses.
    monkeypatch.setitem(
        policy_module.__dict__,
        "moe_offload_layer_bytes",
        lambda model, fraction, mtp=False: (GIB,) * model.layer_count,
    )
    model = layout()
    nodes = [facts("a"), facts("b")]

    with pytest.raises(NoFeasiblePlanError):
        solve(
            nodes,
            layout(),
            policy=ResourcePolicy(context_tokens=8192),
            forced_node_ids=["a", "b"],
        )

    selection = solve(nodes, model, saved=SavedPolicy(moe_expert_offload_enabled=True))
    assert selection.policy.moe_expert_offload is True
    assert 0 < selection.policy.moe_resident_fraction <= 1


def test_expert_offload_is_priced_into_the_stage_plan(monkeypatch) -> None:
    """The lever is only real if the planner holds the offloaded bytes back.

    A plan that names a resident fraction but charges every expert byte would
    still refuse (or refuse later) a model the coordinator can launch, so the
    stage's residency — not just the policy flag — is what this asserts.
    """

    monkeypatch.setitem(
        policy_module.__dict__,
        "moe_offload_layer_bytes",
        lambda model, fraction, mtp=False: (GIB,) * model.layer_count,
    )
    nodes = [
        facts("a", capacity_gib=20.0, admission_gib=20.0, p50_gib=20.0, p95_gib=20.0),
        facts("b", capacity_gib=20.0, admission_gib=20.0, p50_gib=20.0, p95_gib=20.0),
    ]
    model = layout(
        layers=20, layer_gib=1.5, fixed_gib=1.0, kv_bytes_per_token_per_layer=4096
    )

    selection = solve(nodes, model, target_context_tokens=1_048_576)
    assert selection.policy.moe_expert_offload is True
    plan = selection.election.selected.shard_plan
    assert plan is not None
    for assignment in plan.assignments:
        # One GiB per layer is streamed, so only half the layer stays resident.
        assert assignment.moe_offload_weight_bytes == GIB * assignment.layer_count
        assert (
            assignment.moe_expert_offload_resident_fraction
            == selection.policy.moe_resident_fraction
        )
        assert assignment.planned_weight_bytes == (
            int(1.0 * GIB)
            + int(0.5 * GIB) * assignment.layer_count
            + assignment.kv_cache_bytes
        )


def test_raising_the_memory_tier_turns_a_refusal_into_a_plan() -> None:
    nodes = [
        facts("a", admission_gib=10.0, p50_gib=10.0, p95_gib=24.0),
        facts("b", admission_gib=10.0, p50_gib=10.0, p95_gib=24.0),
    ]
    model = layout(layers=20, layer_gib=2.0, fixed_gib=1.0)  # 41 GiB

    with pytest.raises(NoFeasiblePlanError):
        solve(
            nodes,
            model,
            policy=ResourcePolicy(context_tokens=8192),
            forced_node_ids=["a", "b"],
        )

    selection = solve(nodes, model)
    assert selection.policy.memory_guard_tier == "aggressive"
    assert selection.policy.admission == ADMISSION_P50
    codes = {warning.code for warning in selection.warnings}
    assert "tier_raised" in codes


def test_a_shorter_context_turns_a_refusal_into_a_plan() -> None:
    # 16 KiB of KV per token per layer is enough that only the documented floor
    # still fits. The capacity is close to the admission so raising the tier
    # cannot rescue the plan: the context is the lever under test.
    nodes = [
        facts("a", capacity_gib=13.0, admission_gib=12.5),
        facts("b", capacity_gib=13.0, admission_gib=12.5),
    ]
    model = layout(
        layers=20,
        layer_gib=1.0,
        fixed_gib=1.0,
        kv_bytes_per_token_per_layer=16 * 1024,
        max_context_tokens=32768,
    )

    with pytest.raises(NoFeasiblePlanError):
        solve(
            nodes,
            model,
            target_context_tokens=32768,
            policy=ResourcePolicy(context_tokens=32768),
            forced_node_ids=["a", "b"],
        )

    selection = solve(nodes, model, target_context_tokens=32768)
    assert selection.policy.context_tokens < 32768
    assert selection.policy.context_tokens >= 2048
    assert "context_reduced" in {item.code for item in selection.warnings}


def test_never_plans_past_the_models_own_context_limit() -> None:
    nodes = [facts("a", capacity_gib=96.0, admission_gib=80.0),
             facts("b", capacity_gib=96.0, admission_gib=80.0)]
    model = layout(max_context_tokens=16384)

    selection = solve(nodes, model, target_context_tokens=1_048_576)

    assert selection.policy.context_tokens == 16384


def test_the_optimistic_admission_needs_explicit_consent() -> None:
    nodes = [
        facts("a", capacity_gib=13.0, admission_gib=10.0, p50_gib=10.0, p95_gib=12.5),
        facts("b", capacity_gib=13.0, admission_gib=10.0, p50_gib=10.0, p95_gib=12.5),
    ]
    model = layout(layers=24, layer_gib=1.0, fixed_gib=0.0)  # 24 GiB

    with pytest.raises(NoFeasiblePlanError):
        solve(nodes, model)

    selection = solve(nodes, model, allow_optimistic_admission=True)
    assert selection.policy.admission == ADMISSION_P95
    assert "optimistic_admission" in {item.code for item in selection.warnings}


# ---------------------------------------------------------------------------
# Fewest Macs that fit
# ---------------------------------------------------------------------------


def test_a_pair_wins_on_four_macs_and_names_the_two_excluded() -> None:
    nodes = [
        facts("m5", capacity_gib=128.0, admission_gib=96.0),
        facts("m2", capacity_gib=64.0, admission_gib=54.0),
        facts("m1", capacity_gib=32.0, admission_gib=23.0),
        facts("m4", capacity_gib=32.0, admission_gib=23.0),
    ]
    model = layout(layers=60, layer_gib=1.0, fixed_gib=0.0)  # 60 GiB

    selection = solve(nodes, model)

    assert selection.feasible is True
    assert set(selection.election.rank_order) == {"m5", "m2"}
    assert {item.node_id for item in selection.excluded_nodes} == {"m1", "m4"}
    assert "macs_not_needed" in {item.code for item in selection.warnings}
    assert selection.size_refusals == ()


def test_a_model_that_needs_three_returns_three_and_records_why_two_failed() -> None:
    nodes = [
        facts("m5", capacity_gib=64.0, admission_gib=60.0),
        facts("m2", capacity_gib=48.0, admission_gib=40.0),
        facts("m1", capacity_gib=40.0, admission_gib=35.0),
    ]
    model = layout(layers=120, layer_gib=1.0, fixed_gib=0.0)  # 120 GiB

    selection = solve(nodes, model)

    assert len(selection.election.rank_order) == 3
    assert selection.size_refusals
    assert selection.size_refusals[0].node_count == 2
    assert selection.size_refusals[0].reason
    assert selection.excluded_nodes == ()


def test_node_selection_all_forces_every_measured_mac() -> None:
    nodes = [
        facts("m5", capacity_gib=128.0, admission_gib=96.0),
        facts("m2", capacity_gib=64.0, admission_gib=54.0),
        facts("m1", capacity_gib=64.0, admission_gib=50.0),
    ]
    model = layout(layers=40, layer_gib=1.0, fixed_gib=0.0)  # 40 GiB

    minimal = solve(nodes, model)
    everyone = solve(nodes, model, node_selection="all")

    assert len(minimal.election.rank_order) == 2
    assert len(everyone.election.rank_order) == 3


# ---------------------------------------------------------------------------
# The policy is part of the plan identity
# ---------------------------------------------------------------------------


def test_a_model_mlx_lm_cannot_pipeline_is_refused_with_one_action() -> None:
    nodes = [facts("a"), facts("b")]

    with pytest.raises(NoFeasiblePlanError) as excinfo:
        solve(nodes, layout(supports_pipeline=False))

    assert excinfo.value.action == "choose_pipeline_model"
    assert "pipeline" in str(excinfo.value)


def test_the_table_policy_changes_the_plan_signature() -> None:
    nodes = [
        facts("a", capacity_gib=64.0, admission_gib=40.0),
        facts("b", capacity_gib=64.0, admission_gib=40.0),
    ]
    model = layout(table_gib=0.5, table_kind="engram")

    resident = solve(
        nodes,
        model,
        policy=ResourcePolicy(context_tokens=8192),
        forced_node_ids=["a", "b"],
    )
    offloaded = solve(
        nodes,
        model,
        policy=ResourcePolicy(context_tokens=8192, table_offload="engram"),
        forced_node_ids=["a", "b"],
    )

    assert resident.election.plan_signature != offloaded.election.plan_signature


def test_the_settings_patch_only_enables_what_the_policy_uses() -> None:
    assert policy_model_settings(ResourcePolicy(context_tokens=8192)) == {}
    assert policy_model_settings(
        ResourcePolicy(context_tokens=8192, table_offload="ple")
    ) == {"qwen4_ple_ssd_offload": True}
    assert policy_model_settings(
        ResourcePolicy(
            context_tokens=8192,
            table_offload="engram",
            moe_expert_offload=True,
            moe_resident_fraction=0.5,
        )
    ) == {
        "deepseek_v41_engram_ssd_offload": True,
        "moe_expert_offload_enabled": True,
        "moe_expert_offload_resident_fraction": 0.5,
    }
    # An explicit resident plan for a checkpoint that has a table writes the
    # off switch; the plan showed it and Start setup is the confirmation.
    assert policy_model_settings(
        ResourcePolicy(context_tokens=8192, table_kind="engram")
    ) == {"deepseek_v41_engram_ssd_offload": False}


def test_the_ple_and_engram_switches_are_both_honoured(monkeypatch) -> None:
    from omlx.cluster import routes as cluster_routes

    monkeypatch.setitem(
        cluster_routes.__dict__,
        "_saved_model_settings",
        lambda path: SimpleNamespace(qwen4_ple_ssd_offload=True),
    )
    assert cluster_routes._saved_engram_ssd_offload("/models/qwen") is True

    monkeypatch.setitem(
        cluster_routes.__dict__,
        "_saved_model_settings",
        lambda path: SimpleNamespace(deepseek_v41_engram_ssd_offload=True),
    )
    assert cluster_routes._saved_engram_ssd_offload("/models/ds") is True

    monkeypatch.setitem(
        cluster_routes.__dict__,
        "_saved_model_settings",
        lambda path: SimpleNamespace(),
    )
    assert cluster_routes._saved_engram_ssd_offload("/models/plain") is False


def test_the_solver_honours_the_requested_microbatch():
    """B12: the lever plans carry the requested microbatch, not a hardcoded 1.

    The plan's microbatch is what apply/start launch with; ignoring the request
    made the plan and the launch disagree. The default stays 1 when nothing is
    requested, so existing callers are unchanged.
    """

    nodes = [
        facts("a", capacity_gib=64, admission_gib=40),
        facts("b", capacity_gib=64, admission_gib=40),
    ]
    model = layout(layers=20, layer_gib=2.0)

    assert solve(nodes, model, microbatch_size=2).policy.microbatch_size == 2
    assert solve(nodes, model, microbatch_size=4).policy.microbatch_size == 4
    assert solve(nodes, model).policy.microbatch_size == 1


# ---------------------------------------------------------------------------
# Slice 10c: the subset prune and the per-call memo are cost cuts only.
# Every decision — subset, rank order, context, levers, signature, warnings,
# refusals — must stay bit-identical to the full linear enumeration.
# ---------------------------------------------------------------------------


def _plan_snapshot(selection: Any) -> dict[str, Any]:
    """Everything a plan answer carries, in one comparable dict."""

    return {
        "election": selection.election.to_dict(),
        "policy": selection.policy.to_dict(),
        "excluded": [item.to_dict() for item in selection.excluded_nodes],
        "warnings": [item.to_dict() for item in selection.warnings],
        "size_refusals": [item.to_dict() for item in selection.size_refusals],
    }


def _outcome(nodes: Any, model: Any, **overrides: Any) -> tuple[str, Any]:
    try:
        return ("plan", _plan_snapshot(solve(nodes, model, **overrides)))
    except NoFeasiblePlanError as exc:
        return (
            "refusal",
            {"message": str(exc), "action": exc.action, "result": exc.result.to_dict()},
        )


def _never_prune(*args: Any, **kwargs: Any) -> bool:
    return False


class _NoMemo:
    """A pass-through stand-in for ``_SolverMemo``: no caching at all."""

    def moe_bytes(self, model: Any, fraction: float, *, mtp: bool) -> tuple[int, ...]:
        return policy_module.moe_offload_layer_bytes(model, fraction, mtp=mtp)

    def election(self, key: Any, compute: Any) -> Any:
        return compute()


def _differential_scenarios() -> list[tuple[str, Any, Any, dict[str, Any]]]:
    """A representative grid: feasible 2/3/4-Mac plans, every lever, refusals."""

    pair = [facts("a"), facts("b")]
    roomy_pair = [
        facts("a", capacity_gib=64.0, admission_gib=40.0),
        facts("b", capacity_gib=64.0, admission_gib=40.0),
    ]
    tight_pair = [
        facts("a", admission_gib=10.0, p50_gib=10.0, p95_gib=24.0),
        facts("b", admission_gib=10.0, p50_gib=10.0, p95_gib=24.0),
    ]
    kv_pair = [
        facts("a", capacity_gib=13.0, admission_gib=12.5),
        facts("b", capacity_gib=13.0, admission_gib=12.5),
    ]
    four = [
        facts("m5", capacity_gib=128.0, admission_gib=96.0),
        facts("m2", capacity_gib=64.0, admission_gib=54.0),
        facts("m1", capacity_gib=32.0, admission_gib=23.0),
        facts("m4", capacity_gib=32.0, admission_gib=21.0),
    ]
    three = [
        facts("m5", capacity_gib=64.0, admission_gib=60.0),
        facts("m2", capacity_gib=48.0, admission_gib=40.0),
        facts("m1", capacity_gib=40.0, admission_gib=35.0),
    ]
    heavy_tail = replace(
        layout(layers=20, layer_gib=1.0),
        layer_weight_bytes=(int(1.0 * GIB),) * 19 + (int(30.0 * GIB),),
    )
    return [
        ("feasible_pair", pair, layout(), {}),
        ("feasible_three", three, layout(layers=120, layer_gib=1.0, fixed_gib=0.0), {}),
        ("feasible_four_all", four, layout(layers=80, layer_gib=1.0, fixed_gib=0.0),
         {"node_selection": "all"}),
        ("infeasible_huge", pair, layout(layers=120, layer_gib=4.0), {}),
        ("infeasible_four", four, layout(layers=200, layer_gib=4.0), {}),
        ("tier_lever", tight_pair, layout(layers=20, layer_gib=2.0, fixed_gib=1.0), {}),
        ("context_lever", kv_pair,
         layout(layers=20, layer_gib=1.0, fixed_gib=1.0,
                kv_bytes_per_token_per_layer=16 * 1024, max_context_tokens=32768),
         {"target_context_tokens": 32768}),
        ("table_engram",
         [facts("a", capacity_gib=32.0, admission_gib=15.0),
          facts("b", capacity_gib=32.0, admission_gib=15.0)],
         layout(table_gib=1.0, table_kind="engram"), {}),
        ("table_ple",
         [facts("a", capacity_gib=32.0, admission_gib=15.0),
          facts("b", capacity_gib=32.0, admission_gib=15.0)],
         layout(table_gib=1.0, table_kind="ple"), {}),
        ("table_tight_pair", pair, layout(table_gib=1.0, table_kind="engram"), {}),
        ("table_saved_on", pair, layout(table_gib=1.0, table_kind="engram"),
         {"saved": SavedPolicy(engram_ssd_offload=True)}),
        ("p95_consent",
         [facts("a", capacity_gib=13.0, admission_gib=10.0, p50_gib=10.0, p95_gib=12.5),
          facts("b", capacity_gib=13.0, admission_gib=10.0, p50_gib=10.0, p95_gib=12.5)],
         layout(layers=24, layer_gib=1.0, fixed_gib=0.0),
         {"allow_optimistic_admission": True}),
        ("mtp_head", roomy_pair,
         replace(layout(layers=40, layer_gib=1.0, fixed_gib=0.0), mtp_weight_bytes=4 * GIB),
         {"mtp": True}),
        ("rank0_extra", pair, layout(layers=30, fixed_gib=30.0),
         {"rank0_extra_bytes": 8 * GIB}),
        # The head must stay fully resident even with a table offloaded: the
        # prune charges it separately, so these pin that it is never clamped.
        ("table_mtp_extra",
         [facts("a", capacity_gib=32.0, admission_gib=15.0),
          facts("b", capacity_gib=32.0, admission_gib=15.0)],
         replace(layout(table_gib=1.0, table_kind="engram"), mtp_weight_bytes=2 * GIB),
         {"mtp": True, "rank0_extra_bytes": 1 * GIB}),
        ("table_mtp_hopeless",
         [facts("a", capacity_gib=32.0, admission_gib=8.0),
          facts("b", capacity_gib=32.0, admission_gib=8.0)],
         replace(layout(layers=40, table_gib=1.0, table_kind="engram"),
                 mtp_weight_bytes=4 * GIB),
         {"mtp": True, "rank0_extra_bytes": 2 * GIB}),
        ("fabric_line", four, layout(layers=60, layer_gib=1.0, fixed_gib=0.0),
         {"fabric_links": [["m5", "m2"], ["m2", "m1"], ["m1", "m4"]]}),
        ("fabric_incomplete", three, layout(layers=60, layer_gib=1.0, fixed_gib=0.0),
         {"fabric_links": [["m5", "m2"]]}),
        ("forced_subset", four, layout(layers=60, layer_gib=1.0, fixed_gib=0.0),
         {"forced_node_ids": ["m5", "m2"]}),
        ("two_mac_fallback", four, layout(layers=60, layer_gib=1.0, fixed_gib=0.0),
         {"max_size": 2, "two_mac_fallback": True}),
        ("microbatch", roomy_pair, layout(), {"microbatch_size": 4}),
        ("single_node_feasible", [facts("a", capacity_gib=64.0, admission_gib=50.0)],
         layout(), {}),
        ("single_node_infeasible", [facts("a")], layout(layers=120, layer_gib=4.0), {}),
        ("saved_aggressive", tight_pair, layout(layers=20, layer_gib=2.0, fixed_gib=1.0),
         {"saved": SavedPolicy(memory_guard_tier="aggressive")}),
        ("heavy_tail", roomy_pair, heavy_tail, {}),
        ("stage_starts", roomy_pair,
         replace(layout(layers=24, layer_gib=1.0), pipeline_stage_starts=(8,)), {}),
        ("declared_ceiling", roomy_pair,
         layout(layers=20, layer_gib=1.0, fixed_gib=1.0,
                kv_bytes_per_token_per_layer=20480, max_context_tokens=200000),
         {"target_context_tokens": 1_048_576}),
        ("unhealthy_mac", [facts("a"), facts("b"), replace(facts("c"), healthy=False)],
         layout(), {}),
    ]


def test_the_pruned_solver_decides_exactly_like_the_full_descent(monkeypatch) -> None:
    """Signature preservation: optimized vs the original algorithm, bit-identical.

    The reference disables BOTH cost cuts (``_subset_cannot_fit`` and the
    ``_SolverMemo``), i.e. it is the original linear enumeration with no caching.
    Every scenario must produce the same rank order, context, policy, signature,
    warnings, exclusions and refusal payloads, because apply/start re-prove the
    signature and would refuse a plan that drifted by even one byte.
    """

    original_prune = policy_module._subset_cannot_fit
    original_memo = policy_module._SolverMemo
    for name, nodes, model, overrides in _differential_scenarios():
        optimized = _outcome(nodes, model, **overrides)
        monkeypatch.setitem(policy_module.__dict__, "_subset_cannot_fit", _never_prune)
        monkeypatch.setitem(policy_module.__dict__, "_SolverMemo", _NoMemo)
        reference = _outcome(nodes, model, **overrides)
        monkeypatch.setitem(policy_module.__dict__, "_subset_cannot_fit", original_prune)
        monkeypatch.setitem(policy_module.__dict__, "_SolverMemo", original_memo)
        assert optimized == reference, f"scenario {name} drifted"


def test_the_pruned_solver_matches_the_full_descent_with_moe_offload(monkeypatch) -> None:
    """The same preservation proof with routed-expert offload levers active."""

    monkeypatch.setitem(
        policy_module.__dict__,
        "moe_offload_layer_bytes",
        lambda model, fraction, mtp=False: (GIB,) * model.layer_count,
    )
    nodes = [facts("a"), facts("b")]
    scenarios = [
        ("moe_saved", layout(), {"saved": SavedPolicy(moe_expert_offload_enabled=True)}),
        ("moe_ladder", layout(layers=30, layer_gib=2.0), {}),
        ("moe_hopeless", layout(layers=120, layer_gib=4.0),
         {"saved": SavedPolicy(moe_expert_offload_enabled=True)}),
        ("moe_mtp", replace(layout(), mtp_weight_bytes=2 * GIB),
         {"mtp": True, "saved": SavedPolicy(moe_expert_offload_enabled=True)}),
        # MoE + table + head together: the tightest stress on the resident bound.
        ("moe_table_mtp",
         replace(layout(table_gib=1.0, table_kind="engram"), mtp_weight_bytes=2 * GIB),
         {"mtp": True, "rank0_extra_bytes": 1 * GIB,
          "saved": SavedPolicy(moe_expert_offload_enabled=True)}),
        ("moe_table_mtp_hopeless",
         replace(layout(layers=60, layer_gib=3.0, table_gib=1.0, table_kind="engram"),
                 mtp_weight_bytes=4 * GIB),
         {"mtp": True, "rank0_extra_bytes": 2 * GIB,
          "saved": SavedPolicy(moe_expert_offload_enabled=True)}),
    ]
    original_prune = policy_module._subset_cannot_fit
    original_memo = policy_module._SolverMemo
    for name, model, overrides in scenarios:
        optimized = _outcome(nodes, model, **overrides)
        monkeypatch.setitem(policy_module.__dict__, "_subset_cannot_fit", _never_prune)
        monkeypatch.setitem(policy_module.__dict__, "_SolverMemo", _NoMemo)
        reference = _outcome(nodes, model, **overrides)
        monkeypatch.setitem(policy_module.__dict__, "_subset_cannot_fit", original_prune)
        monkeypatch.setitem(policy_module.__dict__, "_SolverMemo", original_memo)
        assert optimized == reference, f"scenario {name} drifted"


def test_a_wrong_bound_never_overrides_the_planner(monkeypatch) -> None:
    """Fail-safe: if the prune ever misfires, the full descent still runs.

    The bound is a cost cut, not an authority. Forcing it to "cannot fit" on a
    feasible cluster must still produce the identical plan, because the
    canonical election succeeds and the solver falls through to the descent.
    """

    nodes = [facts("a"), facts("b")]
    cases = [
        (nodes, layout(), {}),
        (nodes, layout(layers=120, layer_gib=4.0), {}),  # genuinely hopeless
        ([facts("a", admission_gib=10.0, p50_gib=10.0, p95_gib=24.0),
          facts("b", admission_gib=10.0, p50_gib=10.0, p95_gib=24.0)],
         layout(layers=20, layer_gib=2.0, fixed_gib=1.0), {}),
    ]
    for case_nodes, model, overrides in cases:
        reference = _outcome(case_nodes, model, **overrides)
        monkeypatch.setitem(policy_module.__dict__, "_subset_cannot_fit", lambda *a, **k: True)
        assert _outcome(case_nodes, model, **overrides) == reference
        monkeypatch.undo()


def test_the_largest_fitting_context_wins_on_the_full_ladder() -> None:
    """A 10-rung ladder: the chosen context is the largest one that plans.

    KV is sized so 131072 fits the pair but 262144 cannot (weights plus KV
    exceed the summed admission at any split), and the model's own non-preset
    limit adds a 200000 rung that fits. The descent must stop at the largest
    fitting rung, never at the first one that fits.
    """

    # p50/p95 pinned to the floor so the tier lever cannot raise admission:
    # the context, not the tier, is the dimension under test.
    nodes = [
        facts("a", capacity_gib=52.0, admission_gib=51.0, p50_gib=51.0, p95_gib=51.0),
        facts("b", capacity_gib=52.0, admission_gib=51.0, p50_gib=51.0, p95_gib=51.0),
    ]
    model = layout(
        layers=20, layer_gib=1.0, fixed_gib=1.0, kv_bytes_per_token_per_layer=20480
    )

    selection = solve(nodes, model, target_context_tokens=1_048_576)
    assert selection.policy.context_tokens == 131072
    assert selection.policy.target_context_tokens == 1_048_576
    assert "context_reduced" in {item.code for item in selection.warnings}

    # The next rung up is genuinely infeasible, not merely untried.
    with pytest.raises(NoFeasiblePlanError):
        solve(
            nodes,
            model,
            target_context_tokens=1_048_576,
            policy=ResourcePolicy(context_tokens=262144),
            forced_node_ids=list(selection.election.rank_order),
        )

    # The model's own non-preset limit is a ladder rung too, and it fits.
    capped = solve(
        nodes,
        replace(model, max_context_tokens=200000),
        target_context_tokens=1_048_576,
    )
    assert capped.policy.context_tokens == 200000


def test_the_prune_skips_the_ladder_for_hopeless_subsets(monkeypatch) -> None:
    """A model no subset can hold costs one election per subset, not per rung.

    The count is asserted through a ``plan_hybrid`` spy — no wall-clock claims.
    Four Macs give 6 pairs + 4 triples + 1 quad = 11 subsets; each pruned
    subset asks exactly the canonical question (2/6/24 rank orders).
    """

    nodes = [
        facts("m5", capacity_gib=128.0, admission_gib=96.0),
        facts("m2", capacity_gib=64.0, admission_gib=54.0),
        facts("m1", capacity_gib=32.0, admission_gib=23.0),
        facts("m4", capacity_gib=32.0, admission_gib=21.0),
    ]
    model = layout(layers=120, layer_gib=4.0)  # 480 GiB: hopeless everywhere

    calls: list[int] = []
    real_plan = election_module.plan_hybrid

    def spy(plan_model, budgets, **kwargs):
        calls.append(1)
        return real_plan(plan_model, budgets, **kwargs)

    monkeypatch.setitem(election_module.__dict__, "plan_hybrid", spy)

    with pytest.raises(NoFeasiblePlanError) as pruned_error:
        solve(nodes, model, target_context_tokens=32768)
    pruned_calls = len(calls)

    calls.clear()
    monkeypatch.setitem(policy_module.__dict__, "_subset_cannot_fit", _never_prune)
    with pytest.raises(NoFeasiblePlanError) as full_error:
        solve(nodes, model, target_context_tokens=32768)
    full_calls = len(calls)

    # One canonical election per subset: 6*2 + 4*6 + 1*24 rank orders.
    assert pruned_calls == 60
    # The full descent pays every ladder rung and every lever on top.
    assert full_calls > pruned_calls * 5
    # And the refusal the user sees is unchanged, message and payload.
    assert str(pruned_error.value) == str(full_error.value)
    assert pruned_error.value.result.to_dict() == full_error.value.result.to_dict()


def test_a_subset_that_exactly_fits_the_summed_admission_is_not_pruned() -> None:
    """The bound is a necessary condition: equality must keep the subset."""

    nodes = [
        facts("a", capacity_gib=32.0, admission_gib=20.0),
        facts("b", capacity_gib=32.0, admission_gib=20.0),
    ]
    model = layout(layers=20, layer_gib=1.9, fixed_gib=0.0)  # 38 GiB vs 40 summed

    selection = solve(nodes, model)
    assert selection.feasible is True
    assert selection.election.plan_signature

    original_prune = policy_module._subset_cannot_fit
    seen: list[bool] = []

    def watching_prune(*args, **kwargs):
        answer = original_prune(*args, **kwargs)
        seen.append(answer)
        return answer

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(policy_module, "_subset_cannot_fit", watching_prune)
        again = solve(nodes, model)
    assert seen and not any(seen)
    assert again.election.plan_signature == selection.election.plan_signature


def test_the_memo_never_re_evaluates_an_identical_subproblem(monkeypatch) -> None:
    """Correctness of the per-call memo: identical answers, no duplicate work."""

    import json

    real_elect = policy_module.elect
    keys: list[str] = []

    def spy(nodes, model, **kwargs):
        keys.append(
            json.dumps(
                [[n.node_id for n in nodes], kwargs], sort_keys=True, default=str
            )
        )
        return real_elect(nodes, model, **kwargs)

    monkeypatch.setitem(policy_module.__dict__, "elect", spy)

    scenarios = [
        ([facts("a"), facts("b")], layout(), {}),
        ([facts("a", admission_gib=10.0, p50_gib=10.0, p95_gib=24.0),
          facts("b", admission_gib=10.0, p50_gib=10.0, p95_gib=24.0)],
         layout(layers=20, layer_gib=2.0, fixed_gib=1.0), {}),
        ([facts("a", capacity_gib=13.0, admission_gib=12.5),
          facts("b", capacity_gib=13.0, admission_gib=12.5)],
         layout(layers=20, layer_gib=1.0, fixed_gib=1.0,
                kv_bytes_per_token_per_layer=16 * 1024, max_context_tokens=32768),
         {"target_context_tokens": 32768}),
    ]
    original_memo = policy_module._SolverMemo
    for nodes, model, overrides in scenarios:
        keys.clear()
        with_memo = _outcome(nodes, model, **overrides)
        assert len(keys) == len(set(keys)), "a subproblem was evaluated twice"

        keys.clear()
        monkeypatch.setitem(policy_module.__dict__, "_SolverMemo", _NoMemo)
        without_memo = _outcome(nodes, model, **overrides)
        monkeypatch.setitem(policy_module.__dict__, "_SolverMemo", original_memo)
        assert with_memo == without_memo


def test_the_memo_does_not_leak_across_solver_calls(monkeypatch) -> None:
    """Each solve re-elects from its own facts; nothing is cached globally."""

    real_elect = policy_module.elect
    calls: list[int] = []

    def spy(nodes, model, **kwargs):
        calls.append(1)
        return real_elect(nodes, model, **kwargs)

    monkeypatch.setitem(policy_module.__dict__, "elect", spy)
    nodes = [facts("a"), facts("b")]
    model = layout()

    first = solve(nodes, model)
    first_calls = len(calls)
    second = solve(nodes, model)
    assert first_calls > 0
    assert len(calls) == 2 * first_calls  # the second call re-evaluated everything
    assert second.election.plan_signature == first.election.plan_signature


def test_the_memo_reads_the_expert_layout_once_per_fraction(monkeypatch) -> None:
    """The MoE helper hits the checkpoint headers; the memo collapses repeats."""

    reads: list[tuple[float, bool]] = []

    def fake(model, fraction, mtp=False):
        reads.append((fraction, mtp))
        return (GIB,) * model.layer_count

    monkeypatch.setitem(policy_module.__dict__, "moe_offload_layer_bytes", fake)
    nodes = [facts("a"), facts("b")]
    model = layout()

    selection = solve(nodes, model, saved=SavedPolicy(moe_expert_offload_enabled=True))
    assert selection.policy.moe_expert_offload is True
    assert reads, "the expert layout was never read"
    assert len(reads) == len(set(reads)), "a (fraction, mtp) layout was read twice"

    reads.clear()
    monkeypatch.setitem(policy_module.__dict__, "_SolverMemo", _NoMemo)
    without_memo = solve(
        nodes, model, saved=SavedPolicy(moe_expert_offload_enabled=True)
    )
    assert len(reads) > len(set(reads))  # without the memo the reads repeat
    assert _plan_snapshot(without_memo) == _plan_snapshot(selection)


def test_a_lever_plan_re_proves_to_the_same_signature() -> None:
    """apply/start re-elect with the stored policy: the signature must hold."""

    nodes = [
        facts("a", admission_gib=10.0, p50_gib=10.0, p95_gib=24.0),
        facts("b", admission_gib=10.0, p50_gib=10.0, p95_gib=24.0),
    ]
    model = layout(layers=20, layer_gib=2.0, fixed_gib=1.0)

    selection = solve(nodes, model)
    assert selection.policy.memory_guard_tier == "aggressive"
    again = solve(
        nodes,
        model,
        policy=selection.policy,
        forced_node_ids=list(selection.election.rank_order),
    )
    assert again.election.plan_signature == selection.election.plan_signature
    assert again.policy == selection.policy


def test_an_mtp_plan_re_proves_to_the_same_signature() -> None:
    nodes = [
        facts("m5", capacity_gib=128.0, admission_gib=96.0),
        facts("m2", capacity_gib=64.0, admission_gib=54.0),
    ]
    model = replace(
        layout(layers=40, layer_gib=1.0, fixed_gib=0.0), mtp_weight_bytes=4 * GIB
    )

    selection = solve(nodes, model, mtp=True)
    again = solve(
        nodes,
        model,
        mtp=True,
        policy=selection.policy,
        forced_node_ids=list(selection.election.rank_order),
    )
    assert again.election.plan_signature == selection.election.plan_signature


def test_a_table_offload_plan_re_proves_to_the_same_signature() -> None:
    # 41 GiB resident does not fit the 30 GiB pair; the 21 GiB left with the
    # Engram table on SSD does — so the plan must use the offload lever.
    nodes = [
        facts("a", capacity_gib=32.0, admission_gib=15.0),
        facts("b", capacity_gib=32.0, admission_gib=15.0),
    ]
    model = layout(table_gib=1.0, table_kind="engram")

    selection = solve(nodes, model)
    assert selection.policy.table_offload == "engram"
    again = solve(
        nodes,
        model,
        policy=selection.policy,
        forced_node_ids=list(selection.election.rank_order),
    )
    assert again.election.plan_signature == selection.election.plan_signature


def test_a_table_capable_model_offloads_its_lookup_table_by_default() -> None:
    """Decision 2026-11-14: offload Engram/PLE by default so experts stay resident.

    The solver's base lever (the least invasive) already offloads the table the
    checkpoint supports, so the plan shows it and Start writes it. A model with
    no such table is unaffected.
    """

    engram_model = layout(table_gib=1.0, table_kind="engram")
    levers = policy_module._levers(
        policy_module.SavedPolicy(), engram_model, allow_optimistic=False
    )
    assert levers[0].table_offload == "engram"

    plain = layout()
    plain_levers = policy_module._levers(
        policy_module.SavedPolicy(), plain, allow_optimistic=False
    )
    assert plain_levers[0].table_offload == "none"


def test_quality_is_kept_by_default_over_a_smaller_context(monkeypatch) -> None:
    """Decision 2026-11-14: never silently trade quality (context) for speed.

    Weights plus KV at 131072 need routed-expert offload, while 65536 would fit
    with experts resident. The default keeps the larger context and offloads
    experts (quality stays, speed pays); the faster alternative is surfaced to
    the user as a choice, not chosen here. Lookup tables are already offloaded
    by default (see the table-default test), which has no speed cost.
    """

    gib_local = 1024**3
    monkeypatch.setitem(
        policy_module.__dict__,
        "moe_offload_layer_bytes",
        lambda model, fraction, mtp=False: (gib_local,) * model.layer_count,
    )
    nodes = [
        facts("a", capacity_gib=20.0, admission_gib=20.0, p50_gib=20.0, p95_gib=20.0),
        facts("b", capacity_gib=20.0, admission_gib=20.0, p50_gib=20.0, p95_gib=20.0),
    ]
    model = layout(
        layers=20, layer_gib=1.5, fixed_gib=1.0, kv_bytes_per_token_per_layer=4096
    )

    selection = solve(nodes, model, target_context_tokens=1_048_576)
    # Quality first: a context above the preferred floor is kept, and the plan
    # is willing to offload routed experts to keep it.
    assert selection.policy.context_tokens > 32768
    assert selection.policy.moe_expert_offload is True
    # The plan is honest about the tradeoff it took.
    assert selection.policy.moe_resident_fraction < 1.0


def test_excluding_expert_offload_yields_the_speed_alternative(monkeypatch) -> None:
    """The speed side of the tradeoff: same Macs, smaller context, experts resident."""

    gib_local = 1024**3
    monkeypatch.setitem(
        policy_module.__dict__,
        "moe_offload_layer_bytes",
        lambda model, fraction, mtp=False: (gib_local,) * model.layer_count,
    )
    nodes = [
        facts("a", capacity_gib=20.0, admission_gib=20.0, p50_gib=20.0, p95_gib=20.0),
        facts("b", capacity_gib=20.0, admission_gib=20.0, p50_gib=20.0, p95_gib=20.0),
    ]
    model = layout(
        layers=20, layer_gib=1.5, fixed_gib=1.0, kv_bytes_per_token_per_layer=4096
    )

    quality = solve(nodes, model, target_context_tokens=1_048_576)
    speed = solve(
        nodes, model, target_context_tokens=1_048_576, exclude_expert_offload=True
    )
    assert quality.policy.moe_expert_offload is True
    assert speed.policy.moe_expert_offload is False
    # Keeping experts resident costs context, never a larger one.
    assert speed.policy.context_tokens <= quality.policy.context_tokens
