# SPDX-License-Identifier: Apache-2.0
"""Coordinator-election tests.

These are the tests the first milestone is defined by (PROJECT-BRIEF §7,
Appendix A.8). They run off-cluster: no Mac, model, GPU or SSH is involved,
only measured numbers, because the election is a pure function and the thing
that must be provable is the *decision*.

The scenarios use the reference cluster's shapes:

===========  ==================  ==========
Mac          RAM                 role
===========  ==================  ==========
``m5``       128 GiB, fast        large/fast candidate
``m2``       64 GiB              medium candidate
``m1``       32-64 GiB, less     small candidate
===========  ==================  ==========

Every test states which §7.5 rule it pins down, because the scoring order is
the part of this module most likely to be "improved" into a free-memory sort.
"""

from __future__ import annotations

import json

import pytest

from omlx.cluster.assistant.election import (
    RANK0_ADMISSION_MARGIN_BYTES,
    ComputeProfile,
    ElectionPolicy,
    ElectionResult,
    NodeFacts,
    NoFeasiblePlanError,
    PlanCandidate,
    StagePlan,
    UnsupportedParallelismError,
    budget_for,
    elect,
    rank_candidates,
    wired_limit_binds,
)
from omlx.cluster.planner import ModelLayout, ShardPlan

GIB = 1024**3


def facts(
    node_id: str,
    *,
    capacity_gib: float,
    admission_gib: float,
    decode_gib_per_s: float | None = None,
    prefill_gib_per_s: float | None = None,
    host: str = "",
    **kwargs,
) -> NodeFacts:
    """One Mac as the measurement layer would report it."""

    compute = None
    if decode_gib_per_s is not None:
        # decode_gib_per_s is a RELATIVE speed signal in these tests. Map it
        # into the plausible memory-bandwidth window the election's
        # plausibility guard expects (E4=B): x100 turns the 0.8-4.0 calibration
        # values into ~86-430 GB/s, preserving their relative ordering.
        compute = ComputeProfile(
            decode_weight_bytes_per_second=decode_gib_per_s * 100 * GIB,
            prefill_weight_bytes_per_second=(
                (prefill_gib_per_s or decode_gib_per_s * 2) * 100 * GIB
            ),
        )
    return NodeFacts(
        node_id=node_id,
        capacity_gib=capacity_gib,
        admission_p05_gib=admission_gib,
        compute=compute,
        endpoint_host=host,
        **kwargs,
    )


def test_rank0_budget_leaves_a_safety_margin_above_the_measured_floor() -> None:
    """A live 32 GB Studio missed a 21.2 GiB plan by 0.1 GiB at Start."""

    node = facts(
        "m1",
        capacity_gib=32,
        admission_gib=20,
        admission_p50_gib=21.2,
    )

    rank0 = budget_for(node, rank=0, admission="p50")
    rank1 = budget_for(node, rank=1, admission="p50")

    assert rank0.reserve_bytes - rank1.reserve_bytes == RANK0_ADMISSION_MARGIN_BYTES
    assert rank0.reserve_bytes <= node.capacity_bytes - node.admission_bytes


def test_rank0_margin_never_goes_below_the_p05_floor() -> None:
    node = facts("tiny", capacity_gib=32, admission_gib=0.125)

    rank0 = budget_for(node, rank=0)

    assert rank0.reserve_bytes == node.capacity_bytes - node.admission_bytes


def layout(
    *,
    layers: int = 32,
    layer_gib: float = 2.0,
    fixed_gib: float = 1.0,
    last_layer_gib: float | None = None,
    mtp_gib: float = 0.0,
    stage_starts: tuple[int, ...] = (),
    activation_bytes_per_token: int = 0,
) -> ModelLayout:
    """A checkpoint with uniform layers, or a deliberately heavy tail layer."""

    sizes = [int(layer_gib * GIB)] * layers
    if last_layer_gib is not None:
        sizes[-1] = int(last_layer_gib * GIB)
    return ModelLayout(
        source="test-model",
        fixed_weight_bytes=int(fixed_gib * GIB),
        layer_weight_bytes=tuple(sizes),
        mtp_weight_bytes=int(mtp_gib * GIB),
        pipeline_stage_starts=stage_starts,
        activation_bytes_per_token=activation_bytes_per_token,
    )


def m5(**kwargs) -> NodeFacts:
    return facts(
        "m5",
        capacity_gib=128,
        admission_gib=100,
        decode_gib_per_s=2.0,
        prefill_gib_per_s=4.0,
        host="172.16.1.11",
        **kwargs,
    )


def m1(**kwargs) -> NodeFacts:
    return facts(
        "m1",
        capacity_gib=64,
        admission_gib=40,
        decode_gib_per_s=0.8,
        prefill_gib_per_s=1.6,
        host="172.16.1.14",
        **kwargs,
    )


def assert_valid_rank_order(result: ElectionResult, node_ids: set[str]) -> None:
    """The plan is a legal pipeline split of every layer across every Mac."""

    assert result.feasible
    assert result.selected is not None
    assert set(result.rank_order) == node_ids
    assert len(result.rank_order) == len(node_ids)
    assert result.coordinator_node == result.rank_order[0]
    assert result.endpoint is not None
    assert result.endpoint.node_id == result.coordinator_node

    stages = result.selected.stages
    assert [stage.rank for stage in stages] == list(range(len(stages)))
    assert [stage.node_id for stage in stages] == list(result.rank_order)

    layers = result.selected.shard_plan.model.layer_count
    ordered = sorted(stages, key=lambda stage: stage.first_layer)
    assert ordered[0].first_layer == 0
    assert ordered[-1].last_layer == layers
    for previous, current in zip(ordered, ordered[1:]):
        assert current.first_layer == previous.last_layer
    assert sum(result.selected.groups) == layers
    assert all(stage.layer_count >= 1 for stage in stages)


def test_app_origin_does_not_decide_rank0() -> None:
    """§7.2/exit criterion: the Mac running the app has no vote.

    The origin is recorded for support, and must produce a bit-identical
    election whether the user opened the app on the small Mac, the large one,
    or a machine that is not even part of the cluster.
    """

    nodes = [m1(), m5()]
    model = layout()

    elections = {}
    for origin in ("m1", "m5", "", "origin-laptop"):
        result = elect(nodes, model, context_tokens=8192, origin_node=origin)
        assert result.origin_node == origin
        elections[origin] = (
            result.coordinator_node,
            result.rank_order,
            result.plan_signature,
        )

    assert len(set(elections.values())) == 1
    chosen = elections["m1"][0]
    assert chosen == "m5"
    assert elections["m1"][0] != "m1"  # not the local-first answer


def test_infeasible_local_rank0_is_replaced_by_a_remote_one() -> None:
    """§7.6: the M1 tail does not fit, the M5 tail does.

    The local Mac is the app origin *and* the only Mac that cannot hold the
    tail layer plus head. Election must pick the remote M5, not fall back.
    """

    # The final layer carries the output projection: 42 GiB on its own, so the
    # 40 GiB M1 cannot hold the tail even with a single layer.
    model = layout(last_layer_gib=42)
    result = elect([m1(), m5()], model, context_tokens=8192, origin_node="m1")

    assert_valid_rank_order(result, {"m1", "m5"})
    assert result.coordinator_node == "m5"
    assert result.rank_order == ("m5", "m1")
    assert [item.node_id for item in result.rejected] == ["m1"]
    assert "no legal pipeline split" in result.rejected[0].reason
    # The small Mac is still part of the plan — as a worker.
    assert result.selected.stages[-1].node_id == "m1"
    # No local fallback: nothing in the result names the origin as coordinator.
    assert result.selected.rank_order[0] != "m1"


def test_best_feasible_plan_wins_over_the_most_free_admission() -> None:
    """§7.5 steps 1-2 and §7.6: feasibility and speed outrank free memory.

    A KV/owner boundary pins the single legal cut, so the Mac in the tail
    position holds 16 of 24 layers. Putting the fast Mac there is ~2x faster;
    the slow Mac has more free admission (120 vs 100 GiB) and would win a
    free-memory sort of rank 0. It must not.
    """

    fast = facts(
        "fast",
        capacity_gib=128,
        admission_gib=100,
        decode_gib_per_s=4.0,
        prefill_gib_per_s=8.0,
        host="10.0.0.1",
    )
    roomy = facts(
        "roomy",
        capacity_gib=128,
        admission_gib=120,
        decode_gib_per_s=1.0,
        prefill_gib_per_s=2.0,
        host="10.0.0.2",
    )
    model = layout(layers=24, stage_starts=(8,))

    result = elect([roomy, fast], model, context_tokens=8192)

    assert_valid_rank_order(result, {"fast", "roomy"})
    assert result.coordinator_node == "fast"
    assert result.selected.rank0_admission_bytes == 100 * GIB

    by_coordinator = {item.coordinator_node: item for item in result.considered}
    assert set(by_coordinator) == {"fast", "roomy"}  # both were feasible
    roomy_plan = by_coordinator["roomy"]
    assert roomy_plan.rank0_admission_bytes > result.selected.rank0_admission_bytes
    # ...and it also has more rank-0 headroom, so the decision was made on the
    # predicted critical stage alone, as §7.5 requires.
    assert roomy_plan.rank0_headroom_bytes > result.selected.rank0_headroom_bytes
    assert roomy_plan.critical_stage_seconds > result.selected.critical_stage_seconds
    assert result.rejected == ()


def test_no_feasible_candidate_is_an_error_and_never_a_local_fallback() -> None:
    """§7.7: no split for any Mac is an explicit failure, not the origin Mac."""

    model = layout(layers=32, layer_gib=6.0)  # 192 GiB across 140 GiB of admission
    with pytest.raises(NoFeasiblePlanError) as caught:
        elect([m1(), m5()], model, context_tokens=8192, origin_node="m1")

    error = caught.value
    assert "no feasible coordinator" in str(error)
    assert error.result.feasible is False
    assert error.result.coordinator_node == ""
    assert error.result.rank_order == ()
    assert error.result.selected is None
    assert error.result.endpoint is None
    assert error.result.origin_node == "m1"
    assert {item.node_id for item in error.result.rejected} == {"m1", "m5"}
    # The serializable payload an API would return must not skip validation
    # into "start the local Mac".
    payload = error.result.to_dict()
    assert payload["feasible"] is False
    assert payload["coordinator_node"] == ""
    assert json.loads(json.dumps(payload))["selected"] is None


def test_rank0_extra_bytes_change_feasibility() -> None:
    """§7.5 step 1: tail + LM head + MTP + coordinator memory is a hard limit.

    The same checkpoint and the same two Macs elect a plan with no extra
    rank-0 bytes, and lose the fast candidate once rank 0 must also hold
    the head/coordinator memory.
    """

    fast = facts(
        "fast",
        capacity_gib=128,
        admission_gib=100,
        decode_gib_per_s=4.0,
        prefill_gib_per_s=8.0,
    )
    big = facts(
        "big",
        capacity_gib=256,
        admission_gib=200,
        decode_gib_per_s=1.0,
        prefill_gib_per_s=2.0,
    )
    model = layout(layers=32)  # 65 GiB in total

    without = elect([fast, big], model, context_tokens=8192)
    assert "fast" in {item.coordinator_node for item in without.considered}

    with_extra = elect(
        [fast, big], model, context_tokens=8192, rank0_extra_bytes=120 * GIB
    )
    assert "fast" not in {item.coordinator_node for item in with_extra.considered}
    assert [item.node_id for item in with_extra.rejected] == ["fast"]
    assert with_extra.coordinator_node == "big"
    assert with_extra.selected.rank0_extra_bytes == 120 * GIB


def test_rank0_extra_bytes_are_charged_against_rank0_headroom() -> None:
    """The head/coordinator bytes come out of rank 0's admission, not nowhere."""

    node = m5()
    model = layout(layers=32)  # 65 GiB, all on one Mac
    baseline = elect([node], model, context_tokens=8192)
    charged = elect([node], model, context_tokens=8192, rank0_extra_bytes=20 * GIB)

    assert charged.selected.groups == baseline.selected.groups == (32,)
    assert (
        baseline.selected.rank0_headroom_bytes - charged.selected.rank0_headroom_bytes
        == 20 * GIB
    )
    with pytest.raises(NoFeasiblePlanError):
        elect([node], model, context_tokens=8192, rank0_extra_bytes=60 * GIB)


def test_mtp_head_only_counts_when_the_plan_enables_it() -> None:
    """The Lightning MTP head is rank-0 weight, and only when MTP is planned."""

    node = m5()  # 100 GiB admission, 65 GiB model
    model = layout(layers=32, mtp_gib=20)  # 85 GiB with the draft head

    pp_only = elect([node], model, context_tokens=8192)
    assert pp_only.selected.rank0_extra_bytes == 0

    with_mtp = elect([node], model, context_tokens=8192, mtp=True)
    assert with_mtp.selected.rank0_extra_bytes == 20 * GIB
    assert (
        pp_only.selected.rank0_headroom_bytes - with_mtp.selected.rank0_headroom_bytes
        == 20 * GIB
    )

    too_big = layout(layers=32, mtp_gib=40)  # 105 GiB: fits only without MTP
    assert elect([node], too_big, context_tokens=8192).selected.rank0_extra_bytes == 0
    with pytest.raises(NoFeasiblePlanError):
        elect([node], too_big, context_tokens=8192, mtp=True)


def test_three_node_rank_orders_are_valid() -> None:
    """§7.4: every permutation is tried; the result is a legal 3-stage split."""

    nodes = [
        m5(),
        facts(
            "m2",
            capacity_gib=64,
            admission_gib=52,
            decode_gib_per_s=1.6,
            prefill_gib_per_s=3.2,
            host="172.16.1.13",
        ),
        m1(),
    ]
    model = layout(layers=32, layer_gib=0.75)  # 25 GiB

    result = elect(nodes, model, context_tokens=8192, origin_node="m1")

    assert_valid_rank_order(result, {"m1", "m2", "m5"})
    # 4 rank-0 candidates would be 4 * 2! ; with 3 Macs it is 3 * 2!.
    assert len(result.considered) >= 1
    assert all(len(item.rank_order) == 3 for item in result.considered)
    assert all(len(item.stages) == 3 for item in result.considered)


def test_four_node_rank_orders_are_valid() -> None:
    """§4.1: the 4-Mac ring is supported, and every Mac can be rank 0."""

    nodes = [
        m5(),
        facts(
            "m2",
            capacity_gib=64,
            admission_gib=52,
            decode_gib_per_s=1.6,
            prefill_gib_per_s=3.2,
            host="172.16.1.13",
        ),
        m1(),
        facts(
            "m1b",
            capacity_gib=64,
            admission_gib=44,
            decode_gib_per_s=0.9,
            prefill_gib_per_s=1.8,
            host="172.16.1.12",
        ),
    ]
    model = layout(layers=32, layer_gib=0.6)  # 20 GiB

    result = elect(nodes, model, context_tokens=8192, origin_node="m1b")

    assert_valid_rank_order(result, {"m1", "m1b", "m2", "m5"})
    # 4 rank-0 candidates * 3! peer orders, all resolvable to a legal split.
    assert {item.coordinator_node for item in result.considered} == {
        "m1",
        "m1b",
        "m2",
        "m5",
    }
    assert len(result.considered) == 24


def test_tensor_parallel_request_is_rejected() -> None:
    """§8.1/§4.2: TP is out of scope and must be refused, not silently PP'd."""

    for degree in (2, 3, 4):
        with pytest.raises(UnsupportedParallelismError) as caught:
            elect(
                [m5(), m1()], layout(), context_tokens=8192, tensor_parallel_size=degree
            )
        assert "pipeline parallelism only" in str(caught.value)
        assert "tensor parallelism" in str(caught.value)

    with pytest.raises(ValueError):
        elect([m5()], layout(), context_tokens=8192, tensor_parallel_size=0)


def test_free_admission_breaks_an_otherwise_exact_tie() -> None:
    """§7.5 step 5: admission is the last preference, never the first.

    Built as candidates rather than as a model because a real Mac pair almost
    always differs in headroom as well; this pins the ordering itself. The
    candidate with the faster critical stage wins even with less admission, and
    only the exact tie is decided by free admission.
    """

    nodes = [
        facts("a", capacity_gib=128, admission_gib=100),
        facts("b", capacity_gib=128, admission_gib=90),
        facts("c", capacity_gib=128, admission_gib=100),
    ]

    def candidate(
        node_id: str,
        *,
        critical: float | None,
        headroom_gib: float,
        admission_gib: float,
    ) -> PlanCandidate:
        return PlanCandidate(
            rank_order=(node_id,),
            stages=(
                StagePlan(
                    rank=0,
                    node_id=node_id,
                    first_layer=0,
                    last_layer=1,
                    layer_count=1,
                    resident_bytes=0,
                    admission_bytes=int(admission_gib * GIB),
                    headroom_bytes=int(headroom_gib * GIB),
                ),
            ),
            feasible=True,
            reason="synthetic",
            rank0_headroom_bytes=int(headroom_gib * GIB),
            rank0_admission_bytes=int(admission_gib * GIB),
            min_headroom_bytes=int(headroom_gib * GIB),
            critical_stage_seconds=critical,
        )

    slower_roomy = candidate("a", critical=12.0, headroom_gib=40, admission_gib=100)
    faster_tight = candidate("b", critical=9.0, headroom_gib=10, admission_gib=90)
    tie_a = candidate("a", critical=10.0, headroom_gib=20, admission_gib=100)
    tie_b = candidate("b", critical=10.0, headroom_gib=20, admission_gib=90)

    ranked = rank_candidates([slower_roomy, faster_tight], nodes)
    assert [item.coordinator_node for item in ranked] == ["b", "a"]

    tied = rank_candidates([tie_b, tie_a], nodes)
    assert [item.coordinator_node for item in tied] == ["a", "b"]

    no_preference = rank_candidates(
        [tie_b, tie_a], nodes, policy=ElectionPolicy(free_admission_preference=False)
    )
    # Without the preference the rank order itself is the deterministic tie-break.
    assert [item.coordinator_node for item in no_preference] == ["a", "b"]


def test_an_unmeasured_plan_never_outranks_a_measured_one() -> None:
    """§17.2: no throughput promises — "unknown" is not "fast"."""

    nodes = [
        facts("measured", capacity_gib=128, admission_gib=100),
        facts("unmeasured", capacity_gib=256, admission_gib=240),
    ]
    measured = PlanCandidate(
        rank_order=("measured",),
        stages=(),
        feasible=True,
        reason="synthetic",
        critical_stage_seconds=30.0,
    )
    unmeasured = PlanCandidate(
        rank_order=("unmeasured",),
        stages=(),
        feasible=True,
        reason="synthetic",
        min_headroom_bytes=200 * GIB,
        rank0_headroom_bytes=200 * GIB,
    )

    ranked = rank_candidates([unmeasured, measured], nodes)
    assert [item.coordinator_node for item in ranked] == ["measured", "unmeasured"]


def test_measured_admission_is_the_budget_not_installed_ram() -> None:
    """§7.3: a plan is built against measured admission (p05), not capacity."""

    node = facts("hot", capacity_gib=128, admission_gib=60)  # 68 GiB still in use
    model = layout(layers=16, layer_gib=2.0)  # 33 GiB

    result = elect([node], model, context_tokens=8192)
    stage = result.selected.stages[0]
    assert stage.admission_bytes == 60 * GIB
    assert stage.resident_bytes == 33 * GIB
    assert stage.headroom_bytes == 27 * GIB
    assert stage.headroom_bytes == stage.admission_bytes - stage.resident_bytes


def test_unhealthy_and_unstageable_macs_are_rejected_with_a_reason() -> None:
    """§7.4 candidates: paired, healthy, model present or stageable."""

    sick = facts("sick", capacity_gib=128, admission_gib=100, healthy=False)
    no_disk = facts(
        "no-disk",
        capacity_gib=128,
        admission_gib=100,
        model_present=False,
        disk_free_gib=5,
    )
    stageable = facts(
        "stageable",
        capacity_gib=128,
        admission_gib=100,
        model_present=False,
        disk_free_gib=500,
    )
    model = layout(layers=16, layer_gib=1.0)  # 17 GiB

    result = elect([sick, no_disk, stageable], model, context_tokens=8192)

    assert result.coordinator_node == "stageable"
    rejected = {item.node_id: item.reason for item in result.rejected}
    assert rejected["sick"] == "Mac is not healthy"
    assert "not enough disk" in rejected["no-disk"]
    assert "stageable" not in rejected


def test_election_is_serializable_and_keeps_the_planner_plan() -> None:
    """§13.3: the result must survive an API round trip, and keep its plan."""

    result = elect([m1(), m5()], layout(), context_tokens=8192, origin_node="m1")
    payload = json.loads(json.dumps(result.to_dict()))

    assert payload["feasible"] is True
    assert payload["coordinator_node"] == "m5"
    assert payload["rank_order"] == ["m5", "m1"]
    assert payload["plan_signature"] == result.selected.plan_hash
    assert len(payload["plan_signature"]) == 64
    assert payload["endpoint"]["node_id"] == "m5"
    assert payload["endpoint"]["url"] == "http://172.16.1.11"
    assert payload["policy"]["name"] == "feasibility_first"
    assert payload["reasons"]

    selected = result.selected
    assert isinstance(selected.shard_plan, ShardPlan)
    assert [item.rank for item in selected.shard_plan.assignments] == [0, 1]
    assert "shard_plan" not in payload["selected"]

    restored = NodeFacts.from_dict(m5().to_dict())
    assert restored == m5()
    assert restored.compute is not None
    assert (
        restored.compute.relative_speed(restored.compute.decode_weight_bytes_per_second)
        == 1.0
    )


def test_election_rejects_duplicate_and_excessive_nodes() -> None:
    """Input contract: unique Macs, at most the MVP's four."""

    model = layout(layers=16, layer_gib=0.5)
    with pytest.raises(ValueError):
        elect([m5(), m5()], model, context_tokens=8192)
    with pytest.raises(ValueError):
        elect([], model, context_tokens=8192)
    with pytest.raises(ValueError):
        elect([m5()], model, context_tokens=0)

    five = [
        facts(f"node-{index}", capacity_gib=128, admission_gib=100)
        for index in range(5)
    ]
    with pytest.raises(ValueError) as caught:
        elect(five, model, context_tokens=8192)
    assert "at most 4 Macs" in str(caught.value)


def test_compute_plausibility_window() -> None:
    """E4=B: a bandwidth-bound rate is plausible; latency-bound garbage is not."""

    bandwidth_bound = ComputeProfile(
        decode_weight_bytes_per_second=400 * GIB,
        prefill_weight_bytes_per_second=800 * GIB,
    )
    assert bandwidth_bound.plausible()
    # The old rows=1 GEMV reported single-digit GB/s — hardware-impossible.
    latency_bound = ComputeProfile(
        decode_weight_bytes_per_second=10 * GIB,
        prefill_weight_bytes_per_second=20 * GIB,
    )
    assert not latency_bound.plausible()
    absurd = ComputeProfile(
        decode_weight_bytes_per_second=2000 * GIB,
        prefill_weight_bytes_per_second=4000 * GIB,
    )
    assert not absurd.plausible()


def test_implausible_compute_is_treated_as_unmeasured() -> None:
    """An implausible probe must not drive the budget's performance profile."""

    # facts() maps the relative signal x100; 0.01 -> ~1 GB/s (implausible),
    # 2.0 -> ~215 GB/s (plausible).
    implausible = facts("m1", capacity_gib=128, admission_gib=100, decode_gib_per_s=0.01)
    assert budget_for(implausible, rank=1, admission="p50").performance is None
    plausible = facts("m2", capacity_gib=128, admission_gib=100, decode_gib_per_s=2.0)
    assert budget_for(plausible, rank=1, admission="p50").performance is not None


def test_the_wired_limit_only_counts_when_it_is_what_caps_the_mac() -> None:
    """R5: a Mac whose ceiling sits at its GPU cap can be helped by raising it.

    ``capacity_bytes`` is the minimum of the static, live and GPU ceilings, so
    being *at* the cap is the one case where more limit means more plan. A Mac
    capped by RAM or by live pressure is not sent to an admin dialog.
    """

    at_cap = facts("cap", capacity_gib=48, admission_gib=44, wired_limit_mb=48 * 1024)
    below_cap = facts(
        "ram", capacity_gib=40, admission_gib=36, wired_limit_mb=48 * 1024
    )
    unmeasured = facts("unknown", capacity_gib=48, admission_gib=44)
    never_raised = facts("default", capacity_gib=48, admission_gib=44, wired_limit_mb=0)

    assert wired_limit_binds(at_cap) is True
    assert wired_limit_binds(below_cap) is False
    assert wired_limit_binds(unmeasured) is False
    # 0 is the macOS default (nobody raised it) — the case the preflight exists
    # for. "Not measured" is None and never reads as too low.
    assert wired_limit_binds(never_raised) is True

def test_a_pinned_coordinator_is_rank_zero_and_says_so() -> None:
    """Pin: the endpoint host is the user's choice, not the ranking's."""

    model = layout(layers=24, layer_gib=1.0)
    big = facts("big", capacity_gib=128, admission_gib=100, decode_gib_per_s=2.0)
    small = facts("small", capacity_gib=64, admission_gib=48, decode_gib_per_s=1.0)

    automatic = elect([big, small], model, context_tokens=8192)
    pinned = elect(
        [big, small], model, context_tokens=8192, preferred_coordinator="small"
    )

    assert pinned.coordinator_node == "small"
    assert pinned.rank_order[0] == "small"
    assert pinned.endpoint is not None and pinned.endpoint.node_id == "small"
    assert any("pinned" in reason for reason in pinned.reasons)
    assert automatic.coordinator_node  # the ranking still answers on its own


def test_an_unknown_pin_is_a_refusal_not_a_silent_election() -> None:
    with pytest.raises(ValueError, match="preferred_coordinator"):
        elect(
            [m1(), m5()],
            layout(layers=8, layer_gib=1.0),
            context_tokens=1024,
            preferred_coordinator="not-a-measured-mac",
        )


def test_a_pin_that_cannot_hold_rank_zero_is_named_in_the_refusal() -> None:
    """The frozen rule: never silently elect a different Mac than the pin."""

    model = layout(layers=8, layer_gib=4.0)
    tiny = facts("tiny", capacity_gib=16, admission_gib=8)
    big = facts("big", capacity_gib=128, admission_gib=100)

    with pytest.raises(NoFeasiblePlanError) as caught:
        elect(
            [tiny, big],
            model,
            context_tokens=8192,
            rank0_extra_bytes=10 * GIB,
            preferred_coordinator="tiny",
        )

    assert "pinned to tiny" in str(caught.value)



def test_a_refusal_names_the_wired_limit_and_offers_to_raise_it() -> None:
    """A too-small Mac gets the click that helps, not only "choose a model"."""

    model = layout(layers=32, layer_gib=6.0)  # needs far more than 48 GiB
    capped = facts("capped", capacity_gib=48, admission_gib=44, wired_limit_mb=48_000)
    other = facts("other", capacity_gib=48, admission_gib=44, wired_limit_mb=48_000)

    with pytest.raises(NoFeasiblePlanError) as caught:
        elect([capped, other], model, context_tokens=8192)

    error = caught.value
    assert error.action == "raise_wired_limit"
    assert "Performance" in str(error)
    assert "48,000 MB" in str(error)


def test_a_refusal_stays_adjust_model_when_the_cap_is_not_the_constraint() -> None:
    model = layout(layers=32, layer_gib=6.0)
    pressured = facts(
        "pressure", capacity_gib=40, admission_gib=36, wired_limit_mb=96 * 1024
    )
    other = facts("other", capacity_gib=40, admission_gib=36, wired_limit_mb=96 * 1024)

    with pytest.raises(NoFeasiblePlanError) as caught:
        elect([pressured, other], model, context_tokens=8192)

    assert caught.value.action == "adjust_model_or_context"
    assert "Performance" not in str(caught.value)
