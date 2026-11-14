# SPDX-License-Identifier: Apache-2.0
"""Which model should this measured cluster run?

The recommendation is a pure function of stored hardware facts plus candidate
layouts, so these tests run off-cluster like the election's: no Mac, model, GPU
or SSH. They pin the six things the answer must never get wrong:

* the model is chosen by what fits, not by what the user opened the app on, and
  the same facts still elect the same Mac with the origin on the other one;
* a model that cannot fit, cannot be split into pipeline stages, or cannot be
  read is reported with a machine-readable reason and exactly one action —
  never dropped and never offered as tensor parallelism;
* the context is the largest ladder value that still plans, capped by the
  caller's target and the model's declared limit, never invented;
* staging is reported per Mac, with the bytes to copy, and unknown disk stays
  unknown instead of becoming zero;
* without measured compute there is no rate and no "fast";
* ordering is stable and idempotent: feasible candidates rank by usable
  context, then measured critical stage, then size; near misses stay smallest
  first.
"""

from __future__ import annotations

import itertools

import pytest

from omlx.cluster.assistant import recommendation as recommendation_module
from omlx.cluster.assistant.election import (
    ComputeProfile,
    NodeFacts,
    budget_for,
    elect,
)
from omlx.cluster.assistant.recommendation import (
    ACTION_CHECK_MODEL_PATH,
    ACTION_CHOOSE_PIPELINE_MODEL,
    ACTION_CHOOSE_SMALLER_MODEL,
    ACTION_FREE_DISK_SPACE,
    FAILURE_MEMORY,
    FAILURE_MODEL_UNREADABLE,
    FAILURE_PIPELINE_UNSUPPORTED,
    WARNING_CONTEXT_BELOW_TARGET,
    WARNING_DISK_TOO_SMALL,
    WARNING_PRESENCE_UNMEASURED,
    WARNING_SPEED_UNKNOWN,
    WARNING_STAGING_REQUIRED,
    ModelCandidate,
    load_model_candidate,
    recommend,
)
from omlx.cluster.planner import ModelLayout, plan_hybrid

GIB = 1024**3


def facts(
    node_id: str,
    *,
    capacity_gib: float,
    admission_gib: float,
    decode_gib_per_s: float | None = None,
    model_present: bool = True,
    disk_free_gib: float | None = 300.0,
    host: str = "",
    **kwargs,
) -> NodeFacts:
    """One Mac as the measurement layer would report it."""

    compute = None
    if decode_gib_per_s is not None:
        # Relative speed signal mapped into the plausible bandwidth window the
        # election's plausibility guard expects (E4=B); ordering preserved.
        compute = ComputeProfile(
            decode_weight_bytes_per_second=decode_gib_per_s * 100 * GIB,
            prefill_weight_bytes_per_second=decode_gib_per_s * 2 * 100 * GIB,
        )
    return NodeFacts(
        node_id=node_id,
        capacity_gib=capacity_gib,
        admission_p05_gib=admission_gib,
        compute=compute,
        model_present=model_present,
        disk_free_gib=disk_free_gib,
        endpoint_host=host,
        **kwargs,
    )


def m5(**kwargs) -> NodeFacts:
    """The reference cluster's large/fast Mac (128 GiB, 96.75 GiB admission)."""

    return facts(
        "m5",
        capacity_gib=128,
        admission_gib=96.75,
        decode_gib_per_s=2.0,
        host="172.16.1.11",
        **kwargs,
    )


def m1(**kwargs) -> NodeFacts:
    """The reference cluster's small Mac (60 GiB, 54 GiB admission)."""

    return facts(
        "m1",
        capacity_gib=60,
        admission_gib=54.0,
        decode_gib_per_s=0.8,
        host="172.16.1.14",
        **kwargs,
    )


def layout(
    *,
    total_gib: float = 40.0,
    layers: int = 20,
    name: str = "model",
    supports_pipeline: bool = True,
    supports_tensor_parallel: bool = False,
    kv_bytes_per_token_per_layer: int = 4096,
) -> ModelLayout:
    """A checkpoint with uniform layers and a known KV shape."""

    per_layer = int(total_gib * GIB) // layers
    return ModelLayout(
        source=name,
        fixed_weight_bytes=0,
        layer_weight_bytes=(per_layer,) * layers,
        supports_pipeline=supports_pipeline,
        supports_tensor_parallel=supports_tensor_parallel,
        kv_bytes_per_token_per_layer=kv_bytes_per_token_per_layer,
    )


def candidate(path: str, **kwargs) -> ModelCandidate:
    """A candidate whose layout is already resolved (no disk involved)."""

    kwargs.setdefault("layout", layout(name=path))
    return ModelCandidate(path, **kwargs)


# ---------------------------------------------------------------------------
# The answer
# ---------------------------------------------------------------------------


def test_the_forty_gib_model_is_recommended_with_m5_as_rank_zero() -> None:
    """The reference pair: M1 cannot hold this as coordinator, M5 can."""

    result = recommend([m1(), m5()], [candidate("/models/big")], context_tokens=8192)

    assert result.feasible is True
    assert result.action == ""
    selected = result.selected
    assert selected is not None
    assert selected.model_id == "/models/big"
    assert selected.feasible is True
    assert selected.pp_supported is True
    assert selected.rank0_node == "m5"
    assert selected.rank_order == ("m5", "m1")
    assert selected.pipeline_stages == 2
    assert selected.nodes_required == 2
    assert selected.max_context_tokens == 8192
    assert selected.headroom_bytes > 0
    # Largest that fits, at one context, with a signature to compare later.
    assert result.candidates[0] is selected
    assert len(selected.plan_signature) == 64

    # ...and the recommendation did not bend the election: the same facts still
    # elect M5, from either origin.
    for origin in ("m1", "m5", None):
        elected = elect(
            [m1(), m5()],
            layout(name="/models/big"),
            context_tokens=8192,
            origin_node=origin,
        )
        assert elected.coordinator_node == "m5"
        assert elected.plan_signature == selected.plan_signature


def test_the_origin_of_the_app_is_not_an_input_to_the_recommendation() -> None:
    """There is no origin parameter at all: the function cannot prefer it."""

    result = recommend([m5(), m1()], [candidate("/models/big")], context_tokens=4096)

    assert result.selected is not None
    assert result.selected.rank0_node == "m5"
    assert "origin" not in {
        name for name in vars(recommendation_module) if not name.startswith("_")
    }


def test_a_model_that_cannot_fit_is_refused_with_exactly_one_action() -> None:
    result = recommend(
        [m1(), m5()], [candidate("/models/huge", layout=layout(total_gib=400))]
    )

    assert result.feasible is False
    assert result.selected is None
    refusal = result.candidates[0]
    assert refusal.feasible is False
    assert refusal.failure_kind == FAILURE_MEMORY
    assert refusal.pp_supported is True
    assert refusal.action == ACTION_CHOOSE_SMALLER_MODEL
    assert refusal.reasons
    assert refusal.max_context_tokens == 0
    assert refusal.rank_order == ()
    # One action for the whole answer, not one per Mac.
    assert result.action == ACTION_CHOOSE_SMALLER_MODEL


def test_a_model_that_would_need_tensor_parallelism_is_refused_not_offered() -> None:
    """§8.1: PP only. A TP-shaped candidate gets a reason and an action."""

    tensor_only = candidate(
        "/models/tensor",
        layout=layout(
            total_gib=40, supports_pipeline=False, supports_tensor_parallel=True
        ),
    )

    result = recommend([m1(), m5()], [tensor_only])

    assert result.feasible is False
    refusal = result.candidates[0]
    assert refusal.feasible is False
    assert refusal.pp_supported is False
    assert refusal.failure_kind == FAILURE_PIPELINE_UNSUPPORTED
    assert refusal.action == ACTION_CHOOSE_PIPELINE_MODEL
    assert "tensor parallelism" in refusal.reasons[0]
    assert "pipeline parallelism only" in refusal.reasons[0]
    # Reported, never silently dropped, and never offered as a plan.
    assert refusal.stages == ()
    assert refusal.pipeline_stages == 0


def test_a_model_mlx_lm_cannot_split_at_all_is_refused() -> None:
    unsplittable = candidate("/models/fixed", layout=layout(supports_pipeline=False))

    result = recommend([m1(), m5()], [unsplittable])

    refusal = result.candidates[0]
    assert refusal.feasible is False
    assert refusal.failure_kind == FAILURE_PIPELINE_UNSUPPORTED
    assert refusal.action == ACTION_CHOOSE_PIPELINE_MODEL
    assert "cannot split" in refusal.reasons[0]


def test_a_model_that_fits_one_mac_is_still_a_pipeline_plan_there() -> None:
    """PP width one is not FP: a single-stage plan is legal PP."""

    result = recommend([m5()], [candidate("/models/big")])

    selected = result.selected
    assert selected is not None
    assert selected.pipeline_stages == 1
    assert selected.rank_order == ("m5",)
    assert selected.rank0_node == "m5"


def test_an_unreadable_model_is_reported_with_a_reason_and_an_action() -> None:
    unreadable = ModelCandidate(
        "/models/broken", layout_error="could not read the model: not a directory"
    )

    result = recommend([m1(), m5()], [unreadable])

    refusal = result.candidates[0]
    assert refusal.feasible is False
    assert refusal.failure_kind == FAILURE_MODEL_UNREADABLE
    assert refusal.action == ACTION_CHECK_MODEL_PATH
    assert "not a directory" in refusal.reasons[0]
    assert result.action == ACTION_CHECK_MODEL_PATH


def test_load_model_candidate_reports_a_path_it_cannot_read(tmp_path) -> None:
    """The one filesystem access: a missing model is a candidate, not an error."""

    candidate_model = load_model_candidate(
        str(tmp_path / "not-a-model"), friendly_name="Ghost"
    )

    assert candidate_model.layout is None
    assert candidate_model.layout_error.startswith("could not read the model:")
    assert candidate_model.label == "Ghost"


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


def test_the_largest_context_that_still_plans_is_recommended() -> None:
    result = recommend([m1(), m5()], [candidate("/models/big")], context_tokens=32768)

    selected = result.selected
    assert selected is not None
    assert selected.max_context_tokens == 32768
    assert selected.context_limited is False
    # The reported context is the one the elected plan was actually built for.
    elected = elect(
        [m1(), m5()],
        layout(name="/models/big"),
        context_tokens=selected.max_context_tokens,
    )
    assert elected.plan_signature == selected.plan_signature


def test_a_smaller_target_caps_the_context() -> None:
    result = recommend([m1(), m5()], [candidate("/models/big")], context_tokens=4096)

    selected = result.selected
    assert selected is not None
    assert selected.max_context_tokens == 4096
    assert not any(
        warning.code == WARNING_CONTEXT_BELOW_TARGET for warning in selected.warnings
    )


def test_the_model_declared_limit_is_respected_and_reported() -> None:
    result = recommend(
        [m1(), m5()],
        [candidate("/models/big", declared_context_tokens=8192)],
        context_tokens=32768,
    )

    selected = result.selected
    assert selected is not None
    assert selected.max_context_tokens == 8192
    assert selected.declared_context_tokens == 8192
    assert selected.context_limited is False
    assert any(
        warning.code == WARNING_CONTEXT_BELOW_TARGET for warning in selected.warnings
    )


def test_memory_can_limit_the_context_below_the_declared_limit() -> None:
    """A big KV cache, not the model, sets the usable context: say so."""

    result = recommend(
        [m1(), m5()],
        [
            candidate(
                "/models/heavy-kv",
                layout=layout(kv_bytes_per_token_per_layer=200_000),
                declared_context_tokens=1_048_576,
            )
        ],
    )

    selected = result.selected
    assert selected is not None
    assert 0 < selected.max_context_tokens < 1_048_576
    assert selected.declared_context_tokens == 1_048_576
    assert selected.context_limited is True


# ---------------------------------------------------------------------------
# Staging and disk
# ---------------------------------------------------------------------------


def test_staging_reports_the_bytes_to_copy_to_each_mac() -> None:
    result = recommend(
        [m1(), m5()],
        [candidate("/models/big", present_on=("m5",))],
    )

    selected = result.selected
    assert selected is not None
    staging = {node.node_id: node for node in selected.nodes}
    assert staging["m5"].model_present is True
    assert staging["m5"].stage_bytes == 0
    assert staging["m1"].model_present is False
    assert staging["m1"].stage_bytes == 40 * GIB
    assert selected.download_bytes == 40 * GIB
    assert any(
        warning.code == WARNING_STAGING_REQUIRED for warning in selected.warnings
    )


def test_unmeasured_presence_is_not_reported_as_present() -> None:
    result = recommend([m1(), m5()], [candidate("/models/big")])

    selected = result.selected
    assert selected is not None
    assert all(not node.model_present for node in selected.nodes)
    assert all(node.stage_bytes == 40 * GIB for node in selected.nodes)
    assert any(
        warning.code == WARNING_PRESENCE_UNMEASURED for warning in selected.warnings
    )


def test_staging_warning_only_names_selected_macs() -> None:
    nodes = [
        m5(model_present=True),
        m1(model_present=False),
        facts(
            "m13",
            capacity_gib=32,
            admission_gib=5,
            model_present=False,
            disk_free_gib=300.0,
        ),
    ]

    result = recommend(
        nodes,
        [candidate("/models/big", present_on=("m5",))],
        context_tokens=8192,
    )

    selected = result.selected
    assert selected is not None
    assert "m13" in selected.excluded_nodes
    staging = [
        warning
        for warning in selected.warnings
        if warning.code == WARNING_STAGING_REQUIRED
    ]
    assert len(staging) == 1
    assert "m1" in staging[0].message
    assert "m13" not in staging[0].message


def test_unknown_disk_stays_unknown_instead_of_becoming_zero() -> None:
    result = recommend(
        [m1(disk_free_gib=None), m5(disk_free_gib=None)],
        [candidate("/models/big", present_on=("m5",))],
    )

    selected = result.selected
    assert selected is not None
    staging = {node.node_id: node for node in selected.nodes}
    assert staging["m1"].disk_free_bytes is None
    # "Not measured" is not "full": the copy is assumed possible.
    assert staging["m1"].can_stage is True


def test_a_mac_without_room_for_the_copy_is_flagged_with_one_action() -> None:
    result = recommend(
        [m1(disk_free_gib=10.0), m5()],
        [candidate("/models/big", present_on=("m5",))],
    )

    selected = result.selected
    assert selected is not None
    flagged = [
        warning
        for warning in selected.warnings
        if warning.code == WARNING_DISK_TOO_SMALL
    ]
    assert len(flagged) == 1
    assert flagged[0].node_id == "m1"
    assert flagged[0].action == ACTION_FREE_DISK_SPACE
    assert "10.0 GiB free" in flagged[0].message


# ---------------------------------------------------------------------------
# No invented speed
# ---------------------------------------------------------------------------


def test_without_measured_compute_there_is_no_rate_anywhere() -> None:
    """§17.2: unknown is not fast."""

    result = recommend(
        [
            facts("m5", capacity_gib=128, admission_gib=96.75),
            facts("m1", capacity_gib=60, admission_gib=54.0),
        ],
        [candidate("/models/big")],
    )

    selected = result.selected
    assert selected is not None
    assert selected.measured_speed is False
    assert selected.critical_stage_seconds is None
    assert selected.critical_stage_rank is not None
    assert all(stage.predicted_seconds is None for stage in selected.stages)
    assert any(warning.code == WARNING_SPEED_UNKNOWN for warning in selected.warnings)

    text = str(result.candidates[0].to_dict()).lower()
    for forbidden in ("tok_per_s", "tok/s", "tokens per second", "fast", "ttft"):
        assert forbidden not in text


def test_measured_compute_yields_stage_seconds_not_a_token_rate() -> None:
    result = recommend([m1(), m5()], [candidate("/models/big")], context_tokens=8192)

    selected = result.selected
    assert selected is not None
    assert selected.measured_speed is True
    assert selected.critical_stage_seconds is not None
    assert selected.critical_stage_seconds > 0
    assert all(stage.predicted_seconds is not None for stage in selected.stages)
    # Stage seconds only: the plan object has no field for a rate before smoke.
    assert not any(
        warning.code == WARNING_SPEED_UNKNOWN for warning in selected.warnings
    )
    assert "tok_per_s" not in str(selected.to_dict())


# ---------------------------------------------------------------------------
# Ordering, stability and purity
# ---------------------------------------------------------------------------


def test_ordering_is_feasible_first_then_near_misses_smallest_first() -> None:
    models = [
        ModelCandidate("/models/huge", layout=layout(total_gib=400)),
        candidate("/models/big", layout=layout(total_gib=40)),
        candidate("/models/small", layout=layout(total_gib=8, layers=8)),
        ModelCandidate("/models/broken", layout_error="could not read the model"),
        candidate(
            "/models/tensor",
            layout=layout(supports_pipeline=False, supports_tensor_parallel=True),
        ),
    ]

    result = recommend([m1(), m5()], models, context_tokens=4096)

    labels = [item.model_id for item in result.candidates]
    # Feasible first. Both plan at the 4096 target, so the measured critical
    # stage breaks the tie: the smaller checkpoint's stages are lighter and
    # faster. (Before B11 this was largest-checkpoint-first, which recommended
    # a huge model capped at a tiny context over a smaller usable one.)
    assert labels[0] == "/models/small"
    assert labels[1] == "/models/big"
    # Then near misses, smallest first (an unreadable layout counts as 0 bytes,
    # exactly as the existing catalogue orders it).
    assert labels[2] == "/models/broken"
    assert labels[3] == "/models/tensor"
    assert labels[4] == "/models/huge"


def test_a_huge_model_capped_at_a_small_context_ranks_below_a_usable_one() -> None:
    """B11: ranking is fits + usable context, not raw checkpoint size.

    Live, dev-v41-15L-real (253.9 GiB, max 4k context) was placed above
    Qwen3.8 (117.2 GiB, 32k) because the order was ``weight_bytes`` descending,
    so the assistant recommended the worse model. Here the huge checkpoint
    fits the pair but its own declared limit caps it at 4k, while the smaller
    one honours the 32k target: the usable one must be selected and rank first.
    """

    huge = candidate(
        "/models/huge-capped",
        layout=layout(total_gib=120, name="/models/huge-capped"),
        declared_context_tokens=4096,
    )
    usable = candidate("/models/usable", layout=layout(name="/models/usable"))

    result = recommend([m1(), m5()], [huge, usable], context_tokens=32768)

    assert result.feasible is True
    selected = result.selected
    assert selected is not None
    assert selected.model_id == "/models/usable"
    assert selected.max_context_tokens == 32768
    assert result.candidates[0] is selected
    # The huge model is not dropped or refused: it fits, just at 4k, and it is
    # the runner-up. Size alone would have put it first — that was the bug.
    runner_up = result.candidates[1]
    assert runner_up.model_id == "/models/huge-capped"
    assert runner_up.feasible is True
    assert runner_up.max_context_tokens == 4096
    assert runner_up.weight_bytes > selected.weight_bytes


def test_equal_context_ranks_the_faster_measured_critical_stage_first() -> None:
    """Same usable context: the measured critical stage breaks the tie.

    Both checkpoints weigh the same, so the size tie-break cannot decide and
    the model ID would sort ``a-coarse`` first. The finer-grained checkpoint
    balances its stages better across the unequal Macs, so its measured
    critical stage is faster — and that, not the name, must rank it first.
    """

    fine = candidate(
        "/models/z-fine", layout=layout(total_gib=40, layers=40, name="/models/z-fine")
    )
    coarse = candidate(
        "/models/a-coarse",
        layout=layout(total_gib=40, layers=20, name="/models/a-coarse"),
    )

    result = recommend([m1(), m5()], [coarse, fine], context_tokens=4096)

    first, second = result.candidates[0], result.candidates[1]
    assert first.max_context_tokens == second.max_context_tokens == 4096
    assert first.weight_bytes == second.weight_bytes
    assert first.measured_speed is True
    assert second.measured_speed is True
    assert first.critical_stage_seconds is not None
    assert second.critical_stage_seconds is not None
    assert first.critical_stage_seconds < second.critical_stage_seconds
    assert first.model_id == "/models/z-fine"
    assert result.selected is first


def test_equal_context_without_compute_falls_back_to_smaller_size() -> None:
    """No measured compute: size is the last resort, never an invented rate."""

    nodes = [
        facts("m5", capacity_gib=128, admission_gib=96.75),
        facts("m1", capacity_gib=60, admission_gib=54.0),
    ]
    big = candidate("/models/big", layout=layout(total_gib=40))
    small = candidate("/models/small", layout=layout(total_gib=8, layers=8))

    result = recommend(nodes, [big, small], context_tokens=4096)

    first, second = result.candidates[0], result.candidates[1]
    assert first.model_id == "/models/small"
    assert second.model_id == "/models/big"
    assert first.max_context_tokens == second.max_context_tokens == 4096
    # Unknown is not fast: no stage seconds appear, and nothing estimates one.
    for item in (first, second):
        assert item.measured_speed is False
        assert item.critical_stage_seconds is None
        assert all(stage.predicted_seconds is None for stage in item.stages)


def test_the_recommendation_is_stable_and_idempotent() -> None:
    nodes = [m1(), m5()]
    models = [
        candidate("/models/big"),
        candidate("/models/small", layout=layout(total_gib=8, layers=8)),
        ModelCandidate("/models/huge", layout=layout(total_gib=400)),
    ]
    first = recommend(nodes, models, context_tokens=4096)

    for permutation in itertools.permutations(models):
        again = recommend(nodes, list(permutation), context_tokens=4096)
        assert [item.to_dict() for item in again.candidates] == [
            item.to_dict() for item in first.candidates
        ]
    # ...and calling it twice with the same input changes nothing.
    assert recommend(nodes, models, context_tokens=4096) == first


def test_an_empty_candidate_list_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one candidate"):
        recommend([m1(), m5()], [])


def test_the_module_imports_no_network_ssh_or_write_surface() -> None:
    """Purity is a property of the imports, not a promise in the docstring."""

    forbidden = {
        "socket",
        "subprocess",
        "paramiko",
        "httpx",
        "requests",
        "urllib",
        "launch",
        "staging",
        "deployment",
        "modelsync",
        "provisioning",
        "start",
        "supervisor",
    }
    assert forbidden & set(vars(recommendation_module)) == set()


# ---------------------------------------------------------------------------
# One budget builder
# ---------------------------------------------------------------------------


def test_one_budget_builder_feeds_the_election_and_any_caller() -> None:
    """Admission becomes a ``NodeBudget`` in exactly one place.

    The elected plan's hash must be reproducible from ``budget_for`` alone, or a
    recommendation that re-plans a stored fact set would be measuring against a
    different rule than the election that produced the number.
    """

    nodes = [m5(), m1()]
    model = layout(name="/models/big")
    elected = elect(nodes, model, context_tokens=8192)

    budgets = [
        budget_for(facts_by_id, rank=rank) for rank, facts_by_id in enumerate(nodes)
    ]
    direct = plan_hybrid(model, budgets, context_tokens=8192)

    assert direct.plan_hash == elected.plan_signature
    assert budgets[0].usable_bytes == nodes[0].admission_bytes
    assert budgets[0].reserve_bytes == (
        nodes[0].capacity_bytes - nodes[0].admission_bytes
    )
    # A ranking built on the same budgets cannot disagree about admission.
    assert budget_for(nodes[0]).usable_bytes == nodes[0].admission_bytes


def test_the_single_action_ignores_an_unreadable_smallest_near_miss():
    """A model that is merely absent must not answer for the whole group.

    Live: the smallest near miss was a checkpoint that was not installed on the
    origin, so its ``model_unreadable`` action ("check the model path") was
    reported while the real blocker was memory — and that action was not one the
    card knew, so the button did nothing.
    """

    from types import SimpleNamespace

    from omlx.cluster.assistant.recommendation import (
        ACTION_CHECK_MODEL_PATH,
        ACTION_CHOOSE_SMALLER_MODEL,
        FAILURE_MEMORY,
        FAILURE_MODEL_UNREADABLE,
        RecommendationResult,
    )

    def near_miss(kind: str, action: str) -> SimpleNamespace:
        # ``RecommendationResult.action`` reads exactly these three attributes.
        return SimpleNamespace(feasible=False, failure_kind=kind, action=action)

    unreadable = near_miss(FAILURE_MODEL_UNREADABLE, ACTION_CHECK_MODEL_PATH)
    memory = near_miss(FAILURE_MEMORY, ACTION_CHOOSE_SMALLER_MODEL)

    def result(candidates: tuple) -> RecommendationResult:
        return RecommendationResult(
            candidates=candidates,
            target_context_tokens=32768,
            mtp=False,
            execution_profile="balanced",
            reasons=(),
        )

    assert result((unreadable, memory)).action == ACTION_CHOOSE_SMALLER_MODEL
    assert result((unreadable,)).action == ACTION_CHECK_MODEL_PATH


def test_presence_comes_from_the_measured_catalogue():
    """Any candidate, not only the checkpoint that was measured.

    Live, a recommendation assumed GLM had to be copied to both Macs (and
    warned about disk space) although the measurement's own inventory listed it
    on every one of them, because presence was only recorded for the measured
    checkpoint.
    """

    from omlx.cluster.assistant.recommendation import _presence

    nodes = (
        m5(model_paths=("/models/glm", "/models/qwen")),
        m1(model_paths=("/models/glm",)),
    )

    assert _presence(nodes, candidate("/models/glm")) == ("m5", "m1")
    assert _presence(nodes, candidate("/models/nothing")) == ()


def test_presence_is_unknown_when_the_catalogue_was_not_measured():
    from omlx.cluster.assistant.recommendation import _presence

    nodes = (m5(model_paths=()),)

    assert _presence(nodes, candidate("/models/glm")) is None


# ---------------------------------------------------------------------------
# Single-node option (E6, PHASE-2B-RUN §22)
# ---------------------------------------------------------------------------


def test_small_model_gets_a_single_node_option() -> None:
    # A model that fits on one Mac at the target context gets a single-node
    # option: the local engine is faster than a pipeline (no ring overhead).
    small = candidate(
        "/models/small", layout=layout(total_gib=2, name="/models/small")
    )
    result = recommend([m1(), m5()], [small], context_tokens=8192)

    assert result.feasible is True
    assert result.single_node is not None
    assert result.single_node.mode == "single"
    assert result.single_node.nodes_required == 1
    assert result.single_node.pipeline_stages == 1
    assert len(result.single_node.rank_order) == 1
    assert result.single_node.max_context_tokens == 8192
    # The best single Mac is the one with the most headroom (m5).
    assert result.single_node.rank0_node == "m5"
    # The cluster recommendation stays the compatible primary answer.
    assert result.selected is not None
    assert result.selected.mode == "cluster"


def test_model_too_big_for_one_mac_has_no_single_node_option() -> None:
    # 100 GiB fits the pair (54+96 admission) but no single Mac (m5 ~96): the
    # cluster is recommended and there is no single-node option.
    big = candidate("/models/big", layout=layout(total_gib=100, name="/models/big"))
    result = recommend([m1(), m5()], [big], context_tokens=8192)

    assert result.feasible is True
    assert result.selected.mode == "cluster"
    assert result.single_node is None


def test_single_node_option_is_none_when_nothing_fits() -> None:
    huge = candidate("/models/huge", layout=layout(total_gib=400, name="/models/huge"))
    result = recommend([m1(), m5()], [huge], context_tokens=8192)
    assert result.feasible is False
    assert result.single_node is None


def test_a_table_the_plan_offloads_is_not_charged_to_single_node_ram() -> None:
    """The assistant's fit must ask the engine's question, table offload included.

    A 100 GiB checkpoint whose 30 GiB lookup table stays on SSD needs ~70 GiB in
    RAM: the engine's local admission prices exactly that (``mmap_bytes``), and
    the plan treats the table lever as free. Charging the whole table here would
    say "no single Mac" about a run the engine can serve.
    """

    layers = 20
    per_layer = int(100 * GIB) // layers
    table = int(30 * GIB) // layers
    offloaded = ModelLayout(
        source="/models/wide",
        fixed_weight_bytes=0,
        layer_weight_bytes=(per_layer,) * layers,
        tensor_parallel_ssd_layer_bytes=(table,) * layers,
        table_offload_kind="ple",
        supports_pipeline=True,
        kv_bytes_per_token_per_layer=4096,
    )
    wide = candidate("/models/wide", layout=offloaded)

    # 100 GiB does not fit m5's 96.75 GiB admission, with or without a pipeline;
    # 70 GiB does.
    result = recommend([m1(), m5()], [wide], context_tokens=8192)

    assert result.single_node is not None
    assert result.single_node.rank0_node == "m5"
    assert result.single_node.pipeline_stages == 1


def test_a_resident_table_still_blocks_a_single_node_run() -> None:
    """Without the offload bytes there is no discount: 100 GiB is still 100 GiB."""

    resident = layout(total_gib=100, name="/models/wide")
    result = recommend(
        [m1(), m5()], [candidate("/models/wide", layout=resident)], context_tokens=8192
    )

    assert result.single_node is None
