# SPDX-License-Identifier: Apache-2.0
"""A rank must refuse a stage too large for its Mac, before and while loading."""

import threading
import time

import pytest

from omlx.cluster.memory_guard import (
    LoadMemoryWatchdog,
    admission_budget,
    ceiling_breakdown,
    check_rank_fits,
    guard_rank_load,
    load_peak_bytes,
    stage_budget,
    watch_rank_load,
)
from omlx.exceptions import InsufficientMemoryError

GIB = 1024**3


def _deterministic_machine(monkeypatch):
    """Pin every memory input so ceilings do not depend on the host.

    The tier tests below assert tier semantics (reserves, reclaim ratios,
    operator clamping). On a small-RAM CI runner the dynamic vm_stat
    ceiling binds instead and fluctuates between calls, which is not what
    they are about. Profile: 64 GiB RAM, 2 GiB oMLX footprint, 16/8/24 GiB
    free/inactive/active, 48 GiB Metal cap.
    """
    import omlx.process_memory_enforcer as enforcer_module
    import omlx.settings as settings_module
    from omlx.process_memory_enforcer import ProcessMemoryEnforcer

    monkeypatch.setattr(settings_module, "get_system_memory", lambda: 64 * GIB)
    monkeypatch.setattr(enforcer_module, "get_phys_footprint", lambda: 2 * GIB)
    monkeypatch.setattr(
        enforcer_module,
        "get_macos_vm_stats",
        lambda: {"free": 16 * GIB, "inactive": 8 * GIB, "active": 24 * GIB},
    )
    monkeypatch.setattr(
        ProcessMemoryEnforcer,
        "_get_effective_metal_cap_bytes",
        lambda self: 48 * GIB,
    )



def test_a_stage_that_fits_is_admitted():
    ceiling = check_rank_fits(50 * GIB, rank=0, ceiling_bytes=100 * GIB)
    assert ceiling == 100 * GIB


def test_cuda_ceiling_uses_device_memory_instead_of_host_ram(monkeypatch):
    import mlx.core as mx

    monkeypatch.setattr(mx.cuda, "is_available", lambda: True)
    monkeypatch.setattr(mx, "device_info", lambda: {"memory_size": 128 * GIB})
    monkeypatch.setattr(
        "omlx.cluster.memory_guard._operator_memory_settings",
        lambda: ("balanced", 0.0, True),
    )

    breakdown = ceiling_breakdown()

    assert breakdown["metal_cap"] == 128 * GIB
    assert breakdown["hard_limit"] == int(128 * GIB * 0.90)


def test_cuda_ceiling_respects_live_free_memory(monkeypatch):
    import mlx.core as mx

    monkeypatch.setattr(mx.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        mx,
        "device_info",
        lambda: {
            "total_memory": 128 * GIB,
            "free_memory": 8 * GIB,
        },
    )
    monkeypatch.setattr(
        "omlx.cluster.memory_guard._operator_memory_settings",
        lambda: ("balanced", 0.0, True),
    )

    breakdown = ceiling_breakdown()

    assert breakdown["metal_cap"] == 128 * GIB
    assert breakdown["static"] == int(128 * GIB * 0.90)
    assert breakdown["dynamic"] == int(8 * GIB * 0.90)
    assert breakdown["hard_limit"] == int(8 * GIB * 0.90)


def test_a_stage_that_overruns_the_ceiling_is_refused():
    """The case that OOM'd a MacBook: 101 GiB onto a 107 GiB Mac in use."""

    with pytest.raises(InsufficientMemoryError) as excinfo:
        check_rank_fits(
            101 * GIB,
            rank=0,
            node_id="test-mbp",
            role="workstation",
            ceiling_bytes=107 * GIB,
        )

    message = str(excinfo.value)
    assert "test-mbp" in message
    assert "101.0 GiB" in message
    assert "fewer layers" in message, "the error must say what to do about it"


def test_a_manual_slider_is_the_rank_guard_limit_even_on_a_workstation():
    """Role selects automatic memory; an explicit slider value supersedes it."""

    from omlx.cluster.planner import PipelineAssignment

    assignment = PipelineAssignment(
        node_id="test-mbp",
        rank=0,
        start_layer=0,
        end_layer=1,
        layer_weight_bytes=79 * GIB,
        fixed_weight_bytes=1 * GIB,
        reserve_bytes=18 * GIB,
        capacity_bytes=108 * GIB,
        manual_memory_limit=True,
        role="workstation",
    )

    # Automatic Workstation policy admits 54 GiB here; the explicit slider
    # says 90 GiB and both planning and the live rank must honour that number.
    assert guard_rank_load(
        assignment,
        rank=0,
        ceiling_bytes=108 * GIB,
    ) == 108 * GIB


def test_the_safety_margin_leaves_room_for_kv_and_activations():
    """Admitting at 100% of the ceiling leaves nothing for the first request."""

    # 95 GiB of weights under a 100 GiB ceiling fits arithmetically but not
    # once cache and activations land on top.
    with pytest.raises(InsufficientMemoryError):
        check_rank_fits(95 * GIB, rank=1, ceiling_bytes=100 * GIB, safety=0.90)

    # Same stage passes if the caller deliberately widens the margin.
    assert check_rank_fits(95 * GIB, rank=1, ceiling_bytes=100 * GIB, safety=1.0)


def test_an_unmeasurable_host_does_not_block_the_load():
    """Matches single-node behaviour: no hard limits when unguarded."""

    assert check_rank_fits(500 * GIB, rank=0, ceiling_bytes=0) == 0


def test_the_error_carries_the_numbers_for_the_caller():
    with pytest.raises(InsufficientMemoryError) as excinfo:
        check_rank_fits(200 * GIB, rank=2, ceiling_bytes=100 * GIB)

    assert excinfo.value.required == 200 * GIB
    assert excinfo.value.current == 100 * GIB


def test_it_works_straight_from_a_planner_assignment():
    class Assignment:
        node_id = "mac-studio"
        planned_weight_bytes = 300 * GIB

    with pytest.raises(InsufficientMemoryError, match="mac-studio"):
        guard_rank_load(Assignment(), rank=1, ceiling_bytes=200 * GIB)


def test_the_real_ceiling_is_readable_on_this_machine():
    """Reuses oMLX's own enforcer rather than reimplementing the arithmetic."""

    from omlx.cluster.memory_guard import ceiling_breakdown

    ceiling = int(ceiling_breakdown().get("hard_limit", 0))
    assert ceiling >= 0


def test_a_plan_tier_cannot_admit_above_the_operators_own_ceiling(monkeypatch):
    """The plan configures the deployment, not the ceiling of someone's Mac.

    A worker capped at 8 GiB by its operator must admit at 8 GiB even when the
    coordinator's plan carries tier "balanced" for every node.
    """

    _deterministic_machine(monkeypatch)

    from omlx.cluster import memory_guard

    monkeypatch.setattr(
        memory_guard, "_operator_memory_settings", lambda: ("custom", 8.0, True)
    )

    local = memory_guard.ceiling_breakdown()
    planned = memory_guard.ceiling_breakdown("balanced")

    assert local["hard_limit"] <= 8 * GIB
    assert planned["hard_limit"] == local["hard_limit"]


def test_a_disabled_local_guard_is_not_resurrected_by_a_plan_tier(monkeypatch):
    """Guard off is an explicit opt-out of hard limits, not a value of zero."""

    _deterministic_machine(monkeypatch)

    from omlx.cluster import memory_guard

    monkeypatch.setattr(
        memory_guard, "_operator_memory_settings", lambda: ("custom", 4.0, False)
    )

    planned = memory_guard.ceiling_breakdown("balanced")

    assert planned["hard_limit"] > 4 * GIB


# ---------------------------------------------------------------------------
# The guard must track memory actually available, not just installed capacity.
# A ceiling equal to total RAM admits a load that cannot possibly fit when
# other applications already hold half the machine.
# ---------------------------------------------------------------------------


def test_the_binding_limit_is_named_in_the_error(monkeypatch):
    from omlx.cluster import memory_guard

    # Other apps have taken the machine down to 30 GiB reclaimable, even though
    # 122 GiB is installed and the GPU cap allows 107.
    monkeypatch.setattr(
        memory_guard,
        "ceiling_breakdown",
        lambda tier="balanced": {
            "static": 122 * GIB,
            "dynamic": 30 * GIB,
            "metal_cap": 107 * GIB,
            "hard_limit": 30 * GIB,
        },
    )

    with pytest.raises(InsufficientMemoryError) as excinfo:
        check_rank_fits(50 * GIB, rank=0, node_id="test-mbp")

    message = str(excinfo.value)
    assert "memory currently available" in message, "must name what is binding"
    assert "close other apps" in message


def test_a_gpu_capped_machine_says_so(monkeypatch):
    from omlx.cluster import memory_guard

    monkeypatch.setattr(
        memory_guard,
        "ceiling_breakdown",
        lambda tier="balanced": {
            "static": 122 * GIB,
            "dynamic": 115 * GIB,
            "metal_cap": 107 * GIB,
            "hard_limit": 107 * GIB,
        },
    )

    with pytest.raises(InsufficientMemoryError, match="GPU allocation cap"):
        check_rank_fits(115 * GIB, rank=0)


def test_pressure_from_other_apps_changes_the_verdict(monkeypatch):
    """The same stage is admitted on an idle Mac and refused on a busy one."""

    from omlx.cluster import memory_guard

    idle = {"static": 122 * GIB, "dynamic": 115 * GIB,
            "metal_cap": 107 * GIB, "hard_limit": 107 * GIB}
    busy = {"static": 122 * GIB, "dynamic": 40 * GIB,
            "metal_cap": 107 * GIB, "hard_limit": 40 * GIB}

    monkeypatch.setattr(memory_guard, "ceiling_breakdown", lambda tier="balanced": idle)
    assert check_rank_fits(60 * GIB, rank=0)

    monkeypatch.setattr(memory_guard, "ceiling_breakdown", lambda tier="balanced": busy)
    with pytest.raises(InsufficientMemoryError):
        check_rank_fits(60 * GIB, rank=0)


def test_the_live_breakdown_has_all_three_components():
    from omlx.cluster.memory_guard import ceiling_breakdown

    breakdown = ceiling_breakdown()
    assert set(breakdown) >= {"static", "dynamic", "metal_cap", "hard_limit"}
    non_zero = [v for k, v in breakdown.items() if k != "hard_limit" and v]
    if non_zero:
        assert breakdown["hard_limit"] == min(non_zero), "hard limit is the minimum"


def test_a_host_without_the_full_engine_stack_loads_unguarded_not_crashed(caplog):
    """The guard exists to prevent a crash; it must not become one.

    A worker-only environment can lack the engine imports ProcessMemoryEnforcer
    pulls in — one Mac here fails on a missing mlx_vlm. Proceeding is the
    documented behaviour for an unmeasurable host, but it is said out loud.
    """

    import logging

    from omlx.cluster import memory_guard

    def _boom(_tier):
        raise ModuleNotFoundError("No module named 'mlx_vlm'")

    original = memory_guard.ceiling_breakdown
    memory_guard.ceiling_breakdown = _boom
    try:
        with caplog.at_level(logging.WARNING):
            assert memory_guard.check_rank_fits(500 * 1024**3, rank=1) == 0
    finally:
        memory_guard.ceiling_breakdown = original

    assert "unguarded" in caplog.text
    assert "mlx_vlm" in caplog.text


# ---------------------------------------------------------------------------
# Admission is a prediction about weights nobody has read yet. It has to keep
# being checked while they arrive: a rank admitted at 56 GiB still took a
# MacBook down, with its marker saying "loading" the whole way.
# ---------------------------------------------------------------------------


def test_a_load_that_stays_inside_its_budget_is_left_alone():
    watchdog = LoadMemoryWatchdog(100 * GIB, usage=lambda: 60 * GIB)

    assert watchdog.run_once() == 60 * GIB


def test_a_rank_that_grows_past_its_budget_mid_load_is_given_up():
    climb = iter([50 * GIB, 80 * GIB, 110 * GIB])
    watchdog = LoadMemoryWatchdog(
        100 * GIB, rank=0, node_id="test-mbp", usage=lambda: next(climb)
    )

    assert watchdog.run_once() == 50 * GIB
    assert watchdog.run_once() == 80 * GIB
    with pytest.raises(InsufficientMemoryError) as excinfo:
        watchdog.run_once()

    message = str(excinfo.value)
    assert "test-mbp" in message
    assert "while loading" in message, "must distinguish this from admission"
    assert "110.0 GiB" in message
    assert "fewer layers" in message, "the error must say what to do about it"
    assert excinfo.value.required == 110 * GIB
    assert excinfo.value.current == 100 * GIB


def test_the_largest_admitted_stage_is_one_the_watcher_will_not_abort():
    """The two budgets have to be derived from each other, or the guard admits
    a stage the watcher then kills at its own load peak.

    ``stage_budget`` is what a rank may still hold when the load finishes;
    ``admission_budget`` is what it may touch on the way there. The second is
    the first plus the cost of getting in, capped at the Mac.
    """

    ceiling = 100 * GIB
    for role in ("headless", "workstation"):
        stage = stage_budget(ceiling, role=role)
        watched = admission_budget(ceiling, role=role)

        # The biggest stage this role admits, admitted.
        check_rank_fits(stage, rank=0, role=role, ceiling_bytes=ceiling)
        # ...and the watcher does not abort it while it loads.
        assert min(load_peak_bytes(stage), ceiling) <= watched
        watchdog = LoadMemoryWatchdog(watched, usage=lambda seen=watched: seen)
        assert watchdog.run_once() == watched
        assert watched <= ceiling, "nothing may be watched above the Mac itself"


def test_an_overrun_aborts_the_rank_while_it_is_still_loading():
    """Not after it: the point is to stop before macOS stops the Mac."""

    interrupted = threading.Event()

    with pytest.raises(InsufficientMemoryError, match="test-mbp"), watch_rank_load(
        1 * GIB,
        rank=0,
        node_id="test-mbp",
        interval=0.0,
        usage=lambda: 9 * GIB,
        interrupt=interrupted.set,
    ):
        assert interrupted.wait(5), "the loading thread must be interrupted"


def test_a_load_that_finishes_leaves_no_watcher_behind():
    samples: list[int] = []

    def sample() -> int:
        samples.append(len(samples))
        if len(samples) == 3:
            watchdog.stop()
        return 1 * GIB

    watchdog = LoadMemoryWatchdog(100 * GIB, interval=0.0, usage=sample)
    watchdog.run()

    assert len(samples) == 3, "sampling must not continue past the load"
    assert watchdog.breach is None


def test_the_watcher_never_makes_the_load_wait_on_it():
    """A monitor that adds a poll interval to every load is worse than none."""

    started = time.perf_counter()
    with watch_rank_load(100 * GIB, interval=30.0, usage=lambda: 1 * GIB):
        pass

    assert time.perf_counter() - started < 1.0


def test_an_unmeasurable_host_loads_watched_by_nothing_rather_than_blocked():
    """Same rule as admission: a Mac we cannot measure is never blocked."""

    def explode() -> int:
        raise AssertionError("nothing to compare a sample against")

    with watch_rank_load(0, rank=0, usage=explode) as watchdog:
        pass

    assert watchdog.active is False
    assert watchdog.breach is None


def test_a_sample_that_cannot_be_read_does_not_kill_a_healthy_rank():
    """This guard exists to prevent a crash; it must not become one."""

    calls: list[int] = []

    def sample() -> int:
        calls.append(len(calls))
        if len(calls) == 1:
            raise OSError("rusage failed")
        watchdog.stop()
        return 1 * GIB

    watchdog = LoadMemoryWatchdog(100 * GIB, interval=0.0, usage=sample)
    watchdog.run()

    assert watchdog.breach is None
    assert len(calls) == 2, "an unreadable sample is skipped, not fatal"


def test_what_the_rank_is_holding_is_measured_not_guessed():
    from omlx.cluster.memory_guard import current_usage_bytes

    assert current_usage_bytes() > 0


# ---------------------------------------------------------------------------
# Loading a stage costs more than the stage. Dequantisation buffers, sanitize
# copies and a MoE's per-shard staging are alive on the way in and gone once
# the weights are resident, so the plan is checked against a size the machine
# never actually has to hold on its own.
# ---------------------------------------------------------------------------


def _mac(monkeypatch, ceiling: int) -> None:
    """A Mac whose GPU can address ``ceiling`` and which is otherwise idle."""

    from omlx.cluster import memory_guard

    monkeypatch.setattr(
        memory_guard,
        "ceiling_breakdown",
        lambda tier="balanced": {
            "static": 122 * GIB,
            "dynamic": 122 * GIB,
            "metal_cap": ceiling,
            "hard_limit": ceiling,
        },
    )


def test_the_stage_that_took_the_macbook_down_is_still_refused(monkeypatch):
    """The incident, in the bytes it actually had.

    107.5 GiB the GPU can address, 56.1 GiB of stage — 52% of the Mac — on a
    laptop someone was working on. A Workstation node may hold half its Mac,
    so this is refused by 2.4 GiB. The role is the thing that refuses it now,
    not a 1.3x inflation charged on top of a different fraction.
    """

    _mac(monkeypatch, 115_427_246_080)

    with pytest.raises(InsufficientMemoryError) as excinfo:
        check_rank_fits(
            60_262_615_040, rank=0, node_id="test-mbp", role="workstation"
        )

    assert excinfo.value.required == 60_262_615_040
    assert excinfo.value.current == 115_427_246_080 // 2
    assert "56.1 GiB" in str(excinfo.value)


def test_a_mac_with_the_room_still_gets_the_stage(monkeypatch):
    """The role must not cost a node nobody is using."""

    _mac(monkeypatch, int(107.5 * GIB))

    assert check_rank_fits(56 * GIB, rank=0, role="headless")


def test_a_load_peak_the_watchdog_would_abort_is_warned_about_not_refused(
    monkeypatch, caplog
):
    """The inverse failure: refusing on a guess costs more than it saves.

    A headless rank admits what this Mac admits alone. Its load may still peak
    past the ceiling — that is real, and the watchdog measures it — but
    predicting it with a 1.3x bound refused stages that load perfectly well,
    which is how adding a second Mac made the first one accept less.
    """

    import logging

    _mac(monkeypatch, int(107.5 * GIB))
    planned = 95 * GIB
    assert load_peak_bytes(planned) > admission_budget(
        int(107.5 * GIB), role="headless"
    )

    with caplog.at_level(logging.WARNING):
        assert check_rank_fits(planned, rank=0, node_id="studio", role="headless")

    assert "peaking near" in caplog.text, "the prediction is still made, and said"
    assert "watchdog" in caplog.text

    # A caller that would rather lose the load early can still ask for the
    # strict reading.
    with pytest.raises(InsufficientMemoryError, match="peaking near"):
        check_rank_fits(planned, rank=0, role="headless", charge_load_peak=True)


def test_a_small_stage_still_pays_the_fixed_part_of_a_load():
    """The runtime, the graph and the buffer pool do not shrink with the stage."""

    assert load_peak_bytes(1 * GIB) >= 3 * GIB
    # ...and the rest of the cost does scale with the weights being read.
    assert load_peak_bytes(100 * GIB) - 100 * GIB > load_peak_bytes(10 * GIB) - 10 * GIB


def test_an_empty_kv_reservation_is_not_charged_as_if_it_were_loading():
    """26 GiB of pure over-charge at long context, and it was the dominant term.

    ``planned_weight_bytes`` includes the KV cache this rank will hold at the
    planned context. Nothing about a reservation for a prompt nobody has sent
    costs anything to *load*, so the multiplier belongs on the weights.
    """

    weights, cache = 40 * GIB, 20 * GIB
    planned = weights + cache

    told_the_split = load_peak_bytes(planned, cache)
    told_nothing = load_peak_bytes(planned)

    assert told_the_split == weights + int(weights * 0.30) + cache
    assert told_nothing - told_the_split == int(cache * 0.30)
    # Still the safe direction when the split is unknown.
    assert told_nothing > told_the_split


def test_a_caller_with_its_own_arithmetic_is_charged_what_it_asked_about():
    """An explicit margin means the caller has already decided."""

    assert check_rank_fits(88 * GIB, rank=0, ceiling_bytes=100 * GIB)
    assert check_rank_fits(95 * GIB, rank=0, ceiling_bytes=100 * GIB, safety=1.0)
    assert stage_budget(100 * GIB, safety=0.5) == 50 * GIB

    # The same stage, charged for its load, does not fit that Mac.
    with pytest.raises(InsufficientMemoryError):
        check_rank_fits(
            88 * GIB, rank=0, ceiling_bytes=100 * GIB, charge_load_peak=True
        )


# ---------------------------------------------------------------------------
# The planner and the guard were derived independently and disagreed by ~22
# GiB: the planner would offer a workstation 75.5 GiB of a 107.5 GiB Mac and
# the rank would refuse anything past 53.75. Every test that missed it called
# ``check_rank_fits`` by hand. These start from a plan the planner built and
# carry it to the rank through the wire the launcher actually uses.
# ---------------------------------------------------------------------------

# Measured on the MacBook this was written for.
_MACBOOK_CEILING_BYTES = 115_427_246_080  # 107.500 GiB
_MACBOOK_INCIDENT_STAGE = 60_262_615_040  # 56.124 GiB, the stage that took it down


def _plan_for(role: str, *, ceiling: int = _MACBOOK_CEILING_BYTES):
    """A real plan: role -> reserve -> NodeBudget -> planner -> assignments.

    The reserve is derived exactly as ``routes._model_and_nodes`` derives it,
    so the node budget the planner sees is the one a deployment really gets.
    """

    from omlx.cluster.node_role import role_for
    from omlx.cluster.planner import (
        NodeBudget,
        plan_unequal_pipeline,
        synthetic_model_layout,
    )

    node = NodeBudget(
        node_id="macbook",
        capacity_bytes=ceiling,
        reserve_bytes=role_for(role).reserve_for(ceiling),
        rank=0,
        role=role,
    )
    peer = NodeBudget(
        node_id="studio",
        capacity_bytes=512 * GIB,
        reserve_bytes=role_for("headless").reserve_for(512 * GIB),
        rank=1,
        role="headless",
    )
    # Big enough that the planner fills the MacBook to its usable limit: the
    # boundary is the only interesting place for this question.
    model = synthetic_model_layout(total_weight_bytes=400 * GIB, layer_count=60)
    return plan_unequal_pipeline(model, [node, peer], context_tokens=4096)


def _as_the_rank_sees_it(assignments):
    """Round-trip through the encoded worker contract, like a launch does."""

    from omlx.cluster.deployment import (
        ClusterDeployment,
        ClusterHost,
        decode_worker_contract,
    )

    assignments = tuple(assignments)
    hosts = tuple(
        ClusterHost(
            item.node_id,
            "127.0.0.1" if index == 0 else f"user@{item.node_id}.local",
            (f"10.0.0.{index + 1}",),
        )
        for index, item in enumerate(assignments)
    )
    deployment = ClusterDeployment(
        deployment_id="cluster-guard",
        model="org/model",
        backend="ring",
        hosts=hosts,
        assignments=assignments,
        plan_hash="d" * 64,
    )
    _plan_hash, decoded, _profiles, _tp = decode_worker_contract(
        deployment.encode_worker_plan()
    )
    return decoded


@pytest.mark.parametrize("role", ["headless", "workstation"])
def test_a_plan_the_planner_built_is_a_plan_the_rank_admits(role):
    """(a) Nothing the planner can produce may be refused at load.

    A guard stricter than the planner turns every launch into a mystery: the
    user approves a stage, the rank rejects it, and the deployment dies after
    the weights have been staged.
    """

    plan = _plan_for(role)
    macbook = next(item for item in _as_the_rank_sees_it(plan.assignments)
                   if item.node_id == "macbook")

    assert macbook.role == role, "the role has to survive the wire to matter"
    assert macbook.planned_weight_bytes > 0
    # No role passed by hand: the rank reads it off the plan.
    assert guard_rank_load(
        macbook, rank=0, ceiling_bytes=_MACBOOK_CEILING_BYTES
    ) == _MACBOOK_CEILING_BYTES


@pytest.mark.parametrize("role", ["headless", "workstation"])
def test_the_largest_stage_the_planner_could_assign_is_admitted(role):
    """The same invariant at the boundary, where a real plan may not land.

    ``plan_unequal_pipeline`` refuses a stage larger than ``usable_bytes``
    (planner.py, "stage does not fit node"), so that is the most this rank can
    ever be handed. It is built here as a real assignment and carried through
    the encoded contract, because the boundary is the only place the answer to
    "do these two agree" is interesting.
    """

    from omlx.cluster.node_role import role_for
    from omlx.cluster.planner import NodeBudget, PipelineAssignment

    ceiling = _MACBOOK_CEILING_BYTES
    node_role = role_for(role)
    budget = NodeBudget(
        node_id="macbook",
        capacity_bytes=ceiling,
        reserve_bytes=node_role.reserve_for(ceiling),
        rank=0,
        role=role,
    )
    # The planner's own limit and the role's are the same number, or the rest
    # of this proves nothing.
    assert budget.usable_bytes == node_role.usable_for(ceiling)

    largest = PipelineAssignment(
        node_id="macbook",
        rank=0,
        start_layer=40,
        end_layer=60,
        fixed_weight_bytes=1 * GIB,
        layer_weight_bytes=budget.usable_bytes - 5 * GIB,
        kv_cache_bytes=4 * GIB,
        reserve_bytes=budget.reserve_bytes,
        capacity_bytes=ceiling,
        role=role,
    )
    assert largest.planned_weight_bytes == budget.usable_bytes

    peer = PipelineAssignment(
        node_id="studio",
        rank=1,
        start_layer=0,
        end_layer=40,
        fixed_weight_bytes=1 * GIB,
        layer_weight_bytes=100 * GIB,
        reserve_bytes=51 * GIB,
        capacity_bytes=512 * GIB,
        role="headless",
    )
    decoded = next(
        item
        for item in _as_the_rank_sees_it([largest, peer])
        if item.node_id == "macbook"
    )
    assert guard_rank_load(decoded, rank=0, ceiling_bytes=ceiling) == ceiling


@pytest.mark.parametrize("role", ["headless", "workstation"])
@pytest.mark.parametrize("capacity_gib", [24, 32, 64, 96, 128, 192, 512])
def test_the_guard_never_admits_less_than_the_planner_may_assign(role, capacity_gib):
    """The same invariant, stated on the arithmetic rather than one plan.

    ``reserve_for`` and ``admission_bytes`` are the planner's and the guard's
    only inputs. If the first ever leaves more usable than the second admits,
    some plan somewhere is unlaunchable.
    """

    from omlx.cluster.node_role import role_for

    capacity = capacity_gib * GIB
    node_role = role_for(role)

    assert stage_budget(capacity, role=role) >= node_role.usable_for(capacity)
    assert node_role.usable_for(capacity) > 0, "a role that contributes nothing"


def test_a_headless_rank_admits_what_the_same_mac_admits_on_its_own():
    """(b) Adding a second Mac must not reduce what the first will accept.

    ``EnginePool._admit_or_evict`` admits when ``max(active, phys, tracked) +
    resident_size <= ceiling`` — no fraction, no load-peak inflation — and for
    a cluster entry ``_entry_resident_size`` charges rank 0's
    ``planned_weight_bytes``: the very number this guard is handed. A rank
    that refused 33 GiB the pool would have admitted was a regression wearing
    a safety label.
    """

    ceiling = _MACBOOK_CEILING_BYTES

    def single_node_admits(resident: int, current: int = 0) -> bool:
        return current + resident <= ceiling

    for planned in (1 * GIB, 50 * GIB, 96 * GIB, ceiling):
        assert single_node_admits(planned)
        assert check_rank_fits(
            planned, rank=0, role="headless", ceiling_bytes=ceiling
        ) == ceiling

    assert stage_budget(ceiling, role="headless") == ceiling
    # ...and the pool's own ceiling still binds: past the Mac is past the Mac.
    with pytest.raises(InsufficientMemoryError):
        check_rank_fits(ceiling + GIB, rank=0, role="headless", ceiling_bytes=ceiling)


def test_the_incident_stage_is_refused_on_a_mac_someone_is_using():
    """(c) The protection that was 3 GiB away and unreachable, driven from a plan."""

    from omlx.cluster.node_role import role_for

    ceiling = _MACBOOK_CEILING_BYTES
    assert role_for("workstation").usable_for(ceiling) < _MACBOOK_INCIDENT_STAGE, (
        "the planner must not propose the incident stage in the first place"
    )

    with pytest.raises(InsufficientMemoryError) as excinfo:
        check_rank_fits(
            _MACBOOK_INCIDENT_STAGE,
            rank=0,
            node_id="macbook",
            role="workstation",
            ceiling_bytes=ceiling,
        )

    assert excinfo.value.required == _MACBOOK_INCIDENT_STAGE
    assert excinfo.value.current == ceiling // 2
    # The same stage on a Mac nobody is using is fine, and says so by loading.
    assert check_rank_fits(
        _MACBOOK_INCIDENT_STAGE, rank=0, role="headless", ceiling_bytes=ceiling
    )


def test_a_plan_that_reserved_nothing_does_not_widen_the_guard():
    """The role has to be a floor, not a suggestion.

    A deployment can arrive with the reserve set by hand — the dashboard sends
    a slider value, an auto-tuned re-plan can drop it to zero. The guard reads
    the *role's* reserve, never the plan's, so a plan that held nothing back
    is still bounded by what the Mac's role allows.
    """

    from omlx.cluster.planner import PipelineAssignment

    reserved_nothing = PipelineAssignment(
        node_id="macbook",
        rank=0,
        start_layer=40,
        end_layer=60,
        fixed_weight_bytes=1 * GIB,
        layer_weight_bytes=_MACBOOK_INCIDENT_STAGE - 1 * GIB,
        reserve_bytes=0,
        capacity_bytes=_MACBOOK_CEILING_BYTES,
        role="workstation",
    )

    assert reserved_nothing.planned_weight_bytes == _MACBOOK_INCIDENT_STAGE
    with pytest.raises(InsufficientMemoryError, match="Workstation"):
        guard_rank_load(
            reserved_nothing, rank=0, ceiling_bytes=_MACBOOK_CEILING_BYTES
        )


def test_the_refusal_states_a_comparison_that_is_actually_true():
    """The message used to contradict itself.

    "would load 56.1 GiB ... above the 69.9 GiB this Mac can admit" — 56.1 is
    not above 69.9. The number doing the refusing was a third figure the
    sentence never showed. Every number in the message now takes part in the
    comparison it claims.
    """

    import re

    with pytest.raises(InsufficientMemoryError) as excinfo:
        check_rank_fits(
            _MACBOOK_INCIDENT_STAGE,
            rank=0,
            node_id="macbook",
            role="workstation",
            ceiling_bytes=_MACBOOK_CEILING_BYTES,
        )

    message = str(excinfo.value)
    held, admitted = (
        float(value) for value in re.findall(r"([\d.]+) GiB", message)[:2]
    )
    assert held > admitted, message

    share = int(re.search(r"(\d+)% of this Mac's ([\d.]+) GiB", message).group(1))
    ceiling = float(re.search(r"(\d+)% of this Mac's ([\d.]+) GiB", message).group(2))
    assert abs(ceiling * share / 100 - admitted) < 0.1, (
        f"the message's own arithmetic does not hold: {message}"
    )


# ---------------------------------------------------------------------------
# The ceiling the guard measures must be the one the operator configured. It
# used to hard-code "balanced", so the only control that says "cap this
# machine" reached every part of oMLX except a cluster rank.
# ---------------------------------------------------------------------------


def _operator_memory(monkeypatch, **fields):
    from types import SimpleNamespace

    import omlx.settings as settings_module

    settings = {
        "memory_guard_tier": "balanced",
        "memory_guard_custom_ceiling_gb": 0.0,
        "prefill_memory_guard": True,
    }
    settings.update(fields)
    memory = SimpleNamespace(**settings)
    monkeypatch.setattr(
        settings_module,
        "get_settings",
        lambda: SimpleNamespace(memory=memory),
    )


def test_a_custom_ceiling_the_operator_set_binds_the_rank(monkeypatch):
    """A user who capped their Mac at 8 GiB got a rank admitting against 107."""

    from omlx.cluster.memory_guard import ceiling_breakdown

    _operator_memory(
        monkeypatch, memory_guard_tier="custom", memory_guard_custom_ceiling_gb=8.0
    )
    capped = int(ceiling_breakdown().get("hard_limit", 0))

    assert 0 < capped <= 8 * GIB
    with pytest.raises(InsufficientMemoryError):
        check_rank_fits(40 * GIB, rank=0, role="headless")


def test_a_custom_tier_with_no_ceiling_typed_in_is_still_guarded(monkeypatch):
    """0 would read as "unmeasurable host, load unguarded" — the strictest
    setting in the product becoming the only one with no rank guard at all."""

    from omlx.cluster.memory_guard import ceiling_breakdown

    _operator_memory(
        monkeypatch, memory_guard_tier="custom", memory_guard_custom_ceiling_gb=0.0
    )

    assert ceiling_breakdown().get("hard_limit", 0) > 0


def test_the_tier_the_operator_chose_is_the_tier_that_is_measured(monkeypatch):
    """safe reclaims 20% of active pages, aggressive 80%. A rank used to get
    balanced's 50% whatever the operator had chosen."""

    _deterministic_machine(monkeypatch)

    from omlx.cluster.memory_guard import ceiling_breakdown

    _operator_memory(monkeypatch, memory_guard_tier="safe")
    safe = ceiling_breakdown()
    _operator_memory(monkeypatch, memory_guard_tier="aggressive")
    aggressive = ceiling_breakdown()

    assert safe["static"] < aggressive["static"], "the tier reserve must differ"
    assert safe["hard_limit"] <= aggressive["hard_limit"]
    # An explicit tier from a caller still wins over the operator's default.
    assert ceiling_breakdown("safe")["static"] == safe["static"]


def test_operator_memory_settings_reads_real_settings_when_uninitialized(
    monkeypatch, tmp_path
):
    """A fresh worker/probe process never calls init_settings, so the operator's
    persisted tier must come from a real settings.json read, and the read must
    not publish the process-wide singleton as a side effect."""
    import json

    import omlx.settings as settings_module
    from omlx.cluster.memory_guard import _operator_memory_settings

    (tmp_path / "settings.json").write_text(
        json.dumps(
            {
                "memory": {
                    "memory_guard_tier": "custom",
                    "memory_guard_custom_ceiling_gb": 44.0,
                }
            }
        )
    )
    monkeypatch.setenv("OMLX_BASE_PATH", str(tmp_path))
    monkeypatch.setattr(settings_module, "_global_settings", None)

    tier, custom_gb, enabled = _operator_memory_settings()

    assert tier == "custom"
    assert custom_gb == 44.0
    assert enabled is True
    assert settings_module._global_settings is None


def test_operator_memory_settings_falls_back_when_load_fails(monkeypatch):
    import omlx.settings as settings_module
    from omlx.cluster.memory_guard import _operator_memory_settings

    def fake_get_settings():
        raise RuntimeError("Settings not initialized")

    def fake_load(cls, *args, **kwargs):
        raise OSError("settings.json unreadable")

    monkeypatch.setattr(settings_module, "get_settings", fake_get_settings)
    monkeypatch.setattr(
        settings_module.GlobalSettings, "load", classmethod(fake_load)
    )

    tier, custom_gb, enabled = _operator_memory_settings()
    assert tier == "balanced"
    assert custom_gb == 0.0
    assert enabled is True
