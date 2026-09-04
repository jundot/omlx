# SPDX-License-Identifier: Apache-2.0
"""A Mac someone is working on must not be planned like a shelf appliance."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from omlx.cluster.node_role import (
    HEADLESS,
    ROLES,
    WORKSTATION,
    metal_cap_bytes,
    role_for,
    suggest_budget,
)

GiB = 1024**3


def _sysctl(value, returncode=0):
    def run(command, **_):
        return SimpleNamespace(stdout=str(value), returncode=returncode)
    return run


# --- Capacity comes from the GPU cap, not installed RAM ---------------------


def test_capacity_is_what_the_gpu_can_address():
    """128 GiB installed, 107.5 addressable — planning on 128 is how a Mac dies."""

    budget = suggest_budget(role="headless", ssh_target="peer", runner=_sysctl(110080))
    assert budget.capacity_bytes == 110080 * 1024 * 1024
    assert budget.capacity_source == "metal_cap"


def test_installed_ram_is_the_fallback_and_says_so():
    calls = []

    def run(command, **_):
        calls.append(command)
        # First call is the GPU cap and fails; second is hw.memsize.
        if "iogpu.wired_limit_mb" in " ".join(command):
            return SimpleNamespace(stdout="", returncode=1)
        return SimpleNamespace(stdout=str(128 * GiB), returncode=0)

    budget = suggest_budget(role="headless", ssh_target="peer", runner=run)
    assert budget.capacity_source == "installed_ram"
    assert "GPU cap unreadable" in budget.describe()


def test_an_unreadable_machine_reports_nothing_rather_than_guessing():
    budget = suggest_budget(role="headless", ssh_target="peer", runner=_sysctl("", returncode=1))
    assert budget.capacity_bytes == 0
    assert budget.usable_bytes == 0


def test_the_gpu_cap_is_read_over_ssh_for_a_peer():
    seen = {}

    def run(command, **_):
        seen["command"] = command
        return SimpleNamespace(stdout="249037", returncode=0)

    metal_cap_bytes(ssh_target="studio", runner=run)
    assert seen["command"][0] == "ssh"
    assert "iogpu.wired_limit_mb" in " ".join(seen["command"])


def test_localhost_is_not_reached_over_ssh():
    seen = {}

    def run(command, **_):
        seen["command"] = command
        return SimpleNamespace(stdout="110080", returncode=0)

    metal_cap_bytes(ssh_target="127.0.0.1", runner=run)
    assert seen["command"][0] == "sysctl"


# --- The roles --------------------------------------------------------------


def test_a_workstation_keeps_enough_back_to_stay_usable():
    """Half the Mac, and never less than the 32 GiB a person needs to work."""

    budget = suggest_budget(role="workstation", ssh_target="peer", runner=_sysctl(110080))
    assert budget.reserve_bytes == budget.capacity_bytes // 2
    assert budget.reserve_bytes > 32 * GiB
    assert "for your work" in budget.describe()


def test_a_small_workstation_still_keeps_the_flat_reserve():
    """Half of a 48 GiB Mac is less than a person needs; the floor wins."""

    budget = suggest_budget(role="workstation", capacity_bytes=48 * GiB)
    assert budget.reserve_bytes == 32 * GiB


def test_a_headless_node_gives_almost_everything():
    budget = suggest_budget(role="headless", ssh_target="peer", runner=_sysctl(249037))
    assert budget.reserve_bytes == int(budget.capacity_bytes * 0.10)
    assert budget.usable_bytes > budget.capacity_bytes * 0.85


def test_a_workstation_contributes_less_than_the_same_mac_headless():
    """The trade must be real, or the setting is decoration."""

    headless = suggest_budget(role="headless", ssh_target="peer", runner=_sysctl(110080))
    workstation = suggest_budget(role="workstation", ssh_target="peer", runner=_sysctl(110080))
    assert workstation.usable_bytes < headless.usable_bytes
    # Roughly 20 GiB of reach, which is what the user is buying back.
    assert (headless.usable_bytes - workstation.usable_bytes) > 18 * GiB


def test_a_small_mac_still_gets_a_fractional_reserve_not_just_the_flat_one():
    """A 32 GiB mini cannot hold 32 GiB back — it would contribute nothing."""

    budget = suggest_budget(role="workstation", capacity_bytes=32 * GiB)
    assert budget.usable_bytes > 0
    assert budget.reserve_bytes < budget.capacity_bytes


def test_a_big_headless_mac_reserves_proportionally_not_a_flat_amount():
    small = suggest_budget(role="headless", capacity_bytes=64 * GiB)
    large = suggest_budget(role="headless", capacity_bytes=512 * GiB)
    assert large.reserve_bytes > small.reserve_bytes


def test_an_unknown_role_falls_back_to_the_safe_default():
    assert role_for("nonsense") is HEADLESS
    assert role_for(None) is HEADLESS
    assert role_for("WORKSTATION") is WORKSTATION


def test_every_role_explains_itself_for_the_tooltip():
    for role in ROLES.values():
        assert role.label and role.summary
        assert len(role.detail) > 80, "the tooltip must say why, not just what"


def test_a_budget_serialises_for_the_interface():
    payload = suggest_budget(role="workstation", ssh_target="peer", runner=_sysctl(110080)).to_dict()
    assert payload["role"] == "workstation"
    assert payload["capacity_source"] == "metal_cap"
    assert payload["usable_bytes"] > 0
    assert "your work" in payload["summary"]


def test_the_local_budget_agrees_with_what_the_guard_admits(monkeypatch):
    """Two definitions of "the cap" is how a plan gets refused at load."""

    monkeypatch.setattr(
        "omlx.cluster.memory_guard.ceiling_breakdown",
        lambda *_a, **_k: {"hard_limit": 100 * GiB},
    )
    budget = suggest_budget(role="headless", runner=_sysctl(999999))
    assert budget.capacity_bytes == 100 * GiB, "must not use the larger raw cap"
    assert budget.capacity_source == "admission_ceiling"
    assert "can admit" in budget.describe()


def test_a_peer_still_falls_back_to_its_gpu_cap(monkeypatch):
    """The enforcer reads *this* process; a remote Mac needs the sysctl."""

    monkeypatch.setattr(
        "omlx.cluster.memory_guard.ceiling_breakdown",
        lambda *_a, **_k: {"hard_limit": 100 * GiB},
    )
    budget = suggest_budget(role="headless", ssh_target="studio", runner=_sysctl(249037))
    assert budget.capacity_source == "metal_cap"
    assert budget.capacity_bytes == 249037 * 1024 * 1024


# --- The role must bind at admission, not only at planning ------------------


def test_a_workstation_admits_less_than_a_headless_mac():
    """The gap that took a MacBook down.

    The role promised 77 GiB. The guard admitted against 90% of the ceiling —
    96.8 GiB — because it had never heard of the role. A reserve nothing
    enforces is not a reserve.

    The fraction is the incident, not a preference: 56.1 GiB of stage on a
    107.5 GiB ceiling is 52% of the Mac, and that is what went down. The cap
    has to sit below what has been measured to fail.
    """

    assert WORKSTATION.admission_fraction < HEADLESS.admission_fraction
    assert WORKSTATION.admission_fraction < 60_262_615_040 / 115_427_246_080


def test_the_two_numbers_a_role_carries_cannot_drift_apart():
    """The planner reads ``reserve_for``; the guard reads ``admission_bytes``.

    They were derived independently and disagreed by ~22 GiB — the planner
    offering a workstation 75.5 GiB of a 107.5 GiB Mac that the rank refused
    past 53.75 GiB. Whatever the capacity, the guard may not admit less than
    the planner may assign.
    """

    for role in ROLES.values():
        for capacity in (16 * GiB, 24 * GiB, 32 * GiB, 64 * GiB, 128 * GiB, 512 * GiB):
            assert role.admission_bytes(capacity) >= role.usable_for(capacity)
            assert role.usable_for(capacity) == capacity - role.reserve_for(capacity)
            assert role.admission_bytes(capacity) <= capacity


def test_a_headless_rank_is_not_stricter_than_the_same_mac_serving_alone():
    """``engine_pool`` admits a model's resident size against the whole
    ceiling. A headless rank that admitted less made adding a second Mac
    reduce what the first would accept."""

    ceiling = 115_427_246_080
    assert HEADLESS.admission_bytes(ceiling) == ceiling
    # The planner still keeps its own 10% back — that is planning headroom,
    # not a limit the guard enforces.
    assert HEADLESS.usable_for(ceiling) == ceiling - int(ceiling * 0.10)


def test_the_guard_uses_the_roles_fraction():
    from omlx.cluster.memory_guard import check_rank_fits
    from omlx.exceptions import InsufficientMemoryError

    ceiling = int(107.5 * GiB)
    # 80 GiB: fine on a headless Mac, refused on one someone is using.
    check_rank_fits(80 * GiB, rank=0, role="headless", ceiling_bytes=ceiling)
    with pytest.raises(InsufficientMemoryError):
        check_rank_fits(80 * GiB, rank=0, role="workstation", ceiling_bytes=ceiling)


def test_an_unset_role_keeps_the_previous_behaviour():
    """Callers that never passed a role must not silently get stricter."""

    from omlx.cluster.memory_guard import check_rank_fits

    check_rank_fits(90 * GiB, rank=0, ceiling_bytes=int(107.5 * GiB))


def test_the_refusal_names_the_way_out():
    from omlx.cluster.memory_guard import check_rank_fits
    from omlx.exceptions import InsufficientMemoryError

    with pytest.raises(InsufficientMemoryError) as excinfo:
        check_rank_fits(
            100 * GiB, rank=0, node_id="mbp", role="workstation",
            ceiling_bytes=int(107.5 * GiB),
        )
    assert "Headless" in str(excinfo.value)
