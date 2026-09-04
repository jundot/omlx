# SPDX-License-Identifier: Apache-2.0
"""What a Mac should contribute, given whether someone is using it.

Two numbers were wrong wherever a node budget was set by hand.

**Capacity was installed RAM.** A 128 GiB MacBook reports 128 GiB, but the GPU
can only address ``iogpu.wired_limit_mb`` — 107.5 GiB on this one. Every plan
built from installed RAM was ~20 GiB optimistic, proposed a stage the memory
guard then refused at load, and before that guard existed, took the machine
down.

**Reserve did not know the Mac was in use.** A dedicated Studio and a laptop
someone is working on are not the same node. Run a laptop to the same
utilisation as a headless box and the model wins every contention against the
person typing on it.

So a node declares a role and the reserve follows from it. This is not a new
planner concept — ``NodeBudget.reserve_bytes`` already means "memory the model
may not touch". It was simply never set from anything real.

Note what the dynamic ceiling can and cannot do. ``ceiling_breakdown()`` reads
``vm_stat``, so a rank *loading* while 40 GiB of applications are open is
already refused. It cannot help once the weights are resident and the apps open
afterwards — nothing is reclaimable then. That is why a workstation reserve is
static and held back up front rather than inferred at load time.

**One number, two consumers.** A role names a single share of the Mac —
``admission_fraction`` — and both sides read it:

* the planner, through ``reserve_for()``: what it holds back is what the role
  does not allow a rank to hold;
* the memory guard, through ``admission_bytes()``: what a rank may hold once
  loaded, whatever the plan says.

``admission_bytes()`` is deliberately ``max(what the plan may assign, the
role's own share)``, so the guard can never refuse a stage the planner just
built. They used to be derived independently and disagreed by ~22 GiB: the
planner would hand a workstation 75.5 GiB of a 107.5 GiB Mac and the guard
would refuse anything past 53.75 GiB, so the plan the user approved could not
launch. Change one of these numbers and the other follows.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from typing import Any

from .ssh_policy import cluster_ssh_options

GiB = 1024**3

# Enough for a browser, an editor, a chat client and the OS to stay responsive
# while a rank holds the rest. A floor, not the whole reserve: on any Mac big
# enough to matter the fractional reserve below is larger. Below roughly this
# the machine starts swapping under its user, which is the failure this exists
# to prevent.
_WORKSTATION_RESERVE_BYTES = 32 * GiB

# A headless node still cannot be *planned* to its cap: the KV cache the plan
# does not count, activations and the runtime sit on top of the weights. This
# is planning headroom only. It is deliberately NOT the guard's limit — see
# ``_HEADLESS_ADMISSION_FRACTION``.
_HEADLESS_RESERVE_FRACTION = 0.10

# Half of what the GPU can address, and no more. The rest belongs to the person
# at the keyboard *and* to the load: a stage costs about 1.3x its resident size
# while it is arriving (see ``memory_guard._LOAD_PEAK_FRACTION``), so a rank
# capped at half the ceiling still touches roughly two thirds of it mid-load.
#
# The number is the incident, not a preference. 56.1 GiB planned onto a 107.5
# GiB ceiling — 52% — is what took the MacBook down, so the cap sits below what
# has been measured to fail rather than above it.
_WORKSTATION_ADMISSION_FRACTION = 0.50

# A headless rank admits exactly what oMLX admits on the same Mac with no
# cluster at all: ``engine_pool`` charges the model's resident size against the
# full ceiling (``engine_pool.py``, ``_admit_or_evict``), with no fraction and
# no load-peak inflation, and for a cluster entry the size it charges is rank
# 0's ``planned_weight_bytes`` — the very number this guard sees.
#
# Anything less makes a second Mac *reduce* what the first will accept: a stage
# a Mac serves fine alone was refused as a rank, which is a regression wearing
# a safety label. The planner's 10% above is what keeps the normal path off the
# ceiling; this is the last line, and a last line that refuses what works alone
# is not one. The load peak is watched instead of predicted here — see
# ``memory_guard.check_rank_fits``.
_HEADLESS_ADMISSION_FRACTION = 1.00


@dataclass(frozen=True)
class NodeRole:
    """How much of a Mac the cluster may take, and why."""

    key: str
    label: str
    summary: str
    detail: str
    reserve_bytes: int = 0
    reserve_fraction: float = 0.0
    # The share of the Mac a rank of this role may hold once loaded. Enforced
    # by the memory guard at load, not just used for planning — a reserve the
    # guard does not know about is a promise nothing keeps. A MacBook whose
    # role promised 77 GiB was admitted at 96.8 GiB and went down.
    #
    # This is the resident cap, not the load-time one: getting a stage in
    # costs more than keeping it (``memory_guard.load_peak_bytes``), and the
    # guard derives that allowance from this number rather than the reverse.
    admission_fraction: float = _HEADLESS_ADMISSION_FRACTION

    def reserve_for(self, capacity_bytes: int) -> int:
        """Bytes held back on a node of this size.

        The larger of the two rules: a workstation reserve is an absolute
        amount someone needs to keep working, and the fractional reserve is
        headroom that scales with the machine. Neither alone is right on both
        a 32 GiB mini and a 512 GiB Studio.
        """

        floor = int(capacity_bytes * self.reserve_fraction)
        reserve = max(self.reserve_bytes, floor)
        # Always leave the model somewhere to live, even on a small Mac.
        return min(reserve, max(0, capacity_bytes - GiB))

    def usable_for(self, capacity_bytes: int) -> int:
        """What the planner may assign a node of this size.

        The same arithmetic ``NodeBudget.usable_bytes`` does, kept here so the
        guard can ask the question without importing the planner.
        """

        return max(0, int(capacity_bytes) - self.reserve_for(capacity_bytes))

    def admission_bytes(self, capacity_bytes: int) -> int:
        """What a rank of this role may hold, whatever the plan says.

        Never below ``usable_for``: the guard exists to stop a load the plan
        did not anticipate, not to refuse the plan itself. A guard that
        refuses what the planner just built turns every launch into a mystery
        — the user approves a stage, the rank rejects it, and nothing in the
        message says the two were computed from different rules.

        Never below the role's own share either, which is what makes a
        hand-written reserve of zero on a Workstation still bounded.
        """

        capacity = max(0, int(capacity_bytes))
        return min(
            capacity,
            max(self.usable_for(capacity), int(capacity * self.admission_fraction)),
        )


HEADLESS = NodeRole(
    key="headless",
    label="Headless",
    summary="Nobody is using this Mac",
    detail=(
        "Give the cluster everything the GPU can address, less a 10% margin "
        "for KV cache, activations and the runtime. Choose this for a Mac "
        "sitting on a shelf serving models — a Studio or a mini you do not sit "
        "in front of. A rank here admits exactly what this Mac would admit "
        "serving the same model on its own, so joining a cluster never costs "
        "it reach."
    ),
    reserve_fraction=_HEADLESS_RESERVE_FRACTION,
    admission_fraction=_HEADLESS_ADMISSION_FRACTION,
)

WORKSTATION = NodeRole(
    key="workstation",
    label="Workstation",
    summary="You work on this Mac while it serves",
    detail=(
        "Hold half this Mac back so your browser, editor and everything else "
        "stay responsive while a rank runs — at least 32 GiB, and more on a "
        "large machine because getting a stage in costs more than holding it. "
        "A model that fills a laptop wins every contention against the person "
        "using it, and macOS starts swapping under you. This costs the cluster "
        "real reach — roughly half of what this Mac could otherwise "
        "contribute — and it is almost always the right trade on a machine you "
        "are actually working on."
    ),
    reserve_bytes=_WORKSTATION_RESERVE_BYTES,
    # The planner holds back exactly what the guard will not admit, so the
    # stage the user approves is the stage that launches. These two being
    # derived separately is what let the planner offer a workstation 75.5 GiB
    # of a 107.5 GiB Mac that the guard then refused at 53.75 GiB.
    reserve_fraction=1.0 - _WORKSTATION_ADMISSION_FRACTION,
    admission_fraction=_WORKSTATION_ADMISSION_FRACTION,
)

ROLES: dict[str, NodeRole] = {HEADLESS.key: HEADLESS, WORKSTATION.key: WORKSTATION}
DEFAULT_ROLE = HEADLESS.key


def role_for(key: str | None) -> NodeRole:
    """The named role, falling back to headless for anything unrecognised."""

    return ROLES.get(str(key or "").strip().lower(), HEADLESS)


_LOCAL_HOSTS = {"", "127.0.0.1", "localhost", "::1"}


def _is_local(ssh_target: str | None) -> bool:
    return (ssh_target or "") in _LOCAL_HOSTS


def _enforcer_ceiling_bytes() -> int:
    """The ceiling oMLX admits against on this Mac. 0 when unavailable.

    Only meaningful locally — it reads this process's own memory state — and
    only importable where the full oMLX environment is installed, which a
    worker-only peer may not have.
    """

    try:
        from .memory_guard import ceiling_breakdown

        return int(ceiling_breakdown().get("hard_limit", 0))
    except Exception:
        return 0


def _parse_wired_limit_mb(text: str) -> int:
    match = re.search(r"(\d+)", text or "")
    return int(match.group(1)) if match else 0


def metal_cap_bytes(*, ssh_target: str | None = None, runner: Any = None) -> int:
    """What the GPU can actually address on a Mac. 0 when unreadable.

    ``iogpu.wired_limit_mb`` is the real ceiling for model weights; installed
    RAM is not. Returns 0 rather than guessing, so callers can fall back to
    installed RAM explicitly instead of silently planning against a number
    nobody checked.
    """

    runner = runner or subprocess.run
    command = ["sysctl", "-n", "iogpu.wired_limit_mb"]
    if ssh_target and ssh_target not in {"127.0.0.1", "localhost", "::1"}:
        command = [
            "ssh",
            *cluster_ssh_options(connect_timeout=10),
            ssh_target,
            "sysctl -n iogpu.wired_limit_mb",
        ]
    try:
        result = runner(command, capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.SubprocessError):
        return 0
    if getattr(result, "returncode", 1) != 0:
        return 0
    return _parse_wired_limit_mb(getattr(result, "stdout", "")) * 1024 * 1024


def installed_memory_bytes(*, ssh_target: str | None = None, runner: Any = None) -> int:
    """Total RAM, used only when the GPU cap cannot be read."""

    runner = runner or subprocess.run
    command = ["sysctl", "-n", "hw.memsize"]
    if ssh_target and ssh_target not in {"127.0.0.1", "localhost", "::1"}:
        command = [
            "ssh",
            *cluster_ssh_options(connect_timeout=10),
            ssh_target,
            "sysctl -n hw.memsize",
        ]
    try:
        result = runner(command, capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.SubprocessError):
        return 0
    return _parse_wired_limit_mb(getattr(result, "stdout", ""))


@dataclass(frozen=True)
class NodeBudgetSuggestion:
    """A budget a node can actually honour, and where the numbers came from."""

    capacity_bytes: int
    reserve_bytes: int
    role: str
    capacity_source: str  # "admission_ceiling" | "recommended_working_set" | ...

    @property
    def usable_bytes(self) -> int:
        return max(0, self.capacity_bytes - self.reserve_bytes)

    def describe(self) -> str:
        where = {
            "admission_ceiling": "this Mac can admit",
            "recommended_working_set": "this Mac can admit for MLX",
            "metal_cap": "what the GPU can address",
        }.get(self.capacity_source, "installed RAM (GPU cap unreadable)")
        line = (
            f"{self.usable_bytes / GiB:.0f} GiB for the cluster "
            f"of {self.capacity_bytes / GiB:.0f} GiB {where}"
        )
        if self.role == WORKSTATION.key:
            line += (
                f", leaving {self.reserve_bytes / GiB:.0f} GiB for your work "
                "and for the load's peak"
            )
        return line

    def to_dict(self) -> dict[str, Any]:
        return {
            "capacity_bytes": self.capacity_bytes,
            "reserve_bytes": self.reserve_bytes,
            "usable_bytes": self.usable_bytes,
            "role": self.role,
            "capacity_source": self.capacity_source,
            "summary": self.describe(),
        }


def suggest_budget(
    *,
    role: str = DEFAULT_ROLE,
    ssh_target: str | None = None,
    capacity_bytes: int = 0,
    capacity_source: str | None = None,
    runner: Any = None,
) -> NodeBudgetSuggestion:
    """What this Mac should offer the cluster, measured rather than assumed."""

    node_role = role_for(role)
    source = capacity_source or (
        "admission_ceiling" if capacity_bytes else "metal_cap"
    )
    capacity = capacity_bytes
    if not capacity and _is_local(ssh_target):
        # Prefer the ceiling the memory guard admits against, so the planner
        # and the guard cannot disagree. The raw sysctl is higher than what
        # ``ProcessMemoryEnforcer`` will actually allow, and planning against
        # the larger number is how a stage gets refused at load.
        capacity = _enforcer_ceiling_bytes()
        if capacity:
            source = "admission_ceiling"
    if not capacity:
        capacity = metal_cap_bytes(ssh_target=ssh_target, runner=runner)
    if capacity <= 0:
        capacity = installed_memory_bytes(ssh_target=ssh_target, runner=runner)
        source = "installed_ram"
    if capacity <= 0:
        return NodeBudgetSuggestion(0, 0, node_role.key, source)
    return NodeBudgetSuggestion(
        capacity_bytes=capacity,
        reserve_bytes=node_role.reserve_for(capacity),
        role=node_role.key,
        capacity_source=source,
    )
