# SPDX-License-Identifier: Apache-2.0
"""Stop a cluster rank loading more than its Mac can hold.

oMLX already refuses a single-node load that will not fit: the engine pool
consults ``ProcessMemoryEnforcer.get_admission_ceiling()`` and raises
``InsufficientMemoryError`` rather than letting the machine thrash. The
distributed worker never had that check, so a rank loaded until the OS gave up —
which is exactly how a 22-layer stage with 7.5 GiB of planned headroom took a
128 GiB MacBook down.

This applies the same ceiling to a rank before it loads. It deliberately reuses
``ProcessMemoryEnforcer`` rather than reimplementing the arithmetic: the tier
reserve, the Metal cap and the vm_stat dynamic ceiling are subtle and should
have one definition.

There are two budgets here, because there are two questions:

``stage_budget``
    What this rank may still be holding when the load finishes. It is the
    node role's own number (``NodeRole.admission_bytes``) — the same one the
    planner's reserve comes from — so a plan the planner produced is always a
    plan this admits. Charged against ``planned_weight_bytes``, exactly the
    quantity and exactly the ceiling single-node oMLX charges in
    ``engine_pool``.

``admission_budget``
    What this rank may *touch*, load peak included. Dequantisation buffers,
    ``sanitize`` copies and a MoE's per-shard staging are all alive on the way
    in and gone once the stage is resident, so a stage that fits perfectly
    well can still exceed the Mac while it is loading. This is the number
    ``LoadMemoryWatchdog`` aborts at.

The peak used to be charged at admission instead of watched, on top of the
role fraction, which made the effective cap 0.90/1.30 = 0.69 of the ceiling
while the planner planned against 0.90 and single-node oMLX admits at 1.0.
Three rules for one machine: a Mac would refuse as a rank a stage it served
fine alone, and the planner would keep proposing stages the guard refused. Now
the peak is *measured* rather than predicted — predicted, it cost 33 GiB of
reach on every headless Mac to catch a case the watchdog catches anyway, with
a real number instead of a 1.3x guess.

Admission is a prediction, so it is not enough on its own. ``LoadMemoryWatchdog``
keeps sampling while the weights come in and aborts the rank the moment it
crosses the budget it was admitted at — a rank planned for 56 GiB was admitted
and still took a MacBook down, because nothing watched after the check.
"""

from __future__ import annotations

import _thread
import logging
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

# What a load costs beyond what it leaves resident. ``planned_weight_bytes`` is
# the resident size, so charging only that charges for the wrong instant.
# Getting there costs more: MLX-LM assembles the whole state dict before
# ``load_weights`` and ``sanitize`` rewrites it — a 128-expert MoE stacks its
# per-expert tensors into one array per projection, so a stacked copy and the
# tensors it was built from are alive together — quantised weights need
# dequantisation scratch on the way in, and every transient MLX frees stays in
# its buffer pool until it is returned to the OS.
#
# The peak of a given model on a given Mac is not knowable from a plan, so both
# numbers below are bounds rather than measurements. They set the headroom the
# watchdog allows above ``stage_budget``, because that is what the load will
# actually be measured against: watching at less than the load legitimately
# needs kills a rank halfway through a load that would have settled fine.
#
# The fraction sits well below the 2x a wholly doubled stage would cost — MLX
# evaluates the sanitize graph in pieces and recycles buffers as it goes — and
# far enough above zero that a stage sized to the role's cap is not aborted the
# moment its buffers double.
_LOAD_PEAK_FRACTION = 0.30

# The part of a load that does not shrink with the stage: the runtime, the model
# graph, the pool residue. 2 GiB is what ``ProcessMemoryEnforcer`` already
# treats as transient nobody can attribute — the margin it allows a live process
# past its ceiling before it starts aborting requests
# (``_EMERGENCY_OVER_CEILING_MARGIN_BYTES``), on these same machines.
_LOAD_PEAK_FLOOR_BYTES = 2 * 1024**3

# Weights arrive over seconds to minutes, so twice a second is soon enough to
# stop a rank and rare enough that the sample never shows up in load time.
_DEFAULT_SAMPLE_INTERVAL = 0.5

# The watcher wakes on ``stop()``, so this only bounds a thread that is wedged.
_JOIN_TIMEOUT = 2.0

_REMEDY = (
    "Give this node fewer layers, set it Headless if nobody is using it, "
    "close other apps, or add a Mac."
)

_CUDA_CEILING_FRACTION = {
    "safe": 0.85,
    "balanced": 0.90,
    "aggressive": 0.95,
    "custom": 0.90,
}


def _operator_memory_settings() -> tuple[str, float, bool]:
    """The memory-guard settings this Mac's operator actually chose.

    Returns ``(tier, custom_ceiling_gb, guard_enabled)``, reading settings
    from the local base path when the process never initialized them, and
    falling back to stock defaults only when settings cannot be loaded.

    The uninitialized-process read is deliberately ``GlobalSettings.load()``,
    not ``init_settings()``: a memory-guard helper must not publish the
    process-wide settings singleton as a side effect of one preference read.
    """

    try:
        from omlx.settings import GlobalSettings, get_settings

        try:
            settings = get_settings()
        except RuntimeError:
            settings = GlobalSettings.load()
        memory = settings.memory
        return (
            str(getattr(memory, "memory_guard_tier", "") or "balanced"),
            float(getattr(memory, "memory_guard_custom_ceiling_gb", 0.0) or 0.0),
            bool(getattr(memory, "prefill_memory_guard", True)),
        )
    except Exception as exc:  # pragma: no cover - settings not initialised
        logger.debug("No operator memory settings available: %s", exc)
        return ("balanced", 0.0, True)


def _cuda_ceiling_breakdown(
    tier: str,
    *,
    custom_ceiling_gb: float,
) -> dict[str, int] | None:
    """Return a CUDA-device ceiling without substituting host RAM for VRAM.

    The macOS enforcer deliberately reasons about system RAM and Metal's wired
    limit. On a discrete CUDA machine that would advertise host RAM as model
    capacity, which can exceed VRAM by hundreds of GiB. Modern MLX exposes both
    total and live free CUDA memory; the free-memory limit is essential when a
    separate service (for example vLLM) already owns most of a unified-memory
    GB10. Older MLX builds without that field retain the physical-size bound.
    """

    try:
        import mlx.core as mx

        if not mx.cuda.is_available():
            return None
        info = mx.device_info()
        physical = int(
            info.get("total_memory") or info.get("memory_size") or 0
        )
        raw_free = info.get("free_memory")
        free = int(raw_free or 0) if raw_free is not None else None
    except Exception:
        return None
    if physical <= 0:
        return None
    normalized = tier if tier in _CUDA_CEILING_FRACTION else "balanced"
    if normalized == "custom" and custom_ceiling_gb > 0:
        budget = min(physical, int(custom_ceiling_gb * 1024**3))
    else:
        budget = int(physical * _CUDA_CEILING_FRACTION[normalized])
    dynamic = (
        max(0, int(free * _CUDA_CEILING_FRACTION[normalized]))
        if free is not None
        else budget
    )
    return {
        "static": budget,
        "dynamic": dynamic,
        # Kept for the existing status/error schema. On CUDA this is the
        # accelerator's physical allocation cap, not a Metal wired limit.
        "metal_cap": physical,
        "hard_limit": min(physical, budget, dynamic),
    }


def ceiling_breakdown(
    memory_guard_tier: str = "",
    *,
    custom_ceiling_gb: float | None = None,
    guard_enabled: bool | None = None,
) -> dict[str, int]:
    """The three limits oMLX admits against, and which one binds.

    ``static`` is total RAM minus the tier reserve, ``dynamic`` is what is
    actually reclaimable right now from ``vm_stat`` (so other applications'
    pressure shows up immediately), and ``metal_cap`` is the effective GPU
    allocation cap. ``hard_limit`` is the minimum — the real answer to "will
    this load fit".

    Anything not named by the caller comes from the operator's own memory
    settings. This used to hard-code ``balanced`` and nothing else, so the one
    control that says "cap this machine" — tier ``custom`` with a ceiling in
    GB — reached every part of oMLX except a cluster rank: a user who capped
    their Mac at 40 GiB got a rank admitting against 107.5 GiB, 67 GiB above
    what they asked for, and a reclaim ratio of 0.5 where they had chosen 0.2.

    ``ProcessMemoryEnforcer`` takes an engine pool for eviction, which a rank
    does not have, but none of the ceiling computations touch it. Passing
    ``None`` keeps one definition of the limit instead of a second copy here.
    """

    operator_tier, operator_custom_gb, operator_enabled = _operator_memory_settings()
    tier = str(memory_guard_tier or "").strip().lower() or operator_tier
    resolved_custom_gb = (
        operator_custom_gb if custom_ceiling_gb is None else float(custom_ceiling_gb)
    )
    cuda = _cuda_ceiling_breakdown(
        tier,
        custom_ceiling_gb=resolved_custom_gb,
    )
    if cuda is not None:
        return cuda

    # CUDA workers return above and should not need the macOS/server engine
    # dependency graph merely to read device memory.  In particular, a
    # lightweight worker may intentionally omit optional mlx-vlm packages.
    from omlx.process_memory_enforcer import ProcessMemoryEnforcer

    enforcer = ProcessMemoryEnforcer(
        engine_pool=None,  # type: ignore[arg-type]
        memory_guard_tier=tier,
        memory_guard_custom_ceiling_gb=resolved_custom_gb,
        prefill_memory_guard=(
            operator_enabled if guard_enabled is None else bool(guard_enabled)
        ),
    )
    breakdown = dict(enforcer._get_ceiling_breakdown())
    if not breakdown.get("hard_limit"):
        # Guard disabled: fall back to the same static ceiling the engine pool
        # uses, so admission still works rather than silently accepting.
        breakdown["hard_limit"] = int(enforcer.get_admission_ceiling())
    if not breakdown.get("hard_limit"):
        # Tier "custom" with no ceiling set: the enforcer's dynamic ceiling is
        # the number the user did not type, so the minimum is 0. Falling
        # through to "unmeasurable, load unguarded" would turn the strictest
        # setting in the product into the only one with no rank guard at all.
        breakdown["hard_limit"] = int(enforcer._get_static_ceiling())
    # A plan's tier configures the deployment; it must not admit this Mac
    # above what its own operator chose. When the caller names a different
    # tier, the operator's settings still bind from below. A disabled local
    # guard is an explicit opt-out of hard limits, not a value to clamp to,
    # and a caller passing explicit overrides is taking full control.
    if (
        custom_ceiling_gb is None
        and guard_enabled is None
        and operator_enabled
        and tier != operator_tier
    ):
        local = ceiling_breakdown(operator_tier)
        if 0 < local.get("hard_limit", 0) < breakdown.get("hard_limit", 0):
            return local
    return breakdown


def _binding_limit(breakdown: dict[str, int]) -> str:
    """Which constraint is actually stopping us, for the error message."""

    hard = breakdown.get("hard_limit", 0)
    for name in ("dynamic", "metal_cap", "static"):
        if breakdown.get(name) and breakdown[name] == hard:
            return {
                "dynamic": "memory currently available",
                "metal_cap": "the GPU allocation cap",
                "static": "installed RAM minus the reserve",
            }[name]
    return "the admission ceiling"


def _where(rank: int, node_id: str) -> str:
    return f"rank {rank}" + (f" ({node_id})" if node_id else "")


def stage_budget(
    ceiling_bytes: int,
    *,
    role: str = "",
    safety: float | None = None,
) -> int:
    """The bytes a rank may still hold once its stage has loaded.

    This is the number the plan is checked against, and it is the node role's
    own — ``NodeRole.admission_bytes``, which is never below what the planner
    was allowed to assign a node of that role. One definition, so the guard
    cannot refuse a plan the planner just built.

    ``safety`` is for a caller doing its own arithmetic; it replaces the role
    entirely rather than compounding with it.
    """

    ceiling = max(0, int(ceiling_bytes))
    if safety is not None:
        return max(0, min(ceiling, int(ceiling * safety)))

    from .node_role import role_for

    return role_for(role).admission_bytes(ceiling)


def admission_budget(
    ceiling_bytes: int,
    *,
    role: str = "",
    safety: float | None = None,
) -> int:
    """The bytes a rank may *touch*, load peak included.

    ``stage_budget`` plus what it costs to get there, and never above the Mac's
    own ceiling — past that the machine is gone whatever the role says.

    This is what ``LoadMemoryWatchdog`` aborts at, so it has to be the wider of
    the two: watch at the resident cap and every load large enough to matter is
    killed at its own peak, having read the weights twice for nothing.
    """

    ceiling = max(0, int(ceiling_bytes))
    return min(ceiling, load_peak_bytes(stage_budget(ceiling, role=role, safety=safety)))


def load_peak_bytes(planned_weight_bytes: int, kv_cache_bytes: int = 0) -> int:
    """What a stage of this size needs while it loads, not once it has loaded.

    The planner's resident figure includes this rank's KV cache, and the cache
    is empty while the weights arrive — nothing about a 20 GiB reservation for
    a context nobody has sent yet costs 26 GiB to *load*. So the multiplier is
    charged on the weights and the cache is added flat. Told the split, this
    stops over-stating the peak by tens of GiB at long context; told nothing,
    it behaves as before and over-states, which is the safe direction for a
    bound whose error the Mac pays for.
    """

    resident = max(0, int(planned_weight_bytes))
    if resident == 0:
        return 0
    cache = min(resident, max(0, int(kv_cache_bytes)))
    weights = resident - cache
    return weights + max(_LOAD_PEAK_FLOOR_BYTES, int(weights * _LOAD_PEAK_FRACTION)) + cache


def current_usage_bytes() -> int:
    """What this rank is holding right now, by the reckoning oMLX uses.

    ``phys_footprint`` is the ledger jetsam compares against and it counts
    MLX's Metal allocations; ``get_active_memory`` sees the allocator before
    the kernel ledger catches up. Neither alone is the whole truth mid-load.
    """

    active = 0
    try:
        import mlx.core as mx

        active = int(mx.get_active_memory())
    except Exception:  # pragma: no cover - MLX absent or too old
        pass
    try:
        from omlx.utils.proc_memory import get_phys_footprint

        phys = max(0, get_phys_footprint())
    except Exception:  # pragma: no cover - non-Darwin or libproc unavailable
        phys = 0
    return max(active, phys)


def _gib(value: int) -> str:
    return f"{value / 1024**3:.1f} GiB"


def _holder(role: str, safety: float | None) -> str:
    """Who the budget belongs to, for a message that explains its own numbers."""

    if safety is not None:
        return "this rank"
    from .node_role import role_for

    return f"a {role_for(role).label} node"


def _cap_clause(
    cap: int,
    ceiling: int,
    binding: str,
    *,
    role: str,
    safety: float | None,
    verb: str,
) -> str:
    """Where the cap came from, stated so the arithmetic can be checked.

    The old message said a stage of 56.1 GiB was "above the 69.9 GiB this Mac
    can admit", which is not true of either number a reader can see — the
    comparison was against a third, inflated figure it never showed. Every
    number in these sentences now takes part in the comparison being made.
    """

    if cap >= ceiling:
        return f"this Mac's whole ceiling (limited by {binding})"
    share = cap / ceiling if ceiling else 0.0
    return (
        f"{share:.0%} of this Mac's {_gib(ceiling)} ceiling "
        f"(limited by {binding}), which is what {_holder(role, safety)} {verb}"
    )


def check_rank_fits(
    planned_weight_bytes: int,
    *,
    rank: int,
    node_id: str = "",
    safety: float | None = None,
    role: str = "",
    memory_guard_tier: str = "",
    ceiling_bytes: int | None = None,
    kv_cache_bytes: int = 0,
    charge_load_peak: bool = False,
) -> int:
    """Raise ``InsufficientMemoryError`` if this rank's stage will not fit.

    The plan is charged as planned, against ``stage_budget`` — the same
    quantity, and for a headless node the same ceiling, that the single-node
    engine pool admits against. A stage this Mac would serve alone is never
    refused because a second Mac joined.

    The load peak is *watched* rather than predicted (``LoadMemoryWatchdog``).
    A plan whose peak will exceed what the watchdog allows is logged here and
    admitted, because the alternative — refusing on a 1.3x guess — is what made
    a rank stricter than the same Mac on its own. ``charge_load_peak`` turns
    that warning into a refusal for a caller that would rather lose the load
    early than late.

    Returns the ceiling used, so callers can report it. A ceiling of 0 means
    the guard is unavailable on this host; the load proceeds, matching the
    single-node "no hard limits when unguarded" behaviour rather than blocking
    a machine we cannot measure.
    """

    from omlx.exceptions import InsufficientMemoryError

    planned = max(0, int(planned_weight_bytes))
    if ceiling_bytes is not None:
        ceiling, binding = max(0, int(ceiling_bytes)), "the admission ceiling"
    else:
        try:
            breakdown = ceiling_breakdown(memory_guard_tier)
        except Exception as exc:
            # The enforcer pulls in the whole engine stack, which a worker-only
            # environment may not have installed — on one Mac here it fails on
            # a missing mlx_vlm. That must not kill the rank: the guard exists
            # to prevent a crash, not to cause one. Proceeding unguarded is the
            # documented behaviour for an unmeasurable host, but it is the
            # reason a Mac was OOMed once, so it is said loudly rather than
            # swallowed.
            logger.warning(
                "Cannot measure the admission ceiling on this host (%s); "
                "rank %s will load unguarded. Install the full oMLX "
                "environment on this Mac to restore the memory guard.",
                exc,
                rank,
            )
            return 0
        ceiling = int(breakdown.get("hard_limit", 0))
        binding = _binding_limit(breakdown)
    if ceiling <= 0:
        return 0

    stage = stage_budget(ceiling, role=role, safety=safety)
    if planned > stage:
        raise InsufficientMemoryError(
            required=planned,
            current=stage,
            message=(
                f"{_where(rank, node_id)} is planned to hold {_gib(planned)}, "
                f"above the {_gib(stage)} this Mac can admit right now — "
                f"{_cap_clause(stage, ceiling, binding, role=role, safety=safety, verb='may hold')}. "
                f"{_REMEDY}"
            ),
        )

    peak = load_peak_bytes(planned, kv_cache_bytes)
    budget = admission_budget(ceiling, role=role, safety=safety)
    if peak > budget:
        cap_clause = _cap_clause(
            budget,
            ceiling,
            binding,
            role=role,
            safety=safety,
            verb="may reach while its weights arrive",
        )
        detail = (
            f"{_where(rank, node_id)} is planned to hold {_gib(planned)}, "
            f"peaking near {_gib(peak)} as it loads, above the {_gib(budget)} "
            f"this Mac can reach — {cap_clause}"
        )
        if charge_load_peak:
            raise InsufficientMemoryError(
                required=peak,
                current=budget,
                message=f"{detail}. {_REMEDY}",
            )
        # Admitted, because single-node oMLX admits the same stage on the same
        # Mac and a rank that refuses it is a regression. Said out loud because
        # the watchdog may well stop this load, and an operator reading the
        # logs afterwards should find the prediction that saw it coming.
        logger.warning(
            "%s. It is admitted because this Mac would admit the same stage "
            "serving alone, but the load watchdog will stop this rank if it "
            "gets there. %s",
            detail,
            _REMEDY,
        )
    return ceiling


def assignment_memory_safety(assignment: Any) -> float | None:
    """The operator-selected resident share encoded in an assignment.

    Roles choose automatic defaults. Once the memory slider is moved, its
    capacity-minus-reserve value is the limit the page promises and the plan
    enforces. Express it as a fraction so the rank still scales it down when
    the live Metal/dynamic ceiling is lower than it was during planning.
    """

    if not bool(getattr(assignment, "manual_memory_limit", False)):
        return None
    capacity = max(0, int(getattr(assignment, "capacity_bytes", 0) or 0))
    if capacity <= 0:
        return 0.0
    reserve = max(0, int(getattr(assignment, "reserve_bytes", 0) or 0))
    return max(0.0, min(1.0, (capacity - reserve) / capacity))


def guard_rank_load(assignment: Any, *, rank: int, **kwargs: Any) -> int:
    """Convenience wrapper over a planner assignment.

    Reads the role off the assignment when the caller does not name one: the
    launcher emits a single argv for every host, so the plan — which is
    per-rank — is the only channel that can say "studio headless, macbook
    workstation". An explicit ``role=`` still wins, for a rank started by hand.
    """

    role = kwargs.pop("role", "") or getattr(assignment, "role", "") or ""
    safety = assignment_memory_safety(assignment)
    if safety is not None:
        kwargs.setdefault("safety", safety)
    kwargs.setdefault(
        "kv_cache_bytes", max(0, int(getattr(assignment, "kv_cache_bytes", 0) or 0))
    )
    return check_rank_fits(
        int(assignment.planned_weight_bytes),
        rank=rank,
        node_id=getattr(assignment, "node_id", ""),
        role=role,
        **kwargs,
    )


class LoadMemoryWatchdog:
    """End a rank that crosses its budget while its weights are still arriving.

    ``check_rank_fits`` runs once, against a prediction, before the first byte
    is read. Nothing watched afterwards, so a rank whose real footprint came in
    heavier than planned climbed past the ceiling with its marker still saying
    "loading" — and the Mac went down instead of the rank.

    Shaped like ``PeerWatchdog``: a thread calls ``run``, ``stop`` ends it, and
    ``run_once`` holds the whole decision so a test never needs a thread.
    Waiting is an ``Event`` rather than a sleep because the load finishes at an
    arbitrary moment and the watcher must not outlive it by an interval.
    """

    def __init__(
        self,
        budget_bytes: int,
        *,
        rank: int = 0,
        node_id: str = "",
        interval: float = _DEFAULT_SAMPLE_INTERVAL,
        usage: Callable[[], int] = current_usage_bytes,
        on_sample: Callable[[int], None] | None = None,
        interrupt: Callable[[], None] = _thread.interrupt_main,
    ) -> None:
        self._budget = max(0, int(budget_bytes))
        self._rank = int(rank)
        self._node_id = node_id
        self._interval = max(0.0, float(interval))
        self._usage = usage
        self._on_sample = on_sample
        self._interrupt = interrupt
        self._stop = threading.Event()
        self.breach: Exception | None = None

    @property
    def active(self) -> bool:
        """False on a host with no readable ceiling, where this is a no-op."""

        return self._budget > 0

    def stop(self) -> None:
        self._stop.set()

    def run_once(self) -> int:
        """Sample the rank, raising ``InsufficientMemoryError`` past the budget."""

        from omlx.exceptions import InsufficientMemoryError

        observed = max(0, int(self._usage()))
        if self._on_sample is not None:
            self._on_sample(observed)
        if observed <= self._budget:
            return observed
        raise InsufficientMemoryError(
            required=observed,
            current=self._budget,
            message=(
                f"{_where(self._rank, self._node_id)} reached "
                f"{observed / 1024**3:.1f} GiB while loading, above the "
                f"{self._budget / 1024**3:.1f} GiB it was admitted at — "
                f"stopping this rank before macOS stops the Mac. {_REMEDY}"
            ),
        )

    def run(self) -> None:
        """Sample until the load finishes or the rank has to be given up."""

        from omlx.exceptions import InsufficientMemoryError

        while not self._stop.wait(self._interval):
            try:
                self.run_once()
            except InsufficientMemoryError as exc:
                self.breach = exc
                self._stop.set()
                # Recorded before the interrupt lands, so whoever catches the
                # interrupt can tell why it was asked for.
                self._interrupt()
                return
            except Exception as exc:  # pragma: no cover - defensive
                # A sample that cannot be read must not kill a rank that
                # already passed admission: this guard exists to prevent a
                # crash, not to become one.
                logger.debug("Could not sample rank memory while loading: %s", exc)


@contextmanager
def watch_rank_load(
    budget_bytes: int,
    *,
    rank: int = 0,
    node_id: str = "",
    interval: float = _DEFAULT_SAMPLE_INTERVAL,
    usage: Callable[[], int] = current_usage_bytes,
    on_sample: Callable[[int], None] | None = None,
    interrupt: Callable[[], None] = _thread.interrupt_main,
) -> Iterator[LoadMemoryWatchdog]:
    """Watch memory for the duration of a load, and abort a rank that overruns.

    The load runs in the caller's thread deep inside MLX, so the watcher cannot
    raise there directly: it interrupts the loading thread and this re-raises
    the recorded breach on the way out, which is what the rank fails with.
    """

    watchdog = LoadMemoryWatchdog(
        budget_bytes,
        rank=rank,
        node_id=node_id,
        interval=interval,
        usage=usage,
        on_sample=on_sample,
        interrupt=interrupt,
    )
    if not watchdog.active:
        # Matches admission: a Mac whose ceiling cannot be measured loads
        # unguarded rather than being blocked by a limit nobody knows.
        yield watchdog
        return
    thread = threading.Thread(
        target=watchdog.run, name="omlx-cluster-load-watchdog", daemon=True
    )
    thread.start()
    try:
        yield watchdog
    finally:
        watchdog.stop()
        try:
            thread.join(timeout=_JOIN_TIMEOUT)
        except KeyboardInterrupt:
            # The interrupt this watchdog asked for, arriving after the load
            # had already returned. The recorded breach is the verdict.
            if watchdog.breach is None:
                raise
        if watchdog.breach is not None:
            raise watchdog.breach
