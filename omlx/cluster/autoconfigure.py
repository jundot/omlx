# SPDX-License-Identifier: Apache-2.0
"""Turn a set of paired Macs into a ready-to-activate cluster proposal.

This is the brain behind one-click activation. Given peers, their memory budgets
and a model, it picks the parallelism split and the collective backend, and
explains why — so the dashboard can show a sentence rather than asking the user
what tensor parallelism is.

Everything here is a pure function over plain data. Transport detection and SSH
probing happen in their own modules and are passed in, which keeps this testable
without a second machine.
"""

from __future__ import annotations

import ast
import json
import re
import shlex
import subprocess
import sys
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache, lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any

from .link_bandwidth import bandwidth_between, bandwidth_graph, slowest_link_in
from .planner import ModelLayout, NodeBudget, PlanningError, ShardPlan, plan_hybrid
from .transport import shared_link_addresses

# Tensor parallelism is the chatty axis: an all-reduce per layer per token. It is
# only worth spanning nodes that share a fast link.
_FAST_TRANSPORTS = {"rdma", "thunderbolt"}


@dataclass(frozen=True)
class ParallelismChoice:
    """A chosen split, the plan it produces, and why it was chosen."""

    tensor_parallel_size: int
    pipeline_stages: int
    plan: ShardPlan
    reason: str
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _divides_heads(model: ModelLayout, tensor_parallel_size: int) -> bool:
    """Whether this architecture can be split N ways at all.

    ``shard()`` divides both head counts by N, and most mlx-lm architectures do
    not implement ``shard()`` in the first place. A model layout built without a
    config.json reports ``supports_tensor_parallel=False``, which correctly
    yields tp=1 rather than a guess.
    """

    if tensor_parallel_size == 1:
        return True
    if not model.supports_tensor_parallel:
        return False
    return all(
        divisor > 0 and divisor % tensor_parallel_size == 0
        for divisor in model.tensor_parallel_divisors
    )


def candidate_tensor_parallel_sizes(
    model: ModelLayout,
    node_count: int,
) -> tuple[int, ...]:
    """Runtime-supported TP degrees, largest first.

    The pinned MLX model ``shard()`` implementations operate on a complete
    model. They cannot safely shard a partial pipeline stage, so the product
    offers either all nodes tensor-parallel (one stage) or pipeline-only. The
    pure planner can model hybrid layouts, but exposing one here would produce
    a plan the worker deliberately refuses before launch.
    """

    candidates = []
    if node_count > 1 and _divides_heads(model, node_count):
        candidates.append(node_count)
    candidates.append(1)
    return tuple(candidates)


def transports_are_fast_enough(
    transports: Sequence[Any],
    tensor_parallel_size: int,
) -> bool:
    """Whether every detected link is fast enough to carry TP traffic.

    With no transport information we do not assume the worst — the caller may
    simply not have probed yet — but we also do not claim TP is a good idea, so
    the reason string says so.
    """

    if tensor_parallel_size == 1 or not transports:
        return True
    return all(
        getattr(transport, "kind", "unknown") in _FAST_TRANSPORTS
        for transport in transports
    )


# What each strategy is for, in the user's terms. Surfaced in the UI so the
# choice can be made on intent ("make this faster" / "run something bigger")
# rather than on knowing what an all-reduce is.
STRATEGIES = {
    "auto": {
        "label": "Automatic",
        "summary": "Pick the best split for this model and link",
        "detail": (
            "Uses tensor parallelism when the model can be split evenly and the "
            "link is fast enough, otherwise falls back to pipeline stages."
        ),
    },
    "tensor": {
        "label": "Tensor — faster responses",
        "summary": "Both Macs work on every token",
        "detail": (
            "Splits every layer's weights across your Macs, so both compute each "
            "token together. Measured 1.78x on a 27B across two Macs over "
            "Thunderbolt RDMA. Needs a fast link and a model whose attention "
            "heads divide evenly by the number of Macs."
        ),
    },
    "pipeline": {
        "label": "Pipeline — bigger models",
        "summary": "Each Mac holds different layers",
        "detail": (
            "Gives each Mac a slice of the layers, so a model too large for one "
            "Mac fits across several. Works on any link and any model, loads "
            "faster and uses less memory, but a single response is no quicker "
            "than one Mac."
        ),
    },
}


def choose_parallelism(
    model: ModelLayout,
    nodes: Sequence[NodeBudget],
    *,
    transports: Sequence[Any] = (),
    prefer: str = "speed",
    strategy: str = "auto",
    measurements: Mapping[int, Any] | None = None,
    workload_profile: str = "balanced",
    context_tokens: int = 8192,
) -> ParallelismChoice:
    """Pick the tensor-parallel degree and produce the plan it implies.

    ``prefer="speed"`` takes the largest workable TP degree: splitting each
    layer across more nodes cuts per-token compute, which is what a user who
    just connected two Macs is usually after. ``prefer="capacity"`` takes the
    smallest, spending the extra nodes on pipeline stages so the largest
    possible model fits.

    Raises ``PlanningError`` if no split fits, with the reason from the last
    attempt rather than a generic failure.
    """

    if len(nodes) < 1:
        raise PlanningError("at least one node is required")
    if prefer not in {"speed", "capacity"}:
        raise ValueError("prefer must be 'speed' or 'capacity'")
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {sorted(STRATEGIES)}")

    candidates = candidate_tensor_parallel_sizes(model, len(nodes))
    if strategy == "pipeline":
        # Explicit request for layer-wise splitting only.
        if (
            len(nodes) > 1
            and model.source != "synthetic"
            and not model.supports_pipeline
        ):
            raise PlanningError(
                "pipeline parallelism is not possible for this model: "
                "the architecture does not implement the MLX-LM pipeline "
                "forward path"
            )
        candidates = (1,)
    elif strategy == "tensor":
        # Explicit request for tensor parallelism: refuse rather than silently
        # falling back, so the user learns *why* it is not possible.
        candidates = tuple(size for size in candidates if size > 1)
        if not candidates:
            raise PlanningError(
                f"tensor parallelism is not possible for this model on "
                f"{len(nodes)} nodes: "
                + (
                    "the architecture does not implement sharding in MLX-LM"
                    if not model.supports_tensor_parallel
                    else f"no split divides {len(nodes)} nodes across every "
                    "architecture-specific attention/Mamba head group "
                    f"{model.tensor_parallel_divisors}"
                )
            )
    elif prefer == "capacity":
        candidates = tuple(reversed(candidates))

    if (
        strategy == "auto"
        and len(nodes) > 1
        and model.source != "synthetic"
        and not model.supports_pipeline
    ):
        # A one-Mac fallback is not a cluster strategy. Keep only genuine
        # tensor-parallel choices so Automatic cannot return a signed plan
        # that the worker will reject after staging.
        candidates = tuple(size for size in candidates if size > 1)
        if not candidates:
            raise PlanningError(
                "this model cannot use more than one Mac: its architecture "
                "implements neither MLX-LM pipeline nor a compatible "
                "tensor-parallel split"
            )

    measured_choice = False
    if strategy == "auto" and len(candidates) > 1 and measurements:
        available = {
            size: measurements[size]
            for size in candidates
            if size in measurements
        }
        # A single measurement is not a comparison. Require every candidate
        # so an old pipeline run cannot beat an unmeasured tensor path (or vice
        # versa) merely by existing.
        if len(available) == len(candidates):
            response_tokens = {
                "interactive": 64,
                "balanced": 256,
                "throughput": 1024,
            }.get(workload_profile, 256)

            def measured_seconds(size: int) -> float:
                outcome = available[size]
                ttft = float(outcome.time_to_first_token_seconds)
                decode = float(outcome.decode_tokens_per_second)
                # TTFT already includes the first generated token.
                return ttft + max(0, response_tokens - 1) / decode

            candidates = tuple(sorted(candidates, key=measured_seconds))
            measured_choice = True

    warnings: list[str] = []
    last_error: PlanningError | None = None

    for tensor_parallel_size in candidates:
        fast_enough = transports_are_fast_enough(
            transports,
            tensor_parallel_size,
        )
        measured_on_this_path = (
            measured_choice and tensor_parallel_size in (measurements or {})
        )
        if (
            strategy == "auto"
            and not fast_enough
            and not measured_on_this_path
        ):
            continue
        try:
            plan = plan_hybrid(
                model,
                nodes,
                tensor_parallel_size=tensor_parallel_size,
                workload_profile=workload_profile,
                context_tokens=context_tokens,
            )
        except PlanningError as exc:
            last_error = exc
            continue

        stages = len(nodes) // tensor_parallel_size
        if tensor_parallel_size == 1:
            reason = (
                f"{len(nodes)} nodes as {stages} pipeline stages. Tensor "
                f"parallelism was not used"
            )
            if not model.supports_tensor_parallel:
                reason += (
                    " because this model architecture does not implement "
                    "tensor-parallel sharding in MLX-LM"
                )
            elif any(
                value % max(len(nodes), 1)
                for value in model.tensor_parallel_divisors
            ):
                reason += (
                    " because an architecture-specific head group does not "
                    "divide evenly"
                )
            elif transports and not transports_are_fast_enough(transports, len(nodes)):
                reason += " because the link between nodes is too slow for it"
            else:
                reason += " because no larger split fit in memory"
        else:
            reason = (
                f"{len(nodes)} nodes as {stages} pipeline "
                f"stage{'s' if stages != 1 else ''} of {tensor_parallel_size}-way "
                f"tensor parallelism — each layer's weights are split across "
                f"{tensor_parallel_size} Macs"
            )
        if measured_choice:
            outcome = measurements[tensor_parallel_size]
            reason += (
                " — selected from measured end-to-end results "
                f"({outcome.prompt_tokens_per_second:.1f} prompt tok/s, "
                f"{outcome.decode_tokens_per_second:.1f} decode tok/s, "
                f"{outcome.time_to_first_token_seconds:.2f}s to first token)"
            )
        if tensor_parallel_size > 1 and not transports:
            warnings.append(
                "Transport was not probed, so tensor parallelism was chosen "
                "without knowing the link speed. Run transport detection for a "
                "better choice."
            )
        elif tensor_parallel_size > 1 and not fast_enough:
            warnings.append(
                "Tensor parallelism was explicitly selected across a slow "
                "link; every layer's all-reduce may make it slower than one Mac."
            )
        return ParallelismChoice(
            tensor_parallel_size=tensor_parallel_size,
            pipeline_stages=stages,
            plan=plan,
            reason=reason,
            warnings=tuple(warnings),
        )

    if last_error is not None:
        raise PlanningError(
            f"no workable split for {len(nodes)} nodes: {last_error}"
        )
    raise PlanningError(
        f"no tensor-parallel degree divides {len(nodes)} nodes across "
        f"architecture dimensions {model.tensor_parallel_divisors}"
    )


def choose_backend(transports: Sequence[Any]) -> tuple[str, str]:
    """Pick the collective backend from detected transports, with a reason.

    Mirrors what ``mlx.distributed_config`` does: RDMA over Thunderbolt is the
    fast path, plain Thunderbolt rings use jaccl-ring, and everything else falls
    back to the TCP ring.
    """

    kinds = {getattr(transport, "kind", "unknown") for transport in transports}
    if "rdma" in kinds:
        return "jaccl", "RDMA over Thunderbolt detected"
    if "thunderbolt" in kinds:
        return "jaccl-ring", "Thunderbolt link detected, without RDMA"
    if not transports:
        return "ring", "no transport detected; using the TCP ring backend"
    return "ring", "Ethernet link; using the TCP ring backend"


@dataclass(frozen=True)
class RDMAMatrix:
    """The per-peer RDMA paths a jaccl deployment needs, or why there are none."""

    rows: tuple[tuple[str | None, ...], ...] = ()
    reason: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "rows": [list(row) for row in self.rows],
            "reason": self.reason,
        }


def build_rdma_matrix(interfaces: Sequence[Any]) -> RDMAMatrix:
    """Assemble the RDMA connectivity matrix ``ClusterDeployment`` demands.

    ``choose_backend`` answers "jaccl" the moment an RDMA device is detected,
    but ``ClusterDeployment`` rejects a jaccl backend whose hosts carry no
    ``rdma`` entry per peer — so the two disagree and activation dies in a
    constructor with nothing the user can act on. This derives the matrix, and
    when it cannot, names what is missing so the caller can fall back to ring
    out loud instead of by accident.

    ``interfaces`` is one ``HostInterfaces`` per host in rank order, not the
    detected ``TransportInfo`` tuple: transport detection knows a Thunderbolt
    link exists but not what either end of it is called — its ``interface``
    field holds a peer index and its ``rdma_device`` a placeholder — and jaccl
    needs the device name each rank uses to reach each peer. Reading it off the
    hosts is also the only form that survives macOS renumbering a port.

    Entry ``[i][j]`` is the RDMA device rank ``i`` uses to reach rank ``j``,
    with a null diagonal, matching MLX's own hostfile.
    """

    if len(interfaces) < 2:
        return RDMAMatrix(reason="a distributed cluster needs at least two hosts")

    without_devices = [host.host for host in interfaces if not host.rdma_interfaces]
    if without_devices:
        return RDMAMatrix(
            reason=(
                f"{', '.join(without_devices)} reports no RDMA device. RDMA is "
                f"off by default and can only be enabled from macOS Recovery, so "
                f"this cluster cannot use jaccl yet."
            )
        )

    rows: list[tuple[str | None, ...]] = []
    for index, source in enumerate(interfaces):
        row: list[str | None] = []
        for peer_index, peer in enumerate(interfaces):
            if index == peer_index:
                row.append(None)
                continue
            link = shared_link_addresses(source, peer)
            endpoint = link.source
            if endpoint is None:
                return RDMAMatrix(reason=link.reason)
            if link.kind != "rdma":
                return RDMAMatrix(
                    reason=(
                        f"{source.host} reaches {peer.host} over "
                        f"{endpoint.interface}, which is not an RDMA device. "
                        f"Give the Thunderbolt ports between them an address so "
                        f"the RDMA path is the one they share."
                    )
                )
            row.append(f"rdma_{endpoint.interface}")
        rows.append(tuple(row))
    return RDMAMatrix(
        rows=tuple(rows),
        reason=f"every pair of {len(interfaces)} hosts has an RDMA path",
    )


def describe_transports(transports: Sequence[Any]) -> str:
    """One human sentence about the fabric, for the activation summary."""

    if not transports:
        return "Link speed unknown"
    speeds = [
        getattr(transport, "link_speed_gbps", None) for transport in transports
    ]
    known = [speed for speed in speeds if speed]
    versions = {
        getattr(transport, "tb_version", None)
        for transport in transports
        if getattr(transport, "tb_version", None)
    }
    label = "/".join(sorted(versions)) if versions else next(
        iter({getattr(t, "kind", "unknown") for t in transports})
    )
    if known:
        return f"{label} at {min(known)} Gb/s"
    return str(label)


@dataclass(frozen=True)
class Placement:
    """A rank ordering for the hosts, and why it was chosen."""

    hosts: tuple[str, ...]
    reason: str
    warnings: tuple[str, ...] = field(default_factory=tuple)


def order_hosts_for_topology(
    hosts: Sequence[str],
    transports: Sequence[Any],
    tensor_parallel_size: int,
    profiles: Sequence[Any] | None = None,
) -> Placement:
    """Order hosts so each tensor-parallel group sits on the fastest links.

    Ranks are laid out ``rank = stage * tp_size + tp_rank``, so consecutive
    blocks of ``tp_size`` hosts form one TP group. TP all-reduces once per layer
    per token while a pipeline boundary sends one activation tensor per stage —
    so the fast links belong *inside* the blocks, and the slow hop belongs
    between them.

    Grouping is by measured bandwidth rather than link kind: two Thunderbolt
    links can differ fourfold, and a link that merely *looks* fast is how a
    tensor-parallel group ends up straddling 0.5 GB/s. Passing ``profiles``
    (oMLX's collective probe) upgrades the estimate from the cable's label to
    what the fabric actually delivered.

    Falls back to the given order, with a warning, when the fabric cannot be
    partitioned that way.
    """

    hosts = tuple(hosts)
    if tensor_parallel_size <= 1 or len(hosts) <= tensor_parallel_size:
        return Placement(hosts, "single tensor-parallel group; no placement choice")
    if len(hosts) % tensor_parallel_size:
        return Placement(
            hosts,
            "host count is not a multiple of the tensor-parallel size",
            ("Rank placement was left unchanged.",),
        )

    graph = bandwidth_graph(transports, profiles)
    fast = {key: link for key, link in graph.items() if link.fast}
    if not fast:
        return Placement(
            hosts,
            "no link fast enough for tensor parallelism; rank order left unchanged",
            (
                "Tensor-parallel groups may span slow links. Connect a "
                "Thunderbolt cable, or run transport detection to measure the "
                "fabric.",
            ),
        )

    remaining = list(hosts)
    ordered: list[str] = []
    while remaining:
        seed = remaining.pop(0)
        block = [seed]
        # Take the fastest peer that is fast to *every* member so far, so the
        # group is mutually fast rather than merely chained. Ties break on host
        # order, keeping placement deterministic.
        while len(block) < tensor_parallel_size:
            candidates = [
                host
                for host in remaining
                if all(
                    bandwidth_between(fast, host, member) > 0 for member in block
                )
            ]
            if not candidates:
                break
            chosen = max(
                candidates,
                key=lambda host: (
                    min(bandwidth_between(fast, host, m) for m in block),
                    -remaining.index(host),
                ),
            )
            block.append(chosen)
            remaining.remove(chosen)
        if len(block) < tensor_parallel_size:
            return Placement(
                hosts,
                "the fabric does not split into fast tensor-parallel groups",
                (
                    f"Could not place {tensor_parallel_size} mutually fast-linked "
                    f"hosts in every group; rank order left unchanged and tensor "
                    f"parallelism may cross a slow link.",
                ),
            )
        ordered.extend(block)

    groups = [
        ordered[index : index + tensor_parallel_size]
        for index in range(0, len(ordered), tensor_parallel_size)
    ]
    described = []
    evidence = set()
    for group in groups:
        link = slowest_link_in(graph, group)
        if link is None:
            described.append("+".join(group))
            continue
        described.append(f"{'+'.join(group)} at {link.gigabytes_per_second:.2f} GB/s")
        evidence.add(link.source)

    basis = (
        "measured bandwidth"
        if evidence == {"measured"}
        else "link speed" if "measured" not in evidence
        else "measured bandwidth where available"
    )
    warnings: tuple[str, ...] = ()
    if "measured" not in evidence:
        warnings = (
            "Bandwidth was inferred from the link type, not measured. Run the "
            "performance probe to place ranks on what the fabric delivers.",
        )
    return Placement(
        tuple(ordered),
        f"tensor-parallel groups placed by {basis}: " + "; ".join(described),
        warnings,
    )


def tp_groups_spanning_slow_links(
    hosts: Sequence[str],
    transports: Sequence[Any],
    tensor_parallel_size: int,
    profiles: Sequence[Any] | None = None,
) -> tuple[tuple[str, ...], ...]:
    """Groups whose all-reduce would run over a link too slow to carry it.

    Used to warn at plan time: a TP group over Ethernet will work and be slow,
    which is worth saying out loud rather than silently shipping. Judged on
    bandwidth, so a Thunderbolt link that negotiated down is caught too.
    """

    if tensor_parallel_size <= 1:
        return ()
    graph = bandwidth_graph(transports, profiles)
    if not graph:
        return ()
    offenders = []
    hosts = tuple(hosts)
    for index in range(0, len(hosts), tensor_parallel_size):
        group = hosts[index : index + tensor_parallel_size]
        if len(group) < 2:
            continue
        slowest = slowest_link_in(graph, group)
        if slowest is None or not slowest.fast:
            offenders.append(group)
    return tuple(offenders)


@dataclass(frozen=True)
class PreflightIssue:
    """One reason a proposed cluster would fail on activation.

    ``remediation`` is a command the user can paste into a terminal as-is.
    Not every issue has one — a version skew is resolved by updating both
    Macs, which is a sentence rather than a command — but an issue that can
    be fixed by one command should carry it rather than describe it.
    """

    node_id: str
    kind: str  # "model_missing", "runtime_mismatch", "unreachable", "import_missing"
    detail: str
    remediation: str = ""


def preflight_issues(
    peer_statuses: dict[str, Any],
    *,
    model_path: str | None,
    local_versions: dict[str, str] | None = None,
) -> tuple[PreflightIssue, ...]:
    """Check what would break before proposing, not after launching.

    ``peer_statuses`` maps node id to the probe payload for that peer, or to
    ``None`` when it could not be reached. Catching a missing model or an MLX
    version skew here saves the user a full distributed load that ends in a
    rank-zero traceback.
    """

    issues: list[PreflightIssue] = []
    for node_id, status in peer_statuses.items():
        if not status:
            issues.append(
                PreflightIssue(node_id, "unreachable", "peer did not answer a probe")
            )
            continue

        payload = status.get("status") if isinstance(status.get("status"), dict) else status
        runtime = payload.get("runtime") or {}
        reported_runtime_mismatch = status.get("runtime_compatible") is False
        if reported_runtime_mismatch:
            issues.append(
                PreflightIssue(
                    node_id,
                    "runtime_mismatch",
                    "; ".join(status.get("runtime_mismatches") or ())
                    or "worker runtime differs from the coordinator",
                )
            )
        if local_versions and not reported_runtime_mismatch:
            for package, expected in local_versions.items():
                actual = runtime.get(package)
                if actual and expected and actual != expected:
                    issues.append(
                        PreflightIssue(
                            node_id,
                            "runtime_mismatch",
                            f"{package} is {actual} here and {expected} locally",
                        )
                    )

        node = payload.get("node") or {}
        if node.get("accelerator") == "cuda" and node.get("memory_kind") == "vram":
            issues.append(
                PreflightIssue(
                    node_id,
                    "cuda_memory_guard_unavailable",
                    "discrete CUDA VRAM cannot yet be monitored during model loading",
                )
            )

        models = payload.get("models")
        if model_path and isinstance(models, (list, tuple)) and model_path not in models:
            issues.append(
                PreflightIssue(
                    node_id,
                    "model_missing",
                    f"{model_path} was not found on this peer",
                )
            )
    return tuple(issues)


def describe_preflight(issues: Sequence[PreflightIssue]) -> str:
    """One sentence a user can act on, or a clear all-clear.

    Where an issue carries a command that fixes it, the command is appended
    verbatim: a summary that names the problem but hides the one-line fix
    sends the user back to the logs to find it.
    """

    if not issues:
        return "All peers are reachable and ready."
    by_kind: dict[str, list[str]] = {}
    for issue in issues:
        by_kind.setdefault(issue.kind, []).append(issue.node_id)
    parts = []
    labels = {
        "unreachable": "unreachable",
        "model_missing": "missing the model",
        "runtime_mismatch": "running a different MLX version",
        "import_missing": "missing a Python package this model needs",
        "cuda_memory_guard_unavailable": "using unsupported discrete CUDA memory",
    }
    for kind, nodes in by_kind.items():
        parts.append(f"{', '.join(sorted(set(nodes)))} {labels.get(kind, kind)}")
    sentence = "Cannot activate yet: " + "; ".join(parts) + "."
    commands = list(
        dict.fromkeys(issue.remediation for issue in issues if issue.remediation)
    )
    if commands:
        sentence += " Run: " + "  ".join(commands)
    return sentence


# A rank does not fail on the model it cannot fit — that is guarded. It fails on
# the import it cannot perform: a peer whose environment was installed for
# serving text dies half way through loading a VLM with "No module named
# 'mlx_vlm'", after every other rank has already paid for its weights.
#
# Which imports a rank needs is not a property of oMLX, it is a property of the
# model: config.json decides which branches of ``maybe_apply_pre_load_patches``
# run, and each patch module's own imports say what those branches need present.
# Both halves are read here rather than restated, so a model type added to the
# dispatcher is covered without anyone remembering to update this file.

_OMLX_ROOT = Path(__file__).resolve().parent.parent
_DISPATCHER = _OMLX_ROOT / "utils" / "model_loading.py"
_DISPATCH_FUNCTION = "maybe_apply_pre_load_patches"

# The dispatcher branches on the top-level model_type and on the one nested in
# ``text_config``; a model matching either takes the branch.
_MODEL_TYPE_NAMES = frozenset({"model_type", "text_model_type"})

# Exception types that make an import optional: the patch logs and carries on,
# so the module it wanted is not something a rank has to have.
_IMPORT_TOLERANT = frozenset(
    {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}
)

_LOCAL_SSH_TARGETS = frozenset({"127.0.0.1", "localhost", "::1"})


@dataclass(frozen=True)
class _Guard:
    """The conditions under which the dispatcher reaches one patch import.

    ``model_types`` is one entry per nested ``if``, each holding the
    alternatives that test allows — ``("eq", "laguna")`` or
    ``("prefix", "deepseek_v4")``. Every entry must be satisfied, any
    alternative within one will do.
    """

    model_types: tuple[frozenset[tuple[str, str]], ...] = ()
    for_vlm: bool | None = None

    def narrowed(
        self,
        alternatives: frozenset[tuple[str, str]],
        for_vlm: bool | None,
    ) -> _Guard:
        return _Guard(
            self.model_types + ((alternatives,) if alternatives else ()),
            self.for_vlm if for_vlm is None else for_vlm,
        )

    def matches(self, model_types: Collection[str], for_vlm: bool) -> bool:
        if self.for_vlm is not None and self.for_vlm is not for_vlm:
            return False
        return all(
            any(
                declared == literal
                if kind == "eq"
                else declared.startswith(literal)
                for kind, literal in alternatives
                for declared in model_types
            )
            for alternatives in self.model_types
        )


def _model_type_alternatives(
    test: ast.expr,
    aliases: Mapping[str, frozenset[str]],
) -> frozenset[tuple[str, str]]:
    """Model types one ``if`` test admits, as ("eq"|"prefix", literal) pairs.

    ``and`` is unioned with ``or`` rather than intersected. The dispatcher only
    ever ``and``s a model-type test with a type check or the VLM flag, so the
    union is exact there, and over-admitting is the safe direction: it asks a
    peer for one import too many rather than missing the one that kills it.
    """

    if isinstance(test, ast.BoolOp):
        return frozenset().union(
            *(_model_type_alternatives(value, aliases) for value in test.values)
        )
    if isinstance(test, ast.Compare) and isinstance(test.left, ast.Name):
        if test.left.id not in _MODEL_TYPE_NAMES:
            return frozenset()
        found: set[tuple[str, str]] = set()
        for operator, comparator in zip(test.ops, test.comparators):
            if isinstance(operator, ast.Eq) and isinstance(comparator, ast.Constant):
                found.add(("eq", str(comparator.value)))
            elif isinstance(operator, ast.In):
                found.update(("eq", value) for value in _literals(comparator, aliases))
        return frozenset(found)
    if (
        isinstance(test, ast.Call)
        and isinstance(test.func, ast.Attribute)
        and test.func.attr == "startswith"
        and isinstance(test.func.value, ast.Name)
        and test.func.value.id in _MODEL_TYPE_NAMES
        and test.args
        and isinstance(test.args[0], ast.Constant)
    ):
        return frozenset({("prefix", str(test.args[0].value))})
    return frozenset()


def _literals(
    node: ast.expr,
    aliases: Mapping[str, frozenset[str]],
) -> frozenset[str]:
    """String constants in a collection literal, or in a name bound to one."""

    if isinstance(node, ast.Name):
        return aliases.get(node.id, frozenset())
    if isinstance(node, (ast.Set, ast.Tuple, ast.List)):
        return frozenset(
            str(element.value)
            for element in node.elts
            if isinstance(element, ast.Constant)
        )
    return frozenset()


def _vlm_polarity(test: ast.expr) -> bool | None:
    """What an ``if`` test says about ``for_vlm``, or None if it says nothing."""

    if isinstance(test, ast.Name) and test.id == "for_vlm":
        return True
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        inner = _vlm_polarity(test.operand)
        return None if inner is None else not inner
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
        for value in test.values:
            polarity = _vlm_polarity(value)
            if polarity is not None:
                return polarity
    return None


def _walk_dispatch(
    body: Sequence[ast.stmt],
    guard: _Guard,
    aliases: dict[str, frozenset[str]],
    found: list[tuple[_Guard, str]],
) -> None:
    """Record every ``omlx.patches`` import with the guard that reaches it."""

    for node in body:
        if isinstance(node, ast.Assign):
            # The dispatcher names one set of model types before testing it.
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                literals = _literals(node.value, aliases)
                if literals:
                    aliases[node.targets[0].id] = literals
        elif isinstance(node, ast.If):
            _walk_dispatch(
                node.body,
                guard.narrowed(
                    _model_type_alternatives(node.test, aliases),
                    _vlm_polarity(node.test),
                ),
                aliases,
                found,
            )
            _walk_dispatch(node.orelse, guard, aliases, found)
        elif isinstance(node, ast.ImportFrom) and node.module:
            # ``from ..patches.step3p7 import ...`` and the absolute form both
            # name the same package; the level lives outside ``module``.
            module = node.module.removeprefix("omlx.")
            if module.startswith("patches."):
                found.append((guard, f"omlx.{module}"))
        else:
            for attribute in ("body", "orelse", "finalbody"):
                inner = getattr(node, attribute, None)
                if isinstance(inner, list):
                    _walk_dispatch(inner, guard, aliases, found)


@lru_cache(maxsize=1)
def _dispatch_rules() -> tuple[tuple[_Guard, str], ...]:
    """Read the patch dispatch out of ``maybe_apply_pre_load_patches``.

    Reading the dispatcher beats restating it. The list of model types that
    need a patch changes every time a new architecture lands, and a copy here
    would be wrong the first time someone forgot about it — which is the same
    failure mode this whole check exists to remove.
    """

    try:
        tree = ast.parse(_DISPATCHER.read_text())
    except (OSError, SyntaxError, ValueError):
        return ()
    dispatcher = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == _DISPATCH_FUNCTION
        ),
        None,
    )
    if dispatcher is None:
        return ()
    found: list[tuple[_Guard, str]] = []
    _walk_dispatch(dispatcher.body, _Guard(), {}, found)
    return tuple(found)


def _handles_import_error(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    raised = (
        handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    )
    return any(
        isinstance(node, ast.Name) and node.id in _IMPORT_TOLERANT for node in raised
    )


def _walk_imports(
    body: Sequence[ast.stmt],
    optional: bool,
    names: set[str],
) -> None:
    """Collect top-level module names an import failure would be fatal for."""

    for node in body:
        if isinstance(node, ast.Try):
            tolerated = optional or any(
                _handles_import_error(handler) for handler in node.handlers
            )
            _walk_imports(node.body, tolerated, names)
            for handler in node.handlers:
                _walk_imports(handler.body, True, names)
            _walk_imports(node.orelse, optional, names)
            _walk_imports(node.finalbody, optional, names)
        elif isinstance(node, ast.If) and _is_type_checking(node.test):
            # Never executes; a rank cannot fail on it.
            _walk_imports(node.orelse, optional, names)
        elif isinstance(node, ast.Import):
            if not optional:
                names.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if not optional and node.level == 0 and node.module:
                names.add(node.module.partition(".")[0])
        else:
            for attribute in ("body", "orelse", "finalbody"):
                inner = getattr(node, attribute, None)
                if isinstance(inner, list):
                    _walk_imports(inner, optional, names)


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


@cache
def _third_party_imports(patch_module: str) -> tuple[str, ...]:
    """Modules outside oMLX and the standard library that a patch must import.

    Scans the patch's own source, including the imports it defers into
    functions — a patch that reaches for mlx-vlm two frames down still needs
    mlx-vlm installed, and deferring the import is how the failure ends up in
    the middle of a load instead of at the top of one.
    """

    relative = patch_module.removeprefix("omlx.").replace(".", "/")
    base = _OMLX_ROOT / relative
    sources = sorted(base.rglob("*.py")) if base.is_dir() else [base.with_suffix(".py")]
    names: set[str] = set()
    for source in sources:
        try:
            _walk_imports(ast.parse(source.read_text()).body, False, names)
        except (OSError, SyntaxError, ValueError):
            continue
    return tuple(
        sorted(
            name
            for name in names
            if name != "omlx"
            and not name.startswith("_")
            and name not in sys.stdlib_module_names
        )
    )


def _canonical(name: str) -> str:
    """PEP 503 name normalisation, so mlx_vlm and mlx-vlm compare equal."""

    return re.sub(r"[-_.]+", "-", name).lower()


@lru_cache(maxsize=1)
def _declared_requirements() -> dict[str, str]:
    """The requirement strings oMLX declares, keyed by canonical distribution.

    Several are pinned to a git commit. A peer told to install the bare name
    lands on whatever the index serves today, and a rank running a different
    build of mlx-vlm fails in a new way rather than not at all — so the fix we
    hand the user is oMLX's own pin, not our guess at it.
    """

    try:
        declared = metadata.requires("omlx") or ()
    except metadata.PackageNotFoundError:
        return {}
    pins: dict[str, str] = {}
    for requirement in declared:
        # Environment markers ("; python_version >= '3.11'") are not part of
        # what a peer installs.
        specifier = requirement.partition(";")[0].strip()
        name = re.split(r"[\s<>=!~\[@]", specifier, maxsplit=1)[0]
        if name:
            pins.setdefault(_canonical(name), specifier)
    return pins


@lru_cache(maxsize=1)
def _distributions_by_module() -> dict[str, list[str]]:
    try:
        return metadata.packages_distributions()
    except Exception:
        # An unreadable metadata directory is not a reason to refuse to name
        # the package; the module name is a usable fallback.
        return {}


def _distribution_for(module: str) -> str:
    """The installable name for an importable one — PIL ships as Pillow."""

    candidates = _distributions_by_module().get(module, [])
    for candidate in candidates:
        if _canonical(candidate) == _canonical(module):
            return candidate
    return candidates[0] if candidates else module.replace("_", "-")


@dataclass(frozen=True)
class ImportRequirement:
    """One module a rank must be able to import to serve this model."""

    module: str
    distribution: str
    reason: str


def _model_config(model_path: Path) -> dict[str, Any]:
    try:
        config = json.loads((model_path / "config.json").read_text())
    except (OSError, ValueError):
        return {}
    return config if isinstance(config, dict) else {}


def _declared_model_types(config: Mapping[str, Any]) -> frozenset[str]:
    text_config = config.get("text_config")
    candidates = (
        config.get("model_type"),
        text_config.get("model_type") if isinstance(text_config, dict) else None,
    )
    return frozenset(value for value in candidates if isinstance(value, str))


def required_imports(
    model_path: str | Path,
    *,
    for_vlm: bool = False,
) -> tuple[ImportRequirement, ...]:
    """Modules a rank will import to serve the model at ``model_path``.

    Derived, not listed: ``config.json`` says which branches of the pre-load
    patch dispatch a rank takes, and each patch module's own source says what
    those branches import. A model directory that cannot be read yields the
    unconditional requirements, which a rank needs regardless.

    ``for_vlm`` defaults to False because a rank is an ``mlx_lm.server`` — that
    is the whole reason ``omlx.patches.minimax_m3_mlx_lm`` exists — so it takes
    the same branches ``inference_worker`` takes. Pass True to describe a
    single-node mlx-vlm load instead.
    """

    root = Path(model_path).expanduser()
    model_types = _declared_model_types(_model_config(root))

    sources: dict[str, set[str]] = {}
    for guard, patch_module in _dispatch_rules():
        if not guard.matches(model_types, for_vlm):
            continue
        for module in _third_party_imports(patch_module):
            sources.setdefault(module, set()).add(patch_module)

    return tuple(
        ImportRequirement(
            module=module,
            distribution=_distribution_for(module),
            reason="needed by " + ", ".join(sorted(reasons)),
        )
        for module, reasons in sorted(sources.items())
    )


# Runs on the peer under the interpreter a rank would use. ``find_spec`` rather
# than a real import: resolution is the question — "No module named 'mlx_vlm'"
# is a resolution failure — and importing mlx here would spin up Metal on a
# machine that is probably already serving.
_IMPORT_PROBE = """
import importlib.util, json, os, shutil, sys

report = {"python": sys.executable, "found": [], "missing": []}
for name in sys.argv[1].split(","):
    try:
        resolved = importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        resolved = False
    report["found" if resolved else "missing"].append(name)

pip = os.path.join(sys.prefix, "bin", "pip")
report["pip"] = pip if os.path.exists(pip) else None
uv = shutil.which("uv") or os.path.expanduser("~/.local/bin/uv")
report["uv"] = uv if os.path.exists(uv) else None
print(json.dumps(report))
"""


def _install_command(
    ssh_target: str,
    *,
    python_executable: str,
    distributions: Sequence[str],
    uv: str | None,
    pip: str | None,
) -> str:
    """One pasteable line that puts the missing packages on the peer.

    The installer is whatever the peer reported having. A worker venv built by
    uv ships no ``pip`` at all, so the obvious ``python -m pip install`` is a
    command that cannot work on the machine it is aimed at; ``ensurepip`` is
    the last resort because it needs nothing fetched before it runs.
    """

    packages = " ".join(
        shlex.quote(_declared_requirements().get(_canonical(name), name))
        # Two missing modules can ship in one distribution; installing it twice
        # is not wrong, but a command with a repeated package reads like a bug.
        for name in dict.fromkeys(distributions)
    )
    if uv:
        inner = f"{uv} pip install --python {python_executable} {packages}"
    elif pip:
        inner = f"{python_executable} -m pip install {packages}"
    else:
        inner = (
            f"{python_executable} -m ensurepip && "
            f"{python_executable} -m pip install {packages}"
        )
    return f'ssh {ssh_target} "{inner}"'


def peer_import_issues(
    hosts: Mapping[str, str],
    *,
    model_path: str | Path,
    python_executable: str = sys.executable,
    python_by_node: Mapping[str, str] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    timeout: float = 30.0,
) -> tuple[PreflightIssue, ...]:
    """Ask every peer whether it can perform the imports this model needs.

    ``hosts`` maps node id to SSH target; local targets are skipped because the
    process asking the question is the answer for them. One round trip per
    peer, before a single rank is started — the alternative is finding out from
    rank 1 after every other rank has loaded its weights.

    Runs over the same strict SSH policy the launch itself uses, so a peer this
    check trusts is a peer the launch will also trust.
    """

    from .launch import DistributedLaunchError, _run_cluster_ssh

    requirements = required_imports(model_path)
    if not requirements:
        return ()
    by_module = {requirement.module: requirement for requirement in requirements}
    issues: list[PreflightIssue] = []
    for node_id, ssh_target in hosts.items():
        if ssh_target in _LOCAL_SSH_TARGETS:
            continue
        peer_python = (python_by_node or {}).get(node_id) or python_executable
        command = shlex.join(
            [peer_python, "-c", _IMPORT_PROBE, ",".join(sorted(by_module))]
        )
        try:
            completed = _run_cluster_ssh(
                ssh_target, command, timeout=timeout, runner=runner
            )
        except (DistributedLaunchError, ValueError) as exc:
            issues.append(
                PreflightIssue(
                    node_id,
                    "unreachable",
                    str(exc),
                    remediation=f"ssh {ssh_target} true",
                )
            )
            continue
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()[:500]
            issues.append(
                PreflightIssue(
                    node_id,
                    "unreachable",
                    f"{peer_python} could not run on this peer: {detail}",
                    remediation=f'ssh {ssh_target} "{peer_python} -V"',
                )
            )
            continue
        try:
            # A login banner or an MOTD lands ahead of the payload.
            report = json.loads(completed.stdout.strip().splitlines()[-1])
            if not isinstance(report, dict):
                raise ValueError("import report is not an object")
        except (ValueError, IndexError):
            issues.append(
                PreflightIssue(
                    node_id,
                    "unreachable",
                    "peer did not return an import report",
                    remediation=f"ssh {ssh_target} true",
                )
            )
            continue

        missing = [
            by_module[module]
            for module in report.get("missing", ())
            if module in by_module
        ]
        if not missing:
            continue
        # The peer reports the interpreter it actually ran under, so the fix
        # names that one rather than the path we hoped it would be.
        peer_python = report.get("python") or peer_python
        issues.append(
            PreflightIssue(
                node_id,
                "import_missing",
                f"{peer_python} cannot import "
                + "; ".join(
                    f"{requirement.module} ({requirement.reason})"
                    for requirement in missing
                ),
                remediation=_install_command(
                    ssh_target,
                    python_executable=peer_python,
                    distributions=[requirement.distribution for requirement in missing],
                    uv=report.get("uv"),
                    pip=report.get("pip"),
                ),
            )
        )
    return tuple(issues)
