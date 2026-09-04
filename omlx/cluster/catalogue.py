# SPDX-License-Identifier: Apache-2.0
"""Which of these models will actually run on this cluster, and how far.

The question anyone asks first — "can my two Macs run this?" — had no answer
short of trying it and watching a rank die. This answers it before anything is
staged or launched.

Every verdict comes from ``plan_hybrid``, the same planner activation uses. That
is the whole point: a second, simpler estimate would be a promise the cluster
does not keep. If planning raises, the model does not fit, and the planner's own
message says which node ran out and by how much.

Two things are worth more than a yes/no:

**How the model fits.** A model that needs both Macs pipelined is a different
proposition from one that fits on either alone — the first cannot survive a
peer going to sleep.

**How much context it fits.** Weights are the floor, not the cost. KV cache
grows with the prompt, so "fits" without a context length is close to
meaningless: a model can load happily and then fail on the first long request.
The largest context that still plans is found by bisection over the planner, so
the number is one the cluster can honour rather than an estimate of one.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .planner import (
    ModelLayout,
    NodeBudget,
    PlanningError,
    inspect_safetensors_layout,
    plan_hybrid,
)

# Contexts worth reporting. Bisection lands on one of these rather than an
# arbitrary token count, because "fits at 32768" is actionable and "fits at
# 41297" only looks precise.
_CONTEXT_LADDER = (
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1_048_576,
)

_MIN_CONTEXT_TOKENS = _CONTEXT_LADDER[0]


@dataclass(frozen=True)
class ModelFit:
    """Whether one model runs on this cluster, how, and how far."""

    model_id: str
    weight_bytes: int
    fits: bool
    reason: str
    tensor_parallel_size: int = 1
    pipeline_stages: int = 1
    nodes_required: int = 0
    max_context_tokens: int = 0
    declared_context_tokens: int | None = None
    splittable: bool = True
    headroom_bytes: int = 0
    model_path: str = ""
    warnings: tuple[str, ...] = field(default_factory=tuple)
    # Machine-readable reason for a negative verdict. The dashboard must not
    # turn every refusal into "Too large": an unreadable model, an
    # unsplittable architecture, and a genuine memory shortfall need different
    # fixes.
    failure_kind: str = ""
    # The most achievable topology is more useful than whichever one happened
    # to be tried last. These keep memory refusals actionable in the GUI.
    shortfall_bytes: int = 0
    closest_strategy: str = ""
    closest_nodes_required: int = 0
    # An unsplittable model may still run perfectly well on a larger peer by
    # itself. That is useful information even though the current multi-rank
    # cluster cannot combine memory for it.
    standalone_node_id: str = ""
    standalone_max_context_tokens: int = 0
    # What the architecture can do, independent of the winning topology. The
    # fit above is the FEWEST-node plan; a model that fits one Mac reports
    # tensor_parallel_size=1 even on a 2-node cluster. The dashboard needs
    # these to know a multi-node deploy of a pipeline-incapable model can
    # only be tensor. Both default False (fail closed, like the planner's
    # _supports_pipeline for an unknown model_type) — a constructor that
    # skips them must not advertise capabilities nobody verified.
    supports_pipeline: bool = False
    supports_tensor_parallel: bool = False

    @property
    def strategy(self) -> str:
        if not self.fits:
            return ""
        if self.tensor_parallel_size > 1 and self.pipeline_stages > 1:
            return "hybrid"
        if self.tensor_parallel_size > 1:
            return "tensor"
        if self.pipeline_stages > 1:
            return "pipeline"
        return "single node"

    @property
    def context_is_limited(self) -> bool:
        """Does memory, not the model, set the usable context?"""

        declared = self.declared_context_tokens
        return bool(
            self.fits and declared and self.max_context_tokens < declared
        )

    def describe(self) -> str:
        """One line a person can act on."""

        if not self.fits:
            return f"{self.model_id}: will not fit — {self.reason}"
        where = (
            "on one node"
            if self.nodes_required == 1
            else f"across {self.nodes_required} nodes ({self.strategy})"
        )
        if not self.max_context_tokens:
            return f"{self.model_id}: fits {where}; context length unknown"
        line = (
            f"{self.model_id}: fits {where}, "
            f"up to {self.max_context_tokens:,} tokens of context"
        )
        if self.context_is_limited:
            line += f" (model supports {self.declared_context_tokens:,})"
        return line

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_path": self.model_path,
            "weight_bytes": self.weight_bytes,
            "fits": self.fits,
            "reason": self.reason,
            "strategy": self.strategy,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_stages": self.pipeline_stages,
            "nodes_required": self.nodes_required,
            "max_context_tokens": self.max_context_tokens,
            "declared_context_tokens": self.declared_context_tokens,
            "context_is_limited": self.context_is_limited,
            "splittable": self.splittable,
            "headroom_bytes": self.headroom_bytes,
            "summary": self.describe(),
            "warnings": list(self.warnings),
            "failure_kind": self.failure_kind,
            "shortfall_bytes": self.shortfall_bytes,
            "closest_strategy": self.closest_strategy,
            "closest_nodes_required": self.closest_nodes_required,
            "standalone_node_id": self.standalone_node_id,
            "standalone_max_context_tokens": self.standalone_max_context_tokens,
            "supports_pipeline": self.supports_pipeline,
            "supports_tensor_parallel": self.supports_tensor_parallel,
        }


def _splits(
    node_count: int,
    *,
    tensor_parallel_ok: bool,
    pipeline_ok: bool = True,
) -> tuple[tuple[int, int], ...]:
    """Every (tensor_parallel_size, pipeline_stages) this cluster can form.

    Ordered fewest-nodes-first, then pipeline before tensor parallelism at
    equal width: pipeline loads faster, uses less memory, and survives a
    slow link, so it is the better answer when both fit.
    """

    options: list[tuple[int, int]] = []
    for used in range(1, node_count + 1):
        for tp in range(1, used + 1):
            if used % tp:
                continue
            if tp > 1 and not tensor_parallel_ok:
                continue
            if used // tp > 1 and not pipeline_ok:
                # mlx-lm cannot split this architecture into stages, however
                # well it fits. Offering a multi-stage plan would stage the
                # weights and then fail at load.
                continue
            options.append((tp, used // tp))
    return tuple(sorted(options, key=lambda item: (item[0] * item[1], item[0])))


_SHORTFALL_RE = re.compile(r"\bat least (\d+) additional bytes required\b")


def _failed_strategy(tp_size: int, stages: int) -> str:
    if tp_size > 1 and stages > 1:
        return "hybrid"
    if tp_size > 1:
        return "tensor"
    if stages > 1:
        return "pipeline"
    return "single node"


def _shortfall_bytes(reason: str) -> int:
    match = _SHORTFALL_RE.search(reason)
    return int(match.group(1)) if match else 0


def _closest_failure(
    failures: Sequence[tuple[int, int, int, str]],
) -> tuple[str, int, str, int]:
    """Return the most achievable failed topology and a human explanation."""

    if not failures:
        return "no node budgets were supplied", 0, "", 0

    def score(item: tuple[int, int, int, str]) -> tuple[int, int, int, int]:
        tp_size, _stages, used, reason = item
        shortfall = _shortfall_bytes(reason)
        if shortfall:
            # The smallest missing amount is the useful answer. Pipeline wins
            # a tie because it is the preferred successful topology too.
            return (0, shortfall, tp_size, -used)
        return (1, 0, tp_size, -used)

    tp_size, stages, used, raw_reason = min(failures, key=score)
    shortfall = _shortfall_bytes(raw_reason)
    strategy = _failed_strategy(tp_size, stages)
    if not shortfall:
        return raw_reason, 0, strategy, used

    amount = shortfall / 1024**3
    topology = "one Mac" if used == 1 else f"a {used}-Mac {strategy} plan"
    reason = (
        f"Needs at least {amount:.1f} GiB more usable model memory for "
        f"{topology} at the minimum {_MIN_CONTEXT_TOKENS:,}-token context."
    )
    return reason, shortfall, strategy, used


def _plan_or_none(
    layout: ModelLayout,
    nodes: Sequence[NodeBudget],
    *,
    tensor_parallel_size: int,
    context_tokens: int,
    workload_profile: str = "balanced",
) -> tuple[Any | None, str]:
    """Plan, or the planner's own reason it could not."""

    try:
        plan = plan_hybrid(
            layout,
            list(nodes),
            tensor_parallel_size=tensor_parallel_size,
            workload_profile=workload_profile,  # type: ignore[arg-type]
            context_tokens=context_tokens,
        )
    except (PlanningError, ValueError) as exc:
        return None, str(exc)
    return plan, ""


def largest_context_that_fits(
    layout: ModelLayout,
    nodes: Sequence[NodeBudget],
    *,
    tensor_parallel_size: int = 1,
    ceiling_tokens: int | None = None,
    workload_profile: str = "balanced",
) -> int:
    """The biggest context on the ladder that still plans. 0 if none do.

    Bisection over the real planner rather than dividing spare bytes by a
    per-token estimate — the planner already knows about MLA caches, uneven
    stages and per-node reserves, and would disagree with any shortcut here.
    """

    if not layout.kv_bytes_per_token_per_layer:
        # Nothing is known about this model's KV growth — a synthetic layout
        # built from a download size, typically. Bisecting would return the
        # top of the ladder for every model, which is not an answer, it is a
        # promise the cluster has no basis for. Say we do not know.
        return 0

    ladder = [
        tokens
        for tokens in _CONTEXT_LADDER
        if ceiling_tokens is None or tokens <= ceiling_tokens
    ]
    # The model's own limit is the ideal automatic choice when memory can
    # serve it, even if it is not a power-of-two preset. Lower manual choices
    # still come from the clean ladder.
    if ceiling_tokens is not None and ceiling_tokens > 0:
        ladder.append(min(ceiling_tokens, _CONTEXT_LADDER[-1]))
        ladder = sorted(set(ladder))
    if not ladder:
        ladder = [_MIN_CONTEXT_TOKENS]

    def plans_at(tokens: int) -> bool:
        plan, _ = _plan_or_none(
            layout,
            nodes,
            tensor_parallel_size=tensor_parallel_size,
            context_tokens=tokens,
            workload_profile=workload_profile,
        )
        return plan is not None

    if not plans_at(ladder[0]):
        return 0

    low, high = 0, len(ladder) - 1
    while low < high:
        middle = (low + high + 1) // 2
        if plans_at(ladder[middle]):
            low = middle
        else:
            high = middle - 1
    return ladder[low]


def assess_model(
    layout: ModelLayout,
    nodes: Sequence[NodeBudget],
    *,
    model_id: str = "",
    declared_context_tokens: int | None = None,
    tensor_parallel_ok: bool | None = None,
    pipeline_ok: bool | None = None,
    workload_profile: str = "balanced",
) -> ModelFit:
    """Whether this model runs here, using the fewest nodes that work.

    ``tensor_parallel_ok`` defaults to what the layout already determined, so
    a model mlx-lm cannot shard is never offered a tensor-parallel plan.
    """

    if tensor_parallel_ok is None:
        tensor_parallel_ok = bool(layout.supports_tensor_parallel)
    if pipeline_ok is None:
        pipeline_ok = bool(layout.supports_pipeline)
    weight_bytes = layout.fixed_weight_bytes + sum(layout.layer_weight_bytes)
    name = model_id or Path(layout.source).name
    nodes = list(nodes)
    last_reason = "no node budgets were supplied"
    failures: list[tuple[int, int, int, str]] = []

    splits = _splits(
        len(nodes),
        tensor_parallel_ok=tensor_parallel_ok,
        pipeline_ok=pipeline_ok,
    )
    for tp_size, stages in splits:
        used = tp_size * stages
        # Nodes are rank-ordered; a narrower split uses the first `used` of
        # them, which is where the fast links were placed. Ranks are positional
        # to the planner and must be contiguous from zero, so a subset — or a
        # caller's single node that happened to be rank 1 — is renumbered.
        subset = [
            replace(node, rank=index) for index, node in enumerate(nodes[:used])
        ]
        plan, reason = _plan_or_none(
            layout,
            subset,
            tensor_parallel_size=tp_size,
            context_tokens=_MIN_CONTEXT_TOKENS,
            workload_profile=workload_profile,
        )
        if plan is None:
            last_reason = reason
            failures.append((tp_size, stages, used, reason))
            continue

        context = largest_context_that_fits(
            layout,
            subset,
            tensor_parallel_size=tp_size,
            ceiling_tokens=declared_context_tokens,
            workload_profile=workload_profile,
        )
        warnings: list[str] = []
        if not context and layout.kv_bytes_per_token_per_layer:
            # Weights fit at the smallest context but nothing above it.
            context = _MIN_CONTEXT_TOKENS
        if not layout.kv_bytes_per_token_per_layer:
            warnings.append(
                "Context length is unknown: this model was planned from its "
                "size, not its config. Download it for an exact answer."
            )
        if used > 1:
            warnings.append(
                f"Needs {used} nodes; the model cannot run if one goes away."
            )
        headroom = plan.cluster_capacity_bytes - plan.cluster_resident_weight_bytes

        return ModelFit(
            model_id=name,
            weight_bytes=weight_bytes,
            fits=True,
            reason=f"planned across {used} node(s)",
            tensor_parallel_size=tp_size,
            pipeline_stages=stages,
            nodes_required=used,
            max_context_tokens=context,
            declared_context_tokens=declared_context_tokens,
            headroom_bytes=max(0, headroom),
            model_path=layout.source,
            warnings=tuple(warnings),
            supports_pipeline=bool(pipeline_ok),
            supports_tensor_parallel=bool(tensor_parallel_ok),
        )

    (
        last_reason,
        shortfall_bytes,
        closest_strategy,
        closest_nodes_required,
    ) = _closest_failure(failures)
    splittable = bool(tensor_parallel_ok or pipeline_ok)
    failure_kind = "memory"
    standalone_node_id = ""
    standalone_context = 0
    if not splittable and len(nodes) > 1:
        shortfall_bytes = 0
        closest_strategy = ""
        closest_nodes_required = 0
        # The rank-zero Mac may be too small while a larger peer can still run
        # the model alone. Probe every other Mac as a standalone rank before
        # saying the model does not fit. This is deliberately an alternative,
        # not a successful cluster plan: a one-rank launch on a peer needs a
        # different coordinator topology.
        for node in nodes[1:]:
            standalone = replace(node, rank=0)
            plan, _ = _plan_or_none(
                layout,
                [standalone],
                tensor_parallel_size=1,
                context_tokens=_MIN_CONTEXT_TOKENS,
                workload_profile=workload_profile,
            )
            if plan is None:
                continue
            standalone_node_id = node.node_id
            standalone_context = largest_context_that_fits(
                layout,
                [standalone],
                ceiling_tokens=declared_context_tokens,
                workload_profile=workload_profile,
            )
            if not standalone_context and layout.kv_bytes_per_token_per_layer:
                standalone_context = _MIN_CONTEXT_TOKENS
            break

        if standalone_node_id:
            failure_kind = "single_node_only"
            context = (
                f" at up to {standalone_context:,} tokens"
                if standalone_context
                else ""
            )
            last_reason = (
                "This model cannot combine Macs because mlx-lm supports neither "
                "pipelining nor tensor parallelism for this architecture. "
                f"It does fit on {standalone_node_id} by itself{context}."
            )
        else:
            failure_kind = "cannot_split"
            # It failed on memory for every individual node, and adding Macs
            # cannot help because the architecture has no way to be split.
            last_reason = (
                "This model cannot combine Macs because mlx-lm supports neither "
                "pipelining nor tensor parallelism for this architecture, and it "
                "does not fit on any one Mac."
            )
    return ModelFit(
        model_id=name,
        weight_bytes=weight_bytes,
        fits=False,
        reason=last_reason,
        declared_context_tokens=declared_context_tokens,
        splittable=splittable,
        model_path=layout.source,
        failure_kind=failure_kind,
        shortfall_bytes=shortfall_bytes,
        closest_strategy=closest_strategy,
        closest_nodes_required=closest_nodes_required,
        standalone_node_id=standalone_node_id,
        standalone_max_context_tokens=standalone_context,
        supports_pipeline=bool(pipeline_ok),
        supports_tensor_parallel=bool(tensor_parallel_ok),
    )


def assess_model_path(
    model_path: str | Path,
    nodes: Sequence[NodeBudget],
    *,
    workload_profile: str = "balanced",
) -> ModelFit:
    """Assess a model directory, reading its real layout and config."""

    from omlx.model_discovery import _read_model_context_length

    root = Path(model_path)
    try:
        layout = inspect_safetensors_layout(root)
    except (PlanningError, OSError, ValueError) as exc:
        return ModelFit(
            model_id=root.name,
            weight_bytes=0,
            fits=False,
            reason=f"could not read the model: {exc}",
            model_path=str(root),
            failure_kind="model_unreadable",
        )

    # MLX conversions often omit ``max_position_embeddings``; oMLX already
    # knows where else to look (nested text configs, tokenizer_config, and the
    # int(1e30) "no cap" sentinel), so use that rather than a second guess.
    declared = _read_model_context_length(root)
    return assess_model(
        layout,
        nodes,
        model_id=root.name,
        declared_context_tokens=declared,
        workload_profile=workload_profile,
    )


def catalogue_for_cluster(
    model_paths: Sequence[str | Path],
    nodes: Sequence[NodeBudget],
    *,
    workload_profile: str = "balanced",
) -> tuple[ModelFit, ...]:
    """Assess every model, largest first among those that fit.

    Ordering is deliberate: the most capable model this cluster can actually
    run is the thing the user came to find out, so it goes at the top. Models
    that do not fit follow, smallest first — those are the near misses.
    """

    fits = [
        assess_model_path(path, nodes, workload_profile=workload_profile)
        for path in model_paths
    ]
    runnable = sorted(
        (fit for fit in fits if fit.fits),
        key=lambda fit: -fit.weight_bytes,
    )
    rejected = sorted(
        (fit for fit in fits if not fit.fits),
        key=lambda fit: fit.weight_bytes,
    )
    return tuple(runnable) + tuple(rejected)
