# SPDX-License-Identifier: Apache-2.0
"""Unequal-memory planning for contiguous MLX-LM pipeline stages."""

from __future__ import annotations

import hashlib
import json
import re
import struct
import subprocess
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .node_role import ROLES
from .performance import ExecutionProfileName, NodePerformanceProfile
from .staging import DEFAULT_REMOTE_PYTHON, is_local_host, run_remote_python
from .tensor_strategies import (
    native_shard_is_layer_local,
    supports_model_type,
)

_MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024
_MAX_SAFETENSORS_INDEX_BYTES = 16 * 1024 * 1024
_MAX_METADATA_FILE_BYTES = 256 * 1024 * 1024
_MAX_PIPELINE_LAYERS = 2048
_LAYER_PATTERNS = (re.compile(r"(?:^|\.)(?:layers|h|blocks|block)\.(\d+)(?:\.|$)"),)

# Context length a plan reserves KV cache for. Planning for 0 is how a model
# that fits its weights still dies on the first long prompt.
_DEFAULT_CONTEXT_TOKENS = 8192
_MEMORY_GUARD_TIERS = frozenset({"safe", "balanced", "aggressive", "custom"})


class PlanningError(ValueError):
    """Raised when a model cannot be admitted to the supplied node budgets."""


def normalize_node_role(value: Any) -> str:
    """The canonical key of a node role, or ``""`` when none was chosen.

    An unrecognised role raises rather than falling back to headless. The
    fallback is the *permissive* direction — headless admits at 0.90 of the
    Mac's ceiling, a workstation at 0.65 — so a typo that quietly became
    "headless" would read as a working control while filling the machine
    someone is typing on. ``role_for()`` in ``node_role`` deliberately keeps
    the lenient fallback for display; the plan, which is what a rank actually
    admits against, does not get to guess.
    """

    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(
            f"node role must be a string, got {type(value).__name__}"
        )
    text = value.strip().lower()
    if not text:
        return ""
    if text not in ROLES:
        raise ValueError(
            f"unknown node role: {value!r} (expected one of "
            f"{', '.join(sorted(ROLES))})"
        )
    return text


def normalize_memory_guard_tier(value: Any) -> str:
    """A rank-safe memory policy name.

    This value is part of the signed per-rank plan. Silently turning a typo
    into ``balanced`` would make the GUI say one policy while the remote Mac
    admits against another, so plan construction and decoding both reject it.
    """

    if not isinstance(value, str):
        raise ValueError(
            f"memory guard tier must be a string, got {type(value).__name__}"
        )
    text = value.strip().lower()
    if text not in _MEMORY_GUARD_TIERS:
        raise ValueError(
            f"unknown memory guard tier: {value!r} (expected one of "
            f"{', '.join(sorted(_MEMORY_GUARD_TIERS))})"
        )
    return text


@dataclass(frozen=True)
class ModelLayout:
    source: str
    fixed_weight_bytes: int
    layer_weight_bytes: tuple[int, ...]
    tensor_count: int = 0
    activation_bytes_per_token: int = 0
    tensor_parallel_heads: int = 1
    # 0 means "same as tensor_parallel_heads" — models without grouped-query
    # attention omit num_key_value_heads entirely.
    tensor_parallel_kv_heads: int = 0
    # Every architecture-specific head/group dimension that its sharder
    # partitions. Qwen3-Next linear attention and Nemotron-H Mamba layers have
    # additional dimensions beyond the regular attention/KV pair.
    tensor_parallel_divisors: tuple[int, ...] = ()
    # Resident KV-cache bytes one layer adds per token. Reserved per node in
    # proportion to the layers it holds, so a plan that fits the weights
    # still fits once a long prompt fills the cache.
    kv_bytes_per_token_per_layer: int = 0
    # MLA models keep one latent cache per layer that every tensor-parallel
    # member holds whole; sharding divides the heads, not this cache.
    kv_replicated_across_tp: bool = False
    supports_tensor_parallel: bool = False
    # Whether mlx-lm can split this architecture into pipeline stages. False
    # means the model runs on one node or not at all, however well it fits.
    supports_pipeline: bool = False

    def __post_init__(self) -> None:
        if self.fixed_weight_bytes < 0:
            raise ValueError("fixed_weight_bytes must be non-negative")
        if not self.layer_weight_bytes:
            raise ValueError("at least one layer is required")
        if len(self.layer_weight_bytes) > _MAX_PIPELINE_LAYERS:
            raise ValueError(
                f"layer count exceeds the {_MAX_PIPELINE_LAYERS} layer limit"
            )
        if any(size < 0 for size in self.layer_weight_bytes):
            raise ValueError("layer weights must be non-negative")
        if self.tensor_count < 0:
            raise ValueError("tensor_count must be non-negative")
        if self.activation_bytes_per_token < 0:
            raise ValueError("activation_bytes_per_token must be non-negative")
        if self.tensor_parallel_heads < 1:
            raise ValueError("tensor_parallel_heads must be at least 1")
        if self.tensor_parallel_kv_heads < 0:
            raise ValueError("tensor_parallel_kv_heads must be non-negative")
        if self.kv_bytes_per_token_per_layer < 0:
            raise ValueError("kv_bytes_per_token_per_layer must be non-negative")
        if self.tensor_parallel_kv_heads == 0:
            object.__setattr__(
                self, "tensor_parallel_kv_heads", self.tensor_parallel_heads
            )
        if not self.tensor_parallel_divisors:
            object.__setattr__(
                self,
                "tensor_parallel_divisors",
                tuple(
                    dict.fromkeys(
                        (
                            self.tensor_parallel_heads,
                            self.tensor_parallel_kv_heads,
                        )
                    )
                ),
            )
        if any(value < 1 for value in self.tensor_parallel_divisors):
            raise ValueError("tensor_parallel_divisors must be positive")

    @property
    def layer_count(self) -> int:
        return len(self.layer_weight_bytes)

    @property
    def total_weight_bytes(self) -> int:
        return self.fixed_weight_bytes + sum(self.layer_weight_bytes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "tensor_count": self.tensor_count,
            "layer_count": self.layer_count,
            "fixed_weight_bytes": self.fixed_weight_bytes,
            "layer_weight_bytes": list(self.layer_weight_bytes),
            "total_weight_bytes": self.total_weight_bytes,
            "activation_bytes_per_token": self.activation_bytes_per_token,
            "tensor_parallel_heads": self.tensor_parallel_heads,
            "tensor_parallel_kv_heads": self.tensor_parallel_kv_heads,
            "tensor_parallel_divisors": list(self.tensor_parallel_divisors),
            "supports_tensor_parallel": self.supports_tensor_parallel,
            "supports_pipeline": self.supports_pipeline,
            "kv_bytes_per_token_per_layer": self.kv_bytes_per_token_per_layer,
            "kv_replicated_across_tp": self.kv_replicated_across_tp,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ModelLayout:
        """Rebuild a layout another node measured and sent back as JSON.

        ``layer_count`` and ``total_weight_bytes`` are dropped: they are
        derived, and trusting a peer's arithmetic over our own is how two nodes
        end up planning different models.
        """

        if not isinstance(payload, dict):
            raise PlanningError("model layout must be a JSON object")
        layers = payload.get("layer_weight_bytes")
        if not isinstance(layers, list) or any(
            not isinstance(size, int) or isinstance(size, bool) for size in layers
        ):
            raise PlanningError("model layout has no layer_weight_bytes")
        try:
            return cls(
                source=str(payload.get("source", "")),
                fixed_weight_bytes=int(payload["fixed_weight_bytes"]),
                layer_weight_bytes=tuple(layers),
                tensor_count=int(payload.get("tensor_count", 0)),
                activation_bytes_per_token=int(
                    payload.get("activation_bytes_per_token", 0)
                ),
                tensor_parallel_heads=int(payload.get("tensor_parallel_heads", 1)),
                tensor_parallel_kv_heads=int(
                    payload.get("tensor_parallel_kv_heads", 0)
                ),
                tensor_parallel_divisors=tuple(
                    int(value)
                    for value in payload.get("tensor_parallel_divisors", ())
                ),
                kv_bytes_per_token_per_layer=int(
                    payload.get("kv_bytes_per_token_per_layer", 0)
                ),
                kv_replicated_across_tp=bool(
                    payload.get("kv_replicated_across_tp", False)
                ),
                supports_tensor_parallel=bool(
                    payload.get("supports_tensor_parallel", False)
                ),
                supports_pipeline=bool(payload.get("supports_pipeline", False)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PlanningError(f"model layout is malformed: {exc}") from exc


@dataclass(frozen=True)
class NodeBudget:
    node_id: str
    capacity_bytes: int
    reserve_bytes: int = 0
    # True when the operator moved this node's memory slider. Roles provide
    # automatic defaults; an explicit limit is already the safety decision.
    manual_memory_limit: bool = False
    rank: int = 0
    performance: NodePerformanceProfile | None = None
    # A hand-chosen ceiling on how much *model* goes here, from the split
    # control. 0 means "let the planner balance it". Deliberately separate from
    # reserve_bytes: a reserve is memory the model may not touch at all, while
    # this caps only the weights, leaving the rest of the node for KV cache.
    # That is the whole point of moving the split — a Mac holding fewer layers
    # can hold a much longer context.
    max_weight_bytes: int = 0
    # Preferred model-weight allocation from the multi-node balance control.
    # Unlike max_weight_bytes this is a soft target: indivisible layers may
    # land either side of it, but the planner chooses the closest feasible
    # contiguous split. 0 keeps fully automatic balancing.
    target_weight_bytes: int = 0
    # "headless" or "workstation" — whether someone is using this Mac. It rides
    # the plan onto the assignment because the rank that enforces it is a
    # process on another machine: the launcher emits one argv that every host
    # runs identically, so a command-line flag cannot say "studio=headless,
    # macbook=workstation". The plan is already per-rank, so it can.
    # "" means the caller did not choose, and the guard falls back to headless.
    role: str = ""
    # The process-level memory policy chosen in the GUI. This must travel on
    # the per-rank plan for the same reason as ``role``: mlx.launch sends one
    # common argv to every host.
    memory_guard_tier: str = "balanced"

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", normalize_node_role(self.role))
        object.__setattr__(
            self,
            "memory_guard_tier",
            normalize_memory_guard_tier(self.memory_guard_tier),
        )
        if not self.node_id:
            raise ValueError("node_id is required")
        if self.capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        if self.reserve_bytes < 0:
            raise ValueError("reserve_bytes must be non-negative")
        if self.reserve_bytes >= self.capacity_bytes:
            raise ValueError("reserve_bytes must be smaller than capacity_bytes")
        if self.rank < 0:
            raise ValueError("rank must be non-negative")
        if self.max_weight_bytes < 0:
            raise ValueError("max_weight_bytes must be non-negative")
        if self.target_weight_bytes < 0:
            raise ValueError("target_weight_bytes must be non-negative")
        if self.target_weight_bytes > self.usable_bytes:
            raise ValueError("target_weight_bytes exceeds usable node memory")
        if self.performance is not None and (
            self.performance.node_id != self.node_id
            or self.performance.rank != self.rank
        ):
            raise ValueError("node performance profile must match its node ID and rank")

    @property
    def usable_bytes(self) -> int:
        return self.capacity_bytes - self.reserve_bytes

    @property
    def weight_ceiling_bytes(self) -> int:
        """The most model weight this node will be given.

        The node's own memory unless the split control has pinned it lower. A
        cap above what the node can hold is meaningless, so it is clamped.
        """

        if self.max_weight_bytes <= 0:
            return self.usable_bytes
        return min(self.usable_bytes, self.max_weight_bytes)


@dataclass(frozen=True)
class PipelineAssignment:
    node_id: str
    rank: int
    start_layer: int
    end_layer: int
    layer_weight_bytes: int
    fixed_weight_bytes: int
    reserve_bytes: int
    capacity_bytes: int
    manual_memory_limit: bool = False
    # How much of this Mac the cluster may take, carried per rank because that
    # is the only channel that reaches the rank. ``build_mlx_launch_argv``
    # emits one argument vector that mlx.launch runs unchanged on every host,
    # so a ``--node-role`` flag can only say one thing for the whole cluster;
    # the encoded plan is indexed by rank on arrival (``assignments[rank]``),
    # so it can say a different thing for each Mac. Empty means unset, which
    # the guard reads as headless.
    role: str = ""
    memory_guard_tier: str = "balanced"
    tensor_parallel_rank: int = 0
    tensor_parallel_size: int = 1
    sharded_weight_bytes: int = 0
    # KV cache this node must hold at the planned context length. Counted as
    # resident memory: weights that fit alone still OOM once a long prompt
    # fills the cache, which is how a 22-layer stage took a 128 GiB Mac down.
    kv_cache_bytes: int = 0
    # Bytes of KV this stage adds per token of context, and the largest context
    # this node could hold if the rest of its memory went to cache. The second
    # is the number that decides whether a long prompt works, so it is reported
    # rather than left to the caller to rederive.
    kv_bytes_per_token: int = 0
    max_context_tokens: int = 0
    predicted_compute_seconds: float | None = None
    predicted_send_seconds: float | None = None
    predicted_stage_seconds: float | None = None

    def __post_init__(self) -> None:
        # Normalised on the way in, not read leniently on the way out: this
        # object is decoded from a command line on a machine that will size its
        # own admission from it, and a role that arrives misspelled must fail
        # the launch rather than silently become the permissive one.
        object.__setattr__(self, "role", normalize_node_role(self.role))
        object.__setattr__(
            self,
            "memory_guard_tier",
            normalize_memory_guard_tier(self.memory_guard_tier),
        )

    @property
    def layer_count(self) -> int:
        return self.end_layer - self.start_layer

    @property
    def planned_weight_bytes(self) -> int:
        # ``layer_weight_bytes`` is what this node actually holds: under tensor
        # parallelism the stage's layers are already divided by the TP degree,
        # because shard_linear splits the projections inside each layer. Adding
        # ``sharded_weight_bytes`` on top double-counted the same weights.
        return self.fixed_weight_bytes + self.layer_weight_bytes + self.kv_cache_bytes

    @property
    def headroom_bytes(self) -> int:
        return self.capacity_bytes - self.reserve_bytes - self.planned_weight_bytes

    @property
    def utilization(self) -> float:
        return (self.reserve_bytes + self.planned_weight_bytes) / self.capacity_bytes

    def to_dict(self) -> dict[str, Any]:
        result = {
            "node_id": self.node_id,
            "rank": self.rank,
            "start_layer": self.start_layer,
            "end_layer": self.end_layer,
            "layer_count": self.layer_count,
            "layer_weight_bytes": self.layer_weight_bytes,
            "fixed_weight_bytes": self.fixed_weight_bytes,
            "planned_weight_bytes": self.planned_weight_bytes,
            "reserve_bytes": self.reserve_bytes,
            "capacity_bytes": self.capacity_bytes,
            "manual_memory_limit": self.manual_memory_limit,
            "headroom_bytes": self.headroom_bytes,
            "utilization": self.utilization,
            "role": self.role,
            "memory_guard_tier": self.memory_guard_tier,
            "tensor_parallel_rank": self.tensor_parallel_rank,
            "tensor_parallel_size": self.tensor_parallel_size,
            "sharded_weight_bytes": self.sharded_weight_bytes,
            "kv_cache_bytes": self.kv_cache_bytes,
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "max_context_tokens": self.max_context_tokens,
        }
        if self.predicted_stage_seconds is not None:
            result["predicted_compute_seconds"] = self.predicted_compute_seconds
            result["predicted_send_seconds"] = self.predicted_send_seconds
            result["predicted_stage_seconds"] = self.predicted_stage_seconds
        return result


@dataclass(frozen=True)
class ShardPlan:
    model: ModelLayout
    assignments: tuple[PipelineAssignment, ...]
    plan_hash: str
    optimization: str = "memory"
    workload_profile: ExecutionProfileName = "balanced"
    performance_profiles: tuple[NodePerformanceProfile, ...] = ()
    tensor_parallel_size: int = 1
    pipeline_stages: int = 1
    # The context that was actually reserved when this immutable plan was
    # built. Keep it explicit even when a model's KV shape is unknown: in that
    # case the per-rank byte reservation is zero, but the operator's requested
    # runtime limit must still be visible and part of the plan identity.
    target_context_tokens: int = _DEFAULT_CONTEXT_TOKENS

    @property
    def max_context_tokens(self) -> int:
        """The longest context this cluster can serve.

        A pipeline runs the same request through every stage, so the shortest
        stage sets the limit — reporting anything else would promise a context
        the weakest node cannot hold. 0 means the model's KV shape is unknown.
        """

        limits = [item.max_context_tokens for item in self.assignments]
        if not limits or any(limit <= 0 for limit in limits):
            return 0
        return min(limits)

    @property
    def cluster_kv_cache_bytes(self) -> int:
        return sum(item.kv_cache_bytes for item in self.assignments)

    @property
    def cluster_capacity_bytes(self) -> int:
        return sum(item.capacity_bytes for item in self.assignments)

    @property
    def cluster_reserve_bytes(self) -> int:
        return sum(item.reserve_bytes for item in self.assignments)

    @property
    def cluster_resident_weight_bytes(self) -> int:
        return sum(item.planned_weight_bytes for item in self.assignments)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_version": 1,
            "plan_hash": self.plan_hash,
            "strategy": (
                "performance_aware_unequal_contiguous_pipeline"
                if self.optimization == "performance"
                else "unequal_contiguous_pipeline"
            ),
            "optimization": self.optimization,
            "workload_profile": self.workload_profile,
            "model": self.model.to_dict(),
            "cluster": {
                "capacity_bytes": self.cluster_capacity_bytes,
                "reserve_bytes": self.cluster_reserve_bytes,
                "resident_weight_bytes": self.cluster_resident_weight_bytes,
                "kv_cache_bytes": self.cluster_kv_cache_bytes,
                "target_context_tokens": self.target_context_tokens,
                "max_context_tokens": self.max_context_tokens,
            },
            "assignments": [item.to_dict() for item in self.assignments],
            "performance_profiles": [
                profile.to_dict() for profile in self.performance_profiles
            ],
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_stages": self.pipeline_stages,
        }


def synthetic_model_layout(*, total_weight_bytes: int, layer_count: int) -> ModelLayout:
    """Create an even-layer model layout for planning before a download."""

    if total_weight_bytes <= 0:
        raise ValueError("total_weight_bytes must be positive")
    if layer_count <= 0:
        raise ValueError("layer_count must be positive")
    if layer_count > _MAX_PIPELINE_LAYERS:
        raise ValueError(f"layer_count exceeds the {_MAX_PIPELINE_LAYERS} layer limit")
    base, remainder = divmod(total_weight_bytes, layer_count)
    layers = tuple(
        base + (1 if index < remainder else 0) for index in range(layer_count)
    )
    return ModelLayout(
        source="synthetic",
        fixed_weight_bytes=0,
        layer_weight_bytes=layers,
    )


def _tensor_layer_index(name: str) -> int | None:
    for pattern in _LAYER_PATTERNS:
        if match := pattern.search(name):
            return int(match.group(1))
    return None


def _activation_bytes_per_token(model_path: Path) -> int:
    """Best-effort FP16/BF16 hidden-state size used by the link cost model."""

    config_path = model_path / "config.json"
    if not config_path.is_file():
        return 0
    config = _bounded_json_object(config_path, limit=_MAX_METADATA_FILE_BYTES)
    candidates = [config]
    for key in ("text_config", "language_config", "llm_config"):
        value = config.get(key)
        if isinstance(value, dict):
            candidates.append(value)
    hidden_size = next(
        (
            value
            for candidate in candidates
            if isinstance((value := candidate.get("hidden_size")), int)
            and not isinstance(value, bool)
            and 0 < value <= 1_000_000
        ),
        0,
    )
    # Pipeline activations in the pinned MLX-LM path are floating point even
    # when weights are quantized. Two bytes is the conservative common case.
    return hidden_size * 2


def _model_config(model_path: Path) -> dict[str, Any]:
    """Load config.json and its nested variants, or an empty dict."""

    config_path = model_path / "config.json"
    if not config_path.is_file():
        return {}
    return _bounded_json_object(config_path, limit=_MAX_METADATA_FILE_BYTES)


def _config_int(
    config: dict[str, Any],
    key: str,
    default: int,
    *,
    maximum: int = 4096,
) -> int:
    """Read a positive bounded integer from the model's text config.

    Most callers read counts or per-head dimensions, for which 4096 is a
    useful corruption bound. Whole-model widths are routinely larger than
    that, however, so those callers must opt into an appropriate ceiling
    instead of silently treating a valid value as absent.
    """

    candidates = [config]
    for nested in ("text_config", "language_config", "llm_config"):
        value = config.get(nested)
        if isinstance(value, dict):
            candidates.append(value)
    return next(
        (
            value
            for candidate in candidates
            if isinstance((value := candidate.get(key)), int)
            and not isinstance(value, bool)
            and 0 < value <= maximum
        ),
        default,
    )


def _config_nonnegative_int(config: dict[str, Any], key: str) -> int | None:
    """Read an explicitly present, bounded integer that may be zero."""
    candidates = [config]
    for nested in ("text_config", "language_config", "llm_config"):
        value = config.get(nested)
        if isinstance(value, dict):
            candidates.append(value)
    return next(
        (
            value
            for candidate in candidates
            if isinstance((value := candidate.get(key)), int)
            and not isinstance(value, bool)
            and 0 <= value <= 4096
        ),
        None,
    )


def _supports_tensor_parallel(config: dict[str, Any]) -> bool:
    """Whether this architecture has a runtime-safe tensor strategy.

    A method named ``shard`` is insufficient: progressive loading can invoke a
    native method one layer at a time only when all mutation is confined to its
    layer loop. Apply the same AST proof used by the worker here so the GUI
    never offers Tensor for a method the loader will later reject.
    """

    model_type = config.get("model_type")
    if not isinstance(model_type, str):
        return False
    if supports_model_type(model_type):
        return True
    if not model_type.replace("_", "").isalnum():
        return False
    try:
        import importlib

        module = importlib.import_module(f"mlx_lm.models.{model_type}")
        model_class = getattr(module, "Model", None)
        shard = getattr(model_class, "shard", None)
    except (ImportError, AttributeError):
        return False
    supported, _reason = native_shard_is_layer_local(shard)
    return supported


def _tensor_parallel_divisors(config: dict[str, Any]) -> tuple[int, ...]:
    """All model dimensions an architecture-specific TP strategy divides."""

    heads = _config_int(config, "num_attention_heads", 1)
    kv_heads = _config_int(config, "num_key_value_heads", heads)
    values = [heads, kv_heads]
    model_type = config.get("model_type")
    if model_type in {"qwen3_next", "qwen3_next_moe", "qwen3_5", "qwen3_5_moe"}:
        values.extend(
            (
                _config_int(config, "linear_num_key_heads", 1),
                _config_int(config, "linear_num_value_heads", 1),
            )
        )
    if model_type == "nemotron_h":
        values.extend(
            (
                _config_int(config, "mamba_num_heads", 1),
                _config_int(config, "n_groups", 1),
            )
        )
        # Quantized row-parallel projections split with the even
        # ``shard_inplace`` path (attention/Mamba ``out_proj`` and the
        # shared-expert ``down_proj``) require a quant-group count divisible by
        # the TP degree, or the split raises mid-load. Contribute those counts
        # so a degree the head counts would allow but the quantization forbids
        # is refused up front. The routed-expert ``fc1``/``fc2`` are excluded on
        # purpose: they use the custom uneven split in ``tensor_strategies`` and
        # only need ``group_count >= degree`` (see D1b in the cluster plan).
        quant = config.get("quantization")
        if isinstance(quant, dict):
            # oQ mixed-precision checkpoints add per-module override dicts
            # inside ``quantization`` (see _patch_mlx_lm_load_config). The
            # top-level group_size alone under-constrains those: a down_proj
            # overridden to a coarser group can have a prime group count the
            # top-level size hides, approving a degree that then raises
            # mid-load. Constrain against every group size present; a size
            # that does not divide a projection cannot have quantized it.
            group_sizes = {_config_int(quant, "group_size", 0)}
            for override in quant.values():
                if isinstance(override, dict):
                    group_sizes.add(_config_int(override, "group_size", 0))
            group_sizes.discard(0)
            if group_sizes:
                head_dim = _config_int(config, "head_dim", 0) or (
                    _config_int(config, "hidden_size", 0, maximum=1_000_000)
                    // max(_config_int(config, "num_attention_heads", 1), 1)
                )
                attn_dim = _config_int(config, "num_attention_heads", 0) * head_dim
                mamba_dim = _config_int(
                    config, "mamba_num_heads", 0
                ) * _config_int(config, "mamba_head_dim", 0)
                shared_dim = _config_int(
                    config,
                    "moe_shared_expert_intermediate_size",
                    0,
                    maximum=1_000_000,
                )
                for dim in (attn_dim, mamba_dim, shared_dim):
                    for group_size in group_sizes:
                        if dim > 0 and dim % group_size == 0:
                            values.append(dim // group_size)
    return tuple(dict.fromkeys(values))


def _model_source(model_type: str) -> str:
    """Source of the mlx-lm module for this model type, or "".

    Consults ``sys.modules`` first so architectures oMLX registers itself —
    ``minimax_m3_vl``, ``deepseek_v4`` — are judged on the module that will
    actually be imported rather than a file mlx-lm does not ship.
    """

    if not model_type.replace("_", "").isalnum():
        return ""

    import sys

    registered = sys.modules.get(f"mlx_lm.models.{model_type}")
    origin = getattr(getattr(registered, "__spec__", None), "origin", None)
    candidates = []
    if origin:
        candidates.append(Path(origin))
    try:
        import mlx_lm.models

        candidates.append(Path(mlx_lm.models.__file__).parent / f"{model_type}.py")
    except ImportError:  # pragma: no cover - mlx-lm is a hard dependency
        pass

    for path in candidates:
        try:
            if path.is_file():
                return path.read_text()
        except OSError:
            continue
    return ""


def _supports_pipeline(config: dict[str, Any]) -> bool:
    """Whether this architecture can be split into pipeline stages.

    mlx-lm gates on ``hasattr(model, "model") and hasattr(model.model,
    "pipeline")``: a model must both blank the layers a rank does not own and
    run a forward pass that hands activations between ranks. That is
    per-architecture work, not a launcher flag — a model written for one Mac
    has none of it.

    Checked here because the alternative is finding out late and expensively:
    MiniMax-M3 was reported as fitting across two Macs on memory alone, and
    that cost 61.7 GiB of staging and two launches before the load raised
    "The model does not support pipelining but a pipeline_group was provided".
    """

    model_type = config.get("model_type")
    if not isinstance(model_type, str):
        return False
    # An architecture oMLX explicitly vouches for wins, even a VLM: the
    # minimax_m3_vl patch ships its own ``pipeline()`` and sets
    # ``SUPPORTS_PIPELINE = True``. Honour that before the vision guard below.
    import sys

    declared = getattr(
        sys.modules.get(f"mlx_lm.models.{model_type}"), "SUPPORTS_PIPELINE", None
    )
    if declared is not None:
        return bool(declared)
    # A checkpoint carrying a vision sub-config is served by mlx-vlm, whose
    # loaded wrapper never exposes ``model.model.pipeline`` — the exact
    # attribute progressive_loading gates on. Its text backbone's source-level
    # ``pipeline()`` belongs to the mlx-lm implementation this model does not
    # use, so trusting it (as ``_declares_pipeline`` does) is the false positive
    # that offered pipeline for Qwen3.5/3.6-family VLMs and then failed at load.
    from omlx.model_discovery import _has_vision_subconfig

    if _has_vision_subconfig(config):
        return False
    return _declares_pipeline(model_type)


# Only about seven of mlx-lm's ~120 architectures can be pipelined, so the
# answer is nearly always no — worth knowing before staging a hundred
# gigabytes on the assumption that it is yes.
_PIPELINE_MARKERS = ("def pipeline(", "PipelineMixin")

# ``from .deepseek_v32 import Model as DSV32Model`` — glm_moe_dsa inherits its
# pipeline support this way, so a source-only check would wrongly refuse it.
_MODEL_BASE_IMPORT = re.compile(r"^from \.(\w+) import .*\bModel\b", re.MULTILINE)


def _declares_pipeline(model_type: str, *, seen: frozenset[str] = frozenset()) -> bool:
    """Does this architecture, or one it inherits from, implement pipelining?"""

    if model_type in seen or len(seen) > 4:
        return False

    # A module oMLX registers may install pipelining by patching a vendored
    # class, which no amount of source-reading will reveal. Let it say so.
    import sys

    declared = getattr(
        sys.modules.get(f"mlx_lm.models.{model_type}"), "SUPPORTS_PIPELINE", None
    )
    if declared is not None:
        return bool(declared)
    source = _model_source(model_type)
    if not source:
        return False
    if any(marker in source for marker in _PIPELINE_MARKERS):
        return True
    return any(
        _declares_pipeline(base, seen=seen | {model_type})
        for base in _MODEL_BASE_IMPORT.findall(source)
    )


def _kv_cache_replicated_across_tp(config: dict[str, Any]) -> bool:
    """Whether every tensor-parallel member holds this model's full KV cache.

    ``shard()`` splits attention heads, but an MLA latent cache is not
    per-head: GLM/DeepSeek-style layers store one compressed latent plus RoPE
    key that every member of the group needs whole. Dividing that reservation
    by the TP degree under-reserved each rank by exactly that factor.
    """

    return (
        _config_int(config, "kv_lora_rank", 0) > 0
        and _config_nonnegative_int(config, "qk_rope_head_dim") is not None
    )


def _kv_bytes_per_token_per_layer(config: dict[str, Any]) -> int:
    """Resident KV-cache bytes each layer adds per token.

    Mirrors the shape ``MemoryMonitor.set_model_info`` uses —
    ``num_kv_heads * head_dim * 2 * dtype_size`` — with the same MLA exception
    it documents: GLM/DeepSeek-style models store a latent key plus a RoPE
    value under a single KV head, so the uniform formula over-counts them by
    more than an order of magnitude.

    Returns 0 when the config does not describe the cache, so callers can tell
    "no KV reservation" from "zero KV".
    """

    # Stored cache is fp16/bf16 even when weights are quantised.
    dtype_size = 2

    if _kv_cache_replicated_across_tp(config):
        # One KV head holding latent + RoPE, not expanded K/V tensors.
        kv_lora_rank = _config_int(config, "kv_lora_rank", 0)
        rope_dim = _config_nonnegative_int(config, "qk_rope_head_dim") or 0
        text_config = config.get("text_config")
        layer_config = text_config if isinstance(text_config, dict) else config
        layer_types = layer_config.get("layer_types")
        if config.get("model_type") == "glm5_next" and isinstance(
            layer_types, list
        ):
            num_layers = _config_int(config, "num_hidden_layers", 0)
            sparse_layers = sum(
                layer_type != "linear_attention" for layer_type in layer_types
            )
            index_dim = _config_int(config, "index_head_dim", 0)
            index_pool = _config_int(config, "index_kpool", 1)
            if num_layers > 0 and sparse_layers > 0:
                # ModelLayout stores one integral average per layer. Keep the
                # exact full-model total conservative by rounding upward.
                numerator = (
                    sparse_layers
                    * ((kv_lora_rank + rope_dim) * index_pool + index_dim)
                    * dtype_size
                )
                denominator = num_layers * index_pool
                return (numerator + denominator - 1) // denominator
        return (kv_lora_rank + rope_dim) * dtype_size

    heads = _config_int(config, "num_attention_heads", 0)
    kv_heads = _config_int(config, "num_key_value_heads", heads)
    head_dim = _config_int(config, "head_dim", 0)
    if head_dim <= 0:
        hidden = _config_int(config, "hidden_size", 0, maximum=1_000_000)
        head_dim = (
            hidden // heads
            if hidden and heads and hidden % heads == 0
            else 0
        )
    if head_dim > 4096:
        return 0
    if kv_heads <= 0 or head_dim <= 0:
        return 0
    return kv_heads * head_dim * 2 * dtype_size


def _attention_head_count(model_path: Path) -> int:
    """Attention heads per layer, which bounds the tensor-parallel degree.

    ``shard()`` does ``n_heads //= N``, so a TP degree that does not divide the
    head count silently reshapes attention wrongly — the planner refuses such a
    split, which means a model whose head count we cannot read can never use
    tensor parallelism. Returns 1 (the "unknown, assume no TP" value) when the
    config is missing or unreadable rather than guessing a plausible number.
    """

    config_path = model_path / "config.json"
    if not config_path.is_file():
        return 1
    config = _bounded_json_object(config_path, limit=_MAX_METADATA_FILE_BYTES)
    candidates = [config]
    for key in ("text_config", "language_config", "llm_config"):
        value = config.get(key)
        if isinstance(value, dict):
            candidates.append(value)
    return next(
        (
            value
            for candidate in candidates
            if isinstance((value := candidate.get("num_attention_heads")), int)
            and not isinstance(value, bool)
            and 0 < value <= 4096
        ),
        1,
    )


def _safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        encoded_size = stream.read(8)
        if len(encoded_size) != 8:
            raise PlanningError(f"{path.name} is not a valid safetensors file")
        header_size = struct.unpack("<Q", encoded_size)[0]
        if not 2 <= header_size <= _MAX_SAFETENSORS_HEADER_BYTES:
            raise PlanningError(f"{path.name} has an invalid safetensors header size")
        if 8 + header_size > file_size:
            raise PlanningError(f"{path.name} has a truncated safetensors header")
        try:
            header = json.loads(stream.read(header_size))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PlanningError(
                f"{path.name} has an invalid safetensors header"
            ) from exc
    if not isinstance(header, dict):
        raise PlanningError(f"{path.name} safetensors header must be an object")
    return header, file_size - 8 - header_size


def _bounded_json_object(path: Path, *, limit: int) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            encoded = stream.read(limit + 1)
    except OSError as exc:
        raise PlanningError(f"could not read {path.name}: {exc}") from exc
    if len(encoded) > limit:
        raise PlanningError(f"{path.name} exceeds the {limit} byte limit")
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PlanningError(f"could not read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise PlanningError(f"{path.name} must contain a JSON object")
    return value


def _model_weight_files(model_path: Path) -> tuple[Path, ...]:
    index_path = model_path / "model.safetensors.index.json"
    names: set[str] = set()
    if index_path.is_file():
        index = _bounded_json_object(
            index_path,
            limit=_MAX_SAFETENSORS_INDEX_BYTES,
        )
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise PlanningError(f"{index_path.name} has no weight_map object")
        if not weight_map:
            raise PlanningError(f"{index_path.name} has an empty weight_map")
        values = tuple(weight_map.values())
        if any(
            not isinstance(value, str) or not value.endswith(".safetensors")
            for value in values
        ):
            raise PlanningError(
                f"{index_path.name} contains an invalid weight filename"
            )
        names = set(values)
    else:
        names = {path.name for path in model_path.glob("*.safetensors")}

    root = model_path.resolve()
    files: list[Path] = []
    for name in sorted(names):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise PlanningError(f"weight file escapes model directory: {name}")
        candidate = root / relative
        if not candidate.is_file():
            raise PlanningError(f"weight file is missing: {name}")
        files.append(candidate)
    if not files:
        raise PlanningError(f"no safetensors weights found in {model_path}")
    return tuple(files)


def inspect_safetensors_layout(model_path: str | Path) -> ModelLayout:
    """Read only safetensors headers and total weights by transformer layer."""

    root = Path(model_path).expanduser()
    if not root.is_dir():
        raise PlanningError(f"model path is not a directory: {root}")

    fixed_bytes = 0
    layer_sizes: dict[int, int] = {}
    tensor_names: set[str] = set()
    tensor_count = 0
    for weight_file in _model_weight_files(root):
        header, payload_bytes = _safetensors_header(weight_file)
        intervals: list[tuple[int, int, str]] = []
        for name, spec in header.items():
            if name == "__metadata__":
                continue
            if name in tensor_names:
                raise PlanningError(f"duplicate tensor name: {name}")
            tensor_names.add(name)
            if not isinstance(spec, dict):
                raise PlanningError(f"invalid tensor metadata for {name}")
            offsets = spec.get("data_offsets")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(
                    isinstance(value, int) and not isinstance(value, bool)
                    for value in offsets
                )
                or offsets[0] < 0
                or offsets[1] < offsets[0]
                or offsets[1] > payload_bytes
            ):
                raise PlanningError(f"invalid data offsets for tensor {name}")
            if offsets[1] > offsets[0]:
                intervals.append((offsets[0], offsets[1], name))
            tensor_bytes = offsets[1] - offsets[0]
            layer_index = _tensor_layer_index(name)
            if layer_index is None:
                fixed_bytes += tensor_bytes
            else:
                layer_sizes[layer_index] = (
                    layer_sizes.get(layer_index, 0) + tensor_bytes
                )
            tensor_count += 1
        intervals.sort()
        for previous, current in zip(intervals, intervals[1:]):
            if current[0] < previous[1]:
                raise PlanningError(
                    "overlapping safetensors data offsets for "
                    f"{previous[2]} and {current[2]}"
                )

    if not layer_sizes:
        raise PlanningError(
            "could not identify transformer layers in safetensors names"
        )
    # Checkpoints may carry layers past the declared depth: DeepSeek/GLM-style
    # multi-token-prediction heads live at index num_hidden_layers and up, and
    # the runtime model never instantiates them. Counting them as decoder
    # layers put a stage boundary over weights that do not exist at runtime,
    # and the last stage failed activation with end_layer beyond the model.
    declared_depth = _config_int(
        _model_config(root), "num_hidden_layers", 0
    )
    if declared_depth > 0 and max(layer_sizes) >= declared_depth:
        trimmed = {
            index: size
            for index, size in layer_sizes.items()
            if index < declared_depth
        }
        if trimmed:
            layer_sizes = trimmed
    maximum_layer = max(layer_sizes)
    if maximum_layer >= _MAX_PIPELINE_LAYERS:
        raise PlanningError(
            f"safetensors layer index exceeds {_MAX_PIPELINE_LAYERS - 1}"
        )
    expected_indices = list(range(maximum_layer + 1))
    if sorted(layer_sizes) != expected_indices:
        raise PlanningError(
            "safetensors layer indices must be contiguous and start at zero"
        )
    return ModelLayout(
        source=str(root.resolve()),
        fixed_weight_bytes=fixed_bytes,
        layer_weight_bytes=tuple(layer_sizes[index] for index in expected_indices),
        tensor_count=tensor_count,
        activation_bytes_per_token=_activation_bytes_per_token(root),
        tensor_parallel_heads=_attention_head_count(root),
        tensor_parallel_kv_heads=_config_int(
            _model_config(root), "num_key_value_heads", 0
        ),
        tensor_parallel_divisors=_tensor_parallel_divisors(_model_config(root)),
        supports_tensor_parallel=_supports_tensor_parallel(_model_config(root)),
        supports_pipeline=_supports_pipeline(_model_config(root)),
        kv_bytes_per_token_per_layer=_kv_bytes_per_token_per_layer(
            _model_config(root)
        ),
        kv_replicated_across_tp=_kv_cache_replicated_across_tp(
            _model_config(root)
        ),
    )


# Directory mtime + config mtime + per-shard (name, mtime, size). The shard
# stats matter: an in-place shard overwrite changes neither the directory's
# mtime (no entry added or removed) nor config.json's, and a stale layout
# would silently mis-size every plan built from it.
_LayoutFingerprint = tuple[float, float, tuple[tuple[str, float, int], ...]]
_LAYOUT_CACHE: dict[str, tuple[_LayoutFingerprint, ModelLayout]] = {}
_LAYOUT_CACHE_LOCK = threading.Lock()


def complete_model_layout(model_path: str | Path) -> ModelLayout:
    """Layout of a model this node holds in full, refusing one it holds part of.

    A pipeline rank keeps its own stage's shards and nothing else. Where the
    model ships an index that is loud — ``weight file is missing`` — and where
    it does not, it is silent: the headers present describe layers 0 to 55, and
    nothing in them says the other 22 exist on another Mac. Planning from that
    number splits a model that does not exist, so a layout is measured against
    the depth declared in config.json — a sidecar every node is staged.

    The planner recomputes this for every discovered model on every
    autoconfigure tick (the admin dashboard's cluster tab polls every ~10s
    while open), so opening and re-parsing every safetensors shard's header
    each call put real, sustained disk I/O and CPU load on a node that may be
    mid-inference. Layouts are cached per resolved path and only recomputed
    when the model directory or its config.json actually changed.
    """

    root = Path(model_path).expanduser()
    # Some architectures are registered into mlx-lm by oMLX immediately before
    # loading (MiniMax-M3 and DeepSeek-V4 are examples).  Capability detection
    # must inspect that registered module, not conclude that the model cannot be
    # split because stock mlx-lm has no file for it.  This does not load model
    # weights; the dispatchers are idempotent compatibility registrations.
    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    maybe_apply_pre_load_patches(str(root))

    resolved = str(root.resolve())
    try:
        shard_stats = []
        for shard_path in sorted(root.glob("*.safetensors")):
            stat = shard_path.stat()
            shard_stats.append((shard_path.name, stat.st_mtime, stat.st_size))
        fingerprint: _LayoutFingerprint | None = (
            root.stat().st_mtime,
            (root / "config.json").stat().st_mtime,
            tuple(shard_stats),
        )
    except OSError:
        fingerprint = None

    layout: ModelLayout | None = None
    if fingerprint is not None:
        with _LAYOUT_CACHE_LOCK:
            cached = _LAYOUT_CACHE.get(resolved)
        if cached is not None and cached[0] == fingerprint:
            layout = cached[1]

    if layout is None:
        layout = inspect_safetensors_layout(root)
        if fingerprint is not None:
            with _LAYOUT_CACHE_LOCK:
                _LAYOUT_CACHE[resolved] = (fingerprint, layout)

    declared = _config_int(_model_config(root), "num_hidden_layers", 0)
    # Multi-token-prediction and draft heads add layers past the declared
    # depth, so only a shortfall means missing weights.
    if declared and layout.layer_count < declared:
        raise PlanningError(
            f"{root} holds {layout.layer_count} of {declared} layers: this node "
            "has its own stage of the model, not the whole model"
        )
    return layout


_REMOTE_LAYOUT_SNIPPET = (
    "import json,sys;"
    "from omlx.cluster.planner import complete_model_layout;"
    "print(json.dumps(complete_model_layout(sys.argv[1]).to_dict()))"
)


def remote_model_layout(
    ssh_target: str,
    model_dir: str,
    *,
    python_executable: str = DEFAULT_REMOTE_PYTHON,
    timeout: float = 600.0,
) -> ModelLayout:
    """Measure a model that lives on a peer, on the peer.

    The same treatment staging gives shard indexing: the node that has the
    weights runs the identical code and sends its answer back, instead of the
    answer being worked out by hand and carried across as JSON.
    """

    try:
        payload = run_remote_python(
            ssh_target,
            _REMOTE_LAYOUT_SNIPPET,
            str(model_dir),
            description="read the model layout",
            python_executable=python_executable,
            timeout=timeout,
        )
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        raise PlanningError(str(exc)) from exc
    return ModelLayout.from_dict(payload)


LOCAL_NODE = "local"


@dataclass(frozen=True)
class ModelHolder:
    """A node that turned out to hold the whole model, and what it read.

    ``node`` is ``LOCAL_NODE`` or the ssh target that answered — which is also
    the host to stage the model from, since it is the one that has it.
    """

    node: str
    layout: ModelLayout

    @property
    def is_local(self) -> bool:
        return self.node == LOCAL_NODE


def locate_model_layout(
    model_path: str | Path,
    hosts: Sequence[str] = (),
    *,
    python_executable: str = DEFAULT_REMOTE_PYTHON,
    timeout: float = 600.0,
) -> ModelHolder:
    """Read a model's layout from whichever node holds all of it.

    Which node that is cannot be assumed: the Mac being planned for is often
    the one holding a single stage, and the Mac holding the model is whichever
    one it was downloaded to. So every candidate is asked in turn — this node
    first, because that costs no ssh — and the first that can read a complete
    model is the one that plans. ``model_path`` is the path on every node, so
    a ``~``-relative one is expanded by each against its own home.
    """

    refusals: list[str] = []
    try:
        return ModelHolder(node=LOCAL_NODE, layout=complete_model_layout(model_path))
    except (PlanningError, OSError) as exc:
        refusals.append(f"{LOCAL_NODE}: {exc}")

    for host in hosts:
        # A peer that is this machine has already answered above.
        if not host.strip() or is_local_host(host):
            continue
        try:
            layout = remote_model_layout(
                host,
                str(model_path),
                python_executable=python_executable,
                timeout=timeout,
            )
        except PlanningError as exc:
            refusals.append(f"{host}: {exc}")
            continue
        return ModelHolder(node=host, layout=layout)

    raise PlanningError(
        f"no node holds a complete copy of {model_path} ({'; '.join(refusals)})"
    )


def _partition_layers(
    layer_sizes: Sequence[int],
    pipeline_nodes: Sequence[NodeBudget],
    *,
    fixed_weight_bytes: int,
    layer_resident_sizes: Sequence[int] | None = None,
    activation_bytes_per_token: int = 0,
    workload_profile: ExecutionProfileName = "balanced",
    microbatch_size: int = 1,
) -> tuple[tuple[int, int], ...]:
    """Minimize the bottleneck stage while retaining contiguous layers.

    ``layer_sizes`` is model weight and remains the quantity used for compute
    prediction and manual weight targets. ``layer_resident_sizes`` may include
    a layer's context-dependent KV cache as well. Keeping the two separate is
    essential: a performance pass is allowed to move weights toward a faster
    Mac, but never so far that the resulting stage cannot hold its KV cache.
    """

    layer_count = len(layer_sizes)
    node_count = len(pipeline_nodes)
    resident_sizes = (
        tuple(layer_sizes)
        if layer_resident_sizes is None
        else tuple(layer_resident_sizes)
    )
    if len(resident_sizes) != layer_count:
        raise ValueError("layer_resident_sizes must match layer_sizes")
    if node_count > layer_count:
        raise PlanningError(
            f"{node_count} nodes cannot each receive a layer from {layer_count} layers"
        )

    weight_prefix = [0]
    resident_prefix = [0]
    for weight_size, resident_size in zip(layer_sizes, resident_sizes):
        if resident_size < weight_size:
            raise ValueError("resident layer size cannot be smaller than its weights")
        weight_prefix.append(weight_prefix[-1] + weight_size)
        resident_prefix.append(resident_prefix[-1] + resident_size)

    performance_aware = all(node.performance is not None for node in pipeline_nodes)
    target_aware = any(node.target_weight_bytes > 0 for node in pipeline_nodes)
    # State value: (lexicographic score, cuts). The no-profile score is kept
    # exactly equivalent to the original memory-only planner.
    base_score = (0.0, 0.0, 0.0, 0.0) if performance_aware else (0.0, 0.0)
    states: dict[int, tuple[tuple[float, ...], tuple[int, ...]]] = {
        0: (((0.0, 0.0) + base_score if target_aware else base_score), (0,))
    }
    for node_index, node in enumerate(pipeline_nodes):
        next_states: dict[
            int,
            tuple[tuple[float, ...], tuple[int, ...]],
        ] = {}
        remaining_nodes = node_count - node_index - 1
        minimum_end = node_index + 1
        maximum_end = layer_count - remaining_nodes
        for previous_end, (previous_score, cuts) in states.items():
            start = previous_end
            for end in range(max(start + 1, minimum_end), maximum_end + 1):
                layer_bytes = weight_prefix[end] - weight_prefix[start]
                resident_layer_bytes = (
                    resident_prefix[end] - resident_prefix[start]
                )
                planned_weight_bytes = fixed_weight_bytes + layer_bytes
                planned_resident_bytes = fixed_weight_bytes + resident_layer_bytes
                if planned_weight_bytes > node.weight_ceiling_bytes:
                    continue
                if planned_resident_bytes > node.usable_bytes:
                    continue
                utilization = (
                    node.reserve_bytes + planned_resident_bytes
                ) / node.capacity_bytes
                score_offset = 2 if target_aware else 0
                if performance_aware:
                    _, _, stage_seconds = _predict_stage_seconds(
                        layer_bytes,
                        node,
                        activation_bytes_per_token=activation_bytes_per_token,
                        workload_profile=workload_profile,
                        microbatch_size=microbatch_size,
                    )
                    score = (
                        max(previous_score[score_offset], stage_seconds),
                        previous_score[score_offset + 1]
                        + stage_seconds * stage_seconds,
                        max(previous_score[score_offset + 2], utilization),
                        previous_score[score_offset + 3]
                        + utilization * utilization,
                    )
                else:
                    score = (
                        max(previous_score[score_offset], utilization),
                        previous_score[score_offset + 1]
                        + utilization * utilization,
                    )
                if target_aware:
                    target = node.target_weight_bytes
                    deviation = (
                        abs(planned_weight_bytes - target) / max(1, target)
                        if target > 0
                        else 0.0
                    )
                    score = (
                        max(previous_score[0], deviation),
                        previous_score[1] + deviation * deviation,
                    ) + score
                existing = next_states.get(end)
                if existing is None or score < existing[0]:
                    next_states[end] = (score, cuts + (end,))
        states = next_states
        if not states:
            break

    final = states.get(layer_count)
    if final is None:
        resident_capacity = sum(
            max(0, node.usable_bytes - fixed_weight_bytes)
            for node in pipeline_nodes
        )
        weight_capacity = sum(
            max(0, node.weight_ceiling_bytes - fixed_weight_bytes)
            for node in pipeline_nodes
        )
        weight_shortfall = max(0, sum(layer_sizes) - weight_capacity)
        resident_shortfall = max(0, sum(resident_sizes) - resident_capacity)
        # Preserve the catalogue's long-standing model-weight shortfall when
        # the weights themselves cannot fit. KV becomes the reported shortfall
        # only when weights fit and context is the actual blocker.
        shortfall = weight_shortfall or resident_shortfall
        raise PlanningError(
            "model does not fit the supplied per-node budgets"
            + (
                f" (at least {shortfall} additional bytes required)"
                if shortfall
                else ""
            )
        )
    cuts = final[1]
    return tuple((cuts[index], cuts[index + 1]) for index in range(node_count))


def _predict_stage_seconds(
    layer_weight_bytes: int,
    node: NodeBudget,
    *,
    activation_bytes_per_token: int,
    workload_profile: ExecutionProfileName,
    microbatch_size: int,
) -> tuple[float, float, float]:
    profile = node.performance
    if profile is None:
        return 0.0, 0.0, 0.0
    workload_weights = {
        "interactive": (0.8, 0.2),
        "balanced": (0.5, 0.5),
        "throughput": (0.25, 0.75),
    }
    if workload_profile not in workload_weights:
        raise ValueError("workload profile is invalid")
    decode_weight, prefill_weight = workload_weights[workload_profile]
    decode_seconds = layer_weight_bytes / profile.decode_weight_bytes_per_second
    prefill_seconds = layer_weight_bytes / profile.prefill_weight_bytes_per_second
    compute_seconds = decode_weight * decode_seconds + prefill_weight * prefill_seconds
    send_seconds = 0.0
    if node.rank != 0 and activation_bytes_per_token > 0:
        send_seconds = profile.collective_latency_seconds + (
            activation_bytes_per_token
            * max(1, microbatch_size)
            / profile.collective_bandwidth_bytes_per_second
        )
    return compute_seconds, send_seconds, compute_seconds + send_seconds


def plan_unequal_pipeline(
    model: ModelLayout,
    nodes: Sequence[NodeBudget],
    *,
    workload_profile: ExecutionProfileName = "balanced",
    microbatch_size: int = 1,
    context_tokens: int = _DEFAULT_CONTEXT_TOKENS,
) -> ShardPlan:
    """Assign contiguous layers in MLX pipeline order across unequal nodes."""

    if not nodes:
        raise ValueError("at least one node is required")
    if workload_profile not in {"interactive", "balanced", "throughput"}:
        raise ValueError("workload profile is invalid")
    if (
        not isinstance(microbatch_size, int)
        or isinstance(microbatch_size, bool)
        or microbatch_size <= 0
    ):
        raise ValueError("microbatch_size must be a positive integer")
    ranks = [node.rank for node in nodes]
    if len(set(ranks)) != len(ranks) or set(ranks) != set(range(len(nodes))):
        raise ValueError("node ranks must be unique and contiguous from zero")
    node_ids = [node.node_id for node in nodes]
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("node IDs must be unique")
    for node in nodes:
        if model.fixed_weight_bytes > node.usable_bytes:
            raise PlanningError(
                f"replicated fixed weights do not fit node {node.node_id}"
            )

    # MLX-LM sends activations from the highest rank (early layers) down to
    # rank zero (late layers / HTTP coordinator), so partition in reverse rank.
    pipeline_nodes = tuple(sorted(nodes, key=lambda item: item.rank, reverse=True))
    performance_aware = all(node.performance is not None for node in pipeline_nodes)
    kv_bytes_per_layer = _kv_bytes_for_stage(model, 1, context_tokens)
    try:
        ranges = _partition_layers(
            model.layer_weight_bytes,
            pipeline_nodes,
            fixed_weight_bytes=model.fixed_weight_bytes,
            layer_resident_sizes=tuple(
                weight_bytes + kv_bytes_per_layer
                for weight_bytes in model.layer_weight_bytes
            ),
            activation_bytes_per_token=model.activation_bytes_per_token,
            workload_profile=workload_profile,
            microbatch_size=microbatch_size,
        )
    except PlanningError as exc:
        if kv_bytes_per_layer:
            raise PlanningError(
                f"model weights and KV cache for {context_tokens} tokens do not "
                f"fit the supplied per-node budgets: {exc}"
            ) from exc
        raise
    assignments: list[PipelineAssignment] = []
    for node, (start, end) in zip(pipeline_nodes, ranges):
        layer_weight_bytes = sum(model.layer_weight_bytes[start:end])
        kv_bytes = _kv_bytes_for_stage(model, end - start, context_tokens)
        planned = model.fixed_weight_bytes + layer_weight_bytes + kv_bytes
        if planned > node.usable_bytes:
            raise PlanningError(
                f"stage does not fit node {node.node_id} once KV cache for "
                f"{context_tokens} tokens is reserved: {planned} > "
                f"{node.usable_bytes} (weights {layer_weight_bytes}, "
                f"KV {kv_bytes})"
            )
        compute_seconds, send_seconds, stage_seconds = _predict_stage_seconds(
            layer_weight_bytes,
            node,
            activation_bytes_per_token=model.activation_bytes_per_token,
            workload_profile=workload_profile,
            microbatch_size=microbatch_size,
        )
        assignments.append(
            PipelineAssignment(
                node_id=node.node_id,
                rank=node.rank,
                start_layer=start,
                end_layer=end,
                layer_weight_bytes=layer_weight_bytes,
                fixed_weight_bytes=model.fixed_weight_bytes,
                reserve_bytes=node.reserve_bytes,
                capacity_bytes=node.capacity_bytes,
                manual_memory_limit=node.manual_memory_limit,
                role=node.role,
                memory_guard_tier=node.memory_guard_tier,
                kv_cache_bytes=kv_bytes,
                kv_bytes_per_token=_kv_bytes_per_token_for_stage(model, end - start),
                max_context_tokens=_max_context_for_stage(
                    model,
                    node,
                    layer_count=end - start,
                    weight_bytes=model.fixed_weight_bytes + layer_weight_bytes,
                ),
                predicted_compute_seconds=(
                    compute_seconds if performance_aware else None
                ),
                predicted_send_seconds=(send_seconds if performance_aware else None),
                predicted_stage_seconds=(stage_seconds if performance_aware else None),
            )
        )
    assignments.sort(key=lambda item: item.rank)

    hash_payload = {
        "model": model.to_dict(),
        "context_tokens": context_tokens,
        "optimization": "performance" if performance_aware else "memory",
        "workload_profile": workload_profile,
        "microbatch_size": microbatch_size,
        "performance_profiles": [
            node.performance.to_dict()
            for node in sorted(nodes, key=lambda item: item.rank)
            if node.performance is not None
        ],
        "assignments": [
            {
                "node_id": item.node_id,
                "rank": item.rank,
                "start_layer": item.start_layer,
                "end_layer": item.end_layer,
                "capacity_bytes": item.capacity_bytes,
                "reserve_bytes": item.reserve_bytes,
                "manual_memory_limit": item.manual_memory_limit,
                "target_weight_bytes": next(
                    node.target_weight_bytes
                    for node in nodes
                    if node.rank == item.rank
                ),
                # Part of the plan's identity: the same layers on the same Mac
                # mean a different deployment depending on whether someone is
                # using it, because the rank admits against a different
                # fraction. Two plans that differ only in role must not share
                # a hash, or an approved headless plan and a launched
                # workstation plan look identical to every staleness check.
                "role": item.role,
                "memory_guard_tier": item.memory_guard_tier,
            }
            for item in assignments
        ],
    }
    plan_hash = hashlib.sha256(
        json.dumps(hash_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return ShardPlan(
        model=model,
        assignments=tuple(assignments),
        plan_hash=plan_hash,
        optimization="performance" if performance_aware else "memory",
        workload_profile=workload_profile,
        performance_profiles=tuple(
            node.performance
            for node in sorted(nodes, key=lambda item: item.rank)
            if node.performance is not None
        ),
        target_context_tokens=context_tokens,
    )


def apply_pipeline_assignment(
    pipeline_model: Any,
    group: Any,
    assignments: Sequence[PipelineAssignment],
) -> None:
    """Apply one planned range to an MLX-LM ``PipelineMixin`` instance."""

    assignments_by_rank = {item.rank: item for item in assignments}
    rank = group.rank()
    size = group.size()
    if size != len(assignments_by_rank) or rank not in assignments_by_rank:
        raise PlanningError("runtime distributed group does not match shard plan")
    assignment = assignments_by_rank[rank]
    original_layer_count = len(pipeline_model.layers)
    if not 0 <= assignment.start_layer < assignment.end_layer <= original_layer_count:
        raise PlanningError("planned layer range does not match runtime model")

    pipeline_model.pipeline_rank = rank
    pipeline_model.pipeline_size = size
    pipeline_model.start_idx = assignment.start_layer
    pipeline_model.end_idx = assignment.end_layer
    pipeline_model.layers = pipeline_model.layers[: assignment.end_layer]
    pipeline_model.layers[: assignment.start_layer] = [None] * assignment.start_layer


@contextmanager
def install_unequal_pipeline_plan(
    assignments: Sequence[PipelineAssignment],
) -> Iterator[None]:
    """Temporarily make MLX-LM's pipeline hook honor explicit layer ranges."""

    try:
        from mlx_lm.models.pipeline import PipelineMixin
    except ImportError as exc:
        raise PlanningError("mlx-lm is required to install a pipeline plan") from exc

    original = PipelineMixin.pipeline

    def planned_pipeline(pipeline_model: Any, group: Any) -> None:
        apply_pipeline_assignment(pipeline_model, group, assignments)

    # The worker's pre-load memory guard looks for this exact contract. A
    # model-specific ``pipeline`` method that merely exists is not enough:
    # DeepSeek-V3.2 has one and still computes an unpinned even split.
    planned_pipeline._omlx_honors_pipeline_assignment = True
    PipelineMixin.pipeline = planned_pipeline
    try:
        yield
    finally:
        PipelineMixin.pipeline = original


def format_shard_plan(plan: ShardPlan) -> str:
    """Render a compact human-readable unequal-memory plan."""

    gib = 1024**3
    lines = [
        "oMLX unequal pipeline plan",
        f"Plan:        {plan.plan_hash[:16]}",
        (
            f"Model:       {plan.model.total_weight_bytes / gib:.2f} GiB weights / "
            f"{plan.model.layer_count} layers"
        ),
        (
            f"Cluster:     {plan.cluster_capacity_bytes / gib:.2f} GiB capacity / "
            f"{plan.cluster_resident_weight_bytes / gib:.2f} GiB resident weights"
        ),
    ]
    for item in plan.assignments:
        lines.append(
            f"Rank {item.rank}:       {item.node_id} layers "
            f"[{item.start_layer}, {item.end_layer}) "
            f"({item.layer_count} layers, "
            f"{item.planned_weight_bytes / gib:.2f} GiB weights, "
            f"{item.headroom_bytes / gib:.2f} GiB headroom)"
        )
    return "\n".join(lines)


def _kv_bytes_for_stage(
    model: ModelLayout,
    layer_count: int,
    context_tokens: int,
    tensor_parallel_size: int = 1,
) -> int:
    """KV bytes a node holds for its layers at the planned context length.

    KV is per layer, so a node reserves in proportion to the layers it carries.
    Under tensor parallelism the heads of each layer are split across the
    group, so each member holds its share — except an MLA latent cache, which
    is not per-head and stays whole on every member.
    """

    if model.kv_bytes_per_token_per_layer <= 0 or context_tokens <= 0:
        return 0
    total = model.kv_bytes_per_token_per_layer * layer_count * context_tokens
    if model.kv_replicated_across_tp:
        return total
    return total // max(1, tensor_parallel_size)


def _kv_bytes_per_token_for_stage(
    model: ModelLayout,
    layer_count: int,
    tensor_parallel_size: int = 1,
) -> int:
    """What one more token of context costs this node."""

    if model.kv_bytes_per_token_per_layer <= 0:
        return 0
    per_token = model.kv_bytes_per_token_per_layer * layer_count
    if model.kv_replicated_across_tp:
        return per_token
    return per_token // max(1, tensor_parallel_size)


def _max_context_for_stage(
    model: ModelLayout,
    node: NodeBudget,
    *,
    layer_count: int,
    weight_bytes: int,
    tensor_parallel_size: int = 1,
) -> int:
    """Longest context this node could hold once its weights are resident.

    Uses the node's whole usable memory, not the ceiling the split control
    pinned: capping the *weights* on a Mac is precisely how its remaining
    memory is freed for cache, so the cap must not also limit the answer.

    Returns 0 when the model's KV shape is unknown, which callers read as
    "not known" rather than "unlimited".
    """

    per_token = _kv_bytes_per_token_for_stage(
        model, layer_count, tensor_parallel_size
    )
    if per_token <= 0:
        return 0
    spare = node.usable_bytes - weight_bytes
    return max(0, spare // per_token)


def _tp_stage_budget(
    group: Sequence[NodeBudget],
    stage: int,
    *,
    fixed_weight_bytes: int,
    tensor_parallel_size: int,
) -> NodeBudget:
    """One synthetic budget standing in for a whole tensor-parallel stage.

    ``_partition_layers`` reasons about one node per stage. Under TP a stage is
    ``tensor_parallel_size`` nodes that each hold the replicated fixed weights
    plus a 1/N share of the stage's layers, so the stage can carry
    ``N * (weakest_usable - fixed)`` layer bytes before its weakest member runs
    out.

    Compute rates are scaled by N because the members work on their shards in
    parallel. Every stage shares the same N here, so this does not skew the
    relative split — it only keeps the absolute predicted seconds honest. The
    per-layer all-reduce TP adds is *not* yet modelled, so predictions are
    optimistic for TP stages.
    """

    weakest = min(group, key=lambda node: node.usable_bytes)
    stage_layer_capacity = tensor_parallel_size * (
        weakest.usable_bytes - fixed_weight_bytes
    )

    performance = None
    if all(node.performance is not None for node in group):
        slowest = min(
            group,
            key=lambda node: node.performance.decode_weight_bytes_per_second,
        ).performance
        performance = replace(
            slowest,
            node_id=f"tp-stage-{stage}",
            rank=stage,
            decode_weight_bytes_per_second=(
                slowest.decode_weight_bytes_per_second * tensor_parallel_size
            ),
            prefill_weight_bytes_per_second=(
                slowest.prefill_weight_bytes_per_second * tensor_parallel_size
            ),
        )

    return NodeBudget(
        node_id=f"tp-stage-{stage}",
        capacity_bytes=fixed_weight_bytes + max(stage_layer_capacity, 1),
        reserve_bytes=0,
        rank=stage,
        performance=performance,
    )


def plan_hybrid(
    model: ModelLayout,
    nodes: Sequence[NodeBudget],
    *,
    tensor_parallel_size: int = 1,
    workload_profile: ExecutionProfileName = "balanced",
    microbatch_size: int = 1,
    context_tokens: int = _DEFAULT_CONTEXT_TOKENS,
) -> ShardPlan:
    """Plan hybrid pipeline + tensor parallelism across nodes.

    Combines contiguous pipeline stages with tensor-parallel weight sharding.
    The world is 2D: pipeline_stages * tensor_parallel_size == len(nodes).

    Rank convention:
        pipeline_stage = rank // tensor_parallel_size
        tp_rank        = rank %  tensor_parallel_size

    TP groups are contiguous in rank order so they can be placed on the fastest
    link. Pipeline boundaries carry the slow-axis traffic.
    """

    if not nodes:
        raise ValueError("at least one node is required")
    if tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be at least 1")
    if len(nodes) % tensor_parallel_size != 0:
        raise PlanningError(
            f"world size {len(nodes)} is not divisible by tensor_parallel_size "
            f"{tensor_parallel_size}"
        )
    pipeline_stages = len(nodes) // tensor_parallel_size
    if pipeline_stages < 1:
        raise PlanningError("pipeline_stages must be at least 1")

    if tensor_parallel_size == 1:
        # There is no hybrid axis in this case. Delegating is more than a
        # shortcut: `_tp_stage_budget` normalises each stage to its usable
        # capacity for real TP groups, while the ordinary pipeline planner
        # scores against the Mac's actual capacity and reserve. Running the TP
        # abstraction at width one therefore produced a different layer cut
        # in autoconfigure than /deployments recomputed moments later. The
        # approval guard correctly refused that drift.
        pipeline = plan_unequal_pipeline(
            model,
            nodes,
            workload_profile=workload_profile,
            microbatch_size=microbatch_size,
            context_tokens=context_tokens,
        )
        return replace(
            pipeline,
            tensor_parallel_size=1,
            pipeline_stages=pipeline_stages,
        )

    # Refuse impossible tensor parallelism here rather than after every rank
    # has loaded its weights. shard() does n_heads //= N and n_kv_heads //= N,
    # and most mlx-lm architectures do not implement shard() at all.
    if tensor_parallel_size > 1:
        incompatible = tuple(
            value
            for value in model.tensor_parallel_divisors
            if value % tensor_parallel_size
        )
        if incompatible:
            raise PlanningError(
                "tensor-parallel architecture dimensions "
                f"{incompatible} are not divisible by tensor_parallel_size "
                f"({tensor_parallel_size})"
            )

    # Validate ranks
    ranks = [node.rank for node in nodes]
    if len(set(ranks)) != len(ranks) or set(ranks) != set(range(len(nodes))):
        raise ValueError("node ranks must be unique and contiguous from zero")
    node_ids = [node.node_id for node in nodes]
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("node IDs must be unique")

    # Validate fixed weights fit every node
    for node in nodes:
        if model.fixed_weight_bytes > node.usable_bytes:
            raise PlanningError(
                f"replicated fixed weights do not fit node {node.node_id}"
            )

    # Group ranks into stages. rank = stage * tp_size + tp_rank, so consecutive
    # ranks form one tensor-parallel group and every member of that group must
    # end up with the SAME layer range — they hold one stage's layers between
    # them, split within each layer by shard_linear.
    ordered_nodes = sorted(nodes, key=lambda item: item.rank)
    stage_groups = [
        tuple(ordered_nodes[stage * tensor_parallel_size : (stage + 1) * tensor_parallel_size])
        for stage in range(pipeline_stages)
    ]

    # MLX-LM sends activations from the highest rank (early layers) down to rank
    # zero, so partition stages in reverse stage order, matching plan_pipeline.
    partition_order = tuple(reversed(range(pipeline_stages)))
    stage_budgets = tuple(
        _tp_stage_budget(
            stage_groups[stage],
            stage,
            fixed_weight_bytes=model.fixed_weight_bytes,
            tensor_parallel_size=tensor_parallel_size,
        )
        for stage in partition_order
    )
    performance_aware = all(node.performance is not None for node in ordered_nodes)
    # ``stage_budgets`` represent aggregate TP capacity after each member's
    # replicated fixed weights. A layer therefore contributes its full weights
    # plus its full-head KV bytes here; both are divided across TP members by
    # the same degree when assignments are materialised below. A replicated
    # MLA cache is the exception: every member pays it whole, so at the
    # aggregate level one layer costs the group N caches.
    kv_bytes_per_layer = _kv_bytes_for_stage(model, 1, context_tokens)
    if model.kv_replicated_across_tp:
        kv_bytes_per_layer *= tensor_parallel_size
    try:
        ranges = _partition_layers(
            model.layer_weight_bytes,
            stage_budgets,
            fixed_weight_bytes=model.fixed_weight_bytes,
            layer_resident_sizes=tuple(
                weight_bytes + kv_bytes_per_layer
                for weight_bytes in model.layer_weight_bytes
            ),
            activation_bytes_per_token=model.activation_bytes_per_token,
            workload_profile=workload_profile,
            microbatch_size=microbatch_size,
        )
    except PlanningError as exc:
        if kv_bytes_per_layer:
            raise PlanningError(
                f"model weights and KV cache for {context_tokens} tokens do not "
                f"fit the supplied per-node budgets: {exc}"
            ) from exc
        raise
    stage_ranges = dict(zip(partition_order, ranges))

    assignments: list[PipelineAssignment] = []
    for stage, group in enumerate(stage_groups):
        start, end = stage_ranges[stage]
        stage_layer_bytes = sum(model.layer_weight_bytes[start:end])
        # The stage's layer weights are split across its TP members; the first
        # `remainder` ranks carry one extra byte so the parts sum exactly.
        per_member = stage_layer_bytes // tensor_parallel_size
        remainder = stage_layer_bytes % tensor_parallel_size
        kv_bytes = _kv_bytes_for_stage(
            model, end - start, context_tokens, tensor_parallel_size
        )
        for tp_rank, node in enumerate(group):
            held_layer_bytes = per_member + (1 if tp_rank < remainder else 0)
            planned = model.fixed_weight_bytes + held_layer_bytes + kv_bytes
            if planned > node.usable_bytes:
                raise PlanningError(
                    f"hybrid shard does not fit node {node.node_id}: "
                    f"{planned} > {node.usable_bytes}"
                )
            compute_seconds, send_seconds, stage_seconds = _predict_stage_seconds(
                held_layer_bytes,
                node,
                activation_bytes_per_token=model.activation_bytes_per_token,
                workload_profile=workload_profile,
                microbatch_size=microbatch_size,
            )
            assignments.append(
                PipelineAssignment(
                    node_id=node.node_id,
                    rank=node.rank,
                    start_layer=start,
                    end_layer=end,
                    layer_weight_bytes=held_layer_bytes,
                    fixed_weight_bytes=model.fixed_weight_bytes,
                    reserve_bytes=node.reserve_bytes,
                    capacity_bytes=node.capacity_bytes,
                    manual_memory_limit=node.manual_memory_limit,
                    role=node.role,
                    memory_guard_tier=node.memory_guard_tier,
                    tensor_parallel_rank=tp_rank,
                    tensor_parallel_size=tensor_parallel_size,
                    kv_cache_bytes=kv_bytes,
                    kv_bytes_per_token=_kv_bytes_per_token_for_stage(
                        model, end - start, tensor_parallel_size
                    ),
                    max_context_tokens=_max_context_for_stage(
                        model,
                        node,
                        layer_count=end - start,
                        weight_bytes=model.fixed_weight_bytes + held_layer_bytes,
                        tensor_parallel_size=tensor_parallel_size,
                    ),
                    sharded_weight_bytes=(
                        held_layer_bytes if tensor_parallel_size > 1 else 0
                    ),
                    predicted_compute_seconds=(
                        compute_seconds if performance_aware else None
                    ),
                    predicted_send_seconds=(
                        send_seconds if performance_aware else None
                    ),
                    predicted_stage_seconds=(
                        stage_seconds if performance_aware else None
                    ),
                )
            )

    # Sort by rank for consistent ordering
    assignments.sort(key=lambda a: a.rank)

    # Build performance profiles if available
    profiles = tuple(
        node.performance for node in ordered_nodes
        if node.performance is not None
    ) if performance_aware else ()

    # Compute plan hash covering all assignment fields
    hash_payload = {
        "model": model.to_dict(),
        "context_tokens": context_tokens,
        "assignments": [a.to_dict() for a in assignments],
        "tensor_parallel_size": tensor_parallel_size,
        "pipeline_stages": pipeline_stages,
        "workload_profile": workload_profile,
        "microbatch_size": microbatch_size,
    }
    plan_hash = hashlib.sha256(
        json.dumps(hash_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    return ShardPlan(
        model=model,
        assignments=tuple(assignments),
        plan_hash=plan_hash,
        optimization="performance" if performance_aware else "memory",
        workload_profile=workload_profile,
        performance_profiles=profiles,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_stages=pipeline_stages,
        target_context_tokens=context_tokens,
    )
