# SPDX-License-Identifier: Apache-2.0
"""Fixed KV-cache memory planning without importing MLX.

The prelaunch planner reads a local ``config.json`` and describes the exact
cache tensors that OMLX's fixed-cache allocator is expected to reserve.  It
does not claim that config metadata is a runtime measurement.  A separate
duck-typed probe helper can describe concrete arrays after a model cache has
been instantiated and materialized.
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

_CACHE_STEP = 256
_SCHEMA_VERSION = 1
_SYSCTL = "/usr/sbin/sysctl"
_CONTEXT_LIMIT_KEYS = (
    "max_position_embeddings",
    "max_seq_len",
    "max_seq_length",
    "max_sequence_length",
    "model_max_length",
    "context_length",
    "seq_length",
    "n_positions",
)


class FixedKVPlanningError(ValueError):
    """The model metadata cannot support an honest fixed-cache plan."""


def validate_fixed_kv_runtime_features(
    settings: object | None,
    *,
    distributed: bool = False,
) -> None:
    """Reject decode modes whose extra caches are outside the fixed plan.

    These paths allocate cache state independently of the target model's
    ``make_prompt_cache`` tree. Until each has a matching descriptor and
    fixed-pool adapter, allowing it would make the launch total incomplete and
    permit memory growth after the main pool had been committed.
    """

    def value(name: str) -> Any:
        if settings is None:
            return None
        if isinstance(settings, Mapping):
            return settings.get(name)
        return getattr(settings, name, None)

    if distributed:
        raise FixedKVPlanningError(
            "Fixed KV allocation is not yet supported for distributed models: "
            "each rank must commit its own cache pool. Use a local engine for "
            "this model and launch again."
        )
    if bool(value("dflash_enabled")):
        raise FixedKVPlanningError(
            "Fixed KV allocation is not compatible with DFlash. Disable DFlash "
            "for this model and launch again."
        )
    if bool(value("turboquant_kv_enabled")):
        raise FixedKVPlanningError(
            "Fixed KV allocation is not compatible with TurboQuant KV. Disable "
            "TurboQuant for this model and launch again."
        )
    if bool(value("specprefill_enabled")) and value("specprefill_draft_model"):
        raise FixedKVPlanningError(
            "Fixed KV allocation is not yet compatible with SpecPrefill: its "
            "draft-model weights and prompt cache are outside the committed "
            "memory plan. Disable SpecPrefill for this model and launch again."
        )
    if bool(value("vlm_mtp_enabled")) and value("vlm_mtp_draft_model"):
        raise FixedKVPlanningError(
            "Fixed KV allocation is not yet compatible with VLM MTP: its external "
            "drafter weights and cache are outside the committed memory plan. "
            "Disable VLM MTP for this model and launch again."
        )


@dataclass(frozen=True)
class CacheTensorDescriptor:
    """One physical tensor in a per-session cache allocation manifest."""

    path: str
    cache_kind: str
    role: str
    shape: tuple[int, ...]
    dtype: str
    dtype_bytes: int
    nbytes: int
    logical_tokens: int | None
    physical_tokens: int | None
    capacity_kind: str
    provenance: str
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a stable, JSON-compatible representation."""

        return {
            "schema_version": _SCHEMA_VERSION,
            "path": self.path,
            "cache_kind": self.cache_kind,
            "role": self.role,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "dtype_bytes": self.dtype_bytes,
            "bytes": self.nbytes,
            "logical_tokens": self.logical_tokens,
            "physical_tokens": self.physical_tokens,
            "capacity_kind": self.capacity_kind,
            "provenance": self.provenance,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ModelMemoryPlan:
    """Prelaunch model memory estimate and fixed-cache session plan."""

    model_path: str
    model_type: str
    model_context_limit: int | None
    context_window: int
    weights_bytes: int
    other_fixed_bytes: int
    cache_tensors: tuple[CacheTensorDescriptor, ...]
    per_session_kv_bytes: int
    pool_scratch_bytes: int
    requested_session_slots: int
    reserved_session_slots: int
    max_feasible_session_slots: int | None
    cache_layout_max_session_slots: int | None
    fixed_kv_cache_bytes: int
    estimated_total_bytes: int
    detected_unified_memory_bytes: int | None
    detected_unified_memory_source: str | None
    available_memory_bytes: int | None
    available_memory_source: str | None
    memory_ceiling_bytes: int | None
    memory_ceiling_source: str | None
    binding_memory_bytes: int | None
    binding_memory_source: str | None
    remaining_memory_bytes: int | None
    fits: bool | None
    requested_configuration_fits: bool | None
    configured_concurrency_capped: bool
    fit_reason: str | None
    provenance: tuple[str, ...]

    @property
    def native_mtp_kv_bytes_per_session(self) -> int:
        """Return per-session cache bytes owned by a native MTP head."""

        return sum(
            tensor.nbytes
            for tensor in self.cache_tensors
            if tensor.path.startswith(("mtp.", "mtp_clone."))
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a stable, JSON-compatible representation."""

        return {
            "schema_version": _SCHEMA_VERSION,
            "model_path": self.model_path,
            "model_type": self.model_type,
            "model_context_limit": self.model_context_limit,
            "context_window": self.context_window,
            "weights_bytes": self.weights_bytes,
            "other_fixed_bytes": self.other_fixed_bytes,
            "per_session_kv_bytes": self.per_session_kv_bytes,
            "pool_scratch_bytes": self.pool_scratch_bytes,
            "requested_session_slots": self.requested_session_slots,
            "reserved_session_slots": self.reserved_session_slots,
            "max_feasible_session_slots": self.max_feasible_session_slots,
            "cache_layout_max_session_slots": self.cache_layout_max_session_slots,
            "fixed_kv_cache_bytes": self.fixed_kv_cache_bytes,
            "estimated_total_bytes": self.estimated_total_bytes,
            "detected_unified_memory_bytes": self.detected_unified_memory_bytes,
            "unified_memory_bytes": self.detected_unified_memory_bytes,
            "detected_unified_memory_source": self.detected_unified_memory_source,
            "available_memory_bytes": self.available_memory_bytes,
            "available_memory_source": self.available_memory_source,
            "memory_ceiling_bytes": self.memory_ceiling_bytes,
            "memory_ceiling_source": self.memory_ceiling_source,
            "binding_memory_bytes": self.binding_memory_bytes,
            "binding_memory_source": self.binding_memory_source,
            "remaining_memory_bytes": self.remaining_memory_bytes,
            "projected_remaining_bytes": self.remaining_memory_bytes,
            "remaining_memory_source": (
                f"estimated from {self.binding_memory_source}"
                if self.remaining_memory_bytes is not None
                and self.binding_memory_source is not None
                else None
            ),
            "fits": self.fits,
            "requested_configuration_fits": self.requested_configuration_fits,
            "configured_concurrency_capped": self.configured_concurrency_capped,
            "fit_reason": self.fit_reason,
            "cache_tensors": [tensor.to_dict() for tensor in self.cache_tensors],
            "native_mtp_kv_bytes_per_session": self.native_mtp_kv_bytes_per_session,
            "components": (
                (
                    [
                        {
                            "name": "native MTP cache per session",
                            "bytes": self.native_mtp_kv_bytes_per_session,
                            "provenance": "config-driven native MTP cache manifest",
                        }
                    ]
                    if self.native_mtp_kv_bytes_per_session
                    else []
                )
                + ([
                    {
                        "name": "fixed KV workspace and rollback storage",
                        "bytes": self.pool_scratch_bytes,
                        "provenance": (
                            "materialized row-compaction and architecture-specific "
                            "rollback workspace"
                        ),
                    }
                ]
                if self.pool_scratch_bytes
                else [])
            ),
            "lifecycle": "estimated",
            "provenance": list(self.provenance),
        }


@dataclass(frozen=True)
class _SystemMemory:
    total: int | None
    total_source: str | None
    available: int | None
    available_source: str | None


def _positive_int(config: Mapping[str, Any], key: str, default: int = 0) -> int:
    value = config.get(key, default)
    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _first_positive(config: Mapping[str, Any], *keys: str) -> int:
    """Return the first positive integer exposed under an architecture alias."""

    for key in keys:
        value = _positive_int(config, key)
        if value:
            return value
    return 0


def _context_limit(config: Mapping[str, Any]) -> int:
    for key in _CONTEXT_LIMIT_KEYS:
        value = _positive_int(config, key)
        if value:
            return value
    return 0


def _round_cache_tokens(tokens: int) -> int:
    return ((tokens + _CACHE_STEP - 1) // _CACHE_STEP) * _CACHE_STEP


def _shape_bytes(shape: Sequence[int], dtype_bytes: int) -> int:
    return math.prod(shape) * dtype_bytes


def _dtype(config: Mapping[str, Any]) -> tuple[str, int, str]:
    raw = config.get("kv_cache_dtype")
    source = "config.kv_cache_dtype"
    if raw is None:
        raw = config.get("torch_dtype", config.get("dtype"))
        source = "config compute dtype"
    if raw is None or str(raw).lower() in {"auto", "none"}:
        return "float16", 2, "assumed float16 cache dtype"

    name = str(raw).lower().replace("torch.", "")
    sizes = {
        "float16": 2,
        "fp16": 2,
        "half": 2,
        "bfloat16": 2,
        "bf16": 2,
        "float32": 4,
        "fp32": 4,
    }
    if name not in sizes:
        raise FixedKVPlanningError(
            f"Unsupported KV-cache dtype {raw!r}; provide a supported explicit "
            "dtype before enabling fixed cache allocation."
        )
    canonical = {
        "fp16": "float16",
        "half": "float16",
        "bf16": "bfloat16",
        "fp32": "float32",
    }.get(name, name)
    return canonical, sizes[name], source


def _load_config(model_path: Path) -> tuple[dict[str, Any], str]:
    path = model_path / "config.json"
    if not path.is_file():
        raise FixedKVPlanningError(f"Missing model config: {path}")
    try:
        root = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FixedKVPlanningError(f"Cannot read model config {path}: {exc}") from exc
    if not isinstance(root, dict):
        raise FixedKVPlanningError(f"Model config must be a JSON object: {path}")

    candidates: list[tuple[str, dict[str, Any]]] = [("config", root)]
    for key in ("text_config", "language_config", "llm_config"):
        nested = root.get(key)
        if isinstance(nested, dict):
            candidates.append((f"config.{key}", nested))
    source, selected = max(
        candidates,
        key=lambda item: sum(
            key in item[1]
            for key in (
                "num_hidden_layers",
                "num_attention_heads",
                "kv_lora_rank",
                "layer_types",
            )
        ),
    )
    config = dict(root)
    config.update(selected)
    if isinstance(root.get("model_type"), str):
        config["_root_model_type"] = root["model_type"]
    if "model_type" not in selected and isinstance(root.get("model_type"), str):
        config["model_type"] = root["model_type"]
    return config, source


def _descriptor(
    *,
    path: str,
    cache_kind: str,
    role: str,
    shape: tuple[int, ...],
    dtype: str,
    dtype_bytes: int,
    logical_tokens: int | None,
    physical_tokens: int | None,
    capacity_kind: str,
    provenance: str,
    notes: str | None = None,
) -> CacheTensorDescriptor:
    return CacheTensorDescriptor(
        path=path,
        cache_kind=cache_kind,
        role=role,
        shape=shape,
        dtype=dtype,
        dtype_bytes=dtype_bytes,
        nbytes=_shape_bytes(shape, dtype_bytes),
        logical_tokens=logical_tokens,
        physical_tokens=physical_tokens,
        capacity_kind=capacity_kind,
        provenance=provenance,
        notes=notes,
    )


def _kv_pair(
    layer: int,
    *,
    heads: int,
    key_dim: int,
    value_dim: int,
    logical_tokens: int,
    physical_tokens: int,
    dtype: str,
    dtype_bytes: int,
    cache_kind: str,
    capacity_kind: str,
    provenance: str,
    prefix: str = "",
) -> list[CacheTensorDescriptor]:
    base = f"layers.{layer}.{prefix}" if prefix else f"layers.{layer}."
    return [
        _descriptor(
            path=f"{base}keys",
            cache_kind=cache_kind,
            role="keys",
            shape=(1, heads, physical_tokens, key_dim),
            dtype=dtype,
            dtype_bytes=dtype_bytes,
            logical_tokens=logical_tokens,
            physical_tokens=physical_tokens,
            capacity_kind=capacity_kind,
            provenance=provenance,
        ),
        _descriptor(
            path=f"{base}values",
            cache_kind=cache_kind,
            role="values",
            shape=(1, heads, physical_tokens, value_dim),
            dtype=dtype,
            dtype_bytes=dtype_bytes,
            logical_tokens=logical_tokens,
            physical_tokens=physical_tokens,
            capacity_kind=capacity_kind,
            provenance=provenance,
            notes="Zero-width values are intentional." if value_dim == 0 else None,
        ),
    ]


def _standard_dimensions(config: Mapping[str, Any]) -> tuple[int, int, int]:
    attention_heads = _first_positive(
        config, "num_attention_heads", "n_head", "n_heads", "num_heads"
    )
    kv_heads = _first_positive(
        config,
        "num_key_value_heads",
        "num_kv_heads",
        "n_kv_heads",
        "num_kv_head",
    ) or attention_heads
    head_dim = _first_positive(config, "head_dim", "hidden_size_per_head")
    hidden_size = _first_positive(
        config, "hidden_size", "hidden_dim", "n_embd", "d_model", "dim"
    )
    if head_dim <= 0 and hidden_size > 0 and attention_heads > 0:
        if hidden_size % attention_heads:
            raise FixedKVPlanningError(
                "hidden_size is not divisible by num_attention_heads; the cache "
                "head dimension cannot be inferred safely."
            )
        head_dim = hidden_size // attention_heads
    if attention_heads <= 0 or kv_heads <= 0 or head_dim <= 0:
        raise FixedKVPlanningError(
            "config.json does not provide enough attention shape metadata "
            "(num_attention_heads, num_key_value_heads, and head_dim or "
            "hidden_size)."
        )
    return attention_heads, kv_heads, head_dim


def _layer_capacities(
    config: Mapping[str, Any],
    model_type: str,
    layers: int,
    context: int,
    prefill_step_size: int,
) -> list[tuple[str, int]]:
    """Return ``(kind, physical_tokens)`` for generic K/V cache layers."""

    linear = ("linear", _round_cache_tokens(context))
    layer_types = config.get("layer_types")
    sliding_window = _positive_int(config, "sliding_window")

    if isinstance(layer_types, list):
        if len(layer_types) != layers:
            raise FixedKVPlanningError(
                "layer_types length does not match num_hidden_layers; fixed cache "
                "allocation cannot choose the correct per-layer layout."
            )
        capacities: list[tuple[str, int]] = []
        for index, raw in enumerate(layer_types):
            kind = str(raw).lower()
            if any(token in kind for token in ("linear", "gdn", "mamba", "ssm")):
                raise FixedKVPlanningError(
                    f"Layer {index} uses recurrent layout {raw!r}. Its fixed state "
                    "shape requires a live cache probe before launch."
                )
            if any(token in kind for token in ("sliding", "window", "local")):
                if sliding_window <= 0:
                    raise FixedKVPlanningError(
                        f"Layer {index} is sliding-window attention but "
                        "sliding_window is missing or invalid."
                    )
                capacities.append(
                    ("rotating", min(sliding_window, _round_cache_tokens(context)))
                )
            elif any(token in kind for token in ("full", "global", "attention")):
                capacities.append(linear)
            else:
                raise FixedKVPlanningError(
                    f"Unknown layer_types entry {raw!r} at layer {index}; fixed "
                    "cache planning refuses to guess its storage layout."
                )
        return capacities

    hybrid_pattern = config.get("hybrid_layer_pattern")
    if isinstance(hybrid_pattern, list):
        if len(hybrid_pattern) != layers:
            raise FixedKVPlanningError(
                "hybrid_layer_pattern length does not match num_hidden_layers."
            )
        window = _positive_int(
            config, "sliding_window_size", _positive_int(config, "sliding_window")
        )
        if window <= 0:
            raise FixedKVPlanningError(
                "hybrid_layer_pattern is set but its sliding-window size is missing."
            )
        return [
            ("rotating", min(window, _round_cache_tokens(context)))
            if int(value) == 1
            else linear
            for value in hybrid_pattern
        ]

    raw_pattern = config.get("sliding_window_pattern")
    if isinstance(raw_pattern, str):
        pattern_text = raw_pattern.upper()
        if not pattern_text or any(char not in "LGFS" for char in pattern_text):
            raise FixedKVPlanningError(
                "sliding_window_pattern must contain only local/global entries."
            )
        if sliding_window <= 0:
            raise FixedKVPlanningError(
                "sliding_window_pattern is set but sliding_window is missing."
            )
        return [
            ("rotating", min(sliding_window, _round_cache_tokens(context)))
            if pattern_text[index % len(pattern_text)] in "LS"
            else linear
            for index in range(layers)
        ]

    pattern = _positive_int(config, "sliding_window_pattern")
    if pattern:
        if sliding_window <= 0:
            raise FixedKVPlanningError(
                "sliding_window_pattern is set but sliding_window is missing."
            )
        return [
            linear
            if index % pattern == pattern - 1
            else ("rotating", min(sliding_window, _round_cache_tokens(context)))
            for index in range(layers)
        ]

    if model_type.startswith("olmo3"):
        if sliding_window <= 0:
            raise FixedKVPlanningError("OLMo 3 requires sliding_window metadata.")
        return [
            linear
            if (index + 1) % 4 == 0
            else ("rotating", min(sliding_window, _round_cache_tokens(context)))
            for index in range(layers)
        ]

    if model_type.startswith("gpt_oss"):
        if sliding_window <= 0:
            raise FixedKVPlanningError("GPT-OSS requires sliding_window metadata.")
        return [
            ("rotating", min(sliding_window, _round_cache_tokens(context)))
            if index % 2 == 0
            else linear
            for index in range(layers)
        ]

    sliding_ids = config.get("sliding_window_layers")
    if isinstance(sliding_ids, list):
        if "baichuan_m1" in model_type:
            raise FixedKVPlanningError(
                "Baichuan M1 also stores recurrent ArraysCache state. Use a live "
                "cache probe before enabling fixed allocation."
            )
        if sliding_window <= 0:
            raise FixedKVPlanningError(
                "sliding_window_layers is set but sliding_window is missing."
            )
        ids = {int(value) for value in sliding_ids}
        return [
            ("rotating", min(sliding_window, _round_cache_tokens(context)))
            if index in ids
            else linear
            for index in range(layers)
        ]

    if model_type.startswith("llama4"):
        chunk = _positive_int(config, "attention_chunk_size")
        if chunk <= 0:
            raise FixedKVPlanningError(
                "Llama 4 requires attention_chunk_size for fixed cache planning."
            )
        incoming = min(context, max(1, int(prefill_step_size)))
        chunk_capacity = min(
            _round_cache_tokens(context),
            chunk + _round_cache_tokens(incoming),
        )
        return [
            linear if (index + 1) % 4 == 0 else ("chunked", chunk_capacity)
            for index in range(layers)
        ]

    return [linear] * layers


def _generic_manifest(
    config: Mapping[str, Any],
    model_type: str,
    context: int,
    dtype: str,
    dtype_bytes: int,
    provenance: str,
    prefill_step_size: int,
) -> list[CacheTensorDescriptor]:
    total_layers = _first_positive(
        config, "num_hidden_layers", "n_layer", "n_layers", "num_layers"
    )
    if total_layers <= 0:
        raise FixedKVPlanningError("num_hidden_layers is missing or invalid.")
    _, kv_heads, head_dim = _standard_dimensions(config)

    cache_layers = total_layers
    if model_type.startswith(("gemma3n", "gemma4")):
        shared = int(config.get("num_kv_shared_layers", 0) or 0)
        if shared < 0 or shared > total_layers:
            raise FixedKVPlanningError(
                "num_kv_shared_layers is outside the layer range."
            )
        cache_layers -= shared

    capacities = _layer_capacities(
        config,
        model_type,
        total_layers,
        context,
        prefill_step_size,
    )
    capacities = capacities[:cache_layers]
    tensors: list[CacheTensorDescriptor] = []
    for layer, (kind, physical) in enumerate(capacities):
        logical = context if kind in {"linear", "chunked"} else min(context, physical)
        tensors.extend(
            _kv_pair(
                layer,
                heads=kv_heads,
                key_dim=head_dim,
                value_dim=head_dim,
                logical_tokens=logical,
                physical_tokens=physical,
                dtype=dtype,
                dtype_bytes=dtype_bytes,
                cache_kind={
                    "linear": "KVCache",
                    "rotating": "RotatingKVCache",
                    "chunked": "ChunkedKVCache",
                }[kind],
                capacity_kind=kind,
                provenance=provenance,
            )
        )
    return tensors


def _qwen_gated_delta_manifest(
    config: Mapping[str, Any],
    context: int,
    dtype: str,
    dtype_bytes: int,
    provenance: str,
) -> list[CacheTensorDescriptor]:
    """Describe Qwen3-Next/Qwen3.5 full-attention and GatedDeltaNet state."""

    layers = _positive_int(config, "num_hidden_layers")
    interval = _positive_int(config, "full_attention_interval", 4)
    key_heads = _positive_int(config, "linear_num_key_heads")
    value_heads = _positive_int(config, "linear_num_value_heads")
    key_dim = _positive_int(config, "linear_key_head_dim")
    value_dim = _positive_int(config, "linear_value_head_dim")
    conv_kernel = _positive_int(config, "linear_conv_kernel_dim")
    if not all(
        (layers, interval, key_heads, value_heads, key_dim, value_dim, conv_kernel)
    ):
        raise FixedKVPlanningError(
            "Qwen gated-delta cache planning requires num_hidden_layers, "
            "full_attention_interval, linear_num_key_heads, "
            "linear_num_value_heads, linear_key_head_dim, "
            "linear_value_head_dim, and linear_conv_kernel_dim."
        )
    if value_heads % key_heads:
        raise FixedKVPlanningError(
            "linear_num_value_heads must be divisible by linear_num_key_heads."
        )

    _, kv_heads, attention_head_dim = _standard_dimensions(config)
    physical = _round_cache_tokens(context)
    conv_dim = 2 * key_heads * key_dim + value_heads * value_dim
    tensors: list[CacheTensorDescriptor] = []
    for layer in range(layers):
        if (layer + 1) % interval == 0:
            tensors.extend(
                _kv_pair(
                    layer,
                    heads=kv_heads,
                    key_dim=attention_head_dim,
                    value_dim=attention_head_dim,
                    logical_tokens=context,
                    physical_tokens=physical,
                    dtype=dtype,
                    dtype_bytes=dtype_bytes,
                    cache_kind="KVCache",
                    capacity_kind="linear",
                    provenance=provenance,
                )
            )
            continue

        tensors.append(
            _descriptor(
                path=f"layers.{layer}.state.0",
                cache_kind="ArraysCache",
                role="state_0",
                shape=(1, conv_kernel - 1, conv_dim),
                dtype=dtype,
                dtype_bytes=dtype_bytes,
                logical_tokens=None,
                physical_tokens=None,
                capacity_kind="fixed_state",
                provenance=(
                    f"{provenance}; mlx-lm Qwen GatedDeltaNet conv state"
                ),
            )
        )
        tensors.append(
            _descriptor(
                path=f"layers.{layer}.state.1",
                cache_kind="ArraysCache",
                role="state_1",
                shape=(1, value_heads, value_dim, key_dim),
                dtype="float32",
                dtype_bytes=4,
                logical_tokens=None,
                physical_tokens=None,
                capacity_kind="fixed_state",
                provenance=(
                    f"{provenance}; mlx-lm gated_delta_update recurrent state"
                ),
            )
        )
    return tensors


def _nemotron_h_manifest(
    config: Mapping[str, Any],
    context: int,
    dtype: str,
    dtype_bytes: int,
    provenance: str,
) -> list[CacheTensorDescriptor]:
    """Describe Nemotron-H's interleaved Mamba2 and attention caches."""

    layers = _positive_int(config, "num_hidden_layers")
    attention_heads = _positive_int(config, "num_attention_heads")
    kv_heads = _positive_int(config, "num_key_value_heads", attention_heads)
    head_dim = _positive_int(config, "head_dim")
    hidden = _positive_int(config, "hidden_size")
    mamba_heads = _positive_int(config, "mamba_num_heads")
    mamba_head_dim = _positive_int(config, "mamba_head_dim")
    state_size = _positive_int(config, "ssm_state_size")
    groups = _positive_int(config, "n_groups")
    conv_kernel = _positive_int(config, "conv_kernel")
    if head_dim <= 0 and hidden and attention_heads and hidden % attention_heads == 0:
        head_dim = hidden // attention_heads
    pattern = config.get("hybrid_override_pattern")
    if pattern is None:
        block_types = config.get("layers_block_type")
        mapping = {"mamba": "M", "attention": "*", "moe": "E", "mlp": "-"}
        if isinstance(block_types, list):
            try:
                pattern = [mapping[str(value).lower()] for value in block_types]
            except KeyError as exc:
                raise FixedKVPlanningError(
                    f"Unknown Nemotron-H block type {exc.args[0]!r}."
                ) from exc
    if (
        not all(
            (
                layers,
                attention_heads,
                kv_heads,
                head_dim,
                mamba_heads,
                mamba_head_dim,
                state_size,
                groups,
                conv_kernel,
            )
        )
        or not isinstance(pattern, list)
        or len(pattern) != layers
    ):
        raise FixedKVPlanningError(
            "Nemotron-H cache dimensions or layer pattern are missing or invalid."
        )

    state_dtype = str(config.get("mamba_ssm_cache_dtype", "float32")).lower()
    state_sizes = {"float32": 4, "fp32": 4, "bfloat16": 2, "bf16": 2}
    if state_dtype not in state_sizes:
        raise FixedKVPlanningError(
            f"Unsupported Nemotron-H Mamba state dtype {state_dtype!r}."
        )
    state_dtype = {"fp32": "float32", "bf16": "bfloat16"}.get(
        state_dtype, state_dtype
    )
    conv_dim = mamba_heads * mamba_head_dim + 2 * groups * state_size
    physical = _round_cache_tokens(context)
    tensors: list[CacheTensorDescriptor] = []
    cache_index = 0
    for block in pattern:
        kind = str(block)
        if kind == "M":
            tensors.append(
                _fixed_state_tensor(
                    cache_index,
                    0,
                    (1, conv_kernel - 1, conv_dim),
                    dtype=dtype,
                    dtype_bytes=dtype_bytes,
                    provenance=provenance,
                    notes="Nemotron-H Mamba2 depthwise-convolution history.",
                )
            )
            tensors.append(
                _fixed_state_tensor(
                    cache_index,
                    1,
                    (1, mamba_heads, mamba_head_dim, state_size),
                    dtype=state_dtype,
                    dtype_bytes=state_sizes[state_dtype],
                    provenance=provenance,
                    notes="Nemotron-H Mamba2 recurrent SSM state.",
                )
            )
            cache_index += 1
        elif kind == "*":
            tensors.extend(
                _kv_pair(
                    cache_index,
                    heads=kv_heads,
                    key_dim=head_dim,
                    value_dim=head_dim,
                    logical_tokens=context,
                    physical_tokens=physical,
                    dtype=dtype,
                    dtype_bytes=dtype_bytes,
                    cache_kind="KVCache",
                    capacity_kind="linear",
                    provenance=provenance,
                )
            )
            cache_index += 1
        elif kind not in {"E", "-"}:
            raise FixedKVPlanningError(
                f"Unknown Nemotron-H layer pattern entry {kind!r}."
            )
    if not tensors:
        raise FixedKVPlanningError("Nemotron-H config has no cache-bearing layers.")
    return tensors


def _deepseek_v2_manifest(
    config: Mapping[str, Any],
    context: int,
    dtype: str,
    dtype_bytes: int,
    provenance: str,
) -> list[CacheTensorDescriptor]:
    layers = _positive_int(config, "num_hidden_layers")
    heads = _positive_int(config, "num_attention_heads")
    nope = _positive_int(config, "qk_nope_head_dim")
    rope = _positive_int(config, "qk_rope_head_dim")
    value = _positive_int(config, "v_head_dim")
    if not all((layers, heads, nope, rope, value)):
        raise FixedKVPlanningError(
            "DeepSeek V2 expanded K/V dimensions are incomplete."
        )
    physical = _round_cache_tokens(context)
    tensors: list[CacheTensorDescriptor] = []
    for layer in range(layers):
        tensors.extend(
            _kv_pair(
                layer,
                heads=heads,
                key_dim=nope + rope,
                value_dim=value,
                logical_tokens=context,
                physical_tokens=physical,
                dtype=dtype,
                dtype_bytes=dtype_bytes,
                cache_kind="KVCache",
                capacity_kind="linear",
                provenance=provenance,
            )
        )
    return tensors


def _minimax_m3_manifest(
    config: Mapping[str, Any],
    context: int,
    dtype: str,
    dtype_bytes: int,
    provenance: str,
) -> list[CacheTensorDescriptor]:
    """Describe MiniMax M3 dense K/V plus its sparse-attention side index."""

    layers = _positive_int(config, "num_hidden_layers")
    _, kv_heads, head_dim = _standard_dimensions(config)
    if layers <= 0:
        raise FixedKVPlanningError("MiniMax M3 layer count is missing or invalid.")

    sparse_raw = config.get("sparse_attention_config")
    sparse = sparse_raw if isinstance(sparse_raw, Mapping) else {}
    sparse_config_absent = not isinstance(sparse_raw, Mapping)
    layer_frequency = None
    if isinstance(config.get("layer_types"), list):
        layer_frequency = [
            1 if str(value).lower() == "minimax_m3_sparse" else 0
            for value in config["layer_types"]
        ]
    frequency = sparse.get("sparse_attention_freq")
    disable_frequency = sparse.get("sparse_disable_index_value")
    if sparse_config_absent:
        frequency = layer_frequency
        if frequency is None:
            frequency = [0] * min(3, layers) + [1] * max(0, layers - 3)
        sparse_default = True
    else:
        sparse_default = layer_frequency is not None
        if frequency is None and layer_frequency is not None:
            frequency = layer_frequency
        # TextConfig only lets the disable-index alias imply sparse mode when
        # no explicit sparse_attention_freq was supplied.
        if frequency is None and isinstance(disable_frequency, list):
            frequency = disable_frequency
            sparse_default = True
    use_sparse = bool(sparse.get("use_sparse_attention", sparse_default))
    if frequency is None:
        # A present sparse config with no schedule remains dense even when its
        # explicit use flag is true: runtime has_sparse_index requires a list.
        frequency = [0] * layers
    if not isinstance(frequency, list) or len(frequency) != layers:
        raise FixedKVPlanningError(
            "MiniMax M3 sparse_attention_freq must contain one entry per layer."
        )

    index_dim = _positive_int(
        sparse,
        "sparse_index_dim",
        _positive_int(config, "index_head_dim", 128),
    )
    if use_sparse and index_dim <= 0:
        raise FixedKVPlanningError(
            "MiniMax M3 sparse_index_dim is missing or invalid."
        )

    physical = _round_cache_tokens(context)
    tensors: list[CacheTensorDescriptor] = []
    for layer in range(layers):
        has_index = use_sparse and bool(frequency[layer])
        cache_kind = "MiniMaxM3KVCache" if has_index else "KVCache"
        tensors.extend(
            _kv_pair(
                layer,
                heads=kv_heads,
                key_dim=head_dim,
                value_dim=head_dim,
                logical_tokens=context,
                physical_tokens=physical,
                dtype=dtype,
                dtype_bytes=dtype_bytes,
                cache_kind=cache_kind,
                capacity_kind="linear",
                provenance=provenance,
            )
        )
        if has_index:
            tensors.append(
                _descriptor(
                    path=f"layers.{layer}.index_keys",
                    cache_kind=cache_kind,
                    role="index_keys",
                    shape=(1, 1, physical, index_dim),
                    dtype=dtype,
                    dtype_bytes=dtype_bytes,
                    logical_tokens=context,
                    physical_tokens=physical,
                    capacity_kind="linear",
                    provenance=provenance,
                    notes="MiniMax M3 sparse-attention key index.",
                )
            )
    return tensors


def _ring_sliding_manifest(
    config: Mapping[str, Any],
    context: int,
    dtype: str,
    dtype_bytes: int,
    provenance: str,
) -> list[CacheTensorDescriptor]:
    """Describe Unlimited OCR's prompt-retaining decode-ring cache."""

    layers = _positive_int(config, "num_hidden_layers")
    _, kv_heads, head_dim = _standard_dimensions(config)
    window = _positive_int(
        config,
        "sliding_window_size",
        _positive_int(config, "sliding_window"),
    )
    if layers <= 0 or window <= 0:
        raise FixedKVPlanningError(
            "Unlimited OCR ring-cache dimensions are missing or invalid."
        )
    physical = _round_cache_tokens(context)
    tensors: list[CacheTensorDescriptor] = []
    for layer in range(layers):
        pair = _kv_pair(
            layer,
            heads=kv_heads,
            key_dim=head_dim,
            value_dim=head_dim,
            logical_tokens=context,
            physical_tokens=physical,
            dtype=dtype,
            dtype_bytes=dtype_bytes,
            cache_kind="RingSlidingKVCache",
            capacity_kind="prompt_plus_decode_ring",
            provenance=provenance,
        )
        tensors.extend(
            CacheTensorDescriptor(
                **{
                    **tensor.__dict__,
                    "notes": (
                        f"Retains the prompt plus a {window}-token decode ring; "
                        "the prompt may itself occupy the configured context."
                    ),
                }
            )
            for tensor in pair
        )
    return tensors


def _fixed_state_tensor(
    layer: int,
    index: int,
    shape: tuple[int, ...],
    *,
    dtype: str,
    dtype_bytes: int,
    provenance: str,
    notes: str,
) -> CacheTensorDescriptor:
    return _descriptor(
        path=f"layers.{layer}.state.{index}",
        cache_kind="ArraysCache",
        role=f"state_{index}",
        shape=shape,
        dtype=dtype,
        dtype_bytes=dtype_bytes,
        logical_tokens=None,
        physical_tokens=None,
        capacity_kind="fixed_state",
        provenance=provenance,
        notes=notes,
    )


def _mamba1_state_pair(
    layer: int,
    *,
    intermediate: int,
    state_size: int,
    kernel: int,
    dtype: str,
    dtype_bytes: int,
    provenance: str,
) -> list[CacheTensorDescriptor]:
    if not all((intermediate, state_size, kernel)):
        raise FixedKVPlanningError("Mamba state dimensions are missing or invalid.")
    return [
        _fixed_state_tensor(
            layer,
            0,
            (1, kernel - 1, intermediate),
            dtype=dtype,
            dtype_bytes=dtype_bytes,
            provenance=provenance,
            notes="Mamba depthwise-convolution history.",
        ),
        _fixed_state_tensor(
            layer,
            1,
            (1, intermediate, state_size),
            dtype=dtype,
            dtype_bytes=dtype_bytes,
            provenance=provenance,
            notes="Mamba recurrent SSM state.",
        ),
    ]


def _mamba2_state_pair(
    layer: int,
    *,
    heads: int,
    head_dim: int,
    state_size: int,
    groups: int,
    kernel: int,
    dtype: str,
    dtype_bytes: int,
    provenance: str,
    conv_dim: int | None = None,
) -> list[CacheTensorDescriptor]:
    if not all((heads, head_dim, state_size, groups, kernel)):
        raise FixedKVPlanningError("Mamba2 state dimensions are missing or invalid.")
    convolution_channels = conv_dim or (
        heads * head_dim + 2 * groups * state_size
    )
    return [
        _fixed_state_tensor(
            layer,
            0,
            (1, kernel - 1, convolution_channels),
            dtype=dtype,
            dtype_bytes=dtype_bytes,
            provenance=provenance,
            notes="Mamba2 depthwise-convolution history.",
        ),
        _fixed_state_tensor(
            layer,
            1,
            (1, heads, head_dim, state_size),
            dtype="float32",
            dtype_bytes=4,
            provenance=provenance,
            notes="Mamba2 recurrent SSM state is accumulated in float32.",
        ),
    ]


def _mamba_manifest(
    config: Mapping[str, Any],
    dtype: str,
    dtype_bytes: int,
    provenance: str,
) -> list[CacheTensorDescriptor]:
    layers = _first_positive(config, "num_hidden_layers", "n_layer", "n_layers")
    intermediate = _first_positive(config, "intermediate_size", "d_inner")
    state_size = _first_positive(config, "state_size", "d_state")
    kernel = _first_positive(config, "conv_kernel", "d_conv")
    if layers <= 0:
        raise FixedKVPlanningError("Mamba layer count is missing or invalid.")
    return [
        tensor
        for layer in range(layers)
        for tensor in _mamba1_state_pair(
            layer,
            intermediate=intermediate,
            state_size=state_size,
            kernel=kernel,
            dtype=dtype,
            dtype_bytes=dtype_bytes,
            provenance=provenance,
        )
    ]


def _mamba2_manifest(
    config: Mapping[str, Any],
    dtype: str,
    dtype_bytes: int,
    provenance: str,
) -> list[CacheTensorDescriptor]:
    layers = _positive_int(config, "num_hidden_layers")
    heads = _positive_int(config, "num_heads")
    head_dim = _positive_int(config, "head_dim")
    state_size = _positive_int(
        config, "ssm_state_size", _positive_int(config, "state_size")
    )
    groups = _positive_int(config, "n_groups")
    kernel = _positive_int(config, "conv_kernel")
    if layers <= 0:
        raise FixedKVPlanningError("Mamba2 layer count is missing or invalid.")
    return [
        tensor
        for layer in range(layers)
        for tensor in _mamba2_state_pair(
            layer,
            heads=heads,
            head_dim=head_dim,
            state_size=state_size,
            groups=groups,
            kernel=kernel,
            dtype=dtype,
            dtype_bytes=dtype_bytes,
            provenance=provenance,
        )
    ]


def _jamba_manifest(
    config: Mapping[str, Any],
    context: int,
    dtype: str,
    dtype_bytes: int,
    provenance: str,
) -> list[CacheTensorDescriptor]:
    layers = _positive_int(config, "num_hidden_layers")
    hidden = _positive_int(config, "hidden_size")
    state_size = _positive_int(config, "mamba_d_state")
    kernel = _positive_int(config, "mamba_d_conv")
    expand = _positive_int(config, "mamba_expand")
    layer_types = config.get("layers_block_type")
    if not isinstance(layer_types, list):
        period = _positive_int(config, "attn_layer_period")
        offset = int(config.get("attn_layer_offset", 0) or 0)
        if period <= 0:
            raise FixedKVPlanningError("Jamba layer pattern is missing or invalid.")
        layer_types = [
            "attention" if index % period == offset else "mamba"
            for index in range(layers)
        ]
    if len(layer_types) != layers or not all((layers, hidden, expand)):
        raise FixedKVPlanningError("Jamba cache dimensions or layer pattern are invalid.")
    _, kv_heads, head_dim = _standard_dimensions(config)
    physical = _round_cache_tokens(context)
    tensors: list[CacheTensorDescriptor] = []
    for layer, raw in enumerate(layer_types):
        if str(raw).lower() == "attention":
            tensors.extend(
                _kv_pair(
                    layer,
                    heads=kv_heads,
                    key_dim=head_dim,
                    value_dim=head_dim,
                    logical_tokens=context,
                    physical_tokens=physical,
                    dtype=dtype,
                    dtype_bytes=dtype_bytes,
                    cache_kind="KVCache",
                    capacity_kind="linear",
                    provenance=provenance,
                )
            )
        elif str(raw).lower() == "mamba":
            tensors.extend(
                _mamba1_state_pair(
                    layer,
                    intermediate=expand * hidden,
                    state_size=state_size,
                    kernel=kernel,
                    dtype=dtype,
                    dtype_bytes=dtype_bytes,
                    provenance=provenance,
                )
            )
        else:
            raise FixedKVPlanningError(f"Unknown Jamba layer type {raw!r}.")
    return tensors


def _mamba2_hybrid_manifest(
    config: Mapping[str, Any],
    context: int,
    dtype: str,
    dtype_bytes: int,
    provenance: str,
    *,
    falcon_parallel_attention: bool = False,
) -> list[CacheTensorDescriptor]:
    layers = _positive_int(config, "num_hidden_layers")
    layer_types = config.get("layer_types")
    if falcon_parallel_attention:
        layer_types = ["parallel"] * layers
    if not isinstance(layer_types, list) or len(layer_types) != layers:
        raise FixedKVPlanningError("Hybrid Mamba2 layer_types are missing or invalid.")
    heads = _positive_int(config, "mamba_n_heads")
    head_dim = _positive_int(config, "mamba_d_head")
    state_size = _positive_int(config, "mamba_d_state")
    groups = _positive_int(config, "mamba_n_groups")
    kernel = _positive_int(config, "mamba_d_conv")
    _, kv_heads, attention_head_dim = _standard_dimensions(config)
    physical = _round_cache_tokens(context)
    tensors: list[CacheTensorDescriptor] = []
    for layer, raw in enumerate(layer_types):
        kind = str(raw).lower()
        if kind in {"mamba", "parallel"}:
            tensors.extend(
                _mamba2_state_pair(
                    layer,
                    heads=heads,
                    head_dim=head_dim,
                    state_size=state_size,
                    groups=groups,
                    kernel=kernel,
                    dtype=dtype,
                    dtype_bytes=dtype_bytes,
                    provenance=provenance,
                )
            )
        if kind in {"attention", "parallel"}:
            tensors.extend(
                _kv_pair(
                    layer,
                    heads=kv_heads,
                    key_dim=attention_head_dim,
                    value_dim=attention_head_dim,
                    logical_tokens=context,
                    physical_tokens=physical,
                    dtype=dtype,
                    dtype_bytes=dtype_bytes,
                    cache_kind="KVCache",
                    capacity_kind="linear",
                    provenance=provenance,
                    prefix="attention." if kind == "parallel" else "",
                )
            )
        if kind not in {"mamba", "attention", "parallel"}:
            raise FixedKVPlanningError(f"Unknown hybrid Mamba2 layer type {raw!r}.")
    return tensors


def _lfm2_manifest(
    config: Mapping[str, Any],
    context: int,
    dtype: str,
    dtype_bytes: int,
    provenance: str,
) -> list[CacheTensorDescriptor]:
    layers = _positive_int(config, "num_hidden_layers")
    hidden = _positive_int(config, "hidden_size")
    kernel = _positive_int(config, "conv_L_cache")
    full_ids = config.get("full_attn_idxs")
    if not isinstance(full_ids, list):
        layer_types = config.get("layer_types")
        if not isinstance(layer_types, list) or len(layer_types) != layers:
            raise FixedKVPlanningError("LFM2 attention layer pattern is missing.")
        full_ids = [
            index
            for index, value in enumerate(layer_types)
            if str(value).lower() == "full_attention"
        ]
    full = {int(value) for value in full_ids}
    _, kv_heads, head_dim = _standard_dimensions(config)
    physical = _round_cache_tokens(context)
    tensors: list[CacheTensorDescriptor] = []
    for layer in range(layers):
        if layer in full:
            tensors.extend(
                _kv_pair(
                    layer,
                    heads=kv_heads,
                    key_dim=head_dim,
                    value_dim=head_dim,
                    logical_tokens=context,
                    physical_tokens=physical,
                    dtype=dtype,
                    dtype_bytes=dtype_bytes,
                    cache_kind="KVCache",
                    capacity_kind="linear",
                    provenance=provenance,
                )
            )
        else:
            tensors.append(
                _fixed_state_tensor(
                    layer,
                    0,
                    (1, kernel - 1, hidden),
                    dtype=dtype,
                    dtype_bytes=dtype_bytes,
                    provenance=provenance,
                    notes="LFM2 short-convolution history.",
                )
            )
    return tensors


def _kimi_linear_manifest(
    config: Mapping[str, Any],
    context: int,
    dtype: str,
    dtype_bytes: int,
    provenance: str,
) -> list[CacheTensorDescriptor]:
    layers = _positive_int(config, "num_hidden_layers")
    linear_raw = config.get("linear_attn_config")
    linear = linear_raw if isinstance(linear_raw, Mapping) else {}
    kda_layers = linear.get("kda_layers")
    heads = _positive_int(linear, "num_heads")
    head_dim = _positive_int(linear, "head_dim")
    kernel = _positive_int(linear, "short_conv_kernel_size", 4)
    latent = _positive_int(config, "kv_lora_rank")
    rope = _positive_int(config, "qk_rope_head_dim")
    if (
        layers <= 0
        or not isinstance(kda_layers, list)
        or not all((heads, head_dim, kernel, latent))
    ):
        raise FixedKVPlanningError("Kimi Linear cache dimensions are incomplete.")
    linear_ids = {int(value) - 1 for value in kda_layers}
    physical = _round_cache_tokens(context)
    tensors: list[CacheTensorDescriptor] = []
    for layer in range(layers):
        if layer not in linear_ids:
            tensors.extend(
                _kv_pair(
                    layer,
                    heads=1,
                    key_dim=latent,
                    value_dim=rope,
                    logical_tokens=context,
                    physical_tokens=physical,
                    dtype=dtype,
                    dtype_bytes=dtype_bytes,
                    cache_kind="KVCache",
                    capacity_kind="linear",
                    provenance=provenance,
                )
            )
            continue
        projection = heads * head_dim
        for index, name in enumerate(("query", "key", "value")):
            tensors.append(
                _fixed_state_tensor(
                    layer,
                    index,
                    (1, kernel - 1, projection),
                    dtype=dtype,
                    dtype_bytes=dtype_bytes,
                    provenance=provenance,
                    notes=f"Kimi Linear {name} short-convolution history.",
                )
            )
        tensors.append(
            _fixed_state_tensor(
                layer,
                3,
                (1, heads, head_dim, head_dim),
                dtype="float32",
                dtype_bytes=4,
                provenance=provenance,
                notes="Kimi Linear gated-delta recurrent state.",
            )
        )
    return tensors


def _recurrent_gemma_manifest(
    config: Mapping[str, Any],
    context: int,
    dtype: str,
    dtype_bytes: int,
    provenance: str,
) -> list[CacheTensorDescriptor]:
    layers = _positive_int(config, "num_hidden_layers")
    hidden = _positive_int(config, "hidden_size")
    kernel = _positive_int(config, "conv1d_width")
    window = _positive_int(config, "attention_window_size")
    pattern = config.get("block_types", config.get("_block_types"))
    if not isinstance(pattern, list) or not pattern or not all((layers, hidden, kernel)):
        raise FixedKVPlanningError("Recurrent Gemma cache metadata is incomplete.")
    heads = _positive_int(config, "num_attention_heads")
    if heads <= 0 or hidden % heads:
        raise FixedKVPlanningError("Recurrent Gemma attention dimensions are invalid.")
    head_dim = hidden // heads
    physical = min(window, _round_cache_tokens(context))
    tensors: list[CacheTensorDescriptor] = []
    for layer in range(layers):
        kind = str(pattern[layer % len(pattern)]).lower()
        if kind == "recurrent":
            tensors.extend(
                [
                    _fixed_state_tensor(
                        layer,
                        0,
                        (1, kernel - 1, hidden),
                        dtype=dtype,
                        dtype_bytes=dtype_bytes,
                        provenance=provenance,
                        notes="Griffin temporal-convolution history.",
                    ),
                    _fixed_state_tensor(
                        layer,
                        1,
                        (1, hidden),
                        dtype=dtype,
                        dtype_bytes=dtype_bytes,
                        provenance=provenance,
                        notes="Griffin RG-LRU recurrent state.",
                    ),
                ]
            )
        elif kind == "attention":
            tensors.extend(
                _kv_pair(
                    layer,
                    heads=1,
                    key_dim=head_dim,
                    value_dim=head_dim,
                    logical_tokens=min(context, window),
                    physical_tokens=physical,
                    dtype=dtype,
                    dtype_bytes=dtype_bytes,
                    cache_kind="RotatingKVCache",
                    capacity_kind="rotating",
                    provenance=provenance,
                )
            )
        else:
            raise FixedKVPlanningError(f"Unknown Recurrent Gemma block type {kind!r}.")
    return tensors


def _rwkv7_manifest(
    config: Mapping[str, Any],
    dtype: str,
    dtype_bytes: int,
    provenance: str,
) -> list[CacheTensorDescriptor]:
    layers = _positive_int(config, "num_hidden_layers")
    hidden = _positive_int(config, "hidden_size")
    head_dim = _positive_int(config, "head_dim")
    if not all((layers, hidden, head_dim)) or hidden % head_dim:
        raise FixedKVPlanningError("RWKV7 cache dimensions are missing or invalid.")
    heads = hidden // head_dim
    tensors: list[CacheTensorDescriptor] = []
    for layer in range(layers):
        for index, shape, note in (
            (0, (1, 1, hidden), "RWKV7 time-mixing token shift."),
            (1, (1, heads, head_dim, head_dim), "RWKV7 WKV recurrent matrix."),
            (2, (1, 1, hidden), "RWKV7 channel-mixing token shift."),
        ):
            tensors.append(
                _fixed_state_tensor(
                    layer,
                    index,
                    shape,
                    dtype=dtype,
                    dtype_bytes=dtype_bytes,
                    provenance=provenance,
                    notes=note,
                )
            )
    return tensors


def _plamo2_manifest(
    config: Mapping[str, Any],
    context: int,
    dtype: str,
    dtype_bytes: int,
    provenance: str,
) -> list[CacheTensorDescriptor]:
    layers = _positive_int(config, "num_hidden_layers")
    enabled = bool(config.get("mamba_enabled", True))
    step = _positive_int(config, "mamba_step", 2)
    heads = _positive_int(config, "mamba_num_heads")
    head_dim = _positive_int(config, "hidden_size_per_head")
    state_size = _positive_int(config, "mamba_d_state")
    kernel = _positive_int(config, "mamba_d_conv")
    # PLaMo2's attention implementation uses hidden_size_per_head directly.
    # A stray generic head_dim field must not override the runtime shape.
    _, kv_heads, _ = _standard_dimensions(config)
    attention_head_dim = _positive_int(config, "hidden_size_per_head")
    if attention_head_dim <= 0:
        raise FixedKVPlanningError(
            "PLaMo2 hidden_size_per_head is missing or invalid."
        )
    physical = _round_cache_tokens(context)
    tensors: list[CacheTensorDescriptor] = []
    for layer in range(layers):
        is_mamba = enabled and (
            layer != layers - 1
            if layers <= step // 2
            else layer % step != step // 2
        )
        if is_mamba:
            tensors.extend(
                _mamba2_state_pair(
                    layer,
                    heads=heads,
                    head_dim=head_dim,
                    state_size=state_size,
                    groups=1,
                    kernel=kernel,
                    dtype=dtype,
                    dtype_bytes=dtype_bytes,
                    provenance=provenance,
                    conv_dim=heads * head_dim,
                )
            )
        else:
            tensors.extend(
                _kv_pair(
                    layer,
                    heads=kv_heads,
                    key_dim=attention_head_dim,
                    value_dim=attention_head_dim,
                    logical_tokens=context,
                    physical_tokens=physical,
                    dtype=dtype,
                    dtype_bytes=dtype_bytes,
                    cache_kind="KVCache",
                    capacity_kind="linear",
                    provenance=provenance,
                )
            )
    return tensors


def _bailing_moe_linear_manifest(
    config: Mapping[str, Any],
    context: int,
    dtype: str,
    dtype_bytes: int,
    provenance: str,
) -> list[CacheTensorDescriptor]:
    layers = _positive_int(config, "num_hidden_layers")
    hidden = _positive_int(config, "hidden_size")
    heads = _positive_int(config, "num_attention_heads")
    group = _positive_int(config, "layer_group_size")
    if not all((layers, hidden, heads, group)) or hidden % heads:
        raise FixedKVPlanningError("Bailing linear-cache dimensions are invalid.")
    _, kv_heads, attention_head_dim = _standard_dimensions(config)
    linear_head_dim = hidden // heads
    physical = _round_cache_tokens(context)
    tensors: list[CacheTensorDescriptor] = []
    for layer in range(layers):
        is_global = (layer + 1) % group == 0 or layer >= (layers // group) * group
        if is_global:
            tensors.extend(
                _kv_pair(
                    layer,
                    heads=kv_heads,
                    key_dim=attention_head_dim,
                    value_dim=attention_head_dim,
                    logical_tokens=context,
                    physical_tokens=physical,
                    dtype=dtype,
                    dtype_bytes=dtype_bytes,
                    cache_kind="KVCache",
                    capacity_kind="linear",
                    provenance=provenance,
                )
            )
        else:
            tensors.append(
                _fixed_state_tensor(
                    layer,
                    0,
                    (1, heads, linear_head_dim, linear_head_dim),
                    dtype=dtype,
                    dtype_bytes=dtype_bytes,
                    provenance=provenance,
                    notes="Bailing recurrent GLA state.",
                )
            )
    return tensors


def _longcat_manifest(
    config: Mapping[str, Any],
    context: int,
    dtype: str,
    dtype_bytes: int,
    provenance: str,
    *,
    ngram: bool,
) -> list[CacheTensorDescriptor]:
    layers = _positive_int(config, "num_layers")
    latent = _positive_int(config, "kv_lora_rank")
    rope = _positive_int(config, "qk_rope_head_dim")
    if not all((layers, latent, rope)):
        raise FixedKVPlanningError("LongCat MLA cache dimensions are incomplete.")
    physical = _round_cache_tokens(context)
    tensors: list[CacheTensorDescriptor] = []
    if ngram:
        neighbors = _positive_int(config, "emb_neighbor_num", 4)
        history = min(context, max(0, neighbors - 1))
        tensors.append(
            _descriptor(
                path="layers.0.state.0",
                cache_kind="ArraysCache",
                role="state_0",
                shape=(1, history),
                dtype="int64",
                dtype_bytes=8,
                logical_tokens=None,
                physical_tokens=None,
                capacity_kind="fixed_state",
                provenance=provenance,
                notes="LongCat N-gram token history.",
            )
        )
    cache_offset = 1 if ngram else 0
    for layer in range(layers):
        for branch in range(2):
            tensors.extend(
                _kv_pair(
                    layer + cache_offset,
                    heads=1,
                    key_dim=latent,
                    value_dim=rope,
                    logical_tokens=context,
                    physical_tokens=physical,
                    dtype=dtype,
                    dtype_bytes=dtype_bytes,
                    cache_kind="KVCache",
                    capacity_kind="linear",
                    provenance=provenance,
                    prefix=f"branch_{branch}.",
                )
            )
    return tensors


def _afm7_manifest(
    config: Mapping[str, Any],
    context: int,
    dtype: str,
    dtype_bytes: int,
    provenance: str,
) -> list[CacheTensorDescriptor]:
    layers = _positive_int(config, "num_layers")
    heads = _positive_int(config, "num_heads")
    kv_heads = _positive_int(config, "num_kv_heads")
    hidden = _positive_int(config, "hidden_dim")
    if not all((layers, heads, kv_heads, hidden)) or hidden % heads:
        raise FixedKVPlanningError("AFM7 cache dimensions are missing or invalid.")
    head_dim = hidden // heads
    physical = _round_cache_tokens(context)
    return [
        tensor
        for layer in range(layers)
        for tensor in _kv_pair(
            layer,
            heads=kv_heads,
            key_dim=head_dim,
            value_dim=head_dim,
            logical_tokens=context,
            physical_tokens=physical,
            dtype=dtype,
            dtype_bytes=dtype_bytes,
            cache_kind="KVCache",
            capacity_kind="linear",
            provenance=provenance,
        )
    ]


def _mimo_v2_manifest(
    config: Mapping[str, Any],
    context: int,
    dtype: str,
    dtype_bytes: int,
    provenance: str,
) -> list[CacheTensorDescriptor]:
    layers = _positive_int(config, "num_hidden_layers")
    pattern = config.get("hybrid_layer_pattern")
    if not isinstance(pattern, list) or len(pattern) != layers:
        raise FixedKVPlanningError("MiMo V2 hybrid_layer_pattern is invalid.")
    window = _positive_int(config, "sliding_window_size")
    physical = _round_cache_tokens(context)
    tensors: list[CacheTensorDescriptor] = []
    for layer, raw in enumerate(pattern):
        sliding = int(raw) == 1
        kv_heads = _positive_int(
            config, "swa_num_key_value_heads" if sliding else "num_key_value_heads"
        )
        key_dim = _positive_int(config, "swa_head_dim" if sliding else "head_dim")
        value_dim = _positive_int(
            config, "swa_v_head_dim" if sliding else "v_head_dim"
        )
        if not all((kv_heads, key_dim, value_dim)) or (sliding and window <= 0):
            raise FixedKVPlanningError("MiMo V2 attention dimensions are invalid.")
        layer_physical = min(window, physical) if sliding else physical
        tensors.extend(
            _kv_pair(
                layer,
                heads=kv_heads,
                key_dim=key_dim,
                value_dim=value_dim,
                logical_tokens=min(context, window) if sliding else context,
                physical_tokens=layer_physical,
                dtype=dtype,
                dtype_bytes=dtype_bytes,
                cache_kind="RotatingKVCache" if sliding else "KVCache",
                capacity_kind="rotating" if sliding else "linear",
                provenance=provenance,
            )
        )
    return tensors


def _iquest_loop_manifest(
    config: Mapping[str, Any],
    context: int,
    dtype: str,
    dtype_bytes: int,
    provenance: str,
) -> list[CacheTensorDescriptor]:
    layers = _positive_int(config, "num_hidden_layers")
    _, kv_heads, head_dim = _standard_dimensions(config)
    window = _positive_int(config, "loop_window_size")
    if layers <= 0 or window <= 0 or _positive_int(config, "loop_num", 2) != 2:
        raise FixedKVPlanningError("IQuest loop-cache metadata is invalid.")
    physical = _round_cache_tokens(context)
    tensors: list[CacheTensorDescriptor] = []
    for loop in range(2):
        for layer in range(layers):
            rotating = loop == 1
            tensors.extend(
                _kv_pair(
                    loop * layers + layer,
                    heads=kv_heads,
                    key_dim=head_dim,
                    value_dim=head_dim,
                    logical_tokens=min(context, window) if rotating else context,
                    physical_tokens=min(window, physical) if rotating else physical,
                    dtype=dtype,
                    dtype_bytes=dtype_bytes,
                    cache_kind="RotatingKVCache" if rotating else "KVCache",
                    capacity_kind="rotating" if rotating else "linear",
                    provenance=provenance,
                )
            )
    return tensors


def _nemotron_nas_manifest(
    config: Mapping[str, Any],
    context: int,
    dtype: str,
    dtype_bytes: int,
    provenance: str,
) -> list[CacheTensorDescriptor]:
    layers = _positive_int(config, "num_hidden_layers")
    heads = _positive_int(config, "num_attention_heads")
    hidden = _positive_int(config, "hidden_size")
    blocks = config.get("block_configs")
    if (
        not isinstance(blocks, list)
        or len(blocks) != layers
        or not all((layers, heads, hidden))
        or hidden % heads
    ):
        raise FixedKVPlanningError("Nemotron-NAS cache metadata is incomplete.")
    head_dim = hidden // heads
    physical = _round_cache_tokens(context)
    tensors: list[CacheTensorDescriptor] = []
    cache_index = 0
    for layer, block in enumerate(blocks):
        if not isinstance(block, Mapping):
            raise FixedKVPlanningError(
                f"Nemotron-NAS block_configs[{layer}] must be an object."
            )
        attention = block.get("attention")
        attention = attention if isinstance(attention, Mapping) else {}
        if bool(attention.get("no_op")):
            continue
        if bool(attention.get("replace_with_linear")):
            # mlx-lm retains a cache-list position for the linear replacement,
            # but the block deliberately ignores it and stores zero tensors.
            cache_index += 1
            continue
        group = _positive_int(attention, "n_heads_in_group")
        if group <= 0 or heads % group:
            raise FixedKVPlanningError(
                f"Nemotron-NAS layer {layer} has an invalid GQA group size."
            )
        tensors.extend(
            _kv_pair(
                cache_index,
                heads=heads // group,
                key_dim=head_dim,
                value_dim=head_dim,
                logical_tokens=context,
                physical_tokens=physical,
                dtype=dtype,
                dtype_bytes=dtype_bytes,
                cache_kind="KVCache",
                capacity_kind="linear",
                provenance=provenance,
            )
        )
        cache_index += 1
    return tensors


def _bailing_hybrid_manifest(
    config: Mapping[str, Any],
    context: int,
    dtype: str,
    dtype_bytes: int,
    provenance: str,
) -> list[CacheTensorDescriptor]:
    """Describe Ling/Bailing's interleaved MLA and recurrent KDA layers."""

    layers = _positive_int(config, "num_hidden_layers")
    heads = _positive_int(config, "num_attention_heads")
    head_dim = _positive_int(config, "head_dim")
    latent = _positive_int(config, "kv_lora_rank")
    rope = _positive_int(config, "qk_rope_head_dim")
    group = _positive_int(config, "layer_group_size")
    kernel = _positive_int(config, "short_conv_kernel_size", 4)
    if not all((layers, heads, head_dim, latent, rope, group, kernel)):
        raise FixedKVPlanningError(
            "Bailing hybrid cache dimensions are missing or invalid."
        )
    physical = _round_cache_tokens(context)
    tensors: list[CacheTensorDescriptor] = []
    for layer in range(layers):
        is_global = (layer + 1) % group == 0 or layer >= (layers // group) * group
        if is_global:
            tensors.extend(
                _kv_pair(
                    layer,
                    heads=1,
                    key_dim=latent,
                    value_dim=rope,
                    logical_tokens=context,
                    physical_tokens=physical,
                    dtype=dtype,
                    dtype_bytes=dtype_bytes,
                    cache_kind="KVCache",
                    capacity_kind="linear",
                    provenance=provenance,
                )
            )
            continue

        channel_dim = heads * head_dim
        tensors.append(
            _fixed_state_tensor(
                layer,
                0,
                (1, heads, head_dim, head_dim),
                dtype="float32",
                dtype_bytes=4,
                provenance=provenance,
                notes="Bailing recurrent KDA state is accumulated in float32.",
            )
        )
        for index, name in enumerate(("query", "key", "value"), start=1):
            tensors.append(
                _fixed_state_tensor(
                    layer,
                    index,
                    (1, channel_dim, kernel),
                    dtype=dtype,
                    dtype_bytes=dtype_bytes,
                    provenance=provenance,
                    notes=f"Bailing {name} depthwise-convolution state.",
                )
            )
    return tensors


def _inkling_manifest(
    config: Mapping[str, Any],
    context: int,
    dtype: str,
    dtype_bytes: int,
    provenance: str,
) -> list[CacheTensorDescriptor]:
    """Describe Inkling's per-layer K/V plus four short-convolution states."""

    layers = _positive_int(config, "num_hidden_layers")
    heads = _positive_int(config, "num_key_value_heads")
    head_dim = _positive_int(config, "head_dim")
    swa_heads = _positive_int(config, "swa_num_key_value_heads")
    swa_head_dim = _positive_int(config, "swa_head_dim")
    hidden = _positive_int(config, "hidden_size")
    kernel = _positive_int(config, "sconv_kernel_size", 4)
    if not all((layers, heads, head_dim, swa_heads, swa_head_dim, hidden, kernel)):
        raise FixedKVPlanningError("Inkling cache dimensions are missing or invalid.")

    layer_types = config.get("layer_types")
    local_ids = config.get("local_layer_ids")
    if isinstance(layer_types, list) and len(layer_types) != layers:
        raise FixedKVPlanningError(
            "Inkling layer_types must contain one entry per layer."
        )
    local_set = (
        {int(value) for value in local_ids}
        if isinstance(local_ids, list)
        else None
    )

    physical = _round_cache_tokens(context)
    tensors: list[CacheTensorDescriptor] = []
    for layer in range(layers):
        if isinstance(layer_types, list):
            sliding = str(layer_types[layer]).lower() == "hybrid_sliding"
        elif local_set is not None:
            sliding = layer in local_set
        else:
            sliding = bool((layer + 1) % 6)
        kv_heads = swa_heads if sliding else heads
        kv_head_dim = swa_head_dim if sliding else head_dim
        tensors.extend(
            _kv_pair(
                layer,
                heads=kv_heads,
                key_dim=kv_head_dim,
                value_dim=kv_head_dim,
                logical_tokens=context,
                physical_tokens=physical,
                dtype=dtype,
                dtype_bytes=dtype_bytes,
                cache_kind="KVCache",
                capacity_kind=("sliding_attention_linear_storage" if sliding else "linear"),
                provenance=provenance,
            )
        )
        conv_history = kernel - 1
        channels = kv_heads * kv_head_dim
        for index, channel_count in enumerate(
            (channels, channels, hidden, hidden)
        ):
            tensors.append(
                _fixed_state_tensor(
                    layer,
                    index,
                    (1, conv_history, channel_count),
                    dtype="float32",
                    dtype_bytes=4,
                    provenance=provenance,
                    notes="Inkling short-convolution history is stored in float32.",
                )
            )
    return tensors


def _baichuan_m1_manifest(
    config: Mapping[str, Any],
    context: int,
    dtype: str,
    dtype_bytes: int,
    provenance: str,
) -> list[CacheTensorDescriptor]:
    """Describe Baichuan M1 convolution state plus global/sliding K/V."""

    layers = _positive_int(config, "num_hidden_layers")
    hidden = _positive_int(config, "hidden_size")
    global_heads = _positive_int(config, "num_attention_heads")
    global_kv_heads = _positive_int(config, "num_key_value_heads")
    swa_heads = _positive_int(config, "num_swa_attention_heads", global_heads)
    swa_kv_heads = _positive_int(
        config,
        "num_swa_key_value_heads",
        global_kv_heads,
    )
    window = _positive_int(config, "sliding_window")
    conv_window = _positive_int(config, "conv_window")
    sliding_ids = config.get("sliding_window_layers")
    if (
        not all(
            (
                layers,
                hidden,
                global_heads,
                global_kv_heads,
                swa_heads,
                swa_kv_heads,
                window,
                conv_window,
            )
        )
        or not isinstance(sliding_ids, list)
    ):
        raise FixedKVPlanningError(
            "Baichuan M1 cache dimensions are missing or invalid."
        )
    if hidden % global_heads or hidden % swa_heads:
        raise FixedKVPlanningError(
            "Baichuan M1 hidden_size is not divisible by its attention head counts."
        )

    sliding = {int(value) for value in sliding_ids}
    physical = _round_cache_tokens(context)
    tensors: list[CacheTensorDescriptor] = []
    for layer in range(layers):
        is_sliding = layer in sliding
        heads = swa_kv_heads if is_sliding else global_kv_heads
        head_dim = hidden // (swa_heads if is_sliding else global_heads)
        for index, name in enumerate(("key", "value")):
            tensors.append(
                _fixed_state_tensor(
                    layer,
                    index,
                    (1, heads, conv_window - 1, head_dim),
                    dtype=dtype,
                    dtype_bytes=dtype_bytes,
                    provenance=provenance,
                    notes=f"Baichuan M1 previous {name} convolution sample.",
                )
            )
        tensors.extend(
            _kv_pair(
                layer,
                heads=heads,
                key_dim=head_dim,
                value_dim=head_dim,
                logical_tokens=(min(context, window) if is_sliding else context),
                physical_tokens=(
                    min(window, physical) if is_sliding else physical
                ),
                dtype=dtype,
                dtype_bytes=dtype_bytes,
                cache_kind=("RotatingKVCache" if is_sliding else "KVCache"),
                capacity_kind=("rotating" if is_sliding else "linear"),
                provenance=provenance,
            )
        )
    return tensors


def _deepseek_v3_manifest(
    config: Mapping[str, Any],
    context: int,
    dtype: str,
    dtype_bytes: int,
    provenance: str,
    *,
    with_index: bool,
) -> list[CacheTensorDescriptor]:
    layers = _positive_int(config, "num_hidden_layers")
    latent = _positive_int(config, "kv_lora_rank")
    rope = _positive_int(config, "qk_rope_head_dim")
    if not all((layers, latent, rope)):
        raise FixedKVPlanningError("DeepSeek MLA cache dimensions are incomplete.")
    physical = _round_cache_tokens(context)
    index_dim = _positive_int(config, "index_head_dim") if with_index else 0
    if with_index and index_dim <= 0:
        raise FixedKVPlanningError("DSA index_head_dim is missing or invalid.")

    # deepseek_v32.make_cache, inherited by GLM MoE DSA, creates the latent
    # and index KVCache pair for every layer.  Indexer patterns control use,
    # not whether persistent cache storage exists.
    index_layers = set(range(layers))

    tensors: list[CacheTensorDescriptor] = []
    for layer in range(layers):
        tensors.extend(
            _kv_pair(
                layer,
                heads=1,
                key_dim=latent,
                value_dim=rope,
                logical_tokens=context,
                physical_tokens=physical,
                dtype=dtype,
                dtype_bytes=dtype_bytes,
                cache_kind="KVCache",
                capacity_kind="linear",
                provenance=provenance,
                prefix="main.",
            )
        )
        if with_index and layer in index_layers:
            tensors.extend(
                _kv_pair(
                    layer,
                    heads=1,
                    key_dim=index_dim,
                    value_dim=0,
                    logical_tokens=context,
                    physical_tokens=physical,
                    dtype=dtype,
                    dtype_bytes=dtype_bytes,
                    cache_kind="KVCache",
                    capacity_kind="linear",
                    provenance=provenance,
                    prefix="index.",
                )
            )
    return tensors


def _pool_tensor(
    layer: int,
    pool_name: str,
    role: str,
    shape: tuple[int, ...],
    *,
    dtype: str,
    dtype_bytes: int,
    logical_tokens: int | None,
    physical_tokens: int | None,
    provenance: str,
) -> CacheTensorDescriptor:
    return _descriptor(
        path=f"layers.{layer}.{pool_name}.{role}",
        cache_kind="PoolingCache",
        role=role,
        shape=shape,
        dtype=dtype,
        dtype_bytes=dtype_bytes,
        logical_tokens=logical_tokens,
        physical_tokens=physical_tokens,
        capacity_kind="pooled" if role == "pooled" else "fixed_state",
        provenance=provenance,
    )


def _deepseek_v4_manifest(
    config: Mapping[str, Any],
    context: int,
    dtype: str,
    dtype_bytes: int,
    provenance: str,
) -> list[CacheTensorDescriptor]:
    layers = _positive_int(config, "num_hidden_layers")
    head_dim = _positive_int(config, "head_dim")
    window = _positive_int(config, "sliding_window")
    index_dim = _positive_int(config, "index_head_dim", 128)
    if not all((layers, head_dim, window, index_dim)):
        raise FixedKVPlanningError("DeepSeek V4 cache dimensions are incomplete.")

    ratios = config.get("compress_ratios")
    if not ratios:
        ratios = [0] + [4 if index % 2 else 128 for index in range(max(layers - 2, 0))]
        if layers >= 2:
            ratios.append(0)
    if not isinstance(ratios, list) or len(ratios) < layers:
        raise FixedKVPlanningError("compress_ratios must contain one entry per layer.")
    try:
        # DeepSeek checkpoints can include entries for auxiliary/MTP blocks.
        # The runtime ModelArgs deliberately truncates those entries before
        # constructing the serving layers, so the planner must do the same.
        ratios = [int(value) for value in ratios[:layers]]
    except (TypeError, ValueError) as exc:
        raise FixedKVPlanningError(
            "compress_ratios contains a non-integer value."
        ) from exc
    if any(ratio not in (0, 4, 128) for ratio in ratios):
        raise FixedKVPlanningError(
            "DeepSeek V4 fixed planning supports compression ratios 0, 4, and 128 only."
        )

    local_physical = min(window, _round_cache_tokens(context))
    local_logical = min(context, window)
    tensors: list[CacheTensorDescriptor] = []
    for layer, ratio in enumerate(ratios):
        tensors.extend(
            _kv_pair(
                layer,
                heads=1,
                key_dim=head_dim,
                value_dim=0,
                logical_tokens=local_logical,
                physical_tokens=local_physical,
                dtype=dtype,
                dtype_bytes=dtype_bytes,
                cache_kind="RotatingKVCache",
                capacity_kind="rotating",
                provenance=provenance,
                prefix="local.",
            )
        )
        if ratio == 0:
            continue

        pools = [("main_pool", head_dim, ratio == 4)]
        if ratio == 4:
            pools.append(("index_pool", index_dim, True))
        for pool_name, pooled_dim, overlap in pools:
            pooled_tokens = context // ratio
            projection_dim = pooled_dim * (2 if overlap else 1)
            tensors.append(
                _pool_tensor(
                    layer,
                    pool_name,
                    "pooled",
                    (1, pooled_tokens, pooled_dim),
                    dtype=dtype,
                    dtype_bytes=dtype_bytes,
                    logical_tokens=pooled_tokens,
                    physical_tokens=pooled_tokens,
                    provenance=provenance,
                )
            )
            for role in ("buffer_kv", "buffer_gate"):
                tensors.append(
                    _pool_tensor(
                        layer,
                        pool_name,
                        role,
                        (1, ratio, projection_dim),
                        dtype=dtype,
                        dtype_bytes=dtype_bytes,
                        logical_tokens=None,
                        physical_tokens=ratio,
                        provenance=provenance,
                    )
                )
            if overlap and context >= ratio:
                for role in ("previous_window_kv", "previous_window_gate"):
                    tensors.append(
                        _pool_tensor(
                            layer,
                            pool_name,
                            role,
                            (1, 1, ratio, projection_dim),
                            dtype=dtype,
                            dtype_bytes=dtype_bytes,
                            logical_tokens=None,
                            physical_tokens=ratio,
                            provenance=provenance,
                        )
                    )
    return tensors


def _prefixed_manifest(
    tensors: Sequence[CacheTensorDescriptor],
    prefix: str,
    *,
    note: str,
) -> list[CacheTensorDescriptor]:
    return [
        replace(
            tensor,
            path=f"{prefix}.{tensor.path}",
            provenance=f"{tensor.provenance}; {note}",
        )
        for tensor in tensors
    ]


def _native_mtp_manifest(
    config: Mapping[str, Any],
    context: int,
    dtype: str,
    dtype_bytes: int,
    provenance: str,
) -> list[CacheTensorDescriptor]:
    """Describe auxiliary caches created by OMLX's built-in native MTP heads.

    The cache factory remains the runtime authority. This config-time manifest
    mirrors every native MTP family OMLX can attach, and launch verifies the
    resulting materialized byte count before exposing the engine.
    """

    model_type = str(config.get("model_type", "")).lower()
    main_layers = _positive_int(config, "num_hidden_layers")
    mtp_layers = _positive_int(
        config,
        "mtp_num_hidden_layers",
        _positive_int(config, "num_nextn_predict_layers"),
    )
    target_ids = config.get("dspark_target_layer_ids")
    target_count = len(target_ids) if isinstance(target_ids, list) else 0
    dspark_stages = _positive_int(config, "n_mtp_layers") or target_count
    dspark = bool(
        _positive_int(config, "dspark_block_size")
        and target_count
        and dspark_stages
    )

    if model_type.startswith("deepseek_v4") and dspark:
        head_dim = _positive_int(config, "head_dim")
        window = _positive_int(config, "sliding_window")
        if head_dim <= 0 or window <= 0:
            raise FixedKVPlanningError(
                "DeepSeek DSpark MTP cache dimensions are incomplete."
            )
        physical = min(context, window)
        return [
            _descriptor(
                path=f"mtp.layers.{stage}.keys",
                cache_kind="DSparkContextCache",
                role="keys",
                shape=(1, 1, physical, head_dim),
                dtype=dtype,
                dtype_bytes=dtype_bytes,
                logical_tokens=min(context, window),
                physical_tokens=physical,
                capacity_kind="rotating",
                provenance=f"{provenance}; embedded DSpark MTP context cache",
            )
            for stage in range(dspark_stages)
        ]

    if mtp_layers <= 0:
        # Stateless assistant heads such as Gemma 4 intentionally return an
        # empty MTP cache. A checkpoint that declares no head layers adds no
        # auxiliary cache allocation.
        return []

    if model_type.startswith("deepseek_v4"):
        ratios = config.get("compress_ratios")
        mtp_ratios: list[int] = []
        if isinstance(ratios, list):
            try:
                mtp_ratios = [
                    int(value)
                    for value in ratios[main_layers : main_layers + mtp_layers]
                ]
            except (TypeError, ValueError) as exc:
                raise FixedKVPlanningError(
                    "DeepSeek V4 MTP compress_ratios contains a non-integer value."
                ) from exc
        if len(mtp_ratios) < mtp_layers:
            mtp_ratios.extend([0] * (mtp_layers - len(mtp_ratios)))
        mtp_config = dict(config)
        mtp_config["num_hidden_layers"] = mtp_layers
        mtp_config["compress_ratios"] = mtp_ratios
        persistent = _prefixed_manifest(
            _deepseek_v4_manifest(
                mtp_config, context, dtype, dtype_bytes, provenance
            ),
            "mtp",
            note="native MTP persistent head cache",
        )
        # Lightning DeepSeek clones its head cache for each speculative chain
        # because a rotated cache cannot be trimmed exactly. Reserve that copy
        # at launch as part of the fixed pool instead of cloning dynamically.
        clone = _prefixed_manifest(
            [replace(tensor, path=tensor.path.removeprefix("mtp.")) for tensor in persistent],
            "mtp_clone",
            note="native MTP speculative cache copy",
        )
        return persistent + clone

    if model_type.startswith("glm") and "dsa" in model_type:
        mtp_config = dict(config)
        mtp_config["model_type"] = "mtp_dsa"
        mtp_config["num_hidden_layers"] = mtp_layers
        persistent = _deepseek_v3_manifest(
            mtp_config,
            context,
            dtype,
            dtype_bytes,
            provenance,
            with_index=True,
        )
        return _prefixed_manifest(
            persistent,
            "mtp",
            note="native GLM MTP latent and index caches",
        )

    _, kv_heads, head_dim = _standard_dimensions(config)
    physical = _round_cache_tokens(context)
    cache_kind = "KVCache"
    capacity_kind = "linear"
    logical = context
    if model_type.startswith(("step3p5", "step3p7")):
        layer_types = config.get("layer_types")
        raw_kind = (
            str(layer_types[main_layers]).lower()
            if isinstance(layer_types, list) and len(layer_types) > main_layers
            else ("sliding" if main_layers % 2 == 0 else "full")
        )
        if any(token in raw_kind for token in ("sliding", "window", "local")):
            window = _positive_int(config, "sliding_window")
            if window <= 0:
                raise FixedKVPlanningError(
                    "Step native MTP uses sliding attention but sliding_window is missing."
                )
            physical = min(window, physical)
            logical = min(context, window)
            cache_kind = "RotatingKVCache"
            capacity_kind = "rotating"

    tensors: list[CacheTensorDescriptor] = []
    for layer in range(mtp_layers):
        tensors.extend(
            _kv_pair(
                layer,
                heads=kv_heads,
                key_dim=head_dim,
                value_dim=head_dim,
                logical_tokens=logical,
                physical_tokens=physical,
                dtype=dtype,
                dtype_bytes=dtype_bytes,
                cache_kind=cache_kind,
                capacity_kind=capacity_kind,
                provenance=provenance,
            )
        )
    return _prefixed_manifest(
        tensors,
        "mtp",
        note="native MTP full-attention head cache",
    )


def estimate_cache_tensors_from_config(
    config: Mapping[str, Any],
    context_window: int,
    *,
    provenance: str = "config.json estimate",
    prefill_step_size: int = 2048,
) -> tuple[CacheTensorDescriptor, ...]:
    """Build a per-session cache manifest from supported config metadata."""

    if context_window <= 0:
        raise FixedKVPlanningError("context_window must be positive.")
    native_limit = _context_limit(config)
    if native_limit and context_window > native_limit:
        raise FixedKVPlanningError(
            f"Requested context {context_window:,} exceeds the model limit "
            f"of {native_limit:,} tokens."
        )

    model_type = str(config.get("model_type", "")).lower()
    if not model_type:
        raise FixedKVPlanningError("model_type is missing from config.json.")
    dtype, dtype_bytes, dtype_source = _dtype(config)
    tensor_provenance = f"{provenance}; {dtype_source}"

    root_model_type = str(config.get("_root_model_type", model_type)).lower()
    qwen_gated_delta = model_type.startswith(
        ("qwen3_next", "qwen3_5", "qwen3_6", "qwen3_8")
    )

    if root_model_type in {"unlimited_ocr", "unlimited-ocr"}:
        tensors = _ring_sliding_manifest(
            config, context_window, dtype, dtype_bytes, tensor_provenance
        )
    elif model_type.startswith("longcat_flash_ngram"):
        tensors = _longcat_manifest(
            config,
            context_window,
            dtype,
            dtype_bytes,
            tensor_provenance,
            ngram=True,
        )
    elif model_type.startswith("longcat_flash"):
        tensors = _longcat_manifest(
            config,
            context_window,
            dtype,
            dtype_bytes,
            tensor_provenance,
            ngram=False,
        )
    elif model_type.startswith("kimi_linear"):
        tensors = _kimi_linear_manifest(
            config, context_window, dtype, dtype_bytes, tensor_provenance
        )
    elif model_type.startswith("falcon_h1"):
        tensors = _mamba2_hybrid_manifest(
            config,
            context_window,
            dtype,
            dtype_bytes,
            tensor_provenance,
            falcon_parallel_attention=True,
        )
    elif model_type.startswith("granitemoehybrid"):
        tensors = _mamba2_hybrid_manifest(
            config, context_window, dtype, dtype_bytes, tensor_provenance
        )
    elif model_type.startswith("jamba"):
        tensors = _jamba_manifest(
            config, context_window, dtype, dtype_bytes, tensor_provenance
        )
    elif model_type.startswith(("lfm2", "lfm2_moe")):
        tensors = _lfm2_manifest(
            config, context_window, dtype, dtype_bytes, tensor_provenance
        )
    elif model_type.startswith("plamo2"):
        tensors = _plamo2_manifest(
            config, context_window, dtype, dtype_bytes, tensor_provenance
        )
    elif model_type.startswith("recurrent_gemma"):
        tensors = _recurrent_gemma_manifest(
            config, context_window, dtype, dtype_bytes, tensor_provenance
        )
    elif model_type.startswith("rwkv7"):
        tensors = _rwkv7_manifest(config, dtype, dtype_bytes, tensor_provenance)
    elif model_type.startswith("mamba2"):
        tensors = _mamba2_manifest(config, dtype, dtype_bytes, tensor_provenance)
    elif model_type.startswith(("mamba", "falcon_mamba")):
        tensors = _mamba_manifest(config, dtype, dtype_bytes, tensor_provenance)
    elif model_type.startswith("bailing_moe_linear"):
        tensors = _bailing_moe_linear_manifest(
            config, context_window, dtype, dtype_bytes, tensor_provenance
        )
    elif model_type.startswith("mimo_v2_flash"):
        tensors = _mimo_v2_manifest(
            config, context_window, dtype, dtype_bytes, tensor_provenance
        )
    elif model_type.startswith("iquestloopcoder"):
        tensors = _iquest_loop_manifest(
            config, context_window, dtype, dtype_bytes, tensor_provenance
        )
    elif model_type.startswith("afm7"):
        tensors = _afm7_manifest(
            config, context_window, dtype, dtype_bytes, tensor_provenance
        )
    elif model_type.startswith("nemotron-nas"):
        tensors = _nemotron_nas_manifest(
            config, context_window, dtype, dtype_bytes, tensor_provenance
        )
    elif qwen_gated_delta:
        tensors = _qwen_gated_delta_manifest(
            config, context_window, dtype, dtype_bytes, tensor_provenance
        )
    elif model_type.startswith("nemotron_h"):
        tensors = _nemotron_h_manifest(
            config, context_window, dtype, dtype_bytes, tensor_provenance
        )
    elif model_type.startswith("bailing_hybrid"):
        tensors = _bailing_hybrid_manifest(
            config, context_window, dtype, dtype_bytes, tensor_provenance
        )
    elif model_type.startswith("inkling"):
        tensors = _inkling_manifest(
            config, context_window, dtype, dtype_bytes, tensor_provenance
        )
    elif model_type.startswith("baichuan_m1"):
        tensors = _baichuan_m1_manifest(
            config, context_window, dtype, dtype_bytes, tensor_provenance
        )
    elif model_type.startswith("minimax_m3"):
        tensors = _minimax_m3_manifest(
            config, context_window, dtype, dtype_bytes, tensor_provenance
        )
    elif model_type.startswith("deepseek_v4"):
        tensors = _deepseek_v4_manifest(
            config, context_window, dtype, dtype_bytes, tensor_provenance
        )
    elif model_type.startswith("deepseek_v32") or "moe_dsa" in model_type:
        tensors = _deepseek_v3_manifest(
            config,
            context_window,
            dtype,
            dtype_bytes,
            tensor_provenance,
            with_index=True,
        )
    elif model_type.startswith("deepseek_v3") or model_type in {
        "joyai_llm_flash",
        "kimi_k2",
    }:
        tensors = _deepseek_v3_manifest(
            config,
            context_window,
            dtype,
            dtype_bytes,
            tensor_provenance,
            with_index=False,
        )
    elif model_type.startswith("deepseek_v2"):
        tensors = _deepseek_v2_manifest(
            config, context_window, dtype, dtype_bytes, tensor_provenance
        )
    else:
        if _positive_int(config, "kv_lora_rank"):
            raise FixedKVPlanningError(
                f"Model type {model_type!r} advertises a compressed or hybrid MLA "
                "cache but has no verified fixed-cache layout adapter."
            )
        tensors = _generic_manifest(
            config,
            model_type,
            context_window,
            dtype,
            dtype_bytes,
            tensor_provenance,
            prefill_step_size,
        )
    return tuple(tensors)


def _detect_system_memory(available_override: int | None) -> _SystemMemory:
    total: int | None = None
    total_source: str | None = None
    available = available_override
    available_source = (
        "caller supplied available-memory snapshot" if available is not None else None
    )

    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                [_SYSCTL, "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                check=True,
                timeout=2,
            )
            total = int(result.stdout.strip())
            total_source = "macOS sysctl hw.memsize"
        except (OSError, ValueError, subprocess.SubprocessError):
            pass

    try:
        import psutil

        memory = psutil.virtual_memory()
        if total is None:
            total = int(memory.total)
            total_source = "psutil.virtual_memory.total"
        if available is None:
            available = int(memory.available)
            available_source = "psutil.virtual_memory.available snapshot"
    except (ImportError, OSError, ValueError):
        pass

    return _SystemMemory(total, total_source, available, available_source)


def _binding_limit(
    available: int | None, ceiling: int | None
) -> tuple[int | None, str | None]:
    candidates = []
    if available is not None:
        candidates.append((available, "available-memory snapshot"))
    if ceiling is not None:
        candidates.append((ceiling, "caller supplied Metal/admission memory ceiling"))
    return min(candidates, key=lambda item: item[0]) if candidates else (None, None)


def _pool_row_scratch_bytes(
    tensors: Sequence[CacheTensorDescriptor],
    *,
    mandatory_only: bool = False,
) -> int:
    """Return the shared row workspace required by the live batch allocator.

    Layers with the same role, dtype, and non-batch shape share one scratch
    array because mlx-lm filters cache layers sequentially. Key and value rows
    remain separate so both can be materialized before either pool tensor is
    updated.
    """

    unique: dict[tuple[str, str, tuple[int, ...]], int] = {}
    for tensor in tensors:
        if mandatory_only:
            mandatory = tensor.cache_kind in {
                "ChunkedKVCache",
                "DSparkContextCache",
            }
            if not mandatory:
                continue
        signature = (tensor.role, tensor.dtype, tensor.shape[1:])
        unique.setdefault(signature, tensor.nbytes)
    return sum(unique.values())


def _pooling_undo_bytes(tensors: tuple[CacheTensorDescriptor, ...]) -> int:
    """Return per-session storage needed for exact pooled-cache rollback."""

    return sum(
        tensor.nbytes
        for tensor in tensors
        if tensor.cache_kind == "PoolingCache"
        and tensor.role in {"buffer_kv", "buffer_gate"}
    )


def estimate_model_memory(
    model_path: str | Path,
    context_window: int,
    *,
    weights_bytes: int,
    requested_session_slots: int,
    available_memory_bytes: int | None = None,
    memory_ceiling_bytes: int | None = None,
    other_fixed_bytes: int = 0,
    prefill_step_size: int = 2048,
    native_mtp_enabled: bool = False,
) -> ModelMemoryPlan:
    """Estimate fixed model memory and choose feasible per-session cache slots.

    ``context_window`` applies to every session. ``weights_bytes`` and
    ``other_fixed_bytes`` are caller-supplied estimates. The optional available
    value is a point-in-time system snapshot; the optional ceiling is treated
    as the maximum launch allocation for weights, known fixed memory, and the
    fixed cache. Neither is presented as a guaranteed amount of RAM.
    """

    path = Path(model_path).expanduser().resolve()
    if weights_bytes < 0 or other_fixed_bytes < 0:
        raise FixedKVPlanningError(
            "weights_bytes and other_fixed_bytes cannot be negative."
        )
    if requested_session_slots <= 0:
        raise FixedKVPlanningError("requested_session_slots must be positive.")
    if available_memory_bytes is not None and available_memory_bytes < 0:
        raise FixedKVPlanningError("available_memory_bytes cannot be negative.")
    if memory_ceiling_bytes is not None and memory_ceiling_bytes < 0:
        raise FixedKVPlanningError("memory_ceiling_bytes cannot be negative.")

    config, config_source = _load_config(path)
    tensors = list(estimate_cache_tensors_from_config(
        config,
        context_window,
        provenance=f"{config_source} shape metadata",
        prefill_step_size=prefill_step_size,
    ))
    if native_mtp_enabled:
        dtype, dtype_bytes, dtype_source = _dtype(config)
        tensors.extend(
            _native_mtp_manifest(
                config,
                context_window,
                dtype,
                dtype_bytes,
                f"{config_source} shape metadata; {dtype_source}",
            )
        )
    tensors_tuple = tuple(tensors)
    unsupported_cache_kinds = sorted(
        {
            tensor.cache_kind
            for tensor in tensors
            if tensor.cache_kind
            not in {
                "KVCache",
                "RotatingKVCache",
                "ArraysCache",
                "PoolingCache",
                "MiniMaxM3KVCache",
                "ChunkedKVCache",
                "RingSlidingKVCache",
                "DSparkContextCache",
            }
        }
    )
    if unsupported_cache_kinds:
        kinds = ", ".join(unsupported_cache_kinds)
        raise FixedKVPlanningError(
            f"The cache manifest contains {kinds}, but the launch-time fixed-pool "
            "allocator has no adapter for that layout. Fixed allocation is "
            "refused before model weights are loaded; use a supported checkpoint "
            "or add a matching runtime pool adapter."
        )
    per_session = sum(tensor.nbytes for tensor in tensors_tuple)
    if per_session < 0:
        raise FixedKVPlanningError("The supported cache manifest has invalid storage.")

    system = _detect_system_memory(available_memory_bytes)
    binding, binding_source = _binding_limit(system.available, memory_ceiling_bytes)
    base = weights_bytes + other_fixed_bytes
    row_scratch = _pool_row_scratch_bytes(tensors_tuple)
    pooling_undo = _pooling_undo_bytes(tensors_tuple)
    mandatory_scratch = _pool_row_scratch_bytes(
        tensors_tuple,
        mandatory_only=True,
    )
    cache_layout_max_slots = 1 if any(
        tensor.cache_kind == "ChunkedKVCache" for tensor in tensors_tuple
    ) else None

    if per_session == 0:
        max_slots = requested_session_slots
        reserved_slots = requested_session_slots
        capped = False
        requested_fits = None if binding is None else base <= binding
        fits = None if binding is None else requested_fits
        if binding is None:
            reason = (
                "This architecture has no persistent per-session cache tensors; "
                "weight fit remains unknown because no reliable memory limit was supplied."
            )
        elif requested_fits:
            reason = None
        else:
            max_slots = 0
            reserved_slots = 0
            capped = True
            reason = (
                f"Model weights and other fixed allocations need {base:,} bytes, "
                f"above the binding {binding_source} of {binding:,} bytes."
            )
    elif binding is None:
        max_slots = cache_layout_max_slots
        reserved_slots = min(
            requested_session_slots,
            cache_layout_max_slots or requested_session_slots,
        )
        fits = None
        requested_fits = (
            None
            if cache_layout_max_slots is None
            else requested_session_slots <= cache_layout_max_slots
        )
        capped = reserved_slots < requested_session_slots
        reason = (
            "Llama 4 chunked attention is serialized by the current runtime, so "
            "concurrency was capped to one; memory fit remains unknown because no "
            "reliable available-memory snapshot or launch ceiling was supplied."
            if capped and cache_layout_max_slots == 1
            else (
                "No reliable available-memory snapshot or launch ceiling was "
                "supplied; session fit is unknown."
            )
        )
    else:
        kv_budget = max(0, binding - base)
        # MLX arrays are immutable. Continuous batching therefore needs one
        # materialized row workspace to compact a surviving row or
        # move a newly admitted row without transiently duplicating the whole
        # pool. A single-session engine never moves rows and needs no scratch.
        first_slot_cost = per_session + pooling_undo + mandatory_scratch
        if kv_budget < first_slot_cost:
            max_slots = 0
        else:
            max_slots = 1
            slot_cost = per_session + pooling_undo
            if kv_budget >= 2 * slot_cost + row_scratch:
                max_slots = max(2, (kv_budget - row_scratch) // slot_cost)
        if cache_layout_max_slots is not None:
            max_slots = min(max_slots, cache_layout_max_slots)
        reserved_slots = min(requested_session_slots, max_slots)
        fits = reserved_slots > 0
        requested_fits = max_slots >= requested_session_slots
        capped = reserved_slots < requested_session_slots
        if reserved_slots == 0:
            reason = (
                f"No session slot fits: weights and other fixed allocations use "
                f"{base:,} bytes, while the binding {binding_source} is "
                f"{binding:,} bytes. Free memory, lower the context window, or "
                "choose a smaller model."
            )
        elif capped and cache_layout_max_slots == 1:
            reason = (
                "Llama 4 chunked attention is serialized by the current runtime, "
                "so requested concurrency was capped to one session."
            )
        elif capped:
            reason = (
                f"Requested concurrency {requested_session_slots} was capped to "
                f"{reserved_slots}. At {per_session:,} KV bytes per session, the "
                f"binding {binding_source} allows at most {max_slots} session "
                "slots after weights and other fixed allocations. Lower the "
                "context window or free memory to restore the requested concurrency."
            )
        else:
            reason = None

    fixed_kv = per_session * reserved_slots
    pool_scratch = pooling_undo * reserved_slots
    if reserved_slots > 1:
        pool_scratch += row_scratch
    elif reserved_slots == 1:
        pool_scratch += mandatory_scratch
    estimated_total = base + fixed_kv + pool_scratch
    remaining = None if binding is None else binding - estimated_total
    provenance = (
        f"weights: caller supplied ({weights_bytes} bytes)",
        f"cache: config-driven estimate from {path / 'config.json'}",
        f"other fixed: caller supplied ({other_fixed_bytes} bytes)",
        f"fixed-pool workspace and rollback: allocator derived ({pool_scratch} bytes)",
        "available memory is a volatile snapshot, not a reservation guarantee",
    )
    return ModelMemoryPlan(
        model_path=str(path),
        model_type=str(config.get("model_type", "")),
        model_context_limit=(_context_limit(config) or None),
        context_window=context_window,
        weights_bytes=weights_bytes,
        other_fixed_bytes=other_fixed_bytes + pool_scratch,
        cache_tensors=tensors_tuple,
        per_session_kv_bytes=per_session,
        pool_scratch_bytes=pool_scratch,
        requested_session_slots=requested_session_slots,
        reserved_session_slots=reserved_slots,
        max_feasible_session_slots=max_slots,
        cache_layout_max_session_slots=cache_layout_max_slots,
        fixed_kv_cache_bytes=fixed_kv,
        estimated_total_bytes=estimated_total,
        detected_unified_memory_bytes=system.total,
        detected_unified_memory_source=system.total_source,
        available_memory_bytes=system.available,
        available_memory_source=system.available_source,
        memory_ceiling_bytes=memory_ceiling_bytes,
        memory_ceiling_source=(
            "caller supplied Metal/admission launch memory ceiling"
            if memory_ceiling_bytes is not None
            else None
        ),
        binding_memory_bytes=binding,
        binding_memory_source=binding_source,
        remaining_memory_bytes=remaining,
        fits=fits,
        requested_configuration_fits=requested_fits,
        configured_concurrency_capped=capped,
        fit_reason=reason,
        provenance=provenance,
    )


def _array_shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None or isinstance(value, (str, bytes, bytearray)):
        return None
    try:
        result = tuple(int(dimension) for dimension in shape)
    except (TypeError, ValueError):
        return None
    return result if all(dimension >= 0 for dimension in result) else None


def _array_dtype(value: Any) -> tuple[str, int]:
    dtype = getattr(value, "dtype", None)
    name = str(dtype) if dtype is not None else "unknown"
    itemsize = getattr(dtype, "itemsize", None)
    if itemsize is None:
        itemsize = getattr(value, "itemsize", None)
    if itemsize is None:
        size = 0
    else:
        try:
            size = int(itemsize)
        except (TypeError, ValueError):
            size = 0
    return name, max(size, 0)


def build_cache_manifest(
    cache_tree: Any, *, provenance: str = "materialized live-cache probe"
) -> tuple[CacheTensorDescriptor, ...]:
    """Describe concrete backing arrays in a live cache tree.

    The helper is deliberately duck typed. Tests and offline tooling can use
    simple objects with ``shape``, ``dtype``, and ``nbytes`` attributes. It
    prefers backing attributes over ``state`` so logical slices do not hide
    over-allocation.
    """

    descriptors: list[CacheTensorDescriptor] = []
    seen_objects: set[int] = set()
    seen_arrays: set[int] = set()
    backing_names = (
        "keys",
        "values",
        "left_padding",
        "lengths",
        "_pool_buf",
        "buf_kv",
        "buf_gate",
        "prev_win_kv",
        "prev_win_gate",
    )

    def add_array(
        value: Any, path: str, cache_kind: str, role: str, owner: Any
    ) -> None:
        shape = _array_shape(value)
        if shape is None or id(value) in seen_arrays:
            return
        seen_arrays.add(id(value))
        dtype, dtype_bytes = _array_dtype(value)
        raw_nbytes = getattr(value, "nbytes", None)
        if raw_nbytes is None:
            nbytes = _shape_bytes(shape, dtype_bytes)
        else:
            try:
                nbytes = int(raw_nbytes)
            except (TypeError, ValueError):
                nbytes = _shape_bytes(shape, dtype_bytes)
        if dtype_bytes <= 0 and math.prod(shape) > 0:
            raise FixedKVPlanningError(
                f"Live cache tensor {path} does not expose a usable dtype item size."
            )

        lower_kind = cache_kind.lower()
        lower_role = role.lower()
        if "rotating" in lower_kind:
            capacity_kind = "rotating"
        elif "chunked" in lower_kind:
            capacity_kind = "chunked"
        elif "pool" in lower_kind:
            capacity_kind = "pooled" if "pool" in lower_role else "fixed_state"
        elif "array" in lower_kind:
            capacity_kind = "fixed_state"
        else:
            capacity_kind = "linear"

        token_axis: int | None = None
        if lower_role in {"keys", "values"} and len(shape) >= 3:
            token_axis = 2
        elif role == "_pool_buf" and len(shape) >= 2:
            token_axis = 1
        physical = shape[token_axis] if token_axis is not None else None
        if role == "_pool_buf":
            offset = getattr(owner, "_pool_len", getattr(owner, "_pool_extent", None))
        else:
            offset = getattr(owner, "offset", None)
        try:
            logical = int(offset) if offset is not None else physical
        except (TypeError, ValueError):
            logical = physical
        descriptors.append(
            CacheTensorDescriptor(
                path=path,
                cache_kind=cache_kind,
                role=role,
                shape=shape,
                dtype=dtype,
                dtype_bytes=dtype_bytes,
                nbytes=nbytes,
                logical_tokens=logical if token_axis is not None else None,
                physical_tokens=physical,
                capacity_kind=capacity_kind,
                provenance=provenance,
                notes=None,
            )
        )

    def walk(value: Any, path: str, owner_kind: str = "cache") -> None:
        if value is None:
            return
        shape = _array_shape(value)
        if shape is not None:
            add_array(value, path, owner_kind, path.rsplit(".", 1)[-1], value)
            return
        if isinstance(value, Mapping):
            for key in sorted(value, key=str):
                walk(value[key], f"{path}.{key}", owner_kind)
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(item, f"{path}.{index}", owner_kind)
            return
        if not hasattr(value, "__dict__") or id(value) in seen_objects:
            return
        seen_objects.add(id(value))
        kind = type(value).__name__
        found_backing = False
        caches = getattr(value, "caches", None)
        if isinstance(caches, (list, tuple)):
            found_backing = True
            for index, item in enumerate(caches):
                walk(item, f"{path}.caches.{index}", kind)
        array_cache = getattr(value, "cache", None)
        if isinstance(array_cache, (list, tuple)):
            found_backing = True
            for index, item in enumerate(array_cache):
                walk(item, f"{path}.cache.{index}", kind)
        for name in backing_names:
            item = getattr(value, name, None)
            if _array_shape(item) is not None:
                found_backing = True
                add_array(item, f"{path}.{name}", kind, name, value)
        handled = {"caches", "cache", *backing_names}
        for name, item in sorted(vars(value).items()):
            if name in handled:
                continue
            if _array_shape(item) is not None:
                found_backing = True
                add_array(item, f"{path}.{name}", kind, name, value)
            elif isinstance(item, (list, tuple, Mapping)):
                before = len(descriptors)
                walk(item, f"{path}.{name}", kind)
                found_backing = found_backing or len(descriptors) > before
        if not found_backing:
            try:
                state = value.state
            except Exception:
                state = None
            if state is not None:
                walk(state, f"{path}.state", kind)

    walk(cache_tree, "cache")
    if not descriptors:
        raise FixedKVPlanningError(
            "The live cache probe exposed no concrete arrays. Run one model step "
            "and materialize the cache before building its manifest."
        )
    return tuple(descriptors)


__all__ = [
    "CacheTensorDescriptor",
    "FixedKVPlanningError",
    "ModelMemoryPlan",
    "build_cache_manifest",
    "estimate_cache_tensors_from_config",
    "estimate_model_memory",
]
