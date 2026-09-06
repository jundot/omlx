# SPDX-License-Identifier: Apache-2.0
"""Capability-only seam for optional DeepSeek V4 kernel providers.

This module deliberately does not register a provider or alter model dispatch.
It freezes the contract an experimental DS4 Metal primitive must satisfy before
production code may call it:

* the exact DeepSeek-V4-Flash model geometry is required;
* the provider consumes the existing MLX safetensors representation;
* experiments are off unless the caller explicitly enables them;
* capability failures select the caller's existing MLX implementation; and
* once a provider call begins, its exceptions propagate.  Retrying a fallback
  after Metal work was encoded could duplicate cache mutation or split JACCL
  collective ordering.

The provider probe must therefore be side-effect free.  Model patches remain
responsible for their operation-specific dtype, shape, and symbol gates.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Generic, Protocol, TypeVar, runtime_checkable

# This is the published DeepSeek-V4-Flash-0731 structure exercised by the
# current oMLX patch and the isolated routed-MoE prototype. Keep it deliberately
# stricter than ``model_type.startswith("deepseek_v4")``: kernels specialized
# for Flash must not silently run on PRO or a future V4 variant with similar
# headline dimensions.
DS4_FLASH_FINGERPRINT: tuple[tuple[str, object], ...] = (
    ("architectures", ("DeepseekV4ForCausalLM",)),
    ("model_type", "deepseek_v4"),
    ("vocab_size", 129_280),
    ("hidden_size", 4_096),
    ("moe_intermediate_size", 2_048),
    ("num_hidden_layers", 43),
    ("num_attention_heads", 64),
    ("num_key_value_heads", 1),
    ("head_dim", 512),
    ("qk_rope_head_dim", 64),
    ("q_lora_rank", 1_024),
    ("o_lora_rank", 1_024),
    ("n_shared_experts", 1),
    ("n_routed_experts", 256),
    ("num_experts_per_tok", 6),
    ("num_hash_layers", 3),
    ("num_nextn_predict_layers", 1),
    ("hc_mult", 4),
    ("hc_eps", 1e-6),
    ("hc_sinkhorn_iters", 20),
    ("o_groups", 8),
    ("sliding_window", 128),
    ("max_position_embeddings", 1_048_576),
    ("index_n_heads", 64),
    ("index_head_dim", 128),
    ("index_topk", 512),
    ("swiglu_limit", 10.0),
    ("routed_scaling_factor", 1.5),
    ("scoring_func", "sqrtsoftplus"),
    ("topk_method", "noaux_tc"),
    ("norm_topk_prob", True),
    (
        "compress_ratios",
        (
            0,
            0,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            0,
            0,
            0,
        ),
    ),
    ("dspark_block_size", 5),
    ("dspark_noise_token_id", 128_799),
    ("dspark_target_layer_ids", (40, 41, 42)),
    ("dspark_markov_rank", 256),
)


class DS4CheckpointLayout(str, Enum):  # noqa: UP042 - mypy targets Python 3.10
    """Weight representation accepted by this in-MLX provider seam."""

    MLX_SAFETENSORS = "mlx_safetensors"
    DS4_GGUF = "ds4_gguf"


class DS4ExecutionMode(str, Enum):  # noqa: UP042 - mypy targets Python 3.10
    """Execution contexts in which an MLX-array primitive may be probed."""

    LOCAL = "local"
    JACCL_TENSOR_PARALLEL = "jaccl_tensor_parallel"


class DS4ProviderDecisionCode(str, Enum):  # noqa: UP042 - mypy targets Python 3.10
    """Stable reason codes for provider selection and fallback telemetry."""

    SELECTED = "selected"
    EXPERIMENT_DISABLED = "experiment_disabled"
    MODEL_MISMATCH = "model_mismatch"
    CHECKPOINT_LAYOUT_UNSUPPORTED = "checkpoint_layout_unsupported"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_REJECTED = "provider_rejected"
    PROVIDER_PROBE_ERROR = "provider_probe_error"


@dataclass(frozen=True)
class DS4KernelRequest:
    """Immutable common inputs to a DS4 provider capability probe.

    Operation-specific tensor metadata belongs in ``metadata``.  Keeping the
    core request free of MLX arrays makes it cheap and safe to unit test and
    ensures capability probing cannot accidentally encode GPU work.
    """

    operation: str
    model_config: Mapping[str, object]
    checkpoint_layout: DS4CheckpointLayout
    execution_mode: DS4ExecutionMode
    experimental_enabled: bool = False
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.operation, str) or not self.operation.strip():
            raise ValueError("DS4 kernel operation must be a non-empty string")
        if not isinstance(self.model_config, Mapping):
            raise TypeError("DS4 model_config must be a mapping")
        if not isinstance(self.checkpoint_layout, DS4CheckpointLayout):
            raise TypeError("checkpoint_layout must be DS4CheckpointLayout")
        if not isinstance(self.execution_mode, DS4ExecutionMode):
            raise TypeError("execution_mode must be DS4ExecutionMode")
        if not isinstance(self.experimental_enabled, bool):
            raise TypeError("experimental_enabled must be bool")
        if self.metadata is not None and not isinstance(self.metadata, Mapping):
            raise TypeError("DS4 provider metadata must be a mapping or None")


@dataclass(frozen=True)
class DS4ProviderSupport:
    """Result returned by a provider's pure capability probe."""

    supported: bool
    detail: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.supported, bool):
            raise TypeError("provider support flag must be bool")
        if not isinstance(self.detail, str):
            raise TypeError("provider support detail must be str")


@runtime_checkable
class DS4KernelProvider(Protocol):
    """Minimal protocol implemented by an optional DS4 kernel provider."""

    name: str

    def probe(self, request: DS4KernelRequest) -> DS4ProviderSupport:
        """Return support without allocating tensors or encoding GPU work."""


@dataclass(frozen=True)
class DS4ProviderCapability:
    """Resolved provider decision passed to the strict dispatcher."""

    selected: bool
    code: DS4ProviderDecisionCode
    operation: str
    provider_name: str | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        if self.selected != (self.code is DS4ProviderDecisionCode.SELECTED):
            raise ValueError("selected must agree with the provider decision code")


T = TypeVar("T")


@dataclass(frozen=True)
class DS4DispatchResult(Generic[T]):
    """A dispatch value paired with its observable provider decision."""

    value: T
    capability: DS4ProviderCapability

    @property
    def used_provider(self) -> bool:
        return self.capability.selected


def _same_fingerprint_value(actual: object, expected: object) -> bool:
    # JSON may spell an integral model field as 10 or 10.0.  Treat those as
    # the same value while ensuring booleans never pass as integers.
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            return False
        return float(actual) == float(expected)
    if isinstance(expected, tuple):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            return False
        return all(
            _same_fingerprint_value(actual_value, expected_value)
            for actual_value, expected_value in zip(actual, expected)
        )
    return type(actual) is type(expected) and actual == expected


def ds4_flash_fingerprint_mismatches(
    config: Mapping[str, object],
) -> tuple[str, ...]:
    """Return the required DS4-Flash fields absent or different in ``config``."""

    if not isinstance(config, Mapping):
        return tuple(key for key, _expected in DS4_FLASH_FINGERPRINT)
    return tuple(
        key
        for key, expected in DS4_FLASH_FINGERPRINT
        if key not in config or not _same_fingerprint_value(config[key], expected)
    )


def is_exact_ds4_flash_config(config: Mapping[str, object]) -> bool:
    """Whether ``config`` matches the complete provider-level fingerprint."""

    return not ds4_flash_fingerprint_mismatches(config)


def _decision(
    request: DS4KernelRequest,
    code: DS4ProviderDecisionCode,
    *,
    provider_name: str | None = None,
    detail: str = "",
) -> DS4ProviderCapability:
    return DS4ProviderCapability(
        selected=code is DS4ProviderDecisionCode.SELECTED,
        code=code,
        operation=request.operation,
        provider_name=provider_name,
        detail=detail,
    )


def resolve_ds4_provider(
    provider: DS4KernelProvider | None,
    request: DS4KernelRequest,
) -> DS4ProviderCapability:
    """Resolve a provider without raising for an optional capability failure.

    The common safety gates run before provider code.  A broken optional probe
    becomes an ordinary fallback decision; ``KeyboardInterrupt`` and other
    ``BaseException`` subclasses are intentionally not swallowed.
    """

    if not request.experimental_enabled:
        return _decision(request, DS4ProviderDecisionCode.EXPERIMENT_DISABLED)

    mismatches = ds4_flash_fingerprint_mismatches(request.model_config)
    if mismatches:
        return _decision(
            request,
            DS4ProviderDecisionCode.MODEL_MISMATCH,
            detail=",".join(mismatches),
        )

    if request.checkpoint_layout is not DS4CheckpointLayout.MLX_SAFETENSORS:
        return _decision(
            request,
            DS4ProviderDecisionCode.CHECKPOINT_LAYOUT_UNSUPPORTED,
            detail=request.checkpoint_layout.value,
        )

    if provider is None:
        return _decision(request, DS4ProviderDecisionCode.PROVIDER_UNAVAILABLE)

    try:
        provider_name = provider.name
        if not isinstance(provider_name, str) or not provider_name.strip():
            raise TypeError("provider name must be a non-empty string")
        support = provider.probe(request)
        if not isinstance(support, DS4ProviderSupport):
            raise TypeError("provider probe returned an invalid result")
    except Exception as exc:  # noqa: BLE001 - optional probe must fail soft
        return _decision(
            request,
            DS4ProviderDecisionCode.PROVIDER_PROBE_ERROR,
            provider_name=(
                provider_name
                if "provider_name" in locals() and isinstance(provider_name, str)
                else None
            ),
            detail=f"{type(exc).__name__}: {exc}",
        )

    if not support.supported:
        return _decision(
            request,
            DS4ProviderDecisionCode.PROVIDER_REJECTED,
            provider_name=provider_name,
            detail=support.detail,
        )

    return _decision(
        request,
        DS4ProviderDecisionCode.SELECTED,
        provider_name=provider_name,
        detail=support.detail,
    )


def dispatch_ds4_kernel(
    provider: DS4KernelProvider | None,
    request: DS4KernelRequest,
    *,
    provider_call: Callable[[], T],
    fallback: Callable[[], T],
) -> DS4DispatchResult[T]:
    """Call exactly one implementation according to the capability decision.

    Provider-call exceptions intentionally propagate and do *not* trigger the
    fallback.  A native call may already have encoded work on an MLX stream;
    transparently running the reference afterward is unsafe for stateful model
    operations and distributed collective ordering.
    """

    capability = resolve_ds4_provider(provider, request)
    value = provider_call() if capability.selected else fallback()
    return DS4DispatchResult(value=value, capability=capability)


__all__ = [
    "DS4CheckpointLayout",
    "DS4DispatchResult",
    "DS4ExecutionMode",
    "DS4_FLASH_FINGERPRINT",
    "DS4KernelProvider",
    "DS4KernelRequest",
    "DS4ProviderCapability",
    "DS4ProviderDecisionCode",
    "DS4ProviderSupport",
    "dispatch_ds4_kernel",
    "ds4_flash_fingerprint_mismatches",
    "is_exact_ds4_flash_config",
    "resolve_ds4_provider",
]
