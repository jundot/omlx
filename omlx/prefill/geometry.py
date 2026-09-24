"""Model-independent dimensions shared by adapters and memory admission.

This module and the model argument readers deliberately have no MLX imports.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PrefillExpertGeometry:
    """Routed and shared expert workspace, orthogonal to the cache layout."""

    num_experts: int
    experts_per_token: int
    intermediate_size: int
    shared_intermediate_size: int


@dataclass(frozen=True)
class PrefillModelGeometry:
    """Verified attention dimensions with optional expert costs."""

    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    compute_dtype_size: int
    cache_dtype_size: int
    cache_step: int
    experts: PrefillExpertGeometry | None = None


def positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def get_argument(arguments: Any, name: str) -> Any:
    if isinstance(arguments, Mapping):
        return arguments.get(name)
    return getattr(arguments, name, None)


def valid_geometry(geometry: PrefillModelGeometry) -> bool:
    dimensions = (
        geometry.num_layers,
        geometry.num_attention_heads,
        geometry.num_kv_heads,
        geometry.head_dim,
        geometry.hidden_size,
        geometry.intermediate_size,
        geometry.vocab_size,
        geometry.compute_dtype_size,
        geometry.cache_dtype_size,
        geometry.cache_step,
    )
    if geometry.experts is not None:
        experts = geometry.experts
        if not isinstance(experts, PrefillExpertGeometry) or not (
            all(
                positive_int(value)
                for value in (
                    experts.num_experts,
                    experts.experts_per_token,
                    experts.intermediate_size,
                )
            )
            and experts.experts_per_token <= experts.num_experts
            and nonnegative_int(experts.shared_intermediate_size)
        ):
            return False
    return (
        all(positive_int(value) for value in dimensions)
        and geometry.num_attention_heads % geometry.num_kv_heads == 0
        and geometry.compute_dtype_size in (2, 4)
        and geometry.cache_dtype_size in (2, 4)
    )


def make_geometry(
    arguments: Any,
    *,
    head_dim: int,
    kv_heads: int,
    experts: PrefillExpertGeometry | None = None,
    dtype_size: int = 4,
    cache_dtype_size: int | None = None,
    cache_step: int = 256,
) -> PrefillModelGeometry | None:
    """Normalize verified architecture fields; unknown dtypes default to fp32."""
    geometry = PrefillModelGeometry(
        num_layers=get_argument(arguments, "num_hidden_layers"),
        num_attention_heads=get_argument(arguments, "num_attention_heads"),
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        hidden_size=get_argument(arguments, "hidden_size"),
        intermediate_size=get_argument(arguments, "intermediate_size"),
        vocab_size=get_argument(arguments, "vocab_size"),
        compute_dtype_size=dtype_size,
        cache_dtype_size=dtype_size if cache_dtype_size is None else cache_dtype_size,
        cache_step=cache_step,
        experts=experts,
    )
    return geometry if valid_geometry(geometry) else None
