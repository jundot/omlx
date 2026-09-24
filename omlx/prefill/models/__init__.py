"""Explicit architecture adapters; importing the registry does not import MLX.

Add an argument reader and live structure validator together. Registering a
name alone never grants batching: capabilities also verify the exact MLX class,
execution flags and actual request caches.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ..geometry import PrefillModelGeometry, get_argument
from . import dense, hy_v3


@dataclass(frozen=True)
class PrefillModelAdapter:
    module_name: str
    geometry_from_args: Callable[..., PrefillModelGeometry | None]
    validate: Callable[[Any, Any, PrefillModelGeometry], str | None]


MODEL_ADAPTERS = MappingProxyType(
    {
        name: PrefillModelAdapter(
            f"mlx_lm.models.{name}", adapter.geometry_from_args, adapter.validate
        )
        for name, adapter in (
            ("llama", dense),
            ("qwen2", dense),
            ("qwen3", dense),
            ("hy_v3", hy_v3),
        )
    }
)


def model_geometry_from_args(
    arguments: Any,
    *,
    dtype_size: int = 4,
    cache_dtype_size: int | None = None,
    cache_step: int = 256,
) -> PrefillModelGeometry | None:
    """Read supported schemas without loading model weights or a GPU runtime."""
    model_type = get_argument(arguments, "model_type")
    if not isinstance(model_type, str):
        return None
    adapter = MODEL_ADAPTERS.get(model_type)
    if adapter is None:
        return None
    return adapter.geometry_from_args(
        arguments,
        dtype_size=dtype_size,
        cache_dtype_size=cache_dtype_size,
        cache_step=cache_step,
    )
