"""Verified model contracts composed from cache and compute capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .geometry import PrefillModelGeometry
from .models import MODEL_ADAPTERS
from .models.common import execution_fallback


@dataclass(frozen=True)
class PrefillModelCapabilities:
    """One verified execution contract; model names alone never grant support."""

    geometry: PrefillModelGeometry
    cache_types: tuple[type, ...]


def inspect_prefill_model(model: Any) -> tuple[PrefillModelCapabilities | None, str]:
    """Resolve supported runtime classes and verify their live post-load structure.

    Model support is independent of enabling batching. New families must prove
    their cache/decode contract and workspace bounds before entering this table.
    """
    try:
        return _inspect_prefill_model(model)
    except ImportError:
        return None, "unsupported_model"
    except (AttributeError, TypeError, ValueError):
        return None, "unsupported_geometry"


def _inspect_prefill_model(model: Any) -> tuple[PrefillModelCapabilities | None, str]:
    import importlib

    from mlx_lm.models.cache import KVCache

    adapter = MODEL_ADAPTERS.get(
        getattr(getattr(model, "args", None), "model_type", None)
    )
    if adapter is None or type(model).__module__ != adapter.module_name:
        return None, "unsupported_model"
    module = importlib.import_module(adapter.module_name)
    if type(model) is not module.Model:
        return None, "unsupported_model"
    parts = (model, model.model)
    reason = execution_fallback(parts)
    if reason:
        return None, reason
    if any(getattr(part, "_uses_mrope", False) for part in parts):
        return None, "unsupported_position_state"
    scaling = getattr(model.args, "rope_scaling", None) or {}
    if (scaling.get("type") or scaling.get("rope_type")) == "mrope":
        return None, "unsupported_position_state"
    if any(
        kind != "full_attention"
        for kind in (getattr(model.args, "layer_types", None) or ())
    ):
        return None, "unsupported_attention"
    geometry = adapter.geometry_from_args(model.args)
    if geometry is None:
        return None, "unsupported_geometry"
    reason = adapter.validate(model, module, geometry)
    if reason:
        return None, reason
    return (
        PrefillModelCapabilities(geometry, (KVCache,) * geometry.num_layers),
        "supported",
    )
