# SPDX-License-Identifier: Apache-2.0
"""Expert streaming (SSD) patch for MoE models (glm_moe_dsa, deepseek_v4, ...).

Thin façade: the implementation lives in leaf modules — ``settings``
(budget/override policy and schema coercion), ``lifecycle`` (backing
traversal + engine load/teardown) — plus ``model_hooks``/``residency``/
``governor``/``conversion``/``streaming_switch`` for the mechanism
itself. This file only re-exports the package's public surface so
existing consumers keep working.
"""

from __future__ import annotations

import logging
from typing import Any as Any

# Conversion pipeline lives in conversion.py (stacked-key resolution, the
# switch-MLP rewrite, the orchestrator, transition-profile persistence).
# These names stay importable from the package root: tests and engines
# import them from here, and lifecycle.shutdown_expert_streaming resolves
# save_transition_profile through this namespace. conversion.py top-level
# only touches leaf modules, so this cannot cycle.
from .conversion import (
    _resolve_moe_dims as _resolve_moe_dims,
)
from .conversion import (
    _resolve_stacked_key as _resolve_stacked_key,
)
from .conversion import (
    convert_model_to_streaming as convert_model_to_streaming,
)
from .conversion import (
    load_transition_profile as load_transition_profile,
)
from .conversion import (
    save_transition_profile as save_transition_profile,
)
from .lifecycle import (
    _BACKING_HOPS as _BACKING_HOPS,
)
from .lifecycle import (
    _BACKING_WALK_DEPTH as _BACKING_WALK_DEPTH,
)
from .lifecycle import (
    _ENGINE_BACKING_HOPS as _ENGINE_BACKING_HOPS,
)
from .lifecycle import (
    _MODEL_BACKING_HOPS as _MODEL_BACKING_HOPS,
)
from .lifecycle import (
    _backing_holders as _backing_holders,
)
from .lifecycle import (
    _engine_holders as _engine_holders,
)
from .lifecycle import (
    ensure_streaming_backing_or_raise as ensure_streaming_backing_or_raise,
)
from .lifecycle import (
    expert_streaming_summary as expert_streaming_summary,
)
from .lifecycle import (
    find_model_streaming_backing as find_model_streaming_backing,
)
from .lifecycle import (
    find_streaming_backing as find_streaming_backing,
)
from .lifecycle import (
    gate_up_fusion_blocked as gate_up_fusion_blocked,
)
from .lifecycle import (
    log_expert_streaming_summary as log_expert_streaming_summary,
)
from .lifecycle import (
    post_load_offload_pipeline as post_load_offload_pipeline,
)
from .lifecycle import (
    resolve_streaming_backing as resolve_streaming_backing,
)
from .lifecycle import (
    save_expert_pin_profile as save_expert_pin_profile,
)
from .lifecycle import (
    shutdown_expert_streaming as shutdown_expert_streaming,
)
from .lifecycle import (
    streaming_cache_of as streaming_cache_of,
)
from .lifecycle import (
    streaming_offload_load as streaming_offload_load,
)
from .lifecycle import (
    streaming_summary_backing as streaming_summary_backing,
)
from .lifecycle import (
    teardown_expert_streaming as teardown_expert_streaming,
)
from .model_hooks import (
    DEFAULT_PREFIX_TEMPLATES as DEFAULT_PREFIX_TEMPLATES,
    find_moe_container as find_moe_container,
    find_moe_owner as find_moe_owner,
    find_mtp_stages as find_mtp_stages,
    hooks_for as hooks_for,
)
from .residency import (
    SUPPORTED_TYPES,
    load_config_model_type,
    normalize_model_type,
    streaming_owns_model,
)
from .settings import (
    _BUDGET_GIB_ATTRS as _BUDGET_GIB_ATTRS,
)
from .settings import (
    _BUDGET_MIB_ATTRS as _BUDGET_MIB_ATTRS,
)
from .settings import (
    _IO_OVERRIDE_KEYS as _IO_OVERRIDE_KEYS,
)
from .settings import (
    _IO_OVERRIDE_VALIDATORS as _IO_OVERRIDE_VALIDATORS,
)
from .settings import (
    MAX_EXPERT_STREAMING_BUDGET_BYTES as MAX_EXPERT_STREAMING_BUDGET_BYTES,
)
from .settings import (
    STREAMING_SETTING_SCHEMA as STREAMING_SETTING_SCHEMA,
)
from .settings import (
    _auto_budget_bytes as _auto_budget_bytes,
)
from .settings import (
    _budget_is_pinned as _budget_is_pinned,
)
from .settings import (
    _clamp_budget_bytes as _clamp_budget_bytes,
)
from .settings import (
    _dynamic_armed as _dynamic_armed,
)
from .settings import (
    _io_overrides as _io_overrides,
)
from .settings import (
    _prior_usable as _prior_usable,
)
from .settings import (
    is_supported_model_type as is_supported_model_type,
)
from .settings import (
    resolve_budget_bytes as resolve_budget_bytes,
)
from .settings import (
    validate_streaming_setting as validate_streaming_setting,
)

try:
    # Public cache-policy API at the package root. Re-exported here (not
    # in settings) so ``settings._dynamic_armed`` can resolve it through
    # this namespace at call time — tests monkeypatch the package attr.
    from .governor import dynamic_residency_enabled
except Exception:  # pragma: no cover - governor has no mlx dependency
    dynamic_residency_enabled = lambda: False  # type: ignore[assignment]

try:
    # Leaf imports (expert_cache/cache_policies) — the pre-split spelling
    # ``from .streaming_switch import ...`` is equivalent, the leaves are
    # where they live now; the guard keeps mlx-less imports working.
    from .cache_policies import S3FIFOExpertCache
    from .expert_cache import make_expert_cache
except Exception:  # pragma: no cover - these leaves import mlx
    S3FIFOExpertCache = None  # type: ignore[assignment]
    make_expert_cache = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

__all__ = [
    "convert_model_to_streaming",
    "ensure_streaming_backing_or_raise",
    "find_streaming_backing",
    "find_model_streaming_backing",
    "gate_up_fusion_blocked",
    "load_config_model_type",
    "post_load_offload_pipeline",
    "resolve_budget_bytes",
    "save_expert_pin_profile",
    "streaming_owns_model",
    "teardown_expert_streaming",
    "validate_streaming_setting",
    "is_supported_model_type",
    "normalize_model_type",
    "SUPPORTED_TYPES",
]
