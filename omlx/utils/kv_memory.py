# SPDX-License-Identifier: Apache-2.0
"""
KV cache memory budget utilities.

Calculates per-token KV cache bytes for Standard/GQA models and converts
user-facing memory budgets into BatchGenerator's ``max_kv_size`` (token count).

Note: ``max_kv_size`` only applies to models **without** ``make_cache()``.
Models with ``make_cache()`` (MLA, hybrid, ArraysCache) already manage their
own bounded cache, and mlx-lm's BatchGenerator ignores ``max_kv_size`` for them.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Minimum max_kv_size to guarantee basic functionality
MIN_MAX_KV_SIZE = 512

# OS / system overhead reserved from total memory (bytes)
_OS_RESERVED_BYTES = 8 * 1024**3  # 8 GB

# Safety margin applied to auto-calculated KV budget
_AUTO_BUDGET_SAFETY_FACTOR = 0.9


# ---------------------------------------------------------------------------
# System memory helpers
# ---------------------------------------------------------------------------


def get_system_available_memory_bytes() -> int:
    """Return usable memory for model + KV cache (system RAM minus OS reserve).

    Returns:
        Available bytes, at least 4 GB.
    """
    from .hardware import get_total_memory_bytes

    total = get_total_memory_bytes()
    available = total - _OS_RESERVED_BYTES
    return max(available, 4 * 1024**3)


# ---------------------------------------------------------------------------
# Per-token KV cache size
# ---------------------------------------------------------------------------


def _get_dtype_bytes(model: Any) -> int:
    """Return byte-width of the model's dtype (2 for float16/bfloat16, 4 for float32)."""
    try:
        import mlx.core as mx

        model_dtype = getattr(model, "dtype", None)
        if model_dtype == mx.float32:
            return 4
    except Exception:
        pass
    return 2  # float16 / bfloat16 default


def _get_model_config(model: Any) -> Optional[Any]:
    """Extract config object from model (tries .config then .args)."""
    config = getattr(model, "config", None)
    if config is not None:
        return config
    return getattr(model, "args", None)


def calculate_per_token_kv_bytes(model: Any) -> Optional[int]:
    """Calculate KV cache bytes consumed per token across all layers.

    Uses the standard formula: ``n_kv_heads × head_dim × dtype_bytes × 2``
    per layer (K + V).  GQA models naturally get a smaller value because
    ``num_key_value_heads < num_attention_heads``.

    Args:
        model: A loaded mlx-lm model instance (without ``make_cache()``).

    Returns:
        Bytes per token, or None if config is incomplete.
    """
    config = _get_model_config(model)
    if config is None:
        return None

    num_layers = getattr(config, "num_hidden_layers", None) or getattr(
        config, "n_layer", None
    )
    if not num_layers:
        return None

    dtype_bytes = _get_dtype_bytes(model)

    num_kv_heads = (
        getattr(config, "num_key_value_heads", None)
        or getattr(config, "num_attention_heads", None)
        or getattr(config, "n_head", None)
    )
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        hidden_size = getattr(config, "hidden_size", None) or getattr(
            config, "n_embd", None
        )
        num_heads = getattr(config, "num_attention_heads", None) or num_kv_heads
        if hidden_size and num_heads:
            head_dim = hidden_size // num_heads

    if not num_kv_heads or not head_dim:
        return None

    # K + V per layer
    per_layer = num_kv_heads * head_dim * dtype_bytes * 2
    return num_layers * per_layer


# ---------------------------------------------------------------------------
# Budget resolution
# ---------------------------------------------------------------------------

_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(GB|MB|TB)\s*$", re.IGNORECASE)


def parse_memory_string(s: str) -> Optional[int]:
    """Parse a human-readable memory string like ``"8GB"`` into bytes.

    Returns:
        Bytes, or None if the string cannot be parsed.
    """
    m = _SIZE_RE.match(s)
    if not m:
        return None
    value = float(m.group(1))
    unit = m.group(2).upper()
    multipliers = {"MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
    return int(value * multipliers[unit])


def resolve_kv_cache_memory_bytes(
    setting: str,
    model_weight_bytes: int = 0,
    available_override: Optional[int] = None,
) -> Optional[int]:
    """Convert a user-facing setting string into a KV cache byte budget.

    Args:
        setting: One of ``"auto"``, ``"disabled"``, or a size like ``"8GB"``.
        model_weight_bytes: Estimated model weight size in bytes (for auto).
        available_override: If set, used instead of system-level detection
            (useful for multi-model scenarios where remaining memory is known).

    Returns:
        Budget in bytes, or None for disabled (no limit).
    """
    setting = setting.strip().lower()

    if setting == "disabled":
        return None

    if setting == "auto":
        if available_override is not None:
            available = available_override
        else:
            available = get_system_available_memory_bytes()
        budget = available - model_weight_bytes
        budget = int(budget * _AUTO_BUDGET_SAFETY_FACTOR)
        return max(budget, 1 * 1024**3)  # at least 1 GB

    # Explicit size
    parsed = parse_memory_string(setting)
    if parsed is not None:
        return parsed

    logger.warning(
        f"Unrecognised max_kv_cache_memory value '{setting}', treating as 'auto'"
    )
    available = available_override or get_system_available_memory_bytes()
    budget = int((available - model_weight_bytes) * _AUTO_BUDGET_SAFETY_FACTOR)
    return max(budget, 1 * 1024**3)


# ---------------------------------------------------------------------------
# Final max_kv_size calculation
# ---------------------------------------------------------------------------


def calculate_max_kv_size(
    model: Any,
    budget_bytes: int,
    max_concurrent: int,
) -> Optional[int]:
    """Convert a KV memory budget into ``max_kv_size`` (tokens per request).

    Args:
        model: A loaded mlx-lm model instance.
        budget_bytes: Total KV cache memory budget in bytes.
        max_concurrent: Maximum concurrent requests (completion_batch_size).

    Returns:
        Token limit per request, at least ``MIN_MAX_KV_SIZE`` (512),
        or None if per-token bytes cannot be determined.
    """
    per_token = calculate_per_token_kv_bytes(model)
    if not per_token or max_concurrent <= 0:
        return None

    per_request_budget = budget_bytes // max(max_concurrent, 1)
    tokens = per_request_budget // per_token
    return max(tokens, MIN_MAX_KV_SIZE)
