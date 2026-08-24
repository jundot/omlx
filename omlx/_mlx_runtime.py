# SPDX-License-Identifier: Apache-2.0
"""Early MLX Metal runtime defaults for throughput-oriented serving.

MLX reads these variables while its Metal runtime is initialized, so server
startup must apply them before importing modules that import ``mlx.core``.
Every setting uses ``setdefault``: an operator-provided MLX value always wins.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping

_HIGH_MEMORY_BYTES = 64 * 1024**3
_ULTRA_MEMORY_BYTES = 192 * 1024**3
_FALSE_VALUES = frozenset(("0", "false", "no", "off"))


def _physical_memory_bytes() -> int | None:
    """Return physical memory without importing platform-specific packages."""
    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def configure_mlx_runtime_environment(
    environ: MutableMapping[str, str] | None = None,
    *,
    memory_bytes: int | None = None,
) -> dict[str, str]:
    """Apply validated Metal defaults and return the resulting policy values.

    Fast CPU/GPU synchronization is safe on all supported hosts. Larger command
    buffers are enabled only on systems with at least 64 GiB because they trade
    memory and scheduling latency for decode throughput. Set
    ``OMLX_MLX_RUNTIME_TUNING=0`` to disable the policy entirely, or set any
    individual MLX variable before startup to override that default.
    """
    target = os.environ if environ is None else environ
    enabled = target.get("OMLX_MLX_RUNTIME_TUNING", "auto").strip().lower()
    if enabled in _FALSE_VALUES:
        return {}

    target.setdefault("MLX_METAL_FAST_SYNCH", "1")

    available = _physical_memory_bytes() if memory_bytes is None else memory_bytes
    if available is not None and available >= _HIGH_MEMORY_BYTES:
        target.setdefault("MLX_MAX_MB_PER_BUFFER", "512")
        target.setdefault("MLX_MAX_OPS_PER_BUFFER", "100")
    if available is not None and available >= _ULTRA_MEMORY_BYTES:
        # At 131k/262k, head-dim-256 Qwen prefill otherwise selects the
        # memory-safe tiled SDPA before scheduler headroom is available. Ultra
        # hosts have room for MLX's materially faster unfused score matrix.
        target.setdefault("OMLX_SDPA256_TILED", "0")
        target.setdefault("OMLX_QWEN35_ANE_FUSED_VIEWS", "1")
        target.setdefault("OMLX_SPECPREFILL_DRAFT_CLEAR_EVERY_2", "1")
        target.setdefault("OMLX_SPECPREFILL_DRAFT_PREALLOCATE", "1")
        target.setdefault("OMLX_SPECPREFILL_DRAFT_STEP", "32768")
        target.setdefault("OMLX_SPECPREFILL_KV_STEP", "512")

    keys = (
        "MLX_METAL_FAST_SYNCH",
        "MLX_MAX_MB_PER_BUFFER",
        "MLX_MAX_OPS_PER_BUFFER",
        "OMLX_SDPA256_TILED",
        "OMLX_QWEN35_ANE_FUSED_VIEWS",
        "OMLX_SPECPREFILL_DRAFT_CLEAR_EVERY_2",
        "OMLX_SPECPREFILL_DRAFT_PREALLOCATE",
        "OMLX_SPECPREFILL_DRAFT_STEP",
        "OMLX_SPECPREFILL_KV_STEP",
    )
    return {key: target[key] for key in keys if key in target}


__all__ = ["configure_mlx_runtime_environment"]
