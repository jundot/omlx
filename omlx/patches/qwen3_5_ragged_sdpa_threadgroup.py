# SPDX-License-Identifier: Apache-2.0
"""Guard the qwen3_5 ragged-decode SDPA fast kernel against GPUs whose per-kernel
threadgroup limit is below its hardcoded 1024-thread launch.

``mlx_vlm.models.qwen3_5.language._qwen3_5_ragged_decode_attention`` dispatches a
custom Metal kernel (``_qwen3_5_ragged_sdpa_*``) with ``threadgroup=(1024, 1, 1)``
— 32 simdgroups x 32 lanes (``constexpr int BN = 32; BD = 32``). That 1024 is
STRUCTURAL: the kernel's final cross-simdgroup reduction maps the 32 per-simdgroup
partials onto the 32 lanes of a single simdgroup, so the launch cannot be clamped
below 32*32 without corrupting attention outputs.

On GPUs where that compiled kernel's pipeline ``maxTotalThreadsPerThreadgroup`` is
< 1024 (e.g. an M2 Ultra reports 896 for it, driven by register pressure), EVERY
dispatch hard-crashes:

    RuntimeError: Thread group size (1024) is greater than the maximum allowed
    threads per threadgroup (896).

That 500s any request whose head_dim=256 full-attention layers reach this path —
i.e. essentially all mid-length chat prompts (~1k-8k prompt tokens). It surfaces
async via ``engine_core._raise_request_output_error`` (the model builds the graph
lazily; the kernel error is captured on the RequestOutput at ``mx.async_eval``
time), so the traceback points at the sampling/eval step, not the kernel.

Fix: neutralize the fast kernel so full-attention falls back to MLX's portable
vector SDPA. head_dim 256 IS supported by MLX's fused vector kernel
(``_SDPA_VECTOR_SUPPORTED_HEAD_DIMS`` includes 256) and it respects the per-kernel
threadgroup limit, so the fallback is correct and crash-free. The only cost is the
ragged-batch decode micro-optimization on affected GPUs.

Default: disabled (the ragged kernel is neutralized) — it is fragile by
construction (a hardcoded >warp*32 threadgroup that many Apple GPUs cap below
1024). Set ``OMLX_QWEN35_RAGGED_SDPA_KEEP=1`` to keep the fast kernel on GPUs
where it is known to run (its pipeline supports 1024 threads).
"""

import logging
import os

logger = logging.getLogger(__name__)

_PATCHED = False


def apply_qwen3_5_ragged_sdpa_threadgroup_fix() -> bool:
    """Neutralize the qwen3_5 ragged-decode fast SDPA kernel unless explicitly
    kept, so full-attention falls back to the threadgroup-safe portable SDPA.

    Idempotent; returns True if the neutralizing patch was installed."""
    global _PATCHED
    if _PATCHED:
        return False
    if os.environ.get("OMLX_QWEN35_RAGGED_SDPA_KEEP") == "1":
        logger.info(
            "qwen3_5 ragged SDPA kernel kept (OMLX_QWEN35_RAGGED_SDPA_KEEP=1)"
        )
        return False

    try:
        from mlx_vlm.models.qwen3_5 import language as q35
    except ImportError:
        return False

    if not hasattr(q35, "_qwen3_5_ragged_decode_attention"):
        return False

    def _ragged_decode_disabled(queries, keys, values, pads, scale):
        # Return None -> caller falls back to portable per-pad-group
        # scaled_dot_product_attention (MLX vector kernel, head_dim=256 safe).
        return None

    q35._qwen3_5_ragged_decode_attention = _ragged_decode_disabled
    _PATCHED = True
    logger.info(
        "qwen3_5 ragged SDPA fast kernel disabled (threadgroup 1024 > per-kernel "
        "limit on this GPU); using portable vector SDPA fallback"
    )
    return True
