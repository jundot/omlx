# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 ``glm_moe_dsa`` monkey-patch for mlx-lm.

Vendors the oMLX GLM-5.2 optimized mlx-lm model code without modifying the
pinned mlx-lm package. The public module name stays
``mlx_lm.models.glm_moe_dsa`` so mlx-lm's normal model loader can find it, but
the optimized helper modules remain private to ``omlx.patches.glm_moe_dsa`` and
do not replace ``mlx_lm.models.deepseek_v32`` or the shared MoE layers used by
other model families.
"""

from __future__ import annotations

import importlib
import logging
import sys

from .kernels import fast as glm_fast

logger = logging.getLogger(__name__)

PATCH_SOURCE = "mlxlm-glm optimized-final mlx-lm snapshot"
NATIVE_KERNELS_PACKAGE = "omlx.custom_kernels.glm_moe_dsa"

_APPLIED = False


def _missing_fast_symbols() -> list[str]:
    """Return expected fast symbols missing from native or patched MLX runtime."""
    required = (
        "dsa_indexer_scores",
        "dsa_topk_indices",
        "glm_dsa_sparse_mla_attention",
        "glm_dsa_exact_block_attention",
        "glm_dsa_q8_vup_flat",
        "glm_moe_weighted_sum",
    )
    return glm_fast.missing(required)


def _register_module() -> None:
    qualname = "mlx_lm.models.glm_moe_dsa"
    existing = sys.modules.get(qualname)
    if getattr(existing, "_OMLX_GLM_DSA_OPTIMIZED", False):
        module = existing
    else:
        module = importlib.import_module(f"{__name__}.glm_moe_dsa_model")
        module._OMLX_GLM_DSA_OPTIMIZED = True

    sys.modules[qualname] = module

    import mlx_lm.models as models_pkg

    models_pkg.glm_moe_dsa = module
    logger.info("Registered %s from oMLX optimized vendored module", qualname)


def apply_glm_moe_dsa_patch() -> bool:
    """Apply the GLM MoE DSA patch. Idempotent.

    Must run before ``mlx_lm.load()`` imports ``mlx_lm.models.glm_moe_dsa``.

    Returns True when oMLX registered its optimized vendored module, False when
    the patch was already applied or mlx-lm is unavailable.
    """
    global _APPLIED
    if _APPLIED:
        return False

    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        logger.debug("mlx_lm not importable - glm_moe_dsa patch skipped")
        return False

    _register_module()
    from .generate_patch import apply_glm_moe_dsa_generate_patch

    apply_glm_moe_dsa_generate_patch()
    _APPLIED = True
    missing = _missing_fast_symbols()
    if missing:
        native_error = glm_fast.native_import_error()
        logger.warning(
            "GLM MoE DSA optimized patch applied, but fast kernel symbols are "
            "missing from %s and mx.fast: %s. Prefill for glm_moe_dsa models "
            "will run a much slower non-fused fallback with higher memory use "
            "(see issue #2137). Build the native extension for the accelerated "
            "path (pip install with OMLX_WITH_CUSTOM_KERNEL=1; requires the "
            "Metal toolchain) or use an official release build.%s",
            NATIVE_KERNELS_PACKAGE,
            ", ".join(missing),
            f" Native import error: {native_error}" if native_error else "",
        )
    elif glm_fast.native_available():
        logger.info(
            "GLM MoE DSA native kernels available from %s",
            NATIVE_KERNELS_PACKAGE,
        )
    logger.info("GLM MoE DSA optimized mlx-lm patch applied")
    return True


def is_applied() -> bool:
    return _APPLIED


__all__ = [
    "apply_glm_moe_dsa_patch",
    "is_applied",
    "PATCH_SOURCE",
    "NATIVE_KERNELS_PACKAGE",
]
