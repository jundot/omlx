# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 ``glm_moe_dsa`` monkey-patch for mlx-lm.

Brings ml-explore/mlx-lm#1410 into oMLX without modifying the pinned
mlx-lm package. The upstream change turns the stock bare DeepSeek-V3.2
subclass into a GLM-5.2-aware model with native DSA indexer sharing:
full layers compute top-k indices, shared layers reuse the previous full
layer's top-k, and shared layers carry no indexer KV cache.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

PR_HEAD_SHA = "3cb18b51892a7800678923116f5b4920a7666208"
PR_URL = "https://github.com/ml-explore/mlx-lm/pull/1410"

_APPLIED = False


def _upstream_has_glm_moe_dsa_support() -> bool:
    """Return True when the installed mlx-lm already has PR #1410 support."""
    try:
        module = importlib.import_module("mlx_lm.models.glm_moe_dsa")
    except Exception:
        return False

    fields = getattr(getattr(module, "ModelArgs", None), "__dataclass_fields__", {})
    return "indexer_types" in fields and hasattr(module, "GlmMoeDsaModel")


def _register_module() -> None:
    qualname = "mlx_lm.models.glm_moe_dsa"
    here = Path(__file__).parent
    file_path = here / "glm_moe_dsa_model.py"
    spec = importlib.util.spec_from_file_location(qualname, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create spec for {qualname} from {file_path}")

    module = importlib.util.module_from_spec(spec)
    module.__package__ = "mlx_lm.models"
    sys.modules[qualname] = module
    spec.loader.exec_module(module)

    import mlx_lm.models as models_pkg

    models_pkg.glm_moe_dsa = module
    logger.info("Registered %s from %s", qualname, file_path.name)


def apply_glm_moe_dsa_patch() -> bool:
    """Apply the GLM MoE DSA patch. Idempotent.

    Must run before ``mlx_lm.load()`` imports ``mlx_lm.models.glm_moe_dsa``.

    Returns True when oMLX registered its vendored module, False when the
    patch was already applied, mlx-lm is unavailable, or upstream already
    ships equivalent support.
    """
    global _APPLIED
    if _APPLIED:
        return False

    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        logger.debug("mlx_lm not importable - glm_moe_dsa patch skipped")
        return False

    if _upstream_has_glm_moe_dsa_support():
        _APPLIED = True
        logger.debug("mlx_lm.models.glm_moe_dsa already supports GLM-5.2 sharing")
        return False

    _register_module()
    _APPLIED = True
    logger.info("GLM MoE DSA mlx-lm patch applied (PR 1410 head %s)", PR_HEAD_SHA[:8])
    return True


def is_applied() -> bool:
    return _APPLIED


__all__ = ["apply_glm_moe_dsa_patch", "is_applied", "PR_HEAD_SHA", "PR_URL"]
