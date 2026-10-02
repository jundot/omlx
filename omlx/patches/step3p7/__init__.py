# SPDX-License-Identifier: Apache-2.0
"""Step 3.7 Flash monkey-patch for mlx-lm.

Brings ml-explore/mlx-lm#1325 into oMLX without modifying the pinned
mlx-lm package. The upstream change adds ``mlx_lm.models.step3p7`` as a
text-only wrapper around the existing Step 3.5 language model.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

PR_HEAD_SHA = "7794077a7e51d8ca9721cd15dbc01a38e55bc4f1"
PR_URL = "https://github.com/ml-explore/mlx-lm/pull/1325"

_APPLIED = False


def _register_module() -> None:
    qualname = "mlx_lm.models.step3p7"
    if qualname in sys.modules:
        return

    here = Path(__file__).parent
    file_path = here / "step3p7_model.py"
    spec = importlib.util.spec_from_file_location(qualname, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create spec for {qualname} from {file_path}")

    module = importlib.util.module_from_spec(spec)
    module.__package__ = "mlx_lm.models"
    sys.modules[qualname] = module
    spec.loader.exec_module(module)

    import mlx_lm.models as models_pkg

    models_pkg.step3p7 = module
    logger.info("Registered %s from %s", qualname, file_path.name)


def apply_step3p7_patch() -> bool:
    """Register ``mlx_lm.models.step3p7`` when upstream does not provide it."""
    global _APPLIED
    if _APPLIED:
        return False

    try:
        module = importlib.import_module("mlx_lm.models.step3p7")
    except ImportError:
        try:
            import mlx_lm  # noqa: F401
        except ImportError:
            logger.debug("mlx_lm not importable - step3p7 patch skipped")
            return False
        _register_module()
        _APPLIED = True
        logger.info("Step 3.7 mlx-lm patch applied (PR 1325 head %s)", PR_HEAD_SHA[:8])
        return True

    import mlx_lm.models as models_pkg

    models_pkg.step3p7 = module
    _APPLIED = True
    logger.debug("mlx_lm.models.step3p7 already available upstream")
    return False


def is_applied() -> bool:
    return _APPLIED


__all__ = ["apply_step3p7_patch", "is_applied", "PR_HEAD_SHA", "PR_URL"]
