# SPDX-License-Identifier: Apache-2.0
"""Laguna XS.2 monkey-patch for mlx-lm.

Brings ml-explore/mlx-lm#1223 into oMLX without modifying the pinned
mlx-lm package. The upstream change adds ``mlx_lm.models.laguna`` as a
text-only model for the Laguna XS.2 architecture.

MLX-LM resolves model architectures and tokenizer-configured tool parsers by
importing modules under its own namespace. Until the upstream module ships, we
register the vendored files under those names so the normal loader path remains
unchanged.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

PR_HEAD_SHA = "0857ee1cf1f4ba7c43e73b836309d9c884529ca8"
PR_URL = "https://github.com/ml-explore/mlx-lm/pull/1223"

_APPLIED = False


def _register_module(qualname: str, filename: str, package: str) -> None:
    if qualname in sys.modules:
        return

    file_path = Path(__file__).parent / filename
    spec = importlib.util.spec_from_file_location(qualname, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create spec for {qualname} from {file_path}")

    module = importlib.util.module_from_spec(spec)
    module.__package__ = package
    # The loader subsequently imports this fully-qualified MLX-LM name. Insert
    # it before executing so intra-module imports and repeated patch attempts
    # see the same module object.
    sys.modules[qualname] = module
    try:
        spec.loader.exec_module(module)

        parent_module = importlib.import_module(package)
        # Also expose it on the package for ``from mlx_lm.models import laguna``
        # and the equivalent tool-parser import style.
        setattr(parent_module, qualname.rsplit(".", 1)[1], module)
    except BaseException:
        # Do not poison future imports with a partially initialized module.
        # Only remove the object this registration attempt installed.
        if sys.modules.get(qualname) is module:
            sys.modules.pop(qualname)
        raise

    logger.info("Registered %s from %s", qualname, filename)


def _register_tool_parser() -> bool:
    """Register the parser named by Laguna's tokenizer configuration if needed."""
    try:
        importlib.import_module("mlx_lm.tool_parsers.laguna")
    except ModuleNotFoundError as error:
        # Fall back only when this exact upstream module is absent. An import
        # failure from inside an installed upstream parser is actionable and
        # must not be hidden by silently replacing it.
        if error.name != "mlx_lm.tool_parsers.laguna":
            raise
        _register_module(
            "mlx_lm.tool_parsers.laguna",
            "tool_parser.py",
            "mlx_lm.tool_parsers",
        )
        return True
    return False


def apply_laguna_patch() -> bool:
    """Register Laguna support before MLX-LM dynamically loads model assets.

    Returning ``True`` means at least one vendored module was installed. Once
    MLX-LM releases its own modules, this function becomes a no-op.
    """
    global _APPLIED
    if _APPLIED:
        return False

    try:
        importlib.import_module("mlx_lm.models.laguna")
    except ModuleNotFoundError as error:
        if error.name == "mlx_lm":
            logger.debug("mlx_lm not importable - laguna patch skipped")
            return False
        # As above, only a missing Laguna module justifies the vendored
        # fallback; surface errors caused by an installed upstream module.
        if error.name != "mlx_lm.models.laguna":
            raise
        _register_module(
            "mlx_lm.models.laguna",
            "laguna_model.py",
            "mlx_lm.models",
        )
        applied = True
    else:
        applied = False

    if _register_tool_parser():
        applied = True

    _APPLIED = True
    if applied:
        logger.info("Laguna mlx-lm patch applied (PR 1223 head %s)", PR_HEAD_SHA[:8])
        return True

    logger.debug("mlx_lm.models.laguna already available upstream")
    return False


def is_applied() -> bool:
    return _APPLIED


__all__ = ["apply_laguna_patch", "is_applied", "PR_HEAD_SHA", "PR_URL"]
