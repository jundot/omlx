# SPDX-License-Identifier: Apache-2.0
"""Register K2 Horizon until mlx-lm provides native support."""

import importlib
import sys

_APPLIED = False


def apply_k2_horizon_patch() -> bool:
    global _APPLIED
    if _APPLIED:
        return False
    for target, source in (
        ("mlx_lm.models.k2_horizon", ".k2_horizon_model"),
        ("mlx_lm.tool_parsers.k2_horizon", ".tool_parser"),
    ):
        try:
            importlib.import_module(target)
        except ModuleNotFoundError as error:
            if error.name != target:
                raise
            module = importlib.import_module(source, __name__)
            sys.modules[target] = module
            package, name = target.rsplit(".", 1)
            setattr(importlib.import_module(package), name, module)
    from .checkpoint import apply_checkpoint_patch

    apply_checkpoint_patch()
    _APPLIED = True
    return True


def validate_chat_template_kwargs(kwargs):
    from ...exceptions import InvalidRequestError

    if kwargs.get("reasoning_effort", "high") not in ("low", "medium", "high"):
        raise InvalidRequestError("K2 reasoning_effort must be low, medium or high.")
    if kwargs.get("enable_thinking") is False:
        raise InvalidRequestError(
            "K2 does not support enable_thinking=false. Use Thinking Budget to limit reasoning."
        )
