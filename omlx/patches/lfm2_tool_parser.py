"""LFM2/LFM2.5 tool parser compatibility patch for mlx-lm.

Some LFM2/LFM2.5 chat templates emit Pythonic tool calls using:

    <|tool_call_start|>[function_name(arg='value')]<|tool_call_end|>

The pinned mlx-lm version may only infer the ``pythonic`` parser when the
chat template contains ``<|tool_list_start|>``. As a result, valid model
tool calls can be returned as plain assistant content instead of structured
OpenAI-compatible ``tool_calls``.

This patch makes mlx-lm infer the ``pythonic`` parser when both
``<|tool_call_start|>`` and ``<|tool_call_end|>`` are present.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_APPLIED = False


def apply_lfm2_tool_parser_patch() -> bool:
    """Patch ``mlx_lm.tokenizer_utils._infer_tool_parser`` for LFM2/LFM2.5.

    Returns:
        True if the patch is applied or was already applied, False if mlx-lm
        is not available or the expected function cannot be found.
    """

    global _APPLIED

    if _APPLIED:
        return True

    try:
        import mlx_lm.tokenizer_utils as tokenizer_utils
    except Exception as exc:
        logger.debug(
            "mlx_lm.tokenizer_utils not available; skipping LFM2 tool parser patch: %s",
            exc,
        )
        return False

    original = getattr(tokenizer_utils, "_infer_tool_parser", None)
    if original is None:
        logger.debug(
            "mlx_lm.tokenizer_utils._infer_tool_parser not found; "
            "skipping LFM2 tool parser patch"
        )
        return False

    if getattr(original, "_omlx_lfm2_tool_parser_patched", False):
        _APPLIED = True
        return True

    def patched_infer_tool_parser(chat_template: Any):
        if (
            isinstance(chat_template, str)
            and "<|tool_call_start|>" in chat_template
            and "<|tool_call_end|>" in chat_template
        ):
            return "pythonic"

        return original(chat_template)

    patched_infer_tool_parser._omlx_lfm2_tool_parser_patched = True
    patched_infer_tool_parser._omlx_original = original

    tokenizer_utils._infer_tool_parser = patched_infer_tool_parser

    _APPLIED = True
    logger.info("mlx-lm LFM2/LFM2.5 pythonic tool parser detection patch applied")
    return True
