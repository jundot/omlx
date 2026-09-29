# SPDX-License-Identifier: Apache-2.0
"""Repair generic JSON tool-parser labels contradicted by a model's template."""

import importlib
import logging
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger(__name__)


def repair_tool_parser(tokenizer: Any) -> str | None:
    """Reconcile MLX-LM's generic parser with a clearly recognized template.

    Keep specific/custom parsers and ambiguous templates unchanged. Resolve and
    encode every replacement field before mutating the wrapper. Model files and
    the underlying tokenizer configuration are never modified.
    """
    from mlx_lm.tokenizer_utils import TokenizerWrapper, _infer_tool_parser

    if not isinstance(tokenizer, TokenizerWrapper):
        return None
    parser = tokenizer.tool_parser
    if getattr(parser, "__module__", None) != "mlx_lm.tool_parsers.json_tools":
        return None
    if getattr(tokenizer, "_chat_template", None) is not None:
        return None
    template = tokenizer.chat_template
    if not isinstance(template, str) or not template:
        return None
    # Use template evidence only: vocabulary tokens alone do not establish
    # the grammar that this template instructs the model to emit.
    inferred = _infer_tool_parser(
        SimpleNamespace(chat_template=template, get_vocab=lambda: {})
    )
    if inferred in (None, "json_tools"):
        return None
    module = importlib.import_module(f"mlx_lm.tool_parsers.{inferred}")
    start, end = module.tool_call_start, module.tool_call_end
    start_tokens = tuple(tokenizer.encode(start, add_special_tokens=False))
    end_tokens = tuple(tokenizer.encode(end, add_special_tokens=False))
    if not start_tokens or (end and not end_tokens):
        raise ValueError("nonempty tool-call markers must encode to nonempty tokens")
    tokenizer._tool_parser = module.parse_tool_call
    tokenizer._tool_call_start = start
    tokenizer._tool_call_end = end
    tokenizer._tool_call_start_tokens = start_tokens
    tokenizer._tool_call_end_tokens = end_tokens
    logger.warning(
        "Tool parser json_tools conflicts with the chat template; using %s in memory",
        inferred,
    )
    return inferred
