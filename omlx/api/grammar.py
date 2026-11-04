# SPDX-License-Identifier: Apache-2.0
"""Grammar-constrained decoding via xgrammar.

Provides a logits processor that enforces grammar constraints by masking
invalid tokens at sampling time.  Follows the same ``__call__(tokens, logits)``
interface used by :class:`ThinkingBudgetProcessor`.

Phase-awareness (thinking vs. output) is handled by the *grammar itself*
via xgrammar's structural tag API, not by this processor.  For thinking
models the grammar is compiled as a ``sequence`` of
``[tag(<think>, any_text, </think>), constrained_schema]`` so that the
bitmask is permissive during reasoning and constrained during output.
This keeps the processor simple and enables uniform batched bitmask
computation (parallel model forward || bitmask fill).

The processor supports two usage modes:

1. **Per-request** (original): call ``processor(tokens, logits)`` directly.
   Handles accept + bitmask fill + mask application in one call.

2. **Batched**: call ``processor.advance(tokens)`` to accept the previous
   token, then use ``BatchGrammarMatcher.batch_fill_next_token_bitmask``
   with the exposed ``matcher`` property to fill bitmasks in parallel
   across the batch, and apply the combined bitmask externally.
"""

import logging
import re
from typing import List, Optional

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


def with_tool_call_alternative(output_format: dict, tokenizer, tools=None) -> dict:
    """Allow a tool response independently of the final answer's format.

    Prefer xgrammar's tool formats for the loaded parser. Other paired
    protocols retain parser validation, including literal markers in arguments.
    Never infer a protocol from the model name or the reasoning parser.
    """
    begin = getattr(tokenizer, "tool_call_start", None)
    end = getattr(tokenizer, "tool_call_end", None)
    if not isinstance(begin, str) or not begin or not isinstance(end, str):
        raise ValueError(
            "Cannot combine tools and structured output without tool-call delimiters"
        )

    parser_module = getattr(getattr(tokenizer, "tool_parser", None), "__module__", "")
    parser = (
        parser_module.rsplit(".", 1)[-1]
        if parser_module.startswith(("mlx_lm.tool_parsers.", "mlx_vlm.tools.parsers."))
        else None
    )
    # Only formats whose disabled-reasoning builtin is a standalone call can
    # be composed here. Others retain their tokenizer's envelope below.
    builtin = {
        "qwen3_coder": "qwen_3_5",
        "glm47": "glm_4_7",
        "kimi_k2": "kimi",
    }.get(parser)
    if tools and builtin:
        from .._torch_stub import install as _install_torch_stub

        _install_torch_stub()
        import xgrammar as xgr

        tool_format = xgr.get_model_structural_tag(
            builtin,
            tools=tools,
            tool_choice="required",
            reasoning=False,
            max_whitespace_cnt=32,
        ).model_dump()["format"]
        # The final-answer alternative owns text output. Keep only actual calls
        # from a builtin that otherwise permits prose around its triggers.
        if tool_format["type"] == "triggered_tags":
            tool_format = {"type": "or", "elements": tool_format["tags"]}
    elif tools and (parser == "json_tools" or (begin == "[TOOL_CALLS]" and not end)):
        calls = []
        headers = []
        for tool in tools:
            function = tool["function"]
            name = function["name"]
            parameters = function.get("parameters")
            if parameters is None or function.get("strict") is False:
                parameters = {"type": "object"}
            calls.append(
                {
                    "type": "object",
                    "properties": {"name": {"const": name}, "arguments": parameters},
                    "required": ["name", "arguments"],
                }
            )
            if not end:
                headers.append(
                    {
                        "type": "tag",
                        "begin": f"{begin}{name}[ARGS]",
                        "content": {
                            "type": "json_schema",
                            "json_schema": parameters,
                            "max_whitespace_cnt": 32,
                        },
                        "end": "",
                    }
                )
        call_schema = {"anyOf": calls}
        json_call = {
            "type": "tag",
            "begin": begin,
            "content": {
                "type": "json_schema",
                "json_schema": (
                    {"type": "array", "items": call_schema, "minItems": 1}
                    if not end
                    else {
                        "anyOf": [
                            call_schema,
                            {"type": "array", "items": call_schema, "minItems": 1},
                        ]
                    }
                ),
                "max_whitespace_cnt": 32,
            },
            "end": end,
        }
        tool_format = {"type": "or", "elements": [json_call, *headers]}
    elif end:
        # AnyText inside a Tag eagerly consumes the first closing marker,
        # even when it belongs to a quoted argument. A regular language for
        # the complete envelope preserves such payloads for the output parser.
        tool_format = {
            "type": "regex",
            "pattern": re.escape(begin) + r"[\s\S]*" + re.escape(end),
        }
    else:
        raise ValueError("Unsupported one-sided tool-call protocol")

    whitespace = {"type": "regex", "pattern": r"[ \t\r\n]{0,32}"}
    tool_calls = {
        "type": "sequence",
        "elements": [
            whitespace,
            {
                "type": "plus",
                "content": {
                    "type": "sequence",
                    "elements": [
                        tool_format,
                        whitespace,
                    ],
                },
            },
        ],
    }
    if output_format["type"] == "json_schema":
        # JSON permits surrounding whitespace. It must remain a valid prefix
        # for the final answer too, or a newline commits decoding to tools.
        output_format = {
            "type": "sequence",
            "elements": [
                whitespace,
                {**output_format, "max_whitespace_cnt": 32},
                whitespace,
            ],
        }
    return {"type": "or", "elements": [output_format, tool_calls]}


def create_grammar_compiler(tokenizer, model, *, cache_limit_bytes=-1):
    """Create an xgrammar GrammarCompiler for the given tokenizer and model.

    Returns None if vocab_size cannot be determined.
    """
    from .._torch_stub import install as _install_torch_stub
    _install_torch_stub()
    import xgrammar as xgr

    from ..utils.tokenizer import resolve_vocab_size, unwrap_tokenizer

    hf_tokenizer = unwrap_tokenizer(tokenizer)
    vocab_size = resolve_vocab_size(model)
    kwargs = {}
    if vocab_size is not None:
        kwargs["vocab_size"] = vocab_size

    tokenizer_info = xgr.TokenizerInfo.from_huggingface(hf_tokenizer, **kwargs)
    return xgr.GrammarCompiler(tokenizer_info, cache_limit_bytes=cache_limit_bytes)


def mark_grammar_thinking_phase(
    compiled_grammar, *, enabled: bool, optional: bool = False
):
    """Record whether oMLX compiled a separate reasoning phase.

    Metadata travels with the Python CompiledGrammar wrapper through every
    engine path. Bare or externally compiled grammars are conservative by
    default: their accepted tokens need not include a forced thinking close.
    An optional phase is opened by the model, not by the prompt, so a thinking
    budget must wait for the opener before it counts.
    """
    from .._torch_stub import install as _install_torch_stub

    _install_torch_stub()
    import xgrammar as xgr

    if isinstance(compiled_grammar, xgr.CompiledGrammar):
        compiled_grammar._omlx_has_thinking_phase = enabled
        compiled_grammar._omlx_thinking_phase_optional = optional
    return compiled_grammar


class GrammarConstraintProcessor:
    """Logits processor that enforces grammar constraints via xgrammar bitmask.

    Args:
        compiled_grammar: An ``xgrammar.CompiledGrammar`` instance.  For
            thinking models this should already encode the thinking phase
            (compiled from a structural tag).
        vocab_size: Model vocabulary size (from model config, not tokenizer).
    """

    def __init__(self, compiled_grammar, vocab_size: int):
        from .._torch_stub import install as _install_torch_stub
        _install_torch_stub()
        import xgrammar as xgr
        from xgrammar.kernels.apply_token_bitmask_mlx import apply_token_bitmask_mlx

        self._matcher = xgr.GrammarMatcher(compiled_grammar)
        self._vocab_size = vocab_size
        self._apply_mask = apply_token_bitmask_mlx

        bitmask_width = (vocab_size + 31) // 32
        self._bitmask = np.full((1, bitmask_width), -1, dtype=np.int32)
        self._terminated = False
        self._first_call = True
        self._pending = False

    # ------------------------------------------------------------------
    # Per-request mode (original interface)
    # ------------------------------------------------------------------

    def __call__(self, tokens, logits: mx.array) -> mx.array:
        """Fill bitmask and apply to logits.

        Accept is handled by the monkey-patched GenerationBatch._step(),
        which is the only place that can see which token was sampled from
        the result.  This method only fills the bitmask, applies it, and
        records that a token is now in flight for this row (see
        :attr:`pending`).
        """
        if self._terminated:
            return logits

        self._pending = True
        self._bitmask.fill(-1)
        self._matcher.fill_next_token_bitmask(self._bitmask)

        mx_bitmask = mx.array(self._bitmask)
        return self._apply_mask(mx_bitmask, logits, self._vocab_size)

    def discard_pending(self) -> None:
        """Forget the in-flight token; its row was dropped before it was sent."""
        self._pending = False

    def accept_token(self, token_id: int) -> None:
        """Accept a generated token to advance matcher state."""
        self._pending = False
        if self._terminated:
            return
        if not self._matcher.accept_token(token_id):
            logger.warning("GrammarMatcher rejected token %d", token_id)
        if self._matcher.is_terminated():
            self._terminated = True

    @property
    def pending(self) -> bool:
        """True when a token was sampled for this row and not yet accepted.

        The flag lives on the processor rather than on the generation batch
        because ``GenerationBatch.extend()`` merges rows primed by a
        *different* batch instance, each with its own token already sampled;
        only per-row state survives that merge and the matching ``filter()``.
        """
        return self._pending

    # ------------------------------------------------------------------
    # Batched mode helpers
    # ------------------------------------------------------------------

    @property
    def matcher(self):
        """Return the underlying ``xgrammar.GrammarMatcher``."""
        return self._matcher

    @property
    def is_terminated(self) -> bool:
        return self._terminated

    def advance(self, tokens: mx.array) -> bool:
        """Accept the previous token and advance grammar state.

        Call this *instead of* ``__call__`` when using batched bitmask
        filling.  Returns ``True`` if the matcher is still active (not
        terminated) and should participate in the next
        ``batch_fill_next_token_bitmask`` call.
        """
        if self._terminated:
            return False

        if self._first_call:
            self._first_call = False
        elif len(tokens) > 0:
            last_token = int(tokens[-1])
            if not self._matcher.accept_token(last_token):
                logger.warning("GrammarMatcher rejected token %d", last_token)
            if self._matcher.is_terminated():
                self._terminated = True
                return False

        return True
