# SPDX-License-Identifier: Apache-2.0
"""Request-owned K2 tool names, compiled by oMLX's existing grammar backend."""

from ...exceptions import InvalidRequestError


def validate_tool_prefix(messages, tools, is_partial):
    if tools and is_partial and messages:
        prefix = str(messages[-1].get("content") or "")
        if prefix.rfind("<ifm|tool_calls>") > prefix.rfind("</ifm|tool_calls>"):
            raise InvalidRequestError(
                "K2 tool-name constraints require a complete assistant tool-call prefix."
            )


def _token(value):
    return {"type": "token", "token": value}


def _sequence(*elements):
    return {"type": "sequence", "elements": list(elements)}


def compile_tool_grammar(compiler, tools, existing=None):
    """Constrain names in native XML/JSON envelopes without inspecting descriptions."""
    if not tools:
        return existing
    if existing is not None:
        raise InvalidRequestError(
            "K2 tool names and structured output cannot be constrained together."
        )
    if compiler is None:
        raise InvalidRequestError(
            "K2 tool-name constraints require xgrammar. Install omlx[grammar]."
        )
    names = sorted({tool["function"]["name"] for tool in tools})
    whitespace = {"type": "grammar", "grammar": r"root ::= [ \t\r\n]*"}
    xml = _sequence(
        whitespace,
        {
            "type": "or",
            "elements": [{"type": "const_string", "value": name} for name in names],
        },
        whitespace,
        {
            "type": "optional",
            "content": _sequence(
                _token("<ifm|arg_key>"),
                {"type": "any_tokens", "exclude_tokens": ["</ifm|tool_call>"]},
            ),
        },
    )
    properties = {
        "name": {"type": "string", "enum": names},
        "arguments": {"type": "object"},
    }
    json_formats = [
        {
            "type": "json_schema",
            "json_schema": {
                "type": "object",
                "properties": {key: properties[key] for key in order},
                "required": ["name", "arguments"],
                "additionalProperties": False,
            },
        }
        for order in (("name", "arguments"), ("arguments", "name"))
    ]
    call = {
        "type": "tag",
        "begin": _token("<ifm|tool_call>"),
        "content": {"type": "or", "elements": [xml, *json_formats]},
        "end": _token("</ifm|tool_call>"),
    }
    group = {
        "type": "tag",
        "begin": _token("<ifm|tool_calls>"),
        "content": _sequence(
            whitespace, {"type": "plus", "content": _sequence(call, whitespace)}
        ),
        "end": _token("</ifm|tool_calls>"),
    }
    return compiler.compile_structural_tag(
        {
            "type": "structural_tag",
            "format": {
                "type": "token_triggered_tags",
                "trigger_tokens": ["<ifm|tool_calls>"],
                "tags": [group],
            },
        }
    )


class UnoToolConstraint:
    """Advance speculative matchers independently and commit only accepted tokens."""

    def __init__(self, compiled_grammar, vocab_size):
        import numpy as np
        import xgrammar as xgr
        from xgrammar.kernels.apply_token_bitmask_mlx import apply_token_bitmask_mlx

        self.matcher = xgr.GrammarMatcher(compiled_grammar)
        self.bitmask = np.full((1, (vocab_size + 31) // 32), -1, dtype=np.int32)
        self.apply_mask = apply_token_bitmask_mlx
        self.vocab_size = vocab_size

    def seed(self, logits):
        import mlx.core as mx

        self.bitmask.fill(-1)
        self.matcher.fill_next_token_bitmask(self.bitmask)
        first = self.apply_mask(mx.array(self.bitmask), logits[:1], self.vocab_size)
        return mx.concatenate([first, logits[1:]])

    def verify(self, logits, proposals):
        import mlx.core as mx
        import numpy as np

        matcher = self.matcher.fork()
        vocab_size = logits.shape[-1]
        masks = np.full((len(proposals), (vocab_size + 31) // 32), -1, dtype=np.int32)
        for row, token in enumerate(proposals):
            if matcher.is_terminated() or not matcher.accept_token(token):
                break
            if not matcher.is_terminated():
                matcher.fill_next_token_bitmask(masks, row)
        return self.apply_mask(mx.array(masks), logits, vocab_size)

    def commit(self, tokens):
        for token in tokens:
            if not self.matcher.accept_token(token):
                raise RuntimeError("Uno committed a token outside the K2 tool grammar")
