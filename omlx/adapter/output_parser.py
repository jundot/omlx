# SPDX-License-Identifier: Apache-2.0
"""Generic streamed output parser sessions.

This module provides a tiny scheduler-facing abstraction for protocol-specific
output parsing.  A parser session owns any protocol state needed while a single
request is generating (e.g. Harmony channel parsing or Gemma 4 reasoning marker
suppression) and exposes a uniform token-by-token interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

try:
    from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer
except ImportError:
    NaiveStreamingDetokenizer = None

from .harmony import HarmonyStreamingParser, parse_tool_calls_from_tokens
from ..utils.tokenizer import is_gemma4_model, is_harmony_model


@dataclass
class OutputParserTokenResult:
    """Per-token parser result returned during streaming."""

    stream_text: str = ""
    visible_text: str = ""
    is_stop: bool = False
    record_token: bool | None = None


@dataclass
class OutputParserFinalizeResult:
    """Final parser result returned once a request finishes."""

    stream_text: str = ""
    visible_text: str = ""
    output_text_prefix: str = ""
    tool_calls: list[dict[str, str]] = field(default_factory=list)
    finish_reason: str | None = None


class OutputParserSession(Protocol):
    """Protocol implemented by per-request output parser sessions."""

    def process_token(self, token_id: int) -> OutputParserTokenResult:
        """Process one generated token."""

    def finalize(self) -> OutputParserFinalizeResult:
        """Flush any buffered output when generation ends."""


@dataclass(frozen=True)
class OutputParserFactory:
    """Factory for creating per-request parser sessions."""

    kind: str
    create_session: Callable[[Any], OutputParserSession]
    stop_token_ids: set[int] = field(default_factory=set)


class HarmonyOutputParserSession:
    """Scheduler-facing wrapper around ``HarmonyStreamingParser``."""

    def __init__(self, tokenizer: Any):
        self._tokenizer = tokenizer
        self._parser = HarmonyStreamingParser(tokenizer)
        self._raw_token_ids: list[int] = []

        if hasattr(tokenizer, "detokenizer"):
            self._detokenizer = tokenizer.detokenizer
        elif NaiveStreamingDetokenizer is not None:
            self._detokenizer = NaiveStreamingDetokenizer(tokenizer)
        else:
            self._detokenizer = None

        if self._detokenizer is not None:
            self._detokenizer.reset()

    def process_token(self, token_id: int) -> OutputParserTokenResult:
        control_text, stream_token, visible_token, is_stop = self._parser.process_token(
            token_id
        )
        self._raw_token_ids.append(token_id)

        stream_text = control_text
        visible_text = ""

        if stream_token is not None:
            if self._detokenizer is not None:
                self._detokenizer.add_token(stream_token)
                decoded_text = self._detokenizer.last_segment
            else:
                decoded_text = self._tokenizer.decode([stream_token])

            stream_text += decoded_text
            if visible_token is not None:
                visible_text += decoded_text
        elif visible_token is not None:
            if self._detokenizer is not None:
                self._detokenizer.add_token(visible_token)
                visible_text += self._detokenizer.last_segment
            else:
                visible_text += self._tokenizer.decode([visible_token])

        return OutputParserTokenResult(
            stream_text=stream_text,
            visible_text=visible_text,
            is_stop=is_stop,
            record_token=True,
        )

    def finalize(self) -> OutputParserFinalizeResult:
        stream_text = self._parser.finalize()
        visible_text = ""

        if self._detokenizer is not None:
            self._detokenizer.finalize()
            final_text = self._detokenizer.last_segment
            if final_text:
                stream_text += final_text
                if self._parser.current_channel == "final":
                    visible_text += final_text

        _, analysis_text, tool_calls = parse_tool_calls_from_tokens(
            self._raw_token_ids
        )
        finish_reason = "tool_calls" if tool_calls else None

        output_text_prefix = (
            f"<think>\n{analysis_text}\n</think>\n" if analysis_text else ""
        )

        return OutputParserFinalizeResult(
            stream_text=stream_text,
            visible_text=visible_text,
            output_text_prefix=output_text_prefix,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )


def detect_output_parser(
    model_name: str,
    tokenizer: Any,
    model_config: Optional[dict[str, Any]] = None,
) -> OutputParserFactory | None:
    """Detect a protocol-specific output parser for the model, if needed."""

    if is_harmony_model(model_name, model_config):
        temp_parser = HarmonyStreamingParser(tokenizer)
        return OutputParserFactory(
            kind="harmony",
            create_session=HarmonyOutputParserSession,
            stop_token_ids=temp_parser.get_stop_token_ids(),
        )

    if is_gemma4_model(model_name, model_config):
        from .gemma4 import Gemma4OutputParserSession

        return OutputParserFactory(
            kind="gemma4",
            create_session=Gemma4OutputParserSession,
            stop_token_ids=set(),
        )

    return None


def detect_message_extractor(
    model_name: str,
    model_config: Optional[dict[str, Any]] = None,
) -> Callable:
    """Return the appropriate message extractor function for the model.

    The returned callable has the signature::

        extractor(messages, max_tool_result_tokens=None, tokenizer=None) -> list[dict]

    This mirrors how ``detect_output_parser`` decouples model-specific
    knowledge from the server layer — the engine stores the extractor at
    load time and the server just calls ``engine.message_extractor(...)``.
    """
    if is_harmony_model(model_name, model_config):
        from ..api.utils import extract_harmony_messages

        return extract_harmony_messages

    if is_gemma4_model(model_name, model_config):
        from .gemma4 import extract_gemma4_messages

        return extract_gemma4_messages

    # Default: caller decides between extract_text_content and
    # extract_multimodal_content based on engine type (VLM vs text).
    return None
