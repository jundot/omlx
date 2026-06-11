# SPDX-License-Identifier: Apache-2.0
"""Generic streamed output parser sessions.

This module provides a tiny scheduler-facing abstraction for protocol-specific
output parsing.  A parser session owns any protocol state needed while a single
request is generating (e.g. Harmony channel parsing or Gemma 4 reasoning marker
suppression) and exposes a uniform token-by-token interface.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..utils.tokenizer import (
    create_streaming_detokenizer,
    is_gemma4_model,
    is_harmony_model,
)
from .harmony import HarmonyStreamingParser, parse_tool_calls_from_tokens

logger = logging.getLogger(__name__)


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
    thinking_end_text: str | None = None
    thinking_end_trailing_text: str | None = None


class HarmonyOutputParserSession:
    """Scheduler-facing wrapper around ``HarmonyStreamingParser``."""

    def __init__(self, tokenizer: Any, model_path: str | None = None):
        self._tokenizer = tokenizer
        self._parser = HarmonyStreamingParser(tokenizer)
        self._raw_token_ids: list[int] = []

        self._detokenizer = create_streaming_detokenizer(tokenizer, model_path)
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

        _, analysis_text, tool_calls = parse_tool_calls_from_tokens(self._raw_token_ids)
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


def _is_cohere2_moe_model(
    model_name: str,
    model_config: dict[str, Any] | None = None,
) -> bool:
    return (
        model_config is not None
        and model_config.get("model_type") == "cohere2_moe"
    )


def _create_cohere2_moe_filter():
    try:
        from cohere_melody import PyFilter, PyFilterOptions
    except ImportError:
        return None

    return PyFilter(PyFilterOptions().cmd4().stream_tool_actions())


class Cohere2MoeOutputParserSession:
    """Parser session for Cohere2 MoE / Command-style Melody output."""

    def __init__(self, tokenizer: Any, model_path: str | None = None):
        self._tokenizer = tokenizer
        self._melody = _create_cohere2_moe_filter()
        if self._melody is None:
            raise RuntimeError("cohere_melody is not installed")

        self._detokenizer = create_streaming_detokenizer(tokenizer, model_path)
        if self._detokenizer is not None:
            self._detokenizer.reset()

        self._thinking_started = False
        self._thinking_closed = False
        self._tool_calls: dict[int, dict[str, str]] = {}

    def _decode_token(self, token_id: int) -> str:
        if self._detokenizer is not None:
            self._detokenizer.add_token(token_id)
            return self._detokenizer.last_segment
        try:
            return self._tokenizer.decode([token_id], skip_special_tokens=False)
        except TypeError:
            return self._tokenizer.decode([token_id])

    def _accumulate_tool_calls(self, tool_calls: list[Any]) -> None:
        for tool_call in tool_calls:
            index = int(getattr(tool_call, "index", 0) or 0)
            current = self._tool_calls.setdefault(
                index,
                {"id": "", "name": "", "arguments": ""},
            )
            current["id"] += getattr(tool_call, "id", "") or ""
            current["name"] += getattr(tool_call, "name", "") or ""
            current["arguments"] += getattr(tool_call, "arguments", "") or ""

    def _apply_melody_result(self, result: Any) -> tuple[str, str]:
        stream_text = ""
        visible_text = ""

        reasoning = getattr(result, "reasoning", None)
        if reasoning:
            if not self._thinking_started:
                self._thinking_started = True
                stream_text += "<think>\n"
                visible_text += "<think>\n"
            stream_text += reasoning
            visible_text += reasoning

        content = getattr(result, "content", None)
        if content:
            if self._thinking_started and not self._thinking_closed:
                self._thinking_closed = True
                stream_text += "</think>\n"
                visible_text += "</think>\n"
            stream_text += content
            visible_text += content

        self._accumulate_tool_calls(getattr(result, "tool_calls", []) or [])
        return stream_text, visible_text

    def process_token(self, token_id: int) -> OutputParserTokenResult:
        decoded_text = self._decode_token(token_id)
        if not decoded_text:
            return OutputParserTokenResult(record_token=True)

        result = self._melody.write_decoded(decoded_text)
        stream_text, visible_text = self._apply_melody_result(result)
        return OutputParserTokenResult(
            stream_text=stream_text,
            visible_text=visible_text,
            record_token=True,
        )

    def finalize(self) -> OutputParserFinalizeResult:
        stream_text = ""
        visible_text = ""

        if self._detokenizer is not None:
            self._detokenizer.finalize()
            final_text = self._detokenizer.last_segment
            if final_text:
                result = self._melody.write_decoded(final_text)
                s_text, v_text = self._apply_melody_result(result)
                stream_text += s_text
                visible_text += v_text

        result = self._melody.flush_partials()
        s_text, v_text = self._apply_melody_result(result)
        stream_text += s_text
        visible_text += v_text

        if self._thinking_started and not self._thinking_closed:
            self._thinking_closed = True
            stream_text += "</think>\n"
            visible_text += "</think>\n"

        tool_calls = [
            {
                "id": value["id"],
                "name": value["name"],
                "arguments": value["arguments"] or "{}",
            }
            for _, value in sorted(self._tool_calls.items())
            if value["name"]
        ]

        return OutputParserFinalizeResult(
            stream_text=stream_text,
            visible_text=visible_text,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else None,
        )


def detect_output_parser(
    model_name: str,
    tokenizer: Any,
    model_config: dict[str, Any] | None = None,
) -> OutputParserFactory | None:
    """Detect a protocol-specific output parser for the model, if needed."""

    if is_harmony_model(model_name, model_config):
        temp_parser = HarmonyStreamingParser(tokenizer)
        return OutputParserFactory(
            kind="harmony",
            create_session=lambda session_tokenizer: HarmonyOutputParserSession(
                session_tokenizer,
                model_path=model_name,
            ),
            stop_token_ids=temp_parser.get_stop_token_ids(),
            thinking_end_text="<|end|>",
            thinking_end_trailing_text="<|start|>assistant<|channel|>final<|message|>",
        )

    if is_gemma4_model(model_name, model_config):
        from .gemma4 import Gemma4OutputParserSession

        return OutputParserFactory(
            kind="gemma4",
            create_session=lambda session_tokenizer: Gemma4OutputParserSession(
                session_tokenizer,
                model_path=model_name,
            ),
            stop_token_ids=set(),
            thinking_end_text="<channel|>",
        )

    if _is_cohere2_moe_model(model_name, model_config):
        if _create_cohere2_moe_filter() is None:
            logger.warning(
                "cohere_melody is not installed; Cohere2 MoE output parser "
                "is disabled for %s",
                model_name,
            )
            return None

        return OutputParserFactory(
            kind="cohere2_moe",
            create_session=lambda session_tokenizer: Cohere2MoeOutputParserSession(
                session_tokenizer,
                model_path=model_name,
            ),
            stop_token_ids=set(),
            thinking_end_text="</think>",
        )

    return None


def detect_message_extractor(
    model_name: str,
    model_config: dict[str, Any] | None = None,
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
