# SPDX-License-Identifier: Apache-2.0
"""Generic streamed output parser sessions.

This module provides a tiny scheduler-facing abstraction for protocol-specific
output parsing.  A parser session owns any protocol state needed while a single
request is generating (e.g. Harmony channel parsing or Gemma 4 reasoning marker
suppression) and exposes a uniform token-by-token interface.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..utils.tokenizer import (
    create_streaming_detokenizer,
    is_gemma4_model,
    is_harmony_model,
)
from .harmony import HarmonyStreamingParser, parse_tool_calls_from_tokens


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


_COHERE_COMMAND_MODEL_TYPES = {"cohere2_moe", "cohere2_vision"}
_COHERE_END_OF_TURN_TOKEN = "<|END_OF_TURN_TOKEN|>"


def _is_cohere_command_model(
    model_name: str,
    model_config: dict[str, Any] | None = None,
) -> bool:
    if model_config is not None:
        model_type = str(model_config.get("model_type") or "").lower()
        if model_type in _COHERE_COMMAND_MODEL_TYPES:
            return True

    name = (model_name or "").lower()
    return "north-mini-code" in name or "command-a-plus" in name


def _make_cohere_command_filter():
    try:
        from cohere_melody import PyFilter, PyFilterOptions
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Cohere Command/North response parsing requires "
            "cohere_melody>=0.9.0. Install or sync oMLX dependencies."
        ) from exc

    return PyFilter(PyFilterOptions().cmd4().stream_tool_actions())


def _cohere_stop_token_ids(tokenizer: Any) -> set[int]:
    ids: set[int] = set()

    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert):
        try:
            token_id = convert(_COHERE_END_OF_TURN_TOKEN)
            if isinstance(token_id, int) and token_id >= 0:
                ids.add(token_id)
        except Exception:
            pass

    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            encoded = encode(_COHERE_END_OF_TURN_TOKEN, add_special_tokens=False)
        except TypeError:
            try:
                encoded = encode(_COHERE_END_OF_TURN_TOKEN)
            except Exception:
                encoded = None
        except Exception:
            encoded = None
        if encoded:
            ids.update(int(token_id) for token_id in encoded)

    return ids


class CohereCommandOutputParserSession:
    """Parser session backed by Cohere's Melody parser for Command 4 markup."""

    def __init__(self, tokenizer: Any, model_path: str | None = None):
        self._tokenizer = tokenizer
        self._filter = _make_cohere_command_filter()
        self._detokenizer = create_streaming_detokenizer(tokenizer, model_path)
        if self._detokenizer is not None:
            self._detokenizer.reset()

        self._thinking_open = False
        self._thinking_started = False
        self._tool_calls: dict[int, dict[str, str]] = {}
        self._stop_token_ids = _cohere_stop_token_ids(tokenizer)

    @staticmethod
    def _is_special_token(text: str) -> bool:
        return text.startswith("<|") and text.endswith("|>")

    def _decode_token(self, token_id: int) -> str:
        decoded = ""
        try:
            decoded = self._tokenizer.decode([token_id], skip_special_tokens=False)
        except TypeError:
            decoded = self._tokenizer.decode([token_id])
        except Exception:
            decoded = ""

        if decoded and self._is_special_token(decoded):
            return decoded

        if self._detokenizer is not None:
            self._detokenizer.add_token(token_id)
            return self._detokenizer.last_segment

        return decoded

    def _remember_tool_calls(self, result: Any) -> None:
        for call in getattr(result, "tool_calls", []) or []:
            try:
                index = int(getattr(call, "index", len(self._tool_calls)))
            except Exception:
                index = len(self._tool_calls)
            stored = self._tool_calls.setdefault(
                index,
                {"id": f"call_{index}", "name": "", "arguments": ""},
            )
            if call_id := str(getattr(call, "id", "") or ""):
                stored["id"] = call_id
            if name := str(getattr(call, "name", "") or ""):
                stored["name"] = name
            if arguments := str(getattr(call, "arguments", "") or ""):
                stored["arguments"] += arguments

    def _format_result(self, result: Any) -> tuple[str, str]:
        self._remember_tool_calls(result)

        text = []
        reasoning = getattr(result, "reasoning", None)
        content = getattr(result, "content", None)

        if reasoning:
            if not self._thinking_open:
                text.append("<think>\n")
                self._thinking_open = True
                self._thinking_started = True
            text.append(str(reasoning))

        if content:
            if self._thinking_open:
                text.append("</think>\n")
                self._thinking_open = False
            text.append(str(content))

        out = "".join(text)
        return out, out

    def process_token(self, token_id: int) -> OutputParserTokenResult:
        decoded = self._decode_token(token_id)
        if token_id in self._stop_token_ids or decoded == _COHERE_END_OF_TURN_TOKEN:
            return OutputParserTokenResult(
                is_stop=True,
                record_token=False,
            )

        if not decoded:
            return OutputParserTokenResult(record_token=True)

        result = self._filter.write_decoded(decoded)
        stream_text, visible_text = self._format_result(result)
        return OutputParserTokenResult(
            stream_text=stream_text,
            visible_text=visible_text,
            record_token=True,
        )

    def finalize(self) -> OutputParserFinalizeResult:
        result = self._filter.flush_partials()
        stream_text, visible_text = self._format_result(result)

        if self._thinking_open and self._thinking_started:
            stream_text += "</think>"
            visible_text += "</think>"
            self._thinking_open = False

        tool_calls = [self._tool_calls[index] for index in sorted(self._tool_calls)]
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

    if _is_cohere_command_model(model_name, model_config):
        return OutputParserFactory(
            kind="cohere_command4",
            create_session=lambda session_tokenizer: CohereCommandOutputParserSession(
                session_tokenizer,
                model_path=model_name,
            ),
            stop_token_ids=_cohere_stop_token_ids(tokenizer),
            thinking_end_text="<|END_THINKING|>",
            thinking_end_trailing_text="<|START_TEXT|>",
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
