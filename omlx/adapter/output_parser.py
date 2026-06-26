# SPDX-License-Identifier: Apache-2.0
"""Generic streamed output parser sessions.

This module provides a tiny scheduler-facing abstraction for protocol-specific
output parsing.  A parser session owns any protocol state needed while a single
request is generating (e.g. Harmony channel parsing or Gemma 4 reasoning marker
suppression) and exposes a uniform token-by-token interface.
"""

from __future__ import annotations

import json
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
    thinking_start_text: str | None = None
    thinking_start_output_text: str | None = None
    thinking_end_text: str | None = None
    thinking_end_trailing_text: str | None = None
    # Marker strings that must survive special-token stripping so the
    # parser session can see them in the text stream.  Engines that strip
    # special tokens during detokenization (e.g. the serial diffusion
    # lane) preserve the token ids of these markers and let the parser
    # session remove them instead.
    protocol_marker_texts: tuple[str, ...] = ()


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


_MINIMAX_M3_MODEL_TYPES = {"minimax_m3", "minimax_m3_vl"}
_MINIMAX_THINK_START = "<mm:think>"
_MINIMAX_THINK_END = "</mm:think>"
_MINIMAX_EOS_TOKEN = "[e~["
_MINIMAX_SPECIAL_TOKENS = (_MINIMAX_EOS_TOKEN, "]~b]", "]~!b[", "]!p~[", "]!d~[")
_MINIMAX_TOOL_CALL_START = "]<]minimax[>[<tool_call>"
_MINIMAX_TOOL_CALL_END = "]<]minimax[>[</tool_call>"


def _serialize_minimax_tool_arguments(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments or "{}"
    if arguments is None:
        return "{}"
    try:
        return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    except TypeError:
        return str(arguments)


def _is_minimax_m3_model(
    model_name: str,
    model_config: dict[str, Any] | None = None,
) -> bool:
    model_type = model_config.get("model_type") if model_config else None
    if model_type in _MINIMAX_M3_MODEL_TYPES:
        return True
    lowered = model_name.lower()
    return "minimax" in lowered and "m3" in lowered


class _MiniMaxM3ProtocolNormalizer:
    """Normalize MiniMax M3 protocol markers to oMLX-visible markers."""

    _REPLACEMENTS = (
        (_MINIMAX_THINK_START, "<think>"),
        (_MINIMAX_THINK_END, "</think>"),
        *tuple((token, "") for token in _MINIMAX_SPECIAL_TOKENS),
    )
    _MARKERS = tuple(marker for marker, _ in _REPLACEMENTS)

    def __init__(self) -> None:
        self._buffer = ""

    @classmethod
    def _replace_markers(cls, text: str) -> str:
        for marker, replacement in cls._REPLACEMENTS:
            text = text.replace(marker, replacement)
        return text

    @classmethod
    def _partial_suffix_len(cls, text: str) -> int:
        max_len = min(len(text), max(len(marker) for marker in cls._MARKERS) - 1)
        for size in range(max_len, 0, -1):
            suffix = text[-size:]
            if any(marker.startswith(suffix) for marker in cls._MARKERS):
                return size
        return 0

    def feed(self, text: str) -> str:
        if not text:
            return ""

        self._buffer += text
        keep = self._partial_suffix_len(self._buffer)
        if keep:
            ready = self._buffer[:-keep]
            self._buffer = self._buffer[-keep:]
        else:
            ready = self._buffer
            self._buffer = ""
        return self._replace_markers(ready)

    def finish(self) -> str:
        text = self._replace_markers(self._buffer)
        self._buffer = ""
        return text


def _token_id_for_text(tokenizer: Any, text: str) -> int | None:
    try:
        token_id = tokenizer.convert_tokens_to_ids(text)
    except (AttributeError, KeyError, TypeError, ValueError):
        token_id = None
    if token_id is not None and token_id != getattr(tokenizer, "unk_token_id", None):
        try:
            return int(token_id)
        except (TypeError, ValueError):
            pass

    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        try:
            token_ids = tokenizer.encode(text)
        except Exception:
            return None
    except Exception:
        return None

    if len(token_ids) == 1:
        try:
            return int(token_ids[0])
        except (TypeError, ValueError):
            return None
    return None


class MiniMaxM3OutputParserSession:
    """Parser session for MiniMax M3 XML-style tool calls."""

    def __init__(self, tokenizer: Any, model_path: str | None = None):
        self._tokenizer = tokenizer
        self._raw_text = ""
        self._detokenizer = create_streaming_detokenizer(tokenizer, model_path)
        if self._detokenizer is not None:
            self._detokenizer.reset()

        try:
            from ..api.tool_calling import ToolCallStreamFilter

            self._stream_filter = ToolCallStreamFilter(tokenizer)
            self._visible_filter = ToolCallStreamFilter(tokenizer)
        except Exception as e:  # noqa: BLE001
            logger.debug("MiniMax M3 stream filter unavailable: %s", e)
            self._stream_filter = None
            self._visible_filter = None
        self._stream_normalizer = _MiniMaxM3ProtocolNormalizer()
        self._visible_normalizer = _MiniMaxM3ProtocolNormalizer()

    def _decode_token(self, token_id: int) -> str:
        if self._detokenizer is not None:
            self._detokenizer.add_token(token_id)
            return self._detokenizer.last_segment
        try:
            return self._tokenizer.decode([token_id], skip_special_tokens=False)
        except TypeError:
            return self._tokenizer.decode([token_id])

    def _filtered_text(
        self,
        text: str,
        tool_filter: Any,
        normalizer: _MiniMaxM3ProtocolNormalizer,
    ) -> str:
        if not text:
            return ""
        if tool_filter is not None:
            text = tool_filter.feed(text)
        return normalizer.feed(text)

    def _finish_filtered_text(
        self,
        tool_filter: Any,
        normalizer: _MiniMaxM3ProtocolNormalizer,
    ) -> str:
        text = ""
        if tool_filter is not None:
            text += normalizer.feed(tool_filter.finish())
        text += normalizer.finish()
        return text

    def process_token(self, token_id: int) -> OutputParserTokenResult:
        decoded_text = self._decode_token(token_id)
        self._raw_text += decoded_text
        is_stop = decoded_text == _MINIMAX_EOS_TOKEN
        return OutputParserTokenResult(
            stream_text=self._filtered_text(
                decoded_text,
                self._stream_filter,
                self._stream_normalizer,
            ),
            visible_text=self._filtered_text(
                decoded_text,
                self._visible_filter,
                self._visible_normalizer,
            ),
            is_stop=is_stop,
            record_token=not is_stop,
        )

    def finalize(self) -> OutputParserFinalizeResult:
        stream_text = ""
        visible_text = ""
        if self._detokenizer is not None:
            self._detokenizer.finalize()
            final_text = self._detokenizer.last_segment
            if final_text:
                self._raw_text += final_text
                stream_text += self._filtered_text(
                    final_text,
                    self._stream_filter,
                    self._stream_normalizer,
                )
                visible_text += self._filtered_text(
                    final_text,
                    self._visible_filter,
                    self._visible_normalizer,
                )

        stream_text += self._finish_filtered_text(
            self._stream_filter,
            self._stream_normalizer,
        )
        visible_text += self._finish_filtered_text(
            self._visible_filter,
            self._visible_normalizer,
        )

        tool_calls: list[dict[str, str]] = []
        if _MINIMAX_TOOL_CALL_START in self._raw_text:
            try:
                from ..patches.mlx_vlm_minimax_m3_compat import (
                    apply_mlx_vlm_minimax_m3_compat_patch,
                )

                apply_mlx_vlm_minimax_m3_compat_patch()

                from mlx_vlm.tool_parsers.minimax_m3 import parse_tool_call

                parsed = parse_tool_call(self._raw_text)
                parsed_calls = parsed if isinstance(parsed, list) else [parsed]
                tool_calls = [
                    {
                        "name": str(call.get("name", "")),
                        "arguments": _serialize_minimax_tool_arguments(
                            call.get("arguments")
                        ),
                    }
                    for call in parsed_calls
                    if isinstance(call, dict) and call.get("name")
                ]
            except Exception as e:  # noqa: BLE001
                logger.debug("MiniMax M3 tool-call parse failed: %s", e)

        return OutputParserFinalizeResult(
            stream_text=stream_text,
            visible_text=visible_text,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else None,
        )


def _create_cohere2_moe_filter():
    try:
        from cohere_melody import PyFilter, PyFilterOptions
    except ImportError:
        return None

    return PyFilter(PyFilterOptions().cmd4().stream_tool_actions())


def _reserialize_cohere_tool_arguments(args: str) -> str:
    if not args:
        return "{}"
    try:
        return json.dumps(
            json.loads(args, strict=False),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    except (json.JSONDecodeError, ValueError):
        return args or "{}"


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
                "arguments": _reserialize_cohere_tool_arguments(value["arguments"]),
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
        from .gemma4 import (
            _CLOSE_MARKER,
            _OPEN_MARKER_BARE,
            _TOOL_RESPONSE_CLOSE,
            _TOOL_RESPONSE_OPEN,
            _TURN_END_MARKER,
            Gemma4OutputParserSession,
        )

        return OutputParserFactory(
            kind="gemma4",
            create_session=lambda session_tokenizer: Gemma4OutputParserSession(
                session_tokenizer,
                model_path=model_name,
            ),
            stop_token_ids=set(),
            thinking_end_text="<channel|>",
            protocol_marker_texts=(
                _OPEN_MARKER_BARE,
                _CLOSE_MARKER,
                _TURN_END_MARKER,
                _TOOL_RESPONSE_OPEN,
                _TOOL_RESPONSE_CLOSE,
            ),
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

    if _is_minimax_m3_model(model_name, model_config):
        minimax_stop_ids = set()
        eos_id = _token_id_for_text(tokenizer, _MINIMAX_EOS_TOKEN)
        if eos_id is not None:
            minimax_stop_ids.add(eos_id)

        return OutputParserFactory(
            kind="minimax_m3",
            create_session=lambda session_tokenizer: MiniMaxM3OutputParserSession(
                session_tokenizer,
                model_path=model_name,
            ),
            stop_token_ids=minimax_stop_ids,
            thinking_start_text=_MINIMAX_THINK_START,
            thinking_start_output_text="<think>\n",
            thinking_end_text=_MINIMAX_THINK_END,
            protocol_marker_texts=(
                _MINIMAX_THINK_START,
                _MINIMAX_THINK_END,
                _MINIMAX_TOOL_CALL_START,
                _MINIMAX_TOOL_CALL_END,
            ),
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
