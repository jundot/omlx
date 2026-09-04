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
from contextlib import suppress
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
    create_session_with_tools: (
        Callable[[Any, list[dict] | None], OutputParserSession] | None
    ) = None
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
    return model_config is not None and model_config.get("model_type") == "cohere2_moe"


_MINIMAX_M3_MODEL_TYPES = {"minimax_m3", "minimax_m3_vl"}
_MINIMAX_THINK_START = "<mm:think>"
_MINIMAX_THINK_END = "</mm:think>"
_MINIMAX_EOS_TOKEN = "[e~["
_MINIMAX_SPECIAL_TOKENS = (_MINIMAX_EOS_TOKEN, "]~b]", "]~!b[", "]!p~[", "]!d~[")
_MINIMAX_TOOL_CALL_START = "]<]minimax[>[<tool_call>"
_MINIMAX_TOOL_CALL_END = "]<]minimax[>[</tool_call>"
_DEEPSEEK_V4_TOOL_CALL_START = "<｜DSML｜tool_calls>"
_DEEPSEEK_V4_TOOL_CALL_END = "</｜DSML｜tool_calls>"
_BAILING_HYBRID_MODEL_TYPE = "bailing_hybrid"
_BAILING_ROLE_MARKERS = ("<role>", "</role>")


def _is_deepseek_v4_model(
    model_name: str,
    tokenizer: Any,
    model_config: dict[str, Any] | None = None,
) -> bool:
    model_type = str(model_config.get("model_type", "")) if model_config else ""
    if model_type.startswith("deepseek_v4"):
        return True

    if (
        getattr(tokenizer, "tool_call_start", None) == _DEEPSEEK_V4_TOOL_CALL_START
        and getattr(tokenizer, "tool_call_end", None) == _DEEPSEEK_V4_TOOL_CALL_END
    ):
        return True

    return "deepseek-v4" in model_name.lower() or "deepseek_v4" in model_name.lower()


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


class BailingHybridOutputParserSession:
    """Suppress Ling role markers and XML tool-call protocol envelopes."""

    def __init__(
        self,
        tokenizer: Any,
        marker_token_ids: set[int],
        model_path: str | None = None,
        tools: list[dict] | None = None,
    ):
        self._tokenizer = tokenizer
        self._marker_token_ids = marker_token_ids
        self._tools = tools
        self._raw_text = ""
        self._detokenizer = create_streaming_detokenizer(tokenizer, model_path)
        if self._detokenizer is not None:
            self._detokenizer.reset()

        try:
            from ..api.tool_calling import ToolCallStreamFilter

            self._stream_filter = ToolCallStreamFilter(tokenizer)
            self._visible_filter = ToolCallStreamFilter(tokenizer)
        except Exception as e:  # noqa: BLE001
            logger.debug("Ling stream filter unavailable: %s", e)
            self._stream_filter = None
            self._visible_filter = None

    def _decode_token(self, token_id: int) -> str:
        if self._detokenizer is not None:
            self._detokenizer.add_token(token_id)
            return self._detokenizer.last_segment
        try:
            return self._tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
            )
        except TypeError:
            return self._tokenizer.decode([token_id])

    @staticmethod
    def _filtered_text(text: str, tool_filter: Any) -> str:
        if not text:
            return ""
        if tool_filter is not None:
            return tool_filter.feed(text)
        return text

    @staticmethod
    def _finish_filtered_text(tool_filter: Any) -> str:
        if tool_filter is None:
            return ""
        return tool_filter.finish()

    def process_token(self, token_id: int) -> OutputParserTokenResult:
        if token_id in self._marker_token_ids:
            return OutputParserTokenResult(record_token=True)

        text = self._decode_token(token_id)
        self._raw_text += text

        return OutputParserTokenResult(
            stream_text=self._filtered_text(text, self._stream_filter),
            visible_text=self._filtered_text(text, self._visible_filter),
            record_token=True,
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
                )
                visible_text += self._filtered_text(
                    final_text,
                    self._visible_filter,
                )

        stream_text += self._finish_filtered_text(self._stream_filter)
        visible_text += self._finish_filtered_text(self._visible_filter)

        tool_calls: list[dict[str, str]] = []
        if self._tools:
            try:
                from ..api.tool_calling import parse_tool_calls

                _, parsed_calls = parse_tool_calls(
                    self._raw_text,
                    self._tokenizer,
                    self._tools,
                )
                valid_names = {
                    function["name"]
                    for tool in self._tools
                    if isinstance(tool, dict)
                    and isinstance((function := tool.get("function")), dict)
                    and isinstance(function.get("name"), str)
                    and function["name"]
                }
                for call in parsed_calls or []:
                    if call.function.name not in valid_names:
                        logger.warning(
                            "Dropping unregistered Ling tool call %r",
                            call.function.name,
                        )
                        continue
                    tool_calls.append(
                        {
                            "id": getattr(call, "id", ""),
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        }
                    )
            except Exception as e:  # noqa: BLE001
                logger.debug("Ling tool-call parse failed: %s", e)

        return OutputParserFinalizeResult(
            stream_text=stream_text,
            visible_text=visible_text,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else None,
        )


def install_minimax_m3_tokenizer_protocol(
    tokenizer: Any,
    model_name: str,
    model_config: dict[str, Any] | None = None,
) -> bool:
    """Install oMLX's MiniMax M3 protocol on any serving tokenizer wrapper.

    Both the normal VLM engine and the distributed MLX-LM server use this
    adapter. MLX-LM otherwise infers ``json_tools`` from MiniMax's template,
    although the model emits namespaced XML, and recognizes ``<think>`` rather
    than MiniMax's ``<mm:think>`` markers.

    ``mlx_lm`` exposes these values as read-only properties backed by private
    constructor fields; ``mlx_vlm`` exposes ordinary instance attributes.
    Populate both representations so every oMLX serving lane shares the same
    parser and marker contract.
    """

    if not _is_minimax_m3_model(model_name, model_config):
        return False

    from ..patches.mlx_vlm_minimax_m3_compat import (
        apply_mlx_vlm_minimax_m3_compat_patch,
    )

    apply_mlx_vlm_minimax_m3_compat_patch()
    from mlx_vlm.tool_parsers.minimax_m3 import (
        parse_tool_call as parse_native_tool_call,
    )

    def parse_tool_call(text: str, tools: Any = None) -> Any:
        parsed = parse_native_tool_call(text, tools)
        calls = parsed if isinstance(parsed, list) else [parsed]
        normalized: list[dict[str, Any]] = []
        for call in calls:
            if not isinstance(call, dict):
                raise ValueError("MiniMax M3 tool parser returned a non-object call")
            arguments = call.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments or "{}")
            if not isinstance(arguments, dict):
                raise ValueError("MiniMax M3 tool parser returned non-object arguments")
            normalized.append(
                {
                    "name": str(call.get("name", "")),
                    "arguments": arguments,
                }
            )
        return normalized if isinstance(parsed, list) else normalized[0]

    def encoded(marker: str) -> tuple[int, ...]:
        tokens = tokenizer.encode(marker, add_special_tokens=False)
        if not tokens:
            raise RuntimeError(
                f"MiniMax M3 protocol marker {marker!r} produced no tokens"
            )
        return tuple(int(token) for token in tokens)

    values = {
        "tool_parser": parse_tool_call,
        "tool_call_start": _MINIMAX_TOOL_CALL_START,
        "tool_call_end": _MINIMAX_TOOL_CALL_END,
        "tool_call_start_tokens": encoded(_MINIMAX_TOOL_CALL_START),
        "tool_call_end_tokens": encoded(_MINIMAX_TOOL_CALL_END),
        "think_start": _MINIMAX_THINK_START,
        "think_end": _MINIMAX_THINK_END,
        "think_start_tokens": encoded(_MINIMAX_THINK_START),
        "think_end_tokens": encoded(_MINIMAX_THINK_END),
    }
    for name, value in values.items():
        setattr(tokenizer, f"_{name}", value)
        # mlx_lm TokenizerWrapper exposes a read-only property.
        with suppress(AttributeError, TypeError):
            setattr(tokenizer, name, value)
    for name in ("has_tool_calling", "has_thinking"):
        with suppress(AttributeError, TypeError):
            setattr(tokenizer, name, True)
    return True


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


class DeepSeekV4OutputParserSession:
    """Parser session for DeepSeek V4 DSML tool-call output.

    A completed DSML tool-call block ends the assistant turn. Without a
    parser-owned stop, batched decode keeps the row alive after
    ``</｜DSML｜tool_calls>`` and the model may emit additional or malformed
    DSML fragments as visible assistant text.
    """

    def __init__(self, tokenizer: Any, model_path: str | None = None):
        self._tokenizer = tokenizer
        self._raw_text = ""
        self._stopped = False
        self._detokenizer = create_streaming_detokenizer(tokenizer, model_path)
        if self._detokenizer is not None:
            self._detokenizer.reset()

        try:
            from ..api.tool_calling import ToolCallStreamFilter

            self._stream_filter = ToolCallStreamFilter(tokenizer)
            self._visible_filter = ToolCallStreamFilter(tokenizer)
        except Exception as e:  # noqa: BLE001
            logger.debug("DeepSeek V4 stream filter unavailable: %s", e)
            self._stream_filter = None
            self._visible_filter = None

    def _decode_token(self, token_id: int) -> str:
        if self._detokenizer is not None:
            self._detokenizer.add_token(token_id)
            return self._detokenizer.last_segment
        try:
            return self._tokenizer.decode([token_id], skip_special_tokens=False)
        except TypeError:
            return self._tokenizer.decode([token_id])

    def _filtered_text(self, text: str, tool_filter: Any) -> str:
        if not text:
            return ""
        if tool_filter is not None:
            return tool_filter.feed(text)
        return text

    def _finish_filtered_text(self, tool_filter: Any) -> str:
        if tool_filter is None:
            return ""
        return tool_filter.finish()

    def _trim_at_first_tool_block_end(self, text: str) -> tuple[str, bool]:
        start_idx = text.find(_DEEPSEEK_V4_TOOL_CALL_START)
        if start_idx < 0:
            return text, False
        end_idx = text.find(_DEEPSEEK_V4_TOOL_CALL_END, start_idx)
        if end_idx < 0:
            return text, False
        cutoff = end_idx + len(_DEEPSEEK_V4_TOOL_CALL_END)
        return text[:cutoff], True

    def process_token(self, token_id: int) -> OutputParserTokenResult:
        if self._stopped:
            return OutputParserTokenResult(is_stop=True, record_token=False)

        decoded_text = self._decode_token(token_id)
        combined = self._raw_text + decoded_text
        trimmed, is_stop = self._trim_at_first_tool_block_end(combined)

        feed_text = trimmed[len(self._raw_text) :]
        self._raw_text = trimmed
        self._stopped = is_stop

        return OutputParserTokenResult(
            stream_text=self._filtered_text(feed_text, self._stream_filter),
            visible_text=self._filtered_text(feed_text, self._visible_filter),
            is_stop=is_stop,
            record_token=True,
        )

    def finalize(self) -> OutputParserFinalizeResult:
        stream_text = ""
        visible_text = ""
        if self._detokenizer is not None and not self._stopped:
            self._detokenizer.finalize()
            final_text = self._detokenizer.last_segment
            if final_text:
                prev_len = len(self._raw_text)
                combined = self._raw_text + final_text
                self._raw_text, self._stopped = self._trim_at_first_tool_block_end(
                    combined
                )
                final_text = self._raw_text[prev_len:]
                stream_text += self._filtered_text(final_text, self._stream_filter)
                visible_text += self._filtered_text(final_text, self._visible_filter)

        stream_text += self._finish_filtered_text(self._stream_filter)
        visible_text += self._finish_filtered_text(self._visible_filter)

        tool_calls: list[dict[str, str]] = []
        try:
            from ..api.tool_calling import parse_tool_calls

            _, parsed_calls = parse_tool_calls(self._raw_text, self._tokenizer)
            for call in parsed_calls or []:
                tool_calls.append(
                    {
                        "id": getattr(call, "id", ""),
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    }
                )
        except Exception as e:  # noqa: BLE001
            logger.debug("DeepSeek V4 tool-call parse failed: %s", e)

        return OutputParserFinalizeResult(
            stream_text=stream_text,
            visible_text=visible_text,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else None,
        )


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


_INKLING_MODEL_TYPES = {"inkling", "inkling_mm_model"}
_INKLING_MESSAGE_MODEL = "<|message_model|>"
_INKLING_CONTENT_THINKING = "<|content_thinking|>"
_INKLING_CONTENT_TEXT = "<|content_text|>"
_INKLING_CONTENT_XML = "<|content_xml|>"
_INKLING_CONTENT_TOOL_JSON = "<|content_invoke_tool_json|>"
_INKLING_END_MESSAGE = "<|end_message|>"
_INKLING_END_SAMPLING = "<|content_model_end_sampling|>"
_INKLING_MARKERS = (
    _INKLING_MESSAGE_MODEL,
    _INKLING_CONTENT_THINKING,
    _INKLING_CONTENT_TEXT,
    _INKLING_CONTENT_XML,
    _INKLING_CONTENT_TOOL_JSON,
    _INKLING_END_MESSAGE,
    _INKLING_END_SAMPLING,
)


def _is_inkling_model(
    model_name: str,
    model_config: dict[str, Any] | None = None,
) -> bool:
    model_type = model_config.get("model_type") if model_config else None
    if model_type in _INKLING_MODEL_TYPES:
        return True
    return "inkling" in model_name.lower()


def _is_muse_glimmer_model(
    model_name: str,
    model_config: dict[str, Any] | None = None,
) -> bool:
    model_type = model_config.get("model_type") if model_config else None
    if model_type == "muse_glimmer":
        return True
    lowered = model_name.lower()
    return "muse" in lowered and "glimmer" in lowered


def _append_missing_json_object_closers(payload: str) -> str | None:
    """Append missing object closers without counting braces in strings."""
    depth = 0
    in_string = False
    escaped = False

    for char in payload:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            if depth == 0:
                return None
            depth -= 1

    if in_string or depth <= 0:
        return None
    return payload + "}" * depth


class _InklingChannelSplitter:
    """Streaming splitter for inkling's channel protocol.

    The assistant turn is a sequence of blocks::

        [<|message_model|>][HEAD]<|content_*|>BODY<|end_message|> ... \
<|content_model_end_sampling|>

    ``HEAD`` only occurs for tool calls (the function name before
    ``<|content_invoke_tool_json|>``). Thinking bodies surface on the
    stream inside oMLX's ``<think>``/``</think>`` markers, text bodies on
    stream+visible, tool JSON is suppressed (parsed at finalize from the
    raw text).
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._channel: str | None = None
        self._head = ""
        self._think_open = False
        self.stopped = False

    def _partial_suffix_len(self, text: str) -> int:
        max_len = min(len(text), max(len(m) for m in _INKLING_MARKERS) - 1)
        for size in range(max_len, 0, -1):
            suffix = text[-size:]
            if any(m.startswith(suffix) for m in _INKLING_MARKERS):
                return size
        return 0

    def _emit_body(self, text: str) -> tuple[str, str]:
        if not text:
            return "", ""
        if self._channel in ("text", "xml"):
            return text, text
        if self._channel == "thinking":
            # Thinking flows to BOTH channels wrapped in <think> markers
            # (minimax pattern): the scheduler accumulates only
            # visible_text into request.output_text, and the API layer
            # extracts reasoning_content from the <think> block there.
            return text, text
        if self._channel == "tool":
            return "", ""
        # Block head: hold until the next marker classifies it.
        self._head += text
        return "", ""

    def _flush_head_as_text(self) -> tuple[str, str]:
        head, self._head = self._head, ""
        if not head:
            return "", ""
        return head, head

    def _handle_marker(self, marker: str) -> tuple[str, str]:
        stream = visible = ""
        if marker == _INKLING_CONTENT_THINKING:
            s, v = self._flush_head_as_text()
            stream += s
            visible += v
            if not self._think_open:
                stream += "<think>"
                visible += "<think>"
                self._think_open = True
            self._channel = "thinking"
        elif marker in (_INKLING_CONTENT_TEXT, _INKLING_CONTENT_XML):
            s, v = self._flush_head_as_text()
            stream += s
            visible += v
            if self._think_open:
                # A text block after an unterminated thinking block still
                # closes the visible thinking span.
                stream += "</think>"
                visible += "</think>"
                self._think_open = False
            self._channel = "text" if marker == _INKLING_CONTENT_TEXT else "xml"
        elif marker == _INKLING_CONTENT_TOOL_JSON:
            # Head was the tool name; the JSON payload is parsed at
            # finalize from the raw text.
            self._head = ""
            self._channel = "tool"
        elif marker == _INKLING_END_MESSAGE:
            if self._channel == "thinking" and self._think_open:
                stream += "</think>"
                visible += "</think>"
                self._think_open = False
            s, v = self._flush_head_as_text()
            stream += s
            visible += v
            self._channel = None
        elif marker == _INKLING_MESSAGE_MODEL:
            s, v = self._flush_head_as_text()
            stream += s
            visible += v
            self._channel = None
        elif marker == _INKLING_END_SAMPLING:
            self.stopped = True
            self._channel = None
        return stream, visible

    def feed(self, text: str) -> tuple[str, str]:
        if not text:
            return "", ""
        self._buffer += text
        stream = visible = ""
        while True:
            first_idx = -1
            first_marker = None
            for marker in _INKLING_MARKERS:
                idx = self._buffer.find(marker)
                if idx >= 0 and (first_idx < 0 or idx < first_idx):
                    first_idx = idx
                    first_marker = marker
            if first_marker is None:
                break
            s, v = self._emit_body(self._buffer[:first_idx])
            stream += s
            visible += v
            m_s, m_v = self._handle_marker(first_marker)
            stream += m_s
            visible += m_v
            self._buffer = self._buffer[first_idx + len(first_marker) :]

        keep = self._partial_suffix_len(self._buffer)
        ready = self._buffer[: len(self._buffer) - keep]
        self._buffer = self._buffer[len(self._buffer) - keep :]
        s, v = self._emit_body(ready)
        return stream + s, visible + v

    def finish(self) -> tuple[str, str]:
        stream = visible = ""
        s, v = self._emit_body(self._buffer)
        stream += s
        visible += v
        self._buffer = ""
        s, v = self._flush_head_as_text()
        stream += s
        visible += v
        if self._think_open:
            stream += "</think>"
            visible += "</think>"
            self._think_open = False
        return stream, visible


class InklingOutputParserSession:
    """Parser session for inkling channel output (thinking / text / tool)."""

    _TOOL_RE = None  # compiled lazily

    def __init__(self, tokenizer: Any, model_path: str | None = None):
        import re

        self._tokenizer = tokenizer
        self._raw_text = ""
        self._splitter = _InklingChannelSplitter()
        self._detokenizer = create_streaming_detokenizer(tokenizer, model_path)
        if self._detokenizer is not None:
            self._detokenizer.reset()
        if InklingOutputParserSession._TOOL_RE is None:
            InklingOutputParserSession._TOOL_RE = re.compile(
                re.escape(_INKLING_CONTENT_TOOL_JSON)
                + r"(.*?)(?:"
                + re.escape(_INKLING_END_MESSAGE)
                + r"|"
                + re.escape(_INKLING_END_SAMPLING)
                + r"|\Z)",
                re.S,
            )

    def _decode_token(self, token_id: int) -> str:
        if self._detokenizer is not None:
            self._detokenizer.add_token(token_id)
            return self._detokenizer.last_segment
        try:
            return self._tokenizer.decode([token_id], skip_special_tokens=False)
        except TypeError:
            return self._tokenizer.decode([token_id])

    def process_token(self, token_id: int) -> OutputParserTokenResult:
        if self._splitter.stopped:
            return OutputParserTokenResult(is_stop=True, record_token=False)
        decoded_text = self._decode_token(token_id)
        self._raw_text += decoded_text
        stream_text, visible_text = self._splitter.feed(decoded_text)
        is_stop = self._splitter.stopped
        return OutputParserTokenResult(
            stream_text=stream_text,
            visible_text=visible_text,
            is_stop=is_stop,
            record_token=not is_stop,
        )

    def finalize(self) -> OutputParserFinalizeResult:
        stream_text = ""
        visible_text = ""
        if self._detokenizer is not None and not self._splitter.stopped:
            self._detokenizer.finalize()
            final_text = self._detokenizer.last_segment
            if final_text:
                self._raw_text += final_text
                s, v = self._splitter.feed(final_text)
                stream_text += s
                visible_text += v
        s, v = self._splitter.finish()
        stream_text += s
        visible_text += v

        tool_calls: list[dict[str, str]] = []
        for match in InklingOutputParserSession._TOOL_RE.finditer(self._raw_text):
            payload = match.group(1).strip()
            if not payload:
                continue
            try:
                parsed = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                # Quantized checkpoints occasionally emit the payload with the
                # final closing brace(s) missing (observed: a complete nested
                # args object short exactly one "}" before <|end_message|>).
                # Brace-balance repair only runs after strict parsing failed,
                # so well-formed payloads are never touched.
                repaired_payload = _append_missing_json_object_closers(payload)
                if repaired_payload is not None:
                    try:
                        parsed = json.loads(repaired_payload)
                    except (json.JSONDecodeError, ValueError):
                        logger.debug("Inkling tool-call payload not valid JSON")
                        continue
                else:
                    logger.debug("Inkling tool-call payload not valid JSON")
                    continue
            if not isinstance(parsed, dict) or not parsed.get("name"):
                continue
            # Accept both payload conventions: Inkling-native {"name", "args":
            # {...}} and the OpenAI wire format {"name", "arguments": "<json>"}
            # that quantized checkpoints sometimes emit (both are abundant in
            # tool-call training data). A JSON-encoded string is decoded; only
            # a non-object result falls back to {}.
            args = parsed.get("args")
            if args is None:
                args = parsed.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    args = None
            if not isinstance(args, dict):
                args = {}
            tool_calls.append(
                {
                    "name": str(parsed["name"]),
                    "arguments": json.dumps(
                        args if isinstance(args, dict) else {},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            )

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
    model_path: str | None = None,
) -> OutputParserFactory | None:
    """Detect a protocol-specific output parser for the model, if needed.

    ``model_name`` drives detection (string matching) and may be a display
    id rather than a directory since #2178. Pass ``model_path`` when the
    filesystem path is available so parser sessions can locate
    tokenizer.json for their streaming detokenizers.
    """
    session_model_path = model_path or model_name

    model_type = model_config.get("model_type") if model_config else None
    if model_type == _BAILING_HYBRID_MODEL_TYPE:
        marker_token_ids = {
            token_id
            for marker in _BAILING_ROLE_MARKERS
            if (token_id := _token_id_for_text(tokenizer, marker)) is not None
        }
        if marker_token_ids:
            return OutputParserFactory(
                kind="bailing_hybrid",
                create_session=lambda session_tokenizer: (
                    BailingHybridOutputParserSession(
                        session_tokenizer,
                        marker_token_ids,
                        model_path=session_model_path,
                    )
                ),
                create_session_with_tools=lambda session_tokenizer, tools: (
                    BailingHybridOutputParserSession(
                        session_tokenizer,
                        marker_token_ids,
                        model_path=session_model_path,
                        tools=tools,
                    )
                ),
                thinking_start_text="<think>",
                thinking_start_output_text="<think>\n",
                protocol_marker_texts=_BAILING_ROLE_MARKERS,
            )

    if is_harmony_model(model_name, model_config):
        temp_parser = HarmonyStreamingParser(tokenizer)
        return OutputParserFactory(
            kind="harmony",
            create_session=lambda session_tokenizer: HarmonyOutputParserSession(
                session_tokenizer,
                model_path=session_model_path,
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
                model_path=session_model_path,
            ),
            stop_token_ids=set(),
            thinking_start_text="<|channel>thought",
            thinking_start_output_text="<think>\n",
            thinking_end_text="<channel|>",
            protocol_marker_texts=(
                _OPEN_MARKER_BARE,
                _CLOSE_MARKER,
                _TURN_END_MARKER,
                _TOOL_RESPONSE_OPEN,
                _TOOL_RESPONSE_CLOSE,
            ),
        )

    if _is_deepseek_v4_model(model_name, tokenizer, model_config):
        return OutputParserFactory(
            kind="deepseek_v4",
            create_session=lambda session_tokenizer: DeepSeekV4OutputParserSession(
                session_tokenizer,
                model_path=session_model_path,
            ),
            stop_token_ids=set(),
            protocol_marker_texts=(
                _DEEPSEEK_V4_TOOL_CALL_START,
                _DEEPSEEK_V4_TOOL_CALL_END,
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
                model_path=session_model_path,
            ),
            stop_token_ids=set(),
            thinking_end_text="</think>",
        )

    if _is_inkling_model(model_name, model_config):
        inkling_stop_ids = set()
        end_sampling_id = _token_id_for_text(tokenizer, _INKLING_END_SAMPLING)
        if end_sampling_id is not None:
            inkling_stop_ids.add(end_sampling_id)

        return OutputParserFactory(
            kind="inkling",
            create_session=lambda session_tokenizer: InklingOutputParserSession(
                session_tokenizer,
                model_path=session_model_path,
            ),
            stop_token_ids=inkling_stop_ids,
            thinking_start_text=_INKLING_CONTENT_THINKING,
            thinking_start_output_text="<think>\n",
            thinking_end_text=_INKLING_END_MESSAGE,
            thinking_end_trailing_text=(
                _INKLING_MESSAGE_MODEL + _INKLING_CONTENT_TEXT
            ),
            protocol_marker_texts=_INKLING_MARKERS,
        )

    if _is_muse_glimmer_model(model_name, model_config):
        from .muse_glimmer import (
            _MUSE_END_OF_TEXT,
            _MUSE_EOM,
            _MUSE_EOT,
            _MUSE_MARKERS,
            _MUSE_MESSAGE,
            _MUSE_START,
            MuseGlimmerOutputParserSession,
        )

        muse_stop_ids = set()
        for stop_marker in (_MUSE_EOT, _MUSE_END_OF_TEXT):
            stop_id = _token_id_for_text(tokenizer, stop_marker)
            if stop_id is not None:
                muse_stop_ids.add(stop_id)

        return OutputParserFactory(
            kind="muse_glimmer",
            create_session=lambda session_tokenizer: MuseGlimmerOutputParserSession(
                session_tokenizer,
                model_path=session_model_path,
            ),
            create_session_with_tools=lambda session_tokenizer, tools: (
                MuseGlimmerOutputParserSession(
                    session_tokenizer,
                    model_path=session_model_path,
                    tools=tools,
                )
            ),
            stop_token_ids=muse_stop_ids,
            thinking_start_output_text="<think>\n",
            # A forced thinking close ends the reasoning message and opens
            # the visible-answer message.
            thinking_end_text=_MUSE_EOM,
            thinking_end_trailing_text=(
                _MUSE_START + "assistant to=user" + _MUSE_MESSAGE
            ),
            protocol_marker_texts=_MUSE_MARKERS,
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
                model_path=session_model_path,
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
