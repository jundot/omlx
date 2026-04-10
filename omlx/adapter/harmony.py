# SPDX-License-Identifier: Apache-2.0
"""
Harmony format streaming parser for gpt-oss models.

Uses the official openai-harmony package for robust parsing.

Harmony protocol uses special tokens to structure messages:
- <|start|>: Begin message header
- <|channel|>: Mark channel type
- <|message|>: Transition to content
- <|end|>: End message
- <|return|>: Model completion signal
- <|call|>: Tool invocation signal

Message structure: <|start|>{header}<|channel|>{channel_name}<|message|>{content}<|end|>

Channels:
- final: User-visible response (plain text)
- analysis: Chain-of-thought reasoning (wrapped in <think>...</think> for streaming)
- commentary: Tool/function calls (non-streaming only)
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from openai_harmony import (
    HarmonyEncoding,
    Role,
    StreamableParser,
    load_harmony_encoding,
)

logger = logging.getLogger(__name__)

# Pattern to match <think>...</think> blocks
_THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

# Pre-allocated constants
_THINK_OPEN = "<think>\n"
_THINK_CLOSE = "</think>\n"

# Harmony special tokens that should not be streamed
_HARMONY_SPECIAL_TOKENS = [
    "<|start|>",
    "<|end|>",
    "<|message|>",
    "<|channel|>",
    "<|return|>",
    "<|call|>",
    "<|constrain|>",
]


def preprocess_harmony_messages(
    messages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Preprocess messages for Harmony (gpt-oss) models.

    - Strips <think> tags from assistant messages
    - Keeps tool role messages unchanged (chat_template handles conversion)

    The chat_template expects standard OpenAI format:
    - role: "tool" with tool_call_id and content
    - It uses last_tool_call.name from the previous assistant message
    - Generates: <|start|>functions.{name} to=assistant<|channel|>commentary<|message|>{content|tojson}<|end|>

    Args:
        messages: List of message dicts with 'role' and 'content' keys

    Returns:
        Messages preprocessed for Harmony format
    """
    if not messages:
        return []

    result = []

    for msg in messages:
        # Validate message is a dict
        if not isinstance(msg, dict):
            logger.warning(f"Skipping non-dict message: {type(msg)}")
            continue

        role = msg.get("role")

        if role == "assistant":
            content = msg.get("content", "")
            # Ensure content is a string (could be list in some formats)
            if isinstance(content, str):
                # Strip <think> tags
                if content and "<think>" in content:
                    content = _THINK_TAG_PATTERN.sub("", content).strip()
                    msg = {**msg, "content": content}
            elif content is not None:
                # Non-string content (e.g., list) - log but don't modify
                logger.debug(f"Assistant message has non-string content: {type(content)}")

            result.append(msg)

        else:
            # Pass through all other messages (user, tool, system, etc.) unchanged
            # Chat template handles tool messages directly using last_tool_call.name
            result.append(msg)

    return result


def _get_special_token_ids(tokenizer: Any) -> set[int]:
    """
    Get special token IDs from model tokenizer.

    Args:
        tokenizer: The model's tokenizer

    Returns:
        Set of special token IDs
    """
    special_ids = set()
    for token in _HARMONY_SPECIAL_TOKENS:
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if isinstance(token_id, int) and token_id >= 0:
                special_ids.add(token_id)
            else:
                logger.debug(f"Harmony special token '{token}' not found in tokenizer")
        except Exception as e:
            logger.debug(f"Failed to get ID for Harmony token '{token}': {e}")
    return special_ids


@dataclass
class HarmonyStreamingParser:
    """
    Streaming parser for Harmony format using official openai-harmony package.

    Parses tokens incrementally and routes them to appropriate channels.
    Returns token IDs instead of decoded text to allow proper UTF-8 handling
    via streaming detokenizer in the caller.

    Output routing:
    - analysis channel -> stream only (wrapped in <think>...</think>)
    - final channel -> stream and visible (stored in output_text)
    - commentary channel -> buffered for tool calls (non-streaming)

    The parser returns:
    - control_text: Control strings like <think>, </think>
    - stream_token: Token ID to stream (None if not streaming)
    - visible_token: Token ID to store (None if not storing)
    - is_stop: Whether this is a stop signal
    """

    tokenizer: Any

    # Internal state (initialized in __post_init__)
    _encoding: HarmonyEncoding = field(init=False, repr=False)
    _parser: StreamableParser = field(init=False, repr=False)
    _stop_tokens: set[int] = field(init=False, default_factory=set)
    _special_tokens: set[int] = field(init=False, default_factory=set)

    # <think> tag state
    _in_think_tag: bool = field(init=False, default=False)
    _prev_channel: str | None = field(init=False, default=None)

    # Passthrough mode: activated when streaming parser encounters an
    # unrecoverable error.  Tokens are still accumulated by the scheduler
    # (request.append_output_token) so parse_tool_calls_from_tokens can
    # extract tool calls at finalization.
    _passthrough_mode: bool = field(init=False, default=False)

    def __post_init__(self):
        """Initialize the official Harmony parser."""
        self._encoding = load_harmony_encoding("HarmonyGptOss")
        # role=None allows the parser to handle tool-call headers
        # (e.g. "assistant to=functions.Write") which Role.ASSISTANT rejects.
        self._parser = StreamableParser(self._encoding, None, strict=False)
        self._stop_tokens = set(self._encoding.stop_tokens_for_assistant_actions())
        self._special_tokens = _get_special_token_ids(self.tokenizer)

        # Prime the parser with "<|start|>assistant" tokens.  The chat
        # template already includes these in the prompt, so the model's
        # first output token is <|channel|>, not <|start|>.  Without
        # priming, the parser rejects <|channel|> as unexpected.
        self._prime_parser(self._parser)

        logger.info(
            f"Harmony parser initialized: {len(self._special_tokens)} special tokens, "
            f"{len(self._stop_tokens)} stop tokens"
        )

    def _prime_parser(self, parser: StreamableParser) -> None:
        """Feed '<|start|>assistant' header tokens so parser expects <|channel|> next."""
        start_tokens = self._encoding.encode(
            "<|start|>assistant", allowed_special="all"
        )
        for t in start_tokens:
            parser.process(t)

    def process_token(
        self, token_id: int
    ) -> tuple[str, int | None, int | None, bool]:
        """
        Process a single token and return routing information.

        This method routes tokens to appropriate channels without decoding.
        The caller should use streaming detokenizer to decode the returned
        token IDs for proper UTF-8 handling.

        Args:
            token_id: The token ID to process.

        Returns:
            Tuple of:
            - control_text: Control strings (<think>, </think>, etc.)
            - stream_token: Token ID to stream (None to skip)
            - visible_token: Token ID to store in output_text (None to skip)
            - is_stop: True if this is a stop token
        """
        # Check if this is a special token (should not be streamed)
        is_special_token = token_id in self._special_tokens
        is_stop = token_id in self._stop_tokens

        # Passthrough: parser crashed earlier, buffer all tokens silently.
        # Tokens are still tracked by the scheduler for non-streaming tool
        # call extraction at finalization.
        if self._passthrough_mode:
            return "", None, None, is_stop

        try:
            self._parser.process(token_id)
        except Exception as e:
            logger.warning(
                f"Harmony streaming parser error, switching to passthrough: {e}"
            )
            self._passthrough_mode = True
            control_text = ""
            if self._in_think_tag:
                control_text = _THINK_CLOSE
                self._in_think_tag = False
            return control_text, None, None, is_stop

        channel = self._parser.current_channel
        control_text = ""

        # Handle channel transitions for <think> tags
        if channel != self._prev_channel:
            # Close previous analysis channel
            if self._in_think_tag and self._prev_channel == "analysis":
                control_text += _THINK_CLOSE
                self._in_think_tag = False
            # Open new analysis channel
            if channel == "analysis" and not self._in_think_tag:
                control_text += _THINK_OPEN
                self._in_think_tag = True
            self._prev_channel = channel

        # Special tokens should never be streamed or stored
        if is_special_token:
            return control_text, None, None, is_stop

        # Route based on channel
        if channel == "final":
            # final: stream AND store (same token for both)
            return control_text, token_id, token_id, is_stop
        elif channel == "analysis":
            # analysis: stream only (wrapped in <think>)
            return control_text, token_id, None, is_stop
        elif channel is None:
            # Channel not yet determined (still in header parsing)
            # Buffer token but don't stream
            return control_text, None, None, is_stop
        else:
            # commentary etc: buffer only (for tool calls)
            return control_text, None, None, is_stop

    def get_stop_token_ids(self) -> set[int]:
        """Get Harmony stop token IDs."""
        return self._stop_tokens

    def get_tool_calls(self) -> list[dict[str, str]]:
        """Get accumulated tool calls from parsed messages."""
        tool_calls = []
        try:
            messages = self._parser.messages
            if not messages:
                return tool_calls

            for msg in messages:
                if not msg.recipient or not msg.recipient.startswith("functions."):
                    continue

                name = msg.recipient[10:]  # Remove "functions." prefix
                content = ""

                # Safely iterate over content
                msg_content = getattr(msg, "content", None)
                if msg_content is not None:
                    for c in msg_content:
                        text = getattr(c, "text", None)
                        if isinstance(text, str):
                            content += text

                tool_calls.append({"name": name, "arguments": content})
                logger.info(f"Extracted tool call: {name}, arguments={content}")

        except Exception as e:
            logger.warning(f"Error extracting tool calls: {e}")

        return tool_calls

    def finalize(self) -> str:
        """
        Finalize parsing and close any open tags.

        Returns:
            Any remaining control text (e.g., closing </think> tag).
        """
        try:
            self._parser.process_eos()
        except Exception as e:
            # Can fail if message is incomplete (e.g., missing <|end|>)
            # This is expected in some cases, so just log and continue
            logger.debug(f"Harmony parser process_eos failed (expected for incomplete messages): {e}")

        if self._in_think_tag:
            self._in_think_tag = False
            return _THINK_CLOSE
        return ""

    def reset(self) -> None:
        """Reset parser state for a new request."""
        self._parser = StreamableParser(self._encoding, None, strict=False)
        self._prime_parser(self._parser)
        self._in_think_tag = False
        self._prev_channel = None
        self._passthrough_mode = False

    @property
    def current_channel(self) -> str | None:
        """Get current channel name."""
        return self._parser.current_channel

    @property
    def current_recipient(self) -> str | None:
        """Get current recipient (for tool calls)."""
        return self._parser.current_recipient


def parse_tool_calls_from_tokens(
    token_ids: list[int],
    prepend_start: bool = True,
) -> tuple[str, str, list[dict[str, str]]]:
    """
    Parse a complete Harmony token sequence (non-streaming).

    Args:
        token_ids: Model output token ID list
        prepend_start: Whether to prepend "<|start|>assistant" tokens.
            Set to False if token_ids already includes start tokens.

    Returns:
        (output_text, analysis_text, tool_calls)
        - output_text: Text from the final channel
        - analysis_text: Chain-of-thought text from the analysis channel
        - tool_calls: [{"name": "...", "arguments": "..."}]
    """
    if not token_ids:
        return "", "", []

    try:
        encoding = load_harmony_encoding("HarmonyGptOss")

        # The model's chat template includes "<|start|>assistant" in the prompt,
        # so the model generates starting from "<|channel|>".
        # We need to prepend "<|start|>assistant" for proper parsing.
        if prepend_start:
            start_tokens = encoding.encode("<|start|>assistant", allowed_special="all")
            full_token_ids = start_tokens + list(token_ids)
        else:
            full_token_ids = list(token_ids)

        # Decode tokens for debugging
        decoded_text = encoding.decode(full_token_ids)
        logger.info(f"parse_tool_calls input ({len(full_token_ids)} tokens): {decoded_text[:300]}...")

        messages = encoding.parse_messages_from_completion_tokens(
            full_token_ids,
            role=Role.ASSISTANT,
            strict=False,
        )

        logger.info(f"Parsed {len(messages)} messages")
        for i, msg in enumerate(messages):
            content_count = len(msg.content) if msg.content else 0
            logger.info(
                f"Message {i}: channel={msg.channel}, recipient={msg.recipient}, "
                f"content_count={content_count}"
            )

        output_text = ""
        analysis_text = ""
        tool_calls = []

        for msg in messages:
            # Safely get content
            msg_content = getattr(msg, "content", None)
            if msg_content is None:
                continue

            if msg.channel == "final":
                # Extract text from final channel
                for content in msg_content:
                    text = getattr(content, "text", None)
                    if isinstance(text, str):
                        output_text += text

            elif msg.channel == "analysis":
                # Extract chain-of-thought text from analysis channel
                for content in msg_content:
                    text = getattr(content, "text", None)
                    if isinstance(text, str):
                        analysis_text += text

            elif msg.recipient and msg.recipient.startswith("functions."):
                # Extract tool calls from commentary channel
                name = msg.recipient[10:]  # Remove "functions." prefix
                arguments = ""
                for content in msg_content:
                    text = getattr(content, "text", None)
                    if isinstance(text, str):
                        arguments += text
                tool_calls.append({"name": name, "arguments": arguments})

        return output_text, analysis_text, tool_calls

    except Exception as e:
        logger.warning(f"Error parsing tool calls from tokens: {e}")
        return "", "", []
