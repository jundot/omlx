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

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncoding,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    StreamableParser,
    SystemContent,
    ToolDescription,
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


_REASONING_EFFORT_MAP = {
    "low": ReasoningEffort.LOW,
    "medium": ReasoningEffort.MEDIUM,
    "high": ReasoningEffort.HIGH,
}

# Shared across all render_harmony_prompt() calls — the encoding is
# stateless and cheap to reuse (same pattern as parse_tool_calls_from_tokens
# below, which reloads on every call; here we pay the cost once).
_GPT_OSS_ENCODING: HarmonyEncoding | None = None


def _get_gpt_oss_encoding() -> HarmonyEncoding:
    global _GPT_OSS_ENCODING
    if _GPT_OSS_ENCODING is None:
        _GPT_OSS_ENCODING = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return _GPT_OSS_ENCODING


def _extract_text(content: Any) -> str:
    """Flatten OpenAI content (str or list of blocks) into plain text."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif "text" in block:
                parts.append(block["text"])
        elif isinstance(block, str):
            parts.append(block)
    return "".join(parts)


def _tool_description_from_openai(tool: dict[str, Any]) -> ToolDescription | None:
    """Convert an OpenAI tool spec (``{"type":"function","function":{...}}``) to a ToolDescription."""
    spec = tool.get("function") if tool.get("type") == "function" else tool
    if not isinstance(spec, dict):
        return None
    name = spec.get("name")
    if not name:
        return None
    description = spec.get("description", "") or ""
    parameters = spec.get("parameters") or spec.get("input_schema") or {
        "type": "object",
        "properties": {},
    }
    return ToolDescription.new(name, description, parameters=parameters)


def render_harmony_prompt(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> str:
    """Render chat messages + tools into a Harmony-format prompt string.

    Used as a tokenizer-free fallback for gpt-oss models whose packaged
    tokenizer does not ship with a ``chat_template``.  The official
    ``openai-harmony`` library does the heavy lifting; this wrapper
    translates the OpenAI-style inputs oMLX already uses.

    Args:
        messages: OpenAI chat messages.  ``system`` becomes developer
            instructions; ``assistant`` with a ``tool_calls`` field is
            emitted on the commentary channel; ``tool`` messages are
            rendered as tool responses.
        tools: Optional OpenAI-format function tools.  Also accepts raw
            function specs (``{"name", "description", "parameters"}``).
        chat_template_kwargs: Optional template-style kwargs.  Supports
            ``reasoning_effort`` ("low"/"medium"/"high") and
            ``conversation_start_date``.

    Returns:
        A decoded Harmony prompt string ready to feed into ``generate``.
    """
    ct = chat_template_kwargs or {}

    system_content = SystemContent.new()
    effort = ct.get("reasoning_effort")
    if isinstance(effort, str):
        mapped = _REASONING_EFFORT_MAP.get(effort.lower())
        if mapped is not None:
            system_content = system_content.with_reasoning_effort(mapped)
    start_date = ct.get("conversation_start_date")
    if isinstance(start_date, str) and start_date:
        system_content = system_content.with_conversation_start_date(start_date)

    system_texts: list[str] = []
    convo_msgs: list[Message] = []
    # Maps tool_call_id -> function name for resolving ``role=tool`` messages
    # whose ``name`` field is omitted (OpenAI spec allows this).
    tool_call_names: dict[str, str] = {}

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")

        if role == "system":
            text = _extract_text(msg.get("content"))
            if text:
                system_texts.append(text)
            continue

        if role == "user":
            text = _extract_text(msg.get("content"))
            convo_msgs.append(Message.from_role_and_content(Role.USER, text))
            continue

        if role == "assistant":
            text = _extract_text(msg.get("content"))
            if text:
                convo_msgs.append(
                    Message.from_role_and_content(Role.ASSISTANT, text)
                        .with_channel("final")
                )
            for tc in msg.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function")
                if not isinstance(fn, dict):
                    continue
                name = fn.get("name") or ""
                if not name:
                    continue
                tc_id = tc.get("id")
                if isinstance(tc_id, str):
                    tool_call_names[tc_id] = name
                args = fn.get("arguments")
                if isinstance(args, (dict, list)):
                    args_str = json.dumps(args, ensure_ascii=False)
                else:
                    args_str = args or ""
                convo_msgs.append(
                    Message.from_role_and_content(Role.ASSISTANT, args_str)
                        .with_channel("commentary")
                        .with_recipient(f"functions.{name}")
                        .with_content_type("<|constrain|> json")
                )
            continue

        if role == "tool":
            text = _extract_text(msg.get("content"))
            # Prefer an explicit ``name``; fall back to the function name
            # recorded when the matching assistant tool_call was emitted.
            name = msg.get("name")
            if not name:
                tc_id = msg.get("tool_call_id")
                if isinstance(tc_id, str):
                    name = tool_call_names.get(tc_id)
            if not name:
                # No recoverable name — skip rather than fabricate one, which
                # would confuse the model about which function it called.
                logger.warning(
                    "Skipping tool message: no ``name`` and no matching "
                    "tool_call_id in earlier assistant message."
                )
                continue
            convo_msgs.append(
                Message.from_author_and_content(
                    Author.new(Role.TOOL, f"functions.{name}"),
                    text,
                ).with_channel("commentary")
            )
            continue

        # Unknown roles: drop silently — openai-harmony would reject them.

    developer_content: DeveloperContent | None = None
    instructions = "\n\n".join(t for t in system_texts if t)
    if instructions:
        developer_content = DeveloperContent.new().with_instructions(instructions)
    if tools:
        tool_descs = [td for t in tools if (td := _tool_description_from_openai(t))]
        if tool_descs:
            developer_content = (developer_content or DeveloperContent.new()) \
                .with_function_tools(tool_descs)

    conv_messages: list[Message] = [
        Message.from_role_and_content(Role.SYSTEM, system_content),
    ]
    if developer_content is not None:
        conv_messages.append(Message.from_role_and_content(Role.DEVELOPER, developer_content))
    conv_messages.extend(convo_msgs)

    encoding = _get_gpt_oss_encoding()
    tokens = encoding.render_conversation_for_completion(
        Conversation.from_messages(conv_messages), Role.ASSISTANT
    )
    return encoding.decode(tokens)


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
