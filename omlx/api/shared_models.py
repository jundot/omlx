# SPDX-License-Identifier: Apache-2.0
"""Shared models and utilities for API responses."""

import time
import uuid
from enum import Enum

from pydantic import BaseModel


class IDPrefix(str, Enum):
    """Prefixes for generated IDs."""

    CHAT_COMPLETION = "chatcmpl"
    COMPLETION = "cmpl"
    MESSAGE = "msg"
    EMBEDDING = "emb"
    RERANK = "rerank"
    RESPONSE = "resp"
    FUNCTION_CALL = "fc"


def generate_id(prefix: IDPrefix, length: int = 8) -> str:
    """Generate a unique ID with the given prefix.

    Args:
        prefix: The ID prefix to use
        length: Length of the random suffix (default 8)

    Returns:
        Generated ID string (e.g., "chatcmpl-abc12345")
    """
    if prefix == IDPrefix.MESSAGE:
        # Anthropic style: msg_<24-char-hex>
        return f"msg_{uuid.uuid4().hex[:24]}"
    if prefix == IDPrefix.RESPONSE:
        return f"resp_{uuid.uuid4().hex[:24]}"
    if prefix == IDPrefix.FUNCTION_CALL:
        return f"fc_{uuid.uuid4().hex[:8]}"
    return f"{prefix.value}-{uuid.uuid4().hex[:length]}"


def get_unix_timestamp() -> int:
    """Get current Unix timestamp.

    Returns:
        Current time as Unix timestamp (integer seconds since epoch)
    """
    return int(time.time())


class BaseUsage(BaseModel):
    """Base class for token usage statistics.

    This provides a foundation for both OpenAI-style (prompt_tokens/completion_tokens)
    and Anthropic-style (input_tokens/output_tokens) usage tracking.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def model_post_init(self, __context) -> None:
        """Calculate total_tokens and sync Anthropic-style aliases."""
        if self.total_tokens == 0 and (self.prompt_tokens > 0 or self.completion_tokens > 0):
            object.__setattr__(
                self,
                "total_tokens",
                self.prompt_tokens + self.completion_tokens,
            )
        if self.input_tokens == 0 and self.prompt_tokens > 0:
            object.__setattr__(self, "input_tokens", self.prompt_tokens)
        if self.output_tokens == 0 and self.completion_tokens > 0:
            object.__setattr__(self, "output_tokens", self.completion_tokens)
