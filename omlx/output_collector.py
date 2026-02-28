# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-mlx (https://github.com/vllm-project/vllm-mlx).
"""
Output collector for streaming with low-latency optimizations.

This module implements the RequestOutputCollector pattern from vLLM,
providing non-blocking output collection with intelligent aggregation.
"""

import asyncio
from dataclasses import dataclass, field
from typing import List, Optional

from .request import RequestOutput


class RequestOutputCollector:
    """
    Per-request output collector with smart buffering.

    This class implements the vLLM pattern for efficient streaming:
    - Non-blocking get_nowait() to avoid unnecessary task switches
    - Output aggregation when producer is faster than consumer
    - Event-based signaling for efficient waiting
    - Tracking of active consumers for yield optimization

    Usage:
        collector = RequestOutputCollector()

        # Producer side (engine loop)
        collector.put(output)

        # Consumer side (streaming generator)
        output = collector.get_nowait() or await collector.get()
    """

    # Global counter of collectors with waiting consumers
    # Used to optimize: only yield when someone is waiting
    _waiting_consumers: int = 0

    def __init__(self, aggregate: bool = True):
        """
        Initialize the collector.

        Args:
            aggregate: If True, merge outputs when producer gets ahead.
                       This prevents buffer explosion under load.
        """
        self.output: Optional[RequestOutput] = None
        self.ready = asyncio.Event()
        self.aggregate = aggregate
        self._is_waiting = False

    def put(self, output: RequestOutput) -> None:
        """
        Put an output into the collector (non-blocking).

        If aggregation is enabled and an output already exists,
        the new output is merged with the existing one.

        Args:
            output: The RequestOutput to store
        """
        if self.output is None:
            self.output = output
        elif self.aggregate:
            # Merge: combine tokens when producer is ahead
            self.output = self._merge_outputs(self.output, output)
        else:
            # Replace: just use the new output
            self.output = output
        self.ready.set()

    def get_nowait(self) -> Optional[RequestOutput]:
        """
        Get output without blocking.

        This avoids task switching when output is available,
        reducing latency under load.

        Returns:
            The output if available, None otherwise
        """
        output = self.output
        if output is not None:
            self.output = None
            self.ready.clear()
        return output

    async def get(self) -> RequestOutput:
        """
        Get output, blocking only if none available.

        This method blocks until an output is available.
        For low-latency streaming, prefer:
            output = collector.get_nowait() or await collector.get()

        Returns:
            The RequestOutput
        """
        # Track that we're waiting (for yield optimization)
        if not self._is_waiting:
            self._is_waiting = True
            RequestOutputCollector._waiting_consumers += 1
        try:
            while self.output is None:
                await self.ready.wait()
            output = self.get_nowait()
            # This should never be None after wait, but satisfy type checker
            assert output is not None
            return output
        finally:
            if self._is_waiting:
                self._is_waiting = False
                RequestOutputCollector._waiting_consumers -= 1

    def _merge_outputs(
        self,
        existing: RequestOutput,
        new: RequestOutput,
    ) -> RequestOutput:
        """
        Merge two outputs when producer gets ahead of consumer.

        This combines the token lists and text, keeping the latest
        status information.

        Args:
            existing: The existing output in the buffer
            new: The new output to merge

        Returns:
            Merged RequestOutput
        """
        # Combine new tokens
        merged_new_token_ids = existing.new_token_ids + new.new_token_ids
        merged_new_text = existing.new_text + new.new_text

        return RequestOutput(
            request_id=new.request_id,
            new_token_ids=merged_new_token_ids,
            new_text=merged_new_text,
            output_token_ids=new.output_token_ids,  # Use latest cumulative
            output_text=new.output_text,  # Use latest cumulative
            finished=new.finished,
            finish_reason=new.finish_reason,
            prompt_tokens=new.prompt_tokens,
            completion_tokens=new.completion_tokens,
            tool_calls=new.tool_calls,  # Preserve tool_calls for Harmony models
            cached_tokens=new.cached_tokens,
            error=new.error or existing.error,
        )

    def clear(self) -> None:
        """Clear any pending output."""
        self.output = None
        self.ready.clear()
        if self._is_waiting:
            self._is_waiting = False
            RequestOutputCollector._waiting_consumers -= 1

    @classmethod
    def has_waiting_consumers(cls) -> bool:
        """Check if any collector has waiting consumers.

        Used by engine to optimize: only yield when someone is waiting.
        """
        return cls._waiting_consumers > 0


@dataclass
class RequestStreamState:
    """
    Tracks streaming state for a request.

    This is used to implement stream_interval batching,
    allowing tokens to be accumulated before sending.
    """

    stream_interval: int = 1
    sent_tokens: int = 0

    def should_send(self, total_tokens: int, finished: bool) -> bool:
        """
        Determine if output should be sent based on stream_interval.

        Args:
            total_tokens: Total tokens generated so far
            finished: Whether generation is complete

        Returns:
            True if output should be sent
        """
        # Always send on finish
        if finished:
            return True
        # Always send first token (for low TTFT)
        if self.sent_tokens == 0:
            return True
        # Send if we've accumulated enough tokens
        return (total_tokens - self.sent_tokens) >= self.stream_interval

    def mark_sent(self, total_tokens: int) -> None:
        """
        Update state after sending output.

        Args:
            total_tokens: Total tokens at time of send
        """
        self.sent_tokens = total_tokens
