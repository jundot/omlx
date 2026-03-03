# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-mlx (https://github.com/vllm-project/vllm-mlx).
"""
Request management for oMLX continuous batching.

This module provides Request and RequestStatus classes adapted from vLLM's
request management system, simplified for MLX backend.
"""

import enum
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from .cache.paged_cache import BlockTable


class RequestStatus(enum.IntEnum):
    """Status of a request in the scheduling system."""

    # Request is waiting to be scheduled
    WAITING = enum.auto()
    # Request is currently being processed (generating tokens)
    RUNNING = enum.auto()
    # Request was preempted and needs to be resumed
    PREEMPTED = enum.auto()
    # Request finished successfully (hit stop token)
    FINISHED_STOPPED = enum.auto()
    # Request finished due to max_tokens limit
    FINISHED_LENGTH_CAPPED = enum.auto()
    # Request was aborted by user
    FINISHED_ABORTED = enum.auto()

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        """Check if the status indicates a finished request."""
        return status > RequestStatus.PREEMPTED

    @staticmethod
    def get_finish_reason(status: "RequestStatus") -> Optional[str]:
        """Get the finish reason string for a finished status."""
        if status == RequestStatus.FINISHED_STOPPED:
            return "stop"
        elif status == RequestStatus.FINISHED_LENGTH_CAPPED:
            return "length"
        elif status == RequestStatus.FINISHED_ABORTED:
            return "abort"
        return None


@dataclass
class SamplingParams:
    """Sampling parameters for text generation."""

    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0  # 0 means disabled
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    stop: Optional[List[str]] = None
    stop_token_ids: Optional[List[int]] = None

    # Logprobs settings (memory optimization: disabled by default)
    logprobs: bool = False  # Whether to return logprobs
    top_logprobs: Optional[int] = None  # Number of top logprobs (1-20)

    def __post_init__(self):
        if self.stop is None:
            self.stop = []
        if self.stop_token_ids is None:
            self.stop_token_ids = []


@dataclass
class Request:
    """
    Represents a single inference request in the scheduling system.

    Adapted from vLLM's Request class with simplifications for MLX backend.

    Attributes:
        request_id: Unique identifier for this request
        prompt: The input prompt (string or token ids)
        prompt_token_ids: Tokenized prompt
        sampling_params: Parameters for generation
        arrival_time: When the request was received
        status: Current status of the request
        num_prompt_tokens: Number of tokens in the prompt
        num_computed_tokens: Number of tokens processed so far
        output_token_ids: Generated token ids
        output_text: Generated text (decoded)
    """

    request_id: str
    prompt: Union[str, List[int]]
    sampling_params: SamplingParams
    arrival_time: float = field(default_factory=time.time)
    priority: int = 0  # Lower is higher priority

    # Set after tokenization
    prompt_token_ids: Optional[List[int]] = None
    num_prompt_tokens: int = 0

    # Generation state
    status: RequestStatus = RequestStatus.WAITING
    num_computed_tokens: int = 0
    output_token_ids: List[int] = field(default_factory=list)
    output_text: str = ""

    # For BatchGenerator integration
    batch_uid: Optional[int] = None  # UID assigned by BatchGenerator

    # Prefix cache fields
    prompt_cache: Optional[List[Any]] = None  # Cached KV state from prefix cache
    cached_tokens: int = 0  # Number of tokens retrieved from cache
    remaining_tokens: Optional[List[int]] = None  # Tokens still needing processing

    # Paged cache fields (for BlockAwarePrefixCache)
    block_table: Optional["BlockTable"] = None  # Block table for paged cache
    shared_prefix_blocks: int = 0  # Number of shared prefix blocks

    # Multimodal content (images, video)
    images: Optional[List[Any]] = None
    videos: Optional[List[Any]] = None

    # VLM (Vision-Language Model) fields
    vlm_inputs_embeds: Optional[Any] = None  # Pre-computed vision+text embeddings (mx.array)
    vlm_extra_kwargs: Optional[Dict[str, Any]] = None  # Model-specific kwargs (e.g., position_ids)
    vlm_image_hash: Optional[str] = None  # SHA256 hash of images for prefix cache

    # Metadata
    finish_reason: Optional[str] = None

    # Reasoning model support (for models with <think> tags)
    needs_think_prefix: bool = False    # True if prompt ends with <think> token
    think_prefix_sent: bool = False     # Track if prefix already sent

    # Harmony model support (gpt-oss models)
    is_harmony_model: bool = False      # True if model uses Harmony format

    @property
    def num_output_tokens(self) -> int:
        """Number of output tokens generated so far."""
        return len(self.output_token_ids)

    @property
    def num_tokens(self) -> int:
        """Total number of tokens (prompt + output)."""
        return self.num_prompt_tokens + self.num_output_tokens

    @property
    def max_tokens(self) -> int:
        """Maximum output tokens for this request."""
        return self.sampling_params.max_tokens

    def is_finished(self) -> bool:
        """Check if request has finished."""
        return RequestStatus.is_finished(self.status)

    def get_finish_reason(self) -> Optional[str]:
        """Get the finish reason if finished."""
        if self.finish_reason:
            return self.finish_reason
        return RequestStatus.get_finish_reason(self.status)

    def append_output_token(self, token_id: int) -> None:
        """Append a generated token to the output."""
        self.output_token_ids.append(token_id)
        self.num_computed_tokens += 1

    def set_finished(self, status: RequestStatus, reason: Optional[str] = None) -> None:
        """Mark the request as finished."""
        self.status = status
        self.finish_reason = reason or RequestStatus.get_finish_reason(status)

    def __lt__(self, other: "Request") -> bool:
        """Compare requests for priority queue ordering."""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.arrival_time < other.arrival_time

    def __hash__(self) -> int:
        return hash(self.request_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Request):
            return False
        return self.request_id == other.request_id


@dataclass
class RequestOutput:
    """
    Output for a single request after a generation step.

    This is returned by the engine to communicate results back to the API layer.
    """

    request_id: str
    # New tokens generated in this step
    new_token_ids: List[int] = field(default_factory=list)
    new_text: str = ""
    # Cumulative output
    output_token_ids: List[int] = field(default_factory=list)
    output_text: str = ""
    # Status
    finished: bool = False
    finish_reason: Optional[str] = None
    # Timing
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # Tool calls (for Harmony and other models with tool calling support)
    tool_calls: Optional[List[Dict[str, str]]] = None
    # Prefix cache stats
    cached_tokens: int = 0
    # Error message (set when engine encounters an unrecoverable error)
    error: Optional[str] = None

    @property
    def usage(self) -> Dict[str, int]:
        """Return usage statistics compatible with OpenAI API."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }
