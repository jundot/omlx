"""Qwen3.5/3.6 prefill kernels used by oMLX runtime patches."""

from . import fast
from .gdn import gated_delta_blocked_seq, gated_delta_chunked_metal

__all__ = ["fast", "gated_delta_blocked_seq", "gated_delta_chunked_metal"]
