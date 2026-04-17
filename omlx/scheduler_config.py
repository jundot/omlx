# SPDX-License-Identifier: Apache-2.0
"""Lightweight scheduler config types.

These definitions are kept separate from ``omlx.scheduler`` so callers that
only need configuration objects do not import MLX/Metal runtime code.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SchedulingPolicy(Enum):
    """Scheduling policy for request ordering."""

    FCFS = "fcfs"
    PRIORITY = "priority"


@dataclass
class SchedulerConfig:
    """Configuration for the scheduler."""

    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    policy: SchedulingPolicy = SchedulingPolicy.FCFS
    completion_batch_size: int = 32
    prefill_step_size: int = 2048
    paged_cache_block_size: int = 256
    max_cache_blocks: Optional[int] = None
    initial_cache_blocks: int = 256
    paged_ssd_cache_dir: Optional[str] = None
    paged_ssd_cache_max_size: int = 100 * 1024 * 1024 * 1024
    hot_cache_max_size: int = 0
    model_name: str = ""
    gc_cleanup_interval: int = 0
    mlx_cache_cleanup_interval: int = 512
