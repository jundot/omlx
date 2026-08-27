# SPDX-License-Identifier: Apache-2.0
"""Deterministic prompt/decode budgets for continuous batching.

The policy is deliberately pure: distributed ranks given the same mirrored
batch rows choose the same quantum without clocks, device probes, or memory
reads. Runtime memory guards may only shrink the returned prompt quantum.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _positive_env_int(name: str, default: int) -> int:
    """Read a positive integer without making module import a crash surface."""

    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


DEFAULT_MIXED_PREFILL_QUANTUM = _positive_env_int("OMLX_CONTENDED_PREFILL_CHUNK", 512)
MIN_MIXED_PREFILL_QUANTUM = _positive_env_int("OMLX_MIXED_PREFILL_MIN_QUANTUM", 128)
MIXED_PREFILL_GRID = 64


@dataclass(frozen=True)
class ContinuousBatchBudget:
    """Separate prompt-token/prompt-row and decode-row budgets for one turn.

    Zero prompt budgets mean unbounded idle processing, preserving the
    existing wide-chunk path when no decoder is active.
    """

    active_decode_rows: int
    decode_row_budget: int
    prompt_token_budget: int
    prompt_row_budget: int
    prefill_quantum: int

    @property
    def mixed(self) -> bool:
        return self.active_decode_rows > 0


def _decode_pressure_divisor(active_decode_rows: int) -> int:
    """Power-of-two pressure tiers: B1=1, B2/B3=2, B4+=4+."""

    rows = max(1, int(active_decode_rows))
    return 1 << (rows.bit_length() - 1)


def continuous_batch_budget(
    *,
    configured_prefill_tokens: int,
    active_decode_rows: int,
    decode_row_budget: int,
    max_prompt_tokens: int,
    mixed_prefill_quantum: int = DEFAULT_MIXED_PREFILL_QUANTUM,
    min_mixed_quantum: int = MIN_MIXED_PREFILL_QUANTUM,
    grid: int = MIXED_PREFILL_GRID,
    decode_rows_per_pressure_tier: int = 1,
) -> ContinuousBatchBudget:
    """Plan one exact scheduler turn without reducing idle throughput.

    Under contention, one prompt row receives a token quantum inversely
    proportional to the active decode-row pressure. The runtime's existing KV
    and transient-memory guards remain authoritative and may shrink it further.
    """

    configured = max(1, int(configured_prefill_tokens))
    decode_rows = max(0, int(active_decode_rows))
    decode_budget = max(1, int(decode_row_budget))
    max_prompt = max(1, int(max_prompt_tokens))
    if decode_rows == 0:
        return ContinuousBatchBudget(
            active_decode_rows=0,
            decode_row_budget=decode_budget,
            prompt_token_budget=0,
            prompt_row_budget=0,
            prefill_quantum=configured,
        )

    grid = max(1, int(grid))
    base = min(configured, max_prompt, max(1, int(mixed_prefill_quantum)))
    floor = min(base, max(grid, int(min_mixed_quantum)))
    rows_per_tier = max(1, int(decode_rows_per_pressure_tier))
    pressure_rows = (decode_rows + rows_per_tier - 1) // rows_per_tier
    quantum = max(floor, base // _decode_pressure_divisor(pressure_rows))
    quantum = max(grid, (quantum // grid) * grid)
    quantum = min(configured, max_prompt, quantum)
    return ContinuousBatchBudget(
        active_decode_rows=decode_rows,
        decode_row_budget=decode_budget,
        prompt_token_budget=quantum,
        prompt_row_budget=1,
        prefill_quantum=quantum,
    )
