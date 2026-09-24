"""Conservative admission estimates for verified text prefill capabilities.

These estimates depend on explicit batch geometry, not observations collected
from single-request prefill. Current process occupancy already includes model
weights and the live group cache; only additional allocations are returned.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .geometry import (
    PrefillModelGeometry,
    nonnegative_int,
    positive_int,
    valid_geometry,
)


@dataclass(frozen=True)
class PrefillMemoryCost:
    """Additional occupancy beyond the caller's current physical footprint."""

    new_kv_bytes: int
    temporary_workspace_bytes: int
    cache_transition_bytes: int

    @property
    def additional_peak_bytes(self) -> int:
        return (
            self.new_kv_bytes
            + self.temporary_workspace_bytes
            + self.cache_transition_bytes
        )

    def peak_bytes(self, current_usage_bytes: int) -> int:
        """Add current physical occupancy once, including existing weights/KV."""
        if not nonnegative_int(current_usage_bytes):
            raise ValueError("current_usage_bytes must be a nonnegative integer")
        return current_usage_bytes + self.additional_peak_bytes


class PrefillTransition(StrEnum):
    HANDOFF = "handoff"
    COMPACT = "compact"
    DEMOTE = "demote"


@dataclass(frozen=True)
class PrefillMemoryContext:
    """One fresh engine observation; no model weights or live KV are added twice."""

    geometry: PrefillModelGeometry | None
    current_usage_bytes: int
    limit_bytes: int
    current_cache_bytes: int = 0
    decode_batch_size: int = 0
    decode_max_tokens: int = 0

    @property
    def headroom_bytes(self) -> int:
        return max(0, self.limit_bytes - self.current_usage_bytes)

    def estimate(
        self, batch_size: int, chunk_tokens: int, max_prompt_tokens: int
    ) -> PrefillMemoryCost | None:
        if self.limit_bytes <= 0:
            return None
        return estimate_batched_prefill_memory(
            self.geometry,
            batch_size=batch_size,
            query_tokens=chunk_tokens,
            max_prompt_tokens=max_prompt_tokens,
            current_cache_bytes=self.current_cache_bytes,
            decode_batch_size=self.decode_batch_size,
            decode_max_tokens=self.decode_max_tokens,
        )

    def estimate_transition(
        self,
        transition: PrefillTransition,
        *,
        physical_rows: int,
        owned_rows: int,
        cache_tokens: int,
    ) -> PrefillMemoryCost | None:
        if self.limit_bytes <= 0:
            return None
        return estimate_prefill_transition_memory(
            self.geometry,
            transition=transition,
            physical_rows=physical_rows,
            owned_rows=owned_rows,
            cache_tokens=cache_tokens,
            current_cache_bytes=self.current_cache_bytes,
            decode_batch_size=self.decode_batch_size,
            decode_max_tokens=self.decode_max_tokens,
        )


def estimate_batched_prefill_memory(
    geometry: PrefillModelGeometry | None,
    *,
    batch_size: int,
    query_tokens: int,
    max_prompt_tokens: int,
    current_cache_bytes: int = 0,
    decode_batch_size: int = 0,
    decode_max_tokens: int = 0,
    evaluate_logits: bool = False,
) -> PrefillMemoryCost | None:
    """Price a group through its longest prompt and subsequent decode handoff.

    ``max_prompt_tokens`` includes the final prompt token reserved for decode.
    Shorter rows are charged at this length even after they leave the group;
    admission therefore reserves future cache growth, not just the first chunk.
    ``current_cache_bytes`` is the group's allocated K/V storage
    already present in current physical occupancy. It excludes other groups
    and decode state. Existing KV is credited once; its incremental charge
    is never negative.
    Allocator slack covers growth from unaligned offsets, where rounding the
    final context length to ``cache_step`` alone would underestimate capacity.

    ``decode_batch_size`` and ``decode_max_tokens`` describe the existing decode
    batch that will receive completed rows. Its existing storage remains in the
    caller's baseline. The merge first pads its inputs, then concatenates them:
    both padded inputs and the final output can coexist. Charge a joined-cache
    allocation for each, since the old decode batch can be wider than the group.
    Retained group capacity also bounds the joined allocation, even if its
    logical length is shorter than its storage after a cache transformation.
    A group-sized temporary covers extracted rows and filtering while the old
    group remains alive. These conservative copy allowances also cover cache
    reallocation, and are retained even if handoff can avoid a particular copy.

    Temporary workspace includes fp32 attention scores and probabilities, masks,
    Q/K/V projections, dense MLP/residual activations, and verified routed/shared-expert workspace. It does not assume a
    fused kernel is available for batch > 1. Logits are charged only when the
    executor evaluates them; cache-only evaluation leaves that lazy branch idle.
    Kernel-private scratch and unrelated concurrent allocations still require
    the existing process memory ceiling and a check before every forward.
    """
    if not isinstance(geometry, PrefillModelGeometry) or not valid_geometry(geometry):
        return None

    if not all(
        positive_int(value) for value in (batch_size, query_tokens, max_prompt_tokens)
    ):
        return None
    if query_tokens > max_prompt_tokens:
        return None
    if not all(
        nonnegative_int(value)
        for value in (current_cache_bytes, decode_batch_size, decode_max_tokens)
    ):
        return None
    if (decode_batch_size == 0) != (decode_max_tokens == 0):
        return None
    if not isinstance(evaluate_logits, bool):
        return None

    kv_bytes_per_token = (
        2
        * geometry.num_layers
        * geometry.num_kv_heads
        * geometry.head_dim
        * geometry.cache_dtype_size
    )
    capacity_tokens = max_prompt_tokens + geometry.cache_step - 1
    new_kv_bytes = max(
        0, batch_size * capacity_tokens * kv_bytes_per_token - current_cache_bytes
    )
    group_cache_bytes = max(
        batch_size * capacity_tokens * kv_bytes_per_token,
        current_cache_bytes,
    )
    cache_bytes_per_position = batch_size * kv_bytes_per_token
    retained_capacity = (
        current_cache_bytes + cache_bytes_per_position - 1
    ) // cache_bytes_per_position
    joined_capacity = max(
        max(max_prompt_tokens, decode_max_tokens) + geometry.cache_step - 1,
        retained_capacity,
    )
    joined_cache_bytes = (batch_size + decode_batch_size) * (
        joined_capacity * kv_bytes_per_token
    )
    cache_transition_bytes = group_cache_bytes + 2 * joined_cache_bytes

    token_count = batch_size * query_tokens
    attention_scores = (
        2 * token_count * geometry.num_attention_heads * max_prompt_tokens * 4
    )
    attention_mask = token_count * max_prompt_tokens * 4
    projection_width = (
        geometry.num_attention_heads + 2 * geometry.num_kv_heads
    ) * geometry.head_dim
    activations = (
        token_count
        * (
            2 * projection_width
            + 4 * geometry.intermediate_size
            + 6 * geometry.hidden_size
        )
        * max(4, geometry.compute_dtype_size)
    )
    logits = (
        token_count * geometry.vocab_size * max(4, geometry.compute_dtype_size)
        if evaluate_logits
        else 0
    )
    temporary_workspace_bytes = attention_scores + attention_mask + activations + logits
    if geometry.experts is not None:
        experts = geometry.experts
        temporary_workspace_bytes += token_count * (
            4 * experts.num_experts * 4
            + experts.experts_per_token
            * (4 * experts.intermediate_size + 3 * geometry.hidden_size)
            * max(4, geometry.compute_dtype_size)
            + experts.experts_per_token * 4 * 8
            + 4 * experts.shared_intermediate_size * max(4, geometry.compute_dtype_size)
        )

    return PrefillMemoryCost(
        new_kv_bytes=new_kv_bytes,
        temporary_workspace_bytes=temporary_workspace_bytes,
        cache_transition_bytes=cache_transition_bytes,
    )


def estimate_prefill_transition_memory(
    geometry: PrefillModelGeometry | None,
    *,
    transition: PrefillTransition,
    physical_rows: int,
    owned_rows: int,
    cache_tokens: int,
    current_cache_bytes: int,
    decode_batch_size: int = 0,
    decode_max_tokens: int = 0,
) -> PrefillMemoryCost | None:
    """Bound a cache transition, not another forward or future prompt growth.

    Current occupancy includes the entire physical source, even rows already
    committed to decode. Compaction/demotion allocate at most one source-sized
    copy per remaining row. Handoff additionally reserves padded inputs and a
    joined decode cache at the current position plus generation kickoff/slack.
    Only still-owned rows can join decode; committed rows are already included
    in decode_batch_size and must not be charged as new arrivals again.

    All owned rows are reserved for handoff, including unfinished survivors.
    This bounds several queued insertions before decode materializes the join.
    Later forwards still obtain full-prompt admission; this estimate grants no
    permission for future KV growth or model execution. Metadata uses an int64
    bound for offsets, padding and selection indices.
    """
    if not isinstance(geometry, PrefillModelGeometry) or not valid_geometry(geometry):
        return None
    if not isinstance(transition, PrefillTransition):
        return None
    if not positive_int(physical_rows) or not positive_int(owned_rows):
        return None
    if owned_rows > physical_rows or not all(
        nonnegative_int(value)
        for value in (
            cache_tokens,
            current_cache_bytes,
            decode_batch_size,
            decode_max_tokens,
        )
    ):
        return None
    if (decode_batch_size == 0) != (decode_max_tokens == 0):
        return None
    if cache_tokens > 0 and current_cache_bytes == 0:
        return None

    row_bytes = (current_cache_bytes + physical_rows - 1) // physical_rows
    copy_bytes = owned_rows * row_bytes
    metadata_bytes = owned_rows * (2 * geometry.num_layers + 1) * 8
    if transition is PrefillTransition.HANDOFF:
        kv_bytes_per_token = (
            2
            * geometry.num_layers
            * geometry.num_kv_heads
            * geometry.head_dim
            * geometry.cache_dtype_size
        )
        retained_capacity = (row_bytes + kv_bytes_per_token - 1) // kv_bytes_per_token
        joined_capacity = max(
            max(cache_tokens + 1, decode_max_tokens) + geometry.cache_step - 1,
            retained_capacity,
        )
        joined_bytes = (owned_rows + decode_batch_size) * (
            joined_capacity * kv_bytes_per_token
        )
        copy_bytes += 2 * joined_bytes

    return PrefillMemoryCost(
        new_kv_bytes=0,
        temporary_workspace_bytes=metadata_bytes,
        cache_transition_bytes=copy_bytes,
    )
