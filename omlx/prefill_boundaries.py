"""Shared block-boundary rules for external and resumable prefill."""


def clamp_prefill_chunk_to_boundary(
    chunk_tokens: int, *, cache_tokens: int, block_size: int
) -> int:
    """Cap a non-empty chunk at the next cache block boundary.

    ``cache_tokens`` includes any reused prefix; ``block_size`` must be
    positive. Callers handle empty input and apply memory guards afterward.
    """
    next_boundary = ((cache_tokens // block_size) + 1) * block_size
    return max(1, min(chunk_tokens, next_boundary - cache_tokens))


def spread_throttled_chunk(
    throttled_tokens: int, span_tokens: int, grid_tokens: int = 1
) -> int:
    """Split a throttled span that ends at a stop into even chunks.

    A span ends at a stop (block boundary, tail position or prompt end) and
    must be covered before the next chunk can start past it. Trimming it to
    ``throttled_tokens`` leaves a remainder chunk, often a few dozen tokens,
    that costs a full forward. At long context much of a chunk's transient
    does not scale with its width, so that short chunk also records a huge
    per-token sample. Covering the span in the same number of forwards with
    even chunks avoids both. Every chunk stays within ``throttled_tokens``
    and, when the throttle allows it, on the ``grid_tokens`` grid.
    """
    if throttled_tokens <= 0 or throttled_tokens >= span_tokens:
        return throttled_tokens
    grid = grid_tokens if throttled_tokens >= grid_tokens > 0 else 1
    limit_units = throttled_tokens // grid
    span_units = -(-span_tokens // grid)
    forwards = -(-span_units // limit_units)
    return -(-span_units // forwards) * grid


def should_emit_prefill_boundary(
    *, total_tokens: int, block_size: int, last_emitted_tokens: int
) -> bool:
    """Identify a new, non-empty block boundary with a positive block size.

    Snapshot emission and updates to the last-emitted count belong to callers.
    """
    return (
        total_tokens > 0
        and total_tokens % block_size == 0
        and last_emitted_tokens < total_tokens
    )
