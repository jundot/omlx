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
