"""Unit tests for the shared prefill block-boundary rules."""

import pytest

from omlx.prefill_boundaries import (
    clamp_prefill_chunk_to_boundary,
    should_emit_prefill_boundary,
)


@pytest.mark.parametrize(
    "chunk_tokens,cache_tokens,block_size,expected",
    [
        (24, 0, 8, 8),
        (24, 3, 8, 5),
        (4, 3, 8, 4),
        (24, 8, 8, 8),
        (1, 7, 8, 1),
        (1, 8, 8, 1),
        (257, 31, 256, 225),
        (64, 240, 256, 16),
        (1024, 513, 256, 255),
        (0, 0, 8, 1),
        (-1, 7, 8, 1),
    ],
)
def test_clamp_prefill_chunk_to_boundary(
    chunk_tokens, cache_tokens, block_size, expected
):
    assert (
        clamp_prefill_chunk_to_boundary(
            chunk_tokens, cache_tokens=cache_tokens, block_size=block_size
        )
        == expected
    )


@pytest.mark.parametrize("block_size", [1, 3, 4, 256])
def test_clamped_chunks_advance_without_crossing_a_boundary(block_size):
    for cache_tokens in range(3 * block_size):
        for chunk_tokens in [1, 2, block_size, block_size + 1, 2 * block_size]:
            clamped = clamp_prefill_chunk_to_boundary(
                chunk_tokens, cache_tokens=cache_tokens, block_size=block_size
            )
            next_boundary = (cache_tokens // block_size + 1) * block_size
            assert 1 <= clamped <= chunk_tokens
            assert cache_tokens + clamped <= next_boundary
            assert clamped == chunk_tokens or cache_tokens + clamped == next_boundary


@pytest.mark.parametrize(
    "total_tokens,block_size,last_emitted_tokens,expected",
    [
        (0, 4, -1, False),
        (3, 4, -1, False),
        (4, 4, -1, True),
        (8, 4, 4, True),
        (8, 4, 8, False),
        (8, 4, 12, False),
        (9, 4, 8, False),
        (1, 1, -1, True),
        (6, 3, 3, True),
        (512, 256, 256, True),
        (512, 256, 512, False),
    ],
)
def test_should_emit_prefill_boundary(
    total_tokens, block_size, last_emitted_tokens, expected
):
    assert (
        should_emit_prefill_boundary(
            total_tokens=total_tokens,
            block_size=block_size,
            last_emitted_tokens=last_emitted_tokens,
        )
        is expected
    )
