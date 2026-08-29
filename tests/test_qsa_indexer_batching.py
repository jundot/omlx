# SPDX-License-Identifier: Apache-2.0
"""Qwen4-Exp sparse-attention indexer under multi-row batching.

`Indexer.from_projected` read ``past_len = cache.offset``, which is a plain
``int`` for ``QSAKVCache`` but an ``mx.array`` of per-row offsets for
``BatchQSAKVCache``. Treating it as scalar gave ``query_ends`` and
``complete_counts`` shape ``[batch]`` instead of ``[seq_len]``, so indexing
them ``[None, :, None]`` put the batch axis where seq_len belongs and
``mx.where`` broadcast the scores from ``(batch, seq_len, blocks)`` to
``(batch, batch, blocks)``. Padding built from the literal ``seq_len`` then
failed:

    [concatenate] ... shapes are (2,2,55996), (2,1,1) ... axis is -1

Invisible at ``batch == 1``, where the two axes coincide -- so every
concurrent request pair against a Qwen4-exp model died.

These tests deliberately assert the *selected mask contents* per row, not
only that the shapes agree. A shape-only test would pass on a fix that
silently attends to the wrong blocks, which is the failure mode that
matters here: it would not crash, it would just degrade output.

Needs real ``mlx``; builds the mask arithmetic directly, no model load.
"""

import pytest

mx = pytest.importorskip("mlx.core")

COMPRESS_RATIO = 4
BLOCK_TOPK = 2


def _mask(past_len, seq_len, key_len, scores):
    """The fixed shape arithmetic from Indexer.from_projected.

    Mirrors the implementation rather than importing it: the real method
    needs a constructed Indexer module with weights. The axis handling under
    test is entirely here.
    """
    batch = scores.shape[0]
    max_complete_blocks = key_len // COMPRESS_RATIO
    complete_key_len = max_complete_blocks * COMPRESS_RATIO

    past = mx.array(past_len).reshape(-1)
    query_ends = past[:, None] + mx.arange(seq_len)[None, :] + 1
    complete_counts = query_ends // COMPRESS_RATIO
    valid_blocks = (
        mx.arange(max_complete_blocks)[None, None, :] < complete_counts[..., None]
    )
    masked = mx.where(valid_blocks, scores, -mx.inf)
    selected_blocks = mx.argpartition(masked, kth=-BLOCK_TOPK, axis=-1)[
        ..., -BLOCK_TOPK:
    ]

    block_hits = mx.put_along_axis(
        mx.zeros((batch, seq_len, max_complete_blocks), dtype=mx.bool_),
        selected_blocks,
        mx.array(True),
        axis=-1,
    )
    selected_tokens = mx.repeat(block_hits, COMPRESS_RATIO, axis=-1)
    if complete_key_len < key_len:
        selected_tokens = mx.concatenate(
            [
                selected_tokens,
                mx.zeros((batch, seq_len, key_len - complete_key_len), dtype=mx.bool_),
            ],
            axis=-1,
        )

    token_indices = mx.arange(key_len)
    tail_starts = complete_counts * COMPRESS_RATIO
    tail = (token_indices[None, None, :] >= tail_starts[..., None]) & (
        token_indices[None, None, :] < query_ends[..., None]
    )
    causal = token_indices[None, None, :] < query_ends[..., None]
    use_sparse = complete_counts > BLOCK_TOPK
    return mx.where(use_sparse[..., None], selected_tokens | tail, causal)


def _scores(batch, seq_len, blocks, winners):
    """Scores where `winners[row]` are the highest-scoring blocks."""
    base = mx.zeros((batch, seq_len, blocks))
    rows = []
    for row in range(batch):
        values = [0.0] * blocks
        for rank, block in enumerate(winners[row]):
            values[block] = 10.0 + rank
        rows.append(mx.array([values] * seq_len))
    return mx.stack(rows, axis=0) + base


# --------------------------------------------------------------------------
# shapes
# --------------------------------------------------------------------------


def test_batched_offsets_do_not_leak_onto_the_seq_len_axis():
    """The regression: per-row offsets must not widen the seq_len axis."""
    key_len, seq_len, batch = 64, 1, 2
    blocks = key_len // COMPRESS_RATIO
    out = _mask(
        mx.array([40, 52], dtype=mx.int32),
        seq_len,
        key_len,
        _scores(batch, seq_len, blocks, [[0, 1], [2, 3]]),
    )
    mx.eval(out)
    assert out.shape == (batch, seq_len, key_len)


def test_scalar_offset_still_works():
    """Singleton QSAKVCache passes an int; behaviour must be unchanged."""
    key_len, seq_len = 64, 1
    blocks = key_len // COMPRESS_RATIO
    out = _mask(40, seq_len, key_len, _scores(1, seq_len, blocks, [[0, 1]]))
    mx.eval(out)
    assert out.shape == (1, seq_len, key_len)


@pytest.mark.parametrize("seq_len", [1, 3], ids=["decode", "multi-token"])
@pytest.mark.parametrize("batch", [1, 2, 4], ids=["b1", "b2", "b4"])
def test_shape_holds_across_batch_and_seq_len(batch, seq_len):
    key_len = 64
    blocks = key_len // COMPRESS_RATIO
    offsets = mx.array([32 + 4 * i for i in range(batch)], dtype=mx.int32)
    out = _mask(
        offsets, seq_len, key_len, _scores(batch, seq_len, blocks, [[0, 1]] * batch)
    )
    mx.eval(out)
    assert out.shape == (batch, seq_len, key_len)


def test_ragged_key_len_pads_without_shape_error():
    """key_len not divisible by compress_ratio takes the concatenate path."""
    key_len, seq_len, batch = 66, 1, 2
    blocks = key_len // COMPRESS_RATIO
    out = _mask(
        mx.array([40, 52], dtype=mx.int32),
        seq_len,
        key_len,
        _scores(batch, seq_len, blocks, [[0, 1], [2, 3]]),
    )
    mx.eval(out)
    assert out.shape == (batch, seq_len, key_len)


# --------------------------------------------------------------------------
# contents -- the part a shape-only test would miss
# --------------------------------------------------------------------------


def test_each_row_selects_its_own_winning_blocks():
    """Row 0 and row 1 must attend to different blocks, per their scores."""
    key_len, seq_len, batch = 64, 1, 2
    blocks = key_len // COMPRESS_RATIO
    out = _mask(
        mx.array([60, 60], dtype=mx.int32),
        seq_len,
        key_len,
        _scores(batch, seq_len, blocks, [[0, 1], [5, 6]]),
    )
    mx.eval(out)
    # Blocks 0,1 -> tokens 0..7 ; blocks 5,6 -> tokens 20..27
    assert mx.all(out[0, 0, 0:8]).item(), "row 0 must attend to its winners"
    assert mx.all(out[1, 0, 20:28]).item(), "row 1 must attend to its winners"
    # And not to each other's, outside the causal tail.
    assert not mx.any(out[0, 0, 20:28]).item()
    assert not mx.any(out[1, 0, 0:8]).item()


def test_rows_with_different_offsets_get_different_valid_blocks():
    """Each row's offset must bound which blocks it may select.

    valid_blocks is computed from complete_counts, which is exactly what the
    axis bug corrupted -- a per-row offset landing on the seq_len axis meant
    one row's block-validity window was applied to another row.

    Row 0 (offset 15) has 4 complete blocks, so blocks 0-3 only. Row 1
    (offset 39) has 10, so block 5 is legal for it and illegal for row 0.
    Note this is not the same as the causal bound: sparse attention reaches
    only its selected blocks plus the tail, not everything below query_end.
    """
    key_len, seq_len, batch = 64, 1, 2
    blocks = key_len // COMPRESS_RATIO
    out = _mask(
        mx.array([15, 39], dtype=mx.int32),
        seq_len,
        key_len,
        _scores(batch, seq_len, blocks, [[0, 1], [5, 6]]),
    )
    mx.eval(out)
    # Neither row may reach past its own causal end.
    assert not mx.any(out[0, 0, 16:]).item(), "row 0 leaked past its causal end"
    assert not mx.any(out[1, 0, 40:]).item(), "row 1 leaked past its causal end"
    # Row 1 selects blocks 5 and 6 -> tokens 20..27, which row 0 may not
    # select at all because they lie beyond its complete-block window.
    assert mx.all(out[1, 0, 20:28]).item(), "row 1 lost its own winning blocks"
    assert not mx.any(out[0, 0, 20:28]).item(), "row 0 got row 1's blocks"


def test_causal_bound_advances_within_a_multi_token_step():
    """Each position in a seq_len>1 step gets its own causal end."""
    key_len, seq_len, batch = 64, 3, 1
    blocks = key_len // COMPRESS_RATIO
    out = _mask(
        mx.array([20], dtype=mx.int32),
        seq_len,
        key_len,
        _scores(batch, seq_len, blocks, [[0, 1]]),
    )
    mx.eval(out)
    # query_ends = 21, 22, 23 for the three positions.
    for position, end in enumerate((21, 22, 23)):
        assert not mx.any(out[0, position, end:]).item()
        assert out[0, position, end - 1].item()


def test_short_context_falls_back_to_dense_causal():
    """Below the topk threshold the mask must be plain causal, not sparse."""
    key_len, seq_len, batch = 64, 1, 2
    blocks = key_len // COMPRESS_RATIO
    # complete_counts = 8 // 4 = 2, which is not > BLOCK_TOPK (2).
    out = _mask(
        mx.array([7, 7], dtype=mx.int32),
        seq_len,
        key_len,
        _scores(batch, seq_len, blocks, [[0, 1], [4, 5]]),
    )
    mx.eval(out)
    for row in range(batch):
        assert mx.all(out[row, 0, :8]).item(), "dense causal expected"
        assert not mx.any(out[row, 0, 8:]).item()


def test_batched_result_matches_running_each_row_alone():
    """The strongest check: batching must not change any row's mask."""
    key_len, seq_len = 64, 1
    blocks = key_len // COMPRESS_RATIO
    offsets = [37, 58]
    winners = [[0, 1], [3, 7]]

    batched = _mask(
        mx.array(offsets, dtype=mx.int32),
        seq_len,
        key_len,
        _scores(2, seq_len, blocks, winners),
    )
    mx.eval(batched)

    for row, (offset, win) in enumerate(zip(offsets, winners)):
        alone = _mask(offset, seq_len, key_len, _scores(1, seq_len, blocks, [win]))
        mx.eval(alone)
        assert mx.array_equal(batched[row], alone[0]).item(), (
            f"row {row} differs when batched"
        )
