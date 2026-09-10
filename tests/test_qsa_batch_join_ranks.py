# SPDX-License-Identifier: Apache-2.0
"""BatchQSAKVCache joins: mixed text/MRoPE ranks and KV-vs-indexer offsets.

Re-verification of #3294 items 2 and 4 against current main, after #3219
normalized the *reconstruct* path and the singleton trim fix landed. The
*runtime* join path in BatchQSAKVCache still carries both defects:

Item 2 — ``extend`` picks ``sample_positions`` from whichever operand is
first non-None and derives ``position_axis`` from its rank. Joining a
text-only row (2-D ``[B, S]``) with an image row (3-D ``[3, B, S]``) either
raises on concatenate or joins on the wrong axis, depending on operand order.
The promotion rule already exists for the update path
(``_append_indexer_positions``) but never runs here.

Item 4 — ``merge`` passes the KV ``offset`` to ``_pad_index`` as
``index_offset``, which uses it as the indexer length. Any divergence between
KV length and indexer length is silently clamped into a mis-sized join. Batch
inputs also need to be expanded into singleton rows before ``BatchKVCache``
can merge their underlying KV state.

Small warmed caches with real KV and indexer tensors, no model load. Needs MLX.
"""

import pytest

mx = pytest.importorskip("mlx.core")

from omlx.patches.mlx_vlm_qwen4_exp_compat import (  # noqa: E402
    apply_mlx_vlm_qwen4_exp_compat_patch,
)

apply_mlx_vlm_qwen4_exp_compat_patch()

from mlx_vlm.models.qwen4_exp.language import (  # noqa: E402
    BatchQSAKVCache,
    QSAKVCache,
)

D = 4


def _singleton(length: int, *, mrope: bool, start: int = 0) -> QSAKVCache:
    values = mx.arange(start, start + 2 * length * D, dtype=mx.float32).reshape(
        1, 2, length, D
    )
    index_keys = mx.arange(start, start + length * D, dtype=mx.float32).reshape(
        1, length, D
    )
    positions = mx.arange(start, start + length, dtype=mx.int32)[None]
    if mrope:
        positions = mx.repeat(positions[None], 3, axis=0)

    cache = QSAKVCache()
    cache.state = (values, values + 1000, index_keys, positions)
    return cache


def _batch_text(length: int, start: int = 0) -> BatchQSAKVCache:
    """Warm batch cache whose indexer positions are 2-D text [B, S]."""
    return _singleton(length, mrope=False, start=start).to_batch([0])


def _batch_mrope(length: int, start: int = 0) -> BatchQSAKVCache:
    """Warm batch cache whose positions are 3-D MRoPE [C, B, S]."""
    return _singleton(length, mrope=True, start=start).to_batch([0])


class TestExtendMixedRanks:
    """#3294 item 2 — text row joined with MRoPE row."""

    def test_text_self_image_other(self):
        b = _batch_text(4)
        b.extend(_batch_mrope(4))  # must not raise
        assert b.index_position_ids.ndim == 3

    def test_image_self_text_other(self):
        b = _batch_mrope(4)
        b.extend(_batch_text(4))  # must not raise
        assert b.index_position_ids.ndim == 3

    def test_join_width_correct(self):
        """Two rows of 4 tokens => index_keys [2, 4, D]; positions must
        carry both rows, 8 columns total, at the widest rank."""
        b = _batch_text(4)
        b.extend(_batch_mrope(4))
        assert b.index_keys.shape == (2, 4, D)
        assert b.index_offset == 4
        # the widest rank in the join is MRoPE 3-D; the text row must be
        # promoted to it, not concatenated on the wrong axis
        assert b.index_position_ids.ndim == 3
        assert b.index_position_ids.shape == (3, 2, 4)

    def test_empty_left_batch_keeps_kv_and_indexer_row_counts_equal(self):
        """KV extension must not change the empty indexer's source row count."""
        batch = BatchQSAKVCache([0])

        batch.extend(_batch_text(4))

        mx.eval(batch.offset, batch.left_padding, batch.index_keys)
        assert batch.offset.tolist() == [0, 4]
        assert batch.left_padding.tolist() == [4, 0]
        assert batch.index_keys.shape == (2, 4, D)
        assert batch.index_position_ids.shape == (2, 4)
        assert [batch.extract(idx).offset for idx in range(2)] == [0, 4]

    def test_extend_rejects_divergent_indexer_width(self):
        """A partial indexer cannot safely describe the full KV columns."""
        left = _batch_text(8)
        left.index_keys = left.index_keys[:, :6]
        left.index_position_ids = left.index_position_ids[..., :6]
        left.index_offset = 6

        with pytest.raises(ValueError, match="requires aligned KV and indexer widths"):
            left.extend(_batch_text(4))


class TestMergeOffsetSemantics:
    """#3294 item 4 — merge confuses KV offset with indexer length."""

    def test_merge_singleton_offsets(self):
        """A single text cache merges at its length."""
        c = _singleton(4, mrope=False)
        out = BatchQSAKVCache.merge([c])
        mx.eval(out.offset, out.index_keys)
        assert out.offset.tolist() == [4]
        assert out.kv_cache.size() == 4
        assert int(out.index_offset) == 4
        assert out.index_keys.shape == (1, 4, D)

    def test_merge_warm_batch_inputs(self):
        """Existing batches are flattened to rows before their KV is merged."""
        originals = [
            _singleton(4, mrope=False, start=10),
            _singleton(2, mrope=False, start=30),
            _singleton(3, mrope=True, start=50),
        ]
        left = BatchQSAKVCache.merge(originals[:2])
        right = BatchQSAKVCache.merge(originals[2:])

        out = BatchQSAKVCache.merge([left, right])

        mx.eval(out.offset, out.left_padding, out.index_keys, out.index_position_ids)
        assert out.offset.tolist() == [4, 2, 3]
        assert out.left_padding.tolist() == [0, 2, 1]
        assert out.kv_cache.size() == 4
        assert out.index_offset == 4
        assert out.index_keys.shape == (3, 4, D)
        assert out.index_position_ids.shape == (3, 3, 4)
        for idx, original in enumerate(originals):
            extracted = out.extract(idx)
            assert extracted.offset == original.offset
            assert mx.array_equal(extracted.index_keys, original.index_keys).item()

    @pytest.mark.parametrize("batched", [False, True], ids=["singleton", "batch"])
    def test_merge_rejects_divergent_indexer_length(self, batched):
        """Missing indexer columns cannot be reconstructed from the KV cache."""
        cache = _singleton(8, mrope=False)
        if batched:
            cache = cache.to_batch([0])
            cache.index_keys = cache.index_keys[:, :6]
            cache.index_position_ids = cache.index_position_ids[..., :6]
            cache.index_offset = 6
        else:
            cache.index_keys = cache.index_keys[:, :6]
            cache.index_position_ids = cache.index_position_ids[..., :6]

        with pytest.raises(ValueError, match="requires aligned KV and indexer lengths"):
            BatchQSAKVCache.merge([cache])
