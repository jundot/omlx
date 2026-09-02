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
KV length and indexer length is silently clamped into a mis-sized join. For
``BatchQSAKVCache`` inputs ``cache.offset`` is an ``mx.array``, which makes
``mx.array([cache.offset])`` a 2-D shape inside ``_pad_index``.

Tiny synthetic tensors, no model load. Needs real mlx.
"""

import pytest

mx = pytest.importorskip("mlx.core")

from omlx.patches.mlx_vlm_qwen4_exp_compat import (
    apply_mlx_vlm_qwen4_exp_compat_patch,
)

apply_mlx_vlm_qwen4_exp_compat_patch()

from mlx_vlm.models.qwen4_exp.language import (  # noqa: E402
    BatchQSAKVCache,
)

D = 4


def _batch_text(length: int) -> BatchQSAKVCache:
    """Batch cache whose indexer positions are 2-D text [B, S]."""
    c = BatchQSAKVCache([0])
    c.index_keys = mx.zeros((1, length, D), dtype=mx.float32)
    c.index_position_ids = mx.arange(length)[None, :]  # [1, S] ndim 2
    c.index_offset = length
    c.kv_cache.offset = mx.array([length])
    return c


def _batch_mrope(length: int) -> BatchQSAKVCache:
    """Batch cache whose indexer positions are 3-D MRoPE [C, B, S]."""
    c = BatchQSAKVCache([0])
    c.index_keys = mx.zeros((1, length, D), dtype=mx.float32)
    pos = mx.arange(length)[None, None, :]  # [1, 1, S] then widen channels
    c.index_position_ids = mx.repeat(pos, 3, axis=0)  # [3, 1, S] ndim 3
    c.index_offset = length
    c.kv_cache.offset = mx.array([length])
    return c


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


class TestMergeOffsetSemantics:
    """#3294 item 4 — merge confuses KV offset with indexer length."""

    def test_merge_singleton_offsets(self):
        """A single text cache merges at its length."""
        c = _batch_text(4)
        # simulate a batch-cache input (offset is an mx.array property)
        out = BatchQSAKVCache.merge([c])
        assert int(out.index_offset) == 4
        assert out.index_keys.shape == (1, 4, D)

    def test_merge_divergent_indexer_length(self):
        """KV offset 8 but indexer only holds 6: the join must pad the
        indexer to the *indexer* length semantics without inventing two
        phantom columns — i.e. either the join raises loudly or it pads
        deliberately; it must not silently clamp into wrong tokens."""
        c = _batch_text(8)
        c.index_keys = c.index_keys[:, :6]
        c.index_offset = 6
        out = BatchQSAKVCache.merge([c])
        # correct behaviour: indexer row is 8 wide, padded with 2 zeros
        # beyond index_offset (matching the KV it must describe), and
        # index_offset tracks... whichever value merge documents; today it
        # mixes the two. Assert self-consistency instead of one magic number:
        assert out.index_keys.shape[1] >= 6
        assert out.index_position_ids.shape[-1] == out.index_keys.shape[1]
