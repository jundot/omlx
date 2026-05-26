# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx/models/base_model.py — pure-math helpers used by
omlx/models/xlm_roberta.py (the reranker model). Pin the masking and
normalization semantics so a refactor doesn't silently change
embedding output.
"""

from __future__ import annotations

import math

import mlx.core as mx

from omlx.models.base_model import (
    BaseModelArgs,
    BaseModelOutput,
    mean_pooling,
    normalize_embeddings,
)


class TestBaseModelDataclasses:
    def test_base_model_args_instantiable(self):
        """Empty marker dataclass — subclasses extend it."""
        BaseModelArgs()  # must not raise

    def test_output_required_field(self):
        out = BaseModelOutput(last_hidden_state=mx.zeros((1, 4, 8)))
        assert out.text_embeds is None
        assert out.pooler_output is None
        assert out.hidden_states is None

    def test_output_with_all_fields(self):
        hs = mx.zeros((1, 4, 8))
        emb = mx.ones((1, 8))
        pool = mx.ones((1, 8)) * 0.5
        all_hs = (hs, hs)
        out = BaseModelOutput(
            last_hidden_state=hs,
            text_embeds=emb,
            pooler_output=pool,
            hidden_states=all_hs,
        )
        assert out.text_embeds is emb
        assert out.pooler_output is pool
        assert out.hidden_states is all_hs


class TestMeanPooling:
    def test_uniform_mask_averages_all_positions(self):
        """When every position is unmasked, mean pooling = simple mean."""
        # batch=1, seq=4, hidden=3
        hs = mx.array([[[1.0, 2.0, 3.0],
                        [2.0, 4.0, 6.0],
                        [3.0, 6.0, 9.0],
                        [4.0, 8.0, 12.0]]])
        mask = mx.array([[1.0, 1.0, 1.0, 1.0]])
        pooled = mean_pooling(hs, mask)
        # Mean across seq axis: (1+2+3+4)/4=2.5, (2+4+6+8)/4=5, (3+6+9+12)/4=7.5
        assert pooled.shape == (1, 3)
        result = pooled.tolist()
        assert math.isclose(result[0][0], 2.5, rel_tol=1e-5)
        assert math.isclose(result[0][1], 5.0, rel_tol=1e-5)
        assert math.isclose(result[0][2], 7.5, rel_tol=1e-5)

    def test_partial_mask_excludes_padded_positions(self):
        """Padded positions (mask=0) must not contribute to the mean.
        This is the load-bearing invariant — pre-mask sums would let
        padding tokens corrupt the embedding for short inputs."""
        hs = mx.array([[[1.0, 1.0],
                        [2.0, 2.0],
                        [99.0, 99.0],   # padded — must NOT be counted
                        [99.0, 99.0]]])
        mask = mx.array([[1.0, 1.0, 0.0, 0.0]])
        pooled = mean_pooling(hs, mask)
        # Only first two positions count: mean(1,2)=1.5
        result = pooled.tolist()
        assert math.isclose(result[0][0], 1.5, rel_tol=1e-5)
        assert math.isclose(result[0][1], 1.5, rel_tol=1e-5)

    def test_all_zero_mask_does_not_divide_by_zero(self):
        """If the entire mask is zero (pathological but possible from
        upstream), the function must not produce NaN/Inf — the
        ``clip(..., a_min=1e-9)`` guard exists for this."""
        hs = mx.array([[[5.0, 5.0], [5.0, 5.0]]])
        mask = mx.array([[0.0, 0.0]])
        pooled = mean_pooling(hs, mask)
        # Both sum_embeddings AND sum_mask are 0 → 0 / 1e-9 = 0, not NaN
        result = pooled.tolist()
        assert all(math.isfinite(v) for v in result[0])

    def test_batch_dimension_preserved(self):
        """Batch dim should pass through — each row pooled
        independently."""
        hs = mx.array([
            [[1.0, 0.0], [3.0, 0.0]],
            [[2.0, 0.0], [4.0, 0.0]],
        ])
        mask = mx.array([[1.0, 1.0], [1.0, 1.0]])
        pooled = mean_pooling(hs, mask)
        assert pooled.shape == (2, 2)
        result = pooled.tolist()
        assert math.isclose(result[0][0], 2.0, rel_tol=1e-5)  # (1+3)/2
        assert math.isclose(result[1][0], 3.0, rel_tol=1e-5)  # (2+4)/2

    def test_works_with_float16_dtype(self):
        """Reranker inference often runs in fp16. Mask cast to the
        hidden states' dtype is the whole point of the
        ``mask_expanded.astype(hidden_states.dtype)`` line."""
        hs = mx.array([[[1.0, 1.0], [3.0, 3.0]]], dtype=mx.float16)
        mask = mx.array([[1.0, 1.0]])  # default float32
        pooled = mean_pooling(hs, mask)
        assert pooled.dtype == mx.float16


class TestNormalizeEmbeddings:
    def test_unit_norm_after_normalize(self):
        emb = mx.array([[3.0, 4.0]])  # |v| = 5
        out = normalize_embeddings(emb)
        # Each row should have L2 norm = 1
        norms = mx.linalg.norm(out, axis=-1).tolist()
        assert math.isclose(norms[0], 1.0, rel_tol=1e-5)

    def test_normalizes_along_last_axis_only(self):
        """The ``axis=-1`` is load-bearing — normalizing across the
        wrong axis would silently destroy similarity comparisons. Test
        with shape (batch=2, hidden=3)."""
        emb = mx.array([[1.0, 0.0, 0.0],
                        [3.0, 4.0, 0.0]])
        out = normalize_embeddings(emb)
        # Row 0 was already unit length
        # Row 1 should become (3/5, 4/5, 0)
        result = out.tolist()
        assert math.isclose(result[0][0], 1.0, rel_tol=1e-5)
        assert math.isclose(result[1][0], 0.6, rel_tol=1e-5)
        assert math.isclose(result[1][1], 0.8, rel_tol=1e-5)

    def test_preserves_shape(self):
        """Higher-rank inputs supported — (batch, seq, hidden) for
        per-token embeddings."""
        emb = mx.ones((2, 5, 8))
        out = normalize_embeddings(emb)
        assert out.shape == (2, 5, 8)

    def test_already_normalized_input_is_idempotent(self):
        """Normalizing twice gives the same result — basic mathematical
        invariant that catches accidental sign flips or scaling bugs."""
        emb = mx.array([[1.0, 2.0, 2.0]])
        once = normalize_embeddings(emb)
        twice = normalize_embeddings(once)
        # Compare as Python floats since mx.array doesn't have __eq__ that
        # produces a scalar bool
        a = once.tolist()
        b = twice.tolist()
        for x, y in zip(a[0], b[0]):
            assert math.isclose(x, y, abs_tol=1e-6)
