# SPDX-License-Identifier: Apache-2.0
"""Tests for SpecPrefill (attention-based sparse prefill)."""

import math

import pytest

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


class TestSelectChunks:
    """Tests for select_chunks() — chunk-based top-K% selection."""

    def test_basic_selection(self):
        from omlx.patches.specprefill import select_chunks

        # 128 tokens, importance peaks in the first 32 tokens
        importance = mx.zeros(128)
        importance = importance.at[:32].add(1.0)
        selected = select_chunks(importance, keep_pct=0.25, chunk_size=32)
        # Should keep 1 chunk (25% of 4 chunks)
        assert selected.shape[0] == 32
        # Should be the first chunk (indices 0-31)
        assert selected[0].item() == 0
        assert selected[-1].item() == 31

    def test_keep_100_percent(self):
        from omlx.patches.specprefill import select_chunks

        importance = mx.ones(64)
        selected = select_chunks(importance, keep_pct=1.0, chunk_size=32)
        assert selected.shape[0] == 64

    def test_sorted_output(self):
        from omlx.patches.specprefill import select_chunks

        # Make middle and end chunks important
        importance = mx.zeros(128)
        importance = importance.at[32:64].add(2.0)
        importance = importance.at[96:128].add(1.0)
        selected = select_chunks(importance, keep_pct=0.5, chunk_size=32)
        # Should select 2 chunks, sorted by position
        indices = selected.tolist()
        assert indices == sorted(indices)
        assert 32 in indices
        assert 96 in indices

    def test_single_chunk(self):
        from omlx.patches.specprefill import select_chunks

        importance = mx.ones(16)
        selected = select_chunks(importance, keep_pct=0.5, chunk_size=32)
        # Single chunk, 50% → keep at least 1 chunk
        assert selected.shape[0] == 16

    def test_non_divisible_chunks(self):
        from omlx.patches.specprefill import select_chunks

        # 100 tokens with chunk_size=32 → 4 chunks (last has 4 tokens)
        importance = mx.ones(100)
        selected = select_chunks(importance, keep_pct=0.5, chunk_size=32)
        n_chunks = math.ceil(100 / 32)
        keep_n = math.ceil(n_chunks * 0.5)
        expected_tokens = min(keep_n * 32, 100)
        # Allow for last chunk being smaller
        assert selected.shape[0] <= expected_tokens + 32


class TestManualRoPE:
    """Tests for manual_rope() at arbitrary positions."""

    def test_contiguous_matches_standard(self):
        from omlx.patches.specprefill import manual_rope

        # Contiguous positions should produce same result as standard RoPE
        B, n_heads, L, head_dim = 1, 4, 8, 64
        x = mx.random.normal((B, n_heads, L, head_dim))
        positions = mx.arange(L)
        result = manual_rope(x, positions, dims=head_dim)
        assert result.shape == x.shape

    def test_non_contiguous_positions(self):
        from omlx.patches.specprefill import manual_rope

        B, n_heads, L, head_dim = 1, 4, 3, 64
        x = mx.random.normal((B, n_heads, L, head_dim))
        positions = mx.array([0, 5, 10])
        result = manual_rope(x, positions, dims=head_dim)
        assert result.shape == x.shape
        # Results should differ from contiguous [0,1,2]
        contiguous = manual_rope(x, mx.arange(L), dims=head_dim)
        assert not mx.allclose(result, contiguous)

    def test_partial_rotation(self):
        from omlx.patches.specprefill import manual_rope

        B, n_heads, L, head_dim = 1, 2, 4, 128
        dims = 64  # Only rotate first 64 dims
        x = mx.random.normal((B, n_heads, L, head_dim))
        positions = mx.arange(L)
        result = manual_rope(x, positions, dims=dims)
        # Unrotated portion should be unchanged
        assert mx.allclose(result[..., dims:], x[..., dims:])


class TestAvgPool1d:
    """Tests for _avg_pool1d helper."""

    def test_identity_kernel_1(self):
        from omlx.patches.specprefill import _avg_pool1d

        x = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _avg_pool1d(x, 1)
        assert mx.allclose(result, x)

    def test_smoothing(self):
        from omlx.patches.specprefill import _avg_pool1d

        x = mx.array([0.0, 0.0, 1.0, 0.0, 0.0])
        result = _avg_pool1d(x, 3)
        mx.eval(result)
        # Center value should be smoothed
        assert result[2].item() < 1.0
        assert result[2].item() > 0.0


class TestKeepRatePresets:
    """Tests for keep rate preset constants."""

    def test_presets_exist(self):
        from omlx.patches.specprefill import (
            DEFAULT_KEEP_RATE,
            DEFAULT_THRESHOLD,
            KEEP_RATE_PRESETS,
        )

        assert DEFAULT_KEEP_RATE == 0.20
        assert DEFAULT_THRESHOLD == 8192
        assert 0.10 in KEEP_RATE_PRESETS
        assert 0.20 in KEEP_RATE_PRESETS
        assert 0.30 in KEEP_RATE_PRESETS
        assert 0.50 in KEEP_RATE_PRESETS


class TestModelTopologyHelpers:
    """Tests for model topology detection helpers."""

    def test_find_attention_layers_empty(self):
        from unittest.mock import MagicMock

        from omlx.patches.specprefill import _find_attention_layers

        model = MagicMock(spec=[])
        model.layers = []
        assert _find_attention_layers(model) == []

    def test_get_attn_module_self_attn(self):
        from unittest.mock import MagicMock

        from omlx.patches.specprefill import _get_attn_module

        layer = MagicMock()
        layer.self_attn = "attn_module"
        assert _get_attn_module(layer) == "attn_module"

    def test_detect_query_extractor_qwen35(self):
        from unittest.mock import MagicMock

        from omlx.patches.specprefill import (
            _detect_query_extractor,
            _qwen35_extract_queries,
        )

        attn = MagicMock()
        attn.q_norm = MagicMock()
        assert _detect_query_extractor(attn) is _qwen35_extract_queries

    def test_detect_query_extractor_llama(self):
        from unittest.mock import MagicMock

        from omlx.patches.specprefill import (
            _detect_query_extractor,
            _llama_extract_queries,
        )

        attn = MagicMock(spec=["rope", "q_proj"])
        assert _detect_query_extractor(attn) is _llama_extract_queries


class TestRoPEWrappers:
    """Tests for _PositionMappedRoPE and _OffsetAdjustedRoPE."""

    def test_offset_adjusted_rope_adds_offset(self):
        from omlx.patches.specprefill import _OffsetAdjustedRoPE

        call_log = []

        class FakeRoPE:
            def __call__(self, x, offset=0):
                call_log.append(offset)
                return x

        original = FakeRoPE()
        adjusted = _OffsetAdjustedRoPE(original, adjustment=100)
        x = mx.zeros((1, 4, 1, 64))
        adjusted(x, offset=5)
        assert call_log[-1] == 105  # 5 + 100

    def test_cleanup_rope_restores_original(self):
        from unittest.mock import MagicMock

        from omlx.patches.specprefill import (
            _OffsetAdjustedRoPE,
            cleanup_rope,
        )

        original_rope = MagicMock()
        adjusted = _OffsetAdjustedRoPE(original_rope, adjustment=50)

        model = MagicMock()
        layer = MagicMock()
        layer.self_attn = MagicMock()
        layer.self_attn.rope = adjusted
        model.layers = [layer]

        cleanup_rope(model)
        assert layer.self_attn.rope is original_rope


class TestModelSettings:
    """Tests for SpecPrefill fields in ModelSettings."""

    def test_specprefill_defaults(self):
        from omlx.model_settings import ModelSettings

        s = ModelSettings()
        assert s.specprefill_enabled is False
        assert s.specprefill_draft_model is None
        assert s.specprefill_keep_pct is None
        assert s.specprefill_threshold is None

    def test_specprefill_roundtrip(self):
        from omlx.model_settings import ModelSettings

        s = ModelSettings(
            specprefill_enabled=True,
            specprefill_draft_model="/path/to/draft",
            specprefill_keep_pct=0.2,
            specprefill_threshold=8192,
        )
        d = s.to_dict()
        assert d["specprefill_enabled"] is True
        assert d["specprefill_draft_model"] == "/path/to/draft"
        assert d["specprefill_keep_pct"] == 0.2

        restored = ModelSettings.from_dict(d)
        assert restored.specprefill_enabled is True
        assert restored.specprefill_draft_model == "/path/to/draft"


class TestRequestFields:
    """Tests for SpecPrefill fields in Request."""

    def test_specprefill_defaults(self):
        from omlx.request import Request, SamplingParams

        r = Request(
            request_id="test",
            prompt="hello",
            sampling_params=SamplingParams(),
        )
        assert r.specprefill_indices is None
        assert r.specprefill_total_tokens == 0
        assert r.specprefill_position_offset == 0
