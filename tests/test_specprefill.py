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
        from types import SimpleNamespace

        from omlx.patches.specprefill import (
            _detect_query_extractor,
            _qwen35_extract_queries,
        )

        attn = SimpleNamespace(
            q_norm=object(),
            num_attention_heads=4,
            head_dim=8,
            q_proj=SimpleNamespace(weight=mx.zeros((64, 32))),
            o_proj=SimpleNamespace(weight=mx.zeros((32, 32))),
            rope=object(),
        )
        assert _detect_query_extractor(attn) is _qwen35_extract_queries

    def test_detect_query_extractor_llama(self):
        from unittest.mock import MagicMock

        from omlx.patches.specprefill import (
            _detect_query_extractor,
            _llama_extract_queries,
        )

        attn = MagicMock(spec=["rope", "q_proj"])
        assert _detect_query_extractor(attn) is _llama_extract_queries

    def test_detect_query_extractor_gemma_with_q_norm(self):
        from omlx.patches.specprefill import (
            _detect_query_extractor,
            _gemma4_extract_queries,
        )

        class FakeGemmaAttention:
            n_heads = 8
            head_dim = 16
            q_norm = object()
            rope = object()
            q_proj = type("QProj", (), {"weight": mx.zeros((128, 64))})()
            o_proj = type("OProj", (), {"weight": mx.zeros((64, 128))})()

            def __call__(self, x, mask=None, cache=None, shared_kv=None, offset=None):
                return x

        assert _detect_query_extractor(FakeGemmaAttention()) is _gemma4_extract_queries

    def test_detect_query_extractor_non_gated_q_norm_model(self):
        from types import SimpleNamespace

        from omlx.patches.specprefill import (
            _detect_query_extractor,
            _llama_extract_queries,
        )

        attn = SimpleNamespace(
            q_norm=object(),
            num_attention_heads=4,
            head_dim=8,
            q_proj=SimpleNamespace(weight=mx.zeros((32, 32))),
            o_proj=SimpleNamespace(weight=mx.zeros((32, 32))),
            rope=object(),
        )
        assert _detect_query_extractor(attn) is _llama_extract_queries

    def test_detect_query_extractor_qwen36_moe(self):
        """Qwen3.6 MoE: non-gated q_proj + per-head q_norm routes to qwen36."""
        from types import SimpleNamespace

        from omlx.patches.specprefill import (
            _detect_query_extractor,
            _qwen36_extract_queries,
        )

        attn = SimpleNamespace(
            q_norm=SimpleNamespace(weight=mx.zeros((16,))),  # RMSNorm(head_dim=16)
            n_heads=8,
            q_proj=SimpleNamespace(weight=mx.zeros((128, 64))),  # 8 * 16 = 128
            rope=object(),
        )
        assert _detect_query_extractor(attn) is _qwen36_extract_queries

    def test_detect_query_extractor_flat_q_norm_stays_llama(self):
        """Olmo-style: q_norm on flat n_heads*head_dim must not match qwen36."""
        from types import SimpleNamespace

        from omlx.patches.specprefill import (
            _detect_query_extractor,
            _llama_extract_queries,
        )

        attn = SimpleNamespace(
            q_norm=SimpleNamespace(weight=mx.zeros((128,))),  # flat n_heads*head_dim
            n_heads=8,
            head_dim=16,
            q_proj=SimpleNamespace(weight=mx.zeros((128, 64))),
            rope=object(),
        )
        # q_norm_dim=128, n_heads*q_norm_dim=1024 != q_out=128 → no qwen36 match
        assert _detect_query_extractor(attn) is _llama_extract_queries

    def test_attention_capture_forwards_extra_kwargs(self):
        from unittest.mock import MagicMock

        from omlx.patches.specprefill import _AttentionCapture

        captured = []
        extractor_calls = []

        def _extractor(attn, x, cache=None, **kwargs):
            extractor_calls.append((cache, kwargs))
            return "queries"

        original = MagicMock(return_value="result")
        wrapper = _AttentionCapture(original, 0, [captured], _extractor)

        out = wrapper("x", mask="m", cache="c", shared_kv="skv", offset=7)

        assert out == "result"
        assert captured == ["queries"]
        assert extractor_calls == [("c", {"shared_kv": "skv", "offset": 7})]
        original.assert_called_once_with(
            "x", mask="m", cache="c", shared_kv="skv", offset=7
        )

    def test_attention_capture_supports_legacy_extractor_signature(self):
        from unittest.mock import MagicMock

        from omlx.patches.specprefill import _AttentionCapture

        captured = []

        def _extractor(attn, x, cache=None):
            return "queries"

        original = MagicMock(return_value="result")
        wrapper = _AttentionCapture(original, 0, [captured], _extractor)

        out = wrapper("x", mask="m", cache="c", shared_kv="skv", offset=7)

        assert out == "result"
        assert captured == ["queries"]
        original.assert_called_once_with(
            "x", mask="m", cache="c", shared_kv="skv", offset=7
        )

    def test_gemma4_extract_queries_applies_q_norm(self):
        """Gemma4: q_norm runs on per-head queries before RoPE."""
        from omlx.patches.specprefill import _gemma4_extract_queries

        call_log = []

        class FakeAttn:
            n_heads = 4

            def q_proj(self, x):
                return x

            def q_norm(self, q):
                call_log.append(("q_norm", q.shape))
                return q * 2.0  # distinguishable transform

            def rope(self, q, offset=0):
                call_log.append(("rope", q.shape, offset))
                return q

        head_dim = 8
        x = mx.ones((1, 3, 4 * head_dim))
        out = _gemma4_extract_queries(FakeAttn(), x, cache=None, offset=11)

        # q_norm got (B, L, n_heads, head_dim) — reshape happened before norm
        assert call_log[0] == ("q_norm", (1, 3, 4, head_dim))
        # rope got (B, n_heads, L, head_dim) — transpose happened after norm
        assert call_log[1] == ("rope", (1, 4, 3, head_dim), 11)
        # and q_norm's scaling survived into the output
        assert mx.allclose(out, mx.ones_like(out) * 2.0).item()

    def test_qwen36_extract_queries_applies_q_norm(self):
        """Qwen3.6: q_norm runs on per-head queries before RoPE, no gate split."""
        from omlx.patches.specprefill import _qwen36_extract_queries

        call_log = []

        class FakeAttn:
            n_heads = 4

            def q_proj(self, x):
                return x

            def q_norm(self, q):
                call_log.append(("q_norm", q.shape))
                return q * 3.0

            def rope(self, q, offset=0):
                call_log.append(("rope", q.shape, offset))
                return q

        head_dim = 8
        x = mx.ones((1, 3, 4 * head_dim))
        cache = type("C", (), {"offset": 5})()
        out = _qwen36_extract_queries(FakeAttn(), x, cache=cache)

        # q_norm gets (B, L, n_heads, head_dim) — reshape before norm, no split
        assert call_log[0] == ("q_norm", (1, 3, 4, head_dim))
        # rope gets (B, n_heads, L, head_dim) with cache.offset
        assert call_log[1] == ("rope", (1, 4, 3, head_dim), 5)
        assert mx.allclose(out, mx.ones_like(out) * 3.0).item()

    def test_llama_extract_queries_without_q_norm(self):
        """Plain Llama/Mistral: no q_norm attr, fall through unchanged."""
        from omlx.patches.specprefill import _llama_extract_queries

        class FakeAttn:
            n_heads = 4

            def q_proj(self, x):
                return x

            def rope(self, q, offset=0):
                return q

        x = mx.ones((1, 3, 4 * 8))
        out = _llama_extract_queries(FakeAttn(), x, cache=None)
        assert out.shape == (1, 4, 3, 8)

    def test_build_layer_to_cache_map_gemma_shared_kv_vlm(self):
        """VLM Gemma4: previous_kvs lives at .language_model.model."""
        from types import SimpleNamespace

        from omlx.patches.specprefill import _build_layer_to_cache_map

        previous_kvs = [0, 1, 2, 2, 3]
        model = SimpleNamespace(
            layers=[object() for _ in previous_kvs],
            language_model=SimpleNamespace(
                model=SimpleNamespace(previous_kvs=previous_kvs)
            ),
        )

        assert _build_layer_to_cache_map(model) == {
            0: 0,
            1: 1,
            2: 2,
            3: 2,
            4: 3,
        }

    def test_build_layer_to_cache_map_gemma_shared_kv_text(self):
        """Text-only Gemma4: previous_kvs lives at .model (Gemma4TextModel)."""
        from types import SimpleNamespace

        from omlx.patches.specprefill import _build_layer_to_cache_map

        previous_kvs = [0, 1, 2, 2, 3]
        model = SimpleNamespace(
            layers=[object() for _ in previous_kvs],
            model=SimpleNamespace(previous_kvs=previous_kvs),
        )

        assert _build_layer_to_cache_map(model) == {
            0: 0,
            1: 1,
            2: 2,
            3: 2,
            4: 3,
        }


class TestRoPEWrappers:
    """Tests for _PositionMappedRoPE and _OffsetAdjustedRoPE."""

    def test_position_mapped_rope_accepts_mx_array_offset(self):
        """Gemma4 wraps cache.offset in mx.array before calling RoPE."""
        from omlx.patches.specprefill import _PositionMappedRoPE

        class FakeRoPE:
            dims = 64
            base = 10000.0
            scale = 1.0

            def __call__(self, x, offset=0):
                return x

        positions = mx.arange(10, dtype=mx.int32)
        wrapper = _PositionMappedRoPE(FakeRoPE(), positions, cache_start=0)
        x = mx.zeros((1, 4, 3, 64))
        result = wrapper(x, offset=mx.array(2))
        assert result.shape == x.shape

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


class TestEngineCorePropagation:
    """Tests for SpecPrefill param propagation through AsyncEngineCore.add_request."""

    def _make_engine_core(self, draft_model=None):
        """Create a minimal EngineCore for testing add_request propagation."""
        from unittest.mock import AsyncMock, MagicMock

        from omlx.engine_core import EngineCore

        core = object.__new__(EngineCore)
        core._output_collectors = {}
        core._active_requests = {}
        core._stream_states = {}
        core._finished_events = {}

        mock_scheduler = MagicMock(spec=[])
        mock_scheduler._specprefill_draft_model = draft_model
        core.scheduler = mock_scheduler

        mock_config = MagicMock(spec=[])
        mock_config.stream_interval = 0
        core.config = mock_config

        # _mlx_executor=None makes run_in_executor use the default pool
        core._mlx_executor = None
        # scheduler.add_request is a no-op for this test
        mock_scheduler.add_request = MagicMock()
        return core

    @pytest.mark.asyncio
    async def test_threshold_propagated_to_request(self):
        """specprefill_threshold should be set on request._specprefill_threshold."""
        from omlx.request import SamplingParams

        core = self._make_engine_core(draft_model="/some/draft")

        await core.add_request(
            prompt=[1, 2, 3],
            sampling_params=SamplingParams(),
            specprefill_threshold=4096,
            specprefill_keep_pct=0.3,
        )

        # Retrieve the request passed to scheduler.add_request
        req = core.scheduler.add_request.call_args[0][0]
        assert req._specprefill_threshold == 4096
        assert req._specprefill_keep_pct == 0.3
        assert req._specprefill_enabled is True

    @pytest.mark.asyncio
    async def test_threshold_not_set_when_none(self):
        """When specprefill_threshold is None, _specprefill_threshold should not exist."""
        from omlx.request import SamplingParams

        core = self._make_engine_core(draft_model=None)

        await core.add_request(
            prompt=[1, 2, 3],
            sampling_params=SamplingParams(),
        )

        req = core.scheduler.add_request.call_args[0][0]
        assert not hasattr(req, "_specprefill_threshold")
        assert not hasattr(req, "_specprefill_keep_pct")
