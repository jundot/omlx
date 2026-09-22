# SPDX-License-Identifier: Apache-2.0
"""Tests for SpecPrefill (attention-based sparse prefill)."""

import logging

import pytest

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


class TestSelectChunks:
    """Tests for select_chunks() — top-K% selection with a mandatory tail."""

    def test_basic_selection(self):
        from omlx.patches.specprefill import select_chunks

        # 4096 tokens (128 chunks), importance peaks in the first chunk.
        importance = mx.zeros(4096)
        importance = importance.at[:32].add(1.0)
        selected = select_chunks(importance, keep_pct=0.25, chunk_size=32)
        # Keeps 32 chunks (25% of 128): the tail window plus top-ranked ones.
        assert selected.shape[0] == 32 * 32
        indices = set(selected.tolist())
        # The high-importance first chunk wins a ranked slot.
        assert set(range(32)) <= indices

    def test_keep_100_percent(self):
        from omlx.patches.specprefill import select_chunks

        importance = mx.ones(64)
        selected = select_chunks(importance, keep_pct=1.0, chunk_size=32)
        assert selected.shape[0] == 64

    def test_sorted_output(self):
        from omlx.patches.specprefill import select_chunks

        # Make two early chunks important; 4096 tokens total.
        importance = mx.zeros(4096)
        importance = importance.at[32:64].add(2.0)
        importance = importance.at[96:128].add(1.0)
        selected = select_chunks(importance, keep_pct=0.25, chunk_size=32)
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

        # 4100 tokens with chunk_size=32 → 129 chunks (last has 4 tokens).
        # keep_n = 65 chunks; 64 full chunks + the 4-token final chunk.
        importance = mx.ones(4100)
        selected = select_chunks(importance, keep_pct=0.5, chunk_size=32)
        assert selected.shape[0] == 64 * 32 + 4

    def test_tail_floor_kept_when_unimportant(self):
        from omlx.patches.specprefill import select_chunks

        # Importance mass entirely in the front half; the tail scores zero.
        # Without the mandatory tail window the final chunks (chat-template
        # closers + generation prompt) would be dropped — the 2439 failure.
        importance = mx.zeros(8192)
        importance = importance.at[:2048].add(1.0)
        selected = select_chunks(importance, keep_pct=0.2, chunk_size=32)
        indices = set(selected.tolist())
        assert set(range(8192 - 512, 8192)) <= indices

    def test_tail_floor_within_budget_at_scale(self):
        from omlx.patches.specprefill import select_chunks

        # When the keep budget covers the tail, the tail comes out of the
        # budget and the selected-token count matches the plain top-K
        # formula: 8192 tokens → 256 chunks, keep 20% → 52 chunks → 1664.
        importance = (mx.arange(8192) % 97).astype(mx.float32)
        selected = select_chunks(importance, keep_pct=0.2, chunk_size=32)
        assert selected.shape[0] == 1664

    def test_tail_floor_dominates_small_input(self):
        from omlx.patches.specprefill import select_chunks

        # Just above the admission threshold with a small keep budget the
        # floor wins over keep_pct: exactly the 16 tail chunks (512 tokens,
        # indices 544..1055) are selected — keep_n (4) is below the tail.
        importance = mx.zeros(1056)
        selected = select_chunks(importance, keep_pct=0.1, chunk_size=32)
        indices = set(selected.tolist())
        assert indices == set(range(1056 - 512, 1056))

    def test_tail_floor_non_aligned_input(self):
        from omlx.patches.specprefill import select_chunks

        # Non-chunk-aligned M: the tail window is chunk-aligned, so it
        # covers the final partial chunk plus the 15 full chunks before it
        # (481 trailing tokens here — at least tail_tokens - chunk_size + 1).
        importance = mx.zeros(8193)
        importance = importance.at[:2048].add(1.0)
        selected = select_chunks(importance, keep_pct=0.2, chunk_size=32)
        indices = set(selected.tolist())
        assert set(range(241 * 32, 8193)) <= indices

    def test_tail_floor_disabled(self):
        from omlx.patches.specprefill import select_chunks

        # tail_tokens=0 restores pure top-K: an unimportant tail is dropped
        # and the budget is exactly keep_n chunks (32 of 128 → 1024 tokens).
        importance = mx.zeros(4096)
        importance = importance.at[:2048].add(1.0)
        selected = select_chunks(
            importance, keep_pct=0.25, chunk_size=32, tail_tokens=0
        )
        indices = set(selected.tolist())
        assert (4096 - 1) not in indices
        assert selected.shape[0] == 32 * 32


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


class TestManualRopeWithFreqs:
    """Tests for manual_rope_with_freqs() — RoPE from a precomputed freq table."""

    @staticmethod
    def _reference_partial_rotary(x, positions, dims, freqs):
        """Independent oracle for the model's real partial rotary, written as an
        explicit per-pair rotation in numpy so it shares NO code with the MLX
        implementation under test. Pairs dim ``j`` with dim ``j + dims // 2``
        over the full head and rotates ONLY the first ``len(freqs)`` pairs;
        every other lane passes through. The old contiguous ``2 * len(freqs)``
        pairing diverges from this by construction, which is what caught the
        bug (jundot, PR #2295)."""
        import numpy as np

        xn = np.array(x, dtype=np.float64)
        half = dims // 2
        n = int(freqs.shape[-1])
        inv = 1.0 / np.array(freqs, dtype=np.float64)
        pos = np.array(positions, dtype=np.float64)
        out = xn.copy()
        for j in range(n):
            ang = pos * inv[j]
            c, s = np.cos(ang), np.sin(ang)
            a = xn[..., j]
            b = xn[..., j + half]
            out[..., j] = a * c - b * s
            out[..., j + half] = a * s + b * c
        return out

    def test_partial_rotary_matches_reference_rope(self):
        # The corrected fix must match the model's real rope: pair dim i with
        # dim i + dims//2 and rotate only the first len(freqs) pairs. The earlier
        # 2*len(freqs) contiguous pairing diverged by ~5.9 abs and wrote
        # misrotated KV on every Gemma-4 global layer.
        import numpy as np

        from omlx.patches.specprefill import manual_rope_with_freqs

        B, n_heads, L, head_dim = 1, 2, 8, 256
        n_freqs = 64  # rotary sub-dim 128 < head_dim 256 (Gemma-4 style)
        freqs = mx.arange(1, n_freqs + 1, dtype=mx.float32) * 1000.0
        x = mx.random.normal((B, n_heads, L, head_dim))
        positions = mx.arange(L)

        got = np.array(
            manual_rope_with_freqs(x, positions, dims=head_dim, freqs=freqs),
            dtype=np.float64,
        )
        want = self._reference_partial_rotary(x, positions, head_dim, freqs)
        assert got.shape == tuple(x.shape)
        assert np.max(np.abs(got - want)) < 1e-4

    def test_partial_rotary_rotates_only_first_freqs_of_each_half(self):
        # Exactly the lanes the real rope touches change, and no others: for
        # head_dim 256 (half 128, n_freqs 64), dims [0:64] and [128:192] rotate;
        # [64:128] and [192:256] pass through (the zero-angle, unrotated pairs).
        from omlx.patches.specprefill import manual_rope_with_freqs

        B, n_heads, L, head_dim = 1, 2, 8, 256
        n_freqs, half = 64, 128
        freqs = mx.arange(1, n_freqs + 1, dtype=mx.float32) * 1000.0
        x = mx.random.normal((B, n_heads, L, head_dim))
        result = manual_rope_with_freqs(x, mx.arange(L), dims=head_dim, freqs=freqs)

        assert result.shape == x.shape
        # Rotated lanes: the first n_freqs of each half of the head.
        assert not mx.allclose(result[..., 0:n_freqs], x[..., 0:n_freqs])
        assert not mx.allclose(
            result[..., half : half + n_freqs], x[..., half : half + n_freqs]
        )
        # Untouched lanes: the remainder of each half.
        assert mx.allclose(result[..., n_freqs:half], x[..., n_freqs:half])
        assert mx.allclose(result[..., half + n_freqs :], x[..., half + n_freqs :])

    def test_full_rotary_matches_reference_rope(self):
        # Full rotary (len(freqs) == dims//2): no zero-padding path, every pair
        # rotates, and it still matches the independent oracle exactly -- proving
        # the fix is a no-op for full-rotary custom-_freqs models.
        import numpy as np

        from omlx.patches.specprefill import manual_rope_with_freqs

        B, n_heads, L, head_dim = 1, 2, 4, 64
        n_freqs = head_dim // 2
        freqs = mx.arange(1, n_freqs + 1, dtype=mx.float32) * 1000.0
        x = mx.random.normal((B, n_heads, L, head_dim))
        positions = mx.arange(L)

        got = np.array(
            manual_rope_with_freqs(x, positions, dims=head_dim, freqs=freqs),
            dtype=np.float64,
        )
        want = self._reference_partial_rotary(x, positions, head_dim, freqs)
        assert got.shape == tuple(x.shape)
        assert np.max(np.abs(got - want)) < 1e-4


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

    def test_cleanup_rope_unwraps_nested(self):
        from unittest.mock import MagicMock

        from omlx.patches.specprefill import (
            _OffsetAdjustedRoPE,
            cleanup_rope,
        )

        original_rope = MagicMock()
        nested = _OffsetAdjustedRoPE(
            _OffsetAdjustedRoPE(original_rope, adjustment=3), adjustment=5
        )

        model = MagicMock()
        layer = MagicMock()
        layer.self_attn = MagicMock()
        layer.self_attn.rope = nested
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


class TestRoPEReWrap:
    """Regression tests for #766 — re-wrapping a leftover _OffsetAdjustedRoPE.

    If a prior sparse_prefill left an _OffsetAdjustedRoPE installed (cleanup_rope
    not called, e.g. an aborted request or a multi-turn partial cache hit), the
    next sparse_prefill used to capture that wrapper as `original` and re-wrap it
    in _PositionMappedRoPE, whose __init__ dereferenced `original_rope.dims` and
    raised `'_OffsetAdjustedRoPE' object has no attribute 'dims'`.
    """

    class _GenuineRoPE:
        dims = 128
        base = 10000.0
        scale = 1.0

        def __call__(self, x, offset=0):
            return x

    def test_offset_adjusted_delegates_attrs(self):
        from omlx.patches.specprefill import _OffsetAdjustedRoPE

        wrapped = _OffsetAdjustedRoPE(self._GenuineRoPE(), adjustment=5)
        # unknown attrs delegate to the wrapped rope
        assert wrapped.dims == 128
        assert wrapped.base == 10000.0

    def test_unwrap_peels_to_genuine(self):
        from omlx.patches.specprefill import (
            _OffsetAdjustedRoPE,
            _PositionMappedRoPE,
            _unwrap_rope,
        )

        genuine = self._GenuineRoPE()
        positions = mx.arange(16)
        assert _unwrap_rope(genuine) is genuine
        assert _unwrap_rope(_OffsetAdjustedRoPE(genuine, 5)) is genuine
        nested = _PositionMappedRoPE(_OffsetAdjustedRoPE(genuine, 5), positions)
        assert _unwrap_rope(nested) is genuine

    def test_rewrap_leftover_does_not_crash(self):
        from omlx.patches.specprefill import (
            _OffsetAdjustedRoPE,
            _PositionMappedRoPE,
        )

        genuine = self._GenuineRoPE()
        leftover = _OffsetAdjustedRoPE(genuine, adjustment=5)
        # Previously raised AttributeError on original_rope.dims (#766)
        pm = _PositionMappedRoPE(leftover, mx.arange(16))
        assert pm._dims == 128


class TestTargetPrefillLeftoverCleanup:
    """run_specprefill_target_prefill restores RoPE at entry (#766 follow-up).

    A stale _OffsetAdjustedRoPE left by an aborted specprefill request must be
    removed before the system prompt prefill runs, otherwise system KV is
    written at offset-shifted positions.
    """

    def test_leftover_unwrapped_before_prefill(self, monkeypatch):
        from unittest.mock import MagicMock

        import omlx.patches.specprefill as patches
        import omlx.specprefill.target as target_mod
        from omlx.specprefill.planning import SpecPrefillTargetPlan

        class FakeRoPE:
            dims = 64
            base = 10000.0
            scale = 1.0

            def __call__(self, x, offset=0):
                return x

        genuine = FakeRoPE()
        layer = MagicMock()
        layer.self_attn = MagicMock()
        layer.self_attn.rope = patches._OffsetAdjustedRoPE(genuine, adjustment=7)
        model = MagicMock()
        model.layers = [layer]

        seen = {}

        def fake_sparse_prefill(m, tokens, selected, cache, **kwargs):
            seen["rope"] = layer.self_attn.rope
            return mx.zeros((1, 1))

        monkeypatch.setattr(patches, "sparse_prefill", fake_sparse_prefill)
        monkeypatch.setattr(target_mod, "make_prompt_cache", lambda m: [])

        plan = SpecPrefillTargetPlan(
            system_token_count=0,
            conversation_tokens=list(range(8)),
            conversation_token_count=8,
            generation_kickoff_index=7,
            remove_kickoff_index=False,
            sparse_selected_token_count=4,
            total_tracker_prefill_count=4,
            position_offset=0,
        )
        request = MagicMock()
        request.num_prompt_tokens = 8
        request.cached_tokens = 0

        target_mod.run_specprefill_target_prefill(
            target_model=model,
            request=request,
            plan=plan,
            all_tokens=list(range(8)),
            selected_indices=mx.array([0, 2, 4, 7]),
            prefill_step_size=4,
            stream=mx.cpu,
            check_abort=lambda n: None,
            report_system_progress=lambda p, t: None,
            report_sparse_progress=lambda p, t: None,
            sync_and_clear_cache=lambda: None,
            log=MagicMock(),
        )

        # Entry cleanup must restore the genuine rope before prefill runs
        assert seen["rope"] is genuine
        assert layer.self_attn.rope is genuine


class TestLogicalCacheOffset:
    """Tests for _logical_cache_offset() — the token position of a cache.

    Hybrid GDN models put a recurrent layer at index 0. That layer holds a
    fixed-size summary and exposes no ``offset``, so reading ``cache[0].offset``
    reports 0 for a fully restored cache: the prompt is then re-prefilled on
    top of the state that already holds it.
    """

    @staticmethod
    def _model(layer_kinds):
        """Model whose layers are attention ("a") or recurrent ("r")."""
        from types import SimpleNamespace

        layers = [
            SimpleNamespace(self_attn=object()) if kind == "a" else SimpleNamespace()
            for kind in layer_kinds
        ]
        return SimpleNamespace(layers=layers)

    @staticmethod
    def _kv(offset):
        from types import SimpleNamespace

        return SimpleNamespace(offset=offset, state=(1, 1))

    @staticmethod
    def _recurrent():
        """Stands in for ArraysCache: real state, no ``offset``."""
        from types import SimpleNamespace

        return SimpleNamespace(state=(1, 1), empty=lambda: False)

    def test_pure_kv_cache_is_unchanged(self):
        from omlx.patches.specprefill import _logical_cache_offset

        model = self._model("aaaa")
        cache = [self._kv(512) for _ in range(4)]
        assert _logical_cache_offset(model, cache) == 512

    def test_empty_pure_kv_cache_is_zero(self):
        from omlx.patches.specprefill import _logical_cache_offset

        model = self._model("aa")
        assert _logical_cache_offset(model, [self._kv(0), self._kv(0)]) == 0

    def test_hybrid_cache_reads_past_the_recurrent_layer_zero(self):
        from omlx.patches.specprefill import _logical_cache_offset

        # layer 0 is recurrent, as on Qwen3.5: cache[0] has no offset at all.
        model = self._model("rrra")
        cache = [
            self._recurrent(),
            self._recurrent(),
            self._recurrent(),
            self._kv(12288),
        ]
        assert not hasattr(cache[0], "offset")
        assert _logical_cache_offset(model, cache) == 12288

    def test_real_mlx_caches_are_read_correctly(self):
        """The same question, asked of the cache objects mlx-lm builds.

        The stand-ins elsewhere in this class cannot catch a mismatch between
        what a real cache exposes and what this module expects of it.
        """
        from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

        from omlx.patches.specprefill import (
            _cache_entry_is_bounded,
            _cache_entry_is_empty,
            _logical_cache_offset,
        )

        recurrent, kv = ArraysCache(1), KVCache()
        kv.update_and_fetch(
            mx.zeros((1, 2, 12, 8), dtype=mx.float32),
            mx.zeros((1, 2, 12, 8), dtype=mx.float32),
        )
        recurrent[0] = mx.ones((1, 2, 4), dtype=mx.float32)

        assert not hasattr(recurrent, "offset")
        assert not _cache_entry_is_empty(recurrent)
        assert not _cache_entry_is_bounded(kv)
        assert _cache_entry_is_bounded(RotatingKVCache(max_size=8))
        assert _logical_cache_offset(self._model("ra"), [recurrent, kv]) == 12

    def test_real_recurrent_only_cache_is_refused(self):
        """A real ArraysCache holding state must raise, not answer 0."""
        from mlx_lm.models.cache import ArraysCache

        from omlx.patches.specprefill import (
            IndeterminateCacheOffsetError,
            _logical_cache_offset,
        )

        held = ArraysCache(1)
        held[0] = mx.ones((1, 2, 4), dtype=mx.float32)
        with pytest.raises(IndeterminateCacheOffsetError):
            _logical_cache_offset(self._model("rr"), [held, held])

        assert _logical_cache_offset(self._model("rr"), [ArraysCache(1)]) == 0

    def test_disagreeing_layers_take_the_smallest(self, caplog):
        from omlx.patches.specprefill import _logical_cache_offset

        # One sequence cannot be at two positions; re-prefilling a token is
        # recoverable, skipping one is not.
        model = self._model("rara")
        cache = [self._recurrent(), self._kv(8192), self._recurrent(), self._kv(12288)]
        with caplog.at_level(logging.WARNING, logger="omlx.patches.specprefill"):
            assert _logical_cache_offset(model, cache) == 8192
        # A silent minimum would hide the inconsistency it is working around.
        assert "disagree on sequence position" in caplog.text
        assert "8192" in caplog.text and "12288" in caplog.text

    def test_agreeing_layers_log_nothing(self, caplog):
        from omlx.patches.specprefill import _logical_cache_offset

        model = self._model("rara")
        cache = [self._recurrent(), self._kv(8192), self._recurrent(), self._kv(8192)]
        with caplog.at_level(logging.WARNING, logger="omlx.patches.specprefill"):
            assert _logical_cache_offset(model, cache) == 8192
        assert caplog.text == ""

    @staticmethod
    def _rotating(offset, max_size=32):
        """Stands in for RotatingKVCache: offset counts tokens ever seen."""
        from types import SimpleNamespace

        return SimpleNamespace(offset=offset, max_size=max_size, state=(1, 1))

    def test_bounded_layer_does_not_answer_beside_an_unbounded_one(self, caplog):
        from omlx.patches.specprefill import _logical_cache_offset

        # A wrapped sliding window reports tokens ever processed, not tokens
        # held, so comparing it with a KVCache would mean comparing two
        # different quantities. prefix_cache refuses that comparison too.
        model = self._model("aa")
        cache = [self._kv(40), self._rotating(100)]
        with caplog.at_level(logging.WARNING, logger="omlx.patches.specprefill"):
            assert _logical_cache_offset(model, cache) == 40
        assert caplog.text == ""

    def test_bounded_only_model_still_gets_an_answer(self):
        from omlx.patches.specprefill import _logical_cache_offset

        # Nothing to compare against, so the offset is the position.
        model = self._model("aa")
        assert (
            _logical_cache_offset(model, [self._rotating(64), self._rotating(64)]) == 64
        )

    def test_state_bearing_cache_without_any_offset_is_refused(self):
        from omlx.patches.specprefill import (
            IndeterminateCacheOffsetError,
            _logical_cache_offset,
        )

        # Silently answering 0 here is what duplicates the prefix.
        model = self._model("rr")
        with pytest.raises(IndeterminateCacheOffsetError):
            _logical_cache_offset(model, [self._recurrent(), self._recurrent()])

    def test_cold_all_recurrent_cache_is_zero_not_refused(self):
        from types import SimpleNamespace

        from omlx.patches.specprefill import _logical_cache_offset

        # Nothing reports a position, but nothing holds state either: that is
        # a cold start, not the unreadable case.
        model = self._model("rr")
        cold = SimpleNamespace(state=None, empty=lambda: True)
        assert _logical_cache_offset(model, [cold, cold]) == 0

    def test_unreadable_emptiness_probe_is_not_treated_as_empty(self):
        from types import SimpleNamespace

        from omlx.patches.specprefill import (
            IndeterminateCacheOffsetError,
            _logical_cache_offset,
        )

        def _raises():
            raise RuntimeError("probe unavailable")

        # A probe that fails says nothing; assuming empty would answer 0.
        model = self._model("rr")
        broken = SimpleNamespace(state=(1, 1), empty=_raises)
        with pytest.raises(IndeterminateCacheOffsetError):
            _logical_cache_offset(model, [broken, broken])

    def test_cache_without_an_emptiness_probe_is_not_treated_as_empty(self):
        from types import SimpleNamespace

        from omlx.patches.specprefill import (
            IndeterminateCacheOffsetError,
            _logical_cache_offset,
        )

        # No empty(), so emptiness cannot be established. ArraysCache.state is
        # (list_of_arrays, padding, lengths), whose first element is a list
        # with no .size -- inspecting it would read a full cache as empty.
        model = self._model("rr")
        opaque = SimpleNamespace(state=([1, 2], None, None))
        with pytest.raises(IndeterminateCacheOffsetError):
            _logical_cache_offset(model, [opaque, opaque])

    def test_composite_cache_offset_is_found_in_a_sub_cache(self):
        from types import SimpleNamespace

        from omlx.patches.specprefill import _logical_cache_offset

        model = self._model("ra")
        composite = SimpleNamespace(caches=[self._kv(4096), self._recurrent()])
        assert _logical_cache_offset(model, [self._recurrent(), composite]) == 4096


class _HybridFixture:
    """Smallest model that reproduces the hybrid cache layout.

    Layer 0 is recurrent and exposes no ``offset``; a later layer is attention
    and does. Any code that asks ``cache[0]`` for the token position gets 0 no
    matter how much state the cache holds.
    """

    VOCAB = 16
    HEADS = 2
    HEAD_DIM = 8

    class RecurrentCache:
        """ArraysCache-shaped: fixed-size summary, deliberately no ``offset``."""

        def __init__(self):
            self.cache = [None, None]
            self.seen = 0

        @property
        def state(self):
            return tuple(self.cache)

        def empty(self):
            return self.seen == 0

    class KVCache:
        def __init__(self):
            self.keys = None
            self.values = None
            self.offset = 0

        def update_and_fetch(self, keys, values):
            self.keys = (
                keys if self.keys is None else mx.concatenate([self.keys, keys], axis=2)
            )
            self.values = (
                values
                if self.values is None
                else mx.concatenate([self.values, values], axis=2)
            )
            self.offset = self.keys.shape[2]
            return self.keys, self.values

        @property
        def state(self):
            return (self.keys, self.values)

        def empty(self):
            return self.offset == 0

    class Attn:
        def __init__(self, seed):
            self.n_heads = _HybridFixture.HEADS
            self.n_kv_heads = _HybridFixture.HEADS
            self.head_dim = _HybridFixture.HEAD_DIM
            g = mx.random.key(seed)
            self.wq = mx.random.normal((16, self.n_heads * self.head_dim), key=g)
            self.wk = mx.random.normal((16, self.n_heads * self.head_dim), key=g)

        def project(self, x):
            b, t, _ = x.shape
            q = (x @ self.wq).reshape(b, t, self.n_heads, self.head_dim)
            k = (x @ self.wk).reshape(b, t, self.n_heads, self.head_dim)
            return q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3)

        def __call__(self, x, mask=None, cache=None, **kwargs):
            q, k = self.project(x)
            if cache is not None:
                keys, values = cache.update_and_fetch(k, k)
            else:
                keys, values = k, k
            scores = mx.softmax(
                (q @ keys.transpose(0, 1, 3, 2)) * self.head_dim**-0.5, axis=-1
            )
            out = (scores @ values).transpose(0, 2, 1, 3)
            return out.reshape(x.shape[0], x.shape[1], -1)[..., :16]

    class AttnLayer:
        def __init__(self, seed):
            self.self_attn = _HybridFixture.Attn(seed)

        def __call__(self, x, cache=None):
            return x + self.self_attn(x, cache=cache)

    class RecurrentLayer:
        def __call__(self, x, cache=None):
            if cache is not None:
                cache.seen += x.shape[1]
                cache.cache = [mx.sum(x, axis=1), mx.sum(x, axis=1)]
            return x * 1.0

    def __init__(self, kinds="rara"):
        self.kinds = kinds
        self.layers = [
            (
                _HybridFixture.AttnLayer(i)
                if kind == "a"
                else _HybridFixture.RecurrentLayer()
            )
            for i, kind in enumerate(kinds)
        ]
        self.embed = mx.random.normal((self.VOCAB, 16), key=mx.random.key(99))
        self.prefilled_tokens = []

    def make_cache(self):
        return [
            _HybridFixture.KVCache() if kind == "a" else _HybridFixture.RecurrentCache()
            for kind in self.kinds
        ]

    def __call__(self, tokens, cache=None):
        self.prefilled_tokens.append(int(tokens.shape[-1]))
        h = self.embed[tokens]
        for idx, layer in enumerate(self.layers):
            h = layer(h, cache=None if cache is None else cache[idx])
        return h @ self.embed.T


def _query_extractor(attn, x, cache, **kwargs):
    q, _ = attn.project(x)
    return q


class TestScoreTokensHybridCacheReuse:
    """score_tokens() must honour a restored cache on a hybrid model."""

    @staticmethod
    def _tokens(n):
        return [(i * 7 + 3) % _HybridFixture.VOCAB for i in range(n)]

    def test_layer_zero_reports_no_offset(self):
        fixture = _HybridFixture()
        cache = fixture.make_cache()
        assert not hasattr(cache[0], "offset")
        assert hasattr(cache[1], "offset")

    def test_warm_cache_prefills_only_the_suffix(self):
        from omlx.patches.specprefill import _prefill_draft, score_tokens

        tokens = self._tokens(64)
        fixture = _HybridFixture()
        cache = fixture.make_cache()

        # Restore the first 40 tokens, as a prefix-cache hit would.
        _prefill_draft(fixture, tokens[:40], cache, step_size=8)
        fixture.prefilled_tokens.clear()

        score_tokens(
            fixture,
            tokens,
            n_lookahead=2,
            prefill_step_size=8,
            temp=0.0,
            query_extractor=_query_extractor,
            existing_cache=cache,
        )

        # 24 suffix tokens, not 64. Before the fix this was the whole prompt.
        # The trailing n_lookahead calls are decode steps, not prefill.
        prefill_tokens = sum(fixture.prefilled_tokens[:-2])
        assert prefill_tokens == 24

    def test_exact_hit_does_not_leave_a_duplicate_token(self):
        """A fully cached prompt must come back exactly n_prompt long.

        The exact-hit branch runs one token to obtain logits, so the cache is
        briefly a token longer than the prompt. Before the offset fix this
        branch was unreachable on a hybrid model, because cached_len was
        always 0; reading the position correctly makes it live, and the
        lookahead trim has to remove that token.
        """
        from omlx.patches.specprefill import _prefill_draft, score_tokens

        fixture = _HybridFixture()
        tokens = list(range(64))
        cache = fixture.make_cache()
        _prefill_draft(fixture, tokens, cache, step_size=8)
        assert cache[1].offset == len(tokens)

        score_tokens(
            fixture,
            tokens,
            existing_cache=cache,
            n_lookahead=2,
            query_extractor=_query_extractor,
            prefill_step_size=8,
        )

        assert cache[1].offset == len(tokens)
        assert cache[1].keys.shape[2] == len(tokens)

    def test_cache_ahead_of_the_prompt_is_clamped(self):
        """An offset past the prompt must not slice tokens negatively."""
        from omlx.patches.specprefill import (
            _logical_cache_offset,
            _prefill_draft,
            score_tokens,
        )

        fixture = _HybridFixture()
        tokens = list(range(64))
        cache = fixture.make_cache()
        _prefill_draft(fixture, list(range(80)), cache, step_size=8)
        assert _logical_cache_offset(fixture, cache) > len(tokens)

        score_tokens(
            fixture,
            tokens,
            existing_cache=cache,
            n_lookahead=2,
            query_extractor=_query_extractor,
            prefill_step_size=8,
        )

        assert cache[1].offset == len(tokens)

    def test_cold_and_warm_scoring_agree(self):
        from omlx.patches.specprefill import _prefill_draft, score_tokens

        tokens = self._tokens(64)

        cold_fixture = _HybridFixture()
        cold_importance, _ = score_tokens(
            cold_fixture,
            tokens,
            n_lookahead=2,
            prefill_step_size=8,
            temp=0.0,
            query_extractor=_query_extractor,
        )

        warm_fixture = _HybridFixture()
        warm_cache = warm_fixture.make_cache()
        _prefill_draft(warm_fixture, tokens[:40], warm_cache, step_size=8)
        warm_importance, _ = score_tokens(
            warm_fixture,
            tokens,
            n_lookahead=2,
            prefill_step_size=8,
            temp=0.0,
            query_extractor=_query_extractor,
            existing_cache=warm_cache,
        )

        assert cold_importance.shape == warm_importance.shape == (64,)
        assert mx.allclose(cold_importance, warm_importance, atol=1e-4).item()

    def test_cold_and_warm_select_the_same_tokens(self):
        from omlx.patches.specprefill import (
            _prefill_draft,
            score_tokens,
            select_chunks,
        )

        tokens = self._tokens(128)

        cold_fixture = _HybridFixture()
        cold_importance, _ = score_tokens(
            cold_fixture,
            tokens,
            n_lookahead=2,
            prefill_step_size=16,
            temp=0.0,
            query_extractor=_query_extractor,
        )

        warm_fixture = _HybridFixture()
        warm_cache = warm_fixture.make_cache()
        _prefill_draft(warm_fixture, tokens[:96], warm_cache, step_size=16)
        warm_importance, _ = score_tokens(
            warm_fixture,
            tokens,
            n_lookahead=2,
            prefill_step_size=16,
            temp=0.0,
            query_extractor=_query_extractor,
            existing_cache=warm_cache,
        )

        cold_sel = select_chunks(cold_importance, keep_pct=0.5, chunk_size=8)
        warm_sel = select_chunks(warm_importance, keep_pct=0.5, chunk_size=8)
        assert cold_sel.tolist() == warm_sel.tolist()

    def test_warm_cache_does_not_duplicate_the_prefix(self):
        from omlx.patches.specprefill import _prefill_draft, score_tokens

        tokens = self._tokens(64)
        fixture = _HybridFixture()
        cache = fixture.make_cache()
        _prefill_draft(fixture, tokens[:40], cache, step_size=8)

        score_tokens(
            fixture,
            tokens,
            n_lookahead=2,
            prefill_step_size=8,
            temp=0.0,
            query_extractor=_query_extractor,
            existing_cache=cache,
        )

        # Length alone does not discriminate: the trim clamps to n_prompt
        # either way. What a cached_len of 0 leaves behind is 104 keys trimmed
        # to the FIRST 64 -- the 40 restored ones plus 24 of the duplicates --
        # so compare content against a cache built the honest way.
        cold_fixture = _HybridFixture()
        cold_cache = cold_fixture.make_cache()
        _prefill_draft(cold_fixture, tokens, cold_cache, step_size=8)

        attn_cache = cache[1]
        assert attn_cache.offset == 64
        assert attn_cache.keys.shape[2] == 64
        assert mx.allclose(attn_cache.keys, cold_cache[1].keys, atol=1e-4).item()
