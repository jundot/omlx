# SPDX-License-Identifier: Apache-2.0
"""
Tests for hybrid caching with RotatingKVCache support.

This module tests the last-block-only storage strategy for RotatingKVCache
layers and strict partial-prefix rejection for boundary-safe restore.
"""

import math
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None

from omlx.cache.hybrid_cache import (
    LayerCacheConfig,
    ModelCacheConfig,
    create_default_kvcache_config,
)
from omlx.cache.paged_cache import (
    BlockTable,
    PagedCacheManager,
    compute_block_hash,
)
from omlx.cache.prefix_cache import BlockAwarePrefixCache, BlockCacheEntry
from omlx.cache.type_handlers import CacheType


class MockModel:
    """Mock model for testing."""

    def __init__(self, num_layers: int = 4):
        self._num_layers = num_layers
        self.layers = [MagicMock() for _ in range(num_layers)]

    @property
    def args(self):
        mock_args = MagicMock()
        mock_args.num_hidden_layers = self._num_layers
        return mock_args


class TestModelCacheConfigWindowSize:
    """Tests for ModelCacheConfig window size functionality."""

    def test_get_max_window_size_default(self):
        """Test get_max_window_size returns 0 when not set."""
        config = create_default_kvcache_config(4)
        assert config.get_max_window_size() == 0

    def test_get_max_window_size_set(self):
        """Test get_max_window_size returns stored value."""
        config = create_default_kvcache_config(4)
        config._max_window_size = 1024
        assert config.get_max_window_size() == 1024

    def test_from_cache_list_extracts_window_size(self):
        """Test from_cache_list extracts window_size from RotatingKVCache."""
        # Create mock cache objects with spec to avoid spurious attributes
        mock_kv = MagicMock(spec=[])
        mock_kv.__class__ = type("KVCache", (), {})
        mock_kv.state = (MagicMock(), MagicMock())
        mock_kv.offset = 64
        mock_kv.keys = MagicMock()
        mock_kv.values = MagicMock()

        mock_rotating = MagicMock(spec=[])
        mock_rotating.__class__ = type("RotatingKVCache", (), {})
        mock_rotating.state = (MagicMock(), MagicMock())
        mock_rotating.max_size = 1024
        mock_rotating._idx = 0
        mock_rotating.offset = 0

        cache_list = [mock_kv, mock_rotating, mock_rotating, mock_kv]
        config = ModelCacheConfig.from_cache_list(cache_list)

        assert config.is_hybrid is True
        assert config.has_rotating_layers() is True
        assert config.get_max_window_size() == 1024

    def test_from_cache_list_no_rotating(self):
        """Test from_cache_list with KVCache-only model."""
        mock_kv = MagicMock(spec=[])
        mock_kv.__class__ = type("KVCache", (), {})
        mock_kv.state = (MagicMock(), MagicMock())
        mock_kv.offset = 64
        mock_kv.keys = MagicMock()
        mock_kv.values = MagicMock()

        cache_list = [mock_kv, mock_kv, mock_kv, mock_kv]
        config = ModelCacheConfig.from_cache_list(cache_list)

        assert config.is_hybrid is False
        assert config.has_rotating_layers() is False
        assert config.get_max_window_size() == 0

    def test_from_cache_list_max_window_size_selected(self):
        """Test from_cache_list picks the maximum window_size."""
        mock_rotating_small = MagicMock()
        mock_rotating_small.__class__ = type("RotatingKVCache", (), {})
        mock_rotating_small.__class__.__name__ = "RotatingKVCache"
        mock_rotating_small.max_size = 512
        mock_rotating_small._idx = 0

        mock_rotating_large = MagicMock()
        mock_rotating_large.__class__ = type("RotatingKVCache", (), {})
        mock_rotating_large.__class__.__name__ = "RotatingKVCache"
        mock_rotating_large.max_size = 2048
        mock_rotating_large._idx = 0

        cache_list = [mock_rotating_small, mock_rotating_large]
        config = ModelCacheConfig.from_cache_list(cache_list)

        assert config.get_max_window_size() == 2048


class TestApplyWindowPadding:
    """Tests for _apply_window_padding method."""

    @pytest.fixture
    def prefix_cache(self):
        """Create a BlockAwarePrefixCache with block_size=256."""
        paged_cache = PagedCacheManager(
            block_size=256,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )
        model = MockModel(num_layers=4)
        return BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
        )

    def test_no_config(self, prefix_cache):
        """Test with no model_cache_config returns all blocks."""
        result = prefix_cache._apply_window_padding(16, None)
        assert result == 16

    def test_kvcache_only_model(self, prefix_cache):
        """Test KVCache-only model returns all blocks."""
        config = create_default_kvcache_config(4)
        result = prefix_cache._apply_window_padding(16, config)
        assert result == 16

    def test_hybrid_model_with_rotating(self, prefix_cache):
        """Test hybrid model subtracts padding blocks."""
        # Gemma3-like: window_size=1024, block_size=256
        config = ModelCacheConfig.from_type_list(
            ["KVCache", "RotatingKVCache", "RotatingKVCache", "KVCache"]
        )
        config._max_window_size = 1024

        # 16 blocks matched -> restore 12 (16 - ceil(1024/256) = 16 - 4 = 12)
        result = prefix_cache._apply_window_padding(16, config)
        assert result == 12

    def test_padding_exceeds_matched(self, prefix_cache):
        """Test when padding blocks exceed matched blocks, returns 0."""
        config = ModelCacheConfig.from_type_list(
            ["KVCache", "RotatingKVCache"]
        )
        config._max_window_size = 1024

        # 3 blocks matched, need 4 for padding -> 0
        result = prefix_cache._apply_window_padding(3, config)
        assert result == 0

    def test_exact_padding(self, prefix_cache):
        """Test when matched blocks exactly equal padding blocks."""
        config = ModelCacheConfig.from_type_list(
            ["KVCache", "RotatingKVCache"]
        )
        config._max_window_size = 1024

        # 4 blocks matched, need 4 for padding -> 0
        result = prefix_cache._apply_window_padding(4, config)
        assert result == 0

    def test_window_size_not_multiple_of_block_size(self, prefix_cache):
        """Test with window_size not a multiple of block_size."""
        config = ModelCacheConfig.from_type_list(
            ["KVCache", "RotatingKVCache"]
        )
        config._max_window_size = 1000  # Not a multiple of 256

        # ceil(1000/256) = 4 padding blocks
        result = prefix_cache._apply_window_padding(16, config)
        assert result == 12

    def test_window_size_zero(self, prefix_cache):
        """Test with window_size=0 returns all blocks."""
        config = ModelCacheConfig.from_type_list(
            ["KVCache", "RotatingKVCache"]
        )
        config._max_window_size = 0

        result = prefix_cache._apply_window_padding(16, config)
        assert result == 16

    def test_small_block_size(self):
        """Test with small block_size (more padding blocks needed)."""
        paged_cache = PagedCacheManager(
            block_size=64,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )
        model = MockModel(num_layers=4)
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
        )

        config = ModelCacheConfig.from_type_list(
            ["KVCache", "RotatingKVCache"]
        )
        config._max_window_size = 1024

        # ceil(1024/64) = 16 padding blocks
        result = cache._apply_window_padding(20, config)
        assert result == 4


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestExtractBlockTensorSliceLastBlock:
    """Tests for _extract_block_tensor_slice with is_last_block parameter."""

    @pytest.fixture
    def prefix_cache(self):
        """Create a BlockAwarePrefixCache."""
        paged_cache = PagedCacheManager(
            block_size=4,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )
        model = MockModel(num_layers=4)
        return BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
        )

    def _make_kvcache_layer(self, seq_len=64):
        """Create a mock KVCache layer state."""
        return {
            'state': (mx.zeros((1, 8, seq_len, 64)), mx.zeros((1, 8, seq_len, 64))),
            'cache_type': 'KVCache',
        }

    def _make_rotating_layer(self, max_size=256):
        """Create a mock RotatingKVCache layer state."""
        return {
            'state': (mx.zeros((1, 8, max_size, 64)), mx.zeros((1, 8, max_size, 64))),
            'cache_type': 'RotatingKVCache',
        }

    def test_kvcache_only_slicing_unchanged(self, prefix_cache):
        """Test KVCache-only model slicing is unchanged regardless of is_last_block."""
        cache_data = [self._make_kvcache_layer(64) for _ in range(4)]

        # Non-last block
        result_non_last = prefix_cache._extract_block_tensor_slice(
            cache_data, 0, 4, is_last_block=False
        )
        assert result_non_last is not None
        assert len(result_non_last) == 4
        for keys, values in result_non_last:
            assert keys.shape == (1, 8, 4, 64)

        # Last block
        result_last = prefix_cache._extract_block_tensor_slice(
            cache_data, 0, 4, is_last_block=True
        )
        assert result_last is not None
        assert len(result_last) == 4
        for keys, values in result_last:
            assert keys.shape == (1, 8, 4, 64)

    def test_rotating_non_last_block_placeholder(self, prefix_cache):
        """Test RotatingKVCache non-last block stores placeholder."""
        cache_data = [
            self._make_kvcache_layer(64),
            self._make_rotating_layer(256),
        ]
        config = ModelCacheConfig.from_type_list(["KVCache", "RotatingKVCache"])

        result = prefix_cache._extract_block_tensor_slice(
            cache_data, 0, 4, model_cache_config=config, is_last_block=False
        )

        assert result is not None
        assert len(result) == 2

        # KVCache layer: normal slice
        keys0, values0 = result[0]
        assert keys0.shape == (1, 8, 4, 64)

        # RotatingKVCache layer: placeholder
        keys1, values1 = result[1]
        assert keys1.shape == (1,)
        assert values1.shape == (1,)

    def test_rotating_last_block_full_state(self, prefix_cache):
        """Test RotatingKVCache last block stores full state."""
        cache_data = [
            self._make_kvcache_layer(64),
            self._make_rotating_layer(256),
        ]
        config = ModelCacheConfig.from_type_list(["KVCache", "RotatingKVCache"])

        result = prefix_cache._extract_block_tensor_slice(
            cache_data, 0, 4, model_cache_config=config, is_last_block=True
        )

        assert result is not None
        assert len(result) == 2

        # KVCache layer: normal slice
        keys0, values0 = result[0]
        assert keys0.shape == (1, 8, 4, 64)

        # RotatingKVCache layer: full state
        keys1, values1 = result[1]
        assert keys1.shape == (1, 8, 256, 64)  # Full RotatingKVCache state
        assert values1.shape == (1, 8, 256, 64)

    def test_hybrid_model_multiple_blocks(self, prefix_cache):
        """Test storing multiple blocks for a hybrid model."""
        # Simulate Gemma3-like model: KVCache + RotatingKVCache
        cache_data = [
            self._make_kvcache_layer(16),  # KVCache with 16 tokens
            self._make_rotating_layer(256),  # RotatingKVCache with window=256
        ]
        config = ModelCacheConfig.from_type_list(["KVCache", "RotatingKVCache"])

        # Block 0 (non-last): KVCache sliced, RotatingKVCache placeholder
        block0 = prefix_cache._extract_block_tensor_slice(
            cache_data, 0, 4, model_cache_config=config, is_last_block=False
        )
        assert block0 is not None
        assert block0[0][0].shape == (1, 8, 4, 64)  # KVCache slice
        assert block0[1][0].shape == (1,)  # RotatingKVCache placeholder

        # Block 1 (non-last): KVCache sliced, RotatingKVCache placeholder
        block1 = prefix_cache._extract_block_tensor_slice(
            cache_data, 4, 8, model_cache_config=config, is_last_block=False
        )
        assert block1 is not None
        assert block1[0][0].shape == (1, 8, 4, 64)
        assert block1[1][0].shape == (1,)

        # Block 3 (last): KVCache sliced, RotatingKVCache full state
        block3 = prefix_cache._extract_block_tensor_slice(
            cache_data, 12, 16, model_cache_config=config, is_last_block=True
        )
        assert block3 is not None
        assert block3[0][0].shape == (1, 8, 4, 64)
        assert block3[1][0].shape == (1, 8, 256, 64)  # Full state


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestValidateBlockCacheDataPlaceholder:
    """Tests for _validate_block_cache_data with RotatingKVCache placeholders."""

    @pytest.fixture
    def prefix_cache(self):
        """Create a BlockAwarePrefixCache."""
        paged_cache = PagedCacheManager(
            block_size=4,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )
        model = MockModel(num_layers=4)
        return BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
        )

    def test_placeholder_passes_validation(self, prefix_cache):
        """Test placeholder (1D tensor) passes validation for RotatingKVCache."""
        # KVCache: normal 4D tensor, RotatingKVCache: placeholder
        cache_data = [
            (mx.zeros((1, 8, 4, 64)), mx.zeros((1, 8, 4, 64))),  # KVCache
            (mx.zeros((1,)), mx.zeros((1,))),  # RotatingKVCache placeholder
        ]
        layer_cache_types = ["KVCache", "RotatingKVCache"]

        result = prefix_cache._validate_block_cache_data(cache_data, layer_cache_types)
        assert result is True

    def test_full_rotating_state_passes_validation(self, prefix_cache):
        """Test full RotatingKVCache state passes validation."""
        cache_data = [
            (mx.zeros((1, 8, 4, 64)), mx.zeros((1, 8, 4, 64))),  # KVCache
            (mx.zeros((1, 8, 256, 64)), mx.zeros((1, 8, 256, 64))),  # RotatingKVCache full
        ]
        layer_cache_types = ["KVCache", "RotatingKVCache"]

        result = prefix_cache._validate_block_cache_data(cache_data, layer_cache_types)
        assert result is True


class TestDetectWindowPaddingFromBlocks:
    """Tests for _detect_window_padding_from_blocks method."""

    @pytest.fixture
    def paged_cache(self):
        """Create a PagedCacheManager."""
        return PagedCacheManager(
            block_size=256,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )

    @pytest.fixture
    def mock_ssd_cache(self):
        """Create a mock SSD cache manager."""
        mock = MagicMock()
        mock.save_block.return_value = True
        mock.load_block.return_value = None
        mock.load_block_with_metadata.return_value = (None, None)
        return mock

    @pytest.fixture
    def prefix_cache(self, paged_cache, mock_ssd_cache):
        """Create a BlockAwarePrefixCache with SSD."""
        model = MockModel(num_layers=4)
        return BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd_cache,
        )

    def test_no_block_ids(self, prefix_cache):
        """Test with empty block_ids returns None."""
        result = prefix_cache._detect_window_padding_from_blocks([])
        assert result is None

    def test_no_ssd_cache(self, paged_cache):
        """Test without SSD cache returns None."""
        model = MockModel(num_layers=4)
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=None,
        )
        result = cache._detect_window_padding_from_blocks([1, 2, 3])
        assert result is None

    def test_no_rotating_kvcache_in_metadata(self, prefix_cache, mock_ssd_cache, paged_cache):
        """Test with KVCache-only metadata returns None."""
        # Allocate a block
        block = paged_cache.allocate_block()
        block.block_hash = b"test_hash"

        mock_ssd_cache.load_block_with_metadata.return_value = (
            [(MagicMock(), MagicMock())],
            {
                'layer_cache_types': ['KVCache', 'KVCache'],
                'layer_meta_states': [(64,), (64,)],
            }
        )

        result = prefix_cache._detect_window_padding_from_blocks([block.block_id])
        assert result is None

    def test_rotating_kvcache_detected(self, prefix_cache, mock_ssd_cache, paged_cache):
        """Test RotatingKVCache is detected and config built correctly."""
        block = paged_cache.allocate_block()
        block.block_hash = b"test_hash"

        mock_ssd_cache.load_block_with_metadata.return_value = (
            [(MagicMock(), MagicMock()), (MagicMock(), MagicMock())],
            {
                'layer_cache_types': ['KVCache', 'RotatingKVCache'],
                'layer_meta_states': [
                    (64,),  # KVCache: (offset,)
                    (4, 1024, 500, 100),  # RotatingKVCache: (keep, max_size, offset, _idx)
                ],
            }
        )

        result = prefix_cache._detect_window_padding_from_blocks([block.block_id])

        assert result is not None
        assert result.has_rotating_layers() is True
        assert result.get_max_window_size() == 1024

    def test_max_window_size_from_multiple_rotating_layers(
        self, prefix_cache, mock_ssd_cache, paged_cache
    ):
        """Test max window_size is selected from multiple RotatingKVCache layers."""
        block = paged_cache.allocate_block()
        block.block_hash = b"test_hash"

        mock_ssd_cache.load_block_with_metadata.return_value = (
            [(MagicMock(), MagicMock())] * 3,
            {
                'layer_cache_types': ['RotatingKVCache', 'KVCache', 'RotatingKVCache'],
                'layer_meta_states': [
                    (4, 512, 500, 100),  # RotatingKVCache: max_size=512
                    (64,),  # KVCache
                    (4, 2048, 500, 100),  # RotatingKVCache: max_size=2048
                ],
            }
        )

        result = prefix_cache._detect_window_padding_from_blocks([block.block_id])

        assert result is not None
        assert result.get_max_window_size() == 2048


class TestFetchCachePrefixMatching:
    """Tests for fetch_cache prefix matching behavior."""

    @pytest.fixture
    def paged_cache(self):
        """Create a PagedCacheManager."""
        return PagedCacheManager(
            block_size=256,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )

    @pytest.fixture
    def mock_ssd_cache(self):
        """Create a mock SSD cache manager."""
        mock = MagicMock()
        mock.save_block.return_value = True
        mock.load_block.return_value = None
        mock.load_block_with_metadata.return_value = (None, None)
        return mock

    @pytest.fixture
    def prefix_cache(self, paged_cache, mock_ssd_cache):
        """Create a BlockAwarePrefixCache with SSD."""
        model = MockModel(num_layers=4)
        return BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd_cache,
        )

    def test_fetch_cache_miss_no_padding(self, prefix_cache):
        """Test cache miss doesn't trigger window padding."""
        tokens = list(range(1024))
        block_table, remaining = prefix_cache.fetch_cache("req-001", tokens)

        assert block_table is None
        assert remaining == tokens

    def test_fetch_cache_hit_no_rotating(self, prefix_cache, paged_cache, mock_ssd_cache):
        """Test cache hit without RotatingKVCache doesn't apply padding."""
        # Set up blocks in paged cache for shared prefix
        tokens = list(range(256))
        block = paged_cache.allocate_block()
        block.token_count = 256
        parent_hash = compute_block_hash(None, tokens, model_name="test-model")
        block.block_hash = parent_hash
        paged_cache.register_block_hash(block, tokens, None)

        # SSD returns KVCache-only metadata
        mock_ssd_cache.load_block_with_metadata.return_value = (
            [(MagicMock(), MagicMock())],
            {
                'layer_cache_types': ['KVCache'],
                'layer_meta_states': [(64,)],
            }
        )

        # Extend tokens so there are remaining
        extended_tokens = tokens + list(range(256, 512))
        block_table, remaining = prefix_cache.fetch_cache("req-001", extended_tokens)

        assert block_table is not None
        assert len(remaining) == 256

    def test_fetch_cache_hit_with_rotating_keeps_matched_blocks(
        self, prefix_cache, paged_cache, mock_ssd_cache
    ):
        """Rotating models no longer prune matched blocks via window padding."""
        tokens = list(range(256))
        block = paged_cache.allocate_block()
        block.token_count = 256
        parent_hash = compute_block_hash(None, tokens, model_name="test-model")
        block.block_hash = parent_hash
        paged_cache.register_block_hash(block, tokens, None)

        # SSD returns RotatingKVCache metadata with window_size=1024
        mock_ssd_cache.load_block_with_metadata.return_value = (
            [(MagicMock(), MagicMock()), (MagicMock(), MagicMock())],
            {
                'layer_cache_types': ['KVCache', 'RotatingKVCache'],
                'layer_meta_states': [
                    (64,),
                    (4, 1024, 500, 100),
                ],
            }
        )

        extended_tokens = tokens + list(range(256, 512))
        block_table, remaining = prefix_cache.fetch_cache("req-001", extended_tokens)

        assert block_table is not None
        assert block_table.num_tokens == 256
        assert remaining == extended_tokens[256:]


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestCreateEmptyRotatingCache:
    """Tests for _create_empty_rotating_cache method."""

    @pytest.fixture
    def prefix_cache(self):
        """Create a BlockAwarePrefixCache."""
        paged_cache = PagedCacheManager(
            block_size=256,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )
        model = MockModel(num_layers=4)
        return BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
        )

    def test_create_with_valid_meta_state_and_shape(self, prefix_cache):
        """Test creating empty RotatingKVCache with zero-length keys."""
        meta_state = (4, 1024, 500, 100)  # (keep, max_size, offset, _idx)
        kv_shape_ref = (8, 64)  # (kv_heads, head_dim)
        cache = prefix_cache._create_empty_rotating_cache(
            meta_state, kvcache_offset=768, kv_shape_ref=kv_shape_ref,
        )

        assert cache is not None
        assert cache.max_size == 1024
        assert cache.keep == 4
        assert cache.offset == 768
        # Zero-length keys so empty() returns False
        assert cache.keys is not None
        assert cache.keys.shape == (1, 8, 0, 64)
        assert cache.values is not None
        assert cache.values.shape == (1, 8, 0, 64)
        assert cache._idx == 0
        assert not cache.empty()

    def test_create_without_shape_ref(self, prefix_cache):
        """Test creating with no shape ref falls back to keys=None."""
        meta_state = (0, 512)
        cache = prefix_cache._create_empty_rotating_cache(meta_state)

        assert cache is not None
        assert cache.max_size == 512
        assert cache.keep == 0
        # Without shape ref, keys remain None
        assert cache.keys is None

    def test_create_with_none_meta_state(self, prefix_cache):
        """Test creating with None meta_state returns None."""
        cache = prefix_cache._create_empty_rotating_cache(None)
        assert cache is None

    def test_create_with_empty_meta_state(self, prefix_cache):
        """Test creating with empty tuple returns None."""
        cache = prefix_cache._create_empty_rotating_cache(())
        assert cache is None

    def test_create_with_short_meta_state(self, prefix_cache):
        """Test creating with meta_state shorter than 2 returns None."""
        cache = prefix_cache._create_empty_rotating_cache((64,))
        assert cache is None

    def test_empty_cache_reports_size_zero(self, prefix_cache):
        """Test that empty RotatingKVCache reports size=0, not min(offset, max_size).

        This is critical for BatchRotatingKVCache.merge(): standard
        RotatingKVCache.size() returns min(offset, max_size) which incorrectly
        claims data exists when keys are zero-length, causing merge() to
        create unmasked zero-filled buffers that dilute attention scores.
        """
        meta_state = (0, 128, 500, 64)  # (keep, max_size, offset, _idx)
        kv_shape_ref = (8, 64)
        cache = prefix_cache._create_empty_rotating_cache(
            meta_state, kvcache_offset=512, kv_shape_ref=kv_shape_ref,
        )

        assert cache is not None
        # Standard RotatingKVCache.size() would return min(512, 128) = 128
        # Our subclass must return 0 for zero-length keys
        assert cache.size() == 0
        # offset is still correct for RoPE alignment
        assert cache.offset == 512

    def test_prefill_ready_subclass_type(self, prefix_cache):
        """Test _create_empty_rotating_cache returns the correct subclass type."""
        from mlx_lm.models.cache import RotatingKVCache

        meta_state = (0, 128)
        kv_shape_ref = (8, 64)
        cache = prefix_cache._create_empty_rotating_cache(
            meta_state, kvcache_offset=256, kv_shape_ref=kv_shape_ref,
        )

        assert cache is not None
        # Must be a RotatingKVCache (for merge/empty compatibility)
        assert isinstance(cache, RotatingKVCache)
        # But with overridden size()
        assert cache.size() == 0

    def test_prefill_ready_size_normal_after_data(self, prefix_cache):
        """Test that size() returns normal value after keys have data."""
        meta_state = (0, 128)
        kv_shape_ref = (8, 64)
        cache = prefix_cache._create_empty_rotating_cache(
            meta_state, kvcache_offset=256, kv_shape_ref=kv_shape_ref,
        )

        assert cache.size() == 0

        # Simulate adding data (as happens during prefill)
        cache.keys = mx.zeros((1, 8, 64, 64))
        cache.values = mx.zeros((1, 8, 64, 64))
        cache.offset = 64
        # Now size() should return normal value: min(64, 128) = 64
        assert cache.size() == 64


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestPrefillReadyMergeBehavior:
    """Tests for correct merge behavior with _PrefillReadyRotatingKVCache.

    Verifies that BatchRotatingKVCache.merge() creates zero-length buffers
    (not zero-filled) when merging empty RotatingKVCaches from SSD restore.
    """

    @pytest.fixture
    def prefix_cache(self):
        paged_cache = PagedCacheManager(
            block_size=256,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )
        model = MockModel(num_layers=4)
        return BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
        )

    def test_merge_creates_zero_length_buffer(self, prefix_cache):
        """Test merge creates zero-length buffer, not zero-filled.

        Without the fix, merge creates a (1, H, 128, D) zero-filled buffer
        with left_padding=0, causing 128 unmasked zero positions in attention.
        With the fix, merge creates a (1, H, 0, D) buffer.
        """
        from mlx_lm.models.cache import BatchRotatingKVCache

        meta_state = (0, 128, 500, 64)
        kv_shape_ref = (8, 64)
        cache = prefix_cache._create_empty_rotating_cache(
            meta_state, kvcache_offset=512, kv_shape_ref=kv_shape_ref,
        )

        # Merge single cache (as happens with single request in batch)
        batch_cache = BatchRotatingKVCache.merge([cache])

        # Buffer should be zero-length, NOT 128-element zero-filled
        assert batch_cache.keys.shape[2] == 0
        assert batch_cache.values.shape[2] == 0
        # Offset should be preserved for correct RoPE positions
        assert batch_cache.offset.item() == 512
        # _offset should be 0 (no actual data processed)
        assert batch_cache._offset == 0

    def test_merge_old_behavior_would_create_zero_filled(self, prefix_cache):
        """Demonstrate what the old (buggy) behavior would produce.

        With standard RotatingKVCache.size() = min(512, 128) = 128,
        merge creates a 128-element zero-filled buffer.
        """
        from mlx_lm.models.cache import RotatingKVCache, BatchRotatingKVCache

        # Create using standard RotatingKVCache (NOT our subclass)
        old_cache = RotatingKVCache(max_size=128, keep=0)
        old_cache.offset = 512
        old_cache.keys = mx.zeros((1, 8, 0, 64))
        old_cache.values = mx.zeros((1, 8, 0, 64))
        old_cache._idx = 0

        # Standard size() reports 128 (the bug)
        assert old_cache.size() == 128

        # Merge creates 128-element zero-filled buffer (the bug effect)
        old_batch = BatchRotatingKVCache.merge([old_cache])
        assert old_batch.keys.shape[2] == 128  # 128 zero positions!
        # All positions are unmasked (left_padding = 0)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestFindKVShapeRef:
    """Tests for _find_kv_shape_ref helper."""

    @pytest.fixture
    def prefix_cache(self):
        paged_cache = PagedCacheManager(
            block_size=256, max_blocks=100,
            model_name="test-model", initial_blocks=100,
        )
        return BlockAwarePrefixCache(
            model=MockModel(num_layers=4),
            paged_cache_manager=paged_cache,
        )

    def test_finds_shape_from_kvcache_layer(self, prefix_cache):
        """Find kv_heads and head_dim from a KVCache layer."""
        block_data = [
            [(mx.zeros((1, 8, 256, 64)), mx.zeros((1, 8, 256, 64))),  # KVCache
             (mx.zeros((1,)), mx.zeros((1,)))],  # RotatingKVCache placeholder
        ]
        result = prefix_cache._find_kv_shape_ref(
            block_data, ['KVCache', 'RotatingKVCache']
        )
        assert result == (8, 64)

    def test_skips_non_kvcache_layers(self, prefix_cache):
        """Skip RotatingKVCache layers when finding shape ref."""
        block_data = [
            [(mx.zeros((1, 4, 256, 128)), mx.zeros((1, 4, 256, 128))),  # Rot placeholder
             (mx.zeros((1, 8, 256, 64)), mx.zeros((1, 8, 256, 64)))],   # KVCache
        ]
        result = prefix_cache._find_kv_shape_ref(
            block_data, ['RotatingKVCache', 'KVCache']
        )
        assert result == (8, 64)

    def test_returns_none_for_empty_data(self, prefix_cache):
        """Return None when no block data available."""
        assert prefix_cache._find_kv_shape_ref([], None) is None


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestReconstructCachePartialRestore:
    """Tests for reconstruct_cache with partial restore (placeholder handling)."""

    @pytest.fixture
    def paged_cache(self):
        """Create a PagedCacheManager."""
        return PagedCacheManager(
            block_size=256,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )

    @pytest.fixture
    def mock_ssd_cache(self):
        """Create a mock SSD cache manager."""
        mock = MagicMock()
        mock.save_block.return_value = True
        return mock

    def test_partial_restore_rejects_rotating_placeholder(
        self, paged_cache, mock_ssd_cache
    ):
        """Partial prefix match with Rotating placeholder must be rejected."""
        model = MockModel(num_layers=2)
        prefix_cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd_cache,
        )

        # Allocate 2 blocks (both non-last, so RotatingKVCache has placeholders)
        block1 = paged_cache.allocate_block()
        block1.block_hash = b"block1_hash"
        block1.token_count = 256

        block2 = paged_cache.allocate_block()
        block2.block_hash = b"block2_hash"
        block2.token_count = 256

        block_table = paged_cache.create_block_table("req-001")
        block_table.block_ids = [block1.block_id, block2.block_id]
        block_table.num_tokens = 512

        # Mock SSD: 2 layers - KVCache (4D slice) + RotatingKVCache (placeholder)
        kv_keys = mx.zeros((1, 8, 256, 64))
        kv_values = mx.zeros((1, 8, 256, 64))
        rot_placeholder = mx.zeros((1,))

        block_data = [
            (kv_keys, kv_values),       # Layer 0: KVCache slice
            (rot_placeholder, rot_placeholder),  # Layer 1: RotatingKVCache placeholder
        ]
        block_metadata = {
            'layer_cache_types': ['KVCache', 'RotatingKVCache'],
            'layer_meta_states': [
                (256,),              # KVCache: (offset,)
                (4, 1024, 500, 100), # RotatingKVCache: (keep, max_size, offset, _idx)
            ],
            'model_name': 'test-model',
            'num_layers': 2,
        }

        mock_ssd_cache.load_block_with_metadata.return_value = (block_data, block_metadata)

        result = prefix_cache.reconstruct_cache(block_table)

        assert result is None

    def test_full_restore_reconstructs_rotating_state(
        self, paged_cache, mock_ssd_cache
    ):
        """Exact block match should restore full RotatingKVCache state."""
        model = MockModel(num_layers=2)
        prefix_cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd_cache,
        )

        # Allocate 1 block (the last block with full RotatingKVCache data)
        block = paged_cache.allocate_block()
        block.block_hash = b"block_hash"
        block.token_count = 256

        block_table = paged_cache.create_block_table("req-001")
        block_table.block_ids = [block.block_id]
        block_table.num_tokens = 256

        # Mock SSD: last block has full RotatingKVCache data
        kv_keys = mx.zeros((1, 8, 256, 64))
        kv_values = mx.zeros((1, 8, 256, 64))
        rot_keys = mx.ones((1, 8, 1024, 64))  # Full RotatingKVCache state
        rot_values = mx.ones((1, 8, 1024, 64))

        block_data = [
            (kv_keys, kv_values),  # Layer 0: KVCache
            (rot_keys, rot_values),  # Layer 1: RotatingKVCache full state
        ]
        block_metadata = {
            'layer_cache_types': ['KVCache', 'RotatingKVCache'],
            'layer_meta_states': [
                (256,),
                (4, 1024, 500, 100),
            ],
            'model_name': 'test-model',
            'num_layers': 2,
        }

        mock_ssd_cache.load_block_with_metadata.return_value = (block_data, block_metadata)

        result = prefix_cache.reconstruct_cache(block_table)

        assert result is not None
        assert len(result) == 2

        # Layer 0 (KVCache): normal reconstruct
        kv_cache = result[0]
        assert kv_cache.keys.shape[2] == 256

        # Layer 1 (RotatingKVCache): reconstructed from stored state
        rot_cache = result[1]
        assert rot_cache.max_size == 1024
        assert rot_cache.keep == 4
        assert rot_cache.offset == 500
        assert rot_cache.keys is not None
        assert rot_cache.keys.shape == (1, 8, 1024, 64)
        assert rot_cache.values.shape == (1, 8, 1024, 64)
        assert rot_cache._idx == 100


class TestModelCacheConfigCacheList:
    """Tests for CacheList support in ModelCacheConfig."""

    def test_from_cache_list_with_cache_list(self):
        """Test from_cache_list detects CacheList."""
        from omlx.cache.type_handlers import CacheType
        from omlx.cache.type_registry import CacheTypeRegistry

        # CacheList mock: has .caches attribute
        mock_cache_list = MagicMock(spec=[])
        mock_sub_kv = MagicMock(spec=[])
        mock_sub_kv.__class__ = type("KVCache", (), {})
        mock_cache_list.caches = (mock_sub_kv,)
        mock_cache_list.__class__ = type("CacheList", (), {})

        mock_kv = MagicMock(spec=[])
        mock_kv.__class__ = type("KVCache", (), {})
        mock_kv.keys = MagicMock()
        mock_kv.values = MagicMock()

        config = ModelCacheConfig.from_cache_list(
            [mock_cache_list, mock_kv], model_name="test"
        )

        assert config.num_layers == 2
        assert config.is_hybrid is True
        assert config.layer_configs[0].cache_type == CacheType.CACHE_LIST
        assert config.layer_configs[1].cache_type == CacheType.KVCACHE

    def test_from_cache_list_cache_list_with_rotating_sub(self):
        """Test from_cache_list extracts window_size from CacheList sub-caches."""
        mock_rotating = MagicMock(spec=[])
        mock_rotating.__class__ = type("RotatingKVCache", (), {})
        mock_rotating.max_size = 256
        mock_rotating._idx = 0

        mock_cache_list = MagicMock(spec=[])
        mock_cache_list.__class__ = type("CacheList", (), {})
        mock_cache_list.caches = (mock_rotating,)

        config = ModelCacheConfig.from_cache_list([mock_cache_list])

        assert config.get_max_window_size() == 256

    def test_get_meta_states_cache_list(self):
        """Test get_meta_states extracts composite meta_state for CacheList."""
        from omlx.cache.type_handlers import CacheType

        # Use real thin classes so type(obj).__name__ returns the correct name
        # (MagicMock(spec=[]) with __class__ override does not affect type().__name__)
        KVCacheStub = type("KVCache", (), {
            "state": (MagicMock(), MagicMock()),
            "meta_state": (32,),
        })
        mock_sub = KVCacheStub()

        CacheListStub = type("CacheList", (), {
            "caches": (mock_sub,),
        })
        mock_cache_list = CacheListStub()

        config = ModelCacheConfig.from_cache_list([mock_cache_list])
        meta_states = config.get_meta_states([mock_cache_list])

        assert len(meta_states) == 1
        # CacheList meta_state is (class_names, sub_meta_states)
        assert isinstance(meta_states[0], tuple)
        assert len(meta_states[0]) == 2
        class_names, sub_metas = meta_states[0]
        assert class_names == ["KVCache"]
