# SPDX-License-Identifier: Apache-2.0
"""
Block-Aware Prefix Cache for oMLX.

Provides prefix caching using PagedCacheManager for block-based storage
with SSD persistence. oMLX only supports paged SSD-based caching.
"""

import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from .interface import CacheManager
from .paged_ssd_cache import PagedSSDCacheManager
from .paged_cache import BlockTable, CacheBlock, PagedCacheManager, compute_block_hash
from .stats import BaseCacheStats, PrefixCacheStats
from .type_handlers import CacheType, CacheTypeHandler
from .type_registry import CacheTypeRegistry
from .hybrid_cache import ModelCacheConfig, LayerCacheConfig

logger = logging.getLogger(__name__)


_PrefillReadyRotatingKVCache = None  # Lazily initialized RotatingKVCache subclass


@dataclass
class BlockCacheEntry:
    """Entry mapping a token sequence to cache blocks."""

    block_table: BlockTable
    last_access: float


class BlockAwarePrefixCache(CacheManager):
    """
    Prefix cache that uses PagedCacheManager for block-based storage.

    Features:
    - Block-level prefix sharing (256 tokens per block)
    - paged SSD-only storage via PagedSSDCacheManager
    - Hash-based deduplication across requests
    - Reference counting for memory efficiency

    Implements the CacheManager ABC interface for consistency with other
    cache implementations in oMLX.

    In paged SSD-only mode:
    - All KV cache data is stored on paged SSD via PagedSSDCacheManager
    - PagedCacheManager only stores metadata (no cache_data in blocks)
    - Cache data is loaded from paged SSD when needed for inference

    Example:
        cold_manager = PagedSSDCacheManager(cache_dir=Path("/tmp/cache"), ...)
        paged_manager = PagedCacheManager(block_size=256, max_blocks=1000)
        cache = BlockAwarePrefixCache(model, paged_manager, cold_manager)

        # Check for cached prefix
        block_table, remaining_tokens = cache.fetch_cache(request_id, tokens)

        # After generation, store cache
        cache.store_cache(request_id, tokens, kv_cache_data)

        # Clean up when request completes
        cache.release_cache(request_id)
    """

    def __init__(
        self,
        model: Any,
        paged_cache_manager: PagedCacheManager,
        paged_ssd_cache_manager: Optional[PagedSSDCacheManager] = None,
    ):
        """
        Initialize block-aware prefix cache.

        Args:
            model: The MLX model (used for identification)
            paged_cache_manager: The PagedCacheManager instance for block management
            paged_ssd_cache_manager: The PagedSSDCacheManager for SSD storage (required for paged SSD-only mode)
        """
        self.model = model
        self.model_key = id(model)
        self.paged_cache = paged_cache_manager
        self.paged_ssd_cache = paged_ssd_cache_manager
        self.block_size = paged_cache_manager.block_size

        # Expected number of layers for cache validation
        self.expected_num_layers = self._get_model_num_layers(model)

        # Hash table for quick prefix lookup
        # Maps chain-hash(prefix) -> (prefix_len, block_ids, num_blocks)
        self._prefix_index: Dict[bytes, Tuple[int, List[int], int]] = {}

        # Request to block table mapping
        self._request_tables: Dict[str, BlockCacheEntry] = {}

        # Callback for restoring cold blocks (deprecated in paged SSD-only mode)
        # Kept for API compatibility
        self._cold_restore_callback: Optional[Callable[[int, bytes], bool]] = None

        # Statistics
        self._hits = 0
        self._misses = 0
        self._tokens_saved = 0

    def _get_model_num_layers(self, model: Any) -> int:
        """
        Get the number of layers in the model.

        Args:
            model: The MLX model

        Returns:
            Number of layers, or 0 if cannot be determined
        """
        # Try various ways to get num_layers
        if hasattr(model, 'layers'):
            return len(model.layers)
        if hasattr(model, 'args') and hasattr(model.args, 'num_hidden_layers'):
            return model.args.num_hidden_layers
        if hasattr(model, 'config') and hasattr(model.config, 'num_hidden_layers'):
            return model.config.num_hidden_layers
        # Cannot determine, return 0 to skip validation
        logger.debug("Cannot determine model num_layers, cache layer validation disabled")
        return 0

    def set_paged_ssd_cache_manager(self, paged_ssd_cache_manager: Optional[PagedSSDCacheManager]) -> None:
        """
        Set the PagedSSDCacheManager for SSD storage.

        This allows setting the SSD cache after initialization,
        which is useful when the scheduler creates it later.

        Args:
            paged_ssd_cache_manager: The PagedSSDCacheManager instance.
        """
        self.paged_ssd_cache = paged_ssd_cache_manager
        if paged_ssd_cache_manager is not None:
            logger.info("PagedSSDCacheManager connected to BlockAwarePrefixCache")

    def _detect_window_padding_from_blocks(
        self,
        block_ids: List[int],
    ) -> Optional[ModelCacheConfig]:
        """Detect if blocks contain RotatingKVCache data and build config for padding.

        Checks block metadata from SSD to determine if the cached model uses
        RotatingKVCache layers. If so, builds a ModelCacheConfig with window_size
        for use with _apply_window_padding().

        Args:
            block_ids: List of block IDs to check

        Returns:
            ModelCacheConfig if RotatingKVCache detected, None otherwise
        """
        if not block_ids or self.paged_ssd_cache is None:
            return None

        first_block = self.paged_cache.allocated_blocks.get(block_ids[0])
        if not first_block or not first_block.block_hash:
            return None

        _, metadata = self.paged_ssd_cache.load_block_with_metadata(first_block.block_hash)
        if not metadata:
            return None

        layer_cache_types = metadata.get('layer_cache_types')
        # Note: CacheList layers containing RotatingKVCache sub-caches do NOT need
        # window padding. CacheList uses last-block-only storage with reject-on-partial
        # strategy, so the sliding window state is either fully restored (exact match)
        # or the entire cache is rejected (partial match).
        if not layer_cache_types or 'RotatingKVCache' not in layer_cache_types:
            return None

        model_cache_config = ModelCacheConfig.from_type_list(
            layer_cache_types, model_name=""
        )

        # Extract window_size from layer meta_states
        layer_meta_states = metadata.get('layer_meta_states', [])
        max_window_size = 0
        for idx, meta in enumerate(layer_meta_states):
            if not meta or len(meta) < 2:
                continue
            # Check if this layer is RotatingKVCache
            if idx < len(layer_cache_types) and layer_cache_types[idx] == 'RotatingKVCache':
                # RotatingKVCache meta_state: (keep, max_size, offset, _idx)
                window_size = int(meta[1])
                if window_size > max_window_size:
                    max_window_size = window_size

        if max_window_size > 0:
            model_cache_config._max_window_size = max_window_size

        return model_cache_config

    def fetch_cache(
        self,
        request_id: str,
        tokens: List[int],
    ) -> Tuple[Optional[BlockTable], List[int]]:
        """
        Find cached prefix blocks for the given tokens.

        Args:
            request_id: Unique request identifier
            tokens: Input token sequence

        Returns:
            Tuple of (block_table, remaining_tokens)
            - block_table: BlockTable if prefix found, None otherwise
            - remaining_tokens: Tokens that need processing
        """
        if not tokens:
            return None, tokens

        # Try to find shared prefix blocks
        shared_block_ids, remaining = self.paged_cache.find_shared_prefix(tokens)

        if shared_block_ids:
            # Create block table for this request with shared blocks
            block_table = self.paged_cache.create_block_table(request_id)

            for block_id in shared_block_ids:
                # Increment ref count for sharing
                self.paged_cache.increment_ref(block_id)
                block = self.paged_cache.allocated_blocks.get(block_id)
                if block:
                    block_table.block_ids.append(block_id)
                    block_table.num_tokens += block.token_count

            num_prefix_tokens = len(tokens) - len(remaining)
            self._hits += 1
            self._tokens_saved += num_prefix_tokens

            logger.debug(
                f"Cache hit for {request_id}: "
                f"{len(shared_block_ids)} blocks, {num_prefix_tokens} tokens"
            )

            return block_table, remaining

        # Try prefix index for longer matches
        best_match = self._find_best_prefix_match(tokens)
        if best_match:
            prefix_len, matched_block_ids, num_blocks = best_match

            # Fork the matched blocks
            block_table = self.paged_cache.create_block_table(request_id)
            for block_id in matched_block_ids[:num_blocks]:
                self.paged_cache.increment_ref(block_id)
                block = self.paged_cache.allocated_blocks.get(block_id)
                if block:
                    block_table.block_ids.append(block_id)
                    block_table.num_tokens += block.token_count

            remaining = tokens[prefix_len:]
            self._hits += 1
            self._tokens_saved += prefix_len

            logger.debug(
                f"Prefix index hit for {request_id}: "
                f"{prefix_len} tokens matched"
            )

            return block_table, remaining

        # No cache hit
        self._misses += 1
        logger.debug(f"Cache miss for {request_id}")
        return None, tokens

    def store_cache(
        self,
        request_id: str,
        tokens: List[int],
        cache_data: List[Any],
        model_cache_config: Optional[ModelCacheConfig] = None,
    ) -> Optional[BlockTable]:
        """
        Store computed cache for future reuse.

        In paged SSD-only mode, this method:
        1. Allocates block metadata in PagedCacheManager
        2. Extracts tensor slices for each block
        3. Saves each block's data to paged SSD via PagedSSDCacheManager

        Args:
            request_id: Unique request identifier
            tokens: Token sequence that was processed
            cache_data: The computed KV cache to store. Can be:
                - List of KVCache objects (legacy)
                - List of dicts with 'state': (keys, values) tensors (preferred)
            model_cache_config: Optional cache configuration with per-layer type
                information. If None, assumes all layers use KVCache.

        Returns:
            BlockTable for the stored cache, or None on failure
        """
        if not tokens:
            return None

        # Check if cache_data contains extracted tensor states
        is_tensor_data = (
            cache_data and
            isinstance(cache_data, list) and
            len(cache_data) > 0 and
            isinstance(cache_data[0], dict) and
            'state' in cache_data[0]
        )

        # Extract cache type information for SSD storage
        layer_cache_types = None
        layer_meta_states = None
        if model_cache_config:
            layer_cache_types = model_cache_config.get_type_names()
            # Extract meta_states if available in cache_data
            layer_meta_states = [
                cache_data[i].get('meta_state', ())
                if i < len(cache_data) else ()
                for i in range(model_cache_config.num_layers)
            ]
        elif is_tensor_data:
            # Try to extract type info from cache_data itself
            layer_cache_types = [
                layer_state.get('cache_type', 'KVCache')
                for layer_state in cache_data
            ]
            layer_meta_states = [
                layer_state.get('meta_state', ())
                for layer_state in cache_data
            ]

        # Get or create block table
        block_table = self.paged_cache.get_block_table(request_id)
        if not block_table:
            block_table = self.paged_cache.create_block_table(request_id)

        # Determine tokens we need to cache (not already in block_table)
        existing_tokens = block_table.num_tokens
        new_tokens = tokens[existing_tokens:]

        if not new_tokens:
            # All tokens already cached
            return block_table

        # Allocate only full blocks (skip partial trailing block).
        # get_computed_blocks() matches full blocks only (floor division),
        # so partial block data is never used during cache lookup.
        # Skipping partial blocks also ensures is_last_block points to
        # the last full block, which is critical for non-sliceable caches
        # (ArraysCache/RotatingKVCache) that use last-block-only storage.
        num_new_blocks = len(new_tokens) // self.block_size
        blocks_saved_to_ssd = 0

        for i in range(num_new_blocks):
            start_idx = i * self.block_size
            end_idx = min(start_idx + self.block_size, len(new_tokens))
            block_tokens = new_tokens[start_idx:end_idx]

            # Token range in the original sequence (accounting for existing tokens)
            global_start = existing_tokens + start_idx
            global_end = existing_tokens + end_idx

            # Compute parent hash for chain-based lookup
            parent_hash = None
            if block_table.block_ids:
                prev_block_id = block_table.block_ids[-1]
                prev_block = self.paged_cache.allocated_blocks.get(prev_block_id)
                if prev_block and prev_block.block_hash:
                    parent_hash = prev_block.block_hash

            # Check if this block already exists (deduplication)
            if len(block_tokens) == self.block_size:
                existing_block = self.paged_cache.find_cached_block(block_tokens, parent_hash)
                if existing_block:
                    # Reuse existing block
                    self.paged_cache.increment_ref(existing_block.block_id)
                    block_table.block_ids.append(existing_block.block_id)
                    block_table.num_tokens += len(block_tokens)
                    continue

            # Allocate new block
            block = self.paged_cache.allocate_block()
            if not block:
                # Handle memory pressure
                if not self.paged_cache.handle_memory_pressure(1):
                    logger.warning(f"Cannot allocate block for {request_id}")
                    break
                block = self.paged_cache.allocate_block()
                if not block:
                    break

            # Set block metadata
            block.token_count = len(block_tokens)
            block_table.block_ids.append(block.block_id)
            block_table.num_tokens += len(block_tokens)

            # Compute chain hash for this block
            block.block_hash = compute_block_hash(
                parent_hash, block_tokens, model_name=self.paged_cache.model_name
            )

            # Register hash for full blocks (for deduplication)
            if len(block_tokens) == self.block_size:
                self.paged_cache.register_block_hash(block, block_tokens, parent_hash)

            # Extract tensor slice and save to paged SSD
            if is_tensor_data and HAS_MLX and self.paged_ssd_cache is not None:
                cache_seq_len = self._get_cache_seq_len(cache_data)

                # Determine whether extracted cache_data uses:
                # - global indices (full sequence cache, includes reused prefix), or
                # - relative indices (only newly processed suffix).
                #
                # BatchGenerator.extract_cache() currently returns full-sequence cache.
                # When existing_tokens > 0, slicing with relative indices would save
                # wrong KV ranges for new blocks and corrupt future cache hits.
                cache_uses_global_indices = (
                    existing_tokens > 0 and cache_seq_len >= (existing_tokens + 1)
                )
                if cache_uses_global_indices:
                    cache_start = global_start
                    cache_end = global_end
                else:
                    cache_start = start_idx
                    cache_end = end_idx

                # Check cache continuity for the selected slice mode.
                if cache_seq_len > 0 and cache_start >= cache_seq_len:
                    logger.debug(
                        f"Cache continuity broken: cache only has {cache_seq_len} tokens, "
                        f"cannot store block at cache indices [{cache_start}:{cache_end}] "
                        f"(global [{global_start}:{global_end}]). Stopping block allocation."
                    )
                    # Free the block we just allocated (it has no data)
                    self.paged_cache.free_block(block.block_id)
                    block_table.block_ids.pop()
                    block_table.num_tokens -= len(block_tokens)
                    break

                is_last_block = (i == num_new_blocks - 1)
                block_kv_data = self._extract_block_tensor_slice(
                    cache_data, cache_start, cache_end, model_cache_config,
                    is_last_block=is_last_block,
                )

                if block_kv_data and block.block_hash:
                    # Save to paged SSD via PagedSSDCacheManager with cache type info
                    saved = self.paged_ssd_cache.save_block(
                        block_hash=block.block_hash,
                        cache_data=block_kv_data,
                        token_count=block.token_count,
                        model_name=self.paged_cache.model_name,
                        layer_cache_types=layer_cache_types,
                        layer_meta_states=layer_meta_states,
                    )
                    if saved:
                        blocks_saved_to_ssd += 1
                        logger.debug(
                            f"Saved block {block.block_id} to paged SSD: "
                            f"tokens [{global_start}:{global_end}], {len(block_kv_data)} layers"
                        )
                    else:
                        logger.warning(
                            f"Failed to save block {block.block_id} to paged SSD"
                        )
                else:
                    # Failed to extract tensor data - free block and stop
                    logger.debug(
                        f"Failed to extract tensor slice [{global_start}:{global_end}], "
                        f"freeing block {block.block_id} and stopping."
                    )
                    self.paged_cache.free_block(block.block_id)
                    block_table.block_ids.pop()
                    block_table.num_tokens -= len(block_tokens)
                    break

        # Update prefix index
        self._update_prefix_index(tokens, block_table.block_ids)

        # Store entry for request tracking
        self._request_tables[request_id] = BlockCacheEntry(
            block_table=block_table,
            last_access=time.time(),
        )

        logger.debug(
            f"Stored cache for {request_id}: "
            f"{len(block_table.block_ids)} blocks ({blocks_saved_to_ssd} saved to paged SSD), "
            f"{block_table.num_tokens} tokens"
        )

        return block_table

    def _get_cache_seq_len(self, cache_data: List[Dict[str, Any]]) -> int:
        """
        Get the sequence length from cache data.

        For hybrid models (e.g., gpt-oss, gemma3 with KVCache + RotatingKVCache layers),
        this finds a standard KVCache layer (full attention) to determine the actual
        seq_len. RotatingKVCache layers use sliding window and have limited seq_len.
        ArraysCache layers don't have a sequence dimension.

        Args:
            cache_data: List of layer states, each containing 'state': (keys, values)

        Returns:
            Sequence length from first sliceable KVCache layer, or max seq_len as fallback
        """
        if not cache_data:
            return 0

        # Non-sliceable cache types use sliding window or have no sequence dimension
        # RotatingKVCache: sliding window, seq_len limited to max_size
        # ArraysCache: no traditional sequence dimension
        non_sliceable_types = {'RotatingKVCache', 'ArraysCache', 'CacheList'}

        # Step 1: Search for a sliceable KVCache layer (full attention)
        for layer_idx, layer_state in enumerate(cache_data):
            try:
                if 'state' not in layer_state:
                    continue

                # Skip non-sliceable cache types (e.g., RotatingKVCache)
                cache_type = layer_state.get('cache_type', '')
                class_name = layer_state.get('class_name', '')
                if cache_type in non_sliceable_types or class_name in non_sliceable_types:
                    continue

                keys, _ = layer_state['state']
                if not hasattr(keys, 'shape'):
                    continue

                # KVCache: shape (batch, n_kv_heads, seq_len, head_dim) - 4D
                if len(keys.shape) == 4:
                    seq_len = keys.shape[2]
                    logger.debug(
                        f"Found KVCache at layer {layer_idx} with seq_len={seq_len}"
                    )
                    return seq_len

            except Exception:
                continue

        # Step 2: Fallback - find max seq_len among all 4D tensors
        # This handles pure RotatingKVCache models or unknown cache types.
        # Only skip cache types that do not expose a sequence dimension here.
        # RotatingKVCache must be included because pure RotatingKVCache models
        # have no sliceable KVCache layers for Step 1 to find.
        step2_skip_types = {'ArraysCache', 'CacheList'}
        max_seq_len = 0
        for layer_idx, layer_state in enumerate(cache_data):
            try:
                if 'state' not in layer_state:
                    continue
                cache_type = layer_state.get('cache_type', '')
                class_name = layer_state.get('class_name', '')
                if cache_type in step2_skip_types or class_name in step2_skip_types:
                    continue
                keys, _ = layer_state['state']
                if hasattr(keys, 'shape') and len(keys.shape) == 4:
                    max_seq_len = max(max_seq_len, keys.shape[2])
            except Exception:
                continue

        if max_seq_len > 0:
            logger.debug(f"Using fallback max seq_len={max_seq_len}")
            return max_seq_len

        # Step 3: CacheList fallback — check sub-states for seq_len
        # This handles all-CacheList models (e.g., deepseek_v32)
        for layer_state in cache_data:
            if (layer_state.get('cache_type') == 'CacheList'
                    or layer_state.get('class_name') == 'CacheList'):
                sub_states = layer_state.get('state', [])
                for sub_state in sub_states:
                    if isinstance(sub_state, (list, tuple)) and len(sub_state) >= 2:
                        sub_keys = sub_state[0]
                        if hasattr(sub_keys, 'shape') and len(sub_keys.shape) == 4:
                            seq_len = sub_keys.shape[2]
                            logger.debug(
                                f"Using CacheList sub-cache seq_len={seq_len}"
                            )
                            return seq_len

        return 0

    def _extract_block_tensor_slice(
        self,
        cache_data: List[Dict[str, Any]],
        start_idx: int,
        end_idx: int,
        model_cache_config: Optional[ModelCacheConfig] = None,
        is_last_block: bool = False,
    ) -> Optional[List[Tuple[Any, Any]]]:
        """
        Extract tensor slices for a single block from cache data.

        Supports different cache types (KVCache, RotatingKVCache, ArraysCache)
        with type-aware slicing. For non-sliceable types like ArraysCache,
        returns the full state.

        For RotatingKVCache layers specifically:
        - Last block: stores the full RotatingKVCache state (keys, values)
        - Non-last blocks: stores a placeholder (mx.zeros((1,)), mx.zeros((1,)))
          to preserve layer count while minimizing storage

        During restore, a partial prefix match that ends on a placeholder block
        is rejected. Exact block-boundary matches restore the last full rotating
        state directly (no window-padding recomputation).

        Args:
            cache_data: List of layer states, each containing 'state': (keys, values)
                or other cache-type-specific format
            start_idx: Start token index in the sequence
            end_idx: End token index in the sequence
            model_cache_config: Optional model cache configuration with per-layer
                type information
            is_last_block: If True, this is the last block being stored. For
                RotatingKVCache layers, only the last block stores full state.

        Returns:
            List of (keys_slice, values_slice) for each layer, or None on failure
        """
        if not HAS_MLX or not cache_data:
            return None

        try:
            block_slices = []
            for layer_idx, layer_state in enumerate(cache_data):
                if 'state' not in layer_state:
                    continue

                # Determine cache type for this layer
                cache_type_name = layer_state.get('cache_type', 'KVCache')
                if model_cache_config and layer_idx < len(model_cache_config.layer_configs):
                    cache_type_name = model_cache_config.layer_configs[layer_idx].class_name

                handler = CacheTypeRegistry.get_handler_by_class_name(cache_type_name)

                if handler.supports_block_slicing:
                    # Standard 4D KV cache slicing
                    keys, values = layer_state['state']

                    # KV cache shape: (batch, n_kv_heads, seq_len, head_dim)
                    # Slice along seq_len dimension (axis 2)
                    if not hasattr(keys, 'shape') or len(keys.shape) < 4:
                        # Handle 3D case (no batch dimension)
                        if hasattr(keys, 'shape') and len(keys.shape) == 3:
                            seq_len = keys.shape[1]  # (n_kv_heads, seq_len, head_dim)
                            actual_end = min(end_idx, seq_len)
                            if start_idx >= actual_end:
                                continue
                            keys_slice = keys[:, start_idx:actual_end, :]
                            values_slice = values[:, start_idx:actual_end, :]
                        else:
                            logger.debug(
                                f"Layer {layer_idx}: unexpected tensor shape for {cache_type_name}"
                            )
                            continue
                    else:
                        seq_len = keys.shape[2]
                        if end_idx > seq_len:
                            logger.debug(
                                f"Block slice [{start_idx}:{end_idx}] exceeds seq_len {seq_len}"
                            )
                            actual_end = min(end_idx, seq_len)
                            if start_idx >= actual_end:
                                continue
                            keys_slice = keys[:, :, start_idx:actual_end, :]
                            values_slice = values[:, :, start_idx:actual_end, :]
                        else:
                            keys_slice = keys[:, :, start_idx:end_idx, :]
                            values_slice = values[:, :, start_idx:end_idx, :]

                    # Detach slices so block-level eviction can free memory
                    block_slices.append(
                        (self._clone_tensor(keys_slice), self._clone_tensor(values_slice))
                    )
                elif cache_type_name == 'RotatingKVCache':
                    # RotatingKVCache: last-block-only storage strategy
                    if is_last_block:
                        # Last block: store full RotatingKVCache state
                        state = layer_state['state']
                        if isinstance(state, (list, tuple)) and len(state) >= 2:
                            keys = state[0]
                            values = state[1]
                            block_slices.append(
                                (self._clone_tensor(keys), self._clone_tensor(values))
                            )
                        else:
                            logger.debug(
                                f"Layer {layer_idx}: RotatingKVCache unexpected state format"
                            )
                            block_slices.append((mx.zeros((1,)), mx.zeros((1,))))
                    else:
                        # Non-last block: store placeholder to preserve layer count
                        block_slices.append((mx.zeros((1,)), mx.zeros((1,))))
                elif cache_type_name == 'CacheList':
                    state = layer_state['state']  # List[sub_state]
                    if not isinstance(state, list) or len(state) == 0:
                        block_slices.append((mx.zeros((1,)), mx.zeros((1,))))
                        continue

                    # Check if all sub-caches are sliceable 4D KVCache tensors
                    all_sub_sliceable = all(
                        isinstance(ss, (list, tuple)) and len(ss) >= 2
                        and hasattr(ss[0], 'shape') and len(ss[0].shape) == 4
                        for ss in state
                    )

                    if all_sub_sliceable:
                        # Per-block slicing: slice each sub-cache along seq_len
                        sub_tensors = []
                        for sub_state in state:
                            sub_keys, sub_values = sub_state[0], sub_state[1]
                            seq_len = sub_keys.shape[2]
                            actual_end = min(end_idx, seq_len)
                            if start_idx >= actual_end:
                                sub_tensors.append((
                                    self._clone_tensor(sub_keys[:, :, 0:0, :]),
                                    self._clone_tensor(sub_values[:, :, 0:0, :]),
                                ))
                            else:
                                sub_tensors.append((
                                    self._clone_tensor(sub_keys[:, :, start_idx:actual_end, :]),
                                    self._clone_tensor(sub_values[:, :, start_idx:actual_end, :]),
                                ))
                        block_slices.append(('__cache_list__', sub_tensors))
                    else:
                        # Non-sliceable sub-caches: last-block-only storage
                        if is_last_block:
                            sub_tensors = []
                            for sub_state in state:
                                if isinstance(sub_state, (list, tuple)) and len(sub_state) >= 2:
                                    sub_tensors.append((
                                        self._clone_tensor(sub_state[0]),
                                        self._clone_tensor(sub_state[1]),
                                    ))
                            block_slices.append(('__cache_list__', sub_tensors))
                        else:
                            block_slices.append((mx.zeros((1,)), mx.zeros((1,))))
                else:
                    # Other non-sliceable cache (ArraysCache/MambaCache)
                    # Last-block-only storage strategy (same as RotatingKVCache).
                    # GDN recurrent state summarizes the ENTIRE sequence in a
                    # fixed-size matrix. Storing the same final state in every
                    # block is both wasteful and dangerous: a partial prefix
                    # match would restore stale state containing future token
                    # information. Only the last block carries the valid state;
                    # non-last blocks get a placeholder so partial matches are
                    # detected and rejected during reconstruction.
                    if is_last_block:
                        state = layer_state['state']
                        if isinstance(state, (list, tuple)) and len(state) >= 2:
                            conv_state = state[0] if state[0] is not None else mx.array([])
                            ssm_state = state[1] if state[1] is not None else mx.array([])
                            block_slices.append(
                                (self._clone_tensor(conv_state), self._clone_tensor(ssm_state))
                            )
                        else:
                            logger.debug(
                                f"Layer {layer_idx}: {cache_type_name} unexpected state format"
                            )
                            block_slices.append((mx.zeros((1,)), mx.zeros((1,))))
                    else:
                        # Non-last block: store placeholder to preserve layer count
                        block_slices.append((mx.zeros((1,)), mx.zeros((1,))))

            return block_slices if block_slices else None

        except Exception as e:
            logger.warning(f"Failed to extract block tensor slice: {e}")
            return None

    @staticmethod
    def _is_placeholder_state(data) -> bool:
        """Check if block layer data is a last-block-only placeholder.

        Non-sliceable cache types (ArraysCache, CacheList with non-sliceable
        sub-caches) store real state only in the last block of each save
        operation. All other blocks get a placeholder: ``(mx.zeros((1,)),
        mx.zeros((1,)))``.

        Returns True if *data* is such a placeholder.
        """
        # CacheList real sub-cache data is stored as a list, never a placeholder
        if isinstance(data, list):
            return False
        if isinstance(data, tuple) and len(data) == 2:
            first = data[0]
            if hasattr(first, 'shape') and first.shape == (1,):
                return True
        return False

    def _find_walk_back_truncation_point(
        self,
        all_block_data: List[List[Any]],
        layer_cache_types: Optional[List[str]],
    ) -> Optional[int]:
        """Find the latest block where all non-sliceable layers have valid state.

        In multi-turn conversations, intermediate blocks can accumulate real
        ArraysCache/CacheList state from prior save operations while later
        blocks only have placeholders.  This method walks backwards from the
        last loaded block to locate the most recent block where **every**
        non-sliceable layer (excluding RotatingKVCache, which has its own
        empty-cache recovery) carries real state.

        Returns:
            0-based block index (inclusive) to truncate to, or ``None`` if
            no truncation is needed (last block already valid) or no valid
            fallback block exists.
        """
        if not all_block_data or not layer_cache_types:
            return None

        num_layers = len(all_block_data[0])
        last_idx = len(all_block_data) - 1

        # Identify "problematic" layers: placeholder in the last block AND
        # not RotatingKVCache (which has its own empty-cache + window-padding
        # recovery path).
        problematic_layers: List[int] = []
        for layer_idx in range(num_layers):
            cache_type = (
                layer_cache_types[layer_idx]
                if layer_idx < len(layer_cache_types)
                else 'KVCache'
            )
            # KVCache is always per-block sliced; RotatingKVCache has its own
            # recovery.  Only check types that use last-block-only storage
            # without a built-in fallback.
            if cache_type in ('KVCache', 'RotatingKVCache'):
                continue
            if layer_idx < len(all_block_data[last_idx]):
                if self._is_placeholder_state(all_block_data[last_idx][layer_idx]):
                    problematic_layers.append(layer_idx)

        if not problematic_layers:
            return None  # Last block already has valid state for all layers

        # Walk backwards to find the latest block where ALL problematic
        # layers have real (non-placeholder) state.
        for block_idx in range(last_idx - 1, -1, -1):
            block_data = all_block_data[block_idx]
            all_valid = True
            for layer_idx in problematic_layers:
                if layer_idx < len(block_data) and self._is_placeholder_state(
                    block_data[layer_idx]
                ):
                    all_valid = False
                    break
            if all_valid:
                return block_idx

        return None  # No valid fallback -- fall through to existing rejection

    def _clone_tensor(self, tensor: Any) -> Any:
        """Clone a tensor slice to avoid holding the full backing buffer."""
        try:
            if hasattr(mx, "copy"):
                return mx.copy(tensor)
        except Exception:
            pass

        if hasattr(tensor, "copy"):
            try:
                return tensor.copy()
            except Exception:
                pass

        return mx.array(tensor)

    def _apply_window_padding(
        self,
        matched_blocks: int,
        model_cache_config: Optional[ModelCacheConfig] = None,
    ) -> int:
        """Calculate safe restore limit with window padding for hybrid models.

        For models with RotatingKVCache (sliding window attention), we need to
        ensure the sliding window is fully populated when generation starts.
        This means restoring fewer blocks and reprocessing the padding tokens.

        Example (Gemma3: window_size=1024, block_size=256):
            16 blocks matched -> restore 12 blocks (padding 4 blocks)
            The 4 padding blocks (1024 tokens) will be reprocessed to fill
            the RotatingKVCache sliding window.

        Args:
            matched_blocks: Number of matched cache blocks
            model_cache_config: Model cache configuration

        Returns:
            Number of blocks to actually restore (may be less than matched_blocks)
        """
        if model_cache_config is None or not model_cache_config.has_rotating_layers():
            return matched_blocks

        window_size = model_cache_config.get_max_window_size()
        if window_size <= 0:
            return matched_blocks

        padding_blocks = math.ceil(window_size / self.block_size)
        blocks_to_restore = max(0, matched_blocks - padding_blocks)

        if blocks_to_restore < matched_blocks:
            logger.debug(
                f"Window padding: {matched_blocks} blocks matched, "
                f"restoring {blocks_to_restore} blocks "
                f"(padding {padding_blocks} blocks for window_size={window_size})"
            )

        return blocks_to_restore

    def get_cache_for_generation(
        self,
        request_id: str,
    ) -> Tuple[Optional[List[Any]], bool]:
        """
        Get cache data for generation, loading from paged SSD if needed.

        In paged SSD-only mode, cache data is always loaded from paged SSD via
        reconstruct_cache().

        Args:
            request_id: Request identifier

        Returns:
            Tuple of (cache_data, was_loaded_from_ssd)
        """
        entry = self._request_tables.get(request_id)
        if not entry:
            return None, False

        # Get blocks with COW
        _, was_copied = self.paged_cache.get_blocks_for_generation(entry.block_table)

        # In paged SSD-only mode, always reconstruct from paged SSD
        cache_data = self.reconstruct_cache(entry.block_table)
        if cache_data is None:
            return None, False

        entry.last_access = time.time()
        return cache_data, True

    def release_cache(self, request_id: str) -> None:
        """
        Release cache blocks for a completed request.

        Args:
            request_id: Request identifier
        """
        entry = self._request_tables.pop(request_id, None)
        if entry:
            self.paged_cache.delete_block_table(request_id)
            logger.debug(f"Released cache for {request_id}")

    def clear_request_entry(self, request_id: str) -> None:
        """
        Clear request entry from tracking without freeing blocks.

        This removes the request from _request_tables but keeps the cached
        blocks available for prefix matching. Use this after store_cache()
        when the request is complete but cache should remain for future reuse.

        Args:
            request_id: Request identifier
        """
        entry = self._request_tables.pop(request_id, None)
        if entry:
            logger.debug(f"Cleared request entry for {request_id} (blocks retained)")

    def fork_cache(
        self,
        source_request_id: str,
        new_request_id: str,
    ) -> Optional[BlockTable]:
        """
        Fork cache from one request to another (COW).

        In paged SSD-only mode, cache data is always on paged SSD, so we just
        increment reference counts for the blocks.

        Args:
            source_request_id: Source request ID
            new_request_id: New request ID

        Returns:
            Forked BlockTable, or None if source not found
        """
        source_entry = self._request_tables.get(source_request_id)
        if not source_entry:
            return None

        # Fork block table (increments ref counts)
        forked_table = self.paged_cache.fork_block_table(
            source_entry.block_table,
            new_request_id,
        )

        # Create new entry (cache data is on paged SSD)
        self._request_tables[new_request_id] = BlockCacheEntry(
            block_table=forked_table,
            last_access=time.time(),
        )

        logger.debug(
            f"Forked cache: {source_request_id} -> {new_request_id}"
        )

        return forked_table

    def reconstruct_cache(
        self,
        block_table: BlockTable,
    ) -> Optional[List[Any]]:
        """
        Reconstruct cache objects from paged SSD-stored block data.

        This method supports multiple cache types (KVCache, RotatingKVCache,
        ArraysCache) and uses stored type information for proper reconstruction.

        In paged SSD-only mode, this method:
        1. Loads block tensor data from paged SSD via PagedSSDCacheManager
        2. Gets cache type info from paged SSD metadata
        3. Concatenates tensors for each layer (or uses full state for non-sliceable)
        4. Creates appropriate cache objects for inference

        If some blocks cannot be loaded, this method will use only the valid
        prefix blocks and update block_table in-place.

        Args:
            block_table: BlockTable containing block IDs to reconstruct from.
                        Will be modified in-place if partial reconstruction.

        Returns:
            List of reconstructed cache objects (one per layer),
            or None if reconstruction fails completely
        """
        if not block_table or not block_table.block_ids:
            return None

        if not HAS_MLX:
            logger.warning("Cannot reconstruct cache: MLX not available")
            return None

        if self.paged_ssd_cache is None:
            logger.warning("Cannot reconstruct cache: PagedSSDCacheManager not configured")
            return None

        try:
            # Collect cache data from valid blocks (stop at first invalid)
            all_block_data = []
            valid_block_count = 0
            valid_token_count = 0

            # Cache type information from blocks
            layer_cache_types = None
            first_block_meta_states = None   # meta_states from first block
            last_block_meta_states = None    # meta_states from last block (for non-sliceable caches)
            all_block_meta_states = []       # per-block meta_states for walk-back truncation

            for idx, block_id in enumerate(block_table.block_ids):
                block = self.paged_cache.allocated_blocks.get(block_id)
                if not block:
                    logger.debug(
                        f"Block {block_id} not found, using {valid_block_count} "
                        f"valid blocks ({valid_token_count} tokens)"
                    )
                    break  # Stop at first missing block, use valid prefix

                # Load block data from paged SSD
                if block.block_hash is None:
                    logger.debug(
                        f"Block {block_id} has no block_hash, "
                        f"using {valid_block_count} valid blocks"
                    )
                    break  # Stop here, use valid prefix

                # Load with metadata for type information
                block_data, block_metadata = self.paged_ssd_cache.load_block_with_metadata(
                    block.block_hash
                )
                if block_data is None:
                    logger.debug(
                        f"Failed to load block {block_id} from paged SSD, "
                        f"using {valid_block_count} valid blocks"
                    )
                    # Remove failed block from hash cache to prevent future false hits
                    if block.block_hash is not None:
                        self.paged_cache.cached_block_hash_to_block.pop(
                            block.block_hash, block.block_id
                        )
                        logger.debug(
                            f"Removed missing block {block_id} from hash cache"
                        )
                    break  # Stop here, use valid prefix

                # Validate model_name to prevent cross-model cache contamination
                if block_metadata:
                    block_model_name = block_metadata.get('model_name', '')
                    current_model_name = self.paged_cache.model_name

                    # If current model has a name, validate against block's model
                    if current_model_name:
                        if not block_model_name:
                            # Block was saved without model_name (old cache), skip it
                            logger.warning(
                                f"Block has no model_name (legacy cache), "
                                f"current model is '{current_model_name}'. Invalidating cache hit."
                            )
                            break  # Stop here, don't use this block
                        elif block_model_name != current_model_name:
                            # Block is from a different model
                            logger.warning(
                                f"Cache model mismatch: block is for '{block_model_name}', "
                                f"current model is '{current_model_name}'. Invalidating cache hit."
                            )
                            break  # Stop here, don't use this block

                    # Validate num_layers to catch cross-model cache issues
                    block_num_layers = block_metadata.get('num_layers', 0)
                    if self.expected_num_layers > 0 and block_num_layers > 0:
                        if block_num_layers != self.expected_num_layers:
                            logger.warning(
                                f"Cache layer count mismatch: block has {block_num_layers} layers, "
                                f"expected {self.expected_num_layers}. Invalidating cache hit."
                            )
                            break  # Stop here, don't use this block

                # Extract type info from block metadata
                if block_metadata:
                    if layer_cache_types is None:
                        layer_cache_types = block_metadata.get('layer_cache_types')

                    # Track meta_states from first and last blocks
                    # Non-sliceable caches (RotatingKVCache) need last block's meta_state
                    block_layer_meta_states = block_metadata.get('layer_meta_states')
                    if first_block_meta_states is None:
                        first_block_meta_states = block_layer_meta_states
                    # Always update last to track the most recent
                    last_block_meta_states = block_layer_meta_states
                    all_block_meta_states.append(block_layer_meta_states)

                # Validate loaded data (pass cache types for hybrid models)
                if not self._validate_block_cache_data(block_data, layer_cache_types):
                    logger.debug(
                        f"Block {block_id} has invalid layer data from paged SSD, "
                        f"using {valid_block_count} valid blocks"
                    )
                    break  # Stop here, use valid prefix

                all_block_data.append(block_data)
                valid_block_count += 1
                valid_token_count += block.token_count

            # If we have fewer valid blocks than requested, update block_table
            if valid_block_count < len(block_table.block_ids):
                if valid_block_count == 0:
                    # Free ref_counts for all blocks before returning
                    for bid in block_table.block_ids:
                        self.paged_cache.free_block(bid)
                    return None  # No valid blocks at all

                # Free ref_counts for blocks we are about to drop
                for bid in block_table.block_ids[valid_block_count:]:
                    self.paged_cache.free_block(bid)

                # Truncate block_table to valid prefix
                original_blocks = len(block_table.block_ids)
                block_table.block_ids = block_table.block_ids[:valid_block_count]
                block_table.num_tokens = valid_token_count
                logger.info(
                    f"Partial cache reconstruction: {valid_block_count}/{original_blocks} "
                    f"blocks, {valid_token_count} tokens"
                )

            if not all_block_data:
                return None

            # Get number of layers from first block
            num_layers = len(all_block_data[0])
            if num_layers == 0:
                return None

            # --- Pre-scan: walk-back truncation for non-sliceable caches ---
            # If the last loaded block has a placeholder for any non-sliceable
            # layer (ArraysCache, non-sliceable CacheList), walk backwards to
            # find the latest block where ALL such layers carry real state.
            # This recovers intermediate blocks from multi-turn conversations
            # instead of rejecting the entire cache.
            if all_block_data and layer_cache_types:
                trunc_idx = self._find_walk_back_truncation_point(
                    all_block_data, layer_cache_types
                )
                if trunc_idx is not None:
                    new_count = trunc_idx + 1
                    dropped_count = len(all_block_data) - new_count

                    # Free ref_counts for dropped blocks
                    for bid in block_table.block_ids[new_count:]:
                        self.paged_cache.free_block(bid)

                    # Truncate data structures
                    all_block_data = all_block_data[:new_count]
                    block_table.block_ids = block_table.block_ids[:new_count]
                    valid_token_count = sum(
                        self.paged_cache.allocated_blocks[bid].token_count
                        for bid in block_table.block_ids
                        if bid in self.paged_cache.allocated_blocks
                    )
                    block_table.num_tokens = valid_token_count

                    # Update meta_states to the truncation-point block
                    if trunc_idx < len(all_block_meta_states):
                        last_block_meta_states = all_block_meta_states[trunc_idx]

                    logger.info(
                        f"Walk-back truncation: dropped {dropped_count} trailing "
                        f"block(s) with placeholder non-sliceable state, keeping "
                        f"{new_count} block(s) ({valid_token_count} tokens)"
                    )

            # Build model cache config if we have type info
            model_cache_config = None
            if layer_cache_types and len(layer_cache_types) == num_layers:
                model_cache_config = ModelCacheConfig.from_type_list(
                    layer_cache_types, model_name=""
                )

            # Reconstruct caches for each layer
            reconstructed_caches = []

            for layer_idx in range(num_layers):
                # Determine cache type for this layer
                cache_type_name = "KVCache"
                if layer_cache_types and layer_idx < len(layer_cache_types):
                    cache_type_name = layer_cache_types[layer_idx]

                handler = CacheTypeRegistry.get_handler_by_class_name(cache_type_name)

                # === CacheList: dedicated branch (before standard 2-tuple unpack) ===
                if cache_type_name == 'CacheList':
                    last_block_layer_data = all_block_data[-1][layer_idx]

                    # Placeholder detection (partial match → reject for
                    # non-sliceable CacheList, e.g. containing ArraysCache)
                    if (isinstance(last_block_layer_data, tuple)
                            and len(last_block_layer_data) == 2
                            and hasattr(last_block_layer_data[0], 'shape')
                            and last_block_layer_data[0].shape == (1,)):
                        logger.info(
                            f"CacheList layer {layer_idx}: partial prefix match "
                            f"detected (placeholder). Rejecting cache."
                        )
                        return None

                    # Collect CacheList data from all blocks that have List[Tuple]
                    cl_block_data = []
                    for block_data in all_block_data:
                        bd = block_data[layer_idx]
                        if isinstance(bd, list) and all(
                            isinstance(t, (list, tuple)) and len(t) >= 2
                            for t in bd
                        ):
                            cl_block_data.append(bd)

                    if not cl_block_data:
                        logger.error(
                            f"CacheList layer {layer_idx}: no valid block data found"
                        )
                        return None

                    # Determine sub-cache count from first valid block
                    num_sub_caches = len(cl_block_data[0])

                    if len(cl_block_data) > 1:
                        # Per-block sliced CacheList: concatenate sub-caches
                        concatenated_sub_states = []
                        for j in range(num_sub_caches):
                            all_keys = [bd[j][0] for bd in cl_block_data]
                            all_values = [bd[j][1] for bd in cl_block_data]
                            cat_keys = mx.concatenate(all_keys, axis=2)
                            # Handle zero-dim values (e.g., DSA indexer head_dim=0)
                            if any(d == 0 for d in all_values[0].shape):
                                shape = list(all_values[0].shape)
                                shape[2] = sum(v.shape[2] for v in all_values)
                                cat_values = mx.zeros(tuple(shape))
                            else:
                                cat_values = mx.concatenate(all_values, axis=2)
                            concatenated_sub_states.append((cat_keys, cat_values))
                    else:
                        # Single block: use directly
                        concatenated_sub_states = cl_block_data[0]

                    # Build meta_state with correct offsets for reconstructed
                    # sequence length (may differ from original if partial match)
                    meta_state = None
                    if last_block_meta_states and layer_idx < len(last_block_meta_states):
                        meta_state = last_block_meta_states[layer_idx]

                    if meta_state and isinstance(meta_state, (list, tuple)) and len(meta_state) >= 2:
                        # Adjust sub-cache offsets to actual concatenated seq_len
                        class_names = meta_state[0]
                        adjusted_sub_metas = []
                        for j in range(num_sub_caches):
                            actual_seq_len = concatenated_sub_states[j][0].shape[2]
                            if j < len(meta_state[1]):
                                orig_sub_meta = meta_state[1][j]
                                if isinstance(orig_sub_meta, (list, tuple)) and len(orig_sub_meta) > 0:
                                    # Replace offset (first element) with actual seq_len
                                    adjusted_sub_metas.append(
                                        (actual_seq_len,) + tuple(orig_sub_meta[1:])
                                    )
                                else:
                                    adjusted_sub_metas.append((actual_seq_len,))
                            else:
                                adjusted_sub_metas.append((actual_seq_len,))
                        meta_state = (class_names, adjusted_sub_metas)

                    cache = handler.reconstruct_cache(
                        {'sub_states': concatenated_sub_states}, meta_state
                    )
                    if cache is None:
                        logger.error(f"CacheList layer {layer_idx}: reconstruction failed")
                        return None
                    reconstructed_caches.append(cache)
                    continue

                # Collect layer data from all blocks
                layer_states = []
                for block_data in all_block_data:
                    if layer_idx < len(block_data):
                        keys_slice, values_slice = block_data[layer_idx]
                        if keys_slice is not None and values_slice is not None:
                            layer_states.append({
                                'keys': keys_slice,
                                'values': values_slice,
                            })

                if not layer_states:
                    logger.debug(
                        f"Layer {layer_idx} has no data, cannot reconstruct cache"
                    )
                    return None

                # Get meta_state for this layer based on cache type
                meta_state = None
                if not handler.supports_block_slicing:
                    # Non-sliceable caches (RotatingKVCache, ArraysCache): use LAST block's meta_state
                    # because we use the last block's data (layer_states[-1])
                    if last_block_meta_states and layer_idx < len(last_block_meta_states):
                        meta_state = last_block_meta_states[layer_idx]
                else:
                    # Sliceable caches (KVCache): first block's meta_state is fine
                    if first_block_meta_states and layer_idx < len(first_block_meta_states):
                        meta_state = first_block_meta_states[layer_idx]

                # Reconstruct using appropriate handler
                if handler.supports_block_slicing:
                    # Standard concatenation for KVCache
                    concat_state = handler.concatenate_states(layer_states)
                    cache = handler.reconstruct_cache(concat_state, meta_state)
                else:
                    # Non-sliceable cache: use latest state
                    # States were stored as full state, use last one
                    latest_keys = layer_states[-1].get('keys')
                    latest_values = layer_states[-1].get('values')

                    if cache_type_name == 'RotatingKVCache':
                        # RotatingKVCache: strict last-block restore.
                        # If the last matched block is a placeholder, we only
                        # had a partial prefix hit and must reject.
                        if (hasattr(latest_keys, 'shape')
                                and latest_keys.shape == (1,)):
                            logger.info(
                                f"RotatingKVCache layer {layer_idx}: partial prefix "
                                f"match detected (placeholder in last matched "
                                f"block). Rejecting cache to prevent stale "
                                f"sliding-window state."
                            )
                            return None

                        latest_state = {
                            'keys': latest_keys,
                            'values': latest_values,
                            'meta_state': meta_state,
                        }
                        cache = handler.reconstruct_cache(latest_state, meta_state)
                    else:
                        # ArraysCache/MambaCache: detect placeholder from
                        # last-block-only storage. If the last matched block
                        # has placeholder shape (1,), this is a partial prefix
                        # match — the real state lives in a later block that
                        # was not matched. We must reject the entire cache
                        # because GDN recurrent state cannot be partially
                        # reconstructed.
                        if (hasattr(latest_keys, 'shape')
                                and latest_keys.shape == (1,)):
                            logger.info(
                                f"ArraysCache layer {layer_idx}: partial prefix "
                                f"match detected (placeholder in last matched "
                                f"block). Rejecting cache to prevent stale GDN "
                                f"state. Request will reprocess from scratch."
                            )
                            return None

                        # Exact match: last block has full state
                        latest_state = {
                            'states': [latest_keys, latest_values],
                        }
                        # Pass token_count for proper SizedArraysCache wrapping
                        cache = handler.reconstruct_cache(
                            latest_state,
                            meta_state,
                            token_count=valid_token_count,
                        )

                if cache is None:
                    # Fallback to simple KVCache reconstruction
                    cache = self._fallback_reconstruct_layer(
                        layer_states, cache_type_name
                    )

                if cache is None:
                    logger.debug(
                        f"Layer {layer_idx}: failed to reconstruct {cache_type_name}"
                    )
                    return None

                reconstructed_caches.append(cache)

            if not reconstructed_caches:
                return None

            # Verify all layers were reconstructed
            if len(reconstructed_caches) != num_layers:
                logger.warning(
                    f"Incomplete cache reconstruction: got {len(reconstructed_caches)} "
                    f"layers, expected {num_layers}"
                )
                return None

            logger.debug(
                f"Reconstructed cache from paged SSD: {len(reconstructed_caches)} layers, "
                f"{block_table.num_tokens} tokens from {len(block_table.block_ids)} blocks"
            )

            return reconstructed_caches

        except Exception as e:
            logger.warning(f"Failed to reconstruct cache: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def _fallback_reconstruct_layer(
        self,
        layer_states: List[Dict[str, Any]],
        cache_type_name: str,
    ) -> Optional[Any]:
        """
        Fallback layer reconstruction when handler fails.

        Args:
            layer_states: List of state dicts with 'keys' and 'values'
            cache_type_name: Name of the cache type

        Returns:
            Reconstructed cache object or None
        """
        try:
            # Collect keys and values
            layer_keys = [s['keys'] for s in layer_states if s.get('keys') is not None]
            layer_values = [s['values'] for s in layer_states if s.get('values') is not None]

            if not layer_keys or not layer_values:
                return None

            # Try to concatenate (works for 4D KV caches)
            try:
                concat_keys = mx.concatenate(layer_keys, axis=2)
                concat_values = mx.concatenate(layer_values, axis=2)
            except Exception:
                # If concatenation fails, might be 3D tensors
                try:
                    concat_keys = mx.concatenate(layer_keys, axis=1)
                    concat_values = mx.concatenate(layer_values, axis=1)
                except Exception:
                    # Last resort: use single state
                    concat_keys = layer_keys[-1]
                    concat_values = layer_values[-1]

            # Create appropriate cache object
            try:
                from mlx_lm.models.cache import KVCache
                cache = KVCache()
                cache.keys = concat_keys
                cache.values = concat_values
                if len(concat_keys.shape) >= 3:
                    cache.offset = concat_keys.shape[2] if len(concat_keys.shape) == 4 else concat_keys.shape[1]
                else:
                    cache.offset = 0
                return cache
            except ImportError:
                # Simple fallback
                class SimpleKVCache:
                    def __init__(self, keys, values):
                        self.keys = keys
                        self.values = values
                        self.offset = keys.shape[2] if len(keys.shape) >= 3 else 0

                    @property
                    def state(self):
                        return (self.keys, self.values)

                return SimpleKVCache(concat_keys, concat_values)

        except Exception as e:
            logger.debug(f"Fallback reconstruction failed: {e}")
            return None

    def _find_kv_shape_ref(
        self,
        all_block_data: List[List[Tuple[Any, Any]]],
        layer_cache_types: Optional[List[str]] = None,
    ) -> Optional[Tuple[int, int]]:
        """Find (kv_heads, head_dim) from a KVCache layer's stored data.

        Used to create zero-length RotatingKVCache tensors with the correct shape.

        Args:
            all_block_data: All loaded block data
            layer_cache_types: Per-layer cache type names

        Returns:
            (kv_heads, head_dim) tuple, or None if not found
        """
        if not all_block_data:
            return None

        for layer_idx, layer_data in enumerate(all_block_data[0]):
            # Skip non-KVCache layers
            if layer_cache_types and layer_idx < len(layer_cache_types):
                if layer_cache_types[layer_idx] != 'KVCache':
                    continue
            # Guard against non-tuple formats (CacheList stores List[Tuple])
            if not isinstance(layer_data, tuple) or len(layer_data) != 2:
                continue
            keys, _ = layer_data
            if hasattr(keys, 'shape') and len(keys.shape) == 4:
                return (keys.shape[1], keys.shape[3])

        return None

    def _create_empty_rotating_cache(
        self,
        meta_state: Optional[tuple] = None,
        kvcache_offset: int = 0,
        kv_shape_ref: Optional[Tuple[int, int]] = None,
    ) -> Optional[Any]:
        """
        Create an empty RotatingKVCache for partial prefix restore.

        Creates a RotatingKVCache with zero-length keys/values (not None) and
        offset matching the KVCache layers. This ensures:
        1. mlx-lm's empty() returns False → Continuation mode (not Fresh Start)
        2. Position IDs (RoPE) are correct for all layers
        3. The merge creates a zero-length buffer (not zero-filled) so that
           no phantom attention positions exist during window padding reprocessing

        Uses _PrefillReadyRotatingKVCache which overrides size() to return 0
        for zero-length keys, preventing BatchRotatingKVCache.merge() from
        allocating a zero-filled buffer without left_padding masking.

        Args:
            meta_state: RotatingKVCache meta_state tuple (keep, max_size, offset, _idx).
            kvcache_offset: Offset to match KVCache layers (= restored token count).
            kv_shape_ref: (kv_heads, head_dim) from a KVCache layer for tensor shape.

        Returns:
            RotatingKVCache with zero-length keys/values, or None on failure.
        """
        global _PrefillReadyRotatingKVCache

        try:
            from mlx_lm.models.cache import RotatingKVCache
        except ImportError:
            logger.error("mlx_lm not available for empty RotatingKVCache creation")
            return None

        # Lazily create subclass that overrides size() for correct merge behavior.
        # Standard RotatingKVCache.size() returns min(offset, max_size) which
        # incorrectly reports data size > 0 for zero-length keys, causing
        # BatchRotatingKVCache.merge() to create unmasked zero-filled buffers
        # that dilute attention scores during prefill.
        if _PrefillReadyRotatingKVCache is None:
            class _Impl(RotatingKVCache):
                """RotatingKVCache that reports actual buffer size for merge."""
                def size(self):
                    if self.keys is not None and self.keys.shape[2] == 0:
                        return 0
                    return super().size()
            _PrefillReadyRotatingKVCache = _Impl

        if meta_state and len(meta_state) >= 2:
            keep = int(meta_state[0])
            max_size = int(meta_state[1])
        else:
            logger.warning(
                "Cannot create empty RotatingKVCache: meta_state missing or incomplete"
            )
            return None

        cache = _PrefillReadyRotatingKVCache(max_size=max_size, keep=keep)
        cache.offset = kvcache_offset

        # Set zero-length keys/values so empty() returns False.
        # This prevents mlx-lm from entering Fresh Start mode which
        # would discard all cached KVCache data.
        if kv_shape_ref and HAS_MLX:
            kv_heads, head_dim = kv_shape_ref
            cache.keys = mx.zeros((1, kv_heads, 0, head_dim))
            cache.values = mx.zeros((1, kv_heads, 0, head_dim))
            cache._idx = 0
            logger.debug(
                f"Created empty RotatingKVCache: max_size={max_size}, keep={keep}, "
                f"offset={kvcache_offset}, kv_heads={kv_heads}, head_dim={head_dim}"
            )
        else:
            logger.debug(
                f"Created empty RotatingKVCache: max_size={max_size}, keep={keep} "
                f"(no shape ref, keys=None)"
            )

        return cache

    def _validate_block_cache_data(
        self,
        cache_data: List[Tuple[Any, Any]],
        layer_cache_types: Optional[List[str]] = None,
    ) -> bool:
        """
        Validate that block's cache_data has valid data for all layers.

        A block's cache_data is a list of (keys, values) tuples, one per layer.
        This validates that:
        1. cache_data is not empty
        2. Each layer has non-None keys and values
        3. Each layer has consistent shapes (for sliceable cache types)

        Args:
            cache_data: List of (keys, values) tuples from a block
            layer_cache_types: Optional list of cache type names per layer.
                ArraysCache layers are excluded from seq_len consistency check.

        Returns:
            True if valid, False otherwise
        """
        if not cache_data:
            return False

        # Cache types that don't support block slicing (have different shapes)
        # ArraysCache: generic array cache used by some hybrid models (e.g., Qwen3-Next)
        # RotatingKVCache: uses circular buffer with fixed max_size, cannot be sliced
        # CacheList: composite cache with List[Tuple] format, not (keys, values) tuple
        non_sliceable_types = {'ArraysCache', 'RotatingKVCache', 'CacheList'}

        expected_seq_len = None

        for layer_idx, layer_data in enumerate(cache_data):
            try:
                # Determine cache type first to handle CacheList before tuple unpack
                cache_type = None
                if layer_cache_types and layer_idx < len(layer_cache_types):
                    cache_type = layer_cache_types[layer_idx]

                # CacheList: sub-cache list format, skip standard (keys, values) unpacking
                if cache_type == 'CacheList':
                    # CacheList data is either List[Tuple] (last block) or Tuple (placeholder)
                    if isinstance(layer_data, list):
                        continue  # Sub-cache list — valid
                    # Fall through to standard check for placeholder (zeros tuple)

                keys, values = layer_data

                # Check for None
                if keys is None or values is None:
                    logger.debug(
                        f"Block validation failed: layer {layer_idx} has None keys/values"
                    )
                    return False

                # Skip seq_len check for non-sliceable types (e.g., ArraysCache, RotatingKVCache)
                # This includes placeholder entries (1D tensors from non-last blocks)
                # used by the last-block-only RotatingKVCache storage strategy
                if cache_type in non_sliceable_types:
                    continue

                # Check shape consistency for sliceable types (KVCache, RotatingKVCache)
                if hasattr(keys, 'shape') and len(keys.shape) >= 3:
                    seq_len = keys.shape[2]
                    if expected_seq_len is None:
                        expected_seq_len = seq_len
                    elif seq_len != expected_seq_len:
                        logger.debug(
                            f"Block validation failed: layer {layer_idx} has "
                            f"seq_len {seq_len}, expected {expected_seq_len}"
                        )
                        return False
            except (TypeError, ValueError) as e:
                logger.debug(f"Block validation failed: layer {layer_idx} error: {e}")
                return False

        return True

    def _find_best_prefix_match(
        self,
        tokens: List[int],
    ) -> Optional[Tuple[int, List[int], int]]:
        """Find best matching prefix in the index."""
        best_match = None
        best_len = 0

        parent_hash = b""
        prefix_len = 0
        num_blocks = 0

        for start in range(0, len(tokens), self.block_size):
            end = min(start + self.block_size, len(tokens))
            block_tokens = tokens[start:end]
            if not block_tokens:
                break

            parent_hash = compute_block_hash(
                parent_hash,
                block_tokens,
                model_name=self.paged_cache.model_name,
            )
            prefix_len += len(block_tokens)
            num_blocks += 1

            entry = self._prefix_index.get(parent_hash)
            if entry and entry[0] == prefix_len and prefix_len > best_len:
                best_match = entry
                best_len = prefix_len

        return best_match

    def _update_prefix_index(
        self,
        tokens: List[int],
        block_ids: List[int],
    ) -> None:
        """Update prefix index with new token sequence."""
        # Index prefixes using chain hashes (avoid O(n^2) full-prefix hashing).
        parent_hash = b""
        prefix_len = 0

        for i, block_id in enumerate(block_ids):
            start = i * self.block_size
            end = min(start + self.block_size, len(tokens))
            block_tokens = tokens[start:end]
            if not block_tokens:
                break

            block = self.paged_cache.allocated_blocks.get(block_id)
            block_hash = block.block_hash if block is not None else None
            if block_hash is None:
                block_hash = compute_block_hash(
                    parent_hash,
                    block_tokens,
                    model_name=self.paged_cache.model_name,
                )
                if block is not None:
                    block.block_hash = block_hash

            parent_hash = block_hash
            prefix_len += len(block_tokens)
            self._prefix_index[block_hash] = (prefix_len, block_ids, i + 1)

    def get_stats(self) -> PrefixCacheStats:
        """
        Get cache statistics.

        Returns:
            PrefixCacheStats with cache metrics.
        """
        return PrefixCacheStats(
            hits=self._hits,
            misses=self._misses,
            evictions=self.paged_cache.stats.evictions,
            tokens_saved=self._tokens_saved,
        )

    def get_stats_dict(self) -> Dict[str, Any]:
        """
        Get cache statistics as a dictionary.

        This method provides the legacy dictionary format for compatibility.

        Returns:
            Dictionary with cache statistics.
        """
        paged_stats = self.paged_cache.get_memory_usage()
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0,
            "tokens_saved": self._tokens_saved,
            "active_requests": len(self._request_tables),
            **paged_stats,
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._hits = 0
        self._misses = 0
        self._tokens_saved = 0
        self.paged_cache.reset_stats()

    def clear(self) -> int:
        """
        Clear all cached data.

        Returns:
            Number of entries cleared.
        """
        cleared_count = len(self._request_tables) + len(self._prefix_index)
        self._request_tables.clear()
        self._prefix_index.clear()
        self.paged_cache.clear()
        self.reset_stats()
        return cleared_count

    def set_cold_restore_callback(
        self,
        callback: Optional[Callable[[int, bytes], bool]],
    ) -> None:
        """
        Set callback for restoring cold blocks.

        The callback is invoked when reconstruct_cache() encounters a cold block
        that needs to be restored from paged SSD.

        Args:
            callback: Function with signature (block_id: int, block_hash: bytes) -> bool
                      Returns True if restoration was successful.
        """
        self._cold_restore_callback = callback

    def __len__(self) -> int:
        """Return number of active request entries."""
        return len(self._request_tables)

    # =========================================================================
    # CacheManager ABC Interface Implementation
    # =========================================================================

    def fetch(self, key: Any) -> Tuple[Optional[Any], bool]:
        """
        Fetch cached prefix for a request.

        Args:
            key: Tuple of (request_id: str, tokens: List[int]).

        Returns:
            Tuple of ((block_table, remaining_tokens), True) if found,
            (None, False) otherwise.
        """
        if not isinstance(key, tuple) or len(key) != 2:
            return None, False

        request_id, tokens = key
        if not isinstance(request_id, str) or not isinstance(tokens, list):
            return None, False

        block_table, remaining = self.fetch_cache(request_id, tokens)
        if block_table is not None:
            return (block_table, remaining), True
        return None, False

    def store(self, key: Any, value: Any) -> bool:
        """
        Store cache for a request.

        Args:
            key: Tuple of (request_id: str, tokens: List[int]).
            value: Cache data (List[Any]).

        Returns:
            True if stored successfully.
        """
        if not isinstance(key, tuple) or len(key) != 2:
            return False

        request_id, tokens = key
        if not isinstance(request_id, str) or not isinstance(tokens, list):
            return False

        block_table = self.store_cache(request_id, tokens, value)
        return block_table is not None

    def evict(self, key: Any) -> bool:
        """
        Evict cache for a specific request.

        Args:
            key: Request ID (str) to evict.

        Returns:
            True if evicted, False if not found.
        """
        if not isinstance(key, str):
            return False

        if key in self._request_tables:
            self.release_cache(key)
            return True
        return False

    @property
    def size(self) -> int:
        """
        Get the current number of cached entries.

        Returns:
            Number of active request entries.
        """
        return len(self._request_tables)

    @property
    def max_size(self) -> int:
        """
        Get the maximum cache capacity.

        Returns:
            Maximum number of blocks from the underlying PagedCacheManager.
        """
        return self.paged_cache.max_blocks
