# SPDX-License-Identifier: Apache-2.0
"""Mock classes for testing scheduler and engine components.

These mocks replace mlx-lm's BatchGenerator and related cache components
to enable unit testing without loading actual ML models.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set


@dataclass
class MockResponse:
    """Mock BatchGenerator response.

    Simulates the Response object returned by BatchGenerator.next().
    """

    uid: int
    token: int
    finish_reason: Optional[str] = None
    prompt_cache: Optional[Any] = None


class MockBatchGenerator:
    """Mock mlx-lm BatchGenerator for testing.

    Simulates BatchGenerator's insert/next/remove interface without
    requiring actual model inference.
    """

    def __init__(self, model: Any = None, tokenizer: Any = None, **kwargs: Any):
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = kwargs.get("max_tokens", 100)
        self.stop_tokens: Set[int] = set()
        self._queues: Dict[int, deque] = {}
        self._uid_counter = 0
        self._preset_responses: List[List[MockResponse]] = []
        self._response_index = 0

        # Store constructor arguments for testing
        self.prefill_batch_size = kwargs.get("prefill_batch_size", 1)
        self.completion_batch_size = kwargs.get("completion_batch_size", 32)
        self.prefill_step_size = kwargs.get("prefill_step_size", 2048)

    def insert(
        self,
        token_sequences: List[List[int]],
        max_tokens: Optional[List[int]] = None,
        caches: Optional[List[Any]] = None,
        samplers: Optional[List[Callable]] = None,
        logits_processors: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> List[int]:
        """Insert sequences into the batch for generation.

        Args:
            token_sequences: List of token sequences to insert
            max_tokens: Optional max tokens per sequence
            caches: Optional cached KV states
            samplers: Optional per-sequence samplers
            logits_processors: Optional logits processors

        Returns:
            List of UIDs assigned to each sequence
        """
        uids = []
        for i, tokens in enumerate(token_sequences):
            uid = self._uid_counter
            # Default tokens for mock generation
            self._queues[uid] = deque([100, 101, 102])
            self._uid_counter += 1
            uids.append(uid)
        return uids

    def next(self) -> List[MockResponse]:
        """Generate the next token for all active sequences.

        Returns:
            List of MockResponse objects for each active sequence
        """
        # Use preset responses if available
        if self._preset_responses and self._response_index < len(self._preset_responses):
            responses = self._preset_responses[self._response_index]
            self._response_index += 1
            return responses

        # Generate default responses from queues
        responses = []
        for uid, queue in list(self._queues.items()):
            if queue:
                token = queue.popleft()
                finish = "stop" if not queue else None
                responses.append(MockResponse(uid=uid, token=token, finish_reason=finish))
                if not queue:
                    del self._queues[uid]
        return responses

    def remove(self, uids: List[int]) -> None:
        """Remove sequences from the batch.

        Args:
            uids: List of UIDs to remove
        """
        for uid in uids:
            self._queues.pop(uid, None)

    def set_responses(self, responses: List[List[MockResponse]]) -> None:
        """Set preset responses for testing.

        Args:
            responses: List of response batches to return in order
        """
        self._preset_responses = responses
        self._response_index = 0


@dataclass
class MockBlockTable:
    """Mock BlockTable for paged cache testing."""

    block_ids: List[int] = field(default_factory=list)
    num_tokens: int = 0


class MockPagedCacheManager:
    """Mock PagedCacheManager for testing.

    Simulates block-based KV cache management without actual memory operations.
    """

    def __init__(self, **kwargs: Any):
        self.blocks: List[Any] = []
        self.request_tables: Dict[str, MockBlockTable] = {}
        self.block_size = kwargs.get("block_size", 256)
        self.max_blocks = kwargs.get("max_blocks", 1000)
        self.cold_block_count = 0

    def get_block_table(self, request_id: str) -> Optional[MockBlockTable]:
        """Get the block table for a request.

        Args:
            request_id: The request ID

        Returns:
            BlockTable if exists, None otherwise
        """
        return self.request_tables.get(request_id)

    def release_for_eviction(self, block_ids: List[int]) -> int:
        """Release blocks for eviction.

        Args:
            block_ids: List of block IDs to release

        Returns:
            Number of blocks released
        """
        return len(block_ids)

    def delete_block_table(self, request_id: str) -> None:
        """Delete a request's block table.

        Args:
            request_id: The request ID
        """
        self.request_tables.pop(request_id, None)

    def get_evictable_blocks(self, count: int) -> List[Any]:
        """Get evictable blocks in LRU order.

        Args:
            count: Number of blocks to get

        Returns:
            List of evictable blocks
        """
        return []

    def evict_block_permanently(self, block_id: int) -> bool:
        """Evict a block permanently.

        Args:
            block_id: The block ID to evict

        Returns:
            True if evicted successfully
        """
        return True

    def set_paged_ssd_cache_manager(self, manager: Any) -> None:
        """Set the SSD cache manager.

        Args:
            manager: The SSD cache manager
        """
        pass


class MockBlockAwarePrefixCache:
    """Mock BlockAwarePrefixCache for testing.

    Simulates prefix caching without actual cache storage.
    """

    def __init__(self, **kwargs: Any):
        self._entries: Dict[str, Any] = {}
        self._cold_restore_callback: Optional[Callable] = None

    def fetch_cache(
        self, request_id: str, tokens: List[int]
    ) -> tuple[Optional[MockBlockTable], List[int]]:
        """Fetch cached prefix for a request.

        Args:
            request_id: The request ID
            tokens: The token sequence

        Returns:
            Tuple of (block_table, remaining_tokens)
        """
        return None, tokens

    def reconstruct_cache(self, block_table: MockBlockTable) -> Optional[Any]:
        """Reconstruct cache from block table.

        Args:
            block_table: The block table

        Returns:
            Reconstructed cache or None
        """
        return None

    def store_cache(
        self,
        request_id: str,
        tokens: List[int],
        cache: Any,
        model_cache_config: Any = None,
        **kwargs: Any,
    ) -> Optional[MockBlockTable]:
        """Store cache for a request.

        Args:
            request_id: The request ID
            tokens: The token sequence
            cache: The cache to store
            model_cache_config: Optional model cache configuration

        Returns:
            BlockTable if stored successfully, None otherwise
        """
        return None

    def clear_request_entry(self, request_id: str) -> None:
        """Clear cache entry for a request.

        Args:
            request_id: The request ID
        """
        self._entries.pop(request_id, None)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._entries.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary of cache statistics
        """
        return {"entries": len(self._entries)}

    def set_cold_restore_callback(self, callback: Callable) -> None:
        """Set callback for restoring blocks from cold storage.

        Args:
            callback: The callback function
        """
        self._cold_restore_callback = callback

    def set_paged_ssd_cache_manager(self, manager: Any) -> None:
        """Set the SSD cache manager.

        Args:
            manager: The SSD cache manager
        """
        pass


class MockSSDCacheManager:
    """Mock SSD cache manager for testing."""

    def __init__(self, **kwargs: Any):
        self._blocks: Dict[bytes, Any] = {}

    def has_block(self, block_hash: bytes) -> bool:
        """Check if a block exists in SSD cache.

        Args:
            block_hash: The block hash

        Returns:
            True if block exists
        """
        return block_hash in self._blocks

    def get_stats(self) -> Dict[str, Any]:
        """Get SSD cache statistics.

        Returns:
            Dictionary of cache statistics
        """
        return {"blocks": len(self._blocks)}


class MockMemoryMonitor:
    """Mock memory monitor for testing."""

    def __init__(self, **kwargs: Any):
        self._model_info: Dict[str, Any] = {}

    def set_model_info(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype_size: int = 2,
    ) -> None:
        """Set model info for memory estimation.

        Args:
            num_layers: Number of transformer layers
            num_kv_heads: Number of KV heads
            head_dim: Head dimension
            dtype_size: Size of dtype in bytes
        """
        self._model_info = {
            "num_layers": num_layers,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "dtype_size": dtype_size,
        }

    def estimate_blocks_to_free(self, bytes_to_free: int, block_size: int) -> int:
        """Estimate number of blocks to free.

        Args:
            bytes_to_free: Target bytes to free
            block_size: Size of each block in tokens

        Returns:
            Estimated number of blocks to evict
        """
        return 1

    def estimate_block_memory(self, block_size: int) -> int:
        """Estimate memory usage of a block.

        Args:
            block_size: Block size in tokens

        Returns:
            Estimated memory in bytes
        """
        return block_size * 1024  # Simple estimation
