# SPDX-License-Identifier: Apache-2.0
"""
Cache Recovery Manager for oMLX.

This module handles error recovery for cache corruption and other cache-related
failures, enabling the scheduler to continue processing after encountering errors.
"""

import gc
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..exceptions import is_cache_corruption_error

if TYPE_CHECKING:
    from ..request import Request, RequestStatus
    from ..prefix_cache import BlockAwarePrefixCache

logger = logging.getLogger(__name__)


class CacheRecoveryManager:
    """
    Manages cache error recovery for the scheduler.

    This class provides methods to detect and recover from cache corruption,
    including clearing caches and rescheduling affected requests.
    """

    def __init__(
        self,
        block_aware_cache: Optional["BlockAwarePrefixCache"] = None,
    ):
        """
        Initialize the cache recovery manager.

        Args:
            block_aware_cache: Block-aware prefix cache (paged SSD-based).
        """
        self.block_aware_cache = block_aware_cache

    def is_cache_corruption(self, error: Exception) -> bool:
        """
        Check if an error indicates cache corruption.

        Args:
            error: The exception to check.

        Returns:
            True if the error appears to be cache corruption.
        """
        return is_cache_corruption_error(error)

    def recover(
        self,
        batch_generator_holder: Any,
        request_id_to_uid: Dict[str, int],
        uid_to_request_id: Dict[int, str],
        request_detokenizers: Dict[str, Any],
    ) -> None:
        """
        Recover from cache corruption error.

        This method clears the batch generator and all caches, resetting
        the system to a clean state.

        Args:
            batch_generator_holder: Object holding the batch_generator reference.
            request_id_to_uid: Mapping of request IDs to UIDs.
            uid_to_request_id: Mapping of UIDs to request IDs.
            request_detokenizers: Dict of request detokenizers.
        """
        # Clear batch generator (this is the source of the corruption)
        batch_generator_holder.batch_generator = None
        if hasattr(batch_generator_holder, '_current_sampler_params'):
            batch_generator_holder._current_sampler_params = None

        # Clear cache
        if self.block_aware_cache is not None:
            self.block_aware_cache.clear()

        # Clear UID mappings
        request_id_to_uid.clear()
        uid_to_request_id.clear()

        # Clear detokenizer state to prevent contamination after recovery
        request_detokenizers.clear()

        # Force garbage collection
        gc.collect()

        logger.info("Cache recovery completed")

    def reschedule_running_requests(
        self,
        running: Dict[str, "Request"],
        waiting: Any,  # deque[Request]
        request_status_waiting: "RequestStatus",
    ) -> int:
        """
        Move running requests back to waiting queue for retry.

        Args:
            running: Dictionary of running requests by ID.
            waiting: Deque of waiting requests.
            request_status_waiting: The WAITING status enum value.

        Returns:
            Number of requests rescheduled.
        """
        count = len(running)

        for request_id, request in list(running.items()):
            # Reset request state
            request.status = request_status_waiting
            request.batch_uid = None
            request.prompt_cache = None
            request.cached_tokens = 0
            request.prefill_resumed_tokens = 0
            request.remaining_tokens = request.prompt_token_ids

            # Move to waiting queue (at front for priority)
            waiting.appendleft(request)
            del running[request_id]

        if count > 0:
            logger.info(f"Rescheduled {count} requests for retry")

        return count
