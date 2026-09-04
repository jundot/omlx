# SPDX-License-Identifier: Apache-2.0
"""
Block-Aware Prefix Cache for oMLX.

Provides prefix caching using PagedCacheManager for block-based storage
with SSD persistence. oMLX only supports paged SSD-based caching.
"""

import logging
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from ._rotating_subclass import PrefillReadyRotatingKVCache
from .hybrid_cache import ModelCacheConfig
from .interface import CacheManager
from .paged_cache import (
    BlockHash,
    BlockTable,
    CacheBlock,
    PagedCacheManager,
    compute_block_hash,
    resolve_block_extra_keys,
)
from .paged_ssd_cache import _PM_SLICEABLE_SUB_CLASSES, PagedSSDCacheManager
from .pooling_delta import POOLING_CACHE_DELTA_CLASS
from .stats import PrefixCacheStats
from .type_registry import CacheTypeRegistry

# Non-sliceable CacheList member classes safe for per-member block storage.
# PoolingCache (delta chains) and rotating families keep the legacy
# cumulative path; ArraysCache state is small, positionless, and
# self-contained at every boundary.
_PM_SAFE_NON_SLICEABLE_SUBS = frozenset({"ArraysCache", "SizedArraysCache"})


def cachelist_pm_member_plan(
    sub_class_names: list[str],
    sub_states: list[Any],
) -> list[str] | None:
    """Per-member storage plan for a mixed CacheList layer, or None.

    Returns a list aligned with the sub-caches: ``"slice"`` for 4D
    sliceable KV members (stored as true per-block slices), ``"boundary"``
    for ArraysCache-style members (stored from the block's boundary
    snapshot). ``None`` means the layer is not eligible for per-member
    storage (missing class info, unknown / PoolingCache / rotating
    members, or nothing to slice) and must use the legacy cumulative
    last-block-only path.

    Without this split, a mixed CacheList (e.g. inkling's
    ``CacheList(KVCache, ArraysCache)``) stores the FULL cumulative state
    of every member — including the giant KV prefix — in every block:
    quadratic in context length on both the allocator pool and the SSD.
    """
    if not sub_class_names or len(sub_class_names) != len(sub_states):
        return None

    plan: list[str] = []
    for class_name, sub_state in zip(sub_class_names, sub_states):
        shape_sliceable = (
            isinstance(sub_state, (list, tuple))
            and len(sub_state) >= 2
            and hasattr(sub_state[0], "shape")
            and len(sub_state[0].shape) == 4
        )
        if class_name in _PM_SLICEABLE_SUB_CLASSES and shape_sliceable:
            plan.append("slice")
        elif class_name in _PM_SAFE_NON_SLICEABLE_SUBS:
            plan.append("boundary")
        else:
            return None

    if "slice" not in plan or "boundary" not in plan:
        return None

    return plan

logger = logging.getLogger(__name__)

# Cap on the supersede-on-extend lineage map (tip hash -> previous tip hash).
# Each entry is two 32-byte hashes; the cap only guards against unbounded
# growth from many distinct conversation chains over a long-lived process.
_TIP_LINEAGE_MAX_ENTRIES = 4096

# Cap for the per-session set of dedup'd block hashes whose payloads were
# already inspected for boundary-snapshot backfill (see
# _maybe_backfill_dedup_block_payload). Cleared on overflow, like the tip
# lineage map.
_BACKFILL_CHECKED_MAX_ENTRIES = 4096

# Lightning-MTP prompt priming has one small attention cache of its own.  The
# normal prefix cache stores only the backbone cache, so a warm trunk restore
# used to lose the MTP history and restart speculative decode unprimed.  Keep a
# deliberately tiny in-memory sidecar of full-block boundary snapshots.  Four
# entries bounds the worst-case Qwen4 100K-context footprint to roughly the
# same order as one ordinary request cache while covering the recent branches
# of an interactive conversation.  The entries are keyed by the *same* chain
# hash as the backbone block and are dropped with that block/hash lifecycle.
_MTP_PREFIX_SNAPSHOT_MAX_ENTRIES = 4


def _wrap_cachelist_sub_marker(
    sub_idx: int,
    elements: list[Any],
    sub_class_names: list[str],
    storage_class_name: str | None = None,
) -> tuple[Any, ...]:
    """Wrap one CacheList sub-state for block storage.

    Length-2 element lists round-trip as legacy ``(keys, values)`` so
    existing callers (prefix cache reconstruct, tests) keep their shape.
    Real N-tuple sub-states (PoolingCache, BatchKVCache) surface as
    ``__nstate__`` markers.
    """
    if len(elements) == 2:
        return (elements[0], elements[1])
    name = storage_class_name
    if name is None and sub_idx < len(sub_class_names):
        name = sub_class_names[sub_idx]
    return ("__nstate__", name, list(elements))
_EXACT_PREFIX_TERMINAL_KEY = "specprefill-static-exact-v1"
_POOLING_CACHE_SUB_CLASSES = frozenset({"PoolingCache", "BatchPoolingCache"})


def _contains_pooling_cache_state(cache_data: list[Any]) -> bool:
    """Return whether extracted cache data contains a pooling CacheList sub-cache."""
    for layer_state in cache_data:
        if not isinstance(layer_state, dict):
            continue
        if (
            layer_state.get("cache_type") != "CacheList"
            and layer_state.get("class_name") != "CacheList"
        ):
            continue
        sub_class_names = layer_state.get("sub_class_names") or []
        if not sub_class_names:
            meta_state = layer_state.get("meta_state")
            if (
                isinstance(meta_state, (list, tuple))
                and meta_state
                and isinstance(meta_state[0], (list, tuple))
            ):
                sub_class_names = meta_state[0]
        if any(str(name) in _POOLING_CACHE_SUB_CLASSES for name in sub_class_names):
            return True
    return False


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
        paged_ssd_cache_manager: PagedSSDCacheManager | None = None,
        gdn_ssd_split_enabled: bool = False,
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
        self._gdn_ssd_split_enabled = bool(gdn_ssd_split_enabled)
        self._gdn_split_fallback_reported = False
        self._gdn_checkpoint_loader: Callable[[Any], list[dict[str, Any]] | None] | None = None

        # Expected number of layers for cache validation
        self.expected_num_layers = self._get_model_num_layers(model)

        # Hash table for quick prefix lookup
        # Maps chain-hash(prefix) -> (prefix_len, block_ids, num_blocks)
        self._prefix_index: dict[bytes, tuple[int, tuple[int, ...], int]] = {}

        # Tie the index lifecycle to the paged cache's hash associations.
        # Without these hooks the index only ever grew (entries were dropped
        # solely by clear()), leaking Python heap for the process lifetime
        # on long-uptime servers.
        paged_cache_manager.on_block_hash_dropped = self._on_block_hash_dropped
        paged_cache_manager.on_hash_map_cleared = self._on_hash_map_cleared

        # Request to block table mapping
        self._request_tables: dict[str, BlockCacheEntry] = {}

        # Supersede-on-extend lineage for rotating (sliding-window) models:
        # newest tip block hash -> previous tip block hash. When a chain is
        # extended again, the entry two generations back is stripped of its
        # rotating payload (see _strip_rotating_payload); the immediate
        # previous tip is kept intact as the walk-back fallback.
        self._rotating_tip_lineage: dict[bytes, bytes] = {}

        # Hashes this session stored as tip blocks. A lineage entry is only
        # recorded when the block preceding the new blocks really was a tip:
        # for a store that diverged from an existing chain, that block is a
        # shared-prefix interior block (often a boundary-snapshot block) and
        # stripping it would destroy partial-match walk-back restores.
        self._store_tip_hashes: set[bytes] = set()

        # Dedup'd block hashes whose payloads were conclusively inspected
        # for boundary-snapshot backfill this session (real state found or
        # rewrite saved). Each hash is inspected at most once per run.
        self._backfill_checked_hashes: set[bytes] = set()

        # Full-block Lightning-MTP prompt-history snapshots.  Values are
        # opaque to the generic prefix cache; prompt_priming owns their shape.
        # Access can race with the asynchronous backbone store worker's hash
        # callbacks, so use a private lock rather than relying on the GIL for
        # the multi-step LRU operations.
        self._mtp_prefix_snapshots: OrderedDict[bytes, tuple[int, Any]] = (
            OrderedDict()
        )
        self._mtp_prefix_snapshot_lock = threading.RLock()

        # Callback for restoring cold blocks (deprecated in paged SSD-only mode)
        # Kept for API compatibility
        self._cold_restore_callback: Callable[[int, bytes], bool] | None = None

        # Statistics
        self._hits = 0
        self._misses = 0
        self._tokens_saved = 0
        self._partial_block_skips = 0
        self._partial_tokens_skipped = 0
        self._tokens_matched_total = 0
        self._tokens_requested_total = 0
        self._last_partial_tokens_skipped = 0
        self._last_tokens_to_next_block = 0
        self._exact_prefix_hits = 0
        self._exact_prefix_misses = 0
        self._exact_prefix_tokens_restored = 0
        self._exact_prefix_stores = 0
        self._exact_prefix_store_failures = 0
        self._gdn_checkpoint_loads = 0
        self._gdn_checkpoint_walkbacks = 0
        self._last_gdn_restore: dict[str, Any] | None = None
        self._gdn_dequantization_counter: Callable[[], int] | None = None

    def set_gdn_checkpoint_loader(
        self,
        loader: Callable[[Any], list[dict[str, Any]] | None] | None,
        dequantization_counter: Callable[[], int] | None = None,
    ) -> None:
        """Attach the scheduler-owned recurrent checkpoint deserializer."""
        self._gdn_checkpoint_loader = loader
        self._gdn_dequantization_counter = dequantization_counter

    def _gdn_dequantization_count(self) -> int | None:
        """Read the store counter without making restore depend on metrics."""
        if self._gdn_dequantization_counter is None:
            return None
        try:
            return int(self._gdn_dequantization_counter())
        except Exception:
            logger.debug("Failed to read GDN dequantization counter", exc_info=True)
            return None

    def _gdn_split_layout_supported(
        self, layer_cache_types: list[str] | tuple[str, ...] | None
    ) -> bool:
        """Return whether the current layout is safe for external GDN state.

        V1 intentionally supports only top-level ArraysCache mixed with
        block-sliceable attention caches. Composite CacheList, pooling, and
        rotating families stay on the legacy embedded-snapshot path.
        """
        if not self._gdn_ssd_split_enabled or not layer_cache_types:
            return False
        saw_arrays = any(
            CacheTypeRegistry.is_arrays_family(type_name)
            for type_name in layer_cache_types
        )
        for type_name in layer_cache_types:
            if CacheTypeRegistry.is_arrays_family(type_name):
                continue
            if type_name == "CacheList" or CacheTypeRegistry.is_rotating_family(type_name):
                if saw_arrays and not self._gdn_split_fallback_reported:
                    logger.info(
                        "GDN SSD sidecar is unsupported for layout %s; "
                        "falling back to embedded GDN snapshots",
                        list(layer_cache_types),
                    )
                    self._gdn_split_fallback_reported = True
                return False
            handler = CacheTypeRegistry.get_handler_by_class_name(type_name)
            if not handler.supports_block_slicing:
                if saw_arrays and not self._gdn_split_fallback_reported:
                    logger.info(
                        "GDN SSD sidecar is unsupported for layout %s; "
                        "falling back to embedded GDN snapshots",
                        list(layer_cache_types),
                    )
                    self._gdn_split_fallback_reported = True
                return False
        return saw_arrays

    @staticmethod
    def _validated_gdn_snapshot_layers(
        snapshot: Any,
        layer_cache_types: list[str] | tuple[str, ...] | None,
    ) -> dict[int, tuple[Any, Any]] | None:
        """Return validated Arrays-family payloads, or reject the snapshot."""
        if not isinstance(snapshot, (list, tuple)):
            return None

        payloads: dict[int, tuple[Any, Any]] = {}
        for layer_idx, type_name in enumerate(layer_cache_types or []):
            if not CacheTypeRegistry.is_arrays_family(type_name):
                continue
            if layer_idx >= len(snapshot):
                return None
            layer_state = snapshot[layer_idx]
            if not isinstance(layer_state, dict):
                return None
            state = layer_state.get("state", ())
            if not isinstance(state, (list, tuple)) or len(state) < 2:
                return None
            if len(state) == 2:
                cache_data = (state[0], state[1])
            else:
                cache_data = ("__nstate__", type_name, list(state))
            payloads[layer_idx] = (cache_data, layer_state.get("meta_state", ()))
        return payloads

    def _has_split_gdn_checkpoint(
        self,
        block_hash: bytes,
        layer_cache_types: list[str] | tuple[str, ...] | None,
    ) -> bool:
        """Return whether a recurrent sidecar exists for this exact layout."""
        manager = self.paged_ssd_cache
        signature_builder = getattr(manager, "gdn_cache_signature_for", None)
        if not callable(signature_builder):
            signature_builder = getattr(manager, "cache_signature_for", None)
        checkpoint_checker = getattr(manager, "has_gdn_checkpoint", None)
        if not callable(signature_builder) or not callable(checkpoint_checker):
            return False
        try:
            cache_signature = signature_builder(
                model_name=self.paged_cache.model_name,
                num_layers=len(layer_cache_types or []),
                block_size=self.block_size,
                layer_cache_types=layer_cache_types or [],
            )
            return bool(checkpoint_checker(block_hash, cache_signature))
        except Exception:
            logger.exception("Failed to query split-GDN checkpoint availability")
            return False

    def _commit_split_gdn_checkpoint(
        self,
        boundary_snapshots: Any,
        token_count: int,
        block_hash: bytes,
        layer_cache_types: list[str] | tuple[str, ...] | None,
        layer_meta_states: list[Any] | None,
    ) -> bool:
        """Commit one staged recurrent checkpoint through its provider."""
        commit_checkpoint = getattr(
            boundary_snapshots, "commit_gdn_checkpoint", None
        )
        if not callable(commit_checkpoint):
            return False
        try:
            return bool(
                commit_checkpoint(
                    token_count,
                    block_hash,
                    layer_cache_types=layer_cache_types,
                    layer_meta_states=layer_meta_states,
                    model_name=self.paged_cache.model_name,
                    block_size=self.block_size,
                )
            )
        except Exception:
            logger.exception(
                "Failed to commit split-GDN checkpoint for block hash %s",
                block_hash.hex()[:16],
            )
            return False

    def _get_model_num_layers(self, model: Any) -> int:
        """
        Get the expected number of *cache layers* for validation.

        For hybrid models, the number of cache entries (from ``make_cache()``)
        may be smaller than the architectural layer count (``model.layers``),
        because some layer types do not produce cache state.

        Args:
            model: The MLX model

        Returns:
            Number of cache layers, or 0 if cannot be determined
        """
        # Prefer cache-layer count when available (hybrid-model safe).
        make_cache = getattr(model, "make_cache", None)
        if callable(make_cache):
            try:
                cache_list = make_cache()
                if isinstance(cache_list, list) and len(cache_list) > 0:
                    return len(cache_list)
            except Exception as e:
                logger.debug(
                    f"Could not determine cache layer count via make_cache(): {e}"
                )

        # Fallback to architectural layer count for non-hybrid models.
        if hasattr(model, "layers"):
            return len(model.layers)
        if hasattr(model, "args") and hasattr(model.args, "num_hidden_layers"):
            return model.args.num_hidden_layers
        if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
            return model.config.num_hidden_layers

        # Cannot determine, return 0 to skip validation
        logger.debug(
            "Cannot determine model/cache num_layers, cache layer validation disabled"
        )
        return 0

    def set_paged_ssd_cache_manager(
        self, paged_ssd_cache_manager: PagedSSDCacheManager | None
    ) -> None:
        """
        Set the PagedSSDCacheManager for SSD storage.

        This allows setting the SSD cache after initialization,
        which is useful when the scheduler creates it later.

        Args:
            paged_ssd_cache_manager: The PagedSSDCacheManager instance.
        """
        self.paged_ssd_cache = paged_ssd_cache_manager
        if paged_ssd_cache_manager is not None:
            # If the manager already knows its layer-cache signature (e.g.,
            # eager scheduler plumb-through), purge any indexed blocks left
            # over from a prior cache-config run for the current model.
            # When the signature is unset, this is a no-op; the manager
            # adopts on the first save and sweeps then.
            try:
                paged_ssd_cache_manager.invalidate_stale_layer_signature()
            except Exception as e:
                logger.warning("Stale-signature sweep on manager attach failed: %s", e)
            logger.info("PagedSSDCacheManager connected to BlockAwarePrefixCache")

    def _forget_incompatible_ssd_block(
        self,
        block_hash: bytes | None,
        block_id: int | None = None,
    ) -> None:
        """Remove an incompatible block from this manager without unlinking it.

        The SSD cache directory can be shared by multiple loaded models. A
        block that is stale for this prefix cache may still be valid for
        another model, so mismatch handling must clear local indexes only.
        """
        if block_hash is None:
            return
        try:
            if block_id is None:
                block = self.paged_cache.cached_block_hash_to_block.get_block(
                    block_hash
                )
                block_id = block.block_id if block is not None else None
            if block_id is not None:
                self.paged_cache.cached_block_hash_to_block.pop(block_hash, block_id)
                # Keep the prefix index in step with the hash map, or the
                # index keeps retrying this dead chain until the block is
                # freed (same last-block-standing rule as the paged hooks).
                self.paged_cache._notify_hash_dropped(block_hash)
        except Exception as e:
            logger.debug(f"Failed to forget incompatible paged block: {e}")
        if self.paged_ssd_cache is None:
            return
        try:
            self.paged_ssd_cache.forget_block(block_hash)
        except Exception as e:
            logger.debug(f"Failed to forget incompatible SSD block: {e}")

    @staticmethod
    def _canonical_layer_cache_types(
        layer_cache_types: list[str] | tuple[str, ...] | None,
    ) -> list[str] | None:
        """Normalize wrapper class names for metadata compatibility checks.

        Thin wrapper around the canonical implementation in
        :mod:`omlx.cache.paged_ssd_cache` so the two sites that compare
        signatures stay in lock-step.
        """
        from .paged_ssd_cache import _canonicalize_layer_cache_types

        return _canonicalize_layer_cache_types(layer_cache_types)

    def _detect_window_padding_from_blocks(
        self,
        block_ids: list[int],
    ) -> ModelCacheConfig | None:
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

        _, metadata = self.paged_ssd_cache.load_block_with_metadata(
            first_block.block_hash
        )
        if not metadata:
            return None

        layer_cache_types = metadata.get("layer_cache_types")
        # Note: CacheList layers containing RotatingKVCache sub-caches do NOT need
        # window padding. CacheList uses last-block-only storage with reject-on-partial
        # strategy, so the sliding window state is either fully restored (exact match)
        # or the entire cache is rejected (partial match).
        if not layer_cache_types or not any(
            CacheTypeRegistry.is_rotating_family(t) for t in layer_cache_types
        ):
            return None

        model_cache_config = ModelCacheConfig.from_type_list(
            layer_cache_types, model_name=""
        )

        # Extract window_size from layer meta_states
        layer_meta_states = metadata.get("layer_meta_states", [])
        max_window_size = 0
        for idx, meta in enumerate(layer_meta_states):
            if not meta or len(meta) < 2:
                continue
            # Check if this layer is RotatingKVCache
            if idx < len(layer_cache_types) and CacheTypeRegistry.is_rotating_family(
                layer_cache_types[idx]
            ):
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
        tokens: list[int],
        extra_keys: tuple[Any, ...] | None = None,
        extra_key_token_start: int | None = None,
        extra_key_ranges: list[tuple[int, tuple[Any, ...]]] | None = None,
    ) -> tuple[BlockTable | None, list[int]]:
        """
        Find cached prefix blocks for the given tokens.

        Args:
            request_id: Unique request identifier
            tokens: Input token sequence
            extra_keys: Additional keys for hash (e.g., VLM image hash)

        Returns:
            Tuple of (block_table, remaining_tokens)
            - block_table: BlockTable if prefix found, None otherwise
            - remaining_tokens: Tokens that need processing
        """
        if not tokens:
            return None, tokens

        # Try to find shared prefix blocks
        shared_block_ids, shared_block_hashes, _ = self.paged_cache.find_shared_prefix(
            tokens,
            extra_keys=extra_keys,
            extra_key_token_start=extra_key_token_start,
            extra_key_ranges=extra_key_ranges,
        )

        if shared_block_ids:
            # Create block table for this request with shared blocks
            block_table = self.paged_cache.create_block_table(request_id)

            for block_id, expected_hash in zip(
                shared_block_ids, shared_block_hashes
            ):
                # Acquire atomically under the paged-cache lock, re-checking
                # the hash this block still holds -- closes the TOCTOU
                # window between find_shared_prefix's lookup (outside any
                # lock the caller holds) and this reference, where a
                # concurrent eviction + reallocation could otherwise splice
                # a foreign block's content into the returned chain
                # (docs/qwen35-hardening-and-optimization.md A2). Stop at
                # the first mismatch rather than skipping it: skip-and-
                # continue would leave a hole mid-chain while still
                # reporting the full prefix length.
                block = self.paged_cache.acquire_cached_block(
                    block_id, expected_hash
                )
                if block is None:
                    break
                block_table.block_ids.append(block.block_id)
                block_table.num_tokens += block.token_count

            if block_table.block_ids:
                num_prefix_tokens = block_table.num_tokens
                remaining = tokens[num_prefix_tokens:]
                self._hits += 1
                self._tokens_saved += num_prefix_tokens
                self._tokens_matched_total += num_prefix_tokens
                self._tokens_requested_total += len(tokens)

                logger.debug(
                    f"Cache hit for {request_id}: "
                    f"{len(block_table.block_ids)} blocks, {num_prefix_tokens} tokens"
                )

                return block_table, remaining

            # Every shared block was already gone or reassigned by the time
            # we tried to acquire it -- discard the empty table and fall
            # through to the prefix-index path below instead of returning
            # a hollow hit.
            self.paged_cache.delete_block_table(request_id)

        # Try prefix index for longer matches
        best_match = self._find_best_prefix_match(tokens, extra_keys=extra_keys)
        if best_match:
            prefix_len, matched_block_ids, num_blocks, chain_hashes = best_match

            # Re-validate the stored chain before reuse: index entries can
            # outlive their blocks (eviction, reuse for other content), and
            # handing out a reassigned block would splice foreign KV into
            # this request. Acquire block by block and stop at the first
            # mismatch -- the validated prefix up to that point is still
            # correct. The old code skipped missing blocks but kept the full
            # prefix_len, leaving holes in the block table.
            acquired: list[CacheBlock] = []
            for block_id, expected_hash in zip(
                matched_block_ids[:num_blocks], chain_hashes
            ):
                block = self.paged_cache.acquire_cached_block(block_id, expected_hash)
                if block is None:
                    # Chain broken: self-heal the stale index entry so the
                    # next lookup does not walk the same dead chain.
                    self._prefix_index.pop(chain_hashes[num_blocks - 1], None)
                    break
                acquired.append(block)

            if acquired:
                block_table = self.paged_cache.create_block_table(request_id)
                for block in acquired:
                    block_table.block_ids.append(block.block_id)
                    block_table.num_tokens += block.token_count

                matched_tokens = block_table.num_tokens
                remaining = tokens[matched_tokens:]
                self._hits += 1
                self._tokens_saved += matched_tokens
                self._tokens_matched_total += matched_tokens
                self._tokens_requested_total += len(tokens)

                logger.debug(
                    f"Prefix index hit for {request_id}: "
                    f"{matched_tokens} tokens matched"
                )

                return block_table, remaining

        # No cache hit
        self._misses += 1
        self._tokens_requested_total += len(tokens)
        logger.debug(f"Cache miss for {request_id}")
        return None, tokens

    def store_cache(
        self,
        request_id: str,
        tokens: list[int],
        cache_data: list[Any],
        model_cache_config: ModelCacheConfig | None = None,
        boundary_snapshots: dict[int, list[Any]] | None = None,
        extra_keys: tuple[Any, ...] | None = None,
        extra_key_token_start: int | None = None,
        extra_key_ranges: list[tuple[int, tuple[Any, ...]]] | None = None,
        hot_cache_write_back: bool = True,
        _store_exact_terminal: bool = False,
    ) -> BlockTable | None:
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
            boundary_snapshots: Optional mapping of token_count -> extracted cache
                states for intermediate block boundaries. Used to store per-block
                ArraysCache state instead of placeholders in hybrid models.
            hot_cache_write_back: When False, SSD-backed hot cache is bypassed
                for newly stored dirty blocks.
            _store_exact_terminal: Internal exact-prefix mode that persists the
                trailing partial block and isolates the terminal hash from
                ordinary prefix matching.

        Returns:
            BlockTable for the stored cache, or None on failure
        """
        if not tokens:
            return None

        # Check if cache_data contains extracted tensor states
        is_tensor_data = (
            cache_data
            and isinstance(cache_data, list)
            and len(cache_data) > 0
            and isinstance(cache_data[0], dict)
            and "state" in cache_data[0]
        )

        # Extract cache type information for SSD storage
        layer_cache_types = None
        layer_meta_states = None
        if model_cache_config:
            layer_cache_types = model_cache_config.get_type_names()
            # Extract meta_states if available in cache_data
            layer_meta_states = [
                cache_data[i].get("meta_state", ()) if i < len(cache_data) else ()
                for i in range(model_cache_config.num_layers)
            ]
        elif is_tensor_data:
            # Try to extract type info from cache_data itself
            layer_cache_types = [
                # Prefer class_name for TurboQuant (cache_type maps to 'KVCache'),
                # fall back to cache_type for all standard mlx-lm types.
                (
                    layer_state.get(
                        "class_name", layer_state.get("cache_type", "KVCache")
                    )
                    if layer_state.get("class_name", "")
                    in ("TurboQuantKVCache", "BatchTurboQuantKVCache")
                    else layer_state.get("cache_type", "KVCache")
                )
                for layer_state in cache_data
            ]
            layer_meta_states = [
                layer_state.get("meta_state", ()) for layer_state in cache_data
            ]

        split_gdn_layout = self._gdn_split_layout_supported(layer_cache_types)

        # Get or create block table
        block_table = self.paged_cache.get_block_table(request_id)
        if not block_table:
            block_table = self.paged_cache.create_block_table(request_id)

        # Determine tokens we need to cache (not already in block_table)
        existing_tokens = block_table.num_tokens
        new_tokens = tokens[existing_tokens:]

        if not new_tokens:
            # All tokens already cached
            self._last_partial_tokens_skipped = 0
            self._last_tokens_to_next_block = 0
            return block_table

        # Allocate only full blocks (skip partial trailing block).
        # get_computed_blocks() matches full blocks only (floor division),
        # so partial block data is never used during cache lookup.
        # Skipping partial blocks also ensures is_last_block points to
        # the last full block, which is critical for non-sliceable caches
        # (ArraysCache/RotatingKVCache) that use last-block-only storage.
        num_new_blocks = (
            math.ceil(len(new_tokens) / self.block_size)
            if _store_exact_terminal
            else len(new_tokens) // self.block_size
        )
        trailing_partial_tokens = len(new_tokens) % self.block_size
        self._last_partial_tokens_skipped = (
            0 if _store_exact_terminal else trailing_partial_tokens
        )
        self._last_tokens_to_next_block = (
            self.block_size - trailing_partial_tokens
            if trailing_partial_tokens > 0 and not _store_exact_terminal
            else 0
        )
        if trailing_partial_tokens > 0 and not _store_exact_terminal:
            self._partial_block_skips += 1
            self._partial_tokens_skipped += trailing_partial_tokens
            logger.debug(
                "Skipping trailing partial block for %s: %s token(s) not persisted "
                "(block_size=%s, needs +%s token(s) to fill next block)",
                request_id,
                trailing_partial_tokens,
                self.block_size,
                self._last_tokens_to_next_block,
            )

        blocks_saved_to_ssd = 0
        # Per-member CacheList layers need a boundary snapshot for every
        # non-last block. A missing middle snapshot must TRUNCATE the store
        # at that boundary: falling through to a placeholder produces a
        # per-member/placeholder/per-member chain whose reconstruction
        # silently skips the placeholder block's KV tokens (restored KV
        # shorter than block_table.num_tokens -> scheduler skips tokens).
        pm_layers_present = False
        if is_tensor_data:
            for layer_state in cache_data:
                if not isinstance(layer_state, dict):
                    continue
                type_name = str(
                    layer_state.get("class_name")
                    or layer_state.get("cache_type")
                    or ""
                )
                if type_name != "CacheList":
                    continue
                state_list = layer_state.get("state")
                if isinstance(state_list, list) and any(
                    isinstance(ss, (list, tuple)) and len(ss) == 0
                    for ss in state_list
                ):
                    # A CacheList store source with a blanked member has no
                    # storable state for that sub (a member-filtered snapshot
                    # promoted without refill). The legacy branch would
                    # silently drop the sub, and the short-payload blocks
                    # then poison this prefix through token-hash dedup, so
                    # refuse the whole store.
                    logger.warning(
                        "store_cache aborted for %s: CacheList layer has a "
                        "blanked member state; refusing to store a partial "
                        "composite",
                        request_id,
                    )
                    return None
                if pm_layers_present:
                    continue
                names = layer_state.get("sub_class_names") or []
                if not names:
                    meta = layer_state.get("meta_state")
                    if (
                        isinstance(meta, (list, tuple))
                        and len(meta) >= 1
                        and isinstance(meta[0], (list, tuple))
                    ):
                        names = [str(n) for n in meta[0]]
                if (
                    cachelist_pm_member_plan(
                        list(names), layer_state.get("state") or []
                    )
                    is not None
                ):
                    pm_layers_present = True
        require_contiguous_pooling_snapshots = bool(
            boundary_snapshots
        ) and _contains_pooling_cache_state(cache_data)
        # Supersede-on-extend tracking (rotating models only, see below).
        first_new_block_idx: int | None = None
        tip_block_saved = False

        for i in range(num_new_blocks):
            start_idx = i * self.block_size
            end_idx = min(start_idx + self.block_size, len(new_tokens))
            block_tokens = new_tokens[start_idx:end_idx]

            # Token range in the original sequence (accounting for existing tokens)
            global_start = existing_tokens + start_idx
            global_end = existing_tokens + end_idx
            is_last_block = i == num_new_blocks - 1
            has_boundary_snapshot = (
                boundary_snapshots is not None and global_end in boundary_snapshots
            )

            if pm_layers_present and not is_last_block and not has_boundary_snapshot:
                logger.info(
                    "store_cache truncated for %s at block %d/%d: missing "
                    "boundary snapshot at token %d for a per-member CacheList "
                    "layer (stored prefix stays valid; remaining blocks are "
                    "not persisted)",
                    request_id,
                    i,
                    num_new_blocks,
                    global_end,
                )
                break

            # PoolingCache boundary snapshots form an absolute-range delta
            # chain. Publishing a placeholder for a missing middle boundary
            # makes every later block unreconstructable and surfaces as a
            # CacheList signature mismatch on the next restore. Keep the valid
            # prefix only; the MTP boundary path should normally provide every
            # intermediate snapshot.
            if (
                require_contiguous_pooling_snapshots
                and not is_last_block
                and not has_boundary_snapshot
            ):
                logger.warning(
                    "Stopping PoolingCache prefix store for %s at %d tokens: "
                    "missing boundary snapshot at %d",
                    request_id,
                    global_start,
                    global_end,
                )
                break

            if split_gdn_layout and not callable(
                getattr(boundary_snapshots, "commit_gdn_checkpoint", None)
            ):
                logger.warning(
                    "Stopping split-GDN prefix store for %s at %d tokens: "
                    "boundary sidecar commit is unavailable",
                    request_id,
                    global_start,
                )
                break

            # Compute parent hash for chain-based lookup
            parent_hash = None
            if block_table.block_ids:
                prev_block_id = block_table.block_ids[-1]
                prev_block = self.paged_cache.allocated_blocks.get(prev_block_id)
                if prev_block and prev_block.block_hash:
                    parent_hash = prev_block.block_hash

            is_exact_terminal = _store_exact_terminal and i == num_new_blocks - 1
            block_extra_keys: tuple[Any, ...] | None
            if is_exact_terminal:
                block_extra_keys = (_EXACT_PREFIX_TERMINAL_KEY,)
            else:
                block_extra_keys = resolve_block_extra_keys(
                    global_end,
                    extra_keys=extra_keys,
                    extra_key_token_start=extra_key_token_start,
                    extra_key_ranges=extra_key_ranges,
                )

            # Check if this block already exists (deduplication)
            if len(block_tokens) == self.block_size and not is_exact_terminal:
                existing_block = self.paged_cache.find_cached_block(
                    block_tokens,
                    parent_hash,
                    extra_keys=block_extra_keys,
                )
                if existing_block:
                    if split_gdn_layout and not self._has_split_gdn_checkpoint(
                        existing_block.block_hash, layer_cache_types
                    ):
                        checkpoint_committed = self._commit_split_gdn_checkpoint(
                            boundary_snapshots,
                            global_end,
                            existing_block.block_hash,
                            layer_cache_types,
                            layer_meta_states,
                        )
                        if not checkpoint_committed:
                            logger.warning(
                                "Stopping split-GDN prefix store for %s at %d "
                                "tokens: deduplicated block has no recurrent "
                                "checkpoint",
                                request_id,
                                global_start,
                            )
                            break
                    # Dedup'd blocks are never rewritten, so a block first
                    # stored without snapshot coverage keeps placeholder
                    # non-sliceable payloads forever. When this store carries
                    # a boundary snapshot for the block, repair the payload
                    # so partial-match walk-back can restore here again.
                    if (
                        not split_gdn_layout
                        and is_tensor_data
                        and HAS_MLX
                        and self.paged_ssd_cache is not None
                        and has_boundary_snapshot
                        and existing_block.block_hash is not None
                        and existing_block.block_hash
                        not in self._backfill_checked_hashes
                    ):
                        snapshot_cache_data = boundary_snapshots[global_end]
                        backfilled = (
                            snapshot_cache_data
                            and self._maybe_backfill_dedup_block_payload(
                                existing_block.block_hash,
                                global_end,
                                snapshot_cache_data,
                                hot_cache_write_back=hot_cache_write_back,
                            )
                        )
                        if backfilled:
                            self._backfill_checked_hashes.add(
                                existing_block.block_hash
                            )
                            if (
                                len(self._backfill_checked_hashes)
                                > _BACKFILL_CHECKED_MAX_ENTRIES
                            ):
                                self._backfill_checked_hashes.clear()
                    # Reuse existing block
                    self.paged_cache.increment_ref(existing_block.block_id)
                    block_table.block_ids.append(existing_block.block_id)
                    block_table.num_tokens += len(block_tokens)
                    continue

            # Allocate new block
            if first_new_block_idx is None:
                first_new_block_idx = len(block_table.block_ids)
            block = self.paged_cache.allocate_block()
            if not block:
                # Retry once: a concurrent free() could have returned a
                # block to the pool between the first attempt and here.
                # There is no other source of capacity to fall back to --
                # the free_block_queue only ever holds blocks already
                # eligible for allocation, so a bare retry sees everything
                # the old handle_memory_pressure/evict_lru_blocks pair
                # could (they only recycled blocks already in that same
                # free queue, never anything held by a live request).
                # See docs/qwen35-hardening-and-optimization.md F3.
                block = self.paged_cache.allocate_block()
                if not block:
                    logger.warning(f"Cannot allocate block for {request_id}")
                    break

            # Set block metadata
            block.token_count = len(block_tokens)
            block_table.block_ids.append(block.block_id)
            block_table.num_tokens += len(block_tokens)

            # Compute chain hash for this block
            block.block_hash = compute_block_hash(
                parent_hash,
                block_tokens,
                extra_keys=block_extra_keys,
                model_name=self.paged_cache.model_name,
            )

            # Register ordinary full blocks for general prefix matching. The
            # exact terminal uses a separate hash domain so it can carry the
            # complete non-sliceable state without colliding with a normal
            # block that may contain placeholders.
            if len(block_tokens) == self.block_size and not is_exact_terminal:
                self.paged_cache.register_block_hash(
                    block, block_tokens, parent_hash, extra_keys=block_extra_keys
                )
            elif is_exact_terminal and block.block_hash is not None:
                self.paged_cache.cached_block_hash_to_block.insert(
                    block.block_hash, block
                )

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
                cache_uses_global_indices = existing_tokens > 0 and cache_seq_len >= (
                    existing_tokens + 1
                )
                if cache_uses_global_indices:
                    cache_start = global_start
                    cache_end = global_end
                else:
                    cache_start = start_idx
                    cache_end = end_idx

                # Look up boundary snapshot BEFORE the continuity check.
                # Snapshots are self-contained — they carry the full cache
                # state at this boundary, so the live-cache seq_len gate
                # below does not apply when a snapshot covers this block.
                block_boundary_tc = existing_tokens + end_idx
                snapshot_cache_data = None
                if (
                    not split_gdn_layout
                    and boundary_snapshots
                    and block_boundary_tc in boundary_snapshots
                ):
                    snapshot_cache_data = boundary_snapshots[block_boundary_tc]

                # True (not just "last") for the last block only when
                # nothing beyond it was skipped: exact-terminal mode
                # allocates with ceil() (no trailing partial skipped at
                # all), and an exact-multiple-of-block_size store has no
                # trailing partial to begin with. Ordinary store_cache's
                # last FULL block otherwise may sit behind skipped trailing
                # tokens the live state has already ingested -- see A1.
                live_state_at_true_end = is_last_block and (
                    _store_exact_terminal or trailing_partial_tokens == 0
                )

                # Continuity check applies only when we will slice live
                # cache_data for this block. Skipped when:
                #   1. A boundary snapshot exists for this block — snapshots
                #      are self-contained, so the live-cache seq_len gate
                #      does not apply.
                #   2. is_last_block is True — for sliceable hybrid models
                #      the step-1 path already returns the full prefill
                #      length, so the gate would not fire here anyway.
                #      Non-sliceable types now require a snapshot regardless
                #      of is_last_block (see A1 above), so this branch no
                #      longer needs cache_data's live seq_len for them.
                if (
                    snapshot_cache_data is None
                    and not is_last_block
                    and cache_seq_len > 0
                    and cache_start >= cache_seq_len
                ):
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

                block_kv_data = self._extract_block_tensor_slice(
                    cache_data,
                    cache_start,
                    cache_end,
                    model_cache_config,
                    is_last_block=is_last_block,
                    snapshot_cache_data=snapshot_cache_data,
                    externalize_arrays=split_gdn_layout,
                    live_state_at_true_end=live_state_at_true_end,
                )

                if block_kv_data and not self._block_token_count_matches_slices(
                    block_kv_data, block.token_count, layer_cache_types
                ):
                    # A3: a sliceable layer's own tensor length disagrees
                    # with what this block claims to hold -- persisting it
                    # would silently corrupt a later restore. Stop here,
                    # keeping the already-stored valid prefix, the same
                    # discipline the continuity-break path above uses.
                    logger.warning(
                        "Rejecting block for %s: token_count=%d does not "
                        "match stored tensor length (global [%d:%d])",
                        request_id,
                        block.token_count,
                        global_start,
                        global_end,
                    )
                    self.paged_cache.free_block(block.block_id)
                    block_table.block_ids.pop()
                    block_table.num_tokens -= len(block_tokens)
                    break

                if block_kv_data and block.block_hash:
                    # Use per-block meta_states from boundary snapshot when
                    # available. The shared layer_meta_states comes from the
                    # final cache extraction and carries the end-of-request
                    # offset (e.g. 4479) which is wrong for earlier blocks
                    # whose tensor data was captured at an earlier boundary
                    # (e.g. offset=512). Boundary snapshots record the
                    # correct per-boundary meta_state synchronously during
                    # prefill, so we prefer those.
                    block_meta = layer_meta_states
                    if (
                        snapshot_cache_data is not None
                        and layer_meta_states is not None
                    ):
                        per_block = []
                        for lidx in range(len(layer_meta_states)):
                            if (
                                lidx < len(snapshot_cache_data)
                                and isinstance(snapshot_cache_data[lidx], dict)
                                and snapshot_cache_data[lidx].get("meta_state")
                                and snapshot_cache_data[lidx]["meta_state"] != ()
                            ):
                                per_block.append(
                                    snapshot_cache_data[lidx]["meta_state"]
                                )
                            else:
                                per_block.append(layer_meta_states[lidx])
                        block_meta = per_block

                    # Save to paged SSD via PagedSSDCacheManager with cache type info
                    if hot_cache_write_back:
                        saved = self.paged_ssd_cache.save_block(
                            block_hash=block.block_hash,
                            cache_data=block_kv_data,
                            token_count=block.token_count,
                            model_name=self.paged_cache.model_name,
                            layer_cache_types=layer_cache_types,
                            layer_meta_states=block_meta,
                            replace_existing=False,
                        )
                    else:
                        saved = self.paged_ssd_cache.save_block(
                            block_hash=block.block_hash,
                            cache_data=block_kv_data,
                            token_count=block.token_count,
                            model_name=self.paged_cache.model_name,
                            layer_cache_types=layer_cache_types,
                            layer_meta_states=block_meta,
                            hot_cache_write_back=False,
                            replace_existing=False,
                        )
                    if saved:
                        if split_gdn_layout:
                            checkpoint_committed = (
                                self._commit_split_gdn_checkpoint(
                                    boundary_snapshots,
                                    block_boundary_tc,
                                    block.block_hash,
                                    layer_cache_types,
                                    layer_meta_states,
                                )
                            )
                            if not checkpoint_committed:
                                logger.warning(
                                    "Rejecting split-GDN placeholder block %s "
                                    "because its recurrent checkpoint was not "
                                    "committed",
                                    block.block_id,
                                )
                                block_removed = False
                                delete_block = getattr(
                                    self.paged_ssd_cache, "delete_block", None
                                )
                                if callable(delete_block):
                                    try:
                                        block_removed = bool(
                                            delete_block(block.block_hash)
                                        )
                                    except Exception:
                                        logger.exception(
                                            "Failed to delete rejected split-GDN "
                                            "block %s",
                                            block.block_id,
                                        )
                                if not block_removed:
                                    forget_block = getattr(
                                        self.paged_ssd_cache, "forget_block", None
                                    )
                                    if callable(forget_block):
                                        try:
                                            forget_block(block.block_hash)
                                        except Exception:
                                            logger.exception(
                                                "Failed to forget rejected split-GDN "
                                                "block %s",
                                                block.block_id,
                                            )
                                self.paged_cache.free_block(block.block_id)
                                block_table.block_ids.pop()
                                block_table.num_tokens -= len(block_tokens)
                                break
                        blocks_saved_to_ssd += 1
                        if is_last_block:
                            tip_block_saved = True
                        logger.debug(
                            f"Saved block {block.block_id} to tiered cache: "
                            f"tokens [{global_start}:{global_end}], {len(block_kv_data)} layers"
                        )
                    else:
                        logger.warning(
                            f"Failed to save block {block.block_id} to tiered cache"
                        )
                        # Persistence failed: roll back metadata so we don't
                        # retain a block that cannot be reconstructed later.
                        self.paged_cache.free_block(block.block_id)
                        block_table.block_ids.pop()
                        block_table.num_tokens -= len(block_tokens)
                        break
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

        # Supersede-on-extend: on rotating (sliding-window) models every store
        # of a growing conversation writes one tip block carrying the full
        # sliding-window state of all rotating layers (hundreds of MB fp16 on
        # a gemma3-class model). Restore only ever consumes the newest such
        # block, and the immediate previous tip is kept intact as the
        # walk-back fallback — so the tip two generations back is dead
        # weight. Without stripping it, those blocks fill the hot cache after
        # ~10-20 turns and LRU eviction breaks the prefix chain (multi-turn
        # cache hit collapses to 0%). Steady state after stripping: two heavy
        # blocks per chain.
        if (
            tip_block_saved
            and first_new_block_idx is not None
            and first_new_block_idx < len(block_table.block_ids)
            and layer_cache_types
            and any(CacheTypeRegistry.is_rotating_family(t) for t in layer_cache_types)
        ):
            new_tip_id = block_table.block_ids[-1]
            new_tip = self.paged_cache.allocated_blocks.get(new_tip_id)
            prev_tip = None
            if first_new_block_idx > 0:
                prev_tip_id = block_table.block_ids[first_new_block_idx - 1]
                prev_tip = self.paged_cache.allocated_blocks.get(prev_tip_id)
            if new_tip is not None and new_tip.block_hash is not None:
                # Only treat prev as a superseded tip when it actually was
                # one. `first_new_block_idx - 1` is merely the last reused
                # block of THIS store: on a store that diverged from an
                # existing chain it is a shared-prefix interior block (often
                # a boundary-snapshot block), and recording it here would get
                # it stripped two stores later, permanently breaking
                # partial-match walk-back restores.
                if (
                    prev_tip is not None
                    and prev_tip.block_hash is not None
                    and prev_tip.block_hash in self._store_tip_hashes
                ):
                    superseded = self._rotating_tip_lineage.pop(
                        prev_tip.block_hash, None
                    )
                    if superseded is not None:
                        self._strip_rotating_payload(superseded)
                    self._rotating_tip_lineage[new_tip.block_hash] = (
                        prev_tip.block_hash
                    )
                    if len(self._rotating_tip_lineage) > _TIP_LINEAGE_MAX_ENTRIES:
                        self._rotating_tip_lineage.clear()
                self._store_tip_hashes.add(new_tip.block_hash)
                if len(self._store_tip_hashes) > _TIP_LINEAGE_MAX_ENTRIES:
                    self._store_tip_hashes.clear()

        # Exact terminal blocks are discoverable only through
        # fetch_exact_prefix(); never expose them to general prefix matching.
        if not _store_exact_terminal:
            self._update_prefix_index(
                tokens, block_table.block_ids, extra_keys=extra_keys
            )

        # Store entry for request tracking
        self._request_tables[request_id] = BlockCacheEntry(
            block_table=block_table,
            last_access=time.time(),
        )

        logger.debug(
            f"Stored cache for {request_id}: "
            f"{len(block_table.block_ids)} blocks ({blocks_saved_to_ssd} saved to tiered cache), "
            f"{block_table.num_tokens} tokens"
        )

        return block_table

    def store_exact_prefix(
        self,
        request_id: str,
        tokens: list[int],
        cache_data: list[Any],
        model_cache_config: ModelCacheConfig | None = None,
    ) -> BlockTable | None:
        """Persist a complete prefix, including its exact terminal boundary.

        Ordinary prefix matching remains full-block-only. This path stores one
        domain-separated terminal block, excludes it from general matching,
        and restores it only when the complete static-prefix token chain
        matches. That captures an arbitrary system/tool boundary without
        making the target model repeatedly prefill its uncached suffix.
        SSD-backed configurations use write-through so the hot tier is never
        the only durable copy.
        """
        if not tokens or self.paged_ssd_cache is None:
            return None

        # Exact-prefix storage has no boundary-snapshot provider to commit
        # the recurrent state. Refuse split-GDN entries before store_cache()
        # allocates a terminal block; otherwise the terminal would contain
        # only the structural ArraysCache placeholder and could never be
        # reconstructed.
        exact_layer_cache_types = None
        if model_cache_config:
            exact_layer_cache_types = model_cache_config.get_type_names()
        elif (
            cache_data
            and isinstance(cache_data[0], dict)
            and "state" in cache_data[0]
        ):
            exact_layer_cache_types = [
                (
                    layer_state.get(
                        "class_name", layer_state.get("cache_type", "KVCache")
                    )
                    if layer_state.get("class_name", "")
                    in ("TurboQuantKVCache", "BatchTurboQuantKVCache")
                    else layer_state.get("cache_type", "KVCache")
                )
                for layer_state in cache_data
            ]
        if self._gdn_split_layout_supported(exact_layer_cache_types):
            logger.info(
                "Skipping exact-prefix store for %s: split-GDN terminal "
                "sidecar commit is unavailable",
                request_id,
            )
            self._exact_prefix_store_failures += 1
            return None

        block_table = self.store_cache(
            request_id,
            tokens,
            cache_data,
            model_cache_config=model_cache_config,
            hot_cache_write_back=False,
            _store_exact_terminal=True,
        )
        if block_table is None or block_table.num_tokens != len(tokens):
            self.paged_cache.delete_block_table(request_id)
            self._request_tables.pop(request_id, None)
            self._exact_prefix_store_failures += 1
            return None

        stored_table = block_table.copy(request_id)
        self._request_tables.pop(request_id, None)
        self.paged_cache.request_tables.pop(request_id, None)
        # Drop request ownership while retaining indexed, zero-reference block
        # metadata so the exact chain remains discoverable and evictable.
        self.paged_cache.release_for_eviction(block_table.block_ids)
        self._exact_prefix_stores += 1
        return stored_table

    def _exact_prefix_hashes(self, tokens: list[int]) -> list[tuple[BlockHash, int]]:
        """Return deterministic block hashes and token counts for an exact prefix."""
        block_hashes_and_token_counts: list[tuple[BlockHash, int]] = []
        parent_hash: BlockHash | None = None
        terminal_start = ((len(tokens) - 1) // self.block_size) * self.block_size

        for block_start in range(0, len(tokens), self.block_size):
            block_tokens = tokens[block_start : block_start + self.block_size]
            extra_keys = (
                (_EXACT_PREFIX_TERMINAL_KEY,) if block_start == terminal_start else None
            )
            block_hash = compute_block_hash(
                parent_hash,
                block_tokens,
                extra_keys=extra_keys,
                model_name=self.paged_cache.model_name,
            )
            block_hashes_and_token_counts.append((block_hash, len(block_tokens)))
            parent_hash = block_hash

        return block_hashes_and_token_counts

    def fetch_exact_prefix(
        self,
        request_id: str,
        tokens: list[int],
    ) -> BlockTable | None:
        """Acquire an all-or-nothing exact prefix chain from hot cache or SSD."""
        if not tokens or self.paged_ssd_cache is None:
            return None

        acquired_blocks: list[CacheBlock] = []
        for block_hash, token_count in self._exact_prefix_hashes(tokens):
            cached_block = self.paged_cache.cached_block_hash_to_block.get_block(
                block_hash
            )
            if cached_block is None:
                if not self.paged_ssd_cache.has_block(block_hash):
                    for block_to_release in acquired_blocks:
                        self.paged_cache.free_block(block_to_release.block_id)
                    return None
                # Recreate only the paged-manager metadata; reconstruction
                # still loads the tensor payload from the existing cache tier.
                cached_block = self.paged_cache.allocate_block()
                if cached_block is None:
                    for block_to_release in acquired_blocks:
                        self.paged_cache.free_block(block_to_release.block_id)
                    return None
                cached_block.block_hash = block_hash
                cached_block.token_count = token_count
                cached_block.ref_count = 0
                self.paged_cache.cached_block_hash_to_block.insert(
                    block_hash, cached_block
                )

            acquired_block = self.paged_cache.acquire_cached_block(
                cached_block.block_id, block_hash
            )
            if acquired_block is None:
                for previous_block in acquired_blocks:
                    self.paged_cache.free_block(previous_block.block_id)
                return None
            acquired_blocks.append(acquired_block)

        block_table = self.paged_cache.create_block_table(request_id)
        for acquired_block in acquired_blocks:
            block_table.add_block(acquired_block.block_id, acquired_block.token_count)
        self._request_tables[request_id] = BlockCacheEntry(
            block_table=block_table,
            last_access=time.time(),
        )
        return block_table

    def restore_exact_prefix(
        self,
        request_id: str,
        tokens: list[int],
        *,
        promote_to_hot_cache: bool,
    ) -> list[Any] | None:
        """Reconstruct a complete exact prefix and release temporary block refs."""
        block_table = self.fetch_exact_prefix(request_id, tokens)
        if block_table is None:
            self._exact_prefix_misses += 1
            return None

        try:
            if promote_to_hot_cache:
                self.preload_blocks(block_table)
            restored_cache = self.reconstruct_cache(
                block_table,
                promote_to_hot_cache=promote_to_hot_cache,
            )
            if restored_cache is None or block_table.num_tokens != len(tokens):
                self._exact_prefix_misses += 1
                return None
            self._exact_prefix_hits += 1
            self._exact_prefix_tokens_restored += len(tokens)
            return restored_cache
        finally:
            self.release_cache(request_id)

    @staticmethod
    def _block_token_count_matches_slices(
        block_kv_data: list[tuple[Any, Any]] | None,
        expected_token_count: int,
        layer_cache_types: list[str] | None = None,
    ) -> bool:
        """Reject a block whose stated token_count disagrees with what its
        own sliceable tensors actually hold (A3: a bug anywhere upstream of
        this point could otherwise silently persist a mismatched block,
        with no error until some later restore replays the wrong number of
        tokens against it). Only block-sliced (keys, values) 4D tensor
        entries carry a real per-block sequence length to check against;
        markers (__cache_list__, __nstate__) and placeholders ((1,)-shaped)
        don't. Non-sliceable layers (RotatingKVCache, ArraysCache) store
        full window/recurrent state whose length is legitimately unrelated
        to token_count -- the aligned block size is a MULTIPLE of the
        rotating window, not the window itself -- so when layer types are
        known, only layers whose handler supports block slicing are
        checked. When entry count and layer types disagree (an abnormal
        extraction already dropped entries), attribution is ambiguous and
        the check is skipped rather than risking a false reject.
        """
        if not block_kv_data:
            return True
        if layer_cache_types is not None and len(layer_cache_types) != len(
            block_kv_data
        ):
            return True
        for layer_idx, entry in enumerate(block_kv_data):
            if layer_cache_types is not None:
                handler = CacheTypeRegistry.get_handler_by_class_name(
                    layer_cache_types[layer_idx]
                )
                if not handler.supports_block_slicing:
                    continue
            if isinstance(entry, tuple) and len(entry) == 2:
                keys, values = entry
                keys_are_sliced = hasattr(keys, "shape") and len(keys.shape) == 4
                values_are_sliced = hasattr(values, "shape") and len(values.shape) == 4
                if not keys_are_sliced and not values_are_sliced:
                    continue
                # A sliceable KV entry must contain a matched 4D key/value
                # pair. Checking only keys lets an independently truncated
                # value tensor persist and fail much later during restore.
                if not keys_are_sliced or not values_are_sliced:
                    return False
                if (
                    keys.shape[2] != expected_token_count
                    or values.shape[2] != expected_token_count
                ):
                    return False
        return True

    def _get_cache_seq_len(self, cache_data: list[dict[str, Any]]) -> int:
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
        non_sliceable_types = {
            "ArraysCache",
            "CacheList",
            "MiniMaxM3BatchKVCache",
        }

        # Step 1: Search for a sliceable KVCache layer (full attention)
        for layer_idx, layer_state in enumerate(cache_data):
            try:
                if "state" not in layer_state:
                    continue

                # Skip non-sliceable cache types (e.g., RotatingKVCache)
                cache_type = layer_state.get("cache_type", "")
                class_name = layer_state.get("class_name", "")
                if (
                    cache_type in non_sliceable_types
                    or class_name in non_sliceable_types
                    or CacheTypeRegistry.is_rotating_family(cache_type)
                    or CacheTypeRegistry.is_rotating_family(class_name)
                ):
                    continue

                state = layer_state["state"]
                keys = state[0] if isinstance(state, (list, tuple)) else state
                # TurboQuant v2: NamedTuple state with .norms attribute
                if hasattr(keys, "norms") and hasattr(keys.norms, "shape"):
                    seq_len = keys.norms.shape[2]
                    logger.debug(
                        f"Found TurboQuantKVCache at layer {layer_idx} with seq_len={seq_len}"
                    )
                    return seq_len
                # TurboQuant v2: SplitState with .low/.high sub-states
                if hasattr(keys, "low") and hasattr(keys.low, "norms"):
                    seq_len = keys.low.norms.shape[2]
                    logger.debug(
                        f"Found TurboQuantKVCache (split) at layer {layer_idx} with seq_len={seq_len}"
                    )
                    return seq_len
                if not hasattr(keys, "shape"):
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
        step2_skip_types = {
            "ArraysCache",
            "CacheList",
            "MiniMaxM3BatchKVCache",
        }
        max_seq_len = 0
        for layer_idx, layer_state in enumerate(cache_data):
            try:
                if "state" not in layer_state:
                    continue
                cache_type = layer_state.get("cache_type", "")
                class_name = layer_state.get("class_name", "")
                if cache_type in step2_skip_types or class_name in step2_skip_types:
                    continue
                state_tuple = layer_state["state"]
                if not isinstance(state_tuple, (list, tuple)) or not state_tuple:
                    continue
                # N-tuple safe: only the first element (the keys-shaped tensor
                # in legacy KVCache, or buf_kv-shaped in PoolingCache) is
                # consulted for seq length here. Caches whose first element is
                # not a 4D KVCache-style tensor naturally skip via the shape
                # check below.
                keys = state_tuple[0]
                if hasattr(keys, "shape") and len(keys.shape) == 4:
                    max_seq_len = max(max_seq_len, keys.shape[2])
            except Exception:
                continue

        if max_seq_len > 0:
            # Normal result for all-non-sliceable-KVCache models
            # (e.g. DeepSeek V4 with RotatingKVCache + PoolingCache).
            # Returns the sliding-window length, not a sequence length —
            # callers that depend on full prefill length (continuity
            # check, reconstruction concat) bypass this layer's
            # contribution via the snapshot / is_last_block paths.
            logger.debug(
                f"Cache seq_len resolved from non-sliceable layer "
                f"(window={max_seq_len})"
            )
            return max_seq_len

        # Step 3: CacheList fallback — check sub-states for seq_len
        # This handles all-CacheList models (e.g., deepseek_v32)
        for layer_state in cache_data:
            if (
                layer_state.get("cache_type") == "CacheList"
                or layer_state.get("class_name") == "CacheList"
            ):
                sub_states = layer_state.get("state", [])
                for sub_state in sub_states:
                    if isinstance(sub_state, (list, tuple)) and len(sub_state) >= 2:
                        sub_keys = sub_state[0]
                        if hasattr(sub_keys, "shape") and len(sub_keys.shape) == 4:
                            seq_len = sub_keys.shape[2]
                            logger.debug(f"Using CacheList sub-cache seq_len={seq_len}")
                            return seq_len

        return 0

    def _strip_rotating_payload(self, block_hash: bytes) -> bool:
        """Replace a superseded tip block's rotating payload with placeholders.

        Sliceable layers (KVCache/TurboQuant slices) in the block are kept —
        only RotatingKVCache-family layer states are replaced with the same
        ``(mx.zeros((1,)), mx.zeros((1,)))`` placeholder that non-tip blocks
        receive at store time, so restore treats the stripped block exactly
        like any other placeholder block (walk-back or reject).

        The rewrite goes through ``forget_block()`` + ``save_block()``: the
        hot-cache entry is removed via ``_hot_cache_remove`` (which also
        forgets the shared-budget accounting) and the slim payload re-enters
        via ``_hot_cache_put`` (which re-registers it), so the hot-cache and
        shared-budget byte counters stay consistent. In SSD mode the slim
        payload is re-enqueued and overwrites the same hash-derived file
        path.

        Returns:
            True if the block was rewritten with at least one layer stripped.
        """
        if self.paged_ssd_cache is None or not HAS_MLX:
            return False
        try:
            data, meta = self.paged_ssd_cache.load_block_with_metadata(block_hash)
            if not data or not meta:
                return False
            types = meta.get("layer_cache_types") or []
            new_data: list[Any] = []
            stripped = 0
            for i, layer in enumerate(data):
                type_name = types[i] if i < len(types) else "KVCache"
                if (
                    CacheTypeRegistry.is_rotating_family(type_name)
                    and isinstance(layer, (list, tuple))
                    and len(layer) >= 2
                    and hasattr(layer[0], "shape")
                    and tuple(layer[0].shape) != (1,)
                ):
                    new_data.append((mx.zeros((1,)), mx.zeros((1,))))
                    stripped += 1
                elif type_name == "CacheList" and isinstance(layer, list):
                    # load_block_with_metadata() exposes CacheList payloads as
                    # their legacy list shape.  Restore the storage marker
                    # before re-saving so save_block keeps sub_count metadata
                    # and can stamp the same cachelist_subtypes signature.
                    new_data.append(("__cache_list__", layer))
                else:
                    new_data.append(layer)
            if stripped == 0:
                return False
            # save_block dedups on an existing hash, so drop the old entry
            # (hot cache, pending writes, SSD index) first. The brief gap is
            # benign: a concurrent restore either already loaded the old
            # payload or sees the placeholder version, which the
            # walk-back/reject path handles like any partial match.
            self.paged_ssd_cache.forget_block(block_hash)
            saved = self.paged_ssd_cache.save_block(
                block_hash=block_hash,
                cache_data=new_data,
                token_count=int(meta.get("token_count") or self.block_size),
                model_name=meta.get("model_name", self.paged_cache.model_name),
                layer_cache_types=types or None,
                layer_meta_states=meta.get("layer_meta_states"),
            )
            if saved:
                logger.debug(
                    "Stripped rotating payload from superseded tip block %s "
                    "(%d of %d layers)",
                    block_hash.hex()[:16],
                    stripped,
                    len(data),
                )
            return bool(saved)
        except Exception as e:
            logger.debug(
                "Rotating payload strip failed for %s: %s", block_hash.hex()[:16], e
            )
            return False

    def _cachelist_snapshot_sub_tensors(
        self,
        source_layer: dict[str, Any],
        source_state: Any,
        sub_class_names: list[str],
    ) -> list[Any] | None:
        """Build ``__cache_list__`` sub-tensors from an extracted layer dict.

        Clones *all* sub-state elements (PoolingCache's third element
        ``pooled`` must survive the round-trip) and wraps PoolingCache subs
        that carry a delta range as ``PoolingCacheDelta`` storage entries.
        Returns None when the state is not a usable list.
        """
        if not isinstance(source_state, list):
            return None
        pooling_delta_ranges = source_layer.get("pooling_delta_ranges", {})
        sub_tensors: list[Any] = []
        for sub_idx, sub_state in enumerate(source_state):
            if not (isinstance(sub_state, (list, tuple)) and len(sub_state) >= 1):
                continue
            cloned = [
                self._clone_tensor(elem) if hasattr(elem, "shape") else elem
                for elem in sub_state
            ]
            delta_range = pooling_delta_ranges.get(str(sub_idx))
            sub_class = (
                sub_class_names[sub_idx] if sub_idx < len(sub_class_names) else None
            )
            if (
                sub_class == "PoolingCache"
                and isinstance(delta_range, (list, tuple))
                and len(delta_range) == 2
            ):
                cloned.append(mx.array(delta_range, dtype=mx.int64))
                sub_tensors.append(
                    _wrap_cachelist_sub_marker(
                        sub_idx,
                        cloned,
                        sub_class_names,
                        POOLING_CACHE_DELTA_CLASS,
                    )
                )
            else:
                sub_tensors.append(
                    _wrap_cachelist_sub_marker(sub_idx, cloned, sub_class_names)
                )
        return sub_tensors

    def _maybe_backfill_dedup_block_payload(
        self,
        block_hash: bytes,
        boundary_tc: int,
        snapshot_cache_data: list[dict[str, Any]],
        hot_cache_write_back: bool = True,
    ) -> bool:
        """Rewrite a dedup'd placeholder block with boundary-snapshot state.

        Deduplicated blocks are never rewritten by store_cache, so a block
        first stored without snapshot coverage (or stripped since) keeps
        placeholder non-sliceable payloads forever — every partial prefix
        match that ends inside such a region is rejected and the request
        re-prefills from scratch. When the current store re-processed the
        same tokens and carries a boundary snapshot for this block's
        boundary, rewrite the placeholder layers with the same
        snapshot-derived payload a fresh snapshot-covered block would get,
        making walk-back restores at this boundary possible again.

        Strip interaction: _strip_rotating_payload only targets blocks that
        still hold real rotating state, and this method only replaces
        placeholder layers, so the two never rewrite the same block in one
        store. A previously stripped ex-tip may be deliberately re-backfilled
        by a later covered store.

        Returns True when the block needs no further inspection this session
        (payload already real, unusable snapshot, or rewrite saved); False
        on transient load/save failure so a later store retries.
        """
        if self.paged_ssd_cache is None or not HAS_MLX:
            return True
        try:
            # Inspection load only: do not promote into the hot cache. A
            # legacy-cumulative block (pm-ineligible CacheList layouts, e.g.
            # CacheList(KVCache, PoolingCache)) carries the full cumulative
            # KV prefix; promoting every inspected block retains those
            # oversized payloads in RAM for nothing when the common outcome
            # is "payload already real, no rewrite" (#3226).
            data, meta = self.paged_ssd_cache.load_block_with_metadata(
                block_hash,
                promote_to_hot_cache=False,
            )
            if not data or not meta:
                return False
            types = meta.get("layer_cache_types") or []
            new_data: list[Any] = []
            replaced_indices: list[int] = []
            for i, layer in enumerate(data):
                type_name = types[i] if i < len(types) else "KVCache"
                handler = CacheTypeRegistry.get_handler_by_class_name(type_name)
                if handler.supports_block_slicing or not self._is_placeholder_state(
                    layer
                ):
                    if type_name == "CacheList" and isinstance(layer, list):
                        # load_block_with_metadata() exposes CacheList
                        # payloads as their legacy list shape; restore the
                        # storage marker before re-saving (same as strip).
                        new_data.append(("__cache_list__", layer))
                    else:
                        new_data.append(layer)
                    continue

                snap_layer = (
                    snapshot_cache_data[i]
                    if i < len(snapshot_cache_data)
                    else None
                )
                if not isinstance(snap_layer, dict) or "state" not in snap_layer:
                    return True
                snap_state = snap_layer["state"]
                if CacheTypeRegistry.is_rotating_family(type_name):
                    if not (
                        isinstance(snap_state, (list, tuple))
                        and len(snap_state) >= 2
                    ):
                        return True
                    new_data.append(
                        (
                            self._clone_tensor(snap_state[0]),
                            self._clone_tensor(snap_state[1]),
                        )
                    )
                elif type_name == "CacheList":
                    names = snap_layer.get("sub_class_names") or []
                    if not names:
                        snap_meta = snap_layer.get("meta_state")
                        if (
                            isinstance(snap_meta, (list, tuple))
                            and len(snap_meta) >= 2
                            and isinstance(snap_meta[0], (list, tuple))
                        ):
                            names = [str(n) for n in snap_meta[0]]
                    subs = self._cachelist_snapshot_sub_tensors(
                        snap_layer, snap_state, names
                    )
                    if subs is None:
                        return True
                    new_data.append(("__cache_list__", subs))
                else:
                    # Other non-sliceable layouts (ArraysCache family) keep
                    # today's behavior; never persist a partially-filled
                    # block.
                    return True
                replaced_indices.append(i)

            if not replaced_indices:
                return True

            new_meta = list(meta.get("layer_meta_states") or [])
            while len(new_meta) < len(data):
                new_meta.append(())
            for i in replaced_indices:
                snap_meta_state = snapshot_cache_data[i].get("meta_state")
                if snap_meta_state:
                    new_meta[i] = snap_meta_state

            self.paged_ssd_cache.forget_block(block_hash)
            saved = self.paged_ssd_cache.save_block(
                block_hash=block_hash,
                cache_data=new_data,
                token_count=int(meta.get("token_count") or self.block_size),
                model_name=meta.get("model_name", self.paged_cache.model_name),
                layer_cache_types=types or None,
                layer_meta_states=new_meta,
                hot_cache_write_back=hot_cache_write_back,
            )
            if not saved:
                return False
            logger.info(
                "Backfilled dedup block %s at boundary %d (%d layer(s))",
                block_hash.hex()[:16],
                boundary_tc,
                len(replaced_indices),
            )
            return True
        except Exception as e:
            logger.debug(
                "Dedup block backfill failed for %s: %s", block_hash.hex()[:16], e
            )
            return False

    def _extract_block_tensor_slice(
        self,
        cache_data: list[dict[str, Any]],
        start_idx: int,
        end_idx: int,
        model_cache_config: ModelCacheConfig | None = None,
        is_last_block: bool = False,
        snapshot_cache_data: list[dict[str, Any]] | None = None,
        externalize_arrays: bool = False,
        live_state_at_true_end: bool = False,
    ) -> list[tuple[Any, Any]] | None:
        """
        Extract tensor slices for a single block from cache data.

        Supports different cache types (KVCache, RotatingKVCache, ArraysCache)
        with type-aware slicing. For non-sliceable types like ArraysCache,
        returns the full state.

        For RotatingKVCache and other non-sliceable (recurrent/summary state)
        layers specifically:
        - Boundary snapshot: if a snapshot was captured at this block's
          boundary, the snapshot state is used
        - live_state_at_true_end: if True (last block AND the caller has
          verified no tokens beyond this block were skipped -- e.g. an
          exact-multiple-of-block_size store, or exact-terminal mode), the
          live cache_data state is used as if it were the snapshot
        - Otherwise: stores a placeholder (mx.zeros((1,)), mx.zeros((1,))),
          even on the last block. A block is only "last" among the FULL
          blocks -- store_cache skips trailing partial blocks by default --
          so the live end-of-request state may already have ingested tokens
          past this boundary; storing it as this block's state would
          silently double-ingest those tokens on restore. Losing reuse of
          that one block is strictly better than corrupting it
          (docs/qwen35-hardening-and-optimization.md A1).

        During restore, a partial prefix match that ends on a placeholder block
        first attempts walk-back truncation to the latest block with valid
        non-sliceable state. If no such block exists, the cache hit is rejected.

        Args:
            cache_data: List of layer states, each containing 'state': (keys, values)
                or other cache-type-specific format
            start_idx: Start token index in the sequence
            end_idx: End token index in the sequence
            model_cache_config: Optional model cache configuration with per-layer
                type information
            is_last_block: If True, this is the last full block being stored.
                Not sufficient by itself to justify storing live state for
                non-sliceable layers (see live_state_at_true_end).
            snapshot_cache_data: Boundary snapshot cache data for this block.
                Lets non-sliceable layers store real state without needing
                live_state_at_true_end.
            live_state_at_true_end: The caller's assertion that, for the
                last block, cache_data's live state has NOT ingested any
                tokens beyond this block's boundary (e.g. token count is an
                exact multiple of block_size, or exact-terminal mode).
                Combined with is_last_block, lets non-sliceable layers use
                live state as a snapshot substitute when no real snapshot
                was captured. False by default -- callers must verify the
                invariant before setting it.

        Returns:
            List of (keys_slice, values_slice) for each layer, or None on failure
        """
        if not HAS_MLX or not cache_data:
            return None

        try:
            block_slices = []
            for layer_idx, layer_state in enumerate(cache_data):
                if "state" not in layer_state:
                    continue

                # Determine cache type for this layer
                cache_type_name = layer_state.get("cache_type", "KVCache")
                if model_cache_config and layer_idx < len(
                    model_cache_config.layer_configs
                ):
                    cache_type_name = model_cache_config.layer_configs[
                        layer_idx
                    ].class_name

                handler = CacheTypeRegistry.get_handler_by_class_name(cache_type_name)

                if cache_type_name in ("TurboQuantKVCache", "BatchTurboQuantKVCache"):
                    # TurboQuant v2: NamedTuple state from mlx-vlm
                    from ..turboquant_kv import _slice_state_range, _state_length

                    state = layer_state["state"]
                    if not isinstance(state, (list, tuple)) or len(state) < 2:
                        block_slices.append((mx.zeros((1,)), mx.zeros((1,))))
                        continue
                    k_state, v_state = state[0], state[1]
                    # Unwrap _QuantizedStateProxy if present
                    if hasattr(k_state, "_state"):
                        k_state = k_state._state
                    if hasattr(v_state, "_state"):
                        v_state = v_state._state
                    seq_len = _state_length(k_state)
                    actual_end = min(end_idx, seq_len)
                    if start_idx >= actual_end:
                        block_slices.append((mx.zeros((1,)), mx.zeros((1,))))
                        continue
                    ks = _slice_state_range(k_state, start_idx, actual_end)
                    vs = _slice_state_range(v_state, start_idx, actual_end)
                    block_slices.append(
                        (
                            "__turboquant_v2__",
                            (ks, vs),
                        )
                    )
                elif handler.supports_block_slicing:
                    # Standard 4D KV cache slicing
                    state = layer_state["state"]
                    if not isinstance(state, (list, tuple)) or len(state) < 2:
                        # Placeholder from boundary snapshot (skipped sliceable layer).
                        continue

                    axis_info = handler.get_state_axis_info()
                    if len(state) != 2 or len(axis_info) != 2:
                        seq_len = handler.get_state_seq_len_from_tuple(tuple(state))
                        actual_end = min(end_idx, seq_len)
                        if seq_len <= 0 or start_idx >= actual_end:
                            continue

                        sliced_elements = []
                        for info, elem in zip(axis_info, state):
                            if elem is None:
                                sliced_elements.append(None)
                                continue
                            if (
                                info.sliceable
                                and info.sequence_axis is not None
                                and hasattr(elem, "shape")
                                and info.sequence_axis < len(elem.shape)
                            ):
                                slices = [slice(None)] * len(elem.shape)
                                slices[info.sequence_axis] = slice(
                                    start_idx, actual_end
                                )
                                sliced_elements.append(
                                    self._clone_tensor(elem[tuple(slices)])
                                )
                            else:
                                sliced_elements.append(
                                    self._clone_tensor(elem)
                                    if hasattr(elem, "shape")
                                    else elem
                                )
                        block_slices.append(
                            ("__nstate__", cache_type_name, sliced_elements)
                        )
                        continue

                    keys, values = state

                    # KV cache shape: (batch, n_kv_heads, seq_len, head_dim)
                    # Slice along seq_len dimension (axis 2)
                    if not hasattr(keys, "shape") or len(keys.shape) < 4:
                        # Handle 3D case (no batch dimension)
                        if hasattr(keys, "shape") and len(keys.shape) == 3:
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
                        (
                            self._clone_tensor(keys_slice),
                            self._clone_tensor(values_slice),
                        )
                    )
                elif CacheTypeRegistry.is_rotating_family(cache_type_name):
                    # RotatingKVCache: boundary-snapshot strategy, or live
                    # state only when the caller has verified nothing was
                    # skipped past this block (live_state_at_true_end) -- a
                    # block is only ever "last" among the FULL blocks, and
                    # store_cache skips trailing partial blocks by default,
                    # so the live end-of-request state may already have
                    # ingested tokens past this boundary. Storing it as the
                    # boundary state in that case would silently
                    # double-ingest those tokens on restore
                    # (docs/qwen35-hardening-and-optimization.md A1) -- a
                    # missing snapshot without that verified guarantee falls
                    # to the placeholder below even on the last block, since
                    # losing reuse of that one block is strictly better than
                    # corrupting it.
                    has_snapshot = (
                        snapshot_cache_data is not None
                        and layer_idx < len(snapshot_cache_data)
                        and "state" in snapshot_cache_data[layer_idx]
                    )
                    has_valid_state = has_snapshot or (
                        is_last_block and live_state_at_true_end
                    )
                    if has_valid_state:
                        state = (
                            snapshot_cache_data[layer_idx]["state"]
                            if has_snapshot
                            else layer_state["state"]
                        )
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
                        # Non-last block without snapshot: store placeholder
                        block_slices.append((mx.zeros((1,)), mx.zeros((1,))))
                elif cache_type_name == "CacheList":
                    state = layer_state["state"]  # List[sub_state]
                    sub_class_names = layer_state.get("sub_class_names") or []
                    if not sub_class_names:
                        # Snapshot entries and older extract paths carry the
                        # sub class names only inside the composite
                        # meta_state ([class_names], [sub_meta_states]).
                        meta = layer_state.get("meta_state")
                        if (
                            isinstance(meta, (list, tuple))
                            and len(meta) >= 2
                            and isinstance(meta[0], (list, tuple))
                        ):
                            sub_class_names = [str(n) for n in meta[0]]
                    if not isinstance(state, list) or len(state) == 0:
                        block_slices.append((mx.zeros((1,)), mx.zeros((1,))))
                        continue

                    # Check if all sub-caches are sliceable 4D KVCache tensors.
                    # PoolingCache fails this check (its first element buf_kv
                    # is 3D), so a CacheList containing a PoolingCache falls
                    # to the non-sliceable last-block-only branch below.
                    all_sub_sliceable = all(
                        isinstance(ss, (list, tuple))
                        and len(ss) >= 2
                        and hasattr(ss[0], "shape")
                        and len(ss[0].shape) == 4
                        for ss in state
                    )

                    def _wrap_sub_marker(sub_idx, elements, storage_class_name=None):
                        return _wrap_cachelist_sub_marker(
                            sub_idx, elements, sub_class_names, storage_class_name
                        )

                    def _slice_sub_elements(sub_state):
                        # Per-block slicing along sequence axis. Generic over
                        # the full sub-state tuple length so 4D N-tuple caches
                        # (BatchKVCache: 4 elements with offset/padding meta
                        # at indices 2/3) round-trip without dropping
                        # elements past index 1.
                        seq_len = sub_state[0].shape[2]
                        actual_end = min(end_idx, seq_len)
                        sliced_elements = []
                        for elem in sub_state:
                            if (
                                hasattr(elem, "shape")
                                and len(elem.shape) == 4
                                and elem.shape[2] == seq_len
                            ):
                                if start_idx >= actual_end:
                                    sliced_elements.append(
                                        self._clone_tensor(elem[:, :, 0:0, :])
                                    )
                                else:
                                    sliced_elements.append(
                                        self._clone_tensor(
                                            elem[:, :, start_idx:actual_end, :]
                                        )
                                    )
                            else:
                                # Non-sequence element (e.g. BatchKVCache
                                # offset/left_padding metadata). Pass
                                # through unsliced.
                                sliced_elements.append(
                                    self._clone_tensor(elem)
                                    if hasattr(elem, "shape")
                                    else elem
                                )
                        return sliced_elements

                    # Per-member storage plan for mixed CacheLists (sliceable
                    # KV subs + ArraysCache-style subs). Eligible layers
                    # store per-block KV slices plus the boundary snapshot's
                    # small non-sliceable state — linear instead of the
                    # legacy quadratic cumulative-per-block layout.
                    pm_plan = cachelist_pm_member_plan(sub_class_names, state)
                    pm_snapshot_state = None
                    if (
                        snapshot_cache_data is not None
                        and layer_idx < len(snapshot_cache_data)
                        and isinstance(
                            snapshot_cache_data[layer_idx].get("state"), list
                        )
                    ):
                        pm_snapshot_state = snapshot_cache_data[layer_idx]["state"]
                    # See the rotating-family branch above (A1): a missing
                    # boundary snapshot must fall to the placeholder path
                    # even on the last block, unless the caller has verified
                    # nothing was skipped past this block
                    # (live_state_at_true_end), in which case sub_state IS
                    # the boundary state.
                    pm_has_snapshot = pm_plan is not None and (
                        pm_snapshot_state is not None
                        and all(
                            sub_idx < len(pm_snapshot_state)
                            and isinstance(
                                pm_snapshot_state[sub_idx], (list, tuple)
                            )
                            and len(pm_snapshot_state[sub_idx]) >= 1
                            for sub_idx, mode in enumerate(pm_plan)
                            if mode == "boundary"
                        )
                    )
                    pm_source_ok = pm_has_snapshot or (
                        pm_plan is not None
                        and is_last_block
                        and live_state_at_true_end
                    )

                    if all_sub_sliceable:
                        sub_tensors = []
                        for sub_idx, sub_state in enumerate(state):
                            sub_tensors.append(
                                _wrap_sub_marker(
                                    sub_idx, _slice_sub_elements(sub_state)
                                )
                            )
                        block_slices.append(("__cache_list__", sub_tensors))
                    elif pm_source_ok:
                        sub_tensors = []
                        for sub_idx, sub_state in enumerate(state):
                            if pm_plan[sub_idx] == "slice":
                                sub_tensors.append(
                                    _wrap_sub_marker(
                                        sub_idx, _slice_sub_elements(sub_state)
                                    )
                                )
                                continue
                            # Boundary member: use the snapshot when one
                            # exists; otherwise pm_source_ok already
                            # verified live_state_at_true_end, so sub_state
                            # itself is the boundary state.
                            source = (
                                pm_snapshot_state[sub_idx]
                                if pm_has_snapshot
                                else sub_state
                            )
                            cloned = [
                                (
                                    self._clone_tensor(elem)
                                    if hasattr(elem, "shape")
                                    else elem
                                )
                                for elem in source
                            ]
                            sub_tensors.append(_wrap_sub_marker(sub_idx, cloned))
                        block_slices.append(("__cache_list_pm__", sub_tensors))
                    else:
                        # Non-sliceable sub-caches: last-block-only or snapshot.
                        # This is the path PoolingCache takes (3D buf_kv).
                        # Critical fix: clone *all* sub_state elements, not
                        # just the first two, so PoolingCache's third element
                        # `pooled` survives the round-trip. Dropping it was
                        # the V4 cross-session corruption root cause.
                        # See the rotating-family branch above (A1): a
                        # missing boundary snapshot must fall to the
                        # placeholder path even on the last block, unless
                        # the caller verified live_state_at_true_end.
                        has_snapshot = (
                            snapshot_cache_data is not None
                            and layer_idx < len(snapshot_cache_data)
                            and "state" in snapshot_cache_data[layer_idx]
                        )
                        has_valid_state = has_snapshot or (
                            is_last_block and live_state_at_true_end
                        )
                        if has_valid_state:
                            source_layer = (
                                snapshot_cache_data[layer_idx]
                                if has_snapshot
                                else layer_state
                            )
                            source_state = source_layer["state"]
                            sub_tensors = self._cachelist_snapshot_sub_tensors(
                                source_layer, source_state, sub_class_names
                            )
                            if sub_tensors is not None:
                                block_slices.append(("__cache_list__", sub_tensors))
                            else:
                                block_slices.append((mx.zeros((1,)), mx.zeros((1,))))
                        else:
                            block_slices.append((mx.zeros((1,)), mx.zeros((1,))))
                elif externalize_arrays and CacheTypeRegistry.is_arrays_family(
                    cache_type_name
                ):
                    # Split-GDN blocks keep only the structural placeholder;
                    # the complete recurrent state is committed separately
                    # from the boundary snapshot by store_cache().
                    block_slices.append((mx.zeros((1,)), mx.zeros((1,))))
                else:
                    # Other non-sliceable cache (ArraysCache/MambaCache or
                    # model-specific caches such as MiniMax M3). N-tuple
                    # caches keep every element via the SSD V3 marker format.
                    # GDN recurrent state summarizes the ENTIRE sequence in a
                    # fixed-size matrix. Each block boundary snapshot captures
                    # the state at that point in the sequence. Without a snapshot,
                    # non-last blocks get a placeholder so partial matches are
                    # detected and rejected during reconstruction.
                    # See the rotating-family branch above (A1): GDN's
                    # recurrent state summarizes the ENTIRE sequence, so the
                    # live end-of-request state may already have ingested
                    # tokens past this block's boundary when a trailing
                    # partial block was skipped -- a missing snapshot must
                    # fall to the placeholder path even on the last block,
                    # unless the caller verified live_state_at_true_end.
                    has_snapshot = (
                        snapshot_cache_data is not None
                        and layer_idx < len(snapshot_cache_data)
                        and "state" in snapshot_cache_data[layer_idx]
                    )
                    has_valid_state = has_snapshot or (
                        is_last_block and live_state_at_true_end
                    )
                    if has_valid_state:
                        state = (
                            snapshot_cache_data[layer_idx]["state"]
                            if has_snapshot
                            else layer_state["state"]
                        )
                        if isinstance(state, (list, tuple)) and len(state) > 2:
                            cloned = [
                                (
                                    self._clone_tensor(elem)
                                    if hasattr(elem, "shape")
                                    else elem
                                )
                                for elem in state
                            ]
                            block_slices.append(("__nstate__", cache_type_name, cloned))
                        elif isinstance(state, (list, tuple)) and len(state) >= 2:
                            conv_state = (
                                state[0] if state[0] is not None else mx.array([])
                            )
                            ssm_state = (
                                state[1] if state[1] is not None else mx.array([])
                            )
                            block_slices.append(
                                (
                                    self._clone_tensor(conv_state),
                                    self._clone_tensor(ssm_state),
                                )
                            )
                        else:
                            logger.debug(
                                f"Layer {layer_idx}: {cache_type_name} unexpected state format"
                            )
                            block_slices.append((mx.zeros((1,)), mx.zeros((1,))))
                    else:
                        # Non-last block without snapshot: store placeholder
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
            if hasattr(first, "shape") and first.shape == (1,):
                return True
        return False

    def _find_walk_back_truncation_point(
        self,
        all_block_data: list[list[Any]],
        layer_cache_types: list[str] | None,
    ) -> int | None:
        """Find the latest block where all non-sliceable layers have valid state.

        In multi-turn conversations, intermediate blocks can accumulate real
        non-sliceable state (ArraysCache/RotatingKVCache/CacheList) from prior
        save operations while later blocks only have placeholders. This method
        walks backwards from the last loaded block to locate the most recent
        block where **every** non-sliceable layer carries real state.

        Returns:
            0-based block index (inclusive) to truncate to, or ``None`` if
            no truncation is needed (last block already valid) or no valid
            fallback block exists.
        """
        if not all_block_data or not layer_cache_types:
            return None

        num_layers = len(all_block_data[0])
        last_idx = len(all_block_data) - 1

        # Identify "problematic" layers: non-sliceable layer type with
        # placeholder state in the last matched block.
        problematic_layers: list[int] = []
        for layer_idx in range(num_layers):
            cache_type = (
                layer_cache_types[layer_idx]
                if layer_idx < len(layer_cache_types)
                else "KVCache"
            )
            handler = CacheTypeRegistry.get_handler_by_class_name(cache_type)
            if handler.supports_block_slicing:
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
        model_cache_config: ModelCacheConfig | None = None,
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
    ) -> tuple[list[Any] | None, bool]:
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
    ) -> BlockTable | None:
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

        logger.debug(f"Forked cache: {source_request_id} -> {new_request_id}")

        return forked_table

    def preload_blocks(self, block_table: BlockTable) -> int:
        """
        Pre-load matched blocks from SSD into hot cache in parallel.

        Call this between fetch_cache() and reconstruct_cache() to
        convert cold-SSD reads into hot-cache hits. Warm-start requests
        (blocks already in hot cache) return 0 with no I/O.

        Args:
            block_table: BlockTable from fetch_cache() containing matched block IDs.

        Returns:
            Number of blocks successfully preloaded into hot cache.
        """
        if self.paged_ssd_cache is None:
            return 0
        if not block_table or not block_table.block_ids:
            return 0

        block_hashes = []
        for block_id in block_table.block_ids:
            block = self.paged_cache.allocated_blocks.get(block_id)
            if block and block.block_hash is not None:
                block_hashes.append(block.block_hash)

        if not block_hashes:
            return 0

        return self.paged_ssd_cache.preload_matched_blocks(block_hashes)

    def reconstruct_cache(
        self,
        block_table: BlockTable,
        promote_to_hot_cache: bool = True,
    ) -> list[Any] | None:
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
            promote_to_hot_cache: When False, SSD-loaded blocks are not retained
                in hot cache after active KV reconstruction.

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
            logger.warning(
                "Cannot reconstruct cache: PagedSSDCacheManager not configured"
            )
            return None

        try:
            # Collect cache data from valid blocks (stop at first invalid)
            all_block_data = []
            valid_block_count = 0
            valid_token_count = 0

            # Cache type information from blocks.
            # Anchor the per-block comparison to the live model's signature
            # rather than block 0's metadata. Bootstrapping from block 0
            # means a stale block (saved before TurboQuant/MTP toggled)
            # silently becomes the truth and every newer, correctly-typed
            # block trips the mismatch warning forever. With the live
            # signature as the reference, the stale block 0 itself gets
            # forgotten on the first failed comparison and reuse can
            # extend past it on the next request.
            manager_expected = (
                getattr(self.paged_ssd_cache, "_expected_layer_cache_types", None)
                if self.paged_ssd_cache is not None
                else None
            )
            # Only adopt when the manager really has a concrete signature.
            # Guard against MagicMock auto-attrs in tests and empty lists
            # (which would unanimously trip the mismatch check); fall back
            # to the historical block-0 bootstrap in those cases.
            if isinstance(manager_expected, (list, tuple)) and manager_expected:
                layer_cache_types = list(manager_expected)
            else:
                layer_cache_types = None
            first_block_meta_states = None  # meta_states from first block
            last_block_meta_states = (
                None  # meta_states from last block (for non-sliceable caches)
            )
            all_block_meta_states = []  # per-block meta_states for walk-back truncation

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
                if promote_to_hot_cache:
                    block_data, block_metadata = (
                        self.paged_ssd_cache.load_block_with_metadata(block.block_hash)
                    )
                else:
                    block_data, block_metadata = (
                        self.paged_ssd_cache.load_block_with_metadata(
                            block.block_hash,
                            promote_to_hot_cache=False,
                        )
                    )
                if block_data is None:
                    logger.debug(
                        f"Failed to load block {block_id} from tiered cache, "
                        f"using {valid_block_count} valid blocks"
                    )
                    # Remove failed block from hash cache to prevent future false hits
                    if block.block_hash is not None:
                        self.paged_cache.cached_block_hash_to_block.pop(
                            block.block_hash, block.block_id
                        )
                        self.paged_cache._notify_hash_dropped(block.block_hash)
                        logger.debug(
                            f"Removed missing block {block_id} from hash cache"
                        )
                    break  # Stop here, use valid prefix

                # Validate model_name to prevent cross-model cache contamination
                if block_metadata:
                    block_model_name = block_metadata.get("model_name", "")
                    current_model_name = self.paged_cache.model_name

                    # If current model has a name, validate against block's model
                    if current_model_name:
                        if not block_model_name:
                            # Block was saved without model_name (old cache), skip it
                            logger.warning(
                                f"Block has no model_name (legacy cache), "
                                f"current model is '{current_model_name}'. Invalidating cache hit."
                            )
                            self._forget_incompatible_ssd_block(block.block_hash)
                            break  # Stop here, don't use this block
                        elif block_model_name != current_model_name:
                            # Block is from a different model
                            logger.warning(
                                f"Cache model mismatch: block is for '{block_model_name}', "
                                f"current model is '{current_model_name}'. Invalidating cache hit."
                            )
                            self._forget_incompatible_ssd_block(block.block_hash)
                            break  # Stop here, don't use this block

                    # Validate num_layers to catch cross-model cache issues
                    block_num_layers = block_metadata.get("num_layers", 0)
                    if self.expected_num_layers > 0 and block_num_layers > 0:
                        if block_num_layers != self.expected_num_layers:
                            logger.warning(
                                f"Cache layer count mismatch: block has {block_num_layers} layers, "
                                f"expected {self.expected_num_layers}. Invalidating cache hit."
                            )
                            self._forget_incompatible_ssd_block(block.block_hash)
                            break  # Stop here, don't use this block

                    if "block_size" in block_metadata:
                        block_size = block_metadata.get("block_size", 0)
                        if block_size and block_size != self.block_size:
                            logger.warning(
                                f"Cache block size mismatch: block has block_size={block_size}, "
                                f"expected {self.block_size}. Invalidating cache hit."
                            )
                            self._forget_incompatible_ssd_block(block.block_hash)
                            break  # Stop here, don't use this block
                        if not block_size and self.block_size:
                            logger.warning(
                                "Block has no block_size metadata (legacy cache), "
                                f"current block_size is {self.block_size}. Invalidating cache hit."
                            )
                            self._forget_incompatible_ssd_block(block.block_hash)
                            break  # Stop here, don't use this block

                # Extract type info from block metadata
                if block_metadata:
                    block_layer_cache_types = block_metadata.get("layer_cache_types")
                    if layer_cache_types is None:
                        layer_cache_types = block_layer_cache_types
                    elif (
                        block_layer_cache_types is not None
                        and self._canonical_layer_cache_types(block_layer_cache_types)
                        != self._canonical_layer_cache_types(layer_cache_types)
                    ):
                        logger.warning(
                            "Cache layer type mismatch at block %s: got %s, "
                            "expected %s. Truncating cached prefix before this "
                            "block.",
                            block_id,
                            block_layer_cache_types,
                            layer_cache_types,
                        )
                        self._forget_incompatible_ssd_block(
                            block.block_hash, block.block_id
                        )
                        break

                    # Expectation-gated signature fields (TurboQuant depth,
                    # CacheList sub composition). Hot-cache / pending-write
                    # loads bypass the manager's index-scan compatibility
                    # check, so the restore loop must gate them here.
                    signature_gate = getattr(
                        self.paged_ssd_cache, "is_signature_compatible", None
                    )
                    if callable(signature_gate) and not signature_gate(
                        block_metadata.get("cache_signature", "")
                    ):
                        mismatch_reason = (
                            "TurboQuant depth or CacheList sub composition"
                        )
                        reason_getter = getattr(
                            self.paged_ssd_cache,
                            "signature_mismatch_reason",
                            None,
                        )
                        if callable(reason_getter):
                            candidate = reason_getter(
                                block_metadata.get("cache_signature", "")
                            )
                            if isinstance(candidate, str) and candidate:
                                mismatch_reason = candidate
                        logger.warning(
                            "Cache signature mismatch at block %s: %s. "
                            "Truncating cached prefix before this block.",
                            block_id,
                            mismatch_reason,
                        )
                        self._forget_incompatible_ssd_block(
                            block.block_hash, block.block_id
                        )
                        break

                    # Track meta_states from first and last blocks
                    # Non-sliceable caches (RotatingKVCache) need last block's meta_state
                    block_layer_meta_states = block_metadata.get("layer_meta_states")
                    if first_block_meta_states is None:
                        first_block_meta_states = block_layer_meta_states
                    # Always update last to track the most recent
                    last_block_meta_states = block_layer_meta_states
                    all_block_meta_states.append(block_layer_meta_states)
                else:
                    # A block that loads with data but no metadata at all
                    # cannot be trusted: it skipped every per-block validation
                    # gate above (model_name, num_layers, block_size, layer
                    # types), and continuing would leave the first/last
                    # meta_state trackers stale (pairing this block's tensors
                    # with an earlier block's meta). Truncate the chain here
                    # and use the valid prefix, like the load-failure path.
                    logger.warning(
                        f"Block {block_id} loaded without metadata. "
                        f"Truncating cached prefix before this block "
                        f"({valid_block_count} valid blocks)."
                    )
                    self._forget_incompatible_ssd_block(
                        block.block_hash, block.block_id
                    )
                    break

                # Validate loaded data (pass cache types for hybrid models)
                if not self._validate_block_cache_data(block_data, layer_cache_types):
                    logger.debug(
                        f"Block {block_id} has invalid layer data from tiered cache, "
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
                    block_table.block_ids.clear()
                    block_table.num_tokens = 0
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

            # Split-GDN restore: select the newest endpoint whose KV prefix is
            # contiguous and whose recurrent sidecar matches the live cache
            # signature.  Only that one sidecar is materialized; historical
            # GDN checkpoints stay on SSD.
            if self._gdn_split_layout_supported(layer_cache_types):
                # This is a per-attempt diagnostic.  Clear the previous
                # successful restore before looking for a new endpoint so an
                # admin poll cannot attribute an old endpoint to a miss.
                self._last_gdn_restore = None
                signature_builder = getattr(
                    self.paged_ssd_cache, "gdn_cache_signature_for", None
                )
                if not callable(signature_builder):
                    signature_builder = getattr(
                        self.paged_ssd_cache, "cache_signature_for", None
                    )
                sidecar_lookup_getter = getattr(
                    self.paged_ssd_cache,
                    "get_gdn_checkpoint_file_with_diagnostic",
                    None,
                )
                sidecar_getter = getattr(
                    self.paged_ssd_cache, "get_gdn_checkpoint_file", None
                )
                if not callable(signature_builder) or (
                    not callable(sidecar_lookup_getter)
                    and not callable(sidecar_getter)
                ):
                    logger.warning(
                        "Split GDN cache enabled but sidecar manager API is unavailable"
                    )
                    return None

                cache_signature = signature_builder(
                    model_name=self.paged_cache.model_name,
                    num_layers=len(layer_cache_types or []),
                    block_size=self.block_size,
                    layer_cache_types=layer_cache_types,
                )
                chosen_idx = None
                chosen_payloads = None
                chosen_diagnostic = None
                for idx in range(len(block_table.block_ids) - 1, -1, -1):
                    block = self.paged_cache.allocated_blocks.get(
                        block_table.block_ids[idx]
                    )
                    if block is None or block.block_hash is None:
                        continue
                    # A corrupt reduced sidecar is forgotten and the same
                    # endpoint is retried once, allowing its legacy FP32
                    # candidate to be selected before walking back a block.
                    for _attempt in range(2):
                        lookup_diagnostic = None
                        if callable(sidecar_lookup_getter):
                            lookup = sidecar_lookup_getter(
                                block.block_hash, cache_signature
                            )
                            checkpoint_path = (
                                getattr(lookup, "file_path", None)
                                if lookup is not None
                                else None
                            )
                            if checkpoint_path is not None:
                                lookup_diagnostic = {
                                    "requested_state_dtype": getattr(
                                        lookup, "requested_state_dtype", None
                                    ),
                                    "effective_state_codec": getattr(
                                        lookup, "effective_state_codec", None
                                    ),
                                    "used_legacy_fp32_fallback": getattr(
                                        lookup,
                                        "used_legacy_fp32_fallback",
                                        None,
                                    ),
                                }
                        else:
                            checkpoint_path = sidecar_getter(
                                block.block_hash, cache_signature
                            )
                        if checkpoint_path is None:
                            break
                        if self._gdn_checkpoint_loader is None:
                            logger.warning(
                                "Split GDN cache enabled without checkpoint loader"
                            )
                            return None
                        load_started = time.perf_counter()
                        dequantizations_before = self._gdn_dequantization_count()
                        snapshot = self._gdn_checkpoint_loader(checkpoint_path)
                        dequantizations_after = self._gdn_dequantization_count()
                        load_latency_ms = (
                            time.perf_counter() - load_started
                        ) * 1000.0
                        if snapshot is None:
                            forgetter = getattr(
                                self.paged_ssd_cache,
                                "forget_gdn_checkpoint",
                                None,
                            )
                            if callable(forgetter):
                                forgetter(block.block_hash, cache_signature)
                            continue
                        candidate_payloads = self._validated_gdn_snapshot_layers(
                            snapshot, layer_cache_types
                        )
                        if candidate_payloads is None:
                            logger.warning(
                                "Ignoring structurally invalid recurrent checkpoint "
                                "for block %s",
                                block.block_id,
                            )
                            forgetter = getattr(
                                self.paged_ssd_cache,
                                "forget_gdn_checkpoint",
                                None,
                            )
                            if callable(forgetter):
                                forgetter(block.block_hash, cache_signature)
                            continue
                        chosen_idx = idx
                        chosen_payloads = candidate_payloads
                        chosen_endpoint_tokens = sum(
                            self.paged_cache.allocated_blocks[bid].token_count
                            for bid in block_table.block_ids[: idx + 1]
                            if bid in self.paged_cache.allocated_blocks
                        )
                        source_hash = block.block_hash
                        chosen_diagnostic = {
                            "chosen_endpoint_tokens": chosen_endpoint_tokens,
                            "checkpoint_load_latency_ms": round(load_latency_ms, 3),
                            "walkback_blocks": len(all_block_data) - idx - 1,
                            "source_block_hash": (
                                source_hash.hex()[:16]
                                if isinstance(source_hash, (bytes, bytearray))
                                else str(source_hash)[:16]
                            ),
                            "requested_state_dtype": (
                                lookup_diagnostic or {}
                            ).get("requested_state_dtype"),
                            "effective_state_codec": (
                                lookup_diagnostic or {}
                            ).get("effective_state_codec"),
                            "used_legacy_fp32_fallback": (
                                lookup_diagnostic or {}
                            ).get("used_legacy_fp32_fallback"),
                            "dequantized_state_count": (
                                max(
                                    0,
                                    dequantizations_after
                                    - dequantizations_before,
                                )
                                if dequantizations_before is not None
                                and dequantizations_after is not None
                                else None
                            ),
                        }
                        break
                    if chosen_idx is not None:
                        break

                if chosen_idx is None or chosen_payloads is None:
                    logger.info(
                        "Split GDN restore found no compatible recurrent checkpoint"
                    )
                    return None

                if chosen_idx + 1 < len(all_block_data):
                    self._gdn_checkpoint_walkbacks += (
                        len(all_block_data) - chosen_idx - 1
                    )
                    for bid in block_table.block_ids[chosen_idx + 1 :]:
                        self.paged_cache.free_block(bid)
                    all_block_data = all_block_data[: chosen_idx + 1]
                    block_table.block_ids = block_table.block_ids[: chosen_idx + 1]
                    block_table.num_tokens = sum(
                        self.paged_cache.allocated_blocks[bid].token_count
                        for bid in block_table.block_ids
                        if bid in self.paged_cache.allocated_blocks
                    )
                    valid_token_count = block_table.num_tokens
                    all_block_meta_states = all_block_meta_states[: chosen_idx + 1]

                if chosen_idx < len(all_block_meta_states):
                    last_block_meta_states = all_block_meta_states[chosen_idx]

                # Overlay only Arrays-family layer payloads on the endpoint
                # block. Sliceable KV layers continue to come exclusively
                # from the main hot/SSD block chain.
                endpoint_data = all_block_data[-1]
                endpoint_meta = list(last_block_meta_states or [])
                for layer_idx, (cache_data, meta_state) in chosen_payloads.items():
                    endpoint_data[layer_idx] = cache_data
                    while len(endpoint_meta) <= layer_idx:
                        endpoint_meta.append(())
                    endpoint_meta[layer_idx] = meta_state
                last_block_meta_states = endpoint_meta
                if all_block_meta_states:
                    all_block_meta_states[-1] = endpoint_meta
                # Publish the per-attempt diagnostic only after the snapshot
                # has passed structural validation and has been overlaid on
                # every Arrays-family layer.  A malformed sidecar must not be
                # observable as a successful restore.
                self._last_gdn_restore = chosen_diagnostic
                self._gdn_checkpoint_loads += 1
            # Get number of layers from first block
            num_layers = len(all_block_data[0])
            if num_layers == 0:
                return None

            # --- Pre-scan: walk-back truncation for non-sliceable caches ---
            # If the last loaded block has a placeholder for any non-sliceable
            # layer (ArraysCache/RotatingKVCache/non-sliceable CacheList), walk
            # backwards to find the latest block where ALL such layers carry
            # real state. This recovers intermediate blocks from multi-turn
            # conversations instead of rejecting the entire cache.
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

            # Reconstruct caches for each layer
            reconstructed_caches = []
            healed_tq_layers = 0

            for layer_idx in range(num_layers):
                # Determine cache type for this layer
                cache_type_name = "KVCache"
                if layer_cache_types and layer_idx < len(layer_cache_types):
                    cache_type_name = layer_cache_types[layer_idx]

                handler = CacheTypeRegistry.get_handler_by_class_name(cache_type_name)

                # === CacheList: dedicated branch (before standard 2-tuple unpack) ===
                if cache_type_name == "CacheList":
                    last_block_layer_data = all_block_data[-1][layer_idx]

                    # Placeholder detection (partial match → reject for
                    # non-sliceable CacheList, e.g. containing ArraysCache)
                    if (
                        isinstance(last_block_layer_data, tuple)
                        and len(last_block_layer_data) == 2
                        and hasattr(last_block_layer_data[0], "shape")
                        and last_block_layer_data[0].shape == (1,)
                    ):
                        logger.info(
                            f"CacheList layer {layer_idx}: partial prefix match "
                            f"detected (placeholder). Rejecting cache."
                        )
                        return None

                    # Each sub_state in block_data may be either:
                    # - a legacy 2-tuple ``(keys, values)``, or
                    # - an ``('__nstate__', class_name, [elements])`` marker
                    #   emitted by the N-tuple-aware extract path (preserves
                    #   PoolingCache's full 3-tuple state).
                    # _sub_state_elements normalizes both to a raw element
                    # list so downstream concat / unpack does not have to
                    # branch on marker shape.
                    def _sub_state_elements(sub_state):
                        if (
                            isinstance(sub_state, tuple)
                            and len(sub_state) >= 3
                            and isinstance(sub_state[0], str)
                            and sub_state[0] == "__nstate__"
                        ):
                            return list(sub_state[2])
                        if isinstance(sub_state, (list, tuple)) and len(sub_state) >= 1:
                            return list(sub_state)
                        return None

                    def _sub_state_class(sub_state):
                        if (
                            isinstance(sub_state, tuple)
                            and len(sub_state) >= 3
                            and sub_state[0] == "__nstate__"
                        ):
                            return sub_state[1]
                        return None

                    # Collect CacheList data from all blocks that have List[sub_state].
                    # Per-member blocks arrive as ``('__cache_list_pm__', [subs])``
                    # (re-tagged by load_block from sidecar metadata); unwrap
                    # and remember which blocks used the per-member layout.
                    cl_block_data = []
                    cl_block_pm_flags = []
                    for block_data in all_block_data:
                        bd = block_data[layer_idx]
                        bd_pm = (
                            isinstance(bd, tuple)
                            and len(bd) == 2
                            and isinstance(bd[0], str)
                            and bd[0] == "__cache_list_pm__"
                        )
                        if bd_pm:
                            bd = bd[1]
                        if isinstance(bd, list) and all(
                            _sub_state_elements(t) is not None for t in bd
                        ):
                            cl_block_data.append(bd)
                            cl_block_pm_flags.append(bd_pm)

                    if not cl_block_data:
                        logger.error(
                            f"CacheList layer {layer_idx}: no valid block data found"
                        )
                        return None

                    # Determine sub-cache count from first valid block
                    num_sub_caches = len(cl_block_data[0])

                    # Per-sub-cache class dispatch: sliceable sub-caches
                    # (KVCache) concatenate per-block slices into the full
                    # sequence; non-sliceable sub-caches (RotatingKVCache,
                    # PoolingCache, ArraysCache, BatchPoolingCache) keep
                    # the last block's full state, since each saved block
                    # already snapshots the cache up to its boundary.
                    sub_class_names_for_layer: list[str] = []
                    if (
                        last_block_meta_states
                        and layer_idx < len(last_block_meta_states)
                        and isinstance(last_block_meta_states[layer_idx], (list, tuple))
                        and len(last_block_meta_states[layer_idx]) >= 1
                        and isinstance(
                            last_block_meta_states[layer_idx][0], (list, tuple)
                        )
                    ):
                        sub_class_names_for_layer = list(
                            last_block_meta_states[layer_idx][0]
                        )

                    non_sliceable_sub_classes = {
                        "PoolingCache",
                        "ArraysCache",
                        "SizedArraysCache",
                        "BatchPoolingCache",
                    }

                    def _is_non_sliceable_sub_class(class_name: str) -> bool:
                        return (
                            class_name in non_sliceable_sub_classes
                            or CacheTypeRegistry.is_rotating_family(class_name)
                        )

                    # The store path (_extract_block_tensor_slice) picks ONE
                    # storage mode for the whole layer: per-block slices only
                    # when every sub-state is a >=2-element tuple of 4D
                    # sequence tensors and no sub is a known non-sliceable
                    # class; otherwise EVERY block stores the full cumulative
                    # state of ALL subs at that block's boundary. Restore
                    # mirrors that layer-level decision here. Dispatching per
                    # sub (concat KVCache subs, last-block the rest) silently
                    # concatenated a mixed CacheList's cumulative KV snapshots
                    # into a duplicated sequence (e.g. inkling-style
                    # CacheList(KVCache, ArraysCache): 4+8+12 tokens instead
                    # of 12).
                    any_non_sliceable_sub = any(
                        _is_non_sliceable_sub_class(
                            sub_class_names_for_layer[j]
                            if j < len(sub_class_names_for_layer)
                            else ""
                        )
                        for j in range(num_sub_caches)
                    )
                    last_block_elements = [
                        _sub_state_elements(s) for s in cl_block_data[-1]
                    ]
                    stored_slice_mode = not any_non_sliceable_sub and all(
                        elems is not None
                        and len(elems) >= 2
                        and hasattr(elems[0], "shape")
                        and len(elems[0].shape) == 4
                        for elems in last_block_elements
                    )

                    pm_mode = any(cl_block_pm_flags)
                    if pm_mode and not all(cl_block_pm_flags):
                        # A chain mixing legacy cumulative blocks with
                        # per-member blocks cannot be restored coherently:
                        # concatenating a cumulative KV snapshot duplicates
                        # the sequence, and last-blocking a per-member chain
                        # drops KV. Reject; the request re-prefills.
                        logger.info(
                            f"CacheList layer {layer_idx}: mixed legacy/"
                            f"per-member block chain. Rejecting cache."
                        )
                        return None

                    if pm_mode:
                        # Per-member restore: sliceable KV subs concatenate
                        # their per-block slices along the sequence axis;
                        # boundary members (ArraysCache-style) take the last
                        # matched block's state, which snapshots the cache at
                        # exactly that boundary. Never apply the generic
                        # column concat to boundary members — their 3D conv
                        # tensors would concatenate along channels.
                        concatenated_sub_states = []
                        pm_slice_sub_indices: list[int] = []
                        for j in range(num_sub_caches):
                            per_block_elements = [
                                _sub_state_elements(bd[j]) for bd in cl_block_data
                            ]
                            if any(e is None for e in per_block_elements):
                                logger.error(
                                    f"CacheList layer {layer_idx}: invalid "
                                    f"per-member sub {j} state"
                                )
                                return None
                            elems_last = per_block_elements[-1]
                            class_j = (
                                sub_class_names_for_layer[j]
                                if j < len(sub_class_names_for_layer)
                                else ""
                            )
                            is_slice_sub = (
                                class_j in _PM_SLICEABLE_SUB_CLASSES
                                if class_j
                                else (
                                    len(elems_last) >= 2
                                    and hasattr(elems_last[0], "shape")
                                    and len(elems_last[0].shape) == 4
                                )
                            )
                            if is_slice_sub:
                                pm_slice_sub_indices.append(j)
                            if is_slice_sub and len(per_block_elements) > 1:
                                cat_elements = []
                                for k in range(len(elems_last)):
                                    column = [pb[k] for pb in per_block_elements]
                                    first = column[0]
                                    if (
                                        hasattr(first, "shape")
                                        and len(first.shape) == 4
                                        and all(
                                            hasattr(c, "shape")
                                            and len(c.shape) == 4
                                            and c.shape[:2] == first.shape[:2]
                                            for c in column
                                        )
                                    ):
                                        if any(d == 0 for d in first.shape):
                                            shape = list(first.shape)
                                            shape[2] = sum(
                                                c.shape[2] for c in column
                                            )
                                            cat_elements.append(
                                                mx.zeros(
                                                    tuple(shape),
                                                    dtype=first.dtype,
                                                )
                                            )
                                        else:
                                            cat_elements.append(
                                                mx.concatenate(column, axis=2)
                                            )
                                    else:
                                        # Non-sequence element (BatchKVCache
                                        # offset/padding metadata): last
                                        # block's value.
                                        cat_elements.append(column[-1])
                                concatenated_sub_states.append(tuple(cat_elements))
                            else:
                                concatenated_sub_states.append(tuple(elems_last))

                        # Defense-in-depth (#2550 review): the restored KV
                        # length must equal the matched token count. A chain
                        # with a skipped/short block would otherwise hand the
                        # scheduler a shorter cache than block_table.num_tokens
                        # claims, silently skipping the difference.
                        for j in pm_slice_sub_indices:
                            sub_state = concatenated_sub_states[j]
                            if (
                                not sub_state
                                or not hasattr(sub_state[0], "shape")
                                or len(sub_state[0].shape) != 4
                                or sub_state[0].shape[2] != block_table.num_tokens
                            ):
                                got = (
                                    sub_state[0].shape[2]
                                    if sub_state
                                    and hasattr(sub_state[0], "shape")
                                    and len(sub_state[0].shape) == 4
                                    else "?"
                                )
                                logger.warning(
                                    f"CacheList layer {layer_idx}: per-member "
                                    f"restored KV length {got} != expected "
                                    f"{block_table.num_tokens} tokens. "
                                    f"Rejecting cache."
                                )
                                return None
                    elif len(cl_block_data) > 1 and stored_slice_mode:
                        # Per-block slices: concatenate every sub along the
                        # sequence axis into the full sequence.
                        concatenated_sub_states = []
                        for j in range(num_sub_caches):
                            per_block_elements = [
                                _sub_state_elements(bd[j]) for bd in cl_block_data
                            ]
                            num_elems = len(per_block_elements[-1])
                            cat_elements = []
                            for k in range(num_elems):
                                # Concat sequence-axis tensors; non-sequence
                                # elements (axis-2 mismatch or scalars) take
                                # the last block's value.
                                column = [pb[k] for pb in per_block_elements]
                                first = column[0]
                                if (
                                    hasattr(first, "shape")
                                    and len(first.shape) >= 3
                                    and all(
                                        hasattr(c, "shape")
                                        and c.shape[:2] == first.shape[:2]
                                        for c in column
                                    )
                                ):
                                    if any(d == 0 for d in first.shape):
                                        shape = list(first.shape)
                                        shape[2] = sum(c.shape[2] for c in column)
                                        cat_elements.append(mx.zeros(tuple(shape)))
                                    else:
                                        cat_elements.append(
                                            mx.concatenate(column, axis=2)
                                        )
                                else:
                                    cat_elements.append(column[-1])
                            concatenated_sub_states.append(tuple(cat_elements))
                    else:
                        # Cumulative snapshots (or a single block): the last
                        # matched block already holds the complete state of
                        # every sub at its boundary. PoolingCache V4 markers
                        # are the exception: their append-only pooled tensor
                        # is stored as an absolute-range delta per block and
                        # must be rebuilt in order. A legacy full snapshot may
                        # appear before V4 deltas in an existing cache chain;
                        # it becomes the reconstruction base.
                        concatenated_sub_states = []
                        for j, last_elements in enumerate(last_block_elements):
                            has_pooling_deltas = any(
                                _sub_state_class(bd[j]) == POOLING_CACHE_DELTA_CLASS
                                for bd in cl_block_data
                            )
                            if not has_pooling_deltas:
                                concatenated_sub_states.append(tuple(last_elements))
                                continue

                            pooled_parts = []
                            pooled_length = 0
                            last_pooling_state = None
                            valid_pooling_chain = True
                            for block_cache_list in cl_block_data:
                                sub_state = block_cache_list[j]
                                elements = _sub_state_elements(sub_state)
                                marker_class = _sub_state_class(sub_state)
                                if elements is None:
                                    valid_pooling_chain = False
                                    break

                                if marker_class != POOLING_CACHE_DELTA_CLASS:
                                    if len(elements) < 3:
                                        valid_pooling_chain = False
                                        break
                                    if elements[2] is None:
                                        pooled_parts = []
                                        pooled_length = 0
                                    elif (
                                        hasattr(elements[2], "shape")
                                        and len(elements[2].shape) >= 2
                                    ):
                                        pooled_parts = [elements[2]]
                                        pooled_length = int(elements[2].shape[1])
                                    else:
                                        valid_pooling_chain = False
                                        break
                                    last_pooling_state = list(elements)
                                    continue

                                if (
                                    len(elements) not in (4, 6)
                                    or not hasattr(elements[2], "shape")
                                    or len(elements[2].shape) < 2
                                    or not hasattr(elements[-1], "tolist")
                                ):
                                    valid_pooling_chain = False
                                    break
                                try:
                                    delta_range = elements[-1].tolist()
                                    delta_start, delta_end = (
                                        int(delta_range[0]),
                                        int(delta_range[1]),
                                    )
                                except (IndexError, TypeError, ValueError):
                                    valid_pooling_chain = False
                                    break
                                if (
                                    delta_start != pooled_length
                                    or delta_end < delta_start
                                    or int(elements[2].shape[1])
                                    != delta_end - delta_start
                                ):
                                    valid_pooling_chain = False
                                    break
                                pooled_parts.append(elements[2])
                                pooled_length = delta_end
                                last_pooling_state = list(elements[:-1])

                            if (
                                not valid_pooling_chain
                                or not pooled_parts
                                or last_pooling_state is None
                            ):
                                logger.info(
                                    "CacheList layer %d sub-cache %d: invalid "
                                    "PoolingCache delta chain. Rejecting cache.",
                                    layer_idx,
                                    j,
                                )
                                return None

                            pooled = (
                                pooled_parts[0]
                                if len(pooled_parts) == 1
                                else mx.concatenate(pooled_parts, axis=1)
                            )
                            last_pooling_state[2] = pooled
                            concatenated_sub_states.append(tuple(last_pooling_state))

                    # Build meta_state with correct offsets for reconstructed
                    # sequence length (may differ from original if partial match)
                    meta_state = None
                    if last_block_meta_states and layer_idx < len(
                        last_block_meta_states
                    ):
                        meta_state = last_block_meta_states[layer_idx]

                    if (
                        meta_state
                        and isinstance(meta_state, (list, tuple))
                        and len(meta_state) >= 2
                    ):
                        # Adjust sub-cache offsets to actual concatenated seq_len.
                        # Sliceable sub-caches (KVCache) need offset replaced
                        # with the post-concat seq_len. Non-sliceable
                        # sub-caches (RotatingKVCache, PoolingCache, ...)
                        # keep their last-block meta intact — the sliding
                        # window offset / pool length already encode the
                        # boundary state from the original snapshot.
                        class_names = meta_state[0]
                        adjusted_sub_metas = []
                        for j in range(num_sub_caches):
                            orig_sub_meta = (
                                meta_state[1][j] if j < len(meta_state[1]) else ""
                            )
                            sub_class = (
                                sub_class_names_for_layer[j]
                                if j < len(sub_class_names_for_layer)
                                else ""
                            )
                            if _is_non_sliceable_sub_class(sub_class):
                                adjusted_sub_metas.append(
                                    orig_sub_meta if orig_sub_meta else ""
                                )
                                continue
                            sub_elements = concatenated_sub_states[j]
                            actual_seq_len = None
                            if (
                                sub_elements
                                and len(sub_elements) > 0
                                and hasattr(sub_elements[0], "shape")
                                and len(sub_elements[0].shape) >= 3
                            ):
                                actual_seq_len = sub_elements[0].shape[2]
                            if (
                                actual_seq_len is not None
                                and isinstance(orig_sub_meta, (list, tuple))
                                and len(orig_sub_meta) > 0
                            ):
                                adjusted_sub_metas.append(
                                    (actual_seq_len,) + tuple(orig_sub_meta[1:])
                                )
                            else:
                                adjusted_sub_metas.append(
                                    orig_sub_meta if orig_sub_meta else ""
                                )
                        meta_state = (class_names, adjusted_sub_metas)

                    cache = handler.reconstruct_cache(
                        {"sub_states": concatenated_sub_states}, meta_state
                    )
                    if cache is None:
                        logger.error(
                            f"CacheList layer {layer_idx}: reconstruction failed"
                        )
                        return None
                    reconstructed_caches.append(cache)
                    continue

                # === TurboQuant KV: payload-driven per-block handling ===
                # One block chain can mix payload formats: blocks stored
                # while TurboQuant KV conversion was active carry tagged
                # ('__turboquant_v2__', (ks, vs)) NamedTuple states, while
                # blocks stored without the conversion (chains written by
                # versions that skipped it on chunked-prefill completion,
                # or stored under a different TQ setting) carry plain dense
                # (keys, values) tensors in the SAME chain via dedup.
                # The per-block layer_cache_types mismatch check above
                # truncates such chains when every block's type metadata is
                # present and accurate, but block metadata is optional at
                # load time (missing/corrupt metadata files, failed
                # layer_cache_types round-trips, chains typed from a later
                # block when the first block lacks types) — so the payload
                # itself is the ground truth. Each block must be dispatched
                # on its own payload format: typing the whole chain from
                # the chain-level metadata feeds dense arrays into
                # TurboQuant _concat_state (AttributeError: 'array' object
                # has no attribute 'norms') and rejects the hit. The
                # reverse mixing direction (dense-typed chain with
                # TQ-tagged blocks appended later) is routed here by payload
                # scan for the same reason.
                if cache_type_name in (
                    "TurboQuantKVCache",
                    "BatchTurboQuantKVCache",
                ) or (
                    handler.supports_block_slicing
                    and self._layer_has_turboquant_payload(all_block_data, layer_idx)
                ):
                    entries: list[tuple[int, str, Any, Any]] = []
                    for block_idx, block_data in enumerate(all_block_data):
                        if layer_idx >= len(block_data):
                            continue
                        bd = block_data[layer_idx]
                        if not (isinstance(bd, tuple) and len(bd) == 2):
                            logger.warning(
                                f"TQ layer {layer_idx}: block {block_idx} has "
                                f"unsupported payload type "
                                f"{type(bd).__name__}. Rejecting cache hit."
                            )
                            return None
                        if isinstance(bd[0], str) and bd[0] == "__turboquant_v2__":
                            ks, vs = bd[1]
                            entries.append((block_idx, "tq", ks, vs))
                        elif self._is_placeholder_state(bd):
                            # Empty-slice placeholder written by the TQ store
                            # path: this block carries no KV for the layer,
                            # so the chain cannot be reconstructed.
                            logger.info(
                                f"TQ layer {layer_idx}: block {block_idx} is "
                                f"an empty-slice placeholder. Rejecting cache "
                                f"hit; request will reprocess from scratch."
                            )
                            return None
                        elif (
                            hasattr(bd[0], "shape")
                            and len(bd[0].shape) == 4
                            and hasattr(bd[1], "shape")
                        ):
                            # Plain dense KV slice stored without TQ
                            # conversion (e.g. chunked-prefill completion).
                            entries.append((block_idx, "plain", bd[0], bd[1]))
                        else:
                            logger.warning(
                                f"TQ layer {layer_idx}: block {block_idx} "
                                f"payload is neither a tagged TurboQuant "
                                f"state nor a dense KV slice. Rejecting "
                                f"cache hit."
                            )
                            return None
                    if not entries:
                        logger.debug(f"TQ layer {layer_idx}: no block data")
                        return None

                    # Group consecutive TQ blocks that share (bits, seed) so
                    # each run concatenates in quantized form and dequantizes
                    # once. A homogeneous TQ chain stays a single run — one
                    # concat + one codec rebuild, restored as a quantized
                    # TurboQuantKVCache (upstream #1842: restored long-context
                    # prefixes stay quantized, avoiding full-state
                    # materialization). Mixed chains (plain dense blocks, or
                    # runs with differing (bits, seed)) are dequantized per run
                    # and merged into a dense KVCache, cast back to the dense
                    # blocks' stored dtype: dequantize() emits float32, and an
                    # uncast fp32 layer would silently promote the whole merged
                    # batch cache (2x KV memory) on servers where the TurboQuant
                    # requantize epilogue does not run (TQ disabled, skip_last).
                    # The healed request trades TurboQuant's memory savings for
                    # that one restored prefix.
                    try:
                        from mlx_lm.models.cache import KVCache

                        groups: list[tuple[str, Any, Any]] = []
                        for block_idx, kind, part_k, part_v in entries:
                            if kind == "plain":
                                groups.append(("plain", part_k, part_v))
                                continue
                            params = self._tq_block_params(
                                all_block_meta_states,
                                first_block_meta_states,
                                block_idx,
                                layer_idx,
                            )
                            if params is None:
                                logger.warning(
                                    f"TQ layer {layer_idx}: block {block_idx} "
                                    f"has no TurboQuant (bits, seed) metadata "
                                    f"and no server-configured KV bit depth "
                                    f"is available. Rejecting cache hit "
                                    f"instead of guessing the codec width."
                                )
                                return None
                            if (
                                groups
                                and groups[-1][0] == "tq"
                                and groups[-1][2] == params
                            ):
                                groups[-1][1].append((part_k, part_v))
                            else:
                                groups.append(("tq", [(part_k, part_v)], params))

                        # Homogeneous TQ chain (single quantized run, no dense
                        # blocks): keep it quantized end-to-end like the
                        # pre-payload-driven path, so restored prefixes don't
                        # materialize full state.
                        if len(groups) == 1 and groups[0][0] == "tq":
                            from mlx_vlm.turboquant import TurboQuantKVCache

                            from ..turboquant_kv import (
                                _concat_state_token_axis,
                                _rebuild_codecs,
                                _state_length,
                            )

                            tq_bits, tq_seed = groups[0][2]
                            run_states = groups[0][1]
                            cat_ks = _concat_state_token_axis(
                                [ks for ks, _ in run_states]
                            )
                            cat_vs = _concat_state_token_axis(
                                [vs for _, vs in run_states]
                            )
                            tq = TurboQuantKVCache(bits=tq_bits, seed=tq_seed)
                            tq.keys = cat_ks
                            tq.values = cat_vs
                            tq.offset = _state_length(cat_ks)
                            _rebuild_codecs(tq, cat_ks, cat_vs)
                            reconstructed_caches.append(tq)
                            continue

                        key_parts = []
                        value_parts = []
                        dense_dtype = None
                        for group in groups:
                            if group[0] == "plain":
                                key_parts.append(group[1])
                                value_parts.append(group[2])
                                if dense_dtype is None:
                                    dense_dtype = group[1].dtype
                                continue
                            tq_bits, tq_seed = group[2]
                            dq = self._dequantize_tq_run(
                                group[1], tq_bits, tq_seed, layer_idx
                            )
                            if dq is None:
                                return None
                            key_parts.append(dq[0])
                            value_parts.append(dq[1])

                        keys = mx.concatenate(key_parts, axis=2)
                        values = mx.concatenate(value_parts, axis=2)
                        if dense_dtype is not None and keys.dtype != dense_dtype:
                            keys = keys.astype(dense_dtype)
                            values = values.astype(dense_dtype)
                        cache = KVCache()
                        cache.keys = keys
                        cache.values = values
                        cache.offset = keys.shape[2]
                        reconstructed_caches.append(cache)
                        healed_tq_layers += 1
                        if healed_tq_layers == 1:
                            n_tq = sum(1 for g in groups if g[0] == "tq")
                            n_plain = len(groups) - n_tq
                            logger.info(
                                f"Healed mixed-format TQ chain: layer "
                                f"{layer_idx} has {n_tq} quantized run(s) and "
                                f"{n_plain} dense block group(s); restoring "
                                f"affected layers as dense KVCache "
                                f"(dtype={keys.dtype})"
                            )
                    except Exception as e:
                        logger.error(
                            f"TQ layer {layer_idx}: reconstruction failed: {e}"
                        )
                        return None
                    continue

                # === Generic N-tuple sliceable cache: concatenate block slices ===
                last_block_layer_data = all_block_data[-1][layer_idx]
                if (
                    handler.supports_block_slicing
                    and isinstance(last_block_layer_data, tuple)
                    and len(last_block_layer_data) >= 3
                    and last_block_layer_data[0] == "__nstate__"
                ):
                    marker_class = last_block_layer_data[1] or cache_type_name
                    marker_handler = CacheTypeRegistry.get_handler_by_class_name(
                        marker_class
                    )
                    axis_info = marker_handler.get_state_axis_info()
                    layer_states = []
                    for block_data in all_block_data:
                        if layer_idx >= len(block_data):
                            logger.debug(
                                f"Layer {layer_idx}: missing block data for "
                                f"{marker_class}"
                            )
                            return None
                        block_layer_data = block_data[layer_idx]
                        if (
                            not isinstance(block_layer_data, tuple)
                            or len(block_layer_data) < 3
                            or block_layer_data[0] != "__nstate__"
                        ):
                            logger.debug(
                                f"Layer {layer_idx}: expected N-tuple block data "
                                f"for {marker_class}"
                            )
                            return None
                        elements = tuple(block_layer_data[2])
                        state_dict = {
                            "states": elements,
                            "cache_type": marker_class,
                        }
                        for info, elem in zip(axis_info, elements):
                            state_dict[info.name] = elem
                        layer_states.append(state_dict)

                    concat_state = marker_handler.concatenate_states(layer_states)
                    cache = marker_handler.reconstruct_cache(concat_state, None)
                    if cache is None:
                        logger.error(
                            f"Layer {layer_idx}: failed to reconstruct {marker_class}"
                        )
                        return None
                    reconstructed_caches.append(cache)
                    continue

                # === Generic N-tuple non-sliceable cache: use latest boundary ===
                if (
                    isinstance(last_block_layer_data, tuple)
                    and len(last_block_layer_data) >= 3
                    and last_block_layer_data[0] == "__nstate__"
                ):
                    marker_class = last_block_layer_data[1] or cache_type_name
                    elements = last_block_layer_data[2]
                    marker_handler = CacheTypeRegistry.get_handler_by_class_name(
                        marker_class
                    )
                    meta_state = None
                    if last_block_meta_states and layer_idx < len(
                        last_block_meta_states
                    ):
                        meta_state = last_block_meta_states[layer_idx]
                    if marker_handler.is_variable_length_state():
                        cache = marker_handler.reconstruct_cache(
                            {"states": list(elements)},
                            meta_state,
                            token_count=valid_token_count,
                        )
                    else:
                        cache = marker_handler.deserialize_state(
                            tuple(elements), meta_state
                        )
                    if cache is None:
                        logger.error(
                            f"Layer {layer_idx}: failed to reconstruct {marker_class}"
                        )
                        return None
                    reconstructed_caches.append(cache)
                    continue

                # Collect layer data from all blocks
                layer_states = []
                for block_data in all_block_data:
                    if layer_idx < len(block_data):
                        keys_slice, values_slice = block_data[layer_idx]
                        if keys_slice is not None and values_slice is not None:
                            layer_states.append(
                                {
                                    "keys": keys_slice,
                                    "values": values_slice,
                                }
                            )

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
                    if last_block_meta_states and layer_idx < len(
                        last_block_meta_states
                    ):
                        meta_state = last_block_meta_states[layer_idx]
                else:
                    # Sliceable caches (KVCache): first block's meta_state is fine
                    if first_block_meta_states and layer_idx < len(
                        first_block_meta_states
                    ):
                        meta_state = first_block_meta_states[layer_idx]

                # Reconstruct using appropriate handler
                if handler.supports_block_slicing:
                    # Standard concatenation for KVCache
                    concat_state = handler.concatenate_states(layer_states)
                    cache = handler.reconstruct_cache(concat_state, meta_state)
                else:
                    # Non-sliceable cache: use latest state
                    # States were stored as full state, use last one
                    latest_keys = layer_states[-1].get("keys")
                    latest_values = layer_states[-1].get("values")

                    if CacheTypeRegistry.is_rotating_family(cache_type_name):
                        # RotatingKVCache: strict last-block restore.
                        # If the last matched block is a placeholder, we only
                        # had a partial prefix hit and must reject.
                        if hasattr(latest_keys, "shape") and latest_keys.shape == (1,):
                            logger.info(
                                f"RotatingKVCache layer {layer_idx}: partial prefix "
                                f"match detected (placeholder in last matched "
                                f"block). Rejecting cache to prevent stale "
                                f"sliding-window state."
                            )
                            return None

                        latest_state = {
                            "keys": latest_keys,
                            "values": latest_values,
                            "meta_state": meta_state,
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
                        if hasattr(latest_keys, "shape") and latest_keys.shape == (1,):
                            logger.info(
                                f"ArraysCache layer {layer_idx}: partial prefix "
                                f"match detected (placeholder in last matched "
                                f"block). Rejecting cache to prevent stale GDN "
                                f"state. Request will reprocess from scratch."
                            )
                            return None

                        # Exact match: last block has full state
                        latest_state = {
                            "states": [latest_keys, latest_values],
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

            # Verify KVCache offset consistency across KVCache-typed layers.
            # All KVCache layers must have the same offset (they process
            # the same tokens). A mismatch causes broadcast_shapes errors
            # when the model creates a single attention mask from one layer
            # and applies it to all attention layers.
            # NOTE: only check layers explicitly typed as 'KVCache'.
            # RotatingKVCache also has 'offset' but its meaning differs
            # (total tokens ever processed, not buffer size), so mixing
            # them would produce false positives.
            if layer_cache_types:
                kv_offsets = set()
                for idx, c in enumerate(reconstructed_caches):
                    if (
                        idx < len(layer_cache_types)
                        and layer_cache_types[idx] == "KVCache"
                        and hasattr(c, "offset")
                        and isinstance(getattr(c, "offset", None), int)
                    ):
                        kv_offsets.add(c.offset)
                if len(kv_offsets) > 1:
                    logger.warning(
                        f"KVCache offset inconsistency after reconstruction: "
                        f"{kv_offsets}. Rejecting cache to prevent "
                        f"broadcast_shapes errors."
                    )
                    return None

            logger.debug(
                f"Reconstructed cache from tiered cache: {len(reconstructed_caches)} layers, "
                f"{block_table.num_tokens} tokens from {len(block_table.block_ids)} blocks"
            )

            return reconstructed_caches

        except Exception as e:
            logger.warning(f"Failed to reconstruct cache: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            return None

    @staticmethod
    def _layer_has_turboquant_payload(
        all_block_data: list[list[Any]],
        layer_idx: int,
    ) -> bool:
        """True if any block's payload for this layer is TurboQuant-tagged.

        Used to route dense-typed (KVCache) layers whose chain later
        accumulated TQ-converted blocks into the payload-driven TurboQuant
        reconstruction path. layer_cache_types comes from the first block
        only, so the chain-level type cannot detect this mixing.
        """
        for block_data in all_block_data:
            if layer_idx >= len(block_data):
                continue
            bd = block_data[layer_idx]
            if (
                isinstance(bd, tuple)
                and len(bd) == 2
                and isinstance(bd[0], str)
                and bd[0] == "__turboquant_v2__"
            ):
                return True
        return False

    def _tq_block_params(
        self,
        all_block_meta_states: list[Any],
        first_block_meta_states: Any,
        block_idx: int,
        layer_idx: int,
    ) -> tuple[float, int] | None:
        """Resolve (bits, seed) for one block's TurboQuant payload.

        Prefers the block's own per-block meta_state (TurboQuantKVCache
        stores ``(offset, bits, seed)``; mixed chains can carry different
        parameters per block), then the first block's chain-level meta, then
        the server-configured TurboQuant KV bit depth (blocks admitted for
        reconstruction already passed the bit-depth eligibility check, and
        seed 0 is the universal codec default). Returns None when none of
        these resolve: rebuilding a codec at a guessed width dequantizes to
        plausible-but-wrong tensors — silent output corruption after a cache
        hit — so the caller must reject the hit and re-prefill instead.
        """
        candidates = []
        if block_idx < len(all_block_meta_states):
            bms = all_block_meta_states[block_idx]
            if bms and layer_idx < len(bms):
                candidates.append(bms[layer_idx])
        if first_block_meta_states and layer_idx < len(first_block_meta_states):
            candidates.append(first_block_meta_states[layer_idx])
        for ms in candidates:
            if isinstance(ms, (list, tuple)) and len(ms) >= 3:
                try:
                    return float(ms[1]), int(ms[2])
                except (TypeError, ValueError):
                    continue
        expected_bits = getattr(
            self.paged_ssd_cache, "_expected_turboquant_kv_bits", None
        )
        if isinstance(expected_bits, (int, float)):
            from mlx_vlm.turboquant import DEFAULT_TURBOQUANT_SEED

            return float(expected_bits), int(DEFAULT_TURBOQUANT_SEED)
        return None

    def _dequantize_tq_run(
        self,
        run_states: list[tuple[Any, Any]],
        bits: float,
        seed: int,
        layer_idx: int,
    ) -> tuple[Any, Any] | None:
        """Dequantize a run of consecutive TurboQuant block slices to dense KV.

        Concatenates the quantized (ks, vs) states in block order and
        dequantizes once, with codecs rebuilt deterministically from
        (bits, seed). TurboQuant states are per-token (no cross-token
        coupling), so dequantizing a run equals dequantizing its blocks
        individually and concatenating — which makes payload-driven
        per-block reconstruction valid.

        Returns:
            (keys, values) dense float32 arrays, or None when the run cannot
            be decoded (caller rejects the cache hit).
        """
        from mlx_vlm.turboquant import TurboQuantKVCache

        from ..turboquant_kv import _concat_state_token_axis, _rebuild_codecs

        try:
            cat_ks = _concat_state_token_axis([ks for ks, _ in run_states])
            cat_vs = _concat_state_token_axis([vs for _, vs in run_states])
            tq = TurboQuantKVCache(bits=bits, seed=seed)
            _rebuild_codecs(tq, cat_ks, cat_vs)
            return tq.dequantize(cat_ks, cat_vs)
        except Exception as e:
            logger.error(
                f"TQ layer {layer_idx}: failed to dequantize "
                f"{len(run_states)} block state(s) "
                f"(bits={bits}, seed={seed}): {e}"
            )
            return None

    def _fallback_reconstruct_layer(
        self,
        layer_states: list[dict[str, Any]],
        cache_type_name: str,
    ) -> Any | None:
        """
        Fallback layer reconstruction when handler fails.

        Args:
            layer_states: List of state dicts with 'keys' and 'values'
            cache_type_name: Name of the cache type

        Returns:
            Reconstructed cache object or None
        """
        # The fallback rebuilds a plain KVCache, which is the wrong cache
        # class for every other type: a rotating buffer or recurrent state
        # restored as KVCache carries wrong positions and merge-unsafe
        # state. When a non-KVCache handler declares failure, reject the
        # cached prefix instead of papering over it with a guessed rebuild.
        if cache_type_name != "KVCache":
            logger.warning(
                f"Handler reconstruction failed for {cache_type_name}; "
                f"rejecting the cached prefix instead of rebuilding it as "
                f"a plain KVCache."
            )
            return None
        try:
            # Collect keys and values
            layer_keys = [s["keys"] for s in layer_states if s.get("keys") is not None]
            layer_values = [
                s["values"] for s in layer_states if s.get("values") is not None
            ]

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
                    cache.offset = (
                        concat_keys.shape[2]
                        if len(concat_keys.shape) == 4
                        else concat_keys.shape[1]
                    )
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
        all_block_data: list[list[tuple[Any, Any]]],
        layer_cache_types: list[str] | None = None,
    ) -> tuple[int, int] | None:
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
                if layer_cache_types[layer_idx] != "KVCache":
                    continue
            # Guard against non-tuple formats (CacheList stores List[Tuple])
            if not isinstance(layer_data, tuple) or len(layer_data) != 2:
                continue
            keys, _ = layer_data
            if hasattr(keys, "shape") and len(keys.shape) == 4:
                return (keys.shape[1], keys.shape[3])

        return None

    def _create_empty_rotating_cache(
        self,
        meta_state: tuple | None = None,
        kvcache_offset: int = 0,
        kv_shape_ref: tuple[int, int] | None = None,
    ) -> Any | None:
        """
        Create an empty RotatingKVCache for partial prefix restore.

        Creates a RotatingKVCache with zero-length keys/values (not None) and
        offset matching the KVCache layers. This ensures:
        1. mlx-lm's empty() returns False → Continuation mode (not Fresh Start)
        2. Position IDs (RoPE) are correct for all layers
        3. The merge creates a zero-length buffer (not zero-filled) so that
           no phantom attention positions exist during window padding reprocessing

        Uses PrefillReadyRotatingKVCache (clamped size() by buffer length)
        so BatchRotatingKVCache.merge() never reads beyond the actual buffer
        and never sees zero-padded positions as valid attention keys.

        Args:
            meta_state: RotatingKVCache meta_state tuple (keep, max_size, offset, _idx).
            kvcache_offset: Offset to match KVCache layers (= restored token count).
            kv_shape_ref: (kv_heads, head_dim) from a KVCache layer for tensor shape.

        Returns:
            RotatingKVCache with zero-length keys/values, or None on failure.
        """
        if meta_state and len(meta_state) >= 2:
            keep = int(meta_state[0])
            max_size = int(meta_state[1])
        else:
            logger.warning(
                "Cannot create empty RotatingKVCache: meta_state missing or incomplete"
            )
            return None

        cache = PrefillReadyRotatingKVCache(max_size=max_size, keep=keep)
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
        cache_data: list[tuple[Any, Any]],
        layer_cache_types: list[str] | None = None,
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

        # Cache types that support per-block KV slicing. When cache type
        # metadata is present, do not let non-KV hybrid states define the
        # expected seq_len for KVCache layers.
        sliceable_types = {
            "KVCache",
            "BatchKVCache",
            "TurboQuantKVCache",
            "BatchTurboQuantKVCache",
            "MiniMaxM3KVCache",
        }
        non_sliceable_types = {
            "ArraysCache",
            "RotatingKVCache",
            "BatchRotatingKVCache",
            "CacheList",
            "MiniMaxM3BatchKVCache",
        }

        expected_seq_len = None

        for layer_idx, layer_data in enumerate(cache_data):
            try:
                # Determine cache type first to handle CacheList before tuple unpack
                cache_type = None
                if layer_cache_types and layer_idx < len(layer_cache_types):
                    cache_type = layer_cache_types[layer_idx]

                # CacheList: sub-cache list format, skip standard (keys, values) unpacking
                if cache_type == "CacheList":
                    # CacheList data is either List[Tuple] (last block) or Tuple (placeholder)
                    if isinstance(layer_data, list):
                        continue  # Sub-cache list — valid
                    # Fall through to standard check for placeholder (zeros tuple)

                if not isinstance(layer_data, (list, tuple)) or len(layer_data) < 2:
                    logger.debug(
                        f"Block validation failed: layer {layer_idx} has "
                        f"unsupported data type {type(layer_data).__name__}"
                    )
                    return False

                if (
                    isinstance(layer_data, tuple)
                    and len(layer_data) >= 3
                    and layer_data[0] == "__nstate__"
                ):
                    elements = layer_data[2]
                    if not isinstance(elements, (list, tuple)) or len(elements) < 2:
                        logger.debug(
                            f"Block validation failed: layer {layer_idx} has "
                            "invalid N-tuple state"
                        )
                        return False
                    keys, values = elements[0], elements[1]
                else:
                    keys, values = layer_data[0], layer_data[1]

                # Skip seq_len check for non-sliceable types (e.g., ArraysCache, RotatingKVCache)
                # This includes placeholder entries (1D tensors from non-last blocks)
                # used by the last-block-only RotatingKVCache storage strategy
                if cache_type in non_sliceable_types:
                    continue
                if layer_cache_types and cache_type not in sliceable_types:
                    continue

                # Check for None after non-sliceable N-tuple caches have been
                # accepted. Pooling-style states can legitimately store None
                # in their first elements.
                if keys is None or values is None:
                    logger.debug(
                        f"Block validation failed: layer {layer_idx} has None keys/values"
                    )
                    return False

                # Check shape consistency for sliceable types (KVCache, RotatingKVCache)
                if hasattr(keys, "shape") and len(keys.shape) >= 3:
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
        tokens: list[int],
        extra_keys: tuple[Any, ...] | None = None,
    ) -> tuple[int, tuple[int, ...], int, list[bytes]] | None:
        """Find best matching prefix in the index.

        Returns:
            Tuple of (prefix_len, block_ids, num_blocks, chain_hashes) where
            chain_hashes holds the per-block chain hash for each matched
            block, or None when nothing matched. The hashes let the caller
            re-validate that the indexed blocks still hold the content they
            were indexed for before reusing them.
        """
        best_match = None
        best_len = 0

        parent_hash = b""
        prefix_len = 0
        num_blocks = 0
        chain_hashes: list[bytes] = []

        for start in range(0, len(tokens), self.block_size):
            end = min(start + self.block_size, len(tokens))
            block_tokens = tokens[start:end]
            if not block_tokens:
                break

            parent_hash = compute_block_hash(
                parent_hash,
                block_tokens,
                extra_keys=extra_keys,
                model_name=self.paged_cache.model_name,
            )
            prefix_len += len(block_tokens)
            num_blocks += 1
            chain_hashes.append(parent_hash)

            entry = self._prefix_index.get(parent_hash)
            if entry and entry[0] == prefix_len and prefix_len > best_len:
                best_match = entry
                best_len = prefix_len

        if best_match is None:
            return None
        return (*best_match, chain_hashes[: best_match[2]])

    def _update_prefix_index(
        self,
        tokens: list[int],
        block_ids: list[int],
        extra_keys: tuple[Any, ...] | None = None,
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
            if block is None:
                # Never index an unallocated block id: the entry could not
                # be dropped by the hash-lifecycle hooks (no block owns the
                # hash) and the chain past this point is unverifiable.
                break
            block_hash = block.block_hash
            if block_hash is None:
                block_hash = compute_block_hash(
                    parent_hash,
                    block_tokens,
                    extra_keys=extra_keys,
                    model_name=self.paged_cache.model_name,
                )
                block.block_hash = block_hash

            parent_hash = block_hash
            prefix_len += len(block_tokens)
            self._prefix_index[block_hash] = (
                prefix_len,
                tuple(block_ids[: i + 1]),
                i + 1,
            )

    def _on_block_hash_dropped(self, block_hash: bytes) -> None:
        """Drop the prefix-index entry for a dead block-hash association.

        Registered as PagedCacheManager.on_block_hash_dropped and invoked
        (possibly with the paged cache lock held) whenever a (hash -> block)
        association ceases to exist. dict.pop is GIL-atomic, so this is safe
        from any thread; it must not call back into the paged cache.
        """
        self._prefix_index.pop(block_hash, None)
        with self._mtp_prefix_snapshot_lock:
            self._mtp_prefix_snapshots.pop(block_hash, None)

    def _on_hash_map_cleared(self) -> None:
        """Drop all prefix-index entries after a wholesale hash-map clear."""
        self._prefix_index.clear()
        with self._mtp_prefix_snapshot_lock:
            self._mtp_prefix_snapshots.clear()

    def _mtp_prefix_chain_tip(
        self,
        tokens: list[int],
        boundary_tokens: int,
        *,
        extra_keys: tuple[Any, ...] | None = None,
        extra_key_token_start: int | None = None,
        extra_key_ranges: list[tuple[int, tuple[Any, ...]]] | None = None,
    ) -> bytes | None:
        """Return the ordinary full-block chain hash for an MTP sidecar.

        VLM requests can change hidden states without changing token ids.  The
        generic prefix cache supports per-range media keys, but
        ``compute_chain_hashes`` cannot reproduce that resolution from a lone
        boundary snapshot.  Fail closed for those two complex forms; text-only
        requests and a single request-wide ``extra_keys`` tuple remain exact.
        """
        boundary_tokens = int(boundary_tokens)
        if (
            boundary_tokens <= 0
            or boundary_tokens > len(tokens)
            or boundary_tokens % self.block_size != 0
            or extra_key_token_start is not None
            or extra_key_ranges is not None
        ):
            return None
        parent_hash: BlockHash | None = None
        for start in range(0, boundary_tokens, self.block_size):
            parent_hash = compute_block_hash(
                parent_hash,
                tokens[start : start + self.block_size],
                extra_keys=extra_keys,
                model_name=self.paged_cache.model_name,
            )
        return parent_hash

    def store_mtp_prefix_snapshot(
        self,
        tokens: list[int],
        boundary_tokens: int,
        snapshot: Any,
        *,
        extra_keys: tuple[Any, ...] | None = None,
        extra_key_token_start: int | None = None,
        extra_key_ranges: list[tuple[int, tuple[Any, ...]]] | None = None,
    ) -> bool:
        """Keep an exact Lightning-MTP history at one backbone block boundary.

        This is intentionally memory-only.  SSD persistence would require a
        versioned serialization contract for every model family's MTP cache;
        an unsupported or stale sidecar must merely fall back to unprimed MTP,
        never make the target model consume an invented history.
        """
        tip = self._mtp_prefix_chain_tip(
            tokens,
            boundary_tokens,
            extra_keys=extra_keys,
            extra_key_token_start=extra_key_token_start,
            extra_key_ranges=extra_key_ranges,
        )
        if tip is None or snapshot is None:
            return False
        with self._mtp_prefix_snapshot_lock:
            self._mtp_prefix_snapshots[tip] = (int(boundary_tokens), snapshot)
            self._mtp_prefix_snapshots.move_to_end(tip)
            while len(self._mtp_prefix_snapshots) > _MTP_PREFIX_SNAPSHOT_MAX_ENTRIES:
                self._mtp_prefix_snapshots.popitem(last=False)
        return True

    def restore_mtp_prefix_snapshot(
        self,
        tokens: list[int],
        boundary_tokens: int,
        *,
        extra_keys: tuple[Any, ...] | None = None,
        extra_key_token_start: int | None = None,
        extra_key_ranges: list[tuple[int, tuple[Any, ...]]] | None = None,
    ) -> Any | None:
        """Return the exact MTP sidecar matching a reconstructed trunk prefix."""
        tip = self._mtp_prefix_chain_tip(
            tokens,
            boundary_tokens,
            extra_keys=extra_keys,
            extra_key_token_start=extra_key_token_start,
            extra_key_ranges=extra_key_ranges,
        )
        if tip is None:
            return None
        # A snapshot published before completion is not reusable until the
        # matching backbone block has actually been stored and remains live.
        if self.paged_cache.cached_block_hash_to_block.get_block(tip) is None:
            return None
        with self._mtp_prefix_snapshot_lock:
            entry = self._mtp_prefix_snapshots.get(tip)
            if entry is None or entry[0] != int(boundary_tokens):
                return None
            self._mtp_prefix_snapshots.move_to_end(tip)
            return entry[1]

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
            partial_block_skips=self._partial_block_skips,
            partial_tokens_skipped=self._partial_tokens_skipped,
            block_size=self.block_size,
            last_partial_tokens_skipped=self._last_partial_tokens_skipped,
            last_tokens_to_next_block=self._last_tokens_to_next_block,
            tokens_matched_total=self._tokens_matched_total,
            tokens_requested_total=self._tokens_requested_total,
            exact_prefix_hits=self._exact_prefix_hits,
            exact_prefix_misses=self._exact_prefix_misses,
            exact_prefix_tokens_restored=self._exact_prefix_tokens_restored,
            exact_prefix_stores=self._exact_prefix_stores,
            exact_prefix_store_failures=self._exact_prefix_store_failures,
        )

    def get_stats_dict(self) -> dict[str, Any]:
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
            "hit_rate": (
                self._hits / (self._hits + self._misses)
                if (self._hits + self._misses) > 0
                else 0
            ),
            "tokens_saved": self._tokens_saved,
            "partial_block_skips": self._partial_block_skips,
            "partial_tokens_skipped": self._partial_tokens_skipped,
            "block_size": self.block_size,
            "last_partial_tokens_skipped": self._last_partial_tokens_skipped,
            "last_tokens_to_next_block": self._last_tokens_to_next_block,
            "tokens_matched_total": self._tokens_matched_total,
            "tokens_requested_total": self._tokens_requested_total,
            "exact_prefix_hits": self._exact_prefix_hits,
            "exact_prefix_misses": self._exact_prefix_misses,
            "exact_prefix_tokens_restored": self._exact_prefix_tokens_restored,
            "exact_prefix_stores": self._exact_prefix_stores,
            "exact_prefix_store_failures": self._exact_prefix_store_failures,
            "gdn_checkpoint_loads": self._gdn_checkpoint_loads,
            "gdn_checkpoint_walkbacks": self._gdn_checkpoint_walkbacks,
            "gdn_last_restore": (
                dict(self._last_gdn_restore)
                if self._last_gdn_restore is not None
                else None
            ),
            "active_requests": len(self._request_tables),
            **paged_stats,
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._hits = 0
        self._misses = 0
        self._tokens_saved = 0
        self._partial_block_skips = 0
        self._partial_tokens_skipped = 0
        self._tokens_matched_total = 0
        self._tokens_requested_total = 0
        self._last_partial_tokens_skipped = 0
        self._last_tokens_to_next_block = 0
        self._exact_prefix_hits = 0
        self._exact_prefix_misses = 0
        self._exact_prefix_tokens_restored = 0
        self._exact_prefix_stores = 0
        self._exact_prefix_store_failures = 0
        self._gdn_checkpoint_loads = 0
        self._gdn_checkpoint_walkbacks = 0
        self._last_gdn_restore = None
        self.paged_cache.reset_stats()

    def clear(self) -> int:
        """
        Clear all cached data.

        Returns:
            Number of entries cleared.
        """
        with self._mtp_prefix_snapshot_lock:
            mtp_snapshot_count = len(self._mtp_prefix_snapshots)
        cleared_count = (
            len(self._request_tables) + len(self._prefix_index) + mtp_snapshot_count
        )
        self._request_tables.clear()
        self._prefix_index.clear()
        with self._mtp_prefix_snapshot_lock:
            self._mtp_prefix_snapshots.clear()
        self.paged_cache.clear()
        self.reset_stats()
        return cleared_count

    def set_cold_restore_callback(
        self,
        callback: Callable[[int, bytes], bool] | None,
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

    def fetch(self, key: Any) -> tuple[Any | None, bool]:
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
