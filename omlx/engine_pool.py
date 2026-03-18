# SPDX-License-Identifier: Apache-2.0
"""
Engine pool for oMLX multi-model serving.

This module manages multiple model engines with LRU-based eviction
when memory limits are exceeded. It supports:

- Pre-load memory checking to ensure models fit before loading
- LRU eviction of least recently used models
- Model pinning to keep specific models always loaded
- BatchedEngine for all LLM models (continuous batching)
"""

from __future__ import annotations

import asyncio
import gc
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .model_settings import ModelSettingsManager

import mlx.core as mx

from .engine import BaseEngine, BatchedEngine
from .engine.embedding import EmbeddingEngine
from .engine.reranker import RerankerEngine
from .engine.vlm import VLMBatchedEngine
from .exceptions import (
    EnginePoolError,
    InsufficientMemoryError,
    ModelLoadingError,
    ModelNotFoundError,
    ModelTooLargeError,
)
from .model_discovery import DiscoveredModel, discover_models, format_size
from .engine_core import get_mlx_executor
from .scheduler import SchedulerConfig

logger = logging.getLogger(__name__)


@dataclass
class EngineEntry:
    """Per-model state in the engine pool."""

    model_id: str  # Directory name (e.g., "llama-3b")
    model_path: str  # Full path to model directory
    model_type: Literal["llm", "vlm", "embedding", "reranker"]  # Model type
    engine_type: Literal["batched", "simple", "embedding", "reranker", "vlm"]  # Engine type to use
    estimated_size: int  # Pre-calculated from safetensors (bytes)
    config_model_type: str = ""  # Raw model_type from config.json (e.g., "deepseekocr_2")
    engine: BaseEngine | EmbeddingEngine | RerankerEngine | None = None  # Loaded engine instance
    last_access: float = 0.0  # Timestamp for LRU (0 if never loaded)
    is_loading: bool = False  # Prevent concurrent loads
    is_pinned: bool = False  # Never evict if True
    abort_loading: bool = False  # Set by memory enforcer to abort in-progress load


class EnginePool:
    """
    Manages multiple model engines with LRU-based memory management.

    Features:
    - Pre-load memory checking (evict before load, not after)
    - LRU eviction when memory limit is exceeded
    - Model pinning to prevent eviction
    - Automatic engine type selection based on model type
    """

    def __init__(
        self,
        max_model_memory: int | None,
        scheduler_config: SchedulerConfig | None = None,
    ):
        """
        Initialize the engine pool.

        Args:
            max_model_memory: Maximum memory for loaded models in bytes,
                or None for no limit (disabled)
            scheduler_config: Configuration for BatchedEngine schedulers
        """
        self._entries: dict[str, EngineEntry] = {}
        self._lock = asyncio.Lock()
        self._max_model_memory = max_model_memory
        self._current_model_memory = 0
        self._scheduler_config = scheduler_config or SchedulerConfig()
        self._process_memory_enforcer: object | None = None  # Set by server
        self._settings_manager: object | None = None  # Set by server

    @property
    def max_model_memory(self) -> int | None:
        """Maximum memory for loaded models in bytes, or None if disabled."""
        return self._max_model_memory

    @property
    def current_model_memory(self) -> int:
        """Current memory used by loaded models in bytes."""
        return self._current_model_memory

    @property
    def model_count(self) -> int:
        """Total number of discovered models."""
        return len(self._entries)

    @property
    def loaded_model_count(self) -> int:
        """Number of currently loaded models."""
        return sum(1 for e in self._entries.values() if e.engine is not None)

    def discover_models(
        self, model_dirs: str | list[str], pinned_models: list[str] | None = None
    ) -> None:
        """
        Discover models in the specified directory or directories.

        Args:
            model_dirs: Path or list of paths to directories containing model subdirectories
            pinned_models: List of model IDs to pin (never evict)
        """
        from pathlib import Path

        from .model_discovery import discover_models_from_dirs

        if isinstance(model_dirs, str):
            dirs = [Path(model_dirs)]
        else:
            dirs = [Path(d) for d in model_dirs]

        if len(dirs) == 1:
            discovered = discover_models(dirs[0])
        else:
            discovered = discover_models_from_dirs(dirs)

        pinned_set = set(pinned_models or [])

        for model_id, info in discovered.items():
            existing = self._entries.get(model_id)
            if existing is not None and existing.engine is not None:
                # Loaded model: preserve runtime state, only update pinned flag
                existing.is_pinned = model_id in pinned_set
            else:
                # New or unloaded model: create fresh entry
                self._entries[model_id] = EngineEntry(
                    model_id=model_id,
                    model_path=info.model_path,
                    model_type=info.model_type,
                    engine_type=info.engine_type,
                    estimated_size=info.estimated_size,
                    config_model_type=getattr(info, "config_model_type", ""),
                    is_pinned=model_id in pinned_set,
                )

            if model_id in pinned_set:
                logger.info(f"Pinned model: {model_id}")

        # Remove entries no longer discovered and not loaded
        discovered_ids = set(discovered.keys())
        stale = [
            mid
            for mid in self._entries
            if mid not in discovered_ids and self._entries[mid].engine is None
        ]
        for mid in stale:
            del self._entries[mid]

        # Warn about pinned models not found
        found_models = set(self._entries.keys())
        for model_id in pinned_set:
            if model_id not in found_models:
                logger.warning(f"Pinned model not found: {model_id}")

        mem_display = "disabled" if self._max_model_memory is None else format_size(self._max_model_memory)
        logger.info(
            f"Discovered {len(self._entries)} models, "
            f"max memory: {mem_display}"
        )

    _MODEL_TYPE_TO_ENGINE: dict[str, str] = {
        "llm": "batched",
        "vlm": "vlm",
        "embedding": "embedding",
        "reranker": "reranker",
    }

    def apply_settings_overrides(
        self, settings_manager: "ModelSettingsManager"
    ) -> None:
        """Apply model_type_override from persisted settings to discovered entries."""
        for model_id, entry in self._entries.items():
            settings = settings_manager.get_settings(model_id)
            if settings.model_type_override:
                entry.model_type = settings.model_type_override
                entry.engine_type = self._MODEL_TYPE_TO_ENGINE.get(
                    settings.model_type_override, "batched"
                )
                logger.info(
                    f"Applied model_type override for {model_id}: "
                    f"type={entry.model_type}, engine={entry.engine_type}"
                )

    def get_model_ids(self) -> list[str]:
        """Get list of all discovered model IDs."""
        return list(self._entries.keys())

    def get_loaded_model_ids(self) -> list[str]:
        """Get list of currently loaded model IDs."""
        return [mid for mid, e in self._entries.items() if e.engine is not None]

    def get_entry(self, model_id: str) -> EngineEntry | None:
        """Get entry for a specific model, or None if not found."""
        return self._entries.get(model_id)

    def set_pinned(self, model_id: str, pinned: bool) -> bool:
        """
        Set the pinned status for a model.

        Args:
            model_id: The model ID to update
            pinned: Whether to pin (True) or unpin (False) the model

        Returns:
            True if successful, False if model not found.
        """
        entry = self._entries.get(model_id)
        if entry is None:
            return False
        entry.is_pinned = pinned
        return True

    def _case_insensitive_entry_match(self, name: str) -> str | None:
        """Find a model entry matching *name* case-insensitively.

        Returns the actual model_id if found, None otherwise.
        """
        lower = name.lower()
        for mid in self._entries:
            if mid.lower() == lower:
                return mid
        return None

    def resolve_model_id(self, model_id_or_alias: str, settings_manager) -> str:
        """Resolve a model alias to its actual model_id (directory name).

        Tries exact match in _entries first, then case-insensitive match,
        then scans model settings for alias match. If those fail and input
        contains a provider prefix (e.g. "omlx/my-model"), strips the prefix
        and retries. Returns the original string if no match found.
        """
        if model_id_or_alias in self._entries:
            return model_id_or_alias

        # Case-insensitive fallback
        ci_match = self._case_insensitive_entry_match(model_id_or_alias)
        if ci_match is not None:
            return ci_match

        all_settings = None
        if settings_manager is not None:
            all_settings = settings_manager.get_all_settings()
            for mid, ms in all_settings.items():
                if ms.model_alias and ms.model_alias == model_id_or_alias:
                    return mid

        # Strip provider prefix (e.g. "omlx/qwen3.5-35b" -> "qwen3.5-35b")
        if "/" in model_id_or_alias:
            stripped = model_id_or_alias.split("/", 1)[1]
            if stripped in self._entries:
                return stripped
            ci_match = self._case_insensitive_entry_match(stripped)
            if ci_match is not None:
                return ci_match
            if all_settings is not None:
                for mid, ms in all_settings.items():
                    if ms.model_alias and ms.model_alias == stripped:
                        return mid

        return model_id_or_alias

    async def get_engine(self, model_id: str) -> BaseEngine | EmbeddingEngine | RerankerEngine:
        """
        Get or load engine for the specified model.

        This method implements pre-load memory checking:
        1. Check if model is already loaded → return immediately
        2. Check if model is too large for memory limit → raise error
        3. Evict LRU models until there's enough space
        4. Load the model
        5. Return the engine

        Args:
            model_id: The model ID to get engine for

        Returns:
            The loaded engine (BaseEngine for LLM, EmbeddingEngine for embeddings)

        Raises:
            ModelNotFoundError: If model is not discovered
            ModelTooLargeError: If model exceeds memory limit
            InsufficientMemoryError: If can't free enough memory (all pinned)
            ModelLoadingError: If model is already being loaded
        """
        async with self._lock:
            entry = self._entries.get(model_id)
            if not entry:
                raise ModelNotFoundError(model_id, list(self._entries.keys()))

            # Already loaded - just update access time
            if entry.engine is not None:
                entry.last_access = time.time()
                return entry.engine

            # Check if model is too large for memory limit
            if (
                self._max_model_memory is not None
                and entry.estimated_size > self._max_model_memory
            ):
                raise ModelTooLargeError(
                    model_id, entry.estimated_size, self._max_model_memory
                )

            # Pre-load eviction: reserve 25% extra for KV cache headroom
            # so other models get evicted earlier, leaving room for context.
            # Always try to evict with headroom first. If all evictable models
            # are gone and the model still fits without headroom, allow it.
            # Skip entirely when model memory limit is disabled (None).
            if self._max_model_memory is not None:
                kv_headroom = int(entry.estimated_size * 0.25)
                required_with_headroom = entry.estimated_size + kv_headroom
                try:
                    await self._ensure_memory_available(required_with_headroom)
                except InsufficientMemoryError:
                    # Can't fit with headroom even after evicting everything possible.
                    # Fall back to weights-only if that fits.
                    if self._current_model_memory + entry.estimated_size <= self._max_model_memory:
                        logger.info(
                            f"Loading {model_id} without KV headroom "
                            f"(need {format_size(required_with_headroom)}, "
                            f"available {format_size(self._max_model_memory - self._current_model_memory)})"
                        )
                    else:
                        await self._ensure_memory_available(entry.estimated_size)

            # Check process memory limit before loading.
            # Try evicting LRU models first to free actual Metal memory.
            # max_bytes <= 0 means enforcement is disabled (no limit).
            if self._process_memory_enforcer is not None:
                enforcer = self._process_memory_enforcer
                if enforcer.max_bytes > 0:
                    while True:
                        current_active = mx.get_active_memory()
                        projected = current_active + entry.estimated_size
                        if projected <= enforcer.max_bytes:
                            break
                        # Try to evict an LRU model to free memory
                        victim = self._find_lru_victim()
                        if victim is not None:
                            logger.info(
                                f"Evicting '{victim}' to fit '{model_id}' "
                                f"within process memory limit "
                                f"({format_size(projected)} > "
                                f"{format_size(enforcer.max_bytes)})"
                            )
                            await self._unload_engine(victim)
                            continue
                        # No more victims — cannot fit
                        raise InsufficientMemoryError(
                            required=entry.estimated_size,
                            current=current_active,
                            message=(
                                f"Cannot load {model_id}: projected memory "
                                f"{format_size(projected)} would exceed process "
                                f"limit {format_size(enforcer.max_bytes)} "
                                f"(current: {format_size(current_active)}, "
                                f"model: {format_size(entry.estimated_size)})"
                            ),
                        )

            # Now load the model
            await self._load_engine(model_id)

            return self._entries[model_id].engine

    async def _ensure_memory_available(self, required: int) -> None:
        """
        Evict LRU models BEFORE loading to ensure we don't exceed memory limit.

        Args:
            required: Required memory in bytes

        Raises:
            InsufficientMemoryError: If can't free enough memory
        """
        if self._max_model_memory is None:
            return  # No model memory limit
        while self._current_model_memory + required > self._max_model_memory:
            victim = self._find_lru_victim()
            if not victim:
                raise InsufficientMemoryError(
                    required=required,
                    current=self._current_model_memory,
                    message=(
                        f"Cannot free enough memory. "
                        f"Need {format_size(required)}, "
                        f"current usage {format_size(self._current_model_memory)}, "
                        f"all loaded models are pinned."
                    ),
                )
            await self._unload_engine(victim)

    def _find_lru_victim(self) -> str | None:
        """
        Find the least recently used non-pinned loaded model.

        Returns:
            Model ID of the LRU victim, or None if all models are pinned
        """
        candidates = [
            (e.last_access, mid)
            for mid, e in self._entries.items()
            if e.engine is not None and not e.is_pinned
        ]
        if not candidates:
            return None
        candidates.sort()  # Sort by last_access (oldest first)
        return candidates[0][1]

    async def _unload_engine(self, model_id: str) -> None:
        """
        Immediately stop and unload an engine.

        This aborts any in-progress requests.

        Args:
            model_id: The model ID to unload
        """
        entry = self._entries.get(model_id)
        if not entry or entry.engine is None:
            return

        logger.info(f"Unloading model: {model_id} (immediate abort)")

        try:
            await entry.engine.stop()
        except Exception as e:
            logger.warning(f"Error stopping engine for {model_id}: {e}")

        # Release memory tracking
        self._current_model_memory -= entry.estimated_size

        # Clear engine reference
        entry.engine = None
        entry.last_access = 0.0

        # Force garbage collection to release memory.
        # Run mx.clear_cache on the global MLX executor to avoid concurrent
        # Metal operations with running engines. See issue #85.
        # Synchronize before clearing to prevent releasing Metal buffers
        # still referenced by in-flight command buffers. See issue #300.
        gc.collect()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            get_mlx_executor(), lambda: (mx.synchronize(), mx.clear_cache())
        )

        logger.info(
            f"Unloaded model: {model_id}, "
            f"memory usage: {format_size(self._current_model_memory)}"
        )

    async def _load_engine(self, model_id: str) -> None:
        """
        Load an engine for the specified model.

        Args:
            model_id: The model ID to load

        Raises:
            ModelLoadingError: If model is already being loaded
        """
        entry = self._entries[model_id]
        if entry.is_loading:
            raise ModelLoadingError(model_id)

        entry.is_loading = True
        entry.abort_loading = False
        try:
            logger.info(f"Loading model: {model_id}")

            # Retrieve per-model settings for post-load transforms
            model_settings = None
            if self._settings_manager is not None:
                model_settings = self._settings_manager.get_settings(model_id)

            # Create engine based on engine type
            if entry.engine_type == "embedding":
                # EmbeddingEngine for embedding models
                engine = EmbeddingEngine(model_name=entry.model_path)
            elif entry.engine_type == "reranker":
                # RerankerEngine for reranker models
                engine = RerankerEngine(model_name=entry.model_path)
            elif entry.engine_type == "vlm":
                # VLMBatchedEngine for vision-language models
                engine = VLMBatchedEngine(
                    model_name=entry.model_path,
                    scheduler_config=self._scheduler_config,
                )
            else:
                # BatchedEngine with continuous batching (default)
                engine = BatchedEngine(
                    model_name=entry.model_path,
                    scheduler_config=self._scheduler_config,
                    model_settings=model_settings,
                )

            try:
                await engine.start()
            except Exception as start_error:
                if entry.engine_type == "vlm":
                    # VLM loading failed — fall back to LLM (BatchedEngine)
                    logger.warning(
                        f"VLM loading failed for {model_id}, "
                        f"falling back to LLM: {start_error}"
                    )
                    try:
                        await engine.stop()
                    except Exception:
                        pass
                    gc.collect()
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        get_mlx_executor(),
                        lambda: (mx.synchronize(), mx.clear_cache()),
                    )

                    engine = BatchedEngine(
                        model_name=entry.model_path,
                        scheduler_config=self._scheduler_config,
                        model_settings=model_settings,
                    )
                    await engine.start()

                    entry.model_type = "llm"
                    entry.engine_type = "batched"
                    logger.info(
                        f"Successfully loaded {model_id} as LLM "
                        f"(fallback from VLM)"
                    )
                else:
                    raise

            # Check if memory enforcer requested abort during loading
            if entry.abort_loading:
                logger.warning(
                    f"Model load aborted by memory enforcer: {model_id}"
                )
                try:
                    await engine.stop()
                except Exception as e:
                    logger.warning(
                        f"Error stopping aborted engine for {model_id}: {e}"
                    )
                gc.collect()
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    get_mlx_executor(),
                    lambda: (mx.synchronize(), mx.clear_cache()),
                )
                raise ModelLoadingError(
                    f"Model {model_id} load aborted: "
                    f"process memory limit exceeded"
                )

            entry.engine = engine
            entry.last_access = time.time()
            self._current_model_memory += entry.estimated_size

            # Propagate memory limit to new engine's scheduler
            if self._process_memory_enforcer is not None:
                self._process_memory_enforcer._propagate_memory_limit()

            logger.info(
                f"Loaded model: {model_id} "
                f"(estimated: {format_size(entry.estimated_size)}, "
                f"total: {format_size(self._current_model_memory)})"
            )
        finally:
            entry.is_loading = False
            entry.abort_loading = False

    async def preload_pinned_models(self) -> None:
        """
        Preload all pinned models at startup.

        This ensures pinned models are always available.
        """
        pinned_models = [
            model_id for model_id, e in self._entries.items() if e.is_pinned
        ]

        for model_id in pinned_models:
            try:
                logger.info(f"Preloading pinned model: {model_id}")
                await self.get_engine(model_id)
            except Exception as e:
                logger.error(f"Failed to preload pinned model {model_id}: {e}")

    async def shutdown(self) -> None:
        """Shutdown all engines gracefully."""
        async with self._lock:
            for model_id in list(self._entries.keys()):
                entry = self._entries.get(model_id)
                if entry and entry.engine is not None:
                    try:
                        await self._unload_engine(model_id)
                    except Exception as e:
                        logger.error(f"Error unloading {model_id} during shutdown: {e}")

        logger.info("Engine pool shutdown complete")

    def get_status(self) -> dict:
        """
        Get pool status for monitoring endpoints.

        Returns:
            Dictionary with pool status information
        """
        return {
            "max_model_memory": self._max_model_memory,
            "current_model_memory": self._current_model_memory,
            "model_count": len(self._entries),
            "loaded_count": sum(1 for e in self._entries.values() if e.engine is not None),
            "models": [
                {
                    "id": mid,
                    "loaded": e.engine is not None,
                    "is_loading": e.is_loading,
                    "estimated_size": e.estimated_size,
                    "pinned": e.is_pinned,
                    "engine_type": e.engine_type,
                    "model_type": e.model_type,
                    "config_model_type": e.config_model_type,
                    "last_access": e.last_access if e.last_access > 0 else None,
                }
                for mid, e in sorted(self._entries.items())
            ],
        }

    async def check_ttl_expirations(
        self, settings_manager: ModelSettingsManager
    ) -> list[str]:
        """Check and unload models that have exceeded their TTL.

        Pinned models are skipped (TTL is ignored for pinned models).
        Models with active requests are skipped and their last_access is refreshed.

        Args:
            settings_manager: The settings manager to read TTL values from.

        Returns:
            List of model IDs that were unloaded.
        """
        now = time.time()
        expired: list[str] = []

        async with self._lock:
            for model_id, entry in self._entries.items():
                if entry.engine is None or entry.is_loading or entry.is_pinned:
                    continue

                settings = settings_manager.get_settings(model_id)
                if settings.ttl_seconds is None:
                    continue

                idle_time = now - entry.last_access
                if idle_time < settings.ttl_seconds:
                    continue

                # Check if model has active requests
                has_active = False
                if isinstance(entry.engine, BatchedEngine):
                    engine_core = getattr(entry.engine, "_engine", None)
                    if engine_core is not None:
                        inner = getattr(engine_core, "engine", None)
                        if inner is not None:
                            collectors = getattr(inner, "_output_collectors", {})
                            if len(collectors) > 0:
                                has_active = True

                if has_active:
                    entry.last_access = now
                    continue

                logger.info(
                    f"TTL expired for model '{model_id}' "
                    f"(idle {idle_time:.0f}s > ttl {settings.ttl_seconds}s)"
                )
                await self._unload_engine(model_id)
                expired.append(model_id)

        return expired
