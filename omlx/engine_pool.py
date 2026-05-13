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
from .engine.stt import STTEngine
from .engine.sts import STSEngine
from .engine.tts import TTSEngine
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
from .utils.proc_memory import get_phys_footprint

logger = logging.getLogger(__name__)


@dataclass
class EngineEntry:
    """Per-model state in the engine pool."""

    model_id: str  # Directory name (e.g., "llama-3b")
    model_path: str  # Full path to model directory
    model_type: Literal["llm", "vlm", "embedding", "reranker", "audio_stt", "audio_tts", "audio_sts"]  # Model type
    engine_type: Literal["batched", "simple", "embedding", "reranker", "vlm", "audio_stt", "audio_tts", "audio_sts"]  # Engine type to use
    estimated_size: int  # Pre-calculated from safetensors (bytes)
    config_model_type: str = ""  # Raw model_type from config.json (e.g., "deepseekocr_2")
    thinking_default: bool | None = None  # True if model thinks by default, False if not, None if unknown
    preserve_thinking_default: bool | None = None  # True when template supports preserve_thinking (Qwen 3.6+)
    engine: BaseEngine | EmbeddingEngine | RerankerEngine | STTEngine | STSEngine | TTSEngine | None = None  # Loaded engine instance
    last_access: float = 0.0  # Timestamp for LRU (0 if never loaded)
    is_loading: bool = False  # Prevent concurrent loads
    loading_started_at: float | None = None  # Timestamp when current load started
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
        self._suppress_ttl: bool = False  # Suppress TTL during benchmarks
        self._load_seconds_per_gb_ema: float | None = None
        self._load_time_observations: int = 0

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
                    thinking_default=getattr(info, "thinking_default", None),
                    preserve_thinking_default=getattr(info, "preserve_thinking_default", None),
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
        "audio_stt": "audio_stt",
        "audio_tts": "audio_tts",
        "audio_sts": "audio_sts",
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

    async def get_engine(
        self, model_id: str, force_lm: bool = False,
    ) -> BaseEngine | EmbeddingEngine | RerankerEngine | STTEngine | STSEngine | TTSEngine:
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
            force_lm: Force loading as LM (BatchedEngine) even for VLM models.
                Useful for text-only tasks like accuracy benchmarks.

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
                # If force_lm requested but current engine is VLM, unload and reload
                if force_lm and isinstance(entry.engine, VLMBatchedEngine):
                    logger.info(
                        f"Unloading VLM engine for {model_id} "
                        f"(force_lm=True, reloading as LM)"
                    )
                    await self._unload_engine(model_id)
                else:
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
            # Audio engines (STT/TTS) don't use KV cache, so headroom is 0.
            if self._max_model_memory is not None:
                if entry.engine_type in ("audio_stt", "audio_tts", "audio_sts"):
                    kv_headroom = 0
                else:
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
            # max(active, phys_footprint) matches what jetsam sees and what
            # ProcessMemoryEnforcer uses, so load decisions are consistent.
            if self._process_memory_enforcer is not None:
                enforcer = self._process_memory_enforcer
                if enforcer.max_bytes > 0:
                    while True:
                        current = max(mx.get_active_memory(), get_phys_footprint())
                        projected = current + entry.estimated_size
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
                            current=current,
                            message=(
                                f"Cannot load {model_id}: projected memory "
                                f"{format_size(projected)} would exceed process "
                                f"limit {format_size(enforcer.max_bytes)} "
                                f"(current: {format_size(current)}, "
                                f"model: {format_size(entry.estimated_size)})"
                            ),
                        )

            # Now load the model
            await self._load_engine(model_id, force_lm=force_lm)

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

        Skips models with active inference requests to avoid interrupting
        in-flight generation.

        Returns:
            Model ID of the LRU victim, or None if no evictable model found
        """
        candidates = []
        for mid, e in self._entries.items():
            if e.engine is None or e.is_pinned:
                continue
            try:
                if e.engine.has_active_requests():
                    logger.debug(
                        f"Skipping victim '{mid}': has active requests"
                    )
                    continue
            except AttributeError:
                pass
            candidates.append((e.last_access, mid))
        if not candidates:
            return None
        candidates.sort()  # Sort by last_access (oldest first)
        return candidates[0][1]

    async def _unload_engine(self, model_id: str) -> None:
        """
        Immediately stop and unload an engine with memory settle barrier.

        After stopping the engine, polls mx.get_active_memory() to verify
        Metal buffers are actually reclaimed before updating the memory
        tracking counter.

        Args:
            model_id: The model ID to unload
        """
        entry = self._entries.get(model_id)
        if not entry or entry.engine is None:
            return

        logger.info(f"Unloading model: {model_id} (immediate abort)")
        pre_unload_active = mx.get_active_memory()

        try:
            await entry.engine.stop()
        except Exception as e:
            logger.warning(f"Error stopping engine for {model_id}: {e}")

        # Clear engine reference before settle barrier
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

        # Memory settle barrier: poll actual freed memory instead of
        # trusting the cumulative _current_model_memory estimate.
        # Scale tolerance with model size: estimated_size includes a 5%
        # overhead factor (model_discovery.py) that may not be reflected in
        # actual freed memory. Use 2 GB floor for small models. See #768.
        settle_tolerance = max(2 * 1024**3, int(entry.estimated_size * 0.05))
        min_expected_freed = max(0, entry.estimated_size - settle_tolerance)
        settled = False
        for _settle_round in range(10):
            active_now = mx.get_active_memory()
            actual_freed = pre_unload_active - active_now
            if actual_freed >= min_expected_freed:
                settled = True
                logger.debug(
                    f"Settle round {_settle_round + 1} for '{model_id}': "
                    f"freed={format_size(actual_freed)} "
                    f"(need>={format_size(min_expected_freed)}) - settled"
                )
                break
            logger.debug(
                f"Settle round {_settle_round + 1} for '{model_id}': "
                f"freed={format_size(actual_freed)} "
                f"(need>={format_size(min_expected_freed)}) - retry"
            )
            await asyncio.sleep(0.5)
            gc.collect()
            await loop.run_in_executor(
                get_mlx_executor(), lambda: (mx.synchronize(), mx.clear_cache())
            )

        # Release memory tracking AFTER barrier
        self._current_model_memory -= entry.estimated_size

        if settled:
            logger.info(
                f"Unloaded model: {model_id}, "
                f"freed={format_size(actual_freed)} "
                f"(expected>={format_size(min_expected_freed)}), "
                f"active_memory: {format_size(active_now)} (settled)"
            )
        else:
            # Barrier timed out - try emergency reclaim
            logger.warning(
                f"Settle barrier timed out for '{model_id}': "
                f"freed={format_size(actual_freed)} "
                f"(need>={format_size(min_expected_freed)})"
            )
            for _ in range(3):
                gc.collect()
                await loop.run_in_executor(
                    get_mlx_executor(),
                    lambda: (mx.synchronize(), mx.clear_cache()),
                )
                await asyncio.sleep(1.0)
            active_after = mx.get_active_memory()
            if active_after > self._current_model_memory + 5 * 1024**3:
                logger.error(
                    f"Emergency reclaim failed for '{model_id}': "
                    f"active_memory={format_size(active_after)} "
                    f"exceeds safe threshold "
                    f"({format_size(self._current_model_memory + 5 * 1024**3)})"
                )
            else:
                logger.info(
                    f"Emergency reclaim succeeded: "
                    f"active_memory={format_size(active_after)}"
                )

    async def _load_engine(self, model_id: str, force_lm: bool = False) -> None:
        """
        Load an engine for the specified model.

        Args:
            model_id: The model ID to load
            force_lm: Force loading as BatchedEngine even for VLM models.

        Raises:
            ModelLoadingError: If model is already being loaded
        """
        entry = self._entries[model_id]
        if entry.is_loading:
            raise ModelLoadingError(model_id)

        entry.is_loading = True
        entry.loading_started_at = time.monotonic()
        load_started_at = entry.loading_started_at
        load_completed = False
        entry.abort_loading = False
        try:
            effective_type = entry.engine_type
            if force_lm and effective_type == "vlm":
                effective_type = "batched"
                logger.info(f"Loading model as LM (force_lm=True): {model_id}")
            else:
                logger.info(f"Loading model: {model_id}")

            # Retrieve per-model settings for post-load transforms
            model_settings = None
            if self._settings_manager is not None:
                model_settings = self._settings_manager.get_settings(model_id)

            # Native MTP forces LM-only dispatch even for VLM models. Vision
            # encoder weights are ignored because the patched mtp_forward only
            # exists on the language model path. mtp_enabled was already
            # validated as mutually exclusive with dflash / turboquant in
            # metal-knowledge: with the mlx-vlm runtime MTP patch (see
            # omlx/patches/mlx_vlm_mtp/qwen35_moe_vlm_runtime.py) VLM models
            # can run MTP natively while keeping vision intact. The old
            # force-LM-dispatch shortcut here is obsolete for patched
            # model families; let VLMBatchedEngine handle MTP-enabled VLMs.
            pass

            # Check if DFlash is enabled — takes priority over engine type
            # since DFlash has its own model loading pipeline
            engine = None
            if model_settings is not None:
                dflash_enabled = getattr(model_settings, "dflash_enabled", False)
                dflash_draft = getattr(model_settings, "dflash_draft_model", None)
                if dflash_enabled and dflash_draft:
                    try:
                        from .engine.dflash import DFlashEngine
                        engine = DFlashEngine(
                            model_name=entry.model_path,
                            draft_model_path=dflash_draft,
                            draft_quant_enabled=getattr(model_settings, "dflash_draft_quant_enabled", False),
                            draft_quant_weight_bits=getattr(model_settings, "dflash_draft_quant_weight_bits", 4),
                            draft_quant_activation_bits=getattr(model_settings, "dflash_draft_quant_activation_bits", 16),
                            draft_quant_group_size=getattr(model_settings, "dflash_draft_quant_group_size", 64),
                            model_settings=model_settings,
                            fallback_engine_type=effective_type,
                            scheduler_config=self._scheduler_config,
                            omlx_ssd_cache_dir=getattr(
                                self._scheduler_config, "paged_ssd_cache_dir", None
                            ),
                        )
                        logger.info(f"DFlash enabled for {model_id}, draft={dflash_draft}")
                    except ImportError:
                        logger.warning(
                            f"DFlash enabled for {model_id} but dflash-mlx is not installed. "
                            f"Falling back to default engine."
                        )
                    except Exception as e:
                        logger.warning(
                            f"DFlash init failed for {model_id}: {e}. "
                            f"Falling back to default engine."
                        )

            # Per-model trust_remote_code (security opt-in, issue #926).
            # When unset, defaults to False — repos with custom modeling_*.py
            # will fail to load until the user explicitly toggles this on
            # in the admin UI's model settings modal.
            trc = bool(getattr(model_settings, "trust_remote_code", False)) if model_settings else False

            # Create engine based on engine type (if DFlash not active)
            if engine is None:
                if effective_type == "embedding":
                    engine = EmbeddingEngine(
                        model_name=entry.model_path,
                        trust_remote_code=trc,
                    )
                elif effective_type == "reranker":
                    engine = RerankerEngine(
                        model_name=entry.model_path,
                        trust_remote_code=trc,
                    )
                elif effective_type == "vlm":
                    engine = VLMBatchedEngine(
                        model_name=entry.model_path,
                        trust_remote_code=trc,
                        scheduler_config=self._scheduler_config,
                        model_settings=model_settings,
                    )
                elif entry.engine_type == "audio_stt":
                    engine = STTEngine(model_name=entry.model_path)
                elif entry.engine_type == "audio_tts":
                    engine = TTSEngine(model_name=entry.model_path)
                elif entry.engine_type == "audio_sts":
                    engine = STSEngine(
                        model_name=entry.model_path,
                        config_model_type=entry.config_model_type,
                    )
                else:
                    engine = BatchedEngine(
                        model_name=entry.model_path,
                        trust_remote_code=trc,
                        scheduler_config=self._scheduler_config,
                        model_settings=model_settings,
                    )

            _is_dflash_engine = engine is not None and type(engine).__name__ == "DFlashEngine"

            try:
                await engine.start()
            except Exception as start_error:
                if _is_dflash_engine:
                    # DFlash engine failed to start — fall back to the
                    # model's natural engine type (VLM or Batched)
                    logger.warning(
                        f"DFlash start failed for {model_id}: {start_error}. "
                        f"Falling back to {effective_type} engine."
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

                    if effective_type == "vlm":
                        engine = VLMBatchedEngine(
                            model_name=entry.model_path,
                            trust_remote_code=trc,
                            scheduler_config=self._scheduler_config,
                            model_settings=model_settings,
                        )
                    else:
                        engine = BatchedEngine(
                            model_name=entry.model_path,
                            trust_remote_code=trc,
                            scheduler_config=self._scheduler_config,
                            model_settings=model_settings,
                        )
                    await engine.start()
                    logger.info(
                        f"Successfully loaded {model_id} as {effective_type} "
                        f"(fallback from DFlash)"
                    )

                elif force_lm and entry.engine_type == "vlm":
                    # force_lm created a BatchedEngine but mlx-lm can't
                    # load this VLM model — fall back to VLMBatchedEngine.
                    logger.warning(
                        f"LM loading failed for VLM model {model_id} "
                        f"(force_lm=True), falling back to VLM engine: "
                        f"{start_error}"
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

                    engine = VLMBatchedEngine(
                        model_name=entry.model_path,
                        trust_remote_code=trc,
                        scheduler_config=self._scheduler_config,
                        model_settings=model_settings,
                    )
                    await engine.start()

                    logger.info(
                        f"Successfully loaded {model_id} as VLM "
                        f"(fallback from force_lm)"
                    )
                elif entry.engine_type == "vlm":
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
                        trust_remote_code=trc,
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
            load_completed = True

            # VLM MTP: load gemma4_assistant drafter and attach to engine.
            # Fail-soft — drafter load issues never block the target engine.
            if (
                model_settings is not None
                and getattr(model_settings, "vlm_mtp_enabled", False)
                and getattr(model_settings, "vlm_mtp_draft_model", None)
                and hasattr(engine, "set_vlm_mtp_drafter")
            ):
                drafter_id = model_settings.vlm_mtp_draft_model
                drafter_entry = self._entries.get(drafter_id)
                drafter_path = (
                    drafter_entry.model_path if drafter_entry else drafter_id
                )

                def _load_drafter_sync(path: str = drafter_path):
                    from .speculative.vlm_mtp import load_vlm_mtp_drafter
                    return load_vlm_mtp_drafter(path)

                loop = asyncio.get_running_loop()
                try:
                    drafter = await loop.run_in_executor(
                        get_mlx_executor(), _load_drafter_sync
                    )
                except Exception as e:
                    logger.warning(
                        f"VLM MTP drafter load raised for {model_id} "
                        f"(drafter={drafter_id}): {e} — toggle ignored"
                    )
                    drafter = None
                if drafter is not None:
                    engine.set_vlm_mtp_drafter(drafter)
                    logger.info(
                        f"VLM MTP enabled for {model_id}, drafter={drafter_id}"
                    )
                else:
                    logger.warning(
                        f"VLM MTP toggle on for {model_id} but drafter "
                        f"load failed; toggle ignored"
                    )

            # Propagate memory limit to new engine's scheduler
            if self._process_memory_enforcer is not None:
                self._process_memory_enforcer._propagate_memory_limit()

            # Release intermediate Metal buffers from model loading.
            # mlx_lm.load() creates large temporaries (weight transforms,
            # quantization intermediates) that stay in the Metal buffer pool
            # because mx.set_cache_limit(total_mem) prevents automatic release.
            # Without this, memory stays at ~2x model size until the first
            # inference request triggers a clear. (#429)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                get_mlx_executor(),
                lambda: (mx.synchronize(), mx.clear_cache()),
            )

            logger.info(
                f"Loaded model: {model_id} "
                f"(estimated: {format_size(entry.estimated_size)}, "
                f"total: {format_size(self._current_model_memory)})"
            )
        finally:
            if load_completed and load_started_at is not None and entry.estimated_size > 0:
                elapsed = max(0.0, time.monotonic() - load_started_at)
                size_gb = entry.estimated_size / (1024 ** 3)
                if size_gb > 0 and elapsed > 0:
                    sample = elapsed / size_gb
                    if self._load_seconds_per_gb_ema is None:
                        self._load_seconds_per_gb_ema = sample
                    else:
                        self._load_seconds_per_gb_ema = (
                            self._load_seconds_per_gb_ema * 0.9 + sample * 0.1
                        )
                    self._load_time_observations += 1
                    logger.debug(
                        f"Observed model load speed: {sample:.2f}s/GB "
                        f"for {model_id} ({elapsed:.1f}s, {format_size(entry.estimated_size)}); "
                        f"EMA={self._load_seconds_per_gb_ema:.2f}s/GB"
                    )
            entry.is_loading = False
            entry.loading_started_at = None
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
            "load_seconds_per_gb_estimate": self._load_seconds_per_gb_ema,
            "load_time_observations": self._load_time_observations,
            "models": [
                {
                    "id": mid,
                    "model_path": e.model_path,
                    "loaded": e.engine is not None,
                    "is_loading": e.is_loading,
                    "loading_started_at": e.loading_started_at,
                    "estimated_size": e.estimated_size,
                    "pinned": e.is_pinned,
                    "engine_type": e.engine_type,
                    "model_type": e.model_type,
                    "config_model_type": e.config_model_type,
                    "thinking_default": e.thinking_default,
                    "preserve_thinking_default": e.preserve_thinking_default,
                    "last_access": e.last_access if e.last_access > 0 else None,
                }
                for mid, e in sorted(self._entries.items())
            ],
        }

    async def check_ttl_expirations(
        self,
        settings_manager: ModelSettingsManager,
        global_idle_timeout_seconds: int | None = None,
    ) -> list[str]:
        """Check and unload models that have exceeded their TTL.

        Pinned models are skipped (TTL is ignored for pinned models).
        Models with active requests are skipped and their last_access is refreshed.
        Suppressed during benchmark runs via _suppress_ttl flag.

        Args:
            settings_manager: The settings manager to read TTL values from.
            global_idle_timeout_seconds: Global idle timeout fallback (None = no global TTL).

        Returns:
            List of model IDs that were unloaded.
        """
        if self._suppress_ttl:
            return []

        now = time.time()
        expired: list[str] = []

        async with self._lock:
            for model_id, entry in self._entries.items():
                if entry.engine is None or entry.is_loading or entry.is_pinned:
                    continue

                settings = settings_manager.get_settings(model_id)
                effective_ttl = settings.ttl_seconds
                if effective_ttl is None:
                    effective_ttl = global_idle_timeout_seconds
                if effective_ttl is None:
                    continue

                idle_time = now - entry.last_access
                if idle_time < effective_ttl:
                    continue

                # Check if model has active requests
                has_active = entry.engine.has_active_requests()

                if has_active:
                    entry.last_access = now
                    continue

                logger.info(
                    f"TTL expired for model '{model_id}' "
                    f"(idle {idle_time:.0f}s > ttl {effective_ttl}s)"
                )
                await self._unload_engine(model_id)
                expired.append(model_id)

        return expired
