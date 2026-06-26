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
import json
import logging
import time
from contextlib import asynccontextmanager
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
    ModelBusyError,
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
    model_type: Literal[
        "llm", "vlm", "embedding", "reranker", "audio_stt", "audio_tts", "audio_sts"
    ]  # Model type
    engine_type: Literal[
        "batched",
        "simple",
        "embedding",
        "reranker",
        "vlm",
        "audio_stt",
        "audio_tts",
        "audio_sts",
    ]  # Engine type to use
    estimated_size: int  # Pre-calculated from safetensors (bytes)
    actual_size: int | None = None  # Observed process-memory delta after load settles
    config_model_type: str = (
        ""  # Raw model_type from config.json (e.g., "deepseekocr_2")
    )
    thinking_default: bool | None = (
        None  # True if model thinks by default, False if not, None if unknown
    )
    preserve_thinking_default: bool | None = (
        None  # True when template supports preserve_thinking (Qwen 3.6+)
    )
    model_context_length: int | None = (
        None  # Declared context length from config.json (None if unknown)
    )
    source_type: str = "local"
    source_repo_id: str | None = None
    engine: (
        BaseEngine
        | EmbeddingEngine
        | RerankerEngine
        | STTEngine
        | STSEngine
        | TTSEngine
        | None
    ) = None  # Loaded engine instance
    last_access: float = 0.0  # Timestamp for LRU (0 if never loaded)
    is_loading: bool = False  # Prevent concurrent loads
    loading_started_at: float | None = None  # Timestamp when current load started
    is_pinned: bool = False  # Never evict if True
    abort_loading: bool = False  # Set by memory enforcer to abort in-progress load
    in_use: int = 0  # in-flight acquire/use lease count; never evict while > 0
    abort_requested: bool = False  # Set under hard pressure for leased requests
    pending_unload_reason: str | None = None  # Unload as soon as leases/activity drain
    runtime_settings_signature: tuple[tuple[str, str], ...] | None = None


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
        scheduler_config: SchedulerConfig | None = None,
    ):
        """
        Initialize the engine pool.

        Args:
            scheduler_config: Configuration for BatchedEngine schedulers

        Note:
            Pre-load admission consults `enforcer.get_final_ceiling()` via
            the `_get_final_ceiling` callback set by `server.init_server()`.
            Until the callback is wired up the pool admits unconditionally.
        """
        self._entries: dict[str, EngineEntry] = {}
        self._lock = asyncio.Lock()
        self._current_model_memory = 0
        self._scheduler_config = scheduler_config or SchedulerConfig()
        self._process_memory_enforcer: object | None = None  # Set by server
        self._get_final_ceiling: object | None = None  # Set by server
        self._settings_manager: object | None = None  # Set by server
        self._suppress_ttl: bool = False  # Suppress TTL during benchmarks
        self._load_seconds_per_gb_ema: float | None = None
        self._load_time_observations: int = 0
        self.configure_hot_cache_budget()

    @property
    def current_model_memory(self) -> int:
        """Current memory used by loaded models in bytes."""
        return self._current_model_memory

    def configure_hot_cache_budget(self) -> None:
        """Ensure loaded schedulers share one process-wide hot cache budget."""
        hot_max = int(getattr(self._scheduler_config, "hot_cache_max_size", 0) or 0)
        if hot_max <= 0:
            self._scheduler_config.hot_cache_budget = None
            return

        current = getattr(self._scheduler_config, "hot_cache_budget", None)
        if current is not None and getattr(current, "max_bytes", None) == hot_max:
            return

        from .cache.paged_ssd_cache import SharedHotCacheBudget

        self._scheduler_config.hot_cache_budget = SharedHotCacheBudget(hot_max)

    def _current_ceiling(self) -> int:
        """Resolve the current memory ceiling via the enforcer callback.

        Returns 0 when no callback is wired up (treated by callers as
        "no limit").
        """
        cb = self._get_final_ceiling
        if cb is None:
            return 0
        try:
            return int(cb())
        except Exception:  # noqa: BLE001
            return 0

    def _wake_process_memory_enforcer(self, *, active: bool = False) -> None:
        enforcer = self._process_memory_enforcer
        wake = getattr(enforcer, "wake", None) if enforcer is not None else None
        if callable(wake):
            wake(active=active)

    @staticmethod
    def _canonical_signature_value(value: object) -> str:
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, sort_keys=True, separators=(",", ":"))
        return repr(value)

    def _engine_runtime_signature(
        self,
        model_id: str,
        runtime_settings: object | None = None,
        *,
        loaded_engine: object | None = None,
    ) -> tuple[tuple[str, str], ...] | None:
        settings = runtime_settings
        if settings is None and self._settings_manager is not None:
            get_settings = getattr(self._settings_manager, "get_settings", None)
            if callable(get_settings):
                settings = get_settings(model_id)
        if settings is None:
            return None

        to_dict = getattr(settings, "to_dict", None)
        data = to_dict() if callable(to_dict) else {}
        entry = self._entries.get(model_id)
        is_diffusion = bool(entry and self._entry_is_diffusion_model(entry))
        loaded_engine_name = (
            type(loaded_engine).__name__ if loaded_engine is not None else None
        )

        def has_value(key: str) -> bool:
            value = data.get(key)
            return value is not None and value != ""

        def normalized_index_cache_freq() -> int | None:
            value = data.get("index_cache_freq")
            try:
                freq = int(value) if value is not None else None
            except (TypeError, ValueError):
                return None
            return freq if freq is not None and freq >= 2 else None

        signature: list[tuple[str, str]] = []

        def add(key: str, value: object) -> None:
            signature.append((key, self._canonical_signature_value(value)))

        # Security/load gates.
        add("trust_remote_code", bool(data.get("trust_remote_code", False)))
        add("index_cache_freq", normalized_index_cache_freq())

        # Load-time model variants. Dependent fields only matter when their
        # feature is active; stale draft paths or tuning defaults must not
        # force a reload when the corresponding feature is disabled.
        mtp_active = bool(data.get("mtp_enabled", False))
        add("mtp_enabled", mtp_active)

        turboquant_active = bool(data.get("turboquant_kv_enabled", False))
        add("turboquant_kv_enabled", turboquant_active)
        if turboquant_active:
            add("turboquant_kv_bits", data.get("turboquant_kv_bits", 4))
            add("turboquant_skip_last", data.get("turboquant_skip_last", True))

        specprefill_active = bool(data.get("specprefill_enabled", False)) and has_value(
            "specprefill_draft_model"
        )
        add("specprefill_enabled", specprefill_active)
        if specprefill_active:
            add("specprefill_draft_model", data.get("specprefill_draft_model"))
            add("specprefill_keep_pct", data.get("specprefill_keep_pct", 0.2))
            add("specprefill_threshold", data.get("specprefill_threshold"))

        dflash_active = (
            bool(data.get("dflash_enabled", False))
            and has_value("dflash_draft_model")
            and not is_diffusion
        )
        if loaded_engine_name is not None:
            dflash_active = loaded_engine_name == "DFlashEngine"
        add("dflash_enabled", dflash_active)
        if dflash_active:
            add("dflash_draft_model", data.get("dflash_draft_model"))
            add(
                "dflash_draft_quant_enabled",
                bool(data.get("dflash_draft_quant_enabled", False)),
            )
            if data.get("dflash_draft_quant_enabled", False):
                add(
                    "dflash_draft_quant_weight_bits",
                    data.get("dflash_draft_quant_weight_bits", 4),
                )
                add(
                    "dflash_draft_quant_activation_bits",
                    data.get("dflash_draft_quant_activation_bits", 16),
                )
                add(
                    "dflash_draft_quant_group_size",
                    data.get("dflash_draft_quant_group_size", 64),
                )
            add("dflash_max_ctx", data.get("dflash_max_ctx"))
            add("dflash_in_memory_cache", data.get("dflash_in_memory_cache", True))
            add(
                "dflash_in_memory_cache_max_entries",
                data.get("dflash_in_memory_cache_max_entries", 4),
            )
            add(
                "dflash_in_memory_cache_max_bytes",
                data.get("dflash_in_memory_cache_max_bytes"),
            )
            add("dflash_ssd_cache", bool(data.get("dflash_ssd_cache", False)))
            if data.get("dflash_ssd_cache", False):
                add(
                    "dflash_ssd_cache_max_bytes", data.get("dflash_ssd_cache_max_bytes")
                )
            add("dflash_draft_window_size", data.get("dflash_draft_window_size"))
            add("dflash_draft_sink_size", data.get("dflash_draft_sink_size"))
            add("dflash_verify_mode", data.get("dflash_verify_mode"))

        vlm_mtp_active = bool(data.get("vlm_mtp_enabled", False)) and has_value(
            "vlm_mtp_draft_model"
        )
        if loaded_engine is not None and vlm_mtp_active:
            drafter = getattr(loaded_engine, "vlm_mtp_drafter", None)
            if callable(drafter):
                drafter = drafter()
            vlm_mtp_active = drafter is not None
        add("vlm_mtp_enabled", vlm_mtp_active)
        if vlm_mtp_active:
            add("vlm_mtp_draft_model", data.get("vlm_mtp_draft_model"))
            add("vlm_mtp_draft_block_size", data.get("vlm_mtp_draft_block_size"))

        return tuple(signature)

    @property
    def model_count(self) -> int:
        """Total number of discovered models."""
        return len(self._entries)

    @property
    def loaded_model_count(self) -> int:
        """Number of currently loaded models."""
        return sum(1 for e in self._entries.values() if e.engine is not None)

    async def apply_embedding_batch_size(self, batch_size: int) -> None:
        """Apply embedding batch size to future and currently loaded embedding engines."""
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("embedding batch size must be > 0")

        async with self._lock:
            self._scheduler_config.embedding_batch_size = batch_size
            for entry in list(self._entries.values()):
                engine = entry.engine if entry is not None else None
                if isinstance(engine, EmbeddingEngine):
                    engine._batch_size = batch_size

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
                    preserve_thinking_default=getattr(
                        info, "preserve_thinking_default", None
                    ),
                    model_context_length=getattr(info, "model_context_length", None),
                    source_type=getattr(info, "source_type", "local"),
                    source_repo_id=getattr(info, "source_repo_id", None),
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

        logger.info(f"Discovered {len(self._entries)} models")

    _MODEL_TYPE_TO_ENGINE: dict[str, str] = {
        "llm": "batched",
        "vlm": "vlm",
        "embedding": "embedding",
        "reranker": "reranker",
        "audio_stt": "audio_stt",
        "audio_tts": "audio_tts",
        "audio_sts": "audio_sts",
    }

    @staticmethod
    def _entry_is_diffusion_model(entry: EngineEntry) -> bool:
        model_type = (entry.config_model_type or "").lower().replace("-", "_")
        return model_type == "diffusion_gemma"

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
        then exposed profile model IDs, then scans model settings for alias
        match. If those fail and input contains a provider prefix (e.g.
        "omlx/my-model"), strips the prefix and retries. Returns the
        original string if no match found.
        """
        if model_id_or_alias in self._entries:
            return model_id_or_alias

        # Case-insensitive fallback
        ci_match = self._case_insensitive_entry_match(model_id_or_alias)
        if ci_match is not None:
            return ci_match

        all_settings = None
        if settings_manager is not None:
            # Exposed profiles resolve to the physical model they overlay
            # (handles provider prefixes internally).
            if hasattr(settings_manager, "get_exposed_profile_source_model_id"):
                profile_source = settings_manager.get_exposed_profile_source_model_id(
                    model_id_or_alias
                )
                if profile_source is not None:
                    return profile_source
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

    @staticmethod
    def _entry_has_active_requests(entry: EngineEntry) -> bool:
        engine = entry.engine
        if engine is None:
            return False
        has_active_requests = getattr(engine, "has_active_requests", None)
        if not callable(has_active_requests):
            return False
        try:
            return has_active_requests() is True
        except Exception:
            return True

    def _entry_is_busy(self, entry: EngineEntry) -> bool:
        return entry.in_use > 0 or self._entry_has_active_requests(entry)

    def _raise_if_reload_busy(self, entry: EngineEntry, operation: str) -> None:
        if self._entry_is_busy(entry):
            raise ModelBusyError(entry.model_id, operation)

    def _mark_pending_unload_locked(
        self,
        model_id: str,
        reason: str,
        *,
        abort_requested: bool = False,
    ) -> bool:
        """Mark a loaded non-pinned model for unload once it is no longer busy.

        Caller must hold ``self._lock``. Returns True when a pending marker was
        installed. The method deliberately does not unload by itself; call
        ``_unload_pending_if_idle_locked`` after abort/release state changes.
        """
        entry = self._entries.get(model_id)
        if entry is None or entry.engine is None or entry.is_loading or entry.is_pinned:
            return False
        entry.pending_unload_reason = reason
        if abort_requested:
            entry.abort_requested = True
        return True

    def _find_pending_unload_ready_locked(self) -> str | None:
        candidates: list[tuple[float, str]] = []
        for mid, entry in self._entries.items():
            if not entry.pending_unload_reason:
                continue
            if (
                entry.engine is None
                or entry.is_loading
                or entry.is_pinned
                or self._entry_is_busy(entry)
            ):
                continue
            candidates.append((entry.last_access, mid))
        if not candidates:
            return None
        candidates.sort()
        return candidates[0][1]

    async def _unload_pending_if_idle_locked(self, model_id: str) -> bool:
        """Unload a pending model if all leases and active requests have drained.

        Caller must hold ``self._lock``.
        """
        entry = self._entries.get(model_id)
        if (
            entry is None
            or entry.engine is None
            or not entry.pending_unload_reason
            or entry.is_loading
            or entry.is_pinned
            or self._entry_is_busy(entry)
        ):
            return False

        reason = entry.pending_unload_reason
        entry.pending_unload_reason = None
        entry.abort_requested = False
        logger.warning(
            "Unloading pending model '%s' after activity drained (%s)",
            model_id,
            reason,
        )
        await self._unload_engine(model_id)
        return True

    def is_abort_requested(self, model_id: str | None) -> bool:
        if model_id is None:
            return False
        entry = self._entries.get(model_id)
        return bool(entry and entry.abort_requested)

    async def get_engine(
        self,
        model_id: str,
        force_lm: bool = False,
        _lease: bool = False,
        runtime_settings: object | None = None,
    ) -> (
        BaseEngine
        | EmbeddingEngine
        | RerankerEngine
        | STTEngine
        | STSEngine
        | TTSEngine
    ):
        """
        Get or load engine for the specified model.

        This method implements pre-load memory checking:
        1. Check if model is already loaded -> return immediately
        2. Check if model is too large for memory limit -> raise error
        3. Evict LRU models until there's enough space
        4. Load the model
        5. Return the engine

        Args:
            model_id: The model ID to get engine for
            force_lm: Force loading as LM (BatchedEngine) even for VLM models.
                Useful for text-only tasks like accuracy benchmarks.
            runtime_settings: Optional transient settings used for this engine
                load. When its engine-construction signature differs from the
                currently loaded engine, the old engine is unloaded and the new
                variant is loaded without mutating persisted model settings.

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
            expected_signature = self._engine_runtime_signature(
                model_id,
                runtime_settings,
            )

            # Already loaded - just update access time
            if entry.engine is not None:
                if (
                    expected_signature is not None
                    and entry.runtime_settings_signature is not None
                    and entry.runtime_settings_signature != expected_signature
                ) or (
                    runtime_settings is not None
                    and entry.runtime_settings_signature is None
                ):
                    self._raise_if_reload_busy(
                        entry,
                        "reload runtime settings variant",
                    )
                    logger.info(
                        "Runtime settings variant changed for %s; "
                        "unloading before reload.",
                        model_id,
                    )
                    await self._unload_engine(model_id)
                # If force_lm requested but current engine is VLM, unload and reload
                if (
                    entry.engine is not None
                    and force_lm
                    and isinstance(entry.engine, VLMBatchedEngine)
                ):
                    self._raise_if_reload_busy(entry, "reload as LM")
                    logger.info(
                        f"Unloading VLM engine for {model_id} "
                        f"(force_lm=True, reloading as LM)"
                    )
                    await self._unload_engine(model_id)
                elif entry.engine is not None:
                    if entry.runtime_settings_signature is None:
                        entry.runtime_settings_signature = expected_signature
                    entry.last_access = time.time()
                    if _lease:
                        entry.in_use += 1
                    return entry.engine

            # Pre-load admission against the memory ceiling from the
            # process memory enforcer (min of static and dynamic). Try
            # evicting LRU non-pinned models first; if the model still
            # cannot fit after evicting everything available, raise.
            #
            # ceiling == 0 means the enforcer is off (guard disabled or
            # not yet wired up), so we admit unconditionally.
            ceiling = self._current_ceiling()
            if ceiling > 0:
                while True:
                    # Consult the tracked accumulator alongside live memory:
                    # after a model settles or idles, mx.get_active_memory() and
                    # the process footprint can read well below the model's true
                    # resident size, while _current_model_memory still reflects
                    # the committed total. Using only live memory lets a second
                    # large model load without evicting the first, over-
                    # committing past the ceiling (#1623).
                    current = max(
                        mx.get_active_memory(),
                        get_phys_footprint(),
                        self._current_model_memory,
                    )
                    projected = current + entry.estimated_size
                    if projected <= ceiling:
                        break
                    victim = self._find_lru_victim()
                    if victim is not None:
                        logger.info(
                            f"Evicting '{victim}' to fit '{model_id}' "
                            f"under memory ceiling "
                            f"({format_size(projected)} > "
                            f"{format_size(ceiling)})"
                        )
                        await self._unload_engine(victim)
                        continue
                    # Nothing else to evict -- model cannot fit. Use
                    # ModelTooLargeError when the model alone exceeds the
                    # ceiling (no chance of fitting), InsufficientMemoryError
                    # when the model would fit on a clean process but the
                    # current usage leaves no room.
                    if entry.estimated_size > ceiling:
                        raise ModelTooLargeError(
                            model_id, entry.estimated_size, ceiling
                        )
                    raise InsufficientMemoryError(
                        required=entry.estimated_size,
                        current=current,
                        message=(
                            f"Cannot load {model_id}: projected memory "
                            f"{format_size(projected)} would exceed the memory "
                            f"ceiling {format_size(ceiling)} "
                            f"(current: {format_size(current)}, "
                            f"model: {format_size(entry.estimated_size)}). "
                            "Free system memory or lower memory_guard_tier."
                        ),
                    )

            # Now load the model
            await self._load_engine(
                model_id,
                force_lm=force_lm,
                runtime_settings=runtime_settings,
            )

            loaded = self._entries[model_id]
            if _lease:
                loaded.in_use += 1
            return loaded.engine

    async def release_engine(self, model_id: str) -> None:
        """Release one in-use lease previously taken via get_engine(_lease=True)."""
        async with self._lock:
            e = self._entries.get(model_id)
            if e is not None and e.in_use > 0:
                e.in_use -= 1
            await self._unload_pending_if_idle_locked(model_id)

    async def unload_if_idle_unpinned(self, model_id: str) -> bool:
        """Unload a loaded engine only when it is idle and not pinned."""
        async with self._lock:
            entry = self._entries.get(model_id)
            if (
                entry is None
                or entry.engine is None
                or entry.is_loading
                or entry.is_pinned
                or entry.in_use > 0
            ):
                return False

            if self._entry_has_active_requests(entry):
                entry.last_access = time.time()
                return False

            await self._unload_engine(model_id)
            return True

    @asynccontextmanager
    async def acquire(self, model_id: str, force_lm: bool = False):
        """Acquire an engine with an atomic in-use lease.

        The lease is taken under the pool lock at acquire time and always
        released in finally, so the engine cannot be evicted mid-request even
        on exception.
        """
        engine = await self.get_engine(model_id, force_lm=force_lm, _lease=True)
        try:
            yield engine
        finally:
            await self.release_engine(model_id)

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
            if e.in_use > 0:
                continue
            if self._entry_has_active_requests(e):
                logger.debug(f"Skipping victim '{mid}': has active requests")
                continue
            candidates.append((e.last_access, mid))
        if not candidates:
            return None
        candidates.sort()  # Sort by last_access (oldest first)
        return candidates[0][1]

    async def _unload_other_dflash_engines(self, model_id: str) -> None:
        """Unload other idle DFlash engines before starting a new one.

        dflash-mlx installs target hooks on shared Python classes and owns a
        process-global runtime cache manager, so multiple loaded DFlash engines
        can leak state across model switches.
        """
        victims: list[str] = []
        blocked: list[str] = []
        for mid, e in self._entries.items():
            if mid == model_id or e.engine is None:
                continue
            if type(e.engine).__name__ != "DFlashEngine":
                continue
            if e.is_loading or e.in_use > 0:
                blocked.append(mid)
                continue
            try:
                if e.engine.has_active_requests():
                    blocked.append(mid)
                    continue
            except AttributeError:
                pass
            if e.is_pinned:
                blocked.append(f"{mid} (pinned)")
                continue
            victims.append(mid)

        if blocked:
            raise RuntimeError(
                "Cannot load DFlash model "
                f"'{model_id}' while another DFlash engine is active: "
                f"{', '.join(blocked)}"
            )

        for victim in victims:
            logger.info(
                "Unloading DFlash model '%s' before loading '%s' because "
                "dflash runtime hooks/cache are process-global",
                victim,
                model_id,
            )
            await self._unload_engine(victim)

    @staticmethod
    def _resolve_scheduler_from_engine(engine: object) -> object | None:
        scheduler = getattr(engine, "scheduler", None)
        if scheduler is not None:
            return scheduler
        try:
            return engine._engine.engine.scheduler  # type: ignore[attr-defined]
        except AttributeError:
            return None

    def _is_idle_for_prefill_eviction(self, entry: EngineEntry) -> bool:
        engine = entry.engine
        if engine is None or entry.is_pinned or entry.is_loading or entry.in_use > 0:
            return False
        if self._entry_has_active_requests(entry):
            return False

        scheduler = self._resolve_scheduler_from_engine(engine)
        if scheduler is None:
            return True
        for attr in ("running", "waiting", "prefilling", "requests"):
            value = getattr(scheduler, attr, None)
            if value:
                return False
        return True

    def _find_lru_prefill_eviction_victim(self, *, exclude_model_id: str) -> str | None:
        candidates = []
        for mid, entry in self._entries.items():
            if mid == exclude_model_id:
                continue
            if self._is_idle_for_prefill_eviction(entry):
                candidates.append((entry.last_access, mid))
        if not candidates:
            return None
        candidates.sort()
        return candidates[0][1]

    async def _evict_idle_lru_for_prefill(
        self,
        exclude_model_id: str,
        eviction_request: object,
    ) -> bool:
        """Evict idle LRU models until the requested prefill step should fit."""
        target = int(getattr(eviction_request, "target_cap_bytes", 0) or 0)
        predicted = int(getattr(eviction_request, "predicted_transient_bytes", 0) or 0)
        request_id = str(getattr(eviction_request, "request_id", ""))
        if target <= 0 or predicted <= 0:
            return False

        evicted_any = False
        async with self._lock:
            while True:
                current = max(
                    mx.get_active_memory(),
                    get_phys_footprint(),
                    self._current_model_memory,
                )
                if current + predicted <= target:
                    return evicted_any

                victim = self._find_lru_prefill_eviction_victim(
                    exclude_model_id=exclude_model_id
                )
                if victim is None:
                    if evicted_any:
                        logger.info(
                            "Prefill eviction for request %s stopped with no "
                            "more idle victims (current=%s, predicted=%s, "
                            "target=%s)",
                            request_id,
                            format_size(current),
                            format_size(predicted),
                            format_size(target),
                        )
                    return evicted_any

                logger.info(
                    "Evicting idle model '%s' for prefill headroom on '%s' "
                    "(request=%s, projected=%s > target=%s)",
                    victim,
                    exclude_model_id,
                    request_id,
                    format_size(current + predicted),
                    format_size(target),
                )
                await self._unload_engine(victim)
                evicted_any = True

    def _other_entries_serving(self, model_id: str) -> bool:
        """True when any loaded entry other than ``model_id`` is serving.

        Used by the settle barrier in ``_unload_engine``: the barrier's
        freed-memory check is a delta of the process-global
        ``mx.get_active_memory()`` gauge, which only measures THIS unload
        while no other engine is allocating concurrently.
        """
        # Snapshot the items: admin unload routes call _unload_engine without
        # the pool lock, so discover_models() can mutate _entries mid-iteration.
        for mid, e in list(self._entries.items()):
            if mid == model_id or e.engine is None:
                continue
            if e.in_use > 0:
                return True
            if self._entry_has_active_requests(e):
                return True
        return False

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

        # #1595: the immediate-abort stop() above tears the engine down without the normal
        # per-request completion callbacks, so a non-streaming engine's active_requests
        # counter can leak a phantom count (a stale engine then looks permanently busy).
        # Reset it on teardown so has_active_requests() and the status API stay consistent.
        reset = getattr(entry.engine, "_reset_activity_tracking", None)
        if callable(reset):
            try:
                reset()
            except Exception as e:
                logger.warning(f"Error resetting activity counter for {model_id}: {e}")

        # Yield to the event loop before dropping the engine reference.
        #
        # When abort_all_requests() fires before _unload_engine(), it sets
        # asyncio Events for each active request.  Server-side streaming
        # generators are then scheduled in the asyncio ready queue, but they
        # cannot run until the event loop gets control.  EngineCore.close()
        # (called inside stop()) blocks the event loop with synchronous
        # .result() calls on the MLX executor -- scheduler.shutdown() and
        # scheduler.deep_reset() -- so those generators are still suspended
        # when stop() returns.
        #
        # If we set entry.engine = None and call gc.collect() immediately,
        # the generators are still alive with a local 'engine' variable
        # referencing the BatchedEngine, keeping its refcount above zero.
        # The model's ~20 GB of MLX weight tensors therefore remain "active"
        # in Metal memory, the settle barrier times out, and subsequent load
        # attempts fail with 507 because the ceiling is still exceeded.
        #
        # A few asyncio.sleep(0) calls drain the ready queue -- generator
        # tear-down is at most a few frames deep -- so that by the time we
        # clear entry.engine and run gc.collect(), no coroutine frame holds
        # a stale engine reference.
        for _ in range(5):
            await asyncio.sleep(0)

        # Clear engine reference before settle barrier
        entry.engine = None
        entry.last_access = 0.0
        entry.actual_size = None
        entry.abort_requested = False
        entry.pending_unload_reason = None
        entry.runtime_settings_signature = None

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
        settle_indeterminate = False
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
            if self._other_entries_serving(model_id):
                # actual_freed is a delta of the process-global MLX gauge,
                # so while another engine allocates (prefill/KV growth) the
                # amount freed by THIS unload is unmeasurable — the delta can
                # even read negative. Burning settle rounds here serializes
                # gc/synchronize/clear_cache against live decode for seconds,
                # under memory pressure, with the enforcer holding the pool
                # lock. Bail out instead: pre-load admission re-reads the
                # live gauge, so nothing downstream trusts this sample.
                settle_indeterminate = True
                logger.info(
                    f"Settle for '{model_id}' indeterminate under concurrent "
                    f"activity (freed={format_size(actual_freed)}, "
                    f"need>={format_size(min_expected_freed)}); skipping "
                    f"settle wait"
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
        elif settle_indeterminate:
            # Settle wait skipped (logged above). Emergency reclaim is
            # deliberately skipped too: its gc + synchronize + clear_cache
            # rounds would stall the live engines that made the measurement
            # indeterminate in the first place. Recovery is not lost:
            # _wake_process_memory_enforcer() below triggers an immediate
            # enforcer re-poll, and pre-load admission re-reads the live gauge
            # alongside the tracked accumulator (the #1623 max() in
            # get_engine), so any unreleased memory stays visible to both.
            pass
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

        self._wake_process_memory_enforcer()

    async def _load_engine(
        self,
        model_id: str,
        force_lm: bool = False,
        runtime_settings: object | None = None,
    ) -> None:
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
        self._wake_process_memory_enforcer(active=True)
        load_started_at = entry.loading_started_at
        load_completed = False
        entry.abort_loading = False
        pre_load_memory = max(mx.get_active_memory(), get_phys_footprint())
        try:
            effective_type = entry.engine_type
            if force_lm and effective_type == "vlm":
                effective_type = "batched"
                logger.info(f"Loading model as LM (force_lm=True): {model_id}")
            else:
                logger.info(f"Loading model: {model_id}")

            # Retrieve per-model settings for post-load transforms
            model_settings = runtime_settings
            if model_settings is None and self._settings_manager is not None:
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

            # Check if DFlash is enabled -- takes priority over engine type
            # since DFlash has its own model loading pipeline
            engine = None
            if model_settings is not None:
                dflash_enabled = getattr(model_settings, "dflash_enabled", False)
                dflash_draft = getattr(model_settings, "dflash_draft_model", None)
                if (
                    dflash_enabled
                    and dflash_draft
                    and self._entry_is_diffusion_model(entry)
                ):
                    logger.warning(
                        "DFlash is not supported for diffusion models; "
                        "loading %s with its native VLM engine",
                        model_id,
                    )
                elif dflash_enabled and dflash_draft:
                    try:
                        from .engine.dflash import DFlashEngine

                        engine = DFlashEngine(
                            model_name=entry.model_path,
                            draft_model_path=dflash_draft,
                            draft_quant_enabled=getattr(
                                model_settings, "dflash_draft_quant_enabled", False
                            ),
                            draft_quant_weight_bits=getattr(
                                model_settings, "dflash_draft_quant_weight_bits", 4
                            ),
                            draft_quant_activation_bits=getattr(
                                model_settings, "dflash_draft_quant_activation_bits", 16
                            ),
                            draft_quant_group_size=getattr(
                                model_settings, "dflash_draft_quant_group_size", 64
                            ),
                            model_settings=model_settings,
                            fallback_engine_type=effective_type,
                            scheduler_config=self._scheduler_config,
                            omlx_ssd_cache_dir=getattr(
                                self._scheduler_config, "paged_ssd_cache_dir", None
                            ),
                        )
                        logger.info(
                            f"DFlash enabled for {model_id}, draft={dflash_draft}"
                        )
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
            # When unset, defaults to False -- repos with custom modeling_*.py
            # will fail to load until the user explicitly toggles this on
            # in the admin UI's model settings modal.
            trc = (
                bool(getattr(model_settings, "trust_remote_code", False))
                if model_settings
                else False
            )

            async def prefill_eviction_callback(
                eviction_request: object,
                *,
                _model_id: str = model_id,
            ) -> bool:
                return await self._evict_idle_lru_for_prefill(
                    exclude_model_id=_model_id,
                    eviction_request=eviction_request,
                )

            # Create engine based on engine type (if DFlash not active)
            if engine is None:
                if effective_type == "embedding":
                    engine = EmbeddingEngine(
                        model_name=entry.model_path,
                        trust_remote_code=trc,
                        scheduler_config=self._scheduler_config,
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
                        prefill_eviction_callback=prefill_eviction_callback,
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
                        prefill_eviction_callback=prefill_eviction_callback,
                    )

            _is_dflash_engine = (
                engine is not None and type(engine).__name__ == "DFlashEngine"
            )
            if _is_dflash_engine:
                await self._unload_other_dflash_engines(model_id)

            try:
                await engine.start()
            except Exception as start_error:
                if _is_dflash_engine:
                    # DFlash engine failed to start -- fall back to the
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
                            prefill_eviction_callback=prefill_eviction_callback,
                        )
                    else:
                        engine = BatchedEngine(
                            model_name=entry.model_path,
                            trust_remote_code=trc,
                            scheduler_config=self._scheduler_config,
                            model_settings=model_settings,
                            prefill_eviction_callback=prefill_eviction_callback,
                        )
                    try:
                        await engine.start()
                    except Exception as fallback_error:
                        raise RuntimeError(
                            f"DFlash load failed: {start_error}; "
                            f"{effective_type} fallback also failed: {fallback_error}"
                        ) from start_error
                    logger.info(
                        f"Successfully loaded {model_id} as {effective_type} "
                        f"(fallback from DFlash)"
                    )

                elif force_lm and entry.engine_type == "vlm":
                    # force_lm created a BatchedEngine but mlx-lm can't
                    # load this VLM model -- fall back to VLMBatchedEngine.
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
                        prefill_eviction_callback=prefill_eviction_callback,
                    )
                    try:
                        await engine.start()
                    except Exception as fallback_error:
                        raise RuntimeError(
                            f"LM load failed (force_lm=True): {start_error}; "
                            f"VLM fallback also failed: {fallback_error}"
                        ) from start_error

                    logger.info(
                        f"Successfully loaded {model_id} as VLM "
                        f"(fallback from force_lm)"
                    )
                elif entry.engine_type == "vlm":
                    # VLM loading failed -- fall back to LLM (BatchedEngine)
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
                        prefill_eviction_callback=prefill_eviction_callback,
                    )
                    try:
                        await engine.start()
                    except Exception as fallback_error:
                        raise RuntimeError(
                            f"VLM load failed: {start_error}; "
                            f"LLM fallback also failed: {fallback_error}"
                        ) from start_error

                    entry.model_type = "llm"
                    entry.engine_type = "batched"
                    logger.info(
                        f"Successfully loaded {model_id} as LLM " f"(fallback from VLM)"
                    )
                else:
                    raise

            # Check if memory enforcer requested abort during loading
            if entry.abort_loading:
                logger.warning(f"Model load aborted by memory enforcer: {model_id}")
                try:
                    await engine.stop()
                except Exception as e:
                    logger.warning(f"Error stopping aborted engine for {model_id}: {e}")
                gc.collect()
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    get_mlx_executor(),
                    lambda: (mx.synchronize(), mx.clear_cache()),
                )
                raise ModelLoadingError(
                    f"Model {model_id} load aborted: " f"process memory limit exceeded"
                )

            entry.engine = engine
            entry.last_access = time.time()
            self._current_model_memory += entry.estimated_size
            load_completed = True

            # VLM MTP: load MTP drafter (gemma4_assistant or qwen3_5_mtp) and attach to engine.
            # Fail-soft -- drafter load issues never block the target engine.
            if (
                model_settings is not None
                and getattr(model_settings, "vlm_mtp_enabled", False)
                and getattr(model_settings, "vlm_mtp_draft_model", None)
                and hasattr(engine, "set_vlm_mtp_drafter")
            ):
                drafter_id = model_settings.vlm_mtp_draft_model
                drafter_entry = self._entries.get(drafter_id)
                drafter_path = drafter_entry.model_path if drafter_entry else drafter_id

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
                        f"(drafter={drafter_id}): {e} -- toggle ignored"
                    )
                    drafter = None
                if drafter is not None:
                    engine.set_vlm_mtp_drafter(drafter)
                    logger.info(f"VLM MTP enabled for {model_id}, drafter={drafter_id}")
                else:
                    logger.warning(
                        f"VLM MTP toggle on for {model_id} but drafter "
                        f"load failed; toggle ignored"
                    )

            entry.runtime_settings_signature = self._engine_runtime_signature(
                model_id,
                model_settings,
                loaded_engine=engine,
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

            post_load_memory = max(mx.get_active_memory(), get_phys_footprint())
            observed_delta = max(0, post_load_memory - pre_load_memory)
            entry.actual_size = observed_delta or entry.estimated_size

            logger.info(
                f"Loaded model: {model_id} "
                f"(actual: {format_size(entry.actual_size)}, "
                f"estimated: {format_size(entry.estimated_size)}, "
                f"total: {format_size(self._current_model_memory)})"
            )
        finally:
            if (
                load_completed
                and load_started_at is not None
                and entry.estimated_size > 0
            ):
                elapsed = max(0.0, time.monotonic() - load_started_at)
                size_gb = entry.estimated_size / (1024**3)
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
            self._wake_process_memory_enforcer()

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
            "final_ceiling": self._current_ceiling(),
            "current_model_memory": self._current_model_memory,
            "model_count": len(self._entries),
            "loaded_count": sum(
                1 for e in self._entries.values() if e.engine is not None
            ),
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
                    "actual_size": e.actual_size,
                    "pinned": e.is_pinned,
                    "engine_type": e.engine_type,
                    "model_type": e.model_type,
                    "config_model_type": e.config_model_type,
                    "thinking_default": e.thinking_default,
                    "preserve_thinking_default": e.preserve_thinking_default,
                    "source_type": e.source_type,
                    "source_repo_id": e.source_repo_id,
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
                has_active = entry.engine.has_active_requests() or entry.in_use > 0

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
