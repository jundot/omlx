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
import copy
import gc
import json
import logging
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .cluster.deployment import ClusterDeployment
    from .cluster.registry import ClusterRegistry
    from .model_settings import ModelSettingsManager

import mlx.core as mx

from .engine import BaseEngine, BatchedEngine
from .engine.embedding import EmbeddingEngine
from .engine.reranker import RerankerEngine
from .engine.sts import STSEngine
from .engine.stt import STTEngine
from .engine.tts import TTSEngine
from .engine.vlm import VLMBatchedEngine
from .engine_core import get_mlx_executor
from .exceptions import (
    DEFAULT_CEILING_ADVICE,
    InsufficientMemoryError,
    ModelBusyError,
    ModelLoadingError,
    ModelNotFoundError,
    ModelTooLargeError,
    ModelUnavailableError,
    describe_ceiling_binding,
)
from .model_discovery import discover_models, format_size, is_realtime_stt_model
from .scheduler import SchedulerConfig
from .utils.proc_memory import get_phys_footprint

logger = logging.getLogger(__name__)

_FP16_BYTES = 2
_MAX_AFFINE_BYTES_PER_WEIGHT = 1.0625  # q8 plus fp16 scale/bias per group
_CPU_SHARE_MATERIALIZATION_HEADROOM = 1.5


def _positive_int(value: object) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def _aligned_share_rows(outputs: int, fraction: float) -> int:
    if outputs <= 0 or fraction <= 0:
        return 0
    return min(outputs, (int(outputs * fraction) // 64) * 64)


def _qwen35_cpu_share_estimated_bytes(
    model_path: str,
    settings: object | None,
) -> int | None:
    """Estimate peak bytes added while materializing Qwen CPU-share rows.

    The source checkpoint retains its packed weights. Gate/up and GDN add
    eager FP16 row slices, while down sharing additionally retains a copied
    quantized GPU suffix. The final multiplier covers the per-layer
    dequantize/concatenate scratch observed during eager preparation. ``None``
    means CPU sharing was requested for a Qwen checkpoint whose geometry could
    not be established safely; callers must use a conservative fallback.
    """

    if (
        settings is None
        or not bool(getattr(settings, "qwen35_ane_prefill_enabled", False))
        or not bool(getattr(settings, "qwen35_ane_prefill_cpu_enabled", False))
    ):
        return 0

    config_path = Path(model_path) / "config.json"
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(config, dict):
        return None
    text = config.get("text_config")
    if not isinstance(text, dict):
        text = config
    model_type = str(text.get("model_type") or config.get("model_type") or "")
    if not any(token in model_type for token in ("qwen3_5", "qwen3_6", "qwen3_8")):
        return 0

    hidden = _positive_int(text.get("hidden_size"))
    intermediate = _positive_int(text.get("intermediate_size"))
    layer_count = _positive_int(text.get("num_hidden_layers"))
    if not hidden or not intermediate or not layer_count:
        return None
    layer_count = min(
        layer_count,
        max(0, int(getattr(settings, "qwen35_ane_prefill_max_layers", 64) or 0)),
    )

    extra = 0.0
    gate_fraction = float(
        getattr(settings, "qwen35_ane_prefill_cpu_fraction", 0.0) or 0.0
    )
    gate_rows = _aligned_share_rows(intermediate, gate_fraction)
    if gate_rows:
        extra += layer_count * 2 * gate_rows * hidden * _FP16_BYTES
        if bool(getattr(settings, "qwen35_ane_prefill_fused_down", False)):
            # The fused CPU branch keeps the matching hidden-channel columns
            # of down_proj in FP16 as well as the gate/up rows above.
            extra += layer_count * gate_rows * hidden * _FP16_BYTES

    down_fraction = float(
        getattr(settings, "qwen35_ane_prefill_cpu_down_fraction", 0.0) or 0.0
    )
    down_rows = _aligned_share_rows(hidden, down_fraction)
    if down_rows and down_rows < hidden:
        cpu_weight = down_rows * intermediate * _FP16_BYTES
        gpu_suffix = (hidden - down_rows) * intermediate * _MAX_AFFINE_BYTES_PER_WEIGHT
        extra += layer_count * (cpu_weight + gpu_suffix)

    gdn_fraction = float(
        getattr(settings, "qwen35_ane_prefill_cpu_gdn_fraction", 0.0) or 0.0
    )
    if gdn_fraction > 0 and bool(getattr(settings, "qwen35_ane_prefill_gdn", True)):
        key_heads = _positive_int(text.get("linear_num_key_heads"))
        key_dim = _positive_int(text.get("linear_key_head_dim"))
        value_heads = _positive_int(text.get("linear_num_value_heads"))
        value_dim = _positive_int(text.get("linear_value_head_dim"))
        qkv_outputs = 2 * key_heads * key_dim + value_heads * value_dim
        z_outputs = value_heads * value_dim
        if not qkv_outputs or not z_outputs:
            return None
        gdn_rows = min(
            qkv_outputs,
            _aligned_share_rows(qkv_outputs + z_outputs, gdn_fraction),
        )
        layer_types = text.get("layer_types")
        if isinstance(layer_types, list):
            gdn_layers = sum(
                "linear" in str(layer_type).lower() for layer_type in layer_types
            )
        else:
            # Qwen hybrid checkpoints use full attention periodically. Without
            # the explicit map, charging every layer is the safe estimate.
            gdn_layers = _positive_int(text.get("num_hidden_layers"))
        gdn_layers = min(
            gdn_layers,
            max(
                0,
                int(getattr(settings, "qwen35_ane_prefill_gdn_max_layers", 48) or 0),
            ),
        )
        extra += gdn_layers * gdn_rows * hidden * _FP16_BYTES

    return int(extra * _CPU_SHARE_MATERIALIZATION_HEADROOM)


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
    text_only_size: int = 0  # Language-only estimate for VLM checkpoints (0 = n/a)
    actual_size: int | None = None  # Observed process-memory delta after load settles
    runtime_estimated_size: int | None = None  # Includes active load-time variants
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
    is_helper: bool = False  # Speculative-decoding drafter (dFlash/Assistant/MTP)
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
    pending_unload_allow_pinned: bool = False  # Explicit unload may override pinning
    # Requested load-time variant. This deliberately tracks the settings that
    # produced the engine, even when an optional accelerator fails soft and the
    # engine falls back, so identical requests keep reusing that fallback.
    runtime_settings_signature: tuple[tuple[str, str], ...] | None = None
    load_failed: bool = False  # Sticky until the next discovery refresh
    load_failure_message: str | None = None
    load_failure_at: float | None = None


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
            When that reads 0 (memory guard disabled), the pool falls back
            to `enforcer.get_admission_ceiling()` via `_get_admission_ceiling`
            for best-effort LRU eviction (#2290). Idle-model eviction
            starts at `enforcer.get_admission_soft_target()` via
            `_get_admission_soft_target` so the old model is unloaded
            before the new weights allocate (#2319). Until the callbacks
            are wired up the pool admits unconditionally.
        """
        self._entries: dict[str, EngineEntry] = {}
        self._lock = asyncio.Lock()
        self._current_model_memory = 0
        # Scanned model roots, kept for org-qualified display/upload names.
        self._model_dirs: list[Path] = []
        self._scheduler_config = scheduler_config or SchedulerConfig()
        self._process_memory_enforcer: object | None = None  # Set by server
        self._get_final_ceiling: object | None = None  # Set by server
        self._get_admission_ceiling: object | None = None  # Set by server
        self._get_admission_soft_target: object | None = None  # Set by server
        self._settings_manager: object | None = None  # Set by server
        self._cluster_registry: ClusterRegistry | None = None  # Set by server
        self._suppress_ttl: bool = False  # Suppress TTL during benchmarks
        # Requests whose prefill already got a pooled-buffer reclaim pass.
        # Prefill continuously refills MLX's buffer cache, so the reclaim
        # rung can "succeed" marginally on every pass of a long prompt while
        # the durable rung behind it (ANE bank release) is never reached; a
        # request coming back for more headroom escalates instead of
        # reclaiming again first. Bounded FIFO — request ids are transient.
        self._prefill_headroom_recurring: OrderedDict[str, None] = OrderedDict()
        self._load_seconds_per_gb_ema: float | None = None
        self._load_time_observations: int = 0
        self._lease_release_tasks: set[asyncio.Task[None]] = set()
        self._pending_unload_tasks: dict[str, asyncio.Task[None]] = {}
        self._shutting_down = False
        self.configure_hot_cache_budget()

    def _distributed_deployment_for_entry(
        self, entry: EngineEntry
    ) -> ClusterDeployment | None:
        if entry.engine is not None:
            # A newly activated registry record must not retroactively relabel
            # an already-loaded local engine. The deployment applies only
            # after that engine goes through the normal unload/load lifecycle.
            from .cluster.deployment import ClusterDeployment

            deployment = getattr(entry.engine, "deployment", None)
            return deployment if isinstance(deployment, ClusterDeployment) else None
        registry = self._cluster_registry
        if registry is None or entry.engine_type != "batched":
            return None
        return registry.get_for_model(entry.model_path)

    def _entry_resident_size(self, entry: EngineEntry) -> int:
        """Return this process's planned weights, not the full cluster model."""

        if entry.runtime_estimated_size is not None:
            return entry.runtime_estimated_size

        deployment = self._distributed_deployment_for_entry(entry)
        if deployment is None:
            return entry.estimated_size
        assignment = next(
            (item for item in deployment.assignments if item.rank == 0),
            None,
        )
        return (
            assignment.planned_weight_bytes
            if assignment is not None
            else entry.estimated_size
        )

    def _entry_runtime_resident_size(
        self,
        entry: EngineEntry,
        runtime_settings: object | None,
        *,
        base_size: int | None = None,
    ) -> int:
        """Include eager CPU-share storage in load and prefill accounting."""

        base = self._entry_resident_size(entry) if base_size is None else base_size
        if self._distributed_deployment_for_entry(entry) is not None:
            return base
        qwen4_offload, _, qwen4_estimate = self._qwen4_ple_offload_status(
            entry, runtime_settings
        )
        if qwen4_offload and qwen4_estimate is not None:
            base = min(base, qwen4_estimate.mmap_bytes)
        extra = _qwen35_cpu_share_estimated_bytes(entry.model_path, runtime_settings)
        if extra is None:
            # An enabled CPU path with unreadable geometry must not silently
            # retain the quantized estimate. One additional model-sized charge
            # is conservative and lets normal admission produce useful errors.
            extra = entry.estimated_size
            logger.warning(
                "Could not determine Qwen CPU-share geometry for %s; "
                "reserving one additional model-sized memory allowance",
                entry.model_id,
            )
        if extra > 0:
            logger.info(
                "Qwen CPU sharing adds %s to the projected memory for %s",
                format_size(extra),
                entry.model_id,
            )
        return base + extra

    def _qwen4_ple_offload_status(
        self,
        entry: EngineEntry,
        settings: object | None,
    ) -> tuple[bool, bool, object | None]:
        """Resolve requested/forced Qwen4 PLE mmap mode for this process."""

        model_type = (entry.config_model_type or "").replace("-", "_").lower()
        if model_type != "qwen4_exp":
            return False, False, None
        try:
            from .patches.mlx_vlm_qwen4_exp_compat.residency import (
                qwen4_exp_residency_estimate,
            )

            estimate = qwen4_exp_residency_estimate(entry.model_path)
        except (OSError, TypeError, ValueError):
            logger.debug(
                "Could not inspect Qwen4-Exp PLE residency for %s",
                entry.model_id,
                exc_info=True,
            )
            return False, False, None
        ceiling = self._fallback_admission_ceiling()
        if ceiling <= 0:
            ceiling = self._current_ceiling()
        forced = estimate.force_ssd_offload(ceiling)
        requested = bool(
            settings is not None and getattr(settings, "qwen4_ple_ssd_offload", False)
        )
        return requested or forced, forced, estimate if estimate.supported else None

    def _effective_qwen4_model_settings(
        self,
        entry: EngineEntry,
        settings: object | None,
    ) -> object | None:
        """Apply a forced mmap decision without mutating persisted settings."""

        enabled, forced, _ = self._qwen4_ple_offload_status(entry, settings)
        if not enabled or not forced or settings is None:
            return settings
        effective = copy.copy(settings)
        setattr(effective, "qwen4_ple_ssd_offload", True)
        return effective

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

    def _fallback_admission_ceiling(self) -> int:
        """Best-effort admission ceiling used when `_current_ceiling()` is 0.

        Wired to `enforcer.get_admission_ceiling`, which keeps returning
        the static ceiling while the memory guard is disabled so a model
        swap still evicts LRU models instead of overcommitting physical
        memory (#2290). Returns 0 when no callback is wired up (standalone
        pools admit unconditionally).
        """
        cb = self._get_admission_ceiling
        if cb is None:
            return 0
        try:
            return int(cb())
        except Exception:  # noqa: BLE001
            return 0

    def _admission_soft_target(self) -> int:
        """Soft watermark that pre-load eviction targets (#2319).

        Wired to `enforcer.get_admission_soft_target`. Eviction of idle
        LRU models starts once the projected total exceeds this, so an
        old model is unloaded *before* the new weights allocate instead
        of after the first request's prefill guard fires. Returns 0 when
        no callback is wired up (callers fall back to the ceiling).
        """
        cb = self._get_admission_soft_target
        if cb is None:
            return 0
        try:
            return int(cb())
        except Exception:  # noqa: BLE001
            return 0

    def _ceiling_binding_and_advice(
        self, *, ceiling: int, current: int, tail: str
    ) -> tuple[str | None, str | None]:
        """Name the ceiling that refused this load and the knob that moves it.

        A load refusal used to say "free system memory or lower
        memory_guard_tier" with no breakdown, which is the wrong knob twice
        over: the tier ladder runs safe → balanced → aggressive, so lowering
        it shrinks the ceiling further, and on a dynamic-bound machine the
        static / metal caps the user is likely to be shown elsewhere have
        room to spare. Reuses the scheduler's advice ladder so both
        rejection paths name the same constraint.

        Returns ``(None, None)`` when the enforcer is not wired up or its
        breakdown is unreadable; callers fall back to the generic advice.
        """
        enforcer = self._process_memory_enforcer
        getter = (
            getattr(enforcer, "get_ceiling_breakdown", None)
            if enforcer is not None
            else None
        )
        if not callable(getter):
            return None, None
        try:
            breakdown = getter()
            static = int(breakdown["static"])
            dynamic = int(breakdown["dynamic"])
            metal_cap = int(breakdown["metal_cap"])
        except Exception:  # noqa: BLE001
            return None, None
        if max(static, dynamic, metal_cap) <= 0:
            return None, None
        return describe_ceiling_binding(
            static=static,
            dynamic=dynamic,
            metal_cap=metal_cap,
            tier=str(getattr(enforcer, "memory_guard_tier", "") or ""),
            current=current,
            fmt=format_size,
            tail=tail,
        )

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
        if entry is not None:
            qwen4_offload, _, _ = self._qwen4_ple_offload_status(entry, settings)
            add("qwen4_ple_ssd_offload", qwen4_offload)

        turboquant_active = bool(data.get("turboquant_kv_enabled", False))
        add("turboquant_kv_enabled", turboquant_active)
        if turboquant_active:
            add("turboquant_kv_bits", data.get("turboquant_kv_bits", 4))
            add("turboquant_skip_last", data.get("turboquant_skip_last", True))

        qwen_ane_active = bool(data.get("qwen35_ane_prefill_enabled", False))
        add("qwen35_ane_prefill_enabled", qwen_ane_active)
        if qwen_ane_active:
            add(
                "qwen35_ane_prefill_sequence_length",
                data.get("qwen35_ane_prefill_sequence_length", 2048),
            )
            add(
                "qwen35_ane_prefill_tail_padding_min_tokens",
                data.get("qwen35_ane_prefill_tail_padding_min_tokens", 0),
            )
            add("qwen35_ane_prefill_fraction", data.get("qwen35_ane_prefill_fraction", 0.53))
            add(
                "qwen35_ane_prefill_fused_down",
                data.get("qwen35_ane_prefill_fused_down", False),
            )
            add("qwen35_ane_prefill_max_layers", data.get("qwen35_ane_prefill_max_layers", 64))
            add("qwen35_ane_prefill_dual_ane", data.get("qwen35_ane_prefill_dual_ane", True))
            add("qwen35_ane_prefill_gdn", data.get("qwen35_ane_prefill_gdn", True))
            if data.get("qwen35_ane_prefill_gdn", True):
                add(
                    "qwen35_ane_prefill_gdn_fraction",
                    data.get("qwen35_ane_prefill_gdn_fraction", 0.50),
                )
                add(
                    "qwen35_ane_prefill_gdn_max_layers",
                    data.get("qwen35_ane_prefill_gdn_max_layers", 48),
                )
            cpu_active = bool(data.get("qwen35_ane_prefill_cpu_enabled", False))
            add("qwen35_ane_prefill_cpu_enabled", cpu_active)
            if cpu_active:
                add(
                    "qwen35_ane_prefill_cpu_fraction",
                    data.get("qwen35_ane_prefill_cpu_fraction", 0.135),
                )
                add(
                    "qwen35_ane_prefill_cpu_down_fraction",
                    data.get("qwen35_ane_prefill_cpu_down_fraction", 0.0),
                )
                add(
                    "qwen35_ane_prefill_cpu_gdn_fraction",
                    data.get("qwen35_ane_prefill_cpu_gdn_fraction", 0.0),
                )
                add(
                    "qwen35_ane_prefill_cpu_threads",
                    data.get("qwen35_ane_prefill_cpu_threads", 8),
                )
                add(
                    "qwen35_ane_prefill_cpu_shared_resource",
                    data.get("qwen35_ane_prefill_cpu_shared_resource", True),
                )

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
            add("dflash_block_size", data.get("dflash_block_size"))
            add("dflash_verify_mode", data.get("dflash_verify_mode"))

        vlm_mtp_active = bool(data.get("vlm_mtp_enabled", False)) and has_value(
            "vlm_mtp_draft_model"
        )
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
        self._model_dirs = dirs

        if len(dirs) == 1:
            discovered = discover_models(dirs[0])
        else:
            discovered = discover_models_from_dirs(dirs)

        pinned_set = set(pinned_models or [])

        for model_id, info in discovered.items():
            existing = self._entries.get(model_id)
            if existing is not None and (
                existing.engine is not None or existing.is_loading
            ):
                # Loaded or loading model: preserve runtime state, only
                # update the pinned flag. Replacing an in-flight entry would
                # orphan the engine the load attaches on completion (#2307).
                existing.is_pinned = model_id in pinned_set
            else:
                # New or unloaded model: create fresh entry
                self._entries[model_id] = EngineEntry(
                    model_id=model_id,
                    model_path=info.model_path,
                    model_type=info.model_type,
                    engine_type=info.engine_type,
                    estimated_size=info.estimated_size,
                    text_only_size=getattr(info, "text_only_size", 0),
                    config_model_type=getattr(info, "config_model_type", ""),
                    thinking_default=getattr(info, "thinking_default", None),
                    preserve_thinking_default=getattr(
                        info, "preserve_thinking_default", None
                    ),
                    model_context_length=getattr(info, "model_context_length", None),
                    source_type=getattr(info, "source_type", "local"),
                    source_repo_id=getattr(info, "source_repo_id", None),
                    is_helper=getattr(info, "is_helper", False),
                    is_pinned=model_id in pinned_set,
                )

            if model_id in pinned_set:
                logger.info(f"Pinned model: {model_id}")

        # Remove entries no longer discovered and neither loaded nor loading
        discovered_ids = set(discovered.keys())
        stale = [
            mid
            for mid in self._entries
            if mid not in discovered_ids
            and self._entries[mid].source_type != "cluster"
            and self._entries[mid].engine is None
            and not self._entries[mid].is_loading
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

    @staticmethod
    def _select_cluster_path_match(
        matches: list[tuple[str, EngineEntry]],
    ) -> tuple[str, EngineEntry]:
        """Choose one public alias when discovery names one path twice.

        Hugging Face snapshots can be discovered once from an explicit model
        directory (the snapshot hash) and once from the HF cache (the repo
        ID). Those are aliases, not ambiguous model contents. Preserve the
        fail-closed behavior only when the aliases disagree about how the
        model must be served.
        """

        signatures = {
            (entry.model_type, entry.engine_type, entry.config_model_type)
            for _, entry in matches
        }
        if len(signatures) > 1:
            raise ValueError(
                "cluster model path is ambiguous across incompatible public "
                "model IDs: "
                + ", ".join(sorted(model_id for model_id, _ in matches))
            )
        return min(
            matches,
            key=lambda item: (
                item[1].engine is None,
                item[1].source_repo_id is None,
                item[0],
            ),
        )

    def resolve_cluster_model_id(self, model_path: str) -> str:
        """Resolve one downloaded LLM path to its public oMLX model ID.

        Cluster deployments are keyed by canonical model path while the public
        API and the engine pool are keyed by model ID.  Activation must join
        those namespaces before it starts a launcher; guessing from the final
        path component can select the wrong model when multiple model roots
        contain the same directory name.
        """

        candidate = Path(model_path).expanduser()
        candidate_key = (
            str(candidate.resolve()) if candidate.exists() else str(candidate)
        )
        matches = []
        for model_id, entry in self._entries.items():
            entry_path = Path(entry.model_path).expanduser()
            entry_key = (
                str(entry_path.resolve()) if entry_path.exists() else str(entry_path)
            )
            if entry_key == candidate_key:
                matches.append((model_id, entry))
        if not matches:
            raise ModelNotFoundError(model_path, list(self._entries.keys()))
        model_id, entry = self._select_cluster_path_match(matches)
        if entry.engine_type != "batched":
            raise ValueError(
                f"Model '{model_id}' is a {entry.model_type} model. "
                "Distributed cluster inference currently supports text LLM "
                "models only."
            )
        return model_id

    def register_cluster_model(
        self,
        model_path: str,
        *,
        estimated_size: int,
    ) -> tuple[str, bool]:
        """Register a staged remote model without advertising it as local.

        Rank zero may hold only its pipeline stage, so regular discovery
        correctly omits the directory. The distributed engine still needs an
        ``EngineEntry`` for API routing and lifecycle management. This
        synthetic entry always dispatches through ``DistributedBatchedEngine``;
        it can never load the partial directory as a standalone model.
        """

        path = Path(model_path).expanduser().resolve()
        config_path = path / "config.json"
        if not path.is_dir() or not config_path.is_file():
            raise ModelNotFoundError(model_path, list(self._entries.keys()))
        if estimated_size <= 0:
            raise ValueError("cluster model estimated size must be positive")

        candidate_key = str(path)
        exact = [
            (model_id, entry)
            for model_id, entry in self._entries.items()
            if str(Path(entry.model_path).expanduser().resolve()) == candidate_key
        ]
        if exact:
            model_id, entry = self._select_cluster_path_match(exact)
            if entry.engine_type != "batched":
                raise ValueError(
                    f"Model '{model_id}' is already registered as "
                    f"{entry.model_type}; stop or remove that local model "
                    "before activating it as a text cluster model."
                )
            return model_id, False

        try:
            config = json.loads(config_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cluster model config is unreadable: {exc}") from exc
        if not isinstance(config, dict):
            raise ValueError("cluster model config.json must contain an object")

        base_id = path.name or "cluster-model"
        model_id = base_id
        if model_id in self._entries:
            model_id = f"cluster--{base_id}"
        suffix = 2
        while model_id in self._entries:
            model_id = f"cluster--{base_id}-{suffix}"
            suffix += 1

        context = config.get("max_position_embeddings")
        if not isinstance(context, int) or context <= 0:
            context = None
        self._entries[model_id] = EngineEntry(
            model_id=model_id,
            model_path=str(path),
            model_type="llm",
            engine_type="batched",
            estimated_size=int(estimated_size),
            config_model_type=str(config.get("model_type") or ""),
            model_context_length=context,
            source_type="cluster",
        )
        logger.info(
            "Registered remote-sourced cluster model %s at %s",
            model_id,
            path,
        )
        return model_id, True

    def unregister_cluster_model(self, model_id: str) -> bool:
        """Remove an unloaded synthetic cluster entry after rollback/stop."""

        entry = self._entries.get(model_id)
        if entry is None or entry.source_type != "cluster":
            return False
        if entry.engine is not None or entry.is_loading:
            raise RuntimeError(
                f"cluster model '{model_id}' is still loaded and cannot be removed"
            )
        registry = self._cluster_registry
        if registry is not None and registry.get_for_model(entry.model_path) is not None:
            return False
        del self._entries[model_id]
        logger.info("Removed cluster-only model registration %s", model_id)
        return True

    async def prepare_cluster_reload(self, model_id: str) -> None:
        """Make a discovered model ready to adopt its registry deployment.

        An engine that was already loaded locally cannot be relabelled as a
        distributed engine after the registry changes.  Unload it under the
        pool lock, while refusing to interrupt any active request.  Pinned
        models may be reloaded because pinning is a residency preference, not
        a request to keep using the old execution topology.
        """

        async with self._lock:
            entry = self._entries.get(model_id)
            if entry is None:
                raise ModelNotFoundError(model_id, list(self._entries.keys()))
            if entry.engine is None:
                self._clear_load_failure(entry)
                return
            self._raise_if_reload_busy(entry, "activate distributed cluster")
            await self._unload_engine(model_id)
            self._clear_load_failure(entry)

    def _clear_load_failure(self, entry: EngineEntry) -> None:
        entry.load_failed = False
        entry.load_failure_message = None
        entry.load_failure_at = None

    def _mark_load_failure(self, entry: EngineEntry, exc: BaseException) -> None:
        entry.load_failed = True
        entry.load_failure_message = str(exc) or type(exc).__name__
        entry.load_failure_at = time.time()

    def _raise_if_model_path_missing_locked(
        self, model_id: str, entry: EngineEntry
    ) -> None:
        """Drop stale unloaded entries whose backing model directory vanished."""
        model_path = Path(entry.model_path)
        if model_path.exists() and (model_path / "config.json").exists():
            return

        if entry.engine is None:
            self._entries.pop(model_id, None)
        available = [mid for mid in self._entries if mid != model_id]
        raise ModelNotFoundError(model_id, available)

    def _raise_if_load_failed(self, model_id: str, entry: EngineEntry) -> None:
        if not entry.load_failed:
            return
        detail = entry.load_failure_message or "previous load attempt failed"
        logger.warning(
            "Skipping load retry for '%s' after cached failure: %s",
            model_id,
            detail,
        )
        raise ModelUnavailableError(
            model_id,
            f"Model '{model_id}' is unavailable after a previous load failure: {detail}. "
            "Reload models after fixing the files to retry.",
        )

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
        then active cluster deployment IDs, exposed profile model IDs, and
        model settings aliases. If those fail and input contains a provider
        prefix (e.g. "omlx/my-model"), strips the prefix and retries. Returns
        the original string if no match is found.
        """
        if model_id_or_alias in self._entries:
            return model_id_or_alias

        # Case-insensitive fallback
        ci_match = self._case_insensitive_entry_match(model_id_or_alias)
        if ci_match is not None:
            return ci_match

        # Cluster deployment IDs are private runtime handles, but older Chat
        # sessions and clients may have persisted one before the public model
        # ID was returned consistently. Accept an active deployment ID as a
        # compatibility alias while keeping /v1/models canonical.
        registry = self._cluster_registry
        if registry is not None:
            deployment = registry.get(model_id_or_alias)
            if deployment is not None:
                try:
                    return self.resolve_cluster_model_id(deployment.model)
                except (ModelNotFoundError, ValueError):
                    # A stale registry record must not make unrelated model
                    # resolution fail. The normal not-found path below will
                    # return the caller's original value.
                    pass

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

    def _entry_has_scheduler_work(self, entry: EngineEntry) -> bool:
        """Return True until deferred aborts have actually left the scheduler."""
        scheduler = self._resolve_scheduler_from_engine(entry.engine)
        if scheduler is None:
            return False
        has_requests = getattr(scheduler, "has_requests", None)
        if callable(has_requests):
            try:
                if has_requests():
                    return True
            except Exception:
                return True
        for attr in ("running", "waiting", "prefilling", "requests"):
            if getattr(scheduler, attr, None):
                return True
        return False

    def _entry_is_quiescent(self, entry: EngineEntry) -> bool:
        """Return True only after leases, collectors, and scheduler work drain."""
        return not (
            entry.in_use > 0
            or self._entry_has_active_requests(entry)
            or self._entry_has_scheduler_work(entry)
        )

    def _raise_if_reload_busy(self, entry: EngineEntry, operation: str) -> None:
        if self._entry_is_busy(entry):
            raise ModelBusyError(entry.model_id, operation)

    @staticmethod
    def _engine_has_usable_tokenizer(engine: object) -> bool:
        tokenizer = getattr(engine, "tokenizer", None)
        return tokenizer is not None and callable(getattr(tokenizer, "encode", None))

    def _validate_llm_engine_ready(self, model_id: str, engine: object | None) -> None:
        if engine is None:
            raise ModelLoadingError(
                model_id,
                f"Model '{model_id}' did not return a loaded engine.",
            )
        llm_engine_types = [BaseEngine]
        if isinstance(VLMBatchedEngine, type):
            llm_engine_types.append(VLMBatchedEngine)
        if isinstance(engine, tuple(llm_engine_types)) and not (
            self._engine_has_usable_tokenizer(engine)
        ):
            raise ModelLoadingError(
                model_id,
                f"Model '{model_id}' loaded without a usable tokenizer.",
            )

    def _mark_pending_unload_locked(
        self,
        model_id: str,
        reason: str,
        *,
        abort_requested: bool = False,
        allow_pinned: bool = False,
    ) -> bool:
        """Mark a loaded model for unload once it is no longer busy.

        Caller must hold ``self._lock``. Returns True when a pending marker was
        installed. The method deliberately does not unload by itself; call
        ``_unload_pending_if_idle_locked`` after abort/release state changes.
        Pinning is respected unless an explicit caller opts out.
        """
        entry = self._entries.get(model_id)
        if (
            entry is None
            or entry.engine is None
            or entry.is_loading
            or (entry.is_pinned and not allow_pinned)
        ):
            return False
        entry.pending_unload_reason = reason
        entry.pending_unload_allow_pinned = allow_pinned
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
                or (entry.is_pinned and not entry.pending_unload_allow_pinned)
                or not self._entry_is_quiescent(entry)
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
            or (entry.is_pinned and not entry.pending_unload_allow_pinned)
            or not self._entry_is_quiescent(entry)
        ):
            return False

        reason = entry.pending_unload_reason
        entry.pending_unload_reason = None
        entry.pending_unload_allow_pinned = False
        entry.abort_requested = False
        logger.warning(
            "Unloading pending model '%s' after activity drained (%s)",
            model_id,
            reason,
        )
        await self._unload_engine(model_id)
        return True

    def _finish_pending_unload_task(
        self,
        model_id: str,
        task: asyncio.Task[None],
    ) -> None:
        if self._pending_unload_tasks.get(model_id) is task:
            self._pending_unload_tasks.pop(model_id, None)
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Pending unload task failed for '%s'", model_id)

    def _schedule_pending_unload_locked(self, model_id: str) -> None:
        current = self._pending_unload_tasks.get(model_id)
        if current is not None and not current.done():
            return
        task = asyncio.create_task(
            self._wait_for_pending_unload(model_id),
            name=f"engine-pending-unload:{model_id}",
        )
        self._pending_unload_tasks[model_id] = task
        task.add_done_callback(
            lambda completed, mid=model_id: self._finish_pending_unload_task(
                mid, completed
            )
        )

    async def _wait_for_pending_unload(self, model_id: str) -> None:
        """Poll scheduler state without ever tearing down an in-flight MLX step."""
        while not self._shutting_down:
            async with self._lock:
                entry = self._entries.get(model_id)
                if (
                    entry is None
                    or entry.engine is None
                    or not entry.pending_unload_reason
                ):
                    return
                if await self._unload_pending_if_idle_locked(model_id):
                    return
            await asyncio.sleep(0.1)

    async def request_unload(
        self,
        model_id: str,
        *,
        reason: str = "manual unload",
    ) -> bool:
        """Unload now when idle, otherwise abort and unload after quiescence.

        Returns True when the engine was unloaded before this call returned and
        False when teardown was queued. New acquisitions are rejected while the
        pending marker is installed, so the engine can drain deterministically.
        """
        async with self._lock:
            entry = self._entries.get(model_id)
            if entry is None or entry.engine is None:
                return True
            if entry.is_loading:
                raise ModelLoadingError(
                    model_id,
                    f"Model '{model_id}' is still loading and cannot be unloaded yet",
                )
            if self._entry_is_quiescent(entry):
                await self._unload_engine(model_id)
                return True

            self._mark_pending_unload_locked(
                model_id,
                reason,
                abort_requested=True,
                allow_pinned=True,
            )
            abort_all = getattr(entry.engine, "abort_all_requests", None)
            if callable(abort_all):
                try:
                    await abort_all(
                        reason=(
                            f"Request aborted because model '{model_id}' is being unloaded"
                        ),
                        error_code="model_unloading",
                    )
                except TypeError:
                    # Non-batched engines may expose the older no-argument hook.
                    await abort_all()
                except Exception:
                    logger.warning(
                        "Failed to request abort before unloading '%s'",
                        model_id,
                        exc_info=True,
                    )

            if await self._unload_pending_if_idle_locked(model_id):
                return True
            self._schedule_pending_unload_locked(model_id)
            logger.warning(
                "Queued unload for model '%s' until active scheduler work drains",
                model_id,
            )
            return False

    def is_abort_requested(self, model_id: str | None) -> bool:
        if model_id is None:
            return False
        entry = self._entries.get(model_id)
        return bool(entry and entry.abort_requested)

    def get_abort_requested_reason(self, model_id: str | None) -> str | None:
        if model_id is None:
            return None
        entry = self._entries.get(model_id)
        if entry is None or not entry.abort_requested:
            return None
        return entry.pending_unload_reason or "request abort"

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
            if entry.pending_unload_reason:
                raise ModelBusyError(model_id, "start work while unload is pending")
            expected_signature = self._engine_runtime_signature(
                model_id,
                runtime_settings,
            )
            unloaded_for_admission = False

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
                    unloaded_for_admission = True
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
                    unloaded_for_admission = True
                elif entry.engine is not None:
                    self._validate_llm_engine_ready(model_id, entry.engine)
                    if entry.runtime_settings_signature is None:
                        entry.runtime_settings_signature = expected_signature
                    entry.last_access = time.time()
                    if _lease:
                        entry.in_use += 1
                    return entry.engine

            self._raise_if_model_path_missing_locked(model_id, entry)
            self._raise_if_load_failed(model_id, entry)

            # Pre-load admission against the memory ceiling from the
            # process memory enforcer (min of static and dynamic). Try
            # evicting LRU non-pinned models first; if the model still
            # cannot fit after evicting everything available, raise.
            #
            # Eviction starts at the enforcer's *soft* watermark, not the
            # ceiling (#2319): the soft..ceiling band is exactly the
            # hard-pressure zone, and a second model admitted into it kept
            # both models resident through the load (swapping for minutes)
            # only to have the first request's prefill guard evict the old
            # one anyway. Evicting down to the same soft target *before*
            # the new weights allocate fixes the ordering; refusing a load
            # still requires exceeding the ceiling.
            #
            # ceiling == 0 means the guard is disabled or the enforcer is
            # not wired up. Eviction on model swap must not die with the
            # guard (#2290): fall back to the best-effort admission
            # ceiling (static, guard-independent) and keep evicting, but
            # never refuse the load under it — with the guard off the
            # user opted out of hard limits.
            # A distributed coordinator admits only rank zero's planned shard,
            # not the complete model. A local VLM-shaped checkpoint served by
            # the text engine (force_lm or a model_type_override that flipped
            # engine_type to "batched") loads only its language weights, so
            # admit that path by the text-only estimate instead of the
            # vision-inclusive file size (#2385).
            deployment = self._distributed_deployment_for_entry(entry)
            admission_size = self._entry_resident_size(entry)
            if (
                deployment is None
                and entry.text_only_size
                and (force_lm or entry.engine_type == "batched")
            ):
                admission_size = entry.text_only_size
            admission_settings = runtime_settings
            if admission_settings is None and self._settings_manager is not None:
                get_settings = getattr(self._settings_manager, "get_settings", None)
                if callable(get_settings):
                    admission_settings = get_settings(model_id)
            admission_size = self._entry_runtime_resident_size(
                entry,
                admission_settings,
                base_size=admission_size,
            )
            admission_kind = "local shard" if deployment is not None else "model"
            ceiling = self._current_ceiling()
            best_effort = False
            if ceiling <= 0:
                ceiling = self._fallback_admission_ceiling()
                best_effort = ceiling > 0
            if ceiling > 0:
                soft_target = self._admission_soft_target()
                evict_target = min(soft_target, ceiling) if soft_target > 0 else ceiling
                evicted_any = unloaded_for_admission
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
                    projected = current + admission_size
                    if projected <= evict_target:
                        break
                    victim = self._find_lru_victim()
                    if victim is not None:
                        logger.info(
                            f"Evicting '{victim}' to fit '{model_id}' "
                            f"under the admission soft target "
                            f"({format_size(projected)} > "
                            f"{format_size(evict_target)})"
                        )
                        await self._unload_engine(victim)
                        evicted_any = True
                        continue
                    if projected <= ceiling:
                        # Above the soft target with nothing left to
                        # evict, but still under the ceiling: admit. The
                        # soft target only decides when eviction starts
                        # (#2319); refusal keeps the ceiling-only
                        # contract.
                        if evict_target < ceiling:
                            logger.info(
                                f"Admitting '{model_id}' above the "
                                f"admission soft target with no idle "
                                f"model left to evict "
                                f"({format_size(projected)} > "
                                f"{format_size(evict_target)}, ceiling "
                                f"{format_size(ceiling)})"
                            )
                        break
                    failure_current = current
                    failure_projected = projected
                    failure_label = "current"

                    if evicted_any:
                        # Nothing else to evict after unloading at least one
                        # model in this get_engine() call. Before failing,
                        # re-test against the *tracked committed* baseline.
                        # The phys_footprint term folded into `current` is the
                        # macOS kernel ledger, which can still count
                        # reclaimable residue from models we just evicted.
                        # Pinned/in-use models that could not be evicted remain
                        # counted in _current_model_memory, preserving the
                        # #1623 undercount guard. Without a local eviction,
                        # keep trusting phys_footprint because it may be
                        # unrelated process pressure rather than model residue.
                        committed = max(
                            mx.get_active_memory(), self._current_model_memory
                        )
                        committed_projected = committed + admission_size
                        if committed_projected <= ceiling:
                            logger.info(
                                f"Admitting '{model_id}': committed baseline "
                                f"{format_size(committed_projected)} fits ceiling "
                                f"{format_size(ceiling)} "
                                f"(live footprint {format_size(projected)} included "
                                "reclaimable residue from evicted models)"
                            )
                            break
                        failure_current = committed
                        failure_projected = committed_projected
                        failure_label = "committed"

                    if best_effort:
                        # Memory guard is off: evicting was all we could
                        # do. Admit over the static ceiling instead of
                        # refusing, matching the unguarded no-hard-limit
                        # contract.
                        logger.warning(
                            f"Loading '{model_id}' past the static memory "
                            f"ceiling with the memory guard disabled "
                            f"(projected {format_size(failure_projected)} > "
                            f"ceiling {format_size(ceiling)}, "
                            f"{failure_label} baseline) and nothing left to "
                            f"evict; the system may swap heavily."
                        )
                        break

                    # Still over budget under the applicable baseline. Use
                    # ModelTooLargeError when the model alone exceeds the
                    # ceiling (no chance of fitting), InsufficientMemoryError
                    # when current usage leaves no room.
                    if admission_size > ceiling:
                        binding, advice = self._ceiling_binding_and_advice(
                            ceiling=ceiling,
                            current=failure_current,
                            tail="use a smaller model",
                        )
                        raise ModelTooLargeError(
                            model_id,
                            admission_size,
                            ceiling,
                            binding=binding,
                            advice=advice,
                        )
                    binding, advice = self._ceiling_binding_and_advice(
                        ceiling=ceiling,
                        current=failure_current,
                        tail="unload another model",
                    )
                    label = f"{binding} memory ceiling" if binding else "memory ceiling"
                    raise InsufficientMemoryError(
                        required=admission_size,
                        current=failure_current,
                        message=(
                            f"Cannot load {model_id}: projected memory "
                            f"{format_size(failure_projected)} would exceed "
                            f"the {label} {format_size(ceiling)} "
                            f"({failure_label}: {format_size(failure_current)}, "
                            f"{admission_kind}: {format_size(admission_size)}). "
                            f"{advice or DEFAULT_CEILING_ADVICE}."
                        ),
                    )

            # Now load the model
            await self._load_engine(
                model_id,
                force_lm=force_lm,
                runtime_settings=runtime_settings,
            )

            loaded = self._entries[model_id]
            self._validate_llm_engine_ready(model_id, loaded.engine)
            if _lease:
                loaded.in_use += 1
            return loaded.engine

    async def _release_engine_lease(self, model_id: str) -> None:
        async with self._lock:
            e = self._entries.get(model_id)
            if e is not None and e.in_use > 0:
                e.in_use -= 1
            await self._unload_pending_if_idle_locked(model_id)

    def _finish_lease_release_task(self, task: asyncio.Task[None]) -> None:
        self._lease_release_tasks.discard(task)
        try:
            task.result()
        except asyncio.CancelledError:
            logger.warning("Engine lease release task was cancelled")
        except Exception:
            logger.exception("Engine lease release task failed")

    async def _drain_lease_release_tasks(self) -> None:
        while self._lease_release_tasks:
            tasks = tuple(self._lease_release_tasks)
            await asyncio.gather(*tasks, return_exceptions=True)

    async def release_engine(self, model_id: str) -> None:
        """Release one in-use lease even if the caller is cancelled.

        ASGI disconnect cancellation can arrive while this release is waiting
        for the pool lock. Run the lock-taking operation in its own task so the
        lease still drains after the cancelled request task exits.
        """
        task = asyncio.create_task(
            self._release_engine_lease(model_id),
            name=f"engine-lease-release:{model_id}",
        )
        self._lease_release_tasks.add(task)
        task.add_done_callback(self._finish_lease_release_task)
        await asyncio.shield(task)

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
        reclaim_attempted = False
        ane_release_attempted = False
        # Snapshot once per call: "a PREVIOUS pass for this request already
        # got a reclaim". Marking inside this call must not flip it.
        recurring = request_id in self._prefill_headroom_recurring
        async with self._lock:
            while True:
                current = max(
                    mx.get_active_memory(),
                    get_phys_footprint(),
                    self._current_model_memory,
                )
                if current + predicted <= target:
                    # A model eviction and/or the pooled-buffer reclaim below
                    # created enough headroom; signal the caller to re-admit.
                    # reclaim_attempted stands in for "the reclaim freed
                    # memory": the helper's own footprint delta is
                    # process-wide and can be masked by concurrent
                    # allocation, but reaching this check with headroom
                    # after an attempt means admission will now succeed.
                    return evicted_any or reclaim_attempted or ane_release_attempted

                victim = self._find_lru_prefill_eviction_victim(
                    exclude_model_id=exclude_model_id
                )
                if victim is None:
                    # No idle model left to evict -- the "No idle model
                    # evicted" case that used to reject outright even when
                    # tens of GB were reclaimable. Two rungs remain: the
                    # cheap pooled-buffer reclaim, and shedding the
                    # requesting model's own ANE prefill banks. Ordering
                    # matters: prefill continuously refills MLX's buffer
                    # cache, so on a long prompt the reclaim can "succeed"
                    # by a marginal few GB on every pass while the durable
                    # rung is never reached — a request that already had a
                    # reclaim pass and is back for more headroom escalates
                    # straight to the bank release instead.
                    if recurring and not ane_release_attempted:
                        ane_release_attempted = True
                        await self._release_ane_prefill_for_headroom(
                            exclude_model_id, request_id
                        )
                        # Re-measure regardless of the reported delta: the
                        # footprint reading is process-wide, so concurrent
                        # allocation can mask a real release as 0 bytes.
                        continue
                    if not reclaim_attempted:
                        # Return MLX's pooled Metal buffers (freed by
                        # finished requests but still cached, so
                        # get_phys_footprint stays high) to the OS on the
                        # requesting engine's own MLX thread, then let the
                        # loop re-measure.
                        reclaim_attempted = True
                        self._prefill_headroom_recurring[request_id] = None
                        while len(self._prefill_headroom_recurring) > 512:
                            self._prefill_headroom_recurring.popitem(last=False)
                        await self._reclaim_pooled_buffers_for_prefill(
                            exclude_model_id, request_id
                        )
                        # Re-measure regardless of the reported delta: the
                        # helper measures a process-wide footprint, so
                        # concurrent allocation on another engine can mask a
                        # real reclaim as 0 bytes freed. The loop re-checks
                        # the target with a fresh reading; reclaim_attempted
                        # keeps this branch from running twice.
                        continue
                    if not ane_release_attempted:
                        # Last rung before giving up: shed the requesting
                        # model's own ANE prefill banks. They hold the packed
                        # weight blobs mapped into the native programs (~13 GB
                        # for a 27B at the default fractions) and are purely
                        # an accelerator — the per-module failure-latch
                        # fallback serves the same modules on GPU — so at
                        # long context the trade is a slower-but-unthrottled
                        # prefill instead of chunks collapsing to the floor.
                        # Banks come back at the model's next load.
                        ane_release_attempted = True
                        await self._release_ane_prefill_for_headroom(
                            exclude_model_id, request_id
                        )
                        # Re-measure regardless of the reported delta, for
                        # the same reason as the pooled reclaim above.
                        continue
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

    @staticmethod
    def _resolve_engine_core_from_engine(engine: object) -> object | None:
        """Resolve the EngineCore owning an entry's scheduler and MLX thread."""
        if getattr(engine, "scheduler", None) is not None:
            return engine
        try:
            return engine._engine.engine  # type: ignore[attr-defined]
        except AttributeError:
            return None

    async def _reclaim_pooled_buffers_for_prefill(
        self, model_id: str, request_id: str
    ) -> int:
        """Return MLX's pooled Metal buffers to the OS; report bytes freed.

        A warm server's resident baseline creeps between requests: finished
        requests free their KV / activation arrays into MLX's buffer *cache*
        (retained for reuse) rather than handing the pages back to the OS, so
        ``get_phys_footprint`` -- the resident figure the prefill guard reads
        -- stays high even though the bytes are reclaimable.

        The clear runs on the requesting engine's own MLX thread through the
        scheduler's ``_reclaim_prefill_headroom``, which synchronizes that
        engine's generation stream under ``_mx_buffer_access_lock`` before
        clearing (issues #300, #888, #1106) -- clearing from any other thread
        could release cached buffers still referenced by in-flight
        ``mx.async_eval`` command buffers. The engine's step loop is parked
        awaiting this eviction callback, so its executor is free. A failing
        reclaim is contained (#435 class): the request is then rejected
        exactly as if nothing had been reclaimable. ``clear_cache`` releases
        only unused cached buffers, so arrays an in-flight request still
        references are never touched, and the shared hot / prefix cache is
        left intact (it is SSD-backed and reclaimed separately by the
        enforcer).

        Returns:
            Bytes handed back to the OS (``get_phys_footprint`` delta, >= 0).
        """
        entry = self._entries.get(model_id)
        engine = entry.engine if entry is not None else None
        core = (
            self._resolve_engine_core_from_engine(engine)
            if engine is not None
            else None
        )
        scheduler = getattr(core, "scheduler", None)
        reclaim = getattr(scheduler, "_reclaim_prefill_headroom", None)
        executor = getattr(core, "_mlx_executor", None)
        if not callable(reclaim) or executor is None:
            # Engine mid-teardown (executor dropped at close) or an engine
            # shape without the scheduler helper: skip -- reject as before.
            return 0

        def _reclaim_on_engine_thread() -> None:
            gc.collect()
            reclaim()

        before = get_phys_footprint()
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(executor, _reclaim_on_engine_thread)
        except Exception as e:
            logger.warning(
                "Pooled-buffer reclaim failed for prefill request %s: %s",
                request_id,
                e,
            )
            return 0
        freed = max(0, before - get_phys_footprint())
        if freed > 0:
            logger.info(
                "Reclaimed %s of pooled Metal buffers for prefill request %s "
                "(no idle model to evict)",
                format_size(freed),
                request_id,
            )
        return freed

    async def _release_ane_prefill_for_headroom(
        self, model_id: str, request_id: str
    ) -> int:
        """Release the requesting model's ANE prefill banks; report bytes freed.

        The compiled banks keep the packed weight blobs mapped into the
        native ANE programs for the model's whole residency, so a config
        tuned at a short calibration length silently competes with the KV
        cache at long context — on a 27B at mlp 0.35 / gdn 0.45 the banks
        hold ~13 GB, which is the difference between full 2048-token prefill
        chunks and the guard throttling to the floor. Shedding them is safe:
        the release latches every sliced module through the existing
        per-module failure flags, so the dispatch sites fall back to stock
        GPU compute exactly as they do after a warmup failure. The banks are
        rebuilt at the model's next load.

        Runs on the requesting engine's own MLX thread; its step loop is
        parked awaiting this eviction callback, so no prefill dispatch is in
        flight while references are dropped. A failing release is contained:
        the request is then throttled exactly as if nothing had been
        releasable.

        Returns:
            Bytes handed back to the OS (``get_phys_footprint`` delta, >= 0),
            0 when nothing was released.
        """
        entry = self._entries.get(model_id)
        engine = entry.engine if entry is not None else None
        core = (
            self._resolve_engine_core_from_engine(engine)
            if engine is not None
            else None
        )
        executor = getattr(core, "_mlx_executor", None)
        model = (
            getattr(engine, "_model", None) or getattr(engine, "_vlm_model", None)
            if engine is not None
            else None
        )
        if executor is None or model is None:
            logger.info(
                "ANE bank release skipped for request %s: %s not resolvable "
                "on '%s'",
                request_id,
                "executor" if executor is None else "model object",
                model_id,
            )
            return 0
        try:
            from .patches.qwen35_ane_prefill import release_qwen35_ane_prefill
        except Exception:  # noqa: BLE001 - patch optional at runtime
            return 0

        def _release_on_engine_thread() -> tuple[int, int]:
            released, programs = release_qwen35_ane_prefill(model)
            if released:
                gc.collect()
            return released, programs

        before = get_phys_footprint()
        loop = asyncio.get_running_loop()
        try:
            released, programs = await loop.run_in_executor(
                executor, _release_on_engine_thread
            )
        except Exception as e:
            logger.warning(
                "ANE prefill bank release failed for request %s: %s",
                request_id,
                e,
            )
            return 0
        if not released:
            logger.info(
                "No ANE prefill slices to release on '%s' for request %s",
                model_id,
                request_id,
            )
            return 0
        freed = max(0, before - get_phys_footprint())
        # The load-time admission reservation priced these I/O surfaces; drop
        # it so later passes stop pausing for memory that no longer exists.
        # The next load re-prices it from the rebuilt banks.
        monitor = getattr(getattr(core, "scheduler", None), "memory_monitor", None)
        if monitor is not None and hasattr(monitor, "clear_ane_prefill_transient"):
            monitor.clear_ane_prefill_transient()
        logger.warning(
            "Released ANE prefill banks on '%s' for prefill headroom "
            "(request=%s, %d modules, %d programs, freed %s); the model "
            "serves GPU-only prefill until its next load",
            model_id,
            request_id,
            released,
            programs,
            format_size(freed),
        )
        return freed

    def _other_entries_serving(self, model_id: str) -> bool:
        """True when any other entry is serving or loading.

        Used by the settle barrier in ``_unload_engine``: the barrier's
        freed-memory check is a delta of the process-global
        ``mx.get_active_memory()`` gauge, which only measures THIS unload
        while no other engine is allocating concurrently. A loading entry
        (``is_loading=True``, ``engine`` still None) allocates weights at
        full speed, so it must count as concurrent activity too — else the
        barrier burns its rounds against a gauge that can even read
        negative and logs a bogus timeout (#2312).
        """
        # Snapshot the items: admin unload routes call _unload_engine without
        # the pool lock, so discover_models() can mutate _entries mid-iteration.
        for mid, e in list(self._entries.items()):
            if mid == model_id:
                continue
            if e.is_loading:
                return True
            if e.engine is None:
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
        distributed = self._distributed_deployment_for_entry(entry) is not None
        resident_size = self._entry_resident_size(entry)
        pre_unload_active = 0 if distributed else mx.get_active_memory()

        try:
            await entry.engine.stop()
        except Exception as e:
            if distributed:
                # Keep the supervisor reachable and the planned memory
                # accounted so a later unload can retry process teardown.
                logger.error(
                    f"Distributed teardown failed for {model_id}; "
                    "keeping the engine registered for retry",
                    exc_info=True,
                )
                self._wake_process_memory_enforcer()
                raise
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
        entry.pending_unload_allow_pinned = False
        entry.runtime_settings_signature = None
        entry.runtime_estimated_size = None

        if distributed:
            # Cluster weights live in supervised rank processes, not this
            # process's Metal allocator. Successful supervisor teardown is
            # the memory barrier; polling mx.get_active_memory() here would
            # wait against an unrelated gauge and then run emergency reclaim.
            gc.collect()
            self._current_model_memory = max(
                0,
                self._current_model_memory - resident_size,
            )
            logger.info(
                f"Unloaded distributed model: {model_id}, "
                f"released local shard process "
                f"({format_size(resident_size)} planned)"
            )
            self._wake_process_memory_enforcer()
            return

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
        settle_tolerance = max(2 * 1024**3, int(resident_size * 0.05))
        min_expected_freed = max(0, resident_size - settle_tolerance)
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
        self._current_model_memory = max(0, self._current_model_memory - resident_size)

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

    def _schedule_failed_load_reclaim(
        self, model_id: str, pre_load_memory: int
    ) -> None:
        """Reclaim memory left behind by a failed model load.

        When a load raises partway through (e.g. weights loaded, processor
        construction failed), the weights are often reachable only via the
        propagating exception's traceback frames. Spawn a background task
        that waits briefly for the exception to be handled and dropped, then
        runs gc + synchronize + clear_cache rounds until the live memory
        reading returns near its pre-load level (or rounds are exhausted).
        """

        async def _reclaim() -> None:
            loop = asyncio.get_running_loop()
            # 2 GB slack over the pre-load level mirrors the unload settle
            # barrier's small-model tolerance floor.
            target = pre_load_memory + 2 * 1024**3
            current = 0
            for _round in range(6):
                await asyncio.sleep(0.5 if _round == 0 else 1.0)
                gc.collect()
                await loop.run_in_executor(
                    get_mlx_executor(),
                    lambda: (mx.synchronize(), mx.clear_cache()),
                )
                current = max(mx.get_active_memory(), get_phys_footprint())
                if current <= target:
                    logger.info(
                        f"Reclaimed memory after failed load of '{model_id}': "
                        f"current={format_size(current)} "
                        f"(pre-load={format_size(pre_load_memory)})"
                    )
                    self._wake_process_memory_enforcer()
                    return
            logger.warning(
                f"Post-failed-load reclaim for '{model_id}' did not settle: "
                f"current={format_size(current)} "
                f"(pre-load={format_size(pre_load_memory)}). A server restart "
                f"may be required to release the leaked memory."
            )
            self._wake_process_memory_enforcer()

        # Keep a reference so the task isn't garbage-collected mid-flight;
        # one slot suffices -- a newer failure supersedes the previous task.
        self._failed_load_reclaim_task = asyncio.get_running_loop().create_task(
            _reclaim()
        )

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
        entry_detached = False
        entry.abort_loading = False
        resident_size = self._entry_resident_size(entry)
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
            model_settings = self._effective_qwen4_model_settings(entry, model_settings)

            deployment = self._distributed_deployment_for_entry(entry)
            base_resident_size = self._entry_resident_size(entry)
            if (
                deployment is None
                and entry.text_only_size
                and (force_lm or entry.engine_type == "batched")
            ):
                base_resident_size = entry.text_only_size
            resident_size = self._entry_runtime_resident_size(
                entry,
                model_settings,
                base_size=base_resident_size,
            )
            entry.runtime_estimated_size = resident_size

            # Wire the correct model_id / model_path into the shared scheduler
            # config so every engine (Batched/VLM/DFlash/Embedding) sees the
            # right values when it builds `SchedulerConfig` internally.
            self._scheduler_config.model_name = model_id
            self._scheduler_config.model_path = entry.model_path

            # Native MTP forces LM-only dispatch even for VLM models. Vision
            # encoder weights are ignored because the patched mtp_forward only
            # exists on the language model path. mtp_enabled was already
            # validated as mutually exclusive with dflash in
            # metal-knowledge: with the mlx-vlm runtime MTP patch (see
            # omlx/patches/mlx_vlm_mtp/qwen35_moe_vlm_runtime.py) VLM models
            # can run MTP natively while keeping vision intact. The old
            # force-LM-dispatch shortcut here is obsolete for patched
            # model families; let VLMBatchedEngine handle MTP-enabled VLMs.
            pass

            # Check if DFlash is enabled -- takes priority over engine type
            # since DFlash has its own model loading pipeline
            engine = None
            deployment = deployment if effective_type == "batched" else None
            if deployment is None and model_settings is not None:
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
                if deployment is not None:
                    from .engine.distributed import DistributedBatchedEngine

                    deployment = replace(
                        deployment,
                        trust_remote_code=trc,
                    )
                    engine = DistributedBatchedEngine(
                        deployment,
                        enable_thinking=getattr(
                            model_settings, "enable_thinking", None
                        ),
                        model_settings=model_settings,
                    )
                    logger.info(
                        "Distributed inference enabled for %s: ranks=%d "
                        "backend=%s plan=%s",
                        model_id,
                        deployment.world_size,
                        deployment.backend,
                        deployment.plan_hash[:16],
                    )
                elif effective_type == "embedding":
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
                        f"Successfully loaded {model_id} as LLM (fallback from VLM)"
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
                    model_id,
                    f"Model '{model_id}' load aborted: process memory limit exceeded",
                )

            self._validate_llm_engine_ready(model_id, engine)
            entry.engine = engine
            entry.last_access = time.time()
            self._current_model_memory += resident_size
            load_completed = True
            self._clear_load_failure(entry)

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

            # Keep the requested construction variant as the reuse key. DFlash
            # and VLM MTP are fail-soft: either can leave a normal engine in
            # place. Recording that effective engine as a different variant
            # makes the next identical concurrent request attempt a reload and
            # fail with ModelBusyError before it reaches the scheduler (#2406).
            entry.runtime_settings_signature = self._engine_runtime_signature(
                model_id, model_settings
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
            entry.actual_size = observed_delta or resident_size

            # Registry consistency check: a lockless mutator (a
            # discover_models() rescan or the runtime model-directory
            # reload) may have replaced or dropped this model's entry at an
            # await point above. The engine was then attached to a stale
            # object unreachable from the pool; keeping it running would
            # strand the weights until a process restart (#2307).
            if self._entries.get(model_id) is not entry:
                entry_detached = True
                logger.warning(
                    f"Registry entry for '{model_id}' changed during load; "
                    f"releasing the freshly loaded engine "
                    f"({format_size(resident_size)})"
                )
                entry.engine = None
                self._current_model_memory = max(
                    0, self._current_model_memory - resident_size
                )
                try:
                    await engine.stop()
                except Exception as e:
                    logger.warning(
                        f"Error stopping orphaned engine for {model_id}: {e}"
                    )
                gc.collect()
                await loop.run_in_executor(
                    get_mlx_executor(),
                    lambda: (mx.synchronize(), mx.clear_cache()),
                )
                raise ModelLoadingError(
                    model_id,
                    f"Model '{model_id}' was removed or replaced while it "
                    "was loading; the loaded engine was released. Retry "
                    "the request.",
                )

            logger.info(
                f"Loaded model: {model_id} "
                f"(actual: {format_size(entry.actual_size)}, "
                f"local estimate: {format_size(resident_size)}, "
                f"full model: {format_size(entry.estimated_size)}, "
                f"total: {format_size(self._current_model_memory)})"
            )
        except Exception as exc:
            # A failed load can leave tens of GB of just-loaded weights
            # reachable only through the propagating exception's traceback
            # frames (loader-internal locals). Running gc/clear_cache
            # synchronously here is useless -- the exception is still alive
            # in the caller. Schedule a deferred reclaim that runs after the
            # exception has been handled and dropped, so the buffers are
            # actually released; otherwise the process footprint stays
            # inflated and the memory-ceiling admission check rejects all
            # subsequent loads until a server restart.
            self._schedule_failed_load_reclaim(model_id, pre_load_memory)
            if not entry.abort_loading and not entry_detached:
                self._mark_load_failure(entry, exc)
                logger.exception(
                    "Model load failed for '%s'; caching failure until next discovery refresh",
                    model_id,
                )
                raise ModelUnavailableError(
                    model_id,
                    f"Model '{model_id}' failed to load: {entry.load_failure_message}. "
                    "Reload models after fixing the files to retry.",
                ) from exc
            raise
        finally:
            if load_completed and load_started_at is not None and resident_size > 0:
                elapsed = max(0.0, time.monotonic() - load_started_at)
                size_gb = resident_size / (1024**3)
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
                        f"for {model_id} ({elapsed:.1f}s, "
                        f"{format_size(resident_size)} local); "
                        f"EMA={self._load_seconds_per_gb_ema:.2f}s/GB"
                    )
            entry.is_loading = False
            entry.loading_started_at = None
            entry.abort_loading = False
            if not load_completed:
                entry.runtime_estimated_size = None
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
        self._shutting_down = True
        pending_tasks = tuple(self._pending_unload_tasks.values())
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)
        await self._drain_lease_release_tasks()
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
                    "resident_estimated_size": self._entry_resident_size(e),
                    "distributed": (
                        self._distributed_deployment_for_entry(e) is not None
                    ),
                    "actual_size": e.actual_size,
                    "pinned": e.is_pinned,
                    "engine_type": e.engine_type,
                    "model_type": e.model_type,
                    "config_model_type": e.config_model_type,
                    "realtime_stt": is_realtime_stt_model(
                        e.model_type, e.config_model_type
                    ),
                    "model_context_length": e.model_context_length,
                    "is_helper": e.is_helper,
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
