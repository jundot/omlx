# SPDX-License-Identifier: Apache-2.0
"""
DFlash engine for block diffusion speculative decoding.

This engine wraps dflash-mlx (>= 0.1.5) to provide faster decoding on Apple
Silicon for Qwen, Gemma4, and Laguna model families. By default it serves all
requests through dflash; setting ``model_settings.dflash_max_ctx`` opts into
evicting the dflash models and delegating long-context requests to omlx's
BatchedEngine/VLMBatchedEngine (paged cache, SSD cache, continuous batching).
"""

import asyncio
import copy
import gc
import json
import logging
import math
import re
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import mlx.core as mx

from ..adapter.output_parser import detect_output_parser
from ..api.tool_calling import convert_tools_for_template
from ..api.utils import clean_special_tokens, detect_and_strip_partial
from ..cache.observability import CacheRateTracker
from ..memory_monitor import (
    MemoryMonitor,
    raise_if_prefill_exceeds,
    set_model_info_from_model,
)
from ..reasoning_effort import apply_chat_template_with_reasoning_effort_fallback
from ..utils.generation_config import load_generation_config_token_ids
from ..utils.model_loading import maybe_apply_pre_load_patches
from ..utils.proc_memory import get_phys_footprint
from ..utils.tokenizer import create_streaming_detokenizer
from .base import (
    ActivityTrackingMixin,
    BaseEngine,
    GenerationOutput,
    _warn_scheduler_unreachable_once,
)

logger = logging.getLogger(__name__)


def is_dflash_compatible(model_path: str | Path) -> tuple[bool, str]:
    """Decide whether ``model_path`` can run on the current dflash backend.

    DFlash 0.1.10+omlx.4 registers QwenGdnTargetOps, Gemma4TargetOps, and
    MuseGlimmerTargetOps; oMLX adds a
    Laguna target/draft adapter. The top-level ``model_type`` is the canonical
    discriminator: Gemma4 multimodal
    configs use ``gemma4`` at the top, while MTP-only variants (e.g. the
    Gemma4 ``-assistant`` checkpoint) declare ``gemma4_assistant`` even
    though their nested ``text_config.model_type`` is still ``gemma4_text``.
    Reading top-level only keeps the gate aligned with what dflash will
    actually load.

    ``gemma4_unified`` (current mlx-community/lmstudio Gemma 4 exports,
    #2153) is accepted too: mlx-lm's MODEL_REMAPPING loads those
    checkpoints through the same ``gemma4`` module — the vision/audio
    towers are dropped in sanitize and the text stack DFlash drives is
    identical, which live-testing against z-lab's 12B drafter confirmed.

    ``muse_glimmer`` is a VLM at the top level, but dflash-mlx ships a
    text-only mlx-lm module for it (vision weights dropped in sanitize);
    image requests keep flowing through the VLM fallback engine. Note
    Meta named its DFlash drafter ``-assistant`` — drafter routing keys
    on ``config_model_type == "muse_glimmer_assistant"``, not on oMLX's
    historical "-assistant means MTP" name convention.

    Returns:
        (is_compatible, reason). ``reason`` is empty when compatible.
    """
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return False, f"config.json not found at {config_path}"
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return False, f"failed to read config.json: {e}"

    model_type = str(cfg.get("model_type") or "").lower()

    is_qwen = "qwen" in model_type
    is_gemma4 = model_type in ("gemma4", "gemma4_text", "gemma4_unified")
    is_laguna = model_type == "laguna"
    is_muse = model_type in ("muse_glimmer", "muse_glimmer_text")
    if not (is_qwen or is_gemma4 or is_laguna or is_muse):
        return False, (
            f"DFlash supports only Qwen, Gemma4, Laguna, and Muse Glimmer "
            f"models (model_type='{cfg.get('model_type', '')}')"
        )
    return True, ""


def _format_phase_timings(phase_timings_us: object) -> str:
    """Compact per-phase summary for the completion log, in milliseconds.

    The dflash SummaryEvent reports where each cycle's time went
    (prefill / draft / verify / replay / commit); without surfacing it the
    server log gives no way to tell which phase dominates a slow request.
    """
    if not isinstance(phase_timings_us, dict) or not phase_timings_us:
        return ""

    def _ms(key: str) -> str:
        try:
            return f"{float(phase_timings_us.get(key, 0.0)) / 1000.0:.1f}"
        except (TypeError, ValueError):
            return "0.0"

    return (
        f", phases[prefill={_ms('prefill')}ms"
        f" draft={_ms('draft')}ms"
        f"(first={_ms('draft_prefill')}/incr={_ms('draft_incremental')})"
        f" verify={_ms('verify')}ms"
        f" replay={_ms('replay')}ms"
        f" commit={_ms('commit')}ms]"
    )


# Precision suffixes Poolside uses for drafts trained against a quantized
# target ("Laguna-S-2.1-DFlash-NVFP4" etc.). Anything else after "-DFlash-"
# (e.g. z-lab's "-b16" block-size suffix) makes no precision claim.
_DRAFT_PRECISION_TAGS = frozenset(
    {"NVFP4", "INT4", "INT8", "FP8", "FP4", "MXFP4"}
)


def _canonical_precision_tag(value: object) -> str | None:
    """Normalize an explicit precision label from a name or config value."""
    text = str(value or "").upper().replace("_", "-")
    for tag in ("NVFP4", "MXFP4", "INT4", "INT8", "FP8", "FP4"):
        if tag in text.replace("-", ""):
            return tag
    if re.search(r"(?:^|[-/])OQ\d", text):
        return "OQ"
    if "BFLOAT16" in text or "BF16" in text:
        return "BF16"
    if "FLOAT16" in text or re.search(r"(?:^|[-/])FP16(?:$|[-/])", text):
        return "FP16"
    return None


def _target_precision_tag(
    target_name: str,
    target_config: dict | None,
) -> str | None:
    """Best-effort target precision, preferring checkpoint metadata to paths."""
    if isinstance(target_config, dict):
        for section_name in ("quantization", "quantization_config"):
            section = target_config.get(section_name)
            if not isinstance(section, dict):
                continue
            for key in ("mode", "format", "quant_method", "method"):
                tag = _canonical_precision_tag(section.get(key))
                if tag is not None:
                    return tag

        # An explicit floating dtype plus no quantization metadata is a useful
        # signal for base checkpoints. Do not infer INT4 from `bits: 4` alone:
        # oQ4, NVFP4, and several other formats share that width.
        if not target_config.get("quantization") and not target_config.get(
            "quantization_config"
        ):
            tag = _canonical_precision_tag(target_config.get("torch_dtype"))
            if tag is not None:
                return tag

    # Local model directories normally retain their Hub basename, but this is
    # deliberately only a fallback because users can rename them.
    return _canonical_precision_tag(Path(str(target_name).rstrip("/")).name)


def check_draft_target_precision_pairing(
    target_name: str,
    target_config: dict | None,
    draft_path: str,
) -> str | None:
    """Heuristic precision-pairing check between a DFlash draft and its target.

    Some Poolside drafts declare a target precision in their suffix, for
    example ``*-DFlash-NVFP4``. Draft checkpoints do not otherwise record the
    target they were trained against, so only warn when both sides make an
    explicit, contradictory precision claim. A generic ``*-DFlash`` name is
    intentionally treated as unknown rather than assumed to be BF16-only.

    Returns a warning message, or None when the pair looks matched or either
    side carries no reliable precision claim.
    """
    draft_base = Path(str(draft_path).rstrip("/")).name
    match = re.search(r"-DFlash(?:-([A-Za-z0-9]+))?$", draft_base, re.IGNORECASE)
    if match is None:
        return None
    suffix = (match.group(1) or "").upper()
    draft_tag = suffix if suffix in _DRAFT_PRECISION_TAGS else ""

    if not draft_tag:
        return None

    target_tag = _target_precision_tag(target_name, target_config)
    if target_tag is None or target_tag == draft_tag:
        return None

    target_base = Path(str(target_name).rstrip("/")).name
    return (
        f"Possible DFlash precision mismatch: draft '{draft_base}' declares "
        f"{draft_tag}, while target '{target_base}' appears to be {target_tag}. "
        "This may reduce acceptance and make speculative decoding slower. "
        "Use a target/draft pair recommended by the checkpoint's model card."
    )


class _DFlashPrefillGuard:
    """Prefill-memory guard target for DFlash's primary (speculative) path,
    which bypasses the Scheduler entirely.

    Holds a ``MemoryMonitor`` (built from the target model's dims) plus the two
    watermark attrs the ``ProcessMemoryEnforcer`` writes on schedulers each tick
    (``_prefill_memory_guard``, ``_memory_hard_limit_bytes``). The enforcer
    resolves this object via ``_resolve_scheduler`` so DFlash receives the same
    ceiling as scheduler-driven engines, and ``preflight_or_raise`` delegates to
    the shared ``raise_if_prefill_exceeds`` so the estimate + HTTP-400 mapping
    match ``Scheduler.preflight_or_raise`` exactly.
    """

    def __init__(self, memory_monitor: MemoryMonitor, prefill_step_size: int):
        self.memory_monitor = memory_monitor
        self._prefill_step_size = prefill_step_size
        self._last_mlx_active_memory_bytes: int = 0
        # Written by ProcessMemoryEnforcer._propagate_memory_limit each tick.
        self._prefill_memory_guard: bool = False
        self._memory_hard_limit_bytes: int = 0
        self._memory_hot_cache_used_bytes: int = 0
        # Component breakdown behind the ceiling above, so a rejection can
        # name the binding constraint instead of generic tier advice.
        self._memory_static_ceiling_bytes: int = 0
        self._memory_dynamic_ceiling_bytes: int = 0
        self._memory_metal_cap_bytes: int = 0
        self._memory_guard_tier: str = ""

    def record_mlx_active_memory(self, active_bytes: int) -> None:
        self._last_mlx_active_memory_bytes = max(0, int(active_bytes))

    def _current_usage_bytes(self) -> int:
        # The enforcer already subtracts a hot-cache reservation from the
        # ``_memory_hard_limit_bytes`` it propagates here, so serialized
        # hot-cache CPU bytes must not ALSO be counted in usage via
        # phys_footprint — that charges them twice and over-rejects by the
        # hot-cache size. Mirrors ``Scheduler._current_usage_bytes``.
        phys = max(
            0, get_phys_footprint() - max(0, int(self._memory_hot_cache_used_bytes))
        )
        return max(self._last_mlx_active_memory_bytes, phys)

    def preflight_or_raise(
        self,
        *,
        num_prompt_tokens: int,
        request_id: str | None = None,
    ) -> None:
        # Deliberately no cached_tokens parameter: a DFlash prefix-cache hit
        # reconstructs the matched KV into active memory, so the full prompt
        # must always be charged (see DFlashEngine.preflight_chat).
        raise_if_prefill_exceeds(
            self.memory_monitor,
            prefill_memory_guard=self._prefill_memory_guard,
            hard_limit_bytes=self._memory_hard_limit_bytes,
            current_usage_bytes=self._current_usage_bytes(),
            prefill_step_size=self._prefill_step_size,
            num_prompt_tokens=num_prompt_tokens,
            request_id=request_id,
            static_ceiling_bytes=self._memory_static_ceiling_bytes,
            dynamic_ceiling_bytes=self._memory_dynamic_ceiling_bytes,
            metal_cap_bytes=self._memory_metal_cap_bytes,
            memory_guard_tier=self._memory_guard_tier,
        )


class DFlashEngine(ActivityTrackingMixin, BaseEngine):
    """
    DFlash speculative decoding engine with optional batched fallback.

    For prompts within ``model_settings.dflash_max_ctx`` (or always, when the
    threshold is None), uses block diffusion speculative decoding for 3-4x
    faster generation. When the threshold is exceeded, evicts dflash models
    from memory and delegates to a fallback engine (BatchedEngine or
    VLMBatchedEngine) that provides paged cache, SSD cache, and continuous
    batching.
    """

    def __init__(
        self,
        model_name: str,
        draft_model_path: str,
        draft_quant_enabled: bool | None = None,
        draft_quant_weight_bits: int | None = None,
        draft_quant_activation_bits: int | None = None,
        draft_quant_group_size: int | None = None,
        model_settings: Any | None = None,
        fallback_engine_type: str = "batched",
        scheduler_config: Any | None = None,
        omlx_ssd_cache_dir: str | Path | None = None,
    ):
        super().__init__()
        self._model_name = model_name
        self._draft_model_path = draft_model_path
        self._draft_quant_enabled = draft_quant_enabled
        self._draft_quant_weight_bits = draft_quant_weight_bits
        self._draft_quant_activation_bits = draft_quant_activation_bits
        self._draft_quant_group_size = draft_quant_group_size
        self._model_settings = model_settings
        self._fallback_engine_type = fallback_engine_type
        # Snapshot the scheduler config now: the engine pool mutates the
        # shared instance (model_name/model_path) on every model load, and
        # the fallback engine may start long after this engine was built.
        self._scheduler_config = (
            copy.copy(scheduler_config) if scheduler_config else scheduler_config
        )
        self._omlx_ssd_cache_dir = (
            Path(omlx_ssd_cache_dir) if omlx_ssd_cache_dir else None
        )

        self._target_model = None
        self._target_ops = None
        self._draft_model = None
        self._draft_backend = None
        self._tokenizer_obj = None
        self._executor_tokenizer = None
        self._loaded = False
        self._active_request = False
        self._model_type_str = None
        self._fallback_engine: BaseEngine | None = None
        self._in_fallback_mode = False
        self._fallback_lock = asyncio.Lock()
        # Primary-mode prefill memory guard. DFlash bypasses the scheduler, so
        # it can't receive the enforcer's watermarks through one; this holder
        # stands in (built in start(), resolved by the enforcer).
        self._prefill_guard: _DFlashPrefillGuard | None = None
        self._runtime_context: Any | None = None
        self._dflash_prefix_cache: Any | None = None
        self._suppress_token_ids: set[int] = set()
        # Protocol-specific output parser factory (gemma4 / harmony).
        # Detected once in start() after the target model is loaded; None means
        # the streaming detokenizer is used as-is (qwen, llama, etc.).
        self._output_parser_factory: Any | None = None
        # Session-scope hit/eviction rates for the admin cache observability
        # panel, fed from the dflash-mlx runtime cache manager counters.
        self._cache_rate_tracker = CacheRateTracker()
        # Draft/target precision pairing warning (set in start(), surfaced in
        # the dashboard) and per-session speculation counters fed from each
        # request's SummaryEvent.
        self._pairing_warning: str | None = None
        self._spec_stats_lock = threading.Lock()
        self._spec_last: dict[str, Any] | None = None
        self._spec_totals = {
            "requests": 0,
            "speculative_requests": 0,
            "fallback_requests": 0,
            "generation_tokens": 0,
            "accepted_draft_tokens": 0,
            "cycles": 0,
        }

        self._max_dflash_ctx = (
            getattr(model_settings, "dflash_max_ctx", None) if model_settings else None
        )
        self._in_memory_cache_enabled = (
            bool(getattr(model_settings, "dflash_in_memory_cache", True))
            if model_settings
            else True
        )
        self._in_memory_cache_max_entries = int(
            getattr(model_settings, "dflash_in_memory_cache_max_entries", 4)
            if model_settings
            else 4
        )
        self._in_memory_cache_max_bytes = int(
            getattr(model_settings, "dflash_in_memory_cache_max_bytes", 8 * 1024**3)
            if model_settings
            else 8 * 1024**3
        )
        self._ssd_cache_requested = (
            bool(getattr(model_settings, "dflash_ssd_cache", False))
            if model_settings
            else False
        )
        self._ssd_cache_max_bytes = int(
            getattr(model_settings, "dflash_ssd_cache_max_bytes", 20 * 1024**3)
            if model_settings
            else 20 * 1024**3
        )
        self._draft_window_size = (
            getattr(model_settings, "dflash_draft_window_size", None)
            if model_settings
            else None
        )
        draft_sink_size = (
            getattr(model_settings, "dflash_draft_sink_size", None)
            if model_settings
            else None
        )
        self._draft_sink_size = 0 if draft_sink_size is None else draft_sink_size
        self._block_size = (
            getattr(model_settings, "dflash_block_size", None)
            if model_settings
            else None
        )
        self._verify_mode = (
            getattr(model_settings, "dflash_verify_mode", None)
            if model_settings
            else None
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer_obj

    @property
    def model_type(self) -> str | None:
        return self._model_type_str

    @staticmethod
    def _build_quant_spec(
        weight_bits: int | None,
        activation_bits: int | None,
        group_size: int | None,
    ) -> str:
        """Convert draft quantization config into dflash 0.1.5's spec string format.

        None values fall back to dflash defaults (w4a16:gs64), so the spec stays
        valid when a profile or external API sets `enabled=True` without filling
        in every bit value.
        """
        wb = weight_bits if weight_bits is not None else 4
        ab = activation_bits if activation_bits is not None else 16
        gs = group_size if group_size is not None else 64
        return f"w{wb}a{ab}:gs{gs}"

    def _resolve_dflash_l2_dir(self, quiet: bool = False) -> Path | None:
        """Compute the dflash L2 cache directory under the omlx SSD cache root.

        ``quiet`` suppresses the misconfiguration warnings for callers that
        poll (the admin stats path); the load-time callers keep them.
        """
        if not self._ssd_cache_requested:
            return None
        if self._omlx_ssd_cache_dir is None:
            if not quiet:
                logger.warning(
                    "DFlash SSD cache requested but omlx paged SSD cache directory "
                    "is not configured; disabling L2."
                )
            return None
        if not self._in_memory_cache_enabled:
            if not quiet:
                logger.warning(
                    "DFlash SSD cache requires in-memory cache; disabling L2."
                )
            return None
        return self._omlx_ssd_cache_dir / "dflash_l2"

    def _build_runtime_context(self) -> Any:
        from dflash_mlx.runtime.config import runtime_config_from_defaults
        from dflash_mlx.runtime.context import build_runtime_context

        l2_dir = self._resolve_dflash_l2_dir()
        l2_enabled = l2_dir is not None
        cfg = runtime_config_from_defaults(
            prefix_cache=self._in_memory_cache_enabled,
            prefix_cache_max_entries=self._in_memory_cache_max_entries,
            prefix_cache_max_bytes=self._in_memory_cache_max_bytes,
            prefix_cache_l2=l2_enabled,
            prefix_cache_l2_dir=str(l2_dir) if l2_dir else "",
            # Per-model L2 disk budget. dflash-mlx's _evict_to_budget drops the
            # oldest snapshots once dflash_l2/ exceeds this, so the directory
            # stays bounded instead of filling the disk (issue #1326).
            prefix_cache_l2_max_bytes=self._ssd_cache_max_bytes if l2_enabled else 0,
            # None → dflash-mlx fills in DEFAULT_RUNTIME_CONFIG values.
            draft_window_size=self._draft_window_size,
            draft_sink_size=self._draft_sink_size,
            verify_mode=self._verify_mode,
        )
        return build_runtime_context(cfg)

    @staticmethod
    def _checkpoint_draft_window_size(draft_meta: Any) -> int | None:
        """Return a positive ``config.sliding_window`` from draft metadata."""
        if not isinstance(draft_meta, dict):
            return None
        config = draft_meta.get("config")
        value = (
            config.get("sliding_window")
            if isinstance(config, dict)
            else getattr(config, "sliding_window", None)
        )
        if value is None or isinstance(value, bool):
            return None
        try:
            window_size = int(value)
        except (TypeError, ValueError):
            return None
        return window_size if window_size > 0 else None

    def _resolve_draft_window_size(self, draft_meta: Any) -> int | None:
        """Keep an explicit setting; otherwise use the checkpoint config."""
        if self._draft_window_size is not None:
            return self._draft_window_size
        return self._checkpoint_draft_window_size(draft_meta)

    async def start(self) -> None:
        if self._loaded:
            return

        from ..engine_core import get_mlx_executor

        loop = asyncio.get_running_loop()
        runtime_context = self._build_runtime_context()

        def _load_models():
            from dflash_mlx.draft_backend import EagerDraftBackend
            from dflash_mlx.engine.target_ops import bind_draft_to_target
            from dflash_mlx.runtime.loading import (
                load_draft_bundle,
                load_target_bundle,
            )

            # Apply the same pre-load patches BatchedEngine uses before
            # mlx_lm.load() runs. MTP-bearing targets (e.g. Qwen3.6 *-mtp)
            # need the MTP-compat sanitize patch or stock mlx-lm double-shifts
            # the already-converted norm and emits garbage. dflash and mtp are
            # mutually exclusive per model_settings, so this never attaches an
            # MTP head; it only fixes sanitize. See issue #1318.
            maybe_apply_pre_load_patches(
                self._model_name, model_settings=self._model_settings
            )

            # dflash-mlx 0.1.10 has no Laguna backend. Register oMLX's strict
            # TargetOps plus the official gated Laguna drafter specialization
            # before load_target_bundle resolves either architecture.
            from ..patches.dflash_laguna import install_dflash_laguna_backend

            install_dflash_laguna_backend()

            # Wrap dflash's hook installers so we can revert the class-level
            # __call__ patches when this engine stops. Without this, a later
            # Native MTP load on the same process would see leftover dflash
            # hooks and crash with TypeError on n_confirmed (issue #1388).
            # Idempotent — only wraps once per process.
            from ..patches.dflash_lifecycle import install_dflash_lifecycle_wrap

            install_dflash_lifecycle_wrap()

            target_bundle = load_target_bundle(
                self._model_name,
                quantize_kv_cache=bool(
                    getattr(runtime_context.runtime, "quantize_kv_cache", False)
                ),
                verify_config=getattr(runtime_context, "verify", None),
            )

            # Keep DFlash targets on the same post-load MoE fast path as the
            # regular batched engine.  Laguna uses mlx-lm's SwitchGLU for its
            # routed experts, so fusing gate+up removes one gather_qmm launch
            # per MoE layer in both target prefill and block verification.
            if (
                getattr(
                    self._model_settings,
                    "moe_gate_up_fusion_enabled",
                    True,
                )
                is not False
            ):
                try:
                    from ..patches.qwen35_moe_gate_up import (
                        apply_qwen35_moe_gate_up_fusion,
                    )

                    apply_qwen35_moe_gate_up_fusion(target_bundle.model)
                except Exception:
                    logger.debug(
                        "DFlash target MoE gate+up fusion not applied",
                        exc_info=True,
                    )
            draft, draft_meta = load_draft_bundle(
                self._draft_model_path,
                draft_quant=(
                    self._build_quant_spec(
                        self._draft_quant_weight_bits,
                        self._draft_quant_activation_bits,
                        self._draft_quant_group_size,
                    )
                    if self._draft_quant_enabled
                    else None
                ),
            )
            bind_draft_to_target(
                draft,
                target_bundle.model,
                target_ops=target_bundle.target_ops,
            )
            draft_backend = EagerDraftBackend()
            return target_bundle, draft, draft_backend, draft_meta

        result = await loop.run_in_executor(get_mlx_executor(), _load_models)
        target_bundle, self._draft_model, self._draft_backend, draft_meta = result
        self._draft_window_size = self._resolve_draft_window_size(draft_meta)
        runtime_context = self._build_runtime_context()
        self._runtime_context = runtime_context
        self._target_model = target_bundle.model
        self._tokenizer_obj = target_bundle.tokenizer
        self._target_ops = target_bundle.target_ops
        target_meta = target_bundle.meta

        # Deep-copy tokenizer for executor-thread usage (dflash generation).
        # The original self._tokenizer_obj stays for event-loop operations
        # (encode, apply_chat_template, count_chat_tokens).
        # See: https://github.com/huggingface/tokenizers/issues/537
        self._executor_tokenizer = copy.deepcopy(self._tokenizer_obj)

        # Extract model_type from config
        config = target_meta.get("config", {})
        if isinstance(config, dict):
            self._model_type_str = config.get("model_type")
        elif hasattr(config, "model_type"):
            self._model_type_str = config.model_type

        self._pairing_warning = check_draft_target_precision_pairing(
            self._model_name,
            config if isinstance(config, dict) else None,
            self._draft_model_path,
        )
        if self._pairing_warning:
            logger.warning(self._pairing_warning)

        suppress_ref = (
            getattr(self._executor_tokenizer, "name_or_path", None) or self._model_name
        )
        suppress_ids = load_generation_config_token_ids(suppress_ref, "suppress_tokens")
        self._suppress_token_ids = suppress_ids or set()
        if self._suppress_token_ids:
            logger.info(
                "DFlash loaded %d suppress token(s) from generation_config.json: %s",
                len(self._suppress_token_ids),
                self._suppress_token_ids,
            )

        # Detect protocol-specific output parser (gemma4 channel markers,
        # harmony channels). Scheduler-driven engines apply this via
        # OutputParserSession.process_token per request; dflash bypasses the
        # scheduler so we do the same wiring inline in our two generate paths.
        parser_config = config if isinstance(config, dict) else None
        try:
            self._output_parser_factory = detect_output_parser(
                self._model_name, self._executor_tokenizer, parser_config
            )
        except Exception as exc:
            logger.debug(f"output parser detect failed: {exc}")
            self._output_parser_factory = None

        # Build the primary-mode prefill memory guard from the target model's
        # dims (same estimation Scheduler uses). Best-effort: on failure the
        # guard stays None and preflight degrades to a no-op — the request runs
        # unguarded rather than being falsely rejected.
        self._prefill_guard = None
        if self._target_model is not None:
            try:
                monitor = MemoryMonitor(
                    max_kv_cache_memory=None, eviction_enabled=False
                )
                set_model_info_from_model(monitor, self._target_model)
                step = (
                    getattr(self._scheduler_config, "prefill_step_size", 2048) or 2048
                )
                self._prefill_guard = _DFlashPrefillGuard(monitor, step)
            except Exception as exc:
                # Warn (not debug): a missing guard silently disables the
                # prefill OOM protection this engine relies on.
                logger.warning(f"DFlash prefill guard init failed: {exc}")
                self._prefill_guard = None

        self._loaded = True
        self._in_fallback_mode = False
        max_ctx_display = (
            "unlimited" if self._max_dflash_ctx is None else self._max_dflash_ctx
        )
        # Resolved values dflash-mlx actually ended up using (None settings → dflash default).
        runtime_cfg = getattr(self._runtime_context, "runtime", None)
        window_used = getattr(runtime_cfg, "draft_window_size", "?")
        sink_used = getattr(runtime_cfg, "draft_sink_size", "?")
        verify_used = getattr(runtime_cfg, "verify_mode", "?")
        logger.info(
            f"DFlashEngine loaded: target={self._model_name}, "
            f"draft={self._draft_model_path}, "
            f"max_ctx={max_ctx_display}, "
            f"fallback={self._fallback_engine_type}, "
            f"l1_cache={self._in_memory_cache_enabled}, "
            f"l2_cache={self._resolve_dflash_l2_dir() is not None}, "
            f"draft_window={window_used}, draft_sink={sink_used}, verify={verify_used}"
        )

    def _record_prefill_guard_active_memory(self) -> None:
        guard = self._prefill_guard
        if guard is None:
            return
        try:
            guard.record_mlx_active_memory(mx.get_active_memory())
        except Exception as exc:
            logger.debug(f"DFlash active-memory sample failed: {exc}")

    @staticmethod
    def _begin_runtime_cache_request() -> Any | None:
        """Mark a DFlash cache request boundary when supported by dflash-mlx."""
        try:
            from dflash_mlx.cache.manager import current_runtime_cache_manager
        except ImportError:
            return None
        try:
            manager = current_runtime_cache_manager()
        except Exception as exc:
            logger.debug(f"current_runtime_cache_manager failed: {exc}")
            return None
        if manager is None:
            return None
        begin = getattr(manager, "begin_request", None)
        if not callable(begin):
            return None
        try:
            begin()
        except Exception as exc:
            logger.debug(f"dflash cache begin_request failed: {exc}")
            return None
        return manager

    @staticmethod
    def _end_runtime_cache_request(manager: Any | None) -> None:
        if manager is None:
            return
        end = getattr(manager, "end_request", None)
        if not callable(end):
            return
        try:
            end()
        except Exception as exc:
            logger.debug(f"dflash cache end_request failed: {exc}")

    async def _evict_dflash_and_start_fallback(self) -> None:
        """Evict dflash models from memory, verify release, then start fallback engine."""
        from dflash_mlx.cache.manager import shutdown_runtime_cache_manager

        from ..engine_core import get_mlx_executor

        loop = asyncio.get_running_loop()
        pre_active = mx.get_active_memory()

        # Release dflash model and cache references
        shutdown_runtime_cache_manager()
        self._dflash_prefix_cache = None
        self._runtime_context = None
        self._target_model = None
        self._target_ops = None
        self._draft_model = None
        self._draft_backend = None
        self._executor_tokenizer = None
        self._output_parser_factory = None
        # Deliberately keep self._prefill_guard alive across the transition.
        # Its MemoryMonitor holds only dims (no model ref), so it stays valid
        # and cheap. Nulling it here would open a window — guard gone, fallback
        # scheduler not yet started — where _resolve_scheduler returns None
        # (spurious "scheduler unreachable" warning + an unguarded admission
        # gap). The enforcer prefers the fallback scheduler once it is up, and
        # fallback-mode preflight delegates to the fallback engine, so the
        # stale guard is simply never consulted. Cleared in stop().
        # The fallback engine (BatchedEngine / VLMBatchedEngine) starts next.
        # Revert dflash's class patches now so the fallback's model loads
        # onto clean linear_attn / self_attn classes (issue #1388).
        try:
            from ..patches.dflash_lifecycle import restore_dflash_class_patches

            restore_dflash_class_patches()
        except Exception as exc:
            logger.debug(f"restore_dflash_class_patches (evict): {exc}")

        # Force memory reclaim with settle barrier
        gc.collect()
        await loop.run_in_executor(
            get_mlx_executor(),
            lambda: (mx.synchronize(), mx.clear_cache()),
        )

        # Poll for actual memory release (same pattern as engine_pool._unload_engine)
        for settle_round in range(10):
            active_now = mx.get_active_memory()
            freed = pre_active - active_now
            if freed > 0:
                logger.info(
                    f"DFlash models evicted: freed={freed / 1024**3:.2f}GB "
                    f"(round {settle_round + 1})"
                )
                break
            await asyncio.sleep(0.5)
            gc.collect()
            await loop.run_in_executor(
                get_mlx_executor(),
                lambda: (mx.synchronize(), mx.clear_cache()),
            )
        else:
            logger.warning("DFlash model eviction: memory settle timed out")

        # Start fallback engine
        if self._fallback_engine_type == "vlm":
            from .vlm import VLMBatchedEngine

            self._fallback_engine = VLMBatchedEngine(
                model_name=self._model_name,
                scheduler_config=self._scheduler_config,
                model_settings=self._model_settings,
            )
        else:
            from .batched import BatchedEngine

            self._fallback_engine = BatchedEngine(
                model_name=self._model_name,
                scheduler_config=self._scheduler_config,
                model_settings=self._model_settings,
            )
        await self._fallback_engine.start()
        self._in_fallback_mode = True
        logger.info(f"DFlash fallback engine started: {self._fallback_engine_type}")

    async def stop(self) -> None:
        from dflash_mlx.cache.manager import shutdown_runtime_cache_manager

        if self._fallback_engine is not None:
            await self._fallback_engine.stop()
            self._fallback_engine = None
        try:
            shutdown_runtime_cache_manager()
        except Exception as exc:
            logger.debug(f"shutdown_runtime_cache_manager: {exc}")
        self._dflash_prefix_cache = None
        self._runtime_context = None
        self._target_model = None
        self._target_ops = None
        self._draft_model = None
        self._draft_backend = None
        self._tokenizer_obj = None
        self._executor_tokenizer = None
        self._output_parser_factory = None
        self._prefill_guard = None
        self._in_fallback_mode = False
        self._loaded = False
        # Revert class-level __call__ patches dflash installed during start().
        # Required so a subsequent Native MTP load on the same process sees
        # clean classes instead of leftover dflash hooks (issue #1388).
        try:
            from ..patches.dflash_lifecycle import restore_dflash_class_patches

            restore_dflash_class_patches()
        except Exception as exc:
            logger.debug(f"restore_dflash_class_patches: {exc}")
        logger.info("DFlashEngine stopped")

    def _apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        is_partial: bool | None = None,
    ) -> str:
        """Apply chat template to messages.

        Args:
            messages: List of chat messages
            tools: Optional tool definitions
            chat_template_kwargs: Optional kwargs for the chat template
                (e.g. enable_thinking, reasoning_effort).
            is_partial: Explicit partial-mode signal from the API server.
                ``True``/``False`` — server has already decided; the ``partial``
                key is cleaned from message dicts but no detection is performed.
                ``None`` (default) — auto-detect from messages for backward
                compatibility with direct engine callers.
        """
        if hasattr(self._tokenizer_obj, "apply_chat_template"):
            if is_partial is None:
                is_partial = detect_and_strip_partial(messages)
            else:
                # Server already resolved partial; just clean residual keys
                # so the chat template never sees the non-standard field.
                for msg in messages:
                    msg.pop("partial", None)
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": not is_partial,
            }
            if is_partial:
                template_kwargs["continue_final_message"] = True
            if tools:
                template_kwargs["tools"] = tools
            if chat_template_kwargs:
                template_kwargs.update(chat_template_kwargs)
            try:
                return apply_chat_template_with_reasoning_effort_fallback(
                    self._tokenizer_obj,
                    messages,
                    template_kwargs,
                    is_harmony=self.model_type == "gpt_oss",
                )
            except TypeError:
                if chat_template_kwargs:
                    for key in chat_template_kwargs:
                        template_kwargs.pop(key, None)
                template_kwargs.pop("tools", None)
                template_kwargs.pop("enable_thinking", None)
                return self._tokenizer_obj.apply_chat_template(
                    messages, **template_kwargs
                )
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            return prompt + "\nassistant:"

    def count_chat_tokens(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        is_partial: bool | None = None,
    ) -> int:
        """Count prompt tokens for chat messages after applying chat template.

        Args:
            messages: List of chat messages
            tools: Optional tool definitions
            chat_template_kwargs: Optional kwargs for chat template
            is_partial: Explicit partial-mode signal (see _apply_chat_template).

        Returns:
            Number of prompt tokens
        """
        template_tools = convert_tools_for_template(tools) if tools else None
        prompt = self._apply_chat_template(
            messages,
            template_tools,
            chat_template_kwargs=chat_template_kwargs,
            is_partial=is_partial,
        )
        return len(self._tokenizer_obj.encode(prompt))

    async def preflight_chat(
        self,
        messages: list,
        tools: list | None = None,
        request_id: str | None = None,
        **kwargs,
    ) -> None:
        """Prefill-memory preflight for chat requests.

        DFlash bypasses the scheduler, so it implements the front-door guard
        itself (BaseEngine's no-op would leave primary-mode prefills
        unprotected). In fallback mode it delegates to the fallback engine,
        whose scheduler runs the full guard. Raises ``PrefillMemoryExceededError``
        (→ HTTP 400) when the prompt's prefill peak would exceed the ceiling.
        Mirrors ``BatchedEngine.preflight_chat``.
        """
        if not self._loaded:
            await self.start()
        if self._in_fallback_mode and self._fallback_engine is not None:
            await self._fallback_engine.preflight_chat(
                messages, tools=tools, request_id=request_id, **kwargs
            )
            return
        if self._prefill_guard is None:
            _warn_scheduler_unreachable_once(
                self, "preflight_chat", "primary-mode prefill guard unavailable"
            )
            return
        try:
            num_tokens = self.count_chat_tokens(
                messages,
                tools,
                chat_template_kwargs=kwargs.get("chat_template_kwargs"),
                is_partial=kwargs.get("is_partial"),
            )
        except Exception as e:
            logger.warning(
                "DFlashEngine.preflight_chat: token count raised %s; skipping "
                "prefill memory check, real chat path will surface the error",
                type(e).__name__,
            )
            return
        # Deliberately no cached_tokens: a DFlash prefix-cache hit
        # *reconstructs* the matched KV into active memory (dflash_mlx
        # ``hydrate_target_cache`` clones every array), so the full prompt's
        # KV is allocated this request — unlike the scheduler's resident
        # paged cache. Subtracting hit tokens here would under-count and
        # defeat the OOM guard.
        self._prefill_guard.preflight_or_raise(
            num_prompt_tokens=num_tokens, request_id=request_id
        )

    async def preflight_completion(
        self,
        prompt: str,
        request_id: str | None = None,
        **kwargs,
    ) -> None:
        """Prefill-memory preflight for plain completions. See ``preflight_chat``."""
        if not self._loaded:
            await self.start()
        if self._in_fallback_mode and self._fallback_engine is not None:
            await self._fallback_engine.preflight_completion(
                prompt, request_id=request_id, **kwargs
            )
            return
        if self._prefill_guard is None:
            _warn_scheduler_unreachable_once(
                self,
                "preflight_completion",
                "primary-mode prefill guard unavailable",
            )
            return
        try:
            num_tokens = len(self._tokenizer_obj.encode(prompt))
        except Exception as e:
            logger.warning(
                "DFlashEngine.preflight_completion: tokenizer.encode raised %s; "
                "skipping prefill memory check, real completion path will "
                "surface the error",
                type(e).__name__,
            )
            return
        # Deliberately no cached_tokens — see preflight_chat: a prefix-cache
        # hit reconstructs KV into active memory, so the full prompt is
        # charged.
        self._prefill_guard.preflight_or_raise(
            num_prompt_tokens=num_tokens, request_id=request_id
        )

    @property
    def supports_multimodal_fallback(self) -> bool:
        return self._fallback_engine_type == "vlm"

    _MULTIMODAL_TYPES = frozenset({"image", "image_url", "input_image"})

    @staticmethod
    def _has_multimodal_content(messages: list[dict]) -> bool:
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if (
                        isinstance(part, dict)
                        and part.get("type") in DFlashEngine._MULTIMODAL_TYPES
                    ):
                        return True
        return False

    def _should_fallback(self, prompt_tokens: list[int]) -> bool:
        if self._max_dflash_ctx is None:
            return False
        return len(prompt_tokens) >= self._max_dflash_ctx

    def _get_think_token_id(self, attr: str) -> int | None:
        """Safely read think_start_id / think_end_id from the tokenizer."""
        try:
            return getattr(self._tokenizer_obj, attr, None)
        except (ValueError, TypeError):
            return None

    def _detect_needs_think_prefix(self, prompt_tokens: list[int]) -> bool:
        """Detect if prompt ends with an open <think> tag (thinking enabled).

        DFlash bypasses the scheduler, so the ``<think>\\n`` prefix that the
        scheduler normally prepends to the first chunk for reasoning models
        must be reproduced here. Mirrors ``Scheduler._detect_needs_think_prefix``.

        Returns False for disabled-thinking patterns like <think></think>
        where </think> immediately follows <think> in the prompt tail.
        """
        if not prompt_tokens:
            return False

        think_start_id = self._get_think_token_id("think_start_id")
        if think_start_id is None and self._tokenizer_obj is not None:
            try:
                tid = self._tokenizer_obj.convert_tokens_to_ids("<think>")
                if tid == getattr(self._tokenizer_obj, "unk_token_id", None):
                    return False
                think_start_id = tid
            except (AttributeError, KeyError, TypeError):
                return False

        if not think_start_id:
            return False

        last_tokens = list(prompt_tokens[-3:])
        if think_start_id not in last_tokens:
            return False

        last_idx = len(last_tokens) - 1 - last_tokens[::-1].index(think_start_id)
        after_start = last_tokens[last_idx + 1 :]

        if after_start:
            think_end_id = self._get_think_token_id("think_end_id")
            if think_end_id is not None and think_end_id in after_start:
                return False
            if self._tokenizer_obj is not None:
                try:
                    tid = self._tokenizer_obj.convert_tokens_to_ids("</think>")
                    unk = getattr(self._tokenizer_obj, "unk_token_id", None)
                    if tid != unk and tid in after_start:
                        return False
                except (AttributeError, KeyError, TypeError):
                    pass

        return True

    def _think_prefix_text(self) -> str:
        """Return the opening think tag string to prepend (e.g. '<think>\\n')."""
        tag = getattr(self._tokenizer_obj, "think_start", "<think>")
        return f"{tag}\n"

    def _stream_dflash_events(
        self,
        prompt_tokens: list[int],
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        repetition_context_size: int = 20,
    ):
        """Build the dflash event iterator with prefix cache plumbed in."""
        from dflash_mlx.runtime import get_stop_token_ids, stream_dflash_generate
        from dflash_mlx.server.prefix_cache_flow import PrefixCacheFlow

        stop_ids = get_stop_token_ids(self._executor_tokenizer)

        # Build a minimal model_provider shim for the prefix cache flow.
        # ``model_key`` is consumed as a tuple where index 0 = target id and
        # index 2 = draft id; the middle slot is unused on the dflash side.
        # ``tokenizer`` and ``cli_args`` are required since dflash-mlx 1ba6713 —
        # build_prefix_key hashes the chat template / policy. cli_args=None
        # makes chat_template_args fall back to {}.
        class _ModelProviderShim:
            model_key = (self._model_name, None, self._draft_model_path)
            model = self._target_model
            target_ops = self._target_ops
            tokenizer = self._executor_tokenizer
            cli_args = None

        prefix_flow = PrefixCacheFlow.for_request(
            model_provider=_ModelProviderShim(),
            draft_model=self._draft_model,
            tokenizer=self._executor_tokenizer,
            prompt=prompt_tokens,
            max_new_tokens=max_tokens,
            runtime_context=self._runtime_context,
        )

        event_iter = stream_dflash_generate(
            target_model=self._target_model,
            target_ops=self._target_ops,
            tokenizer=self._executor_tokenizer,
            draft_model=self._draft_model,
            draft_backend=self._draft_backend,
            prompt="",
            max_new_tokens=max_tokens,
            stop_token_ids=stop_ids,
            suppress_token_ids=(
                sorted(self._suppress_token_ids) if self._suppress_token_ids else None
            ),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
            block_tokens=self._block_size,
            prompt_tokens_override=prompt_tokens,
            prefix_snapshot=prefix_flow.snapshot,
            snapshot_service=prefix_flow.snapshot_service,
            stable_prefix_len=prefix_flow.stable_prefix_len,
            prefix_cache_active=prefix_flow.cache_active,
            publish_generation_snapshot=prefix_flow.publish_generation_snapshot,
            prefix_hit_kind=str(getattr(prefix_flow, "hit_kind", "miss") or "miss"),
            runtime_context=self._runtime_context,
        )
        if hasattr(prefix_flow, "snapshot"):
            prefix_flow.snapshot = None
        return event_iter, prefix_flow, stop_ids

    @staticmethod
    def _cached_tokens_from_flow(prefix_flow) -> int:
        """Prompt tokens served from the DFlash prefix snapshot (prefill skipped).

        On a hit ``PrefixCacheFlow.hit_tokens`` is the number of matched prompt
        tokens; surfacing it as ``cached_tokens`` makes DFlash report prefix-cache
        hits like the batched engine does instead of always reporting 0 (#1441).
        """
        if prefix_flow is None:
            return 0
        return max(0, int(getattr(prefix_flow, "hit_tokens", 0) or 0))

    def _create_output_parser_session(self, tools: list[dict] | None) -> Any | None:
        """Create a request-local parser, including its tool schemas."""
        factory = self._output_parser_factory
        if factory is None:
            return None
        create_with_tools = getattr(factory, "create_session_with_tools", None)
        if create_with_tools is not None:
            return create_with_tools(self._executor_tokenizer, tools)
        return factory.create_session(self._executor_tokenizer)

    def _run_generate_streaming(
        self,
        prompt_tokens: list[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        repetition_penalty: float,
        repetition_context_size: int,
        seed: int | None,
        tools: list[dict] | None,
        queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
        stop_event: threading.Event,
    ) -> None:
        """Run dflash generation with streaming on MLX executor thread.

        ``stop_event`` is set by the async consumer when it stops reading
        (client disconnect / abort). Polling it between events lets the loop
        return promptly so the single MLX executor thread is freed for the
        next request.
        """
        from dflash_mlx.engine.events import SummaryEvent, TokenEvent

        event_iter = None
        cache_manager = None
        try:
            self._record_prefill_guard_active_memory()
            if seed is not None:
                # Best-effort per-request reproducibility, matching the
                # batched engine's mx.random.seed handling.
                mx.random.seed(int(seed))
            event_iter, prefix_flow, stop_ids = self._stream_dflash_events(
                prompt_tokens=prompt_tokens,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size,
            )
            cache_manager = self._begin_runtime_cache_request()
            self._record_prefill_guard_active_memory()

            # Protocol-specific parser (gemma4 channel markers → <think> tags,
            # harmony channels → <think>/visible split). When active it owns
            # detokenization too, so the standard streaming detokenizer is
            # only created when no parser is available.
            parser_session = self._create_output_parser_session(tools)
            detokenizer = None
            if parser_session is None:
                detokenizer = create_streaming_detokenizer(
                    self._executor_tokenizer,
                    model_path=self._model_name,
                )
                if detokenizer is not None:
                    detokenizer.reset()

            for event in event_iter:
                if stop_event.is_set():
                    logger.info("DFlash generation aborted by client")
                    break

                if isinstance(event, TokenEvent):
                    token_id = int(event.token_id)
                    # Skip EOS/stop tokens from output
                    if token_id in stop_ids:
                        continue
                    if parser_session is not None:
                        result = parser_session.process_token(token_id)
                        text = result.stream_text
                    elif detokenizer is not None:
                        detokenizer.add_token(token_id)
                        text = detokenizer.last_segment
                    else:
                        text = self._executor_tokenizer.decode([token_id])
                    # Parser sessions can emit empty stream_text on protocol
                    # marker tokens — skip the chunk so clients don't see a
                    # flood of empty deltas.
                    if not text:
                        continue
                    asyncio.run_coroutine_threadsafe(
                        queue.put((text, [token_id], False, None)), loop
                    )

                elif isinstance(event, SummaryEvent):
                    self._record_speculation_summary(event)
                    # Flush any buffered tail from the parser (e.g. close an
                    # unterminated <think> block) before the metrics chunk so
                    # the client sees a well-formed final delta.
                    parser_final = None
                    if parser_session is not None:
                        parser_final = parser_session.finalize()
                        tail = parser_final.stream_text
                        if tail:
                            asyncio.run_coroutine_threadsafe(
                                queue.put((tail, [], False, None)), loop
                            )

                    gen_tokens = int(event.generation_tokens)
                    accept_ratio = float(event.acceptance_ratio)
                    cycles = int(event.cycles_completed)
                    elapsed_us = int(event.elapsed_us)
                    elapsed_s = elapsed_us / 1e6 if elapsed_us else 0
                    gen_tps = gen_tokens / elapsed_s if elapsed_s > 0 else 0
                    fallback = bool(event.fallback_ar)
                    phase_summary = _format_phase_timings(
                        getattr(event, "phase_timings_us", None)
                    )
                    logger.info(
                        f"DFlash generation complete: "
                        f"{gen_tokens} tokens, "
                        f"{gen_tps:.1f} tok/s, "
                        f"acceptance={accept_ratio:.1%}, "
                        f"cycles={cycles}"
                        f"{', fallback=AR' if fallback else ''}"
                        f"{phase_summary}"
                    )
                    metrics: dict[str, Any] = {
                        "prompt_tokens": int(event.prompt_token_count),
                        "completion_tokens": gen_tokens,
                        "acceptance_ratio": accept_ratio,
                        "cycles_completed": cycles,
                        # Prefix-snapshot hit count, surfaced on the final
                        # (usage) chunk so the API reports cached_tokens (#1441).
                        "cached_tokens": self._cached_tokens_from_flow(prefix_flow),
                    }
                    if parser_final is not None:
                        if parser_final.tool_calls:
                            metrics["tool_calls"] = parser_final.tool_calls
                        if parser_final.finish_reason:
                            metrics["finish_reason"] = parser_final.finish_reason
                    asyncio.run_coroutine_threadsafe(
                        queue.put(("", [], True, metrics)), loop
                    )

                # Cycle, memory, prefill, and snapshot events are consumed by the
                # runtime cache manager and metrics layers — omlx does not surface
                # them so all other event types are intentionally ignored.

        except Exception as e:
            logger.error(f"DFlash streaming generation error: {e}")
            asyncio.run_coroutine_threadsafe(
                queue.put(("", [], True, {"error": str(e)})), loop
            )
        finally:
            # Closing the dflash generator throws GeneratorExit on its next
            # yield, releasing kernel state and any draft cache it holds.
            self._record_prefill_guard_active_memory()
            if event_iter is not None:
                close = getattr(event_iter, "close", None)
                if close is not None:
                    try:
                        close()
                    except Exception as exc:
                        logger.debug(f"event_iter.close() raised: {exc}")
            self._end_runtime_cache_request(cache_manager)
            # Always send a sentinel so the async consumer doesn't deadlock
            # when an abort happened before the dflash summary was emitted.
            asyncio.run_coroutine_threadsafe(
                queue.put(("", [], True, {"aborted": stop_event.is_set()})),
                loop,
            )
            self._active_request = False

    def _tokenize_prompt(self, prompt: str | list[int]) -> list[int]:
        """Return prompt IDs without re-tokenizing an already-tokenized prompt."""
        if isinstance(prompt, list):
            return list(prompt)
        return list(self._tokenizer_obj.encode(prompt))

    async def generate(
        self,
        prompt: str | list[int],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        if not self._loaded:
            await self.start()

        prompt_tokens = self._tokenize_prompt(prompt)

        # Fallback: evict dflash models, start LLM/VLM engine
        if self._should_fallback(prompt_tokens):
            async with self._fallback_lock:
                if not self._in_fallback_mode:
                    logger.info(
                        f"DFlash context fallback: {len(prompt_tokens)} >= {self._max_dflash_ctx}, "
                        f"evicting dflash models and switching to {self._fallback_engine_type} engine"
                    )
                    await self._evict_dflash_and_start_fallback()
            return await self._fallback_engine.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                stop=stop,
                **kwargs,
            )

        # Already in fallback mode but short context came in.
        # Stay in fallback mode (reloading dflash models is expensive).
        if self._in_fallback_mode:
            return await self._fallback_engine.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                stop=stop,
                **kwargs,
            )

        tools = kwargs.pop("tools", None)
        seed = kwargs.pop("seed", None)
        repetition_context_size = kwargs.pop("repetition_context_size", None)
        if repetition_context_size is None:
            repetition_context_size = 20

        from ..engine_core import get_mlx_executor

        loop = asyncio.get_running_loop()
        stop_event = threading.Event()
        # Admin visibility: DFlash bypasses the scheduler, so the Active
        # Models card reads this activity instead of a scheduler snapshot.
        activity_id = self._begin_activity("generate", detail="generating")

        def _run():
            from dflash_mlx.engine.events import SummaryEvent, TokenEvent

            event_iter = None
            cache_manager = None
            # Per-request parser session (gemma4 channel markers, harmony
            # channels). Lives only inside the executor thread so the parser
            # state cannot leak across requests.
            parser_session = self._create_output_parser_session(tools)
            try:
                self._record_prefill_guard_active_memory()
                if seed is not None:
                    # Best-effort per-request reproducibility, matching the
                    # batched engine's mx.random.seed handling.
                    mx.random.seed(int(seed))
                event_iter, prefix_flow, stop_ids = self._stream_dflash_events(
                    prompt_tokens=prompt_tokens,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    repetition_penalty=repetition_penalty,
                    repetition_context_size=int(repetition_context_size),
                )
                cache_manager = self._begin_runtime_cache_request()
                self._record_prefill_guard_active_memory()
                tokens: list[int] = []
                parsed_visible_parts: list[str] = []
                summary: SummaryEvent | None = None
                first_token_at: float | None = None
                parser_final = None
                for event in event_iter:
                    if stop_event.is_set():
                        logger.info("DFlash generation aborted by client")
                        break
                    if isinstance(event, TokenEvent):
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        token_id = int(event.token_id)
                        if token_id in stop_ids:
                            continue
                        tokens.append(token_id)
                        self._update_activity(activity_id, token_count=len(tokens))
                        if parser_session is not None:
                            result = parser_session.process_token(token_id)
                            if result.visible_text:
                                parsed_visible_parts.append(result.visible_text)
                    elif isinstance(event, SummaryEvent):
                        summary = event
                        self._record_speculation_summary(event)
                if parser_session is not None:
                    parser_final = parser_session.finalize()
                    if parser_final.visible_text:
                        parsed_visible_parts.append(parser_final.visible_text)
                return (
                    summary,
                    tokens,
                    parser_session,
                    parser_final,
                    parsed_visible_parts,
                    prefix_flow,
                    first_token_at,
                )
            finally:
                self._record_prefill_guard_active_memory()
                if event_iter is not None:
                    close = getattr(event_iter, "close", None)
                    if close is not None:
                        try:
                            close()
                        except Exception as exc:
                            logger.debug(f"event_iter.close() raised: {exc}")
                self._end_runtime_cache_request(cache_manager)
                self._active_request = False

        self._active_request = True
        future = loop.run_in_executor(get_mlx_executor(), _run)
        try:
            try:
                (
                    summary,
                    generated,
                    parser_session,
                    parser_final,
                    parsed_visible_parts,
                    prefix_flow,
                    first_token_at,
                ) = await asyncio.shield(asyncio.wrap_future(future))
            except asyncio.CancelledError:
                stop_event.set()
                logger.info("DFlash generate cancelled, waiting for executor to drain")
                try:
                    await asyncio.wait_for(asyncio.wrap_future(future), timeout=10.0)
                except TimeoutError:
                    logger.warning(
                        "DFlash executor did not exit within 10s after abort"
                    )
                except Exception:
                    pass
                raise
        finally:
            self._end_activity(activity_id)

        if parser_session is not None:
            # Parser already converted protocol markers to <think>...</think>
            # and stripped channel marker tokens, so just join the visible
            # segments. Don't re-decode the raw token list — that would
            # reintroduce the raw markers and double-buffer detokenization.
            text = "".join(parsed_visible_parts)
        else:
            text = self._tokenizer_obj.decode(generated, skip_special_tokens=True)
            text = clean_special_tokens(text)

            # Reasoning models (Qwen3.x with enable_thinking, DeepSeek,
            # MiniMax, ...) have <think>\n at the END of the prompt, so the
            # model's first generated token is already INSIDE the thinking
            # block. The opening tag never appears in the output, which would
            # prevent extract_thinking / ThinkingParser from separating
            # reasoning from content. Prepend the tag here so the API layer
            # can split them correctly. Skipped when a parser session is
            # active because gemma4/harmony parsers already emit <think> tags
            # themselves and prepending would double the marker.
            if self._detect_needs_think_prefix(prompt_tokens):
                text = self._think_prefix_text() + text

        prompt_token_count = (
            int(summary.prompt_token_count)
            if summary is not None
            else len(prompt_tokens)
        )
        completion_token_count = (
            int(summary.generation_tokens) if summary is not None else len(generated)
        )
        return GenerationOutput(
            text=text,
            tokens=generated,
            prompt_tokens=prompt_token_count,
            completion_tokens=completion_token_count,
            cached_tokens=self._cached_tokens_from_flow(prefix_flow),
            finish_reason=(
                parser_final.finish_reason
                if parser_final is not None and parser_final.finish_reason
                else "stop"
            ),
            tool_calls=(
                parser_final.tool_calls
                if parser_final is not None and parser_final.tool_calls
                else None
            ),
            first_token_at=first_token_at,
        )

    async def stream_generate(
        self,
        prompt: str | list[int],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        if not self._loaded:
            await self.start()

        prompt_tokens = self._tokenize_prompt(prompt)

        # Fallback: evict dflash models, start LLM/VLM engine
        if self._should_fallback(prompt_tokens):
            async with self._fallback_lock:
                if not self._in_fallback_mode:
                    logger.info(
                        f"DFlash context fallback: {len(prompt_tokens)} >= {self._max_dflash_ctx}, "
                        f"evicting dflash models and switching to {self._fallback_engine_type} engine"
                    )
                    await self._evict_dflash_and_start_fallback()
            async for output in self._fallback_engine.stream_generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                stop=stop,
                **kwargs,
            ):
                yield output
            return

        # Already in fallback mode — stay there
        if self._in_fallback_mode:
            async for output in self._fallback_engine.stream_generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                stop=stop,
                **kwargs,
            ):
                yield output
            return

        tools = kwargs.pop("tools", None)
        seed = kwargs.pop("seed", None)
        repetition_context_size = kwargs.pop("repetition_context_size", None)
        if repetition_context_size is None:
            repetition_context_size = 20

        prompt_len = len(prompt_tokens)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        stop_event = threading.Event()

        # Reasoning models put <think>\n at the end of the prompt, so dflash
        # generates tokens already inside the thinking block. The streaming
        # ThinkingParser starts in _in_thinking=False, so without prepending
        # the opening tag on the first chunk the whole reasoning block leaks
        # into content. Mirror Scheduler._detect_needs_think_prefix here.
        # When a protocol-aware parser session is active (gemma4 / harmony),
        # the parser emits <think> tags itself, so prepending here would
        # double the opening marker — gate it on factory absence.
        needs_think_prefix = (
            self._output_parser_factory is None
            and self._detect_needs_think_prefix(prompt_tokens)
        )
        think_prefix_pending = needs_think_prefix

        from ..engine_core import get_mlx_executor

        # Admin visibility: DFlash bypasses the scheduler, so the Active
        # Models card reads this activity instead of a scheduler snapshot.
        activity_id = self._begin_activity("generate", detail="generating")
        self._active_request = True
        future = loop.run_in_executor(
            get_mlx_executor(),
            self._run_generate_streaming,
            prompt_tokens,
            max_tokens,
            temperature,
            top_p,
            top_k,
            min_p,
            repetition_penalty,
            int(repetition_context_size),
            seed,
            tools,
            queue,
            loop,
            stop_event,
        )

        total_text = ""
        total_completion = 0
        finished_normally = False

        try:
            while True:
                new_text, new_tokens, finished, metrics = await queue.get()

                if think_prefix_pending and new_text:
                    new_text = self._think_prefix_text() + new_text
                    think_prefix_pending = False

                total_text += new_text
                total_completion += len(new_tokens)
                self._update_activity(activity_id, token_count=total_completion)

                finish_reason = None
                if finished:
                    if metrics and metrics.get("completion_tokens") is not None:
                        total_completion = int(metrics["completion_tokens"])
                    if metrics and metrics.get("error"):
                        finish_reason = "error"
                    else:
                        finish_reason = (metrics or {}).get("finish_reason", "stop")
                    finished_normally = True

                yield GenerationOutput(
                    text=total_text,
                    new_text=new_text,
                    tokens=new_tokens,
                    prompt_tokens=prompt_len,
                    completion_tokens=total_completion,
                    # Carried only on the final (usage) chunk's metrics; 0 on
                    # token deltas so the server's per-chunk sum isn't inflated.
                    cached_tokens=(metrics or {}).get("cached_tokens", 0),
                    finished=finished,
                    finish_reason=finish_reason,
                    tool_calls=(metrics or {}).get("tool_calls"),
                )

                if finished:
                    break
        finally:
            # End the admin activity before the drain await below: a second
            # cancellation delivered during that await would skip anything
            # placed after it and leak a phantom active count.
            self._end_activity(activity_id)
            # Signal the executor to stop so the next request isn't blocked
            # behind a cancelled generation. Wait briefly for the dflash loop
            # to break out at its next event boundary; the timeout caps how
            # long the next request has to wait if the model is mid-cycle.
            if not finished_normally:
                stop_event.set()
                logger.info("DFlash stream cancelled, waiting for executor to drain")
            try:
                await asyncio.wait_for(asyncio.wrap_future(future), timeout=10.0)
            except TimeoutError:
                logger.warning(
                    "DFlash executor did not exit within 10s after abort; "
                    "next request may still be queued"
                )
            except Exception as exc:
                logger.debug(f"DFlash executor future raised: {exc}")

    async def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        if not self._loaded:
            await self.start()

        if self._in_fallback_mode:
            return await self._fallback_engine.chat(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                tools=tools,
                **kwargs,
            )

        if self._fallback_engine_type == "vlm" and self._has_multimodal_content(
            messages
        ):
            async with self._fallback_lock:
                if not self._in_fallback_mode:
                    logger.info(
                        "DFlash multimodal fallback: image content detected, "
                        "switching to VLM engine"
                    )
                    await self._evict_dflash_and_start_fallback()
            return await self._fallback_engine.chat(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                tools=tools,
                **kwargs,
            )

        template_tools = convert_tools_for_template(tools) if tools else None
        ct_kwargs = kwargs.pop("chat_template_kwargs", None)
        is_partial = kwargs.pop("is_partial", None)
        prompt = self._apply_chat_template(
            messages,
            template_tools,
            chat_template_kwargs=ct_kwargs,
            is_partial=is_partial,
        )

        return await self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            tools=tools,
            **kwargs,
        )

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        if not self._loaded:
            await self.start()

        if self._in_fallback_mode:
            async for output in self._fallback_engine.stream_chat(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                tools=tools,
                **kwargs,
            ):
                yield output
            return

        if self._fallback_engine_type == "vlm" and self._has_multimodal_content(
            messages
        ):
            async with self._fallback_lock:
                if not self._in_fallback_mode:
                    logger.info(
                        "DFlash multimodal fallback: image content detected, "
                        "switching to VLM engine"
                    )
                    await self._evict_dflash_and_start_fallback()
            async for output in self._fallback_engine.stream_chat(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                tools=tools,
                **kwargs,
            ):
                yield output
            return

        template_tools = convert_tools_for_template(tools) if tools else None
        ct_kwargs = kwargs.pop("chat_template_kwargs", None)
        is_partial = kwargs.pop("is_partial", None)
        prompt = self._apply_chat_template(
            messages,
            template_tools,
            chat_template_kwargs=ct_kwargs,
            is_partial=is_partial,
        )

        async for output in self.stream_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            tools=tools,
            **kwargs,
        ):
            yield output

    @property
    def scheduler(self) -> Any | None:
        fallback = self._fallback_engine
        if fallback is None:
            return None

        scheduler = getattr(fallback, "scheduler", None)
        if scheduler is not None:
            return scheduler

        inner = getattr(fallback, "_engine", None)
        if inner is None:
            return None
        inner_engine = getattr(inner, "engine", None)
        if inner_engine is None:
            return None
        return getattr(inner_engine, "scheduler", None)

    def has_active_requests(self) -> bool:
        if (
            self._fallback_engine is not None
            and self._fallback_engine.has_active_requests()
        ):
            return True
        if self._active_request:
            return True
        with self._active_lock:
            return self._active_count > 0

    def get_stats(self) -> dict[str, Any]:
        return {
            "engine_type": "dflash",
            "model_name": self._model_name,
            "draft_model": self._draft_model_path,
            "max_dflash_ctx": self._max_dflash_ctx,
            "fallback_engine_type": self._fallback_engine_type,
            "in_fallback_mode": self._in_fallback_mode,
            "loaded": self._loaded,
            "in_memory_cache": self._in_memory_cache_enabled,
            "ssd_cache": self._resolve_dflash_l2_dir() is not None,
            "pairing_warning": self._pairing_warning,
            "speculation": self.get_speculation_stats(),
        }

    @property
    def pairing_warning(self) -> str | None:
        """Draft/target precision pairing warning from load time, if any."""
        return self._pairing_warning

    def _record_speculation_summary(self, summary: Any) -> None:
        """Fold one request's SummaryEvent into the session speculation stats.

        Called from the MLX executor thread in both generate paths, hence the
        lock. Best-effort: a malformed event is dropped, never raised.
        """
        try:
            gen_tokens = int(summary.generation_tokens)
            cycles = int(summary.cycles_completed)
            ratio = float(summary.acceptance_ratio)
            accepted = int(summary.accepted_from_draft)
            tokens_per_cycle = float(summary.tokens_per_cycle)
        except (AttributeError, OverflowError, TypeError, ValueError):
            return
        fallback_ar = bool(getattr(summary, "fallback_ar", False))
        if (
            gen_tokens <= 0
            or cycles < 0
            or accepted < 0
            or accepted > gen_tokens
            or not math.isfinite(ratio)
            or not 0.0 <= ratio <= 1.0
            or not math.isfinite(tokens_per_cycle)
            or tokens_per_cycle < 0.0
        ):
            return
        last = {
            "generation_tokens": gen_tokens,
            "cycles": cycles,
            "acceptance_ratio": ratio,
            "accepted_draft_tokens": accepted,
            "tokens_per_cycle": tokens_per_cycle if cycles > 0 else None,
            "accepted_draft_tokens_per_cycle": (
                accepted / cycles if cycles > 0 else None
            ),
            "fallback_ar": fallback_ar,
            "fallback_reason": getattr(summary, "fallback_reason", None),
        }
        with self._spec_stats_lock:
            self._spec_last = last
            self._spec_totals["requests"] += 1
            if fallback_ar:
                self._spec_totals["fallback_requests"] += 1
                return
            if cycles <= 0:
                # A non-fallback summary without speculative cycles is not a
                # valid contribution to acceptance/tokens-per-cycle totals.
                return
            self._spec_totals["speculative_requests"] += 1
            self._spec_totals["generation_tokens"] += gen_tokens
            self._spec_totals["accepted_draft_tokens"] += accepted
            self._spec_totals["cycles"] += cycles

    def get_speculation_stats(self) -> dict[str, Any] | None:
        """Session speculation counters for the admin dashboard.

        The runtime's acceptance ratio is the share of output tokens supplied
        by the draft, while tokens_per_cycle includes the target-owned token.
        Expose accepted_draft_tokens_per_cycle separately so the dashboard does
        not conflate those two quantities (issue #2398).
        """
        with self._spec_stats_lock:
            if self._spec_totals["requests"] == 0:
                return None
            totals = dict(self._spec_totals)
            last = dict(self._spec_last) if self._spec_last else None
        gen_tokens = totals["generation_tokens"]
        cycles = totals["cycles"]
        totals["acceptance_ratio"] = (
            totals["accepted_draft_tokens"] / gen_tokens if gen_tokens > 0 else None
        )
        totals["tokens_per_cycle"] = gen_tokens / cycles if cycles > 0 else None
        totals["accepted_draft_tokens_per_cycle"] = (
            totals["accepted_draft_tokens"] / cycles if cycles > 0 else None
        )
        return {"last": last, "totals": totals}

    def get_cache_stats(self) -> dict[str, Any] | None:
        if self._fallback_engine is not None:
            return self._fallback_engine.get_cache_stats()
        return None

    def _scan_dflash_l2_files(self) -> tuple[int, int]:
        """Count L2 snapshot files and their total bytes on disk.

        The dflash-mlx L2 stats only track bytes written this session, so the
        directory scan is the source of truth — it also covers snapshots
        persisted by previous runs. Dot-prefixed temp files from in-flight
        writes are skipped.
        """
        l2_dir = self._resolve_dflash_l2_dir(quiet=True)
        if l2_dir is None:
            return 0, 0
        num_files = 0
        total_bytes = 0
        try:
            if not l2_dir.exists():
                return 0, 0
            for path in l2_dir.rglob("*.safetensors"):
                if path.name.startswith("."):
                    continue
                try:
                    total_bytes += path.stat().st_size
                except OSError:
                    continue
                num_files += 1
        except OSError as exc:
            logger.debug(f"DFlash L2 cache scan failed: {exc}")
        return num_files, total_bytes

    def get_runtime_cache_stats(self) -> dict[str, Any] | None:
        """Runtime cache stats for the admin observability panel.

        Returns the same shape as ``Scheduler.get_ssd_cache_stats()`` so the
        admin route can render DFlash models with the existing pipeline: the
        dflash-mlx L1 in-memory cache maps to the hot tier and the L2
        snapshot directory to the SSD tier. Block fields are omitted because
        the dflash cache is snapshot-based, not block-based.
        """
        if self._in_fallback_mode:
            # The fallback engine's scheduler owns the caches in this mode;
            # the admin route reads it through the ``scheduler`` property.
            return None
        if not self._in_memory_cache_enabled:
            return None

        manager_stats: dict[str, Any] | None = None
        try:
            from dflash_mlx.cache.manager import (
                RuntimeCacheManagerClosed,
                current_runtime_cache_manager,
            )

            manager = current_runtime_cache_manager()
            if manager is not None:
                try:
                    manager_stats = manager.stats()
                except RuntimeCacheManagerClosed:
                    manager_stats = None
        except ImportError:
            return None
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"DFlash runtime cache stats unavailable: {exc}")

        num_files, total_size_bytes = self._scan_dflash_l2_files()
        l2_enabled = self._resolve_dflash_l2_dir(quiet=True) is not None

        # The manager is created lazily on the first request; before that,
        # fall back to the configured budgets so the panel shows real caps.
        hot_max = self._in_memory_cache_max_bytes
        hot_size = 0
        hot_entries = 0
        stats: dict[str, Any] = {}
        if manager_stats is not None:
            hot_max = int(manager_stats.get("max_bytes", hot_max) or 0)
            hot_size = int(manager_stats.get("current_bytes", 0) or 0)
            hot_entries = int(manager_stats.get("current_entries", 0) or 0)
            stats["cache_rates"] = self._cache_rate_tracker.snapshot_and_get_rates(
                self._map_cache_counters(manager_stats)
            )
        stats["ssd_cache"] = {
            "num_files": num_files,
            "total_size_bytes": total_size_bytes,
            "max_size_bytes": self._ssd_cache_max_bytes if l2_enabled else 0,
            "hot_cache_max_bytes": hot_max,
            "hot_cache_size_bytes": hot_size,
            "hot_cache_entries": hot_entries,
        }
        return stats

    @staticmethod
    def _map_cache_counters(manager_stats: dict[str, Any]) -> dict[str, int]:
        """Map dflash-mlx runtime cache counters onto the scheduler's
        CacheRateTracker keys: L1 acts as the hot tier and L2 as the disk
        tier. ``exact_hits``/``prefix_hits``/``misses`` are already merged
        across both tiers by the dflash-mlx snapshot store, so the L1-only
        hit count is recovered by subtracting ``l2_hits``.
        """

        def _int(value: Any) -> int:
            try:
                return int(value or 0)
            except (TypeError, ValueError):
                return 0

        l2 = manager_stats.get("l2")
        if not isinstance(l2, dict):
            l2 = {}
        merged_hits = _int(manager_stats.get("exact_hits")) + _int(
            manager_stats.get("prefix_hits")
        )
        misses = _int(manager_stats.get("misses"))
        l1_evictions = _int(manager_stats.get("evictions"))
        l2_hits = _int(manager_stats.get("l2_hits"))
        l2_evictions = _int(l2.get("evictions"))
        return {
            "prefix_hits": merged_hits,
            "prefix_misses": misses,
            "prefix_tokens_saved": _int(manager_stats.get("prefill_tokens_saved")),
            "evictions": l1_evictions + l2_evictions,
            "ssd_hot_hits": max(0, merged_hits - l2_hits),
            "ssd_disk_loads": l2_hits,
            "ssd_saves": _int(l2.get("writes")),
            "hot_cache_evictions": l1_evictions,
            "hot_cache_promotions": l2_hits,
        }
