# SPDX-License-Identifier: Apache-2.0
"""
DFlash engine for block diffusion speculative decoding (Path A layout).

Wraps dflash-mlx (>= 0.1.5) to provide 3-4x faster decoding on Apple Silicon
for Qwen and Gemma4 model families.

Path A layout: eagerly stands up **both** an embedded ``VLMBatchedEngine``
(BG path: paged cache + SSD cache + continuous batching) **and** a DFlash
drafter attached to the same target weights (dflash path: speculative
decode). Per-request the engine routes between them based on concurrency,
KV pressure, and context length; weights are shared (not re-loaded), so
the only extra cost over plain VLM is the small drafter checkpoint.

This replaces the pre-Path-A one-way eviction layout where exceeding
``dflash_max_ctx`` permanently tore down dflash and started a fallback
engine. The ``_in_fallback_mode`` flag and ``_evict_dflash_and_start_fallback``
helper are gone; both paths coexist for the engine's lifetime.
"""

import asyncio
import copy
import gc
import json
import logging
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import mlx.core as mx

from ..api.tool_calling import convert_tools_for_template
from ..api.utils import clean_special_tokens, detect_and_strip_partial
from .base import BaseEngine, GenerationOutput

logger = logging.getLogger(__name__)


def is_dflash_compatible(model_path: str | Path) -> tuple[bool, str]:
    """Decide whether ``model_path`` can run on the current dflash backend.

    DFlash 0.1.5 registers QwenGdnTargetOps and Gemma4TargetOps. The
    top-level ``model_type`` is the canonical discriminator: Gemma4 multimodal
    configs use ``gemma4`` at the top, while MTP-only variants (e.g. the
    Gemma4 ``-assistant`` checkpoint) declare ``gemma4_assistant`` even
    though their nested ``text_config.model_type`` is still ``gemma4_text``.
    Reading top-level only keeps the gate aligned with what dflash will
    actually load.

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
    is_gemma4 = model_type in ("gemma4", "gemma4_text")
    if not (is_qwen or is_gemma4):
        return False, (
            f"DFlash supports only Qwen and Gemma4 models "
            f"(model_type='{cfg.get('model_type', '')}')"
        )
    return True, ""


class DFlashEngine(BaseEngine):
    """
    DFlash speculative decoding engine with a long-lived embedded BG engine.

    Path A layout: ``start()`` brings up both an embedded
    ``VLMBatchedEngine``/``BatchedEngine`` (paged cache, SSD cache,
    continuous batching) AND a DFlash drafter attached to the **same**
    target weights via ``DFlashVLMTargetWrapper``. Per-request, ``_route``
    decides between the dflash decode path (fast, capped concurrency,
    bounded context) and the BG path (everything else). Weights are not
    duplicated; the only extra memory cost is the small drafter.
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
        fallback_engine_type: str = "auto",
        scheduler_config: Any | None = None,
        omlx_ssd_cache_dir: str | Path | None = None,
        dflash_max_concurrent: int = 4,
        dflash_kv_pressure_threshold: float = 0.7,
        dflash_lazy_drafter: bool = False,
    ):
        self._model_name = model_name
        self._draft_model_path = draft_model_path
        self._draft_quant_enabled = draft_quant_enabled
        self._draft_quant_weight_bits = draft_quant_weight_bits
        self._draft_quant_activation_bits = draft_quant_activation_bits
        self._draft_quant_group_size = draft_quant_group_size
        self._model_settings = model_settings
        self._fallback_engine_type = self._resolve_fallback_engine_type(fallback_engine_type, model_name)
        self._scheduler_config = scheduler_config
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
        self._active_count = 0
        self._model_type_str = None
        self._target_ops: Any | None = None
        self._draft_backend: Any | None = None
        self._draft_meta: dict[str, Any] | None = None
        # Path A double-engine layout: embedded VLM stays up for the
        # engine's lifetime; the dflash bundle hooks into the same
        # already-loaded target weights.
        self._embedded_vlm: BaseEngine | None = None
        self._dflash_bundle: Any | None = None
        self._runtime_context: Any | None = None
        self._dflash_prefix_cache: Any | None = None
        # Routing counters (read by get_stats; also used by smoke test).
        self._dflash_routed_count = 0
        self._bg_routed_count = 0
        self._last_route: str | None = None

        self._max_dflash_ctx = (
            getattr(model_settings, "dflash_max_ctx", None) if model_settings else None
        )
        # Path A behavioural change: previously this defaulted to None
        # (unlimited concurrent dflash requests, only context fallback
        # ever bumped them off); Path A defaults to 1 so the BG path is
        # actually exercised under concurrency. The model_settings value
        # still wins when set explicitly.
        settings_concurrent = (
            getattr(model_settings, "dflash_max_concurrent", None) if model_settings else None
        )
        self._max_dflash_concurrent = (
            int(settings_concurrent) if settings_concurrent is not None
            else int(dflash_max_concurrent)
        )
        self._kv_pressure_threshold = float(dflash_kv_pressure_threshold)
        # Lazy drafter loading: defer wrapper + factory call until first
        # dflash-routed request. Saves ~28% throughput when workload is
        # bg-heavy (drafter co-loaded in Metal causes contention even when
        # idle, confirmed by D5 bench 2026-05-12). Cold-start cost: first
        # dflash request includes ~3s drafter load latency.
        self._dflash_lazy_drafter = bool(dflash_lazy_drafter)
        # Lazily created in start() — asyncio.Semaphore needs a running event
        # loop in some Python versions, and __init__ is sync.
        self._concurrent_sem: asyncio.Semaphore | None = None
        # Created in start() too; guards lazy-drafter race when multiple
        # concurrent requests trigger first load simultaneously.
        self._drafter_load_lock: asyncio.Lock | None = None
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
    def _resolve_fallback_engine_type(requested: str, model_name: str) -> str:
        """Resolve fallback_engine_type='auto' by inspecting model config.

        Path A is initially Gemma 4 focused (multimodal). Defaulting to
        'batched' was wrong for VLM models because mlx_lm cannot load
        Gemma 4 ConditionalGeneration architecture. Auto-detect via
        omlx.speculative.detect_fallback_engine_type (vision_config /
        audio_config markers in config.json).
        """
        if requested != "auto":
            return requested
        from ..speculative import detect_fallback_engine_type
        return detect_fallback_engine_type(model_name)

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

    def _resolve_dflash_l2_dir(self) -> Path | None:
        """Compute the dflash L2 cache directory under the omlx SSD cache root."""
        if not self._ssd_cache_requested:
            return None
        if self._omlx_ssd_cache_dir is None:
            logger.warning(
                "DFlash SSD cache requested but omlx paged SSD cache directory is "
                "not configured; disabling L2."
            )
            return None
        if not self._in_memory_cache_enabled:
            logger.warning(
                "DFlash SSD cache requires in-memory cache; disabling L2."
            )
            return None
        return self._omlx_ssd_cache_dir / "dflash_l2"

    def _build_runtime_context(self) -> Any:
        from dflash_mlx.runtime.context import (
            build_runtime_context,
            runtime_config_from_profile,
        )

        l2_dir = self._resolve_dflash_l2_dir()
        l2_enabled = l2_dir is not None
        cfg = runtime_config_from_profile(
            profile="balanced",
            prefix_cache=self._in_memory_cache_enabled,
            prefix_cache_max_entries=self._in_memory_cache_max_entries,
            prefix_cache_max_bytes=self._in_memory_cache_max_bytes,
            prefix_cache_l2=l2_enabled,
            prefix_cache_l2_dir=str(l2_dir) if l2_dir else "",
            # 1 TiB sentinel — disk usage is bounded by the omlx SSD cache
            # configuration, so dflash's own byte limit is intentionally large.
            prefix_cache_l2_max_bytes=1 << 40 if l2_enabled else 0,
        )
        return build_runtime_context(cfg)

    async def start(self) -> None:
        if self._loaded:
            return

        from ..engine_core import get_mlx_executor

        loop = asyncio.get_running_loop()

        # Build runtime context first — the dflash factory consults it for
        # verify_config and cache setup.
        self._runtime_context = self._build_runtime_context()

        # 1) Bring up the embedded BG engine. This is the canonical owner
        #    of the target weights; the dflash drafter will attach to the
        #    SAME ``_vlm_model`` instance, so memory does not double.
        if self._fallback_engine_type == "vlm":
            from .vlm import VLMBatchedEngine
            self._embedded_vlm = VLMBatchedEngine(
                model_name=self._model_name,
                scheduler_config=self._scheduler_config,
                model_settings=self._model_settings,
            )
        else:
            from .batched import BatchedEngine
            self._embedded_vlm = BatchedEngine(
                model_name=self._model_name,
                scheduler_config=self._scheduler_config,
                model_settings=self._model_settings,
            )
        await self._embedded_vlm.start()

        # Discover the loaded model + tokenizer on the embedded engine.
        # VLMBatchedEngine: ``_vlm_model`` / ``_tokenizer``.
        # BatchedEngine: ``_model`` / ``_tokenizer``.
        embedded_model = getattr(self._embedded_vlm, "_vlm_model", None) \
            or getattr(self._embedded_vlm, "_model", None)
        if embedded_model is None:
            raise RuntimeError(
                "DFlashEngine: embedded engine did not expose a loaded model "
                "after start() — neither _vlm_model nor _model is set"
            )
        self._tokenizer_obj = getattr(self._embedded_vlm, "_tokenizer", None) \
            or getattr(self._embedded_vlm, "tokenizer", None)
        # Deep-copy tokenizer for executor-thread usage (dflash generation).
        # See: https://github.com/huggingface/tokenizers/issues/537
        self._executor_tokenizer = copy.deepcopy(self._tokenizer_obj)

        # 2-3) Drafter loading. Eager path runs now; lazy path defers
        # until first dflash-routed request via _ensure_drafter_loaded.
        self._drafter_load_lock = asyncio.Lock()
        if self._dflash_lazy_drafter:
            logger.info(
                "DFlashEngine: lazy_drafter mode — drafter NOT loaded yet "
                "(loads on first dflash-routed request)"
            )
        else:
            await self._load_drafter_bundle(embedded_model)

        # Extract model_type from the embedded engine's config so the API
        # layer's reasoning detection still works.
        cfg = getattr(embedded_model, "config", None)
        if cfg is not None:
            if isinstance(cfg, dict):
                self._model_type_str = cfg.get("model_type")
            else:
                self._model_type_str = getattr(cfg, "model_type", None)

        self._loaded = True
        if self._max_dflash_concurrent:
            self._concurrent_sem = asyncio.Semaphore(self._max_dflash_concurrent)
        max_ctx_display = "unlimited" if self._max_dflash_ctx is None else self._max_dflash_ctx
        logger.info(
            f"DFlashEngine loaded (Path A double-engine): target={self._model_name}, "
            f"draft={self._draft_model_path}, "
            f"embedded_engine={self._fallback_engine_type}, "
            f"max_ctx={max_ctx_display}, "
            f"max_concurrent={self._max_dflash_concurrent}, "
            f"kv_pressure_threshold={self._kv_pressure_threshold}, "
            f"l1_cache={self._in_memory_cache_enabled}, "
            f"l2_cache={self._resolve_dflash_l2_dir() is not None}"
        )

    async def _load_drafter_bundle(self, embedded_model: Any | None = None) -> None:
        """Load the dflash drafter bundle (wrapper + factory attach).

        Used by start() in eager mode and _ensure_drafter_loaded() in lazy
        mode. Caller is responsible for serialization (start() runs once;
        lazy path holds self._drafter_load_lock).

        ``embedded_model`` is passed when called from start() (avoids a
        second getattr); lazy path resolves it from self._embedded_vlm.
        """
        from ..speculative.dflash_factory import attach_dflash_to_loaded_target
        from ..engine_core import get_mlx_executor

        if embedded_model is None:
            embedded_model = getattr(self._embedded_vlm, "_vlm_model", None) \
                or getattr(self._embedded_vlm, "_model", None)
            if embedded_model is None:
                raise RuntimeError(
                    "DFlashEngine._load_drafter_bundle: embedded engine "
                    "did not expose a loaded model"
                )

        # Path A generalization: probe upstream dflash_mlx target_ops directly
        # first. Apply the mlx_vlm→mlx_lm shape wrapper only when the upstream
        # ops can't recognize the model.
        #
        # Currently:
        #   - QwenGdnTargetOps walks `target.language_model` + uses structural
        #     hasattr checks, so it accepts both mlx_lm-loaded Qwen and
        #     mlx_vlm-loaded Qwen 3.x natively. NO wrapper.
        #   - Gemma4TargetOps reads `text_wrapper.args.layer_types` and
        #     `inner._get_per_layer_inputs` (mlx_lm-only attribute names), so
        #     mlx_vlm-loaded Gemma 4 falls through. WRAPPER required.
        #
        # The try/except below auto-routes each family without family
        # hardcoding here. When `bstnxbt/dflash-mlx` upstream generalizes
        # Gemma4TargetOps to match QwenGdnTargetOps's VLM-aware pattern,
        # the wrapper path goes idle for Gemma 4 too and we can eventually
        # delete the wrapper module.
        target_for_dflash = embedded_model
        try:
            from dflash_mlx.engine.target_ops import resolve_target_ops
            resolve_target_ops(target_for_dflash)
            logger.info(
                "DFlashEngine: upstream dflash_mlx ops resolved embedded model "
                "directly — no wrapper needed (family: %s)",
                type(target_for_dflash).__name__,
            )
        except Exception as e:  # NotImplementedError or other rejection
            from ..speculative.dflash_vlm_target_wrap import DFlashVLMTargetWrapper
            logger.info(
                "DFlashEngine: upstream ops rejected embedded model "
                "(%s: %s); applying DFlashVLMTargetWrapper for mlx_vlm→mlx_lm "
                "shape bridge",
                type(e).__name__, str(e)[:120],
            )
            target_for_dflash = DFlashVLMTargetWrapper(embedded_model)

        self._target_model = target_for_dflash
        draft_quant_spec = (
            self._build_quant_spec(
                self._draft_quant_weight_bits,
                self._draft_quant_activation_bits,
                self._draft_quant_group_size,
            )
            if self._draft_quant_enabled
            else None
        )

        def _attach_drafter() -> Any:
            return attach_dflash_to_loaded_target(
                target_model=target_for_dflash,
                draft_path=self._draft_model_path,
                draft_quant=draft_quant_spec,
                runtime_context=self._runtime_context,
            )

        loop = asyncio.get_running_loop()
        self._dflash_bundle = await loop.run_in_executor(
            get_mlx_executor(), _attach_drafter
        )
        self._draft_model = self._dflash_bundle.draft_model
        self._target_ops = self._dflash_bundle.target_ops
        self._draft_backend = self._dflash_bundle.draft_backend

    async def _ensure_drafter_loaded(self) -> None:
        """Lazy-load drafter on first dflash-routed request. Idempotent and
        concurrent-safe via self._drafter_load_lock (double-checked).

        Called from generate / stream_generate just before entering the
        DFlash decode path. No-op if drafter already loaded (eager mode or
        previous lazy invocation).
        """
        if self._dflash_bundle is not None:
            return
        # _drafter_load_lock is created in start(); if we got here without
        # start() having run, the engine is in an invalid state.
        if self._drafter_load_lock is None:
            raise RuntimeError(
                "DFlashEngine._ensure_drafter_loaded called before start()"
            )
        async with self._drafter_load_lock:
            if self._dflash_bundle is not None:  # double-check after lock
                return
            logger.info(
                "DFlashEngine: loading drafter on first dflash-routed request "
                "(lazy mode)"
            )
            await self._load_drafter_bundle()
            logger.info("DFlashEngine: drafter loaded")

    async def stop(self) -> None:
        from dflash_mlx.cache.manager import shutdown_runtime_cache_manager

        # Tear down dflash drafter side first (releases prefix-cache /
        # snapshot service / kernel state) before the embedded engine
        # disposes of the shared target weights.
        try:
            shutdown_runtime_cache_manager()
        except Exception as exc:
            logger.debug(f"shutdown_runtime_cache_manager: {exc}")
        self._dflash_prefix_cache = None
        self._runtime_context = None
        self._dflash_bundle = None
        self._draft_model = None
        self._target_ops = None
        self._draft_backend = None
        # Wrapper; underlying weights belong to embedded_vlm and get torn
        # down when the embedded engine stops below.
        self._target_model = None
        self._tokenizer_obj = None
        self._executor_tokenizer = None

        # Force a barrier so MLX releases any draft buffers before the
        # embedded engine starts its own teardown.
        gc.collect()
        try:
            from ..engine_core import get_mlx_executor
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                get_mlx_executor(),
                lambda: (mx.synchronize(), mx.clear_cache()),
            )
        except Exception as exc:
            logger.debug(f"DFlashEngine.stop barrier: {exc}")

        if self._embedded_vlm is not None:
            await self._embedded_vlm.stop()
            self._embedded_vlm = None

        self._runtime_context = None
        self._tokenizer_obj = None
        self._loaded = False
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
                return self._tokenizer_obj.apply_chat_template(
                    messages, **template_kwargs
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
            messages, template_tools,
            chat_template_kwargs=chat_template_kwargs,
            is_partial=is_partial,
        )
        return len(self._tokenizer_obj.encode(prompt))

    def _kv_pressure(self) -> float | None:
        """Read the embedded engine's paged KV cache usage ratio.

        Returns a float in [0.0, 1.0] or None if the cache isn't exposed yet
        (engine still starting, or accessor path drifted on an omlx upgrade).
        The accessor chain — ``_embedded_vlm._engine.engine.scheduler.
        paged_cache_manager.usage`` — was verified against omlx 0.x
        (cache/paged_cache.py: ``PagedCacheManager.usage`` is a @property).
        Falls back through a few common attr names just in case the upstream
        renames it, so routing still works without a hot fix.

        Wrapped in a broad except: this is best-effort telemetry feeding
        a routing heuristic; it must never break the inference path.
        """
        try:
            scheduler = self._embedded_vlm._engine.engine.scheduler  # type: ignore[union-attr]
        except (AttributeError, RuntimeError):
            return None
        # 1) Most direct: PagedCacheManager.usage (current omlx 0.x).
        mgr = getattr(scheduler, "paged_cache_manager", None)
        if mgr is not None:
            # NOTE: do NOT use mgr.usage property — it computes
            # 1 - free_block_queue.num_free_blocks/(max_blocks-1), but
            # num_free_blocks is the bounded FREE QUEUE size (capped ~256),
            # not the unallocated block count. On a near-empty cache it
            # still returns ~0.997 because the free queue size << max_blocks.
            # Correct: allocated_count / max_blocks.
            try:
                max_blocks = getattr(mgr, "max_blocks", None)
                if max_blocks and max_blocks > 0:
                    alloc_count = getattr(mgr, "_current_allocated_count", None)
                    if alloc_count is None:
                        allocated = getattr(mgr, "allocated_blocks", None) or {}
                        alloc_count = len(allocated)
                    if alloc_count is not None:
                        return float(alloc_count) / float(max_blocks)
            except (AttributeError, TypeError, ZeroDivisionError):
                pass
        return None

    def _route(self, prompt_tokens: list[int]) -> str:
        """Decide whether this request runs on dflash or the embedded BG engine.

        Path A signals (D3 layout):

          * If we're already at the dflash concurrency cap, route to BG.
          * If the embedded engine's paged KV cache usage exceeds the
            configured pressure threshold, route to BG (avoid evicting
            unrelated requests just to fit a dflash decode).
          * If the prompt is at or past ``dflash_max_ctx``, route to BG.
          * Otherwise route to dflash.

        ``_active_count`` is sampled (not held); the increment for the
        accepted request happens in the dflash decode path under
        ``_concurrent_sem``, so the cap is enforced even when several
        requests race here. Reading without locking is OK: the worst case
        is a marginal request getting routed to dflash and then blocking
        on the semaphore, which is the same behaviour the BG-route would
        produce.

        Side effect: records the decision via ``_record_route`` (counters
        + jsonl metric line). Callers should NOT call ``_record_route``
        again from ``generate`` / ``stream_generate``.
        """
        ctx_len = len(prompt_tokens)
        if self._max_dflash_concurrent is not None \
                and self._active_count >= self._max_dflash_concurrent:
            self._record_route("bg", "concurrency", ctx_len, None)
            return "bg"

        kv_pressure = self._kv_pressure()
        if kv_pressure is not None and kv_pressure > self._kv_pressure_threshold:
            self._record_route("bg", "kv_pressure", ctx_len, kv_pressure)
            return "bg"

        if self._max_dflash_ctx is not None and ctx_len >= self._max_dflash_ctx:
            self._record_route("bg", "max_ctx", ctx_len, kv_pressure)
            return "bg"

        self._record_route("dflash", "default", ctx_len, kv_pressure)
        return "dflash"

    def _record_route(
        self,
        routed_to: str,
        reason: str,
        ctx_len: int,
        kv_pressure: float | None,
    ) -> None:
        """Bookkeeping for one routing decision: counters + jsonl metric.

        Metric write is best-effort (size-capped, env-disable-able); see
        ``omlx.metrics.dflash_routing`` for the size guard and disable
        knobs. ``request_id`` is intentionally omitted — dflash.py has no
        handle on the API-layer request id; threading one through is a
        D3.x cleanup.
        """
        import time

        from ..metrics.dflash_routing import write_routing_decision

        self._last_route = routed_to
        if routed_to == "dflash":
            self._dflash_routed_count += 1
        else:
            self._bg_routed_count += 1

        write_routing_decision({
            "ts": time.time(),
            "model_name": self._model_name,
            "ctx_len": ctx_len,
            "active_count": self._active_count,
            "kv_usage_ratio": kv_pressure,
            "projected_kv_after": None,  # spec A.2 — D3.x placeholder.
            "routed_to": routed_to,
            "reason": reason,
        })

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

        think_start_id = self._get_think_token_id('think_start_id')
        if think_start_id is None and self._tokenizer_obj is not None:
            try:
                tid = self._tokenizer_obj.convert_tokens_to_ids("<think>")
                if tid == getattr(self._tokenizer_obj, 'unk_token_id', None):
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
        after_start = last_tokens[last_idx + 1:]

        if after_start:
            think_end_id = self._get_think_token_id('think_end_id')
            if think_end_id is not None and think_end_id in after_start:
                return False
            if self._tokenizer_obj is not None:
                try:
                    tid = self._tokenizer_obj.convert_tokens_to_ids("</think>")
                    unk = getattr(self._tokenizer_obj, 'unk_token_id', None)
                    if tid != unk and tid in after_start:
                        return False
                except (AttributeError, KeyError, TypeError):
                    pass

        return True

    def _think_prefix_text(self) -> str:
        """Return the opening think tag string to prepend (e.g. '<think>\\n')."""
        tag = getattr(self._tokenizer_obj, 'think_start', '<think>')
        return f"{tag}\n"

    def _stream_dflash_events(
        self,
        prompt_tokens: list[int],
        max_tokens: int,
    ):
        """Build the dflash event iterator with prefix cache plumbed in."""
        from dflash_mlx.runtime import get_stop_token_ids, stream_dflash_generate
        from dflash_mlx.server.prefix_cache_flow import PrefixCacheFlow

        stop_ids = get_stop_token_ids(self._executor_tokenizer)

        # Build a minimal model_provider shim for the prefix cache flow.
        # ``model_key`` is consumed as a tuple where index 0 = target id and
        # index 2 = draft id; the middle slot is unused on the dflash side.
        class _ModelProviderShim:
            model_key = (self._model_name, None, self._draft_model_path)

        prefix_flow = PrefixCacheFlow.for_request(
            model_provider=_ModelProviderShim(),
            draft_model=self._draft_model,
            tokenizer=self._executor_tokenizer,
            prompt=prompt_tokens,
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
            prompt_tokens_override=prompt_tokens,
            prefix_snapshot=prefix_flow.snapshot,
            snapshot_service=prefix_flow.snapshot_service,
            stable_prefix_len=prefix_flow.stable_prefix_len,
            prefix_cache_active=prefix_flow.cache_active,
            publish_generation_snapshot=prefix_flow.publish_generation_snapshot,
            runtime_context=self._runtime_context,
        )
        return event_iter, prefix_flow, stop_ids

    def _run_generate_streaming(
        self,
        prompt_tokens: list[int],
        max_tokens: int,
        temperature: float,
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
        try:
            event_iter, prefix_flow, stop_ids = self._stream_dflash_events(
                prompt_tokens=prompt_tokens,
                max_tokens=max_tokens,
            )

            # Use streaming detokenizer for proper UTF-8 handling (CJK etc.)
            detokenizer = None
            try:
                from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer
                detokenizer = NaiveStreamingDetokenizer(self._executor_tokenizer)
            except ImportError:
                pass

            from dflash_mlx.engine.events import TokenEvent, SummaryEvent

            prompt_token_count = 0
            for event in event_iter:
                if stop_event.is_set():
                    logger.info("DFlash generation aborted by client")
                    break

                if isinstance(event, TokenEvent):
                    token_id = int(event.token_id)
                    # Skip EOS/stop tokens from output
                    if token_id in stop_ids:
                        continue
                    if detokenizer is not None:
                        detokenizer.add_token(token_id)
                        text = detokenizer.last_segment
                    else:
                        text = self._executor_tokenizer.decode([token_id])
                    asyncio.run_coroutine_threadsafe(
                        queue.put((text, [token_id], False, None)), loop
                    )

                elif isinstance(event, SummaryEvent):
                    gen_tokens = int(event.generation_tokens)
                    accept_ratio = float(event.acceptance_ratio)
                    cycles = int(event.cycles_completed)
                    elapsed_us = int(event.elapsed_us)
                    elapsed_s = elapsed_us / 1e6 if elapsed_us else 0
                    gen_tps = gen_tokens / elapsed_s if elapsed_s > 0 else 0
                    fallback = bool(event.fallback_ar)
                    logger.info(
                        f"DFlash generation complete: "
                        f"{gen_tokens} tokens, "
                        f"{gen_tps:.1f} tok/s, "
                        f"acceptance={accept_ratio:.1%}, "
                        f"cycles={cycles}"
                        f"{', fallback=AR' if fallback else ''}"
                    )
                    metrics = {
                        "prompt_tokens": int(event.prompt_token_count),
                        "completion_tokens": gen_tokens,
                        "acceptance_ratio": accept_ratio,
                        "cycles_completed": cycles,
                    }
                    asyncio.run_coroutine_threadsafe(
                        queue.put(("", [], True, metrics)), loop
                    )
                # Other event types (PrefillProgressEvent, PrefillCompleteEvent,
                # SnapshotPublishedEvent, etc.) are informational — skip silently.
                # snapshot_service handles snapshot lifecycle automatically.

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
            if event_iter is not None:
                close = getattr(event_iter, "close", None)
                if close is not None:
                    try:
                        close()
                    except Exception as exc:
                        logger.debug(f"event_iter.close() raised: {exc}")
            # Always send a sentinel so the async consumer doesn't deadlock
            # when an abort happened before the dflash summary was emitted.
            asyncio.run_coroutine_threadsafe(
                queue.put(("", [], True, {"aborted": stop_event.is_set()})),
                loop,
            )
            self._active_count -= 1

    async def generate(
        self,
        prompt: str,
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

        prompt_tokens = self._tokenizer_obj.encode(prompt)

        # Path A routing: decide between dflash and the long-lived embedded
        # BG engine. Both stay loaded for the engine's lifetime, so no
        # eviction / reload is involved on either branch. ``_route`` records
        # the decision internally (counters + jsonl metric); callers must
        # NOT invoke ``_record_route`` again.
        route = self._route(prompt_tokens)
        if route == "bg":
            return await self._embedded_vlm.generate(
                prompt=prompt, max_tokens=max_tokens, temperature=temperature,
                top_p=top_p, top_k=top_k, min_p=min_p,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty, stop=stop, **kwargs,
            )

        # Lazy drafter: load now if not yet loaded (no-op in eager mode).
        await self._ensure_drafter_loaded()

        # Concurrent cap: hold at most ``dflash_max_concurrent`` requests
        # inside the DFlash decode path. ``_route`` already bounced excess
        # callers to the BG engine, but the semaphore stays as a guard
        # against in-flight races where ``_active_count`` was sampled
        # before the previous request incremented it.
        if self._concurrent_sem is not None:
            await self._concurrent_sem.acquire()

        try:
            from ..engine_core import get_mlx_executor

            loop = asyncio.get_running_loop()
            stop_event = threading.Event()

            def _run():
                event_iter = None
                try:
                    event_iter, prefix_flow, stop_ids = self._stream_dflash_events(
                        prompt_tokens=prompt_tokens,
                        max_tokens=max_tokens,
                    )
                    from dflash_mlx.engine.events import TokenEvent, SummaryEvent

                    tokens: list[int] = []
                    summary: Any = None
                    for event in event_iter:
                        if stop_event.is_set():
                            logger.info("DFlash generation aborted by client")
                            break
                        if isinstance(event, TokenEvent):
                            token_id = int(event.token_id)
                            if token_id in stop_ids:
                                continue
                            tokens.append(token_id)
                        elif isinstance(event, SummaryEvent):
                            summary = event
                        # Other events (progress, snapshots) are informational.
                    return summary, tokens
                finally:
                    if event_iter is not None:
                        close = getattr(event_iter, "close", None)
                        if close is not None:
                            try:
                                close()
                            except Exception as exc:
                                logger.debug(f"event_iter.close() raised: {exc}")
                    self._active_count -= 1

            self._active_count += 1
            future = loop.run_in_executor(get_mlx_executor(), _run)
            try:
                summary, generated = await asyncio.shield(asyncio.wrap_future(future))
            except asyncio.CancelledError:
                stop_event.set()
                logger.info("DFlash generate cancelled, waiting for executor to drain")
                try:
                    await asyncio.wait_for(asyncio.wrap_future(future), timeout=10.0)
                except asyncio.TimeoutError:
                    logger.warning("DFlash executor did not exit within 10s after abort")
                except Exception:
                    pass
                raise
            # summary is a SummaryEvent dataclass (upstream API) or None.

            text = self._tokenizer_obj.decode(generated, skip_special_tokens=True)
            text = clean_special_tokens(text)

            # Reasoning models (Qwen3.x with enable_thinking, DeepSeek, MiniMax, ...)
            # have <think>\n at the END of the prompt, so the model's first
            # generated token is already INSIDE the thinking block. The opening
            # tag never appears in the output, which would prevent extract_thinking
            # / ThinkingParser from separating reasoning from content. Prepend
            # the tag here so the API layer can split them correctly.
            if self._detect_needs_think_prefix(prompt_tokens):
                text = self._think_prefix_text() + text

            # summary is a SummaryEvent dataclass (upstream API) or None if
            # generation ended before reaching the summary event.
            prompt_tokens_count = (
                int(summary.prompt_token_count) if summary is not None else len(prompt_tokens)
            )
            completion_tokens_count = (
                int(summary.generation_tokens) if summary is not None else len(generated)
            )

            return GenerationOutput(
                text=text,
                tokens=generated,
                prompt_tokens=prompt_tokens_count,
                completion_tokens=completion_tokens_count,
                finish_reason="stop",
            )
        finally:
            if self._concurrent_sem is not None:
                self._concurrent_sem.release()

    async def stream_generate(
        self,
        prompt: str,
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

        prompt_tokens = self._tokenizer_obj.encode(prompt)

        # Path A routing: see ``generate``. Streaming mirrors the same
        # routing decision; the dflash side keeps its concurrency cap via
        # the semaphore released in the finally clause below. ``_route``
        # records the decision; do not call ``_record_route`` again here.
        route = self._route(prompt_tokens)
        if route == "bg":
            async for output in self._embedded_vlm.stream_generate(
                prompt=prompt, max_tokens=max_tokens, temperature=temperature,
                top_p=top_p, top_k=top_k, min_p=min_p,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty, stop=stop, **kwargs,
            ):
                yield output
            return

        # Lazy drafter: load now if not yet loaded (no-op in eager mode).
        await self._ensure_drafter_loaded()

        # Concurrent cap: hold at most ``dflash_max_concurrent`` requests
        # inside the DFlash streaming path. Released in the finally clause
        # below so it fires even when the async generator is cancelled
        # mid-iteration.
        if self._concurrent_sem is not None:
            await self._concurrent_sem.acquire()

        prompt_len = len(prompt_tokens)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        stop_event = threading.Event()

        # Reasoning models put <think>\n at the end of the prompt, so dflash
        # generates tokens already inside the thinking block. The streaming
        # ThinkingParser starts in _in_thinking=False, so without prepending
        # the opening tag on the first chunk the whole reasoning block leaks
        # into content. Mirror Scheduler._detect_needs_think_prefix here.
        needs_think_prefix = self._detect_needs_think_prefix(prompt_tokens)
        think_prefix_pending = needs_think_prefix

        from ..engine_core import get_mlx_executor
        self._active_count += 1
        future = loop.run_in_executor(
            get_mlx_executor(),
            self._run_generate_streaming,
            prompt_tokens,
            max_tokens,
            temperature,
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

                finish_reason = None
                if finished:
                    finish_reason = "stop"
                    if metrics and metrics.get("error"):
                        finish_reason = "error"
                    finished_normally = True

                yield GenerationOutput(
                    text=total_text,
                    new_text=new_text,
                    tokens=new_tokens,
                    prompt_tokens=prompt_len,
                    completion_tokens=total_completion,
                    finished=finished,
                    finish_reason=finish_reason,
                )

                if finished:
                    break
        finally:
            # Signal the executor to stop so the next request isn't blocked
            # behind a cancelled generation. Wait briefly for the dflash loop
            # to break out at its next event boundary; the timeout caps how
            # long the next request has to wait if the model is mid-cycle.
            if not finished_normally:
                stop_event.set()
                logger.info("DFlash stream cancelled, waiting for executor to drain")
            try:
                await asyncio.wait_for(asyncio.wrap_future(future), timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "DFlash executor did not exit within 10s after abort; "
                    "next request may still be queued"
                )
            except Exception as exc:
                logger.debug(f"DFlash executor future raised: {exc}")
            if self._concurrent_sem is not None:
                self._concurrent_sem.release()

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

        template_tools = convert_tools_for_template(tools) if tools else None
        ct_kwargs = kwargs.pop("chat_template_kwargs", None)
        is_partial = kwargs.pop("is_partial", None)
        prompt = self._apply_chat_template(
            messages, template_tools,
            chat_template_kwargs=ct_kwargs, is_partial=is_partial,
        )

        return await self.generate(
            prompt=prompt, max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty, **kwargs,
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

        template_tools = convert_tools_for_template(tools) if tools else None
        ct_kwargs = kwargs.pop("chat_template_kwargs", None)
        is_partial = kwargs.pop("is_partial", None)
        prompt = self._apply_chat_template(
            messages, template_tools,
            chat_template_kwargs=ct_kwargs, is_partial=is_partial,
        )

        async for output in self.stream_generate(
            prompt=prompt, max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty, **kwargs,
        ):
            yield output

    def has_active_requests(self) -> bool:
        if self._embedded_vlm is not None and self._embedded_vlm.has_active_requests():
            return True
        return self._active_count > 0

    def get_stats(self) -> dict[str, Any]:
        return {
            "engine_type": "dflash",
            "model_name": self._model_name,
            "draft_model": self._draft_model_path,
            "max_dflash_ctx": self._max_dflash_ctx,
            "max_dflash_concurrent": self._max_dflash_concurrent,
            "kv_pressure_threshold": self._kv_pressure_threshold,
            "active_count": self._active_count,
            "embedded_engine_type": self._fallback_engine_type,
            "last_route": self._last_route,
            "dflash_routed_count": self._dflash_routed_count,
            "bg_routed_count": self._bg_routed_count,
            "concurrent_sem_locked": (
                self._concurrent_sem.locked() if self._concurrent_sem is not None else False
            ),
            "loaded": self._loaded,
            "in_memory_cache": self._in_memory_cache_enabled,
            "ssd_cache": self._resolve_dflash_l2_dir() is not None,
        }

    def get_cache_stats(self) -> dict[str, Any] | None:
        if self._embedded_vlm is not None:
            return self._embedded_vlm.get_cache_stats()
        return None
