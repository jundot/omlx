# SPDX-License-Identifier: Apache-2.0
"""
DFlash engine for block diffusion speculative decoding.

This engine wraps dflash-mlx (>= 0.1.5) to provide 3-4x faster decoding on
Apple Silicon. By default it serves all requests through dflash; setting
``model_settings.dflash_max_ctx`` opts into evicting the dflash models and
delegating long-context requests to omlx's BatchedEngine/VLMBatchedEngine
(paged cache, SSD cache, continuous batching).
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

    DFlash 0.1.5 only registers ``QwenGdnTargetOps``, so any non-Qwen target
    raises ``NotImplementedError`` from ``resolve_target_ops`` at load time.
    Mirroring dflash's heuristic here lets the admin UI disable the toggle
    upfront with a clear reason instead of letting the backend crash.

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
    model_type = str(cfg.get("model_type", "")).lower()
    if "qwen" not in model_type:
        return False, (
            f"DFlash currently supports only Qwen models "
            f"(model_type='{cfg.get('model_type', '')}')"
        )
    return True, ""


class DFlashEngine(BaseEngine):
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
        draft_quant_bits: int | None = None,
        model_settings: Any | None = None,
        fallback_engine_type: str = "batched",
        scheduler_config: Any | None = None,
        omlx_ssd_cache_dir: str | Path | None = None,
    ):
        self._model_name = model_name
        self._draft_model_path = draft_model_path
        self._draft_quant_bits = draft_quant_bits
        self._model_settings = model_settings
        self._fallback_engine_type = fallback_engine_type
        self._scheduler_config = scheduler_config
        self._omlx_ssd_cache_dir = (
            Path(omlx_ssd_cache_dir) if omlx_ssd_cache_dir else None
        )

        self._target_model = None
        self._draft_model = None
        self._tokenizer_obj = None
        self._executor_tokenizer = None
        self._loaded = False
        self._active_request = False
        self._model_type_str = None
        self._fallback_engine: BaseEngine | None = None
        self._in_fallback_mode = False
        self._runtime_context: Any | None = None
        self._dflash_prefix_cache: Any | None = None

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
    def _bits_to_quant_spec(bits: int | None) -> str | None:
        """Convert legacy bits config into dflash 0.1.5's spec string format."""
        if bits is None:
            return None
        if bits == 4:
            return "w4"  # dflash defaults: act_bits=16, group_size=64
        if bits == 8:
            return "w8"
        raise ValueError(f"unsupported draft_quant_bits: {bits}")

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
        from dflash_mlx.runtime_context import (
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

        def _load_models():
            from dflash_mlx.runtime import load_draft_bundle, load_target_bundle

            model, tokenizer, meta = load_target_bundle(self._model_name)
            draft, draft_meta = load_draft_bundle(
                self._draft_model_path,
                draft_quant=self._bits_to_quant_spec(self._draft_quant_bits),
            )
            return model, tokenizer, meta, draft

        result = await loop.run_in_executor(get_mlx_executor(), _load_models)
        self._target_model, self._tokenizer_obj, target_meta, self._draft_model = result

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

        self._runtime_context = self._build_runtime_context()

        self._loaded = True
        self._in_fallback_mode = False
        max_ctx_display = "unlimited" if self._max_dflash_ctx is None else self._max_dflash_ctx
        logger.info(
            f"DFlashEngine loaded: target={self._model_name}, "
            f"draft={self._draft_model_path}, "
            f"max_ctx={max_ctx_display}, "
            f"fallback={self._fallback_engine_type}, "
            f"l1_cache={self._in_memory_cache_enabled}, "
            f"l2_cache={self._resolve_dflash_l2_dir() is not None}"
        )

    async def _evict_dflash_and_start_fallback(self) -> None:
        """Evict dflash models from memory, verify release, then start fallback engine."""
        from dflash_mlx.server.prefix_cache_flow import shutdown_dflash_prefix_cache

        from ..engine_core import get_mlx_executor

        loop = asyncio.get_running_loop()
        pre_active = mx.get_active_memory()

        # Release dflash model and cache references
        shutdown_dflash_prefix_cache()
        self._dflash_prefix_cache = None
        self._runtime_context = None
        self._target_model = None
        self._draft_model = None
        self._executor_tokenizer = None

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
        logger.info(
            f"DFlash fallback engine started: {self._fallback_engine_type}"
        )

    async def stop(self) -> None:
        from dflash_mlx.server.prefix_cache_flow import shutdown_dflash_prefix_cache

        if self._fallback_engine is not None:
            await self._fallback_engine.stop()
            self._fallback_engine = None
        try:
            shutdown_dflash_prefix_cache()
        except Exception as exc:
            logger.debug(f"shutdown_dflash_prefix_cache: {exc}")
        self._dflash_prefix_cache = None
        self._runtime_context = None
        self._target_model = None
        self._draft_model = None
        self._tokenizer_obj = None
        self._executor_tokenizer = None
        self._in_fallback_mode = False
        self._loaded = False
        logger.info("DFlashEngine stopped")

    def _apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> str:
        if hasattr(self._tokenizer_obj, "apply_chat_template"):
            is_partial = detect_and_strip_partial(messages)
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
    ) -> int:
        template_tools = convert_tools_for_template(tools) if tools else None
        prompt = self._apply_chat_template(
            messages, template_tools, chat_template_kwargs=chat_template_kwargs
        )
        return len(self._tokenizer_obj.encode(prompt))

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
            tokenizer=self._executor_tokenizer,
            draft_model=self._draft_model,
            prompt="",
            max_new_tokens=max_tokens,
            stop_token_ids=stop_ids,
            prompt_tokens_override=prompt_tokens,
            prefix_snapshot=prefix_flow.snapshot,
            stable_prefix_len=prefix_flow.stable_prefix_len,
            prefix_cache=prefix_flow.cache,
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

            for event in event_iter:
                if stop_event.is_set():
                    logger.info("DFlash generation aborted by client")
                    break

                event_type = event.get("event")

                if event_type == "prefill_snapshot_ready":
                    prefix_flow.handle_prefill_snapshot(event)
                    continue
                if event_type == "generation_snapshot_ready":
                    prefix_flow.handle_generation_snapshot(event)
                    continue

                if event_type == "token":
                    token_id = event["token_id"]
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

                elif event_type == "summary":
                    gen_tokens = event.get("generation_tokens", 0)
                    accept_ratio = event.get("acceptance_ratio", 0)
                    cycles = event.get("cycles_completed", 0)
                    elapsed_us = event.get("elapsed_us", 0)
                    elapsed_s = elapsed_us / 1e6 if elapsed_us else 0
                    gen_tps = gen_tokens / elapsed_s if elapsed_s > 0 else 0
                    fallback = event.get("fallback_ar", False)
                    logger.info(
                        f"DFlash generation complete: "
                        f"{gen_tokens} tokens, "
                        f"{gen_tps:.1f} tok/s, "
                        f"acceptance={accept_ratio:.1%}, "
                        f"cycles={cycles}"
                        f"{', fallback=AR' if fallback else ''}"
                    )
                    metrics = {
                        "prompt_tokens": event.get("prompt_token_count", 0),
                        "completion_tokens": gen_tokens,
                        "acceptance_ratio": accept_ratio,
                        "cycles_completed": cycles,
                    }
                    asyncio.run_coroutine_threadsafe(
                        queue.put(("", [], True, metrics)), loop
                    )

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
            self._active_request = False

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

        # Fallback: evict dflash models, start LLM/VLM engine
        if self._should_fallback(prompt_tokens):
            if not self._in_fallback_mode:
                logger.info(
                    f"DFlash context fallback: {len(prompt_tokens)} >= {self._max_dflash_ctx}, "
                    f"evicting dflash models and switching to {self._fallback_engine_type} engine"
                )
                await self._evict_dflash_and_start_fallback()
            return await self._fallback_engine.generate(
                prompt=prompt, max_tokens=max_tokens, temperature=temperature,
                top_p=top_p, top_k=top_k, min_p=min_p,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty, stop=stop, **kwargs,
            )

        # Already in fallback mode but short context came in.
        # Stay in fallback mode (reloading dflash models is expensive).
        if self._in_fallback_mode:
            return await self._fallback_engine.generate(
                prompt=prompt, max_tokens=max_tokens, temperature=temperature,
                top_p=top_p, top_k=top_k, min_p=min_p,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty, stop=stop, **kwargs,
            )

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
                tokens: list[int] = []
                summary: dict[str, Any] | None = None
                for event in event_iter:
                    if stop_event.is_set():
                        logger.info("DFlash generation aborted by client")
                        break
                    event_type = event.get("event")
                    if event_type == "prefill_snapshot_ready":
                        prefix_flow.handle_prefill_snapshot(event)
                        continue
                    if event_type == "generation_snapshot_ready":
                        prefix_flow.handle_generation_snapshot(event)
                        continue
                    if event_type == "token":
                        token_id = int(event["token_id"])
                        if token_id in stop_ids:
                            continue
                        tokens.append(token_id)
                    elif event_type == "summary":
                        summary = event
                return summary, tokens
            finally:
                if event_iter is not None:
                    close = getattr(event_iter, "close", None)
                    if close is not None:
                        try:
                            close()
                        except Exception as exc:
                            logger.debug(f"event_iter.close() raised: {exc}")
                self._active_request = False

        self._active_request = True
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
        summary = summary or {}

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

        return GenerationOutput(
            text=text,
            tokens=generated,
            prompt_tokens=summary.get("prompt_token_count", len(prompt_tokens)),
            completion_tokens=summary.get("generation_tokens", len(generated)),
            finish_reason="stop",
        )

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

        # Fallback: evict dflash models, start LLM/VLM engine
        if self._should_fallback(prompt_tokens):
            if not self._in_fallback_mode:
                logger.info(
                    f"DFlash context fallback: {len(prompt_tokens)} >= {self._max_dflash_ctx}, "
                    f"evicting dflash models and switching to {self._fallback_engine_type} engine"
                )
                await self._evict_dflash_and_start_fallback()
            async for output in self._fallback_engine.stream_generate(
                prompt=prompt, max_tokens=max_tokens, temperature=temperature,
                top_p=top_p, top_k=top_k, min_p=min_p,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty, stop=stop, **kwargs,
            ):
                yield output
            return

        # Already in fallback mode — stay there
        if self._in_fallback_mode:
            async for output in self._fallback_engine.stream_generate(
                prompt=prompt, max_tokens=max_tokens, temperature=temperature,
                top_p=top_p, top_k=top_k, min_p=min_p,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty, stop=stop, **kwargs,
            ):
                yield output
            return

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
        self._active_request = True
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
        prompt = self._apply_chat_template(
            messages, template_tools, chat_template_kwargs=ct_kwargs
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
        prompt = self._apply_chat_template(
            messages, template_tools, chat_template_kwargs=ct_kwargs
        )

        async for output in self.stream_generate(
            prompt=prompt, max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty, **kwargs,
        ):
            yield output

    def has_active_requests(self) -> bool:
        if self._fallback_engine is not None and self._fallback_engine.has_active_requests():
            return True
        return self._active_request

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
        }

    def get_cache_stats(self) -> dict[str, Any] | None:
        if self._fallback_engine is not None:
            return self._fallback_engine.get_cache_stats()
        return None
