# SPDX-License-Identifier: Apache-2.0
"""Serial serving of released K2 Uno conditional diffusion adapters."""

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import gc
import secrets
import threading
import time

import mlx.core as mx

from ..adapter.output_parser import detect_output_parser
from ..api.tool_calling import convert_tools_for_template
from ..api.utils import detect_and_strip_partial
from ..engine_core import get_mlx_executor
from ..exceptions import InvalidRequestError
from ..memory_monitor import (
    MemoryMonitor,
    raise_if_prefill_exceeds,
    set_model_info_from_model,
)
from ..patches.k2_horizon import apply_k2_horizon_patch
from ..patches.k2_horizon.uno_adapter import load_uno_adapter
from ..patches.k2_horizon.uno_decode import UnoDecoder
from ..uno_bundle import UnoBundle, resolve_uno_bundle
from ..utils.model_loading import lm_load_compat
from ..utils.proc_memory import get_phys_footprint
from ..utils.tokenizer import get_tokenizer_config
from .base import ActivityTrackingMixin, BaseEngine, GenerationOutput


class _UnoPrefillGuard:
    """Apply the process enforcer's watermarks to Uno prefill."""

    def __init__(self, memory_monitor: MemoryMonitor, prefill_step_size: int):
        self.memory_monitor = memory_monitor
        self._prefill_step_size = prefill_step_size
        self._last_mlx_active_memory_bytes: int = 0
        self._prefill_memory_guard: bool = False
        self._memory_hard_limit_bytes: int = 0
        self._memory_hot_cache_used_bytes: int = 0
        self._memory_static_ceiling_bytes: int = 0
        self._memory_dynamic_ceiling_bytes: int = 0
        self._memory_metal_cap_bytes: int = 0
        self._memory_guard_tier: str = ""

    def record_mlx_active_memory(self, active_bytes: int) -> None:
        self._last_mlx_active_memory_bytes = max(0, int(active_bytes))

    def _current_usage_bytes(self) -> int:
        phys = max(
            0, get_phys_footprint() - max(0, int(self._memory_hot_cache_used_bytes))
        )
        return max(self._last_mlx_active_memory_bytes, phys)

    def preflight_or_raise(
        self,
        *,
        num_prompt_tokens: int,
        request_id: str | None = None,
        extra_bytes: int = 0,
    ) -> None:
        raise_if_prefill_exceeds(
            self.memory_monitor,
            prefill_memory_guard=self._prefill_memory_guard,
            hard_limit_bytes=self._memory_hard_limit_bytes,
            current_usage_bytes=self._current_usage_bytes() + max(0, extra_bytes),
            prefill_step_size=self._prefill_step_size,
            num_prompt_tokens=num_prompt_tokens,
            request_id=request_id,
            static_ceiling_bytes=self._memory_static_ceiling_bytes,
            dynamic_ceiling_bytes=self._memory_dynamic_ceiling_bytes,
            metal_cap_bytes=self._memory_metal_cap_bytes,
            memory_guard_tier=self._memory_guard_tier,
        )


class _StopBuffer:
    """Hold any suffix that could become a stop string across token boundaries."""

    def __init__(self, stops):
        self.stops = stops
        self.pending = ""
        self.stopped = False

    def feed(self, text, *, final=False):
        if self.stopped:
            return ""
        self.pending += text
        positions = [self.pending.find(stop) for stop in self.stops]
        found = [position for position in positions if position >= 0]
        if found:
            self.stopped = True
            text, self.pending = self.pending[: min(found)], ""
            return text
        keep = 0
        if not final:
            for stop in self.stops:
                for size in range(1, min(len(stop), len(self.pending) + 1)):
                    if self.pending.endswith(stop[:size]):
                        keep = max(keep, size)
        split = len(self.pending) - keep
        text, self.pending = self.pending[:split], self.pending[split:]
        return text


class UnoEngine(ActivityTrackingMixin, BaseEngine):
    """Serve committed Uno tokens with request-local KV on the shared MLX executor."""

    is_uno_model = True

    def __init__(
        self, model_name, *, adapter_path, scheduler_config=None, model_settings=None
    ):
        super().__init__()
        self._model_name = str(model_name)
        self._adapter_path = adapter_path
        self._settings = model_settings
        self._scheduler_config = scheduler_config
        self._model = self._tokenizer = self._executor_tokenizer = None
        self._bundle = self._adapter_info = self._output_parser_factory = None
        self._prefill_guard = None
        self._lock = asyncio.Lock()
        self._events = set()
        self._closing = False
        self._prefill_step = 512
        self._last_speculation = {}

    @property
    def model_name(self):
        return self._model_name

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model_type(self):
        return "k2_horizon"

    def _load_model(self, bundle: UnoBundle):
        """Load the base and conditional adapter on the MLX executor."""
        apply_k2_horizon_patch()
        model, tokenizer = lm_load_compat(
            str(bundle.base_path),
            trust_remote_code=False,
            tokenizer_config=get_tokenizer_config(str(bundle.base_path)),
        )
        info = load_uno_adapter(
            model, bundle.adapter_path, base_model_id=bundle.base_model_id
        )
        mx.eval(model.parameters())
        monitor = MemoryMonitor(max_kv_cache_memory=None, eviction_enabled=False)
        set_model_info_from_model(monitor, model)
        guard = _UnoPrefillGuard(monitor, self._prefill_step)
        guard.record_mlx_active_memory(mx.get_active_memory())
        factory = detect_output_parser(str(bundle.base_path), tokenizer, bundle.config)
        if factory is None or factory.kind != "k2_horizon":
            raise ValueError("Uno requires the K2 Horizon output parser and tokenizer")
        return model, tokenizer, copy.deepcopy(tokenizer), info, guard, factory

    async def start(self):
        async with self._lock:
            if self._model is not None:
                return
            bundle = resolve_uno_bundle(self._model_name, self._adapter_path)

            result = await asyncio.get_running_loop().run_in_executor(
                get_mlx_executor(), self._load_model, bundle
            )
            (
                self._model,
                self._tokenizer,
                self._executor_tokenizer,
                self._adapter_info,
                self._prefill_guard,
                self._output_parser_factory,
            ) = result
            self._bundle = bundle
            self._closing = False

    async def stop(self):
        self._closing = True
        for event in tuple(self._events):
            event.set()
        async with self._lock:

            def unload():
                self._model = self._executor_tokenizer = self._tokenizer = None
                self._output_parser_factory = self._prefill_guard = None
                gc.collect()
                mx.clear_cache()

            await asyncio.get_running_loop().run_in_executor(get_mlx_executor(), unload)

    def _prompt_ids(self, prompt):
        if self._tokenizer is None:
            raise RuntimeError("Uno engine is not loaded")
        ids = (
            self._tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
        )
        vocab = self._bundle.config["vocab_size"]
        if not ids or any(
            type(token) is not int or not 0 <= token < vocab for token in ids
        ):
            raise InvalidRequestError(
                "Uno requires a nonempty prompt of valid vocabulary token IDs"
            )
        return ids

    @staticmethod
    def _validate_options(
        max_tokens=256,
        temperature=0.7,
        top_p=0.9,
        top_k=0,
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        stop=None,
        **kwargs,
    ):
        import math

        if type(max_tokens) is not int or max_tokens < 0:
            raise InvalidRequestError("Uno max_tokens must be a nonnegative integer")
        if not math.isfinite(temperature) or temperature < 0:
            raise InvalidRequestError("Uno temperature must be finite and nonnegative")
        if not math.isfinite(top_p) or not 0 < top_p <= 1:
            raise InvalidRequestError("Uno top_p must be in (0, 1]")
        if type(top_k) is not int or top_k < 0:
            raise InvalidRequestError("Uno top_k must be a nonnegative integer")
        for name, value, default in (
            ("min_p", min_p, 0),
            ("repetition_penalty", repetition_penalty, 1),
            ("presence_penalty", presence_penalty, 0),
            ("frequency_penalty", kwargs.get("frequency_penalty", 0), 0),
            ("xtc_probability", kwargs.get("xtc_probability", 0), 0),
        ):
            if value != default:
                raise InvalidRequestError(
                    f"Uno does not yet support {name}; its p/q sampler must implement the same transform"
                )
        accepted = {
            "seed",
            "tools",
            "request_id",
            "frequency_penalty",
            "xtc_probability",
            "xtc_threshold",
            "repetition_context_size",
        }
        for name, value in kwargs.items():
            if name not in accepted and value is not None:
                raise InvalidRequestError(f"Uno does not support request option {name}")
        seed = kwargs.get("seed")
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
            raise InvalidRequestError("Uno seed must be an integer in [0, 2**32)")
        stops = [stop] if isinstance(stop, str) else list(stop or [])
        if any(not isinstance(item, str) or not item for item in stops):
            raise InvalidRequestError("Uno stop strings must be nonempty strings")
        return stops

    def _preflight(self, ids, *, max_tokens=256, request_id=None, **kwargs):
        self._validate_options(max_tokens=max_tokens, request_id=request_id, **kwargs)
        if kwargs.get("top_k", 0) > self._bundle.config["vocab_size"]:
            raise InvalidRequestError("Uno top_k exceeds vocabulary size")
        context = self._bundle.context_length
        prompt_limit = getattr(self._settings, "max_context_window", None)
        if prompt_limit and len(ids) > prompt_limit:
            raise InvalidRequestError(
                f"Uno prompt exceeds configured context limit {prompt_limit}"
            )
        if len(ids) + max_tokens > context:
            raise InvalidRequestError(
                f"Uno prompt plus max_tokens exceeds context length {context}"
            )
        # Account for future KV, proposal distributions and dense prefill logits.
        transient = (
            max(self._prefill_step, self._bundle.block_size)
            * self._bundle.config["vocab_size"]
            * 4
            * 6
        )
        guard = self._prefill_guard
        guard.preflight_or_raise(
            num_prompt_tokens=len(ids) + max_tokens,
            request_id=request_id,
            extra_bytes=transient,
        )

    def _chat_prompt(
        self, messages, tools=None, chat_template_kwargs=None, is_partial=None
    ):
        messages = copy.deepcopy(messages)
        if is_partial is None:
            is_partial = detect_and_strip_partial(messages)
        else:
            for message in messages:
                message.pop("partial", None)
        from ..patches.k2_horizon import validate_chat_template_kwargs

        options = dict(chat_template_kwargs or {})
        validate_chat_template_kwargs(options)
        options.update(tokenize=False, add_generation_prompt=not is_partial)
        if is_partial:
            options["continue_final_message"] = True
        if tools:
            options["tools"] = convert_tools_for_template(tools)
        return self._tokenizer.apply_chat_template(messages, **options)

    def count_chat_tokens(
        self, messages, tools=None, chat_template_kwargs=None, is_partial=None
    ):
        return len(
            self._prompt_ids(
                self._chat_prompt(messages, tools, chat_template_kwargs, is_partial)
            )
        )

    async def preflight_completion(self, prompt, request_id=None, **kwargs):
        self._preflight(self._prompt_ids(prompt), request_id=request_id, **kwargs)

    async def preflight_chat(self, messages, tools=None, request_id=None, **kwargs):
        prompt = self._chat_prompt(
            messages,
            tools,
            kwargs.pop("chat_template_kwargs", None),
            kwargs.pop("is_partial", None),
        )
        self._preflight(
            self._prompt_ids(prompt), tools=tools, request_id=request_id, **kwargs
        )

    def _run(self, ids, options, stops, cancelled, publish):
        tokenizer = self._executor_tokenizer
        factory = self._output_parser_factory
        parser = factory.create_session_with_tools(tokenizer, options.get("tools"))
        # A template may leave one of the three IFM reasoning channels open.
        prompt_text = tokenizer.decode(ids)
        opened = any(
            prompt_text.rfind(a) > prompt_text.rfind(b)
            for a, b in factory.thinking_marker_pairs
        )
        if opened:
            parser.notify_prefilled_thought()
        decoder = UnoDecoder(
            self._model,
            eos_token_ids=tokenizer.eos_token_ids,
            block_size=self._bundle.block_size,
            temperature=options["temperature"],
            top_p=options["top_p"],
            top_k=options["top_k"] or None,
            seed=(
                options.get("seed")
                if options.get("seed") is not None
                else secrets.randbits(32)
            ),
            prefill_step_size=self._prefill_step,
        )
        buffer = _StopBuffer(stops)
        text = ""
        tokens = []
        first = None
        started = time.perf_counter()
        cycles = accepted = proposals = forwards = 0
        finish = "length"

        def emit(delta, *, finished=False, tool_calls=None):
            nonlocal text, first
            text += delta
            now = time.perf_counter()
            if first is None and tokens:
                first = now
            publish(
                GenerationOutput(
                    text=text,
                    tokens=list(tokens),
                    new_text=delta,
                    finished=finished,
                    prompt_tokens=len(ids),
                    completion_tokens=len(tokens),
                    finish_reason=finish if finished else None,
                    tool_calls=tool_calls,
                    generation_tps=len(tokens) / max(now - started, 1e-9),
                    generated_at=started,
                    generated_until=now,
                    first_token_at=first,
                )
            )

        if opened and options["max_tokens"]:
            emit(buffer.feed("<think>\n"))
        iterator = decoder.generate(
            ids, max_tokens=options["max_tokens"], cancelled=cancelled.is_set
        )
        try:
            for cycle in iterator:
                cycles += 1
                before_cycle = len(tokens)
                proposals += cycle.proposed_tokens
                forwards += cycle.forwards
                for token in cycle.tokens:
                    if cancelled.is_set():
                        return
                    tokens.append(token)
                    if token in tokenizer.eos_token_ids:
                        finish = "stop"
                        break
                    result = parser.process_token(token)
                    delta = buffer.feed(result.stream_text)
                    if delta:
                        emit(delta)
                    if buffer.stopped or result.is_stop:
                        finish = "stop"
                        break
                accepted += min(
                    cycle.accepted_proposals, max(0, len(tokens) - before_cycle - 1)
                )
                self._prefill_guard.record_mlx_active_memory(mx.get_active_memory())
                if finish == "stop":
                    break
                if cycle.finish_reason:
                    finish = cycle.finish_reason
        finally:
            iterator.close()
        if cancelled.is_set():
            return
        final = parser.finalize()
        delta = buffer.feed(final.stream_text, final=True)
        if buffer.stopped:
            finish = "stop"
        elif final.finish_reason:
            finish = final.finish_reason
        self._last_speculation = {
            "cycles": cycles,
            "accepted_proposals": accepted,
            "proposed_tokens": proposals,
            "forwards": forwards,
            "committed_tokens": len(tokens),
            "tokens_per_forward": len(tokens) / forwards if forwards else 0,
        }
        emit(delta, finished=True, tool_calls=final.tool_calls or None)

    async def stream_generate(
        self,
        prompt,
        max_tokens=256,
        temperature=0.7,
        top_p=0.9,
        top_k=0,
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        stop=None,
        **kwargs,
    ):
        options = dict(
            kwargs,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
        )
        stops = self._validate_options(stop=stop, **options)
        ids = self._prompt_ids(prompt)
        event = threading.Event()
        self._events.add(event)
        activity = self._begin_activity(
            "generation", detail="Uno", total_items=max_tokens
        )
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue(maxsize=8)

        def publish(value):
            future = asyncio.run_coroutine_threadsafe(queue.put(value), loop)
            while not event.is_set():
                try:
                    future.result(timeout=0.1)
                    return
                except concurrent.futures.TimeoutError:
                    continue
            future.cancel()

        async def produce():
            try:
                async with self._lock:
                    if event.is_set():
                        return
                    if self._closing:
                        raise RuntimeError("Uno engine is stopping")
                    self._preflight(ids, **options)

                    def run():
                        try:
                            self._run(ids, options, stops, event, publish)
                        finally:
                            self._prefill_guard.record_mlx_active_memory(
                                mx.get_active_memory()
                            )

                    worker = loop.run_in_executor(get_mlx_executor(), run)
                    try:
                        await asyncio.shield(worker)
                    finally:
                        event.set()
                        await asyncio.shield(worker)
            finally:
                self._events.discard(event)
                self._end_activity(activity)

        # Keep lock ownership with the producer so paused clients cannot block stop().
        producer = asyncio.create_task(produce())
        try:
            while True:
                if producer.done() and queue.empty():
                    producer.result()
                    break
                receive = asyncio.create_task(queue.get())
                try:
                    done, _ = await asyncio.wait(
                        (receive, producer), return_when=asyncio.FIRST_COMPLETED
                    )
                    if receive not in done:
                        producer.result()
                        continue
                    yield receive.result()
                finally:
                    if not receive.done():
                        receive.cancel()
                        await asyncio.gather(receive, return_exceptions=True)
        finally:
            event.set()
            await asyncio.shield(producer)

    async def generate(self, prompt, **kwargs):
        result = None
        async for output in self.stream_generate(prompt, **kwargs):
            result = output
        if result is None or not result.finished:
            raise RuntimeError("Uno request ended without a completion")
        return result

    async def stream_chat(self, messages, tools=None, **kwargs):
        prompt = self._chat_prompt(
            messages,
            tools,
            kwargs.pop("chat_template_kwargs", None),
            kwargs.pop("is_partial", None),
        )
        stream = self.stream_generate(prompt, tools=tools, **kwargs)
        try:
            async for output in stream:
                yield output
        finally:
            await stream.aclose()

    async def chat(self, messages, tools=None, **kwargs):
        result = None
        async for output in self.stream_chat(messages, tools=tools, **kwargs):
            result = output
        if result is None or not result.finished:
            raise RuntimeError("Uno request ended without a completion")
        return result

    def get_stats(self):
        return {
            "engine_type": "uno",
            "model_name": self.model_name,
            "loaded": self._model is not None,
            "adapter": self._adapter_info,
            "speculation": dict(self._last_speculation),
        }

    def get_cache_stats(self):
        return None
