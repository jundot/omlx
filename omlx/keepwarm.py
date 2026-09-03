# SPDX-License-Identifier: Apache-2.0
"""Opt-in, idle-gated Metal keepwarm for local inference engines.

The state machine is adapted from ThunderMLX's Apache-2.0 keepwarm design and
adds oMLX-specific admission, cache-clear, shutdown, and live-settings gates.
A loaded engine submits one safe one-thread Metal pulse on a dedicated stream
owned by its existing serialized MLX executor.  The pulse never receives or
retains model, KV-cache, request, tokenizer, or SSD-cache state.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from typing import Any


def _enabled(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _float(name: str, default: float, *, minimum: float = 0.0) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    return max(minimum, value)


def _int(
    name: str,
    default: int,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    value = max(minimum, value)
    return min(value, maximum) if maximum is not None else value


@dataclass(frozen=True)
class KeepwarmConfig:
    """Bounded local keepwarm policy, disabled unless explicitly enabled."""

    enabled: bool = False
    interval_seconds: float = 10.0
    idle_after_seconds: float = 2.0
    matrix_size: int = 1
    repeats: int = 1
    request_start_enabled: bool = True
    request_start_idle_seconds: float = 2.0
    request_start_matrix_size: int = 128
    post_response_enabled: bool = True
    post_response_delay_seconds: float = 5.0
    post_response_matrix_size: int = 128
    large_cache_tokens: int = 8192
    large_cache_interval_seconds: float = 60.0
    slow_threshold_seconds: float = 1.0
    slow_backoff_seconds: float = 60.0
    prompt_tail_prewarm_enabled: bool = False
    prompt_tail_prewarm_delay_seconds: float = 1.0
    prompt_tail_prewarm_min_tokens: int = 256
    prompt_tail_prewarm_max_suffix_tokens: int = 4096
    prompt_tail_prewarm_max_tokens: int = 262144
    prompt_tail_prewarm_chunk_size: int = 128

    @classmethod
    def from_env(cls) -> KeepwarmConfig:
        return cls(
            enabled=_enabled("OMLX_KEEPWARM", False),
            interval_seconds=_float(
                "OMLX_KEEPWARM_INTERVAL_SECONDS", 10.0, minimum=0.25
            ),
            idle_after_seconds=_float("OMLX_KEEPWARM_IDLE_AFTER_SECONDS", 2.0),
            matrix_size=_int("OMLX_KEEPWARM_MATRIX_SIZE", 1, minimum=1, maximum=1024),
            repeats=_int("OMLX_KEEPWARM_REPEATS", 1, minimum=1, maximum=16),
            request_start_enabled=_enabled("OMLX_KEEPWARM_REQUEST_START", True),
            request_start_idle_seconds=_float(
                "OMLX_KEEPWARM_REQUEST_START_IDLE_SECONDS", 2.0
            ),
            request_start_matrix_size=_int(
                "OMLX_KEEPWARM_REQUEST_START_MATRIX_SIZE",
                128,
                minimum=1,
                maximum=1024,
            ),
            post_response_enabled=_enabled("OMLX_KEEPWARM_POST_RESPONSE", True),
            post_response_delay_seconds=_float(
                "OMLX_KEEPWARM_POST_RESPONSE_DELAY_SECONDS", 5.0
            ),
            post_response_matrix_size=_int(
                "OMLX_KEEPWARM_POST_RESPONSE_MATRIX_SIZE",
                128,
                minimum=1,
                maximum=1024,
            ),
            large_cache_tokens=_int("OMLX_KEEPWARM_LARGE_CACHE_TOKENS", 8192),
            large_cache_interval_seconds=_float(
                "OMLX_KEEPWARM_LARGE_CACHE_INTERVAL_SECONDS",
                60.0,
                minimum=0.25,
            ),
            slow_threshold_seconds=_float("OMLX_KEEPWARM_SLOW_THRESHOLD_SECONDS", 1.0),
            slow_backoff_seconds=_float("OMLX_KEEPWARM_SLOW_BACKOFF_SECONDS", 60.0),
            prompt_tail_prewarm_enabled=_enabled(
                "OMLX_KEEPWARM_PROMPT_TAIL",
                False,
            ),
            prompt_tail_prewarm_delay_seconds=_float(
                "OMLX_KEEPWARM_PROMPT_TAIL_DELAY_SECONDS",
                1.0,
                minimum=0.0,
            ),
            prompt_tail_prewarm_min_tokens=_int(
                "OMLX_KEEPWARM_PROMPT_TAIL_MIN_TOKENS",
                256,
                minimum=2,
            ),
            prompt_tail_prewarm_max_suffix_tokens=_int(
                "OMLX_KEEPWARM_PROMPT_TAIL_MAX_SUFFIX_TOKENS",
                4096,
                minimum=1,
            ),
            prompt_tail_prewarm_max_tokens=_int(
                "OMLX_KEEPWARM_PROMPT_TAIL_MAX_TOKENS",
                262144,
                minimum=2,
            ),
            prompt_tail_prewarm_chunk_size=_int(
                "OMLX_KEEPWARM_PROMPT_TAIL_CHUNK_SIZE",
                128,
                minimum=64,
                maximum=4096,
            ),
        )

    @classmethod
    def for_local_engine(cls) -> KeepwarmConfig:
        """Resolve the physically qualified asynchronous local cadence.

        Local engines submit one one-thread pulse asynchronously, so a two
        second periodic cadence and one second post-response arm keep Metal out
        of its measured idle cliff without blocking inference. Explicit
        environment values remain authoritative.
        """

        return replace(
            cls.from_env(),
            interval_seconds=_float(
                "OMLX_KEEPWARM_INTERVAL_SECONDS", 2.0, minimum=0.25
            ),
            post_response_delay_seconds=_float(
                "OMLX_KEEPWARM_POST_RESPONSE_DELAY_SECONDS", 1.0
            ),
            large_cache_interval_seconds=_float(
                "OMLX_KEEPWARM_LARGE_CACHE_INTERVAL_SECONDS",
                2.0,
                minimum=0.25,
            ),
        )


@dataclass(frozen=True)
class KeepwarmAction:
    kind: str
    matrix_size: int
    repeats: int
    idle_seconds: float
    cache_tokens: int = 0


@dataclass(frozen=True)
class MetalKeepwarmTouchResult:
    elapsed_seconds: float
    execution_mode: str


class KeepwarmController:
    """Thread-safe request/idle state machine with bounded telemetry."""

    def __init__(
        self,
        config: KeepwarmConfig | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or KeepwarmConfig.from_env()
        self._clock = clock
        self._lock = threading.Lock()
        now = float(clock())
        self._last_activity_at = now
        self._last_touch_at = float("-inf")
        self._slow_until = float("-inf")
        self._request_active = False
        self._cache_armed = False
        self._cache_tokens = 0
        self._clear_inhibited = False
        self._post_response_pending = False
        self._closed = False
        self._count = 0
        self._failures = 0
        self._skips = 0
        self._slow_count = 0
        self._last_event: dict[str, Any] | None = None

    def configure(self, enabled: bool) -> None:
        """Apply the master switch live without replacing request state."""

        with self._lock:
            self.config = replace(
                self.config,
                enabled=bool(enabled),
                prompt_tail_prewarm_enabled=bool(enabled),
            )
            if not enabled:
                self._post_response_pending = False

    def should_execute(self, action: KeepwarmAction) -> bool:
        """Final gate for an action selected before a live state change."""

        with self._lock:
            if self._closed or not self.config.enabled:
                return False
            if action.kind == "request_start":
                return self._request_active
            return self._cache_armed and not self._request_active

    def request_start_action(self) -> KeepwarmAction | None:
        now = float(self._clock())
        with self._lock:
            idle = max(0.0, now - self._last_activity_at)
            self._request_active = True
            self._post_response_pending = False
            self._last_activity_at = now
            if (
                self._closed
                or not self.config.enabled
                or not self._cache_armed
                or not self.config.request_start_enabled
                or idle < self.config.request_start_idle_seconds
                or now < self._slow_until
            ):
                return None
            return KeepwarmAction(
                kind="request_start",
                matrix_size=self.config.request_start_matrix_size,
                repeats=self.config.repeats,
                idle_seconds=idle,
                cache_tokens=self._cache_tokens,
            )

    def observe_request_state(
        self,
        active: bool,
        *,
        cache_tokens: int | None = None,
    ) -> None:
        """Record real request transitions and arm only after useful activity."""

        now = float(self._clock())
        with self._lock:
            if self._closed:
                return
            if cache_tokens is not None:
                self._cache_tokens = max(0, int(cache_tokens))
                if self._cache_tokens > 0 and not self._clear_inhibited:
                    self._cache_armed = True
            if active:
                self._request_active = True
                self._post_response_pending = False
                return
            if self._request_active:
                self._request_active = False
                # A completed real request is enough to arm Metal warming even
                # when prefix caching is disabled. Cache accounting, when
                # available, only selects the safer long-context cadence.
                self._clear_inhibited = False
                self._cache_armed = True
                self._post_response_pending = True
                self._last_activity_at = now

    def cancel_unstarted_request(self) -> None:
        """Roll back an exclusive admission that failed before scheduler entry."""

        with self._lock:
            self._request_active = False
            self._post_response_pending = False

    def disarm_cache(self) -> None:
        """Stop latent warming after an explicit hot-cache clear."""

        with self._lock:
            self._cache_armed = False
            self._cache_tokens = 0
            self._clear_inhibited = True
            self._post_response_pending = False

    def shutdown(self) -> None:
        """Make every future action a no-op before engine teardown."""

        with self._lock:
            self._closed = True
            self._request_active = False
            self._cache_armed = False
            self._cache_tokens = 0
            self._clear_inhibited = True
            self._post_response_pending = False

    def idle_action(self, *, cache_tokens: int | None = None) -> KeepwarmAction | None:
        now = float(self._clock())
        with self._lock:
            if cache_tokens is not None:
                self._cache_tokens = max(0, int(cache_tokens))
                if self._cache_tokens > 0 and not self._clear_inhibited:
                    self._cache_armed = True
            if (
                self._closed
                or not self.config.enabled
                or not self._cache_armed
                or self._request_active
                or now < self._slow_until
            ):
                return None
            idle = max(0.0, now - self._last_activity_at)
            if (
                self._post_response_pending
                and self.config.post_response_enabled
                and idle >= self.config.post_response_delay_seconds
            ):
                self._post_response_pending = False
                return KeepwarmAction(
                    kind="post_response",
                    matrix_size=self.config.post_response_matrix_size,
                    repeats=self.config.repeats,
                    idle_seconds=idle,
                    cache_tokens=self._cache_tokens,
                )
            interval = self.config.interval_seconds
            if (
                self.config.large_cache_tokens > 0
                and self._cache_tokens >= self.config.large_cache_tokens
            ):
                interval = self.config.large_cache_interval_seconds
            if (
                idle < self.config.idle_after_seconds
                or now - self._last_touch_at < interval
            ):
                return None
            return KeepwarmAction(
                kind="idle",
                matrix_size=self.config.matrix_size,
                repeats=self.config.repeats,
                idle_seconds=idle,
                cache_tokens=self._cache_tokens,
            )

    def record(
        self,
        action: KeepwarmAction,
        *,
        elapsed_seconds: float,
        ok: bool,
        execution_mode: str | None = None,
        error: str | None = None,
    ) -> None:
        now = float(self._clock())
        elapsed_seconds = max(0.0, float(elapsed_seconds))
        event: dict[str, Any] = {
            "ok": bool(ok),
            "action": action.kind,
            "at_monotonic": now,
            "elapsed_ms": elapsed_seconds * 1000.0,
            "idle_seconds": max(0.0, action.idle_seconds),
            "matrix_size": action.matrix_size,
            "repeats": action.repeats,
            "cache_tokens": action.cache_tokens,
        }
        if error:
            event["error"] = str(error)[:500]
        if execution_mode:
            event["execution_mode"] = str(execution_mode)[:50]
        with self._lock:
            self._last_touch_at = now
            self._count += 1
            if not ok:
                self._failures += 1
                self._slow_until = now + self.config.slow_backoff_seconds
                event["failure_backoff_seconds"] = self.config.slow_backoff_seconds
            if elapsed_seconds >= self.config.slow_threshold_seconds:
                self._slow_count += 1
                self._slow_until = now + self.config.slow_backoff_seconds
                event["slow_backoff_seconds"] = self.config.slow_backoff_seconds
            self._last_event = event

    def skip(self, reason: str) -> None:
        with self._lock:
            self._skips += 1
            self._last_event = {
                "ok": True,
                "action": "skip",
                "reason": str(reason)[:200],
                "at_monotonic": float(self._clock()),
            }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.config.enabled,
                "policy": asdict(self.config),
                "request_active": self._request_active,
                "cache_armed": self._cache_armed,
                "cache_tokens": self._cache_tokens,
                "clear_inhibited": self._clear_inhibited,
                "closed": self._closed,
                "count": self._count,
                "failures": self._failures,
                "skips": self._skips,
                "slow_count": self._slow_count,
                "last_event": dict(self._last_event) if self._last_event else None,
            }


class CompiledMetalKeepwarmTouch:
    """Per-engine, lazy one-thread Metal pulse for the serialized MLX lane.

    Idle/post-response work lazily creates a dedicated MLX stream and JITs one
    safe ``mx.fast.metal_kernel`` on the engine worker thread. Request admission
    can only consume an already prepared pulse; a miss skips instead of creating
    a stream, compiling, or allocating. The isolated stream prevents the pulse
    fence from draining unrelated model/cache work queued on the inference
    stream. The only retained operand is one four-byte fp32 array. This object
    never receives or retains model, KV-cache, request, tokenizer, or SSD-cache
    state.
    """

    MIN_MATRIX_SIZE = 1
    MAX_MATRIX_SIZE = 1024
    MIN_REPEATS = 1
    MAX_REPEATS = 16
    KERNEL_SOURCE = """
        uint elem = thread_position_in_grid.x;
        out[elem] = inp[elem] + (T)1e-7;
    """

    def __init__(
        self,
        mx_module: Any,
        *,
        stream: Any | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._mx = mx_module
        self._stream = stream
        self._clock = clock
        self._input: Any | None = None
        self._kernel: Callable[..., Any] | None = None
        self._latest_output: Any | None = None
        self._prepared = False
        self._closed = False

    @classmethod
    def _validate_action(cls, action: KeepwarmAction) -> tuple[int, int]:
        matrix_size = int(action.matrix_size)
        repeats = int(action.repeats)
        if not cls.MIN_MATRIX_SIZE <= matrix_size <= cls.MAX_MATRIX_SIZE:
            raise ValueError(
                "keepwarm matrix size must be between "
                f"{cls.MIN_MATRIX_SIZE} and {cls.MAX_MATRIX_SIZE}"
            )
        if not cls.MIN_REPEATS <= repeats <= cls.MAX_REPEATS:
            raise ValueError(
                "keepwarm repeats must be between "
                f"{cls.MIN_REPEATS} and {cls.MAX_REPEATS}"
            )
        if action.kind not in {"idle", "post_response", "request_start"}:
            raise ValueError(f"unsupported keepwarm action: {action.kind}")
        return matrix_size, repeats

    def _make_output(
        self,
        kernel: Callable[..., Any],
        input_value: Any,
    ) -> Any:
        mx_module = self._mx
        if mx_module is None or self._closed:
            raise RuntimeError("Metal keepwarm pulse is closed")
        outputs = kernel(
            inputs=[input_value],
            template=[("T", mx_module.float32)],
            grid=(1, 1, 1),
            threadgroup=(1, 1, 1),
            output_shapes=[input_value.shape],
            output_dtypes=[input_value.dtype],
        )
        if not outputs:
            raise RuntimeError("Metal keepwarm pulse returned no output")
        return outputs[0]

    @staticmethod
    def _validate_output_metadata(output: Any, input_value: Any) -> None:
        if tuple(output.shape) != tuple(input_value.shape):
            raise RuntimeError(
                "Metal keepwarm pulse returned an unexpected output shape"
            )
        if output.dtype != input_value.dtype:
            raise RuntimeError(
                "Metal keepwarm pulse returned an unexpected output dtype"
            )

    def _submit_async(
        self,
        kernel: Callable[..., Any],
        input_value: Any,
    ) -> None:
        mx_module = self._mx
        if mx_module is None or self._closed:
            raise RuntimeError("Metal keepwarm pulse is closed")
        async_eval = getattr(mx_module, "async_eval", None)
        if not callable(async_eval):
            raise RuntimeError("mx.async_eval is unavailable")
        output = self._make_output(kernel, input_value)
        self._validate_output_metadata(output, input_value)
        async_eval(output)
        # Keep exactly one four-byte output alive. Replacing a prior reference
        # never inserts a global or stream fence; the bounded cadence gives the
        # preceding one-thread command ample time to retire independently.
        self._latest_output = output

    def _prepare(self) -> None:
        mx_module = self._mx
        if mx_module is None or self._closed:
            raise RuntimeError("Metal keepwarm pulse is closed")
        fast_module = getattr(mx_module, "fast", None)
        metal_kernel = getattr(fast_module, "metal_kernel", None)
        if not callable(metal_kernel):
            raise RuntimeError("mx.fast.metal_kernel is unavailable")

        input_value = mx_module.array([1.0], dtype=mx_module.float32)
        kernel = metal_kernel(
            name="omlx_keepwarm_pulse",
            input_names=["inp"],
            output_names=["out"],
            source=self.KERNEL_SOURCE,
            header="",
            ensure_row_contiguous=True,
            atomic_outputs=False,
            compile_options={"math_mode": "safe"},
        )
        # Fixed source/template plus shape/dtype metadata are validated without
        # a host read or completion fence. Unit and isolated integration tests
        # validate the exact 1 + 1e-7 value; production never blocks here.
        self._submit_async(kernel, input_value)
        self._input = input_value
        self._kernel = kernel
        self._prepared = True

    def _ensure_stream(self) -> Any:
        mx_module = self._mx
        if mx_module is None or self._closed:
            raise RuntimeError("Metal keepwarm pulse is closed")
        if self._stream is None:
            new_stream = getattr(mx_module, "new_stream", None)
            if not callable(new_stream):
                raise RuntimeError("mx.new_stream is unavailable")
            # Called only from touch(), which EngineCore dispatches on its
            # existing one-worker MLX executor. Creation, use, and close thus
            # share thread ownership without sharing the model's command queue.
            self._stream = new_stream(mx_module.default_device())
        return self._stream

    def touch(self, action: KeepwarmAction) -> MetalKeepwarmTouchResult | None:
        """Prepare synchronously off-path or submit a bounded async pulse."""

        _matrix_size, repeats = self._validate_action(action)
        if self._closed:
            raise RuntimeError("Metal keepwarm pulse is closed")
        if action.kind == "request_start" and not self._prepared:
            return None
        started = self._clock()
        mx_module = self._mx
        if mx_module is None:
            raise RuntimeError("Metal keepwarm pulse is closed")
        pulse_stream = self._ensure_stream()
        stream_context = mx_module.stream(pulse_stream)
        try:
            with stream_context:
                prepared_now = not self._prepared
                if prepared_now:
                    self._prepare()
                kernel = self._kernel
                input_value = self._input
                if kernel is None or input_value is None:
                    raise RuntimeError("Metal keepwarm pulse was not prepared")
                # _prepare() already submitted the first pulse. Every later
                # pulse is also submission-only.
                remaining = repeats - 1 if prepared_now else repeats
                for _ in range(remaining):
                    self._submit_async(kernel, input_value)
        except Exception:
            # Do not retain or repeatedly call a partial/broken JIT. The outer
            # controller records failure and applies bounded retry backoff.
            self._prepared = False
            self._kernel = None
            self._input = None
            self._latest_output = None
            raise
        return MetalKeepwarmTouchResult(
            elapsed_seconds=max(0.0, float(self._clock()) - started),
            execution_mode=("async_prepared" if prepared_now else "async_submitted"),
        )

    def close(self) -> None:
        """Drop every synthetic array/function reference before teardown."""

        self._closed = True
        mx_module = self._mx
        pulse_stream = self._stream
        try:
            if mx_module is not None and pulse_stream is not None:
                synchronize = getattr(mx_module, "synchronize", None)
                if not callable(synchronize):
                    raise RuntimeError("mx.synchronize is unavailable")
                synchronize(pulse_stream)
        finally:
            self._prepared = False
            self._kernel = None
            self._input = None
            self._latest_output = None
            self._stream = None
            self._mx = None
