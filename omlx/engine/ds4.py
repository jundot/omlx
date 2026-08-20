# SPDX-License-Identifier: Apache-2.0
"""EnginePool adapter for OMLX-managed DS4 subprocesses.

This adapter owns DS4 lifecycle management and byte-preserving proxy helpers for
protocol endpoints that have been wired through OMLX.  BaseEngine generation
methods still raise clear errors for endpoints that are not proxied yet.
"""

from __future__ import annotations

import asyncio
import re
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import requests

DS4_STREAM_FLUSH_BYTES = 2048

from ..ds4_process import DS4_HOST, DS4LaunchConfig, DS4ManagedProcess
from ..settings import DEFAULT_BASE_PATH, DS4Settings
from .base import BaseEngine, GenerationOutput


# DS4 backend HTTP timeouts (seconds).  The connect timeout is generous
# enough for a cold-start fork on a loaded machine; the read timeout
# must accommodate long streaming responses (e.g. max-token generations
# on slow hardware).
_DS4_CONNECT_TIMEOUT = 10.0
_DS4_READ_TIMEOUT = 600.0

class DS4ProxyError(RuntimeError):
    """Raised when OMLX cannot contact the managed DS4 backend."""


_DS4_NUMBER_RE = r"[-+]?\d+(?:\.\d+)?"
_DS4_PREFILL_PROGRESS_RE = re.compile(
    rf"ds4-server: (?P<kind>chat|completion) ctx=.*? "
    rf"(?:[A-Z_,]+ )?(?P<phase>.+?) chunk (?P<current>\d+)/(?P<total>\d+) "
    rf"\((?P<percent>{_DS4_NUMBER_RE})%\) "
    rf"chunk=(?P<chunk_tps>{_DS4_NUMBER_RE}) t/s "
    rf"avg=(?P<avg_tps>{_DS4_NUMBER_RE}) t/s "
    rf"(?P<elapsed>{_DS4_NUMBER_RE})s"
)
_DS4_DECODE_PROGRESS_RE = re.compile(
    rf"ds4-server: (?P<kind>chat|completion) ctx=.*? "
    rf"gen=(?P<generated>\d+).*? decoding "
    rf"chunk=(?P<chunk_tps>{_DS4_NUMBER_RE}) t/s "
    rf"avg=(?P<avg_tps>{_DS4_NUMBER_RE}) t/s "
    rf"(?P<elapsed>{_DS4_NUMBER_RE})s"
)


def _ds4_progress_float(value: str) -> float:
    return round(float(value), 3)


def parse_ds4_progress_log_line(line: str) -> dict[str, Any] | None:
    """Parse one DS4 server progress log line into admin/UI metrics."""
    if match := _DS4_PREFILL_PROGRESS_RE.search(line):
        return {
            "kind": match.group("kind"),
            "phase": match.group("phase").strip(),
            "phase_type": "prefill",
            "current_tokens": int(match.group("current")),
            "total_tokens": int(match.group("total")),
            "percent": _ds4_progress_float(match.group("percent")),
            "chunk_tokens_per_second": _ds4_progress_float(match.group("chunk_tps")),
            "average_tokens_per_second": _ds4_progress_float(match.group("avg_tps")),
            "elapsed_seconds": _ds4_progress_float(match.group("elapsed")),
        }
    if match := _DS4_DECODE_PROGRESS_RE.search(line):
        return {
            "kind": match.group("kind"),
            "phase": "decoding",
            "phase_type": "generation",
            "generated_tokens": int(match.group("generated")),
            "chunk_tokens_per_second": _ds4_progress_float(match.group("chunk_tps")),
            "average_tokens_per_second": _ds4_progress_float(match.group("avg_tps")),
            "elapsed_seconds": _ds4_progress_float(match.group("elapsed")),
        }
    return None


# Regex patterns for DS4 KV cache log lines.
_DS4_KV_STORE_RE = re.compile(
    r"ds4-server: kv cache stored tokens=(?P<tokens>\d+) "
    r"trimmed=(?P<trimmed>\d+) reason=(?P<reason>\S+) "
    r"key=(?P<key>\S+) size=(?P<size_mb>" + _DS4_NUMBER_RE + r") MiB "
    r"save=(?P<save_ms>" + _DS4_NUMBER_RE + r") ms"
)
_DS4_KV_HIT_RE = re.compile(
    r"ds4-server: kv cache hit "
    r"text(?P<respproto> RESPPROTO)? tokens=(?P<text_tokens>\d+) text=(?P<text_chars>\d+) "
    r"quant=(?P<quant>\d+) key=(?P<key>\S+) "
    r"load=(?P<load_ms>" + _DS4_NUMBER_RE + r") ms (?:consumed )?file=(?P<file>\S+)"
)
_DS4_KV_EVICTED_RE = re.compile(
    r"ds4-server: kv cache evicted "
    r"reason=(?P<reason>\S+) tokens=(?P<tokens>\d+) "
    r"hits=(?P<hits>\d+) size=(?P<size_mb>" + _DS4_NUMBER_RE + r") MiB "
    r"file=(?P<file>\S+)"
)
_DS4_KV_LOAD_FAILED_RE = re.compile(
    r"ds4-server: kv cache load failed"
)
_DS4_KV_SKIPPED_RE = re.compile(
    r"ds4-server: kv cache skipped"
)


def parse_ds4_kv_log_line(line: str) -> dict[str, Any] | None:
    """Parse one DS4 KV cache event line into structured metrics."""
    if match := _DS4_KV_STORE_RE.search(line):
        return {
            "event": "store",
            "tokens": int(match.group("tokens")),
            "size_mb": _ds4_progress_float(match.group("size_mb")),
            "save_ms": _ds4_progress_float(match.group("save_ms")),
        }
    if match := _DS4_KV_HIT_RE.search(line):
        return {
            "event": "hit",
            "text_tokens": int(match.group("text_tokens")),
            "load_ms": _ds4_progress_float(match.group("load_ms")),
        }
    if match := _DS4_KV_EVICTED_RE.search(line):
        return {
            "event": "evicted",
            "tokens": int(match.group("tokens")),
            "hits": int(match.group("hits")),
            "size_mb": _ds4_progress_float(match.group("size_mb")),
        }
    if _DS4_KV_LOAD_FAILED_RE.search(line):
        return {"event": "load_failed"}
    if _DS4_KV_SKIPPED_RE.search(line):
        return {"event": "skipped"}
    return None


@dataclass(frozen=True)
class DS4ProxyResponse:
    """Raw non-streaming DS4 HTTP response."""

    status_code: int
    headers: dict[str, str]
    body: bytes


@dataclass
class DS4StreamingProxyResponse:
    """Raw streaming DS4 HTTP response with close-time cleanup."""

    status_code: int
    headers: dict[str, str]
    response: requests.Response
    on_close: Callable[[], None]
    session: requests.Session | None = None
    closed: bool = False

    def close(self) -> None:
        """Close the upstream response and run cleanup exactly once."""
        if self.closed:
            return
        self.closed = True
        self.response.close()
        if self.session is not None:
            self.session.close()
        self.on_close()

    @staticmethod
    def _raw_read(raw: object, amount: int) -> bytes:
        """Read at most ``amount`` raw bytes without content decoding."""
        read = getattr(raw, "read")
        try:
            return read(amount, decode_content=False)
        except TypeError:
            return read(amount)

    @classmethod
    def _iter_low_latency_raw_bytes(cls, raw: object) -> Iterator[bytes]:
        """Yield raw SSE bytes as soon as an event boundary is observed.

        Reads in buffered chunks (up to DS4_STREAM_FLUSH_BYTES at a time)
        and scans for ``\n\n`` / ``\r\n\r\n`` SSE event separators,
        yielding each complete event immediately.  Partial data is kept in
        the buffer and yielded at EOF.  This avoids the per-byte system
        call overhead of the old single-byte read loop.
        """
        buffer = bytearray()
        while True:
            chunk = cls._raw_read(raw, DS4_STREAM_FLUSH_BYTES)
            if not chunk:
                break
            buffer.extend(chunk)
            # Scan for event boundaries within the accumulated buffer.
            # Yield complete events as soon as the separator is found,
            # retaining any trailing partial data in the buffer.
            sep1 = b"\n\n"
            sep2 = b"\r\n\r\n"
            while True:
                pos1 = buffer.find(sep1)
                pos2 = buffer.find(sep2)
                # Use the earliest separator to keep latency low.
                if pos1 != -1 and (pos2 == -1 or pos1 <= pos2):
                    end = pos1 + len(sep1)
                elif pos2 != -1:
                    end = pos2 + len(sep2)
                else:
                    break
                yield bytes(buffer[:end])
                del buffer[:end]
            # Flush on size threshold as a safety valve (no separator found
            # but the buffer has grown large — e.g. a malformed stream).
            if len(buffer) >= DS4_STREAM_FLUSH_BYTES:
                yield bytes(buffer)
                buffer.clear()
        if buffer:
            yield bytes(buffer)

    def iter_bytes(self) -> Iterator[bytes]:
        """Yield DS4 response bytes without parsing or reformatting them."""
        if self.closed:
            return
        try:
            raw = getattr(self.response, "raw", None)
            if raw is not None and hasattr(raw, "read"):
                yield from self._iter_low_latency_raw_bytes(raw)
            elif raw is not None and hasattr(raw, "stream"):
                for chunk in raw.stream(DS4_STREAM_FLUSH_BYTES, decode_content=False):
                    if chunk:
                        yield chunk
            else:
                for chunk in self.response.iter_content(chunk_size=DS4_STREAM_FLUSH_BYTES):
                    if chunk:
                        yield chunk
        finally:
            self.close()


class DS4ProcessEngine(BaseEngine):
    """Minimal BaseEngine wrapper around one managed ds4-server process."""

    def __init__(
        self,
        *,
        model_id: str,
        model_path: str | Path,
        settings: DS4Settings | None = None,
        base_path: str | Path | None = None,
        context_tokens: int | None = None,
        auto_enable_ssd_streaming: bool = False,
        model_settings: Any | None = None,
    ):
        self.model_id = model_id
        self._model_path = Path(model_path)
        self.settings = settings or DS4Settings()
        self.base_path = Path(base_path) if base_path is not None else DEFAULT_BASE_PATH
        self.context_tokens = context_tokens
        self.auto_enable_ssd_streaming = auto_enable_ssd_streaming
        self.model_settings = model_settings
        self.process: DS4ManagedProcess | None = None
        self._lifecycle_lock = asyncio.Lock()
        self._crash_count = 0
        self._restart_count = 0
        self._last_crash_exit_code: int | None = None
        self._last_crash_logs = ""
        self._last_crash_monotonic_time: float | None = None
        self._recorded_crash_process_id: int | None = None
        self._active_requests = 0
        self._active_requests_lock = threading.Lock()
        # Incremental KV cache stats — updated lazily in get_cache_stats().
        self._kv_stats_process_key: int | None = None
        self._kv_stats_cursor: int = 0
        self._kv_hits: int = 0
        self._kv_stores: int = 0
        self._kv_evictions: int = 0
        self._kv_load_failures: int = 0
        self._kv_skips: int = 0
        self._kv_tokens_stored: int = 0
        self._kv_tokens_restored: int = 0
        self._kv_bytes_stored: int = 0
        self._kv_last_store_ms: float | None = None
        self._kv_last_load_ms: float | None = None

    @property
    def model_name(self) -> str:
        """Return the GGUF model path used to launch DS4."""
        return str(self._model_path)

    @property
    def tokenizer(self) -> Any:
        """DS4 tokenization stays inside the subprocess."""
        return None

    @property
    def model_type(self) -> str:
        """Expose a stable model type for server-side type checks."""
        return "ds4"

    @property
    def port(self) -> int | None:
        """Localhost port selected for the running DS4 process."""
        return self.process.port if self.process is not None else None

    @property
    def pid(self) -> int | None:
        """Subprocess PID, if DS4 has been started."""
        process = self.process.process if self.process is not None else None
        return process.pid if process is not None else None

    @property
    def is_running(self) -> bool:
        """Return True while the managed DS4 subprocess is alive."""
        return self.process is not None and self.process.is_running

    def _process_returncode(self) -> int | None:
        process = self.process.process if self.process is not None else None
        return process.returncode if process is not None else None

    def has_crashed(self) -> bool:
        """Return True when the managed DS4 subprocess exited unexpectedly."""
        process = self.process.process if self.process is not None else None
        return process is not None and process.returncode is not None

    async def start(self) -> None:
        """Start DS4 and wait for readiness."""
        if self.is_running:
            return
        mtp_path, mtp_kind, mtp_draft, mtp_margin = self._mtp_launch_args(
            self.model_settings
        )
        config = DS4LaunchConfig(
            model_id=self.model_id,
            gguf_path=self._model_path,
            settings=self.settings,
            base_path=self.base_path,
            context_tokens=self.context_tokens,
            auto_enable_ssd_streaming=self.auto_enable_ssd_streaming,
            mtp_path=mtp_path,
            mtp_kind=mtp_kind,
            mtp_draft=mtp_draft,
            mtp_margin=mtp_margin,
        )
        self.process = DS4ManagedProcess(config)
        try:
            await self.process.start()
        except Exception:
            self.process = None
            raise

    async def stop(self) -> None:
        """Stop the managed DS4 subprocess.

        Serialised with the lifecycle lock so that a concurrent request
        cannot observe a half-stopped backend.
        """
        async with self._lifecycle_lock:
            await self._stop_locked()

    async def _stop_locked(self) -> None:
        """Stop the subprocess without acquiring _lifecycle_lock.

        Callers that already hold _lifecycle_lock (e.g. ensure_min_context,
        restart_with_context) must use this instead of stop().
        """
        process = self.process
        if process is not None:
            await process.stop()
        self.process = None

    def effective_context_tokens(self) -> int:
        """Return the context token count DS4 will launch with."""
        return self.context_tokens or self.settings.get_auto_context_tokens()

    @staticmethod
    def _mtp_launch_args(
        model_settings: Any | None,
    ) -> tuple[
        Path | None,
        Literal["legacy_mtp", "dspark"] | None,
        int | None,
        float | None,
    ]:
        if model_settings is None or not getattr(model_settings, "ds4_mtp_enabled", False):
            return None, None, None, None
        mtp_path = getattr(model_settings, "ds4_mtp_path", None)
        if not mtp_path:
            return None, None, None, None
        resolved_path = Path(str(mtp_path)).expanduser().resolve()
        from ..ds4_gguf import detect_ds4_mtp_sidecar_kind

        mtp_kind = detect_ds4_mtp_sidecar_kind(resolved_path) or "legacy_mtp"
        return (
            resolved_path,
            mtp_kind,
            getattr(model_settings, "ds4_mtp_draft", None),
            getattr(model_settings, "ds4_mtp_margin", None),
        )

    async def ensure_min_context(self, min_tokens: int) -> bool:
        """Raise DS4 context for this loaded engine if needed.

        The new context persists while the engine stays loaded; it is not
        automatically lowered after the triggering request completes.
        Use :meth:`restart_with_context` to explicitly reset the context.
        """
        async with self._lifecycle_lock:
            if self.effective_context_tokens() >= min_tokens:
                return False
            if self.has_active_requests():
                raise DS4ProxyError(
                    "DS4 Think Max requires a backend restart with a larger context, "
                    "but the backend is currently serving another request; retry when idle"
                )
            was_running = self.is_running
            if was_running:
                await self._stop_locked()
            self.context_tokens = min_tokens
            if was_running:
                await self.start()
            return True

    async def restart_with_context(self, context_tokens: int | None) -> bool:
        """Restart this loaded DS4 engine with a new per-engine context override."""
        return await self.restart_with_launch_settings(
            context_tokens=context_tokens,
            model_settings=self.model_settings,
            reason="context change",
        )

    async def restart_with_launch_settings(
        self,
        *,
        context_tokens: int | None,
        model_settings: Any | None,
        reason: str = "launch setting change",
        force: bool = False,
    ) -> bool:
        """Restart this loaded DS4 engine with new launch-time settings."""
        async with self._lifecycle_lock:
            if (
                not force
                and self.context_tokens == context_tokens
                and self.model_settings is model_settings
                and self.is_running
            ):
                return False
            if self.has_active_requests():
                raise DS4ProxyError(
                    f"DS4 {reason} requires a backend restart, but the backend "
                    "is currently serving another request; retry when idle"
                )
            old_context_tokens = self.context_tokens
            old_model_settings = self.model_settings
            was_loaded = self.process is not None
            if was_loaded:
                await self._stop_locked()
                if self.has_active_requests():
                    self.context_tokens = old_context_tokens
                    self.model_settings = old_model_settings
                    await self.start()
                    raise DS4ProxyError(
                        f"DS4 {reason} requires a backend restart, but the "
                        "backend is currently serving another request; retry when idle"
                    )
            self.context_tokens = context_tokens
            self.model_settings = model_settings
            if was_loaded:
                await self.start()
                if self.has_active_requests():
                    await self._stop_locked()
                    self.context_tokens = old_context_tokens
                    self.model_settings = old_model_settings
                    await self.start()
                    raise DS4ProxyError(
                        f"DS4 {reason} requires a backend restart, but the "
                        "backend is currently serving another request; retry when idle"
                    )
            return was_loaded

    def _record_crashed_process(self) -> None:
        process = self.process
        process_obj = process.process if process is not None else None
        if process is None or process_obj is None or process_obj.returncode is None:
            return
        process_id = id(process_obj)
        if self._recorded_crash_process_id == process_id:
            return
        self._recorded_crash_process_id = process_id
        self._crash_count += 1
        self._last_crash_exit_code = process_obj.returncode
        self._last_crash_logs = process.recent_log_text()
        self._last_crash_monotonic_time = time.monotonic()

    async def _restart_stopped_locked(self) -> bool:
        """Restart a stopped/crashed DS4 subprocess while lifecycle-locked."""
        if self.is_running:
            return False
        if self.has_active_requests():
            raise DS4ProxyError(
                "DS4 backend process exited while another request is active; "
                "retry when idle"
            )
        had_process = self.process is not None
        if had_process:
            self._record_crashed_process()
            await self._stop_locked()
        try:
            await self.start()
        except Exception as exc:  # noqa: BLE001 - normalize restart failures
            raise DS4ProxyError(
                f"DS4 backend restart after crash failed: {exc}"
            ) from exc
        if had_process:
            self._restart_count += 1
        return had_process

    async def restart_if_crashed(self) -> bool:
        """Restart an idle DS4 subprocess that exited after being loaded."""
        async with self._lifecycle_lock:
            if not self.has_crashed():
                return False
            return await self._restart_stopped_locked()

    async def _ensure_running_for_request_locked(self) -> None:
        if self.is_running:
            return
        await self._restart_stopped_locked()

    def _increment_active_requests(self) -> None:
        with self._active_requests_lock:
            self._active_requests += 1

    async def begin_proxy_request_window(self) -> None:
        """Mark a DS4 request active before endpoint-specific proxy startup."""
        self._increment_active_requests()
        try:
            async with self._lifecycle_lock:
                pass
        except BaseException:
            self._decrement_active_requests()
            raise

    def end_proxy_request_window(self) -> None:
        """Release a request-start activity marker."""
        self._decrement_active_requests()

    def _decrement_active_requests(self) -> None:
        with self._active_requests_lock:
            if self._active_requests > 0:
                self._active_requests -= 1

    def _active_request_count(self) -> int:
        with self._active_requests_lock:
            return self._active_requests

    def has_active_requests(self) -> bool:
        """Return True while OMLX is proxying requests to DS4."""
        return self._active_request_count() > 0

    def _backend_url(self, path: str) -> str:
        if self.port is None or not self.is_running:
            raise DS4ProxyError("DS4 backend process is not running")
        return f"http://{DS4_HOST}:{self.port}{path}"

    @staticmethod
    def _response_headers(response: requests.Response) -> dict[str, str]:
        """Return response headers that are safe to reflect through OMLX."""
        excluded = {"connection", "transfer-encoding"}
        return {
            key: value
            for key, value in response.headers.items()
            if key.lower() not in excluded
        }

    def _proxy_json_request_blocking(
        self,
        path: str,
        body: dict[str, Any],
        *,
        stream: bool,
    ) -> tuple[requests.Session, requests.Response]:
        session = requests.Session()
        session.trust_env = False
        try:
            response = session.post(
                self._backend_url(path),
                json=body,
                stream=stream,
                timeout=(_DS4_CONNECT_TIMEOUT, _DS4_READ_TIMEOUT),
                headers={
                    "Content-Type": "application/json",
                    "Accept-Encoding": "identity",
                },
            )
        except requests.RequestException as exc:
            session.close()
            raise DS4ProxyError(f"DS4 backend request failed: {exc}") from exc
        return session, response

    @staticmethod
    def _raw_body(response: requests.Response) -> bytes:
        raw = getattr(response, "raw", None)
        if raw is not None and hasattr(raw, "read"):
            return raw.read(decode_content=False)
        return response.content

    def _proxy_json_response_blocking(
        self,
        path: str,
        body: dict[str, Any],
    ) -> DS4ProxyResponse:
        try:
            session, response = self._proxy_json_request_blocking(
                path,
                body,
                stream=True,
            )
            try:
                try:
                    raw_body = self._raw_body(response)
                except Exception as exc:  # noqa: BLE001 - normalize backend I/O failures
                    raise DS4ProxyError(
                        f"DS4 backend response read failed: {exc}"
                    ) from exc
                return DS4ProxyResponse(
                    status_code=response.status_code,
                    headers=self._response_headers(response),
                    body=raw_body,
                )
            finally:
                response.close()
                session.close()
        finally:
            self._decrement_active_requests()

    async def _proxy_json_endpoint(
        self,
        path: str,
        body: dict[str, Any],
    ) -> DS4ProxyResponse:
        async with self._lifecycle_lock:
            await self._ensure_running_for_request_locked()
            self._increment_active_requests()
            task = asyncio.create_task(
                asyncio.to_thread(
                    self._proxy_json_response_blocking,
                    path,
                    body,
                )
            )
        return await asyncio.shield(task)

    async def proxy_chat_completion(self, body: dict[str, Any]) -> DS4ProxyResponse:
        """Forward one non-streaming OpenAI chat completion request to DS4."""
        return await self._proxy_json_endpoint("/v1/chat/completions", body)

    async def proxy_completion(self, body: dict[str, Any]) -> DS4ProxyResponse:
        """Forward one non-streaming OpenAI text completion request to DS4."""
        return await self._proxy_json_endpoint("/v1/completions", body)

    async def proxy_response(self, body: dict[str, Any]) -> DS4ProxyResponse:
        """Forward one non-streaming OpenAI Responses API request to DS4."""
        return await self._proxy_json_endpoint("/v1/responses", body)

    async def proxy_anthropic_message(self, body: dict[str, Any]) -> DS4ProxyResponse:
        """Forward one non-streaming Anthropic Messages API request to DS4."""
        return await self._proxy_json_endpoint("/v1/messages", body)

    async def _open_json_stream(
        self,
        path: str,
        body: dict[str, Any],
    ) -> DS4StreamingProxyResponse:
        async with self._lifecycle_lock:
            await self._ensure_running_for_request_locked()
            self._increment_active_requests()
            task = asyncio.create_task(
                asyncio.to_thread(
                    self._proxy_json_request_blocking,
                    path,
                    body,
                    stream=True,
                )
            )
        claimed = False

        def _cleanup_unclaimed(done: asyncio.Task) -> None:
            nonlocal claimed
            if claimed:
                return
            try:
                session, response = done.result()
            except Exception:
                self._decrement_active_requests()
                return
            response.close()
            session.close()
            self._decrement_active_requests()

        try:
            session, response = await asyncio.shield(task)
            claimed = True
        except asyncio.CancelledError:
            task.add_done_callback(_cleanup_unclaimed)
            raise
        except Exception:
            claimed = True
            self._decrement_active_requests()
            raise

        def _decrement_active() -> None:
            self._decrement_active_requests()

        return DS4StreamingProxyResponse(
            status_code=response.status_code,
            headers=self._response_headers(response),
            response=response,
            on_close=_decrement_active,
            session=session,
        )

    async def open_chat_completion_stream(
        self, body: dict[str, Any]
    ) -> DS4StreamingProxyResponse:
        """Open a streaming OpenAI chat completion request to DS4."""
        return await self._open_json_stream("/v1/chat/completions", body)

    async def open_completion_stream(
        self, body: dict[str, Any]
    ) -> DS4StreamingProxyResponse:
        """Open a streaming OpenAI text completion request to DS4."""
        return await self._open_json_stream("/v1/completions", body)

    async def open_response_stream(
        self, body: dict[str, Any]
    ) -> DS4StreamingProxyResponse:
        """Open a streaming OpenAI Responses API request to DS4."""
        return await self._open_json_stream("/v1/responses", body)

    async def open_anthropic_message_stream(
        self, body: dict[str, Any]
    ) -> DS4StreamingProxyResponse:
        """Open a streaming Anthropic Messages API request to DS4."""
        return await self._open_json_stream("/v1/messages", body)

    def get_process_rss_bytes(self) -> int | None:
        """Return the DS4 subprocess RSS when psutil can observe it."""
        pid = self.pid
        if pid is None:
            return None
        try:
            import psutil

            return int(psutil.Process(pid).memory_info().rss)
        except Exception:  # noqa: BLE001 - status should be best-effort only
            return None

    def _latest_progress_from_logs(self) -> dict[str, Any] | None:
        process = self.process
        if process is None:
            return None
        for log_line in reversed(getattr(process, "logs", [])):
            progress = parse_ds4_progress_log_line(str(getattr(log_line, "text", "")))
            if progress is None:
                continue
            monotonic_time = getattr(log_line, "monotonic_time", None)
            if isinstance(monotonic_time, int | float):
                progress["last_activity_age_seconds"] = max(
                    0.0,
                    time.monotonic() - float(monotonic_time),
                )
            return progress
        return None

    def get_activity_snapshot(self) -> dict[str, Any]:
        """Return best-effort DS4 proxy progress for the Active Models UI."""
        active_requests = self._active_request_count()
        if active_requests <= 0:
            return {"active_requests": 0, "activities": []}

        progress = self._latest_progress_from_logs() or {}
        phase = progress.get("phase") or "proxying"
        detail = f"DS4 {phase}"
        percent = progress.get("percent")
        if isinstance(percent, int | float):
            detail = f"{detail} {percent:.1f}%"
        if active_requests > 1:
            detail = f"{detail} · {active_requests} req"

        activity: dict[str, Any] = {
            "request_id": f"ds4-{self.model_id}",
            "kind": "ds4_proxy",
            "detail": detail,
            "active_requests": active_requests,
        }
        if progress:
            activity.update(
                {
                    "elapsed_seconds": progress.get("elapsed_seconds"),
                    "last_activity_age_seconds": progress.get(
                        "last_activity_age_seconds"
                    ),
                    "current_tokens": progress.get("current_tokens"),
                    "total_tokens": progress.get("total_tokens"),
                    "token_count": progress.get("generated_tokens")
                    or progress.get("current_tokens"),
                    "tokens_per_second": progress.get("average_tokens_per_second"),
                    "chunk_tokens_per_second": progress.get(
                        "chunk_tokens_per_second"
                    ),
                    "ds4_phase": progress.get("phase"),
                    "ds4_phase_type": progress.get("phase_type"),
                }
            )
        return {"active_requests": active_requests, "activities": [activity]}

    def get_stats(self) -> dict[str, Any]:
        """Return DS4 lifecycle/status fields for admin/status endpoints."""
        if self.has_crashed():
            self._record_crashed_process()
        command = self.process.command if self.process is not None else None
        logs = self.process.recent_log_text() if self.process is not None else ""
        log_path = getattr(self.process, "log_path", None)
        active_requests = self._active_request_count()
        progress = self._latest_progress_from_logs() if active_requests > 0 else None
        return {
            "backend": "ds4",
            "host": DS4_HOST,
            "port": self.port,
            "pid": self.pid,
            "running": self.is_running,
            "crashed": self.has_crashed(),
            "exit_code": self._process_returncode(),
            "crash_count": self._crash_count,
            "restart_count": self._restart_count,
            "last_crash_exit_code": self._last_crash_exit_code,
            "last_crash_logs": self._last_crash_logs,
            "last_crash_monotonic_time": self._last_crash_monotonic_time,
            "log_path": str(log_path) if log_path is not None else None,
            "rss_bytes": self.get_process_rss_bytes(),
            "context_tokens": self.effective_context_tokens(),
            "active_requests": active_requests,
            "progress": progress,
            "command": command,
            "recent_logs": logs,
        }

    def get_cache_stats(self) -> dict[str, Any] | None:
        """Aggregate DS4 KV cache metrics from captured ds4-server logs.

        Counters are accumulated incrementally so events that fell out of the
        500-line ring buffer are not lost.  The cursor resets when the managed
        process object changes (restart, context switch).
        """
        process = self.process
        if process is None:
            return None
        process_key = id(process)
        log_total: int = getattr(process, "log_total", len(process.logs))
        logs = process.logs
        if process_key != self._kv_stats_process_key:
            # New process — reset counters; skip lines already pruned from ring.
            ring_start = log_total - len(logs)
            self._kv_stats_process_key = process_key
            self._kv_stats_cursor = ring_start
            self._kv_hits = 0
            self._kv_stores = 0
            self._kv_evictions = 0
            self._kv_load_failures = 0
            self._kv_skips = 0
            self._kv_tokens_stored = 0
            self._kv_tokens_restored = 0
            self._kv_bytes_stored = 0
            self._kv_last_store_ms = None
            self._kv_last_load_ms = None
        ring_start = log_total - len(logs)
        start_idx = max(0, self._kv_stats_cursor - ring_start)
        for log_line in logs[start_idx:]:
            text = getattr(log_line, "text", "")
            parsed = parse_ds4_kv_log_line(str(text))
            if parsed is None:
                continue
            event = parsed["event"]
            if event == "hit":
                self._kv_hits += 1
                self._kv_tokens_restored += parsed.get("text_tokens", 0)
                self._kv_last_load_ms = parsed.get("load_ms")
            elif event == "store":
                self._kv_stores += 1
                self._kv_tokens_stored += parsed.get("tokens", 0)
                self._kv_bytes_stored += int(parsed.get("size_mb", 0) * 1024 * 1024)
                self._kv_last_store_ms = parsed.get("save_ms")
            elif event == "evicted":
                self._kv_evictions += 1
            elif event == "load_failed":
                self._kv_load_failures += 1
            elif event == "skipped":
                self._kv_skips += 1
        self._kv_stats_cursor = log_total
        return {
            "hits": self._kv_hits,
            "stores": self._kv_stores,
            "evictions": self._kv_evictions,
            "load_failures": self._kv_load_failures,
            "skips": self._kv_skips,
            "tokens_stored": self._kv_tokens_stored,
            "tokens_restored": self._kv_tokens_restored,
            "bytes_stored": self._kv_bytes_stored,
            "last_store_ms": self._kv_last_store_ms,
            "last_load_ms": self._kv_last_load_ms,
        }

    def _protocol_not_implemented(self) -> RuntimeError:
        return RuntimeError(
            "DS4 backend lifecycle is available, but protocol forwarding has not "
            "been implemented yet"
        )

    async def generate(self, *args, **kwargs) -> GenerationOutput:
        """Text completions are proxied through the server route."""
        raise self._protocol_not_implemented()

    async def stream_generate(self, *args, **kwargs) -> AsyncIterator[GenerationOutput]:
        """Streaming completions are proxied through the server route."""
        raise self._protocol_not_implemented()
        yield  # pragma: no cover - keeps this method an async iterator

    async def chat(self, *args, **kwargs) -> GenerationOutput:
        """Chat completions are forwarded in a later DS4 protocol slice."""
        raise self._protocol_not_implemented()

    async def stream_chat(self, *args, **kwargs) -> AsyncIterator[GenerationOutput]:
        """Streaming chat is forwarded in a later DS4 protocol slice."""
        raise self._protocol_not_implemented()
        yield  # pragma: no cover - keeps this method an async iterator
