from __future__ import annotations

import asyncio
import http.client as http_client
import json
import os
import ssl
import sys
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from mitmproxy import ctx, http  # type: ignore  # noqa: E402

# A standalone Homebrew mitmdump embeds an isolated Python runtime and ignores
# PYTHONPATH. Source checkouts therefore need the repository root explicitly.
_package_root = str(Path(__file__).resolve().parents[2])
if _package_root not in sys.path:  # pragma: no cover - mitmdump execution path
    sys.path.insert(0, _package_root)

from omlx.codex_interceptor.privacy import (  # noqa: E402
    AuditRecorder,
    audit_header_bytes,
    is_cloud_auxiliary_inference_request,
)
from omlx.codex_interceptor.protocol import (  # noqa: E402
    LOOP_GUARD_MODES,
    SAFE_EVENT_FIELDS,
    SENSITIVE_HEADER_NAMES,
    SSELocalOutputSanitizer,
    SSEToolNameNormalizer,
    SSEUsageObserver,
    adapt_client_tool_call_sse,
    adapt_compaction_response,
    adapt_compaction_sse_response,
    adapt_gpt_56_pro_request,
    adapt_model_catalog_for_local_tools,
    adapt_textual_tool_call_sse,
    add_gpt_56_pro_catalog_entry,
    apply_loop_guard,
    canonical_digest,
    compaction_response_sse,
    detect_churn_run,
    detect_tool_call_loop,
    expose_client_tools_for_local_model,
    extract_advertised_tools,
    extract_namespace_tools,
    flatten_namespace_tools_for_local_model,
    input_type_counts,
    is_compaction_request,
    is_compaction_trigger_request,
    is_intercepted_request,
    is_intercepted_websocket,
    local_incompatible_input_item_count,
    non_array_content_item_count,
    normalize_client_tool_calls,
    normalize_invalid_local_reasoning_items,
    normalize_legacy_compaction_item_ids,
    normalize_response_tool_names,
    opaque_compaction_item_count,
    repair_local_request,
    response_is_empty,
    responses_request_model,
    sanitize_local_response_items,
    select_lowest_visible_codex_model,
    should_route_responses_locally,
    source_fingerprint,
    summarize_responses_request,
    synthetic_compaction_response,
    transform_compaction_request,
    transform_responses_request,
    typeless_input_item_count,
    usage_fields,
    usage_from_response_event,
    validate_local_request,
)
from omlx.codex_interceptor.websocket import (  # noqa: E402
    ResponsesWebSocketState,
    SSEEventDecoder,
    decode_json_events,
)

MAX_STATUS_BYTES = 16 * 1024 * 1024
# Records are per-request milestones, so a few seconds of slack before the
# size check re-runs is harmless and avoids a stat() syscall per record.
STATUS_SIZE_CHECK_SECONDS = 5.0
# A local request that already failed twice will fail the same way a third time,
# and each attempt is a full local inference.
LOCAL_FAILURE_ATTEMPTS = 2
LOCAL_FAILURE_TTL_SECONDS = 300
# A cancelled or reset stream makes Codex re-send a turn the local model has
# already answered. Replaying the stored answer costs nothing; re-running it costs
# another full inference.
LOCAL_REPLAY_ENTRIES = 6
LOCAL_REPLAY_TTL_SECONDS = 300
LOCAL_REPLAY_MAX_BYTES = 4 * 1024 * 1024
PREFIX_PREFILL_ENTRIES = 8
PREFIX_PREFILL_TTL_SECONDS = 15 * 60
PREFIX_PREFILL_MIN_BYTES = 8 * 1024
PREFIX_PREFILL_TIMEOUT_SECONDS = float(
    os.environ.get("OMLX_CODEX_INTERCEPTOR_PREFIX_PREFILL_TIMEOUT", "180")
)
MODEL_RESIDENCY_INTERVAL_SECONDS = float(
    os.environ.get("OMLX_CODEX_INTERCEPTOR_RESIDENCY_INTERVAL", str(12 * 60))
)
MODEL_RESIDENCY_POLL_SECONDS = min(
    30.0,
    max(1.0, MODEL_RESIDENCY_INTERVAL_SECONDS / 4),
)
# Applied per socket read, so keepalive comments from a slow local upstream keep
# resetting it. Abandoning a live stream costs a full inference: the upstream
# finishes and bills the work anyway, and Codex then retries the same turn.
LOCAL_UPSTREAM_READ_TIMEOUT_SECONDS = float(
    os.environ.get("OMLX_CODEX_INTERCEPTOR_UPSTREAM_READ_TIMEOUT", "600")
)
REMOTE_RESPONSE_TERMINAL_TYPES = {
    "response.completed",
    "response.failed",
    "response.incomplete",
    "response.cancelled",
    "error",
}


class LocalUpstreamHTTPError(RuntimeError):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def _is_prompt_size_rejection(status: Any) -> bool:
    # A 413 is rejected before any local inference runs, so a retry costs
    # nothing. Counting it toward LOCAL_FAILURE_ATTEMPTS hides the real
    # prompt-size error behind a 422 lockout for no saving.
    return status == 413


class PersistentUpstreamPool:
    """Small HTTP/1.1 connection pool for one private Responses endpoint."""

    def __init__(self, url: str, *, max_idle: int = 2) -> None:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(f"invalid local upstream URL: {url}")
        self.url = url
        self.scheme = parsed.scheme
        self.host = parsed.hostname
        self.port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self.target = parsed.path or "/"
        if parsed.query:
            self.target += f"?{parsed.query}"
        self.max_idle = max(1, max_idle)
        self._idle: list[http_client.HTTPConnection] = []
        self._lock = threading.Lock()

    def _new(self, timeout: float) -> http_client.HTTPConnection:
        if self.scheme == "https":
            return http_client.HTTPSConnection(
                self.host,
                self.port,
                timeout=timeout,
                context=ssl.create_default_context(),
            )
        return http_client.HTTPConnection(self.host, self.port, timeout=timeout)

    def _acquire(self, timeout: float) -> tuple[http_client.HTTPConnection, bool]:
        with self._lock:
            if self._idle:
                connection = self._idle.pop()
                connection.timeout = timeout
                return connection, True
        return self._new(timeout), False

    def _discard(self, connection: http_client.HTTPConnection) -> None:
        with suppress(OSError):
            connection.close()

    def _release(self, connection: http_client.HTTPConnection) -> None:
        with self._lock:
            if len(self._idle) < self.max_idle:
                self._idle.append(connection)
                return
        self._discard(connection)

    @contextmanager
    def stream(
        self,
        body: bytes,
        headers: dict[str, str],
        *,
        timeout: float,
    ):
        response = None
        connection = None
        reused = False
        connect_ms = 0
        for attempt in range(2):
            connection, reused = self._acquire(timeout)
            try:
                if connection.sock is None:
                    connect_started = time.monotonic()
                    connection.connect()
                    connect_ms = max(
                        0, round((time.monotonic() - connect_started) * 1000)
                    )
                elif connection.sock is not None:
                    connection.sock.settimeout(timeout)
                connection.request("POST", self.target, body=body, headers=headers)
                response = connection.getresponse()
                break
            except (
                BrokenPipeError,
                ConnectionError,
                http_client.HTTPException,
                OSError,
            ):
                self._discard(connection)
                connection = None
                if not reused or attempt:
                    raise
        if connection is None or response is None:
            raise ConnectionError("local upstream connection unavailable")
        try:
            yield (
                response,
                {
                    "connection_reused": reused,
                    "connect_ms": connect_ms,
                },
            )
        finally:
            if response.isclosed() and not response.will_close:
                self._release(connection)
            else:
                self._discard(connection)

    def close(self) -> None:
        with self._lock:
            idle, self._idle = self._idle, []
        for connection in idle:
            self._discard(connection)


class CodexLocalInterceptor:
    def __init__(self) -> None:
        self.upstream_url = _required_env("OMLX_CODEX_INTERCEPTOR_UPSTREAM_URL")
        self.model = _required_env("OMLX_CODEX_INTERCEPTOR_MODEL")
        self.local_slot = os.environ.get("OMLX_CODEX_INTERCEPTOR_LOCAL_SLOT") or None
        self.local_slot_auto = os.environ.get(
            "OMLX_CODEX_INTERCEPTOR_LOCAL_SLOT_AUTO", "0"
        ) in {"1", "true", "True"}
        self.local_label = os.environ.get(
            "OMLX_CODEX_INTERCEPTOR_LOCAL_LABEL", "Local · oMLX"
        )
        self.local_server = os.environ.get(
            "OMLX_CODEX_INTERCEPTOR_LOCAL_SERVER", "oMLX"
        )
        self.api_key = os.environ.get("OMLX_CODEX_INTERCEPTOR_UPSTREAM_API_KEY")
        self.auth_header = os.environ.get(
            "OMLX_CODEX_INTERCEPTOR_AUTH_HEADER", "1"
        ) not in {"0", "false", "False"}
        self.status_path = Path(
            _required_env("OMLX_CODEX_INTERCEPTOR_STATUS_PATH")
        ).expanduser()
        self.session_id = _required_env("OMLX_CODEX_INTERCEPTOR_SESSION_ID")
        self.websocket_bridge = os.environ.get(
            "OMLX_CODEX_INTERCEPTOR_WEBSOCKET_BRIDGE", "1"
        ) not in {"0", "false", "False"}
        self.websocket_upstream_override = os.environ.get(
            "OMLX_CODEX_INTERCEPTOR_WEBSOCKET_UPSTREAM_URL"
        )
        self.gpt_56_pro_enabled = os.environ.get(
            "OMLX_CODEX_INTERCEPTOR_GPT56_PRO", "0"
        ) in {"1", "true", "True"}
        # A hosted compaction returns ciphertext the local model cannot read, so
        # compaction is summarized locally regardless of the model it names.
        self.local_compaction = os.environ.get(
            "OMLX_CODEX_INTERCEPTOR_LOCAL_COMPACTION", "1"
        ) not in {"0", "false", "False"}
        mode = os.environ.get("OMLX_CODEX_INTERCEPTOR_LOOP_GUARD", "guard")
        self.loop_guard_mode = mode if mode in LOOP_GUARD_MODES else "guard"
        # Advertised to Codex for the local slot so it auto-compacts against the
        # local model's real window instead of the hosted slot's.
        self.local_context_window = _positive_int_env(
            "OMLX_CODEX_INTERCEPTOR_LOCAL_CONTEXT_WINDOW"
        )
        self.websocket_states: dict[str, ResponsesWebSocketState] = {}
        self.websocket_tasks: set[asyncio.Task] = set()
        capabilities_path = os.environ.get("OMLX_CODEX_INTERCEPTOR_CAPABILITIES_PATH")
        self.capabilities_path = (
            Path(capabilities_path).expanduser() if capabilities_path else None
        )
        self._capabilities_mtime_ns: int | None = None
        self._native_function_calls = False
        self._native_capability_reported = False
        self.prefix_prefills: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self.prefix_prefill_enabled = os.environ.get(
            "OMLX_CODEX_INTERCEPTOR_PREFIX_PREFILL", "1"
        ) not in {"0", "false", "False"}
        prefix_cache_path = os.environ.get("OMLX_CODEX_INTERCEPTOR_PREFIX_CACHE_PATH")
        self.prefix_cache_path = (
            Path(prefix_cache_path).expanduser() if prefix_cache_path else None
        )
        self._load_prefix_prefills()
        self.residency_keepalive_enabled = os.environ.get(
            "OMLX_CODEX_INTERCEPTOR_RESIDENCY_KEEPALIVE", "1"
        ) not in {"0", "false", "False"}
        self.last_local_activity = time.monotonic()
        self.upstream_pool = PersistentUpstreamPool(self.upstream_url)
        audit_dir = os.environ.get("OMLX_CODEX_INTERCEPTOR_CLOUD_AUDIT_DIR")
        self.cloud_audit_include_credentials = os.environ.get(
            "OMLX_CODEX_INTERCEPTOR_CLOUD_AUDIT_CREDENTIALS", "0"
        ) in {"1", "true", "True"}
        self.cloud_audit = (
            AuditRecorder(
                Path(audit_dir),
                self.session_id,
                include_credentials=self.cloud_audit_include_credentials,
            )
            if audit_dir
            else None
        )
        self._cloud_audit_failed = False
        self._status_size_checked_at = 0.0
        self.local_failures: dict[str, tuple[float, int]] = {}
        self.local_replays: OrderedDict[str, tuple[float, str, bytes]] = OrderedDict()
        self.local_event_replays: OrderedDict[str, tuple[float, list]] = OrderedDict()
        self.code_fingerprint = source_fingerprint(
            [
                Path(__file__).resolve(),
                Path(__file__).resolve().with_name("protocol.py"),
            ]
        )
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.status_path.touch(mode=0o600, exist_ok=True)
        os.chmod(self.status_path, 0o600)
        self._record(
            "interceptor_ready",
            local_model=self.model,
            local_server=self.local_server,
            local_slot=self.local_slot,
            code_fingerprint=self.code_fingerprint,
            advertised_context_window=self.local_context_window,
            prefix_prefill_status=("cached" if self.prefix_prefills else "cold"),
            residency_status=(
                "monitoring" if self.residency_keepalive_enabled else "disabled"
            ),
        )

    async def running(self) -> None:
        if not self.residency_keepalive_enabled:
            return
        task = asyncio.create_task(self._residency_loop())
        self.websocket_tasks.add(task)
        task.add_done_callback(self.websocket_tasks.discard)

    def done(self) -> None:
        for task in tuple(self.websocket_tasks):
            task.cancel()
        self.upstream_pool.close()

    def _pool(self) -> PersistentUpstreamPool:
        if self.upstream_pool.url != self.upstream_url:
            self.upstream_pool.close()
            self.upstream_pool = PersistentUpstreamPool(self.upstream_url)
        return self.upstream_pool

    def _load_prefix_prefills(self) -> None:
        self.prefix_prefills = OrderedDict()
        path = self.prefix_cache_path
        if path is None:
            return
        try:
            decoded = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return
        if not isinstance(decoded, dict):
            return
        if decoded.get("model") != self.model:
            return
        entries = decoded.get("entries")
        if not isinstance(entries, list):
            return
        now = time.time()
        for entry in entries[-PREFIX_PREFILL_ENTRIES:]:
            if not isinstance(entry, dict):
                continue
            digest = entry.get("digest")
            expires_at = entry.get("expires_at")
            if (
                isinstance(digest, str)
                and isinstance(expires_at, (int, float))
                and expires_at > now
            ):
                self.prefix_prefills[digest] = (float(expires_at), "ok")

    def _save_prefix_prefills(self) -> None:
        path = self.prefix_cache_path
        if path is None:
            return
        now = time.time()
        entries = [
            {"digest": digest, "expires_at": expires_at}
            for digest, (expires_at, status) in self.prefix_prefills.items()
            if status == "ok" and expires_at > now
        ][-PREFIX_PREFILL_ENTRIES:]
        try:
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(path.parent, 0o700)
            temporary = path.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "model": self.model,
                        "entries": entries,
                        "updated_at": time.time(),
                    },
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
            os.chmod(temporary, 0o600)
            temporary.replace(path)
        except OSError:
            return

    async def _residency_loop(self) -> None:
        while True:
            await asyncio.sleep(MODEL_RESIDENCY_POLL_SECONDS)
            idle_seconds = time.monotonic() - self.last_local_activity
            if idle_seconds < MODEL_RESIDENCY_INTERVAL_SECONDS:
                continue
            await self._run_residency_keepalive()

    async def _run_residency_keepalive(self) -> None:
        started = time.monotonic()
        body = json.dumps(
            {
                "model": self.model,
                "stream": True,
                "input": "Reply with OK.",
                "max_output_tokens": 1,
            },
            separators=(",", ":"),
        ).encode("utf-8")

        def worker() -> tuple[int, dict[str, Any]]:
            headers = self._local_headers("omlx-codex-interceptor-residency/0.1")
            with self._pool().stream(
                body,
                headers,
                timeout=PREFIX_PREFILL_TIMEOUT_SECONDS,
            ) as (response, connection_metrics):
                raw = response.read(512 * 1024)
                if not 200 <= response.status < 300:
                    raise LocalUpstreamHTTPError(
                        response.status,
                        _upstream_error_message(raw)
                        or f"local upstream returned HTTP {response.status}",
                    )
                return response.status, connection_metrics

        try:
            status, connection_metrics = await asyncio.to_thread(worker)
            self.last_local_activity = time.monotonic()
            self._record(
                "residency_keepalive_completed",
                local_model=self.model,
                local_server=self.local_server,
                status=status,
                duration_ms=round((time.monotonic() - started) * 1000),
                residency_status="resident",
                **connection_metrics,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # A keepalive is an optimization and never interrupts user traffic.
            self.last_local_activity = time.monotonic()
            self._record(
                "residency_keepalive_failed",
                local_model=self.model,
                local_server=self.local_server,
                duration_ms=round((time.monotonic() - started) * 1000),
                residency_status="unknown",
                error=type(exc).__name__,
                error_detail=_safe_preview(str(exc)),
            )

    def _local_headers(self, user_agent: str) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream, application/json",
            "Accept-Encoding": "identity",
            "User-Agent": user_agent,
        }
        if self.api_key and self.auth_header:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def request(self, flow: http.HTTPFlow) -> None:
        request = flow.request
        original_host = request.host
        original_path = request.path
        safe_path = original_path.split("?", 1)[0][:512] or "/"
        flow.metadata["harness_request_started"] = time.monotonic()
        flow.metadata["harness_original_host"] = original_host
        flow.metadata["harness_original_path"] = safe_path
        flow.metadata["harness_original_full_path"] = original_path[:4096]
        flow.metadata["harness_local_routed_at"] = time.monotonic()
        flow.metadata["harness_original_method"] = request.method.upper()
        if (
            is_intercepted_websocket(request.method, original_host, original_path)
            and request.headers.get("Upgrade", "").lower() == "websocket"
        ):
            if not self.websocket_bridge:
                flow.response = http.Response.make(
                    404,
                    b"Responses WebSocket unavailable; use Responses over HTTP",
                    {"Content-Type": "text/plain", "Cache-Control": "no-store"},
                )
                self._record(
                    "websocket_forced_to_http",
                    original_host=original_host,
                    original_path=safe_path,
                    method=request.method.upper(),
                    local_model=self.model,
                    status=404,
                    retryable=False,
                )
                return
            # Keep the authenticated OpenAI WebSocket intact. Local response.create
            # frames are dropped and answered from oMLX in websocket_message();
            # frames for every other Codex model pass through unchanged.
            flow.metadata["harness_responses_websocket"] = True
            if self.websocket_upstream_override:
                request.url = self.websocket_upstream_override
            return
        if is_cloud_auxiliary_inference_request(
            request.method, original_host, original_path
        ):
            flow.metadata["harness_passthrough"] = True
            flow.metadata["harness_remote_inference"] = True
            flow.metadata["harness_remote_auxiliary_inference"] = True
            self._audit_cloud_http_request(
                flow,
                model=None,
                inference_kind="image_generation",
            )
            self._record(
                "remote_inference_routed",
                original_host=original_host,
                original_path=safe_path,
                method=request.method.upper(),
                requested_model=None,
                local_slot=self.local_slot,
                inference_kind="image_generation",
            )
            return
        if not is_intercepted_request(request.method, original_host, original_path):
            flow.metadata["harness_passthrough"] = True
            flow.metadata["harness_model_catalog"] = (
                request.method.upper() == "GET"
                and original_host.lower() == "chatgpt.com"
                and safe_path == "/backend-api/codex/models"
            )
            self._record(
                "request_passthrough",
                original_host=original_host,
                original_path=safe_path,
                method=request.method.upper(),
            )
            return

        try:
            raw = request.get_content(strict=False)
            payload = json.loads(raw)
            payload, repaired_compaction_ids = normalize_legacy_compaction_item_ids(
                payload
            )
            payload, repaired_reasoning_items = normalize_invalid_local_reasoning_items(
                payload
            )
            if repaired_compaction_ids or repaired_reasoning_items:
                request.decode(strict=False)
                request.set_content(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
            if repaired_compaction_ids:
                self._record(
                    "legacy_compaction_id_repaired",
                    repaired_compaction_id_count=repaired_compaction_ids,
                    transport="http",
                )
            if repaired_reasoning_items:
                self._record(
                    "local_reasoning_item_repaired",
                    repaired_local_reasoning_item_count=repaired_reasoning_items,
                    transport="http",
                )
            requested_model = responses_request_model(payload)
            if self.local_slot is not None and requested_model is None:
                raise ValueError("Responses request is missing a valid model")
            route_locally = should_route_responses_locally(
                payload, self.local_slot, local_compaction=self.local_compaction
            )
            forced_local_compaction = (
                route_locally
                and self.local_slot is not None
                and requested_model != self.local_slot
            )
            if not route_locally:
                outbound, pro_mode = adapt_gpt_56_pro_request(payload)
                if pro_mode:
                    request.decode(strict=False)
                    request.set_content(_json_bytes(outbound))
                effective_model = responses_request_model(outbound)
                flow.metadata["harness_remote_inference"] = True
                flow.metadata["harness_requested_model"] = requested_model
                flow.metadata["harness_effective_model"] = effective_model
                flow.metadata["harness_pro_mode"] = pro_mode
                self._audit_cloud_http_request(
                    flow,
                    model=effective_model,
                    inference_kind="responses_pro" if pro_mode else "responses",
                )
                self._record(
                    "remote_inference_routed",
                    original_host=original_host,
                    original_path=safe_path,
                    method=request.method.upper(),
                    requested_model=requested_model,
                    effective_model=effective_model,
                    pro_mode=pro_mode,
                    local_slot=self.local_slot,
                )
                return
            request.decode(strict=False)
            request_summary = summarize_responses_request(payload, len(raw))
            compaction_request = is_compaction_request(payload)
            client_stream = payload.get("stream") is True
            if is_compaction_trigger_request(payload):
                transformed, tool_names = transform_compaction_request(
                    payload, self.model
                )
            else:
                transformed, tool_names = transform_responses_request(
                    payload, self.model
                )
            advertised_tool_names, advertised_tool_types = extract_advertised_tools(
                transformed
            )
            if compaction_request:
                transformed["stream"] = False
                transformed.pop("stream_options", None)
            namespace_tools = extract_namespace_tools(transformed)
            transformed, flattened_names = flatten_namespace_tools_for_local_model(
                transformed
            )
            transformed, client_tool_mappings, client_tool_names = (
                expose_client_tools_for_local_model(transformed)
            )
            local_tool_names, _ = extract_advertised_tools(transformed)
            transformed, loop_action, loop_detection = self._guard_loop(
                transformed, compaction_request
            )
            transformed, repaired_rules = repair_local_request(transformed)
        except Exception as exc:
            flow.response = http.Response.make(
                400,
                json.dumps(
                    {
                        "error": {
                            "message": f"local interceptor rejected request: {exc}",
                            "type": "invalid_request_error",
                        }
                    }
                ),
                {"Content-Type": "application/json"},
            )
            self._record("request_rejected", error=type(exc).__name__)
            return

        request_digest = canonical_digest(transformed)
        flow.metadata["harness_request_digest"] = request_digest
        replay = self._local_replay(request_digest)
        if replay is not None:
            content_type, body = replay
            flow.response = http.Response.make(
                200,
                body,
                {"Content-Type": content_type, "Cache-Control": "no-store"},
            )
            self._record(
                "local_response_replayed",
                local_model=self.model,
                requested_model=requested_model,
                local_slot=self.local_slot,
                status=200,
                request_bytes=len(body),
            )
            return
        if self._local_failures_exhausted(request_digest):
            # The same request already failed twice. Re-running it costs another
            # full local inference (60-200s on a large prompt) to reach the same
            # rejection, which is exactly how eight minutes went missing.
            flow.response = http.Response.make(
                422,
                json.dumps(
                    {
                        "error": {
                            "message": (
                                "local interceptor refused a request that already "
                                f"failed {LOCAL_FAILURE_ATTEMPTS} times"
                            ),
                            "type": "invalid_request_error",
                        }
                    }
                ),
                {"Content-Type": "application/json", "Cache-Control": "no-store"},
            )
            self._record(
                "local_request_refused",
                local_model=self.model,
                requested_model=requested_model,
                local_slot=self.local_slot,
                status=422,
            )
            return
        if repaired_rules:
            self._record(
                "local_request_repaired",
                local_model=self.model,
                requested_model=requested_model,
                local_slot=self.local_slot,
                transport="http",
                local_request_rules_broken=sorted(repaired_rules),
            )
        broken_rules = validate_local_request(transformed)
        if broken_rules:
            flow.response = http.Response.make(
                422,
                json.dumps(
                    {
                        "error": {
                            "message": (
                                "local interceptor refused an invalid request: "
                                + ", ".join(sorted(broken_rules))
                            ),
                            "type": "invalid_request_error",
                        }
                    }
                ),
                {"Content-Type": "application/json", "Cache-Control": "no-store"},
            )
            self._record(
                "local_request_invalid",
                local_model=self.model,
                requested_model=requested_model,
                local_slot=self.local_slot,
                local_request_rules_broken=sorted(broken_rules),
            )
            return
        flow.metadata["harness_intercepted"] = True
        flow.metadata["harness_requested_model"] = requested_model
        flow.metadata["harness_registered_tools"] = sorted(tool_names)
        flow.metadata["harness_emittable_tools"] = sorted(
            tool_names | flattened_names | client_tool_names
        )
        flow.metadata["harness_namespace_tools"] = namespace_tools
        flow.metadata["harness_client_tool_mappings"] = client_tool_mappings
        flow.metadata["harness_compaction_request"] = compaction_request
        if compaction_request:
            # Kept so a failed compaction can still be summarized from the real
            # conversation instead of a placeholder.
            flow.metadata["harness_compaction_payload"] = payload
        flow.metadata["harness_client_stream"] = client_stream
        flow.metadata["harness_original_host"] = original_host
        flow.metadata["harness_original_path"] = safe_path

        request.url = self.upstream_url
        for name in list(request.headers.keys()):
            if name.lower() in SENSITIVE_HEADER_NAMES or name.lower().startswith(
                "x-openai-"
            ):
                request.headers.pop(name, None)
        request.headers.pop("Content-Encoding", None)
        request.headers.pop("Content-Length", None)
        request.headers["Content-Type"] = "application/json"
        request.headers["Accept"] = (
            "application/json"
            if compaction_request
            else "text/event-stream, application/json"
        )
        request.headers["Accept-Encoding"] = "identity"
        request.headers["User-Agent"] = "omlx-codex-interceptor/0.1"
        if self.api_key and self.auth_header:
            request.headers["Authorization"] = f"Bearer {self.api_key}"
        request.set_content(
            json.dumps(transformed, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        self._record(
            "inference_routed",
            original_host=original_host,
            original_path=safe_path,
            method=request.method.upper(),
            local_model=self.model,
            local_server=self.local_server,
            requested_model=requested_model,
            local_slot=self.local_slot,
            tool_count=len(tool_names | flattened_names | client_tool_names),
            tool_names=sorted(local_tool_names),
            advertised_tool_names=sorted(advertised_tool_names),
            advertised_tool_types=sorted(advertised_tool_types),
            typeless_input_item_count=typeless_input_item_count(
                transformed.get("input")
            ),
            non_array_content_item_count=non_array_content_item_count(
                transformed.get("input")
            ),
            input_type_counts=input_type_counts(transformed.get("input")),
            coerced_input_item_count=local_incompatible_input_item_count(
                payload.get("input")
            ),
            opaque_compaction_item_count=opaque_compaction_item_count(
                payload.get("input")
            ),
            forced_local_compaction=forced_local_compaction,
            **request_summary,
        )
        if loop_detection is not None:
            self._record(
                "doom_loop_detected",
                local_model=self.model,
                requested_model=requested_model,
                local_slot=self.local_slot,
                transport="http",
                **_loop_guard_fields(loop_detection, loop_action),
            )

    def _guard_loop(
        self,
        transformed: dict[str, Any],
        compaction: bool,
    ) -> tuple[dict[str, Any], str | None, dict[str, Any] | None]:
        """Detect and optionally break a repeated tool call on a local turn.

        Compaction is exempt: it repeats no calls and its prompt must stay
        exactly as built.
        """
        if compaction or self.loop_guard_mode == "off":
            return transformed, None, None
        items = transformed.get("input")
        detection = detect_tool_call_loop(items) or detect_churn_run(items)
        if detection is None:
            return transformed, None, None
        if self.loop_guard_mode == "observe":
            return transformed, None, detection
        guarded, action = apply_loop_guard(transformed, detection)
        return guarded, action, detection

    def _native_tool_streaming_enabled(self) -> bool:
        override = os.environ.get("OMLX_CODEX_INTERCEPTOR_NATIVE_TOOL_STREAMING")
        if override is not None:
            return override in {"1", "true", "True"}
        path = self.capabilities_path
        if path is None:
            return False
        try:
            mtime_ns = path.stat().st_mtime_ns
            if mtime_ns != self._capabilities_mtime_ns:
                decoded = json.loads(path.read_text(encoding="utf-8"))
                capabilities = (
                    decoded.get("capabilities") if isinstance(decoded, dict) else None
                )
                self._native_function_calls = bool(
                    isinstance(decoded, dict)
                    and decoded.get("model") == self.model
                    and isinstance(capabilities, dict)
                    and capabilities.get("native_function_call") is True
                )
                self._capabilities_mtime_ns = mtime_ns
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return self._native_function_calls
        if self._native_function_calls and not self._native_capability_reported:
            self._native_capability_reported = True
            self._record(
                "native_tool_streaming_enabled",
                local_model=self.model,
                local_server=self.local_server,
            )
        return self._native_function_calls

    def _record_performance_warnings(
        self,
        *,
        request_bytes: Any,
        input_tokens: Any,
        cache_hit_percent: Any,
        transform_ms: Any,
        first_byte_ms: Any,
        first_visible_ms: Any,
        connect_ms: Any,
        connection_reused: Any,
    ) -> None:
        warnings: list[tuple[str, float | int, float | int]] = []
        if isinstance(request_bytes, int) and request_bytes >= 128 * 1024:
            warnings.append(("oversized_request", request_bytes, 128 * 1024))
        if isinstance(transform_ms, int) and transform_ms >= 100:
            warnings.append(("slow_transform", transform_ms, 100))
        if (
            connection_reused is False
            and isinstance(connect_ms, int)
            and connect_ms >= 250
        ):
            warnings.append(("slow_connection", connect_ms, 250))
        if isinstance(first_byte_ms, int) and first_byte_ms >= 5000:
            warnings.append(("slow_prefill", first_byte_ms, 5000))
        if (
            isinstance(first_visible_ms, int)
            and isinstance(first_byte_ms, int)
            and first_visible_ms - first_byte_ms >= 1500
        ):
            warnings.append(
                ("delayed_visible_output", first_visible_ms - first_byte_ms, 1500)
            )
        if (
            isinstance(input_tokens, int)
            and input_tokens >= 8192
            and isinstance(cache_hit_percent, (int, float))
            and cache_hit_percent < 25
        ):
            warnings.append(("low_prefix_cache_hit", cache_hit_percent, 25))
        for warning_code, warning_value, warning_threshold in warnings:
            self._record(
                "performance_warning",
                local_model=self.model,
                local_server=self.local_server,
                warning_code=warning_code,
                warning_value=warning_value,
                warning_threshold=warning_threshold,
            )

    def _schedule_prefix_prefill(
        self,
        payload: dict[str, Any] | None = None,
        *,
        requested_model: str | None,
        transformed: dict[str, Any] | None = None,
        digest: str | None = None,
    ) -> None:
        if not self.prefix_prefill_enabled:
            return
        try:
            if transformed is None:
                transformed, _ = transform_responses_request(payload, self.model)
                transformed, _ = flatten_namespace_tools_for_local_model(transformed)
                transformed, _, _ = expose_client_tools_for_local_model(transformed)
                transformed, _ = repair_local_request(transformed)
                broken_rules = validate_local_request(transformed)
                if broken_rules:
                    return
            if digest is None:
                # Keyed like the serve path's request digest, so a prewarm and
                # the turn it precedes share one prefill entry.
                digest = canonical_digest(transformed)
        except (TypeError, ValueError):
            return

        now = time.time()
        self.prefix_prefills = OrderedDict(
            (key, entry)
            for key, entry in self.prefix_prefills.items()
            if entry[0] > now
        )
        existing = self.prefix_prefills.get(digest)
        if existing is not None:
            self.prefix_prefills.move_to_end(digest)
            self._record(
                "prefix_prefill_deduplicated",
                local_model=self.model,
                local_server=self.local_server,
                requested_model=requested_model,
                prefix_prefill_status=existing[1],
            )
            return
        try:
            body = json.dumps(
                {**transformed, "stream": True, "max_output_tokens": 1},
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError):
            return
        if len(body) < PREFIX_PREFILL_MIN_BYTES:
            return

        self.prefix_prefills[digest] = (
            now + PREFIX_PREFILL_TTL_SECONDS,
            "running",
        )
        while len(self.prefix_prefills) > PREFIX_PREFILL_ENTRIES:
            self.prefix_prefills.popitem(last=False)
        self._record(
            "prefix_prefill_started",
            local_model=self.model,
            local_server=self.local_server,
            requested_model=requested_model,
            request_bytes=len(body),
        )
        task = asyncio.create_task(
            self._run_prefix_prefill(
                digest,
                body,
                requested_model=requested_model,
            )
        )
        self.websocket_tasks.add(task)
        task.add_done_callback(self.websocket_tasks.discard)

    async def _run_prefix_prefill(
        self,
        digest: str,
        body: bytes,
        *,
        requested_model: str | None,
    ) -> None:
        started = time.monotonic()

        def worker() -> tuple[int, int, dict[str, Any]]:
            headers = self._local_headers("omlx-codex-interceptor-prefix-prefill/0.2")
            first_byte_ms = 0
            with self._pool().stream(
                body,
                headers,
                timeout=PREFIX_PREFILL_TIMEOUT_SECONDS,
            ) as (response, connection_metrics):
                while True:
                    chunk = response.readline()
                    if not chunk:
                        break
                    if not first_byte_ms and not _is_sse_keepalive(chunk):
                        first_byte_ms = max(
                            1,
                            round((time.monotonic() - started) * 1000),
                        )
                if not 200 <= response.status < 300:
                    raise LocalUpstreamHTTPError(
                        response.status,
                        f"local upstream returned HTTP {response.status}",
                    )
                return response.status, first_byte_ms, connection_metrics

        try:
            status, first_byte_ms, connection_metrics = await asyncio.to_thread(worker)
            duration_ms = round((time.monotonic() - started) * 1000)
            self.prefix_prefills[digest] = (
                time.time() + PREFIX_PREFILL_TTL_SECONDS,
                "ok",
            )
            self._save_prefix_prefills()
            self.last_local_activity = time.monotonic()
            self._record(
                "prefix_prefill_completed",
                local_model=self.model,
                local_server=self.local_server,
                requested_model=requested_model,
                status=status,
                request_bytes=len(body),
                duration_ms=duration_ms,
                first_byte_ms=first_byte_ms,
                **connection_metrics,
            )
        except Exception as exc:
            self.prefix_prefills[digest] = (
                time.time() + 30,
                "failed",
            )
            self._save_prefix_prefills()
            self._record(
                "prefix_prefill_failed",
                local_model=self.model,
                local_server=self.local_server,
                requested_model=requested_model,
                request_bytes=len(body),
                duration_ms=round((time.monotonic() - started) * 1000),
                error=type(exc).__name__,
                error_detail=_safe_preview(str(exc)),
            )

    def websocket_start(self, flow: http.HTTPFlow) -> None:
        if not flow.metadata.get("harness_responses_websocket"):
            return
        self.websocket_states[flow.id] = ResponsesWebSocketState()
        self._record(
            "websocket_bridge_started",
            original_host=flow.metadata.get("harness_original_host"),
            original_path=flow.metadata.get("harness_original_path"),
            method="GET",
            local_model=self.model,
            local_slot=self.local_slot,
        )

    def websocket_message(self, flow: http.HTTPFlow) -> None:
        if (
            not flow.metadata.get("harness_responses_websocket")
            or flow.websocket is None
        ):
            return
        message = flow.websocket.messages[-1]
        if message.injected:
            return
        if not message.is_text:
            if flow.metadata.get("harness_ws_remote_model"):
                self._audit_cloud_bytes(
                    "cloud.websocket.frame",
                    _websocket_message_bytes(message),
                    flow_id=flow.id,
                    direction=(
                        "codex_to_openai" if message.from_client else "openai_to_codex"
                    ),
                    transport="websocket",
                    opcode="binary",
                    model=flow.metadata.get("harness_ws_remote_model"),
                )
            return
        try:
            payload = json.loads(message.content)
        except (UnicodeDecodeError, json.JSONDecodeError):
            if flow.metadata.get("harness_ws_remote_model"):
                self._audit_cloud_bytes(
                    "cloud.websocket.frame",
                    _websocket_message_bytes(message),
                    flow_id=flow.id,
                    direction=(
                        "codex_to_openai" if message.from_client else "openai_to_codex"
                    ),
                    transport="websocket",
                    opcode="text",
                    model=flow.metadata.get("harness_ws_remote_model"),
                )
            return
        if not message.from_client:
            if flow.metadata.get("harness_ws_remote_model"):
                self._audit_cloud_bytes(
                    "cloud.websocket.frame",
                    _websocket_message_bytes(message),
                    flow_id=flow.id,
                    direction="openai_to_codex",
                    transport="websocket",
                    opcode="text",
                    message_type=payload.get("type"),
                    model=flow.metadata.get("harness_ws_remote_model"),
                )
            message_type = payload.get("type")
            if message_type in REMOTE_RESPONSE_TERMINAL_TYPES and flow.metadata.get(
                "harness_ws_remote_model"
            ):
                requested_model = flow.metadata.pop("harness_ws_remote_model", None)
                started = flow.metadata.pop("harness_ws_remote_at", time.monotonic())
                event = (
                    "remote_inference_completed"
                    if message_type == "response.completed"
                    else "remote_inference_error"
                )
                self._record(
                    event,
                    requested_model=requested_model,
                    local_slot=self.local_slot,
                    status=200 if message_type == "response.completed" else None,
                    transport="websocket",
                    terminal_event=message_type,
                    duration_ms=max(
                        0,
                        round((time.monotonic() - started) * 1000),
                    ),
                )
            return
        if not ResponsesWebSocketState.is_response_create(payload):
            if flow.metadata.get("harness_ws_remote_model"):
                self._audit_cloud_bytes(
                    "cloud.websocket.frame",
                    _websocket_message_bytes(message),
                    flow_id=flow.id,
                    direction="codex_to_openai",
                    transport="websocket",
                    opcode="text",
                    message_type=payload.get("type"),
                    model=flow.metadata.get("harness_ws_remote_model"),
                )
            return
        payload, repaired_compaction_ids = normalize_legacy_compaction_item_ids(payload)
        payload, repaired_reasoning_items = normalize_invalid_local_reasoning_items(
            payload
        )
        if repaired_compaction_ids or repaired_reasoning_items:
            message.content = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        if repaired_compaction_ids:
            self._record(
                "legacy_compaction_id_repaired",
                repaired_compaction_id_count=repaired_compaction_ids,
                transport="websocket",
            )
        if repaired_reasoning_items:
            self._record(
                "local_reasoning_item_repaired",
                repaired_local_reasoning_item_count=repaired_reasoning_items,
                transport="websocket",
            )
        requested_model = responses_request_model(payload)
        if not should_route_responses_locally(
            payload, self.local_slot, local_compaction=self.local_compaction
        ):
            outbound, pro_mode = adapt_gpt_56_pro_request(payload)
            if pro_mode:
                message.content = _json_bytes(outbound)
            effective_model = responses_request_model(outbound)
            flow.metadata["harness_ws_remote_model"] = requested_model
            flow.metadata["harness_ws_effective_model"] = effective_model
            flow.metadata["harness_ws_pro_mode"] = pro_mode
            flow.metadata["harness_ws_remote_at"] = time.monotonic()
            self._audit_cloud_bytes(
                "cloud.websocket.frame",
                _websocket_message_bytes(message),
                flow_id=flow.id,
                direction="codex_to_openai",
                transport="websocket",
                opcode="text",
                message_type=outbound.get("type"),
                model=effective_model,
                requested_model=requested_model,
                pro_mode=pro_mode,
            )
            self._audit_cloud_websocket_handshake(
                flow,
                model=effective_model,
            )
            self._record(
                "remote_inference_routed",
                original_host=flow.metadata.get("harness_original_host"),
                original_path=flow.metadata.get("harness_original_path"),
                method="WEBSOCKET",
                requested_model=requested_model,
                effective_model=effective_model,
                pro_mode=pro_mode,
                local_slot=self.local_slot,
                transport="websocket",
            )
            return
        message.drop()
        state = self.websocket_states.setdefault(flow.id, ResponsesWebSocketState())
        if state.is_prewarm(payload):
            prefill_payload = state.expand(payload)
            response_id, events = state.acknowledge_prewarm(payload)
            for event in events:
                self._inject_websocket_event(flow, event)
            self._record(
                "websocket_prewarm_completed",
                local_model=self.model,
                local_server=self.local_server,
                requested_model=requested_model,
                local_slot=self.local_slot,
                response_id=response_id,
                duration_ms=0,
            )
            self._schedule_prefix_prefill(
                prefill_payload,
                requested_model=requested_model,
            )
            return
        task = asyncio.create_task(
            self._serve_local_websocket_request(flow, state, payload, message.timestamp)
        )
        self.websocket_tasks.add(task)
        task.add_done_callback(self.websocket_tasks.discard)

    def websocket_end(self, flow: http.HTTPFlow) -> None:
        self.websocket_states.pop(flow.id, None)
        requested_model = flow.metadata.pop("harness_ws_remote_model", None)
        if requested_model:
            self._audit_cloud_bytes(
                "cloud.websocket.lifecycle.end",
                _json_bytes(_websocket_end_metadata(flow)),
                flow_id=flow.id,
                direction="transport",
                transport="websocket",
                model=requested_model,
            )
            started = flow.metadata.pop("harness_ws_remote_at", time.monotonic())
            self._record(
                "remote_inference_error",
                requested_model=requested_model,
                local_slot=self.local_slot,
                transport="websocket",
                terminal_event="websocket.closed_before_terminal_event",
                duration_ms=max(0, round((time.monotonic() - started) * 1000)),
            )

    async def _serve_local_websocket_request(
        self,
        flow: http.HTTPFlow,
        state: ResponsesWebSocketState,
        payload: dict[str, Any],
        received_at: float,
    ) -> None:
        started = time.monotonic()
        requested_model = responses_request_model(payload)
        expanded = payload
        request_digest: str | None = None
        try:
            expanded = state.expand(payload)
            if is_compaction_trigger_request(expanded):
                transformed, tool_names = transform_compaction_request(
                    expanded, self.model
                )
            else:
                transformed, tool_names = transform_responses_request(
                    expanded, self.model
                )
            advertised_tool_names, advertised_tool_types = extract_advertised_tools(
                transformed
            )
            namespace_tools = extract_namespace_tools(transformed)
            transformed, flattened_names = flatten_namespace_tools_for_local_model(
                transformed
            )
            transformed, client_tool_mappings, client_tool_names = (
                expose_client_tools_for_local_model(transformed)
            )
            local_tool_names, _ = extract_advertised_tools(transformed)
            transformed, loop_action, loop_detection = self._guard_loop(
                transformed, is_compaction_request(expanded)
            )
            emittable_tools = tool_names | flattened_names | client_tool_names
            native_tool_streaming = bool(
                emittable_tools and self._native_tool_streaming_enabled()
            )
            transformed, repaired_rules = repair_local_request(transformed)
            if repaired_rules:
                self._record(
                    "local_request_repaired",
                    local_model=self.model,
                    requested_model=requested_model,
                    local_slot=self.local_slot,
                    transport="websocket_bridge",
                    local_request_rules_broken=sorted(repaired_rules),
                )
            broken_rules = validate_local_request(transformed)
            if broken_rules:
                self._record(
                    "local_request_invalid",
                    local_model=self.model,
                    requested_model=requested_model,
                    local_slot=self.local_slot,
                    transport="websocket_bridge",
                    local_request_rules_broken=sorted(broken_rules),
                )
                self._inject_websocket_event(
                    flow,
                    {
                        "type": "error",
                        "status": 422,
                        "error": {
                            "type": "invalid_request_error",
                            "message": (
                                "local interceptor refused an invalid request: "
                                + ", ".join(sorted(broken_rules))
                            ),
                        },
                    },
                )
                return
            request_digest = canonical_digest(transformed)
            replayed = self._local_event_replay(request_digest)
            if replayed is not None:
                for event in replayed:
                    self._inject_websocket_event(flow, event)
                state.remember(expanded, replayed)
                self._record(
                    "local_response_replayed",
                    local_model=self.model,
                    requested_model=requested_model,
                    local_slot=self.local_slot,
                    transport="websocket_bridge",
                    status=200,
                    duration_ms=round((time.monotonic() - started) * 1000),
                )
                return
            if self._local_failures_exhausted(request_digest):
                self._inject_websocket_event(
                    flow,
                    {
                        "type": "error",
                        "status": 422,
                        "error": {
                            "type": "invalid_request_error",
                            "message": (
                                "local interceptor refused a request that already "
                                f"failed {LOCAL_FAILURE_ATTEMPTS} times"
                            ),
                        },
                    },
                )
                self._record(
                    "local_request_refused",
                    local_model=self.model,
                    requested_model=requested_model,
                    local_slot=self.local_slot,
                    transport="websocket_bridge",
                    status=422,
                )
                return
            transform_ms = max(0, round((time.monotonic() - started) * 1000))
            # Serialize the request once: the byte count, the upstream body the
            # worker sends, and the audit field all come from this single dump.
            request_body = json.dumps(
                transformed, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            request_summary = summarize_responses_request(expanded, len(request_body))
            if not is_compaction_request(expanded):
                self._schedule_prefix_prefill(
                    requested_model=requested_model,
                    transformed=transformed,
                    digest=request_digest,
                )
            self._record(
                "inference_routed",
                original_host=flow.metadata.get("harness_original_host"),
                original_path=flow.metadata.get("harness_original_path"),
                method="WEBSOCKET",
                local_model=self.model,
                local_server=self.local_server,
                requested_model=requested_model,
                local_slot=self.local_slot,
                transport="websocket_bridge",
                native_tool_streaming=native_tool_streaming,
                tool_count=len(emittable_tools),
                tool_names=sorted(local_tool_names),
                advertised_tool_names=sorted(advertised_tool_names),
                advertised_tool_types=sorted(advertised_tool_types),
                client_to_bridge_ms=max(0, round((time.time() - received_at) * 1000)),
                transform_ms=transform_ms,
                opaque_compaction_item_count=opaque_compaction_item_count(
                    expanded.get("input")
                ),
                **request_summary,
            )
            if loop_detection is not None:
                self._record(
                    "doom_loop_detected",
                    local_model=self.model,
                    requested_model=requested_model,
                    local_slot=self.local_slot,
                    transport="websocket_bridge",
                    **_loop_guard_fields(loop_detection, loop_action),
                )
            first_visible_ms = 0
            upstream_metrics: dict[str, Any] = {}

            def record_connection(metrics: dict[str, Any]) -> None:
                upstream_metrics.update(metrics)
                self._record(
                    "inference_upstream_connected",
                    local_model=self.model,
                    local_server=self.local_server,
                    requested_model=requested_model,
                    local_slot=self.local_slot,
                    transport="websocket_bridge",
                    dispatch_ms=transform_ms,
                    **metrics,
                )

            def emit_event(event: dict[str, Any]) -> None:
                nonlocal first_visible_ms
                if not first_visible_ms and _is_visible_response_event(event):
                    first_visible_ms = max(
                        1,
                        round((time.monotonic() - started) * 1000),
                    )
                    self._record(
                        "inference_first_visible_event",
                        local_model=self.model,
                        local_server=self.local_server,
                        requested_model=requested_model,
                        local_slot=self.local_slot,
                        transport="websocket_bridge",
                        first_visible_ms=first_visible_ms,
                    )
                self._inject_websocket_event(flow, event)

            (
                events,
                first_byte_ms,
                remapped,
                textual_repair,
                replay,
            ) = await self._local_sse_events(
                transformed,
                registered_tools=tool_names,
                namespace_tools=namespace_tools,
                emittable_tools=emittable_tools,
                client_tool_mappings=client_tool_mappings,
                compaction=is_compaction_request(expanded),
                on_event=emit_event,
                on_first_byte=lambda milliseconds: self._record(
                    "inference_first_byte",
                    local_model=self.model,
                    local_server=self.local_server,
                    requested_model=requested_model,
                    local_slot=self.local_slot,
                    transport="websocket_bridge",
                    first_byte_ms=milliseconds,
                ),
                on_connection=record_connection,
                request_started=started,
                request_body=request_body,
            )
            response_id = state.remember(expanded, events)
            duration_ms = round((time.monotonic() - started) * 1000)
            if textual_repair:
                self._record("textual_tool_call_repaired", local_model=self.model)
            self.local_failures.pop(request_digest, None)
            raw_events, raw_events_size = replay
            self._store_local_event_replay(request_digest, raw_events, raw_events_size)
            completed = next(
                (
                    event.get("response")
                    for event in reversed(events)
                    if isinstance(event, dict)
                    and event.get("type") == "response.completed"
                ),
                None,
            )
            usage = next(
                (
                    found
                    for found in (
                        usage_from_response_event(event) for event in reversed(events)
                    )
                    if found is not None
                ),
                None,
            )
            self._record(
                "inference_completed",
                status=200,
                local_model=self.model,
                local_server=self.local_server,
                requested_model=requested_model,
                local_slot=self.local_slot,
                transport="websocket_bridge",
                duration_ms=duration_ms,
                first_byte_ms=first_byte_ms,
                first_visible_ms=first_visible_ms or None,
                bridge_overhead_ms=max(0, duration_ms - first_byte_ms),
                transform_ms=transform_ms,
                route_to_first_byte_ms=max(0, first_byte_ms - transform_ms),
                **upstream_metrics,
                tool_names_remapped=remapped,
                response_id=response_id,
                empty_model_output=response_is_empty(completed) or None,
                **usage_fields(usage, duration_ms=duration_ms),
            )
            self._record_performance_warnings(
                request_bytes=request_summary.get("request_bytes"),
                input_tokens=(usage or {}).get("input_tokens"),
                cache_hit_percent=usage_fields(usage, duration_ms=duration_ms).get(
                    "cache_hit_percent"
                ),
                transform_ms=transform_ms,
                first_byte_ms=first_byte_ms,
                first_visible_ms=first_visible_ms,
                connect_ms=upstream_metrics.get("connect_ms"),
                connection_reused=upstream_metrics.get("connection_reused"),
            )
        except Exception as exc:
            detail = _safe_preview(str(exc))
            if self._serve_synthetic_compaction(
                flow,
                state,
                expanded,
                started=started,
                requested_model=requested_model,
                reason=type(exc).__name__,
                detail=detail,
            ):
                return
            self._inject_websocket_event(
                flow,
                {
                    "type": "error",
                    "status": getattr(exc, "code", 502),
                    "error": {
                        "type": "server_error",
                        "message": (
                            f"Local {self.local_server} bridge failed: "
                            f"{detail or type(exc).__name__}"
                        ),
                    },
                },
            )
            self._record(
                "inference_error",
                status=getattr(exc, "code", None),
                local_model=self.model,
                requested_model=requested_model,
                local_slot=self.local_slot,
                transport="websocket_bridge",
                duration_ms=round((time.monotonic() - started) * 1000),
                error=type(exc).__name__,
                error_detail=_safe_preview(str(exc)),
                local_failure_count=(
                    self._record_local_failure(request_digest)
                    if request_digest
                    and not _is_prompt_size_rejection(getattr(exc, "code", None))
                    else None
                ),
            )

    def _serve_synthetic_compaction(
        self,
        flow: http.HTTPFlow,
        state: ResponsesWebSocketState,
        payload: dict[str, Any],
        *,
        started: float,
        requested_model: str | None,
        reason: str,
        detail: str,
    ) -> bool:
        """Answer a failed compaction turn locally instead of failing it.

        Compaction is maintenance Codex runs on its own schedule, and a failure
        surfaces as a hard error on a turn the user never asked for. The summary
        only has to be a valid compaction item, so a local one is always better
        than propagating the backend's error.
        """
        if not is_compaction_request(payload):
            return False
        try:
            response = synthetic_compaction_response(payload, self.model)
            body, framed = compaction_response_sse(response)
            if not framed:
                return False
            events = decode_json_events(SSEEventDecoder().feed(body))
        except Exception:
            return False
        if not events:
            return False
        for event in events:
            self._inject_websocket_event(flow, event)
        response_id = state.remember(payload, events)
        self._record(
            "compaction_short_circuited",
            local_model=self.model,
            requested_model=requested_model,
            local_slot=self.local_slot,
            transport="websocket_bridge",
            duration_ms=round((time.monotonic() - started) * 1000),
            response_id=response_id,
            error=reason,
            error_detail=detail,
        )
        return True

    async def _local_sse_events(
        self,
        payload: dict[str, Any],
        *,
        registered_tools: set[str],
        namespace_tools: dict[str, Any],
        emittable_tools: set[str],
        client_tool_mappings: dict[str, dict[str, str]],
        compaction: bool,
        on_event,
        on_first_byte,
        on_connection=None,
        request_started: float | None = None,
        request_body: bytes | None = None,
    ) -> tuple[list[dict[str, Any]], int, int, bool, tuple[list[bytes], int]]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        started = request_started if request_started is not None else time.monotonic()

        def worker() -> None:
            headers = self._local_headers("omlx-codex-interceptor-websocket-bridge/0.3")
            body = (
                request_body
                if request_body is not None
                else json.dumps(
                    payload, ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
            )
            try:
                with self._pool().stream(
                    body,
                    headers,
                    timeout=LOCAL_UPSTREAM_READ_TIMEOUT_SECONDS,
                ) as (response, connection_metrics):
                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        ("connection", connection_metrics),
                    )
                    if not 200 <= response.status < 300:
                        raw = response.read()
                        message = _upstream_error_message(raw)
                        wrapped = LocalUpstreamHTTPError(
                            response.status,
                            message
                            or f"local upstream returned HTTP {response.status}",
                        )
                        loop.call_soon_threadsafe(queue.put_nowait, ("error", wrapped))
                        return
                    first = True
                    while True:
                        chunk = response.readline()
                        if not chunk:
                            break
                        if first and not _is_sse_keepalive(chunk):
                            first = False
                            loop.call_soon_threadsafe(
                                queue.put_nowait,
                                ("first", round((time.monotonic() - started) * 1000)),
                            )
                        loop.call_soon_threadsafe(queue.put_nowait, ("data", chunk))
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

        thread_future = asyncio.create_task(asyncio.to_thread(worker))
        raw = bytearray()
        first_byte_ms = 0
        connection_metrics: dict[str, Any] = {}
        events: list[dict[str, Any]] = []
        raw_events: list[bytes] = []
        raw_events_size = 0
        stream_directly = not compaction and (
            not emittable_tools or self._native_tool_streaming_enabled()
        )
        normalizer = SSEToolNameNormalizer(
            registered_tools,
            namespace_tools,
            client_tool_mappings,
        )
        sanitizer = SSELocalOutputSanitizer()
        decoder = SSEEventDecoder()

        def emit_payloads(payloads) -> None:
            nonlocal raw_events_size
            for raw_event in payloads:
                decoded = json.loads(raw_event)
                if isinstance(decoded, dict):
                    events.append(decoded)
                    raw_events.append(raw_event)
                    raw_events_size += len(raw_event)
                    on_event(decoded)

        while True:
            kind, value = await queue.get()
            if kind == "first":
                first_byte_ms = int(value)
                on_first_byte(first_byte_ms)
            elif kind == "connection":
                connection_metrics = dict(value)
                if on_connection is not None:
                    on_connection(connection_metrics)
            elif kind == "data":
                if stream_directly:
                    emit_payloads(decoder.feed(sanitizer.feed(normalizer.feed(value))))
                else:
                    raw.extend(value)
            elif kind == "error":
                raise value
            elif kind == "done":
                break
        await thread_future
        textual_repair = False
        if stream_directly:
            emit_payloads(
                decoder.feed(sanitizer.feed(normalizer.feed(b"")))
                + decoder.feed(sanitizer.feed(b""))
                + decoder.finish()
            )
        else:
            body = bytes(raw)
            if emittable_tools:
                body, textual_repair = adapt_textual_tool_call_sse(
                    body, emittable_tools
                )
                body, _ = adapt_client_tool_call_sse(body, client_tool_mappings)
            body = normalizer.feed(body) + normalizer.feed(b"")
            body = sanitizer.feed(body) + sanitizer.feed(b"")
            if compaction:
                # adapt_compaction_sse_response returns the response object, not
                # a stream, so it has to be re-encoded before the SSE decoder
                # sees it.
                adapted, converted = adapt_compaction_sse_response(body)
                encoded, framed = (
                    compaction_response_sse(adapted) if converted else (b"", False)
                )
                body = encoded if framed else body
            emit_payloads(decoder.feed(body) + decoder.finish())
        if not any(event.get("type") == "response.completed" for event in events):
            raise RuntimeError("local stream ended before response.completed")
        self.last_local_activity = time.monotonic()
        return (
            events,
            first_byte_ms,
            len(normalizer.remapped),
            textual_repair,
            (raw_events, raw_events_size),
        )

    @staticmethod
    def _inject_websocket_event(flow: http.HTTPFlow, event: dict[str, Any]) -> None:
        ctx.master.commands.call(
            "inject.websocket",
            flow,
            True,
            json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            ),
        )

    def responseheaders(self, flow: http.HTTPFlow) -> None:
        if flow.metadata.get("harness_remote_inference") and flow.response is not None:
            self._audit_cloud_http_response_headers(flow)
        if not flow.metadata.get("harness_intercepted") or flow.response is None:
            return
        routed_at = flow.metadata.get("harness_local_routed_at")
        if isinstance(routed_at, (int, float)):
            flow.metadata["harness_route_to_first_byte_ms"] = max(
                0, round((time.monotonic() - routed_at) * 1000)
            )
            self._record(
                "inference_first_byte",
                local_model=self.model,
                local_server=self.local_server,
                requested_model=flow.metadata.get("harness_requested_model"),
                local_slot=self.local_slot,
                transport="http",
                first_byte_ms=flow.metadata["harness_route_to_first_byte_ms"],
            )
        content_type = flow.response.headers.get("Content-Type", "").lower()
        if "text/event-stream" not in content_type:
            return
        if flow.metadata.get("harness_compaction_request"):
            # Buffer compaction streams so response() can convert the local
            # assistant text into the single compaction item Codex expects.
            flow.response.headers.pop("Content-Length", None)
            return
        # Tool-name repair may change byte length. Streaming responses must not
        # retain the upstream Content-Length after that transformation.
        flow.response.headers.pop("Content-Length", None)
        normalizer = SSEToolNameNormalizer(
            flow.metadata.get("harness_registered_tools", []),
            flow.metadata.get("harness_namespace_tools", {}),
            flow.metadata.get("harness_client_tool_mappings", {}),
        )
        sanitizer = SSELocalOutputSanitizer()
        observer = SSEUsageObserver()
        completed_capture = _SSECompletedEventCapture()
        flow.metadata["harness_sse_normalizer"] = normalizer
        flow.metadata["harness_sse_sanitizer"] = sanitizer
        flow.metadata["harness_usage_observer"] = observer
        flow.metadata["harness_sse_completed_capture"] = completed_capture
        native_tool_streaming = self._native_tool_streaming_enabled()
        flow.metadata["harness_native_tool_streaming"] = native_tool_streaming
        if flow.metadata.get("harness_emittable_tools") and not native_tool_streaming:
            flow.metadata["harness_buffer_tool_stream"] = True
        else:
            flow.response.stream = lambda chunk: observer.feed(
                completed_capture.feed(sanitizer.feed(normalizer.feed(chunk)))
            )
        self._record(
            "sse_normalizer_installed",
            status=flow.response.status_code,
            content_type=content_type,
            registered_tool_count=len(
                flow.metadata.get("harness_registered_tools", [])
            ),
            native_tool_streaming=native_tool_streaming,
        )

    def response(self, flow: http.HTTPFlow) -> None:
        if flow.metadata.get("harness_remote_inference") and flow.response is not None:
            self._audit_cloud_http_response_headers(flow)
            self._audit_cloud_http_response_body(flow)
            self._record(
                "remote_inference_completed",
                requested_model=flow.metadata.get("harness_requested_model"),
                local_slot=self.local_slot,
                status=flow.response.status_code,
                duration_ms=self._duration_ms(flow),
                inference_kind=(
                    "image_generation"
                    if flow.metadata.get("harness_remote_auxiliary_inference")
                    else "responses"
                ),
            )
            return
        if flow.metadata.get("harness_passthrough") and flow.response is not None:
            if flow.metadata.get("harness_model_catalog"):
                patched_count = self._adapt_model_catalog(flow.response)
                self._record(
                    "model_catalog_adapted",
                    status=flow.response.status_code,
                    patched_model_count=patched_count,
                    advertised_context_window=self.local_context_window,
                )
            self._record(
                "response_passthrough",
                original_host=flow.metadata.get("harness_original_host"),
                original_path=flow.metadata.get("harness_original_path"),
                method=flow.metadata.get("harness_original_method"),
                status=flow.response.status_code,
                duration_ms=self._duration_ms(flow),
            )
            return
        if not flow.metadata.get("harness_intercepted") or flow.response is None:
            return
        normalizer = flow.metadata.get("harness_sse_normalizer")
        if normalizer is not None:
            if flow.metadata.get("harness_buffer_tool_stream"):
                raw = flow.response.get_content(strict=False)
                repaired, textual_call_repaired = adapt_textual_tool_call_sse(
                    raw, flow.metadata.get("harness_emittable_tools", [])
                )
                repaired, _ = adapt_client_tool_call_sse(
                    repaired,
                    flow.metadata.get("harness_client_tool_mappings", {}),
                )
                normalized = normalizer.feed(repaired) + normalizer.feed(b"")
                sanitizer = flow.metadata.get("harness_sse_sanitizer")
                if sanitizer is not None:
                    normalized = sanitizer.feed(normalized) + sanitizer.feed(b"")
                flow.response.headers.pop("Content-Length", None)
                flow.response.set_content(normalized)
                if textual_call_repaired:
                    self._record(
                        "textual_tool_call_repaired",
                        local_model=self.model,
                    )
            remapped = normalizer.remapped
        else:
            remapped = []
            content_type = flow.response.headers.get("Content-Type", "").lower()
            if flow.metadata.get("harness_compaction_request"):
                try:
                    raw = flow.response.get_content(strict=False)
                    if not 200 <= flow.response.status_code < 300:
                        # Codex treats a failed compaction as a failed task, so
                        # summarize locally rather than forward the error.
                        upstream_error = _response_error_message(flow.response)
                        transformed = synthetic_compaction_response(
                            flow.metadata.get("harness_compaction_payload") or {},
                            self.model,
                        )
                        adapted = True
                        flow.response.status_code = 200
                        flow.response.reason = "OK"
                        self._record(
                            "compaction_short_circuited",
                            local_model=self.model,
                            requested_model=flow.metadata.get(
                                "harness_requested_model"
                            ),
                            local_slot=self.local_slot,
                            transport="http",
                            status=200,
                            upstream_error_message=upstream_error,
                        )
                    elif "text/event-stream" in content_type:
                        transformed, adapted = adapt_compaction_sse_response(raw)
                    else:
                        decoded = json.loads(raw)
                        transformed, adapted = adapt_compaction_response(decoded)
                    if adapted:
                        flow.response.headers.pop("Content-Length", None)
                        if flow.metadata.get("harness_client_stream"):
                            body, _ = compaction_response_sse(transformed)
                            flow.response.headers["Content-Type"] = "text/event-stream"
                            flow.response.set_content(body)
                        else:
                            flow.response.headers["Content-Type"] = "application/json"
                            flow.response.set_content(
                                json.dumps(
                                    transformed,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                ).encode("utf-8")
                            )
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                    pass
            elif "application/json" in content_type:
                try:
                    decoded = json.loads(flow.response.get_content(strict=False))
                    decoded, _ = sanitize_local_response_items(decoded)
                    transformed, remapped = normalize_response_tool_names(
                        decoded,
                        flow.metadata.get("harness_registered_tools", []),
                        flow.metadata.get("harness_namespace_tools", {}),
                    )
                    transformed, _ = normalize_client_tool_calls(
                        transformed,
                        flow.metadata.get("harness_client_tool_mappings", {}),
                    )
                    flow.response.set_content(
                        json.dumps(
                            transformed,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    )
                except (UnicodeDecodeError, json.JSONDecodeError):
                    pass
        status = flow.response.status_code
        event = "inference_completed" if 200 <= status < 300 else "inference_error"
        duration_ms = self._duration_ms(flow)
        fields = {
            "status": flow.response.status_code,
            "local_model": self.model,
            "local_server": self.local_server,
            "duration_ms": duration_ms,
            "first_byte_ms": flow.metadata.get("harness_route_to_first_byte_ms"),
            "tool_names_remapped": len(remapped),
        }
        digest = flow.metadata.get("harness_request_digest")
        if event == "inference_error":
            fields["upstream_error_message"] = _response_error_message(flow.response)
            if isinstance(digest, str) and not _is_prompt_size_rejection(status):
                fields["local_failure_count"] = self._record_local_failure(digest)
        else:
            if isinstance(digest, str):
                self.local_failures.pop(digest, None)
            fields.update(
                usage_fields(self._response_usage(flow), duration_ms=duration_ms)
            )
            if self._response_was_empty(flow):
                fields["empty_model_output"] = True
            if isinstance(digest, str):
                self._store_local_replay(digest, flow)
        self._record(event, **fields)

    def _local_replay(self, digest: str) -> tuple[str, bytes] | None:
        entry = self.local_replays.get(digest)
        if entry is None:
            return None
        expires, content_type, body = entry
        if expires <= time.monotonic():
            self.local_replays.pop(digest, None)
            return None
        self.local_replays.move_to_end(digest)
        return content_type, body

    def _store_local_replay(self, digest: str, flow: http.HTTPFlow) -> None:
        """Keep an answer we already hold in full, for an identical retry.

        Only bodies the addon has already buffered are stored, so nothing is
        accumulated that was not going to be in memory anyway.
        """
        if flow.response is None:
            return
        try:
            body = flow.response.get_content(strict=False)
        except Exception:
            return
        if not body or len(body) > LOCAL_REPLAY_MAX_BYTES:
            return
        content_type = flow.response.headers.get("Content-Type", "application/json")
        self.local_replays[digest] = (
            time.monotonic() + LOCAL_REPLAY_TTL_SECONDS,
            content_type,
            bytes(body),
        )
        self.local_replays.move_to_end(digest)
        while len(self.local_replays) > LOCAL_REPLAY_ENTRIES:
            self.local_replays.popitem(last=False)

    def _local_event_replay(self, digest: str) -> list | None:
        entry = self.local_event_replays.get(digest)
        if entry is None:
            return None
        expires, raw_events = entry
        if expires <= time.monotonic():
            self.local_event_replays.pop(digest, None)
            return None
        self.local_event_replays.move_to_end(digest)
        # The stored payloads are immutable raw bytes; each replay parses them
        # fresh, so callers can never mutate shared state.
        return decode_json_events(raw_events)

    def _store_local_event_replay(
        self, digest: str, raw_events: list[bytes], size: int
    ) -> None:
        """Keep a bridged answer for an identical retry of the same turn.

        The raw event payloads are stored as emitted; the byte total is
        tracked while they are appended, so nothing is re-serialized or
        copied here. They are re-parsed only if an identical turn actually
        replays, which is the rare path.
        """
        if not raw_events or size > LOCAL_REPLAY_MAX_BYTES:
            return
        self.local_event_replays[digest] = (
            time.monotonic() + LOCAL_REPLAY_TTL_SECONDS,
            raw_events,
        )
        self.local_event_replays.move_to_end(digest)
        while len(self.local_event_replays) > LOCAL_REPLAY_ENTRIES:
            self.local_event_replays.popitem(last=False)

    def _local_failures_exhausted(self, digest: str) -> bool:
        entry = self.local_failures.get(digest)
        if entry is None:
            return False
        expires, count = entry
        if expires <= time.monotonic():
            self.local_failures.pop(digest, None)
            return False
        return count >= LOCAL_FAILURE_ATTEMPTS

    def _record_local_failure(self, digest: str) -> int:
        now = time.monotonic()
        expires, count = self.local_failures.get(digest, (0.0, 0))
        count = count + 1 if expires > now else 1
        self.local_failures = {
            key: value for key, value in self.local_failures.items() if value[0] > now
        }
        self.local_failures[digest] = (now + LOCAL_FAILURE_TTL_SECONDS, count)
        return count

    def _response_usage(self, flow: http.HTTPFlow) -> dict[str, Any] | None:
        observer = flow.metadata.get("harness_usage_observer")
        if observer is not None and observer.usage is not None:
            return observer.usage
        return self._usage_from_body(flow)

    def _usage_from_body(self, flow: http.HTTPFlow) -> dict[str, Any] | None:
        decoded = self._decoded_response(flow)
        if isinstance(decoded, dict):
            return usage_from_response_event(decoded)
        if isinstance(decoded, list):
            for event in reversed(decoded):
                usage = usage_from_response_event(event)
                if usage is not None:
                    return usage
        return None

    def _response_was_empty(self, flow: http.HTTPFlow) -> bool:
        decoded = self._decoded_response(flow)
        if isinstance(decoded, dict):
            response = decoded.get("response") if "response" in decoded else decoded
            return response_is_empty(response)
        if isinstance(decoded, list):
            for event in reversed(decoded):
                if (
                    isinstance(event, dict)
                    and event.get("type") == "response.completed"
                ):
                    return response_is_empty(event.get("response"))
        return False

    def _decoded_response(self, flow: http.HTTPFlow) -> Any:
        if "harness_decoded_response" in flow.metadata:
            return flow.metadata["harness_decoded_response"]
        capture = flow.metadata.get("harness_sse_completed_capture")
        if capture is not None:
            # The stream wrapper already retained the terminal event; both
            # consumers (usage and emptiness) only need that one event.
            completed = capture.completed_event()
            if completed is not None:
                flow.metadata["harness_decoded_response"] = completed
                return completed
        decoded: Any = None
        try:
            raw = flow.response.get_content(strict=False) if flow.response else b""
        except Exception:
            raw = b""
        if raw:
            content_type = (
                flow.response.headers.get("Content-Type", "").lower()
                if flow.response
                else ""
            )
            if "text/event-stream" in content_type:
                decoded = decode_json_events(SSEEventDecoder().feed(raw))
            else:
                try:
                    decoded = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    decoded = None
        flow.metadata["harness_decoded_response"] = decoded
        return decoded

    def error(self, flow: http.HTTPFlow) -> None:
        if flow.metadata.get("harness_remote_inference"):
            self._audit_cloud_bytes(
                "cloud.transport.error",
                _json_bytes(
                    {
                        "error": type(flow.error).__name__ if flow.error else "unknown",
                        "detail": _error_detail(flow.error),
                    }
                ),
                flow_id=flow.id,
                direction="transport",
                transport="http",
                model=flow.metadata.get("harness_requested_model"),
            )
            self._record(
                "remote_inference_error",
                requested_model=flow.metadata.get("harness_requested_model"),
                local_slot=self.local_slot,
                duration_ms=self._duration_ms(flow),
                error=type(flow.error).__name__ if flow.error else "unknown",
                error_detail=_error_detail(flow.error),
            )
            return
        if flow.metadata.get("harness_passthrough"):
            self._record(
                "response_passthrough_error",
                original_host=flow.metadata.get("harness_original_host"),
                original_path=flow.metadata.get("harness_original_path"),
                method=flow.metadata.get("harness_original_method"),
                duration_ms=self._duration_ms(flow),
            )
            return
        if flow.metadata.get("harness_intercepted"):
            normalizer = flow.metadata.get("harness_sse_normalizer")
            response = flow.response
            event = (
                "inference_stream_closed"
                if response is not None
                and 200 <= response.status_code < 300
                and normalizer is not None
                else "inference_error"
            )
            fields = {
                "status": response.status_code if response is not None else None,
                "local_model": self.model,
                "duration_ms": self._duration_ms(flow),
                "tool_names_remapped": (
                    len(normalizer.remapped) if normalizer is not None else 0
                ),
            }
            if event == "inference_error":
                fields["error"] = type(flow.error).__name__ if flow.error else "unknown"
                fields["error_detail"] = _error_detail(flow.error)
            self._record(event, **fields)

    def _audit_cloud_websocket_handshake(
        self,
        flow: http.HTTPFlow,
        *,
        model: str | None,
    ) -> None:
        if flow.metadata.get("harness_cloud_websocket_handshake"):
            return
        flow.metadata["harness_cloud_websocket_handshake"] = True
        common = {
            "flow_id": flow.id,
            "transport": "websocket",
            "model": model,
        }
        self._audit_cloud_bytes(
            "cloud.websocket.handshake.request.metadata",
            _json_bytes(_http_request_metadata(flow)),
            direction="codex_to_openai",
            **common,
        )
        self._audit_cloud_bytes(
            "cloud.websocket.handshake.request.start_line",
            _http_request_start_line(flow.request),
            direction="codex_to_openai",
            **common,
        )
        self._audit_cloud_bytes(
            "cloud.websocket.handshake.headers",
            audit_header_bytes(
                flow.request.headers,
                include_credentials=self.cloud_audit_include_credentials,
            ),
            direction="codex_to_openai",
            **common,
        )
        if flow.response is not None:
            self._audit_cloud_bytes(
                "cloud.websocket.handshake.response.metadata",
                _json_bytes(_http_response_metadata(flow)),
                direction="openai_to_codex",
                **common,
            )
            self._audit_cloud_bytes(
                "cloud.websocket.handshake.response.start_line",
                _http_response_start_line(flow.response),
                direction="openai_to_codex",
                **common,
            )
            self._audit_cloud_bytes(
                "cloud.websocket.handshake.response.headers",
                audit_header_bytes(
                    flow.response.headers,
                    include_credentials=self.cloud_audit_include_credentials,
                ),
                direction="openai_to_codex",
                **common,
            )

    def _audit_cloud_http_request(
        self,
        flow: http.HTTPFlow,
        *,
        model: str | None,
        inference_kind: str,
    ) -> None:
        request = flow.request
        common = {
            "flow_id": flow.id,
            "direction": "codex_to_openai",
            "transport": "http",
            "method": request.method.upper(),
            "host": flow.metadata.get("harness_original_host", request.host),
            "path": flow.metadata.get("harness_original_path", request.path),
            "model": model,
            "inference_kind": inference_kind,
        }
        self._audit_cloud_bytes(
            "cloud.http.request.metadata",
            _json_bytes(_http_request_metadata(flow)),
            **common,
        )
        self._audit_cloud_bytes(
            "cloud.http.request.start_line",
            _http_request_start_line(request),
            **common,
        )
        self._audit_cloud_bytes(
            "cloud.http.request.headers",
            audit_header_bytes(
                request.headers,
                include_credentials=self.cloud_audit_include_credentials,
            ),
            **common,
        )
        raw = _http_message_bytes(request)
        self._audit_cloud_bytes("cloud.http.request.body", raw, **common)
        decoded = _http_decoded_message_bytes(request)
        if decoded != raw:
            self._audit_cloud_bytes(
                "cloud.http.request.body.decoded",
                decoded,
                **common,
            )

    def _audit_cloud_http_response_headers(self, flow: http.HTTPFlow) -> None:
        if flow.response is None or flow.metadata.get("harness_cloud_response_headers"):
            return
        flow.metadata["harness_cloud_response_headers"] = True
        response = flow.response
        common = {
            "flow_id": flow.id,
            "direction": "openai_to_codex",
            "transport": "http",
            "status": response.status_code,
            "model": flow.metadata.get("harness_requested_model"),
            "inference_kind": (
                "image_generation"
                if flow.metadata.get("harness_remote_auxiliary_inference")
                else "responses"
            ),
        }
        self._audit_cloud_bytes(
            "cloud.http.response.metadata",
            _json_bytes(_http_response_metadata(flow)),
            **common,
        )
        self._audit_cloud_bytes(
            "cloud.http.response.start_line",
            _http_response_start_line(response),
            **common,
        )
        self._audit_cloud_bytes(
            "cloud.http.response.headers",
            audit_header_bytes(
                response.headers,
                include_credentials=self.cloud_audit_include_credentials,
            ),
            **common,
        )

    def _audit_cloud_http_response_body(self, flow: http.HTTPFlow) -> None:
        if flow.response is None:
            return
        response = flow.response
        common = {
            "flow_id": flow.id,
            "direction": "openai_to_codex",
            "transport": "http",
            "status": response.status_code,
            "model": flow.metadata.get("harness_requested_model"),
            "inference_kind": (
                "image_generation"
                if flow.metadata.get("harness_remote_auxiliary_inference")
                else "responses"
            ),
        }
        raw = _http_message_bytes(response)
        self._audit_cloud_bytes("cloud.http.response.body", raw, **common)
        decoded = _http_decoded_message_bytes(response)
        if decoded != raw:
            self._audit_cloud_bytes(
                "cloud.http.response.body.decoded",
                decoded,
                **common,
            )

    def _audit_cloud_bytes(self, event: str, data: bytes, **fields: Any) -> None:
        if self.cloud_audit is None:
            return
        try:
            self.cloud_audit.record_bytes(event, data, **fields)
        except Exception as exc:
            if self._cloud_audit_failed:
                return
            self._cloud_audit_failed = True
            self._record(
                "cloud_audit_error",
                error=type(exc).__name__,
                error_detail=_safe_preview(str(exc)),
            )

    @staticmethod
    def _duration_ms(flow: http.HTTPFlow) -> int | None:
        started = flow.metadata.get("harness_request_started")
        if not isinstance(started, (int, float)):
            return None
        return max(0, round((time.monotonic() - started) * 1000))

    def _adapt_model_catalog(self, response: http.Response) -> int:
        if not 200 <= response.status_code < 300:
            return 0
        try:
            response.decode(strict=False)
            decoded = json.loads(response.get_content(strict=False))
            self._cache_model_catalog(decoded)
            selected = select_lowest_visible_codex_model(decoded)
            available = {
                item.get("slug")
                for item in decoded.get("models", [])
                if isinstance(item, dict) and isinstance(item.get("slug"), str)
            }
            should_recover = (
                selected is not None
                and (
                    self.local_slot_auto
                    or self.local_slot is None
                    or self.local_slot not in available
                )
                and selected["slug"] != self.local_slot
            )
            if should_recover:
                previous = self.local_slot
                self.local_slot = selected["slug"]
                self._record(
                    "local_slot_recovered",
                    previous_local_slot=previous,
                    local_slot=self.local_slot,
                    local_label=self.local_label,
                )
            adapted, count = adapt_model_catalog_for_local_tools(
                decoded,
                local_slot=self.local_slot,
                display_name=self.local_label if self.local_slot else None,
                description=(
                    f"Served locally by {self.local_server}: {self.model}"
                    if self.local_slot
                    else None
                ),
                context_window=self.local_context_window,
            )
            if self.gpt_56_pro_enabled:
                adapted, pro_count = add_gpt_56_pro_catalog_entry(adapted)
                count += pro_count
        except (UnicodeDecodeError, json.JSONDecodeError):
            return 0
        if not count:
            return 0
        response.headers.pop("Content-Encoding", None)
        response.headers.pop("Content-Length", None)
        response.set_content(
            json.dumps(adapted, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        return count

    def _cache_model_catalog(self, decoded: Any) -> None:
        if not isinstance(decoded, dict) or not isinstance(decoded.get("models"), list):
            return
        path = self.status_path.parent / "model-catalog.json"
        temporary = path.with_suffix(".tmp")
        try:
            temporary.write_text(
                json.dumps(decoded, ensure_ascii=False, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            os.chmod(temporary, 0o600)
            temporary.replace(path)
        except OSError:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)

    def _record(self, event: str, **fields: Any) -> None:
        # This file is user-visible diagnostics, not a traffic capture. Keep
        # the privacy boundary at the writer so prompt/response content cannot
        # leak even if a future call site accidentally passes it here.
        payload = {
            "time": time.time(),
            "event": event,
            "session_id": self.session_id,
            **{key: value for key, value in fields.items() if key in SAFE_EVENT_FIELDS},
        }
        try:
            now = time.monotonic()
            if now >= self._status_size_checked_at:
                self._status_size_checked_at = now + STATUS_SIZE_CHECK_SECONDS
                if self.status_path.stat().st_size >= MAX_STATUS_BYTES:
                    # Keep the same inode so a live dashboard can detect
                    # truncation and resume without exposing an unbounded
                    # request history.
                    with self.status_path.open("w", encoding="utf-8"):
                        pass
                    os.chmod(self.status_path, 0o600)
            with self.status_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
        except OSError:
            # emit_event records mid-stream; a deleted or unwritable status
            # file must never take down a live turn.
            pass


class _SSECompletedEventCapture:
    """Retain the terminal ``response.completed`` event seen while streaming.

    The stream wrapper already forwards every chunk, so a C-level marker scan
    per chunk is far cheaper than re-decoding the whole body at end of turn
    just to find this one event. Anything ambiguous (a marker straddling
    chunks before the capture started, a missing terminal event) simply
    yields ``None`` and the caller falls back to decoding the body.
    """

    _MARKER = b"response.completed"

    def __init__(self) -> None:
        self._pending = b""
        self._tail: bytearray | None = None
        self._event: dict[str, Any] | None = None
        self._parsed = False

    def feed(self, chunk: bytes) -> bytes:
        if self._tail is not None:
            self._tail.extend(chunk)
            return chunk
        marker = chunk.find(self._MARKER)
        if marker < 0:
            if self._pending:
                combined = self._pending + chunk[: len(self._MARKER) - 1]
                if self._MARKER in combined:
                    # The marker straddles a chunk boundary, so the start of
                    # the event line was not retained. Capture from here and
                    # let the decoder drop the partial first event.
                    self._tail = bytearray(chunk)
                    self._pending = b""
                    return chunk
            self._pending = chunk[1 - len(self._MARKER) :]
            return chunk
        start = chunk.rfind(b"\n", 0, marker) + 1
        self._pending = b""
        self._tail = bytearray(chunk[start:])
        return chunk

    def completed_event(self) -> dict[str, Any] | None:
        if not self._parsed:
            self._parsed = True
            if self._tail:
                for event in reversed(
                    decode_json_events(SSEEventDecoder().feed(bytes(self._tail)))
                ):
                    if event.get("type") == "response.completed":
                        self._event = event
                        break
        return self._event


def _loop_guard_fields(detection: dict, action: str | None) -> dict:
    return {
        "loop_tool_name": detection.get("name"),
        "loop_repeats": detection.get("repeats"),
        "loop_identical_outputs": detection.get("identical_outputs"),
        "loop_arguments_digest": detection.get("arguments_digest"),
        "loop_guard_action": action or "observed",
        "loop_kind": detection.get("kind") or "identical",
    }


def _is_visible_response_event(event: Any) -> bool:
    if not isinstance(event, dict):
        return False
    event_type = event.get("type")
    if event_type in {
        "response.output_text.delta",
        "response.function_call_arguments.delta",
        "response.custom_tool_call_input.delta",
    }:
        return bool(event.get("delta"))
    if event_type not in {
        "response.output_item.added",
        "response.output_item.done",
        "response.content_part.added",
        "response.content_part.done",
    }:
        return False
    item = event.get("item") or event.get("part")
    if not isinstance(item, dict):
        return False
    if item.get("type") in {
        "function_call",
        "custom_tool_call",
        "computer_call",
        "tool_search_call",
    }:
        return True
    content = item.get("content")
    if isinstance(content, list):
        return any(
            isinstance(part, dict)
            and isinstance(part.get("text"), str)
            and bool(part["text"])
            for part in content
        )
    return isinstance(item.get("text"), str) and bool(item["text"])


def _positive_int_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def _error_detail(error: Any) -> str | None:
    if error is None:
        return None
    message = getattr(error, "msg", None) or str(error)
    if not isinstance(message, str) or not message:
        return None
    return _safe_preview(message)


def _response_error_message(response: http.Response) -> str | None:
    try:
        raw = response.get_content(strict=False)
    except Exception:
        return None
    if not raw:
        return None
    try:
        decoded = json.loads(raw[:2000].decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return None
    if isinstance(decoded, dict):
        error = decoded.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str):
                return _safe_preview(message)
    return None


def _safe_preview(value: str, *, limit: int = 500) -> str:
    return "".join(
        character if character.isprintable() else " " for character in value
    )[:limit]


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _http_request_start_line(request: Any) -> bytes:
    method = getattr(request, "method", "GET")
    path = getattr(request, "path", "/")
    version = getattr(request, "http_version", "HTTP/1.1")
    return f"{method} {path} {version}\r\n".encode("utf-8", errors="surrogateescape")


def _http_response_start_line(response: Any) -> bytes:
    version = getattr(response, "http_version", "HTTP/1.1")
    status = getattr(response, "status_code", 0)
    reason = getattr(response, "reason", "") or ""
    return f"{version} {status} {reason}\r\n".encode("utf-8", errors="surrogateescape")


def _http_request_metadata(flow: Any) -> dict[str, Any]:
    request = flow.request
    return {
        "method": getattr(request, "method", None),
        "scheme": getattr(request, "scheme", None),
        "host": getattr(request, "host", None),
        "port": getattr(request, "port", None),
        "path": getattr(request, "path", None),
        "url": getattr(request, "url", None),
        "http_version": getattr(request, "http_version", None),
        "timestamp_start": getattr(request, "timestamp_start", None),
        "timestamp_end": getattr(request, "timestamp_end", None),
        "client_connection": _connection_metadata(getattr(flow, "client_conn", None)),
        "server_connection": _connection_metadata(getattr(flow, "server_conn", None)),
        "fidelity": "decrypted application-layer metadata exposed by mitmproxy",
    }


def _http_response_metadata(flow: Any) -> dict[str, Any]:
    response = flow.response
    return {
        "status_code": getattr(response, "status_code", None),
        "reason": getattr(response, "reason", None),
        "http_version": getattr(response, "http_version", None),
        "timestamp_start": getattr(response, "timestamp_start", None),
        "timestamp_end": getattr(response, "timestamp_end", None),
        "client_connection": _connection_metadata(getattr(flow, "client_conn", None)),
        "server_connection": _connection_metadata(getattr(flow, "server_conn", None)),
        "fidelity": "decrypted application-layer metadata exposed by mitmproxy",
    }


def _connection_metadata(connection: Any) -> dict[str, Any] | None:
    if connection is None:
        return None
    alpn = getattr(connection, "alpn", None)
    if isinstance(alpn, bytes):
        alpn = alpn.decode("ascii", errors="replace")
    return {
        "id": getattr(connection, "id", None),
        "peername": getattr(connection, "peername", None),
        "sockname": getattr(connection, "sockname", None),
        "tls": getattr(connection, "tls", None),
        "tls_version": getattr(connection, "tls_version", None),
        "alpn": alpn,
        "cipher": getattr(connection, "cipher", None),
        "sni": getattr(connection, "sni", None),
        "timestamp_start": getattr(connection, "timestamp_start", None),
        "timestamp_tls_setup": getattr(connection, "timestamp_tls_setup", None),
    }


def _websocket_end_metadata(flow: Any) -> dict[str, Any]:
    websocket = getattr(flow, "websocket", None)
    return {
        "close_code": getattr(websocket, "close_code", None),
        "close_reason": getattr(websocket, "close_reason", None),
        "closed_by_client": getattr(websocket, "closed_by_client", None),
        "timestamp_end": getattr(websocket, "timestamp_end", None),
        "message_count": len(getattr(websocket, "messages", []) or []),
        "client_connection": _connection_metadata(getattr(flow, "client_conn", None)),
        "server_connection": _connection_metadata(getattr(flow, "server_conn", None)),
    }


def _http_decoded_message_bytes(message: Any) -> bytes:
    try:
        content = message.get_content(strict=False)
    except Exception:
        return b""
    if isinstance(content, bytes):
        return content
    return bytes(content or b"")


def _http_message_bytes(message: Any) -> bytes:
    raw = getattr(message, "raw_content", None)
    if isinstance(raw, bytes):
        return raw
    try:
        content = message.get_content(strict=False)
    except Exception:
        return b""
    if isinstance(content, bytes):
        return content
    return bytes(content or b"")


def _is_sse_keepalive(chunk: bytes) -> bool:
    stripped = chunk.strip()
    return not stripped or stripped.startswith(b":")


def _upstream_error_message(raw: bytes) -> str:
    message = raw.decode("utf-8", errors="replace").strip()
    try:
        decoded = json.loads(message)
    except json.JSONDecodeError:
        return message
    if isinstance(decoded, dict):
        error = decoded.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"]
    return message


def _websocket_message_bytes(message: Any) -> bytes:
    content = getattr(message, "content", b"")
    if isinstance(content, bytes):
        return content
    if isinstance(content, str):
        return content.encode("utf-8")
    return bytes(content)


addons = [CodexLocalInterceptor()]
