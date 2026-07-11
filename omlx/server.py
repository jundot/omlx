# SPDX-License-Identifier: Apache-2.0
"""
OpenAI-compatible API server for oMLX.

This module provides a FastAPI server that exposes an OpenAI-compatible
API for LLM inference using MLX on Apple Silicon.

Features:
- Multi-model serving with LRU-based memory management
- Continuous batching for high throughput
- Paged KV cache with prefix sharing
- OpenAI-compatible chat/completions API
- Anthropic Messages API compatibility
- Streaming responses
- MCP (Model Context Protocol) tool integration
- Tool calling (Qwen/Llama formats)
- Structured output (JSON schema validation)

Usage:
    # Multi-model serving
    omlx serve --model-dir /path/to/models --max-model-memory 32GB

    # With pinned models
    omlx serve --model-dir /path/to/models --max-model-memory 48GB --pin llama-3b,qwen-7b

    # With MCP tools
    omlx serve --model-dir /path/to/models --max-model-memory 32GB --mcp-config mcp.json

The server provides:
    - POST /v1/completions - Text completions
    - POST /v1/chat/completions - Chat completions
    - POST /v1/messages - Anthropic Messages API
    - POST /v1/responses - OpenAI Responses API (Codex compatibility)
    - GET /v1/models - List available models (with load status)
    - GET /health - Health check
    - GET /v1/mcp/tools - List MCP tools
    - GET /v1/mcp/servers - MCP server status
    - POST /v1/mcp/execute - Execute MCP tool
"""

import argparse
import asyncio
import inspect
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union

from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi import Request as FastAPIRequest
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from omlx._version import __version__

from .api.anthropic_models import (
    MessagesRequest as AnthropicMessagesRequest,
)
from .api.anthropic_models import (
    TokenCountRequest,
    TokenCountResponse,
)
from .api.anthropic_utils import (
    convert_anthropic_to_internal,
    convert_anthropic_to_internal_harmony,
    convert_anthropic_tools_to_internal,
    convert_internal_to_anthropic_response,
    create_content_block_start_event,
    create_content_block_stop_event,
    create_error_event,
    create_input_json_delta_event,
    create_message_delta_event,
    create_message_start_event,
    create_message_stop_event,
    create_text_delta_event,
    create_thinking_delta_event,
    map_finish_reason_to_stop_reason,
    request_has_cache_control,
)
from .api.embedding_models import (
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
)
from .api.embedding_utils import (
    encode_embedding_base64,
    normalize_embedding_items,
    normalize_input,
    truncate_embedding,
)
from .api.markitdown import (
    MARKITDOWN_MODEL_ID,
    MarkItDownRequestError,
    convert_messages_to_markdown_async,
    is_markitdown_model,
    markitdown_model_visible,
    preprocess_markitdown_file_parts_async,
    request_has_file_parts,
    stream_messages_to_markdown_async,
)

# Import from new modular API
from .api.openai_models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    ModelInfo,
    ModelsResponse,
    PromptTokensDetails,
    Usage,
)
from .api.parser_tool_calls import (
    convert_parser_tool_calls as _convert_parser_tool_calls,
)
from .api.rerank_models import (
    RerankRequest,
    RerankResponse,
    RerankResult,
    RerankUsage,
)
from .api.responses_models import (
    OutputItem,
    ResponseObject,
    ResponsesRequest,
)
from .api.responses_utils import (
    ResponseStateCorruptError,
    ResponseStateNotFoundError,
    ResponseStore,
    build_function_call_output_item,
    build_message_output_item,
    build_reasoning_output_item,
    build_response_store_record,
    build_response_usage,
    convert_responses_input_to_messages,
    convert_responses_tools,
    format_sse_event,
    normalize_response_output_to_messages,
)
from .api.thinking import ThinkingParser, extract_thinking, prompt_opens_thinking
from .api.tool_calling import (
    ToolCallStreamFilter,
    build_json_system_prompt,
    convert_tools_for_template,
    enrich_tool_params_for_gemma4,
    extract_tool_calls_with_thinking,
    parse_json_output,
    restore_gemma4_param_names,
    sanitize_tool_call_markup,
)
from .api.utils import (
    clean_special_tokens,
    detect_and_strip_partial,
    extract_multimodal_content,
    extract_text_content,
    has_nonleading_system_message,
    prepare_system_messages_for_template,
    uses_native_reasoning_content,
)
from .engine import BaseEngine, VLMBatchedEngine
from .engine.embedding import EmbeddingEngine
from .engine.reranker import RerankerEngine
from .engine_pool import EnginePool
from .exceptions import (
    EnginePoolError,
    InsufficientMemoryError,
    InvalidRequestError,
    ModelBusyError,
    ModelLoadingError,
    ModelNotFoundError,
    ModelTooLargeError,
    ModelUnavailableError,
    PrefillMemoryExceededError,
    SchedulerQueueFullError,
)
from .server_metrics import get_server_metrics, reset_server_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Security bearer for API key authentication
security = HTTPBearer(auto_error=False)


# =============================================================================
# Server State
# =============================================================================


class EngineType(Enum):
    """Type of engine to retrieve."""

    LLM = "llm"
    EMBEDDING = "embedding"
    RERANKER = "reranker"


@dataclass
class SamplingDefaults:
    """Default sampling parameters."""

    # Fallback context length used by ``get_max_context_window`` only
    # when neither a per-model override nor a model-config-discovered
    # native context length is available. Setting this does NOT cap
    # models that declare their own context — use
    # ``max_context_window_policy`` for the operator-policy cap.
    max_context_window: int = 32768
    # Optional operator policy cap. When set, models whose native
    # context length is discovered get ``min(native, policy)``. Per-model
    # overrides and the fallback default above are not affected — those
    # represent explicit choices that the policy cannot override
    # without surprising migration semantics for existing
    # ``settings.json`` files.
    max_context_window_policy: int | None = None
    max_tokens: int = 32768
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 0
    repetition_penalty: float = 1.0
    force_sampling: bool = False


@dataclass
class ServerState:
    """
    Encapsulated server state.

    This class holds all global state for the server, making it easier
    to manage and test.
    """

    engine_pool: Optional[EnginePool] = None
    default_model: Optional[str] = None
    mcp_manager: Optional[object] = None
    mcp_executor: Optional[object] = None
    sampling: SamplingDefaults = field(default_factory=SamplingDefaults)
    api_key: Optional[str] = None
    settings_manager: Optional[object] = None  # ModelSettingsManager
    global_settings: Optional[object] = None  # GlobalSettings
    hf_downloader: Optional[object] = None  # HFDownloader
    ms_downloader: Optional[object] = None  # MSDownloader
    process_memory_enforcer: Optional[object] = None  # ProcessMemoryEnforcer
    responses_store: ResponseStore = field(default_factory=ResponseStore)
    oq_manager: Optional[object] = None  # OQManager
    hf_uploader: Optional[object] = None  # HFUploader
    # False while the startup pinned-model preload is still running.
    # /health returns 503 with status "loading" until it flips to True so
    # port watchdogs see liveness instead of a closed port (#2184).
    pinned_preload_complete: bool = True


# Global server state instance
_server_state: ServerState = ServerState()


def get_server_state() -> ServerState:
    """Get the global server state."""
    return _server_state


def get_engine_pool() -> EnginePool:
    """Get the engine pool, raising error if not initialized."""
    if _server_state.engine_pool is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    return _server_state.engine_pool


def get_mcp_manager():
    """Get the MCP manager instance (may be None)."""
    return _server_state.mcp_manager


async def verify_api_key(
    request: FastAPIRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> bool:
    """Verify API key if configured.

    Checks the provided Bearer token against the main API key and all sub keys.
    Also accepts the x-api-key header as a fallback (Anthropic SDK compatibility).
    """
    from .admin.auth import fingerprint_key, verify_any_api_key

    # No auth required if no API key is configured
    if _server_state.api_key is None:
        return True

    # Skip verification if enabled
    if (
        _server_state.global_settings is not None
        and _server_state.global_settings.auth.skip_api_key_verification
    ):
        return True

    # Extract API key from Bearer token or x-api-key header
    if credentials is not None:
        api_key_value = credentials.credentials
    else:
        # Fallback: check x-api-key header (Anthropic SDK compatibility)
        api_key_value = request.headers.get("x-api-key")
        if api_key_value is None:
            raise HTTPException(status_code=401, detail="API key required")

    # Check main key and sub keys
    sub_keys = (
        _server_state.global_settings.auth.sub_keys
        if _server_state.global_settings is not None
        else []
    )
    if not verify_any_api_key(api_key_value, _server_state.api_key, sub_keys):
        logger.warning("Rejected API key (fp=%s)", fingerprint_key(api_key_value))
        raise HTTPException(status_code=401, detail="Invalid API key")

    return True


def _reset_boundary_snapshots_for_server() -> None:
    """Reset ephemeral boundary snapshots at server lifecycle boundaries."""
    engine_pool = _server_state.engine_pool
    if engine_pool is None:
        return

    scheduler_config = getattr(engine_pool, "_scheduler_config", None)
    cache_dir = getattr(scheduler_config, "paged_ssd_cache_dir", None)
    if not cache_dir:
        return

    try:
        from .cache.boundary_snapshot_store import reset_boundary_snapshot_root

        reset_boundary_snapshot_root(Path(cache_dir))
    except Exception as exc:  # pragma: no cover - best-effort cleanup
        logger.warning("Failed to reset boundary snapshot directory: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan for startup/shutdown events."""
    # Startup: Auto-populate server aliases for the admin dashboard
    # so users get sensible hostname/IP options for API URL hints
    # without manual configuration. Only runs when the persisted list
    # is empty so user-curated aliases are never overwritten.
    if (
        _server_state.global_settings is not None
        and not _server_state.global_settings.server.server_aliases
    ):
        try:
            from .utils.network import detect_server_aliases

            detected = detect_server_aliases(
                host=_server_state.global_settings.server.host
            )
            if detected:
                _server_state.global_settings.server.server_aliases = detected
                try:
                    _server_state.global_settings.save()
                except Exception as save_exc:  # pragma: no cover - filesystem race
                    logger.warning(
                        "Auto-detected server aliases but could not persist: %s",
                        save_exc,
                    )
                logger.info("Auto-detected server aliases: %s", detected)
        except Exception as exc:  # pragma: no cover - never block startup
            logger.warning("Server alias auto-detection failed: %s", exc)

    _reset_boundary_snapshots_for_server()

    # Start process memory enforcer if configured
    if (
        _server_state.global_settings is not None
        and _server_state.engine_pool is not None
    ):
        from .process_memory_enforcer import ProcessMemoryEnforcer

        memory_settings = _server_state.global_settings.memory
        enforcer = ProcessMemoryEnforcer(
            engine_pool=_server_state.engine_pool,
            memory_guard_tier=memory_settings.memory_guard_tier,
            memory_guard_custom_ceiling_gb=memory_settings.memory_guard_custom_ceiling_gb,
            settings_manager=_server_state.settings_manager,
            prefill_memory_guard=memory_settings.prefill_memory_guard,
            global_settings=_server_state.global_settings,
            soft_threshold=memory_settings.soft_threshold,
            hard_threshold=memory_settings.hard_threshold,
            prefill_safe_zone_ratio=memory_settings.prefill_safe_zone_ratio,
            prefill_min_chunk_tokens=memory_settings.prefill_min_chunk_tokens,
        )
        _server_state.process_memory_enforcer = enforcer
        _server_state.engine_pool._process_memory_enforcer = enforcer
        # Engine pool consults the enforcer for the pre-load ceiling.
        _server_state.engine_pool._get_final_ceiling = enforcer.get_final_ceiling
        enforcer.start()

    # Startup: Preload pinned models in the background so uvicorn binds the
    # HTTP port immediately after this lifespan yields. A large pinned
    # preload (hundreds of GB) otherwise keeps the port closed for minutes,
    # and anything watchdogging the port hard-kills the process mid-load —
    # the worst moment for the kernel wired-memory stranding in #2184.
    # /health answers 503 with status "loading" until the preload finishes.
    # Runs after the enforcer wiring above so the preload sees the final
    # memory ceiling.
    preload_task = None
    if _server_state.engine_pool is not None:
        _server_state.pinned_preload_complete = False

        async def _preload_pinned() -> None:
            try:
                await _server_state.engine_pool.preload_pinned_models()
            finally:
                _server_state.pinned_preload_complete = True

        preload_task = asyncio.create_task(_preload_pinned())

    # Start TTL-only checker if process memory enforcer is not running
    # (enforcer already includes TTL checks in its polling loop)
    ttl_task = None
    if (
        _server_state.process_memory_enforcer is None
        and _server_state.engine_pool is not None
    ):

        async def _ttl_check_loop():
            while True:
                try:
                    if _server_state.settings_manager is not None:
                        await _server_state.engine_pool.check_ttl_expirations(
                            _server_state.settings_manager,
                            global_idle_timeout_seconds=(
                                _server_state.global_settings.idle_timeout.idle_timeout_seconds
                                if _server_state.global_settings
                                else None
                            ),
                        )
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"TTL check error: {e}")
                await asyncio.sleep(1.0)

        ttl_task = asyncio.create_task(_ttl_check_loop())

    # Initialize MCP if config provided
    # Priority: env var > settings.json
    mcp_config = os.environ.get("OMLX_MCP_CONFIG")
    if not mcp_config and _server_state.global_settings:
        mcp_config = _server_state.global_settings.mcp.config_path
    if mcp_config:
        await init_mcp(mcp_config)

    yield

    # Shutdown: Save all-time stats, stop TTL task, process memory enforcer, etc.
    if preload_task is not None and not preload_task.done():
        # SIGTERM arrived while pinned models were still loading. Cancel the
        # await; engine_pool.shutdown() below unloads whatever finished.
        preload_task.cancel()
        with suppress(asyncio.CancelledError):
            await preload_task
    get_server_metrics().save_alltime()
    if ttl_task is not None:
        ttl_task.cancel()
        try:
            await ttl_task
        except asyncio.CancelledError:
            pass
    if _server_state.process_memory_enforcer is not None:
        await _server_state.process_memory_enforcer.stop()
        if _server_state.engine_pool is not None:
            _server_state.engine_pool._process_memory_enforcer = None
        logger.info("Process memory enforcer stopped")
    if _server_state.hf_downloader is not None:
        await _server_state.hf_downloader.shutdown()
        logger.info("HF Downloader stopped")
    if _server_state.ms_downloader is not None:
        await _server_state.ms_downloader.shutdown()
        logger.info("MS Downloader stopped")
    if _server_state.mcp_manager is not None:
        await _server_state.mcp_manager.stop()
        logger.info("MCP manager stopped")
    if _server_state.engine_pool is not None:
        await _server_state.engine_pool.shutdown()
        _reset_boundary_snapshots_for_server()
        logger.info("Engine pool shutdown")


app = FastAPI(
    title="oMLX API",
    description="LLM inference, optimized for your Mac",
    version=__version__,
    lifespan=lifespan,
)

# Include MCP routes
from .api.mcp_routes import router as mcp_router
from .api.mcp_routes import set_mcp_manager_getter

set_mcp_manager_getter(get_mcp_manager)
app.include_router(mcp_router, dependencies=[Depends(verify_api_key)])

# Include audio routes only when mlx-audio is installed.
# audio_routes.py itself only imports fastapi/stdlib at module level, so it
# would always import successfully — we need an explicit mlx-audio check.
try:
    import mlx_audio as _  # noqa: F401

    from .api.audio_routes import router as audio_router

    app.include_router(audio_router, dependencies=[Depends(verify_api_key)])
    del _
except ImportError:
    pass

# Include admin routes
from .admin.auth import _RedirectToLogin
from .admin.routes import router as admin_router
from .admin.routes import set_admin_getters

set_admin_getters(
    get_server_state,
    get_engine_pool,
    lambda: _server_state.settings_manager,
    lambda: _server_state.global_settings,
)
app.include_router(admin_router)


@app.exception_handler(_RedirectToLogin)
async def redirect_to_login_handler(request, exc):
    """Redirect unauthenticated browser requests to the admin login page."""
    return RedirectResponse(url="/admin", status_code=302)


def _status_to_error_type(status_code: int) -> str:
    """Map HTTP status code to OpenAI error type string."""
    if status_code == 401:
        return "authentication_error"
    if status_code == 404:
        return "not_found_error"
    if status_code == 413:
        # Body-size rejections are still request-shape errors.
        return "invalid_request_error"
    if status_code == 429:
        return "rate_limit_error"
    if status_code >= 500:
        return "server_error"
    return "invalid_request_error"


def _is_api_route(request: FastAPIRequest) -> bool:
    """Check if request targets an OpenAI-compatible API route.

    Path-prefix only. This assumes the FastAPI app is mounted at root
    (the oMLX deployment shape) and that route paths are case-sensitive
    — both true today. If a future deployment mounts this app under a
    prefix (``app.mount("/api", ...)``), ``request.url.path`` returns
    the full mounted path and every ``/v1/...`` route would be
    classified as non-API. Switch to ``request.scope.get("route")``
    matching at that point.
    """
    return request.url.path.startswith("/v1/")


def _openai_error_body(message, status_code: int, param=None, code=None) -> dict:
    """Build an OpenAI-compatible error response body."""
    return {
        "error": {
            "message": message,
            "type": _status_to_error_type(status_code),
            "param": param,
            "code": code,
        }
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request: FastAPIRequest, exc: HTTPException):
    """Log all HTTP errors (4xx/5xx) before returning the response."""
    # Admin session expiry from dashboard polling — not worth logging.
    # But keep /admin/api/login 401s visible (possible brute force attempts).
    _is_admin_session_expiry = (
        request.url.path.startswith("/admin/")
        and request.url.path != "/admin/api/login"
        and exc.status_code == 401
    )
    if not _is_admin_session_expiry:
        logger.warning(
            "%s %s → %d: %s",
            request.method,
            request.url.path,
            exc.status_code,
            exc.detail,
        )
    if _is_api_route(request):
        content = _openai_error_body(exc.detail, exc.status_code)
    else:
        content = {"detail": exc.detail}
    return JSONResponse(status_code=exc.status_code, content=content)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: FastAPIRequest, exc: RequestValidationError
):
    """Log request validation errors (422) before returning the response."""
    logger.warning(
        "%s %s → 422: %s",
        request.method,
        request.url.path,
        exc.errors(),
    )
    if _is_api_route(request):
        errors = exc.errors()
        parts = []
        for err in errors:
            loc = " -> ".join(str(x) for x in err.get("loc", []))
            msg = err.get("msg", "")
            parts.append(f"{loc}: {msg}" if loc else msg)
        detail_str = "; ".join(parts)
        param = errors[0].get("loc", [None])[-1] if errors else None
        content = _openai_error_body(detail_str, 422, param=param)
    else:
        content = {"detail": exc.errors()}
    return JSONResponse(status_code=422, content=content)


@app.exception_handler(InvalidRequestError)
async def invalid_request_error_handler(
    request: FastAPIRequest, exc: InvalidRequestError
):
    """Map internal request validation failures to OpenAI-compatible 400s."""
    logger.warning(
        "%s %s -> 400: %s",
        request.method,
        request.url.path,
        exc,
    )
    if _is_api_route(request):
        content = _openai_error_body(str(exc), 400, param=exc.field)
    else:
        content = {"detail": str(exc)}
    return JSONResponse(status_code=400, content=content)


@app.exception_handler(SchedulerQueueFullError)
async def scheduler_queue_full_handler(
    request: FastAPIRequest, exc: SchedulerQueueFullError
):
    """Map scheduler queue cap exhaustion to HTTP 503 + Retry-After."""
    logger.warning(
        "%s %s → 503: %s",
        request.method,
        request.url.path,
        exc,
    )
    detail = (
        f"Scheduler waiting queue full ({exc.current_depth}/{exc.max_depth}). "
        f"Try again shortly."
    )
    if _is_api_route(request):
        content = _openai_error_body(detail, 503)
    else:
        content = {"detail": detail}
    return JSONResponse(
        status_code=503,
        content=content,
        headers={"Retry-After": "1"},
    )


def _prefill_memory_error_detail(exc: PrefillMemoryExceededError) -> str:
    return (
        "oMLX prefill memory guard rejected this prompt: "
        f"{str(exc)} "
        "To continue, set Memory Guard to aggressive, raise the custom "
        "memory guard ceiling, free system memory, or compact/reduce context."
    )


def _prefill_memory_openai_error_body(
    exc: PrefillMemoryExceededError,
    *,
    status_code: int = 400,
) -> dict:
    content = _openai_error_body(
        _prefill_memory_error_detail(exc),
        status_code,
        code="prefill_memory_exceeded",
    )
    content["type"] = "error"
    content["error"]["omlx_code"] = "prefill_memory_exceeded"
    if exc.estimated_bytes is not None:
        content["error"]["estimated_bytes"] = exc.estimated_bytes
    if exc.limit_bytes is not None:
        content["error"]["limit_bytes"] = exc.limit_bytes
    return content


@app.exception_handler(PrefillMemoryExceededError)
async def prefill_memory_exceeded_handler(
    request: FastAPIRequest, exc: PrefillMemoryExceededError
):
    """Map prefill peak overshoot to HTTP 400 with a clear JSON body.

    The synchronous prefill memory guard in ``Scheduler.add_request`` raises
    this when the estimated KV+SDPA peak for a request would push memory
    past the user-configured memory guard ceiling. The caller's prompt
    fits in the model's context window but is too large for the host's
    headroom.

    This is an actionable request rejection, not an HTTP body-size
    rejection. HTTP 400 also prevents Anthropic clients from collapsing
    this oMLX memory-guard failure into Anthropic's generic
    "Request too large (max 32MB)" body-size error.
    """
    detail = _prefill_memory_error_detail(exc)
    status_code = 400
    logger.warning(
        "%s %s → %d: %s",
        request.method,
        request.url.path,
        status_code,
        detail,
    )
    if _is_api_route(request):
        # code="prefill_memory_exceeded" lets OpenAI-SDK clients branch
        # on the failure mode. Without it, "context window too small"
        # and "host has no memory headroom" both surface as
        # invalid_request_error with code=None and clients can only
        # tell the user "shorten your prompt" — which is wrong when
        # the actual fix is to loosen the memory guard.
        # Surface the structured fields so clients can branch on
        # numeric values instead of regex-matching the human message.
        # OpenAI clients ignore unknown error fields so this is a
        # forward-compatible extension.
        content = _prefill_memory_openai_error_body(exc, status_code=status_code)
    else:
        content = {
            "detail": detail,
            "omlx_code": "prefill_memory_exceeded",
        }
        if exc.estimated_bytes is not None:
            content["estimated_bytes"] = exc.estimated_bytes
        if exc.limit_bytes is not None:
            content["limit_bytes"] = exc.limit_bytes
    return JSONResponse(status_code=status_code, content=content)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: FastAPIRequest, exc: Exception):
    """Log unhandled exceptions as 500 errors."""
    logger.error(
        "%s %s → 500 (unhandled): %s",
        request.method,
        request.url.path,
        exc,
    )
    if _is_api_route(request):
        content = _openai_error_body("Internal server error", 500)
    else:
        content = {"detail": "Internal server error"}
    return JSONResponse(status_code=500, content=content)


class DebugRequestLoggingMiddleware:
    """Pure ASGI middleware for trace-level request body logging.

    Uses raw ASGI protocol instead of BaseHTTPMiddleware to avoid
    wrapping StreamingResponse in an intermediate pipe layer, which
    causes connection corruption on HTTP keep-alive connections.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or not logger.isEnabledFor(5)
            or scope.get("method") != "POST"
        ):
            await self.app(scope, receive, send)
            return

        # Read and cache the request body for logging
        body_parts = []
        while True:
            message = await receive()
            body_parts.append(message)
            if not message.get("more_body", False):
                break

        body = b"".join(part.get("body", b"") for part in body_parts)
        logger.log(
            5,
            "Incoming %s %s — body: %s",
            scope["method"],
            scope["path"],
            body.decode("utf-8", errors="replace"),
        )

        # Replay cached body for inner app, then forward real receive
        body_sent = False

        async def cached_receive():
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, cached_receive, send)


app.add_middleware(DebugRequestLoggingMiddleware)


# =============================================================================
# Engine Getters
# =============================================================================


def _wake_process_memory_enforcer(*, active: bool = False) -> None:
    enforcer = _server_state.process_memory_enforcer
    wake = getattr(enforcer, "wake", None) if enforcer is not None else None
    if callable(wake):
        wake(active=active)


async def get_engine(
    model_id: str | None = None,
    engine_type: EngineType = EngineType.LLM,
    _lease: bool = False,
    _leased_out: list | None = None,
) -> Union[BaseEngine, EmbeddingEngine, RerankerEngine]:
    """
    Get engine for the specified model and type.

    This is the unified engine getter that handles LLM, embedding, and reranker models.

    Args:
        model_id: Model ID to get engine for, or None for default (LLM only)
        engine_type: Type of engine to retrieve (LLM, EMBEDDING, or RERANKER)
        _lease: When True, take an atomic in-use lease on the engine that the
            pool actually loaded (eviction-proof until released). The caller
            MUST release exactly one lease per successful leased call.
        _leased_out: When _lease is True, the EXACT pool model_id that was
            leased is appended to this list. Release using that id (not the
            request model) so the lease/release ids always match even when the
            pool falls back to the default model.

    Returns:
        The loaded engine of the appropriate type

    Raises:
        HTTPException: If model not found, wrong type, or memory error
    """
    pool = get_engine_pool()

    # Default model only applies to LLM
    if model_id is None:
        if engine_type != EngineType.LLM:
            raise HTTPException(
                status_code=400,
                detail=f"Model ID is required for {engine_type.value} engines",
            )
        model_id = _server_state.default_model

    if model_id is None:
        raise HTTPException(
            status_code=400, detail="No model specified and no default model set"
        )

    # Resolve alias/profile request to the physical model. Exposed profiles
    # may carry engine-construction settings (MTP/DFlash/etc.); pass those
    # transient settings to the pool so the loaded variant can switch without
    # mutating the base model's persisted settings.
    requested_model_id = model_id
    runtime_settings = None
    sm = _server_state.settings_manager
    if (
        engine_type == EngineType.LLM
        and sm is not None
        and hasattr(sm, "get_exposed_profile_runtime_settings_for_request")
    ):
        runtime = sm.get_exposed_profile_runtime_settings_for_request(
            requested_model_id
        )
        if runtime is not None:
            model_id, runtime_settings = runtime
        else:
            model_id = pool.resolve_model_id(model_id, sm)
    else:
        model_id = pool.resolve_model_id(model_id, sm)
    _wake_process_memory_enforcer(active=True)

    # Only thread optional kwargs through when they are needed, so the common
    # path keeps the original pool.get_engine(model_id) call shape.
    _lease_kwargs = {"_lease": True} if _lease else {}
    if runtime_settings is not None:
        _lease_kwargs["runtime_settings"] = runtime_settings
    try:
        engine = await pool.get_engine(model_id, **_lease_kwargs)
        if _lease and _leased_out is not None:
            _leased_out.append(model_id)
    except ModelNotFoundError as e:
        # Fallback to default model if enabled (LLM only)
        if (
            engine_type == EngineType.LLM
            and _server_state.global_settings
            and _server_state.global_settings.model.model_fallback
            and _server_state.default_model
        ):
            logger.info(
                f"Model '{model_id}' not found, falling back to "
                f"default model '{_server_state.default_model}'"
            )
            try:
                _wake_process_memory_enforcer(active=True)
                _fallback_kwargs = {"_lease": True} if _lease else {}
                fb_engine = await pool.get_engine(
                    _server_state.default_model, **_fallback_kwargs
                )
                if _lease and _leased_out is not None:
                    _leased_out.append(_server_state.default_model)
                return fb_engine
            except Exception:
                pass  # Fall through to original 404

        # Show aliases instead of directory names for user-friendly display
        available = e.available_models
        sm = _server_state.settings_manager
        if sm:
            display = []
            for mid in available:
                ms = sm.get_settings(mid)
                display.append(ms.model_alias if ms.model_alias else mid)
            available = display
        detail = (
            f"Model '{model_id}' not found. "
            f"Available models: {', '.join(available) if available else '(none)'}"
        )
        raise HTTPException(status_code=404, detail=detail)
    except ModelTooLargeError as e:
        raise HTTPException(status_code=507, detail=str(e))
    except InsufficientMemoryError as e:
        raise HTTPException(status_code=507, detail=str(e))
    except ModelUnavailableError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except ModelLoadingError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ModelBusyError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except EnginePoolError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Validate engine type. If a lease was taken above but validation fails,
    # release it before raising so a rejected request never leaks an in_use
    # count (which would pin the engine non-evictable forever).
    try:
        if engine_type == EngineType.EMBEDDING:
            if not isinstance(engine, EmbeddingEngine):
                raise HTTPException(
                    status_code=400,
                    detail=f"Model '{model_id}' is not an embedding model. "
                    f"Use /v1/chat/completions for LLM models.",
                )
        elif engine_type == EngineType.RERANKER:
            if not isinstance(engine, RerankerEngine):
                raise HTTPException(
                    status_code=400,
                    detail=f"Model '{model_id}' is not a reranker model. "
                    f"Use a SequenceClassification model for reranking.",
                )
        elif engine_type == EngineType.LLM:
            # #507: non-LLM engines (STT/TTS/STS/Embedding/Reranker) previously
            # fell through and crashed on `engine.model_type` with an unhandled
            # 500. Reject with a clear 400 pointing the caller at the right
            # endpoint.
            if not isinstance(engine, BaseEngine):
                _endpoint_hint = _suggest_endpoint_for_engine(engine)
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Model '{model_id}' is not an LLM / chat model. "
                        f"{_endpoint_hint}"
                    ),
                )
    except BaseException:
        if _lease and _leased_out:
            await pool.release_engine(_leased_out.pop())
        raise

    return engine


def _suggest_endpoint_for_engine(engine: object) -> str:
    """Return a one-line hint pointing at the correct endpoint for a non-LLM engine."""
    # Import audio engine classes lazily so that oMLX without the [audio]
    # extra still imports this module.
    try:
        from omlx.engine.stt import STTEngine as stt_engine_cls
    except Exception:  # pragma: no cover - defensive
        stt_engine_cls = None
    try:
        from omlx.engine.tts import TTSEngine as tts_engine_cls
    except Exception:  # pragma: no cover - defensive
        tts_engine_cls = None
    try:
        from omlx.engine.sts import STSEngine as sts_engine_cls
    except Exception:  # pragma: no cover - defensive
        sts_engine_cls = None

    if stt_engine_cls is not None and isinstance(engine, stt_engine_cls):
        return "Use /v1/audio/transcriptions for speech-to-text models."
    if tts_engine_cls is not None and isinstance(engine, tts_engine_cls):
        return "Use /v1/audio/speech for text-to-speech models."
    if sts_engine_cls is not None and isinstance(engine, sts_engine_cls):
        return "Use /v1/audio/process for speech-to-speech / audio processing models."
    if isinstance(engine, EmbeddingEngine):
        return "Use /v1/embeddings for embedding models."
    if isinstance(engine, RerankerEngine):
        return "Use /v1/rerank for reranker models."
    return "Use the model's dedicated endpoint (see /v1/models)."


@dataclass
class _LLMEngineLease:
    """Release handle for an LLM engine lease taken from EnginePool."""

    model_id: str | None = None
    released: bool = False

    async def release(self) -> None:
        if self.released:
            return
        self.released = True
        if self.model_id is not None:
            await get_engine_pool().release_engine(self.model_id)

    def abort_requested(self) -> bool:
        if self.model_id is None or self.released:
            return False
        pool = _server_state.engine_pool
        if pool is None:
            return False
        is_abort_requested = getattr(pool, "is_abort_requested", None)
        if not callable(is_abort_requested):
            return False
        return bool(is_abort_requested(self.model_id))


async def _raise_if_llm_lease_abort_requested(lease: _LLMEngineLease) -> None:
    if lease.abort_requested():
        raise HTTPException(
            status_code=507,
            detail=(
                "Request aborted before scheduling because process memory "
                "pressure requested this model to unload. Retry with a shorter "
                "context or after memory pressure drops."
            ),
        )


async def _release_after_stream(
    generator: AsyncIterator[str],
    lease: _LLMEngineLease,
) -> AsyncIterator[str]:
    try:
        await _raise_if_llm_lease_abort_requested(lease)
        async for chunk in generator:
            yield chunk
    finally:
        await lease.release()


async def get_engine_for_model(
    model: str | None = None,
    *,
    lease: _LLMEngineLease | None = None,
) -> BaseEngine:
    """
    Get LLM engine for the specified model (or default).

    This is a convenience wrapper around get_engine() for LLM models.

    Args:
        model: Model ID to get engine for, or None for default

    Returns:
        The loaded engine

    Raises:
        HTTPException: If model not found or memory error
    """
    if lease is None:
        return await get_engine(model, EngineType.LLM)

    leased: list[str] = []
    engine = await get_engine(
        model,
        EngineType.LLM,
        _lease=True,
        _leased_out=leased,
    )
    if leased:
        lease.model_id = leased[0]
    return engine


async def get_embedding_engine(model: str) -> EmbeddingEngine:
    """
    Get embedding engine for the specified model.

    This is a convenience wrapper around get_engine() for embedding models.

    Args:
        model: Model ID to get engine for

    Returns:
        The loaded embedding engine

    Raises:
        HTTPException: If model not found, is not an embedding model, or memory error
    """
    return await get_engine(model, EngineType.EMBEDDING)


async def get_reranker_engine(model: str) -> RerankerEngine:
    """
    Get reranker engine for the specified model.

    This is a convenience wrapper around get_engine() for reranker models.

    Args:
        model: Model ID to get engine for

    Returns:
        The loaded reranker engine

    Raises:
        HTTPException: If model not found, is not a reranker model, or memory error
    """
    return await get_engine(model, EngineType.RERANKER)


@asynccontextmanager
async def acquire_embedding_engine(model: str):
    """Acquire an embedding engine with an atomic, eviction-proof in-use lease.

    Resolves + loads + validates exactly like get_embedding_engine, but holds
    the engine non-evictable for the duration of the request and releases the
    lease on the EXACT pool model_id the pool loaded (handles default-model
    fallback) in finally.
    """
    leased: list = []
    engine = await get_engine(
        model, EngineType.EMBEDDING, _lease=True, _leased_out=leased
    )
    try:
        yield engine
    finally:
        if leased:
            await get_engine_pool().release_engine(leased[0])


@asynccontextmanager
async def acquire_reranker_engine(model: str):
    """Acquire a reranker engine with an atomic, eviction-proof in-use lease.

    See acquire_embedding_engine for the lease/release contract.
    """
    leased: list = []
    engine = await get_engine(
        model, EngineType.RERANKER, _lease=True, _leased_out=leased
    )
    try:
        yield engine
    finally:
        if leased:
            await get_engine_pool().release_engine(leased[0])


def get_sampling_params(
    req_temperature: float | None,
    req_top_p: float | None,
    model_id: str | None = None,
    req_top_k: int | None = None,
    req_repetition_penalty: float | None = None,
    req_min_p: float | None = None,
    req_presence_penalty: float | None = None,
    req_frequency_penalty: float | None = None,
    req_max_tokens: int | None = None,
    ocr_defaults: dict | None = None,
    req_xtc_probability: float | None = None,
    req_xtc_threshold: float | None = None,
) -> tuple[float, float, int, float, float, float, float, int, float, float]:
    """
    Get effective sampling parameters with per-model settings support.

    Priority:
    - If force_sampling is True (global or model level): force sampling knobs
      that affect token selection.
    - max_tokens is an output length cap, so it always uses
      request > model settings > ocr_defaults > global defaults.
    - Otherwise: request > model settings > ocr_defaults > global defaults.

    Returns:
        tuple of (temperature, top_p, top_k, repetition_penalty, min_p, presence_penalty, frequency_penalty, max_tokens, xtc_probability, xtc_threshold)
    """
    global_sampling = _server_state.sampling

    # Get per-model (or exposed-profile) settings if available
    model_settings = get_model_settings_for_request(model_id)

    # Resolve alias so physical-model defaults can still be found by real model ID
    model_id = resolve_model_id(model_id)

    # Resolve OCR defaults if not provided by caller
    if ocr_defaults is None and model_id:
        ocr_defaults = _get_ocr_defaults(model_id)

    # Check force at any level
    force = global_sampling.force_sampling or (
        model_settings and model_settings.force_sampling
    )

    if force:
        # Forced mode: use model settings if available, else global
        if model_settings and model_settings.temperature is not None:
            temperature = model_settings.temperature
        elif ocr_defaults and "temperature" in ocr_defaults:
            temperature = ocr_defaults["temperature"]
        else:
            temperature = global_sampling.temperature

        if model_settings and model_settings.top_p is not None:
            top_p = model_settings.top_p
        else:
            top_p = global_sampling.top_p

        if model_settings and model_settings.top_k is not None:
            top_k = model_settings.top_k
        else:
            top_k = global_sampling.top_k
    else:
        # Normal mode: priority request > model > ocr_defaults > global
        if req_temperature is not None:
            temperature = req_temperature
        elif model_settings and model_settings.temperature is not None:
            temperature = model_settings.temperature
        elif ocr_defaults and "temperature" in ocr_defaults:
            temperature = ocr_defaults["temperature"]
        else:
            temperature = global_sampling.temperature

        if req_top_p is not None:
            top_p = req_top_p
        elif model_settings and model_settings.top_p is not None:
            top_p = model_settings.top_p
        else:
            top_p = global_sampling.top_p

        if req_top_k is not None:
            top_k = req_top_k
        elif model_settings and model_settings.top_k is not None:
            top_k = model_settings.top_k
        elif ocr_defaults and "top_k" in ocr_defaults:
            top_k = ocr_defaults["top_k"]
        else:
            top_k = global_sampling.top_k

    # Repetition penalty: request > model settings > ocr_defaults > global (1.0)
    if req_repetition_penalty is not None:
        repetition_penalty = req_repetition_penalty
    elif model_settings and model_settings.repetition_penalty is not None:
        repetition_penalty = model_settings.repetition_penalty
    elif ocr_defaults and "repetition_penalty" in ocr_defaults:
        repetition_penalty = ocr_defaults["repetition_penalty"]
    else:
        repetition_penalty = getattr(global_sampling, "repetition_penalty", 1.0)

    # Min P: request > model settings > default (0.0)
    if req_min_p is not None:
        min_p = req_min_p
    elif model_settings and getattr(model_settings, "min_p", None) is not None:
        min_p = model_settings.min_p
    else:
        min_p = 0.0

    # Presence penalty: request > model settings > default (0.0)
    if req_presence_penalty is not None:
        presence_penalty = req_presence_penalty
    elif (
        model_settings and getattr(model_settings, "presence_penalty", None) is not None
    ):
        presence_penalty = model_settings.presence_penalty
    else:
        presence_penalty = 0.0

    # Frequency penalty: request > model settings > default (0.0)
    if req_frequency_penalty is not None:
        frequency_penalty = req_frequency_penalty
    elif (
        model_settings
        and getattr(model_settings, "frequency_penalty", None) is not None
    ):
        frequency_penalty = model_settings.frequency_penalty
    else:
        frequency_penalty = 0.0

    # Max tokens is an output length cap, not a sampling knob. Honor request
    # bounds even when force_sampling pins token-selection parameters.
    if req_max_tokens is not None:
        max_tokens = req_max_tokens
    elif model_settings and model_settings.max_tokens is not None:
        max_tokens = model_settings.max_tokens
    elif ocr_defaults and "max_tokens" in ocr_defaults:
        max_tokens = ocr_defaults["max_tokens"]
    else:
        max_tokens = global_sampling.max_tokens

    # XTC probability: request > default (0.0 = disabled)
    xtc_probability = req_xtc_probability if req_xtc_probability is not None else 0.0

    # XTC threshold: request > default (0.1 = safe default when probability is set)
    xtc_threshold = req_xtc_threshold if req_xtc_threshold is not None else 0.1

    logger.debug(
        f"Sampling params: temperature={temperature}, top_p={top_p}, top_k={top_k}, "
        f"repetition_penalty={repetition_penalty}, min_p={min_p}, presence_penalty={presence_penalty}, "
        f"frequency_penalty={frequency_penalty}, max_tokens={max_tokens}, "
        f"xtc_probability={xtc_probability}, xtc_threshold={xtc_threshold}"
        f"{' (forced)' if force else ''}"
        f"{f' (model: {model_id})' if model_id else ''}"
    )
    return (
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        min_p,
        presence_penalty,
        frequency_penalty,
        max_tokens,
        xtc_probability,
        xtc_threshold,
    )


def _strip_synthetic_think_prefix(chunk_text: str, think_tag: str) -> str:
    """Drop the scheduler's synthetic think opener from a raw completions chunk.

    Raw completions are a pure continuation of the prompt. When the prompt
    itself ends with an open think tag, the scheduler still prepends a
    synthetic ``"<think>\\n"`` to the first streamed chunk (chat streams rely
    on it to rebuild the reasoning block), but the opener belongs to the
    prompt and the non-streaming completions path never returns it. Stripping
    it keeps both completion paths returning the same continuation.
    """
    prefix = f"{think_tag}\n"
    return chunk_text[len(prefix) :] if chunk_text.startswith(prefix) else chunk_text


def _resolve_thinking_budget(request, model_id: str | None) -> int | None:
    """Resolve thinking budget: request param > model settings > None."""
    # Check request-level override (OpenAI format)
    req_budget = getattr(request, "thinking_budget", None)
    # For Anthropic: check thinking.budget_tokens
    if req_budget is None and hasattr(request, "thinking") and request.thinking:
        req_budget = getattr(request.thinking, "budget_tokens", None)
    if req_budget is not None:
        return req_budget
    ms = get_model_settings_for_request(model_id)
    if ms and ms.thinking_budget_enabled and ms.thinking_budget_tokens:
        return ms.thinking_budget_tokens
    return None


def get_model_settings_for_request(model_id: str | None):
    """Return settings for the requested API model name via ModelSettingsManager."""
    sm = _server_state.settings_manager
    if not model_id or sm is None:
        return None

    resolved_model_id = resolve_model_id(model_id)
    if not hasattr(sm, "get_settings_for_request"):
        return sm.get_settings(resolved_model_id or model_id)

    return sm.get_settings_for_request(
        model_id,
        resolved_model_id=resolved_model_id,
    )


def resolve_model_id(model_id: str | None) -> str | None:
    """Resolve a model alias to its real model ID.

    Returns the resolved ID, or the original value if no alias match.
    """
    if model_id is None:
        return None
    pool = _server_state.engine_pool
    if pool is None:
        return model_id
    return pool.resolve_model_id(model_id, _server_state.settings_manager)


async def _ensure_tokenizer_for_system_probe(
    engine: BaseEngine, messages: list
) -> None:
    """Load lazy engines before probing mid-conversation system placement."""
    if not has_nonleading_system_message(messages):
        return
    if getattr(engine, "tokenizer", None) is not None:
        return
    await engine.start()


def _unsupported_mid_system_policy() -> str:
    settings = _server_state.global_settings
    preserve_cache = True
    if settings is not None:
        preserve_cache = bool(
            getattr(settings.server, "preserve_mid_system_cache", True)
        )
    return "user_note_safe" if preserve_cache else "strict"


def _format_generation_speed_for_log(
    output,
    tokens_per_sec: float,
    *,
    is_diffusion: bool,
) -> str:
    if not is_diffusion:
        return f"{tokens_per_sec:.1f} tok/s"

    parts = [f"{tokens_per_sec:.1f} tok/s e2e"]
    output_tps = float(getattr(output, "generation_tps", 0.0) or 0.0)
    if output_tps > 0:
        parts.append(f"output={output_tps:.1f} tok/s")
    canvas_tps = float(getattr(output, "diffusion_canvas_tps", 0.0) or 0.0)
    if canvas_tps > 0:
        parts.append(f"canvas={canvas_tps:.1f} tok/s")
    prompt_tps = float(getattr(output, "prompt_tps", 0.0) or 0.0)
    if prompt_tps > 0:
        parts.append(f"prompt={prompt_tps:.1f} tok/s")
    work_tps = float(getattr(output, "diffusion_work_tps", 0.0) or 0.0)
    if work_tps > 0:
        parts.append(f"work={work_tps:.1f} tok/s")
    steps = int(getattr(output, "diffusion_denoising_steps", 0) or 0)
    if steps > 0:
        parts.append(f"steps={steps}")
    return ", ".join(parts)


def _resolve_metric_durations(
    output,
    *,
    is_diffusion: bool,
    prefill_duration: float = 0.0,
    generation_duration: float = 0.0,
) -> tuple[float, float]:
    if not is_diffusion:
        return prefill_duration, generation_duration

    prompt_tps = float(getattr(output, "prompt_tps", 0.0) or 0.0)
    if prompt_tps > 0:
        prefill_duration = output.prompt_tokens / prompt_tps

    generation_tps = float(getattr(output, "generation_tps", 0.0) or 0.0)
    if generation_tps > 0:
        generation_duration = output.completion_tokens / generation_tps

    return prefill_duration, generation_duration


def _get_ocr_defaults(model_id: str | None) -> dict | None:
    """Get OCR generation defaults for a model, or None if not an OCR model."""
    if model_id is None:
        return None
    pool = _server_state.engine_pool
    if pool is None:
        return None
    entry = pool.get_entry(model_id)
    if entry is None:
        return None
    from .engine.vlm import OCR_MODEL_GENERATION_DEFAULTS, OCR_MODEL_TYPES

    cmt = getattr(entry, "config_model_type", "")
    if cmt in OCR_MODEL_TYPES:
        return OCR_MODEL_GENERATION_DEFAULTS.get(cmt)
    return None


def get_max_context_window(model_id: str | None = None) -> int | None:
    """
    Get effective max context window limit.

    Resolution:
        1. **Per-model override** (admin UI / settings.json) — always
           wins. An operator who has set a per-model number knows what
           they want; ``max_context_window_policy`` does not clamp it.
        2. **Model-config-discovered native context length** (#1308),
           optionally clamped by the operator policy: if
           ``sampling.max_context_window_policy`` is set, return
           ``min(native, policy)``; otherwise return ``native`` as-is.
        3. **Fallback default** from ``SamplingSettings.max_context_window``
           — only used when neither tier 1 nor tier 2 yields a value.
           Treated as a default, NOT capped by the policy; existing
           ``settings.json`` files carrying the historical ``32768``
           default keep working unchanged after upgrade.

    The policy field is intentionally nullable and unset by default so
    no existing install behavior shifts. Setting it engages
    ``min(native, policy)`` across every model whose native context is
    discoverable; per-model overrides remain the operator's escape
    hatch for individual models that should exceed the policy.

    Returns:
        Max context window token count, or ``None`` if no tier resolves
        (only possible when neither the model nor the global default
        provides a value, which shouldn't happen in practice).
    """
    # Resolve alias for physical model metadata, but keep requested alias settings.
    requested_model_id = model_id
    model_settings = get_model_settings_for_request(requested_model_id)
    model_id = resolve_model_id(model_id)

    # Priority 1: explicit per-model override (not capped by policy)
    if model_settings and model_settings.max_context_window is not None:
        return model_settings.max_context_window

    # Priority 2: model-native context, optionally clamped by policy
    pool = _server_state.engine_pool
    if model_id and pool is not None:
        entry = pool.get_entry(model_id)
        if entry is not None and entry.model_context_length is not None:
            native = entry.model_context_length
            policy = getattr(_server_state.sampling, "max_context_window_policy", None)
            if policy is not None and policy > 0:
                return min(native, policy)
            return native

    # Priority 3: fallback default (not capped — preserves legacy
    # settings.json behavior).
    return _server_state.sampling.max_context_window


def get_embedding_max_length(
    model_id: str | None = None,
    request_max_length: int | None = None,
) -> int | None:
    """Get max token length for embedding requests.

    Returns ``None`` when neither the request nor the server's
    ``max_context_window`` pins a limit, so the embedding model resolves its
    own configured context length (``max_position_embeddings`` / tokenizer
    ``model_max_length`` in ``MLXEmbeddingModel._resolve_max_length``) instead
    of re-truncating long-context models at the legacy 512-token cap (#1687).
    """
    if request_max_length is not None:
        return request_max_length

    return get_max_context_window(model_id)


def scale_anthropic_tokens(token_count: int, model_id: str | None = None) -> int:
    """
    Scale token count for Anthropic API response if context scaling is enabled.

    Adjusts reported token counts so that Claude Code's auto-compact
    triggers at the correct timing when using models with smaller context
    windows than the target (default 200k).

    Formula: scaled = token_count * (target_context_size / actual_context_size)

    Args:
        token_count: Original token count to scale.
        model_id: Model ID to get context window for.

    Returns:
        Scaled token count, or original if scaling not applicable.
    """
    global_settings = _server_state.global_settings
    if global_settings is None:
        return token_count

    cc = global_settings.claude_code
    if not cc.context_scaling_enabled:
        return token_count

    actual = get_max_context_window(model_id)
    if not actual or actual >= cc.target_context_size:
        return token_count

    return int(token_count * cc.target_context_size / actual)


def validate_context_window(
    num_prompt_tokens: int, model_id: str | None = None
) -> None:
    """
    Validate that prompt token count does not exceed max context window.

    Raises HTTPException 400 if the prompt is too long.
    """
    max_ctx = get_max_context_window(model_id)
    if max_ctx and num_prompt_tokens > max_ctx:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Prompt too long: {num_prompt_tokens} tokens exceeds "
                f"max context window of {max_ctx} tokens"
            ),
        )


def init_server(
    model_dirs: str | list[str],
    scheduler_config=None,
    api_key: str | None = None,
    global_settings: object | None = None,
):
    """
    Initialize server with model directories for multi-model serving.

    Args:
        model_dirs: Path or list of paths to directories containing model subdirectories
        scheduler_config: Scheduler config for BatchedEngine
        api_key: API key for authentication (optional)
        global_settings: GlobalSettings instance (optional)

    Note:
        - Pinned models and default model are managed via admin page (model_settings.json)
        - Sampling parameters (max_tokens, temperature, etc.) are per-model settings

    Raises:
        ValueError: If model directory doesn't exist or no models found
    """
    from pathlib import Path

    from .model_settings import ModelSettingsManager

    # Store API key
    _server_state.api_key = api_key
    _server_state.global_settings = global_settings
    response_state_dir = None
    if global_settings:
        response_state_dir = (
            global_settings.cache.get_ssd_cache_dir(global_settings.base_path)
            / "response-state"
        )
    _server_state.responses_store = ResponseStore(state_dir=response_state_dir)

    # Refresh i18n with loaded language setting
    from .admin.routes import _refresh_i18n_globals

    _refresh_i18n_globals()

    # Initialize auth with persistent secret key
    if global_settings:
        if not global_settings.auth.secret_key:
            import secrets as _secrets

            global_settings.auth.secret_key = _secrets.token_hex(32)
            global_settings.save()
            logger.info("Generated and saved new auth secret key")
        from .admin.auth import init_auth

        init_auth(
            global_settings.auth.secret_key, lambda: _server_state.global_settings
        )

    # Configure CORS middleware from settings
    cors_origins = global_settings.server.cors_origins if global_settings else ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info(f"CORS origins: {cors_origins}")

    # Initialize model settings manager
    base_path = (
        Path(global_settings.base_path) if global_settings else Path.home() / ".omlx"
    )
    _server_state.settings_manager = ModelSettingsManager(base_path)

    # Get pinned models from settings file only (managed via admin page)
    pinned_models = _server_state.settings_manager.get_pinned_model_ids()

    # Get default model from settings file only (managed via admin page)
    settings_default = _server_state.settings_manager.get_default_model_id()

    # Load default sampling values from global settings
    # Per-model settings will override these via get_sampling_params()
    if global_settings and global_settings.sampling:
        _server_state.sampling = SamplingDefaults(
            max_context_window=global_settings.sampling.max_context_window,
            max_context_window_policy=getattr(
                global_settings.sampling, "max_context_window_policy", None
            ),
            max_tokens=global_settings.sampling.max_tokens,
            temperature=global_settings.sampling.temperature,
            top_p=global_settings.sampling.top_p,
            top_k=global_settings.sampling.top_k,
            repetition_penalty=getattr(
                global_settings.sampling, "repetition_penalty", 1.0
            ),
        )
    else:
        _server_state.sampling = SamplingDefaults()

    # Normalize model_dirs to list
    if isinstance(model_dirs, str):
        dir_list = [model_dirs]
    else:
        dir_list = list(model_dirs)
    if global_settings and hasattr(global_settings, "get_effective_model_dirs"):
        dir_list = [str(d) for d in global_settings.get_effective_model_dirs()]

    # Create directories if needed
    for md in dir_list:
        model_path = Path(md)
        if not model_path.exists():
            model_path.mkdir(parents=True, exist_ok=True)
            logger.warning(f"Model directory created (empty): {md}")

    # Create engine pool. The pool consults enforcer.get_final_ceiling()
    # for pre-load admission — wired up later in startup once the enforcer
    # is constructed.
    _server_state.engine_pool = EnginePool(
        scheduler_config=scheduler_config,
    )

    # Discover models (use pinned models from settings file)
    _server_state.engine_pool._settings_manager = _server_state.settings_manager
    _server_state.engine_pool.discover_models(dir_list, pinned_models)
    _server_state.engine_pool.apply_settings_overrides(_server_state.settings_manager)

    if _server_state.engine_pool.model_count == 0:
        logger.warning(
            f"No models found in {', '.join(dir_list)}. Add models to serve them."
        )

    # Set default model (from settings file, fallback to first model)
    available_models = _server_state.engine_pool.get_model_ids()
    if available_models:
        if settings_default:
            if settings_default in available_models:
                _server_state.default_model = settings_default
            else:
                logger.warning(
                    f"Default model '{settings_default}' not found, using first model"
                )
                _server_state.default_model = available_models[0]
        else:
            _server_state.default_model = available_models[0]
    else:
        _server_state.default_model = None

    # Reset server metrics for fresh start (with all-time persistence)
    stats_path = base_path / "stats.json"
    reset_server_metrics(stats_path=stats_path)

    logger.info(
        f"Server initialized with {_server_state.engine_pool.model_count} models"
    )
    if _server_state.default_model:
        logger.info(f"Default model: {_server_state.default_model}")
    else:
        logger.info("No default model (no models available)")
    if global_settings and getattr(global_settings, "memory", None):
        logger.info(
            f"Memory guard tier: {global_settings.memory.memory_guard_tier} "
            f"(guard {'on' if global_settings.memory.prefill_memory_guard else 'off'})"
        )
    logger.info(f"Default max tokens: {_server_state.sampling.max_tokens}")
    if api_key:
        logger.info("API key authentication: enabled")

    # Initialize HuggingFace downloader
    from .admin.hf_downloader import HFDownloader
    from .admin.routes import set_hf_downloader

    async def _refresh_models_after_download():
        """Re-discover models when a HuggingFace download completes."""
        if _server_state.engine_pool and _server_state.settings_manager:
            pinned = _server_state.settings_manager.get_pinned_model_ids()
            _server_state.engine_pool.discover_models(dir_list, pinned)
            _server_state.engine_pool.apply_settings_overrides(
                _server_state.settings_manager
            )
            logger.info("Model pool refreshed after download completion")

    _server_state.hf_downloader = HFDownloader(
        model_dir=dir_list[0],  # Downloads go to primary directory
        on_complete=_refresh_models_after_download,
    )
    set_hf_downloader(_server_state.hf_downloader)
    logger.info("HF Downloader initialized")

    # Initialize ModelScope downloader (optional - requires modelscope SDK)
    try:
        from .admin.ms_downloader import MS_SDK_AVAILABLE, MSDownloader

        if MS_SDK_AVAILABLE:
            from .admin.routes import set_ms_downloader

            _server_state.ms_downloader = MSDownloader(
                model_dir=dir_list[0],
                on_complete=_refresh_models_after_download,
            )
            set_ms_downloader(_server_state.ms_downloader)
            logger.info("ModelScope Downloader initialized")
        else:
            logger.info("ModelScope SDK not installed, MS downloader disabled")
    except ImportError:
        logger.info("ModelScope support not available")

    # Initialize oQ Quantizer
    from .admin.oq_manager import OQManager
    from .admin.routes import set_oq_manager

    _server_state.oq_manager = OQManager(
        model_dirs=[str(d) for d in dir_list],
        on_complete=_refresh_models_after_download,
    )
    set_oq_manager(_server_state.oq_manager)
    logger.info("oQ Quantizer initialized")

    # Initialize HuggingFace uploader
    from .admin.hf_uploader import HFUploader
    from .admin.routes import set_hf_uploader

    _server_state.hf_uploader = HFUploader(
        model_dirs=[str(d) for d in dir_list],
    )
    set_hf_uploader(_server_state.hf_uploader)
    logger.info("HF Uploader initialized")


_KEEPALIVE_SENTINEL = object()

_KEEPALIVE_COMMENT = ": keep-alive\n\n"
# The delta carries "role":"assistant" because this frame is the FIRST event
# of every stream and some accumulators type the whole stream from the first
# chunk's role: LangChain.js builds a generic ChatMessageChunk when it is
# absent, and merging the real chunks into it silently discards all
# tool_call_chunks (#2074, hits n8n AI Agent workflows). Cloud APIs open every
# stream with a role chunk, so clients rely on it; the OpenAI SDKs, openai-go,
# and LangChain all tolerate the duplicate role in later chunks.
_KEEPALIVE_CHAT_CHUNK = (
    'data: {"id":"chatcmpl-keepalive","object":"chat.completion.chunk",'
    '"created":0,"model":"keepalive",'
    '"choices":[{"index":0,"delta":{"role":"assistant","content":""},'
    '"finish_reason":null}]}\n\n'
)
_KEEPALIVE_COMPLETION_CHUNK = (
    'data: {"id":"cmpl-keepalive","object":"text_completion","created":0,'
    '"model":"keepalive",'
    '"choices":[{"index":0,"text":"","logprobs":null,"finish_reason":null}]}\n\n'
)
_KEEPALIVE_ANTHROPIC_PING = 'event: ping\ndata: {"type":"ping"}\n\n'


def _resolve_keepalive(protocol: str) -> Optional[str]:
    """Pick a wire-level keepalive frame for the given API protocol.

    Returns None when the configured mode disables keepalive for this protocol.
    Modes: "chunk" (default, protocol-aware), "comment" (legacy SSE comment),
    "off" (no keepalive). Some clients (e.g. OpenClaw / WorkBuddy) cannot parse
    SSE comment lines, so the chunk mode emits valid no-op events instead.
    """
    global_settings = _server_state.global_settings
    mode = "chunk"
    if global_settings is not None:
        mode = getattr(global_settings.server, "sse_keepalive_mode", "chunk")
    if mode == "off":
        return None
    if mode == "comment":
        return _KEEPALIVE_COMMENT
    if protocol == "openai_chat":
        return _KEEPALIVE_CHAT_CHUNK
    if protocol == "openai_completion":
        return _KEEPALIVE_COMPLETION_CHUNK
    if protocol == "anthropic":
        return _KEEPALIVE_ANTHROPIC_PING
    if protocol == "openai_responses":
        return None
    return None


def _chat_keepalive_chunk(response_id: str) -> str:
    """Keepalive frame that shares the stream's completion id.

    The static ``_KEEPALIVE_CHAT_CHUNK`` carries a sentinel id
    (``chatcmpl-keepalive``) that differs from the real completion chunks.
    Strict OpenAI stream accumulators (e.g. the official ``openai-go`` SDK)
    assume every chunk in one streamed completion shares a single ``id``: they
    latch the first chunk's id and silently drop later chunks whose id differs,
    discarding the real ``tool_calls``/``finish_reason``/``usage``. Emitting the
    keepalive with the stream's own ``response_id`` makes it a true no-op for
    those clients while remaining a parseable data event for clients that can't
    handle SSE comment lines.

    The delta must also carry ``"role":"assistant"`` — see the comment on
    ``_KEEPALIVE_CHAT_CHUNK`` (accumulators that type the stream from the
    first chunk's role drop tool_call_chunks without it, #2074).
    """
    return (
        'data: {"id":"' + response_id + '","object":"chat.completion.chunk",'
        '"created":0,"model":"keepalive",'
        '"choices":[{"index":0,"delta":{"role":"assistant","content":""},'
        '"finish_reason":null}]}\n\n'
    )


async def _safe_anext(ait):
    """Wrapper for __anext__ that converts StopAsyncIteration to a sentinel.

    StopAsyncIteration cannot propagate through asyncio.Task (raises RuntimeError),
    so we catch it here and return a sentinel value instead.
    """
    try:
        return await ait.__anext__()
    except StopAsyncIteration:
        return _KEEPALIVE_SENTINEL


async def _with_sse_keepalive(
    generator: AsyncIterator[str],
    http_request: Optional["FastAPIRequest"] = None,
    interval: float = 10.0,
    disconnect_poll: float = 2.0,
    keepalive_chunk: Optional[str] = _KEEPALIVE_COMMENT,
) -> AsyncIterator[str]:
    """Wrap an SSE generator to send periodic keepalive frames.

    During long prefill (e.g. 90k tokens), no SSE events are emitted,
    causing clients with read timeouts (like Claude Code) to disconnect.
    This wrapper periodically yields a keepalive frame to hold the
    connection open. The frame format depends on caller-supplied
    keepalive_chunk: a legacy SSE comment, a protocol-aware no-op event,
    or None to disable emission entirely.

    When http_request is provided, also polls for client disconnect
    between prefill steps. This detects cancellation during long prefills
    where uvicorn's ASGI disconnect message is not delivered until after
    the generator yields.
    """
    ait = generator.__aiter__()
    task = None
    keepalive_elapsed = 0.0

    # Send initial keepalive immediately so clients with short read
    # timeouts (e.g. openclaw ~15s) don't disconnect during prefill.
    if keepalive_chunk is not None:
        yield keepalive_chunk

    try:
        while True:
            task = asyncio.ensure_future(_safe_anext(ait))
            keepalive_elapsed = 0.0
            while not task.done():
                # Use shorter poll interval for disconnect detection,
                # accumulate time for keepalive emission
                wait_time = disconnect_poll if http_request else interval
                done, _ = await asyncio.wait({task}, timeout=wait_time)
                if done:
                    break
                # Check for client disconnect
                if http_request is not None:
                    try:
                        disconnected = await http_request.is_disconnected()
                        if disconnected:
                            logger.info(
                                "Client disconnected during streaming (is_disconnected), cancelling"
                            )
                            task.cancel()
                            try:
                                await task
                            except (asyncio.CancelledError, StopAsyncIteration):
                                pass
                            return
                    except Exception as e:
                        logger.debug(f"is_disconnected() check failed: {e}")
                        pass  # is_disconnected() can fail if scope is already closed
                # Send keepalive at the configured interval
                keepalive_elapsed += wait_time
                if keepalive_elapsed >= interval:
                    keepalive_elapsed = 0.0
                    if keepalive_chunk is not None:
                        yield keepalive_chunk
            if task.done():
                try:
                    result = task.result()
                except Exception as e:
                    if isinstance(e, PrefillMemoryExceededError):
                        logger.warning(f"SSE generator prefill rejected: {e}")
                        error_data = _prefill_memory_openai_error_body(e)
                    else:
                        logger.error(f"SSE generator error: {e}")
                        error_data = {
                            "error": {"message": str(e), "type": "server_error"}
                        }
                    yield f"data: {json.dumps(error_data)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                if result is _KEEPALIVE_SENTINEL:
                    return
                yield result
    finally:
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, StopAsyncIteration):
                pass
        if hasattr(ait, "aclose"):
            await ait.aclose()


async def _run_with_disconnect_guard(
    http_request: FastAPIRequest,
    coro,
    poll_interval: float = 1.0,
):
    """Run a coroutine with client disconnect detection.

    For non-streaming requests, FastAPI/uvicorn does NOT automatically cancel
    the handler coroutine when a client disconnects. This helper polls
    is_disconnected() periodically and cancels the task on disconnect,
    which triggers CancelledError -> abort_request() in EngineCore.generate()
    to free scheduler/GPU resources.
    """
    task = asyncio.create_task(coro)
    while not task.done():
        done, _ = await asyncio.wait({task}, timeout=poll_interval)
        if done:
            break
        if await http_request.is_disconnected():
            logger.info("Client disconnected, cancelling generation task")
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return None
    return task.result()


async def _with_json_keepalive(
    http_request: FastAPIRequest,
    coro,
    interval: float = 10.0,
    disconnect_poll: float = 2.0,
) -> AsyncIterator[str]:
    """Wrap a coroutine to send keepalive spaces while waiting for completion.

    For non-streaming requests, the HTTP response body is buffered until
    generation finishes, causing client read timeouts on long prefills.
    This wrapper uses StreamingResponse to send space characters as
    keepalive. JSON parsers ignore leading whitespace, so the final
    response parses normally.
    """
    task = asyncio.ensure_future(coro)
    keepalive_elapsed = 0.0

    yield " "

    try:
        while not task.done():
            done, _ = await asyncio.wait({task}, timeout=disconnect_poll)
            if done:
                break
            if http_request is not None:
                try:
                    disconnected = await http_request.is_disconnected()
                    if disconnected:
                        logger.info(
                            "Client disconnected during non-streaming response, cancelling"
                        )
                        task.cancel()
                        try:
                            await task
                        except (asyncio.CancelledError, StopAsyncIteration):
                            pass
                        return
                except Exception:
                    pass
            keepalive_elapsed += disconnect_poll
            if keepalive_elapsed >= interval:
                keepalive_elapsed = 0.0
                yield " "
        try:
            result = task.result()
        except PrefillMemoryExceededError as e:
            logger.warning(f"JSON keepalive prefill rejected: {e}")
            yield json.dumps(_prefill_memory_openai_error_body(e))
            return
        if result is not None:
            yield result
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, StopAsyncIteration):
                pass


@app.get("/health")
async def health(response: Response):
    """Health check endpoint.

    Answers 503 with status "loading" while the startup pinned-model
    preload is still running: the port is already bound (liveness for
    watchdogs, #2184) but the server is not ready to serve those models.
    """
    mcp_info = None
    if _server_state.mcp_manager is not None:
        connected = sum(
            1
            for s in _server_state.mcp_manager.get_server_status()
            if s.state.value == "connected"
        )
        total = len(_server_state.mcp_manager.get_server_status())
        mcp_info = {
            "enabled": True,
            "servers_connected": connected,
            "servers_total": total,
            "tools_available": len(_server_state.mcp_manager.get_all_tools()),
        }

    pool_status = None
    if _server_state.engine_pool is not None:
        enforcer = _server_state.process_memory_enforcer
        ceiling = 0
        if enforcer is not None:
            try:
                ceiling = enforcer.get_final_ceiling()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Health memory ceiling unavailable: %s", exc)
        pool_status = {
            "model_count": _server_state.engine_pool.model_count,
            "loaded_count": _server_state.engine_pool.loaded_model_count,
            "final_ceiling": ceiling,
            "current_model_memory": _server_state.engine_pool.current_model_memory,
        }

    loading = not _server_state.pinned_preload_complete
    if loading:
        response.status_code = 503
    return {
        "status": "loading" if loading else "healthy",
        "default_model": _server_state.default_model,
        "engine_pool": pool_status,
        "mcp": mcp_info,
    }


@app.get("/api/status")
async def server_status(_: bool = Depends(verify_api_key)):
    """Lightweight status endpoint for external tool polling (statuslines, scripts)."""
    from .model_discovery import format_size
    from .server_metrics import get_server_metrics

    metrics = get_server_metrics()
    snapshot = metrics.get_snapshot()

    pool = _server_state.engine_pool

    models_discovered = 0
    models_loaded = 0
    models_loading = 0
    loaded_models = []
    model_memory_used = 0
    model_memory_max = None

    if pool is not None:
        models_discovered = pool.model_count
        models_loaded = pool.loaded_model_count
        loaded_models = pool.get_loaded_model_ids()
        model_memory_used = pool.current_model_memory
        enforcer = _server_state.process_memory_enforcer
        if enforcer is not None:
            try:
                model_memory_max = enforcer.get_final_ceiling()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Status memory ceiling unavailable: %s", exc)
        for entry in pool._entries.values():
            if entry.is_loading:
                models_loading += 1

    # Aggregate active/waiting requests across all loaded engines
    active_requests = 0
    waiting_requests = 0
    if pool is not None:
        for entry in pool._entries.values():
            engine = entry.engine
            if engine is None:
                continue
            async_core = getattr(engine, "_engine", None)
            if async_core is None:
                continue
            core = getattr(async_core, "engine", None)
            if core is None:
                continue
            active_requests += len(getattr(core, "_output_collectors", {}))
            sched = getattr(core, "scheduler", None)
            if sched is not None:
                waiting_requests += len(getattr(sched, "waiting", []))

    return {
        "status": "ok",
        "version": __version__,
        "uptime_seconds": snapshot["uptime_seconds"],
        "models_discovered": models_discovered,
        "models_loaded": models_loaded,
        "models_loading": models_loading,
        "default_model": _server_state.default_model,
        "loaded_models": loaded_models,
        "total_requests": snapshot["total_requests"],
        "active_requests": active_requests,
        "waiting_requests": waiting_requests,
        "total_prompt_tokens": snapshot["total_prompt_tokens"],
        "total_completion_tokens": snapshot["total_completion_tokens"],
        "total_cached_tokens": snapshot["total_cached_tokens"],
        "cache_efficiency": snapshot["cache_efficiency"],
        "avg_prefill_tps": snapshot["avg_prefill_tps"],
        "avg_generation_tps": snapshot["avg_generation_tps"],
        "model_memory_used": model_memory_used,
        "model_memory_max": model_memory_max,
        "model_memory_used_formatted": (
            format_size(model_memory_used) if model_memory_used else "0B"
        ),
        "model_memory_max_formatted": (
            format_size(model_memory_max) if model_memory_max else "unlimited"
        ),
    }


def _markitdown_virtual_model_status() -> dict:
    return {
        "id": MARKITDOWN_MODEL_ID,
        "model_path": "builtin://markitdown",
        "loaded": True,
        "is_loading": False,
        "loading_started_at": None,
        "estimated_size": 0,
        "actual_size": 0,
        "pinned": False,
        "engine_type": "markitdown",
        "model_type": "markitdown",
        "config_model_type": "markitdown",
        "thinking_default": None,
        "preserve_thinking_default": None,
        "source_type": "builtin",
        "source_repo_id": None,
        "last_access": None,
    }


def _markitdown_is_visible() -> bool:
    return markitdown_model_visible(_server_state.global_settings)


def _with_markitdown_status(status: dict) -> dict:
    if not _markitdown_is_visible():
        return status

    augmented = dict(status)
    models = list(augmented.get("models", []))
    if not any(m.get("id") == MARKITDOWN_MODEL_ID for m in models):
        models.append(_markitdown_virtual_model_status())
    augmented["models"] = models
    augmented["model_count"] = len(models)
    augmented["loaded_count"] = sum(1 for m in models if m.get("loaded"))
    return augmented


def _with_exposed_profile_status(status: dict) -> dict:
    settings_manager = _server_state.settings_manager
    if settings_manager is None:
        return status

    list_profiles = getattr(settings_manager, "list_exposed_profile_models", None)
    if not callable(list_profiles):
        return status

    augmented = dict(status)
    models = [dict(m) for m in augmented.get("models", [])]
    by_id = {m.get("id"): m for m in models}
    existing_ids = set(by_id)
    for profile in list_profiles():
        source_model_id = profile.get("source_model_id")
        profile_model_id = profile.get("model_id")
        if (
            not source_model_id
            or not profile_model_id
            or source_model_id not in by_id
            or profile_model_id in existing_ids
        ):
            continue
        profile_status = dict(by_id[source_model_id])
        profile_status.update(
            {
                "id": profile_model_id,
                "source_model_id": source_model_id,
                "profile_name": profile.get("name"),
                "profile_api_name": profile.get("api_name"),
                "profile_display_name": profile.get("display_name"),
            }
        )
        models.append(profile_status)
        existing_ids.add(profile_model_id)

    augmented["models"] = models
    augmented["model_count"] = len(models)
    augmented["loaded_count"] = sum(1 for m in models if m.get("loaded"))
    return augmented


async def _preprocess_markitdown_files_for_llm(
    request: ChatCompletionRequest,
) -> ChatCompletionRequest:
    if not request_has_file_parts(request.messages):
        return request

    try:
        messages = await preprocess_markitdown_file_parts_async(
            request.messages,
            global_settings=_server_state.global_settings,
            engine_pool=_server_state.engine_pool,
            settings_manager=_server_state.settings_manager,
            get_sampling_params=get_sampling_params,
            fail_when_disabled=True,
            allow_missing_historical_files=True,
        )
    except MarkItDownRequestError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return request.model_copy(update={"messages": messages})


def _build_markitdown_chat_response(
    request: ChatCompletionRequest,
    markdown: str,
) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        model=request.model,
        choices=[
            ChatCompletionChoice(
                message=AssistantMessage(content=markdown),
                finish_reason="stop",
            )
        ],
        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
    )


async def _stream_markitdown_chat_response(
    request: ChatCompletionRequest,
    markdown_chunks: AsyncIterator[str],
    response_id: str | None = None,
) -> AsyncIterator[str]:
    response_id = response_id or f"chatcmpl-{uuid.uuid4().hex[:8]}"
    role_chunk = ChatCompletionChunk(
        id=response_id,
        model=request.model,
        choices=[
            ChatCompletionChunkChoice(
                delta=ChatCompletionChunkDelta(role="assistant"),
            )
        ],
    )
    yield f"data: {role_chunk.model_dump_json(exclude_none=True)}\n\n"

    emitted = False
    async for markdown in markdown_chunks:
        if not markdown:
            continue
        emitted = True
        content_chunk = ChatCompletionChunk(
            id=response_id,
            model=request.model,
            choices=[
                ChatCompletionChunkChoice(
                    delta=ChatCompletionChunkDelta(content=markdown),
                )
            ],
        )
        yield f"data: {content_chunk.model_dump_json(exclude_none=True)}\n\n"

    if not emitted:
        raise MarkItDownRequestError(
            "No text or supported file content found for MarkItDown.",
            status_code=400,
        )

    final_chunk = ChatCompletionChunk(
        id=response_id,
        model=request.model,
        choices=[
            ChatCompletionChunkChoice(
                delta=ChatCompletionChunkDelta(),
                finish_reason="stop",
            )
        ],
    )
    yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n"

    if request.stream_options and request.stream_options.include_usage:
        usage_chunk = ChatCompletionChunk(
            id=response_id,
            model=request.model,
            choices=[],
            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )
        yield f"data: {usage_chunk.model_dump_json(exclude_none=True)}\n\n"

    yield "data: [DONE]\n\n"


async def _create_markitdown_chat_completion(
    request: ChatCompletionRequest,
    http_request: FastAPIRequest,
):
    if not _markitdown_is_visible():
        raise HTTPException(
            status_code=404,
            detail=f"Model not found: {MARKITDOWN_MODEL_ID}",
        )

    if request.stream:
        response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        keepalive = _resolve_keepalive("openai_chat")
        if keepalive == _KEEPALIVE_CHAT_CHUNK:
            keepalive = _chat_keepalive_chunk(response_id)
        markdown_chunks = stream_messages_to_markdown_async(
            request.messages,
            global_settings=_server_state.global_settings,
            engine_pool=_server_state.engine_pool,
            settings_manager=_server_state.settings_manager,
            get_sampling_params=get_sampling_params,
            latest_user_only=True,
        )
        return StreamingResponse(
            _with_sse_keepalive(
                _stream_markitdown_chat_response(
                    request,
                    markdown_chunks,
                    response_id=response_id,
                ),
                http_request=http_request,
                keepalive_chunk=keepalive,
            ),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )

    async def _build_markitdown_completion():
        try:
            markdown = await convert_messages_to_markdown_async(
                request.messages,
                global_settings=_server_state.global_settings,
                engine_pool=_server_state.engine_pool,
                settings_manager=_server_state.settings_manager,
                get_sampling_params=get_sampling_params,
                latest_user_only=True,
            )
        except MarkItDownRequestError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        if not markdown:
            raise HTTPException(
                status_code=400,
                detail="No text or supported file content found for MarkItDown.",
            )

        logger.info("MarkItDown completion converted request to markdown")
        return _build_markitdown_chat_response(
            request,
            markdown,
        ).model_dump_json(exclude_none=True)

    return StreamingResponse(
        _with_json_keepalive(http_request, _build_markitdown_completion()),
        media_type="application/json",
    )


@app.get("/v1/models")
async def list_models(_: bool = Depends(verify_api_key)) -> ModelsResponse:
    """List all available models with load status."""
    models = []
    favorite_ids: set[str] = set()

    if _server_state.engine_pool is not None:
        status = _server_state.engine_pool.get_status()
        settings_manager = _server_state.settings_manager

        hide_helpers = bool(
            _server_state.global_settings is not None
            and _server_state.global_settings.model.hide_helper_models
        )
        # Set of draft-model references (paths / repo ids) pointed at by other
        # models' speculative settings — used to flag "helper" drafters that
        # only differ from a chat model by being referenced elsewhere.
        referenced_drafts: set[str] = set()
        if hide_helpers and settings_manager:
            for _ms in settings_manager.get_all_settings().values():
                for ref in (
                    _ms.specprefill_draft_model,
                    _ms.dflash_draft_model,
                    _ms.vlm_mtp_draft_model,
                ):
                    if ref:
                        referenced_drafts.add(ref)

        excluded_model_ids: set[str] = set()
        for m in status["models"]:
            model_id = m["id"]
            display_id = model_id
            ms = None
            if settings_manager:
                ms = settings_manager.get_settings(model_id)
                if ms.model_alias:
                    display_id = ms.model_alias
            # Per-model hide: user-selected, always applied.
            is_hidden = ms is not None and ms.is_hidden
            # Global helper hide: skip drafters when the toggle is on. A model
            # is a drafter if intrinsically flagged at discovery (config marker)
            # or referenced as another model's draft.
            is_hidden_helper = hide_helpers and (
                m.get("is_helper")
                or model_id in referenced_drafts
                or m.get("model_path") in referenced_drafts
                or (m.get("source_repo_id") in referenced_drafts)
            )
            if is_hidden or is_hidden_helper:
                excluded_model_ids.add(model_id)
                continue
            if ms is not None and ms.is_favorite:
                favorite_ids.add(display_id)
            models.append(
                ModelInfo(
                    id=display_id,
                    owned_by="omlx",
                    max_model_len=get_max_context_window(model_id),
                )
            )
        if settings_manager:
            physical_ids = {m["id"] for m in status["models"]}
            existing_ids = {m.id for m in models}
            for profile in settings_manager.list_exposed_profile_models():
                source_model_id = profile["source_model_id"]
                profile_model_id = profile["model_id"]
                if (
                    source_model_id not in physical_ids
                    or source_model_id in excluded_model_ids
                    or profile_model_id in existing_ids
                ):
                    continue
                models.append(
                    ModelInfo(
                        id=profile_model_id,
                        owned_by="omlx",
                        max_model_len=get_max_context_window(profile_model_id),
                    )
                )
                existing_ids.add(profile_model_id)

    if _markitdown_is_visible() and not any(
        m.id == MARKITDOWN_MODEL_ID for m in models
    ):
        models.append(ModelInfo(id=MARKITDOWN_MODEL_ID, owned_by="omlx"))

    # Favorites first; stable sort keeps alphabetical order within groups.
    if favorite_ids:
        models.sort(key=lambda m: m.id not in favorite_ids)

    return ModelsResponse(data=models)


@app.get("/v1/models/status")
async def list_models_status(_: bool = Depends(verify_api_key)):
    """
    List all available models with detailed status.

    Extended endpoint that provides more information than /v1/models.
    """
    if _server_state.engine_pool is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    status = _with_exposed_profile_status(
        _with_markitdown_status(_server_state.engine_pool.get_status())
    )
    for m in status["models"]:
        model_id = m["id"]
        if is_markitdown_model(model_id):
            m["max_context_window"] = None
            m["max_tokens"] = None
            continue

        m["max_context_window"] = get_max_context_window(model_id)
        source_model_id = m.get("source_model_id") or model_id

        # Resolve effective max_tokens: model setting > global default
        max_tokens = _server_state.sampling.max_tokens
        if _server_state.settings_manager:
            sm = _server_state.settings_manager
            if hasattr(sm, "get_settings_for_request"):
                ms = sm.get_settings_for_request(
                    model_id,
                    resolved_model_id=source_model_id,
                )
            else:
                ms = sm.get_settings(source_model_id)
            base_ms = sm.get_settings(source_model_id)
            if base_ms and base_ms.model_alias and source_model_id == model_id:
                m["model_alias"] = base_ms.model_alias
            if ms and ms.max_tokens is not None:
                max_tokens = ms.max_tokens
        m["max_tokens"] = max_tokens
    return status


@app.post("/v1/models/{model_id}/unload")
async def unload_model(model_id: str, _: bool = Depends(verify_api_key)):
    """Manually unload a model from memory."""
    if _server_state.engine_pool is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    entry = _server_state.engine_pool.get_entry(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    if entry.engine is None:
        raise HTTPException(status_code=400, detail=f"Model not loaded: {model_id}")

    await _server_state.engine_pool._unload_engine(model_id)
    return {"status": "ok", "model_id": model_id}


@app.post("/v1/models/{model_id}/load")
async def load_model_public(model_id: str, _: bool = Depends(verify_api_key)):
    """Load a discovered model into memory. Blocks until loading completes."""
    if _server_state.engine_pool is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    entry = _server_state.engine_pool.get_entry(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    if entry.engine is not None:
        return {
            "status": "ok",
            "model_id": model_id,
            "message": f"Already loaded: {model_id}",
        }

    try:
        await _server_state.engine_pool.get_engine(model_id)
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ModelTooLargeError as e:
        raise HTTPException(status_code=507, detail=str(e)) from e
    except InsufficientMemoryError as e:
        raise HTTPException(status_code=507, detail=str(e)) from e
    except ModelUnavailableError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except ModelLoadingError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except ModelBusyError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except EnginePoolError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return {"status": "ok", "model_id": model_id, "message": f"Loaded {model_id}"}


# =============================================================================
# Embeddings Endpoint
# =============================================================================


@app.post("/v1/embeddings")
async def create_embeddings(
    request: EmbeddingRequest,
    http_request: FastAPIRequest,
    _: bool = Depends(verify_api_key),
):
    """
    Create embeddings for input text(s).

    OpenAI-compatible endpoint for generating text embeddings.

    Example request:
    ```json
    {
        "model": "all-MiniLM-L6-v2",
        "input": ["Hello, world!", "How are you?"],
        "encoding_format": "float"
    }
    ```

    Supports:
    - Single text or list of texts
    - float or base64 encoding format
    - Optional dimension reduction (with renormalization)
    """
    oq_manager = getattr(_server_state, "oq_manager", None)
    if oq_manager and oq_manager.is_quantizing:
        raise HTTPException(
            status_code=503,
            detail="Server is busy with oQ quantization. Please try again after quantization completes.",
        )

    # Validate the model up front (resolves + loads + type-checks) so a bad
    # model still 400/404s before we start the streaming response. The actual
    # eviction-proof lease is taken inside _build_embeddings, which is where
    # the engine is used (the StreamingResponse runs that coroutine later).
    await get_embedding_engine(request.model)

    if request.items is not None:
        embedding_inputs = normalize_embedding_items(request.items)
    elif request.input is not None:
        embedding_inputs = normalize_input(request.input)
    else:
        embedding_inputs = []

    if not embedding_inputs:
        raise HTTPException(status_code=400, detail="Input cannot be empty")

    max_length = get_embedding_max_length(request.model, request.max_length)

    async def _build_embeddings():
        start_time = time.perf_counter()
        try:
            async with acquire_embedding_engine(request.model) as engine:
                output = await engine.embed(
                    embedding_inputs,
                    max_length=max_length,
                    truncation=request.truncation,
                )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except TypeError as e:
            raise HTTPException(status_code=400, detail=str(e))

        elapsed = time.perf_counter() - start_time
        logger.info(
            f"Embedding: {len(embedding_inputs)} inputs, {output.dimensions} dims, "
            f"{output.total_tokens} tokens, max_length={max_length}, "
            f"truncation={request.truncation} in {elapsed:.3f}s"
        )
        get_server_metrics().record_request_complete(
            prompt_tokens=output.total_tokens,
            completion_tokens=0,
            cached_tokens=0,
            prefill_duration=elapsed,
            model_id=resolve_model_id(request.model) or request.model,
        )

        data = []
        for i, embedding in enumerate(output.embeddings):
            if request.dimensions and request.dimensions < len(embedding):
                embedding = truncate_embedding(embedding, request.dimensions)

            if request.encoding_format == "base64":
                formatted_embedding = encode_embedding_base64(embedding)
            else:
                formatted_embedding = embedding

            data.append(
                EmbeddingData(
                    index=i,
                    embedding=formatted_embedding,
                )
            )

        return EmbeddingResponse(
            data=data,
            model=request.model,
            usage=EmbeddingUsage(
                prompt_tokens=output.total_tokens,
                total_tokens=output.total_tokens,
            ),
        ).model_dump_json()

    return StreamingResponse(
        _with_json_keepalive(http_request, _build_embeddings()),
        media_type="application/json",
    )


# =============================================================================
# Rerank Endpoint
# =============================================================================


def normalize_documents(documents: list[str] | list[dict]) -> list[str]:
    """Normalize document input to list of strings."""
    result = []
    for doc in documents:
        if isinstance(doc, str):
            result.append(doc)
        elif isinstance(doc, dict):
            result.append(doc.get("text", ""))
        else:
            result.append(str(doc))
    return result


@app.post("/v1/rerank")
async def create_rerank(
    request: RerankRequest,
    _: bool = Depends(verify_api_key),
) -> RerankResponse:
    """
    Rerank documents by relevance to a query.

    Cohere/Jina-compatible endpoint for document reranking.

    Example request:
    ```json
    {
        "model": "bge-reranker-v2-m3",
        "query": "What is machine learning?",
        "documents": [
            "Machine learning is a subset of AI...",
            "The weather today is sunny...",
            "Deep learning uses neural networks..."
        ],
        "top_n": 2
    }
    ```

    Supports:
    - String documents or dict documents with 'text' field
    - Optional top_n to limit results
    - Optional return_documents to include document text in response
    """
    if _server_state.oq_manager and _server_state.oq_manager.is_quantizing:
        raise HTTPException(
            status_code=503,
            detail="Server is busy with oQ quantization. Please try again after quantization completes.",
        )

    # Validate the model up front (resolves + loads + type-checks). The
    # eviction-proof lease is held only around the actual rerank() call below.
    await get_reranker_engine(request.model)

    # Preserve original structure for the engine (multimodal rerankers need
    # dicts with 'image'), but keep a normalized text view for logging and
    # emptiness checks.
    documents_raw = request.documents
    documents_text = normalize_documents(documents_raw)

    if not documents_text:
        raise HTTPException(status_code=400, detail="Documents cannot be empty")

    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Perform reranking
    start_time = time.perf_counter()

    async with acquire_reranker_engine(request.model) as engine:
        output = await engine.rerank(
            query=request.query,
            documents=documents_raw,
            top_n=request.top_n,
        )

    elapsed = time.perf_counter() - start_time
    logger.info(
        f"Rerank: {len(documents_raw)} docs, "
        f"{output.total_tokens} tokens in {elapsed:.3f}s"
    )
    get_server_metrics().record_request_complete(
        prompt_tokens=output.total_tokens,
        completion_tokens=0,
        cached_tokens=0,
        prefill_duration=elapsed,
        model_id=resolve_model_id(request.model) or request.model,
    )

    # Format response - results sorted by score (descending). Strings wrap
    # into {"text": "..."}; dict inputs pass through as-is so multimodal
    # callers get their original 'image' back.
    results = []
    for idx in output.indices:
        if request.return_documents:
            orig = documents_raw[idx]
            display_doc = orig if isinstance(orig, dict) else {"text": orig}
        else:
            display_doc = None
        result = RerankResult(
            index=idx,
            relevance_score=output.scores[idx],
            document=display_doc,
        )
        results.append(result)

    return RerankResponse(
        results=results,
        model=request.model,
        usage=RerankUsage(total_tokens=output.total_tokens),
    )


# =============================================================================
# Completion Endpoints
# =============================================================================


@app.post("/v1/completions")
async def create_completion(
    request: CompletionRequest,
    http_request: FastAPIRequest,
    _: bool = Depends(verify_api_key),
):
    """Create a text completion."""
    if _server_state.oq_manager and _server_state.oq_manager.is_quantizing:
        raise HTTPException(
            status_code=503,
            detail="Server is busy with oQ quantization. Please try again after quantization completes.",
        )
    lease = _LLMEngineLease()
    try:
        load_start = time.perf_counter()
        engine = await get_engine_for_model(request.model, lease=lease)
        model_load_duration = time.perf_counter() - load_start

        # Handle single prompt or list of prompts
        prompts = (
            request.prompt if isinstance(request.prompt, list) else [request.prompt]
        )

        # Validate context window for each prompt
        prompt_token_ids_by_prompt = []
        for prompt in prompts:
            prompt_token_ids = list(engine.tokenizer.encode(prompt))
            prompt_token_ids_by_prompt.append(prompt_token_ids)
            validate_context_window(len(prompt_token_ids), request.model)

        # Pre-flight prefill memory guard — see create_chat_completion for
        # the reason this must precede any StreamingResponse return.
        # Thread the client-provided X-Request-ID when present so the 400
        # log line and the FastAPI handler trace correlate with whatever
        # the client is using on its side.
        upstream_request_id = http_request.headers.get("x-request-id")
        await _raise_if_llm_lease_abort_requested(lease)
        for prompt in prompts:
            await engine.preflight_completion(prompt, request_id=upstream_request_id)
        await _raise_if_llm_lease_abort_requested(lease)

        if request.stream:
            return StreamingResponse(
                _release_after_stream(
                    _with_sse_keepalive(
                        stream_completion(
                            engine,
                            prompts[0],
                            request,
                            model_load_duration=model_load_duration,
                            prompt_token_ids=prompt_token_ids_by_prompt[0],
                        ),
                        http_request=http_request,
                        keepalive_chunk=_resolve_keepalive("openai_completion"),
                    ),
                    lease,
                ),
                media_type="text/event-stream",
                headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
            )

        # Non-streaming response with keepalive during prefill
        async def _build_completion():
            await _raise_if_llm_lease_abort_requested(lease)
            start_time = time.perf_counter()
            choices = []
            total_completion_tokens = 0
            total_prompt_tokens = 0
            total_cached_tokens = 0

            (
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                min_p,
                presence_penalty,
                frequency_penalty,
                max_tokens,
                xtc_probability,
                xtc_threshold,
            ) = get_sampling_params(
                request.temperature,
                request.top_p,
                request.model,
                req_top_k=getattr(request, "top_k", None),
                req_repetition_penalty=getattr(request, "repetition_penalty", None),
                req_min_p=getattr(request, "min_p", None),
                req_presence_penalty=getattr(request, "presence_penalty", None),
                req_frequency_penalty=getattr(request, "frequency_penalty", None),
                req_max_tokens=request.max_tokens,
                req_xtc_probability=getattr(request, "xtc_probability", None),
                req_xtc_threshold=getattr(request, "xtc_threshold", None),
            )

            gen_kwargs = {}
            thinking_budget = _resolve_thinking_budget(request, request.model)
            if thinking_budget is not None:
                gen_kwargs["thinking_budget"] = thinking_budget

            for i, prompt in enumerate(prompts):
                output = await engine.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    repetition_penalty=repetition_penalty,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    xtc_probability=xtc_probability,
                    xtc_threshold=xtc_threshold,
                    stop=request.stop,
                    seed=request.seed,
                    **gen_kwargs,
                )

                choices.append(
                    CompletionChoice(
                        index=i,
                        text=output.text,
                        finish_reason=output.finish_reason,
                    )
                )
                total_completion_tokens += output.completion_tokens
                total_prompt_tokens += output.prompt_tokens
                total_cached_tokens += output.cached_tokens

            elapsed = time.perf_counter() - start_time
            tokens_per_sec = total_completion_tokens / elapsed if elapsed > 0 else 0
            logger.info(
                f"Completion: {total_completion_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s), prompt: {total_prompt_tokens}"
            )

            get_server_metrics().record_request_complete(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                cached_tokens=total_cached_tokens,
                generation_duration=elapsed,
                model_id=resolve_model_id(request.model) or request.model,
            )

            return CompletionResponse(
                model=request.model,
                choices=choices,
                usage=Usage(
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=total_completion_tokens,
                    total_tokens=total_prompt_tokens + total_completion_tokens,
                    prompt_tokens_details=PromptTokensDetails(
                        cached_tokens=total_cached_tokens,
                    ),
                    model_load_duration=(
                        round(model_load_duration, 2)
                        if model_load_duration > 1.0
                        else None
                    ),
                    total_time=round(elapsed, 2),
                ),
            ).model_dump_json(exclude_none=True)

        return StreamingResponse(
            _release_after_stream(
                _with_json_keepalive(http_request, _build_completion()),
                lease,
            ),
            media_type="application/json",
        )
    except BaseException:
        await lease.release()
        raise


@app.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    http_request: FastAPIRequest,
    _: bool = Depends(verify_api_key),
):
    """
    Create a chat completion.

    Structured output (JSON mode):
    ```json
    response_format={"type": "json_object"}
    ```

    Structured output (JSON Schema):
    ```json
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "my_schema",
            "schema": {"type": "object", "properties": {...}}
        }
    }
    ```
    """
    # Log incoming request summary at debug, message content at trace
    logger.debug(
        f"Chat completion request received: model={request.model}, "
        f"messages={len(request.messages)}, stream={request.stream}, "
        f"max_tokens={request.max_tokens}, temp={request.temperature}"
    )
    if logger.isEnabledFor(5):
        for i, msg in enumerate(request.messages):
            content_preview = str(msg.content)[:200] if msg.content else "(empty)"
            logger.log(
                5, "  Message[%d]: role=%s, content=%s...", i, msg.role, content_preview
            )

    if is_markitdown_model(request.model):
        return await _create_markitdown_chat_completion(request, http_request)

    request = await _preprocess_markitdown_files_for_llm(request)

    # Block inference during quantization to prevent GPU Metal errors
    if _server_state.oq_manager and _server_state.oq_manager.is_quantizing:
        raise HTTPException(
            status_code=503,
            detail="Server is busy with oQ quantization. Please try again after quantization completes.",
        )

    lease = _LLMEngineLease()
    try:
        load_start = time.perf_counter()
        engine = await get_engine_for_model(request.model, lease=lease)
        model_load_duration = time.perf_counter() - load_start

        # Resolve alias to real model ID for settings lookups
        resolved_model = resolve_model_id(request.model) or request.model

        # Get per-model settings
        max_tool_result_tokens = None
        merged_ct_kwargs = {}
        forced_keys: set[str] = set()
        reasoning_parser = None
        settings_guided_grammar = None
        ms = get_model_settings_for_request(request.model)
        if ms:
            max_tool_result_tokens = ms.max_tool_result_tokens
            reasoning_parser = ms.reasoning_parser
            settings_guided_grammar = _settings_guided_grammar(ms)
            if ms.chat_template_kwargs:
                merged_ct_kwargs.update(ms.chat_template_kwargs)
            forced_keys = set(ms.forced_ct_kwargs or [])
            # Dedicated enable_thinking toggle takes precedence over chat_template_kwargs
            if ms.enable_thinking is not None:
                merged_ct_kwargs["enable_thinking"] = ms.enable_thinking
            # preserve_thinking: keep <think> blocks in historical turns (Qwen 3.6+)
            if ms.preserve_thinking is not None:
                merged_ct_kwargs["preserve_thinking"] = ms.preserve_thinking
        # Per-request kwargs override model settings (except forced keys)
        if request.chat_template_kwargs:
            for k, v in request.chat_template_kwargs.items():
                if k not in forced_keys:
                    merged_ct_kwargs[k] = v

        # Extract messages - different engines need different content handling.
        # Templates that expose message.reasoning_content natively (Qwen 3.6+)
        # get reasoning as a separate field; others fall back to <think> inlined
        # in content.
        _entry = get_engine_pool().get_entry(resolved_model)
        native_reasoning = uses_native_reasoning_content(
            resolved_model,
            config_model_type=(
                getattr(_entry, "config_model_type", None)
                if _entry is not None
                else None
            ),
            engine_model_type=getattr(engine, "model_type", None),
            preserve_thinking_default=(
                getattr(_entry, "preserve_thinking_default", None)
                if _entry is not None
                else None
            ),
        )
        is_vlm = isinstance(engine, VLMBatchedEngine)
        is_dflash_vlm = not is_vlm and getattr(
            engine, "supports_multimodal_fallback", False
        )
        extractor = getattr(engine, "message_extractor", None)
        merge_system_fallback_roles = not (is_vlm or is_dflash_vlm)
        if extractor is not None:
            extractor_kwargs = {}
            try:
                if (
                    "consolidate_system_messages"
                    in inspect.signature(extractor).parameters
                ):
                    extractor_kwargs["consolidate_system_messages"] = False
            except (TypeError, ValueError):
                pass
            messages = extractor(
                request.messages,
                max_tool_result_tokens,
                engine.tokenizer,
                **extractor_kwargs,
            )
            merge_system_fallback_roles = True
        elif is_vlm or is_dflash_vlm:
            # VLM or DFlash with VLM fallback: preserve image_url content parts
            messages = extract_multimodal_content(
                request.messages,
                max_tool_result_tokens,
                engine.tokenizer,
                native_reasoning_content=native_reasoning,
                consolidate_system_messages=False,
            )
        else:
            messages = extract_text_content(
                request.messages,
                max_tool_result_tokens,
                engine.tokenizer,
                native_reasoning_content=native_reasoning,
                consolidate_system_messages=False,
            )

        # Detect and strip partial mode at the API boundary — exactly once,
        # before any chat template application.  The boolean result is forwarded
        # as an explicit parameter so the engine never has to re-derive it.
        is_partial = detect_and_strip_partial(messages)

        # Compile grammar for structured output (logit-level enforcement).
        # Grammar compilation needs the tokenizer, so ensure the engine is loaded.
        response_format = request.response_format
        guided_grammar = _effective_guided_grammar(
            structured_outputs=request.structured_outputs,
            response_format=response_format,
            request_guided_grammar=request.guided_grammar,
            settings_guided_grammar=settings_guided_grammar,
        )
        structured_outputs = _normalize_structured_outputs(
            request.structured_outputs,
            guided_grammar,
        )
        _reject_diffusion_structured_outputs(
            engine,
            response_format=response_format,
            structured_outputs=structured_outputs,
            guided_grammar=guided_grammar,
        )
        if structured_outputs is not None or response_format:
            await engine.start()
        compiled_grammar = _compile_grammar_for_request(
            engine,
            structured_outputs=structured_outputs,
            response_format=response_format,
            chat_template_kwargs=merged_ct_kwargs or None,
            reasoning_parser=reasoning_parser,
        )
        # Fall back to prompt injection when grammar is not compiled. The degrade
        # is also surfaced to the caller as a Warning response header (#1241).
        # Only response formats that actually request grammar-constrained JSON
        # (json_object / json_schema) can be "unenforced"; a plain text format
        # never asked for enforcement, so it must not warn (#1241 review).
        response_format_warning = None
        if compiled_grammar is None and _response_format_requests_grammar(
            response_format
        ):
            response_format_warning = _response_format_warning_header(response_format)
            json_instruction = build_json_system_prompt(response_format)
            if json_instruction:
                messages = _inject_json_instruction(messages, json_instruction)

        # Merge MCP tools with user-provided tools unless the request explicitly
        # disables tool use.
        tools_disabled = request.tool_choice == "none"
        if getattr(engine, "is_diffusion_model", False) and not getattr(
            engine, "supports_tool_calling", False
        ):
            if request.tools and not tools_disabled:
                raise InvalidRequestError(
                    "Tool calling is not supported for this diffusion model "
                    "(no tool parser matched its chat template).",
                    field="tools",
                )
            tools_disabled = True
        effective_tools = None if tools_disabled else request.tools
        if _server_state.mcp_manager and not tools_disabled:
            # Convert Pydantic ToolDefinition models to dicts for merge_tools
            user_tools_dicts = (
                [t.model_dump() for t in request.tools] if request.tools else None
            )
            effective_tools = _server_state.mcp_manager.get_merged_tools(
                user_tools_dicts
            )

        # Validate context window before sending to model
        tools_for_template = (
            convert_tools_for_template(effective_tools) if effective_tools else None
        )
        # Gemma 4 drops required params that lack descriptions — enrich them
        if tools_for_template and "gemma" in (resolved_model or "").lower():
            tools_for_template = enrich_tool_params_for_gemma4(tools_for_template)
        await _ensure_tokenizer_for_system_probe(engine, messages)
        messages = prepare_system_messages_for_template(
            messages,
            engine.tokenizer,
            tools=tools_for_template,
            chat_template_kwargs=merged_ct_kwargs or None,
            is_partial=is_partial,
            merge_consecutive_roles=merge_system_fallback_roles,
            unsupported_mid_system_policy=_unsupported_mid_system_policy(),
        )
        try:
            num_prompt_tokens = engine.count_chat_tokens(
                messages,
                tools_for_template,
                chat_template_kwargs=merged_ct_kwargs or None,
                is_partial=is_partial,
            )
        except Exception as e:
            # Catch chat template rendering failures: Jinja2 TemplateError,
            # AssertionError from strict role validation, ValueError, etc.
            err_name = type(e).__name__.lower()
            err_msg = str(e).lower()
            if (
                "template" in err_name
                or "template" in err_msg
                or isinstance(e, (AssertionError, ValueError))
            ):
                raise HTTPException(status_code=400, detail=f"Chat template error: {e}")
            raise
        validate_context_window(num_prompt_tokens, request.model)

        # Prepare kwargs
        (
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            min_p,
            presence_penalty,
            frequency_penalty,
            max_tokens,
            xtc_probability,
            xtc_threshold,
        ) = get_sampling_params(
            request.temperature,
            request.top_p,
            request.model,
            req_top_k=getattr(request, "top_k", None),
            req_repetition_penalty=getattr(request, "repetition_penalty", None),
            req_min_p=getattr(request, "min_p", None),
            req_presence_penalty=getattr(request, "presence_penalty", None),
            req_frequency_penalty=getattr(request, "frequency_penalty", None),
            req_max_tokens=request.max_tokens,
            req_xtc_probability=getattr(request, "xtc_probability", None),
            req_xtc_threshold=getattr(request, "xtc_threshold", None),
        )
        chat_kwargs = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "repetition_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "xtc_probability": xtc_probability,
            "xtc_threshold": xtc_threshold,
        }

        # Add seed for reproducible generation (best-effort)
        if request.seed is not None:
            chat_kwargs["seed"] = request.seed

        # Add thinking budget if applicable
        thinking_budget = _resolve_thinking_budget(request, request.model)
        if thinking_budget is not None:
            chat_kwargs["thinking_budget"] = thinking_budget

        # Auto-set enable_thinking in chat template kwargs when a thinking
        # budget is active (from request or model settings).  Some chat
        # templates (e.g. Gemma 4) explicitly suppress thinking unless this
        # kwarg is True.
        if thinking_budget is not None and "enable_thinking" not in merged_ct_kwargs:
            merged_ct_kwargs["enable_thinking"] = True

        # Auto-set preserve_thinking only when the template advertises support
        # for it (Qwen 3.6+). Other templates silently ignore unknown kwargs
        # today but strict templates could raise, so gate on the detected flag.
        _entry = get_engine_pool().get_entry(resolved_model)
        if (
            _entry is not None
            and _entry.preserve_thinking_default is True
            and merged_ct_kwargs.get("enable_thinking") is not False
            and "preserve_thinking" not in merged_ct_kwargs
        ):
            merged_ct_kwargs["preserve_thinking"] = True

        # Add compiled grammar for logit-level structured output.
        # When a reasoning_parser is configured, the structural tag includes
        # a thinking phase — auto-set a thinking_budget so the model exits
        # the reasoning phase and the grammar can activate.
        if compiled_grammar is not None:
            chat_kwargs["compiled_grammar"] = compiled_grammar
            if reasoning_parser and "thinking_budget" not in chat_kwargs:
                default_budget = min(max_tokens // 2, 4096)
                chat_kwargs["thinking_budget"] = default_budget
                logger.debug(
                    "Auto-set thinking_budget=%d for grammar-constrained request",
                    default_budget,
                )

        # Add tools if provided (includes MCP tools)
        if tools_for_template:
            chat_kwargs["tools"] = tools_for_template

        # Add chat template kwargs
        if merged_ct_kwargs:
            chat_kwargs["chat_template_kwargs"] = merged_ct_kwargs

        # Forward partial-mode decision to the engine explicitly
        chat_kwargs["is_partial"] = is_partial

        # SpecPrefill: per-request overrides (fall back to model_settings)
        if request.specprefill is not None:
            chat_kwargs["specprefill"] = request.specprefill
        if request.specprefill_keep_pct is not None:
            chat_kwargs["specprefill_keep_pct"] = request.specprefill_keep_pct
        elif _server_state.settings_manager and ms.specprefill_keep_pct is not None:
            chat_kwargs["specprefill_keep_pct"] = ms.specprefill_keep_pct
        if getattr(request, "specprefill_threshold", None) is not None:
            chat_kwargs["specprefill_threshold"] = request.specprefill_threshold
        elif _server_state.settings_manager and ms.specprefill_threshold is not None:
            chat_kwargs["specprefill_threshold"] = ms.specprefill_threshold

        if request.stop:
            chat_kwargs["stop"] = request.stop

        # Pre-flight prefill memory guard. Must run BEFORE either branch wraps
        # the response in a StreamingResponse — starlette emits
        # http.response.start (status 200) before iterating the body generator,
        # so a typed exception thrown later by add_request lands as "Caught
        # handled exception, but response already started" and the client sees
        # an incomplete chunked read. Running the check here lets
        # prefill_memory_exceeded_handler return a clean HTTP 400.
        await _raise_if_llm_lease_abort_requested(lease)
        await engine.preflight_chat(
            messages,
            request_id=http_request.headers.get("x-request-id"),
            **chat_kwargs,
        )

        await _raise_if_llm_lease_abort_requested(lease)

        if request.stream:
            # Pre-mint the completion id so the keepalive frame (emitted before the
            # generator starts) can share it. See _chat_keepalive_chunk.
            response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            keepalive = _resolve_keepalive("openai_chat")
            if keepalive == _KEEPALIVE_CHAT_CHUNK:
                keepalive = _chat_keepalive_chunk(response_id)
            sse_headers = {"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}
            if response_format_warning:
                sse_headers["Warning"] = response_format_warning
            return StreamingResponse(
                _release_after_stream(
                    _with_sse_keepalive(
                        stream_chat_completion(
                            engine,
                            messages,
                            request,
                            model_load_duration=model_load_duration,
                            resolved_model=resolved_model,
                            response_id=response_id,
                            **chat_kwargs,
                        ),
                        http_request=http_request,
                        keepalive_chunk=keepalive,
                    ),
                    lease,
                ),
                media_type="text/event-stream",
                headers=sse_headers,
            )

        # Non-streaming response with keepalive during prefill
        async def _build_chat_completion():
            await _raise_if_llm_lease_abort_requested(lease)
            start_time = time.perf_counter()

            output = await engine.chat(messages=messages, **chat_kwargs)

            elapsed = time.perf_counter() - start_time
            tokens_per_sec = output.completion_tokens / elapsed if elapsed > 0 else 0
            is_diffusion = getattr(engine, "is_diffusion_model", False)
            speed_text = _format_generation_speed_for_log(
                output,
                tokens_per_sec,
                is_diffusion=is_diffusion,
            )
            logger.info(
                f"Chat completion: {output.completion_tokens} tokens in {elapsed:.2f}s "
                f"({speed_text}), prompt: {output.prompt_tokens}, "
                f"finish_reason={output.finish_reason}, max_tokens={max_tokens}, "
                f"request_max_tokens={request.max_tokens}"
            )
            metric_prefill_duration, metric_gen_duration = _resolve_metric_durations(
                output,
                is_diffusion=is_diffusion,
                generation_duration=elapsed,
            )

            get_server_metrics().record_request_complete(
                prompt_tokens=output.prompt_tokens,
                completion_tokens=output.completion_tokens,
                cached_tokens=output.cached_tokens,
                prefill_duration=metric_prefill_duration,
                generation_duration=metric_gen_duration,
                model_id=resolved_model,
            )

            # Separate thinking from content
            raw_text = clean_special_tokens(output.text) if output.text else ""
            thinking_content, regular_content = extract_thinking(raw_text)
            cleaned_thinking = sanitize_tool_call_markup(
                thinking_content, engine.tokenizer
            )

            # Protocol parsers can return structured tool_calls directly.
            if output.tool_calls:
                tool_calls = _convert_parser_tool_calls(output.tool_calls)
                cleaned_text = regular_content
            else:
                extraction = extract_tool_calls_with_thinking(
                    thinking_content,
                    regular_content,
                    tokenizer=engine.tokenizer,
                    tools=tools_for_template,
                )
                cleaned_text = extraction.cleaned_text
                tool_calls = extraction.tool_calls
                cleaned_thinking = extraction.cleaned_thinking

            # Process response_format if specified
            if response_format and not tool_calls:
                cleaned_text, parsed_json, is_valid, error = parse_json_output(
                    cleaned_text or regular_content, response_format
                )
                if parsed_json is not None:
                    cleaned_text = json.dumps(parsed_json)
                if not is_valid:
                    logger.warning(f"JSON validation failed: {error}")

            # Reverse Gemma 4 parameter renaming (param_description -> description)
            if tool_calls and "gemma" in (resolved_model or "").lower():
                for tc in tool_calls:
                    if tc.function and tc.function.arguments:
                        try:
                            args = json.loads(tc.function.arguments)
                            args = restore_gemma4_param_names(args)
                            tc.function.arguments = json.dumps(args, ensure_ascii=False)
                        except (json.JSONDecodeError, AttributeError):
                            pass

            finish_reason = "tool_calls" if tool_calls else output.finish_reason

            return ChatCompletionResponse(
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        message=AssistantMessage(
                            content=cleaned_text.strip() if cleaned_text else None,
                            reasoning_content=(
                                cleaned_thinking if cleaned_thinking else None
                            ),
                            tool_calls=tool_calls,
                        ),
                        finish_reason=finish_reason,
                    )
                ],
                usage=Usage(
                    prompt_tokens=output.prompt_tokens,
                    completion_tokens=output.completion_tokens,
                    total_tokens=output.prompt_tokens + output.completion_tokens,
                    prompt_tokens_details=PromptTokensDetails(
                        cached_tokens=output.cached_tokens,
                    ),
                    model_load_duration=(
                        round(model_load_duration, 2)
                        if model_load_duration > 1.0
                        else None
                    ),
                    total_time=round(elapsed, 2),
                ),
            ).model_dump_json(exclude_none=True)

        json_headers = (
            {"Warning": response_format_warning} if response_format_warning else None
        )
        return StreamingResponse(
            _release_after_stream(
                _with_json_keepalive(http_request, _build_chat_completion()),
                lease,
            ),
            media_type="application/json",
            headers=json_headers,
        )

    except BaseException:
        await lease.release()
        raise


def _inject_json_instruction(messages: list, instruction: str) -> list:
    """
    Inject JSON instruction into messages.

    If a system message exists, append to it. Otherwise, prepend a new system message.
    """
    messages = list(messages)  # Make a copy

    # Only attach to a leading system message. A mid-conversation system
    # message may be intentionally placed there to preserve prefix cache hits.
    system_idx = None
    if messages:
        first = messages[0]
        role = (
            first.get("role")
            if isinstance(first, dict)
            else getattr(first, "role", None)
        )
        if role == "system":
            system_idx = 0

    if system_idx is not None:
        # Append to existing system message
        msg = messages[system_idx]
        if isinstance(msg, dict):
            existing = msg.get("content", "")
            msg["content"] = f"{existing}\n\n{instruction}"
        else:
            existing = getattr(msg, "content", "") or ""
            msg.content = f"{existing}\n\n{instruction}"
    else:
        # Prepend new system message
        messages.insert(0, {"role": "system", "content": instruction})

    return messages


def _normalize_structured_outputs(
    structured_outputs=None, guided_grammar: str | None = None
):
    """Fold guided_grammar into the existing structured_outputs grammar shape."""
    if structured_outputs is not None:
        return structured_outputs
    if guided_grammar:
        return {"grammar": guided_grammar}
    return None


def _reject_diffusion_structured_outputs(
    engine: BaseEngine,
    *,
    response_format=None,
    structured_outputs=None,
    guided_grammar: str | None = None,
) -> None:
    if not getattr(engine, "is_diffusion_model", False):
        return
    # ``response_format`` (json_object / json_schema) is NOT rejected here:
    # it degrades to prompt-injected JSON with a Warning header, the same
    # fallback used when xgrammar is unavailable (#1241).  Only explicit
    # grammar requests — ``structured_outputs`` and ``guided_grammar`` —
    # are rejected, because logit-mask enforcement has no equivalent in
    # the parallel denoising loop.
    if structured_outputs is None and not guided_grammar:
        return
    raise InvalidRequestError(
        "structured_outputs and guided grammar are not supported "
        "with diffusion models (response_format degrades to "
        "prompt-injected JSON).",
        field="response_format",
    )


def _settings_guided_grammar(settings) -> str | None:
    """Return a non-empty enabled model-level guided grammar."""
    if not settings:
        return None
    if not getattr(settings, "guided_grammar_enabled", False):
        return None
    grammar = getattr(settings, "guided_grammar", None)
    if not grammar:
        return None
    grammar = grammar.strip()
    return grammar or None


def _effective_guided_grammar(
    structured_outputs=None,
    response_format=None,
    request_guided_grammar: str | None = None,
    settings_guided_grammar: str | None = None,
) -> str | None:
    """Choose the request grammar alias or eligible model default."""
    if request_guided_grammar:
        return request_guided_grammar
    if structured_outputs is None and response_format is None:
        return settings_guided_grammar
    return None


def _build_format_element(structured_outputs=None, response_format=None):
    """Build an xgrammar structural-tag format element from the request.

    Returns a format dict (e.g. ``{"type": "json_schema", ...}``) suitable
    for embedding in a structural tag, or ``None`` if no grammar is needed.
    Also returns ``"bare"`` compilation hint when the grammar should be
    compiled directly (EBNF / regex / choice) rather than via structural tag.
    """
    import json as _json

    from .api.openai_models import StructuredOutputOptions

    if structured_outputs is not None:
        if isinstance(structured_outputs, dict):
            structured_outputs = StructuredOutputOptions(**structured_outputs)

        if structured_outputs.json_schema is not None:
            schema = structured_outputs.json_schema
            if isinstance(schema, str):
                schema = _json.loads(schema)
            return {"type": "json_schema", "json_schema": schema}
        if structured_outputs.grammar is not None:
            return {"type": "grammar", "grammar": structured_outputs.grammar}
        if structured_outputs.regex is not None:
            return {"type": "regex", "pattern": structured_outputs.regex}
        if structured_outputs.choice is not None:
            ebnf = "root ::= " + " | ".join(
                _json.dumps(c) for c in structured_outputs.choice
            )
            return {"type": "grammar", "grammar": ebnf}

    if response_format is not None:
        rf = response_format
        rf_type = rf.get("type") if isinstance(rf, dict) else getattr(rf, "type", None)
        if rf_type == "json_schema":
            js = (
                rf.get("json_schema")
                if isinstance(rf, dict)
                else getattr(rf, "json_schema", None)
            )
            if js is not None:
                schema = (
                    js.get("schema")
                    if isinstance(js, dict)
                    else getattr(js, "schema_", None)
                )
                if schema is not None:
                    return {"type": "json_schema", "json_schema": schema}
        elif rf_type == "json_object":
            return {"type": "json_schema", "json_schema": {}}

    return None


def _patch_output_format(tag_dict: dict, user_grammar: dict) -> bool:
    """Replace the output ``any_text`` slot in a builtin structural tag.

    Walks the structural tag dict produced by
    ``xgrammar.get_builtin_structural_tag`` and swaps the ``any_text``
    element that represents the model's output with ``user_grammar``.

    Returns ``True`` if a replacement was made.
    """
    fmt = tag_dict.get("format", tag_dict)

    if fmt.get("type") == "any_text":
        tag_dict["format"] = user_grammar
        return True

    if fmt.get("type") == "sequence":
        for i in range(len(fmt["elements"]) - 1, -1, -1):
            if fmt["elements"][i].get("type") == "any_text":
                fmt["elements"][i] = user_grammar
                return True

    if fmt.get("type") == "tags_with_separator":
        for tag in reversed(fmt["tags"]):
            if tag.get("type") == "tag" and "final" in tag.get("begin", ""):
                tag["content"] = user_grammar
                return True
        if fmt["tags"]:
            fmt["tags"][-1]["content"] = user_grammar
            return True

    return False


def _compile_with_structural_tag(
    compiler, fmt: dict, reasoning_parser: str, chat_template_kwargs: dict | None
):
    """Compile a grammar wrapped in an xgrammar builtin structural tag.

    Uses ``xgrammar.get_builtin_structural_tag`` to obtain the model's
    protocol structure (thinking tags, channel markers, etc.) and patches
    the user's grammar into the output slot.
    """
    from omlx._torch_stub import install as _install_torch_stub

    _install_torch_stub()
    import xgrammar as xgr

    reasoning = not (
        chat_template_kwargs and chat_template_kwargs.get("enable_thinking") is False
    )
    tag = xgr.get_builtin_structural_tag(reasoning_parser, reasoning=reasoning)
    tag_dict = tag.model_dump()
    if not _patch_output_format(tag_dict, fmt):
        logger.warning(
            "Could not patch output format for reasoning_parser=%s, "
            "compiling structural tag as-is",
            reasoning_parser,
        )
    return compiler.compile_structural_tag(tag_dict)


def _compile_bare_grammar(compiler, fmt: dict):
    """Compile a grammar without any structural tag wrapping."""
    if fmt["type"] == "json_schema":
        import json as _json

        schema = fmt["json_schema"]
        if not schema:
            return compiler.compile_builtin_json_grammar()
        schema_str = _json.dumps(schema) if isinstance(schema, dict) else schema
        return compiler.compile_json_schema(schema_str)
    elif fmt["type"] == "grammar":
        return compiler.compile_grammar(fmt["grammar"])
    elif fmt["type"] == "regex":
        return compiler.compile_regex(fmt["pattern"])
    return None


def _response_format_requests_strict(response_format) -> bool:
    """True when an OpenAI ``response_format`` demands strict json_schema output.

    A ``json_schema`` response_format with ``strict: true`` signals that the
    caller expects schema-conformant output, not best-effort.  When
    grammar-constrained decoding is unavailable the request still falls back to
    prompt injection, but the downgrade is logged at a level that names the
    unhonored ``strict`` intent so it is not silent (issue #1241).
    """
    if response_format is None:
        return False
    rf = response_format
    rf_type = rf.get("type") if isinstance(rf, dict) else getattr(rf, "type", None)
    if rf_type != "json_schema":
        return False
    js = (
        rf.get("json_schema")
        if isinstance(rf, dict)
        else getattr(rf, "json_schema", None)
    )
    if js is None:
        return False
    strict = js.get("strict") if isinstance(js, dict) else getattr(js, "strict", None)
    return bool(strict)


def _response_format_requests_grammar(response_format) -> bool:
    """True when an OpenAI ``response_format`` maps to grammar-constrained JSON.

    Delegates to :func:`_build_format_element` so the unenforced-degrade signal
    stays in sync with what actually gets compiled: a format earns the
    Warning header / prompt-injection fallback only when a grammar element would
    have been built for it.  That is non-``None`` exactly for ``json_object``
    and a ``json_schema`` carrying a schema; a plain ``{"type": "text"}`` (or a
    json_schema with no schema) maps to nothing and must not warn.  Sharing the
    one source of truth keeps the header consistent with the server-side warn
    log and avoids claiming "grammar-constrained decoding unavailable" for a
    request that never described an enforceable grammar (#1241 review).
    """
    if response_format is None:
        return False
    return _build_format_element(response_format=response_format) is not None


def _compile_grammar_for_request(
    engine: BaseEngine,
    structured_outputs=None,
    response_format=None,
    chat_template_kwargs=None,
    reasoning_parser=None,
):
    """Compile a grammar from structured_outputs or response_format.

    When ``reasoning_parser`` is set (e.g. ``"qwen"``, ``"harmony"``),
    the user's grammar is wrapped in an xgrammar builtin structural tag
    so that protocol tokens (thinking tags, channel markers) are handled
    automatically.  When not set, the grammar is compiled bare.

    Returns a compiled grammar object or ``None``.  ``structured_outputs``
    raises :class:`HTTPException` when grammar is unavailable or fails to
    compile.  A ``response_format`` degrades to ``None`` so the caller can fall
    back to prompt injection; the downgrade is logged (and named as an
    unhonored strict request when ``strict: true`` was set) rather than being
    silent (#1241).
    """
    compiler = getattr(engine, "grammar_compiler", None)

    fmt = _build_format_element(structured_outputs, response_format)
    if fmt is None:
        return None

    if compiler is None:
        if structured_outputs is not None:
            from omlx.utils.install import get_install_method

            method = get_install_method()
            if method == "homebrew":
                detail = (
                    "Structured output requires xgrammar. "
                    "Reinstall with: brew reinstall omlx --with-grammar"
                )
            elif method == "dmg":
                # DMG bundles xgrammar with a torch stub; reaching this
                # branch means the bundled load failed (e.g. native binding
                # incompatibility). Surface it instead of pointing users to
                # a different install method.
                detail = (
                    "Structured output is unavailable: xgrammar failed to "
                    "load in this build. Please report this issue."
                )
            else:
                detail = (
                    "Structured output requires xgrammar. "
                    "Install with: pip install 'omlx[grammar]'"
                )
            raise HTTPException(status_code=400, detail=detail)
        if response_format is not None:
            _warn_response_format_not_enforced(response_format)
        return None

    try:
        if reasoning_parser:
            return _compile_with_structural_tag(
                compiler,
                fmt,
                reasoning_parser,
                chat_template_kwargs,
            )
        return _compile_bare_grammar(compiler, fmt)
    except Exception as e:
        if structured_outputs is not None:
            raise HTTPException(
                status_code=400,
                detail=f"Grammar compilation error: {e}",
            )
        _warn_response_format_not_enforced(response_format, error=e)
    return None


def _warn_response_format_not_enforced(response_format, error=None):
    """Log that a ``response_format`` request fell back to prompt injection.

    Previously a ``response_format`` that could not be grammar-constrained
    (no compiler available, or a compilation error) degraded to best-effort
    prompt injection silently, giving the client no signal that the schema was
    not enforced (#1241).  A ``strict: true`` request gets a message that names
    the unhonored strict intent.
    """
    reason = f" ({error})" if error is not None else ""
    if _response_format_requests_strict(response_format):
        logger.warning(
            "response_format requested strict json_schema output but "
            "grammar-constrained decoding is unavailable; strict enforcement "
            "cannot be honored, falling back to best-effort prompt injection "
            "(output is NOT schema-enforced)%s.",
            reason,
        )
    else:
        logger.warning(
            "response_format requested but grammar-constrained decoding is "
            "unavailable; output will not be schema-enforced (falling back to "
            "prompt injection)%s.",
            reason,
        )


def _response_format_warning_header(response_format) -> str:
    """Build an RFC 7234 ``Warning`` header for an unenforced response_format.

    The server already logs the downgrade (see
    :func:`_warn_response_format_not_enforced`), but that signal is only
    visible to the operator.  This header surfaces the same fact to the API
    caller so a client can tell that ``response_format`` fell back to
    best-effort prompt injection rather than schema-enforced output (#1241).
    Header values must be single-line ASCII, so the text is terse.
    """
    if _response_format_requests_strict(response_format):
        text = (
            "response_format strict json_schema not enforced; "
            "grammar-constrained decoding unavailable, output is "
            "best-effort and NOT schema-enforced"
        )
    else:
        text = (
            "response_format not enforced; grammar-constrained decoding "
            "unavailable, output is best-effort"
        )
    return f'199 omlx "{text}"'


# =============================================================================
# Streaming Helpers
# =============================================================================


async def stream_completion(
    engine: BaseEngine,
    prompt: str,
    request: CompletionRequest,
    model_load_duration: float = 0.0,
    prompt_token_ids: list[int] | None = None,
) -> AsyncIterator[str]:
    """Stream completion response."""
    start_time = time.perf_counter()
    first_token_time = None
    last_output = None
    # Parity with the non-streaming path: when the prompt opens a thinking
    # block, the first chunk carries the scheduler's synthetic think opener;
    # strip it once so the stream is a pure continuation of the prompt.
    pending_think_prefix_strip, think_tag = prompt_opens_thinking(
        getattr(engine, "tokenizer", None), prompt, prompt_token_ids=prompt_token_ids
    )

    (
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        min_p,
        presence_penalty,
        frequency_penalty,
        max_tokens,
        xtc_probability,
        xtc_threshold,
    ) = get_sampling_params(
        request.temperature,
        request.top_p,
        request.model,
        req_top_k=getattr(request, "top_k", None),
        req_repetition_penalty=getattr(request, "repetition_penalty", None),
        req_min_p=getattr(request, "min_p", None),
        req_presence_penalty=getattr(request, "presence_penalty", None),
        req_frequency_penalty=getattr(request, "frequency_penalty", None),
        req_max_tokens=request.max_tokens,
        req_xtc_probability=getattr(request, "xtc_probability", None),
        req_xtc_threshold=getattr(request, "xtc_threshold", None),
    )
    gen_kwargs = {}
    thinking_budget = _resolve_thinking_budget(request, request.model)
    if thinking_budget is not None:
        gen_kwargs["thinking_budget"] = thinking_budget
    try:
        async for output in engine.stream_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            xtc_probability=xtc_probability,
            xtc_threshold=xtc_threshold,
            stop=request.stop,
            seed=request.seed,
            **gen_kwargs,
        ):
            if first_token_time is None and output.new_text:
                first_token_time = time.perf_counter()
            last_output = output

            chunk_text = output.new_text
            if pending_think_prefix_strip and chunk_text:
                chunk_text = _strip_synthetic_think_prefix(chunk_text, think_tag)
                pending_think_prefix_strip = False

            data = {
                "id": f"cmpl-{uuid.uuid4().hex[:8]}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "text": chunk_text,
                        "finish_reason": (
                            output.finish_reason if output.finished else None
                        ),
                    }
                ],
            }
            yield f"data: {json.dumps(data)}\n\n"
    except Exception as e:
        logger.error(f"Error during completion streaming: {e}")
        error_data = {"error": {"message": str(e), "type": "server_error"}}
        yield f"data: {json.dumps(error_data)}\n\n"
        yield "data: [DONE]\n\n"
        return

    # Record metrics
    if last_output and last_output.finished:
        end_time = time.perf_counter()
        total_duration = end_time - start_time
        ttft = (first_token_time - start_time) if first_token_time else total_duration
        is_diffusion = getattr(engine, "is_diffusion_model", False)
        if is_diffusion:
            gen_duration = total_duration
        else:
            gen_duration = end_time - (first_token_time or start_time)
        metric_prefill_duration, metric_gen_duration = _resolve_metric_durations(
            last_output,
            is_diffusion=is_diffusion,
            prefill_duration=ttft,
            generation_duration=gen_duration,
        )
        get_server_metrics().record_request_complete(
            prompt_tokens=last_output.prompt_tokens,
            completion_tokens=last_output.completion_tokens,
            cached_tokens=last_output.cached_tokens,
            prefill_duration=metric_prefill_duration,
            generation_duration=metric_gen_duration,
            model_id=resolve_model_id(request.model) or request.model,
        )
        speed_duration = total_duration if is_diffusion else gen_duration
        tokens_per_sec = (
            last_output.completion_tokens / speed_duration if speed_duration > 0 else 0
        )
        speed_text = _format_generation_speed_for_log(
            last_output,
            tokens_per_sec,
            is_diffusion=is_diffusion,
        )
        logger.info(
            f"Completion: {last_output.completion_tokens} tokens in "
            f"{total_duration:.2f}s ({speed_text}), "
            f"prompt: {last_output.prompt_tokens}"
        )

        # Emit usage chunk if requested
        if request.stream_options and request.stream_options.include_usage:
            total_time = end_time - start_time
            pt = last_output.prompt_tokens
            ct = last_output.completion_tokens
            usage_data = {
                "id": f"cmpl-{uuid.uuid4().hex[:8]}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [],
                "usage": Usage(
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    total_tokens=pt + ct,
                    prompt_tokens_details=PromptTokensDetails(
                        cached_tokens=last_output.cached_tokens,
                    ),
                    model_load_duration=(
                        round(model_load_duration, 2)
                        if model_load_duration > 1.0
                        else None
                    ),
                    time_to_first_token=round(ttft, 2),
                    total_time=round(total_time, 2),
                    prompt_eval_duration=round(metric_prefill_duration, 2),
                    generation_duration=round(metric_gen_duration, 2),
                    prompt_tokens_per_second=(
                        round(pt / metric_prefill_duration, 2)
                        if metric_prefill_duration > 0
                        else None
                    ),
                    generation_tokens_per_second=(
                        round(ct / metric_gen_duration, 2)
                        if metric_gen_duration > 0
                        else None
                    ),
                ).model_dump(exclude_none=True),
            }
            yield f"data: {json.dumps(usage_data)}\n\n"

    yield "data: [DONE]\n\n"


def _copy_chat_template_messages(messages: list) -> list:
    return [
        dict(message) if isinstance(message, dict) else message for message in messages
    ]


def _render_chat_prompt_for_thinking_detection(
    engine: BaseEngine,
    messages: list,
    kwargs: dict,
) -> tuple[str, list[int] | None]:
    tokenizer = getattr(engine, "tokenizer", None)
    if tokenizer is None:
        return "", None

    template_messages = _copy_chat_template_messages(messages)
    tools = kwargs.get("tools")
    chat_template_kwargs = kwargs.get("chat_template_kwargs")
    is_partial = kwargs.get("is_partial")
    engine_renderer = getattr(engine, "_apply_chat_template", None)

    if is_partial is not None:
        for message in template_messages:
            if isinstance(message, dict):
                message.pop("partial", None)

    if callable(engine_renderer):
        prompt = engine_renderer(
            template_messages,
            tools,
            chat_template_kwargs=chat_template_kwargs,
            is_partial=is_partial,
        )
    else:
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": not bool(is_partial),
        }
        if is_partial:
            template_kwargs["continue_final_message"] = True
        if tools:
            template_kwargs["tools"] = tools
        if chat_template_kwargs:
            template_kwargs.update(chat_template_kwargs)

        try:
            prompt = tokenizer.apply_chat_template(template_messages, **template_kwargs)
        except TypeError:
            if chat_template_kwargs:
                for key in chat_template_kwargs:
                    template_kwargs.pop(key, None)
            template_kwargs.pop("tools", None)
            template_kwargs.pop("enable_thinking", None)
            prompt = tokenizer.apply_chat_template(template_messages, **template_kwargs)

    if isinstance(prompt, str):
        return prompt, None
    if isinstance(prompt, list):
        try:
            return "", [int(token_id) for token_id in prompt]
        except (TypeError, ValueError):
            return str(prompt), None
    return str(prompt), None


async def stream_chat_completion(
    engine: BaseEngine,
    messages: list,
    request: ChatCompletionRequest,
    model_load_duration: float = 0.0,
    resolved_model: Optional[str] = None,
    response_id: Optional[str] = None,
    **kwargs,
) -> AsyncIterator[str]:
    """Stream chat completion response.

    Streams content tokens with reasoning/thinking separation, then at
    completion parses tool calls from accumulated text and emits them
    as structured tool_calls chunks (OpenAI streaming format).
    """
    start_time = time.perf_counter()
    first_token_time = None
    last_output = None
    accumulated_text = ""
    has_tools = bool(kwargs.get("tools"))
    start_in_thinking = False
    try:
        tokenizer = getattr(engine, "tokenizer", None)
        if tokenizer is not None:
            prompt, prompt_token_ids = _render_chat_prompt_for_thinking_detection(
                engine, messages, kwargs
            )
            start_in_thinking, _ = prompt_opens_thinking(
                tokenizer, prompt, prompt_token_ids=prompt_token_ids
            )
    except Exception as exc:
        logger.debug("Could not detect chat stream thinking state: %s", exc)
    thinking_parser = ThinkingParser(start_in_thinking=start_in_thinking)

    # Reuse the id pre-minted by the caller (so the keepalive frame can share
    # it); otherwise mint one for direct/non-streaming callers.
    response_id = response_id or f"chatcmpl-{uuid.uuid4().hex[:8]}"

    # First chunk with role
    first_chunk = ChatCompletionChunk(
        id=response_id,
        model=request.model,
        choices=[
            ChatCompletionChunkChoice(
                delta=ChatCompletionChunkDelta(role="assistant"),
            )
        ],
    )
    yield f"data: {first_chunk.model_dump_json(exclude_none=True)}\n\n"

    # Stream content token-by-token. When tools are present, a
    # ToolCallStreamFilter suppresses known tool-call control markup so
    # clients do not see raw envelopes/tags in assistant content deltas.
    tool_filter = None
    thinking_filter = None
    stream_content = True
    if has_tools:
        _content_filter = ToolCallStreamFilter(engine.tokenizer)
        _thinking_filter = ToolCallStreamFilter(engine.tokenizer)
        if _content_filter.active:
            tool_filter = _content_filter
            thinking_filter = _thinking_filter
        else:
            stream_content = False
    try:
        async for output in engine.stream_chat(messages=messages, **kwargs):
            if first_token_time is None and output.new_text:
                first_token_time = time.perf_counter()
            last_output = output
            if output.new_text:
                accumulated_text += output.new_text

            if stream_content and output.new_text:
                thinking_delta, content_delta = thinking_parser.feed(output.new_text)

                # Emit reasoning_content delta
                if thinking_delta:
                    if thinking_filter:
                        thinking_delta = thinking_filter.feed(thinking_delta)
                    chunk = ChatCompletionChunk(
                        id=response_id,
                        model=request.model,
                        choices=[
                            ChatCompletionChunkChoice(
                                delta=ChatCompletionChunkDelta(
                                    reasoning_content=thinking_delta
                                ),
                                finish_reason=None,
                            )
                        ],
                    )
                    if thinking_delta:
                        yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

                # Emit content delta — filter out tool-call markup when
                # tools are present so clients see clean streamed text.
                if content_delta:
                    if tool_filter:
                        content_delta = tool_filter.feed(content_delta)
                    if content_delta:
                        chunk = ChatCompletionChunk(
                            id=response_id,
                            model=request.model,
                            choices=[
                                ChatCompletionChunkChoice(
                                    delta=ChatCompletionChunkDelta(
                                        content=content_delta
                                    ),
                                    finish_reason=None,
                                )
                            ],
                        )
                        yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
    except Exception as e:
        logger.error(f"Error during chat streaming: {e}")
        error_data = {"error": {"message": str(e), "type": "server_error"}}
        yield f"data: {json.dumps(error_data)}\n\n"
        yield "data: [DONE]\n\n"
        return

    # Flush remaining buffered content from thinking/tool-call parsers
    if stream_content:
        thinking_delta, content_delta = thinking_parser.finish()
        if thinking_delta:
            if thinking_filter:
                thinking_delta = thinking_filter.feed(thinking_delta)
            if thinking_delta:
                chunk = ChatCompletionChunk(
                    id=response_id,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            delta=ChatCompletionChunkDelta(
                                reasoning_content=thinking_delta
                            ),
                            finish_reason=None,
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
        if thinking_filter:
            remaining_thinking = thinking_filter.finish()
            if remaining_thinking:
                chunk = ChatCompletionChunk(
                    id=response_id,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            delta=ChatCompletionChunkDelta(
                                reasoning_content=remaining_thinking
                            ),
                            finish_reason=None,
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
        if content_delta:
            if tool_filter:
                content_delta = tool_filter.feed(content_delta)
            if content_delta:
                chunk = ChatCompletionChunk(
                    id=response_id,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            delta=ChatCompletionChunkDelta(content=content_delta),
                            finish_reason=None,
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

        if tool_filter:
            remaining = tool_filter.finish()
            if remaining:
                chunk = ChatCompletionChunk(
                    id=response_id,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            delta=ChatCompletionChunkDelta(content=remaining),
                            finish_reason=None,
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

    # Parse tool calls from accumulated text
    tool_calls = None
    cleaned_text = accumulated_text
    if last_output and last_output.tool_calls:
        # Protocol parser already extracted structured tool calls.
        tool_calls = _convert_parser_tool_calls(last_output.tool_calls)
        cleaned_text = ""
    elif has_tools and accumulated_text:
        # Separate thinking from content, then parse tool calls from content
        # (falls back to thinking content for small models)
        thinking_content, regular_content = extract_thinking(accumulated_text)
        extraction = extract_tool_calls_with_thinking(
            thinking_content,
            regular_content,
            tokenizer=engine.tokenizer,
            tools=kwargs.get("tools"),
        )
        cleaned_text = extraction.cleaned_text
        tool_calls = extraction.tool_calls
        cleaned_thinking = extraction.cleaned_thinking

        # Process response_format if specified
        if request.response_format and not tool_calls:
            cleaned_text, parsed_json, is_valid, error = parse_json_output(
                cleaned_text, request.response_format
            )
            if parsed_json is not None:
                cleaned_text = json.dumps(parsed_json)
            if not is_valid:
                logger.warning(f"JSON validation failed: {error}")

        # Buffered mode: emit thinking and cleaned content now
        if not stream_content:
            if cleaned_thinking:
                chunk = ChatCompletionChunk(
                    id=response_id,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            delta=ChatCompletionChunkDelta(
                                reasoning_content=cleaned_thinking
                            ),
                            finish_reason=None,
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
            if cleaned_text:
                chunk = ChatCompletionChunk(
                    id=response_id,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            delta=ChatCompletionChunkDelta(content=cleaned_text),
                            finish_reason=None,
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

    # Reverse Gemma 4 parameter renaming for streaming path
    if tool_calls and "gemma" in (resolved_model or request.model or "").lower():
        for tc in tool_calls:
            if tc.function and tc.function.arguments:
                try:
                    args = json.loads(tc.function.arguments)
                    args = restore_gemma4_param_names(args)
                    tc.function.arguments = json.dumps(args, ensure_ascii=False)
                except (json.JSONDecodeError, AttributeError):
                    pass

    # Emit tool call chunks if found
    if tool_calls:
        for i, tc in enumerate(tool_calls):
            tc_chunk = ChatCompletionChunk(
                id=response_id,
                model=request.model,
                choices=[
                    ChatCompletionChunkChoice(
                        delta=ChatCompletionChunkDelta(
                            tool_calls=[
                                {
                                    "index": i,
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                            ],
                        ),
                    )
                ],
            )
            yield f"data: {tc_chunk.model_dump_json(exclude_none=True)}\n\n"

    # Final chunk with finish_reason
    finish_reason = (
        "tool_calls"
        if tool_calls
        else (last_output.finish_reason if last_output else "stop")
    )
    final_chunk = ChatCompletionChunk(
        id=response_id,
        model=request.model,
        choices=[
            ChatCompletionChunkChoice(
                delta=ChatCompletionChunkDelta(),
                finish_reason=finish_reason,
            )
        ],
    )
    yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n"

    # Record metrics and emit usage chunk
    if last_output and last_output.finished:
        end_time = time.perf_counter()
        total_duration = end_time - start_time
        ttft = (first_token_time - start_time) if first_token_time else total_duration
        is_diffusion = getattr(engine, "is_diffusion_model", False)
        if is_diffusion:
            gen_duration = total_duration
        else:
            gen_duration = end_time - (first_token_time or start_time)
        metric_prefill_duration, metric_gen_duration = _resolve_metric_durations(
            last_output,
            is_diffusion=is_diffusion,
            prefill_duration=ttft,
            generation_duration=gen_duration,
        )
        get_server_metrics().record_request_complete(
            prompt_tokens=last_output.prompt_tokens,
            completion_tokens=last_output.completion_tokens,
            cached_tokens=last_output.cached_tokens,
            prefill_duration=metric_prefill_duration,
            generation_duration=metric_gen_duration,
            model_id=resolved_model or request.model,
        )
        speed_duration = total_duration if is_diffusion else gen_duration
        tokens_per_sec = (
            last_output.completion_tokens / speed_duration if speed_duration > 0 else 0
        )
        speed_text = _format_generation_speed_for_log(
            last_output,
            tokens_per_sec,
            is_diffusion=is_diffusion,
        )
        logger.info(
            f"Chat completion: {last_output.completion_tokens} tokens in "
            f"{total_duration:.2f}s ({speed_text}), "
            f"prompt: {last_output.prompt_tokens}, finish_reason={finish_reason}, "
            f"max_tokens={kwargs.get('max_tokens')}, "
            f"request_max_tokens={request.max_tokens}"
        )

        # Emit usage chunk if requested
        if request.stream_options and request.stream_options.include_usage:
            total_time = end_time - start_time
            pt = last_output.prompt_tokens
            ct = last_output.completion_tokens
            usage_chunk = ChatCompletionChunk(
                id=response_id,
                model=request.model,
                choices=[],
                usage=Usage(
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    total_tokens=pt + ct,
                    prompt_tokens_details=PromptTokensDetails(
                        cached_tokens=last_output.cached_tokens,
                    ),
                    model_load_duration=(
                        round(model_load_duration, 2)
                        if model_load_duration > 1.0
                        else None
                    ),
                    time_to_first_token=round(ttft, 2),
                    total_time=round(total_time, 2),
                    prompt_eval_duration=round(metric_prefill_duration, 2),
                    generation_duration=round(metric_gen_duration, 2),
                    prompt_tokens_per_second=(
                        round(pt / metric_prefill_duration, 2)
                        if metric_prefill_duration > 0
                        else None
                    ),
                    generation_tokens_per_second=(
                        round(ct / metric_gen_duration, 2)
                        if metric_gen_duration > 0
                        else None
                    ),
                ),
            )
            yield f"data: {usage_chunk.model_dump_json(exclude_none=True)}\n\n"

    yield "data: [DONE]\n\n"


# =============================================================================
# Anthropic Messages API
# =============================================================================


async def stream_anthropic_messages(
    engine: BaseEngine,
    messages: list,
    request: AnthropicMessagesRequest,
    resolved_model: Optional[str] = None,
    **kwargs,
) -> AsyncIterator[str]:
    """
    Stream Anthropic Messages API response.

    For Harmony models (gpt-oss), separates analysis and final channels:
    - index=0: analysis channel (<think>...</think>) - displayed as thinking
    - index=1: final channel (response text) - displayed as message

    For other models:
    - index=0: all text

    Emits events in Anthropic SSE format:
    1. message_start - Initial message
    2. content_block_start - Start block(s)
    3. content_block_delta - Text chunks
    4. content_block_stop - End block(s)
    5. (tool blocks if present)
    6. message_delta - Final stop_reason and usage
    7. message_stop - End marker
    """
    start_time = time.perf_counter()
    first_token_time = None

    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    accumulated_text = ""

    # Track content blocks with thinking separation. Some templates open the
    # thinking block in the prompt itself, so the generated text starts with
    # reasoning body and only later emits </think>.
    start_in_thinking = False
    try:
        tokenizer = getattr(engine, "tokenizer", None)
        if tokenizer is not None:
            prompt, prompt_token_ids = _render_chat_prompt_for_thinking_detection(
                engine, messages, kwargs
            )
            start_in_thinking, _ = prompt_opens_thinking(
                tokenizer, prompt, prompt_token_ids=prompt_token_ids
            )
    except Exception as exc:
        logger.debug("Could not detect Anthropic stream thinking state: %s", exc)
    thinking_parser = ThinkingParser(start_in_thinking=start_in_thinking)
    thinking_block_started = False
    text_block_started = False
    block_index = 0
    last_output = None  # Track last output for tool_calls and token counts

    # Filter tool-call markup from streamed content when tools are present.
    has_tools = bool(kwargs.get("tools"))
    tool_filter = None
    thinking_filter = None
    if has_tools:
        _content_filter = ToolCallStreamFilter(engine.tokenizer)
        _thinking_filter = ToolCallStreamFilter(engine.tokenizer)
        if _content_filter.active:
            tool_filter = _content_filter
            thinking_filter = _thinking_filter

    # Does the client opt into Anthropic's cache_control accounting?
    # When yes, message_start.input_tokens reports the post-partition value
    # (0 here, since we approximate the whole prompt as belonging to the
    # cache_control region — the final message_delta refines with the real
    # cache hit count). When no, input_tokens carries the full prompt count.
    uses_cache_control = request_has_cache_control(request)

    # Calculate input tokens before streaming starts
    # This is needed for message_start event
    estimated_input_tokens = 0
    try:
        if hasattr(engine, "tokenizer") and engine.tokenizer is not None:
            # Build the prompt using chat template
            template_kwargs = {"tokenize": False, "add_generation_prompt": True}
            if kwargs.get("tools"):
                template_kwargs["tools"] = kwargs["tools"]
            if kwargs.get("chat_template_kwargs"):
                template_kwargs.update(kwargs["chat_template_kwargs"])
            prompt = engine.tokenizer.apply_chat_template(messages, **template_kwargs)
            # Tokenize to count
            tokens = engine.tokenizer.encode(prompt)
            estimated_input_tokens = len(tokens)
    except Exception as e:
        logger.debug(f"Could not estimate input tokens: {e}")

    # 1. Send message_start with estimated input tokens
    yield create_message_start_event(
        message_id=message_id,
        model=request.model,
        input_tokens=(
            0
            if uses_cache_control
            else scale_anthropic_tokens(estimated_input_tokens, request.model)
        ),
    )

    # 3. Stream content with thinking/content separation
    try:
        async for output in engine.stream_chat(messages=messages, **kwargs):
            last_output = output  # Keep reference for tool_calls and token counts

            if first_token_time is None and output.new_text:
                first_token_time = time.perf_counter()

            if output.new_text:
                accumulated_text += output.new_text
                thinking_delta, content_delta = thinking_parser.feed(output.new_text)

                # Emit thinking content as thinking block
                if thinking_delta:
                    if thinking_filter:
                        thinking_delta = thinking_filter.feed(thinking_delta)
                    if thinking_delta:
                        # Close any open text block before starting a new
                        # thinking block at a fresh index. Anthropic SDKs
                        # reject mixed-type content_block events at the same
                        # index — this transition handles a model that emits
                        # a second thinking section after some text.
                        if text_block_started:
                            yield create_content_block_stop_event(index=block_index)
                            block_index += 1
                            text_block_started = False
                        if not thinking_block_started:
                            yield create_content_block_start_event(
                                index=block_index, block_type="thinking"
                            )
                            thinking_block_started = True
                        yield create_thinking_delta_event(
                            index=block_index, thinking=thinking_delta
                        )

                # Emit regular content as text block — filter tool-call
                # markup when a known start marker is available.
                if content_delta:
                    if tool_filter:
                        content_delta = tool_filter.feed(content_delta)
                    if content_delta:
                        # When tools are requested AND we haven't yet opened
                        # a text block, drop pure-whitespace deltas. Models
                        # often emit a leading newline around <tool_call>
                        # envelopes that tool_filter passes through
                        # (whitespace isn't part of the envelope markers).
                        # Without this guard, the `\n` opens a text block
                        # that then holds only whitespace — surfacing as
                        # a phantom empty-ish text block before the
                        # tool_use blocks.
                        if (
                            not text_block_started
                            and kwargs.get("tools")
                            and not content_delta.strip()
                        ):
                            pass  # drop leading whitespace adjacent to tool envelopes
                        else:
                            # Close thinking block if transitioning to text
                            if thinking_block_started and not text_block_started:
                                yield create_content_block_stop_event(index=block_index)
                                block_index += 1
                                thinking_block_started = False
                            if not text_block_started:
                                yield create_content_block_start_event(
                                    index=block_index, block_type="text"
                                )
                                text_block_started = True
                            yield create_text_delta_event(
                                index=block_index, text=content_delta
                            )

            if output.finished:
                break
    except Exception as e:
        logger.error(f"Error during Anthropic streaming: {e}")
        yield create_error_event("api_error", str(e))
        yield create_message_stop_event()
        return

    # Flush remaining buffered content from thinking parser
    thinking_delta, content_delta = thinking_parser.finish()
    if thinking_delta:
        if thinking_filter:
            thinking_delta = thinking_filter.feed(thinking_delta)
        if thinking_delta:
            if text_block_started:
                yield create_content_block_stop_event(index=block_index)
                block_index += 1
                text_block_started = False
            if not thinking_block_started:
                yield create_content_block_start_event(
                    index=block_index, block_type="thinking"
                )
                thinking_block_started = True
            yield create_thinking_delta_event(
                index=block_index, thinking=thinking_delta
            )
    if thinking_filter:
        remaining_thinking = thinking_filter.finish()
        if remaining_thinking:
            if text_block_started:
                yield create_content_block_stop_event(index=block_index)
                block_index += 1
                text_block_started = False
            if not thinking_block_started:
                yield create_content_block_start_event(
                    index=block_index, block_type="thinking"
                )
                thinking_block_started = True
            yield create_thinking_delta_event(
                index=block_index, thinking=remaining_thinking
            )
    if content_delta:
        if tool_filter:
            content_delta = tool_filter.feed(content_delta)
        if content_delta:
            if thinking_block_started and not text_block_started:
                yield create_content_block_stop_event(index=block_index)
                block_index += 1
                thinking_block_started = False
            if not text_block_started:
                yield create_content_block_start_event(
                    index=block_index, block_type="text"
                )
                text_block_started = True
            yield create_text_delta_event(index=block_index, text=content_delta)

    # Flush any remaining buffered content from the tool-call filter
    if tool_filter:
        remaining = tool_filter.finish()
        if remaining:
            if not text_block_started:
                if thinking_block_started:
                    yield create_content_block_stop_event(index=block_index)
                    block_index += 1
                    thinking_block_started = False
                yield create_content_block_start_event(
                    index=block_index, block_type="text"
                )
                text_block_started = True
            yield create_text_delta_event(index=block_index, text=remaining)

    # 5. Handle tool calls (moved before block-closing so empty-text-block
    # emission can skip when tool_use blocks will follow).
    # For Harmony models, use tool_calls from output (parsed by HarmonyStreamingParser)
    # For other models, parse from accumulated text
    tool_calls = None
    if last_output and last_output.tool_calls:
        # Protocol parser already extracted structured tool calls.
        tool_calls = _convert_parser_tool_calls(last_output.tool_calls)
    elif kwargs.get("tools"):
        # Non-Harmony: separate thinking, then parse tool calls from content
        # (falls back to thinking content for small models)
        thinking_content, regular_content = extract_thinking(accumulated_text)
        extraction = extract_tool_calls_with_thinking(
            thinking_content,
            regular_content,
            tokenizer=engine.tokenizer,
            tools=kwargs.get("tools"),
        )
        tool_calls = extraction.tool_calls

    # 4. Close open blocks
    if thinking_block_started and not text_block_started:
        # Only thinking was emitted. Keep block_index on the just-closed
        # block so following tool_use blocks start at the next contiguous index.
        yield create_content_block_stop_event(index=block_index)
    if text_block_started:
        yield create_content_block_stop_event(index=block_index)
    elif not thinking_block_started and not tool_calls:
        # No content AND no tool_calls — emit an empty text block so the
        # message is well-formed. When tool_calls will follow, skip this —
        # the tool_use blocks carry the semantic content, and an empty
        # preceding text block confuses SDK clients that treat content[0]
        # as authoritative.
        yield create_content_block_start_event(index=block_index, block_type="text")
        yield create_content_block_stop_event(index=block_index)

    # Reverse Gemma 4 parameter renaming
    if tool_calls and "gemma" in (resolved_model or request.model or "").lower():
        for tc in tool_calls:
            if tc.function and tc.function.arguments:
                try:
                    args = json.loads(tc.function.arguments)
                    args = restore_gemma4_param_names(args)
                    tc.function.arguments = json.dumps(args, ensure_ascii=False)
                except (json.JSONDecodeError, AttributeError):
                    pass

    # Emit tool_use blocks if present
    # When neither text nor thinking was streamed AND the empty-text-block
    # emission was skipped (because tool_calls are about to follow), the
    # tool_use block takes index 0. Otherwise it follows the last emitted
    # text/thinking block at block_index+1.
    if not text_block_started and not thinking_block_started:
        tool_block_start = 0
    else:
        tool_block_start = block_index + 1
    if tool_calls:
        for i, tc in enumerate(tool_calls, start=tool_block_start):
            # Start tool_use block
            yield create_content_block_start_event(
                index=i,
                block_type="tool_use",
                id=tc.id,
                name=tc.function.name,
            )
            # Send input as delta
            yield create_input_json_delta_event(
                index=i, partial_json=tc.function.arguments
            )
            # Close tool block
            yield create_content_block_stop_event(index=i)

    # 6. Send message_delta with stop_reason and actual token counts
    stop_reason = map_finish_reason_to_stop_reason(
        output.finish_reason if output else "stop", bool(tool_calls)
    )
    # Use actual token counts from the last output
    actual_input_tokens = scale_anthropic_tokens(
        last_output.prompt_tokens if last_output else 0, request.model
    )
    actual_output_tokens = scale_anthropic_tokens(
        last_output.completion_tokens if last_output else 0, request.model
    )
    actual_cached_tokens = scale_anthropic_tokens(
        last_output.cached_tokens if last_output else 0, request.model
    )
    yield create_message_delta_event(
        stop_reason=stop_reason,
        output_tokens=actual_output_tokens,
        input_tokens=actual_input_tokens,
        cached_tokens=actual_cached_tokens,
        request_uses_cache_control=uses_cache_control,
    )

    # Record metrics
    if last_output:
        end_time = time.perf_counter()
        total_duration = end_time - start_time
        ttft = (first_token_time - start_time) if first_token_time else total_duration
        if getattr(engine, "is_diffusion_model", False):
            gen_duration = total_duration
        else:
            gen_duration = end_time - (first_token_time or start_time)
        get_server_metrics().record_request_complete(
            prompt_tokens=last_output.prompt_tokens,
            completion_tokens=last_output.completion_tokens,
            cached_tokens=last_output.cached_tokens,
            prefill_duration=ttft,
            generation_duration=gen_duration,
            model_id=resolved_model or request.model,
        )

    # 7. Send message_stop
    yield create_message_stop_event()


@app.post("/v1/messages")
async def create_anthropic_message(
    request: AnthropicMessagesRequest,
    http_request: FastAPIRequest,
    _: bool = Depends(verify_api_key),
):
    """
    Create a message using Anthropic Messages API format.

    This endpoint provides compatibility with Anthropic's Messages API,
    allowing clients that use Anthropic SDK to work with oMLX.

    Example request:
    ```json
    {
        "model": "claude-3-sonnet",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ]
    }
    ```

    Streaming is supported with `stream: true`.
    """
    logger.debug(
        f"Anthropic Messages request: model={request.model}, "
        f"messages={len(request.messages)}, stream={request.stream}, "
        f"max_tokens={request.max_tokens}"
    )

    if _server_state.oq_manager and _server_state.oq_manager.is_quantizing:
        raise HTTPException(
            status_code=503,
            detail="Server is busy with oQ quantization. Please try again after quantization completes.",
        )

    lease = _LLMEngineLease()
    try:
        engine = await get_engine_for_model(request.model, lease=lease)

        # Resolve alias to real model ID for settings lookups
        resolved_model = resolve_model_id(request.model) or request.model

        # Get per-model settings
        max_tool_result_tokens = None
        merged_ct_kwargs = {}
        forced_keys: set[str] = set()
        ms = get_model_settings_for_request(request.model)
        if ms:
            max_tool_result_tokens = ms.max_tool_result_tokens
            if ms.chat_template_kwargs:
                merged_ct_kwargs.update(ms.chat_template_kwargs)
            forced_keys = set(ms.forced_ct_kwargs or [])
            # Dedicated enable_thinking toggle takes precedence over chat_template_kwargs
            if ms.enable_thinking is not None:
                merged_ct_kwargs["enable_thinking"] = ms.enable_thinking
            # preserve_thinking: keep <think> blocks in historical turns (Qwen 3.6+)
            if ms.preserve_thinking is not None:
                merged_ct_kwargs["preserve_thinking"] = ms.preserve_thinking
        # Per-request kwargs override model settings (except forced keys)
        if request.chat_template_kwargs:
            for k, v in request.chat_template_kwargs.items():
                if k not in forced_keys:
                    merged_ct_kwargs[k] = v

        # Pass Anthropic thinking config to chat template (except forced keys)
        if hasattr(request, "thinking") and request.thinking:
            if "enable_thinking" not in forced_keys:
                thinking_type = getattr(request.thinking, "type", None)
                if thinking_type in ("enabled", "adaptive"):
                    merged_ct_kwargs["enable_thinking"] = True
                elif thinking_type == "disabled":
                    merged_ct_kwargs["enable_thinking"] = False

        logger.debug(
            f"Tool result truncation config: max_tokens={max_tool_result_tokens}, "
            f"has_tokenizer={engine.tokenizer is not None}"
        )

        # Convert Anthropic format to internal format
        # Harmony models need special handling to preserve tool format
        is_vlm = isinstance(engine, VLMBatchedEngine)
        is_dflash_vlm = not is_vlm and getattr(
            engine, "supports_multimodal_fallback", False
        )
        _entry = get_engine_pool().get_entry(resolved_model)
        native_reasoning = uses_native_reasoning_content(
            resolved_model,
            config_model_type=(
                getattr(_entry, "config_model_type", None)
                if _entry is not None
                else None
            ),
            engine_model_type=getattr(engine, "model_type", None),
            preserve_thinking_default=(
                getattr(_entry, "preserve_thinking_default", None)
                if _entry is not None
                else None
            ),
        )
        if engine.model_type == "gpt_oss":
            messages = convert_anthropic_to_internal_harmony(
                request,
                max_tool_result_tokens,
                engine.tokenizer,
                consolidate_system_messages=False,
            )
        else:
            messages = convert_anthropic_to_internal(
                request,
                max_tool_result_tokens,
                engine.tokenizer,
                preserve_images=is_vlm or is_dflash_vlm,
                native_reasoning_content=native_reasoning,
                consolidate_system_messages=False,
            )

        # Apply model-specific message extraction (e.g. Gemma 4 converts
        # role=tool messages into tool_responses on assistant turns).
        extractor = getattr(engine, "message_extractor", None)
        merge_system_fallback_roles = not (is_vlm or is_dflash_vlm)
        if extractor is not None:
            extractor_kwargs = {}
            try:
                if (
                    "consolidate_system_messages"
                    in inspect.signature(extractor).parameters
                ):
                    extractor_kwargs["consolidate_system_messages"] = False
            except (TypeError, ValueError):
                pass
            messages = extractor(
                messages,
                max_tool_result_tokens,
                engine.tokenizer,
                **extractor_kwargs,
            )
            merge_system_fallback_roles = True

        # Detect and strip partial mode at the API boundary — exactly once.
        is_partial = detect_and_strip_partial(messages)

        # Prepare kwargs
        (
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            min_p,
            presence_penalty,
            frequency_penalty,
            max_tokens,
            xtc_probability,
            xtc_threshold,
        ) = get_sampling_params(
            request.temperature,
            request.top_p,
            request.model,
            req_top_k=getattr(request, "top_k", None),
            req_repetition_penalty=getattr(request, "repetition_penalty", None),
            req_max_tokens=request.max_tokens,
        )

        chat_kwargs = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "repetition_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "xtc_probability": xtc_probability,
            "xtc_threshold": xtc_threshold,
        }

        # Add thinking budget if applicable
        thinking_budget = _resolve_thinking_budget(request, request.model)
        if thinking_budget is not None:
            chat_kwargs["thinking_budget"] = thinking_budget

        # Auto-set enable_thinking in chat template kwargs when a thinking
        # budget is active but enable_thinking was not already set (e.g. via
        # the Anthropic thinking.type field above or model settings).
        if thinking_budget is not None and "enable_thinking" not in merged_ct_kwargs:
            merged_ct_kwargs["enable_thinking"] = True

        # Auto-set preserve_thinking only when the template advertises support
        # for it (Qwen 3.6+). Gated on detection so other templates don't
        # receive an unknown kwarg.
        _entry = get_engine_pool().get_entry(resolved_model)
        if (
            _entry is not None
            and _entry.preserve_thinking_default is True
            and merged_ct_kwargs.get("enable_thinking") is not False
            and "preserve_thinking" not in merged_ct_kwargs
        ):
            merged_ct_kwargs["preserve_thinking"] = True

        # Merge MCP tools with user-provided Anthropic tools
        user_internal = convert_anthropic_tools_to_internal(request.tools)
        if getattr(engine, "is_diffusion_model", False) and not getattr(
            engine, "supports_tool_calling", False
        ):
            if user_internal:
                raise InvalidRequestError(
                    "Tool calling is not supported for this diffusion model "
                    "(no tool parser matched its chat template).",
                    field="tools",
                )
            internal_tools = None
        elif _server_state.mcp_manager:
            mcp_openai_tools = _server_state.mcp_manager.get_all_tools_openai()
            combined = (mcp_openai_tools or []) + (user_internal or [])
            # Deduplicate by function name (user tools take precedence)
            if combined:
                seen = {}
                for tool in combined:
                    name = tool.get("function", {}).get("name", "")
                    seen[name] = tool
                internal_tools = list(seen.values())
            else:
                internal_tools = None
        else:
            internal_tools = user_internal
        # Gemma 4 drops required params that lack descriptions — enrich them
        if internal_tools and "gemma" in (resolved_model or "").lower():
            internal_tools = enrich_tool_params_for_gemma4(internal_tools)
        if internal_tools:
            chat_kwargs["tools"] = internal_tools

        # Add chat template kwargs
        if merged_ct_kwargs:
            chat_kwargs["chat_template_kwargs"] = merged_ct_kwargs

        # Forward partial-mode decision to the engine explicitly
        chat_kwargs["is_partial"] = is_partial

        await _ensure_tokenizer_for_system_probe(engine, messages)
        messages = prepare_system_messages_for_template(
            messages,
            engine.tokenizer,
            tools=internal_tools,
            chat_template_kwargs=merged_ct_kwargs or None,
            is_partial=is_partial,
            merge_consecutive_roles=merge_system_fallback_roles,
            unsupported_mid_system_policy=_unsupported_mid_system_policy(),
        )

        # Validate context window before sending to model
        try:
            num_prompt_tokens = engine.count_chat_tokens(
                messages,
                internal_tools,
                chat_template_kwargs=merged_ct_kwargs or None,
                is_partial=is_partial,
            )
        except Exception as e:
            err_name = type(e).__name__.lower()
            err_msg = str(e).lower()
            if (
                "template" in err_name
                or "template" in err_msg
                or isinstance(e, (AssertionError, ValueError))
            ):
                raise HTTPException(status_code=400, detail=f"Chat template error: {e}")
            raise
        validate_context_window(num_prompt_tokens, request.model)

        # Add stop sequences
        if request.stop_sequences:
            chat_kwargs["stop"] = request.stop_sequences

        # Pre-flight prefill memory guard — must precede any StreamingResponse
        # return so PrefillMemoryExceededError can be mapped to HTTP 400.
        await _raise_if_llm_lease_abort_requested(lease)
        await engine.preflight_chat(
            messages,
            request_id=http_request.headers.get("x-request-id"),
            **chat_kwargs,
        )
        await _raise_if_llm_lease_abort_requested(lease)

        if request.stream:
            return StreamingResponse(
                _release_after_stream(
                    _with_sse_keepalive(
                        stream_anthropic_messages(
                            engine,
                            messages,
                            request,
                            resolved_model=resolved_model,
                            **chat_kwargs,
                        ),
                        http_request=http_request,
                        keepalive_chunk=_resolve_keepalive("anthropic"),
                    ),
                    lease,
                ),
                media_type="text/event-stream",
                headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
            )

        # Non-streaming response with keepalive during prefill
        async def _build_anthropic_message():
            await _raise_if_llm_lease_abort_requested(lease)
            start_time = time.perf_counter()

            output = await engine.chat(messages=messages, **chat_kwargs)

            elapsed = time.perf_counter() - start_time
            tokens_per_sec = output.completion_tokens / elapsed if elapsed > 0 else 0
            logger.info(
                f"Anthropic message: {output.completion_tokens} tokens in {elapsed:.2f}s "
                f"({tokens_per_sec:.1f} tok/s)"
            )

            get_server_metrics().record_request_complete(
                prompt_tokens=output.prompt_tokens,
                completion_tokens=output.completion_tokens,
                cached_tokens=output.cached_tokens,
                generation_duration=elapsed,
                model_id=resolved_model,
            )

            # Separate thinking from content
            raw_text = clean_special_tokens(output.text) if output.text else ""
            thinking_content, regular_content = extract_thinking(raw_text)
            cleaned_thinking = sanitize_tool_call_markup(
                thinking_content, engine.tokenizer
            )

            # Protocol parsers can return structured tool_calls directly.
            if output.tool_calls:
                tool_calls = _convert_parser_tool_calls(output.tool_calls)
                cleaned_text = regular_content
            else:
                extraction = extract_tool_calls_with_thinking(
                    thinking_content,
                    regular_content,
                    tokenizer=engine.tokenizer,
                    tools=internal_tools,
                )
                cleaned_text = extraction.cleaned_text
                tool_calls = extraction.tool_calls
                cleaned_thinking = extraction.cleaned_thinking

            # Reverse Gemma 4 parameter renaming
            if tool_calls and "gemma" in (resolved_model or "").lower():
                for tc in tool_calls:
                    if tc.function and tc.function.arguments:
                        try:
                            args = json.loads(tc.function.arguments)
                            args = restore_gemma4_param_names(args)
                            tc.function.arguments = json.dumps(args, ensure_ascii=False)
                        except (json.JSONDecodeError, AttributeError):
                            pass

            response = convert_internal_to_anthropic_response(
                text=cleaned_text.strip() if cleaned_text else "",
                model=request.model,
                prompt_tokens=scale_anthropic_tokens(
                    output.prompt_tokens, request.model
                ),
                completion_tokens=scale_anthropic_tokens(
                    output.completion_tokens, request.model
                ),
                finish_reason=output.finish_reason,
                tool_calls=tool_calls,
                thinking=cleaned_thinking if cleaned_thinking else None,
                cached_tokens=scale_anthropic_tokens(
                    output.cached_tokens, request.model
                ),
                request_uses_cache_control=request_has_cache_control(request),
            )

            return response.model_dump_json()

        return StreamingResponse(
            _release_after_stream(
                _with_json_keepalive(http_request, _build_anthropic_message()),
                lease,
            ),
            media_type="application/json",
        )

    except BaseException:
        await lease.release()
        raise


@app.post("/v1/messages/count_tokens")
async def count_anthropic_tokens(
    request: TokenCountRequest,
    _: bool = Depends(verify_api_key),
):
    """
    Count tokens in a message request.

    Uses the loaded model's tokenizer to accurately count tokens
    including system prompt, messages, and tools.

    This is compatible with Anthropic's token counting API.
    """
    if _server_state.oq_manager and _server_state.oq_manager.is_quantizing:
        raise HTTPException(
            status_code=503,
            detail="Server is busy with oQ quantization. Please try again after quantization completes.",
        )

    lease = _LLMEngineLease()
    try:
        engine = await get_engine_for_model(request.model, lease=lease)
        await _raise_if_llm_lease_abort_requested(lease)

        # Convert Anthropic format to internal format
        # Create a temporary MessagesRequest to reuse existing conversion logic
        temp_request = AnthropicMessagesRequest(
            model=request.model,
            max_tokens=1,  # Dummy value, not used for token counting
            messages=request.messages,
            system=request.system,
            tools=request.tools,
            tool_choice=request.tool_choice,
            thinking=request.thinking,
        )
        messages = convert_anthropic_to_internal(temp_request)

        # Convert tools if present
        internal_tools = convert_anthropic_tools_to_internal(request.tools)

        # Apply chat template to get prompt
        tokenizer = engine.tokenizer
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if internal_tools:
            template_kwargs["tools"] = internal_tools

        try:
            prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
        except Exception as e:
            logger.warning(
                f"Failed to apply chat template: {e}, using simple concatenation"
            )
            # Fallback: simple concatenation
            prompt = "\n".join(
                f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                for msg in messages
            )

        # Tokenize to count tokens
        if isinstance(prompt, str):
            token_ids = tokenizer.encode(prompt)
        else:
            token_ids = prompt  # Already tokenized

        input_tokens = scale_anthropic_tokens(len(token_ids), request.model)
        logger.debug(f"Token count: {input_tokens} tokens for {len(messages)} messages")

        return TokenCountResponse(input_tokens=input_tokens)

    finally:
        await lease.release()


# =============================================================================
# Responses API (/v1/responses) — OpenAI Codex compatibility
# =============================================================================


def _should_store_response(store_flag: Optional[bool]) -> bool:
    """OpenAI Responses defaults to storing responses unless explicitly disabled."""
    return store_flag is not False


def _resolve_previous_response_messages(previous_response_id: str) -> list[dict]:
    """Resolve a previous_response_id chain into chat messages."""
    try:
        return _server_state.responses_store.resolve_chain_messages(
            previous_response_id
        )
    except ResponseStateNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=(
                "Response state not found for previous_response_id. "
                "It may have been deleted, evicted, or lost after restart."
            ),
        ) from exc
    except ResponseStateCorruptError as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                "Stored response state is incomplete or corrupted for "
                "previous_response_id."
            ),
        ) from exc


def _store_response_state(
    public_response: dict,
    input_messages: list[dict],
) -> None:
    """Persist the response object and the normalized conversation state."""
    output_messages = normalize_response_output_to_messages(
        public_response.get("output", [])
    )
    record = build_response_store_record(
        public_response,
        input_messages=input_messages,
        output_messages=output_messages,
    )
    _server_state.responses_store.put(public_response["id"], record)


@app.post("/v1/responses")
async def create_response(
    request: ResponsesRequest,
    http_request: FastAPIRequest,
    _: bool = Depends(verify_api_key),
):
    """Create a response (OpenAI Responses API)."""
    if _server_state.oq_manager and _server_state.oq_manager.is_quantizing:
        raise HTTPException(
            status_code=503,
            detail="Server is busy with oQ quantization. Please try again after quantization completes.",
        )

    logger.debug(
        f"Responses API request: model={request.model}, stream={request.stream}"
    )

    load_start = time.perf_counter()
    lease = _LLMEngineLease()
    try:
        engine = await get_engine_for_model(request.model, lease=lease)
        model_load_duration = time.perf_counter() - load_start

        resolved_model = resolve_model_id(request.model) or request.model

        current_input_messages = convert_responses_input_to_messages(
            request.input,
            consolidate_system_messages=False,
        )

        # Build previous context from previous_response_id
        previous_messages = None
        if request.previous_response_id:
            previous_messages = _resolve_previous_response_messages(
                request.previous_response_id
            )

        # Convert Responses API input → internal messages
        messages = convert_responses_input_to_messages(
            request.input,
            request.instructions,
            previous_messages,
            consolidate_system_messages=False,
        )

        # Convert tools: flat → nested
        openai_tools = convert_responses_tools(request.tools)
        if (
            getattr(engine, "is_diffusion_model", False)
            and not getattr(engine, "supports_tool_calling", False)
            and openai_tools
        ):
            raise InvalidRequestError(
                "Tool calling is not supported for this diffusion model "
                "(no tool parser matched its chat template).",
                field="tools",
            )

        # Get per-model settings
        merged_ct_kwargs = {}
        reasoning_parser = None
        ms = get_model_settings_for_request(request.model)
        if ms:
            reasoning_parser = ms.reasoning_parser
            if ms.chat_template_kwargs:
                merged_ct_kwargs.update(ms.chat_template_kwargs)
            # Dedicated enable_thinking toggle takes precedence over chat_template_kwargs
            if ms.enable_thinking is not None:
                merged_ct_kwargs["enable_thinking"] = ms.enable_thinking
            # preserve_thinking: keep <think> blocks in historical turns (Qwen 3.6+)
            if ms.preserve_thinking is not None:
                merged_ct_kwargs["preserve_thinking"] = ms.preserve_thinking

        # Note: extract_text_content/extract_harmony_messages/extract_multimodal_content
        # are NOT called here because convert_responses_input_to_messages() already
        # returns plain dicts in {"role": str, "content": str} format.
        # Those extract functions expect Pydantic Message objects from OpenAI/Anthropic requests.

        # Handle text.format (structured output)
        response_format = None
        compiled_grammar = None
        if request.text and request.text.format:
            fmt = request.text.format
            if fmt.type == "json_object":
                response_format = {"type": "json_object"}
            elif fmt.type == "json_schema":
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": fmt.name or "response",
                        "schema": fmt.schema_ or {},
                        "strict": fmt.strict or False,
                    },
                }
            if response_format:
                from .api.openai_models import ResponseFormat

                _reject_diffusion_structured_outputs(
                    engine,
                    response_format=response_format,
                )
                await engine.start()
                rf = ResponseFormat(**response_format)
                compiled_grammar = _compile_grammar_for_request(
                    engine,
                    response_format=rf,
                    chat_template_kwargs=merged_ct_kwargs or None,
                    reasoning_parser=reasoning_parser,
                )
                if compiled_grammar is None:
                    json_instruction = build_json_system_prompt(rf)
                    if json_instruction:
                        messages = _inject_json_instruction(messages, json_instruction)
            else:
                compiled_grammar = None

        # Merge MCP tools
        effective_tools = (
            None
            if (
                getattr(engine, "is_diffusion_model", False)
                and not getattr(engine, "supports_tool_calling", False)
            )
            else openai_tools
        )
        if _server_state.mcp_manager and effective_tools:
            effective_tools = _server_state.mcp_manager.get_merged_tools(openai_tools)

        # Convert tools for chat template
        tools_for_template = (
            convert_tools_for_template(effective_tools) if effective_tools else None
        )
        # Gemma 4 drops required params that lack descriptions — enrich them
        if tools_for_template and "gemma" in (resolved_model or "").lower():
            tools_for_template = enrich_tool_params_for_gemma4(tools_for_template)
        await _ensure_tokenizer_for_system_probe(engine, messages)
        messages = prepare_system_messages_for_template(
            messages,
            engine.tokenizer,
            tools=tools_for_template,
            chat_template_kwargs=merged_ct_kwargs or None,
            is_partial=False,
            merge_consecutive_roles=True,
            unsupported_mid_system_policy=_unsupported_mid_system_policy(),
        )

        # Validate context window
        try:
            num_prompt_tokens = engine.count_chat_tokens(
                messages,
                tools_for_template,
                chat_template_kwargs=merged_ct_kwargs or None,
            )
        except Exception as e:
            err_name = type(e).__name__.lower()
            err_msg = str(e).lower()
            if (
                "template" in err_name
                or "template" in err_msg
                or isinstance(e, (AssertionError, ValueError))
            ):
                raise HTTPException(status_code=400, detail=f"Chat template error: {e}")
            raise
        validate_context_window(num_prompt_tokens, request.model)

        # Build sampling kwargs
        (
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            min_p,
            presence_penalty,
            frequency_penalty,
            max_tokens,
            xtc_probability,
            xtc_threshold,
        ) = get_sampling_params(
            request.temperature,
            request.top_p,
            request.model,
            req_top_k=getattr(request, "top_k", None),
            req_repetition_penalty=getattr(request, "repetition_penalty", None),
            req_max_tokens=request.max_output_tokens,
        )
        chat_kwargs = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "repetition_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "xtc_probability": xtc_probability,
            "xtc_threshold": xtc_threshold,
        }

        # Add seed for reproducible generation (best-effort)
        if request.seed is not None:
            chat_kwargs["seed"] = request.seed

        # Add thinking budget if applicable
        thinking_budget = _resolve_thinking_budget(request, request.model)
        if thinking_budget is not None:
            chat_kwargs["thinking_budget"] = thinking_budget

        # Auto-set enable_thinking when thinking budget is active.
        if thinking_budget is not None and "enable_thinking" not in merged_ct_kwargs:
            merged_ct_kwargs["enable_thinking"] = True

        # Auto-set preserve_thinking only when the template advertises support
        # for it (Qwen 3.6+). Gated on detection so other templates don't
        # receive an unknown kwarg.
        _entry = get_engine_pool().get_entry(resolved_model)
        native_reasoning = bool(_entry and _entry.preserve_thinking_default is True)
        if (
            native_reasoning
            and merged_ct_kwargs.get("enable_thinking") is not False
            and "preserve_thinking" not in merged_ct_kwargs
        ):
            merged_ct_kwargs["preserve_thinking"] = True

        # Add compiled grammar for logit-level structured output.
        if compiled_grammar is not None:
            chat_kwargs["compiled_grammar"] = compiled_grammar
            if reasoning_parser and "thinking_budget" not in chat_kwargs:
                default_budget = min(max_tokens // 2, 4096)
                chat_kwargs["thinking_budget"] = default_budget
                logger.debug(
                    "Auto-set thinking_budget=%d for grammar-constrained request",
                    default_budget,
                )

        if tools_for_template:
            chat_kwargs["tools"] = tools_for_template
        if merged_ct_kwargs:
            chat_kwargs["chat_template_kwargs"] = merged_ct_kwargs

        # Pre-flight prefill memory guard — must precede any StreamingResponse
        # return so PrefillMemoryExceededError can be mapped to HTTP 400.
        await _raise_if_llm_lease_abort_requested(lease)
        await engine.preflight_chat(
            messages,
            request_id=http_request.headers.get("x-request-id"),
            **chat_kwargs,
        )
        await _raise_if_llm_lease_abort_requested(lease)

        if request.stream:
            return StreamingResponse(
                _release_after_stream(
                    _with_sse_keepalive(
                        stream_responses_api(
                            engine,
                            messages,
                            request,
                            input_messages=current_input_messages,
                            store_response=_should_store_response(request.store),
                            model_load_duration=model_load_duration,
                            resolved_model=resolved_model,
                            response_format=response_format,
                            native_reasoning=native_reasoning,
                            **chat_kwargs,
                        ),
                        http_request=http_request,
                        keepalive_chunk=_resolve_keepalive("openai_responses"),
                    ),
                    lease,
                ),
                media_type="text/event-stream",
                headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
            )

        # Non-streaming with keepalive during prefill
        async def _build_responses_api():
            await _raise_if_llm_lease_abort_requested(lease)
            start_time = time.perf_counter()
            output = await engine.chat(messages=messages, **chat_kwargs)

            elapsed = time.perf_counter() - start_time
            tokens_per_sec = output.completion_tokens / elapsed if elapsed > 0 else 0
            logger.info(
                f"Responses API: {output.completion_tokens} tokens in {elapsed:.2f}s "
                f"({tokens_per_sec:.1f} tok/s)"
            )

            get_server_metrics().record_request_complete(
                prompt_tokens=output.prompt_tokens,
                completion_tokens=output.completion_tokens,
                cached_tokens=output.cached_tokens,
                generation_duration=elapsed,
                model_id=resolved_model,
            )

            # Process output text
            raw_text = clean_special_tokens(output.text) if output.text else ""
            thinking_content, regular_content = extract_thinking(raw_text)

            # Parse tool calls
            if output.tool_calls:
                tool_calls = _convert_parser_tool_calls(output.tool_calls)
                cleaned_text = regular_content
                cleaned_thinking = sanitize_tool_call_markup(
                    thinking_content, engine.tokenizer
                )
            else:
                extraction = extract_tool_calls_with_thinking(
                    thinking_content,
                    regular_content,
                    tokenizer=engine.tokenizer,
                    tools=tools_for_template,
                )
                cleaned_text = extraction.cleaned_text
                tool_calls = extraction.tool_calls
                cleaned_thinking = extraction.cleaned_thinking

            # Reverse Gemma 4 parameter renaming
            if tool_calls and "gemma" in (resolved_model or "").lower():
                for tc in tool_calls:
                    fn = getattr(tc, "function", None)
                    if fn and fn.arguments:
                        try:
                            args = json.loads(fn.arguments)
                            args = restore_gemma4_param_names(args)
                            fn.arguments = json.dumps(args, ensure_ascii=False)
                        except (json.JSONDecodeError, AttributeError):
                            pass

            # Process response_format if specified
            if response_format and not tool_calls:
                cleaned_text, parsed_json, is_valid, error = parse_json_output(
                    cleaned_text or regular_content, response_format
                )
                if parsed_json is not None:
                    cleaned_text = json.dumps(parsed_json)
                if not is_valid:
                    logger.warning(f"JSON validation failed: {error}")

            # Build output items
            output_items: list[OutputItem] = []
            reasoning_text = (cleaned_thinking or "").strip()
            if reasoning_text:
                output_items.append(build_reasoning_output_item(reasoning_text))
            output_items.append(
                build_message_output_item(cleaned_text.strip() if cleaned_text else "")
            )

            if tool_calls:
                for tc in tool_calls:
                    if hasattr(tc, "function"):
                        call_id = tc.id
                        name = tc.function.name
                        arguments = tc.function.arguments
                    elif isinstance(tc, dict):
                        call_id = tc.get(
                            "call_id", tc.get("id", f"call_{uuid.uuid4().hex[:8]}")
                        )
                        name = tc.get("name", "")
                        arguments = tc.get("arguments", "{}")
                    else:
                        continue
                    output_items.append(
                        build_function_call_output_item(
                            name=name,
                            arguments=arguments,
                            call_id=call_id,
                        )
                    )

            reasoning_token_count = (
                len(engine.tokenizer.encode(reasoning_text)) if reasoning_text else 0
            )
            usage = build_response_usage(
                input_tokens=output.prompt_tokens,
                output_tokens=output.completion_tokens,
                reasoning_tokens=reasoning_token_count,
                cached_tokens=output.cached_tokens,
            )

            response_obj = ResponseObject(
                model=request.model,
                status="completed",
                output=output_items,
                usage=usage,
                tools=request.tools or [],
                tool_choice=request.tool_choice or "auto",
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=request.max_output_tokens,
                previous_response_id=request.previous_response_id,
            )

            # Store response
            if _should_store_response(request.store):
                _store_response_state(
                    response_obj.model_dump(exclude_none=True),
                    input_messages=current_input_messages,
                )

            return response_obj.model_dump_json()

        return StreamingResponse(
            _release_after_stream(
                _with_json_keepalive(http_request, _build_responses_api()),
                lease,
            ),
            media_type="application/json",
        )

    except BaseException:
        await lease.release()
        raise


async def stream_responses_api(
    engine: BaseEngine,
    messages: list,
    request: ResponsesRequest,
    input_messages: Optional[list[dict]] = None,
    store_response: bool = True,
    model_load_duration: float = 0.0,
    resolved_model: Optional[str] = None,
    response_format=None,
    native_reasoning: bool = False,
    **kwargs,
) -> AsyncIterator[str]:
    """Stream Responses API events (SSE with named event types)."""
    from .api.shared_models import IDPrefix, generate_id

    start_time = time.perf_counter()
    first_token_time = None
    last_output = None
    accumulated_text = ""
    accumulated_reasoning = ""
    has_tools = bool(kwargs.get("tools"))
    thinking_parser = ThinkingParser(start_in_thinking=native_reasoning)
    seq = 0

    response_id = generate_id(IDPrefix.RESPONSE)
    msg_id = generate_id(IDPrefix.MESSAGE)
    reasoning_id = generate_id(IDPrefix.REASONING)

    # Lazy item emission state — items are opened on first token
    reasoning_opened = False
    reasoning_closed = False
    message_opened = False
    next_output_index = 0
    reasoning_output_index: Optional[int] = None  # captured when reasoning opens
    msg_output_index: Optional[int] = None  # captured when message opens

    # Build initial response object (in_progress, empty output)
    initial_response = ResponseObject(
        id=response_id,
        model=request.model,
        status="in_progress",
        output=[],
        tools=request.tools or [],
        tool_choice=request.tool_choice or "auto",
        temperature=request.temperature,
        top_p=request.top_p,
        max_output_tokens=request.max_output_tokens,
        previous_response_id=request.previous_response_id,
    )
    initial_data = initial_response.model_dump(exclude_none=True)

    # 1. response.created
    seq += 1
    yield format_sse_event(
        "response.created",
        {
            "type": "response.created",
            "response": initial_data,
            "sequence_number": seq,
        },
    )

    # 2. response.in_progress
    seq += 1
    yield format_sse_event(
        "response.in_progress",
        {
            "type": "response.in_progress",
            "response": initial_data,
            "sequence_number": seq,
        },
    )

    # --- helper closures for lazy item emission ----------------------
    def _open_reasoning():
        nonlocal seq, reasoning_opened, reasoning_output_index, next_output_index
        if reasoning_opened:
            return []
        reasoning_opened = True
        reasoning_output_index = next_output_index
        next_output_index += 1
        events = []
        seq += 1
        events.append(
            format_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": reasoning_output_index,
                    "item": {
                        "type": "reasoning",
                        "id": reasoning_id,
                        "status": "in_progress",
                        "summary": [],
                    },
                    "sequence_number": seq,
                },
            )
        )
        seq += 1
        events.append(
            format_sse_event(
                "response.reasoning_summary_part.added",
                {
                    "type": "response.reasoning_summary_part.added",
                    "item_id": reasoning_id,
                    "output_index": reasoning_output_index,
                    "summary_index": 0,
                    "part": {"type": "summary_text", "text": ""},
                    "sequence_number": seq,
                },
            )
        )
        return events

    def _close_reasoning():
        nonlocal seq, reasoning_closed
        if reasoning_closed or not reasoning_opened:
            return []
        reasoning_closed = True
        reasoning_text = accumulated_reasoning
        events = []
        seq += 1
        events.append(
            format_sse_event(
                "response.reasoning_summary_text.done",
                {
                    "type": "response.reasoning_summary_text.done",
                    "item_id": reasoning_id,
                    "output_index": reasoning_output_index,
                    "summary_index": 0,
                    "text": reasoning_text,
                    "sequence_number": seq,
                },
            )
        )
        seq += 1
        events.append(
            format_sse_event(
                "response.reasoning_summary_part.done",
                {
                    "type": "response.reasoning_summary_part.done",
                    "item_id": reasoning_id,
                    "output_index": reasoning_output_index,
                    "summary_index": 0,
                    "part": {"type": "summary_text", "text": reasoning_text},
                    "sequence_number": seq,
                },
            )
        )
        seq += 1
        events.append(
            format_sse_event(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": reasoning_output_index,
                    "item": {
                        "type": "reasoning",
                        "id": reasoning_id,
                        "status": "completed",
                        "summary": [{"type": "summary_text", "text": reasoning_text}],
                    },
                    "sequence_number": seq,
                },
            )
        )
        return events

    def _open_message():
        nonlocal seq, message_opened, next_output_index, msg_output_index
        if message_opened:
            return []
        message_opened = True
        msg_output_index = next_output_index
        next_output_index += 1
        events = []
        seq += 1
        events.append(
            format_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": msg_output_index,
                    "item": {
                        "type": "message",
                        "id": msg_id,
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                    },
                    "sequence_number": seq,
                },
            )
        )
        seq += 1
        events.append(
            format_sse_event(
                "response.content_part.added",
                {
                    "type": "response.content_part.added",
                    "item_id": msg_id,
                    "output_index": msg_output_index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                    "sequence_number": seq,
                },
            )
        )
        return events

    def _emit_reasoning_delta(delta: str):
        nonlocal seq, accumulated_reasoning
        if not delta:
            return []
        accumulated_reasoning += delta
        events = []
        events.extend(_open_reasoning())
        seq += 1
        events.append(
            format_sse_event(
                "response.reasoning_summary_text.delta",
                {
                    "type": "response.reasoning_summary_text.delta",
                    "item_id": reasoning_id,
                    "output_index": reasoning_output_index,
                    "summary_index": 0,
                    "delta": delta,
                    "sequence_number": seq,
                },
            )
        )
        return events

    # -----------------------------------------------------------------

    # Open message/reasoning items lazily so non-native <think> blocks can still
    # become a leading Responses reasoning item.

    # Stream tokens
    tool_filter = None
    thinking_filter = None
    stream_content = True
    if has_tools:
        _content_filter = ToolCallStreamFilter(engine.tokenizer)
        _thinking_filter = ToolCallStreamFilter(engine.tokenizer)
        if _content_filter.active:
            tool_filter = _content_filter
            thinking_filter = _thinking_filter
        else:
            stream_content = False

    try:
        async for output in engine.stream_chat(messages=messages, **kwargs):
            if first_token_time is None and output.new_text:
                first_token_time = time.perf_counter()
            last_output = output
            if output.new_text:
                accumulated_text += output.new_text

            if stream_content and output.new_text:
                thinking_delta, content_delta = thinking_parser.feed(output.new_text)

                if thinking_delta:
                    if thinking_filter:
                        thinking_delta = thinking_filter.feed(thinking_delta)
                    for ev in _emit_reasoning_delta(thinking_delta):
                        yield ev

                if content_delta:
                    if reasoning_opened and not reasoning_closed:
                        for ev in _close_reasoning():
                            yield ev
                    for ev in _open_message():
                        yield ev
                    if tool_filter:
                        content_delta = tool_filter.feed(content_delta)
                    if content_delta:
                        seq += 1
                        yield format_sse_event(
                            "response.output_text.delta",
                            {
                                "type": "response.output_text.delta",
                                "item_id": msg_id,
                                "output_index": msg_output_index,
                                "content_index": 0,
                                "delta": content_delta,
                                "sequence_number": seq,
                            },
                        )
    except Exception as e:
        logger.error(f"Error during Responses API streaming: {e}")
        seq += 1
        yield format_sse_event(
            "response.failed",
            {
                "type": "response.failed",
                "response": {**initial_data, "status": "failed"},
                "sequence_number": seq,
            },
        )
        return

    # Flush remaining content from parsers
    if stream_content:
        thinking_delta, content_delta = thinking_parser.finish()
        if thinking_delta:
            if thinking_filter:
                thinking_delta = thinking_filter.feed(thinking_delta)
            for ev in _emit_reasoning_delta(thinking_delta):
                yield ev
        if thinking_filter:
            remaining_thinking = thinking_filter.finish()
            for ev in _emit_reasoning_delta(remaining_thinking):
                yield ev
        if content_delta:
            if reasoning_opened and not reasoning_closed:
                for ev in _close_reasoning():
                    yield ev
            for ev in _open_message():
                yield ev
            if tool_filter:
                content_delta = tool_filter.feed(content_delta)
            if content_delta:
                seq += 1
                yield format_sse_event(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "item_id": msg_id,
                        "output_index": msg_output_index,
                        "content_index": 0,
                        "delta": content_delta,
                        "sequence_number": seq,
                    },
                )
        if tool_filter:
            remaining = tool_filter.finish()
            if remaining:
                if reasoning_opened and not reasoning_closed:
                    for ev in _close_reasoning():
                        yield ev
                for ev in _open_message():
                    yield ev
                seq += 1
                yield format_sse_event(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "item_id": msg_id,
                        "output_index": msg_output_index,
                        "content_index": 0,
                        "delta": remaining,
                        "sequence_number": seq,
                    },
                )

    # Parse tool calls from accumulated text
    tool_calls = None
    cleaned_text = accumulated_text
    if last_output and last_output.tool_calls:
        tool_calls = _convert_parser_tool_calls(last_output.tool_calls)
        cleaned_text = ""
    elif has_tools and accumulated_text:
        thinking_content, regular_content = extract_thinking(accumulated_text)
        extraction = extract_tool_calls_with_thinking(
            thinking_content,
            regular_content,
            tokenizer=engine.tokenizer,
            tools=kwargs.get("tools"),
        )
        cleaned_text = extraction.cleaned_text
        tool_calls = extraction.tool_calls
        if not stream_content:
            cleaned_thinking = (extraction.cleaned_thinking or "").strip()
            for ev in _emit_reasoning_delta(cleaned_thinking):
                yield ev
            if reasoning_opened and not reasoning_closed:
                for ev in _close_reasoning():
                    yield ev
        if not stream_content and cleaned_text:
            for ev in _open_message():
                yield ev
            seq += 1
            yield format_sse_event(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "item_id": msg_id,
                    "output_index": msg_output_index,
                    "content_index": 0,
                    "delta": cleaned_text,
                    "sequence_number": seq,
                },
            )
    else:
        # No tools — use raw accumulated text minus thinking.
        thinking_content, regular_content = extract_thinking(accumulated_text)
        cleaned_text = clean_special_tokens(regular_content) if regular_content else ""

    # Reverse Gemma 4 parameter renaming
    if tool_calls and "gemma" in (resolved_model or request.model or "").lower():
        for tc in tool_calls:
            fn = getattr(tc, "function", None)
            if fn and fn.arguments:
                try:
                    args = json.loads(fn.arguments)
                    args = restore_gemma4_param_names(args)
                    fn.arguments = json.dumps(args, ensure_ascii=False)
                except (json.JSONDecodeError, AttributeError):
                    pass

    final_text = cleaned_text.strip() if cleaned_text else ""

    # Process response_format if specified
    if response_format and not tool_calls:
        _, parsed_json, is_valid, error = parse_json_output(final_text, response_format)
        if parsed_json is not None:
            final_text = json.dumps(parsed_json)
        if not is_valid:
            logger.warning(f"JSON validation failed: {error}")

    if reasoning_opened and not reasoning_closed:
        for ev in _close_reasoning():
            yield ev

    # Ensure message item is opened (even if no content was streamed).
    for ev in _open_message():
        yield ev

    # response.output_text.done
    seq += 1
    yield format_sse_event(
        "response.output_text.done",
        {
            "type": "response.output_text.done",
            "item_id": msg_id,
            "output_index": msg_output_index,
            "content_index": 0,
            "text": final_text,
            "sequence_number": seq,
        },
    )

    # response.content_part.done
    seq += 1
    yield format_sse_event(
        "response.content_part.done",
        {
            "type": "response.content_part.done",
            "item_id": msg_id,
            "output_index": msg_output_index,
            "content_index": 0,
            "part": {"type": "output_text", "text": final_text, "annotations": []},
            "sequence_number": seq,
        },
    )

    # response.output_item.done (message)
    seq += 1
    yield format_sse_event(
        "response.output_item.done",
        {
            "type": "response.output_item.done",
            "output_index": msg_output_index,
            "item": {
                "type": "message",
                "id": msg_id,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": final_text, "annotations": []}
                ],
            },
            "sequence_number": seq,
        },
    )

    # Build output items for final response
    output_items = []
    reasoning_text = accumulated_reasoning
    if reasoning_text:
        output_items.append(
            {
                "type": "reasoning",
                "id": reasoning_id,
                "status": "completed",
                "summary": [{"type": "summary_text", "text": reasoning_text}],
            }
        )
    output_items.append(
        {
            "type": "message",
            "id": msg_id,
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": final_text, "annotations": []}],
        }
    )

    # Emit function call items if present
    if tool_calls:
        output_index = next_output_index
        for tc in tool_calls:
            if hasattr(tc, "function"):
                call_id = tc.id
                name = tc.function.name
                arguments = tc.function.arguments
            elif isinstance(tc, dict):
                call_id = tc.get(
                    "call_id", tc.get("id", f"call_{uuid.uuid4().hex[:8]}")
                )
                name = tc.get("name", "")
                arguments = tc.get("arguments", "{}")
            else:
                continue

            fc_id = generate_id(IDPrefix.FUNCTION_CALL)
            fc_item = {
                "type": "function_call",
                "id": fc_id,
                "call_id": call_id,
                "name": name,
                "arguments": "",
                "status": "in_progress",
            }

            # output_item.added
            seq += 1
            yield format_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": fc_item,
                    "sequence_number": seq,
                },
            )

            # function_call_arguments.delta
            seq += 1
            yield format_sse_event(
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": fc_id,
                    "output_index": output_index,
                    "delta": arguments,
                    "sequence_number": seq,
                },
            )

            # function_call_arguments.done
            seq += 1
            yield format_sse_event(
                "response.function_call_arguments.done",
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": fc_id,
                    "output_index": output_index,
                    "arguments": arguments,
                    "sequence_number": seq,
                },
            )

            # output_item.done
            completed_fc = {
                "type": "function_call",
                "id": fc_id,
                "call_id": call_id,
                "name": name,
                "arguments": arguments,
                "status": "completed",
            }
            seq += 1
            yield format_sse_event(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": output_index,
                    "item": completed_fc,
                    "sequence_number": seq,
                },
            )

            output_items.append(completed_fc)
            output_index += 1
            next_output_index = output_index

    # Record metrics
    usage_data = None
    if last_output and last_output.finished:
        end_time = time.perf_counter()
        total_duration = end_time - start_time
        ttft = (first_token_time - start_time) if first_token_time else total_duration
        if getattr(engine, "is_diffusion_model", False):
            gen_duration = total_duration
        else:
            gen_duration = end_time - (first_token_time or start_time)
        get_server_metrics().record_request_complete(
            prompt_tokens=last_output.prompt_tokens,
            completion_tokens=last_output.completion_tokens,
            cached_tokens=last_output.cached_tokens,
            prefill_duration=ttft,
            generation_duration=gen_duration,
            model_id=resolved_model or request.model,
        )
        reasoning_token_count = (
            len(engine.tokenizer.encode(reasoning_text)) if reasoning_text else 0
        )
        usage_data = {
            "input_tokens": last_output.prompt_tokens,
            "output_tokens": last_output.completion_tokens,
            "total_tokens": last_output.prompt_tokens + last_output.completion_tokens,
            "input_tokens_details": {"cached_tokens": last_output.cached_tokens},
            "output_tokens_details": {"reasoning_tokens": reasoning_token_count},
        }

    # 13. response.completed — MUST always be sent
    final_response = {
        "id": response_id,
        "object": "response",
        "created_at": initial_response.created_at,
        "model": request.model,
        "status": "completed",
        "output": output_items,
        "usage": usage_data,
        "tool_choice": request.tool_choice or "auto",
        "tools": (
            [t.model_dump(exclude_none=True) for t in request.tools]
            if request.tools
            else []
        ),
        "temperature": request.temperature,
        "top_p": request.top_p,
        "max_output_tokens": request.max_output_tokens,
    }
    if request.previous_response_id:
        final_response["previous_response_id"] = request.previous_response_id

    seq += 1
    yield format_sse_event(
        "response.completed",
        {
            "type": "response.completed",
            "response": final_response,
            "sequence_number": seq,
        },
    )

    # Store for future previous_response_id usage
    if store_response:
        _store_response_state(final_response, input_messages=input_messages or [])


@app.get("/v1/responses/{response_id}")
async def get_response(
    response_id: str,
    _: bool = Depends(verify_api_key),
):
    """Retrieve a stored response."""
    data = _server_state.responses_store.get(response_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Response not found")
    return data


@app.delete("/v1/responses/{response_id}")
async def delete_response(
    response_id: str,
    _: bool = Depends(verify_api_key),
):
    """Delete a stored response."""
    if not _server_state.responses_store.delete(response_id):
        raise HTTPException(status_code=404, detail="Response not found")
    return {"id": response_id, "object": "response.deleted", "deleted": True}


# =============================================================================
# MCP Initialization
# =============================================================================


async def init_mcp(config_path: str):
    """Initialize MCP manager from config file."""
    try:
        from omlx.mcp import MCPClientManager, ToolExecutor, load_mcp_config

        config = load_mcp_config(config_path)
        _server_state.mcp_manager = MCPClientManager(config)
        await _server_state.mcp_manager.start()

        _server_state.mcp_executor = ToolExecutor(_server_state.mcp_manager)

        logger.info(
            f"MCP initialized with {len(_server_state.mcp_manager.get_all_tools())} tools"
        )

    except ImportError:
        logger.warning(
            "MCP SDK not installed. MCP features disabled. "
            "Install with: pip install mcp"
        )
        return
    except Exception as e:
        logger.error(
            f"Failed to initialize MCP: {e}. "
            "MCP features disabled. Fix your MCP config and restart."
        )
        return


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Run the server (use omlx CLI instead)."""
    parser = argparse.ArgumentParser(
        description="oMLX multi-model serving for Apple Silicon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Multi-model serving
    python -m omlx.server --model-dir /path/to/models

    # With pinned models
    python -m omlx.server --model-dir /path/to/models --pin llama-3b,qwen-7b

    # With MCP tools
    python -m omlx.server --model-dir /path/to/models --mcp-config mcp.json

Note: Use the omlx CLI for full feature support.
        """,
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing model subdirectories",
    )
    parser.add_argument(
        "--pin",
        type=str,
        default=None,
        help="Comma-separated model names to keep always loaded",
    )
    parser.add_argument(
        "--default-model",
        type=str,
        default=None,
        help="Default model when not specified in request",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to",
    )
    parser.add_argument(
        "--mcp-config",
        type=str,
        default=None,
        help="Path to MCP configuration file (JSON/YAML)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="Default max tokens for generation",
    )

    args = parser.parse_args()

    # Set MCP config for lifespan
    if args.mcp_config:
        os.environ["OMLX_MCP_CONFIG"] = args.mcp_config

    # Parse pinned models
    pinned_models = args.pin.split(",") if args.pin else []
    # Initialize server
    init_server(
        model_dir=args.model_dir,
        pinned_models=pinned_models,
        default_model=args.default_model,
        max_tokens=args.max_tokens,
    )

    # Start server
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
