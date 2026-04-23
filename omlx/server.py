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
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union

from fastapi import Depends, FastAPI, HTTPException, Request as FastAPIRequest
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
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
from .api.embedding_models import (
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingData,
    EmbeddingUsage,
)
from .api.embedding_utils import (
    encode_embedding_base64,
    normalize_embedding_items,
    truncate_embedding,
    normalize_input,
)
from .api.rerank_models import (
    RerankRequest,
    RerankResponse,
    RerankResult,
    RerankUsage,
)
from .api.responses_models import (
    OutputContent,
    OutputItem,
    ResponseObject,
    ResponsesRequest,
    ResponsesTool,
    ResponseUsage,
    TextConfig,
)
from .api.responses_utils import (
    ResponseStore,
    ResponseStateCorruptError,
    ResponseStateNotFoundError,
    build_function_call_output_item,
    build_message_output_item,
    build_response_store_record,
    build_response_usage,
    convert_responses_input_to_messages,
    convert_responses_tools,
    format_sse_event,
    normalize_response_output_to_messages,
)
from .api.tool_calling import (
    ToolCallStreamFilter,
    build_json_system_prompt,
    convert_tools_for_template,
    enrich_tool_params_for_gemma4,
    restore_gemma4_param_names,
    extract_tool_calls_with_thinking,
    parse_json_output,
    parse_tool_calls,
    parse_tool_calls_with_thinking_fallback,
    sanitize_tool_call_markup,
)
from .api.thinking import ThinkingParser, extract_thinking
from .api.utils import clean_output_text, clean_special_tokens, extract_multimodal_content, extract_text_content
from .engine import BaseEngine, BatchedEngine, VLMBatchedEngine
from .engine.embedding import EmbeddingEngine
from .engine.reranker import RerankerEngine
from .engine_pool import EnginePool
from .exceptions import (
    EnginePoolError,
    InsufficientMemoryError,
    ModelLoadingError,
    ModelNotFoundError,
    ModelTooLargeError,
)
from .model_discovery import format_size
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

    max_context_window: int = 32768
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
    from .admin.auth import verify_any_api_key

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
        logger.warning("Rejected API key: %r", api_key_value)
        raise HTTPException(status_code=401, detail="Invalid API key")

    return True


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

    # Startup: Preload pinned models
    if _server_state.engine_pool is not None:
        await _server_state.engine_pool.preload_pinned_models()

    # Start process memory enforcer if configured
    if (
        _server_state.global_settings is not None
        and _server_state.engine_pool is not None
    ):
        max_bytes = _server_state.global_settings.memory.get_max_process_memory_bytes()
        if max_bytes is not None:
            from .process_memory_enforcer import ProcessMemoryEnforcer

            enforcer = ProcessMemoryEnforcer(
                engine_pool=_server_state.engine_pool,
                max_bytes=max_bytes,
                settings_manager=_server_state.settings_manager,
                prefill_memory_guard=_server_state.global_settings.memory.prefill_memory_guard,
                global_settings=_server_state.global_settings,
            )
            _server_state.process_memory_enforcer = enforcer
            _server_state.engine_pool._process_memory_enforcer = enforcer
            enforcer.start()

    # Start TTL-only checker if process memory enforcer is not running
    # (enforcer already includes TTL checks in its polling loop)
    ttl_task = None
    if _server_state.process_memory_enforcer is None and _server_state.engine_pool is not None:
        async def _ttl_check_loop():
            while True:
                try:
                    if _server_state.settings_manager is not None:
                        await _server_state.engine_pool.check_ttl_expirations(
                            _server_state.settings_manager,
                            global_idle_timeout_seconds=_server_state.global_settings.idle_timeout.idle_timeout_seconds
                            if _server_state.global_settings else None,
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
        logger.info("Engine pool shutdown")


app = FastAPI(
    title="oMLX API",
    description="LLM inference, optimized for your Mac",
    version=__version__,
    lifespan=lifespan,
)

# Include MCP routes
from .api.mcp_routes import router as mcp_router, set_mcp_manager_getter
set_mcp_manager_getter(get_mcp_manager)
app.include_router(mcp_router)

# Include audio routes only when mlx-audio is installed.
# audio_routes.py itself only imports fastapi/stdlib at module level, so it
# would always import successfully — we need an explicit mlx-audio check.
try:
    import mlx_audio as _  # noqa: F401
    from .api.audio_routes import router as audio_router
    app.include_router(audio_router)
    del _
except ImportError:
    pass

# Include admin routes
from .admin.routes import router as admin_router, set_admin_getters
from .admin.auth import _RedirectToLogin
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
    if status_code == 429:
        return "rate_limit_error"
    if status_code >= 500:
        return "server_error"
    return "invalid_request_error"


def _is_api_route(request: FastAPIRequest) -> bool:
    """Check if request targets an OpenAI-compatible API route."""
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


async def get_engine(
    model_id: str | None = None,
    engine_type: EngineType = EngineType.LLM,
) -> Union[BaseEngine, EmbeddingEngine, RerankerEngine]:
    """
    Get engine for the specified model and type.

    This is the unified engine getter that handles LLM, embedding, and reranker models.

    Args:
        model_id: Model ID to get engine for, or None for default (LLM only)
        engine_type: Type of engine to retrieve (LLM, EMBEDDING, or RERANKER)

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
                detail=f"Model ID is required for {engine_type.value} engines"
            )
        model_id = _server_state.default_model

    if model_id is None:
        raise HTTPException(
            status_code=400,
            detail="No model specified and no default model set"
        )

    # Resolve alias to real model_id
    model_id = pool.resolve_model_id(model_id, _server_state.settings_manager)

    try:
        engine = await pool.get_engine(model_id)
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
                return await pool.get_engine(_server_state.default_model)
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
    except ModelLoadingError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except EnginePoolError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Validate engine type
    if engine_type == EngineType.EMBEDDING:
        if not isinstance(engine, EmbeddingEngine):
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_id}' is not an embedding model. "
                f"Use /v1/chat/completions for LLM models."
            )
    elif engine_type == EngineType.RERANKER:
        if not isinstance(engine, RerankerEngine):
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_id}' is not a reranker model. "
                f"Use a SequenceClassification model for reranking."
            )

    return engine


async def get_engine_for_model(model: str | None = None) -> BaseEngine:
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
    return await get_engine(model, EngineType.LLM)


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


def get_sampling_params(
    req_temperature: float | None,
    req_top_p: float | None,
    model_id: str | None = None,
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
    - If force_sampling is True (global or model level): use forced values
    - Otherwise: request > model settings > ocr_defaults > global defaults

    Returns:
        tuple of (temperature, top_p, top_k, repetition_penalty, min_p, presence_penalty, frequency_penalty, max_tokens, xtc_probability, xtc_threshold)
    """
    global_sampling = _server_state.sampling

    # Resolve alias so per-model settings are found by real model ID
    model_id = resolve_model_id(model_id)

    # Get per-model settings if available
    model_settings = None
    if model_id and _server_state.settings_manager:
        model_settings = _server_state.settings_manager.get_settings(model_id)

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

        if model_settings and model_settings.top_k is not None:
            top_k = model_settings.top_k
        else:
            top_k = global_sampling.top_k

    # Repetition penalty: model settings > ocr_defaults > global default (1.0)
    if model_settings and model_settings.repetition_penalty is not None:
        repetition_penalty = model_settings.repetition_penalty
    elif ocr_defaults and "repetition_penalty" in ocr_defaults:
        repetition_penalty = ocr_defaults["repetition_penalty"]
    else:
        repetition_penalty = getattr(global_sampling, 'repetition_penalty', 1.0)

    # Min P: request > model settings > default (0.0)
    if req_min_p is not None:
        min_p = req_min_p
    elif model_settings and getattr(model_settings, 'min_p', None) is not None:
        min_p = model_settings.min_p
    else:
        min_p = 0.0

    # Presence penalty: request > model settings > default (0.0)
    if req_presence_penalty is not None:
        presence_penalty = req_presence_penalty
    elif model_settings and getattr(model_settings, 'presence_penalty', None) is not None:
        presence_penalty = model_settings.presence_penalty
    else:
        presence_penalty = 0.0

    # Frequency penalty: request > model settings > default (0.0)
    if req_frequency_penalty is not None:
        frequency_penalty = req_frequency_penalty
    elif model_settings and getattr(model_settings, 'frequency_penalty', None) is not None:
        frequency_penalty = model_settings.frequency_penalty
    else:
        frequency_penalty = 0.0

    # Max tokens: same hierarchy as other params
    if force:
        if model_settings and model_settings.max_tokens is not None:
            max_tokens = model_settings.max_tokens
        elif ocr_defaults and "max_tokens" in ocr_defaults:
            max_tokens = ocr_defaults["max_tokens"]
        else:
            max_tokens = global_sampling.max_tokens
    else:
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
    return temperature, top_p, top_k, repetition_penalty, min_p, presence_penalty, frequency_penalty, max_tokens, xtc_probability, xtc_threshold


def _resolve_thinking_budget(request, model_id: str | None) -> int | None:
    """Resolve thinking budget: request param > model settings > None."""
    # Check request-level override (OpenAI format)
    req_budget = getattr(request, 'thinking_budget', None)
    # For Anthropic: check thinking.budget_tokens
    if req_budget is None and hasattr(request, 'thinking') and request.thinking:
        req_budget = getattr(request.thinking, 'budget_tokens', None)
    if req_budget is not None:
        return req_budget
    # Check model settings
    resolved = resolve_model_id(model_id)
    if resolved and _server_state.settings_manager:
        ms = _server_state.settings_manager.get_settings(resolved)
        if ms.thinking_budget_enabled and ms.thinking_budget_tokens:
            return ms.thinking_budget_tokens
    return None


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

    Priority: model setting > global setting.

    Returns:
        Max context window token count, or None if not set.
    """
    # Resolve alias so per-model settings are found by real model ID
    model_id = resolve_model_id(model_id)

    model_settings = None
    if model_id and _server_state.settings_manager:
        model_settings = _server_state.settings_manager.get_settings(model_id)

    if model_settings and model_settings.max_context_window is not None:
        return model_settings.max_context_window

    return _server_state.sampling.max_context_window


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
    max_model_memory: int | None,
    scheduler_config=None,
    api_key: str | None = None,
    global_settings: object | None = None,
):
    """
    Initialize server with model directories for multi-model serving.

    Args:
        model_dirs: Path or list of paths to directories containing model subdirectories
        max_model_memory: Maximum memory for loaded models in bytes, or None for no limit
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

        init_auth(global_settings.auth.secret_key, lambda: _server_state.global_settings)

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
    base_path = Path(global_settings.base_path) if global_settings else Path(model_dir)
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
            max_tokens=global_settings.sampling.max_tokens,
            temperature=global_settings.sampling.temperature,
            top_p=global_settings.sampling.top_p,
            top_k=global_settings.sampling.top_k,
            repetition_penalty=getattr(global_settings.sampling, 'repetition_penalty', 1.0),
        )
    else:
        _server_state.sampling = SamplingDefaults()

    # Normalize model_dirs to list
    if isinstance(model_dirs, str):
        dir_list = [model_dirs]
    else:
        dir_list = list(model_dirs)

    # Create directories if needed
    for md in dir_list:
        model_path = Path(md)
        if not model_path.exists():
            model_path.mkdir(parents=True, exist_ok=True)
            logger.warning(f"Model directory created (empty): {md}")

    # Create engine pool
    _server_state.engine_pool = EnginePool(
        max_model_memory=max_model_memory,
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

    logger.info(f"Server initialized with {_server_state.engine_pool.model_count} models")
    if _server_state.default_model:
        logger.info(f"Default model: {_server_state.default_model}")
    else:
        logger.info("No default model (no models available)")
    if max_model_memory is None:
        logger.info("Max model memory: disabled (no limit)")
    else:
        logger.info(f"Max model memory: {format_size(max_model_memory)}")
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
        from .admin.ms_downloader import MSDownloader, MS_SDK_AVAILABLE

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
) -> AsyncIterator[str]:
    """Wrap an SSE generator to send periodic keep-alive comments.

    During long prefill (e.g. 90k tokens), no SSE events are emitted,
    causing clients with read timeouts (like Claude Code) to disconnect.
    This wrapper sends SSE comments (: keep-alive) that are ignored by
    SSE parsers but keep the HTTP connection alive.

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
    yield ": keep-alive\n\n"

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
                            logger.info("Client disconnected during streaming (is_disconnected), cancelling")
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
                    yield ": keep-alive\n\n"
            if task.done():
                try:
                    result = task.result()
                except Exception as e:
                    logger.error(f"SSE generator error: {e}")
                    error_data = {"error": {"message": str(e), "type": "server_error"}}
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
        if hasattr(ait, 'aclose'):
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
                        logger.info("Client disconnected during non-streaming response, cancelling")
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
        result = task.result()
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
async def health():
    """Health check endpoint."""
    mcp_info = None
    if _server_state.mcp_manager is not None:
        connected = sum(1 for s in _server_state.mcp_manager.get_server_status() if s.state.value == "connected")
        total = len(_server_state.mcp_manager.get_server_status())
        mcp_info = {
            "enabled": True,
            "servers_connected": connected,
            "servers_total": total,
            "tools_available": len(_server_state.mcp_manager.get_all_tools()),
        }

    pool_status = None
    if _server_state.engine_pool is not None:
        pool_status = {
            "model_count": _server_state.engine_pool.model_count,
            "loaded_count": _server_state.engine_pool.loaded_model_count,
            "max_model_memory": _server_state.engine_pool.max_model_memory,
            "current_model_memory": _server_state.engine_pool.current_model_memory,
        }

    return {
        "status": "healthy",
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
        model_memory_max = pool.max_model_memory
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
        "model_memory_used_formatted": format_size(model_memory_used) if model_memory_used else "0B",
        "model_memory_max_formatted": format_size(model_memory_max) if model_memory_max else "unlimited",
    }


@app.get("/v1/models")
async def list_models(_: bool = Depends(verify_api_key)) -> ModelsResponse:
    """List all available models with load status."""
    models = []

    if _server_state.engine_pool is not None:
        status = _server_state.engine_pool.get_status()
        settings_manager = _server_state.settings_manager
        for m in status["models"]:
            model_id = m["id"]
            display_id = model_id
            if settings_manager:
                ms = settings_manager.get_settings(model_id)
                if ms.model_alias:
                    display_id = ms.model_alias
            models.append(
                ModelInfo(
                    id=display_id,
                    owned_by="omlx",
                )
            )

    return ModelsResponse(data=models)


@app.get("/v1/models/status")
async def list_models_status(_: bool = Depends(verify_api_key)):
    """
    List all available models with detailed status.

    Extended endpoint that provides more information than /v1/models.
    """
    if _server_state.engine_pool is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    status = _server_state.engine_pool.get_status()
    for m in status["models"]:
        model_id = m["id"]
        m["max_context_window"] = get_max_context_window(model_id)

        # Resolve effective max_tokens: model setting > global default
        max_tokens = _server_state.sampling.max_tokens
        if _server_state.settings_manager:
            ms = _server_state.settings_manager.get_settings(model_id)
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

    engine = await get_embedding_engine(request.model)

    if request.items is not None:
        embedding_inputs = normalize_embedding_items(request.items)
    elif request.input is not None:
        embedding_inputs = normalize_input(request.input)
    else:
        embedding_inputs = []

    if not embedding_inputs:
        raise HTTPException(status_code=400, detail="Input cannot be empty")

    async def _build_embeddings():
        start_time = time.perf_counter()
        try:
            output = await engine.embed(embedding_inputs)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except TypeError as e:
            raise HTTPException(status_code=400, detail=str(e))

        elapsed = time.perf_counter() - start_time
        logger.info(
            f"Embedding: {len(embedding_inputs)} inputs, {output.dimensions} dims, "
            f"{output.total_tokens} tokens in {elapsed:.3f}s"
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

    engine = await get_reranker_engine(request.model)

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
    load_start = time.perf_counter()
    engine = await get_engine_for_model(request.model)
    model_load_duration = time.perf_counter() - load_start

    # Handle single prompt or list of prompts
    prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]

    # Validate context window for each prompt
    for prompt in prompts:
        num_tokens = len(engine.tokenizer.encode(prompt))
        validate_context_window(num_tokens, request.model)

    if request.stream:
        return StreamingResponse(
            _with_sse_keepalive(
                stream_completion(engine, prompts[0], request, model_load_duration=model_load_duration),
                http_request=http_request,
            ),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )

    # Non-streaming response with keepalive during prefill
    async def _build_completion():
        start_time = time.perf_counter()
        choices = []
        total_completion_tokens = 0
        total_prompt_tokens = 0
        total_cached_tokens = 0

        temperature, top_p, top_k, repetition_penalty, min_p, presence_penalty, frequency_penalty, max_tokens, xtc_probability, xtc_threshold = get_sampling_params(
            request.temperature, request.top_p, request.model,
            req_min_p=getattr(request, 'min_p', None),
            req_presence_penalty=getattr(request, 'presence_penalty', None),
            req_frequency_penalty=getattr(request, 'frequency_penalty', None),
            req_max_tokens=request.max_tokens,
            req_xtc_probability=getattr(request, 'xtc_probability', None),
            req_xtc_threshold=getattr(request, 'xtc_threshold', None),
        )

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
            )

            choices.append(CompletionChoice(
                index=i,
                text=output.text,
                finish_reason=output.finish_reason,
            ))
            total_completion_tokens += output.completion_tokens
            total_prompt_tokens += output.prompt_tokens
            total_cached_tokens += output.cached_tokens

        elapsed = time.perf_counter() - start_time
        tokens_per_sec = total_completion_tokens / elapsed if elapsed > 0 else 0
        logger.info(f"Completion: {total_completion_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)")

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
            ),
        ).model_dump_json()

    return StreamingResponse(
        _with_json_keepalive(http_request, _build_completion()),
        media_type="application/json",
    )


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
    logger.debug(f"Chat completion request received: model={request.model}, "
                 f"messages={len(request.messages)}, stream={request.stream}, "
                 f"max_tokens={request.max_tokens}, temp={request.temperature}")
    if logger.isEnabledFor(5):
        for i, msg in enumerate(request.messages):
            content_preview = str(msg.content)[:200] if msg.content else "(empty)"
            logger.log(5, "  Message[%d]: role=%s, content=%s...", i, msg.role, content_preview)

    # Block inference during quantization to prevent GPU Metal errors
    if _server_state.oq_manager and _server_state.oq_manager.is_quantizing:
        raise HTTPException(
            status_code=503,
            detail="Server is busy with oQ quantization. Please try again after quantization completes.",
        )

    load_start = time.perf_counter()
    engine = await get_engine_for_model(request.model)
    model_load_duration = time.perf_counter() - load_start

    # Resolve alias to real model ID for settings lookups
    resolved_model = resolve_model_id(request.model) or request.model

    # Get per-model settings
    max_tool_result_tokens = None
    merged_ct_kwargs = {}
    forced_keys: set[str] = set()
    reasoning_parser = None
    if _server_state.settings_manager:
        ms = _server_state.settings_manager.get_settings(resolved_model)
        max_tool_result_tokens = ms.max_tool_result_tokens
        reasoning_parser = ms.reasoning_parser
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
    native_reasoning = bool(_entry and _entry.preserve_thinking_default is True)
    is_vlm = isinstance(engine, VLMBatchedEngine)
    extractor = getattr(engine, "message_extractor", None)
    if extractor is not None:
        messages = extractor(request.messages, max_tool_result_tokens, engine.tokenizer)
    elif is_vlm:
        # VLM: preserve image_url content parts for vision processing
        messages = extract_multimodal_content(
            request.messages,
            max_tool_result_tokens,
            engine.tokenizer,
            native_reasoning_content=native_reasoning,
        )
    else:
        messages = extract_text_content(
            request.messages,
            max_tool_result_tokens,
            engine.tokenizer,
            native_reasoning_content=native_reasoning,
        )

    # Compile grammar for structured output (logit-level enforcement).
    # Grammar compilation needs the tokenizer, so ensure the engine is loaded.
    response_format = request.response_format
    if request.structured_outputs is not None or response_format:
        await engine.start()
    compiled_grammar = _compile_grammar_for_request(
        engine,
        structured_outputs=request.structured_outputs,
        response_format=response_format,
        chat_template_kwargs=merged_ct_kwargs or None,
        reasoning_parser=reasoning_parser,
    )
    # Fall back to prompt injection when grammar is not compiled
    if compiled_grammar is None and response_format:
        json_instruction = build_json_system_prompt(response_format)
        if json_instruction:
            messages = _inject_json_instruction(messages, json_instruction)

    # Merge MCP tools with user-provided tools
    effective_tools = request.tools
    if _server_state.mcp_manager:
        # Convert Pydantic ToolDefinition models to dicts for merge_tools
        user_tools_dicts = [t.model_dump() for t in request.tools] if request.tools else None
        effective_tools = _server_state.mcp_manager.get_merged_tools(user_tools_dicts)

    # Validate context window before sending to model
    tools_for_template = convert_tools_for_template(effective_tools) if effective_tools else None
    # Gemma 4 drops required params that lack descriptions — enrich them
    if tools_for_template and "gemma" in (resolved_model or "").lower():
        tools_for_template = enrich_tool_params_for_gemma4(tools_for_template)
    try:
        num_prompt_tokens = engine.count_chat_tokens(
            messages, tools_for_template,
            chat_template_kwargs=merged_ct_kwargs or None,
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
            raise HTTPException(
                status_code=400, detail=f"Chat template error: {e}"
            )
        raise
    validate_context_window(num_prompt_tokens, request.model)

    # Prepare kwargs
    temperature, top_p, top_k, repetition_penalty, min_p, presence_penalty, frequency_penalty, max_tokens, xtc_probability, xtc_threshold = get_sampling_params(
        request.temperature, request.top_p, request.model,
        req_min_p=getattr(request, 'min_p', None),
        req_presence_penalty=getattr(request, 'presence_penalty', None),
        req_frequency_penalty=getattr(request, 'frequency_penalty', None),
        req_max_tokens=request.max_tokens,
        req_xtc_probability=getattr(request, 'xtc_probability', None),
        req_xtc_threshold=getattr(request, 'xtc_threshold', None),
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

    if request.stream:
        return StreamingResponse(
            _with_sse_keepalive(
                stream_chat_completion(engine, messages, request, model_load_duration=model_load_duration, resolved_model=resolved_model, **chat_kwargs),
                http_request=http_request,
            ),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )

    # Non-streaming response with keepalive during prefill
    async def _build_chat_completion():
        start_time = time.perf_counter()

        output = await engine.chat(messages=messages, **chat_kwargs)

        elapsed = time.perf_counter() - start_time
        tokens_per_sec = output.completion_tokens / elapsed if elapsed > 0 else 0
        logger.info(f"Chat completion: {output.completion_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)")

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
        cleaned_thinking = sanitize_tool_call_markup(thinking_content, engine.tokenizer)

        # For Harmony (gpt-oss) models, tool_calls are already extracted by the parser
        # For other models, parse from text output
        if engine.model_type == "gpt_oss" and output.tool_calls:
            from .api.openai_models import ToolCall, FunctionCall
            tool_calls = [
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    type="function",
                    function=FunctionCall(
                        name=tc["name"],
                        arguments=tc["arguments"],
                    ),
                )
                for tc in output.tool_calls
            ]
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
                cleaned_text or regular_content,
                response_format
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
            choices=[ChatCompletionChoice(
                message=AssistantMessage(
                    content=cleaned_text.strip() if cleaned_text else None,
                    reasoning_content=cleaned_thinking if cleaned_thinking else None,
                    tool_calls=tool_calls,
                ),
                finish_reason=finish_reason,
            )],
            usage=Usage(
                prompt_tokens=output.prompt_tokens,
                completion_tokens=output.completion_tokens,
                total_tokens=output.prompt_tokens + output.completion_tokens,
                prompt_tokens_details=PromptTokensDetails(
                    cached_tokens=output.cached_tokens,
                ),
            ),
        ).model_dump_json()

    return StreamingResponse(
        _with_json_keepalive(http_request, _build_chat_completion()),
        media_type="application/json",
    )


def _inject_json_instruction(messages: list, instruction: str) -> list:
    """
    Inject JSON instruction into messages.

    If a system message exists, append to it. Otherwise, prepend a new system message.
    """
    messages = list(messages)  # Make a copy

    # Find existing system message
    system_idx = None
    for i, msg in enumerate(messages):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role == "system":
            system_idx = i
            break

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
        rf_type = (
            rf.get("type") if isinstance(rf, dict)
            else getattr(rf, "type", None)
        )
        if rf_type == "json_schema":
            js = (
                rf.get("json_schema") if isinstance(rf, dict)
                else getattr(rf, "json_schema", None)
            )
            if js is not None:
                schema = (
                    js.get("schema") if isinstance(js, dict)
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


def _compile_with_structural_tag(compiler, fmt: dict, reasoning_parser: str,
                                  chat_template_kwargs: dict | None):
    """Compile a grammar wrapped in an xgrammar builtin structural tag.

    Uses ``xgrammar.get_builtin_structural_tag`` to obtain the model's
    protocol structure (thinking tags, channel markers, etc.) and patches
    the user's grammar into the output slot.
    """
    import xgrammar as xgr

    reasoning = not (
        chat_template_kwargs
        and chat_template_kwargs.get("enable_thinking") is False
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

    Returns a compiled grammar object or ``None``.  Raises
    :class:`HTTPException` on compilation errors or when xgrammar is
    required but not installed.
    """
    compiler = getattr(engine, 'grammar_compiler', None)

    fmt = _build_format_element(structured_outputs, response_format)
    if fmt is None:
        return None

    if compiler is None:
        if structured_outputs is not None:
            from omlx.utils.install import get_install_method

            method = get_install_method()
            if method == "dmg":
                detail = (
                    "Structured output is not available in the DMG version. "
                    "xgrammar requires torch which significantly increases app size. "
                    "Use the pip or Homebrew version for structured output support."
                )
            elif method == "homebrew":
                detail = (
                    "Structured output requires xgrammar. "
                    "Reinstall with: brew reinstall omlx --with-grammar"
                )
            else:
                detail = (
                    "Structured output requires xgrammar. "
                    "Install with: pip install 'omlx[grammar]'"
                )
            raise HTTPException(status_code=400, detail=detail)
        return None

    try:
        if reasoning_parser:
            return _compile_with_structural_tag(
                compiler, fmt, reasoning_parser, chat_template_kwargs,
            )
        return _compile_bare_grammar(compiler, fmt)
    except Exception as e:
        if structured_outputs is not None:
            raise HTTPException(
                status_code=400,
                detail=f"Grammar compilation error: {e}",
            )
        logger.warning("Grammar compilation from response_format failed, "
                       "falling back to prompt injection: %s", e)
    return None


# =============================================================================
# Streaming Helpers
# =============================================================================

async def stream_completion(
    engine: BaseEngine,
    prompt: str,
    request: CompletionRequest,
    model_load_duration: float = 0.0,
) -> AsyncIterator[str]:
    """Stream completion response."""
    start_time = time.perf_counter()
    first_token_time = None
    last_output = None

    temperature, top_p, top_k, repetition_penalty, min_p, presence_penalty, frequency_penalty, max_tokens, xtc_probability, xtc_threshold = get_sampling_params(
        request.temperature, request.top_p, request.model,
        req_min_p=getattr(request, 'min_p', None),
        req_presence_penalty=getattr(request, 'presence_penalty', None),
        req_frequency_penalty=getattr(request, 'frequency_penalty', None),
        req_max_tokens=request.max_tokens,
        req_xtc_probability=getattr(request, 'xtc_probability', None),
        req_xtc_threshold=getattr(request, 'xtc_threshold', None),
    )
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
        ):
            if first_token_time is None and output.new_text:
                first_token_time = time.perf_counter()
            last_output = output

            data = {
                "id": f"cmpl-{uuid.uuid4().hex[:8]}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "text": output.new_text,
                    "finish_reason": output.finish_reason if output.finished else None,
                }],
            }
            yield f"data: {json.dumps(data)}\n\n"
    except Exception as e:
        logger.error(f"Error during completion streaming: {e}")
        error_data = {
            "error": {"message": str(e), "type": "server_error"}
        }
        yield f"data: {json.dumps(error_data)}\n\n"
        yield "data: [DONE]\n\n"
        return

    # Record metrics
    if last_output and last_output.finished:
        end_time = time.perf_counter()
        ttft = (first_token_time - start_time) if first_token_time else (end_time - start_time)
        gen_duration = end_time - (first_token_time or start_time)
        get_server_metrics().record_request_complete(
            prompt_tokens=last_output.prompt_tokens,
            completion_tokens=last_output.completion_tokens,
            cached_tokens=last_output.cached_tokens,
            prefill_duration=ttft,
            generation_duration=gen_duration,
            model_id=resolve_model_id(request.model) or request.model,
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
                    model_load_duration=round(model_load_duration, 2) if model_load_duration > 1.0 else None,
                    time_to_first_token=round(ttft, 2),
                    total_time=round(total_time, 2),
                    prompt_eval_duration=round(ttft, 2),
                    generation_duration=round(gen_duration, 2),
                    prompt_tokens_per_second=round(pt / ttft, 2) if ttft > 0 else None,
                    generation_tokens_per_second=round(ct / gen_duration, 2) if gen_duration > 0 else None,
                ).model_dump(exclude_none=True),
            }
            yield f"data: {json.dumps(usage_data)}\n\n"

    yield "data: [DONE]\n\n"


async def stream_chat_completion(
    engine: BaseEngine,
    messages: list,
    request: ChatCompletionRequest,
    model_load_duration: float = 0.0,
    resolved_model: Optional[str] = None,
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
    thinking_parser = ThinkingParser()

    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    # First chunk with role
    first_chunk = ChatCompletionChunk(
        id=response_id,
        model=request.model,
        choices=[ChatCompletionChunkChoice(
            delta=ChatCompletionChunkDelta(role="assistant"),
        )],
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
                        choices=[ChatCompletionChunkChoice(
                            delta=ChatCompletionChunkDelta(reasoning_content=thinking_delta),
                            finish_reason=None,
                        )],
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
                            choices=[ChatCompletionChunkChoice(
                                delta=ChatCompletionChunkDelta(content=content_delta),
                                finish_reason=None,
                            )],
                        )
                        yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
    except Exception as e:
        logger.error(f"Error during chat streaming: {e}")
        error_data = {
            "error": {"message": str(e), "type": "server_error"}
        }
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
                    choices=[ChatCompletionChunkChoice(
                        delta=ChatCompletionChunkDelta(reasoning_content=thinking_delta),
                        finish_reason=None,
                    )],
                )
                yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
        if thinking_filter:
            remaining_thinking = thinking_filter.finish()
            if remaining_thinking:
                chunk = ChatCompletionChunk(
                    id=response_id,
                    model=request.model,
                    choices=[ChatCompletionChunkChoice(
                        delta=ChatCompletionChunkDelta(reasoning_content=remaining_thinking),
                        finish_reason=None,
                    )],
                )
                yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
        if content_delta:
            if tool_filter:
                content_delta = tool_filter.feed(content_delta)
            if content_delta:
                chunk = ChatCompletionChunk(
                    id=response_id,
                    model=request.model,
                    choices=[ChatCompletionChunkChoice(
                        delta=ChatCompletionChunkDelta(content=content_delta),
                        finish_reason=None,
                    )],
                )
                yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

        if tool_filter:
            remaining = tool_filter.finish()
            if remaining:
                chunk = ChatCompletionChunk(
                    id=response_id,
                    model=request.model,
                    choices=[ChatCompletionChunkChoice(
                        delta=ChatCompletionChunkDelta(content=remaining),
                        finish_reason=None,
                    )],
                )
                yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

    # Parse tool calls from accumulated text
    tool_calls = None
    cleaned_text = accumulated_text
    if last_output and last_output.tool_calls:
        # Harmony model — tool_calls already extracted by parser
        from .api.openai_models import ToolCall, FunctionCall
        tool_calls = [
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                type="function",
                function=FunctionCall(
                    name=tc["name"],
                    arguments=tc["arguments"],
                ),
            )
            for tc in last_output.tool_calls
        ]
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
                    choices=[ChatCompletionChunkChoice(
                        delta=ChatCompletionChunkDelta(reasoning_content=cleaned_thinking),
                        finish_reason=None,
                    )],
                )
                yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
            if cleaned_text:
                chunk = ChatCompletionChunk(
                    id=response_id,
                    model=request.model,
                    choices=[ChatCompletionChunkChoice(
                        delta=ChatCompletionChunkDelta(content=cleaned_text),
                        finish_reason=None,
                    )],
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
                choices=[ChatCompletionChunkChoice(
                    delta=ChatCompletionChunkDelta(
                        tool_calls=[{
                            "index": i,
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }],
                    ),
                )],
            )
            yield f"data: {tc_chunk.model_dump_json(exclude_none=True)}\n\n"

    # Final chunk with finish_reason
    finish_reason = "tool_calls" if tool_calls else (
        last_output.finish_reason if last_output else "stop"
    )
    final_chunk = ChatCompletionChunk(
        id=response_id,
        model=request.model,
        choices=[ChatCompletionChunkChoice(
            delta=ChatCompletionChunkDelta(),
            finish_reason=finish_reason,
        )],
    )
    yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n"

    # Record metrics and emit usage chunk
    if last_output and last_output.finished:
        end_time = time.perf_counter()
        ttft = (first_token_time - start_time) if first_token_time else (end_time - start_time)
        gen_duration = end_time - (first_token_time or start_time)
        get_server_metrics().record_request_complete(
            prompt_tokens=last_output.prompt_tokens,
            completion_tokens=last_output.completion_tokens,
            cached_tokens=last_output.cached_tokens,
            prefill_duration=ttft,
            generation_duration=gen_duration,
            model_id=resolved_model or request.model,
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
                    model_load_duration=round(model_load_duration, 2) if model_load_duration > 1.0 else None,
                    time_to_first_token=round(ttft, 2),
                    total_time=round(total_time, 2),
                    prompt_eval_duration=round(ttft, 2),
                    generation_duration=round(gen_duration, 2),
                    prompt_tokens_per_second=round(pt / ttft, 2) if ttft > 0 else None,
                    generation_tokens_per_second=round(ct / gen_duration, 2) if gen_duration > 0 else None,
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

    # Track content blocks with thinking separation
    thinking_parser = ThinkingParser()
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

    # Calculate input tokens before streaming starts
    # This is needed for message_start event
    estimated_input_tokens = 0
    try:
        if hasattr(engine, 'tokenizer') and engine.tokenizer is not None:
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
        input_tokens=scale_anthropic_tokens(estimated_input_tokens, request.model),
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
                    if not thinking_block_started:
                        if thinking_delta:
                            yield create_content_block_start_event(
                                index=block_index, block_type="thinking"
                            )
                            thinking_block_started = True
                    if thinking_delta:
                        yield create_thinking_delta_event(
                            index=block_index, thinking=thinking_delta
                        )

                # Emit regular content as text block — filter tool-call
                # markup when a known start marker is available.
                if content_delta:
                    if tool_filter:
                        content_delta = tool_filter.feed(content_delta)
                    if content_delta:
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
                        yield create_text_delta_event(index=block_index, text=content_delta)

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
            if not thinking_block_started:
                yield create_content_block_start_event(
                    index=block_index, block_type="thinking"
                )
                thinking_block_started = True
            yield create_thinking_delta_event(index=block_index, thinking=thinking_delta)
    if thinking_filter:
        remaining_thinking = thinking_filter.finish()
        if remaining_thinking:
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

    # 4. Close open blocks
    if thinking_block_started and not text_block_started:
        # Only thinking was emitted, close it
        yield create_content_block_stop_event(index=block_index)
        block_index += 1
    if text_block_started:
        yield create_content_block_stop_event(index=block_index)
    elif not thinking_block_started:
        # No content at all - create empty text block
        yield create_content_block_start_event(index=block_index, block_type="text")
        yield create_content_block_stop_event(index=block_index)

    # 5. Handle tool calls
    # For Harmony models, use tool_calls from output (parsed by HarmonyStreamingParser)
    # For other models, parse from accumulated text
    tool_calls = None
    if last_output and last_output.tool_calls:
        # Harmony model - tool_calls already extracted by parser
        from .api.openai_models import ToolCall, FunctionCall
        tool_calls = [
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                type="function",
                function=FunctionCall(
                    name=tc["name"],
                    arguments=tc["arguments"],
                ),
            )
            for tc in last_output.tool_calls
        ]
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
        cleaned_text = extraction.cleaned_text
        tool_calls = extraction.tool_calls

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
            yield create_input_json_delta_event(index=i, partial_json=tc.function.arguments)
            # Close tool block
            yield create_content_block_stop_event(index=i)

    # 6. Send message_delta with stop_reason and actual token counts
    stop_reason = map_finish_reason_to_stop_reason(
        output.finish_reason if output else "stop",
        bool(tool_calls)
    )
    # Use actual token counts from the last output
    actual_input_tokens = scale_anthropic_tokens(
        last_output.prompt_tokens if last_output else 0, request.model
    )
    actual_output_tokens = scale_anthropic_tokens(
        last_output.completion_tokens if last_output else 0, request.model
    )
    yield create_message_delta_event(
        stop_reason=stop_reason,
        output_tokens=actual_output_tokens,
        input_tokens=actual_input_tokens,
    )

    # Record metrics
    if last_output:
        end_time = time.perf_counter()
        ttft = (first_token_time - start_time) if first_token_time else (end_time - start_time)
        get_server_metrics().record_request_complete(
            prompt_tokens=last_output.prompt_tokens,
            completion_tokens=last_output.completion_tokens,
            cached_tokens=last_output.cached_tokens,
            prefill_duration=ttft,
            generation_duration=end_time - (first_token_time or start_time),
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

    engine = await get_engine_for_model(request.model)

    # Resolve alias to real model ID for settings lookups
    resolved_model = resolve_model_id(request.model) or request.model

    # Get per-model settings
    max_tool_result_tokens = None
    merged_ct_kwargs = {}
    forced_keys: set[str] = set()
    if _server_state.settings_manager:
        ms = _server_state.settings_manager.get_settings(resolved_model)
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
    if hasattr(request, 'thinking') and request.thinking:
        if "enable_thinking" not in forced_keys:
            thinking_type = getattr(request.thinking, 'type', None)
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
    _entry = get_engine_pool().get_entry(resolved_model)
    native_reasoning = bool(_entry and _entry.preserve_thinking_default is True)
    if engine.model_type == "gpt_oss":
        messages = convert_anthropic_to_internal_harmony(
            request, max_tool_result_tokens, engine.tokenizer
        )
    else:
        messages = convert_anthropic_to_internal(
            request, max_tool_result_tokens, engine.tokenizer,
            preserve_images=is_vlm,
            native_reasoning_content=native_reasoning,
        )

    # Apply model-specific message extraction (e.g. Gemma 4 converts
    # role=tool messages into tool_responses on assistant turns).
    extractor = getattr(engine, "message_extractor", None)
    if extractor is not None:
        messages = extractor(messages, max_tool_result_tokens, engine.tokenizer)

    # Prepare kwargs
    temperature, top_p, top_k, repetition_penalty, min_p, presence_penalty, frequency_penalty, max_tokens, xtc_probability, xtc_threshold = get_sampling_params(
        request.temperature, request.top_p, request.model,
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
    if _server_state.mcp_manager:
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

    # Validate context window before sending to model
    try:
        num_prompt_tokens = engine.count_chat_tokens(
            messages, internal_tools,
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
            raise HTTPException(
                status_code=400, detail=f"Chat template error: {e}"
            )
        raise
    validate_context_window(num_prompt_tokens, request.model)

    # Add stop sequences
    if request.stop_sequences:
        chat_kwargs["stop"] = request.stop_sequences

    if request.stream:
        return StreamingResponse(
            _with_sse_keepalive(
                stream_anthropic_messages(engine, messages, request, resolved_model=resolved_model, **chat_kwargs),
                http_request=http_request,
            ),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )

    # Non-streaming response with keepalive during prefill
    async def _build_anthropic_message():
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
        cleaned_thinking = sanitize_tool_call_markup(thinking_content, engine.tokenizer)

        # For Harmony (gpt-oss) models, tool_calls are already extracted by the parser
        # For other models, parse from text output
        if engine.model_type == "gpt_oss" and output.tool_calls:
            from .api.openai_models import ToolCall, FunctionCall
            tool_calls = [
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    type="function",
                    function=FunctionCall(
                        name=tc["name"],
                        arguments=tc["arguments"],
                    ),
                )
                for tc in output.tool_calls
            ]
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
            text=cleaned_text.strip() if cleaned_text else regular_content,
            model=request.model,
            prompt_tokens=scale_anthropic_tokens(output.prompt_tokens, request.model),
            completion_tokens=scale_anthropic_tokens(output.completion_tokens, request.model),
            finish_reason=output.finish_reason,
            tool_calls=tool_calls,
            thinking=cleaned_thinking if cleaned_thinking else None,
        )

        return response.model_dump_json()

    return StreamingResponse(
        _with_json_keepalive(http_request, _build_anthropic_message()),
        media_type="application/json",
    )


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

    engine = await get_engine_for_model(request.model)

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
        logger.warning(f"Failed to apply chat template: {e}, using simple concatenation")
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


# =============================================================================
# Responses API (/v1/responses) — OpenAI Codex compatibility
# =============================================================================


def _should_store_response(store_flag: Optional[bool]) -> bool:
    """OpenAI Responses defaults to storing responses unless explicitly disabled."""
    return store_flag is not False


def _resolve_previous_response_messages(previous_response_id: str) -> list[dict]:
    """Resolve a previous_response_id chain into chat messages."""
    try:
        return _server_state.responses_store.resolve_chain_messages(previous_response_id)
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
    engine = await get_engine_for_model(request.model)
    model_load_duration = time.perf_counter() - load_start

    resolved_model = resolve_model_id(request.model) or request.model

    current_input_messages = convert_responses_input_to_messages(request.input)

    # Build previous context from previous_response_id
    previous_messages = None
    if request.previous_response_id:
        previous_messages = _resolve_previous_response_messages(
            request.previous_response_id
        )

    # Convert Responses API input → internal messages
    messages = convert_responses_input_to_messages(
        request.input, request.instructions, previous_messages
    )

    # Convert tools: flat → nested
    openai_tools = convert_responses_tools(request.tools)

    # Get per-model settings
    max_tool_result_tokens = None
    merged_ct_kwargs = {}
    forced_keys: set[str] = set()
    reasoning_parser = None
    if _server_state.settings_manager:
        ms = _server_state.settings_manager.get_settings(resolved_model)
        max_tool_result_tokens = ms.max_tool_result_tokens
        reasoning_parser = ms.reasoning_parser
        if ms.chat_template_kwargs:
            merged_ct_kwargs.update(ms.chat_template_kwargs)
        forced_keys = set(ms.forced_ct_kwargs or [])
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

            await engine.start()
            rf = ResponseFormat(**response_format)
            compiled_grammar = _compile_grammar_for_request(
                engine, response_format=rf,
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
    effective_tools = openai_tools
    if _server_state.mcp_manager and openai_tools:
        effective_tools = _server_state.mcp_manager.get_merged_tools(openai_tools)

    # Convert tools for chat template
    tools_for_template = (
        convert_tools_for_template(effective_tools) if effective_tools else None
    )
    # Gemma 4 drops required params that lack descriptions — enrich them
    if tools_for_template and "gemma" in (resolved_model or "").lower():
        tools_for_template = enrich_tool_params_for_gemma4(tools_for_template)

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
            raise HTTPException(
                status_code=400, detail=f"Chat template error: {e}"
            )
        raise
    validate_context_window(num_prompt_tokens, request.model)

    # Build sampling kwargs
    temperature, top_p, top_k, repetition_penalty, min_p, presence_penalty, frequency_penalty, max_tokens, xtc_probability, xtc_threshold = (
        get_sampling_params(request.temperature, request.top_p, request.model, req_max_tokens=request.max_output_tokens)
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
    if (
        _entry is not None
        and _entry.preserve_thinking_default is True
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

    if request.stream:
        return StreamingResponse(
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
                    **chat_kwargs,
                ),
                http_request=http_request,
            ),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )

    # Non-streaming with keepalive during prefill
    async def _build_responses_api():
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
        if engine.model_type == "gpt_oss" and output.tool_calls:
            tool_calls = output.tool_calls
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
                cleaned_text or regular_content,
                response_format
            )
            if parsed_json is not None:
                cleaned_text = json.dumps(parsed_json)
            if not is_valid:
                logger.warning(f"JSON validation failed: {error}")

        # Build output items
        output_items: list[OutputItem] = []
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
                    call_id = tc.get("call_id", tc.get("id", f"call_{uuid.uuid4().hex[:8]}"))
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

        usage = build_response_usage(output.prompt_tokens, output.completion_tokens)

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
        _with_json_keepalive(http_request, _build_responses_api()),
        media_type="application/json",
    )


async def stream_responses_api(
    engine: BaseEngine,
    messages: list,
    request: ResponsesRequest,
    input_messages: Optional[list[dict]] = None,
    store_response: bool = True,
    model_load_duration: float = 0.0,
    resolved_model: Optional[str] = None,
    response_format=None,
    **kwargs,
) -> AsyncIterator[str]:
    """Stream Responses API events (SSE with named event types)."""
    from .api.shared_models import IDPrefix, generate_id

    start_time = time.perf_counter()
    first_token_time = None
    last_output = None
    accumulated_text = ""
    has_tools = bool(kwargs.get("tools"))
    thinking_parser = ThinkingParser()
    seq = 0

    response_id = generate_id(IDPrefix.RESPONSE)
    msg_id = generate_id(IDPrefix.MESSAGE)

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
    yield format_sse_event("response.created", {
        "type": "response.created",
        "response": initial_data,
        "sequence_number": seq,
    })

    # 2. response.in_progress
    seq += 1
    yield format_sse_event("response.in_progress", {
        "type": "response.in_progress",
        "response": initial_data,
        "sequence_number": seq,
    })

    # 3. response.output_item.added (message)
    msg_item = {
        "type": "message",
        "id": msg_id,
        "status": "in_progress",
        "role": "assistant",
        "content": [],
    }
    seq += 1
    yield format_sse_event("response.output_item.added", {
        "type": "response.output_item.added",
        "output_index": 0,
        "item": msg_item,
        "sequence_number": seq,
    })

    # 4. response.content_part.added
    content_part = {"type": "output_text", "text": "", "annotations": []}
    seq += 1
    yield format_sse_event("response.content_part.added", {
        "type": "response.content_part.added",
        "item_id": msg_id,
        "output_index": 0,
        "content_index": 0,
        "part": content_part,
        "sequence_number": seq,
    })

    # 5. Stream tokens
    tool_filter = None
    stream_content = True
    if has_tools:
        _f = ToolCallStreamFilter(engine.tokenizer)
        if _f.active:
            tool_filter = _f
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
                _thinking, content_delta = thinking_parser.feed(output.new_text)
                if content_delta:
                    if tool_filter:
                        content_delta = tool_filter.feed(content_delta)
                    if content_delta:
                        seq += 1
                        yield format_sse_event("response.output_text.delta", {
                            "type": "response.output_text.delta",
                            "item_id": msg_id,
                            "output_index": 0,
                            "content_index": 0,
                            "delta": content_delta,
                            "sequence_number": seq,
                        })
    except Exception as e:
        logger.error(f"Error during Responses API streaming: {e}")
        seq += 1
        yield format_sse_event("response.failed", {
            "type": "response.failed",
            "response": {**initial_data, "status": "failed"},
            "sequence_number": seq,
        })
        return

    # Flush remaining content from parsers
    if stream_content:
        _thinking, content_delta = thinking_parser.finish()
        if content_delta:
            if tool_filter:
                content_delta = tool_filter.feed(content_delta)
            if content_delta:
                seq += 1
                yield format_sse_event("response.output_text.delta", {
                    "type": "response.output_text.delta",
                    "item_id": msg_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": content_delta,
                    "sequence_number": seq,
                })
        if tool_filter:
            remaining = tool_filter.finish()
            if remaining:
                seq += 1
                yield format_sse_event("response.output_text.delta", {
                    "type": "response.output_text.delta",
                    "item_id": msg_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": remaining,
                    "sequence_number": seq,
                })

    # Parse tool calls from accumulated text
    tool_calls = None
    cleaned_text = accumulated_text
    if last_output and last_output.tool_calls:
        tool_calls = last_output.tool_calls
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
        if not stream_content and cleaned_text:
            seq += 1
            yield format_sse_event("response.output_text.delta", {
                "type": "response.output_text.delta",
                "item_id": msg_id,
                "output_index": 0,
                "content_index": 0,
                "delta": cleaned_text,
                "sequence_number": seq,
            })
    else:
        # No tools — use raw accumulated text minus thinking
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
        _, parsed_json, is_valid, error = parse_json_output(
            final_text, response_format
        )
        if parsed_json is not None:
            final_text = json.dumps(parsed_json)
        if not is_valid:
            logger.warning(f"JSON validation failed: {error}")

    # 6. response.output_text.done
    seq += 1
    yield format_sse_event("response.output_text.done", {
        "type": "response.output_text.done",
        "item_id": msg_id,
        "output_index": 0,
        "content_index": 0,
        "text": final_text,
        "sequence_number": seq,
    })

    # 7. response.content_part.done
    seq += 1
    yield format_sse_event("response.content_part.done", {
        "type": "response.content_part.done",
        "item_id": msg_id,
        "output_index": 0,
        "content_index": 0,
        "part": {"type": "output_text", "text": final_text, "annotations": []},
        "sequence_number": seq,
    })

    # 8. response.output_item.done (message)
    seq += 1
    yield format_sse_event("response.output_item.done", {
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {
            "type": "message",
            "id": msg_id,
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": final_text, "annotations": []}],
        },
        "sequence_number": seq,
    })

    # Build output items for final response
    output_items = [
        {
            "type": "message",
            "id": msg_id,
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": final_text, "annotations": []}],
        }
    ]

    # 9-12. Emit function call items if present
    if tool_calls:
        output_index = 1
        for tc in tool_calls:
            if hasattr(tc, "function"):
                call_id = tc.id
                name = tc.function.name
                arguments = tc.function.arguments
            elif isinstance(tc, dict):
                call_id = tc.get("call_id", tc.get("id", f"call_{uuid.uuid4().hex[:8]}"))
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
            yield format_sse_event("response.output_item.added", {
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": fc_item,
                "sequence_number": seq,
            })

            # function_call_arguments.delta
            seq += 1
            yield format_sse_event("response.function_call_arguments.delta", {
                "type": "response.function_call_arguments.delta",
                "item_id": fc_id,
                "output_index": output_index,
                "delta": arguments,
                "sequence_number": seq,
            })

            # function_call_arguments.done
            seq += 1
            yield format_sse_event("response.function_call_arguments.done", {
                "type": "response.function_call_arguments.done",
                "item_id": fc_id,
                "output_index": output_index,
                "arguments": arguments,
                "sequence_number": seq,
            })

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
            yield format_sse_event("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": output_index,
                "item": completed_fc,
                "sequence_number": seq,
            })

            output_items.append(completed_fc)
            output_index += 1

    # Record metrics
    usage_data = None
    if last_output and last_output.finished:
        end_time = time.perf_counter()
        ttft = (first_token_time - start_time) if first_token_time else (end_time - start_time)
        gen_duration = end_time - (first_token_time or start_time)
        get_server_metrics().record_request_complete(
            prompt_tokens=last_output.prompt_tokens,
            completion_tokens=last_output.completion_tokens,
            cached_tokens=last_output.cached_tokens,
            prefill_duration=ttft,
            generation_duration=gen_duration,
            model_id=resolved_model or request.model,
        )
        usage_data = {
            "input_tokens": last_output.prompt_tokens,
            "output_tokens": last_output.completion_tokens,
            "total_tokens": last_output.prompt_tokens + last_output.completion_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
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
        "tools": [t.model_dump(exclude_none=True) for t in request.tools] if request.tools else [],
        "temperature": request.temperature,
        "top_p": request.top_p,
        "max_output_tokens": request.max_output_tokens,
    }
    if request.previous_response_id:
        final_response["previous_response_id"] = request.previous_response_id

    seq += 1
    yield format_sse_event("response.completed", {
        "type": "response.completed",
        "response": final_response,
        "sequence_number": seq,
    })

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

        logger.info(f"MCP initialized with {len(_server_state.mcp_manager.get_all_tools())} tools")

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
    from .config import parse_size

    parser = argparse.ArgumentParser(
        description="oMLX multi-model serving for Apple Silicon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Multi-model serving
    python -m omlx.server --model-dir /path/to/models --max-model-memory 32GB

    # With pinned models
    python -m omlx.server --model-dir /path/to/models --max-model-memory 48GB --pin llama-3b,qwen-7b

    # With MCP tools
    python -m omlx.server --model-dir /path/to/models --max-model-memory 32GB --mcp-config mcp.json

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
        "--max-model-memory",
        type=str,
        default="32GB",
        help="Maximum memory for loaded models (e.g., 32GB). KV cache uses additional memory.",
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
        max_model_memory=parse_size(args.max_model_memory),
        pinned_models=pinned_models,
        default_model=args.default_model,
        max_tokens=args.max_tokens,
    )

    # Start server
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
