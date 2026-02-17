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

import secrets

from fastapi import Depends, FastAPI, HTTPException, Request as FastAPIRequest
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse as _BaseStreamingResponse
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
    truncate_embedding,
    normalize_input,
)
from .api.rerank_models import (
    RerankRequest,
    RerankResponse,
    RerankResult,
    RerankUsage,
)
from .api.tool_calling import (
    build_json_system_prompt,
    convert_tools_for_template,
    parse_json_output,
    parse_tool_calls,
)
from .api.utils import clean_output_text, extract_harmony_messages, extract_text_content
from .engine import BaseEngine, BatchedEngine
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


class StreamingResponse(_BaseStreamingResponse):
    """StreamingResponse that aborts generation when client disconnects.

    Monitors the ASGI receive channel for http.disconnect and closes
    the body iterator, propagating GeneratorExit through the engine's
    stream_generate which calls abort_request().
    """

    async def __call__(self, scope, receive, send):
        disconnected = asyncio.Event()

        async def _monitor_disconnect():
            while True:
                message = await receive()
                if message.get("type") == "http.disconnect":
                    disconnected.set()
                    return

        monitor_task = asyncio.create_task(_monitor_disconnect())

        inner = self.body_iterator

        async def _disconnect_aware():
            try:
                async for chunk in inner:
                    if disconnected.is_set():
                        logger.info("Client disconnected, stopping stream")
                        return
                    yield chunk
            finally:
                if hasattr(inner, "aclose"):
                    await inner.aclose()

        self.body_iterator = _disconnect_aware()
        try:
            await super().__call__(scope, receive, send)
        finally:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass


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
    top_k: int = 40
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
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> bool:
    """Verify API key if configured."""
    # No auth required if no API key is configured
    if _server_state.api_key is None:
        return True

    # Check if credentials provided
    if credentials is None:
        raise HTTPException(status_code=401, detail="API key required")

    # Constant-time comparison
    if not secrets.compare_digest(credentials.credentials, _server_state.api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan for startup/shutdown events."""
    # Startup: Preload pinned models
    if _server_state.engine_pool is not None:
        await _server_state.engine_pool.preload_pinned_models()

    # Initialize MCP if config provided
    mcp_config = os.environ.get("OMLX_MCP_CONFIG")
    if mcp_config:
        await init_mcp(mcp_config)

    yield

    # Shutdown: Stop HF downloader, MCP connections, and engines
    if _server_state.hf_downloader is not None:
        await _server_state.hf_downloader.shutdown()
        logger.info("HF Downloader stopped")
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

# Include admin routes
from .admin.routes import router as admin_router, set_admin_getters
set_admin_getters(
    get_server_state,
    get_engine_pool,
    lambda: _server_state.settings_manager,
    lambda: _server_state.global_settings,
)
app.include_router(admin_router)


@app.middleware("http")
async def debug_request_logging(request: FastAPIRequest, call_next):
    """Log full request body for POST requests when debug logging is enabled."""
    if logger.isEnabledFor(5) and request.method == "POST":
        body = await request.body()
        logger.log(
            5,
            "Incoming %s %s — body: %s",
            request.method, request.url.path,
            body.decode("utf-8", errors="replace"),
        )
    response = await call_next(request)
    return response


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

    try:
        engine = await pool.get_engine(model_id)
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
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
) -> tuple[float, float, int]:
    """
    Get effective sampling parameters with per-model settings support.

    Priority:
    - If force_sampling is True (global or model level): use forced values
    - Otherwise: request > model settings > global defaults

    Returns:
        tuple of (temperature, top_p, top_k)
    """
    global_sampling = _server_state.sampling

    # Get per-model settings if available
    model_settings = None
    if model_id and _server_state.settings_manager:
        model_settings = _server_state.settings_manager.get_settings(model_id)

    # Check force at any level
    force = global_sampling.force_sampling or (
        model_settings and model_settings.force_sampling
    )

    if force:
        # Forced mode: use model settings if available, else global
        if model_settings and model_settings.temperature is not None:
            temperature = model_settings.temperature
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
        # Normal mode: priority request > model > global
        if req_temperature is not None:
            temperature = req_temperature
        elif model_settings and model_settings.temperature is not None:
            temperature = model_settings.temperature
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

    logger.debug(
        f"Sampling params: temperature={temperature}, top_p={top_p}, top_k={top_k}"
        f"{' (forced)' if force else ''}"
        f"{f' (model: {model_id})' if model_id else ''}"
    )
    return temperature, top_p, top_k


def get_max_context_window(model_id: str | None = None) -> int | None:
    """
    Get effective max context window limit.

    Priority: model setting > global setting.

    Returns:
        Max context window token count, or None if not set.
    """
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
    model_dir: str,
    max_model_memory: int,
    scheduler_config=None,
    api_key: str | None = None,
    global_settings: object | None = None,
):
    """
    Initialize server with model directory for multi-model serving.

    Args:
        model_dir: Path to directory containing model subdirectories
        max_model_memory: Maximum memory for loaded models in bytes
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
        )
    else:
        _server_state.sampling = SamplingDefaults()

    model_path = Path(model_dir)
    if not model_path.exists():
        # Create directory if it doesn't exist (for first-time setup)
        model_path.mkdir(parents=True, exist_ok=True)
        logger.warning(f"Model directory created (empty): {model_dir}")

    # Create engine pool
    _server_state.engine_pool = EnginePool(
        max_model_memory=max_model_memory,
        scheduler_config=scheduler_config,
    )

    # Discover models (use pinned models from settings file)
    _server_state.engine_pool.discover_models(model_dir, pinned_models)

    if _server_state.engine_pool.model_count == 0:
        logger.warning(f"No models found in {model_dir}. Add models to serve them.")

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

    # Reset server metrics for fresh start
    reset_server_metrics()

    logger.info(f"Server initialized with {_server_state.engine_pool.model_count} models")
    if _server_state.default_model:
        logger.info(f"Default model: {_server_state.default_model}")
    else:
        logger.info("No default model (no models available)")
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
            _server_state.engine_pool.discover_models(model_dir, pinned)
            logger.info("Model pool refreshed after download completion")

    _server_state.hf_downloader = HFDownloader(
        model_dir=model_dir,
        on_complete=_refresh_models_after_download,
    )
    set_hf_downloader(_server_state.hf_downloader)
    logger.info("HF Downloader initialized")


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
    interval: float = 30.0,
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
                result = task.result()
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


@app.get("/v1/models")
async def list_models(_: bool = Depends(verify_api_key)) -> ModelsResponse:
    """List all available models with load status."""
    models = []

    if _server_state.engine_pool is not None:
        status = _server_state.engine_pool.get_status()
        for m in status["models"]:
            models.append(
                ModelInfo(
                    id=m["id"],
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

    return _server_state.engine_pool.get_status()


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
    _: bool = Depends(verify_api_key),
) -> EmbeddingResponse:
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
    engine = await get_embedding_engine(request.model)

    # Normalize input to list
    texts = normalize_input(request.input)

    if not texts:
        raise HTTPException(status_code=400, detail="Input cannot be empty")

    # Generate embeddings
    start_time = time.perf_counter()

    output = await engine.embed(texts)

    elapsed = time.perf_counter() - start_time
    logger.info(
        f"Embedding: {len(texts)} texts, {output.dimensions} dims, "
        f"{output.total_tokens} tokens in {elapsed:.3f}s"
    )

    # Format response
    data = []
    for i, embedding in enumerate(output.embeddings):
        # Apply dimension truncation if specified
        if request.dimensions and request.dimensions < len(embedding):
            embedding = truncate_embedding(embedding, request.dimensions)

        # Apply encoding format
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
    engine = await get_reranker_engine(request.model)

    # Normalize documents to list of strings
    documents = normalize_documents(request.documents)

    if not documents:
        raise HTTPException(status_code=400, detail="Documents cannot be empty")

    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Perform reranking
    start_time = time.perf_counter()

    output = await engine.rerank(
        query=request.query,
        documents=documents,
        top_n=request.top_n,
    )

    elapsed = time.perf_counter() - start_time
    logger.info(
        f"Rerank: {len(documents)} docs, "
        f"{output.total_tokens} tokens in {elapsed:.3f}s"
    )

    # Format response - results sorted by score (descending)
    results = []
    for idx in output.indices:
        result = RerankResult(
            index=idx,
            relevance_score=output.scores[idx],
            document={"text": documents[idx]} if request.return_documents else None,
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
    engine = await get_engine_for_model(request.model)

    # Handle single prompt or list of prompts
    prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]

    # Validate context window for each prompt
    for prompt in prompts:
        num_tokens = len(engine.tokenizer.encode(prompt))
        validate_context_window(num_tokens, request.model)

    if request.stream:
        return StreamingResponse(
            _with_sse_keepalive(
                stream_completion(engine, prompts[0], request),
                http_request=http_request,
            ),
            media_type="text/event-stream",
        )

    # Non-streaming response with timing
    start_time = time.perf_counter()
    choices = []
    total_completion_tokens = 0
    total_prompt_tokens = 0
    total_cached_tokens = 0

    temperature, top_p, top_k = get_sampling_params(
        request.temperature, request.top_p, request.model
    )

    for i, prompt in enumerate(prompts):
        output = await _run_with_disconnect_guard(
            http_request,
            engine.generate(
                prompt=prompt,
                max_tokens=request.max_tokens or _server_state.sampling.max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop=request.stop,
            ),
        )
        if output is None:
            return  # Client disconnected

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

    # Record metrics
    get_server_metrics().record_request_complete(
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        cached_tokens=total_cached_tokens,
        generation_duration=elapsed,
        model_id=request.model,
    )

    return CompletionResponse(
        model=request.model,
        choices=choices,
        usage=Usage(
            completion_tokens=total_completion_tokens,
            total_tokens=total_completion_tokens,
        ),
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

    engine = await get_engine_for_model(request.model)

    # Get max_tool_result_tokens from model settings
    max_tool_result_tokens = None
    if _server_state.settings_manager:
        ms = _server_state.settings_manager.get_settings(request.model)
        max_tool_result_tokens = ms.max_tool_result_tokens

    # Extract messages - Harmony models need special handling to preserve tool format
    if engine.model_type == "gpt_oss":
        messages = extract_harmony_messages(
            request.messages, max_tool_result_tokens, engine.tokenizer
        )
    else:
        messages = extract_text_content(
            request.messages, max_tool_result_tokens, engine.tokenizer
        )

    # Handle response_format - inject system prompt if needed
    response_format = request.response_format
    if response_format:
        json_instruction = build_json_system_prompt(response_format)
        if json_instruction:
            # Inject JSON instruction into messages
            messages = _inject_json_instruction(messages, json_instruction)

    # Validate context window before sending to model
    tools_for_template = convert_tools_for_template(request.tools) if request.tools else None
    num_prompt_tokens = engine.count_chat_tokens(messages, tools_for_template)
    validate_context_window(num_prompt_tokens, request.model)

    # Prepare kwargs
    temperature, top_p, top_k = get_sampling_params(
        request.temperature, request.top_p, request.model
    )
    chat_kwargs = {
        "max_tokens": request.max_tokens or _server_state.sampling.max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }

    # Add tools if provided
    if request.tools:
        chat_kwargs["tools"] = tools_for_template

    if request.stream:
        return StreamingResponse(
            _with_sse_keepalive(
                stream_chat_completion(engine, messages, request, **chat_kwargs),
                http_request=http_request,
            ),
            media_type="text/event-stream",
        )

    # Non-streaming response with timing
    start_time = time.perf_counter()

    output = await _run_with_disconnect_guard(
        http_request,
        engine.chat(messages=messages, **chat_kwargs),
    )
    if output is None:
        return  # Client disconnected

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = output.completion_tokens / elapsed if elapsed > 0 else 0
    logger.info(f"Chat completion: {output.completion_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)")

    # Record metrics
    get_server_metrics().record_request_complete(
        prompt_tokens=output.prompt_tokens,
        completion_tokens=output.completion_tokens,
        cached_tokens=output.cached_tokens,
        generation_duration=elapsed,
        model_id=request.model,
    )

    # For Harmony (gpt-oss) models, tool_calls are already extracted by the parser
    # For other models, parse from text output
    if engine.model_type == "gpt_oss" and output.tool_calls:
        # Harmony model with tool calls - convert format
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
        cleaned_text = clean_output_text(output.text) if output.text else ""
    else:
        # Parse tool calls from output using mlx-lm's tool parser
        cleaned_text, tool_calls = parse_tool_calls(
            output.text,
            tokenizer=engine.tokenizer,
            tools=convert_tools_for_template(request.tools),
        )

    # Process response_format if specified
    if response_format and not tool_calls:
        cleaned_text, parsed_json, is_valid, error = parse_json_output(
            cleaned_text or output.text,
            response_format
        )
        if parsed_json is not None:
            # Return JSON as string
            cleaned_text = json.dumps(parsed_json)
        if not is_valid:
            logger.warning(f"JSON validation failed: {error}")

    # Determine finish reason
    finish_reason = "tool_calls" if tool_calls else output.finish_reason

    return ChatCompletionResponse(
        model=request.model,
        choices=[ChatCompletionChoice(
            message=AssistantMessage(
                content=clean_output_text(cleaned_text) if cleaned_text else None,
                tool_calls=tool_calls,
            ),
            finish_reason=finish_reason,
        )],
        usage=Usage(
            prompt_tokens=output.prompt_tokens,
            completion_tokens=output.completion_tokens,
            total_tokens=output.prompt_tokens + output.completion_tokens,
        ),
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


# =============================================================================
# Streaming Helpers
# =============================================================================

async def stream_completion(
    engine: BaseEngine,
    prompt: str,
    request: CompletionRequest,
) -> AsyncIterator[str]:
    """Stream completion response."""
    start_time = time.perf_counter()
    first_token_time = None
    last_output = None

    temperature, top_p, top_k = get_sampling_params(
        request.temperature, request.top_p, request.model
    )
    async for output in engine.stream_generate(
        prompt=prompt,
        max_tokens=request.max_tokens or _server_state.sampling.max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        stop=request.stop,
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

    # Record metrics
    if last_output and last_output.finished:
        end_time = time.perf_counter()
        ttft = (first_token_time - start_time) if first_token_time else (end_time - start_time)
        get_server_metrics().record_request_complete(
            prompt_tokens=last_output.prompt_tokens,
            completion_tokens=last_output.completion_tokens,
            cached_tokens=last_output.cached_tokens,
            prefill_duration=ttft,
            generation_duration=end_time - (first_token_time or start_time),
            model_id=request.model,
        )

    yield "data: [DONE]\n\n"


async def stream_chat_completion(
    engine: BaseEngine,
    messages: list,
    request: ChatCompletionRequest,
    **kwargs,
) -> AsyncIterator[str]:
    """Stream chat completion response.

    Streams content tokens, then at completion parses tool calls from
    accumulated text and emits them as structured tool_calls chunks
    (OpenAI streaming format).
    """
    start_time = time.perf_counter()
    first_token_time = None
    last_output = None
    accumulated_text = ""
    has_tools = bool(kwargs.get("tools"))

    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    # First chunk with role
    first_chunk = ChatCompletionChunk(
        id=response_id,
        model=request.model,
        choices=[ChatCompletionChunkChoice(
            delta=ChatCompletionChunkDelta(role="assistant"),
        )],
    )
    yield f"data: {first_chunk.model_dump_json()}\n\n"

    # Stream content — buffer when tools are present so we can strip
    # tool call markup before emitting (prevents clients from seeing
    # tool calls in both content and structured tool_calls chunks).
    async for output in engine.stream_chat(messages=messages, **kwargs):
        if first_token_time is None and output.new_text:
            first_token_time = time.perf_counter()
        last_output = output
        if output.new_text:
            accumulated_text += output.new_text

        if not has_tools:
            chunk = ChatCompletionChunk(
                id=response_id,
                model=request.model,
                choices=[ChatCompletionChunkChoice(
                    delta=ChatCompletionChunkDelta(content=output.new_text if output.new_text else None),
                    finish_reason=None,
                )],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

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
        # Parse from accumulated text using mlx-lm's tool parser
        cleaned_text, tool_calls = parse_tool_calls(
            accumulated_text,
            tokenizer=engine.tokenizer,
            tools=kwargs.get("tools"),
        )

    # When tools were requested, emit buffered content now (cleaned of markup)
    if has_tools and cleaned_text:
        content_chunk = ChatCompletionChunk(
            id=response_id,
            model=request.model,
            choices=[ChatCompletionChunkChoice(
                delta=ChatCompletionChunkDelta(content=cleaned_text),
                finish_reason=None,
            )],
        )
        yield f"data: {content_chunk.model_dump_json()}\n\n"

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
            yield f"data: {tc_chunk.model_dump_json()}\n\n"

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
    yield f"data: {final_chunk.model_dump_json()}\n\n"

    # Record metrics
    if last_output and last_output.finished:
        end_time = time.perf_counter()
        ttft = (first_token_time - start_time) if first_token_time else (end_time - start_time)
        get_server_metrics().record_request_complete(
            prompt_tokens=last_output.prompt_tokens,
            completion_tokens=last_output.completion_tokens,
            cached_tokens=last_output.cached_tokens,
            prefill_duration=ttft,
            generation_duration=end_time - (first_token_time or start_time),
            model_id=request.model,
        )

    yield "data: [DONE]\n\n"


# =============================================================================
# Anthropic Messages API
# =============================================================================


async def stream_anthropic_messages(
    engine: BaseEngine,
    messages: list,
    request: AnthropicMessagesRequest,
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

    # Track content blocks - skip <think> content entirely for Harmony
    in_thinking_block = False
    text_block_started = False
    last_output = None  # Track last output for tool_calls and token counts

    # Calculate input tokens before streaming starts
    # This is needed for message_start event
    estimated_input_tokens = 0
    try:
        if hasattr(engine, 'tokenizer') and engine.tokenizer is not None:
            # Build the prompt using chat template
            template_kwargs = {"tokenize": False, "add_generation_prompt": True}
            if kwargs.get("tools"):
                template_kwargs["tools"] = kwargs["tools"]
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

    # 3. Stream content - for Harmony, skip <think> content entirely
    try:
        async for output in engine.stream_chat(messages=messages, **kwargs):
            last_output = output  # Keep reference for tool_calls and token counts

            if first_token_time is None and output.new_text:
                first_token_time = time.perf_counter()

            if output.new_text:
                chunk = output.new_text

                # Check if entering thinking mode - skip <think> content
                if '<think>' in chunk:
                    in_thinking_block = True
                    # Don't accumulate or send thinking content
                    continue

                # Check if exiting thinking mode
                if '</think>' in chunk:
                    in_thinking_block = False
                    # Don't send the </think> tag either
                    continue

                # Skip all thinking content
                if in_thinking_block:
                    continue

                # Regular text (outside thinking) - accumulate and send
                accumulated_text += chunk

                # Start text block if not started
                if not text_block_started:
                    yield create_content_block_start_event(index=0, block_type="text")
                    text_block_started = True
                yield create_text_delta_event(index=0, text=chunk)

            if output.finished:
                break
    except Exception as e:
        logger.error(f"Error during Anthropic streaming: {e}")
        yield create_error_event("api_error", str(e))
        yield create_message_stop_event()
        return

    # 4. Close text block if started, or create empty block
    if text_block_started:
        yield create_content_block_stop_event(index=0)
    else:
        # No text was sent - start and close an empty text block
        yield create_content_block_start_event(index=0, block_type="text")
        yield create_content_block_stop_event(index=0)

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
        # Non-Harmony: parse from accumulated text
        cleaned_text, tool_calls = parse_tool_calls(
            accumulated_text,
            tokenizer=engine.tokenizer,
            tools=kwargs.get("tools"),
        )

    # Emit tool_use blocks if present
    if tool_calls:
        for i, tc in enumerate(tool_calls, start=1):
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
            model_id=request.model,
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

    engine = await get_engine_for_model(request.model)

    # Get max_tool_result_tokens from model settings
    max_tool_result_tokens = None
    if _server_state.settings_manager:
        ms = _server_state.settings_manager.get_settings(request.model)
        max_tool_result_tokens = ms.max_tool_result_tokens

    logger.debug(
        f"Tool result truncation config: max_tokens={max_tool_result_tokens}, "
        f"has_tokenizer={engine.tokenizer is not None}"
    )

    # Convert Anthropic format to internal format
    # Harmony models need special handling to preserve tool format
    if engine.model_type == "gpt_oss":
        messages = convert_anthropic_to_internal_harmony(
            request, max_tool_result_tokens, engine.tokenizer
        )
    else:
        messages = convert_anthropic_to_internal(
            request, max_tool_result_tokens, engine.tokenizer
        )

    # Prepare kwargs
    temperature, top_p, top_k = get_sampling_params(
        request.temperature, request.top_p, request.model
    )

    # Apply max_tokens from model settings if force_sampling is enabled
    max_tokens = request.max_tokens
    if _server_state.settings_manager:
        ms = _server_state.settings_manager.get_settings(request.model)
        force = _server_state.sampling.force_sampling or (ms and ms.force_sampling)
        if force and ms and ms.max_tokens is not None:
            max_tokens = ms.max_tokens

    chat_kwargs = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }

    # Add tools if provided
    internal_tools = convert_anthropic_tools_to_internal(request.tools)
    if internal_tools:
        chat_kwargs["tools"] = internal_tools

    # Validate context window before sending to model
    num_prompt_tokens = engine.count_chat_tokens(messages, internal_tools)
    validate_context_window(num_prompt_tokens, request.model)

    # Add stop sequences
    if request.stop_sequences:
        chat_kwargs["stop"] = request.stop_sequences

    if request.stream:
        return StreamingResponse(
            _with_sse_keepalive(
                stream_anthropic_messages(engine, messages, request, **chat_kwargs),
                http_request=http_request,
            ),
            media_type="text/event-stream",
        )

    # Non-streaming response
    start_time = time.perf_counter()

    output = await _run_with_disconnect_guard(
        http_request,
        engine.chat(messages=messages, **chat_kwargs),
    )
    if output is None:
        return  # Client disconnected

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = output.completion_tokens / elapsed if elapsed > 0 else 0
    logger.info(
        f"Anthropic message: {output.completion_tokens} tokens in {elapsed:.2f}s "
        f"({tokens_per_sec:.1f} tok/s)"
    )

    # Record metrics
    get_server_metrics().record_request_complete(
        prompt_tokens=output.prompt_tokens,
        completion_tokens=output.completion_tokens,
        cached_tokens=output.cached_tokens,
        generation_duration=elapsed,
        model_id=request.model,
    )

    # For Harmony (gpt-oss) models, tool_calls are already extracted by the parser
    # For other models, parse from text output
    if engine.model_type == "gpt_oss" and output.tool_calls:
        # Harmony model with tool calls - convert format
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
        cleaned_text = clean_output_text(output.text) if output.text else ""
    else:
        # Parse tool calls from output text (non-Harmony models)
        cleaned_text, tool_calls = parse_tool_calls(
            output.text,
            tokenizer=engine.tokenizer,
            tools=internal_tools,
        )
        # Clean output text
        cleaned_text = clean_output_text(cleaned_text) if cleaned_text else output.text

    # Convert to Anthropic response format
    response = convert_internal_to_anthropic_response(
        text=cleaned_text,
        model=request.model,
        prompt_tokens=scale_anthropic_tokens(output.prompt_tokens, request.model),
        completion_tokens=scale_anthropic_tokens(output.completion_tokens, request.model),
        finish_reason=output.finish_reason,
        tool_calls=tool_calls,
    )

    return response


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
        logger.error("MCP SDK not installed. Install with: pip install mcp")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize MCP: {e}")
        raise


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
