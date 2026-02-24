# SPDX-License-Identifier: Apache-2.0
"""Admin panel routes for oMLX server configuration.

This module provides HTTP routes for the admin panel including:
- Login/logout with API key authentication
- Dashboard for server monitoring
- Model settings management (per-model sampling parameters, pinning, default)
- Global settings management
"""

import asyncio
import logging
import os
import shutil
from collections import deque
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .auth import verify_api_key, validate_api_key, create_session_token, require_admin

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models
# =============================================================================


class LoginRequest(BaseModel):
    """Request model for admin login."""

    api_key: str


class SetupApiKeyRequest(BaseModel):
    """Request model for initial API key setup."""

    api_key: str
    api_key_confirm: str


class ModelSettingsRequest(BaseModel):
    """Request model for updating per-model settings."""

    max_context_window: Optional[int] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    force_sampling: Optional[bool] = None
    max_tool_result_tokens: Optional[int] = None
    is_pinned: Optional[bool] = None
    is_default: Optional[bool] = None


class GlobalSettingsRequest(BaseModel):
    """Request model for updating global server settings."""

    # Server settings
    host: Optional[str] = None
    port: Optional[int] = None
    log_level: Optional[str] = None

    # Model settings
    model_dirs: Optional[List[str]] = None
    model_dir: Optional[str] = None  # Deprecated: kept for backward compatibility
    max_model_memory: Optional[str] = None

    # Scheduler settings
    max_num_seqs: Optional[int] = None
    prefill_batch_size: Optional[int] = None
    completion_batch_size: Optional[int] = None

    # Cache settings
    cache_enabled: Optional[bool] = None
    ssd_cache_dir: Optional[str] = None
    ssd_cache_max_size: Optional[str] = None

    # MCP settings
    mcp_config: Optional[str] = None

    # Sampling defaults
    sampling_max_context_window: Optional[int] = None
    sampling_max_tokens: Optional[int] = None
    sampling_temperature: Optional[float] = None
    sampling_top_p: Optional[float] = None
    sampling_top_k: Optional[int] = None
    sampling_repetition_penalty: Optional[float] = None

    # Claude Code settings
    claude_code_context_scaling_enabled: Optional[bool] = None
    claude_code_target_context_size: Optional[int] = None

    # Auth settings
    api_key: Optional[str] = None


class HFDownloadRequest(BaseModel):
    """Request model for starting a HuggingFace model download."""

    repo_id: str
    hf_token: str = ""


# =============================================================================
# Runtime Settings Application Functions
# =============================================================================


def _format_cache_size(size_bytes: int) -> str:
    """Format cache size in bytes to human-readable string (e.g., '100GB')."""
    gb = size_bytes / (1024 ** 3)
    if gb >= 1:
        return f"{gb:.0f}GB"
    mb = size_bytes / (1024 ** 2)
    return f"{mb:.0f}MB"


def _apply_log_level_runtime(level: str) -> None:
    """Apply log level change at runtime to all oMLX loggers and handlers."""
    level_name = level.upper()
    log_level = 5 if level_name == "TRACE" else getattr(logging, level_name, logging.INFO)

    # Update root logger level and all its handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    for handler in root_logger.handlers:
        handler.setLevel(log_level)

    # Update omlx-related loggers
    omlx_loggers = [
        "omlx",
        "omlx.scheduler",
        "omlx.paged_ssd_cache",
        "omlx.memory_monitor",
        "omlx.paged_cache",
        "omlx.prefix_cache",
        "omlx.engine_pool",
        "omlx.model_discovery",
        "omlx.engine_core",
        "omlx.engine",
        "omlx.server",
        "omlx.admin",
    ]

    for logger_name in omlx_loggers:
        logging.getLogger(logger_name).setLevel(log_level)

    # Also update uvicorn logger
    logging.getLogger("uvicorn").setLevel(log_level)
    logging.getLogger("uvicorn.access").setLevel(log_level)


async def _apply_model_dirs_runtime(model_dirs: list[str]) -> tuple[bool, str]:
    """
    Apply model directories change at runtime by re-scanning models.

    This will:
    1. Validate all directories
    2. Unload all currently loaded models
    3. Clear the entries dictionary
    4. Re-discover models from the new directories

    Returns:
        Tuple of (success, message)
    """
    from pathlib import Path
    from ..server import _server_state

    if _server_state.engine_pool is None:
        return False, "Engine pool not initialized"

    # Validate all model directories
    for model_dir in model_dirs:
        model_path = Path(model_dir).expanduser().resolve()
        if not model_path.exists():
            return False, f"Model directory does not exist: {model_dir}"
        if not model_path.is_dir():
            return False, f"Path is not a directory: {model_dir}"

    pool = _server_state.engine_pool

    # Get pinned models from settings_manager
    pinned_models = []
    if _server_state.settings_manager is not None:
        pinned_models = _server_state.settings_manager.get_pinned_model_ids()

    # Unload all loaded models
    loaded_models = pool.get_loaded_model_ids()
    for model_id in loaded_models:
        try:
            await pool._unload_engine(model_id)
        except Exception as e:
            logger.warning(f"Error unloading {model_id}: {e}")

    # Clear entries
    pool._entries.clear()
    pool._current_model_memory = 0

    # Re-discover models from new directories
    try:
        pool.discover_models(model_dirs, pinned_models)
    except Exception as e:
        return False, f"Failed to discover models: {e}"

    dir_count = len(model_dirs)
    return True, (
        f"Re-discovered {pool.model_count} models "
        f"from {dir_count} director{'ies' if dir_count > 1 else 'y'}"
    )


async def _apply_max_model_memory_runtime(max_memory_bytes: int) -> tuple[bool, str]:
    """
    Apply max model memory change at runtime.

    If current usage exceeds new limit, unloads LRU models until within limit.

    Returns:
        Tuple of (success, message)
    """
    from ..server import _server_state
    from ..model_discovery import format_size

    if _server_state.engine_pool is None:
        return False, "Engine pool not initialized"

    pool = _server_state.engine_pool
    old_limit = pool._max_model_memory
    pool._max_model_memory = max_memory_bytes

    # If current usage exceeds new limit, unload LRU models
    unloaded = []
    while pool._current_model_memory > max_memory_bytes:
        victim = pool._find_lru_victim()
        if not victim:
            # All models are pinned, can't free more memory
            break
        await pool._unload_engine(victim)
        unloaded.append(victim)

    msg = f"Max model memory changed: {format_size(old_limit)} -> {format_size(max_memory_bytes)}"
    if unloaded:
        msg += f", unloaded: {', '.join(unloaded)}"

    return True, msg


async def _apply_cache_settings_runtime(
    enabled: Optional[bool],
    ssd_cache_dir: Optional[str],
    ssd_cache_max_size: Optional[str],
    global_settings,
) -> tuple[bool, str]:
    """
    Apply cache settings at runtime.

    Updates the scheduler_config and unloads all models so they
    will use the new cache settings when reloaded.

    Returns:
        Tuple of (success, message)
    """
    from ..server import _server_state
    from ..config import parse_size

    if _server_state.engine_pool is None:
        return False, "Engine pool not initialized"

    pool = _server_state.engine_pool

    # Update scheduler config based on cache settings
    if enabled is False or (enabled is None and not global_settings.cache.enabled):
        pool._scheduler_config.paged_ssd_cache_dir = None
        pool._scheduler_config.paged_ssd_cache_max_size = 0
    else:
        # Cache is enabled
        if ssd_cache_dir is not None:
            pool._scheduler_config.paged_ssd_cache_dir = ssd_cache_dir
        elif global_settings.cache.ssd_cache_dir:
            pool._scheduler_config.paged_ssd_cache_dir = global_settings.cache.ssd_cache_dir
        else:
            # Use default cache dir
            pool._scheduler_config.paged_ssd_cache_dir = str(
                global_settings.cache.get_ssd_cache_dir(global_settings.base_path)
            )

        if ssd_cache_max_size is not None:
            # Handle "auto" value
            if ssd_cache_max_size.lower() == "auto":
                pool._scheduler_config.paged_ssd_cache_max_size = (
                    global_settings.cache.get_ssd_cache_max_size_bytes(global_settings.base_path)
                )
            else:
                pool._scheduler_config.paged_ssd_cache_max_size = parse_size(ssd_cache_max_size)
        elif global_settings.cache.ssd_cache_max_size:
            # Use settings value (handles "auto")
            pool._scheduler_config.paged_ssd_cache_max_size = (
                global_settings.cache.get_ssd_cache_max_size_bytes(global_settings.base_path)
            )
        elif global_settings.cache.ssd_cache_max_size:
            pool._scheduler_config.paged_ssd_cache_max_size = parse_size(
                global_settings.cache.ssd_cache_max_size
            )

    # Unload all loaded models so they use new config when reloaded
    loaded_models = pool.get_loaded_model_ids()
    for model_id in loaded_models:
        try:
            await pool._unload_engine(model_id)
        except Exception as e:
            logger.warning(f"Error unloading {model_id}: {e}")

    return True, f"Cache settings updated. Unloaded {len(loaded_models)} models."


def _apply_sampling_settings_runtime(
    max_context_window: Optional[int],
    max_tokens: Optional[int],
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    repetition_penalty: Optional[float] = None,
) -> tuple[bool, str]:
    """
    Apply sampling default settings at runtime.

    Updates _server_state.sampling which is used for all new API requests.

    Returns:
        Tuple of (success, message)
    """
    from ..server import _server_state

    changes = []

    if max_context_window is not None:
        _server_state.sampling.max_context_window = max_context_window
        changes.append(f"max_context_window={max_context_window}")

    if max_tokens is not None:
        _server_state.sampling.max_tokens = max_tokens
        changes.append(f"max_tokens={max_tokens}")

    if temperature is not None:
        _server_state.sampling.temperature = temperature
        changes.append(f"temperature={temperature}")

    if top_p is not None:
        _server_state.sampling.top_p = top_p
        changes.append(f"top_p={top_p}")

    if top_k is not None:
        _server_state.sampling.top_k = top_k
        changes.append(f"top_k={top_k}")

    if repetition_penalty is not None:
        _server_state.sampling.repetition_penalty = repetition_penalty
        changes.append(f"repetition_penalty={repetition_penalty}")

    if changes:
        return True, f"Sampling defaults updated: {', '.join(changes)}"
    return True, "No sampling changes"


# =============================================================================
# Router and Templates
# =============================================================================

router = APIRouter(prefix="/admin", tags=["admin"])
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
static_dir = Path(__file__).parent / "static"


def _static_version(path: str) -> str:
    """Append file mtime as query string for cache busting."""
    file_path = static_dir / path
    if file_path.is_file():
        mtime = int(file_path.stat().st_mtime)
        return f"/admin/static/{path}?v={mtime}"
    return f"/admin/static/{path}"


templates.env.globals["static"] = _static_version


# =============================================================================
# State Getters (set by server.py)
# =============================================================================

_get_server_state = None
_get_engine_pool = None
_get_settings_manager = None
_get_global_settings = None
_hf_downloader = None


def set_admin_getters(
    state_getter,
    pool_getter,
    settings_manager_getter,
    global_settings_getter,
):
    """
    Set the getter functions for accessing server state.

    This function must be called during server initialization to provide
    access to the server state objects.

    Args:
        state_getter: Function that returns the ServerState instance.
        pool_getter: Function that returns the EnginePool instance.
        settings_manager_getter: Function that returns the ModelSettingsManager.
        global_settings_getter: Function that returns the GlobalSettings.
    """
    global _get_server_state, _get_engine_pool, _get_settings_manager, _get_global_settings
    _get_server_state = state_getter
    _get_engine_pool = pool_getter
    _get_settings_manager = settings_manager_getter
    _get_global_settings = global_settings_getter


def set_hf_downloader(downloader):
    """Set the HFDownloader instance for admin routes.

    Args:
        downloader: HFDownloader instance created during server initialization.
    """
    global _hf_downloader
    _hf_downloader = downloader


# =============================================================================
# Helper Functions
# =============================================================================


def format_size(size_bytes: int) -> str:
    """
    Format a byte size as a human-readable string.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Human-readable string (e.g., "1.5 GB").
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.1f} MB"
    elif size_bytes < 1024**4:
        return f"{size_bytes / 1024**3:.2f} GB"
    else:
        return f"{size_bytes / 1024**4:.2f} TB"


def get_ssd_disk_info(cache_dir: str) -> dict:
    """
    Get disk information for the SSD cache directory.

    Returns:
        Dictionary with total_bytes, free_bytes, total_formatted, free_formatted.
    """
    try:
        stat = shutil.disk_usage(cache_dir)
        return {
            "total_bytes": stat.total,
            "free_bytes": stat.free,
            "total_formatted": format_size(stat.total),
            "free_formatted": format_size(stat.free),
        }
    except Exception as e:
        logger.warning(f"Failed to get disk info for {cache_dir}: {e}")
        return {
            "total_bytes": 0,
            "free_bytes": 0,
            "total_formatted": "Unknown",
            "free_formatted": "Unknown",
        }


def get_system_memory_info() -> dict:
    """
    Get system memory information.

    Returns:
        Dictionary with total_bytes, total_formatted, auto_limit_bytes,
        and auto_limit_formatted (80% of total).
    """
    try:
        # macOS: use sysctl to get physical memory
        import subprocess

        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        total_bytes = int(result.stdout.strip())
    except Exception:
        # Fallback: try os.sysconf (works on some Unix systems)
        try:
            total_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except Exception:
            total_bytes = 0

    auto_limit_bytes = int(total_bytes * 0.8)

    return {
        "total_bytes": total_bytes,
        "total_formatted": format_size(total_bytes),
        "auto_limit_bytes": auto_limit_bytes,
        "auto_limit_formatted": format_size(auto_limit_bytes),
    }


# =============================================================================
# HTML Page Routes
# =============================================================================


@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    """
    Render the admin login page or setup page.

    If no API key is configured, the page will show the initial setup form.
    Otherwise, it shows the standard login form.

    Returns:
        HTML login/setup page.
    """
    global_settings = _get_global_settings()
    api_key_configured = bool(global_settings and global_settings.auth.api_key)
    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "api_key_configured": api_key_configured,
        },
    )


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request, is_admin: bool = Depends(require_admin)):
    """
    Render the admin dashboard page.

    Requires admin authentication via session cookie.

    Returns:
        HTML dashboard page with server status and model list.
    """
    return templates.TemplateResponse("dashboard.html", {"request": request})


@router.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request, is_admin: bool = Depends(require_admin)):
    """
    Render the chat page for interacting with models.

    Requires admin authentication via session cookie.
    The API key is injected into the template context so that
    the chat page can auto-set it in localStorage, bypassing
    the manual API key entry modal.

    Returns:
        HTML chat page.
    """
    global_settings = _get_global_settings()
    api_key = global_settings.auth.api_key if global_settings else ""
    return templates.TemplateResponse(
        "chat.html", {"request": request, "api_key": api_key or ""}
    )


@router.get("/static/{path:path}")
async def admin_static(path: str):
    """Serve static files for admin panel (CSS, JS, fonts, logos, etc.)."""
    file_path = static_dir / path
    if not file_path.is_file() or not file_path.resolve().is_relative_to(
        static_dir.resolve()
    ):
        raise HTTPException(status_code=404, detail="File not found")
    media_types = {
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".ico": "image/x-icon",
        ".css": "text/css",
        ".js": "application/javascript",
        ".woff2": "font/woff2",
        ".woff": "font/woff",
        ".ttf": "font/ttf",
    }
    media_type = media_types.get(file_path.suffix, "application/octet-stream")
    return FileResponse(file_path, media_type=media_type)


# =============================================================================
# Authentication API Routes
# =============================================================================


@router.post("/api/login")
async def login(request: LoginRequest, response: Response):
    """
    Authenticate with API key and create session.

    Requires an API key to be configured on the server. If no API key
    is configured, returns 400 directing the user to set one up first.

    Args:
        request: LoginRequest containing the API key.
        response: FastAPI response object for setting cookies.

    Returns:
        JSON response with success status.

    Raises:
        HTTPException: 400 if no API key configured, 401 if invalid.
    """
    global_settings = _get_global_settings()
    server_api_key = global_settings.auth.api_key if global_settings else None

    # Reject login if no API key is configured (must use setup first)
    if not server_api_key:
        raise HTTPException(
            status_code=400,
            detail="No API key configured. Please set up an API key first.",
        )

    if not verify_api_key(request.api_key, server_api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )

    # Create session token and set cookie
    token = create_session_token()
    response.set_cookie(
        key="omlx_admin_session",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=86400,  # 24 hours
    )

    return {"success": True}


@router.post("/api/setup-api-key")
async def setup_api_key(request: SetupApiKeyRequest, response: Response):
    """
    Set up the initial API key when none is configured.

    This endpoint is only available when no API key is currently set.
    After successful setup, a session is created so the user is
    immediately logged in.

    Args:
        request: SetupApiKeyRequest with api_key and api_key_confirm.
        response: FastAPI response object for setting cookies.

    Returns:
        JSON response with success status.

    Raises:
        HTTPException: 400 if key already configured, validation fails,
                      or keys don't match.
    """
    from ..server import _server_state

    global_settings = _get_global_settings()

    # Only allow setup if no API key is currently configured
    if global_settings and global_settings.auth.api_key:
        raise HTTPException(
            status_code=400,
            detail="API key is already configured. Use settings to change it.",
        )

    # Validate confirmation match
    if request.api_key != request.api_key_confirm:
        raise HTTPException(status_code=400, detail="API keys do not match")

    # Validate key format
    is_valid, error_msg = validate_api_key(request.api_key)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)

    # Apply to settings and runtime
    global_settings.auth.api_key = request.api_key
    _server_state.api_key = request.api_key

    # Persist to file
    try:
        global_settings.save()
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to save settings: {e}"
        )

    logger.info("API key configured via initial setup")

    # Create session token and set cookie (auto-login after setup)
    token = create_session_token()
    response.set_cookie(
        key="omlx_admin_session",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=86400,  # 24 hours
    )

    return {"success": True, "message": "API key configured successfully"}


@router.post("/api/logout")
async def logout(response: Response):
    """
    Clear session cookie and logout.

    Args:
        response: FastAPI response object for clearing cookies.

    Returns:
        JSON response with success status.
    """
    response.delete_cookie(key="omlx_admin_session")
    return {"success": True}


@router.get("/auto-login")
async def auto_login(key: str = "", redirect: str = "/admin/dashboard"):
    """
    Auto-login using API key and redirect to the target admin page.

    Used by the macOS menubar app to open admin pages with automatic
    authentication, bypassing the manual login form.

    Args:
        key: The API key for authentication.
        redirect: The path to redirect to after login. Must start with /admin.

    Returns:
        HTTP 302 redirect with session cookie set.
    """
    if not redirect.startswith("/admin"):
        raise HTTPException(status_code=400, detail="Invalid redirect path")

    global_settings = _get_global_settings()
    server_api_key = global_settings.auth.api_key if global_settings else None

    if not key or not server_api_key or not verify_api_key(key, server_api_key):
        return RedirectResponse(url="/admin", status_code=302)

    token = create_session_token()
    response = RedirectResponse(url=redirect, status_code=302)
    response.set_cookie(
        key="omlx_admin_session",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=86400,
    )
    return response


# =============================================================================
# Models API Routes
# =============================================================================


@router.get("/api/models")
async def list_models(is_admin: bool = Depends(require_admin)):
    """
    List all models with their settings.

    Returns model information from the engine pool combined with
    per-model settings from the settings manager.

    Returns:
        JSON list of models with their status and settings.

    Raises:
        HTTPException: 401 if not authenticated, 503 if server not initialized.
    """
    engine_pool = _get_engine_pool()
    settings_manager = _get_settings_manager()
    server_state = _get_server_state()

    if engine_pool is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    # Get engine pool status
    status = engine_pool.get_status()
    models_status = status.get("models", [])

    # Get all model settings
    all_settings = settings_manager.get_all_settings() if settings_manager else {}

    # Combine model info with settings
    models = []
    for model_info in models_status:
        model_id = model_info["id"]
        settings = all_settings.get(model_id)

        model_data = {
            "id": model_id,
            "loaded": model_info.get("loaded", False),
            "estimated_size": model_info.get("estimated_size", 0),
            "estimated_size_formatted": format_size(model_info.get("estimated_size", 0)),
            "pinned": model_info.get("pinned", False),
            "is_default": server_state.default_model == model_id if server_state else False,
            "engine_type": model_info.get("engine_type", "batched"),
            "model_type": model_info.get("model_type", "llm"),
            "last_access": model_info.get("last_access"),
        }

        # Add settings if available
        if settings:
            model_data["settings"] = {
                "max_context_window": settings.max_context_window,
                "max_tokens": settings.max_tokens,
                "temperature": settings.temperature,
                "top_p": settings.top_p,
                "top_k": settings.top_k,
                "repetition_penalty": settings.repetition_penalty,
                "force_sampling": settings.force_sampling,
                "max_tool_result_tokens": settings.max_tool_result_tokens,
                "is_pinned": settings.is_pinned,
                "is_default": settings.is_default,
                "display_name": settings.display_name,
                "description": settings.description,
            }

        models.append(model_data)

    return {"models": models}


@router.post("/api/models/{model_id}/unload")
async def unload_model(
    model_id: str,
    is_admin: bool = Depends(require_admin),
):
    """Manually unload a model from memory."""
    engine_pool = _get_engine_pool()
    if engine_pool is None:
        raise HTTPException(status_code=503, detail="Engine pool not initialized")

    entry = engine_pool.get_entry(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    if entry.engine is None:
        raise HTTPException(status_code=400, detail=f"Model not loaded: {model_id}")

    await engine_pool._unload_engine(model_id)
    logger.info(f"Manually unloaded model: {model_id}")
    return {"status": "ok", "model_id": model_id, "message": f"Unloaded {model_id}"}


@router.put("/api/models/{model_id}/settings")
async def update_model_settings(
    model_id: str,
    request: ModelSettingsRequest,
    is_admin: bool = Depends(require_admin),
):
    """
    Update settings for a specific model.

    Updates are persisted to the settings file and applied immediately
    to the engine pool where applicable (e.g., pinned status).

    Args:
        model_id: The model identifier.
        request: ModelSettingsRequest with the new settings.

    Returns:
        JSON response with success status and updated settings.

    Raises:
        HTTPException: 401 if not authenticated, 404 if model not found.
    """
    engine_pool = _get_engine_pool()
    settings_manager = _get_settings_manager()
    server_state = _get_server_state()

    if engine_pool is None or settings_manager is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    # Check if model exists
    entry = engine_pool.get_entry(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    # Get current settings
    from ..model_settings import ModelSettings
    current_settings = settings_manager.get_settings(model_id)

    # Apply updates — use model_fields_set to distinguish "sent as null"
    # (clear to default) from "not sent" (don't touch).
    sent = request.model_fields_set
    if "max_context_window" in sent:
        current_settings.max_context_window = request.max_context_window
    if "max_tokens" in sent:
        current_settings.max_tokens = request.max_tokens
    if "temperature" in sent:
        current_settings.temperature = request.temperature
    if "top_p" in sent:
        current_settings.top_p = request.top_p
    if "top_k" in sent:
        current_settings.top_k = request.top_k
    if "repetition_penalty" in sent:
        current_settings.repetition_penalty = request.repetition_penalty
    if "force_sampling" in sent:
        current_settings.force_sampling = request.force_sampling
    if "max_tool_result_tokens" in sent:
        # 0 means disable (reset to None)
        current_settings.max_tool_result_tokens = (
            request.max_tool_result_tokens if request.max_tool_result_tokens and request.max_tool_result_tokens > 0 else None
        )
    if request.is_pinned is not None:
        current_settings.is_pinned = request.is_pinned
        # Also update the engine pool entry
        entry.is_pinned = request.is_pinned
    if request.is_default is not None:
        current_settings.is_default = request.is_default
        # Update server_state.default_model if setting as default
        if request.is_default and server_state:
            server_state.default_model = model_id

    # Persist settings
    settings_manager.set_settings(model_id, current_settings)

    return {
        "success": True,
        "model_id": model_id,
        "settings": current_settings.to_dict(),
    }


@router.get("/api/models/{model_id}/generation_config")
async def get_generation_config(
    model_id: str,
    is_admin: bool = Depends(require_admin),
):
    """
    Read model config files and return recommended defaults.

    Reads generation_config.json for sampling parameters and config.json
    for max_context_window (max_position_embeddings).

    Args:
        model_id: The model identifier.

    Returns:
        JSON with recommended parameters from the model's config files.

    Raises:
        HTTPException: 404 if model not found or no config files exist.
    """
    import json as json_module

    engine_pool = _get_engine_pool()
    if engine_pool is None:
        raise HTTPException(status_code=503, detail="Engine pool not initialized")

    entry = engine_pool.get_entry(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    model_path = Path(entry.model_path)
    result = {}

    # Read generation_config.json for sampling parameters
    gen_config_path = model_path / "generation_config.json"
    if gen_config_path.exists():
        try:
            with open(gen_config_path, "r", encoding="utf-8") as f:
                gen_config = json_module.load(f)

            # Temperature: if do_sample is false, effective temperature is 0
            do_sample = gen_config.get("do_sample", True)
            if "temperature" in gen_config:
                result["temperature"] = 0.0 if not do_sample else gen_config["temperature"]

            if "top_p" in gen_config:
                result["top_p"] = gen_config["top_p"]

            if "top_k" in gen_config:
                result["top_k"] = gen_config["top_k"]

            if "repetition_penalty" in gen_config:
                result["repetition_penalty"] = gen_config["repetition_penalty"]

        except (json_module.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to parse generation_config.json for {model_id}: {e}")

    # Read config.json for max_position_embeddings → max_context_window
    config_path = model_path / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                model_config = json_module.load(f)

            max_pos = (
                model_config.get("max_position_embeddings")
                or model_config.get("max_seq_len")
                or model_config.get("seq_length")
                or model_config.get("n_positions")
            )
            if max_pos and isinstance(max_pos, int):
                result["max_context_window"] = max_pos

        except (json_module.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to parse config.json for {model_id}: {e}")

    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No config files with defaults found for {model_id}",
        )

    return result


# =============================================================================
# Global Settings API Routes
# =============================================================================


@router.get("/api/global-settings")
async def get_global_settings(is_admin: bool = Depends(require_admin)):
    """
    Get current global server settings.

    Returns the full global settings including server, model, scheduler,
    cache, and MCP configurations.

    Returns:
        JSON object with global settings.

    Raises:
        HTTPException: 401 if not authenticated, 503 if server not initialized.
    """
    global_settings = _get_global_settings()

    if global_settings is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    # Get system memory info for auto calculation
    memory_info = get_system_memory_info()

    # Get SSD disk info for cache directory
    cache_dir = global_settings.cache.ssd_cache_dir or str(
        global_settings.cache.get_ssd_cache_dir(global_settings.base_path)
    )
    disk_info = get_ssd_disk_info(cache_dir)

    return {
        "base_path": str(global_settings.base_path),
        "server": {
            "host": global_settings.server.host,
            "port": global_settings.server.port,
            "log_level": global_settings.server.log_level,
        },
        "model": {
            "model_dirs": [
                str(d) for d in global_settings.model.get_model_dirs(global_settings.base_path)
            ],
            "model_dir": str(global_settings.model.get_model_dir(global_settings.base_path)),
            "max_model_memory": global_settings.model.max_model_memory,
        },
        "scheduler": {
            "max_num_seqs": global_settings.scheduler.max_num_seqs,
            "prefill_batch_size": global_settings.scheduler.prefill_batch_size,
            "completion_batch_size": global_settings.scheduler.completion_batch_size,
        },
        "cache": {
            "enabled": global_settings.cache.enabled,
            "ssd_cache_dir": cache_dir,
            # Resolve "auto" to actual value (10% of SSD capacity)
            "ssd_cache_max_size": _format_cache_size(
                global_settings.cache.get_ssd_cache_max_size_bytes(global_settings.base_path)
            ),
        },
        "mcp": {
            "config_path": global_settings.mcp.config_path,
        },
        "sampling": {
            "max_context_window": global_settings.sampling.max_context_window,
            "max_tokens": global_settings.sampling.max_tokens,
            "temperature": global_settings.sampling.temperature,
            "top_p": global_settings.sampling.top_p,
            "top_k": global_settings.sampling.top_k,
            "repetition_penalty": global_settings.sampling.repetition_penalty,
        },
        "auth": {
            "api_key_set": bool(global_settings.auth.api_key),
            "api_key": global_settings.auth.api_key or "",
        },
        "claude_code": {
            "context_scaling_enabled": global_settings.claude_code.context_scaling_enabled,
            "target_context_size": global_settings.claude_code.target_context_size,
        },
        "system": {
            "total_memory_bytes": memory_info["total_bytes"],
            "total_memory": memory_info["total_formatted"],
            "auto_model_memory": memory_info["auto_limit_formatted"],
            "ssd_total_bytes": disk_info["total_bytes"],
            "ssd_total": disk_info["total_formatted"],
            "ssd_free_bytes": disk_info["free_bytes"],
            "ssd_free": disk_info["free_formatted"],
        },
    }


@router.post("/api/global-settings")
async def update_global_settings(
    request: GlobalSettingsRequest,
    is_admin: bool = Depends(require_admin),
):
    """
    Update global server settings.

    Updates are persisted to the global settings file. Some settings
    (log_level, model_dir, max_model_memory, cache) are applied immediately,
    while others (host, port, scheduler, mcp) require server restart.

    Args:
        request: GlobalSettingsRequest with the new settings.

    Returns:
        JSON response with success status, message, and list of runtime-applied settings.

    Raises:
        HTTPException: 401 if not authenticated, 503 if server not initialized,
                      400 if validation fails.
    """
    from ..config import parse_size

    global_settings = _get_global_settings()

    if global_settings is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    # Track which settings were applied at runtime
    runtime_applied: List[str] = []

    # Apply server settings
    if request.host is not None:
        global_settings.server.host = request.host
    if request.port is not None:
        global_settings.server.port = request.port
    if request.log_level is not None:
        global_settings.server.log_level = request.log_level
        # Apply log level at runtime
        _apply_log_level_runtime(request.log_level)
        runtime_applied.append("log_level")

    # Apply model settings
    new_dirs = None
    if request.model_dirs is not None:
        new_dirs = [d for d in request.model_dirs if d.strip()]
    elif request.model_dir is not None:
        new_dirs = [request.model_dir]

    if new_dirs is not None:
        old_dirs = global_settings.model.model_dirs
        if new_dirs != old_dirs:
            success, msg = await _apply_model_dirs_runtime(new_dirs)
            if success:
                global_settings.model.model_dirs = new_dirs
                global_settings.model.model_dir = new_dirs[0] if new_dirs else None
                runtime_applied.append("model_dirs")
                logger.info(msg)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to change model directories: {msg}"
                )

    if request.max_model_memory is not None and request.max_model_memory != "":
        global_settings.model.max_model_memory = request.max_model_memory
        # Apply at runtime
        try:
            max_bytes = parse_size(request.max_model_memory)
            success, msg = await _apply_max_model_memory_runtime(max_bytes)
            if success:
                runtime_applied.append("max_model_memory")
                logger.info(msg)
            else:
                logger.warning(f"Failed to apply max_model_memory runtime: {msg}")
        except ValueError as e:
            logger.warning(f"Invalid max_model_memory format: {e}")

    # Apply scheduler settings (restart required)
    if request.max_num_seqs is not None:
        global_settings.scheduler.max_num_seqs = request.max_num_seqs
    if request.prefill_batch_size is not None:
        global_settings.scheduler.prefill_batch_size = request.prefill_batch_size
    if request.completion_batch_size is not None:
        global_settings.scheduler.completion_batch_size = request.completion_batch_size

    # Apply cache settings
    cache_changed = False
    if request.cache_enabled is not None:
        global_settings.cache.enabled = request.cache_enabled
        cache_changed = True
    if request.ssd_cache_dir is not None:
        global_settings.cache.ssd_cache_dir = request.ssd_cache_dir
        cache_changed = True
    if request.ssd_cache_max_size is not None:
        global_settings.cache.ssd_cache_max_size = request.ssd_cache_max_size
        cache_changed = True

    if cache_changed:
        success, msg = await _apply_cache_settings_runtime(
            request.cache_enabled,
            request.ssd_cache_dir,
            request.ssd_cache_max_size,
            global_settings,
        )
        if success:
            runtime_applied.append("cache")
            logger.info(msg)
        else:
            logger.warning(f"Failed to apply cache settings runtime: {msg}")

    # Apply MCP settings (restart required)
    if request.mcp_config is not None:
        global_settings.mcp.config_path = request.mcp_config if request.mcp_config else None

    # Apply sampling settings (Live - immediately applied)
    sampling_changed = False
    if request.sampling_max_context_window is not None:
        global_settings.sampling.max_context_window = request.sampling_max_context_window
        sampling_changed = True
    if request.sampling_max_tokens is not None:
        global_settings.sampling.max_tokens = request.sampling_max_tokens
        sampling_changed = True
    if request.sampling_temperature is not None:
        global_settings.sampling.temperature = request.sampling_temperature
        sampling_changed = True
    if request.sampling_top_p is not None:
        global_settings.sampling.top_p = request.sampling_top_p
        sampling_changed = True
    if request.sampling_top_k is not None:
        global_settings.sampling.top_k = request.sampling_top_k
        sampling_changed = True
    if request.sampling_repetition_penalty is not None:
        global_settings.sampling.repetition_penalty = request.sampling_repetition_penalty
        sampling_changed = True

    if sampling_changed:
        success, msg = _apply_sampling_settings_runtime(
            request.sampling_max_context_window,
            request.sampling_max_tokens,
            request.sampling_temperature,
            request.sampling_top_p,
            request.sampling_top_k,
            request.sampling_repetition_penalty,
        )
        if success:
            runtime_applied.append("sampling")
            logger.info(msg)

    # Apply Claude Code settings (Live - immediately applied)
    claude_code_changed = False
    if request.claude_code_context_scaling_enabled is not None:
        global_settings.claude_code.context_scaling_enabled = (
            request.claude_code_context_scaling_enabled
        )
        claude_code_changed = True
    if request.claude_code_target_context_size is not None:
        global_settings.claude_code.target_context_size = (
            request.claude_code_target_context_size
        )
        claude_code_changed = True

    if claude_code_changed:
        runtime_applied.append("claude_code")
        logger.info(
            f"Claude Code settings updated: "
            f"scaling={'enabled' if global_settings.claude_code.context_scaling_enabled else 'disabled'}, "
            f"target={global_settings.claude_code.target_context_size}"
        )

    # Apply auth settings (API key change)
    if request.api_key is not None:
        from ..server import _server_state

        is_valid, error_msg = validate_api_key(request.api_key)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        global_settings.auth.api_key = request.api_key
        _server_state.api_key = request.api_key
        runtime_applied.append("api_key")
        logger.info("API key updated via admin settings")

    # Validate settings
    errors = global_settings.validate()
    if errors:
        raise HTTPException(status_code=400, detail=errors)

    # Persist to file
    try:
        global_settings.save()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save settings: {e}")

    # Build response message
    message = "Settings saved successfully."

    return {
        "success": True,
        "message": message,
        "runtime_applied": runtime_applied,
    }


# =============================================================================
# Logs API Routes
# =============================================================================


def _tail_file(file_path: Path, num_lines: int) -> tuple[str, int]:
    """
    Read the last N lines of a file efficiently.

    Uses a deque to efficiently keep only the last N lines in memory.

    Args:
        file_path: Path to the log file.
        num_lines: Number of lines to return.

    Returns:
        Tuple of (content_string, total_line_count)
    """
    if not file_path.exists():
        return "", 0

    # Use deque for efficient tail operation
    lines = deque(maxlen=num_lines)
    total_lines = 0

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            lines.append(line)
            total_lines += 1

    return "".join(lines), total_lines


def _get_available_log_files(log_dir: Path) -> List[str]:
    """
    Get list of available log files sorted by modification time.

    Args:
        log_dir: Directory containing log files.

    Returns:
        List of log file names, newest first.
    """
    if not log_dir.exists():
        return []

    files = []
    for f in log_dir.iterdir():
        # Match server.log and server.log.YYYY-MM-DD patterns
        if f.name.startswith("server") and (f.suffix == ".log" or ".log." in f.name):
            files.append(f.name)

    # Sort by modification time (newest first)
    files.sort(key=lambda x: (log_dir / x).stat().st_mtime, reverse=True)
    return files


@router.get("/api/logs")
async def get_logs(
    lines: int = 100,
    file: Optional[str] = None,
    is_admin: bool = Depends(require_admin),
):
    """
    Get server logs.

    Returns the last N lines of the specified log file (or current log).
    Supports viewing historical rotated log files.

    Args:
        lines: Number of lines to return (default: 100, max: 10000).
        file: Optional specific log file name. If not specified, uses current log.

    Returns:
        JSON response with log content and metadata:
        - logs: The log content string
        - total_lines: Total number of lines in the file
        - log_file: Name of the log file being read
        - available_files: List of available log files

    Raises:
        HTTPException: 401 if not authenticated, 503 if server not initialized,
                      400 if invalid file name, 404 if log file not found.
    """
    global_settings = _get_global_settings()

    if global_settings is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    # Limit lines to prevent memory issues
    lines = min(max(1, lines), 10000)

    log_dir = global_settings.logging.get_log_dir(global_settings.base_path)

    # Get available log files
    available_files = _get_available_log_files(log_dir)

    # Determine which file to read
    if file:
        # Validate file name (prevent path traversal)
        if "/" in file or "\\" in file or ".." in file:
            raise HTTPException(status_code=400, detail="Invalid file name")
        log_file = log_dir / file
        if not log_file.exists():
            raise HTTPException(status_code=404, detail=f"Log file not found: {file}")
    else:
        # Default to current log file
        log_file = log_dir / "server.log"

    # Read log content
    if log_file.exists():
        content, total_lines = _tail_file(log_file, lines)
    else:
        content = ""
        total_lines = 0

    return {
        "logs": content,
        "total_lines": total_lines,
        "log_file": log_file.name,
        "available_files": available_files,
    }


# =============================================================================
# Stats API Routes
# =============================================================================


def _get_engine_info() -> dict:
    """Get commit SHA and GitHub URL for engine packages.

    Fallback chain:
    1. PEP 610 direct_url.json (pip install git+https://...)
    2. _engine_commits.json (generated by build.py for app bundle)
    3. Parse pyproject.toml at runtime (dev environment)
    """
    import importlib.metadata

    engines = {}
    packages = {
        "mlx-lm": "https://github.com/ml-explore/mlx-lm",
        "mlx-embeddings": "https://github.com/Blaizzy/mlx-embeddings",
    }

    fallback_commits = _load_fallback_commits(packages)

    for pkg_name, default_url in packages.items():
        info = {"name": pkg_name, "version": None, "commit": None, "url": None}
        try:
            dist = importlib.metadata.distribution(pkg_name)
            info["version"] = dist.version

            # Method 1: PEP 610 direct_url.json
            commit_info = _get_commit_from_direct_url(dist, default_url)
            if not commit_info:
                # Methods 2+3: _engine_commits.json or pyproject.toml
                commit_info = fallback_commits.get(pkg_name)

            if commit_info:
                info["commit"] = commit_info["commit"]
                info["url"] = commit_info["url"]
        except Exception:
            pass
        engines[pkg_name] = info

    return engines


def _get_commit_from_direct_url(dist, default_url: str) -> dict | None:
    """Extract commit SHA from PEP 610 direct_url.json."""
    import json

    try:
        direct_url_text = dist.read_text("direct_url.json")
        if direct_url_text:
            direct_url = json.loads(direct_url_text)
            vcs_info = direct_url.get("vcs_info", {})
            commit = vcs_info.get("commit_id")
            if commit:
                repo_url = direct_url.get("url", default_url).rstrip("/")
                if repo_url.endswith(".git"):
                    repo_url = repo_url[:-4]
                return {"commit": commit, "url": f"{repo_url}/commit/{commit}"}
    except Exception:
        pass
    return None


def _load_fallback_commits(packages: dict[str, str]) -> dict:
    """Load commit SHAs from fallback sources.

    Tries in order:
    1. _engine_commits.json (generated by build.py, lives in omlx package dir)
    2. pyproject.toml (dev environment, lives one level above package dir)
    """
    import json
    from pathlib import Path

    # This file is at omlx/admin/routes.py → package dir is omlx/
    pkg_dir = Path(__file__).resolve().parent.parent

    # Method 2: _engine_commits.json (written by build.py for app bundle)
    commits_file = pkg_dir / "_engine_commits.json"
    if commits_file.is_file():
        try:
            data = json.loads(commits_file.read_text())
            result = {}
            for pkg_name, entry in data.items():
                if isinstance(entry, dict) and "commit" in entry:
                    commit = entry["commit"]
                    repo_url = entry.get("url", packages.get(pkg_name, ""))
                    if "/commit/" not in repo_url:
                        repo_url = f"{repo_url}/commit/{commit}"
                    result[pkg_name] = {"commit": commit, "url": repo_url}
            if result:
                return result
        except Exception:
            pass

    # Method 3: Parse pyproject.toml (dev environment)
    pyproject = pkg_dir.parent / "pyproject.toml"
    if pyproject.is_file():
        try:
            return _parse_commits_from_pyproject(pyproject, packages)
        except Exception:
            pass

    return {}


def _parse_commits_from_pyproject(
    pyproject_path, packages: dict[str, str]
) -> dict:
    """Extract commit SHAs from git+https:// URLs in pyproject.toml."""
    import re
    from pathlib import Path

    content = Path(pyproject_path).read_text()
    commits = {}
    # Match: "mlx-lm @ git+https://github.com/.../mlx-lm@<sha>"
    pattern = r'"(\S+)\s*@\s*git\+https://[^@"]+@([0-9a-f]{7,40})"'
    for match in re.finditer(pattern, content):
        pkg_name = match.group(1).strip().lower()
        sha = match.group(2)
        if pkg_name in packages:
            repo_url = packages[pkg_name]
            commits[pkg_name] = {
                "commit": sha,
                "url": f"{repo_url}/commit/{sha}",
            }
    return commits


@router.get("/api/stats")
async def get_server_stats(
    model: str = "",
    is_admin: bool = Depends(require_admin),
):
    """Get server serving stats for the Status dashboard."""
    from ..server_metrics import get_server_metrics

    metrics = get_server_metrics()
    snapshot = metrics.get_snapshot(model_id=model)

    global_settings = _get_global_settings()
    port = global_settings.server.port if global_settings else 8000
    api_key = global_settings.auth.api_key if global_settings else ""

    return {
        **snapshot,
        "port": port,
        "api_key": api_key or "",
        "claude_code_context_scaling_enabled": (
            global_settings.claude_code.context_scaling_enabled
            if global_settings
            else False
        ),
        "claude_code_target_context_size": (
            global_settings.claude_code.target_context_size
            if global_settings
            else 200000
        ),
        "engines": _get_engine_info(),
    }


@router.post("/api/stats/clear")
async def clear_server_stats(is_admin: bool = Depends(require_admin)):
    """Clear all server metrics."""
    from ..server_metrics import get_server_metrics

    get_server_metrics().clear_metrics()
    return {"status": "ok"}


# =============================================================================
# HuggingFace Downloader API Routes
# =============================================================================


@router.post("/api/hf/download")
async def start_hf_download(
    request: HFDownloadRequest,
    is_admin: bool = Depends(require_admin),
):
    """Start downloading a model from HuggingFace."""
    if _hf_downloader is None:
        raise HTTPException(status_code=503, detail="Downloader not initialized")

    try:
        task = await _hf_downloader.start_download(
            request.repo_id, request.hf_token
        )
        return {"success": True, "task": task.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/api/hf/tasks")
async def list_hf_tasks(is_admin: bool = Depends(require_admin)):
    """List all download tasks."""
    if _hf_downloader is None:
        raise HTTPException(status_code=503, detail="Downloader not initialized")

    return {"tasks": _hf_downloader.get_tasks()}


@router.post("/api/hf/cancel/{task_id}")
async def cancel_hf_download(
    task_id: str,
    is_admin: bool = Depends(require_admin),
):
    """Cancel an active download."""
    if _hf_downloader is None:
        raise HTTPException(status_code=503, detail="Downloader not initialized")

    success = await _hf_downloader.cancel_download(task_id)
    if not success:
        raise HTTPException(
            status_code=404, detail="Task not found or not cancellable"
        )
    return {"success": True}


@router.delete("/api/hf/task/{task_id}")
async def remove_hf_task(
    task_id: str,
    is_admin: bool = Depends(require_admin),
):
    """Remove a completed, failed, or cancelled task."""
    if _hf_downloader is None:
        raise HTTPException(status_code=503, detail="Downloader not initialized")

    success = _hf_downloader.remove_task(task_id)
    if not success:
        raise HTTPException(
            status_code=404, detail="Task not found or still active"
        )
    return {"success": True}


@router.get("/api/hf/recommended")
async def get_recommended_models(is_admin: bool = Depends(require_admin)):
    """Get recommended mlx-community models filtered by system memory."""
    if _hf_downloader is None:
        raise HTTPException(status_code=503, detail="Downloader not initialized")

    memory_info = get_system_memory_info()
    max_memory = memory_info["total_bytes"] or 16 * 1024**3

    from .hf_downloader import HFDownloader

    result = await HFDownloader.get_recommended_models(max_memory_bytes=max_memory)
    return result


@router.get("/api/hf/models")
async def list_hf_models(is_admin: bool = Depends(require_admin)):
    """List models in all model directories with disk size info."""
    global_settings = _get_global_settings()
    if global_settings is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    model_dirs = global_settings.model.get_model_dirs(global_settings.base_path)

    models = []
    seen_names: set[str] = set()
    for model_dir in model_dirs:
        if not model_dir.exists():
            continue
        for subdir in sorted(model_dir.iterdir()):
            if not subdir.is_dir() or subdir.name.startswith("."):
                continue

            if (subdir / "config.json").exists():
                # Level 1: direct model folder
                if subdir.name in seen_names:
                    continue
                seen_names.add(subdir.name)
                total_size = sum(
                    f.stat().st_size for f in subdir.rglob("*") if f.is_file()
                )
                models.append(
                    {
                        "name": subdir.name,
                        "path": str(subdir),
                        "size": total_size,
                        "size_formatted": format_size(total_size),
                    }
                )
            else:
                # Level 2: organization folder — scan children
                for child in sorted(subdir.iterdir()):
                    if not child.is_dir() or child.name.startswith("."):
                        continue
                    if not (child / "config.json").exists():
                        continue
                    if child.name in seen_names:
                        continue
                    seen_names.add(child.name)

                    total_size = sum(
                        f.stat().st_size
                        for f in child.rglob("*")
                        if f.is_file()
                    )
                    models.append(
                        {
                            "name": child.name,
                            "path": str(child),
                            "size": total_size,
                            "size_formatted": format_size(total_size),
                        }
                    )

    return {"models": models}


@router.delete("/api/hf/models/{model_name}")
async def delete_hf_model(
    model_name: str,
    is_admin: bool = Depends(require_admin),
):
    """Delete a downloaded model from disk and refresh the model pool."""
    global_settings = _get_global_settings()
    engine_pool = _get_engine_pool()

    if global_settings is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    model_dirs = global_settings.model.get_model_dirs(global_settings.base_path)

    # Search for model across all directories in both flat and org-folder layouts
    model_path = None
    parent_model_dir = None
    for model_dir in model_dirs:
        if not model_dir.exists():
            continue
        candidate = model_dir / model_name
        if candidate.is_dir() and (candidate / "config.json").exists():
            model_path = candidate
            parent_model_dir = model_dir
            break
        # Try two-level: search inside organization folders
        for subdir in model_dir.iterdir():
            if not subdir.is_dir() or subdir.name.startswith("."):
                continue
            candidate = subdir / model_name
            if candidate.is_dir() and (candidate / "config.json").exists():
                model_path = candidate
                parent_model_dir = model_dir
                break
        if model_path is not None:
            break

    if model_path is None:
        raise HTTPException(status_code=404, detail="Model not found")

    # Validate path traversal against parent model directory
    try:
        if not model_path.resolve().is_relative_to(parent_model_dir.resolve()):
            raise HTTPException(status_code=400, detail="Invalid model name")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid model name")

    if not model_path.is_dir():
        raise HTTPException(status_code=400, detail="Not a model directory")

    # Unload model if loaded
    if engine_pool is not None:
        loaded_ids = engine_pool.get_loaded_model_ids()
        if model_name in loaded_ids:
            try:
                await engine_pool._unload_engine(model_name)
                logger.info(f"Unloaded model '{model_name}' before deletion")
            except Exception as e:
                logger.warning(f"Failed to unload model '{model_name}': {e}")

    # Delete from disk
    shutil.rmtree(model_path)
    logger.info(f"Deleted model directory: {model_path}")

    # Re-discover models
    if engine_pool is not None:
        settings_manager = _get_settings_manager()
        pinned_models = []
        if settings_manager:
            pinned_models = settings_manager.get_pinned_model_ids()

        engine_pool._entries.pop(model_name, None)
        engine_pool.discover_models(
            [str(d) for d in model_dirs], pinned_models
        )
        logger.info("Model pool refreshed after deletion")

    return {"success": True, "message": f"Model '{model_name}' deleted"}


# =============================================================================
# Benchmark API Routes
# =============================================================================


@router.post("/api/bench/start")
async def start_benchmark(
    request: Request,
    is_admin: bool = Depends(require_admin),
):
    """Start a benchmark run.

    Validates the model, creates a benchmark run, and starts it
    as an asyncio background task.
    """
    from .benchmark import (
        BenchmarkRequest,
        cleanup_old_runs,
        create_run,
        run_benchmark,
    )

    engine_pool = _get_engine_pool()
    if engine_pool is None:
        raise HTTPException(status_code=503, detail="Engine pool not initialized")

    body = await request.json()
    try:
        bench_request = BenchmarkRequest(**body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Validate model exists and is an LLM
    entry = engine_pool.get_entry(bench_request.model_id)
    if entry is None:
        raise HTTPException(
            status_code=404, detail=f"Model not found: {bench_request.model_id}"
        )
    if entry.model_type not in ("llm", None):
        raise HTTPException(
            status_code=400,
            detail=f"Model {bench_request.model_id} is not an LLM (type: {entry.model_type})",
        )

    # Cleanup old runs
    cleanup_old_runs()

    # Create and start the benchmark
    run = create_run(bench_request)
    total_tests = len(bench_request.prompt_lengths) + len(bench_request.batch_sizes) * 2

    run.task = asyncio.create_task(run_benchmark(run, engine_pool))

    logger.info(
        f"Benchmark started: {run.bench_id} model={bench_request.model_id} "
        f"tests={total_tests}"
    )

    return {
        "bench_id": run.bench_id,
        "status": "started",
        "total_tests": total_tests,
    }


@router.get("/api/bench/{bench_id}/stream")
async def stream_benchmark(
    bench_id: str,
    is_admin: bool = Depends(require_admin),
):
    """Stream benchmark progress via Server-Sent Events."""
    import json

    from fastapi.responses import StreamingResponse

    from .benchmark import get_run

    run = get_run(bench_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Benchmark not found: {bench_id}")

    async def event_generator():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(run.queue.get(), timeout=60.0)
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield ": keepalive\n\n"
                    continue

                yield f"data: {json.dumps(event)}\n\n"

                # Stop streaming on terminal events
                if event.get("type") in ("done", "error"):
                    break
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/api/bench/{bench_id}/cancel")
async def cancel_benchmark(
    bench_id: str,
    is_admin: bool = Depends(require_admin),
):
    """Cancel a running benchmark."""
    from .benchmark import get_run

    run = get_run(bench_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Benchmark not found: {bench_id}")

    if run.status != "running":
        raise HTTPException(
            status_code=400,
            detail=f"Benchmark is not running (status: {run.status})",
        )

    if run.task and not run.task.done():
        run.task.cancel()

    return {"status": "cancelled", "bench_id": bench_id}


@router.get("/api/bench/{bench_id}/results")
async def get_benchmark_results(
    bench_id: str,
    is_admin: bool = Depends(require_admin),
):
    """Get results from a completed benchmark."""
    from .benchmark import get_run

    run = get_run(bench_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Benchmark not found: {bench_id}")

    return {
        "bench_id": run.bench_id,
        "status": run.status,
        "results": run.results,
        "error": run.error_message if run.error_message else None,
    }
