# SPDX-License-Identifier: Apache-2.0
"""Admin panel routes for oMLX server configuration.

This module provides HTTP routes for the admin panel including:
- Login/logout with API key authentication
- Dashboard for server monitoring
- Model settings management (per-model sampling parameters, pinning, default)
- Global settings management
"""

import asyncio
import inspect
import json
import logging
import os
import re
import secrets
import shutil
import signal
import sys
import time
from collections import deque
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import requests
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from ..model_profiles import EXCLUDED_FROM_PROFILES
from ..settings import SubKeyEntry
from ..utils.release_check import select_latest_stable_release
from .auth import (
    REMEMBER_ME_MAX_AGE,
    SESSION_MAX_AGE,
    create_session_token,
    require_admin,
    validate_api_key,
    verify_api_key,
    verify_session,
)

logger = logging.getLogger(__name__)

PRESET_REMOTE_URL = "https://omlx.ai/assets/omlx_preset.json"


# =============================================================================
# Pydantic Models
# =============================================================================


class LoginRequest(BaseModel):
    """Request model for admin login."""

    api_key: str
    remember: bool = False


class SetupApiKeyRequest(BaseModel):
    """Request model for initial API key setup."""

    api_key: str
    api_key_confirm: str


class CreateSubKeyRequest(BaseModel):
    """Request model for creating a sub API key."""

    key: str
    name: str = ""


class DeleteSubKeyRequest(BaseModel):
    """Request model for deleting a sub API key."""

    key: str


class CacheProbeRequest(BaseModel):
    """Request model for probing per-prompt cache state.

    Tokenizes a chat message list with the target model's tokenizer, then
    classifies each block's location in the cache hierarchy:
    - Hot SSD (in-RAM copy of SSD cache, ready to mount without disk read)
    - Disk SSD (persisted only, needs disk read to reuse)
    - Cold (fully uncached — would require full prefill)
    """

    model_id: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    chat_template_kwargs: dict[str, Any] | None = None


class ModelSettingsRequest(BaseModel):
    """Request model for updating per-model settings."""

    model_alias: str | None = None
    model_type_override: str | None = None
    max_context_window: int | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    force_sampling: bool | None = None
    max_tool_result_tokens: int | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    forced_ct_kwargs: list[str] | None = None
    ttl_seconds: int | None = None
    index_cache_freq: int | None = None
    enable_thinking: bool | None = None
    thinking_budget_enabled: bool | None = None
    thinking_budget_tokens: int | None = None
    # TurboQuant KV cache (mlx-vlm backend)
    turboquant_kv_enabled: bool | None = None
    turboquant_kv_bits: float | None = None
    # SpecPrefill (experimental)
    specprefill_enabled: bool | None = None
    specprefill_draft_model: str | None = None
    specprefill_keep_pct: float | None = None
    specprefill_threshold: int | None = None
    # DFlash (block diffusion speculative decoding)
    dflash_enabled: bool | None = None
    dflash_draft_model: str | None = None
    dflash_draft_quant_enabled: bool | None = None
    dflash_draft_quant_weight_bits: int | None = None
    dflash_draft_quant_activation_bits: int | None = None
    dflash_draft_quant_group_size: int | None = None
    dflash_max_ctx: int | None = None
    dflash_in_memory_cache: bool | None = None
    dflash_in_memory_cache_max_entries: int | None = None
    dflash_in_memory_cache_max_bytes: int | None = None
    dflash_ssd_cache: bool | None = None
    # Native MTP (mlx-lm PR 990 / PR 15 monkey-patch)
    mtp_enabled: bool | None = None
    # VLM MTP speculative decoding via external assistant drafter (mlx-vlm 191d7c8+)
    vlm_mtp_enabled: bool | None = None
    vlm_mtp_draft_model: str | None = None
    vlm_mtp_draft_block_size: int | None = None
    reasoning_parser: str | None = None
    is_pinned: bool | None = None
    is_default: bool | None = None
    # Security: per-model opt-in for trust_remote_code (issue #926)
    trust_remote_code: bool | None = None


class CreateProfileRequest(BaseModel):
    """Request body for creating a per-model profile."""
    name: str
    display_name: str
    description: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)
    also_save_as_template: bool = False
    source_template: str | None = None


class UpdateProfileRequest(BaseModel):
    """Request body for updating/renaming a per-model profile."""
    new_name: str | None = None
    display_name: str | None = None
    description: str | None = None
    settings: dict[str, Any] | None = None
    source_template: str | None = None
    also_save_as_template: bool = False


class CreateTemplateRequest(BaseModel):
    """Request body for creating a global template."""
    name: str
    display_name: str
    description: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)


class UpdateTemplateRequest(BaseModel):
    """Request body for updating/renaming a global template."""
    new_name: str | None = None
    display_name: str | None = None
    description: str | None = None
    settings: dict[str, Any] | None = None


class GlobalSettingsRequest(BaseModel):
    """Request model for updating global server settings."""

    # Server settings
    host: str | None = None
    port: int | None = None
    log_level: str | None = None
    server_aliases: list[str] | None = None
    sse_keepalive_mode: str | None = None

    # Model settings
    model_dirs: list[str] | None = None
    model_dir: str | None = None  # Deprecated: kept for backward compatibility
    max_model_memory: str | None = None
    model_fallback: bool | None = None

    # Memory enforcement
    max_process_memory: str | None = None  # "auto", "disabled", or "XX%"
    memory_prefill_memory_guard: bool | None = None

    # Scheduler settings
    max_concurrent_requests: int | None = None
    chunked_prefill: bool | None = None

    # Cache settings
    cache_enabled: bool | None = None
    ssd_cache_dir: str | None = None
    ssd_cache_max_size: str | None = None
    hot_cache_only: bool | None = None
    hot_cache_max_size: str | None = None  # "0" = disabled, "8GB", etc.
    initial_cache_blocks: int | None = None  # Starting blocks (requires restart)

    # MCP settings
    mcp_config: str | None = None

    # HuggingFace settings
    hf_endpoint: str | None = None

    # ModelScope settings
    ms_endpoint: str | None = None

    # Network settings
    network_http_proxy: str | None = None
    network_https_proxy: str | None = None
    network_no_proxy: str | None = None
    network_ca_bundle: str | None = None

    # Sampling defaults
    sampling_max_context_window: int | None = None
    sampling_max_tokens: int | None = None
    sampling_temperature: float | None = None
    sampling_top_p: float | None = None
    sampling_top_k: int | None = None
    sampling_repetition_penalty: float | None = None

    # Claude Code settings
    claude_code_context_scaling_enabled: bool | None = None
    claude_code_target_context_size: int | None = None
    claude_code_mode: str | None = None
    claude_code_opus_model: str | None = None
    claude_code_sonnet_model: str | None = None
    claude_code_haiku_model: str | None = None

    # Other integrations settings
    integrations_copilot_model: str | None = None
    integrations_codex_model: str | None = None
    integrations_opencode_model: str | None = None
    integrations_openclaw_model: str | None = None
    integrations_hermes_model: str | None = None
    integrations_pi_model: str | None = None
    integrations_openclaw_tools_profile: Literal["minimal", "coding", "messaging", "full"] | None = None

    # UI settings
    ui_language: str | None = None

    # Idle timeout settings. null disables the global fallback.
    idle_timeout_seconds: int | None = Field(default=None, ge=60)

    # Auth settings
    api_key: str | None = None
    skip_api_key_verification: bool | None = None


class HFDownloadRequest(BaseModel):
    """Request model for starting a HuggingFace model download."""

    repo_id: str
    hf_token: str = ""


class HFRetryRequest(BaseModel):
    """Request model for retrying a HuggingFace model download."""

    hf_token: str = ""


class MSDownloadRequest(BaseModel):
    """Request model for starting a ModelScope model download."""

    model_id: str
    ms_token: str = ""


class MSRetryRequest(BaseModel):
    """Request model for retrying a ModelScope model download."""

    ms_token: str = ""


class OQStartRequest(BaseModel):
    """Request model for starting an oQ quantization task."""

    model_path: str
    oq_level: float
    group_size: int = 64
    sensitivity_model_path: str = ""
    text_only: bool = False
    dtype: str = "bfloat16"
    preserve_mtp: bool = False
    auto_proxy_sensitivity: bool = True


class HFUploadRequest(BaseModel):
    """Request model for starting a HuggingFace upload task."""

    model_path: str
    repo_id: str
    hf_token: str
    readme_source_path: str = ""
    auto_readme: bool = True
    redownload_notice: bool = False
    private: bool = False


class HFValidateTokenRequest(BaseModel):
    """Request model for validating a HuggingFace token."""

    hf_token: str


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


_PAROQUANT_REASON = (
    "Not supported on paroquant models yet (compatibility not verified)"
)


def _paroquant_compat_for_model(model_info: dict) -> tuple[bool, str]:
    """Detect whether a model is paroquant-quantized.

    Returns ``(is_paroquant, reason)``. ``is_paroquant`` is True iff
    ``config.json`` declares ``quantization_config.quant_method == "paroquant"``.
    Reason is the user-facing string surfaced as a tooltip/banner on the
    admin model settings modal when paroquant gates an experimental toggle.
    """
    import json
    from pathlib import Path

    model_path = model_info.get("model_path") or ""
    if not model_path:
        return False, ""
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.exists():
        return False, ""
    try:
        cfg = json.loads(cfg_path.read_text())
    except Exception:
        return False, ""
    qcfg = cfg.get("quantization_config") or {}
    method = (qcfg.get("quant_method") or "").lower()
    if method == "paroquant":
        return True, _PAROQUANT_REASON
    return False, ""


def _dflash_compat_for_model(model_info: dict) -> tuple[bool, str]:
    """Resolve dflash compatibility for an engine_pool model dict.

    Returns ``(False, "")`` when dflash-mlx is not installed so the UI hides
    the compat hint instead of pointing the user at an unrelated reason.
    """
    is_paro, paro_reason = _paroquant_compat_for_model(model_info)
    if is_paro:
        return False, paro_reason
    try:
        from ..engine.dflash import is_dflash_compatible
    except ImportError:
        return False, ""
    model_path = model_info.get("model_path") or ""
    if not model_path:
        return False, "model_path missing"
    return is_dflash_compatible(model_path)


def _mtp_compat_for_model(model_info: dict) -> tuple[bool, str]:
    """Mirror of ``_dflash_compat_for_model`` for the native MTP toggle.

    Returns ``(compatible, reason)``. Reason is empty on success and
    suitable for surfacing to users (admin UI shows it under the toggle).

    The check is conservative: even when the config declares MTP layers
    we also peek at the safetensors weight index to verify that the
    converter actually preserved the ``mtp.*`` tensors. Default mlx-lm
    converters strip them; PR 990 ships a separate path that keeps them.
    """
    import json
    from pathlib import Path

    from ..utils.model_loading import _has_mtp_heads, _is_mtp_compatible

    is_paro, paro_reason = _paroquant_compat_for_model(model_info)
    if is_paro:
        return False, paro_reason

    model_path = model_info.get("model_path") or ""
    if not model_path:
        return False, "model_path missing"
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.exists():
        return False, "config.json not found"
    try:
        cfg = json.loads(cfg_path.read_text())
    except Exception as e:
        return False, f"failed to read config: {e}"
    model_type = cfg.get("model_type")
    if not _has_mtp_heads(cfg):
        return False, "model has no MTP heads in config"
    if not _is_mtp_compatible(cfg, model_type):
        return False, (
            f"model_type={model_type!r} is not on the MTP whitelist "
            "(supported: qwen3_5*, qwen3_6*, deepseek_v4*)"
        )
    if not _model_has_mtp_weight_tensors(Path(model_path)):
        return False, (
            "Config declares MTP layers but the converted weights are missing "
            "mtp.* tensors. Re-convert from HF with a converter that preserves "
            "MTP weights."
        )
    return True, ""


def _model_has_mtp_weight_tensors(model_dir) -> bool:
    """Return True iff the model directory's weight files contain ``mtp.*`` keys.

    Uses ``model.safetensors.index.json`` when present (cheap — only reads
    the weight_map). Falls back to opening each ``*.safetensors`` and
    checking its keys when no index is present (single-shard models).
    Returns False on any error (we treat the model as incompatible rather
    than risking a confusing load failure mid-inference).
    """
    import json
    from pathlib import Path

    try:
        from safetensors import safe_open
    except ImportError:
        # Library should be installed via mlx-lm deps; if it's not we can't
        # peek the weights. Stay conservative and assume incompatible.
        return False

    model_dir = Path(model_dir)

    # Preferred path: read the index file's weight_map (no tensor data loaded).
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text())
            weight_map = index.get("weight_map", {})
            return any("mtp." in key for key in weight_map.keys())
        except Exception:
            return False

    # Single-shard fallback: enumerate keys via safe_open metadata. We
    # short-circuit on the first ``mtp.*`` key.
    for path in model_dir.glob("*.safetensors"):
        try:
            with safe_open(str(path), framework="numpy") as f:  # type: ignore[arg-type]
                for key in f.keys():
                    if "mtp." in key:
                        return True
        except Exception:
            continue
    return False


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

    # Update downloader model directories
    global _hf_downloader, _ms_downloader
    if model_dirs:
        primary_dir = model_dirs[0]
        if _hf_downloader is not None:
            _hf_downloader.update_model_dir(primary_dir)
        if _ms_downloader is not None:
            _ms_downloader.update_model_dir(primary_dir)

    # Re-discover models from new directories
    try:
        pool.discover_models(model_dirs, pinned_models)
        if _server_state.settings_manager is not None:
            pool.apply_settings_overrides(_server_state.settings_manager)
    except Exception as e:
        return False, f"Failed to discover models: {e}"

    dir_count = len(model_dirs)
    return True, (
        f"Re-discovered {pool.model_count} models "
        f"from {dir_count} director{'ies' if dir_count > 1 else 'y'}"
    )


async def _reload_models() -> tuple[bool, str]:
    """
    Reload models: re-read model_settings.json, re-scan dirs, re-apply overrides,
    and preload pinned models.

    This does NOT re-read settings.json (global settings). It only refreshes
    the model inventory and per-model settings.

    Returns:
        Tuple of (success, message)
    """
    from ..server import _server_state

    if _server_state.engine_pool is None:
        return False, "Engine pool not initialized"

    global_settings = _get_global_settings()
    if global_settings is None:
        return False, "Global settings not initialized"

    # Re-read model_settings.json from disk
    settings_manager = _get_settings_manager()
    if settings_manager is not None:
        settings_manager._load()

    # Get current model_dirs from global settings
    model_dirs = global_settings.model.model_dirs or []
    if not model_dirs and global_settings.model.model_dir:
        model_dirs = [global_settings.model.model_dir]

    # Unload all, re-discover, re-apply overrides
    success, msg = await _apply_model_dirs_runtime(model_dirs)
    if not success:
        return False, msg

    # Preload pinned models
    pool = _server_state.engine_pool
    if pool is not None:
        await pool.preload_pinned_models()

    return True, msg


async def _apply_max_model_memory_runtime(
    max_memory_bytes: int | None,
) -> tuple[bool, str]:
    """
    Apply max model memory change at runtime.

    If current usage exceeds new limit, unloads LRU models until within limit.
    If None, disables model memory limiting.

    Returns:
        Tuple of (success, message)
    """
    from ..model_discovery import format_size
    from ..server import _server_state

    if _server_state.engine_pool is None:
        return False, "Engine pool not initialized"

    pool = _server_state.engine_pool
    old_limit = pool._max_model_memory
    pool._max_model_memory = max_memory_bytes

    old_display = format_size(old_limit) if old_limit is not None else "disabled"

    if max_memory_bytes is None:
        msg = f"Max model memory changed: {old_display} -> disabled (no limit)"
        return True, msg

    # If current usage exceeds new limit, unload LRU models
    unloaded = []
    while pool._current_model_memory > max_memory_bytes:
        victim = pool._find_lru_victim()
        if not victim:
            # All models are pinned, can't free more memory
            break
        await pool._unload_engine(victim)
        unloaded.append(victim)

    msg = f"Max model memory changed: {old_display} -> {format_size(max_memory_bytes)}"
    if unloaded:
        msg += f", unloaded: {', '.join(unloaded)}"

    return True, msg


async def _apply_max_process_memory_runtime(
    max_process_memory: str,
) -> tuple[bool, str]:
    """
    Apply max process memory change at runtime.

    Starts, stops, or updates the ProcessMemoryEnforcer based on the new value.

    Returns:
        Tuple of (success, message)
    """
    from ..server import _server_state
    from ..settings import get_system_memory

    if max_process_memory.lower() == "disabled":
        # Stop enforcer if running
        if _server_state.process_memory_enforcer is not None:
            await _server_state.process_memory_enforcer.stop()
            _server_state.process_memory_enforcer = None
            if _server_state.engine_pool is not None:
                _server_state.engine_pool._process_memory_enforcer = None
        return True, "Process memory enforcement disabled"

    # Calculate max bytes
    value = max_process_memory.strip().lower()
    if value == "auto":
        from ..settings import _adaptive_system_reserve

        total = get_system_memory()
        reserve = _adaptive_system_reserve(total)
        max_bytes = total - reserve
    else:
        percent_str = value.rstrip("%")
        try:
            percent = int(percent_str)
            max_bytes = int(get_system_memory() * percent / 100)
        except ValueError:
            from ..config import parse_size
            max_bytes = parse_size(max_process_memory)

    if _server_state.process_memory_enforcer is not None:
        # Update existing enforcer's limit
        _server_state.process_memory_enforcer.max_bytes = max_bytes
        # Trigger immediate check
        await _server_state.process_memory_enforcer._check_and_enforce()
        return True, (
            f"Process memory limit updated to "
            f"{max_bytes / 1024**3:.1f}GB"
        )
    else:
        # Create and start new enforcer
        if _server_state.engine_pool is None:
            return False, "Engine pool not initialized"
        from ..process_memory_enforcer import ProcessMemoryEnforcer

        enforcer = ProcessMemoryEnforcer(
            engine_pool=_server_state.engine_pool,
            max_bytes=max_bytes,
        )
        _server_state.process_memory_enforcer = enforcer
        _server_state.engine_pool._process_memory_enforcer = enforcer
        enforcer.start()
        return True, (
            f"Process memory enforcement enabled at "
            f"{max_bytes / 1024**3:.1f}GB"
        )


async def _apply_cache_settings_runtime(
    enabled: bool | None,
    ssd_cache_dir: str | None,
    ssd_cache_max_size: str | None,
    global_settings,
    hot_cache_max_size: str | None = None,
) -> tuple[bool, str]:
    """
    Apply cache settings at runtime.

    Updates the scheduler_config and unloads all models so they
    will use the new cache settings when reloaded.

    Returns:
        Tuple of (success, message)
    """
    from ..config import parse_size
    from ..server import _server_state

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

    # Apply hot cache max size
    if hot_cache_max_size is not None:
        hot_bytes = 0 if hot_cache_max_size == "0" else parse_size(hot_cache_max_size)
        old_hot = pool._scheduler_config.hot_cache_max_size
        pool._scheduler_config.hot_cache_max_size = hot_bytes
        if hot_bytes != old_hot:
            from ..utils.formatting import format_bytes
            old_str = "Off" if old_hot == 0 else format_bytes(old_hot)
            new_str = "Off" if hot_bytes == 0 else format_bytes(hot_bytes)
            logger.info(f"Hot cache max size changed: {old_str} -> {new_str}")
    elif global_settings.cache.hot_cache_max_size:
        pool._scheduler_config.hot_cache_max_size = (
            global_settings.cache.get_hot_cache_max_size_bytes()
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
    max_context_window: int | None,
    max_tokens: int | None,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    repetition_penalty: float | None = None,
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

from omlx._version import __version__ as _omlx_version

templates.env.globals["version"] = _omlx_version

# i18n defaults (English) — overridden once set_admin_getters is called
_i18n_dir = Path(__file__).parent / "i18n"
_en_locale: dict = {}
try:
    _en_locale = json.loads((_i18n_dir / "en.json").read_text(encoding="utf-8"))
except Exception:
    pass
templates.env.globals["t"] = lambda key: _en_locale.get(key, key)
templates.env.globals["locale_json"] = json.dumps(_en_locale, ensure_ascii=False)
templates.env.globals["current_lang"] = "en"


def _load_locale(language: str) -> dict:
    """Load locale dict for a given language code. Falls back to en on error."""
    path = _i18n_dir / f"{language}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            return json.loads((_i18n_dir / "en.json").read_text(encoding="utf-8"))
        except Exception:
            return {}


def _make_t(locale: dict):
    """Return a Jinja2-compatible t() function for the given locale dict."""

    def t(key: str) -> str:
        return locale.get(key, key)

    return t


def _refresh_i18n_globals() -> None:
    """Reload i18n globals from current settings. Called on startup and language change."""
    lang = "en"
    try:
        settings = _get_global_settings() if _get_global_settings else None
        if settings:
            lang = settings.ui.language
    except Exception:
        pass
    locale = _load_locale(lang)
    templates.env.globals["t"] = _make_t(locale)
    templates.env.globals["locale_json"] = json.dumps(locale, ensure_ascii=False)
    templates.env.globals["current_lang"] = lang


# =============================================================================
# State Getters (set by server.py)
# =============================================================================

_get_server_state = None
_get_engine_pool = None
_get_settings_manager = None
_get_global_settings = None
_hf_downloader = None
_ms_downloader = None
_oq_manager = None
_hf_uploader = None


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
    _refresh_i18n_globals()


def set_hf_downloader(downloader):
    """Set the HFDownloader instance for admin routes.

    Args:
        downloader: HFDownloader instance created during server initialization.
    """
    global _hf_downloader
    _hf_downloader = downloader


def set_ms_downloader(downloader):
    """Set the MSDownloader instance for admin routes.

    Args:
        downloader: MSDownloader instance created during server initialization.
    """
    global _ms_downloader
    _ms_downloader = downloader


def set_oq_manager(manager):
    """Set the OQManager instance for admin routes.

    Args:
        manager: OQManager instance created during server initialization.
    """
    global _oq_manager
    _oq_manager = manager


def set_hf_uploader(uploader):
    """Set the HFUploader instance for admin routes.

    Args:
        uploader: HFUploader instance created during server initialization.
    """
    global _hf_uploader
    _hf_uploader = uploader


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
        Dictionary with total_bytes, total_formatted.
    """
    try:
        check_path = Path(cache_dir).expanduser().resolve()
        while not check_path.exists() and check_path.parent != check_path:
            check_path = check_path.parent
        stat = shutil.disk_usage(check_path)
        return {
            "total_bytes": stat.total,
            "total_formatted": format_size(stat.total),
        }
    except Exception as e:
        logger.warning(f"Failed to get disk info for {cache_dir}: {e}")
        return {
            "total_bytes": 0,
            "total_formatted": "Unknown",
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
    # Redirect to dashboard if already authenticated
    from .auth import verify_session

    if verify_session(request):
        return RedirectResponse(url="/admin/dashboard", status_code=302)

    global_settings = _get_global_settings()

    # Skip login page when skip_api_key_verification is enabled
    if (
        global_settings is not None
        and global_settings.auth.skip_api_key_verification
    ):
        return RedirectResponse(url="/admin/dashboard", status_code=302)

    api_key_configured = bool(global_settings and global_settings.auth.api_key)
    return templates.TemplateResponse(
        request,
        "login.html",
        {"api_key_configured": api_key_configured},
    )


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request, is_admin: bool = Depends(require_admin)):
    """
    Render the admin dashboard page.

    Requires admin authentication via session cookie.

    Returns:
        HTML dashboard page with server status and model list.
    """
    return templates.TemplateResponse(request, "dashboard.html", {})


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
        request, "chat.html", {"api_key": api_key or ""}
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

    # Main key only — sub keys must not grant admin login
    if not verify_api_key(request.api_key, server_api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )

    # Create session token and set cookie
    token = create_session_token(remember=request.remember)
    cookie_max_age = REMEMBER_ME_MAX_AGE if request.remember else SESSION_MAX_AGE
    response.set_cookie(
        key="omlx_admin_session",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=cookie_max_age,
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

    # Main key only — sub keys must not grant admin login
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
# Sub Key Management Routes
# =============================================================================


@router.post("/api/sub-keys")
async def create_sub_key(
    request: CreateSubKeyRequest, is_admin: bool = Depends(require_admin)
):
    """Create a new sub API key.

    Sub keys can only be used for API authentication, not admin login.

    Args:
        request: CreateSubKeyRequest with key and optional name.

    Returns:
        JSON with the created sub key entry.

    Raises:
        HTTPException: 400 if validation fails or key already exists.
    """
    global_settings = _get_global_settings()
    if global_settings is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    # Validate key format
    is_valid, error_msg = validate_api_key(request.key)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)

    # Check for duplicate (against main key and existing sub keys)
    if global_settings.auth.api_key and secrets.compare_digest(
        request.key, global_settings.auth.api_key
    ):
        raise HTTPException(
            status_code=400, detail="Sub key cannot be the same as the main key"
        )

    for sk in global_settings.auth.sub_keys:
        if sk.key and secrets.compare_digest(request.key, sk.key):
            raise HTTPException(
                status_code=400, detail="This key already exists"
            )

    entry = SubKeyEntry(
        key=request.key,
        name=request.name or "",
        created_at=datetime.now(UTC).isoformat(),
    )
    global_settings.auth.sub_keys.append(entry)

    try:
        global_settings.save()
    except Exception as e:
        # Rollback
        global_settings.auth.sub_keys.pop()
        raise HTTPException(
            status_code=500, detail=f"Failed to save settings: {e}"
        )

    logger.info(f"Sub key created: {request.name or '(unnamed)'}")
    return {"success": True, "sub_key": entry.to_dict()}


@router.delete("/api/sub-keys")
async def delete_sub_key(
    request: DeleteSubKeyRequest, is_admin: bool = Depends(require_admin)
):
    """Delete a sub API key.

    Args:
        request: DeleteSubKeyRequest with the key to delete.

    Returns:
        JSON with success status.

    Raises:
        HTTPException: 404 if key not found.
    """
    global_settings = _get_global_settings()
    if global_settings is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    # Find and remove the key
    for i, sk in enumerate(global_settings.auth.sub_keys):
        if sk.key and secrets.compare_digest(request.key, sk.key):
            removed = global_settings.auth.sub_keys.pop(i)
            try:
                global_settings.save()
            except Exception as e:
                global_settings.auth.sub_keys.insert(i, removed)
                raise HTTPException(
                    status_code=500, detail=f"Failed to save settings: {e}"
                )
            logger.info(f"Sub key deleted: {sk.name or '(unnamed)'}")
            return {"success": True}

    raise HTTPException(status_code=404, detail="Sub key not found")


# =============================================================================
# Grammar API Routes
# =============================================================================


_SUPPORTED_MODELS_DOC_RE = re.compile(
    r"Supported models:\s*\n((?:\s*-\s*\S.*\n?)+)",
)


def _models_from_docstring(fn) -> list[str]:
    """Extract the ``Supported models:`` bullet list from an xgrammar 0.1.34+
    structural-tag function's docstring. Returns ``[]`` if the section is
    absent or unparseable."""
    doc = inspect.getdoc(fn) or ""
    match = _SUPPORTED_MODELS_DOC_RE.search(doc)
    if not match:
        return []
    return [
        line.strip().lstrip("-").strip()
        for line in match.group(1).splitlines()
        if line.strip().startswith("-")
    ]


@router.get("/api/grammar/parsers")
async def list_grammar_parsers(is_admin: bool = Depends(require_admin)):
    """Return available reasoning parser names from xgrammar.

    Supports both API generations:

    - **xgrammar 0.1.34+** exposes a per-model registry at
      ``xgrammar.builtin_structural_tag._structural_tag_registry``; supported
      model names are pulled from each function's docstring.
    - **xgrammar 0.1.32–0.1.33** exposes the now-removed helper
      ``get_builtin_structural_tag_supported_models()``.

    Returns ``[]`` if xgrammar is missing, fails to load (e.g. broken native
    binding on macOS arm64), or has neither API available.
    """
    # Prefer the 0.1.34+ registry so newer parsers (qwen3_6, gemma4,
    # deepseek_v4, ...) are exposed.
    try:
        from xgrammar.builtin_structural_tag import _structural_tag_registry

        return [
            {"value": style, "label": style, "models": _models_from_docstring(fn)}
            for style, fn in _structural_tag_registry.items()
        ]
    except Exception as e:
        logger.debug("xgrammar 0.1.34+ registry unavailable: %s", e)

    # Fall back to the pre-0.1.34 helper.
    try:
        from xgrammar import get_builtin_structural_tag_supported_models

        supported = get_builtin_structural_tag_supported_models()
        return [
            {"value": style, "label": style, "models": models}
            for style, models in supported.items()
        ]
    except Exception as e:
        logger.warning("xgrammar parser discovery unavailable: %s", e)
        return []


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

    # SSD cache dir is set on the scheduler_config when the user enables paged
    # SSD caching; admin UI consumes it to gate the dflash SSD toggle.
    ssd_cache_dir = getattr(
        getattr(engine_pool, "_scheduler_config", None),
        "paged_ssd_cache_dir",
        None,
    )
    dflash_ssd_cache_available = bool(ssd_cache_dir)

    # Combine model info with settings
    models = []
    for model_info in models_status:
        model_id = model_info["id"]
        settings = all_settings.get(model_id)

        is_paroquant, paroquant_reason = _paroquant_compat_for_model(model_info)
        compat_ok, compat_reason = _dflash_compat_for_model(model_info)
        mtp_compat_ok, mtp_compat_reason = _mtp_compat_for_model(model_info)

        model_data = {
            "id": model_id,
            "model_path": model_info.get("model_path", ""),
            "loaded": model_info.get("loaded", False),
            "is_loading": model_info.get("is_loading", False),
            "estimated_size": model_info.get("estimated_size", 0),
            "estimated_size_formatted": format_size(model_info.get("estimated_size", 0)),
            "pinned": model_info.get("pinned", False),
            "is_default": server_state.default_model == model_id if server_state else False,
            "engine_type": model_info.get("engine_type", "batched"),
            "model_type": model_info.get("model_type", "llm"),
            "config_model_type": model_info.get("config_model_type", ""),
            "thinking_default": model_info.get("thinking_default"),
            "preserve_thinking_default": model_info.get("preserve_thinking_default"),
            "last_access": model_info.get("last_access"),
            "dflash_compatible": compat_ok,
            "dflash_compatibility_reason": compat_reason,
            "dflash_ssd_cache_available": dflash_ssd_cache_available,
            "mtp_compatible": mtp_compat_ok,
            "mtp_compatibility_reason": mtp_compat_reason,
            "is_paroquant": is_paroquant,
            "paroquant_reason": paroquant_reason,
        }

        # Add settings if available
        if settings:
            model_data["settings"] = asdict(settings)

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


async def _require_admin_or_bearer(request: Request) -> bool:
    """Allow admin session OR a valid Bearer API key (for CLI use)."""
    gs = _get_global_settings() if _get_global_settings else None

    # No-auth mode: always allow
    if gs is not None and gs.auth.skip_api_key_verification:
        return True

    # Valid admin session cookie
    if verify_session(request):
        return True

    # Bearer token matching the configured API key
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer ") and gs is not None:
        token = auth_header[7:]
        server_key = gs.auth.api_key or ""
        sub_keys = gs.auth.sub_keys or []
        if verify_api_key(token, server_key):
            return True
        for sk in sub_keys:
            if verify_api_key(token, getattr(sk, "key", "")):
                return True

    raise HTTPException(
        status_code=401,
        detail="Admin authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )


@router.post("/api/models/{model_id}/load")
async def load_model(
    model_id: str,
    is_admin: bool = Depends(_require_admin_or_bearer),
):
    """Manually load a model into memory."""
    engine_pool = _get_engine_pool()
    if engine_pool is None:
        raise HTTPException(status_code=503, detail="Engine pool not initialized")

    entry = engine_pool.get_entry(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    if entry.engine is not None:
        return {"status": "ok", "model_id": model_id, "message": f"Already loaded: {model_id}"}
    if entry.is_loading:
        raise HTTPException(status_code=409, detail=f"Model is already loading: {model_id}")

    try:
        await engine_pool.get_engine(model_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    logger.info(f"Manually loaded model: {model_id}")
    return {"status": "ok", "model_id": model_id, "message": f"Loaded {model_id}"}


@router.post("/api/reload")
async def reload_models(is_admin: bool = Depends(require_admin)):
    """Reload models: re-read model settings, re-discover models, preload pinned."""
    success, message = await _reload_models()
    if success:
        return {"status": "ok", "message": message}
    raise HTTPException(status_code=500, detail=message)


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
    current_settings = settings_manager.get_settings(model_id)

    # Apply updates — use model_fields_set to distinguish "sent as null"
    # (clear to default) from "not sent" (don't touch).
    sent = request.model_fields_set
    prev_engine_type = entry.engine_type  # Track for requires_reload check
    if "model_alias" in sent:
        alias_value = request.model_alias.strip() if request.model_alias else None
        if alias_value == "":
            alias_value = None
        if alias_value is not None:
            all_settings = settings_manager.get_all_settings()
            for mid, ms in all_settings.items():
                if mid != model_id and ms.model_alias == alias_value:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Alias '{alias_value}' is already used by model '{mid}'",
                    )
            for mid in engine_pool._entries:
                if mid != model_id and mid == alias_value:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Alias '{alias_value}' conflicts with model directory name '{mid}'",
                    )
        current_settings.model_alias = alias_value
    if "model_type_override" in sent:
        valid_types = {"llm", "vlm", "embedding", "reranker", "audio_stt", "audio_tts", "audio_sts"}
        # Treat empty string as None (auto-detect)
        override_value = request.model_type_override or None
        if override_value is not None and override_value not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model_type_override: {request.model_type_override}",
            )
        current_settings.model_type_override = override_value
        # Update engine pool entry type immediately
        type_to_engine = {
            "llm": "batched",
            "vlm": "vlm",
            "embedding": "embedding",
            "reranker": "reranker",
            "audio_stt": "audio_stt",
            "audio_tts": "audio_tts",
            "audio_sts": "audio_sts",
        }
        if override_value:
            entry.model_type = override_value
            entry.engine_type = type_to_engine.get(override_value, "batched")
        else:
            # Reset to auto-detected type
            from pathlib import Path

            from ..model_discovery import detect_model_type

            detected_type = detect_model_type(Path(entry.model_path))
            entry.model_type = detected_type
            entry.engine_type = type_to_engine.get(detected_type, "batched")
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
    if "min_p" in sent:
        current_settings.min_p = request.min_p
    if "presence_penalty" in sent:
        current_settings.presence_penalty = request.presence_penalty
    if "force_sampling" in sent:
        current_settings.force_sampling = request.force_sampling
    if "max_tool_result_tokens" in sent:
        # 0 means disable (reset to None)
        current_settings.max_tool_result_tokens = (
            request.max_tool_result_tokens if request.max_tool_result_tokens and request.max_tool_result_tokens > 0 else None
        )
    if "enable_thinking" in sent:
        current_settings.enable_thinking = request.enable_thinking
    if "thinking_budget_enabled" in sent:
        current_settings.thinking_budget_enabled = request.thinking_budget_enabled or False
    if "thinking_budget_tokens" in sent:
        current_settings.thinking_budget_tokens = (
            request.thinking_budget_tokens if request.thinking_budget_tokens and request.thinking_budget_tokens > 0 else None
        )
    if "chat_template_kwargs" in sent:
        current_settings.chat_template_kwargs = request.chat_template_kwargs
    if "forced_ct_kwargs" in sent:
        current_settings.forced_ct_kwargs = request.forced_ct_kwargs
    if "ttl_seconds" in sent:
        current_settings.ttl_seconds = request.ttl_seconds
    if "index_cache_freq" in sent:
        # 0 means disable (reset to None)
        current_settings.index_cache_freq = (
            request.index_cache_freq
            if request.index_cache_freq and request.index_cache_freq >= 2
            else None
        )
    # TurboQuant KV cache settings
    if "turboquant_kv_enabled" in sent:
        current_settings.turboquant_kv_enabled = request.turboquant_kv_enabled or False
    if "turboquant_kv_bits" in sent:
        current_settings.turboquant_kv_bits = request.turboquant_kv_bits or 4
    # SpecPrefill settings
    if "specprefill_enabled" in sent:
        current_settings.specprefill_enabled = request.specprefill_enabled or False
    if "specprefill_draft_model" in sent:
        current_settings.specprefill_draft_model = request.specprefill_draft_model or None
    if "specprefill_keep_pct" in sent:
        current_settings.specprefill_keep_pct = request.specprefill_keep_pct or None
    if "specprefill_threshold" in sent:
        current_settings.specprefill_threshold = request.specprefill_threshold or None
    # DFlash settings
    if "dflash_enabled" in sent:
        new_dflash_enabled = bool(request.dflash_enabled)
        if new_dflash_enabled:
            from ..engine.dflash import is_dflash_compatible

            compat_ok, compat_reason = is_dflash_compatible(entry.model_path)
            if not compat_ok:
                raise HTTPException(status_code=400, detail=compat_reason)
        current_settings.dflash_enabled = new_dflash_enabled
    if "dflash_draft_model" in sent:
        current_settings.dflash_draft_model = request.dflash_draft_model or None
    if "dflash_draft_quant_enabled" in sent:
        current_settings.dflash_draft_quant_enabled = bool(request.dflash_draft_quant_enabled) if request.dflash_draft_quant_enabled is not None else None
    if "dflash_draft_quant_weight_bits" in sent:
        current_settings.dflash_draft_quant_weight_bits = int(request.dflash_draft_quant_weight_bits) if request.dflash_draft_quant_weight_bits is not None else None
    if "dflash_draft_quant_activation_bits" in sent:
        current_settings.dflash_draft_quant_activation_bits = int(request.dflash_draft_quant_activation_bits) if request.dflash_draft_quant_activation_bits is not None else None
    if "dflash_draft_quant_group_size" in sent:
        current_settings.dflash_draft_quant_group_size = int(request.dflash_draft_quant_group_size) if request.dflash_draft_quant_group_size is not None else None
    if "dflash_max_ctx" in sent:
        # 0/None means "unlimited" — the engine treats None as no fallback threshold
        value = request.dflash_max_ctx
        current_settings.dflash_max_ctx = value if value and value > 0 else None
    if "dflash_in_memory_cache" in sent:
        current_settings.dflash_in_memory_cache = bool(request.dflash_in_memory_cache)
    if "dflash_in_memory_cache_max_entries" in sent:
        value = request.dflash_in_memory_cache_max_entries
        current_settings.dflash_in_memory_cache_max_entries = (
            int(value) if value and value > 0 else 4
        )
    if "dflash_in_memory_cache_max_bytes" in sent and request.dflash_in_memory_cache_max_bytes:
        current_settings.dflash_in_memory_cache_max_bytes = int(
            request.dflash_in_memory_cache_max_bytes
        )
    if "dflash_ssd_cache" in sent:
        ssd_requested = bool(request.dflash_ssd_cache)
        if ssd_requested:
            in_mem_after = (
                bool(request.dflash_in_memory_cache)
                if "dflash_in_memory_cache" in sent
                else current_settings.dflash_in_memory_cache
            )
            if not in_mem_after:
                raise HTTPException(
                    status_code=400,
                    detail="DFlash SSD cache requires the in-memory cache to be enabled.",
                )
            ssd_dir = getattr(
                getattr(_get_engine_pool(), "_scheduler_config", None),
                "paged_ssd_cache_dir",
                None,
            )
            if not ssd_dir:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "DFlash SSD cache requires oMLX paged SSD cache to be enabled "
                        "(set --paged-ssd-cache-dir or configure it in settings)."
                    ),
                )
        current_settings.dflash_ssd_cache = ssd_requested

    # Native MTP (mlx-lm PR 990 / PR 15 monkey-patch)
    if "mtp_enabled" in sent:
        new_mtp_enabled = bool(request.mtp_enabled)
        if new_mtp_enabled:
            # Compatibility check: the model needs MTP heads in config.json AND
            # the model_type must be one PR 990 / PR 15 covers AND the weight
            # files must actually contain mtp.* tensors. The last check is
            # the one that catches mlx-community converted weights where the
            # default sanitize path stripped the MTP heads.
            import json
            from pathlib import Path

            from ..utils.model_loading import _is_mtp_compatible

            cfg_path = Path(entry.model_path) / "config.json"
            if not cfg_path.exists():
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"MTP enabled but config.json missing at {cfg_path}; "
                        "cannot verify MTP compatibility."
                    ),
                )
            try:
                cfg = json.loads(cfg_path.read_text())
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"MTP enabled but failed to read model config: {e}",
                )
            model_type = cfg.get("model_type")
            if not _is_mtp_compatible(cfg, model_type):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Model is not MTP-compatible (model_type={model_type!r}, "
                        f"mtp_num_hidden_layers={cfg.get('mtp_num_hidden_layers', 0)}). "
                        "Native MTP requires Qwen3.5/3.6 or DeepSeek-V4 with MTP heads."
                    ),
                )
            if not _model_has_mtp_weight_tensors(Path(entry.model_path)):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Config declares MTP layers but the converted weights are "
                        "missing mtp.* tensors. Re-convert from HF with a converter "
                        "that preserves MTP weights. The default "
                        "mlx-lm sanitize() path strips them."
                    ),
                )
            # Mutual exclusion with DFlash / TurboQuant — ModelSettings.__post_init__
            # also enforces this, but we surface a clearer error here.
            dflash_after = (
                bool(request.dflash_enabled)
                if "dflash_enabled" in sent
                else current_settings.dflash_enabled
            )
            if dflash_after:
                raise HTTPException(
                    status_code=400,
                    detail="MTP and DFlash cannot both be enabled; choose one speculative-decoding path.",
                )
            tq_after = (
                bool(request.turboquant_kv_enabled)
                if "turboquant_kv_enabled" in sent
                else current_settings.turboquant_kv_enabled
            )
            if tq_after:
                raise HTTPException(
                    status_code=400,
                    detail="MTP and TurboQuant KV cannot both be enabled; TurboQuant patches the attention path MTP relies on.",
                )
        current_settings.mtp_enabled = new_mtp_enabled

    # VLM MTP (mlx-vlm f96138e+, gemma4_assistant drafter)
    if "vlm_mtp_enabled" in sent:
        new_vlm_mtp = bool(request.vlm_mtp_enabled)
        if new_vlm_mtp:
            drafter_after = (
                request.vlm_mtp_draft_model
                if "vlm_mtp_draft_model" in sent
                else current_settings.vlm_mtp_draft_model
            )
            if not drafter_after:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "vlm_mtp_enabled requires vlm_mtp_draft_model "
                        "(path to a gemma4_assistant drafter, "
                        "e.g. 'gemma-4-26B-A4B-it-assistant')."
                    ),
                )
            # Mutex enforced again at ModelSettings.__post_init__ for
            # last-mile safety, but surface a clearer error here.
            for other_field, other_label in (
                ("dflash_enabled", "DFlash"),
                ("specprefill_enabled", "SpecPrefill"),
                ("mtp_enabled", "MTP"),
                ("turboquant_kv_enabled", "TurboQuant KV"),
            ):
                other_after = (
                    bool(getattr(request, other_field))
                    if other_field in sent
                    else getattr(current_settings, other_field)
                )
                if other_after:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"vlm_mtp_enabled and {other_label} cannot both be "
                            "enabled; choose one speculative-decoding path."
                        ),
                    )
        current_settings.vlm_mtp_enabled = new_vlm_mtp
    if "vlm_mtp_draft_model" in sent:
        current_settings.vlm_mtp_draft_model = request.vlm_mtp_draft_model or None
    if "vlm_mtp_draft_block_size" in sent:
        current_settings.vlm_mtp_draft_block_size = request.vlm_mtp_draft_block_size

    if "reasoning_parser" in sent:
        current_settings.reasoning_parser = request.reasoning_parser or None
    if request.is_pinned is not None:
        current_settings.is_pinned = request.is_pinned
        # Also update the engine pool entry
        entry.is_pinned = request.is_pinned
    if request.is_default is not None:
        current_settings.is_default = request.is_default
        # Update server_state.default_model if setting as default
        if request.is_default and server_state:
            server_state.default_model = model_id
    if "trust_remote_code" in sent:
        current_settings.trust_remote_code = bool(request.trust_remote_code)

    # If an active profile was set, clear it when the user's save diverges
    # from the profile's stored values.  Only compare fields present in
    # both the profile and the current settings — new fields in the model
    # settings that the profile doesn't have are silently merged in, and
    # removed fields (no longer in the profile) are skipped.
    if current_settings.active_profile_name:
        profile = settings_manager.get_profile(
            model_id, current_settings.active_profile_name
        )
        if profile is None:
            current_settings.active_profile_name = None
        else:
            profile_settings = profile.get("settings", {}) or {}
            candidate = current_settings.to_dict()
            diverged = False
            for key, expected in profile_settings.items():
                # Profile None means "unconstrained" — candidate.to_dict()
                # drops None, so treat profile None as no constraint to
                # keep the comparison symmetric.
                if expected is None:
                    continue
                if key not in candidate:
                    diverged = True
                    break
                if candidate[key] != expected:
                    diverged = True
                    break
            if diverged:
                current_settings.active_profile_name = None
            else:
                new_fields = {
                    k: v for k, v in candidate.items()
                    if k not in profile_settings and k not in EXCLUDED_FROM_PROFILES
                }
                if new_fields:
                    profile_settings.update(new_fields)
                    profile["settings"] = profile_settings
                    settings_manager.update_profile(
                        model_id,
                        current_settings.active_profile_name,
                        settings=profile_settings,
                    )

    # Persist settings
    settings_manager.set_settings(model_id, current_settings)

    # Auto-unload (and re-load if pinned) when a setting that only takes
    # effect at engine construction time is changed on a loaded model.
    requires_reload = (
        entry.engine is not None
        and (
            ("model_type_override" in sent and entry.engine_type != prev_engine_type)
            or "index_cache_freq" in sent
            or "dflash_enabled" in sent
            or "dflash_draft_model" in sent
            or "dflash_draft_quant_enabled" in sent
            or "dflash_draft_quant_weight_bits" in sent
            or "dflash_draft_quant_activation_bits" in sent
            or "dflash_draft_quant_group_size" in sent
            or "dflash_max_ctx" in sent
            or "dflash_in_memory_cache" in sent
            or "dflash_in_memory_cache_max_entries" in sent
            or "dflash_in_memory_cache_max_bytes" in sent
            or "dflash_ssd_cache" in sent
            # trust_remote_code is plumbed at model load time; toggling it on
            # an already-loaded engine has no effect until reload.
            or "trust_remote_code" in sent
        )
    )
    auto_unloaded = False
    auto_reloaded = False
    if requires_reload:
        was_pinned = entry.is_pinned
        try:
            logger.info(
                f"Settings changed for loaded model {model_id}, auto-unloading."
            )
            await engine_pool._unload_engine(model_id)
            auto_unloaded = True
        except Exception as e:
            logger.warning(f"Auto-unload failed for {model_id}: {e}")
        if auto_unloaded and was_pinned:
            try:
                await engine_pool._load_engine(model_id)
                auto_reloaded = True
                logger.info(f"Auto-reloaded pinned model {model_id} with new settings.")
            except Exception as e:
                logger.warning(f"Auto-reload failed for pinned model {model_id}: {e}")

    return {
        "success": True,
        "model_id": model_id,
        "settings": current_settings.to_dict(),
        "model_type": entry.model_type,
        "engine_type": entry.engine_type,
        "requires_reload": requires_reload,
        "auto_unloaded": auto_unloaded,
        "auto_reloaded": auto_reloaded,
    }


# =============================================================================
# Profile & Template endpoints
# =============================================================================


def _require_settings_manager():
    mgr = _get_settings_manager()
    if mgr is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    return mgr


def _require_model(model_id: str):
    pool = _get_engine_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="Engine pool not initialized")
    entry = pool.get_entry(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    return entry


@router.get("/api/models/{model_id}/profiles")
async def list_model_profiles(
    model_id: str,
    is_admin: bool = Depends(require_admin),
):
    mgr = _require_settings_manager()
    _require_model(model_id)
    return {"profiles": mgr.list_profiles(model_id)}


@router.post("/api/models/{model_id}/profiles")
async def create_model_profile(
    model_id: str,
    request: CreateProfileRequest,
    is_admin: bool = Depends(require_admin),
):
    from ..model_profiles import InvalidProfileNameError, filter_universal_fields

    mgr = _require_settings_manager()
    _require_model(model_id)
    try:
        profile = mgr.save_profile(
            model_id=model_id,
            name=request.name,
            display_name=request.display_name,
            description=request.description,
            settings=request.settings or {},
            source_template=request.source_template,
        )
    except InvalidProfileNameError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    if request.also_save_as_template:
        try:
            mgr.upsert_template(
                name=request.name,
                display_name=request.display_name,
                description=request.description,
                settings=filter_universal_fields(request.settings or {}),
            )
        except InvalidProfileNameError as e:
            raise HTTPException(status_code=400, detail=str(e))
    return {"profile": profile}


@router.put("/api/models/{model_id}/profiles/{name}")
async def update_model_profile(
    model_id: str,
    name: str,
    request: UpdateProfileRequest,
    is_admin: bool = Depends(require_admin),
):
    from ..model_profiles import InvalidProfileNameError, filter_universal_fields

    mgr = _require_settings_manager()
    _require_model(model_id)
    try:
        updated = mgr.update_profile(
            model_id=model_id,
            name=name,
            new_name=request.new_name,
            display_name=request.display_name,
            description=request.description,
            settings=request.settings,
            source_template=request.source_template,
        )
    except InvalidProfileNameError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    if updated is None:
        raise HTTPException(status_code=404, detail=f"Profile not found: {name}")

    if request.also_save_as_template and request.settings is not None:
        try:
            mgr.upsert_template(
                name=updated["name"],
                display_name=updated["display_name"],
                description=updated.get("description"),
                settings=filter_universal_fields(request.settings),
            )
        except InvalidProfileNameError as e:
            raise HTTPException(status_code=400, detail=str(e))
    return {"profile": updated}


@router.delete("/api/models/{model_id}/profiles/{name}")
async def delete_model_profile(
    model_id: str,
    name: str,
    is_admin: bool = Depends(require_admin),
):
    mgr = _require_settings_manager()
    _require_model(model_id)
    if not mgr.delete_profile(model_id, name):
        raise HTTPException(status_code=404, detail=f"Profile not found: {name}")
    return {"deleted": True, "name": name}


@router.post("/api/models/{model_id}/profiles/{name}/apply")
async def apply_model_profile(
    model_id: str,
    name: str,
    is_admin: bool = Depends(require_admin),
):
    mgr = _require_settings_manager()
    _require_model(model_id)
    applied = mgr.apply_profile(model_id, name)
    if applied is None:
        raise HTTPException(status_code=404, detail=f"Profile not found: {name}")
    return {"model_id": model_id, "settings": applied.to_dict()}


@router.get("/api/profile-fields")
async def get_profile_fields(is_admin: bool = Depends(require_admin)):
    from ..model_profiles import (
        MODEL_SPECIFIC_PROFILE_FIELDS,
        UNIVERSAL_PROFILE_FIELDS,
    )

    return {
        "universal": list(UNIVERSAL_PROFILE_FIELDS),
        "model_specific": list(MODEL_SPECIFIC_PROFILE_FIELDS),
    }


@router.get("/api/profile-templates")
async def list_templates(is_admin: bool = Depends(require_admin)):
    mgr = _require_settings_manager()
    return {"templates": mgr.list_templates()}


@router.post("/api/profile-templates")
async def create_template(
    request: CreateTemplateRequest,
    is_admin: bool = Depends(require_admin),
):
    from ..model_profiles import InvalidProfileNameError

    mgr = _require_settings_manager()
    try:
        tmpl = mgr.save_template(
            name=request.name,
            display_name=request.display_name,
            description=request.description,
            settings=request.settings or {},
        )
    except InvalidProfileNameError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"template": tmpl}


@router.put("/api/profile-templates/{name}")
async def update_template(
    name: str,
    request: UpdateTemplateRequest,
    is_admin: bool = Depends(require_admin),
):
    from ..model_profiles import InvalidProfileNameError

    mgr = _require_settings_manager()
    try:
        updated = mgr.update_template(
            name=name,
            new_name=request.new_name,
            display_name=request.display_name,
            description=request.description,
            settings=request.settings,
        )
    except InvalidProfileNameError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    if updated is None:
        raise HTTPException(status_code=404, detail=f"Template not found: {name}")
    return {"template": updated}


@router.delete("/api/profile-templates/{name}")
async def delete_template(
    name: str,
    is_admin: bool = Depends(require_admin),
):
    mgr = _require_settings_manager()
    if not mgr.delete_template(name):
        raise HTTPException(status_code=404, detail=f"Template not found: {name}")
    return {"deleted": True, "name": name}


# =============================================================================
# Preset refresh (proxy to omlx.ai to avoid CORS)
# =============================================================================


@router.post("/api/presets/refresh")
async def refresh_presets(is_admin: bool = Depends(require_admin)):
    """Fetch the latest preset bundle from omlx.ai and return it.

    The client uses this instead of fetching omlx.ai directly so we do not
    depend on CORS headers on the remote host. Any failure is surfaced as 502
    so the client can silently fall back to the bundled presets.
    """
    try:
        resp = await asyncio.to_thread(
            requests.get,
            PRESET_REMOTE_URL,
            timeout=10,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Fetch failed: {e}")
    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Remote returned {resp.status_code}",
        )
    try:
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Invalid JSON: {e}")


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
            with open(gen_config_path, encoding="utf-8") as f:
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
            with open(config_path, encoding="utf-8") as f:
                model_config = json_module.load(f)

            max_pos = (
                model_config.get("max_position_embeddings")
                or model_config.get("max_seq_len")
                or model_config.get("seq_length")
                or model_config.get("n_positions")
            )

            # Nested config fallback (VLM, MoE models like Qwen3.5, GLM-4V)
            if not max_pos:
                text_config = model_config.get("text_config", {})
                if isinstance(text_config, dict):
                    max_pos = (
                        text_config.get("max_position_embeddings")
                        or text_config.get("max_seq_len")
                        or text_config.get("seq_length")
                        or text_config.get("n_positions")
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


@router.get("/api/server-info")
async def get_server_info(is_admin: bool = Depends(require_admin)):
    """Return server connectivity metadata for the dashboard.

    Provides the configured host, port, and the list of user-facing
    aliases (hostnames/IPs) that the dashboard can use to render
    selectable API URL hints.

    Returns:
        JSON object with ``host``, ``port``, and ``aliases``.

    Raises:
        HTTPException: 401 if not authenticated, 503 if server not initialized.
    """
    from ..utils.network import detect_server_aliases

    global_settings = _get_global_settings()
    if global_settings is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    configured = list(global_settings.server.server_aliases)
    if configured:
        aliases = configured
    else:
        # Fall back to live detection if persisted list is empty.
        aliases = detect_server_aliases(host=global_settings.server.host)

    return {
        "host": global_settings.server.host,
        "port": global_settings.server.port,
        "aliases": aliases,
    }


def _schedule_self_terminate(delay: float = 0.5) -> None:
    """Schedule ``os.kill(getpid(), SIGTERM)`` on the running loop.

    Extracted from the restart handler so tests can patch this seam
    instead of mocking ``asyncio.get_running_loop`` globally (which
    interferes with FastAPI's TestClient portal).
    """
    pid = os.getpid()

    def _kill() -> None:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            # Already exited (e.g. concurrent SIGTERM) — nothing to do.
            pass
        except Exception:  # pragma: no cover — best-effort signal.
            logger.exception("Failed to self-terminate for restart")

    asyncio.get_running_loop().call_later(delay, _kill)


@router.post("/api/server/restart")
async def restart_server(is_admin: bool = Depends(require_admin)):
    """Trigger a server restart via the menubar supervisor.

    The handler does not perform the restart itself — it returns 202 and
    schedules ``os.kill(os.getpid(), SIGTERM)`` 500ms after the response
    is queued. The menubar app's ``ServerManager._health_check_loop``
    detects the process exit and respawns the server with a short
    backoff (~5s).

    Gated by the ``OMLX_SUPERVISED`` environment variable so plain
    ``omlx serve`` (no supervisor) returns 503 rather than killing the
    server with no respawn path.
    """
    supervisor = os.environ.get("OMLX_SUPERVISED")
    if not supervisor:
        raise HTTPException(
            status_code=503,
            detail=(
                "Server is not running under a supervisor that can "
                "respawn it. Restart unavailable — use the menu bar "
                "app's Restart, or restart from your shell."
            ),
        )

    _schedule_self_terminate(0.5)
    logger.warning("Server restart requested (supervisor=%s)", supervisor)

    # 5s backoff in ServerManager + ~1-2s startup = ~7s downtime budget.
    return JSONResponse(
        status_code=202,
        content={
            "status": "restarting",
            "supervisor": supervisor,
            "expected_downtime_seconds": 7,
        },
    )


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
            "server_aliases": list(global_settings.server.server_aliases),
            "sse_keepalive_mode": global_settings.server.sse_keepalive_mode,
        },
        "model": {
            "model_dirs": [
                str(d) for d in global_settings.model.get_model_dirs(global_settings.base_path)
            ],
            "model_dir": str(global_settings.model.get_model_dir(global_settings.base_path)),
            "max_model_memory": global_settings.model.max_model_memory,
            "model_fallback": global_settings.model.model_fallback,
        },
        "memory": {
            "max_process_memory": global_settings.memory.max_process_memory,
            "prefill_memory_guard": global_settings.memory.prefill_memory_guard,
        },
        "scheduler": {
            "max_concurrent_requests": global_settings.scheduler.max_concurrent_requests,
            "chunked_prefill": global_settings.scheduler.chunked_prefill,
        },
        "cache": {
            "enabled": global_settings.cache.enabled,
            "ssd_cache_dir": cache_dir,
            # Resolve "auto" to actual value (10% of SSD capacity)
            "ssd_cache_max_size": _format_cache_size(
                global_settings.cache.get_ssd_cache_max_size_bytes(global_settings.base_path)
            ),
            "hot_cache_only": global_settings.cache.hot_cache_only,
            "hot_cache_max_size": global_settings.cache.hot_cache_max_size,
            "initial_cache_blocks": global_settings.cache.initial_cache_blocks,
        },
        "mcp": {
            "config_path": global_settings.mcp.config_path,
        },
        "huggingface": {
            "endpoint": global_settings.huggingface.endpoint,
        },
        "modelscope": {
            "endpoint": global_settings.modelscope.endpoint,
        },
        "network": {
            "http_proxy": global_settings.network.http_proxy,
            "https_proxy": global_settings.network.https_proxy,
            "no_proxy": global_settings.network.no_proxy,
            "ca_bundle": global_settings.network.ca_bundle,
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
            "skip_api_key_verification": global_settings.auth.skip_api_key_verification,
            "sub_keys": [sk.to_dict() for sk in global_settings.auth.sub_keys],
        },
        "claude_code": {
            "context_scaling_enabled": global_settings.claude_code.context_scaling_enabled,
            "target_context_size": global_settings.claude_code.target_context_size,
            "mode": global_settings.claude_code.mode,
            "opus_model": global_settings.claude_code.opus_model,
            "sonnet_model": global_settings.claude_code.sonnet_model,
            "haiku_model": global_settings.claude_code.haiku_model,
        },
        "integrations": {
            "codex_model": global_settings.integrations.codex_model,
            "opencode_model": global_settings.integrations.opencode_model,
            "openclaw_model": global_settings.integrations.openclaw_model,
            "hermes_model": global_settings.integrations.hermes_model,
            "pi_model": global_settings.integrations.pi_model,
            "copilot_model": global_settings.integrations.copilot_model,
            "openclaw_tools_profile": global_settings.integrations.openclaw_tools_profile,
        },
        "system": {
            "total_memory_bytes": memory_info["total_bytes"],
            "total_memory": memory_info["total_formatted"],
            "auto_model_memory": memory_info["auto_limit_formatted"],
            "ssd_total_bytes": disk_info["total_bytes"],
            "ssd_total": disk_info["total_formatted"],
        },
        "ui": {
            "language": global_settings.ui.language,
        },
        "idle_timeout": {
            "idle_timeout_seconds": global_settings.idle_timeout.idle_timeout_seconds,
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
    runtime_applied: list[str] = []

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
    if request.sse_keepalive_mode is not None:
        valid_modes = {"chunk", "comment", "off"}
        if request.sse_keepalive_mode not in valid_modes:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid sse_keepalive_mode: {request.sse_keepalive_mode} "
                f"(must be one of {sorted(valid_modes)})",
            )
        global_settings.server.sse_keepalive_mode = request.sse_keepalive_mode
        runtime_applied.append("sse_keepalive_mode")

    if request.server_aliases is not None:
        from ..utils.network import is_valid_alias

        cleaned: list[str] = []
        seen: set[str] = set()
        for alias in request.server_aliases:
            if not isinstance(alias, str):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid server alias: each alias must be a string",
                )
            value = alias.strip()
            if not value or value in seen:
                continue
            if not is_valid_alias(value):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid server alias: {value!r} (must be a hostname or IP address)",
                )
            seen.add(value)
            cleaned.append(value)
        global_settings.server.server_aliases = cleaned
        runtime_applied.append("server_aliases")

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
            if request.max_model_memory.lower() == "disabled":
                max_bytes = None
            elif request.max_model_memory.lower() == "auto":
                max_bytes = global_settings.model.get_max_model_memory_bytes()
            else:
                max_bytes = parse_size(request.max_model_memory)
            success, msg = await _apply_max_model_memory_runtime(max_bytes)
            if success:
                runtime_applied.append("max_model_memory")
                logger.info(msg)
            else:
                logger.warning(f"Failed to apply max_model_memory runtime: {msg}")
        except ValueError as e:
            logger.warning(f"Invalid max_model_memory format: {e}")

    if request.model_fallback is not None:
        global_settings.model.model_fallback = request.model_fallback
        runtime_applied.append("model_fallback")

    # Apply process memory enforcement settings (Live)
    if request.max_process_memory is not None:
        global_settings.memory.max_process_memory = request.max_process_memory
        try:
            success, msg = await _apply_max_process_memory_runtime(
                request.max_process_memory
            )
            if success:
                runtime_applied.append("max_process_memory")
                logger.info(msg)
            else:
                logger.warning(f"Failed to apply max_process_memory: {msg}")
        except Exception as e:
            logger.warning(f"Error applying max_process_memory: {e}")

    # Apply prefill memory guard setting (Live)
    if request.memory_prefill_memory_guard is not None:
        global_settings.memory.prefill_memory_guard = (
            request.memory_prefill_memory_guard
        )
        from ..server import _server_state

        if _server_state.process_memory_enforcer is not None:
            _server_state.process_memory_enforcer.prefill_memory_guard = (
                request.memory_prefill_memory_guard
            )
        runtime_applied.append("prefill_memory_guard")
        logger.info(
            f"Prefill memory guard "
            f"{'enabled' if request.memory_prefill_memory_guard else 'disabled'}"
        )

    # Apply scheduler settings (restart required)
    if request.max_concurrent_requests is not None:
        global_settings.scheduler.max_concurrent_requests = (
            request.max_concurrent_requests
        )

    # Apply chunked prefill setting (Live)
    if request.chunked_prefill is not None:
        global_settings.scheduler.chunked_prefill = request.chunked_prefill
        from ..server import _server_state

        pool = _server_state.engine_pool
        if pool is not None:
            for mid, entry in pool._entries.items():
                if entry is None or entry.engine is None:
                    continue
                async_core = getattr(entry.engine, "_engine", None)
                core = getattr(async_core, "engine", None) if async_core is not None else None
                scheduler = getattr(core, "scheduler", None) if core is not None else None
                if scheduler is not None and hasattr(scheduler, "config"):
                    scheduler.config.chunked_prefill = request.chunked_prefill
        runtime_applied.append("chunked_prefill")
        logger.info(
            f"Chunked prefill {'enabled' if request.chunked_prefill else 'disabled'}"
        )

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
    if request.hot_cache_only is not None:
        global_settings.cache.hot_cache_only = request.hot_cache_only
    if request.hot_cache_max_size is not None:
        global_settings.cache.hot_cache_max_size = request.hot_cache_max_size
        cache_changed = True
    if request.initial_cache_blocks is not None:
        global_settings.cache.initial_cache_blocks = request.initial_cache_blocks

    if cache_changed:
        success, msg = await _apply_cache_settings_runtime(
            request.cache_enabled,
            request.ssd_cache_dir,
            request.ssd_cache_max_size,
            global_settings,
            hot_cache_max_size=request.hot_cache_max_size,
        )
        if success:
            runtime_applied.append("cache")
            logger.info(msg)
        else:
            logger.warning(f"Failed to apply cache settings runtime: {msg}")

    # Apply MCP settings (restart required)
    if request.mcp_config is not None:
        global_settings.mcp.config_path = request.mcp_config if request.mcp_config else None

    # Apply HuggingFace settings (Live - immediately applied via env var)
    if request.hf_endpoint is not None:
        global_settings.huggingface.endpoint = request.hf_endpoint
        if request.hf_endpoint:
            os.environ["HF_ENDPOINT"] = request.hf_endpoint
        elif "HF_ENDPOINT" in os.environ:
            del os.environ["HF_ENDPOINT"]
        runtime_applied.append("hf_endpoint")
        logger.info(
            f"HuggingFace endpoint updated to: "
            f"{request.hf_endpoint or '(default)'}"
        )

    # Apply ModelScope settings (Live - immediately applied via env var)
    if request.ms_endpoint is not None:
        global_settings.modelscope.endpoint = request.ms_endpoint
        if request.ms_endpoint:
            os.environ["MODELSCOPE_DOMAIN"] = request.ms_endpoint
        elif "MODELSCOPE_DOMAIN" in os.environ:
            del os.environ["MODELSCOPE_DOMAIN"]
        runtime_applied.append("ms_endpoint")
        logger.info(
            f"ModelScope endpoint updated to: "
            f"{request.ms_endpoint or '(default)'}"
        )

    # Apply network settings (Live - immediately applied via env vars)
    network_changed = False
    if request.network_http_proxy is not None:
        global_settings.network.http_proxy = request.network_http_proxy
        if request.network_http_proxy:
            os.environ["HTTP_PROXY"] = request.network_http_proxy
            os.environ["http_proxy"] = request.network_http_proxy
        else:
            os.environ.pop("HTTP_PROXY", None)
            os.environ.pop("http_proxy", None)
        network_changed = True

    if request.network_https_proxy is not None:
        global_settings.network.https_proxy = request.network_https_proxy
        if request.network_https_proxy:
            os.environ["HTTPS_PROXY"] = request.network_https_proxy
            os.environ["https_proxy"] = request.network_https_proxy
        else:
            os.environ.pop("HTTPS_PROXY", None)
            os.environ.pop("https_proxy", None)
        network_changed = True

    if request.network_no_proxy is not None:
        global_settings.network.no_proxy = request.network_no_proxy
        if request.network_no_proxy:
            os.environ["NO_PROXY"] = request.network_no_proxy
            os.environ["no_proxy"] = request.network_no_proxy
        else:
            os.environ.pop("NO_PROXY", None)
            os.environ.pop("no_proxy", None)
        network_changed = True

    if request.network_ca_bundle is not None:
        global_settings.network.ca_bundle = request.network_ca_bundle
        if request.network_ca_bundle:
            os.environ["REQUESTS_CA_BUNDLE"] = request.network_ca_bundle
            os.environ["SSL_CERT_FILE"] = request.network_ca_bundle
        else:
            os.environ.pop("REQUESTS_CA_BUNDLE", None)
            os.environ.pop("SSL_CERT_FILE", None)
        network_changed = True

    if network_changed:
        runtime_applied.append("network")
        logger.info("Network settings updated")

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
    # mode: standard is-not-None check is correct — mode must never be null
    if request.claude_code_mode is not None:
        global_settings.claude_code.mode = request.claude_code_mode
        claude_code_changed = True
    # model fields: use model_fields_set to distinguish "field absent from POST body"
    # from "field explicitly sent as null" — null must clear the field to None.
    # DO NOT use `is not None` here: that would prevent clearing a model field to null.
    if "claude_code_opus_model" in request.model_fields_set:
        global_settings.claude_code.opus_model = request.claude_code_opus_model
        claude_code_changed = True
    if "claude_code_sonnet_model" in request.model_fields_set:
        global_settings.claude_code.sonnet_model = request.claude_code_sonnet_model
        claude_code_changed = True
    if "claude_code_haiku_model" in request.model_fields_set:
        global_settings.claude_code.haiku_model = request.claude_code_haiku_model
        claude_code_changed = True

    if claude_code_changed:
        runtime_applied.append("claude_code")
        logger.info(
            f"Claude Code settings updated: "
            f"scaling={'enabled' if global_settings.claude_code.context_scaling_enabled else 'disabled'}, "
            f"target={global_settings.claude_code.target_context_size}, "
            f"mode={global_settings.claude_code.mode}, "
            f"opus={global_settings.claude_code.opus_model}, "
            f"sonnet={global_settings.claude_code.sonnet_model}, "
            f"haiku={global_settings.claude_code.haiku_model}"
        )

    # Apply integrations settings (Live - immediately applied)
    integrations_changed = False
    if "integrations_copilot_model" in request.model_fields_set:
        global_settings.integrations.copilot_model = request.integrations_copilot_model
        integrations_changed = True
    if "integrations_codex_model" in request.model_fields_set:
        global_settings.integrations.codex_model = request.integrations_codex_model
        integrations_changed = True
    if "integrations_opencode_model" in request.model_fields_set:
        global_settings.integrations.opencode_model = (
            request.integrations_opencode_model
        )
        integrations_changed = True
    if "integrations_openclaw_model" in request.model_fields_set:
        global_settings.integrations.openclaw_model = (
            request.integrations_openclaw_model
        )
        integrations_changed = True
    if "integrations_hermes_model" in request.model_fields_set:
        global_settings.integrations.hermes_model = request.integrations_hermes_model
        integrations_changed = True
    if "integrations_pi_model" in request.model_fields_set:
        global_settings.integrations.pi_model = request.integrations_pi_model
        integrations_changed = True
    if "integrations_openclaw_tools_profile" in request.model_fields_set:
        global_settings.integrations.openclaw_tools_profile = (
            request.integrations_openclaw_tools_profile
        )
        integrations_changed = True

    if integrations_changed:
        runtime_applied.append("integrations")
        logger.info(
            f"Integration settings updated: "
            f"copilot={global_settings.integrations.copilot_model}, "
            f"codex={global_settings.integrations.codex_model}, "
            f"opencode={global_settings.integrations.opencode_model}, "
            f"openclaw={global_settings.integrations.openclaw_model}, "
            f"hermes={global_settings.integrations.hermes_model}, "
            f"pi={global_settings.integrations.pi_model}"
        )

    # Apply UI settings
    if request.ui_language is not None:
        global_settings.ui.language = request.ui_language
        runtime_applied.append("ui_language")
        _refresh_i18n_globals()
        logger.info(f"UI language changed to: {request.ui_language}")

    # Apply idle timeout settings (Live)
    # Use model_fields_set to distinguish "explicitly sent as null" (disable)
    # from "not sent" (don't touch).
    if "idle_timeout_seconds" in request.model_fields_set:
        global_settings.idle_timeout.idle_timeout_seconds = request.idle_timeout_seconds
        runtime_applied.append("idle_timeout_seconds")
        if request.idle_timeout_seconds:
            logger.info(f"Idle timeout set to: {request.idle_timeout_seconds}s")
        else:
            logger.info("Idle timeout disabled")

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

    if request.skip_api_key_verification is not None:
        global_settings.auth.skip_api_key_verification = request.skip_api_key_verification
        runtime_applied.append("skip_api_key_verification")

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

    with open(file_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            lines.append(line)
            total_lines += 1

    return "".join(lines), total_lines


def _get_available_log_files(log_dir: Path) -> list[str]:
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
    file: str | None = None,
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
        "mlx-vlm": "https://github.com/Blaizzy/mlx-vlm",
        "mlx-embeddings": "https://github.com/Blaizzy/mlx-embeddings",
        "mlx-audio": "https://github.com/Blaizzy/mlx-audio",
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


def _build_runtime_cache_observability(
    global_settings,
    model_filter: str = "",
) -> dict:
    """Build runtime cache observability payload for dashboard.

    Includes the effective runtime paths and per-model SSD cache runtime stats
    from loaded schedulers, so users can verify real cache state without manual
    process inspection.
    """
    if global_settings is None:
        return {
            "base_path": "",
            "ssd_cache_dir": "",
            "response_state_dir": "",
            "models": [],
            "total_num_files": 0,
            "total_size_bytes": 0,
            "effective_block_sizes": [],
        }

    cache_dir = global_settings.cache.get_ssd_cache_dir(global_settings.base_path)
    payload = {
        "base_path": str(global_settings.base_path),
        "ssd_cache_dir": str(cache_dir),
        "response_state_dir": str(cache_dir / "response-state"),
        "models": [],
        "total_num_files": 0,
        "total_size_bytes": 0,
        "effective_block_sizes": [],
    }

    engine_pool = _get_engine_pool()
    if engine_pool is None:
        return payload

    block_sizes = set()

    for model_info in engine_pool.get_status().get("models", []):
        model_id = model_info.get("id")
        if not model_id:
            continue
        if model_filter and model_id != model_filter:
            continue
        if not model_info.get("loaded"):
            continue

        entry = engine_pool._entries.get(model_id)
        if entry is None or entry.engine is None:
            continue

        async_core = getattr(entry.engine, "_engine", None)
        core = getattr(async_core, "engine", None) if async_core is not None else None
        scheduler = getattr(core, "scheduler", None) if core is not None else None

        runtime_stats = None
        if scheduler is not None and hasattr(scheduler, "get_ssd_cache_stats"):
            try:
                runtime_stats = scheduler.get_ssd_cache_stats()
            except Exception as exc:
                logger.warning(
                    "Failed to collect runtime cache stats for model '%s': %s",
                    model_id,
                    exc,
                )
                continue

        if not runtime_stats:
            continue

        block_size = runtime_stats.get("block_size")
        indexed_blocks = runtime_stats.get("indexed_blocks")

        ssd_stats = runtime_stats.get("ssd_cache")
        if is_dataclass(ssd_stats):
            ssd_stats = asdict(ssd_stats)
        elif hasattr(ssd_stats, "to_dict"):
            ssd_stats = ssd_stats.to_dict()
        elif not isinstance(ssd_stats, dict):
            ssd_stats = {}

        ssd_manager = getattr(scheduler, "paged_ssd_cache_manager", None)
        scheduler_model_name = getattr(getattr(scheduler, "config", None), "model_name", "")
        if ssd_manager is not None and hasattr(ssd_manager, "get_stats_for_model"):
            try:
                scoped_ssd_stats = ssd_manager.get_stats_for_model(
                    scheduler_model_name or model_id
                )
                if is_dataclass(scoped_ssd_stats):
                    ssd_stats = asdict(scoped_ssd_stats)
                elif isinstance(scoped_ssd_stats, dict):
                    ssd_stats = scoped_ssd_stats
            except Exception as exc:
                logger.warning(
                    "Failed to collect model-scoped SSD cache stats for model '%s': %s",
                    model_id,
                    exc,
                )

        prefix_stats = runtime_stats.get("prefix_cache")
        if is_dataclass(prefix_stats):
            prefix_stats = asdict(prefix_stats)
        elif hasattr(prefix_stats, "to_dict"):
            prefix_stats = prefix_stats.to_dict()
        elif not isinstance(prefix_stats, dict):
            prefix_stats = {}

        indexed_blocks_value = indexed_blocks if isinstance(indexed_blocks, int) else 0
        if not isinstance(block_size, int) or block_size <= 0:
            block_size = int(prefix_stats.get("block_size", 0) or 0)

        partial_block_skips = int(prefix_stats.get("partial_block_skips", 0) or 0)
        partial_tokens_skipped = int(prefix_stats.get("partial_tokens_skipped", 0) or 0)
        last_partial_tokens_skipped = int(
            prefix_stats.get("last_partial_tokens_skipped", 0) or 0
        )
        last_tokens_to_next_block = int(
            prefix_stats.get("last_tokens_to_next_block", 0) or 0
        )

        has_sub_block_cache = (
            indexed_blocks_value == 0
            and isinstance(block_size, int)
            and block_size > 0
            and partial_block_skips > 0
        )

        model_payload = {
            "id": model_id,
            "block_size": block_size,
            "indexed_blocks": indexed_blocks_value,
            "indexed_blocks_display": (
                f"<{block_size}" if has_sub_block_cache else str(indexed_blocks_value)
            ),
            "has_sub_block_cache": has_sub_block_cache,
            "partial_block_skips": partial_block_skips,
            "partial_tokens_skipped": partial_tokens_skipped,
            "last_partial_tokens_skipped": last_partial_tokens_skipped,
            "last_tokens_to_next_block": last_tokens_to_next_block,
            "num_files": int(ssd_stats.get("num_files", 0) or 0),
            "total_size_bytes": int(ssd_stats.get("total_size_bytes", 0) or 0),
            "hot_cache_max_bytes": int(ssd_stats.get("hot_cache_max_bytes", 0) or 0),
            "hot_cache_size_bytes": int(ssd_stats.get("hot_cache_size_bytes", 0) or 0),
            "hot_cache_entries": int(ssd_stats.get("hot_cache_entries", 0) or 0),
        }

        payload["models"].append(model_payload)
        payload["total_num_files"] += model_payload["num_files"]
        payload["total_size_bytes"] += model_payload["total_size_bytes"]

        if isinstance(block_size, int) and block_size > 0:
            block_sizes.add(block_size)

    payload["effective_block_sizes"] = sorted(block_sizes)

    # Fallback: if no loaded models contributed stats, scan the cache
    # directory directly so the dashboard still shows real disk usage.
    if payload["total_num_files"] == 0 and cache_dir.exists():
        try:
            num_files = 0
            total_bytes = 0
            for subdir in "0123456789abcdef":
                subdir_path = cache_dir / subdir
                if not subdir_path.exists():
                    continue
                for f in subdir_path.glob("*.safetensors"):
                    num_files += 1
                    total_bytes += f.stat().st_size
            payload["total_num_files"] = num_files
            payload["total_size_bytes"] = total_bytes
        except Exception as exc:
            logger.warning("Failed to scan SSD cache directory: %s", exc)

    return payload


@router.get("/api/stats")
async def get_server_stats(
    model: str = "",
    scope: str = "session",
    is_admin: bool = Depends(require_admin),
):
    """Get server serving stats for the Status dashboard.

    Args:
        model: Filter by model ID. Empty string returns global aggregate.
        scope: "session" for current session, "alltime" for persisted totals.
    """
    from ..server import resolve_model_id
    from ..server_metrics import get_server_metrics

    metrics = get_server_metrics()
    resolved_model = resolve_model_id(model) or model if model else ""
    snapshot = metrics.get_snapshot(model_id=resolved_model, scope=scope)

    global_settings = _get_global_settings()
    host = global_settings.server.host if global_settings else "127.0.0.1"
    port = global_settings.server.port if global_settings else 8000
    api_key = global_settings.auth.api_key if global_settings else ""

    from ..utils.install import get_cli_prefix

    # Build active_models data for the dashboard card.
    active_models_data = _build_active_models_data()
    runtime_cache_data = _build_runtime_cache_observability(
        global_settings,
        model_filter=model,
    )

    return {
        **snapshot,
        "host": host,
        "port": port,
        "api_key": api_key or "",
        "cli_prefix": get_cli_prefix(),
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
        "active_models": active_models_data,
        "runtime_cache": runtime_cache_data,
    }


def _build_active_models_data() -> dict:
    """Build active models status for the dashboard Active Models card."""
    from ..model_discovery import format_size
    from ..prefill_progress import get_prefill_tracker

    engine_pool = _get_engine_pool()
    if engine_pool is None:
        return {
            "models": [],
            "model_memory_used": 0,
            "model_memory_max": 0,
            "total_active_requests": 0,
            "total_waiting_requests": 0,
        }

    now = time.monotonic()
    tracker = get_prefill_tracker()
    status = engine_pool.get_status()
    models = []
    total_active = 0
    total_waiting = 0

    for model_info in status.get("models", []):
        if not model_info.get("loaded") and not model_info.get("is_loading"):
            continue

        model_id = model_info["id"]
        active_requests = 0
        waiting_requests = 0
        running_by_id = {}
        waiting_ids = set()
        waiting = []
        activities = []

        # Get per-model active/waiting request counts.
        # Follow the same pattern as server.py /api/status endpoint.
        active_request_ids: set = set()
        entry = engine_pool._entries.get(model_id)
        if entry and entry.engine is not None:
            async_core = getattr(entry.engine, "_engine", None)
            if async_core is not None:
                core = getattr(async_core, "engine", None)
                if core is not None:
                    collectors = getattr(core, "_output_collectors", {})
                    try:
                        active_request_ids = set(collectors.keys())
                        active_requests = len(collectors)
                    except RuntimeError:
                        # Scheduler state is mutated from the engine executor;
                        # keep the dashboard endpoint best-effort rather than
                        # failing on a concurrent dict resize.
                        active_request_ids = set()
                        active_requests = len(collectors)

                    sched = getattr(core, "scheduler", None)
                    if sched is not None and hasattr(sched, "snapshot_for_admin"):
                        snap = sched.snapshot_for_admin()
                        running_by_id = snap["running_by_id"]
                        waiting_queue = snap["waiting"]
                        waiting_requests = len(waiting_queue)
                        waiting_ids = {req.request_id for req in waiting_queue}
                        waiting = [
                            {
                                "request_id": req.request_id,
                                "queue_position": idx,
                                "elapsed_seconds": max(0.0, now - req.arrival_time),
                                "prompt_tokens": getattr(req, "num_prompt_tokens", 0),
                            }
                            for idx, req in enumerate(waiting_queue, start=1)
                        ]
            elif hasattr(entry.engine, "get_activity_snapshot"):
                snapshot = entry.engine.get_activity_snapshot()
                active_requests = snapshot.get("active_requests", 0)
                activities = snapshot.get("activities", [])

        prefilling = tracker.get_model_progress(model_id)
        prefilling_ids = {p["request_id"] for p in prefilling}

        # Generating = active requests that finished prefill.
        generating = []
        for rid in sorted(active_request_ids - prefilling_ids - waiting_ids):
            req = running_by_id.get(rid)
            generated_tokens = getattr(req, "num_output_tokens", 0) if req else 0
            started_at = getattr(req, "generation_started_at", None) if req else None
            last_activity_at = getattr(req, "last_activity_at", None) if req else None
            elapsed = max(0.0, now - started_at) if started_at else None
            last_activity_age = (
                max(0.0, now - last_activity_at) if last_activity_at else None
            )
            tokens_per_second = (
                generated_tokens / elapsed if elapsed and elapsed > 0 else 0.0
            )
            generating.append(
                {
                    "request_id": rid,
                    "elapsed_seconds": elapsed,
                    "generated_tokens": generated_tokens,
                    "tokens_per_second": tokens_per_second,
                    "last_activity_age_seconds": last_activity_age,
                    "prompt_tokens": getattr(req, "num_prompt_tokens", 0) if req else 0,
                    "max_tokens": getattr(req, "max_tokens", None) if req else None,
                }
            )

        loading_started_at = model_info.get("loading_started_at")
        loading_elapsed_seconds = (
            max(0.0, now - loading_started_at) if loading_started_at else None
        )
        loading_estimated_seconds = None
        loading_remaining_seconds_estimate = None
        if loading_elapsed_seconds is not None:
            estimated_size_gb = model_info.get("estimated_size", 0) / (1024 ** 3)
            # Model loaders do not expose byte-level progress, so use a
            # deliberately conservative elapsed-time estimate and cap below
            # complete until the model is actually loaded.
            observed_seconds_per_gb = status.get("load_seconds_per_gb_estimate")
            observations = status.get("load_time_observations", 0)
            if observed_seconds_per_gb and observations >= 2:
                # Adapt to this machine/session once we have more than a
                # single potentially-misleading sample.
                loading_estimated_seconds = max(
                    3.0,
                    1.0 + estimated_size_gb * float(observed_seconds_per_gb),
                )
                if loading_elapsed_seconds < loading_estimated_seconds:
                    loading_remaining_seconds_estimate = max(
                        0.0, loading_estimated_seconds - loading_elapsed_seconds
                    )

        models.append({
            "id": model_id,
            "estimated_size": model_info.get("estimated_size", 0),
            "estimated_size_formatted": format_size(
                model_info.get("estimated_size", 0)
            ),
            "pinned": model_info.get("pinned", False),
            "is_loading": model_info.get("is_loading", False),
            "loading_elapsed_seconds": loading_elapsed_seconds,
            "loading_estimated_seconds": loading_estimated_seconds,
            "loading_remaining_seconds_estimate": loading_remaining_seconds_estimate,
            "active_requests": active_requests,
            "waiting_requests": waiting_requests,
            "waiting": waiting,
            "activities": activities,
            "prefilling": prefilling,
            "generating": generating,
        })

        total_active += active_requests
        total_waiting += waiting_requests

    return {
        "models": models,
        "model_memory_used": status.get("current_model_memory", 0),
        "model_memory_max": status.get("max_model_memory", 0),
        "total_active_requests": total_active,
        "total_waiting_requests": total_waiting,
    }


@router.post("/api/stats/clear")
async def clear_server_stats(is_admin: bool = Depends(require_admin)):
    """Clear session server metrics."""
    from ..server_metrics import get_server_metrics

    get_server_metrics().clear_metrics()
    return {"status": "ok"}


@router.post("/api/stats/clear-alltime")
async def clear_alltime_stats(is_admin: bool = Depends(require_admin)):
    """Clear all-time server metrics and delete persisted stats file."""
    from ..server_metrics import get_server_metrics

    get_server_metrics().clear_alltime_metrics()
    return {"status": "ok"}


@router.post("/api/ssd-cache/clear")
async def clear_ssd_cache(is_admin: bool = Depends(require_admin)):
    """Clear all SSD cache files for all loaded models.

    Uses loaded models' SSD cache managers when available.  Falls back to
    direct filesystem deletion so caches can be wiped even when no model
    is loaded.
    """
    total_deleted = 0

    # Phase 1: clear via loaded models' cache managers (updates in-memory index)
    engine_pool = _get_engine_pool()
    if engine_pool is not None:
        for model_info in engine_pool.get_status().get("models", []):
            model_id = model_info.get("id")
            if not model_id or not model_info.get("loaded"):
                continue

            entry = engine_pool._entries.get(model_id)
            if entry is None or entry.engine is None:
                continue

            async_core = getattr(entry.engine, "_engine", None)
            core = (
                getattr(async_core, "engine", None) if async_core is not None else None
            )
            scheduler = (
                getattr(core, "scheduler", None) if core is not None else None
            )

            if scheduler is not None:
                ssd_manager = getattr(scheduler, "paged_ssd_cache_manager", None)
                if ssd_manager is not None:
                    try:
                        deleted = ssd_manager.clear()
                        total_deleted += deleted
                    except Exception as exc:
                        logger.warning(
                            "Failed to clear SSD cache for model '%s': %s",
                            model_id,
                            exc,
                        )

    # Phase 2: remove any remaining files on disk (covers unloaded models)
    global_settings = _get_global_settings()
    if global_settings is not None:
        cache_dir = global_settings.cache.get_ssd_cache_dir(
            global_settings.base_path,
        )
        if cache_dir.exists():
            try:
                for subdir in "0123456789abcdef":
                    subdir_path = cache_dir / subdir
                    if not subdir_path.exists():
                        continue
                    for f in subdir_path.glob("*.safetensors"):
                        try:
                            f.unlink()
                            total_deleted += 1
                        except OSError:
                            pass
            except Exception as exc:
                logger.warning("Failed to clean SSD cache directory: %s", exc)

    return {"status": "ok", "total_deleted": total_deleted}


@router.post("/api/cache/probe")
async def probe_cache(
    request: CacheProbeRequest,
    is_admin: bool = Depends(require_admin),
):
    """Probe cache state for a chat message list.

    Classifies each block of the rendered prompt into one of three buckets:
    - ``blocks_ssd_hot``: in the SSD manager's hot cache (RAM copy of cold
      blocks, ready to mount without disk read)
    - ``blocks_ssd_disk``: only in the SSD index on disk
    - ``blocks_cold``: not cached anywhere (requires full prefill)

    The split is computed via a walk of the chain-hashed block sequence — the
    same hashing the scheduler uses at prefill time. The model must be loaded
    for the probe to run; unloaded models return ``model_loaded: false``.
    """
    engine_pool = _get_engine_pool()
    if engine_pool is None:
        raise HTTPException(status_code=503, detail="Engine pool not initialized")

    entry = engine_pool._entries.get(request.model_id)
    if entry is None:
        raise HTTPException(
            status_code=404, detail=f"Model not found: {request.model_id}"
        )
    if entry.engine is None:
        return {
            "model_id": request.model_id,
            "model_loaded": False,
            "reason": "Model is not loaded — load it to enable cache probing.",
        }

    engine = entry.engine
    tokenizer = getattr(engine, "_tokenizer", None)
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        raise HTTPException(
            status_code=400,
            detail="Model tokenizer does not support chat templating.",
        )

    # Reach into the scheduler to access the prefix index and SSD manager.
    async_core = getattr(engine, "_engine", None)
    core = getattr(async_core, "engine", None) if async_core is not None else None
    scheduler = getattr(core, "scheduler", None) if core is not None else None
    if scheduler is None:
        raise HTTPException(
            status_code=500, detail="Scheduler unavailable for loaded model."
        )

    prefix_cache = getattr(scheduler, "block_aware_cache", None)
    ssd_manager = getattr(scheduler, "paged_ssd_cache_manager", None)
    paged_cache = getattr(scheduler, "paged_cache_manager", None)
    block_size = getattr(
        getattr(scheduler, "config", None), "paged_cache_block_size", 0
    )
    if not block_size and prefix_cache is not None:
        block_size = getattr(prefix_cache, "block_size", 0)
    if not block_size:
        raise HTTPException(
            status_code=500,
            detail="Cache block size unavailable — cache may not be enabled.",
        )

    # Render + tokenize the prompt using the same path as generation so the
    # hashes line up with what the scheduler would produce at prefill.
    try:
        messages = request.messages
        if hasattr(engine, "_preprocess_messages"):
            messages = engine._preprocess_messages(messages)
        try:
            from ..api.tool_calling import convert_tools_for_template  # type: ignore
            template_tools = (
                convert_tools_for_template(request.tools) if request.tools else None
            )
        except Exception:
            template_tools = request.tools or None
        if hasattr(engine, "_apply_chat_template"):
            prompt = engine._apply_chat_template(
                messages,
                template_tools,
                chat_template_kwargs=request.chat_template_kwargs,
            )
        else:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        token_ids = list(tokenizer.encode(prompt))
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Failed to tokenize messages: {exc}"
        )

    total_tokens = len(token_ids)
    if total_tokens == 0:
        return {
            "model_id": request.model_id,
            "model_loaded": True,
            "total_tokens": 0,
            "block_size": block_size,
            "total_blocks": 0,
            "blocks_ssd_hot": 0,
            "blocks_ssd_disk": 0,
            "blocks_cold": 0,
            "ssd_hit_tokens": 0,
            "cold_tokens": 0,
        }

    # Compute chain-hashed block sequence.
    from ..cache.paged_cache import compute_block_hash

    model_name = getattr(paged_cache, "model_name", None) if paged_cache else None
    ssd_index = getattr(ssd_manager, "_index", None) if ssd_manager else None
    ssd_hot = getattr(ssd_manager, "_hot_cache", None) if ssd_manager else None

    # The cache is a contiguous prefix (each block chain-hashed from the
    # previous), so we walk block-by-block until the first retrievability
    # miss — after that, every subsequent block is necessarily cold.
    #
    # Ground truth for "cached" in paged-SSD mode is retrievability:
    # hot_cache (RAM copy) OR ssd_index (on disk). BlockAwarePrefixCache's
    # internal prefix index is deliberately NOT consulted — it tracks every
    # hash the scheduler has seen and isn't cleared by clear_ssd_cache(),
    # so relying on it would report false positives after a manual wipe.
    blocks_ssd_hot = 0
    blocks_ssd_disk = 0
    ssd_hit_tokens = 0

    parent_hash = b""
    total_blocks = (total_tokens + block_size - 1) // block_size

    for start in range(0, total_tokens, block_size):
        end = min(start + block_size, total_tokens)
        block_tokens = token_ids[start:end]
        if not block_tokens:
            break

        block_hash = compute_block_hash(
            parent_hash,
            block_tokens,
            extra_keys=None,
            model_name=model_name,
        )
        parent_hash = block_hash

        in_ssd_hot = ssd_hot is not None and block_hash in ssd_hot
        in_ssd_disk = False
        if ssd_index is not None:
            try:
                in_ssd_disk = ssd_index.contains(block_hash)
            except Exception:
                in_ssd_disk = False

        if not (in_ssd_hot or in_ssd_disk):
            break

        if in_ssd_hot:
            blocks_ssd_hot += 1
        else:
            blocks_ssd_disk += 1
        ssd_hit_tokens += len(block_tokens)

    cached_blocks = blocks_ssd_hot + blocks_ssd_disk
    blocks_cold = max(total_blocks - cached_blocks, 0)

    return {
        "model_id": request.model_id,
        "model_loaded": True,
        "total_tokens": total_tokens,
        "block_size": block_size,
        "total_blocks": total_blocks,
        "blocks_ssd_hot": blocks_ssd_hot,
        "blocks_ssd_disk": blocks_ssd_disk,
        "blocks_cold": blocks_cold,
        "ssd_hit_tokens": ssd_hit_tokens,
        "cold_tokens": max(total_tokens - ssd_hit_tokens, 0),
    }


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


@router.post("/api/hf/retry/{task_id}")
async def retry_hf_download(
    task_id: str,
    request: HFRetryRequest = HFRetryRequest(),
    is_admin: bool = Depends(require_admin),
):
    """Retry a failed or cancelled download, resuming from existing files."""
    if _hf_downloader is None:
        raise HTTPException(status_code=503, detail="Downloader not initialized")

    try:
        task = await _hf_downloader.retry_download(task_id, request.hf_token)
        return {"success": True, "task": task.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


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
async def get_recommended_models(
    mlx_only: bool = True,
    is_admin: bool = Depends(require_admin),
):
    """Get recommended models filtered by system memory."""
    if _hf_downloader is None:
        raise HTTPException(status_code=503, detail="Downloader not initialized")

    memory_info = get_system_memory_info()
    max_memory = memory_info["total_bytes"] or 16 * 1024**3

    from .hf_downloader import HFDownloader

    try:
        result = await HFDownloader.get_recommended_models(
            max_memory_bytes=max_memory, result_limit=50, mlx_only=mlx_only
        )
        return result
    except TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="HuggingFace API request timed out. The service may be temporarily unavailable.",
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/api/hf/search")
async def search_hf_models(
    q: str = "",
    sort: str = "trending",
    limit: int = 100,
    mlx_only: bool = True,
    is_admin: bool = Depends(require_admin),
):
    """Search HuggingFace models by query."""
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query parameter 'q' is required")

    from .hf_downloader import HFDownloader

    try:
        result = await HFDownloader.search_models(
            query=q.strip(),
            sort=sort,
            limit=min(limit, 100),
            mlx_only=mlx_only,
        )
        return result
    except TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="HuggingFace API request timed out. The service may be temporarily unavailable.",
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/api/hf/model-info")
async def get_hf_model_info(
    repo_id: str = "",
    is_admin: bool = Depends(require_admin),
):
    """Get detailed model information from HuggingFace."""
    if not repo_id.strip():
        raise HTTPException(
            status_code=400, detail="Query parameter 'repo_id' is required"
        )

    from huggingface_hub.utils import RepositoryNotFoundError

    from .hf_downloader import HFDownloader

    try:
        result = await HFDownloader.get_model_info(repo_id=repo_id.strip())
        return result
    except TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="HuggingFace API request timed out. The service may be temporarily unavailable.",
        )
    except RepositoryNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Model '{repo_id.strip()}' not found"
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/api/hf/models")
async def list_hf_models(is_admin: bool = Depends(require_admin)):
    """List models in all model directories with disk size info."""
    global_settings = _get_global_settings()
    if global_settings is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    model_dirs = global_settings.model.get_model_dirs(global_settings.base_path)

    from ..model_discovery import _resolve_hf_cache_entry

    def _add_model(model_path: Path, model_name: str) -> None:
        if model_name in seen_names:
            return
        seen_names.add(model_name)
        total_size = sum(
            f.stat().st_size for f in model_path.rglob("*") if f.is_file()
        )
        models.append(
            {
                "name": model_name,
                "path": str(model_path),
                "size": total_size,
                "size_formatted": format_size(total_size),
            }
        )

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
                _add_model(subdir, subdir.name)
            else:
                # HF Hub cache entry: models--Org--Name/snapshots/<hash>/
                hf_resolved = _resolve_hf_cache_entry(subdir)
                if hf_resolved is not None:
                    snapshot_path, model_name = hf_resolved
                    if (snapshot_path / "config.json").exists():
                        _add_model(snapshot_path, model_name)
                    continue

                # Level 2: organization folder — scan children
                for child in sorted(subdir.iterdir()):
                    if not child.is_dir() or child.name.startswith("."):
                        continue
                    if (child / "config.json").exists():
                        _add_model(child, child.name)

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
    # Handle macOS resource fork files (._*) that may disappear on non-native
    # filesystems (exFAT, NTFS). Use onexc (Python 3.12+) to avoid
    # DeprecationWarning, with onerror fallback for older versions.
    def _handle_onexc(func, path, exc):
        if isinstance(exc, FileNotFoundError) and Path(path).name.startswith("._"):
            logger.debug(f"Ignoring missing resource fork file: {path}")
            return
        raise exc

    def _handle_onerror(func, path, exc_info):
        if exc_info[0] == FileNotFoundError and Path(path).name.startswith("._"):
            logger.debug(f"Ignoring missing resource fork file: {path}")
            return
        raise exc_info[1].with_traceback(exc_info[2])

    try:
        if sys.version_info >= (3, 12):
            shutil.rmtree(model_path, onexc=_handle_onexc)
        else:
            shutil.rmtree(model_path, onerror=_handle_onerror)
        logger.info(f"Deleted model directory: {model_path}")
    except Exception as e:
        logger.error(f"Failed to delete model directory {model_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {e}")

    # If the model was inside an org folder (organized layout) and that
    # folder is now empty, drop it so the listing stays tidy.
    parent = model_path.parent
    if (
        parent != parent_model_dir
        and parent.exists()
        and not any(parent.iterdir())
    ):
        try:
            parent.rmdir()
            logger.info(f"Removed empty org folder: {parent}")
        except OSError as e:
            logger.debug(f"Could not remove empty org folder {parent}: {e}")

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
        if settings_manager:
            engine_pool.apply_settings_overrides(settings_manager)
        logger.info("Model pool refreshed after deletion")

    return {"success": True, "message": f"Model '{model_name}' deleted"}


# =============================================================================
# ModelScope Downloader API Routes
# =============================================================================


@router.get("/api/ms/status")
async def ms_status(is_admin: bool = Depends(require_admin)):
    """Check if ModelScope downloader is available."""
    return {"available": _ms_downloader is not None}


@router.post("/api/ms/download")
async def start_ms_download(
    request: MSDownloadRequest,
    is_admin: bool = Depends(require_admin),
):
    """Start downloading a model from ModelScope."""
    if _ms_downloader is None:
        raise HTTPException(status_code=503, detail="ModelScope downloader not initialized")

    try:
        task = await _ms_downloader.start_download(
            request.model_id, request.ms_token
        )
        return {"success": True, "task": task.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/api/ms/tasks")
async def list_ms_tasks(is_admin: bool = Depends(require_admin)):
    """List all ModelScope download tasks."""
    if _ms_downloader is None:
        raise HTTPException(status_code=503, detail="ModelScope downloader not initialized")

    return {"tasks": _ms_downloader.get_tasks()}


@router.post("/api/ms/cancel/{task_id}")
async def cancel_ms_download(
    task_id: str,
    is_admin: bool = Depends(require_admin),
):
    """Cancel an active ModelScope download."""
    if _ms_downloader is None:
        raise HTTPException(status_code=503, detail="ModelScope downloader not initialized")

    success = await _ms_downloader.cancel_download(task_id)
    if not success:
        raise HTTPException(
            status_code=404, detail="Task not found or not cancellable"
        )
    return {"success": True}


@router.post("/api/ms/retry/{task_id}")
async def retry_ms_download(
    task_id: str,
    request: MSRetryRequest = MSRetryRequest(),
    is_admin: bool = Depends(require_admin),
):
    """Retry a failed or cancelled ModelScope download."""
    if _ms_downloader is None:
        raise HTTPException(status_code=503, detail="ModelScope downloader not initialized")

    try:
        task = await _ms_downloader.retry_download(task_id, request.ms_token)
        return {"success": True, "task": task.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/api/ms/task/{task_id}")
async def remove_ms_task(
    task_id: str,
    is_admin: bool = Depends(require_admin),
):
    """Remove a completed, failed, or cancelled ModelScope task."""
    if _ms_downloader is None:
        raise HTTPException(status_code=503, detail="ModelScope downloader not initialized")

    success = _ms_downloader.remove_task(task_id)
    if not success:
        raise HTTPException(
            status_code=404, detail="Task not found or still active"
        )
    return {"success": True}


@router.get("/api/ms/recommended")
async def get_ms_recommended_models(
    mlx_only: bool = True,
    is_admin: bool = Depends(require_admin),
):
    """Get recommended models from ModelScope filtered by system memory."""
    if _ms_downloader is None:
        raise HTTPException(status_code=503, detail="ModelScope downloader not initialized")

    memory_info = get_system_memory_info()
    max_memory = memory_info["total_bytes"] or 16 * 1024**3

    from .ms_downloader import MSDownloader

    try:
        result = await MSDownloader.get_recommended_models(
            max_memory_bytes=max_memory, result_limit=50, mlx_only=mlx_only
        )
        return result
    except TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="ModelScope API request timed out. The service may be temporarily unavailable.",
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/api/ms/search")
async def search_ms_models(
    q: str = "",
    sort: str = "trending",
    limit: int = 100,
    mlx_only: bool = True,
    is_admin: bool = Depends(require_admin),
):
    """Search ModelScope models by query."""
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query parameter 'q' is required")

    from .ms_downloader import MSDownloader

    try:
        result = await MSDownloader.search_models(
            query=q.strip(),
            sort=sort,
            limit=min(limit, 100),
            mlx_only=mlx_only,
        )
        return result
    except TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="ModelScope API request timed out. The service may be temporarily unavailable.",
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/api/ms/model-info")
async def get_ms_model_info(
    model_id: str = "",
    is_admin: bool = Depends(require_admin),
):
    """Get detailed model information from ModelScope."""
    if not model_id.strip():
        raise HTTPException(
            status_code=400, detail="Query parameter 'model_id' is required"
        )

    from .ms_downloader import MSDownloader

    try:
        result = await MSDownloader.get_model_info(model_id=model_id.strip())
        return result
    except TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="ModelScope API request timed out. The service may be temporarily unavailable.",
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        if "NotExistError" in type(e).__name__ or "404" in str(e):
            raise HTTPException(
                status_code=404, detail=f"Model '{model_id.strip()}' not found"
            )
        raise HTTPException(status_code=502, detail=str(e))


# =============================================================================
# Accuracy Benchmark API Routes (MUST be before throughput {bench_id} routes)
# =============================================================================


@router.post("/api/bench/accuracy/queue/add")
async def add_to_accuracy_queue(
    request: Request,
    is_admin: bool = Depends(require_admin),
):
    """Add a model to the accuracy benchmark queue and start if idle."""
    from .accuracy_benchmark import (
        AccuracyBenchmarkRequest,
        add_to_queue,
        get_queue_status,
        start_next_from_queue,
    )

    engine_pool = _get_engine_pool()
    if engine_pool is None:
        raise HTTPException(status_code=503, detail="Engine pool not initialized")

    body = await request.json()
    try:
        bench_request = AccuracyBenchmarkRequest(**body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    entry = engine_pool.get_entry(bench_request.model_id)
    if entry is None:
        raise HTTPException(
            status_code=404, detail=f"Model not found: {bench_request.model_id}"
        )
    if entry.model_type not in ("llm", "vlm", None):
        raise HTTPException(
            status_code=400,
            detail=f"Model {bench_request.model_id} is not a supported model (type: {entry.model_type})",
        )

    add_to_queue(bench_request)

    logger.info(
        f"Accuracy queue: added {bench_request.model_id} "
        f"benchmarks={list(bench_request.benchmarks.keys())}"
    )

    # Start processing if not already running (synchronous — sets bench_id immediately)
    start_next_from_queue(engine_pool)

    return get_queue_status()


@router.get("/api/bench/accuracy/queue/status")
async def get_accuracy_queue_status(
    is_admin: bool = Depends(require_admin),
):
    """Get accuracy benchmark queue status."""
    from .accuracy_benchmark import get_queue_status

    return get_queue_status()


@router.delete("/api/bench/accuracy/queue/{idx}")
async def remove_from_accuracy_queue(
    idx: int,
    is_admin: bool = Depends(require_admin),
):
    """Remove an item from the accuracy benchmark queue."""
    from .accuracy_benchmark import get_queue_status, remove_from_queue

    if not remove_from_queue(idx):
        raise HTTPException(status_code=404, detail=f"Queue index {idx} not found")

    return get_queue_status()


@router.get("/api/bench/accuracy/results")
async def get_accumulated_accuracy_results(
    is_admin: bool = Depends(require_admin),
):
    """Get all accumulated accuracy benchmark results."""
    from .accuracy_benchmark import get_accumulated_results, get_queue_status

    status = get_queue_status()
    return {
        "results": get_accumulated_results(),
        "running": status["running"],
        "current_model": status["current_model"],
        "current_bench_id": status["current_bench_id"],
    }


@router.post("/api/bench/accuracy/results/reset")
async def reset_accuracy_results(
    is_admin: bool = Depends(require_admin),
):
    """Clear all accumulated accuracy benchmark results."""
    from .accuracy_benchmark import reset_accumulated_results

    reset_accumulated_results()
    return {"status": "reset"}


@router.post("/api/bench/accuracy/cancel")
async def cancel_accuracy_queue(
    is_admin: bool = Depends(require_admin),
):
    """Cancel the current run and clear the queue."""
    from .accuracy_benchmark import cancel_queue

    await cancel_queue()
    return {"status": "cancelled"}


@router.get("/api/bench/accuracy/{bench_id}/stream")
async def stream_accuracy_benchmark(
    bench_id: str,
    is_admin: bool = Depends(require_admin),
):
    """Stream accuracy benchmark progress via Server-Sent Events."""
    import json

    from fastapi.responses import StreamingResponse

    from .accuracy_benchmark import get_run

    run = get_run(bench_id)
    if run is None:
        raise HTTPException(
            status_code=404, detail=f"Accuracy benchmark not found: {bench_id}"
        )

    async def event_generator():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(run.queue.get(), timeout=60.0)
                except TimeoutError:
                    yield ": keepalive\n\n"
                    continue

                yield f"data: {json.dumps(event)}\n\n"

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


# =============================================================================
# Benchmark API Routes (Throughput)
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
    if entry.model_type not in ("llm", "vlm", None):
        raise HTTPException(
            status_code=400,
            detail=f"Model {bench_request.model_id} is not a supported model (type: {entry.model_type})",
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
                except TimeoutError:
                    # Send keepalive
                    yield ": keepalive\n\n"
                    continue

                yield f"data: {json.dumps(event)}\n\n"

                # Stop streaming on terminal events
                if event.get("type") in ("upload_done", "error"):
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


@router.get("/api/device-info")
async def get_device_info(
    is_admin: bool = Depends(require_admin),
):
    """Get device hardware info and owner_hash for omlx.ai integration."""
    from ..utils.hardware import (
        compute_owner_hash,
        get_chip_name,
        get_gpu_core_count,
        get_io_platform_uuid,
        get_total_memory_gb,
        parse_chip_info,
    )

    chip_string = get_chip_name()
    chip_name, chip_variant = parse_chip_info(chip_string)
    memory_gb = round(get_total_memory_gb())
    gpu_cores = get_gpu_core_count()

    owner_hash = None
    io_uuid = get_io_platform_uuid()
    if io_uuid:
        full_hash = compute_owner_hash(io_uuid, chip_name, gpu_cores, memory_gb)
        owner_hash = full_hash[:-1]  # Strip verify character for URL

    return {
        "chip_name": chip_name,
        "chip_variant": chip_variant,
        "memory_gb": memory_gb,
        "gpu_cores": gpu_cores,
        "owner_hash": owner_hash,
    }


# =============================================================================
# Update Check
# =============================================================================

_update_cache: dict[str, Any] | None = None
_update_cache_time: float = 0.0
_UPDATE_CACHE_TTL = 3600  # 1 hour


@router.get("/api/update-check")
async def check_update(
    is_admin: bool = Depends(require_admin),
):
    """Check GitHub Releases for newer oMLX version (cached 24h)."""
    global _update_cache, _update_cache_time

    now = time.time()
    if _update_cache is not None and now - _update_cache_time < _UPDATE_CACHE_TTL:
        return _update_cache

    no_update = {
        "update_available": False,
        "latest_version": None,
        "release_url": None,
    }

    try:
        # Use the releases list (not /releases/latest) and pick the highest
        # stable PEP 440 tag. Dev/rc tags here have historically been
        # published with the GitHub prerelease flag unset, which makes
        # /releases/latest return them as if they were stable.
        resp = await asyncio.to_thread(
            requests.get,
            "https://api.github.com/repos/jundot/omlx/releases",
            params={"per_page": 20},
            timeout=5,
        )
        if resp.status_code != 200:
            _update_cache = no_update
            _update_cache_time = now
            return _update_cache

        data = select_latest_stable_release(resp.json())
        if data is None:
            _update_cache = no_update
            _update_cache_time = now
            return _update_cache

        latest = data["tag_name"].lstrip("v")

        try:
            from packaging.version import Version

            update_available = Version(latest) > Version(_omlx_version)
        except Exception:
            update_available = False

        if update_available:
            _update_cache = {
                "update_available": True,
                "latest_version": latest,
                "release_url": data.get("html_url"),
            }
        else:
            _update_cache = no_update

        _update_cache_time = now
    except Exception:
        _update_cache = no_update
        _update_cache_time = now

    return _update_cache


# =============================================================================
# oQ Quantization API Routes
# =============================================================================


@router.get("/api/oq/models")
async def list_oq_models(is_admin: bool = Depends(require_admin)):
    """List non-quantized models available for oQ quantization."""
    if _oq_manager is None:
        raise HTTPException(
            status_code=503, detail="oQ quantizer not initialized"
        )
    source_models, all_models = await _oq_manager.list_quantizable_models()
    return {"models": source_models, "all_models": all_models}


@router.get("/api/oq/estimate")
async def estimate_oq(
    model_path: str,
    oq_level: float,
    preserve_mtp: bool = False,
    is_admin: bool = Depends(require_admin),
):
    """Estimate effective bpw and output size for a model at given oQ level."""
    from ..oq import estimate_bpw_and_size

    try:
        result = await asyncio.to_thread(
            estimate_bpw_and_size,
            model_path,
            oq_level,
            64,  # group_size (default)
            preserve_mtp,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/oq/start")
async def start_oq_quantization(
    request: OQStartRequest,
    is_admin: bool = Depends(require_admin),
):
    """Start an oQ quantization task."""
    if _oq_manager is None:
        raise HTTPException(
            status_code=503, detail="oQ quantizer not initialized"
        )
    if request.oq_level not in (2, 3, 3.5, 4, 5, 6, 8):
        raise HTTPException(
            status_code=400,
            detail="Invalid oQ level. Must be 2, 3, 4, 5, 6, or 8",
        )
    if request.dtype not in ("bfloat16", "float16"):
        raise HTTPException(
            status_code=400,
            detail="Invalid dtype. Must be 'bfloat16' or 'float16'",
        )
    is_paro, _ = _paroquant_compat_for_model({"model_path": request.model_path})
    if is_paro:
        raise HTTPException(
            status_code=400,
            detail=(
                "Model is already quantized with paroquant; "
                "oQ re-quantization is not supported"
            ),
        )
    try:
        task = await _oq_manager.start_quantization(
            model_path=request.model_path,
            oq_level=request.oq_level,
            group_size=request.group_size,
            sensitivity_model_path=request.sensitivity_model_path,
            text_only=request.text_only,
            dtype=request.dtype,
            preserve_mtp=request.preserve_mtp,
            auto_proxy_sensitivity=request.auto_proxy_sensitivity,
        )
        return {"success": True, "task": task.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/api/oq/tasks")
async def list_oq_tasks(is_admin: bool = Depends(require_admin)):
    """List all quantization tasks."""
    if _oq_manager is None:
        raise HTTPException(
            status_code=503, detail="oQ quantizer not initialized"
        )
    return {"tasks": _oq_manager.get_tasks()}


@router.post("/api/oq/cancel/{task_id}")
async def cancel_oq_task(
    task_id: str, is_admin: bool = Depends(require_admin)
):
    """Cancel an active quantization task."""
    if _oq_manager is None:
        raise HTTPException(
            status_code=503, detail="oQ quantizer not initialized"
        )
    success = await _oq_manager.cancel_quantization(task_id)
    if not success:
        raise HTTPException(
            status_code=404, detail="Task not found or not cancellable"
        )
    return {"success": True}


@router.delete("/api/oq/task/{task_id}")
async def remove_oq_task(
    task_id: str, is_admin: bool = Depends(require_admin)
):
    """Remove a completed/failed/cancelled task."""
    if _oq_manager is None:
        raise HTTPException(
            status_code=503, detail="oQ quantizer not initialized"
        )
    success = _oq_manager.remove_task(task_id)
    if not success:
        raise HTTPException(
            status_code=404, detail="Task not found or still active"
        )
    return {"success": True}


# =============================================================================
# HuggingFace Upload Endpoints
# =============================================================================


@router.post("/api/upload/validate-token")
async def validate_upload_token(
    request: HFValidateTokenRequest,
    is_admin: bool = Depends(require_admin),
):
    """Validate a HuggingFace token and return user info."""
    if _hf_uploader is None:
        raise HTTPException(
            status_code=503, detail="HF Uploader not initialized"
        )
    try:
        result = await _hf_uploader.validate_token(request.hf_token)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/api/upload/oq-models")
async def list_upload_oq_models(is_admin: bool = Depends(require_admin)):
    """List local oQ models available for upload."""
    if _hf_uploader is None:
        raise HTTPException(
            status_code=503, detail="HF Uploader not initialized"
        )
    oq_models = await _hf_uploader.list_oq_models()
    all_models = await _hf_uploader.list_all_models()
    return {"oq_models": oq_models, "all_models": all_models}


@router.post("/api/upload/start")
async def start_upload(
    request: HFUploadRequest,
    is_admin: bool = Depends(require_admin),
):
    """Start an upload task to HuggingFace Hub."""
    if _hf_uploader is None:
        raise HTTPException(
            status_code=503, detail="HF Uploader not initialized"
        )
    try:
        task = await _hf_uploader.start_upload(
            model_path=request.model_path,
            repo_id=request.repo_id,
            token=request.hf_token,
            readme_source_path=request.readme_source_path,
            auto_readme=request.auto_readme,
            redownload_notice=request.redownload_notice,
            private=request.private,
        )
        return {"success": True, "task": task.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/api/upload/tasks")
async def list_upload_tasks(is_admin: bool = Depends(require_admin)):
    """List all upload tasks."""
    if _hf_uploader is None:
        raise HTTPException(
            status_code=503, detail="HF Uploader not initialized"
        )
    return {"tasks": _hf_uploader.get_tasks()}


@router.post("/api/upload/cancel/{task_id}")
async def cancel_upload_task(
    task_id: str, is_admin: bool = Depends(require_admin)
):
    """Cancel an active or pending upload task."""
    if _hf_uploader is None:
        raise HTTPException(
            status_code=503, detail="HF Uploader not initialized"
        )
    success = await _hf_uploader.cancel_upload(task_id)
    if not success:
        raise HTTPException(
            status_code=404, detail="Task not found or not cancellable"
        )
    return {"success": True}


@router.delete("/api/upload/task/{task_id}")
async def remove_upload_task(
    task_id: str, is_admin: bool = Depends(require_admin)
):
    """Remove a completed/failed/cancelled upload task."""
    if _hf_uploader is None:
        raise HTTPException(
            status_code=503, detail="HF Uploader not initialized"
        )
    success = _hf_uploader.remove_task(task_id)
    if not success:
        raise HTTPException(
            status_code=404, detail="Task not found or still active"
        )
    return {"success": True}
