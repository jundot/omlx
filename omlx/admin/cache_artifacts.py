# SPDX-License-Identifier: Apache-2.0
"""Admin API: export/import paged-SSD cache artifacts (issue #3612).

An artifact is a zip of the SSD blocks (and GDN sidecars) behind a prompt
prefix; an agent stores it next to its own session and hands it back to
oMLX after a restart instead of paying a full re-prefill. Core logic and the
artifact format live in ``omlx/cache/artifact_store.py``.

Both endpoints are admin-only, single-node, and additive: they never touch
cache matching, admission or the scheduler, only read the SSD manager's
index and register verified files back into it.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..cache import artifact_store
from ..cache.artifact_store import ArtifactError
from .auth import require_admin
from .routes import _normalize_probe_tool_calls, _probe_chat_template_kwargs

logger = logging.getLogger(__name__)

router = APIRouter()

_get_engine_pool = None
_get_settings_manager = None


def set_cache_artifact_getters(
    pool_getter, settings_manager_getter
) -> None:
    """Wire server getters (same pattern as ``set_admin_getters``)."""
    global _get_engine_pool, _get_settings_manager
    _get_engine_pool = pool_getter
    _get_settings_manager = settings_manager_getter


class CacheArtifactExportRequest(BaseModel):
    """Export the restorable prefix behind this conversation into a zip.

    ``messages``/``tools`` are rendered and tokenized exactly like a real
    turn (same path as ``/api/cache/probe``) so the block chain lines up
    with what a re-prefill would build.
    """

    model_id: str
    messages: list[dict]
    tools: list[dict] | None = None
    chat_template_kwargs: dict | None = None
    thinking_budget: int | None = None
    # Where to write the artifact. Defaults to a ``cache_artifacts``
    # directory next to the SSD cache dir.
    output_dir: str | None = None


class CacheArtifactImportRequest(BaseModel):
    """Validate and register a previously exported artifact."""

    model_id: str
    artifact_path: str


def _resolve_context(model_id: str) -> tuple:
    """Engine + SSD-manager plumbing, mirroring ``probe_cache`` in routes.py.

    Returns (engine, entry, ssd_manager, model_name, block_size, tokenizer).
    """
    engine_pool = _get_engine_pool()
    if engine_pool is None:
        raise HTTPException(status_code=503, detail="Engine pool not initialized")

    entry = engine_pool._entries.get(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    if entry.engine is None:
        raise HTTPException(
            status_code=409,
            detail=f"Model is not loaded — load it to use cache artifacts: {model_id}",
        )

    engine = entry.engine
    tokenizer = getattr(engine, "_tokenizer", None)
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        raise HTTPException(
            status_code=400,
            detail="Model tokenizer does not support chat templating.",
        )

    async_core = getattr(engine, "_engine", None)
    core = getattr(async_core, "engine", None) if async_core is not None else None
    scheduler = getattr(core, "scheduler", None) if core is not None else None
    if scheduler is None:
        raise HTTPException(
            status_code=500, detail="Scheduler unavailable for loaded model."
        )

    ssd_manager = getattr(scheduler, "paged_ssd_cache_manager", None)
    if ssd_manager is None:
        raise HTTPException(
            status_code=409,
            detail="Paged SSD cache unavailable — the cache may not be enabled.",
        )
    paged_cache = getattr(scheduler, "paged_cache_manager", None)
    prefix_cache = getattr(scheduler, "block_aware_cache", None)
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
    model_name = (
        getattr(paged_cache, "model_name", None)
        if paged_cache is not None
        else None
    ) or model_id
    return engine, entry, ssd_manager, model_name, block_size, tokenizer


def _require_ssd_tier(ssd_manager) -> None:
    if getattr(ssd_manager, "hot_cache_only", False):
        raise HTTPException(
            status_code=409,
            detail=(
                "SSD cache tier is disabled (cache.hot_cache_only=true) — "
                "blocks are RAM-only and not portable. Enable the SSD tier "
                "first."
            ),
        )


def _tokenize_prompt(
    engine, entry, tokenizer, request: CacheArtifactExportRequest
) -> list[int]:
    """Render + tokenize exactly like ``probe_cache`` (keep in sync)."""
    try:
        messages = _normalize_probe_tool_calls(request.messages)
        if hasattr(engine, "_preprocess_messages"):
            messages = engine._preprocess_messages(messages)
        try:
            from ..api.tool_calling import convert_tools_for_template  # type: ignore

            template_tools = (
                convert_tools_for_template(request.tools)
                if request.tools
                else None
            )
        except Exception:
            template_tools = request.tools or None
        if hasattr(engine, "_apply_chat_template"):
            prompt = engine._apply_chat_template(
                messages,
                template_tools,
                chat_template_kwargs=_probe_chat_template_kwargs(
                    request,
                    preserve_thinking_default=getattr(
                        entry, "preserve_thinking_default", None
                    ),
                ),
            )
        else:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return list(tokenizer.encode(prompt))
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Failed to tokenize messages: {exc}"
        ) from exc


@router.post("/api/cache/artifact/export")
def export_cache_artifact(
    request: CacheArtifactExportRequest,
    is_admin: bool = Depends(require_admin),
):
    """Write the longest restorable prefix of the conversation to a zip.

    The walk stops at the first block that is not durable on disk, so the
    artifact always holds a contiguous, importable chain. Returns the
    artifact path, size and manifest (block list, token counts).
    """
    engine, entry, ssd_manager, model_name, block_size, tokenizer = (
        _resolve_context(request.model_id)
    )
    _require_ssd_tier(ssd_manager)
    token_ids = _tokenize_prompt(engine, entry, tokenizer, request)
    if not token_ids:
        raise HTTPException(
            status_code=400, detail="Messages tokenized to zero tokens."
        )

    if request.output_dir:
        output_dir = Path(request.output_dir).expanduser()
    else:
        output_dir = Path(ssd_manager._cache_dir).parent / "cache_artifacts"

    try:
        result = artifact_store.export_artifact(
            ssd_manager,
            model_name=model_name,
            token_ids=token_ids,
            block_size=block_size,
            output_dir=output_dir,
        )
    except OSError as exc:
        raise HTTPException(
            status_code=500, detail=f"Cache artifact export failed: {exc}"
        ) from exc

    if result["manifest"]["restorable_tokens"] == 0:
        import os

        with contextlib.suppress(OSError):
            os.unlink(result["artifact_path"])
        raise HTTPException(
            status_code=409,
            detail=(
                "No restorable prefix found — the SSD cache holds nothing "
                "for this prompt (not yet prefilled, evicted, or the SSD "
                "tier was off when it ran)."
            ),
        )
    return result


@router.post("/api/cache/artifact/import")
def import_cache_artifact(
    request: CacheArtifactImportRequest,
    is_admin: bool = Depends(require_admin),
):
    """Validate an artifact and register its blocks with the SSD cache.

    Fails closed with 422 on any mismatch (model, format, integrity,
    compatibility). A fully imported artifact makes the prompt prefix
    restorable without a re-prefill.
    """
    _engine, _entry, ssd_manager, model_name, _block_size, _tokenizer = (
        _resolve_context(request.model_id)
    )
    _require_ssd_tier(ssd_manager)

    artifact_path = Path(request.artifact_path).expanduser()
    if not artifact_path.is_absolute():
        raise HTTPException(
            status_code=422, detail="artifact_path must be an absolute path."
        )
    if not artifact_path.is_file():
        raise HTTPException(
            status_code=404, detail=f"Artifact not found: {artifact_path}"
        )

    try:
        result = artifact_store.import_artifact(
            ssd_manager, artifact_path=artifact_path, model_name=model_name
        )
    except ArtifactError as exc:
        raise HTTPException(status_code=422, detail=f"Artifact rejected: {exc}") from exc
    return result
