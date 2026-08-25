# SPDX-License-Identifier: Apache-2.0
"""ModelScope model downloader for oMLX admin panel.

Downloads models from ModelScope Hub using the modelscope SDK's snapshot_download
with directory-size-based progress polling.
"""

import asyncio
import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

import requests

from .hf_downloader import (
    DownloadStatus,
    DownloadTask,
    _format_model_size,
    _format_param_count,
)

logger = logging.getLogger(__name__)

# Check if modelscope SDK is available
MS_SDK_AVAILABLE = False
try:
    from modelscope import snapshot_download as ms_snapshot_download
    from modelscope.hub.api import HubApi as MSHubApi

    MS_SDK_AVAILABLE = True
except ImportError:
    ms_snapshot_download = None  # type: ignore[assignment]
    MSHubApi = None  # type: ignore[assignment, misc]

# Timeout for ModelScope API calls (seconds).
_MS_API_TIMEOUT = 15

# Seconds with no download progress before considering the download stalled.
_STALL_TIMEOUT = 300

# Default ModelScope API base URL.
_DEFAULT_MS_ENDPOINT = "https://modelscope.cn"

# Minimum downloads to be included in recommendations.
_MIN_DOWNLOADS = 50

# ModelScope's public OpenAPI names and current web UI sort values.
_MS_OPENAPI_SORT_MAP = {
    "trending": "default",
    "downloads": "downloads",
    "likes": "likes",
}
_MS_WEB_SORT_MAP = {
    "trending": "Default",
    "downloads": "DownloadsCount",
    "likes": "StarsCount",
}
_MS_EXPERIENCE_FILTERS = frozenset(
    {"api_inference", "model_demo", "restful_inference"}
)

_MS_DOMAIN_LABELS = {
    "cv": "Computer Vision",
    "nlp": "Natural Language Processing",
    "audio": "Audio",
    "multi-modal": "Multimodal",
    "scientific-computing": "Scientific Computing",
}
_MS_DOMAIN_ORDER = tuple(_MS_DOMAIN_LABELS)
_MS_POPULAR_TASKS = (
    "text-generation",
    "text-to-image-synthesis",
    "text-to-speech",
)


def _get_ms_endpoint() -> str:
    """Get the configured ModelScope endpoint URL."""
    # Check environment variable first (set by CLI/settings)
    endpoint = os.environ.get("MODELSCOPE_DOMAIN", "")
    if endpoint:
        return endpoint.rstrip("/")

    try:
        from ..settings import get_settings

        endpoint = get_settings().modelscope.endpoint
        if endpoint:
            return endpoint.rstrip("/")
    except (RuntimeError, AttributeError):
        pass

    return _DEFAULT_MS_ENDPOINT


def _get_ms_api():
    """Create ModelScope HubApi instance.

    Returns:
        HubApi instance or None if SDK not available.
    """
    if not MS_SDK_AVAILABLE:
        return None
    return MSHubApi()


def _extract_model_size_from_files(file_list: list) -> int:
    """Calculate total file size from a list of file metadata dicts."""
    total = 0
    for f in file_list:
        size = f.get("Size") or f.get("size") or 0
        if isinstance(size, (int, float)):
            total += int(size)
    return total


# ---------------------------------------------------------------------------
# Per-model enrichment (size + param count)
#
# ModelScope's list_models endpoint returns Path/Name/Downloads/Likes/Stars
# but rarely populates StorageSize, and never returns a parameter count.
# To match HuggingFace's recommended-models card data, we enrich each entry
# with a config.json fetch (for params) and — when StorageSize was missing —
# a model-detail fetch (for size).
#
# Cached in-process for 24 hours since config.json content for a model
# doesn't change in practice; this keeps subsequent page loads of the
# Downloads tab essentially free.

_ENRICH_CACHE: dict[str, tuple[float, dict]] = {}
_ENRICH_CACHE_TTL = 24 * 3600  # 24h — config.json is effectively immutable
_ENRICH_CACHE_MAX = 1024       # bound memory under aggressive search/list use
_ENRICH_CONCURRENCY = 8        # parallel fetches per recommended/search call


def _enrich_cache_get(model_id: str) -> Optional[dict]:
    entry = _ENRICH_CACHE.get(model_id)
    if entry is None:
        return None
    ts, data = entry
    if time.time() - ts > _ENRICH_CACHE_TTL:
        _ENRICH_CACHE.pop(model_id, None)
        return None
    return data


def _enrich_cache_put(model_id: str, data: dict) -> None:
    if len(_ENRICH_CACHE) >= _ENRICH_CACHE_MAX:
        # Drop the oldest entry. O(N) on eviction but N is bounded at MAX
        # and evictions are rare in practice (24h TTL >> page-load rate).
        oldest = min(_ENRICH_CACHE, key=lambda k: _ENRICH_CACHE[k][0])
        _ENRICH_CACHE.pop(oldest, None)
    _ENRICH_CACHE[model_id] = (time.time(), data)


def _estimate_params_from_config(config: Optional[dict]) -> int:
    """Estimate decoder-transformer parameter count from a HF-style config.

    Handles dense Llama/Qwen/Mistral families and MoE variants
    (num_local_experts / num_experts). Returns 0 when required fields are
    missing — caller should render a blank rather than display a wrong
    number. The estimate is intentionally a rough headline figure (≈±5%);
    the goal is to surface "~7B" vs "~14B", not to match the checkpoint
    byte-for-byte.
    """
    if not isinstance(config, dict):
        return 0
    try:
        vocab_size = int(config.get("vocab_size", 0))
        hidden_size = int(config.get("hidden_size", 0))
        num_layers = int(config.get("num_hidden_layers", 0))
    except (TypeError, ValueError):
        return 0

    if not (vocab_size and hidden_size and num_layers):
        return 0

    try:
        intermediate_size = int(config.get("intermediate_size", 0))
        num_heads = int(config.get("num_attention_heads", 0))
        num_kv = int(config.get("num_key_value_heads", num_heads))
        head_dim = int(config.get("head_dim", 0)) or (
            hidden_size // num_heads if num_heads else 0
        )
        num_experts = int(
            config.get("num_local_experts")
            or config.get("num_experts")
            or 1
        )
        tie_embeddings = bool(config.get("tie_word_embeddings", True))
    except (TypeError, ValueError):
        return 0

    embeddings = vocab_size * hidden_size

    # Attention: Q + O are full hidden_size; K + V are reduced for GQA.
    if num_heads and head_dim:
        attn = (
            2 * hidden_size * (num_heads * head_dim)
            + 2 * hidden_size * (num_kv * head_dim)
        )
    else:
        attn = 4 * hidden_size * hidden_size

    # Gated MLP (Llama/Qwen style): gate + up + down projections.
    # MoE multiplies the FFN by the number of experts.
    if intermediate_size:
        ffn = num_experts * 3 * hidden_size * intermediate_size
    else:
        ffn = 8 * hidden_size * hidden_size

    layer_norms = 2 * hidden_size
    per_layer = attn + ffn + layer_norms

    total = embeddings + num_layers * per_layer + hidden_size
    if not tie_embeddings:
        total += vocab_size * hidden_size  # untied LM head

    return total


async def _fetch_model_config(model_id: str) -> Optional[dict]:
    """Fetch and parse a model's config.json from ModelScope.

    Returns None on any error (network, non-200, non-JSON) so callers can
    treat the field as absent without raising.
    """
    if not model_id:
        return None
    import json

    endpoint = _get_ms_endpoint()
    url = (
        f"{endpoint}/api/v1/models/{model_id}/repo"
        f"?FilePath=config.json&Revision=master"
    )
    try:
        resp = await asyncio.wait_for(
            asyncio.to_thread(requests.get, url, timeout=_MS_API_TIMEOUT),
            timeout=_MS_API_TIMEOUT + 5,
        )
        if resp.status_code != 200:
            return None
        return json.loads(resp.text)
    except Exception as e:
        logger.debug(f"config.json fetch failed for {model_id}: {e}")
        return None


async def _fetch_model_detail_size(model_id: str) -> int:
    """Fetch a model's storage size via the detail endpoint.

    Used as a fallback when list_models didn't populate StorageSize.
    Prefers ModelInfos.safetensor.model_size (weights only) and falls
    back to the repository StorageSize (weights + tokenizer + readme).
    Returns 0 on any error.
    """
    if not model_id:
        return 0
    endpoint = _get_ms_endpoint()
    url = f"{endpoint}/api/v1/models/{model_id}"
    try:
        resp = await asyncio.wait_for(
            asyncio.to_thread(requests.get, url, timeout=_MS_API_TIMEOUT),
            timeout=_MS_API_TIMEOUT + 5,
        )
        if resp.status_code != 200:
            return 0
        data = resp.json().get("Data") or {}
        model_infos = data.get("ModelInfos") or {}
        st = model_infos.get("safetensor") or {}
        size = st.get("model_size") or data.get("StorageSize") or 0
        return int(size) if isinstance(size, (int, float, str)) else 0
    except (TypeError, ValueError):
        return 0
    except Exception as e:
        logger.debug(f"model detail fetch failed for {model_id}: {e}")
        return 0


async def _enrich_ms_entry(entry: dict, sem: asyncio.Semaphore) -> dict:
    """Add size + params to a parsed model entry.

    Concurrent fetches are gated by `sem`; per-model results are cached
    in-process for 24h so subsequent page loads don't re-issue requests.
    Mutates and returns the same dict for ergonomic gather() pipelines.
    """
    model_id = entry.get("repo_id") or ""
    if not model_id:
        return entry

    cached = _enrich_cache_get(model_id)
    if cached is not None:
        c_size = cached.get("size") or 0
        c_params = cached.get("params") or 0
        if c_size and not entry.get("size"):
            entry["size"] = c_size
            entry["size_formatted"] = _format_model_size(c_size)
        if c_params:
            entry["params"] = c_params
            entry["params_formatted"] = _format_param_count(c_params)
        return entry

    async with sem:
        config_task = asyncio.create_task(_fetch_model_config(model_id))
        need_size = (entry.get("size") or 0) <= 0
        detail_task = (
            asyncio.create_task(_fetch_model_detail_size(model_id))
            if need_size else None
        )

        config = await config_task
        params = _estimate_params_from_config(config)

        size = entry.get("size") or 0
        if detail_task is not None:
            size = await detail_task

    _enrich_cache_put(model_id, {"size": size, "params": params})

    if size and not entry.get("size"):
        entry["size"] = size
        entry["size_formatted"] = _format_model_size(size)
    if params:
        entry["params"] = params
        entry["params_formatted"] = _format_param_count(params)
    return entry


def _parse_ms_model_entry(entry: dict) -> dict:
    """Parse a ModelScope API model entry into a normalized dict.

    Args:
        entry: Raw model dict from ModelScope API.

    Returns:
        Normalized model dict matching the HF format.
    """
    # Path is the organization/owner, Name is the model name
    # repo_id should be "owner/model" format
    path = entry.get("Path") or ""
    name = entry.get("Name") or ""
    if path and name:
        model_id = f"{path}/{name}"
    elif name:
        model_id = name
    else:
        model_id = path

    downloads = entry.get("Downloads") or 0
    likes = entry.get("Likes") or entry.get("Stars") or 0
    # StorageSize is the total size in bytes
    size = entry.get("StorageSize") or 0

    return {
        "repo_id": model_id,
        "name": name or model_id.split("/")[-1],
        "downloads": downloads,
        "likes": likes,
        "trending_score": 0,
        "size": size,
        "size_formatted": _format_model_size(size) if size > 0 else "",
        "params": None,
        "params_formatted": None,
    }


def _parse_ms_openapi_entry(entry: dict) -> dict:
    """Normalize one model from ModelScope's public OpenAPI."""
    model_id = entry.get("id") or ""
    name = entry.get("display_name") or model_id.split("/")[-1]
    size = entry.get("file_size") or 0
    params = entry.get("params") or 0

    return {
        "repo_id": model_id,
        "name": name,
        "downloads": entry.get("downloads") or 0,
        "likes": entry.get("likes") or 0,
        "trending_score": 0,
        "size": size,
        "size_formatted": _format_model_size(size) if size > 0 else "",
        "params": params if params > 0 else None,
        "params_formatted": (
            _format_param_count(params) if params > 0 else None
        ),
    }


def _build_ms_web_browse_payload(
    sort: str,
    page_size: int,
    mlx_only: bool,
    experiences: tuple[str, ...] = (),
    task: str = "",
) -> dict:
    """Build the request body used by ModelScope's current model browser."""
    criterion: list[dict] = []
    single_criterion: list[dict] = []

    if mlx_only:
        criterion.append(
            {
                "category": "organizations",
                "predicate": "contains",
                "values": ["mlx-community"],
            }
        )

    if task:
        parent, separator, child = task.partition(":")
        criterion.append(
            {
                "category": "tasks",
                "predicate": "contains",
                "values": [parent] if parent else [],
                "sub_values": [child] if separator and child else [],
            }
        )

    selected = set(experiences) & _MS_EXPERIENCE_FILTERS
    if "api_inference" in selected:
        single_criterion.append(
            {
                "category": "inference_type",
                "DateType": "int",
                "predicate": "equal",
                "IntValue": 1,
            }
        )
    if "model_demo" in selected:
        criterion.append(
            {
                "category": "demo_service",
                "predicate": "contains",
                "values": ['{"enabled":true}'],
            }
        )
        single_criterion.append(
            {
                "category": "support_experience",
                "DateType": "int",
                "predicate": "equal",
                "IntValue": 1,
            }
        )
    if "restful_inference" in selected:
        single_criterion.append(
            {
                "category": "support_api_inference",
                "DateType": "int",
                "predicate": "equal",
                "IntValue": 1,
            }
        )

    return {
        "PageSize": min(max(page_size, 1), 50),
        "PageNumber": 1,
        "SortBy": _MS_WEB_SORT_MAP.get(sort, "Default"),
        "Criterion": criterion,
        "SingleCriterion": single_criterion,
    }


async def _fetch_ms_models_openapi(
    sort: str,
    page_size: int,
    mlx_only: bool,
    task: str = "",
) -> tuple[list[dict], int]:
    """Fetch a sorted ModelScope page through the documented OpenAPI."""
    endpoint = _get_ms_endpoint()
    url = f"{endpoint}/openapi/v1/models"
    params: dict[str, object] = {
        "sort": _MS_OPENAPI_SORT_MAP.get(sort, "default"),
        "page_number": 1,
        "page_size": min(max(page_size, 1), 50),
    }
    if mlx_only:
        params["owner"] = "mlx-community"
    if task:
        params["filter.task"] = task.split(":")[-1]

    try:
        resp = await asyncio.wait_for(
            asyncio.to_thread(
                requests.get, url, params=params, timeout=_MS_API_TIMEOUT
            ),
            timeout=_MS_API_TIMEOUT + 5,
        )
        if resp.status_code == 200:
            data = resp.json().get("data") or {}
            models = data.get("models") or []
            return models, int(data.get("total_count") or len(models))
        logger.warning(
            "ModelScope OpenAPI browse failed with status %s", resp.status_code
        )
    except Exception as e:
        logger.warning(f"ModelScope OpenAPI browse failed: {e}")
    return [], 0


async def _fetch_ms_models_web(payload: dict) -> tuple[list[dict], int]:
    """Fetch a page through the endpoint used by ModelScope's web browser."""
    endpoint = _get_ms_endpoint()
    url = f"{endpoint}/api/v1/dolphin/models"
    try:
        resp = await asyncio.wait_for(
            asyncio.to_thread(
                requests.put, url, json=payload, timeout=_MS_API_TIMEOUT
            ),
            timeout=_MS_API_TIMEOUT + 5,
        )
        if resp.status_code == 200:
            body = resp.json()
            if body.get("Code", 200) != 200:
                logger.warning(
                    "ModelScope web browse failed: %s", body.get("Message", "")
                )
                return [], 0
            data = ((body.get("Data") or {}).get("Model") or {})
            models = data.get("Models") or []
            return models, int(data.get("TotalCount") or len(models))
        logger.warning(
            "ModelScope web browse failed with status %s", resp.status_code
        )
    except Exception as e:
        logger.warning(f"ModelScope web browse failed: {e}")
    return [], 0


def _format_ms_task_label(value: str) -> str:
    """Turn a ModelScope task identifier into a compact English label."""
    special = {
        "3d": "3D",
        "nlp": "NLP",
        "nli": "NLI",
        "ocr": "OCR",
    }
    words = value.replace("_", "-").split("-")
    return " ".join(special.get(word.lower(), word.capitalize()) for word in words)


async def _fetch_ms_task_groups() -> list[dict]:
    """Fetch ModelScope's current top-level task taxonomy."""
    endpoint = _get_ms_endpoint()
    url = f"{endpoint}/api/v1/tasks"
    try:
        resp = await asyncio.wait_for(
            asyncio.to_thread(requests.get, url, timeout=_MS_API_TIMEOUT),
            timeout=_MS_API_TIMEOUT + 5,
        )
        if resp.status_code != 200:
            return []
        domains = (resp.json().get("Data") or {}).get("Domains") or []
    except Exception as e:
        logger.warning(f"ModelScope task taxonomy fetch failed: {e}")
        return []

    groups = []
    task_lookup: dict[str, dict] = {}
    for domain in domains:
        domain_id = domain.get("DomainName") or ""
        tasks = []
        for task in domain.get("Tasks") or []:
            value = task.get("Name") or ""
            if not value or task.get("IsRetrieval") is False:
                continue
            item = {"value": value, "label": _format_ms_task_label(value)}
            tasks.append(item)
            task_lookup[value] = item
        if tasks:
            groups.append(
                {
                    "id": domain_id,
                    "label": _MS_DOMAIN_LABELS.get(
                        domain_id, _format_ms_task_label(domain_id)
                    ),
                    "tasks": tasks,
                }
            )

    order = {domain_id: index for index, domain_id in enumerate(_MS_DOMAIN_ORDER)}
    groups.sort(key=lambda group: order.get(group["id"], len(order)))
    popular = [task_lookup[value] for value in _MS_POPULAR_TASKS if value in task_lookup]
    if popular:
        groups.insert(0, {"id": "popular", "label": "Popular", "tasks": popular})
    return groups


async def _fetch_ms_models_rest(
    query: str = "",
    page_size: int = 200,
) -> list[dict]:
    """Fetch models from ModelScope REST API without org restriction.

    Used when mlx_only is disabled to search across all organizations.

    Args:
        query: Optional search query to filter by model name.
        page_size: Number of models to fetch.

    Returns:
        List of raw model entry dicts from the API response.
    """
    endpoint = _get_ms_endpoint()
    url = f"{endpoint}/api/v1/models/"
    payload: dict = {"PageSize": page_size}
    if query:
        payload["Name"] = query
    try:
        resp = await asyncio.wait_for(
            asyncio.to_thread(
                requests.put, url, json=payload, timeout=_MS_API_TIMEOUT
            ),
            timeout=_MS_API_TIMEOUT + 5,
        )
        if resp.status_code == 200:
            data = resp.json().get("Data", {})
            return data.get("Models", data.get("models", []))
    except Exception as e:
        logger.warning(f"ModelScope REST API fetch failed: {e}")
    return []


class MSDownloader:
    """Manages ModelScope model downloads with progress tracking.

    Uses modelscope.snapshot_download() for actual downloads and polls
    the target directory size to estimate progress.

    Args:
        model_dir: Directory where downloaded models are stored.
        on_complete: Async callback invoked when a download completes successfully.
    """

    @staticmethod
    async def browse_models(
        max_memory_bytes: int,
        sort: str = "trending",
        limit: int = 50,
        mlx_only: bool = True,
        experiences: tuple[str, ...] = (),
        task: str = "",
    ) -> dict:
        """Browse ModelScope models with native sorting and filters.

        The documented OpenAPI handles the normal sort, owner, and task path.
        ModelScope's three Experience controls are only implemented by the
        endpoint used by the current web app, so filtered requests use that
        endpoint and retain its server ordering.
        """
        result_limit = min(max(limit, 1), 50)
        selected_experiences = tuple(
            value for value in experiences if value in _MS_EXPERIENCE_FILTERS
        )

        if selected_experiences:
            payload = _build_ms_web_browse_payload(
                sort=sort,
                page_size=50,
                mlx_only=mlx_only,
                experiences=selected_experiences,
                task=task,
            )
            raw_models, upstream_total = await _fetch_ms_models_web(payload)
            models = [_parse_ms_model_entry(entry) for entry in raw_models]
            needs_enrichment = True
        else:
            raw_models, upstream_total = await _fetch_ms_models_openapi(
                sort=sort,
                page_size=50,
                mlx_only=mlx_only,
                task=task,
            )
            models = [_parse_ms_openapi_entry(entry) for entry in raw_models]
            needs_enrichment = False

        models = [
            model
            for model in models
            if model.get("repo_id")
            and model.get("downloads", 0) >= _MIN_DOWNLOADS
            and (
                model.get("size", 0) == 0
                or model.get("size", 0) <= max_memory_bytes
            )
        ]

        if needs_enrichment and models:
            sem = asyncio.Semaphore(_ENRICH_CONCURRENCY)
            enriched = await asyncio.gather(
                *(_enrich_ms_entry(model, sem) for model in models),
                return_exceptions=True,
            )
            models = [model for model in enriched if isinstance(model, dict)]
            models = [
                model
                for model in models
                if model.get("size", 0) == 0
                or model.get("size", 0) <= max_memory_bytes
            ]

        return {
            "models": models[:result_limit],
            "total": upstream_total,
        }

    @staticmethod
    async def get_filter_options() -> dict:
        """Return the current ModelScope top-level task taxonomy."""
        return {"task_groups": await _fetch_ms_task_groups()}

    @staticmethod
    async def get_recommended_models(
        max_memory_bytes: int,
        limit: int = 60,
        result_limit: int = 50,
        mlx_only: bool = True,
    ) -> dict:
        """Fetch trending and popular models from ModelScope.

        When mlx_only is True, uses SDK to list models from mlx-community
        organization. When False, uses REST API to search all organizations.

        Args:
            max_memory_bytes: Maximum model size in bytes (typically system memory).
            limit: Number of models to fetch per category.
            result_limit: Maximum number of models to return per category.
            mlx_only: If True, restrict to mlx-community organization.

        Returns:
            Dict with 'trending' and 'popular' lists.
        """

        async def _fetch() -> list[dict]:
            if mlx_only:
                api = _get_ms_api()
                if api is None:
                    logger.warning("ModelScope SDK not available")
                    return []
                try:
                    data = await asyncio.wait_for(
                        asyncio.to_thread(
                            api.list_models,
                            "mlx-community",
                            page_size=limit,
                        ),
                        timeout=_MS_API_TIMEOUT + 5,
                    )
                except Exception as e:
                    logger.warning(f"ModelScope recommended fetch failed: {e}")
                    return []
                models_data = data.get("Models", [])
                if not models_data:
                    models_data = data.get("models", [])
            else:
                models_data = await _fetch_ms_models_rest(page_size=limit)

            results = []
            for entry in models_data:
                m = _parse_ms_model_entry(entry)
                downloads = m.get("downloads", 0)
                size = m.get("size", 0)
                # Filter by minimum downloads
                if downloads < _MIN_DOWNLOADS:
                    continue
                # Filter by memory size (only when list_models already had
                # a size — enrichment may reveal more below).
                if size > 0 and size > max_memory_bytes:
                    continue
                results.append(m)
                if len(results) >= result_limit * 2:
                    break

            return results

        models = await _fetch()

        # Enrich with size + params from per-model config.json / detail
        # fetches. Bounded concurrency keeps the call to ~1–2s for a full
        # page; results are cached in-process so subsequent loads are free.
        if models:
            sem = asyncio.Semaphore(_ENRICH_CONCURRENCY)
            enriched = await asyncio.gather(
                *(_enrich_ms_entry(m, sem) for m in models),
                return_exceptions=True,
            )
            models = [m for m in enriched if isinstance(m, dict)]

            # Re-apply the memory filter now that enrichment may have
            # supplied a real size for entries that list_models reported
            # as 0. Entries that still have no size are kept (better to
            # show with a blank size than hide a candidate the user has
            # enough RAM for).
            models = [
                m for m in models
                if (m.get("size", 0) == 0) or (m["size"] <= max_memory_bytes)
            ]

        # Sort by downloads for popular, keep original order for trending
        trending = models[:result_limit]
        popular = sorted(models, key=lambda x: x.get("downloads", 0), reverse=True)[:result_limit]

        return {
            "trending": trending,
            "popular": popular,
        }

    @staticmethod
    async def search_models(
        query: str,
        sort: str = "trending",
        limit: int = 100,
        mlx_only: bool = True,
    ) -> dict:
        """Search models on ModelScope.

        When mlx_only is True, uses SDK to list models from mlx-community
        and filters by query string. When False, uses REST API to search
        across all organizations.

        Args:
            query: Search query string.
            sort: Sort order (trending/downloads/created/updated).
            limit: Maximum number of results to return.
            mlx_only: If True, restrict to mlx-community organization.

        Returns:
            Dict with 'models' list and 'total' count.
        """
        if mlx_only:
            api = _get_ms_api()
            if api is None:
                logger.warning("ModelScope SDK not available")
                return {"models": [], "total": 0}

            try:
                data = await asyncio.wait_for(
                    asyncio.to_thread(
                        api.list_models,
                        "mlx-community",
                        page_size=200,
                    ),
                    timeout=_MS_API_TIMEOUT + 5,
                )
            except Exception as e:
                logger.error(f"ModelScope search failed: {e}")
                return {"models": [], "total": 0}

            models_data = data.get("Models", [])
            if not models_data:
                models_data = data.get("models", [])
        else:
            models_data = await _fetch_ms_models_rest(
                query=query, page_size=200
            )

        # Filter by query string (case-insensitive)
        query_lower = query.lower()
        filtered = []
        for entry in models_data:
            name = entry.get("Name", "")
            if query_lower in name.lower():
                m = _parse_ms_model_entry(entry)
                filtered.append(m)

        # Sort results
        if sort == "downloads":
            filtered.sort(key=lambda x: x.get("downloads", 0), reverse=True)
        elif sort == "created":
            pass  # Keep original order (newest first by default)
        elif sort == "updated":
            pass  # Keep original order

        # Limit results
        results = filtered[:limit]

        return {
            "models": results,
            "total": len(filtered),
        }

    @staticmethod
    async def get_model_info(model_id: str) -> dict:
        """Fetch detailed model information from ModelScope.

        Args:
            model_id: ModelScope model ID (e.g., "qwen/Qwen2.5-7B-Instruct-MLX").

        Returns:
            Dict with model details including description, files, tags, etc.
        """
        api = _get_ms_api()
        if api is None:
            raise RuntimeError("ModelScope SDK not available")

        # Get model metadata
        model_data = await asyncio.wait_for(
            asyncio.to_thread(api.get_model, model_id),
            timeout=_MS_API_TIMEOUT,
        )

        # get_model may return a string or dict depending on SDK version
        if isinstance(model_data, str):
            import json

            try:
                model_data = json.loads(model_data)
            except (json.JSONDecodeError, TypeError):
                model_data = {}

        if not isinstance(model_data, dict):
            model_data = {}

        # Get file list
        files = []
        total_file_size = 0
        try:
            file_list = await asyncio.wait_for(
                asyncio.to_thread(api.get_model_files, model_id),
                timeout=_MS_API_TIMEOUT,
            )
            for f in file_list or []:
                fname = f.get("Name") or f.get("Path", "")
                fsize = f.get("Size") or 0
                if isinstance(fsize, str):
                    try:
                        fsize = int(fsize)
                    except ValueError:
                        fsize = 0
                total_file_size += fsize
                files.append(
                    {
                        "name": fname,
                        "size": fsize,
                        "size_formatted": (
                            _format_model_size(fsize) if fsize > 0 else ""
                        ),
                    }
                )
        except Exception as e:
            logger.warning(f"Could not fetch file list for {model_id}: {e}")

        # Fetch model card (README.md) content
        model_card = ""
        try:
            endpoint = _get_ms_endpoint()
            readme_url = f"{endpoint}/api/v1/models/{model_id}/repo?FilePath=README.md&Revision=master"
            resp = await asyncio.wait_for(
                asyncio.to_thread(
                    requests.get,
                    readme_url,
                    timeout=_MS_API_TIMEOUT,
                ),
                timeout=_MS_API_TIMEOUT + 5,
            )
            if resp.status_code == 200:
                card_text = resp.text
                # Strip YAML front matter (between --- markers)
                if card_text.startswith("---"):
                    end = card_text.find("---", 3)
                    if end != -1:
                        card_text = card_text[end + 3:].strip()
                model_card = card_text
        except Exception:
            pass  # README not available

        # Extract metadata
        name = model_data.get("Name") or model_id
        downloads = model_data.get("Downloads") or 0
        likes = model_data.get("Likes") or model_data.get("Stars") or 0
        tags = model_data.get("Tags") or []
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]

        return {
            "repo_id": model_id,
            "name": name,
            "model_card": model_card,
            "description": model_data.get("Description", ""),
            "files": files,
            "tags": tags,
            "pipeline_tag": model_data.get("Task", ""),
            "params": None,
            "params_formatted": None,
            "size": total_file_size,
            "size_formatted": (
                _format_model_size(total_file_size) if total_file_size > 0 else ""
            ),
            "downloads": downloads,
            "likes": likes,
            "created_at": model_data.get("CreatedTime", ""),
            "updated_at": model_data.get("LastUpdatedTime", ""),
        }

    def __init__(
        self,
        model_dir: str,
        on_complete: Optional[Callable] = None,
    ):
        self._model_dir = Path(model_dir)
        self._tasks: dict[str, DownloadTask] = {}
        self._active_tasks: dict[str, asyncio.Task] = {}
        self._progress_tasks: dict[str, asyncio.Task] = {}
        self._on_complete = on_complete
        self._cancelled: set[str] = set()
        self._download_sem = asyncio.Semaphore(1)

    @property
    def model_dir(self) -> Path:
        return self._model_dir

    def update_model_dir(self, new_dir: str) -> None:
        """Update the model directory path."""
        self._model_dir = Path(new_dir)

    async def start_download(
        self, model_id: str, ms_token: str = ""
    ) -> DownloadTask:
        """Start downloading a model from ModelScope.

        Args:
            model_id: ModelScope model ID (e.g., "qwen/Qwen2.5-7B-Instruct-MLX").
            ms_token: Optional ModelScope token for private models.

        Returns:
            The created DownloadTask.

        Raises:
            ValueError: If model_id format is invalid or download is already queued.
            RuntimeError: If ModelScope SDK is not installed.
        """
        if not MS_SDK_AVAILABLE:
            raise RuntimeError(
                "ModelScope SDK not installed. "
                "Install with: pip install \"omlx[modelscope]\""
            )

        model_id = model_id.strip()
        if "/" not in model_id or len(model_id.split("/")) != 2:
            raise ValueError(
                f"Invalid model ID: '{model_id}'. "
                "Expected format: 'owner/model' (e.g., 'qwen/Qwen2.5-7B-Instruct-MLX')"
            )

        # Check for duplicate active downloads
        for task in self._tasks.values():
            if task.repo_id == model_id and task.status in (
                DownloadStatus.PENDING,
                DownloadStatus.DOWNLOADING,
            ):
                raise ValueError(
                    f"Download for '{model_id}' is already in progress"
                )

        task_id = str(uuid.uuid4())
        task = DownloadTask(task_id=task_id, repo_id=model_id)
        self._tasks[task_id] = task

        # Start download in background
        self._active_tasks[task_id] = asyncio.create_task(
            self._run_download(task_id, ms_token)
        )

        logger.info(f"MS Download queued: {model_id} (task_id={task_id})")
        return task

    async def cancel_download(self, task_id: str) -> bool:
        """Cancel an active download.

        Note: Due to Python threading limitations, the actual download thread
        cannot be interrupted immediately. The download will be marked as
        cancelled and files will be cleaned up when the thread completes.

        Args:
            task_id: The task ID to cancel.

        Returns:
            True if the task was found and cancelled.
        """
        task = self._tasks.get(task_id)
        if task is None:
            return False

        if task.status not in (DownloadStatus.PENDING, DownloadStatus.DOWNLOADING):
            return False

        # Mark as cancelled - the running thread will check this flag
        self._cancelled.add(task_id)
        task.status = DownloadStatus.CANCELLED
        task.error = "Cancellation requested. Download will stop shortly."

        # Stop progress polling
        progress_task = self._progress_tasks.pop(task_id, None)
        if progress_task and not progress_task.done():
            progress_task.cancel()

        # Cancel the download task
        active_task = self._active_tasks.pop(task_id, None)
        if active_task and not active_task.done():
            active_task.cancel()

        logger.info(f"MS Download cancelled: {task.repo_id} (task_id={task_id})")
        return True

    def remove_task(self, task_id: str) -> bool:
        """Remove a completed, failed, or cancelled task from the list.

        Args:
            task_id: The task ID to remove.

        Returns:
            True if the task was found and removed.
        """
        task = self._tasks.get(task_id)
        if task is None:
            return False

        if task.status in (DownloadStatus.PENDING, DownloadStatus.DOWNLOADING):
            return False

        del self._tasks[task_id]
        self._cancelled.discard(task_id)
        return True

    async def retry_download(
        self, task_id: str, ms_token: str = ""
    ) -> DownloadTask:
        """Retry a failed or cancelled download, resuming from existing files.

        Args:
            task_id: The task ID of the failed/cancelled download.
            ms_token: Optional ModelScope token for private models.

        Returns:
            The new DownloadTask.

        Raises:
            ValueError: If task not found or not in retryable state.
        """
        old_task = self._tasks.get(task_id)
        if old_task is None:
            raise ValueError(f"Task not found: {task_id}")

        if old_task.status not in (DownloadStatus.FAILED, DownloadStatus.CANCELLED):
            raise ValueError(
                f"Task {task_id} is not retryable (status: {old_task.status.value})"
            )

        model_id = old_task.repo_id
        old_retry_count = old_task.retry_count

        # Remove old task entry
        del self._tasks[task_id]
        self._cancelled.discard(task_id)

        # Start fresh download (snapshot_download resumes from existing files)
        new_task = await self.start_download(model_id, ms_token)
        new_task.retry_count = old_retry_count + 1
        return new_task

    def get_tasks(self) -> list[dict]:
        """Return all tasks as serializable dicts, ordered by creation time."""
        return [
            task.to_dict()
            for task in sorted(self._tasks.values(), key=lambda t: t.created_at)
        ]

    async def shutdown(self) -> None:
        """Cancel all active downloads and clean up."""
        # Cancel all progress polling tasks
        for task_id, progress_task in list(self._progress_tasks.items()):
            if not progress_task.done():
                progress_task.cancel()
        self._progress_tasks.clear()

        # Cancel all active download tasks
        for task_id, active_task in list(self._active_tasks.items()):
            if not active_task.done():
                active_task.cancel()
                task = self._tasks.get(task_id)
                if task and task.status == DownloadStatus.DOWNLOADING:
                    task.status = DownloadStatus.CANCELLED
        self._active_tasks.clear()

        logger.info("MS Downloader shut down")

    async def _run_download(self, task_id: str, ms_token: str) -> None:
        """Execute a download task.

        Waits for the download semaphore (only one download runs at a time),
        then fetches file info for total size and runs snapshot_download in a
        thread while polling the target directory for progress updates.
        """
        task = self._tasks[task_id]

        try:
            async with self._download_sem:
                # Check if cancelled while waiting in queue
                if task_id in self._cancelled:
                    return

                task.status = DownloadStatus.DOWNLOADING
                task.started_at = time.time()

                # Preserve {owner}/{model} layout to match other tools
                # (LMStudio, huggingface-cli) and avoid duplicate downloads
                # when sharing a model directory.
                target_dir = self._model_dir / task.repo_id

                # Get total file size for progress estimation
                try:
                    api = _get_ms_api()
                    if api:
                        file_list = await asyncio.wait_for(
                            asyncio.to_thread(api.get_model_files, task.repo_id),
                            timeout=_MS_API_TIMEOUT,
                        )
                        if file_list:
                            task.total_size = _extract_model_size_from_files(
                                file_list
                            )
                except Exception as e:
                    logger.warning(
                        f"Could not fetch file info for {task.repo_id}: {e}. "
                        "Progress estimation will be unavailable."
                    )

                # Start progress polling
                self._progress_tasks[task_id] = asyncio.create_task(
                    self._poll_progress(task_id, target_dir)
                )

                # Build download kwargs
                dl_kwargs = {
                    "model_id": task.repo_id,
                    "local_dir": str(target_dir),
                }
                if ms_token:
                    dl_kwargs["token"] = ms_token

                # Run snapshot_download in a thread (blocking call)
                # Note: Thread cannot be interrupted, cancellation is checked after completion
                await asyncio.to_thread(
                    ms_snapshot_download,
                    **dl_kwargs,
                )

                # Check if cancelled while downloading - clean up downloaded files
                if task_id in self._cancelled:
                    logger.info(
                        f"MS Download was cancelled during execution: {task.repo_id}. "
                        "Cleaning up downloaded files..."
                    )
                    if target_dir.exists():
                        try:
                            shutil.rmtree(target_dir)
                            logger.info(f"Cleaned up cancelled download: {target_dir}")
                        except Exception as cleanup_err:
                            logger.warning(f"Failed to clean up {target_dir}: {cleanup_err}")
                    # Drop empty org folder left behind by the cancelled download.
                    parent = target_dir.parent
                    if (
                        parent != self._model_dir
                        and parent.exists()
                        and not any(parent.iterdir())
                    ):
                        try:
                            parent.rmdir()
                        except OSError as cleanup_err:
                            logger.debug(
                                f"Could not remove empty org folder {parent}: "
                                f"{cleanup_err}"
                            )
                    return

                # Success
                task.status = DownloadStatus.COMPLETED
                task.progress = 100.0
                task.downloaded_size = task.total_size or self._get_dir_size(
                    target_dir
                )
                task.completed_at = time.time()

                logger.info(
                    f"MS Download completed: {task.repo_id} -> {target_dir} "
                    f"({time.time() - task.started_at:.1f}s)"
                )

                # Trigger model pool refresh
                if self._on_complete:
                    try:
                        await self._on_complete()
                    except Exception as e:
                        logger.error(
                            f"Error in download completion callback: {e}"
                        )

        except asyncio.CancelledError:
            if task.status not in (
                DownloadStatus.CANCELLED,
                DownloadStatus.FAILED,
            ):
                task.status = DownloadStatus.CANCELLED
        except Exception as e:
            if task_id not in self._cancelled:
                task.status = DownloadStatus.FAILED
                # Provide user-friendly error messages
                err_msg = str(e)
                if "NotExistError" in type(e).__name__ or "404" in err_msg:
                    task.error = (
                        f"Model not found: {task.repo_id}. "
                        "Please check the model ID and try again."
                    )
                elif "401" in err_msg or "403" in err_msg:
                    task.error = (
                        f"Access denied for '{task.repo_id}'. "
                        "Please provide a valid ModelScope token."
                    )
                else:
                    task.error = err_msg
                logger.error(f"MS Download failed for {task.repo_id}: {e}")
        finally:
            # Stop progress polling
            progress_task = self._progress_tasks.pop(task_id, None)
            if progress_task and not progress_task.done():
                progress_task.cancel()

            # Remove from active tasks
            self._active_tasks.pop(task_id, None)

    async def _poll_progress(self, task_id: str, target_dir: Path) -> None:
        """Poll the target directory to estimate download progress.

        Uses both directory size and file modification times to detect
        activity.
        """
        task = self._tasks.get(task_id)
        if task is None:
            return

        last_size = 0
        last_activity_at = time.time()

        try:
            while task.status == DownloadStatus.DOWNLOADING:
                await asyncio.sleep(2)

                if task.status != DownloadStatus.DOWNLOADING:
                    break

                current_size = self._get_dir_size(target_dir)
                task.downloaded_size = current_size

                if task.total_size > 0:
                    # Cap at 99% until snapshot_download confirms completion
                    task.progress = min(
                        (current_size / task.total_size) * 100, 99.0
                    )

                # Activity detection: size change OR file mtime change
                if current_size != last_size:
                    last_size = current_size
                    last_activity_at = time.time()
                else:
                    latest_mtime = self._get_latest_mtime(target_dir)
                    if latest_mtime > last_activity_at:
                        last_activity_at = latest_mtime

                # Stall detection
                if (
                    current_size > 0
                    and (time.time() - last_activity_at) > _STALL_TIMEOUT
                ):
                    task.status = DownloadStatus.FAILED
                    task.error = (
                        f"Download stalled: no progress for {_STALL_TIMEOUT}s. "
                        "Try retrying the download."
                    )
                    logger.warning(
                        f"MS Download stalled for {task.repo_id} "
                        f"(task_id={task_id})"
                    )
                    # Cancel the snapshot_download thread
                    active_task = self._active_tasks.get(task_id)
                    if active_task and not active_task.done():
                        active_task.cancel()
                    break
        except asyncio.CancelledError:
            pass

    @staticmethod
    def _get_latest_mtime(path: Path) -> float:
        """Return the most recent modification time of any file in a directory."""
        if not path.exists():
            return 0.0
        latest = 0.0
        try:
            for f in path.rglob("*"):
                if f.is_file():
                    try:
                        mt = f.stat().st_mtime
                        if mt > latest:
                            latest = mt
                    except OSError:
                        pass
        except OSError:
            pass
        return latest

    @staticmethod
    def _get_dir_size(path: Path) -> int:
        """Calculate total size of all files in a directory."""
        if not path.exists():
            return 0
        total = 0
        try:
            for f in path.rglob("*"):
                if f.is_file():
                    try:
                        total += f.stat().st_size
                    except OSError:
                        pass
        except OSError:
            pass
        return total
