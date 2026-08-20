# SPDX-License-Identifier: Apache-2.0
"""HuggingFace model downloader for oMLX admin panel.

Downloads models from HuggingFace Hub using huggingface_hub's snapshot_download
with directory-size-based progress polling.
"""

import asyncio
import enum
import logging
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable, Optional
from urllib.parse import quote, unquote, urlparse

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils import (
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)
from huggingface_hub.utils import tqdm as _hf_tqdm

# Private-module import; the pyproject floor (huggingface-hub>=1.19.0)
# guarantees it exists. Re-verify the symbol when bumping the hub version.
from huggingface_hub.utils._xet import abort_xet_session

from . import ds4_downloader_catalog as _ds4_catalog

logger = logging.getLogger(__name__)

# Timeout for HuggingFace API calls (seconds).
# Prevents server from hanging when HF is unreachable.
_HF_API_TIMEOUT = 10

# Seconds with no download progress before considering the download stalled.
_STALL_TIMEOUT = 300

# Cache of (configured_endpoint -> resolved_endpoint) so we only probe each
# endpoint once per process lifetime. Mirrors like hf-mirror.com permanently
# 308-redirect to huggingface.co when accessed from IPs outside their region;
# huggingface_hub does NOT follow those cross-origin 308s during HEAD probes,
# so downloads fail. We resolve the redirect chain upfront and pin HfApi to
# the final origin.
_endpoint_resolution_cache: dict[str, str] = {}
_HF_URL_HOSTS = {"huggingface.co", "www.huggingface.co", "hf.co", "www.hf.co"}
_HF_FILE_URL_KINDS = {"blob", "resolve"}


@dataclass(frozen=True)
class _HFDownloadTarget:
    """Normalized HuggingFace repo or single-file download target."""

    repo_id: str
    revision: str | None = None
    filename: str | None = None

    @property
    def is_file(self) -> bool:
        return self.filename is not None

    @property
    def download_ref(self) -> str:
        if not self.filename:
            return self.repo_id
        revision = quote(self.revision or "main", safe="")
        filename = quote(self.filename, safe="/")
        return f"https://huggingface.co/{self.repo_id}/resolve/{revision}/{filename}"

    @property
    def display_name(self) -> str:
        if not self.filename:
            return self.repo_id
        return f"{self.repo_id}/{self.filename}"


def _validate_repo_id(repo_id: str) -> str:
    repo_id = repo_id.strip()
    if "/" not in repo_id or len(repo_id.split("/")) != 2:
        raise ValueError(
            f"Invalid repository ID: '{repo_id}'. "
            "Expected format: 'owner/model' "
            "(e.g., 'mlx-community/Llama-3-8B-4bit') or a HuggingFace file URL"
        )
    owner, name = repo_id.split("/", 1)
    if not owner or not name:
        raise ValueError(
            f"Invalid repository ID: '{repo_id}'. " "Expected format: 'owner/model'"
        )
    return repo_id


def _validate_hf_file_path(filename: str) -> str:
    filename = filename.strip("/")
    path = PurePosixPath(filename)
    if (
        not filename
        or path.is_absolute()
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise ValueError(
            f"Invalid HuggingFace file path: '{filename}'. "
            "Expected a file path inside the repository."
        )
    return filename


def _parse_hf_file_url(value: str) -> _HFDownloadTarget | None:
    parsed = urlparse(value.strip())
    if (
        parsed.scheme not in ("http", "https")
        or parsed.netloc.lower() not in _HF_URL_HOSTS
    ):
        return None

    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if len(parts) < 5 or parts[2] not in _HF_FILE_URL_KINDS:
        raise ValueError(
            "Invalid HuggingFace file URL. Expected "
            "https://huggingface.co/owner/model/resolve/<revision>/<file> "
            "or /blob/<revision>/<file>."
        )

    repo_id = _validate_repo_id(f"{parts[0]}/{parts[1]}")
    revision = parts[3].strip()
    filename = _validate_hf_file_path("/".join(parts[4:]))
    if not revision:
        raise ValueError("Invalid HuggingFace file URL: missing revision.")
    return _HFDownloadTarget(repo_id=repo_id, revision=revision, filename=filename)


def _resolve_download_target(value: str) -> _HFDownloadTarget:
    value = value.strip()
    parsed = urlparse(value)
    if parsed.scheme in ("http", "https"):
        target = _parse_hf_file_url(value)
        if target is not None:
            return target
        raise ValueError(
            "Invalid HuggingFace URL. Expected a huggingface.co file URL "
            "or repository ID in 'owner/model' format."
        )
    return _HFDownloadTarget(repo_id=_validate_repo_id(value))


def _resolve_endpoint(endpoint: str) -> str:
    """Follow permanent (301/308) cross-origin redirects on `endpoint`.

    Returns the final origin (scheme://host[:port]) the endpoint resolves to.
    Used to work around `huggingface_hub`'s inability to follow cross-origin
    308 redirects during file-download HEAD probes.

    Probes a known-stable HF API path (`/api/models/gpt2`) with HEAD; if the
    server returns a 301/308 with a Location pointing at a different host,
    the redirected origin is returned (and cached). Network errors fall back
    to the original endpoint.
    """
    endpoint = endpoint.rstrip("/")
    if endpoint in _endpoint_resolution_cache:
        return _endpoint_resolution_cache[endpoint]

    try:
        import httpx
    except ImportError:
        return endpoint

    probe = f"{endpoint}/api/models/gpt2"
    original_host = urlparse(endpoint).netloc
    resolved = endpoint
    try:
        with httpx.Client(follow_redirects=False, timeout=5.0) as client:
            r = client.head(probe)
            # Walk up to 3 permanent hops; stop on first non-permanent status.
            hops = 0
            current_url = probe
            while r.status_code in (301, 308) and "location" in r.headers:
                hops += 1
                if hops > 3:
                    break
                location = r.headers["location"]
                if location.startswith("/"):
                    # Relative redirect — same origin, no rewrite needed.
                    break
                target = urlparse(location)
                if not target.netloc:
                    break
                if target.netloc != original_host:
                    # Cross-origin permanent redirect: rewrite the endpoint.
                    port = f":{target.port}" if target.port else ""
                    resolved = f"{target.scheme}://{target.hostname}{port}"
                    original_host = target.netloc
                current_url = location
                r = client.head(current_url)
    except Exception as e:  # noqa: BLE001 — probe is best-effort
        logger.debug(f"HF endpoint probe failed for {endpoint}: {e}")
        return endpoint

    if resolved != endpoint:
        logger.info(
            f"HuggingFace endpoint {endpoint} permanently redirects to "
            f"{resolved}; using resolved origin for downloads."
        )
    _endpoint_resolution_cache[endpoint] = resolved
    return resolved


class _DownloadCancelled(Exception):
    """Raised inside the download thread to interrupt a cancelled download."""


def _make_cancellable_tqdm(should_cancel: Callable[[], bool]) -> type:
    """Build a tqdm subclass that aborts the download when cancelled.

    huggingface_hub's http_get calls ``progress.update(len(chunk))`` once per
    downloaded chunk (DOWNLOAD_CHUNK_SIZE, 10MB). A running thread can't be
    force-stopped and snapshot_download takes no cancel token, so we cooperate
    from the progress callback: raising here unwinds the download thread
    cleanly within one chunk, releasing its buffers and connection.

    Note: this only interrupts the Python http_get path, which xet-less repos
    and mirror endpoints still use. On the xet path the Rust side defers a
    callback exception until the whole transfer finishes (issue #1322), so
    cancellation there is driven by ``abort_xet_session()`` instead; this
    class is kept as the raise-on-next-chunk backstop for http_get.
    """

    class _CancellableTqdm(_hf_tqdm):
        def update(self, n=1):
            if should_cancel():
                raise _DownloadCancelled()
            return super().update(n)

    return _CancellableTqdm


def _get_hf_api() -> tuple[HfApi, str | None]:
    """Create HfApi instance with configured endpoint.

    Only the admin UI's `huggingface.endpoint` setting is honored here.
    When that's empty, return `HfApi()` with no explicit endpoint so
    `huggingface_hub` falls back to its own resolution (which already
    honors the `HF_ENDPOINT` env var). The configured endpoint, when
    present, is run through `_resolve_endpoint()` to follow permanent
    cross-origin redirects (e.g. hf-mirror.com → huggingface.co from
    non-CN IPs) so downstream HF library code sees a stable origin.

    Returns:
        Tuple of (HfApi instance, endpoint URL or None).
    """
    endpoint: str | None = None
    try:
        from ..settings import get_settings

        endpoint = get_settings().huggingface.endpoint or None
    except (RuntimeError, AttributeError):
        endpoint = None

    if endpoint:
        resolved = _resolve_endpoint(endpoint)
        return HfApi(endpoint=resolved), resolved
    return HfApi(), None


def _list_models_stale_token_fallback(api: HfApi, kwargs: dict) -> tuple[list, bool]:
    """Drain list_models, retrying anonymously when the stored token is rejected.

    huggingface_hub attaches the locally stored credential (HF_TOKEN env var or
    the hf auth login token file) to every request, so a stale token 401s even
    the public model listing (#2276, #2310). Listing needs no auth, so retry
    once with token=False and report the rejected token to the caller.
    """
    try:
        return list(api.list_models(**kwargs)), False
    except HfHubHTTPError as e:
        if e.response is None or e.response.status_code != 401:
            raise
        logger.warning(
            "HF model listing rejected the stored token (401): %s. "
            "Retrying anonymously.",
            e,
        )
        return list(api.list_models(token=False, **kwargs)), True


class DownloadStatus(str, enum.Enum):
    """Status of a download task."""

    PENDING = "pending"
    DOWNLOADING = "downloading"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class DownloadTask:
    """Represents a single model download task."""

    task_id: str
    repo_id: str
    revision: str = ""
    filename: str = ""
    download_ref: str = ""
    display_name: str = ""
    status: DownloadStatus = DownloadStatus.PENDING
    progress: float = 0.0
    total_size: int = 0
    downloaded_size: int = 0
    error: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0
    retry_count: int = 0

    def to_dict(self) -> dict:
        """Serialize task to a JSON-compatible dict."""
        return {
            "task_id": self.task_id,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "filename": self.filename,
            "download_ref": self.download_ref or self.repo_id,
            "display_name": self.display_name or self.repo_id,
            "status": self.status.value,
            "progress": round(self.progress, 1),
            "total_size": self.total_size,
            "downloaded_size": self.downloaded_size,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "retry_count": self.retry_count,
        }


# Minimum downloads to be included in recommendations.
_MIN_DOWNLOADS = 100

# Backward-compatible helper aliases; tests and ModelScope downloader import
# these from this module while the DS4/GGUF catalog logic lives separately.
_DS4_GGUF_BASE_MODEL_FILTERS = _ds4_catalog.DS4_GGUF_BASE_MODEL_FILTERS
_calc_safetensors_disk_size = _ds4_catalog.calc_safetensors_disk_size
_format_model_size = _ds4_catalog.format_model_size
_format_param_count = _ds4_catalog.format_param_count
_get_param_count = _ds4_catalog.get_param_count
_gguf_size_from_siblings = _ds4_catalog.gguf_size_from_siblings
_has_gguf_files = _ds4_catalog.has_gguf_files
_hf_catalog_item = _ds4_catalog.hf_catalog_item
_hf_tags = _ds4_catalog.hf_tags
_merge_catalog_results = _ds4_catalog.merge_catalog_results
_resolve_hf_catalog_filters = _ds4_catalog.resolve_hf_catalog_filters


def _hf_file_size(info, filename: str) -> int:
    """Return file size for a selected HF sibling when metadata is available."""
    for sibling in _ds4_catalog.hf_siblings(info):
        if _ds4_catalog.hf_sibling_name(sibling) == filename:
            return _ds4_catalog.hf_sibling_size(sibling)
    return 0


def _hf_file_catalog_item(target: _HFDownloadTarget, info) -> dict:
    """Build a DS4-GGUF catalog row for a direct HuggingFace file URL."""
    size = _hf_file_size(info, target.filename or "")
    downloads = getattr(info, "downloads", 0) or 0
    likes = getattr(info, "likes", 0) or 0
    trending_score = getattr(info, "trending_score", 0) or 0
    filename = target.filename or target.repo_id
    return {
        "repo_id": target.repo_id,
        "name": filename,
        "download_ref": target.download_ref,
        "hf_url": target.download_ref,
        "selected_file": target.filename or "",
        "revision": target.revision or "",
        "downloads": downloads,
        "likes": likes,
        "trending_score": trending_score,
        "created_at": _ds4_catalog.hf_timestamp(info, "created_at", "createdAt"),
        "updated_at": _ds4_catalog.hf_timestamp(
            info,
            "last_modified",
            "lastModified",
            "updated_at",
            "updatedAt",
        ),
        "size": size,
        "size_formatted": _format_model_size(size) if size > 0 else "",
        "params": None,
        "params_formatted": None,
        "backend": "ds4",
        "backends": ["ds4"],
        "backend_label": "DS4-GGUF",
        "format": "gguf",
        "compatibility_status": "unverified",
        "compatibility_note": (
            "DeepSeek V4 GGUF candidate; compatibility is unverified and DS4 "
            "will validate the downloaded file at launch."
        ),
    }


# HF API sort field mapping for search.
_SORT_MAP = {
    "trending": "trendingScore",
    "downloads": "downloads",
    "created": "createdAt",
    "updated": "lastModified",
    "most_params": "downloads",  # fetch by downloads, re-sort in Python
    "least_params": "downloads",  # fetch by downloads, re-sort in Python
    "largest": "downloads",  # fetch by downloads, re-sort by size in Python
    "smallest": "downloads",  # fetch by downloads, re-sort by size in Python
}


class HFDownloader:
    """Manages HuggingFace model downloads with progress tracking.

    Uses huggingface_hub.snapshot_download() for actual downloads and polls
    the target directory size to estimate progress.

    Args:
        model_dir: Directory where downloaded models are stored.
        on_complete: Async callback invoked when a download completes successfully.
    """

    @staticmethod
    async def get_recommended_models(
        max_memory_bytes: int,
        limit: int = 60,
        result_limit: int = 50,
        mlx_only: bool = True,
        show_mlx: bool | None = None,
        show_ds4_gguf: bool | None = None,
    ) -> dict:
        """Fetch trending/popular MLX and DS4-GGUF models that fit in memory.

        Args:
            max_memory_bytes: Maximum model size in bytes (typically system memory).
            limit: Number of models to fetch per category from HF API.
            result_limit: Maximum number of models to return per category.
            mlx_only: Legacy flag. When new OR flags are omitted, this keeps
                the old MLX/safetensors query behavior.
            show_mlx: Include MLX catalog results when set.
            show_ds4_gguf: Include DeepSeek V4 DS4-GGUF candidates when set.

        Returns:
            Dict with 'trending' and 'popular' lists, plus 'hf_token_invalid'
            set when the stored HF token was rejected and the listing was
            fetched anonymously instead.
        """
        api, _endpoint = _get_hf_api()
        legacy_filters = show_mlx is None and show_ds4_gguf is None
        include_mlx, include_ds4 = _resolve_hf_catalog_filters(
            mlx_only=mlx_only,
            show_mlx=show_mlx,
            show_ds4_gguf=show_ds4_gguf,
        )
        if not include_mlx and not include_ds4:
            return {"trending": [], "popular": []}

        async def _fetch_mlx(sort: str) -> tuple[list[dict], bool]:
            if not include_mlx:
                return [], False
            kwargs = {
                "sort": sort,
                "limit": limit,
                "expand": ["safetensors", "downloads", "likes", "trendingScore"],
            }
            if legacy_filters:
                if mlx_only:
                    kwargs["author"] = "mlx-community"
            else:
                kwargs["filter"] = "mlx"
            # list_models returns a lazy generator; drain it inside the worker
            # thread so the paginated HTTP calls never block the event loop.
            models, token_rejected = await asyncio.wait_for(
                asyncio.to_thread(_list_models_stale_token_fallback, api, kwargs),
                timeout=_HF_API_TIMEOUT,
            )
            results = []
            for m in models:
                if not m.safetensors or not m.safetensors.get("parameters"):
                    continue
                downloads = m.downloads or 0
                if downloads < _MIN_DOWNLOADS:
                    continue
                size = _calc_safetensors_disk_size(m.safetensors)
                if size <= 0 or size > max_memory_bytes:
                    continue
                results.append(
                    _hf_catalog_item(m, backend="mlx", name=m.id.split("/")[-1])
                )
            return results, token_rejected

        async def _fetch_ds4(sort: str) -> tuple[list[dict], bool]:
            if not include_ds4:
                return [], False
            results = []
            token_rejected = False
            for base_filter in _DS4_GGUF_BASE_MODEL_FILTERS:
                kwargs = {
                    "sort": sort,
                    "limit": limit,
                    "filter": base_filter,
                    "expand": [
                        "siblings",
                        "gguf",
                        "tags",
                        "downloads",
                        "likes",
                        "trendingScore",
                    ],
                }
                models, rejected = await asyncio.wait_for(
                    asyncio.to_thread(_list_models_stale_token_fallback, api, kwargs),
                    timeout=_HF_API_TIMEOUT,
                )
                token_rejected = token_rejected or rejected
                for m in models:
                    downloads = m.downloads or 0
                    if downloads < _MIN_DOWNLOADS or not _has_gguf_files(m):
                        continue
                    size = _gguf_size_from_siblings(m)
                    if size > 0 and size > max_memory_bytes:
                        continue
                    results.append(
                        _hf_catalog_item(
                            m,
                            backend="ds4",
                            name=m.id.split("/")[-1],
                            base_model_filter=base_filter,
                        )
                    )
            key = "downloads" if sort == "downloads" else "trending_score"
            return (
                sorted(
                    _merge_catalog_results(results),
                    key=lambda item: item.get(key, 0),
                    reverse=True,
                ),
                token_rejected,
            )

        async def _fetch(sort: str) -> tuple[list[dict], bool]:
            (mlx_results, mlx_rejected), (ds4_results, ds4_rejected) = await asyncio.gather(
                _fetch_mlx(sort),
                _fetch_ds4(sort),
            )
            key = "downloads" if sort == "downloads" else "trending_score"
            return (
                sorted(
                    _merge_catalog_results(mlx_results + ds4_results),
                    key=lambda item: item.get(key, 0),
                    reverse=True,
                ),
                mlx_rejected or ds4_rejected,
            )

        (trending, trending_rejected), (popular, popular_rejected) = (
            await asyncio.gather(
                _fetch("trendingScore"),
                _fetch("downloads"),
            )
        )

        return {
            "trending": trending[:result_limit],
            "popular": popular[:result_limit],
            "hf_token_invalid": trending_rejected or popular_rejected,
        }

    @staticmethod
    async def search_models(
        query: str,
        sort: str = "trending",
        limit: int = 100,
        mlx_only: bool = True,
        # Filtering options
        min_params: Optional[int] = None,
        max_params: Optional[int] = None,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        # Sorting options
        sort_by_size: bool = False,
        sort_ascending: bool = False,
        show_mlx: bool | None = None,
        show_ds4_gguf: bool | None = None,
    ) -> dict:
        """Search HuggingFace models by query string with OR backend filters.

        Args:
            query: Search query string.
            sort: Sort order (trending/downloads/created/updated/most_params/least_params/largest/smallest).
            limit: Maximum number of results to return.
            mlx_only: Legacy MLX-library filter used when new OR flags are omitted.
            min_params: Minimum parameter count filter.
            max_params: Maximum parameter count filter.
            min_size: Minimum model size in bytes filter.
            max_size: Maximum model size in bytes filter.
            sort_by_size: Sort results by size instead of default sort.
            sort_ascending: Sort in ascending order (for size/params sorting).
            show_mlx: Include MLX catalog results when set.
            show_ds4_gguf: Include DeepSeek V4 DS4-GGUF candidates when set.

        Returns:
            Dict with 'models' list and 'total' count, plus 'hf_token_invalid'
            set when the stored HF token was rejected and the listing was
            fetched anonymously instead.
        """
        api, _endpoint = _get_hf_api()
        legacy_filters = show_mlx is None and show_ds4_gguf is None
        include_mlx, include_ds4 = _resolve_hf_catalog_filters(
            mlx_only=mlx_only,
            show_mlx=show_mlx,
            show_ds4_gguf=show_ds4_gguf,
        )
        if not include_mlx and not include_ds4:
            return {"models": [], "total": 0}

        try:
            file_target = _parse_hf_file_url(query)
        except ValueError:
            file_target = None
        if file_target is not None:
            if (
                not include_ds4
                or not file_target.filename
                or not file_target.filename.lower().endswith(".gguf")
            ):
                return {"models": [], "total": 0}
            info = await asyncio.wait_for(
                asyncio.to_thread(
                    api.model_info,
                    file_target.repo_id,
                    revision=file_target.revision,
                    files_metadata=True,
                ),
                timeout=_HF_API_TIMEOUT,
            )
            item = _hf_file_catalog_item(file_target, info)
            size = item["size"] or 0
            if min_size is not None and size < min_size:
                return {"models": [], "total": 0}
            if max_size is not None and (size <= 0 or size > max_size):
                return {"models": [], "total": 0}
            if min_params is not None or max_params is not None:
                return {"models": [], "total": 0}
            return {"models": [item], "total": 1}

        # Determine base sort - for Python-side sorting, we fetch by downloads
        # which tends to return more results, then sort in Python
        if sort in ("most_params", "least_params", "largest", "smallest"):
            base_sort = "downloads"
        else:
            base_sort = _SORT_MAP.get(sort, "trendingScore")

        async def _fetch_mlx() -> tuple[list[dict], bool]:
            if not include_mlx:
                return [], False
            kwargs = {
                "search": query,
                "sort": base_sort,
                "limit": limit,
                "expand": [
                    "safetensors",
                    "downloads",
                    "likes",
                    "trendingScore",
                    "createdAt",
                    "lastModified",
                ],
            }
            if (legacy_filters and mlx_only) or not legacy_filters:
                kwargs["filter"] = "mlx"
            models, token_rejected = await asyncio.wait_for(
                asyncio.to_thread(_list_models_stale_token_fallback, api, kwargs),
                timeout=_HF_API_TIMEOUT,
            )
            return (
                [_hf_catalog_item(m, backend="mlx", name=m.id) for m in models],
                token_rejected,
            )

        async def _fetch_ds4() -> tuple[list[dict], bool]:
            if not include_ds4:
                return [], False
            results = []
            token_rejected = False
            for base_filter in _DS4_GGUF_BASE_MODEL_FILTERS:
                kwargs = {
                    "search": query,
                    "sort": base_sort,
                    "limit": limit,
                    "filter": base_filter,
                    "expand": [
                        "siblings",
                        "gguf",
                        "tags",
                        "downloads",
                        "likes",
                        "trendingScore",
                        "createdAt",
                        "lastModified",
                    ],
                }
                models, rejected = await asyncio.wait_for(
                    asyncio.to_thread(_list_models_stale_token_fallback, api, kwargs),
                    timeout=_HF_API_TIMEOUT,
                )
                token_rejected = token_rejected or rejected
                for m in models:
                    if _has_gguf_files(m):
                        results.append(
                            _hf_catalog_item(
                                m,
                                backend="ds4",
                                name=m.id,
                                base_model_filter=base_filter,
                            )
                        )
            return _merge_catalog_results(results), token_rejected

        (mlx_results, mlx_rejected), (ds4_results, ds4_rejected) = await asyncio.gather(
            _fetch_mlx(), _fetch_ds4()
        )
        token_rejected = mlx_rejected or ds4_rejected
        results = []
        for item in _merge_catalog_results(mlx_results + ds4_results):
            params = item.get("params")
            size = item.get("size") or 0
            if min_params is not None and (params is None or params < min_params):
                continue
            if max_params is not None and (params is None or params > max_params):
                continue
            if min_size is not None and size < min_size:
                continue
            if max_size is not None and size > max_size:
                continue
            results.append(item)

        def _ds4_or_tiebreaker(item: dict) -> int:
            backends = item.get("backends") or []
            return int(include_mlx and include_ds4 and "ds4" in backends)

        # Apply Python-side sorting
        if sort == "most_params":
            results.sort(key=lambda x: x["params"] or 0, reverse=True)
        elif sort == "least_params":
            results.sort(key=lambda x: x["params"] or 0)
        elif sort in ("largest", "smallest") or sort_by_size:
            # Sort by size, putting unknown-size entries at the end
            results.sort(
                key=lambda x: x["size"] if x["size"] > 0 else -1,
                reverse=(sort == "largest" or (sort_by_size and not sort_ascending)),
            )
        elif sort == "downloads":
            results.sort(key=lambda x: x.get("downloads", 0), reverse=True)
        elif sort == "trending":
            results.sort(key=lambda x: x.get("trending_score", 0), reverse=True)
        elif sort == "created":
            results.sort(
                key=lambda x: (x.get("created_at", 0), _ds4_or_tiebreaker(x)),
                reverse=True,
            )
        elif sort == "updated":
            results.sort(
                key=lambda x: (x.get("updated_at", 0), _ds4_or_tiebreaker(x)),
                reverse=True,
            )
        # Otherwise, keep original HF API ordering from the single selected catalog.

        return {
            "models": results[:limit],
            "total": len(results),
            "hf_token_invalid": token_rejected,
        }

    @staticmethod
    async def get_model_info(repo_id: str) -> dict:
        """Fetch detailed model information from HuggingFace.

        Args:
            repo_id: HuggingFace repository ID or file URL.

        Returns:
            Dict with model details including description, files, tags, etc.
        """
        target = _resolve_download_target(repo_id)
        api, endpoint = _get_hf_api()
        info = await asyncio.wait_for(
            asyncio.to_thread(
                api.model_info,
                target.repo_id,
                revision=target.revision,
                files_metadata=True,
            ),
            timeout=_HF_API_TIMEOUT,
        )

        # Extract file list with sizes
        files = []
        if info.siblings:
            for s in info.siblings:
                files.append(
                    {
                        "name": s.rfilename,
                        "size": s.size or 0,
                        "size_formatted": (
                            _format_model_size(s.size) if s.size else ""
                        ),
                    }
                )
        if target.filename:
            files = [f for f in files if f["name"] == target.filename]

        # Detect LoRA/adapter repos (adapter_config.json is peft standard)
        is_adapter = any(f["name"] == "adapter_config.json" for f in files)
        gguf_size = sum(f["size"] for f in files if f["name"].lower().endswith(".gguf"))
        has_gguf = gguf_size > 0 or any(tag.lower() == "gguf" for tag in _hf_tags(info))

        # Extract params and size from safetensors
        params = None
        params_formatted = None
        size = 0
        safetensors = getattr(info, "safetensors", None)
        if safetensors and not target.filename:
            st_dict = (
                dict(safetensors) if not isinstance(safetensors, dict) else safetensors
            )
            if st_dict.get("parameters"):
                params = _get_param_count(st_dict)
                params_formatted = _format_param_count(params) if params > 0 else None
                size = _calc_safetensors_disk_size(st_dict)
        if target.filename and gguf_size > 0:
            size = gguf_size

        # Fetch model card (README.md) content
        model_card = ""
        try:
            card_path = await asyncio.wait_for(
                asyncio.to_thread(
                    hf_hub_download,
                    repo_id=target.repo_id,
                    filename="README.md",
                    revision=target.revision,
                    endpoint=endpoint,
                ),
                timeout=_HF_API_TIMEOUT,
            )
            if card_path:
                card_text = Path(card_path).read_text(encoding="utf-8")
                # Strip YAML front matter (between --- markers)
                if card_text.startswith("---"):
                    end = card_text.find("---", 3)
                    if end != -1:
                        card_text = card_text[end + 3 :].strip()
                model_card = card_text
        except Exception:
            pass  # README not available

        backends = []
        has_safetensors_model = bool(
            safetensors and getattr(safetensors, "get", lambda _key: None)("parameters")
        )
        if has_safetensors_model or not has_gguf:
            backends.append("mlx")
        if has_gguf:
            backends.append("ds4")
        backend = "mlx+ds4" if len(backends) > 1 else (backends[0] if backends else "")

        return {
            "repo_id": info.id,
            "name": target.filename or info.id,
            "download_ref": target.download_ref,
            "hf_url": (
                target.download_ref
                if target.filename
                else f"https://huggingface.co/{info.id}"
            ),
            "selected_file": target.filename or "",
            "revision": target.revision or "",
            "model_card": model_card,
            "description": "",  # kept for backward compat
            "files": files,
            "tags": list(info.tags) if info.tags else [],
            "pipeline_tag": info.pipeline_tag or "",
            "params": params,
            "params_formatted": params_formatted,
            "size": size,
            "size_formatted": _format_model_size(size) if size > 0 else "",
            "downloads": info.downloads or 0,
            "likes": info.likes or 0,
            "created_at": info.created_at.isoformat() if info.created_at else "",
            "updated_at": (
                info.last_modified.isoformat() if info.last_modified else ""
            ),
            "is_adapter": is_adapter,
            "has_gguf": has_gguf,
            "gguf_size": gguf_size,
            "gguf_size_formatted": (
                _format_model_size(gguf_size) if gguf_size > 0 else ""
            ),
            "backend": backend,
            "backends": backends,
            "backend_label": (
                "MLX + DS4-GGUF"
                if backend == "mlx+ds4"
                else ("DS4-GGUF" if backend == "ds4" else "MLX")
            ),
            "compatibility_status": "unverified" if has_gguf else "verified",
            "compatibility_note": (
                "DeepSeek V4 GGUF candidate; compatibility is unverified and DS4 "
                f"will validate the downloaded {'file' if target.filename else 'repo'} at launch."
                if has_gguf
                else "MLX library model."
            ),
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

    async def start_download(self, repo_id: str, hf_token: str = "") -> DownloadTask:
        """Start downloading a model from HuggingFace.

        Args:
            repo_id: HuggingFace repository ID or file URL.
            hf_token: Optional HuggingFace token for gated models.

        Returns:
            The created DownloadTask.

        Raises:
            ValueError: If repo_id format is invalid or download is already queued.
        """
        target = _resolve_download_target(repo_id)

        # Check for duplicate active downloads
        for task in self._tasks.values():
            if task.status not in (
                DownloadStatus.PENDING,
                DownloadStatus.DOWNLOADING,
            ):
                continue
            if task.repo_id != target.repo_id:
                continue
            active_filename = task.filename or None
            if (
                active_filename is None
                or target.filename is None
                or active_filename == target.filename
            ):
                raise ValueError(
                    f"Download for '{target.display_name}' is already in progress"
                )

        task_id = str(uuid.uuid4())
        task = DownloadTask(
            task_id=task_id,
            repo_id=target.repo_id,
            revision=target.revision or "",
            filename=target.filename or "",
            download_ref=target.download_ref,
            display_name=target.display_name,
        )
        self._tasks[task_id] = task

        # Start download in background
        self._active_tasks[task_id] = asyncio.create_task(
            self._run_download(task_id, hf_token)
        )

        logger.info(f"Download queued: {target.display_name} (task_id={task_id})")
        return task

    async def cancel_download(self, task_id: str) -> bool:
        """Cancel an active download.

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

        was_downloading = task.status == DownloadStatus.DOWNLOADING

        # Mark as cancelled
        self._cancelled.add(task_id)
        task.status = DownloadStatus.CANCELLED

        # A task in DOWNLOADING owns the download semaphore, so the in-flight
        # xet transfer is necessarily this one; aborting the (global) session
        # makes its snapshot_download thread unwind immediately. Pending tasks
        # must not abort, that would kill another task's transfer. The next
        # download lazily creates a fresh session.
        if was_downloading:
            abort_xet_session()

        # Stop progress polling
        progress_task = self._progress_tasks.pop(task_id, None)
        if progress_task and not progress_task.done():
            progress_task.cancel()

        # Cancel the download task
        active_task = self._active_tasks.pop(task_id, None)
        if active_task and not active_task.done():
            active_task.cancel()

        logger.info(f"Download cancelled: {task.repo_id} (task_id={task_id})")
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

    async def retry_download(self, task_id: str, hf_token: str = "") -> DownloadTask:
        """Retry a failed or cancelled download, resuming from existing files.

        Finalized shards are preserved on disk so snapshot_download will
        automatically skip already-completed files.

        Args:
            task_id: The task ID of the failed/cancelled download.
            hf_token: Optional HuggingFace token for gated models.

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

        download_ref = old_task.download_ref or old_task.repo_id
        old_retry_count = old_task.retry_count

        # Remove old task entry
        del self._tasks[task_id]
        self._cancelled.discard(task_id)

        # Start fresh download (snapshot_download resumes from existing files)
        new_task = await self.start_download(download_ref, hf_token)
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

        # Cancel all active download tasks. Mark cancelled first so an
        # in-flight snapshot_download thread aborts via its progress callback;
        # active_task.cancel() only unblocks tasks still waiting on the semaphore.
        for task_id, active_task in list(self._active_tasks.items()):
            self._cancelled.add(task_id)
            if not active_task.done():
                active_task.cancel()
                task = self._tasks.get(task_id)
                if task and task.status == DownloadStatus.DOWNLOADING:
                    task.status = DownloadStatus.CANCELLED
        self._active_tasks.clear()

        # Reap any in-flight xet transfer thread (no-op without a session).
        abort_xet_session()

        logger.info("HF Downloader shut down")

    async def _run_download(self, task_id: str, hf_token: str) -> None:
        """Execute a download task.

        Waits for the download semaphore (only one download runs at a time),
        then fetches repo info for total size and runs snapshot_download in a
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

                api, endpoint = _get_hf_api()

                # Skip pytorch format when safetensors exist to
                # avoid downloading redundant weight files.
                ignore_patterns = None
                st_estimate = 0
                try:
                    model_info = await asyncio.wait_for(
                        asyncio.to_thread(
                            api.model_info,
                            task.repo_id,
                            token=hf_token or None,
                            revision=task.revision or None,
                            expand=["safetensors", "siblings", "tags"],
                        ),
                        timeout=_HF_API_TIMEOUT,
                    )
                    if (
                        not task.filename
                        and model_info.safetensors
                        and model_info.safetensors.get("parameters")
                        and not _has_gguf_files(model_info)
                    ):
                        ignore_patterns = [
                            "*.bin",
                            "original/**",
                            "consolidated.*.pth",
                        ]
                        # Computed inside this try so malformed metadata
                        # (non-int counts) degrades to no estimate instead
                        # of failing the download from the dry-run handler.
                        st_estimate = _calc_safetensors_disk_size(
                            model_info.safetensors
                        )
                except Exception as e:
                    logger.warning(f"Could not fetch repo info for {task.repo_id}: {e}")

                dl_kwargs: dict = {
                    "repo_id": task.repo_id,
                    "local_dir": str(target_dir),
                    "token": hf_token or None,
                    "endpoint": endpoint,
                    "etag_timeout": 30,
                }
                if task.revision:
                    dl_kwargs["revision"] = task.revision
                if task.filename:
                    dl_kwargs["allow_patterns"] = [task.filename]
                if ignore_patterns:
                    dl_kwargs["ignore_patterns"] = ignore_patterns

                # Get accurate total size via dry run so the progress
                # denominator matches what will actually be downloaded.
                size_estimated = False
                try:
                    dry_result = await asyncio.wait_for(
                        asyncio.to_thread(
                            snapshot_download,
                            **dl_kwargs,
                            dry_run=True,
                        ),
                        timeout=30,
                    )
                    task.total_size = sum(f.file_size for f in dry_result)
                except Exception as e:
                    if st_estimate:
                        task.total_size = st_estimate
                        size_estimated = True
                        detail = "Estimated total size from safetensors metadata."
                    else:
                        detail = "Progress estimation will be unavailable."
                    logger.warning(
                        f"Dry run failed for {task.repo_id}: {e}. {detail}"
                    )

                # Start progress polling
                self._progress_tasks[task_id] = asyncio.create_task(
                    self._poll_progress(task_id, target_dir)
                )

                # Run snapshot_download in a thread (blocking call). Cancel
                # reaches the thread two ways: the cancellable tqdm raises on
                # the next chunk of the http_get path, and abort_xet_session()
                # (called by cancel/stall/shutdown) unwinds the xet path with
                # a RuntimeError (the thread itself can't be force-killed).
                await asyncio.to_thread(
                    snapshot_download,
                    **dl_kwargs,
                    tqdm_class=_make_cancellable_tqdm(
                        lambda: task_id in self._cancelled
                    ),
                )

                # Check if cancelled while downloading
                if task_id in self._cancelled:
                    return

                # Success
                task.status = DownloadStatus.COMPLETED
                task.progress = 100.0
                if size_estimated or not task.total_size:
                    # The estimate was only a progress denominator; report
                    # the measured on-disk size once the download is done.
                    task.downloaded_size = self._get_dir_size(target_dir)
                else:
                    task.downloaded_size = task.total_size
                task.completed_at = time.time()

                logger.info(
                    f"Download completed: {task.repo_id} -> {target_dir} "
                    f"({time.time() - task.started_at:.1f}s)"
                )

                # Trigger model pool refresh
                if self._on_complete:
                    try:
                        await self._on_complete()
                    except Exception as e:
                        logger.error(f"Error in download completion callback: {e}")

        except (_DownloadCancelled, asyncio.CancelledError):
            if task.status not in (
                DownloadStatus.CANCELLED,
                DownloadStatus.FAILED,
            ):
                task.status = DownloadStatus.CANCELLED
            try:
                self._cleanup_partial(task)
            except Exception as e:
                logger.error(
                    f"Failed to clean up cancelled download {task.repo_id}: {e}"
                )
        except RepositoryNotFoundError:
            task.status = DownloadStatus.FAILED
            task.error = (
                f"Repository not found: {task.repo_id}. "
                "This may be a gated model that requires HuggingFace authentication."
            )
            logger.error(f"Repository not found: {task.repo_id}")
        except GatedRepoError:
            task.status = DownloadStatus.FAILED
            task.error = (
                f"Repository '{task.repo_id}' is gated. "
                "Please provide a valid HF token with access."
            )
            logger.error(f"Gated repo access denied: {task.repo_id}")
        except Exception as e:
            # Skip when already cancelled (the xet abort surfaces here as a
            # RuntimeError) or already FAILED by the stall detector, whose
            # error message would otherwise be clobbered by the abort error.
            if (
                task_id not in self._cancelled
                and task.status != DownloadStatus.FAILED
            ):
                task.status = DownloadStatus.FAILED
                task.error = str(e)
                logger.error(f"Download failed for {task.repo_id}: {e}")
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
        activity. huggingface_hub pre-allocates large files and fills them
        in, so size alone may not change for extended periods. File mtimes
        are updated on each write syscall and serve as a more reliable
        liveness signal.
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
                    task.progress = min((current_size / task.total_size) * 100, 99.0)

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
                        f"Download stalled for {task.repo_id} " f"(task_id={task_id})"
                    )
                    # Cancel the snapshot_download thread. The task cancel
                    # only unblocks the awaiting coroutine; aborting the xet
                    # session is what actually reaps a wedged transfer thread.
                    active_task = self._active_tasks.get(task_id)
                    if active_task and not active_task.done():
                        active_task.cancel()
                    abort_xet_session()
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

    def _cleanup_partial(self, task: DownloadTask) -> None:
        """Remove in-progress shards while keeping finalized files for resume.

        Hub stages partial downloads inside a hidden ``._____temp`` directory
        and only renames a shard into the target on completion. Wiping the
        whole target dir would also nuke shards the user has already paid
        for; finalized files are visible in the file browser, so users can
        keep them for auto-resume on retry or remove them themselves.
        """
        target_dir = self._model_dir / task.repo_id
        temp_dir = target_dir / "._____temp"
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up in-progress shards: {temp_dir}")
            except Exception as e:
                logger.error(f"Failed to clean up {temp_dir}: {e}")
