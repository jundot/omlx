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
from pathlib import Path
from typing import Callable, Optional

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils import (
    EntryNotFoundError,
    GatedRepoError,
    RepositoryNotFoundError,
)

logger = logging.getLogger(__name__)

# Timeout for HuggingFace API calls (seconds).
# Prevents server from hanging when HF is unreachable.
_HF_API_TIMEOUT = 10

# Seconds with no download progress before considering the download stalled.
_STALL_TIMEOUT = 120


def _get_hf_api() -> tuple[HfApi, str | None]:
    """Create HfApi instance with configured endpoint.

    Returns:
        Tuple of (HfApi instance, endpoint URL or None).
    """
    try:
        from ..settings import get_settings

        endpoint = get_settings().huggingface.endpoint
        if endpoint:
            return HfApi(endpoint=endpoint), endpoint
    except (RuntimeError, AttributeError):
        pass
    return HfApi(), None


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


_DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1,
    "U64": 8, "U32": 4, "U16": 2, "U8": 1,
    "BOOL": 1,
}

# Minimum downloads to be included in recommendations.
_MIN_DOWNLOADS = 100


def _calc_safetensors_disk_size(safetensors: dict) -> int:
    """Calculate actual disk size in bytes from safetensors parameters.

    safetensors.total is the parameter count, not bytes.
    We need to multiply each dtype's parameter count by its byte width.
    """
    params = safetensors.get("parameters", {})
    if not params:
        return 0
    return sum(count * _DTYPE_BYTES.get(dtype, 1) for dtype, count in params.items())


def _format_model_size(size_bytes: int) -> str:
    """Format model size in bytes to a human-readable string."""
    if size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.1f} MB"
    else:
        return f"{size_bytes / 1024**3:.1f} GB"


def _format_param_count(total_params: int) -> str:
    """Format parameter count to a human-readable string (e.g., 7.0B, 13.0B)."""
    if total_params >= 1e12:
        return f"{total_params / 1e12:.1f}T"
    if total_params >= 1e9:
        return f"{total_params / 1e9:.1f}B"
    if total_params >= 1e6:
        return f"{total_params / 1e6:.1f}M"
    return str(total_params)


def _get_param_count(safetensors: dict) -> int:
    """Get total parameter count from safetensors metadata."""
    params = safetensors.get("parameters", {})
    if not params:
        return 0
    return sum(params.values())


# HF API sort field mapping for search.
_SORT_MAP = {
    "trending": "trendingScore",
    "downloads": "downloads",
    "created": "createdAt",
    "updated": "lastModified",
    "most_params": "downloads",  # fetch by downloads, re-sort in Python
    "least_params": "downloads",  # fetch by downloads, re-sort in Python
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
    ) -> dict:
        """Fetch trending and popular mlx-community models that fit in memory.

        Queries HuggingFace Hub for text-generation models from mlx-community,
        filtered by system memory capacity.

        Args:
            max_memory_bytes: Maximum model size in bytes (typically system memory).
            limit: Number of models to fetch per category from HF API.
            result_limit: Maximum number of models to return per category.

        Returns:
            Dict with 'trending' and 'popular' lists.
        """
        api, _endpoint = _get_hf_api()

        async def _fetch(sort: str) -> list[dict]:
            models = await asyncio.wait_for(
                asyncio.to_thread(
                    api.list_models,
                    author="mlx-community",
                    sort=sort,
                    limit=limit,
                    expand=["safetensors", "downloads", "likes", "trendingScore"],
                ),
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
                params = _get_param_count(m.safetensors)
                results.append(
                    {
                        "repo_id": m.id,
                        "name": m.id.split("/")[-1],
                        "downloads": downloads,
                        "likes": m.likes or 0,
                        "trending_score": m.trending_score or 0,
                        "size": size,
                        "size_formatted": _format_model_size(size),
                        "params": params if params > 0 else None,
                        "params_formatted": (
                            _format_param_count(params) if params > 0 else None
                        ),
                    }
                )
            return results

        trending, popular = await asyncio.gather(
            _fetch("trendingScore"),
            _fetch("downloads"),
        )

        return {
            "trending": trending[:result_limit],
            "popular": popular[:result_limit],
        }

    @staticmethod
    async def search_models(
        query: str,
        sort: str = "trending",
        limit: int = 100,
    ) -> dict:
        """Search HuggingFace models by query string.

        Results are restricted to the MLX library so that only MLX-compatible
        models are returned (same as https://huggingface.co/models?library=mlx).

        Args:
            query: Search query string.
            sort: Sort order (trending/downloads/created/updated/most_params/least_params).
            limit: Maximum number of results to return.

        Returns:
            Dict with 'models' list and 'total' count.
        """
        api, _endpoint = _get_hf_api()
        sort_key = _SORT_MAP.get(sort, "trendingScore")

        models = await asyncio.wait_for(
            asyncio.to_thread(
                api.list_models,
                search=query,
                sort=sort_key,
                limit=limit,
                filter="mlx",
                expand=["safetensors", "downloads", "likes", "trendingScore"],
            ),
            timeout=_HF_API_TIMEOUT,
        )

        results = []
        for m in models:
            params = None
            params_formatted = None
            size = 0
            if m.safetensors and m.safetensors.get("parameters"):
                params = _get_param_count(m.safetensors)
                params_formatted = _format_param_count(params) if params > 0 else None
                size = _calc_safetensors_disk_size(m.safetensors)
                if params and params <= 0:
                    params = None

            results.append(
                {
                    "repo_id": m.id,
                    "name": m.id,
                    "downloads": m.downloads or 0,
                    "likes": m.likes or 0,
                    "trending_score": m.trending_score or 0,
                    "size": size,
                    "size_formatted": _format_model_size(size) if size > 0 else "",
                    "params": params,
                    "params_formatted": params_formatted,
                }
            )

        # Re-sort in Python for parameter-based sorting
        if sort == "most_params":
            results.sort(key=lambda x: x["params"] or 0, reverse=True)
        elif sort == "least_params":
            results.sort(key=lambda x: x["params"] or 0)

        return {
            "models": results[:limit],
            "total": len(results),
        }

    @staticmethod
    async def get_model_info(repo_id: str) -> dict:
        """Fetch detailed model information from HuggingFace.

        Args:
            repo_id: HuggingFace repository ID (e.g., "mlx-community/Llama-3-8B-4bit").

        Returns:
            Dict with model details including description, files, tags, etc.
        """
        api, endpoint = _get_hf_api()
        info = await asyncio.wait_for(
            asyncio.to_thread(
                api.model_info,
                repo_id,
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

        # Extract params and size from safetensors
        params = None
        params_formatted = None
        size = 0
        safetensors = getattr(info, "safetensors", None)
        if safetensors:
            st_dict = dict(safetensors) if not isinstance(safetensors, dict) else safetensors
            if st_dict.get("parameters"):
                params = _get_param_count(st_dict)
                params_formatted = _format_param_count(params) if params > 0 else None
                size = _calc_safetensors_disk_size(st_dict)

        # Fetch model card (README.md) content
        model_card = ""
        try:
            card_path = await asyncio.wait_for(
                asyncio.to_thread(
                    hf_hub_download,
                    repo_id=repo_id,
                    filename="README.md",
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
                        card_text = card_text[end + 3:].strip()
                model_card = card_text
        except Exception:
            pass  # README not available

        return {
            "repo_id": info.id,
            "name": info.id,
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
        self, repo_id: str, hf_token: str = ""
    ) -> DownloadTask:
        """Start downloading a model from HuggingFace.

        Args:
            repo_id: HuggingFace repository ID (e.g., "mlx-community/Llama-3-8B-4bit").
            hf_token: Optional HuggingFace token for gated models.

        Returns:
            The created DownloadTask.

        Raises:
            ValueError: If repo_id format is invalid or download is already queued.
        """
        repo_id = repo_id.strip()
        if "/" not in repo_id or len(repo_id.split("/")) != 2:
            raise ValueError(
                f"Invalid repository ID: '{repo_id}'. "
                "Expected format: 'owner/model' (e.g., 'mlx-community/Llama-3-8B-4bit')"
            )

        # Check for duplicate active downloads
        for task in self._tasks.values():
            if task.repo_id == repo_id and task.status in (
                DownloadStatus.PENDING,
                DownloadStatus.DOWNLOADING,
            ):
                raise ValueError(
                    f"Download for '{repo_id}' is already in progress"
                )

        task_id = str(uuid.uuid4())
        task = DownloadTask(task_id=task_id, repo_id=repo_id)
        self._tasks[task_id] = task

        # Start download in background
        self._active_tasks[task_id] = asyncio.create_task(
            self._run_download(task_id, hf_token)
        )

        logger.info(f"Download queued: {repo_id} (task_id={task_id})")
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

        # Mark as cancelled
        self._cancelled.add(task_id)
        task.status = DownloadStatus.CANCELLED

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

    async def retry_download(
        self, task_id: str, hf_token: str = ""
    ) -> DownloadTask:
        """Retry a failed or cancelled download, resuming from existing files.

        Since partial files are preserved on disk, snapshot_download will
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

        repo_id = old_task.repo_id
        old_retry_count = old_task.retry_count

        # Remove old task entry
        del self._tasks[task_id]
        self._cancelled.discard(task_id)

        # Start fresh download (snapshot_download resumes from existing files)
        new_task = await self.start_download(repo_id, hf_token)
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

                # Derive model name from repo_id (last part)
                model_name = task.repo_id.split("/")[-1]
                target_dir = self._model_dir / model_name

                # Get total repo size for progress estimation
                try:
                    api, endpoint = _get_hf_api()
                    model_info = await asyncio.wait_for(
                        asyncio.to_thread(
                            api.model_info,
                            task.repo_id,
                            token=hf_token or None,
                            expand=["safetensors"],
                        ),
                        timeout=_HF_API_TIMEOUT,
                    )
                    if model_info.safetensors and model_info.safetensors.get(
                        "parameters"
                    ):
                        task.total_size = _calc_safetensors_disk_size(
                            dict(model_info.safetensors)
                        )
                    elif model_info.siblings:
                        task.total_size = sum(
                            s.size for s in model_info.siblings if s.size
                        )
                except Exception as e:
                    logger.warning(
                        f"Could not fetch repo info for {task.repo_id}: {e}. "
                        "Progress estimation will be unavailable."
                    )

                # Start progress polling
                self._progress_tasks[task_id] = asyncio.create_task(
                    self._poll_progress(task_id, target_dir)
                )

                # Run snapshot_download in a thread (blocking call)
                await asyncio.to_thread(
                    snapshot_download,
                    repo_id=task.repo_id,
                    local_dir=str(target_dir),
                    token=hf_token or None,
                    endpoint=endpoint,
                    etag_timeout=30,
                )

                # Check if cancelled while downloading
                if task_id in self._cancelled:
                    return

                # Success
                task.status = DownloadStatus.COMPLETED
                task.progress = 100.0
                task.downloaded_size = task.total_size or self._get_dir_size(
                    target_dir
                )
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
                        logger.error(
                            f"Error in download completion callback: {e}"
                        )

        except asyncio.CancelledError:
            if task.status not in (
                DownloadStatus.CANCELLED,
                DownloadStatus.FAILED,
            ):
                task.status = DownloadStatus.CANCELLED
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
            if task_id not in self._cancelled:
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
                        f"Download stalled for {task.repo_id} "
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

    def _cleanup_partial(self, task: DownloadTask) -> None:
        """Remove partially downloaded model directory."""
        model_name = task.repo_id.split("/")[-1]
        target_dir = self._model_dir / model_name
        if target_dir.exists():
            try:
                shutil.rmtree(target_dir)
                logger.info(f"Cleaned up partial download: {target_dir}")
            except Exception as e:
                logger.error(f"Failed to clean up {target_dir}: {e}")
