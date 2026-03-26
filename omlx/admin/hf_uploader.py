# SPDX-License-Identifier: Apache-2.0
"""HuggingFace model uploader for oMLX admin panel.

Uploads oQ-quantized models to HuggingFace Hub with queue-based sequential
processing, following the same pattern as hf_downloader.py.
"""

import asyncio
import enum
import json
import logging
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _format_size(size_bytes: int) -> str:
    """Format size in bytes to human-readable string."""
    if size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.1f} MB"
    else:
        return f"{size_bytes / 1024**3:.1f} GB"


def _is_oq_model(name: str) -> bool:
    """Check if a model name indicates an oQ-quantized model.

    oQ models have 'oQ' in the last 5 characters of their name,
    e.g. 'Qwen3.5-122B-oQ4', 'Llama-3B-oQ4e'.
    """
    return "oQ" in name[-5:]


def _generate_model_card(model_name: str, config: dict) -> str:
    """Generate a minimal HuggingFace model card for an oQ model."""
    model_type = config.get("model_type", "unknown")
    quant = config.get("quantization", {})
    bits = quant.get("bits", "?")
    group_size = quant.get("group_size", "?")

    return f"""---
library_name: mlx
tags:
- mlx
- oq
- quantized
---

# {model_name}

This model was quantized using [oQ](https://github.com/jundot/omlx) mixed-precision quantization.

## Quantization details

- **Model type**: {model_type}
- **Bits**: {bits}
- **Group size**: {group_size}
- **Format**: MLX safetensors
"""


class UploadStatus(str, enum.Enum):
    """Status of an upload task."""

    PENDING = "pending"
    UPLOADING = "uploading"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


_ACTIVE_STATUSES = {UploadStatus.PENDING, UploadStatus.UPLOADING}


@dataclass
class UploadTask:
    """Represents a single model upload task."""

    task_id: str
    model_name: str
    model_path: str
    repo_id: str
    status: UploadStatus = UploadStatus.PENDING
    progress: float = 0.0
    error: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0
    total_size: int = 0
    repo_url: str = ""

    def to_dict(self) -> dict:
        """Serialize task to a JSON-compatible dict."""
        return {
            "task_id": self.task_id,
            "model_name": self.model_name,
            "model_path": self.model_path,
            "repo_id": self.repo_id,
            "status": self.status.value,
            "progress": round(self.progress, 1),
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_size": self.total_size,
            "total_size_formatted": _format_size(self.total_size) if self.total_size else "",
            "repo_url": self.repo_url,
        }


class HFUploader:
    """Manages HuggingFace model uploads with queue-based sequential processing.

    Uses huggingface_hub's upload_folder() with a semaphore to ensure only
    one upload runs at a time. Multiple uploads can be queued.

    Args:
        model_dirs: List of model directory paths to scan for oQ models.
    """

    def __init__(self, model_dirs: list[str]):
        self._model_dirs = [Path(d) for d in model_dirs]
        self._tasks: dict[str, UploadTask] = {}
        self._active_tasks: dict[str, asyncio.Task] = {}
        self._cancelled: set[str] = set()
        self._upload_sem = asyncio.Semaphore(1)

    @staticmethod
    async def validate_token(token: str) -> dict:
        """Validate a HuggingFace token and return user info.

        Args:
            token: HuggingFace write-access token.

        Returns:
            Dict with 'username' and 'orgs' list.

        Raises:
            ValueError: If the token is invalid or lacks write access.
        """
        from huggingface_hub import HfApi

        try:
            api = HfApi()
            info = await asyncio.to_thread(api.whoami, token=token)
        except Exception as e:
            raise ValueError(f"Invalid token: {e}")

        username = info.get("name", "")
        orgs = [
            {"name": org.get("name", "")}
            for org in info.get("orgs", [])
            if org.get("name")
        ]

        # Check for write access
        auth = info.get("auth", {})
        access_token = auth.get("accessToken", {})
        role = access_token.get("role", "")
        if role == "read":
            raise ValueError(
                "Token has read-only access. A write token is required for uploads."
            )

        return {"username": username, "orgs": orgs}

    async def list_oq_models(self) -> list[dict]:
        """Scan model directories and return oQ-quantized models.

        Returns:
            List of dicts with model name, path, size info.
        """

        def _scan() -> list[dict]:
            models = []
            seen: set[str] = set()

            for model_dir in self._model_dirs:
                if not model_dir.exists():
                    continue
                for subdir in sorted(model_dir.iterdir()):
                    if not subdir.is_dir():
                        continue
                    candidates = []
                    if (subdir / "config.json").exists():
                        candidates.append(subdir)
                    else:
                        for child in sorted(subdir.iterdir()):
                            if child.is_dir() and (child / "config.json").exists():
                                candidates.append(child)

                    for path in candidates:
                        if path.name in seen:
                            continue
                        seen.add(path.name)
                        if not _is_oq_model(path.name):
                            continue
                        try:
                            size = sum(
                                f.stat().st_size
                                for f in path.glob("*.safetensors")
                            )
                            if size == 0:
                                continue
                            models.append({
                                "name": path.name,
                                "path": str(path),
                                "size": size,
                                "size_formatted": _format_size(size),
                            })
                        except Exception:
                            continue
            return models

        return await asyncio.to_thread(_scan)

    async def list_all_models(self) -> list[dict]:
        """Scan model directories and return all models (for README source selection).

        Returns:
            List of dicts with model name and path.
        """

        def _scan() -> list[dict]:
            models = []
            seen: set[str] = set()

            for model_dir in self._model_dirs:
                if not model_dir.exists():
                    continue
                for subdir in sorted(model_dir.iterdir()):
                    if not subdir.is_dir():
                        continue
                    candidates = []
                    if (subdir / "config.json").exists():
                        candidates.append(subdir)
                    else:
                        for child in sorted(subdir.iterdir()):
                            if child.is_dir() and (child / "config.json").exists():
                                candidates.append(child)

                    for path in candidates:
                        if path.name in seen:
                            continue
                        seen.add(path.name)
                        has_readme = (path / "README.md").exists()
                        models.append({
                            "name": path.name,
                            "path": str(path),
                            "has_readme": has_readme,
                        })
            return models

        return await asyncio.to_thread(_scan)

    async def start_upload(
        self,
        model_path: str,
        repo_id: str,
        token: str,
        readme_source_path: str = "",
        auto_readme: bool = True,
        private: bool = False,
    ) -> UploadTask:
        """Queue a model upload to HuggingFace Hub.

        Args:
            model_path: Local path to the oQ model directory.
            repo_id: Target HuggingFace repository ID (e.g., 'user/model-oQ4').
            token: HuggingFace write token.
            readme_source_path: Optional path to model whose README.md to copy.
            auto_readme: If True and no readme_source_path, generate a basic README.
            private: If True, create a private repository.

        Returns:
            The created UploadTask.

        Raises:
            ValueError: If model path is invalid or upload is already queued.
        """
        source = Path(model_path)
        if not source.exists() or not source.is_dir():
            raise ValueError(f"Model directory not found: {model_path}")

        if not (source / "config.json").exists():
            raise ValueError(f"Not a valid model directory (no config.json): {model_path}")

        repo_id = repo_id.strip()
        if "/" not in repo_id or len(repo_id.split("/")) != 2:
            raise ValueError(
                f"Invalid repository ID: '{repo_id}'. "
                "Expected format: 'owner/model' (e.g., 'user/Llama-3B-oQ4')"
            )

        # Check for duplicate active uploads
        for task in self._tasks.values():
            if task.repo_id == repo_id and task.status in _ACTIVE_STATUSES:
                raise ValueError(
                    f"Upload to '{repo_id}' is already in progress"
                )

        model_name = source.name
        total_size = sum(
            f.stat().st_size for f in source.rglob("*") if f.is_file()
        )

        task_id = str(uuid.uuid4())
        task = UploadTask(
            task_id=task_id,
            model_name=model_name,
            model_path=model_path,
            repo_id=repo_id,
            total_size=total_size,
        )
        self._tasks[task_id] = task

        self._active_tasks[task_id] = asyncio.create_task(
            self._run_upload(task_id, token, readme_source_path, auto_readme, private)
        )

        logger.info(f"Upload queued: {model_name} -> {repo_id} (task_id={task_id})")
        return task

    async def cancel_upload(self, task_id: str) -> bool:
        """Cancel an active or pending upload.

        Args:
            task_id: The task ID to cancel.

        Returns:
            True if the task was found and cancelled.
        """
        task = self._tasks.get(task_id)
        if task is None:
            return False

        if task.status not in _ACTIVE_STATUSES:
            return False

        self._cancelled.add(task_id)
        task.status = UploadStatus.CANCELLED

        active_task = self._active_tasks.pop(task_id, None)
        if active_task and not active_task.done():
            active_task.cancel()

        logger.info(f"Upload cancelled: {task.model_name} (task_id={task_id})")
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

        if task.status in _ACTIVE_STATUSES:
            return False

        del self._tasks[task_id]
        self._cancelled.discard(task_id)
        return True

    def get_tasks(self) -> list[dict]:
        """Return all tasks as serializable dicts, ordered by creation time."""
        return [
            task.to_dict()
            for task in sorted(self._tasks.values(), key=lambda t: t.created_at)
        ]

    async def shutdown(self) -> None:
        """Cancel all active uploads and clean up."""
        for task_id, active_task in list(self._active_tasks.items()):
            if not active_task.done():
                active_task.cancel()
                task = self._tasks.get(task_id)
                if task and task.status == UploadStatus.UPLOADING:
                    task.status = UploadStatus.CANCELLED
        self._active_tasks.clear()
        logger.info("HF Uploader shut down")

    async def _run_upload(
        self,
        task_id: str,
        token: str,
        readme_source_path: str,
        auto_readme: bool,
        private: bool,
    ) -> None:
        """Execute an upload task with semaphore-guarded sequential processing."""
        from huggingface_hub import HfApi

        task = self._tasks[task_id]
        tmp_readme: Optional[Path] = None

        try:
            async with self._upload_sem:
                if task_id in self._cancelled:
                    return

                task.status = UploadStatus.UPLOADING
                task.started_at = time.time()

                model_path = Path(task.model_path)
                api = HfApi()

                # Create repo (exist_ok handles already-existing repos)
                await asyncio.to_thread(
                    api.create_repo,
                    repo_id=task.repo_id,
                    token=token,
                    exist_ok=True,
                    private=private,
                )

                if task_id in self._cancelled:
                    return

                # Handle README
                readme_in_model = model_path / "README.md"
                if readme_source_path:
                    source_readme = Path(readme_source_path) / "README.md"
                    if source_readme.exists():
                        shutil.copy2(source_readme, readme_in_model)
                        tmp_readme = readme_in_model
                elif auto_readme and not readme_in_model.exists():
                    try:
                        with open(model_path / "config.json") as f:
                            config = json.load(f)
                    except Exception:
                        config = {}
                    readme_content = _generate_model_card(task.model_name, config)
                    readme_in_model.write_text(readme_content, encoding="utf-8")
                    tmp_readme = readme_in_model

                if task_id in self._cancelled:
                    return

                # Upload the entire model folder
                # upload_folder is blocking; run in thread
                task.progress = 10.0  # Signal that upload has started

                await asyncio.to_thread(
                    api.upload_folder,
                    folder_path=str(model_path),
                    repo_id=task.repo_id,
                    token=token,
                    commit_message=f"Upload {task.model_name} via oMLX",
                )

                if task_id in self._cancelled:
                    return

                # Success
                task.status = UploadStatus.COMPLETED
                task.progress = 100.0
                task.completed_at = time.time()
                task.repo_url = f"https://huggingface.co/{task.repo_id}"

                elapsed = task.completed_at - task.started_at
                logger.info(
                    f"Upload completed: {task.model_name} -> {task.repo_id} "
                    f"({elapsed:.0f}s, {_format_size(task.total_size)})"
                )

        except asyncio.CancelledError:
            if task.status not in (UploadStatus.CANCELLED, UploadStatus.FAILED):
                task.status = UploadStatus.CANCELLED
        except Exception as e:
            if task_id not in self._cancelled:
                task.status = UploadStatus.FAILED
                task.error = str(e)
                logger.error(f"Upload failed for {task.model_name}: {e}")
        finally:
            # Clean up copied/generated README if we created it
            if tmp_readme and tmp_readme.exists():
                try:
                    tmp_readme.unlink()
                except Exception:
                    pass
            self._active_tasks.pop(task_id, None)
