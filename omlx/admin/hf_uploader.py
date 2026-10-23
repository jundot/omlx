# SPDX-License-Identifier: Apache-2.0
"""HuggingFace model uploader for oMLX admin panel.

Uploads oQ-quantized models to HuggingFace Hub with queue-based sequential
processing, following the same pattern as hf_downloader.py.
"""

import asyncio
import enum
import json
import logging
import math
import shutil
import struct
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

from ..model_discovery import CLEF_HEAD_WEIGHTS, D3_READOUT_WEIGHTS
from ..oq import (
    _is_audio_tensor,
    _is_mtp_tensor,
    _is_qwen4_exp_ngram_embedding_tensor,
    _is_vision_tensor,
)
from ..patches.minimax_m3_dspark import bundled_dspark_path

logger = logging.getLogger(__name__)


def _format_size(size_bytes: int) -> str:
    """Format size in bytes to human-readable string."""
    if size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.1f} MB"
    else:
        return f"{size_bytes / 1024**3:.1f} GB"


def _has_meaningful_readme(path: Path) -> bool:
    """Check if a README.md exists and has content beyond YAML frontmatter.

    Returns False if the file doesn't exist, is empty, or contains only
    YAML frontmatter (e.g. mlx-lm's default stub).
    """
    readme = path / "README.md"
    if not readme.exists():
        return False
    try:
        text = readme.read_text(encoding="utf-8").strip()
    except Exception:
        return False
    if not text:
        return False
    # Strip YAML frontmatter and check if anything remains
    if text.startswith("---"):
        parts = text.split("---", 2)
        # parts[0] is empty (before first ---), parts[1] is frontmatter
        if len(parts) >= 3:
            body = parts[2].strip()
            return len(body) > 0
        # Only opening --- or unclosed frontmatter
        return False
    return True


def _is_oq_model(name: str) -> bool:
    """Check if a model name indicates an oQ-quantized model.

    Any folder name containing 'oQ' (case-sensitive) is treated as an oQ model,
    e.g. 'Qwen3.5-122B-oQ4', 'Llama-3B-oQ4e', 'Qwen3.6-27B-oQ3.5e'.
    """
    return "oQ" in name


_DTYPE_LABELS = {
    "BF16": "BF16",
    "F16": "FP16",
    "F32": "FP32",
    "F8_E4M3": "FP8 (E4M3)",
    "F8_E5M2": "FP8 (E5M2)",
}
# MX formats fix their group size, so a config may record only bits and mode.
_MX_GROUP_SIZES = {"mxfp4": 32, "mxfp8": 32, "nvfp4": 16}
# Row-lookup tables that oMLX can serve from SSD. The card also reports the
# average without them, because they do not count toward resident memory.
_LOOKUP_TABLES = {
    "N-gram embedding": "the N-gram embedding table",
    "Engram tables": "the Engram tables",
}
_COMPONENT_ORDER = (
    "Language model",
    "N-gram embedding",
    "Engram tables",
    "Vision",
    "Audio",
    "MTP head",
    "Joint schema head",
    "Decision readout",
)
_REDOWNLOAD_NOTICE = (
    "> [!NOTE]\n"
    "> **Updated {date}:** Re-quantized with oMLX v{version}. "
    "Please re-download if you have an earlier copy."
)


def _read_safetensors_header(path: Path) -> dict:
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
    header.pop("__metadata__", None)
    return header


def _quantization_specs(config: dict) -> tuple[dict, dict]:
    """Return the default and per-module quantization specs of an oQ output."""
    v41 = config.get("omlx_deepseek_v41")
    if isinstance(v41, dict):
        modules = dict(v41.get("quantized_modules") or {})
        for table in (v41.get("engram_tables") or {}).values():
            modules[table["weight_key"].rsplit(".", 1)[0]] = {
                "bits": table.get("bits"),
                "group_size": table.get("group_size"),
                "mode": table.get("mode", "affine"),
            }
        return {}, modules
    quant = config.get("quantization") or config.get("quantization_config") or {}
    default = {k: quant[k] for k in ("bits", "group_size", "mode") if k in quant}
    return default, {k: v for k, v in quant.items() if isinstance(v, dict)}


def _word_bits(weight: dict) -> int:
    """MLX packs quantized weights into uint32; V4.1 FP8 tables keep raw bytes."""
    return 32 if weight["dtype"] == "U32" else 8


def _storage_format(
    spec: dict, weight: dict, scales: dict, has_biases: bool
) -> tuple[str, int]:
    """Return the table label and bit width of one quantized module."""
    mode = spec.get("mode") or ("affine" if has_biases else None)
    group = spec.get("group_size") or _MX_GROUP_SIZES.get(mode or "")
    group = group or (64 if has_biases else 32)
    # Each word packs word_bits / bits weights, and each scale covers a group.
    bits = spec.get("bits") or round(
        weight["shape"][-1] * _word_bits(weight) / (scales["shape"][-1] * group)
    )
    mode = mode or f"mxfp{bits}"
    if mode == "affine":
        return f"Affine {bits}-bit, group {group}", bits
    return f"{mode.upper()}, group {group}", bits


def _card_component(key: str, shard: str, config: dict) -> str:
    if shard == CLEF_HEAD_WEIGHTS:
        return "Joint schema head"
    if shard == D3_READOUT_WEIGHTS:
        return "Decision readout"
    if _is_qwen4_exp_ngram_embedding_tensor(key, config):
        return "N-gram embedding"
    if config.get("model_type") == "deepseek_v41" and ".engram.embed" in key:
        return "Engram tables"
    if _is_mtp_tensor(key):
        return "MTP head"
    # V4.1 names its vision encoder and aligner without a vision_ marker.
    v41_vision = config.get("model_type") == "deepseek_v41" and key.startswith(
        ("vision.", "aligner.", "image_")
    )
    if v41_vision or _is_vision_tensor(key):
        return "Vision"
    if _is_audio_tensor(key):
        return "Audio"
    return "Language model"


def _storage_summary(model_path: Path, config: dict) -> tuple[dict, dict]:
    """Sum logical weights and stored bytes by storage format and component.

    Only safetensors headers are read, so this stays cheap for any model size.
    """
    tensors = {}
    for shard in sorted(model_path.glob("*.safetensors")):
        for key, info in _read_safetensors_header(shard).items():
            start, end = info["data_offsets"]
            tensors[key] = {**info, "bytes": end - start, "shard": shard.name}
    default, modules = _quantization_specs(config)
    formats: dict[str, list] = {}  # label -> [weights, bytes, bits]
    components: dict[str, list] = {}  # component -> [weights, bytes]
    for key, info in tensors.items():
        module = key.rsplit(".", 1)[0]
        if key.endswith((".scales", ".biases")) and module + ".weight" in tensors:
            continue
        component = _card_component(key, info["shard"], config)
        scales = tensors.get(module + ".scales")
        if key.endswith(".weight") and scales is not None:
            biases = tensors.get(module + ".biases")
            label, bits = _storage_format(
                {**default, **modules.get(module, {})}, info, scales, biases is not None
            )
            weights = math.prod(info["shape"]) * _word_bits(info) // bits
            stored = info["bytes"] + scales["bytes"] + (biases or {}).get("bytes", 0)
            if component in _LOOKUP_TABLES:
                if not label[1].isupper():
                    label = label[0].lower() + label[1:]
                label = f"{component.split()[0]} {label}"
        else:
            weights = math.prod(info["shape"])
            stored = info["bytes"]
            label = _DTYPE_LABELS.get(info["dtype"], info["dtype"])
            bits = 8 * stored // max(weights, 1)
        row = formats.setdefault(label, [0, 0, bits])
        row[0] += weights
        row[1] += stored
        part = components.setdefault(component, [0, 0])
        part[0] += weights
        part[1] += stored
    return formats, components


def _generate_model_card(
    model_name: str,
    model_path: Path,
    config: dict,
    redownload_notice: bool = False,
) -> str:
    """Generate a HuggingFace model card with the oQ bit distribution."""
    from omlx._version import __version__

    formats, components = _storage_summary(model_path, config)
    total_w = max(sum(row[0] for row in formats.values()), 1)
    total_b = sum(row[1] for row in formats.values())
    gib = 1024**3
    # A bundled DSpark draft keeps its own config and shards in a subfolder.
    draft = bundled_dspark_path(model_path)
    draft_w = draft_b = 0
    if draft is not None:
        draft_formats, _ = _storage_summary(
            draft, json.loads((draft / "config.json").read_text())
        )
        draft_w = sum(row[0] for row in draft_formats.values())
        draft_b = sum(row[1] for row in draft_formats.values())

    has_mtp = "MTP head" in components or draft is not None
    tags = ["mlx", "oq", "quantized"] + (["mtp"] if has_mtp else [])
    lines = ["---", "library_name: mlx", "tags:", *(f"- {tag}" for tag in tags)]
    lines += ["---", "", f"# {model_name}", ""]
    if redownload_notice:
        today = date.today().isoformat()
        lines += [_REDOWNLOAD_NOTICE.format(date=today, version=__version__), ""]
    engram = list(
        ((config.get("omlx_deepseek_v41") or {}).get("engram_tables") or {}).values()
    )
    if engram and all(table.get("mode") == "mxfp8" for table in engram):
        lines += ["N-gram (Engram) tables are kept at the original FP8 precision.", ""]
    lines += [
        "This model was quantized using [oQ](https://github.com/jundot/omlx) "
        f"(oMLX v{__version__}) mixed-precision quantization.",
        "",
    ]

    average = f"{8 * total_b / total_w:.3f} bits/weight"
    scope = "the complete checkpoint" if draft is None else "the model weights"
    with_draft = (
        ""
        if draft is None
        else f", or {8 * (total_b + draft_b) / (total_w + draft_w):.3f} "
        "bits/weight including the DSpark draft"
    )
    tables = [name for name in _LOOKUP_TABLES if name in components]
    if tables:
        table_w = sum(components[name][0] for name in tables)
        table_b = sum(components[name][1] for name in tables)
        excluded = " and ".join(_LOOKUP_TABLES[name] for name in tables)
        lines.append(
            "The effective average is "
            f"{8 * (total_b - table_b) / max(total_w - table_w, 1):.3f} "
            f"bits/weight excluding {excluded}, or {average} for {scope}"
            f"{with_draft}."
        )
    else:
        lines.append(f"The effective average is {average} for {scope}{with_draft}.")

    lines += [
        "",
        "## Quantization and Bit Distribution",
        "",
        "| Storage format | Logical weights | Share of total | Tensor storage "
        "| Effective bits/weight |",
        "|---|---|---|---|---|",
    ]
    table_prefixes = tuple(name.split()[0] + " " for name in tables)
    order = sorted(
        formats,
        key=lambda label: (
            bool(table_prefixes) and label.startswith(table_prefixes),
            formats[label][2],
            label,
        ),
    )
    for label in order:
        weights, stored, _ = formats[label]
        lines.append(
            f"| {label} | {weights / 1e9:.3f}B | {100 * weights / total_w:.3f}% | "
            f"{stored / gib:.3f} GiB | {8 * stored / max(weights, 1):.2f} |"
        )
    lines.append(
        f"| **Total** | **{total_w / 1e9:.3f}B** | **100%** | "
        f"**{total_b / gib:.3f} GiB** | **{8 * total_b / total_w:.3f}** |"
    )

    lines += [
        "",
        "## Storage Breakdown",
        "",
        "| Component | Logical weights | Storage |",
        "|---|---|---|",
    ]
    parts = [
        (name, *components[name]) for name in _COMPONENT_ORDER if name in components
    ]
    if draft is not None:
        parts.append(("DSpark draft (MTP)", draft_w, draft_b))
    for name, weights, stored in parts:
        lines.append(
            f"| {name} | {weights / 1e9:.3f}B | "
            f"{stored / 1e9:.3f} GB / {stored / gib:.3f} GiB |"
        )
    all_w, all_b = total_w + draft_w, total_b + draft_b
    lines.append(
        f"| **Total** | **{all_w / 1e9:.3f}B** | "
        f"**{all_b / 1e9:.3f} GB / {all_b / gib:.3f} GiB** |"
    )
    return "\n".join(lines) + "\n"


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

    def update_model_dirs(self, model_dirs: list[str]) -> None:
        """Update model directory paths."""
        self._model_dirs = [Path(d) for d in model_dirs]

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
                        has_readme = _has_meaningful_readme(path)
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
        redownload_notice: bool = False,
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
            self._run_upload(task_id, token, readme_source_path, auto_readme, redownload_notice, private)
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
        redownload_notice: bool,
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
                elif auto_readme and not _has_meaningful_readme(model_path):
                    try:
                        with open(model_path / "config.json") as f:
                            config = json.load(f)
                    except Exception:
                        config = {}
                    readme_content = _generate_model_card(
                        task.model_name,
                        model_path,
                        config,
                        redownload_notice=redownload_notice,
                    )
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
