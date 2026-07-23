# SPDX-License-Identifier: Apache-2.0
"""oQ Quantization task manager for the admin panel.

Manages quantization tasks with progress tracking, following the same pattern
as hf_downloader.py (DownloadTask / HFDownloader).
"""

import asyncio
import enum
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from ..model_discovery import _has_vision_subconfig

logger = logging.getLogger(__name__)


class _QuantCancelled(Exception):
    """Raised by progress callback when task is cancelled."""

    pass


class QuantStatus(str, enum.Enum):
    """Status of a quantization task."""

    PENDING = "pending"
    LOADING = "loading"
    QUANTIZING = "quantizing"
    SAVING = "saving"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


_ACTIVE_STATUSES = {
    QuantStatus.PENDING,
    QuantStatus.LOADING,
    QuantStatus.QUANTIZING,
    QuantStatus.SAVING,
}


@dataclass
class QuantTask:
    """Represents a single oQ quantization task."""

    task_id: str
    model_name: str
    model_path: str
    oq_level: float
    output_name: str
    output_path: str
    status: QuantStatus = QuantStatus.PENDING
    progress: float = 0.0
    phase: str = ""
    progress_detail: str = ""
    progress_meta: dict = field(default_factory=dict)
    error: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0
    source_size: int = 0
    output_size: int = 0
    group_size: int = 64
    sensitivity_model_path: str = ""
    text_only: bool = False
    dtype: str = "bfloat16"
    preserve_mtp: bool = False
    auto_proxy_sensitivity: bool = True
    enhanced: bool = False
    imatrix_cache_path: str = ""
    imatrix_reuse_cache: bool = True
    imatrix_strict: bool = False
    imatrix_num_samples: int = 128
    imatrix_seq_length: int = 512

    def to_dict(self) -> dict:
        """Serialize task to JSON-compatible dict."""
        return {
            "task_id": self.task_id,
            "model_name": self.model_name,
            "model_path": self.model_path,
            "oq_level": self.oq_level,
            "output_name": self.output_name,
            "output_path": self.output_path,
            "status": self.status.value,
            "progress": round(self.progress, 1),
            "phase": self.phase,
            "progress_detail": self.progress_detail,
            "progress_meta": self.progress_meta,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "source_size": self.source_size,
            "output_size": self.output_size,
            "dtype": self.dtype,
            "enhanced": self.enhanced,
            "imatrix_cache_path": self.imatrix_cache_path,
        }


def _dir_size(path: Path) -> int:
    """Get total size of files in a directory."""
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _format_size(size_bytes: int) -> str:
    """Format byte count as human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.1f} MB"
    else:
        return f"{size_bytes / 1024**3:.1f} GB"


class OQManager:
    """Manages oQ quantization tasks with async execution and progress tracking.

    Follows the same pattern as HFDownloader: semaphore-guarded sequential
    execution, polling-based progress, cooperative cancellation.
    """

    def __init__(
        self,
        model_dirs: list[str],
        on_complete: Optional[Callable] = None,
    ):
        self._model_dirs = [Path(d) for d in model_dirs]
        self._output_dir = self._model_dirs[0] if self._model_dirs else Path(".")
        self._tasks: dict[str, QuantTask] = {}
        self._active_tasks: dict[str, asyncio.Task] = {}
        self._progress_tasks: dict[str, asyncio.Task] = {}
        self._on_complete = on_complete
        self._cancelled: set[str] = set()
        self._quant_sem = asyncio.Semaphore(1)

    def update_model_dirs(self, model_dirs: list[str]) -> None:
        """Update model directory paths."""
        self._model_dirs = [Path(d) for d in model_dirs]
        if self._model_dirs:
            self._output_dir = self._model_dirs[0]

    async def list_quantizable_models(self) -> tuple[list[dict], list[dict]]:
        """Scan all model dirs. Returns (source_models, all_models)."""

        def _scan() -> tuple[list[dict], list[dict]]:
            from ..oq import validate_quantizable, estimate_memory
            from ..utils.model_loading import _checkpoint_has_mtp_weights

            source_models = []
            all_models = []
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
                        try:
                            with open(path / "config.json") as f:
                                config = json.load(f)
                            size = sum(
                                f.stat().st_size for f in path.glob("*.safetensors")
                            )
                            if size == 0:
                                size = sum(f.stat().st_size for f in path.glob("*.bin"))
                            if size == 0:
                                continue
                            tc = config.get("text_config", {})
                            # has_mtp_heads: top-level OR text_config nested
                            config_declares_mtp = (
                                int(config.get("mtp_num_hidden_layers", 0) or 0) > 0
                                or int(config.get("num_nextn_predict_layers", 0) or 0)
                                > 0
                                or int(tc.get("mtp_num_hidden_layers", 0) or 0) > 0
                                or int(tc.get("num_nextn_predict_layers", 0) or 0) > 0
                            )
                            has_mtp = (
                                config_declares_mtp
                                and _checkpoint_has_mtp_weights(path)
                            )
                            info = {
                                "name": path.name,
                                "path": str(path),
                                "size": size,
                                "size_formatted": _format_size(size),
                                "model_type": config.get("model_type", "")
                                or tc.get("model_type", ""),
                                "is_quantized": "quantization" in config,
                                # Treat vision_config / vit_config / mm_vision_tower as VLM
                                # evidence (Molmo / Molmo2 use vit_config; FastVLM uses
                                # mm_vision_tower). Same predicate as model_discovery.
                                "is_vlm": _has_vision_subconfig(config),
                                "has_mtp_heads": has_mtp,
                            }
                            all_models.append(info)
                            if validate_quantizable(config):
                                info_full = dict(info)
                                info_full["num_layers"] = config.get(
                                    "num_hidden_layers", 0
                                ) or tc.get("num_hidden_layers", 0)
                                info_full["num_experts"] = config.get(
                                    "num_local_experts", 0
                                )
                                info_full["memory_streaming"] = estimate_memory(size)
                                source_models.append(info_full)
                        except Exception:
                            continue
            return source_models, all_models

        return await asyncio.to_thread(_scan)

    async def start_quantization(
        self,
        model_path: str,
        oq_level: float,
        group_size: int = 64,
        sensitivity_model_path: str = "",
        text_only: bool = False,
        dtype: str = "bfloat16",
        preserve_mtp: bool = False,
        auto_proxy_sensitivity: bool = True,
        enhanced: bool = False,
        imatrix_cache_path: str = "",
        imatrix_reuse_cache: bool = True,
        imatrix_strict: bool = False,
        imatrix_num_samples: int = 128,
        imatrix_seq_length: int = 512,
    ) -> QuantTask:
        """Start a quantization job.

        Args:
            model_path: Path to source model directory.
            oq_level: oQ level from OQ_LEVELS.
            dtype: Target fp dtype for non-quantized weights and quant
                scales/biases. "bfloat16" (default) or "float16".

        Returns:
            The created QuantTask.

        Raises:
            ValueError: On invalid inputs or output conflict.
        """
        from ..oq import (
            OQ_DTYPES,
            OQ_LEVELS,
            _validate_oq_dtype_for_model,
            resolve_output_name,
        )
        from ..utils.model_loading import _checkpoint_has_mtp_weights

        if oq_level not in OQ_LEVELS:
            raise ValueError(
                f"Invalid oQ level {oq_level}. Must be one of {sorted(OQ_LEVELS)}"
            )
        if dtype not in OQ_DTYPES:
            raise ValueError(f"Invalid dtype {dtype!r}. Must be one of {OQ_DTYPES}")

        source = Path(model_path)
        if not source.exists() or not (source / "config.json").exists():
            raise ValueError(f"Model not found: {model_path}")

        with open(source / "config.json") as f:
            config = json.load(f)
        _validate_oq_dtype_for_model(config, dtype)

        if preserve_mtp and not _checkpoint_has_mtp_weights(source):
            logger.warning(
                "Preserve MTP requested for %s, but no mtp.* tensors were "
                "found in the checkpoint; disabling MTP preservation",
                source.name,
            )
            preserve_mtp = False

        model_name = source.name
        output_name = resolve_output_name(
            model_name,
            oq_level,
            dtype,
            preserve_mtp=preserve_mtp,
            enhanced=enhanced,
        )
        output_path = self._output_dir / output_name

        if output_path.exists():
            raise ValueError(
                f"Output directory already exists: {output_path}. "
                "Delete it first via the Manager tab."
            )

        # Check for duplicate active tasks (same level + dtype combo)
        for task in self._tasks.values():
            if (
                task.model_path == model_path
                and task.oq_level == oq_level
                and task.dtype == dtype
                and task.enhanced == enhanced
                and task.status in _ACTIVE_STATUSES
            ):
                raise ValueError(
                    f"Quantization for '{model_name}' at oQ{oq_level:g}"
                    f"{'e' if enhanced else ''} "
                    f"({dtype}) is already in progress"
                )

        if enhanced:
            if imatrix_num_samples < 1:
                raise ValueError("imatrix_num_samples must be >= 1")
            if imatrix_seq_length < 1:
                raise ValueError("imatrix_seq_length must be >= 1")
            if not imatrix_cache_path:
                digest = hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:12]
                imatrix_cache_path = str(
                    self._output_dir
                    / ".oqe_imatrix"
                    / (
                        f"{model_name}-{digest}-s{int(imatrix_num_samples)}"
                        f"-l{int(imatrix_seq_length)}.npz"
                    )
                )

        source_size = sum(f.stat().st_size for f in source.glob("*.safetensors"))
        if source_size == 0:
            source_size = sum(f.stat().st_size for f in source.glob("*.bin"))

        task_id = str(uuid.uuid4())
        task = QuantTask(
            task_id=task_id,
            model_name=model_name,
            model_path=model_path,
            oq_level=oq_level,
            output_name=output_name,
            output_path=str(output_path),
            source_size=source_size,
            group_size=group_size,
            sensitivity_model_path=sensitivity_model_path,
            text_only=text_only,
            dtype=dtype,
            preserve_mtp=preserve_mtp,
            auto_proxy_sensitivity=auto_proxy_sensitivity,
            enhanced=enhanced,
            imatrix_cache_path=imatrix_cache_path,
            imatrix_reuse_cache=imatrix_reuse_cache,
            imatrix_strict=imatrix_strict,
            imatrix_num_samples=imatrix_num_samples,
            imatrix_seq_length=imatrix_seq_length,
        )
        self._tasks[task_id] = task

        self._active_tasks[task_id] = asyncio.create_task(
            self._run_quantization(task_id)
        )

        logger.info(
            f"oQ quantization queued: {model_name} -> "
            f"oQ{oq_level:g}{'e' if enhanced else ''} "
            f"(task_id={task_id})"
        )
        return task

    async def cancel_quantization(self, task_id: str) -> bool:
        """Cancel an active quantization task."""
        task = self._tasks.get(task_id)
        if task is None:
            return False
        if task.status not in _ACTIVE_STATUSES:
            return False

        self._cancelled.add(task_id)
        task.status = QuantStatus.CANCELLED

        progress_task = self._progress_tasks.pop(task_id, None)
        if progress_task and not progress_task.done():
            progress_task.cancel()

        active_task = self._active_tasks.pop(task_id, None)

        # Clean up partial output
        output = Path(task.output_path)
        if output.exists():
            import shutil

            shutil.rmtree(output, ignore_errors=True)

        # Wait for the quantization thread to actually finish.
        # Do NOT call active_task.cancel() first — that only cancels the
        # asyncio wrapper and causes the await to return immediately while
        # the OS thread continues running Metal commands. Instead, rely on
        # cooperative cancellation: the progress callback raises
        # _QuantCancelled when it sees the flag, terminating quantize_oq
        # at the next callback point (per-layer in GPTQ, per-tensor in
        # streaming).
        if active_task and not active_task.done():
            try:
                await asyncio.wait_for(
                    asyncio.shield(active_task),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                # Thread didn't exit cooperatively (e.g. stuck in long GPTQ
                # block). Force-cancel as last resort and wait a bit for
                # Metal to settle.
                logger.warning(
                    "oQ cancel: cooperative exit timed out, force-cancelling"
                )
                active_task.cancel()
                try:
                    await active_task
                except (asyncio.CancelledError, Exception):
                    pass
                await asyncio.sleep(2.0)
            except (asyncio.CancelledError, Exception):
                pass

        # GPU cleanup after thread is done
        if HAS_MLX:
            for _attempt in range(3):
                try:
                    mx.synchronize()
                    mx.clear_cache()
                    break
                except Exception:
                    await asyncio.sleep(1.0)

        logger.info(f"oQ quantization cancelled: {task.model_name} (task_id={task_id})")
        return True

    def remove_task(self, task_id: str) -> bool:
        """Remove a completed/failed/cancelled task from the list."""
        task = self._tasks.get(task_id)
        if task is None:
            return False
        if task.status in _ACTIVE_STATUSES:
            return False
        del self._tasks[task_id]
        self._cancelled.discard(task_id)
        return True

    def get_tasks(self) -> list[dict]:
        """Return all tasks as serializable dicts."""
        return [t.to_dict() for t in self._tasks.values()]

    @property
    def is_quantizing(self) -> bool:
        """Check if any quantization task is actively running."""
        return any(t.status in _ACTIVE_STATUSES for t in self._tasks.values())

    async def shutdown(self) -> None:
        """Cancel all active tasks."""
        for task_id in list(self._active_tasks):
            await self.cancel_quantization(task_id)

    async def _run_quantization(self, task_id: str) -> None:
        """Execute the quantization pipeline in background."""
        task = self._tasks[task_id]
        try:
            async with self._quant_sem:
                if task_id in self._cancelled:
                    return

                # Ensure GPU is clean before starting (previous task may have been cancelled)
                # Metal command buffers need full sync + cache clear after cancellation
                if HAS_MLX:
                    for _ in range(3):
                        try:
                            mx.synchronize()
                            mx.clear_cache()
                            break
                        except Exception:
                            await asyncio.sleep(1.0)

                # Phase 1: Loading
                task.status = QuantStatus.LOADING
                task.started_at = time.time()
                task.phase = "Loading model..."
                task.progress = 5.0

                def _progress_cb(
                    phase: str,
                    pct: float,
                    detail: str = "",
                    meta: dict | None = None,
                ) -> None:
                    if task_id in self._cancelled:
                        raise _QuantCancelled(f"Task {task_id} cancelled")
                    base_phase = phase.split("|", 1)[0]
                    if base_phase.startswith("quantizing"):
                        task.status = QuantStatus.QUANTIZING
                    elif base_phase == "saving":
                        task.status = QuantStatus.SAVING
                    else:
                        task.status = QuantStatus.LOADING
                    task.phase = self._phase_label(phase, task.oq_level, task.enhanced)
                    task.progress_detail = detail or ""
                    task.progress_meta = meta or {}
                    task.progress = pct
                    task._last_progress_callback_at = time.time()

                # Start time-based progress estimation
                self._progress_tasks[task_id] = asyncio.create_task(
                    self._estimate_progress(task_id)
                )

                from ..oq import quantize_oq_streaming

                await asyncio.to_thread(
                    quantize_oq_streaming,
                    task.model_path,
                    task.output_path,
                    task.oq_level,
                    task.group_size,
                    _progress_cb,
                    task.text_only,
                    None,  # target_bpw
                    None,  # hard_cap_bpw
                    task.sensitivity_model_path,
                    task.dtype,
                    task.preserve_mtp,
                    task.auto_proxy_sensitivity,
                    enhanced=task.enhanced,
                    imatrix_cache_path=task.imatrix_cache_path,
                    imatrix_reuse_cache=task.imatrix_reuse_cache,
                    imatrix_strict=task.imatrix_strict,
                    imatrix_num_samples=task.imatrix_num_samples,
                    imatrix_seq_length=task.imatrix_seq_length,
                )

                if task_id in self._cancelled:
                    return

                # Complete
                task.status = QuantStatus.COMPLETED
                task.progress = 100.0
                task.phase = "Completed"
                task.progress_detail = ""
                task.progress_meta = {}
                task.completed_at = time.time()
                task.output_size = _dir_size(Path(task.output_path))

                elapsed = task.completed_at - task.started_at
                logger.info(
                    f"oQ quantization completed: {task.output_name} "
                    f"({elapsed:.0f}s, {_format_size(task.output_size)})"
                )

                if self._on_complete:
                    try:
                        result = self._on_complete()
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception:
                        logger.exception("on_complete callback failed")

        except asyncio.CancelledError:
            if task.status not in (QuantStatus.CANCELLED, QuantStatus.FAILED):
                task.status = QuantStatus.CANCELLED
        except _QuantCancelled:
            if task.status != QuantStatus.CANCELLED:
                task.status = QuantStatus.CANCELLED
        except Exception as e:
            if task_id not in self._cancelled:
                task.status = QuantStatus.FAILED
                task.error = str(e)
                task.completed_at = time.time()
                logger.exception(f"oQ quantization failed: {task.model_name} -> {e}")
                # Clean up partial output
                output = Path(task.output_path)
                if output.exists():
                    import shutil

                    shutil.rmtree(output, ignore_errors=True)
        finally:
            pt = self._progress_tasks.pop(task_id, None)
            if pt and not pt.done():
                pt.cancel()
            self._active_tasks.pop(task_id, None)

    async def _estimate_progress(self, task_id: str) -> None:
        """Estimate progress by time during quantize phase (30-90%)."""
        task = self._tasks.get(task_id)
        if task is None:
            return

        source_gb = max(task.source_size / (1024**3), 0.1)
        estimated_total = source_gb * 3.0
        start = time.time()

        try:
            while task_id not in self._cancelled and task.status in _ACTIVE_STATUSES:
                await asyncio.sleep(2)
                elapsed = time.time() - start
                if time.time() - getattr(task, "_last_progress_callback_at", 0.0) < 5:
                    continue
                if task.status == QuantStatus.QUANTIZING:
                    if self._has_explicit_quant_progress(task):
                        continue
                    fraction = min(elapsed / estimated_total, 0.95)
                    task.progress = max(task.progress, 30.0 + fraction * 60.0)
                elif task.status == QuantStatus.SAVING:
                    # During save, poll output dir size
                    output = Path(task.output_path)
                    if output.exists() and task.source_size > 0:
                        current = _dir_size(output)
                        # Estimate output as source * (oq_level / 16)
                        expected = task.source_size * task.oq_level / 16
                        if expected > 0:
                            save_frac = min(current / expected, 0.99)
                            task.progress = max(task.progress, 90.0 + save_frac * 10.0)
        except asyncio.CancelledError:
            pass

    @staticmethod
    def _has_explicit_quant_progress(task: QuantTask) -> bool:
        """Return True once the quantizer emits byte-level progress."""
        meta = task.progress_meta if isinstance(task.progress_meta, dict) else {}
        try:
            total_bytes = int(meta.get("total_bytes") or 0)
            processed_bytes = int(meta.get("processed_bytes") or 0)
        except (TypeError, ValueError):
            return False
        return total_bytes > 0 and processed_bytes >= 0

    @staticmethod
    def _phase_label(phase: str, oq_level: float, enhanced: bool = False) -> str:
        """Human-readable phase label."""
        oq_label = f"oQ{oq_level:g}{'e' if enhanced else ''}"
        labels = {
            "loading": "Loading model...",
            "imatrix": "Collecting oQe imatrix...",
            "quantizing": f"Quantizing to {oq_label}...",
            "saving": "Saving quantized model...",
        }
        # Handle progress: "quantizing_eta|792|879|0:02"
        if phase.startswith("quantizing_eta|"):
            parts = phase.split("|")
            current = parts[1] if len(parts) > 1 else "?"
            total = parts[2] if len(parts) > 2 else "?"
            eta = parts[3] if len(parts) > 3 and parts[3] else ""
            pct = (
                int(int(current) / max(int(total), 1) * 100)
                if current.isdigit() and total.isdigit()
                else 0
            )
            label = f"{oq_label}: {pct}%"
            if eta:
                label += f" ({eta} remaining)"
            return label
        return labels.get(phase, phase)
