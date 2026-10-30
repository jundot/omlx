# SPDX-License-Identifier: Apache-2.0
"""mlx-vlm-backed image generation and editing engine for oMLX."""

from __future__ import annotations

import asyncio
import gc
import importlib
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import mlx.core as mx

from ..engine_core import get_mlx_executor
from ..utils.optional_deps import MLX_VLM_MISSING_MESSAGE
from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)

ImageTask = Literal["generation", "edit"]


def _cleanup_mlx_cache() -> None:
    gc.collect()
    mx.synchronize()
    mx.clear_cache()


def _clear_mlx_cache() -> None:
    mx.synchronize()
    mx.clear_cache()


async def _await_image_worker(future: asyncio.Future[Any]) -> Any:
    cancelled = False
    while not future.done():
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            if not cancelled:
                raise
    if cancelled:
        error = future.exception()
        if error is not None:
            logger.error(
                "Cancelled image worker failed",
                exc_info=(type(error), error, error.__traceback__),
            )
        raise asyncio.CancelledError
    return future.result()


@dataclass(frozen=True)
class _ImageAPI:
    load_image_model: Any
    generate_image: Any
    ImageGenerationRequest: Any
    ImageEditRequest: Any


@dataclass(slots=True)
class ImageEngineResult:
    """A route-friendly image result returned by an mlx-vlm model."""

    image: Any
    metadata: dict[str, Any] = field(default_factory=dict)


def _image_api() -> _ImageAPI:
    """Import mlx-vlm's generic image API after registering oMLX shims."""
    try:
        from ..patches.mlx_vlm_image_compat import (
            apply_mlx_vlm_image_compat_patch,
        )

        apply_mlx_vlm_image_compat_patch()
        image = importlib.import_module("mlx_vlm.generate.image")
        edit_image = importlib.import_module("mlx_vlm.generate.edit_image")
    except (AttributeError, ImportError) as exc:
        raise ImportError(MLX_VLM_MISSING_MESSAGE) from exc

    return _ImageAPI(
        load_image_model=image.load_image_model,
        generate_image=image.generate_image,
        ImageGenerationRequest=image.ImageGenerationRequest,
        ImageEditRequest=edit_image.ImageEditRequest,
    )


class ImageEngine(BaseNonStreamingEngine):
    """Non-streaming image engine using mlx-vlm model loaders and requests.

    mlx-vlm owns model detection, per-model sampling defaults and input
    validation; this engine only serializes calls and manages loaded variants.
    """

    def __init__(
        self,
        model_name: str,
        *,
        model_id: str | None = None,
        model_path: str | None = None,
        tasks: list[str] | None = None,
        model_settings: object | None = None,
    ) -> None:
        super().__init__()
        self._model_name = model_name
        self._model_id = model_id or model_name
        self._model_path = model_path or model_name
        self._tasks = self._normalize_tasks(tasks)
        self._model_settings = model_settings
        self._models: dict[ImageTask, Any] = {}
        self._started = False
        self._call_lock = asyncio.Lock()
        self._load_lock = asyncio.Lock()

    @property
    def model_name(self) -> str:
        """Get the API/model name associated with this engine."""
        return self._model_name

    async def start(self) -> None:
        """Load the primary image task."""
        if self._started:
            return
        if not self._tasks:
            raise ValueError(
                f"Image model {self._model_id} declares no supported tasks"
            )

        logger.info(
            "Starting mlx-vlm image engine: model=%s, tasks=%s",
            self._model_id,
            ",".join(self._tasks),
        )
        await self._load_model_for_task(self._primary_task())
        self._started = True

    def _primary_task(self) -> ImageTask:
        """Choose the task whose weights are accounted during pool admission."""
        if "generation" in self._tasks:
            return "generation"
        return "edit"

    async def stop(self) -> None:
        """Drop the loaded image variant and release MLX buffers."""
        await self._stop(clear_mlx_cache=True)

    async def stop_without_global_cleanup(self) -> None:
        """Drop this model without synchronizing another active MLX workload."""
        await self._stop(clear_mlx_cache=False)

    async def _stop(self, *, clear_mlx_cache: bool) -> None:
        if not self._models and not self._started:
            return

        async with self._call_lock:
            logger.info("Stopping image engine: %s", self._model_id)
            self._models.clear()
            self._started = False
            if clear_mlx_cache:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(get_mlx_executor(), _cleanup_mlx_cache)
            else:
                gc.collect()

    async def generate(
        self,
        prompt: str,
        width: int | None = None,
        height: int | None = None,
        seed: int | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        output_format: str = "png",
        **kwargs: Any,
    ) -> ImageEngineResult:
        """Generate one image through mlx-vlm's generic generation API."""
        self._ensure_started()
        self._ensure_task_supported("generation")

        activity_id = self._begin_activity(
            "generating image",
            detail="Generating image",
            metadata={"prompt_length": len(prompt), "width": width, "height": height},
        )
        try:
            async with self._call_lock:
                model = await self._load_model_for_task("generation")
                api = _image_api()
                # mlx-vlm's generation request has non-null size defaults.
                size = {
                    name: int(value)
                    for name, value in (("width", width), ("height", height))
                    if value is not None
                }
                request = api.ImageGenerationRequest(
                    prompt=prompt,
                    seed=0 if seed is None else int(seed),
                    steps=None if steps is None else int(steps),
                    guidance=None if guidance is None else float(guidance),
                    output_format="png",
                    extra=self._request_extra(kwargs),
                    **size,
                )
                raw_result = await self._run_api_call(
                    api.generate_image,
                    model,
                    request,
                    task="generate",
                )
                return self._result_from_raw(
                    raw_result,
                    task="generation",
                    output_format=output_format,
                )
        finally:
            await self._finish_activity(activity_id)

    async def edit(
        self,
        prompt: str,
        image_paths: list[str],
        mask_path: str | None = None,
        width: int | None = None,
        height: int | None = None,
        seed: int | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        output_format: str = "png",
        **kwargs: Any,
    ) -> ImageEngineResult:
        """Edit input images through mlx-vlm's generic edit API."""
        self._ensure_started()
        self._ensure_task_supported("edit")
        if not image_paths:
            raise ValueError("edit requires at least one input image path")
        if mask_path is not None:
            raise ValueError("mlx-vlm image models do not support masks")

        activity_id = self._begin_activity(
            "editing image",
            detail="Editing image",
            metadata={"prompt_length": len(prompt), "width": width, "height": height},
        )
        try:
            async with self._call_lock:
                model = await self._load_model_for_task("edit")
                api = _image_api()
                request = api.ImageEditRequest(
                    prompt=prompt,
                    image_paths=tuple(image_paths),
                    seed=0 if seed is None else int(seed),
                    steps=None if steps is None else int(steps),
                    width=None if width is None else int(width),
                    height=None if height is None else int(height),
                    guidance=None if guidance is None else float(guidance),
                    output_format="png",
                    extra=self._request_extra(kwargs),
                )
                raw_result = await self._run_api_call(
                    api.generate_image,
                    model,
                    request,
                    task="edit",
                )
                return self._result_from_raw(
                    raw_result,
                    task="edit",
                    output_format=output_format,
                    input_image_count=len(image_paths),
                )
        finally:
            await self._finish_activity(activity_id)

    def get_stats(self) -> dict[str, Any]:
        """Get image engine statistics."""
        return {
            "model_name": self._model_name,
            "model_id": self._model_id,
            "loaded": self._started,
            "tasks": list(self._tasks),
            "loaded_tasks": sorted(self._models),
        }

    @staticmethod
    def _normalize_tasks(raw_tasks: list[str] | None) -> list[ImageTask]:
        tasks = set(raw_tasks or ())
        unsupported = tasks - {"generation", "edit"}
        if unsupported:
            raise ValueError(f"Unsupported image model tasks: {sorted(unsupported)}")
        return [task for task in ("generation", "edit") if task in tasks]

    def _ensure_started(self) -> None:
        if not self._started:
            raise RuntimeError("Engine not started. Call start() first.")

    def _ensure_task_supported(self, task: ImageTask) -> None:
        if task not in self._tasks:
            raise ValueError(
                f"Image model {self._model_id} does not support task {task!r}"
            )

    async def _load_model_for_task(self, task: ImageTask) -> Any:
        model = self._models.get(task)
        if model is not None:
            return model

        async with self._load_lock:
            model = self._models.get(task)
            if model is not None:
                return model

            if self._models:
                logger.info(
                    "Unloading previous mlx-vlm image variant before loading: "
                    "model=%s, task=%s",
                    self._model_id,
                    task,
                )
                self._models.clear()
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(get_mlx_executor(), _cleanup_mlx_cache)

            api = _image_api()
            loader_task = "generate" if task == "generation" else "edit"

            def load_sync() -> Any:
                return api.load_image_model(self._model_path, task=loader_task)

            logger.info(
                "Loading mlx-vlm image model: model=%s, task=%s, model_path=%s",
                self._model_id,
                task,
                self._model_path,
            )
            loop = asyncio.get_running_loop()
            try:
                model = await _await_image_worker(
                    loop.run_in_executor(get_mlx_executor(), load_sync)
                )
            except ImportError as exc:
                raise ImportError(MLX_VLM_MISSING_MESSAGE) from exc
            self._models[task] = model
            return model

    @staticmethod
    async def _run_api_call(
        generate_image: Any,
        model: Any,
        request: Any,
        *,
        task: str,
    ) -> Any:
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            get_mlx_executor(),
            lambda: generate_image(model, request, task=task),
        )
        return await _await_image_worker(future)

    @staticmethod
    def _request_extra(kwargs: dict[str, Any]) -> dict[str, Any]:
        extra = {key: value for key, value in kwargs.items() if value is not None}
        image_strength = extra.pop("image_strength", None)
        if image_strength is not None:
            image_strength = float(image_strength)
            # Z-Image consumes ``strength`` while ERNIE-Image consumes
            # ``image_strength``. Other mlx-vlm families safely ignore extras.
            extra["image_strength"] = image_strength
            extra["strength"] = image_strength
        return extra

    def _result_from_raw(
        self,
        raw_result: Any,
        *,
        task: ImageTask,
        output_format: str,
        input_image_count: int | None = None,
    ) -> ImageEngineResult:
        raw_metadata = getattr(raw_result, "metadata", {})
        metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
        metadata.update(
            {
                "model_id": self._model_id,
                "model_name": self._model_name,
                "task": task,
                "output_format": output_format,
                "model_path": self._model_path,
            }
        )
        if input_image_count is not None:
            metadata["input_image_count"] = input_image_count
        for attribute in (
            "seed",
            "prompt",
            "steps",
            "guidance",
            "width",
            "height",
            "model",
            "family",
            "variant",
            "prompt_tokens",
            "peak_memory",
        ):
            value = getattr(raw_result, attribute, None)
            if value is not None:
                metadata[attribute] = value
        return ImageEngineResult(
            image=getattr(raw_result, "image", raw_result),
            metadata=metadata,
        )

    async def _finish_activity(self, activity_id: str) -> None:
        self._end_activity(activity_id)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(get_mlx_executor(), _clear_mlx_cache)

    def __repr__(self) -> str:
        status = "running" if self._started else "stopped"
        return f"<ImageEngine model={self._model_id} status={status}>"
