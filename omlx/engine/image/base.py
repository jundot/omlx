# SPDX-License-Identifier: Apache-2.0
"""
Base image engine wrapping an mflux model.

Concrete families (see ``omlx/engine/image/``) declare their Pydantic
parameter models (``generates_params`` / ``edits_params``); the request
parameters they expose are validated against those models before mflux's
``generate_image`` is called. mflux is imported lazily in ``start()`` so
this module imports cleanly without the optional dependency.
"""

import asyncio
import contextlib
import gc
import io
import logging
import time
from typing import Any

import mlx.core as mx
from PIL import Image
from pydantic import BaseModel

from ...engine_core import get_mlx_executor
from ..base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)


def _mflux_progress(t: int, config: Any, time_steps: Any) -> tuple[int, int]:
    """Compute (current_step, total_steps) for an mflux InLoopCallback."""
    init = getattr(config, "init_time_step", 0) or 0
    total = getattr(config, "num_inference_steps", None)
    if time_steps is not None:
        with contextlib.suppress(TypeError):
            total = len(time_steps)
    if total is None or total <= 0:
        total = 1
    step = max(1, min(total, (t + 1 - init) if init else (t + 1)))
    return step, total


class BaseImageEngine(BaseNonStreamingEngine):
    """Base engine for image generation (no streaming, non-chat)."""

    #: Pydantic model validating text-to-image request parameters.
    generates_params: type[BaseModel]
    #: Pydantic model validating edit request parameters, or None if the
    #: family cannot edit.
    edits_params: type[BaseModel] | None = None

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        super().__init__()
        self._model_name = model_name
        self._model: Any = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def supports_native_streaming(self) -> bool:
        return False

    def validate_generate(self, body: dict[str, Any]) -> BaseModel:
        return self.generates_params.model_validate(body)

    def validate_edit(self, body: dict[str, Any]) -> BaseModel:
        if self.edits_params is None:
            raise ValueError(f"Model {self._model_name} does not support image editing")
        return self.edits_params.model_validate(body)

    def edit_kwargs(self, image_paths: list[str]) -> dict[str, Any]:
        """Map resolved input images to this family's mflux kwargs."""
        return {"image_paths": image_paths} if image_paths else {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._model is not None:
            return

        logger.info(f"Starting ImageEngine: {self._model_name}")

        model_name = self._model_name

        def _load_sync():
            from ...utils.mflux import resolve_mflux_config, resolve_mflux_family

            model_config = resolve_mflux_config(model_name)
            family_cls = resolve_mflux_family(model_config)
            logger.info(
                f"Loading mflux model as {family_cls.__name__} from {model_name}"
            )
            return family_cls(model_path=model_name, model_config=model_config)

        loop = asyncio.get_running_loop()
        try:
            self._model = await loop.run_in_executor(get_mlx_executor(), _load_sync)
        except ImportError as exc:
            raise ImportError(
                "mflux is required for image generation. "
                'Install it with: pip install "omlx[image]"'
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load image model {model_name}: {exc}"
            ) from exc

        logger.info(f"ImageEngine started: {self._model_name}")

    async def stop(self) -> None:
        if self._model is None:
            return

        logger.info(f"Stopping ImageEngine: {self._model_name}")
        self._model = None

        gc.collect()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            get_mlx_executor(), lambda: (mx.synchronize(), mx.clear_cache())
        )

    # ------------------------------------------------------------------
    # Progress
    # ------------------------------------------------------------------

    def _register_progress_callback(self, activity_id: str) -> tuple[Any, Any]:
        """Register a step-progress callback; returns (cb, registry) or (None, None)."""
        model = self._model
        callbacks = getattr(model, "callbacks", None)
        if callbacks is None or not hasattr(callbacks, "register"):
            return None, None
        try:
            from mflux.callbacks.callback import InLoopCallback

            parent = self
            started = time.monotonic()

            class _Progress(InLoopCallback):
                def call_in_loop(
                    self,
                    t: int,
                    seed: int,
                    prompt: str,
                    latents: Any,
                    config: Any,
                    time_steps: Any,
                ) -> None:
                    step, total = _mflux_progress(t, config, time_steps)
                    elapsed = time.monotonic() - started
                    parent._update_activity(
                        activity_id,
                        current_step=step,
                        total_steps=total,
                        steps_per_second=(step / elapsed) if elapsed > 0 else None,
                    )

            progress_cb = _Progress()
            callbacks.register(progress_cb)
            return progress_cb, callbacks
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not register mflux progress callback: %s", exc)
            return None, None

    def _unregister_progress_callback(self, progress_cb: Any, callbacks: Any) -> None:
        if progress_cb is None or callbacks is None:
            return
        try:
            if hasattr(callbacks, "in_loop") and progress_cb in callbacks.in_loop:
                callbacks.in_loop.remove(progress_cb)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def _run_sync(self, activity_id: str, gen_kwargs: dict[str, Any]) -> bytes:
        """Run mflux ``generate_image`` and return PNG bytes."""
        model = self._model
        progress_cb, callbacks = self._register_progress_callback(activity_id)
        try:
            result = model.generate_image(**gen_kwargs)
        finally:
            self._unregister_progress_callback(progress_cb, callbacks)

        img = result if isinstance(result, Image.Image) else result.image
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()

    async def _execute(
        self, params: BaseModel, kind: str, extra_kwargs: dict[str, Any]
    ) -> bytes:
        if self._model is None:
            raise RuntimeError("Engine not started. Call start() first.")

        gen_kwargs = {**params.model_dump(exclude_none=True), **extra_kwargs}
        logger.info(
            "Image %s: model=%s, prompt_len=%d, seed=%s",
            kind,
            self._model_name,
            len(gen_kwargs.get("prompt", "")),
            gen_kwargs.get("seed"),
        )

        t0 = time.monotonic()
        activity_id = self._begin_activity(
            f"{'generating' if kind == 'generation' else 'editing'} image",
            detail=f"Image {kind}",
        )
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                get_mlx_executor(), lambda: self._run_sync(activity_id, gen_kwargs)
            )
            logger.info(
                "Image %s done: model=%s, %.2fs, %d bytes output",
                kind,
                self._model_name,
                time.monotonic() - t0,
                len(result),
            )
            return result
        finally:
            await self._finish_activity(activity_id)

    async def generate(self, params: BaseModel) -> bytes:
        return await self._execute(params, "generation", {})

    async def edit(self, params: BaseModel, image_paths: list[str]) -> bytes:
        if self.edits_params is None:
            raise ValueError(f"Model {self._model_name} does not support image editing")
        return await self._execute(params, "edit", self.edit_kwargs(image_paths))

    def get_stats(self) -> dict[str, Any]:
        return {"model_name": self._model_name, "is_loaded": self._model is not None}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self._model_name})"
