# SPDX-License-Identifier: Apache-2.0
"""PDF OCR processing for the MarkItDown integration."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from typing import Any

from ..exceptions import (
    EnginePoolError,
    InsufficientMemoryError,
    ModelLoadingError,
    ModelNotFoundError,
    ModelTooLargeError,
)
from .markitdown import MarkItDownFile, MarkItDownRequestError, quiet_pdf_parser_loggers

logger = logging.getLogger(__name__)


def resolve_pdf_ocr_model(
    model_id: str,
    *,
    engine_pool: Any | None,
    settings_manager: Any | None,
) -> str:
    if engine_pool is None:
        raise MarkItDownRequestError(
            "PDF OCR processing requires an initialized engine pool.",
            status_code=503,
        )

    resolved = engine_pool.resolve_model_id(model_id, settings_manager)
    entry = engine_pool.get_entry(resolved)
    if entry is None:
        raise MarkItDownRequestError(
            f"MarkItDown PDF OCR model not found: {model_id}",
            status_code=400,
        )

    config_model_type = str(getattr(entry, "config_model_type", "") or "").lower()
    if "ocr" not in config_model_type:
        raise MarkItDownRequestError(
            "MarkItDown PDF OCR model must have OCR in config model_type: "
            f"{model_id}",
            status_code=400,
        )

    if getattr(entry, "engine_type", "") != "vlm":
        raise MarkItDownRequestError(
            f"MarkItDown PDF OCR model must be a VLM: {model_id}",
            status_code=400,
        )

    return resolved


def render_pdf_pages_to_image_data_uris(
    file: MarkItDownFile,
    *,
    resolution: int = 144,
) -> list[str]:
    quiet_pdf_parser_loggers()
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError(
            "pdfplumber is not installed. Install markitdown[pdf]."
        ) from exc

    data_uris: list[str] = []
    with pdfplumber.open(io.BytesIO(file.data)) as pdf:
        for page in pdf.pages:
            try:
                image = page.to_image(
                    resolution=resolution,
                    antialias=True,
                ).original
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                data_uris.append(f"data:image/png;base64,{encoded}")
            finally:
                page.close()
    return data_uris


async def convert_pdf_with_ocr_engine(
    file: MarkItDownFile,
    *,
    engine_model_id: str,
    engine_pool: Any | None,
    settings_manager: Any | None,
    global_settings: Any | None,
    get_sampling_params: Any | None,
) -> str:
    chunks: list[str] = []
    async for chunk in stream_pdf_with_ocr_engine(
        file,
        engine_model_id=engine_model_id,
        engine_pool=engine_pool,
        settings_manager=settings_manager,
        global_settings=global_settings,
        get_sampling_params=get_sampling_params,
    ):
        chunks.append(chunk)
    return "".join(chunks).strip()


async def stream_pdf_with_ocr_engine(
    file: MarkItDownFile,
    *,
    engine_model_id: str,
    engine_pool: Any | None,
    settings_manager: Any | None,
    global_settings: Any | None,
    get_sampling_params: Any | None,
):
    model_id = resolve_pdf_ocr_model(
        engine_model_id,
        engine_pool=engine_pool,
        settings_manager=settings_manager,
    )
    data_uris = await asyncio.to_thread(render_pdf_pages_to_image_data_uris, file)
    if not data_uris:
        raise MarkItDownRequestError(
            f"No pages found in attached PDF: {file.filename}",
            status_code=400,
        )

    logger.info(
        "Using OCR PDF processing engine: filename=%s model=%s pages=%d",
        file.filename,
        model_id,
        len(data_uris),
    )

    if get_sampling_params is None:
        raise RuntimeError("get_sampling_params callback is required for OCR PDF.")

    (
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        min_p,
        presence_penalty,
        frequency_penalty,
        max_tokens,
        xtc_probability,
        xtc_threshold,
    ) = get_sampling_params(None, None, model_id)
    chat_kwargs = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "repetition_penalty": repetition_penalty,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "xtc_probability": xtc_probability,
        "xtc_threshold": xtc_threshold,
    }

    emitted = False
    try:
        async with engine_pool.acquire(model_id) as engine:
            async for page_number, text in _stream_pages_with_ocr(
                engine,
                data_uris,
                global_settings=global_settings,
                chat_kwargs=chat_kwargs,
            ):
                text = text.strip()
                if not text:
                    continue
                emitted = True
                yield f"### Page {page_number}\n\n{text}\n\n"
    except ModelNotFoundError as exc:
        raise MarkItDownRequestError(str(exc), status_code=404) from exc
    except ModelTooLargeError as exc:
        raise MarkItDownRequestError(str(exc), status_code=507) from exc
    except InsufficientMemoryError as exc:
        raise MarkItDownRequestError(str(exc), status_code=507) from exc
    except ModelLoadingError as exc:
        raise MarkItDownRequestError(str(exc), status_code=409) from exc
    except EnginePoolError as exc:
        raise RuntimeError(str(exc)) from exc
    finally:
        unload = getattr(engine_pool, "unload_if_idle_unpinned", None)
        if callable(unload):
            await unload(model_id)

    if not emitted:
        raise MarkItDownRequestError(
            f"OCR PDF processing produced no text for attached PDF: {file.filename}",
            status_code=400,
        )


async def _stream_pages_with_ocr(
    engine: Any,
    data_uris: list[str],
    *,
    global_settings: Any | None,
    chat_kwargs: dict[str, Any],
):
    scheduler = getattr(global_settings, "scheduler", None)
    max_concurrent = int(getattr(scheduler, "max_concurrent_requests", 1) or 1)
    semaphore = asyncio.Semaphore(max(1, max_concurrent))
    queue: asyncio.Queue[tuple[int, str, Exception | None]] = asyncio.Queue()

    async def convert_page(index: int, data_uri: str) -> None:
        try:
            async with semaphore:
                output = await engine.chat(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": data_uri},
                                }
                            ],
                        }
                    ],
                    **dict(chat_kwargs),
                )
                await queue.put((index, (output.text or "").strip(), None))
        except Exception as exc:
            await queue.put((index, "", exc))

    tasks = [
        asyncio.create_task(convert_page(idx, data_uri))
        for idx, data_uri in enumerate(data_uris, 1)
    ]
    pending = len(tasks)
    buffered: dict[int, str] = {}
    next_to_emit = 1

    try:
        while pending:
            page_number, text, error = await queue.get()
            pending -= 1
            if error is not None:
                for task in tasks:
                    task.cancel()
                raise error

            buffered[page_number] = text
            while next_to_emit in buffered:
                yield next_to_emit, buffered.pop(next_to_emit)
                next_to_emit += 1
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
