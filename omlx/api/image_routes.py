# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible image generation and edit routes for oMLX."""

from __future__ import annotations

import asyncio
import base64
import binascii
import http.client
import ipaddress
import logging
import socket
import ssl
import tempfile
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, cast
from urllib.parse import ParseResult, unquote_to_bytes, urljoin, urlparse, urlunparse

from fastapi import APIRouter, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError
from starlette.datastructures import UploadFile as StarletteUploadFile

from omlx.utils.optional_deps import require_mlx_vlm_available

from .image_models import (
    ImageData,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageMultipartEditRequest,
    ImageReference,
    ImageResponse,
    ImageUsage,
)

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_IMAGE_INPUT_BYTES = 25 * 1024 * 1024
MAX_IMAGE_INPUT_PIXELS = 64 * 1024 * 1024
MAX_IMAGE_DOWNLOAD_BYTES = MAX_IMAGE_INPUT_BYTES
IMAGE_DOWNLOAD_TIMEOUT_SECONDS = 20.0
IMAGE_DOWNLOAD_CONNECT_TIMEOUT_SECONDS = 5.0
MAX_IMAGE_REDIRECTS = 3

_IMAGE_SUFFIX_BY_MIME = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
_SAFE_ENGINE_KWARG_FIELDS = {
    "negative_prompt",
    "scheduler",
    "image_strength",
}
_MULTIPART_FIELD_NAMES = {
    "prompt",
    "model",
    "n",
    "size",
    "quality",
    "output_format",
    "response_format",
    "background",
    "style",
    "moderation",
    "user",
    "stream",
    "partial_images",
    "seed",
    "steps",
    "guidance",
    "negative_prompt",
    "scheduler",
    "image_strength",
    "lora_paths",
    "lora_scales",
    "input_fidelity",
}
_MULTI_VALUE_FIELDS = {"lora_paths", "lora_scales"}


@dataclass(frozen=True)
class _ResolvedHTTPURL:
    url: str
    hostname: str
    port: int
    resolved_ip: str
    scheme: str


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(
        self,
        host: str,
        port: int,
        resolved_ip: str,
        connect_timeout: float,
        read_timeout: float,
    ) -> None:
        super().__init__(host=host, port=port, timeout=connect_timeout)
        self._resolved_ip = resolved_ip
        self._read_timeout = read_timeout

    def connect(self) -> None:
        self.sock = socket.create_connection(
            (self._resolved_ip, self.port),
            self.timeout,
            None,
        )
        self.sock.settimeout(self._read_timeout)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(
        self,
        host: str,
        port: int,
        resolved_ip: str,
        connect_timeout: float,
        read_timeout: float,
    ) -> None:
        super().__init__(
            host=host,
            port=port,
            timeout=connect_timeout,
            context=ssl.create_default_context(),
        )
        self._resolved_ip = resolved_ip
        self._read_timeout = read_timeout
        self._ssl_context = ssl.create_default_context()

    def connect(self) -> None:
        sock = socket.create_connection(
            (self._resolved_ip, self.port),
            self.timeout,
            None,
        )
        self.sock = self._ssl_context.wrap_socket(sock, server_hostname=self.host)
        self.sock.settimeout(self._read_timeout)


def _get_engine_pool():
    from omlx.server import _server_state

    pool = _server_state.engine_pool
    if pool is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    return pool


def _require_image_dependency() -> None:
    try:
        require_mlx_vlm_available()
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _resolve_model(model_id: str) -> str:
    from omlx.server import resolve_model_id

    return resolve_model_id(model_id) or model_id


def _validation_detail(exc: ValidationError) -> list[dict[str, Any]]:
    return [
        dict(error) for error in exc.errors(include_url=False, include_context=False)
    ]


def _normalize_task_names(tasks: object) -> set[str]:
    if tasks is None:
        return set()
    items: Iterable[object]
    if isinstance(tasks, str):
        items = [tasks]
    elif isinstance(tasks, Iterable):
        items = tasks
    else:
        return set()

    normalized: set[str] = set()
    for item in items:
        if not isinstance(item, str):
            continue
        task = item.strip().lower().replace("_", "-")
        if task in {"generation", "generate", "text-to-image", "txt2img"}:
            normalized.add("generation")
        elif task in {
            "edit",
            "editing",
            "image-to-image",
            "img2img",
            "inpaint",
            "inpainting",
        }:
            normalized.add("edit")
    return normalized


def _entry_for_model(pool: Any, model_id: str) -> Any | None:
    get_entry = getattr(pool, "get_entry", None)
    if get_entry is None:
        return None
    try:
        return get_entry(model_id)
    except Exception:
        return None


def _validate_image_entry(entry: Any | None, model_id: str, task: str) -> bool:
    if entry is None:
        return False

    model_type = getattr(entry, "model_type", None)
    engine_type = getattr(entry, "engine_type", None)
    entry_is_image = model_type == "image" or engine_type == "image"
    if not entry_is_image:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_id}' is not an image model",
        )

    tasks = _normalize_task_names(getattr(entry, "tasks", None))
    if tasks and task not in tasks:
        raise HTTPException(
            status_code=400,
            detail=f"Image model '{model_id}' does not support task '{task}'",
        )
    return True


def _engine_stats(engine: Any) -> dict[str, Any]:
    get_stats = getattr(engine, "get_stats", None)
    if get_stats is None:
        return {}
    try:
        stats = get_stats()
    except Exception:
        return {}
    return stats if isinstance(stats, dict) else {}


def _validate_image_engine(
    engine: Any,
    model_id: str,
    task: str,
    *,
    entry_is_image: bool,
) -> None:
    stats = _engine_stats(engine)
    stats_tasks = _normalize_task_names(stats.get("tasks"))
    if stats_tasks and task not in stats_tasks:
        raise HTTPException(
            status_code=400,
            detail=f"Image model '{model_id}' does not support task '{task}'",
        )

    method_name = "generate" if task == "generation" else "edit"
    has_task_method = callable(getattr(engine, method_name, None))
    looks_like_image = (
        entry_is_image or bool(stats_tasks) or stats.get("backend") == "mlx-vlm"
    )
    if not looks_like_image or not has_task_method:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_id}' is not an image-capable model",
        )


async def _load_image_engine(model: str, task: str) -> tuple[str, Any, Any]:
    from omlx.exceptions import (
        EnginePoolError,
        InsufficientMemoryError,
        ModelLoadingError,
        ModelNotFoundError,
        ModelTooLargeError,
    )

    pool = _get_engine_pool()
    resolved_model = _resolve_model(model)
    entry_is_image = _validate_image_entry(
        _entry_for_model(pool, resolved_model), resolved_model, task
    )

    try:
        engine = await pool.get_engine(resolved_model, _lease=True)
    except ModelNotFoundError as exc:
        available = (
            ", ".join(exc.available_models) if exc.available_models else "(none)"
        )
        raise HTTPException(
            status_code=404,
            detail=f"Model '{resolved_model}' not found. Available: {available}",
        ) from exc
    except (ModelTooLargeError, InsufficientMemoryError) as exc:
        raise HTTPException(status_code=507, detail=str(exc)) from exc
    except ModelLoadingError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except EnginePoolError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        _validate_image_engine(
            engine,
            resolved_model,
            task,
            entry_is_image=entry_is_image,
        )
    except Exception:
        await pool.release_engine(resolved_model)
        raise
    return resolved_model, engine, pool


def _safe_suffix(filename: str | None = None, content_type: str | None = None) -> str:
    if content_type:
        suffix = _IMAGE_SUFFIX_BY_MIME.get(content_type.split(";")[0].strip().lower())
        if suffix:
            return suffix
    if filename:
        suffix = Path(filename).suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            return suffix
    return ".png"


def _validate_image_bytes(data: bytes) -> None:
    try:
        with Image.open(BytesIO(data)) as image:
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_INPUT_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        "Image dimensions exceed maximum allowed pixels "
                        f"({MAX_IMAGE_INPUT_PIXELS})"
                    ),
                )
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=400, detail="Input is not a valid image"
        ) from exc


def _write_image_bytes(data: bytes, suffix: str, temp_dir: Path) -> str:
    if len(data) > MAX_IMAGE_INPUT_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image input exceeds maximum allowed size ({MAX_IMAGE_INPUT_BYTES} bytes)",
        )
    _validate_image_bytes(data)

    path = temp_dir / f"{uuid.uuid4().hex}{suffix}"
    with path.open("xb") as handle:
        handle.write(data)
    return str(path)


async def _read_upload(file: UploadFile | StarletteUploadFile) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_IMAGE_INPUT_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    "Image upload exceeds maximum allowed size "
                    f"({MAX_IMAGE_INPUT_BYTES} bytes)"
                ),
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _decode_data_uri(uri: str) -> tuple[bytes, str]:
    header, separator, payload = uri.partition(",")
    if not separator or not header.lower().startswith("data:"):
        raise HTTPException(status_code=400, detail="Invalid image data URI")

    meta = header[5:]
    media_type = meta.split(";", 1)[0].lower() if meta else "image/png"
    if media_type and not media_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Data URI must contain an image")

    try:
        if ";base64" in meta.lower():
            if len(payload) > int(MAX_IMAGE_INPUT_BYTES * 1.5):
                raise HTTPException(
                    status_code=413,
                    detail=(
                        "Image data URI exceeds maximum allowed size "
                        f"({MAX_IMAGE_INPUT_BYTES} bytes)"
                    ),
                )
            data = base64.b64decode(payload, validate=True)
        else:
            data = unquote_to_bytes(payload)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid image data URI") from exc

    return data, _safe_suffix(content_type=media_type)


def _is_public_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    mapped = getattr(ip, "ipv4_mapped", None)
    return mapped.is_global if mapped is not None else ip.is_global


def _validate_public_http_url(url: str) -> _ResolvedHTTPURL:
    """Reject internal/private addresses before server-side image fetches."""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid image_url") from exc
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(
            status_code=400,
            detail="image_url must be a data URI or HTTP/HTTPS URL",
        )
    if not hostname:
        raise HTTPException(status_code=400, detail="image_url host is required")

    try:
        addr_infos = socket.getaddrinfo(
            hostname,
            port,
            type=socket.SOCK_STREAM,
        )
    except OSError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to resolve image_url host: {hostname}",
        ) from exc

    try:
        resolved_ips = [
            ipaddress.ip_address(str(sockaddr[0]).split("%", 1)[0])
            for *_, sockaddr in addr_infos
        ]
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to resolve image_url host: {hostname}",
        ) from exc
    if not resolved_ips or any(not _is_public_ip(ip) for ip in resolved_ips):
        raise HTTPException(
            status_code=400,
            detail="image_url must resolve to a public internet address",
        )
    return _ResolvedHTTPURL(
        url=url,
        hostname=hostname,
        port=port,
        resolved_ip=str(resolved_ips[0]),
        scheme=parsed.scheme,
    )


def _host_header(parsed: ParseResult) -> str:
    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    return hostname if parsed.port is None else f"{hostname}:{parsed.port}"


def _request_target(parsed: ParseResult) -> str:
    return urlunparse(("", "", parsed.path or "/", parsed.params, parsed.query, ""))


def _get_pinned_response(
    resolved: _ResolvedHTTPURL,
) -> tuple[http.client.HTTPResponse, http.client.HTTPConnection]:
    parsed = urlparse(resolved.url)
    if resolved.scheme == "https":
        connection: http.client.HTTPConnection = _PinnedHTTPSConnection(
            resolved.hostname,
            resolved.port,
            resolved.resolved_ip,
            IMAGE_DOWNLOAD_CONNECT_TIMEOUT_SECONDS,
            IMAGE_DOWNLOAD_TIMEOUT_SECONDS,
        )
    else:
        connection = _PinnedHTTPConnection(
            resolved.hostname,
            resolved.port,
            resolved.resolved_ip,
            IMAGE_DOWNLOAD_CONNECT_TIMEOUT_SECONDS,
            IMAGE_DOWNLOAD_TIMEOUT_SECONDS,
        )
    try:
        connection.putrequest("GET", _request_target(parsed), skip_host=True)
        connection.putheader("Host", _host_header(parsed))
        connection.putheader("Accept", "image/*,*/*;q=0.8")
        connection.putheader("User-Agent", "omlx-image-fetch/1.0")
        connection.endheaders()
        return connection.getresponse(), connection
    except Exception:
        connection.close()
        raise


def _download_http_image(url: str) -> tuple[bytes, str]:
    current_url = url
    try:
        for _ in range(MAX_IMAGE_REDIRECTS + 1):
            resolved = _validate_public_http_url(current_url)
            response, connection = _get_pinned_response(resolved)
            if 300 <= response.status < 400:
                location = response.getheader("location")
                connection.close()
                if not location:
                    raise HTTPException(
                        status_code=400,
                        detail="Image download redirect missing Location header",
                    )
                next_url = urljoin(current_url, location)
                if (
                    urlparse(current_url).scheme == "https"
                    and urlparse(next_url).scheme != "https"
                ):
                    raise HTTPException(
                        status_code=400,
                        detail="Image download redirects from HTTPS to HTTP are not allowed",
                    )
                current_url = next_url
                continue
            break
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Image download exceeded {MAX_IMAGE_REDIRECTS} redirects",
            )

        try:
            if response.status >= 400:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to download image: HTTP {response.status}",
                )
            content_length = response.getheader("content-length")
            if content_length is not None:
                try:
                    length = int(content_length)
                except ValueError:
                    length = 0
                if length > MAX_IMAGE_DOWNLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Image download exceeds maximum allowed size "
                            f"({MAX_IMAGE_DOWNLOAD_BYTES} bytes)"
                        ),
                    )

            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_IMAGE_DOWNLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Image download exceeds maximum allowed size "
                            f"({MAX_IMAGE_DOWNLOAD_BYTES} bytes)"
                        ),
                    )
                chunks.append(chunk)
            return (
                b"".join(chunks),
                _safe_suffix(
                    filename=urlparse(current_url).path,
                    content_type=response.getheader("content-type"),
                ),
            )
        finally:
            connection.close()
    except TimeoutError as exc:
        raise HTTPException(
            status_code=504, detail="Timed out downloading image"
        ) from exc
    except (OSError, http.client.HTTPException, ssl.SSLError) as exc:
        raise HTTPException(
            status_code=400, detail=f"Failed to download image: {exc}"
        ) from exc


async def _image_reference_to_path(reference: ImageReference, temp_dir: Path) -> str:
    if reference.file_id is not None:
        raise HTTPException(
            status_code=400,
            detail="file_id image references are not supported; no File API store exists",
        )
    image_url = reference.image_url
    if image_url is None:
        raise HTTPException(status_code=400, detail="image_url is required")
    url = image_url if isinstance(image_url, str) else image_url.url
    parsed = urlparse(url)

    if parsed.scheme == "data":
        data, suffix = _decode_data_uri(url)
    elif parsed.scheme in {"http", "https"}:
        data, suffix = await asyncio.to_thread(_download_http_image, url)
    else:
        raise HTTPException(
            status_code=400,
            detail="image_url must be a data URI or HTTP/HTTPS URL",
        )
    return _write_image_bytes(data, suffix, temp_dir)


def _reject_unsupported_lora(
    request: ImageGenerationRequest | ImageEditRequest | ImageMultipartEditRequest,
) -> None:
    if request.lora_paths is not None or request.lora_scales is not None:
        raise HTTPException(
            status_code=400,
            detail="Request-level LoRA fields are not supported for image endpoints yet",
        )


def _reject_unsupported_streaming(
    request: ImageGenerationRequest | ImageEditRequest | ImageMultipartEditRequest,
) -> None:
    if request.stream:
        raise HTTPException(
            status_code=400,
            detail="stream is not supported for image endpoints",
        )
    if request.partial_images is not None:
        raise HTTPException(
            status_code=400,
            detail="partial_images is not supported for image endpoints",
        )


def _validate_image_size(
    request: ImageGenerationRequest | ImageEditRequest | ImageMultipartEditRequest,
) -> None:
    try:
        request.parsed_size()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _reject_unsupported_response_format(
    request: ImageGenerationRequest | ImageEditRequest | ImageMultipartEditRequest,
) -> None:
    if request.response_format == "url":
        raise HTTPException(
            status_code=400,
            detail="response_format='url' is not supported; use 'b64_json'",
        )


def _request_engine_kwargs(
    request: ImageGenerationRequest | ImageEditRequest | ImageMultipartEditRequest,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for field in _SAFE_ENGINE_KWARG_FIELDS:
        value = getattr(request, field, None)
        if value is not None:
            kwargs[field] = value
    return kwargs


def _seed_for_output(seed: int | None, index: int) -> int | None:
    if seed is not None:
        return int(seed) + index
    return None


def _encode_image_b64(image: Any, output_format: str) -> str:
    if not hasattr(image, "save"):
        raise HTTPException(
            status_code=500, detail="Image engine returned no PIL image"
        )

    pil_image = image
    image_format = output_format.upper()
    if image_format == "JPEG":
        if getattr(pil_image, "mode", None) in {"RGBA", "LA", "P"}:
            pil_image = pil_image.convert("RGB")
    elif image_format == "JPG":
        image_format = "JPEG"

    buffer = BytesIO()
    try:
        pil_image.save(buffer, format=image_format)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to encode image as {output_format}: {exc}",
        ) from exc
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _image_data_from_result(
    result: Any,
    request: ImageGenerationRequest | ImageEditRequest | ImageMultipartEditRequest,
) -> ImageData:
    image = getattr(result, "image", result)
    metadata = getattr(result, "metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    return ImageData(
        b64_json=_encode_image_b64(image, request.output_format),
        revised_prompt=metadata.get("revised_prompt"),
        size=(
            f"{image.width}x{image.height}"
            if hasattr(image, "width") and hasattr(image, "height")
            else request.size
        ),
        quality=request.quality,
        output_format=request.output_format,
        background=getattr(request, "background", None),
        steps=metadata.get("steps"),
        guidance=metadata.get("guidance"),
    )


async def _run_generation(
    resolved_model: str,
    engine: Any,
    request: ImageGenerationRequest,
) -> ImageResponse:
    width, height = request.parsed_size()
    data: list[ImageData] = []
    # ImageEngine serializes mlx-vlm/MLX calls for GPU safety, so keep n > 1 sequential.
    for index in range(request.n):
        result = await engine.generate(
            request.prompt,
            width=width,
            height=height,
            seed=_seed_for_output(request.seed, index),
            steps=request.steps,
            guidance=request.guidance,
            output_format=request.output_format,
            **_request_engine_kwargs(request),
        )
        data.append(_image_data_from_result(result, request))

    return ImageResponse(
        data=data,
        model=resolved_model,
        usage=ImageUsage(image_count=len(data)),
    )


async def _run_edit(
    resolved_model: str,
    engine: Any,
    request: ImageEditRequest | ImageMultipartEditRequest,
    image_paths: list[str],
    mask_path: str | None,
) -> ImageResponse:
    width, height = request.parsed_size()
    data: list[ImageData] = []
    # ImageEngine serializes mlx-vlm/MLX calls for GPU safety, so keep n > 1 sequential.
    for index in range(request.n):
        result = await engine.edit(
            request.prompt,
            image_paths=image_paths,
            mask_path=mask_path,
            width=width,
            height=height,
            seed=_seed_for_output(request.seed, index),
            steps=request.steps,
            guidance=request.guidance,
            output_format=request.output_format,
            **_request_engine_kwargs(request),
        )
        data.append(_image_data_from_result(result, request))

    return ImageResponse(
        data=data,
        model=resolved_model,
        usage=ImageUsage(image_count=len(data)),
    )


def _is_upload(value: object) -> bool:
    return isinstance(value, (UploadFile, StarletteUploadFile)) or (
        hasattr(value, "read") and hasattr(value, "filename")
    )


def _form_values(form: Any, name: str) -> list[Any]:
    getlist = getattr(form, "getlist", None)
    if getlist is None:
        value = form.get(name)
        return [] if value is None else [value]
    return list(getlist(name))


def _parse_multi_value(values: list[str]) -> list[str]:
    parsed: list[str] = []
    for value in values:
        parsed.extend(part.strip() for part in value.split(",") if part.strip())
    return parsed


def _multipart_request_from_form(form: Any) -> ImageMultipartEditRequest:
    payload: dict[str, Any] = {}
    for field in _MULTIPART_FIELD_NAMES:
        values = [
            str(value)
            for value in _form_values(form, field)
            if not _is_upload(value) and str(value) != ""
        ]
        if not values:
            continue
        if field in _MULTI_VALUE_FIELDS:
            payload[field] = _parse_multi_value(values)
        else:
            payload[field] = values[-1]
    try:
        return cast(
            ImageMultipartEditRequest,
            ImageMultipartEditRequest.model_validate(payload),
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=_validation_detail(exc)) from exc


async def _upload_to_path(value: object, field_name: str, temp_dir: Path) -> str:
    if not _is_upload(value):
        raise HTTPException(
            status_code=400, detail=f"'{field_name}' must be a file upload"
        )
    upload = cast(UploadFile | StarletteUploadFile, value)
    data = await _read_upload(upload)
    return _write_image_bytes(
        data,
        _safe_suffix(
            filename=getattr(upload, "filename", None),
            content_type=getattr(upload, "content_type", None),
        ),
        temp_dir,
    )


@router.post("/v1/images/generations", response_model=ImageResponse)
async def create_image_generation(request: ImageGenerationRequest) -> ImageResponse:
    """Create images from a prompt using a discovered ImageEngine model."""
    _reject_unsupported_lora(request)
    _reject_unsupported_streaming(request)
    _reject_unsupported_response_format(request)
    _validate_image_size(request)
    _require_image_dependency()
    resolved_model, engine, pool = await _load_image_engine(request.model, "generation")
    try:
        return await _run_generation(resolved_model, engine, request)
    except HTTPException:
        raise
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        await pool.release_engine(resolved_model)


@router.post("/v1/images/edits", response_model=ImageResponse)
async def create_image_edit(request: Request) -> ImageResponse:
    """Create edited images from JSON image refs or multipart uploads."""
    content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
    if content_type == "application/json" or content_type.endswith("+json"):
        return await _create_json_image_edit(request)
    if content_type == "multipart/form-data":
        return await _create_multipart_image_edit(request)
    raise HTTPException(
        status_code=415,
        detail="Content-Type must be application/json or multipart/form-data",
    )


async def _create_json_image_edit(request: Request) -> ImageResponse:
    try:
        payload = await request.json()
        edit_request = ImageEditRequest.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=_validation_detail(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from exc
    _reject_unsupported_lora(edit_request)
    _reject_unsupported_streaming(edit_request)
    _reject_unsupported_response_format(edit_request)
    if edit_request.mask is not None:
        raise HTTPException(
            status_code=400,
            detail="mask is not supported by mlx-vlm image models",
        )
    _validate_image_size(edit_request)
    _require_image_dependency()

    with tempfile.TemporaryDirectory(prefix="omlx-image-") as temp_dir:
        image_dir = Path(temp_dir)
        image_paths: list[str] = []
        try:
            for reference in edit_request.images:
                image_paths.append(await _image_reference_to_path(reference, image_dir))
            resolved_model, engine, pool = await _load_image_engine(
                edit_request.model, "edit"
            )
            try:
                return await _run_edit(
                    resolved_model,
                    engine,
                    edit_request,
                    image_paths=image_paths,
                    mask_path=None,
                )
            finally:
                await pool.release_engine(resolved_model)
        except HTTPException:
            raise
        except ImportError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc


async def _create_multipart_image_edit(request: Request) -> ImageResponse:
    form = await request.form()
    edit_request = _multipart_request_from_form(form)
    _reject_unsupported_lora(edit_request)
    _reject_unsupported_streaming(edit_request)
    _reject_unsupported_response_format(edit_request)
    uploads = _form_values(form, "image") + _form_values(form, "image[]")
    uploads = [value for value in uploads if _is_upload(value)]
    if not uploads:
        raise HTTPException(
            status_code=400, detail="At least one 'image' upload is required"
        )

    mask_values = [value for value in _form_values(form, "mask") if _is_upload(value)]
    if len(mask_values) > 1:
        raise HTTPException(
            status_code=400, detail="Only one 'mask' upload is supported"
        )
    if mask_values:
        raise HTTPException(
            status_code=400,
            detail="mask is not supported by mlx-vlm image models",
        )
    _validate_image_size(edit_request)
    _require_image_dependency()

    with tempfile.TemporaryDirectory(prefix="omlx-image-") as temp_dir:
        image_dir = Path(temp_dir)
        image_paths: list[str] = []
        try:
            for upload in uploads:
                image_paths.append(await _upload_to_path(upload, "image", image_dir))
            resolved_model, engine, pool = await _load_image_engine(
                edit_request.model, "edit"
            )
            try:
                return await _run_edit(
                    resolved_model,
                    engine,
                    edit_request,
                    image_paths=image_paths,
                    mask_path=None,
                )
            finally:
                await pool.release_engine(resolved_model)
        except HTTPException:
            raise
        except ImportError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
