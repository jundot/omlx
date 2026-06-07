# SPDX-License-Identifier: Apache-2.0
"""MarkItDown integration helpers."""

from __future__ import annotations

import base64
import binascii
import io
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .openai_models import ContentPart, Message

logger = logging.getLogger(__name__)
NOISY_PDF_LOGGERS = (
    "pdfminer",
    "pdfminer.cmapdb",
    "pdfminer.converter",
    "pdfminer.layout",
    "pdfminer.pdfdocument",
    "pdfminer.pdfinterp",
    "pdfminer.pdfpage",
    "pdfminer.pdfparser",
    "pdfminer.psparser",
)

MARKITDOWN_MODEL_ID = "MarkItDown"
MARKITDOWN_MODEL_ALIASES = {"markitdown"}
MARKITDOWN_SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx"}
PLAIN_TEXT_SUPPORTED_EXTENSIONS = {".txt", ".md"}
SUPPORTED_ATTACHMENT_EXTENSIONS = (
    MARKITDOWN_SUPPORTED_EXTENSIONS | PLAIN_TEXT_SUPPORTED_EXTENSIONS
)
MARKITDOWN_UNSUPPORTED_SPREADSHEET_EXTENSIONS = {".xls", ".xlsx"}
DOCX_MIME_TYPE = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)
PPTX_MIME_TYPE = (
    "application/vnd.openxmlformats-officedocument.presentationml.presentation"
)
MARKITDOWN_SUPPORTED_MIME_TYPES = {
    "application/pdf",
    DOCX_MIME_TYPE,
    PPTX_MIME_TYPE,
}
PLAIN_TEXT_SUPPORTED_MIME_TYPES = {"text/plain", "text/markdown", "text/x-markdown"}
SUPPORTED_ATTACHMENT_MIME_TYPES = (
    MARKITDOWN_SUPPORTED_MIME_TYPES | PLAIN_TEXT_SUPPORTED_MIME_TYPES
)
MARKITDOWN_SPREADSHEET_MIME_TYPES = {
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}
MARKITDOWN_EMPTY_PDF_MESSAGE = (
    "No extractable text found. This PDF appears to be scanned or image-only. "
    "Select an OCR PDF processing engine for scanned or image-only PDFs."
)
MARKITDOWN_EMPTY_MESSAGE = "No extractable text found in the attached document."
PDF_PROCESSING_MARKITDOWN = "markitdown"
DEFAULT_PDF_PROCESSING_ENGINE = PDF_PROCESSING_MARKITDOWN

_converter_lock = threading.Lock()
_converter: Any | None = None


@dataclass(frozen=True)
class MarkItDownFile:
    filename: str
    mime_type: str
    data: bytes

    @property
    def extension(self) -> str:
        return Path(self.filename).suffix.lower()


class MarkItDownRequestError(ValueError):
    """Client-side request error for document conversion."""

    def __init__(self, detail: str, *, status_code: int = 400) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


def is_markitdown_model(model: str | None) -> bool:
    normalized = (model or "").strip()
    return normalized == MARKITDOWN_MODEL_ID or normalized in MARKITDOWN_MODEL_ALIASES


def markitdown_enabled(global_settings: Any | None) -> bool:
    if global_settings is None:
        return True
    integrations = getattr(global_settings, "integrations", None)
    if integrations is None:
        return True
    return bool(getattr(integrations, "markitdown_enabled", True))


def markitdown_model_visible(global_settings: Any | None) -> bool:
    if not markitdown_enabled(global_settings):
        return False
    integrations = getattr(global_settings, "integrations", None)
    if integrations is None:
        return True
    return bool(getattr(integrations, "markitdown_expose_model", True))


def markitdown_limits(global_settings: Any | None) -> tuple[int, int]:
    integrations = getattr(global_settings, "integrations", None)
    max_mb = getattr(integrations, "markitdown_max_file_size_mb", 25)
    max_files = getattr(integrations, "markitdown_max_files_per_request", 5)
    return int(max_mb or 25), int(max_files or 5)


def quiet_pdf_parser_loggers() -> None:
    for logger_name in NOISY_PDF_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def markitdown_pdf_processing_engine(global_settings: Any | None) -> str:
    integrations = getattr(global_settings, "integrations", None)
    value = getattr(
        integrations,
        "markitdown_pdf_processing_engine",
        DEFAULT_PDF_PROCESSING_ENGINE,
    )
    return normalize_pdf_processing_engine(value)


def normalize_pdf_processing_engine(value: Any) -> str:
    engine = str(value or DEFAULT_PDF_PROCESSING_ENGINE).strip()
    if not engine:
        return DEFAULT_PDF_PROCESSING_ENGINE

    if engine.lower() == PDF_PROCESSING_MARKITDOWN:
        return PDF_PROCESSING_MARKITDOWN
    return engine


def request_has_file_parts(messages: list[Message]) -> bool:
    return any(_iter_file_part_dicts(msg.content) for msg in messages)


def convert_messages_to_markdown(
    messages: list[Message],
    *,
    global_settings: Any | None = None,
    latest_user_only: bool = False,
) -> str:
    """Render request messages into Markdown, converting file parts."""
    if latest_user_only:
        messages = _latest_user_turn(messages)

    converted_messages = preprocess_markitdown_file_parts(
        messages,
        global_settings=global_settings,
        fail_when_disabled=True,
    )
    sections: list[str] = []
    for msg in converted_messages:
        text = _content_text(msg.content).strip()
        if text:
            sections.append(text)
    return "\n\n".join(sections).strip()


async def convert_messages_to_markdown_async(
    messages: list[Message],
    *,
    global_settings: Any | None = None,
    engine_pool: Any | None = None,
    settings_manager: Any | None = None,
    get_sampling_params: Any | None = None,
    latest_user_only: bool = False,
) -> str:
    """Render request messages into Markdown, converting file parts asynchronously."""
    if latest_user_only:
        messages = _latest_user_turn(messages)

    converted_messages = await preprocess_markitdown_file_parts_async(
        messages,
        global_settings=global_settings,
        engine_pool=engine_pool,
        settings_manager=settings_manager,
        get_sampling_params=get_sampling_params,
        fail_when_disabled=True,
    )
    sections: list[str] = []
    for msg in converted_messages:
        text = _content_text(msg.content).strip()
        if text:
            sections.append(text)
    return "\n\n".join(sections).strip()


async def stream_messages_to_markdown_async(
    messages: list[Message],
    *,
    global_settings: Any | None = None,
    engine_pool: Any | None = None,
    settings_manager: Any | None = None,
    get_sampling_params: Any | None = None,
    latest_user_only: bool = False,
):
    """Stream request messages into Markdown chunks, preserving OCR PDF page order."""
    if latest_user_only:
        messages = _latest_user_turn(messages)

    if request_has_file_parts(messages) and not markitdown_enabled(global_settings):
        raise MarkItDownRequestError(
            "MarkItDown integration is disabled.",
            status_code=400,
        )

    max_file_size_mb, max_files = markitdown_limits(global_settings)
    files_seen = 0
    emitted = False

    async def emit_chunk_stream(chunks):
        nonlocal emitted
        started = False
        async for chunk in chunks:
            if not chunk:
                continue
            if not started:
                if emitted:
                    yield "\n\n"
                emitted = True
                started = True
            yield chunk

    for msg in messages:
        content = msg.content
        if isinstance(content, str):
            async for chunk in emit_chunk_stream(_single_chunk(content.strip())):
                yield chunk
            continue
        if not isinstance(content, list):
            async for chunk in emit_chunk_stream(_single_chunk(str(content))):
                yield chunk
            continue

        for part in content:
            part_dict = _part_to_dict(part)
            if part_dict.get("type") in {"text", "input_text"}:
                async for chunk in emit_chunk_stream(
                    _single_chunk(str(part_dict.get("text") or "").strip())
                ):
                    yield chunk
                continue

            if part_dict.get("type") != "file":
                continue

            files_seen += 1
            if files_seen > max_files:
                raise MarkItDownRequestError(
                    f"Too many attached files. Maximum is {max_files} per request.",
                    status_code=400,
                )

            parsed = parse_file_part(part_dict, max_file_size_mb=max_file_size_mb)
            async for chunk in emit_chunk_stream(
                stream_attachment_to_markdown_async(
                    parsed,
                    global_settings=global_settings,
                    engine_pool=engine_pool,
                    settings_manager=settings_manager,
                    get_sampling_params=get_sampling_params,
                )
            ):
                yield chunk


async def _single_chunk(text: str):
    if text:
        yield text


def _latest_user_turn(messages: list[Message]) -> list[Message]:
    for msg in reversed(messages):
        if msg.role == "user":
            return [msg]
    return messages[-1:] if messages else []


def preprocess_markitdown_file_parts(
    messages: list[Message],
    *,
    global_settings: Any | None = None,
    fail_when_disabled: bool = True,
) -> list[Message]:
    """Replace file content parts with Markdown text parts."""
    if not request_has_file_parts(messages):
        return messages

    if not markitdown_enabled(global_settings):
        if fail_when_disabled:
            raise MarkItDownRequestError(
                "MarkItDown integration is disabled.",
                status_code=400,
            )
        return messages

    max_file_size_mb, max_files = markitdown_limits(global_settings)
    files_seen = 0
    processed: list[Message] = []

    for msg in messages:
        content = msg.content
        if not isinstance(content, list):
            processed.append(msg)
            continue

        new_parts: list[ContentPart] = []
        changed = False
        for part in content:
            part_dict = _part_to_dict(part)
            if part_dict.get("type") != "file":
                new_parts.append(ContentPart.model_validate(part_dict))
                continue

            files_seen += 1
            if files_seen > max_files:
                raise MarkItDownRequestError(
                    f"Too many attached files. Maximum is {max_files} per request.",
                    status_code=400,
                )

            parsed = parse_file_part(part_dict, max_file_size_mb=max_file_size_mb)
            markdown = convert_attachment_to_markdown(
                parsed,
                global_settings=global_settings,
            )
            new_parts.append(
                ContentPart(
                    type="text",
                    text=_format_attachment_markdown(parsed.filename, markdown),
                )
            )
            changed = True

        processed.append(
            msg.model_copy(update={"content": new_parts}) if changed else msg
        )

    return processed


async def preprocess_markitdown_file_parts_async(
    messages: list[Message],
    *,
    global_settings: Any | None = None,
    engine_pool: Any | None = None,
    settings_manager: Any | None = None,
    get_sampling_params: Any | None = None,
    fail_when_disabled: bool = True,
) -> list[Message]:
    """Replace file content parts with Markdown text parts asynchronously."""
    if not request_has_file_parts(messages):
        return messages

    if not markitdown_enabled(global_settings):
        if fail_when_disabled:
            raise MarkItDownRequestError(
                "MarkItDown integration is disabled.",
                status_code=400,
            )
        return messages

    max_file_size_mb, max_files = markitdown_limits(global_settings)
    files_seen = 0
    processed: list[Message] = []

    for msg in messages:
        content = msg.content
        if not isinstance(content, list):
            processed.append(msg)
            continue

        new_parts: list[ContentPart] = []
        changed = False
        for part in content:
            part_dict = _part_to_dict(part)
            if part_dict.get("type") != "file":
                new_parts.append(ContentPart.model_validate(part_dict))
                continue

            files_seen += 1
            if files_seen > max_files:
                raise MarkItDownRequestError(
                    f"Too many attached files. Maximum is {max_files} per request.",
                    status_code=400,
                )

            parsed = parse_file_part(part_dict, max_file_size_mb=max_file_size_mb)
            markdown = await convert_attachment_to_markdown_async(
                parsed,
                global_settings=global_settings,
                engine_pool=engine_pool,
                settings_manager=settings_manager,
                get_sampling_params=get_sampling_params,
            )
            new_parts.append(
                ContentPart(
                    type="text",
                    text=_format_attachment_markdown(parsed.filename, markdown),
                )
            )
            changed = True

        processed.append(
            msg.model_copy(update={"content": new_parts}) if changed else msg
        )

    return processed


def parse_file_part(part: dict[str, Any], *, max_file_size_mb: int) -> MarkItDownFile:
    file_obj = part.get("file")
    if not isinstance(file_obj, dict):
        raise MarkItDownRequestError("File content part must include a file object.")

    data_value = file_obj.get("file_data") or file_obj.get("data")
    if not data_value and file_obj.get("file_id"):
        raise MarkItDownRequestError(
            "File content part file_id is not supported. "
            "Send base64 content in file.file_data instead.",
            status_code=400,
        )
    if not isinstance(data_value, str) or not data_value.strip():
        raise MarkItDownRequestError(
            "File content part requires file.file_data.",
            status_code=400,
        )

    data_uri_mime_type = _mime_type_from_data_uri(data_value)
    mime_type = (
        str(file_obj.get("mime_type") or data_uri_mime_type or "").strip().lower()
    )

    filename = str(file_obj.get("filename") or "").strip()
    if not filename:
        extension = _extension_for_mime_type(mime_type)
        if not extension:
            raise MarkItDownRequestError(
                "File content part requires file.filename.",
                status_code=400,
            )
        filename = f"attachment{extension}"

    extension = Path(filename).suffix.lower()
    _validate_supported_file(filename, mime_type)

    data = _decode_data(data_value)
    max_bytes = max_file_size_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise MarkItDownRequestError(
            f"Attached file exceeds the {max_file_size_mb}MB limit: {filename}",
            status_code=400,
        )

    if not mime_type:
        mime_type = _mime_type_for_extension(extension)

    return MarkItDownFile(filename=filename, mime_type=mime_type, data=data)


def convert_attachment_to_markdown(
    file: MarkItDownFile,
    *,
    global_settings: Any | None = None,
) -> str:
    if _is_plain_text_attachment(file):
        return decode_plain_text_attachment(file)
    return convert_file_to_markdown(file, global_settings=global_settings)


async def convert_attachment_to_markdown_async(
    file: MarkItDownFile,
    *,
    global_settings: Any | None = None,
    engine_pool: Any | None = None,
    settings_manager: Any | None = None,
    get_sampling_params: Any | None = None,
) -> str:
    if _is_plain_text_attachment(file):
        return decode_plain_text_attachment(file)

    if file.extension == ".pdf":
        pdf_engine = markitdown_pdf_processing_engine(global_settings)
        if pdf_engine != PDF_PROCESSING_MARKITDOWN:
            from .markitdown_pdf_fallback import convert_pdf_with_ocr_engine

            return await convert_pdf_with_ocr_engine(
                file,
                engine_model_id=pdf_engine,
                engine_pool=engine_pool,
                settings_manager=settings_manager,
                global_settings=global_settings,
                get_sampling_params=get_sampling_params,
            )

    import asyncio

    return await asyncio.to_thread(
        convert_file_to_markdown,
        file,
        global_settings=global_settings,
    )


async def stream_attachment_to_markdown_async(
    file: MarkItDownFile,
    *,
    global_settings: Any | None = None,
    engine_pool: Any | None = None,
    settings_manager: Any | None = None,
    get_sampling_params: Any | None = None,
):
    if _is_plain_text_attachment(file):
        markdown = decode_plain_text_attachment(file)
        yield _format_attachment_markdown(file.filename, markdown)
        return

    if file.extension == ".pdf":
        pdf_engine = markitdown_pdf_processing_engine(global_settings)
        if pdf_engine != PDF_PROCESSING_MARKITDOWN:
            from .markitdown_pdf_fallback import stream_pdf_with_ocr_engine

            yield f"## Attached file: {file.filename}\n\n"
            async for chunk in stream_pdf_with_ocr_engine(
                file,
                engine_model_id=pdf_engine,
                engine_pool=engine_pool,
                settings_manager=settings_manager,
                global_settings=global_settings,
                get_sampling_params=get_sampling_params,
            ):
                yield chunk
            return

    markdown = await convert_attachment_to_markdown_async(
        file,
        global_settings=global_settings,
        engine_pool=engine_pool,
        settings_manager=settings_manager,
        get_sampling_params=get_sampling_params,
    )
    yield _format_attachment_markdown(file.filename, markdown)


def decode_plain_text_attachment(file: MarkItDownFile) -> str:
    try:
        text = file.data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise MarkItDownRequestError(
            f"Text attachment must be UTF-8 encoded: {file.filename}",
            status_code=400,
        ) from exc

    markdown = text.strip()
    if not markdown:
        raise MarkItDownRequestError(MARKITDOWN_EMPTY_MESSAGE, status_code=400)
    return markdown


def convert_file_to_markdown(
    file: MarkItDownFile,
    *,
    global_settings: Any | None = None,
) -> str:
    if file.extension == ".pdf":
        return convert_pdf_file_to_markdown(file)

    markdown = _convert_file_with_markitdown(file)
    if not _has_extractable_text(markdown):
        logger.warning(
            "MarkItDown found no extractable text: filename=%s mime_type=%s",
            file.filename,
            file.mime_type,
        )
        raise MarkItDownRequestError(MARKITDOWN_EMPTY_MESSAGE, status_code=400)

    return markdown


def convert_pdf_file_to_markdown(
    file: MarkItDownFile,
) -> str:
    quiet_pdf_parser_loggers()
    markdown = _convert_file_with_markitdown(file)
    if not _has_extractable_text(markdown):
        logger.warning(
            "MarkItDown found no extractable text: filename=%s mime_type=%s",
            file.filename,
            file.mime_type,
        )
        raise MarkItDownRequestError(MARKITDOWN_EMPTY_PDF_MESSAGE, status_code=400)
    return markdown


def _convert_file_with_markitdown(file: MarkItDownFile) -> str:
    converter = _get_converter()

    try:
        from markitdown import StreamInfo
    except ImportError as exc:
        raise RuntimeError(
            "MarkItDown is not installed. Install markitdown[pdf,docx,pptx]."
        ) from exc

    stream = io.BytesIO(file.data)
    stream_info = StreamInfo(
        extension=file.extension,
        mimetype=file.mime_type or None,
        filename=file.filename,
    )

    try:
        with _converter_lock:
            result = converter.convert_stream(stream, stream_info=stream_info)
    except Exception as exc:
        logger.warning(
            "MarkItDown conversion failed: filename=%s mime_type=%s error=%s",
            file.filename,
            file.mime_type,
            exc,
        )
        raise MarkItDownRequestError(
            f"Failed to convert attached file '{file.filename}': {exc}",
            status_code=400,
        ) from exc

    return _normalize_markdown_text(getattr(result, "markdown", "") or "")


def _normalize_markdown_text(markdown: str) -> str:
    lines = [
        line.rstrip()
        for line in (markdown or "")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .split("\n")
    ]
    normalized: list[str] = []
    previous_blank = False
    for line in lines:
        is_blank = not line.strip()
        if is_blank and previous_blank:
            continue
        normalized.append(line)
        previous_blank = is_blank
    return "\n".join(normalized).strip()


def _get_converter() -> Any:
    global _converter
    if _converter is not None:
        return _converter
    with _converter_lock:
        if _converter is not None:
            return _converter
        try:
            from markitdown import MarkItDown
        except ImportError as exc:
            raise RuntimeError(
                "MarkItDown is not installed. Install markitdown[pdf,docx,pptx]."
            ) from exc
        _converter = MarkItDown(enable_plugins=False)
        return _converter


def _iter_file_part_dicts(content: Any) -> list[dict[str, Any]]:
    if not isinstance(content, list):
        return []
    return [
        part
        for part in (_part_to_dict(item) for item in content)
        if part.get("type") == "file"
    ]


def _part_to_dict(part: Any) -> dict[str, Any]:
    if hasattr(part, "model_dump"):
        return part.model_dump(exclude_none=True)
    if hasattr(part, "dict"):
        return part.dict(exclude_none=True)
    return part if isinstance(part, dict) else {}


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    text_parts: list[str] = []
    for part in content:
        part_dict = _part_to_dict(part)
        if part_dict.get("type") in {"text", "input_text"}:
            text_parts.append(str(part_dict.get("text") or ""))
    return "\n".join(p for p in text_parts if p)


def _format_attachment_markdown(filename: str, markdown: str) -> str:
    return f"## Attached file: {filename}\n\n{markdown.strip()}"


def _validate_supported_file(filename: str, mime_type: str) -> None:
    extension = Path(filename).suffix.lower()
    if (
        extension in MARKITDOWN_UNSUPPORTED_SPREADSHEET_EXTENSIONS
        or mime_type in MARKITDOWN_SPREADSHEET_MIME_TYPES
    ):
        raise MarkItDownRequestError(
            "Spreadsheet attachments are not supported. "
            "Supported formats are PDF, DOCX, PPTX, TXT, and MD.",
            status_code=400,
        )
    if extension not in SUPPORTED_ATTACHMENT_EXTENSIONS:
        raise MarkItDownRequestError(
            "Unsupported attachment type. "
            "Supported formats are PDF, DOCX, PPTX, TXT, and MD.",
            status_code=400,
        )
    if mime_type and mime_type not in SUPPORTED_ATTACHMENT_MIME_TYPES:
        raise MarkItDownRequestError(
            "Unsupported attachment MIME type. "
            "Supported formats are PDF, DOCX, PPTX, TXT, and MD.",
            status_code=400,
        )


def _mime_type_for_extension(extension: str) -> str:
    if extension == ".txt":
        return "text/plain"
    if extension == ".md":
        return "text/markdown"
    if extension == ".pdf":
        return "application/pdf"
    if extension == ".docx":
        return DOCX_MIME_TYPE
    if extension == ".pptx":
        return PPTX_MIME_TYPE
    return "application/octet-stream"


def _extension_for_mime_type(mime_type: str) -> str:
    if mime_type == "application/pdf":
        return ".pdf"
    if mime_type == DOCX_MIME_TYPE:
        return ".docx"
    if mime_type == PPTX_MIME_TYPE:
        return ".pptx"
    if mime_type == "text/plain":
        return ".txt"
    if mime_type in {"text/markdown", "text/x-markdown"}:
        return ".md"
    return ""


def _is_plain_text_attachment(file: MarkItDownFile) -> bool:
    return file.extension in PLAIN_TEXT_SUPPORTED_EXTENSIONS


def _decode_data(value: str) -> bytes:
    data = value.strip()
    if data.startswith("data:"):
        marker = ";base64,"
        idx = data.find(marker)
        if idx < 0:
            raise MarkItDownRequestError("Only base64 data URIs are supported.")
        data = data[idx + len(marker) :]
    try:
        return base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise MarkItDownRequestError("File data must be valid base64.") from exc


def _mime_type_from_data_uri(value: str) -> str:
    data = value.strip()
    if not data.startswith("data:"):
        return ""
    marker = ";base64,"
    idx = data.find(marker)
    if idx < 0:
        return ""
    return data[5:idx].strip().lower()


def _has_extractable_text(markdown: str) -> bool:
    text = markdown.strip()
    if not text:
        return False
    return any(ch.isalnum() for ch in text)
