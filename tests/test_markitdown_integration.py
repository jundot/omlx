# SPDX-License-Identifier: Apache-2.0
"""Tests for the MarkItDown integration."""

from __future__ import annotations

import asyncio
import base64
import logging
import sys
import types
import warnings
from dataclasses import dataclass
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from starlette.responses import StreamingResponse

import omlx.server as server_module
from omlx.api.markitdown import (
    MARKITDOWN_EMPTY_PDF_MESSAGE,
    MARKITDOWN_MODEL_ID,
    MarkItDownFile,
    MarkItDownRequestError,
    convert_file_to_markdown,
    parse_file_part,
    preprocess_markitdown_file_parts,
    preprocess_markitdown_file_parts_async,
    quiet_pdf_parser_loggers,
)
from omlx.api.markitdown_pdf_fallback import (
    convert_pdf_with_ocr_engine,
    resolve_pdf_ocr_model,
    stream_pdf_with_ocr_engine,
)
from omlx.api.openai_models import ChatCompletionRequest, Message
from omlx.engine_pool import EngineEntry, EnginePool
from omlx.server import ServerState, app
from omlx.settings import GlobalSettings


def _data_uri(payload: bytes = b"doc", mime_type: str = "application/pdf") -> str:
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _file_part(
    filename: str = "sample.pdf",
    data: str | None = None,
    mime_type: str = "application/pdf",
) -> dict:
    return {
        "type": "file",
        "file": {
            "filename": filename,
            "mime_type": mime_type,
            "file_data": data or _data_uri(mime_type=mime_type),
        },
    }


class _EmptyPool:
    def get_status(self) -> dict:
        return {
            "final_ceiling": 0,
            "current_model_memory": 0,
            "model_count": 0,
            "loaded_count": 0,
            "models": [],
        }


def test_openai_models_includes_markitdown_when_enabled():
    state = ServerState()
    state.engine_pool = _EmptyPool()
    state.global_settings = GlobalSettings()

    with patch("omlx.server._server_state", state):
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/v1/models")

    assert response.status_code == 200
    ids = [m["id"] for m in response.json()["data"]]
    assert "MarkItDown" in ids


def test_openai_models_hides_markitdown_when_disabled():
    state = ServerState()
    state.engine_pool = _EmptyPool()
    state.global_settings = GlobalSettings()
    state.global_settings.integrations.markitdown_enabled = False

    with patch("omlx.server._server_state", state):
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/v1/models")

    assert response.status_code == 200
    ids = [m["id"] for m in response.json()["data"]]
    assert MARKITDOWN_MODEL_ID not in ids


def test_openai_models_hides_markitdown_when_not_exposed():
    state = ServerState()
    state.engine_pool = _EmptyPool()
    state.global_settings = GlobalSettings()
    state.global_settings.integrations.markitdown_expose_model = False

    with patch("omlx.server._server_state", state):
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/v1/models")

    assert response.status_code == 200
    ids = [m["id"] for m in response.json()["data"]]
    assert MARKITDOWN_MODEL_ID not in ids


def test_markitdown_chat_completion_converts_file(monkeypatch):
    state = ServerState()
    state.engine_pool = _EmptyPool()
    state.global_settings = GlobalSettings()

    def fake_convert(file: MarkItDownFile, **kwargs) -> str:
        assert file.filename == "sample.pdf"
        return "# Converted"

    monkeypatch.setattr("omlx.api.markitdown.convert_file_to_markdown", fake_convert)

    with patch("omlx.server._server_state", state):
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": MARKITDOWN_MODEL_ID,
                "messages": [{"role": "user", "content": [_file_part()]}],
            },
        )

    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    assert "## Attached file: sample.pdf" in content
    assert "# Converted" in content


def test_markitdown_chat_completion_uses_latest_user_turn(monkeypatch):
    state = ServerState()
    state.engine_pool = _EmptyPool()
    state.global_settings = GlobalSettings()

    def fake_convert(file: MarkItDownFile, **kwargs) -> str:
        return f"# Converted {file.filename}"

    monkeypatch.setattr("omlx.api.markitdown.convert_file_to_markdown", fake_convert)

    with patch("omlx.server._server_state", state):
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": MARKITDOWN_MODEL_ID,
                "messages": [
                    {"role": "user", "content": [_file_part("file1.pdf")]},
                    {
                        "role": "assistant",
                        "content": (
                            "## Attached file: file1.pdf\n\n# Converted file1.pdf"
                        ),
                    },
                    {"role": "user", "content": [_file_part("file2.pdf")]},
                ],
            },
        )

    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    assert "file2.pdf" in content
    assert "file1.pdf" not in content


def test_markitdown_chat_completion_disabled_returns_404():
    state = ServerState()
    state.engine_pool = _EmptyPool()
    state.global_settings = GlobalSettings()
    state.global_settings.integrations.markitdown_enabled = False

    with patch("omlx.server._server_state", state):
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": MARKITDOWN_MODEL_ID,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 404
    assert "markitdown" in response.json()["error"]["message"].lower()


def test_markitdown_chat_completion_hidden_model_returns_404():
    state = ServerState()
    state.engine_pool = _EmptyPool()
    state.global_settings = GlobalSettings()
    state.global_settings.integrations.markitdown_expose_model = False

    with patch("omlx.server._server_state", state):
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": MARKITDOWN_MODEL_ID,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 404
    assert "markitdown" in response.json()["error"]["message"].lower()


def test_markitdown_stream_response_starts_before_conversion(monkeypatch):
    state = ServerState()
    state.engine_pool = _EmptyPool()
    state.global_settings = GlobalSettings()
    started = False

    async def fake_stream_messages(*args, **kwargs):
        nonlocal started
        started = True
        yield "Converted markdown"

    monkeypatch.setattr(
        server_module,
        "stream_messages_to_markdown_async",
        fake_stream_messages,
    )
    request = ChatCompletionRequest(
        model=MARKITDOWN_MODEL_ID,
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
    )

    async def exercise():
        with patch("omlx.server._server_state", state):
            response = await server_module._create_markitdown_chat_completion(
                request,
                None,
            )

        assert isinstance(response, StreamingResponse)
        assert started is False

        iterator = response.body_iterator.__aiter__()
        first = await iterator.__anext__()
        assert first.startswith("data: ")
        assert started is False

        role_chunk = await iterator.__anext__()
        assert '"role":"assistant"' in role_chunk
        assert started is False

        content_chunk = await iterator.__anext__()
        assert "Converted markdown" in content_chunk
        assert started is True
        await iterator.aclose()

    asyncio.run(exercise())


def test_markitdown_non_stream_response_starts_before_conversion(monkeypatch):
    state = ServerState()
    state.engine_pool = _EmptyPool()
    state.global_settings = GlobalSettings()
    started = False

    async def fake_convert_messages(*args, **kwargs):
        nonlocal started
        started = True
        return "Converted markdown"

    monkeypatch.setattr(
        server_module,
        "convert_messages_to_markdown_async",
        fake_convert_messages,
    )
    request = ChatCompletionRequest(
        model=MARKITDOWN_MODEL_ID,
        messages=[{"role": "user", "content": "hello"}],
    )

    async def exercise():
        with patch("omlx.server._server_state", state):
            response = await server_module._create_markitdown_chat_completion(
                request,
                None,
            )

        assert isinstance(response, StreamingResponse)
        assert started is False

        iterator = response.body_iterator.__aiter__()
        first = await iterator.__anext__()
        assert first == " "
        chunk = await iterator.__anext__()
        assert "Converted markdown" in chunk
        assert started is True
        await iterator.aclose()

    asyncio.run(exercise())


def test_preprocess_file_parts_for_llm(monkeypatch):
    def fake_convert(file: MarkItDownFile, **kwargs) -> str:
        return "Converted text"

    monkeypatch.setattr("omlx.api.markitdown.convert_file_to_markdown", fake_convert)
    messages = [
        Message(
            role="user",
            content=[
                {"type": "text", "text": "Summarize this."},
                _file_part("paper.pdf"),
            ],
        )
    ]

    processed = preprocess_markitdown_file_parts(
        messages,
        global_settings=GlobalSettings(),
    )

    parts = processed[0].content
    assert isinstance(parts, list)
    assert parts[0].type == "text"
    assert parts[1].type == "text"
    assert "## Attached file: paper.pdf" in (parts[1].text or "")
    assert "Converted text" in (parts[1].text or "")


def test_preprocess_file_parts_works_when_model_not_exposed(monkeypatch):
    def fake_convert(file: MarkItDownFile, **kwargs) -> str:
        return "Converted text"

    monkeypatch.setattr("omlx.api.markitdown.convert_file_to_markdown", fake_convert)
    settings = GlobalSettings()
    settings.integrations.markitdown_expose_model = False

    processed = preprocess_markitdown_file_parts(
        [Message(role="user", content=[_file_part("paper.pdf")])],
        global_settings=settings,
    )

    parts = processed[0].content
    assert isinstance(parts, list)
    assert parts[0].type == "text"
    assert "Converted text" in (parts[0].text or "")


def test_text_and_markdown_file_parts_are_inlined_without_converter(monkeypatch):
    called = False

    def fake_convert(file: MarkItDownFile, **kwargs) -> str:
        nonlocal called
        called = True
        raise AssertionError("plain text attachments should not use MarkItDown")

    monkeypatch.setattr("omlx.api.markitdown.convert_file_to_markdown", fake_convert)

    processed = preprocess_markitdown_file_parts(
        [
            Message(
                role="user",
                content=[
                    _file_part(
                        "notes.txt",
                        data=_data_uri(b"Plain notes", mime_type="text/plain"),
                        mime_type="text/plain",
                    ),
                    _file_part(
                        "guide.md",
                        data=_data_uri(b"# Guide", mime_type="text/markdown"),
                        mime_type="text/markdown",
                    ),
                ],
            )
        ],
        global_settings=GlobalSettings(),
    )

    parts = processed[0].content
    assert isinstance(parts, list)
    rendered = "\n".join(part.text or "" for part in parts)
    assert "## Attached file: notes.txt" in rendered
    assert "Plain notes" in rendered
    assert "## Attached file: guide.md" in rendered
    assert "# Guide" in rendered
    assert called is False


def test_preprocess_file_parts_does_not_create_mixed_content_warning(monkeypatch):
    def fake_convert(file: MarkItDownFile, **kwargs) -> str:
        return "Converted text"

    monkeypatch.setattr("omlx.api.markitdown.convert_file_to_markdown", fake_convert)
    messages = [
        Message(
            role="user",
            content=[
                {"type": "text", "text": "Summarize this."},
                _file_part("paper.pdf"),
            ],
        )
    ]

    processed = preprocess_markitdown_file_parts(
        messages,
        global_settings=GlobalSettings(),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        processed[0].model_dump()

    assert not [
        warning
        for warning in caught
        if "Pydantic serializer warnings" in str(warning.message)
    ]


def test_preprocess_file_parts_rejects_when_disabled():
    settings = GlobalSettings()
    settings.integrations.markitdown_enabled = False

    with pytest.raises(MarkItDownRequestError, match="disabled"):
        preprocess_markitdown_file_parts(
            [Message(role="user", content=[_file_part()])],
            global_settings=settings,
        )


def test_xlsx_is_rejected_without_pandas_dependency():
    part = {
        "type": "file",
        "file": {
            "filename": "sheet.xlsx",
            "mime_type": (
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ),
            "file_data": _data_uri(),
        },
    }

    with pytest.raises(MarkItDownRequestError, match="Spreadsheet"):
        parse_file_part(part, max_file_size_mb=25)


def test_openai_file_data_without_filename_infers_from_data_uri():
    parsed = parse_file_part(
        {
            "type": "file",
            "file": {
                "file_data": _data_uri(
                    b"Plain notes",
                    mime_type="text/plain",
                ),
            },
        },
        max_file_size_mb=25,
    )

    assert parsed.filename == "attachment.txt"
    assert parsed.mime_type == "text/plain"
    assert parsed.data == b"Plain notes"


def test_file_id_is_rejected():
    with pytest.raises(MarkItDownRequestError, match="file_id"):
        parse_file_part(
            {"type": "file", "file": {"file_id": "file_123", "filename": "x.pdf"}},
            max_file_size_mb=25,
        )


def test_empty_pdf_conversion_logs_warning(monkeypatch, caplog):
    @dataclass(frozen=True)
    class FakeStreamInfo:
        extension: str | None = None
        mimetype: str | None = None
        filename: str | None = None

    class FakeResult:
        markdown = ""

    class FakeConverter:
        def convert_stream(self, stream, stream_info=None):
            return FakeResult()

    fake_markitdown = types.ModuleType("markitdown")
    fake_markitdown.StreamInfo = FakeStreamInfo
    monkeypatch.setitem(sys.modules, "markitdown", fake_markitdown)
    monkeypatch.setattr("omlx.api.markitdown._converter", FakeConverter())

    caplog.set_level("WARNING")
    with pytest.raises(MarkItDownRequestError) as exc_info:
        convert_file_to_markdown(
            MarkItDownFile(
                filename="scan.pdf",
                mime_type="application/pdf",
                data=b"%PDF",
            )
        )

    assert exc_info.value.detail == MARKITDOWN_EMPTY_PDF_MESSAGE
    assert "no extractable text" in caplog.text.lower()


def test_pdf_parser_debug_loggers_are_quieted():
    logger_names = ("pdfminer", "pdfminer.psparser", "pdfminer.pdfinterp")
    original_levels = {
        name: logging.getLogger(name).level for name in logger_names
    }
    try:
        for name in logger_names:
            logging.getLogger(name).setLevel(logging.DEBUG)

        quiet_pdf_parser_loggers()

        for name in logger_names:
            assert logging.getLogger(name).level == logging.WARNING
    finally:
        for name, level in original_levels.items():
            logging.getLogger(name).setLevel(level)


def test_ocr_pdf_engine_converts_pages_in_order_limits_concurrency_and_unloads(
    monkeypatch,
):
    class FakePool:
        def __init__(self):
            self.engine = FakeEngine()
            self.unloaded = []

        def resolve_model_id(self, model_id, settings_manager):
            return model_id

        def get_entry(self, model_id):
            return types.SimpleNamespace(
                config_model_type="dots_ocr",
                engine_type="vlm",
            )

        def acquire(self, model_id):
            assert model_id == "OCR-Model"

            class Lease:
                async def __aenter__(lease_self):
                    return self.engine

                async def __aexit__(lease_self, exc_type, exc, tb):
                    return None

            return Lease()

        async def unload_if_idle_unpinned(self, model_id):
            self.unloaded.append(model_id)
            return True

    active = 0
    max_active = 0

    class FakeEngine:
        async def chat(self, messages, **kwargs):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            try:
                await asyncio.sleep(0.01)
                url = messages[0]["content"][0]["image_url"]["url"]
                return types.SimpleNamespace(text=f"markdown for {url[-1]}")
            finally:
                active -= 1

    settings = GlobalSettings()
    settings.scheduler.max_concurrent_requests = 2
    pool = FakePool()

    monkeypatch.setattr(
        "omlx.api.markitdown_pdf_fallback.render_pdf_pages_to_image_data_uris",
        lambda file: ["page1", "page2", "page3"],
    )

    markdown = asyncio.run(
        convert_pdf_with_ocr_engine(
            MarkItDownFile(
                filename="scan.pdf",
                mime_type="application/pdf",
                data=b"%PDF",
            ),
            engine_model_id="OCR-Model",
            engine_pool=pool,
            settings_manager=None,
            global_settings=settings,
            get_sampling_params=lambda *_args: (
                0.0,
                1.0,
                0,
                1.0,
                0.0,
                0.0,
                0.0,
                4096,
                0.0,
                0.1,
            ),
        )
    )

    assert "### Page 1\n\nmarkdown for 1" in markdown
    assert "### Page 2\n\nmarkdown for 2" in markdown
    assert "### Page 3\n\nmarkdown for 3" in markdown
    assert max_active == 2
    assert pool.unloaded == ["OCR-Model"]


def test_ocr_pdf_engine_streams_ready_prefix_in_page_order(monkeypatch):
    class FakePool:
        def __init__(self):
            self.engine = FakeEngine()

        def resolve_model_id(self, model_id, settings_manager):
            return model_id

        def get_entry(self, model_id):
            return types.SimpleNamespace(
                config_model_type="dots_ocr",
                engine_type="vlm",
            )

        def acquire(self, model_id):
            class Lease:
                async def __aenter__(lease_self):
                    return self.engine

                async def __aexit__(lease_self, exc_type, exc, tb):
                    return None

            return Lease()

        async def unload_if_idle_unpinned(self, model_id):
            return True

    class FakeEngine:
        async def chat(self, messages, **kwargs):
            url = messages[0]["content"][0]["image_url"]["url"]
            delay_by_page = {"page1": 0.03, "page2": 0.0, "page3": 0.01}
            await asyncio.sleep(delay_by_page[url])
            return types.SimpleNamespace(text=f"markdown for {url[-1]}")

    settings = GlobalSettings()
    settings.scheduler.max_concurrent_requests = 3
    monkeypatch.setattr(
        "omlx.api.markitdown_pdf_fallback.render_pdf_pages_to_image_data_uris",
        lambda file: ["page1", "page2", "page3"],
    )

    async def collect_chunks():
        chunks = []
        async for chunk in stream_pdf_with_ocr_engine(
            MarkItDownFile(
                filename="scan.pdf",
                mime_type="application/pdf",
                data=b"%PDF",
            ),
            engine_model_id="OCR-Model",
            engine_pool=FakePool(),
            settings_manager=None,
            global_settings=settings,
            get_sampling_params=lambda *_args: (
                0.0,
                1.0,
                0,
                1.0,
                0.0,
                0.0,
                0.0,
                4096,
                0.0,
                0.1,
            ),
        ):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(collect_chunks())

    assert [chunk.split("\n", 1)[0] for chunk in chunks] == [
        "### Page 1",
        "### Page 2",
        "### Page 3",
    ]


def test_ocr_pdf_engine_rejects_non_ocr_config_model_type():
    class FakePool:
        def resolve_model_id(self, model_id, settings_manager):
            return model_id

        def get_entry(self, model_id):
            return types.SimpleNamespace(
                config_model_type="qwen3_vl",
                engine_type="vlm",
            )

    with pytest.raises(MarkItDownRequestError, match="OCR"):
        resolve_pdf_ocr_model(
            "plain-model",
            engine_pool=FakePool(),
            settings_manager=None,
        )


def test_ocr_pdf_engine_ignores_model_id_ocr_without_ocr_config_model_type():
    class FakePool:
        def resolve_model_id(self, model_id, settings_manager):
            return model_id

        def get_entry(self, model_id):
            return types.SimpleNamespace(
                model_id="Invoice-OCR",
                source_repo_id=None,
                config_model_type="qwen2_vl",
                engine_type="vlm",
            )

    with pytest.raises(MarkItDownRequestError, match="config model_type"):
        resolve_pdf_ocr_model(
            "Invoice-OCR",
            engine_pool=FakePool(),
            settings_manager=None,
        )


def test_async_preprocess_uses_ocr_pdf_processing_engine(monkeypatch):
    class FakePool:
        def resolve_model_id(self, model_id, settings_manager):
            return model_id

        def get_entry(self, model_id):
            return types.SimpleNamespace(
                config_model_type="dots_ocr",
                engine_type="vlm",
            )

        def acquire(self, model_id):
            class Lease:
                async def __aenter__(self):
                    return types.SimpleNamespace(
                        chat=lambda **kwargs: None,
                    )

                async def __aexit__(self, exc_type, exc, tb):
                    return None

            return Lease()

    async def fake_convert(file, **kwargs):
        assert file.filename == "paper.pdf"
        assert kwargs["engine_model_id"] == "OCR-Model"
        return "OCR markdown"

    settings = GlobalSettings()
    settings.integrations.markitdown_pdf_processing_engine = "OCR-Model"
    monkeypatch.setattr(
        "omlx.api.markitdown_pdf_fallback.convert_pdf_with_ocr_engine",
        fake_convert,
    )

    processed = asyncio.run(
        preprocess_markitdown_file_parts_async(
            [Message(role="user", content=[_file_part("paper.pdf")])],
            global_settings=settings,
            engine_pool=FakePool(),
            settings_manager=None,
            get_sampling_params=lambda *_args: None,
        )
    )

    parts = processed[0].content
    assert isinstance(parts, list)
    assert "OCR markdown" in (parts[0].text or "")


def test_unload_if_idle_unpinned_skips_pinned_and_unloads_idle(monkeypatch):
    class FakeEngine:
        def has_active_requests(self):
            return False

    pool = EnginePool()
    pool._entries["OCR-Model"] = EngineEntry(
        model_id="OCR-Model",
        model_path="/models/OCR-Model",
        model_type="vlm",
        engine_type="vlm",
        estimated_size=1,
        config_model_type="dots_ocr",
        is_pinned=True,
        engine=FakeEngine(),
    )

    unloaded = []

    async def fake_unload(model_id):
        unloaded.append(model_id)

    monkeypatch.setattr(pool, "_unload_engine", fake_unload)

    assert asyncio.run(pool.unload_if_idle_unpinned("OCR-Model")) is False
    assert unloaded == []

    pool._entries["OCR-Model"].is_pinned = False
    assert asyncio.run(pool.unload_if_idle_unpinned("OCR-Model")) is True
    assert unloaded == ["OCR-Model"]
