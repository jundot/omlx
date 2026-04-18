# SPDX-License-Identifier: Apache-2.0
"""Integration tests for POST /admin/api/chat/attach.

Uses FastAPI TestClient to call the actual route, covering extraction,
validation, auth, and error paths. No live inference server required —
the engine pool and auth dependencies are overridden.
"""
import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# App fixture — build a minimal FastAPI app with the admin router mounted
# and all external dependencies overridden.
# ---------------------------------------------------------------------------

def make_app(engine_pool=None):
    """Return a TestClient-ready FastAPI app with the attach route mounted."""
    from omlx.admin import routes as admin_routes
    from omlx.admin.auth import require_admin

    app = FastAPI()
    app.include_router(admin_routes.router)

    # Bypass authentication — all requests treated as authenticated admin.
    app.dependency_overrides[require_admin] = lambda: True

    # Inject a null or mock engine pool via the module-level setter.
    admin_routes.set_admin_getters(
        state_getter=lambda: None,
        pool_getter=lambda: engine_pool,
        settings_manager_getter=lambda: None,
        global_settings_getter=lambda: MagicMock(
            auth=MagicMock(api_key="test", skip_api_key_verification=False)
        ),
    )

    return app


@pytest.fixture()
def client():
    """TestClient with no engine pool (token count will be approximate)."""
    return TestClient(make_app(engine_pool=None))


@pytest.fixture()
def client_with_tokenizer():
    """TestClient with a mock engine that exposes a tokenizer."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode = lambda text: list(range(max(1, len(text) // 4)))

    mock_engine = MagicMock()
    mock_engine._tokenizer = mock_tokenizer

    mock_entry = MagicMock()
    mock_entry.engine = mock_engine

    mock_pool = MagicMock()
    mock_pool.get_loaded_model_ids.return_value = ["test-model"]
    mock_pool.get_entry.return_value = mock_entry

    return TestClient(make_app(engine_pool=mock_pool))


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _upload(client, content: bytes, filename: str):
    """POST a file to /admin/api/chat/attach."""
    return client.post(
        "/admin/api/chat/attach",
        files={"file": (filename, io.BytesIO(content), "application/octet-stream")},
    )


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class TestAuth:
    def test_unauthenticated_returns_401_or_403(self):
        """Without the dependency override, auth should reject the request."""
        from omlx.admin import routes as admin_routes
        from omlx.admin.auth import require_admin

        app = FastAPI()
        app.include_router(admin_routes.router)
        # No override — require_admin runs normally with no session cookie.
        admin_routes.set_admin_getters(
            state_getter=lambda: None,
            pool_getter=lambda: None,
            settings_manager_getter=lambda: None,
            global_settings_getter=lambda: MagicMock(
                auth=MagicMock(api_key="secret", skip_api_key_verification=False)
            ),
        )
        unauthed_client = TestClient(app, raise_server_exceptions=False)
        resp = _upload(unauthed_client, b"hello", "test.txt")
        assert resp.status_code in (401, 403)


# ---------------------------------------------------------------------------
# Markdown / plain text
# ---------------------------------------------------------------------------

class TestMarkdownAttach:
    def test_valid_markdown(self, client):
        content = b"# Title\n\nSome body text.\n"
        resp = _upload(client, content, "notes.md")
        assert resp.status_code == 200
        data = resp.json()
        assert data["filename"] == "notes.md"
        assert "# Title" in data["content"]
        assert data["char_count"] == len(content.decode())
        assert data["token_count"] > 0
        assert data["token_count_exact"] is False  # no tokenizer in this fixture

    def test_valid_txt(self, client):
        resp = _upload(client, b"Plain text content.\n", "doc.txt")
        assert resp.status_code == 200
        assert resp.json()["filename"] == "doc.txt"

    def test_empty_markdown_rejected(self, client):
        resp = _upload(client, b"   \n\n  ", "empty.md")
        assert resp.status_code == 400
        assert "empty" in resp.json()["detail"].lower()

    def test_non_utf8_rejected(self, client):
        resp = _upload(client, bytes([0xFF, 0xFE, 0x80, 0x81]), "bad.md")
        assert resp.status_code == 422
        assert "utf-8" in resp.json()["detail"].lower()

    def test_oversized_rejected(self, client):
        big = ("x" * 51_000).encode()
        resp = _upload(client, big, "big.md")
        assert resp.status_code == 400
        assert "limit" in resp.json()["detail"].lower()

    def test_exact_tokenizer_when_model_loaded(self, client_with_tokenizer):
        resp = _upload(client_with_tokenizer, b"Hello world.\n", "hello.md")
        assert resp.status_code == 200
        data = resp.json()
        assert data["token_count_exact"] is True
        assert data["token_count"] > 0


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

class TestJsonAttach:
    def test_valid_json_object(self, client):
        payload = json.dumps({"key": "value", "number": 42}).encode()
        resp = _upload(client, payload, "data.json")
        assert resp.status_code == 200
        data = resp.json()
        assert '"key"' in data["content"]
        assert data["char_count"] > 0

    def test_valid_json_array(self, client):
        payload = json.dumps([1, 2, 3]).encode()
        resp = _upload(client, payload, "list.json")
        assert resp.status_code == 200

    def test_invalid_json_rejected(self, client):
        resp = _upload(client, b"{bad json}", "bad.json")
        assert resp.status_code == 422
        detail = resp.json()["detail"].lower()
        assert "json" in detail or "invalid" in detail

    def test_empty_object_rejected(self, client):
        resp = _upload(client, b"{}", "empty.json")
        assert resp.status_code == 422
        assert "empty" in resp.json()["detail"].lower()

    def test_null_rejected(self, client):
        resp = _upload(client, b"null", "null.json")
        assert resp.status_code == 422

    def test_empty_array_rejected(self, client):
        resp = _upload(client, b"[]", "empty.json")
        assert resp.status_code == 422

    def test_raw_size_limit_enforced_before_parse(self, client):
        # A JSON file that is large before parsing
        big = ('{"data": "' + "x" * 51_000 + '"}').encode()
        resp = _upload(client, big, "big.json")
        assert resp.status_code == 400
        assert "limit" in resp.json()["detail"].lower()

    def test_indent_expansion_caught(self, client):
        # Compact JSON under 50k chars that expands past limit after indent=2
        # Build a deeply nested structure that grows when pretty-printed
        inner = {"k" * 10: "v" * 40}
        data = {f"key_{i}": inner for i in range(500)}
        compact = json.dumps(data, separators=(",", ":")).encode()
        # Confirm raw is under the char limit
        assert len(compact) < 50_000
        # But indented version may exceed it
        indented = json.dumps(data, indent=2)
        if len(indented) > 50_000:
            resp = _upload(client, compact, "expanding.json")
            assert resp.status_code == 400
            assert "limit" in resp.json()["detail"].lower()
        else:
            # If this particular structure doesn't expand past limit, skip
            pytest.skip("test data did not expand past limit — adjust structure")


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

class TestPdfAttach:
    def test_pdf_without_pypdf_returns_501(self, client):
        """When pypdf is not installed the route returns 501."""
        with patch.dict("sys.modules", {"pypdf": None}):
            # Force re-import failure inside the route
            with patch("builtins.__import__", side_effect=_make_import_raiser("pypdf")):
                resp = _upload(client, b"%PDF-1.4 fake", "doc.pdf")
        # 501 if pypdf truly absent; 422 if pypdf is installed and rejects the fake bytes
        assert resp.status_code in (501, 422)

    def test_corrupt_pdf_returns_422(self, client):
        pytest.importorskip("pypdf", reason="pypdf not installed — skip PDF tests")
        resp = _upload(client, b"this is not a pdf", "corrupt.pdf")
        assert resp.status_code == 422
        assert "pdf" in resp.json()["detail"].lower()

    def test_valid_text_pdf(self, client):
        """Test with a real minimal PDF containing extractable text."""
        pytest.importorskip("pypdf", reason="pypdf not installed — skip PDF tests")
        pdf_bytes = _make_minimal_pdf("Hello from the PDF.")
        resp = _upload(client, pdf_bytes, "test.pdf")
        assert resp.status_code == 200
        data = resp.json()
        assert data["char_count"] > 0
        assert data["token_count"] > 0

    def test_encrypted_pdf_returns_422(self, client):
        """Encrypted PDF without empty-string password is rejected."""
        pytest.importorskip("pypdf", reason="pypdf not installed — skip PDF tests")
        # Create a mock PdfReader that reports is_encrypted=True and decrypt fails
        mock_reader = MagicMock()
        mock_reader.is_encrypted = True
        mock_reader.decrypt.return_value = 0  # failed decrypt

        with patch("pypdf.PdfReader", return_value=mock_reader):
            resp = _upload(client, b"%PDF-1.4", "encrypted.pdf")
        assert resp.status_code == 422
        assert "password" in resp.json()["detail"].lower()

    def test_scanned_pdf_returns_422(self, client):
        """PDF with no extractable text (scanned image) is rejected."""
        pytest.importorskip("pypdf", reason="pypdf not installed — skip PDF tests")
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""

        mock_reader = MagicMock()
        mock_reader.is_encrypted = False
        mock_reader.pages = [mock_page]

        with patch("pypdf.PdfReader", return_value=mock_reader):
            resp = _upload(client, b"%PDF-1.4", "scanned.pdf")
        assert resp.status_code == 422
        assert "scanned" in resp.json()["detail"].lower() or \
               "no text" in resp.json()["detail"].lower()

    def test_oversized_pdf_returns_400(self, client):
        """PDF whose extracted text exceeds 50k chars is rejected cleanly."""
        pytest.importorskip("pypdf", reason="pypdf not installed — skip PDF tests")
        big_text = "word " * 15_000  # well over 50k chars

        mock_page = MagicMock()
        mock_page.extract_text.return_value = big_text

        mock_reader = MagicMock()
        mock_reader.is_encrypted = False
        mock_reader.pages = [mock_page]

        with patch("pypdf.PdfReader", return_value=mock_reader):
            resp = _upload(client, b"%PDF-1.4", "large.pdf")
        assert resp.status_code == 400
        assert "limit" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# File type validation
# ---------------------------------------------------------------------------

class TestFileTypeValidation:
    def test_wrong_extension_returns_415(self, client):
        resp = _upload(client, b"content", "document.docx")
        assert resp.status_code == 415
        detail = resp.json()["detail"]
        assert ".docx" in detail
        assert "Accepted" in detail

    def test_csv_rejected(self, client):
        resp = _upload(client, b"a,b,c\n1,2,3\n", "data.csv")
        assert resp.status_code == 415

    def test_no_extension_rejected(self, client):
        resp = _upload(client, b"content", "noextension")
        assert resp.status_code == 415

    def test_empty_file_rejected(self, client):
        resp = _upload(client, b"", "empty.md")
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Byte limit
# ---------------------------------------------------------------------------

class TestByteLimit:
    def test_oversized_upload_rejected_before_parsing(self, client):
        """A 21 MB file should be rejected at the byte limit check."""
        from omlx.admin.routes import _ATTACH_MAX_BYTES
        oversized = b"x" * (_ATTACH_MAX_BYTES + 1)
        resp = _upload(client, oversized, "huge.md")
        assert resp.status_code == 413
        assert "MB" in resp.json()["detail"]

    def test_within_byte_limit_proceeds(self, client):
        """A small file passes the byte limit and reaches content validation."""
        resp = _upload(client, b"# Hello\n", "small.md")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Response shape
# ---------------------------------------------------------------------------

class TestResponseShape:
    def test_response_fields_present(self, client):
        resp = _upload(client, b"# Doc\nContent.\n", "doc.md")
        assert resp.status_code == 200
        data = resp.json()
        assert "filename" in data
        assert "content" in data
        assert "char_count" in data
        assert "token_count" in data
        assert "token_count_exact" in data

    def test_filename_preserved(self, client):
        resp = _upload(client, b"# Hello\n", "my-notes.md")
        assert resp.json()["filename"] == "my-notes.md"

    def test_char_count_matches_content(self, client):
        text = "# Header\n\nSome content here.\n"
        resp = _upload(client, text.encode(), "doc.md")
        data = resp.json()
        assert data["char_count"] == len(data["content"])

    def test_token_count_approximate_when_no_model(self, client):
        resp = _upload(client, b"# Hello\nContent.\n", "doc.md")
        data = resp.json()
        assert data["token_count_exact"] is False
        assert data["token_count"] >= 1

    def test_token_count_exact_when_model_loaded(self, client_with_tokenizer):
        resp = _upload(client_with_tokenizer, b"# Hello\nContent.\n", "doc.md")
        data = resp.json()
        assert data["token_count_exact"] is True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_import_raiser(blocked_module):
    """Return a side_effect function that raises ImportError for one module."""
    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") \
        else __import__

    def _import(name, *args, **kwargs):
        if name == blocked_module:
            raise ImportError(f"No module named '{blocked_module}'")
        return real_import(name, *args, **kwargs)

    return _import


def _make_minimal_pdf(text: str) -> bytes:
    """Create a minimal valid PDF with one page containing the given text.

    Uses only the stdlib — no reportlab dependency required.
    The PDF is intentionally minimal and may not render in all viewers,
    but pypdf can extract text from it.
    """
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    stream_len = len(stream)

    pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R "
        b"/MediaBox [0 0 612 792] "
        b"/Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Length " + str(stream_len).encode() + b" >>\n"
        b"stream\n" + stream + b"\nendstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000266 00000 n \n"
        b"0000000360 00000 n \n"
        b"trailer\n<< /Size 6 /Root 1 0 R >>\n"
        b"startxref\n430\n%%EOF\n"
    )
    return pdf


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
