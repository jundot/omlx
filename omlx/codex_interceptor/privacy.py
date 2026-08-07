"""Privacy boundary helpers used by the transparent proxy addon.

The source interceptor can optionally archive cloud request bodies for audit
work.  oMLX deliberately does not ship that capability: this integration's
diagnostics are metadata-only, even if a stale Harness environment variable is
present in the parent process.
"""

from __future__ import annotations

from typing import Any


class AuditRecorder:
    """No-op compatibility shim that can never persist request content."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def record_bytes(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def audit_header_bytes(*_args: Any, **_kwargs: Any) -> bytes:
    """Never serialize headers into oMLX interceptor diagnostics."""

    return b""


def is_cloud_auxiliary_inference_request(method: str, host: str, path: str) -> bool:
    """Classify hosted image generation without recording its content."""

    clean_path = str(path).split("?", 1)[0]
    return (
        str(method).upper() == "POST"
        and str(host).lower() in {"chatgpt.com", "chat.openai.com"}
        and clean_path.startswith("/backend-api/codex/images/")
    )
