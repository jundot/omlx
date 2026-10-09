# SPDX-License-Identifier: Apache-2.0
"""Transport-level cap on request body size.

A Content-Length above the cap gets a 413 before the app reads the body.
Chunked bodies are cut off once they pass the cap.
"""

from __future__ import annotations

import logging

from starlette.requests import Headers
from starlette.responses import JSONResponse

from ..settings import get_settings

logger = logging.getLogger(__name__)

# Fits a 200 MB video sent as base64 (about 267 MB).
DEFAULT_MAX_REQUEST_BODY_BYTES = 512 * 1024 * 1024


def _resolve_limit() -> int:
    """Return the body cap, never below the upload limits users can raise."""
    try:
        settings = get_settings()
        server = settings.server
        integrations = settings.integrations
        uploads = max(
            server.max_audio_upload_bytes(),
            server.max_image_upload_bytes(),
            integrations.markitdown_max_file_size_mb
            * integrations.markitdown_max_files_per_request
            * 1024**2,
        )
        # Uploads can arrive as base64, which adds a third.
        return max(server.max_request_body_bytes(), uploads * 4 // 3 + 1024**2)
    except (RuntimeError, AttributeError, TypeError, ValueError):
        return DEFAULT_MAX_REQUEST_BODY_BYTES


class RequestBodySizeLimitMiddleware:
    """ASGI middleware enforcing ``server.max_request_body_size``."""

    def __init__(self, app, max_bytes: int | None = None):
        self.app = app
        self._fixed_limit = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        limit = self._fixed_limit if self._fixed_limit is not None else _resolve_limit()
        if limit <= 0:
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        content_length = headers.get("content-length")
        if content_length and content_length.isdigit() and int(content_length) > limit:
            logger.warning(
                "Rejected %s %s: Content-Length %s exceeds limit %d",
                scope.get("method"),
                scope.get("path"),
                content_length,
                limit,
            )
            response = JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "message": (
                            "Request body exceeds the maximum allowed size "
                            f"of {limit} bytes."
                        ),
                        "type": "request_too_large",
                    }
                },
            )
            await response(scope, receive, send)
            return

        total = 0
        overflowed = False

        async def counting_receive():
            nonlocal total, overflowed
            message = await receive()
            if overflowed:
                return {"type": "http.disconnect"}
            if message.get("type") == "http.request":
                total += len(message.get("body", b""))
                if total > limit:
                    overflowed = True
                    logger.warning(
                        "Chunked request body on %s %s exceeded limit %d; "
                        "truncating",
                        scope.get("method"),
                        scope.get("path"),
                        limit,
                    )
                    return {"type": "http.disconnect"}
            return message

        await self.app(scope, counting_receive, send)
