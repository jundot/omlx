# SPDX-License-Identifier: Apache-2.0
"""Request body size limiting middleware.

The API previously had no transport-level cap on request bodies: the
per-endpoint limits (audio uploads, image payloads, attachments) each
enforce their own budget, but a plain JSON request could carry an
arbitrarily large ``guided_grammar`` / ``json_schema`` / prompt string and
FastAPI reads and buffers the whole body before any handler code runs.
This middleware rejects oversized bodies before they reach the app.

Two enforcement paths:

- ``Content-Length`` (every mainstream HTTP client sends it): rejected with
  a clean 413 before the app sees anything.
- Chunked / no Content-Length: the wrapped ``receive`` stops delivering
  request data once the cap is exceeded and reports a disconnect, so
  app-side accumulation stays bounded. The app then fails on the truncated
  body (typically a 422/400); a lying client can also stall its own
  upload, which is its own problem.

The limit is read from settings per request so admin-side changes apply
without a restart; if settings are unavailable (embedded/library use), the
module default applies.
"""

from __future__ import annotations

import logging

from starlette.requests import Headers
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# 512 MB covers the largest legitimate payloads: a 200 MB video
# (~267 MB as base64), several max-size images, or a ~100 MB inline
# audio clip — while keeping pathological bodies out of memory.
DEFAULT_MAX_REQUEST_BODY_BYTES = 512 * 1024 * 1024


def _resolve_limit() -> int:
    """Read the configured limit; fall back to the default when unset."""
    try:
        from ..settings import get_settings

        return get_settings().server.max_request_body_bytes()
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

        limit = (
            self._fixed_limit
            if self._fixed_limit is not None
            else _resolve_limit()
        )
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
