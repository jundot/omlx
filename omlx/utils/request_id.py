# SPDX-License-Identifier: Apache-2.0
"""Request correlation ids shared by the HTTP layer and the engines.

A correlation id is what oMLX logs (``server.log``) and the admin stats
endpoint key an in-flight request by, so a client that can name its own id is
able to follow one request through the server instead of inferring it from
timings. The same conservative rule already guards the internal
coordinator-to-rank transport id: ASCII only, at most 128 bytes, and a charset
that is safe in a log line, an HTTP header and a scheduler dict key.
"""

from __future__ import annotations

import uuid
from typing import Any

#: Maximum length in bytes; matches the internal transport id limit.
MAX_REQUEST_ID_BYTES = 128

#: Characters allowed in a correlation id. Deliberately excludes whitespace,
#: quotes, path separators and control characters.
_REQUEST_ID_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.:"
)

#: Scope/state key the request id is published under for handlers to read.
REQUEST_ID_STATE_KEY = "omlx_request_id"


def valid_request_id(value: Any) -> str | None:
    """Return ``value`` when it is a usable correlation id, otherwise ``None``.

    Used by the engines, where an unusable value must fall back to a minted id
    rather than being trusted.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError:
        return None
    if len(encoded) > MAX_REQUEST_ID_BYTES:
        return None
    if any(character not in _REQUEST_ID_CHARS for character in value):
        return None
    return value


def new_request_id() -> str:
    """Mint a fresh correlation id (32 hex characters)."""
    return uuid.uuid4().hex


def resolve_request_id(value: Any) -> str:
    """The caller's id when well-formed, otherwise a freshly minted one."""
    return valid_request_id(value) or new_request_id()
