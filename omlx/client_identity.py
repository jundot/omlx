# SPDX-License-Identifier: Apache-2.0
"""Which client made the current request, as a short label for usage history.

Each request carries two labels: the API key that authenticated it (the main
key, a named sub key, or none when no key was checked) and its peer IP. Only
these labels cross into usage history; never the key itself or request content.
"""

import ipaddress
from contextvars import ContextVar

MAIN_KEY = "main_key"
SUB_KEY = "sub_key"
NO_KEY = "none"
KEY_KINDS = (MAIN_KEY, SUB_KEY, NO_KEY)
MAX_LABEL_LENGTH = 256


class _ClientSlot:
    """Mutable per-request holder. The middleware creates it; auth fills it in.

    Holding one object (rather than setting the ContextVar from the auth
    dependency) keeps the identity visible to tasks Starlette spawns with a
    copied context, such as streaming response bodies.
    """

    __slots__ = ("key_kind", "key_label", "ip")

    def __init__(self, ip: str):
        self.key_kind = NO_KEY
        self.key_label = ""
        self.ip = ip


_current: ContextVar[_ClientSlot | None] = ContextVar(
    "omlx_client_identity", default=None
)


def _peer_label(scope) -> str:
    client = scope.get("client")
    host = client[0] if client else ""
    if not host:
        return "unknown"
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return host[:MAX_LABEL_LENGTH]
    # Dual-stack listeners report IPv4 peers as ::ffff:a.b.c.d.
    mapped = getattr(address, "ipv4_mapped", None)
    return str(mapped or address)


class ClientIdentityMiddleware:
    """Seed each HTTP/WebSocket request with its peer IP.

    The proxy-supplied X-Forwarded-For header is deliberately ignored: any
    client can set it, so behind a reverse proxy requests group under the
    proxy's address unless they authenticate with distinct sub keys.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        token = _current.set(_ClientSlot(_peer_label(scope)))
        try:
            await self.app(scope, receive, send)
        finally:
            _current.reset(token)


def set_key_identity(kind: str, label: str) -> None:
    """Attribute the current request to the API key that authenticated it."""
    slot = _current.get()
    if slot is not None and kind in (MAIN_KEY, SUB_KEY):
        slot.key_kind = kind
        slot.key_label = label[:MAX_LABEL_LENGTH]


def current_client() -> tuple[str, str, str] | None:
    """``(key_kind, key_label, ip)`` for the current request, or None outside one."""
    slot = _current.get()
    return None if slot is None else (slot.key_kind, slot.key_label, slot.ip)
