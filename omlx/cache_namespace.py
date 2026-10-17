# SPDX-License-Identifier: Apache-2.0
"""Authenticated, request-owned namespaces for reusable KV state.

The HTTP task supplies a verified credential identity. Request captures it
before work enters the scheduler or cache writer thread. Client headers and
request bodies cannot select another credential's namespace.
"""

from contextvars import ContextVar
import hashlib

current_cache_namespace: ContextVar[str | None] = ContextVar(
    "omlx_cache_namespace", default=None
)


def namespace_for_api_key(api_key: str) -> str:
    """Stable domain-separated identity; never retain the raw credential."""
    if not isinstance(api_key, str) or not api_key:
        raise ValueError("cache isolation requires a verified API key")
    return hashlib.sha256(b"omlx-api-key-cache-v1\0" + api_key.encode()).hexdigest()


def namespace_keys(namespace: str | None) -> tuple | None:
    if namespace is None:
        return None
    return (("omlx-api-key-cache-v1", namespace),)


def with_namespace(keys: tuple | None, namespace: str | None) -> tuple | None:
    scope = namespace_keys(namespace)
    return (scope + (keys or ())) if scope else keys
