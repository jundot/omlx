"""Transparent, process-scoped Codex routing to the local oMLX server.

The package intentionally does not modify Codex configuration.  A managed
loopback proxy rewrites only selected Responses inference traffic while every
other Codex request keeps its original destination and credentials.
"""

from .manager import CodexInterceptorManager, get_codex_interceptor_manager

__all__ = ["CodexInterceptorManager", "get_codex_interceptor_manager"]
