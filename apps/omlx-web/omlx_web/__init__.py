# SPDX-License-Identifier: Apache-2.0
"""oMLX browser admin UI, mounted by omlx.server unless it runs headless."""

from .routes import WebUIHost, require_admin, router, set_host

__all__ = ["WebUIHost", "require_admin", "router", "set_host"]
