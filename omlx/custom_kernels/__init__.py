# SPDX-License-Identifier: Apache-2.0
"""Optional native custom kernels bundled with oMLX."""

from __future__ import annotations

import importlib

NATIVE_KERNEL_PACKAGES = ("bonsai", "glm_moe_dsa", "minimax_m3", "qwen35_prefill")


def native_kernel_status() -> dict[str, dict[str, object]]:
    """Report availability of every optional native kernel extension.

    Source installs only compile the native extensions when built with
    ``OMLX_WITH_CUSTOM_KERNEL=1`` (which additionally needs the Metal
    toolchain). Without them the affected model families silently fall back
    to much slower generic paths, so availability is surfaced through
    ``GET /api/status`` for diagnosability instead of only a log line.

    Never raises: a package that fails to import is reported as unavailable
    with the stringified error.
    """
    status: dict[str, dict[str, object]] = {}
    for name in NATIVE_KERNEL_PACKAGES:
        try:
            fast = importlib.import_module(f"{__name__}.{name}.fast")
            available = bool(fast.is_native_available())
            error = fast.import_error()
        except Exception as exc:  # noqa: BLE001 - status must never break
            available = False
            error = exc
        status[name] = {
            "available": available,
            "import_error": str(error) if error is not None else None,
        }
    return status
