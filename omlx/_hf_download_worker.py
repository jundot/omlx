# SPDX-License-Identifier: Apache-2.0
"""Isolated Hugging Face HTTP download worker.

The parent sends one JSON request over stdin. Keeping the token off argv and
setting ``HF_HUB_DISABLE_XET`` before importing huggingface_hub isolates the
fallback transport choice from the long-running oMLX server process.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from typing import Any

_ALLOWED_DOWNLOAD_KWARGS = {
    "endpoint",
    "etag_timeout",
    "ignore_patterns",
    "local_dir",
    "repo_id",
    "token",
}


def _download_without_xet(
    kwargs: dict[str, Any],
    download_fn: Callable[..., Any] | None = None,
) -> None:
    """Run snapshot_download after disabling xet before its first import."""
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    if download_fn is None:
        from huggingface_hub import snapshot_download

        download_fn = snapshot_download
    download_fn(**kwargs)


def _read_request() -> dict[str, Any]:
    request = json.load(sys.stdin)
    if not isinstance(request, dict) or not isinstance(request.get("kwargs"), dict):
        raise ValueError("Invalid download worker request")
    kwargs = request["kwargs"]
    unknown = set(kwargs) - _ALLOWED_DOWNLOAD_KWARGS
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unsupported download arguments: {names}")
    return kwargs


def main() -> int:
    try:
        kwargs = _read_request()
        _download_without_xet(kwargs)
    except Exception as error:  # noqa: BLE001 - report child failure to parent
        response = {
            "ok": False,
            "error_type": type(error).__name__,
            "message": str(error),
        }
        print(json.dumps(response), flush=True)
        return 1

    print(json.dumps({"ok": True}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
