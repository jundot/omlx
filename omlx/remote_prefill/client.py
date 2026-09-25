# SPDX-License-Identifier: Apache-2.0
"""Ask the vLLM prefill server to compute a prompt's KV cache and hold it for a handoff."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from .receiver import HandoffError, HandoffTimeoutError
from .settings import RemotePrefillSettings


def request_prefill(
    settings: RemotePrefillSettings,
    tokens: list[int],
    handoff: str,
    export_from: int,
    *,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> None:
    """Return once vLLM has prefilled `tokens` and its connector holds their pages."""
    body = {
        "model": settings.model,
        "prompt": list(tokens),
        "max_tokens": 1,
        "temperature": 0.0,
        "kv_transfer_params": {
            "mcdma_handoff": {"id": handoff, "export_from": export_from}
        },
    }
    headers = {"Content-Type": "application/json"}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"
    request = urllib.request.Request(
        settings.url + "/v1/completions",
        data=json.dumps(body).encode(),
        headers=headers,
        method="POST",
    )
    try:
        with opener(request, timeout=settings.timeout_s) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:300].decode(errors="replace")
        raise HandoffError(f"vLLM answered {exc.code}: {detail}") from exc
    except (urllib.error.URLError, OSError) as exc:
        if isinstance(exc, TimeoutError) or isinstance(
            getattr(exc, "reason", None), TimeoutError
        ):
            raise HandoffTimeoutError(
                f"vLLM at {settings.url} did not answer within {settings.timeout_s:.0f} s"
            ) from exc
        raise HandoffError(f"vLLM at {settings.url} did not answer: {exc}") from exc
