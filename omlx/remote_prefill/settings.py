# SPDX-License-Identifier: Apache-2.0
"""Where remote prefill sends prompts, which local model it serves, and when it is worth it."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from ..cluster.rdma.layout import valid_link_name

PREFIX = "OMLX_REMOTE_PREFILL_"


@dataclass(frozen=True)
class RemotePrefillSettings:
    """One vLLM prefill server, the oMLX model it prefills for, and its MCDMA links in rank order."""

    url: str
    model: str
    local_model: str
    links: tuple[str, ...]
    min_tokens: int = 4096
    timeout_s: float = 120.0
    checksum: bool = True
    api_key: str = ""

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] = os.environ
    ) -> RemotePrefillSettings | None:
        """Settings from OMLX_REMOTE_PREFILL_*, or None when no server is configured."""

        def get(name: str, default: str = "") -> str:
            return environ.get(PREFIX + name, default).strip()

        url = get("URL")
        if not url:
            return None
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"{PREFIX}URL must be an http or https URL")
        links = tuple(
            valid_link_name(name.strip())
            for name in get("LINKS").split(",")
            if name.strip()
        )
        if not links or not get("MODEL") or not get("FOR"):
            raise ValueError(
                f"{PREFIX}URL needs {PREFIX}MODEL, {PREFIX}FOR and {PREFIX}LINKS"
            )
        settings = cls(
            url=url.rstrip("/"),
            model=get("MODEL"),
            local_model=get("FOR"),
            links=links,
            min_tokens=int(get("MIN_TOKENS", "4096")),
            timeout_s=float(get("TIMEOUT", "120")),
            checksum=get("CHECKSUM", "1").lower() not in {"0", "false", "no", "off"},
            api_key=get("API_KEY"),
        )
        if settings.min_tokens < 1 or settings.timeout_s <= 0:
            raise ValueError(f"{PREFIX}MIN_TOKENS and {PREFIX}TIMEOUT must be positive")
        return settings
