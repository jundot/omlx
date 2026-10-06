# SPDX-License-Identifier: Apache-2.0
"""Persist link verifications next to the enrolled-node registry, owner-only."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from contextlib import suppress
from pathlib import Path

from .verification import LinkVerification

logger = logging.getLogger(__name__)
_SCHEMA = 1


class RdmaLinkStore:
    """Thread-safe latest verification per link, written atomically with mode 0600."""

    def __init__(self, base_path: Path) -> None:
        self.path = Path(base_path) / "cluster" / "rdma-links.json"
        self._lock = threading.Lock()
        self._records: dict[str, LinkVerification] = {}
        self.load_error: str | None = None
        # A damaged file must never stop the server starting; it reads as empty and says why.
        try:
            self._load()
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            self._records = {}
            self.load_error = f"could not read {self.path.name}: {exc}"

    def _load(self) -> None:
        if not self.path.exists():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != _SCHEMA:
            raise ValueError("unsupported schema")
        links = payload.get("links", [])
        if not isinstance(links, list) or not all(
            isinstance(item, dict) for item in links
        ):
            raise ValueError("links must be a list of objects")
        records = [LinkVerification.from_dict(item) for item in links]
        self._records = {record.link: record for record in records}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": _SCHEMA,
            "links": [record.to_dict() for _, record in sorted(self._records.items())],
        }
        descriptor, temporary = tempfile.mkstemp(
            prefix=".rdma-links.", suffix=".tmp", dir=self.path.parent
        )
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temporary)

    def record(self, verification: LinkVerification) -> None:
        """Replace one link's verification; a failed write still keeps it in memory."""
        with self._lock:
            self._records[verification.link] = verification
            try:
                self._save()
            except OSError as exc:
                logger.warning(
                    "Could not save RDMA link evidence to %s: %s", self.path, exc
                )

    def get(self, link: str) -> LinkVerification | None:
        with self._lock:
            return self._records.get(link)

    def all(self) -> dict[str, LinkVerification]:
        with self._lock:
            return dict(self._records)


_configured_lock = threading.Lock()
_configured_store: RdmaLinkStore | None = None


def configure_rdma_link_store(base_path: Path) -> RdmaLinkStore:
    """Create the process-wide store under the server's base path."""
    global _configured_store
    with _configured_lock:
        _configured_store = RdmaLinkStore(base_path)
        return _configured_store


def get_rdma_link_store() -> RdmaLinkStore:
    """The process-wide store; raises when the server has not configured one."""
    with _configured_lock:
        if _configured_store is None:
            raise RuntimeError("RDMA link store is not configured")
        return _configured_store
