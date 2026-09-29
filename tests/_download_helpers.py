# SPDX-License-Identifier: Apache-2.0
"""Scaffolding shared by the HuggingFace and ModelScope downloader tests.

Both backends persist the same queue file and hand the same credential to
their run coroutine, so the on-disk row helpers and the neutralised run body
live here instead of being copied per test module.
"""

import json
from pathlib import Path


def _write_rows(path: Path, rows) -> None:
    """Write a persisted queue file the way a previous boot left it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")


def _read_rows(path: Path) -> list:
    """Read back the persisted queue rows."""
    return json.loads(path.read_text(encoding="utf-8"))


async def _noop(self, task_id, token):
    """A stand-in _run_download: it takes the row and does nothing."""
    return None
