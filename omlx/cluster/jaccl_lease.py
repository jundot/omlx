# SPDX-License-Identifier: Apache-2.0
"""Per-Mac process lease for the Thunderbolt JACCL communicator."""

from __future__ import annotations

import fcntl
import json
import os
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

_MAX_OWNER_BYTES = 4096


class JacclCommunicatorBusyError(RuntimeError):
    """Another live process owns this Mac's RDMA communicator."""


class JacclCommunicatorLease:
    """An advisory lock held for one rank process's complete lifetime."""

    def __init__(self, descriptor: int, path: Path) -> None:
        self._descriptor = descriptor
        self.path = path

    @property
    def active(self) -> bool:
        return self._descriptor >= 0

    def close(self) -> None:
        descriptor, self._descriptor = self._descriptor, -1
        if descriptor < 0:
            return
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def __enter__(self) -> JacclCommunicatorLease:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


def acquire_jaccl_communicator_lease(
    *,
    deployment_id: str,
    state_dir: str | Path = "~/.omlx/cluster/runtime",
) -> JacclCommunicatorLease:
    """Acquire the one-communicator-per-Mac safety contract.

    The lock is kernel-owned and releases when a rank crashes or is killed;
    the small JSON body is diagnostics only and is never trusted as liveness.
    """

    root = Path(state_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "jaccl-communicator.lock"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            owner = ""
            try:
                os.lseek(descriptor, 0, os.SEEK_SET)
                owner = os.read(descriptor, _MAX_OWNER_BYTES).decode(
                    "utf-8", errors="replace"
                )
                parsed = json.loads(owner)
                owner = (
                    f"deployment {parsed.get('deployment_id', 'unknown')} "
                    f"(pid {parsed.get('pid', 'unknown')})"
                )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                owner = "another live rank process"
            raise JacclCommunicatorBusyError(
                "Thunderbolt RDMA is already owned by "
                f"{owner or 'another live rank process'}; oMLX permits one "
                "JACCL communicator per Mac until the multi-communicator "
                "lost-completion soak passes"
            ) from exc

        payload = json.dumps(
            {
                "schema_version": 1,
                "deployment_id": str(deployment_id)[:128],
                "pid": os.getpid(),
                "acquired_at": time.time(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.write(descriptor, payload)
        os.fsync(descriptor)
        return JacclCommunicatorLease(descriptor, path)
    except BaseException:
        os.close(descriptor)
        raise


__all__ = [
    "JacclCommunicatorBusyError",
    "JacclCommunicatorLease",
    "acquire_jaccl_communicator_lease",
]
