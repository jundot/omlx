# SPDX-License-Identifier: Apache-2.0
"""Expert reads spread over several copies of one checkpoint.

MoE Expert Offload reads a missing expert from the checkpoint, so decode waits
on the disk the checkpoint is on. A second copy of the checkpoint on another
disk is a second queue: this module checks that the copies hold the same
shards and sends each read to the copy expected to finish it first.

The copies are replicas in a fixed order. Replica 0 is the model directory,
and the mirrors follow in the order they are configured.
"""

from __future__ import annotations

import os
import struct
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeVar

# How long an idle replica's throughput reading stays usable.
_MEASUREMENT_LIFETIME_NS = 1_000_000_000

_T = TypeVar("_T")


class _Replica:
    __slots__ = (
        "queued_reads",
        "queued_bytes",
        "served_bytes",
        "measured_bytes",
        "measured_ns",
        "busy_since",
        "touched_at",
    )

    def __init__(self) -> None:
        self.queued_reads = 0
        self.queued_bytes = 0
        self.served_bytes = 0
        # Completed bytes and busy time; reset when an idle reading expires.
        self.measured_bytes = 0
        self.measured_ns = 0
        # Start of the interval during which this replica has had at least one
        # read in flight. A replica that never falls idle would otherwise
        # report no elapsed time at all, so the open interval is added at the
        # point the reading is asked for.
        self.busy_since = 0
        self.touched_at = 0

    def finish_estimate(self, nbytes: int, now: int) -> float:
        """Time until a read of ``nbytes`` submitted now would finish here.

        An idle, unmeasured replica gets one probe. Until that probe returns,
        replicas with a measured throughput are preferred: more queued work is
        not a new measurement.
        """
        elapsed = self.measured_ns + (now - self.busy_since if self.queued_reads else 0)
        if self.measured_bytes == 0 or elapsed == 0:
            return 0.0 if self.queued_reads == 0 else float("inf")
        return (self.queued_bytes + nbytes) * elapsed / self.measured_bytes

    def is_stale(self, now: int) -> bool:
        """Idle for so long that the reading no longer describes the disk.

        A replica with a read in flight is never stale, however long that read
        takes: it is being measured right now, and sending more reads to a
        disk that is visibly struggling is the opposite of what the estimate
        is for.
        """
        return (
            self.queued_reads == 0 and now - self.touched_at > _MEASUREMENT_LIFETIME_NS
        )


class MirrorRouter:
    """Send each read to the replica with the earliest estimated finish.

    The estimate is (queued bytes + this read) / observed throughput.
    Throughput is completed bytes over busy time for the replica as a whole,
    so concurrent reads do not count the same queueing delay more than once.
    An idle reading expires, which lets a disk that stalled or went unused be
    tried again. A failed read releases its reservation without counting as
    served bytes.
    """

    def __init__(
        self, replica_count: int, now: Callable[[], int] = time.perf_counter_ns
    ) -> None:
        self._replicas = [_Replica() for _ in range(replica_count)]
        self._lock = threading.Lock()
        self._now = now

    def claim(self, nbytes: int) -> int:
        """Reserve a replica for one read of ``nbytes`` and return its index.

        The caller must pass that index and the same ``nbytes`` to
        :meth:`release` when the read ends, including when it fails, or the
        replica is left looking permanently busier than it is.
        """
        with self._lock:
            # Read the clock under the lock: a claim that entered with an
            # older reading than a stored ``busy_since`` would compute a
            # negative open interval.
            now = self._now()
            chosen = self._replicas[0]
            index = 0
            soonest = float("inf")
            for i, replica in enumerate(self._replicas):
                if replica.is_stale(now):
                    replica.measured_bytes = 0
                    replica.measured_ns = 0
                finish = replica.finish_estimate(nbytes, now)
                # If every replica is still being probed, balance their
                # pending bytes instead of treating them as instantly free.
                if finish < soonest or (
                    finish == soonest and replica.queued_bytes < chosen.queued_bytes
                ):
                    soonest = finish
                    chosen = replica
                    index = i
            if chosen.queued_reads == 0:
                chosen.busy_since = now
            chosen.queued_reads += 1
            chosen.queued_bytes += nbytes
            chosen.touched_at = now
            return index

    def release(self, index: int, nbytes: int, completed: bool) -> None:
        """End a read that :meth:`claim` reserved.

        A read that failed passes ``completed=False``: it occupied the
        replica, so its time counts, but bytes it never delivered must not
        count as throughput.
        """
        with self._lock:
            now = self._now()
            replica = self._replicas[index]
            replica.queued_reads -= 1
            replica.queued_bytes -= nbytes
            replica.touched_at = now
            if completed:
                replica.served_bytes += nbytes
                replica.measured_bytes += nbytes
            if replica.queued_reads == 0:
                replica.measured_ns += now - replica.busy_since
                replica.busy_since = 0

    def served_bytes(self) -> list[int]:
        """Bytes each replica has served so far, in replica order."""
        with self._lock:
            return [replica.served_bytes for replica in self._replicas]


class ShardCopies:
    """One shard's descriptor in every replica, with the router they share."""

    __slots__ = ("router", "fds")

    def __init__(self, router: MirrorRouter, primary_fd: int) -> None:
        self.router = router
        self.fds = [primary_fd]

    def read(self, nbytes: int, read_fd: Callable[[int], _T | None]) -> _T | None:
        """Run ``read_fd`` on the descriptor of the replica the router picks.

        A read that raises or returns ``None`` occupied the replica without
        delivering its bytes.
        """
        replica = self.router.claim(nbytes)
        result = None
        try:
            result = read_fd(self.fds[replica])
            return result
        finally:
            self.router.release(replica, nbytes, result is not None)


def _shard_identity(path: Path) -> tuple[int, bytes]:
    """Size plus safetensors header.

    Together they fix every tensor's name, dtype, shape and byte range in the
    shard, which is what a read plan made from one copy relies on in another.
    """
    with open(path, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        prefix = f.read(8)
        # The length is checked before it is used: a file that is not
        # safetensors can declare a header of any size.
        if len(prefix) < 8 or 8 + struct.unpack("<Q", prefix)[0] > size:
            raise ValueError(f"{path} is not a complete safetensors file")
        return size, prefix + f.read(struct.unpack("<Q", prefix)[0])


def require_identical_mirrors(
    model_dir: str | Path, mirrors: Sequence[str | Path]
) -> None:
    """Refuse mirrors that cannot stand in for ``model_dir``.

    Each mirror must hold every shard of the model directory with the same
    size and safetensors header. The tensor bytes are not compared: hashing
    tens of gigabytes at every load is the wrong place for that check, so a
    mirror has to be made with a copy that verifies them.
    """
    model_dir = Path(model_dir)
    seen = [model_dir]
    for mirror in map(Path, mirrors):
        if not mirror.is_dir():
            raise ValueError(f"mirror {mirror} is not a directory")
        if any(mirror.samefile(other) for other in seen):
            raise ValueError(
                f"mirror {mirror} is listed twice or is the model directory "
                f"{model_dir}; a mirror must be another copy on another disk"
            )
        seen.append(mirror)
    for shard in sorted(model_dir.glob("*.safetensors")):
        expected = _shard_identity(shard)
        for mirror in seen[1:]:
            try:
                found = _shard_identity(mirror / shard.name)
            except (OSError, ValueError) as error:
                raise ValueError(
                    f"mirror {mirror} cannot serve {shard.name}: {error}. "
                    f"A mirror must hold the same shards as {model_dir}"
                ) from error
            if found != expected:
                raise ValueError(
                    f"mirror {mirror} holds a different {shard.name} than "
                    f"{model_dir}. Every mirror must be a byte-identical copy "
                    "of the model directory"
                )
