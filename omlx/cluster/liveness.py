# SPDX-License-Identifier: Apache-2.0
"""Notice when a peer Mac goes away, instead of waiting forever.

A collective has no timeout. If a peer sleeps, loses its cable, or its rank
dies, every surviving rank blocks inside the all-reduce and the deployment
hangs until someone kills it by hand — which is exactly what happened when a
Thunderbolt cable was unplugged mid-session.

Ranks already publish a runtime marker with a heartbeat timestamp on each
update. This watches those markers plus the peers themselves, so a vanished
Mac becomes a stated failure with a reason rather than a stall.

Two limits are structural and worth stating rather than papering over.

**A marker is a rank-local file.** The coordinator reads remote markers through
a fixed SSH command, so runtime health means both the Mac and the exact worker
process are alive. Before launch there is no marker yet and reachability alone
is reported as ``status == "unknown"``.

**A marker outlives the rank that wrote it.** Only a clean exit removes one, so
a crash leaves debris that the next activation of the same deployment id reads
as a live-but-silent rank. A marker whose writing process is gone is treated as
absent.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ssh_policy import cluster_ssh_options

# A rank refreshes its marker on every phase change, while serving, and on a
# fixed heartbeat interval even when idle (``RuntimeTelemetry``). Three missed
# intervals distinguishes "busy with a long prefill" from "gone".
#
# Before the idle heartbeat existed this number was a self-destruct timer: a
# healthy cluster that had simply not been asked anything for 45 seconds looked
# exactly like a wedged one.
_DEFAULT_STALE_AFTER = 45.0
_DEFAULT_PROBE_TIMEOUT = 5.0
_MAX_REMOTE_MARKER_BYTES = 256 * 1024

_LOOPBACK_TARGETS = {"127.0.0.1", "localhost", "::1"}


@dataclass(frozen=True)
class PeerHealth:
    """What we last knew about one rank.

    Before launch, ``healthy`` means the host is reachable and ``status`` may
    be ``unknown`` because no worker is expected. During a deployment,
    ``heartbeat_required`` makes health fail closed: the rank marker must be
    visible, fresh, and owned by a live process.
    """

    node_id: str
    rank: int
    reachable: bool
    seconds_since_heartbeat: float | None
    phase: str = ""
    detail: str = ""
    heartbeat_required: bool = False
    process_live: bool | None = None

    @property
    def healthy(self) -> bool:
        if not self.reachable or self.stale:
            return False
        if not self.heartbeat_required:
            return True
        return self.seconds_since_heartbeat is not None and self.process_live is not False

    @property
    def stale(self) -> bool:
        age = self.seconds_since_heartbeat
        return age is not None and age > _DEFAULT_STALE_AFTER

    @property
    def status(self) -> str:
        """``lost`` | ``stale`` | ``unknown`` | ``healthy``.

        ``unknown`` is permitted only before a deployment starts, when SSH
        reachability is the fact being checked. Once ``heartbeat_required`` is
        true, a missing marker is ``missing`` and an exited marker owner is
        ``dead``; neither is healthy.
        """

        if not self.reachable:
            return "lost"
        if self.process_live is False:
            return "dead"
        if self.stale:
            return "stale"
        if self.seconds_since_heartbeat is None:
            return "missing" if self.heartbeat_required else "unknown"
        return "healthy"

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "rank": self.rank,
            "reachable": self.reachable,
            "seconds_since_heartbeat": self.seconds_since_heartbeat,
            "phase": self.phase,
            "healthy": self.healthy,
            "status": self.status,
            "stale": self.stale,
            "heartbeat_required": self.heartbeat_required,
            "process_live": self.process_live,
            "detail": self.detail,
        }


class PeerLostError(RuntimeError):
    """A rank this deployment depends on is no longer answering."""


def probe_peer(
    ssh_target: str,
    *,
    timeout: float = _DEFAULT_PROBE_TIMEOUT,
    runner: Callable[..., Any] = subprocess.run,
) -> bool:
    """Is the peer answering at all? A cheap SSH round trip, not a model call."""

    if ssh_target in _LOOPBACK_TARGETS:
        return True
    try:
        result = runner(
            [
                "ssh",
                *cluster_ssh_options(connect_timeout=timeout),
                ssh_target,
                "true",
            ],
            capture_output=True,
            timeout=timeout + 2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return getattr(result, "returncode", 1) == 0


def read_marker(path: Path) -> dict[str, Any] | None:
    """A rank's runtime marker, or None when it is absent or unreadable."""

    try:
        if not path.is_file():
            return None
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


# Runs on the peer, but is authored here: the coordinator injects this script
# on every query, so the payload producer and consumer are always the same
# build. ``peer_now`` is the peer's own clock, read in the same process that
# read the marker — ages computed against it never cross two Macs' wall
# clocks, which is what made an unsynchronized pair look stale.
_REMOTE_MARKER_SCRIPT = (
    "import json,os,sys,time;"
    "from pathlib import Path;"
    "p=Path(sys.argv[1]).expanduser();"
    "d=json.loads(p.read_text());"
    "pid=d.get('pid');"
    "live=None;"
    "\nif isinstance(pid,int) and not isinstance(pid,bool) and pid>0:"
    "\n try: os.kill(pid,0); live=True"
    "\n except ProcessLookupError: live=False"
    "\n except PermissionError: live=True"
    "\nprint(json.dumps({'marker':d,'process_live':live,'peer_now':time.time()},"
    "separators=(',',':')))"
)


def read_remote_marker(
    ssh_target: str,
    path: str,
    *,
    timeout: float = _DEFAULT_PROBE_TIMEOUT,
    runner: Callable[..., Any] = subprocess.run,
) -> tuple[dict[str, Any] | None, bool | None, float | None, str]:
    """Read one rank marker, its owner status and the peer's clock, over SSH.

    The command is fixed; only the SSH target and a shell-quoted path vary.
    Reading the marker through SSH keeps process health tied to the rank rather
    than to the host's SSH daemon. A Mac can answer SSH after jetsam or SIGKILL
    has removed its worker.
    """

    command = " ".join(
        (
            "python3",
            "-c",
            shlex.quote(_REMOTE_MARKER_SCRIPT),
            shlex.quote(path),
        )
    )
    try:
        result = runner(
            [
                "ssh",
                *cluster_ssh_options(connect_timeout=timeout),
                ssh_target,
                command,
            ],
            capture_output=True,
            timeout=timeout + 2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, None, None, str(exc)
    if getattr(result, "returncode", 1) != 0:
        error = getattr(result, "stderr", b"")
        if isinstance(error, bytes):
            error = error.decode(errors="replace")
        return None, None, None, str(error).strip()
    raw = getattr(result, "stdout", b"")
    if isinstance(raw, str):
        raw = raw.encode()
    if len(raw) > _MAX_REMOTE_MARKER_BYTES:
        return None, None, None, "runtime marker response was too large"
    try:
        payload = json.loads(raw.decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, None, None, f"runtime marker response was invalid: {exc}"
    marker = payload.get("marker")
    process_live = payload.get("process_live")
    peer_now = payload.get("peer_now")
    if not isinstance(marker, dict):
        return None, None, None, "runtime marker response did not contain a marker"
    if process_live not in (True, False, None):
        process_live = None
    if isinstance(peer_now, bool) or not isinstance(peer_now, (int, float)):
        # The script above always emits it; a payload without it is malformed,
        # and quietly substituting the local clock would revive the cross-Mac
        # comparison this field exists to remove.
        return None, None, None, "runtime marker response did not carry the peer clock"
    return marker, process_live, float(peer_now), ""


def _pid_is_live(pid: int) -> bool:
    """Is that process still there? Mirrors ``runtime._process_is_live``.

    Kept local rather than imported so this module stays usable on a
    worker-only install, where ``runtime`` pulls in more than a rank has.
    """

    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OverflowError):
        return True
    except (TypeError, ValueError):
        return False
    return True


def marker_owner_is_live(marker: dict[str, Any]) -> bool:
    """Is the rank that wrote this marker still running on this Mac?

    A marker is removed only on a clean exit. SIGKILL, the OOM reaper, a panic
    and a power cut all leave it behind, and the deployment id is deterministic
    from the model and the plan — so the next activation reads the corpse of the
    last one, calls it stale, and refuses to start with advice ("deactivate and
    activate again") that cannot possibly clear it. There is no reaper anywhere.

    An unreadable or absent pid is treated as live: refusing to start is the
    safe answer when we cannot tell, and only a positively-dead pid should
    unblock a launch.
    """

    pid = marker.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return True
    return _pid_is_live(pid)


def marker_age_seconds(marker: dict[str, Any], *, now: float | None = None) -> float | None:
    """Seconds since a rank last refreshed its marker."""

    from datetime import UTC, datetime

    updated = marker.get("updated_at")
    if not isinstance(updated, str):
        return None
    try:
        stamp = datetime.fromisoformat(updated)
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    current = now if now is not None else datetime.now(UTC).timestamp()
    return max(0.0, current - stamp.timestamp())


def check_peers(
    hosts_by_rank: dict[int, tuple[str, str]],
    *,
    state_dir: str = "~/.omlx/cluster/runtime",
    deployment_id: str = "",
    now: float | None = None,
    probe: Callable[[str], bool] = probe_peer,
    remote_reader: Callable[
        [str, str], tuple[dict[str, Any] | None, bool | None, float | None, str]
    ] = read_remote_marker,
    require_heartbeat: bool = False,
) -> tuple[PeerHealth, ...]:
    """Health of every rank: reachable, and how fresh its heartbeat is.

    ``hosts_by_rank`` maps rank to ``(node_id, ssh_target)``. Before launch,
    callers leave ``require_heartbeat`` false and this is only an SSH
    reachability check. During a deployment it is true: local markers are read
    locally and remote markers are read on their owning Mac through SSH.

    A remote marker's age is computed against the clock returned by the same
    query, so both timestamps come from the peer. Comparing a peer's marker to
    this Mac's clock made an unsynchronized but healthy pair read as stale,
    and the watchdog then shut the deployment down.

    A marker whose writing process is gone is treated as absent rather than
    stale. It is the debris of a crashed run, not evidence about this one, and
    calling it stale wedged every subsequent activation of the same model.
    """

    local_root = Path(state_dir).expanduser()
    remote_root = Path(state_dir)
    health = []
    for rank, (node_id, ssh_target) in sorted(hosts_by_rank.items()):
        reachable = probe(ssh_target)
        marker: dict[str, Any] | None = None
        process_live: bool | None = None
        marker_error = ""
        marker_clock = now
        if reachable and require_heartbeat and deployment_id:
            name = f"{deployment_id}-rank-{rank}.json"
            if ssh_target in _LOOPBACK_TARGETS:
                marker = read_marker(local_root / name)
                process_live = marker_owner_is_live(marker) if marker else None
            else:
                marker, process_live, marker_clock, marker_error = remote_reader(
                    ssh_target, str(remote_root / name)
                )
        age = marker_age_seconds(marker, now=marker_clock) if marker else None
        if not reachable:
            detail = f"{ssh_target} did not answer"
        elif require_heartbeat and process_live is False:
            detail = f"rank {rank} is no longer running on {ssh_target}"
        elif require_heartbeat and marker is None:
            suffix = f": {marker_error}" if marker_error else ""
            detail = f"rank {rank} has no observable runtime heartbeat{suffix}"
        else:
            detail = ""
        health.append(
            PeerHealth(
                node_id=node_id,
                rank=rank,
                reachable=reachable,
                seconds_since_heartbeat=age,
                phase=str(marker.get("phase", "")) if marker else "",
                detail=detail,
                heartbeat_required=require_heartbeat,
                process_live=process_live,
            )
        )
    return tuple(health)


def describe_failure(health: tuple[PeerHealth, ...]) -> str:
    """One actionable sentence about what went wrong with the cluster."""

    unreachable = [h for h in health if not h.reachable]
    dead = [h for h in health if h.process_live is False]
    stale = [h for h in health if h.reachable and h.stale]
    missing = [
        h
        for h in health
        if h.reachable
        and h.heartbeat_required
        and h.seconds_since_heartbeat is None
        and h.process_live is not False
    ]
    if unreachable:
        names = ", ".join(h.node_id or f"rank {h.rank}" for h in unreachable)
        return (
            f"Lost contact with {names}. Check the Mac is awake and its cable "
            f"is connected, then activate the cluster again."
        )
    if dead:
        names = ", ".join(h.node_id or f"rank {h.rank}" for h in dead)
        return (
            f"{names} worker exited while its Mac remained reachable. oMLX "
            f"stopped the cluster so surviving ranks do not hang."
        )
    if stale:
        names = ", ".join(h.node_id or f"rank {h.rank}" for h in stale)
        return (
            f"{names} stopped reporting progress. The rank may have crashed or "
            f"be stuck; deactivate and activate the cluster again."
        )
    if missing:
        names = ", ".join(h.node_id or f"rank {h.rank}" for h in missing)
        return (
            f"{names} stopped publishing its runtime heartbeat. Check oMLX is "
            f"running on that Mac, then activate the cluster again."
        )
    return "All ranks are responding."


def raise_if_peer_lost(health: tuple[PeerHealth, ...]) -> None:
    """Turn a vanished peer into an error instead of an indefinite wait."""

    if any(not h.healthy for h in health):
        raise PeerLostError(describe_failure(health))


# One failed SSH round trip is not a lost Mac. The callback ends the rank
# process, so a single flaky probe during a twenty-minute weight load would
# throw the whole deployment away; two consecutive failures is 30 s of silence
# at the default interval and still well inside a human's patience.
_DEFAULT_FAILURE_TOLERANCE = 2
_DEFAULT_SERVING_INTERVAL = 3.0


class PeerWatchdog:
    """Fail a rank when a peer it depends on stops answering.

    Runs alongside the load and serving. The point is not to repair anything —
    a collective cannot proceed without every rank — but to end the wait with a
    message that names the missing Mac, so the deployment stops instead of
    hanging.

    ``hosts_by_rank`` must not contain the watching rank itself. Watching
    yourself is not a health check: the marker is on local disk and the SSH
    probe short-circuits on loopback, so the only thing it can ever detect is
    your own idleness — which it did, killing every healthy rank 60 s after the
    last token.
    """

    def __init__(
        self,
        hosts_by_rank: dict[int, tuple[str, str]],
        *,
        deployment_id: str,
        state_dir: str = "~/.omlx/cluster/runtime",
        interval: float = 15.0,
        serving_interval: float | None = None,
        on_lost: Callable[[str], None] | None = None,
        failure_tolerance: int = _DEFAULT_FAILURE_TOLERANCE,
    ) -> None:
        self._hosts = hosts_by_rank
        self._deployment_id = deployment_id
        self._state_dir = state_dir
        self._interval = max(0.0, float(interval))
        # Loading a very large model is a noisy, minutes-long cold-start lane:
        # probing it aggressively creates false failures and needless SSH
        # traffic. Once every peer reports ready, a dead cable or process is
        # request-critical and should fail in seconds instead of half a minute.
        # An explicit zero interval is retained for deterministic unit tests.
        self._serving_interval = (
            min(self._interval, _DEFAULT_SERVING_INTERVAL)
            if serving_interval is None
            else max(0.0, float(serving_interval))
        )
        self._on_lost = on_lost
        self._failure_tolerance = max(1, int(failure_tolerance))
        self._consecutive_failures: dict[int, int] = {
            rank: 0 for rank in hosts_by_rank
        }
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run_once(self) -> tuple[PeerHealth, ...]:
        return check_peers(
            self._hosts,
            state_dir=self._state_dir,
            deployment_id=self._deployment_id,
            require_heartbeat=True,
        )

    def run(self, *, sleep: Callable[[float], None] = time.sleep) -> None:
        if not self._hosts:
            # Nothing to watch. A rank alone in its own host map used to watch
            # itself; now it simply has no peers and must not fire.
            return
        poll_interval = self._interval
        while not self._stop:
            sleep(poll_interval)
            if self._stop:
                return
            health = self.run_once()
            # Change lanes only on an entirely healthy observation. A timeout
            # has no phase, and must not silently restore the long cold-start
            # cadence immediately after a ready cluster loses its cable.
            if health and all(item.healthy for item in health):
                poll_interval = (
                    self._serving_interval
                    if all(item.phase == "ready" for item in health)
                    else self._interval
                )
            by_rank = {item.rank: item for item in health}
            for rank in self._hosts:
                item = by_rank.get(rank)
                if item is not None and item.healthy:
                    self._consecutive_failures[rank] = 0
                else:
                    self._consecutive_failures[rank] = (
                        self._consecutive_failures.get(rank, 0) + 1
                    )
            failed_ranks = {
                rank
                for rank, count in self._consecutive_failures.items()
                if count >= self._failure_tolerance
            }
            if not failed_ranks:
                continue
            failed = tuple(item for item in health if item.rank in failed_ranks)
            if self._on_lost is not None:
                self._on_lost(describe_failure(failed))
            return
