# SPDX-License-Identifier: Apache-2.0
"""A vanished peer must become a stated failure, not an indefinite wait."""

import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta

import pytest

from omlx.cluster.liveness import (
    PeerHealth,
    PeerLostError,
    PeerWatchdog,
    check_peers,
    describe_failure,
    marker_age_seconds,
    marker_owner_is_live,
    probe_peer,
    raise_if_peer_lost,
    read_marker,
)

HOSTS = {0: ("test-mbp", "127.0.0.1"), 1: ("mac-studio", "Studio.local")}


def _marker(state_dir, deployment, rank, *, age_seconds=0.0, phase="ready", pid=None):
    stamp = datetime.now(UTC) - timedelta(seconds=age_seconds)
    payload = {"phase": phase, "updated_at": stamp.isoformat(), "rank": rank}
    payload["pid"] = os.getpid() if pid is None else pid
    (state_dir / f"{deployment}-rank-{rank}.json").write_text(json.dumps(payload))


def _reaped_pid() -> int:
    """A pid that certainly no longer exists: one we started and collected."""

    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


def _remote_reader(state_dir, *, peer_clock_offset=0.0):
    """Model the fixed SSH marker query without opening a real connection.

    ``peer_clock_offset`` shifts the reported peer clock relative to this
    Mac's, standing in for an unsynchronized pair.
    """

    def read(_target, path):
        marker = read_marker(state_dir / os.path.basename(path))
        live = marker_owner_is_live(marker) if marker is not None else None
        peer_now = datetime.now(UTC).timestamp() + peer_clock_offset
        return marker, live, peer_now, ""

    return read


def test_the_local_rank_needs_no_ssh():
    """Probing yourself over SSH is both slow and prone to failing on Macs."""

    def explode(*args, **kwargs):
        raise AssertionError("must not ssh to localhost")

    assert probe_peer("127.0.0.1", runner=explode) is True
    assert probe_peer("localhost", runner=explode) is True


def test_an_unreachable_peer_is_reported_not_raised(tmp_path):
    health = check_peers(HOSTS, state_dir=str(tmp_path), deployment_id="d",
                         probe=lambda t: t == "127.0.0.1")

    by_rank = {h.rank: h for h in health}
    assert by_rank[0].reachable is True
    assert by_rank[1].reachable is False
    assert "Studio.local did not answer" in by_rank[1].detail
    assert by_rank[1].healthy is False


def test_a_pulled_cable_produces_an_actionable_message(tmp_path):
    """The exact failure that needed pkill: peer gone mid-generation."""

    health = check_peers(HOSTS, state_dir=str(tmp_path), deployment_id="d",
                         probe=lambda t: t == "127.0.0.1")

    message = describe_failure(health)
    assert "mac-studio" in message
    assert "cable" in message, "must say what to physically check"

    with pytest.raises(PeerLostError, match="mac-studio"):
        raise_if_peer_lost(health)


def test_a_rank_that_stopped_reporting_is_distinguished_from_one_that_vanished(tmp_path):
    """Reachable but silent means crashed or stuck — a different remedy."""

    _marker(tmp_path, "d", 0, age_seconds=0)
    _marker(tmp_path, "d", 1, age_seconds=300)

    health = check_peers(HOSTS, state_dir=str(tmp_path), deployment_id="d",
                         probe=lambda t: True, require_heartbeat=True,
                         remote_reader=_remote_reader(tmp_path))
    by_rank = {h.rank: h for h in health}

    assert by_rank[1].reachable is True
    assert by_rank[1].stale is True
    message = describe_failure(health)
    assert "stopped reporting" in message
    assert "cable" not in message, "a reachable Mac is not a cable problem"


def test_a_busy_rank_is_not_mistaken_for_a_dead_one(tmp_path):
    """A long prefill can go quiet for a while; only sustained silence counts."""

    _marker(tmp_path, "d", 0, age_seconds=0)
    _marker(tmp_path, "d", 1, age_seconds=20)

    health = check_peers(HOSTS, state_dir=str(tmp_path), deployment_id="d",
                         probe=lambda t: True, require_heartbeat=True,
                         remote_reader=_remote_reader(tmp_path))
    assert all(h.healthy for h in health)
    assert describe_failure(health) == "All ranks are responding."
    raise_if_peer_lost(health)  # must not raise


def test_a_remote_rank_without_a_local_marker_is_not_called_stale(tmp_path):
    """Markers are local files; a peer's marker lives on the peer."""

    health = check_peers(HOSTS, state_dir=str(tmp_path), deployment_id="d",
                         probe=lambda t: True)
    remote = next(h for h in health if h.rank == 1)
    assert remote.seconds_since_heartbeat is None
    assert remote.stale is False
    assert remote.healthy is True


def test_marker_age_survives_a_missing_or_broken_timestamp():
    assert marker_age_seconds({}) is None
    assert marker_age_seconds({"updated_at": "not-a-date"}) is None
    now = datetime.now(UTC)
    age = marker_age_seconds(
        {"updated_at": (now - timedelta(seconds=10)).isoformat()},
        now=now.timestamp(),
    )
    assert 9 <= age <= 11


def test_clock_skew_between_macs_does_not_kill_a_healthy_cluster(tmp_path):
    """Ages are peer-clock only: a Thunderbolt pair has no NTP to agree on.

    The peer's clock runs ten minutes behind this Mac. Its marker is five
    seconds old by its own clock, which is the only clock that also stamped
    ``updated_at``. Judging that marker against the local clock read 605
    seconds and shut the deployment down.
    """

    _marker(tmp_path, "d", 0, age_seconds=0)
    _marker(tmp_path, "d", 1, age_seconds=605)

    health = check_peers(HOSTS, state_dir=str(tmp_path), deployment_id="d",
                         probe=lambda t: True, require_heartbeat=True,
                         remote_reader=_remote_reader(
                             tmp_path, peer_clock_offset=-600.0))

    remote = next(h for h in health if h.rank == 1)
    assert remote.seconds_since_heartbeat == pytest.approx(5.0, abs=2.0)
    assert remote.stale is False
    assert all(h.healthy for h in health)


def test_a_rank_that_is_stale_by_its_own_clock_is_still_caught(tmp_path):
    """The skew fix must not blind the watchdog to genuine silence."""

    _marker(tmp_path, "d", 0, age_seconds=0)
    _marker(tmp_path, "d", 1, age_seconds=905)

    health = check_peers(HOSTS, state_dir=str(tmp_path), deployment_id="d",
                         probe=lambda t: True, require_heartbeat=True,
                         remote_reader=_remote_reader(
                             tmp_path, peer_clock_offset=-600.0))

    remote = next(h for h in health if h.rank == 1)
    assert remote.stale is True
    assert "stopped reporting" in describe_failure(health)


def test_the_injected_marker_script_reports_the_peer_clock(tmp_path):
    """Run the exact script SSH would run, minus the SSH."""

    from omlx.cluster.liveness import _REMOTE_MARKER_SCRIPT

    _marker(tmp_path, "d", 0, age_seconds=3)
    path = tmp_path / "d-rank-0.json"

    result = subprocess.run(
        [sys.executable, "-c", _REMOTE_MARKER_SCRIPT, str(path)],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["marker"]["rank"] == 0
    assert payload["process_live"] is True  # the marker carries this test's pid
    age = payload["peer_now"] - datetime.fromisoformat(
        payload["marker"]["updated_at"]
    ).timestamp()
    assert 2.0 <= age <= 8.0


def test_a_marker_response_without_the_peer_clock_is_rejected(tmp_path):
    """A payload the injected script cannot have produced is an error."""

    from omlx.cluster.liveness import read_remote_marker

    fake = subprocess.CompletedProcess(
        args=[], returncode=0,
        stdout=b'{"marker":{"rank":1},"process_live":true}', stderr=b"")

    marker, live, peer_now, error = read_remote_marker(
        "studio.local", "/tmp/x.json", runner=lambda *a, **k: fake
    )

    assert marker is None
    assert live is None
    assert peer_now is None
    assert "peer clock" in error


def test_the_watchdog_reports_once_and_stops(tmp_path):
    """It ends the wait; it does not thrash trying to repair a collective."""

    losses = []
    watchdog = PeerWatchdog(HOSTS, deployment_id="d", state_dir=str(tmp_path),
                            interval=0.0, on_lost=losses.append)
    watchdog.run_once = lambda: (  # type: ignore[method-assign]
        PeerHealth("test-mbp", 0, True, 0.0),
        PeerHealth("mac-studio", 1, False, None, detail="gone"),
    )

    ticks = [0]

    def fake_sleep(_seconds):
        ticks[0] += 1
        if ticks[0] > 5:
            raise AssertionError("watchdog should have stopped after reporting")

    watchdog.run(sleep=fake_sleep)
    assert len(losses) == 1
    assert "mac-studio" in losses[0]


def test_a_healthy_cluster_keeps_the_watchdog_quiet(tmp_path):
    losses = []
    watchdog = PeerWatchdog(HOSTS, deployment_id="d", state_dir=str(tmp_path),
                            interval=0.0, on_lost=losses.append)
    watchdog.run_once = lambda: (  # type: ignore[method-assign]
        PeerHealth("test-mbp", 0, True, 1.0),
        PeerHealth("mac-studio", 1, True, 2.0),
    )

    calls = [0]

    def fake_sleep(_seconds):
        calls[0] += 1
        if calls[0] >= 3:
            watchdog.stop()

    watchdog.run(sleep=fake_sleep)
    assert losses == []


# ---------------------------------------------------------------------------
# A rank must not be its own peer, and a corpse must not outvote a live cluster.
# ---------------------------------------------------------------------------


def test_a_watchdog_with_no_peers_never_fires(tmp_path):
    """The self-kill in one line: watching only yourself is not a health check.

    Every rank built its peer map from *all* assignments including its own.
    Rank 0's SSH target is pinned to loopback (always "reachable") and its
    marker is on local disk, so the only fact that entry ever contributed was
    the age of its own heartbeat — which nothing refreshed while idle.
    """

    losses = []
    watchdog = PeerWatchdog({}, deployment_id="d", state_dir=str(tmp_path),
                            interval=0.0, on_lost=losses.append)

    def fake_sleep(_seconds):
        raise AssertionError("a watchdog with no peers must not even poll")

    watchdog.run(sleep=fake_sleep)
    assert losses == []


def test_one_flaky_probe_does_not_throw_a_loaded_deployment_away(tmp_path):
    """on_lost kills the rank, so it must take more than a single missed ssh.

    The watchdog is armed before the weights are read; a twenty-minute load has
    plenty of room for one SSH round trip to time out.
    """

    losses = []
    gone = PeerHealth("mac-studio", 1, False, None, detail="gone")
    back = PeerHealth("mac-studio", 1, True, 1.0)
    answers = [(gone,), (back,), (back,)]
    watchdog = PeerWatchdog(
        {1: ("mac-studio", "Studio.local")},
        deployment_id="d",
        state_dir=str(tmp_path),
        interval=0.0,
        on_lost=losses.append,
    )

    def next_answer():
        if not answers:
            watchdog.stop()
            return (back,)
        return answers.pop(0)

    watchdog.run_once = next_answer  # type: ignore[method-assign]

    ticks = [0]

    def fake_sleep(_seconds):
        ticks[0] += 1
        if ticks[0] > 6:
            raise AssertionError("watchdog never settled")

    watchdog.run(sleep=fake_sleep)
    assert losses == [], "one failed probe is not a lost Mac"


def test_a_peer_that_stays_gone_is_still_reported(tmp_path):
    losses = []
    watchdog = PeerWatchdog(
        {1: ("mac-studio", "Studio.local")},
        deployment_id="d",
        state_dir=str(tmp_path),
        interval=0.0,
        on_lost=losses.append,
    )
    watchdog.run_once = lambda: (  # type: ignore[method-assign]
        PeerHealth("mac-studio", 1, False, None, detail="gone"),
    )

    ticks = [0]

    def fake_sleep(_seconds):
        ticks[0] += 1
        if ticks[0] > 6:
            raise AssertionError("watchdog should have reported by now")

    watchdog.run(sleep=fake_sleep)
    assert len(losses) == 1
    assert "mac-studio" in losses[0]


def test_watchdog_uses_a_fast_lane_after_every_peer_is_ready(tmp_path):
    """Serving failures are request-critical; cold-start probes remain patient."""

    losses = []
    loading = PeerHealth("mac-studio", 1, True, 1.0, phase="loading")
    ready = PeerHealth("mac-studio", 1, True, 1.0, phase="ready")
    gone = PeerHealth("mac-studio", 1, False, None, detail="gone")
    answers = [(loading,), (ready,), (gone,), (gone,)]
    sleeps = []
    watchdog = PeerWatchdog(
        {1: ("mac-studio", "Studio.local")},
        deployment_id="d",
        interval=15.0,
        serving_interval=3.0,
        on_lost=losses.append,
    )
    watchdog.run_once = lambda: answers.pop(0)  # type: ignore[method-assign]

    watchdog.run(sleep=sleeps.append)

    assert sleeps == [15.0, 15.0, 3.0, 3.0]
    assert len(losses) == 1


def test_one_ready_lane_timeout_does_not_restore_the_cold_start_delay(tmp_path):
    """A missing phase is failure evidence, not a request to slow monitoring."""

    losses = []
    ready = PeerHealth("mac-studio", 1, True, 1.0, phase="ready")
    gone = PeerHealth("mac-studio", 1, False, None, detail="gone")
    answers = [(ready,), (gone,), (gone,)]
    sleeps = []
    watchdog = PeerWatchdog(
        {1: ("mac-studio", "Studio.local")},
        deployment_id="d",
        interval=15.0,
        serving_interval=2.0,
        on_lost=losses.append,
    )
    watchdog.run_once = lambda: answers.pop(0)  # type: ignore[method-assign]

    watchdog.run(sleep=sleeps.append)

    assert sleeps == [15.0, 2.0, 2.0]
    assert len(losses) == 1


def test_a_marker_left_by_a_crashed_rank_does_not_wedge_the_next_activation(tmp_path):
    """SIGKILL, jetsam, panic and power loss all skip the marker cleanup.

    The deployment id is deterministic from the model and the plan hash, so the
    next activation of the same model reads the corpse of the last one, calls it
    stale and returns 409 — advising a deactivate/activate cycle that cannot
    clear a file nothing ever removes. There is no reaper anywhere in oMLX.
    """

    _marker(tmp_path, "d", 1, age_seconds=3600, pid=_reaped_pid())

    health = check_peers(
        {1: ("mac-studio", "Studio.local")},
        state_dir=str(tmp_path),
        deployment_id="d",
        probe=lambda _target: True,
    )

    assert health[0].stale is False
    assert health[0].seconds_since_heartbeat is None
    assert health[0].status == "unknown"
    raise_if_peer_lost(health)  # must not raise


def test_a_running_rank_that_went_silent_is_still_stale(tmp_path):
    """The corpse rule must not swallow the failure it sits next to."""

    _marker(tmp_path, "d", 1, age_seconds=3600, pid=os.getpid())

    health = check_peers(
        {1: ("mac-studio", "Studio.local")},
        state_dir=str(tmp_path),
        deployment_id="d",
        probe=lambda _target: True,
        require_heartbeat=True,
        remote_reader=_remote_reader(tmp_path),
    )

    assert health[0].stale is True
    assert health[0].status == "stale"
    with pytest.raises(PeerLostError, match="stopped reporting"):
        raise_if_peer_lost(health)


def test_a_marker_without_a_pid_is_believed(tmp_path):
    """Older markers carry no pid; refusing to start is the safe unknown."""

    stamp = datetime.now(UTC) - timedelta(seconds=3600)
    (tmp_path / "d-rank-1.json").write_text(
        json.dumps({"phase": "ready", "updated_at": stamp.isoformat()})
    )

    health = check_peers(
        {1: ("mac-studio", "Studio.local")},
        state_dir=str(tmp_path),
        deployment_id="d",
        probe=lambda _target: True,
        require_heartbeat=True,
        remote_reader=_remote_reader(tmp_path),
    )

    assert marker_owner_is_live({"phase": "ready"}) is True
    assert health[0].stale is True


def test_prelaunch_reachability_does_not_require_a_runtime_heartbeat(tmp_path):
    """Before launch there is deliberately no rank heartbeat to require."""

    health = check_peers(
        {0: ("test-mbp", "127.0.0.1"), 1: ("mac-studio", "Studio.local")},
        state_dir=str(tmp_path),
        deployment_id="d",
        probe=lambda _target: True,
    )

    assert [h.status for h in health] == ["unknown", "unknown"]
    assert all(h.healthy for h in health)
    assert all(item["status"] == "unknown" for item in (h.to_dict() for h in health))


def test_a_running_deployment_fails_closed_when_remote_heartbeat_is_missing(tmp_path):
    health = check_peers(
        {1: ("mac-studio", "Studio.local")},
        state_dir=str(tmp_path),
        deployment_id="d",
        probe=lambda _target: True,
        require_heartbeat=True,
        remote_reader=lambda _target, _path: (None, None, None, "not found"),
    )

    assert health[0].status == "missing"
    assert health[0].healthy is False
    with pytest.raises(PeerLostError, match="heartbeat"):
        raise_if_peer_lost(health)


def test_a_reachable_mac_with_a_dead_worker_is_not_healthy(tmp_path):
    marker = {
        "phase": "ready",
        "updated_at": datetime.now(UTC).isoformat(),
        "pid": 999999,
    }
    health = check_peers(
        {1: ("mac-studio", "Studio.local")},
        state_dir=str(tmp_path),
        deployment_id="d",
        probe=lambda _target: True,
        require_heartbeat=True,
        remote_reader=lambda _target, _path: (marker, False, datetime.now(UTC).timestamp(), ""),
    )

    assert health[0].reachable is True
    assert health[0].status == "dead"
    assert health[0].healthy is False
    with pytest.raises(PeerLostError, match="worker exited"):
        raise_if_peer_lost(health)


def test_alternating_failures_on_different_peers_do_not_accumulate(tmp_path):
    losses = []
    studio_gone = (
        PeerHealth("studio", 1, False, None),
        PeerHealth("mini", 2, True, 1.0),
    )
    mini_gone = (
        PeerHealth("studio", 1, True, 1.0),
        PeerHealth("mini", 2, False, None),
    )
    answers = [studio_gone, mini_gone, studio_gone, mini_gone]
    watchdog = PeerWatchdog(
        {1: ("studio", "studio.local"), 2: ("mini", "mini.local")},
        deployment_id="d",
        interval=0.0,
        on_lost=losses.append,
    )

    def next_answer():
        if not answers:
            watchdog.stop()
            return (
                PeerHealth("studio", 1, True, 1.0),
                PeerHealth("mini", 2, True, 1.0),
            )
        return answers.pop(0)

    watchdog.run_once = next_answer  # type: ignore[method-assign]
    watchdog.run(sleep=lambda _seconds: None)
    assert losses == []


def test_status_names_each_way_a_rank_can_be_wrong(tmp_path):
    assert PeerHealth("a", 0, False, None).status == "lost"
    assert PeerHealth("a", 0, True, 300.0).status == "stale"
    assert PeerHealth("a", 0, True, None).status == "unknown"
    assert PeerHealth("a", 0, True, 1.0).status == "healthy"
    assert PeerHealth("a", 0, True, None, heartbeat_required=True).status == "missing"
    assert PeerHealth("a", 0, True, 1.0, process_live=False).status == "dead"
