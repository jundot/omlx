# SPDX-License-Identifier: Apache-2.0
"""Link verification evidence: what invalidates it, how it persists, and the driver identity."""

from __future__ import annotations

import json
import stat
import subprocess
from types import SimpleNamespace

import pytest

from omlx.cluster.rdma.daemon import DaemonStatus
from omlx.cluster.rdma.links import RdmaLink
from omlx.cluster.rdma.store import RdmaLinkStore
from omlx.cluster.rdma.verification import (
    DriverIdentity,
    LinkVerification,
    ProbeMeasurements,
    link_identity,
    read_driver_identity,
)

_MEASURED = ProbeMeasurements(200, 11.5, 19.0, 3 << 26, 29.1, 3 << 26, 49.8)
_LINK = RdmaLink(
    "linka",
    True,
    "",
    "10.0.0.2",
    "spark-a",
    "rdma_mcrdma1",
    4 << 20,
    64 << 20,
    1790000000.0,
)
_STATUS = DaemonStatus("/tmp/x.sock", True, "", "0.1.19")


def _evidence(**changes):
    base = {
        "link": "linka",
        "peer_node_id": "spark-a",
        "verified": True,
        "reason": "",
        "checked_at": 1000.0,
        "identity": link_identity(_LINK, _STATUS, DriverIdentity("0.1.18", "0B30")),
        "measurements": _MEASURED,
    }
    return LinkVerification(**{**base, **changes})


def test_fresh_evidence_with_the_same_identity_holds():
    evidence = _evidence()
    assert evidence.stale_reason(dict(evidence.identity), now=1500.0) is None


def test_evidence_expires_after_a_day():
    evidence = _evidence()
    assert "more than a day old" in evidence.stale_reason(
        dict(evidence.identity), now=1000.0 + 90_000
    )


def test_a_reconnect_or_driver_change_invalidates_evidence():
    evidence = _evidence()
    reconnected = link_identity(
        RdmaLink(
            "linka",
            True,
            "",
            "10.0.0.2",
            "spark-a",
            "rdma_mcrdma1",
            4 << 20,
            64 << 20,
            1790009999.0,
        ),
        _STATUS,
        DriverIdentity("0.1.18", "0B30"),
    )
    assert (
        evidence.stale_reason(reconnected, now=1001.0)
        == "the link changed since it was verified: connected_since"
    )
    new_driver = link_identity(_LINK, _STATUS, DriverIdentity("0.1.19", "AAAA"))
    assert "driver_uuid, driver_version" in evidence.stale_reason(
        new_driver, now=1001.0
    )


def test_failed_evidence_never_holds():
    evidence = _evidence(verified=False, reason="no reply within 10 s")
    assert (
        evidence.stale_reason(dict(evidence.identity), now=1001.0)
        == "no reply within 10 s"
    )


def test_evidence_round_trips_through_its_dict():
    evidence = _evidence()
    assert (
        LinkVerification.from_dict(json.loads(json.dumps(evidence.to_dict())))
        == evidence
    )


def test_the_store_persists_owner_only_and_reloads(tmp_path):
    store = RdmaLinkStore(tmp_path)
    store.record(_evidence())
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    reloaded = RdmaLinkStore(tmp_path)
    assert reloaded.get("linka") == _evidence()
    assert reloaded.load_error is None


def test_a_corrupt_store_starts_empty_and_says_so(tmp_path):
    path = tmp_path / "cluster" / "rdma-links.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    store = RdmaLinkStore(tmp_path)
    assert store.all() == {}
    assert "could not read rdma-links.json" in store.load_error


@pytest.mark.parametrize(
    "payload",
    [
        '{"schema_version": 1, "links": [null]}',
        '{"schema_version": 1, "links": [{"link": "linka"}]}',
        '{"schema_version": 1, "links": "linka"}',
        "[]",
    ],
)
def test_a_malformed_store_never_stops_the_server_starting(tmp_path, payload):
    path = tmp_path / "cluster" / "rdma-links.json"
    path.parent.mkdir()
    path.write_text(payload)
    store = RdmaLinkStore(tmp_path)
    assert store.all() == {}
    assert store.load_error.startswith("could not read rdma-links.json")


def test_the_loaded_driver_is_read_from_kmutil():
    def run(argv, **kwargs):
        assert argv[:2] == ["/usr/bin/kmutil", "showloaded"]
        line = "  252 0 0 0xd73 0xd73 org.mcdma.cx5.native (0.1.18) 1a2b3c4d-0000-4000-8000-00000000abcd <219 73>"
        return SimpleNamespace(stdout=line)

    assert read_driver_identity(run) == DriverIdentity(
        "0.1.18", "1A2B3C4D-0000-4000-8000-00000000ABCD"
    )


def test_an_unloaded_driver_or_missing_kmutil_reads_as_none():
    assert (
        read_driver_identity(lambda argv, **kwargs: SimpleNamespace(stdout="")) is None
    )

    def missing(argv, **kwargs):
        raise FileNotFoundError(argv[0])

    assert read_driver_identity(missing) is None

    def slow(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, 10)

    assert read_driver_identity(slow) is None
