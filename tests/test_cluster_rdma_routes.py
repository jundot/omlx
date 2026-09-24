# SPDX-License-Identifier: Apache-2.0
"""The admin routes that list RDMA links and verify one on demand."""

from __future__ import annotations

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.cluster import routes
from omlx.cluster.rdma import launch_links, link_routes
from omlx.cluster.rdma.daemon import DaemonStatus, PeerStatus
from omlx.cluster.rdma.links import NodeAddress, discover_links
from omlx.cluster.rdma.store import RdmaLinkStore
from omlx.cluster.rdma.verification import (
    LinkVerification,
    ProbeMeasurements,
    link_identity,
)

_NODES = (
    NodeAddress(
        "spark-a", "worker@10.0.0.2", ("10.0.0.2",), python_executable="/opt/py"
    ),
)
_STATUS = DaemonStatus(
    "/tmp/mcdma-rpcd.sock",
    True,
    "1 of 1 peer links up",
    "1.0.0",
    (
        PeerStatus(
            "linka",
            True,
            host="10.0.0.2",
            device="rdma_mcrdma0",
            request_bytes=4 << 20,
            reply_bytes=4 << 20,
            since=5.0,
        ),
    ),
)
_MEASURED = ProbeMeasurements(
    200, 21.5, 40.0, 3 * (64 << 20), 28.1, 3 * (64 << 20), 47.9
)


@pytest.fixture
def client(monkeypatch, tmp_path):
    store = RdmaLinkStore(tmp_path)
    monkeypatch.setattr(link_routes, "read_status", lambda: _STATUS)
    monkeypatch.setattr(link_routes, "enrolled_node_addresses", lambda: _NODES)
    monkeypatch.setattr(link_routes, "_store", lambda: store)
    monkeypatch.setattr(link_routes, "cached_driver_identity", lambda: None)
    monkeypatch.setattr(link_routes, "read_driver_identity", lambda: None)
    monkeypatch.setattr(
        link_routes, "load_word_ops", lambda: (None, "libmcdma-rpc is not installed")
    )
    launch_links.release_links("cluster-rdma")
    app = FastAPI()
    app.include_router(routes.router)
    yield TestClient(app), store
    launch_links.release_links("cluster-rdma")
    launch_links.release_links(launch_links.VERIFYING)


def _evidence(verified: bool = True) -> LinkVerification:
    (link,) = discover_links(_STATUS, _NODES)
    return LinkVerification(
        "linka",
        "spark-a",
        verified,
        "",
        time.time(),
        link_identity(link, _STATUS, None),
        _MEASURED,
    )


def test_inventory_lists_each_link_with_its_evidence(client):
    http, store = client
    store.record(_evidence())
    body = http.get("/admin/api/cluster/rdma-links").json()
    (row,) = body["links"]
    assert (row["name"], row["peer_node_id"], row["usable"]) == (
        "linka",
        "spark-a",
        True,
    )
    assert row["verified"] is True and row["stale_reason"] is None
    assert row["verification"]["measurements"]["from_peer_gbit_s"] == 47.9
    assert body["helper"] == {
        "available": False,
        "reason": "libmcdma-rpc is not installed",
    }
    assert body["daemon"]["reachable"] is True


def test_inventory_says_when_a_link_was_never_verified(client):
    http, _ = client
    (row,) = http.get("/admin/api/cluster/rdma-links").json()["links"]
    assert row["verified"] is False and row["stale_reason"] == "never verified"


def test_inventory_answers_without_a_daemon(client, monkeypatch):
    http, _ = client
    monkeypatch.setattr(
        link_routes,
        "read_status",
        lambda: DaemonStatus(
            "/tmp/mcdma-rpcd.sock", False, "mcdma-rpcd is not running"
        ),
    )
    body = http.get("/admin/api/cluster/rdma-links").json()
    assert (
        body["links"] == [] and body["daemon"]["reason"] == "mcdma-rpcd is not running"
    )


def test_verify_runs_the_full_probe_and_records_it(client, monkeypatch):
    http, store = client
    calls = []

    def verify(link, node, **kwargs):
        calls.append((link.name, node.node_id, kwargs["settings"]))
        return _evidence()

    monkeypatch.setattr(link_routes, "verify_link", verify)
    response = http.post("/admin/api/cluster/rdma-links/verify", json={"link": "linka"})
    assert response.status_code == 200 and response.json()["verified"] is True
    assert calls == [("linka", "spark-a", link_routes.FULL)]
    assert store.get("linka").measurements == _MEASURED


def test_verify_holds_the_link_while_it_probes(client, monkeypatch):
    http, _ = client
    held = []

    def verify(link, node, **kwargs):
        held.append(launch_links.claimed_links())
        return _evidence()

    monkeypatch.setattr(link_routes, "verify_link", verify)
    http.post("/admin/api/cluster/rdma-links/verify", json={"link": "linka"})
    assert held == [{"linka": launch_links.VERIFYING}]
    assert launch_links.claimed_links() == {}


def test_a_probe_that_crashes_still_frees_the_link(client, monkeypatch):
    http, _ = client

    def verify(*_args, **_kwargs):
        raise RuntimeError("probe crashed")

    monkeypatch.setattr(link_routes, "verify_link", verify)
    with pytest.raises(RuntimeError, match="probe crashed"):
        http.post("/admin/api/cluster/rdma-links/verify", json={"link": "linka"})
    assert launch_links.claimed_links() == {}


def test_verify_leaves_other_verifications_alone(client, monkeypatch):
    http, _ = client
    launch_links.claim_link("linkb", launch_links.VERIFYING)
    monkeypatch.setattr(link_routes, "verify_link", lambda *_a, **_k: _evidence())
    http.post("/admin/api/cluster/rdma-links/verify", json={"link": "linka"})
    assert launch_links.claimed_links() == {"linkb": launch_links.VERIFYING}


def test_inventory_marks_a_link_under_verification(client):
    http, _ = client
    launch_links.claim_link("linka", launch_links.VERIFYING)
    (row,) = http.get("/admin/api/cluster/rdma-links").json()["links"]
    assert row["verifying"] is True and row["in_use_by"] is None


@pytest.mark.parametrize(
    ("setup", "status_code", "detail"),
    [
        (
            lambda mp: mp.setattr(
                link_routes,
                "read_status",
                lambda: DaemonStatus("/tmp/x.sock", False, "mcdma-rpcd is not running"),
            ),
            409,
            "not running",
        ),
        (lambda mp: None, 404, "no link named linkz"),
        (
            lambda mp: launch_links.claim_link("linka", "cluster-rdma"),
            409,
            "in use by deployment cluster-rdma",
        ),
        (
            lambda mp: launch_links.claim_link("linka", launch_links.VERIFYING),
            409,
            "in use by a dashboard verification",
        ),
        (
            lambda mp: mp.setattr(link_routes, "enrolled_node_addresses", lambda: ()),
            409,
            "no enrolled node",
        ),
    ],
)
def test_verify_refuses_what_it_cannot_probe_safely(
    client, monkeypatch, setup, status_code, detail
):
    http, _ = client
    setup(monkeypatch)
    name = "linkz" if status_code == 404 else "linka"
    monkeypatch.setattr(
        link_routes, "verify_link", lambda *a, **k: pytest.fail("must not probe")
    )
    response = http.post("/admin/api/cluster/rdma-links/verify", json={"link": name})
    assert response.status_code == status_code
    assert detail in response.json()["detail"]


def test_verify_rejects_names_the_daemon_could_not_serve(client):
    http, _ = client
    assert (
        http.post(
            "/admin/api/cluster/rdma-links/verify", json={"link": "../etc"}
        ).status_code
        == 422
    )
