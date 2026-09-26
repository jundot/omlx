# SPDX-License-Identifier: Apache-2.0
"""Stage links in the deployment contract, and the per-launch decision that sets them."""

from __future__ import annotations

import base64
import json
import zlib
from dataclasses import replace

import pytest

from omlx.cluster.deployment import (
    ClusterDeployment,
    ClusterHost,
    decode_worker_contract,
    decode_worker_stage_links,
)
from omlx.cluster.planner import PipelineAssignment
from omlx.cluster.rdma import launch_links
from omlx.cluster.rdma.daemon import DaemonStatus, PeerStatus
from omlx.cluster.rdma.links import NodeAddress
from omlx.cluster.rdma.stage_plan import StageLink, links_for_rank, validate_stage_links
from omlx.cluster.rdma.verification import LinkVerification

_LINK = StageLink(1, 0, "linka", "/tmp/mcdma-rpcd.linka.sock")


def _deployment(**changes) -> ClusterDeployment:
    base = {
        "deployment_id": "cluster-rdma",
        "model": "org/model",
        "backend": "ring",
        "hosts": (
            ClusterHost("mac", "127.0.0.1", ("10.0.0.1",)),
            ClusterHost("spark-a", "worker@10.0.0.2", ("10.0.0.2",)),
        ),
        "assignments": (
            PipelineAssignment("mac", 0, 3, 8, 5, 1, 1, 16),
            PipelineAssignment("spark-a", 1, 0, 3, 3, 1, 1, 8),
        ),
        "plan_hash": "c" * 64,
    }
    return ClusterDeployment(**{**base, **changes})


def test_stage_links_ride_the_worker_contract_only():
    deployment = _deployment(stage_links=(_LINK,))
    encoded = deployment.encode_worker_plan()
    assert decode_worker_stage_links(encoded) == (_LINK,)
    assert decode_worker_contract(encoded)[0] == "c" * 64
    assert "stage_links" not in deployment.to_dict()
    assert ClusterDeployment.from_dict(deployment.to_dict()).stage_links == ()


def test_stage_links_are_not_part_of_the_deployment_identity():
    assert _deployment(stage_links=(_LINK,)) == _deployment()


def test_older_contracts_decode_with_no_stage_links():
    payload = json.loads(
        zlib.decompress(base64.urlsafe_b64decode(_deployment().encode_worker_plan()))
    )
    del payload["stage_links"]
    encoded = base64.urlsafe_b64encode(
        zlib.compress(json.dumps(payload).encode())
    ).decode()
    assert decode_worker_stage_links(encoded) == ()


@pytest.mark.parametrize(
    "raw",
    [
        [
            {
                "sender_rank": 2,
                "receiver_rank": 0,
                "link": "linka",
                "service_socket": "/tmp/s.sock",
            }
        ],
        [
            {
                "sender_rank": 1,
                "receiver_rank": 0,
                "link": "link a",
                "service_socket": "/tmp/s.sock",
            }
        ],
        [
            {
                "sender_rank": 1,
                "receiver_rank": 0,
                "link": "linka",
                "service_socket": "relative.sock",
            }
        ],
        [_LINK.to_dict(), _LINK.to_dict()],
        [
            {
                "sender_rank": 5,
                "receiver_rank": 4,
                "link": "linkz",
                "service_socket": "/tmp/s.sock",
            }
        ],
        "not a list",
    ],
)
def test_malformed_stage_links_are_refused(raw):
    with pytest.raises(ValueError):
        validate_stage_links(raw, world_size=2)


def test_each_rank_finds_its_incoming_and_outgoing_edge():
    assert links_for_rank((_LINK,), 0) == (_LINK, None)
    assert links_for_rank((_LINK,), 1) == (None, _LINK)
    assert links_for_rank((_LINK,), 2) == (None, None)


_NODES = (
    NodeAddress(
        "spark-a", "worker@10.0.0.2", ("10.0.0.2",), python_executable="/opt/py"
    ),
)


def _up(name="linka", host="10.0.0.2"):
    return DaemonStatus(
        "/tmp/x.sock",
        True,
        "1 of 1 peer links up",
        "0.1.19",
        (PeerStatus(name, True, host=host, since=7.0),),
    )


class _Store:
    def __init__(self):
        self.recorded = []

    def record(self, verification):
        self.recorded.append(verification)


def _verify(verified, reason=""):
    def verify(link, node, **kwargs):
        assert kwargs["settings"].round_trips == 50
        return LinkVerification(link.name, node.node_id, verified, reason, 1.0)

    return verify


@pytest.fixture(autouse=True)
def _no_claims():
    for owner in ("cluster-rdma", "other", launch_links.VERIFYING):
        launch_links.release_links(owner)
    yield
    for owner in ("cluster-rdma", "other", launch_links.VERIFYING):
        launch_links.release_links(owner)


def _attach(deployment=None, **overrides):
    store = _Store()
    kwargs = {
        "status_reader": _up,
        "nodes": lambda: _NODES,
        "verify": _verify(True),
        "store": lambda: store,
    }
    kwargs.update(overrides)
    result = launch_links.attach_stage_links(deployment or _deployment(), **kwargs)
    return (*result, store)


def test_a_verified_link_to_rank_one_becomes_the_stage_link():
    deployment, report, store = _attach()
    assert deployment.stage_links == (
        StageLink(1, 0, "linka", "/tmp/mcdma-rpcd.linka.sock"),
    )
    assert report["active"] and report["reason"] == "rank 1 sends to rank 0 over linka"
    assert store.recorded[0].verified
    assert launch_links.claimed_links() == {"linka": "cluster-rdma"}


def test_the_pre_launch_probe_runs_where_rank_one_runs():
    probed = []

    def verify(link, node, **kwargs):
        probed.append((node.ssh, node.python_executable))
        return LinkVerification(link.name, node.node_id, True, "", 1.0)

    rank_one = ClusterHost(
        "spark-a",
        "worker@10.0.0.9",
        ("10.0.0.9",),
        python_executable="/opt/ranks/bin/python",
    )
    deployment = _deployment(
        hosts=(ClusterHost("mac", "127.0.0.1", ("10.0.0.1",)), rank_one)
    )
    _attach(deployment, verify=verify)
    assert probed == [("worker@10.0.0.9", "/opt/ranks/bin/python")]


def test_a_failed_pre_launch_check_falls_back_to_the_ring_and_frees_the_link():
    deployment, report, store = _attach(verify=_verify(False, "no reply within 10 s"))
    assert deployment.stage_links == ()
    assert not report["active"]
    assert (
        report["reason"]
        == "link linka failed its pre-launch check: no reply within 10 s"
    )
    assert report["edges"][0]["verified"] is False
    assert launch_links.claimed_links() == {}
    assert store.recorded[0].reason == "no reply within 10 s"


def test_a_link_carrying_another_deployment_is_left_alone():
    launch_links.claim_link("linka", "other")

    def never(*args, **kwargs):
        raise AssertionError("a claimed link must not be probed")

    deployment, report, _ = _attach(verify=never)
    assert deployment.stage_links == ()
    assert report["reason"] == "link linka is in use by deployment other"


def test_a_link_under_dashboard_verification_is_left_alone():
    launch_links.claim_link("linka", launch_links.VERIFYING)
    deployment, report, _ = _attach(verify=_verify(True))
    assert deployment.stage_links == ()
    assert report["reason"] == "link linka is in use by a dashboard verification"


@pytest.mark.parametrize(
    ("overrides", "deployment_changes", "reason"),
    [
        (
            {
                "status_reader": lambda: DaemonStatus(
                    "/tmp/x.sock", False, "mcdma-rpcd is not running: no socket"
                )
            },
            {},
            "mcdma-rpcd is not running",
        ),
        ({"nodes": lambda: ()}, {}, "rank 1 (spark-a) is not an enrolled CUDA worker"),
        (
            {"status_reader": lambda: _up(host="10.9.9.9")},
            {},
            "no mcdma-rpcd link from mac reaches spark-a",
        ),
        (
            {},
            {
                "backend": "jaccl",
                "hosts": (
                    ClusterHost("mac", "127.0.0.1", ("10.0.0.1",), (None, "rdma_en1")),
                    ClusterHost(
                        "spark-a", "worker@10.0.0.2", ("10.0.0.2",), ("rdma_en1", None)
                    ),
                ),
            },
            "only multi-node TCP-ring deployments have a stage edge to move",
        ),
    ],
)
def test_launches_without_a_usable_link_say_why(overrides, deployment_changes, reason):
    deployment, report, _ = _attach(_deployment(**deployment_changes), **overrides)
    assert deployment.stage_links == ()
    assert reason in report["reason"]


def test_stored_stage_links_are_never_trusted_for_a_new_launch():
    stale = _deployment(stage_links=(_LINK,))
    deployment, report, _ = _attach(
        stale, status_reader=lambda: DaemonStatus("/tmp/x.sock", False, "down")
    )
    assert deployment.stage_links == ()


def test_operators_can_turn_stage_links_off(monkeypatch):
    monkeypatch.setenv(launch_links.DISABLE_ENV, "0")
    deployment, report, _ = _attach()
    assert deployment.stage_links == ()
    assert report["reason"] == f"disabled by {launch_links.DISABLE_ENV}"


def test_tensor_parallel_deployments_keep_mlx_collectives():
    tp = replace(
        _deployment(),
        tensor_parallel_size=2,
        assignments=(
            PipelineAssignment(
                "mac",
                0,
                0,
                8,
                5,
                1,
                1,
                16,
                tensor_parallel_rank=0,
                tensor_parallel_size=2,
            ),
            PipelineAssignment(
                "spark-a",
                1,
                0,
                8,
                3,
                1,
                1,
                8,
                tensor_parallel_rank=1,
                tensor_parallel_size=2,
            ),
        ),
    )
    deployment, report, _ = _attach(tp)
    assert deployment.stage_links == ()
    assert report["reason"] == "tensor-parallel deployments keep MLX's collectives"
