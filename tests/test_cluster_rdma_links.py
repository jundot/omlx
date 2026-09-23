# SPDX-License-Identifier: Apache-2.0
"""Daemon status parsing, and matching the daemon's peers to enrolled nodes."""

from __future__ import annotations

import os
import socket
import tempfile
import threading

from omlx.cluster.rdma.daemon import DaemonStatus, PeerStatus, parse_status, read_status
from omlx.cluster.rdma.links import NodeAddress, discover_links

_V1 = [
    "VERSION mcdma-rpcd 1 0.1.19",
    "PEER linka up calls 12 failures 0 MiB 3 host=10.0.0.2 port=18555 device=rdma_mcrdma1 req_mib=4 rep_mib=64 since=1790000000",
    "PEER linkb down calls 0 failures 2 MiB 0 host=10.0.0.3 port=18555 device=rdma_mcrdma0 req_mib=4 rep_mib=4 since=0",
    "END",
]


def test_protocol_one_status_carries_host_device_and_sizes():
    version, peers = parse_status(_V1)
    assert version == "0.1.19"
    first = peers[0]
    assert (first.name, first.up, first.calls, first.mib) == ("linka", True, 12, 3)
    assert (first.host, first.port, first.device) == ("10.0.0.2", 18555, "rdma_mcrdma1")
    assert (first.request_bytes, first.reply_bytes, first.since) == (
        4 << 20,
        64 << 20,
        1790000000.0,
    )
    assert peers[1].up is False and peers[1].failures == 2


def test_lab_daemon_status_lines_still_parse_without_extras():
    version, peers = parse_status(["PEER link1s up calls 3 failures 0 MiB 1", "END"])
    assert version is None
    assert peers == (PeerStatus("link1s", True, 3, 0, 1),)


def _serve(lines):
    directory = tempfile.mkdtemp(prefix="rpcd-", dir="/tmp")
    path = os.path.join(directory, "c.sock")
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(path)
    listener.listen(1)

    def answer():
        connection, _ = listener.accept()
        with connection:
            assert connection.recv(64) == b"STATUS\n"
            connection.sendall(("\n".join(lines) + "\n").encode())
        listener.close()

    threading.Thread(target=answer, daemon=True).start()
    return path


def test_status_is_read_over_the_control_socket():
    status = read_status(_serve(_V1))
    assert status.reachable and status.version == "0.1.19"
    assert status.reason == "1 of 2 peer links up"
    assert status.peer("linkb").up is False


def test_a_daemon_speaking_another_protocol_is_not_used():
    lines = [
        "VERSION mcdma-rpcd 2 2.0.0",
        *(line for line in _V1 if line.startswith("PEER")),
        "END",
    ]
    status = read_status("/tmp/x.sock", ask=lambda *_a: lines)
    assert not status.reachable
    assert (
        status.reason
        == "mcdma-rpcd at /tmp/x.sock speaks protocol 2; oMLX needs protocol 1"
    )


def test_an_absent_daemon_is_a_reason_not_an_error():
    # Unix socket paths are short on macOS, so this cannot live under pytest's tmp_path.
    status = read_status(
        os.path.join(tempfile.mkdtemp(prefix="rpcd-", dir="/tmp"), "missing.sock")
    )
    assert not status.reachable
    assert "mcdma-rpcd is not running" in status.reason


def test_a_daemon_with_no_peers_is_reported():
    status = read_status(_serve(["END"]))
    assert status.reachable and not status.peers
    assert "serves no peers" in status.reason


_NODES = (
    NodeAddress(
        "spark-a",
        "worker@10.0.0.2",
        addresses=("10.0.0.2",),
        hostname="spark-a.local",
        python_executable="/opt/py",
    ),
    NodeAddress("spark-b", "worker@spark-b.local", addresses=("10.0.0.3",)),
)


def _status(*peers):
    return DaemonStatus("/tmp/x.sock", True, "", "0.1.19", tuple(peers))


def test_peers_resolve_to_nodes_by_address_hostname_or_ssh_target():
    links = discover_links(
        _status(
            PeerStatus("linka", True, host="spark-a"),
            PeerStatus("linkb", True, host="10.0.0.3"),
        ),
        _NODES,
    )
    assert [(link.name, link.peer_node_id, link.usable) for link in links] == [
        ("linka", "spark-a", True),
        ("linkb", "spark-b", True),
    ]


def test_unusable_links_say_why():
    links = discover_links(
        _status(
            PeerStatus("old", True),
            PeerStatus("down", False, host="10.0.0.2"),
            PeerStatus("stray", True, host="10.9.9.9"),
            PeerStatus("bad name!", True, host="10.0.0.2"),
        ),
        _NODES,
    )
    reasons = {link.name: link.reason for link in links}
    assert reasons["old"] == "mcdma-rpcd is too old to report the peer host"
    assert reasons["down"] == "mcdma-rpcd reports this link down"
    assert reasons["stray"] == "no enrolled node answers to 10.9.9.9"
    assert "invalid RDMA link name" in reasons["bad name!"]
    assert not any(link.usable for link in links)


def test_an_address_shared_by_two_nodes_is_ambiguous():
    twins = (NodeAddress("a", "u@10.0.0.9"), NodeAddress("b", "u@10.0.0.9"))
    (link,) = discover_links(_status(PeerStatus("linkx", True, host="10.0.0.9")), twins)
    assert link.reason == "more than one enrolled node answers to 10.0.0.9"
