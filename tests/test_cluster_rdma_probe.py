# SPDX-License-Identifier: Apache-2.0
"""The live link probe checks every byte both ways, through the real client and service code."""

from __future__ import annotations

import json
import subprocess
import sys
import threading

import pytest
from rdma_loopback import LoopbackLink, PythonWordOps

from omlx.cluster.rdma import probe_wire
from omlx.cluster.rdma.daemon import DaemonStatus
from omlx.cluster.rdma.link_probe import (
    ProbeService,
    ProbeSettings,
    probe_link,
    start_probe_service,
    verify_link,
)
from omlx.cluster.rdma.link_probe_service import serve_probe
from omlx.cluster.rdma.links import NodeAddress, RdmaLink
from omlx.cluster.rdma.mailbox import ClientMailbox, ServiceMailbox
from omlx.cluster.rdma.probe_wire import ProbeError
from omlx.cluster.rdma.words import load_word_ops

_SMALL = ProbeSettings(
    warmup=2, round_trips=20, bulk_bytes=256 * 1024, repeats=2, call_timeout_s=5.0
)


@pytest.fixture
def link():
    loop = LoopbackLink(request_bytes=512 * 1024, reply_bytes=512 * 1024)
    yield loop
    loop.close()


def _serve_in_thread(link, ops=None):
    service = ServiceMailbox.attach(
        link.name,
        link.socket_path,
        ops or PythonWordOps(),
        mailbox_path=link.mailbox_path,
    )
    result = {}

    def run():
        try:
            result["summary"] = serve_probe(service, idle_timeout_s=10, deadline_s=30)
        finally:
            service.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, result


def test_probe_measures_a_clean_link_both_ways(link):
    thread, result = _serve_in_thread(link)
    client = ClientMailbox.attach(link.name, PythonWordOps())
    try:
        measured = probe_link(client, _SMALL)
    finally:
        client.close()
    thread.join(timeout=10)
    assert result["summary"] == {
        "ended": "end",
        "served": {"echo": 22, "sink": 2, "source": 2},
    }
    assert measured.round_trips == 20
    assert measured.to_peer_bytes == 2 * 256 * 1024
    assert measured.from_peer_bytes == 2 * 256 * 1024
    assert (
        measured.latency_p50_us > 0
        and measured.to_peer_gbit_s > 0
        and measured.from_peer_gbit_s > 0
    )


@pytest.mark.skipif(load_word_ops()[0] is None, reason="libmcdma-rpc is not installed")
def test_the_installed_helper_carries_the_probe(link):
    ops, _ = load_word_ops()
    thread, result = _serve_in_thread(link, ops)
    client = ClientMailbox.attach(link.name, ops)
    try:
        measured = probe_link(client, _SMALL)
    finally:
        client.close()
    thread.join(timeout=10)
    assert result["summary"]["ended"] == "end"
    assert measured.round_trips == _SMALL.round_trips


@pytest.mark.skipif(load_word_ops()[0] is None, reason="libmcdma-rpc is not installed")
def test_a_probe_service_whose_coordinator_goes_away_frees_the_link(link):
    service = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "omlx.cluster.rdma.link_probe_service",
            "--name",
            link.name,
            "--socket",
            link.socket_path,
            "--mailbox",
            link.mailbox_path,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert json.loads(service.stdout.readline())["ready"]
        # The coordinator's SSH client dying closes the service's stdin.
        service.stdin.close()
        assert service.wait(timeout=5) == 1
        assert json.loads(service.stdout.readline())["ended"] == "abandoned"
    finally:
        if service.poll() is None:
            service.kill()
            service.wait()


def test_a_short_answer_fails_the_probe_cleanly(link):
    service = ServiceMailbox.attach(
        link.name, link.socket_path, PythonWordOps(), mailbox_path=link.mailbox_path
    )

    def serve():
        while (got := service.next_request(timeout_s=5)) is not None:
            seq, payload = got
            body = bytes(payload[probe_wire.HEADER_BYTES :])
            if probe_wire.unpack_request(payload).kind == probe_wire.ECHO:
                service.reply(seq, (body[::-1],))
            else:
                service.reply(seq, (b"x",))
                return

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    client = ClientMailbox.attach(link.name, PythonWordOps())
    try:
        with pytest.raises(ProbeError, match="short"):
            probe_link(client, _SMALL)
    finally:
        client.close()
        thread.join(timeout=5)
        service.close()


def test_serving_stops_once_the_probe_is_abandoned(link):
    service = ServiceMailbox.attach(
        link.name, link.socket_path, PythonWordOps(), mailbox_path=link.mailbox_path
    )
    try:
        summary = serve_probe(service, idle_timeout_s=30, abandoned=lambda: True)
    finally:
        service.close()
    assert summary["ended"] == "abandoned"


def test_the_service_stays_registered_until_the_daemon_takes_bye(link):
    # mcdma-rpcd drops a staged reply once its service detaches.
    link.hold_replies = True
    thread, result = _serve_in_thread(link)
    client = ClientMailbox.attach(link.name, PythonWordOps())
    try:
        end = probe_wire.pack_request(probe_wire.ProbeRequest(probe_wire.END))
        seq = client.stage((end,))
        thread.join(timeout=0.2)
        assert thread.is_alive()
        link.hold_replies = False
        reply = client.wait(seq, 5.0)
        assert reply is not None and bytes(reply) == probe_wire.BYE
    finally:
        link.hold_replies = False
        client.close()
    thread.join(timeout=10)
    assert result["summary"]["ended"] == "end"


def test_one_flipped_byte_fails_the_probe(link):
    thread, _ = _serve_in_thread(link)
    client = ClientMailbox.attach(link.name, PythonWordOps())
    try:
        link.corrupt_next_reply = True
        with pytest.raises(ProbeError, match="echo reply did not match"):
            probe_link(client, _SMALL)
    finally:
        client.close()
    link.drop()
    thread.join(timeout=10)


class _Service:
    def __init__(self, summary):
        self.summary = summary
        self.closed = False

    def ready(self):
        return {"ready": True}

    def finish(self):
        return self.summary

    def close(self):
        self.closed = True


_NODE = NodeAddress(
    "spark-a", "worker@10.0.0.2", ("10.0.0.2",), python_executable="/opt/py"
)
_LINK = RdmaLink(
    "linka", True, "", "10.0.0.2", "spark-a", "rdma_mcrdma1", 1 << 20, 1 << 20, 5.0
)
_STATUS = DaemonStatus("/tmp/x.sock", True, "", "0.1.19")


def test_verify_link_records_measurements_when_the_probe_passes(link):
    service = _Service({"ended": "end"})
    thread, _ = _serve_in_thread(link)
    renamed = RdmaLink(link.name, True, "", "10.0.0.2", "spark-a")
    evidence = verify_link(
        renamed,
        _NODE,
        status=_STATUS,
        driver=None,
        ops=PythonWordOps(),
        settings=_SMALL,
        start_service=lambda node, name, socket_path: service,
        clock=lambda: 42.0,
    )
    thread.join(timeout=10)
    assert evidence.verified and evidence.reason == ""
    assert evidence.checked_at == 42.0 and evidence.measurements.round_trips == 20
    assert evidence.identity["peer_node_id"] == "spark-a"
    assert service.closed


@pytest.mark.parametrize(
    ("link", "ops", "reason"),
    [
        (
            RdmaLink("linka", False, "mcdma-rpcd reports this link down"),
            PythonWordOps(),
            "mcdma-rpcd reports this link down",
        ),
        (
            RdmaLink("linka", True, "", "10.0.0.9", "spark-z"),
            PythonWordOps(),
            "reaches spark-z, not spark-a",
        ),
        (_LINK, None, "libmcdma-rpc is not installed"),
    ],
)
def test_verify_link_fails_closed_before_touching_the_peer(link, ops, reason):
    def never(*args, **kwargs):
        raise AssertionError("the probe service must not start")

    evidence = verify_link(
        link, _NODE, status=_STATUS, driver=None, ops=ops, start_service=never
    )
    assert not evidence.verified
    assert reason in evidence.reason


def test_verify_link_reports_a_service_that_did_not_finish():
    service = _Service({"ended": "idle"})

    class Box:
        connected = True

        def close(self):
            pass

    evidence = verify_link(
        _LINK,
        _NODE,
        status=_STATUS,
        driver=None,
        ops=PythonWordOps(),
        attach=lambda name, ops: Box(),
        start_service=lambda node, name, socket_path: service,
        probe=lambda mailbox, settings: None,
    )
    assert not evidence.verified
    assert evidence.reason == "the probe service ended with idle"
    assert service.closed


def test_the_probe_service_starts_over_the_cluster_ssh_policy():
    captured = {}

    class Process:
        stdout = iter(['{"ready": true}\n'])

        def poll(self):
            return 0

    def popen(argv, **kwargs):
        captured["argv"] = argv
        captured["stdin"] = kwargs.get("stdin")
        return Process()

    start_probe_service(_NODE, "linka", "/tmp/mcdma-rpcd.linka.sock", popen=popen)
    argv = captured["argv"]
    assert captured["stdin"] is subprocess.PIPE
    assert argv[0].endswith("ssh") and "BatchMode=yes" in argv
    assert argv[-2] == "worker@10.0.0.2"
    assert (
        argv[-1]
        == "/opt/py -m omlx.cluster.rdma.link_probe_service --name linka --socket /tmp/mcdma-rpcd.linka.sock"
    )


def test_noise_on_the_service_output_is_not_mistaken_for_its_answer():
    class Process:
        stdout = iter(["{stray diagnostic}\n", '{"ready": true, "max_request": 1}\n'])

        def poll(self):
            return 0

    assert ProbeService(Process()).ready()["ready"] is True


def test_a_node_without_a_worker_python_cannot_host_the_probe():
    with pytest.raises(ProbeError, match="no recorded worker Python"):
        start_probe_service(
            NodeAddress("spark-a", "worker@10.0.0.2"), "linka", "/tmp/s.sock"
        )
