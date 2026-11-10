# SPDX-License-Identifier: Apache-2.0
"""The helper daemon against the oMLX side of each protocol it speaks."""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omlx_cuda_helper import families, gpus
from omlx_cuda_helper.control import ControlServer
from omlx_cuda_helper.files import ModelStore
from omlx_cuda_helper.pairing import Joiner, PairingError
from omlx_cuda_helper.state import State
from omlx_cuda_helper.supervisor import Supervisor
from omlx_cuda_helper.worker import Upstream

CAPS = {"chip": "2x NVIDIA RTX A6000", "ram_gb": 96.0, "control": "agent"}


def _wait(done, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not done():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


@pytest.fixture
def host(tmp_path):
    """A real oMLX pairing manager behind its HTTP routes."""
    pytest.importorskip("omlx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from omlx.cluster import pairing_routes
    from omlx.cluster.pairing import PairingManager

    calls = []
    manager = PairingManager(
        None,
        None,
        base_path=tmp_path / "host",
        identity={
            "node_id": "host-node",
            "friendly_name": "Studio",
            "created_at": 1.0,
            "schema_version": 1,
        },
        caps_provider=lambda: {"chip": "M3 Ultra", "ram_gb": 512.0},
        address_provider=lambda: ["192.0.2.1"],
        ssh_key_provider=lambda: calls.append("key"),
        ssh_host_key_provider=lambda: calls.append("host key"),
        enrollment_driver=lambda peer: calls.append("enroll"),
        revocation_driver=lambda material: calls.append("revoke"),
    )
    pairing_routes.set_pairing_manager_getter(lambda: manager)
    app = FastAPI()
    app.include_router(pairing_routes.pair_router)

    @app.get("/api/cluster/node_id")
    def node_id():
        return {"node_id": "host-node"}

    client = TestClient(app)

    def http(url, body=None, timeout=5.0):
        path = url.split("/", 3)[3]
        response = (
            client.get(f"/{path}")
            if body is None
            else client.post(f"/{path}", json=body)
        )
        if response.status_code >= 400:
            raise OSError(f"HTTP {response.status_code}: {response.text}")
        return response.json()

    yield manager, http, calls
    pairing_routes.set_pairing_manager_getter(None)


def test_helper_pairs_with_the_code_from_its_log(tmp_path, host, caplog):
    manager, http, ssh_calls = host
    state = State(tmp_path / "helper", name="nvidia1")
    joiner = Joiner(state, lambda: CAPS, 8765, http=http, poll_seconds=0.01)

    with caplog.at_level(logging.INFO, logger="omlx_cuda_helper.pairing"):
        joiner.join("192.0.2.1", 8000, node_id="host-node")
    code = caplog.messages[-1].split("PAIRING CODE ")[1][:6]
    pending = manager.pending_requests()[0]
    assert (pending["friendly_name"], pending["caps"]["control"]) == (
        "nvidia1",
        "agent",
    )

    with pytest.raises(Exception, match="code does not match"):
        manager.approve(state.node_id, "000000" if code != "000000" else "111111")
    manager.approve(state.node_id, code)
    _wait(lambda: "host-node" in state.hosts())

    # Both ends hold the same key and nothing touched SSH on the host.
    assert state.cluster_key("host-node") == manager.control_key(state.node_id)
    assert state.hosts()["host-node"]["name"] == "Studio"
    assert ssh_calls == []
    assert (state.root / "pairing-keys.json").stat().st_mode & 0o777 == 0o600


def test_helper_refuses_another_host_than_the_one_that_invited(tmp_path, host):
    _, http, _ = host
    joiner = Joiner(State(tmp_path / "helper"), lambda: CAPS, 8765, http=http)
    with pytest.raises(PairingError, match="not the oMLX that sent the invite"):
        joiner.join("192.0.2.1", 8000, node_id="someone-else")


def test_the_helper_reports_its_wired_links(monkeypatch):
    addresses = [
        {"ifname": "lo", "flags": ["LOOPBACK", "UP", "LOWER_UP"], "addr_info": []},
        {
            "ifname": "enp5s0",
            "flags": ["UP", "LOWER_UP"],
            "addr_info": [
                {
                    "family": "inet",
                    "local": "192.168.1.230",
                    "prefixlen": 24,
                    "scope": "global",
                },
                {
                    "family": "inet6",
                    "local": "fe80::1",
                    "prefixlen": 64,
                    "scope": "link",
                },
            ],
        },
        {
            "ifname": "enp6s0",
            "flags": ["UP", "LOWER_UP"],
            "addr_info": [
                {
                    "family": "inet6",
                    "local": "fe80::2",
                    "prefixlen": 64,
                    "scope": "link",
                }
            ],
        },
        {"ifname": "enp7s0", "flags": ["UP", "NO-CARRIER"], "addr_info": []},
        {"ifname": "docker0", "flags": ["UP", "LOWER_UP"], "addr_info": []},
    ]
    monkeypatch.setattr(gpus, "_run", lambda *command: json.dumps(addresses))
    monkeypatch.setattr(gpus, "_speed", {"enp5s0": 1000, "enp6s0": 2500}.get)
    assert gpus.links() == [
        {
            "interface": "enp5s0",
            "ipv4": ["192.168.1.230/24"],
            "link_local": "fe80::1",
            "mbps": 1000,
        },
        {"interface": "enp6s0", "ipv4": [], "link_local": "fe80::2", "mbps": 2500},
    ]
    # A control connection over the cable names its address with the port.
    assert gpus.link_mbps("fe80::2%enp6s0") == 2500


def test_omlx_drives_the_helper_over_its_control_channel(monkeypatch):
    pytest.importorskip("omlx")
    from omlx.prefill_helper import agent, launcher

    # Short paths: the stages talk over Unix sockets.
    root = Path(tempfile.mkdtemp(prefix="och-", dir="/tmp"))
    key = bytes(range(32))
    state = State(root / "state", name="nvidia1")
    state.save_host("studio", {"cluster_key": key.hex(), "name": "Studio"})
    gpu = {"name": "NVIDIA RTX A6000", "total": 48 << 30, "free": 48 << 30}
    monkeypatch.setattr(
        gpus, "gpus", lambda only=None: [{"index": 0, **gpu}, {"index": 1, **gpu}]
    )
    cable = {"interface": "enp6s0", "ipv4": [], "link_local": "fe80::2", "mbps": 2500}
    monkeypatch.setattr(gpus, "links", lambda: [cable])
    monkeypatch.setitem(
        families.FAMILIES, "fake", families.Family(contract=7, module="fake_family")
    )
    worker = [sys.executable, str(Path(__file__).parent / "fake_worker.py")]
    supervisor = Supervisor(None, root / "run", worker=worker)
    # Every address: IPv4, and IPv6 for a cable that only has link-local.
    control = ControlServer(
        state, ModelStore(root / "models"), supervisor, {"fake": 7}, bind=""
    )
    control.start()
    me = SimpleNamespace(node_id="studio")
    monkeypatch.setattr(agent, "get_node_identity", lambda: me)
    monkeypatch.setattr(agent, "adapter_for", lambda t: SimpleNamespace(contract=7))
    target = agent.AgentTarget(
        state.node_id, "nvidia1", "127.0.0.1", "ethernet", control.port, key
    )
    try:
        info = agent.remote_info(target)
        assert info["budgets"] == [48 << 30, 48 << 30]
        assert info["contracts"] == {"fake": 7} and info["lending"] is None
        assert info["device"] == "2x NVIDIA RTX A6000"
        assert info["links"] == [cable]

        # Files cross in chunks and appear only when complete.
        model = root / "Model"
        model.mkdir()
        (model / "w.safetensors").write_bytes(os.urandom(70_000))
        (model / "config.json").write_bytes(b"{}")
        sizes = {"w.safetensors": 70_000, "config.json": 2}
        monkeypatch.setattr(agent, "_CHUNK_BYTES", 16_000)
        where = agent.locate_files(target, "Model", sizes)
        assert where["found"] is None and sorted(where["missing"]) == sorted(sizes)
        agent.copy_files(target, model, where["copy"], list(sizes))
        where = agent.locate_files(target, "Model", sizes)
        copied = Path(where["found"])
        assert (copied / "w.safetensors").read_bytes() == (
            model / "w.safetensors"
        ).read_bytes()
        with pytest.raises(launcher.LaunchError, match="unsafe name"):
            agent.locate_files(target, "../escape", {})
        # A file sits where the model folder should be: the helper says what
        # failed and keeps the connection.
        (root / "models" / "Blocked").write_bytes(b"")
        with pytest.raises(launcher.LaunchError, match="File exists"):
            agent.copy_files(target, model, "Blocked", ["config.json"])

        # The session starts on two GPU stages and answers on the data port,
        # which listens where the control connection came in, here over IPv6.
        target = replace(target, address="::1")
        process = agent.launch(
            target, str(copied), "fake", 6, host_name="Studio", warm_tokens=8
        )
        assert process.hello["layers"] == 6
        lending = agent.remote_info(target)["lending"]
        assert lending == {"host": "Studio", "model": "Model"}
        host = Upstream(
            socket.create_connection(("::1", process.port), timeout=10),
            process.token,
        )
        tokens = np.arange(3, 11, dtype=np.int32)
        host.submit(
            {"op": "job", "job_id": "j", "chunks": [8]},
            {"tokens": (tokens.reshape(1, -1), "int32")},
        )
        _, arrays = host.take("j", 0)
        assert arrays["b:h"][0].shape == (1, 8, 4) and "s:5.rows" in arrays
        # The copy a session runs from is never offered for deletion, and
        # neither is a folder the operator put there: only copies are.
        mine = root / "models" / "Mine"
        mine.mkdir()
        assert agent.list_copies(target)["copies"] == []
        agent.delete_copies(target, ["Model"])
        assert copied.is_dir()

        # The helper answers a stop once the workers are gone, so their GPU
        # memory is free for what the host asks next.
        procs = list(supervisor._session.procs)
        process.stop()
        assert supervisor.status() is None
        assert all(proc.poll() is not None for proc in procs)
        copies = agent.list_copies(target)["copies"]
        assert [copy["name"] for copy in copies] == ["Model"]
        agent.delete_copies(target, ["Model", "Mine"])
        assert not copied.exists() and mine.is_dir()

        stranger = agent.AgentTarget(
            state.node_id, "nvidia1", "127.0.0.1", "ethernet", control.port, bytes(32)
        )
        with pytest.raises(launcher.LaunchError, match="Pair it again"):
            agent.remote_info(stranger)
    finally:
        supervisor.stop()
        control.stop()
        shutil.rmtree(root, ignore_errors=True)
