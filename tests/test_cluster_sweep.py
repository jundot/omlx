# SPDX-License-Identifier: Apache-2.0
"""Subnet sweep scanner: CIDR parsing, port probes, merged discovery."""

from __future__ import annotations

import pytest

from omlx.cluster.sweep import (
    _interface_prefix_length,
    default_sweep_ports,
    detect_local_subnet,
    expand_cidr,
    is_port_open,
    sweep_subnet,
)


def test_default_sweep_ports_includes_ssh_and_configured_admin_port(monkeypatch):
    import omlx.cluster.sweep as sweep_module

    monkeypatch.setattr(sweep_module.ServerSettings, "port", 8443)
    assert 22 in default_sweep_ports()
    assert 8443 in default_sweep_ports()


def test_interface_prefix_length_reads_linux_ip_output(monkeypatch):
    import subprocess

    fake_output = (
        "3: wlo1    inet 192.168.1.10/24 brd 192.168.1.255 scope global "
        "dynamic noprefixroute wlo1\n"
    )

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout=fake_output)

    monkeypatch.setattr("omlx.cluster.sweep.subprocess.run", fake_run)
    assert _interface_prefix_length("192.168.1.10") == 24


def test_interface_prefix_length_reads_macos_netmask_hex(monkeypatch):
    import subprocess

    fake_output = "\tinet 192.168.1.64 netmask 0xffffff00 broadcast 192.168.1.255\n"

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout=fake_output)

    monkeypatch.setattr("omlx.cluster.sweep.subprocess.run", fake_run)
    assert _interface_prefix_length("192.168.1.64") == 24


def test_interface_prefix_length_returns_none_when_unparseable(monkeypatch):
    import subprocess

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="")

    monkeypatch.setattr("omlx.cluster.sweep.subprocess.run", fake_run)
    assert _interface_prefix_length("192.168.1.64") is None


def test_detect_local_subnet_returns_cidr(monkeypatch):
    def fake_socket_connect(self, address):
        self._fake_local = "192.168.1.10"

    def fake_getsockname(self):
        return (self._fake_local, 0)

    class FakeSocket:
        def __init__(self, *args, **kwargs):
            self._fake_local = None

        def connect(self, address):
            fake_socket_connect(self, address)

        def getsockname(self):
            return fake_getsockname(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        "omlx.cluster.sweep.socket.socket",
        lambda *a, **k: FakeSocket(),
    )
    monkeypatch.setattr(
        "omlx.cluster.sweep._interface_prefix_length",
        lambda address: 24,
    )
    assert detect_local_subnet() == "192.168.1.0/24"


def test_detect_local_subnet_returns_empty_when_offline(monkeypatch):
    class FakeSocket:
        def __init__(self, *args, **kwargs):
            pass

        def connect(self, address):
            raise OSError("no route")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        "omlx.cluster.sweep.socket.socket",
        lambda *a, **k: FakeSocket(),
    )
    assert detect_local_subnet() == ""


def test_expand_cidr_lists_all_host_addresses():
    addresses = expand_cidr("192.168.1.0/30")
    assert addresses == ["192.168.1.1", "192.168.1.2"]


def test_expand_cidr_rejects_bad_input():
    with pytest.raises(ValueError):
        expand_cidr("not-an-ip")
    with pytest.raises(ValueError):
        expand_cidr("10.0.0.0/99")


def test_is_port_open_returns_bool_for_reachable_and_closed():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        assert is_port_open("127.0.0.1", port, timeout=1) is True
        assert is_port_open("127.0.0.1", 1, timeout=1) is False


def test_is_port_open_rejects_non_ip_target():
    with pytest.raises(ValueError):
        is_port_open("evil;rm -rf /", 22, timeout=1)


def test_sweep_subnet_returns_reachable_hosts_with_ports(monkeypatch):
    def fake_is_port_open(address, port, timeout=0.5):
        return address in {"192.168.1.1"} or port == 8000

    monkeypatch.setattr(
        "omlx.cluster.sweep.is_port_open",
        fake_is_port_open,
    )
    found = sweep_subnet(
        "192.168.1.0/30",
        ports=(22, 8000),
        timeout=1,
        max_workers=8,
    )
    assert isinstance(found, list)
    for host in found:
        assert "address" in host
        assert "open_ports" in host
        assert set(host["open_ports"]).issubset({22, 8000})


def test_interface_prefix_length_returns_none_when_local_ip_unmatched(monkeypatch):
    import subprocess

    # `ip -o` succeeds but the local_ip is on no interface — falls through.
    fake_output = (
        "3: wlo1    inet 10.0.0.1/24 brd 10.0.0.255 scope global wlo1\n"
    )

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout=fake_output)

    monkeypatch.setattr("omlx.cluster.sweep.subprocess.run", fake_run)
    assert _interface_prefix_length("192.168.1.10") is None


def test_interface_prefix_length_reads_decimal_netmask(monkeypatch):
    # Legacy net-tools ifconfig emits decimal netmasks, not hex.
    import subprocess

    fake_output = "\tinet 192.168.1.64 netmask 255.255.255.0 broadcast 192.168.1.255\n"

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout=fake_output)

    monkeypatch.setattr("omlx.cluster.sweep.subprocess.run", fake_run)
    assert _interface_prefix_length("192.168.1.64") == 24


def test_interface_prefix_length_anchors_token_to_prevent_substring_match(monkeypatch):
    """192.168.1.1 must not match a line for 192.168.1.10."""
    import subprocess

    fake_output = "3: wlo1    inet 192.168.1.10/24 brd 192.168.1.255 scope global wlo1\n"

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout=fake_output)

    monkeypatch.setattr("omlx.cluster.sweep.subprocess.run", fake_run)
    assert _interface_prefix_length("192.168.1.1") is None
