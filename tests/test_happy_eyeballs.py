# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.utils.happy_eyeballs.

All fake-based tests run on any OS/CI: the resolver and the socket class
are monkeypatched, so no real network or IPv6 connectivity is needed.
One test uses real sockets against a self-calibrated blackhole IPv6
address and skips itself when the environment cannot produce a hang.
"""

import socket
import time

import pytest

from omlx.utils import happy_eyeballs as he

V6_SA = ("2606:4700:4700::1111", 443, 0, 0)
V4_SA = ("99.84.91.11", 443)


def ai6(sockaddr=V6_SA):
    return (socket.AF_INET6, socket.SOCK_STREAM, 0, "", sockaddr)


def ai4(sockaddr=V4_SA):
    return (socket.AF_INET, socket.SOCK_STREAM, 0, "", sockaddr)


def _behavior(mode):
    """Map a mode name to a connect() behavior."""

    def behavior(sock):
        if mode == "ok":
            return
        if mode == "refused":
            raise ConnectionRefusedError(61, "Connection refused")
        if mode == "hang":
            limit = sock.timeout if sock.timeout else 30
            time.sleep(limit)
            raise socket.timeout("timed out")
        if callable(mode):
            return mode(sock)
        raise AssertionError(f"unknown mode {mode!r}")

    return behavior


class FakeSocket:
    def __init__(self, family, mode, registry):
        self.family = family
        self.timeout = None
        self.closed = False
        self.connected_to = None
        self._mode = mode
        self._registry = registry

    def settimeout(self, value):
        self.timeout = value

    def connect(self, sockaddr):
        self._registry.append(self)
        self.connected_to = sockaddr
        _behavior(self._mode)(self)

    def close(self):
        self.closed = True


class NetStub:
    """Patches socket.socket / socket.getaddrinfo with per-family behavior."""

    def __init__(self, monkeypatch, addrinfos, **modes):
        self.created = []
        self._modes = modes
        infos = addrinfos

        def fake_socket(family, socktype, proto, *args, **kwargs):
            mode = modes.get("v6" if family == socket.AF_INET6 else "v4", "ok")
            return FakeSocket(family, mode, self.created)

        def fake_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
            return infos

        monkeypatch.setattr(socket, "socket", fake_socket)
        monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    def sockets_of(self, family):
        return [s for s in self.created if s.family == family]


# ---------------------------------------------------------------------------
# Race behavior (fake sockets)
# ---------------------------------------------------------------------------


def test_blackholed_v6_v4_wins_within_delay(monkeypatch):
    stub = NetStub(monkeypatch, [ai6(), ai4()], v6="hang", v4="ok")
    t0 = time.monotonic()
    sock = he.connect_addrinfos([ai6(), ai4()], timeout=0.4)
    elapsed = time.monotonic() - t0
    assert sock.family == socket.AF_INET
    # v4 starts at 0.3 s and connects instantly; must not wait for the
    # 0.4 s v6 timeout, let alone the stdlib's sequential sum.
    assert elapsed < 1.0, f"race too slow: {elapsed:.2f}s"


def test_fast_v6_failure_starts_v4_immediately(monkeypatch):
    stub = NetStub(monkeypatch, [ai6(), ai4()], v6="refused", v4="ok")
    t0 = time.monotonic()
    sock = he.connect_addrinfos([ai6(), ai4()], timeout=2.0)
    elapsed = time.monotonic() - t0
    assert sock.family == socket.AF_INET
    # No 300 ms penalty when the first family errors fast.
    assert elapsed < he.HAPPY_EYEBALLS_DELAY, f"delayed v4 start: {elapsed:.2f}s"


def test_healthy_v6_wins_without_touching_v4(monkeypatch):
    stub = NetStub(monkeypatch, [ai6(), ai4()], v6="ok", v4="ok")
    sock = he.connect_addrinfos([ai6(), ai4()], timeout=1.0)
    assert sock.family == socket.AF_INET6
    time.sleep(he.HAPPY_EYEBALLS_DELAY + 0.1)  # give a stray v4 attempt a chance
    assert stub.sockets_of(socket.AF_INET) == [], (
        "IPv4 must not be attempted when IPv6 connects first"
    )


def test_single_family_v4_keeps_sequential_semantics(monkeypatch):
    NetStub(monkeypatch, [ai4(("10.0.0.1", 80)), ai4(("10.0.0.2", 80))], v4="ok")
    sock = he.connect_addrinfos(
        [ai4(("10.0.0.1", 80)), ai4(("10.0.0.2", 80))], timeout=1.0
    )
    assert sock.connected_to == ("10.0.0.1", 80)  # resolver order preserved


def test_single_family_v6_only(monkeypatch):
    NetStub(monkeypatch, [ai6()], v6="ok")
    sock = he.connect_addrinfos([ai6()], timeout=1.0)
    assert sock.family == socket.AF_INET6


def test_both_racers_fail_then_remaining_addresses_are_swept(monkeypatch):
    third = ai4(("10.0.0.3", 80))
    attempts = {"n": 0}

    def v4_mode(s):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ConnectionRefusedError(61, "Connection refused")
        # second v4 candidate (the sweep address) connects fine

    NetStub(monkeypatch, [ai6(), ai4(), third], v6="hang", v4=v4_mode)
    t0 = time.monotonic()
    sock = he.connect_addrinfos([ai6(), ai4(), third], timeout=0.3)
    assert sock.connected_to == ("10.0.0.3", 80)
    assert time.monotonic() - t0 < 2.0


def test_all_fail_raises_last_family_error(monkeypatch):
    NetStub(monkeypatch, [ai6(), ai4()], v6="refused", v4="hang")
    with pytest.raises(socket.timeout):
        he.connect_addrinfos([ai6(), ai4()], timeout=0.2)


def test_empty_addrinfos_raises_oserror():
    with pytest.raises(OSError):
        he.connect_addrinfos([], timeout=1.0)


# ---------------------------------------------------------------------------
# create_connection wrapper and install()
# ---------------------------------------------------------------------------


def test_create_connection_end_to_end(monkeypatch):
    NetStub(monkeypatch, [ai6(), ai4()], v6="hang", v4="ok")
    t0 = time.monotonic()
    sock = he.create_connection(("example.com", 443), timeout=0.4)
    assert sock.family == socket.AF_INET
    assert time.monotonic() - t0 < 1.0


def test_install_is_idempotent_and_restorable(monkeypatch):
    monkeypatch.setattr(socket, "create_connection", socket.create_connection)
    assert he.install() is True
    assert he.install() is False
    assert getattr(socket.create_connection, he._INSTALLED_ATTR, False)
    assert socket.create_connection is he.create_connection


def test_installed_wrapper_races(monkeypatch):
    monkeypatch.setattr(socket, "create_connection", socket.create_connection)
    he.install()
    NetStub(monkeypatch, [ai6(), ai4()], v6="hang", v4="ok")
    t0 = time.monotonic()
    sock = socket.create_connection(("example.com", 443), 0.4)
    assert sock.family == socket.AF_INET
    assert time.monotonic() - t0 < 1.0


def test_source_address_delegates_to_original(monkeypatch):
    calls = []

    def spy(address, *args, **kwargs):
        calls.append((address, args, kwargs))
        return "original-socket"

    monkeypatch.setattr(he, "_ORIGINAL_CREATE_CONNECTION", spy)
    result = he.create_connection(("example.com", 443), 1.0, source_address="1.2.3.4")
    assert result == "original-socket"
    assert calls and calls[0][2].get("source_address") == "1.2.3.4"


def test_malformed_address_delegates_to_original(monkeypatch):
    calls = []

    def spy(address, *args, **kwargs):
        calls.append(address)
        raise ValueError("address should be a (host, port) tuple")

    monkeypatch.setattr(he, "_ORIGINAL_CREATE_CONNECTION", spy)
    with pytest.raises(ValueError):
        he.create_connection(object(), 1.0)
    assert calls == [calls[0]]


def test_stdlib_default_timeout_sentinel_accepted(monkeypatch):
    NetStub(monkeypatch, [ai6(), ai4()], v6="ok", v4="ok")
    sock = he.create_connection(
        ("example.com", 443), socket._GLOBAL_DEFAULT_TIMEOUT
    )
    assert sock.family == socket.AF_INET6


# ---------------------------------------------------------------------------
# Real-kernel test: genuine blackholed IPv6 against a genuine IPv4 listener
# ---------------------------------------------------------------------------


def _blackhole_candidates(port):
    """IPv6 addresses likely to blackhole (SYNs go nowhere, no error)."""
    cands = []
    try:
        for idx, name in socket.if_nameindex():
            if name.startswith("lo"):
                continue
            cands.append(ai6(("fe80::dead:beef", port, 0, idx)))
    except OSError:
        pass
    cands.append(ai6(("2001:db8::1", port, 0, 0)))  # RFC 3849, not routable
    return cands


def test_real_blackholed_ipv6_does_not_block_ipv4():
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    blackhole = None
    for cand in _blackhole_candidates(port):
        probe = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        probe.settimeout(0.5)
        try:
            probe.connect(cand[4])  # connected: not a blackhole
        except socket.timeout:
            blackhole = cand  # SYN swallowed: exactly the failure mode
        except OSError:
            pass  # instant error: the sequential path handles this fine
        finally:
            probe.close()
        if blackhole:
            break
    if blackhole is None:
        listener.close()
        pytest.skip("no blackhole IPv6 address available on this host")

    v4_info = socket.getaddrinfo("127.0.0.1", port, socket.AF_INET, socket.SOCK_STREAM)[0]
    try:
        t0 = time.monotonic()
        sock = he.connect_addrinfos([blackhole, v4_info], timeout=5.0)
        elapsed = time.monotonic() - t0
        assert elapsed < 2.0, f"real race too slow: {elapsed:.2f}s"
        conn, _ = listener.accept()  # proves the winner really connected
        conn.close()
        sock.close()
    finally:
        listener.close()
