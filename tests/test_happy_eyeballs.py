# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.utils.happy_eyeballs.

The race tests replace ``socket.socket`` and the module's ``selectors``
with fakes that complete connects on a schedule, so they run on any
OS/CI without network or IPv6 connectivity.  The real-kernel tests use
loopback listeners and a self-calibrated blackhole IPv6 address, and
skip themselves when the host cannot produce the needed condition.
"""

import errno
import inspect
import selectors
import socket
import threading
import time
from types import SimpleNamespace

import pytest

from omlx.utils import happy_eyeballs as he

V6_SA = ("2606:4700:4700::1111", 443, 0, 0)
V4_SA = ("99.84.91.11", 443)


def ai6(sockaddr=V6_SA):
    return (socket.AF_INET6, socket.SOCK_STREAM, 0, "", sockaddr)


def ai4(sockaddr=V4_SA):
    return (socket.AF_INET, socket.SOCK_STREAM, 0, "", sockaddr)


# A plan says how one connect attempt ends:
#   "ok"       - completes successfully after a short delay
#   "refused"  - fails with ECONNREFUSED after a short delay
#   "reject"   - connect_ex() fails at once with ECONNREFUSED
#   "instant"  - connect_ex() connects at once
#   "hang"     - never completes (blackholed SYN)
#   "kernel"   - the kernel gives up after KERNEL_GIVE_UP (ETIMEDOUT)
#   (mode, seconds) - "ok"/"refused" after that many seconds
FAST = 0.01
KERNEL_GIVE_UP = 0.4


class FakeSocket:
    def __init__(self, family, plan, registry):
        self.family = family
        self.mode, self.after = plan if isinstance(plan, tuple) else (plan, FAST)
        self.timeout = None
        self.closed = False
        self.connected_to = None
        self.ready_at = None
        registry.append(self)

    # -- non-blocking path ------------------------------------------------
    def setblocking(self, flag):
        self.timeout = None if flag else 0.0

    def connect_ex(self, sockaddr):
        self.connected_to = sockaddr
        if self.mode == "reject":
            return errno.ECONNREFUSED
        if self.mode == "instant":
            return 0
        if self.mode == "kernel":
            self.ready_at = time.monotonic() + KERNEL_GIVE_UP
        elif self.mode != "hang":
            self.ready_at = time.monotonic() + self.after
        return errno.EINPROGRESS

    def getsockopt(self, level, option):
        assert (level, option) == (socket.SOL_SOCKET, socket.SO_ERROR)
        return {"ok": 0, "refused": errno.ECONNREFUSED, "kernel": errno.ETIMEDOUT}[
            self.mode
        ]

    # -- blocking path (sequential fallback) --------------------------------
    def settimeout(self, value):
        self.timeout = value

    def connect(self, sockaddr):
        self.connected_to = sockaddr
        if self.mode in ("refused", "reject"):
            raise ConnectionRefusedError(errno.ECONNREFUSED, "Connection refused")
        if self.mode == "hang":
            time.sleep(self.timeout or 0)
            raise TimeoutError("timed out")

    def close(self):
        self.closed = True


class FakeSelector:
    """Reports a FakeSocket writable once its scheduled connect completes."""

    instances = []

    def __init__(self):
        self.socks = []
        self.closed = False
        FakeSelector.instances.append(self)

    def register(self, sock, events):
        assert events == selectors.EVENT_WRITE
        self.socks.append(sock)

    def unregister(self, sock):
        self.socks.remove(sock)

    def select(self, timeout=None):
        now = time.monotonic()
        wake = min((s.ready_at for s in self.socks if s.ready_at), default=None)
        if timeout is not None:
            wake = now + timeout if wake is None else min(wake, now + timeout)
        assert wake is not None, "select() would block forever"
        time.sleep(max(0.0, wake - now))
        now = time.monotonic()
        return [
            (SimpleNamespace(fileobj=s), selectors.EVENT_WRITE)
            for s in list(self.socks)
            if s.ready_at and s.ready_at <= now
        ]

    def close(self):
        self.closed = True


class NetStub:
    """Patches socket.socket / getaddrinfo / selectors with per-family plans."""

    def __init__(self, monkeypatch, addrinfos=(), plans=None, **families):
        self.created = []
        plans = dict(plans or {})

        def fake_socket(family, socktype=0, proto=0, *args, **kwargs):
            index = len(self.created)
            if index in plans:
                plan = plans[index]
            else:
                plan = families.get("v6" if family == socket.AF_INET6 else "v4", "ok")
            return FakeSocket(family, plan, self.created)

        def fake_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
            return list(addrinfos)

        FakeSelector.instances = []
        monkeypatch.setattr(socket, "socket", fake_socket)
        monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
        monkeypatch.setattr(
            he,
            "selectors",
            SimpleNamespace(
                DefaultSelector=FakeSelector, EVENT_WRITE=selectors.EVENT_WRITE
            ),
        )

    def of(self, family):
        return [s for s in self.created if s.family == family]

    def losers(self, winner):
        return [s for s in self.created if s is not winner]


def _timed(fn, *args, **kwargs):
    t0 = time.monotonic()
    result = fn(*args, **kwargs)
    return result, time.monotonic() - t0


# ---------------------------------------------------------------------------
# Race behavior (fake sockets)
# ---------------------------------------------------------------------------


def test_blackholed_v6_v4_wins_after_delay(monkeypatch):
    stub = NetStub(monkeypatch, v6="hang", v4="ok")
    sock, elapsed = _timed(he.connect_addrinfos, [ai6(), ai4()], 10.0)
    assert sock.family == socket.AF_INET
    assert he.HAPPY_EYEBALLS_DELAY <= elapsed < 1.0, f"race timing: {elapsed:.2f}s"
    assert all(s.closed for s in stub.losers(sock)), "loser left open"
    assert sock.timeout == 10.0  # caller's timeout restored on the winner


def test_no_timeout_closes_loser_and_starts_no_thread(monkeypatch):
    # timeout=None is what huggingface_hub passes by default; a blackholed
    # attempt must still be closed when the call returns.
    stub = NetStub(monkeypatch, v6="hang", v4="ok")
    threads_before = threading.active_count()
    sock = he.connect_addrinfos([ai6(), ai4()], None)
    assert sock.family == socket.AF_INET
    assert sock.timeout is None  # blocking, as the stdlib leaves it
    assert all(s.closed for s in stub.losers(sock))
    assert threading.active_count() == threads_before
    assert FakeSelector.instances and all(x.closed for x in FakeSelector.instances)


def test_no_timeout_all_fail_ends_when_kernel_gives_up(monkeypatch):
    stub = NetStub(monkeypatch, v6="kernel", v4="kernel")
    with pytest.raises(TimeoutError):
        _timed(he.connect_addrinfos, [ai6(), ai4()], None)
    assert all(s.closed for s in stub.created)


def test_fast_v6_failure_starts_v4_immediately(monkeypatch):
    NetStub(monkeypatch, v6="refused", v4="ok")
    sock, elapsed = _timed(he.connect_addrinfos, [ai6(), ai4()], 2.0)
    assert sock.family == socket.AF_INET
    assert elapsed < he.HAPPY_EYEBALLS_DELAY, f"delayed v4 start: {elapsed:.2f}s"


def test_immediate_v6_reject_starts_v4_immediately(monkeypatch):
    NetStub(monkeypatch, v6="reject", v4="ok")
    sock, elapsed = _timed(he.connect_addrinfos, [ai6(), ai4()], 2.0)
    assert sock.family == socket.AF_INET
    assert elapsed < he.HAPPY_EYEBALLS_DELAY


def test_immediate_connect_wins(monkeypatch):
    stub = NetStub(monkeypatch, v6="instant", v4="ok")
    sock = he.connect_addrinfos([ai6(), ai4()], 2.0)
    assert sock.family == socket.AF_INET6
    assert stub.of(socket.AF_INET) == []


def test_healthy_v6_wins_without_touching_v4(monkeypatch):
    stub = NetStub(monkeypatch, v6="ok", v4="ok")
    sock = he.connect_addrinfos([ai6(), ai4()], 1.0)
    assert sock.family == socket.AF_INET6
    assert stub.of(socket.AF_INET) == [], "IPv4 must not start when IPv6 wins"


def test_addresses_are_interleaved_and_losers_closed(monkeypatch):
    v6a, v6b = ai6(("2001:db8::a", 443, 0, 0)), ai6(("2001:db8::b", 443, 0, 0))
    v4a, v4b = ai4(("192.0.2.1", 443)), ai4(("192.0.2.2", 443))
    # Fourth attempt (v4b) is the only one that answers.
    stub = NetStub(monkeypatch, plans={0: "hang", 1: "hang", 2: "hang", 3: "ok"})
    sock = he.connect_addrinfos([v6a, v6b, v4a, v4b], 10.0)
    order = [s.connected_to for s in stub.created]
    assert order == [v6a[4], v4a[4], v6b[4], v4b[4]]
    assert sock.connected_to == v4b[4]
    assert all(s.closed for s in stub.losers(sock))


def test_resolver_preference_is_kept_when_v4_comes_first(monkeypatch):
    stub = NetStub(monkeypatch, v6="ok", v4="ok")
    sock = he.connect_addrinfos([ai4(), ai6()], 1.0)
    assert sock.family == socket.AF_INET
    assert stub.of(socket.AF_INET6) == []


def test_per_attempt_timeout_raises_socket_timeout(monkeypatch):
    stub = NetStub(monkeypatch, v6="hang", v4="hang")
    t0 = time.monotonic()
    with pytest.raises(socket.timeout):
        he.connect_addrinfos([ai6(), ai4()], 0.2)
    elapsed = time.monotonic() - t0
    # Last attempt starts at the delay and gets its own 0.2 s, like the
    # stdlib's per-address timeout - never the sequential sum.
    assert elapsed < he.HAPPY_EYEBALLS_DELAY + 0.2 + 0.3
    assert all(s.closed for s in stub.created)


def test_all_fail_raises_last_error(monkeypatch):
    NetStub(monkeypatch, v6="refused", v4="refused")
    with pytest.raises(ConnectionRefusedError):
        he.connect_addrinfos([ai6(), ai4()], 1.0)


def test_all_errors_raises_exception_group(monkeypatch):
    NetStub(monkeypatch, v6="refused", v4="reject")
    with pytest.raises(ExceptionGroup) as info:
        he.connect_addrinfos([ai6(), ai4()], 1.0, all_errors=True)
    assert len(info.value.exceptions) == 2
    assert all(isinstance(e, ConnectionRefusedError) for e in info.value.exceptions)


def test_single_family_v4_keeps_sequential_semantics(monkeypatch):
    infos = [ai4(("10.0.0.1", 80)), ai4(("10.0.0.2", 80))]
    stub = NetStub(monkeypatch, v4="ok")
    sock = he.connect_addrinfos(infos, 1.0)
    assert sock.connected_to == ("10.0.0.1", 80)  # resolver order preserved
    assert FakeSelector.instances == []  # stdlib loop, no race
    assert len(stub.created) == 1


def test_single_family_all_errors(monkeypatch):
    infos = [ai4(("10.0.0.1", 80)), ai4(("10.0.0.2", 80))]
    NetStub(monkeypatch, v4="refused")
    with pytest.raises(ExceptionGroup) as info:
        he.connect_addrinfos(infos, 1.0, all_errors=True)
    assert len(info.value.exceptions) == 2


def test_empty_addrinfos_raises_oserror():
    with pytest.raises(OSError):
        he.connect_addrinfos([], 1.0)


# ---------------------------------------------------------------------------
# create_connection wrapper and install()
# ---------------------------------------------------------------------------


def test_signature_matches_stdlib():
    ours = inspect.signature(he.create_connection).parameters
    stdlib = inspect.signature(he._ORIGINAL_CREATE_CONNECTION).parameters
    assert list(ours) == list(stdlib)
    for name, param in stdlib.items():
        assert ours[name].kind == param.kind, name
        assert ours[name].default is param.default, name


def test_create_connection_end_to_end(monkeypatch):
    NetStub(monkeypatch, [ai6(), ai4()], v6="hang", v4="ok")
    sock, elapsed = _timed(he.create_connection, ("example.com", 443), timeout=5.0)
    assert sock.family == socket.AF_INET
    assert elapsed < 1.0


def test_stdlib_default_timeout_gives_default_on_winner(monkeypatch):
    NetStub(monkeypatch, [ai6(), ai4()], v6="ok", v4="ok")
    monkeypatch.setattr(socket, "getdefaulttimeout", lambda: 7.0)
    sock = he.create_connection(("example.com", 443))
    assert sock.family == socket.AF_INET6
    assert sock.timeout == 7.0


def _uninstalled(monkeypatch):
    # Server-start tests (cli.serve_command, server.main) install the
    # wrapper for real; start from the stdlib function either way.
    monkeypatch.setattr(socket, "create_connection", he._ORIGINAL_CREATE_CONNECTION)


def test_install_is_idempotent(monkeypatch):
    _uninstalled(monkeypatch)
    assert he.install() is True
    assert he.install() is False
    assert socket.create_connection is he.create_connection


def test_installed_wrapper_races(monkeypatch):
    _uninstalled(monkeypatch)
    he.install()
    NetStub(monkeypatch, [ai6(), ai4()], v6="hang", v4="ok")
    sock, elapsed = _timed(socket.create_connection, ("example.com", 443), 5.0)
    assert sock.family == socket.AF_INET
    assert elapsed < 1.0


@pytest.mark.parametrize(
    ("address", "kwargs"),
    [
        (("example.com", 443), {"source_address": ("1.2.3.4", 0)}),
        (object(), {}),
        ((b"example.com", 443), {}),
    ],
)
def test_delegates_to_original(monkeypatch, address, kwargs):
    calls = []

    def spy(*args, **kw):
        calls.append((args, kw))
        return "original-socket"

    monkeypatch.setattr(he, "_ORIGINAL_CREATE_CONNECTION", spy)
    result = he.create_connection(address, 1.0, all_errors=True, **kwargs)
    assert result == "original-socket"
    (args, kw), = calls
    assert args == (address, 1.0, kwargs.get("source_address"))
    assert kw == {"all_errors": True}


# ---------------------------------------------------------------------------
# Real kernel sockets
# ---------------------------------------------------------------------------


class _Recording:
    """socket.socket subclass factory that keeps every socket it made."""

    def __init__(self, monkeypatch):
        self.made = []
        made = self.made
        base = socket.socket

        class Rec(base):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                made.append(self)

        monkeypatch.setattr(socket, "socket", Rec)


def _listener(family, host):
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.bind((host, 0))
    sock.listen(4)
    return sock


def _v6_loopback_or_skip():
    try:
        return _listener(socket.AF_INET6, "::1")
    except OSError:
        pytest.skip("no IPv6 loopback on this host")


def test_real_loopback_v6_wins_when_healthy(monkeypatch):
    l6, l4 = _v6_loopback_or_skip(), _listener(socket.AF_INET, "127.0.0.1")
    rec = _Recording(monkeypatch)
    try:
        v6 = socket.getaddrinfo("::1", l6.getsockname()[1], socket.AF_INET6, socket.SOCK_STREAM)[0]
        v4 = socket.getaddrinfo("127.0.0.1", l4.getsockname()[1], socket.AF_INET, socket.SOCK_STREAM)[0]
        sock = he.connect_addrinfos([v6, v4], 5.0)
        assert sock.family == socket.AF_INET6
        assert sock.gettimeout() == 5.0
        assert len(rec.made) == 1  # IPv4 never started
        sock.close()
    finally:
        l6.close()
        l4.close()


def test_real_refused_v6_falls_back_at_once():
    probe = _v6_loopback_or_skip()
    dead_port = probe.getsockname()[1]
    probe.close()  # nothing listens there now: instant RST
    l4 = _listener(socket.AF_INET, "127.0.0.1")
    try:
        v6 = socket.getaddrinfo("::1", dead_port, socket.AF_INET6, socket.SOCK_STREAM)[0]
        v4 = socket.getaddrinfo("127.0.0.1", l4.getsockname()[1], socket.AF_INET, socket.SOCK_STREAM)[0]
        sock, elapsed = _timed(he.connect_addrinfos, [v6, v4], None)
        assert sock.family == socket.AF_INET
        assert elapsed < he.HAPPY_EYEBALLS_DELAY
        assert sock.gettimeout() is None
        sock.close()
    finally:
        l4.close()


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


def test_real_blackholed_ipv6_does_not_block_ipv4(monkeypatch):
    listener = _listener(socket.AF_INET, "127.0.0.1")
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

    rec = _Recording(monkeypatch)
    v4_info = socket.getaddrinfo("127.0.0.1", port, socket.AF_INET, socket.SOCK_STREAM)[0]
    try:
        sock, elapsed = _timed(he.connect_addrinfos, [blackhole, v4_info], None)
        assert elapsed < 2.0, f"real race too slow: {elapsed:.2f}s"
        conn, _ = listener.accept()  # proves the winner really connected
        conn.close()
        losers = [s for s in rec.made if s.family == socket.AF_INET6]
        assert losers and all(s.fileno() == -1 for s in losers), "loser left open"
        sock.close()
    finally:
        listener.close()
