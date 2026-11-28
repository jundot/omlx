# SPDX-License-Identifier: Apache-2.0
"""Happy Eyeballs (RFC 8305) connection racing for synchronous sockets.

Python's async stack already races IPv4/IPv6 (anyio's ``connect_tcp``
implements Happy Eyeballs since 4.7), but the synchronous path does not:
``socket.create_connection()`` tries resolved addresses strictly one at a
time, and httpcore's sync backend (used by ``huggingface_hub``, which
oMLX calls through ``asyncio.to_thread``) goes straight through it.

On a dual-stack host whose IPv6 path silently blackholes traffic (no
RST, no ICMP - just lost SYNs), the first attempt therefore hangs until
the caller's whole timeout budget expires and the working IPv4 address
is never tried.  This is not theoretical: on a real machine with broken
IPv6 egress every ``/admin/api/hf/*`` call returned 504 after exactly
the 10 s HF API timeout, while browsers on the same machine loaded
huggingface.co in ~150 ms because they race the families.

``install()`` wraps ``socket.create_connection`` with a race on
non-blocking sockets.  Addresses are tried in interleaved family order
(the resolver's first family first), a new attempt starts every
:data:`HAPPY_EYEBALLS_DELAY` seconds or at once when the previous one
fails, the first connection to complete wins, and every other attempt is
closed at once.  All attempts run in the calling thread on one
``selectors`` loop, so no thread or socket outlives the call.  Each
attempt keeps the per-address timeout of the stdlib, so a call never
waits longer than plain ``socket.create_connection`` would.

Single-family names, explicit ``source_address`` bindings and
non-(host, port) addresses go to the original implementation unchanged,
so behaviour on healthy or IPv4-only networks is identical.
"""

from __future__ import annotations

import errno
import itertools
import os
import selectors
import socket
import time
from typing import Any

#: Time a pending attempt gets before the next address starts in
#: parallel.  RFC 8305 section 5 recommends 250 ms; 300 ms is the value
#: Chrome uses for its IPv6 fallback.
HAPPY_EYEBALLS_DELAY = 0.3

#: The stdlib's "caller did not pass a timeout" sentinel.
_DEFAULT_TIMEOUT: Any = getattr(socket, "_GLOBAL_DEFAULT_TIMEOUT", object())

_ORIGINAL_CREATE_CONNECTION = socket.create_connection

_INSTALLED_ATTR = "_omlx_happy_eyeballs"

# connect_ex() results that mean "non-blocking connect is under way".
_IN_PROGRESS = frozenset({errno.EINPROGRESS, errno.EWOULDBLOCK})

_AddrInfo = tuple  # (family, type, proto, canonname, sockaddr) from getaddrinfo()


def _raise_errors(exceptions: list[BaseException], all_errors: bool) -> None:
    """Raise like the stdlib: the last error, or all of them as a group."""
    if not exceptions:
        raise OSError("getaddrinfo returns an empty list")
    try:
        if all_errors:
            raise ExceptionGroup("create_connection failed", exceptions)
        raise exceptions[-1]
    finally:
        exceptions.clear()  # break the reference cycle, as the stdlib does


def _connect_sequential(
    addrinfos: list[_AddrInfo], timeout: Any, all_errors: bool
) -> socket.socket:
    """The stdlib loop: try every address in resolver order."""
    exceptions: list[BaseException] = []
    for family, socktype, proto, _canonname, sockaddr in addrinfos:
        sock = None
        try:
            sock = socket.socket(family, socktype, proto)
            if timeout is not _DEFAULT_TIMEOUT:
                sock.settimeout(timeout)
            sock.connect(sockaddr)
            return sock
        except OSError as exc:
            if not all_errors:
                exceptions.clear()
            exceptions.append(exc)
            if sock is not None:
                sock.close()
    _raise_errors(exceptions, all_errors)
    raise AssertionError("unreachable")  # pragma: no cover


def _interleave(addrinfos: list[_AddrInfo]) -> list[_AddrInfo]:
    """RFC 8305 section 4: alternate families, the resolver's first first."""
    first = addrinfos[0][0]
    preferred = [info for info in addrinfos if info[0] == first]
    other = [info for info in addrinfos if info[0] != first]
    pairs = itertools.zip_longest(preferred, other)
    return [info for pair in pairs for info in pair if info is not None]


def _start_attempt(
    info: _AddrInfo, exceptions: list[BaseException]
) -> tuple[socket.socket | None, bool]:
    """Start a non-blocking connect.

    Returns ``(sock, True)`` when it connected at once, ``(sock, False)``
    when it is under way, and ``(None, False)`` when it failed at once
    (the error is appended to ``exceptions``).
    """
    family, socktype, proto, _canonname, sockaddr = info
    sock = None
    try:
        sock = socket.socket(family, socktype, proto)
        sock.setblocking(False)
        err = sock.connect_ex(sockaddr)
    except OSError as exc:
        if sock is not None:
            sock.close()
        exceptions.append(exc)
        return None, False
    if err == 0:
        return sock, True
    if err in _IN_PROGRESS:
        return sock, False
    sock.close()
    exceptions.append(OSError(err, os.strerror(err)))
    return None, False


def _won(sock: socket.socket, timeout: Any) -> socket.socket:
    """Give the winner the timeout mode the stdlib would have left on it."""
    if timeout is _DEFAULT_TIMEOUT:
        timeout = socket.getdefaulttimeout()
    sock.settimeout(timeout)
    return sock


def _connect_race(
    addrinfos: list[_AddrInfo], timeout: Any, all_errors: bool
) -> socket.socket:
    limit = socket.getdefaulttimeout() if timeout is _DEFAULT_TIMEOUT else timeout
    if limit is not None and limit <= 0:
        # A non-blocking connect request has no time to race in.
        return _connect_sequential(addrinfos, timeout, all_errors)

    queue = _interleave(addrinfos)
    exceptions: list[BaseException] = []
    pending: dict[socket.socket, float | None] = {}  # socket -> deadline
    next_start = 0.0
    selector = selectors.DefaultSelector()
    try:
        while queue or pending:
            now = time.monotonic()
            if queue and (not pending or now >= next_start):
                sock, connected = _start_attempt(queue.pop(0), exceptions)
                if connected:
                    return _won(sock, timeout)
                if sock is not None:
                    selector.register(sock, selectors.EVENT_WRITE)
                    pending[sock] = None if limit is None else now + limit
                    next_start = now + HAPPY_EYEBALLS_DELAY
                continue

            wakeups = [d for d in pending.values() if d is not None]
            if queue:
                wakeups.append(next_start)
            wait = max(0.0, min(wakeups) - now) if wakeups else None
            # With no timeout the kernel still ends every attempt (SYN
            # retries run out), so this wait is bounded like connect().
            for key, _events in selector.select(wait):
                sock = key.fileobj
                selector.unregister(sock)
                del pending[sock]
                try:
                    err = sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
                except OSError as exc:
                    err, failure = -1, exc
                else:
                    failure = OSError(err, os.strerror(err))
                if err == 0:
                    return _won(sock, timeout)
                sock.close()
                exceptions.append(failure)
                next_start = 0.0  # a failure starts the next address at once

            now = time.monotonic()
            for sock, deadline in list(pending.items()):
                if deadline is not None and now >= deadline:
                    selector.unregister(sock)
                    del pending[sock]
                    sock.close()
                    exceptions.append(TimeoutError("timed out"))
                    next_start = 0.0
    finally:
        selector.close()
        for sock in pending:  # every attempt that did not win
            sock.close()
    _raise_errors(exceptions, all_errors)
    raise AssertionError("unreachable")  # pragma: no cover


def connect_addrinfos(
    addrinfos: list[_AddrInfo],
    timeout: Any = _DEFAULT_TIMEOUT,
    *,
    all_errors: bool = False,
) -> socket.socket:
    """Connect to the first reachable address of ``getaddrinfo()`` results.

    Races the families when both IPv4 and IPv6 are present, otherwise
    keeps the stdlib's sequential order and errors.
    """
    families = {info[0] for info in addrinfos}
    if {socket.AF_INET, socket.AF_INET6} <= families:
        return _connect_race(addrinfos, timeout, all_errors)
    return _connect_sequential(addrinfos, timeout, all_errors)


def create_connection(
    address: Any,
    timeout: Any = _DEFAULT_TIMEOUT,
    source_address: Any = None,
    *,
    all_errors: bool = False,
) -> socket.socket:
    """``socket.create_connection`` replacement with Happy Eyeballs."""
    try:
        host, port = address
    except (TypeError, ValueError):
        host = port = None
    if source_address is not None or not isinstance(host, str):
        return _ORIGINAL_CREATE_CONNECTION(
            address, timeout, source_address, all_errors=all_errors
        )
    addrinfos = socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM)
    return connect_addrinfos(addrinfos, timeout, all_errors=all_errors)


def install() -> bool:
    """Route ``socket.create_connection`` through the racing version.

    Idempotent; returns ``True`` when the wrapper was newly installed.
    Call once at server start, before any outbound connection.
    """
    current = getattr(socket, "create_connection", None)
    if getattr(current, _INSTALLED_ATTR, False):
        return False
    setattr(create_connection, _INSTALLED_ATTR, True)  # noqa: B010
    socket.create_connection = create_connection  # type: ignore[assignment]
    return True
