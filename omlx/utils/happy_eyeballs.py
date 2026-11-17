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

``install()`` wraps ``socket.create_connection`` with a two-family race:
the first IPv6 candidate gets a 300 ms head start, then the first IPv4
candidate joins in parallel; the first success wins and the loser is
closed.  Single-family names, explicit ``source_address`` bindings and
non-(host, port) addresses delegate to the original implementation
unchanged, so behaviour on healthy or IPv4-only networks is identical.
"""

from __future__ import annotations

import socket
import threading
from typing import Any, Optional

#: How long to let the resolver-preferred family (IPv6 on dual-stack
#: hosts) connect alone before starting the other family in parallel.
#: Same value browsers use (RFC 8305 recommends ~300 ms).
HAPPY_EYEBALLS_DELAY = 0.3

#: Sentinel for "caller did not pass a timeout", mirroring the semantics
#: of ``socket._GLOBAL_DEFAULT_TIMEOUT``: leave the socket timeout alone.
_UNSET = object()

# The stdlib sentinel, if it exists; comparing against it keeps callers
# that pass it explicitly working unchanged.
_STDLIB_DEFAULT_TIMEOUT = getattr(socket, "_GLOBAL_DEFAULT_TIMEOUT", _UNSET)

_ORIGINAL_CREATE_CONNECTION = socket.create_connection

_INSTALLED_ATTR = "_omlx_happy_eyeballs"

_AddrInfo = tuple  # (family, type, proto, canonname, sockaddr) as returned by getaddrinfo()


def _normalize_timeout(timeout: Any) -> Any:
    if timeout is _STDLIB_DEFAULT_TIMEOUT:
        return _UNSET
    return timeout


def _connect_one(info: _AddrInfo, timeout: Any) -> tuple[Optional[socket.socket], Optional[BaseException]]:
    """Connect a single addrinfo tuple.

    Returns ``(socket, None)`` on success or ``(None, error)`` on
    failure; never raises.
    """
    family, socktype, proto, _canonname, sockaddr = info
    sock: Optional[socket.socket] = None
    try:
        sock = socket.socket(family, socktype, proto)
        if timeout is not _UNSET:
            sock.settimeout(timeout)
        sock.connect(sockaddr)
        return sock, None
    except OSError as exc:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        return None, exc


def _connect_sequential(addrinfos: list[_AddrInfo], timeout: Any) -> socket.socket:
    """Original stdlib semantics: try every address in resolver order,
    raise the last error."""
    last_exc: Optional[BaseException] = None
    for info in addrinfos:
        sock, exc = _connect_one(info, timeout)
        if sock is not None:
            return sock
        last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise OSError("getaddrinfo returns an empty list")


def connect_addrinfos(addrinfos: list[_AddrInfo], timeout: Any = _UNSET) -> socket.socket:
    """Connect to the first reachable address using Happy Eyeballs.

    ``addrinfos`` is a list of tuples as returned by
    ``socket.getaddrinfo``.  When both families are represented, the
    first IPv6 candidate gets :data:`HAPPY_EYEBALLS_DELAY` seconds alone
    before the first IPv4 candidate is attempted in parallel; the first
    success wins and the loser's socket is closed.  If both racing
    candidates fail, the remaining addresses are tried sequentially
    (original semantics) before the last error is raised.
    """
    timeout = _normalize_timeout(timeout)
    if not addrinfos:
        raise OSError("getaddrinfo returns an empty list")

    v6 = [i for i in addrinfos if i[0] == socket.AF_INET6]
    v4 = [i for i in addrinfos if i[0] == socket.AF_INET]
    if not v6 or not v4:
        # Single family: nothing to race, keep stdlib order and errors.
        return _connect_sequential(addrinfos, timeout)

    lock = threading.Lock()
    winner: Optional[socket.socket] = None
    errors: dict[str, BaseException] = {}
    done = {"v6": False, "v4": False}
    v4_started = False
    first_done = threading.Event()  # first racing attempt produced a result
    settled = threading.Event()  # success, or both racing attempts failed

    def attempt(key: str, info: _AddrInfo, signal_first: bool) -> None:
        nonlocal winner
        try:
            sock, exc = _connect_one(info, timeout)
        except BaseException as unexpected:  # defensive: never strand the waiter
            sock, exc = None, unexpected
        with lock:
            if sock is not None:
                if winner is None:
                    winner = sock
                else:
                    try:
                        sock.close()
                    except OSError:
                        pass
            elif exc is not None:
                errors[key] = exc
            done[key] = True
            if winner is not None or (done["v6"] and done["v4"]):
                settled.set()
        if signal_first:
            first_done.set()

    t6 = threading.Thread(
        target=attempt, args=("v6", v6[0], True), daemon=True, name="omlx-happy-eyeballs-v6"
    )
    t6.start()

    # Give the resolver-preferred family a head start, but join the race
    # early if it fails fast.
    first_done.wait(HAPPY_EYEBALLS_DELAY)
    with lock:
        sock = winner
        if sock is None and not v4_started:
            v4_started = True
            t4 = threading.Thread(
                target=attempt,
                args=("v4", v4[0], False),
                daemon=True,
                name="omlx-happy-eyeballs-v4",
            )
            t4.start()
    if sock is not None:
        return sock

    settled.wait()
    with lock:
        sock = winner
        last_exc = errors.get("v4") or errors.get("v6")
    if sock is not None:
        return sock

    # Both racing candidates failed: sweep the remaining addresses with
    # the original sequential logic so we do not try fewer addresses than
    # the stdlib would have.
    rest = [i for i in addrinfos if i is not v6[0] and i is not v4[0]]
    if rest:
        try:
            return _connect_sequential(rest, timeout)
        except BaseException as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise OSError("all connection attempts failed")


def create_connection(
    address: Any, timeout: Any = _UNSET, source_address: Any = None
) -> socket.socket:
    """``socket.create_connection`` replacement with Happy Eyeballs."""
    if source_address is not None:
        if timeout is _UNSET:
            return _ORIGINAL_CREATE_CONNECTION(address, source_address=source_address)
        return _ORIGINAL_CREATE_CONNECTION(address, timeout, source_address=source_address)
    try:
        host, port = address
    except (TypeError, ValueError):
        # Let the original raise its exact error for malformed addresses.
        if timeout is _UNSET:
            return _ORIGINAL_CREATE_CONNECTION(address)
        return _ORIGINAL_CREATE_CONNECTION(address, timeout)
    if not isinstance(host, str):
        if timeout is _UNSET:
            return _ORIGINAL_CREATE_CONNECTION(address)
        return _ORIGINAL_CREATE_CONNECTION(address, timeout)
    timeout = _normalize_timeout(timeout)
    addrinfos = socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM)
    return connect_addrinfos(addrinfos, timeout)


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
