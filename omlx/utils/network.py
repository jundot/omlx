# SPDX-License-Identifier: Apache-2.0
"""Network interface and hostname detection utilities.

Used to auto-populate ``ServerSettings.server_aliases`` so the admin
dashboard can offer dynamic API URL hints (Tailscale mDNS, LAN IP,
localhost, etc.) without manual configuration.
"""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
from collections.abc import Iterable

logger = logging.getLogger(__name__)

# RFC 1123 hostname label: letters, digits, hyphens; 1-63 chars per label.
# Allows trailing dot. Total length capped at 253.
_HOSTNAME_LABEL = re.compile(r"^(?!-)[A-Za-z0-9-]{1,63}(?<!-)$")


def is_valid_hostname(value: str) -> bool:
    """Return True if ``value`` looks like a valid DNS hostname."""
    if not value or len(value) > 253:
        return False
    candidate = value[:-1] if value.endswith(".") else value
    return all(_HOSTNAME_LABEL.match(label) for label in candidate.split("."))


def is_valid_ip(value: str) -> bool:
    """Return True if ``value`` is a usable IPv4 or IPv6 alias address.

    Rejects unspecified bind addresses (``0.0.0.0`` and ``::``) since they
    are not routable as client-facing URL hosts even though they parse as
    valid IP addresses.
    """
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return False
    return not ip.is_unspecified


def is_valid_alias(value: str) -> bool:
    """Validate that ``value`` is a hostname or routable IP address.

    If the value parses as an IP address at all, the IP validity check is
    authoritative — we do not silently fall through to hostname matching.
    Without this guard, an IP-shaped string like ``0.0.0.0`` would slip
    through as a "valid hostname" (digit-only labels are legal) even after
    being rejected as an unspecified bind address by :func:`is_valid_ip`.
    """
    if not isinstance(value, str):
        return False
    value = value.strip()
    if not value:
        return False
    try:
        ipaddress.ip_address(value)
    except ValueError:
        return is_valid_hostname(value)
    return is_valid_ip(value)


def _local_ipv4_addresses() -> list[str]:
    """Best-effort enumeration of non-loopback IPv4 addresses.

    Tries ``psutil`` (most accurate, multi-interface) first, then falls
    back to ``socket.getaddrinfo`` against the local hostname.
    """
    addresses: list[str] = []

    try:
        import psutil  # type: ignore

        for iface_addrs in psutil.net_if_addrs().values():
            for addr in iface_addrs:
                if getattr(addr, "family", None) == socket.AF_INET:
                    ip = addr.address
                    try:
                        if not ipaddress.ip_address(ip).is_loopback:
                            addresses.append(ip)
                    except ValueError:
                        continue
    except Exception as exc:  # pragma: no cover - psutil unavailable
        logger.debug("psutil unavailable for IP discovery: %s", exc)

    if not addresses:
        try:
            host = socket.gethostname()
            for info in socket.getaddrinfo(host, None, family=socket.AF_INET):
                ip = info[4][0]
                try:
                    if not ipaddress.ip_address(ip).is_loopback:
                        addresses.append(ip)
                except ValueError:
                    continue
        except OSError as exc:
            logger.debug("getaddrinfo fallback failed: %s", exc)

    return addresses


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def detect_server_aliases(host: str = "127.0.0.1") -> list[str]:
    """Detect candidate aliases for the running server.

    Args:
        host: The configured server bind host. Used to decide whether
            to include ``localhost``/``127.0.0.1`` first.

    Returns:
        Ordered, de-duplicated list of valid aliases. Order favors
        commonly accessible names: localhost, hostname, mDNS (.local),
        FQDN, then any non-loopback IPv4 addresses.
    """
    candidates: list[str] = []

    # Always offer loopback first when bound to localhost or all interfaces.
    if host in ("127.0.0.1", "localhost", "0.0.0.0", "::"):
        candidates.append("localhost")
        candidates.append("127.0.0.1")

    try:
        hostname = socket.gethostname()
        if hostname:
            candidates.append(hostname)
            # Add Bonjour/mDNS form if not already present.
            if not hostname.endswith(".local"):
                candidates.append(f"{hostname}.local")
    except OSError as exc:
        logger.debug("gethostname failed: %s", exc)

    try:
        fqdn = socket.getfqdn()
        # Skip reverse-DNS PTR records (e.g. "...ip6.arpa", "...in-addr.arpa")
        # which are not user-friendly and not routable as URLs.
        if fqdn and not fqdn.endswith((".ip6.arpa", ".in-addr.arpa")):
            candidates.append(fqdn)
    except OSError as exc:
        logger.debug("getfqdn failed: %s", exc)

    candidates.extend(_local_ipv4_addresses())

    # Filter to valid aliases only and dedupe while preserving order.
    valid = [c for c in candidates if is_valid_alias(c)]
    return _dedupe_preserve_order(valid)
