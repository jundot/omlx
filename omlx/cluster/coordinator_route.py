# SPDX-License-Identifier: Apache-2.0
"""Where each rank reaches rank zero.

Rank zero accepts two bootstrap connections from every other rank: JACCL's
metadata exchange and the rank-control channel. ``mlx.launch`` names rank zero
once for the whole job, by the first communication IP of its hostfile entry.
One address is enough while a single subnet joins every Mac. Macs cabled to
each other directly share a different subnet on every cable, and rank zero's
address on its cable to rank one is not one rank two can reach.

A launch that needs more than one address lists them in an environment
variable, one entry per rank, and each rank reads the entry at its own index.
"""

from __future__ import annotations

import ipaddress
import os
from collections.abc import Sequence

COORDINATOR_ADDRESSES_ENV = "OMLX_COORDINATOR_ADDRESSES"


def encode_coordinator_addresses(addresses: Sequence[str]) -> str:
    """The environment value that names rank zero for every rank, in rank order."""

    return ",".join(addresses)


def coordinator_addresses(world_size: int, default: str) -> tuple[str, ...]:
    """The address each rank reaches rank zero on, indexed by rank.

    Without the variable every rank uses ``default``, the one address the
    launcher announced.
    """

    raw = os.environ.get(COORDINATOR_ADDRESSES_ENV, "").strip()
    if not raw:
        return (default,) * world_size
    addresses = tuple(part.strip() for part in raw.split(","))
    if len(addresses) != world_size:
        raise RuntimeError(
            f"{COORDINATOR_ADDRESSES_ENV} names rank zero for {len(addresses)} "
            f"ranks, but the job has {world_size}"
        )
    for address in addresses:
        try:
            # Both bootstrap listeners are IPv4 sockets.
            ipaddress.IPv4Address(address)
        except ValueError as exc:
            raise RuntimeError(
                f"{COORDINATOR_ADDRESSES_ENV} holds an invalid IPv4 address: "
                f"{address!r}"
            ) from exc
    return addresses


def listen_addresses(addresses: Sequence[str]) -> tuple[str, ...]:
    """Where rank zero accepts the other ranks: each address they use, once.

    Listening on these alone, and not on every interface, keeps a bootstrap
    socket off any network no rank was told to use.
    """

    return tuple(dict.fromkeys(addresses[1:]))
