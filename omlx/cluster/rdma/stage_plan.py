# SPDX-License-Identifier: Apache-2.0
"""Which pipeline stage edges ride a verified RDMA link: the contract entries ranks read."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from . import layout

_MAX_STAGE_LINKS = 64
# sun_path holds 104 bytes on macOS and 108 on Linux, including the terminating NUL.
_MAX_SOCKET_BYTES = 103


@dataclass(frozen=True)
class StageLink:
    """Rank `sender_rank` sends its stage output to `receiver_rank` through mailbox `link`."""

    sender_rank: int
    receiver_rank: int
    link: str
    service_socket: str

    def __post_init__(self) -> None:
        for rank in (self.sender_rank, self.receiver_rank):
            if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
                raise ValueError("stage link ranks must be non-negative integers")
        if self.receiver_rank != self.sender_rank - 1:
            raise ValueError("a stage link must join a rank to the rank before it")
        layout.valid_link_name(self.link)
        socket_path = self.service_socket
        if (
            not isinstance(socket_path, str)
            or "\x00" in socket_path
            or len(socket_path.encode()) > _MAX_SOCKET_BYTES
            or not PurePosixPath(socket_path).is_absolute()
        ):
            raise ValueError(
                "stage link service socket must be an absolute Unix socket path"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sender_rank": self.sender_rank,
            "receiver_rank": self.receiver_rank,
            "link": self.link,
            "service_socket": self.service_socket,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> StageLink:
        if not isinstance(payload, dict):
            raise ValueError("stage link must be an object")
        return cls(
            sender_rank=payload.get("sender_rank"),
            receiver_rank=payload.get("receiver_rank"),
            link=payload.get("link"),
            service_socket=payload.get("service_socket"),
        )


def validate_stage_links(
    raw: Any, world_size: int | None = None
) -> tuple[StageLink, ...]:
    """Parse contract stage links; each rank may appear at most once per direction."""
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)) or len(raw) > _MAX_STAGE_LINKS:
        raise ValueError("stage_links must be a list of at most 64 entries")
    links = tuple(
        item if isinstance(item, StageLink) else StageLink.from_dict(item)
        for item in raw
    )
    senders = [item.sender_rank for item in links]
    if len(set(senders)) != len(senders) or len({item.link for item in links}) != len(
        links
    ):
        raise ValueError("stage_links repeats a rank or a link")
    if world_size is not None and any(item.sender_rank >= world_size for item in links):
        raise ValueError("stage_links names a rank outside the deployment")
    return links


def links_for_rank(
    links: Iterable[StageLink], rank: int
) -> tuple[StageLink | None, StageLink | None]:
    """This rank's (incoming, outgoing) stage links."""
    known = tuple(links)
    incoming = next((item for item in known if item.receiver_rank == rank), None)
    outgoing = next((item for item in known if item.sender_rank == rank), None)
    return incoming, outgoing
