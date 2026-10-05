# SPDX-License-Identifier: Apache-2.0
"""Inside a rank, carry its live RDMA stage edges through mailboxes; the rest stays on MLX."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, closing, contextmanager
from typing import Any

import numpy as np

from . import frames
from .daemon import read_status
from .mailbox import ClientMailbox, MailboxError, ServiceMailbox
from .stage_plan import StageLink, links_for_rank
from .words import WordOps, load_word_ops

logger = logging.getLogger(__name__)
# How long a rank waits for its neighbour before declaring the edge dead.
DEFAULT_TIMEOUT_S = 600.0
# Liveness of the link is rechecked at least this often while waiting.
_POLL_S = 1.0


class LinkDownError(RuntimeError):
    """The RDMA link under a stage edge stopped while a rank depended on it."""


def _dtype_name(dtype: Any) -> str:
    return str(dtype).rsplit(".", 1)[-1]


class StageReceiver:
    """Client end of an edge: requests each frame, pre-posting the next message's first request."""

    def __init__(
        self,
        mx: Any,
        mailbox: ClientMailbox,
        link: StageLink,
        *,
        timeout_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.peer_rank = link.sender_rank
        self._mx = mx
        self._mailbox = mailbox
        self._link = link
        self._timeout_s = timeout_s
        self._clock = clock
        self._message = 0
        self._pending: tuple[tuple[int, int], int] | None = None

    def _request(self, message: int, frame: int) -> int:
        return self._mailbox.stage((frames.pack(frames.request(message, frame)),))

    def _wait(self, seq: int) -> memoryview:
        deadline = self._clock() + self._timeout_s
        while True:
            reply = self._mailbox.wait(seq, _POLL_S)
            if reply is not None:
                return reply
            if not self._mailbox.connected:
                raise LinkDownError(
                    f"RDMA link {self._link.link} went down "
                    f"while waiting for rank {self.peer_rank}"
                )
            if self._mailbox.reconnected:
                raise LinkDownError(
                    f"RDMA link {self._link.link} reconnected while waiting for rank "
                    f"{self.peer_rank}; the request in flight was lost"
                )
            if self._clock() > deadline:
                raise TimeoutError(
                    f"rank {self.peer_rank} sent nothing over {self._link.link} "
                    f"for {self._timeout_s:.0f} s"
                )

    def recv_like(self, template: Any) -> Any:
        """The next stage message, shaped and typed like `template`."""
        shape = tuple(int(dim) for dim in template.shape)
        dtype = _dtype_name(template.dtype)
        total = int(template.nbytes)
        message = self._message + 1
        out = np.empty(total, dtype=np.uint8)
        frame, count, received = 0, 1, 0
        while frame < count:
            key = (message, frame)
            seq = (
                self._pending[1]
                if self._pending and self._pending[0] == key
                else self._request(*key)
            )
            self._pending = None
            reply = self._wait(seq)
            header = frames.unpack(reply)
            expected = (frames.FRAME, message, frame, total, dtype, shape, received)
            actual = (
                header.kind,
                header.message,
                header.frame,
                header.total,
                header.dtype,
                header.shape,
                header.offset,
            )
            if actual != expected:
                raise frames.FrameError(
                    f"stage frame mismatch over {self._link.link}: "
                    f"expected {expected}, got {actual}"
                )
            out[received : received + header.nbytes] = np.frombuffer(
                reply, dtype=np.uint8, count=header.nbytes, offset=frames.HEADER_BYTES
            )
            received += header.nbytes
            count = header.frames
            frame += 1
        if received != total:
            raise frames.FrameError(
                f"stage message over {self._link.link} carried {received} of {total} bytes"
            )
        self._message = message
        self._pending = ((message + 1, 0), self._request(message + 1, 0))
        return self._mx.array(out).view(template.dtype).reshape(shape)


class StageSender:
    """Service end of an edge: answers each frame request with the next slice of the message."""

    def __init__(
        self,
        mx: Any,
        mailbox: ServiceMailbox,
        link: StageLink,
        *,
        timeout_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.peer_rank = link.receiver_rank
        self._mx = mx
        self._mailbox = mailbox
        self._link = link
        self._timeout_s = timeout_s
        self._clock = clock
        self._message = 0

    def _await_request(self, message: int, frame: int) -> int:
        deadline = self._clock() + self._timeout_s
        while True:
            got = self._mailbox.next_request(_POLL_S)
            if got is not None:
                seq, payload = got
                header = frames.unpack(payload)
                if (header.kind, header.message, header.frame) != (
                    frames.REQUEST,
                    message,
                    frame,
                ):
                    raise frames.FrameError(
                        f"rank {self.peer_rank} asked for message {header.message} "
                        f"frame {header.frame}, not {message} frame {frame}"
                    )
                return seq
            if not self._mailbox.alive:
                raise LinkDownError(
                    f"mcdma-rpcd dropped the service end of {self._link.link}"
                )
            if self._clock() > deadline:
                raise TimeoutError(
                    f"rank {self.peer_rank} requested nothing over {self._link.link} "
                    f"for {self._timeout_s:.0f} s"
                )

    def send(self, array: Any) -> Any:
        """Serve `array` to the receiver frame by frame; returns it evaluated."""
        mx = self._mx
        flat = mx.contiguous(array).reshape(-1)
        mx.eval(flat)
        data = np.asarray(flat.view(mx.uint8))
        shape = tuple(int(dim) for dim in array.shape)
        plan = frames.plan(
            data.nbytes, self._mailbox.sizes.max_reply - frames.HEADER_BYTES
        )
        message = self._message + 1
        for index, (offset, nbytes) in enumerate(plan):
            seq = self._await_request(message, index)
            header = frames.Frame(
                frames.FRAME,
                message,
                index,
                len(plan),
                offset,
                data.nbytes,
                nbytes,
                _dtype_name(array.dtype),
                shape,
            )
            self._mailbox.reply(
                seq, (frames.pack(header), data[offset : offset + nbytes])
            )
        self._message = message
        return array


def _listen_link_up(socket_path: str, name: str) -> bool:
    """Whether the listen daemon serving `name` reports its link up; it accepts services either way."""
    peer = read_status(socket_path).peer(name)
    return peer is not None and peer.up


def _edge_state(link: StageLink, active: bool, reason: str) -> dict[str, Any]:
    return {**link.to_dict(), "active": active, "reason": reason}


@contextmanager
def install_stage_links(
    mx: Any,
    group: Any,
    links: tuple[StageLink, ...],
    *,
    rank: int,
    ops_loader: Callable[[], tuple[WordOps | None, str]] = load_word_ops,
    attach_client: Callable[[str, WordOps], ClientMailbox] = ClientMailbox.attach,
    attach_service: Callable[
        [str, str, WordOps], ServiceMailbox
    ] = ServiceMailbox.attach,
    link_up: Callable[[str, str], bool] = _listen_link_up,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> Iterator[dict[str, Any]]:
    """Agree with every rank on which edges are live, then carry this rank's live edges."""
    if not links:
        yield {
            "enabled": False,
            "active": False,
            "reason": "no verified RDMA stage link",
            "edges": [],
        }
        return
    incoming, outgoing = links_for_rank(links, rank)
    with ExitStack() as stack:
        client = service = None
        reasons = {"incoming": "", "outgoing": ""}
        # Every rank must reach the vote below, so local failures only lower this rank's vote.
        try:
            ops, why = ops_loader()
            if ops is None:
                reasons = {"incoming": why, "outgoing": why}
            else:
                if incoming is not None:
                    try:
                        client = stack.enter_context(
                            closing(attach_client(incoming.link, ops))
                        )
                        if not client.connected:
                            reasons["incoming"] = "mcdma-rpcd reports the link down"
                    except (MailboxError, ValueError, OSError) as exc:
                        reasons["incoming"] = str(exc)
                if outgoing is not None:
                    try:
                        service = stack.enter_context(
                            closing(
                                attach_service(
                                    outgoing.link, outgoing.service_socket, ops
                                )
                            )
                        )
                        if not link_up(outgoing.service_socket, outgoing.link):
                            reasons["outgoing"] = "mcdma-rpcd reports the link down"
                    except (MailboxError, ValueError, OSError) as exc:
                        reasons["outgoing"] = str(exc)
        except Exception as exc:
            logger.exception("RDMA stage link preparation failed on rank %d", rank)
            reasons = {"incoming": str(exc), "outgoing": str(exc)}
        local = [
            int(client is not None and not reasons["incoming"]),
            int(service is not None and not reasons["outgoing"]),
        ]
        # The vote also orders every attach before any request, so no live request looks stale.
        votes = mx.distributed.all_gather(mx.array(local, dtype=mx.int32), group=group)
        mx.eval(votes)
        flags = [int(value) for value in votes.tolist()]

        def ready(peer: int, slot: int) -> bool:
            return flags[2 * peer + slot] == 1

        edges = []
        receiver = sender = None
        if incoming is not None:
            live = ready(rank, 0) and ready(incoming.sender_rank, 1)
            reason = reasons["incoming"] or (
                "" if live else f"rank {incoming.sender_rank} could not attach its end"
            )
            edges.append(_edge_state(incoming, live, reason))
            if live and client is not None:
                receiver = StageReceiver(mx, client, incoming, timeout_s=timeout_s)
        if outgoing is not None:
            live = ready(outgoing.receiver_rank, 0) and ready(rank, 1)
            reason = reasons["outgoing"] or (
                ""
                if live
                else f"rank {outgoing.receiver_rank} could not attach its end"
            )
            edges.append(_edge_state(outgoing, live, reason))
            if live and service is not None:
                sender = StageSender(mx, service, outgoing, timeout_s=timeout_s)
        original_send = mx.distributed.send
        original_recv_like = mx.distributed.recv_like

        def send(array: Any, dst: int, *args: Any, **kwargs: Any) -> Any:
            if sender is not None and dst == sender.peer_rank:
                return sender.send(array)
            return original_send(array, dst, *args, **kwargs)

        def recv_like(array: Any, src: int, *args: Any, **kwargs: Any) -> Any:
            if receiver is not None and src == receiver.peer_rank:
                return receiver.recv_like(array)
            return original_recv_like(array, src, *args, **kwargs)

        mx.distributed.send = send
        mx.distributed.recv_like = recv_like
        active = receiver is not None or sender is not None
        try:
            yield {
                "enabled": True,
                "active": active,
                "reason": "stage activations cross RDMA"
                if active
                else "no live RDMA edge at this rank",
                "edges": edges,
            }
        finally:
            mx.distributed.send = original_send
            mx.distributed.recv_like = original_recv_like
