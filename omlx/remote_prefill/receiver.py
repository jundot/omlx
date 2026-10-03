# SPDX-License-Identifier: Apache-2.0
"""Pull one producer rank's exported pages over its MCDMA link: manifest, frames, close."""

from __future__ import annotations

import json
import threading
import time
import zlib
from collections.abc import Callable, Iterator
from contextlib import suppress

from ..cluster.rdma.mailbox import ClientMailbox, MailboxError
from . import wire

# A waiting call rechecks the link at least this often.
_POLL_S = 0.5
# Pause before asking again for a manifest the producer does not have yet.
_RETRY_S = 0.005
# A halted handoff still waits this long for the answer in flight, so the link's request slot is free after it.
_DRAIN_S = 2.0


class HandoffError(RuntimeError):
    """A handoff could not be completed; the request prefills locally instead."""


class HandoffTimeoutError(HandoffError):
    """vLLM or its connector stayed silent, so remote prefill pauses at once instead of waiting again."""


class HandoffStoppedError(HandoffError):
    """The handoff was stopped because its request left the queue or another rank's pull failed."""


class HandoffReceiver:
    """The decoder's end of one producer link, used for one handoff at a time."""

    def __init__(
        self,
        mailbox: ClientMailbox,
        *,
        deadline: float,
        checksum: bool = True,
        halt: threading.Event | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._mailbox = mailbox
        self._deadline = deadline
        self._checksum = checksum
        self._halt = halt or threading.Event()
        self._clock = clock
        # False while a call is unanswered: the link's single request slot is still taken.
        self._idle = True

    def _check_halt(self) -> None:
        if self._halt.is_set():
            raise HandoffStoppedError("the remote prefill was stopped")

    def _call(
        self, header: wire.Header, payload: bytes = b""
    ) -> tuple[wire.Header, memoryview]:
        self._check_halt()
        self._idle = False
        seq = self._mailbox.stage((wire.pack(header), payload))
        drain_until = None
        while True:
            reply = self._mailbox.wait(seq, _POLL_S)
            if reply is not None:
                self._idle = True
                answer = wire.unpack(reply)
                if answer.handoff != header.handoff:
                    raise HandoffError("the producer answered for another handoff")
                if answer.kind == wire.ERROR:
                    reason = bytes(wire.body(reply, answer)).decode(errors="replace")
                    raise HandoffError(f"the producer refused the handoff: {reason}")
                return answer, reply
            if not self._mailbox.connected:
                raise HandoffError(f"RDMA link {self._mailbox.name} went down")
            if self._mailbox.reconnected:
                raise HandoffError(
                    f"RDMA link {self._mailbox.name} reconnected; the call in flight was lost"
                )
            if self._halt.is_set():
                drain_until = drain_until or self._clock() + _DRAIN_S
                if self._clock() > drain_until:
                    raise HandoffStoppedError("the remote prefill was stopped")
            if self._clock() > self._deadline:
                raise HandoffTimeoutError(
                    f"no answer over {self._mailbox.name} before the deadline"
                )

    def _open_request(self, handoff: bytes) -> tuple[wire.Header, bytes]:
        request = json.dumps({"checksum": self._checksum}).encode()
        return wire.Header(wire.OPEN, handoff, nbytes=len(request)), request

    def probe(self, handoff: bytes) -> None:
        """Raise unless a producer serves this link; before vLLM has the prompt, OPEN is answered WAIT."""
        try:
            self._call(*self._open_request(handoff))
        except HandoffTimeoutError as exc:
            raise HandoffTimeoutError(
                f"no MCDMA connector answered on {self._mailbox.name}; "
                "is vLLM running with the MCDMA KV connector?"
            ) from exc

    def open(self, handoff: bytes, *, ready_by: float | None = None) -> wire.Manifest:
        """The handoff's manifest, asked for again until the producer has it or `ready_by` passes."""
        header, request = self._open_request(handoff)
        give_up = self._deadline if ready_by is None else min(self._deadline, ready_by)
        while True:
            answer, reply = self._call(header, request)
            if answer.kind == wire.MANIFEST:
                return wire.Manifest.from_json(wire.body(reply, answer))
            if answer.kind != wire.WAIT:
                raise HandoffError(
                    f"the producer answered OPEN with kind {answer.kind}"
                )
            if self._clock() > give_up:
                raise HandoffTimeoutError("the producer never had the handoff ready")
            time.sleep(_RETRY_S)

    def frames(
        self, handoff: bytes, manifest: wire.Manifest
    ) -> Iterator[tuple[wire.LayerExport, wire.Header, memoryview]]:
        """Every frame in order; a frame's bytes stay valid only until the next is asked for."""
        layers = {layer.index: layer for layer in manifest.layers}
        for frame in range(manifest.frames):
            answer, reply = self._call(wire.Header(wire.PULL, handoff, frame=frame))
            layer = layers.get(answer.layer)
            if (
                answer.kind != wire.DATA
                or answer.frame != frame
                or answer.frames != manifest.frames
                or layer is None
            ):
                raise HandoffError(
                    f"frame {frame} came back as kind {answer.kind} for frame "
                    f"{answer.frame} of {answer.frames}, layer {answer.layer}"
                )
            total = layer.shape[layer.dims.index("block")]
            if (
                answer.rows < 1
                or answer.row_start + answer.rows > total
                or answer.nbytes != answer.rows * layer.row_bytes
            ):
                raise HandoffError(f"frame {frame} does not fit layer {layer.index}")
            data = wire.body(reply, answer)
            if self._checksum and (
                not answer.flags & wire.CHECKED or zlib.crc32(data) != answer.crc
            ):
                raise HandoffError(f"frame {frame} failed its checksum")
            yield layer, answer, data

    def close(self, handoff: bytes) -> None:
        """Let the producer free the handoff's pages."""
        answer, _ = self._call(wire.Header(wire.CLOSE, handoff))
        if answer.kind != wire.ACK:
            raise HandoffError(f"the producer answered CLOSE with kind {answer.kind}")

    def abandon(self, handoff: bytes) -> None:
        """Close a handoff that failed, when the link's request slot is free to say so."""
        if self._idle:
            # Closing is how a stopped or late handoff frees the producer's pages, so it gets its own time.
            self._halt = threading.Event()
            self._deadline = self._clock() + _DRAIN_S
            with suppress(HandoffError, MailboxError, wire.WireError):
                self.close(handoff)
