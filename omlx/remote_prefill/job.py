# SPDX-License-Identifier: Apache-2.0
"""One remote prefill: vLLM computes the prompt's KV cache and every rank's pages are pulled here."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..cluster.rdma.mailbox import ClientMailbox
from . import wire
from .client import request_prefill
from .receiver import HandoffError, HandoffReceiver
from .settings import RemotePrefillSettings

logger = logging.getLogger(__name__)
# Numpy types that copy each element width's bits unchanged.
_NUMPY_CARRIER = {2: np.uint16, 4: np.uint32}


@dataclass(frozen=True)
class Chunk:
    """One frame's pages, copied out of the mailbox and not yet interpreted."""

    rank: int
    layer: wire.LayerExport
    row_start: int
    rows: int
    data: Any


@dataclass(frozen=True)
class PrefillResult:
    """Everything pulled for one handoff."""

    manifests: tuple[wire.Manifest, ...]
    chunks: tuple[Chunk, ...]
    nbytes: int
    prefill_s: float
    transfer_s: float


def _pull(
    mx: Any,
    rank: int,
    mailbox: ClientMailbox,
    handoff: bytes,
    deadline: float,
    checksum: bool,
) -> tuple[wire.Manifest, list[Chunk], int]:
    """Pull rank `rank`'s export; each frame is copied into an MLX array before the next is asked for."""
    receiver = HandoffReceiver(mailbox, deadline=deadline, checksum=checksum)
    manifest = receiver.open(handoff)
    chunks, received = [], 0
    try:
        if manifest.tp_rank != rank:
            raise HandoffError(
                f"link {mailbox.name} serves rank {manifest.tp_rank}, not {rank}"
            )
        for layer, header, data in receiver.frames(handoff, manifest):
            carrier = _NUMPY_CARRIER[wire.DTYPE_BYTES[layer.dtype]]
            # The copy is materialized here; interpreting it waits for the engine's own stream.
            values = mx.array(np.frombuffer(data, dtype=carrier))
            chunks.append(Chunk(rank, layer, header.row_start, header.rows, values))
            received += header.nbytes
    except HandoffError:
        receiver.abandon(handoff)
        raise
    receiver.close(handoff)
    return manifest, chunks, received


class PrefillJob:
    """Runs one handoff on its own thread; the scheduler polls `running` and reads `result`."""

    def __init__(
        self,
        mx: Any,
        settings: RemotePrefillSettings,
        tokens: list[int],
        export_from: int,
        *,
        attach: Callable[[str], ClientMailbox],
        requester: Callable[..., None] = request_prefill,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.handoff = wire.new_handoff()
        self.tokens = list(tokens)
        self.export_from = export_from
        self.state = "running"
        self.error = ""
        self.result: PrefillResult | None = None
        self._mx = mx
        self._settings = settings
        self._attach = attach
        self._requester = requester
        self._clock = clock
        self._thread = threading.Thread(
            target=self._run, name="omlx-remote-prefill", daemon=True
        )

    @property
    def end(self) -> int:
        """Prompt tokens the handoff covers."""
        return len(self.tokens)

    @property
    def running(self) -> bool:
        return self.state == "running"

    def start(self) -> None:
        self._thread.start()

    def wait(self, timeout_s: float | None = None) -> None:
        self._thread.join(timeout_s)

    def _check(self, manifests: list[wire.Manifest]) -> None:
        digest = wire.token_sha256(self.tokens)
        layers = {layer.index for layer in manifests[0].layers}
        for manifest in manifests:
            if manifest.token_sha256 != digest or manifest.prompt_tokens != self.end:
                raise HandoffError("the producer prefilled a different prompt")
            if manifest.tp_size != len(manifests):
                raise HandoffError(
                    f"the producer runs {manifest.tp_size} ranks; {len(manifests)} links are set"
                )
            if manifest.first_token > self.export_from:
                raise HandoffError("the producer exported fewer tokens than asked for")
            if {layer.index for layer in manifest.layers} != layers:
                raise HandoffError("the producer's ranks exported different layers")

    def _run(self) -> None:
        settings = self._settings
        began = self._clock()
        deadline = began + settings.timeout_s
        try:
            # The pool drains before the mailboxes close, so no pull reads a closed mapping.
            with (
                ExitStack() as stack,
                ThreadPoolExecutor(max_workers=len(settings.links)) as pool,
            ):
                mailboxes = [
                    stack.enter_context(closing(self._attach(name)))
                    for name in settings.links
                ]
                for mailbox in mailboxes:
                    if not mailbox.connected:
                        raise HandoffError(f"RDMA link {mailbox.name} is down")
                self._requester(
                    settings, self.tokens, self.handoff.hex(), self.export_from
                )
                prefilled = self._clock()
                pulls = [
                    pool.submit(
                        _pull,
                        self._mx,
                        rank,
                        mailbox,
                        self.handoff,
                        deadline,
                        settings.checksum,
                    )
                    for rank, mailbox in enumerate(mailboxes)
                ]
                outcomes = [pull.result() for pull in pulls]
            manifests = [manifest for manifest, _, _ in outcomes]
            self._check(manifests)
            self.result = PrefillResult(
                manifests=tuple(manifests),
                chunks=tuple(chunk for _, chunks, _ in outcomes for chunk in chunks),
                nbytes=sum(received for _, _, received in outcomes),
                prefill_s=prefilled - began,
                transfer_s=self._clock() - prefilled,
            )
            self.state = "done"
        except Exception as exc:
            logger.warning("Remote prefill failed; prefilling locally: %s", exc)
            self.error = str(exc) or type(exc).__name__
            self.state = "failed"
