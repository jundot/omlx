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

from ..cluster.rdma.mailbox import ClientMailbox, MailboxError
from . import wire
from .client import request_prefill
from .receiver import (
    HandoffError,
    HandoffReceiver,
    HandoffStoppedError,
    HandoffTimeoutError,
)
from .settings import RemotePrefillSettings

logger = logging.getLogger(__name__)
# Numpy types that copy each element width's bits unchanged.
_NUMPY_CARRIER = {2: np.uint16, 4: np.uint32}
# A connector answers OPEN within milliseconds; this much silence means none serves the link.
_PRODUCER_CHECK_S = 5.0
# Once vLLM has answered, its connector registers the export within a step or two.
_READY_S = 30.0


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
    receiver: HandoffReceiver,
    handoff: bytes,
    ready_by: float,
    check: Callable[[int, wire.Manifest], None],
) -> tuple[wire.Manifest, list[Chunk], int]:
    """Pull rank `rank`'s export; each frame is copied into an MLX array before the next is asked for."""
    chunks, received = [], 0
    try:
        manifest = receiver.open(handoff, ready_by=ready_by)
        # A manifest for another prompt or model fails here, before any page crosses the link.
        check(rank, manifest)
        for layer, header, data in receiver.frames(handoff, manifest):
            carrier = _NUMPY_CARRIER[wire.DTYPE_BYTES[layer.dtype]]
            # The copy is materialized here; interpreting it waits for the engine's own stream.
            values = mx.array(np.frombuffer(data, dtype=carrier))
            chunks.append(Chunk(rank, layer, header.row_start, header.rows, values))
            received += header.nbytes
    except (HandoffError, MailboxError, wire.WireError):
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
        links: threading.Lock | None = None,
        check: Callable[[wire.Manifest], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.handoff = wire.new_handoff()
        self.tokens = list(tokens)
        self.export_from = export_from
        self.state = "running"
        self.error = ""
        # True when the failure means vLLM or its connector is missing or stuck.
        self.pause = False
        self.result: PrefillResult | None = None
        self._mx = mx
        self._settings = settings
        self._attach = attach
        self._requester = requester
        # Held while this job uses the links; two handoffs on one link would share its request slot.
        self._links = links or threading.Lock()
        self._model_check = check
        self._clock = clock
        self._halt = threading.Event()
        self._cancelled = False
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

    def cancel(self) -> None:
        """Stop between frames, free the producer's pages, and give the links back."""
        self._cancelled = True
        self._halt.set()

    def _check_manifest(self, rank: int, manifest: wire.Manifest) -> None:
        if manifest.tp_rank != rank:
            raise HandoffError(
                f"link {self._settings.links[rank]} serves rank {manifest.tp_rank}, not {rank}"
            )
        if (
            manifest.token_sha256 != wire.token_sha256(self.tokens)
            or manifest.prompt_tokens != self.end
        ):
            raise HandoffError("the producer prefilled a different prompt")
        if manifest.tp_size != len(self._settings.links):
            raise HandoffError(
                f"the producer runs {manifest.tp_size} ranks; "
                f"{len(self._settings.links)} links are set"
            )
        if manifest.first_token > self.export_from:
            raise HandoffError("the producer exported fewer tokens than asked for")
        if self._model_check is not None:
            self._model_check(manifest)

    @staticmethod
    def _check_ranks(manifests: list[wire.Manifest]) -> None:
        layers = {layer.index for layer in manifests[0].layers}
        for manifest in manifests:
            if {layer.index for layer in manifest.layers} != layers:
                raise HandoffError("the producer's ranks exported different layers")

    def _hold_links(self, deadline: float) -> ExitStack:
        """The links, once no other handoff is using them."""
        stack = ExitStack()
        if not self._links.acquire(timeout=max(0.0, deadline - self._clock())):
            raise HandoffTimeoutError(
                "another remote prefill kept the links until the deadline"
            )
        stack.callback(self._links.release)
        return stack

    def _receivers(
        self, mailboxes: list[ClientMailbox], deadline: float
    ) -> list[HandoffReceiver]:
        return [
            HandoffReceiver(
                mailbox,
                deadline=deadline,
                checksum=self._settings.checksum,
                halt=self._halt,
                clock=self._clock,
            )
            for mailbox in mailboxes
        ]

    def _run(self) -> None:
        settings = self._settings
        began = self._clock()
        deadline = began + settings.timeout_s
        try:
            with ExitStack() as stack:
                mailboxes = [
                    stack.enter_context(closing(self._attach(name)))
                    for name in settings.links
                ]
                for mailbox in mailboxes:
                    if not mailbox.connected:
                        raise HandoffError(f"RDMA link {mailbox.name} is down")
                with self._hold_links(deadline):
                    probe_by = min(deadline, self._clock() + _PRODUCER_CHECK_S)
                    for receiver in self._receivers(mailboxes, probe_by):
                        receiver.probe(self.handoff)
                if self._halt.is_set():
                    raise HandoffStoppedError("the remote prefill was stopped")
                self._requester(
                    settings, self.tokens, self.handoff.hex(), self.export_from
                )
                prefilled = self._clock()
                with self._hold_links(deadline):
                    receivers = self._receivers(mailboxes, deadline)
                    if self._halt.is_set():
                        # vLLM holds the pages until they are closed or expire.
                        for receiver in receivers:
                            receiver.abandon(self.handoff)
                        raise HandoffStoppedError("the remote prefill was stopped")
                    outcomes = self._pull_all(receivers, prefilled + _READY_S)
            manifests = [manifest for manifest, _, _ in outcomes]
            self._check_ranks(manifests)
            self.result = PrefillResult(
                manifests=tuple(manifests),
                chunks=tuple(chunk for _, chunks, _ in outcomes for chunk in chunks),
                nbytes=sum(received for _, _, received in outcomes),
                prefill_s=prefilled - began,
                transfer_s=self._clock() - prefilled,
            )
            self.state = "done"
        except Exception as exc:
            if self._cancelled:
                self.state = "cancelled"
                return
            logger.warning("Remote prefill failed; prefilling locally: %s", exc)
            self.error = str(exc) or type(exc).__name__
            self.pause = isinstance(exc, HandoffTimeoutError)
            self.state = "failed"

    def _pull_all(
        self, receivers: list[HandoffReceiver], ready_by: float
    ) -> list[tuple[wire.Manifest, list[Chunk], int]]:
        """Every rank's pull in parallel; the first failure stops the others between frames."""

        def pull(rank: int, receiver: HandoffReceiver) -> Any:
            try:
                return _pull(
                    self._mx,
                    rank,
                    receiver,
                    self.handoff,
                    ready_by,
                    self._check_manifest,
                )
            except Exception:
                self._halt.set()
                raise

        # The pool drains before the mailboxes close, so no pull reads a closed mapping.
        with ThreadPoolExecutor(max_workers=len(receivers)) as pool:
            pulls = [
                pool.submit(pull, rank, receiver)
                for rank, receiver in enumerate(receivers)
            ]
        errors = [pull.exception() for pull in pulls if pull.exception() is not None]
        # Ranks halted by another rank's failure report that failure, not the halt.
        for error in sorted(
            errors, key=lambda error: isinstance(error, HandoffStoppedError)
        ):
            raise error
        return [pull.result() for pull in pulls]
