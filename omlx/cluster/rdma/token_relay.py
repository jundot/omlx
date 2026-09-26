# SPDX-License-Identifier: Apache-2.0
"""Carry rank zero's sampled tokens up the pipeline over the stage links instead of a ring all-sum."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .stage_transport import StageReceiver, StageSender


class TokenRelay:
    """Rank zero hands each decode step's tokens to rank 1, and every rank passes them on."""

    def __init__(
        self,
        receiver: StageReceiver | None,
        sender: StageSender | None,
        *,
        ready: bool,
        reason: str,
        report: dict[str, Any],
    ) -> None:
        self._receiver = receiver
        self._sender = sender
        self._report = report
        # Every rank computes this from the same vote, so all ranks reach the same decision.
        self.ready = ready
        self.active = False
        self._record(
            False, "waiting for the rank-zero sampling decision" if ready else reason
        )

    def _record(self, active: bool, reason: str) -> None:
        self._report["token_relay"] = {"active": active, "reason": reason}

    def activate(self, sampling_active: bool) -> bool:
        """Use the relay only when every edge is live and rank zero samples alone."""
        self.active = self.ready and sampling_active
        if self.active:
            # Each token request now also asks for the next message, so none is pre-posted.
            if self._receiver is not None:
                self._receiver.prepost = False
            self._record(True, "sampled tokens cross RDMA rank by rank")
        elif self.ready:
            self._record(False, "rank-zero sampling is off, so tokens stay on the ring")
        return self.active

    def broadcast(self, mx: Any, sampled: Any, count: int) -> Any:
        """Rank zero's `sampled` tokens on every rank, as the ring all-sum would return them."""
        if self._sender is None:
            mx.eval(sampled)
            values = np.array(sampled, dtype=np.uint32)
            result = sampled
        else:
            values = self._sender.take_tokens(count)
            result = mx.array(values, dtype=sampled.dtype)
        if self._receiver is not None:
            self._receiver.send_tokens(values)
        return result
