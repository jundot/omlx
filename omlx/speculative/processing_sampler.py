# SPDX-License-Identifier: Apache-2.0
"""Per-request logits processors on the vlm_mtp decode path.

Why this exists
===============

The vlm_mtp decode path (#2399) used to drop per-request logits processors
because ``run_vlm_mtp_decode`` threads only a *sampler* into mlx-vlm's
``_mtp_rounds``, and a bare sampler sees logits without the token history
processors need. The interim fix routed any request carrying processors
back to BatchGenerator, making e.g. ``thinking_budget`` and vlm_mtp
mutually exclusive.

mlx-vlm's verify walk, however, already exposes the application point we
need: when the sampler object provides ``sample_target(logprobs, row_ids,
positions)``, ``_positioned_target_tokens`` / the deferred-greedy walk call
it with the **absolute position of every candidate token**. Two properties
of the MTP round loop make processor application at that hook exact:

1. Every *emitted* token equals the target-side sample at its position —
   an accepted draft token is, by definition of the acceptance walk, equal
   to the target's sample, and a rejected slot emits the target's sample
   directly (``_speculative_walk``). So sampling from processor-modified
   logits at each position reproduces BatchGenerator semantics exactly.
2. Verify logits at slot ``i`` are conditioned on the draft prefix
   ``d[0:i]``; slot ``i``'s sample is only ever *used* when that prefix was
   accepted, in which case the wrapper's own sampled prefix equals
   ``d[0:i]``. Building each slot's history from the wrapper's own samples
   is therefore correct in every case where the slot's token survives.

Statefulness is the remaining hazard: positions past the first rejection
are discarded and re-sampled next round, so a stateful processor must be
rewound. The wrapper checkpoints processor state after every position
(``snapshot_state`` / ``restore_state`` on the processor) and restores the
checkpoint matching ``positions[0]`` at the start of every call. Rounds
only re-sample a *suffix* — committed positions are never revisited — so
the checkpoint for a call's base position always exists.

Cost: token values must be concrete before the next slot's processor call,
so the walk pays one small ``.item()`` sync per candidate slot (block_size
+ 1 per round, on tiny arrays). Draft-side sampling is unaffected — with a
positioned sampler present, mlx-vlm switches eligible drafters to greedy
argmax and never calls the sampler for drafts; any plain ``sampler(...)``
call falls through to the base sampler.

Scope: processors must expose ``snapshot_state()`` / ``restore_state()``
(today: ``ThinkingBudgetProcessor``). Grammar constraints stay on the
BatchGenerator fallback — beyond statefulness, a grammar mask makes the
unconstrained drafter's proposals systematically rejectable, so the
speculative path would buy nothing there anyway.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)


def supports_vlm_mtp_processing(processor: Any) -> bool:
    """True when a logits processor can be applied on the vlm_mtp path."""
    return callable(getattr(processor, "snapshot_state", None)) and callable(
        getattr(processor, "restore_state", None)
    )


class MTPProcessingSampler:
    """Sampler wrapper that applies logits processors at MTP verify time.

    Positions follow mlx-vlm's convention: the first bonus token (sampled
    from the post-prefill logits before the round loop starts) occupies
    position 0, and ``_mtp_rounds`` starts its verify walks at
    ``base_position = 1``.

    Call sequence:

    1. ``process_first_logits(logits)`` — apply processors to the
       post-prefill logits with ``history = prompt``.
    2. ``__call__(logits)`` (plain) — sample the first bonus from them.
    3. ``note_first_bonus(token)`` — append it; checkpoint position 1.
    4. mlx-vlm then calls ``sample_target`` once per verify walk.
    """

    def __init__(
        self,
        base_sampler: Callable[[Any], Any],
        logits_processors: Sequence[Any],
        prompt_token_ids: Sequence[int],
    ) -> None:
        self._base = base_sampler
        self._processors: list[Any] = list(logits_processors)
        self._history: list[int] = [int(t) for t in prompt_token_ids]
        self._prompt_len = len(self._history)
        self._snapshots: dict[int, list[dict]] = {}
        self._initial_snapshot = self._snap()
        self._degraded = False

    # -- plain sampler interface (drafter and non-verify call sites) --------

    def __call__(self, logits: Any) -> Any:
        return self._base(logits)

    # -- lifecycle ----------------------------------------------------------

    def process_first_logits(self, logits: Any) -> Any:
        """Apply processors to the post-prefill logits (history = prompt)."""
        processed = logits
        for proc in self._processors:
            processed = proc(self._history, processed)
        return processed

    def note_first_bonus(self, token: int, position: int = 1) -> None:
        """Record the emitted first-bonus token and checkpoint ``position``."""
        self._history.append(int(token))
        self._snapshots[int(position)] = self._snap()

    def reset_processors(self) -> None:
        """Rewind processors to their pre-routing state.

        Called when vlm_mtp routing fails *after* ``process_first_logits``
        already ran, so the BatchGenerator fallback sees fresh processors.
        """
        self._restore(self._initial_snapshot)
        del self._history[self._prompt_len :]
        self._snapshots.clear()

    # -- mlx-vlm positioned verify hook -------------------------------------

    def sample_target(
        self,
        logprobs: Any,
        row_ids: Sequence[int] | None = None,
        positions: Sequence[int] | None = None,
    ) -> Any:
        """Process-then-sample every candidate slot of a verify walk.

        ``logprobs`` is ``[n_slots, vocab]`` (or ``[vocab]`` for a single
        slot); ``positions`` carries each slot's absolute emitted-token
        position. Slots are processed in order, each with the token history
        reconstructed for its position; processor state is checkpointed
        after every slot so a later call can rewind to any re-sampled
        position.
        """
        if self._degraded or not self._processors:
            return self._base(logprobs)
        if not positions:
            # Positioned contract violated — never sample unprocessed
            # silently (#2399); degrade loudly.
            self._degrade("sample_target called without positions")
            return self._base(logprobs)

        if logprobs.ndim == 1:
            logprobs = logprobs[None, :]
        n_slots = int(logprobs.shape[0])
        if n_slots != len(positions):
            self._degrade(
                f"row/position mismatch (rows={n_slots}, "
                f"positions={len(positions)})"
            )
            return self._base(logprobs)

        base_pos = int(positions[0])
        snapshot = self._snapshots.get(base_pos)
        if snapshot is None:
            self._degrade(f"no checkpoint for base position {base_pos}")
            return self._base(logprobs)

        # Rewind: restore state at base_pos and drop the re-sampled suffix.
        self._restore(snapshot)
        del self._history[self._prompt_len + base_pos :]
        for key in [k for k in self._snapshots if k > base_pos]:
            del self._snapshots[key]

        mx.eval(logprobs)
        sampled: list[int] = []
        for i in range(n_slots):
            pos = int(positions[i])
            if pos != base_pos + i:
                self._degrade(f"non-contiguous positions ({positions!r})")
                # Finish this call unprocessed for the remaining slots.
                rest = self._base(logprobs[i:])
                rest_list = [int(t) for t in mx.reshape(rest, (-1,)).tolist()]
                return mx.array(sampled + rest_list, dtype=mx.int32)
            processed = logprobs[i][None, :]
            for proc in self._processors:
                processed = proc(self._history, processed)
            token_arr = self._base(processed)
            token = int(mx.reshape(token_arr, (-1,))[0].item())
            sampled.append(token)
            self._history.append(token)
            self._snapshots[pos + 1] = self._snap()
        return mx.array(sampled, dtype=mx.int32)

    # -- internals ----------------------------------------------------------

    def _snap(self) -> list[dict]:
        return [proc.snapshot_state() for proc in self._processors]

    def _restore(self, snapshot: list[dict]) -> None:
        for proc, state in zip(self._processors, snapshot):
            proc.restore_state(state)

    def _degrade(self, reason: str) -> None:
        self._degraded = True
        logger.warning(
            "MTPProcessingSampler degraded to plain sampling (%s) — "
            "per-request logits processors are NOT enforced for the rest "
            "of this request. This indicates an mlx-vlm contract change; "
            "please report it.",
            reason,
        )
