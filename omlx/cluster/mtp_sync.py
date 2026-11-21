# SPDX-License-Identifier: Apache-2.0
"""Rank zero's Lightning MTP decisions, replayed by every other rank.

Only rank zero holds the head and samples. The other ranks receive its verify
tokens, accepted counts and sampled tokens over the rank-control channel, in
the same order as request sharing. Each message carries a seed every rank
applies, because drafting draws from rank zero's RNG alone.
"""

from __future__ import annotations

import os
from typing import Any

#: Families whose Lightning MTP chain is proven inside a cluster (a real
#: deployment, not a plan). ``qwen4_exp``: live on 2026-11-20 (54.70 tok/s
#: against 49.26 with MTP off). ``qwen3_5_moe``: proven in the same series.
#:
#: ``glm5_next`` stamps a chain marker of its own but deadlocks in a cluster:
#: both ranks arm MTP, then rank 1 waits ~120 s inside ``_share_object`` while
#: the canary never returns (two runs, identical). Until that handshake exists,
#: cluster MTP is refused by name for the family instead of hanging.
CLUSTER_MTP_FAMILIES = frozenset({"qwen4_exp", "qwen3_5_moe"})


def cluster_mtp_family_supported(model_type: str | None) -> bool:
    """Whether cluster Lightning MTP may arm for this architecture family.

    A text-only load reports the family with a ``_text`` suffix
    (``qwen4_exp_text``, ``glm5_next_text``), while the catalogue and the
    planner use the bare name. Both spellings are the same architecture, so the
    suffix is stripped before the lookup instead of refusing a proven family by
    accident.
    """

    family = str(model_type or "").strip().lower()
    if family.endswith("_text"):
        family = family[: -len("_text")]
    return family in CLUSTER_MTP_FAMILIES


def _new_seed() -> int:
    return int.from_bytes(os.urandom(8), "little") >> 1


class ClusterMtpSync:
    """The decision channel one rank's batch generator reads during MTP."""

    def __init__(self, *, rank: int, control_plane: Any | None = None) -> None:
        self.rank = int(rank)
        self.control_plane = control_plane

    @property
    def is_leader(self) -> bool:
        return self.rank == 0

    def _broadcast(self, kind: str, *fields: Any) -> tuple[Any, ...]:
        plane = self.control_plane
        if plane is None:
            raise RuntimeError("cluster Lightning MTP needs the rank-control channel")
        message = plane.broadcast_object((kind, *fields) if self.is_leader else None)
        if not isinstance(message, tuple) or not message or message[0] != kind:
            got = message[0] if isinstance(message, tuple) and message else message
            raise RuntimeError(f"cluster MTP expected {kind!r}, got {got!r}")
        return message[1:]

    def share_tokens(self, token_ids: list[int] | None) -> tuple[list[int], int]:
        """Rank zero's sampled tokens, plus a seed."""

        ids, seed = self._broadcast("tokens", list(token_ids or ()), _new_seed())
        return [int(token) for token in ids], int(seed)

    def share_cycle_inputs(
        self,
        next_main: list[int] | None,
        drafts: list[int] | None,
        *,
        fallback: str | None = None,
    ) -> tuple[list[int], list[int], str | None]:
        """Rank zero's verify inputs, or the reason it leaves MTP instead."""

        main_ids, draft_ids, reason = self._broadcast(
            "inputs", list(next_main or ()), list(drafts or ()), fallback
        )
        return list(main_ids), list(draft_ids), reason

    def share_batch_plan(self, standard: bool, depth: int) -> tuple[bool, int]:
        """Rank zero's choice for a batch step: ordinary decode, or MTP at this depth."""

        standard, depth = self._broadcast("plan", bool(standard), int(depth))
        return bool(standard), int(depth)

    def share_batch_park(self, park: bool) -> bool:
        """Whether rank zero parks batched MTP after this cycle."""

        (park,) = self._broadcast("park", bool(park))
        return bool(park)

    def share_cycle_result(self, accepted: int, emit_id: int) -> tuple[int, int, int]:
        """Rank zero's accepted draft count and final token, plus a seed."""

        accepted, emit_id, seed = self._broadcast(
            "result", int(accepted), int(emit_id), _new_seed()
        )
        return int(accepted), int(emit_id), int(seed)
