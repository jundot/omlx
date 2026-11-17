# SPDX-License-Identifier: Apache-2.0
"""Wire fine-tuning to a real coordinator completion seam.

The search loop picks the best measured constellation; fine-tuning then sweeps
chunk / sampler / MTP on that winner using REAL completions. This module adapts
a coordinator completion call (body dict -> oMLX reply mapping) to the
``tuning.CompletionSeam`` contract (body -> TuneSample), reading oMLX's own usage
fields. It is pure orchestration over an injected ``complete`` callable; the
transport (local in-process or the coordinator admin seam) is the caller's.

Nothing here invents a rate: a reply with no decode timing is ``ok=False`` and
the sweep skips it.
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable
from typing import Any

from .tuning import CompletionSeam, FineTuneResult, TuneSample, fine_tune

# A coordinator completion call: takes a chat-completions body, returns the oMLX
# reply (usage + choices). Sync or async.
CompleteCall = Callable[[dict], Any | Awaitable[Any]]


def _positive_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def tune_sample_from_reply(reply: Any) -> TuneSample:
    """Read one oMLX completion reply into a TuneSample (never invents a rate)."""

    if not isinstance(reply, dict):
        return TuneSample(ok=False)
    usage = reply.get("usage")
    if not isinstance(usage, dict):
        return TuneSample(ok=False)
    rate = _positive_float(usage.get("generation_tokens_per_second"))
    if rate is None:
        return TuneSample(ok=False)
    answer = ""
    choices = reply.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(message, dict):
            answer = str(message.get("content") or "")
    completion = usage.get("completion_tokens") or 0
    details = usage.get("completion_tokens_details")
    reasoning = 0
    if isinstance(details, dict):
        reasoning = details.get("reasoning_tokens") or 0
    return TuneSample(
        answer=answer,
        completion_tokens=int(completion),
        reasoning_tokens=int(reasoning),
        tok_per_s=rate,
        ok=True,
    )


def completion_seam(complete: CompleteCall) -> CompletionSeam:
    """Adapt a body->reply completion call into tuning's body->TuneSample seam."""

    async def seam(body: dict) -> TuneSample:
        try:
            reply = complete(body)
            if hasattr(reply, "__await__"):
                reply = await reply
        except Exception:  # noqa: BLE001 - a failed completion is skipped, not fatal
            return TuneSample(ok=False)
        return tune_sample_from_reply(reply)

    return seam


async def fine_tune_constellation(
    complete: CompleteCall,
    *,
    model_id: str,
    mtp_capable: bool = True,
    chunk_applicable: bool = True,
    model_signature: str = "",
    hardware_key: str = "",
) -> FineTuneResult:
    """Run the fine-tuning sweeps on the winning constellation's coordinator."""

    return await fine_tune(
        completion_seam(complete),
        model_id=model_id,
        mtp_capable=mtp_capable,
        chunk_applicable=chunk_applicable,
        model_signature=model_signature,
        hardware_key=hardware_key,
    )
