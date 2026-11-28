# SPDX-License-Identifier: Apache-2.0
"""Real-model hardware calibration (replaces the synthetic compute probe).

Decision 2026-11-13 (product owner): ranking must rest on REAL model
measurements, not synthetic microbenchmarks. A small calibration model — the
reference project's ``dev-1.5b`` (Qwen2.5-1.5B-Instruct-4bit, ~1 GB) — is run on
each Mac and its real prefill/decode throughput characterizes that Mac:

* DECODE is memory-bound: decode tok/s tracks unified-memory bandwidth.
* PREFILL is compute-bound: prefill tok/s tracks fp16/AI compute.

The election (``election.py``) and oMLX's planner (``plan_hybrid``) are NOT
changed: the planner still owns the layer split, the election still only picks
rank order / subset / levers. This module produces the per-node
``ComputeProfile`` numbers they consume, derived from a real completion instead
of a GEMV/reduction microbenchmark. The methodology is ported from
LocalInferenz ``console/deploy.chat`` (streamed completion, TTFT/decode split,
cold request discarded) — not its app or planner.

The conversion to the planner's units (model-weight bytes per second):

* decode reads the whole model once per generated token, so
  ``decode_weight_bytes_per_second = decode_tok_per_s * model_weight_bytes``.
* prefill reads the weights once per prompt of ``prompt_tokens`` tokens, so
  ``prefill_weight_bytes_per_second = prefill_tok_per_s * model_weight_bytes /
  prompt_tokens``.

A stage's predicted seconds then come out right: the planner divides the
stage's layer bytes by these rates, i.e. it scales the whole-model time by the
stage's fraction of the model. Nothing here invents a rate: a missing timing is
``None`` (unknown compute), never zero or an estimate.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ..planner import ModelLayout
from .election import ComputeProfile

# The reference project's quick-check model: ~1 GB, proven to run here. It is
# small enough to load on every Mac (even a 32 GiB node) and characterize the
# hardware without competing with the user's model for memory.
CALIBRATION_MODEL_REPO = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
CALIBRATION_MODEL_DIR = "~/Models/dev-1.5b"

# A fixed prompt and token budget so rates are comparable across Macs and runs
# (ported from LocalInferenz tuning.sweep). Enough decode tokens that the number
# means something — a three-token reply measures startup jitter, not decoding.
CALIBRATION_PROMPT = "Write four short sentences about the ocean."
CALIBRATION_MAX_TOKENS = 96
# A short warmup request loads the checkpoint; its timing is discarded so the
# measured request is warm (LocalInferenz profiles.tune_profile).
CALIBRATION_WARMUP_PROMPT = "Reply OK."
CALIBRATION_WARMUP_TOKENS = 8


@dataclass(frozen=True)
class CompletionTimings:
    """Real throughput from one warm completion (oMLX usage fields).

    ``decode_tok_per_s`` is required; ``prefill_tok_per_s`` and ``ttft_seconds``
    are recorded only when the reply carries them. A missing value is unknown,
    never zero.
    """

    decode_tok_per_s: float
    prefill_tok_per_s: float | None = None
    ttft_seconds: float | None = None
    prompt_tokens: int | None = None


def compute_profile_from_timings(
    timings: CompletionTimings,
    *,
    model_weight_bytes: int,
    backend: str = "ring",
    measured_at: str = "",
) -> ComputeProfile | None:
    """Convert real completion throughput into the planner's ComputeProfile.

    Returns ``None`` when there is no decode rate (nothing measured) — the
    election then scores the Mac as unknown compute, never as fast. The decode
    rate is bandwidth-proportional; the prefill rate is compute-proportional
    (normalized by the prompt length, since prefill reads the weights once per
    prompt). When prefill is unreported, the decode rate is reused as a
    conservative stand-in so a measured Mac is not penalized as unmeasured; this
    is documented, not an invented throughput.
    """

    if model_weight_bytes <= 0:
        raise ValueError("model_weight_bytes must be positive")
    decode = timings.decode_tok_per_s
    if decode is None or decode <= 0:
        return None
    decode_rate = decode * model_weight_bytes
    if timings.prefill_tok_per_s and timings.prefill_tok_per_s > 0:
        prompt_tokens = max(1, timings.prompt_tokens or 1)
        prefill_rate = (
            timings.prefill_tok_per_s * model_weight_bytes / prompt_tokens
        )
    else:
        prefill_rate = decode_rate
    return ComputeProfile(
        decode_weight_bytes_per_second=decode_rate,
        prefill_weight_bytes_per_second=prefill_rate,
        backend=backend,
        measured_at=measured_at,
        samples=1,
    )


def calibration_weight_bytes(layout: ModelLayout, *, include_mtp: bool = False) -> int:
    """The weight bytes one completion of this model reads. Pure.

    Decode reads the whole model once per generated token, so the conversion
    needs the fixed weights plus every layer. The Lightning MTP draft head is
    only read when a plan enables MTP; the dev-1.5b calibration model has none,
    so it is excluded unless the caller explicitly includes it.
    """

    total = layout.total_weight_bytes
    if include_mtp:
        total += layout.mtp_weight_bytes
    return total


def calibration_body(model_id: str, *, prompt: str, max_tokens: int) -> dict[str, Any]:
    """One streaming completion body whose final usage chunk is the measurement.

    The same shape the observe seam uses (``start._observe_body`` with
    ``stream=True``): oMLX puts TTFT and the prompt rate in the final usage
    chunk only when usage is requested, so the request always streams with
    ``include_usage``.
    """

    return {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def _positive_float(value: Any) -> float | None:
    # The same rule as start._positive_float: a rate is a finite positive
    # number; a bool or a string is not a measurement.
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _positive_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def timings_from_reply(reply: Any) -> CompletionTimings | None:
    """Read oMLX's usage fields exactly as ``start.observed_from_reply`` does.

    ``generation_tokens_per_second`` (decode) is required: without it there is
    no measurement at all. ``prompt_tokens_per_second`` (prefill),
    ``time_to_first_token`` and ``prompt_tokens`` are recorded only when the
    reply carries them; a missing value stays ``None`` — unknown, never zero.

    The read is replicated rather than imported from ``start``: ``start``
    imports ``provisioning``, which imports ``measurement``, which imports this
    module — importing ``start`` here would close that cycle.
    """

    usage = reply.get("usage") if isinstance(reply, Mapping) else None
    if not isinstance(usage, Mapping):
        return None
    decode = _positive_float(usage.get("generation_tokens_per_second"))
    if decode is None:
        return None
    return CompletionTimings(
        decode_tok_per_s=decode,
        prefill_tok_per_s=_positive_float(usage.get("prompt_tokens_per_second")),
        ttft_seconds=_positive_float(usage.get("time_to_first_token")),
        prompt_tokens=_positive_int(usage.get("prompt_tokens")),
    )


#: One completion through a node's seam: body dict in, reply mapping out (the
#: shape ``CoordinatorOps.observe``/``stream_observe`` produce). Sync or async.
CompletionSeam = Callable[[Mapping[str, Any]], Any]


async def calibrate_node(
    completion_seam: CompletionSeam,
    *,
    model_weight_bytes: int,
    model_id: str = CALIBRATION_MODEL_DIR,
    backend: str = "ring",
    measured_at: str = "",
) -> ComputeProfile | None:
    """Measure one Mac with a real calibration-model completion, or ``None``.

    ``completion_seam`` takes a completion body dict and returns the reply
    mapping whose ``usage`` carries oMLX's timing fields — the same shape the
    observe seam uses. It may be sync or async; an awaitable reply is awaited.
    The caller owns the transport (local in-process, or the coordinator's own
    admin route over the existing SSH seam); this function only runs the two
    completions and converts the timings. It never stages, launches or loads
    anything itself.

    A short warmup loads the checkpoint and its reply is discarded; the
    measured request then runs warm with the fixed calibration prompt and
    token budget, streamed with ``include_usage`` so TTFT/prefill are present.
    A failed or missing reply — warmup or measured — is unknown compute
    (``None``): never an exception into the caller, never an invented rate.
    """

    if model_weight_bytes <= 0:
        raise ValueError("model_weight_bytes must be positive")
    try:
        # The warmup's timing is deliberately discarded: it loads the
        # checkpoint, and a cold request measures startup jitter, not decode.
        await _complete(
            completion_seam,
            calibration_body(
                model_id,
                prompt=CALIBRATION_WARMUP_PROMPT,
                max_tokens=CALIBRATION_WARMUP_TOKENS,
            ),
        )
    except Exception:  # a failed warmup cannot prove a warm measurement
        return None
    try:
        reply = await _complete(
            completion_seam,
            calibration_body(
                model_id,
                prompt=CALIBRATION_PROMPT,
                max_tokens=CALIBRATION_MAX_TOKENS,
            ),
        )
    except Exception:  # a failed completion is unknown compute, not a crash
        return None
    timings = timings_from_reply(reply)
    if timings is None:
        return None
    return compute_profile_from_timings(
        timings,
        model_weight_bytes=model_weight_bytes,
        backend=backend,
        measured_at=measured_at,
    )


async def _complete(seam: CompletionSeam, body: Mapping[str, Any]) -> Any:
    """Run one completion through the seam, whether it is sync or async."""

    reply = seam(body)
    if inspect.isawaitable(reply):
        reply = await reply
    return reply
