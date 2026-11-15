# SPDX-License-Identifier: Apache-2.0
"""Fine-tuning the winning constellation with REAL completions.

After the search loop picks the best measured constellation, fine-tuning sweeps
the knobs that shape quality and speed, ported from LocalInferenz
``console/tuning.py`` / ``profiles.py`` (methodology only, not the app):

* prefill CHUNK size — keep the highest measured tok/s;
* SAMPLER (temperature / top_p / top_k) — scored by a golden prompt with a
  checkable answer (no judge model), selected lexicographically: correctness
  first, then fewer reasoning tokens, then fewer output tokens, then throughput;
* MTP on/off — A/B the measured decode rate, the faster measured side wins.

Everything runs through an INJECTED completion seam (fake-testable); nothing
here launches a server. A failed measurement is skipped, never invented; an
unmeasured side never wins. Presets are keyed by model content signature +
hardware pair and must never be reused across hardware.
"""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

# Golden prompts with checkable answers (no judge model). Ported from the
# reference tuning.GOLDEN_PROMPT / profiles golden set.
GOLDEN_PROMPTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Name the three primary colours of light and separate them with commas. "
        "Answer with the names only.",
        ("red", "green", "blue"),
    ),
    ("What is 17 + 25? Answer with the number only.", ("42",)),
    ("What is the capital of France? Answer with the city name only.", ("paris",)),
)

# (name, temperature, top_p, top_k) — ported from the reference SAMPLERS.
SAMPLERS: tuple[tuple[str, float, float, int], ...] = (
    ("deterministic", 0.0, 0.95, 20),
    ("balanced", 0.7, 0.95, 20),
    ("loose", 1.0, 0.95, 40),
)

CHUNKS: tuple[int, ...] = (2048, 4096, 8192)


@dataclass(frozen=True)
class TuneSample:
    """One real completion's measured outcome (the seam returns this)."""

    answer: str = ""
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    tok_per_s: float = 0.0
    ok: bool = True


# The seam runs one completion body and returns its measured sample. The real
# wiring converts an oMLX reply (usage.completion_tokens,
# completion_tokens_details.reasoning_tokens, generation_tokens_per_second,
# choices[0].message.content) into a TuneSample.
CompletionSeam = Callable[[dict], Awaitable[TuneSample]]


def score_quality(answer: str, expected: tuple[str, ...]) -> float:
    """Fraction of expected tokens present in the answer (case-insensitive)."""

    if not expected:
        return 0.0
    lowered = answer.lower()
    hits = sum(1 for token in expected if token.lower() in lowered)
    return hits / len(expected)


def preset_key(model_signature: str, hardware_key: str) -> str:
    """A stable preset key bound to a model content signature AND hardware pair.

    A measurement from another machine pair must never be reused (the reference
    tuning.hardware_key rule), so both parts are hashed into the key.
    """

    raw = f"{model_signature}|{hardware_key}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def hardware_key(node_ids: Sequence[str]) -> str:
    """The canonical hardware-pair key: the node *set*, independent of rank order.

    "Per model and node config" means the Macs, not which one happened to be
    elected rank 0. A later election may flip rank 0 (measured 2026-11-14: MTP
    off moved it from the M2 Ultra to the M5 Max), and a start must still find
    the tuning preset for the same set of Macs. Sorting makes the key stable.
    """

    return ",".join(sorted(str(node_id) for node_id in node_ids))


def preset_matches(key: str, model_signature: str, hardware_key: str) -> bool:
    return key == preset_key(model_signature, hardware_key)


async def sweep_chunks(
    seam: CompletionSeam,
    *,
    model_id: str,
    chunks: tuple[int, ...] = CHUNKS,
    prompt: str = "Write four short sentences about the ocean.",
    max_tokens: int = 96,
) -> list[dict]:
    """Measure tok/s per prefill chunk; the caller keeps the max. Failed chunks
    are skipped (ok=False), never invented."""

    rows: list[dict] = []
    for chunk in chunks:
        sample = await seam(
            {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "prefill_chunk_size": chunk,
            }
        )
        if sample.ok and sample.tok_per_s > 0:
            rows.append({"chunk": chunk, "tok_per_s": sample.tok_per_s})
    return rows


def best_chunk(rows: list[dict]) -> int | None:
    if not rows:
        return None
    return max(rows, key=lambda row: row["tok_per_s"])["chunk"]


async def sweep_samplers(
    seam: CompletionSeam,
    *,
    model_id: str,
    samplers: tuple[tuple[str, float, float, int], ...] = SAMPLERS,
    max_tokens: int = 64,
) -> list[dict]:
    """Score each sampler on the golden prompts; rows carry the lexicographic
    objective inputs (quality, reasoning_tokens, completion_tokens, tok_per_s)."""

    rows: list[dict] = []
    for name, temperature, top_p, top_k in samplers:
        quality = 0.0
        reasoning = 0
        completion = 0
        rate = 0.0
        measured = 0
        for prompt, expected in GOLDEN_PROMPTS:
            sample = await seam(
                {
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                }
            )
            if not sample.ok:
                continue
            measured += 1
            quality += score_quality(sample.answer, expected)
            reasoning += sample.reasoning_tokens
            completion += sample.completion_tokens
            rate += sample.tok_per_s
        if measured:
            rows.append(
                {
                    "sampler": name,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "quality": quality / measured,
                    "reasoning_tokens": reasoning,
                    "completion_tokens": completion,
                    "tok_per_s": rate / measured,
                }
            )
    return rows


def best_sampler(rows: list[dict]) -> dict | None:
    """Lexicographic: correctness first, then fewer reasoning tokens, then fewer
    output tokens, then higher throughput (the reference profiles objective)."""

    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            row["quality"],
            -row["reasoning_tokens"],
            -row["completion_tokens"],
            row["tok_per_s"],
        ),
    )


async def sweep_mtp(
    seam: CompletionSeam,
    *,
    model_id: str,
    mtp_capable: bool = True,
    prompt: str = "Write four short sentences about the ocean.",
    max_tokens: int = 96,
) -> dict:
    """A/B MTP off vs on; the faster MEASURED decode rate wins. An unmeasured
    side never wins; if neither is measured the choice is None.

    A model without an MTP head (``mtp_capable=False``) cannot draft, so there is
    nothing to A/B: return ``mtp=None`` and ``applicable=False`` instead of
    measuring two identical no-ops as if it were a real comparison.
    """

    if not mtp_capable:
        return {"mtp": None, "rates": {}, "applicable": False}

    rates: dict[bool, float] = {}
    for mtp in (False, True):
        sample = await seam(
            {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "mtp": mtp,
            }
        )
        if sample.ok and sample.tok_per_s > 0:
            rates[mtp] = sample.tok_per_s
    if not rates:
        return {"mtp": None, "rates": rates, "applicable": True}
    winner = max(rates, key=rates.get)
    return {"mtp": winner, "rates": rates, "applicable": True}


@dataclass
class FineTuneResult:
    chunk: int | None = None
    sampler: dict | None = None
    mtp: bool | None = None
    mtp_applicable: bool = True
    chunk_applicable: bool = True
    chunk_rows: list[dict] = field(default_factory=list)
    sampler_rows: list[dict] = field(default_factory=list)
    mtp_rates: dict = field(default_factory=dict)


async def fine_tune(
    seam: CompletionSeam,
    *,
    model_id: str,
    mtp_capable: bool = True,
    chunk_applicable: bool = True,
    model_signature: str = "",
    hardware_key: str = "",
) -> FineTuneResult:
    """Run the sweeps and return the winning knobs (real measurements only).

    Only the sampler is a real per-request knob. The chunk and MTP (on/off and
    depth) are per-model settings: a per-request sweep of them measures noise,
    so the caller passes ``chunk_applicable=False`` for those and the sweep is
    skipped (``chunk=None``); tuning them for real needs a redeploy per config.
    """

    chunk_rows = (
        await sweep_chunks(seam, model_id=model_id) if chunk_applicable else []
    )
    sampler_rows = await sweep_samplers(seam, model_id=model_id)
    mtp = await sweep_mtp(seam, model_id=model_id, mtp_capable=mtp_capable)
    return FineTuneResult(
        chunk=best_chunk(chunk_rows) if chunk_applicable else None,
        sampler=best_sampler(sampler_rows),
        mtp=mtp["mtp"],
        mtp_applicable=bool(mtp.get("applicable", True)),
        chunk_applicable=chunk_applicable,
        chunk_rows=chunk_rows,
        sampler_rows=sampler_rows,
        mtp_rates=mtp["rates"],
    )