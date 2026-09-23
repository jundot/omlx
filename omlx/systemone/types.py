# SPDX-License-Identifier: Apache-2.0
"""Pure data types for structured System One reads.

A *structured read* seeds a block-diffusion canvas with the fixed text of an
answer template, leaves only the answer slots as noise, and runs one read-only
denoise step. The per-slot distribution over the allowed label ids **is** the
answer: nothing is generated, nothing is parsed, so nothing can go off-schema.

The slot/template model follows `razorback16/openjev` (Apache-2.0), which in
turn adapts `vllm-project/vllm#57250`. The noise law is OpenJev's: a fresh
uniform token id from the model vocabulary at each slot position, everything
else in the canvas left as clean template text.

This module is deliberately free of ``mlx`` and of any tokenizer. It is the
seam between the request surface (which owns tokenization and schema) and the
engine (which owns the forward pass): the engine accepts these plain dataclasses
and returns probabilities, and nothing above it touches an ``mx.array``.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field

__all__ = ["ReadSlot", "ReadGroup", "SlotDistribution", "StructuredReadResult"]


@dataclass(frozen=True)
class ReadSlot:
    """One answer position inside a canvas.

    ``position`` indexes the canvas; ``label_ids`` are the only token ids the
    caller cares about, in the order the answer maps back to ``label_keys``.
    """

    key: str
    position: int
    label_ids: tuple[int, ...]
    label_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.label_ids:
            raise ValueError(f"slot {self.key!r} has no label ids")
        if self.position < 0:
            raise ValueError(f"slot {self.key!r} has a negative position")
        if len(self.label_keys) and len(self.label_keys) != len(self.label_ids):
            raise ValueError(
                f"slot {self.key!r} label_keys/label_ids length mismatch "
                f"({len(self.label_keys)} != {len(self.label_ids)})"
            )

    @property
    def n_labels(self) -> int:
        return len(self.label_ids)


@dataclass(frozen=True)
class ReadGroup:
    """A canvas template plus the slots to noise, answered in one forward.

    ``template_ids`` is the *exact* canvas content: it already contains the
    trailing turn-close token and is not padded. Canvas rows attend
    bidirectionally to every other canvas row, so pad rows would perturb the
    label slots — on MLX the canvas is sized exactly and there is nothing to
    perturb.
    """

    template_ids: tuple[int, ...]
    slots: tuple[ReadSlot, ...]
    label: str | None = None
    metadata: dict[str, object] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not self.template_ids:
            raise ValueError("read group has an empty canvas template")
        seen: dict[int, str] = {}
        for slot in self.slots:
            if slot.position >= len(self.template_ids):
                raise ValueError(
                    f"slot {slot.key!r} position {slot.position} is outside a "
                    f"canvas of width {len(self.template_ids)}"
                )
            if slot.position in seen:
                raise ValueError(
                    f"slots {seen[slot.position]!r} and {slot.key!r} share canvas "
                    f"position {slot.position}"
                )
            seen[slot.position] = slot.key

    @property
    def width(self) -> int:
        return len(self.template_ids)

    def canvas(self, rng: random.Random | None, noise_hi: int) -> list[int]:
        """Canvas content for one noise draw.

        With ``rng=None`` the canvas comes back as clean template text (no
        noise) — the deterministic baseline used by tests. Otherwise every slot
        position gets a fresh uniform id in ``[0, noise_hi)`` and every other
        position keeps its template token.
        """

        canvas = list(self.template_ids)
        if rng is not None:
            for slot in self.slots:
                canvas[slot.position] = rng.randrange(noise_hi)
        return canvas


@dataclass(frozen=True)
class SlotDistribution:
    """Calibrated distribution over one slot's labels.

    Probabilities come from **raw** (softcapped, temperature-1) logits. The
    denoise schedule's temperature ramp must never touch these numbers: it
    shares the argmax but destroys the calibration.
    """

    key: str
    label_ids: tuple[int, ...]
    label_keys: tuple[str, ...]
    probabilities: tuple[float, ...]

    @property
    def entropy(self) -> float:
        """Shannon entropy in nats."""

        total = 0.0
        for p in self.probabilities:
            if p > 0.0:
                total -= p * math.log(p)
        return total

    @property
    def confidence(self) -> float:
        """``max(0, 1 - H(p) / ln K)`` — 1 when the mass is all on one label."""

        k = len(self.probabilities)
        if k <= 1:
            return 1.0
        return max(0.0, 1.0 - self.entropy / math.log(k))

    @property
    def argmax_index(self) -> int:
        best = 0
        for i, p in enumerate(self.probabilities):
            if p > self.probabilities[best]:
                best = i
        return best

    @property
    def argmax_id(self) -> int:
        return self.label_ids[self.argmax_index]

    @property
    def argmax_label(self) -> str | None:
        if not self.label_keys:
            return None
        return self.label_keys[self.argmax_index]


@dataclass
class StructuredReadResult:
    """Outcome of one structured read (one prefill, N canvas forwards)."""

    distributions: list[SlotDistribution] = field(default_factory=list)
    per_sample: dict[str, list[tuple[float, ...]]] = field(default_factory=dict)
    seed: int | None = None
    samples: int = 1
    steps: int = 1
    prompt_tokens: int = 0
    canvas_tokens: int = 0
    forwards: int = 0
    prefill_ms: float = 0.0
    decode_ms: float = 0.0

    def by_key(self, key: str) -> SlotDistribution | None:
        for dist in self.distributions:
            if dist.key == key:
                return dist
        return None

    @property
    def max_entropy(self) -> float:
        return max((d.entropy for d in self.distributions), default=0.0)

    @property
    def total_ms(self) -> float:
        return self.prefill_ms + self.decode_ms

    def as_dict(self) -> dict[str, object]:
        """Client-facing shape: key -> {probabilities, confidence, ...}."""

        out: dict[str, object] = {}
        for dist in self.distributions:
            entry: dict[str, object] = {
                "probabilities": [
                    {
                        "label": (
                            dist.label_keys[i]
                            if dist.label_keys
                            else str(dist.label_ids[i])
                        ),
                        "probability": p,
                    }
                    for i, p in enumerate(dist.probabilities)
                ],
                "confidence": dist.confidence,
                "entropy": dist.entropy,
            }
            if dist.label_keys:
                entry["choice"] = dist.argmax_label
            out[dist.key] = entry
        return out


def normalized(probabilities: Sequence[float]) -> tuple[float, ...]:
    """Renormalize a probability vector; uniform when the mass is degenerate."""

    total = float(sum(probabilities))
    if total <= 0.0 or not math.isfinite(total):
        n = max(1, len(probabilities))
        return tuple(1.0 / n for _ in range(n))
    return tuple(float(p) / total for p in probabilities)
