# SPDX-License-Identifier: Apache-2.0
"""Canvas numerics: slot softmax, averaging, confidence, answer shapes. Pure.

Ported from ``openjev/openjev/engine.py`` (razorback16/openjev, Apache-2.0).
The answer shapes (:func:`to_answer`) and :func:`confidence` are upstream's and
are reproduced so a client sees the same JSON it would get from OpenJev.

Where this departs from upstream it is because the in-process engine has
information the HTTP client does not:

* upstream's :func:`slot_distribution` reconstructs probabilities from the
  ``top_logprobs`` the API returns, inventing a floor of
  ``min(top.values()) - 5.0`` for any label outside that set. Here the engine
  hands over the **whole** logits row, so every label gets its exact logit and
  there is no floor to invent.
* upstream's entropy is taken over the returned top-k set and is not normalized,
  because that is all the API gave it. Here it is the entropy of the label
  distribution itself. That matters beyond tidiness: the re-read policy fires on
  ``entropy > auto_threshold``, so the two definitions trigger at different
  points and a port that silently kept upstream's would re-read on a different
  schedule while looking identical.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from .types import normalized

__all__ = [
    "confidence",
    "mean_probabilities",
    "slot_distribution",
    "softmax",
    "to_answer",
]

#: Logits below this many nats under the row maximum are treated as zero mass.
#: Guards ``exp`` against underflow on a wide vocabulary without changing the
#: distribution measurably.
FLOOR_NATS = -30.0


def softmax(logits: Sequence[float]) -> tuple[float, ...]:
    """Numerically stable softmax over raw logits.

    Raw means exactly that: softcapping stays (it is part of the model), the
    denoise schedule's temperature ramp must not be applied — tempering divides
    by a schedule value before anything can observe it, which preserves argmax
    and destroys calibration.
    """

    if not logits:
        return ()
    mx = max(float(x) for x in logits)
    if not math.isfinite(mx):
        return tuple(1.0 / len(logits) for _ in logits)
    ex = [
        math.exp(min(0.0, float(x) - mx)) if float(x) - mx > FLOOR_NATS else 0.0
        for x in logits
    ]
    return normalized(ex)


def slot_distribution(
    logits: Sequence[float], label_ids: Sequence[int]
) -> tuple[float, ...]:
    """Probabilities over ``label_ids``, given the logits at one canvas slot.

    ``logits`` is the full vocabulary row at that position and ``label_ids`` the
    ids the caller cares about, in order. Selecting before the softmax is the
    point: the distribution over labels is what the answer is, and it is
    renormalized over exactly those labels.
    """

    if not label_ids:
        return ()
    row = list(logits)
    picked = [float(row[i]) for i in label_ids]
    return softmax(picked)


def mean_probabilities(draws: Sequence[Sequence[float]]) -> tuple[float, ...]:
    """Average per-draw distributions of equal length.

    An unweighted mean over independent noise draws is the estimator upstream
    uses; it is unbiased for the model's own marginal at that slot because each
    draw is an independent canvas.
    """

    if not draws:
        return ()
    n = len(draws[0])
    if any(len(d) != n for d in draws):
        raise ValueError("draw distributions have differing lengths")
    total = [0.0] * n
    for draw in draws:
        for i, p in enumerate(draw):
            total[i] += float(p)
    return normalized([t / len(draws) for t in total])


def confidence(p: Sequence[float]) -> float:
    """How peaked a distribution is: ``1 - H(p) / ln(K)``.

    1 is certain, 0 uniform. Entropy is over the label distribution itself, in
    nats — see the module docstring for why that is not upstream's top-k
    entropy.
    """

    k = len(p)
    if k <= 1:
        return 1.0
    h = -sum(x * math.log(x) for x in p if x > 0.0)
    return max(0.0, min(1.0, 1.0 - h / math.log(k)))


def entropy(p: Sequence[float]) -> float:
    """Shannon entropy of a distribution, in nats."""

    return -sum(x * math.log(x) for x in p if x > 0.0)


def to_answer(question: dict[str, Any], p: Sequence[float]) -> dict[str, Any]:
    """The client-facing answer for one question, in upstream's shapes.

    A noul carries its yes-mass as a single number; a choice carries the argmax
    name plus the whole distribution; a score carries the expected level, which
    is why score labels are the digits ``0..9`` and the legend echoes the
    criteria exactly as sent.
    """

    kind = question["type"]
    if kind == "noul":
        return {"type": "noul", "noul": float(p[0])}
    if kind == "choice":
        top = max(range(len(p)), key=p.__getitem__)
        return {
            "type": "choice",
            "choice": question["choices"][top][0],
            "probabilities": {c[0]: float(v) for c, v in zip(question["choices"], p)},
            "confidence": confidence(p),
        }
    if kind == "score":
        legend = question.get("legend") or [name for name, _ in question["choices"]]
        return {
            "type": "score",
            "score": sum(i * float(v) for i, v in enumerate(p)),
            "legend": {str(i): legend[i] for i in range(len(p))},
            "probabilities": {str(i): float(v) for i, v in enumerate(p)},
            "confidence": confidence(p),
        }
    raise ValueError(f"unknown question type {kind!r}")
