# SPDX-License-Identifier: Apache-2.0
"""LoopCD-Logits for looped Inkling 2 checkpoints: https://arxiv.org/abs/2610.02185"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

REFERENCE = "loopcd_reference"
MAX_STRENGTH = 0.5
THINKING_TOKEN = 200_008
FIRST_CONTROL_TOKEN = 199_998


def apply_loopcd(model: Any, settings: Any) -> None:
    if not getattr(settings, "loopcd_enabled", False):
        return
    if not hasattr(model, "enable_loopcd"):
        return
    skip_thinking = settings.loopcd_skip_thinking
    enabled = model.enable_loopcd(skip_thinking)
    logger.info("LoopCD enabled=%s skip_thinking=%s", enabled, skip_thinking)


def thinking_flags(tokens, before):
    position = mx.arange(tokens.shape[1])
    is_control = tokens >= FIRST_CONTROL_TOKEN
    last_control = mx.cummax(mx.where(is_control, position, -1), axis=1)
    opens_thinking = mx.take_along_axis(
        tokens == THINKING_TOKEN, mx.maximum(last_control, 0), axis=1
    )
    return mx.where(last_control < 0, before, opens_thinking)


def _top_two_margin(logits):
    probs = mx.softmax(logits, axis=-1)
    top = mx.argmax(probs, axis=-1, keepdims=True)
    others = mx.where(mx.arange(probs.shape[-1]) == top, 0.0, probs)
    return mx.max(probs, axis=-1, keepdims=True) - mx.max(
        others, axis=-1, keepdims=True
    )


def _keep_control_probabilities(final, guided):
    ordinary = mx.arange(final.shape[-1]) < FIRST_CONTROL_TOKEN
    final_mass, guided_mass = (
        mx.logsumexp(mx.where(ordinary, x, -mx.inf), axis=-1, keepdims=True)
        for x in (final, guided)
    )
    return mx.where(ordinary, guided + (final_mass - guided_mass), final)


def guide(final, reference, cap, vocab_bias, thinking):
    def mask_vocab(logits):
        return logits if vocab_bias is None else logits + vocab_bias

    final, reference = (
        mx.tanh(x.astype(mx.float32) / cap) * cap for x in (final, reference)
    )
    weight = MAX_STRENGTH * (1.0 - _top_two_margin(mask_vocab(final)))
    if thinking is None:
        return mask_vocab(final + weight * (final - reference))
    weight = mx.where(thinking[..., None], 0.0, weight)
    guided = final + weight * (final - reference)
    return mask_vocab(_keep_control_probabilities(final, guided))


guide_compiled = mx.compile(guide)
