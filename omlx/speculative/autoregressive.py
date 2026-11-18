# SPDX-License-Identifier: Apache-2.0
"""Greedy full-model speculation for the dense Qwen3.5/3.8 text backbone."""

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache, make_prompt_cache
from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs
from mlx_lm.utils import load_model


class _QwenTextDraft(TextModel):
    def sanitize(self, weights):
        # The ModelOpt export has the transformers language-model prefix
        # even though it declares a standalone text model in config.json.
        prefix = "model.language_model."
        weights = {
            ("model." + key[len(prefix) :] if key.startswith(prefix) else key): value
            for key, value in weights.items()
        }
        return super().sanitize(weights)


def load_qwen_draft(path: Path, target_path: Path):
    """Load the standalone Qwen text checkpoint using mlx-lm's strict loader."""
    if (path / "tokenizer.json").read_bytes() != (
        target_path / "tokenizer.json"
    ).read_bytes():
        raise ValueError(
            "The target and draft must have identical tokenizer.json files"
        )

    def model_classes(config):
        if config["model_type"] != "qwen3_5_text":
            raise ValueError("Expected a standalone qwen3_5_text drafter")
        return _QwenTextDraft, TextModelArgs

    model, _ = load_model(path, get_model_classes=model_classes)
    return model


def prefill(model, prompt, step_size=512):
    """Fill the cache up to the input token that starts the first round."""
    cache = make_prompt_cache(model)
    for start in range(0, len(prompt) - 1, step_size):
        inputs = prompt[start : min(start + step_size, len(prompt) - 1)]
        model(mx.array([inputs]), cache=cache)
        mx.eval([layer.state for layer in cache])
    return cache


def _draft_checkpoint(cache):
    # GDN replaces the arrays inside a mutable list on each forward. Keep
    # that round's array references and record where the attention KV ends.
    return [
        (
            (layer.cache.copy(), layer.left_padding, layer.lengths)
            if type(layer) is ArraysCache
            else layer.offset
        )
        for layer in cache
    ]


def _restore_draft(cache, checkpoint):
    for layer, state in zip(cache, checkpoint, strict=True):
        if type(layer) is ArraysCache:
            layer.state = state
        else:
            trim_count = layer.offset - state
            if layer.trim(trim_count) != trim_count:
                raise RuntimeError("Draft KV cache could not roll back")


@dataclass(frozen=True)
class DraftRound:
    tokens: list[int]
    accepted: int
    proposed: int


class GreedyQwenDraft:
    """Own the target and drafter caches for one prefilled text request."""

    def __init__(self, target, draft, prompt):
        if not prompt:
            raise ValueError("The prompt must have at least one token")
        if not callable(getattr(target, "mtp_partial_rollback", None)):
            raise ValueError("The target needs oMLX's Qwen cache rollback hooks")
        self.target = target
        self.draft = draft
        self.target_cache = prefill(target, prompt)
        self.draft_cache = prefill(draft, prompt)
        if any(type(layer) not in (ArraysCache, KVCache) for layer in self.draft_cache):
            raise ValueError("The draft requires standard Qwen GDN and KV caches")
        self._next_input = prompt[-1]

    def step(self, depth: int) -> DraftRound:
        if depth < 0:
            raise ValueError("Draft depth must be nonnegative")
        seed = self._next_input
        draft_token = seed
        proposals = []
        checkpoints = []
        for _ in range(depth):
            logits = self.draft(mx.array([[draft_token]]), cache=self.draft_cache)
            draft_token = mx.argmax(logits[0, -1]).item()
            proposals.append(draft_token)
            checkpoints.append(_draft_checkpoint(self.draft_cache))

        block = mx.array([[seed, *proposals]])
        logits = self.target(block, cache=self.target_cache, n_confirmed=1)
        verified = mx.argmax(logits[0], axis=-1).tolist()
        accepted = 0
        for proposal, token in zip(proposals, verified):
            if proposal != token:
                break
            accepted += 1

        if accepted < depth:
            if not self.target.mtp_partial_rollback(self.target_cache, accepted, depth):
                raise RuntimeError("Target cache could not roll back")
            _restore_draft(self.draft_cache, checkpoints[accepted])
        else:
            # The draft produced the last proposal from its predecessor.
            # Advance it through that proposal to align with the target.
            self.draft(mx.array([[draft_token]]), cache=self.draft_cache)

        tokens = verified[: accepted + 1]
        self._next_input = tokens[-1]
        return DraftRound(tokens, accepted, depth)
