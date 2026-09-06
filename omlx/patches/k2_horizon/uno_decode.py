# SPDX-License-Identifier: Apache-2.0
"""Native MLX linear two-pass Uno decoding with request-owned KV and RNG."""

# Psi-Spec reference: https://github.com/ifm-ai/uno/blob/main/nano_vllm_uno/engine/two_pass_decoding.py

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache


def probabilities(logits, temperature, top_p=1.0, top_k=None):
    """Compute the exact filtered distribution used to sample and verify."""
    values = logits.astype(mx.float32) / temperature
    order = mx.argsort(-values, axis=-1)
    sorted_values = mx.take_along_axis(values, order, axis=-1)
    if top_k is not None and top_k < values.shape[-1]:
        sorted_values = mx.where(
            mx.arange(values.shape[-1]) < top_k, sorted_values, -mx.inf
        )
    probs = mx.softmax(sorted_values, axis=-1)
    if top_p < 1:
        probs = mx.where(mx.cumsum(probs, axis=-1) - probs > top_p, 0, probs)
        probs = probs / mx.sum(probs, axis=-1, keepdims=True)
    return mx.put_along_axis(mx.zeros_like(probs), order, probs, axis=-1)


def acceptance_and_residual(p, q, proposals, uniforms):
    pt = mx.take_along_axis(p, proposals[..., None], axis=-1).squeeze(-1)
    qt = mx.take_along_axis(q, proposals[..., None], axis=-1).squeeze(-1)
    ratio = mx.where(qt > 0, pt / mx.where(qt > 0, qt, 1), 0)
    accepted = uniforms < mx.minimum(ratio, 1)
    residual = mx.maximum(p - q, 0)
    mass = mx.sum(residual, axis=-1, keepdims=True)
    # Psi-Spec returns p when correction mass is zero.
    residual = mx.where(mass > 0, residual / mx.where(mass > 0, mass, 1), p)
    return accepted, residual


@dataclass(frozen=True)
class UnoCycle:
    tokens: tuple[int, ...]
    accepted_proposals: int
    proposed_tokens: int
    forwards: int
    cache_length: int
    finish_reason: str | None


class UnoDecoder:
    """Generate committed blocks; callers never observe rejected draft tokens."""

    def __init__(
        self,
        model,
        *,
        eos_token_ids,
        block_size=8,
        temperature=1.0,
        top_p=0.95,
        top_k=None,
        seed=0,
        prefill_step_size=512,
    ):
        if not getattr(model, "_uno_adapter_loaded", False):
            raise ValueError("Uno decoding requires a validated conditional adapter")
        # The released Uno recipes use eight-token blocks. Tails may be shorter.
        if type(block_size) is not int or not 1 <= block_size <= 8:
            raise ValueError("Uno block_size must be in [1, 8]")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("Uno temperature must be finite and nonnegative")
        if not math.isfinite(top_p) or not 0 < top_p <= 1:
            raise ValueError("Uno top_p must be in (0, 1]")
        if top_k is not None and (
            type(top_k) is not int or not 0 < top_k <= model.args.vocab_size
        ):
            raise ValueError("Uno top_k must be positive and within vocabulary")
        self.model = model
        self.eos = set(eos_token_ids)
        self.block_size = block_size
        self.temperature, self.top_p, self.top_k = temperature, top_p, top_k
        self.key = mx.random.key(seed)
        if type(prefill_step_size) is not int or prefill_step_size <= 0:
            raise ValueError("Uno prefill_step_size must be a positive integer")
        self.prefill_step_size = prefill_step_size

    def _key(self):
        self.key, key = mx.random.split(self.key)
        return key

    def _sample(self, logits):
        if self.temperature == 0:
            return mx.argmax(logits, axis=-1), None
        probs = probabilities(logits, self.temperature, self.top_p, self.top_k)
        return mx.random.categorical(mx.log(probs), key=self._key()), probs

    @staticmethod
    def _trim(cache, length):
        for layer in cache:
            if layer.offset < length:
                raise RuntimeError(f"Uno KV frontier {layer.offset} precedes {length}")
            layer.trim(layer.offset - length)
            if layer.offset != length:
                raise RuntimeError("Uno KV rollback failed")

    def generate(self, prompt, *, max_tokens, cancelled=None, prompt_cache=None):
        if not prompt or type(max_tokens) is not int or max_tokens < 0:
            raise ValueError(
                "Uno requires a nonempty prompt and nonnegative max_tokens"
            )
        if any(
            type(token) is not int or not 0 <= token < self.model.args.vocab_size
            for token in prompt
        ):
            raise ValueError("Uno prompt token outside vocabulary")
        committed = list(prompt)
        cache = make_prompt_cache(self.model) if prompt_cache is None else prompt_cache
        offset = cache[0].offset
        if not 0 <= offset < len(prompt) or any(c.offset != offset for c in cache):
            raise ValueError("Uno prefix cache must leave at least one uncached token")
        if len(prompt) > 1 and max_tokens:
            for start in range(offset, len(prompt) - 1, self.prefill_step_size):
                if cancelled is not None and cancelled():
                    return
                end = min(len(prompt) - 1, start + self.prefill_step_size)
                self.model(mx.array([prompt[start:end]]), cache=cache)
                mx.eval([layer.state for layer in cache])
                mx.synchronize()
                mx.clear_cache()
        emitted = 0
        while emitted < max_tokens:
            if cancelled is not None and cancelled():
                return
            length = min(self.block_size, max_tokens - emitted)
            frontier = len(committed)
            if any(layer.offset != frontier - 1 for layer in cache):
                raise RuntimeError("Uno draft must start with one uncached seed")
            # Upstream noise.py uses [1, mask_token_id). Released K2 masks equal vocab_size.
            noise = mx.random.randint(
                1, self.model.args.vocab_size, shape=(length - 1,), key=self._key()
            )
            draft = mx.concatenate([mx.array([committed[-1]]), noise])[None]
            row_mask = mx.concatenate([mx.zeros((1,)), mx.ones((length - 1,))])[None]
            draft_logits = self.model(draft, cache=cache, lora_mask=row_mask)[0]
            proposals, q = self._sample(draft_logits)
            mx.eval(proposals, q)
            self._trim(cache, frontier)
            verify_logits = self.model(proposals[None], cache=cache)[0]
            targets, p = self._sample(verify_logits)
            uniforms = residual = None
            if self.temperature == 0:
                flags = proposals[1:] == targets[:-1]
                corrections = targets[:-1]
            elif length > 1:
                uniforms = mx.random.uniform(shape=(length - 1,), key=self._key())
                flags, residual = acceptance_and_residual(
                    p[:-1], q[1:], proposals[1:], uniforms
                )
                corrections = mx.random.categorical(mx.log(residual), key=self._key())
            else:
                flags = mx.array([], dtype=mx.bool_)
                corrections = mx.array([], dtype=mx.int32)
            mx.eval(flags, corrections, targets)
            accepted = 0
            for flag in flags.tolist():
                if not flag:
                    break
                accepted += 1
            proposed = proposals.tolist()
            output = proposed[: accepted + 1]
            if accepted < length - 1:
                output.append(int(corrections[accepted].item()))
            else:
                output.append(int(targets[-1].item()))
            output = output[: max_tokens - emitted]
            finish = None
            for i, token in enumerate(output):
                if token in self.eos:
                    output = output[: i + 1]
                    finish = "stop"
                    break
            if cancelled is not None and cancelled():
                return
            committed.extend(output)
            emitted += len(output)
            self._trim(cache, len(committed) - 1)
            if finish is None and emitted == max_tokens:
                finish = "length"
            # Reuse small allocations and reclaim retired KV buffers.
            if mx.get_cache_memory() > 64 * 1024**2:
                mx.synchronize()
                mx.clear_cache()
            yield UnoCycle(
                tuple(output),
                min(accepted, len(output) - 1),
                length - 1,
                2,
                len(committed) - 1,
                finish,
            )
            if finish:
                return
