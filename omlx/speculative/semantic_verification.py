# SPDX-License-Identifier: Apache-2.0
"""Exact dense-KV target verification for the bounded OoO-Spec lane."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, cast

import mlx.core as mx
from mlx_lm.models.cache import KVCache


class SemanticVerificationError(Exception):
    """The hint lane could not preserve ordinary serial target semantics."""


@dataclass(frozen=True)
class VerifiedContinuation:
    """Target-derived output queue and isolated cache through the whole queue."""

    cache: list[KVCache]
    token_ids: tuple[int, ...]
    logprobs: tuple[Any, ...]
    accepted_tokens: int


class PrimedStateMachine:
    """Public state-machine adapter preserving stop matching across handback."""

    def __init__(self, state_machine: Any, matcher_state: Any) -> None:
        self._state_machine = state_machine
        self._matcher_state = matcher_state

    def make_state(self) -> Any:
        return self._matcher_state

    def match(self, state: Any, token: int) -> tuple[Any, Any, Any]:
        return cast(tuple[Any, Any, Any], self._state_machine.match(state, token))


def dense_kv_offsets(cache: Sequence[Any]) -> tuple[int, ...]:
    """Return offsets only when every layer is the exact ordinary KVCache."""
    if not cache:
        raise SemanticVerificationError("target cache is empty")
    offsets: list[int] = []
    for layer in cache:
        if type(layer) is not KVCache:
            raise SemanticVerificationError(
                f"unsupported target cache layer: {type(layer).__name__}"
            )
        offset = getattr(layer, "offset", None)
        if not isinstance(offset, int) or offset < 0:
            raise SemanticVerificationError("target cache offset is invalid")
        if offset > 0 and (
            getattr(layer, "keys", None) is None
            or getattr(layer, "values", None) is None
        ):
            raise SemanticVerificationError("target cache storage is missing")
        offsets.append(offset)
    return tuple(offsets)


def assert_dense_kv_offset(cache: Sequence[Any], expected: int) -> None:
    offsets = dense_kv_offsets(cache)
    if any(offset != expected for offset in offsets):
        raise SemanticVerificationError(
            f"target cache offset mismatch: expected {expected}, got {offsets}"
        )


def clone_dense_kv_cache(cache: Sequence[Any], expected_offset: int) -> list[KVCache]:
    """Clone dense KV storage so verification cannot mutate baseline cache."""
    assert_dense_kv_offset(cache, expected_offset)
    cloned: list[KVCache] = []
    arrays: list[Any] = []
    for source in cache:
        target = KVCache()
        if expected_offset:
            # ``+ 0`` creates detached MLX storage; a view would let the
            # verifier's in-place cache writes alter the baseline cache.
            target.keys = source.keys[..., :expected_offset, :] + 0
            target.values = source.values[..., :expected_offset, :] + 0
            target.offset = expected_offset
            arrays.extend((target.keys, target.values))
        cloned.append(target)
    if arrays:
        mx.eval(*arrays)
    assert_dense_kv_offset(cloned, expected_offset)
    return cloned


def trim_dense_kv_cache_exact(
    cache: Sequence[Any], n_tokens: int, *, expected_before: int
) -> None:
    """Trim every layer by exactly ``n_tokens`` with before/after assertions."""
    if n_tokens < 0 or n_tokens > expected_before:
        raise SemanticVerificationError("target cache trim amount is invalid")
    assert_dense_kv_offset(cache, expected_before)
    if n_tokens == 0:
        return
    for layer in cache:
        trim = getattr(layer, "trim", None)
        if not callable(trim):
            raise SemanticVerificationError("target cache is not trimmable")
    for layer in cache:
        try:
            trimmed = layer.trim(n_tokens)
        except Exception as exc:
            raise SemanticVerificationError("target cache trim failed") from exc
        if isinstance(trimmed, bool) or int(trimmed) != n_tokens:
            raise SemanticVerificationError("target cache trim was not exact")
    assert_dense_kv_offset(cache, expected_before - n_tokens)


def _stream_context(stream: Any | None):
    return mx.stream(stream) if stream is not None else nullcontext()


def _eval_cache_storage(cache: Sequence[KVCache]) -> None:
    arrays: list[Any] = []
    for layer in cache:
        if layer.keys is not None:
            arrays.extend((layer.keys, layer.values))
    if arrays:
        mx.eval(*arrays)


def _model_logits(model: Any, token_ids: Sequence[int], cache: list[KVCache]) -> Any:
    inputs = mx.array([list(token_ids)])
    logits = model(inputs, cache=cache)
    if hasattr(logits, "logits"):
        logits = logits.logits
    if not isinstance(logits, mx.array) or logits.ndim != 3:
        raise SemanticVerificationError("target model returned invalid logits")
    if logits.shape[0] != 1 or logits.shape[1] != len(token_ids):
        raise SemanticVerificationError("target model logits shape is invalid")
    return logits


def _serial_greedy_step(
    *,
    model: Any,
    input_token: int,
    cache: list[KVCache],
    expected_before: int,
    stream: Any | None,
) -> tuple[int, Any]:
    """Run the exact one-token arithmetic used by ordinary greedy decode."""
    assert_dense_kv_offset(cache, expected_before)
    with _stream_context(stream):
        logits = _model_logits(model, (input_token,), cache)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        row = logprobs[0, 0] + 0
        target = mx.argmax(row)
        mx.eval(row, target)
        _eval_cache_storage(cache)
    assert_dense_kv_offset(cache, expected_before + 1)
    return int(target.item()), row


def verify_greedy_suffix(
    *,
    model: Any,
    baseline_cache: Sequence[Any],
    prompt_token_ids: Sequence[int],
    suffix_token_ids: Sequence[int],
    stream: Any | None = None,
) -> VerifiedContinuation:
    """Verify one semantic suffix with only target logits and isolated KV.

    The returned queue is the accepted candidate prefix followed by one target
    correction/bonus token.  Its cache already contains every queued token;
    the scheduler exposes them one at a time and trims any unexposed tail on
    termination.
    """
    prompt = tuple(int(token) for token in prompt_token_ids)
    suffix = tuple(int(token) for token in suffix_token_ids)
    if not prompt or not suffix:
        raise SemanticVerificationError("prompt and semantic suffix must be nonempty")
    if any(token < 0 for token in (*prompt, *suffix)):
        raise SemanticVerificationError("token ids must be nonnegative")

    baseline_offset = len(prompt) - 1
    verify_cache = clone_dense_kv_cache(baseline_cache, baseline_offset)
    try:
        queue: list[int] = []
        queue_logprobs: list[Any] = []
        accepted = 0
        input_token = prompt[-1]
        offset = baseline_offset

        for candidate in suffix:
            target, logprobs = _serial_greedy_step(
                model=model,
                input_token=input_token,
                cache=verify_cache,
                expected_before=offset,
                stream=stream,
            )
            offset += 1
            queue_logprobs.append(logprobs)
            if target != candidate:
                queue.append(target)
                input_token = target
                break
            queue.append(candidate)
            accepted += 1
            input_token = candidate
        else:
            # The final accepted suffix token still has to be processed to
            # obtain the ordinary serial bonus token.
            target, logprobs = _serial_greedy_step(
                model=model,
                input_token=input_token,
                cache=verify_cache,
                expected_before=offset,
                stream=stream,
            )
            offset += 1
            queue.append(target)
            queue_logprobs.append(logprobs)
            input_token = target

        # Commit the correction/bonus exactly as the next ordinary serial
        # decode step would. Its newly predicted successor remains private;
        # public BatchGenerator recreates it by replaying this final token.
        _serial_greedy_step(
            model=model,
            input_token=input_token,
            cache=verify_cache,
            expected_before=offset,
            stream=stream,
        )
        assert_dense_kv_offset(verify_cache, len(prompt) + len(queue))
        return VerifiedContinuation(
            cache=verify_cache,
            token_ids=tuple(queue),
            logprobs=tuple(queue_logprobs),
            accepted_tokens=accepted,
        )
    except SemanticVerificationError:
        raise
    except Exception as exc:
        raise SemanticVerificationError("target verification failed") from exc


def cold_recompute_dense_kv_cache(
    *,
    model: Any,
    baseline_cache: Sequence[Any],
    baseline_offset: int,
    token_ids: Sequence[int],
    stream: Any | None = None,
) -> list[KVCache]:
    """Clone untouched baseline KV and replay a continuation serially."""
    tokens = tuple(int(token) for token in token_ids)
    if not tokens:
        raise SemanticVerificationError("cold replay requires at least one token")
    cache = clone_dense_kv_cache(baseline_cache, baseline_offset)
    try:
        offset = baseline_offset
        for token in tokens:
            _serial_greedy_step(
                model=model,
                input_token=token,
                cache=cache,
                expected_before=offset,
                stream=stream,
            )
            offset += 1
        assert_dense_kv_offset(cache, baseline_offset + len(tokens))
        return cache
    except SemanticVerificationError:
        raise
    except Exception as exc:
        raise SemanticVerificationError("cold target replay failed") from exc
