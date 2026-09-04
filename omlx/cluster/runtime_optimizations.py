# SPDX-License-Identifier: Apache-2.0
"""Capability-gated optimizations for the pinned MLX-LM pipeline worker."""

from __future__ import annotations

import importlib
import inspect
import math
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from .performance import ExecutionSettings


def _capability(
    *,
    enabled: bool,
    active: bool,
    reason: str,
) -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "active": bool(active),
        "reason": reason,
    }


def _supports_coordinator_sampling(
    pipeline_model: Any,
    *,
    batchable: bool,
    world_size: int,
) -> tuple[bool, str]:
    if world_size < 2:
        return False, "requires more than one pipeline rank"
    if not batchable:
        return False, "model is not compatible with MLX-LM continuous batching"
    call = type(pipeline_model).__dict__.get("__call__")
    if not callable(call):
        return False, "pipeline model has no callable forward path"
    try:
        source = inspect.getsource(call)
    except (OSError, TypeError):
        return False, "pipeline forward source is unavailable for validation"
    required = (
        "pipeline_rank",
        "pipeline_size",
        "distributed.all_gather",
        "distributed.send",
    )
    if any(token not in source for token in required):
        return False, "pipeline forward does not match the validated output contract"
    if source.count("distributed.all_gather") != 1:
        return False, "pipeline forward has an ambiguous collective output path"
    if source.count("distributed.send") != 1:
        return False, "pipeline forward has an ambiguous send path"
    return True, "validated final hidden-state gather replaced by token all-sum"


def _supports_native_async_step(generation_batch: Any) -> bool:
    step = getattr(generation_batch, "_step", None)
    try:
        source = inspect.getsource(step)
    except (OSError, TypeError):
        return False
    return "async_eval" in source and "_next_tokens" in source


def _supports_rank_zero_logits(model: Any) -> tuple[bool, int, str]:
    """Validate that worker ranks may advance the model without an LM head."""

    if not getattr(model, "_omlx_supports_rank_zero_logits", False):
        return False, 0, "model adapter has no rank-zero logits contract"
    call = type(model).__dict__.get("__call__")
    if not callable(call):
        return False, 0, "model adapter has no direct callable forward path"
    try:
        signature = inspect.signature(call)
    except (TypeError, ValueError):
        return False, 0, "model adapter forward signature is unavailable"
    if "skip_logits" not in signature.parameters:
        return False, 0, "model adapter does not explicitly accept skip_logits"
    try:
        vocab_size = int(model._omlx_output_vocab_size)
    except (AttributeError, TypeError, ValueError):
        return False, 0, "model adapter does not declare its output vocabulary"
    if vocab_size < 1:
        return False, 0, "model adapter declared an invalid output vocabulary"
    return (
        True,
        vocab_size,
        "worker ranks skip the vocabulary projection and log-softmax",
    )


def _supports_pipeline_prompt(prompt_batch: Any) -> tuple[bool, str]:
    """Validate the exact MLX-LM prompt loop this module replaces.

    This deliberately checks behavior-bearing source tokens instead of merely
    checking that a method with the right name exists. A future MLX-LM release
    can change cache preparation/finalization or prompt ownership without
    silently running a stale monkeypatch.
    """

    prompt = getattr(prompt_batch, "prompt", None)
    base_prompt = getattr(prompt_batch, "_omlx_base_prompt", prompt)
    known_wrapper = getattr(prompt_batch, "_omlx_prompt_wrapper", prompt)
    if prompt not in {base_prompt, known_wrapper}:
        return False, "another prompt-processing patch owns this model's prefill"
    try:
        source = inspect.getsource(base_prompt)
    except (OSError, TypeError):
        return False, "pinned prompt-processing source is unavailable"
    required = (
        "_right_pad_prompts",
        "self.prefill_step_size",
        "self.model(",
        "c.prepare(",
        "c.finalize()",
        "mx.eval([c.state for c in self.prompt_cache])",
    )
    if any(token not in source for token in required):
        return False, "MLX-LM prompt loop does not match the validated contract"
    return True, "validated staggered chunk scheduler and queued inter-stage sends"


@dataclass(frozen=True)
class PrefillSlot:
    """One rank's work in a fill/drain pipeline timeline."""

    iteration: int
    start: int | None
    end: int | None

    @property
    def is_real(self) -> bool:
        return self.start is not None


def pipeline_prefill_schedule(
    token_count: int,
    prefill_step_size: int,
    *,
    rank: int,
    world_size: int,
) -> tuple[PrefillSlot, ...]:
    """Return the Exo-style staggered fill/steady/drain schedule for one rank.

    Dummy slots do not issue a collective. Pipeline ``recv``/``send`` calls
    provide the actual dependency between adjacent ranks; issuing a different
    collective in a dummy slot would reorder the distributed graph and can
    deadlock. The slots remain explicit so telemetry and tests can prove every
    rank has the same total timeline and the expected offset.
    """

    if token_count < 0:
        raise ValueError("token_count must be non-negative")
    if prefill_step_size < 1:
        raise ValueError("prefill_step_size must be positive")
    if world_size < 2:
        raise ValueError("pipeline prefill requires at least two ranks")
    if not 0 <= rank < world_size:
        raise ValueError("rank must be inside the pipeline world")

    chunks = int(math.ceil(token_count / prefill_step_size))
    slots: list[PrefillSlot] = []
    total = chunks + world_size - 1
    # MLX-LM's pipeline flows from the highest rank to rank zero: rank r
    # receives from r+1 and sends to r-1. Exo's native pipeline numbers the
    # source stage as rank zero, so its ``rank`` leading-dummy formula must be
    # mirrored here. Using it verbatim makes rank zero block in recv while the
    # highest rank is still in a dummy slot.
    leading = world_size - 1 - rank
    for iteration in range(total):
        chunk = iteration - leading
        if 0 <= chunk < chunks:
            start = chunk * prefill_step_size
            slots.append(
                PrefillSlot(
                    iteration=iteration,
                    start=start,
                    end=min(start + prefill_step_size, token_count),
                )
            )
        else:
            slots.append(PrefillSlot(iteration=iteration, start=None, end=None))
    return tuple(slots)


@contextmanager
def install_runtime_optimizations(
    model: Any,
    group: Any,
    execution: ExecutionSettings,
    *,
    batchable: bool,
    pipeline_parallel: bool = True,
) -> Iterator[dict[str, dict[str, Any]]]:
    """Install opt-in token-only output while reporting every capability."""

    import mlx.core as mx

    mlx_generate = importlib.import_module("mlx_lm.generate")

    pipeline_model = getattr(model, "model", None)
    world_size = int(group.size())
    sampling_supported, sampling_reason = _supports_coordinator_sampling(
        pipeline_model,
        batchable=batchable,
        world_size=world_size,
    )
    if not pipeline_parallel:
        sampling_supported = False
        sampling_reason = "pure tensor parallelism keeps MLX-LM's synchronized sampler"
    generation_batch_cls = getattr(mlx_generate, "GenerationBatch", None)
    prompt_batch_cls = getattr(mlx_generate, "PromptProcessingBatch", None)
    native_async = (
        _supports_native_async_step(generation_batch_cls)
        if generation_batch_cls
        else False
    )
    prompt_supported, prompt_reason = (
        _supports_pipeline_prompt(prompt_batch_cls)
        if prompt_batch_cls
        else (False, "MLX-LM has no prompt-processing batch")
    )
    sampling_active = execution.sampling_rank_only and sampling_supported
    (
        rank_zero_logits_supported,
        output_vocab_size,
        rank_zero_logits_reason,
    ) = _supports_rank_zero_logits(model)
    rank_zero_logits_active = sampling_active and rank_zero_logits_supported
    prefill_active = (
        execution.async_overlap
        and sampling_active
        and prompt_supported
        and pipeline_parallel
        and execution.prefill_step_size > 1
    )
    batching_enabled = execution.pipeline_microbatch_size > 1
    batching_active = batching_enabled and batchable
    capabilities = {
        "coalesced_batching": _capability(
            enabled=batching_enabled,
            active=batching_active,
            reason=(
                "MLX-LM continuous batching coalesces up to "
                f"{execution.pipeline_microbatch_size} requests per target batch"
                if batchable
                else (
                    "this model's KV cache cannot be merged, so MLX-LM serves "
                    "requests sequentially"
                )
            ),
        ),
        "sampling_rank_only": _capability(
            enabled=execution.sampling_rank_only,
            active=sampling_active,
            reason=(
                sampling_reason
                if execution.sampling_rank_only
                else "experimental optimization is disabled"
            ),
        ),
        "rank_zero_logits": _capability(
            enabled=execution.sampling_rank_only,
            active=rank_zero_logits_active,
            reason=rank_zero_logits_reason,
        ),
        "async_overlap": _capability(
            enabled=execution.async_overlap,
            active=execution.async_overlap and native_async,
            reason=(
                "pinned MLX-LM GenerationBatch dispatches the next token with "
                "mx.async_eval"
                if native_async
                else "pinned generation step has no validated async dispatch"
            ),
        ),
        "cache_affinity": _capability(
            enabled=execution.cache_affinity,
            active=execution.cache_affinity,
            reason=(
                "all requests for this model stay on one persistent deployment "
                "and its rank-local prompt caches"
                if execution.cache_affinity
                else "deployment cache affinity is disabled"
            ),
        ),
        "pipeline_prefill_overlap": _capability(
            enabled=execution.async_overlap and execution.sampling_rank_only,
            active=prefill_active,
            reason=(
                prompt_reason
                if prefill_active
                else (
                    prompt_reason
                    if sampling_active
                    else (
                        "requires the validated rank-zero sampling path; this model "
                        "keeps MLX-LM's synchronized prefill"
                    )
                )
            ),
        ),
    }
    if not sampling_active:
        yield capabilities
        return

    original_all_gather = mx.distributed.all_gather
    original_send = mx.distributed.send
    original_pipeline_call = type(pipeline_model).__call__
    original_generation_step = mlx_generate.GenerationBatch._step
    original_prompt = (
        mlx_generate.PromptProcessingBatch.prompt if prefill_active else None
    )
    local_state = threading.local()

    def selective_all_gather(value: Any, *args: Any, **kwargs: Any) -> Any:
        if getattr(local_state, "skip_final_gather", False):
            return value
        return original_all_gather(value, *args, **kwargs)

    def local_pipeline_output(
        instance: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        previous = getattr(local_state, "skip_final_gather", False)
        local_state.skip_final_gather = True
        try:
            return original_pipeline_call(instance, *args, **kwargs)
        finally:
            local_state.skip_final_gather = previous

    def queued_pipeline_send(value: Any, *args: Any, **kwargs: Any) -> Any:
        """Materialize a stage output and defer only the transport operation."""

        if not getattr(local_state, "queue_prefill_sends", False):
            return original_send(value, *args, **kwargs)
        # Breaking the graph here is essential. Without it, the send remains
        # entangled with the entire layer graph and the downstream recv cannot
        # make forward progress while this rank starts its next chunk.
        mx.eval(value)
        pending = getattr(local_state, "pending_prefill_sends", None)
        if pending is None:
            pending = []
            local_state.pending_prefill_sends = pending
        pending.append((value, args, kwargs))
        return value

    def flush_prefill_sends() -> None:
        pending = getattr(local_state, "pending_prefill_sends", [])
        local_state.pending_prefill_sends = []
        for value, args, kwargs in pending:
            sent = original_send(value, *args, **kwargs)
            mx.async_eval(sent)

    def staggered_pipeline_prompt(instance: Any, tokens: list[list[int]]) -> None:
        """Pinned PromptProcessingBatch.prompt with pipeline fill/drain."""

        if len(instance.uids) != len(tokens):
            raise ValueError("The batch length doesn't match the number of inputs")
        if not tokens:
            return
        before_prompt = getattr(instance, "_omlx_before_prompt", None)
        if callable(before_prompt):
            before_prompt()

        for stored, incoming in zip(instance.tokens, tokens):
            stored += incoming

        lengths = [len(prompt) for prompt in tokens]
        max_length = max(lengths)
        padding = [max_length - length for length in lengths]
        max_padding = max(padding)
        if max_padding > 0:
            tokens_array = mlx_generate._right_pad_prompts(
                tokens,
                max_length=max_length,
            )
            for cache in instance.prompt_cache:
                cache.prepare(lengths=lengths, right_padding=padding)
        else:
            tokens_array = mx.array(tokens)

        # ``prefill_step_size`` is already the memory-admitted chunk size used
        # by MLX-LM and the rank prefill guard. Dividing it by the world size
        # here made a two-rank 4096-token deployment execute 2048-token chunks,
        # doubling every cache-state barrier and send boundary on long prompts.
        #
        # Staggering still overlaps adjacent stages: it is the rank offset in
        # ``pipeline_prefill_schedule`` that creates fill/steady/drain, not a
        # private reduction of the guarded compute chunk.
        step = max(1, int(instance.prefill_step_size))
        schedule = pipeline_prefill_schedule(
            int(tokens_array.shape[1]),
            step,
            rank=int(group.rank()),
            world_size=world_size,
        )
        local_state.pending_prefill_sends = []
        try:
            for slot in schedule:
                if not slot.is_real:
                    continue
                local_state.queue_prefill_sends = True
                try:
                    instance.model(
                        tokens_array[:, slot.start : slot.end],
                        cache=instance.prompt_cache,
                    )
                finally:
                    local_state.queue_prefill_sends = False
                flush_prefill_sends()
                mx.eval([cache.state for cache in instance.prompt_cache])
                mx.clear_cache()
        finally:
            local_state.queue_prefill_sends = False
            # A cancelled/failed prefill must never leak an old activation into
            # the next request.
            local_state.pending_prefill_sends = []

        if max_padding > 0:
            for cache in instance.prompt_cache:
                cache.finalize()
            mx.eval([cache.state for cache in instance.prompt_cache])
            mx.clear_cache()

    def coordinator_generation_step(instance: Any) -> Any:
        """Pinned GenerationBatch._step with one token collective per batch."""

        instance._current_tokens = instance._next_tokens
        instance._current_logprobs = instance._next_logprobs
        inputs = instance._current_tokens
        coordinator = int(group.rank()) == 0

        if coordinator or not rank_zero_logits_active:
            logits = instance.model(inputs[:, None], cache=instance.prompt_cache)
            logits = logits[:, -1, :]
        else:
            instance.model(
                inputs[:, None],
                cache=instance.prompt_cache,
                skip_logits=True,
            )
            # The token all-sum must be issued after this rank's stage send.
            # MiniMax anchors that lazy send in its last KV cache entry, so
            # materializing the cache state both advances the cache and fixes
            # the distributed operation order without paying for an LM head.
            cache_states = [cache.state for cache in instance.prompt_cache]
            if not cache_states:
                raise RuntimeError(
                    "rank-zero logits requires a cache state to anchor the "
                    "worker-stage send"
                )
            mx.eval(cache_states)
            logits = None

        token_context = []
        if any(instance.logits_processors):
            token_context = [
                token_buffer.update_and_fetch(inputs[index : index + 1])
                for index, token_buffer in enumerate(instance._token_context)
            ]
            if logits is not None:
                processed_logits = []
                for index in range(len(instance.uids)):
                    sample_logits = logits[index : index + 1]
                    for processor in instance.logits_processors[index]:
                        sample_logits = processor(
                            token_context[index],
                            sample_logits,
                        )
                    processed_logits.append(sample_logits)
                logits = mx.concatenate(processed_logits, axis=0)

        if logits is not None:
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        else:
            # Worker ResponseGenerator instances still index this vector while
            # draining their private response queues. Its values never leave
            # the worker, but it must retain the real vocabulary width.
            placeholder = mx.zeros((output_vocab_size,), dtype=mx.float32)
            logprobs = mx.stack([placeholder] * len(instance.uids))

        if coordinator:
            if any(instance.samplers):
                all_samples = []
                for index in range(len(instance.uids)):
                    sampler = instance.samplers[index] or instance.fallback_sampler
                    all_samples.append(sampler(logprobs[index : index + 1]))
                sampled = mx.concatenate(all_samples, axis=0)
            else:
                sampled = instance.fallback_sampler(logprobs)
        else:
            sampled = mx.zeros((len(instance.uids),), dtype=mx.uint32)

        # Rank zero contributes the selected IDs; all other ranks contribute
        # zeros. Every rank therefore advances the same local KV state without
        # gathering a hidden-state tensor.
        sampled = mx.distributed.all_sum(sampled, group=group)
        instance._next_tokens = sampled
        instance._next_logprobs = list(logprobs)
        mx.async_eval(
            instance._next_tokens,
            instance._next_logprobs,
            token_context,
        )

        mx.eval(inputs, instance._current_logprobs)
        input_values = inputs.tolist()
        for sequence_tokens, token in zip(instance.tokens, input_values):
            sequence_tokens.append(token)
        return input_values, instance._current_logprobs

    mx.distributed.all_gather = selective_all_gather
    mx.distributed.send = queued_pipeline_send
    type(pipeline_model).__call__ = local_pipeline_output
    mlx_generate.GenerationBatch._step = coordinator_generation_step
    if prefill_active:
        mlx_generate.PromptProcessingBatch.prompt = staggered_pipeline_prompt
    try:
        yield capabilities
    finally:
        if original_prompt is not None:
            mlx_generate.PromptProcessingBatch.prompt = original_prompt
        mlx_generate.GenerationBatch._step = original_generation_step
        type(pipeline_model).__call__ = original_pipeline_call
        mx.distributed.send = original_send
        mx.distributed.all_gather = original_all_gather
