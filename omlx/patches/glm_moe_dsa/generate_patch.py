# SPDX-License-Identifier: Apache-2.0
"""GLM DSA adaptive prefill patches for ``mlx_lm.generate``.

This ports the small GLM-specific prefill chunking change from the optimized
mlx-lm snapshot without replacing the whole ``mlx_lm.generate`` module. The
patch is intentionally inert for non-GLM models.
"""

from __future__ import annotations

import importlib
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

_APPLIED = False


@dataclass(frozen=True)
class _AdaptivePrefillConfig:
    step_size: int
    after: int
    min_remaining: int


def _glm_dsa_adaptive_prefill_config(
    model: Any, prefill_step_size: int
) -> _AdaptivePrefillConfig | None:
    model_type = getattr(model, "model_type", None) or getattr(
        getattr(model, "args", None), "model_type", None
    )
    if (
        model_type != "glm_moe_dsa"
        or prefill_step_size != 2048
        or os.environ.get("MLX_LM_GLM_DSA_ADAPTIVE_PREFILL_STEP", "1") != "1"
    ):
        return None

    return _AdaptivePrefillConfig(
        step_size=int(
            os.environ.get("MLX_LM_GLM_DSA_ADAPTIVE_PREFILL_STEP_SIZE", "8192")
        ),
        after=int(os.environ.get("MLX_LM_GLM_DSA_ADAPTIVE_PREFILL_AFTER", "0")),
        min_remaining=int(
            os.environ.get("MLX_LM_GLM_DSA_ADAPTIVE_PREFILL_MIN_REMAINING", "0")
        ),
    )


def _prefill_step_size_for_progress(
    prefill_step_size: int,
    processed_tokens: int,
    remaining_tokens: int,
    adaptive_prefill: _AdaptivePrefillConfig | None,
) -> int:
    if (
        adaptive_prefill is not None
        and processed_tokens >= adaptive_prefill.after
        and remaining_tokens >= adaptive_prefill.min_remaining
    ):
        return adaptive_prefill.step_size
    return prefill_step_size


def apply_glm_moe_dsa_generate_patch() -> bool:
    """Patch ``mlx_lm.generate`` with GLM-only adaptive prefill behavior."""
    global _APPLIED
    if _APPLIED:
        return False

    try:
        gen = importlib.import_module("mlx_lm.generate")
    except Exception:
        logger.debug("mlx_lm.generate not importable - GLM prefill patch skipped")
        return False

    PromptProcessingBatch = gen.PromptProcessingBatch
    BatchGenerator = gen.BatchGenerator

    if (
        getattr(PromptProcessingBatch, "_omlx_glm_dsa_adaptive_patched", False)
        and getattr(gen.generate_step, "_omlx_glm_dsa_adaptive_patched", False)
    ):
        _APPLIED = True
        return False

    original_ppb_init = PromptProcessingBatch.__init__
    original_ppb_copy = PromptProcessingBatch._copy
    original_ppb_split = PromptProcessingBatch.split
    original_ppb_prompt = PromptProcessingBatch.prompt
    original_bg_init = BatchGenerator.__init__
    original_bg_next = BatchGenerator._next
    original_generate_step = gen.generate_step

    def patched_ppb_init(self, *args, **kwargs):
        original_ppb_init(self, *args, **kwargs)
        model = getattr(self, "model", None)
        prefill_step_size = getattr(self, "prefill_step_size", 2048)
        self._omlx_glm_dsa_adaptive_prefill = _glm_dsa_adaptive_prefill_config(
            model, prefill_step_size
        )

    def patched_ppb_copy(self):
        new_batch = original_ppb_copy(self)
        new_batch._omlx_glm_dsa_adaptive_prefill = getattr(
            self, "_omlx_glm_dsa_adaptive_prefill", None
        )
        return new_batch

    def patched_ppb_split(self, indices):
        new_batch = original_ppb_split(self, indices)
        if not hasattr(new_batch, "_omlx_glm_dsa_adaptive_prefill"):
            new_batch._omlx_glm_dsa_adaptive_prefill = getattr(
                self, "_omlx_glm_dsa_adaptive_prefill", None
            )
        return new_batch

    def patched_ppb_prompt(self, tokens):
        adaptive_prefill = getattr(self, "_omlx_glm_dsa_adaptive_prefill", None)
        if adaptive_prefill is None:
            return original_ppb_prompt(self, tokens)

        if len(self.uids) != len(tokens):
            raise ValueError("The batch length doesn't match the number of inputs")
        if not tokens:
            return None

        processed_tokens = min(len(sti) for sti in self.tokens)
        for sti, ti in zip(self.tokens, tokens):
            sti += ti

        lengths = [len(p) for p in tokens]
        max_length = max(lengths)
        padding = [max_length - length for length in lengths]
        max_padding = max(padding)

        if max_padding > 0:
            tokens = gen._right_pad_prompts(tokens, max_length=max_length)
            for cache in self.prompt_cache:
                cache.prepare(lengths=lengths, right_padding=padding)
        else:
            tokens = mx.array(tokens)

        while tokens.shape[1] > 0:
            remaining = tokens.shape[1]
            step_size = _prefill_step_size_for_progress(
                self.prefill_step_size,
                processed_tokens,
                remaining,
                adaptive_prefill,
            )
            n_to_process = min(step_size, remaining)
            self.model(tokens[:, :n_to_process], cache=self.prompt_cache)
            mx.eval([cache.state for cache in self.prompt_cache])
            mx.clear_cache()
            tokens = tokens[:, n_to_process:]
            processed_tokens += n_to_process

        if max_padding > 0:
            for cache in self.prompt_cache:
                cache.finalize()
            mx.eval([cache.state for cache in self.prompt_cache])
            mx.clear_cache()
        return None

    def patched_bg_init(self, *args, **kwargs):
        original_bg_init(self, *args, **kwargs)
        model = getattr(self, "model", None)
        prefill_step_size = getattr(self, "prefill_step_size", 2048)
        self._omlx_glm_dsa_adaptive_prefill = _glm_dsa_adaptive_prefill_config(
            model, prefill_step_size
        )
        self._prompt_batch._omlx_glm_dsa_adaptive_prefill = (
            self._omlx_glm_dsa_adaptive_prefill
        )

    def patched_bg_next(self):
        adaptive_prefill = getattr(self, "_omlx_glm_dsa_adaptive_prefill", None)
        if adaptive_prefill is None:
            return original_bg_next(self)

        generation_responses = []
        prompt_responses = []

        if len(self._generation_batch) > 0:
            generation_responses = self._generation_batch.next()
            self._gen_tokens_counter += len(generation_responses)
            self._steps_counter += 1
            if self._steps_counter % 512 == 0:
                mx.clear_cache()

        if len(self._generation_batch) >= self.completion_batch_size:
            return prompt_responses, generation_responses

        n = min(
            self.prefill_batch_size - len(self._prompt_batch),
            self.completion_batch_size - len(self._generation_batch),
            len(self._unprocessed_sequences),
        )
        if n > 0:
            self._prompt_batch.extend(self._make_batch(n))

        keep = []
        split = []
        for i, seq in enumerate(self._currently_processing):
            segments = seq[0]
            if len(segments) == 1 and len(segments[0]) == 1:
                split.append(i)
            else:
                keep.append(i)

        if split:
            last_inputs = [self._currently_processing[i][0][0] for i in split]
            progress = [(self._currently_processing[i][2],) * 2 for i in split]
            self._currently_processing = [self._currently_processing[i] for i in keep]
            gen_batch = self._prompt_batch.split(split).generate(last_inputs)
            for i, progress_item in enumerate(progress):
                prompt_responses.append(
                    gen.PromptProcessingBatch.Response(
                        gen_batch.uids[i],
                        progress_item,
                        True,
                        True,
                    )
                )
            self._generation_batch.extend(gen_batch)

        prompts = []
        for i, seq in enumerate(self._currently_processing):
            response = gen.PromptProcessingBatch.Response(
                self._prompt_batch.uids[i], 0, False, False
            )
            segments = seq[0]
            remaining = len(segments[0])
            step_size = _prefill_step_size_for_progress(
                self.prefill_step_size,
                seq[1],
                remaining,
                adaptive_prefill,
            )
            n = min(remaining, step_size)
            prompts.append(segments[0][:n])
            segments[0] = segments[0][n:]
            if len(segments[0]) == 0:
                segments.pop(0)
                response.end_of_segment = True
            seq[1] += len(prompts[-1])
            response.progress = (seq[1], seq[2])
            prompt_responses.append(response)

        self._prompt_tokens_counter += sum(len(prompt) for prompt in prompts)
        tic = time.perf_counter()
        self._prompt_batch.prompt(prompts)
        toc = time.perf_counter()
        self._prompt_time_counter += toc - tic

        return prompt_responses, generation_responses

    def patched_generate_step(prompt, model, *args, **kwargs):
        prefill_step_size = kwargs.get("prefill_step_size", 2048)
        adaptive_prefill = _glm_dsa_adaptive_prefill_config(
            model, prefill_step_size
        )
        if (
            adaptive_prefill is not None
            and adaptive_prefill.after == 0
            and adaptive_prefill.min_remaining == 0
        ):
            kwargs["prefill_step_size"] = adaptive_prefill.step_size
        return original_generate_step(prompt, model, *args, **kwargs)

    PromptProcessingBatch.__init__ = patched_ppb_init
    PromptProcessingBatch._copy = patched_ppb_copy
    PromptProcessingBatch.split = patched_ppb_split
    PromptProcessingBatch.prompt = patched_ppb_prompt
    BatchGenerator.__init__ = patched_bg_init
    BatchGenerator._next = patched_bg_next
    patched_generate_step._omlx_glm_dsa_adaptive_patched = True
    gen.generate_step = patched_generate_step

    PromptProcessingBatch._omlx_glm_dsa_adaptive_patched = True
    BatchGenerator._omlx_glm_dsa_adaptive_patched = True
    _APPLIED = True
    logger.info("GLM MoE DSA adaptive prefill patch applied to mlx_lm.generate")
    return True


__all__ = ["apply_glm_moe_dsa_generate_patch"]
