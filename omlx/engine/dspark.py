# SPDX-License-Identifier: Apache-2.0
"""DSpark speculative decoding engine for Bonsai 27B.

This engine wraps the Bonsai DSpark drafter + target to provide ~2-3x faster
decoding on Apple Silicon via speculative decode. Architecture:

  Target : Bonsai 27B Qwen3.5 hybrid SSM+FA VLM (loaded via mlx-vlm)
  Drafter: 6-layer cross-attention DSpark drafter with log-SNR conditioning

Configuration (model_settings):
  dspark_enabled          : bool  – enable this engine
  dspark_draft_model      : str   – path to the converted drafter directory
                                    (contains config.json + drafter.safetensors)
                                    OR path to the raw GGUF file (auto-converts)
  dspark_max_draft_tokens : int   – draft cap per round (default 2)
  dspark_log_snr          : float – log-SNR inference scalar (default 10.0)

If ``dspark_draft_model`` points to a ``*.gguf`` file, convert_gguf() is called
automatically the first time and cached next to the GGUF.
"""

from __future__ import annotations

import asyncio
import copy
import gc
import json
import logging
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import mlx.core as mx

from ..utils.model_loading import maybe_apply_pre_load_patches
from .base import BaseEngine, GenerationOutput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_drafter_dir(draft_path: str | Path) -> Path:
    """Return the safetensors directory for the drafter.

    If ``draft_path`` is a ``.gguf`` file, auto-convert it and return the
    output directory (sibling to the GGUF with '_mlx' suffix).
    """
    p = Path(draft_path)
    if p.suffix.lower() == ".gguf":
        out_dir = p.parent / (p.stem + "_mlx")
        if not (out_dir / "config.json").exists():
            logger.info("Auto-converting GGUF drafter: %s → %s", p, out_dir)
            from ..custom_kernels.bonsai.dspark.convert import convert_gguf
            convert_gguf(p, out_dir)
        return out_dir
    return p


def _load_drafter(drafter_dir: Path, target_model) -> "BonsaiDSparkDrafter":  # noqa: F821
    """Load BonsaiDSparkDrafter from a converted directory and bind embeddings."""
    from ..custom_kernels.bonsai.dspark.config import BonsaiDSparkConfig
    from ..custom_kernels.bonsai.dspark.drafter import BonsaiDSparkDrafter

    config = BonsaiDSparkConfig.from_json(drafter_dir / "config.json")
    drafter = BonsaiDSparkDrafter(config)
    drafter.load_weights(str(drafter_dir / "drafter.safetensors"))
    drafter.bind_target_embedding(target_model)
    mx.eval(drafter.parameters())
    return drafter


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class DSParkEngine(BaseEngine):
    """Bonsai DSpark speculative decoding engine.

    Supports batch=1 decoding only. For multi-request concurrency, add a
    queueing layer (the server's request queue) in front of this engine.
    """

    def __init__(
        self,
        model_name: str,
        draft_model_path: str,
        model_settings: Any | None = None,
    ):
        self._model_name = model_name
        self._draft_model_path = draft_model_path
        self._model_settings = model_settings

        self._target = None       # BonsaiTarget
        self._drafter = None      # BonsaiDSparkDrafter
        self._tokenizer_obj = None
        self._executor_tokenizer = None
        self._loaded = False

        self._max_draft_tokens: int = int(
            getattr(model_settings, "dspark_max_draft_tokens", 2) or 2
        )
        self._log_snr: float = float(
            getattr(model_settings, "dspark_log_snr", 10.0) or 10.0
        )

    # -----------------------------------------------------------------------
    # BaseEngine properties
    # -----------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def tokenizer(self):
        return self._tokenizer_obj

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    async def start(self) -> None:
        if self._loaded:
            return

        from ..engine_core import get_mlx_executor

        loop = asyncio.get_running_loop()
        trc = bool(getattr(self._model_settings, "trust_remote_code", False))
        draft_path = self._draft_model_path
        log_snr = self._log_snr

        def _load():
            import mlx_vlm

            maybe_apply_pre_load_patches(
                self._model_name,
                model_settings=self._model_settings,
                for_vlm=True,
            )

            logger.info("DSPark: loading target model from %s", self._model_name)
            model, tokenizer = mlx_vlm.load(
                self._model_name,
                trust_remote_code=trc,
            )

            drafter_dir = _resolve_drafter_dir(draft_path)
            logger.info("DSPark: loading drafter from %s", drafter_dir)
            drafter = _load_drafter(drafter_dir, model)

            from ..custom_kernels.bonsai.dspark.target import BonsaiTarget

            target = BonsaiTarget(model, tokenizer)
            return target, drafter, tokenizer

        target, drafter, tokenizer = await loop.run_in_executor(
            get_mlx_executor(), _load
        )
        self._target = target
        self._drafter = drafter
        self._tokenizer_obj = tokenizer
        self._executor_tokenizer = copy.deepcopy(tokenizer)
        self._loaded = True
        logger.info("DSPark engine loaded for %s", self._model_name)

    async def stop(self) -> None:
        self._target = None
        self._drafter = None
        self._tokenizer_obj = None
        self._executor_tokenizer = None
        self._loaded = False
        gc.collect()

    # -----------------------------------------------------------------------
    # Generation helpers
    # -----------------------------------------------------------------------

    def _encode_messages(self, messages: list[dict], **kwargs) -> list[int]:
        tok = self._executor_tokenizer or self._tokenizer_obj
        try:
            from mlx_dspark.generate import encode_messages
            return encode_messages(tok, messages, **kwargs)
        except ImportError:
            if hasattr(tok, "apply_chat_template"):
                r = tok.apply_chat_template(messages, add_generation_prompt=True)
                if isinstance(r, list):
                    return r
            return list(tok.encode(" ".join(m.get("content", "") for m in messages)))

    def _encode_prompt(self, prompt: str) -> list[int]:
        return self._encode_messages([{"role": "user", "content": prompt}])

    def _eos_ids(self) -> set[int]:
        try:
            from mlx_dspark.generate import eos_token_ids
            return eos_token_ids(self._executor_tokenizer or self._tokenizer_obj)
        except ImportError:
            tok = self._executor_tokenizer or self._tokenizer_obj
            ids: set[int] = set()
            for attr in ("eos_token_id", "eos_token_ids"):
                v = getattr(tok, attr, None)
                if isinstance(v, int):
                    ids.add(v)
                elif v:
                    ids.update(int(x) for x in v)
            return ids

    def _run_generate(
        self,
        prompt_ids: list[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        presence_penalty: float,
        frequency_penalty: float,
        stop: list[str] | None,
        on_text=None,
    ) -> GenerationOutput:
        """Run speculative generate on the executor thread (blocking)."""
        from ..custom_kernels.bonsai.dspark.generate import (
            GenResult,
            speculative_generate,
        )

        result: GenResult = speculative_generate(
            self._target,
            self._executor_tokenizer,
            self._drafter,
            prompt_ids=prompt_ids,
            max_new_tokens=max_tokens,
            max_draft_tokens=self._max_draft_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            stop=stop,
            on_text=on_text,
            log_snr=self._log_snr,
        )
        return GenerationOutput(
            text=result.text,
            tokens=result.token_ids,
            prompt_tokens=len(prompt_ids),
            completion_tokens=result.num_tokens,
            finish_reason=result.finish_reason,
            generation_tps=result.tokens_per_sec,
            new_text=result.text,
            finished=True,
        )

    # -----------------------------------------------------------------------
    # BaseEngine interface
    # -----------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: Optional[list[str]] = None,
        **kwargs,
    ) -> GenerationOutput:
        messages = kwargs.get("messages")
        if messages:
            prompt_ids = self._encode_messages(messages)
        else:
            prompt_ids = self._encode_prompt(prompt)

        loop = asyncio.get_running_loop()
        from ..engine_core import get_mlx_executor

        return await loop.run_in_executor(
            get_mlx_executor(),
            lambda: self._run_generate(
                prompt_ids=prompt_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                stop=stop or [],
            ),
        )

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: Optional[list[str]] = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        messages = kwargs.get("messages")
        if messages:
            prompt_ids = self._encode_messages(messages)
        else:
            prompt_ids = self._encode_prompt(prompt)

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def on_text_cb(text: str):
            loop.call_soon_threadsafe(queue.put_nowait, text)

        from ..engine_core import get_mlx_executor

        fut = loop.run_in_executor(
            get_mlx_executor(),
            lambda: self._run_generate(
                prompt_ids=prompt_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                stop=stop or [],
                on_text=on_text_cb,
            ),
        )

        emitted = ""
        while True:
            try:
                chunk = await asyncio.wait_for(queue.get(), timeout=0.05)
                new_text = chunk[len(emitted):]
                emitted = chunk
                yield GenerationOutput(
                    text=chunk,
                    new_text=new_text,
                    finished=False,
                    finish_reason=None,
                )
            except asyncio.TimeoutError:
                if fut.done():
                    break

        result = await fut
        final_new = result.text[len(emitted):]
        yield GenerationOutput(
            text=result.text,
            new_text=final_new,
            tokens=result.tokens,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            finish_reason=result.finish_reason,
            generation_tps=result.generation_tps,
            finished=True,
        )
