# SPDX-License-Identifier: Apache-2.0
"""Uno setup and request validation for the continuous batching engine."""

from pathlib import Path

import mlx.core as mx

from ..exceptions import InvalidRequestError
from ..model_settings import uno_conflicts
from ..uno_bundle import resolve_uno_bundle
from .batched import BatchedEngine


class UnoEngine(BatchedEngine):
    """Use Uno for one request and ordinary decoding when requests overlap."""

    is_uno_model = True

    def __init__(
        self,
        model_name,
        *,
        adapter_path,
        scheduler_config=None,
        model_settings=None,
        prefill_eviction_callback=None,
    ):
        super().__init__(
            str(model_name),
            scheduler_config=scheduler_config,
            model_settings=model_settings,
            prefill_eviction_callback=prefill_eviction_callback,
        )
        self._adapter_path = adapter_path
        self._bundle = None

    async def _prepare_loaded_model(self) -> None:
        import asyncio

        from ..engine_core import get_mlx_executor

        await asyncio.get_running_loop().run_in_executor(
            get_mlx_executor(), self._prepare_uno_model
        )

    def _prepare_uno_model(self):
        from ..patches.k2_horizon.compiled import (
            can_compile_blocks,
            install_compiled_blocks,
        )
        from ..patches.k2_horizon.quantized import enable_q8_blocks
        from ..patches.k2_horizon.uno_adapter import load_uno_adapter
        from ..patches.k2_horizon.uno_batch import install_cache_hooks

        bundle = resolve_uno_bundle(self._model_name, self._adapter_path)
        load_uno_adapter(
            self._model, bundle.adapter_path, base_model_id=bundle.base_model_id
        )
        mx.eval(self._model.parameters())
        self._model._omlx_uno_q8_block_projections = enable_q8_blocks(self._model)
        if can_compile_blocks(self._model):
            install_compiled_blocks(self._model)
            self._model._omlx_k2_compiled = True
        self._model._omlx_uno_eos = self._tokenizer.eos_token_ids
        self._model._omlx_uno_enabled = True
        self._model._omlx_uno_singleton = False
        install_cache_hooks()
        self._bundle = bundle

    def get_stats(self):
        stats = super().get_stats()
        if self._bundle is not None:
            stats["uno"] = {
                "base": str(Path(self._model_name).resolve()),
                "adapter": str(self._bundle.adapter_path.resolve()),
                "compiled": bool(getattr(self._model, "_omlx_k2_compiled", False)),
                "ane_layers": getattr(self._model, "_omlx_k2_ane_prefill_count", 0),
                "q8_block_projections": getattr(
                    self._model, "_omlx_uno_q8_block_projections", 0
                ),
                **dict(getattr(self._model, "_omlx_uno_stats", {})),
            }
        return stats

    def _validate_request(self, prompt=None, **options) -> None:
        self._validate_options(**options)
        if options.get("compiled_grammar") is not None and not options.get("tools"):
            raise InvalidRequestError("Uno supports K2 tool constraints only")
        if options.get("top_k", 0) >= self._bundle.config["vocab_size"]:
            raise InvalidRequestError("Uno top_k must be smaller than the vocabulary")
        if prompt is None:
            return
        tokens = self._tokenizer.encode(prompt) if isinstance(prompt, str) else prompt
        limit = getattr(self._model_settings, "max_context_window", None)
        if limit and len(tokens) > limit:
            raise InvalidRequestError(
                f"Uno prompt exceeds configured context limit {limit}"
            )
        if len(tokens) + options.get("max_tokens", 256) > self._bundle.context_length:
            raise InvalidRequestError(
                f"Uno prompt plus max_tokens exceeds context length {self._bundle.context_length}"
            )

    @staticmethod
    def _validate_options(
        max_tokens=256,
        temperature=0.7,
        top_p=0.9,
        top_k=0,
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        stop=None,
        **kwargs,
    ):
        import math

        if type(max_tokens) is not int or max_tokens < 0:
            raise InvalidRequestError("Uno max_tokens must be a nonnegative integer")
        if not math.isfinite(temperature) or temperature < 0:
            raise InvalidRequestError("Uno temperature must be finite and nonnegative")
        if not math.isfinite(top_p) or not 0 < top_p <= 1:
            raise InvalidRequestError("Uno top_p must be in (0, 1]")
        if type(top_k) is not int or top_k < 0:
            raise InvalidRequestError("Uno top_k must be a nonnegative integer")
        for name, neutral in uno_conflicts(
            dict(
                kwargs,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
            )
        ).items():
            raise InvalidRequestError(
                f"Uno requires {name}={neutral}. Check request, model, and global sampling settings."
            )
        # Only decoding restrictions belong here. Shared engine metadata such
        # as preserve_reasoning must retain BatchedEngine's normal semantics.
        if kwargs.get("thinking_budget") is not None or kwargs.get("specprefill"):
            raise InvalidRequestError(
                "Uno does not support thinking_budget or specprefill"
            )
        seed = kwargs.get("seed")
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
            raise InvalidRequestError("Uno seed must be an integer in [0, 2**32)")
        stops = [stop] if isinstance(stop, str) else list(stop or [])
        if any(not isinstance(item, str) or not item for item in stops):
            raise InvalidRequestError("Uno stop strings must be nonempty strings")
        return stops
