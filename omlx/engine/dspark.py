"""First-class oMLX dSpark engine.

This is intentionally a thin BatchedEngine specialization: oMLX owns target
loading, Scheduler, request collection, caches, sampling and statistics.  The
only extra object attached at startup is a native speculative draft provider.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import mlx.core as mx

from ..dspark.compat import validate_pair
from ..dspark.provider import NativeDSparkProvider
from ..dspark.smoke import verify_cross_target_smoke
from .batched import BatchedEngine

logger = logging.getLogger(__name__)


class DSparkEngine(BatchedEngine):
    """BatchedEngine with a Scheduler-native dSpark provider."""

    def __init__(
        self,
        *,
        model_name: str,
        draft_model_path: str,
        model_settings: Any,
        scheduler_config: Any | None = None,
        trust_remote_code: bool = False,
        stream_interval: int = 1,
        enable_thinking: bool | None = None,
        prefill_eviction_callback: Any | None = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            trust_remote_code=trust_remote_code,
            scheduler_config=scheduler_config,
            stream_interval=stream_interval,
            enable_thinking=enable_thinking,
            model_settings=model_settings,
            prefill_eviction_callback=prefill_eviction_callback,
        )
        self._draft_model_path = draft_model_path
        self._format_requested = str(getattr(model_settings, "dspark_format", "auto"))
        self._pairing_mode = str(
            getattr(model_settings, "dspark_pairing_mode", "exact")
        )
        self._max_draft_tokens = getattr(
            model_settings, "dspark_max_draft_tokens", "auto"
        )
        self._markov_mode = str(getattr(model_settings, "dspark_markov_mode", "auto"))
        self._compatibility: dict[str, Any] = {}

    async def start(self) -> None:
        if self._loaded:
            return

        compatibility = validate_pair(
            self._model_name,
            self._draft_model_path,
            requested_format=self._format_requested,
            pairing_mode=self._pairing_mode,
        )
        self._compatibility = compatibility.to_dict()
        if not compatibility.compatible:
            reasons = "; ".join(compatibility.blocked_reasons)
            raise ValueError(f"dSpark target/drafter pairing blocked: {reasons}")

        await super().start()
        assert self._engine is not None
        core = self._engine.engine

        def _load_provider() -> NativeDSparkProvider:
            with mx.stream(core._mlx_stream):
                provider = NativeDSparkProvider(
                    target_model=self._model,
                    tokenizer=self._tokenizer,
                    drafter_path=self._draft_model_path,
                    requested_format=self._format_requested,
                    max_draft_tokens=self._max_draft_tokens,
                    markov_mode=self._markov_mode,
                    pairing_mode=self._pairing_mode,
                )
                mx.eval(provider.drafter.parameters())
                if self._pairing_mode == "verified_cross_target":
                    verify_cross_target_smoke(provider, self._tokenizer)
                core.scheduler.set_dspark_provider(provider)
                return provider

        try:
            await asyncio.wrap_future(core._mlx_executor.submit(_load_provider))
        except BaseException:
            await super().stop()
            raise
        logger.info(
            "DSparkEngine loaded natively: target=%s drafter=%s format=%s pairing=%s",
            self._model_name,
            self._draft_model_path,
            compatibility.format,
            self._pairing_mode,
        )

    def get_stats(self) -> dict[str, Any]:
        stats = super().get_stats()
        stats.update(
            {
                "engine_type": "dspark",
                "route": "dspark",
                "draft_model": self._draft_model_path,
                "handler_format": self._compatibility.get("format"),
                "pairing_mode": self._pairing_mode,
                "target_fingerprint": self._compatibility.get("target_fingerprint"),
                "drafter_fingerprint": self._compatibility.get("drafter_fingerprint"),
            }
        )
        return stats
