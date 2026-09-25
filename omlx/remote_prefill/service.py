# SPDX-License-Identifier: Apache-2.0
"""Scheduler side of remote prefill: hold long prompts while vLLM prefills them, then hand over their caches."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from ..cluster.rdma.mailbox import ClientMailbox
from ..cluster.rdma.words import load_word_ops
from .client import request_prefill
from .inject import SUPPORTED_CACHES, extend_caches, layer_updates
from .job import PrefillJob
from .receiver import HandoffError
from .settings import RemotePrefillSettings

logger = logging.getLogger(__name__)
# After this many failures in a row remote prefill pauses, so a broken path costs one wait, not many.
_FAILURES_BEFORE_PAUSE = 3
_PAUSE_S = 60.0


def _activation_dtype(mx: Any, model: Any) -> Any:
    """The dtype the model computes in, read from its token embedding."""
    embed = getattr(getattr(model, "model", None), "embed_tokens", None)
    for name in ("scales", "weight"):
        value = getattr(embed, name, None)
        if value is not None and getattr(value, "dtype", None) in (
            mx.bfloat16,
            mx.float16,
            mx.float32,
        ):
            return value.dtype
    return mx.bfloat16


class RemotePrefill:
    """Decides which requests prefill on the vLLM server and hands their KV caches to the scheduler."""

    def __init__(
        self,
        scheduler: Any,
        settings: RemotePrefillSettings,
        *,
        attach: Callable[[str], ClientMailbox] | None = None,
        requester: Callable[..., None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        import mlx.core as mx

        self._mx = mx
        self._scheduler = scheduler
        self.settings = settings
        self._attach = attach or self._attach_link
        self._requester = requester or request_prefill
        self._clock = clock
        self._jobs: dict[str, PrefillJob] = {}
        self._failures = 0
        self._paused_until = 0.0
        self._cache_reason: str | None = None
        self.last: dict[str, Any] = {}
        self.last_error = ""

    @classmethod
    def for_scheduler(cls, scheduler: Any) -> RemotePrefill | None:
        """The scheduler's remote prefill, or None when none is configured for its model."""
        try:
            settings = RemotePrefillSettings.from_env()
        except ValueError as exc:
            logger.warning("Remote prefill is off: %s", exc)
            return None
        if settings is None or settings.local_model != scheduler.config.model_name:
            return None
        logger.info(
            "Remote prefill for %s from %s over %s",
            settings.local_model,
            settings.url,
            ", ".join(settings.links),
        )
        return cls(scheduler, settings)

    @staticmethod
    def _attach_link(name: str) -> ClientMailbox:
        ops, reason = load_word_ops()
        if ops is None:
            raise HandoffError(reason)
        return ClientMailbox.attach(name, ops)

    def _cache_support(self) -> str:
        """Why the model's caches cannot take a handoff, or an empty string when they can."""
        if self._cache_reason is None:
            from mlx_lm.models.cache import make_prompt_cache

            try:
                kinds = {
                    type(cache).__name__
                    for cache in make_prompt_cache(self._scheduler.model)
                }
            except Exception as exc:
                self._cache_reason = f"its caches could not be probed: {exc}"
            else:
                unsupported = sorted(kinds - SUPPORTED_CACHES)
                self._cache_reason = (
                    f"cache types {unsupported} cannot take a handoff"
                    if unsupported
                    else ""
                )
        return self._cache_reason

    @staticmethod
    def _end(request: Any) -> int:
        """Prompt tokens to prefill remotely: all but the generation prompt, which runs here."""
        prompt = request.prompt_token_ids or []
        start = getattr(request, "generation_prompt_start", 0)
        if isinstance(start, int) and 0 < start < len(prompt):
            return start
        return len(prompt) - 1

    def _local_prefix(self, request: Any, end: int) -> int:
        """Tokens the local prefix cache already holds, so vLLM exports only the rest."""
        cache = getattr(self._scheduler, "block_aware_cache", None)
        if cache is None:
            return 0
        try:
            blocks, _, _ = cache.paged_cache.find_shared_prefix(
                request.prompt_token_ids[:end]
            )
        except Exception:
            return 0
        return min(end, len(blocks) * self._scheduler.config.paged_cache_block_size)

    def _ineligible(self, request: Any, end: int) -> str:
        if self._clock() < self._paused_until:
            return "paused after repeated failures"
        if end < self.settings.min_tokens:
            return "the prompt is short"
        if request.vlm_extra_keys_for_cache or request.specprefill_indices is not None:
            return "multimodal and SpecPrefill prompts prefill here"
        return self._cache_support()

    def defer(self, request: Any) -> bool:
        """Whether `request` waits because its prompt is prefilling remotely; may start that prefill."""
        job = self._jobs.get(request.request_id)
        if job is not None:
            return job.running
        end = self._end(request)
        if self._ineligible(request, end):
            return False
        start = self._local_prefix(request, end)
        if end - start < self.settings.min_tokens:
            return False
        job = PrefillJob(
            self._mx,
            self.settings,
            request.prompt_token_ids[:end],
            start,
            attach=self._attach,
            requester=self._requester,
        )
        self._jobs[request.request_id] = job
        job.start()
        logger.info(
            "Request %s: prefilling %d prompt tokens on %s",
            request.request_id,
            end - start,
            self.settings.url,
        )
        return True

    def _failed(self, reason: str) -> None:
        self.last_error = reason
        self._failures += 1
        if self._failures >= _FAILURES_BEFORE_PAUSE:
            self._paused_until = self._clock() + _PAUSE_S
            logger.warning(
                "Remote prefill paused for %.0f s after %d failures: %s",
                _PAUSE_S,
                self._failures,
                reason,
            )

    def inject(self, request: Any) -> None:
        """Append a finished remote prefill to `request`'s cache; any problem leaves it to prefill here."""
        job = self._jobs.pop(request.request_id, None)
        if job is None:
            return
        if job.state != "done" or job.result is None:
            self._failed(job.error or "the remote prefill did not finish")
            return
        cached = request.cached_tokens or 0
        first = job.result.manifests[0].first_token
        if cached >= job.end:
            return
        if cached < first:
            logger.info(
                "Request %s: the local cache no longer reaches the handoff; prefilling here",
                request.request_id,
            )
            return
        mx = self._mx
        from mlx_lm.models.cache import make_prompt_cache

        caches = request.prompt_cache or make_prompt_cache(self._scheduler.model)
        try:
            updates = layer_updates(
                mx,
                job.result,
                cached,
                job.end,
                _activation_dtype(mx, self._scheduler.model),
            )
            extend_caches(mx, caches, updates)
        except (HandoffError, ValueError) as exc:
            # A restored cache may be half extended; the scheduler restarts it from nothing.
            self._failed(f"the handoff could not be applied: {exc}")
            self._reset(request)
            return
        request.prompt_cache = caches
        request.cached_tokens = job.end
        request.remaining_tokens = request.prompt_token_ids[job.end :]
        self._failures = 0
        result = job.result
        seconds = max(result.transfer_s, 1e-9)
        self.last = {
            "tokens": job.end - cached,
            "prefill_s": round(result.prefill_s, 3),
            "transfer_s": round(result.transfer_s, 3),
            "bytes": result.nbytes,
            "gbit_s": round(8 * result.nbytes / seconds / 1e9, 2),
        }
        logger.info(
            "Request %s: remote prefill of %d tokens, prefill %.2f s, transfer %.2f s (%.1f Gbit/s)",
            request.request_id,
            self.last["tokens"],
            result.prefill_s,
            result.transfer_s,
            self.last["gbit_s"],
        )

    def _reset(self, request: Any) -> None:
        manager = getattr(self._scheduler, "paged_cache_manager", None)
        if manager is not None:
            manager.delete_block_table(request.request_id)
        request.prompt_cache = None
        request.block_table = None
        request.cached_tokens = 0
        request.shared_prefix_blocks = 0
        request.remaining_tokens = request.prompt_token_ids

    def forget(self, request_id: str) -> None:
        """Drop the job of a request that left the queue; its thread finishes on its own."""
        self._jobs.pop(request_id, None)

    def clear(self) -> None:
        self._jobs.clear()

    def status(self) -> dict[str, Any]:
        return {
            "url": self.settings.url,
            "model": self.settings.model,
            "local_model": self.settings.local_model,
            "links": list(self.settings.links),
            "min_tokens": self.settings.min_tokens,
            "paused": self._clock() < self._paused_until,
            "failures": self._failures,
            "running": sum(job.running for job in self._jobs.values()),
            "last": dict(self.last),
            "last_error": self.last_error,
            "unsupported": self._cache_reason or "",
        }
