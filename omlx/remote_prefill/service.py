# SPDX-License-Identifier: Apache-2.0
"""Scheduler side of remote prefill: hold long prompts while vLLM prefills them, then hand over their caches."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from ..cluster.rdma.mailbox import ClientMailbox
from ..cluster.rdma.words import load_word_ops
from .client import request_prefill
from .geometry import KVGeometry, check_manifest, model_geometry
from .inject import SUPPORTED_CACHES, extend_caches, layer_updates
from .job import PrefillJob
from .receiver import HandoffError
from .settings import RemotePrefillSettings

logger = logging.getLogger(__name__)
# After this many failures in a row remote prefill pauses; a silent vLLM or connector pauses it at once.
_FAILURES_BEFORE_PAUSE = 3
# The first pause; each further one before a success doubles, up to the longest.
_PAUSE_S = 60.0
_MAX_PAUSE_S = 900.0


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
        # Requests whose remote prefill already ran; each request gets one at most.
        self._served: set[str] = set()
        # One handoff uses the links at a time, whichever request it belongs to.
        self._links = threading.Lock()
        self._failures = 0
        self._pauses = 0
        self._paused_until = 0.0
        self._cache_reason: str | None = None
        self._geometry: KVGeometry | None = None
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
                caches = make_prompt_cache(self._scheduler.model)
            except Exception as exc:
                self._cache_reason = f"its caches could not be probed: {exc}"
                return self._cache_reason
            unsupported = sorted(
                {type(cache).__name__ for cache in caches} - SUPPORTED_CACHES
            )
            if unsupported:
                self._cache_reason = f"cache types {unsupported} cannot take a handoff"
            else:
                self._geometry, self._cache_reason = model_geometry(
                    self._scheduler.model, len(caches)
                )
        return self._cache_reason

    def _check_model(self, manifest: Any) -> None:
        """Refuse a handoff whose layers do not fit the local model's caches."""
        if self._geometry is not None:
            check_manifest(manifest, self._geometry)

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
        if request.request_id in self._served:
            return False
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
            links=self._links,
            check=self._check_model,
        )
        self._jobs[request.request_id] = job
        self._served.add(request.request_id)
        job.start()
        logger.info(
            "Request %s: prefilling %d prompt tokens on %s",
            request.request_id,
            end - start,
            self.settings.url,
        )
        return True

    def _failed(self, reason: str, *, pause: bool = False) -> None:
        self.last_error = reason
        self._failures += 1
        if pause or self._failures >= _FAILURES_BEFORE_PAUSE:
            seconds = min(_PAUSE_S * 2**self._pauses, _MAX_PAUSE_S)
            self._pauses += 1
            self._paused_until = self._clock() + seconds
            logger.warning(
                "Remote prefill paused for %.0f s after %d failures: %s",
                seconds,
                self._failures,
                reason,
            )

    def inject(self, request: Any) -> None:
        """Append a finished remote prefill to `request`'s cache; any problem leaves it to prefill here."""
        job = self._jobs.pop(request.request_id, None)
        if job is None:
            return
        if job.state != "done" or job.result is None:
            job.cancel()
            self._failed(
                job.error or "the remote prefill did not finish", pause=job.pause
            )
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
        except Exception as exc:
            # Nothing here may stop the scheduler; a restored cache may be half extended, so it restarts.
            expected = isinstance(exc, (HandoffError, ValueError))
            logger.warning(
                "Request %s: the handoff could not be applied; prefilling here: %r",
                request.request_id,
                exc,
                exc_info=not expected,
            )
            self._failed(
                f"the handoff could not be applied: {type(exc).__name__}: {exc}"
            )
            self._reset(request)
            return
        request.prompt_cache = caches
        request.cached_tokens = job.end
        request.remaining_tokens = request.prompt_token_ids[job.end :]
        self._failures = 0
        self._pauses = 0
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
        """Stop and drop the remote prefill of a request that left the queue."""
        self._served.discard(request_id)
        job = self._jobs.pop(request_id, None)
        if job is not None:
            job.cancel()

    def clear(self) -> None:
        """Stop and drop every remote prefill."""
        for job in self._jobs.values():
            job.cancel()
        self._jobs.clear()
        self._served.clear()

    def status(self) -> dict[str, Any]:
        return {
            "url": self.settings.url,
            "model": self.settings.model,
            "local_model": self.settings.local_model,
            "links": list(self.settings.links),
            "min_tokens": self.settings.min_tokens,
            "paused": self._clock() < self._paused_until,
            "paused_for_s": round(max(0.0, self._paused_until - self._clock()), 1),
            "failures": self._failures,
            "running": sum(job.running for job in self._jobs.values()),
            "last": dict(self.last),
            "last_error": self.last_error,
            "unsupported": self._cache_reason or "",
        }
