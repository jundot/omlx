# SPDX-License-Identifier: Apache-2.0
"""Model-call lifecycle for deterministic SSD expert streaming."""

from __future__ import annotations

import dataclasses
import threading
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import mlx.core as mx


@dataclass
class ExpertExecutionStats:
    checked_passes: int = 0
    resident_bypass_passes: int = 0
    decode_passes: int = 0

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


_PATCH_LOCK = threading.RLock()
_TARGETS: dict[int, tuple[weakref.ReferenceType[Any], Any]] = {}
_PATCHED_CLASSES: dict[type, Callable[..., Any]] = {}


def _restore_class_if_unused(target_class: type) -> None:
    for reference, _runtime in _TARGETS.values():
        target = reference()
        if target is not None and type(target) is target_class:
            return
    original = _PATCHED_CLASSES.pop(target_class, None)
    if original is not None:
        target_class.__call__ = original


def attach_execution_runtime(target: Any, runtime: Any) -> None:
    """Route calls to one model instance through its streaming runtime."""

    target_id = id(target)
    target_class = type(target)
    with _PATCH_LOCK:
        if target_class not in _PATCHED_CLASSES:
            original = target_class.__call__

            def streaming_call(self, *args, **kwargs):
                registered = _TARGETS.get(id(self))
                if registered is None or registered[0]() is not self:
                    return original(self, *args, **kwargs)
                return registered[1].execute_call(
                    self,
                    lambda: original(self, *args, **kwargs),
                    args,
                    kwargs,
                )

            _PATCHED_CLASSES[target_class] = original
            target_class.__call__ = streaming_call

        def remove_target(
            _reference, *, identity=target_id, attached_class=target_class
        ):
            with _PATCH_LOCK:
                _TARGETS.pop(identity, None)
                _restore_class_if_unused(attached_class)

        _TARGETS[target_id] = (weakref.ref(target, remove_target), runtime)


def detach_execution_runtime(target: Any) -> None:
    with _PATCH_LOCK:
        _TARGETS.pop(id(target), None)
        _restore_class_if_unused(type(target))


class ExpertStreamingExecution:
    """Apply decode-only streaming state without changing model semantics."""

    def __init__(self, runtime: Any):
        self.runtime = runtime
        self.stats = ExpertExecutionStats()
        self._executing = False
        self._lock = threading.RLock()
        self._targets: list[weakref.ReferenceType[Any]] = []

    def attach(self, target: Any) -> None:
        attach_execution_runtime(target, self)
        self._targets.append(weakref.ref(target))

    def close(self) -> None:
        for reference in self._targets:
            target = reference()
            if target is not None:
                detach_execution_runtime(target)
        self._targets.clear()

    def _set_mode(self, mode: str) -> None:
        for pool in self.runtime.pools:
            pool.set_execution_mode(mode)

    @staticmethod
    def _is_decode(args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
        if not args or not isinstance(args[0], mx.array) or args[0].ndim < 1:
            return False
        if kwargs.get("inputs_embeds") is not None:
            return False
        return bool(
            args[0].shape[-1] == 1
            or kwargs.get("n_confirmed", 0)
            or kwargs.get("gdn_sink") is not None
        )

    def execute_call(
        self,
        target: Any,
        call: Callable[[], Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        del target
        with self._lock:
            if self._executing:
                return call()
            self._executing = True
            is_decode = self._is_decode(args, kwargs)
            if is_decode:
                self.stats.decode_passes += 1
            if self.runtime.native_demand_decode_only:
                for pool in self.runtime.pools:
                    pool.set_native_demand_active(is_decode)
            self.runtime.set_decode_cache_active(is_decode)
            try:
                if self.runtime.pools and all(
                    pool.all_experts_resident for pool in self.runtime.pools
                ):
                    self._set_mode("resident")
                    self.stats.resident_bypass_passes += 1
                else:
                    self._set_mode("checked")
                    self.stats.checked_passes += 1
                return call()
            finally:
                self._set_mode("checked")
                self._executing = False
                if not is_decode:
                    # A streamed prefill chunk allocates short-lived gather
                    # and scratch buffers in many distinct shapes, and the
                    # MLX pool retains all of them: a 2048-token chunk on
                    # Qwen3.8 Flash Next grew the pool by ~13 GB while live
                    # Metal memory rose 3 GB. The scheduler's prefill guard
                    # prices that footprint delta as the next chunk's
                    # transient and refuses long prompts that would fit.
                    # Return the pool at the prefill boundary; decode passes
                    # keep their warm buffers. Metal defers the release of
                    # buffers still referenced by in-flight command buffers,
                    # so drain the stream first or the footprint the guard
                    # samples right after this call still carries the pool.
                    mx.synchronize()
                    mx.clear_cache()
