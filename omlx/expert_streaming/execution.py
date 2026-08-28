# SPDX-License-Identifier: Apache-2.0
"""Whole-forward execution policies for graph-native expert streaming."""

from __future__ import annotations

import dataclasses
import threading
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass
class ExpertExecutionStats:
    checked_passes: int = 0
    speculative_passes: int = 0
    speculative_hits: int = 0
    speculative_retries: int = 0
    speculative_fallbacks: int = 0

    def as_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)


def _clone_state_value(value: Any) -> Any:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, set):
        return set(value)
    if isinstance(value, tuple):
        return tuple(value)
    return value


class CacheSnapshot:
    """Shallow transactional snapshot of MLX cache objects and metadata.

    MLX caches append by replacing array references or by writing beyond their
    previous logical offset. Restoring both the old references and offsets is
    therefore sufficient for rejecting a whole forward pass. A retry writes
    the same append region again before it can be observed.
    """

    def __init__(self, cache: Any):
        self._states: list[tuple[Any, dict[str, Any]]] = []
        self._containers: list[tuple[Any, Any]] = []
        self._seen: set[int] = set()
        self._capture(cache)

    @staticmethod
    def _is_leaf(value: Any) -> bool:
        return value is None or isinstance(
            value, (mx.array, str, bytes, int, float, bool, np.ndarray, type)
        )

    def _capture(self, value: Any) -> None:
        if self._is_leaf(value):
            return
        if id(value) in self._seen:
            return
        self._seen.add(id(value))
        if isinstance(value, dict):
            self._containers.append((value, dict(value)))
            for child in value.values():
                self._capture(child)
            return
        if isinstance(value, list):
            self._containers.append((value, list(value)))
            for child in value:
                self._capture(child)
            return
        if isinstance(value, set):
            self._containers.append((value, set(value)))
            for child in value:
                self._capture(child)
            return
        if isinstance(value, tuple):
            for child in value:
                self._capture(child)
            return
        # A cache should never own a model, but avoid traversing one if a
        # third-party cache keeps a back-reference.
        if isinstance(value, nn.Module):
            return
        state = getattr(value, "__dict__", None)
        if state is None:
            return
        copied = {key: _clone_state_value(item) for key, item in state.items()}
        self._states.append((value, copied))
        for child in state.values():
            self._capture(child)

    def restore(self) -> None:
        for cache_object, state in reversed(self._states):
            current = getattr(cache_object, "__dict__", None)
            if current is None:
                continue
            current.clear()
            current.update(
                {key: _clone_state_value(value) for key, value in state.items()}
            )
        for container, state in reversed(self._containers):
            if isinstance(container, dict):
                container.clear()
                container.update(state)
            elif isinstance(container, list):
                container[:] = state
            elif isinstance(container, set):
                container.clear()
                container.update(state)


def _result_arrays(value: Any, seen: set[int] | None = None) -> list[mx.array]:
    if seen is None:
        seen = set()
    if isinstance(value, mx.array):
        return [value]
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return []
    identity = id(value)
    if identity in seen:
        return []
    seen.add(identity)
    if isinstance(value, dict):
        return [
            array for item in value.values() for array in _result_arrays(item, seen)
        ]
    if isinstance(value, (list, tuple)):
        return [array for item in value for array in _result_arrays(item, seen)]
    arrays: list[mx.array] = []
    for name in ("logits", "hidden_states", "gdn_states"):
        if hasattr(value, name):
            arrays.extend(_result_arrays(getattr(value, name), seen))
    return arrays


_PATCH_LOCK = threading.RLock()
_TARGETS: dict[int, tuple[weakref.ReferenceType[Any], Any]] = {}
_PATCHED_CLASSES: dict[type, Callable[..., Any]] = {}


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

        def remove_target(_reference, *, identity=target_id):
            with _PATCH_LOCK:
                _TARGETS.pop(identity, None)

        _TARGETS[target_id] = (weakref.ref(target, remove_target), runtime)


def detach_execution_runtime(target: Any) -> None:
    with _PATCH_LOCK:
        _TARGETS.pop(id(target), None)


class SpeculativeExecution:
    """Execute resident hits as one MLX graph and retry exact on a miss."""

    def __init__(self, runtime: Any, *, policy: str, max_retries: int = 1):
        if policy not in {"checked", "speculative"}:
            raise ValueError(
                "Expert streaming execution policy must be checked or speculative"
            )
        self.runtime = runtime
        self.policy = policy
        self.max_retries = max(0, int(max_retries))
        self.stats = ExpertExecutionStats()
        self._has_checked_pass = False
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
    def _cache_from_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        if "cache" in kwargs:
            return kwargs["cache"]
        return args[1] if len(args) > 1 else None

    def _can_speculate(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
        if self.policy != "speculative" or not self._has_checked_pass:
            return False
        if kwargs.get("inputs_embeds") is not None:
            return False
        if kwargs.get("vlm_extra_kwargs"):
            return False
        if not args or not isinstance(args[0], mx.array):
            return False
        return args[0].ndim >= 1 and args[0].shape[-1] == 1

    def _checked(self, call: Callable[[], Any]) -> Any:
        self._set_mode("checked")
        result = call()
        self.stats.checked_passes += 1
        self._has_checked_pass = True
        return result

    def _collect_routes(self):
        routes = []
        for pool in self.runtime.pools:
            for indices, missing in pool.take_speculative_routes():
                routes.append((pool, indices, missing))
        return routes

    @staticmethod
    def _evaluate(result: Any, routes: list[tuple[Any, mx.array, mx.array]]) -> None:
        arrays = _result_arrays(result)
        arrays.extend(indices for _, indices, _ in routes)
        arrays.extend(missing for _, _, missing in routes)
        if arrays:
            mx.eval(*arrays)

    def _inspect_routes(self, routes):
        first_miss = None
        for pool, indices, missing in routes:
            values = tuple(
                int(value) for value in np.asarray(indices.tolist()).reshape(-1)
            )
            pool.record_speculative_route(values)
            if bool(mx.any(missing).item()):
                first_miss = (pool, values)
                # Downstream routing was computed from placeholder expert
                # output and is not a valid analytical cache signal.
                break
        return first_miss

    def _speculative(self, call: Callable[[], Any], cache: Any) -> Any:
        snapshot = CacheSnapshot(cache)
        for attempt in range(self.max_retries + 1):
            self._set_mode("speculative")
            result = call()
            routes = self._collect_routes()
            self._evaluate(result, routes)
            self.stats.speculative_passes += 1
            first_miss = self._inspect_routes(routes)
            if first_miss is None:
                self.stats.speculative_hits += 1
                return result

            snapshot.restore()
            pool, values = first_miss
            try:
                pool.promote(values)
            except RuntimeError:
                break
            if attempt < self.max_retries:
                self.stats.speculative_retries += 1
                continue
            break

        snapshot.restore()
        self.stats.speculative_fallbacks += 1
        return self._checked(call)

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
            try:
                if not self._can_speculate(args, kwargs):
                    return self._checked(call)
                return self._speculative(call, self._cache_from_call(args, kwargs))
            except Exception:
                self._set_mode("checked")
                raise
            finally:
                self._executing = False
