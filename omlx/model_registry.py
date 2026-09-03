# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-mlx (https://github.com/vllm-project/vllm-mlx).
"""
Model Registry for oMLX.

Tracks which models are in use by which engines to prevent
BatchKVCache conflicts when multiple engines share a model.

The problem: mlx-lm's BatchGenerator maintains internal KV cache state
tied to the model. When multiple EngineCore instances use the same model,
the cache objects become incompatible, causing NoneType errors.

Solution: Track model ownership and ensure only one engine's BatchGenerator
is active for each model at a time.
"""

import logging
import threading
import weakref
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class ModelOwnershipError(Exception):
    """Raised when a model is already owned by another engine."""
    pass


class ModelRegistry:
    """
    Global registry tracking model ownership.

    Uses weak references to automatically clean up when engines
    are garbage collected.
    """

    _instance: Optional["ModelRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ModelRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        """Initialize registry state."""
        # model_id -> (weak_ref_to_engine, engine_id)
        self._owners: Dict[int, Tuple[weakref.ref, str]] = {}
        self._registry_lock = threading.Lock()

    def acquire(
        self,
        model: Any,
        engine: Any,
        engine_id: str,
        force: bool = False
    ) -> bool:
        """
        Attempt to acquire ownership of a model.

        Args:
            model: The MLX model
            engine: The EngineCore instance
            engine_id: Unique identifier for the engine
            force: Legacy compatibility flag. It may replace a stale weakref,
                but a different live owner must still close explicitly.

        Returns:
            True if ownership acquired

        Raises:
            ModelOwnershipError: If another live engine owns the model.
        """
        model_id = id(model)

        with self._registry_lock:
            if model_id in self._owners:
                weak_ref, owner_id = self._owners[model_id]
                owner = weak_ref()

                if owner is not None and owner_id != engine_id:
                    # Resetting only the prior scheduler is not an ownership
                    # transfer: that core still holds the shared model and its
                    # later close() can release resources under the replacement
                    # engine.  Fail closed even for the legacy force flag; the
                    # owner must complete orderly close/unload first.
                    raise ModelOwnershipError(
                        f"Model is already owned by live engine {owner_id}. "
                        "Close or unload the previous engine before reuse."
                    )

            # Register new owner
            self._owners[model_id] = (weakref.ref(engine), engine_id)
            logger.debug(f"Engine {engine_id} acquired model {model_id}")
            return True

    def release(self, model: Any, engine_id: str) -> bool:
        """
        Release ownership of a model.

        Args:
            model: The MLX model
            engine_id: The engine releasing ownership

        Returns:
            True if ownership was released
        """
        model_id = id(model)

        with self._registry_lock:
            if model_id in self._owners:
                _, owner_id = self._owners[model_id]
                if owner_id == engine_id:
                    del self._owners[model_id]
                    logger.debug(f"Engine {engine_id} released model {model_id}")
                    return True
        return False

    def release_by_id(self, model_id: int, engine_id: str) -> bool:
        """Release ownership using only the registry's non-owning key.

        Engine teardown must preserve its lease through final Metal reclaim,
        but retaining the model object until that point also retains every
        weight.  This owner-checked variant lets teardown keep ordering with
        only ``id(model)`` and no strong reference to the graph.
        """

        model_id = int(model_id)
        with self._registry_lock:
            if model_id in self._owners:
                _, owner_id = self._owners[model_id]
                if owner_id == engine_id:
                    del self._owners[model_id]
                    logger.debug(
                        "Engine %s released model %s by id",
                        engine_id,
                        model_id,
                    )
                    return True
        return False

    def is_owned(self, model: Any) -> Tuple[bool, Optional[str]]:
        """
        Check if a model is owned.

        Returns:
            Tuple of (is_owned, owner_engine_id)
        """
        model_id = id(model)

        with self._registry_lock:
            if model_id in self._owners:
                weak_ref, owner_id = self._owners[model_id]
                if weak_ref() is not None:
                    return (True, owner_id)
                else:
                    # Owner was garbage collected, clean up
                    del self._owners[model_id]
        return (False, None)

    def cleanup(self) -> int:
        """
        Clean up stale entries (owners that were garbage collected).

        Returns:
            Number of entries cleaned up
        """
        cleaned = 0
        with self._registry_lock:
            stale = [
                model_id for model_id, (weak_ref, _) in self._owners.items()
                if weak_ref() is None
            ]
            for model_id in stale:
                del self._owners[model_id]
                cleaned += 1
        if cleaned:
            logger.debug(f"Cleaned up {cleaned} stale registry entries")
        return cleaned

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        with self._registry_lock:
            active = sum(
                1 for _, (ref, _) in self._owners.items()
                if ref() is not None
            )
            return {
                "total_entries": len(self._owners),
                "active_owners": active,
            }


# Global singleton
_registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    """Get the global model registry."""
    return _registry
