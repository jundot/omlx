# SPDX-License-Identifier: Apache-2.0
"""
omlx: LLM inference server, optimized for your Mac.

The package root exposes the public API lazily so lightweight entrypoints
like ``python -m omlx.cli --help`` do not initialize MLX/Metal just by
importing the ``omlx`` package.
"""

from importlib import import_module

from omlx._version import __version__

_LAZY_EXPORTS = {
    "Request": ("omlx.request", "Request"),
    "RequestOutput": ("omlx.request", "RequestOutput"),
    "RequestStatus": ("omlx.request", "RequestStatus"),
    "SamplingParams": ("omlx.request", "SamplingParams"),
    "Scheduler": ("omlx.scheduler", "Scheduler"),
    "SchedulerConfig": ("omlx.scheduler_config", "SchedulerConfig"),
    "SchedulerOutput": ("omlx.scheduler", "SchedulerOutput"),
    "EngineCore": ("omlx.engine_core", "EngineCore"),
    "AsyncEngineCore": ("omlx.engine_core", "AsyncEngineCore"),
    "EngineConfig": ("omlx.engine_core", "EngineConfig"),
    "get_registry": ("omlx.model_registry", "get_registry"),
    "ModelOwnershipError": ("omlx.model_registry", "ModelOwnershipError"),
    "BlockAwarePrefixCache": ("omlx.cache.prefix_cache", "BlockAwarePrefixCache"),
    "PagedCacheManager": ("omlx.cache.paged_cache", "PagedCacheManager"),
    "CacheBlock": ("omlx.cache.paged_cache", "CacheBlock"),
    "BlockTable": ("omlx.cache.paged_cache", "BlockTable"),
    "PrefixCacheStats": ("omlx.cache.stats", "PrefixCacheStats"),
    "PagedCacheStats": ("omlx.cache.stats", "PagedCacheStats"),
}

__all__ = [
    "Request",
    "RequestOutput",
    "RequestStatus",
    "SamplingParams",
    "Scheduler",
    "SchedulerConfig",
    "SchedulerOutput",
    "EngineCore",
    "AsyncEngineCore",
    "EngineConfig",
    "get_registry",
    "ModelOwnershipError",
    "BlockAwarePrefixCache",
    "PagedCacheManager",
    "CacheBlock",
    "BlockTable",
    "PrefixCacheStats",
    "PagedCacheStats",
    "CacheStats",
    "__version__",
]


def __getattr__(name):
    """Lazily import public API objects on first access."""
    if name == "CacheStats":
        value = __getattr__("PagedCacheStats")
        globals()[name] = value
        return value

    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'omlx' has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
