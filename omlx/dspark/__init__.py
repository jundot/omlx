"""Generic dSpark integration for oMLX."""

from .compat import (
    DSparkCompatibility,
    DSparkProbe,
    probe_drafter,
    validate_pair,
)
from .handlers import (
    DeepSpecHandler,
    DrafterHandler,
    DSparkLoadOptions,
    HiggsSidecarHandler,
    SpeculatorsHybridHandler,
    get_handler,
    resolve_handler,
)

__all__ = [
    "DSparkCompatibility",
    "DSparkProbe",
    "probe_drafter",
    "validate_pair",
    "DSparkLoadOptions",
    "DrafterHandler",
    "DeepSpecHandler",
    "SpeculatorsHybridHandler",
    "HiggsSidecarHandler",
    "get_handler",
    "resolve_handler",
]
