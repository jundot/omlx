# SPDX-License-Identifier: MIT
"""DeepSeek V4.1 reference port for oMLX's pinned mlx-vlm runtime."""

import logging

logger = logging.getLogger(__name__)


def apply_patch():
    import sys

    import mlx_lm.models.cache as caches

    from omlx.cache.type_handlers import CacheType
    from omlx.cache.type_registry import CacheTypeRegistry

    from . import model
    from .cache import DeepseekV41Cache, DeepseekV41CacheHandler

    alias = "mlx_vlm.models.deepseek_v41"
    existing = sys.modules.get(alias)
    vendor_path = str(model.__file__)
    if existing is not None and existing is not model:
        # A foreign module under this alias (e.g. an upstream mlx-vlm release
        # shipping deepseek_v41, imported before the patch ran) silently
        # disabled the whole vendor tree once via import-once — GLM-5.x's
        # three rounds of dead-code fixes. The vendor module is the only
        # implementation this runtime knows, so it wins, loudly.
        logger.warning(
            "deepseek_v41 alias %s (%s) replaced by vendor module %s; "
            "check for an upstream mlx-vlm deepseek_v41 fork",
            alias,
            getattr(existing, "__file__", "?"),
            vendor_path,
        )
    sys.modules[alias] = model
    if sys.modules.get(alias) is not model:
        logger.error(
            "deepseek_v41 vendor patch failed to install %s (got %s)",
            vendor_path,
            sys.modules.get(alias),
        )
    caches.DeepseekV41Cache = DeepseekV41Cache
    CacheTypeRegistry.register(DeepseekV41CacheHandler())
    CacheTypeRegistry._class_name_map["DeepseekV41Cache"] = CacheType.DEEPSEEK_V41
