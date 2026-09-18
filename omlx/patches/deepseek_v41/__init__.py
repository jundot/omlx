# SPDX-License-Identifier: MIT
"""DeepSeek V4.1 reference port for oMLX's pinned mlx-vlm runtime."""


def apply_patch():
    import sys

    import mlx_lm.models.cache as caches

    from omlx.cache.type_handlers import CacheType
    from omlx.cache.type_registry import CacheTypeRegistry

    from . import model
    from .cache import DeepseekV41Cache, DeepseekV41CacheHandler

    sys.modules.setdefault("mlx_vlm.models.deepseek_v41", model)
    caches.DeepseekV41Cache = DeepseekV41Cache
    CacheTypeRegistry.register(DeepseekV41CacheHandler())
    CacheTypeRegistry._class_name_map["DeepseekV41Cache"] = CacheType.DEEPSEEK_V41
