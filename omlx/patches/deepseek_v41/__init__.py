# SPDX-License-Identifier: MIT
"""DeepSeek V4.1 reference port for oMLX's pinned mlx-vlm runtime."""


def apply_patch():
    import os
    import sys

    # Opt-in ~27 tok/s stack (SwitchGLU + wsdpa + deferred HC).
    if os.environ.get("OMLX_DSV41_FAST", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        from .fast_path import apply_fast_path

        return apply_fast_path()

    import mlx_lm.models.cache as caches

    from omlx.cache.type_handlers import CacheType
    from omlx.cache.type_registry import CacheTypeRegistry

    from . import model
    from .cache import DeepseekV41Cache, DeepseekV41CacheHandler

    sys.modules.setdefault("mlx_vlm.models.deepseek_v41", model)
    caches.DeepseekV41Cache = DeepseekV41Cache
    CacheTypeRegistry.register(DeepseekV41CacheHandler())
    CacheTypeRegistry._class_name_map["DeepseekV41Cache"] = CacheType.DEEPSEEK_V41
