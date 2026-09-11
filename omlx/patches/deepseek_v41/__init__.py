# SPDX-License-Identifier: MIT
"""DeepSeek V4.1 reference port for oMLX's pinned mlx-vlm runtime.

The SwitchGLU / wsdpa / deferred-HC stack is the default load path for
``deepseek_v41``. Engram backend and hot-row cache budget remain optional
via ``OMLX_DSV41_ENGRAM`` / ``OMLX_DSV41_ENGRAM_CACHE_GB``.
"""


def apply_patch():
    from .fast_path import apply_fast_path

    return apply_fast_path()
