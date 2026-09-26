# SPDX-License-Identifier: Apache-2.0
"""Registry of patches that substitute a model's chat template.

Most models ship their chat template in the checkpoint, so discovery can read
capabilities straight off disk. A few are served by an oMLX patch that replaces
that template with a Python implementation — for those, the template's own
declarations (such as its reasoning-effort vocabulary) live in oMLX's source
and there is nothing on disk to inspect.

Each such patch declares the model_type it serves alongside its template, and
this module asks them in turn. Discovery therefore stays generic: adding
another patched template means declaring it next to that template, not naming
a model family inside the detector.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


def _deepseek_v4_template() -> Any:
    from .patches.deepseek_v4 import chat_template_v4

    return chat_template_v4


# Loaders are lazy: importing a patch module at registry-import time would pull
# mlx-lm into processes that only need discovery.
_TEMPLATE_MODULE_LOADERS: tuple[Callable[[], Any], ...] = (_deepseek_v4_template,)


def reasoning_effort_vocabulary_for(
    model_type: str,
) -> tuple[list[str], str | None] | None:
    """Return ``(levels, default)`` from the patch serving ``model_type``.

    Returns None when no patched template claims this model, meaning discovery
    should fall back to reading the checkpoint's own chat template.
    """
    if not model_type:
        return None

    for load_template in _TEMPLATE_MODULE_LOADERS:
        try:
            template = load_template()
        except Exception as exc:  # noqa: BLE001 - a patch must never break discovery
            logger.debug("chat-template patch unavailable during discovery: %s", exc)
            continue

        prefix = getattr(template, "CHAT_TEMPLATE_MODEL_TYPE_PREFIX", None)
        if not prefix or not model_type.startswith(prefix):
            continue

        prompts = getattr(template, "REASONING_EFFORT_PROMPTS", None)
        if not prompts:
            return None
        default = getattr(template, "DEFAULT_REASONING_EFFORT", None)
        return list(prompts), default

    return None
