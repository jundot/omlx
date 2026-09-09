# SPDX-License-Identifier: Apache-2.0
"""The mlx-vlm half of gemma4 assistant MTP, kept out of ``mlx_lm_mtp``.

Google ships the gemma4 draft head as a separate ``gemma4_assistant``
model, and the only implementation of it lives in mlx-vlm. Serving a
merged checkpoint through mlx-lm therefore needs mlx-vlm installed —
which is true of gemma4 and of nothing else in ``mlx_lm_mtp``.

That distinction is load-bearing, not tidiness.
``cluster.autoconfigure.required_imports`` tells a rank what to install by
scanning each patch package for third-party imports, and it scans the whole
package directory. An ``import mlx_vlm`` anywhere under ``mlx_lm_mtp`` would
therefore ask every rank serving any model — a plain llama included — to
bring mlx-vlm, and over-asking blocks a cluster that would have worked.
Keeping these two functions in their own module, imported from the
dispatcher under a gemma4 guard, asks exactly the ranks that need it.

Deferring the import instead would hide the requirement from that scan and
put the failure back in the middle of a load, which is the bug the scan
exists to prevent.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_draft_model(assistant_config: dict) -> Any:
    """Size the assistant head from the config oQ merged into the checkpoint."""
    from mlx_vlm.speculative.drafters.gemma4_assistant import (
        Gemma4AssistantDraftModel,
        ModelConfig,
    )

    return Gemma4AssistantDraftModel(ModelConfig.from_dict(assistant_config))


def slice_shared_kv_after_reject(shared_kv: dict, rejected: int) -> dict:
    """Drop the rejected tail from a captured K/V stash."""
    from mlx_vlm.speculative.mtp import (  # noqa: SLF001
        _slice_shared_kv_after_reject,
    )

    return _slice_shared_kv_after_reject(shared_kv, rejected)


def warn_if_unavailable(model_name: str) -> bool:
    """Say up front when a merged head cannot be attached, and why.

    Without this the load reaches ``Model.__init__``, fails to import the
    drafter, and the operator sees a ModuleNotFoundError from inside model
    construction rather than a sentence naming the missing package.
    """
    try:
        import mlx_vlm.speculative.drafters.gemma4_assistant  # noqa: F401
    except ImportError:
        logger.warning(
            "Gemma 4 assistant MTP for %s needs mlx-vlm, which supplies the "
            "draft head; speculative decoding will stay inactive",
            model_name,
        )
        return False
    return True
