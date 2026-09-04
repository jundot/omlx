# SPDX-License-Identifier: Apache-2.0
"""Let mlx-lm load MiniMax-M3, so the cluster can serve it.

oMLX already runs MiniMax-M3 on one Mac, through a vendored mlx-vlm. The
cluster cannot: every rank is an ``mlx_lm.server``, and pinned mlx-lm has no
``minimax_m3_vl`` — ``_get_classes`` raises *"Model type minimax_m3_vl not
supported."* before a single weight is read. A 225 GiB model that fits two Macs
comfortably was therefore unservable across them while fitting one Mac only at
~1k tokens of context.

This registers ``mlx_lm.models.minimax_m3_vl`` in ``sys.modules`` so mlx-lm's
own ``importlib.import_module(f"mlx_lm.models.{model_type}")`` finds it —
exactly what ``omlx.patches.deepseek_v4`` does for the same reason. The model
itself is oMLX's existing vendored implementation, not a second copy: one
definition serves both the single-node and clustered paths, so they cannot
drift.

Gated on ``model_type == "minimax_m3_vl"``; other models pay nothing.

Removable in one delete once mlx-lm ships MiniMax-M3 upstream, along with the
dispatch in ``omlx/utils/model_loading.py``.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_QUALNAME = "mlx_lm.models.minimax_m3_vl"
# Converted checkpoints label the text-only export differently; both resolve to
# the same module, matching how deepseek_v4 aliases its MTP variant.
_ALIASES = ("mlx_lm.models.minimax_m3",)


def _register_module(qualname: str, file_name: str) -> None:
    """Load a local file as if it were ``qualname``. Idempotent.

    ``__package__`` is set to ``mlx_lm.models`` so any relative import inside
    the loaded file resolves through the real mlx-lm package, not through omlx.
    """

    if qualname in sys.modules:
        return

    file_path = Path(__file__).parent / file_name
    spec = importlib.util.spec_from_file_location(qualname, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create spec for {qualname} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "mlx_lm.models"
    # Registered before execution so relative imports inside resolve, but a
    # failure must not leave the husk behind: mlx-lm would then find a module
    # with no ``Model`` and fail far away with a confusing AttributeError
    # instead of the real cause. Seen on a peer lacking mlx_vlm.
    sys.modules[qualname] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(qualname, None)
        raise
    logger.info("Registered %s from %s", qualname, file_path.name)


def apply_minimax_m3_mlx_lm_patch() -> bool:
    """Make ``minimax_m3_vl`` loadable by mlx-lm. Safe to call repeatedly."""

    try:
        _register_module(_QUALNAME, "minimax_m3_vl_model.py")
    except Exception as exc:
        logger.warning("Could not register %s: %s", _QUALNAME, exc)
        return False

    base = sys.modules.get(_QUALNAME)
    for alias in _ALIASES:
        if base is not None:
            sys.modules.setdefault(alias, base)
    return True


def is_minimax_m3(config: dict) -> bool:
    """Does this config describe a model this patch is for?"""

    model_type = str(config.get("model_type", ""))
    return model_type.startswith("minimax_m3")
