# SPDX-License-Identifier: Apache-2.0
"""Inkling MTP drafter compatibility layer for the pinned mlx-vlm.

The ``inkling_mtp`` drafter package landed in mlx-vlm v0.6.9, newer than
oMLX's mlx-vlm pin (``78b96eb``), whose ``speculative/drafters`` tree
carries only deepseek_v4_mtp / eagle3 / gemma4_assistant /
gemma4_unified_assistant / qwen3_5_mtp. Without it, loading an Inkling
drafter fails twice over: ``resolve_drafter_kind`` misses the model_type
and falls back to ``DEFAULT_DRAFTER_KIND`` (``dflash``), then
``load_model`` cannot import ``mlx_vlm.speculative.drafters.inkling_mtp``
and raises ``Model type inkling_mtp not supported``.

This vendors the v0.6.13 drafter package verbatim and wires the two
discovery surfaces oMLX needs, mirroring
:mod:`omlx.patches.mlx_vlm_inkling_compat`:

- installs the vendored package onto the real
  ``mlx_vlm.speculative.drafters`` namespace (``__path__`` append) so
  ``load_model`` can import it. The vendor path is searched last, so a
  future pin bump that ships the drafter upstream wins automatically.
- adds ``inkling_mtp -> "mtp"`` to ``DRAFTER_KIND_BY_MODEL_TYPE`` via
  ``setdefault``, so auto-detection resolves the MTP round loop instead
  of degrading to dflash — and again defers to upstream if present.

The drafter's own imports (``....models.inkling.language``,
``....models.inkling.inkling``, ``....models.cache``) resolve against the
real pinned mlx-vlm plus the vendored inkling model package, so
:func:`omlx.patches.mlx_vlm_inkling_compat.apply_mlx_vlm_inkling_compat_patch`
MUST be applied first.

Note this is mlx-vlm's separate-checkpoint drafter, which folds only MTP
block 0. oMLX's in-checkpoint path
(:mod:`omlx.patches.mlx_vlm_mtp.inkling_vlm_runtime`) drives the full
eight-depth chain and remains the faster option where the checkpoint
carries ``mtp.*`` weights.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_VENDOR_MLX_VLM = Path(__file__).resolve().parent / "vendor" / "mlx_vlm"

_MODEL_TYPE = "inkling_mtp"
_DRAFT_KIND = "mtp"

_APPLIED = False


def apply_mlx_vlm_inkling_mtp_compat_patch() -> bool:
    """Install the vendored inkling_mtp drafter and its kind registration."""
    global _APPLIED
    if _APPLIED:
        return False

    try:
        _install_vendor_namespace()
        _register_drafter_kind()
        _import_vendor_modules()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Inkling MTP drafter compat patch failed: %s", exc)
        return False

    _APPLIED = True
    logger.info("Inkling MTP drafter mlx-vlm compatibility patch applied")
    return True


def is_applied() -> bool:
    return _APPLIED


def _install_vendor_namespace() -> None:
    import mlx_vlm.speculative.drafters

    _append_package_path(
        mlx_vlm.speculative.drafters,
        _VENDOR_MLX_VLM / "speculative" / "drafters",
    )


def _append_package_path(package: Any, path: Path) -> None:
    package_path = getattr(package, "__path__", None)
    if package_path is None:
        return
    path_str = str(path)
    if path_str not in package_path:
        package_path.append(path_str)


def _register_drafter_kind() -> None:
    from mlx_vlm.speculative import drafters

    table = getattr(drafters, "DRAFTER_KIND_BY_MODEL_TYPE", None)
    if isinstance(table, dict):
        # setdefault: a pin bump that ships the entry upstream wins.
        table.setdefault(_MODEL_TYPE, _DRAFT_KIND)


def _import_vendor_modules() -> None:
    # Fail loudly here rather than at drafter-load time: the package pulls
    # symbols from the vendored inkling model, so a missing inkling compat
    # surfaces as an ImportError during apply() instead of a confusing
    # "Model type inkling_mtp not supported" mid-request.
    importlib.import_module(f"mlx_vlm.speculative.drafters.{_MODEL_TYPE}")
