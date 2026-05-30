# SPDX-License-Identifier: Apache-2.0
"""Merge Hugging Face-style ``mtp.safetensors`` sidecars during ``mlx_lm.load``.

Official Qwen3.5/3.6 checkpoints (and some third-party quant exports such as
OptiQ) store MTP head weights in ``mtp.safetensors`` while the main index only
references ``model-*.safetensors``. Stock mlx-lm loads ``model*.safetensors``
only, so the MTP module stays at random init unless the sidecar is merged.

This patch wraps ``mlx_lm.utils.load_model`` and, when ``mtp.safetensors``
is needed (no inlined ``mtp.*`` in the main shards/index), temporarily extends
``glob.glob`` so the sidecar is loaded with the main shards *before*
``sanitize`` / ``nn.quantize`` / ``load_weights``. Inlined oQ exports are
left to stock mlx-lm loading even if a leftover sidecar file is present.
"""

from __future__ import annotations

import glob as glob_module
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from ...utils.model_loading import mtp_sidecar_path_for_load

logger = logging.getLogger(__name__)

_PATCHED = False


@contextmanager
def _glob_includes_mtp_sidecar(sidecar: Path) -> Iterator[None]:
    """Append ``mtp.safetensors`` to mlx-lm's ``model*.safetensors`` glob."""
    real_glob = glob_module.glob
    sidecar_str = str(sidecar)

    def patched_glob(pattern: str, *args, **kwargs):
        files = real_glob(pattern, *args, **kwargs)
        if sidecar_str in files:
            return files
        normalized = pattern.replace("\\", "/")
        if normalized.endswith("model*.safetensors"):
            return sorted(files) + [sidecar_str]
        return files

    glob_module.glob = patched_glob
    try:
        yield
    finally:
        glob_module.glob = real_glob


def apply() -> bool:
    """Wrap ``mlx_lm.utils.load_model`` to load ``mtp.safetensors`` sidecars."""
    global _PATCHED
    if _PATCHED:
        return True

    try:
        import mlx_lm.utils as mlx_utils
    except ImportError:
        logger.debug("mlx_lm.utils not importable; skipping mtp sidecar load patch")
        return False

    if getattr(mlx_utils, "_omlx_mtp_load_patched", False):
        _PATCHED = True
        return True

    original: Callable[..., Any] = mlx_utils.load_model

    def patched_load_model(
        model_path: Path,
        lazy: bool = False,
        strict: bool = True,
        model_config: dict | None = None,
        get_model_classes: Callable | None = None,
    ):
        kwargs: dict[str, Any] = {
            "lazy": lazy,
            "strict": strict,
            "model_config": model_config,
        }
        if get_model_classes is not None:
            kwargs["get_model_classes"] = get_model_classes

        model_path = Path(model_path)
        sidecar = mtp_sidecar_path_for_load(model_path)
        if sidecar is not None:
            with _glob_includes_mtp_sidecar(sidecar):
                model, config = original(model_path, **kwargs)
            logger.info("Loaded MTP weights from %s", sidecar.name)
            return model, config

        return original(model_path, **kwargs)

    patched_load_model._omlx_mtp_load_patched = True  # type: ignore[attr-defined]
    mlx_utils.load_model = patched_load_model
    mlx_utils._omlx_mtp_load_patched = True

    _propagate_load_model_binding(mlx_utils, patched_load_model, original)

    _PATCHED = True
    logger.debug("mlx_lm.utils.load_model wrapped for mtp.safetensors sidecars")
    return True


def _propagate_load_model_binding(
    mlx_utils: Any, patched: Callable[..., Any], original: Callable[..., Any]
) -> None:
    """Point stale ``mlx_lm.*`` imports at the new ``load_model``."""
    import sys

    for mod_name, mod in list(sys.modules.items()):
        if mod is None or not mod_name.startswith("mlx_lm"):
            continue
        if mod_name == "mlx_lm.utils":
            continue
        existing = getattr(mod, "load_model", None)
        if existing is not None and existing in (original, patched):
            try:
                mod.load_model = patched
            except Exception:
                pass
