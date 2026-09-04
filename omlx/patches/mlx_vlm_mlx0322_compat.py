# SPDX-License-Identifier: Apache-2.0
"""Backport the narrow mlx-vlm changes required by MLX 0.32.2.

The pinned mlx-vlm revision predates upstream PRs #1949, #1982, and #2006.
MLX 0.32.2 stopped accepting scalar ``mx.array`` objects where Python integer
dimensions are required and changed ``mx.random.state`` from a mutable list to
a proxy whose member arrays must be updated in place. Affected models otherwise
fail in speculative RNG restoration, ``mx.repeat``, shape construction, or
Metal grid dispatch.

Moving the mlx-vlm pin to any of these merge commits would also import hundreds of
unrelated upstream changes. Instead, this module installs a source loader for
the exact affected modules and applies only the upstream integer conversions.
Each replacement is checked before compilation: an unexpected pinned source
fails loudly instead of leaving a partially applied compatibility patch.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import inspect
import logging
import sys
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Replacement:
    old: str
    new: str
    count: int = 1


_REPEAT_GRID = _Replacement(
    "mx.repeat(seq_len, grid_thw[i, 0])",
    "mx.repeat(seq_len, int(grid_thw[i, 0]))",
)

_MODULE_REPLACEMENTS: dict[str, tuple[_Replacement, ...]] = {
    "mlx_vlm.speculative.common": (
        _Replacement(
            "mx.random.state[i] = value",
            "mx.random.state[i][:] = value",
        ),
    ),
    "mlx_vlm.models.aya_vision.vision": (
        _Replacement(
            "height, width = spatial_shapes[i]",
            "height, width = spatial_shapes[i].tolist()",
        ),
    ),
    "mlx_vlm.models.dots_ocr.vision": (_REPEAT_GRID,),
    "mlx_vlm.models.glm4v.vision": (
        _Replacement(
            "mx.repeat(image_shapes[i, 1], lengths[i])",
            "mx.repeat(image_shapes[i, 1], int(lengths[i]))",
        ),
        _Replacement(
            "mx.repeat(image_shapes[i, 2], lengths[i])",
            "mx.repeat(image_shapes[i, 2], int(lengths[i]))",
        ),
    ),
    "mlx_vlm.models.glm4v_moe.vision": (
        _Replacement(
            "mx.repeat(image_shapes[i, 1], lengths[i])",
            "mx.repeat(image_shapes[i, 1], int(lengths[i]))",
        ),
        _Replacement(
            "mx.repeat(image_shapes[i, 2], lengths[i])",
            "mx.repeat(image_shapes[i, 2], int(lengths[i]))",
        ),
    ),
    "mlx_vlm.models.lfm2_vl.lfm2_vl": (
        _Replacement(
            "feature_org_h, feature_org_w = spatial_shapes[img_idx]",
            "feature_org_h, feature_org_w = (\n"
            "                    int(dim) for dim in spatial_shapes[img_idx]\n"
            "                )",
        ),
    ),
    "mlx_vlm.models.paddleocr_vl.vision": (_REPEAT_GRID,),
    "mlx_vlm.models.qwen2_5_vl.vision": (_REPEAT_GRID,),
    "mlx_vlm.models.qwen2_vl.vision": (
        _Replacement(
            "h, w = int(h), int(w)  # Ensure h and w are integers",
            "t, h, w = int(t), int(h), int(w)",
        ),
        _REPEAT_GRID,
    ),
    "mlx_vlm.models.qwen3_omni_moe.vision": (_REPEAT_GRID,),
    "mlx_vlm.models.qwen3_vl.vision": (_REPEAT_GRID,),
    "mlx_vlm.models.qwen3_vl_moe.vision": (_REPEAT_GRID,),
    "mlx_vlm.models.kernels": (
        _Replacement(
            "import mlx.core as mx",
            "import math\n\nimport mlx.core as mx",
        ),
        _Replacement(
            "grid=(mx.prod(mx.array(out_shape)), 1, 1)",
            "grid=(math.prod(out_shape), 1, 1)",
        ),
    ),
}

_INSTALLED = False


def _patch_source(fullname: str, source: str) -> str:
    replacements = _MODULE_REPLACEMENTS[fullname]
    patched = source
    for replacement in replacements:
        # Prefer the already-fixed form first. This matters when ``old`` is a
        # substring of ``new`` (the kernels.py import insertion), and keeps a
        # future upstream-fixed pin from receiving the same edit twice.
        if patched.count(replacement.new) == replacement.count:
            continue
        old_count = patched.count(replacement.old)
        if old_count == replacement.count:
            patched = patched.replace(
                replacement.old, replacement.new, replacement.count
            )
            continue

        # Any other shape is an explicit pin/patch mismatch.
        raise ImportError(
            f"MLX 0.32.2 mlx-vlm compatibility patch mismatch for {fullname}: "
            f"expected {replacement.count} occurrence(s), found {old_count}"
        )
    return patched


class _CompatLoader(importlib.abc.Loader):
    def __init__(self, fullname: str, delegate: importlib.abc.Loader):
        self.fullname = fullname
        self.delegate = delegate

    def create_module(self, spec):
        create_module = getattr(self.delegate, "create_module", None)
        return create_module(spec) if create_module is not None else None

    def exec_module(self, module) -> None:
        get_source = getattr(self.delegate, "get_source", None)
        if get_source is None:
            raise ImportError(f"source unavailable for {self.fullname}")
        source = get_source(self.fullname)
        if source is None:
            raise ImportError(f"source unavailable for {self.fullname}")
        filename = getattr(module.__spec__, "origin", None) or self.fullname
        code = compile(_patch_source(self.fullname, source), filename, "exec")
        exec(code, module.__dict__)


class _CompatFinder(importlib.abc.MetaPathFinder):
    _omlx_mlx_vlm_mlx0322_compat = True

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _MODULE_REPLACEMENTS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _CompatLoader(fullname, spec.loader)
        return spec


def _reload_and_rebind(fullname: str, module) -> None:
    """Patch an early import and update mlx-vlm's copied references."""
    old_objects = {
        name: value
        for name, value in vars(module).items()
        if (inspect.isclass(value) or inspect.isfunction(value))
        and getattr(value, "__module__", None) == fullname
    }
    old_generation_stream = (
        getattr(module, "generation_stream", None)
        if fullname == "mlx_vlm.speculative.common"
        else None
    )
    importlib.reload(module)
    replacements = {
        old: getattr(module, name)
        for name, old in old_objects.items()
        if hasattr(module, name) and getattr(module, name) is not old
    }
    new_generation_stream = (
        getattr(module, "generation_stream", None)
        if old_generation_stream is not None
        else None
    )
    if not replacements and new_generation_stream is old_generation_stream:
        return
    for loaded_name, loaded in list(sys.modules.items()):
        if loaded is None or not loaded_name.startswith("mlx_vlm."):
            continue
        for name, value in list(vars(loaded).items()):
            if value is old_generation_stream:
                setattr(loaded, name, new_generation_stream)
                continue
            if not (inspect.isclass(value) or inspect.isfunction(value)):
                continue
            replacement = replacements.get(value)
            if replacement is not None:
                setattr(loaded, name, replacement)


def apply_mlx_vlm_mlx0322_compat_patch() -> bool:
    """Install the MLX 0.32.2 loader before mlx-vlm model imports."""
    global _INSTALLED
    if _INSTALLED:
        return False

    finder = _CompatFinder()
    sys.meta_path.insert(0, finder)

    # Normal oMLX loading installs this hook before importing a model module.
    # Reload any target imported by an embedding application earlier so the
    # process cannot retain the incompatible code silently.
    try:
        for fullname in _MODULE_REPLACEMENTS:
            module = sys.modules.get(fullname)
            if module is not None:
                _reload_and_rebind(fullname, module)
    except Exception:
        sys.meta_path.remove(finder)
        raise

    _INSTALLED = True
    logger.info("mlx-vlm MLX 0.32.2 compatibility patch installed")
    return True


__all__ = ["apply_mlx_vlm_mlx0322_compat_patch"]
