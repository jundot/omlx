# SPDX-License-Identifier: Apache-2.0
"""Patch mlx-lm's weight-shard glob to pick up indexed MTP sidecar files.

``mlx_lm.utils.load_model`` discovers weight shards with
``glob.glob(str(model_path / "model*.safetensors"))``. MTPLX-forge exports
(e.g. ``wang-yang/Ornith-1.0-35B-MTPLX``) ship the MTP head's weights in a
separate ``mtp.safetensors`` sidecar that ``model.safetensors.index.json``
references but that filename doesn't match the glob. The sidecar is
silently skipped, so the ``mtp.*`` keys it holds never reach
``TextModel.sanitize`` — Native MTP then raises "the converted weights are
missing the mtp.* tensors" even though the checkpoint has them (issue
#2062).

Fix: replace the ``glob`` name ``mlx_lm.utils`` resolves at call time with a
thin proxy. Every pattern except the ``model*.safetensors`` shard-discovery
call passes straight through to the real ``glob.glob``; that one call gets
augmented with sidecar files referenced by ``mtp.*``-prefixed keys in the
model's safetensors index.

This is distinct from issue #1944 / PR #1962, which fix the Native MTP
*compatibility check* (``_checkpoint_has_mtp_weights`` /
``_model_has_mtp_weight_tensors``) for OptiQ checkpoints whose sidecar isn't
referenced by the index at all. Neither of those touches actual weight
loading, so they don't help here.
"""

from __future__ import annotations

import glob as _glob_module
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_EXTRA_TENSOR_FILES: list[str] = []

_MTP_INDEX_PREFIXES = (
    "mtp.",
    "language_model.mtp.",
    "model.mtp.",
    "model.language_model.mtp.",
)


def set_extra_tensor_files(files: list[str]) -> None:
    """Set the process-wide sidecar file list for the next ``mlx_lm.load()``.

    Same construction-time-flag pattern as ``mlx_lm_mtp.set_mtp_active``:
    the caller resets this to ``[]`` before every load so a model without a
    sidecar isn't polluted by a prior load's file list.
    """
    global _EXTRA_TENSOR_FILES
    _EXTRA_TENSOR_FILES = list(files)


def get_extra_tensor_files() -> list[str]:
    return list(_EXTRA_TENSOR_FILES)


def sidecar_files_for(model_path: str | Path) -> list[str]:
    """Return absolute paths of safetensors sidecars holding mtp.* weights.

    Reads ``model.safetensors.index.json``'s ``weight_map``, collects the
    distinct filenames backing ``mtp.*``-prefixed keys, and drops any that
    already match mlx-lm's own ``model*.safetensors`` glob (no need to add
    those — mlx-lm finds them on its own).

    Returns ``[]`` when there's no index, the index is unreadable, or every
    mtp.* key already lives in a ``model*.safetensors`` shard.
    """
    p = Path(model_path)
    index_path = p / "model.safetensors.index.json"
    if not index_path.exists():
        return []
    try:
        weight_map = json.loads(index_path.read_text()).get("weight_map") or {}
    except Exception as e:
        logger.debug("Failed to read %s for sidecar scan: %s", index_path, e)
        return []

    sidecar_names = {
        fname
        for key, fname in weight_map.items()
        if isinstance(key, str)
        and key.startswith(_MTP_INDEX_PREFIXES)
        and isinstance(fname, str)
    }
    if not sidecar_names:
        return []

    already_covered = {
        Path(f).name for f in _glob_module.glob(str(p / "model*.safetensors"))
    }
    return sorted(
        str(p / fname) for fname in sidecar_names if fname not in already_covered
    )


class _GlobProxy:
    """Delegates to the real ``glob`` module, augmenting one call site.

    Only the ``model*.safetensors`` shard-discovery pattern is
    special-cased; every other call (tokenizer/config discovery, checkpoint
    saving, sharded-pipeline file selection, etc.) passes through
    untouched.
    """

    _omlx_extra_tensors_proxy = True

    def glob(self, pattern, *args, **kwargs):
        matches = _glob_module.glob(pattern, *args, **kwargs)
        if pattern.endswith("model*.safetensors") and _EXTRA_TENSOR_FILES:
            existing = set(matches)
            matches = matches + [f for f in _EXTRA_TENSOR_FILES if f not in existing]
        return matches

    def __getattr__(self, name):
        return getattr(_glob_module, name)


def apply() -> bool:
    """Install the glob proxy on ``mlx_lm.utils``. Idempotent."""
    try:
        from mlx_lm import utils as mlx_lm_utils
    except ImportError:
        logger.debug("mlx_lm.utils not importable; skipping extra-tensors patch")
        return False

    if not getattr(mlx_lm_utils.glob, "_omlx_extra_tensors_proxy", False):
        mlx_lm_utils.glob = _GlobProxy()
        logger.info("mlx-lm MTP sidecar glob patch applied")
    return True
