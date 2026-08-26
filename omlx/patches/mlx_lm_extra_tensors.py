# SPDX-License-Identifier: Apache-2.0
"""Patch mlx-lm's weight-shard glob to pick up MTP sidecar files.

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
augmented with sidecar files declared by that model's config or referenced by
``mtp.*``-prefixed keys in its safetensors index.

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
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_MTP_INDEX_PREFIXES = (
    "mtp.",
    "language_model.mtp.",
    "model.mtp.",
    "model.language_model.mtp.",
)


def sidecar_files_for(model_path: str | Path) -> list[str]:
    """Return absolute paths of safetensors sidecars holding mtp.* weights.

    Merges ``config.json``'s ``mlx_lm_extra_tensors.mtp_file`` with filenames
    backing ``mtp.*``-prefixed keys in ``model.safetensors.index.json``. Drops
    exact files already matched by mlx-lm's own ``model*.safetensors`` glob.

    Returns ``[]`` when neither source declares an MTP sidecar or every
    declared file is already covered by mlx-lm's shard glob.
    """
    p = Path(model_path)
    sidecar_names: set[str] = set()

    config_path = p / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
            extra_tensors = config.get("mlx_lm_extra_tensors")
            if isinstance(extra_tensors, dict):
                mtp_file = extra_tensors.get("mtp_file")
                if isinstance(mtp_file, str) and mtp_file:
                    sidecar_names.add(mtp_file)
        except Exception as e:
            logger.debug("Failed to read %s for sidecar scan: %s", config_path, e)

    index_path = p / "model.safetensors.index.json"
    if index_path.exists():
        try:
            weight_map = json.loads(index_path.read_text()).get("weight_map") or {}
            sidecar_names.update(
                fname
                for key, fname in weight_map.items()
                if isinstance(key, str)
                and key.startswith(_MTP_INDEX_PREFIXES)
                and isinstance(fname, str)
            )
        except Exception as e:
            logger.debug("Failed to read %s for sidecar scan: %s", index_path, e)

    if not sidecar_names:
        return []

    model_root = Path(os.path.abspath(p))
    already_covered = {
        Path(os.path.abspath(f))
        for f in _glob_module.glob(str(model_root / "model*.safetensors"))
    }
    sidecars = {
        path
        for fname in sidecar_names
        if (path := Path(os.path.abspath(model_root / fname))).is_relative_to(
            model_root
        )
    }
    return sorted(str(path) for path in sidecars if path not in already_covered)


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
        try:
            pattern_path = Path(pattern)
        except TypeError:
            return matches
        if pattern_path.name == "model*.safetensors":
            sidecars = sidecar_files_for(pattern_path.parent)
            existing = set(matches)
            matches = matches + [f for f in sidecars if f not in existing]
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
