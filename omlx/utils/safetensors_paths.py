# SPDX-License-Identifier: Apache-2.0
"""Path identity for safetensors shards inside symlinked checkpoints.

Several loaders decide whether a checkpoint is ``format=mlx`` from the
safetensors header metadata, and oMLX sometimes hides that marker for a single
load so a model's ``Model.sanitize()`` runs. That override is scoped to the
shards belonging to the model being loaded, so it has to compare the filename
handed to ``safe_open`` against the model directory.

``Path.resolve()`` is the wrong tool for that comparison. Checkpoint layouts
that keep shards as symlinks to a content-addressed store -- HF Hub cache
snapshots, download mirrors, hand-made links -- resolve the shard name out of
the model directory entirely, the comparison silently returns False, and the
marker stays visible. The sanitize that would have dropped the model's
self-managed tensors then never runs, and the load fails later on orphan keys
reported by ``load_weights``.

Match lexically first (symlinks preserved) and fall back to the fully resolved
form, so both a symlinked shard and a symlinked model directory match.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

__all__ = ["model_shard_matcher"]


def _as_filename_str(filename: object) -> str | None:
    """Return *filename* as a filesystem string, or ``None`` if not a path."""
    try:
        return os.fsdecode(os.fspath(filename))
    except TypeError:
        return None


def _candidate_parents(name: str) -> set[Path]:
    """Every spelling of the directory a shard name could belong to."""
    parents = {Path(name).parent, Path(os.path.abspath(name)).parent}
    try:
        parents.add(Path(name).resolve().parent)
    except (OSError, RuntimeError):
        pass
    return parents


def model_shard_matcher(
    model_dir: str | os.PathLike[str],
) -> Callable[[object], bool]:
    """Return a predicate matching ``safe_open`` filenames in *model_dir*.

    A shard matches when any spelling of its directory -- as written, made
    absolute, or followed through symlinks -- equals any spelling of
    *model_dir*. Only direct children match, mirroring the
    ``glob(model_path / "*.safetensors")`` scan the loaders perform.

    The argument is deliberately ``object``: filenames reach ``safe_open`` from
    callers that oMLX does not control, and anything that is not a
    safetensors path must simply not match rather than abort the load.
    """
    raw_dir = os.fsdecode(os.fspath(model_dir))
    targets = {Path(raw_dir), Path(os.path.abspath(raw_dir))}
    try:
        targets.add(Path(raw_dir).resolve())
    except (OSError, RuntimeError):
        pass

    def matches(filename: object) -> bool:
        name = _as_filename_str(filename)
        if name is None or not name.endswith(".safetensors"):
            return False
        return bool(_candidate_parents(name) & targets)

    return matches
