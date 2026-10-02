# SPDX-License-Identifier: Apache-2.0
"""Let pipeline loading accept models whose weights are not all in the index.

``mlx_lm.utils.load`` selects which shards a pipeline rank must fetch by looking
every model parameter up in ``model.safetensors.index.json``::

    for k, _ in tree_flatten(model.parameters()):
        if file_name := weight_index.get(k, None) is None:
            raise ValueError(
                "Pipeline loading is only supported for MLX converted models."
            )
        local_files.add(weight_index[k])

Two lines later the same function calls ``load_model(..., strict=False)``, which
tolerates exactly the parameters this loop rejects. So the guard is stricter
than the code it guards.

That difference matters for architectures oMLX patches in. GLM-5.2
(``glm_moe_dsa``) declares sparse-attention indexer parameters that its
checkpoint legitimately does not carry — oMLX's own patch exists because
"pinned mlx-lm ... cannot load checkpoints whose shared DSA layers carry no
indexer weights". The model loads fine on one Mac and is refused the moment it
is pipelined across two.

Rather than vendor a copy of ``load`` — which would drift from upstream — this
wraps the index as it is read, so an absent parameter resolves to a real shard
name instead of ``None``. Nothing else about loading changes: the weights that
exist are still loaded from the files that hold them, and the ones that do not
exist are still initialised by the model, exactly as ``strict=False`` intends.
"""

from __future__ import annotations

import builtins
import io
import json as _real_json
import struct
from pathlib import Path
from typing import Any

_APPLIED = False
_MAX_HEADER_BYTES = 64 * 1024 * 1024
_real_open = builtins.open


class TolerantWeightMap(dict):
    """A weight map that never reports a parameter as unmapped.

    Unknown keys resolve to an arbitrary real shard name. That name is only
    used to build the set of files to fetch, so naming one extra file is
    harmless — and for a locally staged model the fetch is a no-op anyway.
    Returning ``None`` instead would trip the upstream guard.
    """

    def _fallback(self) -> Any:
        for value in dict.values(self):
            return value
        return "model.safetensors"

    def get(self, key: Any, default: Any = None) -> Any:
        value = dict.get(self, key, None)
        return value if value is not None else self._fallback()

    def __getitem__(self, key: Any) -> Any:
        value = dict.get(self, key, None)
        return value if value is not None else self._fallback()

    def __contains__(self, key: object) -> bool:
        return True


class _JsonProxy:
    """Stand-in for ``json`` inside ``mlx_lm.utils``.

    Delegates everything to the real module and only intervenes when the object
    being loaded is a safetensors index, so config reads and every other use of
    ``json`` in that module are untouched.
    """

    def __getattr__(self, name: str) -> Any:
        return getattr(_real_json, name)

    def load(self, fp: Any, **kwargs: Any) -> Any:
        data = _real_json.load(fp, **kwargs)
        if isinstance(data, dict) and isinstance(data.get("weight_map"), dict):
            data["weight_map"] = TolerantWeightMap(data["weight_map"])
        return data


def _open_with_single_file_index(
    file: Any,
    mode: str = "r",
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Synthesize the optional HF index for one-file safetensors exports.

    ``mlx_lm.utils.load`` requires an index whenever a pipeline group is
    supplied, even though a valid single-file MLX model normally has no index.
    The model directory is not mutated: only that one missing read receives an
    in-memory weight map built from the safetensors header.
    """

    try:
        path = Path(file)
    except TypeError:
        return _real_open(file, mode, *args, **kwargs)
    if (
        path.name != "model.safetensors.index.json"
        or path.exists()
        or "r" not in mode
        or "b" in mode
    ):
        return _real_open(file, mode, *args, **kwargs)

    shards = sorted(path.parent.glob("*.safetensors"))
    if len(shards) != 1:
        return _real_open(file, mode, *args, **kwargs)
    shard = shards[0]
    with _real_open(shard, "rb") as stream:
        encoded = stream.read(8)
        if len(encoded) != 8:
            return _real_open(file, mode, *args, **kwargs)
        header_size = struct.unpack("<Q", encoded)[0]
        if not 0 < header_size <= _MAX_HEADER_BYTES:
            return _real_open(file, mode, *args, **kwargs)
        header = _real_json.loads(stream.read(header_size))
    weight_map = {
        name: shard.name
        for name in header
        if name != "__metadata__"
    }
    return io.StringIO(_real_json.dumps({"weight_map": weight_map}))


def is_applied() -> bool:
    return _APPLIED


def apply_mlx_lm_pipeline_index_patch() -> bool:
    """Make pipeline shard selection tolerant. Idempotent.

    Returns True when the patch is in place. Raises only if ``mlx_lm.utils``
    cannot be imported, since a silent no-op here surfaces much later as
    "Pipeline loading is only supported for MLX converted models".
    """

    global _APPLIED
    if _APPLIED:
        return True

    from mlx_lm import utils as mlx_lm_utils

    mlx_lm_utils.json = _JsonProxy()
    mlx_lm_utils.open = _open_with_single_file_index
    _APPLIED = True
    return True


__all__ = [
    "TolerantWeightMap",
    "apply_mlx_lm_pipeline_index_patch",
    "is_applied",
]
