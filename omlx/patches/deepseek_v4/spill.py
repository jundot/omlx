"""Spill-stacking for per-expert DeepSeek-V4 checkpoints.

JANGQ DeepSeek-V4 checkpoints store routed experts unstacked
(``layers.N.ffn.experts.{i}.w{1,2,3}.*``: 256 experts x 43 layers). The
stock sanitize stacks them into ``switch_mlp`` banks with lazy ``mx.stack``
graphs — cheap to build, but the load-time ``mx.eval`` materializes ~64 GiB
at once and the kernel SIGKILLs the process on a 48 GiB box (the old oQ4e
checkpoint stored experts pre-fused, so it never stacked).

Spill-stacking keeps peak transient memory at one layer (~1.6 GiB): each
layer's banks are stacked, evaluated, saved to a spill shard next to the
model, and reloaded as memory-mapped lazy arrays. The strict load then
sees the exact stacked keys it expects, and expert streaming (which runs
before ``materialize_lazy_state`` and replaces the MoE modules) drops
those arrays before they are ever faulted into RAM.

The spill directory persists across runs: a manifest records the source
shards' sizes+mtimes, so a repeat load skips re-spilling entirely.
Spill layout (all paths outside the checkpoint dir — discovery globs
``**/*.safetensors`` recursively, so nothing may be written inside it)::

    <model-parent>/.omlx_spill/<model-dirname>/
        manifest.json
        spill_layer_00.safetensors ... spill_layer_42.safetensors

Set ``OMLX_DSV4_SPILL=0`` to restore the legacy in-RAM stacking, or
``OMLX_SPILL_DIR`` to relocate the spill root.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SPILL_VERSION = 1
_MANIFEST_NAME = "manifest.json"
_LAYER_FILE = "spill_layer_{idx:02d}.safetensors"

# Load-time context: model_loading sets this from the load_config wrapper
# (which receives the checkpoint path) before mlx_lm.load runs sanitize.
_CTX_MODEL_PATH: str | None = None

# (checkpoint src, stacked dst) projection mapping shared with sanitize.
_PROJ_MAP = (("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj"))
_SUFFIXES = ("weight", "scales", "biases")


def set_spill_model_path(path: str | os.PathLike | None) -> None:
    """Record the checkpoint dir for the in-flight load (or clear it)."""
    global _CTX_MODEL_PATH
    _CTX_MODEL_PATH = str(path) if path is not None else None


def spill_model_path() -> str | None:
    """Checkpoint dir recorded for the in-flight load, if any."""
    return _CTX_MODEL_PATH


def spill_dir_for(model_path: str | os.PathLike) -> Path:
    """Spill directory for a checkpoint (volume-local, outside the dir)."""
    model_path = Path(model_path)
    root = os.environ.get("OMLX_SPILL_DIR")
    base = Path(root).expanduser() if root else model_path.parent / ".omlx_spill"
    return base / model_path.name


def spill_disabled() -> bool:
    """Escape hatch: OMLX_DSV4_SPILL=0 restores in-RAM stacking."""
    return os.environ.get("OMLX_DSV4_SPILL", "1") == "0"


def _source_files(model_path: Path) -> list[Path]:
    return sorted(model_path.glob("model*.safetensors"))


def _source_sig(model_path: Path) -> dict[str, list[int]]:
    sig: dict[str, list[int]] = {}
    for fp in _source_files(model_path):
        try:
            st = fp.stat()
        except OSError:
            continue
        sig[fp.name] = [st.st_size, st.st_mtime_ns]
    return sig


def read_manifest(spill_dir: Path) -> dict[str, Any] | None:
    """Parsed manifest, or None when absent/unreadable."""
    try:
        return json.loads((spill_dir / _MANIFEST_NAME).read_text())
    except Exception:
        return None


def spill_is_valid(model_path: str | os.PathLike) -> Path | None:
    """Spill dir when a fresh spill exists for this checkpoint, else None."""
    mp = Path(model_path)
    sd = spill_dir_for(mp)
    manifest = read_manifest(sd)
    if not manifest or manifest.get("version") != _SPILL_VERSION:
        return None
    if manifest.get("model_path") != str(mp):
        return None
    if manifest.get("source") != _source_sig(mp):
        return None
    for fname in manifest.get("files") or []:
        if not (sd / fname).is_file():
            return None
    return sd


def write_manifest(
    spill_dir: Path,
    model_path: Path,
    files: list[str],
    key_to_file: dict[str, str],
) -> None:
    """Persist the spill manifest after a full re-spill."""
    spill_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": _SPILL_VERSION,
        "model_path": str(model_path),
        "source": _source_sig(model_path),
        "files": files,
        "key_to_file": key_to_file,
    }
    (spill_dir / _MANIFEST_NAME).write_text(json.dumps(manifest, indent=1))


def spill_key_to_file(manifest: dict[str, Any]) -> dict[str, str]:
    """Explicit stacked-key -> spill filename mapping from a manifest."""
    raw = manifest.get("key_to_file") or {}
    return {str(k): str(v) for k, v in raw.items()}


def spill_layer_name(layer_idx: int) -> str:
    """Spill shard filename for one layer."""
    return _LAYER_FILE.format(idx=layer_idx)


def expected_layer_keys(layer_idx: int) -> list[str]:
    """Stacked key names one layer's spill shard must contain."""
    return [
        f"model.layers.{layer_idx}.ffn.switch_mlp.{dst}.{suffix}"
        for _, dst in _PROJ_MAP
        for suffix in _SUFFIXES
    ]


def spill_layer_ok(spill_dir: Path, layer_idx: int) -> bool:
    """True when the layer shard exists with a complete header.

    A kill mid-save leaves a truncated file; the header check (all nine
    stacked keys present) lets a restarted spill resume instead of
    serving a corrupt shard.
    """
    import struct as _struct

    path = spill_dir / spill_layer_name(layer_idx)
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        with path.open("rb") as f:
            hsize = _struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(hsize))
        want = expected_layer_keys(layer_idx)
        return all(k in hdr for k in want)
    except Exception:
        return False


def stack_layer_to_spill(
    weights: dict[str, Any],
    *,
    layer_idx: int,
    n_experts: int,
    spill_dir: Path,
) -> dict[str, Any]:
    """Stack one layer's experts to a spill shard; return mmap arrays.

    Pops ``model.layers.{i}.ffn.experts.{e}.*`` from *weights*, stacks each
    (src, suffix) bank, evaluates just this layer (~2 GiB transient),
    saves ``spill_layer_{i}.safetensors`` under the switch_mlp names, and
    reloads the shard as memory-mapped lazy arrays. The caller merges the
    return value back into *weights*.
    """
    import gc as _gc

    import mlx.core as mx

    prefix = f"model.layers.{layer_idx}.ffn.experts"
    stacked: dict[str, Any] = {}
    for src, dst in _PROJ_MAP:
        for suffix in _SUFFIXES:
            key0 = f"{prefix}.0.{src}.{suffix}"
            if key0 not in weights:
                continue
            parts = [
                weights.pop(f"{prefix}.{e}.{src}.{suffix}")
                for e in range(n_experts)
            ]
            stacked[f"model.layers.{layer_idx}.ffn.switch_mlp.{dst}.{suffix}"] = mx.stack(parts)
            del parts
    if not stacked:
        return {}
    mx.eval(stacked)
    fname = spill_layer_name(layer_idx)
    spill_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(spill_dir / fname), stacked)
    # Reload as memory-mapped lazy arrays: no RAM held past this point.
    reloaded = mx.load(str(spill_dir / fname))
    # Drop the evaluated banks AND flush the Metal buffer pool: without
    # this each layer's ~2 GiB stays wired and 43 layers OOM the box.
    del stacked
    _gc.collect()
    try:
        mx.metal.clear_cache()
    except Exception:
        pass
    return dict(reloaded)


def load_spill_into(weights: dict[str, Any], spill_dir: Path) -> list[str]:
    """Merge every spill shard into *weights* as mmap arrays.

    Returns the stacked key names served (for backing registration).
    """
    import mlx.core as mx

    manifest = read_manifest(spill_dir) or {}
    served: list[str] = []
    for fname in manifest.get("files") or sorted(
        p.name for p in spill_dir.glob("spill_layer_*.safetensors")
    ):
        for k, v in mx.load(str(spill_dir / fname)).items():
            weights[k] = v
            served.append(k)
    return served
