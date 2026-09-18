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
shards' sizes+mtimes plus the spill mode (how many MTP stage banks were
spilled and which key spelling they use), so a repeat load skips
re-spilling entirely — and a load whose mode differs (MTP or DSpark
toggled) misses and re-spills instead of serving mismatched keys.
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

from ..expert_streaming._env import env_bool, env_str

logger = logging.getLogger(__name__)

# Version 3: manifests now record the spill mode (``mode.mtp_stages`` +
# ``mode.block_part`` — how many MTP stage banks were spilled and the
# ``mtp.{i}[.block].ffn.`` key spelling). A load whose expected mode
# differs must miss and re-spill: an MTP-on spill injected into an
# MTP-off model would fail strict load on unexpected mtp.* keys, and an
# MTP-off spill under an MTP-on model would leave the stage banks
# missing. v2 manifests are still accepted — they predate the mode
# block, so the mode is derived from the ``key_to_file`` spelling, which
# records exactly which banks the spill serves. v1 manifests stay
# rejected (they can validate while serving no mtp.* stacked keys).
_SPILL_VERSION = 3
_SPILL_READ_VERSIONS = frozenset((2, _SPILL_VERSION))
_MANIFEST_NAME = "manifest.json"
_LAYER_FILE = "spill_layer_{idx:02d}.safetensors"
_MTP_FILE = "spill_mtp_{idx:02d}.safetensors"

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
    root = env_str("OMLX_SPILL_DIR", None)
    base = Path(root).expanduser() if root else model_path.parent / ".omlx_spill"
    return base / model_path.name


def spill_disabled() -> bool:
    """Escape hatch: OMLX_DSV4_SPILL=0 restores in-RAM stacking."""
    return not env_bool("OMLX_DSV4_SPILL", True)


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


def manifest_mode(manifest: dict[str, Any]) -> tuple[int, str | None]:
    """Spill mode recorded in — or derivable from — a manifest.

    Returns ``(mtp_stages, block_part)``: how many MTP stage banks the
    spill serves and the key spelling they sit under (``".block"`` for
    Lightning MTP, ``""`` for DSpark). ``block_part`` is None iff no MTP
    banks were spilled — with zero stages the spelling serves nothing, so
    it is normalized away.

    v3 manifests carry an explicit ``mode`` block; v2 manifests predate
    it, so the mode is derived from the ``key_to_file`` map (and the
    ``files`` list as a floor), which spells out exactly which banks the
    spill serves. A derived mode can only *mismatch* a load's
    expectation — never hallucinate banks the spill does not serve.
    """
    mode = manifest.get("mode")
    if isinstance(mode, dict):
        try:
            stages = int(mode.get("mtp_stages") or 0)
        except (TypeError, ValueError):
            stages = 0
        bp = mode.get("block_part")
        # Zero stages => the spelling serves nothing; normalize to None.
        return stages, (bp if stages and isinstance(bp, str) else None)
    stage_ids: set[int] = set()
    bp: str | None = None
    raw = manifest.get("key_to_file")
    if not isinstance(raw, dict):
        raw = {}
    for key in raw:
        if not isinstance(key, str) or not key.startswith("mtp."):
            continue
        parts = key.split(".")
        if len(parts) < 3 or not parts[1].isdigit():
            continue
        stage_ids.add(int(parts[1]))
        bp = ".block" if parts[2] == "block" else ""
    if stage_ids:
        return max(stage_ids) + 1, bp
    # key_to_file lists no mtp.* banks; the files list is the floor —
    # spill_mtp_* shards with no recorded keys still mean the writing
    # load spilled stages (spelling then unknown → never matches a
    # block_part-expecting load).
    files = manifest.get("files")
    n_mtp_files = (
        sum(1 for f in files if str(f).startswith("spill_mtp_"))
        if isinstance(files, (list, tuple))
        else 0
    )
    return (n_mtp_files, None) if n_mtp_files else (0, None)


def spill_is_valid(
    model_path: str | os.PathLike,
    *,
    mtp_stages: int | None = None,
    block_part: str | None = None,
) -> Path | None:
    """Spill dir when a fresh spill exists for this checkpoint, else None.

    When ``mtp_stages`` is given, the manifest's spill mode (recorded, or
    derived for pre-v3 manifests) must equal ``(mtp_stages,
    block_part)`` — a mismatch means the shards serve a key set this load
    cannot consume, so the caller must treat the dir as a miss and
    re-spill. Callers that do not know the load's mode (e.g. the
    expert-streaming map absorb) omit it and get the source/freshness
    check only.
    """
    mp = Path(model_path)
    sd = spill_dir_for(mp)
    manifest = read_manifest(sd)
    if not manifest or manifest.get("version") not in _SPILL_READ_VERSIONS:
        return None
    if manifest.get("model_path") != str(mp):
        return None
    if manifest.get("source") != _source_sig(mp):
        return None
    # Require a non-empty file list: an empty/corrupt manifest must not
    # validate (load_spill_into would otherwise glob stale shards).
    _files = manifest.get("files") or []
    if not _files:
        return None
    for fname in _files:
        if not (sd / fname).is_file():
            return None
    if mtp_stages is not None:
        expected = (int(mtp_stages), block_part if mtp_stages else None)
        served = manifest_mode(manifest)
        if served != expected:
            logger.info(
                "dsv4 spill mode mismatch at %s: serves (mtp_stages=%s, "
                "block_part=%r), load expects %s — re-spilling",
                sd,
                served[0],
                served[1],
                expected,
            )
            return None
    return sd


def write_manifest(
    spill_dir: Path,
    model_path: Path,
    files: list[str],
    key_to_file: dict[str, str],
    *,
    mtp_stages: int = 0,
    block_part: str | None = None,
) -> None:
    """Persist the spill manifest after a full re-spill.

    ``mtp_stages``/``block_part`` record the spill mode: how many MTP
    stage banks were spilled and under which key spelling (``".block"``
    for Lightning MTP, ``""`` for DSpark). A later load whose expected
    mode differs treats the dir as a miss and re-spills.
    """
    spill_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": _SPILL_VERSION,
        "model_path": str(model_path),
        "source": _source_sig(model_path),
        "files": files,
        "key_to_file": key_to_file,
        "mode": {
            "mtp_stages": int(mtp_stages),
            "block_part": block_part if mtp_stages else None,
        },
    }
    (spill_dir / _MANIFEST_NAME).write_text(json.dumps(manifest, indent=1))


def spill_key_to_file(manifest: dict[str, Any]) -> dict[str, str]:
    """Explicit stacked-key -> spill filename mapping from a manifest."""
    raw = manifest.get("key_to_file") or {}
    return {str(k): str(v) for k, v in raw.items()}


def spill_layer_name(layer_idx: int) -> str:
    """Spill shard filename for one layer."""
    return _LAYER_FILE.format(idx=layer_idx)


def spill_mtp_name(mtp_idx: int) -> str:
    """Spill shard filename for one MTP stage's stacked expert bank."""
    return _MTP_FILE.format(idx=mtp_idx)


def stack_bank_to_spill(
    weights: dict[str, Any],
    *,
    src_prefix: str,
    dst_prefix: str,
    n_experts: int,
    spill_dir: Path,
    file_name: str,
) -> dict[str, Any]:
    """Stack one expert bank to a spill shard; return mmap arrays.

    Generic form of the backbone layer spill: pops
    ``{src_prefix}.{e}.{w1,w2,w3}.{weight,scales,biases}`` from
    *weights*, stacks each (src, suffix) bank, evaluates just this bank
    (~2 GiB transient), saves ``file_name`` under the
    ``{dst_prefix}.{gate_proj,down_proj,up_proj}.{suffix}`` names, and
    reloads the shard as memory-mapped lazy arrays. The caller merges
    the return value back into *weights*.
    """
    import gc as _gc

    import mlx.core as mx

    stacked: dict[str, Any] = {}
    for src, dst in _PROJ_MAP:
        for suffix in _SUFFIXES:
            key0 = f"{src_prefix}.0.{src}.{suffix}"
            if key0 not in weights:
                continue
            parts = [
                weights.pop(f"{src_prefix}.{e}.{src}.{suffix}")
                for e in range(n_experts)
            ]
            stacked[f"{dst_prefix}.{dst}.{suffix}"] = mx.stack(parts)
            del parts
    if not stacked:
        return {}
    mx.eval(stacked)
    spill_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(spill_dir / file_name), stacked)
    # Reload as memory-mapped lazy arrays: no RAM held past this point.
    reloaded = mx.load(str(spill_dir / file_name))
    # Drop the evaluated banks AND flush the Metal buffer pool: without
    # this each layer's ~2 GiB stays wired and 43 layers OOM the box.
    del stacked
    _gc.collect()
    try:
        mx.metal.clear_cache()
    except Exception:
        pass
    return dict(reloaded)


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
    return stack_bank_to_spill(
        weights,
        src_prefix=f"model.layers.{layer_idx}.ffn.experts",
        dst_prefix=f"model.layers.{layer_idx}.ffn.switch_mlp",
        n_experts=n_experts,
        spill_dir=spill_dir,
        file_name=spill_layer_name(layer_idx),
    )


def load_spill_into(weights: dict[str, Any], spill_dir: Path) -> list[str]:
    """Merge every spill shard into *weights* as mmap arrays.

    Returns the stacked key names served (for backing registration).
    """
    import mlx.core as mx

    manifest = read_manifest(spill_dir) or {}
    served: list[str] = []
    for fname in manifest.get("files") or sorted(
        p.name for p in spill_dir.glob("spill_*.safetensors")
    ):
        for k, v in mx.load(str(spill_dir / fname)).items():
            weights[k] = v
            served.append(k)
    return served
