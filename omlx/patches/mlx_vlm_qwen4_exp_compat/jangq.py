# SPDX-License-Identifier: Apache-2.0
"""JANGQ mixed-precision contract reconstruction for Qwen4-Exp checkpoints."""

from __future__ import annotations

import hashlib
import json
import logging
import struct
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CACHE_DIR = Path.home() / ".cache" / "omlx" / "jangq"
_PACKED_DTYPES = ("U32", "uint32")
_SCALE_DTYPES = ("F16", "BF16", "F32")
_ALLOWED_BITS = (2, 3, 4, 5, 8)
_SKIP_SUBSTRINGS = (".ple.",)
_SKIP_PREFIXES = ("mtp.",)


def infer_packing(
    stored_shape: tuple[int, ...],
    scales_shape: tuple[int, ...],
    logical_shape: tuple[int, ...],
) -> tuple[int, int] | None:
    """Infer ``(bits, group_size)`` for one packed triple.

    ``stored_shape`` is the packed U32 weight, ``scales_shape`` the scales
    tensor, ``logical_shape`` the owning module's dense weight shape.
    Returns ``None`` when the shapes do not describe a supported packing.
    """
    if len(stored_shape) != len(logical_shape) or len(scales_shape) != len(logical_shape):
        return None
    if tuple(stored_shape[:-1]) != tuple(logical_shape[:-1]):
        return None
    if tuple(scales_shape[:-1]) != tuple(logical_shape[:-1]):
        return None
    stored_numel = 1
    for x in stored_shape:
        stored_numel *= x
    logical_numel = 1
    for x in logical_shape:
        logical_numel *= x
    if logical_numel == 0 or (stored_numel * 32) % logical_numel != 0:
        return None
    bits = (stored_numel * 32) // logical_numel
    if bits not in _ALLOWED_BITS:
        return None
    if scales_shape[-1] == 0 or logical_shape[-1] % scales_shape[-1] != 0:
        return None
    group = logical_shape[-1] // scales_shape[-1]
    if group <= 0:
        return None
    return (bits, group)


def _read_headers(model_dir: Path, weight_map: dict[str, str]) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Read safetensors headers only; return key -> (dtype, shape)."""
    by_file: dict[str, list[str]] = {}
    for key, fname in weight_map.items():
        by_file.setdefault(fname, []).append(key)
    tensors: dict[str, tuple[str, tuple[int, ...]]] = {}
    for fname in sorted(by_file):
        path = model_dir / fname
        with open(path, "rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(n))
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            tensors[key] = (meta["dtype"], tuple(meta["shape"]))
    return tensors


def _is_packed_triple(
    key: str, tensors: dict[str, tuple[str, tuple[int, ...]]]
) -> bool:
    if not key.endswith(".weight"):
        return False
    dtype, _ = tensors[key]
    if dtype not in _PACKED_DTYPES:
        return False
    base = key[: -len(".weight")]
    scales = tensors.get(base + ".scales")
    biases = tensors.get(base + ".biases")
    if scales is None or biases is None:
        return False
    return scales[0] in _SCALE_DTYPES


def detect_jangq(model_dir: str | Path) -> dict[str, Any] | None:
    """Detect a JANGQ-packed checkpoint without quantization metadata.

    Returns a fingerprint dict when the directory holds packed U32 triples
    and declares no quantization policy; otherwise ``None``.
    """
    root = Path(model_dir)
    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if config.get("quantization") or config.get("quantization_config"):
        return None
    try:
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    except (OSError, KeyError, json.JSONDecodeError):
        return None
    try:
        tensors = _read_headers(root, weight_map)
    except OSError:
        return None
    triples = [k for k in tensors if _is_packed_triple(k, tensors)]
    if not triples:
        return None
    stat = index_path.stat()
    cfg_stat = config_path.stat()
    fingerprint = hashlib.sha1(
        f"{root.resolve()}|{stat.st_mtime_ns}|{stat.st_size}|"
        f"{cfg_stat.st_mtime_ns}|{len(triples)}".encode()
    ).hexdigest()
    return {
        "model_dir": str(root),
        "fingerprint": fingerprint,
        "triples": len(triples),
    }


def _resolve_module(root: Any, key: str) -> Any | None:
    node = root
    try:
        for part in key.split(".")[:-1]:
            if isinstance(node, (list, tuple)):
                node = node[int(part)]
            elif isinstance(node, dict):
                node = node[part]
            else:
                node = getattr(node, part)
    except (AttributeError, KeyError, IndexError, ValueError):
        return None
    return node


def _runtime_key(raw_key: str) -> str:
    from mlx_vlm.models.qwen4_exp.qwen4_exp import (
        _NGRAM_SHARD_RE,
        _normalize_checkpoint_key,
    )

    try:
        from mlx_vlm.models.qwen3_5.qwen3_5 import sanitize_key
    except ImportError:
        def sanitize_key(k: str) -> str:  # type: ignore[no-redef]
            return k

    key = sanitize_key(_normalize_checkpoint_key(raw_key))
    return _NGRAM_SHARD_RE.sub(r".ngram_embedding.shards.\1", key)


def synthesize_quantization(model_dir: str | Path) -> dict[str, Any]:
    """Rebuild the mlx-vlm quantization policy for a JANGQ checkpoint.

    Audits packed triples against the live module shapes and returns a
    policy dict with a global ``(bits, group_size, mode)`` default plus
    per-module overrides for every projection that differs from it.
    Raises ``ValueError`` when no packed triple resolves.
    """
    root = Path(model_dir)
    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    tensors = _read_headers(root, index["weight_map"])
    raw_config = json.loads((root / "config.json").read_text(encoding="utf-8"))

    from mlx_vlm.models.qwen4_exp.config import ModelConfig
    from mlx_vlm.models.qwen4_exp.qwen4_exp import Model

    model = Model(ModelConfig.from_dict(raw_config))

    votes: Counter[tuple[int, int]] = Counter()
    per_module: dict[str, tuple[int, int]] = {}
    skipped = Counter()
    for key in tensors:
        if not _is_packed_triple(key, tensors):
            continue
        final = _runtime_key(key)
        if any(s in final for s in _SKIP_SUBSTRINGS) or any(
            final.startswith(p) for p in _SKIP_PREFIXES
        ):
            skipped["ple_or_mtp"] += 1
            continue
        module = _resolve_module(model, final)
        weight = getattr(module, "weight", None) if module is not None else None
        if weight is None or not hasattr(weight, "shape"):
            skipped["no_module"] += 1
            continue
        base = key[: -len(".weight")]
        packing = infer_packing(
            tensors[key][1], tensors[base + ".scales"][1], tuple(weight.shape)
        )
        if packing is None:
            skipped["bad_packing"] += 1
            continue
        votes[packing] += 1
        per_module[final[: -len(".weight")]] = packing
    if not votes:
        raise ValueError(f"no JANGQ triple resolved for {root} (skipped={dict(skipped)})")
    global_packing = votes.most_common(1)[0][0]
    policy: dict[str, Any] = {
        "group_size": global_packing[1],
        "bits": global_packing[0],
        "mode": "affine",
    }
    for module_key, (bits, group) in sorted(per_module.items()):
        if (bits, group) != global_packing:
            policy[module_key] = {"bits": bits, "group_size": group, "mode": "affine"}
    logger.info(
        "JANGQ contract for %s: global=%s overrides=%d skipped=%s",
        root,
        global_packing,
        sum(1 for v in policy.values() if isinstance(v, dict)),
        dict(skipped),
    )
    return policy


def _cache_path(fingerprint: str) -> Path:
    return _CACHE_DIR / f"{fingerprint}.json"


def load_cached_policy(fingerprint: str) -> dict[str, Any] | None:
    try:
        return json.loads(_cache_path(fingerprint).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def store_cached_policy(fingerprint: str, policy: dict[str, Any]) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(fingerprint).write_text(json.dumps(policy), encoding="utf-8")
    except OSError:
        logger.debug("JANGQ policy cache write failed", exc_info=True)


def quantization_for_model(model_dir: str | Path) -> dict[str, Any] | None:
    """Return the quantization policy for a JANGQ dir, using the cache."""
    found = detect_jangq(model_dir)
    if found is None:
        return None
    cached = load_cached_policy(found["fingerprint"])
    if cached is not None:
        return cached
    try:
        policy = synthesize_quantization(model_dir)
    except Exception:  # noqa: BLE001
        logger.warning("JANGQ synthesis failed for %s", model_dir, exc_info=True)
        return None
    store_cached_policy(found["fingerprint"], policy)
    return policy
