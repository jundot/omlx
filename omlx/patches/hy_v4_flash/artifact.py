# SPDX-License-Identifier: Apache-2.0
"""Validate a local Flash export before allocating model parameters or wiring memory."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .quant_io import SUPPORTED_BITS, SUPPORTED_TRUNK_BITS, TRUNK_GROUP, TRUNK_LINEAR

SCHEMA = "hy4-flash-oqh/v1"
SOURCE = "tencent/Hy4-flash"
REVISION = "d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959"
INDEX = "model.safetensors.index.json"


def _unique(pairs):
    doc = dict(pairs)
    if len(pairs) != len(doc):
        raise ValueError("duplicate JSON key in model metadata")
    return doc


def _json(path):
    if not 0 < path.stat().st_size <= 32 << 20:
        raise ValueError(f"invalid metadata size: {path.name}")
    return json.loads(path.read_bytes(), object_pairs_hook=_unique)


def member(root, name):
    """The portable manifest may only reference regular files in this directory."""
    if (
        not isinstance(name, str)
        or re.fullmatch(r"[a-zA-Z0-9_][-a-zA-Z0-9_.]*", name) is None
    ):
        raise ValueError(f"invalid model member name: {name!r}")
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or indirect model member: {name}")
    return path


def _validate_files(root, records, *, digest):
    if not isinstance(records, dict) or not records:
        raise ValueError("missing model file descriptions")
    for name, record in records.items():
        path = member(root, name)
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("sha256"), str)
            or re.fullmatch("[a-f0-9]{64}", record["sha256"]) is None
            or type(record.get("size")) is not int
            or record["size"] != path.stat().st_size
        ):
            raise ValueError(f"wrong size or invalid checksum record: {name}")
        if digest:
            with path.open("rb") as f:
                actual = hashlib.file_digest(f, "sha256").hexdigest()
            if actual != record["sha256"]:
                raise ValueError(f"SHA-256 mismatch: {name}")


def _validate_trunk(trunk, index):
    """Optional MLX-quantized trunk: all or none of the eligible linears."""
    eligible = {k.removesuffix(".weight") for k in index if TRUNK_LINEAR.fullmatch(k)}
    if trunk is None:
        if any(m + ".scales" in index for m in eligible):
            raise ValueError("quantized trunk tensors without trunk_quantization")
        return
    modules = trunk.get("modules") if isinstance(trunk, dict) else None
    if (
        trunk.get("bits") not in SUPPORTED_TRUNK_BITS
        or trunk.get("group_size") != TRUNK_GROUP
        or trunk.get("mode") != "affine"
        or not isinstance(modules, list)
        or len(set(modules)) != len(modules)
        or set(modules) != eligible
        or not all(
            m + suffix in index for m in modules for suffix in (".scales", ".biases")
        )
    ):
        raise ValueError("invalid or incomplete trunk_quantization")


@dataclass
class Artifact:
    root: Path
    manifest: dict
    config: dict
    weight_map: dict[str, str]

    @classmethod
    def open(cls, directory: Path, *, verify_core: bool = False):
        root = Path(directory).resolve(strict=True)
        doc = _json(member(root, "manifest.json"))
        src = doc.get("source", {})
        if (
            doc.get("schema") != SCHEMA
            or doc.get("text_only") is not True
            or src.get("repo") != SOURCE
            or src.get("revision") != REVISION
        ):
            raise ValueError(
                "expected the converted public Hy4 Flash instruct text artifact"
            )
        assets = doc.get("assets", {})
        required = {"config.json", INDEX, "tokenizer.json", "chat_template.jinja"}
        if not required.issubset(assets):
            raise ValueError("missing authenticated model, index, or tokenizer assets")
        _validate_files(root, assets, digest=True)
        core = doc.get("core_shards", {})
        if any(not n.endswith(".safetensors") for n in core):
            raise ValueError("unexpected core shard filename")
        _validate_files(root, core, digest=verify_core)
        cfg = _json(root / "config.json")
        if cfg.get("model_type") != "hy_v4_flash" or "vision_config" in cfg:
            raise ValueError("expected a flattened Flash text configuration")
        iq = cfg.get("omlx_oqh", {})
        targets = iq.get("tensors", {})
        if iq.get("version") != 1 or not targets:
            raise ValueError("expected version-1 packed oQh tensors")
        index = _json(root / INDEX).get("weight_map")
        if (
            not isinstance(index, dict)
            or not index
            or set(index.values()) != core.keys()
        ):
            raise ValueError("weight index does not describe exactly the core shards")
        for prefix, spec in targets.items():
            if spec not in [
                {"bits": b, "block": 256, "group_size": 128} for b in SUPPORTED_BITS
            ] or not all(
                prefix + suffix in index
                for suffix in (".weight", ".oqh_signs", ".oqh_scales", ".oqh_biases")
            ):
                raise ValueError(
                    f"invalid or incomplete rotated expert projection: {prefix}"
                )
        expected = {
            f"model.layers.{i}.mlp.experts.{bank}"
            for i, kind in enumerate(cfg.get("mlp_layer_types", []))
            if kind == "sparse"
            for bank in ("gate_up_proj", "down_proj")
        }
        if set(targets) != expected:
            raise ValueError(
                "rotation plan does not cover exactly the routed expert banks"
            )
        _validate_trunk(cfg.get("trunk_quantization"), index)
        return cls(root, doc, cfg, index)

    def summary(self):
        return {
            "source": SOURCE,
            "revision": REVISION,
            "schema": SCHEMA,
            "text_only": True,
            "core_shards": len(self.manifest["core_shards"]),
            "core_tensors": len(self.weight_map),
            "core_bytes": sum(v["size"] for v in self.manifest["core_shards"].values()),
            "rotated_expert_banks": len(self.config["omlx_oqh"]["tensors"]),
            "trunk_bits": (self.config.get("trunk_quantization") or {}).get("bits"),
            "external_bf16_blocks": sorted(
                int(k) for k in self.manifest.get("external_tables", {})
            ),
        }
