# SPDX-License-Identifier: Apache-2.0
"""Validate a local Flash export before allocating model parameters or wiring memory."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .quant_io import (
    SUPPORTED_TRUNK_BITS,
    TRUNK_GROUP,
    TRUNK_LINEAR,
    unique,
    valid_bank_spec,
)
from .residency import table_specs

SCHEMA = "hy4-flash-oqh/v1"
SOURCE = "tencent/Hy4-flash"
REVISION = "d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959"
INDEX = "model.safetensors.index.json"


def _json(path):
    if not 0 < path.stat().st_size <= 32 << 20:
        raise ValueError(f"invalid metadata size: {path.name}")
    doc = json.loads(path.read_bytes(), object_pairs_hook=unique)
    if not isinstance(doc, dict):
        raise ValueError(f"expected a JSON object in {path.name}")
    return doc


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
    if not isinstance(trunk, dict):
        raise ValueError("trunk_quantization must be an object")
    modules = trunk.get("modules")
    if (
        type(trunk.get("bits")) is not int
        or trunk["bits"] not in SUPPORTED_TRUNK_BITS
        or type(trunk.get("group_size")) is not int
        or trunk["group_size"] != TRUNK_GROUP
        or trunk.get("mode") != "affine"
        or not isinstance(modules, list)
        or not all(isinstance(m, str) for m in modules)
        or len(set(modules)) != len(modules)
        or set(modules) != eligible
        or not all(
            m + suffix in index for m in modules for suffix in (".scales", ".biases")
        )
    ):
        raise ValueError("invalid or incomplete trunk_quantization")


def _validate_quantization(manifest, targets, trunk):
    # Older v1 exports may omit the manifest summary, but never a packed trunk.
    if "quantization" not in manifest and trunk is None:
        return
    from .convert import ALGORITHM, trunk_record

    record = manifest.get("quantization")
    bits = {spec["bits"] for spec in targets.values()}
    if not isinstance(record, dict) or len(bits) != 1:
        raise ValueError("invalid manifest quantization record")
    expected = dict(ALGORITHM, bits=next(iter(bits)))
    if trunk is not None:
        expected["trunk"] = trunk_record(trunk["bits"], trunk["modules"])
        if any(
            trunk.get(k) != v for k, v in expected["trunk"].items() if k != "modules"
        ):
            raise ValueError("invalid trunk_quantization format")
    # JSON comparison distinguishes exact integers, floats and booleans.
    if json.dumps(record, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError("manifest quantization disagrees with config and format")


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
        table_specs(doc)
        assets = doc.get("assets", {})
        required = {"config.json", INDEX, "tokenizer.json", "chat_template.jinja"}
        if not isinstance(assets, dict) or not required.issubset(assets):
            raise ValueError("missing authenticated model, index, or tokenizer assets")
        _validate_files(root, assets, digest=True)
        core = doc.get("core_shards", {})
        if not isinstance(core, dict) or any(
            not n.endswith(".safetensors") for n in core
        ):
            raise ValueError("unexpected core shard filename")
        _validate_files(root, core, digest=verify_core)
        cfg = _json(root / "config.json")
        if cfg.get("model_type") != "hy_v4_flash" or "vision_config" in cfg:
            raise ValueError("expected a flattened Flash text configuration")
        iq = cfg.get("omlx_oqh", {})
        targets = iq.get("tensors", {}) if isinstance(iq, dict) else {}
        if (
            not isinstance(targets, dict)
            or not targets
            or type(iq.get("version")) is not int
            or iq["version"] != 1
        ):
            raise ValueError("expected version-1 packed oQh tensors")
        index = _json(root / INDEX).get("weight_map")
        if (
            not isinstance(index, dict)
            or not index
            or not all(isinstance(shard, str) for shard in index.values())
            or set(index.values()) != core.keys()
        ):
            raise ValueError("weight index does not describe exactly the core shards")
        for prefix, spec in targets.items():
            if not valid_bank_spec(spec) or not all(
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
        _validate_quantization(doc, targets, cfg.get("trunk_quantization"))
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
