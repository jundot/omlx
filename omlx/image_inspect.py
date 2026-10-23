# SPDX-License-Identifier: Apache-2.0
"""Read-only structural inspection; a successful report never enables serving."""

import hashlib
import json
import os
import re
import stat
import struct
from pathlib import Path

from .image_bundle import MAX_METADATA, PIPELINE, _pairs, bundle_contract, read_json

RECIPE = "qwen-image-2.1-turbo-euler-8-v1"
COMPONENTS = ("text_encoder", "transformer", "vae")
DTYPES = {"F32": 4, "BF16": 2}


def contained(root, reference):
    if not isinstance(reference, str) or not reference:
        raise ValueError("path must be a nonempty relative string")
    path = Path(reference)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"invalid relative path: {reference}")
    resolved = (root / path).resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError(f"path escapes bundle: {reference}")
    return resolved


def signature(st):
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


def tensor_header(path, verify_hashes):
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"not a regular tensor file: {path.name}")
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated safetensors length")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= MAX_METADATA:
            raise ValueError("invalid safetensors header length")
        raw = stream.read(length)
        if len(raw) != length:
            raise ValueError("truncated safetensors header")
        header = json.loads(raw, object_pairs_hook=_pairs)
        if not isinstance(header, dict):
            raise ValueError("safetensors header must be an object")
        spans = []
        tensors = {}
        for name, item in header.items():
            if name == "__metadata__":
                if not isinstance(item, dict) or not all(
                    isinstance(k, str) and isinstance(v, str) for k, v in item.items()
                ):
                    raise ValueError("invalid safetensors metadata")
                continue
            if not isinstance(item, dict):
                raise ValueError(f"invalid tensor: {name}")
            shape, dtype, offsets = (
                item.get("shape"),
                item.get("dtype"),
                item.get("data_offsets"),
            )
            if (
                not isinstance(shape, list)
                or len(shape) > 8
                or not all(type(x) is int and 0 <= x <= 2**63 - 1 for x in shape)
            ):
                raise ValueError(f"invalid shape: {name}")
            if dtype not in DTYPES:
                raise ValueError(f"unsupported dtype: {name}")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(type(x) is int for x in offsets)
            ):
                raise ValueError(f"invalid offsets: {name}")
            start, end = offsets
            elements = 1
            for dimension in shape:
                elements *= dimension
                if elements > before.st_size:
                    raise ValueError(f"tensor shape exceeds file: {name}")
            if (
                start < 0
                or end < start
                or end > before.st_size - 8 - length
                or end - start != elements * DTYPES[dtype]
            ):
                raise ValueError(f"invalid tensor byte count: {name}")
            spans.append((start, end))
            tensors[name] = {"shape": shape, "dtype": dtype}
        cursor = 0
        for start, end in sorted(spans):
            if start != cursor:
                raise ValueError("overlapping or noncontiguous tensor data")
            cursor = end
        if cursor != before.st_size - 8 - length or not tensors:
            raise ValueError("empty tensors or unaccounted payload")
        digest = None
        if verify_hashes:
            stream.seek(0)
            hasher = hashlib.sha256()
            while chunk := stream.read(1024 * 1024):
                hasher.update(chunk)
            digest = hasher.hexdigest()
        if signature(before) != signature(os.fstat(stream.fileno())) or signature(
            before
        ) != signature(path.stat()):
            raise ValueError("tensor file changed during inspection")
    return tensors, before.st_size, digest, (before.st_dev, before.st_ino)


def inspect_bundle(path, verify_hashes=False):
    report = {
        "structural_status": "invalid",
        "runtime_status": "unvalidated",
        "numerical_status": "unvalidated",
        "serving_enabled": False,
        "components": {},
        "errors": [],
        "physical_weight_bytes": 0,
        "integrity_status": "not_checked",
        "integrity_scope": "weight_files",
    }
    try:
        root = Path(path).resolve(strict=True)
        observed = {}

        def metadata(path):
            before = signature(path.stat())
            result = read_json(path)
            if before != signature(path.stat()):
                raise ValueError(f"metadata changed during inspection: {path.name}")
            observed[path] = before
            return result

        manifest = bundle_contract(root, resolve=contained, metadata=metadata)
        if (
            type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != 1
        ):
            raise ValueError("unsupported schema_version")
        for key, expected in (
            ("pipeline", PIPELINE),
            ("layout", "native-bf16-f32-v1"),
            ("recipe", RECIPE),
        ):
            if manifest.get(key) != expected:
                raise ValueError(f"unsupported {key}")
        components = manifest.get("components")
        if not isinstance(components, dict) or set(components) != set(COMPONENTS):
            raise ValueError("expected text_encoder, transformer and vae components")
        hashes = manifest.get("sha256", {})
        if not isinstance(hashes, dict):
            raise ValueError("sha256 must be an object")
        if not all(
            isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v)
            for v in hashes.values()
        ):
            raise ValueError("sha256 values must be lowercase SHA-256 digests")
        seen_files = set()
        inspected_paths = set()
        all_hashed = True
        for name in COMPONENTS:
            component = components[name]
            if not isinstance(component, dict):
                raise ValueError(f"invalid component: {name}")
            directory = contained(root, component.get("path"))
            if not directory.is_dir():
                raise ValueError(f"not a component directory: {name}")
            config = metadata(
                contained(root, str(directory.relative_to(root) / "config.json"))
            )
            index_path = directory / "model.safetensors.index.json"
            index = None
            if index_path.exists():
                index = metadata(
                    contained(root, str(index_path.relative_to(root)))
                ).get("weight_map")
                if (
                    not isinstance(index, dict)
                    or not index
                    or not all(isinstance(v, str) for v in index.values())
                ):
                    raise ValueError(f"invalid weight_map: {name}")
                shard_names = sorted(set(index.values()))
            else:
                shard_names = ["model.safetensors"]
            auxiliary = component.get("auxiliary", [])
            if not isinstance(auxiliary, list) or not all(
                isinstance(v, str) for v in auxiliary
            ):
                raise ValueError("auxiliary must be a list of paths")
            tensors = {}
            for shard in shard_names + auxiliary:
                file = contained(root, str(directory.relative_to(root) / shard))
                if not file.is_relative_to(directory):
                    raise ValueError("shard escapes component")
                before = signature(file.stat())
                values, size, digest, identity = tensor_header(file, verify_hashes)
                observed[file] = before
                inspected_paths.add(str(file.relative_to(root)))
                if identity not in seen_files:
                    report["physical_weight_bytes"] += size
                    seen_files.add(identity)
                for tensor in values:
                    if tensor in tensors:
                        raise ValueError(f"duplicate tensor: {name}/{tensor}")
                    if (
                        index is not None
                        and shard in shard_names
                        and index.get(tensor) != shard
                    ):
                        raise ValueError(f"index mismatch: {name}/{tensor}")
                tensors.update(values)
                if verify_hashes:
                    expected = hashes.get(str(file.relative_to(root)))
                    if expected is None:
                        all_hashed = False
                    elif expected != digest:
                        raise ValueError(f"checksum mismatch: {file.relative_to(root)}")
            if index is not None and not set(index).issubset(tensors):
                raise ValueError(f"index references missing tensors: {name}")
            report["components"][name] = {
                "tensor_count": len(tensors),
                "config_class": config.get("_class_name"),
                "dtypes": sorted({v["dtype"] for v in tensors.values()}),
            }
        # The tokenizer is part of the artifact even though inspection runs no code.
        processor = contained(root, manifest.get("processor"))
        for filename in ("tokenizer.json", "tokenizer_config.json"):
            metadata(contained(root, str(processor.relative_to(root) / filename)))
        if set(hashes) - inspected_paths:
            raise ValueError("sha256 references an uninspected weight file")
        for file, before in observed.items():
            if signature(file.stat()) != before:
                raise ValueError(
                    f"bundle changed during inspection: {file.relative_to(root)}"
                )
        report["structural_status"] = "valid"
        report["compatibility_status"] = "unvalidated_tensor_semantics"
        if verify_hashes:
            report["integrity_status"] = (
                "matched_manifest" if all_hashed else "missing_reference_hashes"
            )
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        OverflowError,
        RecursionError,
        RuntimeError,
    ) as exc:
        report["errors"].append({"code": "invalid_bundle", "detail": str(exc)})
    return report


def inspect_command(args):
    try:
        report = inspect_bundle(args.path, args.verify_hashes)
    except KeyboardInterrupt:
        print(json.dumps({"structural_status": "cancelled", "serving_enabled": False}))
        return 130
    print(
        json.dumps(report, indent=2)
        if args.json
        else f"Structure: {report['structural_status']}; runtime: unvalidated\n"
        + "\n".join(error["detail"] for error in report["errors"])
    )
    return (
        0
        if report["structural_status"] == "valid"
        and (not args.verify_hashes or report["integrity_status"] == "matched_manifest")
        else 1
    )
