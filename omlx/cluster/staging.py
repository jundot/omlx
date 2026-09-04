# SPDX-License-Identifier: Apache-2.0
"""Put the right weight files on the right Mac, and nothing else.

A pipeline rank loads only its own layers, but the safetensors files holding
those layers still have to exist on that machine's disk. Copying the whole model
to every node wastes most of the transfer: for a 78-layer model split 56/22, the
node holding the tail needs 24 of 76 shards — 96 GiB instead of 309.

This module works out that mapping from the model itself and stages only what
each node is missing. It is the difference between "clustering needs a 300 GiB
copy to each Mac" and "clustering needs the shards you don't already have".
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shlex
import struct
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .ssh_policy import cluster_ssh_options

_LAYER = re.compile(r"(?:^|\.)(?:layers|h|blocks|block)\.(\d+)(?:\.|$)")
_MAX_HEADER_BYTES = 64 * 1024 * 1024
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}

# Where a peer's oMLX checkout keeps its interpreter. Unquoted on the remote
# command line so the peer's shell expands ``~`` to its own home.
DEFAULT_REMOTE_PYTHON = "~/omlx-distributed/.venv/bin/python"


def is_local_host(host: str) -> bool:
    """Whether this address names this machine, so ssh would be a detour."""

    return host.strip() in _LOCAL_HOSTS


@dataclass(frozen=True)
class ShardInfo:
    """One weight file: which layers it holds, and whether it holds shared ones."""

    name: str
    size_bytes: int
    layers: frozenset[int]
    has_shared_tensors: bool  # embeddings, lm_head, norms — not layer-scoped


@dataclass(frozen=True)
class StagingPlan:
    """What one node must receive before it can serve its stage."""

    node_id: str
    start_layer: int
    end_layer: int
    required: tuple[str, ...]
    missing: tuple[str, ...]
    required_bytes: int
    missing_bytes: int
    total_model_bytes: int

    @property
    def saved_bytes(self) -> int:
        """Bytes avoided versus copying the whole model."""

        return self.total_model_bytes - self.required_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "start_layer": self.start_layer,
            "end_layer": self.end_layer,
            "required_files": len(self.required),
            "missing_files": len(self.missing),
            "required_bytes": self.required_bytes,
            "missing_bytes": self.missing_bytes,
            "total_model_bytes": self.total_model_bytes,
            "saved_bytes": self.saved_bytes,
            "ready": not self.missing,
        }


def _safetensors_tensor_names(path: Path) -> list[str]:
    """Tensor names from a safetensors header, without reading the payload."""

    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path.name}: truncated safetensors header")
        length = struct.unpack("<Q", raw)[0]
        if not 0 < length <= _MAX_HEADER_BYTES:
            raise ValueError(f"{path.name}: implausible header length {length}")
        header = json.loads(stream.read(length))
    return [name for name in header if name != "__metadata__"]


def index_shards(model_path: str | Path) -> tuple[ShardInfo, ...]:
    """Describe every weight file in a model directory.

    Uses ``model.safetensors.index.json`` when present, because reading one file
    beats opening dozens. Falls back to scanning each header — mixed-precision
    exports frequently ship without an index.
    """

    root = Path(model_path).expanduser()
    if not root.is_dir():
        raise ValueError(f"model path is not a directory: {root}")
    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise ValueError(f"no safetensors files in {root}")

    names_by_file: dict[str, list[str]] = {}
    index = root / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text()).get("weight_map", {})
        if isinstance(weight_map, dict):
            for tensor, filename in weight_map.items():
                names_by_file.setdefault(str(filename), []).append(str(tensor))

    shards = []
    for path in files:
        names = names_by_file.get(path.name)
        if names is None:
            names = _safetensors_tensor_names(path)
        layers = set()
        shared = False
        for name in names:
            match = _LAYER.search(name)
            if match:
                layers.add(int(match.group(1)))
            else:
                shared = True
        shards.append(
            ShardInfo(
                name=path.name,
                size_bytes=path.stat().st_size,
                layers=frozenset(layers),
                has_shared_tensors=shared,
            )
        )
    return tuple(shards)


def shards_for_stage(
    shards: Sequence[ShardInfo],
    start_layer: int,
    end_layer: int,
) -> tuple[ShardInfo, ...]:
    """Files a rank needs to serve layers ``[start_layer, end_layer)``.

    Shared tensors (embeddings, lm_head, final norm) are always included: which
    rank touches them is a model-specific detail, and they are small next to the
    layer weights, so shipping them everywhere is the cheap, safe choice.
    """

    wanted = set(range(start_layer, end_layer))
    return tuple(
        shard
        for shard in shards
        if (shard.layers & wanted) or shard.has_shared_tensors
    )


# Files a model needs besides its weights. Tiny, so always staged.
_SIDECAR_GLOBS = (
    "*.json",
    "*.txt",
    "*.model",
    "tokenizer*",
    "*.jinja",
)


def sidecar_files(model_path: str | Path) -> tuple[str, ...]:
    """Config and tokenizer files that must accompany the weights."""

    root = Path(model_path).expanduser()
    names: set[str] = set()
    for pattern in _SIDECAR_GLOBS:
        for path in root.glob(pattern):
            if path.is_file() and not path.name.endswith(".safetensors"):
                names.add(path.name)
    return tuple(sorted(names))


def _model_identity_digest(model_path: str | Path) -> str:
    """Identity shared by every stage, without requiring every weight shard.

    Pipeline ranks intentionally hold different safetensors files, so a digest
    of the complete weight directory can never match after selective staging.
    The index, config, tokenizer and processor files are copied to every rank
    and describe which exact model those rank-local weights belong to.
    """

    root = Path(model_path).expanduser()
    hasher = hashlib.sha256()
    for name in sidecar_files(root):
        path = root / name
        encoded = name.encode()
        payload = path.read_bytes()
        hasher.update(struct.pack("<Q", len(encoded)))
        hasher.update(encoded)
        hasher.update(struct.pack("<Q", len(payload)))
        hasher.update(payload)
    return hasher.hexdigest()


def _indexed_shards(model_path: str | Path) -> tuple[ShardInfo, ...] | None:
    """Read the full shard map from the index even on a partially staged rank."""

    root = Path(model_path).expanduser()
    index = root / "model.safetensors.index.json"
    if not index.is_file():
        return None
    payload = json.loads(index.read_text())
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{index.name} does not contain a weight map")

    names_by_file: dict[str, list[str]] = {}
    for tensor, raw_filename in weight_map.items():
        filename = str(raw_filename)
        if Path(filename).name != filename:
            raise ValueError(f"unsafe safetensors filename in index: {filename!r}")
        names_by_file.setdefault(filename, []).append(str(tensor))

    shards: list[ShardInfo] = []
    for filename, names in sorted(names_by_file.items()):
        path = root / filename
        layers: set[int] = set()
        shared = False
        for name in names:
            match = _LAYER.search(name)
            if match:
                layers.add(int(match.group(1)))
            else:
                shared = True
        shards.append(
            ShardInfo(
                name=filename,
                size_bytes=path.stat().st_size if path.is_file() else 0,
                layers=frozenset(layers),
                has_shared_tensors=shared,
            )
        )
    return tuple(shards)


def _validate_safetensors_payload(path: Path) -> None:
    """Reject a missing/truncated staged shard without loading its tensors."""

    with path.open("rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError("truncated header length")
        header_length = struct.unpack("<Q", raw_length)[0]
        if not 0 < header_length <= _MAX_HEADER_BYTES:
            raise ValueError(f"implausible header length {header_length}")
        raw_header = stream.read(header_length)
        if len(raw_header) != header_length:
            raise ValueError("truncated header")
    header = json.loads(raw_header)
    if not isinstance(header, dict):
        raise ValueError("header is not an object")
    payload_bytes = path.stat().st_size - 8 - header_length
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        offsets = metadata.get("data_offsets") if isinstance(metadata, dict) else None
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(value, int) for value in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
            or offsets[1] > payload_bytes
        ):
            raise ValueError(f"{name} has invalid data offsets")


def validate_staged_model(
    model_path: str | Path,
    start_layer: int,
    end_layer: int,
) -> dict[str, Any]:
    """Prove one rank has its assigned stage, not an unnecessary full model.

    The model index is staged to every rank, so indexed models retain the full
    layer→filename map even when most weight files are intentionally absent.
    Exports without an index fall back to the headers that are present and must
    still cover every assigned layer.
    """

    root = Path(model_path).expanduser()
    if not root.is_dir():
        raise ValueError(f"model path is not a directory: {root}")
    if start_layer < 0 or end_layer <= start_layer:
        raise ValueError("stage layer range is invalid")

    indexed = _indexed_shards(root)
    shards = indexed if indexed is not None else index_shards(root)
    required = shards_for_stage(shards, start_layer, end_layer)
    missing = tuple(shard.name for shard in required if not (root / shard.name).is_file())
    corrupt: list[str] = []
    for shard in required:
        path = root / shard.name
        if not path.is_file():
            continue
        try:
            _validate_safetensors_payload(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            corrupt.append(f"{shard.name}: {exc}")

    covered_layers = set().union(*(shard.layers for shard in required))
    missing_layers = sorted(set(range(start_layer, end_layer)) - covered_layers)
    ready = not missing and not corrupt and not missing_layers
    return {
        "model_identity": _model_identity_digest(root),
        "stage_ready": ready,
        "required_files": [shard.name for shard in required],
        "missing_files": list(missing),
        "corrupt_files": corrupt,
        "missing_layers": missing_layers,
    }


def model_staging_inventory(model_path: str | Path) -> dict[str, Any]:
    """Serializable shard/layer map and sidecar sizes for one complete model."""

    root = Path(model_path).expanduser()
    shards = index_shards(root)
    sidecars = sidecar_files(root)
    return {
        "shards": [
            {
                "name": shard.name,
                "size_bytes": shard.size_bytes,
                "layers": sorted(shard.layers),
                "has_shared_tensors": shard.has_shared_tensors,
            }
            for shard in shards
        ],
        "sidecars": {
            name: (root / name).stat().st_size
            for name in sidecars
        },
    }


_REMOTE_STAGING_INVENTORY_SNIPPET = (
    "import json,sys;"
    "from omlx.cluster.staging import model_staging_inventory;"
    "print(json.dumps(model_staging_inventory(sys.argv[1])))"
)


def remote_model_staging_inventory(
    ssh_target: str,
    model_path: str,
    *,
    python_executable: str = DEFAULT_REMOTE_PYTHON,
    timeout: float = 600.0,
) -> tuple[tuple[ShardInfo, ...], dict[str, int]]:
    """Read a complete source model's shard map on the Mac that owns it."""

    payload = run_remote_python(
        ssh_target,
        _REMOTE_STAGING_INVENTORY_SNIPPET,
        model_path,
        description="index the model shards",
        python_executable=python_executable,
        timeout=timeout,
    )
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid model shard inventory from {ssh_target}")
    raw_shards = payload.get("shards")
    raw_sidecars = payload.get("sidecars")
    if not isinstance(raw_shards, list) or not isinstance(raw_sidecars, dict):
        raise RuntimeError(f"incomplete model shard inventory from {ssh_target}")

    shards: list[ShardInfo] = []
    for item in raw_shards:
        if not isinstance(item, dict):
            raise RuntimeError(f"invalid shard entry from {ssh_target}")
        try:
            name = str(item["name"])
            size_bytes = int(item["size_bytes"])
            layers = frozenset(int(layer) for layer in item["layers"])
            shared = item["has_shared_tensors"] is True
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid shard entry from {ssh_target}") from exc
        if Path(name).name != name or size_bytes < 0 or min(layers, default=0) < 0:
            raise RuntimeError(f"unsafe shard entry from {ssh_target}")
        shards.append(
            ShardInfo(
                name=name,
                size_bytes=size_bytes,
                layers=layers,
                has_shared_tensors=shared,
            )
        )

    sidecars: dict[str, int] = {}
    for name, size in raw_sidecars.items():
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
        ):
            raise RuntimeError(f"unsafe sidecar entry from {ssh_target}")
        sidecars[name] = size
    return tuple(shards), sidecars


def plan_staging(
    model_path: str | Path,
    *,
    node_id: str,
    start_layer: int,
    end_layer: int,
    present: dict[str, int] | None = None,
    shards: Sequence[ShardInfo] | None = None,
) -> StagingPlan:
    """Work out what one node still needs.

    ``present`` maps filename to size for what the node already has; a file is
    considered staged only when the size matches, so a truncated transfer is
    re-sent rather than silently trusted.
    """

    shards = tuple(shards) if shards is not None else index_shards(model_path)
    required = shards_for_stage(shards, start_layer, end_layer)
    present = present or {}
    missing = tuple(
        shard for shard in required if present.get(shard.name) != shard.size_bytes
    )
    return StagingPlan(
        node_id=node_id,
        start_layer=start_layer,
        end_layer=end_layer,
        required=tuple(shard.name for shard in required),
        missing=tuple(shard.name for shard in missing),
        required_bytes=sum(shard.size_bytes for shard in required),
        missing_bytes=sum(shard.size_bytes for shard in missing),
        total_model_bytes=sum(shard.size_bytes for shard in shards),
    )


def plan_cluster_staging(
    model_path: str | Path,
    assignments: Sequence[Any],
    *,
    present_by_node: dict[str, dict[str, int]] | None = None,
) -> tuple[StagingPlan, ...]:
    """One staging plan per rank, from the shard plan the planner produced."""

    shards = index_shards(model_path)
    present_by_node = present_by_node or {}
    return tuple(
        plan_staging(
            model_path,
            node_id=assignment.node_id,
            start_layer=assignment.start_layer,
            end_layer=assignment.end_layer,
            present=present_by_node.get(assignment.node_id),
            shards=shards,
        )
        for assignment in assignments
    )


def stage_manifest(
    model_path: str | Path,
    assignments: Sequence[Any],
    hosts_by_node: dict[str, str],
    *,
    source_host: str = "127.0.0.1",
    source_python_executable: str = DEFAULT_REMOTE_PYTHON,
) -> dict[str, Any]:
    """What must move before this plan can run, per node.

    Combines the layer→shard map with what each peer already holds, so the
    answer is "these 24 files, 96 GiB" rather than "copy the model". Sidecars
    (config, tokenizer) are listed separately because they are tiny and always
    required.
    """

    # The worker receives ``model_path`` verbatim on every rank, so readiness
    # must inspect that same path on every host. The previous hard-coded
    # ~/.omlx/models/<basename> probe marked an already-present model missing
    # whenever the catalogue used another configured directory. It also
    # skipped the local host entirely, which made local weights look absent.
    remote_dir = str(Path(model_path).expanduser())
    source_is_local = is_local_host(source_host)
    if source_is_local:
        source_root = Path(remote_dir)
        shards = index_shards(source_root)
        sidecar_sizes = {
            name: (source_root / name).stat().st_size
            for name in sidecar_files(source_root)
        }
    else:
        # Send the ~-form: the peer's inventory snippet expanduser()s it in
        # its own home, so a cross-user source reports its real shard map
        # instead of an empty directory at the coordinator's absolute path.
        shards, sidecar_sizes = remote_model_staging_inventory(
            source_host,
            home_relative_model_path(str(model_path)),
            python_executable=source_python_executable,
        )
    # A peer with a different macOS account has a different $HOME, so the
    # coordinator-absolute path names nothing there. The worker now receives the
    # ~-form and resolves it per-node (see home_relative_model_path / launch),
    # so readiness must probe each peer in its OWN home too — otherwise an
    # already-present model reads as entirely missing and a full re-copy is
    # (needlessly, and here fatally) proposed.
    portable = home_relative_model_path(str(model_path))
    present_by_node = {}
    for assignment in assignments:
        ssh_target = hosts_by_node.get(assignment.node_id)
        if not ssh_target:
            continue
        if is_local_host(ssh_target):
            present_by_node[assignment.node_id] = {
                path.name: path.stat().st_size
                for path in Path(remote_dir).iterdir()
                if path.is_file()
            }
        else:
            peer_dir = remote_model_dir(ssh_target, portable)
            present_by_node[assignment.node_id] = remote_file_sizes(
                ssh_target, peer_dir
            )

    plans = (
        plan_cluster_staging(
            remote_dir,
            assignments,
            present_by_node=present_by_node,
        )
        if source_is_local
        else tuple(
            plan_staging(
                remote_dir,
                node_id=assignment.node_id,
                start_layer=assignment.start_layer,
                end_layer=assignment.end_layer,
                present=present_by_node.get(assignment.node_id),
                shards=shards,
            )
            for assignment in assignments
        )
    )
    sidecars = tuple(sorted(sidecar_sizes))
    nodes = []
    total_missing_bytes = 0
    for plan in plans:
        present = present_by_node.get(plan.node_id, {})
        missing_sidecars = tuple(
            name
            for name, size in sidecar_sizes.items()
            if present.get(name) != size
        )
        missing_sidecar_bytes = sum(sidecar_sizes[name] for name in missing_sidecars)
        total_missing_bytes += plan.missing_bytes + missing_sidecar_bytes
        nodes.append(
            plan.to_dict()
            | {
                "sidecar_files": len(sidecars),
                "missing_sidecars": len(missing_sidecars),
                "missing_sidecar_bytes": missing_sidecar_bytes,
                "ready": not plan.missing and not missing_sidecars,
            }
        )
    return {
        "sidecars": list(sidecars),
        "source_host": (
            "127.0.0.1" if source_is_local else source_host
        ),
        "nodes": nodes,
        "total_missing_bytes": total_missing_bytes,
        "ready": all(node["ready"] for node in nodes),
    }


# AES-GCM is hardware-accelerated on Apple Silicon and measured ~2.5x faster
# than the default cipher on a Thunderbolt link (0.60 vs 0.24 GB/s). Transfers
# run several files at once because scp is bound by per-stream cipher CPU, not
# by the wire.
_FAST_CIPHER = "aes128-gcm@openssh.com"
_DEFAULT_PARALLEL = 4
_REMOTE_INSTALL_SNIPPET = (
    "import os,sys;"
    "temporary,final=sys.argv[1:3];"
    "expected=int(sys.argv[3]);"
    "actual=os.path.getsize(temporary);"
    "\nif expected >= 0 and actual != expected:"
    "\n raise SystemExit(f'transferred size {actual} != expected {expected}')"
    "\nos.replace(temporary,final)"
)
_REMOTE_DISCARD_SNIPPET = (
    "import os,sys;"
    "\ntry: os.unlink(sys.argv[1])"
    "\nexcept FileNotFoundError: pass"
)


def _staging_partial_name() -> str:
    """A hidden, collision-resistant name that is never a runnable model file."""

    return f".omlx-stage-{secrets.token_hex(12)}.part"


def _finish_remote_staged_file(
    host: str,
    temporary_path: str,
    final_path: str,
    *,
    expected_size: int | None = None,
) -> None:
    """Validate and atomically publish one completed remote transfer."""

    command = " ".join(
        (
            "/usr/bin/python3",
            "-c",
            shlex.quote(_REMOTE_INSTALL_SNIPPET),
            shlex.quote(temporary_path),
            shlex.quote(final_path),
            str(expected_size if expected_size is not None else -1),
        )
    )
    result = subprocess.run(
        ["ssh", *cluster_ssh_options(connect_timeout=10), host, command],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"could not install staged file on {host}: "
            f"{result.stderr.strip()[:200]}"
        )


def _discard_remote_staged_file(host: str, temporary_path: str) -> None:
    """Best-effort cleanup of one exact hidden staging file."""

    subprocess.run(
        [
            "ssh",
            *cluster_ssh_options(connect_timeout=10),
            host,
            " ".join(
                (
                    "/usr/bin/python3",
                    "-c",
                    shlex.quote(_REMOTE_DISCARD_SNIPPET),
                    shlex.quote(temporary_path),
                )
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


@dataclass
class StagingResult:
    """Outcome of moving one node's missing files into place."""

    node_id: str
    copied: tuple[str, ...] = ()
    failed: tuple[str, ...] = ()
    bytes_copied: int = 0
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.failed

    @property
    def gigabytes_per_second(self) -> float:
        return self.bytes_copied / self.seconds / 1e9 if self.seconds > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "copied": len(self.copied),
            "failed": list(self.failed),
            "bytes_copied": self.bytes_copied,
            "seconds": round(self.seconds, 2),
            "gigabytes_per_second": round(self.gigabytes_per_second, 2),
            "ok": self.ok,
        }


def scp_push(
    *,
    destination_host: str,
    source_dir: str,
    destination_dir: str,
    filename: str,
    cipher: str = _FAST_CIPHER,
    timeout: float = 3600.0,
) -> None:
    """Push one local model file to the Mac that needs it."""

    if Path(filename).name != filename:
        raise ValueError(f"staging filename must be a basename: {filename!r}")
    source = Path(source_dir).expanduser() / filename
    if not source.is_file():
        raise RuntimeError(f"source model file is missing: {source}")
    remote_dir = shlex.quote(str(Path(destination_dir).expanduser()))
    mkdir = subprocess.run(
        [
            "ssh",
            *cluster_ssh_options(connect_timeout=10),
            destination_host,
            f"mkdir -p {remote_dir}",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if mkdir.returncode != 0:
        raise RuntimeError(
            f"could not create model directory on {destination_host}: "
            f"{mkdir.stderr.strip()[:200]}"
        )
    temporary_path = (
        f"{str(Path(destination_dir).expanduser()).rstrip('/')}/"
        f"{_staging_partial_name()}"
    )
    final_path = (
        f"{str(Path(destination_dir).expanduser()).rstrip('/')}/{filename}"
    )
    installed = False
    try:
        result = subprocess.run(
            [
                "scp",
                "-q",
                *cluster_ssh_options(connect_timeout=10),
                "-c",
                cipher,
                str(source),
                f"{destination_host}:{shlex.quote(temporary_path)}",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"scp of {filename} to {destination_host} failed: "
                f"{result.stderr.strip()[:200]}"
            )
        _finish_remote_staged_file(
            destination_host,
            temporary_path,
            final_path,
            expected_size=source.stat().st_size,
        )
        installed = True
    finally:
        if not installed:
            _discard_remote_staged_file(destination_host, temporary_path)


def _local_file_sizes(model_dir: str | Path) -> dict[str, int]:
    root = Path(model_dir).expanduser()
    if not root.is_dir():
        return {}
    return {
        path.name: path.stat().st_size
        for path in root.iterdir()
        if path.is_file()
    }


def scp_copy(
    *,
    source_host: str,
    destination_host: str,
    source_dir: str,
    destination_dir: str,
    filename: str,
    cipher: str = _FAST_CIPHER,
    timeout: float = 3600.0,
) -> None:
    """Copy one model file between any two enrolled cluster nodes."""

    if Path(filename).name != filename:
        raise ValueError(f"staging filename must be a basename: {filename!r}")
    source_local = is_local_host(source_host)
    destination_local = is_local_host(destination_host)
    if source_local and destination_local:
        source = Path(source_dir).expanduser() / filename
        destination = Path(destination_dir).expanduser() / filename
        if source.resolve() != destination.resolve():
            raise RuntimeError("local-to-local cluster staging is not supported")
        if not source.is_file():
            raise RuntimeError(f"source model file is missing: {source}")
        return
    if source_local:
        scp_push(
            destination_host=destination_host,
            source_dir=source_dir,
            destination_dir=destination_dir,
            filename=filename,
            cipher=cipher,
            timeout=timeout,
        )
        return

    remote_source = shlex.quote(f"{source_dir.rstrip('/')}/{filename}")
    if destination_local:
        destination = Path(destination_dir).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        temporary = destination / _staging_partial_name()
        try:
            result = subprocess.run(
                [
                    "scp",
                    "-q",
                    *cluster_ssh_options(connect_timeout=10),
                    "-c",
                    cipher,
                    f"{source_host}:{remote_source}",
                    str(temporary),
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
            if result.returncode == 0:
                os.replace(temporary, destination / filename)
        finally:
            if temporary.exists():
                temporary.unlink()
    else:
        remote_destination = shlex.quote(destination_dir)
        mkdir = subprocess.run(
            [
                "ssh",
                *cluster_ssh_options(connect_timeout=10),
                destination_host,
                f"mkdir -p {remote_destination}",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        if mkdir.returncode != 0:
            raise RuntimeError(
                f"could not create model directory on {destination_host}: "
                f"{mkdir.stderr.strip()[:200]}"
            )
        temporary_path = (
            f"{destination_dir.rstrip('/')}/{_staging_partial_name()}"
        )
        final_path = f"{destination_dir.rstrip('/')}/{filename}"
        installed = False
        try:
            # -3 routes peer-to-peer copies through the coordinator so neither
            # peer needs SSH credentials for the other.
            result = subprocess.run(
                [
                    "scp",
                    "-3",
                    "-q",
                    *cluster_ssh_options(connect_timeout=10),
                    "-c",
                    cipher,
                    f"{source_host}:{remote_source}",
                    f"{destination_host}:{shlex.quote(temporary_path)}",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
            if result.returncode == 0:
                _finish_remote_staged_file(
                    destination_host,
                    temporary_path,
                    final_path,
                )
                installed = True
        finally:
            if not installed:
                _discard_remote_staged_file(destination_host, temporary_path)
    if result.returncode != 0:
        raise RuntimeError(
            f"scp of {filename} from {source_host} to {destination_host} failed: "
            f"{result.stderr.strip()[:200]}"
        )


def stage_files_from_source(
    plan: StagingPlan,
    *,
    model_path: str | Path,
    source_host: str,
    destination_host: str,
    expected_sizes: dict[str, int],
    destination_dir: str | None = None,
    parallel: int = _DEFAULT_PARALLEL,
    transfer: Any = scp_copy,
    progress: Callable[[str, str, int], None] | None = None,
    clock: Any = None,
) -> StagingResult:
    """Stage one rank from a local or peer model holder and verify every file.

    ``destination_dir`` is where files land on the destination Mac; it defaults
    to the coordinator's own absolute source path, correct only when every Mac
    shares the same $HOME. On a cross-user cluster the caller passes the path
    resolved in the destination's own home so the present-file probe and the
    scp destination address the peer's real directory.
    """

    import time
    from concurrent.futures import ThreadPoolExecutor

    # model_path is the coordinator's own absolute form. On a remote source
    # whose macOS account differs it names nothing, so resolve the ~-form in
    # the source peer's OWN home before using it as the scp source path —
    # the same treatment the destination side already gets from the caller.
    source_dir = (
        str(Path(model_path).expanduser())
        if is_local_host(source_host)
        else remote_model_dir(
            source_host, home_relative_model_path(str(model_path))
        )
    )
    destination_dir = destination_dir or source_dir
    expected = dict(expected_sizes)
    # The caller supplies only this rank's required shards plus common
    # sidecars. Validate that contract here before any disk or network action.
    if any(
        Path(name).name != name
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size < 0
        for name, size in expected.items()
    ):
        raise RuntimeError("staging source returned an unsafe file inventory")
    if is_local_host(source_host) and not is_local_host(destination_host):
        common = tuple(
            name for name in expected if name not in set(plan.required)
        )
        return stage_remote_files(
            plan,
            model_path=model_path,
            destination_host=destination_host,
            destination_dir=destination_dir,
            sidecars=common,
            parallel=parallel,
            progress=progress,
            clock=clock,
        )

    present = (
        _local_file_sizes(destination_dir)
        if is_local_host(destination_host)
        else remote_file_sizes(destination_host, destination_dir)
    )
    missing = tuple(
        name for name, size in expected.items() if present.get(name) != size
    )
    missing_bytes = sum(expected[name] for name in missing)
    disk_plan = replace(plan, missing=missing, missing_bytes=missing_bytes)
    check_disk_for_staging(
        disk_plan,
        destination_dir,
        ssh_target=None if is_local_host(destination_host) else destination_host,
    )
    if not missing:
        return StagingResult(node_id=plan.node_id)
    if source_host == destination_host:
        raise RuntimeError(
            f"{source_host} was selected as the complete model source but "
            "its required files do not match the source inventory"
        )

    clock = clock or time.perf_counter
    started = clock()
    transferred: list[str] = []
    failed: list[str] = []

    def copy(name: str) -> tuple[str, str | None]:
        if progress is not None:
            progress(name, "copying", 0)
        try:
            transfer(
                source_host=source_host,
                destination_host=destination_host,
                source_dir=source_dir,
                destination_dir=destination_dir,
                filename=name,
            )
            return name, None
        except Exception as exc:  # noqa: BLE001 - reported per file
            return name, str(exc)

    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        for name, error in pool.map(copy, missing):
            if error:
                failed.append(name)
                if progress is not None:
                    progress(name, "failed", 0)
            else:
                transferred.append(name)
                if progress is not None:
                    progress(name, "copied", expected[name])

    landed = (
        _local_file_sizes(destination_dir)
        if is_local_host(destination_host)
        else remote_file_sizes(destination_host, destination_dir)
    )
    verified: list[str] = []
    for name in transferred:
        if landed.get(name) != expected[name]:
            failed.append(name)
            if progress is not None:
                progress(name, "failed", 0)
        else:
            verified.append(name)

    return StagingResult(
        node_id=plan.node_id,
        copied=tuple(verified),
        failed=tuple(sorted(set(failed))),
        bytes_copied=sum(expected[name] for name in verified),
        seconds=clock() - started,
    )


def run_remote_python(
    ssh_target: str,
    snippet: str,
    argument: str,
    *,
    description: str,
    python_executable: str = DEFAULT_REMOTE_PYTHON,
    timeout: float = 600.0,
) -> Any:
    """Run one line of oMLX on a peer and return the JSON it printed.

    The peer answers questions only it can answer — which files it holds, how
    the model it holds is shaped — by running the same code this node would
    run. Both the snippet and its argument are shell-quoted: ``model_dir``
    reaches here from an API request, and an unquoted one is a command the
    peer's shell would happily run. Paths keep their ``~`` because the oMLX
    side expands it in Python.
    """

    executable = str(python_executable).strip()
    if executable == DEFAULT_REMOTE_PYTHON:
        # This one fixed internal default intentionally keeps ``~`` for the
        # peer shell. All discovered/user-carried paths must be absolute and
        # are quoted as a single word below.
        executable_word = executable
    else:
        path = Path(executable)
        if not path.is_absolute() or "\x00" in executable or len(executable) > 4096:
            raise ValueError("remote Python executable must be an absolute path")
        executable_word = shlex.quote(executable)
    result = subprocess.run(
        [
            "ssh",
            *cluster_ssh_options(connect_timeout=10),
            ssh_target,
            f"{executable_word} -c {shlex.quote(snippet)} {shlex.quote(argument)}",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"could not {description} on {ssh_target}: {result.stderr.strip()[:200]}"
        )
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise RuntimeError(
            f"unexpected output from {ssh_target}: {result.stdout[:200]}"
        ) from exc


_REMOTE_FILE_SIZES_SNIPPET = (
    "import json,sys;"
    "from pathlib import Path;"
    "p=Path(sys.argv[1]).expanduser();"
    "files=p.iterdir() if p.is_dir() else ();"
    "print(json.dumps({f.name:f.stat().st_size for f in files if f.is_file()}))"
)


def remote_file_sizes(
    ssh_target: str,
    model_dir: str,
    *,
    timeout: float = 120.0,
) -> dict[str, int]:
    """Immediate model-directory files and sizes on one destination Mac."""

    payload = run_remote_python(
        ssh_target,
        _REMOTE_FILE_SIZES_SNIPPET,
        model_dir,
        description="inspect staged model files",
        python_executable="/usr/bin/python3",
        timeout=timeout,
    )
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid model file listing from {ssh_target}")
    return {
        str(name): int(size)
        for name, size in payload.items()
        if isinstance(name, str)
        and isinstance(size, int)
        and not isinstance(size, bool)
        and size >= 0
    }


_REMOTE_EXPAND_SNIPPET = (
    "import json,sys;from pathlib import Path;"
    "print(json.dumps(str(Path(sys.argv[1]).expanduser())))"
)


def home_relative_model_path(model: str) -> str:
    """Re-express a coordinator-absolute model path in portable ~-form.

    A locally-sourced deployment resolves the model to the coordinator's own
    absolute path. Sent verbatim to a peer with a different macOS account that
    path names nothing, so callers send this ~-form for the peer to expand in
    its own home. A path outside the local home is returned unchanged.
    """

    expanded = Path(model).expanduser()
    try:
        return "~/" + str(expanded.relative_to(Path.home()))
    except ValueError:
        return str(expanded)


def remote_model_dir(
    ssh_target: str,
    model_dir: str,
    *,
    timeout: float = 60.0,
) -> str:
    """Expand a ``~``-form model path in the peer's OWN home.

    A cross-user cluster has a different ``$HOME`` per Mac, so the coordinator's
    absolute path names nothing on the peer. Resolving the ``~``-form on the
    peer yields the concrete directory its copy actually lives in, which the
    present-file probe and the scp destination both need.
    """

    path = run_remote_python(
        ssh_target,
        _REMOTE_EXPAND_SNIPPET,
        model_dir,
        description="resolve the model directory",
        python_executable="/usr/bin/python3",
        timeout=timeout,
    )
    if not isinstance(path, str) or not path:
        raise RuntimeError(f"invalid model directory from {ssh_target}")
    return path


def stage_remote_files(
    plan: StagingPlan,
    *,
    model_path: str | Path,
    destination_host: str,
    destination_dir: str | None = None,
    sidecars: Sequence[str] = (),
    parallel: int = _DEFAULT_PARALLEL,
    transfer: Any = scp_push,
    present_reader: Callable[[str, str], dict[str, int]] = remote_file_sizes,
    progress: Callable[[str, str, int], None] | None = None,
    clock: Any = None,
) -> StagingResult:
    """Push and verify every file one remote rank needs.

    The coordinator owns the source model and the job state. That makes the
    direction unambiguous (local source → explicit destination), supports any
    number of nodes, and lets the GUI show file-level progress.

    ``destination_dir`` is where the files land on the peer. It defaults to the
    coordinator's own absolute source path, which is only correct when every
    Mac shares the same $HOME. On a cross-user cluster the caller must pass the
    directory resolved in the peer's own home, or the present-file probe reads
    an empty directory and re-copies everything.
    """

    import time
    from concurrent.futures import ThreadPoolExecutor

    source = Path(model_path).expanduser()
    destination_dir = destination_dir or str(source)
    expected = {
        path.name: path.stat().st_size
        for path in source.iterdir()
        if path.is_file() and (path.name in plan.required or path.name in sidecars)
    }
    present = present_reader(destination_host, destination_dir)
    missing = tuple(
        name for name, size in expected.items() if present.get(name) != size
    )
    missing_bytes = sum(expected[name] for name in missing)
    disk_plan = replace(
        plan,
        missing=missing,
        missing_bytes=missing_bytes,
    )
    check_disk_for_staging(
        disk_plan,
        destination_dir,
        ssh_target=destination_host,
    )
    if not missing:
        return StagingResult(node_id=plan.node_id)

    clock = clock or time.perf_counter
    started = clock()
    transferred: list[str] = []
    failed: list[str] = []

    def push(name: str) -> tuple[str, str | None]:
        if progress is not None:
            progress(name, "copying", 0)
        try:
            transfer(
                destination_host=destination_host,
                source_dir=str(source),
                destination_dir=destination_dir,
                filename=name,
            )
            return name, None
        except Exception as exc:  # noqa: BLE001 - reported per file
            return name, str(exc)

    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        for name, error in pool.map(push, missing):
            if error:
                failed.append(name)
                if progress is not None:
                    progress(name, "failed", 0)
            else:
                transferred.append(name)
                if progress is not None:
                    progress(name, "copied", expected[name])

    landed = present_reader(destination_host, destination_dir)
    verified: list[str] = []
    for name in transferred:
        if landed.get(name) != expected[name]:
            failed.append(name)
            if progress is not None:
                progress(name, "failed", 0)
        else:
            verified.append(name)

    return StagingResult(
        node_id=plan.node_id,
        copied=tuple(verified),
        failed=tuple(sorted(set(failed))),
        bytes_copied=sum(expected[name] for name in verified),
        seconds=clock() - started,
    )


class InsufficientDiskError(RuntimeError):
    """Not enough free space on a node to receive its shards."""


# Filling a disk to the brim breaks the OS, not just the transfer. Leave room.
_DISK_HEADROOM_BYTES = 10 * 1024**3


def free_disk_bytes(path: str | Path, *, ssh_target: str | None = None) -> int:
    """Free bytes on the filesystem holding ``path``, locally or on a peer."""

    if ssh_target and not is_local_host(ssh_target):
        result = subprocess.run(
            # Unquoted so the remote shell expands ~; the path comes from a
            # validated model directory, not user text.
            [
                "ssh",
                *cluster_ssh_options(connect_timeout=10),
                ssh_target,
                f"df -k {path} | tail -1",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        fields = result.stdout.split()
        # df -k: Filesystem 1K-blocks Used Avail ...
        for index in (3, 2):
            if len(fields) > index:
                try:
                    return int(fields[index]) * 1024
                except ValueError:
                    continue
        return 0
    import shutil

    target = Path(path).expanduser()
    while not target.exists() and target.parent != target:
        target = target.parent
    try:
        return int(shutil.disk_usage(target).free)
    except OSError:
        return 0


def check_disk_for_staging(
    plan: StagingPlan,
    destination_dir: str | Path,
    *,
    ssh_target: str | None = None,
    headroom_bytes: int = _DISK_HEADROOM_BYTES,
    free_bytes: int | None = None,
) -> int:
    """Refuse a transfer that would not fit, before moving a single byte.

    Staging can move a hundred gigabytes; discovering there is no room after an
    hour of copying — or filling the disk and destabilising the Mac — is a poor
    way to find out. Returns the free bytes seen, so callers can report it.
    """

    available = (
        free_bytes
        if free_bytes is not None
        else free_disk_bytes(destination_dir, ssh_target=ssh_target)
    )
    if available <= 0:
        # Unmeasurable filesystem: proceed rather than block on a number we
        # could not read, matching how the memory guard treats an unknown host.
        return 0

    needed = plan.missing_bytes + headroom_bytes
    if needed > available:
        where = plan.node_id or str(destination_dir)
        raise InsufficientDiskError(
            f"{where} needs {plan.missing_bytes / 1024**3:.1f} GiB for its "
            f"shards but has {available / 1024**3:.1f} GiB free "
            f"(keeping {headroom_bytes / 1024**3:.0f} GiB for the system). "
            f"Free up "
            f"{(needed - available) / 1024**3:.1f} GiB, or give this node "
            f"fewer layers."
        )
    return available
