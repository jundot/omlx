# SPDX-License-Identifier: Apache-2.0
"""Model synchronization and per-node model paths for cluster v2.

Two responsibilities:

* A bounded, content-describing **manifest** for a locally-known model
  (safetensors index hash + file list + total bytes), served to peers over
  ``GET /api/cluster/models/{model_id}/manifest`` so a coordinator can answer
  "does that node have this model?" without SSHing anywhere.
* A :class:`ModelSyncManager` that compares two manifests and then moves the
  model, either over the enrolled SSH channel (``rsync``, resumable via
  ``--partial --append-verify``) or as a per-node Hugging Face download of
  only the files a shard needs (allow-patterns derived from the weight-map
  index — the exo pattern). Progress events are emitted for the UI.

All transports are injectable so unit tests never open a socket, run rsync,
or touch the Hugging Face hub.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import subprocess
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException

from ..exceptions import ModelNotFoundError
from .ssh_policy import cluster_ssh_options
from .staging import (
    index_shards,
    model_identity_digest,
    shards_for_stage,
    sidecar_files,
)

SyncMethod = Literal["auto", "rsync", "download"]
SyncState = Literal["present", "missing", "partial", "mismatch"]

#: Above this size ``method="auto"`` prefers a direct rsync over the enrolled
#: SSH channel; at or below it, each node downloads its own shard files.
AUTO_RSYNC_THRESHOLD_BYTES = 20 * 1024**3

_MANIFEST_TIMEOUT_SECONDS = 10.0
_MAX_MODEL_ID_LENGTH = 4096
_MAX_PROGRESS_EVENTS = 512

# ``rsync --info=progress2`` emits lines such as:
#   "  1,234,567,890  42%  123.45MB/s    0:01:23"
_PROGRESS2 = re.compile(
    r"^\s*(?P<bytes>[\d,]+)\s+(?P<pct>\d+)%\s+"
    r"(?P<rate>[\d.,]+)\s*(?P<unit>[KMGTP]?B)/s\s+"
    r"(?P<eta>\d+:\d{2}(?::\d{2})?)"
)
_RATE_MULTIPLIERS = {
    "B": 1,
    "KB": 1000,
    "MB": 1000**2,
    "GB": 1000**3,
    "TB": 1000**4,
    "PB": 1000**5,
}


class ModelSyncError(RuntimeError):
    """A sync operation could not start or finish."""


@dataclass(frozen=True)
class ManifestFile:
    """One file in a model directory: name and size, nothing else."""

    name: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "size_bytes": self.size_bytes}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ManifestFile:
        return cls(name=str(payload["name"]), size_bytes=int(payload["size_bytes"]))


@dataclass(frozen=True)
class ModelManifest:
    """Content description of one model directory, cheap to compute.

    Hashing every weight file would take minutes on a 300 GB model, so the
    identity rests on the safetensors weight-map index plus the sidecar
    digest staging already uses, with per-file sizes for drift detection.
    """

    model_id: str
    path: str
    index_sha256: str | None
    identity_sha256: str
    files: tuple[ManifestFile, ...]
    total_bytes: int
    source_repo_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "model_id": self.model_id,
            "path": self.path,
            "index_sha256": self.index_sha256,
            "identity_sha256": self.identity_sha256,
            "files": [item.to_dict() for item in self.files],
            "total_bytes": self.total_bytes,
            "source_repo_id": self.source_repo_id,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ModelManifest:
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("unsupported model manifest schema")
        files = payload.get("files")
        if not isinstance(files, list):
            raise ValueError("model manifest files must be an array")
        return cls(
            model_id=str(payload.get("model_id") or ""),
            path=str(payload.get("path") or ""),
            index_sha256=payload.get("index_sha256"),
            identity_sha256=str(payload.get("identity_sha256") or ""),
            files=tuple(ManifestFile.from_dict(item) for item in files),
            total_bytes=int(payload.get("total_bytes") or 0),
            source_repo_id=payload.get("source_repo_id"),
        )


@dataclass(frozen=True)
class SyncProgress:
    """One progress event for the UI; bytes/s and ETA when measurable."""

    model_id: str
    peer: str
    method: str
    phase: Literal[
        "checking", "transferring", "verifying", "done", "error"
    ]
    bytes_done: int = 0
    bytes_total: int = 0
    bytes_per_second: float | None = None
    eta_seconds: float | None = None
    detail: str = ""
    at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "peer": self.peer,
            "method": self.method,
            "phase": self.phase,
            "bytes_done": self.bytes_done,
            "bytes_total": self.bytes_total,
            "bytes_per_second": self.bytes_per_second,
            "eta_seconds": self.eta_seconds,
            "detail": self.detail,
            "at": self.at,
        }


def build_manifest(
    model_path: str | Path,
    *,
    model_id: str | None = None,
    source_repo_id: str | None = None,
) -> ModelManifest:
    """Describe a local model directory for peer comparison."""

    root = Path(model_path).expanduser()
    if not root.is_dir():
        raise ValueError(f"model path is not a directory: {root}")
    weights = sorted(root.glob("*.safetensors"))
    if not weights:
        raise ValueError(f"no safetensors weights found in {root}")

    index_path = root / "model.safetensors.index.json"
    index_sha256 = (
        hashlib.sha256(index_path.read_bytes()).hexdigest()
        if index_path.is_file()
        else None
    )
    files = [
        ManifestFile(name=path.name, size_bytes=path.stat().st_size)
        for path in weights
    ]
    for name in sidecar_files(root):
        path = root / name
        files.append(ManifestFile(name=name, size_bytes=path.stat().st_size))
    files.sort(key=lambda item: item.name)
    return ModelManifest(
        model_id=model_id or root.name,
        path=str(root),
        index_sha256=index_sha256,
        identity_sha256=model_identity_digest(root),
        files=tuple(files),
        total_bytes=sum(item.size_bytes for item in files),
        source_repo_id=source_repo_id,
    )


def allow_patterns_for_shard(
    model_path: str | Path,
    start_layer: int | None = None,
    end_layer: int | None = None,
) -> tuple[str, ...]:
    """HF ``allow_patterns`` for only the files one shard needs.

    Derived from the weight-map index (or shard headers when no index ships)
    via the same layer→file map staging uses: layer-scoped shards for
    ``[start_layer, end_layer)`` plus the shared sidecars. With no layer
    range, the pattern list covers the whole model.
    """

    shards = index_shards(model_path)
    if start_layer is not None and end_layer is not None:
        if not 0 <= start_layer < end_layer:
            raise ValueError("shard layer range must satisfy 0 <= start < end")
        shards = shards_for_stage(shards, start_layer, end_layer)
    patterns = {shard.name for shard in shards}
    patterns.update(sidecar_files(model_path))
    return tuple(sorted(patterns))


def compare_manifests(local: ModelManifest, peer: ModelManifest | None) -> dict[str, Any]:
    """Whether ``peer`` can serve the model ``local`` describes.

    * ``missing`` — the peer answered 404 or is unreachable.
    * ``mismatch`` — the peer has a different model (weight-map index hash or
      sidecar identity differs).
    * ``partial`` — same model, but files are absent or truncated.
    * ``present`` — every file is there at the expected size.
    """

    if peer is None:
        return {
            "state": "missing",
            "bytes": 0,
            "total_bytes": local.total_bytes,
            "missing": [item.name for item in local.files],
        }
    if (
        local.index_sha256 is not None
        and peer.index_sha256 is not None
        and local.index_sha256 != peer.index_sha256
    ) or local.identity_sha256 != peer.identity_sha256:
        return {
            "state": "mismatch",
            "bytes": 0,
            "total_bytes": local.total_bytes,
            "missing": [item.name for item in local.files],
        }
    peer_sizes = {item.name: item.size_bytes for item in peer.files}
    missing = [
        item.name for item in local.files if peer_sizes.get(item.name) != item.size_bytes
    ]
    matched = sum(
        item.size_bytes for item in local.files if item.name not in set(missing)
    )
    return {
        "state": "partial" if missing else "present",
        "bytes": matched,
        "total_bytes": local.total_bytes,
        "missing": missing,
    }


def parse_rsync_progress(line: str) -> tuple[int, float, float] | None:
    """Parse one ``--info=progress2`` line → (bytes_done, bytes/s, eta_s)."""

    match = _PROGRESS2.match(line.strip())
    if match is None:
        return None
    done = int(match.group("bytes").replace(",", ""))
    rate = float(match.group("rate").replace(",", "")) * _RATE_MULTIPLIERS[
        match.group("unit")
    ]
    parts = [int(part) for part in match.group("eta").split(":")]
    eta = parts[0] * 3600 + parts[1] * 60 + parts[2] if len(parts) == 3 else (
        parts[0] * 60 + parts[1]
    )
    return done, rate, float(eta)


def build_rsync_argv(
    source_dir: str | Path,
    ssh_target: str,
    destination_dir: str | Path,
    *,
    ssh_identity: str | Path | None = None,
) -> list[str]:
    """A resumable rsync over the enrolled SSH channel.

    ``--partial --append-verify`` resumes interrupted transfers and verifies
    the appended bytes, the functional equivalent of ``-P --append-verify``
    without the interactive progress format. The SSH transport uses the one
    shared non-interactive cluster policy (managed identity, accept-new
    host keys, BatchMode) — never a hand-rolled option set.
    """

    ssh_argv = ["ssh", *cluster_ssh_options()]
    if ssh_identity is not None:
        ssh_argv.extend(["-i", str(ssh_identity)])
    source = str(Path(source_dir).expanduser()).rstrip("/") + "/"
    destination = f"{ssh_target}:{destination_dir}"
    return [
        "rsync",
        "-a",
        "--partial",
        "--append-verify",
        "--info=progress2",
        "-e",
        shlex.join(ssh_argv),
        source,
        destination,
    ]


def _default_ssh_trust(ssh_target: str) -> bool:
    """Whether the dedicated cluster key exists for non-interactive SSH."""

    del ssh_target  # the key is shared; the target itself is probed at launch
    return (Path.home() / ".ssh" / "omlx_cluster").is_file()


def _default_hf_download(
    *,
    repo_id: str,
    allow_patterns: Iterable[str],
    local_dir: str | Path,
) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ModelSyncError(
            "huggingface_hub is not installed on this node; install it or use "
            "method='rsync' to copy the model from the coordinator instead"
        ) from exc
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=list(allow_patterns),
        local_dir=str(local_dir),
    )


def _normalize_peer_url(peer: str) -> str:
    peer = peer.strip()
    if not peer:
        raise ValueError("peer must be a host:port or http(s) URL")
    if "://" not in peer:
        if ":" not in peer.rsplit("]", 1)[-1]:
            raise ValueError(
                f"peer {peer!r} has no port; use host:port or a full URL"
            )
        peer = f"http://{peer}"
    return peer.rstrip("/")


class ModelSyncManager:
    """Compare and synchronize model files between cluster nodes.

    Every transport is injectable: ``http_fetch`` for peer manifests,
    ``ssh_trust`` for the auto-method decision, ``rsync_run`` for the copy,
    and ``hf_download`` for per-node downloads. Tests inject fakes; the
    defaults are the real thing and are never exercised offline.
    """

    def __init__(
        self,
        *,
        http_fetch: Callable[[str, float], dict[str, Any] | None] | None = None,
        ssh_trust: Callable[[str], bool] | None = None,
        rsync_run: Callable[..., int] | None = None,
        hf_download: Callable[..., None] | None = None,
        pool_getter: Callable[[], Any] | None = None,
        settings_loader: Callable[[], Any] | None = None,
        manifest_timeout: float = _MANIFEST_TIMEOUT_SECONDS,
    ) -> None:
        self._http_fetch = http_fetch or self._default_http_fetch
        self._ssh_trust = ssh_trust or _default_ssh_trust
        self._rsync_run = rsync_run or self._default_rsync_run
        self._hf_download = hf_download or _default_hf_download
        self._pool_getter = pool_getter
        self._settings_loader = settings_loader
        self._manifest_timeout = manifest_timeout
        self._events: list[SyncProgress] = []
        self._lock = threading.Lock()

    # -- model resolution ------------------------------------------------

    def resolve_local_model_path(self, model_id: str) -> Path:
        """Resolve a model ID, repo ID, or path to a local directory."""

        if not model_id or len(model_id) > _MAX_MODEL_ID_LENGTH or "\x00" in model_id:
            raise ModelNotFoundError(model_id or "<empty>", [])
        candidate = Path(model_id).expanduser()
        if candidate.is_dir():
            return candidate.resolve()
        pool = self._pool_getter() if self._pool_getter is not None else None
        if pool is not None:
            status = pool.get_status()
            for item in status.get("models", []) if isinstance(status, dict) else []:
                if not isinstance(item, dict):
                    continue
                if model_id in {item.get("id"), item.get("source_repo_id"), item.get("model_path")}:
                    path = Path(str(item["model_path"])).expanduser()
                    if path.is_dir():
                        return path.resolve()
        settings = (
            self._settings_loader() if self._settings_loader is not None else None
        )
        if settings is not None:
            from ..model_discovery import discover_models_from_dirs

            model_dirs = [
                Path(entry).expanduser()
                for entry in settings.get_effective_model_dirs()
            ]
            discovered = discover_models_from_dirs(model_dirs)
            for key, entry in discovered.items():
                if model_id in {key, entry.model_id, entry.source_repo_id}:
                    path = Path(entry.model_path).expanduser()
                    if path.is_dir():
                        return path.resolve()
        raise ModelNotFoundError(model_id, [])

    def local_manifest(self, model_id: str) -> ModelManifest:
        path = self.resolve_local_model_path(model_id)
        source_repo_id = None if Path(model_id).expanduser().is_dir() else model_id
        return build_manifest(path, model_id=model_id, source_repo_id=source_repo_id)

    # -- peer manifest ----------------------------------------------------

    @staticmethod
    def _default_http_fetch(url: str, timeout: float) -> dict[str, Any] | None:
        import urllib.error
        import urllib.request

        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise ModelSyncError(f"peer manifest request failed: HTTP {exc.code}") from exc
        except (OSError, urllib.error.URLError) as exc:
            raise ModelSyncError(f"peer is unreachable: {exc}") from exc

    def peer_manifest(self, peer: str, model_id: str) -> ModelManifest | None:
        """Fetch a peer's manifest; ``None`` when the model is unknown there."""

        base = _normalize_peer_url(peer)
        payload = self._http_fetch(
            f"{base}/api/cluster/models/{model_id}/manifest",
            self._manifest_timeout,
        )
        if payload is None:
            return None
        try:
            return ModelManifest.from_dict(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelSyncError(f"peer returned an invalid manifest: {exc}") from exc

    def status(
        self,
        peer: str,
        model_id: str,
        *,
        local: ModelManifest | None = None,
    ) -> dict[str, Any]:
        """{state: present|missing|partial|mismatch, bytes} for a peer."""

        local = local or self.local_manifest(model_id)
        result = compare_manifests(local, self.peer_manifest(peer, model_id))
        result["peer"] = peer
        result["model_id"] = model_id
        return result

    # -- sync --------------------------------------------------------------

    def decide_method(
        self,
        ssh_target: str | None,
        total_bytes: int,
    ) -> Literal["rsync", "download"]:
        """auto = rsync when SSH trust exists AND the model is over 20 GB."""

        if ssh_target and total_bytes > AUTO_RSYNC_THRESHOLD_BYTES:
            if self._ssh_trust(ssh_target):
                return "rsync"
        return "download"

    def sync(
        self,
        peer: str,
        model_id: str,
        method: SyncMethod = "auto",
        *,
        ssh_target: str | None = None,
        destination: str | Path | None = None,
        repo_id: str | None = None,
        start_layer: int | None = None,
        end_layer: int | None = None,
        on_progress: Callable[[SyncProgress], None] | None = None,
    ) -> dict[str, Any]:
        """Bring ``peer`` (or a local destination) to ``present`` for a model.

        ``rsync`` copies the local model directory to ``ssh_target:destination``
        over the enrolled channel. ``download`` fetches only shard-needed files
        from the Hugging Face hub into ``destination`` on this node (run the
        same call on each node, exo-style). Progress events are emitted to
        ``on_progress`` and retained in :attr:`events` for UI polling.
        """

        if method not in {"auto", "rsync", "download"}:
            raise ValueError(f"unsupported sync method: {method!r}")
        local = self.local_manifest(model_id)
        chosen = (
            self.decide_method(ssh_target, local.total_bytes)
            if method == "auto"
            else method
        )
        self._emit(on_progress, local.model_id, peer, chosen, "checking",
                   bytes_total=local.total_bytes, detail="comparing manifests")
        try:
            if chosen == "rsync":
                result = self._sync_rsync(
                    local, peer, ssh_target, destination, on_progress
                )
            else:
                result = self._sync_download(
                    local,
                    model_id,
                    peer,
                    destination,
                    repo_id,
                    start_layer,
                    end_layer,
                    on_progress,
                )
        except Exception as exc:
            self._emit(on_progress, local.model_id, peer, chosen, "error",
                       bytes_total=local.total_bytes, detail=str(exc)[:500])
            raise
        self._emit(on_progress, local.model_id, peer, chosen, "done",
                   bytes_done=local.total_bytes, bytes_total=local.total_bytes)
        return result

    def _sync_rsync(
        self,
        local: ModelManifest,
        peer: str,
        ssh_target: str | None,
        destination: str | Path | None,
        on_progress: Callable[[SyncProgress], None] | None,
    ) -> dict[str, Any]:
        if not ssh_target:
            raise ModelSyncError(
                "rsync sync needs ssh_target (the enrolled SSH destination)"
            )
        if not self._ssh_trust(ssh_target):
            raise ModelSyncError(
                f"no enrolled SSH trust for {ssh_target}; pair the nodes first "
                "or use method='download'"
            )
        # Without an explicit destination the peer keeps the coordinator's
        # absolute path — the legacy shared-path layout, still the default
        # until a path_map says otherwise.
        target_dir = str(destination or local.path)
        argv = build_rsync_argv(local.path, ssh_target, target_dir)

        def progress(done: int, rate: float, eta: float) -> None:
            self._emit(
                on_progress,
                local.model_id,
                peer,
                "rsync",
                "transferring",
                bytes_done=done,
                bytes_total=local.total_bytes,
                bytes_per_second=rate,
                eta_seconds=eta,
            )

        rc = self._rsync_run(argv, on_line=self._progress_line_parser(progress))
        if rc != 0:
            raise ModelSyncError(f"rsync to {ssh_target} exited with status {rc}")
        self._emit(on_progress, local.model_id, peer, "rsync", "verifying",
                   bytes_total=local.total_bytes, detail="verifying peer manifest")
        return {
            "method": "rsync",
            "peer": peer,
            "ssh_target": ssh_target,
            "destination": target_dir,
            "bytes": local.total_bytes,
        }

    def _sync_download(
        self,
        local: ModelManifest,
        model_id: str,
        peer: str,
        destination: str | Path | None,
        repo_id: str | None,
        start_layer: int | None,
        end_layer: int | None,
        on_progress: Callable[[SyncProgress], None] | None,
    ) -> dict[str, Any]:
        repo = repo_id or local.source_repo_id
        if not repo or Path(repo).expanduser().is_dir():
            raise ModelSyncError(
                "download sync needs the model's Hugging Face repo ID; the "
                "local copy has no repository reference, use method='rsync'"
            )
        if destination is None:
            raise ModelSyncError(
                "download sync needs a destination directory on this node"
            )
        patterns = allow_patterns_for_shard(local.path, start_layer, end_layer)
        self._emit(
            on_progress,
            local.model_id,
            peer,
            "download",
            "transferring",
            bytes_total=local.total_bytes,
            detail=f"fetching {len(patterns)} files from {repo}",
        )
        self._hf_download(
            repo_id=repo,
            allow_patterns=patterns,
            local_dir=destination,
        )
        return {
            "method": "download",
            "peer": peer,
            "repo_id": repo,
            "destination": str(destination),
            "allow_patterns": list(patterns),
            "bytes": local.total_bytes,
        }

    # -- progress plumbing ---------------------------------------------------

    @staticmethod
    def _progress_line_parser(
        progress: Callable[[int, float, float], None],
    ) -> Callable[[str], None]:
        def parse(line: str) -> None:
            parsed = parse_rsync_progress(line)
            if parsed is not None:
                progress(*parsed)

        return parse

    @staticmethod
    def _default_rsync_run(argv: list[str], on_line: Callable[[str], None]) -> int:
        process = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            on_line(line)
        return process.wait()

    def _emit(
        self,
        on_progress: Callable[[SyncProgress], None] | None,
        model_id: str,
        peer: str,
        method: str,
        phase: str,
        **fields: Any,
    ) -> SyncProgress:
        event = SyncProgress(
            model_id=model_id,
            peer=peer,
            method=method,
            phase=phase,  # type: ignore[arg-type]
            **fields,
        )
        with self._lock:
            self._events.append(event)
            if len(self._events) > _MAX_PROGRESS_EVENTS:
                del self._events[: len(self._events) - _MAX_PROGRESS_EVENTS]
        if on_progress is not None:
            on_progress(event)
        return event

    @property
    def events(self) -> tuple[SyncProgress, ...]:
        """Bounded recent progress history for UI polling."""

        with self._lock:
            return tuple(self._events)


# -- manifest endpoint ------------------------------------------------------

manifest_router = APIRouter(prefix="/api/cluster", tags=["cluster-modelsync"])

_pool_getter: Callable[[], Any] | None = None
_settings_loader: Callable[[], Any] | None = None


def set_modelsync_getters(
    engine_pool_getter: Callable[[], Any] | None = None,
    settings_loader: Callable[[], Any] | None = None,
) -> None:
    """Inject server-owned dependencies without importing ``omlx.server``."""

    global _pool_getter, _settings_loader
    _pool_getter = engine_pool_getter
    _settings_loader = settings_loader


def _default_settings_loader() -> Any:
    from ..settings import GlobalSettings

    return GlobalSettings.load()


def _manager() -> ModelSyncManager:
    return ModelSyncManager(
        pool_getter=_pool_getter,
        settings_loader=_settings_loader or _default_settings_loader,
    )


def _allowed_model_roots() -> tuple[Path, ...]:
    try:
        settings = (_settings_loader or _default_settings_loader)()
        return tuple(
            Path(entry).expanduser().resolve()
            for entry in settings.get_effective_model_dirs()
        )
    except Exception:  # noqa: BLE001 - a broken settings file must not 500 here
        return ()


def _guard_manifest_path(path: Path) -> None:
    """Path-typed model IDs stay inside the configured model directories.

    Repo-style IDs are resolved through the engine pool / discovery, which
    already only surfaces registered models. A raw path must not let an admin
    token enumerate arbitrary directories.
    """

    roots = _allowed_model_roots()
    if not roots:
        return
    if not any(root == path or root in path.parents for root in roots):
        raise HTTPException(
            status_code=403,
            detail="model path is outside the configured model directories",
        )


@manifest_router.get("/models/{model_id:path}/manifest")
async def cluster_model_manifest(model_id: str) -> dict[str, Any]:
    """Safetensors index hash + file list + total bytes for a local model.

    Peers call this to decide whether they must sync before activation; the
    payload is names and sizes only — never weights, never credentials.
    """

    import asyncio

    manager = _manager()
    try:
        path = manager.resolve_local_model_path(model_id)
    except ModelNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail=f"unknown model: {model_id}"
        ) from exc
    _guard_manifest_path(path)
    try:
        manifest = await asyncio.to_thread(build_manifest, path, model_id=model_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return manifest.to_dict()


__all__ = [
    "AUTO_RSYNC_THRESHOLD_BYTES",
    "ManifestFile",
    "ModelManifest",
    "ModelSyncError",
    "ModelSyncManager",
    "SyncProgress",
    "allow_patterns_for_shard",
    "build_manifest",
    "build_rsync_argv",
    "compare_manifests",
    "manifest_router",
    "parse_rsync_progress",
    "set_modelsync_getters",
]
