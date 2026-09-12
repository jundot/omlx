# SPDX-License-Identifier: Apache-2.0
"""Export/import artifacts for the paged SSD prompt cache.

An artifact is a portable, self-contained zip holding every SSD block (and
GDN sidecar) that makes up a contiguous prefix chain for one model:

    manifest.json
    blocks/<block_hash_hex>.safetensors
    gdn/<signature_digest>/<block_hash_hex>.safetensors

Block files are byte-identical copies of what the paged SSD cache already
writes to disk, so an import is "copy the files back and re-index" — no
re-serialization and no re-prefill. The manifest pins model identity, block
size, per-file sha256 and the engine cache-format version; import validates
all of it and fails closed (nothing is registered) on any mismatch.

The manifest also carries the exact token ids of the prefix so an import can
re-derive the expected block-hash chain and reject a manifest whose blocks do
not line up with it. The token ids are therefore part of the artifact
(don't ship artifacts where the transcript would not be acceptable).

This exists so an external agent (e.g. a coding agent that owns the session
transcript) can materialize a long prompt prefix into a file it controls —
next to its session — and hand it back to oMLX after a restart instead of
paying a full re-prefill. See issue #3612.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
import uuid
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .._version import __version__
from .paged_cache import compute_block_hash
from .paged_ssd_cache import _CACHE_FORMAT_VERSION, _fsync_parent_dir

logger = logging.getLogger(__name__)

ARTIFACT_FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
BLOCKS_DIR = "blocks"
GDN_DIR = "gdn"
_IO_CHUNK = 8 * 1024 * 1024

_BLOCK_PATH_RE = re.compile(r"blocks/[0-9a-f]{64}\.safetensors")
_GDN_PATH_RE = re.compile(r"gdn/[0-9a-f]{64}/[0-9a-f]{64}\.safetensors")


class ArtifactError(Exception):
    """Fail-closed artifact validation or IO error."""


def chain_hashes(
    token_ids: list[int], block_size: int, model_name: str
) -> Iterator[tuple[bytes, list[int]]]:
    """Chain-hashed blocks of ``token_ids``.

    The same hash walk the scheduler and ``/api/cache/probe`` use, so the
    blocks exported here are exactly the blocks a re-prefill would build.
    """
    parent: bytes = b""
    for start in range(0, len(token_ids), block_size):
        chunk = list(token_ids[start : start + block_size])
        parent = compute_block_hash(
            parent, chunk, extra_keys=None, model_name=model_name
        )
        yield parent, chunk


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_IO_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_stream(source, file_out) -> tuple[str, int]:
    """Stream ``source`` into ``file_out``, returning (sha256_hex, size)."""
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = source.read(_IO_CHUNK)
        if not chunk:
            break
        size += len(chunk)
        digest.update(chunk)
        file_out.write(chunk)
    file_out.flush()
    os.fsync(file_out.fileno())
    return digest.hexdigest(), size


def _sanitize_model_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", model_name)[:80] or "model"


def export_artifact(
    manager: Any,
    *,
    model_name: str,
    token_ids: list[int],
    block_size: int,
    output_dir: Path,
) -> dict:
    """Materialize the longest restorable prefix of ``token_ids`` into a zip.

    A block is exportable when it is in the SSD index and its file exists on
    disk (hot-cache-only entries with no durable file are not portable). The
    walk stops at the first non-exportable block: the chain is contiguous by
    construction, so nothing past it is restorable either.

    Returns ``{"artifact_path": str, "size_bytes": int, "manifest": dict}``.
    When nothing is restorable the artifact is still written (with zero
    blocks); the caller decides how to treat it.
    """
    index = manager._index

    block_entries: list[dict[str, Any]] = []
    source_paths: list[Path] = []
    restorable_tokens = 0
    for block_hash, chunk in chain_hashes(token_ids, block_size, model_name):
        if not index.contains(block_hash):
            break
        file_path = manager._get_file_path(block_hash)
        if not file_path.is_file():
            break
        metadata = manager.get_block_metadata(block_hash)
        block_entries.append(
            {
                "hash": block_hash.hex(),
                "path": f"{BLOCKS_DIR}/{block_hash.hex()}.safetensors",
                "size": file_path.stat().st_size,
                "sha256": _sha256_file(file_path),
                "token_count": len(chunk),
                "model_name": str(
                    getattr(metadata, "model_name", "") or model_name
                ),
                "num_layers": int(getattr(metadata, "num_layers", 0)),
                "cache_signature": str(
                    getattr(metadata, "cache_signature", "") or ""
                ),
            }
        )
        source_paths.append(file_path)
        restorable_tokens += len(chunk)

    gdn_entries: list[dict[str, Any]] = []
    if block_entries:
        wanted = {b["hash"] for b in block_entries}
        for entry in manager._gdn_sidecar_index.get_all_metadata():
            source_hash = entry.source_block_hash.hex()
            if source_hash not in wanted:
                continue
            sidecar_path = Path(entry.file_path)
            if not sidecar_path.is_file():
                continue
            gdn_entries.append(
                {
                    "source_hash": source_hash,
                    "sig_digest": entry.cache_signature_digest,
                    "path": (
                        f"{GDN_DIR}/{entry.cache_signature_digest}/"
                        f"{source_hash}.safetensors"
                    ),
                    "size": sidecar_path.stat().st_size,
                    "sha256": _sha256_file(sidecar_path),
                    "token_count": int(entry.token_count),
                }
            )
            source_paths.append(sidecar_path)

    manifest = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "engine_version": __version__,
        "cache_format_version": _CACHE_FORMAT_VERSION,
        "model_name": model_name,
        "block_size": block_size,
        "restorable_tokens": restorable_tokens,
        "token_ids": token_ids,
        "blocks": block_entries,
        "gdn_sidecars": gdn_entries,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    final_path = (
        output_dir
        / f"kvart_{_sanitize_model_name(model_name)}_{restorable_tokens}tok_{stamp}.omlxzip"
    )
    descriptor, temp_path = tempfile.mkstemp(
        prefix=f".{final_path.name}.", suffix=".part", dir=output_dir
    )
    os.close(descriptor)
    try:
        with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr(
                MANIFEST_NAME,
                json.dumps(manifest, separators=(",", ":"), sort_keys=True),
            )
            for entry, source in zip(
                block_entries + gdn_entries, source_paths, strict=True
            ):
                with open(source, "rb") as source_file:
                    archive.writestr(entry["path"], source_file.read())
        os.replace(temp_path, final_path)
        _fsync_parent_dir(final_path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temp_path)
        raise
    return {
        "artifact_path": str(final_path),
        "size_bytes": final_path.stat().st_size,
        "manifest": manifest,
    }


def import_artifact(
    manager: Any,
    *,
    artifact_path: Path,
    model_name: str,
) -> dict:
    """Validate an artifact and register its blocks with ``manager``.

    Fails closed: any integrity or compatibility problem raises
    ``ArtifactError`` and nothing is registered. Files already present with
    a matching size are kept as-is (they are already local cache state).
    """
    artifact_path = Path(artifact_path)
    if not artifact_path.is_file():
        raise ArtifactError(f"artifact not found: {artifact_path}")
    try:
        archive = zipfile.ZipFile(artifact_path)
    except zipfile.BadZipFile as exc:
        raise ArtifactError(f"not a valid artifact zip: {exc}") from exc

    with archive:
        names = set(archive.namelist())
        if MANIFEST_NAME not in names:
            raise ArtifactError("artifact manifest is missing")
        try:
            manifest = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ArtifactError(f"artifact manifest is unreadable: {exc}") from exc
        if not isinstance(manifest, dict):
            raise ArtifactError("artifact manifest has an invalid shape")
        if manifest.get("format_version") != ARTIFACT_FORMAT_VERSION:
            raise ArtifactError(
                f"unsupported artifact format version "
                f"{manifest.get('format_version')!r}"
            )
        if manifest.get("model_name") != model_name:
            raise ArtifactError(
                f"artifact is for model {manifest.get('model_name')!r}, "
                f"not {model_name!r}"
            )
        block_size = int(manifest.get("block_size") or 0)
        if block_size <= 0:
            raise ArtifactError("artifact manifest has no block size")
        token_ids = [int(t) for t in (manifest.get("token_ids") or [])]
        if not token_ids:
            raise ArtifactError("artifact manifest has no token ids")
        block_entries = manifest.get("blocks") or []
        gdn_entries = manifest.get("gdn_sidecars") or []

        # The manifest must line up with the hash chain its token ids imply.
        expected_chain = list(chain_hashes(token_ids, block_size, model_name))
        if [e.get("token_count") for e in block_entries] != [
            len(chunk) for _h, chunk in expected_chain[: len(block_entries)]
        ] or len(block_entries) > len(expected_chain):
            raise ArtifactError(
                "artifact manifest blocks do not match the token chain"
            )
        for entry, (expected_hash, _chunk) in zip(
            block_entries, expected_chain, strict=False
        ):
            if entry.get("hash") != expected_hash.hex():
                raise ArtifactError(
                    f"artifact block hash {str(entry.get('hash'))[:16]}… does "
                    f"not match the token chain"
                )

        for entry in block_entries + gdn_entries:
            if not isinstance(entry, dict) or not entry.get("path"):
                raise ArtifactError("artifact manifest entry is malformed")
            if entry["path"] not in names:
                raise ArtifactError(f"artifact member is missing: {entry['path']}")
            if entry["path"].startswith(BLOCKS_DIR + "/"):
                if not _BLOCK_PATH_RE.fullmatch(entry["path"]):
                    raise ArtifactError(
                        f"artifact block path is inconsistent: {entry['path']}"
                    )
            elif not _GDN_PATH_RE.fullmatch(entry["path"]):
                raise ArtifactError(
                    f"artifact sidecar path is inconsistent: {entry['path']}"
                )

        signature_by_hash = {
            b["hash"]: str(b.get("cache_signature", "") or "")
            for b in block_entries
        }
        for entry in gdn_entries:
            if entry["source_hash"] not in signature_by_hash:
                raise ArtifactError(
                    f"artifact sidecar has no matching block: "
                    f"{entry['source_hash'][:16]}…"
                )
            signature = signature_by_hash[entry["source_hash"]]
            if (
                hashlib.sha256(signature.encode("utf-8")).hexdigest()
                != entry["sig_digest"]
            ):
                raise ArtifactError(
                    f"artifact sidecar signature digest does not match its "
                    f"block: {entry['source_hash'][:16]}…"
                )

        placed_files: list[Path] = []
        staging_dir: Path | None = None
        try:
            # Pass 1: place every payload (atomically) and integrity-check
            # it before anything is registered.
            imported_blocks = 0
            existing_blocks = 0
            for entry in block_entries:
                block_hash = bytes.fromhex(entry["hash"])
                target = manager._get_file_path(block_hash)
                if target.is_file():
                    if target.stat().st_size != int(entry["size"]):
                        raise ArtifactError(
                            f"block {entry['hash'][:16]}… exists locally "
                            f"with a different size — refusing to overwrite"
                        )
                    existing_blocks += 1
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                descriptor, temp_path = tempfile.mkstemp(
                    prefix=f".{entry['hash']}.", suffix=".part", dir=target.parent
                )
                with (
                    os.fdopen(descriptor, "wb") as out,
                    archive.open(entry["path"]) as source,
                ):
                    digest_hex, size = _sha256_stream(source, out)
                if digest_hex != entry["sha256"] or size != int(entry["size"]):
                    os.unlink(temp_path)
                    raise ArtifactError(
                        f"block {entry['hash'][:16]}… failed the sha256 "
                        f"integrity check"
                    )
                os.replace(temp_path, target)
                _fsync_parent_dir(target)
                placed_files.append(target)
                imported_blocks += 1

            # Pass 2: read metadata from disk and gate every block through
            # the same compatibility checks the startup scan uses.
            metadata_by_hash: dict[str, Any] = {}
            for entry in block_entries:
                block_hash = bytes.fromhex(entry["hash"])
                target = manager._get_file_path(block_hash)
                metadata = manager._read_file_metadata(target)
                if metadata is None:
                    raise ArtifactError(
                        f"block {entry['hash'][:16]}… is not a readable "
                        f"cache block"
                    )
                if metadata.block_size != block_size:
                    raise ArtifactError(
                        f"block {entry['hash'][:16]}… was saved with a "
                        f"different block size"
                    )
                if not manager._is_compatible_block(metadata):
                    reason = manager.signature_mismatch_reason(
                        metadata.cache_signature
                    )
                    raise ArtifactError(
                        f"block {entry['hash'][:16]}… is incompatible with "
                        f"the loaded model"
                        + (f": {reason}" if reason else "")
                    )
                metadata_by_hash[entry["hash"]] = metadata

            # Pass 3: register blocks, then GDN sidecars.
            for entry in block_entries:
                manager._index.add(metadata_by_hash[entry["hash"]])

            imported_gdn = 0
            if gdn_entries:
                staging_dir = manager._cache_dir / f".artifact_import_{uuid.uuid4().hex}"
                staging_dir.mkdir(parents=True, exist_ok=True)
                for entry in gdn_entries:
                    descriptor, temp_path = tempfile.mkstemp(
                        prefix=f".{entry['source_hash']}.",
                        suffix=".part",
                        dir=staging_dir,
                    )
                    with (
                        os.fdopen(descriptor, "wb") as out,
                        archive.open(entry["path"]) as source,
                    ):
                        _digest_hex, _size = _sha256_stream(source, out)
                    signature = signature_by_hash[entry["source_hash"]]
                    committed = manager.commit_gdn_checkpoint_file(
                        bytes.fromhex(entry["source_hash"]),
                        Path(temp_path),
                        token_count=int(entry["token_count"]),
                        model_name=model_name,
                        cache_signature=signature,
                        block_size=block_size,
                    )
                    if committed is None:
                        raise ArtifactError(
                            f"could not register GDN sidecar for block "
                            f"{entry['source_hash'][:16]}…"
                        )
                    imported_gdn += 1

            manager.enforce_size_limit()
        except BaseException:
            for path in placed_files:
                with contextlib.suppress(OSError):
                    path.unlink()
            if staging_dir is not None:
                with contextlib.suppress(OSError):
                    shutil.rmtree(staging_dir, ignore_errors=True)
            raise

        # Re-walk the chain to report what is restorable after the import.
        restorable = 0
        for block_hash, chunk in expected_chain:
            if not manager._index.contains(block_hash):
                break
            restorable += len(chunk)

        return {
            "model_name": model_name,
            "block_size": block_size,
            "restorable_tokens": restorable,
            "imported_blocks": imported_blocks,
            "existing_blocks": existing_blocks,
            "imported_gdn_sidecars": imported_gdn,
            "artifact_size_bytes": artifact_path.stat().st_size,
        }
