# SPDX-License-Identifier: Apache-2.0
"""On-disk store of projected DFlash drafter contexts for prefix-cache hits.

A prefix-cache hit restores the target's KV cache for the cached prompt
tokens, but the drafter never sees their layer captures. Within one server a
finished request's context can be reused (``DFlashDrafter._kept``); this store
keeps the window of projected rows at the end of long prompts on disk, so a
hit served from the SSD prefix cache after a restart, or a new branch from a
shared prefix, still drafts with a full window.

Each entry holds the newest projected rows ending at absolute position
``end`` and the prompt tokens ``tokens[:end]``. Files:
``<key>-<end>-<rows>.ctx.npy`` (rows as the bit patterns of the dtype in the
namespace) and ``<key>-<end>-<rows>.tok.npy`` (int32 tokens).
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)

# Entries kept on disk; the oldest by last use is removed past this count.
_MAX_ENTRIES = int(os.environ.get("OMLX_DFLASH_CONTEXT_ENTRIES", "48"))


def store_enabled() -> bool:
    return os.environ.get("OMLX_DFLASH_CONTEXT_STORE", "1") != "0"


def min_prompt_tokens() -> int:
    """Shorter prompts are neither stored nor recomputed."""
    return int(os.environ.get("OMLX_DFLASH_CONTEXT_MIN", "4096"))


def default_root() -> Path:
    root = os.environ.get("OMLX_DFLASH_CONTEXT_DIR")
    if root:
        return Path(root).expanduser()
    return Path.home() / ".omlx" / "dflash-context"


# Rows are stored as the unsigned-integer bit patterns of their dtype.
_BITS = {mx.bfloat16: mx.uint16, mx.float16: mx.uint16, mx.float32: mx.uint32}


@dataclass
class _Entry:
    stem: str
    end: int
    rows: int

    @property
    def start(self) -> int:
        return self.end - self.rows


def _token_key(tokens: Sequence[int]) -> str:
    return hashlib.sha1(np.asarray(tokens, dtype=np.int32).tobytes()).hexdigest()[:24]


class ContextStore:
    """Projected drafter contexts on disk, keyed by the tokens they follow."""

    def __init__(self, root: Path, dtype=mx.bfloat16, max_entries: int = _MAX_ENTRIES):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.dtype = dtype
        self.max_entries = int(max_entries)
        self._lock = threading.Lock()
        self._entries: dict[str, _Entry] = {}
        self._writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dflash-ctx")
        for path in self.root.glob("*.ctx.npy"):
            stem = path.name[: -len(".ctx.npy")]
            try:
                _key, end, rows = stem.rsplit("-", 2)
                entry = _Entry(stem, int(end), int(rows))
            except ValueError:
                continue
            if (self.root / f"{stem}.tok.npy").is_file():
                self._entries[stem] = entry

    def __len__(self) -> int:
        return len(self._entries)

    def lookup(self, prefix: Sequence[int], cached: int) -> tuple[_Entry, int] | None:
        """The entry with the most rows ending at ``cached`` for ``prefix``."""
        cached = int(cached)
        with self._lock:
            candidates = [
                e for e in self._entries.values() if e.start < cached <= e.end
            ]
        best = None
        want = None
        for entry in sorted(candidates, key=lambda e: cached - e.start, reverse=True):
            take = cached - entry.start
            if best is not None and take <= best[1]:
                break
            try:
                tokens = np.load(self.root / f"{entry.stem}.tok.npy", mmap_mode="r")
            except (OSError, ValueError):
                continue
            if want is None:
                want = np.asarray(prefix[:cached], dtype=np.int32)
            if tokens.shape[0] >= cached and np.array_equal(tokens[:cached], want):
                best = (entry, take)
        return best

    def load(self, entry: _Entry, cached: int, take: int) -> mx.array | None:
        """Rows ``[cached - take, cached)`` of an entry, as ``(1, take, D)``."""
        try:
            bits = np.load(self.root / f"{entry.stem}.ctx.npy", mmap_mode="r")
        except (OSError, ValueError):
            return None
        offset = int(cached) - entry.start
        rows = np.ascontiguousarray(bits[:, offset - int(take) : offset])
        try:
            os.utime(self.root / f"{entry.stem}.ctx.npy")
        except OSError:
            pass
        return mx.array(rows).view(self.dtype)

    def save(self, tokens: Sequence[int], context: mx.array, end: int) -> bool:
        """Queue one entry for writing; returns False when it is already kept."""
        end = int(end)
        rows = int(context.shape[1])
        if rows <= 0 or end < rows or len(tokens) < end:
            return False
        stem = f"{_token_key(tokens[:end])}-{end}-{rows}"
        with self._lock:
            if stem in self._entries:
                return False
        # The host copy waits for the GPU; the file write runs off the
        # inference thread.
        bits = np.array(context.astype(self.dtype).view(_BITS[self.dtype]))
        toks = np.asarray(tokens[:end], dtype=np.int32)
        self._writer.submit(self._write, stem, end, rows, bits, toks)
        return True

    def _write(self, stem: str, end: int, rows: int, bits, toks) -> None:
        try:
            for suffix, data in ((".tok.npy", toks), (".ctx.npy", bits)):
                tmp = self.root / f"{stem}{suffix}.tmp"
                with open(tmp, "wb") as f:
                    np.save(f, data)
                os.replace(tmp, self.root / f"{stem}{suffix}")
        except OSError as exc:
            logger.warning("DFlash context store write failed: %r", exc)
            return
        with self._lock:
            self._entries[stem] = _Entry(stem, end, rows)
            self._evict()

    def _evict(self) -> None:
        if len(self._entries) <= self.max_entries:
            return

        def used(stem):
            try:
                return os.path.getmtime(self.root / f"{stem}.ctx.npy")
            except OSError:
                return 0.0

        for stem in sorted(self._entries, key=used)[: len(self._entries) - self.max_entries]:
            self._entries.pop(stem, None)
            for suffix in (".ctx.npy", ".tok.npy"):
                try:
                    os.remove(self.root / f"{stem}{suffix}")
                except OSError:
                    pass

    def flush(self) -> None:
        """Wait for queued writes (tests, shutdown)."""
        self._writer.submit(lambda: None).result()


def namespace_dir(root: Path, parts: Sequence[str]) -> Path:
    digest = hashlib.sha1("\x00".join(parts).encode()).hexdigest()[:16]
    return Path(root) / digest
