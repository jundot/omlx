# SPDX-License-Identifier: Apache-2.0
"""HuggingFace model downloader for oMLX admin panel.

Downloads models with huggingface_hub's snapshot_download, filesystem-based
progress polling fed by both disk allocation and the transfer bar's network
byte counts, and an isolated HTTP retry for xet transport failures.

The task queue persists to a JSON file beside settings.json, so a restart
re-queues interrupted downloads (resume is snapshot_download's own "skip
finalized files" behaviour), and a row that is still pending or downloading
keeps the credential the request supplied so a gated download resumes with
it. Cancelling aborts every recorded xet download group, so a shard-parallel
Rust transfer stops immediately instead of running to completion — including
a group the call only opens after the cancel, once abort_xet_session() has
already replaced its session.
"""

import asyncio
import enum
import errno
import json
import logging
import os
import shutil
import signal
import sys
import threading
import time
from collections import deque
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)
from huggingface_hub.utils import tqdm as _hf_tqdm

# Private-module import; the pyproject floor (huggingface-hub>=1.19.0)
# guarantees it exists. Re-verify the symbol when bumping the hub version.
from huggingface_hub.utils._xet import abort_xet_session

logger = logging.getLogger(__name__)

# Timeout for HuggingFace API calls (seconds).
# Prevents server from hanging when HF is unreachable.
_HF_API_TIMEOUT = 10

# Seconds with no payload activity (disk writes or wire bytes) before
# considering the download stalled.
_STALL_TIMEOUT = 300

# A separate first-activity deadline catches xet token/CAS hangs that never
# move the wire counter or create a payload file. Once any payload activity
# is observed, the more tolerant stall timeout above applies so slow but
# active connections are left alone.
_STARTUP_STALL_TIMEOUT = 120

_PROGRESS_POLL_INTERVAL = 0.5
_SUBPROCESS_TERMINATE_TIMEOUT = 5

# Seconds of byte-count history the reported speed averages over. One second
# makes the readout a true per-second mean: a single interval reads as bursts
# (0 between files, double during one) and an exponential average never
# forgets the past. The display trails reality by at most one window plus one
# poll interval, and stops moving only when the transfer stops.
_SPEED_WINDOW = 1.0


class _SlidingRate:
    """Mean bytes/second over a sliding window of cumulative byte samples.

    Subclasses turn their own signal into a monotone cumulative count and
    hand it to :meth:`_record`; the window arithmetic (and therefore the
    smoothing every readout uses) lives here once.
    """

    def __init__(self, window: float = _SPEED_WINDOW) -> None:
        self.window = window
        self._samples: deque[tuple[float, float]] = deque()

    def _record(self, total: float, now: float) -> float:
        self._samples.append((now, float(total)))
        # Retire samples the window has outgrown, but keep one older sample
        # as the anchor so every reading spans (at least) a full window.
        while len(self._samples) > 2 and self._samples[1][0] < now - self.window:
            self._samples.popleft()
        if len(self._samples) < 2:
            return 0.0
        start_t, start_b = self._samples[0]
        end_t, end_b = self._samples[-1]
        span = end_t - start_t
        if span <= 0:
            return 0.0
        return (end_b - start_b) / span


class _SpeedMeter(_SlidingRate):
    """Sliding-window bytes/second from per-file allocation continuity.

    A sample is the path -> allocated-bytes mapping from one directory walk.
    Only positive growth of paths that were also present in the previous
    sample counts as transfer:

    - A path first seen at non-zero size contributes nothing (a file moved or
      copied in wholesale, a tree that vanished and came back, a walk that
      aborted early and came back partial).
    - A vanished path is forgotten, so if it reappears it starts from zero
      instead of replaying its old size as fresh growth.
    - A truncated path lowers its baseline, so the refill counts once, from
      the new size.

    Both hubs stage inside the target directory, so a real transfer always
    grows paths that are already under watch.
    """

    def __init__(self, window: float = _SPEED_WINDOW) -> None:
        super().__init__(window)
        self._prev: dict[str, int] = {}

    def add(self, files: dict, now: float | None = None) -> float:
        """Record a per-file sample and return the mean rate over the window."""
        now = time.monotonic() if now is None else now
        growth = 0
        prev = self._prev
        for path, allocated in files.items():
            before = prev.get(path)
            if before is not None and allocated > before:
                growth += allocated - before
        # Replace rather than merge: forgotten paths must not carry a stale
        # baseline across a disappearance, and a truncated file registers
        # its new, smaller size as the baseline.
        self._prev = dict(files)
        total = (self._samples[-1][1] if self._samples else 0.0) + growth
        return self._record(total, now)


_NON_XET_WORKER_MODULE = "omlx._hf_download_worker"

_XET_ERROR_MARKERS = (
    "cas service error",
    "cas client",
    "hf_xet",
    "reqwestmiddleware",
    "xet-read-token",
    "xet storage",
    "/xet-",
)

# Cache of (configured_endpoint -> resolved_endpoint) so we only probe each
# endpoint once per process lifetime. Mirrors like hf-mirror.com permanently
# 308-redirect to huggingface.co when accessed from IPs outside their region;
# huggingface_hub does NOT follow those cross-origin 308s during HEAD probes,
# so downloads fail. We resolve the redirect chain upfront and pin HfApi to
# the final origin.
_endpoint_resolution_cache: dict[str, str] = {}


def _resolve_endpoint(endpoint: str) -> str:
    """Follow permanent (301/308) cross-origin redirects on `endpoint`.

    Returns the final origin (scheme://host[:port]) the endpoint resolves to.
    Used to work around `huggingface_hub`'s inability to follow cross-origin
    308 redirects during file-download HEAD probes.

    Probes a known-stable HF API path (`/api/models/gpt2`) with HEAD; if the
    server returns a 301/308 with a Location pointing at a different host,
    the redirected origin is returned (and cached). Network errors fall back
    to the original endpoint.
    """
    endpoint = endpoint.rstrip("/")
    if endpoint in _endpoint_resolution_cache:
        return _endpoint_resolution_cache[endpoint]

    try:
        import httpx
    except ImportError:
        return endpoint

    probe = f"{endpoint}/api/models/gpt2"
    original_host = urlparse(endpoint).netloc
    resolved = endpoint
    try:
        with httpx.Client(follow_redirects=False, timeout=5.0) as client:
            r = client.head(probe)
            # Walk up to 3 permanent hops; stop on first non-permanent status.
            hops = 0
            current_url = probe
            while r.status_code in (301, 308) and "location" in r.headers:
                hops += 1
                if hops > 3:
                    break
                location = r.headers["location"]
                if location.startswith("/"):
                    # Relative redirect — same origin, no rewrite needed.
                    break
                target = urlparse(location)
                if not target.netloc:
                    break
                if target.netloc != original_host:
                    # Cross-origin permanent redirect: rewrite the endpoint.
                    port = f":{target.port}" if target.port else ""
                    resolved = f"{target.scheme}://{target.hostname}{port}"
                    original_host = target.netloc
                current_url = location
                r = client.head(current_url)
    except Exception as e:  # noqa: BLE001 — probe is best-effort
        logger.debug(f"HF endpoint probe failed for {endpoint}: {e}")
        return endpoint

    if resolved != endpoint:
        logger.info(
            f"HuggingFace endpoint {endpoint} permanently redirects to "
            f"{resolved}; using resolved origin for downloads."
        )
    _endpoint_resolution_cache[endpoint] = resolved
    return resolved


class _DownloadCancelled(Exception):
    """Raised inside the download thread to interrupt a cancelled download."""


class _DownloadStalledError(RuntimeError):
    """Raised when a monitored transfer stops making observable progress."""

    def __init__(self, *, transport: str, phase: str, timeout: float):
        self.transport = transport
        self.phase = phase
        self.timeout = timeout
        if phase == "startup":
            detail = "no initial download activity"
        else:
            detail = "no download progress"
        super().__init__(f"{transport} download stalled: {detail} for {timeout:g}s")


class _NonXetDownloadError(RuntimeError):
    """Raised when the isolated HTTP fallback process fails."""


@dataclass(frozen=True)
class _DownloadActivity:
    """Filesystem signals used to distinguish slow transfers from stalls."""

    file_count: int = 0
    logical_size: int = 0
    allocated_size: int = 0
    latest_mtime_ns: int = 0
    # Per-file allocated bytes (path -> blocks), so the speed meter can
    # follow file-level continuity instead of one aggregate number that
    # jumps whenever files appear or vanish between walks.
    files: dict = field(default_factory=dict)


def _is_xet_transport_error(error: BaseException) -> bool:
    """Return whether an exception identifies the xet/CAS transport path."""
    if isinstance(
        error,
        (
            _DownloadCancelled,
            _DownloadStalledError,
            GatedRepoError,
            RepositoryNotFoundError,
        ),
    ):
        return False
    if isinstance(error, OSError) and error.errno in {
        errno.EACCES,
        errno.ENOSPC,
        errno.EROFS,
    }:
        return False
    detail = f"{type(error).__name__}: {error}".lower()
    return any(marker in detail for marker in _XET_ERROR_MARKERS)


class _WireCounter:
    """Cumulative network bytes reported by huggingface_hub's progress bars.

    The xet transfer bar (``snapshot_download.transfer``) receives each
    ``transfer_inc`` as it arrives from the network, while the filesystem
    meter only sees bytes once buffered chunks are flushed to disk — during
    xet's fetch phase that lag can be hundreds of MB of wire traffic
    (``min_reconstruction_fetch_size``, 256MB) showing as 0 B/s on disk.
    Feeding these callbacks into a second window keeps the readout live
    during exactly the stretches where the disk meter reads zero.
    """

    __slots__ = ("_value", "_lock")

    def __init__(self) -> None:
        self._value = 0
        self._lock = threading.Lock()

    def add(self, n: int) -> None:
        if n <= 0:
            return
        with self._lock:
            self._value += int(n)

    @property
    def value(self) -> int:
        with self._lock:
            return self._value


class _WireSpeedMeter(_SlidingRate):
    """Sliding-window bytes/second over a cumulative wire-byte counter.

    Same window semantics as ``_SpeedMeter`` (the readout must not switch
    smoothing between the fetch and reconstruction phases): the mean rate
    over the last ``window`` seconds, so it moves within one poll interval
    of any change and settles to zero once the counter stops advancing.
    """

    def add(self, total: int, now: float | None = None) -> float:
        """Record a cumulative sample and return the mean rate over the window."""
        now = time.monotonic() if now is None else now
        return self._record(max(0.0, float(total)), now)


def _make_cancellable_tqdm(
    should_cancel: Callable[[], bool],
    on_wire_bytes: Callable[[int], None] | None = None,
) -> type:
    """Build a tqdm subclass that aborts the download when cancelled.

    huggingface_hub's http_get calls ``progress.update(len(chunk))`` once per
    downloaded chunk (DOWNLOAD_CHUNK_SIZE, 10MB). A running thread can't be
    force-stopped and snapshot_download takes no cancel token, so we cooperate
    from the progress callback: raising here unwinds the download thread
    cleanly within one chunk, releasing its buffers and connection.

    Note: this only interrupts the Python http_get path, which xet-less repos
    and mirror endpoints still use. On the xet path the Rust side defers a
    callback exception until the whole transfer finishes (issue #1322), so
    cancellation there is driven by ``abort_xet_session()`` instead; this
    class is kept as the raise-on-next-chunk backstop for http_get.

    When ``on_wire_bytes`` is given, snapshot_download's transfer bar
    (``bar_format=XET_TRANSFER_BAR_FORMAT``, which — unlike the reconstruction
    bar and the default per-file/meta formats — carries no ``{total_fmt}``
    because network byte totals are unpredictable) reports each increment as
    it arrives from the network. Every other bar in the pipeline (disk
    reconstruction bytes, per-file counters, the meta file-count bar) only
    participates in cancellation, so wire and disk bytes are never summed
    twice.
    """

    class _CancellableTqdm(_hf_tqdm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            fmt = kwargs.get("bar_format")
            # After init, tqdm keeps bar_format as passed (None for the
            # default format), so classify from the resolved attribute too.
            if fmt is None:
                fmt = getattr(self, "bar_format", None)
            self._wire_bar = (
                on_wire_bytes is not None
                and isinstance(fmt, str)
                and "total_fmt" not in fmt
            )

        def update(self, n=1):
            if self._wire_bar and n:
                try:
                    on_wire_bytes(int(n))
                except Exception:  # noqa: BLE001 — metering must never break a transfer
                    logger.debug("wire-byte hook failed", exc_info=True)
            if should_cancel():
                raise _DownloadCancelled()
            return super().update(n)

    return _CancellableTqdm


def _get_hf_api() -> tuple[HfApi, str | None]:
    """Create HfApi instance with configured endpoint.

    Only the admin UI's `huggingface.endpoint` setting is honored here.
    When that's empty, return `HfApi()` with no explicit endpoint so
    `huggingface_hub` falls back to its own resolution (which already
    honors the `HF_ENDPOINT` env var). The configured endpoint, when
    present, is run through `_resolve_endpoint()` to follow permanent
    cross-origin redirects (e.g. hf-mirror.com → huggingface.co from
    non-CN IPs) so downstream HF library code sees a stable origin.

    Returns:
        Tuple of (HfApi instance, endpoint URL or None).
    """
    endpoint: str | None = None
    try:
        from ..settings import get_settings

        endpoint = get_settings().huggingface.endpoint or None
    except (RuntimeError, AttributeError):
        endpoint = None

    if endpoint:
        resolved = _resolve_endpoint(endpoint)
        return HfApi(endpoint=resolved), resolved
    return HfApi(), None


def _list_models_stale_token_fallback(api: HfApi, kwargs: dict) -> tuple[list, bool]:
    """Drain list_models, retrying anonymously when the stored token is rejected.

    huggingface_hub attaches the locally stored credential (HF_TOKEN env var or
    the hf auth login token file) to every request, so a stale token 401s even
    the public model listing (#2276, #2310). Listing needs no auth, so retry
    once with token=False and report the rejected token to the caller.
    """
    try:
        return list(api.list_models(**kwargs)), False
    except HfHubHTTPError as e:
        if e.response is None or e.response.status_code != 401:
            raise
        logger.warning(
            "HF model listing rejected the stored token (401): %s. "
            "Retrying anonymously.",
            e,
        )
        return list(api.list_models(token=False, **kwargs)), True


class DownloadStatus(str, enum.Enum):
    """Status of a download task."""

    PENDING = "pending"
    DOWNLOADING = "downloading"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class DownloadTask:
    """Represents a single model download task."""

    task_id: str
    repo_id: str
    status: DownloadStatus = DownloadStatus.PENDING
    progress: float = 0.0
    total_size: int = 0
    downloaded_size: int = 0
    # Smoothed transfer rate in bytes/second over the last poll intervals.
    # 0.0 whenever the task is not actively transferring.
    speed_bps: float = 0.0
    error: str = ""
    # Request-supplied credential (HF or MS token). Kept for restart
    # recovery only: _write_tasks_file persists it (owner-only file) so a
    # restored row resumes private/gated downloads with the credentials
    # that queued them. Deliberately absent from to_dict() — the queue API
    # must never hand tokens back to clients.
    token: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0
    retry_count: int = 0

    def to_dict(self) -> dict:
        """Serialize task to a JSON-compatible dict."""
        return {
            "task_id": self.task_id,
            "repo_id": self.repo_id,
            "status": self.status.value,
            "progress": round(self.progress, 1),
            "total_size": self.total_size,
            "downloaded_size": self.downloaded_size,
            "speed_bps": round(self.speed_bps, 1),
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "retry_count": self.retry_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DownloadTask":
        """Rebuild a task from :meth:`to_dict` output.

        Used to restore rows after a restart. An unknown status value (a row
        written by a newer build) folds into FAILED so the entry still
        displays and stays retryable instead of crashing the queue load.
        """
        task = cls(
            task_id=str(data["task_id"]),
            repo_id=str(data["repo_id"]),
        )
        try:
            # A missing status is as unknown as one this build does not
            # recognise: it folds into FAILED rather than reading as a queue
            # row this boot never started and the next boot then starts.
            task.status = DownloadStatus(data.get("status") or "")
        except ValueError:
            task.status = DownloadStatus.FAILED
        task.progress = float(data.get("progress", 0.0))
        task.total_size = int(data.get("total_size", 0))
        task.downloaded_size = int(data.get("downloaded_size", 0))
        # Nothing transfers at boot: a restored row always reads 0 B/s.
        task.speed_bps = 0.0
        task.error = str(data.get("error", ""))
        # Credential for a later retry/resume; rows written before the
        # token field existed simply fall back to hub's env/login lookup.
        task.token = str(data.get("token") or "")
        task.created_at = float(data.get("created_at") or 0.0) or task.created_at
        task.started_at = float(data.get("started_at") or 0.0)
        task.completed_at = float(data.get("completed_at") or 0.0)
        task.retry_count = int(data.get("retry_count") or 0)
        return task


_DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1,
    "U64": 8, "U32": 4, "U16": 2, "U8": 1,
    "BOOL": 1,
}

# Minimum downloads to be included in recommendations.
_MIN_DOWNLOADS = 100

# Follow-up model_info(files_metadata=True) calls for U32-packed repos.
_BLOB_FETCH_CONCURRENCY = 8
_BLOB_FETCH_BATCH_TIMEOUT = 15
_BLOB_SIZE_CACHE_TTL = 3600.0
_BLOB_SIZE_CACHE_MAX = 512
_blob_size_cache: dict[str, tuple[int, float]] = {}


def _safetensors_parameters(safetensors) -> dict:
    """Return the dtype → count map from HF safetensors metadata."""
    if not safetensors:
        return {}
    if isinstance(safetensors, dict):
        params = safetensors.get("parameters") or {}
        return params if isinstance(params, dict) else {}
    params = getattr(safetensors, "parameters", None)
    return params if isinstance(params, dict) else {}


def _histogram_has_packed_u32(safetensors) -> bool:
    """True when HF reports U32 weights — packed MLX quants, not 4-byte params."""
    return "U32" in _safetensors_parameters(safetensors)


def _sum_safetensors_blob_bytes(siblings) -> int | None:
    """Sum current-revision *.safetensors blob sizes from model_info siblings.

    Returns None when siblings are missing or have no sizes (list_models does
    not populate them). Does not use usedStorage, which includes old revisions.
    """
    if not isinstance(siblings, (list, tuple)):
        return None
    total = 0
    saw_size = False
    for sibling in siblings:
        name = getattr(sibling, "rfilename", None) or ""
        if not str(name).endswith(".safetensors"):
            continue
        try:
            size = int(getattr(sibling, "size", None) or 0)
        except (TypeError, ValueError):
            continue
        if size > 0:
            total += size
            saw_size = True
    return total if saw_size else None


def _fetch_safetensors_blob_bytes(api: HfApi, repo_id: str) -> int:
    """Look up *.safetensors blob bytes for one repo. 0 on failure."""
    try:
        info = api.model_info(
            repo_id, files_metadata=True, timeout=_HF_API_TIMEOUT
        )
    except HfHubHTTPError as e:
        if e.response is None or e.response.status_code != 401:
            logger.debug("Could not fetch blob sizes for %s: %s", repo_id, e)
            return 0
        try:
            info = api.model_info(
                repo_id, files_metadata=True, token=False, timeout=_HF_API_TIMEOUT
            )
        except Exception as retry_error:
            logger.debug(
                "Could not fetch blob sizes for %s: %s", repo_id, retry_error
            )
            return 0
    except Exception as e:
        logger.debug("Could not fetch blob sizes for %s: %s", repo_id, e)
        return 0
    return _sum_safetensors_blob_bytes(getattr(info, "siblings", None)) or 0


def _cached_blob_size(repo_id: str) -> int | None:
    """Return a still-fresh cached positive blob size, or None."""
    entry = _blob_size_cache.get(repo_id)
    if entry is None:
        return None
    size, stored_at = entry
    if time.monotonic() - stored_at > _BLOB_SIZE_CACHE_TTL:
        _blob_size_cache.pop(repo_id, None)
        return None
    return size


def _prune_blob_size_cache() -> None:
    """Drop expired entries, then the oldest if still over the cap."""
    now = time.monotonic()
    expired = [
        repo_id
        for repo_id, (_, stored_at) in _blob_size_cache.items()
        if now - stored_at > _BLOB_SIZE_CACHE_TTL
    ]
    for repo_id in expired:
        _blob_size_cache.pop(repo_id, None)
    overflow = len(_blob_size_cache) - _BLOB_SIZE_CACHE_MAX
    if overflow <= 0:
        return
    oldest = sorted(_blob_size_cache.items(), key=lambda item: item[1][1])
    for repo_id, _ in oldest[:overflow]:
        _blob_size_cache.pop(repo_id, None)


def _store_blob_size(repo_id: str, size: int) -> None:
    """Cache Hub blob sizes that actually resolved. Failures stay uncached."""
    if size > 0:
        _blob_size_cache[repo_id] = (size, time.monotonic())
        _prune_blob_size_cache()


async def _blob_bytes_for_repos(api: HfApi, repo_ids: list[str]) -> dict[str, int]:
    """Fetch blob sizes for U32-packed repos, concurrently, with a batch cap."""
    unique_ids = list(dict.fromkeys(repo_ids))
    if not unique_ids:
        return {}

    sizes: dict[str, int] = {}
    to_fetch: list[str] = []
    for repo_id in unique_ids:
        cached = _cached_blob_size(repo_id)
        if cached is not None:
            sizes[repo_id] = cached
        else:
            to_fetch.append(repo_id)
    if not to_fetch:
        return sizes

    semaphore = asyncio.Semaphore(_BLOB_FETCH_CONCURRENCY)

    async def _one(repo_id: str) -> tuple[str, int]:
        async with semaphore:
            try:
                size = await asyncio.to_thread(
                    _fetch_safetensors_blob_bytes, api, repo_id
                )
                return repo_id, size
            except Exception:
                return repo_id, 0

    tasks = [asyncio.create_task(_one(repo_id)) for repo_id in to_fetch]
    done, pending = await asyncio.wait(tasks, timeout=_BLOB_FETCH_BATCH_TIMEOUT)
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
        logger.warning(
            "Timed out fetching safetensors blob sizes for %d U32-packed repo(s)",
            len(pending),
        )

    for repo_id in to_fetch:
        sizes.setdefault(repo_id, 0)
    for task in done:
        try:
            repo_id, size = task.result()
        except Exception:
            continue
        sizes[repo_id] = size
        _store_blob_size(repo_id, size)
    return sizes


def _needs_u32_blob_fetch(model) -> bool:
    """True when the listing has a U32 histogram but no sibling blob sizes."""
    if not _histogram_has_packed_u32(getattr(model, "safetensors", None)):
        return False
    return _sum_safetensors_blob_bytes(getattr(model, "siblings", None)) is None


def _list_disk_size(model, blob_sizes: dict[str, int]) -> int:
    """On-disk size for a list_models row: dtype formula, or U32 blob bytes."""
    safetensors = getattr(model, "safetensors", None)
    params = _safetensors_parameters(safetensors)
    if not params:
        return 0
    if "U32" in params:
        blob = _sum_safetensors_blob_bytes(getattr(model, "siblings", None))
        if blob is not None:
            return blob
        return int(blob_sizes.get(getattr(model, "id", ""), 0) or 0)
    return _calc_safetensors_disk_size({"parameters": params})


def _calc_safetensors_disk_size(safetensors: dict) -> int:
    """Estimate disk size from a dtype histogram.

    Accurate for BF16/F16/F32. Wrong for U32-packed MLX quants: HF reports
    logical parameter counts under U32, not packed word counts (#3401).

    Malformed counts (None, non-int) return 0 rather than raising, so one
    bad Hub row cannot take down Browse/Search/Recommended.
    """
    params = safetensors.get("parameters", {}) or {}
    if not params:
        return 0
    try:
        return sum(
            int(count) * _DTYPE_BYTES.get(dtype, 1)
            for dtype, count in params.items()
        )
    except (TypeError, ValueError):
        return 0


def _format_model_size(size_bytes: int) -> str:
    """Format model size in bytes to a human-readable string."""
    if size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.1f} MB"
    else:
        return f"{size_bytes / 1024**3:.1f} GB"


def _format_param_count(total_params: int) -> str:
    """Format parameter count to a human-readable string (e.g., 7.0B, 13.0B)."""
    if total_params >= 1e12:
        return f"{total_params / 1e12:.1f}T"
    if total_params >= 1e9:
        return f"{total_params / 1e9:.1f}B"
    if total_params >= 1e6:
        return f"{total_params / 1e6:.1f}M"
    return str(total_params)


def _get_param_count(safetensors: dict) -> int:
    """Get total parameter count from safetensors metadata."""
    params = safetensors.get("parameters", {}) or {}
    if not params:
        return 0
    try:
        return sum(int(count) for count in params.values())
    except (TypeError, ValueError):
        return 0


# HF API sort field mapping for search.
_SORT_MAP = {
    "trending": "trendingScore",
    "downloads": "downloads",
    "created": "createdAt",
    "updated": "lastModified",
    "most_params": "downloads",  # fetch by downloads, re-sort in Python
    "least_params": "downloads",  # fetch by downloads, re-sort in Python
    "largest": "downloads",  # fetch by downloads, re-sort by size in Python
    "smallest": "downloads",  # fetch by downloads, re-sort by size in Python
}


def _write_tasks_file(path: Path, tasks) -> None:
    """Atomically persist task rows for restart recovery.

    Written owner-only (mode 0600) via a temp file + rename, so a crash
    mid-write never leaves a truncated queue behind and no other user can
    read the rows while they are being written. Callers gate on their own
    lifecycle flags (restore/shutdown) before getting here.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = []
        for task in sorted(tasks, key=lambda t: t.created_at):
            row = task.to_dict()
            # to_dict() never carries the credential (the API must not echo
            # it), and neither does a finished row: only a row that can
            # still be resumed needs the token the request supplied. A
            # terminal row keeps it in memory for a retry in this process
            # but is written without it, so tokens do not outlive the
            # download that needed them.
            row["token"] = (
                task.token
                if task.status
                in (DownloadStatus.PENDING, DownloadStatus.DOWNLOADING)
                else ""
            )
            payload.append(row)
        tmp_path = path.with_name(path.name + ".tmp")
        fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        # O_CREAT's mode only applies when this call creates the file, and the
        # temp name is reused, so an earlier writer's mode would stand.
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload))
        tmp_path.replace(path)
    except OSError as exc:
        logger.warning(
            "Could not persist download tasks to %s: %s", path, exc
        )


def _read_tasks_file(path: Path | None) -> list[dict]:
    """Read persisted task rows; a missing or corrupt file yields []."""
    if path is None or not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning(
            "Could not read persisted download tasks from %s: %s", path, exc
        )
        return []
    return data if isinstance(data, list) else []


def _restore_created_at(entry: dict) -> float:
    """Order persisted rows oldest first, and carry their queue position over.

    A row a build that stored the field differently left behind reads as 0.0
    instead of raising: inside ``sorted`` a single bad value would abort the
    whole restore, and outside it the assignment would drop every row behind
    it as well as the healing rewrite that follows.
    """
    try:
        return float(entry.get("created_at") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _restore_retry_count(entry: dict) -> int:
    """A retry count this build cannot read reads as none rather than raising."""
    try:
        return int(entry.get("retry_count") or 0)
    except (TypeError, ValueError):
        return 0

# Every xet download group still in flight. snapshot_download shards files
# across hf_thread_map workers, so one snapshot runs many xet_get() calls
# concurrently — one XetFileDownloadGroup per file — and cancel/shutdown must
# abort all of them. Groups register on creation and unregister when their
# `with` block settles, so this registry only holds live transfers.
_xet_groups: list = []
# One token per snapshot_download call that is running right now, plus the
# subset of those calls whose transfer was already aborted. A cancel that
# lands before the call reaches its first xet_get() finds an empty registry,
# and abort_xet_session() cannot cover the gap either: it only drops the
# session, so that call's next get_xet_session() builds a fresh one and the
# group opened on it would never be aborted. Each call registers a token
# before its work starts and drops it when its worker returns.
_xet_live_calls: set = set()
_xet_aborted_calls: set = set()
# The call the current thread is downloading for. hub opens the xet groups on
# its per-file worker threads, so _install_xet_call_marking marks those too;
# attributing a group this way keeps an abort from reaching a download that
# started after the cancel while the cancelled worker is still unwinding.
_xet_call_in_thread = threading.local()
_xet_group_lock = threading.Lock()


def _open_xet_call() -> object:
    """Register one running snapshot_download call; see _xet_live_calls."""
    call = object()
    with _xet_group_lock:
        _xet_live_calls.add(call)
    return call


def _close_xet_call(call: object) -> None:
    """Drop a settled call and any abort flag it was still carrying."""
    with _xet_group_lock:
        _xet_live_calls.discard(call)
        _xet_aborted_calls.discard(call)


def _abort_xet_call(call: object) -> None:
    """Flag one live call: every group it opens from now on is aborted."""
    with _xet_group_lock:
        if call in _xet_live_calls:
            _xet_aborted_calls.add(call)


def _mark_xet_call(call) -> None:
    """Make ``call`` own the xet groups opened on this thread."""
    _xet_call_in_thread.call = call


def _register_xet_group(group) -> bool:
    """Record an in-flight xet download group (identity-deduplicated).

    Returns True when the group belongs to a call that was already aborted;
    its caller then aborts the group instead of leaving it registered.
    Testing the flag and registering share one lock, so a cancel landing in
    between still leaves the group reachable: it is either in the registry
    when that abort snapshots it, or aborted on the spot here.
    """
    call = getattr(_xet_call_in_thread, "call", None)
    with _xet_group_lock:
        if call is not None and call in _xet_aborted_calls:
            return True
        if not any(g is group for g in _xet_groups):
            _xet_groups.append(group)
        return False


def _unregister_xet_group(group) -> None:
    """Drop a group whose download settled; a miss means abort ran first."""
    with _xet_group_lock:
        _xet_groups[:] = [g for g in _xet_groups if g is not group]


def _abort_xet_group(group) -> bool:
    """Abort one group. True if the abort was accepted."""
    try:
        group.abort()
        return True
    except Exception:  # pragma: no cover - stale or already-finished
        logger.debug("xet group abort failed", exc_info=True)
        return False


def _abort_xet_transfers() -> bool:
    """Abort every active recorded xet group. True if one was aborted."""
    with _xet_group_lock:
        groups = list(_xet_groups)
        _xet_groups.clear()
        # A call that has not opened a group yet is unreachable through the
        # registry; flag it so the group it opens later is aborted on sight.
        _xet_aborted_calls.update(_xet_live_calls)
    aborted = False
    for group in groups:
        if _abort_xet_group(group):
            aborted = True
    return aborted


def _tracked_snapshot_download(func, is_cancelled, /, **kwargs):
    """Run one snapshot_download call, tracking it for post-abort groups.

    The call is registered while its worker runs and dropped in a
    ``finally``, so the flag can never outlive the only worker that could
    still open a group for it. ``is_cancelled`` is re-read before the call
    starts: a cancel that landed before this worker registered would not
    have flagged it, and the call would otherwise recreate the session and
    download into the void.
    """
    call = _open_xet_call()
    previous = getattr(_xet_call_in_thread, "call", None)
    _mark_xet_call(call)
    try:
        if is_cancelled():
            _abort_xet_call(call)
        return func(**kwargs)
    finally:
        _mark_xet_call(previous)
        _close_xet_call(call)


def _install_xet_call_marking() -> None:
    """Attribute hub's per-file download threads to the call they serve.

    snapshot_download maps its files across a ThreadPoolExecutor, so the xet
    groups are opened on threads other than the one that owns the call.
    Wrapping hub's thread-map entry point re-marks each worker for the
    duration of one file, which is what lets a group be attributed to the
    transfer that owns it. Best effort, like the group capture above: without
    the marker a group is unattributed and only the registry can abort it, so
    downloads keep working either way.
    """
    try:
        from huggingface_hub import _snapshot_download as hub_snapshot

        original = hub_snapshot.hf_thread_map
        if getattr(original, "_omlx_call_marking", False):
            return

        def hf_thread_map_with_call_marking(fn, iterable, **kwargs):
            # Called on the download's own thread where the marker is set;
            # hub's workers only inherit it through the wrapper below.
            call = getattr(_xet_call_in_thread, "call", None)
            if call is None:
                return original(fn, iterable, **kwargs)

            def marked(item):
                previous = getattr(_xet_call_in_thread, "call", None)
                _mark_xet_call(call)
                try:
                    return fn(item)
                finally:
                    _mark_xet_call(previous)

            return original(marked, iterable, **kwargs)

        hf_thread_map_with_call_marking._omlx_call_marking = True
        hub_snapshot.hf_thread_map = hf_thread_map_with_call_marking
    except Exception:  # pragma: no cover - defensive against hub refactors
        logger.debug("xet call marking not installed", exc_info=True)


class _TrackedXetGroup:
    """Context-manager wrapper that keeps the active-group registry in sync.

    hub opens one group per file (``with session.new_file_download_group()``
    inside ``xet_get``) and snapshot_download runs those concurrently, so
    cancel must be able to reach every group that is still transferring.
    The proxy registers the group at creation; this wrapper unregisters it
    when the ``with`` block exits, so a settled group is never aborted
    later and an active one can never be missed.
    """

    __slots__ = ("_group",)

    def __init__(self, group) -> None:
        self._group = group

    def __enter__(self):
        try:
            entered = self._group.__enter__()
        except BaseException:
            # The `with` statement skips __exit__ when __enter__ raises,
            # so drop the registration here or a failed handshake would
            # leak a stale entry into the registry.
            _unregister_xet_group(self._group)
            raise
        # XetFileDownloadGroup.__enter__ returns itself; hand hub whatever
        # it yields so the real group drives the transfer.
        return self._group if entered is None else entered

    def __exit__(self, *exc_info):
        try:
            return self._group.__exit__(*exc_info)
        finally:
            _unregister_xet_group(self._group)

    def __getattr__(self, name: str):
        return getattr(self._group, name)


class _XetSessionProxy:
    """Delegating XetSession wrapper that records download groups.

    huggingface_hub exposes no cancellation hook for the transfers it
    starts, but ``XetFileDownloadGroup.abort()`` cancels the group's active
    downloads while leaving the session usable — the precise stop that
    ``sigint_abort()`` alone does not deliver for a long reconstruction
    (the Rust side only checks the session flag at task boundaries, so a
    ghost transfer can keep running minutes after a cancel). Every other
    session attribute passes through untouched, so uploads and future hub
    APIs keep working.
    """

    def __init__(self, inner) -> None:
        self._inner = inner

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    def new_file_download_group(self, *args, **kwargs):
        group = self._inner.new_file_download_group(*args, **kwargs)
        if _register_xet_group(group):
            # abort_xet_session() dropped the session this call's cancel
            # aborted, so the group it just opened on the fresh session is
            # the one that abort could not reach: stop it on sight.
            _abort_xet_group(group)
        return _TrackedXetGroup(group)


def _install_xet_group_capture() -> None:
    """Route get_xet_session() through the recording proxy (idempotent).

    hub imports ``get_xet_session`` by name from its module at call time,
    so replacing the module attribute takes effect for the next download.
    Best effort: if a future hub refactors the seam, downloads keep working
    and cancellation just falls back to ``abort_xet_session()``.
    """
    try:
        from huggingface_hub.utils import _xet as hub_xet

        original = hub_xet.get_xet_session
        if getattr(original, "_omlx_group_capture", False):
            return

        def get_xet_session_with_capture():
            return _XetSessionProxy(original())

        get_xet_session_with_capture._omlx_group_capture = True
        hub_xet.get_xet_session = get_xet_session_with_capture
    except Exception:  # pragma: no cover - defensive against hub refactors
        logger.debug("xet group capture not installed", exc_info=True)


class HFDownloader:
    """Manages HuggingFace model downloads with progress tracking.

    Uses huggingface_hub.snapshot_download() for actual downloads and polls
    the target directory size to estimate progress.

    Args:
        model_dir: Directory where downloaded models are stored.
        on_complete: Async callback invoked when a download completes successfully.
    """

    @staticmethod
    async def get_recommended_models(
        max_memory_bytes: int,
        limit: int = 60,
        result_limit: int = 50,
        mlx_only: bool = True,
    ) -> dict:
        """Fetch trending and popular models that fit in memory.

        Queries HuggingFace Hub for models, optionally restricted to
        mlx-community. Filtered by system memory capacity.

        Args:
            max_memory_bytes: Maximum model size in bytes (typically system memory).
            limit: Number of models to fetch per category from HF API.
            result_limit: Maximum number of models to return per category.
            mlx_only: If True, restrict to mlx-community author.

        Returns:
            Dict with 'trending' and 'popular' lists, plus 'hf_token_invalid'
            set when the stored HF token was rejected and the listing was
            fetched anonymously instead.
        """
        api, _endpoint = _get_hf_api()

        async def _list(sort: str) -> tuple[list, bool]:
            kwargs = {
                "sort": sort,
                "limit": limit,
                "expand": ["safetensors", "downloads", "likes", "trendingScore"],
            }
            if mlx_only:
                kwargs["author"] = "mlx-community"
            # list_models returns a lazy generator; drain it inside the worker
            # thread so the paginated HTTP calls never block the event loop.
            return await asyncio.wait_for(
                asyncio.to_thread(_list_models_stale_token_fallback, api, kwargs),
                timeout=_HF_API_TIMEOUT,
            )

        (trending_models, trending_rejected), (popular_models, popular_rejected) = (
            await asyncio.gather(
                _list("trendingScore"),
                _list("downloads"),
            )
        )

        def _eligible(model) -> bool:
            if not _safetensors_parameters(getattr(model, "safetensors", None)):
                return False
            return (model.downloads or 0) >= _MIN_DOWNLOADS

        need_blob: list[str] = []
        for models in (trending_models, popular_models):
            for model in models:
                if _eligible(model) and _needs_u32_blob_fetch(model):
                    need_blob.append(model.id)
        blob_sizes = await _blob_bytes_for_repos(api, need_blob)

        def _build(models) -> list[dict]:
            results = []
            for model in models:
                if not _eligible(model):
                    continue
                size = _list_disk_size(model, blob_sizes)
                if size <= 0 or size > max_memory_bytes:
                    continue
                params = _get_param_count(
                    {"parameters": _safetensors_parameters(model.safetensors)}
                )
                results.append(
                    {
                        "repo_id": model.id,
                        "name": model.id.split("/")[-1],
                        "downloads": model.downloads or 0,
                        "likes": model.likes or 0,
                        "trending_score": model.trending_score or 0,
                        "size": size,
                        "size_formatted": (
                            _format_model_size(size) if size > 0 else ""
                        ),
                        "params": params if params > 0 else None,
                        "params_formatted": (
                            _format_param_count(params) if params > 0 else None
                        ),
                    }
                )
            return results

        return {
            "trending": _build(trending_models)[:result_limit],
            "popular": _build(popular_models)[:result_limit],
            "hf_token_invalid": trending_rejected or popular_rejected,
        }

    @staticmethod
    async def search_models(
        query: str,
        sort: str = "trending",
        limit: int = 100,
        mlx_only: bool = True,
        # Filtering options
        min_params: Optional[int] = None,
        max_params: Optional[int] = None,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        # Sorting options
        sort_by_size: bool = False,
        sort_ascending: bool = False,
    ) -> dict:
        """Search HuggingFace models by query string with filtering and sorting.

        When mlx_only is True, results are restricted to the MLX library
        (same as https://huggingface.co/models?library=mlx).

        Args:
            query: Search query string.
            sort: Sort order (trending/downloads/created/updated/most_params/least_params/largest/smallest).
            limit: Maximum number of results to return.
            mlx_only: If True, restrict to MLX library models only.
            min_params: Minimum parameter count filter.
            max_params: Maximum parameter count filter.
            min_size: Minimum model size in bytes filter.
            max_size: Maximum model size in bytes filter.
            sort_by_size: Sort results by size instead of default sort.
            sort_ascending: Sort in ascending order (for size/params sorting).

        Returns:
            Dict with 'models' list and 'total' count, plus 'hf_token_invalid'
            set when the stored HF token was rejected and the listing was
            fetched anonymously instead.
        """
        api, _endpoint = _get_hf_api()

        # Determine base sort - for Python-side sorting, we fetch by downloads
        # which tends to return more results, then sort in Python
        if sort in ("most_params", "least_params", "largest", "smallest"):
            base_sort = "downloads"
        else:
            base_sort = _SORT_MAP.get(sort, "trendingScore")

        kwargs = {
            "search": query,
            "sort": base_sort,
            "limit": limit,
            "expand": ["safetensors", "downloads", "likes", "trendingScore"],
        }
        if mlx_only:
            kwargs["filter"] = "mlx"

        # list_models returns a lazy generator; drain it inside the worker
        # thread so the paginated HTTP calls never block the event loop.
        models, token_rejected = await asyncio.wait_for(
            asyncio.to_thread(_list_models_stale_token_fallback, api, kwargs),
            timeout=_HF_API_TIMEOUT,
        )

        pending: list[tuple[object, int | None, dict]] = []
        for m in models:
            params = None
            st_params = _safetensors_parameters(getattr(m, "safetensors", None))
            if st_params:
                params = _get_param_count({"parameters": st_params}) or None
            if min_params is not None and (params is None or params < min_params):
                continue
            if max_params is not None and (params is None or params > max_params):
                continue
            pending.append((m, params, st_params))

        need_blob = [m.id for m, _, _ in pending if _needs_u32_blob_fetch(m)]
        blob_sizes = await _blob_bytes_for_repos(api, need_blob)

        results = []
        for m, params, st_params in pending:
            size = 0
            params_formatted = None
            if st_params:
                params_formatted = (
                    _format_param_count(params) if params and params > 0 else None
                )
                size = _list_disk_size(m, blob_sizes)

            if min_size is not None and size < min_size:
                continue
            if max_size is not None and size > max_size:
                continue

            results.append(
                {
                    "repo_id": m.id,
                    "name": m.id,
                    "downloads": m.downloads or 0,
                    "likes": m.likes or 0,
                    "trending_score": m.trending_score or 0,
                    "size": size,
                    "size_formatted": _format_model_size(size) if size > 0 else "",
                    "params": params,
                    "params_formatted": params_formatted,
                }
            )

        # Apply Python-side sorting
        if sort == "most_params":
            results.sort(key=lambda x: x["params"] or 0, reverse=True)
        elif sort == "least_params":
            results.sort(key=lambda x: x["params"] or 0)
        elif sort in ("largest", "smallest") or sort_by_size:
            # Sort by size, putting unknown-size entries at the end
            results.sort(
                key=lambda x: x["size"] if x["size"] > 0 else -1,
                reverse=(sort == "largest" or (sort_by_size and not sort_ascending)),
            )
        # Otherwise, keep original HF API ordering (trending, downloads, created, updated)

        return {
            "models": results[:limit],
            "total": len(results),
            "hf_token_invalid": token_rejected,
        }

    @staticmethod
    async def get_model_info(repo_id: str) -> dict:
        """Fetch detailed model information from HuggingFace.

        Args:
            repo_id: HuggingFace repository ID (e.g., "mlx-community/Llama-3-8B-4bit").

        Returns:
            Dict with model details including description, files, tags, etc.
        """
        api, endpoint = _get_hf_api()
        info = await asyncio.wait_for(
            asyncio.to_thread(
                api.model_info,
                repo_id,
                files_metadata=True,
            ),
            timeout=_HF_API_TIMEOUT,
        )

        # Extract file list with sizes
        files = []
        if info.siblings:
            for s in info.siblings:
                files.append(
                    {
                        "name": s.rfilename,
                        "size": s.size or 0,
                        "size_formatted": (
                            _format_model_size(s.size) if s.size else ""
                        ),
                    }
                )

        # Detect LoRA/adapter repos (adapter_config.json is peft standard)
        is_adapter = any(f["name"] == "adapter_config.json" for f in files)

        # Params from the dtype histogram (logical count). Size from current
        # revision blob bytes — U32 packed quants are not 4 bytes/param (#3401).
        params = None
        params_formatted = None
        size = _sum_safetensors_blob_bytes(info.siblings) or 0
        safetensors = getattr(info, "safetensors", None)
        st_params = _safetensors_parameters(safetensors)
        if st_params:
            params = _get_param_count({"parameters": st_params})
            params_formatted = _format_param_count(params) if params > 0 else None
            if size == 0 and "U32" not in st_params:
                size = _calc_safetensors_disk_size({"parameters": st_params})

        # Fetch model card (README.md) content
        model_card = ""
        try:
            card_path = await asyncio.wait_for(
                asyncio.to_thread(
                    hf_hub_download,
                    repo_id=repo_id,
                    filename="README.md",
                    endpoint=endpoint,
                ),
                timeout=_HF_API_TIMEOUT,
            )
            if card_path:
                card_text = Path(card_path).read_text(encoding="utf-8")
                # Strip YAML front matter (between --- markers)
                if card_text.startswith("---"):
                    end = card_text.find("---", 3)
                    if end != -1:
                        card_text = card_text[end + 3:].strip()
                model_card = card_text
        except Exception:
            pass  # README not available

        return {
            "repo_id": info.id,
            "name": info.id,
            "model_card": model_card,
            "description": "",  # kept for backward compat
            "files": files,
            "tags": list(info.tags) if info.tags else [],
            "pipeline_tag": info.pipeline_tag or "",
            "params": params,
            "params_formatted": params_formatted,
            "size": size,
            "size_formatted": _format_model_size(size) if size > 0 else "",
            "downloads": info.downloads or 0,
            "likes": info.likes or 0,
            "created_at": info.created_at.isoformat() if info.created_at else "",
            "updated_at": (
                info.last_modified.isoformat() if info.last_modified else ""
            ),
            "is_adapter": is_adapter,
        }

    def __init__(
        self,
        model_dir: str,
        on_complete: Optional[Callable] = None,
        tasks_file: str | Path | None = None,
    ):
        self._model_dir = Path(model_dir)
        self._tasks: dict[str, DownloadTask] = {}
        self._active_tasks: dict[str, asyncio.Task] = {}
        self._progress_tasks: dict[str, asyncio.Task] = {}
        self._on_complete = on_complete
        self._cancelled: set[str] = set()
        # Where the queue persists for restart recovery; None disables
        # persistence (tests and embedders manage their own lifecycle).
        self._tasks_file = Path(tasks_file) if tasks_file else None
        # _persist() stays silent while either flag is set: shutdown must
        # leave on-disk rows reading "pending/downloading" so the next boot
        # resumes them, and restore must not rewrite the queue until every
        # interrupted row has been re-queued.
        self._shutting_down = False
        self._restoring = False
        self._stalled: dict[str, _DownloadStalledError] = {}
        self._fallback_processes: dict[str, asyncio.subprocess.Process] = {}
        # Network-byte counters fed by the transfer bar's progress callbacks;
        # a second speed source alongside the filesystem meter so xet's
        # fetch-but-don't-write phases don't read as 0 B/s.
        self._wire_counters: dict[str, _WireCounter] = {}
        self._download_sem = asyncio.Semaphore(1)

    @property
    def model_dir(self) -> Path:
        return self._model_dir

    def update_model_dir(self, new_dir: str) -> None:
        """Update the model directory path."""
        self._model_dir = Path(new_dir)

    async def start_download(
        self, repo_id: str, hf_token: str = ""
    ) -> DownloadTask:
        """Start downloading a model from HuggingFace.

        Args:
            repo_id: HuggingFace repository ID (e.g., "mlx-community/Llama-3-8B-4bit").
            hf_token: Optional HuggingFace token for gated models.

        Returns:
            The created DownloadTask.

        Raises:
            ValueError: If repo_id format is invalid or download is already queued.
        """
        repo_id = repo_id.strip()
        if "/" not in repo_id or len(repo_id.split("/")) != 2:
            raise ValueError(
                f"Invalid repository ID: '{repo_id}'. "
                "Expected format: 'owner/model' (e.g., 'mlx-community/Llama-3-8B-4bit')"
            )

        # Check for duplicate active downloads
        for task in self._tasks.values():
            if task.repo_id == repo_id and task.status in (
                DownloadStatus.PENDING,
                DownloadStatus.DOWNLOADING,
            ):
                raise ValueError(
                    f"Download for '{repo_id}' is already in progress"
                )

        task_id = str(uuid.uuid4())
        task = DownloadTask(task_id=task_id, repo_id=repo_id)
        # Keep the request-supplied credential with the row so a restart
        # can resume gated downloads (the API never returns it).
        task.token = hf_token or ""
        self._tasks[task_id] = task

        # Start download in background
        self._active_tasks[task_id] = asyncio.create_task(
            self._run_download(task_id, hf_token)
        )
        # The queued row must reach disk before anything can crash it, so a
        # restart re-queues this download instead of dropping it.
        self._persist()

        logger.info(f"Download queued: {repo_id} (task_id={task_id})")
        return task

    async def cancel_download(self, task_id: str) -> bool:
        """Cancel an active download.

        Args:
            task_id: The task ID to cancel.

        Returns:
            True if the task was found and cancelled.
        """
        task = self._tasks.get(task_id)
        if task is None:
            return False

        if task.status not in (DownloadStatus.PENDING, DownloadStatus.DOWNLOADING):
            return False

        was_downloading = task.status == DownloadStatus.DOWNLOADING

        # Mark as cancelled
        self._cancelled.add(task_id)
        task.status = DownloadStatus.CANCELLED
        # User intent: persist now so a restart does NOT resume this row.
        self._persist()

        # A task in DOWNLOADING owns the download semaphore, so every group
        # in the registry belongs to this task's shard-parallel transfer;
        # aborting the (global) session makes its snapshot_download thread
        # unwind immediately. Pending tasks must not abort, that would kill
        # another task's transfer. The next download lazily creates a fresh
        # session.
        if was_downloading:
            # Group abort stops every shard's transfer where it stands —
            # snapshot_download runs files concurrently, each with its own
            # group, and all of them are registered. sigint_abort only flips
            # a session flag that a long reconstruction may not check until
            # it finishes (xet-core#1322 ghost downloads), so keep it as the
            # backstop for anything the groups missed.
            _abort_xet_transfers()
            abort_xet_session()

        # Stop progress polling
        progress_task = self._progress_tasks.pop(task_id, None)
        if progress_task and not progress_task.done():
            progress_task.cancel()

        # Cancel the download task
        active_task = self._active_tasks.pop(task_id, None)
        if active_task and not active_task.done():
            active_task.cancel()

        logger.info(f"Download cancelled: {task.repo_id} (task_id={task_id})")
        return True

    def remove_task(self, task_id: str) -> bool:
        """Remove a completed, failed, or cancelled task from the list.

        Args:
            task_id: The task ID to remove.

        Returns:
            True if the task was found and removed.
        """
        task = self._tasks.get(task_id)
        if task is None:
            return False

        if task.status in (DownloadStatus.PENDING, DownloadStatus.DOWNLOADING):
            return False

        del self._tasks[task_id]
        self._cancelled.discard(task_id)
        # The row is gone from memory; drop it from disk too.
        self._persist()
        return True

    async def retry_download(
        self, task_id: str, hf_token: str = ""
    ) -> DownloadTask:
        """Retry a failed or cancelled download, resuming from existing files.

        Finalized shards are preserved on disk so snapshot_download will
        automatically skip already-completed files.

        Args:
            task_id: The task ID of the failed/cancelled download.
            hf_token: Optional HuggingFace token for gated models.

        Returns:
            The new DownloadTask.

        Raises:
            ValueError: If task not found or not in retryable state.
        """
        old_task = self._tasks.get(task_id)
        if old_task is None:
            raise ValueError(f"Task not found: {task_id}")

        if old_task.status not in (DownloadStatus.FAILED, DownloadStatus.CANCELLED):
            raise ValueError(
                f"Task {task_id} is not retryable (status: {old_task.status.value})"
            )

        repo_id = old_task.repo_id
        old_retry_count = old_task.retry_count

        # Remove old task entry
        del self._tasks[task_id]
        self._cancelled.discard(task_id)

        # Start fresh download (snapshot_download resumes from existing files).
        # An empty retry token means "no new credential entered" (the app's
        # retry sends none): keep the stored one instead of wiping it, so a
        # retry after a restart still reaches a private repository.
        new_task = await self.start_download(repo_id, hf_token or old_task.token)
        new_task.retry_count = old_retry_count + 1
        # start_download persisted the row before this bookkeeping; write
        # again so a crash right after a retry keeps the count and the
        # (possibly re-entered) credential on disk too.
        self._persist()
        return new_task

    def get_tasks(self) -> list[dict]:
        """Return all tasks as serializable dicts, ordered by creation time."""
        return [
            task.to_dict()
            for task in sorted(self._tasks.values(), key=lambda t: t.created_at)
        ]

    def _persist(self) -> None:
        """Write the queue to disk — no-op outside normal operation.

        Skipped while restoring (the on-disk queue must stay intact until
        every interrupted row has been re-queued) and during shutdown
        (interrupted rows must keep saying "downloading" so the next boot
        resumes them instead of seeing this process's dying cancelled state).
        """
        if self._tasks_file is None or self._restoring or self._shutting_down:
            return
        _write_tasks_file(self._tasks_file, self._tasks.values())

    async def restore_tasks(self) -> None:
        """Restore the persisted queue after a restart.

        Rows interrupted mid-download (pending/downloading) re-enter through
        start_download, which resumes from files already on disk and
        re-serializes behind the download semaphore; terminal rows come back
        as display-only entries so a failed or cancelled download stays
        retryable. The row's persisted request-supplied token rides along
        with the resumed download (credential recovery for private/gated
        repos); when a row carries none, start_download hands hub an empty
        token, which falls back to HF_TOKEN/the login cache, and a resume
        that still fails on auth lands in FAILED where the user can
        re-enter a credential and retry. Never raises: a corrupt queue file
        only loses the queue.
        """
        entries = _read_tasks_file(self._tasks_file)
        if not entries:
            return
        self._restoring = True
        try:
            for entry in sorted(
                (e for e in entries if isinstance(e, dict)),
                key=_restore_created_at,
            ):
                status = entry.get("status")
                if status in (
                    DownloadStatus.PENDING.value,
                    DownloadStatus.DOWNLOADING.value,
                ):
                    repo_id = entry.get("repo_id")
                    if not isinstance(repo_id, str):
                        continue
                    try:
                        task = await self.start_download(
                            repo_id, str(entry.get("token") or "")
                        )
                    except ValueError as exc:
                        logger.warning(
                            "Could not resume download %s: %s", repo_id, exc
                        )
                        continue
                    # Keep the pre-restart queue position and retry history.
                    # A row this build cannot read keeps the fresh task's own
                    # values instead of aborting the restore for the rows
                    # behind it — the same tolerance from_dict gets.
                    task.created_at = _restore_created_at(entry) or task.created_at
                    task.retry_count = _restore_retry_count(entry)
                    logger.info("Resumed interrupted download: %s", repo_id)
                    continue
                try:
                    task = DownloadTask.from_dict(entry)
                except (KeyError, TypeError, ValueError):
                    logger.warning(
                        "Skipping unpersistable download row: %s",
                        {k: v for k, v in entry.items() if k != "token"},
                    )
                    continue
                self._tasks.setdefault(task.task_id, task)
        finally:
            self._restoring = False
        self._persist()

    async def shutdown(self) -> None:
        """Cancel all active downloads and clean up."""
        # Leave the persisted queue untouched: rows stay "pending/downloading"
        # on disk so the next boot resumes them; the cancelled states below
        # are this process's dying breath, not user intent.
        self._shutting_down = True
        # Cancel all progress polling tasks
        for task_id, progress_task in list(self._progress_tasks.items()):
            if not progress_task.done():
                progress_task.cancel()
        self._progress_tasks.clear()

        # Cancel all active download tasks. Mark cancelled first so an
        # in-flight snapshot_download thread aborts via its progress callback;
        # active_task.cancel() only unblocks tasks still waiting on the semaphore.
        for task_id, active_task in list(self._active_tasks.items()):
            self._cancelled.add(task_id)
            if not active_task.done():
                active_task.cancel()
                task = self._tasks.get(task_id)
                if task and task.status == DownloadStatus.DOWNLOADING:
                    task.status = DownloadStatus.CANCELLED
        self._active_tasks.clear()
        self._wire_counters.clear()

        # Reap any in-flight xet transfer threads: group abort stops every
        # shard's Rust transfer where it stands (sigint alone can leave a
        # reconstruction running, which would hang the non-daemon writer
        # thread on interpreter exit), sigint is the backstop.
        _abort_xet_transfers()
        abort_xet_session()

        logger.info("HF Downloader shut down")

    async def _run_download(self, task_id: str, hf_token: str) -> None:
        """Execute a download task.

        Waits for the download semaphore (only one download runs at a time),
        then fetches repo info for total size and runs snapshot_download in a
        thread while polling the target directory for progress updates.
        """
        task = self._tasks[task_id]

        # Cancellation reaches the Rust transfers through the groups this
        # records; install once, idempotently, before any xet session exists.
        _install_xet_group_capture()
        _install_xet_call_marking()

        try:
            async with self._download_sem:
                # Check if cancelled while waiting in queue
                if task_id in self._cancelled:
                    return

                task.status = DownloadStatus.DOWNLOADING
                task.started_at = time.time()

                # Preserve {owner}/{model} layout to match other tools
                # (LMStudio, huggingface-cli) and avoid duplicate downloads
                # when sharing a model directory.
                target_dir = self._model_dir / task.repo_id

                api, endpoint = _get_hf_api()

                # Skip pytorch format when safetensors exist to
                # avoid downloading redundant weight files.
                ignore_patterns = None
                st_estimate = 0
                try:
                    model_info = await asyncio.wait_for(
                        asyncio.to_thread(
                            api.model_info,
                            task.repo_id,
                            token=hf_token or None,
                            files_metadata=True,
                        ),
                        timeout=_HF_API_TIMEOUT,
                    )
                    st_params = _safetensors_parameters(
                        getattr(model_info, "safetensors", None)
                    )
                    if st_params:
                        ignore_patterns = [
                            "*.bin",
                            "original/**",
                            "consolidated.*.pth",
                        ]
                        # Blob sizes match the download; the U32 dtype
                        # histogram is logical params, not disk bytes (#3401).
                        # Non-int counts degrade to no estimate instead of
                        # failing the download from the dry-run handler.
                        try:
                            blob = _sum_safetensors_blob_bytes(
                                getattr(model_info, "siblings", None)
                            )
                            if blob is not None:
                                st_estimate = blob
                            elif "U32" not in st_params:
                                st_estimate = _calc_safetensors_disk_size(
                                    {"parameters": st_params}
                                )
                        except (TypeError, ValueError):
                            st_estimate = 0
                except Exception as e:
                    logger.warning(
                        f"Could not fetch repo info for {task.repo_id}: {e}"
                    )

                dl_kwargs: dict = {
                    "repo_id": task.repo_id,
                    "local_dir": str(target_dir),
                    "token": hf_token or None,
                    "endpoint": endpoint,
                    "etag_timeout": 30,
                }
                if ignore_patterns:
                    dl_kwargs["ignore_patterns"] = ignore_patterns

                # Get accurate total size via dry run so the progress
                # denominator matches what will actually be downloaded.
                size_estimated = False
                try:
                    dry_result = await asyncio.wait_for(
                        asyncio.to_thread(
                            snapshot_download,
                            **dl_kwargs,
                            dry_run=True,
                        ),
                        timeout=30,
                    )
                    task.total_size = sum(f.file_size for f in dry_result)
                except Exception as e:
                    if st_estimate:
                        task.total_size = st_estimate
                        size_estimated = True
                        detail = "Estimated total size from safetensors metadata."
                    else:
                        detail = "Progress estimation will be unavailable."
                    logger.warning(
                        f"Dry run failed for {task.repo_id}: {e}. {detail}"
                    )

                wire_counter = _WireCounter()
                self._wire_counters[task_id] = wire_counter
                self._progress_tasks[task_id] = asyncio.create_task(
                    self._poll_progress(task_id, target_dir)
                )

                xet_error: Exception | None = None
                try:
                    # Awaiting the thread here is intentional: after a stall,
                    # fallback cannot start until abort_xet_session() has made
                    # the original writer return. The wrapper keeps this call
                    # registered for aborts that arrive before its first
                    # xet_get(), when the group registry is still empty.
                    await asyncio.to_thread(
                        _tracked_snapshot_download,
                        snapshot_download,
                        lambda: task_id in self._cancelled,
                        **dl_kwargs,
                        tqdm_class=_make_cancellable_tqdm(
                            lambda: task_id in self._cancelled,
                            on_wire_bytes=wire_counter.add,
                        ),
                    )
                except Exception as error:
                    stalled = self._stalled.pop(task_id, None)
                    if stalled is not None:
                        xet_error = stalled
                    elif _is_xet_transport_error(error):
                        xet_error = error
                    else:
                        raise
                else:
                    xet_error = self._stalled.pop(task_id, None)

                if xet_error is not None:
                    if task_id in self._cancelled:
                        raise _DownloadCancelled()
                    logger.warning(
                        "Xet download failed for %s: %s. "
                        "Retrying once over HTTP.",
                        task.repo_id,
                        xet_error,
                    )
                    progress_task = self._progress_tasks.pop(task_id, None)
                    if progress_task and not progress_task.done():
                        progress_task.cancel()
                    # The HTTP worker cannot reuse xet's chunk cache: it
                    # refetches the payload into the target dir, so the
                    # fetch-phase wire bytes no longer describe bytes any
                    # file of this task owns. Start the replacement poll
                    # from zero.
                    self._wire_counters[task_id] = _WireCounter()
                    self._progress_tasks[task_id] = asyncio.create_task(
                        self._poll_progress(task_id, target_dir)
                    )
                    try:
                        await self._run_http_fallback(task_id, dl_kwargs)
                    except asyncio.CancelledError:
                        raise
                    except Exception as error:
                        fallback_error = self._stalled.pop(task_id, None) or error
                        raise _NonXetDownloadError(
                            "Xet download failed "
                            f"({xet_error}); HTTP fallback also failed "
                            f"({fallback_error})"
                        ) from error

                # Check if cancelled while downloading
                if task_id in self._cancelled:
                    return

                # Success
                task.status = DownloadStatus.COMPLETED
                task.progress = 100.0
                task.speed_bps = 0.0
                if size_estimated or not task.total_size:
                    # The estimate was only a progress denominator; report
                    # the measured on-disk size once the download is done.
                    task.downloaded_size = self._get_dir_size(target_dir)
                else:
                    task.downloaded_size = task.total_size
                task.completed_at = time.time()

                logger.info(
                    f"Download completed: {task.repo_id} -> {target_dir} "
                    f"({time.time() - task.started_at:.1f}s)"
                )

                # Trigger model pool refresh
                if self._on_complete:
                    try:
                        await self._on_complete()
                    except Exception as e:
                        logger.error(
                            f"Error in download completion callback: {e}"
                        )

        except (_DownloadCancelled, asyncio.CancelledError):
            if task.status not in (
                DownloadStatus.CANCELLED,
                DownloadStatus.FAILED,
            ):
                task.status = DownloadStatus.CANCELLED
            try:
                self._cleanup_partial(task)
            except Exception as e:
                logger.error(
                    f"Failed to clean up cancelled download {task.repo_id}: {e}"
                )
        except RepositoryNotFoundError:
            task.status = DownloadStatus.FAILED
            task.error = (
                f"Repository not found: {task.repo_id}. "
                "This may be a gated model that requires HuggingFace authentication."
            )
            logger.error(f"Repository not found: {task.repo_id}")
        except GatedRepoError:
            task.status = DownloadStatus.FAILED
            task.error = (
                f"Repository '{task.repo_id}' is gated. "
                "Please provide a valid HF token with access."
            )
            logger.error(f"Gated repo access denied: {task.repo_id}")
        except Exception as e:
            # Do not clobber an earlier terminal error or a user cancellation.
            if (
                task_id not in self._cancelled
                and task.status != DownloadStatus.FAILED
            ):
                task.status = DownloadStatus.FAILED
                task.error = str(e)
                logger.error(f"Download failed for {task.repo_id}: {e}")
        finally:
            # Stop progress polling
            progress_task = self._progress_tasks.pop(task_id, None)
            if progress_task and not progress_task.done():
                progress_task.cancel()
            self._stalled.pop(task_id, None)
            self._wire_counters.pop(task_id, None)

            # Remove from active tasks
            self._active_tasks.pop(task_id, None)

            # Persist whatever terminal state the run settled on (completed,
            # failed, or user-cancelled). Skipped during shutdown so an
            # interrupted row survives for the next boot to resume.
            self._persist()

    async def _run_http_fallback(
        self,
        task_id: str,
        dl_kwargs: dict,
    ) -> None:
        env = os.environ.copy()
        env["HF_HUB_DISABLE_XET"] = "1"
        payload = json.dumps({"kwargs": dl_kwargs}).encode()
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            _NON_XET_WORKER_MODULE,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        self._fallback_processes[task_id] = process

        try:
            stdout, _ = await process.communicate(input=payload)
            try:
                response = json.loads(stdout.decode())
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise _NonXetDownloadError(
                    f"HTTP fallback process exited with code {process.returncode}"
                ) from error
            if process.returncode != 0 or not response.get("ok"):
                error_type = response.get("error_type", "DownloadError")
                message = response.get("message", "unknown error")
                raise _NonXetDownloadError(f"{error_type}: {message}")
        except asyncio.CancelledError:
            await self._stop_subprocess(process)
            raise
        finally:
            self._fallback_processes.pop(task_id, None)

    @staticmethod
    async def _stop_subprocess(process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        try:
            process.send_signal(signal.SIGINT)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(
                process.wait(),
                timeout=_SUBPROCESS_TERMINATE_TIMEOUT,
            )
            return
        except TimeoutError:
            try:
                process.terminate()
            except ProcessLookupError:
                return
        try:
            await asyncio.wait_for(
                process.wait(),
                timeout=_SUBPROCESS_TERMINATE_TIMEOUT,
            )
        except TimeoutError:
            try:
                process.kill()
            except ProcessLookupError:
                return
            await process.wait()

    async def _poll_progress(self, task_id: str, target_dir: Path) -> None:
        """Poll the target directory to estimate download progress.

        Uses both directory size and file modification times to detect
        activity. huggingface_hub pre-allocates large files and fills them
        in, so size alone may not change for extended periods. File mtimes
        are updated on each write syscall and serve as a more reliable
        liveness signal.

        The wire counter extends both duties across xet's fetch phase:
        reported progress may come from bytes that are on the network but
        not yet reconstructed on disk, and wire movement keeps the stall
        deadline fresh while the filesystem stays silent.
        """
        task = self._tasks.get(task_id)
        if task is None:
            return

        last_activity = self._get_download_activity(target_dir)
        last_activity_at = time.monotonic()
        observed_activity = False
        # Two speed sources over the same payload, one per pipeline stage:
        # APFS allocated blocks (grow as bytes are written, including into
        # sparse/preallocated files) and the transfer bar's network-byte
        # counter (xet fetch phase, where disk writes lag by up to a 256MB
        # reconstruction block). Prime both windows with baselines taken
        # before the first sleep so the first reading covers real transfer
        # time rather than startup delay.
        speed_meter = _SpeedMeter()
        speed_meter.add(last_activity.files, now=last_activity_at)
        wire_counter = self._wire_counters.get(task_id)
        wire_meter = _WireSpeedMeter()
        wire_meter.add(
            wire_counter.value if wire_counter else 0, now=last_activity_at
        )
        last_wire = wire_counter.value if wire_counter else 0
        # Bytes already on disk when this attempt started. A resumed
        # download re-fetches only what is missing, so the wire counter
        # measures the remainder while the disk already holds the files
        # from earlier attempts: the wire delta is progress on top of this
        # baseline (and is 0 for a fresh download).
        base_logical = last_activity.logical_size

        try:
            while task.status == DownloadStatus.DOWNLOADING:
                await asyncio.sleep(_PROGRESS_POLL_INTERVAL)

                if task.status != DownloadStatus.DOWNLOADING:
                    break

                activity = self._get_download_activity(target_dir)
                wire_now = wire_counter.value if wire_counter else 0

                # Progress follows whichever pipeline stage is further along:
                # fetch-phase bytes exist only on the wire (reconstruction
                # lags by up to a 256MB block) and reconstruction-phase bytes
                # only on disk. Bytes this attempt flushed are already counted
                # by the disk, so only the wire bytes the disk has not caught
                # up with are added: a fresh download reduces to
                # max(disk, wire), and a resume — the disk already holds the
                # earlier files while the wire counts only the remainder —
                # stays live instead of parking on the disk figure for the
                # whole fetch phase. Both readings are cumulative, so the
                # report never regresses, and snapshot_download's completion
                # write has the final word.
                disk_growth = max(0, activity.logical_size - base_logical)
                reported = activity.logical_size + max(0, wire_now - disk_growth)
                if task.total_size > 0:
                    reported = min(reported, task.total_size)
                    # Cap at 99% until snapshot_download confirms completion
                    task.progress = min(
                        (reported / task.total_size) * 100,
                        99.0,
                    )
                task.downloaded_size = reported

                disk_bps = speed_meter.add(activity.files)
                wire_bps = (
                    wire_meter.add(wire_counter.value)
                    if wire_counter
                    else 0.0
                )
                # Report the larger of the two stages, never their sum: the
                # fetch phase advances only the wire counter, the
                # reconstruction flush advances only the disk meter, and a
                # genuine stall leaves both at zero.
                task.speed_bps = max(disk_bps, wire_bps)

                wire_moved = wire_now > last_wire
                last_wire = wire_now

                if activity != last_activity:
                    # A zero-byte temp file or metadata touch is not payload
                    # progress. APFS allocated blocks increase when bytes are
                    # actually written, including into sparse/preallocated files.
                    if activity.allocated_size > last_activity.allocated_size:
                        observed_activity = True
                    last_activity = activity
                    last_activity_at = time.monotonic()
                    continue

                if wire_moved:
                    # Payload on the wire is progress even while xet's fetch
                    # phase keeps the filesystem silent: keep the stall
                    # deadline fresh, and count bytes past the token/CAS
                    # handshake as observed activity — that handshake, not
                    # the silent fetch, is what the startup deadline guards.
                    observed_activity = True
                    last_activity_at = time.monotonic()
                    continue

                timeout = (
                    _STALL_TIMEOUT
                    if observed_activity
                    else _STARTUP_STALL_TIMEOUT
                )
                if time.monotonic() - last_activity_at > timeout:
                    phase = "active" if observed_activity else "startup"
                    process = self._fallback_processes.get(task_id)
                    transport = "HTTP fallback" if process else "Xet"
                    stalled = _DownloadStalledError(
                        transport=transport,
                        phase=phase,
                        timeout=timeout,
                    )
                    self._stalled[task_id] = stalled
                    logger.warning(
                        "%s for %s (task_id=%s)",
                        stalled,
                        task.repo_id,
                        task_id,
                    )
                    if process is not None:
                        await self._stop_subprocess(process)
                    else:
                        abort_xet_session()
                    break
        except asyncio.CancelledError:
            pass
        finally:
            # Terminal states (done, failed, cancelled, stalled) report no
            # rate — only a live transfer has a speed.
            task.speed_bps = 0.0

    @staticmethod
    def _get_download_activity(path: Path) -> _DownloadActivity:
        """Return size, allocation, and mtime signals including Hub temp files."""
        if not path.exists():
            return _DownloadActivity()
        file_count = 0
        logical_size = 0
        allocated_size = 0
        latest_mtime_ns = 0
        files: dict[str, int] = {}
        try:
            for file_path in path.rglob("*"):
                if not file_path.is_file():
                    continue
                try:
                    stat = file_path.stat()
                except OSError:
                    continue
                file_count += 1
                logical_size += stat.st_size
                blocks = getattr(stat, "st_blocks", 0) * 512
                allocated_size += blocks
                files[str(file_path)] = blocks
                latest_mtime_ns = max(latest_mtime_ns, stat.st_mtime_ns)
        except OSError:
            # A walk aborted mid-way (concurrent cleanup) under-counts.
            # Hand the partial `files` map to the meter anyway: paths that
            # drop out are forgotten, so when they show up again they read
            # as first-sight (0) instead of replaying as a giant delta.
            pass
        return _DownloadActivity(
            file_count=file_count,
            logical_size=logical_size,
            allocated_size=allocated_size,
            latest_mtime_ns=latest_mtime_ns,
            files=files,
        )

    @staticmethod
    def _get_dir_size(path: Path) -> int:
        """Calculate total size of all files in a directory."""
        if not path.exists():
            return 0
        total = 0
        try:
            for f in path.rglob("*"):
                if f.is_file():
                    try:
                        total += f.stat().st_size
                    except OSError:
                        pass
        except OSError:
            pass
        return total

    def _cleanup_partial(self, task: DownloadTask) -> None:
        """Remove in-progress shards while keeping finalized files for resume.

        Hub stages partial downloads inside a hidden ``._____temp`` directory
        and only renames a shard into the target on completion. Wiping the
        whole target dir would also nuke shards the user has already paid
        for; finalized files are visible in the file browser, so users can
        keep them for auto-resume on retry or remove them themselves.
        """
        target_dir = self._model_dir / task.repo_id
        temp_dir = target_dir / "._____temp"
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up in-progress shards: {temp_dir}")
            except Exception as e:
                logger.error(f"Failed to clean up {temp_dir}: {e}")
